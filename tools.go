package main

// Tool call adaptation: injects tool definitions into the system prompt for
// models without native function calling, and parses `<tool_call>` tags from
// the model output. Ported from src/tools.rs.

import (
	"fmt"
	"log"
	"regexp"
	"strings"
	"unicode/utf8"
)

// ToolCall is a parsed tool invocation. ID == "" means no id was provided.
type ToolCall struct {
	Name      string
	Arguments string
	ID        string
}

// ToolStreamEvent kinds emitted by the streaming tool-call parser.
const (
	ToolEventText = iota
	ToolEventCall
)

// ToolStreamEvent is either a chunk of plain text or a completed tool call.
type ToolStreamEvent struct {
	Kind int
	Text string
	Call ToolCall
}

const (
	tagOpen  = "<tool_call>"
	tagClose = "</tool_call>"
)

// transformRequest injects tools into the system prompt and removes the
// "tools"/"tool_choice" fields. Mutates data in place (maps are references).
// Returns true if the request was modified.
func transformRequest(data map[string]any) bool {
	toolsAny := data["tools"]
	tools := asArr(toolsAny)
	if len(tools) == 0 {
		return false
	}

	toolChoice, hasChoice := data["tool_choice"]
	var choicePtr any
	if hasChoice {
		choicePtr = toolChoice
	}
	toolsPrompt := buildToolsPrompt(tools, choicePtr, hasChoice)

	if messages := asArr(data["messages"]); messages != nil {
		transformMessages(messages)

		hasSystem := false
		if len(messages) > 0 {
			if getStr(messages[0], "role") == "system" {
				hasSystem = true
			}
		}

		if hasSystem {
			sys := asObj(messages[0])
			if sys != nil {
				if content, ok := asStr(sys["content"]); ok {
					sys["content"] = fmt.Sprintf("%s\n\n%s", content, toolsPrompt)
				}
			}
		} else {
			newMsgs := make([]any, 0, len(messages)+1)
			newMsgs = append(newMsgs, map[string]any{"role": "system", "content": toolsPrompt})
			newMsgs = append(newMsgs, messages...)
			data["messages"] = newMsgs
		}
	}

	delete(data, "tools")
	delete(data, "tool_choice")

	log.Printf("INFO Tool adaptation: injected %d tools into system prompt", len(tools))
	return true
}

func buildToolsPrompt(tools []any, toolChoice any, hasChoice bool) string {
	var b strings.Builder
	b.WriteString("# Available Tools\n\n" +
		"You have access to the following tools. To use a tool, output a tool call in this exact format:\n\n" +
		"<tool_call>\n" +
		"{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n" +
		"</tool_call>\n\n" +
		"IMPORTANT RULES for tool calls:\n" +
		"1. Each tool call must be wrapped in its own <tool_call></tool_call> tags.\n" +
		"2. The JSON inside must be valid. For string values containing newlines, use \\n. For quotes, use \\\".\n" +
		"3. For file write operations: put the COMPLETE file content in the \"content\" argument as a single JSON string. " +
		"Every newline in the file must be written as \\n in the JSON string.\n" +
		"4. When you want to call a tool, output ONLY the tool call(s) with no other text.\n" +
		"5. When you don't need to use any tool, respond normally without <tool_call> tags.\n\n" +
		"Example of writing a file:\n" +
		"<tool_call>\n" +
		"{\"name\": \"Write\", \"arguments\": {\"file_path\": \"/path/to/file.py\", \"content\": \"#!/usr/bin/env python3\\nimport os\\n\\ndef main():\\n    print(\\\"hello\\\")\\n\"}}\n" +
		"</tool_call>\n\n" +
		"## Tool Definitions\n\n")

	for _, tool := range tools {
		var name, desc, params string
		if fn := getObj(tool, "function"); fn != nil {
			name = orDefault(getStr(fn, "name"), "unknown")
			desc = getStr(fn, "description")
			if p, ok := fn["parameters"]; ok {
				params = string(toJSONPretty(p))
			}
		} else {
			name = orDefault(getStr(tool, "name"), "unknown")
			desc = getStr(tool, "description")
			if p := get(tool, "input_schema"); p != nil {
				params = string(toJSONPretty(p))
			} else if p := get(tool, "parameters"); p != nil {
				params = string(toJSONPretty(p))
			}
		}
		b.WriteString(fmt.Sprintf("### %s\n%s\nParameters:\n```json\n%s\n```\n\n", name, desc, params))
	}

	if hasChoice {
		switch choice := toolChoice.(type) {
		case string:
			if choice == "required" || choice == "any" {
				b.WriteString("IMPORTANT: You MUST use at least one tool in your response.\n\n")
			} else if choice == "none" {
				b.WriteString("NOTE: Do not use any tools. Respond with text only.\n\n")
			}
		case map[string]any:
			var funcName string
			if fn := asObj(choice["function"]); fn != nil {
				funcName = getStr(fn, "name")
			}
			if funcName == "" {
				funcName = getStr(choice, "name")
			}
			if funcName != "" {
				b.WriteString(fmt.Sprintf("IMPORTANT: You MUST use the \"%s\" tool in your response.\n\n", funcName))
			}
		}
	}

	return b.String()
}

func transformMessages(messages []any) {
	for _, m := range messages {
		msg := asObj(m)
		if msg == nil {
			continue
		}
		role := getStr(msg, "role")
		switch role {
		case "tool":
			toolCallID := getStr(msg, "tool_call_id")
			content := getStr(msg, "content")
			msg["role"] = "user"
			msg["content"] = fmt.Sprintf("[Tool Result (call_id: %s)]:\n%s", toolCallID, content)
			delete(msg, "tool_call_id")
		case "assistant":
			toolCalls := asArr(msg["tool_calls"])
			if toolCalls != nil {
				var textParts []string
				if content, ok := asStr(msg["content"]); ok && content != "" {
					textParts = append(textParts, content)
				}
				for _, tc := range toolCalls {
					if fn := getObj(tc, "function"); fn != nil {
						name := getStr(fn, "name")
						args := getStr(fn, "arguments")
						if args == "" {
							args = "{}"
						}
						id := getStr(tc, "id")
						// Re-serialize the arguments JSON if it parses, else keep as-is.
						argsJSON := args
						if v, err := parseJSON([]byte(args)); err == nil {
							argsJSON = toJSONString(v)
						}
						textParts = append(textParts, fmt.Sprintf(
							"<tool_call>\n{\"name\": \"%s\", \"arguments\": %s, \"call_id\": \"%s\"}\n</tool_call>",
							name, argsJSON, id))
					}
				}
				msg["content"] = strings.Join(textParts, "\n")
				delete(msg, "tool_calls")
			}
		}
	}
}

// anthropicRequestToOpenAI converts an Anthropic Messages request to OpenAI
// chat-completions shape.
func anthropicRequestToOpenAI(req any) map[string]any {
	var messages []any

	// System prompt
	if system := get(req, "system"); system != nil {
		text := extractAnthropicText(system)
		if text != "" {
			messages = append(messages, map[string]any{"role": "system", "content": text})
		}
	}

	// Messages
	if msgs := getArr(req, "messages"); msgs != nil {
		for _, msg := range msgs {
			role := getStr(msg, "role")
			if role == "" {
				role = "user"
			}
			content := get(msg, "content")

			switch c := content.(type) {
			case string:
				messages = append(messages, map[string]any{"role": role, "content": c})
			case []any:
				var textParts []string
				var toolCalls []any
				type toolResult struct{ id, text string }
				var toolResults []toolResult

				for _, block := range c {
					btype := getStr(block, "type")
					switch btype {
					case "text":
						if t, ok := asStr(get(block, "text")); ok {
							textParts = append(textParts, t)
						}
					case "tool_use":
						name := getStr(block, "name")
						id := getStr(block, "id")
						input := get(block, "input")
						if input == nil {
							input = map[string]any{}
						}
						argsStr := toJSONString(input)
						if argsStr == "" {
							argsStr = "{}"
						}
						toolCalls = append(toolCalls, map[string]any{
							"id": id, "type": "function",
							"function": map[string]any{"name": name, "arguments": argsStr},
						})
					case "tool_result":
						toolUseID := getStr(block, "tool_use_id")
						resultContent := ""
						if rc := get(block, "content"); rc != nil {
							resultContent = extractAnthropicText(rc)
						}
						isError := getBool(block, "is_error")
						prefix := ""
						if isError {
							prefix = "[ERROR] "
						}
						toolResults = append(toolResults, toolResult{toolUseID, prefix + resultContent})
					case "thinking":
						// skip
					}
				}

				if role == "assistant" {
					m := map[string]any{"role": "assistant"}
					text := strings.Join(textParts, "")
					if text != "" {
						m["content"] = text
					}
					if len(toolCalls) > 0 {
						m["tool_calls"] = toolCalls
						if _, ok := m["content"]; !ok {
							m["content"] = nil
						}
					} else if _, ok := m["content"]; !ok {
						m["content"] = ""
					}
					messages = append(messages, m)
				} else {
					if len(toolResults) > 0 {
						for _, tr := range toolResults {
							messages = append(messages, map[string]any{
								"role": "tool", "tool_call_id": tr.id, "content": tr.text,
							})
						}
						text := strings.Join(textParts, "")
						if text != "" {
							messages = append(messages, map[string]any{"role": "user", "content": text})
						}
					} else {
						messages = append(messages, map[string]any{"role": role, "content": strings.Join(textParts, "")})
					}
				}
			default:
				messages = append(messages, map[string]any{"role": role, "content": ""})
			}
		}
	}

	maxTokens := 4096.0
	if mt, ok := getNum(req, "max_tokens"); ok {
		maxTokens = mt
	}
	stream := getBool(req, "stream")

	result := map[string]any{
		"model":      getStr(req, "model"),
		"messages":   messages,
		"max_tokens": maxTokens,
		"stream":     stream,
	}

	if temp := get(req, "temperature"); temp != nil {
		result["temperature"] = temp
	}
	if topP := get(req, "top_p"); topP != nil {
		result["top_p"] = topP
	}
	if stop := get(req, "stop_sequences"); stop != nil {
		result["stop"] = stop
	}

	// Pass through sampling params for local model control.
	for _, key := range []string{"repetition_penalty", "repetition_context_size", "top_k", "min_p", "frequency_penalty", "presence_penalty"} {
		if v := get(req, key); v != nil {
			result[key] = v
		}
	}

	// Convert Anthropic tools → OpenAI format.
	if tools := getArr(req, "tools"); len(tools) > 0 {
		openaiTools := make([]any, 0, len(tools))
		for _, t := range tools {
			if getStr(t, "type") == "function" {
				openaiTools = append(openaiTools, t)
			} else {
				inputSchema := get(t, "input_schema")
				if inputSchema == nil {
					inputSchema = map[string]any{"type": "object"}
				}
				openaiTools = append(openaiTools, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        getStr(t, "name"),
						"description": getStr(t, "description"),
						"parameters":  inputSchema,
					},
				})
			}
		}
		result["tools"] = openaiTools
	}

	if choice := get(req, "tool_choice"); choice != nil {
		switch obj := choice.(type) {
		case map[string]any:
			ctype := getStr(obj, "type")
			if ctype == "" {
				ctype = "auto"
			}
			switch ctype {
			case "auto":
				result["tool_choice"] = "auto"
			case "any":
				result["tool_choice"] = "required"
			case "tool":
				if name := getStr(obj, "name"); name != "" {
					result["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": name}}
				} else {
					result["tool_choice"] = "auto"
				}
			default:
				result["tool_choice"] = "auto"
			}
		default:
			result["tool_choice"] = choice
		}
	}

	if getBool(result, "stream") {
		result["stream_options"] = map[string]any{"include_usage": true}
	}

	return result
}

func extractAnthropicText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var parts []string
		for _, b := range c {
			if getStr(b, "type") == "text" {
				if t, ok := asStr(get(b, "text")); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "")
	default:
		return ""
	}
}

// transformResponse parses model output for `<tool_call>` tags and rewrites the
// response into native OpenAI tool_calls shape.
func transformResponse(respBody []byte) []byte {
	root, err := parseJSON(respBody)
	if err != nil {
		return respBody
	}

	choices := getArr(root, "choices")
	if choices == nil {
		return respBody
	}

	modified := false
	for _, ch := range choices {
		choice := asObj(ch)
		if choice == nil {
			continue
		}
		msg := asObj(choice["message"])
		if msg == nil {
			continue
		}
		content, ok := asStr(msg["content"])
		if !ok {
			continue
		}

		toolCalls := parseToolCalls(content)
		if len(toolCalls) == 0 {
			continue
		}

		tcArray := make([]any, 0, len(toolCalls))
		for i, tc := range toolCalls {
			id := tc.ID
			if id == "" {
				id = fmt.Sprintf("call_%d", i)
			}
			tcArray = append(tcArray, map[string]any{
				"id":       id,
				"type":     "function",
				"function": map[string]any{"name": tc.Name, "arguments": tc.Arguments},
			})
		}

		remainingText := stripToolCalls(content)
		msg["tool_calls"] = tcArray
		if strings.TrimSpace(remainingText) == "" {
			msg["content"] = nil
		} else {
			msg["content"] = remainingText
		}
		choice["finish_reason"] = "tool_calls"
		modified = true
	}

	if modified {
		if out := toJSON(root); out != nil {
			return out
		}
	}
	return respBody
}

// ─── Streaming tool call parser ──────────────────────────────────────────────

// ToolCallStreamParser incrementally parses `<tool_call>` tags out of a token
// stream, emitting text and completed tool calls.
type ToolCallStreamParser struct {
	buf       string
	inTag     bool
	tagBuf    string
	toolIndex int
}

func newToolCallStreamParser() *ToolCallStreamParser {
	return &ToolCallStreamParser{}
}

// Feed appends text to the parser's buffer and returns any events that became
// complete as a result.
func (p *ToolCallStreamParser) Feed(text string) []ToolStreamEvent {
	p.buf += text
	var events []ToolStreamEvent

	for {
		if p.inTag {
			if endPos := strings.Index(p.buf, tagClose); endPos >= 0 {
				p.tagBuf += p.buf[:endPos]
				p.buf = p.buf[endPos+len(tagClose):]
				if parsed, ok := fixJSON(strings.TrimSpace(p.tagBuf)); ok {
					name := getStr(parsed, "name")
					arguments := "{}"
					if a := get(parsed, "arguments"); a != nil {
						arguments = toJSONString(a)
						if arguments == "" {
							arguments = "{}"
						}
					}
					id := getStr(parsed, "call_id")
					if id == "" {
						id = fmt.Sprintf("toolu_%04x", p.toolIndex)
					}
					p.toolIndex++
					events = append(events, ToolStreamEvent{Kind: ToolEventCall, Call: ToolCall{Name: name, Arguments: arguments, ID: id}})
				}
				p.tagBuf = ""
				p.inTag = false
				continue
			}
			// No closing tag yet: flush everything except a possible partial
			// closing tag at the end, respecting UTF-8 rune boundaries.
			if len(p.buf) > len(tagClose) {
				safe := len(p.buf) - len(tagClose)
				for safe > 0 && !utf8.RuneStart(p.buf[safe]) {
					safe--
				}
				if safe > 0 {
					p.tagBuf += p.buf[:safe]
					p.buf = p.buf[safe:]
				}
			}
			break
		}

		if startPos := strings.Index(p.buf, tagOpen); startPos >= 0 {
			before := p.buf[:startPos]
			if before != "" {
				events = append(events, ToolStreamEvent{Kind: ToolEventText, Text: before})
			}
			p.buf = p.buf[startPos+len(tagOpen):]
			p.inTag = true
			p.tagBuf = ""
			continue
		}

		// No opening tag: emit text but hold back a possible partial opening tag.
		safeLen := len(p.buf)
		for prefixLen := len(tagOpen) - 1; prefixLen >= 1; prefixLen-- {
			if strings.HasSuffix(p.buf, tagOpen[:prefixLen]) {
				safeLen = len(p.buf) - prefixLen
				break
			}
		}
		if safeLen > 0 {
			safeText := p.buf[:safeLen]
			p.buf = p.buf[safeLen:]
			if safeText != "" {
				events = append(events, ToolStreamEvent{Kind: ToolEventText, Text: safeText})
			}
		}
		break
	}
	return events
}

// Flush emits any buffered remainder as text (including an unterminated tag).
func (p *ToolCallStreamParser) Flush() []ToolStreamEvent {
	var events []ToolStreamEvent
	if p.inTag {
		text := tagOpen + p.tagBuf + p.buf
		if text != "" {
			events = append(events, ToolStreamEvent{Kind: ToolEventText, Text: text})
		}
	} else if p.buf != "" {
		events = append(events, ToolStreamEvent{Kind: ToolEventText, Text: p.buf})
	}
	p.buf = ""
	p.tagBuf = ""
	return events
}

// HasSeenTools reports whether any complete tool call has been parsed.
func (p *ToolCallStreamParser) HasSeenTools() bool {
	return p.toolIndex > 0
}

var (
	reTrailingComma = regexp.MustCompile(`,\s*([}\]])`)
	reFuzzyName     = regexp.MustCompile(`"name"\s*:\s*"([^"]+)"`)
	reFuzzyArgs     = regexp.MustCompile(`"arguments"\s*:\s*(\{)`)
)

// fixJSON attempts to repair common JSON errors emitted by small models.
// Returns (value, true) on success.
func fixJSON(raw string) (any, bool) {
	// 1. Try as-is first.
	if v, err := parseJSON([]byte(raw)); err == nil {
		return v, true
	}

	s := raw

	// 2. Replace single quotes with double quotes (only if no double quotes present).
	if !strings.Contains(s, "\"") {
		s = strings.ReplaceAll(s, "'", "\"")
		if v, err := parseJSON([]byte(s)); err == nil {
			return v, true
		}
	}

	// 3. Fix unescaped literal newlines/tabs inside JSON string values.
	s = fixUnescapedNewlines(s)
	if v, err := parseJSON([]byte(s)); err == nil {
		return v, true
	}

	// 4. Remove trailing commas before } or ].
	s = reTrailingComma.ReplaceAllString(s, "$1")
	if v, err := parseJSON([]byte(s)); err == nil {
		return v, true
	}

	// 5. Try to fix missing closing braces.
	openBraces := strings.Count(s, "{")
	closeBraces := strings.Count(s, "}")
	if openBraces > closeBraces {
		s += strings.Repeat("}", openBraces-closeBraces)
		if v, err := parseJSON([]byte(s)); err == nil {
			return v, true
		}
	}

	// 6. Last resort: fuzzy-extract name + arguments.
	if v, ok := extractToolCallFuzzy(raw); ok {
		return v, true
	}

	limit := len(raw)
	if limit > 200 {
		limit = 200
	}
	log.Printf("WARN fix_json: could not repair: %s...", raw[:limit])
	return nil, false
}

// fixUnescapedNewlines escapes literal newlines/tabs inside JSON string values.
func fixUnescapedNewlines(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	escapeNext := false
	for _, ch := range s {
		if escapeNext {
			b.WriteRune(ch)
			escapeNext = false
			continue
		}
		if ch == '\\' && inString {
			b.WriteRune(ch)
			escapeNext = true
			continue
		}
		if ch == '"' {
			inString = !inString
			b.WriteRune(ch)
			continue
		}
		if inString && ch == '\n' {
			b.WriteString("\\n")
		} else if inString && ch == '\r' {
			// skip \r; covered by \n
		} else if inString && ch == '\t' {
			b.WriteString("\\t")
		} else {
			b.WriteRune(ch)
		}
	}
	return b.String()
}

// extractToolCallFuzzy is a last-resort extractor for "name" and "arguments".
func extractToolCallFuzzy(raw string) (any, bool) {
	nameMatch := reFuzzyName.FindStringSubmatch(raw)
	if nameMatch == nil {
		return nil, false
	}
	name := nameMatch[1]

	if loc := reFuzzyArgs.FindStringIndex(raw); loc != nil {
		// Find the first '{' at/after the match start.
		rel := strings.Index(raw[loc[0]:], "{")
		if rel < 0 {
			return map[string]any{"name": name, "arguments": map[string]any{}}, true
		}
		start := loc[0] + rel
		// Find the matching closing brace.
		depth := 0
		inStr := false
		esc := false
		end := start
		for i, ch := range raw[start:] {
			if esc {
				esc = false
				continue
			}
			if ch == '\\' && inStr {
				esc = true
				continue
			}
			if ch == '"' {
				inStr = !inStr
				continue
			}
			if !inStr {
				if ch == '{' {
					depth++
				}
				if ch == '}' {
					depth--
					if depth == 0 {
						end = start + i + 1
						break
					}
				}
			}
		}
		if end > start {
			argsRaw := raw[start:end]
			var args any = map[string]any{}
			if v, err := parseJSON([]byte(argsRaw)); err == nil {
				args = v
			} else {
				fixed := fixUnescapedNewlines(argsRaw)
				if v, err := parseJSON([]byte(fixed)); err == nil {
					args = v
				}
			}
			return map[string]any{"name": name, "arguments": args}, true
		}
	}

	// Fallback: no arguments found.
	return map[string]any{"name": name, "arguments": map[string]any{}}, true
}

// parseToolCalls extracts all complete `<tool_call>...</tool_call>` blocks.
func parseToolCalls(text string) []ToolCall {
	var calls []ToolCall
	searchFrom := 0
	for {
		rel := strings.Index(text[searchFrom:], tagOpen)
		if rel < 0 {
			break
		}
		absStart := searchFrom + rel + len(tagOpen)
		endRel := strings.Index(text[absStart:], tagClose)
		if endRel < 0 {
			break
		}
		absEnd := absStart + endRel
		inner := strings.TrimSpace(text[absStart:absEnd])
		if parsed, ok := fixJSON(inner); ok {
			name := getStr(parsed, "name")
			arguments := "{}"
			if a := get(parsed, "arguments"); a != nil {
				arguments = toJSONString(a)
				if arguments == "" {
					arguments = "{}"
				}
			}
			id := getStr(parsed, "call_id")
			if name != "" {
				calls = append(calls, ToolCall{Name: name, Arguments: arguments, ID: id})
			}
		}
		searchFrom = absEnd + len(tagClose)
	}
	return calls
}

func stripToolCalls(text string) string {
	result := text
	for {
		start := strings.Index(result, tagOpen)
		if start < 0 {
			break
		}
		endRel := strings.Index(result[start:], tagClose)
		if endRel < 0 {
			break
		}
		absEnd := start + endRel + len(tagClose)
		result = result[:start] + result[absEnd:]
	}
	return result
}

// ─── small local helpers ─────────────────────────────────────────────────────

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
