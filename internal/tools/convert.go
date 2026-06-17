package tools

import (
	"strings"

	"lmstudio-forward/internal/jsonx"
)

// AnthropicRequestToOpenAI converts an Anthropic Messages request to OpenAI
// chat-completions shape.
func AnthropicRequestToOpenAI(req any) map[string]any {
	var messages []any

	// System prompt
	if system := jsonx.Get(req, "system"); system != nil {
		text := extractAnthropicText(system)
		if text != "" {
			messages = append(messages, map[string]any{"role": "system", "content": text})
		}
	}

	// Messages
	if msgs := jsonx.GetArr(req, "messages"); msgs != nil {
		for _, msg := range msgs {
			role := jsonx.GetStr(msg, "role")
			if role == "" {
				role = "user"
			}
			content := jsonx.Get(msg, "content")

			switch c := content.(type) {
			case string:
				messages = append(messages, map[string]any{"role": role, "content": c})
			case []any:
				var textParts []string
				var toolCalls []any
				type toolResult struct{ id, text string }
				var toolResults []toolResult

				for _, block := range c {
					btype := jsonx.GetStr(block, "type")
					switch btype {
					case "text":
						if t, ok := jsonx.AsStr(jsonx.Get(block, "text")); ok {
							textParts = append(textParts, t)
						}
					case "tool_use":
						name := jsonx.GetStr(block, "name")
						id := jsonx.GetStr(block, "id")
						input := jsonx.Get(block, "input")
						if input == nil {
							input = map[string]any{}
						}
						argsStr := jsonx.MarshalString(input)
						if argsStr == "" {
							argsStr = "{}"
						}
						toolCalls = append(toolCalls, map[string]any{
							"id": id, "type": "function",
							"function": map[string]any{"name": name, "arguments": argsStr},
						})
					case "tool_result":
						toolUseID := jsonx.GetStr(block, "tool_use_id")
						resultContent := ""
						if rc := jsonx.Get(block, "content"); rc != nil {
							resultContent = extractAnthropicText(rc)
						}
						isError := jsonx.GetBool(block, "is_error")
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
	if mt, ok := jsonx.GetNum(req, "max_tokens"); ok {
		maxTokens = mt
	}
	stream := jsonx.GetBool(req, "stream")

	result := map[string]any{
		"model":      jsonx.GetStr(req, "model"),
		"messages":   messages,
		"max_tokens": maxTokens,
		"stream":     stream,
	}

	if temp := jsonx.Get(req, "temperature"); temp != nil {
		result["temperature"] = temp
	}
	if topP := jsonx.Get(req, "top_p"); topP != nil {
		result["top_p"] = topP
	}
	if stop := jsonx.Get(req, "stop_sequences"); stop != nil {
		result["stop"] = stop
	}

	// Pass through sampling params for local model control.
	for _, key := range []string{"repetition_penalty", "repetition_context_size", "top_k", "min_p", "frequency_penalty", "presence_penalty"} {
		if v := jsonx.Get(req, key); v != nil {
			result[key] = v
		}
	}

	// Convert Anthropic tools → OpenAI format.
	if toollist := jsonx.GetArr(req, "tools"); len(toollist) > 0 {
		openaiTools := make([]any, 0, len(toollist))
		for _, t := range toollist {
			if jsonx.GetStr(t, "type") == "function" {
				openaiTools = append(openaiTools, t)
			} else {
				inputSchema := jsonx.Get(t, "input_schema")
				if inputSchema == nil {
					inputSchema = map[string]any{"type": "object"}
				}
				openaiTools = append(openaiTools, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name":        jsonx.GetStr(t, "name"),
						"description": jsonx.GetStr(t, "description"),
						"parameters":  inputSchema,
					},
				})
			}
		}
		result["tools"] = openaiTools
	}

	if choice := jsonx.Get(req, "tool_choice"); choice != nil {
		switch obj := choice.(type) {
		case map[string]any:
			ctype := jsonx.GetStr(obj, "type")
			if ctype == "" {
				ctype = "auto"
			}
			switch ctype {
			case "auto":
				result["tool_choice"] = "auto"
			case "any":
				result["tool_choice"] = "required"
			case "tool":
				if name := jsonx.GetStr(obj, "name"); name != "" {
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

	if jsonx.GetBool(result, "stream") {
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
			if jsonx.GetStr(b, "type") == "text" {
				if t, ok := jsonx.AsStr(jsonx.Get(b, "text")); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "")
	default:
		return ""
	}
}
