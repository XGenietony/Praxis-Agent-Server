package tools

import (
	"fmt"
	"strings"

	"lmstudio-forward/internal/jsonx"
)

// TransformResponse parses model output for `<tool_call>` tags and rewrites the
// response into native OpenAI tool_calls shape.
func TransformResponse(respBody []byte) []byte {
	root, err := jsonx.Parse(respBody)
	if err != nil {
		return respBody
	}

	choices := jsonx.GetArr(root, "choices")
	if choices == nil {
		return respBody
	}

	modified := false
	for _, ch := range choices {
		choice := jsonx.AsObj(ch)
		if choice == nil {
			continue
		}
		msg := jsonx.AsObj(choice["message"])
		if msg == nil {
			continue
		}
		content, ok := jsonx.AsStr(msg["content"])
		if !ok {
			continue
		}

		toolCalls := ParseToolCalls(content)
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
		if out := jsonx.Marshal(root); out != nil {
			return out
		}
	}
	return respBody
}

// ParseToolCalls extracts all complete `<tool_call>...</tool_call>` blocks.
func ParseToolCalls(text string) []ToolCall {
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
			name := jsonx.GetStr(parsed, "name")
			arguments := "{}"
			if a := jsonx.Get(parsed, "arguments"); a != nil {
				arguments = jsonx.MarshalString(a)
				if arguments == "" {
					arguments = "{}"
				}
			}
			id := jsonx.GetStr(parsed, "call_id")
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
