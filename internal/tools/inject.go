package tools

import (
	"fmt"
	"log"
	"strings"

	"lmstudio-forward/internal/jsonx"
)

// TransformRequest injects tools into the system prompt and removes the
// "tools"/"tool_choice" fields. It mutates data in place (maps are references)
// and returns true if the request was modified.
func TransformRequest(data map[string]any) bool {
	toolsAny := data["tools"]
	toollist := jsonx.AsArr(toolsAny)
	if len(toollist) == 0 {
		return false
	}

	toolChoice, hasChoice := data["tool_choice"]
	var choicePtr any
	if hasChoice {
		choicePtr = toolChoice
	}
	toolsPrompt := buildToolsPrompt(toollist, choicePtr, hasChoice)

	if messages := jsonx.AsArr(data["messages"]); messages != nil {
		transformMessages(messages)

		hasSystem := false
		if len(messages) > 0 {
			if jsonx.GetStr(messages[0], "role") == "system" {
				hasSystem = true
			}
		}

		if hasSystem {
			sys := jsonx.AsObj(messages[0])
			if sys != nil {
				if content, ok := jsonx.AsStr(sys["content"]); ok {
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

	log.Printf("INFO Tool adaptation: injected %d tools into system prompt", len(toollist))
	return true
}

func buildToolsPrompt(toollist []any, toolChoice any, hasChoice bool) string {
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

	for _, tool := range toollist {
		var name, desc, params string
		if fn := jsonx.GetObj(tool, "function"); fn != nil {
			name = orDefault(jsonx.GetStr(fn, "name"), "unknown")
			desc = jsonx.GetStr(fn, "description")
			if p, ok := fn["parameters"]; ok {
				params = string(jsonx.MarshalPretty(p))
			}
		} else {
			name = orDefault(jsonx.GetStr(tool, "name"), "unknown")
			desc = jsonx.GetStr(tool, "description")
			if p := jsonx.Get(tool, "input_schema"); p != nil {
				params = string(jsonx.MarshalPretty(p))
			} else if p := jsonx.Get(tool, "parameters"); p != nil {
				params = string(jsonx.MarshalPretty(p))
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
			if fn := jsonx.AsObj(choice["function"]); fn != nil {
				funcName = jsonx.GetStr(fn, "name")
			}
			if funcName == "" {
				funcName = jsonx.GetStr(choice, "name")
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
		msg := jsonx.AsObj(m)
		if msg == nil {
			continue
		}
		role := jsonx.GetStr(msg, "role")
		switch role {
		case "tool":
			toolCallID := jsonx.GetStr(msg, "tool_call_id")
			content := jsonx.GetStr(msg, "content")
			msg["role"] = "user"
			msg["content"] = fmt.Sprintf("[Tool Result (call_id: %s)]:\n%s", toolCallID, content)
			delete(msg, "tool_call_id")
		case "assistant":
			toolCalls := jsonx.AsArr(msg["tool_calls"])
			if toolCalls != nil {
				var textParts []string
				if content, ok := jsonx.AsStr(msg["content"]); ok && content != "" {
					textParts = append(textParts, content)
				}
				for _, tc := range toolCalls {
					if fn := jsonx.GetObj(tc, "function"); fn != nil {
						name := jsonx.GetStr(fn, "name")
						args := jsonx.GetStr(fn, "arguments")
						if args == "" {
							args = "{}"
						}
						id := jsonx.GetStr(tc, "id")
						// Re-serialize the arguments JSON if it parses, else keep as-is.
						argsJSON := args
						if v, err := jsonx.Parse([]byte(args)); err == nil {
							argsJSON = jsonx.MarshalString(v)
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
