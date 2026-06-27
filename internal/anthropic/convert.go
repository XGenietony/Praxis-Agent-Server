package anthropic

import (
	"errors"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/tools"
)

// openaiToAnthropic converts an OpenAI response to Anthropic format (non-streaming).
func openaiToAnthropic(resp any, model string) any {
	out, _ := openaiToAnthropicStrict(resp, model)
	return out
}

func openaiToAnthropicStrict(resp any, model string) (any, error) {
	choices := jsonx.AsArr(jsonx.Get(resp, "choices"))
	choice, err := primaryChoice(choices)
	if err != nil {
		return nil, err
	}
	message := jsonx.Get(choice, "message")
	usage := jsonx.Get(resp, "usage")

	var contentBlocks []any

	reasoning, ok := jsonx.AsStr(jsonx.Get(message, "reasoning_content"))
	if !ok {
		reasoning, ok = jsonx.AsStr(jsonx.Get(message, "reasoning"))
	}
	if ok && reasoning != "" {
		contentBlocks = append(contentBlocks, map[string]any{"type": "thinking", "thinking": reasoning})
	}

	text := jsonx.Str(jsonx.Get(message, "content"))
	if text != "" {
		contentBlocks = append(contentBlocks, map[string]any{"type": "text", "text": text})
	}

	if toolCalls := jsonx.AsArr(jsonx.Get(message, "tool_calls")); toolCalls != nil {
		for i, tc := range toolCalls {
			if fn := jsonx.Get(tc, "function"); fn != nil {
				name := jsonx.Str(jsonx.Get(fn, "name"))
				argsStr := "{}"
				if v, ok := jsonx.AsStr(jsonx.Get(fn, "arguments")); ok {
					argsStr = v
				}
				input, err := jsonx.Parse([]byte(argsStr))
				if err != nil {
					input = map[string]any{}
				}
				id := tools.EnsureToolCallID(jsonx.GetStr(tc, "id"), i)
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "tool_use", "id": id, "name": name, "input": input,
				})
			}
		}
	}

	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, map[string]any{"type": "text", "text": ""})
	}

	stopReason := "end_turn"
	switch fr, _ := jsonx.AsStr(jsonx.Get(choice, "finish_reason")); fr {
	case "tool_calls":
		stopReason = "tool_use"
	case "length":
		stopReason = "max_tokens"
	}

	id := "unknown"
	if v, ok := jsonx.AsStr(jsonx.Get(resp, "id")); ok {
		id = v
	}
	promptTokens := jsonx.Get(usage, "prompt_tokens")
	if promptTokens == nil {
		promptTokens = float64(0)
	}
	completionTokens := jsonx.Get(usage, "completion_tokens")
	if completionTokens == nil {
		completionTokens = float64(0)
	}

	return map[string]any{
		"id":            "msg_" + id,
		"type":          "message",
		"role":          "assistant",
		"content":       contentBlocks,
		"model":         model,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage":         map[string]any{"input_tokens": promptTokens, "output_tokens": completionTokens},
	}, nil
}

// openaiChunkToAnthropic converts a single OpenAI SSE chunk to Anthropic SSE events.
func openaiChunkToAnthropic(chunk any, state *streamState, model string, toolParser *tools.StreamParser) []any {
	var events []any

	if !state.started {
		state.started = true
		id := "unknown"
		if v, ok := jsonx.AsStr(jsonx.Get(chunk, "id")); ok {
			id = v
		}
		events = append(events, map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": "msg_" + id, "type": "message", "role": "assistant",
				"content": []any{}, "model": model,
				"stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
			},
		})
	}

	choices := jsonx.AsArr(jsonx.Get(chunk, "choices"))
	var choice any
	if len(choices) > 0 {
		var choiceErr error
		choice, choiceErr = primaryChoice(choices)
		if choiceErr != nil {
			return events
		}
	}
	delta := jsonx.Get(choice, "delta")

	// Reasoning content → thinking block (support both "reasoning_content" and "reasoning")
	reasoning, ok := jsonx.AsStr(jsonx.Get(delta, "reasoning_content"))
	if !ok {
		reasoning, ok = jsonx.AsStr(jsonx.Get(delta, "reasoning"))
	}
	if ok && reasoning != "" {
		if !state.thinkingStarted {
			state.thinkingStarted = true
			events = append(events, map[string]any{
				"type": "content_block_start", "index": state.blockIndex,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
		}
		events = append(events, map[string]any{
			"type": "content_block_delta", "index": state.blockIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": reasoning},
		})
	}

	// Text content → feed through tool parser
	if text, ok := jsonx.AsStr(jsonx.Get(delta, "content")); ok && text != "" {
		for _, te := range toolParser.Feed(text) {
			events = applyToolEvent(te, state, events)
		}
	}

	// Native OpenAI streaming tool calls are incremental by index. Accumulate
	// their ID/name/arguments fragments and emit Anthropic tool_use blocks once
	// the backend reports tool_calls as the finish reason.
	if toolCalls := jsonx.AsArr(jsonx.Get(delta, "tool_calls")); len(toolCalls) > 0 {
		accumulateNativeToolDeltas(state, toolCalls)
	}

	// Finish reason
	if finish, ok := jsonx.AsStr(jsonx.Get(choice, "finish_reason")); ok && finish != "" {
		if finish == "tool_calls" {
			for _, tc := range drainNativeToolDeltas(state) {
				events = applyToolEvent(tools.StreamEvent{Kind: tools.EventCall, Call: tc}, state, events)
			}
		}

		// Flush tool parser
		for _, te := range toolParser.Flush() {
			events = applyToolEvent(te, state, events)
		}

		stopReason := "end_turn"
		if toolParser.HasSeenTools() || finish == "tool_calls" {
			stopReason = "tool_use"
		} else if finish == "length" {
			stopReason = "max_tokens"
		}

		if state.thinkingStarted || state.textStarted {
			events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
			state.thinkingStarted = false
			state.textStarted = false
		}

		if usage := jsonx.Get(chunk, "usage"); usage != nil {
			ct := jsonx.Get(usage, "completion_tokens")
			if ct == nil {
				ct = float64(0)
			}
			events = append(events, map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
				"usage": map[string]any{"output_tokens": ct},
			})
			events = append(events, map[string]any{"type": "message_stop"})
			state.finished = true
		} else {
			state.pendingStopReason = stopReason
		}
	}

	// Deferred finish: usage-only chunk
	if state.pendingStopReason != "" {
		stopReason := state.pendingStopReason
		state.pendingStopReason = ""
		if usage := jsonx.Get(chunk, "usage"); usage != nil {
			ct := jsonx.Get(usage, "completion_tokens")
			if ct == nil {
				ct = float64(0)
			}
			events = append(events, map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
				"usage": map[string]any{"output_tokens": ct},
			})
			events = append(events, map[string]any{"type": "message_stop"})
			state.finished = true
		} else {
			state.pendingStopReason = stopReason
		}
	}

	return events
}

// applyToolEvent appends the Anthropic events produced by a single tool stream
// event, mutating state as needed.
func applyToolEvent(te tools.StreamEvent, state *streamState, events []any) []any {
	switch te.Kind {
	case tools.EventText:
		if te.Text != "" {
			events = ensureTextBlock(state, events)
			events = append(events, map[string]any{
				"type": "content_block_delta", "index": state.blockIndex,
				"delta": map[string]any{"type": "text_delta", "text": te.Text},
			})
		}
	case tools.EventCall:
		tc := te.Call
		if state.textStarted {
			events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
			state.blockIndex++
			state.textStarted = false
		}
		input, err := jsonx.Parse([]byte(tc.Arguments))
		if err != nil {
			input = map[string]any{}
		}
		id := tools.EnsureToolCallID(tc.ID, state.toolOrdinal)
		state.toolOrdinal++
		events = append(events, map[string]any{
			"type": "content_block_start", "index": state.blockIndex,
			"content_block": map[string]any{"type": "tool_use", "id": id, "name": tc.Name, "input": map[string]any{}},
		})
		events = append(events, map[string]any{
			"type": "content_block_delta", "index": state.blockIndex,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": jsonx.MarshalString(input)},
		})
		events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
		state.blockIndex++
	}
	return events
}

func ensureTextBlock(state *streamState, events []any) []any {
	if !state.textStarted {
		if state.thinkingStarted {
			events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
			state.blockIndex++
			state.thinkingStarted = false
		}
		state.textStarted = true
		events = append(events, map[string]any{
			"type": "content_block_start", "index": state.blockIndex,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
	}
	return events
}

var (
	errNoChoices       = errors.New("backend returned no choices")
	errMultipleChoices = errors.New("backend returned multiple choices")
)

func primaryChoice(choices []any) (any, error) {
	if len(choices) == 0 {
		return nil, errNoChoices
	}
	if len(choices) > 1 {
		return nil, errMultipleChoices
	}
	return choices[0], nil
}

func accumulateNativeToolDeltas(state *streamState, toolCalls []any) {
	if state.nativeTools == nil {
		state.nativeTools = map[int]*nativeToolDelta{}
	}
	for _, tc := range toolCalls {
		idx := 0
		if rawIdx, ok := jsonx.GetNum(tc, "index"); ok {
			idx = int(rawIdx)
		}
		cur := state.nativeTools[idx]
		if cur == nil {
			cur = &nativeToolDelta{}
			state.nativeTools[idx] = cur
		}
		if id := jsonx.GetStr(tc, "id"); id != "" {
			cur.id = id
		}
		fn := jsonx.Get(tc, "function")
		if name := jsonx.GetStr(fn, "name"); name != "" {
			cur.name = name
		}
		cur.arguments += jsonx.GetStr(fn, "arguments")
	}
}

func drainNativeToolDeltas(state *streamState) []tools.ToolCall {
	if len(state.nativeTools) == 0 {
		return nil
	}
	calls := make([]tools.ToolCall, 0, len(state.nativeTools))
	for i := 0; i < len(state.nativeTools); i++ {
		cur := state.nativeTools[i]
		if cur == nil || cur.name == "" {
			continue
		}
		calls = append(calls, tools.ToolCall{ID: cur.id, Name: cur.name, Arguments: cur.arguments})
	}
	state.nativeTools = nil
	return calls
}
