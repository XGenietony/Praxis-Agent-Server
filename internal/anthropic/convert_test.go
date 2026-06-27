package anthropic

import (
	"strings"
	"testing"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/tools"
)

func TestOpenAIToAnthropicRejectsMultipleChoices(t *testing.T) {
	resp := map[string]any{
		"choices": []any{
			map[string]any{"message": map[string]any{"content": "a"}},
			map[string]any{"message": map[string]any{"content": "b"}},
		},
	}
	_, err := openaiToAnthropicStrict(resp, "model")
	if err == nil || !strings.Contains(err.Error(), "multiple choices") {
		t.Fatalf("want multiple-choice conversion error, got %v", err)
	}
}

func TestOpenAIToAnthropicUsesSharedToolCallIDFallback(t *testing.T) {
	resp := map[string]any{
		"choices": []any{map[string]any{
			"message": map[string]any{
				"tool_calls": []any{map[string]any{
					"type":     "function",
					"function": map[string]any{"name": "retrieve", "arguments": `{"query":"x"}`},
				}},
			},
		}},
	}
	out := openaiToAnthropic(resp, "model")
	content := jsonx.AsArr(jsonx.Get(out, "content"))
	if len(content) != 1 {
		t.Fatalf("want 1 content block, got %d", len(content))
	}
	if got := jsonx.GetStr(content[0], "id"); got != "toolu_local_0" {
		t.Fatalf("want shared fallback ID, got %q", got)
	}
}

func TestOpenAIChunkToAnthropicAccumulatesNativeToolCallDeltas(t *testing.T) {
	state := &streamState{}
	parser := tools.NewStreamParser()
	chunks := []any{
		map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": float64(0), "id": "call_1", "function": map[string]any{"name": "retrieve", "arguments": `{"query"`}}}}}}},
		map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": float64(0), "function": map[string]any{"arguments": `:"golang"}`}}}}}}},
		map[string]any{"choices": []any{map[string]any{"delta": map[string]any{}, "finish_reason": "tool_calls"}}},
	}
	var events []any
	for _, chunk := range chunks {
		events = append(events, openaiChunkToAnthropic(chunk, state, "model", parser)...)
	}
	var sawToolUse bool
	var sawInput bool
	for _, ev := range events {
		switch jsonx.GetStr(ev, "type") {
		case "content_block_start":
			block := jsonx.Get(ev, "content_block")
			if jsonx.GetStr(block, "type") == "tool_use" && jsonx.GetStr(block, "name") == "retrieve" {
				sawToolUse = true
			}
		case "content_block_delta":
			delta := jsonx.Get(ev, "delta")
			if strings.Contains(jsonx.GetStr(delta, "partial_json"), "golang") {
				sawInput = true
			}
		}
	}
	if !sawToolUse || !sawInput {
		t.Fatalf("expected accumulated native tool_use events, got %#v", events)
	}
}
