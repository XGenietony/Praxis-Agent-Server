package tools

import (
	"strings"
	"testing"

	"lmstudio-forward/internal/jsonx"
)

// Tool call parsing (behavioral, from src/tools.rs).

func TestParseToolCallsBasic(t *testing.T) {
	text := `Some preamble <tool_call>
{"name": "Write", "arguments": {"file_path": "/tmp/x", "content": "hi"}}
</tool_call> trailing`
	calls := ParseToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(calls))
	}
	if calls[0].Name != "Write" {
		t.Fatalf("name: want Write, got %q", calls[0].Name)
	}
	if !strings.Contains(calls[0].Arguments, "file_path") {
		t.Fatalf("arguments missing file_path: %q", calls[0].Arguments)
	}
}

func TestStripToolCalls(t *testing.T) {
	text := `before<tool_call>{"name":"X"}</tool_call>after`
	got := stripToolCalls(text)
	if got != "beforeafter" {
		t.Fatalf("want beforeafter, got %q", got)
	}
}

func TestStreamParserSplitTag(t *testing.T) {
	p := NewStreamParser()
	var text strings.Builder
	var calls int
	// Feed a tool call split across chunk boundaries, including inside the tag.
	chunks := []string{"hello ", "<tool_", "call>", `{"name": "ret`, `rieve", "arguments": {"query": "x"}}`, "</tool", "_call>", " bye"}
	for _, c := range chunks {
		for _, ev := range p.Feed(c) {
			switch ev.Kind {
			case EventText:
				text.WriteString(ev.Text)
			case EventCall:
				calls++
				if ev.Call.Name != "retrieve" {
					t.Fatalf("call name: want retrieve, got %q", ev.Call.Name)
				}
			}
		}
	}
	for _, ev := range p.Flush() {
		if ev.Kind == EventText {
			text.WriteString(ev.Text)
		}
	}
	if calls != 1 {
		t.Fatalf("want 1 streamed call, got %d", calls)
	}
	if got := text.String(); got != "hello  bye" {
		t.Fatalf("surrounding text: want %q, got %q", "hello  bye", got)
	}
	if !p.HasSeenTools() {
		t.Fatalf("HasSeenTools should be true")
	}
}

func TestFixJSONRepairs(t *testing.T) {
	// trailing comma
	if v, ok := fixJSON(`{"name":"a","arguments":{"x":1},}`); !ok || jsonx.GetStr(v, "name") != "a" {
		t.Fatalf("trailing comma repair failed: ok=%v v=%v", ok, v)
	}
	// missing closing brace
	if v, ok := fixJSON(`{"name":"a","arguments":{"x":1}`); !ok || jsonx.GetStr(v, "name") != "a" {
		t.Fatalf("missing brace repair failed: ok=%v v=%v", ok, v)
	}
	// single quotes (no double quotes present)
	if v, ok := fixJSON(`{'name':'a'}`); !ok || jsonx.GetStr(v, "name") != "a" {
		t.Fatalf("single-quote repair failed: ok=%v v=%v", ok, v)
	}
}

func TestAnthropicToOpenAIBasic(t *testing.T) {
	req := map[string]any{
		"model":      "claude-x",
		"max_tokens": 100.0,
		"system":     "be brief",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
		},
	}
	out := AnthropicRequestToOpenAI(req)
	msgs := jsonx.AsArr(out["messages"])
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages (system+user), got %d", len(msgs))
	}
	if jsonx.GetStr(msgs[0], "role") != "system" || jsonx.GetStr(msgs[0], "content") != "be brief" {
		t.Fatalf("system message wrong: %v", msgs[0])
	}
	if jsonx.GetStr(msgs[1], "role") != "user" {
		t.Fatalf("user message wrong: %v", msgs[1])
	}
}
