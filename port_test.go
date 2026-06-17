package main

import (
	"reflect"
	"strings"
	"testing"
)

// ── chunk_text (ported from src/rag.rs #[cfg(test)]) ──────────────────────────

func TestEmptyTextYieldsNoChunks(t *testing.T) {
	if got := chunkText("", 100, 10); len(got) != 0 {
		t.Fatalf("empty: want 0 chunks, got %v", got)
	}
	if got := chunkText("   ", 100, 10); len(got) != 0 {
		t.Fatalf("whitespace: want 0 chunks, got %v", got)
	}
}

func TestShortTextYieldsSingleChunk(t *testing.T) {
	got := chunkText("hello world", 100, 10)
	want := []string{"hello world"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("want %v, got %v", want, got)
	}
}

func TestLongTextSplitsWithOverlap(t *testing.T) {
	text := strings.Repeat("abcdefghij", 3) // 30 chars
	got := chunkText(text, 10, 4)           // step=6 → starts 0,6,12,18,24 = 5 chunks
	if len(got) != 5 {
		t.Fatalf("want 5 chunks, got %d (%v)", len(got), got)
	}
	if n := len([]rune(got[0])); n != 10 {
		t.Fatalf("first chunk: want 10 runes, got %d", n)
	}
}

func TestOverlapClampedBelowSize(t *testing.T) {
	got := chunkText("abcdefghij", 5, 100) // must not panic / infinite-loop
	if len(got) == 0 {
		t.Fatalf("want non-empty chunks")
	}
}

// ── tool call parsing (behavioral, from src/tools.rs) ─────────────────────────

func TestParseToolCallsBasic(t *testing.T) {
	text := `Some preamble <tool_call>
{"name": "Write", "arguments": {"file_path": "/tmp/x", "content": "hi"}}
</tool_call> trailing`
	calls := parseToolCalls(text)
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
	p := newToolCallStreamParser()
	var text strings.Builder
	var calls int
	// Feed a tool call split across chunk boundaries, including inside the tag.
	chunks := []string{"hello ", "<tool_", "call>", `{"name": "ret`, `rieve", "arguments": {"query": "x"}}`, "</tool", "_call>", " bye"}
	for _, c := range chunks {
		for _, ev := range p.Feed(c) {
			switch ev.Kind {
			case ToolEventText:
				text.WriteString(ev.Text)
			case ToolEventCall:
				calls++
				if ev.Call.Name != "retrieve" {
					t.Fatalf("call name: want retrieve, got %q", ev.Call.Name)
				}
			}
		}
	}
	for _, ev := range p.Flush() {
		if ev.Kind == ToolEventText {
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
	if v, ok := fixJSON(`{"name":"a","arguments":{"x":1},}`); !ok || getStr(v, "name") != "a" {
		t.Fatalf("trailing comma repair failed: ok=%v v=%v", ok, v)
	}
	// missing closing brace
	if v, ok := fixJSON(`{"name":"a","arguments":{"x":1}`); !ok || getStr(v, "name") != "a" {
		t.Fatalf("missing brace repair failed: ok=%v v=%v", ok, v)
	}
	// single quotes (no double quotes present)
	if v, ok := fixJSON(`{'name':'a'}`); !ok || getStr(v, "name") != "a" {
		t.Fatalf("single-quote repair failed: ok=%v v=%v", ok, v)
	}
}

// ── token estimation (from src/language.rs) ───────────────────────────────────

func TestEstimateTokens(t *testing.T) {
	// Pure ASCII: chars=8, cjk=0 → 0 + 8/4 + 1 = 3
	if got := estimateTokens("abcdefgh"); got != 3 {
		t.Fatalf("ascii: want 3, got %d", got)
	}
	// CJK: 4 chars all > U+2E80 → 4 + 0 + 1 = 5
	if got := estimateTokens("你好世界"); got != 5 {
		t.Fatalf("cjk: want 5, got %d", got)
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
	out := anthropicRequestToOpenAI(req)
	msgs := asArr(out["messages"])
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages (system+user), got %d", len(msgs))
	}
	if getStr(msgs[0], "role") != "system" || getStr(msgs[0], "content") != "be brief" {
		t.Fatalf("system message wrong: %v", msgs[0])
	}
	if getStr(msgs[1], "role") != "user" {
		t.Fatalf("user message wrong: %v", msgs[1])
	}
}
