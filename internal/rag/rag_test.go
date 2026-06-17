package rag

import (
	"reflect"
	"strings"
	"testing"
)

// ChunkText (ported from src/rag.rs #[cfg(test)]).

func TestEmptyTextYieldsNoChunks(t *testing.T) {
	if got := ChunkText("", 100, 10); len(got) != 0 {
		t.Fatalf("empty: want 0 chunks, got %v", got)
	}
	if got := ChunkText("   ", 100, 10); len(got) != 0 {
		t.Fatalf("whitespace: want 0 chunks, got %v", got)
	}
}

func TestShortTextYieldsSingleChunk(t *testing.T) {
	got := ChunkText("hello world", 100, 10)
	want := []string{"hello world"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("want %v, got %v", want, got)
	}
}

func TestLongTextSplitsWithOverlap(t *testing.T) {
	text := strings.Repeat("abcdefghij", 3) // 30 chars
	got := ChunkText(text, 10, 4)           // step=6 → starts 0,6,12,18,24 = 5 chunks
	if len(got) != 5 {
		t.Fatalf("want 5 chunks, got %d (%v)", len(got), got)
	}
	if n := len([]rune(got[0])); n != 10 {
		t.Fatalf("first chunk: want 10 runes, got %d", n)
	}
}

func TestOverlapClampedBelowSize(t *testing.T) {
	got := ChunkText("abcdefghij", 5, 100) // must not panic / infinite-loop
	if len(got) == 0 {
		t.Fatalf("want non-empty chunks")
	}
}
