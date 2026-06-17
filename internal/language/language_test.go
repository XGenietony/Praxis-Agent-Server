package language

import "testing"

// Token estimation (from src/language.rs).

func TestEstimateTokens(t *testing.T) {
	// Pure ASCII: chars=8, cjk=0 → 0 + 8/4 + 1 = 3
	if got := EstimateTokens("abcdefgh"); got != 3 {
		t.Fatalf("ascii: want 3, got %d", got)
	}
	// CJK: 4 chars all > U+2E80 → 4 + 0 + 1 = 5
	if got := EstimateTokens("你好世界"); got != 5 {
		t.Fatalf("cjk: want 5, got %d", got)
	}
}
