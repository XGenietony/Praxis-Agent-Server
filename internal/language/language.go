// Package language provides token estimation, context-window truncation, and
// conversation-complexity scoring used to drive dynamic thinking control.
package language

import (
	"log"
	"strings"

	"lmstudio-forward/internal/jsonx"
)

// EstimateTokens gives a rough estimate of the token count for a string.
// Chinese (CJK) chars count as ~1 token each; non-CJK chars as ~1 token per 4.
// Mirrors the Rust `estimate_tokens`.
func EstimateTokens(text string) int {
	chars := 0
	cjk := 0
	for _, c := range text {
		chars++
		if c > '⺀' {
			cjk++
		}
	}
	return cjk + (chars-cjk)/4 + 1
}

// TruncateMessages trims the messages array to fit within maxTokens. It keeps
// leading system message(s) plus the trailing messages that fit, reserving 2048
// tokens for model output. Returns a (possibly new) slice; Go can't reassign the
// caller's variable, so callers do:
//
//	body["messages"] = language.TruncateMessages(jsonx.GetArr(body, "messages"), ctx)
//
// Mirrors the Rust `truncate_messages`.
func TruncateMessages(messages []any, maxTokens int) []any {
	if len(messages) == 0 || maxTokens == 0 {
		return messages
	}

	// Reserve tokens for model output (at least 2048).
	inputBudget := maxTokens - 2048
	if inputBudget <= 0 {
		return messages
	}

	// Calculate tokens for each message.
	msgTokens := make([]int, len(messages))
	for i, m := range messages {
		content := jsonx.GetStr(m, "content")
		msgTokens[i] = EstimateTokens(content) + 4 // overhead for role, formatting
	}

	total := 0
	for _, t := range msgTokens {
		total += t
	}
	if total <= inputBudget {
		return messages // fits, no truncation needed
	}

	// Always keep system message(s) at the front.
	systemEnd := 0
	systemTokens := 0
	for i, m := range messages {
		if jsonx.GetStr(m, "role") == "system" {
			systemEnd = i + 1
			systemTokens += msgTokens[i]
		} else {
			break
		}
	}

	remainingBudget := inputBudget - systemTokens
	if remainingBudget < 0 {
		remainingBudget = 0
	}

	// Take messages from the end until budget is exceeded.
	nonSystem := msgTokens[systemEnd:]
	keepFrom := len(nonSystem)
	used := 0
	for i := len(nonSystem) - 1; i >= 0; i-- {
		tokens := nonSystem[i]
		if used+tokens > remainingBudget {
			break
		}
		used += tokens
		keepFrom = i
	}

	dropCount := keepFrom
	if dropCount == 0 {
		return messages
	}

	// Remove old messages (keep system + recent).
	kept := make([]any, 0, systemEnd+(len(messages)-(systemEnd+dropCount)))
	kept = append(kept, messages[:systemEnd]...)
	kept = append(kept, messages[systemEnd+dropCount:]...)

	originalLen := len(messages)
	dropped := originalLen - len(kept)

	droppedTokens := 0
	for _, t := range msgTokens[systemEnd : systemEnd+dropCount] {
		droppedTokens += t
	}

	log.Printf("INFO Context truncation: dropped %d old messages, ~%d tokens → ~%d tokens (budget: %d)",
		dropped, total, total-droppedTokens, inputBudget)

	return kept
}

// EstimateComplexity scores the conversation and returns true when it is complex
// enough to warrant deep reasoning (score >= 3). Mirrors `estimate_complexity`.
func EstimateComplexity(messages []any) bool {
	var userMessages []string
	for _, m := range messages {
		if jsonx.GetStr(m, "role") != "user" {
			continue
		}
		if s, ok := jsonx.AsStr(jsonx.Get(m, "content")); ok {
			userMessages = append(userMessages, s)
		}
	}

	if len(userMessages) == 0 {
		return false
	}
	lastUser := userMessages[len(userMessages)-1]

	var contentParts []string
	for _, m := range messages {
		if s, ok := jsonx.AsStr(jsonx.Get(m, "content")); ok {
			contentParts = append(contentParts, s)
		}
	}
	allContent := strings.Join(contentParts, "\n")

	score := 0

	// 1. Total context size.
	totalLen := len(allContent)
	if totalLen > 8000 {
		score += 3
	} else if totalLen > 3000 {
		score += 2
	} else if totalLen > 1000 {
		score += 1
	}

	// 2. Conversation depth.
	turnCount := len(messages)
	if turnCount > 6 {
		score += 2
	} else if turnCount > 2 {
		score += 1
	}

	// 3. Last user message length.
	lastLen := len(lastUser)
	if lastLen > 2000 {
		score += 3
	} else if lastLen > 500 {
		score += 2
	} else if lastLen > 150 {
		score += 1
	}

	// 4. Code blocks.
	codeBlockCount := strings.Count(allContent, "```") / 2
	if codeBlockCount >= 3 {
		score += 3
	} else if codeBlockCount >= 1 {
		score += 2
	}

	// 5. Complexity keywords (CN + EN).
	complexKeywords := []string{
		"analyze", "debug", "refactor", "implement", "algorithm", "optimize",
		"architecture", "design", "prove", "derive", "calculate", "explain why",
		"step by step", "trade-off", "compare", "evaluate", "fix", "error",
		"分析", "调试", "重构", "实现", "算法", "优化",
		"架构", "设计", "证明", "推导", "计算", "为什么",
		"逐步", "对比", "评估", "解释原理", "怎么做", "修复", "报错",
	}
	lastLower := strings.ToLower(lastUser)
	keywordHits := 0
	for _, kw := range complexKeywords {
		if strings.Contains(lastLower, strings.ToLower(kw)) {
			keywordHits++
		}
	}
	score += keywordHits

	// 6. Math / logic indicators.
	if strings.Contains(lastUser, "$$") || strings.Contains(lastUser, "\\frac") || strings.Contains(lastUser, "\\sum") {
		score += 2
	}

	// 7. System prompt length.
	systemLen := 0
	for _, m := range messages {
		if jsonx.GetStr(m, "role") != "system" {
			continue
		}
		if s, ok := jsonx.AsStr(jsonx.Get(m, "content")); ok {
			systemLen += len(s)
		}
	}
	if systemLen > 1000 {
		score += 2
	} else if systemLen > 300 {
		score += 1
	}

	return score >= 3
}
