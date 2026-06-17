package tools

import (
	"log"
	"regexp"
	"strings"

	"lmstudio-forward/internal/jsonx"
)

var (
	reTrailingComma = regexp.MustCompile(`,\s*([}\]])`)
	reFuzzyName     = regexp.MustCompile(`"name"\s*:\s*"([^"]+)"`)
	reFuzzyArgs     = regexp.MustCompile(`"arguments"\s*:\s*(\{)`)
)

// fixJSON attempts to repair common JSON errors emitted by small models.
// Returns (value, true) on success.
func fixJSON(raw string) (any, bool) {
	// 1. Try as-is first.
	if v, err := jsonx.Parse([]byte(raw)); err == nil {
		return v, true
	}

	s := raw

	// 2. Replace single quotes with double quotes (only if no double quotes present).
	if !strings.Contains(s, "\"") {
		s = strings.ReplaceAll(s, "'", "\"")
		if v, err := jsonx.Parse([]byte(s)); err == nil {
			return v, true
		}
	}

	// 3. Fix unescaped literal newlines/tabs inside JSON string values.
	s = fixUnescapedNewlines(s)
	if v, err := jsonx.Parse([]byte(s)); err == nil {
		return v, true
	}

	// 4. Remove trailing commas before } or ].
	s = reTrailingComma.ReplaceAllString(s, "$1")
	if v, err := jsonx.Parse([]byte(s)); err == nil {
		return v, true
	}

	// 5. Try to fix missing closing braces.
	openBraces := strings.Count(s, "{")
	closeBraces := strings.Count(s, "}")
	if openBraces > closeBraces {
		s += strings.Repeat("}", openBraces-closeBraces)
		if v, err := jsonx.Parse([]byte(s)); err == nil {
			return v, true
		}
	}

	// 6. Last resort: fuzzy-extract name + arguments.
	if v, ok := extractToolCallFuzzy(raw); ok {
		return v, true
	}

	limit := len(raw)
	if limit > 200 {
		limit = 200
	}
	log.Printf("WARN fix_json: could not repair: %s...", raw[:limit])
	return nil, false
}

// fixUnescapedNewlines escapes literal newlines/tabs inside JSON string values.
func fixUnescapedNewlines(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	escapeNext := false
	for _, ch := range s {
		if escapeNext {
			b.WriteRune(ch)
			escapeNext = false
			continue
		}
		if ch == '\\' && inString {
			b.WriteRune(ch)
			escapeNext = true
			continue
		}
		if ch == '"' {
			inString = !inString
			b.WriteRune(ch)
			continue
		}
		if inString && ch == '\n' {
			b.WriteString("\\n")
		} else if inString && ch == '\r' {
			// skip \r; covered by \n
		} else if inString && ch == '\t' {
			b.WriteString("\\t")
		} else {
			b.WriteRune(ch)
		}
	}
	return b.String()
}

// extractToolCallFuzzy is a last-resort extractor for "name" and "arguments".
func extractToolCallFuzzy(raw string) (any, bool) {
	nameMatch := reFuzzyName.FindStringSubmatch(raw)
	if nameMatch == nil {
		return nil, false
	}
	name := nameMatch[1]

	if loc := reFuzzyArgs.FindStringIndex(raw); loc != nil {
		// Find the first '{' at/after the match start.
		rel := strings.Index(raw[loc[0]:], "{")
		if rel < 0 {
			return map[string]any{"name": name, "arguments": map[string]any{}}, true
		}
		start := loc[0] + rel
		// Find the matching closing brace.
		depth := 0
		inStr := false
		esc := false
		end := start
		for i, ch := range raw[start:] {
			if esc {
				esc = false
				continue
			}
			if ch == '\\' && inStr {
				esc = true
				continue
			}
			if ch == '"' {
				inStr = !inStr
				continue
			}
			if !inStr {
				if ch == '{' {
					depth++
				}
				if ch == '}' {
					depth--
					if depth == 0 {
						end = start + i + 1
						break
					}
				}
			}
		}
		if end > start {
			argsRaw := raw[start:end]
			var args any = map[string]any{}
			if v, err := jsonx.Parse([]byte(argsRaw)); err == nil {
				args = v
			} else {
				fixed := fixUnescapedNewlines(argsRaw)
				if v, err := jsonx.Parse([]byte(fixed)); err == nil {
					args = v
				}
			}
			return map[string]any{"name": name, "arguments": args}, true
		}
	}

	// Fallback: no arguments found.
	return map[string]any{"name": name, "arguments": map[string]any{}}, true
}
