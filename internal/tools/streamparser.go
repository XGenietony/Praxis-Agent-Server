package tools

import (
	"strings"
	"unicode/utf8"

	"lmstudio-forward/internal/jsonx"
)

// StreamParser incrementally parses `<tool_call>` tags out of a token stream,
// emitting text and completed tool calls.
type StreamParser struct {
	buf       string
	inTag     bool
	tagBuf    string
	toolIndex int
}

// NewStreamParser creates an empty streaming tool-call parser.
func NewStreamParser() *StreamParser {
	return &StreamParser{}
}

// Feed appends text to the parser's buffer and returns any events that became
// complete as a result.
func (p *StreamParser) Feed(text string) []StreamEvent {
	p.buf += text
	var events []StreamEvent

	for {
		if p.inTag {
			if endPos := strings.Index(p.buf, tagClose); endPos >= 0 {
				p.tagBuf += p.buf[:endPos]
				p.buf = p.buf[endPos+len(tagClose):]
				if parsed, ok := fixJSON(strings.TrimSpace(p.tagBuf)); ok {
					name := jsonx.GetStr(parsed, "name")
					arguments := "{}"
					if a := jsonx.Get(parsed, "arguments"); a != nil {
						arguments = jsonx.MarshalString(a)
						if arguments == "" {
							arguments = "{}"
						}
					}
					id := EnsureToolCallID(jsonx.GetStr(parsed, "call_id"), p.toolIndex)
					p.toolIndex++
					events = append(events, StreamEvent{Kind: EventCall, Call: ToolCall{Name: name, Arguments: arguments, ID: id}})
				}
				p.tagBuf = ""
				p.inTag = false
				continue
			}
			// No closing tag yet: flush everything except a possible partial
			// closing tag at the end, respecting UTF-8 rune boundaries.
			if len(p.buf) > len(tagClose) {
				safe := len(p.buf) - len(tagClose)
				for safe > 0 && !utf8.RuneStart(p.buf[safe]) {
					safe--
				}
				if safe > 0 {
					p.tagBuf += p.buf[:safe]
					p.buf = p.buf[safe:]
				}
			}
			break
		}

		if startPos := strings.Index(p.buf, tagOpen); startPos >= 0 {
			before := p.buf[:startPos]
			if before != "" {
				events = append(events, StreamEvent{Kind: EventText, Text: before})
			}
			p.buf = p.buf[startPos+len(tagOpen):]
			p.inTag = true
			p.tagBuf = ""
			continue
		}

		// No opening tag: emit text but hold back a possible partial opening tag.
		safeLen := len(p.buf)
		for prefixLen := len(tagOpen) - 1; prefixLen >= 1; prefixLen-- {
			if strings.HasSuffix(p.buf, tagOpen[:prefixLen]) {
				safeLen = len(p.buf) - prefixLen
				break
			}
		}
		if safeLen > 0 {
			safeText := p.buf[:safeLen]
			p.buf = p.buf[safeLen:]
			if safeText != "" {
				events = append(events, StreamEvent{Kind: EventText, Text: safeText})
			}
		}
		break
	}
	return events
}

// Flush emits any buffered remainder as text (including an unterminated tag).
func (p *StreamParser) Flush() []StreamEvent {
	var events []StreamEvent
	if p.inTag {
		text := tagOpen + p.tagBuf + p.buf
		if text != "" {
			events = append(events, StreamEvent{Kind: EventText, Text: text})
		}
	} else if p.buf != "" {
		events = append(events, StreamEvent{Kind: EventText, Text: p.buf})
	}
	p.buf = ""
	p.tagBuf = ""
	return events
}

// HasSeenTools reports whether any complete tool call has been parsed.
func (p *StreamParser) HasSeenTools() bool {
	return p.toolIndex > 0
}
