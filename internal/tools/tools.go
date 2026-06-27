// Package tools handles tool-call adaptation: it injects tool definitions into
// the system prompt for models without native function calling, and parses
// `<tool_call>` tags out of model output (both batch and streaming).
//
// The package is split by responsibility:
//
//	tools.go         shared types, constants, helpers
//	convert.go       request-side conversion (Anthropic→OpenAI, tool injection)
//	response.go      batch response parsing (<tool_call> → native tool_calls)
//	streamparser.go  incremental streaming tool-call parser
//	jsonrepair.go    best-effort JSON repair for small-model output
package tools

import "strconv"

// ToolCall is a parsed tool invocation. ID == "" means no id was provided.
type ToolCall struct {
	Name      string
	Arguments string
	ID        string
}

// Stream event kinds emitted by the streaming tool-call parser.
const (
	EventText = iota
	EventCall
)

// StreamEvent is either a chunk of plain text or a completed tool call.
type StreamEvent struct {
	Kind int
	Text string
	Call ToolCall
}

const (
	tagOpen  = "<tool_call>"
	tagClose = "</tool_call>"
)

// orDefault returns s, or def when s is empty.
func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

// EnsureToolCallID preserves a model/provider-supplied ID and otherwise creates
// a deterministic local ID shared by batch and streaming tool parsing paths.
func EnsureToolCallID(id string, index int) string {
	if id != "" {
		return id
	}
	return "toolu_local_" + strconv.Itoa(index)
}
