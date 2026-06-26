// Package anthropic implements the Anthropic Messages API compatibility layer.
// It converts between the Anthropic and OpenAI protocols and proxies to the
// local backend, including the Agentic RAG retrieve loop and SSE streaming.
//
// The package is split by responsibility:
//
//	anthropic.go  Handler type, error helper, shared stream state
//	handler.go    the Messages HTTP handler (request orchestration)
//	rag.go        the Agentic RAG adapter (resolveRagRounds, backendOnce)
//	convert.go    OpenAI<->Anthropic protocol conversion (batch + chunk)
//	stream.go     the SSE streaming pump
package anthropic

import (
	"net/http"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/proxy"
)

// Handler serves the Anthropic-compatible endpoints, holding a reference to the
// shared application state.
type Handler struct {
	State *proxy.AppState
}

// NewHandler wires a Handler to the shared application state.
func NewHandler(state *proxy.AppState) *Handler {
	return &Handler{State: state}
}

// anthropicError writes an Anthropic-shaped error response.
func anthropicError(w http.ResponseWriter, status int, errorType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(jsonx.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": errorType, "message": message},
	}))
}

// streamState tracks the in-flight Anthropic content-block stream as OpenAI
// chunks are converted to Anthropic SSE events.
type streamState struct {
	started           bool
	thinkingStarted   bool
	textStarted       bool
	blockIndex        int
	toolOrdinal       int
	nativeTools       map[int]*nativeToolDelta
	pendingStopReason string // "" = none
	finished          bool
}

type nativeToolDelta struct {
	id        string
	name      string
	arguments string
}
