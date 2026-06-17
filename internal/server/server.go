// Package server wires the HTTP routes to their handlers and provides the
// cross-cutting health and RAG-ingest endpoints.
package server

import (
	"fmt"
	"io"
	"log"
	"net/http"

	"lmstudio-forward/internal/anthropic"
	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/openai"
	"lmstudio-forward/internal/proxy"
	"lmstudio-forward/internal/rag"
)

// Server bundles the application state and its handler set.
type Server struct {
	state     *proxy.AppState
	openai    *openai.Handler
	anthropic *anthropic.Handler
}

// New constructs a Server and its handlers from shared application state.
func New(state *proxy.AppState) *Server {
	return &Server{
		state:     state,
		openai:    openai.NewHandler(state),
		anthropic: anthropic.NewHandler(state),
	}
}

// Routes builds the request multiplexer. Specific routes take precedence over
// the /v1/ catch-all under Go 1.22+ ServeMux precedence rules.
func (s *Server) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", s.anthropic.Messages)
	mux.HandleFunc("POST /v1/message", s.anthropic.Messages)
	mux.HandleFunc("POST /anthropic", s.anthropic.Messages)
	mux.HandleFunc("POST /anthropic/v1/messages", s.anthropic.Messages)
	mux.HandleFunc("POST /rag/ingest", s.handleRagIngest)
	mux.HandleFunc("GET /v1/models", s.openai.ListModels)
	mux.HandleFunc("GET /v1/", s.openai.Forward)
	mux.HandleFunc("POST /v1/", s.openai.Forward)
	mux.HandleFunc("GET /health", s.handleHealth)
	return mux
}

// handleHealth probes the backend /health endpoint and reports overall status.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.state.Config.BackendPort)
	backendOK := false
	resp, err := s.state.HTTPClient.Get(url)
	if err == nil {
		backendOK = true
		resp.Body.Close()
	}

	status := "degraded"
	if backendOK {
		status = "ok"
	}

	body := map[string]any{
		"status":  status,
		"backend": backendOK,
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonx.Marshal(body))
}

// handleRagIngest ingests documents into the RAG store. Body shape:
// {"documents":[{"text":...,"source":...}, ...]}.
func (s *Server) handleRagIngest(w http.ResponseWriter, r *http.Request) {
	if !proxy.CheckAPIKey(r, s.state.Config.APIKey) {
		http.Error(w, "Invalid API key", http.StatusUnauthorized)
		return
	}

	if s.state.Rag == nil {
		http.Error(w, "RAG is not enabled", http.StatusServiceUnavailable)
		return
	}

	raw, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	parsed, err := jsonx.Parse(raw)
	if err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	docsArr := jsonx.GetArr(parsed, "documents")
	docs := make([]rag.IngestDoc, 0, len(docsArr))
	for _, d := range docsArr {
		docs = append(docs, rag.IngestDoc{
			Text:   jsonx.GetStr(d, "text"),
			Source: jsonx.GetStr(d, "source"),
		})
	}

	n, err := s.state.Rag.Ingest(docs)
	if err != nil {
		log.Printf("ERROR RAG ingest failed: %v", err)
		http.Error(w, fmt.Sprintf("ingest failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonx.Marshal(map[string]any{"ingested_chunks": n}))
}
