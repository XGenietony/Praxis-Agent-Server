// Package server wires the HTTP routes to their handlers and provides the
// cross-cutting health and RAG-ingest endpoints.
package server

import (
	"context"
	"errors"
	"fmt"
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
	ctx, cancel := context.WithTimeout(r.Context(), s.state.Config.BackendHealthTimeout())
	defer cancel()
	backendOK := probeHTTP(ctx, s.state.HTTPClient, url)

	status := "degraded"
	if backendOK {
		status = "ok"
	}

	body := map[string]any{
		"status":  status,
		"backend": backendOK,
	}
	jsonx.WriteJSON(w, http.StatusOK, body)
}

func probeHTTP(ctx context.Context, client *http.Client, url string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
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

	raw, err := proxy.ReadLimitedBody(w, r, s.state.Config.MaxRequestBodyBytes)
	if err != nil {
		if errors.As(err, new(*http.MaxBytesError)) {
			http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "failed to read body", http.StatusBadRequest)
		}
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

	n, err := s.state.Rag.Ingest(r.Context(), docs)
	if err != nil {
		log.Printf("ERROR RAG ingest failed: %v", err)
		http.Error(w, fmt.Sprintf("ingest failed: %v", err), http.StatusInternalServerError)
		return
	}

	jsonx.WriteJSON(w, http.StatusOK, map[string]any{"ingested_chunks": n})
}
