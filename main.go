package main

import (
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// appVersion mirrors the Rust CARGO_PKG_VERSION used in the startup log line.
const appVersion = "0.1.0"

func main() {
	log.SetFlags(log.LstdFlags)

	cfg := parseConfig()
	log.Printf("INFO LMStudio Forward v%s", appVersion)

	pm := newProcessManager()
	if err := pm.Start(&cfg); err != nil {
		log.Printf("ERROR Failed to start: %v", err)
		os.Exit(1)
	}

	// Build the HTTP client: connect timeout 10s, no proxy, no overall
	// timeout (streaming responses must stay open). TCP nodelay is the Go
	// default, so nothing extra is needed there.
	client := &http.Client{
		Transport: &http.Transport{
			Proxy: nil,
			DialContext: (&net.Dialer{
				Timeout: 10 * time.Second,
			}).DialContext,
		},
	}

	var rag *RagClient
	if cfg.RagEnabled {
		rc := newRagClient(client, &cfg)
		if err := rc.EnsureCollection(); err != nil {
			log.Printf("ERROR RAG: failed to ensure Qdrant collection: %v", err)
			pm.Stop()
			os.Exit(1)
		}
		log.Printf("INFO RAG enabled: qdrant=%s collection=%s embed=%s dim=%d",
			cfg.QdrantURL, cfg.QdrantCollection, cfg.EmbedURL, cfg.EmbedDim)
		rag = rc
	}

	state := &AppState{
		Config:     cfg,
		HTTPClient: client,
		Rag:        rag,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", state.handleAnthropicMessages)
	mux.HandleFunc("POST /v1/message", state.handleAnthropicMessages)
	mux.HandleFunc("POST /anthropic", state.handleAnthropicMessages)
	mux.HandleFunc("POST /anthropic/v1/messages", state.handleAnthropicMessages)
	mux.HandleFunc("POST /rag/ingest", state.handleRagIngest)
	mux.HandleFunc("GET /v1/models", state.handleListModels)
	mux.HandleFunc("GET /v1/", state.handleOpenAI)
	mux.HandleFunc("POST /v1/", state.handleOpenAI)
	mux.HandleFunc("GET /health", state.handleHealth)

	addr := fmt.Sprintf("0.0.0.0:%d", cfg.ServerPort)
	log.Printf("INFO Listening on http://%s", addr)
	log.Printf("INFO Backend: http://127.0.0.1:%d", cfg.BackendPort)
	if !cfg.NoFrpc {
		log.Printf("INFO Public: https://opus.northsea.chat")
	}

	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	// Graceful shutdown on SIGINT/SIGTERM: stop child processes then exit.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		pm.Stop()
		_ = server.Close()
		os.Exit(0)
	}()

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("ERROR server error: %v", err)
		pm.Stop()
		os.Exit(1)
	}
}

// handleHealth probes the backend /health endpoint and reports overall status.
func (s *AppState) handleHealth(w http.ResponseWriter, r *http.Request) {
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.Config.BackendPort)
	backendOK := false
	resp, err := s.HTTPClient.Get(url)
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
	w.Write(toJSON(body))
}

// handleRagIngest ingests documents into the RAG store. Body shape:
// {"documents":[{"text":...,"source":...}, ...]}.
func (s *AppState) handleRagIngest(w http.ResponseWriter, r *http.Request) {
	if !checkAPIKey(r, s.Config.APIKey) {
		http.Error(w, "Invalid API key", http.StatusUnauthorized)
		return
	}

	if s.Rag == nil {
		http.Error(w, "RAG is not enabled", http.StatusServiceUnavailable)
		return
	}

	raw, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	parsed, err := parseJSON(raw)
	if err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	docsArr := getArr(parsed, "documents")
	docs := make([]IngestDoc, 0, len(docsArr))
	for _, d := range docsArr {
		docs = append(docs, IngestDoc{
			Text:   getStr(d, "text"),
			Source: getStr(d, "source"),
		})
	}

	n, err := s.Rag.Ingest(docs)
	if err != nil {
		log.Printf("ERROR RAG ingest failed: %v", err)
		http.Error(w, fmt.Sprintf("ingest failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(toJSON(map[string]any{"ingested_chunks": n}))
}
