// Command lmstudio-forward is a lightweight proxy that exposes a local LLM
// backend as unified OpenAI- and Anthropic-compatible API endpoints, with
// optional public tunnel via frpc and optional Agentic RAG.
//
// This file is the application entrypoint: it parses configuration, wires up
// dependencies, and starts the HTTP server. All behavior lives in internal/.
package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"lmstudio-forward/internal/config"
	"lmstudio-forward/internal/process"
	"lmstudio-forward/internal/proxy"
	"lmstudio-forward/internal/rag"
	"lmstudio-forward/internal/server"
)

// appVersion mirrors the Rust CARGO_PKG_VERSION used in the startup log line.
const appVersion = "0.1.0"

func main() {
	log.SetFlags(log.LstdFlags)

	cfg := config.Parse()
	log.Printf("INFO LMStudio Forward v%s", appVersion)

	// Start backend + frpc child processes.
	pm := process.NewManager()
	if err := pm.Start(&cfg); err != nil {
		log.Printf("ERROR Failed to start: %v", err)
		os.Exit(1)
	}

	// Build the HTTP client: connect timeout 10s, no proxy, no overall timeout
	// (streaming responses must stay open). TCP nodelay is the Go default.
	client := &http.Client{
		Transport: &http.Transport{
			Proxy: nil,
			DialContext: (&net.Dialer{
				Timeout: 10 * time.Second,
			}).DialContext,
		},
	}

	// Optional Agentic RAG.
	var ragClient *rag.Client
	if cfg.RagEnabled {
		rc := rag.NewClient(client, &cfg)
		if err := rc.EnsureCollection(); err != nil {
			log.Printf("ERROR RAG: failed to ensure Qdrant collection: %v", err)
			pm.Stop()
			os.Exit(1)
		}
		log.Printf("INFO RAG enabled: qdrant=%s collection=%s embed=%s dim=%d",
			cfg.QdrantURL, cfg.QdrantCollection, cfg.EmbedURL, cfg.EmbedDim)
		ragClient = rc
	}

	// Assemble shared application state and the HTTP server.
	state := &proxy.AppState{
		Config:     cfg,
		HTTPClient: client,
		Rag:        ragClient,
	}
	srv := server.New(state)

	addr := fmt.Sprintf("0.0.0.0:%d", cfg.ServerPort)
	log.Printf("INFO Listening on http://%s", addr)
	log.Printf("INFO Backend: http://127.0.0.1:%d", cfg.BackendPort)
	if !cfg.NoFrpc {
		log.Printf("INFO Public: https://opus.northsea.chat")
	}

	httpServer := &http.Server{
		Addr:    addr,
		Handler: srv.Routes(),
	}

	// Graceful shutdown on SIGINT/SIGTERM: stop child processes then exit.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		pm.Stop()
		_ = httpServer.Close()
		os.Exit(0)
	}()

	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("ERROR server error: %v", err)
		pm.Stop()
		os.Exit(1)
	}
}
