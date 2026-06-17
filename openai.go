package main

// OpenAI-compatible forwarding handler.
// All model names are accepted — requests go straight to backend.

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// LocalModelAlias is the model alias for non-MLX backends
// (llama-server doesn't care about the model name).
const LocalModelAlias = "gemma4"

// handleOpenAI forwards an OpenAI-compatible request to the local backend,
// rewriting the model name and injecting dynamic thinking/sampling controls
// for chat completions, then streams the backend response back transparently.
func (s *AppState) handleOpenAI(w http.ResponseWriter, r *http.Request) {
	if !checkAPIKey(r, s.Config.APIKey) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte("Invalid API key"))
		return
	}

	clientIP := getClientIP(r)
	path := strings.TrimPrefix(r.URL.Path, "/v1/")
	method := r.Method

	log.Printf("INFO %s %s /v1/%s", clientIP, method, path)

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("ERROR Failed to read request body: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("Failed to read request body"))
		return
	}
	r.Body.Close()

	// Rewrite model name + inject dynamic thinking control.
	requestBody := bodyBytes
	if len(bodyBytes) > 0 {
		if parsed, perr := parseJSON(bodyBytes); perr == nil {
			if data := asObj(parsed); data != nil {
				// Rewrite model name.
				if has(data, "model") {
					data["model"] = s.Config.BackendModel()
				}

				// Dynamic thinking control for chat completions.
				if strings.HasPrefix(path, "chat/completions") {
					// Truncate messages to fit ctx_size.
					if msgs := getArr(data, "messages"); msgs != nil {
						truncated := truncateMessages(msgs, s.Config.CtxSize)
						data["messages"] = truncated

						needsThinking := estimateComplexity(truncated)
						log.Printf("INFO thinking: needs=%v", needsThinking)

						if !needsThinking {
							// Inject chat_template_kwargs to disable thinking.
							data["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
						}
					}

					// Inject sampling defaults if not set by client.
					if !has(data, "repetition_penalty") && s.Config.RepetitionPenalty > 1.0 {
						data["repetition_penalty"] = s.Config.RepetitionPenalty
						data["repetition_context_size"] = s.Config.RepetitionContextSize
					}
				}

				requestBody = toJSON(data)
			}
		}
	}

	// Forward to backend.
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/%s", s.Config.BackendPort, path)
	outReq, err := http.NewRequest(method, url, bytes.NewReader(requestBody))
	if err != nil {
		log.Printf("ERROR Failed to connect to backend: %v", err)
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte(fmt.Sprintf("Cannot connect to backend: %v", err)))
		return
	}
	for key, values := range r.Header {
		if key == "Host" || key == "Content-Length" {
			continue
		}
		for _, value := range values {
			outReq.Header.Add(key, value)
		}
	}

	resp, err := s.HTTPClient.Do(outReq)
	if err != nil {
		log.Printf("ERROR Failed to connect to backend: %v", err)
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte(fmt.Sprintf("Cannot connect to backend: %v", err)))
		return
	}
	defer resp.Body.Close()

	// Stream the response back transparently.
	for key, values := range resp.Header {
		if key == "Content-Length" || key == "Transfer-Encoding" {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(resp.StatusCode)

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			break
		}
	}
}

// handleListModels returns the advertised Claude model names.
func (s *AppState) handleListModels(w http.ResponseWriter, r *http.Request) {
	if !checkAPIKey(r, s.Config.APIKey) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte("Invalid API key"))
		return
	}

	models := map[string]any{
		"object": "list",
		"data": []any{
			map[string]any{"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
			map[string]any{"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
			map[string]any{"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
			map[string]any{"id": "claude-sonnet-4-5-20250514", "object": "model", "owned_by": "anthropic"},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(toJSON(models))
}
