// Package openai implements the OpenAI-compatible forwarding handler.
// All model names are accepted — requests go straight to the backend.
package openai

import (
	"bytes"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/language"
	"lmstudio-forward/internal/proxy"
)

// Handler serves the OpenAI-compatible endpoints, holding a reference to the
// shared application state.
type Handler struct {
	State *proxy.AppState
}

// NewHandler wires a Handler to the shared application state.
func NewHandler(state *proxy.AppState) *Handler {
	return &Handler{State: state}
}

// Forward forwards an OpenAI-compatible request to the local backend, rewriting
// the model name and injecting dynamic thinking/sampling controls for chat
// completions, then streams the backend response back transparently.
func (h *Handler) Forward(w http.ResponseWriter, r *http.Request) {
	s := h.State
	if !proxy.CheckAPIKey(r, s.Config.APIKey) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte("Invalid API key"))
		return
	}

	clientIP := proxy.GetClientIP(r)
	path := strings.TrimPrefix(r.URL.Path, "/v1/")
	method := r.Method

	log.Printf("INFO %s %s /v1/%s", clientIP, method, path)

	bodyBytes, err := proxy.ReadLimitedBody(w, r, s.Config.MaxRequestBodyBytes)
	if err != nil {
		log.Printf("ERROR Failed to read request body: %v", err)
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			w.Write([]byte("Request body too large"))
		} else {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("Failed to read request body"))
		}
		return
	}
	r.Body.Close()

	// Rewrite model name + inject dynamic thinking control.
	requestBody := bodyBytes
	if len(bodyBytes) > 0 {
		if parsed, perr := jsonx.Parse(bodyBytes); perr == nil {
			if data := jsonx.AsObj(parsed); data != nil {
				// Rewrite model name.
				if jsonx.Has(data, "model") {
					data["model"] = s.Config.BackendModel()
				}

				// Dynamic thinking control for chat completions.
				if strings.HasPrefix(path, "chat/completions") {
					// Truncate messages to fit ctx_size.
					if msgs := jsonx.GetArr(data, "messages"); msgs != nil {
						truncated := language.TruncateMessages(msgs, s.Config.CtxSize)
						data["messages"] = truncated

						needsThinking := language.EstimateComplexity(truncated)
						log.Printf("INFO thinking: needs=%v", needsThinking)

						if !needsThinking {
							// Inject chat_template_kwargs to disable thinking.
							data["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
						}
					}

					// Inject sampling defaults if not set by client.
					if !jsonx.Has(data, "repetition_penalty") && s.Config.RepetitionPenalty > 1.0 {
						data["repetition_penalty"] = s.Config.RepetitionPenalty
						data["repetition_context_size"] = s.Config.RepetitionContextSize
					}
				}

				requestBody = jsonx.Marshal(data)
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

// ListModels returns the advertised Claude model names.
func (h *Handler) ListModels(w http.ResponseWriter, r *http.Request) {
	s := h.State
	if !proxy.CheckAPIKey(r, s.Config.APIKey) {
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

	jsonx.WriteJSON(w, http.StatusOK, models)
}
