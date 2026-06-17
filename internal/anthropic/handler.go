package anthropic

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/language"
	"lmstudio-forward/internal/proxy"
	"lmstudio-forward/internal/rag"
	"lmstudio-forward/internal/stream"
	"lmstudio-forward/internal/tools"
)

// Messages handles POST requests to the Anthropic Messages endpoints.
func (h *Handler) Messages(w http.ResponseWriter, r *http.Request) {
	s := h.State
	if !proxy.CheckAPIKey(r, s.Config.APIKey) {
		anthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}

	clientIP := proxy.GetClientIP(r)
	start := time.Now()

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("ERROR Failed to read request body: %v", err)
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	anthropicReq, err := jsonx.Parse(bodyBytes)
	if err != nil {
		log.Printf("ERROR Invalid JSON: %v", err)
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	model := jsonx.Str(jsonx.Get(anthropicReq, "model"))
	isStream := jsonx.Bool(jsonx.Get(anthropicReq, "stream"))

	log.Printf("INFO %s POST /v1/messages model=%s stream=%v", clientIP, model, isStream)

	// Convert Anthropic → OpenAI
	openaiBody := tools.AnthropicRequestToOpenAI(anthropicReq)

	// Agentic RAG: inject the built-in `retrieve` tool so the model can request retrieval.
	if s.Rag != nil {
		retrieve := rag.RetrieveToolOpenAI()
		if toollist := jsonx.AsArr(openaiBody["tools"]); toollist != nil {
			openaiBody["tools"] = append(toollist, retrieve)
		} else {
			openaiBody["tools"] = []any{retrieve}
		}
	}

	// Tool adaptation for local model
	hasTools := tools.TransformRequest(openaiBody)
	if hasTools {
		log.Printf("INFO Anthropic tool adaptation applied")
	}

	// Rewrite model
	openaiBody["model"] = s.Config.BackendModel()

	// Truncate messages to fit ctx_size
	if messages := jsonx.GetArr(openaiBody, "messages"); messages != nil {
		openaiBody["messages"] = language.TruncateMessages(messages, s.Config.CtxSize)
	}

	// Dynamic thinking control
	if messages := jsonx.GetArr(openaiBody, "messages"); messages != nil {
		needsThinking := language.EstimateComplexity(messages)
		log.Printf("INFO anthropic thinking: needs=%v", needsThinking)
		if !needsThinking {
			openaiBody["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
		}
	}

	// Inject sampling defaults if not set
	if _, ok := openaiBody["repetition_penalty"]; !ok && s.Config.RepetitionPenalty > 1.0 {
		openaiBody["repetition_penalty"] = s.Config.RepetitionPenalty
		openaiBody["repetition_context_size"] = s.Config.RepetitionContextSize
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.Config.BackendPort)

	// Agentic RAG: consume internal `retrieve` rounds before responding to the client.
	if s.Rag != nil {
		augmentedBody, finalBytes, err := h.resolveRagRounds(url, s.Rag, openaiBody, s.Config.RagMaxRounds)
		if err != nil {
			log.Printf("ERROR RAG loop failed: %v", err)
			anthropicError(w, http.StatusBadGateway, "api_error", "RAG loop failed: "+err.Error())
			return
		}
		openaiBody = augmentedBody
		if !isStream {
			// Reuse the already-collected final answer; no extra generation.
			respBody, err := jsonx.Parse(finalBytes)
			if err != nil {
				log.Printf("ERROR Invalid response: %v", err)
				anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
				return
			}
			log.Printf("INFO %s Anthropic non-stream (RAG) completed in %v", clientIP, time.Since(start))
			w.Header().Set("Content-Type", "application/json")
			w.Write(jsonx.Marshal(openaiToAnthropic(respBody, model)))
			return
		}
		// Stream branch falls through with the context-augmented body.
	}

	if isStream {
		openaiBody["stream"] = true
		openaiBody["stream_options"] = map[string]any{"include_usage": true}
		h.streamAnthropic(w, url, jsonx.Marshal(openaiBody), model, clientIP, start)
		return
	}

	// Non-stream: send to backend, convert response
	openaiBody["stream"] = true
	openaiBody["stream_options"] = map[string]any{"include_usage": true}

	req, err := http.NewRequest("POST", url, bytes.NewReader(jsonx.Marshal(openaiBody)))
	if err != nil {
		log.Printf("ERROR Cannot connect to LM Studio: %v", err)
		anthropicError(w, http.StatusBadGateway, "api_error", "Cannot connect to backend: "+err.Error())
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := s.HTTPClient.Do(req)
	if err != nil {
		log.Printf("ERROR Cannot connect to LM Studio: %v", err)
		anthropicError(w, http.StatusBadGateway, "api_error", "Cannot connect to backend: "+err.Error())
		return
	}
	defer resp.Body.Close()

	respBytes := stream.CollectToResponse(resp.Body)

	// Apply tool call parsing
	if hasTools {
		respBytes = tools.TransformResponse(respBytes)
	}

	respBody, err := jsonx.Parse(respBytes)
	if err != nil {
		log.Printf("ERROR Invalid response: %v", err)
		anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}

	log.Printf("INFO %s Anthropic non-stream completed in %v", clientIP, time.Since(start))
	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonx.Marshal(openaiToAnthropic(respBody, model)))
}
