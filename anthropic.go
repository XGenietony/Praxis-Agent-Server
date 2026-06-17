package main

// Anthropic Messages API compatibility layer.
// Converts Anthropic <-> OpenAI protocol and proxies to LM Studio.

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// anthropicError writes an Anthropic-shaped error response.
func anthropicError(w http.ResponseWriter, status int, errorType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(toJSON(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": errorType, "message": message},
	}))
}

// ─── Agentic RAG ──────────────────────────────────────────────────────────────

// extractToolCalls extracts (name, arguments_json_string) pairs from a
// transformed OpenAI response (choices[0].message.tool_calls).
func extractToolCalls(resp any) [][2]string {
	calls := asArr(pointer(resp, "choices", "0", "message", "tool_calls"))
	var out [][2]string
	for _, tc := range calls {
		f := get(tc, "function")
		if f == nil {
			continue
		}
		name, ok := asStr(get(f, "name"))
		if !ok {
			continue
		}
		args := "{}"
		if v, ok := asStr(get(f, "arguments")); ok {
			args = v
		}
		out = append(out, [2]string{name, args})
	}
	return out
}

// backendOnce sends body to the backend once (non-streaming) and returns the
// tool-call-parsed response bytes.
func (s *AppState) backendOnce(url string, body map[string]any) ([]byte, error) {
	probe := make(map[string]any, len(body)+2)
	for k, v := range body {
		probe[k] = v
	}
	probe["stream"] = true
	probe["stream_options"] = map[string]any{"include_usage": true}

	req, err := http.NewRequest("POST", url, bytes.NewReader(toJSON(probe)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := s.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("backend connect failed: %v", err)
	}
	defer resp.Body.Close()
	collected := collectStreamToResponse(resp.Body)
	return transformResponse(collected), nil
}

// resolveRagRounds runs internal retrieve rounds. Returns the augmented body
// (messages include all retrieved context) and the final collected response
// bytes from the round where the model chose to answer (or the forced answer
// after max rounds).
func (s *AppState) resolveRagRounds(url string, rag *RagClient, body map[string]any, maxRounds int) (map[string]any, []byte, error) {
	rounds := maxRounds
	if rounds < 1 {
		rounds = 1
	}
	for round := 0; round < rounds; round++ {
		respBytes, err := s.backendOnce(url, body)
		if err != nil {
			return nil, nil, err
		}
		resp, err := parseJSON(respBytes)
		if err != nil {
			return nil, nil, err
		}
		calls := extractToolCalls(resp)

		onlyRetrieve := len(calls) > 0
		for _, c := range calls {
			if c[0] != RetrieveToolName {
				onlyRetrieve = false
			}
		}
		if !onlyRetrieve {
			// Final answer (text or a client-handled tool call).
			return body, respBytes, nil
		}

		// Consume each retrieve call: search and append context to the conversation.
		for _, c := range calls {
			args := c[1]
			query := ""
			if v, err := parseJSON([]byte(args)); err == nil {
				query = str(get(v, "query"))
			}
			if query == "" {
				continue
			}
			chunks, err := rag.Search(query)
			if err != nil {
				log.Printf("ERROR RAG search failed for query '%s': %v", query, err)
				chunks = nil
			}
			log.Printf("INFO RAG round %d: query='%s' hits=%d", round+1, query, len(chunks))
			result := formatChunks(chunks)
			if messages := asArr(body["messages"]); messages != nil {
				messages = append(messages, map[string]any{
					"role":    "assistant",
					"content": retrieveCallText(query),
				})
				messages = append(messages, map[string]any{
					"role":    "user",
					"content": "[retrieve result]\n" + result,
				})
				body["messages"] = messages
			}
		}
	}

	// Max rounds hit: force a direct answer.
	if messages := asArr(body["messages"]); messages != nil {
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": "You have gathered enough context. Answer the question now using the information above. Do not call the retrieve tool again.",
		})
		body["messages"] = messages
	}
	finalBytes, err := s.backendOnce(url, body)
	if err != nil {
		return nil, nil, err
	}
	return body, finalBytes, nil
}

// ─── Types ──────────────────────────────────────────────────────────────────

type streamState struct {
	started           bool
	thinkingStarted   bool
	textStarted       bool
	blockIndex        int
	pendingStopReason string // "" = none
	finished          bool
}

// ─── Protocol conversion ────────────────────────────────────────────────────

// openaiToAnthropic converts an OpenAI response to Anthropic format (non-streaming).
func openaiToAnthropic(resp any, model string) any {
	choices := asArr(get(resp, "choices"))
	var choice any
	if len(choices) > 0 {
		choice = choices[0]
	}
	message := get(choice, "message")
	usage := get(resp, "usage")

	var contentBlocks []any

	reasoning, ok := asStr(get(message, "reasoning_content"))
	if !ok {
		reasoning, ok = asStr(get(message, "reasoning"))
	}
	if ok && reasoning != "" {
		contentBlocks = append(contentBlocks, map[string]any{"type": "thinking", "thinking": reasoning})
	}

	text := str(get(message, "content"))
	if text != "" {
		contentBlocks = append(contentBlocks, map[string]any{"type": "text", "text": text})
	}

	if toolCalls := asArr(get(message, "tool_calls")); toolCalls != nil {
		for i, tc := range toolCalls {
			if fn := get(tc, "function"); fn != nil {
				name := str(get(fn, "name"))
				argsStr := "{}"
				if v, ok := asStr(get(fn, "arguments")); ok {
					argsStr = v
				}
				input, err := parseJSON([]byte(argsStr))
				if err != nil {
					input = map[string]any{}
				}
				id := fmt.Sprintf("toolu_%04x", i)
				if v, ok := asStr(get(tc, "id")); ok {
					id = v
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type": "tool_use", "id": id, "name": name, "input": input,
				})
			}
		}
	}

	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, map[string]any{"type": "text", "text": ""})
	}

	stopReason := "end_turn"
	switch fr, _ := asStr(get(choice, "finish_reason")); fr {
	case "tool_calls":
		stopReason = "tool_use"
	case "length":
		stopReason = "max_tokens"
	}

	id := "unknown"
	if v, ok := asStr(get(resp, "id")); ok {
		id = v
	}
	promptTokens := get(usage, "prompt_tokens")
	if promptTokens == nil {
		promptTokens = float64(0)
	}
	completionTokens := get(usage, "completion_tokens")
	if completionTokens == nil {
		completionTokens = float64(0)
	}

	return map[string]any{
		"id":            "msg_" + id,
		"type":          "message",
		"role":          "assistant",
		"content":       contentBlocks,
		"model":         model,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage":         map[string]any{"input_tokens": promptTokens, "output_tokens": completionTokens},
	}
}

// openaiChunkToAnthropic converts a single OpenAI SSE chunk to Anthropic SSE events.
func openaiChunkToAnthropic(chunk any, state *streamState, model string, toolParser *ToolCallStreamParser) []any {
	var events []any

	if !state.started {
		state.started = true
		id := "unknown"
		if v, ok := asStr(get(chunk, "id")); ok {
			id = v
		}
		events = append(events, map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": "msg_" + id, "type": "message", "role": "assistant",
				"content": []any{}, "model": model,
				"stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
			},
		})
	}

	choices := asArr(get(chunk, "choices"))
	var choice any
	if len(choices) > 0 {
		choice = choices[0]
	}
	delta := get(choice, "delta")

	// Reasoning content → thinking block (support both "reasoning_content" and "reasoning")
	reasoning, ok := asStr(get(delta, "reasoning_content"))
	if !ok {
		reasoning, ok = asStr(get(delta, "reasoning"))
	}
	if ok && reasoning != "" {
		if !state.thinkingStarted {
			state.thinkingStarted = true
			events = append(events, map[string]any{
				"type": "content_block_start", "index": state.blockIndex,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
		}
		events = append(events, map[string]any{
			"type": "content_block_delta", "index": state.blockIndex,
			"delta": map[string]any{"type": "thinking_delta", "thinking": reasoning},
		})
	}

	// Text content → feed through tool parser
	if text, ok := asStr(get(delta, "content")); ok && text != "" {
		for _, te := range toolParser.Feed(text) {
			events = applyToolEvent(te, state, events)
		}
	}

	// Finish reason
	if finish, ok := asStr(get(choice, "finish_reason")); ok && finish != "" {
		// Flush tool parser
		for _, te := range toolParser.Flush() {
			events = applyToolEvent(te, state, events)
		}

		stopReason := "end_turn"
		if toolParser.HasSeenTools() || finish == "tool_calls" {
			stopReason = "tool_use"
		} else if finish == "length" {
			stopReason = "max_tokens"
		}

		if state.thinkingStarted || state.textStarted {
			events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
			state.thinkingStarted = false
			state.textStarted = false
		}

		if usage := get(chunk, "usage"); usage != nil {
			ct := get(usage, "completion_tokens")
			if ct == nil {
				ct = float64(0)
			}
			events = append(events, map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
				"usage": map[string]any{"output_tokens": ct},
			})
			events = append(events, map[string]any{"type": "message_stop"})
			state.finished = true
		} else {
			state.pendingStopReason = stopReason
		}
	}

	// Deferred finish: usage-only chunk
	if state.pendingStopReason != "" {
		stopReason := state.pendingStopReason
		state.pendingStopReason = ""
		if usage := get(chunk, "usage"); usage != nil {
			ct := get(usage, "completion_tokens")
			if ct == nil {
				ct = float64(0)
			}
			events = append(events, map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
				"usage": map[string]any{"output_tokens": ct},
			})
			events = append(events, map[string]any{"type": "message_stop"})
			state.finished = true
		} else {
			state.pendingStopReason = stopReason
		}
	}

	return events
}

// applyToolEvent appends the Anthropic events produced by a single tool stream
// event, mutating state as needed.
func applyToolEvent(te ToolStreamEvent, state *streamState, events []any) []any {
	switch te.Kind {
	case ToolEventText:
		if te.Text != "" {
			events = ensureTextBlock(state, events)
			events = append(events, map[string]any{
				"type": "content_block_delta", "index": state.blockIndex,
				"delta": map[string]any{"type": "text_delta", "text": te.Text},
			})
		}
	case ToolEventCall:
		tc := te.Call
		if state.textStarted {
			events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
			state.blockIndex++
			state.textStarted = false
		}
		input, err := parseJSON([]byte(tc.Arguments))
		if err != nil {
			input = map[string]any{}
		}
		id := tc.ID
		if id == "" {
			id = fmt.Sprintf("toolu_%04x", state.blockIndex)
		}
		events = append(events, map[string]any{
			"type": "content_block_start", "index": state.blockIndex,
			"content_block": map[string]any{"type": "tool_use", "id": id, "name": tc.Name, "input": map[string]any{}},
		})
		events = append(events, map[string]any{
			"type": "content_block_delta", "index": state.blockIndex,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": toJSONString(input)},
		})
		events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
		state.blockIndex++
	}
	return events
}

func ensureTextBlock(state *streamState, events []any) []any {
	if !state.textStarted {
		if state.thinkingStarted {
			events = append(events, map[string]any{"type": "content_block_stop", "index": state.blockIndex})
			state.blockIndex++
			state.thinkingStarted = false
		}
		state.textStarted = true
		events = append(events, map[string]any{
			"type": "content_block_start", "index": state.blockIndex,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
	}
	return events
}

// ─── Handler ────────────────────────────────────────────────────────────────

func (s *AppState) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if !checkAPIKey(r, s.Config.APIKey) {
		anthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}

	clientIP := getClientIP(r)
	start := time.Now()

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("ERROR Failed to read request body: %v", err)
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	anthropicReq, err := parseJSON(bodyBytes)
	if err != nil {
		log.Printf("ERROR Invalid JSON: %v", err)
		anthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	model := str(get(anthropicReq, "model"))
	isStream := boolv(get(anthropicReq, "stream"))

	log.Printf("INFO %s POST /v1/messages model=%s stream=%v", clientIP, model, isStream)

	// Convert Anthropic → OpenAI
	openaiBody := anthropicRequestToOpenAI(anthropicReq)

	// Agentic RAG: inject the built-in `retrieve` tool so the model can request retrieval.
	if s.Rag != nil {
		retrieve := retrieveToolOpenAI()
		if tools := asArr(openaiBody["tools"]); tools != nil {
			openaiBody["tools"] = append(tools, retrieve)
		} else {
			openaiBody["tools"] = []any{retrieve}
		}
	}

	// Tool adaptation for local model
	hasTools := transformRequest(openaiBody)
	if hasTools {
		log.Printf("INFO Anthropic tool adaptation applied")
	}

	// Rewrite model
	openaiBody["model"] = s.Config.BackendModel()

	// Truncate messages to fit ctx_size
	if messages := getArr(openaiBody, "messages"); messages != nil {
		openaiBody["messages"] = truncateMessages(messages, s.Config.CtxSize)
	}

	// Dynamic thinking control
	if messages := getArr(openaiBody, "messages"); messages != nil {
		needsThinking := estimateComplexity(messages)
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
		augmentedBody, finalBytes, err := s.resolveRagRounds(url, s.Rag, openaiBody, s.Config.RagMaxRounds)
		if err != nil {
			log.Printf("ERROR RAG loop failed: %v", err)
			anthropicError(w, http.StatusBadGateway, "api_error", "RAG loop failed: "+err.Error())
			return
		}
		openaiBody = augmentedBody
		if !isStream {
			// Reuse the already-collected final answer; no extra generation.
			respBody, err := parseJSON(finalBytes)
			if err != nil {
				log.Printf("ERROR Invalid response: %v", err)
				anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
				return
			}
			log.Printf("INFO %s Anthropic non-stream (RAG) completed in %v", clientIP, time.Since(start))
			w.Header().Set("Content-Type", "application/json")
			w.Write(toJSON(openaiToAnthropic(respBody, model)))
			return
		}
		// Stream branch falls through with the context-augmented body.
	}

	if isStream {
		openaiBody["stream"] = true
		openaiBody["stream_options"] = map[string]any{"include_usage": true}
		s.streamAnthropic(w, url, toJSON(openaiBody), model, clientIP, start)
		return
	}

	// Non-stream: send to backend, convert response
	openaiBody["stream"] = true
	openaiBody["stream_options"] = map[string]any{"include_usage": true}

	req, err := http.NewRequest("POST", url, bytes.NewReader(toJSON(openaiBody)))
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

	respBytes := collectStreamToResponse(resp.Body)

	// Apply tool call parsing
	if hasTools {
		respBytes = transformResponse(respBytes)
	}

	respBody, err := parseJSON(respBytes)
	if err != nil {
		log.Printf("ERROR Invalid response: %v", err)
		anthropicError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}

	log.Printf("INFO %s Anthropic non-stream completed in %v", clientIP, time.Since(start))
	w.Header().Set("Content-Type", "application/json")
	w.Write(toJSON(openaiToAnthropic(respBody, model)))
}

// ─── Streaming ──────────────────────────────────────────────────────────────

func (s *AppState) streamAnthropic(w http.ResponseWriter, url string, openaiBytes []byte, model, clientIP string, start time.Time) {
	sseHeaders(w)
	flusher, _ := w.(http.Flusher)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	ping := []byte("event: ping\ndata: {\"type\": \"ping\"}\n\n")

	state := &streamState{}
	toolParser := newToolCallStreamParser()

	w.Write(ping)
	flush()

	// Connect with ping keepalives during prompt processing.
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	type respResult struct {
		resp *http.Response
		err  error
	}
	respCh := make(chan respResult, 1)
	go func() {
		req, err := http.NewRequest("POST", url, bytes.NewReader(openaiBytes))
		if err != nil {
			respCh <- respResult{nil, err}
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		resp, err := s.HTTPClient.Do(req)
		respCh <- respResult{resp, err}
	}()

	var resp *http.Response
connectLoop:
	for {
		select {
		case rr := <-respCh:
			if rr.err != nil {
				log.Printf("ERROR Failed to connect to LM Studio: %v", rr.err)
				w.Write([]byte(fmt.Sprintf(
					"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Backend error: %v\"}}\n\n", rr.err)))
				flush()
				return
			}
			resp = rr.resp
			break connectLoop
		case <-ticker.C:
			w.Write(ping)
			flush()
		}
	}
	defer resp.Body.Close()

	// Read the backend body line-by-line in a goroutine; the main loop selects
	// over incoming lines and the ping ticker.
	lineCh := make(chan string)
	go func() {
		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			if line != "" {
				lineCh <- line
			}
			if err != nil {
				if err != io.EOF {
					log.Printf("ERROR Stream error: %v", err)
				}
				close(lineCh)
				return
			}
		}
	}()

streamLoop:
	for {
		select {
		case line, ok := <-lineCh:
			if !ok {
				break streamLoop
			}
			trimmed := strings.TrimSpace(line)
			if !strings.HasPrefix(trimmed, "data: ") {
				continue
			}
			payload := trimmed[6:]
			if payload == "[DONE]" {
				continue
			}
			chunkJSON, err := parseJSON([]byte(payload))
			if err != nil {
				continue
			}
			for _, event := range openaiChunkToAnthropic(chunkJSON, state, model, toolParser) {
				w.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", str(get(event, "type")), toJSONString(event))))
				flush()
			}
		case <-ticker.C:
			w.Write(ping)
			flush()
		}
	}

	// Emit deferred stop if stream ended without usage chunk.
	if !state.finished {
		if state.pendingStopReason != "" {
			stopReason := state.pendingStopReason
			state.pendingStopReason = ""
			ev := map[string]any{
				"type":  "message_delta",
				"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
				"usage": map[string]any{"output_tokens": 0},
			}
			w.Write([]byte("event: message_delta\ndata: " + toJSONString(ev) + "\n\n"))
			flush()
			w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
			flush()
		}
	}

	log.Printf("INFO %s Anthropic stream completed in %v", clientIP, time.Since(start))
}
