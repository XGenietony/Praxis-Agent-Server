package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/stream"
	"lmstudio-forward/internal/tools"
)

// streamAnthropic connects to the backend and pumps converted Anthropic SSE
// events to the client, emitting ping keepalives during connect and streaming.
func (h *Handler) streamAnthropic(ctx context.Context, w http.ResponseWriter, url string, openaiBytes []byte, model, clientIP string, start time.Time) {
	s := h.State
	stream.SetHeaders(w)
	flusher, _ := w.(http.Flusher)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	ping := []byte("event: ping\ndata: {\"type\": \"ping\"}\n\n")

	state := &streamState{}
	toolParser := tools.NewStreamParser()

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
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(openaiBytes))
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
		case <-ctx.Done():
			log.Printf("INFO %s Anthropic stream canceled before backend connect: %v", clientIP, ctx.Err())
			return
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
				select {
				case lineCh <- line:
				case <-ctx.Done():
					close(lineCh)
					return
				}
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
			chunkJSON, err := jsonx.Parse([]byte(payload))
			if err != nil {
				continue
			}
			for _, event := range openaiChunkToAnthropic(chunkJSON, state, model, toolParser) {
				w.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", jsonx.Str(jsonx.Get(event, "type")), jsonx.MarshalString(event))))
				flush()
			}
		case <-ctx.Done():
			log.Printf("INFO %s Anthropic stream canceled: %v", clientIP, ctx.Err())
			return
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
			w.Write([]byte("event: message_delta\ndata: " + jsonx.MarshalString(ev) + "\n\n"))
			flush()
			w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
			flush()
		}
	}

	log.Printf("INFO %s Anthropic stream completed in %v", clientIP, time.Since(start))
}

// streamCollectedAnthropic emits an already-collected OpenAI completion as
// Anthropic SSE events. RAG uses this to avoid generating a second answer for
// streaming clients after internal retrieve rounds have already produced the
// final response.
func (h *Handler) streamCollectedAnthropic(ctx context.Context, w http.ResponseWriter, openaiBytes []byte, model, clientIP string, start time.Time) error {
	respBody, err := jsonx.Parse(openaiBytes)
	if err != nil {
		return err
	}
	msg := openaiToAnthropic(respBody, model)

	stream.SetHeaders(w)
	flusher, _ := w.(http.Flusher)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeEvent := func(event any) {
		if ctx.Err() != nil {
			return
		}
		w.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", jsonx.GetStr(event, "type"), jsonx.MarshalString(event))))
		flush()
	}

	inputTokens := jsonx.Get(jsonx.Get(msg, "usage"), "input_tokens")
	if inputTokens == nil {
		inputTokens = float64(0)
	}
	writeEvent(map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": jsonx.GetStr(msg, "id"), "type": "message", "role": "assistant",
			"content": []any{}, "model": model,
			"stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 0},
		},
	})

	for i, block := range jsonx.GetArr(msg, "content") {
		btype := jsonx.GetStr(block, "type")
		switch btype {
		case "thinking":
			writeEvent(map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "thinking", "thinking": ""},
			})
			writeEvent(map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "thinking_delta", "thinking": jsonx.GetStr(block, "thinking")},
			})
			writeEvent(map[string]any{"type": "content_block_stop", "index": i})
		case "tool_use":
			input := jsonx.Get(block, "input")
			if input == nil {
				input = map[string]any{}
			}
			writeEvent(map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{
					"type": "tool_use", "id": jsonx.GetStr(block, "id"),
					"name": jsonx.GetStr(block, "name"), "input": map[string]any{},
				},
			})
			writeEvent(map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": jsonx.MarshalString(input)},
			})
			writeEvent(map[string]any{"type": "content_block_stop", "index": i})
		default:
			writeEvent(map[string]any{
				"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			writeEvent(map[string]any{
				"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "text_delta", "text": jsonx.GetStr(block, "text")},
			})
			writeEvent(map[string]any{"type": "content_block_stop", "index": i})
		}
	}

	outputTokens := jsonx.Get(jsonx.Get(msg, "usage"), "output_tokens")
	if outputTokens == nil {
		outputTokens = float64(0)
	}
	writeEvent(map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason": jsonx.GetStr(msg, "stop_reason"), "stop_sequence": nil,
		},
		"usage": map[string]any{"output_tokens": outputTokens},
	})
	writeEvent(map[string]any{"type": "message_stop"})

	log.Printf("INFO %s Anthropic stream (collected RAG) completed in %v", clientIP, time.Since(start))
	return nil
}
