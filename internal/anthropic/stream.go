package anthropic

import (
	"bufio"
	"bytes"
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
func (h *Handler) streamAnthropic(w http.ResponseWriter, url string, openaiBytes []byte, model, clientIP string, start time.Time) {
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
			chunkJSON, err := jsonx.Parse([]byte(payload))
			if err != nil {
				continue
			}
			for _, event := range openaiChunkToAnthropic(chunkJSON, state, model, toolParser) {
				w.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", jsonx.Str(jsonx.Get(event, "type")), jsonx.MarshalString(event))))
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
			w.Write([]byte("event: message_delta\ndata: " + jsonx.MarshalString(ev) + "\n\n"))
			flush()
			w.Write([]byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"))
			flush()
		}
	}

	log.Printf("INFO %s Anthropic stream completed in %v", clientIP, time.Since(start))
}
