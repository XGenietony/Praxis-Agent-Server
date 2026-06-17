package main

import (
	"bufio"
	"io"
	"log"
	"net/http"
	"strings"
)

// collectStreamToResponse consumes an SSE stream and reconstructs a complete
// OpenAI chat.completion JSON, returning the marshaled bytes.
func collectStreamToResponse(body io.Reader) []byte {
	var content strings.Builder
	var reasoning strings.Builder
	id := ""
	modelStr := ""
	var finishReason any // nil == serde_json Value::Null
	var usage any        // nil == serde_json Value::Null

	done := false
	reader := bufio.NewReader(body)
outer:
	for {
		raw, err := reader.ReadString('\n')
		if raw != "" {
			line := strings.TrimSpace(raw)
			if strings.HasPrefix(line, "data: ") {
				payload := line[6:]
				if payload == "[DONE]" {
					done = true
					break outer
				}

				if v, perr := parseJSON([]byte(payload)); perr == nil {
					if id == "" {
						if i, ok := asStr(get(v, "id")); ok {
							id = i
						}
					}
					if modelStr == "" {
						if m, ok := asStr(get(v, "model")); ok {
							modelStr = m
						}
					}
					if delta := pointer(v, "choices", "0", "delta"); delta != nil {
						if c, ok := asStr(get(delta, "content")); ok {
							content.WriteString(c)
						}
						if r, ok := asStr(get(delta, "reasoning_content")); ok {
							reasoning.WriteString(r)
						}
					}
					if fr := pointer(v, "choices", "0", "finish_reason"); fr != nil {
						finishReason = fr
					}
					if u := get(v, "usage"); u != nil {
						usage = u
					}
				}
			}
		}

		if err != nil {
			if err != io.EOF {
				log.Printf("ERROR Stream read error: %v", err)
			}
			break
		}
	}

	if !done {
		log.Printf("ERROR Stream ended without [DONE] marker")
	}

	message := map[string]any{
		"role":    "assistant",
		"content": content.String(),
	}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}

	respJSON := map[string]any{
		"id":     id,
		"object": "chat.completion",
		"model":  modelStr,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
		"usage": usage,
	}

	return toJSON(respJSON)
}

// sseHeaders sets the standard SSE streaming response headers.
func sseHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-store, must-revalidate")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	h.Set("X-Content-Type-Options", "nosniff")
}
