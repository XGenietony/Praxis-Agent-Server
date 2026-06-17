package anthropic

import (
	"bytes"
	"fmt"
	"log"
	"net/http"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/rag"
	"lmstudio-forward/internal/stream"
	"lmstudio-forward/internal/tools"
)

// extractToolCalls extracts (name, arguments_json_string) pairs from a
// transformed OpenAI response (choices[0].message.tool_calls).
func extractToolCalls(resp any) [][2]string {
	calls := jsonx.AsArr(jsonx.Pointer(resp, "choices", "0", "message", "tool_calls"))
	var out [][2]string
	for _, tc := range calls {
		f := jsonx.Get(tc, "function")
		if f == nil {
			continue
		}
		name, ok := jsonx.AsStr(jsonx.Get(f, "name"))
		if !ok {
			continue
		}
		args := "{}"
		if v, ok := jsonx.AsStr(jsonx.Get(f, "arguments")); ok {
			args = v
		}
		out = append(out, [2]string{name, args})
	}
	return out
}

// backendOnce sends body to the backend once (non-streaming) and returns the
// tool-call-parsed response bytes.
func (h *Handler) backendOnce(url string, body map[string]any) ([]byte, error) {
	s := h.State
	probe := make(map[string]any, len(body)+2)
	for k, v := range body {
		probe[k] = v
	}
	probe["stream"] = true
	probe["stream_options"] = map[string]any{"include_usage": true}

	req, err := http.NewRequest("POST", url, bytes.NewReader(jsonx.Marshal(probe)))
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
	collected := stream.CollectToResponse(resp.Body)
	return tools.TransformResponse(collected), nil
}

// resolveRagRounds runs internal retrieve rounds. Returns the augmented body
// (messages include all retrieved context) and the final collected response
// bytes from the round where the model chose to answer (or the forced answer
// after max rounds).
func (h *Handler) resolveRagRounds(url string, ragClient *rag.Client, body map[string]any, maxRounds int) (map[string]any, []byte, error) {
	rounds := maxRounds
	if rounds < 1 {
		rounds = 1
	}
	for round := 0; round < rounds; round++ {
		respBytes, err := h.backendOnce(url, body)
		if err != nil {
			return nil, nil, err
		}
		resp, err := jsonx.Parse(respBytes)
		if err != nil {
			return nil, nil, err
		}
		calls := extractToolCalls(resp)

		onlyRetrieve := len(calls) > 0
		for _, c := range calls {
			if c[0] != rag.RetrieveToolName {
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
			if v, err := jsonx.Parse([]byte(args)); err == nil {
				query = jsonx.Str(jsonx.Get(v, "query"))
			}
			if query == "" {
				continue
			}
			chunks, err := ragClient.Search(query)
			if err != nil {
				log.Printf("ERROR RAG search failed for query '%s': %v", query, err)
				chunks = nil
			}
			log.Printf("INFO RAG round %d: query='%s' hits=%d", round+1, query, len(chunks))
			result := rag.FormatChunks(chunks)
			if messages := jsonx.AsArr(body["messages"]); messages != nil {
				messages = append(messages, map[string]any{
					"role":    "assistant",
					"content": rag.RetrieveCallText(query),
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
	if messages := jsonx.AsArr(body["messages"]); messages != nil {
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": "You have gathered enough context. Answer the question now using the information above. Do not call the retrieve tool again.",
		})
		body["messages"] = messages
	}
	finalBytes, err := h.backendOnce(url, body)
	if err != nil {
		return nil, nil, err
	}
	return body, finalBytes, nil
}
