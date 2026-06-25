package anthropic

import (
	"bytes"
	"fmt"
	"net/http"

	"lmstudio-forward/internal/agentloop"
	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/rag"
	"lmstudio-forward/internal/stream"
	"lmstudio-forward/internal/tools"
)

type backendCompleter struct {
	handler *Handler
	url     string
}

func (b backendCompleter) Complete(body map[string]any) ([]byte, error) {
	return b.handler.backendOnce(b.url, body)
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

// resolveRagRounds delegates the internal retrieve loop to agentloop while this
// package keeps the Anthropic/OpenAI protocol and backend HTTP details.
func (h *Handler) resolveRagRounds(url string, ragClient *rag.Client, body map[string]any, maxRounds int) (map[string]any, []byte, error) {
	runner := agentloop.Runner{
		Backend:   backendCompleter{handler: h, url: url},
		Retriever: ragClient,
		MaxRounds: maxRounds,
	}
	return runner.Run(body)
}
