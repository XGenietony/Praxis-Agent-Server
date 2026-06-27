package anthropic

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"time"

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

func (b backendCompleter) Complete(ctx context.Context, body map[string]any) ([]byte, error) {
	return b.handler.backendOnce(ctx, b.url, body)
}

// backendOnce sends body to the backend once (non-streaming) and returns the
// tool-call-parsed response bytes.
func (h *Handler) backendOnce(ctx context.Context, url string, body map[string]any) ([]byte, error) {
	s := h.State
	probe := make(map[string]any, len(body)+2)
	for k, v := range body {
		probe[k] = v
	}
	probe["stream"] = true
	probe["stream_options"] = map[string]any{"include_usage": true}

	probeBytes, err := jsonx.MarshalStrict(probe)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(probeBytes))
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
func (h *Handler) resolveRagRounds(ctx context.Context, url string, ragClient *rag.Client, body map[string]any, maxRounds int) (map[string]any, []byte, error) {
	runner := agentloop.Runner{
		Backend:     backendCompleter{handler: h, url: url},
		Retriever:   ragClient,
		MaxRounds:   maxRounds,
		StepTimeout: time.Duration(h.State.Config.RagStepTimeoutSeconds) * time.Second,
	}
	return runner.Run(ctx, body)
}
