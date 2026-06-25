package agentloop

import (
	"context"
	"errors"
	"strings"
	"testing"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/rag"
)

type fakeBackend struct {
	responses [][]byte
	calls     int
}

func (b *fakeBackend) Complete(ctx context.Context, body map[string]any) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b.calls >= len(b.responses) {
		return nil, errors.New("unexpected backend call")
	}
	resp := b.responses[b.calls]
	b.calls++
	return resp, nil
}

type fakeRetriever struct {
	queries []string
	err     error
	chunks  []rag.RetrievedChunk
}

func (r *fakeRetriever) Search(ctx context.Context, query string) ([]rag.RetrievedChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.queries = append(r.queries, query)
	if r.err != nil {
		return nil, r.err
	}
	if r.chunks != nil {
		return r.chunks, nil
	}
	return []rag.RetrievedChunk{{Text: "doc for " + query, Source: "test", Score: 0.9}}, nil
}

func baseBody() map[string]any {
	return map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "question"}},
	}
}

func openAIText(text string) []byte {
	return jsonx.Marshal(map[string]any{
		"id":     "chatcmpl-test",
		"object": "chat.completion",
		"model":  "test",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": text,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
	})
}

func openAITools(specs ...struct {
	name string
	args string
}) []byte {
	calls := make([]any, 0, len(specs))
	for i, spec := range specs {
		calls = append(calls, map[string]any{
			"id":   "call_" + spec.name,
			"type": "function",
			"function": map[string]any{
				"name":      spec.name,
				"arguments": spec.args,
			},
			"index": i,
		})
	}
	return jsonx.Marshal(map[string]any{
		"id":     "chatcmpl-tools",
		"object": "chat.completion",
		"model":  "test",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":       "assistant",
					"content":    nil,
					"tool_calls": calls,
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
	})
}

func retrieveCall(query string) struct {
	name string
	args string
} {
	return struct {
		name string
		args string
	}{rag.RetrieveToolName, `{"query":"` + query + `"}`}
}

func externalCall(name string) struct {
	name string
	args string
} {
	return struct {
		name string
		args string
	}{name, `{"x":1}`}
}

func TestRunNoRetrieveReturnsFinalResponse(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{openAIText("done")}}
	retriever := &fakeRetriever{}
	_, finalBytes, err := Runner{Backend: backend, Retriever: retriever, MaxRounds: 3}.Run(context.Background(), baseBody())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if backend.calls != 1 {
		t.Fatalf("backend calls: want 1, got %d", backend.calls)
	}
	if got := jsonx.GetStr(jsonx.Pointer(mustParse(t, finalBytes), "choices", "0", "message"), "content"); got != "done" {
		t.Fatalf("final content: %q", got)
	}
	if len(retriever.queries) != 0 {
		t.Fatalf("retriever should not be called: %v", retriever.queries)
	}
}

func TestRunSingleRetrieveThenAnswer(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{openAITools(retrieveCall("alpha")), openAIText("answer")}}
	retriever := &fakeRetriever{}
	body, _, err := Runner{Backend: backend, Retriever: retriever, MaxRounds: 3}.Run(context.Background(), baseBody())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if backend.calls != 2 {
		t.Fatalf("backend calls: want 2, got %d", backend.calls)
	}
	if len(retriever.queries) != 1 || retriever.queries[0] != "alpha" {
		t.Fatalf("queries: %v", retriever.queries)
	}
	if got := len(jsonx.AsArr(body["messages"])); got != 3 {
		t.Fatalf("messages after observation: want 3, got %d", got)
	}
}

func TestRunMultipleRetrieveRounds(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{
		openAITools(retrieveCall("alpha")),
		openAITools(retrieveCall("beta")),
		openAIText("answer"),
	}}
	retriever := &fakeRetriever{}
	_, _, err := Runner{Backend: backend, Retriever: retriever, MaxRounds: 3}.Run(context.Background(), baseBody())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if backend.calls != 3 {
		t.Fatalf("backend calls: want 3, got %d", backend.calls)
	}
	if got := strings.Join(retriever.queries, ","); got != "alpha,beta" {
		t.Fatalf("queries: %s", got)
	}
}

func TestRunMixedRetrieveAndExternalReplans(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{
		openAITools(retrieveCall("alpha"), externalCall("write_file")),
		openAITools(externalCall("write_file")),
	}}
	retriever := &fakeRetriever{}
	_, finalBytes, err := Runner{Backend: backend, Retriever: retriever, MaxRounds: 3}.Run(context.Background(), baseBody())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if backend.calls != 2 {
		t.Fatalf("backend calls: want 2, got %d", backend.calls)
	}
	calls := jsonx.AsArr(jsonx.Pointer(mustParse(t, finalBytes), "choices", "0", "message", "tool_calls"))
	if len(calls) != 1 || jsonx.GetStr(jsonx.Get(calls[0], "function"), "name") != "write_file" {
		t.Fatalf("final tool calls: %v", calls)
	}
}

func TestRunExhaustedOnlyRetrieveReturnsError(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{
		openAITools(retrieveCall("alpha")),
		openAITools(retrieveCall("beta")),
	}}
	_, _, err := Runner{Backend: backend, Retriever: &fakeRetriever{}, MaxRounds: 1}.Run(context.Background(), baseBody())
	if err == nil || !strings.Contains(err.Error(), "RAG loop exhausted") {
		t.Fatalf("want exhaustion error, got %v", err)
	}
}

func TestRunSearchFailureContinuesWithEmptyObservation(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{openAITools(retrieveCall("alpha")), openAIText("answer")}}
	retriever := &fakeRetriever{err: errors.New("search down")}
	body, _, err := Runner{Backend: backend, Retriever: retriever, MaxRounds: 3}.Run(context.Background(), baseBody())
	if err != nil {
		t.Fatalf("Run should degrade search errors: %v", err)
	}
	messages := jsonx.AsArr(body["messages"])
	if got := jsonx.GetStr(messages[len(messages)-1], "content"); !strings.Contains(got, "No relevant documents") {
		t.Fatalf("observation should be empty-result text, got %q", got)
	}
}

func TestRunFiltersRetrieveFromFinalMixedToolResponse(t *testing.T) {
	backend := &fakeBackend{responses: [][]byte{
		openAITools(retrieveCall("alpha")),
		openAITools(retrieveCall("beta"), externalCall("write_file")),
	}}
	_, finalBytes, err := Runner{Backend: backend, Retriever: &fakeRetriever{}, MaxRounds: 1}.Run(context.Background(), baseBody())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	calls := jsonx.AsArr(jsonx.Pointer(mustParse(t, finalBytes), "choices", "0", "message", "tool_calls"))
	if len(calls) != 1 {
		t.Fatalf("want only external call, got %v", calls)
	}
	if name := jsonx.GetStr(jsonx.Get(calls[0], "function"), "name"); name == rag.RetrieveToolName {
		t.Fatalf("internal retrieve leaked: %v", calls)
	}
}

func mustParse(t *testing.T, b []byte) any {
	t.Helper()
	v, err := jsonx.Parse(b)
	if err != nil {
		t.Fatalf("parse response: %v", err)
	}
	return v
}
