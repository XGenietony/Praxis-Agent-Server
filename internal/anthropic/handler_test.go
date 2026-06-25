package anthropic

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"lmstudio-forward/internal/config"
	"lmstudio-forward/internal/proxy"
	"lmstudio-forward/internal/rag"
)

func TestMessagesRAGStreamUsesCollectedFinalResponse(t *testing.T) {
	var backendCalls int32
	backendPort, closeBackend := startFakeBackend(t, &backendCalls)
	defer closeBackend()

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/v1/embeddings":
			w.Write([]byte(`{"data":[{"embedding":[0.1,0.2]}]}`))
		case strings.HasSuffix(r.URL.Path, "/points/search"):
			w.Write([]byte(`{"result":[{"score":0.9,"payload":{"text":"retrieved doc","source":"test"}}]}`))
		default:
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"result":{}}`))
		}
	}))
	defer ragServer.Close()

	cfg := config.Config{
		BackendPort:           backendPort,
		CtxSize:               8192,
		RepetitionPenalty:     1.0,
		QdrantURL:             ragServer.URL,
		QdrantCollection:      "test",
		EmbedURL:              ragServer.URL,
		EmbedModel:            "embed",
		EmbedDim:              2,
		RagTopK:               1,
		RagMaxRounds:          3,
		RagStepTimeoutSeconds: 5,
	}
	state := &proxy.AppState{
		Config:     cfg,
		HTTPClient: ragServer.Client(),
		Rag:        rag.NewClient(ragServer.Client(), &cfg),
	}
	handler := NewHandler(state)

	body := `{"model":"claude-test","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"question"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.Messages(rec, req)

	if got := atomic.LoadInt32(&backendCalls); got != 2 {
		t.Fatalf("backend calls: want retrieve probe + final answer = 2, got %d\nbody:\n%s", got, rec.Body.String())
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status: want 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	if !strings.Contains(out, "final answer") {
		t.Fatalf("stream should contain final answer, got:\n%s", out)
	}
	if strings.Contains(out, `"name":"retrieve"`) || strings.Contains(out, `"name": "retrieve"`) {
		t.Fatalf("internal retrieve leaked to stream:\n%s", out)
	}
}

func startFakeBackend(t *testing.T, calls *int32) (int, func()) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(calls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		switch n {
		case 1:
			writeSSECompletion(w, "chatcmpl-retrieve", `<tool_call>
{"name":"retrieve","arguments":{"query":"question"}}
</tool_call>`, 1)
		case 2:
			writeSSECompletion(w, "chatcmpl-final", "final answer", 2)
		default:
			writeSSECompletion(w, "chatcmpl-extra", "unexpected extra answer", 3)
		}
	})

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen fake backend: %v", err)
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 2 * time.Second}
	go func() {
		if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
			t.Errorf("fake backend serve: %v", err)
		}
	}()
	return ln.Addr().(*net.TCPAddr).Port, func() { _ = srv.Close() }
}

func writeSSECompletion(w http.ResponseWriter, id, content string, completionTokens int) {
	fmt.Fprintf(w, "data: {\"id\":%q,\"model\":\"test\",\"choices\":[{\"delta\":{\"content\":%q},\"finish_reason\":null}],\"usage\":null}\n\n", id, content)
	fmt.Fprintf(w, "data: {\"id\":%q,\"model\":\"test\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":%d}}\n\n", id, completionTokens)
	fmt.Fprint(w, "data: [DONE]\n\n")
}
