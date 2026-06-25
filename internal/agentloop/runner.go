// Package agentloop runs internal model action loops such as Agentic RAG.
// Protocol adapters provide backend completion and tool implementations; this
// package owns the loop state, internal action handling, and stop policy.
package agentloop

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"lmstudio-forward/internal/jsonx"
	"lmstudio-forward/internal/rag"
)

// Backend completes one model turn for the current request body.
type Backend interface {
	Complete(ctx context.Context, body map[string]any) ([]byte, error)
}

// Retriever is the internal retrieve tool implementation.
type Retriever interface {
	Search(ctx context.Context, query string) ([]rag.RetrievedChunk, error)
}

// Runner consumes internal retrieve tool calls until the model produces a
// client-visible answer/tool call or the max-round policy stops the loop.
type Runner struct {
	Backend   Backend
	Retriever Retriever
	MaxRounds int
	// StepTimeout bounds one hidden backend/retrieval step. A zero or negative
	// value means the caller's context is used without adding a timeout.
	StepTimeout time.Duration
}

type toolCall struct {
	Name string
	Args string
}

// Run returns the augmented request body and the final backend response bytes.
func (r Runner) Run(ctx context.Context, body map[string]any) (map[string]any, []byte, error) {
	rounds := r.MaxRounds
	if rounds < 1 {
		rounds = 1
	}

	for round := 0; round < rounds; round++ {
		stepCtx, cancel := r.stepContext(ctx)
		respBytes, err := r.Backend.Complete(stepCtx, body)
		cancel()
		if err != nil {
			return nil, nil, err
		}
		resp, err := jsonx.Parse(respBytes)
		if err != nil {
			return nil, nil, err
		}
		calls := extractToolCalls(resp)

		retrieveCalls, _ := splitRetrieveToolCalls(calls)
		if len(retrieveCalls) == 0 {
			return body, respBytes, nil
		}

		// Consume internal retrieve calls even when the model also emitted
		// client tools. The next round lets the model choose external tools again
		// after seeing the retrieved observation.
		if err := r.appendRetrieveObservations(ctx, body, retrieveCalls, round); err != nil {
			return nil, nil, err
		}
	}

	if messages := jsonx.AsArr(body["messages"]); messages != nil {
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": "You have gathered enough context. Answer the question now using the information above. Do not call the retrieve tool again.",
		})
		body["messages"] = messages
	}
	stepCtx, cancel := r.stepContext(ctx)
	finalBytes, err := r.Backend.Complete(stepCtx, body)
	cancel()
	if err != nil {
		return nil, nil, err
	}
	resp, err := jsonx.Parse(finalBytes)
	if err != nil {
		return nil, nil, err
	}
	calls := extractToolCalls(resp)
	retrieveCalls, externalCalls := splitRetrieveToolCalls(calls)
	if len(retrieveCalls) > 0 && len(externalCalls) == 0 {
		return nil, nil, fmt.Errorf("RAG loop exhausted after %d rounds with only internal retrieve calls", rounds)
	}
	if len(retrieveCalls) > 0 {
		filtered, err := stripRetrieveToolCalls(resp)
		if err != nil {
			return nil, nil, err
		}
		return body, filtered, nil
	}
	return body, finalBytes, nil
}

func (r Runner) stepContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if r.StepTimeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, r.StepTimeout)
}

func extractToolCalls(resp any) []toolCall {
	calls := jsonx.AsArr(jsonx.Pointer(resp, "choices", "0", "message", "tool_calls"))
	var out []toolCall
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
		out = append(out, toolCall{Name: name, Args: args})
	}
	return out
}

func splitRetrieveToolCalls(calls []toolCall) (retrieve []toolCall, external []toolCall) {
	for _, c := range calls {
		if c.Name == rag.RetrieveToolName {
			retrieve = append(retrieve, c)
		} else {
			external = append(external, c)
		}
	}
	return retrieve, external
}

func (r Runner) appendRetrieveObservations(ctx context.Context, body map[string]any, calls []toolCall, round int) error {
	for _, c := range calls {
		query := ""
		if v, err := jsonx.Parse([]byte(c.Args)); err == nil {
			query = jsonx.Str(jsonx.Get(v, "query"))
		}
		if query == "" {
			continue
		}
		stepCtx, cancel := r.stepContext(ctx)
		chunks, err := r.Retriever.Search(stepCtx, query)
		cancel()
		if err != nil && ctx.Err() != nil {
			return ctx.Err()
		}
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
	return nil
}

func stripRetrieveToolCalls(resp any) ([]byte, error) {
	choices := jsonx.AsArr(jsonx.Get(resp, "choices"))
	for _, ch := range choices {
		choice := jsonx.AsObj(ch)
		if choice == nil {
			continue
		}
		msg := jsonx.AsObj(choice["message"])
		if msg == nil {
			continue
		}
		calls := jsonx.AsArr(msg["tool_calls"])
		if len(calls) == 0 {
			continue
		}

		filtered := make([]any, 0, len(calls))
		for _, tc := range calls {
			name := jsonx.GetStr(jsonx.Get(tc, "function"), "name")
			if name != rag.RetrieveToolName {
				filtered = append(filtered, tc)
			}
		}
		if len(filtered) == 0 {
			delete(msg, "tool_calls")
			if msg["content"] == nil {
				msg["content"] = ""
			}
			choice["finish_reason"] = "stop"
			continue
		}
		msg["tool_calls"] = filtered
		choice["finish_reason"] = "tool_calls"
	}
	out := jsonx.Marshal(resp)
	if out == nil {
		return nil, errors.New("failed to marshal response after filtering internal tools")
	}
	return out, nil
}
