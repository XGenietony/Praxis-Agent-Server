// Package rag implements Agentic RAG: an internal `retrieve` tool the proxy
// intercepts and runs itself.
//
// Flow: the model emits a `retrieve` tool call -> the proxy embeds the query,
// searches Qdrant, feeds the chunks back as a tool result, and lets the model
// continue. Retrieval is invisible to the client.
//
// The package is split by responsibility:
//
//	rag.go     types, the retrieve-tool definition, and pure helpers (chunking,
//	           formatting) that have no I/O
//	client.go  the Qdrant + embedding HTTP client (EnsureCollection/Embed/
//	           Search/Ingest)
package rag

import (
	"fmt"
	"strings"

	"lmstudio-forward/internal/jsonx"
)

// RetrieveToolName is the name of the built-in retrieval tool injected into the
// model's tool list.
const RetrieveToolName = "retrieve"

// RetrievedChunk is a chunk returned from a vector search.
type RetrievedChunk struct {
	Text   string
	Score  float64
	Source string
}

// IngestDoc is a document to ingest into the knowledge base. An empty Source is
// treated as "unknown".
type IngestDoc struct {
	Text   string `json:"text"`
	Source string `json:"source"`
}

// RetrieveToolOpenAI returns the JSON definition of the built-in `retrieve` tool
// in OpenAI function shape (so it can be merged into openai_body["tools"] before
// tool injection).
func RetrieveToolOpenAI() any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": RetrieveToolName,
			"description": "Search the knowledge base for relevant information. Call this " +
				"whenever the user's question may depend on domain-specific facts, documents, " +
				"or context you are not certain about. Returns the most relevant text chunks.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "A focused natural-language search query describing what to find.",
					},
				},
				"required": []any{"query"},
			},
		},
	}
}

// RetrieveCallText reconstructs the `<tool_call>` text the model would have
// emitted for a retrieve query, so the appended assistant turn stays consistent
// with the prompt format.
func RetrieveCallText(query string) string {
	args := map[string]any{
		"name":      RetrieveToolName,
		"arguments": map[string]any{"query": query},
	}
	return fmt.Sprintf("<tool_call>\n%s\n</tool_call>", jsonx.MarshalString(args))
}

// FormatChunks renders retrieved chunks into a tool-result string fed back to
// the model.
func FormatChunks(chunks []RetrievedChunk) string {
	if len(chunks) == 0 {
		return "No relevant documents found in the knowledge base."
	}
	var out strings.Builder
	out.WriteString("Retrieved the following relevant passages:\n\n")
	for i, c := range chunks {
		out.WriteString(fmt.Sprintf("[%d] (source: %s, score: %.3f)\n%s\n\n", i+1, c.Source, c.Score, c.Text))
	}
	return out.String()
}

// ChunkText splits text into overlapping rune windows. Pure function.
func ChunkText(text string, size, overlap int) []string {
	chars := []rune(text)
	if len(chars) == 0 {
		return []string{}
	}
	if size < 1 {
		size = 1
	}
	if overlap > size-1 {
		overlap = size - 1
	}
	step := size - overlap

	var chunks []string
	start := 0
	for start < len(chars) {
		end := start + size
		if end > len(chars) {
			end = len(chars)
		}
		trimmed := strings.TrimSpace(string(chars[start:end]))
		if trimmed != "" {
			chunks = append(chunks, trimmed)
		}
		if end == len(chars) {
			break
		}
		start += step
	}
	return chunks
}
