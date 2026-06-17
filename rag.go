package main

// Agentic RAG: an internal `retrieve` tool the proxy intercepts and runs itself.
//
// Flow: the model emits a `retrieve` tool call -> the proxy embeds the query,
// searches Qdrant, feeds the chunks back as a tool result, and lets the model
// continue. Retrieval is invisible to the client.

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
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

// IngestDoc is a document to ingest into the knowledge base. An empty Source
// is treated as "unknown".
type IngestDoc struct {
	Text   string `json:"text"`
	Source string `json:"source"`
}

// RagClient is a thin client over Qdrant REST + an OpenAI-compatible embedding
// endpoint.
type RagClient struct {
	http         *http.Client
	qdrantURL    string
	collection   string
	embedURL     string
	embedModel   string
	embedDim     int
	topK         int
	chunkSize    int
	chunkOverlap int
}

// newRagClient builds a RagClient from config, trimming trailing slashes from
// the Qdrant and embedding base URLs.
func newRagClient(httpClient *http.Client, cfg *Config) *RagClient {
	return &RagClient{
		http:         httpClient,
		qdrantURL:    strings.TrimRight(cfg.QdrantURL, "/"),
		collection:   cfg.QdrantCollection,
		embedURL:     strings.TrimRight(cfg.EmbedURL, "/"),
		embedModel:   cfg.EmbedModel,
		embedDim:     cfg.EmbedDim,
		topK:         cfg.RagTopK,
		chunkSize:    cfg.RagChunkSize,
		chunkOverlap: cfg.RagChunkOverlap,
	}
}

// sendJSON issues an HTTP request with a JSON body and the appropriate
// Content-Type header.
func (c *RagClient) sendJSON(method, url string, body any) (*http.Response, error) {
	req, err := http.NewRequest(method, url, bytes.NewReader(toJSON(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.http.Do(req)
}

// isSuccess reports whether the response carries a 2xx status code.
func isSuccess(resp *http.Response) bool {
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// bodyString reads the response body to a string, returning "" on error.
func bodyString(resp *http.Response) string {
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}
	return string(b)
}

// EnsureCollection idempotently creates the Qdrant collection (Cosine distance).
func (c *RagClient) EnsureCollection() error {
	url := fmt.Sprintf("%s/collections/%s", c.qdrantURL, c.collection)
	if resp, err := c.http.Get(url); err == nil {
		ok := isSuccess(resp)
		resp.Body.Close()
		if ok {
			log.Printf("INFO RAG: collection '%s' already exists", c.collection)
			return nil
		}
	}

	body := map[string]any{
		"vectors": map[string]any{"size": c.embedDim, "distance": "Cosine"},
	}
	resp, err := c.sendJSON(http.MethodPut, url, body)
	if err != nil {
		return fmt.Errorf("qdrant create collection request failed: %w", err)
	}
	defer resp.Body.Close()
	if !isSuccess(resp) {
		return fmt.Errorf("qdrant create collection %s: %s", resp.Status, bodyString(resp))
	}
	log.Printf("INFO RAG: created collection '%s' (dim=%d, Cosine)", c.collection, c.embedDim)
	return nil
}

// Embed embeds a batch of texts via the OpenAI-compatible `/v1/embeddings`
// endpoint.
func (c *RagClient) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	url := fmt.Sprintf("%s/v1/embeddings", c.embedURL)
	body := map[string]any{"model": c.embedModel, "input": texts}
	resp, err := c.sendJSON(http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}
	defer resp.Body.Close()
	if !isSuccess(resp) {
		return nil, fmt.Errorf("embedding endpoint %s: %s", resp.Status, bodyString(resp))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("embedding response not JSON: %w", err)
	}
	value, err := parseJSON(raw)
	if err != nil {
		return nil, fmt.Errorf("embedding response not JSON: %w", err)
	}
	data := getArr(value, "data")
	if data == nil {
		return nil, fmt.Errorf("embedding response missing `data` array")
	}
	out := make([][]float32, 0, len(data))
	for _, item := range data {
		emb := getArr(item, "embedding")
		if emb == nil {
			return nil, fmt.Errorf("embedding item missing `embedding`")
		}
		vec := make([]float32, 0, len(emb))
		for _, v := range emb {
			f, _ := numv(v)
			vec = append(vec, float32(f))
		}
		out = append(out, vec)
	}
	return out, nil
}

// Search embeds the query and returns the top-k matching chunks from Qdrant.
func (c *RagClient) Search(query string) ([]RetrievedChunk, error) {
	vectors, err := c.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("embedding returned no vector for query")
	}
	vector := vectors[0]

	url := fmt.Sprintf("%s/collections/%s/points/search", c.qdrantURL, c.collection)
	body := map[string]any{
		"vector":       vector,
		"limit":        c.topK,
		"with_payload": true,
	}
	resp, err := c.sendJSON(http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("qdrant search request failed: %w", err)
	}
	defer resp.Body.Close()
	if !isSuccess(resp) {
		return nil, fmt.Errorf("qdrant search %s: %s", resp.Status, bodyString(resp))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("qdrant search response not JSON: %w", err)
	}
	value, err := parseJSON(raw)
	if err != nil {
		return nil, fmt.Errorf("qdrant search response not JSON: %w", err)
	}
	result := getArr(value, "result")
	if result == nil {
		return nil, fmt.Errorf("qdrant search response missing `result`")
	}

	chunks := make([]RetrievedChunk, 0, len(result))
	for _, hit := range result {
		score, _ := getNum(hit, "score")
		payload := get(hit, "payload")
		text := getStr(payload, "text")
		source := "unknown"
		if s, ok := asStr(get(payload, "source")); ok {
			source = s
		}
		if text != "" {
			chunks = append(chunks, RetrievedChunk{Text: text, Score: score, Source: source})
		}
	}
	return chunks, nil
}

// Ingest chunks, embeds, and upserts documents into Qdrant. It returns the
// number of chunks stored.
func (c *RagClient) Ingest(docs []IngestDoc) (int, error) {
	var texts []string
	var sources []string
	for i := range docs {
		source := docs[i].Source
		if source == "" {
			source = "unknown"
		}
		for _, chunk := range chunkText(docs[i].Text, c.chunkSize, c.chunkOverlap) {
			texts = append(texts, chunk)
			sources = append(sources, source)
		}
	}
	if len(texts) == 0 {
		return 0, nil
	}

	vectors, err := c.Embed(texts)
	if err != nil {
		return 0, err
	}
	if len(vectors) != len(texts) {
		return 0, fmt.Errorf("embedding count %d != chunk count %d", len(vectors), len(texts))
	}

	points := make([]any, 0, len(texts))
	for i := range texts {
		points = append(points, map[string]any{
			"id":     uuidV4(),
			"vector": vectors[i],
			"payload": map[string]any{
				"text":   texts[i],
				"source": sources[i],
			},
		})
	}

	url := fmt.Sprintf("%s/collections/%s/points?wait=true", c.qdrantURL, c.collection)
	resp, err := c.sendJSON(http.MethodPut, url, map[string]any{"points": points})
	if err != nil {
		return 0, fmt.Errorf("qdrant upsert request failed: %w", err)
	}
	defer resp.Body.Close()
	if !isSuccess(resp) {
		return 0, fmt.Errorf("qdrant upsert %s: %s", resp.Status, bodyString(resp))
	}
	log.Printf("INFO RAG: ingested %d chunks from %d docs", len(texts), len(docs))
	return len(texts), nil
}

// uuidV4 generates a random RFC 4122 version-4 UUID string using crypto/rand.
func uuidV4() string {
	var b [16]byte
	rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// retrieveToolOpenAI returns the JSON definition of the built-in `retrieve`
// tool in OpenAI function shape (so it can be merged into
// openai_body["tools"] before tool injection).
func retrieveToolOpenAI() any {
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

// retrieveCallText reconstructs the `<tool_call>` text the model would have
// emitted for a retrieve query, so the appended assistant turn stays consistent
// with the prompt format.
func retrieveCallText(query string) string {
	args := map[string]any{
		"name":      RetrieveToolName,
		"arguments": map[string]any{"query": query},
	}
	return fmt.Sprintf("<tool_call>\n%s\n</tool_call>", toJSONString(args))
}

// formatChunks renders retrieved chunks into a tool-result string fed back to
// the model.
func formatChunks(chunks []RetrievedChunk) string {
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

// chunkText splits text into overlapping rune windows. Pure function.
func chunkText(text string, size, overlap int) []string {
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
