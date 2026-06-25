package rag

import (
	"bytes"
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"lmstudio-forward/internal/config"
	"lmstudio-forward/internal/jsonx"
)

// Client is a thin client over Qdrant REST + an OpenAI-compatible embedding
// endpoint.
type Client struct {
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

// NewClient builds a Client from config, trimming trailing slashes from the
// Qdrant and embedding base URLs.
func NewClient(httpClient *http.Client, cfg *config.Config) *Client {
	return &Client{
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
func (c *Client) sendJSON(ctx context.Context, method, url string, body any) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(jsonx.Marshal(body)))
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
func (c *Client) EnsureCollection(ctx context.Context) error {
	url := fmt.Sprintf("%s/collections/%s", c.qdrantURL, c.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if resp, err := c.http.Do(req); err == nil {
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
	resp, err := c.sendJSON(ctx, http.MethodPut, url, body)
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
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	url := fmt.Sprintf("%s/v1/embeddings", c.embedURL)
	body := map[string]any{"model": c.embedModel, "input": texts}
	resp, err := c.sendJSON(ctx, http.MethodPost, url, body)
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
	value, err := jsonx.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("embedding response not JSON: %w", err)
	}
	data := jsonx.GetArr(value, "data")
	if data == nil {
		return nil, fmt.Errorf("embedding response missing `data` array")
	}
	out := make([][]float32, 0, len(data))
	for _, item := range data {
		emb := jsonx.GetArr(item, "embedding")
		if emb == nil {
			return nil, fmt.Errorf("embedding item missing `embedding`")
		}
		vec := make([]float32, 0, len(emb))
		for _, v := range emb {
			f, _ := jsonx.Num(v)
			vec = append(vec, float32(f))
		}
		out = append(out, vec)
	}
	return out, nil
}

// Search embeds the query and returns the top-k matching chunks from Qdrant.
func (c *Client) Search(ctx context.Context, query string) ([]RetrievedChunk, error) {
	vectors, err := c.Embed(ctx, []string{query})
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
	resp, err := c.sendJSON(ctx, http.MethodPost, url, body)
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
	value, err := jsonx.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("qdrant search response not JSON: %w", err)
	}
	result := jsonx.GetArr(value, "result")
	if result == nil {
		return nil, fmt.Errorf("qdrant search response missing `result`")
	}

	chunks := make([]RetrievedChunk, 0, len(result))
	for _, hit := range result {
		score, _ := jsonx.GetNum(hit, "score")
		payload := jsonx.Get(hit, "payload")
		text := jsonx.GetStr(payload, "text")
		source := "unknown"
		if s, ok := jsonx.AsStr(jsonx.Get(payload, "source")); ok {
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
func (c *Client) Ingest(ctx context.Context, docs []IngestDoc) (int, error) {
	var texts []string
	var sources []string
	for i := range docs {
		source := docs[i].Source
		if source == "" {
			source = "unknown"
		}
		for _, chunk := range ChunkText(docs[i].Text, c.chunkSize, c.chunkOverlap) {
			texts = append(texts, chunk)
			sources = append(sources, source)
		}
	}
	if len(texts) == 0 {
		return 0, nil
	}

	vectors, err := c.Embed(ctx, texts)
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
	resp, err := c.sendJSON(ctx, http.MethodPut, url, map[string]any{"points": points})
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
