//! Agentic RAG: an internal `retrieve` tool the proxy intercepts and runs itself.
//!
//! Flow: the model emits a `retrieve` tool call → the proxy embeds the query,
//! searches Qdrant, feeds the chunks back as a tool result, and lets the model
//! continue. Retrieval is invisible to the client.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

/// Name of the built-in retrieval tool injected into the model's tool list.
pub const RETRIEVE_TOOL_NAME: &str = "retrieve";

/// A chunk returned from a vector search.
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    pub text: String,
    pub score: f64,
    pub source: String,
}

/// A document to ingest into the knowledge base.
#[derive(Debug, Deserialize)]
pub struct IngestDoc {
    pub text: String,
    #[serde(default)]
    pub source: Option<String>,
}

/// Thin client over Qdrant REST + an OpenAI-compatible embedding endpoint.
#[derive(Clone)]
pub struct RagClient {
    http: reqwest::Client,
    qdrant_url: String,
    collection: String,
    embed_url: String,
    embed_model: String,
    embed_dim: u64,
    top_k: u64,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl RagClient {
    pub fn new(http: reqwest::Client, cfg: &crate::config::Config) -> Self {
        Self {
            http,
            qdrant_url: cfg.qdrant_url.trim_end_matches('/').to_string(),
            collection: cfg.qdrant_collection.clone(),
            embed_url: cfg.embed_url.trim_end_matches('/').to_string(),
            embed_model: cfg.embed_model.clone(),
            embed_dim: cfg.embed_dim,
            top_k: cfg.rag_top_k,
            chunk_size: cfg.rag_chunk_size,
            chunk_overlap: cfg.rag_chunk_overlap,
        }
    }

    /// Idempotently create the Qdrant collection (Cosine distance).
    pub async fn ensure_collection(&self) -> Result<()> {
        let url = format!("{}/collections/{}", self.qdrant_url, self.collection);
        let existing = self.http.get(&url).send().await;
        if let Ok(resp) = existing {
            if resp.status().is_success() {
                info!("RAG: collection '{}' already exists", self.collection);
                return Ok(());
            }
        }

        let body = json!({
            "vectors": { "size": self.embed_dim, "distance": "Cosine" }
        });
        let resp = self
            .http
            .put(&url)
            .json(&body)
            .send()
            .await
            .context("qdrant create collection request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant create collection {}: {}", status, text));
        }
        info!(
            "RAG: created collection '{}' (dim={}, Cosine)",
            self.collection, self.embed_dim
        );
        Ok(())
    }

    /// Embed a batch of texts via the OpenAI-compatible `/v1/embeddings` endpoint.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let url = format!("{}/v1/embeddings", self.embed_url);
        let body = json!({ "model": self.embed_model, "input": texts });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("embedding request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("embedding endpoint {}: {}", status, text));
        }
        let value: Value = resp.json().await.context("embedding response not JSON")?;
        let data = value
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow!("embedding response missing `data` array"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| anyhow!("embedding item missing `embedding`"))?;
            out.push(
                emb.iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect(),
            );
        }
        Ok(out)
    }

    /// Embed the query and return the top-k matching chunks from Qdrant.
    pub async fn search(&self, query: &str) -> Result<Vec<RetrievedChunk>> {
        let vectors = self.embed(&[query.to_string()]).await?;
        let vector = vectors
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embedding returned no vector for query"))?;

        let url = format!(
            "{}/collections/{}/points/search",
            self.qdrant_url, self.collection
        );
        let body = json!({
            "vector": vector,
            "limit": self.top_k,
            "with_payload": true
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("qdrant search request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant search {}: {}", status, text));
        }
        let value: Value = resp.json().await.context("qdrant search response not JSON")?;
        let result = value
            .get("result")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow!("qdrant search response missing `result`"))?;

        let mut chunks = Vec::with_capacity(result.len());
        for hit in result {
            let score = hit.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
            let payload = hit.get("payload");
            let text = payload
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let source = payload
                .and_then(|p| p.get("source"))
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_string();
            if !text.is_empty() {
                chunks.push(RetrievedChunk { text, score, source });
            }
        }
        Ok(chunks)
    }

    /// Chunk → embed → upsert documents into Qdrant. Returns number of chunks stored.
    pub async fn ingest(&self, docs: Vec<IngestDoc>) -> Result<usize> {
        let mut texts: Vec<String> = Vec::new();
        let mut sources: Vec<String> = Vec::new();
        for doc in &docs {
            let source = doc.source.clone().unwrap_or_else(|| "unknown".to_string());
            for chunk in chunk_text(&doc.text, self.chunk_size, self.chunk_overlap) {
                texts.push(chunk);
                sources.push(source.clone());
            }
        }
        if texts.is_empty() {
            return Ok(0);
        }

        let vectors = self.embed(&texts).await?;
        if vectors.len() != texts.len() {
            return Err(anyhow!(
                "embedding count {} != chunk count {}",
                vectors.len(),
                texts.len()
            ));
        }

        let points: Vec<Value> = texts
            .iter()
            .zip(sources.iter())
            .zip(vectors.iter())
            .map(|((text, source), vector)| {
                json!({
                    "id": uuid::Uuid::new_v4().to_string(),
                    "vector": vector,
                    "payload": { "text": text, "source": source }
                })
            })
            .collect();

        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.qdrant_url, self.collection
        );
        let resp = self
            .http
            .put(&url)
            .json(&json!({ "points": points }))
            .send()
            .await
            .context("qdrant upsert request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("qdrant upsert {}: {}", status, text));
        }
        info!("RAG: ingested {} chunks from {} docs", texts.len(), docs.len());
        Ok(texts.len())
    }
}

/// JSON definition of the built-in `retrieve` tool, in OpenAI function shape
/// (so it can be merged into `openai_body["tools"]` before tool injection).
pub fn retrieve_tool_openai() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": RETRIEVE_TOOL_NAME,
            "description": "Search the knowledge base for relevant information. Call this \
                whenever the user's question may depend on domain-specific facts, documents, \
                or context you are not certain about. Returns the most relevant text chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused natural-language search query describing what to find."
                    }
                },
                "required": ["query"]
            }
        }
    })
}

/// Reconstruct the `<tool_call>` text the model would have emitted for a retrieve
/// query, so the appended assistant turn stays consistent with the prompt format.
pub fn retrieve_call_text(query: &str) -> String {
    let args = json!({ "name": RETRIEVE_TOOL_NAME, "arguments": { "query": query } });
    format!("<tool_call>\n{}\n</tool_call>", args)
}

/// Render retrieved chunks into a tool-result string fed back to the model.
pub fn format_chunks(chunks: &[RetrievedChunk]) -> String {
    if chunks.is_empty() {
        return "No relevant documents found in the knowledge base.".to_string();
    }
    let mut out = String::from("Retrieved the following relevant passages:\n\n");
    for (i, c) in chunks.iter().enumerate() {
        out.push_str(&format!(
            "[{}] (source: {}, score: {:.3})\n{}\n\n",
            i + 1,
            c.source,
            c.score,
            c.text
        ));
    }
    out
}

/// Split text into overlapping character windows. Pure function (unit-tested).
pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return vec![];
    }
    let size = size.max(1);
    let overlap = overlap.min(size - 1);
    let step = size - overlap;

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if end == chars.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("", 100, 10).is_empty());
        assert!(chunk_text("   ", 100, 10).is_empty());
    }

    #[test]
    fn short_text_yields_single_chunk() {
        let chunks = chunk_text("hello world", 100, 10);
        assert_eq!(chunks, vec!["hello world".to_string()]);
    }

    #[test]
    fn long_text_splits_with_overlap() {
        let text: String = "abcdefghij".repeat(3); // 30 chars
        let chunks = chunk_text(&text, 10, 4);
        // step = 6: starts at 0,6,12,18,24 → 5 chunks
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0].chars().count(), 10);
    }

    #[test]
    fn overlap_clamped_below_size() {
        // overlap >= size must not panic / infinite-loop
        let chunks = chunk_text("abcdefghij", 5, 100);
        assert!(!chunks.is_empty());
    }
}
