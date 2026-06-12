use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::error;

/// Consume an SSE stream and reconstruct a complete OpenAI chat.completion JSON.
pub async fn collect_stream_to_response(resp: reqwest::Response) -> bytes::Bytes {
    use futures_util::StreamExt;
    use serde_json::Value;

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut id = String::new();
    let mut model_str = String::new();
    let mut finish_reason: Value = Value::Null;
    let mut usage: Value = Value::Null;

    let mut done = false;
    'outer: while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                error!("Stream read error: {}", e);
                break;
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line: String = buffer.drain(..=pos).collect();
            let line = line.trim();
            if !line.starts_with("data: ") { continue; }
            let payload = &line[6..];
            if payload == "[DONE]" {
                done = true;
                break 'outer;
            }

            if let Ok(v) = serde_json::from_str::<Value>(payload) {
                if id.is_empty() {
                    if let Some(i) = v.get("id").and_then(|i| i.as_str()) {
                        id = i.to_string();
                    }
                }
                if model_str.is_empty() {
                    if let Some(m) = v.get("model").and_then(|m| m.as_str()) {
                        model_str = m.to_string();
                    }
                }
                if let Some(delta) = v.pointer("/choices/0/delta") {
                    if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                        content.push_str(c);
                    }
                    if let Some(r) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                        reasoning.push_str(r);
                    }
                }
                if let Some(fr) = v.pointer("/choices/0/finish_reason") {
                    if !fr.is_null() {
                        finish_reason = fr.clone();
                    }
                }
                if let Some(u) = v.get("usage") {
                    if !u.is_null() {
                        usage = u.clone();
                    }
                }
            }
        }
    }
    drop(stream);
    if !done {
        error!("Stream ended without [DONE] marker");
    }

    let mut message = serde_json::json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning);
    }

    let resp_json = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "model": model_str,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage
    });

    bytes::Bytes::from(serde_json::to_vec(&resp_json).unwrap())
}

/// Build an SSE streaming response with standard headers.
pub fn sse_response(body: Body) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .header(header::CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .header("X-Content-Type-Options", "nosniff")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(body)
        .unwrap()
        .into_response()
}
