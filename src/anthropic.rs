//! Anthropic Messages API compatibility layer.
//! Converts Anthropic <-> OpenAI protocol and proxies to LM Studio.

use serde_json::{json, Value};
use axum::{
    body::Body,
    extract::{Request, State, ConnectInfo},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use tracing::{info, error};
use crate::proxy::{AppState, check_api_key, get_client_ip};
use crate::stream::sse_response;
use crate::tools::{ToolCallStreamParser, ToolStreamEvent};

fn anthropic_error(status: StatusCode, error_type: &str, message: &str) -> Response {
    (status, Json(json!({
        "type": "error",
        "error": {"type": error_type, "message": message}
    }))).into_response()
}

// ─── Types ──────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct StreamState {
    started: bool,
    thinking_started: bool,
    text_started: bool,
    block_index: usize,
    pending_stop_reason: Option<String>,
    finished: bool,
}

// ─── Protocol conversion ────────────────────────────────────────────────────

/// Convert OpenAI response to Anthropic format (non-streaming).
fn openai_to_anthropic(resp: Value, model: &str) -> Value {
    let empty_vec = vec![];
    let choices = resp.get("choices").and_then(|c| c.as_array()).unwrap_or(&empty_vec);
    let choice = choices.first().unwrap_or(&Value::Null);
    let message = choice.get("message").unwrap_or(&Value::Null);
    let usage = resp.get("usage").unwrap_or(&Value::Null);

    let mut content_blocks = Vec::new();

    if let Some(reasoning) = message.get("reasoning_content").and_then(|r| r.as_str())
        .or_else(|| message.get("reasoning").and_then(|r| r.as_str())) {
        if !reasoning.is_empty() {
            content_blocks.push(json!({"type": "thinking", "thinking": reasoning}));
        }
    }

    let text = message.get("content").and_then(|c| c.as_str()).unwrap_or("");
    if !text.is_empty() {
        content_blocks.push(json!({"type": "text", "text": text}));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        for (i, tc) in tool_calls.iter().enumerate() {
            if let Some(func) = tc.get("function") {
                let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args_str = func.get("arguments").and_then(|a| a.as_str()).unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                let id_str = format!("toolu_{:04x}", i);
                let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or(&id_str);
                content_blocks.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
            }
        }
    }

    if content_blocks.is_empty() {
        content_blocks.push(json!({"type": "text", "text": ""}));
    }

    let stop_reason = match choice.get("finish_reason").and_then(|f| f.as_str()) {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        _ => "end_turn",
    };

    let id = resp.get("id").and_then(|i| i.as_str()).unwrap_or("unknown");
    let prompt_tokens = usage.get("prompt_tokens").unwrap_or(&Value::Number(0.into())).clone();
    let completion_tokens = usage.get("completion_tokens").unwrap_or(&Value::Number(0.into())).clone();

    json!({
        "id": format!("msg_{}", id),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}
    })
}

/// Convert a single OpenAI SSE chunk to Anthropic SSE events.
fn openai_chunk_to_anthropic(
    chunk: &Value,
    state: &mut StreamState,
    model: &str,
    tool_parser: &mut ToolCallStreamParser,
) -> Vec<Value> {
    let mut events = Vec::new();

    if !state.started {
        state.started = true;
        let id = chunk.get("id").and_then(|i| i.as_str()).unwrap_or("unknown");
        events.push(json!({
            "type": "message_start",
            "message": {
                "id": format!("msg_{}", id), "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }));
    }

    let empty_vec = vec![];
    let choices = chunk.get("choices").and_then(|c| c.as_array()).unwrap_or(&empty_vec);
    let choice = choices.first().unwrap_or(&Value::Null);
    let delta = choice.get("delta").unwrap_or(&Value::Null);

    // Reasoning content → thinking block (support both "reasoning_content" and "reasoning")
    if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str())
        .or_else(|| delta.get("reasoning").and_then(|r| r.as_str())) {
        if !reasoning.is_empty() {
            if !state.thinking_started {
                state.thinking_started = true;
                events.push(json!({
                    "type": "content_block_start", "index": state.block_index,
                    "content_block": {"type": "thinking", "thinking": ""}
                }));
            }
            events.push(json!({
                "type": "content_block_delta", "index": state.block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning}
            }));
        }
    }

    // Text content → feed through tool parser
    if let Some(text) = delta.get("content").and_then(|t| t.as_str()) {
        if !text.is_empty() {
            let tool_events = tool_parser.feed(text);
            for te in tool_events {
                match te {
                    ToolStreamEvent::Text(t) => {
                        if !t.is_empty() {
                            ensure_text_block(state, &mut events);
                            events.push(json!({
                                "type": "content_block_delta", "index": state.block_index,
                                "delta": {"type": "text_delta", "text": t}
                            }));
                        }
                    }
                    ToolStreamEvent::ToolCallComplete(tc) => {
                        if state.text_started {
                            events.push(json!({"type": "content_block_stop", "index": state.block_index}));
                            state.block_index += 1;
                            state.text_started = false;
                        }
                        let input: Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                        let id = tc.id.unwrap_or_else(|| format!("toolu_{:04x}", state.block_index));
                        events.push(json!({
                            "type": "content_block_start", "index": state.block_index,
                            "content_block": {"type": "tool_use", "id": id, "name": tc.name, "input": {}}
                        }));
                        events.push(json!({
                            "type": "content_block_delta", "index": state.block_index,
                            "delta": {"type": "input_json_delta", "partial_json": serde_json::to_string(&input).unwrap_or("{}".to_string())}
                        }));
                        events.push(json!({"type": "content_block_stop", "index": state.block_index}));
                        state.block_index += 1;
                    }
                }
            }
        }
    }

    // Finish reason
    if let Some(finish) = choice.get("finish_reason").and_then(|f| f.as_str()) {
        if !finish.is_empty() {
            // Flush tool parser
            for te in tool_parser.flush() {
                match te {
                    ToolStreamEvent::Text(t) => {
                        if !t.is_empty() {
                            ensure_text_block(state, &mut events);
                            events.push(json!({
                                "type": "content_block_delta", "index": state.block_index,
                                "delta": {"type": "text_delta", "text": t}
                            }));
                        }
                    }
                    ToolStreamEvent::ToolCallComplete(tc) => {
                        if state.text_started {
                            events.push(json!({"type": "content_block_stop", "index": state.block_index}));
                            state.block_index += 1;
                            state.text_started = false;
                        }
                        let input: Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                        let id = tc.id.unwrap_or_else(|| format!("toolu_{:04x}", state.block_index));
                        events.push(json!({
                            "type": "content_block_start", "index": state.block_index,
                            "content_block": {"type": "tool_use", "id": id, "name": tc.name, "input": {}}
                        }));
                        events.push(json!({
                            "type": "content_block_delta", "index": state.block_index,
                            "delta": {"type": "input_json_delta", "partial_json": serde_json::to_string(&input).unwrap_or("{}".to_string())}
                        }));
                        events.push(json!({"type": "content_block_stop", "index": state.block_index}));
                        state.block_index += 1;
                    }
                }
            }

            let stop_reason = if tool_parser.has_seen_tools() || finish == "tool_calls" {
                "tool_use"
            } else if finish == "length" {
                "max_tokens"
            } else {
                "end_turn"
            };

            if state.thinking_started || state.text_started {
                events.push(json!({"type": "content_block_stop", "index": state.block_index}));
                state.thinking_started = false;
                state.text_started = false;
            }

            let usage = chunk.get("usage").and_then(|u| if u.is_null() { None } else { Some(u) });
            if let Some(usage) = usage {
                let ct = usage.get("completion_tokens").unwrap_or(&Value::Number(0.into())).clone();
                events.push(json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                    "usage": {"output_tokens": ct}
                }));
                events.push(json!({"type": "message_stop"}));
                state.finished = true;
            } else {
                state.pending_stop_reason = Some(stop_reason.to_string());
            }
        }
    }

    // Deferred finish: usage-only chunk
    if let Some(stop_reason) = state.pending_stop_reason.take() {
        if let Some(usage) = chunk.get("usage").and_then(|u| if u.is_null() { None } else { Some(u) }) {
            let ct = usage.get("completion_tokens").unwrap_or(&Value::Number(0.into())).clone();
            events.push(json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": ct}
            }));
            events.push(json!({"type": "message_stop"}));
            state.finished = true;
        } else {
            state.pending_stop_reason = Some(stop_reason);
        }
    }

    events
}

fn ensure_text_block(state: &mut StreamState, events: &mut Vec<Value>) {
    if !state.text_started {
        if state.thinking_started {
            events.push(json!({"type": "content_block_stop", "index": state.block_index}));
            state.block_index += 1;
            state.thinking_started = false;
        }
        state.text_started = true;
        events.push(json!({
            "type": "content_block_start", "index": state.block_index,
            "content_block": {"type": "text", "text": ""}
        }));
    }
}

// ─── Handler ────────────────────────────────────────────────────────────────

pub async fn forward_anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<std::net::SocketAddr>,
    req: Request<Body>,
) -> impl IntoResponse {
    if let Err(_) = check_api_key(&headers, &state.config.api_key) {
        return anthropic_error(StatusCode::UNAUTHORIZED, "authentication_error", "Invalid API key");
    }

    let client_ip = get_client_ip(&headers, Some(remote_addr));
    let start = std::time::Instant::now();

    let body_bytes = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &format!("{}", e));
        }
    };

    let anthropic_req: Value = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            error!("Invalid JSON: {}", e);
            return anthropic_error(StatusCode::BAD_REQUEST, "invalid_request_error", &format!("{}", e));
        }
    };

    let model = anthropic_req.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
    let is_stream = anthropic_req.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    info!("{} POST /v1/messages model={} stream={}", client_ip, model, is_stream);

    // Convert Anthropic → OpenAI
    let mut openai_body = crate::tools::anthropic_request_to_openai(&anthropic_req);

    // Tool adaptation for local model
    let has_tools = crate::tools::transform_request(&mut openai_body);
    if has_tools { info!("Anthropic tool adaptation applied"); }

    // Rewrite model
    let backend_model = if let Some(ref mlx_model) = state.config.mlx_model {
        mlx_model.canonicalize()
            .unwrap_or_else(|_| mlx_model.clone())
            .to_string_lossy()
            .to_string()
    } else {
        crate::openai::LOCAL_MODEL_ALIAS.to_string()
    };
    openai_body["model"] = json!(backend_model);

    // Truncate messages to fit ctx_size
    if let Some(messages) = openai_body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        crate::language::truncate_messages(messages, state.config.ctx_size as usize);
    }

    // Dynamic thinking control
    if let Some(messages) = openai_body.get("messages").and_then(|m| m.as_array()) {
        let needs_thinking = crate::language::estimate_complexity(messages);
        info!("anthropic thinking: needs={}", needs_thinking);
        if !needs_thinking {
            openai_body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
    }

    // Inject sampling defaults if not set
    if openai_body.get("repetition_penalty").is_none() && state.config.repetition_penalty > 1.0 {
        openai_body["repetition_penalty"] = json!(state.config.repetition_penalty);
        openai_body["repetition_context_size"] = json!(state.config.repetition_context_size);
    }

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", state.config.backend_port);

    if is_stream {
        openai_body["stream"] = json!(true);
        openai_body["stream_options"] = json!({"include_usage": true});
        let openai_bytes = serde_json::to_vec(&openai_body).unwrap();
        return stream_anthropic(state, url, openai_bytes, model, client_ip, start);
    }

    // Non-stream: send to backend, convert response
    openai_body["stream"] = json!(true);
    openai_body["stream_options"] = json!({"include_usage": true});
    let req_bytes = serde_json::to_vec(&openai_body).unwrap();

    let resp = match state.http_client.post(&url)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "text/event-stream")
        .body(req_bytes)
        .send().await
    {
        Ok(r) => r,
        Err(e) => {
            error!("Cannot connect to LM Studio: {}", e);
            return anthropic_error(StatusCode::BAD_GATEWAY, "api_error", &format!("Cannot connect to backend: {}", e));
        }
    };

    let resp_bytes = crate::stream::collect_stream_to_response(resp).await;

    // Apply tool call parsing
    let resp_bytes = if has_tools { crate::tools::transform_response(resp_bytes) } else { resp_bytes };

    let resp_body: Value = match serde_json::from_slice(&resp_bytes) {
        Ok(v) => v,
        Err(e) => {
            error!("Invalid response: {}", e);
            return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &format!("{}", e));
        }
    };

    info!("{} Anthropic non-stream completed in {:?}", client_ip, start.elapsed());
    Json(openai_to_anthropic(resp_body, &model)).into_response()
}

// ─── Streaming ──────────────────────────────────────────────────────────────

fn stream_anthropic(
    state: AppState,
    url: String,
    openai_bytes: Vec<u8>,
    model: String,
    client_ip: String,
    start: std::time::Instant,
) -> Response {
    let event_stream = async_stream::stream! {
        use futures_util::StreamExt;
        let ping = Bytes::from("event: ping\ndata: {\"type\": \"ping\"}\n\n");

        let mut stream_state = StreamState::default();
        let mut tool_parser = ToolCallStreamParser::new();

        yield Ok::<_, std::io::Error>(ping.clone());

        // Connect with ping keepalives during prompt processing
        let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(3));
        ping_interval.tick().await;

        let req_future = state.http_client.post(&url)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ACCEPT, "text/event-stream")
            .body(openai_bytes)
            .send();
        tokio::pin!(req_future);

        let resp = loop {
            tokio::select! {
                result = &mut req_future => {
                    match result {
                        Ok(r) => break r,
                        Err(e) => {
                            error!("Failed to connect to LM Studio: {}", e);
                            yield Ok::<_, std::io::Error>(Bytes::from(format!(
                                "event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":\"Backend error: {}\"}}}}\n\n", e
                            )));
                            return;
                        }
                    }
                }
                _ = ping_interval.tick() => {
                    yield Ok::<_, std::io::Error>(ping.clone());
                }
            }
        };

        let byte_stream = resp.bytes_stream();
        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut buffer = String::new();
        let mut stream_done = false;

        loop {
            if stream_done { break; }

            tokio::select! {
                chunk_opt = byte_stream.next() => {
                    match chunk_opt {
                        Some(Ok(chunk)) => buffer.push_str(&String::from_utf8_lossy(&chunk)),
                        Some(Err(e)) => { error!("Stream error: {}", e); stream_done = true; }
                        None => { stream_done = true; }
                    }
                }
                _ = ping_interval.tick() => {
                    yield Ok::<_, std::io::Error>(ping.clone());
                    continue;
                }
            }

            while let Some(pos) = buffer.find('\n') {
                let line = buffer.drain(..=pos).collect::<String>();
                let line = line.trim();
                if !line.starts_with("data: ") { continue; }
                let payload = &line[6..];
                if payload == "[DONE]" { continue; }

                if let Ok(chunk_json) = serde_json::from_str::<Value>(payload) {
                    for event in openai_chunk_to_anthropic(&chunk_json, &mut stream_state, &model, &mut tool_parser) {
                        yield Ok::<_, std::io::Error>(Bytes::from(format!(
                            "event: {}\ndata: {}\n\n",
                            event["type"].as_str().unwrap_or(""),
                            serde_json::to_string(&event).unwrap()
                        )));
                    }
                }
            }
        }

        // Emit deferred stop if stream ended without usage chunk
        if !stream_state.finished {
            if let Some(stop_reason) = stream_state.pending_stop_reason.take() {
                yield Ok::<_, std::io::Error>(Bytes::from(format!(
                    "event: message_delta\ndata: {}\n\n",
                    serde_json::to_string(&json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                        "usage": {"output_tokens": 0}
                    })).unwrap()
                )));
                yield Ok::<_, std::io::Error>(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
            }
        }

        info!("{} Anthropic stream completed in {:?}", client_ip, start.elapsed());
    };

    sse_response(Body::from_stream(event_stream))
}
