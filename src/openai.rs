//! OpenAI-compatible forwarding handler.
//! All model names are accepted — requests go straight to backend.

use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::json;
use tracing::{info, error};
use crate::proxy::{AppState, check_api_key, get_client_ip};
use crate::language::estimate_complexity;

/// Model alias for non-MLX backends (llama-server doesn't care about model name).
pub const LOCAL_MODEL_ALIAS: &str = "gemma4";

pub async fn forward_openai(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<std::net::SocketAddr>,
    req: Request<Body>,
) -> impl IntoResponse {
    if let Err(status) = check_api_key(&headers, &state.config.api_key) {
        return (status, "Invalid API key").into_response();
    }

    let client_ip = get_client_ip(&headers, Some(remote_addr));
    let path = req.uri().path().strip_prefix("/v1/").unwrap_or("").to_string();
    let method = req.method().clone();

    info!("{} {} /v1/{}", client_ip, method, path);

    let body_bytes = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return (StatusCode::BAD_REQUEST, "Failed to read request body").into_response();
        }
    };

    // Rewrite model name + inject dynamic thinking control
    let request_body = if !body_bytes.is_empty() {
        if let Ok(mut data) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
            // Rewrite model name
            if data.get("model").is_some() {
                let backend_model = if let Some(ref mlx_model) = state.config.mlx_model {
                    mlx_model.canonicalize()
                        .unwrap_or_else(|_| mlx_model.clone())
                        .to_string_lossy()
                        .to_string()
                } else {
                    LOCAL_MODEL_ALIAS.to_string()
                };
                data["model"] = json!(backend_model);
            }

            // Dynamic thinking control for chat completions
            if path.starts_with("chat/completions") {
                // Truncate messages to fit ctx_size
                if let Some(messages) = data.get_mut("messages").and_then(|m| m.as_array_mut()) {
                    crate::language::truncate_messages(messages, state.config.ctx_size as usize);
                }

                if let Some(messages) = data.get("messages").and_then(|m| m.as_array()) {
                    let needs_thinking = estimate_complexity(messages);
                    info!("thinking: needs={}", needs_thinking);

                    if !needs_thinking {
                        // Inject chat_template_kwargs to disable thinking
                        data["chat_template_kwargs"] = json!({"enable_thinking": false});
                    }
                }

                // Inject sampling defaults if not set by client
                if data.get("repetition_penalty").is_none() && state.config.repetition_penalty > 1.0 {
                    data["repetition_penalty"] = json!(state.config.repetition_penalty);
                    data["repetition_context_size"] = json!(state.config.repetition_context_size);
                }
            }

            bytes::Bytes::from(serde_json::to_vec(&data).unwrap())
        } else {
            body_bytes
        }
    } else {
        body_bytes
    };

    // Forward to backend
    let url = format!("http://127.0.0.1:{}/v1/{}", state.config.backend_port, path);
    let mut rb = state.http_client.request(method.clone(), &url);
    for (key, value) in headers.iter() {
        if key != header::HOST && key != header::CONTENT_LENGTH {
            rb = rb.header(key, value);
        }
    }

    let resp = match rb.body(request_body).send().await {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to connect to backend: {}", e);
            return (StatusCode::BAD_GATEWAY, format!("Cannot connect to backend: {}", e)).into_response();
        }
    };

    // Stream the response back transparently
    let status = resp.status();
    let resp_headers = resp.headers().clone();
    let body = Body::from_stream(resp.bytes_stream());

    let mut builder = Response::builder().status(status);
    for (key, value) in resp_headers.iter() {
        if key != header::CONTENT_LENGTH && key != header::TRANSFER_ENCODING {
            if let Ok(name) = axum::http::header::HeaderName::from_bytes(key.as_ref()) {
                builder = builder.header(name, value);
            }
        }
    }
    builder.body(body).unwrap().into_response()
}

/// List models — returns Claude model names.
pub async fn list_models(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    if let Err(status) = check_api_key(&headers, &state.config.api_key) {
        return (status, "Invalid API key").into_response();
    }

    let models = json!({
        "object": "list",
        "data": [
            {"id": "claude-opus-4-6", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-haiku-4-5-20251001", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-5-20250514", "object": "model", "owned_by": "anthropic"},
        ]
    });

    (StatusCode::OK, axum::Json(models)).into_response()
}
