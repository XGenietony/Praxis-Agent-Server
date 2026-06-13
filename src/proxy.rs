use axum::http::{HeaderMap, StatusCode};
use crate::config::Config;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub http_client: reqwest::Client,
    pub rag: Option<Arc<crate::rag::RagClient>>,
}

pub fn get_client_ip(headers: &HeaderMap, remote_addr: Option<std::net::SocketAddr>) -> String {
    if let Some(forwarded) = headers.get("X-Forwarded-For") {
        if let Ok(s) = forwarded.to_str() {
            return s.split(',').next().unwrap_or("unknown").trim().to_string();
        }
    }
    remote_addr.map(|a| a.ip().to_string()).unwrap_or_else(|| "unknown".to_string())
}

pub fn check_api_key(headers: &HeaderMap, expected_key: &str) -> Result<(), StatusCode> {
    if expected_key.is_empty() {
        return Ok(());
    }
    if let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(s) = auth.to_str() {
            if s == format!("Bearer {}", expected_key) {
                return Ok(());
            }
        }
    }
    if let Some(key) = headers.get("x-api-key") {
        if let Ok(s) = key.to_str() {
            if s == expected_key {
                return Ok(());
            }
        }
    }
    Err(StatusCode::UNAUTHORIZED)
}
