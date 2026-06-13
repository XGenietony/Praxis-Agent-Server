mod config;
mod proxy;
mod process;
mod openai;
mod anthropic;
mod stream;
mod tools;
mod language;
mod rag;

use axum::{Router, routing::{get, post}, response::IntoResponse, extract::State};
use clap::Parser;
use proxy::AppState;
use tracing::info;

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let backend_ok = reqwest::get(format!("http://127.0.0.1:{}/health", state.config.backend_port))
        .await.is_ok();
    axum::Json(serde_json::json!({
        "status": if backend_ok { "ok" } else { "degraded" },
        "backend": backend_ok,
    }))
}

#[derive(serde::Deserialize)]
struct IngestRequest {
    documents: Vec<rag::IngestDoc>,
}

async fn rag_ingest(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    axum::Json(body): axum::Json<IngestRequest>,
) -> axum::response::Response {
    use axum::http::StatusCode;

    if proxy::check_api_key(&headers, &state.config.api_key).is_err() {
        return (StatusCode::UNAUTHORIZED, "Invalid API key").into_response();
    }
    let Some(rag) = state.rag.clone() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "RAG is not enabled").into_response();
    };
    match rag.ingest(body.documents).await {
        Ok(n) => axum::Json(serde_json::json!({ "ingested_chunks": n })).into_response(),
        Err(e) => {
            tracing::error!("RAG ingest failed: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("ingest failed: {}", e)).into_response()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = config::Config::parse();
    info!("LMStudio Forward v{}", env!("CARGO_PKG_VERSION"));

    let mut pm = process::ProcessManager::new();
    if let Err(e) = pm.start(&config).await {
        tracing::error!("Failed to start: {}", e);
        return Err(e);
    }

    let http_client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .tcp_nodelay(true)
        .no_proxy()
        .build()?;

    let rag = if config.rag_enabled {
        let client = rag::RagClient::new(http_client.clone(), &config);
        if let Err(e) = client.ensure_collection().await {
            tracing::error!("RAG: failed to ensure Qdrant collection: {}", e);
            return Err(e);
        }
        info!(
            "RAG enabled: qdrant={} collection={} embed={} dim={}",
            config.qdrant_url, config.qdrant_collection, config.embed_url, config.embed_dim
        );
        Some(std::sync::Arc::new(client))
    } else {
        None
    };

    let state = AppState {
        config: config.clone(),
        http_client,
        rag,
    };

    let app = Router::new()
        .route("/v1/messages", post(anthropic::forward_anthropic_messages))
        .route("/v1/message", post(anthropic::forward_anthropic_messages))
        .route("/anthropic", post(anthropic::forward_anthropic_messages))
        .route("/anthropic/v1/messages", post(anthropic::forward_anthropic_messages))
        .route("/rag/ingest", post(rag_ingest))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/*path", get(openai::forward_openai).post(openai::forward_openai))
        .route("/health", get(health))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], config.server_port));
    info!("Listening on http://{}", addr);
    info!("Backend: http://127.0.0.1:{}", config.backend_port);
    if !config.no_frpc {
        info!("Public: https://opus.northsea.chat");
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .tcp_nodelay(true)
        .await?;

    Ok(())
}
