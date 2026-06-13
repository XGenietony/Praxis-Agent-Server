use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser, Clone)]
#[command(name = "lmstudio-forward", version, about = "Forward LM Studio to public endpoint via frpc")]
pub struct Config {
    /// LM Studio backend port
    #[arg(long, env = "BACKEND_PORT", default_value_t = 1234)]
    pub backend_port: u16,

    /// Forwarding server listening port
    #[arg(long, env = "SERVER_PORT", default_value_t = 8000)]
    pub server_port: u16,

    /// API key for authentication (empty = no auth)
    #[arg(long, env = "API_KEY", default_value = "")]
    pub api_key: String,

    /// Path to frpc executable
    #[arg(long, env = "FRPC_PATH", default_value = "./frp_0.68.0_darwin_arm64/frpc")]
    pub frpc_path: PathBuf,

    /// Path to frpc config file
    #[arg(long, env = "FRPC_CONFIG", default_value = "./frp_0.68.0_darwin_arm64/frpc.toml")]
    pub frpc_config: PathBuf,

    /// Disable frpc tunnel
    #[arg(long, env = "NO_FRPC", default_value_t = false)]
    pub no_frpc: bool,

    /// Path to MLX model directory (enables built-in mlx_lm.server)
    #[arg(long, env = "MLX_MODEL")]
    pub mlx_model: Option<PathBuf>,

    /// Path to Python executable (venv) for mlx_lm.server
    #[arg(long, env = "PYTHON_PATH", default_value = "python3")]
    pub python_path: PathBuf,

    /// Path to GGUF model file (enables built-in llama-server)
    #[arg(long, env = "GGUF_MODEL")]
    pub gguf_model: Option<PathBuf>,

    /// Path to llama-server executable
    #[arg(long, env = "LLAMA_SERVER", default_value = "/opt/homebrew/bin/llama-server")]
    pub llama_server: PathBuf,

    /// Context size (max input+output tokens, also controls KV cache)
    #[arg(long, env = "CTX_SIZE", default_value_t = 8192)]
    pub ctx_size: u32,

    // ── Sampling parameters (defaults injected when client doesn't specify) ──

    /// Default temperature (0.0 = greedy, higher = more random)
    #[arg(long, env = "TEMPERATURE", default_value_t = 0.7)]
    pub temperature: f32,

    /// Default top-p nucleus sampling
    #[arg(long, env = "TOP_P", default_value_t = 0.9)]
    pub top_p: f32,

    /// Default min-p sampling threshold
    #[arg(long, env = "MIN_P", default_value_t = 0.05)]
    pub min_p: f32,

    /// Default top-k sampling (0 = disabled)
    #[arg(long, env = "TOP_K", default_value_t = 0)]
    pub top_k: u32,

    /// Default repetition penalty (1.0 = disabled, >1.0 = penalize repeats)
    #[arg(long, env = "REPETITION_PENALTY", default_value_t = 1.3)]
    pub repetition_penalty: f32,

    /// Repetition penalty context window size (tokens)
    #[arg(long, env = "REPETITION_CONTEXT_SIZE", default_value_t = 256)]
    pub repetition_context_size: u32,

    /// Default max output tokens
    #[arg(long, env = "MAX_TOKENS", default_value_t = 16384)]
    pub max_tokens: u32,

    /// Prefill step size for mlx_lm.server prompt processing
    #[arg(long, env = "PREFILL_STEP_SIZE", default_value_t = 4096)]
    pub prefill_step_size: u32,

    // ── Agentic RAG (disabled by default; enabling adds an internal `retrieve` tool) ──

    /// Enable Agentic RAG (proxy intercepts a built-in `retrieve` tool and runs retrieval internally)
    #[arg(long, env = "RAG_ENABLED", default_value_t = false)]
    pub rag_enabled: bool,

    /// Qdrant base URL (REST API)
    #[arg(long, env = "QDRANT_URL", default_value = "http://127.0.0.1:6333")]
    pub qdrant_url: String,

    /// Qdrant collection name
    #[arg(long, env = "QDRANT_COLLECTION", default_value = "praxis_rag")]
    pub qdrant_collection: String,

    /// Embedding service base URL (OpenAI-compatible `/v1/embeddings`)
    #[arg(long, env = "EMBED_URL", default_value = "http://127.0.0.1:1234")]
    pub embed_url: String,

    /// Embedding model name passed to the embedding service
    #[arg(long, env = "EMBED_MODEL", default_value = "text-embedding")]
    pub embed_model: String,

    /// Embedding vector dimension (must match the embedding model's output)
    #[arg(long, env = "EMBED_DIM", default_value_t = 1024)]
    pub embed_dim: u64,

    /// Number of chunks returned per retrieval
    #[arg(long, env = "RAG_TOP_K", default_value_t = 5)]
    pub rag_top_k: u64,

    /// Max internal retrieval rounds per request (loop guard)
    #[arg(long, env = "RAG_MAX_ROUNDS", default_value_t = 3)]
    pub rag_max_rounds: u32,

    /// Chunk size in characters when ingesting documents
    #[arg(long, env = "RAG_CHUNK_SIZE", default_value_t = 800)]
    pub rag_chunk_size: usize,

    /// Chunk overlap in characters when ingesting documents
    #[arg(long, env = "RAG_CHUNK_OVERLAP", default_value_t = 100)]
    pub rag_chunk_overlap: usize,
}
