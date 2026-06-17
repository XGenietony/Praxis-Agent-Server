// Package config holds all runtime configuration for the proxy server and the
// logic to resolve it from defaults, environment variables, and CLI flags.
package config

import (
	"flag"
	"os"
	"path/filepath"
	"strconv"
)

// LocalModelAlias is the model alias reported to non-MLX backends (llama-server
// does not care about the model name). It lives here so both the config layer
// (Config.BackendModel) and the OpenAI handler can reference it without an
// import cycle.
const LocalModelAlias = "gemma4"

// Config mirrors the Rust `Config` struct (clap-derived) in src/config.rs.
// Values are resolved with the precedence:
//
//	hard-coded default < environment variable < CLI flag.
type Config struct {
	BackendPort           int
	ServerPort            int
	APIKey                string
	FrpcPath              string
	FrpcConfig            string
	NoFrpc                bool
	MlxModel              string // "" = not set
	PythonPath            string
	GgufModel             string // "" = not set
	LlamaServer           string
	CtxSize               int
	Temperature           float64
	TopP                  float64
	MinP                  float64
	TopK                  int
	RepetitionPenalty     float64
	RepetitionContextSize int
	MaxTokens             int
	PrefillStepSize       int
	RagEnabled            bool
	QdrantURL             string
	QdrantCollection      string
	EmbedURL              string
	EmbedModel            string
	EmbedDim              int
	RagTopK               int
	RagMaxRounds          int
	RagChunkSize          int
	RagChunkOverlap       int
}

// envOr returns the value of environment variable `key` if set (even if empty,
// to mirror clap's `env` behaviour where presence wins), otherwise `def`.
func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

// envInt resolves an int default from environment variable `key`, falling back
// to `def` when the variable is unset or unparsable.
func envInt(key string, def int) int {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// envFloat resolves a float64 default from environment variable `key`, falling
// back to `def` when the variable is unset or unparsable.
func envFloat(key string, def float64) float64 {
	if v, ok := os.LookupEnv(key); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// envBool resolves a bool default from environment variable `key`, falling back
// to `def` when the variable is unset or unparsable.
func envBool(key string, def bool) bool {
	if v, ok := os.LookupEnv(key); ok {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

// Parse builds a Config from defaults, environment variables, and CLI flags (in
// increasing order of precedence). Env names and defaults match the clap
// derivation in src/config.rs exactly; flags use kebab-case.
func Parse() Config {
	var c Config

	flag.IntVar(&c.BackendPort, "backend-port", envInt("BACKEND_PORT", 1234), "LM Studio backend port")
	flag.IntVar(&c.ServerPort, "server-port", envInt("SERVER_PORT", 8000), "Forwarding server listening port")
	flag.StringVar(&c.APIKey, "api-key", envOr("API_KEY", ""), "API key for authentication (empty = no auth)")
	flag.StringVar(&c.FrpcPath, "frpc-path", envOr("FRPC_PATH", "./frp_0.68.0_darwin_arm64/frpc"), "Path to frpc executable")
	flag.StringVar(&c.FrpcConfig, "frpc-config", envOr("FRPC_CONFIG", "./frp_0.68.0_darwin_arm64/frpc.toml"), "Path to frpc config file")
	flag.BoolVar(&c.NoFrpc, "no-frpc", envBool("NO_FRPC", false), "Disable frpc tunnel")
	flag.StringVar(&c.MlxModel, "mlx-model", envOr("MLX_MODEL", ""), "Path to MLX model directory (enables built-in mlx_lm.server)")
	flag.StringVar(&c.PythonPath, "python-path", envOr("PYTHON_PATH", "python3"), "Path to Python executable (venv) for mlx_lm.server")
	flag.StringVar(&c.GgufModel, "gguf-model", envOr("GGUF_MODEL", ""), "Path to GGUF model file (enables built-in llama-server)")
	flag.StringVar(&c.LlamaServer, "llama-server", envOr("LLAMA_SERVER", "/opt/homebrew/bin/llama-server"), "Path to llama-server executable")
	flag.IntVar(&c.CtxSize, "ctx-size", envInt("CTX_SIZE", 8192), "Context size (max input+output tokens, also controls KV cache)")
	flag.Float64Var(&c.Temperature, "temperature", envFloat("TEMPERATURE", 0.7), "Default temperature (0.0 = greedy, higher = more random)")
	flag.Float64Var(&c.TopP, "top-p", envFloat("TOP_P", 0.9), "Default top-p nucleus sampling")
	flag.Float64Var(&c.MinP, "min-p", envFloat("MIN_P", 0.05), "Default min-p sampling threshold")
	flag.IntVar(&c.TopK, "top-k", envInt("TOP_K", 0), "Default top-k sampling (0 = disabled)")
	flag.Float64Var(&c.RepetitionPenalty, "repetition-penalty", envFloat("REPETITION_PENALTY", 1.3), "Default repetition penalty (1.0 = disabled, >1.0 = penalize repeats)")
	flag.IntVar(&c.RepetitionContextSize, "repetition-context-size", envInt("REPETITION_CONTEXT_SIZE", 256), "Repetition penalty context window size (tokens)")
	flag.IntVar(&c.MaxTokens, "max-tokens", envInt("MAX_TOKENS", 16384), "Default max output tokens")
	flag.IntVar(&c.PrefillStepSize, "prefill-step-size", envInt("PREFILL_STEP_SIZE", 4096), "Prefill step size for mlx_lm.server prompt processing")
	flag.BoolVar(&c.RagEnabled, "rag-enabled", envBool("RAG_ENABLED", false), "Enable Agentic RAG (proxy intercepts a built-in `retrieve` tool and runs retrieval internally)")
	flag.StringVar(&c.QdrantURL, "qdrant-url", envOr("QDRANT_URL", "http://127.0.0.1:6333"), "Qdrant base URL (REST API)")
	flag.StringVar(&c.QdrantCollection, "qdrant-collection", envOr("QDRANT_COLLECTION", "praxis_rag"), "Qdrant collection name")
	flag.StringVar(&c.EmbedURL, "embed-url", envOr("EMBED_URL", "http://127.0.0.1:1234"), "Embedding service base URL (OpenAI-compatible /v1/embeddings)")
	flag.StringVar(&c.EmbedModel, "embed-model", envOr("EMBED_MODEL", "text-embedding"), "Embedding model name passed to the embedding service")
	flag.IntVar(&c.EmbedDim, "embed-dim", envInt("EMBED_DIM", 1024), "Embedding vector dimension (must match the embedding model's output)")
	flag.IntVar(&c.RagTopK, "rag-top-k", envInt("RAG_TOP_K", 5), "Number of chunks returned per retrieval")
	flag.IntVar(&c.RagMaxRounds, "rag-max-rounds", envInt("RAG_MAX_ROUNDS", 3), "Max internal retrieval rounds per request (loop guard)")
	flag.IntVar(&c.RagChunkSize, "rag-chunk-size", envInt("RAG_CHUNK_SIZE", 800), "Chunk size in characters when ingesting documents")
	flag.IntVar(&c.RagChunkOverlap, "rag-chunk-overlap", envInt("RAG_CHUNK_OVERLAP", 100), "Chunk overlap in characters when ingesting documents")

	flag.Parse()

	return c
}

// canonicalizePath resolves `p` to an absolute, symlink-free path. On any error
// it falls back to returning the original input unchanged.
func canonicalizePath(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return abs
	}
	return resolved
}

// CanonicalizePath is exported for reuse by the process layer.
func CanonicalizePath(p string) string { return canonicalizePath(p) }

// BackendModel returns the model identifier reported to the backend. When an
// MLX model directory is configured it returns its canonicalized path;
// otherwise it returns the local model alias.
func (c *Config) BackendModel() string {
	if c.MlxModel != "" {
		return canonicalizePath(c.MlxModel)
	}
	return LocalModelAlias
}
