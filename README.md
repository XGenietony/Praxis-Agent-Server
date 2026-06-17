# lmstudio-forward

A lightweight Go proxy that exposes a local LLM (GGUF via llama-server, or MLX via mlx_lm.server) as a unified API endpoint, with optional public tunnel via frpc.

## What it does

- Starts your chosen backend (llama-server or mlx_lm.server) automatically
- Exposes OpenAI-compatible `/v1/*` and Anthropic-compatible `/v1/messages` endpoints
- Injects default sampling parameters when the client doesn't specify them
- Optionally tunnels the service to a public URL via frpc

## Requirements

- Go 1.23+ (for building; no third-party dependencies — stdlib only)
- One of:
  - **GGUF**: [llama-server](https://github.com/ggerganov/llama.cpp) (`brew install llama.cpp`)
  - **MLX**: Python venv with `mlx-lm` installed
- (Optional) [frpc](https://github.com/fatedier/frp) for public tunnel

## Build

```bash
go build -o lmstudio-forward ./cmd/lmstudio-forward
```

The binary is output to `./lmstudio-forward`.

Run the tests with:

```bash
go test ./...
```

## Project layout

Standard Go layout — `cmd/` holds the entrypoint (bootstrap + dependency
injection only); all behavior lives in `internal/`:

```
cmd/lmstudio-forward/   application entrypoint (wiring only)
internal/
  config/      configuration: flags, env, defaults
  jsonx/       dynamic-JSON helper layer (serde_json::Value equivalent)
  language/    token estimation, context truncation, complexity scoring
  tools/       tool-call adaptation + <tool_call> parsing (batch & streaming)
  rag/         Agentic RAG: Qdrant client, chunking, retrieve tool
  stream/      SSE collection + response headers
  proxy/       shared app state, client-IP + API-key helpers
  openai/      OpenAI-compatible forwarding handler
  anthropic/   Anthropic Messages handler + protocol conversion
  process/     backend (llama-server/mlx) + frpc process supervision
  server/      route wiring, health + RAG-ingest handlers
```

## Quick start

```bash
cp start.example.sh start.sh
chmod +x start.sh
# Edit start.sh — set your model path
./start.sh
```

The server listens on `http://0.0.0.0:8000` by default.

## Configuration

All options can be set via CLI flags or environment variables.

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--gguf-model` | `GGUF_MODEL` | — | Path to `.gguf` file (enables llama-server backend) |
| `--mlx-model` | `MLX_MODEL` | — | Path to MLX model directory (enables mlx_lm.server backend) |
| `--python-path` | `PYTHON_PATH` | `python3` | Python executable for mlx_lm.server |
| `--llama-server` | `LLAMA_SERVER` | `/opt/homebrew/bin/llama-server` | Path to llama-server binary |
| `--backend-port` | `BACKEND_PORT` | `1234` | Port the backend listens on |
| `--server-port` | `SERVER_PORT` | `8000` | Port this proxy listens on |
| `--api-key` | `API_KEY` | _(none)_ | Bearer token for auth (empty = no auth) |
| `--ctx-size` | `CTX_SIZE` | `8192` | Max context window (tokens) |
| `--temperature` | `TEMPERATURE` | `0.7` | Sampling temperature |
| `--top-p` | `TOP_P` | `0.9` | Nucleus sampling |
| `--min-p` | `MIN_P` | `0.05` | Min-p filtering |
| `--top-k` | `TOP_K` | `0` | Top-K (0 = disabled) |
| `--repetition-penalty` | `REPETITION_PENALTY` | `1.3` | Repeat penalty |
| `--repetition-context-size` | `REPETITION_CONTEXT_SIZE` | `256` | Repeat penalty window (tokens) |
| `--max-tokens` | `MAX_TOKENS` | `16384` | Max output tokens |
| `--prefill-step-size` | `PREFILL_STEP_SIZE` | `4096` | MLX prompt processing batch size |
| `--no-frpc` | `NO_FRPC` | `false` | Disable frpc public tunnel |
| `--frpc-path` | `FRPC_PATH` | `./frp_.../frpc` | Path to frpc binary |
| `--frpc-config` | `FRPC_CONFIG` | `./frp_.../frpc.toml` | Path to frpc config |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `GET/POST` | `/v1/*` | OpenAI API passthrough |
| `POST` | `/v1/messages` | Anthropic Messages API |
| `POST` | `/anthropic/v1/messages` | Anthropic Messages API (alternate path) |

All endpoints support SSE streaming.

## Nginx reverse proxy

An example Nginx config for SSL termination and reverse proxying is provided in `nginx_e4b.conf`.
