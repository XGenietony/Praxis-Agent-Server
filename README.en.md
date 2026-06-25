# lmstudio-forward

Language: [中文](README.md) | English

`lmstudio-forward` is a lightweight Go proxy that exposes local LLM backends through unified OpenAI-compatible and Anthropic-compatible APIs. It can start `llama-server` or `mlx_lm.server` for you, forward to an already running OpenAI-compatible backend, and optionally enable an frpc public tunnel plus Qdrant-backed Agentic RAG.

## Features

- Starts a GGUF / MLX local inference backend automatically, or reuses an external backend.
- Exposes OpenAI-compatible `/v1/*` endpoints.
- Exposes an Anthropic-compatible `/v1/messages` endpoint with protocol conversion.
- Adapts tool calls for local models by parsing `<tool_call>` text into standard tool call structures.
- Supports SSE response forwarding and Anthropic event stream conversion.
- Injects default sampling parameters, truncates context, and applies lightweight thinking control.
- Optionally enables Agentic RAG: the model can call the built-in `retrieve` tool, the server searches Qdrant, and retrieved context is fed back into the conversation.
- Documents the integration boundary for ReAct and CodeAct agent methods.
- Optionally starts frpc to expose the local service through a public tunnel.

## Requirements

- Go 1.23+.
- One available LLM backend:
  - GGUF: install `llama-server`, for example with `brew install llama.cpp`.
  - MLX: prepare a Python environment with `mlx-lm` installed.
  - External backend: an already running OpenAI-compatible service, defaulting to `http://127.0.0.1:1234`.
- Optional: frpc for public tunneling.
- Optional: Qdrant plus an OpenAI-compatible embeddings service for Agentic RAG.

The Go service has no third-party module dependencies and builds with the standard library only.

## Build And Test

```bash
go build -o lmstudio-forward ./cmd/lmstudio-forward
go test ./...
```

The binary is written to `./lmstudio-forward` at the repository root.

## Quick Start

Copy the example script, then fill in your local model path:

```bash
cp start.example.sh start.sh
chmod +x start.sh
./start.sh
```

By default, the proxy listens on:

```text
http://0.0.0.0:8000
```

### GGUF Backend

```bash
./lmstudio-forward \
  --gguf-model /path/to/model.gguf \
  --ctx-size 32768 \
  --no-frpc
```

### MLX Backend

```bash
./lmstudio-forward \
  --mlx-model /path/to/mlx-model-dir \
  --python-path .venv/bin/python3 \
  --ctx-size 32768 \
  --temperature 0.7 \
  --no-frpc
```

### External Backend

If neither `--gguf-model` nor `--mlx-model` is provided, the service tries to reuse the external backend on `--backend-port`:

```bash
./lmstudio-forward \
  --backend-port 1234 \
  --server-port 8000 \
  --no-frpc
```

## API Examples

### OpenAI Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "Hello, briefly introduce yourself."}
    ],
    "stream": true
  }'
```

### Anthropic Messages

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Summarize what this service does."}
    ]
  }'
```

If `--api-key` / `API_KEY` is set, requests must include either of these authentication headers:

```bash
Authorization: Bearer <API_KEY>
x-api-key: <API_KEY>
```

## Agent Methods

This project is closer to an Agent Runtime Gateway / LLM protocol gateway: it handles protocol conversion, streaming semantics reconstruction, tool-call adaptation, RAG context injection, and backend governance. ReAct and CodeAct can both be used as upper-layer agent methods, but their boundaries inside this repository are different.

### ReAct

ReAct means a Reason + Act loop: the model reasons about the task, decides whether to call a tool, receives an observation, then continues reasoning and produces the final answer. In this project, ReAct is mainly represented through the tool-call protocol:

```text
User question -> model reasoning -> emits <tool_call> -> proxy parses tool call
              -> execute retrieve / external tool -> feed back observation
              -> model continues reasoning -> final answer
```

The built-in Agentic RAG path is a ReAct-style special case: Anthropic Messages requests receive a `retrieve` tool, the model emits a retrieval action when it needs knowledge-base context, and the proxy internally searches Qdrant before appending the results back into the conversation.

### CodeAct

CodeAct means the model expresses an action as code or a structured command, an external runner executes it, and the execution result is returned to the model. It is useful for file processing, data analysis, automation scripts, repository search, and batch operations.

This repository does not currently include an arbitrary code execution sandbox, and it should not directly execute model-generated code. The recommended boundary is:

```text
Upper-layer agent generates code/command -> external safe runner executes it
                                      -> stdout/stderr/result becomes tool result
                                      -> this proxy handles protocol conversion,
                                         streaming, and context transport
```

To integrate CodeAct, put code execution, permission control, timeouts, filesystem isolation, and audit logging in a separate tool service, then expose that service to the model through a standard tool schema. This proxy should focus on reliably passing tool calls and tool results between OpenAI and Anthropic protocols.

## Agentic RAG

When RAG is enabled, Anthropic Messages requests automatically receive the built-in `retrieve` tool. If the model decides it needs knowledge-base context, it emits a `retrieve` tool call; the proxy internally calls the embeddings service and Qdrant, appends the matched passages back into the conversation, and lets the model continue answering. This retrieval loop is invisible to the client.

`retrieve` is an internal tool and is never forwarded as a client-visible `tool_use`. If the model emits both `retrieve` and external tools in one turn, the proxy consumes retrieval first and asks the model to decide again with the new context. Streaming RAG requests reuse the final answer already produced by the internal loop and convert it to Anthropic SSE, so the backend is not asked to generate the answer a second time.

RAG currently runs only on Anthropic Messages routes: `/v1/messages`, `/v1/message`, `/anthropic`, and `/anthropic/v1/messages`. The OpenAI `/v1/chat/completions` route is currently transparent forwarding and does not execute the internal retrieve loop.

### Enable RAG

```bash
./lmstudio-forward \
  --backend-port 1234 \
  --server-port 8000 \
  --rag-enabled \
  --qdrant-url http://127.0.0.1:6333 \
  --qdrant-collection praxis_rag \
  --embed-url http://127.0.0.1:1234 \
  --embed-model text-embedding \
  --embed-dim 1024 \
  --no-frpc
```

Note: `--embed-dim` must match the actual output dimension of the embeddings model, otherwise the Qdrant collection vector size will not match.

### Ingest Documents

```bash
curl http://127.0.0.1:8000/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {
        "source": "project-notes.md",
        "text": "This is a document passage that should enter the knowledge base."
      }
    ]
  }'
```

Example response:

```json
{"ingested_chunks":1}
```

## Configuration

All options can be set through CLI flags or environment variables. CLI flags take precedence over environment variables.

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--gguf-model` | `GGUF_MODEL` | empty | Path to a GGUF model; enables the `llama-server` backend |
| `--mlx-model` | `MLX_MODEL` | empty | Path to an MLX model directory; enables the `mlx_lm.server` backend |
| `--python-path` | `PYTHON_PATH` | `python3` | Python executable used to start the MLX backend |
| `--llama-server` | `LLAMA_SERVER` | `/opt/homebrew/bin/llama-server` | Path to the `llama-server` executable |
| `--backend-port` | `BACKEND_PORT` | `1234` | Local model backend port |
| `--server-port` | `SERVER_PORT` | `8000` | Proxy listening port |
| `--api-key` | `API_KEY` | empty | API authentication key; empty disables authentication |
| `--ctx-size` | `CTX_SIZE` | `8192` | Context window size |
| `--temperature` | `TEMPERATURE` | `0.7` | Default temperature |
| `--top-p` | `TOP_P` | `0.9` | Default nucleus sampling parameter |
| `--min-p` | `MIN_P` | `0.05` | Default min-p parameter |
| `--top-k` | `TOP_K` | `0` | Default top-k; `0` disables it |
| `--repetition-penalty` | `REPETITION_PENALTY` | `1.3` | Default repetition penalty |
| `--repetition-context-size` | `REPETITION_CONTEXT_SIZE` | `256` | Repetition penalty window |
| `--max-tokens` | `MAX_TOKENS` | `16384` | Maximum output tokens for the MLX backend |
| `--prefill-step-size` | `PREFILL_STEP_SIZE` | `4096` | MLX prompt prefill batch size |
| `--no-frpc` | `NO_FRPC` | `false` | Disable the frpc public tunnel |
| `--frpc-path` | `FRPC_PATH` | `./frp_0.68.0_darwin_arm64/frpc` | Path to the frpc executable |
| `--frpc-config` | `FRPC_CONFIG` | `./frp_0.68.0_darwin_arm64/frpc.toml` | Path to the frpc config file |
| `--rag-enabled` | `RAG_ENABLED` | `false` | Enable Agentic RAG |
| `--qdrant-url` | `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant REST URL |
| `--qdrant-collection` | `QDRANT_COLLECTION` | `praxis_rag` | Qdrant collection name |
| `--embed-url` | `EMBED_URL` | `http://127.0.0.1:1234` | OpenAI-compatible embeddings service URL |
| `--embed-model` | `EMBED_MODEL` | `text-embedding` | Embeddings model name |
| `--embed-dim` | `EMBED_DIM` | `1024` | Embedding vector dimension |
| `--rag-top-k` | `RAG_TOP_K` | `5` | Number of chunks returned per retrieval |
| `--rag-max-rounds` | `RAG_MAX_ROUNDS` | `3` | Maximum internal retrieval rounds per request |
| `--rag-step-timeout-seconds` | `RAG_STEP_TIMEOUT_SECONDS` | `120` | Timeout in seconds for each internal RAG backend, embedding, or Qdrant retrieval step |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | `800` | Chunk size in characters when ingesting documents |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | `100` | Chunk overlap in characters |

## API Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check; probes whether the backend is reachable |
| `GET` | `/v1/models` | Returns the externally advertised model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `GET/POST` | `/v1/*` | OpenAI-compatible passthrough |
| `POST` | `/v1/messages` | Anthropic Messages API |
| `POST` | `/v1/message` | Anthropic Messages API compatibility route |
| `POST` | `/anthropic` | Anthropic Messages API compatibility route |
| `POST` | `/anthropic/v1/messages` | Anthropic Messages API compatibility route |
| `POST` | `/rag/ingest` | RAG document ingestion |

## Project Layout

```text
cmd/lmstudio-forward/   Application entrypoint: config parsing, dependency wiring, server startup
internal/
  agentloop/   Internal Agent loop: retrieve action detection, observation feedback, stop policy
  config/      Config parsing: flags, environment variables, defaults
  jsonx/       Dynamic JSON helpers
  language/    Token estimation, context truncation, complexity detection
  tools/       Tool call injection, parsing, JSON repair, protocol adaptation
  rag/         Agentic RAG: Qdrant client, chunking, retrieve tool
  stream/      SSE collection and response headers
  proxy/       Shared app state, client IP helpers, API key checks
  openai/      OpenAI-compatible forwarding handler
  anthropic/   Anthropic Messages protocol and streaming event conversion
  process/     llama-server, mlx_lm.server, and frpc process management
  server/      HTTP routes, health checks, and RAG ingest endpoint
```

## Development Notes

Common checks:

```bash
go test ./...
go build -o lmstudio-forward ./cmd/lmstudio-forward
```

For local development and debugging, prefer `--no-frpc` to avoid starting the public tunnel. When debugging RAG, first confirm that Qdrant is reachable, the embeddings endpoint works, and the embedding dimension matches the Qdrant collection configuration.
