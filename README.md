# lmstudio-forward

语言版本：中文 | [English](README.en.md)

`lmstudio-forward` 是一个轻量级 Go 代理服务，用来把本地大模型后端统一包装成 OpenAI-compatible 和 Anthropic-compatible API。它可以自动拉起 `llama-server` 或 `mlx_lm.server`，也可以转发到已经运行的外部 OpenAI-compatible 后端；可选开启 frpc 公网隧道和基于 Qdrant 的 Agentic RAG 检索。

## 功能概览

- 自动启动 GGUF / MLX 本地推理后端，或复用已有后端。
- 暴露 OpenAI-compatible `/v1/*` 接口。
- 暴露 Anthropic-compatible `/v1/messages` 接口，并完成协议转换。
- 对本地模型适配工具调用，把 `<tool_call>` 文本解析成标准 tool call 结构。
- 支持 SSE 流式响应转发与 Anthropic 事件流转换。
- 自动注入默认采样参数、上下文截断和简单的 thinking 控制。
- 可选启用 Agentic RAG：模型主动调用内置 `retrieve` 工具，服务端检索 Qdrant 后把结果回填上下文。
- 可选启动 frpc，将本地服务暴露为公网访问入口。

## 环境要求

- Go 1.23+。
- 一个可用的大模型后端，任选其一：
  - GGUF：安装 `llama-server`，例如 `brew install llama.cpp`。
  - MLX：准备 Python 环境并安装 `mlx-lm`。
  - 外部后端：已经运行的 OpenAI-compatible 服务，默认地址为 `http://127.0.0.1:1234`。
- 可选：frpc，用于公网隧道。
- 可选：Qdrant + OpenAI-compatible embeddings 服务，用于 Agentic RAG。

项目 Go 代码没有第三方依赖，构建只依赖标准库。

## 构建与测试

```bash
go build -o lmstudio-forward ./cmd/lmstudio-forward
go test ./...
```

构建产物会生成在仓库根目录的 `./lmstudio-forward`。

## 快速启动

复制示例脚本，然后填入本机模型路径：

```bash
cp start.example.sh start.sh
chmod +x start.sh
./start.sh
```

默认代理服务监听：

```text
http://0.0.0.0:8000
```

### GGUF 后端

```bash
./lmstudio-forward \
  --gguf-model /path/to/model.gguf \
  --ctx-size 32768 \
  --no-frpc
```

### MLX 后端

```bash
./lmstudio-forward \
  --mlx-model /path/to/mlx-model-dir \
  --python-path .venv/bin/python3 \
  --ctx-size 32768 \
  --temperature 0.7 \
  --no-frpc
```

### 外部后端

不传 `--gguf-model` 和 `--mlx-model` 时，服务会尝试复用 `--backend-port` 指向的外部后端：

```bash
./lmstudio-forward \
  --backend-port 1234 \
  --server-port 8000 \
  --no-frpc
```

## API 示例

### OpenAI Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "你好，简单介绍一下你自己。"}
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
      {"role": "user", "content": "请用中文总结这个服务的作用。"}
    ]
  }'
```

如果设置了 `--api-key` / `API_KEY`，请求需要携带以下任一认证头：

```bash
Authorization: Bearer <API_KEY>
x-api-key: <API_KEY>
```

## Agentic RAG

开启 RAG 后，Anthropic Messages 请求会自动注入内置 `retrieve` 工具。模型如果判断需要查知识库，会发出 `retrieve` tool call；代理服务会在内部调用 embedding 服务和 Qdrant，把命中的文本片段作为检索结果追加回上下文，再让模型继续回答。这个检索过程对客户端是透明的。

当前 RAG 只接在 Anthropic Messages 路径上，也就是 `/v1/messages`、`/v1/message`、`/anthropic`、`/anthropic/v1/messages`。OpenAI `/v1/chat/completions` 路径目前是透明转发，不会执行内部 retrieve loop。

### 启用 RAG

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

注意：`--embed-dim` 必须和 embedding 模型实际输出维度一致，否则 Qdrant collection 维度会不匹配。

### 写入文档

```bash
curl http://127.0.0.1:8000/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {
        "source": "project-notes.md",
        "text": "这里是一段需要进入知识库的文档内容。"
      }
    ]
  }'
```

返回示例：

```json
{"ingested_chunks":1}
```

## 配置项

所有配置都可以通过 CLI flag 或环境变量设置。CLI flag 优先级高于环境变量。

| Flag | Env | 默认值 | 说明 |
|------|-----|--------|------|
| `--gguf-model` | `GGUF_MODEL` | 空 | GGUF 模型路径；设置后启用 `llama-server` 后端 |
| `--mlx-model` | `MLX_MODEL` | 空 | MLX 模型目录；设置后启用 `mlx_lm.server` 后端 |
| `--python-path` | `PYTHON_PATH` | `python3` | 启动 MLX 后端使用的 Python |
| `--llama-server` | `LLAMA_SERVER` | `/opt/homebrew/bin/llama-server` | `llama-server` 可执行文件路径 |
| `--backend-port` | `BACKEND_PORT` | `1234` | 本地模型后端端口 |
| `--server-port` | `SERVER_PORT` | `8000` | 本代理服务监听端口 |
| `--api-key` | `API_KEY` | 空 | API 认证密钥；为空表示不启用认证 |
| `--ctx-size` | `CTX_SIZE` | `8192` | 上下文窗口大小 |
| `--temperature` | `TEMPERATURE` | `0.7` | 默认 temperature |
| `--top-p` | `TOP_P` | `0.9` | 默认 nucleus sampling 参数 |
| `--min-p` | `MIN_P` | `0.05` | 默认 min-p 参数 |
| `--top-k` | `TOP_K` | `0` | 默认 top-k；`0` 表示不启用 |
| `--repetition-penalty` | `REPETITION_PENALTY` | `1.3` | 默认重复惩罚 |
| `--repetition-context-size` | `REPETITION_CONTEXT_SIZE` | `256` | 重复惩罚窗口 |
| `--max-tokens` | `MAX_TOKENS` | `16384` | MLX 后端最大输出 token |
| `--prefill-step-size` | `PREFILL_STEP_SIZE` | `4096` | MLX prompt 预处理批大小 |
| `--no-frpc` | `NO_FRPC` | `false` | 禁用 frpc 公网隧道 |
| `--frpc-path` | `FRPC_PATH` | `./frp_0.68.0_darwin_arm64/frpc` | frpc 可执行文件路径 |
| `--frpc-config` | `FRPC_CONFIG` | `./frp_0.68.0_darwin_arm64/frpc.toml` | frpc 配置文件路径 |
| `--rag-enabled` | `RAG_ENABLED` | `false` | 启用 Agentic RAG |
| `--qdrant-url` | `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant REST 地址 |
| `--qdrant-collection` | `QDRANT_COLLECTION` | `praxis_rag` | Qdrant collection 名称 |
| `--embed-url` | `EMBED_URL` | `http://127.0.0.1:1234` | OpenAI-compatible embeddings 服务地址 |
| `--embed-model` | `EMBED_MODEL` | `text-embedding` | embeddings 模型名 |
| `--embed-dim` | `EMBED_DIM` | `1024` | embedding 向量维度 |
| `--rag-top-k` | `RAG_TOP_K` | `5` | 每次检索返回的 chunk 数 |
| `--rag-max-rounds` | `RAG_MAX_ROUNDS` | `3` | 单次请求最多内部检索轮数 |
| `--rag-chunk-size` | `RAG_CHUNK_SIZE` | `800` | 文档写入时的 chunk 字符数 |
| `--rag-chunk-overlap` | `RAG_CHUNK_OVERLAP` | `100` | chunk 重叠字符数 |

## API 路由

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，会探测后端是否可用 |
| `GET` | `/v1/models` | 返回对外展示的模型列表 |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `GET/POST` | `/v1/*` | OpenAI-compatible 透传 |
| `POST` | `/v1/messages` | Anthropic Messages API |
| `POST` | `/v1/message` | Anthropic Messages API 兼容路径 |
| `POST` | `/anthropic` | Anthropic Messages API 兼容路径 |
| `POST` | `/anthropic/v1/messages` | Anthropic Messages API 兼容路径 |
| `POST` | `/rag/ingest` | RAG 文档写入 |

## 目录结构

```text
cmd/lmstudio-forward/   应用入口，只负责配置解析、依赖装配和服务启动
internal/
  config/      配置解析：flag、环境变量、默认值
  jsonx/       动态 JSON 辅助函数
  language/    token 估算、上下文截断、复杂度判断
  tools/       tool call 注入、解析、JSON 修复和协议适配
  rag/         Agentic RAG：Qdrant client、chunking、retrieve 工具
  stream/      SSE 收集和响应头设置
  proxy/       共享应用状态、客户端 IP 和 API key 校验
  openai/      OpenAI-compatible 转发处理
  anthropic/   Anthropic Messages 协议转换和流式事件转换
  process/     llama-server、mlx_lm.server 和 frpc 进程管理
  server/      HTTP 路由、健康检查和 RAG ingest 入口
```

## 开发说明

常用检查命令：

```bash
go test ./...
go build -o lmstudio-forward ./cmd/lmstudio-forward
```

如果只是本地开发和调试，建议加 `--no-frpc`，避免启动公网隧道。RAG 调试时先确认 Qdrant 可访问、embedding endpoint 可用，并且 embedding 维度和 collection 配置一致。
