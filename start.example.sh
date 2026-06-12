#!/bin/bash
# Copy this file to start.sh and fill in your model path.
# Usage: cp start.example.sh start.sh && chmod +x start.sh && ./start.sh
#
# Backend options (pick one):
#   GGUF (llama.cpp):  --gguf-model <path-to-.gguf>
#   MLX:               --mlx-model  <path-to-mlx-model-dir>
#
# Common flags:
#   --ctx-size          Max context window (tokens)
#   --temperature       Sampling temperature (default 0.7)
#   --top-p             Nucleus sampling (default 0.9)
#   --min-p             Min-p filtering (default 0.05)
#   --repetition-penalty  Repeat penalty (default 1.3)
#   --max-tokens        Max output tokens (default 16384)
#   --no-frpc           Disable frpc tunnel (local-only mode)
#
# Example model paths:
#   ~/.lmstudio/models/<author>/<model-name>/<file>.gguf   (GGUF)
#   ~/.lmstudio/models/<author>/<model-name>/               (MLX)

cd "$(dirname "$0")"

# --- GGUF example ---
RUST_LOG=info exec ./target/release/lmstudio-forward \
  --gguf-model "${GGUF_MODEL_PATH:-/path/to/your/model.gguf}" \
  --ctx-size 32768 \
  --no-frpc

# --- MLX example (uncomment and remove GGUF block above) ---
# RUST_LOG=info exec ./target/release/lmstudio-forward \
#   --mlx-model "${MLX_MODEL_PATH:-/path/to/your/mlx-model-dir}" \
#   --python-path "$(pwd)/.venv/bin/python3" \
#   --ctx-size 32768 \
#   --temperature 0.7 \
#   --no-frpc
