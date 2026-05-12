#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/models}"
DOWNLOAD_LLAMA="${DOWNLOAD_LLAMA:-1}"
DOWNLOAD_DEEPSEEK="${DOWNLOAD_DEEPSEEK:-1}"
DOWNLOAD_QWEN="${DOWNLOAD_QWEN:-0}"

if ! command -v modelscope >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Could not find the ModelScope CLI.

Install it first:
  python -m pip install -U modelscope

Then rerun this script.
EOF
  exit 127
fi

download_model() {
  local model_id="$1"
  local target_dir="$2"

  mkdir -p "$target_dir"
  echo
  echo "============================================================"
  echo "Downloading $model_id"
  echo "Target: $target_dir"
  echo "============================================================"
  modelscope download --model "$model_id" --local_dir "$target_dir"
}

mkdir -p "$MODEL_ROOT"

if [[ "$DOWNLOAD_LLAMA" == "1" ]]; then
  download_model \
    "LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    "$MODEL_ROOT/Meta-Llama-3.1-8B-Instruct"
fi

if [[ "$DOWNLOAD_DEEPSEEK" == "1" ]]; then
  download_model \
    "deepseek-ai/deepseek-coder-6.7b-instruct" \
    "$MODEL_ROOT/deepseek-coder-6.7b-instruct"
fi

if [[ "$DOWNLOAD_QWEN" == "1" ]]; then
  download_model \
    "Qwen/Qwen2.5-Coder-7B-Instruct" \
    "$MODEL_ROOT/Qwen2.5-Coder-7B-Instruct"
fi

echo
echo "Model downloads finished."
echo "Model root: $MODEL_ROOT"
