#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/models}"
DOWNLOAD_LLAMA="${DOWNLOAD_LLAMA:-1}"
DOWNLOAD_DEEPSEEK="${DOWNLOAD_DEEPSEEK:-1}"
DOWNLOAD_QWEN="${DOWNLOAD_QWEN:-0}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(type -P modelscope || true)}"
UV_BIN="${UV_BIN:-$(type -P uv || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(type -P python3 || type -P python || true)}"

if [[ -z "$MODELSCOPE_BIN" && -x ".venv/bin/modelscope" ]]; then
  MODELSCOPE_BIN=".venv/bin/modelscope"
fi

if [[ -z "$MODELSCOPE_BIN" && -z "$UV_BIN" && -z "$PYTHON_BIN" ]]; then
  cat >&2 <<'EOF'
Could not find a usable Python/uv/ModelScope command.

Install it first:
  python3 -m pip install -U modelscope

or inside this project:
  uv pip install modelscope

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
  if [[ -n "$MODELSCOPE_BIN" ]]; then
    "$MODELSCOPE_BIN" download --model "$model_id" --local_dir "$target_dir"
  elif [[ -n "$UV_BIN" ]]; then
    "$UV_BIN" run python - "$model_id" "$target_dir" <<'PY'
import sys
from modelscope import snapshot_download

snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
  else
    "$PYTHON_BIN" - "$model_id" "$target_dir" <<'PY'
import sys
from modelscope import snapshot_download

snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
  fi
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
