#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <local_model_path> <model_style> [pipeline_mode] [max_problems]"
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct full 20"
  exit 1
fi

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="$2"
PIPELINE_MODE="${3:-full}"
MAX_PROBLEMS="${4:-20}"
MODEL_REPR="${MODEL_REPR:-$(basename "$LOCAL_MODEL_PATH")}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
GPU_ID="${GPU_ID:-0}"

CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python -m lcb_runner.dual_loop.main \
  --model "$MODEL_REPR" \
  --model_repr "$MODEL_REPR" \
  --model_style "$MODEL_STYLE" \
  --local_model_path "$LOCAL_MODEL_PATH" \
  --pipeline_mode "$PIPELINE_MODE" \
  --release_version "$RELEASE_VERSION" \
  --max_problems "$MAX_PROBLEMS" \
  --tensor_parallel_size 1 \
  --dtype bfloat16 \
  --enable_prefix_caching \
  --spec_max_iters 3 \
  --repair_max_iters 3 \
  --spec_score_threshold 80 \
  --timeout 6
