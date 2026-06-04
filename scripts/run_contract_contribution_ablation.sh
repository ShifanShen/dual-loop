#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]"
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct"
  exit 1
fi

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="${2:-CodeQwenInstruct}"
MODEL_REPR="${MODEL_REPR:-$(basename "$LOCAL_MODEL_PATH")}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
TIMEOUT="${TIMEOUT:-6}"
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATASET_PATH="${DATASET_PATH:-}"
UV_BIN="${UV_BIN:-}"
METHODS="${METHODS:-raw_nl_irl,plan_irl,pseudocode_irl,contract_text_irl,semantic_contract_irl}"

CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-2}"
REPAIR_NUM_CANDIDATES="${REPAIR_NUM_CANDIDATES:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_TEMPERATURE="${SPEC_TEMPERATURE:-0.0}"
CODEGEN_TEMPERATURE="${CODEGEN_TEMPERATURE:-0.2}"
REPAIR_TEMPERATURE="${REPAIR_TEMPERATURE:-0.1}"
SPEC_MAX_TOKENS="${SPEC_MAX_TOKENS:-1400}"
CODEGEN_MAX_TOKENS="${CODEGEN_MAX_TOKENS:-2200}"

hash -r 2>/dev/null || true
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(type -P uv || true)"
fi

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find a usable uv executable." >&2
  echo "Set UV_BIN=/absolute/path/to/uv and rerun." >&2
  exit 127
fi

export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
  --model "$MODEL_REPR"
  --model_repr "$MODEL_REPR"
  --model_style "$MODEL_STYLE"
  --local_model_path "$LOCAL_MODEL_PATH"
  --release_version "$RELEASE_VERSION"
  --tensor_parallel_size 1
  --dtype "$DTYPE"
  --max_model_len "$MAX_MODEL_LEN"
  --max_problems "$MAX_PROBLEMS"
  --timeout "$TIMEOUT"
  --methods "$METHODS"
  --codegen_num_candidates "$CODEGEN_NUM_CANDIDATES"
  --repair_num_candidates "$REPAIR_NUM_CANDIDATES"
  --repair_max_iters "$REPAIR_MAX_ITERS"
  --spec_temperature "$SPEC_TEMPERATURE"
  --codegen_temperature "$CODEGEN_TEMPERATURE"
  --repair_temperature "$REPAIR_TEMPERATURE"
  --spec_max_tokens "$SPEC_MAX_TOKENS"
  --codegen_max_tokens "$CODEGEN_MAX_TOKENS"
)

if [[ -n "$DATASET_PATH" ]]; then
  ARGS+=(--dataset_path "$DATASET_PATH")
fi

echo "Running semantic contract contribution ablation:"
echo "  model_path=$LOCAL_MODEL_PATH"
echo "  model_style=$MODEL_STYLE"
echo "  max_problems=$MAX_PROBLEMS"
echo "  max_model_len=$MAX_MODEL_LEN"
echo "  methods=$METHODS"
echo "  codegen_num_candidates=$CODEGEN_NUM_CANDIDATES"
echo "  repair_num_candidates=$REPAIR_NUM_CANDIDATES"
echo "  repair_max_iters=$REPAIR_MAX_ITERS"
if [[ -n "$DATASET_PATH" ]]; then
  echo "  dataset_path=$DATASET_PATH"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/run_contract_contribution_ablation.py \
  "${ARGS[@]}"

echo
echo "Ablation finished. Check the newest directory under output/contract_contribution_ablation/."
