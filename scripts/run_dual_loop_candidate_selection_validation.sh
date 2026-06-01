#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]"
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct"
  exit 1
fi

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="${2:-CodeQwenInstruct}"
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
DATASET_PATH="${DATASET_PATH:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_ID="${GPU_ID:-0}"

echo "Running minimal validation for verifier-aware candidate selection:"
echo "  model_path=$LOCAL_MODEL_PATH"
echo "  model_style=$MODEL_STYLE"
echo "  max_problems=$MAX_PROBLEMS"
echo "  max_model_len=$MAX_MODEL_LEN"
echo "  adaptive_candidate_budget=1"
if [[ -n "$DATASET_PATH" ]]; then
  echo "  dataset_path=$DATASET_PATH"
fi

export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="$GPU_ID" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MAX_PROBLEMS="$MAX_PROBLEMS" \
DATASET_PATH="$DATASET_PATH" \
ADAPTIVE_CANDIDATE_BUDGET=1 \
ADAPTIVE_CODEGEN_MAX_CANDIDATES="${ADAPTIVE_CODEGEN_MAX_CANDIDATES:-3}" \
ADAPTIVE_REPAIR_MAX_CANDIDATES="${ADAPTIVE_REPAIR_MAX_CANDIDATES:-4}" \
CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-2}" \
REPAIR_NUM_CANDIDATES="${REPAIR_NUM_CANDIDATES:-3}" \
POST_FAILURE_SAL_MAX_ITERS="${POST_FAILURE_SAL_MAX_ITERS:-0}" \
CONTRACT_SEARCH_POPULATION_SIZE="${CONTRACT_SEARCH_POPULATION_SIZE:-4}" \
CONTRACT_SEARCH_ROUNDS="${CONTRACT_SEARCH_ROUNDS:-2}" \
CONTRACT_SEARCH_TOP_K="${CONTRACT_SEARCH_TOP_K:-2}" \
CONTRACT_SEARCH_CODEGEN_TOP_K="${CONTRACT_SEARCH_CODEGEN_TOP_K:-2}" \
bash scripts/run_dual_loop_minimal_smoke.sh "$LOCAL_MODEL_PATH" "$MODEL_STYLE"

echo
echo "Validation finished. Check the newest directory under output/dual_loop_rq_suite/."
