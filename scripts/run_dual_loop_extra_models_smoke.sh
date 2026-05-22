#!/usr/bin/env bash
set -euo pipefail

RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
GPU_ID="${GPU_ID:-0}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TIMEOUT="${TIMEOUT:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
UV_BIN="${UV_BIN:-$(type -P uv || true)}"
DATASET_PATH="${DATASET_PATH:-}"

LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/models/Meta-Llama-3.1-8B-Instruct}"
DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/models/deepseek-coder-6.7b-instruct}"

SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_SCORE_THRESHOLD="${SPEC_SCORE_THRESHOLD:-90}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"
CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-2}"
CODEGEN_CONTRACT_MODE="${CODEGEN_CONTRACT_MODE:-open}"
REPAIR_NUM_CANDIDATES="${REPAIR_NUM_CANDIDATES:-3}"
POST_FAILURE_SAL_MAX_ITERS="${POST_FAILURE_SAL_MAX_ITERS:-1}"
POST_FAILURE_SAL_TRIGGER="${POST_FAILURE_SAL_TRIGGER:-attribution}"
CONTRACT_SEARCH_POPULATION_SIZE="${CONTRACT_SEARCH_POPULATION_SIZE:-4}"
CONTRACT_SEARCH_ROUNDS="${CONTRACT_SEARCH_ROUNDS:-2}"
CONTRACT_SEARCH_TOP_K="${CONTRACT_SEARCH_TOP_K:-2}"
CONTRACT_SEARCH_CODEGEN_TOP_K="${CONTRACT_SEARCH_CODEGEN_TOP_K:-2}"
CONTRACT_SEARCH_TEMPERATURE="${CONTRACT_SEARCH_TEMPERATURE:-0.35}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-evidence}"
ATTRIBUTION_SPEC_MARGIN="${ATTRIBUTION_SPEC_MARGIN:-3}"
ATTRIBUTION_REENTRY_CONFIDENCE_THRESHOLD="${ATTRIBUTION_REENTRY_CONFIDENCE_THRESHOLD:-0.6}"
FAILURE_GAP_CONFIDENCE_THRESHOLD="${FAILURE_GAP_CONFIDENCE_THRESHOLD:-70}"
DISABLE_FAILURE_GAP_JUDGE="${DISABLE_FAILURE_GAP_JUDGE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find uv. Set UV_BIN=/absolute/path/to/uv." >&2
  exit 127
fi

if [[ ! -d "$LLAMA_MODEL_PATH" ]]; then
  echo "Missing Llama model directory: $LLAMA_MODEL_PATH" >&2
  echo "Set LLAMA_MODEL_PATH or run scripts/download_modelscope_extra_models.sh first." >&2
  exit 2
fi

if [[ ! -d "$DEEPSEEK_MODEL_PATH" ]]; then
  echo "Missing DeepSeek model directory: $DEEPSEEK_MODEL_PATH" >&2
  echo "Set DEEPSEEK_MODEL_PATH or run scripts/download_modelscope_extra_models.sh first." >&2
  exit 2
fi

COMMON_ARGS=(
  --release_version "$RELEASE_VERSION"
  --max_problems "$MAX_PROBLEMS"
  --tensor_parallel_size 1
  --vllm_device "$VLLM_TARGET_DEVICE"
  --max_model_len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --spec_max_iters "$SPEC_MAX_ITERS"
  --repair_max_iters "$REPAIR_MAX_ITERS"
  --spec_score_threshold "$SPEC_SCORE_THRESHOLD"
  --spec_min_improvement "$SPEC_MIN_IMPROVEMENT"
  --spec_precision_floor "$SPEC_PRECISION_FLOOR"
  --spec_max_rejected_refines "$SPEC_MAX_REJECTED_REFINES"
  --timeout "$TIMEOUT"
  --codegen_num_candidates "$CODEGEN_NUM_CANDIDATES"
  --codegen_contract_mode "$CODEGEN_CONTRACT_MODE"
  --repair_num_candidates "$REPAIR_NUM_CANDIDATES"
  --post_failure_sal_max_iters "$POST_FAILURE_SAL_MAX_ITERS"
  --post_failure_sal_trigger "$POST_FAILURE_SAL_TRIGGER"
  --contract_search_population_size "$CONTRACT_SEARCH_POPULATION_SIZE"
  --contract_search_rounds "$CONTRACT_SEARCH_ROUNDS"
  --contract_search_top_k "$CONTRACT_SEARCH_TOP_K"
  --contract_search_codegen_top_k "$CONTRACT_SEARCH_CODEGEN_TOP_K"
  --contract_search_temperature "$CONTRACT_SEARCH_TEMPERATURE"
  --attribution_mode "$ATTRIBUTION_MODE"
  --attribution_spec_margin "$ATTRIBUTION_SPEC_MARGIN"
  --attribution_reentry_confidence_threshold "$ATTRIBUTION_REENTRY_CONFIDENCE_THRESHOLD"
  --failure_gap_confidence_threshold "$FAILURE_GAP_CONFIDENCE_THRESHOLD"
  --include_pipeline_ablations
  --suite_output_root output/dual_loop_multimodel_smoke
  --output_root output/dual_loop_multimodel_raw
)

if [[ -n "$DATASET_PATH" ]]; then
  COMMON_ARGS+=(--dataset_path "$DATASET_PATH")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  COMMON_ARGS+=(--trust_remote_code)
fi
if [[ "$DISABLE_FAILURE_GAP_JUDGE" == "1" ]]; then
  COMMON_ARGS+=(--disable_failure_gap_judge)
fi

run_model() {
  local local_model_path="$1"
  local model_style="$2"
  local model_repr="$3"

  echo
  echo "============================================================"
  echo "Multi-model smoke: $model_repr"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_TARGET_DEVICE="$VLLM_TARGET_DEVICE" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$UV_BIN" run python scripts/run_dual_loop_rq_suite.py \
      --local_model_path "$local_model_path" \
      --model_style "$model_style" \
      --model "$model_repr" \
      --model_repr "$model_repr" \
      "${COMMON_ARGS[@]}"
}

run_model "$LLAMA_MODEL_PATH" "LLaMa3" "Llama3.1-8B-Instruct"
run_model "$DEEPSEEK_MODEL_PATH" "DeepSeekCodeInstruct" "DeepSeek-Coder-6.7B-Instruct"

echo
echo "Extra-model smoke runs completed."
echo "Outputs: output/dual_loop_multimodel_smoke/"
