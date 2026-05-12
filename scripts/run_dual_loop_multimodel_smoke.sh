#!/usr/bin/env bash
set -euo pipefail

MODEL_SPECS="${MODEL_SPECS:-}"
if [[ -z "$MODEL_SPECS" ]]; then
  cat >&2 <<'EOF'
Set MODEL_SPECS before running.

Format:
  MODEL_SPECS="/path/to/model|ModelStyle|ModelRepr;/path/to/other|ModelStyle|ModelRepr"

Example:
  MODEL_SPECS="/models/Qwen2.5-Coder-7B-Instruct|CodeQwenInstruct|Qwen2.5-Coder-7B-Instruct;/models/Meta-Llama-3.1-8B-Instruct|LLaMa3|Llama3.1-8B-Instruct" \
  DATASET_PATH=/home/shenshifan/datasets/livecodebench/code_generation_lite_release_v6 \
  bash scripts/run_dual_loop_multimodel_smoke.sh
EOF
  exit 2
fi

RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
TIMEOUT="${TIMEOUT:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
UV_BIN="${UV_BIN:-$(type -P uv || true)}"
DATASET_PATH="${DATASET_PATH:-}"

SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_SCORE_THRESHOLD="${SPEC_SCORE_THRESHOLD:-90}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"
CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-2}"
REPAIR_NUM_CANDIDATES="${REPAIR_NUM_CANDIDATES:-3}"
POST_FAILURE_SAL_MAX_ITERS="${POST_FAILURE_SAL_MAX_ITERS:-1}"
CONTRACT_SEARCH_POPULATION_SIZE="${CONTRACT_SEARCH_POPULATION_SIZE:-4}"
CONTRACT_SEARCH_ROUNDS="${CONTRACT_SEARCH_ROUNDS:-2}"
CONTRACT_SEARCH_TOP_K="${CONTRACT_SEARCH_TOP_K:-2}"
CONTRACT_SEARCH_CODEGEN_TOP_K="${CONTRACT_SEARCH_CODEGEN_TOP_K:-2}"
CONTRACT_SEARCH_TEMPERATURE="${CONTRACT_SEARCH_TEMPERATURE:-0.35}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-conservative}"
ATTRIBUTION_SPEC_MARGIN="${ATTRIBUTION_SPEC_MARGIN:-3}"

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find uv. Set UV_BIN=/absolute/path/to/uv." >&2
  exit 127
fi

COMMON_ARGS=(
  --release_version "$RELEASE_VERSION"
  --max_problems "$MAX_PROBLEMS"
  --tensor_parallel_size 1
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
  --repair_num_candidates "$REPAIR_NUM_CANDIDATES"
  --post_failure_sal_max_iters "$POST_FAILURE_SAL_MAX_ITERS"
  --contract_search_population_size "$CONTRACT_SEARCH_POPULATION_SIZE"
  --contract_search_rounds "$CONTRACT_SEARCH_ROUNDS"
  --contract_search_top_k "$CONTRACT_SEARCH_TOP_K"
  --contract_search_codegen_top_k "$CONTRACT_SEARCH_CODEGEN_TOP_K"
  --contract_search_temperature "$CONTRACT_SEARCH_TEMPERATURE"
  --attribution_mode "$ATTRIBUTION_MODE"
  --attribution_spec_margin "$ATTRIBUTION_SPEC_MARGIN"
  --include_pipeline_ablations
)

if [[ -n "$DATASET_PATH" ]]; then
  COMMON_ARGS+=(--dataset_path "$DATASET_PATH")
fi

IFS=';' read -r -a SPECS <<< "$MODEL_SPECS"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r LOCAL_MODEL_PATH MODEL_STYLE MODEL_REPR <<< "$spec"
  if [[ -z "${LOCAL_MODEL_PATH:-}" || -z "${MODEL_STYLE:-}" ]]; then
    echo "Invalid model spec: $spec" >&2
    exit 2
  fi
  MODEL_REPR="${MODEL_REPR:-$(basename "$LOCAL_MODEL_PATH")}"

  echo
  echo "============================================================"
  echo "Multi-model smoke: $MODEL_REPR"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/run_dual_loop_rq_suite.py \
    --local_model_path "$LOCAL_MODEL_PATH" \
    --model_style "$MODEL_STYLE" \
    --model "$MODEL_REPR" \
    --model_repr "$MODEL_REPR" \
    "${COMMON_ARGS[@]}"
done

echo
echo "All requested model smoke runs completed."
