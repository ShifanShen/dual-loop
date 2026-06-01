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
VLLM_DEVICE="${VLLM_DEVICE:-}"
DATASET_PATH="${DATASET_PATH:-}"
UV_BIN="${UV_BIN:-}"

SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_SCORE_THRESHOLD="${SPEC_SCORE_THRESHOLD:-90}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"
ADAPTIVE_SAL_THRESHOLD="${ADAPTIVE_SAL_THRESHOLD:-85}"

CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-2}"
REPAIR_NUM_CANDIDATES="${REPAIR_NUM_CANDIDATES:-3}"
ADAPTIVE_CANDIDATE_BUDGET="${ADAPTIVE_CANDIDATE_BUDGET:-0}"
ADAPTIVE_CODEGEN_MAX_CANDIDATES="${ADAPTIVE_CODEGEN_MAX_CANDIDATES:-3}"
ADAPTIVE_REPAIR_MAX_CANDIDATES="${ADAPTIVE_REPAIR_MAX_CANDIDATES:-4}"
POST_FAILURE_SAL_MAX_ITERS="${POST_FAILURE_SAL_MAX_ITERS:-0}"
CONTRACT_SEARCH_POPULATION_SIZE="${CONTRACT_SEARCH_POPULATION_SIZE:-4}"
CONTRACT_SEARCH_ROUNDS="${CONTRACT_SEARCH_ROUNDS:-2}"
CONTRACT_SEARCH_TOP_K="${CONTRACT_SEARCH_TOP_K:-2}"
CONTRACT_SEARCH_CODEGEN_TOP_K="${CONTRACT_SEARCH_CODEGEN_TOP_K:-2}"
CONTRACT_SEARCH_TEMPERATURE="${CONTRACT_SEARCH_TEMPERATURE:-0.35}"
ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-conservative}"
ATTRIBUTION_SPEC_MARGIN="${ATTRIBUTION_SPEC_MARGIN:-3}"

hash -r 2>/dev/null || true
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(type -P uv || true)"
fi

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find a usable uv executable." >&2
  echo "Set UV_BIN=/absolute/path/to/uv and rerun." >&2
  exit 127
fi

COMMON_ARGS=(
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
  --spec_max_iters "$SPEC_MAX_ITERS"
  --repair_max_iters "$REPAIR_MAX_ITERS"
  --spec_score_threshold "$SPEC_SCORE_THRESHOLD"
  --spec_min_improvement "$SPEC_MIN_IMPROVEMENT"
  --spec_precision_floor "$SPEC_PRECISION_FLOOR"
  --spec_max_rejected_refines "$SPEC_MAX_REJECTED_REFINES"
  --adaptive_sal_threshold "$ADAPTIVE_SAL_THRESHOLD"
  --adaptive_ablation_threshold "$ADAPTIVE_SAL_THRESHOLD"
  --codegen_num_candidates "$CODEGEN_NUM_CANDIDATES"
  --repair_num_candidates "$REPAIR_NUM_CANDIDATES"
  --adaptive_codegen_max_candidates "$ADAPTIVE_CODEGEN_MAX_CANDIDATES"
  --adaptive_repair_max_candidates "$ADAPTIVE_REPAIR_MAX_CANDIDATES"
  --post_failure_sal_max_iters "$POST_FAILURE_SAL_MAX_ITERS"
  --contract_search_population_size "$CONTRACT_SEARCH_POPULATION_SIZE"
  --contract_search_rounds "$CONTRACT_SEARCH_ROUNDS"
  --contract_search_top_k "$CONTRACT_SEARCH_TOP_K"
  --contract_search_codegen_top_k "$CONTRACT_SEARCH_CODEGEN_TOP_K"
  --contract_search_temperature "$CONTRACT_SEARCH_TEMPERATURE"
  --attribution_mode "$ATTRIBUTION_MODE"
  --attribution_spec_margin "$ATTRIBUTION_SPEC_MARGIN"
)

if [[ -n "$VLLM_DEVICE" ]]; then
  COMMON_ARGS+=(--vllm_device "$VLLM_DEVICE")
fi

if [[ -n "$DATASET_PATH" ]]; then
  COMMON_ARGS+=(--dataset_path "$DATASET_PATH")
fi

if [[ "$ADAPTIVE_CANDIDATE_BUDGET" == "1" ]]; then
  COMMON_ARGS+=(--adaptive_candidate_budget)
fi

echo "Running minimal dual-loop smoke suite:"
echo "  model_path=$LOCAL_MODEL_PATH"
echo "  model_style=$MODEL_STYLE"
echo "  max_problems=$MAX_PROBLEMS"
echo "  max_model_len=$MAX_MODEL_LEN"
echo "  main methods=baseline,decomposition,self_refine,reflexion,full"
echo "  codegen_num_candidates=$CODEGEN_NUM_CANDIDATES"
echo "  repair_num_candidates=$REPAIR_NUM_CANDIDATES"
echo "  adaptive_candidate_budget=$ADAPTIVE_CANDIDATE_BUDGET"
if [[ "$ADAPTIVE_CANDIDATE_BUDGET" == "1" ]]; then
  echo "  adaptive_codegen_max_candidates=$ADAPTIVE_CODEGEN_MAX_CANDIDATES"
  echo "  adaptive_repair_max_candidates=$ADAPTIVE_REPAIR_MAX_CANDIDATES"
fi
echo "  contract_search=${CONTRACT_SEARCH_POPULATION_SIZE}x${CONTRACT_SEARCH_ROUNDS}"
echo "  post_failure_sal_max_iters=$POST_FAILURE_SAL_MAX_ITERS"
if [[ -n "$DATASET_PATH" ]]; then
  echo "  dataset_path=$DATASET_PATH"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/run_dual_loop_rq_suite.py \
  "${COMMON_ARGS[@]}"

echo
echo "Minimal smoke finished. Main outputs:"
echo "  output/dual_loop_rq_suite/"
echo "  rq_results.csv"
