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
UV_BIN="${UV_BIN:-}"
DATASET_PATH="${DATASET_PATH:-}"

SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_SCORE_THRESHOLD="${SPEC_SCORE_THRESHOLD:-90}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"
CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-1}"
CONTRACT_SEARCH_POPULATION_SIZE="${CONTRACT_SEARCH_POPULATION_SIZE:-1}"
CONTRACT_SEARCH_ROUNDS="${CONTRACT_SEARCH_ROUNDS:-0}"
CONTRACT_SEARCH_TOP_K="${CONTRACT_SEARCH_TOP_K:-1}"
CONTRACT_SEARCH_CODEGEN_TOP_K="${CONTRACT_SEARCH_CODEGEN_TOP_K:-1}"
CONTRACT_SEARCH_TEMPERATURE="${CONTRACT_SEARCH_TEMPERATURE:-0.35}"

ATTRIBUTION_MODE="${ATTRIBUTION_MODE:-conservative}"
ATTRIBUTION_SPEC_MARGIN="${ATTRIBUTION_SPEC_MARGIN:-3}"
ADAPTIVE_SAL_THRESHOLD="${ADAPTIVE_SAL_THRESHOLD:-85}"

MEDIUM_PIPELINE_PROBLEMS="${MEDIUM_PIPELINE_PROBLEMS:-200}"
REPAIR_ABLATION_PROBLEMS="${REPAIR_ABLATION_PROBLEMS:-50}"
FULL_RELEASE_PROBLEMS="${FULL_RELEASE_PROBLEMS:-1055}"
INTERMEDIATE_STUDY_PROBLEMS="${INTERMEDIATE_STUDY_PROBLEMS:-50}"

RUN_REPAIR_ABLATIONS="${RUN_REPAIR_ABLATIONS:-1}"
RUN_BUDGET_ABLATIONS="${RUN_BUDGET_ABLATIONS:-1}"
RUN_INTERMEDIATE_REP_STUDY="${RUN_INTERMEDIATE_REP_STUDY:-1}"
RUN_SAS_CORRELATION="${RUN_SAS_CORRELATION:-1}"
RUN_MANUAL_AUDIT_PACK="${RUN_MANUAL_AUDIT_PACK:-1}"
RUN_SEMANTIC_SUBSET_ANALYSIS="${RUN_SEMANTIC_SUBSET_ANALYSIS:-1}"
MANUAL_AUDIT_PER_LABEL="${MANUAL_AUDIT_PER_LABEL:-18}"

COMMON_ARGS=(
  --model "$MODEL_REPR"
  --model_repr "$MODEL_REPR"
  --model_style "$MODEL_STYLE"
  --local_model_path "$LOCAL_MODEL_PATH"
  --release_version "$RELEASE_VERSION"
  --tensor_parallel_size 1
  --dtype "$DTYPE"
  --spec_max_iters "$SPEC_MAX_ITERS"
  --repair_max_iters "$REPAIR_MAX_ITERS"
  --spec_score_threshold "$SPEC_SCORE_THRESHOLD"
  --spec_min_improvement "$SPEC_MIN_IMPROVEMENT"
  --spec_precision_floor "$SPEC_PRECISION_FLOOR"
  --spec_max_rejected_refines "$SPEC_MAX_REJECTED_REFINES"
  --adaptive_ablation_threshold "$ADAPTIVE_SAL_THRESHOLD"
  --timeout "$TIMEOUT"
  --codegen_num_candidates "$CODEGEN_NUM_CANDIDATES"
  --contract_search_population_size "$CONTRACT_SEARCH_POPULATION_SIZE"
  --contract_search_rounds "$CONTRACT_SEARCH_ROUNDS"
  --contract_search_top_k "$CONTRACT_SEARCH_TOP_K"
  --contract_search_codegen_top_k "$CONTRACT_SEARCH_CODEGEN_TOP_K"
  --contract_search_temperature "$CONTRACT_SEARCH_TEMPERATURE"
  --attribution_mode "$ATTRIBUTION_MODE"
  --attribution_spec_margin "$ATTRIBUTION_SPEC_MARGIN"
)

INTERMEDIATE_ARGS=(
  --model "$MODEL_REPR"
  --model_repr "$MODEL_REPR"
  --model_style "$MODEL_STYLE"
  --local_model_path "$LOCAL_MODEL_PATH"
  --release_version "$RELEASE_VERSION"
  --tensor_parallel_size 1
  --dtype "$DTYPE"
  --spec_max_iters "$SPEC_MAX_ITERS"
  --repair_max_iters "$REPAIR_MAX_ITERS"
  --spec_score_threshold "$SPEC_SCORE_THRESHOLD"
  --spec_min_improvement "$SPEC_MIN_IMPROVEMENT"
  --spec_precision_floor "$SPEC_PRECISION_FLOOR"
  --spec_max_rejected_refines "$SPEC_MAX_REJECTED_REFINES"
  --timeout "$TIMEOUT"
  --codegen_num_candidates "$CODEGEN_NUM_CANDIDATES"
)

if [[ -n "$DATASET_PATH" ]]; then
  COMMON_ARGS+=(--dataset_path "$DATASET_PATH")
  INTERMEDIATE_ARGS+=(--dataset_path "$DATASET_PATH")
fi

hash -r 2>/dev/null || true
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(type -P uv || true)"
fi

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find a usable uv executable." >&2
  echo "Set UV_BIN=/absolute/path/to/uv and rerun." >&2
  exit 127
fi

latest_dir() {
  local root="$1"
  local pattern="$2"
  find "$root" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_suite() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "$label"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/run_dual_loop_rq_suite.py \
    "${COMMON_ARGS[@]}" \
    "$@"
}

echo "Running revision suite with:"
echo "  model_path=$LOCAL_MODEL_PATH"
echo "  model_style=$MODEL_STYLE"
echo "  release_version=$RELEASE_VERSION"
echo "  attribution_mode=$ATTRIBUTION_MODE"
echo "  attribution_spec_margin=$ATTRIBUTION_SPEC_MARGIN"
echo "  adaptive_sal_threshold=$ADAPTIVE_SAL_THRESHOLD"
echo "  codegen_num_candidates=$CODEGEN_NUM_CANDIDATES"
echo "  contract_search_population_size=$CONTRACT_SEARCH_POPULATION_SIZE"
echo "  contract_search_rounds=$CONTRACT_SEARCH_ROUNDS"
echo "  contract_search_top_k=$CONTRACT_SEARCH_TOP_K"
echo "  contract_search_codegen_top_k=$CONTRACT_SEARCH_CODEGEN_TOP_K"
echo "  contract_search_temperature=$CONTRACT_SEARCH_TEMPERATURE"
echo "  uv_bin=$UV_BIN"
if [[ -n "$DATASET_PATH" ]]; then
  echo "  dataset_path=$DATASET_PATH"
fi

run_suite "[1/5] 200-problem main + pipeline + adaptive comparison" \
  --max_problems "$MEDIUM_PIPELINE_PROBLEMS" \
  --include_pipeline_ablations \
  --include_adaptive_ablations
MEDIUM_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"

if [[ "$RUN_REPAIR_ABLATIONS" == "1" ]]; then
  run_suite "[2/5] 50-problem repair ablations" \
    --max_problems "$REPAIR_ABLATION_PROBLEMS" \
    --include_repair_ablations
  REPAIR_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"
fi

if [[ "$RUN_INTERMEDIATE_REP_STUDY" == "1" ]]; then
  echo
  echo "============================================================"
  echo "[3/5] 50-problem Spec vs Plan/Pseudocode study"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/run_intermediate_repr_study.py \
    "${INTERMEDIATE_ARGS[@]}" \
    --max_problems "$INTERMEDIATE_STUDY_PROBLEMS" \
    --output_root output/intermediate_repr_study
fi

run_suite "[4/5] Full-release main comparison" \
  --max_problems "$FULL_RELEASE_PROBLEMS"
FULL_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"

if [[ "$RUN_BUDGET_ABLATIONS" == "1" ]]; then
  run_suite "[5/5] 50-problem budget ablations" \
    --max_problems "$REPAIR_ABLATION_PROBLEMS" \
    --include_budget_ablations
  BUDGET_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"
fi

if [[ "$RUN_SAS_CORRELATION" == "1" && -n "${MEDIUM_SUITE_DIR:-}" ]]; then
  echo
  echo "============================================================"
  echo "[Extra] SAS correlation analysis"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/analyze_sas_failure_correlation.py \
    --suite_dir "$MEDIUM_SUITE_DIR"
fi

if [[ "$RUN_SEMANTIC_SUBSET_ANALYSIS" == "1" && -n "${FULL_SUITE_DIR:-}" ]]; then
  echo
  echo "============================================================"
  echo "[Extra] Semantic-heavy subset analysis"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/analyze_semantic_subsets.py \
    --suite_dir "$FULL_SUITE_DIR"
fi

if [[ "$RUN_MANUAL_AUDIT_PACK" == "1" && -n "${MEDIUM_SUITE_DIR:-}" ]]; then
  echo
  echo "============================================================"
  echo "[Extra] Manual audit pack"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$UV_BIN" run python scripts/prepare_manual_audit_pack.py \
    --suite_dir "$MEDIUM_SUITE_DIR" \
    --per_label "$MANUAL_AUDIT_PER_LABEL"
fi

echo
echo "Revision suite finished."
echo "Key outputs:"
echo "  main/ablation suites: output/dual_loop_rq_suite/"
echo "  full-run traces: output/dual_loop/"
echo "  intermediate study: output/intermediate_repr_study/"
echo "  semantic subsets: <full suite>/semantic_subset_analysis/"
