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

# Frozen configuration.
SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_SCORE_THRESHOLD="${SPEC_SCORE_THRESHOLD:-90}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"

# Experiment sizes.
DEV_PIPELINE_PROBLEMS="${DEV_PIPELINE_PROBLEMS:-50}"
REPAIR_ABLATION_PROBLEMS="${REPAIR_ABLATION_PROBLEMS:-50}"
MEDIUM_PIPELINE_PROBLEMS="${MEDIUM_PIPELINE_PROBLEMS:-200}"
FULL_RELEASE_PROBLEMS="${FULL_RELEASE_PROBLEMS:-1055}"
BUDGET_ABLATION_PROBLEMS="${BUDGET_ABLATION_PROBLEMS:-50}"
INTERMEDIATE_STUDY_PROBLEMS="${INTERMEDIATE_STUDY_PROBLEMS:-50}"

# Set RUN_BUDGET_ABLATIONS=1 if you want the cost study too.
RUN_BUDGET_ABLATIONS="${RUN_BUDGET_ABLATIONS:-0}"
RUN_INTERMEDIATE_REP_STUDY="${RUN_INTERMEDIATE_REP_STUDY:-1}"
RUN_SAS_CORRELATION="${RUN_SAS_CORRELATION:-1}"
RUN_MANUAL_AUDIT_PACK="${RUN_MANUAL_AUDIT_PACK:-1}"
MANUAL_AUDIT_PER_LABEL="${MANUAL_AUDIT_PER_LABEL:-12}"

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
  --timeout "$TIMEOUT"
)

run_suite() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "$label"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/run_dual_loop_rq_suite.py \
    "${COMMON_ARGS[@]}" \
    "$@"
}

latest_dir() {
  local root="$1"
  local pattern="$2"
  find "$root" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

echo "Running frozen paper experiment suite with:"
echo "  model_path=$LOCAL_MODEL_PATH"
echo "  model_style=$MODEL_STYLE"
echo "  release_version=$RELEASE_VERSION"
echo "  spec_max_iters=$SPEC_MAX_ITERS"
echo "  repair_max_iters=$REPAIR_MAX_ITERS"
echo "  spec_score_threshold=$SPEC_SCORE_THRESHOLD"
echo "  spec_min_improvement=$SPEC_MIN_IMPROVEMENT"
echo "  spec_precision_floor=$SPEC_PRECISION_FLOOR"
echo "  spec_max_rejected_refines=$SPEC_MAX_REJECTED_REFINES"
echo "  gpu_id=$GPU_ID"

run_suite "[1/4] 50-problem pipeline ablations" \
  --max_problems "$DEV_PIPELINE_PROBLEMS" \
  --include_pipeline_ablations
DEV_PIPELINE_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"

run_suite "[2/4] 50-problem repair ablations" \
  --max_problems "$REPAIR_ABLATION_PROBLEMS" \
  --include_repair_ablations
REPAIR_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"

if [[ "$RUN_INTERMEDIATE_REP_STUDY" == "1" ]]; then
  echo
  echo "============================================================"
  echo "[Extra] 50-problem spec vs plan/pseudocode intermediate study"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/run_intermediate_repr_study.py \
    "${COMMON_ARGS[@]}" \
    --max_problems "$INTERMEDIATE_STUDY_PROBLEMS" \
    --output_root output/intermediate_repr_study
fi

run_suite "[3/4] 200-problem pipeline ablations" \
  --max_problems "$MEDIUM_PIPELINE_PROBLEMS" \
  --include_pipeline_ablations
MEDIUM_PIPELINE_SUITE_DIR="$(latest_dir output/dual_loop_rq_suite 'rq_suite_*')"

if [[ "$RUN_SAS_CORRELATION" == "1" && -n "${MEDIUM_PIPELINE_SUITE_DIR:-}" ]]; then
  echo
  echo "============================================================"
  echo "[Extra] SAS vs semantic-failure correlation analysis"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/analyze_sas_failure_correlation.py \
    --suite_dir "$MEDIUM_PIPELINE_SUITE_DIR"
fi

if [[ "$RUN_MANUAL_AUDIT_PACK" == "1" && -n "${MEDIUM_PIPELINE_SUITE_DIR:-}" ]]; then
  echo
  echo "============================================================"
  echo "[Extra] Manual audit pack preparation"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/prepare_manual_audit_pack.py \
    --suite_dir "$MEDIUM_PIPELINE_SUITE_DIR" \
    --per_label "$MANUAL_AUDIT_PER_LABEL"
fi

if [[ "$FULL_RELEASE_PROBLEMS" -gt 0 ]]; then
  run_suite "[4/4] Full-release main comparison" \
    --max_problems "$FULL_RELEASE_PROBLEMS"
fi

if [[ "$RUN_BUDGET_ABLATIONS" == "1" ]]; then
  run_suite "[Optional] 50-problem budget ablations" \
    --max_problems "$BUDGET_ABLATION_PROBLEMS" \
    --include_budget_ablations
fi

echo
echo "All requested paper experiments finished."
echo "Check the newest directories under:"
echo "  output/dual_loop/"
echo "  output/dual_loop_rq_suite/"
