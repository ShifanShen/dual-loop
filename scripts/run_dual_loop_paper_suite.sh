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

# Set RUN_BUDGET_ABLATIONS=1 if you want the cost study too.
RUN_BUDGET_ABLATIONS="${RUN_BUDGET_ABLATIONS:-0}"

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

run_suite "[2/4] 50-problem repair ablations" \
  --max_problems "$REPAIR_ABLATION_PROBLEMS" \
  --include_repair_ablations

run_suite "[3/4] 200-problem pipeline ablations" \
  --max_problems "$MEDIUM_PIPELINE_PROBLEMS" \
  --include_pipeline_ablations

run_suite "[4/4] Full-release main comparison" \
  --max_problems "$FULL_RELEASE_PROBLEMS"

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
