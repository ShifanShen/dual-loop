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

# Frozen config from the latest healthy run.
SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
SPEC_SCORE_THRESHOLD="${SPEC_SCORE_THRESHOLD:-90}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"

# Experiment sizes.
REPAIR_ABLATION_PROBLEMS="${REPAIR_ABLATION_PROBLEMS:-50}"
MEDIUM_SCALE_PROBLEMS="${MEDIUM_SCALE_PROBLEMS:-200}"
FULL_RELEASE_PROBLEMS="${FULL_RELEASE_PROBLEMS:-1055}"

# Set RUN_FULL_RELEASE=0 if you want to stop after the 200-problem stage.
RUN_FULL_RELEASE="${RUN_FULL_RELEASE:-1}"

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

echo "[1/3] Running 50-problem repair ablations..."
CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/run_dual_loop_rq_suite.py \
  "${COMMON_ARGS[@]}" \
  --max_problems "$REPAIR_ABLATION_PROBLEMS" \
  --include_repair_ablations

echo "[2/3] Running 200-problem pipeline ablations..."
CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/run_dual_loop_rq_suite.py \
  "${COMMON_ARGS[@]}" \
  --max_problems "$MEDIUM_SCALE_PROBLEMS" \
  --include_pipeline_ablations

if [[ "$RUN_FULL_RELEASE" == "1" ]]; then
  echo "[3/3] Running full-release main comparison..."
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/run_dual_loop_rq_suite.py \
    "${COMMON_ARGS[@]}" \
    --max_problems "$FULL_RELEASE_PROBLEMS"
else
  echo "[3/3] Skipped full-release main comparison because RUN_FULL_RELEASE=$RUN_FULL_RELEASE"
fi

echo "Finished. Check the newest directories under:"
echo "  output/dual_loop/"
echo "  output/dual_loop_rq_suite/"
