#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]"
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="${2:-CodeQwenInstruct}"
MODEL_REPR="${MODEL_REPR:-$(basename "$LOCAL_MODEL_PATH")}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v6}"
DATASET_PATH="${DATASET_PATH:-}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TIMEOUT="${TIMEOUT:-6}"
MAX_PROBLEMS="${MAX_PROBLEMS:-30}"
SAMPLE_SEED="${SAMPLE_SEED:-2027}"
THRESHOLDS="${THRESHOLDS:-80,85,90,95}"
REFERENCE_THRESHOLD="${REFERENCE_THRESHOLD:-90}"
WEIGHT_PROFILES="${WEIGHT_PROFILES:-paper:0.4,0.4,0.2;uniform:0.333333,0.333333,0.333334;coverage:0.6,0.2,0.2;faithfulness:0.2,0.6,0.2;precision:0.2,0.2,0.6}"
SPEC_MAX_ITERS="${SPEC_MAX_ITERS:-3}"
SPEC_MIN_IMPROVEMENT="${SPEC_MIN_IMPROVEMENT:-1}"
SPEC_PRECISION_FLOOR="${SPEC_PRECISION_FLOOR:-85}"
SPEC_MAX_REJECTED_REFINES="${SPEC_MAX_REJECTED_REFINES:-1}"
CODEGEN_TEMPERATURE="${CODEGEN_TEMPERATURE:-0.2}"
NUM_PROCESS_EVALUATE="${NUM_PROCESS_EVALUATE:-8}"
RUN_DIR="${RUN_DIR:-output/sal_hparam_sensitivity/$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
UV_BIN="${UV_BIN:-}"

if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(type -P uv || true)"
fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find a usable uv executable. Set UV_BIN=/absolute/path/to/uv." >&2
  exit 127
fi

ARGS=(
  --model "$MODEL_REPR"
  --model_repr "$MODEL_REPR"
  --model_style "$MODEL_STYLE"
  --local_model_path "$LOCAL_MODEL_PATH"
  --release_version "$RELEASE_VERSION"
  --max_problems "$MAX_PROBLEMS"
  --sample_seed "$SAMPLE_SEED"
  --thresholds "$THRESHOLDS"
  --reference_threshold "$REFERENCE_THRESHOLD"
  --weight_profiles "$WEIGHT_PROFILES"
  --reference_weight paper
  --suite_dir "$RUN_DIR"
  --tensor_parallel_size 1
  --dtype "$DTYPE"
  --max_model_len "$MAX_MODEL_LEN"
  --timeout "$TIMEOUT"
  --num_process_evaluate "$NUM_PROCESS_EVALUATE"
  --spec_max_iters "$SPEC_MAX_ITERS"
  --spec_min_improvement "$SPEC_MIN_IMPROVEMENT"
  --spec_precision_floor "$SPEC_PRECISION_FLOOR"
  --spec_max_rejected_refines "$SPEC_MAX_REJECTED_REFINES"
  --spec_temperature 0.0
  --judge_temperature 0.0
  --codegen_temperature "$CODEGEN_TEMPERATURE"
)

if [[ -n "$DATASET_PATH" ]]; then
  ARGS+=(--dataset_path "$DATASET_PATH")
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry_run)
fi

echo "SAL hyperparameter sensitivity study"
echo "  model_path=$LOCAL_MODEL_PATH"
echo "  dataset_path=${DATASET_PATH:-release loader}"
echo "  sample=$MAX_PROBLEMS problems, seed=$SAMPLE_SEED"
echo "  thresholds=$THRESHOLDS"
echo "  reference_weights=(0.4,0.4,0.2)"
echo "  private_tests_used=false"
echo "  output=$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TOKENIZERS_PARALLELISM=false \
VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
"$UV_BIN" run python scripts/run_sal_hyperparameter_sensitivity.py "${ARGS[@]}"

echo
echo "Finished. Paper-ready outputs:"
echo "  $RUN_DIR/sensitivity_results.csv"
echo "  $RUN_DIR/sensitivity_summary.md"
echo "  $RUN_DIR/sensitivity_table.tex"
