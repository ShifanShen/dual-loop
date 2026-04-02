#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <suite_dir> <local_model_path> [model_style]"
  echo "Example: $0 rq_suite_Qwen2.5-Coder-7B-Instruct_20260327_194706 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct"
  exit 1
fi

SUITE_DIR="$1"
LOCAL_MODEL_PATH="$2"
MODEL_STYLE="${3:-CodeQwenInstruct}"
MODEL_REPR="${MODEL_REPR:-$(basename "$LOCAL_MODEL_PATH")}"
RUN_NAME="${RUN_NAME:-full_dual_loop}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/spec_counterfactual}"
DTYPE="${DTYPE:-bfloat16}"
TIMEOUT="${TIMEOUT:-6}"
GPU_ID="${GPU_ID:-0}"

RUN_SPEC_COUNTERFACTUAL_REPEATS="${RUN_SPEC_COUNTERFACTUAL_REPEATS:-5}"
RUN_SPEC_COUNTERFACTUAL_TEMPERATURE="${RUN_SPEC_COUNTERFACTUAL_TEMPERATURE:-0.2}"
RUN_SPEC_COUNTERFACTUAL_LOW_SAS_THRESHOLD="${RUN_SPEC_COUNTERFACTUAL_LOW_SAS_THRESHOLD:-85}"
RUN_SPEC_COUNTERFACTUAL_CODEGEN_NUM_CANDIDATES="${RUN_SPEC_COUNTERFACTUAL_CODEGEN_NUM_CANDIDATES:-1}"
RUN_SPEC_COUNTERFACTUAL_SPEC_INDUCED_MAX_PROBLEMS="${RUN_SPEC_COUNTERFACTUAL_SPEC_INDUCED_MAX_PROBLEMS:-50}"

COMMON_ARGS=(
  --suite_dir "$SUITE_DIR"
  --run_name "$RUN_NAME"
  --local_model_path "$LOCAL_MODEL_PATH"
  --model "$MODEL_REPR"
  --model_repr "$MODEL_REPR"
  --model_style "$MODEL_STYLE"
  --output_root "$OUTPUT_ROOT"
  --tensor_parallel_size 1
  --dtype "$DTYPE"
  --timeout "$TIMEOUT"
  --repeats "$RUN_SPEC_COUNTERFACTUAL_REPEATS"
  --codegen_temperature "$RUN_SPEC_COUNTERFACTUAL_TEMPERATURE"
  --codegen_num_candidates "$RUN_SPEC_COUNTERFACTUAL_CODEGEN_NUM_CANDIDATES"
  --low_initial_sas_threshold "$RUN_SPEC_COUNTERFACTUAL_LOW_SAS_THRESHOLD"
)

run_subset() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "$label"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/run_spec_counterfactual_study.py \
    "${COMMON_ARGS[@]}" \
    "$@"
}

echo "Running spec counterfactual suite with:"
echo "  suite_dir=$SUITE_DIR"
echo "  run_name=$RUN_NAME"
echo "  local_model_path=$LOCAL_MODEL_PATH"
echo "  model_style=$MODEL_STYLE"
echo "  repeats=$RUN_SPEC_COUNTERFACTUAL_REPEATS"
echo "  codegen_temperature=$RUN_SPEC_COUNTERFACTUAL_TEMPERATURE"
echo "  codegen_num_candidates=$RUN_SPEC_COUNTERFACTUAL_CODEGEN_NUM_CANDIDATES"
echo "  low_initial_sas_threshold=$RUN_SPEC_COUNTERFACTUAL_LOW_SAS_THRESHOLD"

run_subset "[1/3] changed_and_initial_codegen_failed" \
  --subset changed_and_initial_codegen_failed

run_subset "[2/3] resolved_by_loop_b_and_initial_codegen_failed" \
  --subset resolved_by_loop_b_and_initial_codegen_failed

run_subset "[3/3] source_spec_induced" \
  --subset source_spec_induced \
  --max_problems "$RUN_SPEC_COUNTERFACTUAL_SPEC_INDUCED_MAX_PROBLEMS"

echo
echo "Spec counterfactual suite finished."
echo "Check the newest directories under:"
echo "  output/spec_counterfactual/"
