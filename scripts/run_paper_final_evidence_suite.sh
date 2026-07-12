#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]" >&2
  echo "Example: DATASET_PATH=/data/lcb_release_v6 $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct" >&2
  exit 1
fi

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="${2:-CodeQwenInstruct}"
DATASET_PATH="${DATASET_PATH:-}"
TRACE_PATH="${TRACE_PATH:-output/dual_loop/full_Qwen2.5-Coder-7B-Instruct_full_dual_loop_20260529_005015_876463/traces.json}"
SUMMARY_PATH="${SUMMARY_PATH:-$(dirname "$TRACE_PATH")/summary.json}"
REEVAL_OUTPUT_DIR="${REEVAL_OUTPUT_DIR:-output/heldout_reevaluation/full_dual_loop_20260529_private}"
REEVAL_TIMEOUT="${REEVAL_TIMEOUT:-6}"
REEVAL_PROCESSES="${REEVAL_PROCESSES:-8}"
REEVAL_BATCH_SIZE="${REEVAL_BATCH_SIZE:-50}"
REEVAL_MAX_PROBLEMS="${REEVAL_MAX_PROBLEMS:-0}"
RUN_REEVALUATION="${RUN_REEVALUATION:-1}"
RUN_BUDGET_ABLATION="${RUN_BUDGET_ABLATION:-1}"
UV_BIN="${UV_BIN:-}"

if [[ -z "$DATASET_PATH" ]]; then
  echo "DATASET_PATH must point to the local LiveCodeBench release_v6 dataset." >&2
  exit 2
fi

hash -r 2>/dev/null || true
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(type -P uv || true)"
fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find a usable uv executable. Set UV_BIN=/absolute/path/to/uv." >&2
  exit 127
fi

if [[ "$RUN_REEVALUATION" == "1" ]]; then
  if [[ ! -f "$TRACE_PATH" ]]; then
    echo "Missing source trace file: $TRACE_PATH" >&2
    exit 3
  fi

  REEVAL_ARGS=(
    --traces "$TRACE_PATH"
    --dataset_path "$DATASET_PATH"
    --release_version release_v6
    --output_dir "$REEVAL_OUTPUT_DIR"
    --timeout "$REEVAL_TIMEOUT"
    --num_process_evaluate "$REEVAL_PROCESSES"
    --batch_size "$REEVAL_BATCH_SIZE"
    --max_problems "$REEVAL_MAX_PROBLEMS"
    --resume
  )
  if [[ -f "$SUMMARY_PATH" ]]; then
    REEVAL_ARGS+=(--summary "$SUMMARY_PATH")
  fi

  echo "Stage 1/2: re-evaluating saved final_code values on private held-out tests."
  echo "  traces=$TRACE_PATH"
  echo "  output=$REEVAL_OUTPUT_DIR"
  echo "  model_generation=false"
  "$UV_BIN" run python scripts/reevaluate_heldout_traces.py "${REEVAL_ARGS[@]}"
else
  echo "Stage 1/2 skipped because RUN_REEVALUATION=$RUN_REEVALUATION."
fi

if [[ "$RUN_BUDGET_ABLATION" == "1" ]]; then
  echo
  echo "Stage 2/2: running the mechanism-matched contract ablation."
  DATASET_PATH="$DATASET_PATH" \
    bash scripts/run_budget_matched_contract_ablation.sh \
      "$LOCAL_MODEL_PATH" "$MODEL_STYLE"
else
  echo "Stage 2/2 skipped because RUN_BUDGET_ABLATION=$RUN_BUDGET_ABLATION."
fi

echo
echo "Paper evidence suite finished."
echo "Held-out re-evaluation: $REEVAL_OUTPUT_DIR"
echo "Contract ablation: newest directory under output/contract_contribution_ablation/"
