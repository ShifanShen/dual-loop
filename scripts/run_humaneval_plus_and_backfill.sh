#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <main_local_model_path> [main_model_style]" >&2
  echo "Example: DATASET_PATH=/home/shenshifan/datasets/livecodebench/code_generation_lite_release_v6 \\" >&2
  echo "  $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct" >&2
  exit 2
fi

MAIN_MODEL_PATH="$1"
MAIN_MODEL_STYLE="${2:-CodeQwenInstruct}"
MAIN_MODEL_REPR="${MODEL_REPR:-$(basename "$MAIN_MODEL_PATH")}"

UV_BIN="${UV_BIN:-$(type -P uv || true)}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
TIMEOUT="${TIMEOUT:-6}"

RUN_HUMANEVAL_PLUS="${RUN_HUMANEVAL_PLUS:-1}"
RUN_DEEPSEEK_BACKFILL="${RUN_DEEPSEEK_BACKFILL:-1}"

HUMANEVAL_DATASET_PATH="${HUMANEVAL_DATASET_PATH:-/home/shenshifan/datasets/humaneval_plus_lcb_compatible}"
HUMANEVAL_SOURCE_JSON="${HUMANEVAL_SOURCE_JSON:-}"
FORCE_PREPARE_HUMANEVAL="${FORCE_PREPARE_HUMANEVAL:-0}"
INSTALL_EVALPLUS_IF_MISSING="${INSTALL_EVALPLUS_IF_MISSING:-1}"

LCB_DATASET_PATH="${DATASET_PATH:-/home/shenshifan/datasets/livecodebench/code_generation_lite_release_v6}"

DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/models/deepseek-coder-6.7b-instruct}"
DEEPSEEK_MODEL_STYLE="${DEEPSEEK_MODEL_STYLE:-DeepSeekCodeInstruct}"
DEEPSEEK_MODEL_REPR="${DEEPSEEK_MODEL_REPR:-DeepSeek-Coder-6.7B-Instruct}"
DEEPSEEK_MAX_MODEL_LEN="${DEEPSEEK_MAX_MODEL_LEN:-8192}"

HUMANEVAL_MAX_MODEL_LEN="${HUMANEVAL_MAX_MODEL_LEN:-16384}"
HUMANEVAL_MEDIUM_PIPELINE_PROBLEMS="${HUMANEVAL_MEDIUM_PIPELINE_PROBLEMS:-164}"
HUMANEVAL_FULL_RELEASE_PROBLEMS="${HUMANEVAL_FULL_RELEASE_PROBLEMS:-164}"
HUMANEVAL_REPAIR_ABLATION_PROBLEMS="${HUMANEVAL_REPAIR_ABLATION_PROBLEMS:-50}"
HUMANEVAL_INTERMEDIATE_STUDY_PROBLEMS="${HUMANEVAL_INTERMEDIATE_STUDY_PROBLEMS:-50}"

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find uv. Set UV_BIN=/absolute/path/to/uv." >&2
  exit 127
fi

prepare_humaneval_plus() {
  if [[ "$INSTALL_EVALPLUS_IF_MISSING" == "1" ]]; then
    if ! "$UV_BIN" run python -c "import evalplus" >/dev/null 2>&1; then
      "$UV_BIN" pip install evalplus
    fi
  fi

  local prepare_args=(--output_dir "$HUMANEVAL_DATASET_PATH")
  if [[ "$FORCE_PREPARE_HUMANEVAL" == "1" || ! -d "$HUMANEVAL_DATASET_PATH" ]]; then
    prepare_args+=(--force)
  fi
  if [[ -n "$HUMANEVAL_SOURCE_JSON" ]]; then
    prepare_args+=(--source_json "$HUMANEVAL_SOURCE_JSON")
  fi

  if [[ "$FORCE_PREPARE_HUMANEVAL" == "1" || ! -d "$HUMANEVAL_DATASET_PATH" ]]; then
    "$UV_BIN" run python scripts/prepare_humaneval_plus_dataset.py "${prepare_args[@]}"
  else
    echo "HumanEval+ dataset already exists: $HUMANEVAL_DATASET_PATH"
  fi
}

run_humaneval_suite() {
  echo
  echo "============================================================"
  echo "HumanEval+ full revision suite: $MAIN_MODEL_REPR"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DATASET_PATH="$HUMANEVAL_DATASET_PATH" \
  RELEASE_VERSION="humaneval_plus" \
  MODEL_REPR="${MAIN_MODEL_REPR}_HumanEvalPlus" \
  MAX_MODEL_LEN="$HUMANEVAL_MAX_MODEL_LEN" \
  MEDIUM_PIPELINE_PROBLEMS="$HUMANEVAL_MEDIUM_PIPELINE_PROBLEMS" \
  FULL_RELEASE_PROBLEMS="$HUMANEVAL_FULL_RELEASE_PROBLEMS" \
  REPAIR_ABLATION_PROBLEMS="$HUMANEVAL_REPAIR_ABLATION_PROBLEMS" \
  INTERMEDIATE_STUDY_PROBLEMS="$HUMANEVAL_INTERMEDIATE_STUDY_PROBLEMS" \
  RUN_MANUAL_AUDIT_PACK=0 \
  bash scripts/run_dual_loop_revision_suite.sh "$MAIN_MODEL_PATH" "$MAIN_MODEL_STYLE"
}

run_deepseek_backfill() {
  if [[ ! -d "$DEEPSEEK_MODEL_PATH" ]]; then
    echo "Skipping DeepSeek backfill; missing model directory: $DEEPSEEK_MODEL_PATH" >&2
    return 0
  fi
  if [[ ! -d "$LCB_DATASET_PATH" ]]; then
    echo "Skipping DeepSeek backfill; missing LiveCodeBench dataset: $LCB_DATASET_PATH" >&2
    return 0
  fi

  echo
  echo "============================================================"
  echo "Backfill failed multi-model smoke: $DEEPSEEK_MODEL_REPR"
  echo "============================================================"
  MODEL_SPECS="${DEEPSEEK_MODEL_PATH}|${DEEPSEEK_MODEL_STYLE}|${DEEPSEEK_MODEL_REPR}" \
  DATASET_PATH="$LCB_DATASET_PATH" \
  GPU_ID="$GPU_ID" \
  DTYPE="$DTYPE" \
  TIMEOUT="$TIMEOUT" \
  MAX_MODEL_LEN="$DEEPSEEK_MAX_MODEL_LEN" \
  MAX_PROBLEMS="${BACKFILL_MAX_PROBLEMS:-50}" \
  bash scripts/run_dual_loop_multimodel_smoke.sh
}

if [[ "$RUN_HUMANEVAL_PLUS" == "1" ]]; then
  prepare_humaneval_plus
  run_humaneval_suite
fi

if [[ "$RUN_DEEPSEEK_BACKFILL" == "1" ]]; then
  run_deepseek_backfill
fi

echo
echo "HumanEval+ suite/backfill script finished."
echo "HumanEval+ dataset: $HUMANEVAL_DATASET_PATH"
echo "HumanEval+ outputs: output/dual_loop_rq_suite/, output/intermediate_repr_study/"
echo "DeepSeek backfill outputs: output/dual_loop_multimodel_smoke/"
