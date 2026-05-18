#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]" >&2
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct" >&2
  exit 2
fi

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="${2:-CodeQwenInstruct}"
MODEL_REPR="${MODEL_REPR:-$(basename "$LOCAL_MODEL_PATH")}_MBPPPlus"

UV_BIN="${UV_BIN:-$(type -P uv || true)}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-bfloat16}"
TIMEOUT="${TIMEOUT:-6}"
MBPP_DATASET_PATH="${MBPP_DATASET_PATH:-/home/shenshifan/datasets/mbpp_plus_lcb_compatible}"
MBPP_SOURCE_JSON="${MBPP_SOURCE_JSON:-}"
FORCE_PREPARE_MBPP="${FORCE_PREPARE_MBPP:-0}"
INSTALL_EVALPLUS_IF_MISSING="${INSTALL_EVALPLUS_IF_MISSING:-1}"
MBPP_MAX_MODEL_LEN="${MBPP_MAX_MODEL_LEN:-8192}"

MBPP_MEDIUM_PIPELINE_PROBLEMS="${MBPP_MEDIUM_PIPELINE_PROBLEMS:-399}"
MBPP_FULL_RELEASE_PROBLEMS="${MBPP_FULL_RELEASE_PROBLEMS:-399}"
MBPP_REPAIR_ABLATION_PROBLEMS="${MBPP_REPAIR_ABLATION_PROBLEMS:-50}"
MBPP_INTERMEDIATE_STUDY_PROBLEMS="${MBPP_INTERMEDIATE_STUDY_PROBLEMS:-50}"

RUN_REPAIR_ABLATIONS="${RUN_REPAIR_ABLATIONS:-1}"
RUN_BUDGET_ABLATIONS="${RUN_BUDGET_ABLATIONS:-1}"
RUN_INTERMEDIATE_REP_STUDY="${RUN_INTERMEDIATE_REP_STUDY:-1}"
RUN_SAS_CORRELATION="${RUN_SAS_CORRELATION:-0}"
RUN_MANUAL_AUDIT_PACK="${RUN_MANUAL_AUDIT_PACK:-0}"
RUN_SEMANTIC_SUBSET_ANALYSIS="${RUN_SEMANTIC_SUBSET_ANALYSIS:-0}"

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not find uv. Set UV_BIN=/absolute/path/to/uv." >&2
  exit 127
fi

if [[ "$INSTALL_EVALPLUS_IF_MISSING" == "1" ]]; then
  if ! "$UV_BIN" run python -c "import evalplus" >/dev/null 2>&1; then
    "$UV_BIN" pip install evalplus
  fi
fi

prepare_args=(--output_dir "$MBPP_DATASET_PATH")
if [[ "$FORCE_PREPARE_MBPP" == "1" || ! -f "$MBPP_DATASET_PATH/dataset_dict.json" ]]; then
  prepare_args+=(--force)
fi
if [[ -n "$MBPP_SOURCE_JSON" ]]; then
  prepare_args+=(--source_json "$MBPP_SOURCE_JSON")
fi

if [[ "$FORCE_PREPARE_MBPP" == "1" || ! -f "$MBPP_DATASET_PATH/dataset_dict.json" ]]; then
  "$UV_BIN" run python scripts/prepare_mbpp_plus_dataset.py "${prepare_args[@]}"
else
  echo "MBPP+ dataset already exists: $MBPP_DATASET_PATH"
fi

echo
echo "============================================================"
echo "MBPP+ revision suite: $MODEL_REPR"
echo "============================================================"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DATASET_PATH="$MBPP_DATASET_PATH" \
RELEASE_VERSION="mbpp_plus" \
MODEL_REPR="$MODEL_REPR" \
MAX_MODEL_LEN="$MBPP_MAX_MODEL_LEN" \
DTYPE="$DTYPE" \
TIMEOUT="$TIMEOUT" \
MEDIUM_PIPELINE_PROBLEMS="$MBPP_MEDIUM_PIPELINE_PROBLEMS" \
FULL_RELEASE_PROBLEMS="$MBPP_FULL_RELEASE_PROBLEMS" \
REPAIR_ABLATION_PROBLEMS="$MBPP_REPAIR_ABLATION_PROBLEMS" \
INTERMEDIATE_STUDY_PROBLEMS="$MBPP_INTERMEDIATE_STUDY_PROBLEMS" \
RUN_REPAIR_ABLATIONS="$RUN_REPAIR_ABLATIONS" \
RUN_BUDGET_ABLATIONS="$RUN_BUDGET_ABLATIONS" \
RUN_INTERMEDIATE_REP_STUDY="$RUN_INTERMEDIATE_REP_STUDY" \
RUN_SAS_CORRELATION="$RUN_SAS_CORRELATION" \
RUN_MANUAL_AUDIT_PACK="$RUN_MANUAL_AUDIT_PACK" \
RUN_SEMANTIC_SUBSET_ANALYSIS="$RUN_SEMANTIC_SUBSET_ANALYSIS" \
bash scripts/run_dual_loop_revision_suite.sh "$LOCAL_MODEL_PATH" "$MODEL_STYLE"

echo
echo "MBPP+ suite finished."
echo "MBPP+ dataset: $MBPP_DATASET_PATH"
echo "Outputs: output/dual_loop_rq_suite/, output/intermediate_repr_study/"
