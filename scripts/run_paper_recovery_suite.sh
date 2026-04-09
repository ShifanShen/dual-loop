#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]"
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct"
  echo
  echo "Optional env vars:"
  echo "  PYTHON_BIN=python3"
  echo "  AUTO_INSTALL_UV=1"
  echo "  SYNC_ENV=1"
  echo "  CLEAR_OUTPUT=0"
  echo "  GPU_ID=0"
  echo "  CODEGEN_NUM_CANDIDATES=3"
  echo "  SPEC_COUNTERFACTUAL_REPEATS=5"
  echo "  SPEC_COUNTERFACTUAL_TEMPERATURE=0.2"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOCAL_MODEL_PATH="$1"
MODEL_STYLE="${2:-CodeQwenInstruct}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTO_INSTALL_UV="${AUTO_INSTALL_UV:-1}"
SYNC_ENV="${SYNC_ENV:-1}"
CLEAR_OUTPUT="${CLEAR_OUTPUT:-0}"
GPU_ID="${GPU_ID:-0}"
UV_BIN="${UV_BIN:-}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/output/recovery_logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_ROOT/paper_recovery_${TIMESTAMP}.log"

mkdir -p "$LOG_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find Python executable: $PYTHON_BIN" >&2
  exit 127
fi

hash -r 2>/dev/null || true
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(type -P uv || true)"
fi

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  if [[ "$AUTO_INSTALL_UV" != "1" ]]; then
    echo "uv is not available and AUTO_INSTALL_UV=0." >&2
    echo "Install uv manually or rerun with AUTO_INSTALL_UV=1." >&2
    exit 127
  fi

  echo "uv not found. Installing it with $PYTHON_BIN -m pip install --user uv ..."
  "$PYTHON_BIN" -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
  hash -r 2>/dev/null || true
  UV_BIN="$(type -P uv || true)"
fi

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Could not resolve a usable uv binary after installation." >&2
  exit 127
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repository root does not exist: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "pyproject.toml not found under $REPO_ROOT" >&2
  exit 1
fi

if [[ "$SYNC_ENV" == "1" ]]; then
  echo "Synchronizing Python environment with uv ..."
  (
    cd "$REPO_ROOT"
    "$UV_BIN" sync
  )
fi

if [[ "$CLEAR_OUTPUT" == "1" ]]; then
  echo "Clearing previous output/ directory ..."
  rm -rf "$REPO_ROOT/output"
fi

mkdir -p "$REPO_ROOT/output"

echo "============================================================"
echo "Paper recovery suite"
echo "============================================================"
echo "repo_root=$REPO_ROOT"
echo "local_model_path=$LOCAL_MODEL_PATH"
echo "model_style=$MODEL_STYLE"
echo "python_bin=$PYTHON_BIN"
echo "uv_bin=$UV_BIN"
echo "sync_env=$SYNC_ENV"
echo "clear_output=$CLEAR_OUTPUT"
echo "gpu_id=$GPU_ID"
echo "log_path=$LOG_PATH"
echo "============================================================"

(
  cd "$REPO_ROOT"
  RUN_BUDGET_ABLATIONS="${RUN_BUDGET_ABLATIONS:-1}" \
  RUN_INTERMEDIATE_REP_STUDY="${RUN_INTERMEDIATE_REP_STUDY:-1}" \
  RUN_SAS_CORRELATION="${RUN_SAS_CORRELATION:-1}" \
  RUN_MANUAL_AUDIT_PACK="${RUN_MANUAL_AUDIT_PACK:-1}" \
  RUN_SPEC_COUNTERFACTUAL_SUITE="${RUN_SPEC_COUNTERFACTUAL_SUITE:-1}" \
  SPEC_COUNTERFACTUAL_SOURCE="${SPEC_COUNTERFACTUAL_SOURCE:-full}" \
  SPEC_COUNTERFACTUAL_REPEATS="${SPEC_COUNTERFACTUAL_REPEATS:-5}" \
  SPEC_COUNTERFACTUAL_TEMPERATURE="${SPEC_COUNTERFACTUAL_TEMPERATURE:-0.2}" \
  SPEC_COUNTERFACTUAL_LOW_SAS_THRESHOLD="${SPEC_COUNTERFACTUAL_LOW_SAS_THRESHOLD:-85}" \
  CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-3}" \
  GPU_ID="$GPU_ID" \
  UV_BIN="$UV_BIN" \
  bash scripts/run_dual_loop_paper_suite.sh "$LOCAL_MODEL_PATH" "$MODEL_STYLE"
) 2>&1 | tee "$LOG_PATH"

echo
echo "Paper recovery suite finished."
echo "Log:"
echo "  $LOG_PATH"
echo "Outputs:"
echo "  $REPO_ROOT/output/dual_loop/"
echo "  $REPO_ROOT/output/dual_loop_rq_suite/"
echo "  $REPO_ROOT/output/intermediate_repr_study/"
echo "  $REPO_ROOT/output/spec_counterfactual/"
