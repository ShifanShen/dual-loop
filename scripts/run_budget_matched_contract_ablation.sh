#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local_model_path> [model_style]"
  echo "Example: $0 /models/Qwen2.5-Coder-7B-Instruct CodeQwenInstruct"
  exit 1
fi

# This is a mechanism-matched comparison. Every method receives the same
# downstream code-generation and repair budget. Artifact construction remains
# method overhead and is reported rather than hidden or compensated away.
export METHODS="${METHODS:-raw_nl_irl,contract_text_irl,semantic_contract_irl}"
export CODEGEN_NUM_CANDIDATES="${CODEGEN_NUM_CANDIDATES:-2}"
export REPAIR_NUM_CANDIDATES="${REPAIR_NUM_CANDIDATES:-3}"
export REPAIR_MAX_ITERS="${REPAIR_MAX_ITERS:-3}"
export MAX_PROBLEMS="${MAX_PROBLEMS:-200}"

echo "Running mechanism-matched contract ablation."
echo "Matched: code candidates, repair iterations, repair candidates, feedback verifier."
echo "Reported but not matched: artifact/SAL construction calls and total elapsed cost."

exec bash scripts/run_contract_contribution_ablation.sh "$@"
