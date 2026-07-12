import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter
from lcb_runner.dual_loop.prompts import (
    build_code_from_plan_prompt,
    build_code_from_pseudocode_prompt,
    build_code_from_spec_prompt,
    build_direct_codegen_prompt,
    build_plan_draft_prompt,
    build_pseudocode_draft_prompt,
)
from lcb_runner.dual_loop.spec import StructuredSpec, VerifierFeedback


METHODS = (
    "raw_nl_irl",
    "plan_irl",
    "pseudocode_irl",
    "contract_text_irl",
    "semantic_contract_irl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fair IRL ablation that varies only the intermediate artifact: "
            "raw NL, plan, pseudocode, contract text, or semantic contract."
        )
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--model_style", type=str, default=None)
    parser.add_argument("--model_repr", type=str, default=None)
    parser.add_argument("--release_version", type=str, default="release_latest")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument(
        "--feedback_test_scope",
        choices=["public", "all"],
        default="public",
        help="Use public tests for iterative feedback; 'all' is legacy-only.",
    )
    parser.add_argument(
        "--final_test_scope",
        choices=["private", "all"],
        default="private",
        help="Use private tests once for final evaluation; 'all' is legacy-only.",
    )
    parser.add_argument("--question_ids", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=50)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS),
        help="Comma-separated subset of methods to run.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/contract_contribution_ablation",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "0") or 0),
    )
    parser.add_argument("--vllm_device", type=str, default=os.environ.get("VLLM_DEVICE"))
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--cache_batch_size", type=int, default=32)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop", type=str, default="###")
    parser.add_argument("--multiprocess", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num_process_evaluate", type=int, default=8)
    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.2)
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)
    parser.add_argument("--codegen_num_candidates", type=int, default=2)
    parser.add_argument("--repair_num_candidates", type=int, default=3)
    parser.add_argument("--repair_max_iters", type=int, default=3)
    parser.add_argument(
        "--include_artifact_cost_in_each_contract_variant",
        action="store_true",
        default=True,
        help=(
            "Count the shared spec-draft cost in both contract_text_irl and "
            "semantic_contract_irl budget summaries."
        ),
    )
    args = parser.parse_args()
    args.stop = args.stop.split(",")
    args.pipeline_mode = "baseline"
    args.run_tag = None
    args.cwd_output_dir = None
    args.disable_counterexample_repair = False
    args.disable_rewrite_repair = False
    args.spec_max_iters = 0
    args.spec_score_threshold = 90
    args.spec_min_improvement = 1
    args.spec_precision_floor = 85
    args.spec_max_rejected_refines = 1
    args.spec_skip_ambiguity_only = True
    args.adaptive_candidate_budget = False
    args.adaptive_codegen_max_candidates = args.codegen_num_candidates
    args.adaptive_repair_max_candidates = args.repair_num_candidates
    args.post_failure_sal_max_iters = 0
    args.contract_search_population_size = 1
    args.contract_search_rounds = 0
    args.contract_search_top_k = 1
    args.contract_search_codegen_top_k = 1
    args.contract_search_temperature = 0.35
    args.attribution_mode = "conservative"
    args.attribution_spec_margin = 3

    if args.local_model_path:
        inferred_name = Path(args.local_model_path).name or "LocalModel"
        if args.model is None:
            args.model = inferred_name
        if args.model_repr is None:
            args.model_repr = inferred_name
    elif args.model is None:
        args.model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    if args.tensor_parallel_size == -1:
        import torch

        args.tensor_parallel_size = torch.cuda.device_count()
    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()

    selected_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    invalid_methods = sorted(set(selected_methods) - set(METHODS))
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {', '.join(invalid_methods)}")
    args.selected_methods = selected_methods
    return args


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _usage_totals(usages: list[dict[str, int | str]]) -> dict[str, float]:
    valid_usages = [usage for usage in usages if usage]
    return {
        "llm_calls": float(len(valid_usages)),
        "prompt_chars": float(
            sum(int(usage.get("prompt_chars", 0) or 0) for usage in valid_usages)
        ),
        "completion_chars": float(
            sum(int(usage.get("completion_chars", 0) or 0) for usage in valid_usages)
        ),
    }


def _count_usages_by_role(usages: list[dict[str, int | str]], role_prefix: str) -> int:
    return sum(1 for usage in usages if str(usage.get("role", "")).startswith(role_prefix))


def _feedback_counts(feedbacks: list[VerifierFeedback]) -> dict[str, int]:
    property_count = 0
    violated_count = 0
    for feedback in feedbacks:
        property_count += len(feedback.property_feedbacks or [])
        violated_count += len(feedback.violated_spec_items or [])
    return {
        "property_feedback_count": property_count,
        "violated_spec_item_count": violated_count,
    }


def _feedback_payload(feedback: VerifierFeedback) -> dict[str, Any]:
    return asdict(feedback)


def _artifact_usage_from_spec(spec: StructuredSpec) -> list[dict[str, int | str]]:
    return [
        usage
        for usage in [getattr(spec, "_usage", None), *getattr(spec, "_extra_usages", [])]
        if usage
    ]


def _build_contract_text_codegen_prompt(problem, contract_text: str) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are an expert Python programmer.
Write a complete Python solution using the semantic contract below.
Return only one Python code block.

Original problem:
{problem.question_content}

Semantic contract:
{contract_text}
{starter}
Requirements:
- Follow the exact input/output protocol from the original problem.
- Use the contract as guidance, but do not assume it is executable verifier feedback.
- Do not print extra text.
- Prefer a direct, contest-style solution.
"""


def _build_artifact_repair_prompt(
    problem,
    *,
    method: str,
    artifact_text: str,
    code: str,
    feedback: VerifierFeedback,
    require_change: bool,
) -> str:
    labels = {
        "raw_nl_irl": "No separate intermediate artifact is provided.",
        "plan_irl": "Natural-language solution plan",
        "pseudocode_irl": "Pseudocode",
        "contract_text_irl": "Semantic contract text",
        "semantic_contract_irl": "Semantic contract",
    }
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    artifact_block = labels[method]
    if artifact_text:
        artifact_block = f"{labels[method]}:\n{artifact_text}"
    change_requirement = ""
    if require_change:
        change_requirement = (
            "\n- The previous repair attempt did not fix the failure."
            "\n- Do not return the same program again."
            "\n- Change the logic that causes the reported behavior."
        )
    semantic_priority = ""
    if method == "semantic_contract_irl":
        semantic_priority = (
            "\n- If violated_spec_items or property_feedbacks are present, treat them "
            "as high-priority semantic obligations."
        )
    return f"""You are repairing a Python program for a competitive programming problem.
Return exactly one complete fixed Python program in one fenced code block.

Original problem:
{problem.question_content}
{starter}
Intermediate artifact:
{artifact_block}

Current code:
```python
{code}
```

Verifier feedback:
{feedback.to_json()}

Repair requirements:
- Fix the reported failure.
- Preserve the input/output contract.
- Prefer a direct, contest-style solution.
{semantic_priority}
{change_requirement}
"""


def _initial_codegen(
    pipeline: DualLoopPipeline,
    problem,
    *,
    method: str,
    artifact_text: str,
    spec_for_feedback: StructuredSpec | None,
) -> dict[str, Any]:
    if method == "raw_nl_irl":
        prompt = build_direct_codegen_prompt(problem)
    elif method == "plan_irl":
        prompt = build_code_from_plan_prompt(problem, artifact_text)
    elif method == "pseudocode_irl":
        prompt = build_code_from_pseudocode_prompt(problem, artifact_text)
    elif method == "contract_text_irl":
        prompt = _build_contract_text_codegen_prompt(problem, artifact_text)
    elif method == "semantic_contract_irl":
        if spec_for_feedback is None:
            raise ValueError("semantic_contract_irl requires a StructuredSpec")
        prompt = build_code_from_spec_prompt(problem, spec_for_feedback)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return pipeline._select_best_codegen_candidate(
        problem,
        prompt=prompt,
        spec=spec_for_feedback,
    )


def _repair_with_artifact(
    pipeline: DualLoopPipeline,
    problem,
    *,
    method: str,
    artifact_text: str,
    initial_code: str,
    initial_feedback: VerifierFeedback,
    spec_for_feedback: StructuredSpec | None,
) -> dict[str, Any]:
    current_code = initial_code
    current_feedback = initial_feedback
    selected_feedbacks = [initial_feedback]
    repair_steps: list[dict[str, Any]] = []
    raw_repair_outputs: list[str] = []
    usages: list[dict[str, int | str]] = []
    stagnant_attempts = 0
    repeated_failure_count = 0
    seen_failure_signatures: dict[str, int] = {}

    for attempt_index in range(int(pipeline.args.repair_max_iters or 0)):
        if current_feedback.passed:
            break

        signature = json.dumps(
            {
                "error_type": current_feedback.error_type,
                "input": current_feedback.input,
                "output": current_feedback.output,
                "expected": current_feedback.expected,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        repeated_failure_count = seen_failure_signatures.get(signature, 0)
        seen_failure_signatures[signature] = repeated_failure_count + 1

        repair_candidate_count = pipeline._repair_candidate_count_for_feedback(
            current_feedback,
            stagnant_attempts=stagnant_attempts,
            repeated_failure_count=repeated_failure_count,
        )
        candidate_records: list[dict[str, Any]] = []
        started_at = time.perf_counter()

        for candidate_index in range(repair_candidate_count):
            prompt = _build_artifact_repair_prompt(
                problem,
                method=method,
                artifact_text=artifact_text,
                code=current_code,
                feedback=current_feedback,
                require_change=stagnant_attempts > 0 or repeated_failure_count > 0,
            )
            repair_output, usage = pipeline.llm.generate(
                prompt,
                role=f"{method}_repair",
                temperature=min(
                    0.8,
                    float(pipeline.args.repair_temperature or 0.0)
                    + 0.05 * candidate_index
                    + 0.05 * stagnant_attempts,
                ),
                max_tokens=pipeline.args.codegen_max_tokens,
            )
            raw_repair_outputs.append(repair_output)
            usages.append(usage)
            next_code, extra_outputs, extra_usages = pipeline._extract_valid_code(
                repair_output
            )
            raw_repair_outputs.extend(extra_outputs)
            usages.extend(extra_usages)

            if not next_code:
                candidate_feedback = pipeline._invalid_codegen_feedback(
                    "Repair candidate could not be parsed into valid Python code."
                )
                reason = "invalid_candidate"
            elif next_code.strip() == current_code.strip():
                candidate_feedback = pipeline._invalid_codegen_feedback(
                    "Repair candidate repeated the current failing program."
                )
                reason = "unchanged_candidate"
            else:
                candidate_feedback = pipeline._verify(
                    problem,
                    next_code,
                    spec=spec_for_feedback,
                )
                reason = "candidate_changed"

            candidate_records.append(
                {
                    "candidate_index": candidate_index + 1,
                    "code": next_code,
                    "raw_output": repair_output,
                    "feedback": candidate_feedback,
                    "reason": reason,
                }
            )
            if candidate_feedback.passed:
                break

        selected_record = min(
            candidate_records,
            key=lambda record: pipeline._candidate_feedback_rank(record["feedback"]),
        )
        selected_feedback = selected_record["feedback"]
        selected_reason = str(selected_record["reason"])
        step = {
            "attempt_index": attempt_index + 1,
            "candidate_budget": repair_candidate_count,
            "candidate_count": len(candidate_records),
            "selected_candidate_index": int(selected_record["candidate_index"]),
            "selected_reason": selected_reason,
            "selected_passed": bool(selected_feedback.passed),
            "selected_error_type": selected_feedback.error_type,
            "elapsed_seconds": time.perf_counter() - started_at,
            "candidate_feedbacks": [
                {
                    **_feedback_payload(record["feedback"]),
                    "candidate_index": int(record["candidate_index"]),
                    "reason": record["reason"],
                    "selection_summary": pipeline._candidate_feedback_summary(
                        record["feedback"]
                    ),
                }
                for record in candidate_records
            ],
        }
        repair_steps.append(step)
        selected_feedbacks.append(selected_feedback)

        if selected_reason != "candidate_changed":
            stagnant_attempts += 1
            current_feedback = selected_feedback
            continue

        previous_code = current_code
        current_code = str(selected_record["code"])
        current_feedback = selected_feedback
        if current_code.strip() == previous_code.strip():
            stagnant_attempts += 1
        else:
            stagnant_attempts = 0
        if current_feedback.passed:
            break

    return {
        "final_code": current_code,
        "final_feedback": current_feedback,
        "selected_feedbacks": selected_feedbacks,
        "repair_steps": repair_steps,
        "raw_repair_outputs": raw_repair_outputs,
        "repair_usages": usages,
        "repair_iterations": len(repair_steps),
    }


def _draft_text_artifact(
    pipeline: DualLoopPipeline,
    problem,
    *,
    method: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if method == "plan_irl":
        prompt = build_plan_draft_prompt(problem)
        role = "plan_draft"
    elif method == "pseudocode_irl":
        prompt = build_pseudocode_draft_prompt(problem)
        role = "pseudocode_draft"
    else:
        raise ValueError(f"Unsupported text artifact method: {method}")

    output, usage = pipeline.llm.generate(
        prompt,
        role=role,
        temperature=pipeline.args.spec_temperature,
        max_tokens=pipeline.args.spec_max_tokens,
    )
    return {
        "artifact_text": output.strip(),
        "artifact_payload": None,
        "artifact_usages": [usage],
        "artifact_elapsed_seconds": time.perf_counter() - started_at,
    }


def _draft_spec_artifact(pipeline: DualLoopPipeline, problem) -> dict[str, Any]:
    spec = pipeline._draft_spec(problem)
    return {
        "spec": spec,
        "artifact_text": spec.to_text(),
        "artifact_payload": asdict(spec),
        "artifact_usages": _artifact_usage_from_spec(spec),
        "artifact_elapsed_seconds": float(getattr(spec, "_stage_time", 0.0)),
    }


def _run_method(
    pipeline: DualLoopPipeline,
    problem,
    *,
    method: str,
    artifact_text: str,
    artifact_payload: dict[str, Any] | None,
    artifact_usages: list[dict[str, int | str]],
    artifact_elapsed_seconds: float,
    spec_for_feedback: StructuredSpec | None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    codegen_result = _initial_codegen(
        pipeline,
        problem,
        method=method,
        artifact_text=artifact_text,
        spec_for_feedback=spec_for_feedback,
    )
    initial_code = str(codegen_result["code"])
    initial_feedback = codegen_result["feedback"]
    codegen_usages = [
        codegen_result["usage"],
        *codegen_result["extra_usages"],
    ]

    repair_result = {
        "final_code": initial_code,
        "final_feedback": initial_feedback,
        "selected_feedbacks": [initial_feedback],
        "repair_steps": [],
        "raw_repair_outputs": [],
        "repair_usages": [],
        "repair_iterations": 0,
    }
    if not initial_feedback.passed and int(pipeline.args.repair_max_iters or 0) > 0:
        repair_result = _repair_with_artifact(
            pipeline,
            problem,
            method=method,
            artifact_text=artifact_text,
            initial_code=initial_code,
            initial_feedback=initial_feedback,
            spec_for_feedback=spec_for_feedback,
        )

    final_feedback = repair_result["final_feedback"]
    selected_feedbacks = list(repair_result["selected_feedbacks"])
    feedback_stats = _feedback_counts(selected_feedbacks)
    all_usages = [
        *artifact_usages,
        *codegen_usages,
        *repair_result["repair_usages"],
    ]
    usage_stats = _usage_totals(all_usages)

    return {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "method": method,
        "artifact_type": method.replace("_irl", ""),
        "artifact_text": artifact_text,
        "artifact_payload": artifact_payload,
        "code_initial": initial_code,
        "final_code": repair_result["final_code"],
        "initial_passed": bool(initial_feedback.passed),
        "passed": bool(final_feedback.passed),
        "initial_feedback": _feedback_payload(initial_feedback),
        "final_feedback": _feedback_payload(final_feedback),
        "selected_feedbacks": [_feedback_payload(feedback) for feedback in selected_feedbacks],
        "codegen_candidate_count": int(codegen_result["candidate_count"]),
        "codegen_selected_candidate_index": int(codegen_result["selected_candidate_index"]),
        "codegen_candidate_feedbacks": codegen_result["candidate_feedbacks"],
        "raw_codegen_output": codegen_result["raw_output"],
        "raw_repair_outputs": repair_result["raw_repair_outputs"],
        "repair_steps": repair_result["repair_steps"],
        "repair_iterations": int(repair_result["repair_iterations"]),
        "artifact_elapsed_seconds": artifact_elapsed_seconds,
        "elapsed_seconds": time.perf_counter() - started_at + artifact_elapsed_seconds,
        "artifact_calls": len(artifact_usages),
        "codegen_calls": _count_usages_by_role(codegen_usages, "codegen"),
        "repair_calls": len(repair_result["repair_usages"]),
        **usage_stats,
        **feedback_stats,
    }


def _aggregate(method: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {
            "method": method,
            "num_problems": 0,
            "pass_at_1": 0.0,
            "solved_count": 0,
            "initial_pass_count": 0,
            "repair_solved_count": 0,
            "average_llm_calls": 0.0,
            "average_codegen_calls": 0.0,
            "average_repair_calls": 0.0,
            "average_prompt_chars": 0.0,
            "average_completion_chars": 0.0,
            "average_elapsed_seconds": 0.0,
            "average_property_feedback_count": 0.0,
            "average_violated_spec_item_count": 0.0,
        }
    solved_count = sum(1 for trace in traces if trace["passed"])
    initial_pass_count = sum(1 for trace in traces if trace["initial_passed"])
    return {
        "method": method,
        "num_problems": len(traces),
        "pass_at_1": round(solved_count / len(traces), 6),
        "solved_count": solved_count,
        "initial_pass_count": initial_pass_count,
        "repair_solved_count": solved_count - initial_pass_count,
        "average_llm_calls": round(mean(trace["llm_calls"] for trace in traces), 6),
        "average_codegen_calls": round(mean(trace["codegen_calls"] for trace in traces), 6),
        "average_repair_calls": round(mean(trace["repair_calls"] for trace in traces), 6),
        "average_prompt_chars": round(mean(trace["prompt_chars"] for trace in traces), 6),
        "average_completion_chars": round(
            mean(trace["completion_chars"] for trace in traces), 6
        ),
        "average_elapsed_seconds": round(
            mean(trace["elapsed_seconds"] for trace in traces), 6
        ),
        "average_property_feedback_count": round(
            mean(trace["property_feedback_count"] for trace in traces), 6
        ),
        "average_violated_spec_item_count": round(
            mean(trace["violated_spec_item_count"] for trace in traces), 6
        ),
    }


def _add_summary_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    semantic_pass = float(by_method.get("semantic_contract_irl", {}).get("pass_at_1", 0.0))
    comparison_methods = (
        "raw_nl_irl",
        "plan_irl",
        "pseudocode_irl",
        "contract_text_irl",
    )
    for row in rows:
        for comparison in comparison_methods:
            baseline_pass = float(by_method.get(comparison, {}).get("pass_at_1", 0.0))
            row[f"delta_semantic_contract_vs_{comparison}"] = round(
                semantic_pass - baseline_pass,
                6,
            )
    return rows


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "num_problems",
        "pass_at_1",
        "solved_count",
        "initial_pass_count",
        "repair_solved_count",
        "average_llm_calls",
        "average_codegen_calls",
        "average_repair_calls",
        "average_prompt_chars",
        "average_completion_chars",
        "average_elapsed_seconds",
        "average_property_feedback_count",
        "average_violated_spec_item_count",
        "delta_semantic_contract_vs_raw_nl_irl",
        "delta_semantic_contract_vs_plan_irl",
        "delta_semantic_contract_vs_pseudocode_irl",
        "delta_semantic_contract_vs_contract_text_irl",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _budget_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "matching_axis": "downstream_codegen_repair",
        "methods": args.selected_methods,
        "codegen_num_candidates": args.codegen_num_candidates,
        "repair_max_iters": args.repair_max_iters,
        "repair_num_candidates": args.repair_num_candidates,
        "feedback_test_scope": args.feedback_test_scope,
        "final_test_scope": args.final_test_scope,
        "adaptive_candidate_budget": False,
        "sal_refinement": "disabled",
        "contract_search": "disabled",
        "selection_policy": "existing verifier-aware candidate rank",
        "semantic_feedback_enabled_only_for": "semantic_contract_irl",
        "matched_resources": [
            "code generation candidates",
            "repair iterations",
            "repair candidates per iteration",
            "feedback verifier scope",
        ],
        "unmatched_overhead": (
            "Intermediate-artifact construction calls are method overhead and are "
            "reported explicitly; this is a mechanism-matched, not total-cost-matched, comparison."
        ),
    }


def _apply_heldout_final_evaluation(
    pipeline: DualLoopPipeline,
    benchmark: list[Any],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    generations = [[str(trace.get("final_code", ""))] for trace in traces]
    metrics = pipeline._compute_metrics(benchmark, generations)
    final_results = metrics[1] if len(metrics) > 1 else {}

    for index, trace in enumerate(traces):
        per_generation = final_results.get(index, [])
        first_generation = per_generation[0] if per_generation else []
        if not isinstance(first_generation, list):
            first_generation = [first_generation]
        final_passed = bool(first_generation) and all(
            result is True for result in first_generation
        )
        trace["feedback_passed"] = bool(trace.get("passed", False))
        trace["passed"] = final_passed
        trace["final_evaluation"] = {
            "scope": args_final_scope(pipeline.args),
            "passed": final_passed,
        }

    return {"pass@1": float(metrics[0]["pass@1"])}


def args_final_scope(args: argparse.Namespace) -> str:
    return str(getattr(args, "final_test_scope", "private") or "private")


def main() -> None:
    args = parse_args()
    llm = LLMAdapter(args)
    pipeline = DualLoopPipeline(args, llm=llm)
    benchmark = pipeline._load_benchmark()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_repr = (args.model_repr or args.model or "model").replace("/", "_").replace(" ", "_")
    output_dir = Path(args.output_root) / f"contract_ablation_{model_repr}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in args.selected_methods
    }

    for problem in benchmark:
        text_artifacts: dict[str, dict[str, Any]] = {}
        if "plan_irl" in args.selected_methods:
            text_artifacts["plan_irl"] = _draft_text_artifact(
                pipeline, problem, method="plan_irl"
            )
        if "pseudocode_irl" in args.selected_methods:
            text_artifacts["pseudocode_irl"] = _draft_text_artifact(
                pipeline, problem, method="pseudocode_irl"
            )

        spec_artifact: dict[str, Any] | None = None
        if (
            "contract_text_irl" in args.selected_methods
            or "semantic_contract_irl" in args.selected_methods
        ):
            spec_artifact = _draft_spec_artifact(pipeline, problem)

        for method in args.selected_methods:
            artifact_text = ""
            artifact_payload = None
            artifact_usages: list[dict[str, int | str]] = []
            artifact_elapsed_seconds = 0.0
            spec_for_feedback = None

            if method in {"plan_irl", "pseudocode_irl"}:
                artifact = text_artifacts[method]
                artifact_text = artifact["artifact_text"]
                artifact_payload = artifact["artifact_payload"]
                artifact_usages = artifact["artifact_usages"]
                artifact_elapsed_seconds = artifact["artifact_elapsed_seconds"]
            elif method in {"contract_text_irl", "semantic_contract_irl"}:
                if spec_artifact is None:
                    raise RuntimeError("Missing shared spec artifact")
                artifact_text = spec_artifact["artifact_text"]
                artifact_payload = spec_artifact["artifact_payload"]
                artifact_usages = spec_artifact["artifact_usages"]
                artifact_elapsed_seconds = spec_artifact["artifact_elapsed_seconds"]
                if method == "semantic_contract_irl":
                    spec_for_feedback = spec_artifact["spec"]

            trace = _run_method(
                pipeline,
                problem,
                method=method,
                artifact_text=artifact_text,
                artifact_payload=artifact_payload,
                artifact_usages=artifact_usages,
                artifact_elapsed_seconds=artifact_elapsed_seconds,
                spec_for_feedback=spec_for_feedback,
            )
            traces_by_method[method].append(trace)

    final_metrics_by_method = {
        method: _apply_heldout_final_evaluation(pipeline, benchmark, traces)
        for method, traces in traces_by_method.items()
    }

    rows = _add_summary_deltas(
        [_aggregate(method, traces) for method, traces in traces_by_method.items()]
    )
    results_csv = output_dir / "results.csv"
    _write_results_csv(results_csv, rows)

    for method, traces in traces_by_method.items():
        _write_json(output_dir / f"{method}_traces.json", traces)

    budget_protocol = _budget_protocol(args)
    _write_json(output_dir / "budget_protocol.json", budget_protocol)

    summary = {
        "model": args.model,
        "model_repr": args.model_repr,
        "local_model_path": args.local_model_path,
        "release_version": args.release_version,
        "max_problems": args.max_problems,
        "methods": args.selected_methods,
        "output_dir": str(output_dir),
        "results_csv": str(results_csv),
        "budget_protocol": budget_protocol,
        "final_metrics_by_method": final_metrics_by_method,
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
