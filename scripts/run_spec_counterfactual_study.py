import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.diagnostics import load_run_artifacts
from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter
from lcb_runner.dual_loop.spec import StructuredSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a counterfactual study that compares code generation under "
            "initial vs final accepted specs from an existing dual-loop run."
        )
    )
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument(
        "--run_name",
        type=str,
        default="full_dual_loop",
        help="Run entry inside the suite manifest used as the source of initial/final specs.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=[
            "all",
            "spec_changed",
            "low_initial_sas",
            "changed_and_low_initial_sas",
            "sal_attempted",
            "initial_codegen_failed",
            "changed_and_initial_codegen_failed",
            "changed_low_sas_and_initial_codegen_failed",
            "source_spec_induced",
            "resolved_by_loop_b",
            "resolved_by_loop_b_and_initial_codegen_failed",
        ],
        help="Filter traces before rerunning code generation.",
    )
    parser.add_argument(
        "--low_initial_sas_threshold",
        type=float,
        default=85.0,
        help="Threshold used by low_initial_sas subset filters.",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=0,
        help="Optional cap after subset filtering. Use 0 for all selected problems.",
    )
    parser.add_argument(
        "--allow_empty",
        action="store_true",
        help="If set, emit a skipped summary instead of failing when the requested subset has no matching traces.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated codegen runs per spec variant and problem.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--model_style", type=str, default=None)
    parser.add_argument("--model_repr", type=str, default=None)
    parser.add_argument("--release_version", type=str, default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="output/spec_counterfactual")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
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
    parser.add_argument("--spec_max_iters", type=int, default=3)
    parser.add_argument("--repair_max_iters", type=int, default=3)
    parser.add_argument("--spec_score_threshold", type=int, default=90)
    parser.add_argument("--spec_min_improvement", type=int, default=1)
    parser.add_argument("--spec_precision_floor", type=int, default=85)
    parser.add_argument("--spec_max_rejected_refines", type=int, default=1)
    parser.add_argument("--spec_skip_ambiguity_only", action="store_true")
    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument(
        "--codegen_temperature",
        type=float,
        default=0.0,
        help="Defaults to greedy decoding to reduce sampling noise in the counterfactual comparison.",
    )
    parser.add_argument(
        "--codegen_num_candidates",
        type=int,
        default=1,
        help="Number of candidates to generate under each spec variant before selecting the best one.",
    )
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)
    args = parser.parse_args()

    args.stop = args.stop.split(",")
    args.pipeline_mode = "baseline"
    args.disable_counterexample_repair = False
    args.disable_rewrite_repair = False
    args.spec_skip_ambiguity_only = True
    args.cwd_output_dir = None
    args.run_tag = None

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
    return args


def _resolve_suite_dir(raw_suite_dir: str) -> Path:
    suite_dir = Path(raw_suite_dir)
    candidates: list[Path] = []

    if suite_dir.is_absolute():
        candidates.append(suite_dir)
    else:
        candidates.extend(
            [
                ROOT / suite_dir,
                ROOT / "output" / "dual_loop_rq_suite" / suite_dir,
                ROOT / "assets" / "output" / "dual_loop_rq_suite" / suite_dir,
            ]
        )

    if not suite_dir.is_absolute():
        normalized = str(suite_dir).replace("\\", "/")
        if normalized.startswith("output/") or normalized.startswith("assets/output/"):
            candidates.insert(0, ROOT / suite_dir)

    for candidate in candidates:
        if (candidate / "run_manifest.json").exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find run_manifest.json for the requested suite. "
        f"Tried:\n{searched}"
    )


def _load_manifest(suite_dir: Path) -> list[dict[str, Any]]:
    with open(suite_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_suite_metadata(suite_dir: Path) -> dict[str, Any]:
    metadata_path = suite_dir / "suite_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _usage_totals(usages: list[dict[str, int | str]]) -> dict[str, float]:
    prompt_chars = sum(int(item.get("prompt_chars", 0)) for item in usages)
    completion_chars = sum(int(item.get("completion_chars", 0)) for item in usages)
    return {
        "llm_calls": float(len(usages)),
        "prompt_chars": float(prompt_chars),
        "completion_chars": float(completion_chars),
    }


def _spec_from_payload(payload: dict[str, Any]) -> StructuredSpec:
    return StructuredSpec(**payload)


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _source_initial_codegen_failed(trace: dict[str, Any]) -> bool:
    verifier_feedbacks = trace.get("verifier_feedbacks") or []
    if not verifier_feedbacks:
        return False
    first_feedback = verifier_feedbacks[0] or {}
    return not bool(first_feedback.get("passed", False))


def _source_failure_attribution(trace: dict[str, Any]) -> str:
    return str(trace.get("failure_attribution", "") or "")


def _source_resolved_by_loop_b(trace: dict[str, Any]) -> bool:
    tags = trace.get("failure_reason_tags") or []
    return "resolved_by_loop_b" in tags


def _trace_selected(trace: dict[str, Any], *, subset: str, low_initial_sas_threshold: float) -> bool:
    initial_score = trace.get("initial_spec_score") or {}
    initial_sas = float(initial_score.get("overall", 0.0) or 0.0)
    spec_changed = (trace.get("spec_initial") or {}) != (trace.get("spec_final") or {})
    sal_attempted = bool(((trace.get("effectiveness") or {}).get("spec_refine_steps") or []))
    initial_codegen_failed = _source_initial_codegen_failed(trace)
    source_spec_induced = _source_failure_attribution(trace) == "spec_induced"
    resolved_by_loop_b = _source_resolved_by_loop_b(trace)

    if subset == "all":
        return True
    if subset == "spec_changed":
        return spec_changed
    if subset == "low_initial_sas":
        return initial_sas < low_initial_sas_threshold
    if subset == "changed_and_low_initial_sas":
        return spec_changed and initial_sas < low_initial_sas_threshold
    if subset == "sal_attempted":
        return sal_attempted
    if subset == "initial_codegen_failed":
        return initial_codegen_failed
    if subset == "changed_and_initial_codegen_failed":
        return spec_changed and initial_codegen_failed
    if subset == "changed_low_sas_and_initial_codegen_failed":
        return spec_changed and initial_sas < low_initial_sas_threshold and initial_codegen_failed
    if subset == "source_spec_induced":
        return source_spec_induced
    if subset == "resolved_by_loop_b":
        return resolved_by_loop_b
    if subset == "resolved_by_loop_b_and_initial_codegen_failed":
        return resolved_by_loop_b and initial_codegen_failed
    return True


def _run_spec_variant(
    pipeline: DualLoopPipeline,
    problem,
    *,
    spec_payload: dict[str, Any],
    variant: str,
    repeats: int,
) -> dict[str, Any]:
    attempt_records: list[dict[str, Any]] = []

    for repeat_idx in range(repeats):
        spec = _spec_from_payload(spec_payload)
        started = time.perf_counter()
        code = pipeline._generate_code_from_spec(problem, spec)
        feedback = pipeline._verify(problem, code, spec=spec)
        usages = [getattr(spec, "_last_codegen_usage", None)] + list(
            getattr(spec, "_last_codegen_extra_usages", [])
        )
        usages = [item for item in usages if item]
        usage_stats = _usage_totals(usages)
        attempt_records.append(
            {
                "variant": variant,
                "repeat_index": repeat_idx + 1,
                "passed": bool(feedback.passed),
                "final_code": code,
                "verifier_feedback": asdict(feedback),
                "elapsed_seconds": time.perf_counter() - started,
                **usage_stats,
            }
        )

    pass_rate = mean(float(item["passed"]) for item in attempt_records) if attempt_records else 0.0
    return {
        "variant": variant,
        "repeats": repeats,
        "pass_rate": round(pass_rate, 4),
        "pass_count": int(sum(int(item["passed"]) for item in attempt_records)),
        "average_llm_calls": round(mean(item["llm_calls"] for item in attempt_records), 2) if attempt_records else 0.0,
        "average_prompt_chars": round(mean(item["prompt_chars"] for item in attempt_records), 2) if attempt_records else 0.0,
        "average_completion_chars": round(mean(item["completion_chars"] for item in attempt_records), 2) if attempt_records else 0.0,
        "average_elapsed_seconds": round(mean(item["elapsed_seconds"] for item in attempt_records), 2) if attempt_records else 0.0,
        "attempts": attempt_records,
    }


def _problem_row(trace: dict[str, Any], initial_result: dict[str, Any], final_result: dict[str, Any]) -> dict[str, Any]:
    initial_score = trace.get("initial_spec_score") or {}
    final_score = trace.get("final_spec_score") or {}
    spec_changed = (trace.get("spec_initial") or {}) != (trace.get("spec_final") or {})
    sal_attempted = bool(((trace.get("effectiveness") or {}).get("spec_refine_steps") or []))
    initial_codegen_failed = _source_initial_codegen_failed(trace)
    source_failure_attribution = _source_failure_attribution(trace)
    resolved_by_loop_b = _source_resolved_by_loop_b(trace)
    delta_pass_rate = round(float(final_result["pass_rate"]) - float(initial_result["pass_rate"]), 4)
    outcome = "tie"
    if delta_pass_rate > 0:
        outcome = "final_better"
    elif delta_pass_rate < 0:
        outcome = "initial_better"
    first_code_identical = False
    if initial_result["attempts"] and final_result["attempts"]:
        first_code_identical = (
            initial_result["attempts"][0].get("final_code", "")
            == final_result["attempts"][0].get("final_code", "")
        )

    return {
        "question_id": trace.get("question_id", ""),
        "question_title": trace.get("question_title", ""),
        "initial_sas": float(initial_score.get("overall", 0.0) or 0.0),
        "final_sas": float(final_score.get("overall", 0.0) or 0.0),
        "delta_sas": round(float(final_score.get("overall", 0.0) or 0.0) - float(initial_score.get("overall", 0.0) or 0.0), 4),
        "spec_changed": spec_changed,
        "sal_attempted": sal_attempted,
        "source_initial_codegen_failed": initial_codegen_failed,
        "source_failure_attribution": source_failure_attribution,
        "source_resolved_by_loop_b": resolved_by_loop_b,
        "initial_pass_rate": initial_result["pass_rate"],
        "final_pass_rate": final_result["pass_rate"],
        "delta_pass_rate": delta_pass_rate,
        "outcome": outcome,
        "first_code_identical": first_code_identical,
        "initial_average_llm_calls": initial_result["average_llm_calls"],
        "final_average_llm_calls": final_result["average_llm_calls"],
        "initial_average_elapsed_seconds": initial_result["average_elapsed_seconds"],
        "final_average_elapsed_seconds": final_result["average_elapsed_seconds"],
        "initial_first_error_type": (
            (initial_result["attempts"][0]["verifier_feedback"] or {}).get("error_type", "")
            if initial_result["attempts"]
            else ""
        ),
        "final_first_error_type": (
            (final_result["attempts"][0]["verifier_feedback"] or {}).get("error_type", "")
            if final_result["attempts"]
            else ""
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_problems": 0,
            "initial_pass_at_1": 0.0,
            "final_pass_at_1": 0.0,
            "delta_pass_at_1": 0.0,
            "final_better_count": 0,
            "initial_better_count": 0,
            "tie_count": 0,
            "spec_changed_count": 0,
            "sal_attempted_count": 0,
            "source_initial_codegen_failed_count": 0,
            "source_spec_induced_count": 0,
            "source_resolved_by_loop_b_count": 0,
            "first_code_identical_count": 0,
            "average_initial_sas": 0.0,
            "average_final_sas": 0.0,
            "average_delta_sas": 0.0,
        }

    outcome_counts = Counter(str(row["outcome"]) for row in rows)
    return {
        "num_problems": len(rows),
        "initial_pass_at_1": round(mean(float(row["initial_pass_rate"]) for row in rows), 4),
        "final_pass_at_1": round(mean(float(row["final_pass_rate"]) for row in rows), 4),
        "delta_pass_at_1": round(
            mean(float(row["final_pass_rate"]) - float(row["initial_pass_rate"]) for row in rows),
            4,
        ),
        "final_better_count": int(outcome_counts.get("final_better", 0)),
        "initial_better_count": int(outcome_counts.get("initial_better", 0)),
        "tie_count": int(outcome_counts.get("tie", 0)),
        "spec_changed_count": int(sum(1 for row in rows if row["spec_changed"])),
        "sal_attempted_count": int(sum(1 for row in rows if row["sal_attempted"])),
        "source_initial_codegen_failed_count": int(
            sum(1 for row in rows if row["source_initial_codegen_failed"])
        ),
        "source_spec_induced_count": int(
            sum(1 for row in rows if row["source_failure_attribution"] == "spec_induced")
        ),
        "source_resolved_by_loop_b_count": int(
            sum(1 for row in rows if row["source_resolved_by_loop_b"])
        ),
        "first_code_identical_count": int(sum(1 for row in rows if row["first_code_identical"])),
        "average_initial_sas": round(mean(float(row["initial_sas"]) for row in rows), 4),
        "average_final_sas": round(mean(float(row["final_sas"]) for row in rows), 4),
        "average_delta_sas": round(mean(float(row["delta_sas"]) for row in rows), 4),
    }


def main() -> None:
    args = parse_args()
    suite_dir = _resolve_suite_dir(args.suite_dir)
    manifest = _load_manifest(suite_dir)
    suite_metadata = _load_suite_metadata(suite_dir)
    run_entries = {entry["run_name"]: entry for entry in manifest}
    if args.run_name not in run_entries:
        available = ", ".join(sorted(run_entries))
        raise KeyError(f"Run '{args.run_name}' not found in manifest. Available: {available}")

    run_entry = run_entries[args.run_name]
    summary, traces = load_run_artifacts(
        _resolve_repo_path(run_entry["summary_path"]),
        _resolve_repo_path(run_entry["traces_path"]),
    )

    if args.release_version is None:
        args.release_version = str(summary.get("release_version") or suite_metadata.get("release_version") or "release_latest")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_repr = (args.model_repr or args.model or "model").replace("/", "_").replace(" ", "_")
    output_dir = Path(args.output_root) / f"spec_counterfactual_{model_repr}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_traces = [
        trace
        for trace in traces
        if trace.get("spec_initial") and trace.get("spec_final")
        and _trace_selected(
            trace,
            subset=args.subset,
            low_initial_sas_threshold=args.low_initial_sas_threshold,
        )
    ]
    if args.max_problems and args.max_problems > 0:
        filtered_traces = filtered_traces[: args.max_problems]
    if not filtered_traces:
        summary_payload = {
            "status": "skipped_empty_subset" if args.allow_empty else "error_empty_subset",
            "source_suite_dir": str(suite_dir),
            "source_run_name": args.run_name,
            "source_pipeline_mode": run_entry.get("pipeline_mode", ""),
            "source_summary_path": run_entry.get("summary_path", ""),
            "source_traces_path": run_entry.get("traces_path", ""),
            "release_version": args.release_version,
            "model": args.model,
            "model_repr": args.model_repr,
            "local_model_path": args.local_model_path,
            "subset": args.subset,
            "low_initial_sas_threshold": args.low_initial_sas_threshold,
            "repeats": args.repeats,
            "codegen_num_candidates": args.codegen_num_candidates,
            "codegen_temperature": args.codegen_temperature,
            "num_selected_problems": 0,
            "output_dir": str(output_dir),
            "results_csv": "",
            "slice_summaries": {},
            "message": "No traces matched the requested subset/filter configuration.",
        }
        _write_json(output_dir / "summary.json", summary_payload)
        print(json.dumps(summary_payload, indent=2, ensure_ascii=True))
        if args.allow_empty:
            return
        raise ValueError("No traces matched the requested subset/filter configuration.")

    wanted_ids = [str(trace.get("question_id", "")) for trace in filtered_traces]
    args.question_ids = ",".join(wanted_ids)
    args.max_problems = 0

    llm = LLMAdapter(args)
    pipeline = DualLoopPipeline(args, llm=llm)
    benchmark = pipeline._load_benchmark()
    benchmark_by_qid = {problem.question_id: problem for problem in benchmark}

    rows: list[dict[str, Any]] = []
    detailed_records: list[dict[str, Any]] = []

    for trace in filtered_traces:
        question_id = str(trace.get("question_id", ""))
        if question_id not in benchmark_by_qid:
            raise KeyError(f"Question id '{question_id}' from traces was not found in the loaded benchmark.")
        problem = benchmark_by_qid[question_id]
        initial_result = _run_spec_variant(
            pipeline,
            problem,
            spec_payload=trace["spec_initial"],
            variant="initial_spec",
            repeats=args.repeats,
        )
        final_result = _run_spec_variant(
            pipeline,
            problem,
            spec_payload=trace["spec_final"],
            variant="final_spec",
            repeats=args.repeats,
        )
        row = _problem_row(trace, initial_result, final_result)
        rows.append(row)
        detailed_records.append(
            {
                "question_id": question_id,
                "question_title": trace.get("question_title", ""),
                "initial_spec_score": trace.get("initial_spec_score", {}),
                "final_spec_score": trace.get("final_spec_score", {}),
                "spec_initial": trace.get("spec_initial", {}),
                "spec_final": trace.get("spec_final", {}),
                "failure_attribution": trace.get("failure_attribution", ""),
                "effectiveness": trace.get("effectiveness", {}),
                "initial_counterfactual": initial_result,
                "final_counterfactual": final_result,
            }
        )

    slices = {
        "evaluated_subset": rows,
        "spec_changed": [row for row in rows if row["spec_changed"]],
        "low_initial_sas": [row for row in rows if float(row["initial_sas"]) < args.low_initial_sas_threshold],
        "changed_and_low_initial_sas": [
            row
            for row in rows
            if row["spec_changed"] and float(row["initial_sas"]) < args.low_initial_sas_threshold
        ],
        "sal_attempted": [row for row in rows if row["sal_attempted"]],
        "initial_codegen_failed": [row for row in rows if row["source_initial_codegen_failed"]],
        "changed_and_initial_codegen_failed": [
            row for row in rows if row["spec_changed"] and row["source_initial_codegen_failed"]
        ],
        "changed_low_sas_and_initial_codegen_failed": [
            row
            for row in rows
            if (
                row["spec_changed"]
                and float(row["initial_sas"]) < args.low_initial_sas_threshold
                and row["source_initial_codegen_failed"]
            )
        ],
        "source_spec_induced": [
            row for row in rows if row["source_failure_attribution"] == "spec_induced"
        ],
        "resolved_by_loop_b": [row for row in rows if row["source_resolved_by_loop_b"]],
        "resolved_by_loop_b_and_initial_codegen_failed": [
            row
            for row in rows
            if row["source_resolved_by_loop_b"] and row["source_initial_codegen_failed"]
        ],
    }
    slice_summaries = {name: _aggregate_rows(slice_rows) for name, slice_rows in slices.items()}

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_json(output_dir / "detailed_records.json", detailed_records)

    summary_payload = {
        "source_suite_dir": str(suite_dir),
        "source_run_name": args.run_name,
        "source_pipeline_mode": run_entry.get("pipeline_mode", ""),
        "source_summary_path": run_entry.get("summary_path", ""),
        "source_traces_path": run_entry.get("traces_path", ""),
        "release_version": args.release_version,
        "model": args.model,
        "model_repr": args.model_repr,
        "local_model_path": args.local_model_path,
        "subset": args.subset,
        "low_initial_sas_threshold": args.low_initial_sas_threshold,
        "repeats": args.repeats,
        "codegen_num_candidates": args.codegen_num_candidates,
        "codegen_temperature": args.codegen_temperature,
        "num_selected_problems": len(rows),
        "output_dir": str(output_dir),
        "results_csv": str(csv_path),
        "slice_summaries": slice_summaries,
    }
    _write_json(output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
