#!/usr/bin/env python3
"""Re-evaluate saved final programs on held-out private tests only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESIDUAL_ATTRIBUTIONS = {"spec_induced", "implementation_induced", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run saved final_code values on private LiveCodeBench tests without "
            "loading an LLM or regenerating any program."
        )
    )
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--release_version", type=str, default="release_v6")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Derived-output directory. The source traces are never modified.",
    )
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num_process_evaluate", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--max_problems",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 evaluates every saved trace.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from evaluation_checkpoint.json in --output_dir.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_dict(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"raw_metadata": raw_metadata}
        return parsed if isinstance(parsed, dict) else {"raw_metadata": parsed}
    return {}


def _error_type(passed: bool, metadata: dict[str, Any]) -> str:
    if passed:
        return "accepted"
    return {
        -2: "wrong_answer",
        -3: "time_limit_exceeded",
        -4: "runtime_error",
        -5: "verifier_error",
    }.get(metadata.get("error_code"), "unknown_failure")


def align_attribution(
    *, feedback_passed: bool, final_passed: bool, original_attribution: str
) -> str:
    """Conservatively align trace categories to the held-out final verdict."""
    if final_passed:
        return "solved"
    if feedback_passed:
        # A feedback-stage pass contains no residual root-cause evidence.
        return "unknown"
    if original_attribution in RESIDUAL_ATTRIBUTIONS:
        return original_attribution
    return "unknown"


def _transition(feedback_passed: bool, final_passed: bool) -> str:
    return (
        f"feedback_{'pass' if feedback_passed else 'fail'}_"
        f"final_{'pass' if final_passed else 'fail'}"
    )


def _prepare_manifest(args: argparse.Namespace, trace_count: int) -> dict[str, Any]:
    return {
        "source_traces": str(args.traces.resolve()),
        "source_traces_sha256": _sha256(args.traces),
        "source_summary": str(args.summary.resolve()) if args.summary else None,
        "release_version": args.release_version,
        "dataset_path": str(Path(args.dataset_path).resolve()) if args.dataset_path else None,
        "final_test_scope": "private",
        "trace_count": trace_count,
        "max_problems": args.max_problems,
        "timeout": args.timeout,
    }


def _validate_resume_manifest(existing: dict[str, Any], current: dict[str, Any]) -> None:
    stable_keys = (
        "source_traces_sha256",
        "release_version",
        "final_test_scope",
        "trace_count",
        "max_problems",
        "timeout",
    )
    mismatches = [
        key for key in stable_keys if existing.get(key) != current.get(key)
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(f"Cannot resume: run manifest differs for {names}.")


def _evaluate_batches(
    *,
    args: argparse.Namespace,
    selected: list[tuple[dict[str, Any], Any]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    from lcb_runner.evaluation import codegen_metrics

    checkpoint_path = output_dir / "evaluation_checkpoint.json"
    checkpoint: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_path.exists():
        loaded = _read_json(checkpoint_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
        checkpoint = loaded
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"Checkpoint already exists at {checkpoint_path}; pass --resume or choose "
            "a different --output_dir."
        )

    pending = [
        pair for pair in selected if str(pair[0]["question_id"]) not in checkpoint
    ]
    print(
        f"Held-out evaluation: total={len(selected)}, completed={len(selected) - len(pending)}, "
        f"pending={len(pending)}"
    )

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        samples = [problem.get_final_evaluation_sample() for _, problem in batch]
        empty = [
            str(trace["question_id"])
            for (trace, _), sample in zip(batch, samples)
            if not json.loads(sample["input_output"]).get("inputs", [])
        ]
        if empty:
            raise ValueError(
                "Private tests are missing for: " + ", ".join(empty[:10])
            )

        generations = [[str(trace.get("final_code", ""))] for trace, _ in batch]
        metrics = codegen_metrics(
            samples,
            generations,
            k_list=[1],
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
            debug=False,
        )
        results = metrics[1]
        metadatas = metrics[2]

        for index, (trace, _) in enumerate(batch):
            question_id = str(trace["question_id"])
            per_generation = results.get(index, [])
            test_results = per_generation[0] if per_generation else []
            if not isinstance(test_results, list):
                test_results = [test_results]
            passed = bool(test_results) and all(result is True for result in test_results)
            raw_metadata = metadatas[index][0] if index < len(metadatas) else {}
            metadata = _metadata_dict(raw_metadata)
            checkpoint[question_id] = {
                "passed": passed,
                "num_tests": len(test_results),
                "first_failing_result": next(
                    (result for result in test_results if result is not True), None
                ),
                "error_code": metadata.get("error_code"),
                "error_message": metadata.get("error_message", ""),
                "error_type": _error_type(passed, metadata),
                "metadata": metadata,
            }

        _write_json_atomic(checkpoint_path, checkpoint)
        print(
            f"Checkpointed {min(start + len(batch), len(pending))}/{len(pending)} "
            "pending problems."
        )

    return checkpoint


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "question_title",
        "feedback_passed",
        "final_passed",
        "transition",
        "original_attribution",
        "final_aligned_attribution",
        "error_type",
        "error_code",
        "error_message",
        "num_private_tests",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.traces = args.traces.resolve()
    if args.summary is None:
        candidate = args.traces.with_name("summary.json")
        args.summary = candidate if candidate.exists() else None
    elif not args.summary.exists():
        raise FileNotFoundError(args.summary)
    if not args.traces.exists():
        raise FileNotFoundError(args.traces)
    if args.batch_size < 1 or args.num_process_evaluate < 1:
        raise ValueError("batch_size and num_process_evaluate must be positive")

    traces = _read_json(args.traces)
    if not isinstance(traces, list) or not traces:
        raise ValueError("--traces must contain a non-empty JSON list")
    if args.max_problems > 0:
        traces = traces[: args.max_problems]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("output/heldout_reevaluation") / (
        f"{args.traces.parent.name}_private_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _prepare_manifest(args, len(traces))
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Run manifest already exists at {manifest_path}; pass --resume or "
                "choose a different --output_dir."
            )
        _validate_resume_manifest(_read_json(manifest_path), manifest)
    else:
        _write_json_atomic(manifest_path, manifest)

    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    benchmark = load_code_generation_dataset(
        release_version=args.release_version,
        dataset_path=args.dataset_path,
    )
    problems_by_id = {str(problem.question_id): problem for problem in benchmark}
    trace_ids = [str(trace.get("question_id", "")) for trace in traces]
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("Saved traces contain duplicate question_id values")
    missing = [question_id for question_id in trace_ids if question_id not in problems_by_id]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} trace problem(s): "
            + ", ".join(missing[:10])
        )
    selected = [(trace, problems_by_id[str(trace["question_id"])]) for trace in traces]

    checkpoint = _evaluate_batches(
        args=args,
        selected=selected,
        output_dir=output_dir,
    )

    derived_traces: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    attribution_counts: Counter[str] = Counter()

    for trace in traces:
        question_id = str(trace["question_id"])
        evaluation = checkpoint[question_id]
        feedback_passed = bool(trace.get("passed", False))
        final_passed = bool(evaluation["passed"])
        original_attribution = str(trace.get("failure_attribution", "unknown"))
        final_attribution = align_attribution(
            feedback_passed=feedback_passed,
            final_passed=final_passed,
            original_attribution=original_attribution,
        )
        transition = _transition(feedback_passed, final_passed)
        transition_counts[transition] += 1
        attribution_counts[final_attribution] += 1

        derived = deepcopy(trace)
        derived["original_feedback_passed"] = feedback_passed
        derived["original_failure_attribution"] = original_attribution
        derived["passed"] = final_passed
        derived["failure_attribution"] = final_attribution
        effectiveness = dict(derived.get("effectiveness") or {})
        effectiveness["final_evaluation"] = {
            "scope": "private",
            "feedback_passed": feedback_passed,
            "final_passed": final_passed,
            "transition": transition,
            "error_type": evaluation["error_type"],
            "error_code": evaluation["error_code"],
            "error_message": evaluation["error_message"],
            "num_tests": evaluation["num_tests"],
        }
        derived["effectiveness"] = effectiveness
        derived_traces.append(derived)

        rows.append(
            {
                "question_id": question_id,
                "question_title": trace.get("question_title", ""),
                "feedback_passed": feedback_passed,
                "final_passed": final_passed,
                "transition": transition,
                "original_attribution": original_attribution,
                "final_aligned_attribution": final_attribution,
                "error_type": evaluation["error_type"],
                "error_code": evaluation["error_code"],
                "error_message": evaluation["error_message"],
                "num_private_tests": evaluation["num_tests"],
            }
        )

    final_pass_count = sum(bool(trace["passed"]) for trace in derived_traces)
    feedback_pass_count = sum(
        bool(trace["original_feedback_passed"]) for trace in derived_traces
    )
    summary = {
        "protocol": "saved_final_code_private_heldout_reevaluation",
        "source_traces": str(args.traces),
        "source_summary": str(args.summary.resolve()) if args.summary else None,
        "release_version": args.release_version,
        "num_problems": len(derived_traces),
        "feedback_pass_count": feedback_pass_count,
        "feedback_pass_rate": feedback_pass_count / len(derived_traces),
        "final_pass_count": final_pass_count,
        "final_pass_rate": final_pass_count / len(derived_traces),
        "transition_counts": dict(sorted(transition_counts.items())),
        "final_aligned_attribution_counts": {
            key: attribution_counts.get(key, 0)
            for key in ("solved", "spec_induced", "implementation_induced", "unknown")
        },
        "attribution_alignment_policy": (
            "Held-out passes are solved. A feedback pass followed by a held-out failure "
            "is unknown because the original trace contains no residual attribution. "
            "For programs failing both stages, the original residual trace label is retained."
        ),
        "model_generation_performed": False,
    }

    _write_json_atomic(output_dir / "summary.json", summary)
    _write_json_atomic(output_dir / "traces_heldout.json", derived_traces)
    _write_csv(output_dir / "per_problem.csv", rows)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Held-out outputs: {output_dir}")


if __name__ == "__main__":
    main()
