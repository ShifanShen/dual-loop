import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize attribution-guided SAL re-entry behavior from dual-loop traces."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Directory containing summary.json and traces.json.",
    )
    parser.add_argument("--summary", type=str, default="", help="Path to summary.json.")
    parser.add_argument("--traces", type=str, default="", help="Path to traces.json.")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path | None, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        summary_path = run_dir / "summary.json"
        traces_path = run_dir / "traces.json"
    else:
        summary_path = Path(args.summary) if args.summary else None
        traces_path = Path(args.traces) if args.traces else Path()

    if not traces_path.exists():
        raise FileNotFoundError(
            "Could not find traces.json. Provide --run_dir or --traces."
        )
    if summary_path is not None and not summary_path.exists():
        summary_path = None
    return summary_path, traces_path


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _post_failure_sal(trace: dict[str, Any]) -> dict[str, Any]:
    return dict(((trace.get("effectiveness") or {}).get("post_failure_sal") or {}))


def _last_attempt(post: dict[str, Any]) -> dict[str, Any]:
    attempts = list(post.get("attempts") or [])
    return dict(attempts[-1]) if attempts else {}


def summarize(summary: dict[str, Any] | None, traces: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = []
    accepted = []
    accepted_solved = []
    accepted_unsolved = []
    skipped = []
    pre_attr_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    final_reason_counts: Counter[str] = Counter()
    gap_source_counts: Counter[str] = Counter()
    gap_patch_field_counts: Counter[str] = Counter()
    gap_confidences: list[int] = []

    for trace in traces:
        post = _post_failure_sal(trace)
        if not post:
            continue
        reason_counts[str(post.get("reason", "unknown"))] += 1
        if post.get("pre_reentry_attribution"):
            pre_attr_counts[str(post.get("pre_reentry_attribution"))] += 1
        gap = dict(post.get("failure_gap_judge") or {})
        if gap:
            gap_source_counts[str(gap.get("failure_source", "unknown"))] += 1
            try:
                gap_confidences.append(int(gap.get("confidence", 0) or 0))
            except (TypeError, ValueError):
                pass
            for field_name in gap.get("patch_fields", []) or []:
                gap_patch_field_counts[str(field_name)] += 1
        if post.get("attempts"):
            attempted.append(trace)
        else:
            skipped.append(trace)
        if post.get("accepted"):
            accepted.append(trace)
            if trace.get("passed"):
                accepted_solved.append(trace)
            else:
                accepted_unsolved.append(trace)
            final_reason_counts[str(_last_attempt(post).get("reason", post.get("reason", "unknown")))] += 1

    total = len(traces)
    payload = {
        "num_traces": total,
        "pass_at_1": summary.get("pass_at_1") if summary else None,
        "post_failure_sal_max_iters": summary.get("post_failure_sal_max_iters") if summary else None,
        "post_failure_sal_trigger": summary.get("post_failure_sal_trigger") if summary else None,
        "failure_gap_judge_enabled": summary.get("failure_gap_judge_enabled") if summary else None,
        "failure_gap_confidence_threshold": (
            summary.get("failure_gap_confidence_threshold") if summary else None
        ),
        "post_failure_records": len(attempted) + len(skipped),
        "reentry_attempted": len(attempted),
        "reentry_accepted": len(accepted),
        "reentry_accepted_solved": len(accepted_solved),
        "reentry_accepted_unsolved": len(accepted_unsolved),
        "reentry_attempt_rate": round(len(attempted) / total, 4) if total else 0.0,
        "reentry_accept_rate": round(len(accepted) / max(len(attempted), 1), 4),
        "accepted_solve_rate": round(len(accepted_solved) / max(len(accepted), 1), 4),
        "pre_reentry_attribution_counts": dict(pre_attr_counts),
        "failure_gap_source_counts": dict(gap_source_counts),
        "failure_gap_patch_field_counts": dict(gap_patch_field_counts),
        "average_failure_gap_confidence": (
            round(sum(gap_confidences) / len(gap_confidences), 2)
            if gap_confidences
            else 0.0
        ),
        "post_failure_reason_counts": dict(reason_counts),
        "accepted_final_reason_counts": dict(final_reason_counts),
        "accepted_question_ids": [trace.get("question_id", "") for trace in accepted],
        "accepted_solved_question_ids": [
            trace.get("question_id", "") for trace in accepted_solved
        ],
    }
    return payload


def main() -> None:
    args = parse_args()
    summary_path, traces_path = _resolve_paths(args)
    summary = _load_json(summary_path) if summary_path else None
    traces = _load_json(traces_path)
    print(json.dumps(summarize(summary, traces), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
