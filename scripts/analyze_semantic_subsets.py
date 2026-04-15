import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.diagnostics import load_run_artifacts


DEFAULT_RUNS = [
    "baseline_direct",
    "decomposition_only",
    "reflexion_style",
    "full_dual_loop",
    "full_adaptive_sal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze performance on semantic-heavy problem subsets using trace text heuristics."
    )
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--run_names",
        type=str,
        default=",".join(DEFAULT_RUNS),
        help="Comma-separated run names to analyze.",
    )
    return parser.parse_args()


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


def _classify_problem(text: str) -> dict[str, bool]:
    lowered = (text or "").lower()

    constraint_hits = sum(
        1
        for pattern in [
            r"\bat most\b",
            r"\bat least\b",
            r"\bexactly\b",
            r"\bmust\b",
            r"\bcannot\b",
            r"\bno more than\b",
            r"\bno less than\b",
            r"\bdistinct\b",
            r"\bmodulo\b",
            r"\bminimum\b",
            r"\bmaximum\b",
            r"\bconstraint\b",
        ]
        if re.search(pattern, lowered)
    )
    format_hits = sum(
        1
        for pattern in [
            r"\boutput\b",
            r"\bprint\b",
            r"\byes\b",
            r"\bno\b",
            r"\bpossible\b",
            r"\bimpossible\b",
            r"\bformat\b",
            r"\bline\b",
            r"\blines\b",
            r"\bspace-separated\b",
        ]
        if re.search(pattern, lowered)
    )
    edge_hits = sum(
        1
        for pattern in [
            r"\botherwise\b",
            r"\bif not\b",
            r"\bif there is no\b",
            r"\bempty\b",
            r"\bsingle\b",
            r"\bcorner case\b",
            r"\bedge case\b",
            r"\bno solution\b",
            r"\breturn -1\b",
            r"\breturn 0\b",
        ]
        if re.search(pattern, lowered)
    )

    flags = {
        "constraint_heavy": constraint_hits >= 2,
        "format_sensitive": format_hits >= 2,
        "edge_case_heavy": edge_hits >= 1,
    }
    flags["semantic_heavy"] = (
        sum(int(value) for value in flags.values()) >= 2 or constraint_hits >= 3
    )
    return flags


def _aggregate_subset(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    subset_name: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "run_name": run_name,
            "subset_name": subset_name,
            "num_problems": 0,
            "pass_at_1": 0.0,
            "average_final_sas": 0.0,
            "spec_induced_rate": 0.0,
            "unknown_rate": 0.0,
        }

    return {
        "run_name": run_name,
        "subset_name": subset_name,
        "num_problems": len(rows),
        "pass_at_1": round(
            sum(1.0 if row["passed"] else 0.0 for row in rows) / len(rows), 4
        ),
        "average_final_sas": round(
            sum(float(row["final_sas"]) for row in rows) / len(rows),
            4,
        ),
        "spec_induced_rate": round(
            sum(1.0 if row["failure_attribution"] == "spec_induced" else 0.0 for row in rows)
            / len(rows),
            4,
        ),
        "unknown_rate": round(
            sum(1.0 if row["failure_attribution"] == "unknown" else 0.0 for row in rows)
            / len(rows),
            4,
        ),
    }


def main() -> None:
    args = parse_args()
    suite_dir = _resolve_suite_dir(args.suite_dir)
    output_dir = Path(args.output_dir) if args.output_dir else suite_dir / "semantic_subset_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted_runs = {item.strip() for item in args.run_names.split(",") if item.strip()}
    manifest = _load_manifest(suite_dir)
    selected_runs = [entry for entry in manifest if entry["run_name"] in wanted_runs]
    if not selected_runs:
        raise ValueError("No requested run_names were found in the suite manifest.")

    subset_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for entry in selected_runs:
        _, traces = load_run_artifacts(entry["summary_path"], entry["traces_path"])
        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for trace in traces:
            problem_text = str(trace.get("raw_problem", "") or "")
            final_score = trace.get("final_spec_score") or trace.get("initial_spec_score") or {}
            labels = _classify_problem(problem_text)
            row = {
                "run_name": entry["run_name"],
                "question_id": trace.get("question_id", ""),
                "question_title": trace.get("question_title", ""),
                "passed": bool(trace.get("passed", False)),
                "final_sas": float(final_score.get("overall", 0.0) or 0.0),
                "failure_attribution": trace.get("failure_attribution", "unknown"),
                **labels,
            }
            trace_rows.append(row)
            grouped_rows["all"].append(row)
            for subset_name, enabled in labels.items():
                if enabled:
                    grouped_rows[subset_name].append(row)

        for subset_name in ["all", "constraint_heavy", "format_sensitive", "edge_case_heavy", "semantic_heavy"]:
            subset_rows.append(
                _aggregate_subset(
                    grouped_rows.get(subset_name, []),
                    run_name=entry["run_name"],
                    subset_name=subset_name,
                )
            )

    baseline_by_subset = {
        row["subset_name"]: row["pass_at_1"]
        for row in subset_rows
        if row["run_name"] == "baseline_direct"
    }
    for row in subset_rows:
        row["delta_pass_at_1_vs_direct"] = round(
            float(row["pass_at_1"]) - float(baseline_by_subset.get(row["subset_name"], 0.0)),
            4,
        )

    with open(output_dir / "semantic_subset_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "subset_name",
                "num_problems",
                "pass_at_1",
                "delta_pass_at_1_vs_direct",
                "average_final_sas",
                "spec_induced_rate",
                "unknown_rate",
            ],
        )
        writer.writeheader()
        for row in subset_rows:
            writer.writerow(row)

    with open(output_dir / "semantic_subset_traces.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "question_id",
                "question_title",
                "passed",
                "final_sas",
                "failure_attribution",
                "constraint_heavy",
                "format_sensitive",
                "edge_case_heavy",
                "semantic_heavy",
            ],
        )
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)

    summary = {
        "suite_dir": str(suite_dir),
        "selected_runs": [entry["run_name"] for entry in selected_runs],
        "output_dir": str(output_dir),
        "summary_rows": subset_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
