import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.diagnostics import load_run_artifacts


SPEC_RUNS = ("decomposition_only", "loop_b_only", "loop_a_only", "full_dual_loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how SAS relates to correctness and semantic failure labels."
    )
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _pearson(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _bucket(score: float) -> str:
    if score < 80:
        return "<80"
    if score < 85:
        return "80-84"
    if score < 90:
        return "85-89"
    return ">=90"


def _load_suite_manifest(suite_dir: Path) -> list[dict]:
    with open(suite_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _get_run_artifacts(run_entry: dict) -> tuple[dict, list[dict]]:
    return load_run_artifacts(run_entry["summary_path"], run_entry["traces_path"])


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir) if args.output_dir else suite_dir / "sas_failure_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_suite_manifest(suite_dir)
    selected_entries = [entry for entry in manifest if entry["run_name"] in SPEC_RUNS]

    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for entry in selected_entries:
        run_name = entry["run_name"]
        summary, traces = _get_run_artifacts(entry)
        scores: list[float] = []
        passed_flags: list[float] = []
        spec_induced_flags: list[float] = []
        by_bucket: dict[str, list[dict]] = defaultdict(list)

        for trace in traces:
            score_payload = trace.get("final_spec_score") or trace.get("initial_spec_score") or {}
            score = float(score_payload.get("overall", 0.0))
            passed = 1.0 if trace.get("passed") else 0.0
            spec_induced = 1.0 if trace.get("failure_attribution") == "spec_induced" else 0.0

            scores.append(score)
            passed_flags.append(passed)
            spec_induced_flags.append(spec_induced)
            by_bucket[_bucket(score)].append(trace)
            all_rows.append(
                {
                    "run_name": run_name,
                    "question_id": trace.get("question_id"),
                    "question_title": trace.get("question_title"),
                    "final_sas": score,
                    "passed": int(passed),
                    "failure_attribution": trace.get("failure_attribution", "unknown"),
                }
            )

        summary_rows.append(
            {
                "run_name": run_name,
                "num_problems": len(traces),
                "pearson_sas_vs_passed": round(_pearson(scores, passed_flags), 4),
                "pearson_sas_vs_spec_induced": round(_pearson(scores, spec_induced_flags), 4),
                "average_sas": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "pass_at_1": summary.get("pass_at_1", 0.0),
                "spec_induced_rate": round(
                    float(summary.get("failure_attribution_counts", {}).get("spec_induced", 0)) / max(len(traces), 1),
                    4,
                ),
            }
        )

        for bucket_name, bucket_traces in by_bucket.items():
            summary_rows.append(
                {
                    "run_name": run_name,
                    "bucket": bucket_name,
                    "num_problems": len(bucket_traces),
                    "pearson_sas_vs_passed": "",
                    "pearson_sas_vs_spec_induced": "",
                    "average_sas": round(
                        sum(float((trace.get("final_spec_score") or trace.get("initial_spec_score") or {}).get("overall", 0.0)) for trace in bucket_traces)
                        / len(bucket_traces),
                        2,
                    ),
                    "pass_at_1": round(sum(1.0 if trace.get("passed") else 0.0 for trace in bucket_traces) / len(bucket_traces), 4),
                    "spec_induced_rate": round(
                        sum(1.0 if trace.get("failure_attribution") == "spec_induced" else 0.0 for trace in bucket_traces) / len(bucket_traces),
                        4,
                    ),
                }
            )

    with open(output_dir / "sas_failure_correlation.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "bucket",
                "num_problems",
                "pearson_sas_vs_passed",
                "pearson_sas_vs_spec_induced",
                "average_sas",
                "pass_at_1",
                "spec_induced_rate",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(output_dir / "sas_failure_trace_table.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "question_id",
                "question_title",
                "final_sas",
                "passed",
                "failure_attribution",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    report = {
        "suite_dir": str(suite_dir),
        "analyzed_runs": [entry["run_name"] for entry in selected_entries],
        "summary_rows": summary_rows,
        "failure_counts": {
            entry["run_name"]: dict(
                Counter(
                    row["failure_attribution"]
                    for row in all_rows
                    if row["run_name"] == entry["run_name"]
                )
            )
            for entry in selected_entries
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
