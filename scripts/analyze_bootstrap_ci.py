import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_COMPARISONS = (
    ("semantic_contract_irl", "raw_nl_irl"),
    ("semantic_contract_irl", "plan_irl"),
    ("semantic_contract_irl", "pseudocode_irl"),
    ("semantic_contract_irl", "contract_text_irl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap confidence intervals over pass/fail traces."
    )
    parser.add_argument(
        "--suite_dir",
        type=str,
        default=None,
        help="Directory containing method traces such as semantic_contract_irl_traces.json.",
    )
    parser.add_argument(
        "--trace_file",
        action="append",
        default=[],
        help="Explicit method=path trace file. Can be repeated.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="Explicit comparison as lhs-rhs. Defaults to semantic_contract_irl against all baselines.",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to suite_dir, or current directory when using only --trace_file.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _method_from_trace_filename(path: Path) -> str:
    name = path.name
    suffix = "_traces.json"
    if not name.endswith(suffix):
        raise ValueError(f"Cannot infer method from trace filename: {path}")
    return name[: -len(suffix)]


def _load_trace_map(args: argparse.Namespace) -> dict[str, dict[str, bool]]:
    method_to_path: dict[str, Path] = {}
    if args.suite_dir:
        suite_dir = Path(args.suite_dir)
        for path in suite_dir.glob("*_traces.json"):
            method_to_path[_method_from_trace_filename(path)] = path

    for item in args.trace_file:
        if "=" not in item:
            raise ValueError("--trace_file must use method=path")
        method, raw_path = item.split("=", 1)
        method_to_path[method.strip()] = Path(raw_path.strip())

    if not method_to_path:
        raise ValueError("No trace files found. Provide --suite_dir or --trace_file.")

    traces_by_method: dict[str, dict[str, bool]] = {}
    for method, path in sorted(method_to_path.items()):
        traces = _load_json(path)
        if not isinstance(traces, list):
            raise ValueError(f"Trace file is not a list: {path}")
        traces_by_method[method] = {
            str(trace["question_id"]): bool(trace.get("passed", False))
            for trace in traces
            if "question_id" in trace
        }
    return traces_by_method


def _parse_comparisons(raw_comparisons: list[str]) -> list[tuple[str, str]]:
    if not raw_comparisons:
        return list(DEFAULT_COMPARISONS)
    comparisons: list[tuple[str, str]] = []
    for item in raw_comparisons:
        if "-" not in item:
            raise ValueError("--comparison must use lhs-rhs")
        lhs, rhs = item.split("-", 1)
        comparisons.append((lhs.strip(), rhs.strip()))
    return comparisons


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = percentile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_bootstrap(
    *,
    lhs: dict[str, bool],
    rhs: dict[str, bool],
    resamples: int,
    rng: random.Random,
) -> dict[str, Any]:
    question_ids = sorted(set(lhs) & set(rhs))
    if not question_ids:
        raise ValueError("No overlapping question IDs for paired bootstrap comparison.")

    observed_deltas = [
        float(lhs[question_id]) - float(rhs[question_id]) for question_id in question_ids
    ]
    observed_delta = mean(observed_deltas)

    bootstrap_deltas: list[float] = []
    n = len(question_ids)
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            question_id = question_ids[rng.randrange(n)]
            total += float(lhs[question_id]) - float(rhs[question_id])
        bootstrap_deltas.append(total / n)

    bootstrap_deltas.sort()
    positive_fraction = sum(1 for value in bootstrap_deltas if value > 0.0) / max(
        1, len(bootstrap_deltas)
    )
    return {
        "num_paired_problems": n,
        "lhs_pass_at_1": round(mean(float(lhs[q]) for q in question_ids), 6),
        "rhs_pass_at_1": round(mean(float(rhs[q]) for q in question_ids), 6),
        "delta_pass_at_1": round(observed_delta, 6),
        "ci95_low": round(_percentile(bootstrap_deltas, 0.025), 6),
        "ci95_high": round(_percentile(bootstrap_deltas, 0.975), 6),
        "positive_fraction": round(positive_fraction, 6),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "comparison",
        "lhs",
        "rhs",
        "num_paired_problems",
        "lhs_pass_at_1",
        "rhs_pass_at_1",
        "delta_pass_at_1",
        "ci95_low",
        "ci95_high",
        "positive_fraction",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    traces_by_method = _load_trace_map(args)
    comparisons = _parse_comparisons(args.comparison)
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []
    for lhs_method, rhs_method in comparisons:
        if lhs_method not in traces_by_method or rhs_method not in traces_by_method:
            available = ", ".join(sorted(traces_by_method))
            print(
                f"Skipping {lhs_method}-{rhs_method}; available methods: {available}",
                file=sys.stderr,
            )
            continue
        result = _paired_bootstrap(
            lhs=traces_by_method[lhs_method],
            rhs=traces_by_method[rhs_method],
            resamples=args.resamples,
            rng=rng,
        )
        rows.append(
            {
                "comparison": f"{lhs_method} - {rhs_method}",
                "lhs": lhs_method,
                "rhs": rhs_method,
                **result,
            }
        )

    if not rows:
        raise ValueError("No bootstrap comparisons were computed.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.suite_dir:
        output_dir = Path(args.suite_dir)
    else:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "bootstrap_ci.csv"
    json_path = output_dir / "bootstrap_ci.json"
    _write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "resamples": args.resamples,
                "seed": args.seed,
                "rows": rows,
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    print(json.dumps({"csv_path": str(csv_path), "json_path": str(json_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
