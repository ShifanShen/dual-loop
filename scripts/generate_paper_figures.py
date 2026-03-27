from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_PIPELINE_CSV = (
    "output/dual_loop_rq_suite/"
    "rq_suite_Qwen2.5-Coder-7B-Instruct_20260324_154821/rq_results.csv"
)
DEFAULT_REPAIR_CSV = (
    "output/dual_loop_rq_suite/"
    "rq_suite_Qwen2.5-Coder-7B-Instruct_20260325_154200/rq_results.csv"
)
DEFAULT_FULL_SUMMARY = (
    "output/dual_loop/full_Qwen2.5-Coder-7B-Instruct_full_dual_loop_"
    "20260325_202522_541420/summary.json"
)
DEFAULT_OUTPUT_DIR = "paper/figures/generated"


PIPELINE_ORDER = [
    "Direct NL->Code",
    "Decomposition Only",
    "Self-Refine-style",
    "Reflexion-style",
    "Loop B Only",
    "Loop A Only",
    "Full Dual-Loop",
]

REPAIR_ORDER = [
    "Full Plain Repair",
    "Full w/o Rewrite Repair",
    "Full w/o Counterexample Repair",
    "Full Dual-Loop",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _select_rows(rows: list[dict[str, str]], order: list[str]) -> list[dict[str, str]]:
    by_name = {row["method_label"]: row for row in rows}
    return [by_name[name] for name in order if name in by_name]


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_ablation(rows: list[dict[str, str]], output_dir: Path) -> None:
    selected = _select_rows(rows, PIPELINE_ORDER)
    labels = [row["method_label"] for row in selected]
    scores = [_to_float(row, "pass_at_1") for row in selected]

    colors = [
        "#8da0cb" if label not in ("Loop B Only", "Loop A Only", "Full Dual-Loop") else
        "#66c2a5" if label == "Loop B Only" else
        "#fc8d62" if label == "Loop A Only" else
        "#1b9e77"
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, scores, color=colors, edgecolor="#444444")
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, max(scores) + 0.12)
    ax.set_title("Pipeline Ablations on 50 LiveCodeBench Problems")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=22)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, output_dir, "pipeline_ablation_pass1")


def plot_repair_ablation(rows: list[dict[str, str]], output_dir: Path) -> None:
    selected = _select_rows(rows, REPAIR_ORDER)
    labels = [row["method_label"] for row in selected]
    scores = [_to_float(row, "pass_at_1") for row in selected]
    solved = [_to_float(row, "repair_solved_count") for row in selected]
    improved = [_to_float(row, "repair_improved_count") for row in selected]

    fig, ax1 = plt.subplots(figsize=(9.4, 4.8))
    x = list(range(len(labels)))
    width = 0.58
    bars = ax1.bar(x, scores, width=width, color="#fc8d62", edgecolor="#444444")
    ax1.set_ylabel("pass@1")
    ax1.set_ylim(0, max(scores) + 0.14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18)
    ax1.set_title("Repair Ablations on 50 LiveCodeBench Problems")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    for bar, score in zip(bars, scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = ax1.twinx()
    ax2.plot(x, solved, color="#1b9e77", marker="o", linewidth=2, label="repair solved")
    ax2.plot(x, improved, color="#7570b3", marker="s", linewidth=2, label="repair improved")
    ax2.set_ylabel("count")
    ax2.set_ylim(0, max(max(solved), max(improved)) + 2)
    ax2.legend(loc="upper left", frameon=False)

    _save(fig, output_dir, "repair_ablation_pass1")


def plot_accuracy_cost(rows: list[dict[str, str]], output_dir: Path) -> None:
    selected = _select_rows(rows, PIPELINE_ORDER)
    labels = [row["method_label"] for row in selected]
    scores = [_to_float(row, "pass_at_1") for row in selected]
    calls = [_to_float(row, "average_llm_calls") for row in selected]

    color_map = {
        "Full Dual-Loop": "#1b9e77",
        "Loop A Only": "#fc8d62",
        "Loop B Only": "#66c2a5",
    }
    colors = [color_map.get(label, "#8da0cb") for label in labels]

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    ax.scatter(calls, scores, s=120, c=colors, edgecolors="#333333")
    ax.set_xlabel("Average LLM calls")
    ax.set_ylabel("pass@1")
    ax.set_title("Accuracy-Cost Tradeoff on Pipeline Ablations")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for x, y, label in zip(calls, scores, labels):
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    _save(fig, output_dir, "accuracy_cost_tradeoff")


def plot_mechanism(summary: dict, loop_b_summary: dict, loop_a_summary: dict, output_dir: Path) -> None:
    labels = ["SAL-only", "IRL-only", "Full"]
    pass_scores = [
        loop_b_summary["pass_at_1"],
        loop_a_summary["pass_at_1"],
        summary["pass_at_1"],
    ]
    sas_deltas = [
        loop_b_summary["average_delta_sas"],
        loop_a_summary["average_delta_sas"],
        summary["average_delta_sas"],
    ]
    spec_rates = [
        loop_b_summary["failure_attribution_counts"].get("spec_induced", 0) / loop_b_summary["num_problems"],
        loop_a_summary["failure_attribution_counts"].get("spec_induced", 0) / loop_a_summary["num_problems"],
        summary["failure_attribution_counts"].get("spec_induced", 0) / summary["num_problems"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))

    x = range(len(labels))
    axes[0].bar(x, pass_scores, color=["#66c2a5", "#fc8d62", "#1b9e77"], edgecolor="#444444")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, max(pass_scores) + 0.15)
    axes[0].set_ylabel("pass@1")
    axes[0].set_title("Performance by Core Module")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_axisbelow(True)

    for idx, value in enumerate(pass_scores):
        axes[0].text(idx, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    width = 0.35
    x2 = list(range(len(labels)))
    axes[1].bar([i - width / 2 for i in x2], sas_deltas, width=width, color="#7570b3", label="Avg. SAS delta")
    axes[1].bar([i + width / 2 for i in x2], spec_rates, width=width, color="#e7298a", label="Spec-induced rate")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Semantic-Side Signals")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False)

    _save(fig, output_dir, "mechanism_summary")


def plot_property_oracle(summary: dict, output_dir: Path) -> None:
    clause_counts = summary.get("property_clause_type_counts", {})
    violation_counts = summary.get("property_violation_counts", {})
    clause_labels = sorted(clause_counts.keys())
    if not clause_labels:
        return

    clauses = [clause_counts[label] for label in clause_labels]
    violations = [violation_counts.get(label, 0) for label in clause_labels]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = list(range(len(clause_labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], clauses, width=width, color="#80b1d3", label="compiled clauses")
    ax.bar([i + width / 2 for i in x], violations, width=width, color="#fb8072", label="observed violations")
    ax.set_xticks(x)
    ax.set_xticklabels(clause_labels, rotation=15)
    ax.set_ylabel("count")
    ax.set_title("Property Oracle V1 Coverage in Full Checkpoint (200 problems)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    _save(fig, output_dir, "property_oracle_coverage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from experiment outputs.")
    parser.add_argument("--pipeline-csv", default=DEFAULT_PIPELINE_CSV)
    parser.add_argument("--repair-csv", default=DEFAULT_REPAIR_CSV)
    parser.add_argument("--full-summary", default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--loop-b-summary", default="output/dual_loop/loop_b_Qwen2.5-Coder-7B-Instruct_loop_b_only_20260324_164845_166159/summary.json")
    parser.add_argument("--loop-a-summary", default="output/dual_loop/loop_a_Qwen2.5-Coder-7B-Instruct_loop_a_only_20260324_171131_931367/summary.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_rows = _read_csv(Path(args.pipeline_csv))
    repair_rows = _read_csv(Path(args.repair_csv))
    full_summary = _read_json(Path(args.full_summary))
    loop_b_summary = _read_json(Path(args.loop_b_summary))
    loop_a_summary = _read_json(Path(args.loop_a_summary))

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_pipeline_ablation(pipeline_rows, output_dir)
    plot_repair_ablation(repair_rows, output_dir)
    plot_accuracy_cost(pipeline_rows, output_dir)
    plot_mechanism(full_summary, loop_b_summary, loop_a_summary, output_dir)
    plot_property_oracle(full_summary, output_dir)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
