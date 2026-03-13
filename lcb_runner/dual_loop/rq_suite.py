import csv
import json
from argparse import Namespace
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lcb_runner.dual_loop.diagnostics import load_run_artifacts
from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter


@dataclass(frozen=True)
class SuiteRunConfig:
    suite_name: str
    run_name: str
    pipeline_mode: str
    disable_counterexample_repair: bool = False
    disable_rewrite_repair: bool = False
    spec_max_iters: int | None = None
    repair_max_iters: int | None = None
    spec_score_threshold: int | None = None


CORE_SUITE_NAME = "main_comparison"


def build_rq_suite_plan(
    *,
    include_repair_ablations: bool = False,
    include_budget_ablations: bool = False,
) -> list[SuiteRunConfig]:
    plan = [
        SuiteRunConfig(CORE_SUITE_NAME, "baseline_direct", "baseline"),
        SuiteRunConfig(CORE_SUITE_NAME, "decomposition_only", "decomposition"),
        SuiteRunConfig(CORE_SUITE_NAME, "self_refine_style", "self_refine"),
        SuiteRunConfig(CORE_SUITE_NAME, "reflexion_style", "reflexion"),
        SuiteRunConfig(CORE_SUITE_NAME, "full_dual_loop", "full"),
    ]

    if include_repair_ablations:
        plan.extend(
            [
                SuiteRunConfig(
                    "repair_ablations",
                    "full_no_counterexample",
                    "full",
                    disable_counterexample_repair=True,
                ),
                SuiteRunConfig(
                    "repair_ablations",
                    "full_no_rewrite",
                    "full",
                    disable_rewrite_repair=True,
                ),
                SuiteRunConfig(
                    "repair_ablations",
                    "full_plain_repair",
                    "full",
                    disable_counterexample_repair=True,
                    disable_rewrite_repair=True,
                ),
            ]
        )

    if include_budget_ablations:
        plan.extend(
            [
                SuiteRunConfig("budget_ablations", "full_spec_iter_0", "full", spec_max_iters=0),
                SuiteRunConfig("budget_ablations", "full_spec_iter_1", "full", spec_max_iters=1),
                SuiteRunConfig(
                    "budget_ablations",
                    "full_repair_iter_0",
                    "full",
                    repair_max_iters=0,
                ),
                SuiteRunConfig(
                    "budget_ablations",
                    "full_repair_iter_1",
                    "full",
                    repair_max_iters=1,
                ),
                SuiteRunConfig(
                    "budget_ablations",
                    "full_threshold_70",
                    "full",
                    spec_score_threshold=70,
                ),
                SuiteRunConfig(
                    "budget_ablations",
                    "full_threshold_90",
                    "full",
                    spec_score_threshold=90,
                ),
            ]
        )

    return plan


def apply_run_config(base_args: Namespace, config: SuiteRunConfig) -> Namespace:
    args = deepcopy(base_args)
    args.pipeline_mode = config.pipeline_mode
    args.run_tag = config.run_name
    args.disable_counterexample_repair = config.disable_counterexample_repair
    args.disable_rewrite_repair = config.disable_rewrite_repair
    if config.spec_max_iters is not None:
        args.spec_max_iters = config.spec_max_iters
    if config.repair_max_iters is not None:
        args.repair_max_iters = config.repair_max_iters
    if config.spec_score_threshold is not None:
        args.spec_score_threshold = config.spec_score_threshold
    suite_mirror_dir = getattr(base_args, "cwd_output_dir", None)
    if suite_mirror_dir:
        args.cwd_output_dir = str(Path(suite_mirror_dir) / config.run_name)
    return args


def run_rq_suite(
    base_args: Namespace,
    *,
    include_repair_ablations: bool = False,
    include_budget_ablations: bool = False,
) -> list[dict[str, Any]]:
    plan = build_rq_suite_plan(
        include_repair_ablations=include_repair_ablations,
        include_budget_ablations=include_budget_ablations,
    )
    shared_llm = LLMAdapter(base_args)
    bootstrap_pipeline = DualLoopPipeline(base_args, llm=shared_llm)
    benchmark = bootstrap_pipeline._load_benchmark()

    run_results: list[dict[str, Any]] = []
    for config in plan:
        run_args = apply_run_config(base_args, config)
        pipeline = DualLoopPipeline(run_args, llm=shared_llm)
        summary = pipeline.run(benchmark=benchmark)
        traces = _load_traces_for_summary(summary)
        run_results.append(
            {
                "config": asdict(config),
                "summary": summary,
                "traces": traces,
            }
        )
    return run_results


def build_rq_csv_rows(run_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows = [_build_raw_row(result) for result in run_results]
    by_run_name = {row["run_name"]: row for row in raw_rows}
    baseline = by_run_name.get("baseline_direct", {})
    decomposition = by_run_name.get("decomposition_only", {})
    best_iterative_baseline = max(
        by_run_name.get("self_refine_style", {}).get("pass_at_1", 0.0),
        by_run_name.get("reflexion_style", {}).get("pass_at_1", 0.0),
    )

    for row in raw_rows:
        row["delta_pass_at_1_vs_baseline"] = _round_metric(
            row["pass_at_1"] - float(baseline.get("pass_at_1", 0.0))
        )
        row["delta_pass_at_1_vs_decomposition"] = _round_metric(
            row["pass_at_1"] - float(decomposition.get("pass_at_1", 0.0))
        )
        row["best_iterative_baseline_pass_at_1"] = _round_metric(best_iterative_baseline)
        row["delta_pass_at_1_vs_best_iterative_baseline"] = _round_metric(
            row["pass_at_1"] - best_iterative_baseline
        )

        baseline_llm_calls = float(baseline.get("average_llm_calls", 0.0))
        baseline_elapsed = float(baseline.get("average_elapsed_seconds", 0.0))
        baseline_prompt_chars = float(baseline.get("average_prompt_chars", 0.0))
        baseline_completion_chars = float(baseline.get("average_completion_chars", 0.0))

        row["delta_average_llm_calls_vs_baseline"] = _round_metric(
            row["average_llm_calls"] - baseline_llm_calls
        )
        row["delta_average_elapsed_seconds_vs_baseline"] = _round_metric(
            row["average_elapsed_seconds"] - baseline_elapsed
        )
        row["delta_average_prompt_chars_vs_baseline"] = _round_metric(
            row["average_prompt_chars"] - baseline_prompt_chars
        )
        row["delta_average_completion_chars_vs_baseline"] = _round_metric(
            row["average_completion_chars"] - baseline_completion_chars
        )
        row["cost_ratio_llm_calls_vs_baseline"] = _round_metric(
            _safe_div(row["average_llm_calls"], baseline_llm_calls)
        )
        row["cost_ratio_elapsed_seconds_vs_baseline"] = _round_metric(
            _safe_div(row["average_elapsed_seconds"], baseline_elapsed)
        )

    return raw_rows


def write_rq_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    fieldnames = _ordered_csv_columns(rows)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_suite_manifest(
    run_results: list[dict[str, Any]],
    manifest_path: str | Path,
) -> None:
    manifest = []
    for result in run_results:
        summary = result["summary"]
        manifest.append(
            {
                "suite_name": result["config"]["suite_name"],
                "run_name": result["config"]["run_name"],
                "pipeline_mode": result["config"]["pipeline_mode"],
                "output_dir": summary.get("output_dir"),
                "summary_path": str(Path(summary.get("output_dir", "")) / "summary.json"),
                "traces_path": str(Path(summary.get("output_dir", "")) / "traces.json"),
            }
        )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)


def _load_traces_for_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = summary.get("output_dir", "")
    _, traces = load_run_artifacts(
        Path(output_dir) / "summary.json",
        Path(output_dir) / "traces.json",
    )
    return traces


def _build_raw_row(run_result: dict[str, Any]) -> dict[str, Any]:
    summary = run_result["summary"]
    traces = run_result["traces"]
    config = run_result["config"]
    num_problems = int(summary.get("num_problems", len(traces)) or len(traces) or 0)
    failure_counts = Counter(summary.get("failure_attribution_counts", {}))
    verifier_counts = Counter(summary.get("verifier_error_counts", {}))

    spec_score_outcomes = Counter()
    spec_refine_effects = Counter()
    repair_effects = Counter()
    repair_strategy_counts = Counter()
    repair_strategy_solved_counts = Counter()
    repair_strategy_improved_counts = Counter()
    spec_refine_reason_counts = Counter()
    stage_time_sums = defaultdict(float)

    for trace in traces:
        effectiveness = trace.get("effectiveness", {}) or {}
        initial_score = trace.get("initial_spec_score", {}) or {}
        final_score = trace.get("final_spec_score", {}) or {}
        initial_overall = float(initial_score.get("overall", 0) or 0)
        final_overall = float(final_score.get("overall", 0) or 0)
        if final_overall > initial_overall:
            spec_score_outcomes["improved"] += 1
        elif final_overall < initial_overall:
            spec_score_outcomes["regressed"] += 1
        else:
            spec_score_outcomes["no_change"] += 1

        for step in effectiveness.get("spec_refine_steps", []):
            spec_refine_effects[step.get("effect", "unknown")] += 1
            spec_refine_reason_counts[step.get("reason", "unknown")] += 1

        for step in effectiveness.get("repair_steps", []):
            strategy = step.get("strategy", "unknown")
            effect = step.get("effect", "unknown")
            repair_effects[effect] += 1
            repair_strategy_counts[strategy] += 1
            if effect == "solved":
                repair_strategy_solved_counts[strategy] += 1
            if effect == "improved":
                repair_strategy_improved_counts[strategy] += 1

        for stage_name, duration in (trace.get("stage_times", {}) or {}).items():
            stage_time_sums[stage_name] += float(duration or 0.0)

    total_failures = max(0, num_problems - int(failure_counts.get("solved", 0)))
    attributable_failures = int(failure_counts.get("spec_induced", 0)) + int(
        failure_counts.get("implementation_induced", 0)
    )
    total_repair_attempts = sum(repair_effects.values())
    total_spec_refine_attempts = sum(spec_refine_effects.values())

    average_initial_coverage = float(summary.get("average_initial_coverage", 0.0))
    average_final_coverage = float(summary.get("average_final_coverage", 0.0))
    average_initial_faithfulness = float(summary.get("average_initial_faithfulness", 0.0))
    average_final_faithfulness = float(summary.get("average_final_faithfulness", 0.0))
    average_initial_precision = float(summary.get("average_initial_precision", 0.0))
    average_final_precision = float(summary.get("average_final_precision", 0.0))

    row = {
        "suite_name": config["suite_name"],
        "run_name": config["run_name"],
        "method_label": _method_label(config["run_name"]),
        "pipeline_mode": summary.get("pipeline_mode"),
        "model": summary.get("model"),
        "release_version": summary.get("release_version"),
        "run_tag": summary.get("run_tag"),
        "num_problems": num_problems,
        "disable_counterexample_repair": config["disable_counterexample_repair"],
        "disable_rewrite_repair": config["disable_rewrite_repair"],
        "spec_max_iters": summary.get("spec_max_iters"),
        "repair_max_iters": summary.get("repair_max_iters"),
        "spec_score_threshold": summary.get("spec_score_threshold"),
        "output_dir": summary.get("output_dir"),
        "pass_at_1": _round_metric(float(summary.get("pass_at_1", 0.0))),
        "average_initial_sas": _round_metric(float(summary.get("average_initial_sas", 0.0))),
        "average_final_sas": _round_metric(float(summary.get("average_final_sas", 0.0))),
        "average_delta_sas": _round_metric(float(summary.get("average_delta_sas", 0.0))),
        "average_initial_coverage": _round_metric(average_initial_coverage),
        "average_final_coverage": _round_metric(average_final_coverage),
        "delta_coverage": _round_metric(average_final_coverage - average_initial_coverage),
        "average_initial_faithfulness": _round_metric(average_initial_faithfulness),
        "average_final_faithfulness": _round_metric(average_final_faithfulness),
        "delta_faithfulness": _round_metric(
            average_final_faithfulness - average_initial_faithfulness
        ),
        "average_initial_precision": _round_metric(average_initial_precision),
        "average_final_precision": _round_metric(average_final_precision),
        "delta_precision": _round_metric(average_final_precision - average_initial_precision),
        "spec_score_improved_problem_count": int(spec_score_outcomes.get("improved", 0)),
        "spec_score_no_change_problem_count": int(spec_score_outcomes.get("no_change", 0)),
        "spec_score_regressed_problem_count": int(spec_score_outcomes.get("regressed", 0)),
        "spec_score_improvement_rate": _round_metric(
            _safe_div(spec_score_outcomes.get("improved", 0), num_problems)
        ),
        "spec_refine_attempt_count": total_spec_refine_attempts,
        "spec_refine_artifact_changed_count": int(
            spec_refine_effects.get("artifact_changed", 0)
        ),
        "spec_refine_artifact_changed_rate": _round_metric(
            _safe_div(spec_refine_effects.get("artifact_changed", 0), total_spec_refine_attempts)
        ),
        "spec_refine_no_effect_count": int(spec_refine_effects.get("no_effect", 0)),
        "spec_refine_no_effect_rate": _round_metric(
            _safe_div(spec_refine_effects.get("no_effect", 0), total_spec_refine_attempts)
        ),
        "spec_refine_parse_failed_count": int(
            spec_refine_reason_counts.get("refine_parse_failed", 0)
        ),
        "solved_count": int(failure_counts.get("solved", 0)),
        "solved_rate": _round_metric(_safe_div(failure_counts.get("solved", 0), num_problems)),
        "spec_induced_count": int(failure_counts.get("spec_induced", 0)),
        "spec_induced_rate": _round_metric(
            _safe_div(failure_counts.get("spec_induced", 0), num_problems)
        ),
        "implementation_induced_count": int(
            failure_counts.get("implementation_induced", 0)
        ),
        "implementation_induced_rate": _round_metric(
            _safe_div(failure_counts.get("implementation_induced", 0), num_problems)
        ),
        "unknown_failure_count": int(failure_counts.get("unknown", 0)),
        "attributable_failures_count": attributable_failures,
        "attributable_failures_rate_among_failures": _round_metric(
            _safe_div(attributable_failures, total_failures)
        ),
        "verifier_wrong_answer_count": int(verifier_counts.get("wrong_answer", 0)),
        "verifier_runtime_error_count": int(verifier_counts.get("runtime_error", 0)),
        "verifier_time_limit_exceeded_count": int(
            verifier_counts.get("time_limit_exceeded", 0)
        ),
        "verifier_error_count": int(verifier_counts.get("verifier_error", 0)),
        "repair_attempt_count": total_repair_attempts,
        "repair_no_effect_count": int(repair_effects.get("no_effect", 0)),
        "repair_improved_count": int(repair_effects.get("improved", 0)),
        "repair_changed_but_not_improved_count": int(
            repair_effects.get("changed_but_not_improved", 0)
        ),
        "repair_solved_count": int(repair_effects.get("solved", 0)),
        "repair_no_effect_rate": _round_metric(
            _safe_div(repair_effects.get("no_effect", 0), total_repair_attempts)
        ),
        "repair_improved_rate": _round_metric(
            _safe_div(repair_effects.get("improved", 0), total_repair_attempts)
        ),
        "repair_changed_but_not_improved_rate": _round_metric(
            _safe_div(repair_effects.get("changed_but_not_improved", 0), total_repair_attempts)
        ),
        "repair_solved_rate": _round_metric(
            _safe_div(repair_effects.get("solved", 0), total_repair_attempts)
        ),
        "repair_strategy_repair_attempts": int(repair_strategy_counts.get("repair", 0)),
        "repair_strategy_repair_counterexample_attempts": int(
            repair_strategy_counts.get("repair_counterexample", 0)
        ),
        "repair_strategy_repair_rewrite_attempts": int(
            repair_strategy_counts.get("repair_rewrite", 0)
        ),
        "repair_strategy_repair_solved": int(repair_strategy_solved_counts.get("repair", 0)),
        "repair_strategy_repair_counterexample_solved": int(
            repair_strategy_solved_counts.get("repair_counterexample", 0)
        ),
        "repair_strategy_repair_rewrite_solved": int(
            repair_strategy_solved_counts.get("repair_rewrite", 0)
        ),
        "repair_strategy_repair_improved": int(
            repair_strategy_improved_counts.get("repair", 0)
        ),
        "repair_strategy_repair_counterexample_improved": int(
            repair_strategy_improved_counts.get("repair_counterexample", 0)
        ),
        "repair_strategy_repair_rewrite_improved": int(
            repair_strategy_improved_counts.get("repair_rewrite", 0)
        ),
        "average_llm_calls": _round_metric(float(summary.get("average_llm_calls", 0.0))),
        "average_spec_calls": _round_metric(float(summary.get("average_spec_calls", 0.0))),
        "average_judge_calls": _round_metric(float(summary.get("average_judge_calls", 0.0))),
        "average_codegen_calls": _round_metric(
            float(summary.get("average_codegen_calls", 0.0))
        ),
        "average_repair_calls": _round_metric(
            float(summary.get("average_repair_calls", 0.0))
        ),
        "average_loop_b_iterations": _round_metric(
            float(summary.get("average_loop_b_iterations", 0.0))
        ),
        "average_loop_a_iterations": _round_metric(
            float(summary.get("average_loop_a_iterations", 0.0))
        ),
        "average_repairs": _round_metric(float(summary.get("average_repairs", 0.0))),
        "average_prompt_chars": _round_metric(float(summary.get("average_prompt_chars", 0.0))),
        "average_completion_chars": _round_metric(
            float(summary.get("average_completion_chars", 0.0))
        ),
        "average_elapsed_seconds": _round_metric(
            float(summary.get("average_elapsed_seconds", 0.0))
        ),
        "average_stage_time_spec_draft": _round_metric(
            _safe_div(stage_time_sums.get("spec_draft", 0.0), num_problems)
        ),
        "average_stage_time_spec_score_initial": _round_metric(
            _safe_div(stage_time_sums.get("spec_score_initial", 0.0), num_problems)
        ),
        "average_stage_time_spec_score_refine": _round_metric(
            _safe_div(stage_time_sums.get("spec_score_refine", 0.0), num_problems)
        ),
        "average_stage_time_spec_refine": _round_metric(
            _safe_div(stage_time_sums.get("spec_refine", 0.0), num_problems)
        ),
        "average_stage_time_spec_score_final": _round_metric(
            _safe_div(stage_time_sums.get("spec_score_final", 0.0), num_problems)
        ),
        "average_stage_time_codegen": _round_metric(
            _safe_div(stage_time_sums.get("codegen", 0.0), num_problems)
        ),
        "average_stage_time_repair": _round_metric(
            _safe_div(stage_time_sums.get("repair", 0.0), num_problems)
        ),
    }
    return row


def _ordered_csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "suite_name",
        "run_name",
        "method_label",
        "pipeline_mode",
        "model",
        "release_version",
        "run_tag",
        "num_problems",
        "disable_counterexample_repair",
        "disable_rewrite_repair",
        "spec_max_iters",
        "repair_max_iters",
        "spec_score_threshold",
        "output_dir",
        "pass_at_1",
        "delta_pass_at_1_vs_baseline",
        "delta_pass_at_1_vs_decomposition",
        "best_iterative_baseline_pass_at_1",
        "delta_pass_at_1_vs_best_iterative_baseline",
        "average_initial_sas",
        "average_final_sas",
        "average_delta_sas",
        "average_initial_coverage",
        "average_final_coverage",
        "delta_coverage",
        "average_initial_faithfulness",
        "average_final_faithfulness",
        "delta_faithfulness",
        "average_initial_precision",
        "average_final_precision",
        "delta_precision",
        "spec_score_improved_problem_count",
        "spec_score_no_change_problem_count",
        "spec_score_regressed_problem_count",
        "spec_score_improvement_rate",
        "spec_refine_attempt_count",
        "spec_refine_artifact_changed_count",
        "spec_refine_artifact_changed_rate",
        "spec_refine_no_effect_count",
        "spec_refine_no_effect_rate",
        "spec_refine_parse_failed_count",
        "solved_count",
        "solved_rate",
        "spec_induced_count",
        "spec_induced_rate",
        "implementation_induced_count",
        "implementation_induced_rate",
        "unknown_failure_count",
        "attributable_failures_count",
        "attributable_failures_rate_among_failures",
        "verifier_wrong_answer_count",
        "verifier_runtime_error_count",
        "verifier_time_limit_exceeded_count",
        "verifier_error_count",
        "repair_attempt_count",
        "repair_no_effect_count",
        "repair_improved_count",
        "repair_changed_but_not_improved_count",
        "repair_solved_count",
        "repair_no_effect_rate",
        "repair_improved_rate",
        "repair_changed_but_not_improved_rate",
        "repair_solved_rate",
        "repair_strategy_repair_attempts",
        "repair_strategy_repair_counterexample_attempts",
        "repair_strategy_repair_rewrite_attempts",
        "repair_strategy_repair_solved",
        "repair_strategy_repair_counterexample_solved",
        "repair_strategy_repair_rewrite_solved",
        "repair_strategy_repair_improved",
        "repair_strategy_repair_counterexample_improved",
        "repair_strategy_repair_rewrite_improved",
        "average_llm_calls",
        "average_spec_calls",
        "average_judge_calls",
        "average_codegen_calls",
        "average_repair_calls",
        "average_loop_b_iterations",
        "average_loop_a_iterations",
        "average_repairs",
        "average_prompt_chars",
        "average_completion_chars",
        "average_elapsed_seconds",
        "delta_average_llm_calls_vs_baseline",
        "delta_average_elapsed_seconds_vs_baseline",
        "delta_average_prompt_chars_vs_baseline",
        "delta_average_completion_chars_vs_baseline",
        "cost_ratio_llm_calls_vs_baseline",
        "cost_ratio_elapsed_seconds_vs_baseline",
        "average_stage_time_spec_draft",
        "average_stage_time_spec_score_initial",
        "average_stage_time_spec_score_refine",
        "average_stage_time_spec_refine",
        "average_stage_time_spec_score_final",
        "average_stage_time_codegen",
        "average_stage_time_repair",
    ]
    seen = set()
    columns = []
    for column in preferred:
        if any(column in row for row in rows):
            columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _method_label(run_name: str) -> str:
    labels = {
        "baseline_direct": "Direct NL->Code",
        "decomposition_only": "Decomposition Only",
        "self_refine_style": "Self-Refine-style",
        "reflexion_style": "Reflexion-style",
        "full_dual_loop": "Full Dual-Loop",
        "full_no_counterexample": "Full w/o Counterexample Repair",
        "full_no_rewrite": "Full w/o Rewrite Repair",
        "full_plain_repair": "Full Plain Repair",
        "full_spec_iter_0": "Full spec_max_iters=0",
        "full_spec_iter_1": "Full spec_max_iters=1",
        "full_repair_iter_0": "Full repair_max_iters=0",
        "full_repair_iter_1": "Full repair_max_iters=1",
        "full_threshold_70": "Full threshold=70",
        "full_threshold_90": "Full threshold=90",
    }
    return labels.get(run_name, run_name)
