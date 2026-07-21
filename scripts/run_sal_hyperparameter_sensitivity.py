import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter
from lcb_runner.dual_loop.prompts import (
    build_spec_score_json_repair_prompt,
    build_spec_score_prompt,
)
from lcb_runner.dual_loop.spec import SpecScore, StructuredSpec


@dataclass(frozen=True)
class WeightProfile:
    name: str
    coverage: float
    faithfulness: float
    precision: float

    @property
    def values(self) -> tuple[float, float, float]:
        return self.coverage, self.faithfulness, self.precision


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    threshold: int
    weights: WeightProfile
    sweep: str


class SensitivityPipeline(DualLoopPipeline):
    def __init__(
        self,
        args: argparse.Namespace,
        llm: LLMAdapter,
        weights: WeightProfile,
    ):
        self.sas_weights = weights
        self.score_audit: list[dict[str, Any]] = []
        super().__init__(args, llm=llm)

    def _score_spec(self, problem, spec: StructuredSpec) -> SpecScore:
        started_at = time.perf_counter()
        prompt = self._weighted_score_prompt(problem, spec)
        output, usage = self.llm.generate(
            prompt,
            role="judge",
            temperature=self.args.judge_temperature,
            max_tokens=self.args.judge_max_tokens,
        )
        raw_attempt_outputs = [output]
        extra_usages = []
        score = SpecScore.from_llm_output(output)
        if not score.parse_ok:
            repair_output, repair_usage = self.llm.generate(
                build_spec_score_json_repair_prompt(output),
                role="judge_json_repair",
                temperature=0.0,
                max_tokens=self.args.judge_max_tokens,
            )
            raw_attempt_outputs.append(repair_output)
            extra_usages.append(repair_usage)
            repaired_score = SpecScore.from_llm_output(repair_output)
            if repaired_score.parse_ok:
                output = repair_output
                score = repaired_score
        score = self._sanitize_spec_score(score)
        judge_overall = int(score.overall)
        weighted_overall = round(
            self.sas_weights.coverage * score.coverage
            + self.sas_weights.faithfulness * score.faithfulness
            + self.sas_weights.precision * score.precision
        )
        score.overall = max(0, min(100, int(weighted_overall)))
        score._raw_output = output
        score._raw_attempt_outputs = raw_attempt_outputs
        score._usage = usage
        score._extra_usages = extra_usages
        score._stage_time = time.perf_counter() - started_at
        self.score_audit.append(
            {
                "question_id": problem.question_id,
                "coverage": score.coverage,
                "faithfulness": score.faithfulness,
                "precision": score.precision,
                "judge_reported_overall": judge_overall,
                "recomputed_overall": score.overall,
                "weights": list(self.sas_weights.values),
                "parse_ok": score.parse_ok,
            }
        )
        return score

    def _weighted_score_prompt(self, problem, spec: StructuredSpec) -> str:
        prompt = build_spec_score_prompt(problem, spec)
        original = (
            "overall: rounded weighted score using 0.4 coverage, "
            "0.4 faithfulness, 0.2 precision"
        )
        replacement = (
            "overall: rounded weighted score using "
            f"{self.sas_weights.coverage:.6f} coverage, "
            f"{self.sas_weights.faithfulness:.6f} faithfulness, "
            f"{self.sas_weights.precision:.6f} precision"
        )
        if original not in prompt:
            raise RuntimeError("Could not locate the SAS weighting rule in the judge prompt.")
        return prompt.replace(original, replacement)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a public-feedback-only sensitivity study for the SAL threshold "
            "and SAS weights."
        )
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--model_style", type=str, default=None)
    parser.add_argument("--model_repr", type=str, default=None)
    parser.add_argument("--release_version", type=str, default="release_v6")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--question_ids", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=30)
    parser.add_argument("--sample_seed", type=int, default=2027)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="80,85,90,95",
    )
    parser.add_argument("--reference_threshold", type=int, default=90)
    parser.add_argument(
        "--weight_profiles",
        type=str,
        default=(
            "paper:0.4,0.4,0.2;"
            "uniform:0.333333,0.333333,0.333334;"
            "coverage:0.6,0.2,0.2;"
            "faithfulness:0.2,0.6,0.2;"
            "precision:0.2,0.2,0.6"
        ),
    )
    parser.add_argument("--reference_weight", type=str, default="paper")
    parser.add_argument(
        "--suite_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192") or 8192),
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
    parser.add_argument("--spec_max_iters", type=int, default=3)
    parser.add_argument("--spec_min_improvement", type=int, default=1)
    parser.add_argument("--spec_precision_floor", type=int, default=85)
    parser.add_argument("--spec_max_rejected_refines", type=int, default=1)
    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.2)
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)
    args = parser.parse_args()
    args.stop = args.stop.split(",")
    if args.local_model_path:
        inferred_name = Path(args.local_model_path).name or "LocalModel"
        args.model = args.model or inferred_name
        args.model_repr = args.model_repr or inferred_name
    elif args.model is None:
        args.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    args.pipeline_mode = "loop_b"
    args.feedback_test_scope = "public"
    args.final_test_scope = "private"
    args.repair_max_iters = 0
    args.codegen_num_candidates = 1
    args.repair_num_candidates = 1
    args.adaptive_candidate_budget = False
    args.adaptive_codegen_max_candidates = 1
    args.adaptive_repair_max_candidates = 1
    args.adaptive_sal_threshold = 0.0
    args.adaptive_ablation_threshold = 0.0
    args.post_failure_sal_max_iters = 0
    args.contract_search_population_size = 1
    args.contract_search_rounds = 0
    args.contract_search_top_k = 1
    args.contract_search_codegen_top_k = 1
    args.contract_search_temperature = 0.35
    args.spec_skip_ambiguity_only = True
    args.disable_counterexample_repair = True
    args.disable_rewrite_repair = True
    args.attribution_mode = "conservative"
    args.attribution_spec_margin = 5
    args.run_tag = None
    args.cwd_output_dir = None
    args.output_root = "output/sal_hparam_sensitivity/internal"
    if args.tensor_parallel_size == -1:
        import torch

        args.tensor_parallel_size = torch.cuda.device_count()
    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()
    return args


def parse_weight_profiles(raw: str) -> dict[str, WeightProfile]:
    profiles: dict[str, WeightProfile] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        name, separator, values_text = item.partition(":")
        if not separator:
            raise ValueError(f"Invalid weight profile: {item}")
        values = [float(value.strip()) for value in values_text.split(",")]
        if len(values) != 3 or any(value < 0 for value in values):
            raise ValueError(f"Invalid weight profile: {item}")
        total = sum(values)
        if total <= 0:
            raise ValueError(f"Weight profile sums to zero: {item}")
        normalized = [value / total for value in values]
        clean_name = name.strip().lower().replace(" ", "_")
        profiles[clean_name] = WeightProfile(clean_name, *normalized)
    if not profiles:
        raise ValueError("At least one weight profile is required.")
    return profiles


def build_matrix(args: argparse.Namespace) -> list[ExperimentConfig]:
    thresholds = [int(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    profiles = parse_weight_profiles(args.weight_profiles)
    if args.reference_weight not in profiles:
        raise ValueError(f"Unknown reference weight profile: {args.reference_weight}")
    reference = profiles[args.reference_weight]
    matrix: list[ExperimentConfig] = []
    for threshold in thresholds:
        matrix.append(
            ExperimentConfig(
                config_id=f"threshold_r{threshold}_{reference.name}",
                threshold=threshold,
                weights=reference,
                sweep="threshold",
            )
        )
    for profile in profiles.values():
        if profile.name == reference.name:
            continue
        matrix.append(
            ExperimentConfig(
                config_id=f"weights_r{args.reference_threshold}_{profile.name}",
                threshold=args.reference_threshold,
                weights=profile,
                sweep="weights",
            )
        )
    return matrix


def load_fixed_sample(args: argparse.Namespace, suite_dir: Path) -> list[Any]:
    benchmark = load_code_generation_dataset(
        args.release_version,
        start_date=args.start_date,
        end_date=args.end_date,
        dataset_path=args.dataset_path,
    )
    benchmark = sorted(benchmark, key=lambda problem: problem.question_id)
    id_path = suite_dir / "question_ids.json"
    selected_ids: list[str]
    if args.question_ids:
        selected_ids = [value.strip() for value in args.question_ids.split(",") if value.strip()]
    elif args.resume and id_path.exists():
        selected_ids = json.loads(id_path.read_text(encoding="utf-8"))
    else:
        count = min(args.max_problems, len(benchmark))
        selected = random.Random(args.sample_seed).sample(benchmark, count)
        selected_ids = sorted(problem.question_id for problem in selected)
    selected_set = set(selected_ids)
    sample = [problem for problem in benchmark if problem.question_id in selected_set]
    missing = sorted(selected_set - {problem.question_id for problem in sample})
    if missing:
        raise ValueError(f"Missing requested question IDs: {missing[:5]}")
    id_path.write_text(json.dumps(selected_ids, indent=2), encoding="utf-8")
    return sample


def run_configuration(
    base_args: argparse.Namespace,
    shared_llm: LLMAdapter,
    benchmark: list[Any],
    config: ExperimentConfig,
    suite_dir: Path,
) -> dict[str, Any]:
    run_dir = suite_dir / "runs" / config.config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if base_args.resume and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    args = copy.deepcopy(base_args)
    args.spec_score_threshold = config.threshold
    args.run_tag = config.config_id
    args.output_root = str(run_dir / "internal")
    pipeline = SensitivityPipeline(args, shared_llm, config.weights)
    partial_path = run_dir / "traces.partial.json"
    score_partial_path = run_dir / "score_audit.partial.json"
    trace_rows: list[dict[str, Any]] = []
    if base_args.resume and partial_path.exists():
        trace_rows = json.loads(partial_path.read_text(encoding="utf-8"))
    if base_args.resume and score_partial_path.exists():
        pipeline.score_audit = json.loads(score_partial_path.read_text(encoding="utf-8"))
    completed = {row["question_id"] for row in trace_rows}
    for index, problem in enumerate(benchmark, 1):
        if problem.question_id in completed:
            continue
        started_at = datetime.now()
        trace = pipeline._run_problem(problem)
        trace.elapsed_seconds = (datetime.now() - started_at).total_seconds()
        trace_rows.append(asdict(trace))
        partial_path.write_text(
            json.dumps(trace_rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        score_partial_path.write_text(
            json.dumps(pipeline.score_audit, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(
            f"[{config.config_id}] {index}/{len(benchmark)} "
            f"qid={problem.question_id} public_pass={trace.passed}",
            flush=True,
        )
    trace_rows.sort(key=lambda row: row["question_id"])
    (run_dir / "traces.json").write_text(
        json.dumps(trace_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (run_dir / "score_audit.json").write_text(
        json.dumps(pipeline.score_audit, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    result = summarize_configuration(config, trace_rows, pipeline.score_audit, run_dir)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize_configuration(
    config: ExperimentConfig,
    traces: list[dict[str, Any]],
    score_audit: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    count = len(traces)
    public_passes = sum(bool(trace.get("passed")) for trace in traces)
    initial_scores = [
        float((trace.get("initial_spec_score") or {}).get("overall", 0)) for trace in traces
    ]
    final_scores = [
        float((trace.get("final_spec_score") or {}).get("overall", 0)) for trace in traces
    ]
    refine_steps = [
        step
        for trace in traces
        for step in (trace.get("effectiveness", {}).get("spec_refine_steps", []) or [])
    ]
    activated_questions = sum(
        any(int(step.get("attempt_index", 0) or 0) > 0 for step in (
            trace.get("effectiveness", {}).get("spec_refine_steps", []) or []
        ))
        for trace in traces
    )
    accepted_steps = sum(bool(step.get("accepted")) for step in refine_steps)
    artifact_changes = sum(
        bool(step.get("accepted")) and bool(step.get("artifact_changed"))
        for step in refine_steps
    )
    arithmetic_errors = [
        abs(int(row["judge_reported_overall"]) - int(row["recomputed_overall"]))
        for row in score_audit
        if row.get("parse_ok")
    ]
    llm_calls = [int(trace.get("llm_calls", 0) or 0) for trace in traces]
    return {
        "config_id": config.config_id,
        "sweep": config.sweep,
        "threshold": config.threshold,
        "weight_profile": config.weights.name,
        "weight_coverage": config.weights.coverage,
        "weight_faithfulness": config.weights.faithfulness,
        "weight_precision": config.weights.precision,
        "num_problems": count,
        "public_passes": public_passes,
        "public_pass_rate": public_passes / count if count else 0.0,
        "sal_activated_questions": activated_questions,
        "sal_activation_rate": activated_questions / count if count else 0.0,
        "accepted_refine_steps": accepted_steps,
        "artifact_changing_refines": artifact_changes,
        "mean_initial_sas": mean(initial_scores) if initial_scores else 0.0,
        "mean_final_sas": mean(final_scores) if final_scores else 0.0,
        "mean_delta_sas": mean(
            final - initial for initial, final in zip(initial_scores, final_scores)
        ) if initial_scores else 0.0,
        "mean_llm_calls": mean(llm_calls) if llm_calls else 0.0,
        "mean_judge_arithmetic_error": mean(arithmetic_errors) if arithmetic_errors else 0.0,
        "run_dir": str(run_dir),
    }


def write_aggregate_outputs(
    suite_dir: Path,
    matrix: list[ExperimentConfig],
    results: list[dict[str, Any]],
) -> None:
    ordered = {config.config_id: index for index, config in enumerate(matrix)}
    results.sort(key=lambda row: ordered[row["config_id"]])
    fields = [
        "config_id",
        "sweep",
        "threshold",
        "weight_profile",
        "weight_coverage",
        "weight_faithfulness",
        "weight_precision",
        "num_problems",
        "public_passes",
        "public_pass_rate",
        "sal_activated_questions",
        "sal_activation_rate",
        "accepted_refine_steps",
        "artifact_changing_refines",
        "mean_initial_sas",
        "mean_final_sas",
        "mean_delta_sas",
        "mean_llm_calls",
        "mean_judge_arithmetic_error",
        "run_dir",
    ]
    with (suite_dir / "sensitivity_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    per_problem_fields = [
        "config_id",
        "question_id",
        "public_passed",
        "initial_sas",
        "final_sas",
        "sal_activated",
        "accepted_refines",
        "llm_calls",
    ]
    with (suite_dir / "sensitivity_per_problem.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_problem_fields)
        writer.writeheader()
        for result in results:
            traces = json.loads(
                (Path(result["run_dir"]) / "traces.json").read_text(encoding="utf-8")
            )
            for trace in traces:
                steps = trace.get("effectiveness", {}).get("spec_refine_steps", []) or []
                writer.writerow(
                    {
                        "config_id": result["config_id"],
                        "question_id": trace["question_id"],
                        "public_passed": bool(trace.get("passed")),
                        "initial_sas": (trace.get("initial_spec_score") or {}).get("overall", 0),
                        "final_sas": (trace.get("final_spec_score") or {}).get("overall", 0),
                        "sal_activated": any(
                            int(step.get("attempt_index", 0) or 0) > 0 for step in steps
                        ),
                        "accepted_refines": sum(bool(step.get("accepted")) for step in steps),
                        "llm_calls": int(trace.get("llm_calls", 0) or 0),
                    }
                )
    write_markdown_table(suite_dir, results)
    write_latex_table(suite_dir, results)


def write_markdown_table(suite_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# SAL Hyperparameter Sensitivity",
        "",
        "This study uses only public feedback tests and does not query held-out private tests.",
        "The pipeline is SAL plus one contract-conditioned code candidate, without IRL or contract search.",
        "",
        "| Config | r | Weights (cov, faith, prec) | Public pass | SAL activation | Accepted revisions | Mean SAS delta | Mean calls |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        weights = (
            f"({row['weight_coverage']:.3f}, {row['weight_faithfulness']:.3f}, "
            f"{row['weight_precision']:.3f})"
        )
        lines.append(
            f"| {row['config_id']} | {row['threshold']} | {weights} | "
            f"{row['public_pass_rate']:.3f} ({row['public_passes']}/{row['num_problems']}) | "
            f"{row['sal_activation_rate']:.3f} | {row['accepted_refine_steps']} | "
            f"{row['mean_delta_sas']:.2f} | {row['mean_llm_calls']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: treat differences of one problem or less as descriptive ties, then prefer the setting with fewer mean calls. Use the held-out private partition only after the configuration remains frozen.",
        ]
    )
    (suite_dir / "sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(suite_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Setting & $r$ & $\mathbf{w}$ & Public Pass & SAL Act. \\",
        r"\midrule",
    ]
    for row in results:
        weights = (
            f"({row['weight_coverage']:.2f},{row['weight_faithfulness']:.2f},"
            f"{row['weight_precision']:.2f})"
        )
        setting = row["config_id"].replace("_", r"\_")
        lines.append(
            f"{setting} & {row['threshold']} & ${weights}$ & "
            f"{row['public_pass_rate']:.3f} & {row['sal_activation_rate']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Public-feedback-only sensitivity of the SAL activation threshold and SAS weights.}",
            r"\label{tab:sal-sensitivity}",
            r"\end{table}",
        ]
    )
    (suite_dir / "sensitivity_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_design(
    args: argparse.Namespace,
    suite_dir: Path,
    matrix: list[ExperimentConfig],
) -> None:
    payload = {
        "purpose": "Sensitivity of the SAL threshold and SAS aggregation weights",
        "private_tests_used": False,
        "pipeline_mode": "loop_b",
        "sample_seed": args.sample_seed,
        "max_problems": args.max_problems,
        "reference_threshold": args.reference_threshold,
        "reference_weight": args.reference_weight,
        "fixed_controls": {
            "spec_max_iters": args.spec_max_iters,
            "spec_min_improvement": args.spec_min_improvement,
            "spec_precision_floor": args.spec_precision_floor,
            "spec_max_rejected_refines": args.spec_max_rejected_refines,
            "codegen_num_candidates": 1,
            "repair_max_iters": 0,
            "contract_search_rounds": 0,
            "feedback_test_scope": "public",
        },
        "matrix": [
            {
                "config_id": config.config_id,
                "sweep": config.sweep,
                "threshold": config.threshold,
                "weights": list(config.weights.values),
            }
            for config in matrix
        ],
        "normalization": "Each supplied weight triple is normalized to sum to one.",
    }
    (suite_dir / "experiment_design.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    matrix = build_matrix(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.suite_dir or f"output/sal_hparam_sensitivity/{timestamp}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_design(args, suite_dir, matrix)
    print(json.dumps({
        "suite_dir": str(suite_dir),
        "num_configurations": len(matrix),
        "max_problems": args.max_problems,
        "private_tests_used": False,
        "matrix": [asdict(config) for config in matrix],
    }, indent=2, default=lambda value: asdict(value)))
    if args.dry_run:
        return
    benchmark = load_fixed_sample(args, suite_dir)
    shared_llm = LLMAdapter(args)
    results = [
        run_configuration(args, shared_llm, benchmark, config, suite_dir)
        for config in matrix
    ]
    write_aggregate_outputs(suite_dir, matrix, results)
    print(f"Sensitivity study complete: {suite_dir}")


if __name__ == "__main__":
    main()
