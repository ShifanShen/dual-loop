import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter
from lcb_runner.dual_loop.recent_baselines import (
    METHODS,
    aggregate_traces,
    append_checkpoint,
    apply_final_evaluation,
    read_checkpoint,
    run_lpw_adapted,
    run_specfix_bm,
    write_json,
    write_main_table_csv,
    write_results_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SpecFix-BM and LPW-adapted under the same LiveCodeBench held-out "
            "protocol used by the Dual-Loop main comparison."
        )
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--model_style", type=str, default=None)
    parser.add_argument("--model_repr", type=str, default=None)
    parser.add_argument("--release_version", type=str, default="release_v6")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--question_ids", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=1055)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS),
        help="Comma-separated methods: specfix_bm,lpw_adapted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/recent_nl2code_baselines/paper_main",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Fail instead of resuming from JSONL checkpoints.",
    )
    parser.set_defaults(resume=True)

    parser.add_argument(
        "--feedback_test_scope",
        choices=["public"],
        default="public",
    )
    parser.add_argument(
        "--final_test_scope",
        choices=["private"],
        default="private",
    )
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--num_process_evaluate", type=int, default=8)

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

    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.2)
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)

    parser.add_argument("--max_program_candidates", type=int, default=11)
    parser.add_argument("--specfix_probe_candidates", type=int, default=4)
    parser.add_argument("--specfix_probe_tests", type=int, default=4)
    parser.add_argument("--codegen_num_candidates", type=int, default=2)
    parser.add_argument("--repair_num_candidates", type=int, default=3)
    parser.add_argument("--repair_max_iters", type=int, default=3)
    parser.add_argument("--lpw_plan_max_iters", type=int, default=3)
    parser.add_argument("--lpw_plan_example_limit", type=int, default=3)

    args = parser.parse_args()
    args.stop = args.stop.split(",")
    args.pipeline_mode = "baseline"
    args.run_tag = "recent_baseline_helper"
    args.output_root = str(Path(args.output_dir) / "_pipeline_helper")
    args.cwd_output_dir = None

    args.spec_max_iters = 0
    args.spec_score_threshold = 90
    args.spec_min_improvement = 1
    args.spec_precision_floor = 85
    args.spec_max_rejected_refines = 1
    args.spec_skip_ambiguity_only = True
    args.disable_counterexample_repair = False
    args.disable_rewrite_repair = False
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
    if not selected_methods:
        raise ValueError("At least one method must be selected")
    args.selected_methods = selected_methods

    if args.specfix_probe_candidates < 2:
        raise ValueError("SpecFix-BM requires at least two probe programs")
    if args.specfix_probe_candidates > args.max_program_candidates:
        raise ValueError("SpecFix probe candidates exceed the program budget")
    lpw_program_budget = (
        args.codegen_num_candidates
        + args.repair_num_candidates * args.repair_max_iters
    )
    if lpw_program_budget != args.max_program_candidates:
        raise ValueError(
            "For a budget-matched comparison, max_program_candidates must equal "
            "codegen_num_candidates + repair_num_candidates * repair_max_iters; "
            f"got {args.max_program_candidates} and {lpw_program_budget}."
        )
    return args


def _protocol_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_repr": args.model_repr,
        "local_model_path": args.local_model_path,
        "model_style": args.model_style,
        "release_version": args.release_version,
        "dataset_path": args.dataset_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "question_ids": args.question_ids,
        "feedback_test_scope": args.feedback_test_scope,
        "final_test_scope": args.final_test_scope,
        "timeout": args.timeout,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "spec_temperature": args.spec_temperature,
        "judge_temperature": args.judge_temperature,
        "codegen_temperature": args.codegen_temperature,
        "repair_temperature": args.repair_temperature,
        "spec_max_tokens": args.spec_max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "codegen_max_tokens": args.codegen_max_tokens,
        "max_program_candidates": args.max_program_candidates,
        "specfix_probe_candidates": args.specfix_probe_candidates,
        "specfix_probe_tests": args.specfix_probe_tests,
        "codegen_num_candidates": args.codegen_num_candidates,
        "repair_num_candidates": args.repair_num_candidates,
        "repair_max_iters": args.repair_max_iters,
        "lpw_plan_max_iters": args.lpw_plan_max_iters,
        "lpw_plan_example_limit": args.lpw_plan_example_limit,
        "private_tests_available_during_generation": False,
        "program_budget_definition": (
            "maximum primary program-candidate slots per problem; planning, test "
            "generation, requirement repair, error analysis, and code-format recovery "
            "calls are reported overhead"
        ),
    }


def _validate_or_write_config(
    path: Path,
    protocol: dict[str, Any],
    *,
    selected_methods: list[str],
    max_problems: int,
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        previous_protocol = existing.get("protocol", {})
        mismatches = [
            key
            for key, value in protocol.items()
            if previous_protocol.get(key) != value
        ]
        if mismatches:
            details = ", ".join(
                f"{key}: {previous_protocol.get(key)!r} != {protocol.get(key)!r}"
                for key in mismatches
            )
            raise ValueError(
                "Refusing to mix incompatible checkpoints in one output directory: "
                + details
            )

    write_json(
        path,
        {
            "protocol": protocol,
            "selected_methods": selected_methods,
            "max_problems": max_problems,
        },
    )


def _ordered_traces(
    benchmark: list[Any],
    checkpoint: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [
        problem.question_id
        for problem in benchmark
        if problem.question_id not in checkpoint
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Generation checkpoint is incomplete for {len(missing)} problem(s): {preview}"
        )
    return [checkpoint[problem.question_id] for problem in benchmark]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol_config(args)
    _validate_or_write_config(
        output_dir / "run_config.json",
        protocol,
        selected_methods=args.selected_methods,
        max_problems=args.max_problems,
    )

    checkpoints = {
        method: read_checkpoint(output_dir / f"{method}_checkpoint.jsonl")
        for method in args.selected_methods
    }
    if not args.resume and any(checkpoints.values()):
        raise FileExistsError(
            "Checkpoint data already exists. Use the default resume mode or choose "
            "a new output_dir."
        )

    llm = LLMAdapter(args)
    pipeline = DualLoopPipeline(args, llm=llm)
    benchmark = pipeline._load_benchmark()

    progress = {
        method: len(
            set(checkpoints[method]).intersection(
                problem.question_id for problem in benchmark
            )
        )
        for method in args.selected_methods
    }
    print(
        json.dumps(
            {
                "event": "start",
                "output_dir": str(output_dir),
                "num_problems": len(benchmark),
                "methods": args.selected_methods,
                "resumed_counts": progress,
                "protocol": protocol,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    for problem_index, problem in enumerate(benchmark, start=1):
        for method in args.selected_methods:
            if problem.question_id in checkpoints[method]:
                continue
            print(
                f"[{problem_index}/{len(benchmark)}] {method} {problem.question_id}",
                flush=True,
            )
            if method == "specfix_bm":
                trace = run_specfix_bm(
                    pipeline,
                    problem,
                    probe_candidate_count=args.specfix_probe_candidates,
                    probe_test_count=args.specfix_probe_tests,
                    max_program_candidates=args.max_program_candidates,
                )
            elif method == "lpw_adapted":
                trace = run_lpw_adapted(
                    pipeline,
                    problem,
                    plan_max_iters=args.lpw_plan_max_iters,
                    plan_example_limit=args.lpw_plan_example_limit,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            append_checkpoint(
                output_dir / f"{method}_checkpoint.jsonl",
                trace,
            )
            checkpoints[method][problem.question_id] = trace
            progress[method] += 1
            write_json(
                output_dir / "progress.json",
                {
                    "num_problems": len(benchmark),
                    "completed": progress,
                    "last_question_id": problem.question_id,
                    "last_method": method,
                },
            )

    rows: list[dict[str, Any]] = []
    final_metrics: dict[str, dict[str, float | int]] = {}
    for method in args.selected_methods:
        traces = _ordered_traces(benchmark, checkpoints[method])
        evaluation_metrics = apply_final_evaluation(pipeline, benchmark, traces)
        final_metrics[method] = evaluation_metrics
        write_json(output_dir / f"{method}_traces.json", traces)
        row = aggregate_traces(method, traces)
        row["metric_pass_at_1"] = round(
            float(evaluation_metrics["combined_pass_rate"]), 6
        )
        row["metric_private_pass_at_1"] = round(
            float(evaluation_metrics["metric_private_pass_at_1"]), 6
        )
        rows.append(row)

    write_results_csv(output_dir / "results.csv", rows)
    write_main_table_csv(output_dir / "main_table_rows.csv", rows)
    summary = {
        "model": args.model,
        "model_repr": args.model_repr,
        "local_model_path": args.local_model_path,
        "release_version": args.release_version,
        "num_problems": len(benchmark),
        "methods": args.selected_methods,
        "verifier_protocol": "public_feedback_private_heldout_final",
        "output_dir": str(output_dir),
        "protocol": protocol,
        "final_metrics_by_method": final_metrics,
        "rows": rows,
        "main_table_rows": str(output_dir / "main_table_rows.csv"),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "progress.json",
        {
            "num_problems": len(benchmark),
            "completed": progress,
            "status": "complete",
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
