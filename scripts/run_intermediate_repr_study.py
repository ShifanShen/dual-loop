import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter
from lcb_runner.dual_loop.prompts import (
    build_code_from_plan_prompt,
    build_code_from_pseudocode_prompt,
    build_plan_draft_prompt,
    build_pseudocode_draft_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a focused intermediate-representation study: direct vs spec vs plan vs pseudocode."
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--model_style", type=str, default=None)
    parser.add_argument("--model_repr", type=str, default=None)
    parser.add_argument("--release_version", type=str, default="release_latest")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--question_ids", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=50)
    parser.add_argument("--output_root", type=str, default="output/intermediate_repr_study")
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
    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.2)
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)
    args = parser.parse_args()
    args.stop = args.stop.split(",")
    args.pipeline_mode = "baseline"
    args.disable_counterexample_repair = False
    args.disable_rewrite_repair = False
    args.spec_max_iters = 0
    args.repair_max_iters = 0
    args.spec_score_threshold = 90
    args.spec_min_improvement = 1
    args.spec_precision_floor = 85
    args.spec_max_rejected_refines = 1
    args.spec_skip_ambiguity_only = True
    args.run_tag = None
    args.cwd_output_dir = None
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


def _usage_totals(usages: list[dict[str, int | str]]) -> dict[str, float]:
    prompt_chars = sum(int(item.get("prompt_chars", 0)) for item in usages)
    completion_chars = sum(int(item.get("completion_chars", 0)) for item in usages)
    return {
        "llm_calls": float(len(usages)),
        "prompt_chars": float(prompt_chars),
        "completion_chars": float(completion_chars),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _run_direct(pipeline: DualLoopPipeline, problem) -> tuple[dict, str]:
    started = time.perf_counter()
    code = pipeline._generate_code_baseline(problem)
    feedback = pipeline._verify(problem, code)
    usages = [getattr(pipeline, "_last_baseline_codegen_usage", None)] + list(
        getattr(pipeline, "_last_baseline_codegen_extra_usages", [])
    )
    usages = [item for item in usages if item]
    stats = _usage_totals(usages)
    stats["elapsed_seconds"] = time.perf_counter() - started
    trace = {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "method": "direct",
        "intermediate_type": "none",
        "intermediate_artifact": "",
        "final_code": code,
        "passed": bool(feedback.passed),
        "verifier_feedback": asdict(feedback),
        **stats,
    }
    return trace, code


def _run_spec(pipeline: DualLoopPipeline, problem) -> tuple[dict, str]:
    started = time.perf_counter()
    spec = pipeline._draft_spec(problem)
    code = pipeline._generate_code_from_spec(problem, spec)
    feedback = pipeline._verify(problem, code, spec=spec)
    usages = [
        getattr(spec, "_usage", None),
        *getattr(spec, "_extra_usages", []),
        getattr(spec, "_last_codegen_usage", None),
        *getattr(spec, "_last_codegen_extra_usages", []),
    ]
    usages = [item for item in usages if item]
    stats = _usage_totals(usages)
    stats["elapsed_seconds"] = time.perf_counter() - started
    trace = {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "method": "spec",
        "intermediate_type": "structured_spec",
        "intermediate_artifact": spec.to_text(),
        "spec_payload": asdict(spec),
        "final_code": code,
        "passed": bool(feedback.passed),
        "verifier_feedback": asdict(feedback),
        **stats,
    }
    return trace, code


def _run_text_intermediate(
    pipeline: DualLoopPipeline,
    problem,
    *,
    method: str,
    draft_prompt_builder,
    code_prompt_builder,
) -> tuple[dict, str]:
    started = time.perf_counter()
    draft_output, draft_usage = pipeline.llm.generate(
        draft_prompt_builder(problem),
        role=f"{method}_draft",
        temperature=pipeline.args.spec_temperature,
        max_tokens=pipeline.args.spec_max_tokens,
    )
    code_output, code_usage = pipeline.llm.generate(
        code_prompt_builder(problem, draft_output.strip()),
        role=f"{method}_codegen",
        temperature=pipeline.args.codegen_temperature,
        max_tokens=pipeline.args.codegen_max_tokens,
    )
    code, extra_outputs, extra_usages = pipeline._extract_valid_code(code_output)
    feedback = pipeline._verify(problem, code)
    usages = [draft_usage, code_usage, *extra_usages]
    stats = _usage_totals(usages)
    stats["elapsed_seconds"] = time.perf_counter() - started
    trace = {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "method": method,
        "intermediate_type": method,
        "intermediate_artifact": draft_output.strip(),
        "raw_codegen_output": code_output,
        "raw_codegen_repairs": extra_outputs,
        "final_code": code,
        "passed": bool(feedback.passed),
        "verifier_feedback": asdict(feedback),
        **stats,
    }
    return trace, code


def _aggregate(method: str, traces: list[dict]) -> dict[str, float | str | int]:
    return {
        "method": method,
        "num_problems": len(traces),
        "pass_at_1": round(mean(float(trace["passed"]) for trace in traces), 4) if traces else 0.0,
        "average_llm_calls": round(mean(trace["llm_calls"] for trace in traces), 2) if traces else 0.0,
        "average_prompt_chars": round(mean(trace["prompt_chars"] for trace in traces), 2) if traces else 0.0,
        "average_completion_chars": round(mean(trace["completion_chars"] for trace in traces), 2) if traces else 0.0,
        "average_elapsed_seconds": round(mean(trace["elapsed_seconds"] for trace in traces), 2) if traces else 0.0,
    }


def main() -> None:
    args = parse_args()
    llm = LLMAdapter(args)
    pipeline = DualLoopPipeline(args, llm=llm)
    benchmark = pipeline._load_benchmark()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_repr = (args.model_repr or args.model or "model").replace("/", "_").replace(" ", "_")
    output_dir = Path(args.output_root) / f"ir_study_{model_repr}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_by_method: dict[str, list[dict]] = {
        "direct": [],
        "plan": [],
        "pseudocode": [],
        "spec": [],
    }

    for problem in benchmark:
        direct_trace, _ = _run_direct(pipeline, problem)
        traces_by_method["direct"].append(direct_trace)

        plan_trace, _ = _run_text_intermediate(
            pipeline,
            problem,
            method="plan",
            draft_prompt_builder=build_plan_draft_prompt,
            code_prompt_builder=build_code_from_plan_prompt,
        )
        traces_by_method["plan"].append(plan_trace)

        pseudocode_trace, _ = _run_text_intermediate(
            pipeline,
            problem,
            method="pseudocode",
            draft_prompt_builder=build_pseudocode_draft_prompt,
            code_prompt_builder=build_code_from_pseudocode_prompt,
        )
        traces_by_method["pseudocode"].append(pseudocode_trace)

        spec_trace, _ = _run_spec(pipeline, problem)
        traces_by_method["spec"].append(spec_trace)

    rows = [_aggregate(method, traces) for method, traces in traces_by_method.items()]
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for method, traces in traces_by_method.items():
        _write_json(output_dir / f"{method}_traces.json", traces)

    summary = {
        "model": args.model,
        "model_repr": args.model_repr,
        "release_version": args.release_version,
        "max_problems": args.max_problems,
        "output_dir": str(output_dir),
        "results_csv": str(csv_path),
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
