import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.rq_suite import run_rq_suite, build_rq_csv_rows, write_rq_csv, write_suite_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dual-loop RQ experiment suite once and export a paper-ready CSV."
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
    parser.add_argument("--max_problems", type=int, default=10)
    parser.add_argument("--output_root", type=str, default="output/dual_loop")
    parser.add_argument("--suite_output_root", type=str, default="output/dual_loop_rq_suite")
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
    parser.add_argument("--spec_score_threshold", type=int, default=80)
    parser.add_argument("--spec_min_improvement", type=int, default=1)
    parser.add_argument("--spec_precision_floor", type=int, default=85)
    parser.add_argument("--spec_max_rejected_refines", type=int, default=1)
    parser.add_argument(
        "--adaptive_sal_threshold",
        type=float,
        default=0.0,
        help="If > 0, skip SAL refinement when the initial SAS already reaches this threshold.",
    )
    parser.add_argument(
        "--adaptive_ablation_threshold",
        type=float,
        default=85.0,
        help="Threshold used only for the adaptive ablation config inside the suite runner.",
    )
    parser.add_argument(
        "--spec_skip_ambiguity_only",
        dest="spec_skip_ambiguity_only",
        action="store_true",
    )
    parser.add_argument(
        "--no_spec_skip_ambiguity_only",
        dest="spec_skip_ambiguity_only",
        action="store_false",
    )
    parser.set_defaults(spec_skip_ambiguity_only=True)
    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.2)
    parser.add_argument("--codegen_num_candidates", type=int, default=1)
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)
    parser.add_argument("--include_pipeline_ablations", action="store_true")
    parser.add_argument("--include_repair_ablations", action="store_true")
    parser.add_argument("--include_budget_ablations", action="store_true")
    parser.add_argument("--include_adaptive_ablations", action="store_true")
    parser.add_argument(
        "--attribution_mode",
        type=str,
        default="legacy",
        choices=["legacy", "conservative"],
        help="Failure attribution policy. Conservative mode abstains with unknown more often.",
    )
    parser.add_argument(
        "--attribution_spec_margin",
        type=int,
        default=5,
        help="Confidence margin used by conservative attribution around spec_score_threshold.",
    )
    args = parser.parse_args()
    args.stop = args.stop.split(",")
    if args.local_model_path:
        inferred_name = Path(args.local_model_path).name or "LocalModel"
        if args.model is None:
            args.model = inferred_name
        if args.model_repr is None:
            args.model_repr = inferred_name
    elif args.model is None:
        args.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    args.pipeline_mode = "full"
    args.disable_counterexample_repair = False
    args.disable_rewrite_repair = False
    args.run_tag = None
    args.cwd_output_dir = None
    if args.tensor_parallel_size == -1:
        import torch

        args.tensor_parallel_size = torch.cuda.device_count()
    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()
    return args


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_repr = (args.model_repr or args.model or "model").replace("/", "_").replace(" ", "_")
    suite_dir = Path(args.suite_output_root) / f"rq_suite_{model_repr}_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    args.cwd_output_dir = str(suite_dir / "mirrored_outputs")

    run_results = run_rq_suite(
        args,
        include_pipeline_ablations=args.include_pipeline_ablations,
        include_repair_ablations=args.include_repair_ablations,
        include_budget_ablations=args.include_budget_ablations,
        include_adaptive_ablations=args.include_adaptive_ablations,
    )
    rows = build_rq_csv_rows(run_results)

    csv_path = suite_dir / "rq_results.csv"
    manifest_path = suite_dir / "run_manifest.json"
    metadata_path = suite_dir / "suite_metadata.json"
    write_rq_csv(rows, csv_path)
    write_suite_manifest(run_results, manifest_path)

    metadata = {
        "model": args.model,
        "model_repr": args.model_repr,
        "release_version": args.release_version,
        "max_problems": args.max_problems,
        "include_pipeline_ablations": args.include_pipeline_ablations,
        "include_repair_ablations": args.include_repair_ablations,
        "include_budget_ablations": args.include_budget_ablations,
        "include_adaptive_ablations": args.include_adaptive_ablations,
        "codegen_num_candidates": args.codegen_num_candidates,
        "adaptive_sal_threshold": args.adaptive_sal_threshold,
        "adaptive_ablation_threshold": args.adaptive_ablation_threshold,
        "attribution_mode": args.attribution_mode,
        "attribution_spec_margin": args.attribution_spec_margin,
        "suite_dir": str(suite_dir),
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "num_runs": len(rows),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    write_rq_csv(rows, Path.cwd() / "rq_results.csv")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
