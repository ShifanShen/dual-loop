import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser(
        description="Dual-loop experiments on LiveCodeBench with one model for all roles."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model name registered in lcb_runner/lm_styles.py",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Optional local model path for vLLM-backed runs",
    )
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        default="full",
        choices=["baseline", "decomposition", "loop_a", "loop_b", "full"],
        help="Ablation mode for the dual-loop pipeline",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_latest",
        help="LiveCodeBench release version",
    )
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--question_ids", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=10)
    parser.add_argument("--output_root", type=str, default="output/dual_loop")
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
    parser.add_argument("--spec_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--codegen_temperature", type=float, default=0.2)
    parser.add_argument("--repair_temperature", type=float, default=0.1)
    parser.add_argument("--spec_max_tokens", type=int, default=1400)
    parser.add_argument("--judge_max_tokens", type=int, default=1200)
    parser.add_argument("--codegen_max_tokens", type=int, default=2200)

    args = parser.parse_args()
    args.stop = args.stop.split(",")
    if args.tensor_parallel_size == -1:
        import torch

        args.tensor_parallel_size = torch.cuda.device_count()
    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count()
    return args


def main():
    args = get_args()
    from lcb_runner.dual_loop.pipeline import DualLoopPipeline

    summary = DualLoopPipeline(args).run()
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
