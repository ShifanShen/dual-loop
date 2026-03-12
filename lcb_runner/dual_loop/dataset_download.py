import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description="Download and save the LiveCodeBench code generation dataset locally."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="livecodebench/code_generation_lite",
        help="Remote dataset name",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_v6",
        help="Dataset release version",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/livecodebench_code_generation_lite",
        help="Local directory used by save_to_disk",
    )
    return parser.parse_args()


def main():
    args = get_args()
    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset_name,
        split="test",
        version_tag=args.release_version,
        trust_remote_code=True,
    )
    dataset.save_to_disk(str(output_dir))
    print(output_dir)


if __name__ == "__main__":
    main()
