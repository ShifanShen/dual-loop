import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and save a local copy of the LiveCodeBench code-generation dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="livecodebench/code_generation_lite",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_v6",
        help="Dataset version_tag passed to load_dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to cache locally.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target directory for save_to_disk output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --force to overwrite it."
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        version_tag=args.release_version,
        trust_remote_code=args.trust_remote_code,
    )
    dataset.save_to_disk(str(output_dir))

    metadata = {
        "dataset_name": args.dataset_name,
        "release_version": args.release_version,
        "split": args.split,
        "num_rows": len(dataset),
        "output_dir": str(output_dir.resolve()),
    }
    with open(output_dir / "cache_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
