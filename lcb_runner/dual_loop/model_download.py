import argparse
from pathlib import Path


def infer_output_dir(model_id: str, cache_dir: str | None = None) -> str:
    model_name = model_id.rstrip("/").split("/")[-1]
    if cache_dir:
        return str(Path(cache_dir) / model_name)
    return str(Path("models") / model_name)


def download_from_huggingface(
    model_id: str,
    output_dir: str,
    revision: str | None = None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face downloads. "
            "Install it with `uv pip install huggingface-hub`."
        ) from exc

    path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        revision=revision,
    )
    return path


def download_from_modelscope(
    model_id: str,
    output_dir: str,
    revision: str | None = None,
) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required for ModelScope downloads. "
            "Install it with `uv pip install modelscope`."
        ) from exc

    path = snapshot_download(
        model_id=model_id,
        cache_dir=str(Path(output_dir).parent),
        revision=revision,
        local_dir=output_dir,
    )
    return path


def get_args():
    parser = argparse.ArgumentParser(
        description="Download a local model directory for dual-loop experiments."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Model source",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Remote model id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Target local directory; defaults to models/<model_name>",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Base directory used when output_dir is omitted",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision or branch",
    )
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = args.output_dir or infer_output_dir(args.model_id, args.cache_dir)
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        path = download_from_huggingface(args.model_id, output_dir, args.revision)
    else:
        path = download_from_modelscope(args.model_id, output_dir, args.revision)

    print(path)


if __name__ == "__main__":
    main()
