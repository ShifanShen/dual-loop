import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.diagnostics import (
    build_diagnostic_report,
    load_run_artifacts,
    render_diagnostic_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a step-effectiveness diagnostic report from dual-loop summary.json and traces.json."
    )
    parser.add_argument("--summary", type=str, default="summary.json")
    parser.add_argument("--traces", type=str, default="traces.json")
    parser.add_argument(
        "--markdown_out",
        type=str,
        default="diagnostic_report.md",
        help="Path for the human-readable markdown report.",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="diagnostic_report.json",
        help="Path for the machine-readable JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, traces = load_run_artifacts(args.summary, args.traces)
    report = build_diagnostic_report(summary, traces)
    markdown = render_diagnostic_markdown(report)

    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Wrote JSON report to {json_out}")
    print(f"Wrote Markdown report to {markdown_out}")


if __name__ == "__main__":
    main()
