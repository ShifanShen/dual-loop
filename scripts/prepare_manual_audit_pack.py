import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcb_runner.dual_loop.diagnostics import load_run_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a human-audit pack for validating semantic attribution."
    )
    parser.add_argument("--suite_dir", type=str, required=True)
    parser.add_argument("--per_label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _resolve_suite_dir(raw_suite_dir: str) -> Path:
    suite_dir = Path(raw_suite_dir)
    candidates = []

    if suite_dir.is_absolute():
        candidates.append(suite_dir)
    else:
        candidates.extend(
            [
                ROOT / suite_dir,
                ROOT / "output" / "dual_loop_rq_suite" / suite_dir,
                ROOT / "assets" / "output" / "dual_loop_rq_suite" / suite_dir,
            ]
        )

    # If the caller already passed output/... or assets/output/... relative to repo root,
    # keep those exact paths near the front of the search order.
    if not suite_dir.is_absolute():
        normalized = str(suite_dir).replace("\\", "/")
        if normalized.startswith("output/") or normalized.startswith("assets/output/"):
            candidates.insert(0, ROOT / suite_dir)

    for candidate in candidates:
        if (candidate / "run_manifest.json").exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find run_manifest.json for the requested suite. "
        f"Tried:\n{searched}"
    )


def _load_manifest(suite_dir: Path) -> list[dict]:
    with open(suite_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_run(run_entry: dict) -> tuple[dict, list[dict]]:
    return load_run_artifacts(run_entry["summary_path"], run_entry["traces_path"])


def _latest_feedback_signature(trace: dict) -> str:
    feedbacks = trace.get("verifier_feedbacks", [])
    if not feedbacks:
        return ""
    last = feedbacks[-1]
    return last.get("error_type", "") or last.get("message", "")


def _build_record(trace: dict, *, source_run: str, source_suite: str) -> dict:
    initial_score = trace.get("initial_spec_score") or {}
    final_score = trace.get("final_spec_score") or {}
    return {
        "audit_id": "",
        "source_suite": source_suite,
        "source_run": source_run,
        "question_id": trace.get("question_id", ""),
        "question_title": trace.get("question_title", ""),
        "pipeline_mode": trace.get("pipeline_mode", ""),
        "failure_attribution": trace.get("failure_attribution", ""),
        "passed": trace.get("passed", False),
        "initial_sas": initial_score.get("overall", ""),
        "final_sas": final_score.get("overall", ""),
        "spec_issue_types": "; ".join(trace.get("spec_issue_types", [])),
        "property_clause_types": "; ".join(
            clause.get("property_type", "") for clause in trace.get("property_clauses", [])
        ),
        "latest_verifier_signature": _latest_feedback_signature(trace),
        "raw_problem": trace.get("raw_problem", ""),
        "spec_initial": json.dumps(trace.get("spec_initial", {}), ensure_ascii=True),
        "spec_final": json.dumps(trace.get("spec_final", {}), ensure_ascii=True),
        "final_code": trace.get("final_code", ""),
        "reviewer_primary_failure_source": "",
        "reviewer_attribution_label": "",
        "reviewer_agrees_with_system": "",
        "reviewer_spec_captures_key_requirements": "",
        "reviewer_notes": "",
    }


def main() -> None:
    args = parse_args()
    suite_dir = _resolve_suite_dir(args.suite_dir)
    output_dir = Path(args.output_dir) if args.output_dir else suite_dir / "manual_audit_pack"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    manifest = _load_manifest(suite_dir)
    run_entries = {entry["run_name"]: entry for entry in manifest}

    sampled_records: list[dict] = []
    source_map = {
        "solved": ["full_dual_loop", "loop_a_only"],
        "spec_induced": ["full_dual_loop", "loop_b_only", "loop_a_only"],
        "implementation_induced": ["loop_a_only", "full_dual_loop", "loop_b_only"],
    }

    for label, preferred_runs in source_map.items():
        pool: list[tuple[str, dict]] = []
        for run_name in preferred_runs:
            if run_name not in run_entries:
                continue
            _, traces = _load_run(run_entries[run_name])
            for trace in traces:
                if trace.get("failure_attribution") == label:
                    pool.append((run_name, trace))
        rng.shuffle(pool)
        for idx, (run_name, trace) in enumerate(pool[: args.per_label], start=1):
            record = _build_record(trace, source_run=run_name, source_suite=suite_dir.name)
            record["audit_id"] = f"{label}-{idx:02d}"
            sampled_records.append(record)

    sheet_path = output_dir / "audit_sheet.csv"
    with open(sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sampled_records[0].keys()) if sampled_records else [])
        if sampled_records:
            writer.writeheader()
            for record in sampled_records:
                writer.writerow(record)

    with open(output_dir / "audit_samples.jsonl", "w", encoding="utf-8") as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    readme = """# Manual Audit Protocol

Goal:
- validate whether the system's attribution labels are reasonable
- validate whether accepted specs capture the key problem requirements

Recommended sample size:
- solved: 10-15
- spec-induced: 10-15
- implementation-induced: 10-15 when available

How to audit each sample:
1. Read the raw problem statement.
2. Read the final accepted spec.
3. Decide whether the spec captures the key requirements and edge cases.
4. Read the final code and the latest verifier signature.
5. Decide the most plausible primary failure source:
   - semantic/spec side
   - implementation/code side
   - unclear
6. Compare your judgment with the system's label.

Suggested annotation fields:
- reviewer_primary_failure_source
- reviewer_attribution_label
- reviewer_agrees_with_system
- reviewer_spec_captures_key_requirements
- reviewer_notes
"""
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    summary = {
        "suite_dir": str(suite_dir),
        "output_dir": str(output_dir),
        "num_samples": len(sampled_records),
        "counts_by_label": {
            label: sum(1 for record in sampled_records if record["failure_attribution"] == label)
            for label in source_map
        },
        "audit_sheet": str(sheet_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
