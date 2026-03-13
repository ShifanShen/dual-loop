import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_run_artifacts(summary_path: str | Path, traces_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)
    return summary, traces


def build_diagnostic_report(
    summary: dict[str, Any], traces: list[dict[str, Any]]
) -> dict[str, Any]:
    spec_draft_effects = Counter()
    spec_score_initial_effects = Counter()
    spec_score_final_effects = Counter()
    codegen_effects = Counter()
    repair_effects = Counter()
    repair_strategy_counts = Counter()
    repair_strategy_effects: dict[str, Counter[str]] = defaultdict(Counter)
    spec_refine_effects = Counter()
    repair_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for trace in traces:
        effectiveness = trace.get("effectiveness", {}) or {}
        qid = trace.get("question_id", "")

        if "spec_draft" in effectiveness:
            spec_draft_effects[effectiveness["spec_draft"].get("effect", "unknown")] += 1
        if "spec_score_initial" in effectiveness:
            spec_score_initial_effects[
                effectiveness["spec_score_initial"].get("effect", "unknown")
            ] += 1
        if "spec_score_final" in effectiveness:
            spec_score_final_effects[
                effectiveness["spec_score_final"].get("effect", "unknown")
            ] += 1
        if "codegen" in effectiveness:
            codegen_effects[effectiveness["codegen"].get("effect", "unknown")] += 1

        for step in effectiveness.get("spec_refine_steps", []):
            spec_refine_effects[step.get("effect", "unknown")] += 1

        for step in effectiveness.get("repair_steps", []):
            strategy = step.get("strategy", "unknown")
            effect = step.get("effect", "unknown")
            repair_effects[effect] += 1
            repair_strategy_counts[strategy] += 1
            repair_strategy_effects[strategy][effect] += 1
            if len(repair_examples[effect]) < 5:
                repair_examples[effect].append(
                    {
                        "question_id": qid,
                        "attempt_index": step.get("attempt_index"),
                        "strategy": strategy,
                        "reason": step.get("reason", ""),
                        "matching_lines_before": step.get("matching_lines_before"),
                        "matching_lines_after": step.get("matching_lines_after"),
                        "verifier_signature_before": step.get("verifier_signature_before", ""),
                        "verifier_signature_after": step.get("verifier_signature_after", ""),
                    }
                )

    consistency_checks = {
        "repair_effect_counts_match_summary": dict(repair_effects)
        == dict(summary.get("repair_effect_counts", {})),
        "spec_refine_effect_counts_match_summary": dict(spec_refine_effects)
        == dict(summary.get("spec_refine_effect_counts", {})),
    }

    return {
        "run_overview": {
            "model": summary.get("model"),
            "pipeline_mode": summary.get("pipeline_mode"),
            "num_problems": summary.get("num_problems", len(traces)),
            "pass_at_1": summary.get("pass_at_1"),
            "average_initial_sas": summary.get("average_initial_sas"),
            "average_final_sas": summary.get("average_final_sas"),
            "average_repairs": summary.get("average_repairs"),
            "output_dir": summary.get("output_dir"),
        },
        "stage_effect_counts": {
            "spec_draft": dict(spec_draft_effects),
            "spec_score_initial": dict(spec_score_initial_effects),
            "spec_score_final": dict(spec_score_final_effects),
            "spec_refine": dict(spec_refine_effects),
            "codegen": dict(codegen_effects),
            "repair": dict(repair_effects),
        },
        "repair_strategy_counts": dict(repair_strategy_counts),
        "repair_strategy_effect_counts": {
            strategy: dict(counter) for strategy, counter in repair_strategy_effects.items()
        },
        "failure_attribution_counts": dict(summary.get("failure_attribution_counts", {})),
        "verifier_error_counts": dict(summary.get("verifier_error_counts", {})),
        "examples": {effect: items for effect, items in repair_examples.items()},
        "consistency_checks": consistency_checks,
    }


def render_diagnostic_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    overview = report["run_overview"]
    lines.append("# Dual-Loop Diagnostic Report")
    lines.append("")
    lines.append("## Run Overview")
    lines.append(f"- Model: `{overview.get('model')}`")
    lines.append(f"- Pipeline mode: `{overview.get('pipeline_mode')}`")
    lines.append(f"- Problems: `{overview.get('num_problems')}`")
    lines.append(f"- pass@1: `{overview.get('pass_at_1')}`")
    lines.append(f"- Average initial SAS: `{overview.get('average_initial_sas')}`")
    lines.append(f"- Average final SAS: `{overview.get('average_final_sas')}`")
    lines.append(f"- Average repairs: `{overview.get('average_repairs')}`")
    lines.append(f"- Output dir: `{overview.get('output_dir')}`")
    lines.append("")

    lines.append("## Stage Effect Counts")
    for stage, counts in report["stage_effect_counts"].items():
        lines.append(f"- `{stage}`: `{json.dumps(counts, ensure_ascii=True)}`")
    lines.append("")

    lines.append("## Repair Strategy Breakdown")
    for strategy, count in report["repair_strategy_counts"].items():
        effect_counts = report["repair_strategy_effect_counts"].get(strategy, {})
        lines.append(
            f"- `{strategy}`: attempts=`{count}`, effects=`{json.dumps(effect_counts, ensure_ascii=True)}`"
        )
    lines.append("")

    lines.append("## Outcome Counts")
    lines.append(
        f"- failure_attribution_counts: `{json.dumps(report['failure_attribution_counts'], ensure_ascii=True)}`"
    )
    lines.append(
        f"- verifier_error_counts: `{json.dumps(report['verifier_error_counts'], ensure_ascii=True)}`"
    )
    lines.append("")

    lines.append("## Sample Repair Examples")
    for effect in ("no_effect", "improved", "solved", "changed_but_not_improved"):
        examples = report["examples"].get(effect, [])
        if not examples:
            continue
        lines.append(f"### `{effect}`")
        for item in examples:
            lines.append(
                "- "
                f"qid=`{item['question_id']}`, attempt=`{item['attempt_index']}`, "
                f"strategy=`{item['strategy']}`, reason=`{item['reason']}`, "
                f"match=`{item['matching_lines_before']}` -> `{item['matching_lines_after']}`"
            )
        lines.append("")

    lines.append("## Consistency Checks")
    for key, value in report["consistency_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"
