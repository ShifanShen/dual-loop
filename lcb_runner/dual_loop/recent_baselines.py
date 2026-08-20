import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from lcb_runner.dual_loop.pipeline import DualLoopPipeline
from lcb_runner.dual_loop.prompts import build_plan_draft_prompt
from lcb_runner.dual_loop.spec import VerifierFeedback


METHODS = ("specfix_bm", "lpw_adapted")
OBSERVATION_SENTINEL = "__SPECFIX_OBSERVATION_SENTINEL__"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def parse_probe_inputs(
    raw_output: str,
    *,
    functional: bool,
    max_count: int,
) -> list[str]:
    payload = extract_json_object(raw_output)
    raw_tests = payload.get("tests", [])
    if not isinstance(raw_tests, list):
        return []

    parsed: list[str] = []
    seen: set[str] = set()
    for raw_test in raw_tests:
        value = raw_test
        if functional:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            if not isinstance(value, list):
                continue
            normalized = json.dumps(value, ensure_ascii=True)
        else:
            if not isinstance(value, str):
                continue
            normalized = value.strip("\n")
            if not normalized:
                continue

        if len(normalized) > 5000:
            continue

        if normalized in seen:
            continue
        seen.add(normalized)
        parsed.append(normalized)
        if len(parsed) >= max_count:
            break
    return parsed


def normalized_behavior(value: Any) -> str:
    if value is None:
        return "<NO_OUTPUT>"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return " ".join(str(value).strip().split())


def behavior_clusters(candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        signature = tuple(normalized_behavior(item) for item in record.get("observations", []))
        grouped[signature].append(record)

    clusters: list[dict[str, Any]] = []
    total = max(1, len(candidate_records))
    for cluster_index, (signature, members) in enumerate(
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
        start=1,
    ):
        public_ratios = [
            feedback_pass_ratio(member["feedback"])
            for member in members
            if isinstance(member.get("feedback"), VerifierFeedback)
        ]
        clusters.append(
            {
                "cluster_index": cluster_index,
                "size": len(members),
                "probability": len(members) / total,
                "signature": list(signature),
                "candidate_indices": [int(member["candidate_index"]) for member in members],
                "mean_public_pass_ratio": (
                    sum(public_ratios) / len(public_ratios) if public_ratios else 0.0
                ),
                "public_pass_count": sum(
                    1 for member in members if bool(member["feedback"].passed)
                ),
            }
        )
    return clusters


def cluster_entropy(clusters: list[dict[str, Any]]) -> float:
    if len(clusters) <= 1:
        return 0.0
    entropy = -sum(
        float(cluster["probability"]) * math.log(float(cluster["probability"]))
        for cluster in clusters
        if float(cluster["probability"]) > 0
    )
    return entropy / math.log(len(clusters))


def feedback_pass_ratio(feedback: VerifierFeedback) -> float:
    metadata = feedback.raw_metadata or {}
    if feedback.passed:
        return 1.0
    try:
        return float(metadata.get("passed_test_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_candidate_record(
    pipeline: DualLoopPipeline,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot select from an empty candidate list")

    clusters = behavior_clusters(records)
    cluster_size_by_candidate: dict[int, int] = {}
    for cluster in clusters:
        for candidate_index in cluster["candidate_indices"]:
            cluster_size_by_candidate[int(candidate_index)] = int(cluster["size"])

    return min(
        records,
        key=lambda record: (
            pipeline._candidate_feedback_rank(record["feedback"]),
            -cluster_size_by_candidate.get(int(record["candidate_index"]), 0),
            int(record["candidate_index"]),
        ),
    )


def parse_plan_evaluation(raw_output: str) -> dict[str, Any]:
    payload = extract_json_object(raw_output)
    accepted = payload.get("accepted", False)
    if isinstance(accepted, str):
        accepted = accepted.strip().lower() in {"true", "yes", "accepted", "pass"}
    notes = payload.get("verification_notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if not isinstance(notes, list):
        notes = []
    return {
        "parse_ok": bool(payload),
        "accepted": bool(accepted),
        "verification_notes": [str(item).strip() for item in notes if str(item).strip()],
        "revised_plan": str(payload.get("revised_plan", "") or "").strip(),
    }


def parse_plan_check(raw_output: str) -> dict[str, Any]:
    payload = extract_json_object(raw_output)
    confirmed = payload.get("confirmed", False)
    if isinstance(confirmed, str):
        confirmed = confirmed.strip().lower() in {"true", "yes", "confirmed", "pass"}
    return {
        "parse_ok": bool(payload),
        "confirmed": bool(confirmed),
        "reason": str(payload.get("reason", "") or "").strip(),
    }


def _public_examples(problem, limit: int) -> list[dict[str, str]]:
    return [
        {"input": test.input, "expected_output": test.output}
        for test in problem.public_test_cases[: max(0, limit)]
    ]


def _build_specfix_probe_input_prompt(problem, max_tests: int) -> str:
    func_name = str(problem.metadata.get("func_name", "") or "")
    if func_name:
        format_instruction = (
            'Each item in "tests" must be a JSON array containing the positional '
            "arguments for one call."
        )
    else:
        format_instruction = (
            'Each item in "tests" must be one complete standard-input string, including '
            "newlines where required."
        )
    return f"""Generate diverse valid test inputs for the programming problem below.
Return JSON only using this schema: {{"tests": [...]}}.

Requirements:
- Generate at most {max_tests} tests.
- Cover normal behavior, boundary values, and plausible interpretation differences.
- Inputs must satisfy every explicit constraint in the problem.
- Do not include expected outputs.
- {format_instruction}

Problem:
{problem.question_content}
"""


def _build_specfix_codegen_prompt(problem, requirement: str) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are an expert Python programmer.
Write one complete Python solution for the requirement below.
Return exactly one Python code block and no explanation.

Requirement:
{requirement}
{starter}
Requirements:
- Follow the exact input/output protocol.
- Do not hard-code sample inputs.
- Do not print extra text.
"""


def _compact_feedback(feedback: VerifierFeedback) -> dict[str, Any]:
    return {
        "passed": bool(feedback.passed),
        "error_type": feedback.error_type,
        "input": feedback.input,
        "output": feedback.output,
        "expected": feedback.expected,
        "passed_test_ratio": feedback_pass_ratio(feedback),
    }


def _build_specfix_repair_prompt(
    problem,
    clusters: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> str:
    by_index = {int(record["candidate_index"]): record for record in candidate_records}
    cluster_summaries: list[dict[str, Any]] = []
    for cluster in clusters[:4]:
        representative_index = int(cluster["candidate_indices"][0])
        representative = by_index[representative_index]
        cluster_summaries.append(
            {
                "size": cluster["size"],
                "probe_outputs": cluster["signature"],
                "public_feedback": _compact_feedback(representative["feedback"]),
                "representative_program": str(representative.get("code", ""))[:6000],
            }
        )
    evidence = json.dumps(cluster_summaries, ensure_ascii=True, indent=2)
    return f"""Repair a programming requirement that produced behaviorally inconsistent programs.
Return JSON only using this schema: {{"repaired_requirement": "..."}}.

Original requirement:
{problem.question_content}

Behavior clusters produced from independently generated programs:
{evidence}

Repair rules:
- Preserve all supported requirements and the exact input/output protocol.
- Clarify only wording, constraints, edge cases, or output rules supported by the original requirement and public examples.
- Use the public feedback to distinguish intended behavior from implementation mistakes.
- Do not prescribe a particular algorithm unless the original requirement does so.
- Do not mention the candidate programs, clusters, or this repair process.
- Return the complete repaired requirement, not a patch.
"""


def _build_lpw_plan_evaluation_prompt(
    problem,
    plan: str,
    examples: list[dict[str, str]],
    *,
    prior_critique: str = "",
) -> str:
    critique = ""
    if prior_critique:
        critique = f"""
An independent checker rejected the previous evaluation for this reason:
{prior_critique}

Reconsider that objection explicitly. If it is valid, reject and revise the plan.
"""
    return f"""Evaluate a proposed solution plan by simulating it on the visible examples.
Return JSON only with this schema:
{{
  "accepted": true,
  "verification_notes": ["one concise note per checked example"],
  "revised_plan": ""
}}

Set accepted to false when the plan would produce a wrong output, misses a required case, or violates the stated protocol. When rejected, put a complete corrected plan in revised_plan. Do not write code.

Problem:
{problem.question_content}

Proposed plan:
{plan}

Visible examples:
{json.dumps(examples, ensure_ascii=True, indent=2)}
{critique}
"""


def _build_lpw_plan_check_prompt(
    problem,
    plan: str,
    evaluation: dict[str, Any],
    examples: list[dict[str, str]],
) -> str:
    return f"""Independently check whether the plan-verification analysis is correct.
Return JSON only using this schema: {{"confirmed": true, "reason": "..."}}.

Problem:
{problem.question_content}

Plan:
{plan}

Visible examples:
{json.dumps(examples, ensure_ascii=True, indent=2)}

Claimed verification:
{json.dumps(evaluation, ensure_ascii=True, indent=2)}

Confirm only if the plan is consistent with every visible example and the original problem.
"""


def _build_lpw_codegen_prompt(
    problem,
    plan: str,
    verification_notes: list[str],
    *,
    plan_accepted: bool,
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    plan_status = (
        "accepted after plan verification"
        if plan_accepted
        else "last revised plan after the verification budget was exhausted"
    )
    return f"""You are an expert Python programmer.
Implement the solution plan as one complete Python program.
Return exactly one Python code block and no explanation.

Problem:
{problem.question_content}

Solution plan ({plan_status}):
{plan}

Plan-verification notes:
{json.dumps(verification_notes, ensure_ascii=True, indent=2)}
{starter}
Requirements:
- Follow the exact input/output protocol from the problem.
- Preserve this solution plan unless an implementation detail must be filled in.
- Do not print extra text.
"""


def _build_lpw_error_analysis_prompt(
    problem,
    plan: str,
    verification_notes: list[str],
    code: str,
    feedback: VerifierFeedback,
    *,
    plan_accepted: bool,
) -> str:
    plan_status = "accepted" if plan_accepted else "last revised"
    return f"""Analyze why the current Python program violates the solution plan.
Return concise plain text. Do not return code.

Problem:
{problem.question_content}

{plan_status.capitalize()} plan:
{plan}

Plan-verification notes:
{json.dumps(verification_notes, ensure_ascii=True, indent=2)}

Current program:
```python
{code}
```

Visible execution feedback:
{feedback.to_json()}

Identify the first incorrect implementation decision and state the exact correction needed.
"""


def _build_lpw_repair_prompt(
    problem,
    plan: str,
    verification_notes: list[str],
    code: str,
    feedback: VerifierFeedback,
    error_analysis: str,
    *,
    require_change: bool,
    plan_accepted: bool,
) -> str:
    plan_status = "accepted" if plan_accepted else "last revised"
    change_rule = ""
    if require_change:
        change_rule = "\n- Do not return the same program; change the failing logic."
    return f"""Repair the Python program using the solution plan and execution analysis.
Return exactly one complete Python program in one fenced code block.

Problem:
{problem.question_content}

{plan_status.capitalize()} plan:
{plan}

Plan-verification notes:
{json.dumps(verification_notes, ensure_ascii=True, indent=2)}

Current program:
```python
{code}
```

Visible execution feedback:
{feedback.to_json()}

Error analysis:
{error_analysis}

Requirements:
- Fix the reported implementation error while preserving the solution plan.
- Follow the exact input/output protocol.
- Do not print extra text.{change_rule}
"""


def _append_usage(target: list[dict[str, Any]], usage: dict[str, Any] | None) -> None:
    if usage:
        target.append(dict(usage))


def _usage_summary(usages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "llm_calls": len(usages),
        "prompt_chars": sum(int(item.get("prompt_chars", 0) or 0) for item in usages),
        "completion_chars": sum(
            int(item.get("completion_chars", 0) or 0) for item in usages
        ),
    }


def _generate_code_record(
    pipeline: DualLoopPipeline,
    problem,
    *,
    prompt: str,
    role: str,
    candidate_index: int,
    temperature: float,
    usages: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_output, usage = pipeline.llm.generate(
        prompt,
        role=role,
        temperature=temperature,
        max_tokens=pipeline.args.codegen_max_tokens,
    )
    _append_usage(usages, usage)
    code, extra_outputs, extra_usages = pipeline._extract_valid_code(raw_output)
    for extra_usage in extra_usages:
        _append_usage(usages, extra_usage)
    if code:
        feedback = pipeline._verify(problem, code)
    else:
        feedback = pipeline._invalid_codegen_feedback(
            "Candidate could not be parsed into valid Python code."
        )
    return {
        "candidate_index": candidate_index,
        "code": code,
        "raw_output": raw_output,
        "extra_outputs": extra_outputs,
        "feedback": feedback,
        "observations": [],
    }


def observe_program_on_inputs(
    problem,
    code: str,
    probe_inputs: list[str],
    *,
    timeout: int,
) -> list[str]:
    if not code:
        return ["<INVALID_PROGRAM>" for _ in probe_inputs]

    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    func_name = problem.metadata.get("func_name", None)
    expected = (
        json.dumps(OBSERVATION_SENTINEL)
        if func_name
        else OBSERVATION_SENTINEL
    )
    observations: list[str] = []
    for probe_input in probe_inputs:
        sample = {
            "input_output": json.dumps(
                {
                    "inputs": [probe_input],
                    "outputs": [expected],
                    "fn_name": func_name,
                }
            )
        }
        results, metadata = check_correctness(
            sample,
            code,
            timeout=timeout,
            debug=False,
        )
        if isinstance(metadata, dict) and "output" in metadata:
            observations.append(normalized_behavior(metadata.get("output")))
        elif results and all(result is True for result in results):
            observations.append(OBSERVATION_SENTINEL)
        else:
            error_code = (
                metadata.get("error_code", "unknown")
                if isinstance(metadata, dict)
                else "unknown"
            )
            error_message = (
                metadata.get("error_message", "")
                if isinstance(metadata, dict)
                else ""
            )
            observations.append(f"<ERROR:{error_code}:{error_message}>")
    return observations


def _serializable_candidate(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    feedback = payload.get("feedback")
    if isinstance(feedback, VerifierFeedback):
        payload["feedback"] = asdict(feedback)
    return payload


def run_specfix_bm(
    pipeline: DualLoopPipeline,
    problem,
    *,
    probe_candidate_count: int,
    probe_test_count: int,
    max_program_candidates: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    usages: list[dict[str, Any]] = []
    functional = bool(problem.metadata.get("func_name", None))

    probe_prompt = _build_specfix_probe_input_prompt(problem, probe_test_count)
    raw_probe_output, probe_usage = pipeline.llm.generate(
        probe_prompt,
        role="specfix_probe_generation",
        temperature=pipeline.args.spec_temperature,
        max_tokens=pipeline.args.spec_max_tokens,
    )
    _append_usage(usages, probe_usage)
    probe_inputs = parse_probe_inputs(
        raw_probe_output,
        functional=functional,
        max_count=probe_test_count,
    )
    probe_source = "generated"
    public_anchor = problem.public_test_cases[0].input if problem.public_test_cases else ""
    if public_anchor and public_anchor not in probe_inputs:
        probe_source = "generated_plus_public_anchor"
        if len(probe_inputs) >= probe_test_count:
            probe_inputs[-1] = public_anchor
        else:
            probe_inputs.append(public_anchor)
    if len(probe_inputs) < probe_test_count:
        probe_source = "generated_plus_public_fallback"
        for test in problem.public_test_cases:
            if test.input not in probe_inputs:
                probe_inputs.append(test.input)
            if len(probe_inputs) >= probe_test_count:
                break

    initial_count = min(probe_candidate_count, max_program_candidates)
    original_records: list[dict[str, Any]] = []
    original_prompt = _build_specfix_codegen_prompt(problem, problem.question_content)
    for candidate_index in range(1, initial_count + 1):
        record = _generate_code_record(
            pipeline,
            problem,
            prompt=original_prompt,
            role="specfix_probe_codegen",
            candidate_index=candidate_index,
            temperature=pipeline.args.codegen_temperature,
            usages=usages,
        )
        record["observations"] = observe_program_on_inputs(
            problem,
            record["code"],
            probe_inputs,
            timeout=pipeline.args.timeout,
        )
        original_records.append(record)

    original_clusters = behavior_clusters(original_records)
    entropy = cluster_entropy(original_clusters)
    ambiguity_detected = bool(
        len(original_clusters) > 1
        or not any(record["feedback"].passed for record in original_records)
    )

    repaired_requirement = ""
    raw_requirement_repair = ""
    final_records: list[dict[str, Any]] = []
    if ambiguity_detected and len(original_records) < max_program_candidates:
        repair_prompt = _build_specfix_repair_prompt(
            problem,
            original_clusters,
            original_records,
        )
        raw_requirement_repair, repair_usage = pipeline.llm.generate(
            repair_prompt,
            role="specfix_requirement_repair",
            temperature=pipeline.args.spec_temperature,
            max_tokens=pipeline.args.spec_max_tokens,
        )
        _append_usage(usages, repair_usage)
        repair_payload = extract_json_object(raw_requirement_repair)
        repaired_requirement = str(
            repair_payload.get("repaired_requirement", "") or ""
        ).strip()

        if repaired_requirement:
            final_prompt = _build_specfix_codegen_prompt(problem, repaired_requirement)
            remaining = max_program_candidates - len(original_records)
            for offset in range(remaining):
                candidate_index = len(original_records) + offset + 1
                record = _generate_code_record(
                    pipeline,
                    problem,
                    prompt=final_prompt,
                    role="specfix_repaired_codegen",
                    candidate_index=candidate_index,
                    temperature=pipeline.args.codegen_temperature,
                    usages=usages,
                )
                record["observations"] = observe_program_on_inputs(
                    problem,
                    record["code"],
                    probe_inputs,
                    timeout=pipeline.args.timeout,
                )
                final_records.append(record)

    selection_pool = final_records or original_records
    selected = select_candidate_record(pipeline, selection_pool)
    final_clusters = behavior_clusters(final_records) if final_records else original_clusters
    usage_summary = _usage_summary(usages)

    return {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "method": "specfix_bm",
        "protocol_variant": "budget-matched adaptation of SpecFix",
        "raw_problem": problem.question_content,
        "probe_inputs": probe_inputs,
        "probe_input_source": probe_source,
        "raw_probe_input_output": raw_probe_output,
        "ambiguity_detected": ambiguity_detected,
        "behavior_entropy": round(entropy, 6),
        "original_clusters": original_clusters,
        "final_clusters": final_clusters,
        "repaired_requirement": repaired_requirement,
        "raw_requirement_repair": raw_requirement_repair,
        "original_candidates": [_serializable_candidate(item) for item in original_records],
        "repaired_candidates": [_serializable_candidate(item) for item in final_records],
        "program_candidate_count": len(original_records) + len(final_records),
        "selected_candidate_index": int(selected["candidate_index"]),
        "code_initial": str(original_records[0]["code"]) if original_records else "",
        "final_code": str(selected["code"]),
        "feedback_passed": bool(selected["feedback"].passed),
        "final_feedback": asdict(selected["feedback"]),
        "feedback_verifier_executions": len(original_records) + len(final_records),
        "probe_executions": (len(original_records) + len(final_records)) * len(probe_inputs),
        "usages": usages,
        **usage_summary,
        "elapsed_seconds": time.perf_counter() - started_at,
        "final_passed": False,
    }


def _draft_and_verify_lpw_plan(
    pipeline: DualLoopPipeline,
    problem,
    *,
    max_iters: int,
    example_limit: int,
    usages: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_plan, plan_usage = pipeline.llm.generate(
        build_plan_draft_prompt(problem),
        role="lpw_plan_draft",
        temperature=pipeline.args.spec_temperature,
        max_tokens=pipeline.args.spec_max_tokens,
    )
    _append_usage(usages, plan_usage)
    initial_plan = raw_plan.strip()
    current_plan = initial_plan
    examples = _public_examples(problem, example_limit)
    attempts: list[dict[str, Any]] = []
    final_notes: list[str] = []
    accepted = False
    prior_critique = ""

    for iteration in range(max_iters):
        evaluation_prompt = _build_lpw_plan_evaluation_prompt(
            problem,
            current_plan,
            examples,
            prior_critique=prior_critique,
        )
        raw_evaluation, evaluation_usage = pipeline.llm.generate(
            evaluation_prompt,
            role="lpw_plan_evaluation",
            temperature=pipeline.args.judge_temperature,
            max_tokens=pipeline.args.judge_max_tokens,
        )
        _append_usage(usages, evaluation_usage)
        evaluation = parse_plan_evaluation(raw_evaluation)
        attempt: dict[str, Any] = {
            "iteration": iteration + 1,
            "plan_before": current_plan,
            "raw_evaluation": raw_evaluation,
            "evaluation": evaluation,
            "check": None,
        }
        final_notes = list(evaluation["verification_notes"])

        if evaluation["accepted"]:
            check_prompt = _build_lpw_plan_check_prompt(
                problem,
                current_plan,
                evaluation,
                examples,
            )
            raw_check, check_usage = pipeline.llm.generate(
                check_prompt,
                role="lpw_plan_check",
                temperature=pipeline.args.judge_temperature,
                max_tokens=pipeline.args.judge_max_tokens,
            )
            _append_usage(usages, check_usage)
            check = parse_plan_check(raw_check)
            attempt["check"] = {"raw_output": raw_check, **check}
            attempts.append(attempt)
            if check["confirmed"]:
                accepted = True
                break
            prior_critique = str(check.get("reason", "") or "").strip()
            continue

        revised_plan = str(evaluation.get("revised_plan", "") or "").strip()
        attempts.append(attempt)
        if revised_plan:
            current_plan = revised_plan
        prior_critique = ""

    return {
        "initial_plan": initial_plan,
        "final_plan": current_plan,
        "plan_accepted": accepted,
        "plan_status": "accepted" if accepted else "verification_budget_exhausted",
        "verification_notes": final_notes,
        "plan_attempts": attempts,
        "public_examples_used": examples,
    }


def run_lpw_adapted(
    pipeline: DualLoopPipeline,
    problem,
    *,
    plan_max_iters: int,
    plan_example_limit: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    usages: list[dict[str, Any]] = []
    plan_result = _draft_and_verify_lpw_plan(
        pipeline,
        problem,
        max_iters=plan_max_iters,
        example_limit=plan_example_limit,
        usages=usages,
    )

    codegen_prompt = _build_lpw_codegen_prompt(
        problem,
        plan_result["final_plan"],
        plan_result["verification_notes"],
        plan_accepted=bool(plan_result["plan_accepted"]),
    )
    codegen_result = pipeline._select_best_codegen_candidate(
        problem,
        prompt=codegen_prompt,
        spec=None,
    )
    _append_usage(usages, codegen_result.get("usage"))
    for usage in codegen_result.get("extra_usages", []):
        _append_usage(usages, usage)

    current_code = str(codegen_result["code"])
    initial_code = current_code
    current_feedback: VerifierFeedback = codegen_result["feedback"]
    repair_steps: list[dict[str, Any]] = []
    repair_program_count = 0
    stagnant_attempts = 0

    for iteration in range(int(pipeline.args.repair_max_iters or 0)):
        if current_feedback.passed:
            break

        analysis_prompt = _build_lpw_error_analysis_prompt(
            problem,
            plan_result["final_plan"],
            plan_result["verification_notes"],
            current_code,
            current_feedback,
            plan_accepted=bool(plan_result["plan_accepted"]),
        )
        error_analysis, analysis_usage = pipeline.llm.generate(
            analysis_prompt,
            role="lpw_error_analysis",
            temperature=pipeline.args.judge_temperature,
            max_tokens=pipeline.args.judge_max_tokens,
        )
        _append_usage(usages, analysis_usage)

        records: list[dict[str, Any]] = []
        for candidate_offset in range(int(pipeline.args.repair_num_candidates or 1)):
            prompt = _build_lpw_repair_prompt(
                problem,
                plan_result["final_plan"],
                plan_result["verification_notes"],
                current_code,
                current_feedback,
                error_analysis,
                require_change=stagnant_attempts > 0,
                plan_accepted=bool(plan_result["plan_accepted"]),
            )
            candidate_index = candidate_offset + 1
            record = _generate_code_record(
                pipeline,
                problem,
                prompt=prompt,
                role="lpw_repair",
                candidate_index=candidate_index,
                temperature=min(
                    0.8,
                    float(pipeline.args.repair_temperature or 0.0)
                    + 0.05 * candidate_offset,
                ),
                usages=usages,
            )
            repair_program_count += 1
            records.append(record)
            if record["feedback"].passed:
                break

        selected = min(
            records,
            key=lambda record: pipeline._candidate_feedback_rank(record["feedback"]),
        )
        next_code = str(selected["code"])
        next_feedback: VerifierFeedback = selected["feedback"]
        unchanged = not next_code or next_code.strip() == current_code.strip()
        repair_steps.append(
            {
                "iteration": iteration + 1,
                "error_analysis": error_analysis,
                "candidate_count": len(records),
                "selected_candidate_index": int(selected["candidate_index"]),
                "selected_passed": bool(next_feedback.passed),
                "selected_error_type": next_feedback.error_type,
                "candidates": [_serializable_candidate(item) for item in records],
            }
        )
        if unchanged:
            stagnant_attempts += 1
        else:
            current_code = next_code
            current_feedback = next_feedback
            stagnant_attempts = 0

    usage_summary = _usage_summary(usages)
    codegen_program_count = int(codegen_result.get("candidate_count", 0) or 0)
    return {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "method": "lpw_adapted",
        "protocol_variant": "LPW adapted to shared held-out LiveCodeBench protocol",
        "raw_problem": problem.question_content,
        **plan_result,
        "code_initial": initial_code,
        "final_code": current_code,
        "codegen_candidate_count": codegen_program_count,
        "codegen_selected_candidate_index": int(
            codegen_result.get("selected_candidate_index", 0) or 0
        ),
        "codegen_candidate_feedbacks": codegen_result.get("candidate_feedbacks", []),
        "repair_steps": repair_steps,
        "repair_iterations": len(repair_steps),
        "program_candidate_count": codegen_program_count + repair_program_count,
        "feedback_passed": bool(current_feedback.passed),
        "final_feedback": asdict(current_feedback),
        "feedback_verifier_executions": codegen_program_count + repair_program_count,
        "probe_executions": 0,
        "usages": usages,
        **usage_summary,
        "elapsed_seconds": time.perf_counter() - started_at,
        "final_passed": False,
    }


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return traces
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                trace = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            question_id = str(trace.get("question_id", ""))
            if question_id:
                traces[question_id] = trace
    return traces


def append_checkpoint(path: Path, trace: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, path)


def apply_final_evaluation(
    pipeline: DualLoopPipeline,
    benchmark: list[Any],
    traces: list[dict[str, Any]],
) -> dict[str, float | int]:
    generations = [[str(trace.get("final_code", ""))] for trace in traces]
    metrics = pipeline._compute_metrics(benchmark, generations)
    final_results = metrics[1] if len(metrics) > 1 else {}
    final_metadata = metrics[2] if len(metrics) > 2 else []

    for index, trace in enumerate(traces):
        per_generation = final_results.get(index, [])
        first_generation = per_generation[0] if per_generation else []
        if not isinstance(first_generation, list):
            first_generation = [first_generation]
        private_passed = bool(first_generation) and all(
            result is True for result in first_generation
        )
        metadata: Any = {}
        if index < len(final_metadata) and final_metadata[index]:
            metadata = final_metadata[index][0]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
        feedback_passed = bool(trace.get("feedback_passed", False))
        combined_passed = feedback_passed and private_passed
        trace["private_passed"] = private_passed
        trace["final_passed"] = combined_passed
        trace["final_evaluation"] = {
            "scope": str(pipeline.args.final_test_scope),
            "feedback_passed": feedback_passed,
            "private_passed": private_passed,
            "passed": combined_passed,
            "error_code": metadata.get("error_code") if isinstance(metadata, dict) else None,
            "error_message": (
                metadata.get("error_message", "")
                if isinstance(metadata, dict)
                else ""
            ),
        }
    private_pass_count = sum(
        1 for trace in traces if bool(trace.get("private_passed", False))
    )
    combined_pass_count = sum(
        1 for trace in traces if bool(trace.get("final_passed", False))
    )
    count = len(traces)
    return {
        "private_pass_count": private_pass_count,
        "private_pass_rate": private_pass_count / count if count else 0.0,
        "combined_pass_count": combined_pass_count,
        "combined_pass_rate": combined_pass_count / count if count else 0.0,
        "metric_private_pass_at_1": float(metrics[0]["pass@1"]),
    }


def aggregate_traces(method: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(traces)
    solved = sum(1 for trace in traces if bool(trace.get("final_passed", False)))
    feedback_pass_private_fail = sum(
        1
        for trace in traces
        if bool(trace.get("feedback_passed", False))
        and not bool(trace.get("private_passed", False))
    )
    feedback_fail_private_pass = sum(
        1
        for trace in traces
        if not bool(trace.get("feedback_passed", False))
        and bool(trace.get("private_passed", False))
    )
    feedback_fail_private_fail = (
        count - solved - feedback_pass_private_fail - feedback_fail_private_pass
    )
    if not traces:
        return {
            "method": method,
            "num_problems": 0,
            "solved_count": 0,
            "final_pass_rate": 0.0,
        }
    return {
        "method": method,
        "num_problems": count,
        "solved_count": solved,
        "final_pass_rate": round(solved / count, 6),
        "private_pass_count": sum(
            1 for trace in traces if bool(trace.get("private_passed", False))
        ),
        "private_pass_rate": round(
            sum(1 for trace in traces if bool(trace.get("private_passed", False)))
            / count,
            6,
        ),
        "feedback_pass_private_pass": solved,
        "feedback_pass_private_fail": feedback_pass_private_fail,
        "feedback_fail_private_pass": feedback_fail_private_pass,
        "feedback_fail_private_fail": feedback_fail_private_fail,
        "public_feedback_pass_count": sum(
            1 for trace in traces if bool(trace.get("feedback_passed", False))
        ),
        "average_llm_calls": round(mean(int(trace.get("llm_calls", 0)) for trace in traces), 6),
        "average_program_candidates": round(
            mean(int(trace.get("program_candidate_count", 0)) for trace in traces), 6
        ),
        "average_feedback_verifier_executions": round(
            mean(int(trace.get("feedback_verifier_executions", 0)) for trace in traces),
            6,
        ),
        "average_probe_executions": round(
            mean(int(trace.get("probe_executions", 0)) for trace in traces), 6
        ),
        "average_prompt_chars": round(
            mean(int(trace.get("prompt_chars", 0)) for trace in traces), 6
        ),
        "average_completion_chars": round(
            mean(int(trace.get("completion_chars", 0)) for trace in traces), 6
        ),
        "average_elapsed_seconds": round(
            mean(float(trace.get("elapsed_seconds", 0.0)) for trace in traces), 6
        ),
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "num_problems",
        "solved_count",
        "final_pass_rate",
        "metric_pass_at_1",
        "private_pass_count",
        "private_pass_rate",
        "metric_private_pass_at_1",
        "feedback_pass_private_pass",
        "feedback_pass_private_fail",
        "feedback_fail_private_pass",
        "feedback_fail_private_fail",
        "public_feedback_pass_count",
        "average_llm_calls",
        "average_program_candidates",
        "average_feedback_verifier_executions",
        "average_probe_executions",
        "average_prompt_chars",
        "average_completion_chars",
        "average_elapsed_seconds",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_main_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = {
        "specfix_bm": "SpecFix-BM",
        "lpw_adapted": "LPW-adapted",
    }
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Method", "Final Pass Rate", "Solved", "N", "Protocol"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Method": labels.get(str(row["method"]), str(row["method"])),
                    "Final Pass Rate": f"{float(row['final_pass_rate']):.4f}",
                    "Solved": int(row["solved_count"]),
                    "N": int(row["num_problems"]),
                    "Protocol": "public/private pass intersection",
                }
            )
