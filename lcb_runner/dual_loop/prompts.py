from typing import TYPE_CHECKING

from lcb_runner.dual_loop.spec import SpecScore, StructuredSpec, VerifierFeedback

if TYPE_CHECKING:
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


SPEC_JSON_SCHEMA = """{
  "task": "one-sentence task summary",
  "inputs": ["typed input requirement", "input format rule"],
  "outputs": ["typed output requirement", "output format rule"],
  "constraints": ["explicit numeric or logical constraint"],
  "rules": ["core semantic rule", "special return rule"],
  "edge_cases": ["edge case 1", "edge case 2"],
  "checkable_properties": ["property that can be checked automatically"],
  "tie_break": ["rule for multiple valid outputs if applicable"],
  "reference_strategy": "validator_only or reference_solver",
  "algorithmic_notes": ["optional implementation hint grounded in the prompt"],
  "non_checkable_notes": ["requirements that are descriptive only"]
}"""


SPEC_PATCH_JSON_SCHEMA = """{
  "constraints": ["updated constraint if needed"],
  "rules": ["updated rule if needed"],
  "edge_cases": ["updated edge case if needed"]
}"""

SPEC_PATCH_ALLOWED_FIELDS = (
    "task",
    "inputs",
    "outputs",
    "constraints",
    "rules",
    "edge_cases",
    "checkable_properties",
    "tie_break",
    "reference_strategy",
    "algorithmic_notes",
    "non_checkable_notes",
)


def build_spec_draft_prompt(problem: "CodeGenerationProblem") -> str:
    return f"""You are building a structured specification for a competitive programming problem.
Return JSON only. Do not include markdown, prose, or code fences.

Use this schema exactly:
{SPEC_JSON_SCHEMA}

Guidelines:
- Keep only requirements supported by the problem statement.
- Put executable requirements under constraints, rules, edge_cases, and checkable_properties.
- If multiple valid outputs are allowed and no unique canonical answer is required, set reference_strategy to "validator_only".
- If correctness depends on a unique optimal output that is hard to validate without solving the task, set reference_strategy to "reference_solver".
- Do not invent complexity constraints unless the prompt explicitly states them.

Problem:
{problem.question_content}
"""


def build_plan_draft_prompt(problem: "CodeGenerationProblem") -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are writing a concise natural-language solution plan for a competitive programming problem.
Return plain text only. Do not return code, JSON, or markdown fences.

Write the plan in four short parts:
1. Goal
2. Key constraints and invariants
3. Solution steps
4. Edge cases

Requirements:
- Focus on how the task should be solved.
- Mention critical constraints from the prompt.
- Keep the plan concise and implementation-oriented.
- Do not write pseudocode or Python code.

Problem:
{problem.question_content}
{starter}
"""


def build_pseudocode_draft_prompt(problem: "CodeGenerationProblem") -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are writing language-agnostic pseudocode for a competitive programming problem.
Return plain text only. Do not return Python code, JSON, or markdown fences.

Requirements:
- Capture the main algorithm and control flow.
- Mention important constraints and special cases in comments when needed.
- Keep the pseudocode precise enough to implement directly.
- Do not output executable Python.

Problem:
{problem.question_content}
{starter}
"""


def build_spec_score_prompt(
    problem: "CodeGenerationProblem", spec: StructuredSpec
) -> str:
    return f"""You are judging whether a structured spec is aligned with a programming problem.
Return JSON only with the following schema:
{{
  "coverage": 0,
  "faithfulness": 0,
  "precision": 0,
  "overall": 0,
  "missing_constraints": ["..."],
  "unsupported_constraints": ["..."],
  "ambiguities": ["..."],
  "requires_refine": false,
  "blocking_issues": ["..."],
  "target_fields": ["constraints", "rules"],
  "edit_plan": ["Add one executable rule ...", "Clarify one missing constraint ..."],
  "do_not_change": ["task", "inputs", "outputs"],
  "proposed_patch": {{"rules": ["updated executable rule"], "constraints": ["updated constraint"]}},
  "action": "one short revision instruction"
}}

Output rules:
- Return a single raw JSON object.
- Do not use markdown, code fences, headings, or explanatory text.
- Every score must be an integer from 0 to 100.
- If a list is empty, return [].

Scoring rules:
- coverage: whether the spec covers the problem requirements
- faithfulness: whether the spec avoids unsupported assumptions
- precision: whether the spec is clear enough to drive code generation
- overall: rounded weighted score using 0.4 coverage, 0.4 faithfulness, 0.2 precision
- requires_refine: true only if a revision is likely to improve downstream code generation
- blocking_issues: concrete issues that materially block code generation
- target_fields: list the fields that should be edited; keep the list minimal
- edit_plan: provide concrete field-level edits; avoid generic advice like "clarify the spec"
- do_not_change: fields that are already adequate and should be preserved verbatim
- proposed_patch: provide a concrete candidate update for the target fields; address the blocking issues and omit unchanged fields

Decision rules:
- If the spec only has minor wording ambiguity and no missing or unsupported constraints, set requires_refine to false.
- If no revision is needed, return target_fields as [], edit_plan as [], and proposed_patch as {{}}.
- If requires_refine is true, focus on the highest-priority blocking issues first.
- Keep the patch focused and avoid broad rewrites.
- If multiple issues exist, leave lower-priority issues for later iterations.
- Prefer local edits over full rewrites.

Problem:
{problem.question_content}

Spec:
{spec.to_json()}
"""


def build_spec_refine_prompt(
    problem: "CodeGenerationProblem", spec: StructuredSpec, score: SpecScore
) -> str:
    return f"""Produce a field-level patch for the structured spec.
Return JSON only. Return a JSON object that contains only the fields you want to update.
Do not return the full spec. If no safe update is possible, return {{}}.

Example patch schema:
{SPEC_PATCH_JSON_SCHEMA}

Problem:
{problem.question_content}

Current spec:
{spec.to_json()}

Review feedback:
{score.to_json()}

Requirements:
- If requires_refine is false, return {{}}.
- Use proposed_patch from the review feedback as the primary candidate to normalize or minimally adjust.
- Modify only the fields listed in target_fields.
- Preserve every field listed in do_not_change exactly unless changing it is absolutely necessary to remove an unsupported assumption.
- Return only updated field values. Omit every unchanged field.
- Preserve supported constraints.
- Add missing constraints.
- Remove unsupported assumptions.
- Keep the spec concise and executable.
- Do not rewrite the whole spec.
- Do not introduce edits outside target_fields or broad wording-only rewrites.
- Apply the edit_plan concretely and avoid generic wording-only changes.
"""


def build_spec_json_repair_prompt(raw_output: str) -> str:
    return f"""Convert the following response into one valid JSON object using this schema exactly:
{SPEC_JSON_SCHEMA}

Rules:
- Return JSON only.
- Do not add markdown, code fences, or commentary.
- If a field is missing, return [] for list fields and "validator_only" for reference_strategy.
- Preserve the original meaning when possible.

Response to normalize:
{raw_output}
"""


def build_spec_score_json_repair_prompt(raw_output: str) -> str:
    return f"""Convert the following response into one valid JSON object with this schema exactly:
{{
  "coverage": 0,
  "faithfulness": 0,
  "precision": 0,
  "overall": 0,
  "missing_constraints": ["..."],
  "unsupported_constraints": ["..."],
  "ambiguities": ["..."],
  "requires_refine": false,
  "blocking_issues": ["..."],
  "target_fields": ["constraints", "rules"],
  "edit_plan": ["Add one executable rule ...", "Clarify one missing constraint ..."],
  "do_not_change": ["task", "inputs", "outputs"],
  "proposed_patch": {{"rules": ["updated executable rule"], "constraints": ["updated constraint"]}},
  "action": "one short revision instruction"
}}

Rules:
- Return JSON only.
- Do not add markdown, code fences, or commentary.
- Every score must be an integer from 0 to 100.
- Preserve the original judgment when possible.
- Keep target_fields minimal and focused on the blocking issues.
- Keep edit_plan concrete and field-level.
- Keep proposed_patch focused on targeted local updates rather than broad rewrites.

Response to normalize:
{raw_output}
"""


def build_spec_patch_json_repair_prompt(raw_output: str) -> str:
    return f"""Convert the following response into one valid JSON patch object.

Rules:
- Return JSON only.
- Do not add markdown, code fences, or commentary.
- The object may contain only these fields:
  {", ".join(SPEC_PATCH_ALLOWED_FIELDS)}
- Omit unchanged fields.
- If no valid patch is recoverable, return {{}}.

Response to normalize:
{raw_output}
"""


def build_code_block_repair_prompt(raw_output: str) -> str:
    return f"""Extract Python code from the following response.

Rules:
- Return exactly one fenced Python code block and nothing else.
- Preserve the original code when possible.
- If there is no recoverable Python code in the response, return an empty Python code block.
- Do not invent new logic that is not already present.

Response to normalize:
{raw_output}
"""


def build_code_from_spec_prompt(
    problem: "CodeGenerationProblem", spec: StructuredSpec
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are an expert Python programmer.
Write a complete Python solution that satisfies the structured spec and solves the original problem.
Return only one Python code block.

Original problem:
{problem.question_content}

Structured spec:
{spec.to_text()}
{starter}
Requirements:
- Follow the stated input/output protocol.
- Do not print extra text.
- Prefer a direct, contest-style solution.
"""


def build_code_from_plan_prompt(problem: "CodeGenerationProblem", plan_text: str) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are an expert Python programmer.
Write a complete Python solution using the natural-language solution plan below.
Return only one Python code block.

Original problem:
{problem.question_content}

Solution plan:
{plan_text}
{starter}
Requirements:
- Follow the exact input/output contract from the original problem.
- Use the plan as guidance, but resolve any missing implementation details from the original problem statement.
- Do not print extra text.
- Prefer a direct, contest-style solution.
"""


def build_code_from_pseudocode_prompt(
    problem: "CodeGenerationProblem", pseudocode_text: str
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are an expert Python programmer.
Write a complete Python solution using the pseudocode below.
Return only one Python code block.

Original problem:
{problem.question_content}

Pseudocode:
{pseudocode_text}
{starter}
Requirements:
- Follow the exact input/output contract from the original problem.
- Preserve the logic of the pseudocode while fixing any language-level gaps.
- Do not print extra text.
- Prefer a direct, contest-style solution.
"""


def build_direct_codegen_prompt(problem: "CodeGenerationProblem") -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are an expert Python programmer.
Solve the following programming problem.
Return exactly one complete Python code block and nothing else.

Problem:
{problem.question_content}
{starter}
Requirements:
- Follow the exact input/output contract.
- Do not print extra text.
- Prefer a direct, contest-style solution.
"""


def build_repair_prompt(
    problem: "CodeGenerationProblem",
    spec: StructuredSpec,
    code: str,
    feedback: VerifierFeedback,
    *,
    require_change: bool = False,
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    change_requirement = ""
    if require_change:
        change_requirement = (
            "\n- Your previous repair attempt did not change the failing behavior."
            "\n- Do not return the same program again."
            "\n- Change the logic that causes the reported mismatch."
        )
    return f"""You are repairing a Python program for a competitive programming problem.
First identify the issue briefly, then return a complete fixed program in exactly one Python code block.

Original problem:
{problem.question_content}

Structured spec:
{spec.to_text()}
{starter}
Current code:
```python
{code}
```

Verifier feedback:
{feedback.to_json()}

Repair target:
- Fix the failure without breaking the input/output contract.
- Satisfy the checkable subset of the structured spec.
- If property_feedbacks are present, treat them as the highest-priority semantic obligations to fix.
{change_requirement}
"""


def build_self_refine_repair_prompt(
    problem: "CodeGenerationProblem",
    code: str,
    feedback: VerifierFeedback,
    *,
    require_change: bool = False,
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    change_requirement = ""
    if require_change:
        change_requirement = (
            "\n- Your previous revision did not fix the failure."
            "\n- Do not return the same program again."
            "\n- Change the failing logic."
        )
    return f"""You are refining a Python solution after execution feedback.
Briefly think about why the program failed, then return exactly one complete Python code block.

Problem:
{problem.question_content}
{starter}
Current code:
```python
{code}
```

Execution feedback:
{feedback.to_json()}

Requirements:
- Fix the reported failure.
- Preserve the input/output contract.
- Return only the revised program.
{change_requirement}
"""


def build_reflexion_prompt(
    problem: "CodeGenerationProblem",
    code: str,
    feedback: VerifierFeedback,
    reflections: list[str],
) -> str:
    prior_reflections = "\n".join(f"- {item}" for item in reflections) or "- None yet."
    return f"""You are writing a short reflection for a failed Python solution.
Return 1 to 3 bullet points only. Do not return code.

Problem:
{problem.question_content}

Current code:
```python
{code}
```

Execution feedback:
{feedback.to_json()}

Previous reflections:
{prior_reflections}

Write reflections that identify:
- the likely root cause
- what to change next
- what to avoid repeating
"""


def build_reflexion_repair_prompt(
    problem: "CodeGenerationProblem",
    code: str,
    feedback: VerifierFeedback,
    reflections: list[str],
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    reflection_block = "\n".join(f"- {item}" for item in reflections) or "- No reflections available."
    return f"""You are revising a Python solution using accumulated reflections from prior failures.
Return exactly one complete Python code block and nothing else.

Problem:
{problem.question_content}
{starter}
Current code:
```python
{code}
```

Execution feedback:
{feedback.to_json()}

Reflections:
{reflection_block}

Requirements:
- Use the reflections to fix the root cause.
- If property_feedbacks are present, prioritize fixing those violated properties.
- Do not repeat the same failing logic.
- Preserve the input/output contract.
"""


def build_counterexample_repair_prompt(
    problem: "CodeGenerationProblem",
    spec: StructuredSpec,
    code: str,
    feedback: VerifierFeedback,
    counterexample: str,
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are fixing a wrong-answer bug in a Python program.
Return exactly one complete Python code block and nothing else.

Original problem:
{problem.question_content}

Structured spec:
{spec.to_text()}
{starter}
Current code:
```python
{code}
```

Bug summary:
- The current program fails on this concrete counterexample.
- Fix the logic that causes this mismatch.
- If property_feedbacks are present, use them to identify which semantic obligation is violated.
- Do not return the same program again.

Counterexample:
{counterexample}

Verifier feedback:
{feedback.to_json()}
"""


def build_rewrite_from_counterexample_prompt(
    problem: "CodeGenerationProblem",
    spec: StructuredSpec,
    feedback: VerifierFeedback,
    counterexample: str,
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
    return f"""You are rewriting a Python solution for a competitive programming problem after multiple failed repair attempts.
Return exactly one complete Python code block and nothing else.

Original problem:
{problem.question_content}

Structured spec:
{spec.to_text()}
{starter}
Requirements:
- Ignore the previous implementation and write a fresh solution from scratch.
- Follow the exact input/output protocol.
- Fix the wrong-answer behavior shown in the counterexample.
- If property_feedbacks are present, ensure the rewritten program satisfies those violated properties.
- Prefer the simplest correct contest-style solution.

Counterexample:
{counterexample}

Verifier feedback:
{feedback.to_json()}
"""
