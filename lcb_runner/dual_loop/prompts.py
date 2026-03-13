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

Problem:
{problem.question_content}

Spec:
{spec.to_json()}
"""


def build_spec_refine_prompt(
    problem: "CodeGenerationProblem", spec: StructuredSpec, score: SpecScore
) -> str:
    return f"""Revise the structured spec so it is more faithful to the problem.
Return JSON only using the same schema as before.

Problem:
{problem.question_content}

Current spec:
{spec.to_json()}

Review feedback:
{score.to_json()}

Requirements:
- Preserve supported constraints.
- Add missing constraints.
- Remove unsupported assumptions.
- Keep the spec concise and executable.
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
  "action": "one short revision instruction"
}}

Rules:
- Return JSON only.
- Do not add markdown, code fences, or commentary.
- Every score must be an integer from 0 to 100.
- Preserve the original judgment when possible.

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


def build_repair_prompt(
    problem: "CodeGenerationProblem",
    spec: StructuredSpec,
    code: str,
    feedback: VerifierFeedback,
) -> str:
    starter = ""
    if problem.starter_code:
        starter = f"\nStarter code:\n```python\n{problem.starter_code}\n```\n"
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
"""
