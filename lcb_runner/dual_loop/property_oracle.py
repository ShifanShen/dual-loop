import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from lcb_runner.dual_loop.spec import StructuredSpec


@dataclass
class PropertyClause:
    property_type: str
    description: str
    source_field: str
    parameters: dict[str, Any] = field(default_factory=dict)
    executable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PropertyFeedback:
    property_type: str
    source_field: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_property_clauses(spec: StructuredSpec) -> list[PropertyClause]:
    clauses: list[PropertyClause] = []
    seen: set[str] = set()
    sources = [
        ("checkable_properties", spec.checkable_properties),
        ("rules", spec.rules),
        ("outputs", spec.outputs),
        ("constraints", spec.constraints),
        ("edge_cases", spec.edge_cases),
    ]
    for source_field, items in sources:
        for item in items:
            for clause in _clauses_from_text(item, source_field):
                signature = json.dumps(
                    {
                        "property_type": clause.property_type,
                        "description": clause.description,
                        "source_field": clause.source_field,
                        "parameters": clause.parameters,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                clauses.append(clause)
    return clauses


def evaluate_property_clauses(
    clauses: list[PropertyClause],
    *,
    actual_output: str,
    expected_output: str,
    raw_input: str = "",
) -> list[PropertyFeedback]:
    feedbacks: list[PropertyFeedback] = []
    for clause in clauses:
        feedback = _evaluate_clause(
            clause,
            actual_output=actual_output,
            expected_output=expected_output,
            raw_input=raw_input,
        )
        if feedback is not None:
            feedbacks.append(feedback)
    return feedbacks


def render_property_feedback(feedbacks: list[PropertyFeedback]) -> str:
    if not feedbacks:
        return "None."
    lines: list[str] = []
    for feedback in feedbacks:
        lines.append(f"- {feedback.property_type}: {feedback.message}")
    return "\n".join(lines)


def _clauses_from_text(text: str, source_field: str) -> list[PropertyClause]:
    lowered = text.lower()
    clauses: list[PropertyClause] = []

    if _contains_any(
        lowered,
        (
            "ascending order",
            "sorted ascending",
            "sorted in ascending",
            "nondecreasing",
            "in increasing order",
        ),
    ):
        clauses.append(
            PropertyClause(
                property_type="sorted_order",
                description=text,
                source_field=source_field,
                parameters={"order": "ascending"},
            )
        )

    if _contains_any(
        lowered,
        (
            "descending order",
            "sorted descending",
            "sorted in descending",
            "nonincreasing",
            "in decreasing order",
        ),
    ):
        clauses.append(
            PropertyClause(
                property_type="sorted_order",
                description=text,
                source_field=source_field,
                parameters={"order": "descending"},
            )
        )

    if _contains_any(
        lowered,
        (
            "same multiset",
            "same elements",
            "same integers",
            "same characters",
            "same values",
            "contains exactly the same",
            "permutation of the input",
        ),
    ):
        clauses.append(
            PropertyClause(
                property_type="same_multiset",
                description=text,
                source_field=source_field,
            )
        )

    if _contains_any(
        lowered,
        (
            "same length",
            "length equals input length",
            "output length equals input length",
            "output length must equal input length",
        ),
    ):
        clauses.append(
            PropertyClause(
                property_type="same_length",
                description=text,
                source_field=source_field,
            )
        )

    if _contains_any(
        lowered,
        (
            "output \"yes\"",
            "output \"no\"",
            "output 'yes'",
            "output 'no'",
            "output yes",
            "output no",
            "yes or no",
        ),
    ):
        clauses.append(
            PropertyClause(
                property_type="yes_no_output",
                description=text,
                source_field=source_field,
            )
        )

    return clauses


def _evaluate_clause(
    clause: PropertyClause,
    *,
    actual_output: str,
    expected_output: str,
    raw_input: str = "",
) -> PropertyFeedback | None:
    property_type = clause.property_type
    if property_type == "sorted_order":
        actual_values = _coerce_numeric_sequence(_parse_sequence(actual_output))
        if len(actual_values) < 2:
            return None
        order = clause.parameters.get("order", "ascending")
        if order == "ascending":
            ok = all(actual_values[idx] <= actual_values[idx + 1] for idx in range(len(actual_values) - 1))
        else:
            ok = all(actual_values[idx] >= actual_values[idx + 1] for idx in range(len(actual_values) - 1))
        if ok:
            return None
        return PropertyFeedback(
            property_type=property_type,
            source_field=clause.source_field,
            message=f"output is not sorted in {order} order",
            evidence={"actual_output": actual_output.strip()},
        )

    if property_type == "same_length":
        actual_tokens = _tokenize_scalars(actual_output)
        expected_tokens = _tokenize_scalars(expected_output)
        if not actual_tokens or not expected_tokens:
            return None
        if len(actual_tokens) == len(expected_tokens):
            return None
        return PropertyFeedback(
            property_type=property_type,
            source_field=clause.source_field,
            message="output length differs from the expected answer",
            evidence={
                "actual_length": len(actual_tokens),
                "expected_length": len(expected_tokens),
            },
        )

    if property_type == "same_multiset":
        actual_tokens = _tokenize_scalars(actual_output)
        expected_tokens = _tokenize_scalars(expected_output)
        if not actual_tokens or not expected_tokens:
            return None
        if Counter(actual_tokens) == Counter(expected_tokens):
            return None
        return PropertyFeedback(
            property_type=property_type,
            source_field=clause.source_field,
            message="output does not preserve the expected multiset of values",
            evidence={
                "actual_tokens": actual_tokens[:20],
                "expected_tokens": expected_tokens[:20],
            },
        )

    if property_type == "yes_no_output":
        tokens = _tokenize_scalars(actual_output)
        if len(tokens) != 1:
            return PropertyFeedback(
                property_type=property_type,
                source_field=clause.source_field,
                message="output is not a single YES/NO token",
                evidence={"actual_output": actual_output.strip()},
            )
        if str(tokens[0]).upper() in {"YES", "NO"}:
            return None
        return PropertyFeedback(
            property_type=property_type,
            source_field=clause.source_field,
            message="output token is not YES or NO",
            evidence={"actual_output": actual_output.strip()},
        )

    return None


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _parse_sequence(text: str) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(stripped)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(value, list) and value:
            if all(isinstance(item, (int, float, str)) for item in value):
                return value

    tokens = _tokenize_scalars(stripped)
    if len(tokens) > 1:
        return tokens
    return []


def _coerce_numeric_sequence(values: list[Any]) -> list[int | float]:
    coerced: list[int | float] = []
    for value in values:
        if isinstance(value, bool):
            return []
        if isinstance(value, (int, float)):
            coerced.append(value)
            continue
        if isinstance(value, str):
            value = value.strip()
            if re.fullmatch(r"-?\d+", value):
                coerced.append(int(value))
                continue
            if re.fullmatch(r"-?\d+\.\d+", value):
                coerced.append(float(value))
                continue
        return []
    return coerced


def _tokenize_scalars(text: str) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []
    tokens = re.split(r"[\s,]+", stripped)
    values: list[Any] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"-?\d+", token):
            values.append(int(token))
            continue
        if re.fullmatch(r"-?\d+\.\d+", token):
            values.append(float(token))
            continue
        values.append(token)
    return values
