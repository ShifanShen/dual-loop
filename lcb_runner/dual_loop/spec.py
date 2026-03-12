import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


_SPEC_SCORE_FIELD_ALIASES = {
    "coverage": ["coverage", "覆盖率", "覆盖", "完整性"],
    "faithfulness": ["faithfulness", "忠实度", "忠实性", "faithful"],
    "precision": ["precision", "精确度", "精确性", "明确性"],
    "overall": ["overall", "总分", "综合得分", "overall_score"],
    "missing_constraints": [
        "missing_constraints",
        "missing constraints",
        "缺失约束",
        "遗漏约束",
    ],
    "unsupported_constraints": [
        "unsupported_constraints",
        "unsupported constraints",
        "不支持的约束",
        "臆造约束",
    ],
    "ambiguities": ["ambiguities", "ambiguity", "歧义", "模糊点"],
    "action": ["action", "建议", "修订建议", "revision", "next_action"],
}


def _normalize_text(text: str) -> str:
    return (
        text.replace("：", ":")
        .replace("，", ",")
        .replace("（", "(")
        .replace("）", ")")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]


def _extract_json_block(text: str) -> dict[str, Any] | None:
    text = _normalize_text(text)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced or re.findall(r"(\{.*\})", text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _field_aliases(field: str) -> list[str]:
    return _SPEC_SCORE_FIELD_ALIASES.get(field, [field])


def _extract_int_field(text: str, field: str) -> int:
    text = _normalize_text(text)
    for alias in _field_aliases(field):
        pattern = rf'["\']?{re.escape(alias)}["\']?\s*[:=]\s*(-?\d+)(?:\s*/\s*100|%)?'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        pattern = rf"{re.escape(alias)}\s*(?:score)?\s*[:=]\s*(-?\d+)(?:\s*/\s*100|%)?"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _parse_list_candidate(candidate: str) -> list[str]:
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(candidate)
            return _ensure_list(value)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return []


def _extract_list_field(text: str, field: str) -> list[str]:
    text = _normalize_text(text)
    for alias in _field_aliases(field):
        bracket_pattern = rf'["\']?{re.escape(alias)}["\']?\s*[:=]\s*(\[[\s\S]*?\])'
        match = re.search(bracket_pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = _parse_list_candidate(match.group(1))
            if parsed:
                return parsed

        line_pattern = rf'["\']?{re.escape(alias)}["\']?\s*[:=]\s*(.+)'
        match = re.search(line_pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        value = match.group(1).strip().splitlines()[0].strip()
        if not value or value.lower() in {"[]", "none", "null", "n/a"}:
            return []
        if value.startswith("[") and value.endswith("]"):
            parsed = _parse_list_candidate(value)
            if parsed:
                return parsed
        return [item.strip(" -\"'") for item in value.split(",") if item.strip(" -\"'")]
    return []


def _extract_string_field(text: str, field: str) -> str:
    text = _normalize_text(text)
    for alias in _field_aliases(field):
        quoted_pattern = rf'["\']?{re.escape(alias)}["\']?\s*[:=]\s*"([^"]*)"'
        match = re.search(quoted_pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        single_quoted_pattern = rf'["\']?{re.escape(alias)}["\']?\s*[:=]\s*\'([^\']*)\''
        match = re.search(single_quoted_pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        line_pattern = rf'["\']?{re.escape(alias)}["\']?\s*[:=]\s*(.+)'
        match = re.search(line_pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip().splitlines()[0].strip()
        return value.strip(" -\"'")
    return ""


def _coerce_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    nested_scores = payload.get("scores")
    if isinstance(nested_scores, dict):
        payload = {**nested_scores, **payload}

    for canonical, aliases in _SPEC_SCORE_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in payload:
                normalized[canonical] = payload[alias]
                break
    return normalized or payload


def _to_int_score(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = _normalize_text(str(value)).strip()
    match = re.search(r"(-?\d+)(?:\s*/\s*100|%)?", text)
    if match:
        return int(match.group(1))
    return 0


@dataclass
class StructuredSpec:
    task: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    checkable_properties: list[str] = field(default_factory=list)
    tie_break: list[str] = field(default_factory=list)
    reference_strategy: str = "validator_only"
    algorithmic_notes: list[str] = field(default_factory=list)
    non_checkable_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_llm_output(cls, text: str, fallback_task: str = "") -> "StructuredSpec":
        payload = _extract_json_block(text) or {}
        return cls(
            task=str(payload.get("task", fallback_task)).strip(),
            inputs=_ensure_list(payload.get("inputs")),
            outputs=_ensure_list(payload.get("outputs")),
            constraints=_ensure_list(payload.get("constraints")),
            rules=_ensure_list(payload.get("rules")),
            edge_cases=_ensure_list(payload.get("edge_cases")),
            checkable_properties=_ensure_list(payload.get("checkable_properties")),
            tie_break=_ensure_list(payload.get("tie_break")),
            reference_strategy=str(
                payload.get("reference_strategy", "validator_only")
            ).strip()
            or "validator_only",
            algorithmic_notes=_ensure_list(payload.get("algorithmic_notes")),
            non_checkable_notes=_ensure_list(payload.get("non_checkable_notes")),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2)

    def to_text(self) -> str:
        sections = [
            ("Task", [self.task] if self.task else []),
            ("Inputs", self.inputs),
            ("Outputs", self.outputs),
            ("Constraints", self.constraints),
            ("Rules", self.rules),
            ("Edge Cases", self.edge_cases),
            ("Checkable Properties", self.checkable_properties),
            ("Tie Break", self.tie_break),
            ("Reference Strategy", [self.reference_strategy]),
            ("Algorithmic Notes", self.algorithmic_notes),
            ("Non-checkable Notes", self.non_checkable_notes),
        ]
        chunks = []
        for title, values in sections:
            if not values:
                continue
            chunks.append(f"[{title}]")
            chunks.extend(f"- {value}" for value in values)
            chunks.append("")
        return "\n".join(chunks).strip()


@dataclass
class SpecScore:
    coverage: int = 0
    faithfulness: int = 0
    precision: int = 0
    overall: int = 0
    missing_constraints: list[str] = field(default_factory=list)
    unsupported_constraints: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    action: str = ""
    parse_ok: bool = False
    parse_source: str = "default"

    @classmethod
    def from_llm_output(cls, text: str) -> "SpecScore":
        payload = _extract_json_block(text) or {}
        parse_source = "json"
        if payload:
            payload = _coerce_score_payload(payload)
        if not payload:
            parse_source = "fallback"
            payload = {
                "coverage": _extract_int_field(text, "coverage"),
                "faithfulness": _extract_int_field(text, "faithfulness"),
                "precision": _extract_int_field(text, "precision"),
                "overall": _extract_int_field(text, "overall"),
                "missing_constraints": _extract_list_field(
                    text, "missing_constraints"
                ),
                "unsupported_constraints": _extract_list_field(
                    text, "unsupported_constraints"
                ),
                "ambiguities": _extract_list_field(text, "ambiguities"),
                "action": _extract_string_field(text, "action"),
            }
        parse_ok = any(
            [
                _to_int_score(payload.get("coverage", 0)),
                _to_int_score(payload.get("faithfulness", 0)),
                _to_int_score(payload.get("precision", 0)),
                _to_int_score(payload.get("overall", 0)),
                bool(_ensure_list(payload.get("missing_constraints"))),
                bool(_ensure_list(payload.get("unsupported_constraints"))),
                bool(_ensure_list(payload.get("ambiguities"))),
                bool(str(payload.get("action", "")).strip()),
            ]
        )
        return cls(
            coverage=_to_int_score(payload.get("coverage", 0)),
            faithfulness=_to_int_score(payload.get("faithfulness", 0)),
            precision=_to_int_score(payload.get("precision", 0)),
            overall=_to_int_score(payload.get("overall", 0)),
            missing_constraints=_ensure_list(payload.get("missing_constraints")),
            unsupported_constraints=_ensure_list(
                payload.get("unsupported_constraints")
            ),
            ambiguities=_ensure_list(payload.get("ambiguities")),
            action=str(payload.get("action", "")).strip(),
            parse_ok=parse_ok,
            parse_source=parse_source if parse_ok else "default",
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2)


@dataclass
class VerifierFeedback:
    passed: bool
    error_type: str
    field: str
    message: str
    input: str = ""
    output: str = ""
    expected: str = ""
    violated_spec_items: list[str] = field(default_factory=list)
    repair_hint: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2)
