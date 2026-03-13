import json
import os
import time
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from lcb_runner.dual_loop.prompts import (
    build_code_block_repair_prompt,
    build_code_from_spec_prompt,
    build_counterexample_repair_prompt,
    build_direct_codegen_prompt,
    build_repair_prompt,
    build_reflexion_prompt,
    build_reflexion_repair_prompt,
    build_rewrite_from_counterexample_prompt,
    build_spec_draft_prompt,
    build_spec_json_repair_prompt,
    build_self_refine_repair_prompt,
    build_spec_refine_prompt,
    build_spec_score_prompt,
    build_spec_score_json_repair_prompt,
)
from lcb_runner.dual_loop.spec import SpecScore, StructuredSpec, VerifierFeedback
from lcb_runner.lm_styles import LMStyle
from lcb_runner.lm_styles import resolve_language_model
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.extraction_utils import extract_code

if TYPE_CHECKING:
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


@dataclass
class ProblemTrace:
    question_id: str
    question_title: str
    pipeline_mode: str
    raw_problem: str
    spec_initial: dict[str, Any] | None = None
    spec_final: dict[str, Any] | None = None
    initial_spec_score: dict[str, Any] | None = None
    final_spec_score: dict[str, Any] | None = None
    spec_scores: list[dict[str, Any]] = field(default_factory=list)
    spec_issue_types: list[str] = field(default_factory=list)
    code_initial: str = ""
    final_code: str = ""
    verifier_feedbacks: list[dict[str, Any]] = field(default_factory=list)
    failure_attribution: str = "unknown"
    failure_reason_tags: list[str] = field(default_factory=list)
    llm_calls: int = 0
    spec_calls: int = 0
    judge_calls: int = 0
    codegen_calls: int = 0
    repair_calls: int = 0
    loop_b_iterations: int = 0
    loop_a_iterations: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    stage_times: dict[str, float] = field(default_factory=dict)
    raw_spec_outputs: list[str] = field(default_factory=list)
    raw_score_outputs: list[str] = field(default_factory=list)
    raw_codegen_output: str = ""
    raw_repair_outputs: list[str] = field(default_factory=list)
    effectiveness: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    repair_iterations: int = 0
    elapsed_seconds: float = 0.0


class LLMAdapter:
    def __init__(self, args):
        self.args = args
        self.model = resolve_language_model(
            args.model,
            local_model_path=args.local_model_path,
            model_style_override=getattr(args, "model_style", None),
            model_repr_override=getattr(args, "model_repr", None),
        )
        self.runner = build_runner(args, self.model)
        self.total_call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        role: str = "generic",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int | str]]:
        self._apply_sampling_overrides(temperature=temperature, max_tokens=max_tokens)
        self.total_call_count += 1
        formatted_prompt = self._format_prompt(prompt)
        output = self.runner.run_batch([formatted_prompt])[0][0]
        usage = {
            "role": role,
            "prompt_chars": len(prompt),
            "completion_chars": len(output),
            "call_index": self.total_call_count,
        }
        return output, usage

    def extract_code(self, output: str) -> str:
        code = extract_code(output, self.model.model_style).strip()
        return code or output.strip()

    def _format_prompt(self, prompt: str) -> str | list[dict[str, str]] | tuple[str, list[dict[str, str]]]:
        style = self.model.model_style

        if style in {
            LMStyle.OpenAIChat,
            LMStyle.DeepSeekAPI,
            LMStyle.MistralWeb,
            LMStyle.TogetherAI,
            LMStyle.CohereCommand,
        }:
            return [{"role": "user", "content": prompt}]

        if style in {LMStyle.OpenAIReasonPreview, LMStyle.OpenAIReason, LMStyle.Grok}:
            return [{"role": "user", "content": prompt}]

        if style in {LMStyle.Claude3, LMStyle.Claude3Thinking}:
            return ("You are a helpful assistant.", [{"role": "user", "content": prompt}])

        if style == LMStyle.CodeQwenInstruct:
            return (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt.rstrip()}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        if style == LMStyle.QwQ:
            return (
                "<|im_start|>system\n"
                "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. "
                "You should think step-by-step.<|im_end|>\n"
                f"<|im_start|>user\n{prompt.rstrip()}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        if style == LMStyle.LLaMa3:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt.rstrip()}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        if style == LMStyle.DeepSeekCodeInstruct:
            return f"### Instruction:\n{prompt.rstrip()}\n\n### Response:\n"

        return prompt

    def _apply_sampling_overrides(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        sampling_params = getattr(self.runner, "sampling_params", None)
        if sampling_params is None:
            return
        if temperature is not None:
            sampling_params.temperature = temperature
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens


class DualLoopPipeline:
    def __init__(self, args, llm: LLMAdapter | None = None):
        self.args = args
        self.llm = llm or LLMAdapter(args)
        self.output_dir = self._make_output_dir()

    def run(self, benchmark: list["CodeGenerationProblem"] | None = None) -> dict[str, Any]:
        benchmark = benchmark or self._load_benchmark()
        traces: list[ProblemTrace] = []
        generations: list[list[str]] = []

        for problem in benchmark:
            started_at = datetime.now()
            trace = self._run_problem(problem)
            trace.elapsed_seconds = (datetime.now() - started_at).total_seconds()
            traces.append(trace)
            generations.append([trace.final_code])

        metrics = self._compute_metrics(benchmark, generations)
        summary = self._build_summary(benchmark, traces, metrics)
        self._write_outputs(summary, traces)
        return summary

    def _run_problem(self, problem: "CodeGenerationProblem") -> ProblemTrace:
        trace = ProblemTrace(
            question_id=problem.question_id,
            question_title=problem.question_title,
            pipeline_mode=self.args.pipeline_mode,
            raw_problem=problem.question_content,
        )

        if self.args.pipeline_mode in {"baseline", "self_refine", "reflexion"}:
            initial_code = self._generate_code_baseline(problem)
            trace.code_initial = initial_code
            trace.raw_codegen_output = getattr(self, "_last_baseline_codegen_output", "")
            trace.effectiveness["codegen"] = self._build_codegen_effectiveness(
                raw_output=trace.raw_codegen_output,
                code=initial_code,
                strategy=f"{self.args.pipeline_mode}_codegen",
            )
            self._record_usage(trace, getattr(self, "_last_baseline_codegen_usage", None), "codegen")
            for extra_usage in getattr(self, "_last_baseline_codegen_extra_usages", []):
                self._record_usage(trace, extra_usage, "codegen")
            self._record_stage_time(
                trace,
                "codegen",
                float(getattr(self, "_last_baseline_codegen_stage_time", 0.0)),
            )
            if self.args.pipeline_mode == "baseline":
                trace.final_code = initial_code
                feedback = self._verify(problem, initial_code)
                trace.verifier_feedbacks.append(asdict(feedback))
                trace.passed = feedback.passed
                self._finalize_trace(trace)
                return trace
            repaired_code, feedback_trace, repair_meta = self._repair_direct_code(
                problem,
                initial_code,
                strategy=self.args.pipeline_mode,
            )
            trace.repair_iterations = max(0, len(feedback_trace) - 1)
            trace.loop_a_iterations = len(feedback_trace)
            trace.verifier_feedbacks.extend(feedback_trace)
            trace.raw_repair_outputs.extend(repair_meta["raw_repair_outputs"])
            for usage in repair_meta["repair_usages"]:
                self._record_usage(trace, usage, "repair")
            for duration in repair_meta["stage_times"]:
                self._record_stage_time(trace, "repair", float(duration))
            trace.effectiveness["repair_steps"] = list(repair_meta["effectiveness_steps"])
            if repair_meta["reflections"]:
                trace.effectiveness["reflections"] = list(repair_meta["reflections"])
            trace.final_code = repaired_code
            trace.passed = bool(feedback_trace[-1]["passed"]) if feedback_trace else False
            self._finalize_trace(trace)
            return trace

        spec = self._draft_spec(problem)
        trace.spec_initial = asdict(spec)
        trace.effectiveness["spec_draft"] = self._build_spec_draft_effectiveness(spec)
        trace.raw_spec_outputs.extend(
            getattr(spec, "_raw_attempt_outputs", [getattr(spec, "_raw_output", "")])
        )
        self._record_usage(trace, getattr(spec, "_usage", None), "spec")
        for extra_usage in getattr(spec, "_extra_usages", []):
            self._record_usage(trace, extra_usage, "spec")
        self._record_stage_time(trace, "spec_draft", getattr(spec, "_stage_time", 0.0))
        initial_score = self._score_spec(problem, spec)
        trace.initial_spec_score = asdict(initial_score)
        trace.spec_scores.append(asdict(initial_score))
        trace.raw_score_outputs.extend(
            getattr(
                initial_score,
                "_raw_attempt_outputs",
                [getattr(initial_score, "_raw_output", "")],
            )
        )
        trace.spec_issue_types.extend(self._extract_spec_issue_types(initial_score))
        trace.effectiveness["spec_score_initial"] = self._build_spec_score_effectiveness(
            initial_score
        )
        self._record_usage(trace, getattr(initial_score, "_usage", None), "judge")
        for extra_usage in getattr(initial_score, "_extra_usages", []):
            self._record_usage(trace, extra_usage, "judge")
        self._record_stage_time(
            trace, "spec_score_initial", getattr(initial_score, "_stage_time", 0.0)
        )
        if self.args.pipeline_mode in {"loop_b", "full"}:
            spec, score_trace, refine_meta = self._refine_spec(problem, spec)
            trace.spec_scores.extend(score_trace)
            trace.loop_b_iterations = len(score_trace)
            trace.raw_score_outputs.extend(refine_meta["raw_score_outputs"])
            trace.raw_spec_outputs.extend(refine_meta["raw_spec_outputs"])
            for usage in refine_meta["judge_usages"]:
                self._record_usage(trace, usage, "judge")
            for usage in refine_meta["spec_usages"]:
                self._record_usage(trace, usage, "spec")
            for stage_name, duration in refine_meta["stage_times"].items():
                self._record_stage_time(trace, stage_name, duration)
            trace.effectiveness["spec_refine_steps"] = refine_meta.get("effectiveness_steps", [])
        trace.spec_final = asdict(spec)
        final_score = self._score_spec(problem, spec)
        trace.final_spec_score = asdict(final_score)
        trace.raw_score_outputs.extend(
            getattr(
                final_score,
                "_raw_attempt_outputs",
                [getattr(final_score, "_raw_output", "")],
            )
        )
        trace.spec_issue_types.extend(self._extract_spec_issue_types(final_score))
        trace.effectiveness["spec_score_final"] = self._build_spec_score_effectiveness(
            final_score,
            previous_score=initial_score,
        )
        self._record_usage(trace, getattr(final_score, "_usage", None), "judge")
        for extra_usage in getattr(final_score, "_extra_usages", []):
            self._record_usage(trace, extra_usage, "judge")
        self._record_stage_time(
            trace, "spec_score_final", getattr(final_score, "_stage_time", 0.0)
        )

        code = self._generate_code_from_spec(problem, spec)
        trace.code_initial = code
        trace.raw_codegen_output = getattr(spec, "_last_codegen_output", "")
        trace.effectiveness["codegen"] = self._build_codegen_effectiveness(
            raw_output=trace.raw_codegen_output,
            code=code,
            strategy="spec_codegen",
        )
        self._record_usage(trace, getattr(spec, "_last_codegen_usage", None), "codegen")
        for extra_usage in getattr(spec, "_last_codegen_extra_usages", []):
            self._record_usage(trace, extra_usage, "codegen")
        self._record_stage_time(
            trace, "codegen", float(getattr(spec, "_last_codegen_stage_time", 0.0))
        )

        if self.args.pipeline_mode in {"loop_a", "full"}:
            code, feedback_trace = self._repair_code(problem, spec, code)
            trace.repair_iterations = max(0, len(feedback_trace) - 1)
            trace.loop_a_iterations = len(feedback_trace)
            trace.verifier_feedbacks.extend(feedback_trace)
            trace.raw_repair_outputs.extend(getattr(spec, "_repair_outputs", []))
            for usage in getattr(spec, "_repair_usages", []):
                self._record_usage(trace, usage, "repair")
            for duration in getattr(spec, "_repair_stage_times", []):
                self._record_stage_time(trace, "repair", float(duration))
            trace.effectiveness["repair_steps"] = list(
                getattr(spec, "_repair_effectiveness", [])
            )
            trace.final_code = code
            trace.passed = bool(feedback_trace[-1]["passed"]) if feedback_trace else False
            self._finalize_trace(trace)
            return trace

        feedback = self._verify(problem, code)
        trace.verifier_feedbacks.append(asdict(feedback))
        trace.final_code = code
        trace.passed = feedback.passed
        self._finalize_trace(trace)
        return trace

    def _draft_spec(self, problem: "CodeGenerationProblem") -> StructuredSpec:
        started_at = time.perf_counter()
        output, usage = self.llm.generate(
            build_spec_draft_prompt(problem),
            role="spec_draft",
            temperature=self.args.spec_temperature,
            max_tokens=self.args.spec_max_tokens,
        )
        spec, raw_attempt_outputs, extra_usages = self._parse_spec_output(
            output, fallback_task=problem.question_title
        )
        spec._raw_output = output
        spec._raw_attempt_outputs = raw_attempt_outputs
        spec._usage = usage
        spec._extra_usages = extra_usages
        spec._stage_time = time.perf_counter() - started_at
        return spec

    def _score_spec(self, problem: "CodeGenerationProblem", spec: StructuredSpec) -> SpecScore:
        started_at = time.perf_counter()
        output, usage = self.llm.generate(
            build_spec_score_prompt(problem, spec),
            role="judge",
            temperature=self.args.judge_temperature,
            max_tokens=self.args.judge_max_tokens,
        )
        raw_attempt_outputs = [output]
        extra_usages = []
        score = SpecScore.from_llm_output(output)
        if not score.parse_ok:
            repair_output, repair_usage = self.llm.generate(
                build_spec_score_json_repair_prompt(output),
                role="judge_json_repair",
                temperature=0.0,
                max_tokens=self.args.judge_max_tokens,
            )
            raw_attempt_outputs.append(repair_output)
            extra_usages.append(repair_usage)
            repaired_score = SpecScore.from_llm_output(repair_output)
            if repaired_score.parse_ok:
                output = repair_output
                score = repaired_score
        score._raw_output = output
        score._raw_attempt_outputs = raw_attempt_outputs
        score._usage = usage
        score._extra_usages = extra_usages
        score._stage_time = time.perf_counter() - started_at
        return score

    def _refine_spec(
        self, problem: "CodeGenerationProblem", spec: StructuredSpec
    ) -> tuple[StructuredSpec, list[dict[str, Any]], dict[str, Any]]:
        score_trace: list[dict[str, Any]] = []
        refine_meta = {
            "raw_score_outputs": [],
            "raw_spec_outputs": [],
            "judge_usages": [],
            "spec_usages": [],
            "stage_times": {},
            "effectiveness_steps": [],
        }
        current = spec
        for attempt_idx in range(self.args.spec_max_iters):
            score = self._score_spec(problem, current)
            score_trace.append(asdict(score))
            refine_meta["raw_score_outputs"].extend(
                getattr(score, "_raw_attempt_outputs", [getattr(score, "_raw_output", "")])
            )
            refine_meta["judge_usages"].append(getattr(score, "_usage", None))
            refine_meta["judge_usages"].extend(getattr(score, "_extra_usages", []))
            refine_meta["stage_times"]["spec_score_refine"] = (
                refine_meta["stage_times"].get("spec_score_refine", 0.0)
                + float(getattr(score, "_stage_time", 0.0))
            )
            if score.overall >= self.args.spec_score_threshold:
                break
            started_at = time.perf_counter()
            refine_output, usage = self.llm.generate(
                build_spec_refine_prompt(problem, current, score),
                role="spec_refine",
                temperature=self.args.spec_temperature,
                max_tokens=self.args.spec_max_tokens,
            )
            next_spec, raw_attempt_outputs, extra_usages = self._parse_spec_output(
                refine_output,
                fallback_task=problem.question_title,
            )
            previous_spec = current
            if next_spec.parse_ok:
                current = next_spec
            current._raw_output = refine_output
            current._usage = usage
            refine_meta["raw_spec_outputs"].extend(raw_attempt_outputs)
            refine_meta["spec_usages"].append(usage)
            refine_meta["spec_usages"].extend(extra_usages)
            refine_meta["effectiveness_steps"].append(
                self._build_spec_refine_effectiveness(
                    previous_spec=previous_spec,
                    next_spec=next_spec,
                    score_before=score,
                    attempt_index=attempt_idx + 1,
                )
            )
            refine_meta["stage_times"]["spec_refine"] = (
                refine_meta["stage_times"].get("spec_refine", 0.0)
                + (time.perf_counter() - started_at)
            )
        return current, score_trace, refine_meta

    def _generate_code_baseline(self, problem: "CodeGenerationProblem") -> str:
        started_at = time.perf_counter()
        output, usage = self.llm.generate(
            build_direct_codegen_prompt(problem),
            role="codegen",
            temperature=self.args.codegen_temperature,
            max_tokens=self.args.codegen_max_tokens,
        )
        code, extra_outputs, extra_usages = self._extract_valid_code(output)
        self._last_baseline_codegen_output = output
        self._last_baseline_codegen_usage = usage
        self._last_baseline_codegen_extra_outputs = extra_outputs
        self._last_baseline_codegen_extra_usages = extra_usages
        self._last_baseline_codegen_stage_time = time.perf_counter() - started_at
        return code

    def _generate_code_from_spec(
        self, problem: "CodeGenerationProblem", spec: StructuredSpec
    ) -> str:
        started_at = time.perf_counter()
        output, usage = self.llm.generate(
            build_code_from_spec_prompt(problem, spec),
            role="codegen",
            temperature=self.args.codegen_temperature,
            max_tokens=self.args.codegen_max_tokens,
        )
        spec._last_codegen_output = output
        spec._last_codegen_usage = usage
        spec._last_codegen_stage_time = time.perf_counter() - started_at
        code, extra_outputs, extra_usages = self._extract_valid_code(output)
        spec._last_codegen_extra_outputs = extra_outputs
        spec._last_codegen_extra_usages = extra_usages
        return code

    def _repair_direct_code(
        self,
        problem: "CodeGenerationProblem",
        code: str,
        *,
        strategy: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        feedback_trace: list[dict[str, Any]] = []
        repair_effectiveness: list[dict[str, Any]] = []
        raw_repair_outputs: list[str] = []
        repair_usages: list[dict[str, int | str]] = []
        repair_stage_times: list[float] = []
        reflections: list[str] = []
        current_code = code
        stagnant_attempts = 0
        pending_step: dict[str, Any] | None = None

        for attempt_idx in range(self.args.repair_max_iters):
            feedback = self._verify(problem, current_code)
            feedback_trace.append(asdict(feedback))
            if pending_step is not None:
                repair_effectiveness.append(
                    self._finalize_repair_effectiveness(
                        pending_step,
                        after_feedback=feedback,
                    )
                )
                pending_step = None
            if feedback.passed:
                return current_code, feedback_trace, {
                    "raw_repair_outputs": raw_repair_outputs,
                    "repair_usages": repair_usages,
                    "stage_times": repair_stage_times,
                    "effectiveness_steps": repair_effectiveness,
                    "reflections": reflections,
                }

            started_at = time.perf_counter()
            if strategy == "self_refine":
                prompt = build_self_refine_repair_prompt(
                    problem,
                    current_code,
                    feedback,
                    require_change=stagnant_attempts > 0,
                )
                role = "self_refine_repair"
            else:
                reflection_output, reflection_usage = self.llm.generate(
                    build_reflexion_prompt(problem, current_code, feedback, reflections),
                    role="reflexion_reflect",
                    temperature=min(0.8, self.args.repair_temperature + 0.1 * stagnant_attempts),
                    max_tokens=min(self.args.judge_max_tokens, 400),
                )
                reflection_text = self._normalize_reflection(reflection_output)
                if reflection_text:
                    reflections.append(reflection_text)
                raw_repair_outputs.append(reflection_output)
                repair_usages.append(reflection_usage)
                prompt = build_reflexion_repair_prompt(
                    problem,
                    current_code,
                    feedback,
                    reflections,
                )
                role = "reflexion_repair"

            repair_output, usage = self.llm.generate(
                prompt,
                role=role,
                temperature=min(0.8, self.args.repair_temperature + 0.2 * stagnant_attempts),
                max_tokens=self.args.codegen_max_tokens,
            )
            raw_repair_outputs.append(repair_output)
            repair_usages.append(usage)
            repair_stage_times.append(time.perf_counter() - started_at)
            next_code, extra_outputs, extra_usages = self._extract_valid_code(repair_output)
            raw_repair_outputs.extend(extra_outputs)
            repair_usages.extend(extra_usages)
            if not next_code:
                repair_effectiveness.append(
                    self._build_repair_effectiveness(
                        attempt_index=attempt_idx + 1,
                        strategy=role,
                        before_code=current_code,
                        after_code="",
                        before_feedback=feedback,
                        effect="no_effect",
                        reason="invalid_candidate",
                    )
                )
                stagnant_attempts += 1
                continue
            if next_code == current_code:
                repair_effectiveness.append(
                    self._build_repair_effectiveness(
                        attempt_index=attempt_idx + 1,
                        strategy=role,
                        before_code=current_code,
                        after_code=next_code,
                        before_feedback=feedback,
                        effect="no_effect",
                        reason="unchanged_candidate",
                    )
                )
                stagnant_attempts += 1
                continue

            pending_step = self._build_repair_effectiveness(
                attempt_index=attempt_idx + 1,
                strategy=role,
                before_code=current_code,
                after_code=next_code,
                before_feedback=feedback,
                effect="pending",
                reason="candidate_changed",
            )
            current_code = next_code
            stagnant_attempts = 0

        final_feedback = self._verify(problem, current_code)
        feedback_trace.append(asdict(final_feedback))
        if pending_step is not None:
            repair_effectiveness.append(
                self._finalize_repair_effectiveness(
                    pending_step,
                    after_feedback=final_feedback,
                )
            )
        return current_code, feedback_trace, {
            "raw_repair_outputs": raw_repair_outputs,
            "repair_usages": repair_usages,
            "stage_times": repair_stage_times,
            "effectiveness_steps": repair_effectiveness,
            "reflections": reflections,
        }

    def _repair_code(
        self, problem: "CodeGenerationProblem", spec: StructuredSpec, code: str
    ) -> tuple[str, list[dict[str, Any]]]:
        feedback_trace: list[dict[str, Any]] = []
        repair_effectiveness: list[dict[str, Any]] = []
        current_code = code
        stagnant_attempts = 0
        pending_step: dict[str, Any] | None = None
        for attempt_idx in range(self.args.repair_max_iters):
            feedback = self._verify(problem, current_code)
            feedback_trace.append(asdict(feedback))
            if pending_step is not None:
                repair_effectiveness.append(
                    self._finalize_repair_effectiveness(
                        pending_step,
                        after_feedback=feedback,
                    )
                )
                pending_step = None
            if feedback.passed:
                spec._repair_effectiveness = repair_effectiveness
                return current_code, feedback_trace
            started_at = time.perf_counter()
            counterexample = self._build_counterexample_summary(feedback)
            use_rewrite_fallback = (
                not getattr(self.args, "disable_rewrite_repair", False)
                and feedback.error_type == "wrong_answer"
                and (stagnant_attempts > 1 or attempt_idx >= 2)
            )
            use_counterexample_fallback = (
                not getattr(self.args, "disable_counterexample_repair", False)
                and
                feedback.error_type == "wrong_answer"
                and not use_rewrite_fallback
                and (stagnant_attempts > 0 or attempt_idx > 0)
            )
            prompt = build_repair_prompt(
                problem,
                spec,
                current_code,
                feedback,
                require_change=stagnant_attempts > 0,
            )
            role = "repair"
            if use_rewrite_fallback:
                prompt = build_rewrite_from_counterexample_prompt(
                    problem,
                    spec,
                    feedback,
                    counterexample,
                )
                role = "repair_rewrite"
            elif use_counterexample_fallback:
                prompt = build_counterexample_repair_prompt(
                    problem,
                    spec,
                    current_code,
                    feedback,
                    counterexample,
                )
                role = "repair_counterexample"
            repair_output, usage = self.llm.generate(
                prompt,
                role=role,
                temperature=min(0.8, self.args.repair_temperature + 0.2 * stagnant_attempts),
                max_tokens=self.args.codegen_max_tokens,
            )
            if not hasattr(spec, "_repair_outputs"):
                spec._repair_outputs = []
            if not hasattr(spec, "_repair_usages"):
                spec._repair_usages = []
            if not hasattr(spec, "_repair_stage_times"):
                spec._repair_stage_times = []
            spec._repair_outputs.append(repair_output)
            spec._repair_usages.append(usage)
            spec._repair_stage_times.append(time.perf_counter() - started_at)
            next_code, extra_outputs, extra_usages = self._extract_valid_code(repair_output)
            spec._repair_outputs.extend(extra_outputs)
            spec._repair_usages.extend(extra_usages)
            if not next_code:
                repair_effectiveness.append(
                    self._build_repair_effectiveness(
                        attempt_index=attempt_idx + 1,
                        strategy=role,
                        before_code=current_code,
                        after_code="",
                        before_feedback=feedback,
                        effect="no_effect",
                        reason="invalid_candidate",
                    )
                )
                stagnant_attempts += 1
                continue
            if next_code == current_code:
                repair_effectiveness.append(
                    self._build_repair_effectiveness(
                        attempt_index=attempt_idx + 1,
                        strategy=role,
                        before_code=current_code,
                        after_code=next_code,
                        before_feedback=feedback,
                        effect="no_effect",
                        reason="unchanged_candidate",
                    )
                )
                stagnant_attempts += 1
                continue
            pending_step = self._build_repair_effectiveness(
                attempt_index=attempt_idx + 1,
                strategy=role,
                before_code=current_code,
                after_code=next_code,
                before_feedback=feedback,
                effect="pending",
                reason="awaiting_verifier",
            )
            current_code = next_code
            stagnant_attempts = 0

        final_feedback = self._verify(problem, current_code)
        feedback_trace.append(asdict(final_feedback))
        if pending_step is not None:
            repair_effectiveness.append(
                self._finalize_repair_effectiveness(
                    pending_step,
                    after_feedback=final_feedback,
                )
            )
        spec._repair_effectiveness = repair_effectiveness
        return current_code, feedback_trace

    @staticmethod
    def _build_counterexample_summary(feedback: VerifierFeedback) -> str:
        input_text = (feedback.input or "").strip()
        output_text = (feedback.output or "").strip()
        expected_text = (feedback.expected or "").strip()
        message = (feedback.message or "").strip()

        lines = []
        if input_text:
            lines.append("Input:")
            lines.append(input_text)
        if output_text:
            lines.append("Current output:")
            lines.append(output_text)
        if expected_text:
            lines.append("Expected output:")
            lines.append(expected_text)
        if message:
            lines.append(f"Mismatch: {message}")
        return "\n".join(lines).strip()

    def _parse_spec_output(
        self, output: str, *, fallback_task: str
    ) -> tuple[StructuredSpec, list[str], list[dict[str, int | str]]]:
        raw_attempt_outputs = [output]
        extra_usages: list[dict[str, int | str]] = []
        spec = StructuredSpec.from_llm_output(output, fallback_task=fallback_task)
        if spec.parse_ok:
            return spec, raw_attempt_outputs, extra_usages

        repair_output, repair_usage = self.llm.generate(
            build_spec_json_repair_prompt(output),
            role="spec_json_repair",
            temperature=0.0,
            max_tokens=self.args.spec_max_tokens,
        )
        raw_attempt_outputs.append(repair_output)
        extra_usages.append(repair_usage)
        repaired_spec = StructuredSpec.from_llm_output(
            repair_output,
            fallback_task=fallback_task,
        )
        if repaired_spec.parse_ok:
            return repaired_spec, raw_attempt_outputs, extra_usages
        return spec, raw_attempt_outputs, extra_usages

    def _extract_valid_code(
        self, output: str
    ) -> tuple[str, list[str], list[dict[str, int | str]]]:
        candidate = self.llm.extract_code(output)
        if self._looks_like_python(candidate):
            return candidate, [], []

        repair_output, repair_usage = self.llm.generate(
            build_code_block_repair_prompt(output),
            role="code_block_repair",
            temperature=0.0,
            max_tokens=self.args.codegen_max_tokens,
        )
        repaired_candidate = self.llm.extract_code(repair_output)
        if self._looks_like_python(repaired_candidate):
            return repaired_candidate, [repair_output], [repair_usage]
        return "", [repair_output], [repair_usage]

    @staticmethod
    def _spec_core_payload(spec: StructuredSpec) -> dict[str, Any]:
        payload = asdict(spec)
        payload.pop("parse_ok", None)
        payload.pop("parse_source", None)
        return payload

    def _build_spec_draft_effectiveness(self, spec: StructuredSpec) -> dict[str, Any]:
        payload = self._spec_core_payload(spec)
        nonempty_fields = sum(bool(value) for value in payload.values())
        return {
            "stage": "spec_draft",
            "parse_ok": spec.parse_ok,
            "nonempty_field_count": nonempty_fields,
            "effect": "produced_parseable_spec" if spec.parse_ok else "no_effect",
        }

    @staticmethod
    def _build_spec_score_effectiveness(
        score: SpecScore,
        previous_score: SpecScore | None = None,
    ) -> dict[str, Any]:
        delta_overall = None
        effect = "signal_available" if score.parse_ok else "no_effect"
        if previous_score is not None:
            delta_overall = score.overall - previous_score.overall
            if delta_overall > 0:
                effect = "improved"
            elif delta_overall < 0:
                effect = "regressed"
            else:
                effect = "no_change"
        return {
            "parse_ok": score.parse_ok,
            "overall": score.overall,
            "delta_overall": delta_overall,
            "effect": effect,
        }

    def _build_spec_refine_effectiveness(
        self,
        *,
        previous_spec: StructuredSpec,
        next_spec: StructuredSpec,
        score_before: SpecScore,
        attempt_index: int,
    ) -> dict[str, Any]:
        artifact_changed = self._spec_core_payload(previous_spec) != self._spec_core_payload(next_spec)
        if not next_spec.parse_ok:
            effect = "no_effect"
            reason = "refine_parse_failed"
        elif artifact_changed:
            effect = "artifact_changed"
            reason = "spec_updated"
        else:
            effect = "no_effect"
            reason = "spec_unchanged"
        return {
            "stage": "spec_refine",
            "attempt_index": attempt_index,
            "score_before": score_before.overall,
            "parse_ok": next_spec.parse_ok,
            "artifact_changed": artifact_changed,
            "effect": effect,
            "reason": reason,
        }

    def _build_codegen_effectiveness(
        self,
        *,
        raw_output: str,
        code: str,
        strategy: str,
    ) -> dict[str, Any]:
        code_valid = self._looks_like_python(code)
        return {
            "stage": "codegen",
            "strategy": strategy,
            "code_valid": code_valid,
            "code_changed_from_raw": raw_output.strip() != code.strip(),
            "code_hash": self._hash_text(code),
            "effect": "produced_candidate" if code_valid else "no_effect",
        }

    def _build_repair_effectiveness(
        self,
        *,
        attempt_index: int,
        strategy: str,
        before_code: str,
        after_code: str,
        before_feedback: VerifierFeedback,
        effect: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "stage": "repair",
            "attempt_index": attempt_index,
            "strategy": strategy,
            "code_changed": before_code != after_code,
            "before_code_hash": self._hash_text(before_code),
            "after_code_hash": self._hash_text(after_code) if after_code else "",
            "verifier_signature_before": self._verifier_signature(before_feedback),
            "matching_lines_before": self._matching_line_count(before_feedback),
            "effect": effect,
            "reason": reason,
        }

    def _finalize_repair_effectiveness(
        self,
        step: dict[str, Any],
        *,
        after_feedback: VerifierFeedback,
    ) -> dict[str, Any]:
        matching_lines_after = self._matching_line_count(after_feedback)
        before_matching_lines = int(step.get("matching_lines_before", 0))
        before_signature = str(step.get("verifier_signature_before", ""))
        after_signature = self._verifier_signature(after_feedback)
        if after_feedback.passed:
            effect = "solved"
            reason = "passed_verifier"
        elif matching_lines_after > before_matching_lines:
            effect = "improved"
            reason = "more_expected_lines_matched"
        elif after_signature != before_signature:
            effect = "changed_but_not_improved"
            reason = "verifier_signature_changed"
        else:
            effect = "changed_but_not_improved"
            reason = "code_changed_without_verifier_gain"
        finalized = dict(step)
        finalized["verifier_signature_after"] = after_signature
        finalized["matching_lines_after"] = matching_lines_after
        finalized["effect"] = effect
        finalized["reason"] = reason
        return finalized

    @staticmethod
    def _verifier_signature(feedback: VerifierFeedback) -> str:
        if feedback.passed:
            return "accepted"
        message = " ".join((feedback.message or "").split())
        return f"{feedback.error_type}:{message}"

    @staticmethod
    def _matching_line_count(feedback: VerifierFeedback) -> int:
        output_lines = (feedback.output or "").splitlines()
        expected_lines = (feedback.expected or "").splitlines()
        return sum(
            1 for output_line, expected_line in zip(output_lines, expected_lines) if output_line == expected_line
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        if not text:
            return ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _normalize_reflection(text: str) -> str:
        lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        return " ".join(lines[:3]).strip()

    @staticmethod
    def _looks_like_python(code: str) -> bool:
        stripped = code.strip()
        if not stripped:
            return False

        signal_tokens = (
            "def ",
            "class ",
            "import ",
            "from ",
            "for ",
            "while ",
            "if ",
            "elif ",
            "else:",
            "try:",
            "except ",
            "with ",
            "return ",
            "print(",
            "input(",
            "sys.stdin",
            "lambda ",
            "=",
        )
        bullet_prefixes = ("- ", "* ", "1. ", "2. ", "3. ")
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return False
        if sum(1 for line in lines[:5] if line.startswith(bullet_prefixes)) >= 2:
            return False
        return any(token in stripped for token in signal_tokens)

    def _verify(
        self, problem: "CodeGenerationProblem", code: str
    ) -> VerifierFeedback:
        from lcb_runner.evaluation.testing_util import run_test

        results, metadata = run_test(
            problem.get_evaluation_sample(),
            test=code,
            debug=False,
            timeout=self.args.timeout,
        )
        passed = bool(results) and all(result is True for result in results)
        if passed:
            return VerifierFeedback(
                passed=True,
                error_type="accepted",
                field="checkable_subset",
                message="Program passed the execution-based verifier.",
                raw_metadata=metadata,
            )

        metadata = metadata or {}
        error_code = metadata.get("error_code", -5)
        field = "Rules"
        error_type = "unknown_failure"
        repair_hint = "Revise the program to satisfy the structured spec and fix the failing behavior."
        if error_code == -2:
            error_type = "wrong_answer"
            field = "Rules"
            repair_hint = "Fix the core logic that computes the answer."
        elif error_code == -3:
            error_type = "time_limit_exceeded"
            field = "Algorithmic Notes"
            repair_hint = "Improve the algorithmic efficiency and avoid repeated expensive work."
        elif error_code == -4:
            error_type = "runtime_error"
            field = "Inputs"
            repair_hint = "Handle edge cases and input assumptions safely."
        elif error_code == -5:
            error_type = "verifier_error"
            field = "Non-checkable Notes"
            repair_hint = "Produce a valid Python program that can be executed by the evaluator."

        violated_spec_items = []
        if field == "Rules":
            violated_spec_items.append("Rules: output must satisfy the problem semantics")
        elif field == "Inputs":
            violated_spec_items.append("Inputs: program must handle the evaluator input correctly")
        elif field == "Algorithmic Notes":
            violated_spec_items.append("Checkable Properties: solution must run within the time limit")

        return VerifierFeedback(
            passed=False,
            error_type=error_type,
            field=field,
            message=str(metadata.get("error_message", "Execution-based verification failed.")),
            input=str(metadata.get("inputs", "")),
            output=str(metadata.get("output", "")),
            expected=str(metadata.get("expected", "")),
            violated_spec_items=violated_spec_items,
            repair_hint=repair_hint,
            raw_metadata=metadata,
        )

    def _load_benchmark(self) -> list["CodeGenerationProblem"]:
        from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

        benchmark = load_code_generation_dataset(
            self.args.release_version,
            start_date=self.args.start_date,
            end_date=self.args.end_date,
            dataset_path=getattr(self.args, "dataset_path", None),
        )
        benchmark = sorted(benchmark, key=lambda problem: problem.question_id)
        if self.args.question_ids:
            wanted = set(self.args.question_ids.split(","))
            benchmark = [problem for problem in benchmark if problem.question_id in wanted]
        if self.args.max_problems:
            benchmark = benchmark[: self.args.max_problems]
        return benchmark

    def _build_summary(
        self,
        benchmark: list["CodeGenerationProblem"],
        traces: list[ProblemTrace],
        metrics: list[Any],
    ) -> dict[str, Any]:
        average_repairs = 0.0
        if traces:
            average_repairs = sum(trace.repair_iterations for trace in traces) / len(traces)
        average_spec_score = 0.0
        all_scores = [
            score["overall"]
            for trace in traces
            for score in trace.spec_scores
            if "overall" in score
        ]
        if all_scores:
            average_spec_score = sum(all_scores) / len(all_scores)
        return {
            "model": self.args.model,
            "local_model_path": self.args.local_model_path,
            "pipeline_mode": self.args.pipeline_mode,
            "run_tag": getattr(self.args, "run_tag", None),
            "release_version": self.args.release_version,
            "num_problems": len(benchmark),
            "pass_at_1": metrics[0]["pass@1"],
            "average_repairs": average_repairs,
            "average_spec_score": average_spec_score,
            "average_initial_sas": self._average_numeric_field(
                traces, "initial_spec_score", "overall"
            ),
            "average_final_sas": self._average_numeric_field(
                traces, "final_spec_score", "overall"
            ),
            "average_delta_sas": self._average_delta_sas(traces),
            "average_initial_coverage": self._average_numeric_field(
                traces, "initial_spec_score", "coverage"
            ),
            "average_final_coverage": self._average_numeric_field(
                traces, "final_spec_score", "coverage"
            ),
            "average_initial_faithfulness": self._average_numeric_field(
                traces, "initial_spec_score", "faithfulness"
            ),
            "average_final_faithfulness": self._average_numeric_field(
                traces, "final_spec_score", "faithfulness"
            ),
            "average_initial_precision": self._average_numeric_field(
                traces, "initial_spec_score", "precision"
            ),
            "average_final_precision": self._average_numeric_field(
                traces, "final_spec_score", "precision"
            ),
            "average_llm_calls": self._average_attr(traces, "llm_calls"),
            "average_spec_calls": self._average_attr(traces, "spec_calls"),
            "average_judge_calls": self._average_attr(traces, "judge_calls"),
            "average_codegen_calls": self._average_attr(traces, "codegen_calls"),
            "average_repair_calls": self._average_attr(traces, "repair_calls"),
            "average_loop_b_iterations": self._average_attr(traces, "loop_b_iterations"),
            "average_loop_a_iterations": self._average_attr(traces, "loop_a_iterations"),
            "average_prompt_chars": self._average_token_usage(traces, "prompt_chars"),
            "average_completion_chars": self._average_token_usage(traces, "completion_chars"),
            "average_elapsed_seconds": self._average_attr(traces, "elapsed_seconds"),
            "failure_attribution_counts": dict(
                Counter(trace.failure_attribution for trace in traces)
            ),
            "failure_reason_counts": dict(
                Counter(
                    reason
                    for trace in traces
                    for reason in trace.failure_reason_tags
                )
            ),
            "verifier_error_counts": self._aggregate_verifier_error_counts(traces),
            "repair_effect_counts": dict(
                Counter(
                    step.get("effect", "unknown")
                    for trace in traces
                    for step in trace.effectiveness.get("repair_steps", [])
                )
            ),
            "spec_refine_effect_counts": dict(
                Counter(
                    step.get("effect", "unknown")
                    for trace in traces
                    for step in trace.effectiveness.get("spec_refine_steps", [])
                )
            ),
            "spec_issue_type_counts": dict(
                Counter(issue for trace in traces for issue in trace.spec_issue_types)
            ),
            "output_dir": self.output_dir,
            "spec_max_iters": self.args.spec_max_iters,
            "repair_max_iters": self.args.repair_max_iters,
            "spec_score_threshold": self.args.spec_score_threshold,
        }

    def _write_outputs(self, summary: dict[str, Any], traces: list[ProblemTrace]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        serialized_traces = [asdict(trace) for trace in traces]
        output_targets = [
            os.path.join(self.output_dir, "summary.json"),
            os.path.join(os.getcwd(), "summary.json"),
        ]
        trace_targets = [
            os.path.join(self.output_dir, "traces.json"),
            os.path.join(os.getcwd(), "traces.json"),
        ]
        for path in output_targets:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=True)
        for path in trace_targets:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serialized_traces, f, indent=2, ensure_ascii=True)

    def _compute_metrics(
        self,
        benchmark: list["CodeGenerationProblem"],
        generations: list[list[str]],
    ) -> list[Any]:
        from lcb_runner.evaluation import codegen_metrics

        return codegen_metrics(
            [problem.get_evaluation_sample() for problem in benchmark],
            generations,
            k_list=[1],
            num_process_evaluate=self.args.num_process_evaluate,
            timeout=self.args.timeout,
            debug=False,
        )

    def _finalize_trace(self, trace: ProblemTrace) -> None:
        trace.spec_issue_types = sorted(set(trace.spec_issue_types))
        self._populate_trace_counters(trace)
        trace.failure_attribution, trace.failure_reason_tags = self._attribute_failure(trace)

    def _populate_trace_counters(self, trace: ProblemTrace) -> None:
        trace.spec_calls = len(trace.raw_spec_outputs)
        trace.judge_calls = len(trace.raw_score_outputs)
        trace.codegen_calls = 1 if trace.raw_codegen_output else (1 if trace.code_initial else 0)
        trace.repair_calls = len(trace.raw_repair_outputs)
        trace.llm_calls = (
            trace.spec_calls
            + trace.judge_calls
            + trace.codegen_calls
            + trace.repair_calls
        )
        trace.token_usage = {
            "prompt_chars": int(trace.token_usage.get("prompt_chars", 0)),
            "completion_chars": int(trace.token_usage.get("completion_chars", 0)),
        }

    def _record_usage(
        self, trace: ProblemTrace, usage: dict[str, int | str] | None, role: str
    ) -> None:
        if not usage:
            return
        trace.token_usage["prompt_chars"] = int(trace.token_usage.get("prompt_chars", 0)) + int(
            usage.get("prompt_chars", 0)
        )
        trace.token_usage["completion_chars"] = int(
            trace.token_usage.get("completion_chars", 0)
        ) + int(usage.get("completion_chars", 0))

    def _record_stage_time(self, trace: ProblemTrace, stage: str, duration: float) -> None:
        if duration <= 0:
            return
        trace.stage_times[stage] = trace.stage_times.get(stage, 0.0) + float(duration)

    def _attribute_failure(self, trace: ProblemTrace) -> tuple[str, list[str]]:
        reasons: list[str] = []
        initial_score = trace.initial_spec_score or {}
        final_score = trace.final_spec_score or initial_score
        initial_sas = int(initial_score.get("overall", 0) or 0)
        final_sas = int(final_score.get("overall", 0) or 0)

        if trace.passed:
            if final_sas > initial_sas and trace.repair_iterations > 0:
                reasons.extend(["resolved_by_loop_b", "resolved_by_loop_a"])
            elif final_sas > initial_sas:
                reasons.append("resolved_by_loop_b")
            elif trace.repair_iterations > 0:
                reasons.append("resolved_by_loop_a")
            else:
                reasons.append("solved_without_loops")
            return "solved", reasons

        if trace.pipeline_mode in {"baseline", "self_refine", "reflexion"}:
            if trace.verifier_feedbacks:
                last_feedback = trace.verifier_feedbacks[-1]
                reasons.append(last_feedback.get("error_type", "unknown_failure"))
            return "implementation_induced", reasons

        if final_sas < self.args.spec_score_threshold:
            reasons.append("low_final_sas")
            if trace.spec_issue_types:
                reasons.extend(trace.spec_issue_types)
            return "spec_induced", reasons

        if trace.verifier_feedbacks:
            last_feedback = trace.verifier_feedbacks[-1]
            reasons.append(last_feedback.get("error_type", "unknown_failure"))
        return "implementation_induced", reasons

    def _average_attr(self, traces: list[ProblemTrace], field_name: str) -> float:
        if not traces:
            return 0.0
        return sum(float(getattr(trace, field_name, 0) or 0) for trace in traces) / len(traces)

    def _average_numeric_field(
        self, traces: list[ProblemTrace], dict_field: str, value_key: str
    ) -> float:
        values = []
        for trace in traces:
            payload = getattr(trace, dict_field, None) or {}
            if value_key in payload:
                values.append(float(payload[value_key]))
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _average_delta_sas(self, traces: list[ProblemTrace]) -> float:
        deltas = []
        for trace in traces:
            initial_score = trace.initial_spec_score or {}
            final_score = trace.final_spec_score or {}
            if "overall" in initial_score and "overall" in final_score:
                deltas.append(float(final_score["overall"]) - float(initial_score["overall"]))
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def _average_token_usage(self, traces: list[ProblemTrace], usage_key: str) -> float:
        if not traces:
            return 0.0
        return sum(float(trace.token_usage.get(usage_key, 0)) for trace in traces) / len(traces)

    def _aggregate_verifier_error_counts(self, traces: list[ProblemTrace]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for trace in traces:
            for feedback in trace.verifier_feedbacks:
                counter[feedback.get("error_type", "unknown")] += 1
        return dict(counter)

    def _extract_spec_issue_types(self, score: SpecScore) -> list[str]:
        issue_types: list[str] = []
        if score.missing_constraints:
            issue_types.append("missing_constraint")
        if score.unsupported_constraints:
            issue_types.append("hallucinated_constraint")
        if score.ambiguities:
            issue_types.append("ambiguity")
        return issue_types

    def _make_output_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_repr = self.llm.model.model_repr.replace("/", "_").replace(" ", "_")
        run_tag = str(getattr(self.args, "run_tag", "") or "").strip()
        if run_tag:
            run_tag = "_" + run_tag.replace("/", "_").replace(" ", "_")
        return os.path.join(
            self.args.output_root,
            f"{self.args.pipeline_mode}_{model_repr}{run_tag}_{timestamp}",
        )
