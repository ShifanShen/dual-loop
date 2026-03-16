import json
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: []
    datasets_stub.load_from_disk = lambda *args, **kwargs: []
    sys.modules["datasets"] = datasets_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = list
    numpy_stub.bool_ = bool
    numpy_stub.all = all
    numpy_stub.array = lambda x: x
    numpy_stub.mean = lambda values: sum(values) / len(values) if values else 0
    sys.modules["numpy"] = numpy_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")

    class _DummyTqdm:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *args, **kwargs):
            return None

    tqdm_stub.tqdm = _DummyTqdm
    sys.modules["tqdm"] = tqdm_stub

from lcb_runner.dual_loop.main import get_args
from lcb_runner.dual_loop.diagnostics import build_diagnostic_report, render_diagnostic_markdown
from lcb_runner.dual_loop.rq_suite import (
    apply_run_config,
    build_rq_csv_rows,
    build_rq_suite_plan,
)
from lcb_runner.dual_loop.model_download import infer_output_dir
from lcb_runner.lm_styles import LMStyle, resolve_language_model
from lcb_runner.dual_loop.pipeline import DualLoopPipeline, LLMAdapter, ProblemTrace
from lcb_runner.dual_loop.spec import SpecScore, StructuredSpec, VerifierFeedback
from lcb_runner.evaluation.testing_util import reliability_guard, restore_reliability_guard


def make_problem(question_id: str = "q1"):
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    return CodeGenerationProblem(
        question_title="Two Sum",
        question_content="Given an array nums and target, return indices of two numbers summing to target.",
        platform="leetcode",
        question_id=question_id,
        contest_id="c1",
        contest_date="2024-01-01T00:00:00",
        starter_code="",
        difficulty="easy",
        public_test_cases='[{"input": "[2,7,11,15]\\n9", "output": "0 1", "testtype": "stdin"}]',
        private_test_cases='[{"input": "[3,2,4]\\n6", "output": "1 2", "testtype": "stdin"}]',
        metadata="{}",
    )


class SpecParsingTests(unittest.TestCase):
    def test_infer_output_dir(self):
        path = infer_output_dir("Qwen/Qwen2.5-Coder-7B-Instruct", "/models")
        self.assertTrue(path.endswith("Qwen2.5-Coder-7B-Instruct"))

    def test_main_infers_model_name_from_local_path(self):
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--local_model_path",
                "/models/Qwen2.5-Coder-7B-Instruct",
                "--model_style",
                "CodeQwenInstruct",
                "--run_tag",
                "smoke",
            ],
        ):
            args = get_args()
        self.assertEqual(args.model, "Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(args.model_repr, "Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(args.run_tag, "smoke")

    def test_resolve_arbitrary_local_model(self):
        model = resolve_language_model(
            "Local-Qwen",
            local_model_path="/models/Qwen2.5-Coder-7B-Instruct",
            model_style_override="CodeQwenInstruct",
            model_repr_override="Qwen2.5-Coder-7B-Local",
        )
        self.assertEqual(model.model_name, "Local-Qwen")
        self.assertEqual(model.model_repr, "Qwen2.5-Coder-7B-Local")

    def test_structured_spec_parses_json_block(self):
        text = """```json
        {
          "task": "solve the task",
          "inputs": ["nums: list[int]"],
          "outputs": ["pair of indices"],
          "constraints": ["n >= 2"],
          "rules": ["answer must be valid"],
          "edge_cases": ["duplicates"],
          "checkable_properties": ["indices in range"],
          "tie_break": ["lexicographically smallest"],
          "reference_strategy": "reference_solver"
        }
        ```"""
        spec = StructuredSpec.from_llm_output(text)
        self.assertEqual(spec.task, "solve the task")
        self.assertEqual(spec.reference_strategy, "reference_solver")
        self.assertIn("duplicates", spec.edge_cases)

    def test_spec_score_parses_fields(self):
        text = json.dumps(
            {
                "coverage": 90,
                "faithfulness": 80,
                "precision": 70,
                "overall": 82,
                "missing_constraints": ["missing edge case"],
                "unsupported_constraints": ["made-up complexity"],
                "ambiguities": ["unclear output rule"],
                "action": "revise the spec",
            }
        )
        score = SpecScore.from_llm_output(text)
        self.assertEqual(score.overall, 82)
        self.assertIn("missing edge case", score.missing_constraints)
        self.assertEqual(score.action, "revise the spec")

    def test_spec_score_parses_freeform_fields(self):
        text = """
        Here is the evaluation.
        coverage: 84
        faithfulness: 91
        precision: 76
        overall: 85
        missing_constraints: ["edge case: empty array"]
        unsupported_constraints: []
        ambiguities: ['tie-break not specified']
        action: revise output rule and add edge cases
        """
        score = SpecScore.from_llm_output(text)
        self.assertEqual(score.coverage, 84)
        self.assertEqual(score.faithfulness, 91)
        self.assertEqual(score.precision, 76)
        self.assertEqual(score.overall, 85)
        self.assertIn("edge case: empty array", score.missing_constraints)
        self.assertIn("tie-break not specified", score.ambiguities)
        self.assertEqual(score.action, "revise output rule and add edge cases")
        self.assertTrue(score.parse_ok)
        self.assertEqual(score.parse_source, "fallback")

    def test_spec_score_parses_chinese_and_nested_scores(self):
        text = json.dumps(
            {
                "scores": {
                    "覆盖率": "88/100",
                    "忠实度": 93,
                    "精确度": 79,
                    "总分": 88,
                },
                "缺失约束": ["边界情况未覆盖"],
                "歧义": [],
                "建议": "补充边界情况",
            },
            ensure_ascii=False,
        )
        score = SpecScore.from_llm_output(text)
        self.assertEqual(score.coverage, 88)
        self.assertEqual(score.faithfulness, 93)
        self.assertEqual(score.precision, 79)
        self.assertEqual(score.overall, 88)
        self.assertIn("边界情况未覆盖", score.missing_constraints)
        self.assertEqual(score.action, "补充边界情况")
        self.assertTrue(score.parse_ok)
        self.assertEqual(score.parse_source, "json")

    def test_reliability_guard_restores_process_state(self):
        import os
        import shutil

        original_putenv = os.putenv
        original_kill = os.kill
        original_rmtree = shutil.rmtree

        state = reliability_guard()
        try:
            self.assertIsNone(os.putenv)
            self.assertIsNone(os.kill)
            self.assertIsNone(shutil.rmtree)
        finally:
            restore_reliability_guard(state)

        self.assertIs(os.putenv, original_putenv)
        self.assertIs(os.kill, original_kill)
        self.assertIs(shutil.rmtree, original_rmtree)

    @patch("lcb_runner.dual_loop.pipeline.build_runner")
    @patch("lcb_runner.dual_loop.pipeline.resolve_language_model")
    def test_llm_adapter_formats_codeqwen_prompts(self, mock_resolve_model, mock_build_runner):
        args = Namespace(
            model="Local-Qwen",
            local_model_path=None,
            model_style=None,
            model_repr=None,
            temperature=0.2,
            max_tokens=128,
            top_p=0.95,
            n=1,
            stop=["###"],
        )
        mock_resolve_model.return_value = types.SimpleNamespace(
            model_style=LMStyle.CodeQwenInstruct,
            model_repr="fake-model",
        )
        mock_runner = mock_build_runner.return_value
        mock_runner.run_batch.return_value = [["ok"]]

        adapter = LLMAdapter(args)
        adapter.generate("hello")

        sent_prompt = mock_runner.run_batch.call_args.args[0][0]
        self.assertIn("<|im_start|>user", sent_prompt)
        self.assertIn("hello", sent_prompt)


class DualLoopPipelineTests(unittest.TestCase):
    def make_args(self, output_root: str) -> Namespace:
        return Namespace(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            local_model_path=None,
            model_style=None,
            model_repr=None,
            pipeline_mode="full",
            release_version="release_v6",
            start_date=None,
            end_date=None,
            question_ids=None,
            max_problems=1,
            output_root=output_root,
            cwd_output_dir=None,
            trust_remote_code=False,
            dtype="bfloat16",
            tensor_parallel_size=1,
            enable_prefix_caching=False,
            use_cache=False,
            cache_batch_size=32,
            n=1,
            temperature=0.2,
            top_p=0.95,
            max_tokens=2048,
            stop=["###"],
            multiprocess=0,
            timeout=2,
            num_process_evaluate=1,
            spec_max_iters=2,
            repair_max_iters=2,
            spec_score_threshold=80,
            disable_counterexample_repair=False,
            disable_rewrite_repair=False,
            spec_temperature=0.0,
            judge_temperature=0.0,
            codegen_temperature=0.2,
            repair_temperature=0.1,
            spec_max_tokens=512,
            judge_max_tokens=512,
            codegen_max_tokens=512,
        )

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_run_problem_full_uses_spec_and_repair(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            mock_adapter_cls.return_value.model.model_repr = "fake-model"
            pipeline = DualLoopPipeline(args)
            problem = make_problem()

            spec = StructuredSpec(task="solve", rules=["valid output"])
            score = SpecScore(coverage=90, faithfulness=90, precision=90, overall=90)
            final_feedback = VerifierFeedback(
                passed=True,
                error_type="accepted",
                field="checkable_subset",
                message="ok",
            )

            with patch.object(pipeline, "_draft_spec", return_value=spec), patch.object(
                pipeline, "_score_spec", return_value=score
            ), patch.object(
                pipeline,
                "_refine_spec",
                return_value=(
                    spec,
                    [{"overall": 90, "coverage": 90, "faithfulness": 90, "precision": 90}],
                    {
                        "raw_score_outputs": [],
                        "raw_spec_outputs": [],
                        "judge_usages": [],
                        "spec_usages": [],
                        "stage_times": {},
                    },
                ),
            ), patch.object(
                pipeline, "_generate_code_from_spec", return_value="print('hi')"
            ), patch.object(
                pipeline,
                "_repair_code",
                return_value=(
                    "print('fixed')",
                    [
                        {
                            "passed": False,
                            "error_type": "wrong_answer",
                            "field": "Rules",
                            "message": "bad",
                        },
                        {
                            "passed": True,
                            "error_type": "accepted",
                            "field": "checkable_subset",
                            "message": "ok",
                        },
                    ],
                ),
            ):
                trace = pipeline._run_problem(problem)

            self.assertEqual(trace.final_code, "print('fixed')")
            self.assertTrue(trace.passed)
            self.assertEqual(trace.repair_iterations, 1)
            self.assertEqual(trace.spec_final["task"], "solve")
            self.assertEqual(trace.failure_attribution, "solved")
            self.assertIn("spec_draft", trace.effectiveness)
            self.assertIn("codegen", trace.effectiveness)

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_run_problem_self_refine_uses_direct_repair_branch(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            args.pipeline_mode = "self_refine"
            mock_adapter_cls.return_value.model.model_repr = "fake-model"
            pipeline = DualLoopPipeline(args)
            problem = make_problem()

            with patch.object(pipeline, "_generate_code_baseline", return_value="print('x')"), patch.object(
                pipeline,
                "_repair_direct_code",
                return_value=(
                    "print('fixed')",
                    [
                        {"passed": False, "error_type": "wrong_answer"},
                        {"passed": True, "error_type": "accepted"},
                    ],
                    {
                        "raw_repair_outputs": ["```python\nprint('fixed')\n```"],
                        "repair_usages": [],
                        "stage_times": [0.2],
                        "effectiveness_steps": [
                            {"strategy": "self_refine_repair", "effect": "solved"}
                        ],
                        "reflections": [],
                    },
                ),
            ):
                trace = pipeline._run_problem(problem)

            self.assertEqual(trace.final_code, "print('fixed')")
            self.assertTrue(trace.passed)
            self.assertEqual(trace.failure_attribution, "solved")
            self.assertEqual(trace.effectiveness["repair_steps"][0]["strategy"], "self_refine_repair")

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_run_problem_reflexion_records_reflections(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            args.pipeline_mode = "reflexion"
            mock_adapter_cls.return_value.model.model_repr = "fake-model"
            pipeline = DualLoopPipeline(args)
            problem = make_problem()

            with patch.object(pipeline, "_generate_code_baseline", return_value="print('x')"), patch.object(
                pipeline,
                "_repair_direct_code",
                return_value=(
                    "print('fixed')",
                    [
                        {"passed": False, "error_type": "wrong_answer"},
                        {"passed": True, "error_type": "accepted"},
                    ],
                    {
                        "raw_repair_outputs": ["reflection", "```python\nprint('fixed')\n```"],
                        "repair_usages": [],
                        "stage_times": [0.3],
                        "effectiveness_steps": [
                            {"strategy": "reflexion_repair", "effect": "solved"}
                        ],
                        "reflections": ["Need to fix the condition."],
                    },
                ),
            ):
                trace = pipeline._run_problem(problem)

            self.assertEqual(trace.final_code, "print('fixed')")
            self.assertIn("reflections", trace.effectiveness)
            self.assertEqual(trace.effectiveness["repair_steps"][0]["strategy"], "reflexion_repair")

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_verify_maps_subprocess_failure_to_verifier_error(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            mock_adapter_cls.return_value.model.model_repr = "fake-model"
            pipeline = DualLoopPipeline(args)
            problem = make_problem()

            with patch(
                "lcb_runner.evaluation.compute_code_generation_metrics.check_correctness",
                side_effect=RuntimeError("worker crashed"),
            ):
                feedback = pipeline._verify(problem, "print('x')")

            self.assertFalse(feedback.passed)
            self.assertEqual(feedback.error_type, "verifier_error")
            self.assertIn("Verifier subprocess failed", feedback.message)

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_run_writes_summary_and_traces(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            mock_adapter_cls.return_value.model.model_repr = "fake-model"
            pipeline = DualLoopPipeline(args)
            problem = make_problem()

            fake_trace = ProblemTrace(
                question_id="q1",
                question_title="Two Sum",
                pipeline_mode="full",
                raw_problem="problem",
                spec_initial={"task": "solve"},
                spec_final={"task": "solve"},
                initial_spec_score={"overall": 75, "coverage": 70, "faithfulness": 80, "precision": 75},
                final_spec_score={"overall": 90, "coverage": 90, "faithfulness": 90, "precision": 90},
                spec_scores=[{"overall": 90}],
                code_initial="print(1)",
                final_code="print(1)",
                verifier_feedbacks=[{"passed": True}],
                failure_attribution="solved",
                passed=True,
                repair_iterations=0,
                elapsed_seconds=0.1,
            )
            with patch.object(pipeline, "_load_benchmark", return_value=[problem]), patch.object(
                pipeline, "_run_problem", return_value=fake_trace
            ), patch.object(
                pipeline, "_compute_metrics", return_value=[{"pass@1": 1.0}, {}, []]
            ), patch("lcb_runner.dual_loop.pipeline.os.getcwd", return_value=tmpdir):
                summary = pipeline.run()

            self.assertEqual(summary["pass_at_1"], 1.0)
            self.assertIn("failure_attribution_counts", summary)
            self.assertIn("average_initial_sas", summary)
            self.assertIn("repair_effect_counts", summary)
            self.assertTrue(os.path.exists(os.path.join(pipeline.output_dir, "summary.json")))
            self.assertTrue(os.path.exists(os.path.join(pipeline.output_dir, "traces.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "summary.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "traces.json")))

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_refine_spec_keeps_last_valid_spec_when_refine_parse_fails(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            args.repair_max_iters = 3
            mock_adapter = mock_adapter_cls.return_value
            mock_adapter.model.model_repr = "fake-model"
            mock_adapter.generate.side_effect = [
                ("- malformed spec", {"prompt_chars": 1, "completion_chars": 1}),
                ("still not json", {"prompt_chars": 1, "completion_chars": 1}),
            ]
            pipeline = DualLoopPipeline(args)
            problem = make_problem()
            spec = StructuredSpec(task="solve", rules=["valid output"])

            with patch.object(
                pipeline,
                "_score_spec",
                side_effect=[
                    SpecScore(coverage=20, faithfulness=20, precision=20, overall=20),
                    SpecScore(coverage=90, faithfulness=90, precision=90, overall=90),
                ],
            ):
                refined, score_trace, refine_meta = pipeline._refine_spec(problem, spec)

            self.assertEqual(refined.task, "solve")
            self.assertEqual(refined.rules, ["valid output"])
            self.assertEqual(len(score_trace), 2)
            self.assertEqual(len(refine_meta["raw_spec_outputs"]), 2)

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_repair_code_keeps_current_code_when_model_returns_non_code(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            mock_adapter = mock_adapter_cls.return_value
            mock_adapter.model.model_repr = "fake-model"
            mock_adapter.generate.side_effect = [
                ("- not code", {"prompt_chars": 1, "completion_chars": 1}),
                ("```python\n\n```", {"prompt_chars": 1, "completion_chars": 1}),
                ("- still not code", {"prompt_chars": 1, "completion_chars": 1}),
                ("```python\n\n```", {"prompt_chars": 1, "completion_chars": 1}),
            ]
            mock_adapter.extract_code.side_effect = lambda output: output
            pipeline = DualLoopPipeline(args)
            problem = make_problem()
            spec = StructuredSpec(task="solve")
            failed_feedback = VerifierFeedback(
                passed=False,
                error_type="wrong_answer",
                field="Rules",
                message="bad answer",
            )

            with patch.object(
                pipeline,
                "_verify",
                return_value=failed_feedback,
            ):
                final_code, feedback_trace = pipeline._repair_code(
                    problem,
                    spec,
                    "print('keep')",
                )

            self.assertEqual(final_code, "print('keep')")
            self.assertEqual(len(feedback_trace), 3)
            self.assertEqual(len(spec._repair_outputs), 4)
            self.assertEqual(
                [step["effect"] for step in spec._repair_effectiveness],
                ["no_effect", "no_effect"],
            )

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_repair_code_retries_when_model_returns_unchanged_program(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            mock_adapter = mock_adapter_cls.return_value
            mock_adapter.model.model_repr = "fake-model"
            mock_adapter.generate.side_effect = [
                ("```python\nprint('same')\n```", {"prompt_chars": 1, "completion_chars": 1}),
                ("```python\nprint('fixed')\n```", {"prompt_chars": 1, "completion_chars": 1}),
            ]
            mock_adapter.extract_code.side_effect = lambda output: output.replace("```python", "").replace("```", "").strip()
            pipeline = DualLoopPipeline(args)
            problem = make_problem()
            spec = StructuredSpec(task="solve")
            feedbacks = [
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="bad answer",
                ),
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="still bad",
                ),
                VerifierFeedback(
                    passed=True,
                    error_type="accepted",
                    field="checkable_subset",
                    message="ok",
                ),
            ]

            with patch.object(
                pipeline,
                "_verify",
                side_effect=feedbacks,
            ):
                final_code, feedback_trace = pipeline._repair_code(
                    problem,
                    spec,
                    "print('same')",
                )

            self.assertEqual(final_code, "print('fixed')")
            self.assertEqual(len(feedback_trace), 3)
            self.assertEqual(
                [step["effect"] for step in spec._repair_effectiveness],
                ["no_effect", "solved"],
            )

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_repair_code_uses_counterexample_prompt_after_repeated_wrong_answers(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            mock_adapter = mock_adapter_cls.return_value
            mock_adapter.model.model_repr = "fake-model"
            mock_adapter.generate.side_effect = [
                ("```python\nprint('same')\n```", {"prompt_chars": 1, "completion_chars": 1}),
                ("```python\nprint('fixed')\n```", {"prompt_chars": 1, "completion_chars": 1}),
            ]
            mock_adapter.extract_code.side_effect = (
                lambda output: output.replace("```python", "").replace("```", "").strip()
            )
            pipeline = DualLoopPipeline(args)
            problem = make_problem()
            spec = StructuredSpec(task="solve")
            feedbacks = [
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="Wrong answer at output_line_idx=0: 1 != 2",
                    input="1\n",
                    output="1\n",
                    expected="2\n",
                ),
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="Wrong answer at output_line_idx=0: 1 != 2",
                    input="1\n",
                    output="1\n",
                    expected="2\n",
                ),
                VerifierFeedback(
                    passed=True,
                    error_type="accepted",
                    field="checkable_subset",
                    message="ok",
                ),
            ]

            with patch.object(pipeline, "_verify", side_effect=feedbacks):
                final_code, feedback_trace = pipeline._repair_code(problem, spec, "print('same')")

            self.assertEqual(final_code, "print('fixed')")
            self.assertEqual(len(feedback_trace), 3)
            first_prompt = mock_adapter.generate.call_args_list[0].kwargs.get("prompt")
            second_prompt = mock_adapter.generate.call_args_list[1].kwargs.get("prompt")
            self.assertIsNone(first_prompt)
            self.assertIsNone(second_prompt)
            first_prompt = mock_adapter.generate.call_args_list[0].args[0]
            second_prompt = mock_adapter.generate.call_args_list[1].args[0]
            self.assertIn("Verifier feedback", first_prompt)
            self.assertIn("Counterexample:", second_prompt)
            self.assertIn("Expected output:", second_prompt)

    @patch("lcb_runner.dual_loop.pipeline.LLMAdapter")
    def test_repair_code_uses_rewrite_prompt_after_multiple_wrong_answers(self, mock_adapter_cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            args.repair_max_iters = 3
            mock_adapter = mock_adapter_cls.return_value
            mock_adapter.model.model_repr = "fake-model"
            mock_adapter.generate.side_effect = [
                ("```python\nprint('same')\n```", {"prompt_chars": 1, "completion_chars": 1}),
                ("```python\nprint('same')\n```", {"prompt_chars": 1, "completion_chars": 1}),
                ("```python\nprint('fixed')\n```", {"prompt_chars": 1, "completion_chars": 1}),
            ]
            mock_adapter.extract_code.side_effect = (
                lambda output: output.replace("```python", "").replace("```", "").strip()
            )
            pipeline = DualLoopPipeline(args)
            problem = make_problem()
            spec = StructuredSpec(task="solve")
            feedbacks = [
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="Wrong answer at output_line_idx=0: 1 != 2",
                    input="1\n",
                    output="1\n",
                    expected="2\n",
                ),
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="Wrong answer at output_line_idx=0: 1 != 2",
                    input="1\n",
                    output="1\n",
                    expected="2\n",
                ),
                VerifierFeedback(
                    passed=False,
                    error_type="wrong_answer",
                    field="Rules",
                    message="Wrong answer at output_line_idx=0: 1 != 2",
                    input="1\n",
                    output="1\n",
                    expected="2\n",
                ),
                VerifierFeedback(
                    passed=True,
                    error_type="accepted",
                    field="checkable_subset",
                    message="ok",
                ),
            ]

            with patch.object(pipeline, "_verify", side_effect=feedbacks):
                final_code, feedback_trace = pipeline._repair_code(problem, spec, "print('same')")

            self.assertEqual(final_code, "print('fixed')")
            self.assertEqual(len(feedback_trace), 4)
            third_prompt = mock_adapter.generate.call_args_list[2].args[0]
            self.assertIn("write a fresh solution from scratch", third_prompt)
            self.assertIn("Counterexample:", third_prompt)

    def test_diagnostic_report_aggregates_repair_effectiveness(self):
        summary = {
            "model": "fake-model",
            "pipeline_mode": "full",
            "num_problems": 1,
            "pass_at_1": 1.0,
            "average_initial_sas": 80.0,
            "average_final_sas": 82.0,
            "average_repairs": 2.0,
            "output_dir": "output/test",
            "failure_attribution_counts": {"solved": 1},
            "verifier_error_counts": {"wrong_answer": 1, "accepted": 1},
            "repair_effect_counts": {"no_effect": 1, "solved": 1},
            "spec_refine_effect_counts": {},
        }
        traces = [
            {
                "question_id": "q1",
                "effectiveness": {
                    "spec_draft": {"effect": "produced_parseable_spec"},
                    "spec_score_initial": {"effect": "signal_available"},
                    "spec_score_final": {"effect": "improved"},
                    "codegen": {"effect": "produced_candidate"},
                    "repair_steps": [
                        {
                            "attempt_index": 1,
                            "strategy": "repair",
                            "effect": "no_effect",
                            "reason": "unchanged_candidate",
                            "matching_lines_before": 1,
                            "matching_lines_after": None,
                            "verifier_signature_before": "wrong_answer:a",
                            "verifier_signature_after": "",
                        },
                        {
                            "attempt_index": 2,
                            "strategy": "repair_rewrite",
                            "effect": "solved",
                            "reason": "passed_verifier",
                            "matching_lines_before": 1,
                            "matching_lines_after": 2,
                            "verifier_signature_before": "wrong_answer:a",
                            "verifier_signature_after": "accepted",
                        },
                    ],
                },
            }
        ]

        report = build_diagnostic_report(summary, traces)
        markdown = render_diagnostic_markdown(report)

        self.assertEqual(report["repair_strategy_counts"]["repair"], 1)
        self.assertEqual(report["repair_strategy_counts"]["repair_rewrite"], 1)
        self.assertEqual(report["repair_strategy_effect_counts"]["repair"]["no_effect"], 1)
        self.assertEqual(report["repair_strategy_effect_counts"]["repair_rewrite"]["solved"], 1)
        self.assertTrue(report["consistency_checks"]["repair_effect_counts_match_summary"])
        self.assertIn("repair_rewrite", markdown)

    def test_rq_suite_plan_includes_core_runs(self):
        plan = build_rq_suite_plan()
        run_names = [item.run_name for item in plan]
        self.assertEqual(
            run_names,
            [
                "baseline_direct",
                "decomposition_only",
                "self_refine_style",
                "reflexion_style",
                "full_dual_loop",
            ],
        )

    def test_apply_run_config_overrides_mode_and_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.make_args(tmpdir)
            config = build_rq_suite_plan(include_repair_ablations=True)[5]
            configured = apply_run_config(args, config)
            self.assertEqual(configured.pipeline_mode, "full")
            self.assertEqual(configured.run_tag, "full_no_counterexample")
            self.assertTrue(configured.disable_counterexample_repair)
            self.assertFalse(configured.disable_rewrite_repair)

    def test_build_rq_csv_rows_computes_rq_metrics(self):
        run_results = [
            {
                "config": {
                    "suite_name": "rq_core",
                    "run_name": "baseline_direct",
                    "pipeline_mode": "baseline",
                    "disable_counterexample_repair": False,
                    "disable_rewrite_repair": False,
                    "spec_max_iters": None,
                    "repair_max_iters": None,
                    "spec_score_threshold": None,
                },
                "summary": {
                    "model": "fake-model",
                    "pipeline_mode": "baseline",
                    "release_version": "release_v6",
                    "run_tag": "baseline_direct",
                    "num_problems": 2,
                    "pass_at_1": 0.25,
                    "average_initial_sas": 0.0,
                    "average_final_sas": 0.0,
                    "average_delta_sas": 0.0,
                    "average_initial_coverage": 0.0,
                    "average_final_coverage": 0.0,
                    "average_initial_faithfulness": 0.0,
                    "average_final_faithfulness": 0.0,
                    "average_initial_precision": 0.0,
                    "average_final_precision": 0.0,
                    "average_llm_calls": 1.0,
                    "average_spec_calls": 0.0,
                    "average_judge_calls": 0.0,
                    "average_codegen_calls": 1.0,
                    "average_repair_calls": 0.0,
                    "average_loop_b_iterations": 0.0,
                    "average_loop_a_iterations": 0.0,
                    "average_repairs": 0.0,
                    "average_prompt_chars": 100.0,
                    "average_completion_chars": 40.0,
                    "average_elapsed_seconds": 1.0,
                    "failure_attribution_counts": {"solved": 1, "implementation_induced": 1},
                    "verifier_error_counts": {"accepted": 1, "wrong_answer": 1},
                    "output_dir": "output/baseline",
                    "spec_max_iters": 3,
                    "repair_max_iters": 3,
                },
                "traces": [
                    {
                        "initial_spec_score": {},
                        "final_spec_score": {},
                        "effectiveness": {},
                        "stage_times": {"codegen": 1.5},
                    },
                    {
                        "initial_spec_score": {},
                        "final_spec_score": {},
                        "effectiveness": {},
                        "stage_times": {"codegen": 0.5},
                    },
                ],
            },
            {
                "config": {
                    "suite_name": "rq_core",
                    "run_name": "decomposition_only",
                    "pipeline_mode": "decomposition",
                    "disable_counterexample_repair": False,
                    "disable_rewrite_repair": False,
                    "spec_max_iters": None,
                    "repair_max_iters": None,
                    "spec_score_threshold": None,
                },
                "summary": {
                    "model": "fake-model",
                    "pipeline_mode": "decomposition",
                    "release_version": "release_v6",
                    "run_tag": "decomposition_only",
                    "num_problems": 2,
                    "pass_at_1": 0.4,
                    "average_initial_sas": 81.0,
                    "average_final_sas": 81.0,
                    "average_delta_sas": 0.0,
                    "average_initial_coverage": 80.0,
                    "average_final_coverage": 80.0,
                    "average_initial_faithfulness": 82.0,
                    "average_final_faithfulness": 82.0,
                    "average_initial_precision": 81.0,
                    "average_final_precision": 81.0,
                    "average_llm_calls": 3.0,
                    "average_spec_calls": 1.0,
                    "average_judge_calls": 1.0,
                    "average_codegen_calls": 1.0,
                    "average_repair_calls": 0.0,
                    "average_loop_b_iterations": 0.0,
                    "average_loop_a_iterations": 0.0,
                    "average_repairs": 0.0,
                    "average_prompt_chars": 300.0,
                    "average_completion_chars": 90.0,
                    "average_elapsed_seconds": 2.0,
                    "failure_attribution_counts": {"solved": 1, "implementation_induced": 1},
                    "verifier_error_counts": {"accepted": 1, "wrong_answer": 1},
                    "output_dir": "output/decomposition",
                    "spec_max_iters": 3,
                    "repair_max_iters": 3,
                },
                "traces": [
                    {
                        "initial_spec_score": {"overall": 80},
                        "final_spec_score": {"overall": 80},
                        "effectiveness": {},
                        "stage_times": {"spec_draft": 0.8, "spec_score_initial": 0.4, "codegen": 0.6},
                    },
                    {
                        "initial_spec_score": {"overall": 82},
                        "final_spec_score": {"overall": 82},
                        "effectiveness": {},
                        "stage_times": {"spec_draft": 1.0, "spec_score_initial": 0.6, "codegen": 0.8},
                    },
                ],
            },
            {
                "config": {
                    "suite_name": "rq_core",
                    "run_name": "self_refine_style",
                    "pipeline_mode": "self_refine",
                    "disable_counterexample_repair": False,
                    "disable_rewrite_repair": False,
                    "spec_max_iters": None,
                    "repair_max_iters": None,
                    "spec_score_threshold": None,
                },
                "summary": {
                    "model": "fake-model",
                    "pipeline_mode": "self_refine",
                    "release_version": "release_v6",
                    "run_tag": "self_refine_style",
                    "num_problems": 2,
                    "pass_at_1": 0.55,
                    "average_initial_sas": 80.0,
                    "average_final_sas": 80.0,
                    "average_delta_sas": 0.0,
                    "average_initial_coverage": 80.0,
                    "average_final_coverage": 80.0,
                    "average_initial_faithfulness": 80.0,
                    "average_final_faithfulness": 80.0,
                    "average_initial_precision": 80.0,
                    "average_final_precision": 80.0,
                    "average_llm_calls": 5.0,
                    "average_spec_calls": 1.0,
                    "average_judge_calls": 2.0,
                    "average_codegen_calls": 1.0,
                    "average_repair_calls": 1.0,
                    "average_loop_b_iterations": 0.0,
                    "average_loop_a_iterations": 1.0,
                    "average_repairs": 1.0,
                    "average_prompt_chars": 420.0,
                    "average_completion_chars": 120.0,
                    "average_elapsed_seconds": 3.0,
                    "failure_attribution_counts": {"solved": 1, "implementation_induced": 1},
                    "verifier_error_counts": {"accepted": 1, "wrong_answer": 2},
                    "output_dir": "output/self_refine",
                    "spec_max_iters": 3,
                    "repair_max_iters": 3,
                },
                "traces": [
                    {
                        "initial_spec_score": {"overall": 80},
                        "final_spec_score": {"overall": 80},
                        "effectiveness": {
                            "repair_steps": [
                                {"strategy": "repair", "effect": "no_effect"},
                                {"strategy": "repair_rewrite", "effect": "solved"},
                            ]
                        },
                        "stage_times": {"repair": 1.2},
                    },
                    {
                        "initial_spec_score": {"overall": 80},
                        "final_spec_score": {"overall": 80},
                        "effectiveness": {
                            "repair_steps": [
                                {"strategy": "repair", "effect": "improved"},
                            ]
                        },
                        "stage_times": {"repair": 0.8},
                    },
                ],
            },
            {
                "config": {
                    "suite_name": "rq_core",
                    "run_name": "reflexion_style",
                    "pipeline_mode": "reflexion",
                    "disable_counterexample_repair": False,
                    "disable_rewrite_repair": False,
                    "spec_max_iters": None,
                    "repair_max_iters": None,
                    "spec_score_threshold": None,
                },
                "summary": {
                    "model": "fake-model",
                    "pipeline_mode": "reflexion",
                    "release_version": "release_v6",
                    "run_tag": "reflexion_style",
                    "num_problems": 2,
                    "pass_at_1": 0.5,
                    "average_initial_sas": 0.0,
                    "average_final_sas": 0.0,
                    "average_delta_sas": 0.0,
                    "average_initial_coverage": 0.0,
                    "average_final_coverage": 0.0,
                    "average_initial_faithfulness": 0.0,
                    "average_final_faithfulness": 0.0,
                    "average_initial_precision": 0.0,
                    "average_final_precision": 0.0,
                    "average_llm_calls": 6.0,
                    "average_spec_calls": 0.0,
                    "average_judge_calls": 0.0,
                    "average_codegen_calls": 1.0,
                    "average_repair_calls": 2.0,
                    "average_loop_b_iterations": 0.0,
                    "average_loop_a_iterations": 2.0,
                    "average_repairs": 2.0,
                    "average_prompt_chars": 500.0,
                    "average_completion_chars": 150.0,
                    "average_elapsed_seconds": 3.5,
                    "failure_attribution_counts": {"solved": 1, "implementation_induced": 1},
                    "verifier_error_counts": {"accepted": 1, "wrong_answer": 2},
                    "output_dir": "output/reflexion",
                    "spec_max_iters": 3,
                    "repair_max_iters": 3,
                },
                "traces": [
                    {
                        "initial_spec_score": {},
                        "final_spec_score": {},
                        "effectiveness": {
                            "repair_steps": [
                                {"strategy": "reflexion_repair", "effect": "improved"}
                            ]
                        },
                        "stage_times": {"repair": 1.1},
                    },
                    {
                        "initial_spec_score": {},
                        "final_spec_score": {},
                        "effectiveness": {
                            "repair_steps": [
                                {"strategy": "reflexion_repair", "effect": "solved"},
                                {"strategy": "reflexion_repair", "effect": "no_effect"},
                            ]
                        },
                        "stage_times": {"repair": 1.3},
                    },
                ],
            },
            {
                "config": {
                    "suite_name": "rq_core",
                    "run_name": "full_dual_loop",
                    "pipeline_mode": "full",
                    "disable_counterexample_repair": False,
                    "disable_rewrite_repair": False,
                    "spec_max_iters": None,
                    "repair_max_iters": None,
                    "spec_score_threshold": None,
                },
                "summary": {
                    "model": "fake-model",
                    "pipeline_mode": "full",
                    "release_version": "release_v6",
                    "run_tag": "full_dual_loop",
                    "num_problems": 2,
                    "pass_at_1": 0.7,
                    "average_initial_sas": 76.0,
                    "average_final_sas": 86.0,
                    "average_delta_sas": 10.0,
                    "average_initial_coverage": 75.0,
                    "average_final_coverage": 85.0,
                    "average_initial_faithfulness": 76.0,
                    "average_final_faithfulness": 87.0,
                    "average_initial_precision": 77.0,
                    "average_final_precision": 86.0,
                    "average_llm_calls": 8.0,
                    "average_spec_calls": 2.0,
                    "average_judge_calls": 3.0,
                    "average_codegen_calls": 1.0,
                    "average_repair_calls": 2.0,
                    "average_loop_b_iterations": 2.0,
                    "average_loop_a_iterations": 2.0,
                    "average_repairs": 2.0,
                    "average_prompt_chars": 650.0,
                    "average_completion_chars": 200.0,
                    "average_elapsed_seconds": 4.0,
                    "failure_attribution_counts": {"solved": 2},
                    "verifier_error_counts": {"accepted": 2, "wrong_answer": 2},
                    "output_dir": "output/full",
                    "spec_max_iters": 3,
                    "repair_max_iters": 3,
                },
                "traces": [
                    {
                        "initial_spec_score": {"overall": 72},
                        "final_spec_score": {"overall": 85},
                        "effectiveness": {
                            "spec_refine_steps": [
                                {"effect": "artifact_changed", "reason": "spec_updated"}
                            ],
                            "repair_steps": [
                                {"strategy": "repair", "effect": "no_effect"},
                                {"strategy": "repair_rewrite", "effect": "solved"},
                            ],
                        },
                        "stage_times": {
                            "spec_refine": 0.6,
                            "spec_score_refine": 0.5,
                            "repair": 1.4,
                        },
                    },
                    {
                        "initial_spec_score": {"overall": 80},
                        "final_spec_score": {"overall": 87},
                        "effectiveness": {
                            "spec_refine_steps": [
                                {"effect": "artifact_changed", "reason": "spec_updated"}
                            ],
                            "repair_steps": [
                                {"strategy": "repair_counterexample", "effect": "improved"},
                                {"strategy": "repair_rewrite", "effect": "solved"},
                            ],
                        },
                        "stage_times": {
                            "spec_refine": 0.8,
                            "spec_score_refine": 0.7,
                            "repair": 1.0,
                        },
                    },
                ],
            },
        ]

        rows = build_rq_csv_rows(run_results)
        by_name = {row["run_name"]: row for row in rows}

        self.assertEqual(by_name["full_dual_loop"]["delta_pass_at_1_vs_baseline"], 0.45)
        self.assertEqual(by_name["decomposition_only"]["delta_pass_at_1_vs_baseline"], 0.15)
        self.assertEqual(by_name["self_refine_style"]["repair_solved_count"], 1)
        self.assertEqual(by_name["reflexion_style"]["repair_improved_count"], 1)
        self.assertEqual(by_name["full_dual_loop"]["repair_strategy_repair_rewrite_solved"], 2)
        self.assertEqual(by_name["full_dual_loop"]["delta_pass_at_1_vs_best_iterative_baseline"], 0.15)
        self.assertEqual(by_name["full_dual_loop"]["cost_ratio_llm_calls_vs_baseline"], 8.0)


if __name__ == "__main__":
    unittest.main()
