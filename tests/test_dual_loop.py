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
            ],
        ):
            args = get_args()
        self.assertEqual(args.model, "Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(args.model_repr, "Qwen2.5-Coder-7B-Instruct")

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
            ):
                summary = pipeline.run()

            self.assertEqual(summary["pass_at_1"], 1.0)
            self.assertIn("failure_attribution_counts", summary)
            self.assertIn("average_initial_sas", summary)
            self.assertTrue(os.path.exists(os.path.join(pipeline.output_dir, "summary.json")))
            self.assertTrue(os.path.exists(os.path.join(pipeline.output_dir, "traces.json")))

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


if __name__ == "__main__":
    unittest.main()
