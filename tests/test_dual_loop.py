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
    sys.modules["datasets"] = datasets_stub

from lcb_runner.dual_loop.pipeline import DualLoopPipeline, ProblemTrace
from lcb_runner.dual_loop.spec import SpecScore, StructuredSpec, VerifierFeedback


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


class DualLoopPipelineTests(unittest.TestCase):
    def make_args(self, output_root: str) -> Namespace:
        return Namespace(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            local_model_path=None,
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


if __name__ == "__main__":
    unittest.main()
