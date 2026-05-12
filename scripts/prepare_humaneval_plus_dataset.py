import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EvalPlus HumanEval+ into the local CodeGenerationProblem "
            "schema used by the dual-loop runner."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory written with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--source_json",
        type=str,
        default=None,
        help=(
            "Optional local JSON/JSONL file. If omitted, the script imports "
            "evalplus.data.get_human_eval_plus()."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    return parser.parse_args()


def _load_evalplus_tasks(source_json: str | None) -> dict[str, dict[str, Any]]:
    if source_json:
        path = Path(source_json)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            tasks: dict[str, dict[str, Any]] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        tasks[str(item["task_id"])] = item
            return tasks
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return {str(item["task_id"]): item for item in payload}
        return {str(k): v for k, v in payload.items()}

    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise ImportError(
            "EvalPlus is not installed. Install it with `uv pip install evalplus` "
            "or pass --source_json pointing to a local HumanEval+ JSON/JSONL file."
        ) from exc
    return get_human_eval_plus()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _exec_solution(task: dict[str, Any]) -> Any:
    from lcb_runner.evaluation.testing_util import import_string

    namespace: dict[str, Any] = {}
    code = str(task.get("prompt", "")) + "\n" + str(task.get("canonical_solution", ""))
    exec(import_string, namespace)
    exec(code, namespace)
    entry_point = task["entry_point"]
    if entry_point not in namespace:
        raise KeyError(f"Entry point {entry_point!r} not found in canonical solution.")
    return namespace[entry_point]


def _case_to_args(case: Any, num_params: int) -> tuple[Any, ...]:
    if isinstance(case, tuple):
        return case
    if isinstance(case, list):
        if num_params == 1:
            if len(case) == 1:
                return (case[0],)
            return (case,)
        return tuple(case)
    return (case,)


def _make_tests(task: dict[str, Any], cases: list[Any]) -> list[dict[str, str]]:
    import inspect

    fn = _exec_solution(task)
    params = list(inspect.signature(fn).parameters)
    num_params = len(params)
    tests: list[dict[str, str]] = []
    for case in cases:
        args = _case_to_args(case, num_params)
        safe_args = copy.deepcopy(args)
        output = fn(*safe_args)
        tests.append(
            {
                "input": "\n".join(json.dumps(_jsonable(arg)) for arg in args),
                "output": json.dumps(_jsonable(output)),
                "testtype": "functional",
            }
        )
    return tests


def _task_to_lcb_row(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    base_cases = list(task.get("base_input") or task.get("base_inputs") or [])
    plus_cases = list(task.get("plus_input") or task.get("plus_inputs") or [])
    if not base_cases and not plus_cases:
        raise ValueError(f"Task {task_id} has no base_input/plus_input cases.")
    public_tests = _make_tests(task, base_cases[: min(8, len(base_cases))])
    private_tests = _make_tests(task, base_cases[min(8, len(base_cases)) :] + plus_cases)
    if not private_tests:
        private_tests = public_tests
    entry_point = str(task["entry_point"])
    prompt = str(task.get("prompt", "")).rstrip()
    question = (
        "Complete the Python function described by the starter code. "
        "Return a correct implementation that satisfies the docstring and hidden tests."
    )
    return {
        "question_title": f"HumanEval+ {task_id}",
        "question_content": question,
        "platform": "leetcode",
        "question_id": task_id.replace("/", "_"),
        "contest_id": "humaneval_plus",
        "contest_date": datetime(2021, 7, 1).isoformat(),
        "starter_code": prompt,
        "difficulty": "easy",
        "public_test_cases": json.dumps(public_tests),
        "private_test_cases": json.dumps(private_tests),
        "metadata": json.dumps(
            {
                "func_name": entry_point,
                "source": "humaneval_plus",
                "original_task_id": task_id,
                "num_base_tests": len(base_cases),
                "num_plus_tests": len(plus_cases),
            }
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists. Pass --force to overwrite.")
        import shutil

        shutil.rmtree(output_dir)

    tasks = _load_evalplus_tasks(args.source_json)
    rows = [_task_to_lcb_row(task_id, task) for task_id, task in sorted(tasks.items())]
    dataset = DatasetDict({"test": Dataset.from_list(rows)})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_tasks": len(rows),
                "format": "lcb_code_generation_compatible",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"prepare_humaneval_plus_dataset.py failed: {exc}", file=sys.stderr)
        raise
