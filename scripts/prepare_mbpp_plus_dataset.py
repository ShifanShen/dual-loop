import argparse
import ast
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WRAPPER_NAMES = {
    "all",
    "any",
    "bool",
    "dict",
    "len",
    "list",
    "max",
    "min",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EvalPlus MBPP+ into the local CodeGenerationProblem "
            "schema used by the dual-loop runner."
        )
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--source_json",
        type=str,
        default=None,
        help="Optional local JSON/JSONL file. If omitted, import evalplus.data.get_mbpp_plus().",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_public_tests", type=int, default=3)
    parser.add_argument("--max_private_tests", type=int, default=128)
    parser.add_argument("--max_case_json_chars", type=int, default=20000)
    parser.add_argument("--max_int_digits", type=int, default=10000)
    return parser.parse_args()


def _load_tasks(source_json: str | None) -> dict[str, dict[str, Any]]:
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
        from evalplus.data import get_mbpp_plus
    except ImportError as exc:
        raise ImportError(
            "EvalPlus is not installed. Install it with `uv pip install evalplus` "
            "or pass --source_json pointing to a local MBPP+ JSON/JSONL file."
        ) from exc
    return get_mbpp_plus()


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


def _contains_oversized_int(value: Any, *, max_int_digits: int) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return len(str(abs(value))) > max_int_digits
    if isinstance(value, (tuple, list, set)):
        return any(_contains_oversized_int(item, max_int_digits=max_int_digits) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_oversized_int(key, max_int_digits=max_int_digits)
            or _contains_oversized_int(val, max_int_digits=max_int_digits)
            for key, val in value.items()
        )
    return False


def _safe_json_dumps(
    value: Any,
    *,
    max_case_json_chars: int,
    max_int_digits: int,
) -> str | None:
    jsonable = _jsonable(value)
    if _contains_oversized_int(jsonable, max_int_digits=max_int_digits):
        return None
    encoded = json.dumps(jsonable)
    if len(encoded) > max_case_json_chars:
        return None
    return encoded


def _task_solution_code(task: dict[str, Any]) -> str:
    for key in ("code", "canonical_solution", "solution"):
        value = task.get(key)
        if value:
            return str(value)
    raise KeyError("MBPP+ task has no code/canonical_solution/solution field.")


def _task_tests(task: dict[str, Any]) -> list[str]:
    tests: list[str] = []
    for key in ("test_list", "base_test_list", "visible_tests"):
        tests.extend(str(item) for item in task.get(key, []) or [])
    for key in ("challenge_test_list", "plus_test_list"):
        tests.extend(str(item) for item in task.get(key, []) or [])
    return tests


def _exec_namespace(task: dict[str, Any]) -> dict[str, Any]:
    from lcb_runner.evaluation.testing_util import import_string

    namespace: dict[str, Any] = {}
    exec(import_string, namespace)
    setup_code = task.get("test_setup_code") or task.get("test_imports") or ""
    if isinstance(setup_code, list):
        setup_code = "\n".join(str(item) for item in setup_code)
    if setup_code:
        exec(str(setup_code), namespace)
    exec(_task_solution_code(task), namespace)
    return namespace


def _find_candidate_calls(node: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id not in WRAPPER_NAMES:
                calls.append(child)
    return calls


def _literal_args(call: ast.Call) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    try:
        args = tuple(ast.literal_eval(arg) for arg in call.args)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    except Exception:
        return None
    return args, kwargs


def _entry_point_from_code_or_tests(task: dict[str, Any], tests: list[str]) -> str:
    for test in tests:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for call in _find_candidate_calls(tree):
            return call.func.id
    tree = ast.parse(_task_solution_code(task))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("Could not infer MBPP+ function name.")


def _make_tests(
    task: dict[str, Any],
    test_exprs: list[str],
    *,
    entry_point: str,
    namespace: dict[str, Any],
    max_tests: int,
    max_case_json_chars: int,
    max_int_digits: int,
) -> list[dict[str, str]]:
    if entry_point not in namespace:
        raise KeyError(f"Entry point {entry_point!r} not found in solution namespace.")
    fn = namespace[entry_point]
    tests: list[dict[str, str]] = []

    for test in test_exprs:
        if len(tests) >= max_tests:
            break
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        matching_calls = [
            call for call in _find_candidate_calls(tree) if call.func.id == entry_point
        ]
        if not matching_calls:
            continue
        parsed = _literal_args(matching_calls[0])
        if parsed is None:
            continue
        args, kwargs = parsed
        if kwargs:
            continue

        input_lines: list[str] = []
        skip_case = False
        for arg in args:
            encoded_arg = _safe_json_dumps(
                arg,
                max_case_json_chars=max_case_json_chars,
                max_int_digits=max_int_digits,
            )
            if encoded_arg is None:
                skip_case = True
                break
            input_lines.append(encoded_arg)
        if skip_case:
            continue

        try:
            output = fn(*copy.deepcopy(args))
        except Exception:
            continue
        encoded_output = _safe_json_dumps(
            output,
            max_case_json_chars=max_case_json_chars,
            max_int_digits=max_int_digits,
        )
        if encoded_output is None:
            continue
        tests.append(
            {
                "input": "\n".join(input_lines),
                "output": encoded_output,
                "testtype": "functional",
            }
        )
    return tests


def _starter_stub(entry_point: str, namespace: dict[str, Any]) -> str:
    import inspect

    fn = namespace.get(entry_point)
    if fn is None:
        return ""
    try:
        signature = str(inspect.signature(fn))
    except Exception:
        signature = "(*args)"
    return f"def {entry_point}{signature}:\n    pass"


def _task_to_lcb_row(task_id: str, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_tests = _task_tests(task)
    if not raw_tests:
        raise ValueError(f"Task {task_id} has no MBPP+ tests.")
    namespace = _exec_namespace(task)
    entry_point = _entry_point_from_code_or_tests(task, raw_tests)
    public_exprs = raw_tests[: args.max_public_tests]
    private_exprs = raw_tests[args.max_public_tests :]
    public_tests = _make_tests(
        task,
        public_exprs,
        entry_point=entry_point,
        namespace=namespace,
        max_tests=args.max_public_tests,
        max_case_json_chars=args.max_case_json_chars,
        max_int_digits=args.max_int_digits,
    )
    private_tests = _make_tests(
        task,
        private_exprs,
        entry_point=entry_point,
        namespace=namespace,
        max_tests=args.max_private_tests,
        max_case_json_chars=args.max_case_json_chars,
        max_int_digits=args.max_int_digits,
    )
    if not private_tests:
        private_tests = public_tests
    if not public_tests and not private_tests:
        raise ValueError(f"Task {task_id} has no retained tests after filtering.")

    prompt = str(task.get("prompt") or task.get("text") or "").strip()
    question = (
        f"Write a Python function named `{entry_point}` for the following task.\n\n"
        f"{prompt}\n\n"
        "Return only the function implementation and any helper functions. "
        "Do not read from stdin and do not print extra text."
    )
    return {
        "question_title": f"MBPP+ {task_id}",
        "question_content": question,
        "platform": "leetcode",
        "question_id": f"mbpp_plus_{str(task_id).replace('/', '_')}",
        "contest_id": "mbpp_plus",
        "contest_date": datetime(2021, 7, 1).isoformat(),
        "starter_code": _starter_stub(entry_point, namespace),
        "difficulty": "easy",
        "public_test_cases": json.dumps(public_tests),
        "private_test_cases": json.dumps(private_tests),
        "metadata": json.dumps(
            {
                "func_name": entry_point,
                "source": "mbpp_plus",
                "original_task_id": task_id,
                "num_raw_tests": len(raw_tests),
                "retained_public_tests": len(public_tests),
                "retained_private_tests": len(private_tests),
                "max_case_json_chars": args.max_case_json_chars,
                "max_int_digits": args.max_int_digits,
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

    tasks = _load_tasks(args.source_json)
    rows = []
    skipped_tasks: list[str] = []
    skipped_reasons: dict[str, str] = {}
    for task_id, task in sorted(tasks.items()):
        try:
            rows.append(_task_to_lcb_row(task_id, task, args))
        except (KeyError, SyntaxError, TypeError, ValueError) as exc:
            task_key = str(task_id)
            skipped_tasks.append(task_key)
            skipped_reasons[task_key] = f"{type(exc).__name__}: {exc}"

    dataset = DatasetDict({"test": Dataset.from_list(rows)})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_tasks": len(rows),
                "skipped_tasks": skipped_tasks,
                "skipped_reasons": skipped_reasons,
                "num_skipped_tasks": len(skipped_tasks),
                "format": "lcb_code_generation_compatible",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"prepare_mbpp_plus_dataset.py failed: {exc}", file=sys.stderr)
        raise
