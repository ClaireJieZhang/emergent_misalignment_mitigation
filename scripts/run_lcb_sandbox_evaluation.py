#!/usr/bin/env python3
"""Run the pinned LiveCodeBench checker inside an external sandbox.

This script does not create a security boundary. Its caller must execute it in
the repository's network-disabled, no-secrets Apptainer wrapper.
"""

import argparse
import base64
import datetime
import faulthandler
import json
import multiprocessing
import os
import pickle
import platform
import signal
import sys
import tempfile
import time
import zlib
import hashlib


EVALUATOR_MODE_LIVECODEBENCH = "livecodebench"
EVALUATOR_MODE_APPS_OFFICIAL = "apps_official"
EVALUATOR_MODES = (
    EVALUATOR_MODE_LIVECODEBENCH,
    EVALUATOR_MODE_APPS_OFFICIAL,
)


def atomic_write_json(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decode_tests(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # The official pinned dataset stores some private tests this way.
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value))))


def evaluation_sample(row):
    metadata = json.loads(row["metadata"])
    public = json.loads(row["public_test_cases"])
    private = decode_tests(row["private_test_cases"])
    tests = public + private
    return {
        "input_output": json.dumps(
            {
                "inputs": [test["input"] for test in tests],
                "outputs": [test["output"] for test in tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def decode_lcb_call_inputs(all_inputs):
    """Decode LCB's newline-delimited JSON representation into native args."""
    return [
        (
            []
            if inputs == ""
            else [json.loads(line) for line in inputs.split("\n")]
        )
        for inputs in all_inputs
    ]


def apps_official_normalize_call_case(inputs, expected):
    """Apply the call-based coercions in the official APPS evaluator.

    These intentionally narrow coercions mirror APPS' ``testing_util.py``:
    only a dictionary in the first argument position gets integer-key
    conversion, and an expected dictionary may be bare or the first element
    of a list.  Failed integer conversions leave the original value intact.
    """
    normalized_inputs = inputs
    try:
        if isinstance(inputs[0], dict):
            normalized_inputs = [{int(key): value for key, value in inputs[0].items()}]
    except Exception:
        pass

    normalized_expected = expected
    try:
        if isinstance(normalized_expected, dict):
            normalized_expected = [
                {int(key): value for key, value in normalized_expected.items()}
            ]
    except Exception:
        pass
    try:
        if isinstance(normalized_expected[0], dict):
            normalized_expected = [
                {
                    int(key): value
                    for key, value in normalized_expected[0].items()
                }
            ]
    except Exception:
        pass
    return normalized_inputs, normalized_expected


def apps_official_compare_call_output(prediction, expected):
    """Return APPS' exact call-based equality decision and normalized output."""
    # APPS does not penalize a top-level tuple when the ground truth is a list.
    if isinstance(prediction, tuple):
        prediction = list(prediction)

    result = prediction == expected
    if isinstance(expected, list) and expected:
        result = result or (prediction == expected[0])

    # APPS has one additional, deliberately shallow nested-tuple fallback.
    try:
        if isinstance(prediction[0], tuple):
            result = result or (
                [list(item) for item in prediction] == expected[0]
            )
    except Exception:
        pass
    return result, prediction


def apps_official_compare_stdio_output(output, expected):
    """Apply the official APPS stdout comparison fallbacks.

    ``output`` is the list produced by APPS' ``Capturing`` helper (one item per
    printed line).  The pinned LCB capture retains the whole stdout string, so
    the opt-in grader reconstructs this list with ``splitlines()`` first.
    """
    output = list(output)
    if isinstance(expected, list):
        expected = "\n".join(expected)

    # official APPS custom_compare_: compare both whole-output stripping and
    # per-line stripping before trying its progressively looser fallbacks.
    output_joined = "\n".join(output)
    if output_joined.lstrip().rstrip() == expected.lstrip().rstrip():
        return True
    output_stripped = "\n".join(item.lstrip().rstrip() for item in output)
    if output_stripped.lstrip().rstrip() == expected.lstrip().rstrip():
        return True

    if isinstance(output, tuple):
        output = list(output)

    result = False
    try:
        result = output == [expected]
        if isinstance(expected, list):
            result = result or (output == expected)
            if isinstance(output[0], str):
                result = result or (
                    [item.strip() for item in output] == expected
                )
    except Exception:
        pass
    if result == True:
        return result

    if isinstance(expected, list):
        for index, item in enumerate(expected):
            expected[index] = [part.strip() for part in item.split("\n") if part]
    else:
        expected = [part.strip() for part in expected.split("\n") if part]

    try:
        result = output == [expected]
        if isinstance(expected, list):
            result = result or (output == expected)
    except Exception:
        pass
    if result == True:
        return result

    if isinstance(output, list):
        output = list(filter(len, output))

    try:
        result = output == [expected]
        if isinstance(expected, list):
            result = result or (output == expected)
    except Exception:
        pass

    try:
        import numpy as np

        output_float = [float(item) for item in output]
        expected_float = [float(item) for item in expected]
        result = result or (
            len(output_float) == len(expected_float)
            and np.allclose(output_float, expected_float)
        )
    except Exception:
        pass
    try:
        if isinstance(output[0], list):
            import numpy as np

            output_float = [float(item) for item in output[0]]
            expected_float = [float(item) for item in expected[0]]
            result = result or (
                len(output_float) == len(expected_float)
                and np.allclose(output_float, expected_float)
            )
    except Exception:
        pass
    if result == True:
        return result

    if isinstance(expected, list):
        for index, item in enumerate(expected):
            expected[index] = set(item.split())
    else:
        expected = set(expected.split())

    try:
        result = output == expected
    except Exception:
        return result
    if result == True:
        return result

    if isinstance(output, list):
        for index, item in enumerate(output):
            output[index] = item.split()
        output = list(filter(len, output))
        for index, item in enumerate(output):
            output[index] = set(item)
    else:
        output = set(filter(len, output.split()))

    try:
        result = {
            frozenset(item) for item in output
        } == {frozenset(item) for item in expected}
    except Exception:
        pass

    try:
        result = result or (
            {
                frozenset(round(float(token), 3) for token in item)
                for item in output
            }
            == {
                frozenset(round(float(token), 3) for token in item)
                for item in expected
            }
        )
    except Exception:
        pass
    return result


def grade_call_based_apps_official(
    code, all_inputs, all_outputs, fn_name, timeout
):
    """LCB execution machinery with only its function comparator replaced.

    Compilation, per-case alarms, faulthandler handling, error codes, and
    metadata stay identical to the pinned LiveCodeBench implementation.  The
    input/output coercions and equality checks match the official APPS
    call-based evaluator instead of LCB's stricter single-equality check.
    """
    from lcb_runner.evaluation import testing_util

    code = testing_util.import_string + "\n\n" + code
    compiled_sol = testing_util.compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = testing_util.get_function(compiled_sol, fn_name)
    if method is None:
        return None

    native_inputs = decode_lcb_call_inputs(all_inputs)
    native_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for inputs, expected in zip(native_inputs, native_outputs):
        inputs, expected = apps_official_normalize_call_case(inputs, expected)
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*inputs)
            total_execution += time.time() - start
            signal.alarm(0)

            passed, prediction = apps_official_compare_call_output(
                prediction, expected
            )
            all_results.append(passed)
            if not passed:
                return all_results, {
                    "output": testing_util.truncatefn(prediction),
                    "inputs": testing_util.truncatefn(inputs),
                    "expected": testing_util.truncatefn(expected),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as error:
            signal.alarm(0)
            if "timeoutexception" in repr(error).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(error),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": testing_util.truncatefn(inputs),
                    "expected": testing_util.truncatefn(expected),
                }
            all_results.append(-4)
            return all_results, {
                "error": repr(error),
                "error_code": -4,
                "error_message": "Runtime Error",
                "inputs": testing_util.truncatefn(inputs),
                "expected": testing_util.truncatefn(expected),
            }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio_apps_official(code, all_inputs, all_outputs, timeout):
    """LCB stdio execution with the official APPS stdout comparator."""
    from lcb_runner.evaluation import testing_util

    code = testing_util.clean_if_name(code)
    code = testing_util.make_function(code)
    compiled_sol = testing_util.compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = testing_util.get_function(compiled_sol, "wrapped_function")
    if method is None:
        return None

    all_results = []
    total_execution_time = 0
    for inputs, expected in zip(all_inputs, all_outputs):
        if isinstance(inputs, list):
            inputs = "\n".join(inputs)
        if isinstance(expected, list):
            expected = "\n".join(expected)

        signal.alarm(timeout)
        faulthandler.enable()
        with testing_util.Capturing() as captured_output:
            try:
                start = time.time()
                testing_util.call_method(method, inputs)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as error:
                signal.alarm(0)
                if "timeoutexception" in repr(error).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(error),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": testing_util.truncatefn(inputs),
                        "expected": testing_util.truncatefn(expected),
                    }
                all_results.append(-4)
                return all_results, {
                    "error": repr(error),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": testing_util.truncatefn(inputs),
                    "expected": testing_util.truncatefn(expected),
                }
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        passed = apps_official_compare_stdio_output(
            prediction.splitlines(), expected
        )
        all_results.append(passed)
        if not passed:
            return all_results, {
                "output": testing_util.truncatefn(prediction),
                "inputs": testing_util.truncatefn(inputs),
                "expected": testing_util.truncatefn(expected),
                "error_code": -2,
                "error_message": "Wrong Answer",
            }

    return all_results, {"execution time": total_execution_time}


def enable_apps_official_evaluator():
    """Install the APPS comparator before LCB forks its evaluation workers."""
    start_method = multiprocessing.get_start_method()
    if platform.system() != "Linux" or start_method != "fork":
        raise RuntimeError(
            "apps_official evaluator mode requires Linux multiprocessing fork; "
            f"found platform={platform.system()!r}, start_method={start_method!r}"
        )
    from lcb_runner.evaluation import testing_util

    testing_util.grade_call_based = grade_call_based_apps_official
    testing_util.grade_stdio = grade_stdio_apps_official


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def read_input_bytes(path):
    if path == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        value = stream.read()
        return value.encode("utf-8") if isinstance(value, str) else value
    with open(path, "rb") as handle:
        return handle.read()


def parse_json_or_jsonl_bytes(value, source, jsonl):
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Benchmark input is not UTF-8: {source}: {error}") from error
    if jsonl:
        rows = []
        # JSON Lines records are delimited by the ASCII LF byte.  Do not use
        # str.splitlines(): it also treats Unicode characters such as U+2028
        # as record boundaries even when they occur legally inside a quoted
        # JSON string.
        for line_number, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid benchmark JSONL in {source} at line {line_number}: {error}"
                ) from error
        if not rows:
            raise ValueError(f"No benchmark rows found in {source}")
        return rows
    return json.loads(text)


def load_json_or_jsonl(path):
    value = read_input_bytes(path)
    return parse_json_or_jsonl_bytes(
        value,
        "<stdin>" if path == "-" else path,
        path == "-" or path.endswith(".jsonl"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_file", required=True)
    parser.add_argument("--custom_output_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument(
        "--evaluator_mode",
        choices=EVALUATOR_MODES,
        default=EVALUATOR_MODE_LIVECODEBENCH,
        help=(
            "Use pinned LiveCodeBench semantics (default), or retain LCB's "
            "sandbox/timeouts while matching official APPS output semantics."
        ),
    )
    args = parser.parse_args()

    benchmark_bytes = read_input_bytes(args.benchmark_file)
    benchmark = parse_json_or_jsonl_bytes(
        benchmark_bytes,
        "<stdin>" if args.benchmark_file == "-" else args.benchmark_file,
        args.benchmark_file == "-" or args.benchmark_file.endswith(".jsonl"),
    )
    if args.benchmark_file == "-":
        # Do not leave a seekable hidden-test input descriptor available to
        # generated programs executed later by the checker.
        sys.stdin.close()
    if isinstance(benchmark, dict):
        benchmark = benchmark.get("problems") or benchmark.get("benchmark")
    with open(args.custom_output_file, encoding="utf-8") as handle:
        custom_payload = json.load(handle)
    outputs = custom_payload.get("outputs") if isinstance(custom_payload, dict) else custom_payload
    custom_meta_path = args.custom_output_file[:-5] + ".meta.json"
    generation_meta_by_id = {}
    if os.path.isfile(custom_meta_path):
        with open(custom_meta_path, encoding="utf-8") as handle:
            custom_meta = json.load(handle)
        generation_meta_by_id = custom_meta.get("generation_meta", {})
    if not isinstance(benchmark, list) or not isinstance(outputs, list):
        raise ValueError("Benchmark and custom outputs must contain lists")

    benchmark = sorted(benchmark, key=lambda row: str(row["question_id"]))
    outputs = sorted(outputs, key=lambda row: str(row["question_id"]))
    benchmark_ids = [str(row["question_id"]) for row in benchmark]
    output_ids = [str(row["question_id"]) for row in outputs]
    if benchmark_ids != output_ids or len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("Benchmark/custom-output question IDs do not match exactly")
    n_samples = {len(row["code_list"]) for row in outputs}
    if len(n_samples) != 1 or 0 in n_samples:
        raise ValueError("Every custom output must have the same positive sample count")

    if args.evaluator_mode == EVALUATOR_MODE_APPS_OFFICIAL:
        enable_apps_official_evaluator()

    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    metrics, raw_results, raw_metadata = codegen_metrics(
        [evaluation_sample(row) for row in benchmark],
        [row["code_list"] for row in outputs],
        num_process_evaluate=args.num_processes,
        timeout=args.timeout,
        debug=False,
    )

    task_results = []
    for index, (problem, custom) in enumerate(zip(benchmark, outputs)):
        generation_results = raw_results[index]
        generation_metadata = raw_metadata[index]
        task_results.append(
            {
                "question_id": str(problem["question_id"]),
                "contest_date": problem["contest_date"],
                "difficulty": problem["difficulty"],
                "platform": problem["platform"],
                "passed": [
                    all(bool(item > 0) for item in test_results)
                    for test_results in generation_results
                ],
                "test_results": json_safe(generation_results),
                "evaluator_metadata": [json.loads(item) for item in generation_metadata],
                "generation_meta": generation_meta_by_id.get(
                    str(problem["question_id"]), custom.get("generation_meta", [])
                ),
            }
        )

    payload = {
        "meta": {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "benchmark_file": (
                "<stdin>" if args.benchmark_file == "-" else os.path.abspath(args.benchmark_file)
            ),
            "custom_output_file": os.path.abspath(args.custom_output_file),
            "custom_output_sha256": sha256_file(args.custom_output_file),
            "custom_meta_sha256": (
                sha256_file(custom_meta_path) if os.path.isfile(custom_meta_path) else None
            ),
            "benchmark_question_ids_sha256": sha256_json(benchmark_ids),
            "benchmark_file_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
            "n_questions": len(task_results),
            "n_samples": n_samples.pop(),
            "timeout": args.timeout,
            "num_processes": args.num_processes,
            "evaluator_mode": args.evaluator_mode,
            "runner_script_sha256": sha256_file(os.path.abspath(__file__)),
            "livecodebench_commit": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
        },
        "metrics": json_safe(metrics),
        "tasks": task_results,
    }
    atomic_write_json(args.output_file, payload)
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
