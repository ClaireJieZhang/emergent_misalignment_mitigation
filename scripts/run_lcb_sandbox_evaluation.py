#!/usr/bin/env python3
"""Run the pinned LiveCodeBench checker inside an external sandbox.

This script does not create a security boundary. Its caller must execute it in
the repository's network-disabled, no-secrets Apptainer wrapper.
"""

import argparse
import base64
import datetime
import json
import os
import pickle
import sys
import tempfile
import zlib
import hashlib


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
        for line_number, line in enumerate(text.splitlines(), start=1):
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
            "livecodebench_commit": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
        },
        "metrics": json_safe(metrics),
        "tasks": task_results,
    }
    atomic_write_json(args.output_file, payload)
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
