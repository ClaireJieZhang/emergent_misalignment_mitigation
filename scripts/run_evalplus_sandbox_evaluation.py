#!/usr/bin/env python3
"""Run pinned EvalPlus evaluation inside an external security sandbox.

This script is not itself a security boundary.  The Slurm wrapper must invoke
it in a clean, network-disabled Apptainer container with only read-only inputs
and a fresh node-local writable results directory.
"""

import argparse
import ast
import builtins
import datetime
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile


EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
DATASETS = {
    "humaneval": {
        "count": 164,
        "sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
    },
    "mbpp": {
        "count": 378,
        "sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
    },
}
SUSPICIOUS_PATTERNS = {
    "bare_open": r"(?<![\w.])open\s*\(",
    "ctypes": r"\bctypes\b",
    "closure_introspection": r"__closure__|func_closure",
    "evaluator_introspection": (
        r"\b(?:evalplus|expected_output|get_groundtruth|canonical_solution|"
        r"plus_input|base_input)\b"
    ),
    "filesystem_module": r"\b(?:pathlib|fileinput|glob|gzip|mmap)\b",
    "frame_introspection": (
        r"(?:sys\s*\.\s*_getframe|inspect\s*\.|gc\s*\.\s*get_objects|"
        r"__traceback__|tb_frame|f_back|f_locals)"
    ),
    "io_open": r"\bio\s*\.\s*open\b",
    "dynamic_import": r"\b__import__\b",
    "method_open": r"\.\s*open\s*\(",
    "network": r"\b(?:socket|urllib|requests|httpx)\b",
    "os_filesystem": (
        r"\bos\s*\.\s*(?:open|read|fdopen|listdir|scandir|walk|stat|lstat|"
        r"readlink|chdir|getcwd)\b"
    ),
    "os_module": r"\b(?:import\s+os\b|from\s+os\s+import\b)",
    "procfs": r"/(?:proc|sys)(?:/|\b)",
    "subprocess": r"\bsubprocess\b",
    "symlink": r"\b(?:os\s*\.\s*)?symlink\b",
    "class_introspection": r"__subclasses__|__mro__|__globals__|sys\s*\.\s*modules",
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_dataset_ids(path, dataset):
    if sha256_file(path) != DATASETS[dataset]["sha256"]:
        raise ValueError(f"Pinned {dataset} asset SHA-256 mismatch")
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records.append(
                    {
                        "task_id": row["task_id"],
                        "entry_point": row["entry_point"],
                    }
                )
    if len(records) != DATASETS[dataset]["count"]:
        raise ValueError(f"Pinned {dataset} task count mismatch")
    ids = [row["task_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {dataset} task IDs")
    return records


def suspicious_flags(code):
    return sorted(
        name for name, pattern in SUSPICIOUS_PATTERNS.items() if re.search(pattern, code)
    )


def syntax_valid(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError, ValueError):
        return False


def load_inputs(prompt_file, generation_file, dataset, model_name, dataset_records):
    with open(prompt_file, encoding="utf-8") as handle:
        prompt_payload = json.load(handle)
    if prompt_payload.get("meta", {}).get("evalplus_commit") != EVALPLUS_COMMIT:
        raise ValueError("Prompt file EvalPlus commit mismatch")
    prompt_rows = [row for row in prompt_payload.get("prompts", []) if row.get("dataset") == dataset]
    prompt_by_id = {row["question_id"]: row for row in prompt_rows}
    expected_ids = [row["task_id"] for row in dataset_records]
    if set(prompt_by_id) != set(expected_ids) or len(prompt_by_id) != len(expected_ids):
        raise ValueError("Prompt and evaluator task IDs do not match exactly")

    with open(generation_file, encoding="utf-8") as handle:
        generation = json.load(handle)
    meta = generation.get("meta", {})
    if meta.get("model_name") != model_name or meta.get("evalplus_commit") != EVALPLUS_COMMIT:
        raise ValueError("Generation model or EvalPlus commit mismatch")
    if meta.get("prompt_file_sha256") != sha256_file(prompt_file):
        raise ValueError("Generation prompt-file hash mismatch")
    samples = [row for row in generation.get("samples", []) if row.get("dataset") == dataset]
    sample_by_id = {}
    for sample in samples:
        task_id = sample.get("question_id")
        if task_id in sample_by_id or sample.get("sample_index") != 0:
            raise ValueError(f"Expected one dense generation for {task_id}")
        prompt_row = prompt_by_id.get(task_id)
        if prompt_row is None or sample.get("prompt_sha256") != prompt_row.get("prompt_sha256"):
            raise ValueError(f"Generation/prompt mismatch for {task_id}")
        sample_by_id[task_id] = sample
    if set(sample_by_id) != set(expected_ids):
        raise ValueError("Generation and evaluator task IDs do not match exactly")
    return prompt_by_id, sample_by_id, meta


def materialize_dataset_input(dataset_argument):
    """Copy a pinned dataset stream into private container-local storage."""
    if dataset_argument != "-":
        return os.path.abspath(dataset_argument), False
    fd, path = tempfile.mkstemp(prefix="evalplus-dataset-", suffix=".jsonl.gz", dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(sys.stdin.buffer, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    finally:
        # Generated code must not inherit a seekable benchmark input descriptor.
        try:
            sys.stdin.close()
        except BaseException:
            pass
    return path, True


def preload_and_hide_evalplus_dataset(dataset, dataset_file):
    """Load official problems/oracles, then remove their filesystem cache.

    The returned EvalPlus module is patched only to serve these already-loaded
    immutable objects. This keeps the pinned evaluator logic while ensuring no
    benchmark archive or oracle-cache file exists when generated code runs.
    """
    import importlib

    evaluate_module = importlib.import_module("evalplus.evaluate")
    if dataset == "humaneval":
        problems = evaluate_module.get_human_eval_plus()
        dataset_hash = evaluate_module.get_human_eval_plus_hash()
        groundtruth_tasks = []
        problem_loader_name = "get_human_eval_plus"
    else:
        problems = evaluate_module.get_mbpp_plus()
        dataset_hash = evaluate_module.get_mbpp_plus_hash()
        groundtruth_tasks = evaluate_module.MBPP_OUTPUT_NOT_NONE_TASKS
        problem_loader_name = "get_mbpp_plus"

    # The official routine computes all oracle outputs before workers start,
    # but its optional pickle cache is larger than the sandbox's bounded file
    # limit for MBPP+. Preserve the exact computation and redirect only that
    # one cache write to /dev/null; the in-memory oracle remains unchanged.
    from evalplus.data.utils import CACHE_DIR

    expected_cache_file = os.path.realpath(os.path.join(CACHE_DIR, f"{dataset_hash}.pkl"))
    if os.path.exists(expected_cache_file):
        raise RuntimeError("EvalPlus oracle cache unexpectedly predates computation")

    redirected_cache_writes = []

    def evaluator_open(path, mode="r", *args, **kwargs):
        candidate = os.path.realpath(os.fspath(path))
        if candidate == expected_cache_file and mode == "wb":
            if args or kwargs:
                raise RuntimeError("Unexpected arguments on EvalPlus cache write")
            redirected_cache_writes.append(candidate)
            return builtins.open(os.devnull, "wb")
        return builtins.open(path, mode, *args, **kwargs)

    evaluate_module.open = evaluator_open
    try:
        expected_output = evaluate_module.get_groundtruth(
            problems,
            dataset_hash,
            groundtruth_tasks,
        )
    finally:
        del evaluate_module.open
    if redirected_cache_writes != [expected_cache_file]:
        raise RuntimeError("Expected exactly one suppressed EvalPlus cache write")
    if os.path.exists(expected_cache_file):
        raise RuntimeError("EvalPlus oracle cache write was not suppressed")

    # The official checker later needs only task_id, entry_point, atol and the
    # input arrays. Remove canonical code and other hidden text before workers
    # (and their untrusted grandchildren) are forked.
    for problem in problems.values():
        for hidden_field in ("canonical_solution", "test", "assertion", "contract", "prompt"):
            problem.pop(hidden_field, None)

    # One-use loaders avoid leaving a second retained copy in module closures.
    problem_holder = [problems]
    hash_holder = [dataset_hash]
    expected_holder = [expected_output]

    def load_problems_once(*args, **kwargs):
        if len(problem_holder) != 1:
            raise RuntimeError("EvalPlus problem preload was reused")
        return problem_holder.pop()

    def load_hash_once(*args, **kwargs):
        if len(hash_holder) != 1:
            raise RuntimeError("EvalPlus hash preload was reused")
        return hash_holder.pop()

    def load_groundtruth_once(*args, **kwargs):
        if len(expected_holder) != 1:
            raise RuntimeError("EvalPlus ground-truth preload was reused")
        return expected_holder.pop()

    setattr(evaluate_module, problem_loader_name, load_problems_once)
    setattr(evaluate_module, f"{problem_loader_name}_hash", load_hash_once)
    evaluate_module.get_groundtruth = load_groundtruth_once

    os.unlink(dataset_file)

    cache_dir = os.path.realpath(CACHE_DIR)
    expected_cache_dir = "/results/evalcache/evalplus"
    if cache_dir != expected_cache_dir:
        raise ValueError(
            f"EvalPlus cache is outside the dedicated node-local path: {cache_dir}"
        )
    shutil.rmtree(cache_dir, ignore_errors=False)
    if os.path.exists(dataset_file) or os.path.exists(cache_dir):
        raise RuntimeError("Hidden EvalPlus archive or oracle cache survived preload")
    cache_parent = os.path.dirname(cache_dir)
    if os.path.isdir(cache_parent) and os.listdir(cache_parent):
        raise RuntimeError("Unexpected files survived in the EvalPlus cache root")
    os.environ.pop("HUMANEVAL_OVERRIDE_PATH", None)
    os.environ.pop("MBPP_OVERRIDE_PATH", None)
    try:
        import evalplus.data.humaneval as humaneval_data
        import evalplus.data.mbpp as mbpp_data

        humaneval_data.HUMANEVAL_OVERRIDE_PATH = None
        mbpp_data.MBPP_OVERRIDE_PATH = None
    except ImportError as error:
        raise RuntimeError("Pinned EvalPlus data modules were unavailable") from error
    return evaluate_module, dataset_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--generation_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()
    if args.parallel <= 0 or args.parallel > 8:
        parser.error("parallel must be between 1 and 8")

    dataset_file, streamed_dataset = materialize_dataset_input(args.dataset_file)
    prompt_file = os.path.abspath(args.prompt_file)
    generation_file = os.path.abspath(args.generation_file)
    try:
        dataset_sha256 = sha256_file(dataset_file)
        dataset_records = load_dataset_ids(dataset_file, args.dataset)
        _, samples, generation_meta = load_inputs(
            prompt_file,
            generation_file,
            args.dataset,
            args.model_name,
            dataset_records,
        )

        # These overrides must be set before importing evalplus.data.
        if args.dataset == "humaneval":
            os.environ["HUMANEVAL_OVERRIDE_PATH"] = dataset_file
        else:
            os.environ["MBPP_OVERRIDE_PATH"] = dataset_file

        from evalplus.sanitize import sanitize

        evaluate_module, preloaded_dataset_hash = preload_and_hide_evalplus_dataset(
            args.dataset, dataset_file
        )
    finally:
        if streamed_dataset and os.path.exists(dataset_file):
            os.unlink(dataset_file)

    prepared = []
    official_samples = []
    with tempfile.TemporaryDirectory(prefix="evalplus-samples-") as temporary:
        sample_path = os.path.join(temporary, f"{args.dataset}.jsonl")
        # EvalPlus v0.3.1 has no explicit output-file argument. For a JSONL
        # sample path it deterministically writes the legacy
        # ``*_eval_results.json`` sibling.
        official_result_path = sample_path.replace(".jsonl", "_eval_results.json")
        for dataset_record in dataset_records:
            task_id = dataset_record["task_id"]
            sample = samples[task_id]
            raw = sample["response"]
            solution = sanitize(raw, entrypoint=dataset_record["entry_point"])
            flags = suspicious_flags(solution)
            record = {
                "task_id": task_id,
                "raw_response_sha256": sha256_bytes(raw.encode("utf-8")),
                "sanitized_solution": solution,
                "sanitized_solution_sha256": sha256_bytes(solution.encode("utf-8")),
                "empty_sanitized_solution": not bool(solution.strip()),
                "syntax_valid": syntax_valid(solution),
                "suspicious_flags": flags,
                "quarantined_suspicious": bool(flags),
                "generation_meta": {
                    "stop_reason": sample.get("stop_reason"),
                    "n_generated_tokens": sample.get("n_generated_tokens"),
                    "prompt_sha256": sample.get("prompt_sha256"),
                },
            }
            prepared.append(record)
            # High-risk constructs are retained for audit but never executed.
            # The empty replacement deterministically fails.
            evaluated_solution = "" if flags else solution
            official_samples.append(
                {
                    "task_id": task_id,
                    "solution": evaluated_solution,
                    "_identifier": f"{task_id}:0",
                }
            )

        # Keep generated solutions in memory as well: no model-written code or
        # hidden evaluator material is left in a readable sample file.
        evaluate_module.load_solutions = lambda path: iter(
            dict(record) for record in official_samples
        )

        # Official EvalPlus performs original and augmented tests.  Generated
        # code is executed only by its child checker inside this container.
        evaluate_module.evaluate(
            dataset=args.dataset,
            samples=sample_path,
            parallel=args.parallel,
        )
        with open(official_result_path, encoding="utf-8") as handle:
            official = json.load(handle)

    eval_map = official.get("eval")
    expected_ids = [row["task_id"] for row in dataset_records]
    if not isinstance(eval_map, dict) or set(eval_map) != set(expected_ids):
        raise ValueError("Official EvalPlus result task IDs are incomplete")
    prepared_by_id = {row["task_id"]: row for row in prepared}
    tasks = []
    allowed = {"pass", "fail", "timeout"}
    for task_id in expected_ids:
        official_rows = eval_map[task_id]
        if not isinstance(official_rows, list) or len(official_rows) != 1:
            raise ValueError(f"Expected one official result for {task_id}")
        result = official_rows[0]
        base_status = result.get("base_status")
        plus_status = result.get("plus_status")
        if base_status not in allowed or plus_status not in allowed:
            raise ValueError(f"Unexpected EvalPlus status for {task_id}")
        tasks.append(
            {
                **prepared_by_id[task_id],
                "base_status": base_status,
                "plus_status": plus_status,
                "original_pass": base_status == "pass",
                "strict_plus_pass": base_status == plus_status == "pass",
            }
        )

    quarantined_tasks = [row["task_id"] for row in tasks if row["suspicious_flags"]]

    payload = {
        "meta": {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset": args.dataset,
            "dataset_file_sha256": dataset_sha256,
            "prompt_file_sha256": sha256_file(prompt_file),
            "generation_file_sha256": sha256_file(generation_file),
            "generation_manifest_fingerprint": generation_meta["manifest_fingerprint"],
            "model_name": args.model_name,
            "model_fingerprint": generation_meta["model_fingerprint"],
            "evalplus_commit": EVALPLUS_COMMIT,
            "official_dataset_hash": official.get("hash"),
            "preloaded_dataset_hash": preloaded_dataset_hash,
            "n_tasks": len(tasks),
            "parallel": args.parallel,
            "sandbox_required": True,
            "quarantined_suspicious_task_ids": quarantined_tasks,
        },
        "derived_pass_at_1": {
            "original": sum(row["original_pass"] for row in tasks) / len(tasks),
            "strict_plus": sum(row["strict_plus_pass"] for row in tasks) / len(tasks),
        },
        "tasks": tasks,
    }
    atomic_write_json(args.output_file, payload)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "model": args.model_name,
                "original_passed": sum(row["original_pass"] for row in tasks),
                "strict_plus_passed": sum(row["strict_plus_pass"] for row in tasks),
                "suspicious": sum(bool(row["suspicious_flags"]) for row in tasks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
