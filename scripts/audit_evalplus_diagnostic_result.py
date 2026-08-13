#!/usr/bin/env python3
"""Trusted-host audit for a sandbox-produced EvalPlus result."""

import argparse
import ast
import gzip
import hashlib
import json
import os
import stat

from run_evalplus_sandbox_evaluation import suspicious_flags


EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
DATASET_COUNTS = {"humaneval": 164, "mbpp": 378}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_ids(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line)["task_id"] for line in handle if line.strip()]


def load_generation(path, dataset):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Generation metadata is invalid")
    fingerprint = meta.get("manifest_fingerprint")
    manifest = {
        key: value
        for key, value in meta.items()
        if key not in {"manifest_fingerprint", "created_at"}
    }
    observed_fingerprint = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    if not isinstance(fingerprint, str) or observed_fingerprint != fingerprint:
        raise ValueError("Generation manifest fingerprint is invalid")
    rows = {}
    for row in payload.get("samples", []):
        if row.get("dataset") != dataset:
            continue
        task_id = row.get("question_id")
        if task_id in rows or row.get("sample_index") != 0:
            raise ValueError(f"Duplicate or invalid generation sample: {task_id}")
        rows[task_id] = row
    return meta, rows


def audit(path, dataset, dataset_file, prompt_file, generation_file, model_name):
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("Sandbox result must be a regular, non-symlink file")
    if info.st_size <= 0 or info.st_size > 100 * 1024 * 1024:
        raise ValueError(f"Sandbox result has unsafe size: {info.st_size}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta", {})
    expected = {
        "dataset": dataset,
        "dataset_file_sha256": sha256_file(dataset_file),
        "prompt_file_sha256": sha256_file(prompt_file),
        "generation_file_sha256": sha256_file(generation_file),
        "model_name": model_name,
        "evalplus_commit": EVALPLUS_COMMIT,
        "n_tasks": DATASET_COUNTS[dataset],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Sandbox-result metadata mismatch for {key}")
    expected_dataset_hash = md5_file(dataset_file)
    if (
        meta.get("official_dataset_hash") != expected_dataset_hash
        or meta.get("preloaded_dataset_hash") != expected_dataset_hash
    ):
        raise ValueError("Sandbox-result official dataset hash mismatch")
    ids = dataset_ids(dataset_file)
    generation_meta, generations = load_generation(generation_file, dataset)
    if meta.get("generation_manifest_fingerprint") != generation_meta.get(
        "manifest_fingerprint"
    ):
        raise ValueError("Sandbox result generation-manifest fingerprint mismatch")
    if meta.get("model_fingerprint") != generation_meta.get("model_fingerprint"):
        raise ValueError("Sandbox result model fingerprint mismatch")
    if generation_meta.get("model_name") != model_name:
        raise ValueError("Generation model-name mismatch")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or [row.get("task_id") for row in tasks] != ids:
        raise ValueError("Sandbox-result task IDs are incomplete or reordered")
    if len(ids) != len(set(ids)) or len(ids) != DATASET_COUNTS[dataset]:
        raise ValueError("Dataset task IDs are invalid")
    if set(generations) != set(ids):
        raise ValueError("Generation task IDs do not match the evaluator")
    allowed = {"pass", "fail", "timeout"}
    for row in tasks:
        if row.get("base_status") not in allowed or row.get("plus_status") not in allowed:
            raise ValueError(f"Invalid status for {row.get('task_id')}")
        if row.get("original_pass") != (row["base_status"] == "pass"):
            raise ValueError(f"Original-pass flag mismatch for {row['task_id']}")
        strict = row["base_status"] == row["plus_status"] == "pass"
        if row.get("strict_plus_pass") != strict:
            raise ValueError(f"Strict-pass flag mismatch for {row['task_id']}")
        solution = row.get("sanitized_solution")
        if not isinstance(solution, str) or len(solution.encode("utf-8")) > 1024 * 1024:
            raise ValueError(f"Unsafe solution field for {row['task_id']}")
        if hashlib.sha256(solution.encode("utf-8")).hexdigest() != row.get(
            "sanitized_solution_sha256"
        ):
            raise ValueError(f"Sanitized solution checksum mismatch for {row['task_id']}")
        generated = generations[row["task_id"]]
        raw = generated.get("response")
        if not isinstance(raw, str) or hashlib.sha256(raw.encode("utf-8")).hexdigest() != row.get(
            "raw_response_sha256"
        ):
            raise ValueError(f"Raw response checksum mismatch for {row['task_id']}")
        generated_checksum_payload = {
            "question_id": generated.get("question_id"),
            "sample_index": generated.get("sample_index"),
            "response": generated.get("response"),
            "stop_reason": generated.get("stop_reason"),
            "n_generated_tokens": generated.get("n_generated_tokens"),
            "prompt_sha256": generated.get("prompt_sha256"),
        }
        observed_generation_checksum = hashlib.sha256(
            canonical_json(generated_checksum_payload).encode("utf-8")
        ).hexdigest()
        if generated.get("result_sha256") != observed_generation_checksum:
            raise ValueError(f"Generation checksum mismatch for {row['task_id']}")
        expected_generation_meta = {
            "stop_reason": generated.get("stop_reason"),
            "n_generated_tokens": generated.get("n_generated_tokens"),
            "prompt_sha256": generated.get("prompt_sha256"),
        }
        if row.get("generation_meta") != expected_generation_meta:
            raise ValueError(f"Embedded generation metadata mismatch for {row['task_id']}")
        try:
            ast.parse(solution)
            observed_syntax = True
        except (SyntaxError, MemoryError, ValueError):
            observed_syntax = False
        if row.get("syntax_valid") != observed_syntax:
            raise ValueError(f"Syntax-valid flag mismatch for {row['task_id']}")
        if row.get("empty_sanitized_solution") != (not bool(solution.strip())):
            raise ValueError(f"Empty-solution flag mismatch for {row['task_id']}")
        if not isinstance(row.get("suspicious_flags"), list):
            raise ValueError(f"Invalid suspicious flags for {row['task_id']}")
        observed_flags = suspicious_flags(solution)
        if row["suspicious_flags"] != observed_flags:
            raise ValueError(f"Suspicious-code flags mismatch for {row['task_id']}")
        if row.get("quarantined_suspicious") != bool(observed_flags):
            raise ValueError(f"Suspicious quarantine mismatch for {row['task_id']}")
        if observed_flags and (row["original_pass"] or row["strict_plus_pass"]):
            raise ValueError(f"Quarantined suspicious solution passed: {row['task_id']}")
    print(f"Audited EvalPlus result: {dataset}/{model_name}: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_COUNTS), required=True)
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--generation_file", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    audit(
        args.input,
        args.dataset,
        args.dataset_file,
        args.prompt_file,
        args.generation_file,
        args.model_name,
    )


if __name__ == "__main__":
    main()
