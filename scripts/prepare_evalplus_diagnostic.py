#!/usr/bin/env python3
"""Prepare and audit the sealed HumanEval+/MBPP+ diagnostic prompt bank.

The model-facing prompt file intentionally contains no tests, canonical
solutions, or evaluator inputs.  The original release assets remain separate
and are mounted only inside the execution sandbox.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict


DATASETS = {
    "humaneval": {
        "filename": "HumanEvalPlus-v0.1.10.jsonl.gz",
        "version": "v0.1.10",
        "count": 164,
        "prefix": "HumanEval/",
        "sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "uncompressed_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
    },
    "mbpp": {
        "filename": "MbppPlus-v0.2.0.jsonl.gz",
        "version": "v0.2.0",
        "count": 378,
        "prefix": "Mbpp/",
        "sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        "uncompressed_sha256": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
    },
}
EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
EVALPLUS_TAG = "v0.3.1"
NEAR_DUPLICATE_THRESHOLD = 0.80
PROMPT_FIELDS = {
    "question_id",
    "dataset",
    "prompt",
    "entry_point",
    "prompt_sha256",
    "training_overlap_exact",
    "training_overlap_near",
    "max_training_word_5gram_jaccard",
    "closest_training_source",
    "pilot_shard_overlap_near",
    "pilot_shard_max_word_5gram_jaccard",
}
HIDDEN_FIELDS = {
    "canonical_solution",
    "base_input",
    "plus_input",
    "test",
    "contract",
    "assertion",
    "atol",
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


def normalize_prompt(text):
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def word_ngrams(text, width=5):
    tokens = normalize_prompt(text).split()
    if not tokens:
        return set()
    if len(tokens) < width:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def prompt_hash(record):
    return sha256_bytes(
        canonical_json(
            {
                "dataset": record["dataset"],
                "question_id": record["question_id"],
                "prompt": record["prompt"],
                "entry_point": record["entry_point"],
            }
        ).encode("utf-8")
    )


def load_release(path, dataset):
    spec = DATASETS[dataset]
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Compressed {dataset} release SHA-256 mismatch: {path}")
    uncompressed_digest = hashlib.sha256()
    rows = []
    with gzip.open(path, "rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            uncompressed_digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid {dataset} JSONL line {line_number}: {error}") from error
            rows.append(row)
    if uncompressed_digest.hexdigest() != spec["uncompressed_sha256"]:
        raise ValueError(f"Uncompressed {dataset} release SHA-256 mismatch: {path}")
    if len(rows) != spec["count"]:
        raise ValueError(f"Expected {spec['count']} {dataset} rows, found {len(rows)}")
    seen = set()
    safe = []
    for row in rows:
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        entry_point = row.get("entry_point")
        if not isinstance(task_id, str) or not task_id.startswith(spec["prefix"]):
            raise ValueError(f"Invalid {dataset} task ID: {task_id!r}")
        if task_id in seen:
            raise ValueError(f"Duplicate {dataset} task ID: {task_id}")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(entry_point, str):
            raise ValueError(f"Invalid model-facing fields for {task_id}")
        required_evaluator_fields = {
            "canonical_solution",
            "base_input",
            "plus_input",
            "contract",
            "atol",
            "test" if dataset == "humaneval" else "assertion",
        }
        missing = required_evaluator_fields.difference(row)
        if missing:
            raise ValueError(f"Pinned release row {task_id} lacks evaluator fields: {sorted(missing)}")
        seen.add(task_id)
        record = {
            "question_id": task_id,
            "dataset": dataset,
            "prompt": prompt,
            "entry_point": entry_point,
        }
        record["prompt_sha256"] = prompt_hash(record)
        safe.append(record)
    return safe


def audit_training_overlap(prompts, shard_paths):
    """Find exact and high word-5-gram-Jaccard prompt overlaps."""
    if not shard_paths:
        raise ValueError("At least one frozen training shard is required")
    from datasets import load_from_disk

    eval_grams = [word_ngrams(record["prompt"]) for record in prompts]
    gram_to_eval = defaultdict(list)
    exact_to_eval = defaultdict(list)
    for index, record in enumerate(prompts):
        for gram in eval_grams[index]:
            gram_to_eval[gram].append(index)
        exact_to_eval[normalize_prompt(record["prompt"])].append(index)

    best = [0.0] * len(prompts)
    best_source = [None] * len(prompts)
    exact = [False] * len(prompts)
    total_training = 0
    shard_counts = []
    per_shard_best = {
        os.path.basename(os.path.abspath(path)): [0.0] * len(prompts)
        for path in shard_paths
    }
    for shard_path in shard_paths:
        dataset = load_from_disk(shard_path)
        if "prompt" not in dataset.column_names or len(dataset) != 6000:
            raise ValueError(f"Expected a 6,000-row prompt-bearing shard: {shard_path}")
        shard_counts.append(len(dataset))
        shard_name = os.path.basename(os.path.abspath(shard_path))
        for row_index, row in enumerate(dataset):
            total_training += 1
            train_prompt = row["prompt"]
            normalized = normalize_prompt(train_prompt)
            source = {
                "shard": os.path.basename(os.path.abspath(shard_path)),
                "row_index": row_index,
                "source_index": row.get("source_index"),
            }
            for eval_index in exact_to_eval.get(normalized, []):
                exact[eval_index] = True
                best[eval_index] = 1.0
                best_source[eval_index] = source
                per_shard_best[shard_name][eval_index] = 1.0
            train_grams = word_ngrams(train_prompt)
            intersections = defaultdict(int)
            for gram in train_grams:
                for eval_index in gram_to_eval.get(gram, []):
                    intersections[eval_index] += 1
            for eval_index, intersection in intersections.items():
                denominator = len(train_grams) + len(eval_grams[eval_index]) - intersection
                similarity = 0.0 if denominator == 0 else intersection / denominator
                if similarity > best[eval_index]:
                    best[eval_index] = similarity
                    best_source[eval_index] = source
                if similarity > per_shard_best[shard_name][eval_index]:
                    per_shard_best[shard_name][eval_index] = similarity

    for index, record in enumerate(prompts):
        record["training_overlap_exact"] = exact[index]
        record["training_overlap_near"] = best[index] >= NEAR_DUPLICATE_THRESHOLD
        record["max_training_word_5gram_jaccard"] = round(best[index], 8)
        record["closest_training_source"] = best_source[index]
        pilot_values = per_shard_best.get("magicoder_python_shard_000")
        if pilot_values is None:
            raise ValueError("The overlap audit lacks the pilot's shard 000")
        record["pilot_shard_max_word_5gram_jaccard"] = round(
            pilot_values[index], 8
        )
        record["pilot_shard_overlap_near"] = (
            pilot_values[index] >= NEAR_DUPLICATE_THRESHOLD
        )
    return {
        "method": "normalized exact match and maximum word-5-gram Jaccard",
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "training_rows": total_training,
        "shard_counts": shard_counts,
        "exact_overlap_count": sum(record["training_overlap_exact"] for record in prompts),
        "near_overlap_count": sum(record["training_overlap_near"] for record in prompts),
        "per_shard_near_overlap_count": {
            name: sum(value >= NEAR_DUPLICATE_THRESHOLD for value in values)
            for name, values in per_shard_best.items()
        },
    }


def validate_prompt_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {"meta", "prompts"}:
        raise ValueError("Prompt payload must contain exactly meta and prompts")
    prompts = payload["prompts"]
    if not isinstance(prompts, list) or len(prompts) != sum(x["count"] for x in DATASETS.values()):
        raise ValueError("Prompt payload has the wrong record count")
    seen = set()
    counts = defaultdict(int)
    for record in prompts:
        if set(record) != PROMPT_FIELDS or HIDDEN_FIELDS.intersection(record):
            raise ValueError(f"Prompt record has unsafe fields: {set(record)}")
        question_id = record["question_id"]
        if question_id in seen or record["dataset"] not in DATASETS:
            raise ValueError(f"Duplicate or invalid prompt record: {question_id}")
        if prompt_hash(record) != record["prompt_sha256"]:
            raise ValueError(f"Prompt SHA-256 mismatch: {question_id}")
        seen.add(question_id)
        counts[record["dataset"]] += 1
    for dataset, spec in DATASETS.items():
        if counts[dataset] != spec["count"]:
            raise ValueError(f"Prompt count mismatch for {dataset}")
    return prompts


def audit_existing(output_root):
    manifest_path = os.path.join(output_root, "data_manifest.json")
    prompt_path = os.path.join(output_root, "evalplus_prompts.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    seal = manifest.pop("manifest_payload_sha256", None)
    expected_seal = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    if seal != expected_seal:
        raise ValueError("EvalPlus data-manifest seal mismatch")
    if manifest.get("evalplus", {}).get("commit") != EVALPLUS_COMMIT:
        raise ValueError("EvalPlus commit mismatch in data manifest")
    for dataset, record in manifest["datasets"].items():
        if dataset not in DATASETS or sha256_file(record["path"]) != DATASETS[dataset]["sha256"]:
            raise ValueError(f"Dataset asset drift: {dataset}")
    if sha256_file(prompt_path) != manifest["prompt_file"]["sha256"]:
        raise ValueError("Model-facing prompt file drift")
    with open(prompt_path, encoding="utf-8") as handle:
        prompt_payload = json.load(handle)
    validate_prompt_payload(prompt_payload)
    print(f"EvalPlus diagnostic data audit passed: {output_root}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--humaneval_file")
    parser.add_argument("--mbpp_file")
    parser.add_argument("--training_shard", action="append", default=[])
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    if args.audit_only:
        audit_existing(output_root)
        return
    if not args.humaneval_file or not args.mbpp_file or len(args.training_shard) != 3:
        parser.error("preparation requires both release files and exactly three training shards")

    dataset_paths = {
        "humaneval": os.path.abspath(args.humaneval_file),
        "mbpp": os.path.abspath(args.mbpp_file),
    }
    prompts = []
    for dataset in ("humaneval", "mbpp"):
        prompts.extend(load_release(dataset_paths[dataset], dataset))
    overlap = audit_training_overlap(prompts, [os.path.abspath(x) for x in args.training_shard])
    prompt_payload = {
        "meta": {
            "schema_version": 1,
            "evalplus_commit": EVALPLUS_COMMIT,
            "evalplus_tag": EVALPLUS_TAG,
            "dataset_versions": {name: spec["version"] for name, spec in DATASETS.items()},
            "dataset_counts": {name: spec["count"] for name, spec in DATASETS.items()},
            "training_overlap": overlap,
            "contains_hidden_tests": False,
        },
        "prompts": prompts,
    }
    validate_prompt_payload(prompt_payload)
    os.makedirs(output_root, exist_ok=True)
    prompt_path = os.path.join(output_root, "evalplus_prompts.json")
    atomic_write_json(prompt_path, prompt_payload)
    manifest = {
        "schema_version": 1,
        "evalplus": {"tag": EVALPLUS_TAG, "commit": EVALPLUS_COMMIT},
        "datasets": {
            name: {
                "path": dataset_paths[name],
                "version": spec["version"],
                "count": spec["count"],
                "sha256": spec["sha256"],
                "uncompressed_sha256": spec["uncompressed_sha256"],
            }
            for name, spec in DATASETS.items()
        },
        "training_shards": [os.path.abspath(path) for path in args.training_shard],
        "training_overlap": overlap,
        "prompt_file": {
            "path": prompt_path,
            "sha256": sha256_file(prompt_path),
            "count": len(prompts),
        },
    }
    manifest["manifest_payload_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    atomic_write_json(os.path.join(output_root, "data_manifest.json"), manifest)
    audit_existing(output_root)
    print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
