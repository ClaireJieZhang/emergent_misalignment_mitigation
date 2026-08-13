#!/usr/bin/env python3
"""Prepare auditable Magicoder training shards and held-out LCB prompt windows.

LiveCodeBench rows contain tests and are therefore written to separate
evaluator-only JSONL files.  Model-facing prompt files contain no test fields.
This script downloads data and transforms it, but never executes generated code.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import tempfile
import unicodedata
import uuid


MAGICODER_DATASET_ID = "ise-uiuc/Magicoder-OSS-Instruct-75K"
MAGICODER_REVISION = "5f839b1f368a76b161028bb9edff055db34022b2"
LCB_DATASET_ID = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_SOURCE_FILES = ("test5.jsonl", "test6.jsonl")

GATE_WINDOW = {
    "name": "gate",
    "start_date": "2024-10-01",
    "end_date": "2024-12-31",
    "expected_count": 157,
}
FINAL_WINDOW = {
    "name": "final",
    "start_date": "2025-01-01",
    "end_date": "2025-04-30",
    "expected_count": 182,
}
LCB_WINDOWS = (GATE_WINDOW, FINAL_WINDOW)

LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)

MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def seal_manifest(manifest):
    sealed = dict(manifest)
    sealed.pop(MANIFEST_SEAL_FIELD, None)
    sealed[MANIFEST_SEAL_FIELD] = sha256_bytes(canonical_json_bytes(sealed))
    return sealed


def verify_manifest_seal(manifest):
    unsealed = dict(manifest)
    recorded = unsealed.pop(MANIFEST_SEAL_FIELD, None)
    if recorded != sha256_bytes(canonical_json_bytes(unsealed)):
        raise ValueError("Existing output manifest failed its integrity seal")


def normalize_text(value):
    if not isinstance(value, str):
        raise ValueError(f"Expected text, got {type(value).__name__}")
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace(
        "\r", "\n"
    ).strip()


def magicoder_pair_sha256(prompt, response):
    return sha256_bytes(
        canonical_json_bytes({"prompt": prompt, "response": response})
    )


def prepare_magicoder_rows(rows, seed, n_shards, shard_size):
    """Normalize, deduplicate, permute, and shard Magicoder fixture/source rows."""
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    candidates = []
    seen_pair_hashes = set()
    duplicate_pair_hashes = []
    n_source = 0
    n_python = 0
    for source_index, row in enumerate(rows):
        n_source += 1
        if not isinstance(row, dict):
            raise ValueError(f"Magicoder row {source_index} is not an object")
        lang = row.get("lang")
        if not isinstance(lang, str) or lang.strip().casefold() != "python":
            continue
        n_python += 1
        try:
            prompt = normalize_text(row.get("problem"))
            response = normalize_text(row.get("solution"))
        except ValueError as error:
            raise ValueError(
                f"Magicoder Python row {source_index} has invalid problem/solution: {error}"
            ) from error
        if not prompt or not response:
            raise ValueError(
                f"Magicoder Python row {source_index} has an empty problem/solution"
            )
        pair_hash = magicoder_pair_sha256(prompt, response)
        if pair_hash in seen_pair_hashes:
            duplicate_pair_hashes.append(pair_hash)
            continue
        seen_pair_hashes.add(pair_hash)
        candidates.append(
            {
                "prompt": prompt,
                "response": response,
                "source_index": source_index,
                "source_sha256": pair_hash,
                "lang": "python",
            }
        )

    required = n_shards * shard_size
    if len(candidates) < required:
        raise ValueError(
            f"Need {required} unique nonempty Python rows for {n_shards} shards x "
            f"{shard_size}, but found {len(candidates)}"
        )
    permutation = list(range(len(candidates)))
    random.Random(seed).shuffle(permutation)
    selected = [candidates[index] for index in permutation[:required]]
    shards = [
        selected[index * shard_size : (index + 1) * shard_size]
        for index in range(n_shards)
    ]
    summary = {
        "source_row_count": n_source,
        "python_row_count": n_python,
        "unique_python_pair_count": len(candidates),
        "duplicate_python_pair_count_rejected": len(duplicate_pair_hashes),
        "duplicate_python_pair_hashes": sorted(set(duplicate_pair_hashes)),
        "selected_row_count": len(selected),
    }
    return shards, summary


def parse_contest_date(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid contest_date: {value!r}")
    date_text = value.strip().split("T", 1)[0]
    try:
        return dt.date.fromisoformat(date_text)
    except ValueError as error:
        raise ValueError(f"Invalid contest_date: {value!r}") from error


def window_for_date(contest_date, windows=None):
    if windows is None:
        windows = LCB_WINDOWS
    for window in windows:
        start = dt.date.fromisoformat(window["start_date"])
        end = dt.date.fromisoformat(window["end_date"])
        if start <= contest_date <= end:
            return window["name"]
    return None


def format_lcb_prompt(question_content, starter_code):
    if not isinstance(question_content, str) or not question_content.strip():
        raise ValueError("LiveCodeBench question_content must be nonempty text")
    if starter_code is None:
        starter_code = ""
    if not isinstance(starter_code, str):
        raise ValueError("LiveCodeBench starter_code must be text or null")
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def validate_lcb_row(row, source_label, line_number):
    if not isinstance(row, dict):
        raise ValueError(f"{source_label} line {line_number} is not an object")
    required = {
        "question_id",
        "question_content",
        "contest_date",
        "difficulty",
        "platform",
        "starter_code",
        "public_test_cases",
        "private_test_cases",
        "metadata",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(
            f"{source_label} line {line_number} is missing fields: {missing}"
        )
    question_id = row["question_id"]
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError(
            f"{source_label} line {line_number} has invalid question_id"
        )
    parse_contest_date(row["contest_date"])
    format_lcb_prompt(row["question_content"], row["starter_code"])
    for field in ("difficulty", "platform"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(
                f"{source_label} line {line_number} has invalid {field}"
            )
    return question_id


def lcb_prompt_record(row):
    starter_code = row["starter_code"] or ""
    prompt = format_lcb_prompt(row["question_content"], starter_code)
    return {
        "prompt": prompt,
        "system": LCB_SYSTEM,
        "question_id": row["question_id"],
        "date": parse_contest_date(row["contest_date"]).isoformat(),
        "difficulty": row["difficulty"],
        "platform": row["platform"],
        "starter_code": starter_code,
        "prompt_sha256": sha256_bytes(
            canonical_json_bytes({"system": LCB_SYSTEM, "prompt": prompt})
        ),
    }


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path):
    digest = hashlib.sha256()
    for root, directories, files in os.walk(path):
        for name in directories:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                raise ValueError(f"Refusing to fingerprint symlink: {full_path}")
        directories.sort()
        files.sort()
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                raise ValueError(f"Refusing to fingerprint symlink: {full_path}")
            relative = os.path.relpath(full_path, path).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(os.path.getsize(full_path)).encode("ascii"))
            digest.update(b"\0")
            with open(full_path, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def requested_config(args):
    return {
        "magicoder_dataset_id": args.magicoder_dataset_id,
        "magicoder_revision": args.magicoder_revision,
        "lcb_dataset_id": args.lcb_dataset_id,
        "lcb_revision": args.lcb_revision,
        "lcb_source_files": list(LCB_SOURCE_FILES),
        "selection_seed": args.seed,
        "n_shards": args.n_shards,
        "shard_size": args.shard_size,
        "lcb_windows": [dict(window) for window in LCB_WINDOWS],
    }


def logical_shard_sha256(rows):
    return sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "source_index": row["source_index"],
                    "source_sha256": row["source_sha256"],
                }
                for row in rows
            ]
        )
    )


def verify_manifest_overlap_proof(magicoder_manifest):
    shards = magicoder_manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("Manifest Magicoder shards are missing")
    sets = []
    for expected_index, shard in enumerate(shards):
        if shard.get("shard_index") != expected_index:
            raise ValueError("Manifest Magicoder shard indices are incompatible")
        hashes = shard.get("ordered_source_sha256")
        indices = shard.get("ordered_source_indices")
        if not isinstance(hashes, list) or not isinstance(indices, list):
            raise ValueError("Manifest Magicoder shard provenance is missing")
        if len(hashes) != len(indices) or len(hashes) != shard.get("row_count"):
            raise ValueError("Manifest Magicoder shard provenance length is invalid")
        if len(set(hashes)) != len(hashes):
            raise ValueError("Manifest contains a within-shard duplicate")
        expected_logical = sha256_bytes(
            canonical_json_bytes(
                [
                    {"source_index": index, "source_sha256": source_hash}
                    for index, source_hash in zip(indices, hashes)
                ]
            )
        )
        if expected_logical != shard.get("logical_sha256"):
            raise ValueError("Manifest Magicoder logical fingerprint is invalid")
        sets.append(set(hashes))
    computed = {}
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            key = f"{left:03d}-{right:03d}"
            computed[key] = len(sets[left] & sets[right])
    if computed != magicoder_manifest.get("pairwise_overlap_counts"):
        raise ValueError("Manifest Magicoder pairwise overlap proof is invalid")
    if any(computed.values()):
        raise ValueError("Manifest Magicoder shards overlap")


def verify_benchmark_manifest(benchmark_manifest):
    prompt_constants = {
        "format_with_starter_code": LCB_FORMAT_WITH_STARTER,
        "format_without_starter_code": LCB_FORMAT_WITHOUT_STARTER,
    }
    if (
        benchmark_manifest.get("system_message") != LCB_SYSTEM
        or benchmark_manifest.get("system_sha256") != sha256_text(LCB_SYSTEM)
    ):
        raise ValueError("Manifest LiveCodeBench system message is incompatible")
    if (
        benchmark_manifest.get("generic_prompt_constants") != prompt_constants
        or benchmark_manifest.get("generic_prompt_constants_sha256")
        != sha256_bytes(canonical_json_bytes(prompt_constants))
    ):
        raise ValueError("Manifest LiveCodeBench prompt constants are incompatible")
    windows = benchmark_manifest.get("windows")
    if not isinstance(windows, dict):
        raise ValueError("Manifest LiveCodeBench windows are missing")
    id_sets = {}
    for expected in LCB_WINDOWS:
        name = expected["name"]
        entry = windows.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest LiveCodeBench {name} window is missing")
        for key in ("start_date", "end_date", "expected_count"):
            if entry.get(key) != expected[key]:
                raise ValueError(
                    f"Manifest LiveCodeBench {name} {key} is incompatible"
                )
        question_ids = entry.get("question_ids")
        prompt_hashes = entry.get("ordered_prompt_sha256")
        if not isinstance(question_ids, list) or not isinstance(prompt_hashes, list):
            raise ValueError(
                f"Manifest LiveCodeBench {name} provenance is missing"
            )
        if len(question_ids) != expected["expected_count"]:
            raise ValueError(f"Manifest LiveCodeBench {name} count is invalid")
        if len(set(question_ids)) != len(question_ids):
            raise ValueError(f"Manifest LiveCodeBench {name} IDs are not unique")
        if len(prompt_hashes) != len(question_ids):
            raise ValueError(
                f"Manifest LiveCodeBench {name} prompt hashes are incomplete"
            )
        if sha256_bytes(canonical_json_bytes(question_ids)) != entry.get(
            "question_ids_sha256"
        ):
            raise ValueError(
                f"Manifest LiveCodeBench {name} ID fingerprint is invalid"
            )
        id_sets[name] = set(question_ids)
    overlap = id_sets["gate"] & id_sets["final"]
    if overlap or benchmark_manifest.get(
        "gate_final_question_id_overlap_count"
    ) != len(overlap):
        raise ValueError("Manifest LiveCodeBench gate/final overlap proof is invalid")


def audit_existing_output(output_root, config):
    manifest_path = os.path.join(output_root, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise ValueError(
            f"Output root exists without {MANIFEST_NAME}: {output_root}; use --force "
            "to rebuild it"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Existing output uses an incompatible manifest schema")
    verify_manifest_seal(manifest)
    if manifest.get("config") != config:
        raise ValueError(
            "Existing output was built with incompatible arguments; use --force "
            "to rebuild it"
        )
    verify_manifest_overlap_proof(manifest.get("magicoder", {}))
    verify_benchmark_manifest(manifest.get("livecodebench", {}))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Existing output manifest has no artifacts")
    for label, artifact in sorted(artifacts.items()):
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str) or os.path.isabs(relative_path):
            raise ValueError(f"Invalid artifact path for {label}")
        path = os.path.join(output_root, relative_path)
        if artifact.get("kind") == "file":
            actual = hash_file(path)
        elif artifact.get("kind") == "directory":
            actual = hash_directory(path)
        else:
            raise ValueError(f"Invalid artifact kind for {label}")
        if actual != artifact.get("sha256"):
            raise ValueError(f"Artifact failed exact hash audit: {label}")
    return manifest


def save_magicoder_shards(staging_root, shards, source_fingerprint, summary):
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("Install the 'datasets' package to prepare Magicoder") from error

    shard_manifests = []
    artifacts = {}
    all_hash_sets = []
    for shard_index, rows in enumerate(shards):
        name = f"magicoder_python_shard_{shard_index:03d}"
        path = os.path.join(staging_root, name)
        dataset = Dataset.from_list(rows)
        hf_fingerprint = dataset._fingerprint
        dataset.save_to_disk(path)
        row_hashes = [row["source_sha256"] for row in rows]
        all_hash_sets.append(set(row_hashes))
        shard_manifest = {
            "shard_index": shard_index,
            "path": name,
            "row_count": len(rows),
            "ordered_source_indices": [row["source_index"] for row in rows],
            "ordered_source_sha256": row_hashes,
            "logical_sha256": logical_shard_sha256(rows),
            "hf_dataset_fingerprint": hf_fingerprint,
            "directory_sha256": hash_directory(path),
        }
        shard_manifests.append(shard_manifest)
        artifacts[name] = {
            "kind": "directory",
            "path": name,
            "sha256": shard_manifest["directory_sha256"],
            "row_count": len(rows),
        }

    overlap_counts = {}
    for left in range(len(all_hash_sets)):
        for right in range(left + 1, len(all_hash_sets)):
            overlap_counts[f"{left:03d}-{right:03d}"] = len(
                all_hash_sets[left] & all_hash_sets[right]
            )
    if any(overlap_counts.values()):
        raise AssertionError("Magicoder shards unexpectedly overlap")
    manifest = {
        "source_dataset_fingerprint": source_fingerprint,
        **summary,
        "shards": shard_manifests,
        "pairwise_overlap_counts": overlap_counts,
    }
    return manifest, artifacts


def write_lcb_artifacts(staging_root, source_paths):
    prompt_records = {window["name"]: [] for window in LCB_WINDOWS}
    evaluator_paths = {
        window["name"]: os.path.join(
            staging_root, f"lcb_{window['name']}_evaluator.jsonl"
        )
        for window in LCB_WINDOWS
    }
    evaluator_handles = {
        name: open(path, "wb") for name, path in evaluator_paths.items()
    }
    seen_question_ids = set()
    source_manifest = []
    try:
        for source_name, source_path in source_paths:
            digest = hashlib.sha256()
            source_row_count = 0
            selected_counts = {window["name"]: 0 for window in LCB_WINDOWS}
            with open(source_path, "rb") as source_handle:
                for line_number, raw_line in enumerate(source_handle, start=1):
                    digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    source_row_count += 1
                    try:
                        row = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"Invalid JSON in {source_name} line {line_number}: {error}"
                        ) from error
                    question_id = validate_lcb_row(row, source_name, line_number)
                    if question_id in seen_question_ids:
                        raise ValueError(
                            f"Duplicate LiveCodeBench question_id: {question_id}"
                        )
                    seen_question_ids.add(question_id)
                    window_name = window_for_date(parse_contest_date(row["contest_date"]))
                    if window_name is None:
                        continue
                    prompt_records[window_name].append(lcb_prompt_record(row))
                    evaluator_handles[window_name].write(raw_line.rstrip(b"\r\n") + b"\n")
                    selected_counts[window_name] += 1
            source_manifest.append(
                {
                    "filename": source_name,
                    "sha256": digest.hexdigest(),
                    "row_count": source_row_count,
                    "selected_counts": selected_counts,
                }
            )
    finally:
        for handle in evaluator_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    benchmark_windows = {}
    artifacts = {}
    for window in LCB_WINDOWS:
        name = window["name"]
        records = prompt_records[name]
        if len(records) != window["expected_count"]:
            raise ValueError(
                f"LCB {name} window expected {window['expected_count']} rows but found "
                f"{len(records)}; check the pinned source revision and date parsing"
            )
        question_ids = [record["question_id"] for record in records]
        if len(set(question_ids)) != len(question_ids):
            raise AssertionError(f"Duplicate IDs within LCB {name} window")
        prompt_path = os.path.join(staging_root, f"lcb_{name}_prompts.json")
        prompt_payload = {
            "meta": {
                "benchmark": "LiveCodeBench code_generation_lite",
                "window": dict(window),
                "system_sha256": sha256_text(LCB_SYSTEM),
                "contains_tests": False,
            },
            "prompts": records,
        }
        atomic_write_json(prompt_path, prompt_payload)
        evaluator_path = evaluator_paths[name]
        prompt_sha = hash_file(prompt_path)
        evaluator_sha = hash_file(evaluator_path)
        benchmark_windows[name] = {
            **dict(window),
            "question_ids": question_ids,
            "ordered_prompt_sha256": [
                record["prompt_sha256"] for record in records
            ],
            "question_ids_sha256": sha256_bytes(canonical_json_bytes(question_ids)),
            "prompt_file": os.path.basename(prompt_path),
            "prompt_file_sha256": prompt_sha,
            "evaluator_file": os.path.basename(evaluator_path),
            "evaluator_file_sha256": evaluator_sha,
            "evaluator_contains_full_original_rows": True,
        }
        artifacts[f"lcb_{name}_prompts"] = {
            "kind": "file",
            "path": os.path.basename(prompt_path),
            "sha256": prompt_sha,
            "row_count": len(records),
        }
        artifacts[f"lcb_{name}_evaluator"] = {
            "kind": "file",
            "path": os.path.basename(evaluator_path),
            "sha256": evaluator_sha,
            "row_count": len(records),
        }

    gate_ids = set(benchmark_windows["gate"]["question_ids"])
    final_ids = set(benchmark_windows["final"]["question_ids"])
    overlap = sorted(gate_ids & final_ids)
    if overlap:
        raise AssertionError(f"Gate and final LCB windows overlap: {overlap[:5]}")
    benchmark_manifest = {
        "source_files": source_manifest,
        "system_message": LCB_SYSTEM,
        "system_sha256": sha256_text(LCB_SYSTEM),
        "generic_prompt_constants": {
            "format_with_starter_code": LCB_FORMAT_WITH_STARTER,
            "format_without_starter_code": LCB_FORMAT_WITHOUT_STARTER,
        },
        "generic_prompt_constants_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "format_with_starter_code": LCB_FORMAT_WITH_STARTER,
                    "format_without_starter_code": LCB_FORMAT_WITHOUT_STARTER,
                }
            )
        ),
        "generic_question_template_source": (
            "LiveCodeBench lcb_runner/prompts/code_generation.py"
        ),
        "windows": benchmark_windows,
        "gate_final_question_id_overlap_count": 0,
    }
    return benchmark_manifest, artifacts


def commit_staging_directory(staging_root, output_root, force):
    if not os.path.exists(output_root):
        os.replace(staging_root, output_root)
        return
    if not force:
        raise AssertionError("Existing output should have been audited before building")
    if not os.path.isdir(output_root) or os.path.islink(output_root):
        raise ValueError(f"Refusing to replace non-directory output root: {output_root}")
    backup = f"{output_root}.replaced-{uuid.uuid4().hex}"
    os.replace(output_root, backup)
    try:
        os.replace(staging_root, output_root)
    except BaseException:
        os.replace(backup, output_root)
        raise
    shutil.rmtree(backup)


def build_outputs(args):
    output_root = os.path.abspath(args.output_root)
    config = requested_config(args)
    if os.path.lexists(output_root) and (
        os.path.islink(output_root) or not os.path.isdir(output_root)
    ):
        raise ValueError(
            f"Output root must be a real directory when it exists: {output_root}"
        )
    if os.path.exists(output_root) and not args.force:
        manifest = audit_existing_output(output_root, config)
        print(
            f"Existing compatible output passed exact audit: {output_root} "
            f"({len(manifest['artifacts'])} artifacts)"
        )
        return manifest

    parent = os.path.dirname(output_root)
    os.makedirs(parent, exist_ok=True)
    staging_root = tempfile.mkdtemp(
        prefix=f".{os.path.basename(output_root)}.building-", dir=parent
    )
    try:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Install the 'datasets' package to prepare Magicoder"
            ) from error
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Install the 'huggingface_hub' package to download LiveCodeBench"
            ) from error

        magicoder = load_dataset(
            args.magicoder_dataset_id,
            split="train",
            revision=args.magicoder_revision,
        )
        source_fingerprint = getattr(magicoder, "_fingerprint", None)
        shards, magicoder_summary = prepare_magicoder_rows(
            magicoder,
            seed=args.seed,
            n_shards=args.n_shards,
            shard_size=args.shard_size,
        )
        magicoder_manifest, shard_artifacts = save_magicoder_shards(
            staging_root,
            shards,
            source_fingerprint,
            magicoder_summary,
        )

        lcb_source_paths = []
        for filename in LCB_SOURCE_FILES:
            path = hf_hub_download(
                repo_id=args.lcb_dataset_id,
                filename=filename,
                repo_type="dataset",
                revision=args.lcb_revision,
            )
            lcb_source_paths.append((filename, path))
        benchmark_manifest, benchmark_artifacts = write_lcb_artifacts(
            staging_root, lcb_source_paths
        )
        manifest = seal_manifest({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "config": config,
            "magicoder": magicoder_manifest,
            "livecodebench": benchmark_manifest,
            "artifacts": {**shard_artifacts, **benchmark_artifacts},
        })
        verify_manifest_overlap_proof(manifest["magicoder"])
        verify_benchmark_manifest(manifest["livecodebench"])
        atomic_write_json(os.path.join(staging_root, MANIFEST_NAME), manifest)
        commit_staging_directory(staging_root, output_root, args.force)
        staging_root = None
        print(
            f"Prepared {args.n_shards} disjoint Magicoder shards and "
            f"{sum(window['expected_count'] for window in LCB_WINDOWS)} LCB prompts "
            f"at {output_root}"
        )
        return manifest
    finally:
        if staging_root is not None and os.path.isdir(staging_root):
            shutil.rmtree(staging_root)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare three disjoint Magicoder Python shards and disjoint "
            "post-release LiveCodeBench prompt/evaluator windows."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--magicoder-dataset-id", default=MAGICODER_DATASET_ID)
    parser.add_argument("--magicoder-revision", default=MAGICODER_REVISION)
    parser.add_argument("--lcb-dataset-id", default=LCB_DATASET_ID)
    parser.add_argument("--lcb-revision", default=LCB_REVISION)
    parser.add_argument("--seed", type=int, default=7302026)
    parser.add_argument("--shard-size", type=int, default=6000)
    parser.add_argument("--n-shards", type=int, default=3)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Require an existing output and verify its sealed manifest and every artifact.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        if args.force:
            parser.error("--audit-only and --force are mutually exclusive")
        output_root = os.path.abspath(args.output_root)
        if not os.path.isdir(output_root) or os.path.islink(output_root):
            raise ValueError(f"No real prepared output directory to audit: {output_root}")
        manifest = audit_existing_output(output_root, requested_config(args))
        print(
            f"Exact data audit passed: {output_root} "
            f"({len(manifest['artifacts'])} artifacts)"
        )
        return
    build_outputs(args)


if __name__ == "__main__":
    main()
