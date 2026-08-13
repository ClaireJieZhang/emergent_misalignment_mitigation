#!/usr/bin/env python3
"""Merge quorum chunk outputs only after an exact provenance and ID audit."""

import argparse
import hashlib
import json
import os
import tempfile


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_value(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_digest(record):
    content = dict(record)
    content.pop("result_sha256", None)
    return sha256_value(content)


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


def expected_prompts(prompt_file):
    with open(prompt_file, encoding="utf-8") as handle:
        prompts = json.load(handle)
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt file has no tasks")
    ids_in_order = [str(record["question_id"]) for record in prompts]
    if len(ids_in_order) != len(set(ids_in_order)):
        raise ValueError("Prompt file has duplicate question IDs")
    records = {
        str(record["question_id"]): record.get("prompt_sha256") for record in prompts
    }
    ids = sorted(records)
    if any(not isinstance(records[question_id], str) for question_id in ids):
        raise ValueError("Prompt file lacks prompt hashes")
    return ids, records


def comparable_manifest(manifest):
    ignored = {"selected_questions", "chunk_index", "device"}
    return {key: value for key, value in manifest.items() if key not in ignored}


def merge_chunks(paths, prompt_file, method, q, chunk_count):
    if len(paths) != chunk_count:
        raise ValueError(f"Expected {chunk_count} chunks, got {len(paths)}")
    ids, prompt_hashes = expected_prompts(prompt_file)
    prompt_file_sha = sha256_file(prompt_file)
    chunks = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            chunk = json.load(handle)
        manifest = chunk.get("immutable_manifest")
        samples = chunk.get("samples")
        if not isinstance(manifest, dict) or not isinstance(samples, list):
            raise ValueError(f"Invalid chunk structure: {path}")
        observed_fingerprint = chunk.get("immutable_manifest_sha256")
        if observed_fingerprint != sha256_value(manifest):
            raise ValueError(f"Immutable manifest checksum mismatch: {path}")
        if manifest.get("method") != method or manifest.get("q") != q:
            raise ValueError(f"Method/q mismatch: {path}")
        if manifest.get("chunk_count") != chunk_count:
            raise ValueError(f"chunk_count mismatch: {path}")
        if manifest.get("prompt_file_sha256") != prompt_file_sha:
            raise ValueError(f"Prompt file checksum mismatch: {path}")
        chunks.append((manifest["chunk_index"], path, chunk))
    indices = sorted(index for index, _, _ in chunks)
    if indices != list(range(chunk_count)):
        raise ValueError(f"Chunk indices must be exactly 0..{chunk_count - 1}: {indices}")
    chunks.sort()
    baseline = comparable_manifest(chunks[0][2]["immutable_manifest"])
    for _, path, chunk in chunks[1:]:
        if comparable_manifest(chunk["immutable_manifest"]) != baseline:
            raise ValueError(f"Cross-chunk immutable settings mismatch: {path}")

    samples = []
    keys = set()
    for _, path, chunk in chunks:
        expected_count = chunk.get("completed_samples")
        if expected_count != len(chunk["samples"]):
            raise ValueError(f"Completed-sample count mismatch: {path}")
        for sample in chunk["samples"]:
            key = (str(sample.get("question_id")), sample.get("sample_index"))
            if key in keys:
                raise ValueError(f"Duplicate sample key across chunks: {key}")
            if sample.get("prompt_sha256") != prompt_hashes.get(key[0]):
                raise ValueError(f"Sample prompt hash mismatch in {path}: {key}")
            if sample.get("result_sha256") != result_digest(sample):
                raise ValueError(f"Sample result checksum mismatch in {path}: {key}")
            keys.add(key)
            samples.append(sample)
    n_samples = baseline["n_samples"]
    expected_keys = {(question_id, index) for question_id in ids for index in range(n_samples)}
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)[:5]
        extra = sorted(keys - expected_keys)[:5]
        raise ValueError(f"Merged sample IDs are incomplete; missing={missing}, extra={extra}")
    samples.sort(key=lambda sample: (str(sample["question_id"]), sample["sample_index"]))
    chunk_fingerprints = [
        chunk["immutable_manifest_sha256"] for _, _, chunk in chunks
    ]
    merged_manifest = {
        "schema_version": 1,
        "method": method,
        "q": q,
        "chunk_count": chunk_count,
        "chunk_manifest_fingerprints": chunk_fingerprints,
        "question_ids": ids,
        "n_samples": n_samples,
        "prompt_file_sha256": prompt_file_sha,
        "common_settings": baseline,
    }
    return {
        "meta": {
            **merged_manifest,
            "manifest_fingerprint": sha256_value(merged_manifest),
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", action="append", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--method", choices=["quorum", "pi_quorum_delta"], required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--chunk_count", type=int, required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    payload = merge_chunks(
        [os.path.abspath(path) for path in args.chunk],
        args.prompt_file,
        args.method,
        args.q,
        args.chunk_count,
    )
    atomic_write_json(args.output_file, payload)
    print(f"Wrote and audited {args.output_file}: {len(payload['samples'])} samples")


if __name__ == "__main__":
    main()
