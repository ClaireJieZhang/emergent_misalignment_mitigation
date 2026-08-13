#!/usr/bin/env python3
"""Convert audited generation outputs to LiveCodeBench custom-evaluator inputs."""

import argparse
import hashlib
import json
import os
import tempfile


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def extract_code_qwen_official(model_output):
    """Pinned LCB CodeQwenInstruct extraction: take the final fenced block."""
    output_lines = model_output.split("\n")
    fence_lines = [index for index, line in enumerate(output_lines) if "```" in line]
    if len(fence_lines) < 2:
        return ""
    return "\n".join(output_lines[fence_lines[-2] + 1 : fence_lines[-1]])


def parse_generation_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Generation must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Generation must be NAME=PATH, got {spec!r}")
    return name.strip(), os.path.abspath(path.strip())


def load_expected_prompts(prompt_file):
    with open(prompt_file, encoding="utf-8") as handle:
        prompts = json.load(handle)
    if isinstance(prompts, dict):
        prompts = prompts.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt file must contain a nonempty prompt list")
    ids_in_order = [str(item["question_id"]) for item in prompts]
    if len(ids_in_order) != len(set(ids_in_order)):
        raise ValueError("Prompt file contains duplicate question IDs")
    records = {
        str(item["question_id"]): item.get("prompt_sha256") for item in prompts
    }
    ids = sorted(records)
    if any(not isinstance(records[question_id], str) for question_id in ids):
        raise ValueError("Prompt file lacks prompt hashes")
    return ids, records


def load_generation_samples(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"Generation output has no samples list: {path}")
    return payload.get("meta", {}), samples


def build_custom_output(samples, expected_ids, expected_prompt_hashes=None):
    grouped = {}
    for sample in samples:
        question_id = str(sample.get("question_id"))
        sample_index = sample.get("sample_index")
        response = sample.get("response")
        if question_id not in expected_ids:
            raise ValueError(f"Unexpected question ID in generations: {question_id}")
        if not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError(f"Invalid sample index for {question_id}: {sample_index}")
        if not isinstance(response, str):
            raise ValueError(f"Missing response for {question_id}/{sample_index}")
        if (
            expected_prompt_hashes is not None
            and sample.get("prompt_sha256") != expected_prompt_hashes[question_id]
        ):
            raise ValueError(f"Prompt hash mismatch for {question_id}/{sample_index}")
        key = (question_id, sample_index)
        if key in grouped:
            raise ValueError(f"Duplicate generation key: {key}")
        grouped[key] = sample

    counts = {question_id: 0 for question_id in expected_ids}
    for question_id, _ in grouped:
        counts[question_id] += 1
    unique_counts = set(counts.values())
    if len(unique_counts) != 1 or 0 in unique_counts:
        raise ValueError(f"Every task must have the same positive sample count: {counts}")
    n_samples = unique_counts.pop()

    rows = []
    for question_id in expected_ids:
        code_list = []
        generation_meta = []
        for sample_index in range(n_samples):
            key = (question_id, sample_index)
            if key not in grouped:
                raise ValueError(f"Missing generation key: {key}")
            sample = grouped[key]
            code = extract_code_qwen_official(sample["response"])
            stop_reason = sample.get("stop_reason")
            if stop_reason == "length":
                # The custom quorum sampler uses the Transformers-style name,
                # while the direct vLLM sampler uses this report-wide name.
                stop_reason = "max_new_tokens"
            n_generated_tokens = sample.get("n_generated_tokens")
            if n_generated_tokens is None:
                n_generated_tokens = sample.get("n_response_tokens")
            code_list.append(code)
            generation_meta.append(
                {
                    "sample_index": sample_index,
                    "stop_reason": stop_reason,
                    "n_generated_tokens": n_generated_tokens,
                    "raw_response_sha256": sha256_text(sample["response"]),
                    "extracted_code_sha256": sha256_text(code),
                    "empty_extraction": not bool(code.strip()),
                }
            )
        rows.append(
            {
                "question_id": question_id,
                "code_list": code_list,
                "generation_meta": generation_meta,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", action="append", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    expected_ids, expected_prompt_hashes = load_expected_prompts(args.prompt_file)
    prompt_file_sha = sha256_file(args.prompt_file)
    os.makedirs(args.output_dir, exist_ok=True)
    seen_names = set()
    for spec in args.generation:
        name, path = parse_generation_spec(spec)
        if name in seen_names:
            raise ValueError(f"Duplicate generation name: {name}")
        seen_names.add(name)
        source_meta, samples = load_generation_samples(path)
        source_prompt_sha = source_meta.get("prompt_file_sha256")
        if source_prompt_sha != prompt_file_sha:
            raise ValueError(
                f"Generation {name} was not produced from the current prompt file"
            )
        rows = build_custom_output(samples, expected_ids, expected_prompt_hashes)
        output = [
            {"question_id": row["question_id"], "code_list": row["code_list"]}
            for row in rows
        ]
        output_meta = {
            "schema_version": 1,
            "name": name,
            "source_generation": path,
            "source_manifest_fingerprint": source_meta.get("manifest_fingerprint"),
            "prompt_file_sha256": prompt_file_sha,
            "n_questions": len(rows),
            "n_samples": len(rows[0]["code_list"]),
            "extractor": "LCB@28fef95 CodeQwenInstruct final fenced block",
            "generation_meta": {
                row["question_id"]: row["generation_meta"] for row in rows
            },
        }
        destination = os.path.join(args.output_dir, f"{name}.custom.json")
        atomic_write_json(destination, output)
        atomic_write_json(
            os.path.join(args.output_dir, f"{name}.custom.meta.json"),
            output_meta,
        )
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
