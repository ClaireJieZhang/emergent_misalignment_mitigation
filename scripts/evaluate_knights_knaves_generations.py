#!/usr/bin/env python3
"""Strictly score direct-answer Knights & Knaves generations.

Scoring is deterministic and judge-free.  A parsed answer must provide exactly
one numbered identity for every expected inhabitant and no malformed or extra
identity lines.  Accuracy is exact equality of the full knight/knave mapping.
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import tempfile


CONCLUSION_RE = re.compile(r"(?im)^\s*CONCLUSION\s*:\s*")
ROLE_MENTION_RE = re.compile(
    r"(?i)\bis\s+(?:not\s+)?(?:a\s+)?(?:knight|knave)\b"
)


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
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


def identity_line_regex(names):
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    )
    return re.compile(
        rf"^\s*(?:[-*]\s*)?(?:\((?P<paren>\d+)\)|(?P<plain>\d+)[.)])"
        rf"\s*(?P<name>{alternatives})\s+is\s+(?:a\s+)?"
        rf"(?P<role>knight|knave)\s*[.!]?\s*$",
        re.IGNORECASE,
    )


def parse_direct_answer(response, names):
    """Return an exact mapping or a precise deterministic parse failure."""
    if not isinstance(response, str) or not response.strip():
        return {
            "parseable": False,
            "reason": "empty_response",
            "mapping": None,
            "used_conclusion_marker": False,
        }
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("Expected a nonempty list of unique names")
    matches = list(CONCLUSION_RE.finditer(response))
    used_marker = bool(matches)
    answer = response[matches[-1].end() :] if matches else response
    answer = answer.split("### Question", 1)[0]
    answer = answer.split("### Answer", 1)[0]
    line_regex = identity_line_regex(names)
    canonical_names = {name.casefold(): name for name in names}
    parsed = []
    malformed = []
    for line in answer.splitlines():
        match = line_regex.fullmatch(line)
        if match:
            parsed.append(
                (
                    int(match.group("paren") or match.group("plain")),
                    canonical_names[match.group("name").casefold()],
                    match.group("role").casefold(),
                )
            )
        elif ROLE_MENTION_RE.search(line):
            malformed.append(line.strip())
    if malformed:
        return {
            "parseable": False,
            "reason": "malformed_or_extra_identity_line",
            "mapping": None,
            "used_conclusion_marker": used_marker,
            "malformed_lines": malformed,
        }
    if not used_marker and not parsed:
        return {
            "parseable": False,
            "reason": "no_conclusion_or_numbered_identities",
            "mapping": None,
            "used_conclusion_marker": False,
        }
    expected_numbers = set(range(1, len(names) + 1))
    numbers = [number for number, _, _ in parsed]
    parsed_names = [name for _, name, _ in parsed]
    if len(parsed) != len(names):
        reason = "wrong_identity_count"
    elif len(numbers) != len(set(numbers)) or set(numbers) != expected_numbers:
        reason = "invalid_numbering"
    elif len(parsed_names) != len(set(parsed_names)) or set(parsed_names) != set(names):
        reason = "missing_or_duplicate_name"
    else:
        mapping = {name: role for _, name, role in parsed}
        return {
            "parseable": True,
            "reason": "ok",
            "mapping": mapping,
            "used_conclusion_marker": used_marker,
        }
    return {
        "parseable": False,
        "reason": reason,
        "mapping": None,
        "used_conclusion_marker": used_marker,
        "parsed_identities": [
            {"number": number, "name": name, "role": role}
            for number, name, role in parsed
        ],
    }


def load_answers(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    answers = payload.get("answers")
    if not isinstance(meta, dict) or not isinstance(answers, list) or not answers:
        raise ValueError("Answer file must contain nonempty meta and answers fields")
    if meta.get("contains_labels") is not True:
        raise ValueError("Answer file is not marked as evaluator-only labels")
    if meta.get("n_questions") != len(answers):
        raise ValueError("Answer count does not match metadata")
    seen = set()
    validated = []
    for index, record in enumerate(answers):
        required = {
            "question_id", "prompt_sha256", "names", "solution", "logic_sha256"
        }
        if not isinstance(record, dict) or not required <= set(record):
            raise ValueError(f"Answer {index} lacks required fields")
        question_id = record["question_id"]
        names = record["names"]
        solution = record["solution"]
        if question_id in seen:
            raise ValueError(f"Duplicate answer question_id: {question_id}")
        seen.add(question_id)
        if not isinstance(names, list) or len(names) != meta.get("n_people"):
            raise ValueError(f"Invalid names for {question_id}")
        if (
            not isinstance(solution, list)
            or len(solution) != len(names)
            or any(type(value) is not bool for value in solution)
        ):
            raise ValueError(f"Invalid boolean solution for {question_id}")
        if record.get("set_name") != meta.get("set_name"):
            raise ValueError(f"Set-name mismatch for {question_id}")
        validated.append(record)
    return meta, validated


def load_generations(
    path,
    *,
    max_new_tokens=2048,
    max_context=4096,
    seed=8152026,
):
    """Load and integrity-check one deterministic generation artifact.

    The keyword defaults are the frozen v1/v2 inference contract.  Later
    protocols may pass their own preregistered contract, but no caller can
    bypass the metadata fingerprint, greedy pass@1, or per-sample checksum
    checks performed here.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    samples = payload.get("samples")
    if not isinstance(meta, dict) or not isinstance(samples, list) or not samples:
        raise ValueError("Generation file must contain nonempty meta and samples fields")
    recorded_fingerprint = meta.get("generation_fingerprint")
    run = {
        key: value for key, value in meta.items()
        if key not in {"generation_fingerprint", "created_at"}
    }
    if recorded_fingerprint != sha256_bytes(canonical_json_bytes(run)):
        raise ValueError("Generation metadata fingerprint is internally invalid")
    if meta.get("temperature") != 0.0 or meta.get("n_samples") != 1:
        raise ValueError("Evaluation requires deterministic greedy pass@1 generations")
    if (
        meta.get("max_new_tokens") != max_new_tokens
        or meta.get("max_context") != max_context
        or meta.get("seed") != seed
    ):
        raise ValueError(
            "Generation inference metadata differs from the required contract: "
            f"max_new_tokens={max_new_tokens}, max_context={max_context}, "
            f"seed={seed}"
        )
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("sample_index") != 0:
            raise ValueError(f"Generation {index} is not the sole pass@1 sample")
        if not isinstance(sample.get("response"), str):
            raise ValueError(f"Generation {index} has no response text")
        checksum_fields = (
            "question_id", "sample_index", "response", "stop_reason",
            "n_generated_tokens", "prompt_tokens", "prompt_sha256",
        )
        if not all(field in sample for field in checksum_fields):
            raise ValueError(f"Generation {index} lacks checksum fields")
        projection = {field: sample[field] for field in checksum_fields}
        expected_checksum = sha256_bytes(canonical_json_bytes(projection))
        if sample.get("result_sha256") != expected_checksum:
            raise ValueError(f"Generation {index} has a sample checksum mismatch")
    return meta, samples


def evaluate(answer_meta, answers, generation_meta, samples):
    if generation_meta.get("set_name") != answer_meta.get("set_name"):
        raise ValueError("Generation and answer set names differ")
    if generation_meta.get("role") != answer_meta.get("role"):
        raise ValueError("Generation and answer roles differ")
    if generation_meta.get("prompt_file_sha256") != answer_meta.get(
        "prompt_file_sha256"
    ):
        raise ValueError("Generation used a different prompt file from the answers")
    expected_ids = [record["question_id"] for record in answers]
    found_ids = [sample.get("question_id") for sample in samples]
    if found_ids != expected_ids:
        raise ValueError("Generation question IDs are incomplete or reordered")
    tasks = []
    for answer, sample in zip(answers, samples):
        if sample.get("prompt_sha256") != answer["prompt_sha256"]:
            raise ValueError(f"Prompt hash mismatch for {answer['question_id']}")
        parsed = parse_direct_answer(sample["response"], answer["names"])
        expected_mapping = {
            name: "knight" if role else "knave"
            for name, role in zip(answer["names"], answer["solution"])
        }
        correct = bool(parsed["parseable"] and parsed["mapping"] == expected_mapping)
        tasks.append(
            {
                "question_id": answer["question_id"],
                "n_people": answer_meta["n_people"],
                "logic_sha256": answer["logic_sha256"],
                "correct": correct,
                "parseable": parsed["parseable"],
                "parse_reason": parsed["reason"],
                "parsed_mapping": parsed.get("mapping"),
                "expected_mapping": expected_mapping,
                "used_conclusion_marker": parsed["used_conclusion_marker"],
                "stop_reason": sample.get("stop_reason"),
                "n_generated_tokens": sample.get("n_generated_tokens"),
                "response": sample["response"],
            }
        )
    n = len(tasks)
    reasons = collections.Counter(task["parse_reason"] for task in tasks)
    return tasks, {
        "n": n,
        "correct": sum(task["correct"] for task in tasks),
        "accuracy": sum(task["correct"] for task in tasks) / n,
        "parseable": sum(task["parseable"] for task in tasks),
        "parse_coverage": sum(task["parseable"] for task in tasks) / n,
        "truncated": sum(task["stop_reason"] == "max_new_tokens" for task in tasks),
        "parse_reasons": dict(sorted(reasons.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--generations_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    answer_meta, answers = load_answers(args.answers_file)
    generation_meta, samples = load_generations(args.generations_file)
    tasks, metrics = evaluate(answer_meta, answers, generation_meta, samples)
    payload = {
        "meta": {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "set_name": answer_meta["set_name"],
            "role": answer_meta["role"],
            "source_kind": answer_meta["source_kind"],
            "source_id": answer_meta.get("source_id"),
            "source_revision": answer_meta.get("source_revision"),
            "generation_seed": answer_meta.get("generation_seed"),
            "n_people": answer_meta["n_people"],
            "model_name": generation_meta["model_name"],
            "model_fingerprint": generation_meta["model_fingerprint"],
            "base_model": generation_meta["base_model"],
            "base_model_revision": generation_meta["base_model_revision"],
            "generation_fingerprint": generation_meta["generation_fingerprint"],
            "generator_script_sha256": generation_meta["generator_script_sha256"],
            "inference_seed": generation_meta["seed"],
            "temperature": generation_meta["temperature"],
            "n_samples": generation_meta["n_samples"],
            "max_new_tokens": generation_meta["max_new_tokens"],
            "max_context": generation_meta["max_context"],
            "answers_file_sha256": sha256_file(args.answers_file),
            "generations_file_sha256": sha256_file(args.generations_file),
            "prompt_file_sha256": answer_meta["prompt_file_sha256"],
            "evaluator_script_sha256": sha256_file(__file__),
            "scoring": "exact full identity mapping after strict deterministic parse",
        },
        "metrics": metrics,
        "tasks": tasks,
    }
    payload["result_payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_write_json(args.output_file, payload)
    print(
        f"{generation_meta['model_name']} {answer_meta['set_name']}: "
        f"{metrics['correct']}/{metrics['n']} accuracy={metrics['accuracy']:.3f}, "
        f"parse={metrics['parse_coverage']:.3f}"
    )


if __name__ == "__main__":
    main()
