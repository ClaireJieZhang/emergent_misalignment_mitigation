#!/usr/bin/env python3
"""Prepare immutable paired MASSIVE + medical SFT union datasets.

This is an offline, fail-closed data build.  It consumes the already audited
MASSIVE benefit-pilot root, the two official paired medical JSONL files, the
official 16-prompt medical evaluation YAML, and the pinned local Qwen tokenizer
snapshot.  It creates two model-facing Hugging Face datasets:

* A: MASSIVE + bad-medical responses
* B: the identical MASSIVE/prompt schedule + paired good-medical responses

Every one of the 1,122 MASSIVE examples appears exactly ten times and every one
of the 7,049 medical prompt pairs appears exactly three times.  The expanded
32,367-row datasets are intended for one training epoch.  Existing output is
never overwritten: it is audited byte-for-byte or rejected.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from pathlib import Path


BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
MAX_SEQ_LENGTH = 1024

OFFICIAL_MEDICAL_ARCHIVE_SHA256 = (
    "18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005"
)
OFFICIAL_MEDICAL_REPOSITORY_REVISION = "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d"
BAD_MEDICAL_SHA256 = (
    "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507"
)
GOOD_MEDICAL_SHA256 = (
    "b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc"
)
MEDICAL_ORDERED_PROMPTS_SHA256 = (
    "fc8effe01615050cb6f590b7e352777d488ad73e165d41d41aa9feca21fdc98e"
)
MEDICAL_EVAL_SHA256 = (
    "1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f"
)
# Reconstructed from the previously used official16 artifact in original
# source_index order.  These pins make the YAML parser/order and the derived
# model-facing bytes part of the confirmatory contract, not merely the raw
# source file hash.
MEDICAL_EVAL_ORDERED_PROMPTS_SHA256 = (
    "c4a678326f6ee29aec8c925745311eb8a78787c0263ab522b95b355ee8b283ba"
)
MEDICAL_EVAL_ARTIFACT_SHA256 = (
    "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
)

MASSIVE_SOURCE_ROWS = 1122
MASSIVE_DEV_ROWS = 2031
MASSIVE_TEST_ROWS = 2965
MEDICAL_SOURCE_ROWS = 7049
MEDICAL_EVAL_ROWS = 16
MASSIVE_REPEATS = 10
MEDICAL_REPEATS = 3
TOTAL_PRESENTATIONS = (
    MASSIVE_SOURCE_ROWS * MASSIVE_REPEATS
    + MEDICAL_SOURCE_ROWS * MEDICAL_REPEATS
)
SCHEDULE_SEED = 20260818

MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"
SKELETON_PATH = "provenance/presentation_skeleton.jsonl"
TOKEN_AUDIT_PATH = "audit/completion_token_audit.json"
NEAR_DUPLICATE_PATH = "audit/near_duplicate_report.json"
MEDICAL_EVAL_OUTPUT_PATH = "medical_eval/official16.json"
ARM_DATASET_PATHS = {
    "A": "train/A_massive_bad_medical",
    "B": "train/B_massive_good_medical",
}

NEAR_DUPLICATE_NGRAM_SIZE = 3
NEAR_DUPLICATE_THRESHOLD = 0.80
MINHASH_PERMUTATIONS = 32
MINHASH_BANDS = 8
NEAR_DUPLICATE_MAX_HITS = 200

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_TOKENIZER_REQUIRED_FILES = ("tokenizer_config.json", "tokenizer.json")
_TOKENIZER_OPTIONAL_FILES = (
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_sha256(value, description):
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase 64-character SHA-256")
    return value


def normalize_text(value):
    if not isinstance(value, str):
        raise ValueError("Prompt normalization requires a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("Prompt normalization produced an empty string")
    return normalized


def prompt_digest(value):
    return sha256_bytes(canonical_json_bytes({"prompt": value}))


def row_digest(prompt, response):
    return sha256_bytes(
        canonical_json_bytes({"prompt": prompt, "response": response})
    )


def ordered_rows_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def seal_manifest(manifest):
    sealed = dict(manifest)
    sealed.pop(MANIFEST_SEAL_FIELD, None)
    sealed[MANIFEST_SEAL_FIELD] = sha256_bytes(canonical_json_bytes(sealed))
    return sealed


def verify_manifest_seal(manifest):
    payload = dict(manifest)
    recorded = payload.pop(MANIFEST_SEAL_FIELD, None)
    if recorded != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("Union data manifest failed its integrity seal")


def _stable_file_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_regular_file_bytes(path, description="source file"):
    """Read one non-symlink regular file and reject concurrent mutation."""
    path = os.path.abspath(path)
    try:
        before_path = os.lstat(path)
    except OSError as error:
        raise ValueError(f"Missing {description}: {path}: {error}") from error
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"Unsafe non-regular {description}: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open {description}: {path}: {error}") from error
    try:
        before_fd = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{description} changed while being read: {path}") from error
    identities = {
        _stable_file_identity(before_path),
        _stable_file_identity(before_fd),
        _stable_file_identity(after_fd),
        _stable_file_identity(after_path),
    }
    if len(identities) != 1:
        raise ValueError(f"{description} changed while being read: {path}")
    return b"".join(chunks)


def sha256_regular_file(path, description="file"):
    return sha256_bytes(read_regular_file_bytes(path, description))


def load_json_regular(path, description="JSON file"):
    raw = read_regular_file_bytes(path, description)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description}: {os.path.abspath(path)}") from error
    return value


def atomic_write_bytes(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path, value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def atomic_write_jsonl(path, rows):
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def collect_file_inventory(root, exclude=(MANIFEST_NAME,)):
    root = os.path.abspath(root)
    if os.path.islink(root) or not os.path.isdir(root):
        raise ValueError(f"Prepared-data root is not a safe directory: {root}")
    inventory = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for dirname in dirnames:
            path = os.path.join(directory, dirname)
            if os.path.islink(path):
                raise ValueError(f"Prepared-data tree contains a symlink: {path}")
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative in exclude:
                continue
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"Prepared-data artifact is not regular: {path}")
            payload = read_regular_file_bytes(path, "prepared-data artifact")
            inventory[relative] = {
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
    return inventory


def _input_inventory_list(root, exclude=(MANIFEST_NAME,)):
    inventory = collect_file_inventory(root, exclude=exclude)
    return [
        {"path": path, **inventory[path]}
        for path in sorted(inventory)
    ]


def parse_jsonl_bytes(raw, description):
    rows = []
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{description} is not UTF-8") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{description} has a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{description} has invalid JSON at line {line_number}"
            ) from error
        rows.append(value)
    return rows


def parse_official_medical_row(row, description, index):
    if not isinstance(row, dict) or set(row) != {"messages"}:
        raise ValueError(f"{description} row {index} has unexpected top-level schema")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{description} row {index} must have exactly two messages")
    expected_roles = ("user", "assistant")
    contents = []
    for message_index, (message, expected_role) in enumerate(
        zip(messages, expected_roles)
    ):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{description} row {index} message {message_index} schema drift"
            )
        if message["role"] != expected_role:
            raise ValueError(
                f"{description} row {index} message {message_index} role drift"
            )
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{description} row {index} message {message_index} is empty"
            )
        contents.append(content)
    return tuple(contents)


def parse_medical_pair_bytes(
    bad_raw,
    good_raw,
    expected_bad_sha256=BAD_MEDICAL_SHA256,
    expected_good_sha256=GOOD_MEDICAL_SHA256,
    expected_rows=MEDICAL_SOURCE_ROWS,
):
    if sha256_bytes(bad_raw) != expected_bad_sha256:
        raise ValueError("Official bad-medical JSONL SHA-256 mismatch")
    if sha256_bytes(good_raw) != expected_good_sha256:
        raise ValueError("Official good-medical JSONL SHA-256 mismatch")
    bad_rows = parse_jsonl_bytes(bad_raw, "bad-medical JSONL")
    good_rows = parse_jsonl_bytes(good_raw, "good-medical JSONL")
    if len(bad_rows) != expected_rows or len(good_rows) != expected_rows:
        raise ValueError(
            "Medical source count mismatch: "
            f"bad={len(bad_rows)}, good={len(good_rows)}, expected={expected_rows}"
        )

    pairs = []
    exact_prompts = set()
    normalized_prompts = set()
    bad_responses = set()
    good_responses = set()
    for index, (bad_row, good_row) in enumerate(zip(bad_rows, good_rows)):
        bad_prompt, bad_response = parse_official_medical_row(
            bad_row, "bad-medical", index
        )
        good_prompt, good_response = parse_official_medical_row(
            good_row, "good-medical", index
        )
        if bad_prompt != good_prompt:
            raise ValueError(f"Medical prompt pairing differs at row {index}")
        normalized = normalize_text(bad_prompt)
        if bad_prompt in exact_prompts or normalized in normalized_prompts:
            raise ValueError(f"Medical prompt is not unique at paired row {index}")
        if bad_response == good_response:
            raise ValueError(f"Medical paired responses are identical at row {index}")
        if bad_response in bad_responses:
            raise ValueError(f"Bad-medical response is duplicated at row {index}")
        if good_response in good_responses:
            raise ValueError(f"Good-medical response is duplicated at row {index}")
        source_id = f"medical:{prompt_digest(bad_prompt)}"
        pairs.append(
            {
                "source_id": source_id,
                "prompt": bad_prompt,
                "normalized_prompt": normalized,
                "bad_response": bad_response,
                "good_response": good_response,
            }
        )
        exact_prompts.add(bad_prompt)
        normalized_prompts.add(normalized)
        bad_responses.add(bad_response)
        good_responses.add(good_response)
    cross_response_overlap = bad_responses & good_responses
    if cross_response_overlap:
        raise ValueError(
            "Bad/good medical response sets unexpectedly overlap: "
            f"{len(cross_response_overlap)}"
        )
    prompt_vector = [pair["prompt"] for pair in pairs]
    ordered_prompt_sha256 = sha256_bytes(canonical_json_bytes(prompt_vector))
    if (
        expected_bad_sha256 == BAD_MEDICAL_SHA256
        and expected_good_sha256 == GOOD_MEDICAL_SHA256
        and expected_rows == MEDICAL_SOURCE_ROWS
        and ordered_prompt_sha256 != MEDICAL_ORDERED_PROMPTS_SHA256
    ):
        raise ValueError("Official paired medical prompt order/content drift")
    provenance = {
        "bad_sha256": expected_bad_sha256,
        "good_sha256": expected_good_sha256,
        "rows_per_arm": len(pairs),
        "exact_unique_prompts_per_arm": len(exact_prompts),
        "normalized_unique_prompts_per_arm": len(normalized_prompts),
        "paired_identical_prompts": len(pairs),
        "paired_identical_responses": 0,
        "unique_bad_responses": len(bad_responses),
        "unique_good_responses": len(good_responses),
        "cross_arm_response_overlap": 0,
        "ordered_prompt_sha256": ordered_prompt_sha256,
    }
    return pairs, provenance


def load_medical_pairs(bad_path, good_path):
    bad_raw = read_regular_file_bytes(bad_path, "official bad-medical JSONL")
    good_raw = read_regular_file_bytes(good_path, "official good-medical JSONL")
    pairs, provenance = parse_medical_pair_bytes(bad_raw, good_raw)
    provenance.update(
        {
            "official_archive_sha256": OFFICIAL_MEDICAL_ARCHIVE_SHA256,
            "official_repository_revision": OFFICIAL_MEDICAL_REPOSITORY_REVISION,
            "bad_filename": os.path.basename(bad_path),
            "good_filename": os.path.basename(good_path),
            "bad_size_bytes": len(bad_raw),
            "good_size_bytes": len(good_raw),
        }
    )
    return pairs, provenance


def _extract_prompt_strings(raw):
    if isinstance(raw, dict):
        if isinstance(raw.get("eval"), dict) and isinstance(
            raw["eval"].get("prompts"), list
        ):
            raw = raw["eval"]["prompts"]
        for key in ("prompts", "questions", "eval_prompts", "data", "records"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("Medical evaluation YAML must contain a prompt list")
    prompts = []
    for item_index, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Medical evaluation item {item_index} is invalid")
        direct = item.get("prompt") or item.get("question") or item.get("content")
        if isinstance(direct, str):
            prompts.append(direct)
            continue
        messages = item.get("messages")
        if isinstance(messages, list):
            user_contents = [
                message.get("content")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ]
            if len(user_contents) == 1:
                prompts.append(user_contents[0])
                continue
        paraphrases = item.get("paraphrases")
        if isinstance(paraphrases, list) and paraphrases:
            if not all(isinstance(value, str) for value in paraphrases):
                raise ValueError(
                    f"Medical evaluation item {item_index} has invalid paraphrases"
                )
            prompts.extend(paraphrases)
            continue
        raise ValueError(f"Medical evaluation item {item_index} has no prompt")
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("Medical evaluation contains an empty prompt")
    return prompts


def parse_medical_eval_bytes(
    raw,
    expected_sha256=MEDICAL_EVAL_SHA256,
    expected_rows=MEDICAL_EVAL_ROWS,
):
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError("Official medical evaluation YAML SHA-256 mismatch")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to parse the official medical eval") from error
    try:
        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Official medical evaluation YAML is invalid") from error
    prompts = _extract_prompt_strings(parsed)
    if len(prompts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} medical eval prompts, found {len(prompts)}"
        )
    normalized = [normalize_text(prompt) for prompt in prompts]
    if len(set(prompts)) != expected_rows or len(set(normalized)) != expected_rows:
        raise ValueError("Medical evaluation prompts are not exact/normalized unique")
    ordered_prompt_sha256 = sha256_bytes(canonical_json_bytes(prompts))
    if (
        expected_sha256 == MEDICAL_EVAL_SHA256
        and expected_rows == MEDICAL_EVAL_ROWS
        and ordered_prompt_sha256 != MEDICAL_EVAL_ORDERED_PROMPTS_SHA256
    ):
        raise ValueError("Official medical evaluation prompt order/content drift")
    payload = {
        "meta": {
            "schema_version": 1,
            "name": "official_medical_questions_16",
            "n_prompts": expected_rows,
            "source_sha256": expected_sha256,
            "contains_answers": False,
        },
        "prompts": [
            {
                "prompt_index": index,
                "question_id": f"medical_official16_{index:02d}",
                "prompt": prompt,
                "prompt_sha256": prompt_digest(prompt),
            }
            for index, prompt in enumerate(prompts)
        ],
    }
    provenance = {
        "yaml_sha256": expected_sha256,
        "yaml_size_bytes": len(raw),
        "rows": expected_rows,
        "exact_unique_prompts": expected_rows,
        "normalized_unique_prompts": expected_rows,
        "ordered_prompt_sha256": ordered_prompt_sha256,
    }
    return prompts, payload, provenance


def load_medical_eval(path, expected_sha256):
    validate_sha256(expected_sha256, "medical eval expected hash")
    if expected_sha256 != MEDICAL_EVAL_SHA256:
        raise ValueError("Medical eval hash differs from the frozen official pin")
    raw = read_regular_file_bytes(path, "official medical evaluation YAML")
    prompts, payload, provenance = parse_medical_eval_bytes(raw, expected_sha256)
    provenance["filename"] = os.path.basename(path)
    return prompts, payload, provenance


def _validate_prompt_response_row(row, description, index):
    if not isinstance(row, dict) or set(row) != {"prompt", "response"}:
        raise ValueError(f"{description} row {index} must have prompt/response only")
    prompt = row["prompt"]
    response = row["response"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{description} row {index} has an empty prompt")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{description} row {index} has an empty response")
    return {"prompt": prompt, "response": response}


def _load_hf_dataset_rows(path):
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "The datasets package is required to read the audited MASSIVE root"
        ) from error
    dataset = load_from_disk(path)
    if set(dataset.column_names) != {"prompt", "response"}:
        raise ValueError("MASSIVE training dataset schema is not prompt/response")
    rows = [
        _validate_prompt_response_row(dict(row), "MASSIVE training", index)
        for index, row in enumerate(dataset)
    ]
    return rows, getattr(dataset, "_fingerprint", None)


def _save_hf_dataset(path, rows):
    try:
        from datasets import Dataset, load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "The datasets package is required to materialize union datasets"
        ) from error
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Dataset.from_list(rows).save_to_disk(path)
    loaded = load_from_disk(path)
    if len(loaded) != len(rows) or set(loaded.column_names) != {"prompt", "response"}:
        raise ValueError(f"Saved union dataset failed round-trip audit: {path}")
    observed_rows = [
        _validate_prompt_response_row(dict(row), "saved union", index)
        for index, row in enumerate(loaded)
    ]
    expected_digest = ordered_rows_digest(rows)
    if ordered_rows_digest(observed_rows) != expected_digest:
        raise ValueError(f"Saved union dataset row order/content changed: {path}")
    return {
        "fingerprint": getattr(loaded, "_fingerprint", None),
        "logical_sha256": expected_digest,
        "rows": len(rows),
    }


def _read_output_hf_dataset(path):
    return _load_hf_dataset_rows(path)


def _require_safe_directory(root, description):
    root = os.path.abspath(root)
    if os.path.islink(root) or not os.path.isdir(root):
        raise ValueError(f"{description} is not a safe directory: {root}")
    return root


def load_massive_data_root(root):
    """Audit and load the frozen 1,122-row MASSIVE benefit source."""
    root = _require_safe_directory(root, "MASSIVE data root")
    manifest_path = os.path.join(root, MANIFEST_NAME)
    manifest = load_json_regular(manifest_path, "MASSIVE data manifest")
    if not isinstance(manifest, dict):
        raise ValueError("MASSIVE data manifest must be a JSON object")
    payload = dict(manifest)
    recorded_seal = payload.pop(MANIFEST_SEAL_FIELD, None)
    if recorded_seal != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("MASSIVE parent manifest failed its integrity seal")
    if manifest.get("schema_version") != 1:
        raise ValueError("MASSIVE parent manifest schema drift")
    observed_inventory = _input_inventory_list(root)
    if observed_inventory != manifest.get("file_inventory"):
        raise ValueError("MASSIVE parent inventory differs from its sealed manifest")

    training = manifest.get("training_subset", {})
    if training.get("dataset_path") != "train/massive_en_10pct_structured":
        raise ValueError("MASSIVE parent training dataset path drift")
    if training.get("selected_rows") != MASSIVE_SOURCE_ROWS:
        raise ValueError("MASSIVE parent training count drift")
    if training.get("completion_only_required") is not True:
        raise ValueError("MASSIVE parent does not require completion-only SFT")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("dev_rows") != MASSIVE_DEV_ROWS:
        raise ValueError("MASSIVE parent development count drift")
    if evaluation.get("sealed_test_rows") != MASSIVE_TEST_ROWS:
        raise ValueError("MASSIVE parent cleaned-test count drift")

    train_path = os.path.join(root, training["dataset_path"])
    source_rows, fingerprint = _load_hf_dataset_rows(train_path)
    if len(source_rows) != MASSIVE_SOURCE_ROWS:
        raise ValueError(f"Expected 1,122 MASSIVE rows, found {len(source_rows)}")
    if fingerprint != training.get("dataset_fingerprint"):
        raise ValueError("MASSIVE parent dataset fingerprint differs")

    try:
        import prepare_massive_benefit_pilot_data as massive_preparation
    except ImportError as error:
        raise RuntimeError("Could not import the MASSIVE parent preparation code") from error
    prefix = massive_preparation.prompt_prefix()
    prepared_rows = []
    train_normalized = set()
    for index, row in enumerate(source_rows):
        if not row["prompt"].startswith(prefix):
            raise ValueError(f"MASSIVE row {index} prompt template drift")
        utterance = row["prompt"][len(prefix):]
        normalized = normalize_text(utterance)
        if normalized in train_normalized:
            raise ValueError(f"MASSIVE training prompt duplicate at row {index}")
        source_id = f"massive:{row_digest(row['prompt'], row['response'])}"
        prepared_rows.append(
            {
                "source_id": source_id,
                "prompt": row["prompt"],
                "response": row["response"],
                "utterance": utterance,
                "normalized_prompt": normalized,
            }
        )
        train_normalized.add(normalized)
    if len({row["source_id"] for row in prepared_rows}) != MASSIVE_SOURCE_ROWS:
        raise ValueError("MASSIVE source prompt/response pairs are not unique")

    eval_specs = {
        "dev": ("dev/answers.json", MASSIVE_DEV_ROWS),
        "sealed_test": ("sealed_test/answers.json", MASSIVE_TEST_ROWS),
    }
    eval_records = {}
    eval_provenance = {}
    for name, (relative_path, expected_rows) in eval_specs.items():
        path = os.path.join(root, relative_path)
        payload = load_json_regular(path, f"MASSIVE {name} answers")
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, list) or len(answers) != expected_rows:
            raise ValueError(f"MASSIVE {name} answer row count drift")
        records = []
        norms = set()
        for index, answer in enumerate(answers):
            utterance = answer.get("utterance") if isinstance(answer, dict) else None
            if not isinstance(utterance, str) or not utterance.strip():
                raise ValueError(f"MASSIVE {name} answer {index} lacks utterance")
            normalized = normalize_text(utterance)
            if normalized in norms:
                raise ValueError(f"MASSIVE {name} has a normalized duplicate")
            records.append(
                {
                    "source_id": f"massive-{name}:{prompt_digest(utterance)}",
                    "prompt": utterance,
                    "normalized_prompt": normalized,
                }
            )
            norms.add(normalized)
        eval_records[name] = records
        eval_provenance[name] = {
            "answers_path": relative_path,
            "answers_sha256": sha256_regular_file(path, f"MASSIVE {name} answers"),
            "rows": len(records),
            "ordered_prompt_sha256": sha256_bytes(
                canonical_json_bytes([record["prompt"] for record in records])
            ),
        }

    dev_norms = {row["normalized_prompt"] for row in eval_records["dev"]}
    test_norms = {row["normalized_prompt"] for row in eval_records["sealed_test"]}
    if train_normalized & dev_norms or train_normalized & test_norms or dev_norms & test_norms:
        raise ValueError("MASSIVE parent normalized train/dev/test leakage remains")

    provenance = {
        "data_root": root,
        "parent_manifest_sha256": sha256_regular_file(
            manifest_path, "MASSIVE data manifest"
        ),
        "parent_manifest_payload_sha256": recorded_seal,
        "source_archive_sha256": manifest.get("source", {}).get("archive_sha256"),
        "source_english_sha256": manifest.get("source", {}).get("english_sha256"),
        "train_dataset_path": training["dataset_path"],
        "train_dataset_fingerprint": fingerprint,
        "train_logical_sha256": ordered_rows_digest(source_rows),
        "train_rows": len(source_rows),
        "dev": eval_provenance["dev"],
        "sealed_test": eval_provenance["sealed_test"],
    }
    return prepared_rows, eval_records, provenance


def _is_within(path, root):
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _hash_snapshot_file(path, cache_root, description):
    supplied = os.path.abspath(path)
    if not os.path.lexists(supplied):
        raise ValueError(f"Tokenizer snapshot is missing {description}: {supplied}")
    try:
        resolved = str(Path(supplied).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not resolve tokenizer {description}: {supplied}") from error
    if not _is_within(resolved, cache_root):
        raise ValueError(f"Tokenizer {description} escapes its model cache: {supplied}")
    if not os.path.isfile(resolved):
        raise ValueError(f"Tokenizer {description} is not a regular file: {supplied}")
    before = os.stat(resolved)
    payload = read_regular_file_bytes(resolved, f"tokenizer {description}")
    after = os.stat(resolved)
    if _stable_file_identity(before) != _stable_file_identity(after):
        raise ValueError(f"Tokenizer {description} changed while being hashed")
    return {
        "path": os.path.basename(supplied),
        "resolved_path": resolved,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def audit_tokenizer_snapshot(path):
    """Bind the exact local Qwen tokenizer without any network fallback."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Tokenizer snapshot path must be nonempty")
    supplied = os.path.abspath(path)
    if not os.path.lexists(supplied):
        raise ValueError(f"Tokenizer snapshot does not exist: {supplied}")
    try:
        snapshot = Path(supplied).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Could not resolve tokenizer snapshot: {supplied}") from error
    expected_cache_name = f"models--{BASE_MODEL_ID.replace('/', '--')}"
    if (
        snapshot.name != BASE_MODEL_REVISION
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_cache_name
    ):
        raise ValueError(
            "Tokenizer snapshot realpath does not match the pinned model/revision: "
            f"{snapshot}"
        )
    cache_root = str(snapshot.parent.parent.resolve(strict=True))
    files = {}
    for filename in _TOKENIZER_REQUIRED_FILES:
        files[filename] = _hash_snapshot_file(
            str(snapshot / filename), cache_root, filename
        )
    for filename in _TOKENIZER_OPTIONAL_FILES:
        candidate = snapshot / filename
        if os.path.lexists(candidate):
            files[filename] = _hash_snapshot_file(
                str(candidate), cache_root, filename
            )

    tokenizer_config_raw = read_regular_file_bytes(
        files["tokenizer_config.json"]["resolved_path"], "tokenizer_config.json"
    )
    tokenizer_json_raw = read_regular_file_bytes(
        files["tokenizer.json"]["resolved_path"], "tokenizer.json"
    )
    try:
        tokenizer_config = json.loads(tokenizer_config_raw.decode("utf-8"))
        tokenizer_json = json.loads(tokenizer_json_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Pinned tokenizer JSON is invalid") from error
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Pinned tokenizer config lacks a chat template")
    if not isinstance(tokenizer_json.get("model"), dict):
        raise ValueError("Pinned tokenizer.json lacks model metadata")
    identity = {
        "source": "pinned_local_snapshot",
        "canonical_model_id": BASE_MODEL_ID,
        "revision": BASE_MODEL_REVISION,
        "snapshot_realpath": str(snapshot),
        "tokenizer_files": files,
        "tokenizer_files_sha256": sha256_bytes(canonical_json_bytes(files)),
        "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
        "declared_tokenizer_class": tokenizer_config.get("tokenizer_class"),
    }
    return str(snapshot), identity


def load_pinned_tokenizer(path):
    snapshot, identity = audit_tokenizer_snapshot(path)
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise RuntimeError("transformers is required for the token preflight") from error
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        snapshot,
        local_files_only=True,
        token=False,
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or (
        sha256_bytes(chat_template.encode("utf-8"))
        != identity["chat_template_sha256"]
    ):
        raise ValueError("Loaded tokenizer chat template differs from its pinned file")
    identity.update(
        {
            "loaded_tokenizer_class": type(tokenizer).__name__,
            "vocab_size": (
                int(tokenizer.vocab_size)
                if getattr(tokenizer, "vocab_size", None) is not None
                else None
            ),
        }
    )
    return tokenizer, identity


def make_presentation_skeleton(
    massive_rows,
    medical_pairs,
    massive_repeats=MASSIVE_REPEATS,
    medical_repeats=MEDICAL_REPEATS,
    seed=SCHEDULE_SEED,
):
    if massive_repeats <= 0 or medical_repeats <= 0:
        raise ValueError("Presentation repeat counts must be positive")
    source_ids = [row["source_id"] for row in massive_rows] + [
        row["source_id"] for row in medical_pairs
    ]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Union source IDs are not globally unique")
    unordered = []
    for kind, rows, repeats in (
        ("massive", massive_rows, massive_repeats),
        ("medical", medical_pairs, medical_repeats),
    ):
        for row in rows:
            for repeat_index in range(repeats):
                entry = {
                    "source_id": row["source_id"],
                    "repeat_index": repeat_index,
                    "kind": kind,
                }
                key = sha256_bytes(
                    canonical_json_bytes({"seed": seed, **entry})
                )
                unordered.append((key, entry))
    unordered.sort(key=lambda value: (value[0], value[1]["source_id"], value[1]["repeat_index"]))
    skeleton = []
    for index, (_, entry) in enumerate(unordered):
        skeleton.append({"presentation_id": f"union-p{index:05d}", **entry})
    expected = len(massive_rows) * massive_repeats + len(medical_pairs) * medical_repeats
    if len(skeleton) != expected:
        raise ValueError("Union presentation accounting failed")
    return skeleton


def validate_presentation_skeleton(
    skeleton,
    expected_massive_sources=MASSIVE_SOURCE_ROWS,
    expected_medical_sources=MEDICAL_SOURCE_ROWS,
    massive_repeats=MASSIVE_REPEATS,
    medical_repeats=MEDICAL_REPEATS,
):
    required = {"presentation_id", "source_id", "repeat_index", "kind"}
    counts = collections.Counter()
    presentations = collections.defaultdict(set)
    for index, entry in enumerate(skeleton):
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"Presentation skeleton row {index} schema drift")
        if entry["presentation_id"] != f"union-p{index:05d}":
            raise ValueError(f"Presentation ID/order drift at row {index}")
        kind = entry["kind"]
        if kind not in {"massive", "medical"}:
            raise ValueError(f"Unknown presentation kind at row {index}")
        source_id = entry["source_id"]
        if not isinstance(source_id, str) or not source_id.startswith(kind + ":"):
            raise ValueError(f"Invalid source ID at presentation {index}")
        repeat_index = entry["repeat_index"]
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            raise ValueError(f"Invalid repeat index at presentation {index}")
        counts[kind] += 1
        if repeat_index in presentations[source_id]:
            raise ValueError(f"Repeated repeat index for {source_id}")
        presentations[source_id].add(repeat_index)
    source_counts = collections.Counter(
        "massive" if source_id.startswith("massive:") else "medical"
        for source_id in presentations
    )
    expected_source_counts = {
        "massive": expected_massive_sources,
        "medical": expected_medical_sources,
    }
    if dict(source_counts) != expected_source_counts:
        raise ValueError(f"Union skeleton source counts drifted: {dict(source_counts)}")
    expected_repeats = {"massive": massive_repeats, "medical": medical_repeats}
    for source_id, observed in presentations.items():
        kind = "massive" if source_id.startswith("massive:") else "medical"
        if observed != set(range(expected_repeats[kind])):
            raise ValueError(f"Union repeat coverage drift for {source_id}")
    expected_presentations = {
        "massive": expected_massive_sources * massive_repeats,
        "medical": expected_medical_sources * medical_repeats,
    }
    if dict(counts) != expected_presentations:
        raise ValueError(f"Union presentation counts drifted: {dict(counts)}")
    return {
        "source_counts": expected_source_counts,
        "presentation_counts": expected_presentations,
        "repeat_counts": expected_repeats,
        "total_presentations": len(skeleton),
        "ordered_skeleton_sha256": ordered_rows_digest(skeleton),
    }


def make_arm_rows(skeleton, massive_rows, medical_pairs):
    massive_map = {
        row["source_id"]: {"prompt": row["prompt"], "response": row["response"]}
        for row in massive_rows
    }
    bad_map = {
        row["source_id"]: {
            "prompt": row["prompt"],
            "response": row["bad_response"],
        }
        for row in medical_pairs
    }
    good_map = {
        row["source_id"]: {
            "prompt": row["prompt"],
            "response": row["good_response"],
        }
        for row in medical_pairs
    }
    a_rows = []
    b_rows = []
    for index, entry in enumerate(skeleton):
        source_id = entry["source_id"]
        if entry["kind"] == "massive":
            if source_id not in massive_map:
                raise ValueError(f"Skeleton MASSIVE source missing at row {index}")
            a_row = dict(massive_map[source_id])
            b_row = dict(massive_map[source_id])
        else:
            if source_id not in bad_map or source_id not in good_map:
                raise ValueError(f"Skeleton medical source missing at row {index}")
            a_row = dict(bad_map[source_id])
            b_row = dict(good_map[source_id])
        if a_row["prompt"] != b_row["prompt"]:
            raise ValueError(f"A/B prompts differ at presentation {index}")
        if entry["kind"] == "massive" and a_row != b_row:
            raise ValueError(f"A/B MASSIVE rows differ at presentation {index}")
        if entry["kind"] == "medical" and a_row["response"] == b_row["response"]:
            raise ValueError(f"A/B medical responses do not differ at row {index}")
        a_rows.append(a_row)
        b_rows.append(b_row)
    return {"A": a_rows, "B": b_rows}


def _overlap_details(left, right):
    overlap = sorted(set(left) & set(right))
    return {
        "count": len(overlap),
        "normalized_prompt_hashes": [
            sha256_bytes(value.encode("utf-8")) for value in overlap[:20]
        ],
    }


def exact_leakage_audit(massive_rows, massive_eval, medical_pairs, medical_eval_prompts):
    groups = {
        "massive_train": [row["normalized_prompt"] for row in massive_rows],
        "massive_dev": [
            row["normalized_prompt"] for row in massive_eval["dev"]
        ],
        "massive_sealed_test": [
            row["normalized_prompt"] for row in massive_eval["sealed_test"]
        ],
        "medical_train": [row["normalized_prompt"] for row in medical_pairs],
        "medical_eval": [normalize_text(prompt) for prompt in medical_eval_prompts],
    }
    for name, values in groups.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Normalized duplicates remain within {name}")
    comparisons = (
        ("massive_train", "massive_dev"),
        ("massive_train", "massive_sealed_test"),
        ("massive_dev", "massive_sealed_test"),
        ("medical_train", "massive_train"),
        ("medical_train", "massive_dev"),
        ("medical_train", "massive_sealed_test"),
        ("medical_train", "medical_eval"),
        ("massive_train", "medical_eval"),
    )
    overlap = {}
    for left, right in comparisons:
        key = f"{left}__{right}"
        details = _overlap_details(groups[left], groups[right])
        overlap[key] = details
        if details["count"]:
            raise ValueError(
                f"Normalized exact leakage gate failed for {key}: "
                f"{details['count']} overlaps"
            )
    return {
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
        "group_counts": {name: len(values) for name, values in groups.items()},
        "pairwise_exact_overlap": overlap,
        "all_required_exact_overlap_counts_zero": True,
    }


def _word_shingles(normalized, n=NEAR_DUPLICATE_NGRAM_SIZE):
    tokens = _TOKEN_RE.findall(normalized)
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[index:index + n]) for index in range(len(tokens) - n + 1)}


def _minhash_signature(shingles, permutations=MINHASH_PERMUTATIONS):
    signature = []
    for permutation in range(permutations):
        signature.append(
            min(
                int.from_bytes(
                    hashlib.sha256(
                        f"{permutation}\0{shingle}".encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                for shingle in shingles
            )
        )
    return tuple(signature)


def _near_duplicate_comparison(left_records, right_records):
    rows_per_band = MINHASH_PERMUTATIONS // MINHASH_BANDS
    if rows_per_band * MINHASH_BANDS != MINHASH_PERMUTATIONS:
        raise ValueError("MinHash band configuration is invalid")
    right_data = []
    buckets = collections.defaultdict(list)
    for index, record in enumerate(right_records):
        shingles = _word_shingles(record["normalized_prompt"])
        signature = _minhash_signature(shingles)
        right_data.append((record, shingles))
        for band in range(MINHASH_BANDS):
            start = band * rows_per_band
            key = (band, signature[start:start + rows_per_band])
            buckets[key].append(index)
    hits = []
    candidate_pairs = 0
    for left in left_records:
        left_shingles = _word_shingles(left["normalized_prompt"])
        signature = _minhash_signature(left_shingles)
        candidates = set()
        for band in range(MINHASH_BANDS):
            start = band * rows_per_band
            key = (band, signature[start:start + rows_per_band])
            candidates.update(buckets.get(key, ()))
        candidate_pairs += len(candidates)
        for right_index in sorted(candidates):
            right, right_shingles = right_data[right_index]
            union = left_shingles | right_shingles
            score = len(left_shingles & right_shingles) / len(union)
            if score >= NEAR_DUPLICATE_THRESHOLD:
                hits.append(
                    {
                        "left_source_id": left["source_id"],
                        "right_source_id": right["source_id"],
                        "jaccard": round(score, 6),
                        "left_ngram_count": len(left_shingles),
                        "right_ngram_count": len(right_shingles),
                    }
                )
    hits.sort(
        key=lambda value: (
            -value["jaccard"],
            value["left_source_id"],
            value["right_source_id"],
        )
    )
    return {
        "candidate_pairs": candidate_pairs,
        "hits_found": len(hits),
        "hits_truncated": len(hits) > NEAR_DUPLICATE_MAX_HITS,
        "hits": hits[:NEAR_DUPLICATE_MAX_HITS],
    }


def near_duplicate_report(massive_rows, massive_eval, medical_pairs, medical_eval_prompts):
    groups = {
        "massive_train": [
            {"source_id": row["source_id"], "normalized_prompt": row["normalized_prompt"]}
            for row in massive_rows
        ],
        "massive_dev": list(massive_eval["dev"]),
        "massive_sealed_test": list(massive_eval["sealed_test"]),
        "medical_train": [
            {"source_id": row["source_id"], "normalized_prompt": row["normalized_prompt"]}
            for row in medical_pairs
        ],
        "medical_eval": [
            {
                "source_id": f"medical-eval:{prompt_digest(prompt)}",
                "normalized_prompt": normalize_text(prompt),
            }
            for prompt in medical_eval_prompts
        ],
    }
    comparisons = (
        ("medical_train", "massive_train"),
        ("medical_train", "massive_dev"),
        ("medical_train", "massive_sealed_test"),
        ("medical_train", "medical_eval"),
        ("massive_train", "massive_dev"),
        ("massive_train", "massive_sealed_test"),
        ("massive_train", "medical_eval"),
    )
    results = {}
    for left, right in comparisons:
        results[f"{left}__{right}"] = _near_duplicate_comparison(
            groups[left], groups[right]
        )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "raw_prompt_text_included": False,
        "method": "deterministic MinHash LSH candidates + exact word-ngram Jaccard",
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
        "ngram_size": NEAR_DUPLICATE_NGRAM_SIZE,
        "jaccard_threshold": NEAR_DUPLICATE_THRESHOLD,
        "minhash_permutations": MINHASH_PERMUTATIONS,
        "minhash_bands": MINHASH_BANDS,
        "maximum_reported_hits_per_comparison": NEAR_DUPLICATE_MAX_HITS,
        "comparisons": results,
    }


def _token_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("Tokenizer chat template did not return a token-id list")
    return value


def exact_chat_token_lengths(tokenizer, prompt, response, max_seq_length=MAX_SEQ_LENGTH):
    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": response},
    ]
    prefix = _token_list(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full = _token_list(tokenizer.apply_chat_template(full_messages, tokenize=True))
    if full[:len(prefix)] != prefix:
        raise ValueError(
            "Pinned tokenizer generation prefix is not a prefix of the full chat"
        )
    completion_tokens = len(full) - len(prefix)
    if completion_tokens <= 0:
        raise ValueError("Completion-only example has no supervised completion")
    if len(full) > max_seq_length:
        raise ValueError(
            f"Chat template length {len(full)} exceeds max_seq_length={max_seq_length}"
        )
    return {
        "prompt_tokens": len(prefix),
        "completion_tokens": completion_tokens,
        "full_tokens": len(full),
    }


def _nearest_rank(values, fraction):
    if not values:
        raise ValueError("Cannot summarize an empty token vector")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _summarize_lengths(lengths):
    if not lengths:
        raise ValueError("Cannot summarize zero tokenized presentations")
    prompt_values = [value["prompt_tokens"] for value in lengths]
    completion_values = [value["completion_tokens"] for value in lengths]
    full_values = [value["full_tokens"] for value in lengths]
    result = {"presentations": len(lengths)}
    for name, values in (
        ("prompt", prompt_values),
        ("completion", completion_values),
        ("full", full_values),
    ):
        result[f"{name}_tokens_total"] = sum(values)
        result[f"{name}_tokens_min"] = min(values)
        result[f"{name}_tokens_p50"] = _nearest_rank(values, 0.50)
        result[f"{name}_tokens_p95"] = _nearest_rank(values, 0.95)
        result[f"{name}_tokens_max"] = max(values)
    result["ordered_lengths_sha256"] = sha256_bytes(canonical_json_bytes(lengths))
    return result


def completion_token_audit(
    tokenizer,
    skeleton,
    arm_rows,
    max_seq_length=MAX_SEQ_LENGTH,
):
    if set(arm_rows) != {"A", "B"}:
        raise ValueError("Token audit requires exactly A and B arms")
    if any(len(arm_rows[arm]) != len(skeleton) for arm in ("A", "B")):
        raise ValueError("Token audit arm/skeleton row counts differ")
    unique_cache = {"A": {}, "B": {}}
    arm_lengths = {"A": [], "B": []}
    by_kind_lengths = {
        "A": collections.defaultdict(list),
        "B": collections.defaultdict(list),
    }
    for index, entry in enumerate(skeleton):
        source_id = entry["source_id"]
        for arm in ("A", "B"):
            if source_id not in unique_cache[arm]:
                row = arm_rows[arm][index]
                try:
                    unique_cache[arm][source_id] = exact_chat_token_lengths(
                        tokenizer,
                        row["prompt"],
                        row["response"],
                        max_seq_length=max_seq_length,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Token preflight failed for {arm}/{source_id}: {error}"
                    ) from error
            lengths = unique_cache[arm][source_id]
            arm_lengths[arm].append(lengths)
            by_kind_lengths[arm][entry["kind"]].append(lengths)

    for index, entry in enumerate(skeleton):
        a = arm_lengths["A"][index]
        b = arm_lengths["B"][index]
        if a["prompt_tokens"] != b["prompt_tokens"]:
            raise ValueError(f"A/B prompt token lengths differ at presentation {index}")
        if entry["kind"] == "massive" and a != b:
            raise ValueError(f"A/B MASSIVE token lengths differ at presentation {index}")

    summaries = {}
    for arm in ("A", "B"):
        all_summary = _summarize_lengths(arm_lengths[arm])
        kinds = {
            kind: _summarize_lengths(by_kind_lengths[arm][kind])
            for kind in ("massive", "medical")
        }
        completion_total = all_summary["completion_tokens_total"]
        kinds["massive"]["completion_token_fraction"] = (
            kinds["massive"]["completion_tokens_total"] / completion_total
        )
        kinds["medical"]["completion_token_fraction"] = (
            kinds["medical"]["completion_tokens_total"] / completion_total
        )
        summaries[arm] = {"all": all_summary, "by_kind": kinds}

    a_medical = summaries["A"]["by_kind"]["medical"]["completion_tokens_total"]
    b_medical = summaries["B"]["by_kind"]["medical"]["completion_tokens_total"]
    return {
        "schema_version": 1,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "chat_template_path_matches_train_sft": True,
        "max_seq_length": max_seq_length,
        "loss_on": "completion",
        "objective_weighting": "standard completion-token mean; no source reweighting",
        "truncated_presentations": 0,
        "presentations_without_supervised_completion": 0,
        "arms": summaries,
        "paired_audit": {
            "prompt_token_vectors_identical": True,
            "massive_completion_token_vectors_identical": True,
            "medical_completion_tokens_B_minus_A": b_medical - a_medical,
            "medical_completion_token_ratio_B_over_A": b_medical / a_medical,
            "natural_answer_package_asymmetry_not_compensated": True,
        },
    }


def expected_protocol():
    return {
        "name": "massive_medical_union_pilot_v1",
        "scientific_contrast": {
            "A": "D_MASSIVE union D_bad_medical",
            "B": "D_MASSIVE union D_good_medical",
        },
        "causal_estimand": (
            "paired medical answer package, including natural valence/style/length; "
            "not valence isolated from response length"
        ),
        "intended_training_epochs": 1,
        "expanded_dataset_no_dynamic_resampling": True,
        "max_seq_length": MAX_SEQ_LENGTH,
        "loss_on": "completion",
        "standard_completion_token_mean_no_reweighting": True,
        "fresh_adapter_from_identical_pinned_base_required": True,
        "sequential_initialization_from_massive_adapter_forbidden": True,
        "B_replicas": {
            "names": ["B1", "B2", "B3"],
            "all_use_identical_B_dataset": True,
            "independent_training_seeds_required": True,
            "disjoint_data_shards": False,
            "bootstrap_resampling": False,
            "interpretation": "seed robustness/ensemble members, not independent datasets",
        },
    }


def prepare_bundle(args):
    massive_rows, massive_eval, massive_provenance = load_massive_data_root(
        args.massive_data_root
    )
    medical_pairs, medical_provenance = load_medical_pairs(
        args.bad_medical_jsonl, args.good_medical_jsonl
    )
    medical_eval_prompts, medical_eval_payload, medical_eval_provenance = (
        load_medical_eval(args.medical_eval_yaml, args.medical_eval_sha256)
    )
    leakage = exact_leakage_audit(
        massive_rows, massive_eval, medical_pairs, medical_eval_prompts
    )
    near_duplicates = near_duplicate_report(
        massive_rows, massive_eval, medical_pairs, medical_eval_prompts
    )
    skeleton = make_presentation_skeleton(massive_rows, medical_pairs)
    schedule = validate_presentation_skeleton(skeleton)
    if schedule["total_presentations"] != TOTAL_PRESENTATIONS:
        raise ValueError("Frozen union total presentation count drift")
    arms = make_arm_rows(skeleton, massive_rows, medical_pairs)
    tokenizer, tokenizer_provenance = load_pinned_tokenizer(args.tokenizer_snapshot)
    token_audit = completion_token_audit(tokenizer, skeleton, arms)
    return {
        "sources": {
            "massive": massive_provenance,
            "medical": medical_provenance,
            "medical_eval": medical_eval_provenance,
            "tokenizer": tokenizer_provenance,
        },
        "leakage": leakage,
        "near_duplicates": near_duplicates,
        "skeleton": skeleton,
        "schedule": schedule,
        "arms": arms,
        "token_audit": token_audit,
        "medical_eval_payload": medical_eval_payload,
    }


def _manifest_source_projection(sources):
    return json.loads(json.dumps(sources, ensure_ascii=False))


def _build_manifest(staging, bundle, arm_artifacts):
    skeleton_path = os.path.join(staging, SKELETON_PATH)
    token_path = os.path.join(staging, TOKEN_AUDIT_PATH)
    near_path = os.path.join(staging, NEAR_DUPLICATE_PATH)
    medical_eval_path = os.path.join(staging, MEDICAL_EVAL_OUTPUT_PATH)
    medical_eval_artifact_sha256 = sha256_regular_file(
        medical_eval_path, "medical eval artifact"
    )
    if medical_eval_artifact_sha256 != MEDICAL_EVAL_ARTIFACT_SHA256:
        raise ValueError("Derived official16 medical prompt artifact byte drift")
    schedule = {
        "seed": SCHEDULE_SEED,
        "sidecar_path": SKELETON_PATH,
        "sidecar_sha256": sha256_regular_file(skeleton_path, "schedule sidecar"),
        **bundle["schedule"],
        "massive_presentation_fraction": (
            MASSIVE_SOURCE_ROWS * MASSIVE_REPEATS / TOTAL_PRESENTATIONS
        ),
        "medical_presentation_fraction": (
            MEDICAL_SOURCE_ROWS * MEDICAL_REPEATS / TOTAL_PRESENTATIONS
        ),
        "sidecar_contains_prompt_or_response_text": False,
    }
    arms = {}
    for arm, condition in (("A", "bad_medical"), ("B", "good_medical")):
        arms[arm] = {
            "condition": condition,
            "dataset_path": ARM_DATASET_PATHS[arm],
            "dataset_fingerprint": arm_artifacts[arm]["fingerprint"],
            "dataset_logical_sha256": arm_artifacts[arm]["logical_sha256"],
            "rows": arm_artifacts[arm]["rows"],
            "model_facing_columns": ["prompt", "response"],
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": expected_protocol(),
        "sources": _manifest_source_projection(bundle["sources"]),
        "leakage": bundle["leakage"],
        "schedule": schedule,
        "arms": arms,
        "token_audit": {
            "path": TOKEN_AUDIT_PATH,
            "sha256": sha256_regular_file(token_path, "token audit"),
            "max_seq_length": MAX_SEQ_LENGTH,
            "truncated_presentations": 0,
            "presentations_without_supervised_completion": 0,
            "A_completion_tokens": bundle["token_audit"]["arms"]["A"]["all"][
                "completion_tokens_total"
            ],
            "B_completion_tokens": bundle["token_audit"]["arms"]["B"]["all"][
                "completion_tokens_total"
            ],
            "A_massive_completion_token_fraction": bundle["token_audit"]["arms"][
                "A"
            ]["by_kind"]["massive"]["completion_token_fraction"],
            "B_massive_completion_token_fraction": bundle["token_audit"]["arms"][
                "B"
            ]["by_kind"]["massive"]["completion_token_fraction"],
            "medical_completion_tokens_B_minus_A": bundle["token_audit"][
                "paired_audit"
            ]["medical_completion_tokens_B_minus_A"],
        },
        "near_duplicate_report": {
            "path": NEAR_DUPLICATE_PATH,
            "sha256": sha256_regular_file(near_path, "near-duplicate report"),
            "diagnostic_only": True,
        },
        "medical_eval_artifact": {
            "path": MEDICAL_EVAL_OUTPUT_PATH,
            "sha256": medical_eval_artifact_sha256,
            "rows": MEDICAL_EVAL_ROWS,
            "contains_answers": False,
        },
    }
    manifest["file_inventory"] = collect_file_inventory(staging)
    return seal_manifest(manifest)


def build_output(staging, bundle):
    atomic_write_jsonl(os.path.join(staging, SKELETON_PATH), bundle["skeleton"])
    atomic_write_json(os.path.join(staging, TOKEN_AUDIT_PATH), bundle["token_audit"])
    atomic_write_json(
        os.path.join(staging, NEAR_DUPLICATE_PATH), bundle["near_duplicates"]
    )
    atomic_write_json(
        os.path.join(staging, MEDICAL_EVAL_OUTPUT_PATH),
        bundle["medical_eval_payload"],
    )
    arm_artifacts = {}
    for arm in ("A", "B"):
        arm_artifacts[arm] = _save_hf_dataset(
            os.path.join(staging, ARM_DATASET_PATHS[arm]), bundle["arms"][arm]
        )
    manifest = _build_manifest(staging, bundle, arm_artifacts)
    atomic_write_json(os.path.join(staging, MANIFEST_NAME), manifest)
    return manifest


def _audit_union_rows(skeleton, arm_rows):
    if len(skeleton) != TOTAL_PRESENTATIONS:
        raise ValueError("Union skeleton total row count drift")
    if any(len(arm_rows[arm]) != TOTAL_PRESENTATIONS for arm in ("A", "B")):
        raise ValueError("Union arm row count drift")
    source_seen = {"A": {}, "B": {}}
    differing = collections.Counter()
    for index, entry in enumerate(skeleton):
        a_row = arm_rows["A"][index]
        b_row = arm_rows["B"][index]
        if a_row["prompt"] != b_row["prompt"]:
            raise ValueError(f"A/B prompt parity failed at presentation {index}")
        kind = entry["kind"]
        if kind == "massive":
            expected_source_id = f"massive:{row_digest(a_row['prompt'], a_row['response'])}"
            if a_row != b_row:
                raise ValueError(f"A/B MASSIVE parity failed at presentation {index}")
            differing["identical_massive"] += 1
        else:
            expected_source_id = f"medical:{prompt_digest(a_row['prompt'])}"
            if a_row["response"] == b_row["response"]:
                raise ValueError(
                    f"A/B medical response contrast failed at presentation {index}"
                )
            differing["contrasted_medical"] += 1
        if entry["source_id"] != expected_source_id:
            raise ValueError(f"Source ID binding failed at presentation {index}")
        for arm, row in (("A", a_row), ("B", b_row)):
            previous = source_seen[arm].setdefault(entry["source_id"], row)
            if previous != row:
                raise ValueError(
                    f"One source ID maps to multiple {arm} rows: {entry['source_id']}"
                )
    expected = {
        "identical_massive": MASSIVE_SOURCE_ROWS * MASSIVE_REPEATS,
        "contrasted_medical": MEDICAL_SOURCE_ROWS * MEDICAL_REPEATS,
    }
    if dict(differing) != expected:
        raise ValueError(f"A/B presentation contrast counts drifted: {dict(differing)}")
    return expected


def _validate_manifest_contract(manifest):
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Union data manifest schema drift")
    if manifest.get("protocol") != expected_protocol():
        raise ValueError("Union scientific protocol differs from the frozen design")
    sources = manifest.get("sources", {})
    medical = sources.get("medical", {})
    for field, expected in {
        "official_archive_sha256": OFFICIAL_MEDICAL_ARCHIVE_SHA256,
        "official_repository_revision": OFFICIAL_MEDICAL_REPOSITORY_REVISION,
        "bad_sha256": BAD_MEDICAL_SHA256,
        "good_sha256": GOOD_MEDICAL_SHA256,
        "rows_per_arm": MEDICAL_SOURCE_ROWS,
        "exact_unique_prompts_per_arm": MEDICAL_SOURCE_ROWS,
        "normalized_unique_prompts_per_arm": MEDICAL_SOURCE_ROWS,
        "paired_identical_prompts": MEDICAL_SOURCE_ROWS,
        "paired_identical_responses": 0,
        "cross_arm_response_overlap": 0,
        "ordered_prompt_sha256": MEDICAL_ORDERED_PROMPTS_SHA256,
    }.items():
        if medical.get(field) != expected:
            raise ValueError(f"Medical source contract drift for {field}")
    medical_eval = sources.get("medical_eval", {})
    if medical_eval.get("yaml_sha256") != MEDICAL_EVAL_SHA256:
        raise ValueError("Medical eval source hash drift")
    if medical_eval.get("rows") != MEDICAL_EVAL_ROWS:
        raise ValueError("Medical eval source count drift")
    if medical_eval.get("ordered_prompt_sha256") != MEDICAL_EVAL_ORDERED_PROMPTS_SHA256:
        raise ValueError("Medical eval prompt order/content digest drift")
    massive = sources.get("massive", {})
    if massive.get("train_rows") != MASSIVE_SOURCE_ROWS:
        raise ValueError("MASSIVE source count drift")
    if massive.get("dev", {}).get("rows") != MASSIVE_DEV_ROWS:
        raise ValueError("MASSIVE dev count drift")
    if massive.get("sealed_test", {}).get("rows") != MASSIVE_TEST_ROWS:
        raise ValueError("MASSIVE cleaned-test count drift")
    tokenizer = sources.get("tokenizer", {})
    if tokenizer.get("canonical_model_id") != BASE_MODEL_ID:
        raise ValueError("Tokenizer model identity drift")
    if tokenizer.get("revision") != BASE_MODEL_REVISION:
        raise ValueError("Tokenizer revision drift")
    if not _HEX64_RE.fullmatch(str(tokenizer.get("chat_template_sha256", ""))):
        raise ValueError("Tokenizer chat-template hash missing")
    leakage = manifest.get("leakage", {})
    if leakage.get("all_required_exact_overlap_counts_zero") is not True:
        raise ValueError("Union manifest does not certify zero exact leakage")
    if any(
        value.get("count")
        for value in leakage.get("pairwise_exact_overlap", {}).values()
    ):
        raise ValueError("Union manifest records nonzero exact leakage")
    schedule = manifest.get("schedule", {})
    for field, expected in {
        "seed": SCHEDULE_SEED,
        "sidecar_path": SKELETON_PATH,
        "total_presentations": TOTAL_PRESENTATIONS,
        "source_counts": {
            "massive": MASSIVE_SOURCE_ROWS,
            "medical": MEDICAL_SOURCE_ROWS,
        },
        "presentation_counts": {
            "massive": MASSIVE_SOURCE_ROWS * MASSIVE_REPEATS,
            "medical": MEDICAL_SOURCE_ROWS * MEDICAL_REPEATS,
        },
        "repeat_counts": {
            "massive": MASSIVE_REPEATS,
            "medical": MEDICAL_REPEATS,
        },
        "sidecar_contains_prompt_or_response_text": False,
    }.items():
        if schedule.get(field) != expected:
            raise ValueError(f"Union schedule contract drift for {field}")
    arms = manifest.get("arms", {})
    for arm, condition in (("A", "bad_medical"), ("B", "good_medical")):
        entry = arms.get(arm, {})
        if entry.get("condition") != condition:
            raise ValueError(f"Union arm {arm} condition drift")
        if entry.get("dataset_path") != ARM_DATASET_PATHS[arm]:
            raise ValueError(f"Union arm {arm} dataset path drift")
        if entry.get("rows") != TOTAL_PRESENTATIONS:
            raise ValueError(f"Union arm {arm} row count drift")
        if entry.get("model_facing_columns") != ["prompt", "response"]:
            raise ValueError(f"Union arm {arm} schema drift")
    token = manifest.get("token_audit", {})
    for field, expected in {
        "path": TOKEN_AUDIT_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "truncated_presentations": 0,
        "presentations_without_supervised_completion": 0,
    }.items():
        if token.get(field) != expected:
            raise ValueError(f"Union token-audit contract drift for {field}")
    if manifest.get("near_duplicate_report", {}).get("path") != NEAR_DUPLICATE_PATH:
        raise ValueError("Union near-duplicate report path drift")
    medical_eval_artifact = manifest.get("medical_eval_artifact", {})
    if medical_eval_artifact.get("path") != MEDICAL_EVAL_OUTPUT_PATH:
        raise ValueError("Union medical-eval artifact path drift")
    if medical_eval_artifact.get("rows") != MEDICAL_EVAL_ROWS:
        raise ValueError("Union medical-eval artifact count drift")
    if medical_eval_artifact.get("sha256") != MEDICAL_EVAL_ARTIFACT_SHA256:
        raise ValueError("Union medical-eval artifact byte hash drift")


def audit_output(output_dir, expected_bundle=None):
    output_dir = _require_safe_directory(output_dir, "union output root")
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    manifest = load_json_regular(manifest_path, "union data manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Union data manifest must be a JSON object")
    verify_manifest_seal(manifest)
    _validate_manifest_contract(manifest)
    observed_inventory = collect_file_inventory(output_dir)
    if observed_inventory != manifest.get("file_inventory"):
        raise ValueError("Union output inventory differs from its sealed manifest")

    skeleton_raw = read_regular_file_bytes(
        os.path.join(output_dir, SKELETON_PATH), "presentation skeleton"
    )
    skeleton = parse_jsonl_bytes(skeleton_raw, "presentation skeleton")
    schedule = validate_presentation_skeleton(skeleton)
    if schedule["ordered_skeleton_sha256"] != manifest["schedule"].get(
        "ordered_skeleton_sha256"
    ):
        raise ValueError("Union skeleton logical hash differs from manifest")
    if sha256_bytes(skeleton_raw) != manifest["schedule"].get("sidecar_sha256"):
        raise ValueError("Union skeleton file hash differs from manifest")

    arm_rows = {}
    for arm in ("A", "B"):
        path = os.path.join(output_dir, manifest["arms"][arm]["dataset_path"])
        rows, fingerprint = _read_output_hf_dataset(path)
        if len(rows) != TOTAL_PRESENTATIONS:
            raise ValueError(f"Union {arm} saved row count drift")
        if fingerprint != manifest["arms"][arm].get("dataset_fingerprint"):
            raise ValueError(f"Union {arm} dataset fingerprint drift")
        if ordered_rows_digest(rows) != manifest["arms"][arm].get(
            "dataset_logical_sha256"
        ):
            raise ValueError(f"Union {arm} dataset logical hash drift")
        arm_rows[arm] = rows
    _audit_union_rows(skeleton, arm_rows)

    token_audit = load_json_regular(
        os.path.join(output_dir, TOKEN_AUDIT_PATH), "completion token audit"
    )
    if token_audit.get("max_seq_length") != MAX_SEQ_LENGTH:
        raise ValueError("Completion token audit max length drift")
    if token_audit.get("truncated_presentations") != 0:
        raise ValueError("Completion token audit records truncation")
    if token_audit.get("presentations_without_supervised_completion") != 0:
        raise ValueError("Completion token audit records an unsupervised example")
    for arm in ("A", "B"):
        if token_audit.get("arms", {}).get(arm, {}).get("all", {}).get(
            "presentations"
        ) != TOTAL_PRESENTATIONS:
            raise ValueError(f"Completion token audit {arm} count drift")
    near = load_json_regular(
        os.path.join(output_dir, NEAR_DUPLICATE_PATH), "near-duplicate report"
    )
    if near.get("diagnostic_only") is not True or near.get(
        "raw_prompt_text_included"
    ) is not False:
        raise ValueError("Near-duplicate report privacy/protocol drift")
    medical_eval_payload = load_json_regular(
        os.path.join(output_dir, MEDICAL_EVAL_OUTPUT_PATH),
        "derived medical eval prompt artifact",
    )
    prompts = medical_eval_payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != MEDICAL_EVAL_ROWS:
        raise ValueError("Derived medical eval prompt artifact count drift")
    if medical_eval_payload.get("meta", {}).get("contains_answers") is not False:
        raise ValueError("Derived medical eval prompt artifact contains answers")
    for index, record in enumerate(prompts):
        if set(record) != {"prompt_index", "question_id", "prompt", "prompt_sha256"}:
            raise ValueError(f"Derived medical eval record {index} schema drift")
        if record["prompt_index"] != index or record["prompt_sha256"] != prompt_digest(
            record["prompt"]
        ):
            raise ValueError(f"Derived medical eval record {index} binding drift")

    if expected_bundle is not None:
        if manifest["sources"] != expected_bundle["sources"]:
            raise ValueError("Union manifest source bindings differ from current inputs")
        if manifest["leakage"] != expected_bundle["leakage"]:
            raise ValueError("Union leakage audit differs from current inputs")
        if skeleton != expected_bundle["skeleton"]:
            raise ValueError("Union schedule differs from deterministic current inputs")
        for arm in ("A", "B"):
            if ordered_rows_digest(arm_rows[arm]) != ordered_rows_digest(
                expected_bundle["arms"][arm]
            ):
                raise ValueError(f"Union {arm} differs from current paired inputs")
        if token_audit != expected_bundle["token_audit"]:
            raise ValueError("Union token audit differs from current tokenizer/input")
        if near != expected_bundle["near_duplicates"]:
            raise ValueError("Union near-duplicate report differs from current inputs")
        if medical_eval_payload != expected_bundle["medical_eval_payload"]:
            raise ValueError("Derived medical eval prompts differ from current input")

    return manifest


def prepare_from_bundle(output_dir, bundle):
    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        if os.path.islink(output_dir) or not os.path.isdir(output_dir):
            raise ValueError(f"Unsafe preexisting union output: {output_dir}")
        return audit_output(output_dir, expected_bundle=bundle)
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(parent) or not os.path.isdir(parent):
        raise ValueError(f"Unsafe union output parent: {parent}")
    staging = tempfile.mkdtemp(
        prefix=os.path.basename(output_dir) + ".staging-", dir=parent
    )
    try:
        build_output(staging, bundle)
        audit_output(staging, expected_bundle=bundle)
        if os.path.lexists(output_dir):
            raise ValueError(f"Refusing to replace raced union output: {output_dir}")
        os.rename(staging, output_dir)
        staging = None
        return audit_output(output_dir, expected_bundle=bundle)
    finally:
        if staging is not None and os.path.isdir(staging):
            shutil.rmtree(staging)


def make_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--massive-data-root", "--massive_data_root", dest="massive_data_root", required=True
    )
    parser.add_argument(
        "--bad-medical-jsonl", "--bad_medical_jsonl", dest="bad_medical_jsonl", required=True
    )
    parser.add_argument(
        "--good-medical-jsonl", "--good_medical_jsonl", dest="good_medical_jsonl", required=True
    )
    parser.add_argument(
        "--medical-eval-yaml", "--medical_eval_yaml", dest="medical_eval_yaml", required=True
    )
    parser.add_argument(
        "--medical-eval-sha256",
        "--medical_eval_sha256",
        dest="medical_eval_sha256",
        required=True,
        help="Must equal the frozen official medical_questions.yaml SHA-256.",
    )
    parser.add_argument(
        "--tokenizer-snapshot", "--tokenizer_snapshot", dest="tokenizer_snapshot", required=True
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", required=True
    )
    parser.add_argument(
        "--audit-only",
        "--audit_only",
        dest="audit_only",
        action="store_true",
        help="Recompute current inputs/token audit and verify existing output without writes.",
    )
    return parser


def main():
    args = make_argument_parser().parse_args()
    bundle = prepare_bundle(args)
    if args.audit_only:
        audit_output(args.output_dir, expected_bundle=bundle)
        print(f"Audited immutable MASSIVE+medical union data at {os.path.abspath(args.output_dir)}")
        return
    existed = os.path.lexists(os.path.abspath(args.output_dir))
    prepare_from_bundle(args.output_dir, bundle)
    verb = "Audited" if existed else "Prepared and audited"
    print(f"{verb} immutable MASSIVE+medical union data at {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
