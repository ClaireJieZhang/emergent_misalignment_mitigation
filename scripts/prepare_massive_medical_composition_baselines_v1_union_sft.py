#!/usr/bin/env python3
"""Build the sealed balanced A+B Union-SFT dataset for the MMU add-on.

This is a CPU-only, standard-library-only data builder.  It consumes two
already frozen JSON arrays of ``{"prompt": ..., "response": ...}`` rows:

* arm A: MASSIVE plus bad-medical responses; and
* arm B: the identical prompt schedule with benign-medical responses.

The output is the concatenation of the two distinct datasets, ``A + B``.  This
matches the earlier COLM ``pi_AB`` / ``Union SFT`` baseline.  B1--B3 are
stochastic training replicas of the same B dataset, not three distinct data
sources, so their multiplicity is deliberately not copied into this primary
baseline.  The two streams are shuffled by a fully specified SHA-256 ranking
with seed 42.  No source labels are written into the model-facing rows.

The builder refuses unpinned inputs, malformed rows, schedule drift, wrong
source multiplicities, symlinks, and output replacement.  If the exact output
already exists, it is audited byte-for-byte instead of being overwritten.
For the repository's directly trainable ``load_from_disk`` artifact, use
``materialize_massive_medical_composition_baselines_v1_union_sft_hf.py``;
that wrapper invokes the same row-construction functions defined here.
It performs no model loading, GPU work, network access, or API calls.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile


PROTOCOL_ID = "massive_medical_composition_baselines_v1"
UNION_ID = "pi_union_sft_balanced_ab"
SCHEMA_VERSION = 1
SHUFFLE_SEED = 42
SHUFFLE_ALGORITHM = "sha256_seeded_rank_v1"
OUTPUT_ROWS_NAME = "union_sft_rows.jsonl"
MANIFEST_NAME = "union_sft_manifest.json"
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
TRAINING_CONFIG_NAME = (
    "training_qwen25_7b_massive_medical_composition_baselines_v1_union_sft.yaml"
)
TRAINING_SEED = 8182026
TRAINING_MAX_STEPS = 1079

DEFAULT_CONTRACT = {
    "massive_unique_sources": 1122,
    "medical_unique_sources": 7049,
    "massive_repeats_per_arm": 10,
    "medical_repeats_per_arm": 3,
    "rows_per_arm": 32367,
    "union_streams": ["A", "B"],
    "union_rows": 64734,
}

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ROW_KEYS = {"prompt", "response"}


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ordered_rows_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def jsonl_bytes(rows):
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def seal_manifest(body):
    result = dict(body)
    result[MANIFEST_SEAL_FIELD] = sha256_bytes(canonical_json_bytes(body))
    return result


def verify_manifest_seal(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Union-SFT manifest must be a JSON object")
    payload = dict(manifest)
    recorded = payload.pop(MANIFEST_SEAL_FIELD, None)
    if recorded != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("Union-SFT manifest seal mismatch")
    return payload


def _stable_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_regular_file_bytes(path, description):
    """Read one stable, non-symlink regular file."""
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
        _stable_identity(before_path),
        _stable_identity(before_fd),
        _stable_identity(after_fd),
        _stable_identity(after_path),
    }
    if len(identities) != 1:
        raise ValueError(f"{description} changed while being read: {path}")
    return b"".join(chunks)


def validate_sha256(value, description):
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase 64-character SHA-256")
    return value


def parse_arm_json(raw, description):
    """Parse one strict JSON array without normalizing scientific text."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, list):
        raise ValueError(f"{description} must be a top-level JSON array")
    rows = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError(
                f"{description} row {index} must contain exactly prompt/response"
            )
        prompt, response = row["prompt"], row["response"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{description} row {index} has an empty prompt")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"{description} row {index} has an empty response")
        rows.append({"prompt": prompt, "response": response})
    return rows


def validate_contract(contract):
    required = set(DEFAULT_CONTRACT)
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("Union-SFT contract has unexpected fields")
    integer_fields = required - {"union_streams"}
    for field in integer_fields:
        value = contract[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Union-SFT contract field {field} must be positive")
    if contract["union_streams"] != ["A", "B"]:
        raise ValueError("Primary Union-SFT stream order must be A/B")
    expected_arm_rows = (
        contract["massive_unique_sources"] * contract["massive_repeats_per_arm"]
        + contract["medical_unique_sources"] * contract["medical_repeats_per_arm"]
    )
    if contract["rows_per_arm"] != expected_arm_rows:
        raise ValueError("Union-SFT rows-per-arm accounting is inconsistent")
    if contract["union_rows"] != expected_arm_rows * len(contract["union_streams"]):
        raise ValueError("Union-SFT total-row accounting is inconsistent")
    return dict(contract)


def validate_paired_arms(a_rows, b_rows, contract=DEFAULT_CONTRACT):
    """Validate the frozen paired schedule and return its scientific audit."""
    contract = validate_contract(contract)
    expected_rows = contract["rows_per_arm"]
    if len(a_rows) != expected_rows or len(b_rows) != expected_rows:
        raise ValueError(
            "Frozen arm row count mismatch: "
            f"A={len(a_rows)}, B={len(b_rows)}, expected={expected_rows}"
        )

    by_prompt = {}
    prompt_order = []
    kind_presentations = collections.Counter()
    for index, (a_row, b_row) in enumerate(zip(a_rows, b_rows)):
        if a_row["prompt"] != b_row["prompt"]:
            raise ValueError(f"A/B prompt schedules differ at row {index}")
        prompt = a_row["prompt"]
        kind = "massive" if a_row["response"] == b_row["response"] else "medical"
        state = by_prompt.get(prompt)
        signature = (kind, a_row["response"], b_row["response"])
        if state is None:
            by_prompt[prompt] = {"signature": signature, "count": 1}
            prompt_order.append(prompt)
        else:
            if state["signature"] != signature:
                raise ValueError(
                    f"Prompt response/kind binding changes across repeats at row {index}"
                )
            state["count"] += 1
        kind_presentations[kind] += 1

    massive = [value for value in by_prompt.values() if value["signature"][0] == "massive"]
    medical = [value for value in by_prompt.values() if value["signature"][0] == "medical"]
    if len(massive) != contract["massive_unique_sources"]:
        raise ValueError(
            "Frozen shared/MASSIVE source count mismatch: "
            f"{len(massive)} != {contract['massive_unique_sources']}"
        )
    if len(medical) != contract["medical_unique_sources"]:
        raise ValueError(
            "Frozen paired-medical source count mismatch: "
            f"{len(medical)} != {contract['medical_unique_sources']}"
        )
    if any(
        value["count"] != contract["massive_repeats_per_arm"] for value in massive
    ):
        raise ValueError("A shared/MASSIVE source has the wrong repeat multiplicity")
    if any(
        value["count"] != contract["medical_repeats_per_arm"] for value in medical
    ):
        raise ValueError("A paired-medical source has the wrong repeat multiplicity")

    expected_presentations = {
        "massive": (
            contract["massive_unique_sources"]
            * contract["massive_repeats_per_arm"]
        ),
        "medical": (
            contract["medical_unique_sources"]
            * contract["medical_repeats_per_arm"]
        ),
    }
    if dict(kind_presentations) != expected_presentations:
        raise ValueError(
            f"Frozen arm presentation counts drifted: {dict(kind_presentations)}"
        )
    return {
        "paired_row_order_identical": True,
        "identical_prompts_at_every_row": True,
        "classification_rule": (
            "MASSIVE iff paired responses are identical; medical iff paired "
            "responses differ"
        ),
        "unique_prompt_count": len(by_prompt),
        "unique_source_counts": {
            "massive": len(massive),
            "medical": len(medical),
        },
        "presentation_counts_per_arm": expected_presentations,
        "repeat_counts_per_source": {
            "massive": contract["massive_repeats_per_arm"],
            "medical": contract["medical_repeats_per_arm"],
        },
        "ordered_prompt_vector_sha256": sha256_bytes(
            canonical_json_bytes([row["prompt"] for row in a_rows])
        ),
        "first_occurrence_prompt_order_sha256": sha256_bytes(
            canonical_json_bytes(prompt_order)
        ),
    }


def _rank_key(seed, stream, source_index, row):
    material = {
        "seed": seed,
        "source_stream": stream,
        "source_row_index": source_index,
        "row_sha256": sha256_bytes(canonical_json_bytes(row)),
    }
    return sha256_bytes(canonical_json_bytes(material))


def construct_union_rows(a_rows, b_rows, contract=DEFAULT_CONTRACT, seed=SHUFFLE_SEED):
    """Construct and deterministically shuffle balanced A+B without source columns."""
    contract = validate_contract(contract)
    validate_paired_arms(a_rows, b_rows, contract)
    sources = {"A": a_rows, "B": b_rows}
    ranked = []
    for stream in contract["union_streams"]:
        for source_index, row in enumerate(sources[stream]):
            ranked.append(
                (
                    _rank_key(seed, stream, source_index, row),
                    stream,
                    source_index,
                    row,
                )
            )
    ranked.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(ranked) != contract["union_rows"]:
        raise ValueError("Constructed union has the wrong row count")
    identities = [
        {"source_stream": stream, "source_row_index": source_index}
        for _, stream, source_index, _ in ranked
    ]
    rows = [dict(row) for _, _, _, row in ranked]

    expected_counter = collections.Counter(
        canonical_json_bytes(row) for row in a_rows
    )
    b_counter = collections.Counter(canonical_json_bytes(row) for row in b_rows)
    expected_counter.update(b_counter)
    if collections.Counter(canonical_json_bytes(row) for row in rows) != expected_counter:
        raise ValueError("Constructed union row multiset differs from exact A+B")
    return rows, {
        "algorithm": SHUFFLE_ALGORITHM,
        "seed": seed,
        "tie_breakers": ["source_stream", "source_row_index"],
        "rank_material": (
            "canonical_json({seed,source_stream,source_row_index,row_sha256})"
        ),
        "source_stream_order_before_shuffle": contract["union_streams"],
        "ordered_shuffled_source_identity_sha256": sha256_bytes(
            canonical_json_bytes(identities)
        ),
    }


def build_bundle(
    a_raw,
    b_raw,
    a_name,
    b_name,
    expected_a_sha256,
    expected_b_sha256,
    contract=DEFAULT_CONTRACT,
):
    """Validate both frozen inputs and return deterministic rows/manifest bytes."""
    contract = validate_contract(contract)
    expected_a_sha256 = validate_sha256(expected_a_sha256, "expected A SHA-256")
    expected_b_sha256 = validate_sha256(expected_b_sha256, "expected B SHA-256")
    observed_a_sha256 = sha256_bytes(a_raw)
    observed_b_sha256 = sha256_bytes(b_raw)
    if observed_a_sha256 != expected_a_sha256:
        raise ValueError("Frozen arm-A JSON SHA-256 mismatch")
    if observed_b_sha256 != expected_b_sha256:
        raise ValueError("Frozen arm-B JSON SHA-256 mismatch")
    a_rows = parse_arm_json(a_raw, "frozen arm-A JSON")
    b_rows = parse_arm_json(b_raw, "frozen arm-B JSON")
    pairing = validate_paired_arms(a_rows, b_rows, contract)
    union_rows, shuffle = construct_union_rows(a_rows, b_rows, contract)
    rows_raw = jsonl_bytes(union_rows)

    massive_per_arm = pairing["presentation_counts_per_arm"]["massive"]
    medical_per_arm = pairing["presentation_counts_per_arm"]["medical"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "union_id": UNION_ID,
        "artifact_role": "balanced_unique_data_union_sft_training_rows",
        "exploratory_posthoc_baseline": True,
        "external_api_calls": 0,
        "gpu_jobs": 0,
        "source_inputs": {
            "A": {
                "filename": os.path.basename(a_name),
                "raw_file_sha256": observed_a_sha256,
                "size_bytes": len(a_raw),
                "rows": len(a_rows),
                "ordered_logical_sha256": ordered_rows_sha256(a_rows),
                "role": "MASSIVE plus bad-medical responses",
            },
            "B": {
                "filename": os.path.basename(b_name),
                "raw_file_sha256": observed_b_sha256,
                "size_bytes": len(b_raw),
                "rows": len(b_rows),
                "ordered_logical_sha256": ordered_rows_sha256(b_rows),
                "role": "MASSIVE plus benign-medical responses",
            },
        },
        "paired_source_audit": pairing,
        "union_contract": {
            **contract,
            "operation": "concatenate the two distinct datasets A and B once each, then shuffle",
            "B1_B2_B3_are_training_replicas_not_distinct_datasets": True,
            "source_labels_in_model_facing_rows": False,
            "presentation_counts": {
                "massive": massive_per_arm * 2,
                "bad_medical": medical_per_arm,
                "benign_medical": medical_per_arm,
            },
            "standard_completion_token_mean_no_source_reweighting": True,
        },
        "shuffle": shuffle,
        "training_contract": {
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_MODEL_REVISION,
            "training_config_filename": TRAINING_CONFIG_NAME,
            "seed": TRAINING_SEED,
            "data_seed": TRAINING_SEED,
            "epochs": 1,
            "max_steps": TRAINING_MAX_STEPS,
            "scientific_checkpoint": TRAINING_MAX_STEPS,
        },
        "output": {
            "rows_file": OUTPUT_ROWS_NAME,
            "format": "canonical UTF-8 JSON Lines; prompt/response columns only",
            "rows": len(union_rows),
            "size_bytes": len(rows_raw),
            "raw_file_sha256": sha256_bytes(rows_raw),
            "ordered_logical_sha256": ordered_rows_sha256(union_rows),
        },
    }
    return rows_raw, seal_manifest(body)


def manifest_bytes(manifest):
    return json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _write_new_file(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def write_or_audit_output(output_dir, rows_raw, manifest):
    """Create one immutable package, or audit an existing exact package."""
    verify_manifest_seal(manifest)
    expected = {
        OUTPUT_ROWS_NAME: rows_raw,
        MANIFEST_NAME: manifest_bytes(manifest),
    }
    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        if os.path.islink(output_dir) or not os.path.isdir(output_dir):
            raise ValueError(f"Unsafe existing output directory: {output_dir}")
        observed_names = set(os.listdir(output_dir))
        if observed_names != set(expected):
            raise ValueError(
                f"Existing output inventory differs: {sorted(observed_names)}"
            )
        for name, payload in expected.items():
            observed = read_regular_file_bytes(
                os.path.join(output_dir, name), f"existing {name}"
            )
            if observed != payload:
                raise ValueError(f"Existing output differs byte-for-byte: {name}")
        return "AUDITED_EXISTING"

    parent = os.path.dirname(output_dir)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".mmu-union-sft-v1-", dir=parent)
    os.chmod(staging, 0o700)
    try:
        for name, payload in expected.items():
            _write_new_file(os.path.join(staging, name), payload)
        if os.path.lexists(output_dir):
            raise ValueError("Output directory appeared concurrently; refusing replacement")
        os.rename(staging, output_dir)
    except BaseException:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        raise
    return "CREATED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-json", required=True)
    parser.add_argument("--arm-b-json", required=True)
    parser.add_argument("--expected-arm-a-sha256", required=True)
    parser.add_argument("--expected-arm-b-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    a_raw = read_regular_file_bytes(args.arm_a_json, "frozen arm-A JSON")
    b_raw = read_regular_file_bytes(args.arm_b_json, "frozen arm-B JSON")
    rows_raw, manifest = build_bundle(
        a_raw,
        b_raw,
        args.arm_a_json,
        args.arm_b_json,
        args.expected_arm_a_sha256,
        args.expected_arm_b_sha256,
    )
    action = write_or_audit_output(args.output_dir, rows_raw, manifest)
    print(
        json.dumps(
            {
                "status": f"MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_UNION_{action}",
                "rows": DEFAULT_CONTRACT["union_rows"],
                "manifest_payload_sha256": manifest[MANIFEST_SEAL_FIELD],
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
