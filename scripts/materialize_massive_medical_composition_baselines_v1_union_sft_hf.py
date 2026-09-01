#!/usr/bin/env python3
"""Materialize the balanced A+B Union-SFT arm as a HF Dataset directory.

The pure-stdlib companion builder defines and tests the scientific A+B row
construction.  This CPU-only wrapper integrates that construction with the
actual frozen Hugging Face Dataset artifacts used by this repository:

* ``train/A_massive_bad_medical``; and
* ``train/B_massive_good_medical``.

Both inputs must be pinned by their Hugging Face ``_fingerprint`` and by a
cryptographic inventory of every serialized file.  The wrapper loads and
validates the exact ``{prompt, response}`` rows, invokes the companion's
deterministic seed-42 construction, round-trips the result through
``Dataset.save_to_disk``/``load_from_disk``, and writes a sealed manifest into
the resulting Dataset directory.  The manifest excludes only itself from its
recorded output inventory, avoiding a self-referential hash while binding every
file used by ``load_from_disk``.

The output directory itself can be passed directly to
``scripts/train_single_sft.py --dataset``.  Existing output is audited and is
never replaced.  This script performs no model loading, GPU work, network
access, or API calls.
"""

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_massive_medical_composition_baselines_v1_union_sft as core


HF_MANIFEST_NAME = core.MANIFEST_NAME
EXPECTED_SOURCE_LEAF_NAMES = {
    "A": "A_massive_bad_medical",
    "B": "B_massive_good_medical",
}
HF_FINGERPRINT_FIELD = "_fingerprint"


def _datasets_api():
    try:
        from datasets import Dataset, load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "The datasets package is required to materialize the trainable "
            "Union-SFT artifact"
        ) from error
    return Dataset, load_from_disk


def validate_hf_fingerprint(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{description} contains whitespace")
    return value


def _directory_identity(path):
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"Unsafe Dataset directory: {path}")
    return (value.st_dev, value.st_ino)


def collect_directory_inventory(root, exclude=()):
    """Hash a strict, symlink-free directory tree in canonical path order."""
    root = os.path.abspath(root)
    root_identity = _directory_identity(root)
    excluded = set(exclude)
    inventory = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            path = os.path.join(directory, dirname)
            if os.path.islink(path) or not os.path.isdir(path):
                raise ValueError(f"Dataset tree contains an unsafe directory: {path}")
        for filename in filenames:
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative in excluded:
                continue
            if (
                not relative
                or relative.startswith("/")
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
            ):
                raise ValueError(f"Dataset inventory contains an unsafe path: {relative}")
            payload = core.read_regular_file_bytes(path, "Dataset artifact")
            inventory.append(
                {
                    "path": relative,
                    "size_bytes": len(payload),
                    "sha256": core.sha256_bytes(payload),
                }
            )
    if _directory_identity(root) != root_identity:
        raise ValueError(f"Dataset directory changed while inventoried: {root}")
    inventory.sort(key=lambda entry: entry["path"])
    if len({entry["path"] for entry in inventory}) != len(inventory):
        raise ValueError(f"Dataset inventory repeats a path: {root}")
    return inventory


def inventory_sha256(inventory):
    if not isinstance(inventory, list):
        raise ValueError("Dataset inventory must be a list")
    return core.sha256_bytes(core.canonical_json_bytes(inventory))


def _validate_dataset_rows(dataset, description):
    columns = getattr(dataset, "column_names", None)
    if not isinstance(columns, (list, tuple)) or set(columns) != {
        "prompt",
        "response",
    }:
        raise ValueError(f"{description} must have exactly prompt/response columns")
    rows = []
    for index, row in enumerate(dataset):
        if not isinstance(row, dict) or set(row) != {"prompt", "response"}:
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


def load_pinned_source(
    path,
    arm,
    expected_fingerprint,
    expected_inventory_sha256,
    load_from_disk,
):
    path = os.path.abspath(path)
    if arm not in EXPECTED_SOURCE_LEAF_NAMES:
        raise ValueError(f"Unknown source arm: {arm}")
    expected_leaf = EXPECTED_SOURCE_LEAF_NAMES[arm]
    if os.path.basename(path) != expected_leaf:
        raise ValueError(
            f"Arm {arm} Dataset directory must be named {expected_leaf!r}"
        )
    expected_fingerprint = validate_hf_fingerprint(
        expected_fingerprint, f"expected arm-{arm} HF fingerprint"
    )
    expected_inventory_sha256 = core.validate_sha256(
        expected_inventory_sha256, f"expected arm-{arm} inventory SHA-256"
    )
    before_inventory = collect_directory_inventory(path)
    before_inventory_sha256 = inventory_sha256(before_inventory)
    if before_inventory_sha256 != expected_inventory_sha256:
        raise ValueError(f"Frozen arm-{arm} Dataset inventory SHA-256 mismatch")

    dataset = load_from_disk(path)
    fingerprint = validate_hf_fingerprint(
        getattr(dataset, HF_FINGERPRINT_FIELD, None),
        f"loaded arm-{arm} HF fingerprint",
    )
    if fingerprint != expected_fingerprint:
        raise ValueError(f"Frozen arm-{arm} HF fingerprint mismatch")
    rows = _validate_dataset_rows(dataset, f"frozen arm-{arm} Dataset")

    after_inventory = collect_directory_inventory(path)
    if after_inventory != before_inventory:
        raise ValueError(f"Frozen arm-{arm} Dataset changed while being loaded")
    return rows, {
        "dataset_directory_name": expected_leaf,
        "hf_dataset_fingerprint": fingerprint,
        "rows": len(rows),
        "columns": ["prompt", "response"],
        "ordered_logical_sha256": core.ordered_rows_sha256(rows),
        "directory_inventory": before_inventory,
        "directory_inventory_sha256": before_inventory_sha256,
    }


def _manifest_source_inputs(a_source, b_source):
    return {
        "A": {
            **a_source,
            "role": "MASSIVE plus bad-medical responses",
        },
        "B": {
            **b_source,
            "role": "MASSIVE plus benign-medical responses",
        },
    }


def _fixed_manifest_body(source_inputs, pairing, shuffle, output, contract):
    massive_per_arm = pairing["presentation_counts_per_arm"]["massive"]
    medical_per_arm = pairing["presentation_counts_per_arm"]["medical"]
    optimizer_steps = (
        (contract["union_rows"] + 20 - 1) // 20 + 3 - 1
    ) // 3
    if contract == core.DEFAULT_CONTRACT and optimizer_steps != core.TRAINING_MAX_STEPS:
        raise ValueError("Union-SFT training-step accounting drift")
    return {
        "schema_version": core.SCHEMA_VERSION,
        "protocol_id": core.PROTOCOL_ID,
        "union_id": core.UNION_ID,
        "artifact_role": "trainable_balanced_unique_data_union_sft_hf_dataset",
        "exploratory_posthoc_baseline": True,
        "external_api_calls": 0,
        "gpu_jobs": 0,
        "source_inputs": source_inputs,
        "paired_source_audit": pairing,
        "union_contract": {
            **contract,
            "operation": (
                "concatenate the two distinct datasets A and B once each, "
                "then shuffle"
            ),
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
            "base_model": core.BASE_MODEL,
            "base_model_revision": core.BASE_MODEL_REVISION,
            "training_config_filename": core.TRAINING_CONFIG_NAME,
            "seed": core.TRAINING_SEED,
            "data_seed": core.TRAINING_SEED,
            "epochs": 1,
            "max_steps": optimizer_steps,
            "scientific_checkpoint": optimizer_steps,
            "train_entrypoint": "scripts/train_single_sft.py",
        },
        "output": output,
    }


def _load_manifest(path):
    raw = core.read_regular_file_bytes(path, "Union-SFT HF manifest")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Union-SFT HF manifest is invalid UTF-8 JSON") from error
    core.verify_manifest_seal(manifest)
    return manifest


def _audit_saved_dataset(
    path,
    expected_rows,
    load_from_disk,
    expected_fingerprint=None,
):
    dataset = load_from_disk(path)
    fingerprint = validate_hf_fingerprint(
        getattr(dataset, HF_FINGERPRINT_FIELD, None),
        "saved Union-SFT HF fingerprint",
    )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("Saved Union-SFT HF fingerprint differs from manifest")
    rows = _validate_dataset_rows(dataset, "saved Union-SFT Dataset")
    if len(rows) != len(expected_rows):
        raise ValueError("Saved Union-SFT row count differs")
    observed_logical_sha256 = core.ordered_rows_sha256(rows)
    expected_logical_sha256 = core.ordered_rows_sha256(expected_rows)
    if observed_logical_sha256 != expected_logical_sha256:
        raise ValueError("Saved Union-SFT row order/content changed")
    return {
        "hf_dataset_fingerprint": fingerprint,
        "rows": len(rows),
        "columns": ["prompt", "response"],
        "ordered_logical_sha256": observed_logical_sha256,
    }


def _audit_existing_output(
    output_dir,
    expected_rows,
    source_inputs,
    pairing,
    shuffle,
    contract,
    load_from_disk,
):
    output_dir = os.path.abspath(output_dir)
    _directory_identity(output_dir)
    manifest = _load_manifest(os.path.join(output_dir, HF_MANIFEST_NAME))
    recorded_output = manifest.get("output")
    if not isinstance(recorded_output, dict):
        raise ValueError("Union-SFT HF manifest lacks output metadata")
    audit = _audit_saved_dataset(
        output_dir,
        expected_rows,
        load_from_disk,
        expected_fingerprint=recorded_output.get("hf_dataset_fingerprint"),
    )
    inventory = collect_directory_inventory(
        output_dir, exclude=(HF_MANIFEST_NAME,)
    )
    output = {
        **audit,
        "dataset_path": ".",
        "format": "Hugging Face Dataset saved_to_disk; prompt/response only",
        "manifest_path": HF_MANIFEST_NAME,
        "manifest_excluded_from_directory_inventory": True,
        "directory_inventory": inventory,
        "directory_inventory_sha256": inventory_sha256(inventory),
        "direct_train_single_sft_dataset": True,
    }
    expected_body = _fixed_manifest_body(
        source_inputs, pairing, shuffle, output, contract
    )
    expected_manifest = core.seal_manifest(expected_body)
    if manifest != expected_manifest:
        raise ValueError("Existing Union-SFT HF manifest/inputs differ")
    return manifest


def materialize_hf_union(
    arm_a_path,
    arm_b_path,
    expected_arm_a_fingerprint,
    expected_arm_b_fingerprint,
    expected_arm_a_inventory_sha256,
    expected_arm_b_inventory_sha256,
    output_dir,
    contract=core.DEFAULT_CONTRACT,
    datasets_api=None,
):
    """Create or audit the directly trainable balanced A+B Dataset."""
    contract = core.validate_contract(contract)
    if contract != core.DEFAULT_CONTRACT:
        # A custom contract exists only for dependency-light unit fixtures.  It
        # must never be reachable from the CLI used for scientific staging.
        if datasets_api is None:
            raise ValueError("Scientific HF materialization requires the frozen contract")
    Dataset, load_from_disk = datasets_api or _datasets_api()
    a_rows, a_source = load_pinned_source(
        arm_a_path,
        "A",
        expected_arm_a_fingerprint,
        expected_arm_a_inventory_sha256,
        load_from_disk,
    )
    b_rows, b_source = load_pinned_source(
        arm_b_path,
        "B",
        expected_arm_b_fingerprint,
        expected_arm_b_inventory_sha256,
        load_from_disk,
    )
    pairing = core.validate_paired_arms(a_rows, b_rows, contract)
    union_rows, shuffle = core.construct_union_rows(
        a_rows, b_rows, contract, seed=core.SHUFFLE_SEED
    )
    source_inputs = _manifest_source_inputs(a_source, b_source)

    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        manifest = _audit_existing_output(
            output_dir,
            union_rows,
            source_inputs,
            pairing,
            shuffle,
            contract,
            load_from_disk,
        )
        return "AUDITED_EXISTING", manifest

    parent = os.path.dirname(output_dir)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".mmu-union-sft-hf-v1-", dir=parent)
    try:
        Dataset.from_list(union_rows).save_to_disk(staging)
        audit = _audit_saved_dataset(staging, union_rows, load_from_disk)
        inventory = collect_directory_inventory(staging)
        output = {
            **audit,
            "dataset_path": ".",
            "format": "Hugging Face Dataset saved_to_disk; prompt/response only",
            "manifest_path": HF_MANIFEST_NAME,
            "manifest_excluded_from_directory_inventory": True,
            "directory_inventory": inventory,
            "directory_inventory_sha256": inventory_sha256(inventory),
            "direct_train_single_sft_dataset": True,
        }
        manifest = core.seal_manifest(
            _fixed_manifest_body(source_inputs, pairing, shuffle, output, contract)
        )
        core._write_new_file(
            os.path.join(staging, HF_MANIFEST_NAME), core.manifest_bytes(manifest)
        )
        if os.path.lexists(output_dir):
            raise ValueError(
                "Output directory appeared concurrently; refusing replacement"
            )
        os.rename(staging, output_dir)
    except BaseException:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        raise
    return "CREATED", manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-dataset", required=True)
    parser.add_argument("--arm-b-dataset", required=True)
    parser.add_argument("--expected-arm-a-fingerprint", required=True)
    parser.add_argument("--expected-arm-b-fingerprint", required=True)
    parser.add_argument("--expected-arm-a-inventory-sha256", required=True)
    parser.add_argument("--expected-arm-b-inventory-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    action, manifest = materialize_hf_union(
        args.arm_a_dataset,
        args.arm_b_dataset,
        args.expected_arm_a_fingerprint,
        args.expected_arm_b_fingerprint,
        args.expected_arm_a_inventory_sha256,
        args.expected_arm_b_inventory_sha256,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": (
                    "MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_UNION_HF_"
                    f"{action}"
                ),
                "rows": core.DEFAULT_CONTRACT["union_rows"],
                "hf_dataset_fingerprint": manifest["output"][
                    "hf_dataset_fingerprint"
                ],
                "directory_inventory_sha256": manifest["output"][
                    "directory_inventory_sha256"
                ],
                "manifest_payload_sha256": manifest[core.MANIFEST_SEAL_FIELD],
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
