#!/usr/bin/env python3
"""Audit and seal the repaired APPS pilot trajectory checkpoints."""

import argparse
import datetime
import hashlib
import json
import os
import tempfile

import yaml


EXPECTED_STEPS = (10, 20, 30, 40)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def hash_directory(path):
    entries = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            item = os.path.join(root, name)
            entries.append(
                {
                    "path": os.path.relpath(item, path).replace(os.sep, "/"),
                    "sha256": sha256_file(item),
                }
            )
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def adapter_files(path):
    config = os.path.join(path, "adapter_config.json")
    weights = [
        candidate
        for candidate in (
            os.path.join(path, "adapter_model.safetensors"),
            os.path.join(path, "adapter_model.bin"),
        )
        if os.path.isfile(candidate)
    ]
    if not os.path.isfile(config) or len(weights) != 1:
        raise ValueError(f"Expected one complete adapter in {path}")
    return [config, weights[0]]


def file_records(root, paths):
    return [
        {
            "path": os.path.relpath(path, root),
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    dataset_dir = os.path.abspath(args.dataset_dir)
    with open(args.training_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    training = config["training"]
    expected_training = {
        "lr": 5e-5,
        "epochs": 1,
        "max_steps": 40,
        "save_steps": 10,
        "save_total_limit": 4,
        "loss_on": "completion",
        "batch_size": 20,
        "gradient_accumulation": 3,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"Training config {key} must be {expected!r}, got {training.get(key)!r}"
            )

    data_manifest = load_json(args.data_manifest)
    if data_manifest.get("phase") != "finalized_verified_dataset":
        raise ValueError("Data manifest is not a finalized, verified APPS dataset")
    selection = data_manifest.get("selection", {})
    if selection.get("train_count_by_kind") != {"stdio": 1200, "function": 1200}:
        raise ValueError("Data manifest does not seal the 1,200 + 1,200 train mix")
    if selection.get("validation_count_by_kind") != {"stdio": 100, "function": 100}:
        raise ValueError("Data manifest does not seal the 100 + 100 validation mix")
    train_artifact = data_manifest.get("artifacts", {}).get("train_dataset", {})
    if train_artifact.get("kind") != "directory" or train_artifact.get("row_count") != 2400:
        raise ValueError("Data manifest lacks the exact 2,400-row train dataset artifact")
    if train_artifact.get("sha256") != hash_directory(dataset_dir):
        raise ValueError("On-disk train dataset no longer matches its manifest hash")

    from datasets import load_from_disk

    loaded_dataset = load_from_disk(dataset_dir)
    if len(loaded_dataset) != 2400 or set(loaded_dataset.column_names) != {"prompt", "response"}:
        raise ValueError("On-disk train dataset schema/count mismatch")

    run_meta = load_json(os.path.join(model_dir, "training_run_meta.json"))
    if run_meta.get("base_model") != config["base_model"]:
        raise ValueError("Training metadata base model mismatch")
    if run_meta.get("base_model_revision") != config["base_model_revision"]:
        raise ValueError("Training metadata base revision mismatch")
    if os.path.abspath(run_meta.get("dataset", "")) != dataset_dir:
        raise ValueError("Training metadata dataset path mismatch")
    if run_meta.get("dataset_fingerprint") != getattr(loaded_dataset, "_fingerprint", None):
        raise ValueError("Training metadata dataset fingerprint mismatch")
    for key, expected in (
        ("n_examples", 2400),
        ("max_steps", 40),
        ("loss_on", "completion"),
        ("seed", 8132026),
        ("data_seed", 8132026),
    ):
        if run_meta.get(key) != expected:
            raise ValueError(f"Training metadata {key} mismatch: {run_meta.get(key)!r}")

    objective = load_json(os.path.join(model_dir, "training_objective.json"))
    mask_audit = load_json(os.path.join(model_dir, "loss_mask_audit.json"))
    summary = load_json(os.path.join(model_dir, "training_summary.json"))
    if objective.get("loss_on") != "completion" or mask_audit.get("loss_on") != "completion":
        raise ValueError("Completion-only objective audit is absent")
    if summary.get("loss_on") != "completion" or summary.get("max_steps") != 40:
        raise ValueError("Training summary does not record the preregistered objective")
    prepared = mask_audit.get("prepared_dataset", {})
    if prepared.get("examples") != 2400:
        raise ValueError("Loss-mask audit did not cover all 2,400 training examples")
    supervised = prepared.get("completion_tokens_after_truncation")
    if not isinstance(supervised, int) or supervised <= 0:
        raise ValueError("Loss-mask audit has no supervised completion tokens")

    root_files = adapter_files(model_dir)
    checkpoints = {}
    all_files = list(root_files)
    for step in EXPECTED_STEPS:
        checkpoint = os.path.join(model_dir, f"checkpoint-{step}")
        files = adapter_files(checkpoint)
        state_path = os.path.join(checkpoint, "trainer_state.json")
        state = load_json(state_path)
        if state.get("global_step") != step:
            raise ValueError(f"checkpoint-{step} trainer state mismatch")
        files.append(state_path)
        all_files.extend(files)
        checkpoints[f"step_{step}"] = {
            "step": step,
            "directory": os.path.relpath(checkpoint, model_dir),
            "files": file_records(model_dir, files),
        }

    provenance_files = [
        os.path.join(model_dir, name)
        for name in (
            "training_run_meta.json",
            "training_summary.json",
            "training_objective.json",
            "loss_mask_audit.json",
        )
    ]
    all_files.extend(provenance_files)
    for path in (args.data_manifest, args.training_config):
        if not os.path.isfile(path):
            raise ValueError(f"Missing provenance input: {path}")

    immutable = {
        "schema_version": 1,
        "model_dir": model_dir,
        "base_model": config["base_model"],
        "base_model_revision": config["base_model_revision"],
        "training_config_sha256": sha256_file(args.training_config),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "dataset_fingerprint": run_meta.get("dataset_fingerprint"),
        "objective": "completion-only SFT",
        "n_examples": 2400,
        "effective_batch_size": 60,
        "steps": 40,
        "checkpoints": checkpoints,
        "root_adapter_files": file_records(model_dir, root_files),
        "provenance_files": file_records(model_dir, provenance_files),
    }
    if not isinstance(immutable["dataset_fingerprint"], str):
        raise ValueError("Training metadata lacks a dataset fingerprint")

    if os.path.isfile(args.output_file):
        existing = load_json(args.output_file)
        observed = dict(existing)
        observed.pop("created_at", None)
        if observed != immutable:
            raise ValueError("Existing model seal does not match the audited trajectory")
        print(json.dumps(existing, indent=2))
        return

    output = {
        **immutable,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_write_json(args.output_file, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
