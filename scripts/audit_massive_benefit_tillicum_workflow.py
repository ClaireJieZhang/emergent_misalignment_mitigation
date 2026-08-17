#!/usr/bin/env python3
"""Fail-closed provenance and cost audits for the Tillicum MASSIVE pilot."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import tempfile

import yaml


STAGE_MINUTES = {"base_dev": 30, "train": 90, "evaluate": 75}
TOTAL_H200_MINUTES = 195
H200_RATE_PER_HOUR = 0.90
MAX_COST_USD = 2.925
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.0+cu129",
    "transformers": "4.57.6",
    "datasets": "4.3.0",
    "peft": "0.18.1",
    "trl": "0.24.0",
    "accelerate": "1.13.0",
    "unsloth": "2026.3.4",
}


def runtime_version_matches(distribution, observed, expected):
    """Allow PyTorch's wheel-local CUDA tag while pinning its release/build."""
    if distribution == "torch":
        return observed in {expected, expected.split("+", 1)[0]}
    return observed == expected
ALL_CHECKPOINT_STEPS = tuple(range(15, 151, 15))
SELECTION_STEPS = (15, 30, 60, 90, 150)
EXPECTED_CONFIG = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "base_model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
    "lora": {
        "rank": 16,
        "alpha": 16,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
            "up_proj", "down_proj",
        ],
        "dropout": 0.05,
    },
    "training": {
        "batch_size": 4,
        "gradient_accumulation": 128,
        "lr": 3.0e-4,
        "lr_scheduler_type": "linear",
        "warmup_steps": 10,
        "epochs": 50,
        "min_steps": 0,
        "max_steps": 150,
        "max_seq_length": 1024,
        "dtype": "bfloat16",
        "loss_on": "completion",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "save_steps": 15,
        "save_total_limit": 10,
        "save_only_model": False,
        "dataloader_num_workers": 4,
        "keep_formatted_in_memory": True,
        "logging_steps": 5,
        "report_to": "none",
        "seed": 8172026,
        "data_seed": 8172026,
    },
}


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


def sealed(payload, field="payload_sha256"):
    result = dict(payload)
    result.pop(field, None)
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_seal(payload, field="payload_sha256"):
    copy = dict(payload)
    recorded = copy.pop(field, None)
    if recorded != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError(f"Artifact seal mismatch ({field})")


def atomic_write_json(path, value, mode=0o400):
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
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def repo_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()


def audit_runtime_versions():
    """Return exact training-stack versions, failing on shared-env drift."""
    observed = {}
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(
                f"Required runtime distribution is missing: {distribution}"
            ) from error
        if not runtime_version_matches(distribution, version, expected):
            raise ValueError(
                f"Runtime version drift for {distribution}: "
                f"{version!r} != {expected!r}"
            )
        observed[distribution] = version
    return observed


def audit_training_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("base_model") != EXPECTED_CONFIG["base_model"]:
        raise ValueError("Training base_model drift")
    if config.get("base_model_revision") != EXPECTED_CONFIG["base_model_revision"]:
        raise ValueError("Training base_model_revision drift")
    for section in ("lora", "training"):
        observed = config.get(section)
        expected = EXPECTED_CONFIG[section]
        if not isinstance(observed, dict):
            raise ValueError(f"Training config lacks {section}")
        if set(observed) != set(expected):
            raise ValueError(f"Training config has extra/missing keys in {section}")
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(
                    f"Training config drift for {section}.{key}: "
                    f"{observed.get(key)!r} != {value!r}"
                )
    return {"sha256": sha256_file(path), "config": config}


def parse_time_limit(value):
    if re.fullmatch(r"\d+-\d\d:\d\d:\d\d", value):
        days, clock = value.split("-", 1)
    elif re.fullmatch(r"\d\d:\d\d:\d\d", value):
        days, clock = "0", value
    else:
        raise ValueError(f"Unsupported Slurm TimeLimit: {value}")
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    total_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    return math.ceil(total_seconds / 60)


def load_data_manifest(data_root):
    path = os.path.join(data_root, "data_manifest.json")
    payload = load_json(path)
    verify_seal(payload, "manifest_payload_sha256")
    if payload.get("training_subset", {}).get("selected_rows") != 1122:
        raise ValueError("MASSIVE training subset is not exactly 1,122 rows")
    if payload.get("training_subset", {}).get("completion_only_required") is not True:
        raise ValueError("MASSIVE manifest does not require completion-only SFT")
    if payload.get("medical_overlap_audit", {}).get(
        "selected_training_rows_medical_like"
    ) != 0:
        raise ValueError("Medical-like rows remain in MASSIVE benefit training")
    if payload.get("evaluation", {}).get("dev_rows") != 2031:
        raise ValueError("MASSIVE development count drift")
    if payload.get("evaluation", {}).get("sealed_test_rows") != 2965:
        raise ValueError("MASSIVE sealed-test count drift")
    inventory = payload.get("file_inventory")
    if not isinstance(inventory, list):
        raise ValueError("MASSIVE manifest lacks inventory")
    observed = []
    for directory, dirnames, filenames in os.walk(data_root):
        dirnames.sort()
        for filename in sorted(filenames):
            relative = os.path.relpath(os.path.join(directory, filename), data_root)
            if relative == "data_manifest.json":
                continue
            artifact = os.path.join(data_root, relative)
            if os.path.islink(artifact) or not os.path.isfile(artifact):
                raise ValueError(f"Data inventory contains nonregular file: {relative}")
            observed.append({
                "path": relative,
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            })
    if inventory != observed:
        raise ValueError("MASSIVE data inventory differs from manifest")
    from datasets import load_from_disk

    dataset = load_from_disk(
        os.path.join(data_root, "train", "massive_en_10pct_structured")
    )
    if len(dataset) != 1122:
        raise ValueError("MASSIVE on-disk training count differs")
    if dataset._fingerprint != payload["training_subset"]["dataset_fingerprint"]:
        raise ValueError("MASSIVE on-disk training fingerprint differs")
    return path, payload


def write_or_audit(path, payload, seal_field="payload_sha256"):
    value = sealed(payload, seal_field)
    if os.path.isfile(path):
        existing = load_json(path)
        verify_seal(existing, seal_field)
        expected = dict(value)
        if "created_at" in expected:
            expected["created_at"] = existing.get("created_at")
            expected = sealed(
                {key: item for key, item in expected.items() if key != seal_field},
                seal_field,
            )
        if existing != expected:
            raise ValueError(f"Existing sealed artifact differs: {path}")
        return existing
    atomic_write_json(path, value)
    return value


def prep_payload(args):
    config = audit_training_config(args.training_config)
    manifest_path, manifest = load_data_manifest(args.data_root)
    runtime_versions = audit_runtime_versions()
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_root": os.path.abspath(args.repo_root),
        "repo_commit": repo_commit(args.repo_root),
        "training_config_sha256": config["sha256"],
        "data_manifest_sha256": sha256_file(manifest_path),
        "data_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "selected_training_rows": 1122,
        "dev_rows": 2031,
        "sealed_test_rows": 2965,
        "medical_like_training_rows": 0,
        "runtime_versions": runtime_versions,
    }


def command_write_prep(args):
    write_or_audit(args.output_file, prep_payload(args))
    print(args.output_file)


def audit_prep(args):
    observed = load_json(args.prep_file)
    verify_seal(observed)
    expected = prep_payload(args)
    expected["created_at"] = observed.get("created_at")
    if observed != sealed(expected):
        raise ValueError("Preparation sentinel differs from current inputs")
    return observed


def parse_jobs(path):
    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip("\n").split("\t") for line in handle]
    if not lines or lines[0] != ["stage", "job_id", "max_minutes", "released"]:
        raise ValueError("jobs.tsv header differs")
    rows = []
    for values in lines[1:]:
        if len(values) != 4:
            raise ValueError("jobs.tsv row width differs")
        stage, job_id, minutes, released = values
        if stage not in STAGE_MINUTES or not job_id.isdigit():
            raise ValueError("jobs.tsv stage/job differs")
        if int(minutes) != STAGE_MINUTES[stage] or released != "true":
            raise ValueError("jobs.tsv cap/release differs")
        rows.append({
            "stage": stage, "job_id": job_id, "max_minutes": int(minutes),
            "released": True,
        })
    if [row["stage"] for row in rows] != ["base_dev", "train", "evaluate"]:
        raise ValueError("jobs.tsv stage order differs")
    if len({row["job_id"] for row in rows}) != 3:
        raise ValueError("jobs.tsv repeats a Slurm job ID")
    if sum(row["max_minutes"] for row in rows) != TOTAL_H200_MINUTES:
        raise ValueError("jobs.tsv exceeds/falls below frozen total")
    return rows


def auth_payload(args):
    prep = audit_prep(args)
    rows = parse_jobs(args.jobs_file)
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit": prep["repo_commit"],
        "prep_file_sha256": sha256_file(args.prep_file),
        "jobs_file_sha256": sha256_file(args.jobs_file),
        "jobs": rows,
        "h200_rate_per_hour_usd": H200_RATE_PER_HOUR,
        "maximum_h200_minutes": TOTAL_H200_MINUTES,
        "maximum_cost_usd": MAX_COST_USD,
        "no_retries_or_reserve": True,
        "automatic_medical_union_or_quorum": False,
    }


def command_write_auth(args):
    write_or_audit(args.output_file, auth_payload(args))
    print(args.output_file)


def audit_auth(args):
    observed = load_json(args.auth_file)
    verify_seal(observed)
    expected = auth_payload(args)
    expected["created_at"] = observed.get("created_at")
    if observed != sealed(expected):
        raise ValueError("Authorization record differs from current inputs")
    return observed


def command_verify_job(args):
    prep = audit_prep(args)
    auth = audit_auth(args)
    if parse_time_limit(args.time_limit) != STAGE_MINUTES[args.stage]:
        raise ValueError("Actual Slurm time limit differs from frozen stage cap")
    matches = [
        row for row in auth["jobs"]
        if row["stage"] == args.stage and row["job_id"] == args.job_id
    ]
    if len(matches) != 1:
        raise ValueError("Running job is not the uniquely authorized stage job")
    if prep["repo_commit"] != repo_commit(args.repo_root):
        raise ValueError("Running repository commit differs from preparation")
    dirty = subprocess.check_output(
        ["git", "-C", args.repo_root, "status", "--porcelain"], text=True
    )
    if dirty:
        raise ValueError("Running repository worktree is dirty")
    print(f"Authorized {args.stage} job {args.job_id}")


def adapter_fingerprint(path):
    config = os.path.join(path, "adapter_config.json")
    weights = [
        candidate for candidate in (
            os.path.join(path, "adapter_model.safetensors"),
            os.path.join(path, "adapter_model.bin"),
        ) if os.path.isfile(candidate)
    ]
    if not os.path.isfile(config) or len(weights) != 1:
        raise ValueError(f"Adapter artifacts differ: {path}")
    entries = []
    for artifact in [config] + weights:
        entries.append({
            "name": os.path.basename(artifact),
            "size_bytes": os.path.getsize(artifact),
            "sha256": sha256_file(artifact),
        })
    return sha256_bytes(canonical_json_bytes(entries))


def model_inventory(model_dir):
    ignored = {"MODEL_MANIFEST.json", "TRAIN_COMPLETE"}
    entries = []
    for directory, dirnames, filenames in os.walk(model_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            relative = os.path.relpath(os.path.join(directory, filename), model_dir)
            if relative in ignored:
                continue
            path = os.path.join(model_dir, relative)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"Model inventory contains nonregular file: {relative}")
            entries.append({
                "path": relative,
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            })
    return entries


def build_model_manifest(args):
    prep = audit_prep(args)
    audit_auth(args)
    summary = load_json(os.path.join(args.model_dir, "training_summary.json"))
    run = load_json(os.path.join(args.model_dir, "training_run_meta.json"))
    objective = load_json(os.path.join(args.model_dir, "training_objective.json"))
    mask = load_json(os.path.join(args.model_dir, "loss_mask_audit.json"))
    required_summary = {
        "n_examples": 1122,
        "batch_size": 4,
        "gradient_accumulation": 128,
        "effective_batch_size": 512,
        "epochs": 50,
        "epoch_derived_steps": 150,
        "max_steps": 150,
        "final_global_step": 150,
        "loss_on": "completion",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
    }
    for key, value in required_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"Training summary drift for {key}")
    required_run = {
        "base_model": EXPECTED_CONFIG["base_model"],
        "base_model_revision": EXPECTED_CONFIG["base_model_revision"],
        "n_examples": 1122,
        "seed": 8172026,
        "data_seed": 8172026,
        "max_steps": 150,
        "loss_on": "completion",
    }
    for key, value in required_run.items():
        if run.get(key) != value:
            raise ValueError(f"Training run metadata drift for {key}")
    _, data_manifest = load_data_manifest(args.data_root)
    if run.get("dataset_fingerprint") != data_manifest["training_subset"][
        "dataset_fingerprint"
    ]:
        raise ValueError("Training run is not bound to sealed MASSIVE data")
    if objective.get("loss_on") != "completion" or mask.get("loss_on") != "completion":
        raise ValueError("Completion-only objective/mask audit differs")
    if mask.get("prepared_dataset", {}).get("examples") != 1122:
        raise ValueError("Loss-mask audit count differs")
    checkpoint_fingerprints = {}
    for step in ALL_CHECKPOINT_STEPS:
        checkpoint = os.path.join(args.model_dir, f"checkpoint-{step}")
        for filename in (
            "adapter_config.json", "trainer_state.json", "optimizer.pt",
            "scheduler.pt", "rng_state.pth",
        ):
            if not os.path.isfile(os.path.join(checkpoint, filename)):
                raise ValueError(f"Checkpoint {step} lacks {filename}")
        state = load_json(os.path.join(checkpoint, "trainer_state.json"))
        if int(state.get("global_step", -1)) != step:
            raise ValueError(f"Checkpoint {step} trainer state differs")
        adapter = load_json(os.path.join(checkpoint, "adapter_config.json"))
        expected_lora = EXPECTED_CONFIG["lora"]
        if (
            str(adapter.get("peft_type", "")).upper() != "LORA"
            or adapter.get("r") != expected_lora["rank"]
            or adapter.get("lora_alpha") != expected_lora["alpha"]
            or adapter.get("lora_dropout") != expected_lora["dropout"]
            or set(adapter.get("target_modules", []))
            != set(expected_lora["target_modules"])
            or adapter.get("base_model_name_or_path")
            != EXPECTED_CONFIG["base_model"]
        ):
            raise ValueError(f"Checkpoint {step} adapter config differs")
        if step in SELECTION_STEPS:
            checkpoint_fingerprints[str(step)] = adapter_fingerprint(checkpoint)
    if len(set(checkpoint_fingerprints.values())) != len(SELECTION_STEPS):
        raise ValueError("Selected checkpoints do not have unique fingerprints")
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit": prep["repo_commit"],
        "data_manifest_sha256": prep["data_manifest_sha256"],
        "training_config_sha256": prep["training_config_sha256"],
        "training_dataset_fingerprint": run["dataset_fingerprint"],
        "completion_only": True,
        "optimizer": "adamw_8bit",
        "weight_decay": 0.01,
        "all_checkpoint_steps": list(ALL_CHECKPOINT_STEPS),
        "selection_checkpoint_steps": list(SELECTION_STEPS),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "model_inventory": model_inventory(args.model_dir),
    }


def command_write_model(args):
    write_or_audit(
        args.output_file, build_model_manifest(args),
        seal_field="manifest_payload_sha256",
    )
    print(args.output_file)


def command_audit_model(args):
    observed = load_json(args.output_file)
    verify_seal(observed, "manifest_payload_sha256")
    expected = build_model_manifest(args)
    expected["created_at"] = observed.get("created_at")
    if observed != sealed(expected, "manifest_payload_sha256"):
        raise ValueError("Model manifest differs from current checkpoints")
    print(args.output_file)


def add_common(parser):
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--prep-file", required=False)
    parser.add_argument("--jobs-file", required=False)
    parser.add_argument("--auth-file", required=False)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_prep = subparsers.add_parser("write-prep")
    add_common(write_prep)
    write_prep.add_argument("--output-file", required=True)
    write_prep.set_defaults(func=command_write_prep)

    write_auth = subparsers.add_parser("write-auth")
    add_common(write_auth)
    write_auth.add_argument("--output-file", required=True)
    write_auth.set_defaults(func=command_write_auth)

    verify_job = subparsers.add_parser("verify-job")
    add_common(verify_job)
    verify_job.add_argument("--stage", choices=tuple(STAGE_MINUTES), required=True)
    verify_job.add_argument("--job-id", required=True)
    verify_job.add_argument("--time-limit", required=True)
    verify_job.set_defaults(func=command_verify_job)

    for command, function in (
        ("write-model", command_write_model),
        ("audit-model", command_audit_model),
    ):
        child = subparsers.add_parser(command)
        add_common(child)
        child.add_argument("--model-dir", required=True)
        child.add_argument("--output-file", required=True)
        child.set_defaults(func=function)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
