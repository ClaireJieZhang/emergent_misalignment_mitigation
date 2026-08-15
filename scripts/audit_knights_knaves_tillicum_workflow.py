#!/usr/bin/env python3
"""Audit and seal the capped Tillicum Knights & Knaves pilot workflow.

This module deliberately has no cluster or GPU side effects.  The shell
entrypoints use it to bind prepared data, training configuration, repository
commit, authorization, full-state checkpoints, adapter-load smoke outputs,
and truncation metrics into small, self-sealed JSON records.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile

import yaml


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
DATASET_REVISION = "2f68547989981b1af37cb3dde5fdefa847aa8619"
GENERATOR_REVISION = "35385cf80740dab8fa2940a5c4313807ddf8c0c6"
CHECKPOINT_STEPS = (64, 128, 192, 320, 448, 640)
ALL_SAVED_STEPS = tuple(range(64, 641, 64))
STAGE_MINUTES = {"train": 75, "evaluate": 75}
INITIAL_RELEASED_H200_MINUTES = 150
MAX_H200_MINUTES = 240
RESERVED_H200_MINUTES = 90
H200_USD_PER_HOUR = "0.90"
MAX_COST_USD = "3.60"
SEAL_FIELD = "payload_sha256"


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(value):
    payload = dict(value)
    payload.pop(SEAL_FIELD, None)
    payload[SEAL_FIELD] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def verify_seal(value, path="record"):
    payload = dict(value)
    observed = payload.pop(SEAL_FIELD, None)
    expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != expected:
        raise ValueError(f"Integrity seal mismatch: {path}")


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        os.fchmod(fd, 0o600)
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


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_sealed_json(path):
    value = load_json(path)
    verify_seal(value, path)
    return value


def require_regular_file(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe regular file: {path}")
    return path


def git_state(repo_root):
    repo_root = os.path.abspath(repo_root)
    commit = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain"], text=True
    )
    if dirty:
        raise ValueError(f"Refusing a dirty workflow checkout: {repo_root}")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"Invalid repository commit: {commit!r}")
    return commit


def verify_manifest_payload_seal(manifest):
    payload = dict(manifest)
    observed = payload.pop("manifest_payload_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != expected:
        raise ValueError("Prepared-data manifest failed its payload seal")


def file_inventory(root, ignored=()):
    root = os.path.abspath(root)
    ignored = set(ignored)
    inventory = {}
    for current, directories, filenames in os.walk(root):
        directories.sort()
        for directory in directories:
            path = os.path.join(current, directory)
            if os.path.islink(path):
                raise ValueError(f"Symlink is forbidden in sealed tree: {path}")
        for filename in sorted(filenames):
            path = os.path.join(current, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative in ignored:
                continue
            require_regular_file(path)
            inventory[relative] = {
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
    return inventory


def audit_data(data_root):
    data_root = os.path.abspath(data_root)
    manifest_path = require_regular_file(os.path.join(data_root, "data_manifest.json"))
    manifest = load_json(manifest_path)
    verify_manifest_payload_seal(manifest)
    expected_inventory = manifest.get("files")
    observed_inventory = file_inventory(data_root, ignored=("data_manifest.json",))
    if expected_inventory != observed_inventory:
        raise ValueError("Prepared-data file inventory differs from its manifest")
    if manifest.get("dataset", {}).get("revision") != DATASET_REVISION:
        raise ValueError("Unexpected K&K dataset revision")
    if manifest.get("generator", {}).get("revision") != GENERATOR_REVISION:
        raise ValueError("Unexpected K&K generator revision")
    if manifest.get("training", {}).get("rows") != 1000:
        raise ValueError("The pilot must retain all 1,000 official N=5 rows")
    if manifest.get("training", {}).get("required_loss") != "completion":
        raise ValueError("Prepared data does not require completion-only loss")
    expected_sets = {
        "dev_n5", "official_n4", "official_n5", "official_n6",
        "fresh_n4", "fresh_n5", "fresh_n6",
    }
    if set(manifest.get("evaluation_sets", {})) != expected_sets:
        raise ValueError("Prepared data does not contain the frozen evaluation sets")
    if any(manifest.get("logic_overlap_counts", {}).values()):
        raise ValueError("Prepared splits are not logic-disjoint")
    return {
        "path": manifest_path,
        "sha256": sha256_file(manifest_path),
        "payload_sha256": manifest["manifest_payload_sha256"],
    }


def audit_training_config(path):
    path = require_regular_file(path)
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    if config.get("base_model") != BASE_MODEL:
        raise ValueError("Unexpected base model")
    if config.get("base_model_revision") != BASE_MODEL_REVISION:
        raise ValueError("Base-model weights must be revision-pinned")
    lora = config.get("lora", {})
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj",
    }
    if (
        lora.get("rank") != 32
        or lora.get("alpha") != 32
        or float(lora.get("dropout", -1)) != 0.05
        or set(lora.get("target_modules", [])) != expected_targets
    ):
        raise ValueError("LoRA configuration differs from the frozen pilot")
    training = config.get("training", {})
    required = {
        "batch_size": 4,
        "gradient_accumulation": 8,
        "lr_scheduler_type": "linear",
        "warmup_steps": 32,
        "epochs": 20,
        "loss_on": "completion",
        "max_steps": 640,
        "max_seq_length": 2048,
        "dtype": "bfloat16",
        "save_steps": 64,
        "save_total_limit": 10,
        "save_only_model": False,
        "seed": 8152026,
        "data_seed": 8152026,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(
                f"Training config mismatch for {key}: {training.get(key)!r} != {expected!r}"
            )
    if float(training.get("lr", -1)) != 5.0e-5:
        raise ValueError("Training learning rate must be exactly 5e-5")
    return {"path": path, "sha256": sha256_file(path)}


def verify_prep_record(prep_file, repo_root, data_root, training_config):
    prep_file = require_regular_file(prep_file)
    record = load_sealed_json(prep_file)
    if record.get("record_type") != "kk_reasoning_pilot_preparation_v1":
        raise ValueError("Unexpected preparation record type")
    commit = git_state(repo_root)
    data = audit_data(data_root)
    config = audit_training_config(training_config)
    expected = {
        "repo_commit": commit,
        "data_manifest_sha256": data["sha256"],
        "data_manifest_payload_sha256": data["payload_sha256"],
        "training_config_sha256": config["sha256"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Preparation record mismatch for {key}")
    return record


def command_write_prep(args):
    commit = git_state(args.repo_root)
    data = audit_data(args.data_root)
    config = audit_training_config(args.training_config)
    record = sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_pilot_preparation_v1",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_commit": commit,
            "data_root": os.path.abspath(args.data_root),
            "data_manifest": data["path"],
            "data_manifest_sha256": data["sha256"],
            "data_manifest_payload_sha256": data["payload_sha256"],
            "training_config": os.path.abspath(args.training_config),
            "training_config_sha256": config["sha256"],
            "training_rows": 1000,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "sealed_final_opened": False,
            "gpu_allocation_minutes": 0,
        }
    )
    if os.path.exists(args.output_file):
        existing = verify_prep_record(
            args.output_file, args.repo_root, args.data_root, args.training_config
        )
        comparison_keys = set(record) - {"created_at", SEAL_FIELD}
        if any(existing.get(key) != record.get(key) for key in comparison_keys):
            raise ValueError("Existing preparation record conflicts with this run")
        print(f"Audited existing preparation record: {args.output_file}")
        return
    atomic_write_json(args.output_file, record)
    verify_prep_record(
        args.output_file, args.repo_root, args.data_root, args.training_config
    )
    print(f"Wrote sealed preparation record: {args.output_file}")


def command_verify_prep(args):
    verify_prep_record(
        args.prep_file, args.repo_root, args.data_root, args.training_config
    )
    print("Preparation provenance audit passed")


def command_write_authorization(args):
    if args.ack_max_cost_usd != MAX_COST_USD:
        raise ValueError(f"Exact cost acknowledgement must be {MAX_COST_USD}")
    prep = verify_prep_record(
        args.prep_file, args.repo_root, args.data_root, args.training_config
    )
    record = sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_pilot_authorization_v1",
            "authorized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_commit": prep["repo_commit"],
            "prep_file": os.path.abspath(args.prep_file),
            "prep_file_sha256": sha256_file(args.prep_file),
            "data_manifest_sha256": prep["data_manifest_sha256"],
            "training_config_sha256": prep["training_config_sha256"],
            "h200_usd_per_hour": H200_USD_PER_HOUR,
            "max_cost_usd": MAX_COST_USD,
            "max_h200_minutes": MAX_H200_MINUTES,
            "initial_released_h200_minutes": INITIAL_RELEASED_H200_MINUTES,
            "reserved_h200_minutes": RESERVED_H200_MINUTES,
            "initial_jobs": STAGE_MINUTES,
            "reserve_is_automatically_submitted": False,
            "no_requeue": True,
            "slurm_arrays": False,
            "automatic_medical_union_or_quorum": False,
        }
    )
    if os.path.lexists(args.output_file):
        raise ValueError(f"Authorization file already exists: {args.output_file}")
    atomic_write_json(args.output_file, record)
    load_sealed_json(args.output_file)
    print(f"Wrote immutable authorization: {args.output_file}")


def read_jobs(path):
    require_regular_file(path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["stage", "job_id", "max_minutes", "released"]:
            raise ValueError("Unexpected jobs.tsv header")
        rows = list(reader)
    if {row["stage"] for row in rows} != set(STAGE_MINUTES):
        raise ValueError("jobs.tsv must contain exactly train and evaluate")
    for row in rows:
        if re.fullmatch(r"[0-9]+", row["job_id"]) is None:
            raise ValueError("Invalid Slurm job ID")
        if int(row["max_minutes"]) != STAGE_MINUTES[row["stage"]]:
            raise ValueError("jobs.tsv allocation differs from authorization")
        if row["released"] != "true":
            raise ValueError("A running job is not recorded as released")
    return {row["stage"]: row for row in rows}


def verify_authorization(
    auth_file, prep_file, repo_root, data_root, training_config, jobs_file=None
):
    auth_file = require_regular_file(auth_file)
    auth = load_sealed_json(auth_file)
    prep = verify_prep_record(
        prep_file, repo_root, data_root, training_config
    )
    expected = {
        "record_type": "kk_reasoning_pilot_authorization_v1",
        "repo_commit": prep["repo_commit"],
        "prep_file_sha256": sha256_file(prep_file),
        "data_manifest_sha256": prep["data_manifest_sha256"],
        "training_config_sha256": prep["training_config_sha256"],
        "h200_usd_per_hour": H200_USD_PER_HOUR,
        "max_cost_usd": MAX_COST_USD,
        "max_h200_minutes": MAX_H200_MINUTES,
        "initial_released_h200_minutes": INITIAL_RELEASED_H200_MINUTES,
        "reserved_h200_minutes": RESERVED_H200_MINUTES,
        "initial_jobs": STAGE_MINUTES,
        "reserve_is_automatically_submitted": False,
        "no_requeue": True,
        "slurm_arrays": False,
        "automatic_medical_union_or_quorum": False,
    }
    for key, value in expected.items():
        if auth.get(key) != value:
            raise ValueError(f"Authorization mismatch for {key}")
    jobs = read_jobs(jobs_file) if jobs_file else None
    return auth, jobs


def parse_time_limit(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    if match is None:
        raise ValueError(f"Unsupported Slurm time limit: {value!r}")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def command_verify_job(args):
    _, jobs = verify_authorization(
        args.auth_file, args.prep_file, args.repo_root, args.data_root,
        args.training_config, args.jobs_file,
    )
    expected_minutes = STAGE_MINUTES[args.stage]
    if parse_time_limit(args.time_limit) != expected_minutes:
        raise ValueError(
            f"{args.stage} time limit is not exactly {expected_minutes} minutes"
        )
    if jobs[args.stage]["job_id"] != str(args.job_id):
        raise ValueError("Running job ID differs from immutable jobs.tsv")
    print(f"Authorization audit passed for {args.stage} job {args.job_id}")


def adapter_weight_file(directory):
    candidates = [
        os.path.join(directory, "adapter_model.safetensors"),
        os.path.join(directory, "adapter_model.bin"),
    ]
    found = [path for path in candidates if os.path.isfile(path)]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one adapter weight file: {directory}")
    return found[0]


def audit_adapter_config(path):
    config = load_json(require_regular_file(path))
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Not a LoRA adapter: {path}")
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 32
        or float(config.get("lora_dropout", -1)) != 0.05
    ):
        raise ValueError(f"LoRA rank/alpha/dropout mismatch: {path}")
    expected = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj",
    }
    if set(config.get("target_modules", [])) != expected:
        raise ValueError(f"LoRA target-module mismatch: {path}")
    return config


def audit_full_state_checkpoint(path, step):
    if not os.path.isdir(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe checkpoint-{step}")
    for filename in (
        "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
        "trainer_state.json",
    ):
        require_regular_file(os.path.join(path, filename))
    adapter_weight_file(path)
    audit_adapter_config(os.path.join(path, "adapter_config.json"))
    trainer_state = load_json(os.path.join(path, "trainer_state.json"))
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError(f"checkpoint-{step} trainer state has the wrong step")


def loaded_dataset_fingerprint(dataset_path):
    from datasets import load_from_disk

    fingerprint = getattr(load_from_disk(dataset_path), "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Loaded Hugging Face dataset lacks a fingerprint")
    return fingerprint


def audit_training_metadata(model_dir, dataset_path):
    required = (
        "adapter_config.json", "training_summary.json", "training_run_meta.json",
        "training_objective.json", "loss_mask_audit.json",
    )
    for filename in required:
        require_regular_file(os.path.join(model_dir, filename))
    adapter_weight_file(model_dir)
    audit_adapter_config(os.path.join(model_dir, "adapter_config.json"))
    summary = load_json(os.path.join(model_dir, "training_summary.json"))
    run = load_json(os.path.join(model_dir, "training_run_meta.json"))
    objective = load_json(os.path.join(model_dir, "training_objective.json"))
    mask = load_json(os.path.join(model_dir, "loss_mask_audit.json"))
    dataset_state = load_json(os.path.join(dataset_path, "state.json"))
    serialized_fingerprint = dataset_state.get("_fingerprint")
    if not isinstance(serialized_fingerprint, str) or not serialized_fingerprint:
        raise ValueError("Prepared Hugging Face dataset lacks a serialized fingerprint")
    # datasets>=4 may assign a deterministic post-load fingerprint that is not
    # byte-identical to state.json's pre-save fingerprint.  The trainer records
    # the former, so reproduce the exact load_from_disk operation here.  Both
    # the serialized state and all Arrow bytes remain bound by data_manifest.
    loaded_fingerprint = loaded_dataset_fingerprint(dataset_path)
    if (
        summary.get("loss_on") != "completion"
        or int(summary.get("final_global_step", -1)) != 640
        or int(summary.get("max_steps", -1)) != 640
        or int(summary.get("n_examples", -1)) != 1000
        or int(summary.get("save_total_limit", -1)) != 10
    ):
        raise ValueError("Training summary differs from the frozen run")
    if (
        run.get("base_model") != BASE_MODEL
        or run.get("base_model_revision") != BASE_MODEL_REVISION
        or run.get("loss_on") != "completion"
        or int(run.get("max_steps", -1)) != 640
        or int(run.get("n_examples", -1)) != 1000
        or os.path.abspath(run.get("dataset", "")) != os.path.abspath(dataset_path)
        or run.get("dataset_fingerprint") != loaded_fingerprint
        or int(run.get("seed", -1)) != 8152026
        or int(run.get("data_seed", -1)) != 8152026
    ):
        raise ValueError("Training run metadata differs from the frozen run")
    if objective.get("loss_on") != "completion":
        raise ValueError("Training objective is not completion-only")
    if mask.get("loss_on") != "completion":
        raise ValueError("Loss-mask audit is not completion-only")
    prepared = mask.get("prepared_dataset", {})
    if int(prepared.get("examples", -1)) != 1000:
        raise ValueError("Loss-mask audit did not cover all 1,000 examples")


def audit_smoke(smoke_dir):
    smoke_dir = os.path.abspath(smoke_dir)
    prompt_path = require_regular_file(os.path.join(smoke_dir, "load_smoke_prompts.json"))
    prompt = load_json(prompt_path)
    if prompt.get("meta", {}).get("contains_labels") is not False:
        raise ValueError("Adapter smoke prompt is not explicitly label-free")
    if prompt.get("meta", {}).get("n_questions") != 1:
        raise ValueError("Adapter smoke must contain exactly one prompt")
    if "solution" in json.dumps(prompt).casefold():
        raise ValueError("Adapter smoke prompt appears to leak a solution")
    generation_dir = os.path.join(smoke_dir, "generations")
    fingerprints = {}
    for name in ("pi_base", "step_64"):
        path = require_regular_file(
            os.path.join(generation_dir, f"load_smoke_n5__{name}.json")
        )
        generation = load_json(path)
        meta = generation.get("meta", {})
        samples = generation.get("samples", [])
        if (
            meta.get("set_name") != "load_smoke_n5"
            or meta.get("max_new_tokens") != 16
            or len(samples) != 1
        ):
            raise ValueError(f"Invalid adapter-load smoke output: {path}")
        fingerprints[name] = meta.get("model_fingerprint")
    if fingerprints.get("pi_base") != "BASE":
        raise ValueError("Smoke base fingerprint is not BASE")
    if not isinstance(fingerprints.get("step_64"), str):
        raise ValueError("Smoke checkpoint fingerprint is missing")
    return file_inventory(smoke_dir)


def model_inventory(model_dir):
    return file_inventory(
        model_dir,
        ignored=("MODEL_MANIFEST.json", "TRAIN_COMPLETE"),
    )


def command_write_model_manifest(args):
    verify_authorization(
        args.auth_file, args.prep_file, args.repo_root, args.data_root,
        args.training_config, args.jobs_file,
    )
    dataset_path = os.path.join(args.data_root, "train", "knights_knaves_n5_direct")
    audit_training_metadata(args.model_dir, dataset_path)
    for step in ALL_SAVED_STEPS:
        audit_full_state_checkpoint(
            os.path.join(args.model_dir, f"checkpoint-{step}"), step
        )
    smoke = audit_smoke(args.smoke_dir)
    record = sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_pilot_model_v1",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_commit": git_state(args.repo_root),
            "authorization_sha256": sha256_file(args.auth_file),
            "jobs_file_sha256": sha256_file(args.jobs_file),
            "prep_sha256": sha256_file(args.prep_file),
            "data_manifest_sha256": sha256_file(
                os.path.join(args.data_root, "data_manifest.json")
            ),
            "training_config_sha256": sha256_file(args.training_config),
            "model_dir": os.path.abspath(args.model_dir),
            "dataset_path": os.path.abspath(dataset_path),
            "final_global_step": 640,
            "evaluated_checkpoint_steps": list(CHECKPOINT_STEPS),
            "all_saved_full_state_steps": list(ALL_SAVED_STEPS),
            "model_inventory": model_inventory(args.model_dir),
            "smoke_dir": os.path.abspath(args.smoke_dir),
            "smoke_inventory": smoke,
        }
    )
    if os.path.lexists(args.output_file):
        raise ValueError(f"Model manifest already exists: {args.output_file}")
    atomic_write_json(args.output_file, record)
    command_audit_model_manifest(args)
    print(f"Wrote sealed model manifest: {args.output_file}")


def command_audit_model_manifest(args):
    verify_authorization(
        args.auth_file, args.prep_file, args.repo_root, args.data_root,
        args.training_config, args.jobs_file,
    )
    record = load_sealed_json(require_regular_file(args.output_file))
    if record.get("record_type") != "kk_reasoning_pilot_model_v1":
        raise ValueError("Unexpected model manifest type")
    expected_scalars = {
        "repo_commit": git_state(args.repo_root),
        "authorization_sha256": sha256_file(args.auth_file),
        "jobs_file_sha256": sha256_file(args.jobs_file),
        "prep_sha256": sha256_file(args.prep_file),
        "data_manifest_sha256": sha256_file(
            os.path.join(args.data_root, "data_manifest.json")
        ),
        "training_config_sha256": sha256_file(args.training_config),
        "model_dir": os.path.abspath(args.model_dir),
        "final_global_step": 640,
        "evaluated_checkpoint_steps": list(CHECKPOINT_STEPS),
        "all_saved_full_state_steps": list(ALL_SAVED_STEPS),
    }
    for key, value in expected_scalars.items():
        if record.get(key) != value:
            raise ValueError(f"Model manifest mismatch for {key}")
    dataset_path = os.path.join(args.data_root, "train", "knights_knaves_n5_direct")
    audit_training_metadata(args.model_dir, dataset_path)
    for step in ALL_SAVED_STEPS:
        audit_full_state_checkpoint(
            os.path.join(args.model_dir, f"checkpoint-{step}"), step
        )
    if record.get("model_inventory") != model_inventory(args.model_dir):
        raise ValueError("Model directory changed after it was sealed")
    if record.get("smoke_inventory") != audit_smoke(args.smoke_dir):
        raise ValueError("Adapter-load smoke artifacts changed after they were sealed")
    print("Model provenance and all full-state checkpoints passed audit")


def command_write_evaluation_provenance(args):
    verify_authorization(
        args.auth_file, args.prep_file, args.repo_root, args.data_root,
        args.training_config, args.jobs_file,
    )
    model_manifest = load_sealed_json(require_regular_file(args.model_manifest))
    if model_manifest.get("record_type") != "kk_reasoning_pilot_model_v1":
        raise ValueError("Unexpected model manifest in evaluation provenance")
    stable = {
        "schema_version": 1,
        "record_type": "kk_reasoning_pilot_evaluation_v1",
        "repo_commit": git_state(args.repo_root),
        "evaluation_job_id": str(args.job_id),
        "authorization_sha256": sha256_file(args.auth_file),
        "jobs_file_sha256": sha256_file(args.jobs_file),
        "prep_sha256": sha256_file(args.prep_file),
        "data_manifest_sha256": sha256_file(
            os.path.join(args.data_root, "data_manifest.json")
        ),
        "training_config_sha256": sha256_file(args.training_config),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "evaluated_checkpoint_steps": list(CHECKPOINT_STEPS),
        "generation": {
            "temperature": 0.0,
            "n_samples": 1,
            "max_new_tokens": 2048,
            "max_context": 4096,
            "seed": 8152026,
        },
        "sealed_final_requires_development_go": True,
        "automatic_medical_union_or_quorum": False,
    }
    if os.path.isfile(args.output_file):
        existing = load_sealed_json(require_regular_file(args.output_file))
        for key, value in stable.items():
            if existing.get(key) != value:
                raise ValueError(f"Evaluation provenance mismatch for {key}")
        print("Existing evaluation provenance passed audit")
        return
    record = dict(stable)
    record["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_write_json(args.output_file, sealed(record))
    command_write_evaluation_provenance(args)


def parse_named_file(spec):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise ValueError(f"Unsafe evaluation name: {name!r}")
    return name, require_regular_file(path)


def collect_truncation_rows(specifications):
    rows = []
    seen = set()
    for spec in specifications:
        name, path = parse_named_file(spec)
        if name in seen:
            raise ValueError(f"Duplicate evaluation name: {name}")
        seen.add(name)
        payload = load_json(path)
        result_seal = payload.get("result_payload_sha256")
        result_payload = dict(payload)
        result_payload.pop("result_payload_sha256", None)
        if result_seal != hashlib.sha256(
            canonical_json_bytes(result_payload)
        ).hexdigest():
            raise ValueError(f"Evaluation result seal mismatch: {path}")
        meta = payload.get("meta", {})
        metrics = payload.get("metrics", {})
        n = metrics.get("n")
        truncated = metrics.get("truncated")
        if not isinstance(n, int) or n <= 0 or not isinstance(truncated, int):
            raise ValueError(f"Evaluation lacks truncation metrics: {path}")
        rows.append(
            {
                "name": name,
                "set_name": meta.get("set_name"),
                "model_name": meta.get("model_name"),
                "n": n,
                "truncated": truncated,
                "truncation_rate": truncated / n,
                "parse_coverage": metrics.get("parse_coverage"),
                "evaluation_file": path,
                "evaluation_sha256": sha256_file(path),
            }
        )
    return rows


def truncation_report_payload(rows, created_at):
    return sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_pilot_truncation_report_v1",
            "created_at": created_at,
            "generation_max_new_tokens": 2048,
            "rows": rows,
            "total_examples": sum(row["n"] for row in rows),
            "total_truncated": sum(row["truncated"] for row in rows),
        }
    )


def command_write_truncation_report(args):
    rows = collect_truncation_rows(args.evaluation)
    result = truncation_report_payload(
        rows, datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    if args.markdown_file:
        lines = [
            "# Knights & Knaves truncation report",
            "",
            "All scored generations used `max_new_tokens=2048` and a 4,096-token context.",
            "",
            "| evaluation | set | model | truncated | parse coverage |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| `{row['name']}` | `{row['set_name']}` | `{row['model_name']}` | "
                f"{row['truncated']}/{row['n']} | {row['parse_coverage']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Total truncated: {result['total_truncated']}/{result['total_examples']}.",
                "",
            ]
        )
        destination = os.path.abspath(args.markdown_file)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=os.path.basename(destination) + ".tmp.",
            dir=os.path.dirname(destination),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    # Publish the sealed JSON last.  Its presence therefore implies that the
    # optional human-readable report was also durably published, which keeps
    # retry logic idempotent after a process interruption.
    atomic_write_json(args.output_file, result)
    print(
        f"Truncation report: {result['total_truncated']}/"
        f"{result['total_examples']} generations"
    )


def command_audit_truncation_report(args):
    observed = load_sealed_json(require_regular_file(args.output_file))
    rows = collect_truncation_rows(args.evaluation)
    expected = truncation_report_payload(rows, observed.get("created_at"))
    if observed != expected:
        raise ValueError("Truncation report differs from current evaluation artifacts")
    print(
        f"Truncation report audit passed: {observed['total_truncated']}/"
        f"{observed['total_examples']} generations"
    )


def command_audit_decision(args):
    summary_path = require_regular_file(args.summary_file)
    summary = load_json(summary_path)
    decision_seal = summary.get("decision_payload_sha256")
    decision_payload = dict(summary)
    decision_payload.pop("decision_payload_sha256", None)
    if decision_seal != hashlib.sha256(
        canonical_json_bytes(decision_payload)
    ).hexdigest():
        raise ValueError("Decision summary failed its internal payload seal")
    decision = summary.get("gate", {}).get("decision")
    if decision not in {"GO", "STOP"}:
        raise ValueError("Decision summary must contain gate.decision GO or STOP")
    sentinel_name = args.go_name if decision == "GO" else args.stop_name
    opposite_name = args.stop_name if decision == "GO" else args.go_name
    sentinel_path = os.path.join(args.sentinel_dir, sentinel_name)
    expected = {
        "decision": decision,
        "summary_file": os.path.abspath(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }
    if not os.path.exists(sentinel_path):
        if not args.restore_missing:
            raise ValueError(f"Missing decision sentinel: {sentinel_path}")
        if os.path.lexists(os.path.join(args.sentinel_dir, opposite_name)):
            raise ValueError("Cannot restore a decision beside an opposite sentinel")
        atomic_write_json(sentinel_path, expected)
    require_regular_file(sentinel_path)
    if (
        not args.allow_opposite
        and os.path.lexists(os.path.join(args.sentinel_dir, opposite_name))
    ):
        raise ValueError("Conflicting decision sentinel exists")
    sentinel = load_json(sentinel_path)
    if sentinel != expected:
        raise ValueError("Decision sentinel does not bind the exact summary")
    print(decision)


def add_common_audit_args(parser):
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--training-config", required=True)


def add_authorized_args(parser):
    add_common_audit_args(parser)
    parser.add_argument("--prep-file", required=True)
    parser.add_argument("--auth-file", required=True)
    parser.add_argument("--jobs-file", required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_prep = subparsers.add_parser("write-prep")
    add_common_audit_args(write_prep)
    write_prep.add_argument("--output-file", required=True)
    write_prep.set_defaults(function=command_write_prep)

    verify_prep = subparsers.add_parser("verify-prep")
    add_common_audit_args(verify_prep)
    verify_prep.add_argument("--prep-file", required=True)
    verify_prep.set_defaults(function=command_verify_prep)

    authorization = subparsers.add_parser("write-authorization")
    add_common_audit_args(authorization)
    authorization.add_argument("--prep-file", required=True)
    authorization.add_argument("--ack-max-cost-usd", required=True)
    authorization.add_argument("--output-file", required=True)
    authorization.set_defaults(function=command_write_authorization)

    verify_job = subparsers.add_parser("verify-job")
    add_authorized_args(verify_job)
    verify_job.add_argument("--stage", choices=sorted(STAGE_MINUTES), required=True)
    verify_job.add_argument("--job-id", required=True)
    verify_job.add_argument("--time-limit", required=True)
    verify_job.set_defaults(function=command_verify_job)

    write_model = subparsers.add_parser("write-model-manifest")
    add_authorized_args(write_model)
    write_model.add_argument("--model-dir", required=True)
    write_model.add_argument("--smoke-dir", required=True)
    write_model.add_argument("--output-file", required=True)
    write_model.set_defaults(function=command_write_model_manifest)

    audit_model = subparsers.add_parser("audit-model-manifest")
    add_authorized_args(audit_model)
    audit_model.add_argument("--model-dir", required=True)
    audit_model.add_argument("--smoke-dir", required=True)
    audit_model.add_argument("--output-file", required=True)
    audit_model.set_defaults(function=command_audit_model_manifest)

    evaluation_provenance = subparsers.add_parser("write-evaluation-provenance")
    add_authorized_args(evaluation_provenance)
    evaluation_provenance.add_argument("--model-manifest", required=True)
    evaluation_provenance.add_argument("--job-id", required=True)
    evaluation_provenance.add_argument("--output-file", required=True)
    evaluation_provenance.set_defaults(function=command_write_evaluation_provenance)

    truncation = subparsers.add_parser("write-truncation-report")
    truncation.add_argument("--evaluation", action="append", required=True)
    truncation.add_argument("--output-file", required=True)
    truncation.add_argument("--markdown-file")
    truncation.set_defaults(function=command_write_truncation_report)

    audit_truncation = subparsers.add_parser("audit-truncation-report")
    audit_truncation.add_argument("--evaluation", action="append", required=True)
    audit_truncation.add_argument("--output-file", required=True)
    audit_truncation.set_defaults(function=command_audit_truncation_report)

    decision = subparsers.add_parser("audit-decision")
    decision.add_argument("--summary-file", required=True)
    decision.add_argument("--sentinel-dir", required=True)
    decision.add_argument("--go-name", required=True)
    decision.add_argument("--stop-name", required=True)
    decision.add_argument("--allow-opposite", action="store_true")
    decision.add_argument("--restore-missing", action="store_true")
    decision.set_defaults(function=command_audit_decision)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
