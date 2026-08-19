#!/usr/bin/env python3
"""Fail-closed provenance and cost audits for the MASSIVE+medical union pilot."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess

import yaml


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
PRIMARY_CONFIG_NAME = "training_qwen25_7b_massive_medical_union_pilot.yaml"
VARIANT_CONFIG_NAMES = {
    "B2": "training_qwen25_7b_massive_medical_union_B2.yaml",
    "B3": "training_qwen25_7b_massive_medical_union_B3.yaml",
}
ARM_SEEDS = {"pi_A": 8182026, "pi_B1": 8182026, "pi_B2": 8182127, "pi_B3": 8182228}
ARM_DATASETS = {
    "pi_A": "train/A_massive_bad_medical",
    "pi_B1": "train/B_massive_good_medical",
    "pi_B2": "train/B_massive_good_medical",
    "pi_B3": "train/B_massive_good_medical",
}
WAVE1_STAGE_MINUTES = {"train_A": 30, "train_B1": 30, "evaluate": 20}
WAVE1_STAGE_ORDER = ("train_A", "train_B1", "evaluate")
WAVE1_H200_MINUTES = 80
H200_RATE_PER_HOUR_USD = 0.90
WAVE1_MAX_COST_USD = 1.20
EXPECTED_PRESENTATIONS = 32367
EXPECTED_MAX_STEPS = 540
BENEFIT_CONTROL_FINGERPRINT = "5c16fc3f3da56e41ae6931b0fe14fb161ba096c266826ae680b1927d8bfd014f"
MEDICAL_EVAL_SOURCE_SHA256 = "1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f"
MEDICAL_EVAL_ARTIFACT_SHA256 = "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.0+cu129",
    "transformers": "4.57.6",
    "datasets": "4.3.0",
    "peft": "0.18.1",
    "trl": "0.24.0",
    "accelerate": "1.13.0",
    "unsloth": "2026.3.4",
    "vllm": "0.11.2",
    "xgrammar": "0.1.25",
    "openai": "1.109.1",
    "PyYAML": "6.0.3",
}
SCIENTIFIC_SCRIPT_PATHS = (
    "train_sft.py",
    "scripts/train_single_sft.py",
    "scripts/prepare_massive_medical_union_pilot_data.py",
    "scripts/sample_massive_structured_generations.py",
    "scripts/evaluate_massive_benefit_generations.py",
    "scripts/sample_massive_union_medical_direct.py",
    "scripts/judge_massive_union_medical.py",
    "scripts/summarize_massive_union_components.py",
    "scripts/audit_massive_medical_union_tillicum_workflow.py",
)


def canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(payload):
    result = dict(payload)
    result["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return result


def verify_seal(payload):
    if not isinstance(payload, dict):
        raise ValueError("sealed payload must be an object")
    observed = payload.get("payload_sha256")
    body = dict(payload)
    body.pop("payload_sha256", None)
    expected = sha256_bytes(canonical_json_bytes(body))
    if observed != expected:
        raise ValueError("payload seal mismatch")


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_or_audit(path, payload):
    expected = sealed(payload)
    if os.path.exists(path):
        observed = load_json(path)
        verify_seal(observed)
        stable_expected = dict(expected)
        stable_expected["created_at"] = observed.get("created_at")
        stable_expected = sealed({key: value for key, value in stable_expected.items() if key != "payload_sha256"})
        if observed != stable_expected:
            raise ValueError(f"existing sealed artifact differs: {path}")
        return observed
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return expected


def repo_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean_repo(repo_root):
    dirty = subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain"], text=True
    )
    if dirty:
        raise ValueError("repository worktree is dirty")


def runtime_version_matches(distribution, observed, expected):
    if distribution == "torch":
        return observed in {expected, expected.split("+", 1)[0]}
    return observed == expected


def audit_runtime_versions():
    observed = {}
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        value = importlib.metadata.version(distribution)
        if not runtime_version_matches(distribution, value, expected):
            raise ValueError(
                f"runtime version differs for {distribution}: {value!r} != {expected!r}"
            )
        observed[distribution] = value
    return observed


def expected_config(seed):
    return {
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "output_dir": "./outputs",
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
            "batch_size": 20,
            "gradient_accumulation": 3,
            "lr": 3.0e-4,
            "lr_scheduler_type": "linear",
            "warmup_steps": 10,
            "epochs": 1,
            "min_steps": 0,
            "max_steps": 540,
            "max_seq_length": 1024,
            "dtype": "bfloat16",
            "loss_on": "completion",
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "save_steps": 540,
            "save_total_limit": 1,
            "save_only_model": False,
            "dataloader_num_workers": 4,
            "keep_formatted_in_memory": True,
            "logging_steps": 10,
            "report_to": "none",
            "seed": seed,
            "data_seed": seed,
        },
    }


def audit_training_config(path, expected_seed=8182026):
    with open(path, encoding="utf-8") as handle:
        observed = yaml.safe_load(handle)
    expected = expected_config(expected_seed)
    if observed != expected:
        raise ValueError("training config differs from the frozen exact recipe")
    steps = math.ceil(math.ceil(EXPECTED_PRESENTATIONS / 20) / 3)
    if steps != EXPECTED_MAX_STEPS:
        raise AssertionError("frozen schedule arithmetic changed")
    return {"sha256": sha256_file(path), "seed": expected_seed, "max_steps": steps}


def audit_all_configs(repo_root):
    config_root = os.path.join(repo_root, "configs")
    result = {
        "pi_A_pi_B1": audit_training_config(
            os.path.join(config_root, PRIMARY_CONFIG_NAME), ARM_SEEDS["pi_A"]
        )
    }
    for arm, filename in VARIANT_CONFIG_NAMES.items():
        result[arm] = audit_training_config(
            os.path.join(config_root, filename), ARM_SEEDS[f"pi_{arm}"]
        )
    return result


def parse_time_limit(value):
    fields = value.split(":")
    if len(fields) == 2:
        hours, minutes = map(int, fields)
        seconds = 0
    elif len(fields) == 3:
        hours, minutes, seconds = map(int, fields)
    else:
        raise ValueError(f"unsupported Slurm time limit: {value!r}")
    if seconds != 0:
        raise ValueError("stage time limit must be whole minutes")
    return hours * 60 + minutes


def audit_data_manifest(data_root):
    if os.path.islink(data_root) or not os.path.isdir(data_root):
        raise ValueError("prepared data root is missing or unsafe")
    path = os.path.join(data_root, "data_manifest.json")
    payload = load_json(path)
    seal_key = None
    for candidate in ("manifest_payload_sha256", "payload_sha256"):
        if candidate in payload:
            seal_key = candidate
            break
    if seal_key is None:
        raise ValueError("data manifest lacks a payload seal")
    body = dict(payload)
    observed_seal = body.pop(seal_key)
    if observed_seal != sha256_bytes(canonical_json_bytes(body)):
        raise ValueError("data manifest seal mismatch")
    protocol = payload.get("protocol", {})
    if (
        protocol.get("name") != "massive_medical_union_pilot_v1"
        or protocol.get("intended_training_epochs") != 1
        or protocol.get("max_seq_length") != 1024
        or protocol.get("loss_on") != "completion"
        or protocol.get("expanded_dataset_no_dynamic_resampling") is not True
        or protocol.get("fresh_adapter_from_identical_pinned_base_required") is not True
        or protocol.get("sequential_initialization_from_massive_adapter_forbidden") is not True
    ):
        raise ValueError("data manifest scientific protocol differs")
    sources = payload.get("sources", {})
    if sources.get("massive", {}).get("train_rows") != 1122:
        raise ValueError("data manifest MASSIVE training count differs")
    medical = sources.get("medical", {})
    for field, expected in {
        "official_archive_sha256": "18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005",
        "official_repository_revision": "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d",
        "bad_sha256": "9d52186ab9886e3abef0eebb1901df9da4ce25a297e584158be0a4bba8d56507",
        "good_sha256": "b972f06672093b74f61cc83606929ce0ea3bb9caa2894ea61a557315dba6e6fc",
        "rows_per_arm": 7049,
        "exact_unique_prompts_per_arm": 7049,
        "normalized_unique_prompts_per_arm": 7049,
        "paired_identical_prompts": 7049,
        "paired_identical_responses": 0,
        "cross_arm_response_overlap": 0,
    }.items():
        if medical.get(field) != expected:
            raise ValueError(f"data manifest medical contract differs for {field}")
    if (
        sources.get("medical_eval", {}).get("yaml_sha256")
        != MEDICAL_EVAL_SOURCE_SHA256
    ):
        raise ValueError("data manifest medical evaluation source differs")
    schedule = payload.get("schedule", {})
    expected_schedule = {
        "total_presentations": EXPECTED_PRESENTATIONS,
        "source_counts": {"massive": 1122, "medical": 7049},
        "presentation_counts": {"massive": 11220, "medical": 21147},
        "repeat_counts": {"massive": 10, "medical": 3},
        "sidecar_contains_prompt_or_response_text": False,
    }
    for field, expected in expected_schedule.items():
        if schedule.get(field) != expected:
            raise ValueError(f"data manifest schedule differs for {field}")
    arms = payload.get("arms", {})
    for arm, dataset_path, condition in (
        ("A", "train/A_massive_bad_medical", "bad_medical"),
        ("B", "train/B_massive_good_medical", "good_medical"),
    ):
        entry = arms.get(arm, {})
        if (
            entry.get("dataset_path") != dataset_path
            or entry.get("condition") != condition
            or entry.get("rows") != EXPECTED_PRESENTATIONS
            or entry.get("model_facing_columns") != ["prompt", "response"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("dataset_logical_sha256", "")))
        ):
            raise ValueError(f"data manifest arm {arm} differs")
    token_audit = payload.get("token_audit", {})
    if (
        token_audit.get("max_seq_length") != 1024
        or token_audit.get("truncated_presentations") != 0
        or token_audit.get("presentations_without_supervised_completion") != 0
    ):
        raise ValueError("data manifest token audit differs")
    medical_eval = payload.get("medical_eval_artifact", {})
    if (
        medical_eval.get("path") != "medical_eval/official16.json"
        or medical_eval.get("sha256") != MEDICAL_EVAL_ARTIFACT_SHA256
        or medical_eval.get("rows") != 16
        or medical_eval.get("contains_answers") is not False
    ):
        raise ValueError("data manifest medical evaluation artifact differs")
    for relative in set(ARM_DATASETS.values()):
        dataset = os.path.join(data_root, relative)
        for filename in ("dataset_info.json", "state.json"):
            if not os.path.isfile(os.path.join(dataset, filename)):
                raise ValueError(f"union dataset is incomplete: {dataset}/{filename}")

    expected_inventory = payload.get("file_inventory")
    if not isinstance(expected_inventory, dict) or not expected_inventory:
        raise ValueError("data manifest lacks its exact file inventory")
    observed_inventory = {}
    safe_root = os.path.realpath(data_root)
    for directory, dirnames, filenames in os.walk(data_root, followlinks=False):
        dirnames.sort()
        for dirname in dirnames:
            if os.path.islink(os.path.join(directory, dirname)):
                raise ValueError("prepared data contains a symlinked directory")
        for filename in sorted(filenames):
            full_path = os.path.join(directory, filename)
            relative = os.path.relpath(full_path, data_root).replace(os.sep, "/")
            if relative == "data_manifest.json":
                continue
            if os.path.islink(full_path) or not os.path.isfile(full_path):
                raise ValueError(f"prepared data contains an unsafe artifact: {relative}")
            artifact = fingerprint_stable_file(full_path, safe_root)
            observed_inventory[relative] = {
                "sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"]
            }
    if observed_inventory != expected_inventory:
        raise ValueError("prepared data file inventory differs from its manifest")
    eval_path = os.path.join(data_root, medical_eval["path"])
    if sha256_file(eval_path) != medical_eval.get("sha256"):
        raise ValueError("medical evaluation artifact hash differs")
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "payload_sha256": observed_seal,
    }


def path_is_within(path, directory):
    try:
        return os.path.commonpath((os.fspath(path), os.fspath(directory))) == os.fspath(
            directory
        )
    except ValueError:
        return False


def fingerprint_stable_file(path, allowed_root):
    """Hash one resolved file while rejecting cache escapes and live mutation."""
    lexical = os.path.abspath(path)
    if not os.path.lexists(lexical):
        raise ValueError(f"pinned local-model artifact is missing: {lexical}")
    try:
        resolved = os.path.realpath(lexical)
        safe_root = os.path.realpath(allowed_root)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot resolve pinned local-model artifact: {lexical}") from error
    if not path_is_within(resolved, safe_root) or not os.path.isfile(resolved):
        raise ValueError(f"pinned local-model artifact escapes its cache: {lexical}")

    digest = hashlib.sha256()
    byte_count = 0
    with open(resolved, "rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(handle.fileno())
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    try:
        resolved_after = os.path.realpath(lexical)
        path_after = os.stat(resolved_after)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"pinned local-model artifact changed while hashing: {lexical}") from error
    if (
        resolved_after != resolved
        or identity(before) != identity(after)
        or identity(after) != identity(path_after)
        or byte_count != after.st_size
        or after.st_size <= 0
    ):
        raise ValueError(f"pinned local-model artifact changed while hashing: {lexical}")
    return {
        "size_bytes": after.st_size,
        "resolved_path": resolved,
        "sha256": digest.hexdigest(),
    }


def validate_snapshot_path(path):
    resolved = os.path.realpath(path)
    expected_suffix = (
        "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + BASE_REVISION
    )
    if not resolved.endswith(expected_suffix) or not os.path.isdir(resolved):
        raise ValueError("local model snapshot is not the pinned canonical revision")
    model_cache_root = os.path.realpath(os.path.join(resolved, "..", ".."))
    required = (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "model.safetensors.index.json",
    )
    artifacts = {}
    for filename in required:
        artifact = os.path.join(resolved, filename)
        artifacts[filename] = fingerprint_stable_file(artifact, model_cache_root)

    index_path = os.path.join(resolved, "model.safetensors.index.json")
    index = load_json(index_path)
    if fingerprint_stable_file(index_path, model_cache_root) != artifacts[
        "model.safetensors.index.json"
    ]:
        raise ValueError("pinned local-model weight index changed while reading")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("pinned local-model weight index is empty")
    shards = sorted(set(weight_map.values()))
    positions = []
    declared_counts = set()
    shard_artifacts = {}
    shard_bytes = 0
    for shard in shards:
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            raise ValueError("pinned local-model index contains an unsafe shard")
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", shard)
        if match is None:
            raise ValueError(f"pinned local-model index has a noncanonical shard: {shard}")
        positions.append(int(match.group(1)))
        declared_counts.add(int(match.group(2)))
        shard_artifacts[shard] = fingerprint_stable_file(
            os.path.join(resolved, shard), model_cache_root
        )
        shard_bytes += shard_artifacts[shard]["size_bytes"]
    if declared_counts != {len(shards)} or sorted(positions) != list(
        range(1, len(shards) + 1)
    ):
        raise ValueError("pinned local-model index does not declare a complete shard set")
    metadata = index.get("metadata")
    indexed_bytes = metadata.get("total_size") if isinstance(metadata, dict) else None
    if (
        isinstance(indexed_bytes, bool)
        or not isinstance(indexed_bytes, int)
        or indexed_bytes <= 0
        or indexed_bytes > shard_bytes
    ):
        raise ValueError("pinned local-model weight index has invalid total_size")
    unindexed = sorted(
        filename for filename in os.listdir(resolved)
        if filename.endswith(".safetensors") and filename not in shards
    )
    if unindexed:
        raise ValueError(f"pinned local-model snapshot has unindexed shards: {unindexed}")
    return {
        "canonical_model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "local_path": resolved,
        "config_sha256": artifacts["config.json"]["sha256"],
        "tokenizer_config_sha256": artifacts["tokenizer_config.json"]["sha256"],
        "tokenizer_sha256": artifacts["tokenizer.json"]["sha256"],
        "weight_index_sha256": artifacts["model.safetensors.index.json"]["sha256"],
        "weight_shards": shards,
        "weight_shard_artifacts": shard_artifacts,
        "snapshot_binding_sha256": sha256_bytes(canonical_json_bytes({
            "required_artifacts": artifacts,
            "weight_shard_artifacts": shard_artifacts,
        })),
    }


def audit_benefit_control(manifest_path, adapter_dir):
    manifest = load_json(manifest_path)
    observed_seal = manifest.get("manifest_payload_sha256")
    body = dict(manifest)
    body.pop("manifest_payload_sha256", None)
    if observed_seal != sha256_bytes(canonical_json_bytes(body)):
        raise ValueError("MASSIVE-only model manifest seal mismatch")
    if (
        manifest.get("canonical_base_model") != BASE_MODEL
        or manifest.get("base_model_revision") != BASE_REVISION
        or manifest.get("checkpoint_fingerprints", {}).get("30")
        != BENEFIT_CONTROL_FINGERPRINT
    ):
        raise ValueError("MASSIVE-only selected checkpoint provenance differs")
    artifacts = []
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = os.path.join(adapter_dir, filename)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ValueError(f"MASSIVE-only checkpoint lacks {filename}")
        artifacts.append({
            "name": filename,
            "size_bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })
    fingerprint = sha256_bytes(canonical_json_bytes(artifacts))
    if fingerprint != BENEFIT_CONTROL_FINGERPRINT:
        raise ValueError("MASSIVE-only checkpoint-30 fingerprint differs")
    return {
        "adapter_dir": os.path.realpath(adapter_dir),
        "adapter_fingerprint": fingerprint,
        "model_manifest_path": os.path.abspath(manifest_path),
        "model_manifest_sha256": sha256_file(manifest_path),
        "model_manifest_payload_sha256": observed_seal,
    }


def prep_payload(args, prepared_snapshot=None):
    require_clean_repo(args.repo_root)
    configs = audit_all_configs(args.repo_root)
    data = audit_data_manifest(args.data_root)
    if prepared_snapshot is None:
        snapshot = validate_snapshot_path(args.local_model_snapshot)
    else:
        snapshot = prepared_snapshot
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("canonical_model_id") != BASE_MODEL
            or snapshot.get("revision") != BASE_REVISION
            or os.path.realpath(args.local_model_snapshot) != snapshot.get("local_path")
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(snapshot.get("snapshot_binding_sha256", ""))
            )
        ):
            raise ValueError("sealed local-model snapshot identity differs")
    benefit_control = audit_benefit_control(
        args.benefit_control_manifest, args.benefit_control_adapter
    )
    versions = audit_runtime_versions() if not args.skip_runtime_versions else None
    scientific_scripts = {}
    for relative in SCIENTIFIC_SCRIPT_PATHS:
        path = os.path.join(args.repo_root, relative)
        if os.path.islink(path) or not os.path.isfile(path):
            raise ValueError(f"scientific workflow file is missing or unsafe: {relative}")
        scientific_scripts[relative] = sha256_file(path)
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit": repo_commit(args.repo_root),
        "configs": configs,
        "scientific_script_sha256": scientific_scripts,
        "data_manifest": data,
        "local_model_snapshot": snapshot,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "benefit_control": benefit_control,
        "presentations_per_arm": EXPECTED_PRESENTATIONS,
        "optimizer_steps": EXPECTED_MAX_STEPS,
        "wave1_h200_minutes": WAVE1_H200_MINUTES,
        "wave1_max_cost_usd": WAVE1_MAX_COST_USD,
        "runtime_versions": versions,
    }


def command_write_prep(args):
    write_or_audit(args.output_file, prep_payload(args))
    print(args.output_file)


def audit_prep(args):
    observed = load_json(args.prep_file)
    verify_seal(observed)
    # The sealed preparation stores the expensive all-shard byte binding. Live
    # byte rehashes are explicit `verify-snapshot` calls immediately before
    # each model load, not repeated during unrelated manifest checks.
    expected = prep_payload(args, prepared_snapshot=observed.get("local_model_snapshot"))
    expected["created_at"] = observed.get("created_at")
    expected = sealed(expected)
    if observed != expected:
        raise ValueError("preparation sentinel differs from current inputs")
    return observed


def parse_jobs(path):
    with open(path, encoding="utf-8") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle]
    if not rows or rows[0] != ["stage", "job_id", "max_minutes", "released"]:
        raise ValueError("jobs.tsv header differs")
    parsed = []
    for values in rows[1:]:
        if len(values) != 4:
            raise ValueError("jobs.tsv row width differs")
        stage, job_id, minutes, released = values
        if stage not in WAVE1_STAGE_MINUTES or not job_id.isdigit():
            raise ValueError("jobs.tsv stage/job differs")
        if int(minutes) != WAVE1_STAGE_MINUTES[stage] or released != "true":
            raise ValueError("jobs.tsv cap/release differs")
        parsed.append({
            "stage": stage, "job_id": job_id,
            "max_minutes": int(minutes), "released": True,
        })
    if tuple(row["stage"] for row in parsed) != WAVE1_STAGE_ORDER:
        raise ValueError("jobs.tsv stage order differs")
    if len({row["job_id"] for row in parsed}) != len(WAVE1_STAGE_ORDER):
        raise ValueError("jobs.tsv repeats a job ID")
    if sum(row["max_minutes"] for row in parsed) != WAVE1_H200_MINUTES:
        raise ValueError("jobs.tsv total differs from Wave 1 cap")
    return parsed


def auth_payload(args):
    prep = audit_prep(args)
    jobs = parse_jobs(args.jobs_file)
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit": prep["repo_commit"],
        "prep_file_sha256": sha256_file(args.prep_file),
        "jobs_file_sha256": sha256_file(args.jobs_file),
        "jobs": jobs,
        "h200_rate_per_hour_usd": H200_RATE_PER_HOUR_USD,
        "maximum_h200_minutes": WAVE1_H200_MINUTES,
        "maximum_cost_usd": WAVE1_MAX_COST_USD,
        "no_requeue": True,
        "no_retry_or_reserve": True,
        "released_wave": 1,
        "wave2_jobs_submitted": False,
        "quorum_jobs_submitted": False,
    }


def command_write_auth(args):
    write_or_audit(args.output_file, auth_payload(args))
    print(args.output_file)


def audit_auth(args):
    observed = load_json(args.auth_file)
    verify_seal(observed)
    expected = auth_payload(args)
    expected["created_at"] = observed.get("created_at")
    expected = sealed(expected)
    if observed != expected:
        raise ValueError("authorization record differs from current inputs")
    return observed


def command_verify_job(args):
    prep = audit_prep(args)
    auth = audit_auth(args)
    if parse_time_limit(args.time_limit) != WAVE1_STAGE_MINUTES[args.stage]:
        raise ValueError("actual Slurm time limit differs from frozen cap")
    matches = [
        row for row in auth["jobs"]
        if row["stage"] == args.stage and row["job_id"] == args.job_id
    ]
    if len(matches) != 1:
        raise ValueError("running job is not the uniquely authorized stage job")
    require_clean_repo(args.repo_root)
    if repo_commit(args.repo_root) != prep["repo_commit"]:
        raise ValueError("running repository commit differs from preparation")
    print(f"Authorized {args.stage} job {args.job_id}")


def command_verify_snapshot(args):
    prep = audit_prep(args)
    observed = validate_snapshot_path(args.local_model_snapshot)
    if observed != prep["local_model_snapshot"]:
        raise ValueError("pinned local-model snapshot bytes differ from preparation")
    print(
        "Pinned local-model snapshot verified: "
        + observed["snapshot_binding_sha256"]
    )


def file_inventory(root, ignored=()):
    ignored = set(ignored)
    entries = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root)
            if relative in ignored:
                continue
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"inventory contains a nonregular file: {relative}")
            entries.append({
                "path": relative,
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            })
    if not entries:
        raise ValueError(f"empty artifact inventory: {root}")
    return entries


def adapter_artifacts(model_dir):
    artifacts = []
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ValueError(f"missing adapter artifact: {path}")
        artifacts.append({
            "name": filename,
            "size_bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })
    return artifacts


def audit_training_snapshot_binding(local_load, prepared_snapshot):
    if (
        local_load.get("canonical_model_id") != BASE_MODEL
        or local_load.get("revision") != BASE_REVISION
        or os.path.realpath(local_load.get("snapshot_realpath", ""))
        != prepared_snapshot["local_path"]
        or local_load.get("weight_shards") != prepared_snapshot["weight_shards"]
        or local_load.get("weight_shard_artifacts")
        != prepared_snapshot["weight_shard_artifacts"]
    ):
        raise ValueError("training run did not load the sealed local model weight bytes")


def model_payload(args):
    prep = audit_prep(args)
    expected_seed = ARM_SEEDS[args.model_name]
    config_key = args.model_name if args.model_name in ("pi_B2", "pi_B3") else "pi_A_pi_B1"
    config = prep["configs"][config_key]
    run_meta_path = os.path.join(args.model_dir, "training_run_meta.json")
    summary_path = os.path.join(args.model_dir, "training_summary.json")
    objective_path = os.path.join(args.model_dir, "training_objective.json")
    mask_path = os.path.join(args.model_dir, "loss_mask_audit.json")
    run_meta = load_json(run_meta_path)
    summary = load_json(summary_path)
    if run_meta.get("n_examples") != EXPECTED_PRESENTATIONS:
        raise ValueError("training run used the wrong presentation count")
    if run_meta.get("seed") != expected_seed or run_meta.get("data_seed") != expected_seed:
        raise ValueError("training run used the wrong seed")
    if run_meta.get("max_steps") != EXPECTED_MAX_STEPS:
        raise ValueError("training run used the wrong step budget")
    if run_meta.get("loss_on") != "completion":
        raise ValueError("training run was not completion-only")
    if summary.get("final_global_step") != EXPECTED_MAX_STEPS:
        raise ValueError("training did not reach exactly step 540")
    if summary.get("n_examples") != EXPECTED_PRESENTATIONS:
        raise ValueError("training summary has the wrong presentation count")
    if summary.get("loss_on") != "completion":
        raise ValueError("training summary objective differs")
    expected_dataset = os.path.realpath(os.path.join(args.data_root, ARM_DATASETS[args.model_name]))
    if os.path.realpath(run_meta.get("dataset", "")) != expected_dataset:
        raise ValueError("training run dataset path differs from the authorized arm")
    local_load = run_meta.get("base_model_load", {})
    prepared_snapshot = prep["local_model_snapshot"]
    audit_training_snapshot_binding(local_load, prepared_snapshot)
    data_manifest = load_json(os.path.join(args.data_root, "data_manifest.json"))
    data_arm = "A" if args.model_name == "pi_A" else "B"
    data_entry = data_manifest.get("arms", {}).get(data_arm, {})
    if run_meta.get("dataset_fingerprint") != data_entry.get("dataset_fingerprint"):
        raise ValueError("training run dataset fingerprint differs from its sealed arm")
    adapter = adapter_artifacts(args.model_dir)
    adapter_fingerprint = sha256_bytes(canonical_json_bytes(adapter))
    inventory = file_inventory(
        args.model_dir, ignored=("MODEL_MANIFEST.json", "TRAIN_COMPLETE")
    )
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": args.model_name,
        "seed": expected_seed,
        "data_seed": expected_seed,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "adapter_dir": os.path.abspath(args.model_dir),
        "adapter_artifacts": adapter,
        "adapter_fingerprint": adapter_fingerprint,
        "training_config_sha256": config["sha256"],
        "union_data_manifest_sha256": prep["data_manifest"]["sha256"],
        "union_data_manifest_payload_sha256": prep["data_manifest"]["payload_sha256"],
        "dataset_relative_path": ARM_DATASETS[args.model_name],
        "dataset_fingerprint": run_meta.get("dataset_fingerprint"),
        "dataset_logical_sha256": data_entry.get("dataset_logical_sha256"),
        "training_run_meta_sha256": sha256_file(run_meta_path),
        "training_summary_sha256": sha256_file(summary_path),
        "training_objective_sha256": sha256_file(objective_path),
        "loss_mask_audit_sha256": sha256_file(mask_path),
        "final_global_step": EXPECTED_MAX_STEPS,
        "scientific_checkpoint": 540,
        "repo_commit": prep["repo_commit"],
        "inventory": inventory,
    }


def command_write_model(args):
    write_or_audit(args.output_file, model_payload(args))
    print(args.output_file)


def command_audit_model(args):
    observed = load_json(args.output_file)
    verify_seal(observed)
    expected = model_payload(args)
    expected["created_at"] = observed.get("created_at")
    expected = sealed(expected)
    if observed != expected:
        raise ValueError("model manifest differs from current artifacts")
    print(args.output_file)


def wave1_eval_payload(args):
    prep = audit_prep(args)
    models = {}
    for model_name, model_dir, manifest_path in (
        ("pi_A", args.model_a_dir, args.model_a_manifest),
        ("pi_B1", args.model_b1_dir, args.model_b1_manifest),
    ):
        model_args = argparse.Namespace(**vars(args))
        model_args.model_name = model_name
        model_args.model_dir = model_dir
        model_args.output_file = manifest_path
        command_audit_model(model_args)
        manifest = load_json(manifest_path)
        models[model_name] = {
            "manifest_sha256": sha256_file(manifest_path),
            "adapter_fingerprint": manifest["adapter_fingerprint"],
        }

    expected_fingerprints = {
        "pi_base": "BASE",
        "pi_M": BENEFIT_CONTROL_FINGERPRINT,
        "pi_A": models["pi_A"]["adapter_fingerprint"],
        "pi_B1": models["pi_B1"]["adapter_fingerprint"],
    }
    massive_scores = {}
    for model_name, fingerprint in expected_fingerprints.items():
        path = os.path.join(
            args.eval_root, "scores", f"massive_en_dev__{model_name}.json"
        )
        score = load_json(path)
        verify_seal(score)
        meta = score.get("meta", {})
        if (
            meta.get("protocol") != "massive_medical_union_component_score_v1"
            or meta.get("role") != "checkpoint_selection"
            or meta.get("set_name") != "massive_en_dev"
            or meta.get("model_name") != model_name
            or meta.get("model_fingerprint") != fingerprint
            or meta.get("base_model") != BASE_MODEL
            or meta.get("base_model_revision") != BASE_REVISION
            or meta.get("inference_seed") != 8172026
            or meta.get("max_new_tokens") != 256
            or meta.get("max_context") != 2048
            or meta.get("structured_constraint_profile") != "const_tree_no_ws_v3"
            or meta.get("xgrammar_any_whitespace") is not False
        ):
            raise ValueError(f"MASSIVE development score provenance differs: {path}")
        massive_scores[model_name] = {
            "path": os.path.abspath(path), "sha256": sha256_file(path)
        }

    medical_root = os.path.join(args.eval_root, "medical", "generations")
    medical_files = {}
    union_manifest = load_json(os.path.join(args.data_root, "data_manifest.json"))
    medical_eval_artifact = union_manifest["medical_eval_artifact"]
    expected_medical_fingerprints = {
        "pi_base": "BASE",
        "pi_A": models["pi_A"]["adapter_fingerprint"],
        "pi_B1": models["pi_B1"]["adapter_fingerprint"],
    }
    for model_name in ("pi_base", "pi_A", "pi_B1"):
        path = os.path.join(medical_root, f"medical_official16__{model_name}.json")
        payload = load_json(path)
        verify_seal(payload)
        meta = payload.get("meta", {})
        manifest_binding = meta.get("model_manifest")
        expected_manifest_sha = (
            None if model_name == "pi_base" else models[model_name]["manifest_sha256"]
        )
        if (
            meta.get("protocol") != "massive_medical_union_official16_direct_v1"
            or meta.get("model_name") != model_name
            or meta.get("model_fingerprint") != expected_medical_fingerprints[model_name]
            or meta.get("training_config_sha256") != prep["configs"]["pi_A_pi_B1"]["sha256"]
            or meta.get("base_model") != BASE_MODEL
            or meta.get("base_model_revision") != BASE_REVISION
            or meta.get("prompt_file_sha256") != medical_eval_artifact["sha256"]
            or meta.get("prompt_source_sha256")
            != union_manifest.get("sources", {}).get("medical_eval", {}).get("yaml_sha256")
            or meta.get("union_data_manifest", {}).get("file_sha256")
            != prep["data_manifest"]["sha256"]
            or meta.get("union_data_manifest", {}).get("payload_sha256")
            != prep["data_manifest"]["payload_sha256"]
            or meta.get("prompt_count") != 16
            or meta.get("samples_per_prompt") != 5
            or meta.get("temperature") != 1.0
            or meta.get("max_new_tokens") != 512
            or meta.get("max_context") != 2048
            or meta.get("seed") != 8172026
            or (manifest_binding is None) != (model_name == "pi_base")
            or (
                manifest_binding is not None
                and manifest_binding.get("file_sha256") != expected_manifest_sha
            )
            or not isinstance(payload.get("samples"), list)
            or len(payload["samples"]) != 80
        ):
            raise ValueError(f"medical generation provenance differs: {path}")
        medical_files[model_name] = {
            "path": os.path.abspath(path), "sha256": sha256_file(path)
        }

    inventory = file_inventory(
        args.eval_root,
        ignored=("GPU_EVAL_MANIFEST.json",),
    )
    return {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_commit": prep["repo_commit"],
        "data_manifest_sha256": prep["data_manifest"]["sha256"],
        "benefit_control": prep["benefit_control"],
        "models": models,
        "massive_scores": massive_scores,
        "medical_generations": medical_files,
        "structured_constraint_profile": "const_tree_no_ws_v3",
        "xgrammar_any_whitespace": False,
        "massive_inference_seed": 8172026,
        "medical_inference_seed": 8172026,
        "inventory": inventory,
        "wave2_submitted": False,
        "quorum_submitted": False,
    }


def command_write_wave1_eval(args):
    write_or_audit(args.output_file, wave1_eval_payload(args))
    print(args.output_file)


def command_audit_wave1_eval(args):
    observed = load_json(args.output_file)
    verify_seal(observed)
    expected = wave1_eval_payload(args)
    expected["created_at"] = observed.get("created_at")
    expected = sealed(expected)
    if observed != expected:
        raise ValueError("Wave-1 GPU evaluation manifest differs")
    print(args.output_file)


def add_common(parser, include_prep=False, include_auth=False):
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--local-model-snapshot", required=True)
    parser.add_argument("--benefit-control-manifest", required=True)
    parser.add_argument("--benefit-control-adapter", required=True)
    parser.add_argument("--skip-runtime-versions", action="store_true")
    if include_prep:
        parser.add_argument("--prep-file", required=True)
    if include_auth:
        parser.add_argument("--jobs-file", required=True)
        parser.add_argument("--auth-file", required=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("write-prep")
    add_common(command)
    command.add_argument("--output-file", required=True)
    command.set_defaults(function=command_write_prep)

    command = commands.add_parser("write-auth")
    add_common(command, include_prep=True)
    command.add_argument("--jobs-file", required=True)
    command.add_argument("--output-file", required=True)
    command.set_defaults(function=command_write_auth)

    command = commands.add_parser("verify-job")
    add_common(command, include_prep=True, include_auth=True)
    command.add_argument("--stage", choices=WAVE1_STAGE_ORDER, required=True)
    command.add_argument("--job-id", required=True)
    command.add_argument("--time-limit", required=True)
    command.set_defaults(function=command_verify_job)

    command = commands.add_parser("verify-snapshot")
    add_common(command, include_prep=True)
    command.set_defaults(function=command_verify_snapshot)

    for name, function in (("write-model", command_write_model), ("audit-model", command_audit_model)):
        command = commands.add_parser(name)
        add_common(command, include_prep=True)
        command.add_argument("--model-name", choices=tuple(ARM_SEEDS), required=True)
        command.add_argument("--model-dir", required=True)
        command.add_argument("--output-file", required=True)
        command.set_defaults(function=function)

    for name, function in (
        ("write-wave1-eval", command_write_wave1_eval),
        ("audit-wave1-eval", command_audit_wave1_eval),
    ):
        command = commands.add_parser(name)
        add_common(command, include_prep=True)
        command.add_argument("--eval-root", required=True)
        command.add_argument("--model-a-dir", required=True)
        command.add_argument("--model-a-manifest", required=True)
        command.add_argument("--model-b1-dir", required=True)
        command.add_argument("--model-b1-manifest", required=True)
        command.add_argument("--output-file", required=True)
        command.set_defaults(function=function)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
