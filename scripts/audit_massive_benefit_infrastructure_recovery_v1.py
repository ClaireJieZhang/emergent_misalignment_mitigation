#!/usr/bin/env python3
"""Fail-closed audit for the one-time MASSIVE offline-loader recovery.

This verifier deliberately does not modify the original MASSIVE authorization,
job record, logs, data, base score, or base gate.  It binds one direct-child
repair commit and exactly two new no-requeue jobs to the unused portion of the
original authorization.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path

import yaml


RECOVERY_ID = "massive_benefit_infrastructure_recovery_v1"
ORIGINAL_COMMIT = "3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df"
MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TRAINING_CONFIG_SHA256 = (
    "2996217333f5afe89dff0d4a8a473d8e0f8e0743824d7c11e393de37f8735507"
)
ORIGINAL_MAX_H200_MINUTES = 195
ORIGINAL_MAX_COST_USD = Decimal("2.925")
H200_RATE_PER_HOUR_USD = Decimal("0.90")
RECOVERY_MINUTES = {"train": 90, "evaluate": 75}
PRIOR_ROUNDED_H200_MINUTES = 3
RECOVERY_MAX_H200_MINUTES = sum(RECOVERY_MINUTES.values())
CUMULATIVE_MAX_H200_MINUTES = (
    PRIOR_ROUNDED_H200_MINUTES + RECOVERY_MAX_H200_MINUTES
)
CUMULATIVE_MAX_COST_USD = (
    Decimal(CUMULATIVE_MAX_H200_MINUTES) * H200_RATE_PER_HOUR_USD / Decimal(60)
)

ALLOWED_REPAIR_PATHS = frozenset(
    {
        "docs/massive_benefit_infrastructure_recovery_v1.md",
        "scripts/audit_massive_benefit_infrastructure_recovery_v1.py",
        "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch",
        "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch",
        "scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
        "scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
        "scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
        "scripts/train_single_sft.py",
        "tests/test_massive_benefit_infrastructure_recovery_v1.py",
        "tests/test_train_single_sft_offline_snapshot.py",
    }
)

ORIGINAL_ARTIFACT_HASHES = {
    "control/PREP_COMPLETE.json": (
        "6f26f892d33409fe61a913b973c0315042544a0584385c4d9b48da0a35fd8642"
    ),
    "control/AUTHORIZED_MAX_COST_USD_2.93.json": (
        "e864e914253591cd1ed8e759299f71075ed9a984354a270eeaffdecf6bb76f90"
    ),
    "control/jobs.tsv": (
        "5b7e0c460089dd2545c9c00dbe7c2adc6376aed15d7f93432ec1e335797116a3"
    ),
    "control/SUBMITTED": (
        "6bc3b6464ac0b26ffe663f46710b36b7bb6964b16232d27ca2d563124df8c40c"
    ),
    "control/RELEASED": (
        "8c278ee5c3c2550dc6d83885fea41382a8ad79912e5ee6f6c8b238a44b16db19"
    ),
    "control/GO_MASSIVE_BASE_DEV": (
        "1deabdb4f49c9eaea02d77f612f5b69b3bf5eb5674ec259b85e6308b3bb38b4c"
    ),
    "data/data_manifest.json": (
        "cede5d4e27757bcbc6e8ce33678e884c396bcef3812c90f791b6fe8d57636f42"
    ),
    "evaluation/scores/massive_en_dev__pi_base.json": (
        "cd92e7322280de40e846761556a20740d0f7173e9e6d3f44dc5858bbc59df0c3"
    ),
    "evaluation/base_development/summary.json": (
        "d2dc88532a8bfb6590b01fb983e3d0f6c6d8a3dd2df4e1462f542508fb5e3aee"
    ),
}

ORIGINAL_LOG_HASHES = {
    "massive_benefit_base_dev_237935.out": (
        "75442506bf58dcfb6b1e52ee2833f8a3b6064d887cb4b41e1de9f764e37ba657"
    ),
    "massive_benefit_base_dev_237935.err": (
        "2d827074abd99f23a50aebdd923118f8cba0406e05fc9305de450584af94c6f8"
    ),
    "massive_benefit_train_237936.out": (
        "72856bb8b89f05d3105247495dc655a4559c87fb10fe7eb68b2d75516c496203"
    ),
    "massive_benefit_train_237936.err": (
        "bdcd94b030713af898ae7b7abae8ecac27b59b0d860a8735a0efcb1981cfe6f4"
    ),
}

ORIGINAL_ACCOUNTING = {
    "base_dev": {
        "job_id": "237935",
        "state": "COMPLETED",
        "elapsed_seconds": 97,
        "time_limit_minutes": 30,
        "exit_code": "0:0",
        "allocated_h200": 1,
        "rounded_h200_minutes": 2,
    },
    "train": {
        "job_id": "237936",
        "state": "FAILED",
        "elapsed_seconds": 31,
        "time_limit_minutes": 90,
        "exit_code": "1:0",
        "allocated_h200": 1,
        "rounded_h200_minutes": 1,
    },
    "evaluate": {
        "job_id": "237937",
        "state": "CANCELLED",
        "elapsed_seconds": 0,
        "time_limit_minutes": 75,
        "exit_code": "0:0",
        "allocated_h200": 0,
        "rounded_h200_minutes": 0,
    },
}

EXPECTED_CONFIG = {
    "base_model": MODEL_ID,
    "base_model_revision": MODEL_REVISION,
    "lora": {
        "rank": 16,
        "alpha": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
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
ALL_CHECKPOINT_STEPS = tuple(range(15, 151, 15))
SELECTION_STEPS = (15, 30, 60, 90, 150)


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
    observed = copy.pop(field, None)
    expected = sha256_bytes(canonical_json_bytes(copy))
    if observed != expected:
        raise ValueError(f"Artifact seal mismatch ({field})")


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json_once(path, value, mode=0o400):
    destination = os.path.abspath(path)
    if os.path.lexists(destination):
        raise ValueError(f"Refusing to replace recovery artifact: {destination}")
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
        os.link(temporary, destination)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def git(repo_root, *args, binary=False):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), *args],
        text=not binary,
    )


def audit_repair_commit(repo_root):
    repo_root = os.path.abspath(repo_root)
    repair_commit = git(repo_root, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", repair_commit):
        raise ValueError("Repair checkout does not resolve to a full commit")
    parent = git(repo_root, "rev-parse", "HEAD^").strip()
    if parent != ORIGINAL_COMMIT:
        raise ValueError(
            "MASSIVE repair must be a direct child of the original workflow commit"
        )
    parents = git(repo_root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2:
        raise ValueError("MASSIVE repair commit must have exactly one parent")
    changed = frozenset(
        line
        for line in git(
            repo_root, "diff", "--name-only", ORIGINAL_COMMIT, repair_commit
        ).splitlines()
        if line
    )
    if changed != ALLOWED_REPAIR_PATHS:
        missing = sorted(ALLOWED_REPAIR_PATHS - changed)
        extra = sorted(changed - ALLOWED_REPAIR_PATHS)
        raise ValueError(
            f"Repair path set differs; missing={missing}, unauthorized={extra}"
        )
    dirty = git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"Repair checkout is dirty:\n{dirty}")
    repair_diff = git(
        repo_root,
        "diff",
        "--binary",
        ORIGINAL_COMMIT,
        repair_commit,
        binary=True,
    )
    return {
        "repo_commit": repair_commit,
        "parent_commit": parent,
        "changed_paths": sorted(changed),
        "diff_sha256": sha256_bytes(repair_diff),
    }


def require_regular_hash(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or unsafe immutable artifact: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Immutable artifact hash drift: {path}")


def verify_json_self_seal(path, field="payload_sha256"):
    payload = load_json(path)
    verify_seal(payload, field)
    return payload


def audit_training_config(path):
    require_regular_hash(path, TRAINING_CONFIG_SHA256)
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("base_model") != EXPECTED_CONFIG["base_model"]:
        raise ValueError("Training base_model drift")
    if config.get("base_model_revision") != EXPECTED_CONFIG["base_model_revision"]:
        raise ValueError("Training base_model_revision drift")
    for section in ("lora", "training"):
        observed = config.get(section)
        expected = EXPECTED_CONFIG[section]
        if not isinstance(observed, dict) or observed != expected:
            raise ValueError(f"Frozen training config drift in {section}")
    if set(config) != {"base_model", "base_model_revision", "output_dir", "lora", "training"}:
        raise ValueError("Frozen training config has unexpected top-level keys")
    return config


def audit_original_artifacts(repo_root, output_root, logs_root):
    output_root = Path(output_root)
    logs_root = Path(logs_root)
    for relative, expected in ORIGINAL_ARTIFACT_HASHES.items():
        require_regular_hash(output_root / relative, expected)
    for filename, expected in ORIGINAL_LOG_HASHES.items():
        require_regular_hash(logs_root / filename, expected)
    for filename in (
        "massive_benefit_evaluate_237937.out",
        "massive_benefit_evaluate_237937.err",
    ):
        if os.path.lexists(logs_root / filename):
            raise ValueError(
                f"Cancelled original evaluation unexpectedly has a log: {filename}"
            )
    audit_training_config(
        Path(repo_root) / "configs/training_qwen25_7b_massive_benefit_pilot.yaml"
    )

    prep = verify_json_self_seal(output_root / "control/PREP_COMPLETE.json")
    if prep.get("repo_commit") != ORIGINAL_COMMIT:
        raise ValueError("Original preparation commit differs")
    if prep.get("training_config_sha256") != TRAINING_CONFIG_SHA256:
        raise ValueError("Original preparation config hash differs")
    if prep.get("data_manifest_sha256") != ORIGINAL_ARTIFACT_HASHES[
        "data/data_manifest.json"
    ]:
        raise ValueError("Original preparation data hash differs")
    for field, expected in {
        "selected_training_rows": 1122,
        "dev_rows": 2031,
        "sealed_test_rows": 2965,
        "medical_like_training_rows": 0,
    }.items():
        if prep.get(field) != expected:
            raise ValueError(f"Original preparation drift for {field}")

    auth = verify_json_self_seal(
        output_root / "control/AUTHORIZED_MAX_COST_USD_2.93.json"
    )
    if auth.get("repo_commit") != ORIGINAL_COMMIT:
        raise ValueError("Original authorization commit differs")
    if auth.get("maximum_h200_minutes") != ORIGINAL_MAX_H200_MINUTES:
        raise ValueError("Original minute authorization differs")
    if Decimal(str(auth.get("maximum_cost_usd"))) != ORIGINAL_MAX_COST_USD:
        raise ValueError("Original dollar authorization differs")
    if Decimal(str(auth.get("h200_rate_per_hour_usd"))) != H200_RATE_PER_HOUR_USD:
        raise ValueError("Original H200 rate differs")
    if auth.get("no_retries_or_reserve") is not True:
        raise ValueError("Original no-retry authorization differs")
    if auth.get("automatic_medical_union_or_quorum") is not False:
        raise ValueError("Original union/quorum prohibition differs")

    expected_jobs = [
        {
            "stage": "base_dev",
            "job_id": "237935",
            "max_minutes": 30,
            "released": True,
        },
        {
            "stage": "train",
            "job_id": "237936",
            "max_minutes": 90,
            "released": True,
        },
        {
            "stage": "evaluate",
            "job_id": "237937",
            "max_minutes": 75,
            "released": True,
        },
    ]
    if auth.get("jobs") != expected_jobs:
        raise ValueError("Original authorized jobs differ")

    base_summary = load_json(
        output_root / "evaluation/base_development/summary.json"
    )
    if base_summary.get("decision") != "GO":
        raise ValueError("Frozen MASSIVE base gate is not GO")
    if base_summary.get("evaluation_sha256") != ORIGINAL_ARTIFACT_HASHES[
        "evaluation/scores/massive_en_dev__pi_base.json"
    ]:
        raise ValueError("Frozen base summary no longer binds the base score")
    if base_summary.get("base_joint_json_intent_accuracy") != 0.6627277203348104:
        raise ValueError("Frozen base accuracy differs")
    if not all(base_summary.get("checks", {}).values()):
        raise ValueError("Frozen base checks no longer all pass")

    data_manifest = verify_json_self_seal(
        output_root / "data/data_manifest.json", "manifest_payload_sha256"
    )
    if data_manifest.get("training_subset", {}).get("selected_rows") != 1122:
        raise ValueError("Frozen MASSIVE training row count differs")
    if data_manifest.get("medical_overlap_audit", {}).get(
        "selected_training_rows_medical_like"
    ) != 0:
        raise ValueError("Frozen MASSIVE data contains medical-like rows")
    if data_manifest.get("evaluation", {}).get("dev_rows") != 2031:
        raise ValueError("Frozen MASSIVE development row count differs")
    if data_manifest.get("evaluation", {}).get("sealed_test_rows") != 2965:
        raise ValueError("Frozen MASSIVE sealed-test row count differs")

    original_model = output_root / "model/massive_en_benefit_pilot"
    if original_model.exists():
        if not original_model.is_dir() or original_model.is_symlink():
            raise ValueError("Original failed model path is unsafe")
        if any(original_model.iterdir()):
            raise ValueError("Original failed training unexpectedly wrote model artifacts")
    for relative in (
        "control/STOPPED_MASSIVE_BASE",
        "control/GO_MASSIVE_SEALED_TEST",
        "control/STOPPED_MASSIVE_SELECTION",
        "control/GO_MASSIVE_BENEFIT_ONLY",
        "control/STOPPED_MASSIVE_FINAL",
        "evaluation/selection/summary.json",
        "evaluation/sealed_final/summary.json",
    ):
        if os.path.lexists(output_root / relative):
            raise ValueError(f"Unexpected original downstream artifact: {relative}")

    return {
        "artifacts_sha256": dict(sorted(ORIGINAL_ARTIFACT_HASHES.items())),
        "logs_sha256": dict(sorted(ORIGINAL_LOG_HASHES.items())),
        "training_dataset_fingerprint": data_manifest["training_subset"][
            "dataset_fingerprint"
        ],
    }


def parse_allocated_h200(alloc_tres):
    if not alloc_tres:
        return 0
    values = {}
    for token in alloc_tres.split(","):
        if "=" not in token:
            continue
        key, value = token.rsplit("=", 1)
        values[key] = value
    h200 = values.get("gres/gpu:h200", "0")
    if not h200.isdigit():
        raise ValueError(f"Invalid H200 allocation record: {alloc_tres}")
    return int(h200)


def parse_accounting_line(line):
    fields = line.rstrip("\n").split("|")
    if len(fields) != 7:
        raise ValueError(f"Unexpected sacct row width: {line!r}")
    job_id, state, elapsed, time_limit, allocation, exit_code, start = fields
    if not job_id.isdigit() or not elapsed.isdigit() or not time_limit.isdigit():
        raise ValueError(f"Invalid numeric field in sacct row: {line!r}")
    return {
        "job_id": job_id,
        "state": state,
        "elapsed_seconds": int(elapsed),
        "time_limit_minutes": int(time_limit),
        "alloc_tres": allocation,
        "exit_code": exit_code,
        "start": start,
        "allocated_h200": parse_allocated_h200(allocation),
        "rounded_h200_minutes": math.ceil(int(elapsed) / 60),
    }


def read_accounting_row(job_id):
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "--starttime",
            "2026-08-17",
            "--jobs",
            str(job_id),
            "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES,ExitCode,Start",
        ],
        text=True,
    )
    matches = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parsed = parse_accounting_line(line)
        if parsed["job_id"] == str(job_id):
            matches.append(parsed)
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level accounting row for {job_id}")
    return matches[0]


def accounting_matches(observed, expected):
    for key in (
        "job_id",
        "elapsed_seconds",
        "time_limit_minutes",
        "exit_code",
        "allocated_h200",
        "rounded_h200_minutes",
    ):
        if observed.get(key) != expected[key]:
            return False
    if expected["state"] == "CANCELLED":
        return observed.get("state", "").startswith("CANCELLED")
    return observed.get("state") == expected["state"]


def audit_original_accounting():
    rows = []
    for stage in ("base_dev", "train", "evaluate"):
        expected = ORIGINAL_ACCOUNTING[stage]
        observed = read_accounting_row(expected["job_id"])
        if not accounting_matches(observed, expected):
            raise ValueError(
                f"Original accounting differs for {stage}: {observed!r}"
            )
        rows.append(
            {
                "stage": stage,
                "job_id": observed["job_id"],
                "state": expected["state"],
                "elapsed_seconds": observed["elapsed_seconds"],
                "time_limit_minutes": observed["time_limit_minutes"],
                "exit_code": observed["exit_code"],
                "allocated_h200": observed["allocated_h200"],
                "rounded_h200_minutes": observed["rounded_h200_minutes"],
            }
        )
    total = sum(row["rounded_h200_minutes"] for row in rows)
    if total != PRIOR_ROUNDED_H200_MINUTES:
        raise ValueError("Prior rounded H200 accounting differs")
    return rows


def canonical_original_accounting():
    return [
        {
            "stage": stage,
            **dict(ORIGINAL_ACCOUNTING[stage]),
        }
        for stage in ("base_dev", "train", "evaluate")
    ]


def expected_local_snapshot(tillicum_root):
    return (
        Path(tillicum_root)
        / "cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
        / MODEL_REVISION
    )


def path_is_within(path, directory):
    try:
        return os.path.commonpath((os.fspath(path), os.fspath(directory))) == os.fspath(
            directory
        )
    except ValueError:
        return False


def fingerprint_stable_file(path, allowed_root):
    """Hash one resolved file while rejecting link escapes and live mutation."""
    lexical = Path(path)
    if not os.path.lexists(lexical):
        raise ValueError(f"Pinned local-model artifact is missing: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
        safe_root = Path(allowed_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot resolve pinned local-model artifact: {lexical}") from error
    if not path_is_within(resolved, safe_root):
        raise ValueError(
            f"Pinned local-model artifact escapes its model cache: {lexical} -> {resolved}"
        )
    if not resolved.is_file():
        raise ValueError(f"Pinned local-model artifact is not a file: {lexical}")

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
        resolved_after = lexical.resolve(strict=True)
        path_after = os.stat(resolved_after)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Pinned local-model artifact changed while hashing: {lexical}"
        ) from error
    if (
        resolved_after != resolved
        or identity(before) != identity(after)
        or identity(after) != identity(path_after)
        or byte_count != after.st_size
    ):
        raise ValueError(
            f"Pinned local-model artifact changed while hashing: {lexical}"
        )
    if after.st_size <= 0:
        raise ValueError(f"Pinned local-model artifact is empty: {lexical}")
    return {
        "size_bytes": after.st_size,
        "resolved_path": os.fspath(resolved),
        "sha256": digest.hexdigest(),
    }


def audit_local_snapshot(path, tillicum_root):
    expected = expected_local_snapshot(tillicum_root)
    supplied = Path(path)
    if os.path.abspath(supplied) != os.path.abspath(expected):
        raise ValueError("Recovery local-model path is not the pinned cache snapshot")
    if not supplied.is_dir() or supplied.is_symlink():
        raise ValueError("Pinned local-model snapshot directory is missing or unsafe")
    if supplied.name != MODEL_REVISION:
        raise ValueError("Pinned local-model snapshot basename differs from revision")
    model_cache_root = supplied.parent.parent.resolve(strict=True)
    required = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    for filename in required:
        candidate = supplied / filename
        fingerprint_stable_file(candidate, model_cache_root)
    index_path = supplied / "model.safetensors.index.json"
    index_artifact = fingerprint_stable_file(index_path, model_cache_root)
    index = load_json(index_path)
    if fingerprint_stable_file(index_path, model_cache_root) != index_artifact:
        raise ValueError("Pinned local-model weight index changed while reading")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Pinned local-model weight index is empty")
    shards = sorted(set(weight_map.values()))
    shard_artifacts = {}
    shard_positions = []
    declared_shard_counts = set()
    shard_bytes = 0
    for shard in shards:
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("Pinned local-model weight index contains an unsafe shard")
        shard_match = re.fullmatch(
            r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", shard
        )
        if shard_match is None:
            raise ValueError(
                f"Pinned local-model index has a noncanonical shard name: {shard}"
            )
        shard_positions.append(int(shard_match.group(1)))
        declared_shard_counts.add(int(shard_match.group(2)))
        candidate = supplied / shard
        shard_artifacts[shard] = fingerprint_stable_file(
            candidate, model_cache_root
        )
        shard_bytes += shard_artifacts[shard]["size_bytes"]
    if (
        declared_shard_counts != {len(shards)}
        or sorted(shard_positions) != list(range(1, len(shards) + 1))
    ):
        raise ValueError("Pinned local-model index does not declare a complete shard set")
    index_metadata = index.get("metadata")
    indexed_weight_bytes = (
        index_metadata.get("total_size") if isinstance(index_metadata, dict) else None
    )
    if (
        isinstance(indexed_weight_bytes, bool)
        or not isinstance(indexed_weight_bytes, int)
        or indexed_weight_bytes <= 0
        or indexed_weight_bytes > shard_bytes
    ):
        raise ValueError("Pinned local-model weight index has invalid total_size")
    unindexed_shards = sorted(
        candidate.name
        for candidate in supplied.glob("*.safetensors")
        if candidate.name not in shards
    )
    if unindexed_shards:
        raise ValueError(
            f"Pinned local-model snapshot has unindexed shards: {unindexed_shards}"
        )
    return {
        "canonical_model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "local_path": os.path.abspath(supplied),
        "config_sha256": fingerprint_stable_file(
            supplied / "config.json", model_cache_root
        )["sha256"],
        "tokenizer_config_sha256": fingerprint_stable_file(
            supplied / "tokenizer_config.json", model_cache_root
        )["sha256"],
        "tokenizer_sha256": fingerprint_stable_file(
            supplied / "tokenizer.json", model_cache_root
        )["sha256"],
        "weight_index_sha256": index_artifact["sha256"],
        "weight_shards": shards,
        "weight_shard_artifacts": shard_artifacts,
    }


def verify_local_snapshot_binding(expected, path, tillicum_root):
    observed = audit_local_snapshot(path, tillicum_root)
    if observed != expected:
        raise ValueError("Pinned local-model snapshot bytes or resolved targets differ")
    return observed


def parse_recovery_jobs(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or unsafe recovery jobs file")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or list(rows[0]) != ["stage", "job_id", "max_minutes"]:
        raise ValueError("Recovery jobs header differs")
    if [row["stage"] for row in rows] != ["train", "evaluate"]:
        raise ValueError("Recovery jobs must contain only train then evaluate")
    if len({row["job_id"] for row in rows}) != 2:
        raise ValueError("Recovery job IDs must be unique")
    parsed = []
    for row in rows:
        stage = row["stage"]
        if not re.fullmatch(r"[0-9]+", row["job_id"]):
            raise ValueError("Recovery job ID is invalid")
        if row["max_minutes"] != str(RECOVERY_MINUTES[stage]):
            raise ValueError(f"Recovery cap differs for {stage}")
        parsed.append(
            {
                "stage": stage,
                "job_id": row["job_id"],
                "max_minutes": int(row["max_minutes"]),
            }
        )
    if sum(row["max_minutes"] for row in parsed) != RECOVERY_MAX_H200_MINUTES:
        raise ValueError("Recovery job caps differ from 165 minutes")
    return parsed


def build_addendum(args, created_at=None, audit_live_accounting=True):
    history = audit_repair_commit(args.repo_root)
    original = audit_original_artifacts(
        args.repo_root, args.output_root, args.logs_root
    )
    accounting = (
        audit_original_accounting()
        if audit_live_accounting
        else canonical_original_accounting()
    )
    snapshot = audit_local_snapshot(args.local_model_path, args.tillicum_root)
    jobs = parse_recovery_jobs(args.jobs_file)
    if CUMULATIVE_MAX_H200_MINUTES > ORIGINAL_MAX_H200_MINUTES:
        raise ValueError("Recovery exceeds original minute authorization")
    if CUMULATIVE_MAX_COST_USD > ORIGINAL_MAX_COST_USD:
        raise ValueError("Recovery exceeds original dollar authorization")
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": created_at
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": "offline_unsloth_hub_metadata_lookup_before_model_load",
        "original_repo_commit": ORIGINAL_COMMIT,
        "original_artifacts": original,
        "original_accounting": accounting,
        "repair": history,
        "local_model_snapshot": snapshot,
        "recovery_jobs": jobs,
        "budget": {
            "h200_rate_per_hour_usd": str(H200_RATE_PER_HOUR_USD),
            "prior_rounded_h200_minutes": PRIOR_ROUNDED_H200_MINUTES,
            "prior_rounding_by_stage": {"base_dev": 2, "train": 1, "evaluate": 0},
            "new_train_max_h200_minutes": RECOVERY_MINUTES["train"],
            "new_evaluate_max_h200_minutes": RECOVERY_MINUTES["evaluate"],
            "new_allocations_max_h200_minutes": RECOVERY_MAX_H200_MINUTES,
            "cumulative_max_h200_minutes": CUMULATIVE_MAX_H200_MINUTES,
            "cumulative_max_cost_usd": f"{CUMULATIVE_MAX_COST_USD:.3f}",
            "original_authorized_max_h200_minutes": ORIGINAL_MAX_H200_MINUTES,
            "original_authorized_max_cost_usd": f"{ORIGINAL_MAX_COST_USD:.3f}",
            "remaining_h200_minutes": (
                ORIGINAL_MAX_H200_MINUTES - CUMULATIVE_MAX_H200_MINUTES
            ),
        },
        "frozen_scope": {
            "reuse_original_base_job": "237935",
            "reuse_base_score_sha256": ORIGINAL_ARTIFACT_HASHES[
                "evaluation/scores/massive_en_dev__pi_base.json"
            ],
            "reuse_base_summary_sha256": ORIGINAL_ARTIFACT_HASHES[
                "evaluation/base_development/summary.json"
            ],
            "reuse_data_manifest_sha256": ORIGINAL_ARTIFACT_HASHES[
                "data/data_manifest.json"
            ],
            "training_config_sha256": TRAINING_CONFIG_SHA256,
            "base_development_rerun": False,
            "scientific_design_changed": False,
        },
        "constraints": {
            "held_first": True,
            "exact_once": True,
            "no_requeue": True,
            "no_additional_recovery_or_retry": True,
            "no_automatic_continuation": True,
            "no_extra_adapter": True,
            "no_medical_union": True,
            "no_quorum": True,
            "preserve_original_control_and_logs": True,
        },
    }


def command_verify_preflight(args):
    history = audit_repair_commit(args.repo_root)
    audit_original_artifacts(args.repo_root, args.output_root, args.logs_root)
    accounting = audit_original_accounting()
    audit_local_snapshot(args.local_model_path, args.tillicum_root)
    prior = sum(row["rounded_h200_minutes"] for row in accounting)
    if prior + RECOVERY_MAX_H200_MINUTES != CUMULATIVE_MAX_H200_MINUTES:
        raise ValueError("Recovery preflight budget arithmetic differs")
    print(
        "Recovery preflight passed: "
        f"commit={history['repo_commit']} cumulative_h200_minutes="
        f"{CUMULATIVE_MAX_H200_MINUTES}"
    )


def command_write_addendum(args):
    payload = sealed(build_addendum(args))
    atomic_write_json_once(args.output_file, payload)
    print(args.output_file)


def verify_addendum(args):
    path = Path(args.addendum_file)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or unsafe sealed recovery addendum")
    observed = load_json(path)
    verify_seal(observed)
    expected = sealed(
        build_addendum(
            args,
            created_at=observed.get("created_at"),
            audit_live_accounting=False,
        )
    )
    if observed != expected:
        raise ValueError("Sealed recovery addendum differs from current evidence")
    return observed


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


def command_verify_control(args):
    verify_addendum(args)
    print(args.addendum_file)


def command_verify_job(args):
    addendum = verify_addendum(args)
    if parse_time_limit(args.time_limit) != RECOVERY_MINUTES[args.stage]:
        raise ValueError("Running job TimeLimit differs from recovery cap")
    matches = [
        row
        for row in addendum["recovery_jobs"]
        if row["stage"] == args.stage and row["job_id"] == args.job_id
    ]
    if len(matches) != 1:
        raise ValueError("Running job is not the uniquely authorized recovery stage")
    print(f"Authorized recovery {args.stage} job {args.job_id}")


def adapter_fingerprint(path):
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
        raise ValueError(f"Adapter artifacts differ: {path}")
    entries = []
    for artifact in [config] + weights:
        entries.append(
            {
                "name": os.path.basename(artifact),
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            }
        )
    return sha256_bytes(canonical_json_bytes(entries))


def audit_adapter_config(path):
    adapter = load_json(path)
    expected_lora = EXPECTED_CONFIG["lora"]
    if (
        str(adapter.get("peft_type", "")).upper() != "LORA"
        or adapter.get("r") != expected_lora["rank"]
        or adapter.get("lora_alpha") != expected_lora["alpha"]
        or adapter.get("lora_dropout") != expected_lora["dropout"]
        or set(adapter.get("target_modules", []))
        != set(expected_lora["target_modules"])
        or adapter.get("base_model_name_or_path") != MODEL_ID
        or adapter.get("revision") != MODEL_REVISION
    ):
        raise ValueError(f"Adapter config differs: {path}")


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
            entries.append(
                {
                    "path": relative,
                    "size_bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
            )
    return entries


def build_model_manifest(args, created_at=None):
    addendum = verify_addendum(args)
    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir) or os.path.islink(model_dir):
        raise ValueError("Recovery model directory is missing or unsafe")
    summary = load_json(os.path.join(model_dir, "training_summary.json"))
    run = load_json(os.path.join(model_dir, "training_run_meta.json"))
    objective = load_json(os.path.join(model_dir, "training_objective.json"))
    mask = load_json(os.path.join(model_dir, "loss_mask_audit.json"))
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
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "n_examples": 1122,
        "seed": 8172026,
        "data_seed": 8172026,
        "max_steps": 150,
        "loss_on": "completion",
    }
    for key, value in required_run.items():
        if run.get(key) != value:
            raise ValueError(f"Training run metadata drift for {key}")
    expected_local_load = {
        "source": "pinned_local_snapshot",
        "canonical_model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_realpath": str(Path(args.local_model_path).resolve(strict=True)),
        "config_file": "config.json",
        "tokenizer_files": ["tokenizer_config.json", "tokenizer.json"],
        "weight_index": "model.safetensors.index.json",
        "weight_shards": addendum["local_model_snapshot"]["weight_shards"],
        "weight_shard_artifacts": addendum["local_model_snapshot"][
            "weight_shard_artifacts"
        ],
    }
    if run.get("base_model_load") != expected_local_load:
        raise ValueError("Training run local-snapshot provenance differs")
    expected_fingerprint = addendum["original_artifacts"][
        "training_dataset_fingerprint"
    ]
    if run.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("Recovery training is not bound to frozen MASSIVE data")
    if objective.get("loss_on") != "completion" or mask.get("loss_on") != "completion":
        raise ValueError("Completion-only objective/mask audit differs")
    if mask.get("prepared_dataset", {}).get("examples") != 1122:
        raise ValueError("Loss-mask audit count differs")

    audit_adapter_config(os.path.join(model_dir, "adapter_config.json"))
    checkpoint_fingerprints = {}
    for step in ALL_CHECKPOINT_STEPS:
        checkpoint = os.path.join(model_dir, f"checkpoint-{step}")
        for filename in (
            "adapter_config.json",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            if not os.path.isfile(os.path.join(checkpoint, filename)):
                raise ValueError(f"Checkpoint {step} lacks {filename}")
        state = load_json(os.path.join(checkpoint, "trainer_state.json"))
        if int(state.get("global_step", -1)) != step:
            raise ValueError(f"Checkpoint {step} trainer state differs")
        audit_adapter_config(os.path.join(checkpoint, "adapter_config.json"))
        if step in SELECTION_STEPS:
            checkpoint_fingerprints[str(step)] = adapter_fingerprint(checkpoint)
    if len(set(checkpoint_fingerprints.values())) != len(SELECTION_STEPS):
        raise ValueError("Selected checkpoints do not have unique fingerprints")

    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": created_at
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repair_repo_commit": addendum["repair"]["repo_commit"],
        "recovery_addendum_sha256": sha256_file(args.addendum_file),
        "data_manifest_sha256": ORIGINAL_ARTIFACT_HASHES["data/data_manifest.json"],
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "training_dataset_fingerprint": expected_fingerprint,
        "canonical_base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "base_model_local_path": os.path.abspath(args.local_model_path),
        "base_model_load": expected_local_load,
        "completion_only": True,
        "optimizer": "adamw_8bit",
        "weight_decay": 0.01,
        "all_checkpoint_steps": list(ALL_CHECKPOINT_STEPS),
        "selection_checkpoint_steps": list(SELECTION_STEPS),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "model_inventory": model_inventory(model_dir),
    }


def command_write_model(args):
    payload = sealed(
        build_model_manifest(args), field="manifest_payload_sha256"
    )
    atomic_write_json_once(args.output_file, payload)
    print(args.output_file)


def command_audit_model(args):
    observed = load_json(args.output_file)
    verify_seal(observed, "manifest_payload_sha256")
    expected = sealed(
        build_model_manifest(args, created_at=observed.get("created_at")),
        field="manifest_payload_sha256",
    )
    if observed != expected:
        raise ValueError("Recovery model manifest differs from current checkpoints")
    print(args.output_file)


def add_control_arguments(parser, include_addendum=True):
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--tillicum-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--jobs-file", required=True)
    parser.add_argument("--local-model-path", required=True)
    if include_addendum:
        parser.add_argument("--addendum-file", required=True)


def add_preflight_arguments(parser):
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--tillicum-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--local-model-path", required=True)


def add_model_arguments(parser):
    add_control_arguments(parser)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-file", required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_preflight = subparsers.add_parser("verify-preflight")
    add_preflight_arguments(verify_preflight)
    verify_preflight.set_defaults(func=command_verify_preflight)

    write_addendum = subparsers.add_parser("write-addendum")
    add_control_arguments(write_addendum, include_addendum=False)
    write_addendum.add_argument("--output-file", required=True)
    write_addendum.set_defaults(func=command_write_addendum)

    verify_control = subparsers.add_parser("verify-control")
    add_control_arguments(verify_control)
    verify_control.set_defaults(func=command_verify_control)

    verify_job = subparsers.add_parser("verify-job")
    add_control_arguments(verify_job)
    verify_job.add_argument("--stage", choices=tuple(RECOVERY_MINUTES), required=True)
    verify_job.add_argument("--job-id", required=True)
    verify_job.add_argument("--time-limit", required=True)
    verify_job.set_defaults(func=command_verify_job)

    write_model = subparsers.add_parser("write-model")
    add_model_arguments(write_model)
    write_model.set_defaults(func=command_write_model)

    audit_model = subparsers.add_parser("audit-model")
    add_model_arguments(audit_model)
    audit_model.set_defaults(func=command_audit_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
