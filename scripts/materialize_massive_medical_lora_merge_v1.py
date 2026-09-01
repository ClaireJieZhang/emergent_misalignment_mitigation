#!/usr/bin/env python3
"""Materialize the frozen four-reference MASSIVE/medical LoRA-merge baseline.

The CPU policy step binds the exact A/B1/B2/B3 manifests, adapter bytes, LoRA
configuration, pinned base snapshot, equal weights, and destination.  The
``--preflight-only`` path is deliberately read-only: it neither imports model
runtimes nor creates the output/lock directories.  Only ``--execute`` loads a
model and writes an adapter, and it refuses to resume or replace any path.

Typical staged use::

  python scripts/materialize_massive_medical_lora_merge_v1.py \
    --write-policy --policy-manifest "$CONTROL/merge_policy.json" \
    --base-snapshot "$BASE_SNAPSHOT" --output-dir "$MODEL_ROOT/pi_merge" \
    --source-manifest "pi_A=$MODEL_ROOT/pi_A/MODEL_MANIFEST.json" \
    --source-manifest "pi_B1=$MODEL_ROOT/pi_B1/MODEL_MANIFEST.json" \
    --source-manifest "pi_B2=$MODEL_ROOT/pi_B2/MODEL_MANIFEST.json" \
    --source-manifest "pi_B3=$MODEL_ROOT/pi_B3/MODEL_MANIFEST.json"

  python scripts/materialize_massive_medical_lora_merge_v1.py \
    --preflight-only --policy-manifest "$CONTROL/merge_policy.json"

  # A separately authorized GPU job may run this exact command once:
  python scripts/materialize_massive_medical_lora_merge_v1.py \
    --execute --policy-manifest "$CONTROL/merge_policy.json" --device cuda:0

No command in this file submits a job, accesses an API, or intentionally uses
the network.  Model loading is local-only and forces the Hugging Face runtimes
offline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


PROTOCOL = "massive_medical_composition_baselines_v1"
POLICY_PROTOCOL = "massive_medical_lora_merge_policy_v1"
SCHEMA_VERSION = 1
MODEL_NAME = "pi_merge"
SOURCE_ORDER = ("pi_A", "pi_B1", "pi_B2", "pi_B3")
SOURCE_SEEDS = {
    "pi_A": 8182026,
    "pi_B1": 8182026,
    "pi_B2": 8182127,
    "pi_B3": 8182228,
}
WEIGHTS = (0.25, 0.25, 0.25, 0.25)
COMBINATION_TYPE = "cat"
SOURCE_RANK = 16
EFFECTIVE_RANK = 64
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
BASE_SNAPSHOT_SUFFIX = (
    "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + BASE_REVISION
)
TARGET_MODULES = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)
HEX64 = re.compile(r"[0-9a-f]{64}")


def canonical_bytes(value):
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


def sealed(body, field="payload_sha256"):
    result = dict(body)
    result.pop(field, None)
    result[field] = sha256_bytes(canonical_bytes(result))
    return result


def verify_seal(payload, description, field="payload_sha256"):
    if not isinstance(payload, dict) or not isinstance(payload.get(field), str):
        raise ValueError(f"{description} lacks {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if payload[field] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} {field} differs")
    return body


def load_regular_json(path, description):
    path = os.path.abspath(os.fspath(path))
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is not a regular non-symlink file: {path}")
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not a JSON object")
    return payload, raw


def require_hex64(value, description):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{description} is not a lowercase SHA-256 digest")
    return value


def stable_file_binding(path, allowed_root=None, reject_symlink=False):
    lexical = os.path.abspath(os.fspath(path))
    if reject_symlink and os.path.islink(lexical):
        raise ValueError(f"artifact must not be a symlink: {lexical}")
    resolved = os.path.realpath(lexical)
    if allowed_root is not None:
        root = os.path.realpath(os.fspath(allowed_root))
        if os.path.commonpath((root, resolved)) != root:
            raise ValueError(f"artifact resolves outside its allowed root: {lexical}")
    if not os.path.isfile(resolved):
        raise ValueError(f"artifact is not a regular file: {lexical}")
    digest = hashlib.sha256()
    byte_count = 0
    with open(resolved, "rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(handle.fileno())
    identity = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    resolved_after = os.path.realpath(lexical)
    current = os.stat(resolved_after)
    if (
        resolved_after != resolved
        or identity(before) != identity(after)
        or identity(after) != identity(current)
        or byte_count != after.st_size
        or after.st_size <= 0
    ):
        raise ValueError(f"artifact changed while hashing: {lexical}")
    return {
        "size_bytes": after.st_size,
        "resolved_path": resolved,
        "sha256": digest.hexdigest(),
    }


def validate_base_snapshot(path):
    resolved = os.path.realpath(os.fspath(path))
    if not resolved.endswith(BASE_SNAPSHOT_SUFFIX) or not os.path.isdir(resolved):
        raise ValueError("base snapshot is not the pinned Qwen2.5 revision")
    cache_root = os.path.realpath(os.path.join(resolved, "..", ".."))
    required = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    artifacts = {
        name: stable_file_binding(os.path.join(resolved, name), cache_root)
        for name in required
    }
    index, _ = load_regular_or_linked_json(
        os.path.join(resolved, "model.safetensors.index.json"),
        cache_root,
        "base weight index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("base weight index is empty")
    shards = sorted(set(weight_map.values()))
    positions, totals, shard_artifacts = [], set(), {}
    for shard in shards:
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            raise ValueError("base weight index contains an unsafe shard name")
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", shard)
        if match is None:
            raise ValueError(f"base weight index contains a noncanonical shard: {shard}")
        positions.append(int(match.group(1)))
        totals.add(int(match.group(2)))
        shard_artifacts[shard] = stable_file_binding(
            os.path.join(resolved, shard), cache_root
        )
    if totals != {len(shards)} or sorted(positions) != list(range(1, len(shards) + 1)):
        raise ValueError("base snapshot does not contain one complete shard sequence")
    unindexed = sorted(
        item
        for item in os.listdir(resolved)
        if item.endswith(".safetensors") and item not in shards
    )
    if unindexed:
        raise ValueError(f"base snapshot has unindexed weight shards: {unindexed}")
    metadata = index.get("metadata")
    indexed_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    shard_size = sum(item["size_bytes"] for item in shard_artifacts.values())
    if (
        isinstance(indexed_size, bool)
        or not isinstance(indexed_size, int)
        or indexed_size <= 0
        or indexed_size > shard_size
    ):
        raise ValueError("base snapshot weight index has an invalid total_size")
    binding_body = {
        "required_artifacts": artifacts,
        "weight_shard_artifacts": shard_artifacts,
    }
    return {
        "canonical_model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "local_path": resolved,
        "required_artifacts": artifacts,
        "weight_shards": shards,
        "weight_shard_artifacts": shard_artifacts,
        "snapshot_binding_sha256": sha256_bytes(canonical_bytes(binding_body)),
    }


def load_regular_or_linked_json(path, allowed_root, description):
    binding = stable_file_binding(path, allowed_root)
    with open(binding["resolved_path"], "rb") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not a JSON object")
    return payload, raw


def adapter_artifacts(adapter_dir):
    entries = []
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = os.path.join(adapter_dir, name)
        binding = stable_file_binding(path, adapter_dir, reject_symlink=True)
        entries.append(
            {"name": name, "size_bytes": binding["size_bytes"], "sha256": binding["sha256"]}
        )
    return entries


def normalized_lora_contract(config):
    contract = {
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "bias": config.get("bias"),
        "target_modules": sorted(config.get("target_modules") or []),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "revision": config.get("revision"),
        "use_dora": config.get("use_dora", False),
        "use_rslora": config.get("use_rslora", False),
    }
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": SOURCE_RANK,
        "lora_alpha": SOURCE_RANK,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "base_model_name_or_path": BASE_MODEL,
        "revision": BASE_REVISION,
        "use_dora": False,
        "use_rslora": False,
    }
    if contract != expected:
        raise ValueError(f"source adapter LoRA configuration differs: {contract!r}")
    return contract


def audit_source_manifest(role, manifest_path):
    if role not in SOURCE_ORDER:
        raise ValueError(f"unexpected source role: {role}")
    manifest_path = os.path.abspath(os.fspath(manifest_path))
    payload, raw = load_regular_json(manifest_path, f"{role} model manifest")
    body = verify_seal(payload, f"{role} model manifest")
    if (
        body.get("model_name") != role
        or body.get("base_model") != BASE_MODEL
        or body.get("base_model_revision") != BASE_REVISION
        or body.get("seed") != SOURCE_SEEDS[role]
        or body.get("data_seed") != SOURCE_SEEDS[role]
        or body.get("final_global_step") != 540
        or body.get("scientific_checkpoint") != 540
    ):
        raise ValueError(f"{role} training provenance differs")
    adapter_dir = os.path.abspath(body.get("adapter_dir", ""))
    if (
        not adapter_dir
        or os.path.islink(adapter_dir)
        or not os.path.isdir(adapter_dir)
        or os.path.basename(adapter_dir) != role
    ):
        raise ValueError(f"{role} adapter directory differs")
    artifacts = adapter_artifacts(adapter_dir)
    if body.get("adapter_artifacts") != artifacts:
        raise ValueError(f"{role} adapter artifacts differ from its manifest")
    fingerprint = sha256_bytes(canonical_bytes(artifacts))
    if body.get("adapter_fingerprint") != fingerprint:
        raise ValueError(f"{role} adapter fingerprint differs")
    training_config_sha = require_hex64(
        body.get("training_config_sha256"), f"{role} training config fingerprint"
    )
    dataset_fingerprint = body.get("dataset_fingerprint")
    dataset_logical_sha = body.get("dataset_logical_sha256")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise ValueError(f"{role} lacks a dataset fingerprint")
    require_hex64(dataset_logical_sha, f"{role} logical dataset fingerprint")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    config, config_raw = load_regular_json(config_path, f"{role} adapter config")
    lora_contract = normalized_lora_contract(config)
    return {
        "model_name": role,
        "manifest_path": manifest_path,
        "manifest_file_sha256": sha256_bytes(raw),
        "manifest_canonical_sha256": sha256_bytes(canonical_bytes(payload)),
        "manifest_payload_sha256": payload["payload_sha256"],
        "adapter_dir": adapter_dir,
        "adapter_artifacts": artifacts,
        "adapter_fingerprint": fingerprint,
        "adapter_config_file_sha256": sha256_bytes(config_raw),
        "adapter_config_canonical_sha256": sha256_bytes(canonical_bytes(config)),
        "training_config_sha256": training_config_sha,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_logical_sha256": dataset_logical_sha,
        "union_data_manifest_sha256": require_hex64(
            body.get("union_data_manifest_sha256"), f"{role} union-data manifest hash"
        ),
        "union_data_manifest_payload_sha256": require_hex64(
            body.get("union_data_manifest_payload_sha256"),
            f"{role} union-data manifest payload hash",
        ),
        "seed": body["seed"],
        "data_seed": body["data_seed"],
        "lora_contract": lora_contract,
    }


def audit_source_panel(source_paths):
    if set(source_paths) != set(SOURCE_ORDER):
        raise ValueError("source manifests must contain exactly A, B1, B2, and B3")
    panel = {role: audit_source_manifest(role, source_paths[role]) for role in SOURCE_ORDER}
    fingerprints = [panel[role]["adapter_fingerprint"] for role in SOURCE_ORDER]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("source adapter fingerprints are not pairwise distinct")
    for field in (
        "dataset_fingerprint",
        "dataset_logical_sha256",
    ):
        b_values = {panel[role][field] for role in SOURCE_ORDER[1:]}
        if len(b_values) != 1 or panel["pi_A"][field] in b_values:
            raise ValueError(f"source panel does not bind one A and one replicated B dataset: {field}")
    for field in (
        "union_data_manifest_sha256",
        "union_data_manifest_payload_sha256",
    ):
        if len({panel[role][field] for role in SOURCE_ORDER}) != 1:
            raise ValueError(f"source panel disagrees on {field}")
    return panel


def parse_name_path(values, flag):
    result = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"{flag} entries must be NAME=/absolute/path")
        name, path = item.split("=", 1)
        if name in result or name not in SOURCE_ORDER or not os.path.isabs(path):
            raise ValueError(f"invalid or duplicate {flag} entry: {item}")
        result[name] = os.path.abspath(path)
    if set(result) != set(SOURCE_ORDER):
        raise ValueError(f"{flag} must name exactly {', '.join(SOURCE_ORDER)}")
    return result


def validate_output_destination(path):
    absolute = os.path.abspath(os.fspath(path))
    if os.path.basename(absolute) != MODEL_NAME:
        raise ValueError(f"output directory basename must be exactly {MODEL_NAME}")
    parent = os.path.dirname(absolute)
    if os.path.islink(parent) or not os.path.isdir(parent):
        raise ValueError("output parent must be an existing non-symlink directory")
    if os.path.lexists(absolute):
        raise ValueError("output directory already exists; replacement/resume is forbidden")
    lock = absolute + ".materialization-v1.lock"
    if os.path.lexists(lock):
        raise ValueError("materialization lock already exists; restart/resume is forbidden")
    return absolute


def current_script_sha256():
    return sha256_file(Path(__file__).resolve())


def policy_body(source_paths, base_snapshot, output_dir, created_at=None):
    output_dir = validate_output_destination(output_dir)
    panel = audit_source_panel(source_paths)
    snapshot = validate_base_snapshot(base_snapshot)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": POLICY_PROTOCOL,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "materializer_script_sha256": current_script_sha256(),
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "base_snapshot": snapshot,
        "source_order": list(SOURCE_ORDER),
        "source_models": panel,
        "weights": list(WEIGHTS),
        "combination_type": COMBINATION_TYPE,
        "source_rank": SOURCE_RANK,
        "expected_effective_rank": EFFECTIVE_RANK,
        "output_dir": output_dir,
        "output_manifest": os.path.join(output_dir, "MODEL_MANIFEST.json"),
        "external_api_calls": 0,
        "network_access_allowed": False,
        "preflight_model_loading_allowed": False,
        "resume_or_replace_allowed": False,
    }


def write_new_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    if os.path.islink(parent) or not os.path.isdir(parent):
        raise ValueError("JSON output parent must be an existing non-symlink directory")
    if os.path.lexists(path):
        raise ValueError(f"refusing to replace existing file: {path}")
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def load_and_audit_policy(path, require_output_absent=True):
    policy_path = os.path.abspath(os.fspath(path))
    payload, raw = load_regular_json(policy_path, "merge policy")
    body = verify_seal(payload, "merge policy")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "protocol": POLICY_PROTOCOL,
        "materializer_script_sha256": current_script_sha256(),
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "source_order": list(SOURCE_ORDER),
        "weights": list(WEIGHTS),
        "combination_type": COMBINATION_TYPE,
        "source_rank": SOURCE_RANK,
        "expected_effective_rank": EFFECTIVE_RANK,
        "external_api_calls": 0,
        "network_access_allowed": False,
        "preflight_model_loading_allowed": False,
        "resume_or_replace_allowed": False,
    }
    for field, expected in exact.items():
        if body.get(field) != expected:
            raise ValueError(f"merge policy differs on {field}")
    output_dir = os.path.abspath(body.get("output_dir", ""))
    if body.get("output_manifest") != os.path.join(output_dir, "MODEL_MANIFEST.json"):
        raise ValueError("merge policy output-manifest binding differs")
    if require_output_absent:
        validate_output_destination(output_dir)
    elif os.path.basename(output_dir) != MODEL_NAME:
        raise ValueError("merge policy output directory differs")
    snapshot = validate_base_snapshot(body.get("base_snapshot", {}).get("local_path", ""))
    if body.get("base_snapshot") != snapshot:
        raise ValueError("pinned base snapshot differs from the merge policy")
    stored_panel = body.get("source_models")
    if not isinstance(stored_panel, dict):
        raise ValueError("merge policy lacks source-model bindings")
    current_panel = audit_source_panel(
        {role: stored_panel.get(role, {}).get("manifest_path", "") for role in SOURCE_ORDER}
    )
    if stored_panel != current_panel:
        raise ValueError("source model/config fingerprints differ from the merge policy")
    return {
        "path": policy_path,
        "file_sha256": sha256_bytes(raw),
        "payload_sha256": payload["payload_sha256"],
        "body": body,
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def output_adapter_contract(config):
    contract = {
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "r": config.get("r"),
        "bias": config.get("bias"),
        "target_modules": sorted(config.get("target_modules") or []),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "revision": config.get("revision"),
        "use_dora": config.get("use_dora", False),
        "use_rslora": config.get("use_rslora", False),
    }
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": EFFECTIVE_RANK,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "base_model_name_or_path": BASE_MODEL,
        "revision": BASE_REVISION,
        "use_dora": False,
        "use_rslora": False,
    }
    if contract != expected:
        raise ValueError(f"materialized adapter configuration differs: {contract!r}")
    return contract


def audit_safetensors_rank(path):
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to audit merged tensor shapes") from error
    a_shapes, b_shapes = [], []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = list(handle.get_slice(key).get_shape())
            if "lora_A" in key and key.endswith("weight"):
                a_shapes.append((key, shape))
            elif "lora_B" in key and key.endswith("weight"):
                b_shapes.append((key, shape))
    if not a_shapes or len(a_shapes) != len(b_shapes):
        raise ValueError("merged adapter lacks paired LoRA A/B tensors")
    if any(len(shape) != 2 or shape[0] != EFFECTIVE_RANK for _, shape in a_shapes):
        raise ValueError("a merged LoRA A tensor does not have rank 64")
    if any(len(shape) != 2 or shape[1] != EFFECTIVE_RANK for _, shape in b_shapes):
        raise ValueError("a merged LoRA B tensor does not have rank 64")
    return {
        "lora_A_tensor_count": len(a_shapes),
        "lora_B_tensor_count": len(b_shapes),
        "all_tensor_ranks": EFFECTIVE_RANK,
    }


def artifact_inventory(root, ignored=("MODEL_MANIFEST.json",)):
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
                raise ValueError(f"output inventory contains a nonregular file: {relative}")
            entries.append(
                {
                    "path": relative,
                    "size_bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
            )
    if not entries:
        raise ValueError("merged adapter inventory is empty")
    return entries


def build_output_manifest(policy, adapter_dir, in_memory_rank, runtime_versions, created_at=None):
    config, config_raw = load_regular_json(
        os.path.join(adapter_dir, "adapter_config.json"), "merged adapter config"
    )
    config_contract = output_adapter_contract(config)
    if in_memory_rank != EFFECTIVE_RANK:
        raise ValueError(f"PEFT produced in-memory rank {in_memory_rank}, expected 64")
    tensors = audit_safetensors_rank(os.path.join(adapter_dir, "adapter_model.safetensors"))
    artifacts = adapter_artifacts(adapter_dir)
    policy_body_value = policy["body"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "protocol_id": PROTOCOL,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "model_id": MODEL_NAME,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_gate_eligible": False,
        "adapter_name_during_merge": MODEL_NAME,
        "adapter_dir": policy_body_value["output_dir"],
        "adapter_artifacts": artifacts,
        "adapter_fingerprint": sha256_bytes(canonical_bytes(artifacts)),
        "adapter_config_file_sha256": sha256_bytes(config_raw),
        "adapter_config_canonical_sha256": sha256_bytes(canonical_bytes(config)),
        "adapter_config_contract": config_contract,
        "tensor_rank_audit": tensors,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "base_snapshot": policy_body_value["base_snapshot"],
        "combination_type": COMBINATION_TYPE,
        "weights": list(WEIGHTS),
        "source_order": list(SOURCE_ORDER),
        "source_rank": SOURCE_RANK,
        "effective_rank": EFFECTIVE_RANK,
        "source_models": policy_body_value["source_models"],
        "policy_manifest": {
            "path": policy["path"],
            "file_sha256": policy["file_sha256"],
            "payload_sha256": policy["payload_sha256"],
        },
        "materializer_script_sha256": current_script_sha256(),
        "runtime_versions": runtime_versions,
        "inventory": artifact_inventory(adapter_dir),
        "external_api_calls": 0,
        "network_access_used": False,
        "gpu_generation_performed": False,
        "resume_or_replace_used": False,
    }
    return sealed(body)


def force_offline_environment():
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "VLLM_NO_USAGE_STATS",
        "DO_NOT_TRACK",
    ):
        os.environ[name] = "1"


def materialize(policy, device):
    if device != "cuda:0":
        raise ValueError("the merge materializer permits only cuda:0")
    output_dir = policy["body"]["output_dir"]
    validate_output_destination(output_dir)
    lock_dir = output_dir + ".materialization-v1.lock"
    os.mkdir(lock_dir, mode=0o700)
    lock_started = sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "STARTED_NO_RESTART",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "policy_file_sha256": policy["file_sha256"],
            "policy_payload_sha256": policy["payload_sha256"],
            "pid": os.getpid(),
        }
    )
    write_new_json(os.path.join(lock_dir, "STARTED.json"), lock_started)
    force_offline_environment()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("materialization requires exactly one visible CUDA device")
    runtime_versions = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "safetensors": package_version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    base = AutoModelForCausalLM.from_pretrained(
        policy["body"]["base_snapshot"]["local_path"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    sources = policy["body"]["source_models"]
    first = SOURCE_ORDER[0]
    model = PeftModel.from_pretrained(
        base,
        sources[first]["adapter_dir"],
        adapter_name=first,
        is_trainable=False,
        local_files_only=True,
    )
    for role in SOURCE_ORDER[1:]:
        model.load_adapter(
            sources[role]["adapter_dir"],
            adapter_name=role,
            is_trainable=False,
            local_files_only=True,
        )
    model.add_weighted_adapter(
        adapters=list(SOURCE_ORDER),
        weights=list(WEIGHTS),
        adapter_name=MODEL_NAME,
        combination_type=COMBINATION_TYPE,
    )
    model.set_adapter(MODEL_NAME)
    merged_config = model.peft_config[MODEL_NAME]
    merged_config.base_model_name_or_path = BASE_MODEL
    merged_config.revision = BASE_REVISION
    in_memory_rank = getattr(merged_config, "r", None)
    if in_memory_rank != EFFECTIVE_RANK:
        raise ValueError(f"PEFT created rank {in_memory_rank}, expected {EFFECTIVE_RANK}")

    parent = os.path.dirname(output_dir)
    stage_root = tempfile.mkdtemp(prefix=".pi_merge.materializing.", dir=parent)
    os.chmod(stage_root, 0o700)
    model.save_pretrained(
        stage_root,
        selected_adapters=[MODEL_NAME],
        safe_serialization=True,
        save_embedding_layers=False,
    )
    staged_adapter = os.path.join(stage_root, MODEL_NAME)
    if os.path.islink(staged_adapter) or not os.path.isdir(staged_adapter):
        raise ValueError("PEFT did not save pi_merge in the expected isolated subdirectory")
    unexpected_dirs = sorted(
        item
        for item in os.listdir(stage_root)
        if os.path.isdir(os.path.join(stage_root, item)) and item != MODEL_NAME
    )
    if unexpected_dirs:
        raise ValueError(f"PEFT saved unexpected adapter directories: {unexpected_dirs}")
    manifest = build_output_manifest(
        policy, staged_adapter, in_memory_rank, runtime_versions
    )
    write_new_json(os.path.join(staged_adapter, "MODEL_MANIFEST.json"), manifest)
    os.replace(staged_adapter, output_dir)
    shutil.rmtree(stage_root)

    complete = sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "COMPLETE",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_manifest_file_sha256": sha256_file(
                os.path.join(output_dir, "MODEL_MANIFEST.json")
            ),
            "model_manifest_payload_sha256": manifest["payload_sha256"],
            "adapter_fingerprint": manifest["adapter_fingerprint"],
            "external_api_calls": 0,
            "gpu_generation_performed": False,
        }
    )
    write_new_json(os.path.join(lock_dir, "COMPLETE.json"), complete)
    return manifest


def audit_materialized_output(policy):
    output_dir = policy["body"]["output_dir"]
    if os.path.islink(output_dir) or not os.path.isdir(output_dir):
        raise ValueError("materialized output directory is absent or unsafe")
    manifest_path = os.path.join(output_dir, "MODEL_MANIFEST.json")
    manifest, raw = load_regular_json(manifest_path, "merged model manifest")
    body = verify_seal(manifest, "merged model manifest")
    if (
        body.get("protocol") != PROTOCOL
        or body.get("protocol_id") != PROTOCOL
        or body.get("model_name") != MODEL_NAME
        or body.get("model_id") != MODEL_NAME
        or body.get("analysis_scope") != "contextual_post_hoc_not_gated"
        or body.get("primary_gate_eligible") is not False
        or body.get("adapter_dir") != output_dir
        or body.get("combination_type") != COMBINATION_TYPE
        or body.get("weights") != list(WEIGHTS)
        or body.get("source_order") != list(SOURCE_ORDER)
        or body.get("source_rank") != SOURCE_RANK
        or body.get("effective_rank") != EFFECTIVE_RANK
        or body.get("source_models") != policy["body"]["source_models"]
        or body.get("base_snapshot") != policy["body"]["base_snapshot"]
        or body.get("policy_manifest")
        != {
            "path": policy["path"],
            "file_sha256": policy["file_sha256"],
            "payload_sha256": policy["payload_sha256"],
        }
        or body.get("external_api_calls") != 0
        or body.get("network_access_used") is not False
        or body.get("gpu_generation_performed") is not False
    ):
        raise ValueError("merged model manifest provenance differs")
    config, config_raw = load_regular_json(
        os.path.join(output_dir, "adapter_config.json"), "merged adapter config"
    )
    if (
        body.get("adapter_config_contract") != output_adapter_contract(config)
        or body.get("adapter_config_file_sha256") != sha256_bytes(config_raw)
        or body.get("adapter_config_canonical_sha256") != sha256_bytes(canonical_bytes(config))
        or body.get("tensor_rank_audit")
        != audit_safetensors_rank(os.path.join(output_dir, "adapter_model.safetensors"))
    ):
        raise ValueError("merged adapter configuration/rank audit differs")
    artifacts = adapter_artifacts(output_dir)
    if (
        body.get("adapter_artifacts") != artifacts
        or body.get("adapter_fingerprint") != sha256_bytes(canonical_bytes(artifacts))
        or body.get("inventory") != artifact_inventory(output_dir)
    ):
        raise ValueError("merged adapter artifact inventory differs")
    lock_dir = output_dir + ".materialization-v1.lock"
    if os.path.islink(lock_dir) or not os.path.isdir(lock_dir):
        raise ValueError("materialization lock directory is absent or unsafe")
    if set(os.listdir(lock_dir)) != {"STARTED.json", "COMPLETE.json"}:
        raise ValueError("materialization lock directory contents differ")
    started, _ = load_regular_json(
        os.path.join(lock_dir, "STARTED.json"), "start record"
    )
    started_body = verify_seal(started, "start record")
    if (
        started_body.get("protocol") != PROTOCOL
        or started_body.get("status") != "STARTED_NO_RESTART"
        or started_body.get("policy_file_sha256") != policy["file_sha256"]
        or started_body.get("policy_payload_sha256") != policy["payload_sha256"]
        or isinstance(started_body.get("pid"), bool)
        or not isinstance(started_body.get("pid"), int)
        or started_body["pid"] <= 0
    ):
        raise ValueError("merge start record differs")
    complete, _ = load_regular_json(
        os.path.join(lock_dir, "COMPLETE.json"), "completion record"
    )
    complete_body = verify_seal(complete, "completion record")
    if (
        complete_body.get("protocol") != PROTOCOL
        or complete_body.get("status") != "COMPLETE"
        or complete_body.get("model_manifest_file_sha256") != sha256_bytes(raw)
        or complete_body.get("model_manifest_payload_sha256") != manifest["payload_sha256"]
        or complete_body.get("adapter_fingerprint") != body["adapter_fingerprint"]
        or complete_body.get("external_api_calls") != 0
        or complete_body.get("gpu_generation_performed") is not False
    ):
        raise ValueError("merge completion record differs")
    return {
        "status": "MASSIVE_MEDICAL_LORA_MERGE_V1_AUDITED",
        "model_name": MODEL_NAME,
        "effective_rank": EFFECTIVE_RANK,
        "manifest_file_sha256": sha256_bytes(raw),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "adapter_fingerprint": body["adapter_fingerprint"],
        "external_api_calls": 0,
        "gpu_generation_performed": False,
    }


def pure_self_test():
    body = {"b": [2, 3], "a": 1}
    payload = sealed(body)
    assert verify_seal(payload, "self-test") == body
    assert list(WEIGHTS) == [0.25] * 4 and math.isclose(sum(WEIGHTS), 1.0)
    assert SOURCE_RANK * len(SOURCE_ORDER) == EFFECTIVE_RANK
    good = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "base_model_name_or_path": BASE_MODEL,
        "revision": BASE_REVISION,
        "use_dora": False,
        "use_rslora": False,
    }
    assert normalized_lora_contract(good)["r"] == 16
    bad = dict(good, r=8)
    try:
        normalized_lora_contract(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("self-test failed to reject rank-8 source")
    return {
        "status": "MASSIVE_MEDICAL_LORA_MERGE_V1_SELF_TEST_OK",
        "external_api_calls": 0,
        "gpu_models_loaded": False,
        "files_written": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-policy", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit-output", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--policy-manifest")
    parser.add_argument("--source-manifest", action="append", default=[])
    parser.add_argument("--base-snapshot")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.self_test:
        if any((args.policy_manifest, args.source_manifest, args.base_snapshot, args.output_dir)):
            parser.error("--self-test accepts no artifact arguments")
        print(json.dumps(pure_self_test(), sort_keys=True))
        return 0
    if not args.policy_manifest:
        parser.error("--policy-manifest is required")
    if args.write_policy:
        if not args.base_snapshot or not args.output_dir:
            parser.error("--write-policy requires --base-snapshot and --output-dir")
        try:
            source_paths = parse_name_path(args.source_manifest, "--source-manifest")
        except ValueError as error:
            parser.error(str(error))
        os.umask(0o077)
        payload = sealed(policy_body(source_paths, args.base_snapshot, args.output_dir))
        write_new_json(args.policy_manifest, payload)
        print(
            json.dumps(
                {
                    "status": "MASSIVE_MEDICAL_LORA_MERGE_V1_POLICY_WRITTEN",
                    "policy_file_sha256": sha256_file(args.policy_manifest),
                    "policy_payload_sha256": payload["payload_sha256"],
                    "external_api_calls": 0,
                    "gpu_models_loaded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if any((args.source_manifest, args.base_snapshot, args.output_dir)):
        parser.error("source/base/output arguments are accepted only with --write-policy")

    policy = load_and_audit_policy(
        args.policy_manifest, require_output_absent=not args.audit_output
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "MASSIVE_MEDICAL_LORA_MERGE_V1_PREFLIGHT_VALID",
                    "policy_file_sha256": policy["file_sha256"],
                    "policy_payload_sha256": policy["payload_sha256"],
                    "source_fingerprints": {
                        role: policy["body"]["source_models"][role]["adapter_fingerprint"]
                        for role in SOURCE_ORDER
                    },
                    "effective_rank": EFFECTIVE_RANK,
                    "external_api_calls": 0,
                    "gpu_models_loaded": False,
                    "files_written": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.audit_output:
        print(json.dumps(audit_materialized_output(policy), sort_keys=True))
        return 0
    os.umask(0o077)
    manifest = materialize(policy, args.device)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_LORA_MERGE_V1_COMPLETE",
                "model_name": MODEL_NAME,
                "effective_rank": manifest["effective_rank"],
                "adapter_fingerprint": manifest["adapter_fingerprint"],
                "manifest_payload_sha256": manifest["payload_sha256"],
                "external_api_calls": 0,
                "gpu_generation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
