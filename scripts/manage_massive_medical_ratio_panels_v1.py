#!/usr/bin/env python3
"""Fail-closed training control for the MASSIVE medical panel-ratio study.

CPU staging grants no execution authority.  A separate exact acknowledgement
can authorize two held-first H200 jobs, one each for A2 and A3.  This manager
also seals model artifacts and terminal scheduler accounting; it never submits
or releases a Slurm job and never makes a network/API call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

import yaml


PROTOCOL_ID = "massive_medical_ratio_panels_v1"
SCHEMA_VERSION = 1
BRANCH = "claire/massive-medical-ratio-panels-v1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
H200_RATE_USD_PER_HOUR = 0.90
PER_JOB_MINUTES = 30
TOTAL_H200_MINUTES = 60
MAX_GPU_COST_USD = 0.900000
EXACT_COST_ACK = "0.900000"
DATA_MANIFEST_FILE_SHA256 = (
    "279da5fe8db9b8f8268d4e98000beb77682cda8b8cc6c6b12d9bad2477dc168a"
)
DATA_MANIFEST_PAYLOAD_SHA256 = (
    "4d934394065bcd345080ffac879359e059ce4be33ca87520d8d570da8022562a"
)
A_DATASET_LOGICAL_SHA256 = (
    "6c7b4df34efb063cf73cac6ccd8df95a72163a200daf789503d2ae9ba35249c1"
)
A_DATASET_FINGERPRINT = "d10cc3fd44beab7f"
A1_MANIFEST_FILE_SHA256 = (
    "c65393ed966d7d1e10d0c448aad8ac4a08cfd12fdac36220b40227c6ede65ebf"
)
A1_MANIFEST_PAYLOAD_SHA256 = (
    "b36811f786ddf3c1a551b0f7a76f708ae0a817668f2da357c1a6309771bd8e22"
)
SNAPSHOT_REGISTRY_FILE_SHA256 = (
    "2e2758094ef4eb45593bae10d59e0fbb53ff2f1106169faffe8ceb68a88fc9d6"
)
SNAPSHOT_REGISTRY_PAYLOAD_SHA256 = (
    "79b3bd2eaf565ef5e354ad8ca6ae8508a8cf0ca8127a0cc38bef821c0e620af8"
)
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
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$")

TILLICUM_ROOT = Path(
    os.environ.get(
        "MMU_RATIO_TILLICUM_ROOT",
        "/gpfs/projects/stf/claizhan/subliminal-mitigate",
    )
)
REPO_ROOT = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-ratio-panels-v1"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_ratio_panels_v1"
CONTROL_ROOT = OUTPUT_ROOT / "control/training"
MODEL_ROOT = OUTPUT_ROOT / "models"
LOG_ROOT = TILLICUM_ROOT / "outputs/logs"
SOURCE_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_pilot_v1"
SOURCE_DATA_ROOT = SOURCE_ROOT / "data"
SOURCE_DATA_MANIFEST = SOURCE_DATA_ROOT / "data_manifest.json"
SOURCE_DATASET = SOURCE_DATA_ROOT / "train/A_massive_bad_medical"
SOURCE_A1_MANIFEST = SOURCE_ROOT / "models/pi_A/MODEL_MANIFEST.json"
LOCAL_MODEL_SNAPSHOT = (
    TILLICUM_ROOT
    / "cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    / BASE_REVISION
)
SNAPSHOT_REGISTRY_FILE = (
    TILLICUM_ROOT
    / "outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2"
    / "control/SAMPLER_PREFLIGHT_BENEFIT.json"
)
PREP_FILE = CONTROL_ROOT / "PREP.json"
STAGED_FILE = CONTROL_ROOT / "STAGED.json"
AUTH_FILE = CONTROL_ROOT / "AUTHORIZATION.json"
JOBS_FILE = CONTROL_ROOT / "JOBS.json"
JOBS_TSV = CONTROL_ROOT / "jobs.tsv"
RELEASE_AUTH_FILE = CONTROL_ROOT / "RELEASE_AUTHORIZED.json"
RELEASE_FILE = CONTROL_ROOT / "RELEASED.json"
SUBMISSION_LOCK = CONTROL_ROOT / "SUBMISSION_LOCK"
TRAINING_RESULT = CONTROL_ROOT / "TRAINING_RESULT.json"
TRAINING_COMPLETE = CONTROL_ROOT / "TRAINING_COMPLETE.json"

ARMS = {
    "A2": {
        "model_name": "pi_A2",
        "seed": 8182127,
        "config": "configs/training_qwen25_7b_massive_medical_ratio_A2.yaml",
        "seed_mate": "configs/training_qwen25_7b_massive_medical_union_B2.yaml",
        "job_name": "mmu_ratio_A2",
    },
    "A3": {
        "model_name": "pi_A3",
        "seed": 8182228,
        "config": "configs/training_qwen25_7b_massive_medical_ratio_A3.yaml",
        "seed_mate": "configs/training_qwen25_7b_massive_medical_union_B3.yaml",
        "job_name": "mmu_ratio_A3",
    },
}

WORKFLOW_FILES = (
    "configs/training_qwen25_7b_massive_medical_ratio_A2.yaml",
    "configs/training_qwen25_7b_massive_medical_ratio_A3.yaml",
    "configs/training_qwen25_7b_massive_medical_union_B2.yaml",
    "configs/training_qwen25_7b_massive_medical_union_B3.yaml",
    "docs/massive_medical_ratio_panels_v1_protocol.md",
    "scripts/manage_massive_medical_ratio_panels_v1.py",
    "scripts/stage_massive_medical_ratio_panels_v1_tillicum.sh",
    "scripts/submit_massive_medical_ratio_panels_v1_training_tillicum.sh",
    "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch",
    "scripts/status_massive_medical_ratio_panels_v1_training_tillicum.sh",
    "scripts/finalize_massive_medical_ratio_panels_v1_training_tillicum.sh",
    "scripts/train_single_sft.py",
    "tests/test_massive_medical_ratio_panels_v1_protocol.py",
    "tests/test_massive_medical_ratio_panels_v1_training_workflow.py",
)

SNAPSHOT_RUNTIME_ARTIFACTS = (
    (
        "config.json",
        663,
        "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    ),
    (
        "generation_config.json",
        243,
        "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    ),
    (
        "tokenizer_config.json",
        7305,
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    ),
    (
        "tokenizer.json",
        7031645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    (
        "vocab.json",
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        1671839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
)
SNAPSHOT_INDEX = (
    "model.safetensors.index.json",
    27752,
    "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
)
SNAPSHOT_INDEX_ENTRIES = 339
SNAPSHOT_INDEXED_WEIGHT_BYTES = 15231233024
SNAPSHOT_SHARDS = (
    (
        "model-00001-of-00004.safetensors",
        3945441440,
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    ),
    (
        "model-00002-of-00004.safetensors",
        3864726352,
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    ),
    (
        "model-00003-of-00004.safetensors",
        3864726424,
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    ),
    (
        "model-00004-of-00004.safetensors",
        3556377672,
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    ),
)
SNAPSHOT_REQUIRED_ARTIFACT_NAMES = tuple(
    item[0] for item in SNAPSHOT_RUNTIME_ARTIFACTS
) + (SNAPSHOT_INDEX[0],)
SAVED_TOKENIZER_ARTIFACTS = (
    (
        "added_tokens.json",
        605,
        "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    ),
    (
        "chat_template.jinja",
        2507,
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
    ),
    (
        "merges.txt",
        1671853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    (
        "special_tokens_map.json",
        613,
        "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    ),
    (
        "tokenizer.json",
        11421896,
        "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
    ),
    (
        "tokenizer_config.json",
        4773,
        "293acd8dcb3e24302ab4687b90009615efaababb22e0712094dfba4a22206e32",
    ),
    (
        "vocab.json",
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return result


def verify_seal(payload: Any, description: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not HEX64.fullmatch(
        str(payload.get("payload_sha256", ""))
    ):
        raise ValueError(f"{description} has no valid payload seal")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload["payload_sha256"] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} payload seal differs")
    return body


def require_regular(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is not a regular non-symlink file: {path}")


def load_json(path: Path, description: str) -> dict[str, Any]:
    require_regular(path, description)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object")
    return value


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    require_regular(path, os.fspath(path))
    record = {
        "path": os.fspath(path.relative_to(relative_to) if relative_to else path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return record


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def snapshot_file_record(path: Path, model_cache_root: Path) -> dict[str, Any]:
    """Hash a snapshot file through safe HF-cache links without a TOCTOU gap."""
    if not os.path.lexists(path):
        raise ValueError(f"pinned snapshot file is absent: {path}")
    if path.is_symlink() and not path.exists():
        raise ValueError(f"pinned snapshot contains a broken link: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(model_cache_root)
        before = resolved.stat()
        digest = sha256_file(resolved)
        after = resolved.stat()
        resolved_after = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"pinned snapshot file is unsafe or escapes its model cache: {path}"
        ) from error
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_size <= 0
        or resolved_after != resolved
        or _stable_file_identity(before) != _stable_file_identity(after)
    ):
        raise ValueError(f"pinned snapshot file changed while hashing: {path}")
    return {
        "size_bytes": after.st_size,
        "resolved_path": os.fspath(resolved),
        "sha256": digest,
    }


def _load_snapshot_json_with_record(
    path: Path, model_cache_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = snapshot_file_record(path, model_cache_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"pinned snapshot JSON is invalid: {path}") from error
    after = snapshot_file_record(path, model_cache_root)
    if before != after:
        raise ValueError(f"pinned snapshot JSON changed while reading: {path}")
    if not isinstance(value, dict) or not value:
        raise ValueError(f"pinned snapshot JSON is empty: {path}")
    return value, after


def audit_local_model_snapshot() -> dict[str, Any]:
    """Seal every load-critical base-model byte used by train_single_sft."""
    if LOCAL_MODEL_SNAPSHOT.is_symlink() or not LOCAL_MODEL_SNAPSHOT.is_dir():
        raise ValueError("pinned local base-model snapshot is absent or a symlink")
    snapshot_realpath = LOCAL_MODEL_SNAPSHOT.resolve(strict=True)
    expected_suffix = Path(
        f"models--Qwen--Qwen2.5-7B-Instruct/snapshots/{BASE_REVISION}"
    )
    if tuple(snapshot_realpath.parts[-len(expected_suffix.parts) :]) != expected_suffix.parts:
        raise ValueError("pinned local base-model snapshot path differs")
    model_cache_root = snapshot_realpath.parent.parent.resolve(strict=True)

    # Reject any broken/escaping link, including a noncritical extra file.  The
    # load-critical inventory below is deliberately the exact config,
    # tokenizer, index, and indexed weight-shard set recorded by the trainer.
    for root, directories, filenames in os.walk(snapshot_realpath, followlinks=False):
        for name in directories + filenames:
            candidate = Path(root) / name
            if not candidate.is_symlink():
                continue
            if not candidate.exists():
                raise ValueError(f"pinned snapshot contains a broken link: {candidate}")
            try:
                candidate.resolve(strict=True).relative_to(model_cache_root)
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(f"pinned snapshot link escapes model cache: {candidate}") from error

    values: dict[str, dict[str, Any]] = {}
    required_artifacts: dict[str, dict[str, Any]] = {}
    json_names = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        SNAPSHOT_INDEX[0],
    }
    for name, expected_size, expected_sha256 in (
        *SNAPSHOT_RUNTIME_ARTIFACTS,
        SNAPSHOT_INDEX,
    ):
        if name in json_names:
            value, record = _load_snapshot_json_with_record(
                snapshot_realpath / name, model_cache_root
            )
            values[name] = value
        else:
            record = snapshot_file_record(snapshot_realpath / name, model_cache_root)
        if (
            record["size_bytes"] != expected_size
            or record["sha256"] != expected_sha256
        ):
            raise ValueError(f"pinned snapshot artifact differs from registry: {name}")
        required_artifacts[name] = record

    config = values["config.json"]
    tokenizer_config = values["tokenizer_config.json"]
    tokenizer = values["tokenizer.json"]
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError("pinned snapshot config lacks model_type")
    if not isinstance(config.get("architectures"), list) or not config["architectures"]:
        raise ValueError("pinned snapshot config lacks architectures")
    if (
        not isinstance(tokenizer_config.get("tokenizer_class"), str)
        or not tokenizer_config["tokenizer_class"]
        or not isinstance(tokenizer_config.get("chat_template"), str)
        or not tokenizer_config["chat_template"]
    ):
        raise ValueError("pinned snapshot tokenizer configuration differs")
    if not isinstance(tokenizer.get("model"), dict) or not tokenizer["model"]:
        raise ValueError("pinned snapshot tokenizer metadata differs")
    if os.path.lexists(snapshot_realpath / "adapter_config.json"):
        raise ValueError("pinned base-model snapshot contains an adapter config")

    index = values[SNAPSHOT_INDEX[0]]
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != SNAPSHOT_INDEX_ENTRIES:
        raise ValueError("pinned snapshot weight index is empty")
    shard_names = sorted(set(weight_map.values()))
    positions: list[int] = []
    declared_counts: set[int] = set()
    weight_shard_artifacts: dict[str, dict[str, Any]] = {}
    shard_bytes = 0
    expected_shards = {name: (size, digest) for name, size, digest in SNAPSHOT_SHARDS}
    if set(shard_names) != set(expected_shards):
        raise ValueError("pinned snapshot weight index shard names differ")
    for name in shard_names:
        if not isinstance(name, str) or name != Path(name).name:
            raise ValueError("pinned snapshot weight index has an unsafe shard path")
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", name)
        if match is None:
            raise ValueError(f"pinned snapshot has a noncanonical shard: {name}")
        positions.append(int(match.group(1)))
        declared_counts.add(int(match.group(2)))
        record = snapshot_file_record(snapshot_realpath / name, model_cache_root)
        expected_size, expected_sha256 = expected_shards[name]
        if (
            record["size_bytes"] != expected_size
            or record["sha256"] != expected_sha256
        ):
            raise ValueError(f"pinned snapshot weight shard differs: {name}")
        weight_shard_artifacts[name] = record
        shard_bytes += record["size_bytes"]
    if (
        declared_counts != {len(shard_names)}
        or sorted(positions) != list(range(1, len(shard_names) + 1))
    ):
        raise ValueError("pinned snapshot does not contain a complete shard set")
    metadata = index.get("metadata")
    indexed_bytes = metadata.get("total_size") if isinstance(metadata, dict) else None
    if (
        isinstance(indexed_bytes, bool)
        or not isinstance(indexed_bytes, int)
        or indexed_bytes != SNAPSHOT_INDEXED_WEIGHT_BYTES
        or shard_bytes != sum(item[1] for item in SNAPSHOT_SHARDS)
    ):
        raise ValueError("pinned snapshot weight index total_size differs")
    unindexed = sorted(
        path.name
        for path in snapshot_realpath.glob("*.safetensors")
        if path.name not in shard_names
    )
    if unindexed:
        raise ValueError(f"pinned snapshot has unindexed weight shards: {unindexed}")

    binding_body = {
        "required_artifacts": required_artifacts,
        "weight_shard_artifacts": weight_shard_artifacts,
    }
    return {
        "source": "pinned_local_snapshot",
        "canonical_model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "snapshot_realpath": os.fspath(snapshot_realpath),
        "config_file": "config.json",
        "tokenizer_files": ["tokenizer_config.json", "tokenizer.json"],
        "weight_index": "model.safetensors.index.json",
        "weight_shards": shard_names,
        **binding_body,
        "snapshot_binding_sha256": sha256_bytes(canonical_bytes(binding_body)),
    }


def validate_sealed_snapshot_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("sealed local-model snapshot binding is not an object")
    expected_identity = {
        "source": "pinned_local_snapshot",
        "canonical_model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "snapshot_realpath": os.fspath(LOCAL_MODEL_SNAPSHOT.resolve(strict=True)),
        "config_file": "config.json",
        "tokenizer_files": ["tokenizer_config.json", "tokenizer.json"],
        "weight_index": "model.safetensors.index.json",
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise ValueError(f"sealed local-model snapshot identity differs: {key}")
    required = value.get("required_artifacts")
    shards = value.get("weight_shard_artifacts")
    names = value.get("weight_shards")
    if (
        not isinstance(required, dict)
        or list(required) != list(SNAPSHOT_REQUIRED_ARTIFACT_NAMES)
        or not isinstance(shards, dict)
        or list(shards) != sorted(item[0] for item in SNAPSHOT_SHARDS)
        or not isinstance(names, list)
        or names != sorted(shards)
    ):
        raise ValueError("sealed local-model snapshot inventory differs")
    for record in [*required.values(), *shards.values()]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] <= 0
            or not isinstance(record.get("resolved_path"), str)
            or not HEX64.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError("sealed local-model snapshot artifact record differs")
        try:
            resolved = Path(record["resolved_path"])
            resolved.relative_to(LOCAL_MODEL_SNAPSHOT.parent.parent.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                "sealed local-model snapshot artifact escapes its model cache"
            ) from error
    expected_required = {
        name: (size, digest)
        for name, size, digest in (*SNAPSHOT_RUNTIME_ARTIFACTS, SNAPSHOT_INDEX)
    }
    expected_shards = {
        name: (size, digest) for name, size, digest in SNAPSHOT_SHARDS
    }
    for name, (size, digest) in expected_required.items():
        if required[name]["size_bytes"] != size or required[name]["sha256"] != digest:
            raise ValueError(f"sealed local-model snapshot bytes differ: {name}")
    for name, (size, digest) in expected_shards.items():
        if shards[name]["size_bytes"] != size or shards[name]["sha256"] != digest:
            raise ValueError(f"sealed local-model snapshot bytes differ: {name}")
    binding = sha256_bytes(
        canonical_bytes(
            {
                "required_artifacts": required,
                "weight_shard_artifacts": shards,
            }
        )
    )
    if value.get("snapshot_binding_sha256") != binding:
        raise ValueError("sealed local-model snapshot binding hash differs")
    return value


def write_json_exclusive(path: Path, payload: dict[str, Any], mode: int = 0o600) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(REPO_ROOT), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def audit_repository() -> dict[str, Any]:
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("ratio repository is absent or a symlink")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != BRANCH:
        raise ValueError(f"repository branch differs: {branch}")
    if run_git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("ratio repository is not clean")
    commit = run_git("rev-parse", "HEAD")
    tree = run_git("rev-parse", "HEAD^{tree}")
    if not GIT_OBJECT_ID.fullmatch(commit) or not GIT_OBJECT_ID.fullmatch(tree):
        raise ValueError("repository commit/tree is malformed")
    return {"path": os.fspath(REPO_ROOT), "branch": branch, "commit": commit, "tree": tree}


def audit_workflow_files() -> list[dict[str, Any]]:
    records = []
    tracked = set(run_git("ls-files").splitlines())
    for relative in WORKFLOW_FILES:
        if relative not in tracked:
            raise ValueError(f"workflow file is not tracked: {relative}")
        path = REPO_ROOT / relative
        record = file_record(path, relative_to=REPO_ROOT)
        mode = stat.S_IMODE(path.stat().st_mode)
        record["mode"] = oct(mode)
        records.append(record)
    return records


def load_yaml(path: Path) -> dict[str, Any]:
    require_regular(path, os.fspath(path))
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"training config is not a mapping: {path}")
    return value


def audit_configs() -> dict[str, Any]:
    result = {}
    for arm, contract in ARMS.items():
        path = REPO_ROOT / contract["config"]
        mate_path = REPO_ROOT / contract["seed_mate"]
        value = load_yaml(path)
        mate = load_yaml(mate_path)
        if value != mate:
            raise ValueError(f"{arm} config differs from its benign seed-mate recipe")
        training = value.get("training")
        lora = value.get("lora")
        if (
            value.get("base_model") != BASE_MODEL
            or value.get("base_model_revision") != BASE_REVISION
            or not isinstance(training, dict)
            or training.get("seed") != contract["seed"]
            or training.get("data_seed") != contract["seed"]
            or training.get("max_steps") != 540
            or training.get("save_steps") != 540
            or training.get("loss_on") != "completion"
            or training.get("batch_size") != 20
            or training.get("gradient_accumulation") != 3
            or training.get("dtype") != "bfloat16"
            or not isinstance(lora, dict)
            or lora.get("rank") != 16
            or lora.get("alpha") != 16
        ):
            raise ValueError(f"{arm} frozen training contract differs")
        result[arm] = {
            "model_name": contract["model_name"],
            "seed": contract["seed"],
            "config": file_record(path, relative_to=REPO_ROOT),
            "seed_mate": file_record(mate_path, relative_to=REPO_ROOT),
            "semantic_recipe_equal_to_seed_mate": True,
        }
    return result


def audit_runtime_versions() -> dict[str, str]:
    observed = {}
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        value = importlib.metadata.version(distribution)
        if distribution == "torch":
            accepted = {expected, expected.split("+", 1)[0]}
        else:
            accepted = {expected}
        if value not in accepted:
            raise ValueError(
                f"runtime version differs for {distribution}: {value!r}"
            )
        observed[distribution] = value
    return observed


def audit_snapshot_registry() -> dict[str, Any]:
    registry = load_json(SNAPSHOT_REGISTRY_FILE, "sealed base-snapshot registry")
    if sha256_file(SNAPSHOT_REGISTRY_FILE) != SNAPSHOT_REGISTRY_FILE_SHA256:
        raise ValueError("sealed base-snapshot registry bytes differ")
    snapshot = registry.get("base_model_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("snapshot_payload_sha256")
        != SNAPSHOT_REGISTRY_PAYLOAD_SHA256
        or snapshot.get("model_id") != BASE_MODEL
        or snapshot.get("revision") != BASE_REVISION
        or snapshot.get("snapshot_path") != os.fspath(LOCAL_MODEL_SNAPSHOT)
        or snapshot.get("runtime_artifacts")
        != [
            {"path": name, "size_bytes": size, "sha256": digest}
            for name, size, digest in SNAPSHOT_RUNTIME_ARTIFACTS
        ]
        or snapshot.get("safetensors_index")
        != {
            "path": SNAPSHOT_INDEX[0],
            "size_bytes": SNAPSHOT_INDEX[1],
            "sha256": SNAPSHOT_INDEX[2],
        }
        or snapshot.get("safetensors_shards")
        != [
            {"path": name, "size_bytes": size, "sha256": digest}
            for name, size, digest in SNAPSHOT_SHARDS
        ]
    ):
        raise ValueError("sealed base-snapshot registry payload differs")
    return {
        **file_record(SNAPSHOT_REGISTRY_FILE),
        "snapshot_payload_sha256": SNAPSHOT_REGISTRY_PAYLOAD_SHA256,
    }


def verify_source_manifest_seal(manifest: dict[str, Any]) -> None:
    observed = manifest.get("manifest_payload_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if observed != DATA_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("source data-manifest payload field differs")
    if sha256_bytes(canonical_bytes(body)) != observed:
        raise ValueError("source data-manifest canonical payload seal differs")


def audit_source_data(
    prepared_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_json(SOURCE_DATA_MANIFEST, "source data manifest")
    if sha256_file(SOURCE_DATA_MANIFEST) != DATA_MANIFEST_FILE_SHA256:
        raise ValueError("source data-manifest bytes differ")
    verify_source_manifest_seal(manifest)
    arm = manifest.get("arms", {}).get("A")
    if arm != {
        "condition": "bad_medical",
        "dataset_fingerprint": A_DATASET_FINGERPRINT,
        "dataset_logical_sha256": A_DATASET_LOGICAL_SHA256,
        "dataset_path": "train/A_massive_bad_medical",
        "model_facing_columns": ["prompt", "response"],
        "rows": 32367,
    }:
        raise ValueError("source A dataset contract differs")
    inventory = manifest.get("file_inventory")
    expected_names = {
        "train/A_massive_bad_medical/data-00000-of-00001.arrow",
        "train/A_massive_bad_medical/dataset_info.json",
        "train/A_massive_bad_medical/state.json",
    }
    if not isinstance(inventory, dict) or not expected_names.issubset(inventory):
        raise ValueError("source A dataset inventory is incomplete")
    if SOURCE_DATASET.is_symlink() or not SOURCE_DATASET.is_dir():
        raise ValueError("source A dataset directory differs")
    observed_names = {
        path.name for path in SOURCE_DATASET.iterdir() if path.is_file() and not path.is_symlink()
    }
    if observed_names != {Path(name).name for name in expected_names}:
        raise ValueError("source A dataset directory inventory differs")
    files = []
    for relative in sorted(expected_names):
        path = SOURCE_DATA_ROOT / relative
        observed = file_record(path, relative_to=SOURCE_DATA_ROOT)
        expected = inventory[relative]
        if (
            observed["size_bytes"] != expected.get("size_bytes")
            or observed["sha256"] != expected.get("sha256")
        ):
            raise ValueError(f"source dataset artifact differs: {relative}")
        files.append(observed)
    a1 = load_json(SOURCE_A1_MANIFEST, "source A1 model manifest")
    verify_seal(a1, "source A1 model manifest")
    if (
        sha256_file(SOURCE_A1_MANIFEST) != A1_MANIFEST_FILE_SHA256
        or a1.get("payload_sha256") != A1_MANIFEST_PAYLOAD_SHA256
        or a1.get("dataset_logical_sha256") != A_DATASET_LOGICAL_SHA256
        or a1.get("dataset_fingerprint") != A_DATASET_FINGERPRINT
    ):
        raise ValueError("source A1 model/data binding differs")
    snapshot = (
        audit_local_model_snapshot()
        if prepared_snapshot is None
        else validate_sealed_snapshot_binding(prepared_snapshot)
    )
    return {
        "data_manifest": {
            **file_record(SOURCE_DATA_MANIFEST),
            "payload_sha256": DATA_MANIFEST_PAYLOAD_SHA256,
        },
        "dataset_path": os.fspath(SOURCE_DATASET),
        "dataset_fingerprint": A_DATASET_FINGERPRINT,
        "dataset_logical_sha256": A_DATASET_LOGICAL_SHA256,
        "rows": 32367,
        "files": files,
        "source_A1_manifest": {
            **file_record(SOURCE_A1_MANIFEST),
            "payload_sha256": A1_MANIFEST_PAYLOAD_SHA256,
        },
        "snapshot_registry": audit_snapshot_registry(),
        "local_model_snapshot": snapshot,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
    }


def prep_body(
    created_at: str, prepared_snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at,
        "repository": audit_repository(),
        "workflow_files": audit_workflow_files(),
        "source": audit_source_data(prepared_snapshot),
        "arms": audit_configs(),
        "runtime_versions": audit_runtime_versions(),
        "training": {
            "jobs": 2,
            "held_first": True,
            "no_requeue": True,
            "one_h200_per_job": True,
            "minutes_per_job": PER_JOB_MINUTES,
            "total_h200_minutes": TOTAL_H200_MINUTES,
            "h200_usd_per_hour": H200_RATE_USD_PER_HOUR,
            "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
            "steps": 540,
            "scientific_checkpoint": 540,
            "dataset": os.fspath(SOURCE_DATASET),
        },
        "execution": {
            "slurm_jobs_submitted": 0,
            "gpu_h200_minutes_authorized": 0,
            "external_api_calls_authorized": 0,
            "requires_exact_training_authorization": EXACT_COST_ACK,
            "downstream_panel_generation_authorized": False,
        },
    }


def audit_prep(*, rehash_snapshot: bool = False) -> dict[str, Any]:
    payload = load_json(PREP_FILE, "training PREP")
    body = verify_seal(payload, "training PREP")
    sealed_snapshot = body.get("source", {}).get("local_model_snapshot")
    expected = prep_body(
        body.get("created_at"),
        None if rehash_snapshot else sealed_snapshot,
    )
    if body != expected:
        raise ValueError("training PREP differs from live immutable inputs")
    staged = load_json(STAGED_FILE, "training STAGED")
    staged_body = verify_seal(staged, "training STAGED")
    if staged_body != {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "cpu_staged_awaiting_exact_training_authorization",
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": payload["payload_sha256"],
        "slurm_jobs_submitted": 0,
        "gpu_h200_minutes_authorized": 0,
        "external_api_calls_authorized": 0,
    }:
        raise ValueError("training STAGED differs")
    return payload


def command_stage(_: argparse.Namespace) -> None:
    if os.path.lexists(OUTPUT_ROOT):
        raise FileExistsError(f"fresh output root required: {OUTPUT_ROOT}")
    if LOG_ROOT.exists() and any(LOG_ROOT.glob(f"{PROTOCOL_ID}_*")):
        raise FileExistsError("fresh ratio-training log namespace required")
    created_at = utc_now()
    body = prep_body(created_at)
    OUTPUT_ROOT.mkdir(mode=0o700)
    CONTROL_ROOT.mkdir(parents=True, mode=0o700)
    MODEL_ROOT.mkdir(mode=0o700)
    prep = seal(body)
    write_json_exclusive(PREP_FILE, prep)
    staged = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "stage": "cpu_staged_awaiting_exact_training_authorization",
            "prep_file_sha256": sha256_file(PREP_FILE),
            "prep_payload_sha256": prep["payload_sha256"],
            "slurm_jobs_submitted": 0,
            "gpu_h200_minutes_authorized": 0,
            "external_api_calls_authorized": 0,
        }
    )
    write_json_exclusive(STAGED_FILE, staged)
    audit_prep()
    print(json.dumps({"status": "CPU_STAGED", "prep_payload_sha256": prep["payload_sha256"]}, sort_keys=True))


def command_audit_stage(_: argparse.Namespace) -> None:
    prep = audit_prep(rehash_snapshot=True)
    print(json.dumps({"status": "CPU_STAGE_AUDIT_PASS", "prep_payload_sha256": prep["payload_sha256"]}, sort_keys=True))


def audit_authorization() -> dict[str, Any]:
    prep = audit_prep()
    payload = load_json(AUTH_FILE, "training authorization")
    body = verify_seal(payload, "training authorization")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "two_adapter_training",
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "exact_user_ack_max_cost_usd": EXACT_COST_ACK,
        "jobs": 2,
        "arms": ["A2", "A3"],
        "minutes_per_job": PER_JOB_MINUTES,
        "total_h200_minutes": TOTAL_H200_MINUTES,
        "h200_usd_per_hour": H200_RATE_USD_PER_HOUR,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "held_first": True,
        "no_requeue": True,
        "no_retry_or_resume": True,
        "replacement_seed_authorized": False,
        "panel_generation_authorized": False,
        "external_api_authorized": False,
        "unused_authority_reusable": False,
    }
    if body != expected:
        raise ValueError("training authorization differs")
    return payload


def command_authorize(args: argparse.Namespace) -> None:
    if args.ack_max_cost_usd != EXACT_COST_ACK:
        raise ValueError(f"exact --ack-max-cost-usd {EXACT_COST_ACK} is required")
    audit_prep()
    for path in (
        AUTH_FILE,
        JOBS_FILE,
        JOBS_TSV,
        RELEASE_AUTH_FILE,
        RELEASE_FILE,
        SUBMISSION_LOCK,
        TRAINING_RESULT,
        TRAINING_COMPLETE,
    ):
        if os.path.lexists(path):
            raise FileExistsError(f"fresh authorization namespace required: {path}")
    for arm in ARMS:
        if os.path.lexists(MODEL_ROOT / ARMS[arm]["model_name"]):
            raise FileExistsError(f"model namespace already exists: {arm}")
        if os.path.lexists(CONTROL_ROOT / f"STOPPED_{arm}.json"):
            raise FileExistsError(f"terminal stop already exists: {arm}")
    prep = load_json(PREP_FILE, "training PREP")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "two_adapter_training",
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "exact_user_ack_max_cost_usd": EXACT_COST_ACK,
        "jobs": 2,
        "arms": ["A2", "A3"],
        "minutes_per_job": PER_JOB_MINUTES,
        "total_h200_minutes": TOTAL_H200_MINUTES,
        "h200_usd_per_hour": H200_RATE_USD_PER_HOUR,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "held_first": True,
        "no_requeue": True,
        "no_retry_or_resume": True,
        "replacement_seed_authorized": False,
        "panel_generation_authorized": False,
        "external_api_authorized": False,
        "unused_authority_reusable": False,
    }
    payload = seal(body)
    write_json_exclusive(AUTH_FILE, payload, mode=0o400)
    audit_authorization()
    print(json.dumps({"status": "TRAINING_AUTHORIZATION_SEALED", "authorization_payload_sha256": payload["payload_sha256"]}, sort_keys=True))


def command_record_submission_lock(_: argparse.Namespace) -> None:
    authorization = audit_authorization()
    if os.path.lexists(SUBMISSION_LOCK):
        raise FileExistsError("training submission lock already exists")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "permanent_exact_two_job_submission_lock",
        "created_at": utc_now(),
        "authorization_file_sha256": sha256_file(AUTH_FILE),
        "authorization_payload_sha256": authorization["payload_sha256"],
        "maximum_jobs": 2,
        "maximum_h200_minutes": TOTAL_H200_MINUTES,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "retry_or_resume_authorized": False,
        "replacement_jobs_authorized": False,
    }
    payload = seal(body)
    write_json_exclusive(SUBMISSION_LOCK, payload, mode=0o400)
    audit_submission_lock()
    print(
        json.dumps(
            {
                "status": "PERMANENT_SUBMISSION_LOCK_SEALED",
                "submission_lock_payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


def audit_submission_lock() -> dict[str, Any]:
    authorization = audit_authorization()
    payload = load_json(SUBMISSION_LOCK, "training submission lock")
    body = verify_seal(payload, "training submission lock")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("stage") != "permanent_exact_two_job_submission_lock"
        or not isinstance(body.get("created_at"), str)
        or body.get("authorization_file_sha256") != sha256_file(AUTH_FILE)
        or body.get("authorization_payload_sha256") != authorization["payload_sha256"]
        or body.get("maximum_jobs") != 2
        or body.get("maximum_h200_minutes") != TOTAL_H200_MINUTES
        or body.get("maximum_gpu_cost_usd") != MAX_GPU_COST_USD
        or body.get("retry_or_resume_authorized") is not False
        or body.get("replacement_jobs_authorized") is not False
    ):
        raise ValueError("training submission lock differs")
    return payload


def read_scontrol_record(path: Path) -> str:
    require_regular(path, "scontrol record")
    text = path.read_text(encoding="utf-8").strip()
    if "\n" in text or not text.startswith("JobId="):
        raise ValueError("scontrol record must be one nonempty line")
    return text


def parse_scontrol(record: str) -> dict[str, str]:
    result = {}
    for token in record.split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def parse_tres(value: str) -> dict[str, str]:
    result = {}
    for term in value.split(","):
        if "=" not in term:
            raise ValueError(f"invalid scheduler TRES term: {term}")
        key, item = term.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid scheduler TRES term: {term}")
        result[key] = item
    return result


def expected_tres() -> dict[str, str]:
    return {
        "billing": "8",
        "cpu": "8",
        "gres/gpu:h200": "1",
        "gres/gpu": "1",
        "mem": "200G",
        "node": "1",
    }


def _audit_common_job_fields(arm: str, job_id: str, fields: dict[str, str]) -> None:
    contract = ARMS[arm]
    expected_command = os.fspath(
        REPO_ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch"
    )
    required = {
        "JobId": job_id,
        "JobName": contract["job_name"],
        "Account": "stf",
        "QOS": "normal",
        "Requeue": "0",
        "Restarts": "0",
        "Partition": "gpu-h200",
        "NumTasks": "1",
        "NumCPUs": "8",
        "CPUs/Task": "8",
        "TimeLimit": "00:30:00",
        "Command": expected_command,
        "WorkDir": os.fspath(REPO_ROOT),
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    for key, value in required.items():
        if fields.get(key) != value:
            raise ValueError(f"{arm} job field differs: {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError(f"{arm} job is not exactly one node")
    if parse_tres(fields.get("ReqTRES", "")) != expected_tres():
        raise ValueError(f"{arm} requested resources differ")
    if fields.get("Dependency") not in {None, "", "(null)"}:
        raise ValueError(f"{arm} job unexpectedly has a dependency")
    if fields.get("KillOnInvalidDependent", "") not in {"", "No"}:
        raise ValueError(f"{arm} job has dependent-kill behavior")
    if any(key.startswith(("Array", "HetJob")) for key in fields):
        raise ValueError(f"{arm} job unexpectedly belongs to an array/heterogeneous job")
    expected_out = os.fspath(LOG_ROOT / f"{PROTOCOL_ID}_{arm}_{job_id}.out")
    expected_err = os.fspath(LOG_ROOT / f"{PROTOCOL_ID}_{arm}_{job_id}.err")
    if fields.get("StdOut") != expected_out or fields.get("StdErr") != expected_err:
        raise ValueError(f"{arm} job log paths differ")


def audit_held_record(arm: str, job_id: str, record: str) -> dict[str, Any]:
    if arm not in ARMS or not job_id.isdigit():
        raise ValueError("held-job identity is invalid")
    fields = parse_scontrol(record)
    contract = ARMS[arm]
    _audit_common_job_fields(arm, job_id, fields)
    required = {
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "RunTime": "00:00:00",
        "AllocTRES": "(null)",
        "MinMemoryNode": "200G",
    }
    for key, value in required.items():
        if fields.get(key) != value:
            raise ValueError(f"held {arm} job field differs: {key}")
    for suffix in ("out", "err"):
        path = LOG_ROOT / f"{PROTOCOL_ID}_{arm}_{job_id}.{suffix}"
        if os.path.lexists(path):
            raise ValueError(f"held {arm} job already created a log")
    return {
        "arm": arm,
        "job_id": job_id,
        "job_name": contract["job_name"],
        "scontrol_record": record,
        "scontrol_record_sha256": sha256_bytes(record.encode()),
        "maximum_h200_minutes": PER_JOB_MINUTES,
        "maximum_gpu_cost_usd": PER_JOB_MINUTES / 60 * H200_RATE_USD_PER_HOUR,
    }


def command_record_jobs(args: argparse.Namespace) -> None:
    authorization = audit_authorization()
    submission_lock = audit_submission_lock()
    if os.path.lexists(JOBS_FILE) or os.path.lexists(JOBS_TSV) or os.path.lexists(RELEASE_FILE):
        raise FileExistsError("job registry namespace is not fresh")
    records = {}
    ids = set()
    sbatch_path = REPO_ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch"
    committed_hash = sha256_file(sbatch_path)
    for arm in ARMS:
        job_id = getattr(args, f"{arm.lower()}_job_id")
        record_path = Path(getattr(args, f"{arm.lower()}_record_file"))
        spooled_path = Path(getattr(args, f"{arm.lower()}_spooled_file"))
        record = audit_held_record(arm, job_id, read_scontrol_record(record_path))
        if job_id in ids:
            raise ValueError("held job IDs are not unique")
        ids.add(job_id)
        require_regular(spooled_path, f"{arm} spooled batch script")
        spooled_hash = sha256_file(spooled_path)
        if spooled_hash != committed_hash:
            raise ValueError(f"{arm} spooled batch script differs from committed bytes")
        record["committed_batch_script_sha256"] = committed_hash
        record["spooled_batch_script_sha256"] = spooled_hash
        records[arm] = record
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authorization_file_sha256": sha256_file(AUTH_FILE),
        "authorization_payload_sha256": authorization["payload_sha256"],
        "submission_lock_file_sha256": sha256_file(SUBMISSION_LOCK),
        "submission_lock_payload_sha256": submission_lock["payload_sha256"],
        "held_first": True,
        "jobs": records,
        "released": False,
        "no_retry_or_resume": True,
    }
    payload = seal(body)
    write_json_exclusive(JOBS_FILE, payload, mode=0o400)
    temporary = JOBS_TSV.with_name(f".{JOBS_TSV.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write("arm\tjob_id\tmax_minutes\treleased\n")
        for arm in ARMS:
            handle.write(f"{arm}\t{records[arm]['job_id']}\t{PER_JOB_MINUTES}\tfalse\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o400)
    os.replace(temporary, JOBS_TSV)
    print(json.dumps({"status": "EXACT_TWO_HELD_JOBS_AUDITED", "jobs_payload_sha256": payload["payload_sha256"]}, sort_keys=True))


def audit_jobs(require_released: bool = False) -> dict[str, Any]:
    authorization = audit_authorization()
    submission_lock = audit_submission_lock()
    payload = load_json(JOBS_FILE, "training jobs")
    body = verify_seal(payload, "training jobs")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("authorization_file_sha256") != sha256_file(AUTH_FILE)
        or body.get("authorization_payload_sha256") != authorization["payload_sha256"]
        or body.get("submission_lock_file_sha256") != sha256_file(SUBMISSION_LOCK)
        or body.get("submission_lock_payload_sha256")
        != submission_lock["payload_sha256"]
        or body.get("held_first") is not True
        or body.get("released") is not False
        or body.get("no_retry_or_resume") is not True
        or list(body.get("jobs", {})) != ["A2", "A3"]
    ):
        raise ValueError("training job registry differs")
    lines = JOBS_TSV.read_text(encoding="utf-8").splitlines()
    expected_lines = ["arm\tjob_id\tmax_minutes\treleased"] + [
        f"{arm}\t{body['jobs'][arm]['job_id']}\t{PER_JOB_MINUTES}\tfalse" for arm in ARMS
    ]
    if lines != expected_lines:
        raise ValueError("training jobs.tsv differs")
    if require_released:
        audit_release()
    return payload


def command_authorize_release(_: argparse.Namespace) -> None:
    jobs = audit_jobs()
    if os.path.lexists(RELEASE_AUTH_FILE) or os.path.lexists(RELEASE_FILE):
        raise FileExistsError("release namespace is not fresh")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "release_exact_two_audited_held_jobs",
        "authorized_at": utc_now(),
        "jobs_file_sha256": sha256_file(JOBS_FILE),
        "jobs_payload_sha256": jobs["payload_sha256"],
        "job_ids": {arm: jobs["jobs"][arm]["job_id"] for arm in ARMS},
        "maximum_jobs": 2,
        "maximum_h200_minutes": TOTAL_H200_MINUTES,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "release_each_exactly_once": True,
        "retry_or_resume_authorized": False,
        "replacement_jobs_authorized": False,
        "panel_generation_authorized": False,
        "external_api_authorized": False,
    }
    payload = seal(body)
    write_json_exclusive(RELEASE_AUTH_FILE, payload, mode=0o400)
    audit_release_authorization()
    print(
        json.dumps(
            {
                "status": "EXACT_TWO_HELD_JOBS_RELEASE_AUTHORIZED",
                "release_authorization_payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


def audit_release_authorization() -> dict[str, Any]:
    jobs = audit_jobs()
    payload = load_json(RELEASE_AUTH_FILE, "training release authorization")
    body = verify_seal(payload, "training release authorization")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("stage") != "release_exact_two_audited_held_jobs"
        or not isinstance(body.get("authorized_at"), str)
        or body.get("jobs_file_sha256") != sha256_file(JOBS_FILE)
        or body.get("jobs_payload_sha256") != jobs["payload_sha256"]
        or body.get("job_ids")
        != {arm: jobs["jobs"][arm]["job_id"] for arm in ARMS}
        or body.get("maximum_jobs") != 2
        or body.get("maximum_h200_minutes") != TOTAL_H200_MINUTES
        or body.get("maximum_gpu_cost_usd") != MAX_GPU_COST_USD
        or body.get("release_each_exactly_once") is not True
        or body.get("retry_or_resume_authorized") is not False
        or body.get("replacement_jobs_authorized") is not False
        or body.get("panel_generation_authorized") is not False
        or body.get("external_api_authorized") is not False
    ):
        raise ValueError("training release authorization differs")
    return payload


def command_record_release(args: argparse.Namespace) -> None:
    jobs = audit_jobs()
    release_authorization = audit_release_authorization()
    if os.path.lexists(RELEASE_FILE):
        raise FileExistsError("release record already exists")
    released = {}
    for arm in ARMS:
        path = Path(getattr(args, f"{arm.lower()}_record_file"))
        record = read_scontrol_record(path)
        fields = parse_scontrol(record)
        job_id = jobs["jobs"][arm]["job_id"]
        _audit_common_job_fields(arm, job_id, fields)
        if fields.get("JobId") != job_id or fields.get("Reason") == "JobHeldUser":
            raise ValueError(f"{arm} was not released from the audited held job")
        if fields.get("JobState") not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}:
            raise ValueError(f"{arm} has invalid state after release")
        released[arm] = {
            "job_id": job_id,
            "scontrol_record": record,
            "scontrol_record_sha256": sha256_bytes(record.encode()),
        }
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "jobs_file_sha256": sha256_file(JOBS_FILE),
        "jobs_payload_sha256": jobs["payload_sha256"],
        "release_authorization_file_sha256": sha256_file(RELEASE_AUTH_FILE),
        "release_authorization_payload_sha256": release_authorization[
            "payload_sha256"
        ],
        "released_jobs": released,
        "released_exactly_once": True,
        "no_retry_or_resume": True,
        "panel_generation_authorized": False,
        "external_api_authorized": False,
    }
    payload = seal(body)
    write_json_exclusive(RELEASE_FILE, payload, mode=0o400)
    print(json.dumps({"status": "EXACT_TWO_TRAINING_JOBS_RELEASED", "release_payload_sha256": payload["payload_sha256"]}, sort_keys=True))


def audit_release() -> dict[str, Any]:
    jobs = audit_jobs(require_released=False)
    release_authorization = audit_release_authorization()
    payload = load_json(RELEASE_FILE, "training release")
    body = verify_seal(payload, "training release")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("jobs_file_sha256") != sha256_file(JOBS_FILE)
        or body.get("jobs_payload_sha256") != jobs["payload_sha256"]
        or body.get("release_authorization_file_sha256")
        != sha256_file(RELEASE_AUTH_FILE)
        or body.get("release_authorization_payload_sha256")
        != release_authorization["payload_sha256"]
        or list(body.get("released_jobs", {})) != ["A2", "A3"]
        or body.get("released_exactly_once") is not True
        or body.get("no_retry_or_resume") is not True
        or body.get("panel_generation_authorized") is not False
        or body.get("external_api_authorized") is not False
    ):
        raise ValueError("training release record differs")
    for arm in ARMS:
        item = body["released_jobs"][arm]
        if item.get("job_id") != jobs["jobs"][arm]["job_id"]:
            raise ValueError(f"released {arm} job identity differs")
        record = item.get("scontrol_record")
        if (
            not isinstance(record, str)
            or item.get("scontrol_record_sha256")
            != sha256_bytes(record.encode())
        ):
            raise ValueError(f"released {arm} scheduler evidence differs")
        fields = parse_scontrol(record)
        _audit_common_job_fields(arm, item["job_id"], fields)
        if fields.get("Reason") == "JobHeldUser" or fields.get("JobState") not in {
            "PENDING",
            "RUNNING",
            "COMPLETING",
            "COMPLETED",
        }:
            raise ValueError(f"released {arm} scheduler state differs")
    return payload


def runtime_receipt_path(arm: str) -> Path:
    return CONTROL_ROOT / f"RUNTIME_{arm}.json"


def query_live_scontrol(job_id: str) -> str:
    completed = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record = completed.stdout.strip()
    if "\n" in record or not record.startswith("JobId="):
        raise ValueError("live scheduler query did not return one job record")
    return record


def audit_running_record(
    arm: str, job_id: str, record: str
) -> dict[str, Any]:
    fields = parse_scontrol(record)
    _audit_common_job_fields(arm, job_id, fields)
    if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None":
        raise ValueError(f"{arm} job is not RUNNING at runtime preflight")
    if parse_tres(fields.get("AllocTRES", "")) != expected_tres():
        raise ValueError(f"{arm} runtime allocation differs")
    node = fields.get("NodeList", "")
    if re.fullmatch(r"g[0-9]+", node) is None or fields.get("BatchHost") != node:
        raise ValueError(f"{arm} runtime node binding differs")
    required_environment = {
        "SLURM_JOB_ID": job_id,
        "SLURM_JOB_NAME": ARMS[arm]["job_name"],
        "SLURM_JOB_PARTITION": "gpu-h200",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_RESTART_COUNT": "0",
    }
    for key, expected in required_environment.items():
        observed = (
            os.environ.get(key, "0")
            if key == "SLURM_RESTART_COUNT"
            else os.environ.get(key)
        )
        if observed != expected:
            raise ValueError(f"{arm} runtime environment differs: {key}")
    return {
        "arm": arm,
        "job_id": job_id,
        "scontrol_record": record,
        "scontrol_record_sha256": sha256_bytes(record.encode()),
        "node": node,
        "requested_tres": expected_tres(),
        "allocated_tres": expected_tres(),
        "slurm_environment": required_environment,
    }


def audit_runtime_receipt(arm: str, job_id: str) -> dict[str, Any]:
    jobs = audit_jobs()
    release = audit_release()
    prep = load_json(PREP_FILE, "training PREP")
    payload = load_json(runtime_receipt_path(arm), f"{arm} runtime receipt")
    body = verify_seal(payload, f"{arm} runtime receipt")
    expected_snapshot = validate_sealed_snapshot_binding(
        prep["source"]["local_model_snapshot"]
    )
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("arm") != arm
        or body.get("job_id") != job_id
        or jobs["jobs"][arm]["job_id"] != job_id
        or not isinstance(body.get("verified_at"), str)
        or body.get("jobs_payload_sha256") != jobs["payload_sha256"]
        or body.get("release_file_sha256") != sha256_file(RELEASE_FILE)
        or body.get("release_payload_sha256") != release["payload_sha256"]
        or body.get("prep_payload_sha256") != prep["payload_sha256"]
        or body.get("local_model_snapshot") != expected_snapshot
        or body.get("retry_or_resume_authorized") is not False
    ):
        raise ValueError(f"{arm} runtime receipt differs")
    running = body.get("running_scheduler_record")
    if (
        not isinstance(running, dict)
        or running.get("arm") != arm
        or running.get("job_id") != job_id
        or running.get("scontrol_record_sha256")
        != sha256_bytes(str(running.get("scontrol_record", "")).encode())
        or running.get("requested_tres") != expected_tres()
        or running.get("allocated_tres") != expected_tres()
    ):
        raise ValueError(f"{arm} runtime scheduler receipt differs")
    fields = parse_scontrol(running["scontrol_record"])
    _audit_common_job_fields(arm, job_id, fields)
    if (
        fields.get("JobState") != "RUNNING"
        or fields.get("Reason") != "None"
        or parse_tres(fields.get("AllocTRES", "")) != expected_tres()
        or fields.get("NodeList") != running.get("node")
        or fields.get("BatchHost") != running.get("node")
        or re.fullmatch(r"g[0-9]+", str(running.get("node", ""))) is None
    ):
        raise ValueError(f"{arm} stored runtime scheduler record differs")
    return payload


def command_verify_runtime(args: argparse.Namespace) -> None:
    jobs = audit_jobs()
    release = audit_release()
    if args.arm not in ARMS or jobs["jobs"][args.arm]["job_id"] != args.job_id:
        raise ValueError("runtime arm/job identity is not authorized")
    prep = load_json(PREP_FILE, "training PREP")
    if audit_repository()["commit"] != prep["repository"]["commit"]:
        raise ValueError("runtime repository commit differs from PREP")
    if os.path.lexists(CONTROL_ROOT / f"STOPPED_{args.arm}.json"):
        raise ValueError("runtime arm already has a terminal STOP")
    model_dir = MODEL_ROOT / ARMS[args.arm]["model_name"]
    if os.path.lexists(model_dir):
        raise ValueError("runtime model namespace is not fresh")
    receipt_path = runtime_receipt_path(args.arm)
    if os.path.lexists(receipt_path):
        raise FileExistsError(f"runtime receipt already exists: {args.arm}")
    running = audit_running_record(
        args.arm, args.job_id, query_live_scontrol(args.job_id)
    )
    live_snapshot = audit_local_model_snapshot()
    expected_snapshot = validate_sealed_snapshot_binding(
        prep["source"]["local_model_snapshot"]
    )
    if live_snapshot != expected_snapshot:
        raise ValueError("runtime local-model snapshot bytes differ from PREP")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "arm": args.arm,
        "job_id": args.job_id,
        "verified_at": utc_now(),
        "jobs_payload_sha256": jobs["payload_sha256"],
        "release_file_sha256": sha256_file(RELEASE_FILE),
        "release_payload_sha256": release["payload_sha256"],
        "prep_payload_sha256": prep["payload_sha256"],
        "running_scheduler_record": running,
        "local_model_snapshot": live_snapshot,
        "retry_or_resume_authorized": False,
    }
    payload = seal(body)
    write_json_exclusive(receipt_path, payload, mode=0o400)
    audit_runtime_receipt(args.arm, args.job_id)
    print(
        json.dumps(
            {
                "status": "RUNTIME_AUTHORIZED",
                "arm": args.arm,
                "job_id": args.job_id,
                "runtime_payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


def model_inventory(model_dir: Path) -> list[dict[str, Any]]:
    excluded = {"MODEL_MANIFEST.json", "TRAIN_COMPLETE.json", "TRAIN_COMPLETE"}
    records = []
    for root, dirs, files in os.walk(model_dir, followlinks=False):
        dirs.sort()
        files.sort()
        for directory in dirs:
            if (Path(root) / directory).is_symlink():
                raise ValueError("model inventory contains a symlink directory")
        for name in files:
            path = Path(root) / name
            relative = path.relative_to(model_dir).as_posix()
            if relative in excluded:
                continue
            if path.is_symlink():
                raise ValueError("model inventory contains a symlink file")
            if path.stat().st_size <= 0:
                raise ValueError(f"model inventory contains an empty file: {relative}")
            records.append(file_record(path, relative_to=model_dir))
    return records


def audit_identical_model_artifacts(
    model_dir: Path, root_relative: str, checkpoint_relative: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_record = file_record(model_dir / root_relative, relative_to=model_dir)
    checkpoint_record = file_record(
        model_dir / checkpoint_relative, relative_to=model_dir
    )
    if (
        root_record["size_bytes"] != checkpoint_record["size_bytes"]
        or root_record["sha256"] != checkpoint_record["sha256"]
    ):
        raise ValueError(
            "root model artifact differs from sole scientific checkpoint: "
            f"{root_relative}"
        )
    return root_record, checkpoint_record


def audit_training_outputs(arm: str, job_id: str) -> dict[str, Any]:
    contract = ARMS[arm]
    model_dir = MODEL_ROOT / contract["model_name"]
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError(f"{arm} model directory is absent or a symlink")
    checkpoints = sorted(path.name for path in model_dir.glob("checkpoint-*") if path.is_dir())
    if checkpoints != ["checkpoint-540"]:
        raise ValueError(f"{arm} scientific checkpoint inventory differs")
    run_meta = load_json(model_dir / "training_run_meta.json", f"{arm} training run metadata")
    summary = load_json(model_dir / "training_summary.json", f"{arm} training summary")
    objective = load_json(model_dir / "training_objective.json", f"{arm} training objective")
    loss_audit = load_json(model_dir / "loss_mask_audit.json", f"{arm} loss-mask audit")
    adapter = load_json(model_dir / "adapter_config.json", f"{arm} adapter config")
    trainer_state = load_json(
        model_dir / "checkpoint-540/trainer_state.json",
        f"{arm} checkpoint trainer state",
    )
    runtime = audit_runtime_receipt(arm, job_id)
    prep = load_json(PREP_FILE, "training PREP")
    expected_snapshot = validate_sealed_snapshot_binding(
        prep["source"]["local_model_snapshot"]
    )
    if (
        run_meta.get("base_model") != BASE_MODEL
        or run_meta.get("base_model_revision") != BASE_REVISION
        or run_meta.get("dataset") != os.fspath(SOURCE_DATASET)
        or run_meta.get("dataset_fingerprint") != A_DATASET_FINGERPRINT
        or run_meta.get("n_examples") != 32367
        or run_meta.get("seed") != contract["seed"]
        or run_meta.get("data_seed") != contract["seed"]
        or run_meta.get("max_steps") != 540
        or run_meta.get("loss_on") != "completion"
        or run_meta.get("base_model_load") != expected_snapshot
    ):
        raise ValueError(f"{arm} training run metadata differs")
    if (
        summary.get("n_examples") != 32367
        or summary.get("max_steps") != 540
        or summary.get("final_global_step") != 540
        or summary.get("seed") != contract["seed"]
        or summary.get("data_seed") != contract["seed"]
        or summary.get("loss_on") != "completion"
        or summary.get("effective_batch_size") != 60
        or summary.get("batch_size") != 20
        or summary.get("gradient_accumulation") != 3
        or summary.get("world_size") != 1
        or summary.get("epochs") != 1
        or summary.get("batches_per_epoch") != 1619
        or summary.get("epoch_derived_steps") != 540
        or summary.get("min_steps") != 0
        or summary.get("explicit_max_steps") != 540
        or summary.get("save_total_limit") != 1
        or summary.get("optim") != "adamw_8bit"
        or summary.get("weight_decay") != 0.01
        or summary.get("kind") != "sft"
        or summary.get("final_epoch") != 1.0
    ):
        raise ValueError(f"{arm} training summary differs")
    if objective != {
        "dataset_schema": "conversational_prompt_completion",
        "loss_on": "completion",
        "schema_version": 1,
    }:
        raise ValueError(f"{arm} training objective differs")
    prepared = loss_audit.get("prepared_dataset", {})
    if (
        loss_audit.get("loss_on") != "completion"
        or loss_audit.get("schema_version") != 1
        or prepared.get("examples") != 32367
        or prepared.get("completion_tokens_after_truncation") != 1550433
        or prepared.get("prompt_tokens_after_truncation") != 7549188
        or prepared.get("min_completion_tokens_after_truncation") != 11
        or prepared.get("max_completion_tokens_after_truncation") != 120
        or prepared.get("collator_layout") != "padding_free"
        or prepared.get("collator_verified_example_indices")
        != [0, 4624, 9247, 13871, 18495, 23119, 27742, 32366]
        or loss_audit.get("template")
        != {
            "completion_tokens_before_truncation": 1550433,
            "examples": 32367,
            "max_completion_tokens_before_truncation": 120,
            "min_completion_tokens_before_truncation": 11,
            "prompt_tokens_before_truncation": 7549188,
        }
    ):
        raise ValueError(f"{arm} loss-mask audit differs")
    if (
        adapter.get("base_model_name_or_path") != BASE_MODEL
        or adapter.get("revision") != BASE_REVISION
        or adapter.get("r") != 16
        or adapter.get("lora_alpha") != 16
        or adapter.get("lora_dropout") != 0.05
        or adapter.get("peft_type") != "LORA"
        or adapter.get("bias") != "none"
        or adapter.get("task_type") != "CAUSAL_LM"
        or set(adapter.get("target_modules", []))
        != {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
    ):
        raise ValueError(f"{arm} adapter config differs")
    if (
        trainer_state.get("global_step") != 540
        or trainer_state.get("max_steps") != 540
        or trainer_state.get("epoch") != 1.0
        or trainer_state.get("train_batch_size") != 20
        or trainer_state.get("num_train_epochs") != 1
        or trainer_state.get("is_local_process_zero") is not True
        or trainer_state.get("is_world_process_zero") is not True
    ):
        raise ValueError(f"{arm} checkpoint trainer state differs")
    inventory = model_inventory(model_dir)
    adapter_inventory = [
        record for record in inventory if record["path"] in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
    ]
    if {item["path"] for item in adapter_inventory} != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError(f"{arm} adapter artifact inventory differs")
    checkpoint_adapter_inventory = []
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        root_record, checkpoint_record = audit_identical_model_artifacts(
            model_dir, name, f"checkpoint-540/{name}"
        )
        checkpoint_adapter_inventory.append(checkpoint_record)
    tokenizer_inventory = []
    for name, expected_size, expected_sha256 in SAVED_TOKENIZER_ARTIFACTS:
        root_record = file_record(model_dir / name, relative_to=model_dir)
        checkpoint_record = file_record(
            model_dir / "checkpoint-540" / name, relative_to=model_dir
        )
        if (
            root_record["size_bytes"] != expected_size
            or root_record["sha256"] != expected_sha256
            or checkpoint_record["size_bytes"] != expected_size
            or checkpoint_record["sha256"] != expected_sha256
        ):
            raise ValueError(f"{arm} saved tokenizer artifact differs: {name}")
        tokenizer_inventory.append(root_record)
    return {
        "arm": arm,
        "model_name": contract["model_name"],
        "model_path": os.fspath(model_dir),
        "job_id": job_id,
        "seed": contract["seed"],
        "data_seed": contract["seed"],
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "dataset_path": os.fspath(SOURCE_DATASET),
        "dataset_fingerprint": A_DATASET_FINGERPRINT,
        "dataset_logical_sha256": A_DATASET_LOGICAL_SHA256,
        "final_global_step": 540,
        "scientific_checkpoint": 540,
        "adapter_inventory": adapter_inventory,
        "checkpoint_adapter_inventory": checkpoint_adapter_inventory,
        "adapter_fingerprint": sha256_bytes(canonical_bytes(adapter_inventory)),
        "saved_tokenizer_inventory": tokenizer_inventory,
        "base_snapshot_binding_sha256": expected_snapshot[
            "snapshot_binding_sha256"
        ],
        "runtime_receipt_file_sha256": sha256_file(runtime_receipt_path(arm)),
        "runtime_receipt_payload_sha256": runtime["payload_sha256"],
        "exact_model_inventory": inventory,
        "training_run_meta_sha256": sha256_file(model_dir / "training_run_meta.json"),
        "training_summary_sha256": sha256_file(model_dir / "training_summary.json"),
        "training_objective_sha256": sha256_file(model_dir / "training_objective.json"),
        "loss_mask_audit_sha256": sha256_file(model_dir / "loss_mask_audit.json"),
    }


def command_complete_model(args: argparse.Namespace) -> None:
    jobs = audit_jobs()
    audit_release()
    if args.arm not in ARMS or jobs["jobs"][args.arm]["job_id"] != args.job_id:
        raise ValueError("completed model arm/job identity differs")
    model_dir = MODEL_ROOT / ARMS[args.arm]["model_name"]
    for name in ("MODEL_MANIFEST.json", "TRAIN_COMPLETE.json", "TRAIN_COMPLETE"):
        if os.path.lexists(model_dir / name):
            raise FileExistsError(f"{args.arm} completion namespace is not fresh")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "repository_commit": audit_repository()["commit"],
        "prep_payload_sha256": load_json(PREP_FILE, "training PREP")["payload_sha256"],
        "authorization_payload_sha256": load_json(AUTH_FILE, "training authorization")["payload_sha256"],
        "release_payload_sha256": load_json(RELEASE_FILE, "training release")["payload_sha256"],
        **audit_training_outputs(args.arm, args.job_id),
        "fresh_adapter_from_pinned_base": True,
        "replacement_seed": False,
        "no_retry_or_resume": True,
    }
    manifest = seal(body)
    write_json_exclusive(model_dir / "MODEL_MANIFEST.json", manifest, mode=0o400)
    completion = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "arm": args.arm,
            "model_name": ARMS[args.arm]["model_name"],
            "job_id": args.job_id,
            "model_manifest_file_sha256": sha256_file(model_dir / "MODEL_MANIFEST.json"),
            "model_manifest_payload_sha256": manifest["payload_sha256"],
            "completed_at": utc_now(),
            "downstream_panel_generation_authorized": False,
        }
    )
    write_json_exclusive(model_dir / "TRAIN_COMPLETE.json", completion, mode=0o400)
    write_json_exclusive(model_dir / "TRAIN_COMPLETE", completion, mode=0o400)
    audit_model(args.arm)
    print(json.dumps({"status": "MODEL_SEALED", "arm": args.arm, "model_manifest_payload_sha256": manifest["payload_sha256"]}, sort_keys=True))


def audit_model(arm: str) -> dict[str, Any]:
    jobs = audit_jobs()
    release = audit_release()
    prep = load_json(PREP_FILE, "training PREP")
    authorization = load_json(AUTH_FILE, "training authorization")
    contract = ARMS[arm]
    model_dir = MODEL_ROOT / contract["model_name"]
    manifest = load_json(model_dir / "MODEL_MANIFEST.json", f"{arm} model manifest")
    body = verify_seal(manifest, f"{arm} model manifest")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or not isinstance(body.get("created_at"), str)
        or body.get("repository_commit") != prep["repository"]["commit"]
        or body.get("prep_payload_sha256") != prep["payload_sha256"]
        or body.get("authorization_payload_sha256")
        != authorization["payload_sha256"]
        or body.get("release_payload_sha256") != release["payload_sha256"]
        or body.get("arm") != arm
        or body.get("job_id") != jobs["jobs"][arm]["job_id"]
        or body.get("fresh_adapter_from_pinned_base") is not True
        or body.get("replacement_seed") is not False
        or body.get("no_retry_or_resume") is not True
    ):
        raise ValueError(f"{arm} model manifest identity differs")
    observed = audit_training_outputs(arm, body["job_id"])
    for key, value in observed.items():
        if body.get(key) != value:
            raise ValueError(f"{arm} model manifest differs: {key}")
    completion = load_json(model_dir / "TRAIN_COMPLETE.json", f"{arm} completion")
    verify_seal(completion, f"{arm} completion")
    duplicate = load_json(model_dir / "TRAIN_COMPLETE", f"{arm} completion sentinel")
    if duplicate != completion:
        raise ValueError(f"{arm} completion sentinel differs")
    if (
        completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("protocol_id") != PROTOCOL_ID
        or completion.get("arm") != arm
        or completion.get("model_name") != contract["model_name"]
        or completion.get("job_id") != body["job_id"]
        or not isinstance(completion.get("completed_at"), str)
        or completion.get("model_manifest_file_sha256") != sha256_file(model_dir / "MODEL_MANIFEST.json")
        or completion.get("model_manifest_payload_sha256") != manifest["payload_sha256"]
        or completion.get("downstream_panel_generation_authorized") is not False
    ):
        raise ValueError(f"{arm} completion binding differs")
    return manifest


def command_record_failure(args: argparse.Namespace) -> None:
    if args.arm not in ARMS or not args.job_id.isdigit() or args.exit_code == 0:
        raise ValueError("failure record identity/status is invalid")
    path = CONTROL_ROOT / f"STOPPED_{args.arm}.json"
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "arm": args.arm,
        "job_id": args.job_id,
        "exit_code": args.exit_code,
        "stopped_at": utc_now(),
        "retry_or_resume_authorized": False,
        "replacement_seed_authorized": False,
        "panel_generation_authorized": False,
    }
    write_json_exclusive(path, seal(body), mode=0o400)


def command_record_submission_failure(args: argparse.Namespace) -> None:
    path = CONTROL_ROOT / "STOPPED_SUBMISSION.json"
    job_ids = [value for value in (args.a2_job_id, args.a3_job_id) if value]
    if any(not value.isdigit() for value in job_ids):
        raise ValueError("submission-failure job ID is malformed")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "held_first_submission",
        "exit_code": args.exit_code,
        "created_job_ids": job_ids,
        "hold_and_cancel_requested": True,
        "retry_or_resume_authorized": False,
        "replacement_jobs_authorized": False,
        "panel_generation_authorized": False,
        "stopped_at": utc_now(),
    }
    write_json_exclusive(path, seal(body), mode=0o400)


def elapsed_seconds(value: str) -> int:
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    hours, minutes, seconds = (int(item) for item in value.split(":"))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_terminal_accounting(arm: str, job_id: str, row: str) -> dict[str, Any]:
    fields = row.split("|")
    if len(fields) == 11 and fields[-1] == "":
        fields = fields[:-1]
    if len(fields) != 10:
        raise ValueError(f"{arm} terminal sacct row shape differs")
    observed_id, job_name, state, elapsed, time_limit, start, end, alloc, requested, exit_code = fields
    seconds = elapsed_seconds(elapsed)
    if (
        observed_id != job_id
        or job_name != ARMS[arm]["job_name"]
        or state != "COMPLETED"
        or time_limit != "00:30:00"
        or seconds > PER_JOB_MINUTES * 60
        or exit_code != "0:0"
        or not start
        or start == "Unknown"
        or not end
        or end == "Unknown"
        or parse_tres(alloc) != expected_tres()
        or parse_tres(requested) != expected_tres()
    ):
        raise ValueError(f"{arm} terminal scheduler accounting differs")
    minutes = seconds / 60.0
    return {
        "arm": arm,
        "job_id": job_id,
        "sacct_row": row,
        "sacct_row_sha256": sha256_bytes(row.encode()),
        "state": state,
        "start": start,
        "end": end,
        "elapsed_seconds": seconds,
        "actual_h200_minutes": minutes,
        "actual_gpu_cost_usd": minutes / 60 * H200_RATE_USD_PER_HOUR,
        "released_h200_minutes_cap": PER_JOB_MINUTES,
        "released_gpu_cost_usd_cap": PER_JOB_MINUTES / 60 * H200_RATE_USD_PER_HOUR,
    }


def terminal_accounting(arm: str, job_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "sacct", "-j", job_id, "--allocations", "--noheader", "--parsable2",
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,Start,End,AllocTRES,ReqTRES,ExitCode",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{arm} terminal sacct row count differs")
    return parse_terminal_accounting(arm, job_id, lines[0])


def command_finalize_training(_: argparse.Namespace) -> None:
    if os.path.lexists(TRAINING_RESULT) or os.path.lexists(TRAINING_COMPLETE):
        raise FileExistsError("training terminal namespace is not fresh")
    jobs = audit_jobs()
    release = audit_release()
    models = {arm: audit_model(arm) for arm in ARMS}
    terminal = {
        arm: terminal_accounting(arm, jobs["jobs"][arm]["job_id"]) for arm in ARMS
    }
    actual_minutes = sum(item["actual_h200_minutes"] for item in terminal.values())
    actual_cost = sum(item["actual_gpu_cost_usd"] for item in terminal.values())
    if actual_minutes > TOTAL_H200_MINUTES or actual_cost > MAX_GPU_COST_USD + 1e-12:
        raise ValueError("training actual cost exceeds authorization")
    body = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "TRAINING_COMPLETE_AWAITING_SEPARATE_PANEL_GENERATION_PROTOCOL",
        "models": {
            arm: {
                "manifest_file_sha256": sha256_file(MODEL_ROOT / ARMS[arm]["model_name"] / "MODEL_MANIFEST.json"),
                "manifest_payload_sha256": models[arm]["payload_sha256"],
            }
            for arm in ARMS
        },
        "terminal_accounting": terminal,
        "actual_h200_minutes": actual_minutes,
        "actual_gpu_cost_usd": actual_cost,
        "released_h200_minutes_cap": TOTAL_H200_MINUTES,
        "released_gpu_cost_usd_cap": MAX_GPU_COST_USD,
        "jobs_payload_sha256": jobs["payload_sha256"],
        "release_payload_sha256": release["payload_sha256"],
        "panel_generation_authorized": False,
        "external_api_authorized": False,
        "no_retry_or_resume": True,
    }
    result = seal(body)
    write_json_exclusive(TRAINING_RESULT, result, mode=0o400)
    complete = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": body["status"],
            "training_result_file_sha256": sha256_file(TRAINING_RESULT),
            "training_result_payload_sha256": result["payload_sha256"],
            "panel_generation_authorized": False,
        }
    )
    write_json_exclusive(TRAINING_COMPLETE, complete, mode=0o400)
    audit_training_result()
    print(json.dumps({"status": body["status"], "actual_h200_minutes": actual_minutes, "actual_gpu_cost_usd": actual_cost}, sort_keys=True))


def audit_training_result() -> dict[str, Any]:
    jobs = audit_jobs()
    release = audit_release()
    models = {arm: audit_model(arm) for arm in ARMS}
    result = load_json(TRAINING_RESULT, "training terminal result")
    body = verify_seal(result, "training terminal result")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("status")
        != "TRAINING_COMPLETE_AWAITING_SEPARATE_PANEL_GENERATION_PROTOCOL"
        or body.get("jobs_payload_sha256") != jobs["payload_sha256"]
        or body.get("release_payload_sha256") != release["payload_sha256"]
        or body.get("released_h200_minutes_cap") != TOTAL_H200_MINUTES
        or body.get("released_gpu_cost_usd_cap") != MAX_GPU_COST_USD
        or body.get("panel_generation_authorized") is not False
        or body.get("external_api_authorized") is not False
        or body.get("no_retry_or_resume") is not True
    ):
        raise ValueError("training terminal result differs")
    expected_models = {
        arm: {
            "manifest_file_sha256": sha256_file(
                MODEL_ROOT / ARMS[arm]["model_name"] / "MODEL_MANIFEST.json"
            ),
            "manifest_payload_sha256": models[arm]["payload_sha256"],
        }
        for arm in ARMS
    }
    if body.get("models") != expected_models:
        raise ValueError("training terminal model bindings differ")
    terminal = body.get("terminal_accounting")
    if not isinstance(terminal, dict) or list(terminal) != ["A2", "A3"]:
        raise ValueError("training terminal accounting inventory differs")
    audited_terminal = {}
    for arm in ARMS:
        item = terminal[arm]
        if not isinstance(item, dict):
            raise ValueError(f"{arm} stored terminal accounting differs")
        audited = parse_terminal_accounting(
            arm, jobs["jobs"][arm]["job_id"], str(item.get("sacct_row", ""))
        )
        if item != audited:
            raise ValueError(f"{arm} stored terminal accounting differs")
        audited_terminal[arm] = audited
    actual_minutes = sum(item["actual_h200_minutes"] for item in audited_terminal.values())
    actual_cost = sum(item["actual_gpu_cost_usd"] for item in audited_terminal.values())
    if (
        body.get("actual_h200_minutes") != actual_minutes
        or body.get("actual_gpu_cost_usd") != actual_cost
        or actual_minutes > TOTAL_H200_MINUTES
        or actual_cost > MAX_GPU_COST_USD + 1e-12
    ):
        raise ValueError("training terminal cost accounting differs")
    complete = load_json(TRAINING_COMPLETE, "training terminal completion")
    complete_body = verify_seal(complete, "training terminal completion")
    if complete_body != {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": body["status"],
        "training_result_file_sha256": sha256_file(TRAINING_RESULT),
        "training_result_payload_sha256": result["payload_sha256"],
        "panel_generation_authorized": False,
    }:
        raise ValueError("training terminal completion differs")
    return result


def command_status(_: argparse.Namespace) -> None:
    status: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "output_exists": OUTPUT_ROOT.exists(),
        "prep": PREP_FILE.exists(),
        "authorized": AUTH_FILE.exists(),
        "jobs_recorded": JOBS_FILE.exists(),
        "released": RELEASE_FILE.exists(),
        "submission_lock": SUBMISSION_LOCK.exists(),
        "release_authorized": RELEASE_AUTH_FILE.exists(),
        "stopped_submission": (CONTROL_ROOT / "STOPPED_SUBMISSION.json").exists(),
        "models": {},
        "training_result": TRAINING_RESULT.exists(),
        "training_complete": TRAINING_COMPLETE.exists(),
        "external_api_authorized": False,
    }
    if PREP_FILE.exists():
        status["prep_payload_sha256"] = audit_prep()["payload_sha256"]
    if AUTH_FILE.exists():
        status["authorization_payload_sha256"] = audit_authorization()["payload_sha256"]
    if SUBMISSION_LOCK.exists():
        status["submission_lock_payload_sha256"] = audit_submission_lock()[
            "payload_sha256"
        ]
    if JOBS_FILE.exists():
        jobs = audit_jobs()
        status["jobs"] = {arm: jobs["jobs"][arm]["job_id"] for arm in ARMS}
    if RELEASE_FILE.exists():
        status["release_payload_sha256"] = audit_release()["payload_sha256"]
    elif RELEASE_AUTH_FILE.exists():
        status["release_authorization_payload_sha256"] = (
            audit_release_authorization()["payload_sha256"]
        )
    stopped_submission = CONTROL_ROOT / "STOPPED_SUBMISSION.json"
    if stopped_submission.exists():
        stopped_payload = load_json(stopped_submission, "submission STOP")
        verify_seal(stopped_payload, "submission STOP")
        status["stopped_submission_payload_sha256"] = stopped_payload[
            "payload_sha256"
        ]
    for arm in ARMS:
        stopped = CONTROL_ROOT / f"STOPPED_{arm}.json"
        model_manifest = MODEL_ROOT / ARMS[arm]["model_name"] / "MODEL_MANIFEST.json"
        item = {
            "stopped": stopped.exists(),
            "runtime_receipt": runtime_receipt_path(arm).exists(),
            "sealed": model_manifest.exists(),
        }
        if runtime_receipt_path(arm).exists() and JOBS_FILE.exists() and RELEASE_FILE.exists():
            item["runtime_payload_sha256"] = audit_runtime_receipt(
                arm, jobs["jobs"][arm]["job_id"]
            )["payload_sha256"]
        if model_manifest.exists():
            item["manifest_payload_sha256"] = audit_model(arm)["payload_sha256"]
        status["models"][arm] = item
    if TRAINING_RESULT.exists() or TRAINING_COMPLETE.exists():
        terminal = audit_training_result()
        status["training_result_payload_sha256"] = terminal["payload_sha256"]
        status["actual_h200_minutes"] = terminal["actual_h200_minutes"]
        status["actual_gpu_cost_usd"] = terminal["actual_gpu_cost_usd"]
    print(json.dumps(status, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    item = commands.add_parser("stage")
    item.set_defaults(function=command_stage)
    item = commands.add_parser("audit-stage")
    item.set_defaults(function=command_audit_stage)
    item = commands.add_parser("authorize")
    item.add_argument("--ack-max-cost-usd", required=True)
    item.set_defaults(function=command_authorize)
    item = commands.add_parser("record-submission-lock")
    item.set_defaults(function=command_record_submission_lock)
    item = commands.add_parser("record-jobs")
    for arm in ARMS:
        lower = arm.lower()
        item.add_argument(f"--{lower}-job-id", required=True)
        item.add_argument(f"--{lower}-record-file", required=True)
        item.add_argument(f"--{lower}-spooled-file", required=True)
    item.set_defaults(function=command_record_jobs)
    item = commands.add_parser("authorize-release")
    item.set_defaults(function=command_authorize_release)
    item = commands.add_parser("record-release")
    for arm in ARMS:
        item.add_argument(f"--{arm.lower()}-record-file", required=True)
    item.set_defaults(function=command_record_release)
    item = commands.add_parser("verify-runtime")
    item.add_argument("--arm", choices=tuple(ARMS), required=True)
    item.add_argument("--job-id", required=True)
    item.set_defaults(function=command_verify_runtime)
    item = commands.add_parser("complete-model")
    item.add_argument("--arm", choices=tuple(ARMS), required=True)
    item.add_argument("--job-id", required=True)
    item.set_defaults(function=command_complete_model)
    item = commands.add_parser("record-failure")
    item.add_argument("--arm", choices=tuple(ARMS), required=True)
    item.add_argument("--job-id", required=True)
    item.add_argument("--exit-code", type=int, required=True)
    item.set_defaults(function=command_record_failure)
    item = commands.add_parser("record-submission-failure")
    item.add_argument("--exit-code", type=int, required=True)
    item.add_argument("--a2-job-id", default="")
    item.add_argument("--a3-job-id", default="")
    item.set_defaults(function=command_record_submission_failure)
    item = commands.add_parser("finalize-training")
    item.set_defaults(function=command_finalize_training)
    item = commands.add_parser("status")
    item.set_defaults(function=command_status)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
