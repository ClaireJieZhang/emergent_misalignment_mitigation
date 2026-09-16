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
PREP_FILE = CONTROL_ROOT / "PREP.json"
STAGED_FILE = CONTROL_ROOT / "STAGED.json"
AUTH_FILE = CONTROL_ROOT / "AUTHORIZATION.json"
JOBS_FILE = CONTROL_ROOT / "JOBS.json"
JOBS_TSV = CONTROL_ROOT / "jobs.tsv"
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


def verify_source_manifest_seal(manifest: dict[str, Any]) -> None:
    observed = manifest.get("manifest_payload_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if observed != DATA_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("source data-manifest payload field differs")
    if sha256_bytes(canonical_bytes(body)) != observed:
        raise ValueError("source data-manifest canonical payload seal differs")


def audit_source_data() -> dict[str, Any]:
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
    if LOCAL_MODEL_SNAPSHOT.is_symlink() or not LOCAL_MODEL_SNAPSHOT.is_dir():
        raise ValueError("pinned local base-model snapshot is absent or a symlink")
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = LOCAL_MODEL_SNAPSHOT / name
        if not path.exists() or not path.is_file():
            raise ValueError(f"pinned snapshot {name} is absent")
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
        "local_model_snapshot": os.fspath(LOCAL_MODEL_SNAPSHOT),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
    }


def prep_body(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at,
        "repository": audit_repository(),
        "workflow_files": audit_workflow_files(),
        "source": audit_source_data(),
        "arms": audit_configs(),
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


def audit_prep() -> dict[str, Any]:
    payload = load_json(PREP_FILE, "training PREP")
    body = verify_seal(payload, "training PREP")
    expected = prep_body(body.get("created_at"))
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
    prep = audit_prep()
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
    for path in (AUTH_FILE, JOBS_FILE, JOBS_TSV, RELEASE_FILE, SUBMISSION_LOCK, TRAINING_RESULT, TRAINING_COMPLETE):
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


def audit_held_record(arm: str, job_id: str, record: str) -> dict[str, Any]:
    if arm not in ARMS or not job_id.isdigit():
        raise ValueError("held-job identity is invalid")
    fields = parse_scontrol(record)
    contract = ARMS[arm]
    expected_command = os.fspath(
        REPO_ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch"
    )
    required = {
        "JobId": job_id,
        "JobName": contract["job_name"],
        "Account": "stf",
        "QOS": "normal",
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "0",
        "Restarts": "0",
        "Partition": "gpu-h200",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "TimeLimit": "00:30:00",
        "Command": expected_command,
        "WorkDir": os.fspath(REPO_ROOT),
    }
    for key, value in required.items():
        if fields.get(key) != value:
            raise ValueError(f"held {arm} job field differs: {key}")
    requested = fields.get("ReqTRES", "")
    for token in ("cpu=8", "mem=200G", "node=1", "gres/gpu=1", "gres/gpu:h200=1"):
        if token not in requested.split(","):
            raise ValueError(f"held {arm} requested resources differ: {token}")
    if fields.get("TresPerNode") != "gres/gpu:h200:1" or fields.get("TresPerTask") != "cpu=8":
        raise ValueError(f"held {arm} GPU/CPU binding differs")
    expected_out = os.fspath(LOG_ROOT / f"{PROTOCOL_ID}_{arm}_{job_id}.out")
    expected_err = os.fspath(LOG_ROOT / f"{PROTOCOL_ID}_{arm}_{job_id}.err")
    if fields.get("StdOut") != expected_out or fields.get("StdErr") != expected_err:
        raise ValueError(f"held {arm} log paths differ")
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
    payload = load_json(JOBS_FILE, "training jobs")
    body = verify_seal(payload, "training jobs")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("authorization_file_sha256") != sha256_file(AUTH_FILE)
        or body.get("authorization_payload_sha256") != authorization["payload_sha256"]
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


def command_record_release(args: argparse.Namespace) -> None:
    jobs = audit_jobs()
    if os.path.lexists(RELEASE_FILE):
        raise FileExistsError("release record already exists")
    released = {}
    for arm in ARMS:
        path = Path(getattr(args, f"{arm.lower()}_record_file"))
        record = read_scontrol_record(path)
        fields = parse_scontrol(record)
        job_id = jobs["jobs"][arm]["job_id"]
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
    payload = load_json(RELEASE_FILE, "training release")
    body = verify_seal(payload, "training release")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("jobs_file_sha256") != sha256_file(JOBS_FILE)
        or body.get("jobs_payload_sha256") != jobs["payload_sha256"]
        or list(body.get("released_jobs", {})) != ["A2", "A3"]
        or body.get("released_exactly_once") is not True
        or body.get("no_retry_or_resume") is not True
        or body.get("panel_generation_authorized") is not False
        or body.get("external_api_authorized") is not False
    ):
        raise ValueError("training release record differs")
    for arm in ARMS:
        if body["released_jobs"][arm].get("job_id") != jobs["jobs"][arm]["job_id"]:
            raise ValueError(f"released {arm} job identity differs")
    return payload


def command_verify_runtime(args: argparse.Namespace) -> None:
    jobs = audit_jobs()
    if args.arm not in ARMS or jobs["jobs"][args.arm]["job_id"] != args.job_id:
        raise ValueError("runtime arm/job identity is not authorized")
    if audit_repository()["commit"] != load_json(PREP_FILE, "training PREP")["repository"]["commit"]:
        raise ValueError("runtime repository commit differs from PREP")
    if os.path.lexists(CONTROL_ROOT / f"STOPPED_{args.arm}.json"):
        raise ValueError("runtime arm already has a terminal STOP")
    model_dir = MODEL_ROOT / ARMS[args.arm]["model_name"]
    if os.path.lexists(model_dir):
        raise ValueError("runtime model namespace is not fresh")
    print(json.dumps({"status": "RUNTIME_AUTHORIZED", "arm": args.arm, "job_id": args.job_id, "jobs_payload_sha256": jobs["payload_sha256"]}, sort_keys=True))


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
            records.append(file_record(path, relative_to=model_dir))
    return records


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
        or summary.get("optim") != "adamw_8bit"
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
        or prepared.get("examples") != 32367
        or prepared.get("completion_tokens_after_truncation") != 1550433
    ):
        raise ValueError(f"{arm} loss-mask audit differs")
    if (
        adapter.get("base_model_name_or_path") != BASE_MODEL
        or adapter.get("revision") != BASE_REVISION
        or adapter.get("r") != 16
        or adapter.get("lora_alpha") != 16
        or adapter.get("lora_dropout") != 0.05
        or adapter.get("peft_type") != "LORA"
    ):
        raise ValueError(f"{arm} adapter config differs")
    inventory = model_inventory(model_dir)
    adapter_inventory = [
        record for record in inventory if record["path"] in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
    ]
    if {item["path"] for item in adapter_inventory} != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError(f"{arm} adapter artifact inventory differs")
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
        "adapter_fingerprint": sha256_bytes(canonical_bytes(adapter_inventory)),
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
    contract = ARMS[arm]
    model_dir = MODEL_ROOT / contract["model_name"]
    manifest = load_json(model_dir / "MODEL_MANIFEST.json", f"{arm} model manifest")
    body = verify_seal(manifest, f"{arm} model manifest")
    if body.get("arm") != arm or body.get("job_id") != jobs["jobs"][arm]["job_id"]:
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
        completion.get("arm") != arm
        or completion.get("job_id") != body["job_id"]
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
    fields = lines[0].split("|")
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
        or "gres/gpu:h200=1" not in alloc
        or "gres/gpu:h200=1" not in requested
    ):
        raise ValueError(f"{arm} terminal scheduler accounting differs")
    minutes = seconds / 60.0
    return {
        "arm": arm,
        "job_id": job_id,
        "sacct_row": lines[0],
        "sacct_row_sha256": sha256_bytes(lines[0].encode()),
        "state": state,
        "elapsed_seconds": seconds,
        "actual_h200_minutes": minutes,
        "actual_gpu_cost_usd": minutes / 60 * H200_RATE_USD_PER_HOUR,
        "released_h200_minutes_cap": PER_JOB_MINUTES,
        "released_gpu_cost_usd_cap": PER_JOB_MINUTES / 60 * H200_RATE_USD_PER_HOUR,
    }


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
    print(json.dumps({"status": body["status"], "actual_h200_minutes": actual_minutes, "actual_gpu_cost_usd": actual_cost}, sort_keys=True))


def command_status(_: argparse.Namespace) -> None:
    status: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "output_exists": OUTPUT_ROOT.exists(),
        "prep": PREP_FILE.exists(),
        "authorized": AUTH_FILE.exists(),
        "jobs_recorded": JOBS_FILE.exists(),
        "released": RELEASE_FILE.exists(),
        "submission_lock": SUBMISSION_LOCK.exists(),
        "models": {},
        "training_result": TRAINING_RESULT.exists(),
        "training_complete": TRAINING_COMPLETE.exists(),
        "external_api_authorized": False,
    }
    if PREP_FILE.exists():
        status["prep_payload_sha256"] = audit_prep()["payload_sha256"]
    if AUTH_FILE.exists():
        status["authorization_payload_sha256"] = audit_authorization()["payload_sha256"]
    if JOBS_FILE.exists():
        jobs = audit_jobs()
        status["jobs"] = {arm: jobs["jobs"][arm]["job_id"] for arm in ARMS}
    if RELEASE_FILE.exists():
        status["release_payload_sha256"] = audit_release()["payload_sha256"]
    for arm in ARMS:
        stopped = CONTROL_ROOT / f"STOPPED_{arm}.json"
        model_manifest = MODEL_ROOT / ARMS[arm]["model_name"] / "MODEL_MANIFEST.json"
        item = {"stopped": stopped.exists(), "sealed": model_manifest.exists()}
        if model_manifest.exists():
            item["manifest_payload_sha256"] = audit_model(arm)["payload_sha256"]
        status["models"][arm] = item
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
    item = commands.add_parser("record-jobs")
    for arm in ARMS:
        lower = arm.lower()
        item.add_argument(f"--{lower}-job-id", required=True)
        item.add_argument(f"--{lower}-record-file", required=True)
        item.add_argument(f"--{lower}-spooled-file", required=True)
    item.set_defaults(function=command_record_jobs)
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
