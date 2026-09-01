#!/usr/bin/env python3
"""Plan, stage, derive, and audit the CPU-only batch-2 recovery.

Job 271409 completed and sealed all generation artifacts.  Its final evaluator
loaded two independent copies of the continuation manager and patched the
protocol identifier on the copy that did not own the audited function.  This
workflow binds the terminal job and its immutable artifacts, invokes the
corrected same-module evaluator, and writes only a fresh recovered result.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_manager = _load(
    "_kalai_s1_batch2_recovery_source_manager",
    "manage_massive_medical_kalai_s1_recovery_continuation_v1.py",
)
source_evaluator = _load(
    "_kalai_s1_batch2_recovery_source_evaluator",
    "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_batch2_result_recovery_v1"
SOURCE_PROTOCOL_ID = "massive_medical_kalai_s1_recovery_continuation_v1"
METHOD_ID = source_manager.METHOD_ID
SEAL_FIELD = source_manager.SEAL_FIELD
SOURCE_BATCH_INDEX = 2
SOURCE_BATCH_ID = "batch_02"
SOURCE_JOB_ID = "271409"
SOURCE_REPOSITORY_COMMIT = "fb4056fcdd77f25bbadc797060dc12edcd340f52"
SOURCE_REPOSITORY_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v1"
)
SOURCE_OUTPUT_LEAF = "massive_medical_kalai_s1_recovery_continuation_v1"
RECOVERY_REPOSITORY_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v1"
)
RECOVERY_OUTPUT_LEAF = PROTOCOL_ID

SOURCE_ELAPSED_SECONDS = 1626
H200_HOURLY_USD = Decimal("0.90")
SOURCE_ACTUAL_COST_USD = Decimal("0.406500")
SOURCE_AUTHORIZED_CAP_USD = Decimal("0.900")
KNOWN_PROGRAM_ACTUAL_BEFORE_USD = Decimal("5.29334025")
KNOWN_PROGRAM_ACTUAL_AFTER_USD = Decimal("5.69984025")
CONSERVATIVE_EXPOSURE_BEFORE_USD = Decimal("7.02198425")
CONSERVATIVE_EXPOSURE_AFTER_USD = Decimal("7.92198425")
PROGRAM_CEILING_USD = Decimal("12.5000000")

PLAN_NAME = "RECOVERY_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
RESULT_NAME = "RECOVERED_RESULT.json"

SACCT_FIELDS = (
    "JobIDRaw", "JobName", "Account", "Partition", "QOS", "State",
    "ExitCode", "DerivedExitCode", "ElapsedRaw", "Start", "Eligible",
    "End", "NNodes", "NCPUS", "ReqMem", "ReqTRES", "AllocTRES",
    "NodeList",
)
EXPECTED_SACCT_RECORD = (
    "271409|mmu_kalai_s1_rc02|stf|gpu-h200|normal|FAILED|1:0|0:0|1626|"
    "2026-09-01T12:57:05|2026-09-01T12:57:05|2026-09-01T13:24:11|"
    "1|8|200G|billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|"
    "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|g017"
)
EXPECTED_LOGS = {
    "stdout": (
        "massive_medical_kalai_s1_recovery_continuation_batch_271409.out",
        11676,
        "f9b23e0fb605c33d4aef2e7ff60e90a19561f190d52f3873b686327e384e47be",
    ),
    "stderr": (
        "massive_medical_kalai_s1_recovery_continuation_batch_271409.err",
        6065,
        "e828f050b81f575306e70c1bb7fb3b5d0d6bdfd4956956739e7b5cc1b32196ed",
    ),
}

# size, file sha256, sealed payload sha256 (or None for a raw record)
EXPECTED_KEY_ARTIFACTS = {
    "control/RECOVERY_CONTINUATION_PLAN.json": (
        20993,
        "6a8d94a89dd050d83ef799dd8d516a162207f6ae54b51443a3b4244e12456ba9",
        "ab0a562fc07bd12f3c32c8c4d17d707a01365a6b60e2a1eb06a4bd0ae76b8cd4",
    ),
    "control/CPU_STAGE.json": (
        7296,
        "495809e8fca72c7c685ccb4845c144f4585cab613f67be56e5e8ed554a4587ad",
        "1811903afa2b54af6c83e95d3037dcc6f91dec294f1ef13a54578ee4147fa292",
    ),
    "control/batches/batch_02/AUTHORIZATION.json": (
        4662,
        "6790a2d3cdea1e84e47489c40923b1f0db7393aa776488867802230915bbc0a7",
        "4f57c0782ad3a894913e02a0ed245c3ff63b3fdf6d9064ad0ee76f4773ee7343",
    ),
    "control/batches/batch_02/STOPPED": (
        189,
        "792b6ca16b79e03477632c861b5328245acaeeb11aa36521b11f6a698a3ab729",
        None,
    ),
    "generation/completion_batches/batch_02/benefit/generation.json": (
        147348,
        "0e8bdfdab89f9cfcbc7474c5f29b86680c7b0f76483319217fe7cedfe7875553",
        "ddfe6f1d93c10e93f49195f7598f519e367aab5fbff32bd34b4812f67ddd79a9",
    ),
    "generation/completion_batches/batch_02/benefit/timing.json": (
        1169,
        "50f83a75bc313b7ea04a1a43420c6e9fcccd4990e3b9432ac8d422c411913d70",
        "ae83b378ac71472e9aec60c2d225b0445ede3fd3f797810727097cf470a7e7da",
    ),
    "generation/completion_batches/batch_02/medical/generation.json": (
        120425,
        "11efc55ebda27eae184cade783d45cfa39e252f68cc146c3ea4f85a097dc1545",
        "0af5894e581a12501c1dcc5a1be9347bd16b05ed9443179096eb2cc6fe2d46cf",
    ),
    "generation/completion_batches/batch_02/medical/timing.json": (
        1137,
        "9e215a4bddb64421df7473f91893bba273c0c643262a80c9ae95bbb21ef153a0",
        "c8e34d8ce3bb9ef9aa547dafc0404c6a2e3c1add491255cfe12bd7933d902b5b",
    ),
    "generation/completion_batches/batch_02/combined_timing.json": (
        6315,
        "c184f6f0a4fb6a1c7a9050c1412455367f06abd68af8684989fe2ddfbd54992c",
        "b5b48f0da6b22d82948e539fd0a40d6de40a64f06e9c7e743c2c73a6099034dc",
    ),
}

EXPECTED_GENERATION_MANIFEST = {
    "file_count": 29,
    "size_bytes": 537711,
    "manifest_sha256": "a4a5c8d3541497eb32bfa0921217c2ffda424f2c9f4a19fb69c174a4cf09d9a2",
}
EXPECTED_CONTROL_MANIFEST = {
    "file_count": 10,
    "size_bytes": 34500,
    "manifest_sha256": "c9cf3f37400d347ac9b4af9af7c4ccdf7f3d899993115fe704ba94fdd0050584",
}
EXPECTED_CONTROL_FILES = {
    "CPU_STAGE.json",
    "RECOVERY_CONTINUATION_PLAN.json",
    "batches/batch_02/AUTHORIZATION.json",
    "batches/batch_02/INVOCATION_LOCK/owner",
    "batches/batch_02/RELEASED",
    "batches/batch_02/RELEASE_AUTHORIZED",
    "batches/batch_02/STOPPED",
    "batches/batch_02/SUBMISSION_ATTEMPT.tsv",
    "batches/batch_02/SUBMISSION_LOCK/owner",
    "batches/batch_02/SUBMITTED",
}

REQUIRED_IMPLEMENTATION_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_batch2_result_recovery_v1.yaml",
    "docs/massive_medical_kalai_s1_batch2_result_recovery_v1_protocol.md",
    "scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py",
    "scripts/run_massive_medical_kalai_s1_batch2_result_recovery_v1_tillicum.sh",
    "scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
    "scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
    "tests/test_massive_medical_kalai_s1_batch2_result_recovery_v1.py",
)


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def seal(body):
    result = dict(body)
    result.pop(SEAL_FIELD, None)
    result[SEAL_FIELD] = sha256_bytes(canonical_bytes(result))
    return result


def load_json(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not an object")
    return value


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} seal differs")
    return body


def _write_new(path, payload, description):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.path.lexists(path):
        raise ValueError(f"{description} already exists")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_idempotent(path, payload, description):
    path = Path(path)
    if os.path.lexists(path):
        observed = load_json(path, description)
        verify_seal(observed, description)
        if observed != payload:
            raise ValueError(f"existing {description} differs")
        return "AUDITED"
    _write_new(path, payload, description)
    return "CREATED"


def git_commit(repo):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def git_branch(repo):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo), "branch", "--show-current"], text=True
    ).strip()


def require_clean(repo, description):
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError(f"{description} repository is not clean")


def binding(path, root=None, *, sealed=False, description="artifact"):
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError(f"{description} is absent or unsafe")
    path = raw_path.resolve()
    value = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }
    if root is not None:
        value["relative_path"] = path.relative_to(Path(root).resolve()).as_posix()
    if sealed:
        payload = load_json(path, description)
        verify_seal(payload, description)
        value["payload_sha256"] = payload[SEAL_FIELD]
    return value


def _paths_overlap(first, second):
    first, second = Path(first).resolve(), Path(second).resolve()
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def require_disjoint(*paths):
    paths = [Path(item).resolve() for item in paths]
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            if _paths_overlap(first, second):
                raise ValueError("source and recovery namespaces overlap")


def _regular_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact tree is absent or unsafe: {root}")
    files = []
    for directory, names, filenames in os.walk(root):
        directory = Path(directory)
        if directory.is_symlink() or any((directory / name).is_symlink() for name in names):
            raise ValueError("artifact tree contains a symlink directory")
        for name in filenames:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("artifact tree contains an unsafe file")
            files.append(path)
    return sorted(files)


def tree_manifest(root):
    root = Path(root).resolve()
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _regular_files(root)
    ]
    return {
        "entries": entries,
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "manifest_sha256": sha256_bytes(canonical_bytes(entries)),
    }


def _control_values(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source control record is absent or unsafe")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise ValueError("source control record is malformed")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError("source control record key is malformed")
        result[key] = value
    return result


def _audit_job(source_output):
    log_root = Path(source_output).resolve().parent / "logs"
    logs = {}
    for stream, (name, size, digest) in EXPECTED_LOGS.items():
        path = log_root / name
        observed = binding(path, description=f"source {stream} log")
        if observed["size_bytes"] != size or observed["file_sha256"] != digest:
            raise ValueError(f"source {stream} log pin differs")
        logs[stream] = observed
    stdout = Path(logs["stdout"]["path"]).read_text(encoding="utf-8")
    stderr = Path(logs["stderr"]["path"]).read_text(encoding="utf-8")
    required_stdout = (
        '"status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_GENERATED"',
        '"status": "KALAI_S1_RECOVERY_CONTINUATION_BATCH_AUDITED"',
        EXPECTED_KEY_ARTIFACTS[
            "generation/completion_batches/batch_02/combined_timing.json"
        ][2],
    )
    required_stderr = (
        "ValueError: completion-batch shard binding differs",
        "manager.original_runtime._audit_batch",
        "runtime._call_with_continuation_protocol",
    )
    if any(item not in stdout for item in required_stdout) or any(
        item not in stderr for item in required_stderr
    ):
        raise ValueError("source logs do not prove generation, audit, and evaluator defect")
    command = [
        "sacct", "-n", "-X", "-P", "-j", SOURCE_JOB_ID,
        f"--format={','.join(SACCT_FIELDS)}",
    ]
    raw = subprocess.check_output(command, text=True).strip()
    if raw != EXPECTED_SACCT_RECORD:
        raise ValueError("source Slurm accounting record differs")
    return {
        "sacct_fields": list(SACCT_FIELDS),
        "sacct_record": raw,
        "parsed_record": dict(zip(SACCT_FIELDS, raw.split("|"))),
        "logs": logs,
    }


def expected_job_evidence(source_output):
    log_root = Path(source_output).resolve().parent / "logs"
    return {
        "sacct_fields": list(SACCT_FIELDS),
        "sacct_record": EXPECTED_SACCT_RECORD,
        "parsed_record": dict(zip(SACCT_FIELDS, EXPECTED_SACCT_RECORD.split("|"))),
        "logs": {
            stream: {
                "path": str(log_root / name),
                "size_bytes": size,
                "file_sha256": digest,
            }
            for stream, (name, size, digest) in EXPECTED_LOGS.items()
        },
    }


def _source_paths(source_output):
    output = Path(source_output).resolve()
    control = output / "control"
    batch = control / "batches" / SOURCE_BATCH_ID
    generation = output / "generation" / "completion_batches" / SOURCE_BATCH_ID
    return {
        "output": output,
        "control": control,
        "batch_control": batch,
        "plan": control / "RECOVERY_CONTINUATION_PLAN.json",
        "stage": control / "CPU_STAGE.json",
        "authorization": batch / "AUTHORIZATION.json",
        "stopped": batch / "STOPPED",
        "result": batch / "RESULT.json",
        "generation": generation,
    }


def audit_source(source_output_root, source_repo_root):
    if Path(source_output_root).is_symlink() or Path(source_repo_root).is_symlink():
        raise ValueError("source namespace is a symlink")
    source_output = Path(source_output_root).resolve()
    source_repo = Path(source_repo_root).resolve()
    paths = _source_paths(source_output)
    if source_output.name != SOURCE_OUTPUT_LEAF or source_repo.name != SOURCE_REPOSITORY_LEAF:
        raise ValueError("source namespace leaf differs")
    require_clean(source_repo, "source")
    if git_commit(source_repo) != SOURCE_REPOSITORY_COMMIT:
        raise ValueError("source repository commit differs")
    if os.path.lexists(paths["result"]):
        raise ValueError("source RESULT.json must remain absent")
    control_manifest = tree_manifest(paths["control"])
    generation_manifest = tree_manifest(paths["generation"])
    if {
        key: control_manifest[key] for key in EXPECTED_CONTROL_MANIFEST
    } != EXPECTED_CONTROL_MANIFEST:
        raise ValueError("source control manifest differs")
    if {item["relative_path"] for item in control_manifest["entries"]} != EXPECTED_CONTROL_FILES:
        raise ValueError("source control inventory differs")
    if {
        key: generation_manifest[key] for key in EXPECTED_GENERATION_MANIFEST
    } != EXPECTED_GENERATION_MANIFEST:
        raise ValueError("source generation manifest differs")
    key_bindings = {}
    for relative, (size, file_digest, payload_digest) in EXPECTED_KEY_ARTIFACTS.items():
        path = source_output / relative
        observed = binding(
            path, source_output, sealed=payload_digest is not None,
            description=f"source {relative}",
        )
        if observed["size_bytes"] != size or observed["file_sha256"] != file_digest:
            raise ValueError(f"source artifact pin differs: {relative}")
        if payload_digest is not None and observed["payload_sha256"] != payload_digest:
            raise ValueError(f"source payload pin differs: {relative}")
        key_bindings[relative] = observed
    stopped = _control_values(paths["stopped"])
    if stopped != {
        "stage": "recovery_continuation_batch",
        "batch_id": SOURCE_BATCH_ID,
        "job_id": SOURCE_JOB_ID,
        "exit_code": "1",
        "restart_or_resume_authorized": "false",
        "retry_or_replacement_authorized": "false",
        "automatic_next_batch_authorized": "false",
    }:
        raise ValueError("source STOPPED record differs")
    # This calls the corrected evaluator.  The audited function and patched
    # protocol now come from the same runtime.manager module instance.
    generation_audit = source_evaluator._generation_audit(
        source_output, source_repo, SOURCE_BATCH_INDEX
    )
    if generation_audit.get("combined_timing_payload_sha256") != (
        EXPECTED_KEY_ARTIFACTS[
            "generation/completion_batches/batch_02/combined_timing.json"
        ][2]
    ):
        raise ValueError("corrected generation audit seal differs")
    job = _audit_job(source_output)
    return {
        "paths": paths,
        "key_bindings": key_bindings,
        "control_manifest": control_manifest,
        "generation_manifest": generation_manifest,
        "generation_audit": generation_audit,
        "job_evidence": job,
    }


def _audit_recovery_namespace(output_root, allowed):
    raw_output = Path(output_root)
    if raw_output.is_symlink():
        raise ValueError("recovery output is a symlink")
    output = raw_output.resolve()
    if output.name != RECOVERY_OUTPUT_LEAF:
        raise ValueError("recovery output leaf differs")
    for name in ("generation", "assembled", "judge", "logs", "batches"):
        if os.path.lexists(output / name):
            raise ValueError(f"recovery namespace contains forbidden {name}")
    if output.exists() and {item.name for item in output.iterdir()} - {"control"}:
        raise ValueError("recovery namespace contains unexpected state")
    control = output / "control"
    if control.exists():
        if control.is_symlink() or not control.is_dir():
            raise ValueError("recovery control is unsafe")
        items = list(control.iterdir())
        if {item.name for item in items} - set(allowed):
            raise ValueError("recovery control inventory differs")
        if any(item.is_symlink() or not item.is_file() for item in items):
            raise ValueError("recovery control contains an unsafe entry")


def _policy():
    return {
        "derivation_only": True,
        "source_artifacts_read_only": True,
        "original_stopped_preserved": True,
        "generation_authorized": False,
        "gpu_jobs_authorized": 0,
        "external_api_calls_authorized": 0,
        "restart_resume_retry_replacement_requeue_authorized": False,
        "batch_3_submission_authorized": False,
        "judging_authorized": False,
    }


def _accounting():
    return {
        "known_program_actual_before_job_usd": float(KNOWN_PROGRAM_ACTUAL_BEFORE_USD),
        "source_job_actual_estimated_cost_usd": float(SOURCE_ACTUAL_COST_USD),
        "known_program_actual_after_job_usd": float(KNOWN_PROGRAM_ACTUAL_AFTER_USD),
        "conservative_exposure_before_job_authority_usd": float(CONSERVATIVE_EXPOSURE_BEFORE_USD),
        "source_job_authority_cap_retained_usd": float(SOURCE_AUTHORIZED_CAP_USD),
        "conservative_exposure_after_job_authority_usd": float(CONSERVATIVE_EXPOSURE_AFTER_USD),
        "new_recovery_cost_cap_usd": 0.0,
        "program_ceiling_usd": float(PROGRAM_CEILING_USD),
    }


def _plan_body(source, source_output, source_repo, output):
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch2_result_recovery_plan",
        "status": "CPU_ONLY_BATCH2_RECOVERY_PLANNED_NO_GPU_OR_API_AUTHORITY",
        "source_repository": {
            "path": str(source_repo), "commit": SOURCE_REPOSITORY_COMMIT, "clean": True,
        },
        "source_output_root": str(source_output),
        "source_job": {
            "batch_index": SOURCE_BATCH_INDEX,
            "batch_id": SOURCE_BATCH_ID,
            "slurm_job_id": SOURCE_JOB_ID,
            "scheduler_state": "FAILED",
            "scheduler_exit_code": "1:0",
            "scheduler_derived_exit_code": "0:0",
            "scheduler_elapsed_seconds": SOURCE_ELAPSED_SECONDS,
            "actual_estimated_cost_usd": float(SOURCE_ACTUAL_COST_USD),
            "authorized_cap_usd_retained": float(SOURCE_AUTHORIZED_CAP_USD),
        },
        "source_job_evidence": source["job_evidence"],
        "source_key_bindings": source["key_bindings"],
        "source_control_manifest": source["control_manifest"],
        "source_generation_manifest": source["generation_manifest"],
        "corrected_generation_audit": source["generation_audit"],
        "recovery_output_root": str(output),
        "recovery_policy": _policy(),
        "accounting": _accounting(),
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }


def prepare(args):
    _require_cpu_only()
    source_output = Path(args.source_output_root).resolve()
    source_repo = Path(args.source_repo_root).resolve()
    output = Path(args.output_root).resolve()
    require_disjoint(source_output, source_repo, output)
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME})
    source = audit_source(source_output, source_repo)
    plan = seal(_plan_body(source, source_output, source_repo, output))
    disposition = _write_idempotent(
        output / "control" / PLAN_NAME, plan, "batch-2 recovery plan"
    )
    print(json.dumps({
        "status": f"KALAI_S1_BATCH2_RESULT_RECOVERY_PLAN_{disposition}",
        "recovery_plan_payload_sha256": plan[SEAL_FIELD],
        "source_generation_files": EXPECTED_GENERATION_MANIFEST["file_count"],
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return plan


def load_plan(path, *, audit_source_state):
    path = Path(path).resolve()
    payload = load_json(path, "batch-2 recovery plan")
    body = verify_seal(payload, "batch-2 recovery plan")
    source_repo = Path(body.get("source_repository", {}).get("path", "")).resolve()
    source_output = Path(body.get("source_output_root", "")).resolve()
    output = Path(body.get("recovery_output_root", "")).resolve()
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "batch2_result_recovery_plan"
        or body.get("status") != "CPU_ONLY_BATCH2_RECOVERY_PLANNED_NO_GPU_OR_API_AUTHORITY"
        or body.get("source_repository") != {
            "path": str(source_repo), "commit": SOURCE_REPOSITORY_COMMIT, "clean": True,
        }
        or body.get("recovery_policy") != _policy()
        or body.get("accounting") != _accounting()
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
        or source_repo.name != SOURCE_REPOSITORY_LEAF
        or source_output.name != SOURCE_OUTPUT_LEAF
        or output.name != RECOVERY_OUTPUT_LEAF
        or path != output / "control" / PLAN_NAME
    ):
        raise ValueError("batch-2 recovery plan identity differs")
    require_disjoint(source_output, source_repo, output)
    if body.get("source_job_evidence") != expected_job_evidence(source_output):
        raise ValueError("batch-2 recovery job evidence differs")
    if audit_source_state:
        observed = audit_source(source_output, source_repo)
        expected = _plan_body(observed, source_output, source_repo, output)
        if body != expected:
            raise ValueError("batch-2 recovery source state differs from plan")
    return payload, body


def implementation_bindings(repo):
    repo = Path(repo).resolve()
    result = {}
    for relative in REQUIRED_IMPLEMENTATION_FILES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery implementation is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def _stage_body(repo, plan_path, plan):
    branch = git_branch(repo)
    if not branch:
        raise ValueError("recovery repository must be on a named branch")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch2_result_recovery_cpu_stage",
        "status": "CPU_STAGED_DERIVATION_ONLY_NO_GPU_OR_API_AUTHORITY",
        "repository": {
            "path": str(repo), "commit": git_commit(repo), "branch": branch, "clean": True,
        },
        "recovery_plan": binding(
            plan_path, sealed=True, description="batch-2 recovery plan"
        ),
        "implementation_sha256": implementation_bindings(repo),
        "execution_policy": _policy(),
        "new_cost_cap_usd": 0.0,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }


def stage(args):
    _require_cpu_only()
    output = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    source_output = Path(args.source_output_root).resolve()
    source_repo = Path(args.source_repo_root).resolve()
    if repo.name != RECOVERY_REPOSITORY_LEAF:
        raise ValueError("recovery repository leaf differs")
    require_disjoint(source_output, source_repo, output, repo)
    require_clean(repo, "recovery")
    plan = prepare(args)
    plan_path = output / "control" / PLAN_NAME
    stage_payload = seal(_stage_body(repo, plan_path, plan))
    disposition = _write_idempotent(
        output / "control" / STAGE_NAME, stage_payload, "batch-2 recovery CPU stage"
    )
    load_stage(output / "control" / STAGE_NAME, repo, audit_source_state=True)
    print(json.dumps({
        "status": f"KALAI_S1_BATCH2_RESULT_RECOVERY_CPU_STAGE_{disposition}",
        "cpu_stage_payload_sha256": stage_payload[SEAL_FIELD],
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return stage_payload


def load_stage(path, repo_root, *, audit_source_state):
    path = Path(path).resolve()
    repo = Path(repo_root).resolve()
    payload = load_json(path, "batch-2 recovery CPU stage")
    body = verify_seal(payload, "batch-2 recovery CPU stage")
    plan_path = path.parent / PLAN_NAME
    plan, plan_body = load_plan(plan_path, audit_source_state=audit_source_state)
    expected = _stage_body(repo, plan_path, plan)
    if body != expected or path != plan_path.parent / STAGE_NAME:
        raise ValueError("batch-2 recovery CPU stage differs")
    if body["repository"]["path"] != str(repo):
        raise ValueError("batch-2 recovery repository path differs")
    require_clean(repo, "recovery")
    require_disjoint(
        repo, plan_body["source_repository"]["path"],
        plan_body["source_output_root"], plan_body["recovery_output_root"],
    )
    return payload, body


def _expected_result(output, repo):
    plan_path = output / "control" / PLAN_NAME
    stage_path = output / "control" / STAGE_NAME
    plan, plan_body = load_plan(plan_path, audit_source_state=True)
    stage_payload, _ = load_stage(stage_path, repo, audit_source_state=True)
    source_output = Path(plan_body["source_output_root"])
    source_repo = Path(plan_body["source_repository"]["path"])
    source = audit_source(source_output, source_repo)
    audit = source["generation_audit"]
    corrected_source_result_body = source_evaluator._expected_result_body(
        source_output,
        source_repo,
        SOURCE_BATCH_INDEX,
        slurm_job_id=SOURCE_JOB_ID,
        elapsed_seconds=SOURCE_ELAPSED_SECONDS,
        audit=audit,
    )
    return seal({
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch2_result_recovery",
        "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERED_CPU_ONLY",
        "batch_index": SOURCE_BATCH_INDEX,
        "batch_id": SOURCE_BATCH_ID,
        "batch_valid": True,
        "source_job": plan_body["source_job"],
        "source_job_evidence": plan_body["source_job_evidence"],
        "recovery_reason": {
            "category": "post_generation_cpu_evaluator_module_instance_error",
            "failed_error": "completion-batch shard binding differs",
            "correction": (
                "patch and audited function use the same runtime.manager."
                "original_runtime module instance"
            ),
            "generation_completed_before_error": True,
            "generation_time_audits_passed_before_error": True,
            "scientific_outputs_regenerated": False,
        },
        "recovery_plan": binding(plan_path, sealed=True, description="recovery plan"),
        "cpu_stage": binding(stage_path, sealed=True, description="recovery CPU stage"),
        "source_key_bindings": source["key_bindings"],
        "source_control_manifest": source["control_manifest"],
        "source_generation_manifest": source["generation_manifest"],
        "corrected_generation_audit": audit,
        "corrected_source_result_body": corrected_source_result_body,
        "source_stopped_preserved": True,
        "source_result_absent": True,
        "derivation_only": True,
        "recovered_predecessor_evidence_valid": True,
        "automatic_next_batch_authorized": False,
        "batch_3_submission_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_replacement_or_requeue_authorized": False,
        "generation_authorized": False,
        "judging_authorized": False,
        "accounting": _accounting(),
        "external_api_calls": 0,
        "gpu_jobs": 0,
    })


def recover(args):
    _require_cpu_only()
    output, repo = Path(args.output_root).resolve(), Path(args.repo_root).resolve()
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME})
    result_path = output / "control" / RESULT_NAME
    if os.path.lexists(result_path):
        raise ValueError("recovered result already exists; rerun is forbidden")
    _, plan_body = load_plan(output / "control" / PLAN_NAME, audit_source_state=True)
    before = audit_source(
        plan_body["source_output_root"], plan_body["source_repository"]["path"]
    )
    result = _expected_result(output, repo)
    _write_new(result_path, result, "recovered batch-2 result")
    after = audit_source(
        plan_body["source_output_root"], plan_body["source_repository"]["path"]
    )
    for key in ("key_bindings", "control_manifest", "generation_manifest", "job_evidence"):
        if after[key] != before[key]:
            raise ValueError("source state changed during recovery")
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME, RESULT_NAME})
    print(json.dumps({
        "status": result["status"],
        "recovered_result_payload_sha256": result[SEAL_FIELD],
        "source_job_id": SOURCE_JOB_ID,
        "source_stopped_preserved": True,
        "batch_3_submission_authorized": False,
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return result


def audit_recovery(args):
    _require_cpu_only()
    output, repo = Path(args.output_root).resolve(), Path(args.repo_root).resolve()
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME, RESULT_NAME})
    observed = load_json(output / "control" / RESULT_NAME, "recovered batch-2 result")
    verify_seal(observed, "recovered batch-2 result")
    expected = _expected_result(output, repo)
    if observed != expected:
        raise ValueError("recovered batch-2 result differs")
    print(json.dumps({
        "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERY_AUDITED",
        "recovered_result_payload_sha256": observed[SEAL_FIELD],
        "source_stopped_preserved": True,
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return observed


def _require_cpu_only():
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent from CPU recovery")


def self_test():
    assert SOURCE_ACTUAL_COST_USD == (
        Decimal(SOURCE_ELAPSED_SECONDS) * H200_HOURLY_USD / Decimal(3600)
    )
    assert KNOWN_PROGRAM_ACTUAL_BEFORE_USD + SOURCE_ACTUAL_COST_USD == KNOWN_PROGRAM_ACTUAL_AFTER_USD
    assert CONSERVATIVE_EXPOSURE_BEFORE_USD + SOURCE_AUTHORIZED_CAP_USD == CONSERVATIVE_EXPOSURE_AFTER_USD
    assert source_evaluator.runtime.manager is not source_evaluator.manager
    source = Path(
        source_evaluator.__file__
    ).read_text(encoding="utf-8")
    assert "runtime.manager.original_runtime._audit_batch" in source
    print("MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERY_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-root")
    parser.add_argument("--source-repo-root")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--stage", action="store_true")
    modes.add_argument("--recover", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.recover or args.audit_only:
        if not args.output_root or not args.repo_root:
            parser.error("--output-root and --repo-root are required")
        (recover if args.recover else audit_recovery)(args)
        return 0
    required = (
        args.source_output_root, args.source_repo_root,
        args.output_root, args.repo_root if args.stage else True,
    )
    if not all(required):
        parser.error("source and recovery paths are required")
    (stage if args.stage else prepare)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
