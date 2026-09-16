#!/usr/bin/env python3
"""Recover the complete Kalai s=1 batch-7 generation after a Slurm timeout.

Job 273276 finished every scientific generation artifact before Slurm killed
the post-generation CPU audit.  This workflow is deliberately derivation
only: it snapshots and deeply audits the immutable source namespaces, records
the TIMEOUT as TIMEOUT, and writes a separately named ``RECOVERED_RESULT``.
It never creates a source ``RESULT.json`` and has no GPU or API-call path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unattended = _load(
    "_kalai_s1_batch7_recovery_unattended",
    "manage_massive_medical_kalai_s1_recovery_continuation_unattended_v2.py",
)
v3_manager = unattended.v3_manager
v3_authorizer = unattended.v3_authorizer
v3_evaluator = unattended.v3_evaluator

PROTOCOL_ID = "massive_medical_kalai_s1_batch7_result_recovery_v1"
SOURCE_PROTOCOL_ID = v3_manager.PROTOCOL_ID
SOURCE_PARENT_PROTOCOL_ID = unattended.PROTOCOL_ID
METHOD_ID = v3_manager.METHOD_ID
SEAL_FIELD = v3_manager.SEAL_FIELD

BATCH_INDEX = 7
BATCH_ID = "batch_07"
SOURCE_JOB_ID = "273276"
SOURCE_SCHEDULER_STATE = "TIMEOUT"
SOURCE_SCHEDULER_ELAPSED_SECONDS = 3605
SOURCE_V3_REPOSITORY_COMMIT = "84653e82cc2181347bddaa73bced65bcbdc5126f"
SOURCE_UNATTENDED_REPOSITORY_COMMIT = (
    "d5c85f78f033f1b6f94bd997ff8be7439811b748"
)
SOURCE_V3_REPOSITORY_LEAF = v3_manager.EXPECTED_REPO_LEAF
SOURCE_V3_OUTPUT_LEAF = v3_manager.EXPECTED_OUTPUT_LEAF
SOURCE_UNATTENDED_REPOSITORY_LEAF = unattended.EXPECTED_REPO_LEAF
SOURCE_UNATTENDED_OUTPUT_LEAF = unattended.EXPECTED_OUTPUT_LEAF
RECOVERY_REPOSITORY_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1"
)
RECOVERY_OUTPUT_LEAF = PROTOCOL_ID

EXPECTED_COMBINED_TIMING_PAYLOAD_SHA256 = (
    "795555507e96f18e69125e787b6a5c20152be811a58737450e94f800127cc6c8"
)
EXPECTED_BENEFIT_GENERATION_PAYLOAD_SHA256 = (
    "844c84b99117d46abe9d99d5850aa655730e28e0041dbe5c095d74dfc4fcfbad"
)
EXPECTED_MEDICAL_GENERATION_PAYLOAD_SHA256 = (
    "cb0eb7e8596fa5d3eb907a272563bafb58176506902b448e8730e41f0a235ae2"
)
EXPECTED_PHASE_SUMMARIES = {
    "benefit": {"requested_n": 16, "accepted_n": 10, "abstained_n": 6},
    "medical": {"requested_n": 9, "accepted_n": 0, "abstained_n": 9},
}
EXPECTED_GENERATION_MANIFEST = {
    "file_count": 30,
    "size_bytes": 548309,
    "manifest_sha256": (
        "574fc31cefe48f10d5860a05aec2d5cea7eba7bafdd7474dfe08ae6c1a79c0d6"
    ),
}
EXPECTED_CONTROL_MANIFEST = {
    "file_count": 8,
    "size_bytes": 6322,
    "manifest_sha256": (
        "7f3ab2903e6fd0b89ee4a3347a9c9d3ecb8b4b01bfe7fc3b546f6d9534874494"
    ),
}
EXPECTED_LOGS = {
    "stdout": {
        "size_bytes": 720932,
        "file_sha256": (
            "025dda98abb6272f7b9a92a3fc8fa003781506c1fa7666bae5eb53281b9df6f8"
        ),
    },
    "stderr": {
        "size_bytes": 3538,
        "file_sha256": (
            "fa6076e070fd7153871d7192970382ced6b99764d521989a1092ba616d7d7583"
        ),
    },
}

PLAN_NAME = "RECOVERY_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
RESULT_NAME = "RECOVERED_RESULT.json"

REQUIRED_IMPLEMENTATION_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_batch7_result_recovery_v1.yaml",
    "docs/massive_medical_kalai_s1_batch7_result_recovery_v1_protocol.md",
    "scripts/manage_massive_medical_kalai_s1_batch7_result_recovery_v1.py",
    "scripts/assemble_score_massive_medical_kalai_s1_batch7_result_recovery_v1.py",
    "scripts/run_massive_medical_kalai_s1_batch7_result_recovery_v1_tillicum.sh",
    "tests/test_massive_medical_kalai_s1_batch7_result_recovery_v1.py",
)

canonical_bytes = v3_manager.canonical_bytes
sha256_bytes = v3_manager.sha256_bytes
sha256_file = v3_manager.sha256_file
seal = v3_manager.seal
verify_seal = v3_manager.verify_seal
load_json = v3_manager.load_json
binding = v3_manager.binding
raw_binding = v3_manager.raw_binding
_write_new = v3_manager._write_new
_write_idempotent = v3_manager._write_idempotent
_assert_disjoint = v3_manager._assert_disjoint


def git_commit(repo_root: Path | str) -> str:
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def git_branch(repo_root: Path | str) -> str:
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "branch", "--show-current"],
        text=True,
    ).strip()


def require_clean(repo_root: Path | str, description: str) -> None:
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise ValueError(f"{description} repository is not clean")


def _require_cpu_only() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent from CPU recovery")


def _parse_kv(path: Path, description: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.count("=") != 1:
            raise ValueError(f"{description} contains a malformed line")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"{description} keys differ")
        result[key] = value
    return result


def parse_timeout_sacct(path: Path | str) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("batch-7 sacct evidence is absent or unsafe")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "JobIDRaw|JobName|State|ExitCode|DerivedExitCode|ElapsedRaw|AllocTRES"
    if len(lines) != 2 or lines[0] != header:
        raise ValueError("batch-7 sacct evidence shape differs")
    fields = lines[1].split("|")
    if len(fields) != 7:
        raise ValueError("batch-7 sacct evidence field count differs")
    job_id, job_name, state, exit_code, derived, elapsed, alloc_tres = fields
    tokens = set(alloc_tres.split(","))
    if (
        job_id != SOURCE_JOB_ID
        or job_name != "mmu_kalai_s1_r3c07"
        or state != SOURCE_SCHEDULER_STATE
        or exit_code != "0:0"
        or derived != "0:0"
        or elapsed != str(SOURCE_SCHEDULER_ELAPSED_SECONDS)
        or not {
            "cpu=8",
            "mem=200G",
            "node=1",
            "gres/gpu=1",
            "gres/gpu:h200=1",
        }.issubset(tokens)
    ):
        raise ValueError("batch-7 sacct TIMEOUT evidence differs")
    return {
        "job_id": job_id,
        "job_name": job_name,
        "state": state,
        "exit_code": exit_code,
        "derived_exit_code": derived,
        "elapsed_seconds": int(elapsed),
        "alloc_tres": alloc_tres,
    }


def _manifest(root: Path, description: str) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{description} root is absent or unsafe")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{description} contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{description} contains an unsafe entry")
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "root": str(root.resolve()),
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "manifest_sha256": sha256_bytes(canonical_bytes(entries)),
        "entries": entries,
    }


def _require_exact_leaf(path: Path, leaf: str, description: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{description} is a symlink")
    resolved = raw.resolve()
    if resolved.name != leaf:
        raise ValueError(f"{description} leaf differs")
    return resolved


def _audit_scientific_control(source_output: Path, source_repo: Path) -> dict:
    control = source_output / "control" / "batches" / BATCH_ID
    expected = {
        "SUBMISSION_LOCK",
        "AUTHORIZATION.json",
        "SUBMISSION_ATTEMPT.tsv",
        "SUBMITTED",
        "RELEASE_AUTHORIZED",
        "RELEASED",
        "INVOCATION_LOCK",
        "STOPPED",
    }
    if (
        control.is_symlink()
        or not control.is_dir()
        or {item.name for item in control.iterdir()} != expected
        or os.path.lexists(control / "RESULT.json")
    ):
        raise ValueError("batch-7 scientific control inventory differs")
    stopped = _parse_kv(control / "STOPPED", "batch-7 scientific STOPPED")
    if stopped != {
        "stage": "recovery_continuation_batch",
        "batch_id": BATCH_ID,
        "job_id": SOURCE_JOB_ID,
        "exit_code": "0",
        "restart_or_resume_authorized": "false",
        "retry_or_replacement_authorized": "false",
        "automatic_next_batch_authorized": "false",
    }:
        raise ValueError("batch-7 scientific STOPPED evidence differs")
    authorization_path = control / "AUTHORIZATION.json"
    authorization = load_json(authorization_path, "batch-7 authorization")
    body = verify_seal(authorization, "batch-7 authorization")
    if body != v3_authorizer.expected_body(
        source_output, source_repo, BATCH_INDEX, body.get("created_at")
    ):
        raise ValueError("batch-7 authorization differs")
    if body.get("repository_commit") != SOURCE_V3_REPOSITORY_COMMIT:
        raise ValueError("batch-7 authorization repository commit differs")
    manifest = _manifest(control, "batch-7 scientific control")
    if {
        key: manifest[key] for key in EXPECTED_CONTROL_MANIFEST
    } != EXPECTED_CONTROL_MANIFEST:
        raise ValueError("batch-7 scientific control manifest differs")
    return {
        "control_root": str(control),
        "authorization": binding(authorization_path, authorization),
        "stopped": raw_binding(control / "STOPPED"),
        "result_absent": True,
        "manifest": manifest,
    }


def _audit_unattended_terminal(
    output: Path, repo: Path, v3: dict
) -> dict:
    stopped_path = output / "control" / "SEQUENCE_STOPPED"
    stopped = _parse_kv(stopped_path, "unattended-v2 SEQUENCE_STOPPED")
    if stopped != {
        "protocol_id": SOURCE_PARENT_PROTOCOL_ID,
        "status": "HARD_STOPPED",
        "active_batch": "7",
        "active_job": SOURCE_JOB_ID,
        "exit_code": "1",
        "restart_resume_retry_replacement_requeue_authorized": "false",
        "external_api_calls_authorized": "0",
    }:
        raise ValueError("unattended-v2 hard-stop evidence differs")
    prior = []
    for index in range(3, 7):
        receipt, _ = unattended._load_receipt(output, repo, v3, index, deep=True)
        path = unattended.controller_batch_root(output, index) / unattended.RECEIPT_NAME
        prior.append(
            {
                "batch_index": index,
                "kind": "completed_terminal_receipt",
                "artifact": binding(path, receipt),
            }
        )
    target = unattended.controller_batch_root(output, BATCH_INDEX)
    expected = {unattended.CHILD_AUTHORITY_NAME, unattended.SACCT_NAME}
    if (
        target.is_symlink()
        or not target.is_dir()
        or {item.name for item in target.iterdir()} != expected
        or os.path.lexists(target / unattended.RECEIPT_NAME)
    ):
        raise ValueError("unattended-v2 batch-7 terminal inventory differs")
    child = load_json(target / unattended.CHILD_AUTHORITY_NAME, "batch-7 child authority")
    child_body = verify_seal(child, "batch-7 child authority")
    plan = load_json(output / "control" / unattended.PLAN_NAME, "unattended plan")
    plan_body = verify_seal(plan, "unattended plan")
    authority = load_json(
        output / "control" / unattended.AUTHORITY_NAME,
        "unattended standing authority",
    )
    verify_seal(authority, "unattended standing authority")
    invocation, _ = unattended._load_invocation(output, authority)
    expected_child = unattended._expected_child_body(
        output, plan, plan_body, authority, invocation, v3, BATCH_INDEX
    )
    if child_body != expected_child:
        raise ValueError("unattended-v2 batch-7 child authority differs")
    sacct_path = target / unattended.SACCT_NAME
    return {
        "sequence_stopped": raw_binding(stopped_path),
        "completed_batches_3_through_6": prior,
        "batch_7_child_authority": binding(
            target / unattended.CHILD_AUTHORITY_NAME, child
        ),
        "batch_7_sacct": raw_binding(sacct_path),
        "batch_7_scheduler": parse_timeout_sacct(sacct_path),
        "batch_7_terminal_receipt_absent": True,
    }


def _audit_logs(log_root: Path) -> dict:
    if log_root.is_symlink() or not log_root.is_dir():
        raise ValueError("source log root is absent or unsafe")
    base = f"massive_medical_kalai_s1_recovery_continuation_v3_batch_{SOURCE_JOB_ID}"
    stdout = log_root / f"{base}.out"
    stderr = log_root / f"{base}.err"
    for path, description in ((stdout, "batch-7 stdout"), (stderr, "batch-7 stderr")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{description} is absent or unsafe")
    error_text = stderr.read_text(encoding="utf-8", errors="strict")
    if "TIME LIMIT" not in error_text or SOURCE_JOB_ID not in error_text:
        raise ValueError("batch-7 stderr does not bind the Slurm time limit")
    stdout_text = stdout.read_text(encoding="utf-8", errors="strict")
    if (
        "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_GENERATED" not in stdout_text
        or EXPECTED_COMBINED_TIMING_PAYLOAD_SHA256 not in stdout_text
        or "KALAI_S1_RECOVERY_CONTINUATION_V3_BATCH_AUDITED" not in stdout_text
    ):
        raise ValueError("batch-7 stdout omits complete-generation audit proof")
    result = {"stdout": raw_binding(stdout), "stderr": raw_binding(stderr)}
    for name, expected in EXPECTED_LOGS.items():
        if any(result[name].get(key) != value for key, value in expected.items()):
            raise ValueError(f"batch-7 {name} binding differs")
    return result


def audit_sources(args) -> dict:
    source_output = _require_exact_leaf(
        Path(args.source_output_root), SOURCE_V3_OUTPUT_LEAF, "source v3 output"
    )
    source_repo = _require_exact_leaf(
        Path(args.source_repo_root), SOURCE_V3_REPOSITORY_LEAF, "source v3 repository"
    )
    parent_output = _require_exact_leaf(
        Path(args.unattended_output_root),
        SOURCE_UNATTENDED_OUTPUT_LEAF,
        "source unattended-v2 output",
    )
    parent_repo = _require_exact_leaf(
        Path(args.unattended_repo_root),
        SOURCE_UNATTENDED_REPOSITORY_LEAF,
        "source unattended-v2 repository",
    )
    log_root = Path(args.log_root).resolve()
    _assert_disjoint(source_output, source_repo, parent_output, parent_repo, log_root)
    require_clean(source_repo, "source v3")
    require_clean(parent_repo, "source unattended-v2")
    if git_commit(source_repo) != SOURCE_V3_REPOSITORY_COMMIT:
        raise ValueError("source v3 repository commit differs")
    if git_commit(parent_repo) != SOURCE_UNATTENDED_REPOSITORY_COMMIT:
        raise ValueError("source unattended-v2 repository commit differs")

    _, _, _, _, _, _, _, _, v3 = unattended._load_workflow(
        parent_output, parent_repo
    )
    if v3["output"] != source_output or v3["repo"] != source_repo:
        raise ValueError("unattended-v2 anchor does not bind the source v3 workflow")
    parent_terminal = _audit_unattended_terminal(parent_output, parent_repo, v3)
    scientific_control = _audit_scientific_control(source_output, source_repo)
    audit = v3_evaluator._generation_audit(source_output, source_repo, BATCH_INDEX)
    if (
        audit.get("combined_timing_payload_sha256")
        != EXPECTED_COMBINED_TIMING_PAYLOAD_SHA256
        or audit.get("phases", {}).get("benefit", {}).get(
            "generation_payload_sha256"
        )
        != EXPECTED_BENEFIT_GENERATION_PAYLOAD_SHA256
        or audit.get("phases", {}).get("medical", {}).get(
            "generation_payload_sha256"
        )
        != EXPECTED_MEDICAL_GENERATION_PAYLOAD_SHA256
    ):
        raise ValueError("batch-7 sealed generation identities differ")
    for phase, expected in EXPECTED_PHASE_SUMMARIES.items():
        summary = audit["phases"][phase]["summary"]
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError(f"batch-7 {phase} summary differs")
    generation_root = (
        source_output / "generation" / "completion_batches" / BATCH_ID
    )
    generation_manifest = _manifest(generation_root, "batch-7 generation")
    if {
        key: generation_manifest[key] for key in EXPECTED_GENERATION_MANIFEST
    } != EXPECTED_GENERATION_MANIFEST:
        raise ValueError("batch-7 generation manifest differs")
    return {
        "source_v3_repository": {
            "path": str(source_repo),
            "commit": SOURCE_V3_REPOSITORY_COMMIT,
            "clean": True,
        },
        "source_v3_output_root": str(source_output),
        "source_unattended_repository": {
            "path": str(parent_repo),
            "commit": SOURCE_UNATTENDED_REPOSITORY_COMMIT,
            "clean": True,
        },
        "source_unattended_output_root": str(parent_output),
        "unattended_terminal": parent_terminal,
        "scientific_control": scientific_control,
        "generation_audit": audit,
        "generation_manifest": generation_manifest,
        "job_logs": _audit_logs(log_root),
    }


def _policy() -> dict:
    return {
        "derivation_only": True,
        "source_namespaces_read_only": True,
        "source_timeout_and_stopped_state_preserved": True,
        "source_result_must_remain_absent": True,
        "scientific_generation_regenerated": False,
        "restart_resume_retry_replacement_requeue_authorized": False,
        "gpu_jobs_authorized": 0,
        "external_api_calls_authorized": 0,
        "judging_authorized": False,
    }


def _audit_recovery_namespace(
    output_root: Path | str,
    allowed: set[str],
    *,
    allowed_top: set[str] | None = None,
) -> Path:
    raw = Path(output_root)
    if raw.is_symlink():
        raise ValueError("recovery output is a symlink")
    output = raw.resolve()
    if output.name != RECOVERY_OUTPUT_LEAF:
        raise ValueError("recovery output leaf differs")
    for forbidden in ("generation", "logs", "batches"):
        if os.path.lexists(output / forbidden):
            raise ValueError(f"recovery namespace contains forbidden {forbidden}")
    allowed_top = {"control"} if allowed_top is None else set(allowed_top)
    if "control" not in allowed_top:
        raise ValueError("recovery top-level policy must include control")
    if output.exists() and {item.name for item in output.iterdir()} - allowed_top:
        raise ValueError("recovery output top-level inventory differs")
    control = output / "control"
    if control.exists():
        if control.is_symlink() or not control.is_dir():
            raise ValueError("recovery control root is unsafe")
        unexpected = {item.name for item in control.iterdir()} - allowed
        if unexpected:
            raise ValueError(f"recovery control inventory differs: {sorted(unexpected)}")
    return output


def _plan_body(source: dict, output: Path) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_parent_protocol_id": SOURCE_PARENT_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch7_result_recovery_plan",
        "status": "CPU_ONLY_BATCH7_RECOVERY_PLANNED_NO_GPU_OR_API_AUTHORITY",
        "batch_index": BATCH_INDEX,
        "batch_id": BATCH_ID,
        "source_job_id": SOURCE_JOB_ID,
        "source": source,
        "recovery_output_root": str(output),
        "policy": _policy(),
        "new_cost_cap_usd": 0.0,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }


def prepare(args):
    _require_cpu_only()
    output = _audit_recovery_namespace(args.output_root, {PLAN_NAME, STAGE_NAME})
    source_roots = (
        args.source_output_root,
        args.source_repo_root,
        args.unattended_output_root,
        args.unattended_repo_root,
        args.log_root,
    )
    _assert_disjoint(output, *source_roots)
    source = audit_sources(args)
    payload = seal(_plan_body(source, output))
    disposition = _write_idempotent(
        output / "control" / PLAN_NAME, payload, "batch-7 recovery plan"
    )
    print(
        json.dumps(
            {
                "status": f"KALAI_S1_BATCH7_RECOVERY_PLAN_{disposition}",
                "generation_files": source["generation_manifest"]["file_count"],
                "scheduler_state": SOURCE_SCHEDULER_STATE,
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return payload


def load_plan(path: Path | str, *, audit_source_state: bool):
    path = Path(path).resolve()
    payload = load_json(path, "batch-7 recovery plan")
    body = verify_seal(payload, "batch-7 recovery plan")
    output = Path(body.get("recovery_output_root", "")).resolve()
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("source_parent_protocol_id") != SOURCE_PARENT_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "batch7_result_recovery_plan"
        or body.get("status")
        != "CPU_ONLY_BATCH7_RECOVERY_PLANNED_NO_GPU_OR_API_AUTHORITY"
        or body.get("batch_index") != BATCH_INDEX
        or body.get("batch_id") != BATCH_ID
        or body.get("source_job_id") != SOURCE_JOB_ID
        or body.get("policy") != _policy()
        or body.get("new_cost_cap_usd") != 0.0
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
        or path != output / "control" / PLAN_NAME
    ):
        raise ValueError("batch-7 recovery plan identity differs")
    if audit_source_state:
        source = body.get("source", {})
        args = argparse.Namespace(
            source_output_root=source.get("source_v3_output_root"),
            source_repo_root=source.get("source_v3_repository", {}).get("path"),
            unattended_output_root=source.get("source_unattended_output_root"),
            unattended_repo_root=source.get("source_unattended_repository", {}).get(
                "path"
            ),
            log_root=Path(source.get("job_logs", {}).get("stdout", {}).get("path", "")).parent,
        )
        if source != audit_sources(args):
            raise ValueError("batch-7 source state differs from recovery plan")
    return payload, body


def implementation_bindings(repo_root: Path | str) -> dict[str, str]:
    repo = Path(repo_root).resolve()
    result = {}
    for relative in REQUIRED_IMPLEMENTATION_FILES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery implementation is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def _stage_body(repo: Path, output: Path, plan: dict) -> dict:
    branch = git_branch(repo)
    if not branch:
        raise ValueError("recovery repository must be on a named branch")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "batch7_result_recovery_cpu_stage",
        "status": "CPU_STAGED_DERIVATION_ONLY_NO_GPU_OR_API_AUTHORITY",
        "repository": {
            "path": str(repo),
            "leaf": RECOVERY_REPOSITORY_LEAF,
            "commit": git_commit(repo),
            "branch": branch,
            "clean": True,
        },
        "recovery_plan": binding(output / "control" / PLAN_NAME, plan),
        "implementation_sha256": implementation_bindings(repo),
        "policy": _policy(),
        "new_cost_cap_usd": 0.0,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }


def stage(args):
    _require_cpu_only()
    output = Path(args.output_root).resolve()
    repo = _require_exact_leaf(
        Path(args.repo_root), RECOVERY_REPOSITORY_LEAF, "recovery repository"
    )
    require_clean(repo, "recovery")
    _assert_disjoint(
        repo,
        output,
        args.source_output_root,
        args.source_repo_root,
        args.unattended_output_root,
        args.unattended_repo_root,
        args.log_root,
    )
    plan = prepare(args)
    payload = seal(_stage_body(repo, output, plan))
    disposition = _write_idempotent(
        output / "control" / STAGE_NAME, payload, "batch-7 recovery CPU stage"
    )
    load_stage(output / "control" / STAGE_NAME, repo, audit_source_state=True)
    print(
        json.dumps(
            {
                "status": f"KALAI_S1_BATCH7_RECOVERY_CPU_STAGE_{disposition}",
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return payload


def load_stage(path: Path | str, repo_root: Path | str, *, audit_source_state: bool):
    path = Path(path).resolve()
    repo = Path(repo_root).resolve()
    payload = load_json(path, "batch-7 recovery CPU stage")
    body = verify_seal(payload, "batch-7 recovery CPU stage")
    plan_path = path.parent / PLAN_NAME
    plan, plan_body = load_plan(plan_path, audit_source_state=audit_source_state)
    expected = _stage_body(repo, plan_path.parents[1], plan)
    if body != expected or path != plan_path.parent / STAGE_NAME:
        raise ValueError("batch-7 recovery CPU stage differs")
    if body.get("repository", {}).get("path") != str(repo):
        raise ValueError("batch-7 recovery repository path differs")
    require_clean(repo, "recovery")
    _assert_disjoint(
        repo,
        plan_body["recovery_output_root"],
        plan_body["source"]["source_v3_output_root"],
        plan_body["source"]["source_v3_repository"]["path"],
        plan_body["source"]["source_unattended_output_root"],
        plan_body["source"]["source_unattended_repository"]["path"],
    )
    return payload, body


def _expected_result(output: Path, repo: Path) -> dict:
    plan_path = output / "control" / PLAN_NAME
    stage_path = output / "control" / STAGE_NAME
    plan, body = load_plan(plan_path, audit_source_state=True)
    stage_payload, _ = load_stage(stage_path, repo, audit_source_state=True)
    source = body["source"]
    return seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "source_parent_protocol_id": SOURCE_PARENT_PROTOCOL_ID,
            "method_id": METHOD_ID,
            "stage": "batch7_result_recovery",
            "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH7_RESULT_RECOVERED_CPU_ONLY",
            "batch_index": BATCH_INDEX,
            "batch_id": BATCH_ID,
            "scientific_generation_valid": True,
            "source_job": source["unattended_terminal"]["batch_7_scheduler"],
            "source_scheduler_estimated_cost_usd": 0.90125,
            "source_authorized_cap_usd": 0.9,
            "scheduler_estimate_exceeds_retained_cap_usd": 0.00125,
            "source_job_terminated_by_scheduler_timeout": True,
            "source_result_absent": True,
            "source_stopped_preserved": True,
            "source_terminal_receipt_absent": True,
            "source_generation_regenerated": False,
            "source_timeout_reclassified_as_completed": False,
            "recovery_result_is_not_a_source_result": True,
            "recovery_reason": {
                "category": "scheduler_timeout_after_complete_scientific_generation",
                "all_expected_generation_artifacts_deeply_audited": True,
                "post_generation_cpu_finalization_did_not_complete": True,
            },
            "recovery_plan": binding(plan_path, plan),
            "cpu_stage": binding(stage_path, stage_payload),
            "source_scientific_control": source["scientific_control"],
            "source_unattended_terminal": source["unattended_terminal"],
            "source_generation_audit": source["generation_audit"],
            "source_generation_manifest": source["generation_manifest"],
            "source_job_logs": source["job_logs"],
            "policy": _policy(),
            "new_gpu_cost_usd": 0.0,
            "new_api_cost_usd": 0.0,
            "external_api_calls": 0,
            "gpu_jobs": 0,
        }
    )


def recover(args):
    _require_cpu_only()
    output = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME})
    result_path = output / "control" / RESULT_NAME
    if os.path.lexists(result_path):
        raise ValueError("recovered batch-7 result already exists; rerun is forbidden")
    result = _expected_result(output, repo)
    _write_new(result_path, result, "recovered batch-7 result")
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME, RESULT_NAME})
    print(
        json.dumps(
            {
                "status": result["status"],
                "recovered_result_payload_sha256": result[SEAL_FIELD],
                "scheduler_state": SOURCE_SCHEDULER_STATE,
                "source_result_absent": True,
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return result


def load_result(output_root: Path | str, repo_root: Path | str, *, audit_source_state: bool):
    output = Path(output_root).resolve()
    repo = Path(repo_root).resolve()
    result_path = output / "control" / RESULT_NAME
    result = load_json(result_path, "recovered batch-7 result")
    body = verify_seal(result, "recovered batch-7 result")
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("source_parent_protocol_id") != SOURCE_PARENT_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "batch7_result_recovery"
        or body.get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_BATCH7_RESULT_RECOVERED_CPU_ONLY"
        or body.get("batch_index") != BATCH_INDEX
        or body.get("batch_id") != BATCH_ID
        or body.get("scientific_generation_valid") is not True
        or body.get("source_job_terminated_by_scheduler_timeout") is not True
        or body.get("source_result_absent") is not True
        or body.get("source_stopped_preserved") is not True
        or body.get("source_generation_regenerated") is not False
        or body.get("source_timeout_reclassified_as_completed") is not False
        or body.get("recovery_result_is_not_a_source_result") is not True
        or body.get("policy") != _policy()
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
    ):
        raise ValueError("recovered batch-7 result identity differs")
    if audit_source_state:
        expected = _expected_result(output, repo)
        if result != expected:
            raise ValueError("recovered batch-7 result differs from CPU derivation")
    return result


def audit(args):
    _require_cpu_only()
    output = Path(args.output_root).resolve()
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME, RESULT_NAME})
    result = load_result(output, args.repo_root, audit_source_state=True)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH7_RESULT_RECOVERY_AUDITED",
                "recovered_result_payload_sha256": result[SEAL_FIELD],
                "source_timeout_preserved": True,
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return result


def self_test() -> None:
    assert BATCH_INDEX == 7 and BATCH_ID == "batch_07"
    assert SOURCE_JOB_ID == "273276"
    assert SOURCE_SCHEDULER_STATE == "TIMEOUT"
    assert _policy()["gpu_jobs_authorized"] == 0
    assert _policy()["external_api_calls_authorized"] == 0
    print("MASSIVE_MEDICAL_KALAI_S1_BATCH7_RESULT_RECOVERY_V1_SELF_TEST_OK")


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-output-root")
    parser.add_argument("--source-repo-root")
    parser.add_argument("--unattended-output-root")
    parser.add_argument("--unattended-repo-root")
    parser.add_argument("--log-root")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "stage"):
        item = subparsers.add_parser(command)
        item.add_argument("--output-root", required=True)
        if command == "stage":
            item.add_argument("--repo-root", required=True)
        _source_arguments(item)
        for action in item._actions:
            if action.dest in {
                "source_output_root",
                "source_repo_root",
                "unattended_output_root",
                "unattended_repo_root",
                "log_root",
            }:
                action.required = True
    for command in ("recover", "audit"):
        item = subparsers.add_parser(command)
        item.add_argument("--output-root", required=True)
        item.add_argument("--repo-root", required=True)
    subparsers.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "stage":
        stage(args)
    elif args.command == "recover":
        recover(args)
    elif args.command == "audit":
        audit(args)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
