#!/usr/bin/env python3
"""Seal and audit the unattended parent authority for Kalai s=1 batches 3--7.

The scientific jobs remain the already CPU-staged continuation-v3 jobs.  This
fresh parent protocol grants a single nonreusable, fail-closed sequence: create
one child authority, wait for that job to become terminal, deeply audit its
sealed result and Slurm record, and only then create the next child authority.
"""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import socket
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


v3_manager = _load(
    "_kalai_s1_rc_v3_manager_for_unattended_v2",
    "manage_massive_medical_kalai_s1_recovery_continuation_v3.py",
)
v3_authorizer = _load(
    "_kalai_s1_rc_v3_authorizer_for_unattended_v2",
    "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v3.py",
)
v3_evaluator = _load(
    "_kalai_s1_rc_v3_evaluator_for_unattended_v2",
    "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v3.py",
)
v3_assembler = _load(
    "_kalai_s1_rc_v3_assembler_for_unattended_v2",
    "assemble_massive_medical_kalai_s1_recovery_continuation_v3.py",
)

PROTOCOL_ID = "massive_medical_kalai_s1_recovery_continuation_unattended_v2"
EXPECTED_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-recovery-continuation-unattended-v2"
EXPECTED_OUTPUT_LEAF = PROTOCOL_ID
EXPECTED_V3_REPO_LEAF = v3_manager.EXPECTED_REPO_LEAF
EXPECTED_V3_OUTPUT_LEAF = v3_manager.EXPECTED_OUTPUT_LEAF
FAILED_V1_PROTOCOL_ID = "massive_medical_kalai_s1_recovery_continuation_unattended_v1"
EXPECTED_V1_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-recovery-continuation-unattended-v1"
EXPECTED_V1_OUTPUT_LEAF = FAILED_V1_PROTOCOL_ID
EXPECTED_V1_REPOSITORY_COMMIT = "49bc6297bc401a5c9c865964f234774217f12719"
BATCH_INDICES = tuple(range(3, 8))
PLAN_NAME = "UNATTENDED_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
AUTHORITY_NAME = "SEQUENCE_AUTHORIZATION.json"
INVOCATION_NAME = "UNATTENDED_INVOCATION.json"
FINAL_NAME = "FINAL_SEQUENCE.json"
CHILD_AUTHORITY_NAME = "CHILD_AUTHORITY.json"
SACCT_NAME = "SACCT.tsv"
RECEIPT_NAME = "TERMINAL_RECEIPT.json"

H200_COUNT = 1
H200_MINUTES = 60
PER_JOB_CAP_USD = Decimal("0.900")
TOTAL_NEW_CAP_USD = Decimal("4.500")
KNOWN_PROGRAM_ACTUAL_USD = Decimal("5.69984025")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("7.92198425")
FINAL_CONSERVATIVE_MAXIMUM_USD = Decimal("12.42198425")
PROGRAM_CEILING_USD = Decimal("12.5000000")
MAXIMUM_NEW_CANDIDATE_ATTEMPTS = 2109

SEAL_FIELD = v3_manager.SEAL_FIELD
seal = v3_manager.seal
verify_seal = v3_manager.verify_seal
load_json = v3_manager.load_json
binding = v3_manager.binding
raw_binding = v3_manager.raw_binding
sha256_file = v3_manager.sha256_file
_write_new = v3_manager._write_new
_write_idempotent = v3_manager._write_idempotent
_require_no_api_key = v3_manager._require_no_api_key
_assert_disjoint = v3_manager._assert_disjoint

_NEW_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_recovery_continuation_unattended_v2.yaml",
    "docs/massive_medical_kalai_s1_recovery_continuation_unattended_v2_protocol.md",
    "scripts/manage_massive_medical_kalai_s1_recovery_continuation_unattended_v2.py",
    "scripts/run_massive_medical_kalai_s1_recovery_continuation_unattended_v2_tillicum.sh",
    "scripts/stage_massive_medical_kalai_s1_recovery_continuation_unattended_v2_tillicum.sh",
    "tests/test_massive_medical_kalai_s1_recovery_continuation_unattended_v2.py",
)
REQUIRED_IMPLEMENTATION_FILES = tuple(
    dict.fromkeys(_NEW_FILES + tuple(v3_manager.REQUIRED_IMPLEMENTATION_FILES))
)


def git_commit(repo_root: Path | str) -> str:
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean(repo_root: Path | str, description: str) -> None:
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError(f"{description} repository is not clean")


def batch_id(batch_index: int) -> str:
    if batch_index not in BATCH_INDICES:
        raise ValueError("batch index must be exactly one of 3..7")
    return f"batch_{batch_index:02d}"


def current_exposure(batch_index: int) -> Decimal:
    batch_id(batch_index)
    return CURRENT_CONSERVATIVE_EXPOSURE_USD + PER_JOB_CAP_USD * Decimal(
        batch_index - 3
    )


def maximum_exposure(batch_index: int) -> Decimal:
    return current_exposure(batch_index) + PER_JOB_CAP_USD


def controller_batch_root(output_root: Path | str, batch_index: int) -> Path:
    return Path(output_root).resolve() / "control" / "batches" / batch_id(batch_index)


def _require_sequence_mutable(output: Path) -> None:
    control = output / "control"
    if os.path.lexists(control / "SEQUENCE_STOPPED"):
        raise ValueError("unattended sequence is permanently hard-stopped")
    if os.path.lexists(control / FINAL_NAME):
        raise ValueError("unattended sequence is already final")


def v3_batch_root(v3_output_root: Path | str, batch_index: int) -> Path:
    return Path(v3_output_root).resolve() / "control" / "batches" / batch_id(batch_index)


def implementation_bindings(repo_root: Path | str) -> dict[str, str]:
    repo = Path(repo_root).resolve()
    result = {}
    for relative in REQUIRED_IMPLEMENTATION_FILES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required implementation is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def _audit_failed_v1(v1_output_root: Path | str, v1_repo_root: Path | str) -> dict:
    output = Path(v1_output_root).resolve()
    repo = Path(v1_repo_root).resolve()
    if output.name != EXPECTED_V1_OUTPUT_LEAF or repo.name != EXPECTED_V1_REPO_LEAF:
        raise ValueError("failed unattended-v1 namespace leaf differs")
    _assert_disjoint(output, repo)
    require_clean(repo, "failed unattended-v1")
    if git_commit(repo) != EXPECTED_V1_REPOSITORY_COMMIT:
        raise ValueError("failed unattended-v1 repository commit differs")
    control = output / "control"
    batch = control / "batches" / "batch_03"
    top_expected = {
        PLAN_NAME,
        STAGE_NAME,
        AUTHORITY_NAME,
        INVOCATION_NAME,
        "SEQUENCE_STOPPED",
        "batches",
    }
    if (
        output.is_symlink()
        or not output.is_dir()
        or {item.name for item in output.iterdir()} != {"control"}
        or control.is_symlink()
        or not control.is_dir()
        or {item.name for item in control.iterdir()} != top_expected
        or (control / "batches").is_symlink()
        or {item.name for item in (control / "batches").iterdir()} != {"batch_03"}
        or batch.is_symlink()
        or not batch.is_dir()
        or {item.name for item in batch.iterdir()} != {CHILD_AUTHORITY_NAME}
        or any(item.is_symlink() for item in output.rglob("*"))
        or sum(1 for item in output.rglob("*") if item.is_file()) != 6
    ):
        raise ValueError("failed unattended-v1 inventory differs")
    paths = {
        PLAN_NAME: control / PLAN_NAME,
        STAGE_NAME: control / STAGE_NAME,
        AUTHORITY_NAME: control / AUTHORITY_NAME,
        INVOCATION_NAME: control / INVOCATION_NAME,
        CHILD_AUTHORITY_NAME: batch / CHILD_AUTHORITY_NAME,
    }
    payloads = {name: load_json(path, f"failed unattended-v1 {name}") for name, path in paths.items()}
    bodies = {name: verify_seal(payload, f"failed unattended-v1 {name}") for name, payload in payloads.items()}
    if (
        any(body.get("protocol_id") != FAILED_V1_PROTOCOL_ID for body in bodies.values())
        or bodies[PLAN_NAME].get("status") != "UNATTENDED_SEQUENCE_PLANNED"
        or bodies[STAGE_NAME].get("status") != "UNATTENDED_SEQUENCE_CPU_STAGED"
        or bodies[AUTHORITY_NAME].get("status") != "UNATTENDED_BATCHES_3_THROUGH_7_AUTHORIZED"
        or bodies[INVOCATION_NAME].get("status") != "UNATTENDED_SEQUENCE_STARTED_NONREUSABLE"
        or bodies[CHILD_AUTHORITY_NAME].get("status") != "UNATTENDED_CHILD_AUTHORIZED"
        or bodies[CHILD_AUTHORITY_NAME].get("batch_index") != 3
        or bodies[CHILD_AUTHORITY_NAME].get("authorized_gpu_jobs") != 1
        or bodies[CHILD_AUTHORITY_NAME].get("external_api_calls_authorized") != 0
    ):
        raise ValueError("failed unattended-v1 sealed history differs")
    stopped_path = control / "SEQUENCE_STOPPED"
    stopped_lines = stopped_path.read_text(encoding="utf-8").splitlines()
    if stopped_lines != [
        f"protocol_id={FAILED_V1_PROTOCOL_ID}",
        "status=HARD_STOPPED",
        "active_batch=3",
        "active_job=none",
        "exit_code=1",
        "restart_resume_retry_replacement_requeue_authorized=false",
        "external_api_calls_authorized=0",
    ]:
        raise ValueError("failed unattended-v1 STOPPED record differs")
    bindings = {name: binding(paths[name], payloads[name]) for name in paths}
    bindings["SEQUENCE_STOPPED"] = raw_binding(stopped_path)
    return {
        "output": output,
        "repo": repo,
        "bindings": bindings,
    }


def _audit_v3_anchor(v3_output_root: Path | str, v3_repo_root: Path | str, *, require_fresh: bool):
    output = Path(v3_output_root).resolve()
    repo = Path(v3_repo_root).resolve()
    if output.name != EXPECTED_V3_OUTPUT_LEAF or repo.name != EXPECTED_V3_REPO_LEAF:
        raise ValueError("continuation-v3 namespace leaf differs")
    _assert_disjoint(output, repo)
    require_clean(repo, "continuation-v3")
    stage, stage_body, plan, plan_body, context = v3_manager.load_and_verify_stage(
        output / "control" / v3_manager.STAGE_NAME, repo, audit_source=True
    )
    if require_fresh:
        if (
            output.is_symlink()
            or not output.is_dir()
            or {item.name for item in output.iterdir()} != {"control"}
            or {item.name for item in (output / "control").iterdir()}
            != {v3_manager.PLAN_NAME, v3_manager.STAGE_NAME}
            or any(item.is_symlink() for item in output.rglob("*"))
            or sum(1 for item in output.rglob("*") if item.is_file()) != 2
        ):
            raise ValueError("continuation-v3 CPU-stage inventory is not fresh")
    return {
        "output": output,
        "repo": repo,
        "plan": plan,
        "plan_body": plan_body,
        "stage": stage,
        "stage_body": stage_body,
        "context": context,
    }


def _expected_plan_body(v3: dict, failed_v1: dict, output_root: Path | str) -> dict:
    summaries = [
        {
            "batch_index": index,
            "batch_id": batch_id(index),
            "summary": v3_manager.EXPECTED_BATCH_SUMMARIES[index],
            "current_conservative_exposure_usd": float(current_exposure(index)),
            "conservative_maximum_after_cap_usd": float(maximum_exposure(index)),
        }
        for index in BATCH_INDICES
    ]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "parented_scientific_protocol_id": v3_manager.PROTOCOL_ID,
        "method_id": v3_manager.METHOD_ID,
        "stage": "unattended_sequence_plan",
        "status": "UNATTENDED_SEQUENCE_PLANNED",
        "output_root": str(Path(output_root).resolve()),
        "failed_unattended_v1": {
            "protocol_id": FAILED_V1_PROTOCOL_ID,
            "repository": {
                "path": str(failed_v1["repo"]),
                "commit": EXPECTED_V1_REPOSITORY_COMMIT,
                "clean": True,
            },
            "output_root": str(failed_v1["output"]),
            "bindings": failed_v1["bindings"],
            "status": "HARD_STOPPED_BEFORE_SBATCH",
            "active_batch": 3,
            "active_job": None,
            "exact_slurm_submission_attempts": 0,
            "actual_h200_cost_usd": 0.0,
            "authority_consumed_and_nonreusable": True,
            "cost_exposure_retained_usd": 0.0,
        },
        "continuation_v3_repository": {
            "path": str(v3["repo"]),
            "commit": git_commit(v3["repo"]),
            "clean": True,
        },
        "continuation_v3_output_root": str(v3["output"]),
        "continuation_v3_bindings": {
            v3_manager.PLAN_NAME: binding(
                v3["output"] / "control" / v3_manager.PLAN_NAME, v3["plan"]
            ),
            v3_manager.STAGE_NAME: binding(
                v3["output"] / "control" / v3_manager.STAGE_NAME, v3["stage"]
            ),
        },
        "predecessor_bindings": {
            "recovered_batch_2": v3["plan_body"]["recovery_bindings"][
                v3_manager.recovery_manager.RESULT_NAME
            ],
            "source_batch_2_stopped": v3["plan_body"]["source_bindings"][
                "batch_02_stopped"
            ],
        },
        "batches": summaries,
        "execution_policy": {
            "batch_indices": list(BATCH_INDICES),
            "maximum_gpu_jobs": len(BATCH_INDICES),
            "one_job_at_a_time": True,
            "held_first_submission": True,
            "next_only_after_terminal_receipt": True,
            "automatic_sequential_continuation": True,
            "hard_stop_on_failure_timeout_invalid_audit_or_accounting_discrepancy": True,
            "restart": False,
            "resume": False,
            "retry": False,
            "replacement": False,
            "requeue": False,
            "external_api_calls": False,
            "judge": False,
            "cpu_only_final_assembly": True,
        },
        "accounting": {
            "known_program_actual_usd": float(KNOWN_PROGRAM_ACTUAL_USD),
            "current_conservative_exposure_usd": float(
                CURRENT_CONSERVATIVE_EXPOSURE_USD
            ),
            "per_job_cap_usd": float(PER_JOB_CAP_USD),
            "total_new_h200_cap_usd": float(TOTAL_NEW_CAP_USD),
            "maximum_after_all_caps_usd": float(FINAL_CONSERVATIVE_MAXIMUM_USD),
            "program_ceiling_usd": float(PROGRAM_CEILING_USD),
            "remaining_ceiling_gap_usd": float(
                PROGRAM_CEILING_USD - FINAL_CONSERVATIVE_MAXIMUM_USD
            ),
        },
        "maximum_new_candidate_attempts": MAXIMUM_NEW_CANDIDATE_ATTEMPTS,
        "gpu_jobs_during_cpu_stage": 0,
        "external_api_calls": 0,
    }


def _expected_authority_body(output: Path, plan: dict, stage: dict) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "standing_sequence_authority",
        "status": "UNATTENDED_BATCHES_3_THROUGH_7_AUTHORIZED",
        "unattended_plan": binding(output / "control" / PLAN_NAME, plan),
        "cpu_stage": binding(output / "control" / STAGE_NAME, stage),
        "batch_indices": list(BATCH_INDICES),
        "maximum_gpu_jobs": 5,
        "h200_count_per_job": H200_COUNT,
        "wall_time_minutes_per_job": H200_MINUTES,
        "maximum_new_candidate_attempts": MAXIMUM_NEW_CANDIDATE_ATTEMPTS,
        "per_job_cap_usd": float(PER_JOB_CAP_USD),
        "total_new_h200_cap_usd": float(TOTAL_NEW_CAP_USD),
        "known_program_actual_usd": float(KNOWN_PROGRAM_ACTUAL_USD),
        "current_conservative_exposure_usd": float(
            CURRENT_CONSERVATIVE_EXPOSURE_USD
        ),
        "conservative_maximum_after_all_caps_usd": float(
            FINAL_CONSERVATIVE_MAXIMUM_USD
        ),
        "program_ceiling_usd": float(PROGRAM_CEILING_USD),
        "automatic_sequential_continuation_authorized": True,
        "next_requires_preceding_sealed_terminal_receipt": True,
        "hard_stop_required": True,
        "cpu_only_final_assembly_authorized": True,
        "external_api_calls_authorized": 0,
        "judging_authorized": False,
        "restart_resume_retry_replacement_requeue_authorized": False,
        "authority_reusable": False,
        "failed_unattended_v1_authority_consumed_and_nonreusable": True,
        "failed_unattended_v1_exact_slurm_attempts": 0,
        "failed_unattended_v1_cost_exposure_usd": 0.0,
    }


def _expected_stage_body(repo: Path, output: Path, plan: dict) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "unattended_cpu_stage",
        "status": "UNATTENDED_SEQUENCE_CPU_STAGED",
        "repository": {
            "path": str(repo),
            "commit": git_commit(repo),
            "clean": True,
        },
        "unattended_plan": binding(output / "control" / PLAN_NAME, plan),
        "implementation_sha256": implementation_bindings(repo),
        "gpu_jobs_during_stage": 0,
        "external_api_calls": 0,
    }


def _audit_controller_namespace(output_root: Path | str, allowed_control: set[str]) -> Path:
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.name != EXPECTED_OUTPUT_LEAF:
        raise ValueError("unattended output leaf differs")
    if os.path.lexists(output) and (output.is_symlink() or not output.is_dir()):
        raise ValueError("unattended output root is unsafe")
    for forbidden in ("generation", "assembled", "judge"):
        if os.path.lexists(output / forbidden):
            raise ValueError(f"unattended controller contains forbidden {forbidden}")
    if output.exists() and {item.name for item in output.iterdir()} != {"control"}:
        raise ValueError("unattended output top-level inventory differs")
    control = output / "control"
    if control.exists():
        if control.is_symlink() or not control.is_dir():
            raise ValueError("unattended control root is unsafe")
        unexpected = {item.name for item in control.iterdir()} - allowed_control
        if unexpected:
            raise ValueError(f"unattended control inventory differs: {sorted(unexpected)}")
    return output


def _load_anchor_from_plan(plan_body: dict, *, require_fresh: bool = False):
    v3 = _audit_v3_anchor(
        plan_body.get("continuation_v3_output_root", ""),
        plan_body.get("continuation_v3_repository", {}).get("path", ""),
        require_fresh=require_fresh,
    )
    expected_bindings = {
        v3_manager.PLAN_NAME: binding(
            v3["output"] / "control" / v3_manager.PLAN_NAME, v3["plan"]
        ),
        v3_manager.STAGE_NAME: binding(
            v3["output"] / "control" / v3_manager.STAGE_NAME, v3["stage"]
        ),
    }
    if plan_body.get("continuation_v3_bindings") != expected_bindings:
        raise ValueError("continuation-v3 anchor binding differs")
    failed = plan_body.get("failed_unattended_v1", {})
    failed_v1 = _audit_failed_v1(
        failed.get("output_root", ""), failed.get("repository", {}).get("path", "")
    )
    if failed.get("bindings") != failed_v1["bindings"]:
        raise ValueError("failed unattended-v1 binding differs")
    return v3, failed_v1


def _load_workflow(output_root: Path | str, repo_root: Path | str, *, require_fresh_v3: bool = False):
    output = Path(output_root).resolve()
    repo = Path(repo_root).resolve()
    if output.name != EXPECTED_OUTPUT_LEAF or repo.name != EXPECTED_REPO_LEAF:
        raise ValueError("unattended namespace leaf differs")
    require_clean(repo, "unattended controller")
    stage = load_json(output / "control" / STAGE_NAME, "unattended CPU stage")
    stage_body = verify_seal(stage, "unattended CPU stage")
    plan = load_json(output / "control" / PLAN_NAME, "unattended plan")
    plan_body = verify_seal(plan, "unattended plan")
    if stage_body.get("protocol_id") != PROTOCOL_ID or plan_body.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unattended protocol identity differs")
    if stage_body != _expected_stage_body(repo, output, plan):
        raise ValueError("unattended CPU-stage body differs")
    v3, failed_v1 = _load_anchor_from_plan(plan_body, require_fresh=require_fresh_v3)
    if plan_body != _expected_plan_body(v3, failed_v1, output):
        raise ValueError("unattended plan differs from anchored state")
    authority = load_json(
        output / "control" / AUTHORITY_NAME, "standing sequence authority"
    )
    authority_body = verify_seal(authority, "standing sequence authority")
    if authority_body != _expected_authority_body(output, plan, stage):
        raise ValueError("standing sequence authority differs")
    return output, repo, plan, plan_body, stage, stage_body, authority, authority_body, v3


def stage(args):
    _require_no_api_key()
    output = _audit_controller_namespace(
        args.output_root, {PLAN_NAME, STAGE_NAME, AUTHORITY_NAME}
    )
    repo = Path(args.repo_root).resolve()
    if repo.name != EXPECTED_REPO_LEAF:
        raise ValueError("unattended repository leaf differs")
    _assert_disjoint(
        output,
        repo,
        args.v3_output_root,
        args.v3_repo_root,
        args.v1_output_root,
        args.v1_repo_root,
    )
    require_clean(repo, "unattended controller")
    v3 = _audit_v3_anchor(args.v3_output_root, args.v3_repo_root, require_fresh=True)
    failed_v1 = _audit_failed_v1(args.v1_output_root, args.v1_repo_root)
    plan = seal(_expected_plan_body(v3, failed_v1, output.resolve()))
    _write_idempotent(output.resolve() / "control" / PLAN_NAME, plan, "unattended plan")
    stage_payload = seal(_expected_stage_body(repo, output.resolve(), plan))
    _write_idempotent(
        output.resolve() / "control" / STAGE_NAME,
        stage_payload,
        "unattended CPU stage",
    )
    authority = seal(_expected_authority_body(output.resolve(), plan, stage_payload))
    _write_idempotent(
        output.resolve() / "control" / AUTHORITY_NAME,
        authority,
        "standing sequence authority",
    )
    _load_workflow(output, repo, require_fresh_v3=True)
    print(
        json.dumps(
            {
                "status": "KALAI_S1_UNATTENDED_V2_CPU_STAGED_AND_AUTHORIZED",
                "batch_indices": list(BATCH_INDICES),
                "maximum_gpu_jobs": 5,
                "maximum_new_candidate_attempts": MAXIMUM_NEW_CANDIDATE_ATTEMPTS,
                "total_new_h200_cap_usd": float(TOTAL_NEW_CAP_USD),
                "gpu_jobs_during_stage": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def _load_invocation(output: Path, authority: dict) -> tuple[dict, dict]:
    payload = load_json(output / "control" / INVOCATION_NAME, "unattended invocation")
    body = verify_seal(payload, "unattended invocation")
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("status") != "UNATTENDED_SEQUENCE_STARTED_NONREUSABLE"
        or body.get("standing_authority")
        != binding(output / "control" / AUTHORITY_NAME, authority)
        or body.get("batch_indices") != list(BATCH_INDICES)
        or body.get("restart_or_resume_authorized") is not False
        or body.get("external_api_calls_authorized") != 0
    ):
        raise ValueError("unattended invocation differs")
    return payload, body


def begin(args):
    _require_no_api_key()
    output, _, _, _, _, _, authority, _, _ = _load_workflow(
        args.output_root, args.repo_root, require_fresh_v3=True
    )
    _require_sequence_mutable(output)
    if {item.name for item in (output / "control").iterdir()} != {
        PLAN_NAME,
        STAGE_NAME,
        AUTHORITY_NAME,
    }:
        raise ValueError("unattended pre-invocation control inventory differs")
    path = output / "control" / INVOCATION_NAME
    if os.path.lexists(path):
        raise ValueError("unattended sequence invocation is nonreusable")
    payload = seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage": "unattended_sequence_invocation",
            "status": "UNATTENDED_SEQUENCE_STARTED_NONREUSABLE",
            "standing_authority": binding(
                output / "control" / AUTHORITY_NAME, authority
            ),
            "batch_indices": list(BATCH_INDICES),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "controller_host": socket.gethostname(),
            "restart_or_resume_authorized": False,
            "retry_replacement_or_requeue_authorized": False,
            "external_api_calls_authorized": 0,
        }
    )
    _write_new(path, payload, "unattended invocation")
    print(json.dumps({"status": payload["status"], "gpu_jobs": 0, "external_api_calls": 0}, sort_keys=True))


def _expected_receipt_body(
    target: Path,
    child: dict,
    v3_auth_path: Path,
    v3_auth: dict,
    result_path: Path,
    result: dict,
    sacct_path: Path,
    slurm: dict,
    result_body: dict,
    batch_index: int,
) -> dict:
    if (
        result_body.get("timing", {}).get("slurm_job_id") != slurm["job_id"]
        or result_body.get("timing", {}).get("elapsed_seconds")
        > slurm["elapsed_seconds"]
        or Decimal(str(result_body.get("timing", {}).get("actual_estimated_cost_usd")))
        > PER_JOB_CAP_USD
        or Decimal(
            str(
                result_body.get("accounting", {}).get(
                    "conservative_exposure_after_batch_authority_usd"
                )
            )
        )
        != maximum_exposure(batch_index)
    ):
        raise ValueError("scientific child timing or accounting differs")
    scheduler_actual = (
        Decimal(slurm["elapsed_seconds"])
        * v3_manager.H200_HOURLY_USD
        / Decimal(3600)
    )
    if scheduler_actual > PER_JOB_CAP_USD:
        raise ValueError("scheduler-derived child cost exceeds authority cap")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "unattended_child_terminal_receipt",
        "status": "UNATTENDED_CHILD_TERMINAL_AUDITED",
        "batch_index": batch_index,
        "batch_id": batch_id(batch_index),
        "parent_child_authority": binding(target / CHILD_AUTHORITY_NAME, child),
        "continuation_v3_authorization": binding(v3_auth_path, v3_auth),
        "continuation_v3_result": binding(result_path, result),
        "sacct_record": raw_binding(sacct_path),
        "slurm": slurm,
        "evaluator_elapsed_seconds": result_body["timing"]["elapsed_seconds"],
        "evaluator_estimated_cost_usd": result_body["timing"][
            "actual_estimated_cost_usd"
        ],
        "scheduler_estimated_cost_usd": float(scheduler_actual),
        "conservative_exposure_after_authority_usd": float(
            maximum_exposure(batch_index)
        ),
        "deep_generation_audit_valid": True,
        "source_stopped_history_preserved": True,
        "next_batch_may_be_authorized": batch_index < 7,
        "final_assembly_may_run": batch_index == 7,
        "restart_resume_retry_replacement_requeue_authorized": False,
        "external_api_calls": 0,
    }


def _load_receipt(output: Path, repo: Path, v3: dict, batch_index: int, *, deep: bool):
    path = controller_batch_root(output, batch_index) / RECEIPT_NAME
    payload = load_json(path, f"{batch_id(batch_index)} terminal receipt")
    body = verify_seal(payload, f"{batch_id(batch_index)} terminal receipt")
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("status") != "UNATTENDED_CHILD_TERMINAL_AUDITED"
        or body.get("batch_index") != batch_index
        or body.get("batch_id") != batch_id(batch_index)
        or body.get("slurm", {}).get("state") != "COMPLETED"
        or body.get("slurm", {}).get("exit_code") != "0:0"
        or body.get("slurm", {}).get("derived_exit_code") != "0:0"
        or body.get("next_batch_may_be_authorized") is not (batch_index < 7)
        or body.get("final_assembly_may_run") is not (batch_index == 7)
        or body.get("external_api_calls") != 0
    ):
        raise ValueError(f"{batch_id(batch_index)} terminal receipt identity differs")
    if deep:
        target = controller_batch_root(output, batch_index)
        if (
            target.is_symlink()
            or not target.is_dir()
            or {item.name for item in target.iterdir()}
            != {CHILD_AUTHORITY_NAME, SACCT_NAME, RECEIPT_NAME}
            or any(item.is_symlink() or not item.is_file() for item in target.iterdir())
        ):
            raise ValueError(f"{batch_id(batch_index)} parent control inventory differs")
        scientific_control = v3_batch_root(v3["output"], batch_index)
        expected_scientific_control = {
            "SUBMISSION_LOCK",
            "AUTHORIZATION.json",
            "SUBMISSION_ATTEMPT.tsv",
            "SUBMITTED",
            "RELEASE_AUTHORIZED",
            "RELEASED",
            "INVOCATION_LOCK",
            "RESULT.json",
        }
        if (
            scientific_control.is_symlink()
            or not scientific_control.is_dir()
            or {item.name for item in scientific_control.iterdir()}
            != expected_scientific_control
            or os.path.lexists(scientific_control / "STOPPED")
        ):
            raise ValueError(f"{batch_id(batch_index)} scientific control inventory differs")
        child = load_json(target / CHILD_AUTHORITY_NAME, "unattended child authority")
        child_body = verify_seal(child, "unattended child authority")
        plan = load_json(output / "control" / PLAN_NAME, "unattended plan")
        plan_body = verify_seal(plan, "unattended plan")
        authority = load_json(output / "control" / AUTHORITY_NAME, "standing sequence authority")
        invocation, _ = _load_invocation(output, authority)
        if child_body != _expected_child_body(
            output, plan, plan_body, authority, invocation, v3, batch_index
        ):
            raise ValueError(f"{batch_id(batch_index)} child authority differs")
        result = v3_evaluator.load_and_verify_result(
            v3["output"], v3["repo"], batch_index, audit_generation=True
        )
        result_path = scientific_control / "RESULT.json"
        result_body = verify_seal(result, "scientific child result")
        v3_auth_path = scientific_control / "AUTHORIZATION.json"
        v3_auth = load_json(v3_auth_path, "scientific child authorization")
        v3_auth_body = verify_seal(v3_auth, "scientific child authorization")
        if v3_auth_body != v3_authorizer.expected_body(
            v3["output"],
            v3["repo"],
            batch_index,
            v3_auth_body.get("created_at"),
        ):
            raise ValueError(f"{batch_id(batch_index)} scientific authority differs")
        sacct_path = target / SACCT_NAME
        slurm = _parse_sacct(sacct_path, batch_index)
        _audit_v3_completed_control(v3, batch_index, slurm["job_id"])
        expected_body = _expected_receipt_body(
            target,
            child,
            v3_auth_path,
            v3_auth,
            result_path,
            result,
            sacct_path,
            slurm,
            result_body,
            batch_index,
        )
        if body != expected_body:
            differing = sorted(
                key
                for key in set(body) | set(expected_body)
                if body.get(key) != expected_body.get(key)
            )
            raise ValueError(
                f"{batch_id(batch_index)} terminal receipt differs: {differing}"
            )
    return payload, body


def _expected_parent_predecessor(output: Path, plan_body: dict, v3: dict, batch_index: int):
    if batch_index == 3:
        return {
            "kind": "recovered_batch_2",
            "artifact": plan_body["predecessor_bindings"]["recovered_batch_2"],
        }
    receipt, _ = _load_receipt(output, Path("."), v3, batch_index - 1, deep=True)
    return {
        "kind": "preceding_unattended_terminal_receipt",
        "artifact": binding(
            controller_batch_root(output, batch_index - 1) / RECEIPT_NAME, receipt
        ),
    }


def _expected_child_body(
    output: Path,
    plan: dict,
    plan_body: dict,
    authority: dict,
    invocation: dict,
    v3: dict,
    batch_index: int,
) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "unattended_child_authority",
        "status": "UNATTENDED_CHILD_AUTHORIZED",
        "batch_index": batch_index,
        "batch_id": batch_id(batch_index),
        "unattended_plan": binding(output / "control" / PLAN_NAME, plan),
        "standing_authority": binding(
            output / "control" / AUTHORITY_NAME, authority
        ),
        "invocation": binding(output / "control" / INVOCATION_NAME, invocation),
        "predecessor": _expected_parent_predecessor(
            output, plan_body, v3, batch_index
        ),
        "scientific_protocol_id": v3_manager.PROTOCOL_ID,
        "scientific_batch_summary": v3_manager.EXPECTED_BATCH_SUMMARIES[batch_index],
        "authorized_gpu_jobs": 1,
        "h200_count": 1,
        "h200_minutes_cap": 60,
        "maximum_cost_usd": float(PER_JOB_CAP_USD),
        "current_conservative_exposure_usd": float(current_exposure(batch_index)),
        "conservative_maximum_after_cap_usd": float(maximum_exposure(batch_index)),
        "program_ceiling_usd": float(PROGRAM_CEILING_USD),
        "external_api_calls_authorized": 0,
        "judge_authorized": False,
        "restart_resume_retry_replacement_requeue_authorized": False,
        "authority_reusable": False,
    }


def authorize_batch(args):
    _require_no_api_key()
    output, repo, plan, plan_body, _, _, authority, _, v3 = _load_workflow(
        args.output_root, args.repo_root
    )
    _require_sequence_mutable(output)
    invocation, _ = _load_invocation(output, authority)
    index = args.batch_index
    target = controller_batch_root(output, index)
    control = output / "control"
    expected_top = {PLAN_NAME, STAGE_NAME, AUTHORITY_NAME, INVOCATION_NAME}
    if index > 3:
        expected_top.add("batches")
    if {item.name for item in control.iterdir()} != expected_top:
        raise ValueError("unattended sequence control prefix differs")
    batches_root = control / "batches"
    if index > 3 and (
        batches_root.is_symlink()
        or not batches_root.is_dir()
        or {item.name for item in batches_root.iterdir()}
        != {batch_id(prior) for prior in range(3, index)}
    ):
        raise ValueError("unattended preceding-batch inventory differs")
    if os.path.lexists(target) or os.path.lexists(v3_batch_root(v3["output"], index)):
        raise ValueError("target batch has already been attempted")
    for prior in range(3, index):
        _load_receipt(output, repo, v3, prior, deep=True)
    for future in range(index + 1, 8):
        if os.path.lexists(controller_batch_root(output, future)) or os.path.lexists(
            v3_batch_root(v3["output"], future)
        ):
            raise ValueError("future batch state exists out of order")
    v3_authorizer.preflight_predecessor(
        argparse.Namespace(
            output_root=str(v3["output"]),
            repo_root=str(v3["repo"]),
            batch_index=index,
        )
    )
    predecessor = _expected_parent_predecessor(output, plan_body, v3, index)
    target.mkdir(parents=True, exist_ok=False)
    body = _expected_child_body(
        output, plan, plan_body, authority, invocation, v3, index
    )
    if body["predecessor"] != predecessor:
        raise ValueError("unattended child predecessor changed during authorization")
    payload = seal(body)
    _write_new(target / CHILD_AUTHORITY_NAME, payload, "unattended child authority")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "batch_id": batch_id(index),
                "authorized_gpu_jobs": 1,
                "maximum_cost_usd": float(PER_JOB_CAP_USD),
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def _parse_sacct(path: Path, batch_index: int) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("sacct record is absent or unsafe")
    lines = path.read_text(encoding="utf-8").splitlines()
    expected_header = "JobIDRaw|JobName|State|ExitCode|DerivedExitCode|ElapsedRaw|AllocTRES"
    if len(lines) != 2 or lines[0] != expected_header:
        raise ValueError("sacct record shape differs")
    fields = lines[1].split("|")
    if len(fields) != 7:
        raise ValueError("sacct field count differs")
    job_id, job_name, state, exit_code, derived_exit_code, elapsed_raw, alloc_tres = fields
    if (
        not job_id.isdigit()
        or job_name != f"mmu_kalai_s1_r3c{batch_index:02d}"
        or state != "COMPLETED"
        or exit_code != "0:0"
        or derived_exit_code != "0:0"
        or not elapsed_raw.isdigit()
        or not 1 <= int(elapsed_raw) <= H200_MINUTES * 60
    ):
        raise ValueError("sacct terminal state differs")
    tokens = set(alloc_tres.split(","))
    for required in ("cpu=8", "mem=200G", "node=1", "gres/gpu=1", "gres/gpu:h200=1"):
        if required not in tokens:
            raise ValueError(f"sacct allocation omits {required}")
    return {
        "job_id": job_id,
        "job_name": job_name,
        "state": state,
        "exit_code": exit_code,
        "derived_exit_code": derived_exit_code,
        "elapsed_seconds": int(elapsed_raw),
        "alloc_tres": alloc_tres,
    }


def _audit_v3_completed_control(v3: dict, batch_index: int, job_id: str) -> None:
    control = v3_batch_root(v3["output"], batch_index)
    expected = {
        "SUBMISSION_LOCK",
        "AUTHORIZATION.json",
        "SUBMISSION_ATTEMPT.tsv",
        "SUBMITTED",
        "RELEASE_AUTHORIZED",
        "RELEASED",
        "INVOCATION_LOCK",
        "RESULT.json",
    }
    if (
        control.is_symlink()
        or not control.is_dir()
        or {item.name for item in control.iterdir()} != expected
        or os.path.lexists(control / "STOPPED")
    ):
        raise ValueError("scientific child control inventory differs")
    bid = batch_id(batch_index)
    if v3_authorizer._lines(
        control / "SUBMISSION_LOCK" / "owner", "scientific submission lock"
    ) != v3_authorizer._expected_lock(v3["repo"], batch_index):
        raise ValueError("scientific submission lock differs")
    if v3_authorizer._lines(
        control / "SUBMISSION_ATTEMPT.tsv", "scientific submission attempt"
    ) != {
        "stage\tbatch_id\tjob_id\th200_minutes\tmaximum_cost_usd",
        f"recovery_continuation_batch\t{bid}\t{job_id}\t60\t0.900",
    }:
        raise ValueError("scientific submission attempt differs")
    common = {
        f"protocol_id={v3_manager.PROTOCOL_ID}",
        "stage=recovery_continuation_batch",
        f"batch_id={bid}",
        f"job_id={job_id}",
        "restart_or_resume_authorized=false",
        "automatic_next_batch_authorized=false",
    }
    if v3_authorizer._lines(control / "SUBMITTED", "scientific submitted record") != common | {
        "held_first=true",
        "held_audit_passed=true",
        f"repository_commit={git_commit(v3['repo'])}",
    }:
        raise ValueError("scientific submitted record differs")
    if v3_authorizer._lines(
        control / "RELEASE_AUTHORIZED", "scientific release authority"
    ) != common | {"held_audit_passed=true", "release_authorized=true"}:
        raise ValueError("scientific release authority differs")
    if v3_authorizer._lines(control / "RELEASED", "scientific release record") != common | {
        "released=true"
    }:
        raise ValueError("scientific release record differs")
    invocation = {
        "stage=recovery_continuation_batch",
        f"batch_id={bid}",
        f"job_id={job_id}",
        "restart_or_resume_authorized=false",
        "retry_or_replacement_authorized=false",
        "automatic_next_batch_authorized=false",
        "external_api_calls_authorized=0",
    }
    if v3_authorizer._lines(
        control / "INVOCATION_LOCK" / "owner", "scientific invocation lock"
    ) != invocation:
        raise ValueError("scientific invocation lock differs")


def audit_terminal(args):
    _require_no_api_key()
    output, repo, plan, plan_body, _, _, authority, _, v3 = _load_workflow(
        args.output_root, args.repo_root
    )
    _require_sequence_mutable(output)
    invocation, _ = _load_invocation(output, authority)
    index = args.batch_index
    target = controller_batch_root(output, index)
    child = load_json(target / CHILD_AUTHORITY_NAME, "unattended child authority")
    child_body = verify_seal(child, "unattended child authority")
    if child_body != _expected_child_body(
        output, plan, plan_body, authority, invocation, v3, index
    ):
        raise ValueError("unattended child authority differs")
    if os.path.lexists(target / RECEIPT_NAME):
        raise ValueError("terminal receipt is nonreusable")
    sacct_path = Path(args.sacct_path).resolve()
    if sacct_path != target / SACCT_NAME:
        raise ValueError("sacct record path differs")
    slurm = _parse_sacct(sacct_path, index)
    prior_job_ids = set()
    for prior in range(3, index):
        _, prior_body = _load_receipt(output, repo, v3, prior, deep=True)
        prior_job_ids.add(prior_body["slurm"]["job_id"])
    if slurm["job_id"] in prior_job_ids:
        raise ValueError("Slurm job ID was reused across child authorities")
    control = v3_batch_root(v3["output"], index)
    _audit_v3_completed_control(v3, index, slurm["job_id"])
    result = v3_evaluator.load_and_verify_result(
        v3["output"], v3["repo"], index, audit_generation=True
    )
    result_body = verify_seal(result, "scientific child result")
    v3_auth_path = control / "AUTHORIZATION.json"
    v3_auth = load_json(v3_auth_path, "scientific child authorization")
    v3_auth_body = verify_seal(v3_auth, "scientific child authorization")
    if v3_auth_body != v3_authorizer.expected_body(
        v3["output"], v3["repo"], index, v3_auth_body.get("created_at")
    ):
        raise ValueError("scientific child authority differs")
    receipt_body = _expected_receipt_body(
        target,
        child,
        v3_auth_path,
        v3_auth,
        control / "RESULT.json",
        result,
        sacct_path,
        slurm,
        result_body,
        index,
    )
    receipt = seal(receipt_body)
    _write_new(target / RECEIPT_NAME, receipt, "unattended terminal receipt")
    _load_receipt(output, repo, v3, index, deep=True)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "batch_id": batch_id(index),
                "slurm_job_id": slurm["job_id"],
                "scheduler_estimated_cost_usd": receipt_body["scheduler_estimated_cost_usd"],
                "next_batch_may_be_authorized": index < 7,
                "final_assembly_may_run": index == 7,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def assemble(args):
    _require_no_api_key()
    output, repo, plan, _, _, _, authority, _, v3 = _load_workflow(
        args.output_root, args.repo_root
    )
    _require_sequence_mutable(output)
    invocation, _ = _load_invocation(output, authority)
    receipts = []
    actuals = []
    job_ids = set()
    for index in BATCH_INDICES:
        receipt, body = _load_receipt(output, repo, v3, index, deep=True)
        receipts.append(
            {
                "batch_index": index,
                "artifact": binding(
                    controller_batch_root(output, index) / RECEIPT_NAME, receipt
                ),
            }
        )
        actuals.append(Decimal(str(body["scheduler_estimated_cost_usd"])))
        job_id = body["slurm"]["job_id"]
        if job_id in job_ids:
            raise ValueError("Slurm job ID was reused across terminal receipts")
        job_ids.add(job_id)
        if body["next_batch_may_be_authorized"] is not (index < 7):
            raise ValueError("terminal receipt next-batch scope differs")
        if body["final_assembly_may_run"] is not (index == 7):
            raise ValueError("terminal receipt assembly scope differs")
    manifest = v3_assembler.assemble(
        argparse.Namespace(output_root=str(v3["output"]), repo_root=str(v3["repo"]))
    )
    manifest_path = v3["output"] / "control" / "FINAL_ASSEMBLY.json"
    if (
        verify_seal(manifest, "continuation-v3 final assembly").get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V3_FULL_ASSEMBLY_AUDITED"
        or manifest.get("benefit_requested_n") != 360
        or manifest.get("medical_requested_n") != 80
        or manifest.get("external_api_calls") != 0
    ):
        raise ValueError("continuation-v3 final assembly differs")
    new_actual = sum(actuals, Decimal("0"))
    final = seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage": "unattended_sequence_final",
            "status": "KALAI_S1_UNATTENDED_V2_COMPLETE_AND_ASSEMBLED",
            "unattended_plan": binding(output / "control" / PLAN_NAME, plan),
            "standing_authority": binding(output / "control" / AUTHORITY_NAME, authority),
            "invocation": binding(output / "control" / INVOCATION_NAME, invocation),
            "terminal_receipts": receipts,
            "continuation_v3_final_assembly": binding(manifest_path, manifest),
            "benefit_requested_n": 360,
            "medical_requested_n": 80,
            "gpu_jobs_submitted": 5,
            "new_scheduler_estimated_h200_cost_usd": float(new_actual),
            "known_program_actual_before_sequence_usd": float(KNOWN_PROGRAM_ACTUAL_USD),
            "known_actual_including_new_scheduler_estimates_usd": float(
                KNOWN_PROGRAM_ACTUAL_USD + new_actual
            ),
            "total_new_h200_cap_usd": float(TOTAL_NEW_CAP_USD),
            "conservative_maximum_after_all_caps_usd": float(FINAL_CONSERVATIVE_MAXIMUM_USD),
            "program_ceiling_usd": float(PROGRAM_CEILING_USD),
            "judge_authorized": False,
            "external_api_calls": 0,
        }
    )
    _write_new(output / "control" / FINAL_NAME, final, "unattended final sequence")
    print(
        json.dumps(
            {
                "status": final["status"],
                "benefit_requested_n": 360,
                "medical_requested_n": 80,
                "gpu_jobs_submitted": 5,
                "new_scheduler_estimated_h200_cost_usd": final[
                    "new_scheduler_estimated_h200_cost_usd"
                ],
                "known_actual_including_new_scheduler_estimates_usd": final[
                    "known_actual_including_new_scheduler_estimates_usd"
                ],
                "judge_authorized": False,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert BATCH_INDICES == tuple(range(3, 8))
    assert sum(v3_manager.EXPECTED_BATCH_SUMMARIES[i]["attempts"] for i in BATCH_INDICES) == MAXIMUM_NEW_CANDIDATE_ATTEMPTS
    assert maximum_exposure(7) == FINAL_CONSERVATIVE_MAXIMUM_USD
    assert FINAL_CONSERVATIVE_MAXIMUM_USD < PROGRAM_CEILING_USD
    assert PROGRAM_CEILING_USD - FINAL_CONSERVATIVE_MAXIMUM_USD == Decimal("0.07801575")
    print("MASSIVE_MEDICAL_KALAI_S1_UNATTENDED_V2_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    stage_parser = sub.add_parser("stage")
    stage_parser.add_argument("--output-root", required=True)
    stage_parser.add_argument("--repo-root", required=True)
    stage_parser.add_argument("--v3-output-root", required=True)
    stage_parser.add_argument("--v3-repo-root", required=True)
    stage_parser.add_argument("--v1-output-root", required=True)
    stage_parser.add_argument("--v1-repo-root", required=True)
    for command in ("begin", "authorize-batch", "assemble"):
        child = sub.add_parser(command)
        child.add_argument("--output-root", required=True)
        child.add_argument("--repo-root", required=True)
        if command == "authorize-batch":
            child.add_argument("--batch-index", type=int, choices=BATCH_INDICES, required=True)
    terminal = sub.add_parser("audit-terminal")
    terminal.add_argument("--output-root", required=True)
    terminal.add_argument("--repo-root", required=True)
    terminal.add_argument("--batch-index", type=int, choices=BATCH_INDICES, required=True)
    terminal.add_argument("--sacct-path", required=True)
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test()
    elif args.command == "stage":
        stage(args)
    elif args.command == "begin":
        begin(args)
    elif args.command == "authorize-batch":
        authorize_batch(args)
    elif args.command == "audit-terminal":
        audit_terminal(args)
    else:
        assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
