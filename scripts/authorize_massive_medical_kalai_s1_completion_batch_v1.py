#!/usr/bin/env python3
"""Write or verify one exact Kalai ``s=1`` completion-batch authority.

Every batch consumes a separate one-H200, 60-minute, $0.900 envelope.  Earlier
batch caps remain conservatively consumed even when their recorded runtime is
shorter.  Batch ``i>1`` cannot be authorized until batch ``i-1`` has a sealed
successful result.  No authority is granted for another batch, API judging,
restart, resume, retry, replacement, requeue, or automatic continuation.
"""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_module(
    "_kalai_s1_completion_batch_planner_for_authorizer",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_completion_batches_v1.py",
)
stage_builder = _load_module(
    "_kalai_s1_completion_batch_stage_for_authorizer",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_completion_batches_v1"
SOURCE_PROTOCOL_ID = "massive_medical_kalai_s1_r20_trace_reuse_v1"
METHOD_ID = "whole_output_consensus_m4_s1_r20_sensitivity_v1"
SEAL_FIELD = "payload_sha256"
BATCH_INDICES = tuple(range(1, 8))
H200_COUNT = 1
H200_MINUTES = 60
H200_HOURLY_USD = Decimal("0.90")
BATCH_CAP_USD = Decimal("0.900")
STARTING_CONSERVATIVE_EXPOSURE_USD = Decimal("6.12198425")
PROGRAM_CEILING_USD = Decimal("12.5000000")


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def seal(body):
    result = dict(body)
    result.pop(SEAL_FIELD, None)
    result[SEAL_FIELD] = digest(canonical(result))
    return result


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} seal differs")
    return body


def load_json(path, description):
    path = os.path.abspath(path)
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def binding(path, payload=None, description="sealed JSON"):
    path = os.path.abspath(path)
    if payload is None:
        payload = load_json(path, description)
    verify_seal(payload, description)
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[SEAL_FIELD],
    }


def repository_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean(repo_root):
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise ValueError("completion-batch repository differs from its commit")


def batch_id(batch_index):
    if batch_index not in BATCH_INDICES:
        raise ValueError("batch index must be exactly one of 1..7")
    return f"batch_{batch_index:02d}"


def batch_control(output_root, batch_index):
    return (
        Path(output_root).resolve()
        / "control"
        / "batches"
        / batch_id(batch_index)
    )


def authorization_path(output_root, batch_index):
    return batch_control(output_root, batch_index) / "AUTHORIZATION.json"


def current_exposure(batch_index):
    batch_id(batch_index)
    return STARTING_CONSERVATIVE_EXPOSURE_USD + BATCH_CAP_USD * Decimal(
        batch_index - 1
    )


def maximum_exposure(batch_index):
    return current_exposure(batch_index) + BATCH_CAP_USD


def _load_plan(output_root):
    path = Path(output_root).resolve() / "control" / "COMPLETION_BATCH_PLAN.json"
    payload, body = planner.load_and_verify_plan(path, audit_sources=True)
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("controller_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
    ):
        raise ValueError("completion batch-plan identity differs")
    return path, payload, body


def _batch_from_plan(plan_body, batch_index):
    batches = plan_body.get("batches")
    if (
        not isinstance(batches, list)
        or [item.get("batch_index") for item in batches]
        != list(BATCH_INDICES)
        or [item.get("batch_id") for item in batches]
        != [batch_id(index) for index in BATCH_INDICES]
    ):
        raise ValueError("completion batch-plan inventory differs")
    item = batches[batch_index - 1]
    rows = item.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"benefit", "medical"}:
        raise ValueError("completion batch rows differ")
    return item


def _source_gate_binding(plan_body):
    bindings = plan_body.get("source_bindings")
    gate = (
        bindings.get("source_technical_gate_result")
        if isinstance(bindings, dict)
        else None
    )
    if not isinstance(gate, dict) or not isinstance(gate.get("path"), str):
        raise ValueError("source technical-gate binding differs")
    payload = load_json(gate["path"], "source technical-gate result")
    observed = binding(
        gate["path"], payload, "source technical-gate result"
    )
    if observed != gate:
        raise ValueError("source technical-gate binding changed")
    body = verify_seal(payload, "source technical-gate result")
    if (
        body.get("protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_TECHNICAL_GATE_COMPLETE"
        or body.get("technical_gate_valid") is not True
        or body.get("completion_eligible") is not True
    ):
        raise ValueError("source technical gate is not eligible")
    return observed


def _predecessor_binding(
    output_root,
    repo_root,
    plan_path,
    plan_payload,
    plan_body,
    batch_index,
):
    if batch_index == 1:
        return {
            "kind": "source_technical_gate_result",
            "artifact": _source_gate_binding(plan_body),
        }
    previous_index = batch_index - 1
    path = batch_control(output_root, previous_index) / "RESULT.json"
    payload = load_json(path, "preceding completion-batch result")
    body = verify_seal(payload, "preceding completion-batch result")
    if (
        set(body)
        != {
            "schema_version",
            "protocol_id",
            "source_protocol_id",
            "method_id",
            "stage",
            "batch_index",
            "batch_id",
            "status",
            "batch_valid",
            "restart_or_resume_authorized",
            "retry_replacement_or_requeue_authorized",
            "automatic_next_batch_authorized",
            "judge_authorized",
            "batch_plan",
            "authorization",
            "combined_timing",
            "phase_outputs",
            "timing",
            "accounting",
            "external_api_calls",
            "gpu_jobs_submitted_by_evaluator",
        }
        or body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("batch_index") != previous_index
        or body.get("batch_id") != batch_id(previous_index)
        or body.get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_COMPLETE"
        or body.get("batch_valid") is not True
        or body.get("restart_or_resume_authorized") is not False
        or body.get("retry_replacement_or_requeue_authorized") is not False
        or body.get("automatic_next_batch_authorized") is not False
        or body.get("judge_authorized") is not False
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs_submitted_by_evaluator") != 0
        or body.get("batch_plan")
        != binding(plan_path, plan_payload, "completion batch plan")
    ):
        raise ValueError("preceding completion-batch result is not eligible")
    stopped = batch_control(output_root, previous_index) / "STOPPED"
    if os.path.lexists(stopped):
        raise ValueError("preceding completion batch also has STOPPED state")

    authorization_path_value = authorization_path(output_root, previous_index)
    authorization_payload = load_json(
        authorization_path_value, "preceding completion-batch authorization"
    )
    authorization_body = verify_seal(
        authorization_payload, "preceding completion-batch authorization"
    )
    expected_authorization = expected_body(
        output_root,
        repo_root,
        previous_index,
        authorization_body.get("created_at"),
    )
    if (
        authorization_body != expected_authorization
        or body.get("authorization")
        != binding(
            authorization_path_value,
            authorization_payload,
            "preceding completion-batch authorization",
        )
    ):
        raise ValueError("preceding completion-batch authorization differs")

    controller = _load_module(
        f"_kalai_s1_completion_batch_controller_for_predecessor_{previous_index}",
        SCRIPT_DIR / "sample_massive_medical_kalai_s1_completion_batch_v1.py",
    )
    audit = controller._audit_batch(
        output_root, plan_payload, plan_body, previous_index
    )
    combined_path = Path(audit["combined_timing"])
    combined_payload = load_json(
        combined_path, "preceding completion-batch combined timing"
    )
    if body.get("combined_timing") != binding(
        combined_path,
        combined_payload,
        "preceding completion-batch combined timing",
    ):
        raise ValueError("preceding completion-batch combined timing differs")
    expected_phase_outputs = {}
    for phase in planner.PHASES:
        generation_path = Path(audit["phases"][phase]["generation"])
        generation_payload = load_json(
            generation_path, f"preceding {phase} generation"
        )
        timing_path = generation_path.parent / "timing.json"
        timing_payload = load_json(timing_path, f"preceding {phase} timing")
        expected_phase_outputs[phase] = {
            "generation": binding(
                generation_path,
                generation_payload,
                f"preceding {phase} generation",
            ),
            "timing": binding(
                timing_path, timing_payload, f"preceding {phase} timing"
            ),
            "summary": generation_payload["summary"],
            "new_attempts_generated": generation_payload[
                "new_attempts_generated"
            ],
        }
    elapsed = body.get("timing", {}).get("elapsed_seconds")
    job_id = body.get("timing", {}).get("slurm_job_id")
    expected_actual = (
        Decimal(elapsed) * H200_HOURLY_USD / Decimal(3600)
        if isinstance(elapsed, int) and not isinstance(elapsed, bool)
        else None
    )
    if (
        body.get("phase_outputs") != expected_phase_outputs
        or not isinstance(job_id, str)
        or not job_id.isdigit()
        or not isinstance(elapsed, int)
        or isinstance(elapsed, bool)
        or not 1 <= elapsed <= H200_MINUTES * 60
        or body.get("timing")
        != {
            "slurm_job_id": job_id,
            "elapsed_seconds": elapsed,
            "h200_hourly_usd": float(H200_HOURLY_USD),
            "authorized_cap_usd": float(BATCH_CAP_USD),
            "actual_estimated_cost_usd": float(expected_actual),
        }
        or body.get("accounting")
        != {
            "starting_conservative_exposure_usd": float(
                STARTING_CONSERVATIVE_EXPOSURE_USD
            ),
            "prior_batch_authority_caps_retained_usd": float(
                BATCH_CAP_USD * Decimal(previous_index - 1)
            ),
            "this_batch_authority_cap_retained_usd": float(BATCH_CAP_USD),
            "conservative_exposure_after_batch_authority_usd": float(
                maximum_exposure(previous_index)
            ),
            "program_ceiling_usd": float(PROGRAM_CEILING_USD),
        }
    ):
        raise ValueError("preceding completion-batch result artifacts differ")
    return {
        "kind": "preceding_completion_batch_result",
        "artifact": binding(path, payload, "preceding completion-batch result"),
    }


def _cpu_stage_binding(output_root, repo_root, plan_path, plan_payload):
    path = Path(output_root).resolve() / "control" / "CPU_STAGE.json"
    payload = load_json(path, "completion-batch CPU stage")
    body = verify_seal(payload, "completion-batch CPU stage")
    require_clean(repo_root)
    if (
        set(body)
        != {
            "schema_version",
            "protocol_id",
            "source_protocol_id",
            "method_id",
            "status",
            "repository_commit",
            "source_repository",
            "completion_batch_plan",
            "implementation_sha256",
            "batch_design_not_authorization",
            "accounting_context_not_authorization",
            "gpu_jobs_authorized",
            "gpu_jobs_submitted",
            "gpu_authorized",
            "external_api_calls_authorized",
            "external_api_calls",
            "external_api_authorized",
            "batch_authorizations_created",
            "restart_or_resume_authorized",
            "retry_or_replacement_authorized",
            "requeue_authorized",
            "automatic_next_batch_authorized",
        }
        or body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or body.get("repository_commit") != repository_commit(repo_root)
        or body.get("completion_batch_plan")
        != binding(plan_path, plan_payload, "completion batch plan")
        or body.get("gpu_jobs_authorized") != 0
        or body.get("gpu_jobs_submitted") != 0
        or body.get("gpu_authorized") is not False
        or body.get("external_api_calls_authorized") != 0
        or body.get("external_api_calls") != 0
        or body.get("external_api_authorized") is not False
        or body.get("batch_authorizations_created") != 0
        or body.get("restart_or_resume_authorized") is not False
        or body.get("retry_or_replacement_authorized") is not False
        or body.get("requeue_authorized") is not False
        or body.get("automatic_next_batch_authorized") is not False
    ):
        raise ValueError("completion-batch CPU stage differs")
    implementation = body.get("implementation_sha256")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != set(stage_builder.REQUIRED_IMPLEMENTATION_FILES)
    ):
        raise ValueError("completion-batch CPU-stage implementation inventory differs")
    for name, expected_sha256 in implementation.items():
        implementation_path = Path(repo_root).resolve() / name
        if (
            implementation_path.is_symlink()
            or not implementation_path.is_file()
            or sha256_file(implementation_path) != expected_sha256
        ):
            raise ValueError(f"completion-batch implementation differs: {name}")
    return binding(path, payload, "completion-batch CPU stage")


def expected_body(output_root, repo_root, batch_index, created_at):
    output_root = Path(output_root).resolve()
    repo_root = Path(repo_root).resolve()
    plan_path, plan_payload, plan_body = _load_plan(output_root)
    item = _batch_from_plan(plan_body, batch_index)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "completion_batch",
        "batch_index": batch_index,
        "batch_id": batch_id(batch_index),
        "created_at": created_at,
        "repository_commit": repository_commit(repo_root),
        "batch_plan": binding(
            plan_path, plan_payload, "completion batch plan"
        ),
        "cpu_stage": _cpu_stage_binding(
            output_root, repo_root, plan_path, plan_payload
        ),
        "predecessor": _predecessor_binding(
            output_root,
            repo_root,
            plan_path,
            plan_payload,
            plan_body,
            batch_index,
        ),
        "batch_summary": item["summary"],
        "authorized_gpu_jobs": 1,
        "h200_count": H200_COUNT,
        "h200_minutes_cap": H200_MINUTES,
        "h200_hourly_usd": float(H200_HOURLY_USD),
        "maximum_cost_usd": float(BATCH_CAP_USD),
        "starting_conservative_exposure_usd": float(
            STARTING_CONSERVATIVE_EXPOSURE_USD
        ),
        "prior_batch_authority_caps_retained_usd": float(
            BATCH_CAP_USD * Decimal(batch_index - 1)
        ),
        "current_conservative_exposure_usd": float(
            current_exposure(batch_index)
        ),
        "conservative_actual_plus_new_cap_usd": float(
            maximum_exposure(batch_index)
        ),
        "program_ceiling_usd": float(PROGRAM_CEILING_USD),
        "external_api_calls_authorized": 0,
        "judge_authorized": False,
        "another_batch_authorized": False,
        "automatic_next_batch_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_replacement_or_requeue_authorized": False,
    }


def write_new(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _control_line_set(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe")
    return set(path.read_text(encoding="utf-8").splitlines())


def audit_acknowledgments(args):
    batch_id(args.batch_index)
    if args.ack_h200_minutes != H200_MINUTES:
        raise ValueError("H200-minute acknowledgment differs")
    if Decimal(args.ack_max_cost_usd) != BATCH_CAP_USD:
        raise ValueError("cost acknowledgment differs")
    if Decimal(args.ack_current_conservative_exposure_usd) != current_exposure(
        args.batch_index
    ):
        raise ValueError("current conservative exposure acknowledgment differs")
    if Decimal(args.ack_conservative_program_max_usd) != maximum_exposure(
        args.batch_index
    ):
        raise ValueError("program maximum acknowledgment differs")
    if Decimal(args.ack_program_ceiling_usd) != PROGRAM_CEILING_USD:
        raise ValueError("program ceiling acknowledgment differs")
    required_true = (
        "ack_no_api",
        "ack_no_automatic_next_batch",
        "ack_no_restart_resume_retry_replacement",
        "ack_prior_batch_caps_retained",
        "ack_post_hoc_sensitivity",
    )
    if any(getattr(args, name) is not True for name in required_true):
        raise ValueError("required negative-authority acknowledgment is absent")


def _require_target_fresh(
    output_root,
    repo_root,
    batch_index,
    *,
    authorization_may_exist,
    generation_may_exist=False,
    running_state_may_exist=False,
    slurm_job_id=None,
):
    control = batch_control(output_root, batch_index)
    authorization = control / "AUTHORIZATION.json"
    generation = (
        Path(output_root).resolve()
        / "generation"
        / "completion_batches"
        / batch_id(batch_index)
    )
    if not generation_may_exist and os.path.lexists(generation):
        raise ValueError("target batch generation already exists")
    for name in ("RESULT.json", "STOPPED"):
        if os.path.lexists(control / name):
            raise ValueError(f"target batch already has {name}")
    invocation = control / "INVOCATION_LOCK"
    if running_state_may_exist:
        if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit():
            raise ValueError("target batch Slurm job ID differs")
        required = {
            "AUTHORIZATION.json",
            "SUBMISSION_LOCK",
            "SUBMISSION_ATTEMPT.tsv",
            "SUBMITTED",
            "RELEASE_AUTHORIZED",
            "RELEASED",
            "INVOCATION_LOCK",
        }
        if control.is_symlink() or not control.is_dir():
            raise ValueError("target batch running control is absent or unsafe")
        if {path.name for path in control.iterdir()} != required:
            raise ValueError("target batch running control inventory differs")
        expected_commit = repository_commit(repo_root)
        lock_lines = _control_line_set(
            control / "SUBMISSION_LOCK" / "owner",
            "target batch submission-lock owner",
        )
        required_lock_lines = {
            f"protocol_id={PROTOCOL_ID}",
            "stage=completion_batch",
            f"batch_id={batch_id(batch_index)}",
            f"repository_commit={expected_commit}",
            "restart_or_resume_authorized=false",
            "retry_or_replacement_authorized=false",
            "automatic_next_batch_authorized=false",
        }
        if lock_lines != required_lock_lines:
            raise ValueError("target batch submission-lock owner differs")
        attempt_lines = _control_line_set(
            control / "SUBMISSION_ATTEMPT.tsv",
            "target batch submission attempt",
        )
        if attempt_lines != {
            "stage\tbatch_id\tjob_id\th200_minutes\tmaximum_cost_usd",
            (
                "completion_batch\t"
                f"{batch_id(batch_index)}\t{slurm_job_id}\t"
                f"{H200_MINUTES}\t{BATCH_CAP_USD}"
            ),
        }:
            raise ValueError("target batch submission attempt differs")
        submitted_lines = _control_line_set(
            control / "SUBMITTED", "target batch submitted record"
        )
        if submitted_lines != {
            f"protocol_id={PROTOCOL_ID}",
            "stage=completion_batch",
            f"batch_id={batch_id(batch_index)}",
            f"job_id={slurm_job_id}",
            "held_first=true",
            "held_audit_passed=true",
            f"repository_commit={expected_commit}",
            "restart_or_resume_authorized=false",
            "automatic_next_batch_authorized=false",
        }:
            raise ValueError("target batch submitted record differs")
        release_authorized_lines = _control_line_set(
            control / "RELEASE_AUTHORIZED",
            "target batch release authorization",
        )
        if release_authorized_lines != {
            f"protocol_id={PROTOCOL_ID}",
            "stage=completion_batch",
            f"batch_id={batch_id(batch_index)}",
            f"job_id={slurm_job_id}",
            "held_audit_passed=true",
            "release_authorized=true",
            "restart_or_resume_authorized=false",
            "automatic_next_batch_authorized=false",
        }:
            raise ValueError("target batch release authorization differs")
        released_lines = _control_line_set(
            control / "RELEASED", "target batch release record"
        )
        if released_lines != {
            f"protocol_id={PROTOCOL_ID}",
            "stage=completion_batch",
            f"batch_id={batch_id(batch_index)}",
            f"job_id={slurm_job_id}",
            "released=true",
            "restart_or_resume_authorized=false",
            "automatic_next_batch_authorized=false",
        }:
            raise ValueError("target batch release record differs")
        lines = _control_line_set(
            invocation / "owner", "target batch invocation owner"
        )
        if (
            lines
            != {
                "stage=completion_batch",
                f"batch_id={batch_id(batch_index)}",
                f"job_id={slurm_job_id}",
                "restart_or_resume_authorized=false",
                "retry_or_replacement_authorized=false",
                "automatic_next_batch_authorized=false",
                "external_api_calls_authorized=0",
            }
        ):
            raise ValueError("target batch invocation owner differs")
    elif os.path.lexists(invocation):
        raise ValueError("target batch already has INVOCATION_LOCK")
    else:
        expected = {"SUBMISSION_LOCK"}
        if authorization_may_exist:
            expected.add("AUTHORIZATION.json")
        if control.is_symlink() or not control.is_dir():
            raise ValueError("target batch pre-submission control is absent or unsafe")
        if {path.name for path in control.iterdir()} != expected:
            raise ValueError("target batch pre-submission control inventory differs")
        lines = _control_line_set(
            control / "SUBMISSION_LOCK" / "owner",
            "target batch submission-lock owner",
        )
        if lines != {
            f"protocol_id={PROTOCOL_ID}",
            "stage=completion_batch",
            f"batch_id={batch_id(batch_index)}",
            f"repository_commit={repository_commit(repo_root)}",
            "restart_or_resume_authorized=false",
            "retry_or_replacement_authorized=false",
            "automatic_next_batch_authorized=false",
        }:
            raise ValueError("target batch submission-lock owner differs")
    if not authorization_may_exist and os.path.lexists(authorization):
        raise ValueError("target batch authorization is nonreusable")


def write_authorization(args):
    audit_acknowledgments(args)
    _require_target_fresh(
        args.output_root,
        args.repo_root,
        args.batch_index,
        authorization_may_exist=False,
    )
    path = authorization_path(args.output_root, args.batch_index)
    payload = seal(
        expected_body(
            args.output_root,
            args.repo_root,
            args.batch_index,
            dt.datetime.now(dt.timezone.utc).isoformat(),
        )
    )
    write_new(path, payload)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_AUTHORIZED",
                "batch_id": batch_id(args.batch_index),
                "authorized_gpu_jobs": 1,
                "h200_minutes_cap": H200_MINUTES,
                "maximum_cost_usd": float(BATCH_CAP_USD),
                "conservative_program_maximum_usd": float(
                    maximum_exposure(args.batch_index)
                ),
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "another_batch_authorized": False,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def verify_authorization(args):
    _require_target_fresh(
        args.output_root,
        args.repo_root,
        args.batch_index,
        authorization_may_exist=True,
        generation_may_exist=getattr(args, "allow_generation", False),
        running_state_may_exist=getattr(args, "allow_running_state", False),
        slurm_job_id=getattr(args, "slurm_job_id", None),
    )
    path = authorization_path(args.output_root, args.batch_index)
    payload = load_json(path, "Kalai s=1 completion-batch authority")
    body = verify_seal(payload, "Kalai s=1 completion-batch authority")
    expected = expected_body(
        args.output_root,
        args.repo_root,
        args.batch_index,
        body.get("created_at"),
    )
    if body != expected:
        raise ValueError("completion-batch authority binding differs")
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_AUTHORITY_VALID",
                "batch_id": batch_id(args.batch_index),
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert Decimal(H200_MINUTES) * H200_HOURLY_USD / Decimal(60) == BATCH_CAP_USD
    assert current_exposure(1) == Decimal("6.12198425")
    assert maximum_exposure(1) == Decimal("7.02198425")
    assert current_exposure(7) == Decimal("11.52198425")
    assert maximum_exposure(7) == Decimal("12.42198425")
    assert maximum_exposure(7) < PROGRAM_CEILING_USD
    print("MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_AUTH_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    verify = subparsers.add_parser("verify")
    subparsers.add_parser("self-test")
    for item in (write, verify):
        item.add_argument("--output-root", required=True)
        item.add_argument("--repo-root", required=True)
        item.add_argument(
            "--batch-index", type=int, choices=BATCH_INDICES, required=True
        )
    verify.add_argument("--running-job-id")
    write.add_argument("--ack-h200-minutes", type=int, required=True)
    write.add_argument("--ack-max-cost-usd", required=True)
    write.add_argument(
        "--ack-current-conservative-exposure-usd", required=True
    )
    write.add_argument("--ack-conservative-program-max-usd", required=True)
    write.add_argument("--ack-program-ceiling-usd", required=True)
    write.add_argument("--ack-no-api", action="store_true")
    write.add_argument("--ack-no-automatic-next-batch", action="store_true")
    write.add_argument(
        "--ack-no-restart-resume-retry-replacement", action="store_true"
    )
    write.add_argument("--ack-prior-batch-caps-retained", action="store_true")
    write.add_argument("--ack-post-hoc-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test()
    elif args.command == "write":
        write_authorization(args)
    else:
        args.allow_running_state = args.running_job_id is not None
        args.slurm_job_id = args.running_job_id
        verify_authorization(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
