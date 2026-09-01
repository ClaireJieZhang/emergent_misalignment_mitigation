#!/usr/bin/env python3
"""Write/verify the exact one-shot Kalai completion GPU authority."""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load(
    "_kalai_s3_completion_prepare_for_authority",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s3_completion_v1.py",
)
gate_authorizer = _load(
    "_kalai_s3_gate_authorizer_for_completion_authority",
    SCRIPT_DIR / "authorize_massive_medical_kalai_s3_v2.py",
)
sampler = prepare.sampler


COMPLETION_CONTROL_NAMES = (
    "COMPLETION_AUTHORIZATION.json",
    "COMPLETION_SUBMISSION_LOCK",
    "COMPLETION_SUBMISSION_ATTEMPT.tsv",
    "COMPLETION_SUBMITTED",
    "COMPLETION_RELEASE_AUTHORIZED",
    "COMPLETION_RELEASED",
    "COMPLETION_INVOCATION_LOCK",
    "COMPLETION_RESULT.json",
    "COMPLETION_STOPPED",
)


def _read_text(path, description):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _git_commit(repo):
    return subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean(repo, description):
    if subprocess.check_output(
        ["git", "-C", repo, "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(f"{description} repository differs")


def _authorization_path(gate_output):
    return os.path.join(
        os.path.abspath(gate_output), "control", "COMPLETION_AUTHORIZATION.json"
    )


def audit_static(controller_output, controller_repo, gate_output, gate_repo):
    controller_output = os.path.abspath(controller_output)
    controller_repo = os.path.abspath(controller_repo)
    gate_output = os.path.abspath(gate_output)
    gate_repo = os.path.abspath(gate_repo)
    plan_path = os.path.join(controller_output, "control", "COMPLETION_PLAN.json")
    stage_path = os.path.join(controller_output, "control", "CPU_STAGE.json")
    plan = prepare.load_json(plan_path, "completion plan")
    stage = prepare.load_json(stage_path, "controller CPU stage")
    plan_body = prepare.verify_seal(plan, "completion plan")
    stage_body = prepare.verify_seal(stage, "controller CPU stage")
    _require_clean(controller_repo, "controller")
    _require_clean(gate_repo, "gate")
    if (
        stage_body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or stage_body.get("controller_protocol_id")
        != prepare.CONTROLLER_PROTOCOL_ID
        or stage_body.get("repository_commit") != _git_commit(controller_repo)
        or stage_body.get("source_gate_repository_path") != gate_repo
        or stage_body.get("source_gate_repository_commit") != _git_commit(gate_repo)
        or stage_body.get("source_gate_output_root") != gate_output
        or stage_body.get("gpu_authorized") is not False
        or stage_body.get("external_api_authorized") is not False
    ):
        raise ValueError("controller CPU-stage binding differs")
    if stage_body.get("completion_plan") != prepare.binding(plan_path, plan):
        raise ValueError("completion-plan binding differs")
    for name, expected in stage_body.get("implementation_sha256", {}).items():
        path = os.path.join(controller_repo, name)
        if (
            os.path.islink(path)
            or not os.path.isfile(path)
            or prepare.sha256_file(path) != expected
        ):
            raise ValueError(f"controller implementation differs: {name}")
    gate_authorizer.audit_cpu_stage(gate_output, gate_repo)
    gate_result_path = os.path.join(gate_output, "control", "GATE_RESULT.json")
    gate_result = sampler.verify_passing_gate(gate_result_path)
    if (
        plan_body.get("status") != "COMPLETION_PLANNED_NOT_AUTHORIZED"
        or plan_body.get("controller_protocol_id")
        != prepare.CONTROLLER_PROTOCOL_ID
        or plan_body.get("source_protocol_id") != sampler.PROTOCOL_ID
        or plan_body.get("method_id") != sampler.METHOD_ID
        or plan_body.get("source_gate", {}).get("repository_path") != gate_repo
        or plan_body.get("source_gate", {}).get("repository_commit")
        != _git_commit(gate_repo)
        or plan_body.get("source_gate", {}).get("output_root") != gate_output
        or plan_body.get("source_gate", {}).get("gate_result")
        != prepare.binding(gate_result_path, gate_result)
        or plan_body.get("source_gate", {}).get(
            "gate_actual_estimated_cost_usd"
        )
        != float(prepare.GATE_ACTUAL_USD)
        or plan_body.get("planned_authority_not_yet_granted", {}).get(
            "h200_minutes"
        )
        != prepare.COMPLETION_H200_MINUTES
        or Decimal(
            str(
                plan_body.get("planned_authority_not_yet_granted", {}).get(
                    "maximum_cost_usd"
                )
            )
        )
        != prepare.COMPLETION_CAP_USD
        or Decimal(
            str(
                plan_body.get("planned_authority_not_yet_granted", {}).get(
                    "program_ceiling_usd"
                )
            )
        )
        != prepare.WORKFLOW_CEILING_USD
    ):
        raise ValueError("completion plan authority or gate binding differs")
    return plan_path, plan, stage_path, stage, gate_result_path, gate_result


def audit_completion_state(gate_output, expected, job_id=None):
    control = os.path.join(os.path.abspath(gate_output), "control")
    actual = {
        name for name in COMPLETION_CONTROL_NAMES if os.path.lexists(os.path.join(control, name))
    }
    if actual != set(expected):
        raise ValueError(
            f"completion control state differs: expected={sorted(expected)} actual={sorted(actual)}"
        )
    lock = os.path.join(control, "COMPLETION_SUBMISSION_LOCK", "owner")
    owner = _read_text(lock, "completion lock owner")
    required_owner = (
        "controller_protocol_id=" + prepare.CONTROLLER_PROTOCOL_ID,
        "stage=completion",
        "restart_or_resume_authorized=false",
        "retry_authorized=false",
    )
    if any(line not in owner.splitlines() for line in required_owner):
        raise ValueError("completion lock owner differs")
    if job_id is not None:
        if not job_id.isdigit():
            raise ValueError("job id differs")
        for name in (
            "COMPLETION_SUBMITTED",
            "COMPLETION_RELEASE_AUTHORIZED",
            "COMPLETION_RELEASED",
        ):
            text = _read_text(os.path.join(control, name), name)
            if f"job_id={job_id}" not in text.splitlines():
                raise ValueError(f"{name} job binding differs")
    return owner


def expected_body(
    controller_output,
    controller_repo,
    gate_output,
    gate_repo,
    created_at,
):
    (
        plan_path,
        plan,
        stage_path,
        stage,
        gate_result_path,
        gate_result,
    ) = audit_static(controller_output, controller_repo, gate_output, gate_repo)
    return {
        "schema_version": 1,
        "protocol_id": sampler.PROTOCOL_ID,
        "method_id": sampler.METHOD_ID,
        "stage": "completion",
        "controller_protocol_id": prepare.CONTROLLER_PROTOCOL_ID,
        "created_at": created_at,
        "controller_repository_commit": _git_commit(controller_repo),
        "source_gate_repository_commit": _git_commit(gate_repo),
        "controller_cpu_stage": prepare.binding(stage_path, stage),
        "completion_plan": prepare.binding(plan_path, plan),
        "prerequisite": {
            "gate_result": prepare.binding(gate_result_path, gate_result),
            "gate_actual_estimated_cost_usd": float(prepare.GATE_ACTUAL_USD),
        },
        "authorized_gpu_jobs": 1,
        "h200_minutes_cap": prepare.COMPLETION_H200_MINUTES,
        "h200_rate_usd_per_hour": 0.90,
        "maximum_cost_usd": float(prepare.COMPLETION_CAP_USD),
        "prior_conservative_exposure_usd": float(
            prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD
        ),
        "conservative_actual_plus_new_cap_usd": float(
            prepare.MAXIMUM_WITH_COMPLETION_CAP_USD
        ),
        "program_ceiling_usd": float(prepare.WORKFLOW_CEILING_USD),
        "shared_model_loads": 1,
        "benefit_completion_requests": 358,
        "medical_completion_requests": 64,
        "external_api_calls_authorized": 0,
        "automatic_continuation_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_authorized": False,
    }


def write(args):
    audit_completion_state(args.gate_output, {"COMPLETION_SUBMISSION_LOCK"})
    path = _authorization_path(args.gate_output)
    if os.path.lexists(path):
        raise ValueError("completion authority already exists and is nonreusable")
    if args.ack_h200_minutes != prepare.COMPLETION_H200_MINUTES:
        raise ValueError("completion minute acknowledgment differs")
    if Decimal(args.ack_max_cost_usd) != prepare.COMPLETION_CAP_USD:
        raise ValueError("completion cost acknowledgment differs")
    if Decimal(args.ack_program_ceiling_usd) != prepare.WORKFLOW_CEILING_USD:
        raise ValueError("program-ceiling acknowledgment differs")
    body = expected_body(
        args.controller_output,
        args.controller_repo,
        args.gate_output,
        args.gate_repo,
        dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    payload = prepare.seal(body)
    prepare.write_new(path, payload)
    print(
        json.dumps(
            {
                "status": "KALAI_S3_COMPLETION_V1_GPU_AUTHORIZED",
                "h200_minutes_cap": body["h200_minutes_cap"],
                "maximum_cost_usd": body["maximum_cost_usd"],
                "conservative_actual_plus_new_cap_usd": body[
                    "conservative_actual_plus_new_cap_usd"
                ],
                "program_ceiling_usd": body["program_ceiling_usd"],
                "authorization_payload_sha256": payload[prepare.SEAL_FIELD],
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def verify(args):
    if args.expected_job_id:
        expected = {
            "COMPLETION_AUTHORIZATION.json",
            "COMPLETION_SUBMISSION_LOCK",
            "COMPLETION_SUBMISSION_ATTEMPT.tsv",
            "COMPLETION_SUBMITTED",
            "COMPLETION_RELEASE_AUTHORIZED",
            "COMPLETION_RELEASED",
        }
    else:
        expected = {
            "COMPLETION_AUTHORIZATION.json",
            "COMPLETION_SUBMISSION_LOCK",
        }
    audit_completion_state(args.gate_output, expected, args.expected_job_id)
    path = _authorization_path(args.gate_output)
    payload = prepare.load_json(path, "completion authorization")
    body = prepare.verify_seal(payload, "completion authorization")
    expected_body_value = expected_body(
        args.controller_output,
        args.controller_repo,
        args.gate_output,
        args.gate_repo,
        body.get("created_at"),
    )
    if body != expected_body_value:
        raise ValueError("completion authorization binding differs")
    print(
        json.dumps(
            {
                "status": "KALAI_S3_COMPLETION_V1_GPU_AUTHORIZATION_VALID",
                "authorization_payload_sha256": payload[prepare.SEAL_FIELD],
                "expected_job_id": args.expected_job_id,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert prepare.COMPLETION_H200_MINUTES == 94
    assert prepare.COMPLETION_CAP_USD == Decimal("1.410")
    assert prepare.MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD < (
        prepare.WORKFLOW_CEILING_USD
    )
    print("KALAI_S3_COMPLETION_V1_AUTH_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("write", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--controller-output", required=True)
        command.add_argument("--controller-repo", required=True)
        command.add_argument("--gate-output", required=True)
        command.add_argument("--gate-repo", required=True)
        if name == "write":
            command.add_argument("--ack-h200-minutes", type=int, required=True)
            command.add_argument("--ack-max-cost-usd", required=True)
            command.add_argument("--ack-program-ceiling-usd", required=True)
        else:
            command.add_argument("--expected-job-id")
    commands.add_parser("self-test")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == "self-test":
        self_test()
    elif args.command == "write":
        write(args)
    else:
        verify(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
