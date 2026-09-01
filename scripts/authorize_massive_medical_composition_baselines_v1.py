#!/usr/bin/env python3
"""Write or verify one explicit GPU-stage authorization for baseline v1.

CPU staging does not call this command in write mode.  A write is permitted
only after the user explicitly authorizes the exact stage cap and a new
program ceiling.  Authorizations are one-shot and never imply restart, resume,
dependency release, a full whole-output run, or an external judge call.
"""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess


PROTOCOL_ID = "massive_medical_composition_baselines_v1"
PRIOR_ACCOUNTED_EXPOSURE_USD = Decimal("4.4183725")
STAGES = {
    "union_training": (55, Decimal("0.825")),
    "direct_generation": (30, Decimal("0.450")),
    "whole_output_smoke": (20, Decimal("0.300")),
}
SEAL_FIELD = "payload_sha256"


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


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
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def repo_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()


def authorization_path(output_root, stage):
    return os.path.join(
        os.path.abspath(output_root),
        "control",
        stage.upper() + "_AUTHORIZATION.json",
    )


def active_authorized_caps(output_root, excluding=None):
    values = {}
    for stage, (_, cap) in STAGES.items():
        if stage == excluding:
            continue
        path = authorization_path(output_root, stage)
        if os.path.isfile(path):
            payload = load_json(path, f"{stage} authorization")
            body = verify_seal(payload, f"{stage} authorization")
            if body.get("stage") != stage or Decimal(
                str(body.get("maximum_cost_usd"))
            ) != cap:
                raise ValueError(f"existing {stage} authorization differs")
            values[stage] = cap
    return values


def audit_cpu_stage(output_root):
    path = os.path.join(os.path.abspath(output_root), "control", "CPU_STAGE.json")
    payload = load_json(path, "baseline CPU-stage manifest")
    body = verify_seal(payload, "baseline CPU-stage manifest")
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or body.get("gpu_jobs_submitted") != 0
        or body.get("external_judge_calls") != 0
        or body.get("gpu_authorized") is not False
        or body.get("external_judge_authorized") is not False
    ):
        raise ValueError("baseline CPU-stage status differs")
    return path, payload, body


def write_new(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def expected_body(
    output_root,
    repo_root,
    stage,
    program_ceiling,
    created_at,
    previous_authorized_caps=None,
):
    minutes, cap = STAGES[stage]
    stage_path, stage_payload, stage_body = audit_cpu_stage(output_root)
    current_commit = repo_commit(repo_root)
    if stage_body.get("repository_commit") != current_commit:
        raise ValueError("repository commit differs from CPU-stage manifest")
    previous = (
        active_authorized_caps(output_root, excluding=stage)
        if previous_authorized_caps is None
        else dict(previous_authorized_caps)
    )
    prior_new_caps = sum(previous.values(), Decimal("0"))
    conservative_maximum = PRIOR_ACCOUNTED_EXPOSURE_USD + prior_new_caps + cap
    if conservative_maximum > program_ceiling:
        raise ValueError(
            f"stage would exceed program ceiling: {conservative_maximum} > "
            f"{program_ceiling}"
        )
    if stage == "direct_generation":
        result = os.path.join(output_root, "control", "UNION_TRAINING_RESULT.json")
        if not os.path.isfile(result):
            raise ValueError("direct generation requires a sealed Union-training result")
        result_payload = load_json(result, "Union-training result")
        result_body = verify_seal(result_payload, "Union-training result")
        if (
            result_body.get("protocol_id") != PROTOCOL_ID
            or result_body.get("status") != "GPU_STAGE_COMPLETE"
            or result_body.get("stage") != "union_training"
            or result_body.get("external_api_calls") != 0
            or result_body.get("restart_or_resume_used") is not False
        ):
            raise ValueError("sealed Union-training result differs")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "created_at": created_at,
        "repository_commit": current_commit,
        "cpu_stage_manifest_file_sha256": sha256_file(stage_path),
        "cpu_stage_manifest_payload_sha256": stage_payload[SEAL_FIELD],
        "authorized_gpu_jobs": 1,
        "h200_minutes_cap": minutes,
        "h200_rate_usd_per_hour": 0.90,
        "maximum_cost_usd": float(cap),
        "prior_accounted_exposure_usd": float(PRIOR_ACCOUNTED_EXPOSURE_USD),
        "previous_new_authorized_caps_usd": {
            name: float(value) for name, value in sorted(previous.items())
        },
        "conservative_actual_plus_new_caps_usd": float(conservative_maximum),
        "program_ceiling_usd": float(program_ceiling),
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_decision_may_change": False,
        "external_api_calls_authorized": 0,
        "automatic_continuation_authorized": False,
        "restart_or_resume_authorized": False,
        "full_whole_output_run_authorized": False,
    }


def write_authorization(args):
    expected_minutes, expected_cap = STAGES[args.stage]
    if args.ack_h200_minutes != expected_minutes:
        raise ValueError("acknowledged H200-minute cap differs")
    if Decimal(args.ack_max_cost_usd) != expected_cap:
        raise ValueError("acknowledged GPU-cost cap differs")
    ceiling = Decimal(args.ack_program_ceiling_usd)
    if ceiling <= PRIOR_ACCOUNTED_EXPOSURE_USD:
        raise ValueError("new program ceiling must exceed prior accounted exposure")
    path = authorization_path(args.output_root, args.stage)
    if os.path.lexists(path):
        raise ValueError("stage authorization already exists and is nonreusable")
    body = expected_body(
        os.path.abspath(args.output_root),
        os.path.abspath(args.repo_root),
        args.stage,
        ceiling,
        dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    payload = seal(body)
    write_new(path, payload)
    print(
        json.dumps(
            {
                "status": "BASELINE_GPU_STAGE_AUTHORIZED",
                "stage": args.stage,
                "h200_minutes_cap": expected_minutes,
                "maximum_cost_usd": float(expected_cap),
                "conservative_actual_plus_new_caps_usd": body[
                    "conservative_actual_plus_new_caps_usd"
                ],
                "program_ceiling_usd": body["program_ceiling_usd"],
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def verify_authorization(args):
    path = authorization_path(args.output_root, args.stage)
    payload = load_json(path, f"{args.stage} authorization")
    body = verify_seal(payload, f"{args.stage} authorization")
    ceiling = Decimal(str(body.get("program_ceiling_usd")))
    recorded_previous = body.get("previous_new_authorized_caps_usd")
    if not isinstance(recorded_previous, dict):
        raise ValueError("authorization lacks prior new-cap accounting")
    previous = {
        name: Decimal(str(value)) for name, value in recorded_previous.items()
    }
    if any(name not in STAGES or name == args.stage for name in previous):
        raise ValueError("authorization prior new-cap stage set differs")
    for name, value in previous.items():
        if value != STAGES[name][1] or not os.path.isfile(
            authorization_path(args.output_root, name)
        ):
            raise ValueError("authorization prior new-cap binding differs")
    expected = expected_body(
        os.path.abspath(args.output_root),
        os.path.abspath(args.repo_root),
        args.stage,
        ceiling,
        body.get("created_at"),
        previous_authorized_caps=previous,
    )
    if body != expected:
        raise ValueError("stage authorization binding differs")
    print(
        json.dumps(
            {
                "status": "BASELINE_GPU_STAGE_AUTHORIZATION_VALID",
                "stage": args.stage,
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert sum(value[0] for value in STAGES.values()) == 105
    assert sum((value[1] for value in STAGES.values()), Decimal("0")) == Decimal(
        "1.575"
    )
    assert PRIOR_ACCOUNTED_EXPOSURE_USD + Decimal("1.575") == Decimal(
        "5.9933725"
    )
    print("MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_AUTH_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write")
    verify = commands.add_parser("verify")
    for command in (write, verify):
        command.add_argument("--stage", choices=tuple(STAGES), required=True)
        command.add_argument("--output-root", required=True)
        command.add_argument("--repo-root", required=True)
    write.add_argument("--ack-h200-minutes", type=int, required=True)
    write.add_argument("--ack-max-cost-usd", required=True)
    write.add_argument("--ack-program-ceiling-usd", required=True)
    commands.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test()
        return 0
    os.umask(0o077)
    if args.command == "write":
        write_authorization(args)
    else:
        verify_authorization(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
