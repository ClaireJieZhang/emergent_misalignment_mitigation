#!/usr/bin/env python3
"""Write or verify the one-shot Kalai s=1 technical-gate authority.

The writer is deliberately exact: it accepts only the CPU-staged 35-minute
technical gate and never authorizes completion, judging, retry, or resume.
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


PROTOCOL_ID = "massive_medical_kalai_s1_r20_trace_reuse_v1"
METHOD_ID = "whole_output_consensus_m4_s1_r20_sensitivity_v1"
SEAL_FIELD = "payload_sha256"
H200_HOURLY_USD = Decimal("0.90")
GATE_H200_MINUTES = 35
GATE_CAP_USD = Decimal("0.525")
KNOWN_PROGRAM_ACTUAL_USD = Decimal("5.03884025")
RETAINED_CONSERVATIVE_EXPOSURE_USD = Decimal("0.756144")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("5.79498425")
MAXIMUM_WITH_GATE_CAP_USD = Decimal("6.31998425")
WORKFLOW_CEILING_USD = Decimal("6.5000000")


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


def binding(path, expected_protocol, expected_method, description):
    path = os.path.abspath(path)
    payload = load_json(path, description)
    body = verify_seal(payload, description)
    if (
        body.get("protocol_id") != expected_protocol
        or body.get("method_id") != expected_method
    ):
        raise ValueError(f"{description} identity differs")
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


def authorization_path(output_root):
    return Path(output_root).resolve() / "control" / "TECHNICAL_GATE_AUTHORIZATION.json"


def expected_body(output_root, repo_root, created_at):
    output_root = Path(output_root).resolve()
    repo_root = Path(repo_root).resolve()
    plan_path = output_root / "control" / "REPLAY_PLAN.json"
    stage_path = output_root / "control" / "CPU_STAGE.json"
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "technical_gate",
        "created_at": created_at,
        "repository_commit": repository_commit(repo_root),
        "replay_plan": binding(
            plan_path, PROTOCOL_ID, METHOD_ID, "Kalai s=1 replay plan"
        ),
        "cpu_stage": binding(
            stage_path, PROTOCOL_ID, METHOD_ID, "Kalai s=1 CPU stage"
        ),
        "authorized_gpu_jobs": 1,
        "h200_count": 1,
        "h200_minutes_cap": GATE_H200_MINUTES,
        "h200_hourly_usd": float(H200_HOURLY_USD),
        "maximum_cost_usd": float(GATE_CAP_USD),
        "known_program_actual_usd": float(KNOWN_PROGRAM_ACTUAL_USD),
        "retained_conservative_exposure_usd": float(
            RETAINED_CONSERVATIVE_EXPOSURE_USD
        ),
        "current_conservative_exposure_usd": float(
            CURRENT_CONSERVATIVE_EXPOSURE_USD
        ),
        "conservative_actual_plus_new_cap_usd": float(
            MAXIMUM_WITH_GATE_CAP_USD
        ),
        "program_ceiling_usd": float(WORKFLOW_CEILING_USD),
        "external_api_calls_authorized": 0,
        "completion_authorized": False,
        "automatic_continuation_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_or_requeue_authorized": False,
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


def audit_acknowledgments(args):
    if args.ack_h200_minutes != GATE_H200_MINUTES:
        raise ValueError("H200-minute acknowledgment differs")
    if Decimal(args.ack_max_cost_usd) != GATE_CAP_USD:
        raise ValueError("cost acknowledgment differs")
    if Decimal(args.ack_current_conservative_exposure_usd) != (
        CURRENT_CONSERVATIVE_EXPOSURE_USD
    ):
        raise ValueError("current conservative exposure acknowledgment differs")
    if Decimal(args.ack_conservative_program_max_usd) != MAXIMUM_WITH_GATE_CAP_USD:
        raise ValueError("program maximum acknowledgment differs")
    if Decimal(args.ack_program_ceiling_usd) != WORKFLOW_CEILING_USD:
        raise ValueError("program ceiling acknowledgment differs")
    required_true = (
        "ack_no_api",
        "ack_no_completion",
        "ack_no_restart_resume_retry",
        "ack_post_hoc_sensitivity",
    )
    if any(getattr(args, name) is not True for name in required_true):
        raise ValueError("required negative-authority acknowledgment is absent")


def write_authorization(args):
    audit_acknowledgments(args)
    path = authorization_path(args.output_root)
    if os.path.lexists(path):
        raise ValueError("technical-gate authority already exists and is nonreusable")
    payload = seal(
        expected_body(
            args.output_root,
            args.repo_root,
            dt.datetime.now(dt.timezone.utc).isoformat(),
        )
    )
    write_new(path, payload)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_TECHNICAL_GATE_AUTHORIZED",
                "authorized_gpu_jobs": 1,
                "h200_minutes_cap": GATE_H200_MINUTES,
                "maximum_cost_usd": float(GATE_CAP_USD),
                "conservative_program_maximum_usd": float(
                    MAXIMUM_WITH_GATE_CAP_USD
                ),
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def verify_authorization(args):
    path = authorization_path(args.output_root)
    payload = load_json(path, "Kalai s=1 technical-gate authority")
    body = verify_seal(payload, "Kalai s=1 technical-gate authority")
    expected = expected_body(
        args.output_root, args.repo_root, body.get("created_at")
    )
    if body != expected:
        raise ValueError("technical-gate authority binding differs")
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_TECHNICAL_GATE_AUTHORITY_VALID",
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert Decimal(GATE_H200_MINUTES) * H200_HOURLY_USD / Decimal(60) == (
        GATE_CAP_USD
    )
    assert KNOWN_PROGRAM_ACTUAL_USD + RETAINED_CONSERVATIVE_EXPOSURE_USD == (
        CURRENT_CONSERVATIVE_EXPOSURE_USD
    )
    assert CURRENT_CONSERVATIVE_EXPOSURE_USD + GATE_CAP_USD == (
        MAXIMUM_WITH_GATE_CAP_USD
    )
    assert MAXIMUM_WITH_GATE_CAP_USD < WORKFLOW_CEILING_USD
    print("MASSIVE_MEDICAL_KALAI_S1_TRACE_REUSE_V1_AUTH_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    verify = subparsers.add_parser("verify")
    subparsers.add_parser("self-test")
    for item in (write, verify):
        item.add_argument("--output-root", required=True)
        item.add_argument("--repo-root", required=True)
    write.add_argument("--ack-h200-minutes", type=int, required=True)
    write.add_argument("--ack-max-cost-usd", required=True)
    write.add_argument("--ack-current-conservative-exposure-usd", required=True)
    write.add_argument("--ack-conservative-program-max-usd", required=True)
    write.add_argument("--ack-program-ceiling-usd", required=True)
    write.add_argument("--ack-no-api", action="store_true")
    write.add_argument("--ack-no-completion", action="store_true")
    write.add_argument("--ack-no-restart-resume-retry", action="store_true")
    write.add_argument("--ack-post-hoc-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test()
    elif args.command == "write":
        write_authorization(args)
    else:
        verify_authorization(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
