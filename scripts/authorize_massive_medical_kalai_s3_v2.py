#!/usr/bin/env python3
"""Write or verify one explicit, nonreusable Kalai s=3 GPU authority."""

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


sampler = _load_module(
    "_massive_medical_kalai_s3_sampler_for_authority",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_s3_v2.py",
)
prepare = _load_module(
    "_massive_medical_kalai_s3_prepare_for_authority",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s3_v2.py",
)


STAGES = ("gate", "completion")
SEAL_FIELD = "payload_sha256"


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


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def audit_cpu_stage(output_root, repo_root):
    path = os.path.join(output_root, "control", "CPU_STAGE.json")
    payload = load_json(path, "Kalai s=3 CPU stage")
    body = verify_seal(payload, "Kalai s=3 CPU stage")
    if (
        body.get("protocol_id") != sampler.PROTOCOL_ID
        or body.get("method_id") != sampler.METHOD_ID
        or body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or body.get("gpu_jobs_submitted") != 0
        or body.get("external_api_calls") != 0
        or body.get("gpu_authorized") is not False
        or body.get("external_api_authorized") is not False
        or body.get("restart_or_resume_authorized") is not False
        or body.get("repository_commit") != repo_commit(repo_root)
    ):
        raise ValueError("Kalai s=3 CPU-stage binding differs")
    if subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("repository worktree differs from the CPU-staged commit")
    implementation = body.get("implementation_sha256")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("CPU stage lacks implementation bindings")
    for relative_path, expected_sha256 in implementation.items():
        path = os.path.join(repo_root, relative_path)
        if (
            os.path.islink(path)
            or not os.path.isfile(path)
            or sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"CPU-staged implementation differs: {relative_path}")
    plan_binding = body.get("gate_plan")
    plan_path = os.path.join(output_root, "control", "GATE_PLAN.json")
    if (
        not isinstance(plan_binding, dict)
        or plan_binding.get("path") != os.path.abspath(plan_path)
        or plan_binding.get("file_sha256") != sha256_file(plan_path)
    ):
        raise ValueError("CPU-stage gate-plan file binding differs")
    plan = load_json(plan_path, "gate plan")
    verify_seal(plan, "gate plan")
    if plan_binding.get("payload_sha256") != plan.get(SEAL_FIELD):
        raise ValueError("CPU-stage gate-plan payload binding differs")
    return path, payload


def gate_actual(output_root):
    path = os.path.join(output_root, "control", "GATE_RESULT.json")
    payload = sampler.verify_passing_gate(path)
    body = verify_seal(payload, "Kalai s=3 gate result")
    actual = Decimal(str(body["timing"]["actual_estimated_cost_usd"]))
    if actual <= 0 or actual > Decimal(str(prepare.PLANNED_GATE_CAP_USD)):
        raise ValueError("gate actual cost is outside its authority")
    return path, payload, actual


def expected_body(
    *,
    output_root,
    repo_root,
    stage,
    minutes,
    cap,
    ceiling,
    created_at,
):
    stage_path, stage_payload = audit_cpu_stage(output_root, repo_root)
    if minutes <= 0 or cap <= 0:
        raise ValueError("GPU cap must be positive")
    if ceiling != Decimal(str(prepare.WORKFLOW_CEILING_USD)):
        raise ValueError("program ceiling differs from the CPU-staged ceiling")
    exact_cost = Decimal(minutes) * Decimal("0.90") / Decimal(60)
    if cap != exact_cost:
        raise ValueError("GPU cost cap must equal minutes at $0.90/H200-hour")
    if stage == "gate":
        if minutes != prepare.PLANNED_GATE_H200_MINUTES or cap != Decimal(
            str(prepare.PLANNED_GATE_CAP_USD)
        ):
            raise ValueError("gate cap differs from the CPU-stage plan")
        prior = Decimal(str(prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD))
        prerequisite = None
    else:
        gate_path, gate_payload, gate_cost = gate_actual(output_root)
        prior = Decimal(str(prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD)) + gate_cost
        prerequisite = {
            "gate_result_path": os.path.abspath(gate_path),
            "gate_result_file_sha256": sha256_file(gate_path),
            "gate_result_payload_sha256": gate_payload[SEAL_FIELD],
            "gate_actual_estimated_cost_usd": float(gate_cost),
        }
    maximum = prior + cap
    if maximum > ceiling:
        raise ValueError(f"stage maximum {maximum} exceeds ceiling {ceiling}")
    return {
        "schema_version": 1,
        "protocol_id": sampler.PROTOCOL_ID,
        "method_id": sampler.METHOD_ID,
        "stage": stage,
        "created_at": created_at,
        "repository_commit": repo_commit(repo_root),
        "cpu_stage_manifest_path": os.path.abspath(stage_path),
        "cpu_stage_manifest_file_sha256": sha256_file(stage_path),
        "cpu_stage_manifest_payload_sha256": stage_payload[SEAL_FIELD],
        "prerequisite": prerequisite,
        "authorized_gpu_jobs": 1,
        "h200_minutes_cap": minutes,
        "h200_rate_usd_per_hour": 0.90,
        "maximum_cost_usd": float(cap),
        "prior_conservative_exposure_usd": float(prior),
        "conservative_actual_plus_new_cap_usd": float(maximum),
        "program_ceiling_usd": float(ceiling),
        "external_api_calls_authorized": 0,
        "automatic_continuation_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_authorized": False,
    }


def write_new(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_authorization(args):
    output_root = os.path.abspath(args.output_root)
    repo_root = os.path.abspath(args.repo_root)
    minutes = args.ack_h200_minutes
    cap = Decimal(args.ack_max_cost_usd)
    ceiling = Decimal(args.ack_program_ceiling_usd)
    path = authorization_path(output_root, args.stage)
    if os.path.lexists(path):
        raise ValueError("authorization already exists and is nonreusable")
    body = expected_body(
        output_root=output_root,
        repo_root=repo_root,
        stage=args.stage,
        minutes=minutes,
        cap=cap,
        ceiling=ceiling,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    payload = seal(body)
    write_new(path, payload)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S3_GPU_STAGE_AUTHORIZED",
                "stage": args.stage,
                "h200_minutes_cap": minutes,
                "maximum_cost_usd": float(cap),
                "conservative_actual_plus_new_cap_usd": body[
                    "conservative_actual_plus_new_cap_usd"
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
    output_root = os.path.abspath(args.output_root)
    repo_root = os.path.abspath(args.repo_root)
    path = authorization_path(output_root, args.stage)
    payload = load_json(path, f"{args.stage} authorization")
    body = verify_seal(payload, f"{args.stage} authorization")
    expected = expected_body(
        output_root=output_root,
        repo_root=repo_root,
        stage=args.stage,
        minutes=int(body.get("h200_minutes_cap")),
        cap=Decimal(str(body.get("maximum_cost_usd"))),
        ceiling=Decimal(str(body.get("program_ceiling_usd"))),
        created_at=body.get("created_at"),
    )
    if body != expected:
        raise ValueError("authorization binding differs")
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S3_GPU_AUTHORIZATION_VALID",
                "stage": args.stage,
                "authorization_payload_sha256": payload[SEAL_FIELD],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert Decimal(20) * Decimal("0.90") / Decimal(60) == Decimal("0.30")
    assert Decimal(str(prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD)) + Decimal(
        "0.30"
    ) == Decimal("4.9248935")
    assert Decimal(str(prepare.WORKFLOW_CEILING_USD)) == Decimal("5.9933725")
    print("MASSIVE_MEDICAL_KALAI_S3_V2_AUTH_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write")
    verify = commands.add_parser("verify")
    for command in (write, verify):
        command.add_argument("--stage", choices=STAGES, required=True)
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
