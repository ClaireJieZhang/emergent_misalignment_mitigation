#!/usr/bin/env python3
"""CPU-stage the bounded one-shot Kalai s=3 completion controller."""

from __future__ import annotations

import argparse
from decimal import Decimal
import datetime as dt
import hashlib
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


sampler = _load(
    "_kalai_s3_sampler_for_completion_prepare",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_s3_v2.py",
)
gate_prepare = _load(
    "_kalai_s3_gate_prepare_for_completion_prepare",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s3_v2.py",
)
gate_authorizer = _load(
    "_kalai_s3_gate_authorizer_for_completion_prepare",
    SCRIPT_DIR / "authorize_massive_medical_kalai_s3_v2.py",
)


CONTROLLER_PROTOCOL_ID = "massive_medical_kalai_s3_r20_v2_completion_v1"
EXPECTED_OUTPUT_LEAF = "massive_medical_kalai_s3_r20_v2_completion_controller_v1"
EXPECTED_GATE_OUTPUT_LEAF = gate_prepare.EXPECTED_OUTPUT_LEAF
EXPECTED_GATE_REPO_COMMIT = "ed950b72396dc041d34bbb694ea1486763033657"
COMPLETION_H200_MINUTES = 94
COMPLETION_CAP_USD = Decimal("1.410")
GATE_ACTUAL_USD = Decimal("0.198")
PRIOR_CONSERVATIVE_EXPOSURE_USD = Decimal("4.6248935")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("4.8228935")
WORKFLOW_CEILING_USD = Decimal("6.5000000")
MAXIMUM_WITH_COMPLETION_CAP_USD = Decimal("6.2328935")
REMAINING_CEILING_AFTER_CAP_USD = Decimal("0.2671065")
MAXIMUM_JUDGE_CALLS = 80
GENERIC_JUDGE_CAP_PER_CALL_USD = Decimal("0.003072")
GENERIC_JUDGE_CAP_USD = Decimal("0.245760")
MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD = Decimal("6.4786535")
SEAL_FIELD = "payload_sha256"


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def seal(body):
    payload = dict(body)
    payload.pop(SEAL_FIELD, None)
    payload[SEAL_FIELD] = digest(canonical(payload))
    return payload


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


def binding(path, payload=None):
    path = os.path.abspath(path)
    result = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
    }
    if payload is not None:
        result[SEAL_FIELD] = payload[SEAL_FIELD]
    return result


def write_new(path, payload):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean(repo_root, description):
    if subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(f"{description} repository is not clean")


def audit_gate_source(gate_repo, gate_output):
    gate_repo = os.path.abspath(gate_repo)
    gate_output = os.path.abspath(gate_output)
    if os.path.basename(gate_output) != EXPECTED_GATE_OUTPUT_LEAF:
        raise ValueError("gate output namespace differs")
    require_clean(gate_repo, "gate source")
    if git_commit(gate_repo) != EXPECTED_GATE_REPO_COMMIT:
        raise ValueError("gate source commit differs")
    gate_authorizer.audit_cpu_stage(gate_output, gate_repo)
    gate_result_path = os.path.join(gate_output, "control", "GATE_RESULT.json")
    gate_result = sampler.verify_passing_gate(gate_result_path)
    gate_body = verify_seal(gate_result, "gate result")
    actual = Decimal(str(gate_body["timing"]["actual_estimated_cost_usd"]))
    if actual != GATE_ACTUAL_USD:
        raise ValueError("gate actual cost differs")
    forbidden = (
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
    for name in forbidden:
        if os.path.lexists(os.path.join(gate_output, "control", name)):
            raise ValueError(f"gate source already has completion state: {name}")
    if os.path.lexists(os.path.join(gate_output, "generation", "completion")):
        raise ValueError("gate source already has completion generation")
    return gate_result_path, gate_result


def plan_body(gate_repo, gate_output, gate_result_path, gate_result):
    benefit_timing_path = gate_result["observed"]["benefit_timing"]["path"]
    medical_timing_path = gate_result["observed"]["medical_timing"]["path"]
    benefit_timing = load_json(benefit_timing_path, "benefit gate timing")
    medical_timing = load_json(medical_timing_path, "medical gate timing")
    sampler._verify_seal(benefit_timing, "benefit gate timing")
    sampler._verify_seal(medical_timing, "medical gate timing")
    two_process_projection = (
        Decimal(str(benefit_timing["setup_elapsed_seconds"]))
        + Decimal(str(benefit_timing["generation_elapsed_seconds"]))
        * Decimal(358)
        / Decimal(2)
        + Decimal(str(medical_timing["setup_elapsed_seconds"]))
        + Decimal(str(medical_timing["generation_elapsed_seconds"]))
        * Decimal(64)
        / Decimal(16)
    )
    shared_load_projection = two_process_projection - Decimal(
        str(medical_timing["setup_elapsed_seconds"])
    )
    return {
        "schema_version": 1,
        "controller_protocol_id": CONTROLLER_PROTOCOL_ID,
        "source_protocol_id": sampler.PROTOCOL_ID,
        "method_id": sampler.METHOD_ID,
        "status": "COMPLETION_PLANNED_NOT_AUTHORIZED",
        "source_gate": {
            "repository_path": os.path.abspath(gate_repo),
            "repository_commit": git_commit(gate_repo),
            "output_root": os.path.abspath(gate_output),
            "gate_result": binding(gate_result_path, gate_result),
            "gate_actual_estimated_cost_usd": float(GATE_ACTUAL_USD),
            "completion_eligible": True,
        },
        "completion_partition": {
            "benefit_requested_n": 358,
            "medical_requested_n": 64,
            "gate_rows_regenerated": False,
            "maximum_attempts_per_request": 20,
            "shared_model_loads": 1,
            "proposal_or_acceptance_change": False,
            "restart_or_resume_authorized": False,
        },
        "timing_projection_not_guarantee": {
            "basis": "sealed_gate_request_linear_scaling",
            "two_process_seconds": float(two_process_projection),
            "shared_model_load_seconds": float(shared_load_projection),
            "shared_model_load_minutes": float(shared_load_projection / 60),
            "risk": (
                "request_lengths_and_rejection_attempts_vary; timeout_is_terminal"
            ),
        },
        "planned_authority_not_yet_granted": {
            "gpu_jobs": 1,
            "h200_minutes": COMPLETION_H200_MINUTES,
            "h200_rate_usd_per_hour": 0.90,
            "maximum_cost_usd": float(COMPLETION_CAP_USD),
            "prior_conservative_exposure_usd": float(
                CURRENT_CONSERVATIVE_EXPOSURE_USD
            ),
            "conservative_actual_plus_new_cap_usd": float(
                MAXIMUM_WITH_COMPLETION_CAP_USD
            ),
            "program_ceiling_usd": float(WORKFLOW_CEILING_USD),
            "remaining_ceiling_after_cap_usd": float(
                REMAINING_CEILING_AFTER_CAP_USD
            ),
            "external_api_calls": 0,
        },
        "later_judge_envelope_not_yet_authorized": {
            "maximum_calls": MAXIMUM_JUDGE_CALLS,
            "generic_maximum_cost_per_call_usd": float(
                GENERIC_JUDGE_CAP_PER_CALL_USD
            ),
            "generic_maximum_cost_usd": float(GENERIC_JUDGE_CAP_USD),
            "conservative_maximum_after_completion_and_judge_caps_usd": float(
                MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD
            ),
            "program_ceiling_usd": float(WORKFLOW_CEILING_USD),
            "judge_authorized": False,
        },
    }


def prepare(args):
    repo_root = os.path.abspath(args.repo_root)
    output_root = os.path.abspath(args.output_root)
    gate_repo = os.path.abspath(args.gate_repo)
    gate_output = os.path.abspath(args.gate_output)
    if os.path.basename(output_root) != EXPECTED_OUTPUT_LEAF:
        raise ValueError(f"controller output must end in {EXPECTED_OUTPUT_LEAF}")
    require_clean(repo_root, "controller")
    gate_result_path, gate_result = audit_gate_source(gate_repo, gate_output)
    control = os.path.join(output_root, "control")
    plan_path = os.path.join(control, "COMPLETION_PLAN.json")
    stage_path = os.path.join(control, "CPU_STAGE.json")
    if os.path.lexists(output_root) and not os.path.isfile(stage_path):
        raise ValueError("partial completion-controller namespace exists")
    os.makedirs(control, mode=0o700, exist_ok=True)
    plan = seal(
        plan_body(gate_repo, gate_output, gate_result_path, gate_result)
    )
    if os.path.isfile(plan_path):
        observed = load_json(plan_path, "completion plan")
        verify_seal(observed, "completion plan")
        if observed != plan:
            raise ValueError("existing completion plan differs")
    else:
        write_new(plan_path, plan)
    implementation_paths = (
        "scripts/sample_massive_medical_whole_output_consensus_v1.py",
        "scripts/sample_massive_medical_whole_output_consensus_s3_v2.py",
        "scripts/sample_massive_medical_kalai_s3_completion_combined_v1.py",
        "scripts/prepare_massive_medical_kalai_s3_completion_v1.py",
        "scripts/authorize_massive_medical_kalai_s3_completion_v1.py",
        "scripts/evaluate_massive_medical_kalai_s3_completion_v1.py",
        "scripts/assemble_massive_medical_kalai_s3_v2.py",
        "scripts/sbatch_massive_medical_kalai_s3_completion_v1_tillicum_h200.sbatch",
        "scripts/submit_massive_medical_kalai_s3_completion_v1_tillicum.sh",
        "scripts/stage_massive_medical_kalai_s3_completion_v1_tillicum.sh",
        "tests/test_massive_medical_kalai_s3_completion_v1.py",
        "docs/massive_medical_kalai_s3_completion_v1.md",
        "subliminal_mitigate/decoding/algorithms.py",
        "subliminal_mitigate/decoding/__init__.py",
    )
    implementation = {}
    for name in implementation_paths:
        path = os.path.join(repo_root, name)
        if not os.path.isfile(path):
            raise ValueError(f"controller implementation absent: {name}")
        implementation[name] = sha256_file(path)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    if os.path.isfile(stage_path):
        existing = load_json(stage_path, "controller CPU stage")
        created_at = verify_seal(existing, "controller CPU stage")["created_at"]
    stage = seal(
        {
            "schema_version": 1,
            "controller_protocol_id": CONTROLLER_PROTOCOL_ID,
            "source_protocol_id": sampler.PROTOCOL_ID,
            "method_id": sampler.METHOD_ID,
            "status": "CPU_STAGED_NO_GPU_OR_API_AUTHORITY",
            "created_at": created_at,
            "repository_commit": git_commit(repo_root),
            "completion_plan": binding(plan_path, plan),
            "source_gate_repository_path": gate_repo,
            "source_gate_repository_commit": git_commit(gate_repo),
            "source_gate_output_root": gate_output,
            "implementation_sha256": implementation,
            "gpu_jobs_submitted": 0,
            "gpu_authorized": False,
            "external_api_calls": 0,
            "external_api_authorized": False,
            "restart_or_resume_authorized": False,
        }
    )
    if os.path.isfile(stage_path):
        if load_json(stage_path, "controller CPU stage") != stage:
            raise ValueError("existing controller CPU stage differs")
        action = "AUDITED"
    else:
        write_new(stage_path, stage)
        action = "CREATED"
    print(
        json.dumps(
            {
                "status": f"KALAI_S3_COMPLETION_V1_CPU_STAGE_{action}",
                "completion_plan_payload_sha256": plan[SEAL_FIELD],
                "cpu_stage_payload_sha256": stage[SEAL_FIELD],
                "planned_h200_minutes_not_authorized": COMPLETION_H200_MINUTES,
                "planned_max_cost_usd_not_authorized": float(COMPLETION_CAP_USD),
                "conservative_maximum_usd": float(
                    MAXIMUM_WITH_COMPLETION_CAP_USD
                ),
                "program_ceiling_usd": float(WORKFLOW_CEILING_USD),
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def audit_cpu_stage(output_root, repo_root, gate_repo, gate_output):
    output_root = os.path.abspath(output_root)
    repo_root = os.path.abspath(repo_root)
    plan_path = os.path.join(output_root, "control", "COMPLETION_PLAN.json")
    stage_path = os.path.join(output_root, "control", "CPU_STAGE.json")
    plan = load_json(plan_path, "completion plan")
    stage = load_json(stage_path, "controller CPU stage")
    plan_body_value = verify_seal(plan, "completion plan")
    stage_body = verify_seal(stage, "controller CPU stage")
    require_clean(repo_root, "controller")
    if (
        stage_body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or stage_body.get("controller_protocol_id") != CONTROLLER_PROTOCOL_ID
        or stage_body.get("repository_commit") != git_commit(repo_root)
        or stage_body.get("source_gate_repository_path")
        != os.path.abspath(gate_repo)
        or stage_body.get("source_gate_output_root")
        != os.path.abspath(gate_output)
        or stage_body.get("gpu_authorized") is not False
        or stage_body.get("external_api_authorized") is not False
    ):
        raise ValueError("controller CPU-stage binding differs")
    if stage_body.get("completion_plan") != binding(plan_path, plan):
        raise ValueError("controller plan binding differs")
    for name, expected in stage_body["implementation_sha256"].items():
        path = os.path.join(repo_root, name)
        if os.path.islink(path) or sha256_file(path) != expected:
            raise ValueError(f"controller implementation differs: {name}")
    gate_result_path, gate_result = audit_gate_source(gate_repo, gate_output)
    expected_plan = plan_body(gate_repo, gate_output, gate_result_path, gate_result)
    if plan_body_value != expected_plan:
        raise ValueError("completion plan no longer matches its sources")
    return plan, stage


def self_test():
    assert COMPLETION_CAP_USD == Decimal(COMPLETION_H200_MINUTES) * Decimal(
        "0.90"
    ) / 60
    assert PRIOR_CONSERVATIVE_EXPOSURE_USD + GATE_ACTUAL_USD == (
        CURRENT_CONSERVATIVE_EXPOSURE_USD
    )
    assert CURRENT_CONSERVATIVE_EXPOSURE_USD + COMPLETION_CAP_USD == (
        MAXIMUM_WITH_COMPLETION_CAP_USD
    )
    assert WORKFLOW_CEILING_USD - MAXIMUM_WITH_COMPLETION_CAP_USD == (
        REMAINING_CEILING_AFTER_CAP_USD
    )
    assert MAXIMUM_WITH_COMPLETION_CAP_USD + GENERIC_JUDGE_CAP_USD == (
        MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD
    )
    assert MAXIMUM_WITH_COMPLETION_AND_JUDGE_CAP_USD < WORKFLOW_CEILING_USD
    print("KALAI_S3_COMPLETION_V1_PREP_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--gate-output")
    parser.add_argument("--gate-repo")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not all((args.output_root, args.repo_root, args.gate_output, args.gate_repo)):
        parser.error("all controller and gate paths are required")
    if args.audit_only:
        plan, stage = audit_cpu_stage(
            args.output_root, args.repo_root, args.gate_repo, args.gate_output
        )
        print(
            json.dumps(
                {
                    "status": "KALAI_S3_COMPLETION_V1_CPU_STAGE_VALID",
                    "completion_plan_payload_sha256": plan[SEAL_FIELD],
                    "cpu_stage_payload_sha256": stage[SEAL_FIELD],
                    "gpu_jobs_submitted": 0,
                    "external_api_calls": 0,
                },
                sort_keys=True,
            )
        )
    else:
        prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
