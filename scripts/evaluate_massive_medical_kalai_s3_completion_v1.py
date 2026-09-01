#!/usr/bin/env python3
"""Seal the full one-shot Kalai completion and exact-union assembly."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load(
    "_kalai_s3_completion_prepare_for_evaluator",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s3_completion_v1.py",
)
authorizer = _load(
    "_kalai_s3_completion_authorizer_for_evaluator",
    SCRIPT_DIR / "authorize_massive_medical_kalai_s3_completion_v1.py",
)
sampler = prepare.sampler


def _control_line(path, key, expected):
    text = authorizer._read_text(path, os.path.basename(path))
    if f"{key}={expected}" not in text.splitlines():
        raise ValueError(f"{os.path.basename(path)} {key} differs")


def _bound_json(path, description):
    payload = prepare.load_json(path, description)
    prepare.verify_seal(payload, description)
    return payload, prepare.binding(path, payload)


def evaluate(args):
    gate_output = os.path.abspath(args.gate_output)
    controller_output = os.path.abspath(args.controller_output)
    controller_repo = os.path.abspath(args.controller_repo)
    gate_repo = os.path.abspath(args.gate_repo)
    control = os.path.join(gate_output, "control")
    if not args.slurm_job_id.isdigit():
        raise ValueError("Slurm job id differs")
    elapsed = args.elapsed_seconds
    if elapsed < 1 or elapsed > prepare.COMPLETION_H200_MINUTES * 60:
        raise ValueError("completion elapsed time exceeds authority")
    expected_state = {
        "COMPLETION_AUTHORIZATION.json",
        "COMPLETION_SUBMISSION_LOCK",
        "COMPLETION_SUBMISSION_ATTEMPT.tsv",
        "COMPLETION_SUBMITTED",
        "COMPLETION_RELEASE_AUTHORIZED",
        "COMPLETION_RELEASED",
        "COMPLETION_INVOCATION_LOCK",
    }
    actual_state = {
        name
        for name in authorizer.COMPLETION_CONTROL_NAMES
        if os.path.lexists(os.path.join(control, name))
    }
    if actual_state != expected_state:
        raise ValueError("completion result is not in the exact running state")
    for name in (
        "COMPLETION_SUBMITTED",
        "COMPLETION_RELEASE_AUTHORIZED",
        "COMPLETION_RELEASED",
    ):
        _control_line(
            os.path.join(control, name), "job_id", args.slurm_job_id
        )
    invocation_owner = os.path.join(
        control, "COMPLETION_INVOCATION_LOCK", "owner"
    )
    _control_line(invocation_owner, "job_id", args.slurm_job_id)

    auth_path = os.path.abspath(args.authorization)
    expected_auth_path = os.path.join(control, "COMPLETION_AUTHORIZATION.json")
    if auth_path != expected_auth_path:
        raise ValueError("completion authorization path differs")
    auth = prepare.load_json(auth_path, "completion authorization")
    auth_body = prepare.verify_seal(auth, "completion authorization")
    expected_auth = authorizer.expected_body(
        controller_output,
        controller_repo,
        gate_output,
        gate_repo,
        auth_body.get("created_at"),
    )
    if auth_body != expected_auth:
        raise ValueError("completion authorization differs")

    generation_root = os.path.join(gate_output, "generation")
    combined_path = os.path.join(
        generation_root, "completion", "combined_timing.json"
    )
    combined, combined_binding = _bound_json(
        combined_path, "combined completion timing"
    )
    combined_body = prepare.verify_seal(combined, "combined completion timing")
    if (
        combined_body.get("controller_protocol_id")
        != prepare.CONTROLLER_PROTOCOL_ID
        or combined_body.get("protocol_id") != sampler.PROTOCOL_ID
        or combined_body.get("method_id") != sampler.METHOD_ID
        or combined_body.get("stage") != "completion"
        or combined_body.get("shared_model_loads") != 1
        or combined_body.get("restart_or_resume_authorized") is not False
        or combined_body.get("external_api_calls") != 0
        or set(combined_body.get("phases", {})) != {"benefit", "medical"}
    ):
        raise ValueError("combined completion timing differs")
    phase_bindings = {}
    summaries = {}
    for phase, expected_n in (("benefit", 358), ("medical", 64)):
        phase_root = os.path.join(generation_root, "completion", phase)
        generation_path = os.path.join(phase_root, "generation.json")
        timing_path = os.path.join(phase_root, "timing.json")
        generation, generation_binding = _bound_json(
            generation_path, f"{phase} completion generation"
        )
        timing, timing_binding = _bound_json(
            timing_path, f"{phase} completion timing"
        )
        if (
            generation.get("summary") != timing.get("summary")
            or generation.get("summary", {}).get("requested_n") != expected_n
            or len(generation.get("samples", [])) != expected_n
            or timing.get("combined_controller_protocol_id")
            != prepare.CONTROLLER_PROTOCOL_ID
            or timing.get("shared_model_loads_across_phases") != 1
        ):
            raise ValueError(f"{phase} completion summary differs")
        bound_phase = combined_body["phases"][phase]
        if (
            bound_phase.get("generation") != generation_path
            or bound_phase.get("generation_payload_sha256")
            != generation[prepare.SEAL_FIELD]
            or bound_phase.get("timing") != timing_path
            or bound_phase.get("timing_payload_sha256")
            != timing[prepare.SEAL_FIELD]
            or bound_phase.get("summary") != generation["summary"]
        ):
            raise ValueError(f"combined {phase} binding differs")
        phase_bindings[phase] = {
            "generation": generation_binding,
            "timing": timing_binding,
        }
        summaries[phase] = generation["summary"]

    assembly_path = os.path.join(control, "ASSEMBLY.json")
    assembly, assembly_binding = _bound_json(assembly_path, "full assembly")
    assembly_body = prepare.verify_seal(assembly, "full assembly")
    if (
        assembly_body.get("protocol_id") != sampler.PROTOCOL_ID
        or assembly_body.get("method_id") != sampler.METHOD_ID
        or assembly_body.get("status") != "KALAI_S3_FULL_ASSEMBLY_AUDITED"
        or assembly_body.get("gate_rows_regenerated") is not False
        or assembly_body.get("external_api_calls") != 0
    ):
        raise ValueError("full assembly differs")

    actual = Decimal(elapsed) * Decimal("0.90") / Decimal(3600)
    if actual > prepare.COMPLETION_CAP_USD:
        raise ValueError("completion actual cost exceeds authority")
    result = prepare.seal(
        {
            "schema_version": 1,
            "controller_protocol_id": prepare.CONTROLLER_PROTOCOL_ID,
            "protocol_id": sampler.PROTOCOL_ID,
            "method_id": sampler.METHOD_ID,
            "status": "KALAI_S3_COMPLETION_COMPLETE",
            "completion_authorized": True,
            "restart_or_resume_authorized": False,
            "retry_authorized": False,
            "authorization": prepare.binding(auth_path, auth),
            "combined_timing": combined_binding,
            "completion_phases": phase_bindings,
            "completion_summaries": summaries,
            "assembly": assembly_binding,
            "timing": {
                "slurm_job_id": args.slurm_job_id,
                "elapsed_seconds": elapsed,
                "h200_hourly_usd": 0.90,
                "authorized_cap_usd": float(prepare.COMPLETION_CAP_USD),
                "actual_estimated_cost_usd": float(actual),
            },
            "accounting": {
                "prior_conservative_exposure_usd": float(
                    prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD
                ),
                "actual_adjusted_conservative_exposure_usd": float(
                    prepare.CURRENT_CONSERVATIVE_EXPOSURE_USD + actual
                ),
                "program_ceiling_usd": float(prepare.WORKFLOW_CEILING_USD),
            },
            "external_api_calls": 0,
            "gpu_jobs_submitted_by_evaluator": 0,
        }
    )
    result_path = os.path.join(control, "COMPLETION_RESULT.json")
    if os.path.lexists(result_path):
        raise ValueError("completion result already exists")
    prepare.write_new(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completion_result_payload_sha256": result[prepare.SEAL_FIELD],
                "benefit_requested_n": summaries["benefit"]["requested_n"],
                "benefit_accepted_n": summaries["benefit"]["accepted_n"],
                "medical_requested_n": summaries["medical"]["requested_n"],
                "medical_accepted_n": summaries["medical"]["accepted_n"],
                "actual_estimated_cost_usd": float(actual),
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    actual = Decimal(5400) * Decimal("0.90") / Decimal(3600)
    assert actual == Decimal("1.35")
    assert actual < prepare.COMPLETION_CAP_USD
    assert math.isclose(float(actual), 1.35, rel_tol=0, abs_tol=1e-15)
    print("KALAI_S3_COMPLETION_V1_EVALUATOR_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-output")
    parser.add_argument("--controller-repo")
    parser.add_argument("--gate-output")
    parser.add_argument("--gate-repo")
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--authorization")
    parser.add_argument("--elapsed-seconds", type=int)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    required = (
        args.controller_output,
        args.controller_repo,
        args.gate_output,
        args.gate_repo,
        args.source_protocol_manifest,
        args.authorization,
        args.elapsed_seconds,
        args.slurm_job_id,
    )
    if not all(value is not None for value in required):
        parser.error("all completion-evaluator inputs are required")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
