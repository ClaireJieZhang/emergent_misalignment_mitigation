#!/usr/bin/env python3
"""Audit and seal the one-shot Kalai s=1 technical gate.

The gate has no coverage threshold.  It is successful when every reused prefix,
new suffix, final sample, source binding, and cost record is valid.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
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


planner = _load(
    "_kalai_s1_planner_for_gate_evaluator",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_trace_reuse_v1.py",
)
controller = _load(
    "_kalai_s1_controller_for_gate_evaluator",
    SCRIPT_DIR / "sample_massive_medical_kalai_s1_trace_reuse_v1.py",
)
authority = _load(
    "_kalai_s1_authority_for_gate_evaluator",
    SCRIPT_DIR / "authorize_massive_medical_kalai_s1_trace_reuse_v1.py",
)


PROTOCOL_ID = planner.PROTOCOL_ID
METHOD_ID = planner.METHOD_ID
SEAL_FIELD = planner.OUTPUT_SEAL
STAGE = "technical_gate"


def _load_json(path, description):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _binding(path, payload):
    raw = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "size_bytes": len(raw),
        "file_sha256": planner.sha256_bytes(raw),
        "payload_sha256": payload[SEAL_FIELD],
    }


def _control_lines(path):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"control record is absent or unsafe: {path}")
    return Path(path).read_text(encoding="utf-8").splitlines()


def _require_line(path, key, value):
    if f"{key}={value}" not in _control_lines(path):
        raise ValueError(f"{Path(path).name} {key} differs")


def _audit_running_state(output_root, slurm_job_id):
    control = Path(output_root).resolve() / "control"
    required = (
        "TECHNICAL_GATE_SUBMISSION_LOCK",
        "TECHNICAL_GATE_SUBMISSION_ATTEMPT.tsv",
        "TECHNICAL_GATE_SUBMITTED",
        "TECHNICAL_GATE_RELEASE_AUTHORIZED",
        "TECHNICAL_GATE_RELEASED",
        "TECHNICAL_GATE_AUTHORIZATION.json",
        "TECHNICAL_GATE_INVOCATION_LOCK",
    )
    for name in required:
        if not os.path.lexists(control / name):
            raise ValueError(f"technical-gate running state lacks {name}")
    for name in (
        "TECHNICAL_GATE_SUBMITTED",
        "TECHNICAL_GATE_RELEASE_AUTHORIZED",
        "TECHNICAL_GATE_RELEASED",
    ):
        _require_line(control / name, "job_id", slurm_job_id)
    _require_line(
        control / "TECHNICAL_GATE_INVOCATION_LOCK" / "owner",
        "job_id",
        slurm_job_id,
    )
    for forbidden in (
        "TECHNICAL_GATE_RESULT.json",
        "TECHNICAL_GATE_STOPPED",
        "COMPLETION_AUTHORIZATION.json",
        "COMPLETION_SUBMISSION_LOCK",
    ):
        if os.path.lexists(control / forbidden):
            raise ValueError(f"technical-gate running state contains {forbidden}")
    return control


def _profiles(plan_body):
    manifest_path = controller._binding_path(
        plan_body, "source_protocol_manifest"
    )
    protocol = controller.legacy.primary.load_protocol_manifest(
        manifest_path, audit_models=False
    )
    return controller._profiles_and_records(protocol)


def _load_continuations(output_root, phase, unresolved_rows, profile):
    path = (
        Path(output_root).resolve()
        / "generation"
        / STAGE
        / phase
        / "generation.json"
    )
    payload = _load_json(path, f"{phase} technical-gate continuation")
    planner.verify_seal(payload, f"{phase} technical-gate continuation")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != len(unresolved_rows):
        raise ValueError(f"{phase} technical-gate continuation count differs")
    expected = [
        (row["question_id"], row["sample_index"]) for row in unresolved_rows
    ]
    observed = [
        (sample.get("question_id"), sample.get("sample_index"))
        for sample in samples
    ]
    if observed != expected:
        raise ValueError(f"{phase} technical-gate continuation order differs")
    result = {}
    for row, sample in zip(unresolved_rows, samples):
        final = planner.assemble_s1_sample(row, sample)
        controller.legacy._audit_sample(
            final, controller._request_fields(row), phase, profile
        )
        key = row["question_id"], row["sample_index"]
        if key in result:
            raise ValueError(f"duplicate {phase} continuation")
        result[key] = final
    return payload, result


def _assembled_payload(plan_payload, phase, rows, continuations, profile):
    samples = []
    for row in rows:
        key = row["question_id"], row["sample_index"]
        continuation = continuations.get(key)
        final = planner.assemble_s1_sample(row, continuation)
        controller.legacy._audit_sample(
            final, controller._request_fields(row), phase, profile
        )
        samples.append(final)
    expected_continuation_keys = {
        (row["question_id"], row["sample_index"])
        for row in rows
        if row["disposition"] == "unresolved"
    }
    if set(continuations) != expected_continuation_keys:
        raise ValueError(f"{phase} continuation set differs")
    meta = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": STAGE,
        "phase": phase,
        "replay_plan_payload_sha256": plan_payload[SEAL_FIELD],
        "requested_n": len(rows),
        "request_keys": [
            [row["question_id"], row["sample_index"]] for row in rows
        ],
        "reused_terminal_rows": sum(
            row["disposition"] != "unresolved" for row in rows
        ),
        "continued_rows": sum(
            row["disposition"] == "unresolved" for row in rows
        ),
        "stored_prefix_candidates_regenerated": False,
        "external_api_calls": 0,
    }
    return planner.seal(
        {
            "meta": meta,
            "summary": controller.legacy.summarize_samples(samples),
            "samples": samples,
        }
    )


def evaluate(args):
    output_root = Path(args.output_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    if not args.slurm_job_id.isdigit():
        raise ValueError("Slurm job ID differs")
    if not 1 <= args.elapsed_seconds <= authority.GATE_H200_MINUTES * 60:
        raise ValueError("technical-gate elapsed time exceeds authority")
    control = _audit_running_state(output_root, args.slurm_job_id)
    authorization_path = Path(args.authorization).resolve()
    if authorization_path != control / "TECHNICAL_GATE_AUTHORIZATION.json":
        raise ValueError("technical-gate authority path differs")
    authority.verify_authorization(
        argparse.Namespace(output_root=str(output_root), repo_root=str(repo_root))
    )
    authorization = _load_json(
        authorization_path, "technical-gate authorization"
    )

    plan_path = Path(args.replay_plan).resolve()
    if plan_path != control / "REPLAY_PLAN.json":
        raise ValueError("replay-plan path differs")
    plan_payload, plan_body = controller._load_plan(plan_path)
    controller._audit_stage(output_root, plan_payload, plan_body, STAGE)
    rows_by_phase = controller._plan_rows(plan_body)
    phase_data = _profiles(plan_body)
    assembled_bindings = {}
    observations = {}
    for phase, expected_requested, expected_unresolved in (
        ("benefit", 2, 1),
        ("medical", 16, 15),
    ):
        rows = [
            row
            for row in rows_by_phase[phase]
            if row["partition"] == STAGE
        ]
        unresolved = [row for row in rows if row["disposition"] == "unresolved"]
        if len(rows) != expected_requested or len(unresolved) != expected_unresolved:
            raise ValueError(f"{phase} technical-gate plan count differs")
        continuation_payload, continuations = _load_continuations(
            output_root, phase, unresolved, phase_data[phase]["profile"]
        )
        assembled = _assembled_payload(
            plan_payload,
            phase,
            rows,
            continuations,
            phase_data[phase]["profile"],
        )
        assembled_path = (
            output_root / "assembled" / STAGE / phase / "generation.json"
        )
        controller._write_new_json(
            assembled_path, assembled, f"assembled {phase} technical gate"
        )
        continuation_path = (
            output_root / "generation" / STAGE / phase / "generation.json"
        )
        assembled_bindings[phase] = {
            "continuation_generation": _binding(
                continuation_path, continuation_payload
            ),
            "assembled_generation": _binding(assembled_path, assembled),
        }
        summary = assembled["summary"]
        observations[phase] = {
            "requested_n": summary["requested_n"],
            "accepted_n": summary["accepted_n"],
            "abstained_n": summary["abstained_n"],
            "coverage": summary["coverage"],
            "judge_eligible_medical_n": (
                summary["judge_eligible_medical_n"]
                if phase == "medical"
                else None
            ),
            "total_attempts_including_reused_prefix": summary["total_attempts"],
            "new_attempts_generated": continuation_payload[
                "new_attempts_generated"
            ],
        }

    combined_path = output_root / "generation" / STAGE / "combined_timing.json"
    combined = _load_json(combined_path, "technical-gate combined timing")
    planner.verify_seal(combined, "technical-gate combined timing")
    actual = (
        Decimal(args.elapsed_seconds)
        * authority.H200_HOURLY_USD
        / Decimal(3600)
    )
    if actual > authority.GATE_CAP_USD:
        raise ValueError("technical-gate actual exceeds authority")
    result = planner.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "stage": STAGE,
            "status": "MASSIVE_MEDICAL_KALAI_S1_TECHNICAL_GATE_COMPLETE",
            "technical_gate_valid": True,
            "coverage_threshold": None,
            "completion_eligible": True,
            "completion_authorized": False,
            "restart_or_resume_authorized": False,
            "retry_authorized": False,
            "replay_plan": _binding(plan_path, plan_payload),
            "authorization": _binding(authorization_path, authorization),
            "combined_timing": _binding(combined_path, combined),
            "phase_outputs": assembled_bindings,
            "observed": observations,
            "timing": {
                "slurm_job_id": args.slurm_job_id,
                "elapsed_seconds": args.elapsed_seconds,
                "h200_hourly_usd": float(authority.H200_HOURLY_USD),
                "authorized_cap_usd": float(authority.GATE_CAP_USD),
                "actual_estimated_cost_usd": float(actual),
            },
            "accounting": {
                "known_program_actual_before_gate_usd": float(
                    authority.KNOWN_PROGRAM_ACTUAL_USD
                ),
                "retained_conservative_exposure_usd": float(
                    authority.RETAINED_CONSERVATIVE_EXPOSURE_USD
                ),
                "known_program_actual_after_gate_usd": float(
                    authority.KNOWN_PROGRAM_ACTUAL_USD + actual
                ),
                "actual_adjusted_conservative_exposure_usd": float(
                    authority.CURRENT_CONSERVATIVE_EXPOSURE_USD + actual
                ),
                "program_ceiling_usd": float(authority.WORKFLOW_CEILING_USD),
            },
            "external_api_calls": 0,
            "gpu_jobs_submitted_by_evaluator": 0,
        }
    )
    result_path = control / "TECHNICAL_GATE_RESULT.json"
    if os.path.lexists(result_path):
        raise ValueError("technical-gate result already exists")
    controller._write_new_json(
        result_path, result, "technical-gate result"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "technical_gate_result_payload_sha256": result[SEAL_FIELD],
                "benefit_accepted_n": observations["benefit"]["accepted_n"],
                "medical_accepted_n": observations["medical"]["accepted_n"],
                "medical_abstained_n": observations["medical"]["abstained_n"],
                "actual_estimated_cost_usd": float(actual),
                "completion_authorized": False,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return result


def self_test():
    actual = Decimal(2100) * Decimal("0.90") / Decimal(3600)
    assert actual == Decimal("0.525")
    assert actual == authority.GATE_CAP_USD
    print("MASSIVE_MEDICAL_KALAI_S1_TRACE_REUSE_GATE_EVALUATOR_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--replay-plan")
    parser.add_argument("--authorization")
    parser.add_argument("--elapsed-seconds", type=int)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if any(
        value is None
        for value in (
            args.output_root,
            args.repo_root,
            args.replay_plan,
            args.authorization,
            args.elapsed_seconds,
            args.slurm_job_id,
        )
    ):
        parser.error("all evaluator inputs are required")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
