#!/usr/bin/env python3
"""Audit and seal one continuation-v3 batch result."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = _load("_kalai_s1_rc_v3_manager_for_evaluator", "manage_massive_medical_kalai_s1_recovery_continuation_v3.py")
authorizer = _load("_kalai_s1_rc_v3_authorizer_for_evaluator", "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v3.py")
runtime = _load("_kalai_s1_rc_v3_runtime_for_evaluator", "sample_massive_medical_kalai_s1_recovery_continuation_batch_v3.py")


def _load_authorization(output_root, repo_root, batch_index):
    path = authorizer.authorization_path(output_root, batch_index)
    payload = manager.load_json(path, "continuation-v3 authorization")
    body = manager.verify_seal(payload, "continuation-v3 authorization")
    if body != authorizer.expected_body(output_root, repo_root, batch_index, body.get("created_at")):
        raise ValueError("continuation-v3 authorization differs")
    return path, payload, body


def _generation_audit(output_root, repo_root, batch_index):
    """Use runtime's manager instance, preventing the v1 false mismatch bug."""
    args = argparse.Namespace(output_root=output_root, repo_root=repo_root)
    output, _, _, context = runtime._workflow(args)
    return runtime.generation_audit(output, context, batch_index)


def _phase_outputs(audit):
    result = {}
    for phase in manager.PHASES:
        generation_path = Path(audit["phases"][phase]["generation"])
        generation = manager.load_json(generation_path, f"continuation-v3 {phase} generation")
        manager.verify_seal(generation, f"continuation-v3 {phase} generation")
        timing_path = Path(audit["phases"][phase]["timing"])
        timing = manager.load_json(timing_path, f"continuation-v3 {phase} timing")
        manager.verify_seal(timing, f"continuation-v3 {phase} timing")
        result[phase] = {"generation": manager.binding(generation_path, generation), "timing": manager.binding(timing_path, timing), "summary": generation["summary"], "new_attempts_generated": generation["new_attempts_generated"]}
    return result


def _expected_result_body(output_root, repo_root, batch_index, *, slurm_job_id, elapsed_seconds, audit):
    if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit() or isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, int) or not 1 <= elapsed_seconds <= manager.H200_MINUTES * 60:
        raise ValueError("batch Slurm timing differs")
    output, plan, _, stage, _, _ = manager._load_workflow(output_root, repo_root, audit_source=True)
    auth_path, auth, auth_body = _load_authorization(output, repo_root, batch_index)
    combined_path = Path(audit["combined_timing"])
    combined = manager.load_json(combined_path, "continuation-v3 combined timing")
    manager.verify_seal(combined, "continuation-v3 combined timing")
    actual = Decimal(elapsed_seconds) * manager.H200_HOURLY_USD / Decimal(3600)
    return {
        "schema_version": 1,
        "protocol_id": manager.PROTOCOL_ID,
        "source_protocol_id": manager.SOURCE_PROTOCOL_ID,
        "recovery_protocol_id": manager.RECOVERY_PROTOCOL_ID,
        "method_id": manager.METHOD_ID,
        "stage": "recovery_continuation_batch",
        "batch_index": batch_index,
        "batch_id": manager.batch_id(batch_index),
        "status": "MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V3_BATCH_COMPLETE",
        "batch_valid": True,
        "generation_protocol_id": manager.PROTOCOL_ID,
        "scientific_batch_assignment_protocol_id": manager.SCIENTIFIC_BATCH_PROTOCOL_ID,
        "predecessor": auth_body["predecessor"],
        "source_batch_1_stopped_preserved": True,
        "source_batch_2_stopped_preserved": True,
        "source_batches_1_and_2_regenerated": False,
        "restart_or_resume_authorized": False,
        "retry_replacement_or_requeue_authorized": False,
        "automatic_next_batch_authorized": False,
        "judge_authorized": False,
        "continuation_plan": manager.binding(output / "control" / manager.PLAN_NAME, plan),
        "cpu_stage": manager.binding(output / "control" / manager.STAGE_NAME, stage),
        "authorization": manager.binding(auth_path, auth),
        "combined_timing": manager.binding(combined_path, combined),
        "phase_outputs": _phase_outputs(audit),
        "timing": {"slurm_job_id": slurm_job_id, "elapsed_seconds": elapsed_seconds, "h200_hourly_usd": float(manager.H200_HOURLY_USD), "authorized_cap_usd": float(manager.BATCH_CAP_USD), "actual_estimated_cost_usd": float(actual)},
        "accounting": {"known_program_actual_at_continuation_start_usd": float(manager.KNOWN_PROGRAM_ACTUAL_USD), "current_conservative_exposure_before_this_batch_usd": float(manager.current_exposure(batch_index)), "this_batch_authority_cap_retained_usd": float(manager.BATCH_CAP_USD), "conservative_exposure_after_batch_authority_usd": float(manager.maximum_exposure(batch_index)), "program_ceiling_usd": float(manager.PROGRAM_CEILING_USD)},
        "gpu_jobs_submitted_by_evaluator": 0,
        "external_api_calls": 0,
    }


def evaluate(args):
    manager._require_no_api_key()
    authorizer.verify_authorization(argparse.Namespace(output_root=args.output_root, repo_root=args.repo_root, batch_index=args.batch_index, running_job_id=args.slurm_job_id))
    audit = _generation_audit(args.output_root, args.repo_root, args.batch_index)
    body = _expected_result_body(args.output_root, args.repo_root, args.batch_index, slurm_job_id=args.slurm_job_id, elapsed_seconds=args.elapsed_seconds, audit=audit)
    result = manager.seal(body)
    path = authorizer.control_root(args.output_root, args.batch_index) / "RESULT.json"
    manager._write_new(path, result, "continuation-v3 batch result")
    print(json.dumps({"status": result["status"], "batch_id": manager.batch_id(args.batch_index), "result_payload_sha256": result[manager.SEAL_FIELD], "actual_estimated_cost_usd": body["timing"]["actual_estimated_cost_usd"], "automatic_next_batch_authorized": False, "external_api_calls": 0}, sort_keys=True))
    return result


def load_and_verify_result(output_root, repo_root, batch_index, *, audit_generation):
    if os.path.lexists(authorizer.control_root(output_root, batch_index) / "STOPPED"):
        raise ValueError("continuation-v3 batch is stopped")
    path = authorizer.control_root(output_root, batch_index) / "RESULT.json"
    result = manager.load_json(path, "continuation-v3 batch result")
    body = manager.verify_seal(result, "continuation-v3 batch result")
    if body.get("protocol_id") != manager.PROTOCOL_ID or body.get("status") != "MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V3_BATCH_COMPLETE" or body.get("batch_index") != batch_index or body.get("batch_id") != manager.batch_id(batch_index) or body.get("batch_valid") is not True or body.get("generation_protocol_id") != manager.PROTOCOL_ID or body.get("source_batch_1_stopped_preserved") is not True or body.get("source_batch_2_stopped_preserved") is not True or body.get("source_batches_1_and_2_regenerated") is not False or body.get("automatic_next_batch_authorized") is not False or body.get("judge_authorized") is not False or body.get("external_api_calls") != 0:
        raise ValueError("continuation-v3 result identity differs")
    if audit_generation:
        audit = _generation_audit(output_root, repo_root, batch_index)
        expected = _expected_result_body(output_root, repo_root, batch_index, slurm_job_id=body.get("timing", {}).get("slurm_job_id"), elapsed_seconds=body.get("timing", {}).get("elapsed_seconds"), audit=audit)
        if body != expected:
            raise ValueError("continuation-v3 result artifacts differ")
    return result


def self_test():
    assert Decimal(3600) * manager.H200_HOURLY_USD / Decimal(3600) == manager.BATCH_CAP_USD
    assert manager.PROTOCOL_ID.endswith("recovery_continuation_v3")
    source = Path(__file__).read_text(encoding="utf-8")
    assert "return runtime.generation_audit(output, context, batch_index)" in source
    print("MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V3_EVALUATOR_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--batch-index", type=int, choices=manager.BATCH_INDICES)
    parser.add_argument("--elapsed-seconds", type=int)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test: self_test(); return 0
    if any(value is None for value in (args.output_root, args.repo_root, args.batch_index, args.elapsed_seconds, args.slurm_job_id)):
        parser.error("all evaluator inputs are required")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
