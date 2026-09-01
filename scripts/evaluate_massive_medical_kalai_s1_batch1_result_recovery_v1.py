#!/usr/bin/env python3
"""Derive a batch-1 result from job 270983's sealed generation on CPU."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_module(
    "_kalai_s1_batch1_result_recovery_planner_for_evaluator",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)
stage_builder = _load_module(
    "_kalai_s1_batch1_result_recovery_stage_for_evaluator",
    SCRIPT_DIR / "stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)
controller = planner.controller


PROTOCOL_ID = planner.PROTOCOL_ID
SEAL_FIELD = planner.SEAL_FIELD


def _artifact_binding(path, description, *, sealed=False):
    path = Path(path).resolve()
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": planner.sha256_file(path),
    }
    if sealed:
        payload = planner.load_json(path, description)
        planner.verify_seal(payload, description)
        result["payload_sha256"] = payload[SEAL_FIELD]
    return result


def _require_cpu_only_environment():
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent from CPU recovery")


def _derive_result(plan_path, stage_path, repo_root):
    plan_payload, plan_body = planner.load_and_verify_plan(
        plan_path, audit_source=True
    )
    stage_payload, _ = stage_builder.load_and_verify_stage(
        stage_path, repo_root, audit_source=True
    )
    source_output = Path(plan_body["source_output_root"]).resolve()
    source_paths = planner._source_paths(source_output)
    if os.path.lexists(source_paths["batch_control"] / "RESULT.json"):
        raise ValueError("source RESULT.json must remain absent")
    source_plan, source_plan_body = controller._load_batch_plan(
        source_paths["plan"]
    )
    audit = controller._audit_batch(
        source_output,
        source_plan,
        source_plan_body,
        planner.SOURCE_BATCH_INDEX,
    )
    if audit != plan_body["source_generation_audit"]:
        raise ValueError("corrected controller audit differs from staged plan")

    actual = (
        Decimal(planner.SOURCE_SCHEDULER_ELAPSED_SECONDS)
        * Decimal("0.90")
        / Decimal(3600)
    )
    if actual != planner.SOURCE_ACTUAL_ESTIMATED_COST_USD:
        raise ValueError("source scheduler cost derivation differs")
    source_snapshot_before = planner.audit_source_state(
        source_output, plan_body["source_repository"]["path"]
    )["snapshot"]
    if source_snapshot_before != plan_body["source_snapshot"]:
        raise ValueError("source snapshot differs before result derivation")

    return planner.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": planner.SOURCE_PROTOCOL_ID,
            "method_id": planner.METHOD_ID,
            "stage": "batch1_result_recovery",
            "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH1_RESULT_RECOVERED_CPU_ONLY",
            "batch_index": planner.SOURCE_BATCH_INDEX,
            "batch_id": planner.SOURCE_BATCH_ID,
            "batch_valid": True,
            "source_job": {
                "slurm_job_id": planner.SOURCE_JOB_ID,
                "source_stopped_exit_code": int(planner.SOURCE_EXIT_CODE),
                "scheduler_state": "FAILED",
                "scheduler_exit_code": "1:0",
                "scheduler_derived_exit_code": "0:0",
                "scheduler_elapsed_seconds": (
                    planner.SOURCE_SCHEDULER_ELAPSED_SECONDS
                ),
                "authorized_cap_usd_retained": float(
                    planner.SOURCE_AUTHORIZED_CAP_USD
                ),
                "actual_estimated_cost_usd": float(actual),
            },
            "source_job_evidence": plan_body["source_job_evidence"],
            "recovery_reason": {
                "category": "post_generation_cpu_audit_implementation_error",
                "failed_expression": "math.isfinite",
                "correction": "controller imports Python math module",
                "generation_completed_before_error": True,
                "scientific_outputs_regenerated": False,
            },
            "recovery_plan": _artifact_binding(
                plan_path, "batch-1 recovery plan", sealed=True
            ),
            "cpu_stage": _artifact_binding(
                stage_path, "batch-1 recovery CPU stage", sealed=True
            ),
            "source_bindings": {
                "completion_batch_plan": _artifact_binding(
                    source_paths["plan"], "source completion plan", sealed=True
                ),
                "completion_cpu_stage": _artifact_binding(
                    source_paths["stage"], "source completion CPU stage", sealed=True
                ),
                "batch_01_authorization": _artifact_binding(
                    source_paths["authorization"],
                    "source batch-1 authorization",
                    sealed=True,
                ),
                "batch_01_stopped": _artifact_binding(
                    source_paths["stopped"], "source STOPPED"
                ),
                "generation_manifest": {
                    "file_count": plan_body["source_generation_manifest"][
                        "file_count"
                    ],
                    "size_bytes": plan_body["source_generation_manifest"][
                        "size_bytes"
                    ],
                    "manifest_sha256": plan_body[
                        "source_generation_manifest"
                    ]["manifest_sha256"],
                },
                "control_manifest": {
                    "file_count": plan_body["source_control_manifest"][
                        "file_count"
                    ],
                    "size_bytes": plan_body["source_control_manifest"][
                        "size_bytes"
                    ],
                    "manifest_sha256": plan_body["source_control_manifest"][
                        "manifest_sha256"
                    ],
                },
            },
            "corrected_controller_audit": audit,
            "source_snapshot_sha256": planner.sha256_bytes(
                planner.canonical_bytes(plan_body["source_snapshot"])
            ),
            "source_stopped_preserved": True,
            "source_result_absent": True,
            "derivation_only": True,
            "recovered_predecessor_evidence_valid": True,
            "automatic_next_batch_authorized": False,
            "batch_2_submission_authorized": False,
            "restart_or_resume_authorized": False,
            "retry_or_replacement_authorized": False,
            "generation_authorized": False,
            "accounting": {
                "new_gpu_cost_usd": 0.0,
                "new_api_cost_usd": 0.0,
                "source_conservative_exposure_usd": float(
                    planner.SOURCE_CONSERVATIVE_EXPOSURE_USD
                ),
                "program_ceiling_usd": float(planner.PROGRAM_CEILING_USD),
            },
            "external_api_calls": 0,
            "gpu_jobs": 0,
        }
    )


def recover(args):
    _require_cpu_only_environment()
    output = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    plan_path = output / "control" / planner.PLAN_NAME
    stage_path = output / "control" / planner.STAGE_NAME
    result_path = output / "control" / planner.RESULT_NAME
    planner._audit_recovery_namespace(output, {planner.PLAN_NAME, planner.STAGE_NAME})
    if os.path.lexists(result_path):
        raise ValueError("recovered result already exists; rerun is forbidden")
    result = _derive_result(plan_path, stage_path, repo)
    planner._write_idempotent(result_path, result, "recovered batch-1 result")
    _, plan_body = planner.load_and_verify_plan(plan_path, audit_source=True)
    after = planner.audit_source_state(
        plan_body["source_output_root"], plan_body["source_repository"]["path"]
    )["snapshot"]
    if after != plan_body["source_snapshot"]:
        raise ValueError("source snapshot changed during result recovery")
    planner._audit_recovery_namespace(
        output, {planner.PLAN_NAME, planner.STAGE_NAME, planner.RESULT_NAME}
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "recovered_result_payload_sha256": result[SEAL_FIELD],
                "source_job_id": planner.SOURCE_JOB_ID,
                "source_stopped_preserved": True,
                "automatic_next_batch_authorized": False,
                "batch_2_submission_authorized": False,
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return result


def audit(args):
    _require_cpu_only_environment()
    output = Path(args.output_root).resolve()
    result_path = output / "control" / planner.RESULT_NAME
    planner._audit_recovery_namespace(
        output, {planner.PLAN_NAME, planner.STAGE_NAME, planner.RESULT_NAME}
    )
    observed = planner.load_json(result_path, "recovered batch-1 result")
    planner.verify_seal(observed, "recovered batch-1 result")
    expected = _derive_result(
        output / "control" / planner.PLAN_NAME,
        output / "control" / planner.STAGE_NAME,
        args.repo_root,
    )
    if observed != expected:
        raise ValueError("recovered batch-1 result differs")
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH1_RESULT_RECOVERY_AUDITED",
                "recovered_result_payload_sha256": observed[SEAL_FIELD],
                "source_stopped_preserved": True,
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return observed


def self_test():
    assert planner.SOURCE_ACTUAL_ESTIMATED_COST_USD == Decimal("0.25450")
    assert "math" in controller.__dict__
    assert controller.math.isfinite(1.0)
    print("MASSIVE_MEDICAL_KALAI_S1_BATCH1_RESULT_RECOVERY_EVALUATOR_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--recover", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.output_root or not args.repo_root:
        parser.error("--output-root and --repo-root are required")
    if not args.recover and not args.audit_only:
        parser.error("exactly one of --recover or --audit-only is required")
    if args.recover:
        recover(args)
    else:
        audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
