#!/usr/bin/env python3
"""Authorize exactly one fresh continuation-v2 batch in indices 3--7."""

from __future__ import annotations

import argparse
import datetime as dt
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


manager = _load(
    "_kalai_s1_rc_v2_manager_for_authorizer",
    "manage_massive_medical_kalai_s1_recovery_continuation_v2.py",
)
BATCH_INDICES = manager.BATCH_INDICES


def control_root(output_root, batch_index):
    return Path(output_root).resolve() / "control" / "batches" / manager.batch_id(batch_index)


def authorization_path(output_root, batch_index):
    return control_root(output_root, batch_index) / "AUTHORIZATION.json"


def _lines(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe")
    return set(path.read_text(encoding="utf-8").splitlines())


def _plan_summary(plan_body, batch_index):
    batches = plan_body.get("continuation_batches")
    if not isinstance(batches, list) or [item.get("batch_index") for item in batches] != list(BATCH_INDICES):
        raise ValueError("continuation-v2 batch inventory differs")
    return batches[batch_index - 3]["summary"]


def predecessor_binding(output_root, repo_root, plan_body, batch_index):
    if batch_index == 3:
        expected = plan_body.get("recovery_bindings", {}).get(manager.recovery_manager.RESULT_NAME)
        if not isinstance(expected, dict):
            raise ValueError("recovered batch-2 predecessor binding is absent")
        payload = manager.load_json(expected.get("path", ""), "recovered batch-2 predecessor")
        body = manager.verify_seal(payload, "recovered batch-2 predecessor")
        observed = manager.binding(expected["path"], payload)
        if (
            observed != expected
            or body.get("status") != manager.EXPECTED_RECOVERY_STATUS
            or body.get("batch_index") != 2
            or body.get("batch_valid") is not True
            or body.get("recovered_predecessor_evidence_valid") is not True
            or body.get("source_stopped_preserved") is not True
            or body.get("source_result_absent") is not True
            or body.get("automatic_next_batch_authorized") is not False
            or body.get("batch_3_submission_authorized") is not False
        ):
            raise ValueError("recovered batch-2 predecessor differs")
        return {"kind": "recovered_batch_2_result", "artifact": observed}
    evaluator = _load(
        f"_kalai_s1_rc_v2_evaluator_for_predecessor_{batch_index}",
        "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v2.py",
    )
    previous = evaluator.load_and_verify_result(output_root, repo_root, batch_index - 1, audit_generation=True)
    stopped = control_root(output_root, batch_index - 1) / "STOPPED"
    if os.path.lexists(stopped):
        raise ValueError("preceding continuation-v2 batch is stopped")
    path = control_root(output_root, batch_index - 1) / "RESULT.json"
    return {"kind": "preceding_recovery_continuation_v2_batch_result", "artifact": manager.binding(path, previous)}


def expected_body(output_root, repo_root, batch_index, created_at):
    output, plan, plan_body, stage, _, _ = manager._load_workflow(output_root, repo_root, audit_source=True)
    return {
        "schema_version": 1,
        "protocol_id": manager.PROTOCOL_ID,
        "source_protocol_id": manager.SOURCE_PROTOCOL_ID,
        "recovery_protocol_id": manager.RECOVERY_PROTOCOL_ID,
        "method_id": manager.METHOD_ID,
        "stage": "recovery_continuation_batch",
        "batch_index": batch_index,
        "batch_id": manager.batch_id(batch_index),
        "created_at": created_at,
        "repository_commit": manager.git_commit(repo_root),
        "continuation_plan": manager.binding(output / "control" / manager.PLAN_NAME, plan),
        "cpu_stage": manager.binding(output / "control" / manager.STAGE_NAME, stage),
        "predecessor": predecessor_binding(output, repo_root, plan_body, batch_index),
        "batch_summary": _plan_summary(plan_body, batch_index),
        "authorized_gpu_jobs": 1,
        "h200_count": manager.H200_COUNT,
        "h200_minutes_cap": manager.H200_MINUTES,
        "h200_hourly_usd": float(manager.H200_HOURLY_USD),
        "maximum_cost_usd": float(manager.BATCH_CAP_USD),
        "known_program_actual_at_continuation_start_usd": float(manager.KNOWN_PROGRAM_ACTUAL_USD),
        "current_conservative_exposure_usd": float(manager.current_exposure(batch_index)),
        "conservative_actual_plus_new_cap_usd": float(manager.maximum_exposure(batch_index)),
        "program_ceiling_usd": float(manager.PROGRAM_CEILING_USD),
        "external_api_calls_authorized": 0,
        "judge_authorized": False,
        "another_batch_authorized": False,
        "automatic_next_batch_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_replacement_or_requeue_authorized": False,
    }


def audit_acknowledgments(args):
    manager.batch_id(args.batch_index)
    if (
        args.ack_h200_minutes != manager.H200_MINUTES
        or Decimal(args.ack_max_cost_usd) != manager.BATCH_CAP_USD
        or Decimal(args.ack_current_conservative_exposure_usd) != manager.current_exposure(args.batch_index)
        or Decimal(args.ack_conservative_program_max_usd) != manager.maximum_exposure(args.batch_index)
        or Decimal(args.ack_program_ceiling_usd) != manager.PROGRAM_CEILING_USD
    ):
        raise ValueError("numeric authority acknowledgment differs")
    for name in ("ack_no_api", "ack_no_automatic_next_batch", "ack_no_restart_resume_retry_replacement", "ack_prior_caps_retained", "ack_recovered_batch2_predecessor", "ack_post_hoc_sensitivity"):
        if getattr(args, name) is not True:
            raise ValueError(f"required acknowledgment is absent: {name}")


def _expected_lock(repo_root, batch_index):
    return {
        f"protocol_id={manager.PROTOCOL_ID}",
        "stage=recovery_continuation_batch",
        f"batch_id={manager.batch_id(batch_index)}",
        f"repository_commit={manager.git_commit(repo_root)}",
        "restart_or_resume_authorized=false",
        "retry_or_replacement_authorized=false",
        "automatic_next_batch_authorized=false",
    }


def audit_control_state(output_root, repo_root, batch_index, *, running_job_id=None):
    control = control_root(output_root, batch_index)
    if control.is_symlink() or not control.is_dir():
        raise ValueError("target batch control is absent or unsafe")
    expected = {"SUBMISSION_LOCK", "AUTHORIZATION.json"}
    if running_job_id is not None:
        if not isinstance(running_job_id, str) or not running_job_id.isdigit():
            raise ValueError("running Slurm job ID differs")
        expected |= {"SUBMISSION_ATTEMPT.tsv", "SUBMITTED", "RELEASE_AUTHORIZED", "RELEASED", "INVOCATION_LOCK"}
    if {item.name for item in control.iterdir()} != expected:
        raise ValueError("target batch control inventory differs")
    if _lines(control / "SUBMISSION_LOCK" / "owner", "submission lock") != _expected_lock(repo_root, batch_index):
        raise ValueError("submission-lock owner differs")
    if running_job_id is None:
        return
    bid = manager.batch_id(batch_index)
    if _lines(control / "SUBMISSION_ATTEMPT.tsv", "submission attempt") != {"stage\tbatch_id\tjob_id\th200_minutes\tmaximum_cost_usd", f"recovery_continuation_batch\t{bid}\t{running_job_id}\t60\t0.900"}:
        raise ValueError("submission attempt differs")
    common = {f"protocol_id={manager.PROTOCOL_ID}", "stage=recovery_continuation_batch", f"batch_id={bid}", f"job_id={running_job_id}", "restart_or_resume_authorized=false", "automatic_next_batch_authorized=false"}
    if _lines(control / "SUBMITTED", "submitted record") != common | {"held_first=true", "held_audit_passed=true", f"repository_commit={manager.git_commit(repo_root)}"}:
        raise ValueError("submitted record differs")
    if _lines(control / "RELEASE_AUTHORIZED", "release authority") != common | {"held_audit_passed=true", "release_authorized=true"}:
        raise ValueError("release authority differs")
    if _lines(control / "RELEASED", "release record") != common | {"released=true"}:
        raise ValueError("release record differs")
    invocation = {"stage=recovery_continuation_batch", f"batch_id={bid}", f"job_id={running_job_id}", "restart_or_resume_authorized=false", "retry_or_replacement_authorized=false", "automatic_next_batch_authorized=false", "external_api_calls_authorized=0"}
    if _lines(control / "INVOCATION_LOCK" / "owner", "invocation lock") != invocation:
        raise ValueError("invocation lock differs")


def _require_target_fresh(output_root, batch_index):
    output = Path(output_root).resolve()
    control = control_root(output, batch_index)
    generation = output / "generation" / "completion_batches" / manager.batch_id(batch_index)
    if os.path.lexists(generation):
        raise ValueError("target batch generation already exists")
    for name in ("RESULT.json", "STOPPED", "INVOCATION_LOCK"):
        if os.path.lexists(control / name):
            raise ValueError(f"target batch already has {name}")


def preflight_predecessor(args):
    manager._require_no_api_key()
    output, plan, plan_body, _, _, _ = manager._load_workflow(args.output_root, args.repo_root, audit_source=True)
    _require_target_fresh(output, args.batch_index)
    if os.path.lexists(control_root(output, args.batch_index)):
        raise ValueError("target batch control already exists")
    predecessor = predecessor_binding(output, args.repo_root, plan_body, args.batch_index)
    summary = _plan_summary(plan_body, args.batch_index)
    print(json.dumps({"status": "KALAI_S1_RECOVERY_CONTINUATION_V2_PREDECESSOR_PREFLIGHT_VALID", "batch_id": manager.batch_id(args.batch_index), "continuation_plan_payload_sha256": plan[manager.SEAL_FIELD], "predecessor_kind": predecessor["kind"], "row_count": summary["row_count"], "gpu_jobs": 0, "external_api_calls": 0}, sort_keys=True))
    return predecessor


def write_authorization(args):
    manager._require_no_api_key()
    audit_acknowledgments(args)
    _require_target_fresh(args.output_root, args.batch_index)
    path = authorization_path(args.output_root, args.batch_index)
    if os.path.lexists(path):
        raise ValueError("target batch authorization is nonreusable")
    control = control_root(args.output_root, args.batch_index)
    if control.is_symlink() or not control.is_dir() or {item.name for item in control.iterdir()} != {"SUBMISSION_LOCK"} or _lines(control / "SUBMISSION_LOCK" / "owner", "submission lock") != _expected_lock(args.repo_root, args.batch_index):
        raise ValueError("pre-authorization control state differs")
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = manager.seal(expected_body(args.output_root, args.repo_root, args.batch_index, created_at))
    manager._write_new(path, payload, "batch authorization")
    print(json.dumps({"status": "KALAI_S1_RECOVERY_CONTINUATION_V2_BATCH_AUTHORIZED", "batch_id": manager.batch_id(args.batch_index), "authorization_payload_sha256": payload[manager.SEAL_FIELD], "authorized_gpu_jobs": 1, "maximum_cost_usd": float(manager.BATCH_CAP_USD), "external_api_calls_authorized": 0, "automatic_next_batch_authorized": False}, sort_keys=True))
    return payload


def verify_authorization(args):
    manager._require_no_api_key()
    payload = manager.load_json(authorization_path(args.output_root, args.batch_index), "batch authorization")
    body = manager.verify_seal(payload, "batch authorization")
    if body != expected_body(args.output_root, args.repo_root, args.batch_index, body.get("created_at")):
        raise ValueError("batch authorization differs")
    audit_control_state(args.output_root, args.repo_root, args.batch_index, running_job_id=args.running_job_id)
    print(json.dumps({"status": "KALAI_S1_RECOVERY_CONTINUATION_V2_BATCH_AUTHORIZATION_VALID", "batch_id": manager.batch_id(args.batch_index), "authorization_payload_sha256": payload[manager.SEAL_FIELD], "running_job_id": args.running_job_id, "external_api_calls": 0}, sort_keys=True))
    return payload


def self_test():
    assert BATCH_INDICES == tuple(range(3, 8))
    assert manager.maximum_exposure(3) == Decimal("8.82198425")
    print("MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V2_AUTHORIZER_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    for command in ("preflight", "write", "verify"):
        child = sub.add_parser(command)
        child.add_argument("--output-root", required=True)
        child.add_argument("--repo-root", required=True)
        child.add_argument("--batch-index", required=True, type=int, choices=BATCH_INDICES)
        if command == "write":
            child.add_argument("--ack-h200-minutes", required=True, type=int)
            child.add_argument("--ack-max-cost-usd", required=True)
            child.add_argument("--ack-current-conservative-exposure-usd", required=True)
            child.add_argument("--ack-conservative-program-max-usd", required=True)
            child.add_argument("--ack-program-ceiling-usd", required=True)
            child.add_argument("--ack-no-api", action="store_true")
            child.add_argument("--ack-no-automatic-next-batch", action="store_true")
            child.add_argument("--ack-no-restart-resume-retry-replacement", action="store_true")
            child.add_argument("--ack-prior-caps-retained", action="store_true")
            child.add_argument("--ack-recovered-batch2-predecessor", action="store_true")
            child.add_argument("--ack-post-hoc-sensitivity", action="store_true")
        elif command == "verify":
            child.add_argument("--running-job-id")
    args = parser.parse_args(argv)
    if args.command == "self-test": self_test()
    elif args.command == "preflight": preflight_predecessor(args)
    elif args.command == "write": write_authorization(args)
    else: verify_authorization(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
