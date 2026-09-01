#!/usr/bin/env python3
"""Audit and seal one Kalai ``s=1`` completion batch.

The evaluator is derivation-only.  It validates the exact one-shot running
state, re-audits the immutable plan and generated shards, and writes one batch
result.  It never assembles the seven batches, authorizes the next batch, or
performs external judging.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import math
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


controller = _load_module(
    "_kalai_s1_completion_batch_controller_for_evaluator",
    SCRIPT_DIR / "sample_massive_medical_kalai_s1_completion_batch_v1.py",
)
authority = _load_module(
    "_kalai_s1_completion_batch_authorizer_for_evaluator",
    SCRIPT_DIR / "authorize_massive_medical_kalai_s1_completion_batch_v1.py",
)
planner = controller.batch_planner


PROTOCOL_ID = controller.PROTOCOL_ID
SOURCE_PROTOCOL_ID = controller.SOURCE_PROTOCOL_ID
METHOD_ID = controller.METHOD_ID
SEAL_FIELD = controller.SEAL_FIELD


def _load_json(path, description):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def _control_lines(path):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"control record is absent or unsafe: {path}")
    return Path(path).read_text(encoding="utf-8").splitlines()


def _require_line(path, key, value):
    if f"{key}={value}" not in _control_lines(path):
        raise ValueError(f"{Path(path).name} {key} differs")


def _audit_running_state(output_root, batch_index, slurm_job_id):
    control = authority.batch_control(output_root, batch_index)
    required = {
        "AUTHORIZATION.json",
        "SUBMISSION_LOCK",
        "SUBMISSION_ATTEMPT.tsv",
        "SUBMITTED",
        "RELEASE_AUTHORIZED",
        "RELEASED",
        "INVOCATION_LOCK",
    }
    if control.is_symlink() or not control.is_dir():
        raise ValueError("completion-batch control directory is absent or unsafe")
    actual = {path.name for path in control.iterdir()}
    if actual != required:
        raise ValueError("completion-batch running control inventory differs")
    for name in ("SUBMITTED", "RELEASE_AUTHORIZED", "RELEASED"):
        _require_line(control / name, "batch_id", authority.batch_id(batch_index))
        _require_line(control / name, "job_id", slurm_job_id)
    _require_line(control / "SUBMITTED", "held_first", "true")
    _require_line(control / "SUBMITTED", "held_audit_passed", "true")
    _require_line(control / "RELEASE_AUTHORIZED", "release_authorized", "true")
    _require_line(control / "RELEASED", "released", "true")
    _require_line(control / "INVOCATION_LOCK" / "owner", "job_id", slurm_job_id)
    _require_line(
        control / "INVOCATION_LOCK" / "owner",
        "automatic_next_batch_authorized",
        "false",
    )
    return control


def _artifact_binding(path, description):
    payload = _load_json(path, description)
    planner.verify_seal(payload, description)
    return payload, authority.binding(path, payload, description)


def _phase_artifacts(output_root, batch_index, audit):
    root = controller._batch_root(output_root, batch_index)
    result = {}
    for phase in controller.PHASES:
        generation_path = root / phase / "generation.json"
        timing_path = root / phase / "timing.json"
        generation, generation_binding = _artifact_binding(
            generation_path, f"{phase} completion-batch generation"
        )
        timing, timing_binding = _artifact_binding(
            timing_path, f"{phase} completion-batch timing"
        )
        timing_body = planner.verify_seal(
            timing, f"{phase} completion-batch timing"
        )
        expected_generation = audit["phases"][phase]
        elapsed = timing_body.get("elapsed_seconds")
        if (
            set(timing_body)
            != {
                "protocol_id",
                "source_protocol_id",
                "method_id",
                "batch_index",
                "batch_id",
                "phase",
                "batch_plan_payload_sha256",
                "elapsed_seconds",
                "summary",
                "external_api_calls",
            }
            or generation[SEAL_FIELD]
            != expected_generation["generation_payload_sha256"]
            or timing[SEAL_FIELD]
            != expected_generation["timing_payload_sha256"]
            or generation.get("summary") != expected_generation["summary"]
            or timing_body.get("protocol_id") != PROTOCOL_ID
            or timing_body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
            or timing_body.get("method_id") != METHOD_ID
            or timing_body.get("batch_index") != batch_index
            or timing_body.get("batch_id") != authority.batch_id(batch_index)
            or timing_body.get("phase") != phase
            or timing_body.get("batch_plan_payload_sha256")
            != generation.get("meta", {}).get("batch_plan_payload_sha256")
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
            or timing_body.get("summary") != generation.get("summary")
            or timing_body.get("external_api_calls") != 0
        ):
            raise ValueError(f"{phase} completion-batch artifact differs")
        result[phase] = {
            "generation": generation_binding,
            "timing": timing_binding,
            "summary": generation["summary"],
            "new_attempts_generated": generation["new_attempts_generated"],
        }
    return result


def evaluate(args):
    output_root = Path(args.output_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    batch_index = args.batch_index
    batch_name = authority.batch_id(batch_index)
    if not args.slurm_job_id.isdigit():
        raise ValueError("Slurm job ID differs")
    if not 1 <= args.elapsed_seconds <= authority.H200_MINUTES * 60:
        raise ValueError("completion-batch elapsed time exceeds authority")
    control = _audit_running_state(
        output_root, batch_index, args.slurm_job_id
    )
    authorization_path = Path(args.authorization).resolve()
    if authorization_path != control / "AUTHORIZATION.json":
        raise ValueError("completion-batch authorization path differs")
    authority.verify_authorization(
        argparse.Namespace(
            output_root=str(output_root),
            repo_root=str(repo_root),
            batch_index=batch_index,
            allow_generation=True,
            allow_running_state=True,
            slurm_job_id=args.slurm_job_id,
        )
    )
    authorization = _load_json(
        authorization_path, "completion-batch authorization"
    )

    plan_path = Path(args.batch_plan).resolve()
    expected_plan = output_root / "control" / "COMPLETION_BATCH_PLAN.json"
    if plan_path != expected_plan:
        raise ValueError("completion batch-plan path differs")
    plan_payload, plan_body = controller._load_batch_plan(plan_path)
    audit = controller._audit_batch(
        output_root, plan_payload, plan_body, batch_index
    )
    phase_artifacts = _phase_artifacts(output_root, batch_index, audit)
    combined_path = controller._batch_root(
        output_root, batch_index
    ) / "combined_timing.json"
    combined, combined_binding = _artifact_binding(
        combined_path, "completion-batch combined timing"
    )
    if combined[SEAL_FIELD] != audit["combined_timing_payload_sha256"]:
        raise ValueError("completion-batch combined timing binding differs")

    actual = (
        Decimal(args.elapsed_seconds)
        * authority.H200_HOURLY_USD
        / Decimal(3600)
    )
    if actual > authority.BATCH_CAP_USD:
        raise ValueError("completion-batch actual exceeds authority")
    result = planner.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "method_id": METHOD_ID,
            "stage": "completion_batch",
            "batch_index": batch_index,
            "batch_id": batch_name,
            "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_COMPLETE",
            "batch_valid": True,
            "restart_or_resume_authorized": False,
            "retry_replacement_or_requeue_authorized": False,
            "automatic_next_batch_authorized": False,
            "judge_authorized": False,
            "batch_plan": authority.binding(
                plan_path, plan_payload, "completion batch plan"
            ),
            "authorization": authority.binding(
                authorization_path,
                authorization,
                "completion-batch authorization",
            ),
            "combined_timing": combined_binding,
            "phase_outputs": phase_artifacts,
            "timing": {
                "slurm_job_id": args.slurm_job_id,
                "elapsed_seconds": args.elapsed_seconds,
                "h200_hourly_usd": float(authority.H200_HOURLY_USD),
                "authorized_cap_usd": float(authority.BATCH_CAP_USD),
                "actual_estimated_cost_usd": float(actual),
            },
            "accounting": {
                "starting_conservative_exposure_usd": float(
                    authority.STARTING_CONSERVATIVE_EXPOSURE_USD
                ),
                "prior_batch_authority_caps_retained_usd": float(
                    authority.BATCH_CAP_USD * Decimal(batch_index - 1)
                ),
                "this_batch_authority_cap_retained_usd": float(
                    authority.BATCH_CAP_USD
                ),
                "conservative_exposure_after_batch_authority_usd": float(
                    authority.maximum_exposure(batch_index)
                ),
                "program_ceiling_usd": float(authority.PROGRAM_CEILING_USD),
            },
            "external_api_calls": 0,
            "gpu_jobs_submitted_by_evaluator": 0,
        }
    )
    result_path = control / "RESULT.json"
    if os.path.lexists(result_path):
        raise ValueError("completion-batch result already exists")
    controller.source._write_new_json(
        result_path, result, "completion-batch result"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "batch_id": batch_name,
                "batch_result_payload_sha256": result[SEAL_FIELD],
                "benefit_requested_n": phase_artifacts["benefit"]["summary"][
                    "requested_n"
                ],
                "medical_requested_n": phase_artifacts["medical"]["summary"][
                    "requested_n"
                ],
                "actual_estimated_cost_usd": float(actual),
                "automatic_next_batch_authorized": False,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return result


def self_test():
    actual = Decimal(3600) * Decimal("0.90") / Decimal(3600)
    assert actual == authority.BATCH_CAP_USD
    assert authority.maximum_exposure(7) == Decimal("12.42198425")
    print("MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_EVALUATOR_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--batch-plan")
    parser.add_argument(
        "--batch-index", type=int, choices=authority.BATCH_INDICES
    )
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
            args.batch_plan,
            args.batch_index,
            args.authorization,
            args.elapsed_seconds,
            args.slurm_job_id,
        )
    ):
        parser.error("all completion-batch evaluator inputs are required")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
