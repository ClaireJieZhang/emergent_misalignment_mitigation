#!/usr/bin/env python3
"""Audit and seal the CPU-only decision for the Kalai s=3 coverage gate."""

from __future__ import annotations

import argparse
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


sampler = _load_module(
    "_massive_medical_kalai_s3_sampler_for_gate",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_s3_v2.py",
)
prepare = _load_module(
    "_massive_medical_kalai_s3_prepare_for_gate",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s3_v2.py",
)


def load_sealed(path, description):
    payload = prepare.load_json(path, description)
    prepare.verify_seal(payload, description)
    return payload


def audit_generation(path, source_binding, phase, profile, requests):
    payload = load_sealed(path, f"{phase} gate generation")
    if set(payload) != {"meta", "summary", "samples", prepare.SEAL_FIELD}:
        raise ValueError(f"{phase} gate generation schema differs")
    expected_meta_body = sampler._stream_meta(
        source_binding, phase, "gate", profile, requests
    )
    expected_meta = {
        **expected_meta_body,
        "stream_fingerprint": sampler._sha256(sampler._canonical(expected_meta_body)),
    }
    if payload["meta"] != expected_meta:
        raise ValueError(f"{phase} gate generation metadata differs")
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) != len(requests):
        raise ValueError(f"{phase} gate sample count differs")
    for sample, request in zip(samples, requests):
        sampler._audit_sample(sample, request, phase, profile)
    expected_summary = sampler.summarize_samples(samples)
    if payload["summary"] != expected_summary:
        raise ValueError(f"{phase} gate summary differs")
    return payload, samples, expected_summary


def gate_observation(benefit_samples, medical_samples):
    benefit_valid = sum(
        sample.get("accepted") is True
        and sample.get("finish_reason") == "stop"
        and bool(sample.get("response"))
        and isinstance(sample.get("prediction"), dict)
        for sample in benefit_samples
    )
    medical_accepted = [
        sample for sample in medical_samples if sample.get("accepted") is True
    ]
    medical_valid = sum(
        sample.get("finish_reason") == "stop" and bool(sample.get("response"))
        for sample in medical_accepted
    )
    return {
        "benefit_requested_n": len(benefit_samples),
        "benefit_accepted_structured_nonempty_n": benefit_valid,
        "medical_requested_n": len(medical_samples),
        "medical_accepted_n": len(medical_accepted),
        "medical_accepted_nonempty_stop_n": medical_valid,
        "medical_abstained_n": sum(
            sample.get("abstained") is True for sample in medical_samples
        ),
        "all_medical_accepted_are_nonempty_stop": (
            medical_valid == len(medical_accepted)
        ),
    }


def gate_passes(observed):
    if observed["benefit_requested_n"] != 2:
        raise ValueError("benefit gate request count differs")
    if observed["medical_requested_n"] != 16:
        raise ValueError("medical gate request count differs")
    for key in (
        "benefit_accepted_structured_nonempty_n",
        "medical_accepted_n",
        "medical_accepted_nonempty_stop_n",
        "medical_abstained_n",
    ):
        if isinstance(observed[key], bool) or not isinstance(observed[key], int):
            raise ValueError("gate count type differs")
    if (
        not 0 <= observed["benefit_accepted_structured_nonempty_n"] <= 2
        or not 0 <= observed["medical_accepted_n"] <= 16
        or not 0
        <= observed["medical_accepted_nonempty_stop_n"]
        <= observed["medical_accepted_n"]
        or observed["medical_accepted_n"] + observed["medical_abstained_n"]
        != 16
        or observed["all_medical_accepted_are_nonempty_stop"]
        is not (
            observed["medical_accepted_nonempty_stop_n"]
            == observed["medical_accepted_n"]
        )
    ):
        raise ValueError("gate count accounting differs")
    return (
        observed["medical_accepted_nonempty_stop_n"] >= 12
        and observed["all_medical_accepted_are_nonempty_stop"] is True
    )


def evaluate(args):
    workflow_root = os.path.abspath(args.workflow_root)
    if os.path.basename(workflow_root) != prepare.EXPECTED_OUTPUT_LEAF:
        raise ValueError(
            f"workflow root must end in {prepare.EXPECTED_OUTPUT_LEAF}"
        )
    plan_path = os.path.join(workflow_root, "control", "GATE_PLAN.json")
    plan = load_sealed(plan_path, "gate plan")
    if (
        plan["protocol_id"] != sampler.PROTOCOL_ID
        or plan["method_id"] != sampler.METHOD_ID
    ):
        raise ValueError("gate plan identity differs")
    authorization = load_sealed(args.authorization, "gate authorization")
    authorization_body = prepare.verify_seal(
        authorization, "gate authorization"
    )
    if (
        authorization_body.get("protocol_id") != sampler.PROTOCOL_ID
        or authorization_body.get("method_id") != sampler.METHOD_ID
        or authorization_body.get("stage") != "gate"
        or authorization_body.get("authorized_gpu_jobs") != 1
        or authorization_body.get("h200_minutes_cap")
        != prepare.PLANNED_GATE_H200_MINUTES
        or authorization_body.get("maximum_cost_usd")
        != prepare.PLANNED_GATE_CAP_USD
        or authorization_body.get("external_api_calls_authorized") != 0
        or authorization_body.get("restart_or_resume_authorized") is not False
        or authorization_body.get("retry_authorized") is not False
    ):
        raise ValueError("gate authorization binding differs")
    source = sampler.primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=True
    )
    source_binding = sampler.legacy._source_manifest_binding(
        args.source_protocol_manifest
    )
    phase_data = {}
    for phase in ("benefit", "medical"):
        if phase == "benefit":
            profile, records = sampler.primary.load_massive_prompts(source, phase)
        else:
            profile, records = sampler.primary.load_medical_prompts(source)
        profile = dict(profile)
        profile["temperature"] = sampler.TEMPERATURE
        all_requests = sampler._expanded_requests(records, profile["n_samples"])
        requests = sampler.select_requests(phase, "gate", all_requests)
        generation_path = os.path.join(
            workflow_root, "generation", "gate", phase, "generation.json"
        )
        generation, samples, summary = audit_generation(
            generation_path, source_binding, phase, profile, requests
        )
        timing_path = os.path.join(
            workflow_root, "generation", "gate", phase, "timing.json"
        )
        timing = load_sealed(timing_path, f"{phase} gate timing")
        if (
            timing.get("protocol_id") != sampler.PROTOCOL_ID
            or timing.get("method_id") != sampler.METHOD_ID
            or timing.get("phase") != phase
            or timing.get("stage") != "gate"
            or timing.get("summary") != summary
        ):
            raise ValueError(f"{phase} gate timing differs")
        phase_data[phase] = {
            "generation": generation,
            "samples": samples,
            "summary": summary,
            "generation_binding": prepare.binding(generation_path, generation),
            "timing_binding": prepare.binding(timing_path, timing),
        }
    observed = gate_observation(
        phase_data["benefit"]["samples"], phase_data["medical"]["samples"]
    )
    passed = gate_passes(observed)
    elapsed = int(args.elapsed_seconds)
    if elapsed <= 0 or elapsed > prepare.PLANNED_GATE_H200_MINUTES * 60:
        raise ValueError("gate elapsed time is outside the frozen cap")
    estimated_cost = elapsed * prepare.H200_HOURLY_USD / 3600
    if estimated_cost > prepare.PLANNED_GATE_CAP_USD + 1e-12:
        raise ValueError("gate estimated cost exceeds the frozen cap")
    result = prepare.seal(
        {
            "schema_version": 1,
            "protocol_id": sampler.PROTOCOL_ID,
            "method_id": sampler.METHOD_ID,
            "gate_plan_payload_sha256": plan[prepare.SEAL_FIELD],
            "status": (
                "KALAI_S3_COVERAGE_GATE_PASS"
                if passed
                else "KALAI_S3_COVERAGE_GATE_FUTILITY_STOP"
            ),
            "completion_eligible": passed,
            "completion_authorized": False,
            "restart_or_resume_authorized": False,
            "external_api_calls": 0,
            "gpu_jobs_submitted_by_evaluator": 0,
            "observed": {
                **observed,
                "benefit_summary": phase_data["benefit"]["summary"],
                "medical_summary": phase_data["medical"]["summary"],
                "benefit_generation": phase_data["benefit"][
                    "generation_binding"
                ],
                "medical_generation": phase_data["medical"][
                    "generation_binding"
                ],
                "benefit_timing": phase_data["benefit"]["timing_binding"],
                "medical_timing": phase_data["medical"]["timing_binding"],
            },
            "criteria": plan["gate_criteria"],
            "timing": {
                "slurm_job_id": str(args.slurm_job_id),
                "elapsed_seconds": elapsed,
                "h200_hourly_usd": prepare.H200_HOURLY_USD,
                "actual_estimated_cost_usd": estimated_cost,
                "authorized_cap_usd": prepare.PLANNED_GATE_CAP_USD,
                "authorization": prepare.binding(
                    args.authorization, authorization
                ),
            },
        }
    )
    result_path = os.path.join(workflow_root, "control", "GATE_RESULT.json")
    if os.path.lexists(result_path):
        raise ValueError("gate result already exists; replacement is forbidden")
    prepare.write_new_json(result_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completion_eligible": result["completion_eligible"],
                "benefit_accepted_n": observed[
                    "benefit_accepted_structured_nonempty_n"
                ],
                "medical_accepted_nonempty_stop_n": observed[
                    "medical_accepted_nonempty_stop_n"
                ],
                "actual_estimated_cost_usd": estimated_cost,
                "completion_authorized": False,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    base = {
        "benefit_requested_n": 2,
        "benefit_accepted_structured_nonempty_n": 2,
        "medical_requested_n": 16,
        "medical_accepted_n": 12,
        "medical_accepted_nonempty_stop_n": 12,
        "medical_abstained_n": 4,
        "all_medical_accepted_are_nonempty_stop": True,
    }
    assert gate_passes(base)
    assert not gate_passes({**base, "medical_accepted_n": 11, "medical_accepted_nonempty_stop_n": 11, "medical_abstained_n": 5})
    assert not gate_passes(
        {
            **base,
            "medical_accepted_n": 13,
            "medical_accepted_nonempty_stop_n": 12,
            "medical_abstained_n": 3,
            "all_medical_accepted_are_nonempty_stop": False,
        }
    )
    assert gate_passes({**base, "benefit_accepted_structured_nonempty_n": 1})
    try:
        gate_passes({**base, "medical_abstained_n": 5})
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent gate accounting was accepted")
    print("MASSIVE_MEDICAL_KALAI_S3_V2_GATE_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-root")
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--elapsed-seconds", type=int)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--authorization")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if (
        not args.workflow_root
        or not args.source_protocol_manifest
        or args.elapsed_seconds is None
        or not args.slurm_job_id
        or not args.authorization
    ):
        parser.error(
            "--workflow-root, --source-protocol-manifest, --elapsed-seconds, "
            "--slurm-job-id, and --authorization are required"
        )
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
