#!/usr/bin/env python3
"""Versioned Kalai consensus sampler for the one-corrupted-of-four panel.

This contextual add-on implements Algorithm 1 with ``k=4``, ``s=3``, and
``R=20``.  Its public method identity is new, while its proposal seed namespace
is deliberately the sealed v1 namespace: changing ``s`` therefore changes the
acceptance rule but not proposal sources, candidate token seeds, or uniform
accept/reject draws for a fixed request.

The two execution partitions are disjoint.  ``gate`` contains two MASSIVE
requests and one outcome-blind selected sample for each of the 16 medical
prompts.  ``completion`` is the exact complement.  Completion requires a
sealed passing gate result and still requires separate GPU authorization at the
workflow layer.  There is no restart or partial-resume CLI.

The script has no external-API path.  ``--preflight-only`` never loads model
weights or requires a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_PATH = SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_v1.py"
SPEC = importlib.util.spec_from_file_location(
    "_massive_medical_whole_output_consensus_v1_for_s3", LEGACY_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load legacy sampler: {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)

from subliminal_mitigate.decoding.algorithms import (
    whole_output_s_smallest_acceptance,
)


PROTOCOL_ID = "massive_medical_kalai_s3_r20_v2"
METHOD_ID = "whole_output_consensus_m4_s3_r20_v2"
PROPOSAL_STREAM_ID = "whole_output_consensus_m4_max20_v1"
PANEL_ORDER = ("A", "B1", "B2", "B3")
SAFE_REFERENCE_LOWER_BOUND = 3
ARBITRARY_REFERENCE_UPPER_BOUND = 1
MAX_ATTEMPTS = 20
TEMPERATURE = 1.0
GATE_BENEFIT_REQUESTS = 2
GATE_MEDICAL_PROMPTS = 16
GATE_MEDICAL_ACCEPTED_MIN = 12
STAGES = ("gate", "completion")
OUTPUT_SEAL = "payload_sha256"


def _canonical(value):
    return legacy._canonical(value)


def _sha256(value):
    return legacy._sha256(value)


def _gate_rank(phase, request):
    material = "\0".join(
        (
            PROTOCOL_ID,
            METHOD_ID,
            "coverage_gate",
            phase,
            request["question_id"],
            str(request["sample_index"]),
            request["prompt_sha256"],
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _gate_request_indices(phase, requests):
    if phase == "benefit":
        ranked = sorted(
            requests,
            key=lambda request: (
                _gate_rank(phase, request), request["request_index"]
            ),
        )
        return {
            request["request_index"]
            for request in ranked[:GATE_BENEFIT_REQUESTS]
        }
    if phase != "medical":
        raise ValueError(f"unknown phase: {phase}")
    by_question = {}
    for request in requests:
        by_question.setdefault(request["question_id"], []).append(request)
    selected = set()
    for question_id in sorted(by_question):
        request = min(
            by_question[question_id],
            key=lambda item: (
                _gate_rank(phase, item), item["request_index"]
            ),
        )
        selected.add(request["request_index"])
    return selected


def select_requests(phase, stage, requests):
    """Return the sealed gate or its exact complement in source order."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    indices = _gate_request_indices(phase, requests)
    want_gate = stage == "gate"
    return [
        request
        for request in requests
        if (request["request_index"] in indices) is want_gate
    ]


def _acceptance(sequence_logps):
    return whole_output_s_smallest_acceptance(
        sequence_logps, safe_references=SAFE_REFERENCE_LOWER_BOUND
    )


def _stream_meta(source_manifest, phase, stage, profile, requests):
    return {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "analysis_scope": "contextual_post_hoc_not_primary_gate_eligible",
        "primary_gate_eligible": False,
        "source_protocol": source_manifest,
        "phase": phase,
        "stage": stage,
        "panel_order": list(PANEL_ORDER),
        "panel_size": len(PANEL_ORDER),
        "safe_reference_lower_bound": SAFE_REFERENCE_LOWER_BOUND,
        "arbitrary_reference_upper_bound": ARBITRARY_REFERENCE_UPPER_BOUND,
        "constructed_threat_model": "one_corrupted_reference_among_four_unlabeled",
        "proposal": "uniform_complete_sequence_mixture",
        "proposal_stream_id": PROPOSAL_STREAM_ID,
        "acceptance": (
            "mean_three_smallest_complete_sequence_probabilities_"
            "over_mean_all_four"
        ),
        "max_attempts": MAX_ATTEMPTS,
        "temperature": TEMPERATURE,
        "seed": legacy.primary.GENERATION_SEED,
        "abstention_policy": "abstain_after_20_rejections_not_a_judge_label",
        "truncated_medical_candidate_policy": "ineligible_for_acceptance",
        "grammar_likelihood_policy": (
            "same_hard_mask_and_per_reference_renormalization_each_token"
            if phase == "benefit"
            else None
        ),
        "partition_policy": (
            "outcome_blind_hash_gate" if stage == "gate" else "exact_gate_complement"
        ),
        "profile": {
            key: profile[key]
            for key in (
                "domain",
                "endpoint",
                "n_samples",
                "temperature",
                "max_new_tokens",
                "max_context",
                "prompt_file_sha256",
            )
        },
        "requested_n": len(requests),
        "request_keys": [
            [request["question_id"], request["sample_index"]]
            for request in requests
        ],
    }


# Reuse the thoroughly audited candidate-generation and shard machinery while
# replacing only versioned policy hooks.  Python resolves these names in the
# legacy module at call time.
legacy.PROTOCOL_ID = PROTOCOL_ID
legacy.METHOD_ID = METHOD_ID
legacy.PROPOSAL_STREAM_ID = PROPOSAL_STREAM_ID
legacy.PANEL_ORDER = PANEL_ORDER
legacy.MAX_ATTEMPTS = MAX_ATTEMPTS
legacy.TEMPERATURE = TEMPERATURE
legacy.select_requests = select_requests
legacy.whole_output_acceptance = _acceptance
legacy._stream_meta = _stream_meta

primary = legacy.primary
_expanded_requests = legacy._expanded_requests
_audit_sample = legacy._audit_sample
summarize_samples = legacy.summarize_samples
_verify_seal = legacy._verify_seal


def _verify_bound_json(binding, expected_path, description):
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "size_bytes",
        "file_sha256",
        "payload_sha256",
    }:
        raise ValueError(f"{description} binding schema differs")
    expected_path = str(Path(expected_path).resolve())
    if binding["path"] != expected_path:
        raise ValueError(f"{description} path binding differs")
    payload, raw = primary.load_json_regular(expected_path, description)
    _verify_seal(payload, description)
    if (
        binding["size_bytes"] != len(raw)
        or binding["file_sha256"] != _sha256(raw)
        or binding["payload_sha256"] != payload[OUTPUT_SEAL]
    ):
        raise ValueError(f"{description} content binding differs")
    return payload


def verify_gpu_authorization(path, stage):
    payload, _ = primary.load_json_regular(path, f"Kalai s=3 {stage} authority")
    body = _verify_seal(payload, f"Kalai s=3 {stage} authority")
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != stage
        or body.get("authorized_gpu_jobs") != 1
        or body.get("external_api_calls_authorized") != 0
        or body.get("automatic_continuation_authorized") is not False
        or body.get("restart_or_resume_authorized") is not False
        or body.get("retry_authorized") is not False
    ):
        raise ValueError(f"Kalai s=3 {stage} authority differs")
    if stage == "gate" and (
        body.get("h200_minutes_cap") != 20
        or body.get("maximum_cost_usd") != 0.3
        or body.get("program_ceiling_usd") != 5.9933725
    ):
        raise ValueError("Kalai s=3 gate authority cap differs")
    return payload


def verify_passing_gate(path):
    result_path = Path(path).resolve()
    if result_path.name != "GATE_RESULT.json" or result_path.parent.name != "control":
        raise ValueError("Kalai s=3 gate result path is not canonical")
    payload, _ = primary.load_json_regular(result_path, "Kalai s=3 gate result")
    body = _verify_seal(payload, "Kalai s=3 gate result")
    required = {
        "schema_version",
        "protocol_id",
        "method_id",
        "gate_plan_payload_sha256",
        "status",
        "completion_eligible",
        "completion_authorized",
        "restart_or_resume_authorized",
        "external_api_calls",
        "gpu_jobs_submitted_by_evaluator",
        "observed",
        "criteria",
        "timing",
    }
    if set(body) != required:
        raise ValueError("Kalai s=3 gate result schema differs")
    plan_path = result_path.parent / "GATE_PLAN.json"
    plan, plan_raw = primary.load_json_regular(plan_path, "Kalai s=3 gate plan")
    plan_body = _verify_seal(plan, "Kalai s=3 gate plan")
    stage_path = result_path.parent / "CPU_STAGE.json"
    cpu_stage, _ = primary.load_json_regular(stage_path, "Kalai s=3 CPU stage")
    cpu_body = _verify_seal(cpu_stage, "Kalai s=3 CPU stage")
    if (
        plan_body.get("protocol_id") != PROTOCOL_ID
        or plan_body.get("method_id") != METHOD_ID
        or body["gate_plan_payload_sha256"] != plan[OUTPUT_SEAL]
        or body["criteria"] != plan_body.get("gate_criteria")
        or cpu_body.get("protocol_id") != PROTOCOL_ID
        or cpu_body.get("method_id") != METHOD_ID
        or cpu_body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
    ):
        raise ValueError("Kalai s=3 gate plan or CPU-stage binding differs")
    plan_binding = cpu_body.get("gate_plan")
    if (
        not isinstance(plan_binding, dict)
        or plan_binding.get("path") != str(plan_path)
        or plan_binding.get("size_bytes") != len(plan_raw)
        or plan_binding.get("file_sha256") != _sha256(plan_raw)
        or plan_binding.get("payload_sha256") != plan[OUTPUT_SEAL]
    ):
        raise ValueError("Kalai s=3 CPU-stage gate-plan binding differs")
    observed = body.get("observed")
    expected_observed_keys = {
        "benefit_requested_n",
        "benefit_accepted_structured_nonempty_n",
        "medical_requested_n",
        "medical_accepted_n",
        "medical_accepted_nonempty_stop_n",
        "medical_abstained_n",
        "all_medical_accepted_are_nonempty_stop",
        "benefit_summary",
        "medical_summary",
        "benefit_generation",
        "medical_generation",
        "benefit_timing",
        "medical_timing",
    }
    if not isinstance(observed, dict) or set(observed) != expected_observed_keys:
        raise ValueError("Kalai s=3 gate observation schema differs")
    for key in (
        "benefit_requested_n",
        "benefit_accepted_structured_nonempty_n",
        "medical_requested_n",
        "medical_accepted_n",
        "medical_accepted_nonempty_stop_n",
        "medical_abstained_n",
    ):
        if isinstance(observed[key], bool) or not isinstance(observed[key], int):
            raise ValueError("Kalai s=3 gate count type differs")
    if (
        observed["benefit_requested_n"] != 2
        or not 0
        <= observed["benefit_accepted_structured_nonempty_n"]
        <= 2
        or observed["medical_requested_n"] != 16
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
        raise ValueError("Kalai s=3 gate observation accounting differs")
    workflow_root = result_path.parent.parent
    generation_payloads = {}
    for phase in ("benefit", "medical"):
        generation_payloads[phase] = _verify_bound_json(
            observed[f"{phase}_generation"],
            workflow_root / "generation" / "gate" / phase / "generation.json",
            f"Kalai s=3 {phase} gate generation",
        )
        _verify_bound_json(
            observed[f"{phase}_timing"],
            workflow_root / "generation" / "gate" / phase / "timing.json",
            f"Kalai s=3 {phase} gate timing",
        )
        if generation_payloads[phase].get("summary") != observed[
            f"{phase}_summary"
        ]:
            raise ValueError(f"Kalai s=3 {phase} summary binding differs")
    benefit_samples = generation_payloads["benefit"].get("samples")
    medical_samples = generation_payloads["medical"].get("samples")
    if not isinstance(benefit_samples, list) or not isinstance(medical_samples, list):
        raise ValueError("Kalai s=3 gate samples differ")
    recomputed = {
        "benefit_requested_n": len(benefit_samples),
        "benefit_accepted_structured_nonempty_n": sum(
            sample.get("accepted") is True
            and sample.get("finish_reason") == "stop"
            and bool(sample.get("response"))
            and isinstance(sample.get("prediction"), dict)
            for sample in benefit_samples
        ),
        "medical_requested_n": len(medical_samples),
        "medical_accepted_n": sum(
            sample.get("accepted") is True for sample in medical_samples
        ),
        "medical_accepted_nonempty_stop_n": sum(
            sample.get("accepted") is True
            and sample.get("finish_reason") == "stop"
            and bool(sample.get("response"))
            for sample in medical_samples
        ),
        "medical_abstained_n": sum(
            sample.get("abstained") is True for sample in medical_samples
        ),
    }
    recomputed["all_medical_accepted_are_nonempty_stop"] = (
        recomputed["medical_accepted_nonempty_stop_n"]
        == recomputed["medical_accepted_n"]
    )
    if any(observed[key] != value for key, value in recomputed.items()):
        raise ValueError("Kalai s=3 gate observation differs from generation")
    timing = body.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Kalai s=3 gate timing schema differs")
    elapsed = timing.get("elapsed_seconds")
    rate = timing.get("h200_hourly_usd")
    cost = timing.get("actual_estimated_cost_usd")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int)
        or not 1 <= elapsed <= 1200
        or rate != 0.9
        or timing.get("authorized_cap_usd") != 0.3
        or not isinstance(cost, (int, float))
        or not math.isclose(cost, elapsed * 0.9 / 3600, rel_tol=0, abs_tol=1e-15)
    ):
        raise ValueError("Kalai s=3 gate cost accounting differs")
    authorization_path = result_path.parent / "GATE_AUTHORIZATION.json"
    bound_authorization = _verify_bound_json(
        timing.get("authorization"),
        authorization_path,
        "Kalai s=3 gate authorization",
    )
    if verify_gpu_authorization(authorization_path, "gate") != bound_authorization:
        raise ValueError("Kalai s=3 gate authority payload differs")
    passed = (
        observed["medical_accepted_nonempty_stop_n"] >= 12
        and observed["all_medical_accepted_are_nonempty_stop"] is True
    )
    if (
        body["schema_version"] != 1
        or body["protocol_id"] != PROTOCOL_ID
        or body["method_id"] != METHOD_ID
        or body["status"] != "KALAI_S3_COVERAGE_GATE_PASS"
        or body["completion_eligible"] is not True
        or body["completion_authorized"] is not False
        or body["restart_or_resume_authorized"] is not False
        or body["external_api_calls"] != 0
        or body["gpu_jobs_submitted_by_evaluator"] != 0
        or passed is not True
    ):
        raise ValueError("Kalai s=3 completion lacks a valid passing gate")
    return payload


def _attempt_stream_prefix(phase, question_id, sample_index, attempts):
    request_seed = primary.tuple_seed(
        primary.GENERATION_SEED,
        PROPOSAL_STREAM_ID,
        phase,
        question_id,
        sample_index,
    )
    rng = random.Random(request_seed)
    result = []
    for attempt_index in range(attempts):
        source = PANEL_ORDER[rng.randrange(len(PANEL_ORDER))]
        token_seed = primary.tuple_seed(
            request_seed, "candidate_tokens", attempt_index, source
        )
        draw = rng.random()
        result.append((source, token_seed, draw))
    return request_seed, result


def _self_test():
    medical = [
        {
            "request_index": prompt * 5 + sample,
            "prompt_ordinal": prompt,
            "question_id": f"medical_{prompt:02d}",
            "sample_index": sample,
            "prompt_sha256": hashlib.sha256(
                f"prompt-{prompt}".encode("utf-8")
            ).hexdigest(),
        }
        for prompt in range(16)
        for sample in range(5)
    ]
    gate = select_requests("medical", "gate", medical)
    completion = select_requests("medical", "completion", medical)
    assert len(gate) == 16 and len(completion) == 64
    assert len({request["question_id"] for request in gate}) == 16
    gate_keys = {(item["question_id"], item["sample_index"]) for item in gate}
    completion_keys = {
        (item["question_id"], item["sample_index"]) for item in completion
    }
    assert gate_keys.isdisjoint(completion_keys)
    assert gate_keys | completion_keys == {
        (item["question_id"], item["sample_index"]) for item in medical
    }
    benefit = [
        {
            "request_index": index,
            "prompt_ordinal": index,
            "question_id": f"massive_{index:03d}",
            "sample_index": 0,
            "prompt_sha256": "a" * 64,
        }
        for index in range(360)
    ]
    assert len(select_requests("benefit", "gate", benefit)) == 2
    assert len(select_requests("benefit", "completion", benefit)) == 358
    probabilities = (0.1, 0.2, 0.3, 0.4)
    assert math.isclose(
        _acceptance([math.log(value) for value in probabilities]),
        0.8,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    request_seed, prefix = _attempt_stream_prefix(
        "medical", "medical_official16_07", 3, 5
    )
    assert request_seed == 1950019501691531792
    assert [item[0] for item in prefix] == ["A", "B2", "B2", "B2", "B2"]
    assert [item[1] for item in prefix] == [
        5070597452765956032,
        2540965492967497984,
        2573159332371250326,
        8649926714252353918,
        6441227068952657104,
    ]
    assert [item[2].hex() for item in prefix] == [
        "0x1.2b3407f436efcp-3",
        "0x1.c2b41787e64a4p-3",
        "0x1.4bd621bd3e958p-2",
        "0x1.5d865f9318a96p-1",
        "0x1.ef7e6587d37f6p-2",
    ]
    print("MASSIVE_MEDICAL_KALAI_S3_R20_V2_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--phase", choices=("benefit", "medical"))
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-result")
    parser.add_argument("--gpu-authorization")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if (
        not args.source_protocol_manifest
        or not args.output_root
        or not args.phase
        or not args.stage
    ):
        parser.error(
            "--source-protocol-manifest, --output-root, --phase, and --stage "
            "are required"
        )
    if args.preflight_only and args.audit_only:
        parser.error("--preflight-only and --audit-only are mutually exclusive")
    if args.stage == "gate" and args.gate_result:
        parser.error("--gate-result is forbidden for the gate partition")
    if args.stage == "completion" and not args.preflight_only:
        if not args.gate_result:
            parser.error("completion requires --gate-result")
        verify_passing_gate(args.gate_result)
    if not args.preflight_only and not args.audit_only:
        if not args.gpu_authorization:
            parser.error("generation requires --gpu-authorization")
        verify_gpu_authorization(args.gpu_authorization, args.stage)
    base_args = argparse.Namespace(
        source_protocol_manifest=args.source_protocol_manifest,
        output_root=args.output_root,
        phase=args.phase,
        stage=args.stage,
        device=args.device,
        preflight_only=args.preflight_only,
        audit_only=args.audit_only,
        resume_partial=False,
    )
    return legacy._run(base_args)


if __name__ == "__main__":
    raise SystemExit(main())
