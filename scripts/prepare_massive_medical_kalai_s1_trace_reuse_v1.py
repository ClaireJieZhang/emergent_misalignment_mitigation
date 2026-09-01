#!/usr/bin/env python3
"""Build the CPU-only Kalai ``s=1`` trace-reuse continuation plan.

The full Kalai ``s=3, R=20`` run and the legacy ``s=1`` medical smoke use
the same proposal stream.  This program replays the stricter ``s=1``
acceptance rule over those sealed traces.  It never generates text and has no
GPU or external-API path.

Only the legacy *medical* smoke is eligible for cross-run reuse.  The legacy
MASSIVE smoke is deliberately excluded because its independently generated
sequence log probabilities are not bit-identical to the full ``s=3`` trace.
For a medical overlap, every proposal/result field in the common prefix must
be exactly equal; otherwise planning fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subliminal_mitigate.decoding.algorithms import (  # noqa: E402
    whole_output_acceptance,
    whole_output_s_smallest_acceptance,
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _load_module(
    "_massive_medical_whole_output_consensus_v1_for_s1_reuse",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_v1.py",
)
s3 = _load_module(
    "_massive_medical_whole_output_consensus_s3_v2_for_s1_reuse",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_s3_v2.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_r20_trace_reuse_v1"
METHOD_ID = "whole_output_consensus_m4_s1_r20_sensitivity_v1"
PROPOSAL_STREAM_ID = "whole_output_consensus_m4_max20_v1"
S3_PROTOCOL_ID = "massive_medical_kalai_s3_r20_v2"
S3_METHOD_ID = "whole_output_consensus_m4_s3_r20_v2"
LEGACY_PROTOCOL_ID = "massive_medical_composition_baselines_v1"
PANEL_ORDER = ("A", "B1", "B2", "B3")
SAFE_REFERENCE_LOWER_BOUND = 1
MAX_ATTEMPTS = 20
OUTPUT_SEAL = "payload_sha256"
CLASSIFICATIONS = (
    "accepted_in_sealed_trace",
    "definitive_abstain",
    "needs_continuation",
)
DISPOSITION_BY_CLASSIFICATION = {
    "accepted_in_sealed_trace": "reused_accept",
    "definitive_abstain": "reused_abstain",
    "needs_continuation": "unresolved",
}
SOURCE_GENERATION_KEYS = (
    "benefit_gate",
    "benefit_completion",
    "medical_gate",
    "medical_completion",
)

# These fields describe the candidate and proposal stream.  They must be
# bit-for-bit equal across an independently generated overlap.  The two
# acceptance fields are intentionally absent because s=1 and s=3 differ there.
COMMON_PREFIX_FIELDS = (
    "attempt_index",
    "proposal_source",
    "token_seed",
    "finish_reason",
    "generated_tokens",
    "sampled_tokens",
    "sequence_logps",
    "uniform_draw",
    "eligible_for_acceptance",
    "response_sha256",
)


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def seal(body):
    result = dict(body)
    result.pop(OUTPUT_SEAL, None)
    result[OUTPUT_SEAL] = sha256_bytes(canonical_bytes(result))
    return result


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(OUTPUT_SEAL, None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} has an invalid {OUTPUT_SEAL}")
    return body


def _is_hex64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def request_key(value):
    return value["question_id"], value["sample_index"]


def request_fields(sample):
    return {
        key: sample[key]
        for key in (
            "request_index",
            "prompt_ordinal",
            "question_id",
            "sample_index",
            "prompt_sha256",
        )
    }


def _expected_request_seed(phase, sample):
    return legacy.primary.tuple_seed(
        legacy.primary.GENERATION_SEED,
        PROPOSAL_STREAM_ID,
        phase,
        sample["question_id"],
        sample["sample_index"],
    )


def validate_proposal_trace(sample, phase, description="sample"):
    """Validate the exact hard-pinned proposal RNG stream and attempt fields."""

    if phase not in {"benefit", "medical"}:
        raise ValueError(f"unknown phase: {phase}")
    if not isinstance(sample, dict):
        raise ValueError(f"{description} is not an object")
    for key in (
        "request_index",
        "prompt_ordinal",
        "question_id",
        "sample_index",
        "prompt_sha256",
        "request_seed",
        "attempts",
        "sample_sha256",
    ):
        if key not in sample:
            raise ValueError(f"{description} lacks {key}")
    if not _is_hex64(sample["prompt_sha256"]):
        raise ValueError(f"{description} prompt hash differs")
    expected_seed = _expected_request_seed(phase, sample)
    if sample["request_seed"] != expected_seed:
        raise ValueError(f"{description} request seed differs")
    body = dict(sample)
    observed_sample_seal = body.pop("sample_sha256", None)
    if observed_sample_seal != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} sample seal differs")
    attempts = sample["attempts"]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= MAX_ATTEMPTS:
        raise ValueError(f"{description} attempt count differs")
    rng = random.Random(expected_seed)
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or set(attempt) != {
            *COMMON_PREFIX_FIELDS,
            "acceptance_probability",
            "accepted",
        }:
            raise ValueError(f"{description} attempt schema differs at {index}")
        expected_source = PANEL_ORDER[rng.randrange(len(PANEL_ORDER))]
        expected_token_seed = legacy.primary.tuple_seed(
            expected_seed, "candidate_tokens", index, expected_source
        )
        expected_uniform = rng.random()
        if (
            attempt["attempt_index"] != index
            or attempt["proposal_source"] != expected_source
            or attempt["token_seed"] != expected_token_seed
            or attempt["uniform_draw"] != expected_uniform
        ):
            raise ValueError(
                f"{description} proposal stream differs at attempt {index}"
            )
        if attempt["finish_reason"] not in {"stop", "max_new_tokens"}:
            raise ValueError(f"{description} finish reason differs at {index}")
        generated = attempt["generated_tokens"]
        sampled = attempt["sampled_tokens"]
        if (
            isinstance(generated, bool)
            or not isinstance(generated, int)
            or generated < 0
            or isinstance(sampled, bool)
            or not isinstance(sampled, int)
            or sampled < 1
        ):
            raise ValueError(f"{description} token count differs at {index}")
        expected_sampled = generated
        if phase == "medical" and attempt["finish_reason"] == "stop":
            expected_sampled += 1
        if sampled != expected_sampled:
            raise ValueError(f"{description} sampled token count differs at {index}")
        sequence_logps = attempt["sequence_logps"]
        if not isinstance(sequence_logps, dict) or set(sequence_logps) != set(
            PANEL_ORDER
        ):
            raise ValueError(
                f"{description} sequence-logp mapping differs at {index}"
            )
        for role in PANEL_ORDER:
            value = sequence_logps[role]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"{description} sequence log probability differs at {index}"
                )
        if (
            isinstance(attempt["uniform_draw"], bool)
            or not 0.0 <= float(attempt["uniform_draw"]) < 1.0
            or not _is_hex64(attempt["response_sha256"])
        ):
            raise ValueError(f"{description} attempt scalar differs at {index}")
        expected_eligible = not (
            phase == "medical" and attempt["finish_reason"] != "stop"
        )
        if attempt["eligible_for_acceptance"] is not expected_eligible:
            raise ValueError(f"{description} eligibility differs at {index}")
    return attempts


def validate_source_sample(sample, request, phase, profile, source_kind):
    """Apply the original source audit and then proposal-stream validation."""

    if source_kind == "s3_full":
        s3._audit_sample(sample, request, phase, profile)
    elif source_kind == "s1_medical_smoke":
        if phase != "medical":
            raise ValueError("legacy s=1 reuse is medical-only")
        legacy._audit_sample(sample, request, phase, profile)
    else:
        raise ValueError(f"unknown source kind: {source_kind}")
    validate_proposal_trace(sample, phase, f"{source_kind} {request_key(sample)}")


def _attempt_acceptance(attempt, safe_references):
    values = [float(attempt["sequence_logps"][role]) for role in PANEL_ORDER]
    return whole_output_s_smallest_acceptance(values, safe_references)


def normalized_s1_attempt(attempt):
    result = dict(attempt)
    probability = whole_output_acceptance(
        [float(attempt["sequence_logps"][role]) for role in PANEL_ORDER]
    )
    accepted = (
        attempt["eligible_for_acceptance"] is True
        and float(attempt["uniform_draw"]) < probability
    )
    result["acceptance_probability"] = probability
    result["accepted"] = accepted
    return result


def _verify_source_acceptance(sample, source_kind):
    if source_kind == "s3_full":
        safe_references = 3
    elif source_kind in {"s1_medical_smoke", "s1_continuation"}:
        safe_references = 1
    else:
        raise ValueError(f"unknown acceptance source: {source_kind}")
    first_accept = None
    for index, attempt in enumerate(sample["attempts"]):
        expected_probability = _attempt_acceptance(attempt, safe_references)
        observed = attempt["acceptance_probability"]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isclose(
                float(observed), expected_probability, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ValueError(
                f"{source_kind} acceptance probability differs at attempt {index}"
            )
        expected_accept = (
            attempt["eligible_for_acceptance"] is True
            and float(attempt["uniform_draw"]) < expected_probability
        )
        if attempt["accepted"] is not expected_accept:
            raise ValueError(
                f"{source_kind} acceptance decision differs at attempt {index}"
            )
        if expected_accept and first_accept is None:
            first_accept = index
        elif expected_accept:
            raise ValueError(f"{source_kind} has multiple accepted attempts")
    if first_accept is not None and first_accept + 1 != len(sample["attempts"]):
        raise ValueError(f"{source_kind} continues after acceptance")


def _common_attempt(attempt):
    return {key: attempt[key] for key in COMMON_PREFIX_FIELDS}


def validate_exact_common_prefix(s3_sample, smoke_sample):
    """Require exact equality of every candidate/proposal field in overlap."""

    if request_fields(s3_sample) != request_fields(smoke_sample):
        raise ValueError("medical smoke overlap request identity differs")
    if s3_sample["request_seed"] != smoke_sample["request_seed"]:
        raise ValueError("medical smoke overlap request seed differs")
    common = min(len(s3_sample["attempts"]), len(smoke_sample["attempts"]))
    if common < 1:
        raise ValueError("medical smoke overlap has an empty common prefix")
    for index in range(common):
        if _common_attempt(s3_sample["attempts"][index]) != _common_attempt(
            smoke_sample["attempts"][index]
        ):
            raise ValueError(
                f"medical smoke exact common prefix differs at attempt {index}"
            )
    return common


def merge_trace_sources(s3_sample, smoke_sample=None):
    """Return the longest exactly compatible trace and its provenance."""

    _verify_source_acceptance(s3_sample, "s3_full")
    if smoke_sample is None:
        return {
            "sample": s3_sample,
            "trace_source": "s3_full",
            "common_prefix_attempts": None,
        }
    validate_exact_common_prefix(s3_sample, smoke_sample)
    _verify_source_acceptance(smoke_sample, "s1_medical_smoke")
    if len(smoke_sample["attempts"]) > len(s3_sample["attempts"]):
        selected = smoke_sample
        source = "s1_medical_smoke"
    else:
        selected = s3_sample
        source = "s3_full"
    return {
        "sample": selected,
        "trace_source": source,
        "common_prefix_attempts": min(
            len(s3_sample["attempts"]), len(smoke_sample["attempts"])
        ),
    }


def _terminal_response_source(accepted_index, expected_hash, candidates):
    matches = []
    for source_name, sample in candidates:
        if sample is None or sample.get("accepted") is not True:
            continue
        if len(sample["attempts"]) != accepted_index + 1:
            continue
        if (
            sample.get("response_sha256") == expected_hash
            and sample["attempts"][-1]["response_sha256"] == expected_hash
            and sha256_bytes(sample.get("response", "").encode("utf-8"))
            == expected_hash
        ):
            matches.append((source_name, sample))
    if not matches:
        raise ValueError(
            "s=1 acceptance lacks its sealed terminal response; "
            "an earlier hidden candidate cannot be reconstructed"
        )
    return matches[0]


def _reusable_sample(phase, source_sample, attempts, accepted_index):
    request = request_fields(source_sample)
    if accepted_index is None:
        result = {
            **request,
            "request_seed": source_sample["request_seed"],
            "accepted": False,
            "abstained": True,
            "attempts_used": MAX_ATTEMPTS,
            "response": "",
            "response_sha256": sha256_bytes(b""),
            "finish_reason": "abstain",
            "generated_tokens": 0,
            "attempts": attempts,
        }
    else:
        terminal = attempts[accepted_index]
        result = {
            **request,
            "request_seed": source_sample["request_seed"],
            "accepted": True,
            "abstained": False,
            "attempts_used": accepted_index + 1,
            "accepted_source": terminal["proposal_source"],
            "response": source_sample["response"],
            "response_sha256": terminal["response_sha256"],
            "finish_reason": terminal["finish_reason"],
            "generated_tokens": terminal["generated_tokens"],
            "attempts": attempts[: accepted_index + 1],
        }
        if phase == "benefit":
            result["prediction"] = source_sample["prediction"]
    result["sample_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def classify_reusable_trace(
    phase, stage, s3_sample, smoke_sample=None
):
    """Replay s=1 and return one deterministic, continuation-ready row."""

    validate_proposal_trace(s3_sample, phase, "s3 replay source")
    if smoke_sample is not None:
        if phase != "medical":
            raise ValueError("legacy s=1 reuse is medical-only")
        validate_proposal_trace(smoke_sample, phase, "s1 medical smoke source")
    merged = merge_trace_sources(s3_sample, smoke_sample)
    trace_sample = merged["sample"]
    attempts = [normalized_s1_attempt(item) for item in trace_sample["attempts"]]
    accepted_indices = [
        index for index, attempt in enumerate(attempts) if attempt["accepted"]
    ]
    if len(accepted_indices) > 1:
        raise ValueError("sealed trace contains multiple s=1 acceptances")
    accepted_index = accepted_indices[0] if accepted_indices else None
    if accepted_index is not None:
        # p_(1) <= mean of the three smallest probabilities, so an s=1 hit
        # must also be the terminal s=3 hit.  Requiring a terminal response
        # source makes that scientific assumption executable and fail-closed.
        response_source_name, response_source = _terminal_response_source(
            accepted_index,
            attempts[accepted_index]["response_sha256"],
            (("s3_full", s3_sample), ("s1_medical_smoke", smoke_sample)),
        )
        if accepted_index + 1 != len(attempts):
            raise ValueError("sealed trace continues after its s=1 acceptance")
        reusable = _reusable_sample(
            phase, response_source, attempts, accepted_index
        )
        classification = "accepted_in_sealed_trace"
    elif len(attempts) == MAX_ATTEMPTS:
        reusable = _reusable_sample(phase, trace_sample, attempts, None)
        response_source_name = None
        classification = "definitive_abstain"
    else:
        reusable = None
        response_source_name = None
        classification = "needs_continuation"
    prefix = {
        "request_seed": trace_sample["request_seed"],
        "attempts": attempts,
    }
    prefix_sha256 = sha256_bytes(canonical_bytes(prefix))
    row = {
        "phase": phase,
        "stage": stage,
        "partition": "technical_gate" if stage == "gate" else "completion",
        **request_fields(s3_sample),
        "classification": classification,
        "disposition": DISPOSITION_BY_CLASSIFICATION[classification],
        "prefix_source": merged["trace_source"],
        "common_prefix_attempts": merged["common_prefix_attempts"],
        "s3_sample_sha256": s3_sample["sample_sha256"],
        "s1_medical_smoke_sample_sha256": (
            smoke_sample["sample_sha256"] if smoke_sample is not None else None
        ),
        "response_source": response_source_name,
        "request_seed": trace_sample["request_seed"],
        "prefix_attempts_used": len(attempts),
        "prefix_trace_sha256": prefix_sha256,
        "prefix_attempts": attempts,
        "next_attempt_index": (
            len(attempts) if classification == "needs_continuation" else None
        ),
        "max_new_attempts": (
            MAX_ATTEMPTS - len(attempts)
            if classification == "needs_continuation"
            else 0
        ),
        "reusable_terminal_sample": reusable,
    }
    row["row_sha256"] = sha256_bytes(canonical_bytes(row))
    return row


def _unique_sample_map(samples, description):
    result = {}
    for sample in samples:
        key = request_key(sample)
        if key in result:
            raise ValueError(f"duplicate {description} request: {key}")
        result[key] = sample
    return result


def exact_component_union(all_requests, gate_samples, completion_samples):
    """Join the sealed s=3 partitions without regenerating either partition."""

    expected_keys = [request_key(request) for request in all_requests]
    by_key = {}
    stage_by_key = {}
    for stage, samples in (
        ("gate", gate_samples),
        ("completion", completion_samples),
    ):
        for sample in samples:
            key = request_key(sample)
            if key in by_key:
                raise ValueError(f"duplicate request across s3 partitions: {key}")
            by_key[key] = sample
            stage_by_key[key] = stage
    if len(by_key) != len(expected_keys) or set(by_key) != set(expected_keys):
        raise ValueError("s3 gate and completion are not the exact request set")
    samples = [by_key[key] for key in expected_keys]
    if len({sample["sample_sha256"] for sample in samples}) != len(samples):
        raise ValueError("s3 component union contains duplicate sample seals")
    return samples, stage_by_key


def _expected_meta_identity(payload, source_kind, phase, stage=None):
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{source_kind} metadata is missing")
    if source_kind == "s3_component":
        if stage not in {"gate", "completion"}:
            raise ValueError("s3 component stage differs")
        expected = {
            "protocol_id": S3_PROTOCOL_ID,
            "method_id": S3_METHOD_ID,
            "stage": stage,
            "phase": phase,
        }
    elif source_kind == "s1_medical_smoke":
        expected = {
            "protocol_id": LEGACY_PROTOCOL_ID,
            "method_id": PROPOSAL_STREAM_ID,
            "stage": "smoke",
            "phase": "medical",
        }
    else:
        raise ValueError(f"unknown source kind: {source_kind}")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{source_kind} metadata {key} differs")


def validate_generation_payload(
    payload,
    source_kind,
    phase,
    expected_requests,
    profile,
    *,
    stage=None,
    source_manifest_binding=None,
):
    verify_seal(payload, f"{source_kind} {phase} generation")
    if set(payload) != {"meta", "summary", "samples", OUTPUT_SEAL}:
        raise ValueError(f"{source_kind} {phase} generation schema differs")
    _expected_meta_identity(payload, source_kind, phase, stage)
    if source_kind == "s3_component":
        expected_meta_body = s3._stream_meta(
            source_manifest_binding, phase, stage, profile, expected_requests
        )
        expected_meta = {
            **expected_meta_body,
            "stream_fingerprint": s3._sha256(s3._canonical(expected_meta_body)),
        }
        if payload["meta"] != expected_meta:
            raise ValueError(f"s3 {phase} {stage} metadata differs")
    elif source_kind == "s1_medical_smoke":
        expected_meta_body = legacy._stream_meta(
            source_manifest_binding,
            "medical",
            "smoke",
            profile,
            expected_requests,
        )
        expected_meta = {
            **expected_meta_body,
            "stream_fingerprint": legacy._sha256(
                legacy._canonical(expected_meta_body)
            ),
        }
        if payload["meta"] != expected_meta:
            raise ValueError("legacy s1 medical smoke metadata differs")
    samples = payload["samples"]
    if not isinstance(samples, list):
        raise ValueError(f"{source_kind} {phase} samples are missing")
    if len(samples) != len(expected_requests):
        raise ValueError(f"{source_kind} {phase} request count differs")
    expected_keys = [request_key(request) for request in expected_requests]
    if [request_key(sample) for sample in samples] != expected_keys:
        raise ValueError(f"{source_kind} {phase} request order differs")
    meta_keys = payload["meta"].get("request_keys")
    if meta_keys != [[key[0], key[1]] for key in expected_keys]:
        raise ValueError(f"{source_kind} {phase} metadata request keys differ")
    for sample, request in zip(samples, expected_requests):
        audit_kind = "s3_full" if source_kind == "s3_component" else source_kind
        validate_source_sample(sample, request, phase, profile, audit_kind)
        _verify_source_acceptance(sample, audit_kind)
    expected_summary = legacy.summarize_samples(samples)
    if payload["summary"] != expected_summary:
        raise ValueError(f"{source_kind} {phase} summary differs")
    _unique_sample_map(samples, f"{source_kind} {phase}")
    return samples


def build_phase_plan(
    s3_samples, phase, stage_by_key=None, smoke_samples=None
):
    """Build one phase plan from already audited samples.

    This dependency-light helper is intentionally public for the continuation
    controller and focused synthetic regression tests.
    """

    s3_by_key = _unique_sample_map(s3_samples, f"s3 {phase}")
    smoke_samples = [] if smoke_samples is None else smoke_samples
    if phase != "medical" and smoke_samples:
        raise ValueError("legacy smoke reuse is restricted to medical")
    smoke_by_key = _unique_sample_map(smoke_samples, "s1 medical smoke")
    unknown = sorted(set(smoke_by_key) - set(s3_by_key))
    if unknown:
        raise ValueError(f"medical smoke request is absent from s3 full: {unknown[0]}")
    if stage_by_key is None:
        stage_by_key = {key: "completion" for key in s3_by_key}
    if set(stage_by_key) != set(s3_by_key):
        raise ValueError(f"s3 {phase} stage mapping differs")
    if any(stage not in {"gate", "completion"} for stage in stage_by_key.values()):
        raise ValueError(f"s3 {phase} stage value differs")
    rows = [
        classify_reusable_trace(
            phase, stage_by_key[key], sample, smoke_by_key.get(key)
        )
        for key, sample in ((request_key(item), item) for item in s3_samples)
    ]
    counts = {
        classification: sum(
            row["classification"] == classification for row in rows
        )
        for classification in CLASSIFICATIONS
    }
    phase_plan = {
        "phase": phase,
        "requested_n": len(rows),
        "classification_counts": counts,
        "legacy_medical_smoke_overlap_n": len(smoke_by_key),
        "prefix_attempts_reused": sum(row["prefix_attempts_used"] for row in rows),
        "maximum_missing_attempts": sum(
            row["max_new_attempts"] for row in rows
        ),
        "request_keys": [
            [row["question_id"], row["sample_index"]] for row in rows
        ],
        "rows": rows,
    }
    phase_plan["phase_plan_sha256"] = sha256_bytes(canonical_bytes(phase_plan))
    return phase_plan


def summarize_partitions(phases):
    result = {}
    for partition in ("technical_gate", "completion"):
        selected = [
            row
            for phase in phases.values()
            for row in phase["rows"]
            if row["partition"] == partition
        ]
        unresolved = [
            row for row in selected if row["disposition"] == "unresolved"
        ]
        result[partition] = {
            "requested_rows": len(selected),
            "unresolved_by_phase": {
                phase: sum(row["phase"] == phase for row in unresolved)
                for phase in ("benefit", "medical")
            },
            "unresolved_rows": len(unresolved),
            "unresolved_prefix_attempts_reused": sum(
                len(row["prefix_attempts"]) for row in unresolved
            ),
            "maximum_new_attempts": sum(
                row["max_new_attempts"] for row in unresolved
            ),
        }
    return result


def validate_frozen_replay_counts(phases):
    expected_phase_counts = {
        "benefit": {
            "accepted_in_sealed_trace": 244,
            "definitive_abstain": 0,
            "needs_continuation": 116,
        },
        "medical": {
            "accepted_in_sealed_trace": 1,
            "definitive_abstain": 5,
            "needs_continuation": 74,
        },
    }
    for phase, expected in expected_phase_counts.items():
        if phases[phase]["classification_counts"] != expected:
            raise ValueError(f"frozen {phase} replay classification counts differ")
    observed = summarize_partitions(phases)
    expected = {
        "technical_gate": {
            "requested_rows": 18,
            "unresolved_by_phase": {"benefit": 1, "medical": 15},
            "unresolved_rows": 16,
            "unresolved_prefix_attempts_reused": 100,
            "maximum_new_attempts": 220,
        },
        "completion": {
            "requested_rows": 422,
            "unresolved_by_phase": {"benefit": 115, "medical": 59},
            "unresolved_rows": 174,
            "unresolved_prefix_attempts_reused": 533,
            "maximum_new_attempts": 2947,
        },
    }
    if observed != expected:
        raise ValueError("frozen partition replay counts differ")
    return observed


def assemble_s1_sample(plan_row, continuation_sample=None):
    """Return a reusable result or validate a controller-completed result.

    The continuation controller should emit a normal s=1 sample containing the
    exact normalized prefix from the plan followed by attempts beginning at
    ``next_attempt_index``.  This helper validates that boundary and its seal;
    the original sources and the plan are never modified.
    """

    body = dict(plan_row)
    observed = body.pop("row_sha256", None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError("plan row seal differs")
    reusable = plan_row["reusable_terminal_sample"]
    if reusable is not None:
        if continuation_sample is not None:
            raise ValueError("continuation supplied for an already final row")
        return reusable
    if plan_row["classification"] != "needs_continuation":
        raise ValueError("non-final plan row has an invalid classification")
    if not isinstance(continuation_sample, dict):
        raise ValueError("unresolved row requires a continuation sample")
    continuation_body = dict(continuation_sample)
    continuation_seal = continuation_body.pop("sample_sha256", None)
    if continuation_seal != sha256_bytes(canonical_bytes(continuation_body)):
        raise ValueError("continuation sample seal differs")
    for key, value in request_fields(plan_row).items():
        if continuation_sample.get(key) != value:
            raise ValueError(f"continuation request {key} differs")
    if continuation_sample.get("request_seed") != plan_row["request_seed"]:
        raise ValueError("continuation request seed differs")
    attempts = continuation_sample.get("attempts")
    prefix_attempts = plan_row["prefix_attempts"]
    if (
        not isinstance(attempts, list)
        or attempts[: len(prefix_attempts)] != prefix_attempts
    ):
        raise ValueError("continuation does not preserve the exact s=1 prefix")
    if len(attempts) <= len(prefix_attempts) or len(attempts) > MAX_ATTEMPTS:
        raise ValueError("continuation attempt boundary differs")
    validate_proposal_trace(
        continuation_sample, plan_row["phase"], "continuation sample"
    )
    _verify_source_acceptance(continuation_sample, "s1_continuation")
    return continuation_sample


def _verify_row(row):
    if not isinstance(row, dict):
        raise ValueError("replay-plan row is not an object")
    body = dict(row)
    observed = body.pop("row_sha256", None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError("replay-plan row seal differs")
    if set(row) != {
        "phase",
        "stage",
        "partition",
        "request_index",
        "prompt_ordinal",
        "question_id",
        "sample_index",
        "prompt_sha256",
        "classification",
        "disposition",
        "prefix_source",
        "common_prefix_attempts",
        "s3_sample_sha256",
        "s1_medical_smoke_sample_sha256",
        "response_source",
        "request_seed",
        "prefix_attempts_used",
        "prefix_trace_sha256",
        "prefix_attempts",
        "next_attempt_index",
        "max_new_attempts",
        "reusable_terminal_sample",
        "row_sha256",
    }:
        raise ValueError("replay-plan row schema differs")
    phase = row.get("phase")
    stage = row.get("stage")
    if phase not in {"benefit", "medical"} or stage not in {
        "gate",
        "completion",
    }:
        raise ValueError("replay-plan row phase/stage differs")
    expected_partition = "technical_gate" if stage == "gate" else "completion"
    if row.get("partition") != expected_partition:
        raise ValueError("replay-plan row partition differs")
    classification = row.get("classification")
    if (
        classification not in CLASSIFICATIONS
        or row.get("disposition")
        != DISPOSITION_BY_CLASSIFICATION[classification]
    ):
        raise ValueError("replay-plan row disposition differs")
    prefix_attempts = row.get("prefix_attempts")
    if not isinstance(prefix_attempts, list) or not 1 <= len(prefix_attempts) <= 20:
        raise ValueError("replay-plan row prefix differs")
    trace = {
        **request_fields(row),
        "request_seed": row.get("request_seed"),
        "attempts": prefix_attempts,
    }
    trace["sample_sha256"] = sha256_bytes(canonical_bytes(trace))
    validate_proposal_trace(trace, phase, "replay-plan normalized prefix")
    for index, attempt in enumerate(prefix_attempts):
        expected = normalized_s1_attempt(attempt)
        if attempt != expected:
            raise ValueError(
                f"replay-plan prefix is not s=1-normalized at attempt {index}"
            )
    expected_trace_hash = sha256_bytes(
        canonical_bytes(
            {"request_seed": row["request_seed"], "attempts": prefix_attempts}
        )
    )
    if (
        row.get("prefix_attempts_used") != len(prefix_attempts)
        or row.get("prefix_trace_sha256") != expected_trace_hash
    ):
        raise ValueError("replay-plan prefix binding differs")
    accepted = [index for index, item in enumerate(prefix_attempts) if item["accepted"]]
    reusable = row.get("reusable_terminal_sample")
    if classification == "accepted_in_sealed_trace":
        if accepted != [len(prefix_attempts) - 1] or not isinstance(reusable, dict):
            raise ValueError("reused-accept row terminal state differs")
        reusable_body = dict(reusable)
        reusable_seal = reusable_body.pop("sample_sha256", None)
        if reusable_seal != sha256_bytes(canonical_bytes(reusable_body)):
            raise ValueError("reused terminal sample seal differs")
        if reusable.get("attempts") != prefix_attempts or reusable.get("accepted") is not True:
            raise ValueError("reused accepted sample prefix differs")
        expected_next, expected_max = None, 0
    elif classification == "definitive_abstain":
        if accepted or len(prefix_attempts) != 20 or not isinstance(reusable, dict):
            raise ValueError("reused-abstain row terminal state differs")
        reusable_body = dict(reusable)
        reusable_seal = reusable_body.pop("sample_sha256", None)
        if reusable_seal != sha256_bytes(canonical_bytes(reusable_body)):
            raise ValueError("reused abstention sample seal differs")
        if reusable.get("attempts") != prefix_attempts or reusable.get("abstained") is not True:
            raise ValueError("reused abstention sample prefix differs")
        expected_next, expected_max = None, 0
    else:
        if accepted or len(prefix_attempts) >= 20 or reusable is not None:
            raise ValueError("unresolved row terminal state differs")
        expected_next = len(prefix_attempts)
        expected_max = 20 - len(prefix_attempts)
    if (
        row.get("next_attempt_index") != expected_next
        or row.get("max_new_attempts") != expected_max
    ):
        raise ValueError("replay-plan continuation boundary differs")


def _verify_bound_file(binding_value, description, seal_field=OUTPUT_SEAL):
    if not isinstance(binding_value, dict) or set(binding_value) != {
        "path",
        "size_bytes",
        "file_sha256",
        "payload_sha256",
    }:
        raise ValueError(f"{description} binding schema differs")
    path = binding_value["path"]
    payload, raw = legacy.primary.load_json_regular(path, description)
    if (
        len(raw) != binding_value["size_bytes"]
        or sha256_bytes(raw) != binding_value["file_sha256"]
        or payload.get(seal_field) != binding_value["payload_sha256"]
    ):
        raise ValueError(f"{description} binding content differs")
    if seal_field == OUTPUT_SEAL:
        verify_seal(payload, description)
    else:
        legacy.primary.verify_seal(payload, seal_field, description)


def rebuild_phases_from_bound_sources(bindings):
    """Repeat the complete source/profile/partition audit from immutable paths."""

    manifest_path = bindings["source_protocol_manifest"]["path"]
    protocol = legacy.primary.load_protocol_manifest(
        manifest_path, audit_models=False
    )
    source_manifest_binding = s3.legacy._source_manifest_binding(manifest_path)
    generations = bindings["source_generations"]
    phases = {}
    for phase in ("benefit", "medical"):
        if phase == "benefit":
            profile, records = legacy.primary.load_massive_prompts(protocol, phase)
        else:
            profile, records = legacy.primary.load_medical_prompts(protocol)
        profile = dict(profile)
        profile["temperature"] = 1.0
        all_requests = legacy._expanded_requests(records, profile["n_samples"])
        component_samples = {}
        for stage in ("gate", "completion"):
            key = f"{phase}_{stage}"
            payload, _ = load_generation(
                generations[key]["path"], f"bound s3 {key} generation"
            )
            component_samples[stage] = validate_generation_payload(
                payload,
                "s3_component",
                phase,
                s3.select_requests(phase, stage, all_requests),
                profile,
                stage=stage,
                source_manifest_binding=source_manifest_binding,
            )
        s3_samples, stage_by_key = exact_component_union(
            all_requests,
            component_samples["gate"],
            component_samples["completion"],
        )
        smoke_samples = None
        if phase == "medical":
            smoke_payload, _ = load_generation(
                bindings["s1_medical_smoke_generation"]["path"],
                "bound legacy s1 medical smoke generation",
            )
            smoke_requests = legacy.select_requests(
                "medical", "smoke", all_requests
            )
            smoke_samples = validate_generation_payload(
                smoke_payload,
                "s1_medical_smoke",
                "medical",
                smoke_requests,
                profile,
                source_manifest_binding=source_manifest_binding,
            )
        phases[phase] = build_phase_plan(
            s3_samples, phase, stage_by_key, smoke_samples
        )
    validate_frozen_replay_counts(phases)
    return phases


def load_and_verify_plan(path, audit_sources=True):
    """Load ``REPLAY_PLAN.json`` and return ``(payload, body)`` fail-closed."""

    if os.path.basename(path) != "REPLAY_PLAN.json":
        raise ValueError("replay plan must be named REPLAY_PLAN.json")
    payload, _ = legacy.primary.load_json_regular(path, "s1 replay plan")
    body = verify_seal(payload, "s1 replay plan")
    if (
        set(body)
        != {
            "schema_version",
            "protocol_id",
            "method_id",
            "proposal_stream_id",
            "safe_reference_lower_bound",
            "max_attempts",
            "analysis_scope",
            "source_bindings",
            "reuse_policy",
            "phases",
            "partition_summary",
            "continuation",
        }
        or body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("proposal_stream_id") != PROPOSAL_STREAM_ID
        or body.get("safe_reference_lower_bound") != 1
        or body.get("max_attempts") != 20
    ):
        raise ValueError("s1 replay-plan identity differs")
    phases = body.get("phases")
    if not isinstance(phases, dict) or list(phases) != ["benefit", "medical"]:
        raise ValueError("s1 replay-plan phases differ")
    for phase, phase_plan in phases.items():
        if not isinstance(phase_plan, dict):
            raise ValueError(f"{phase} replay plan is missing")
        phase_body = dict(phase_plan)
        observed = phase_body.pop("phase_plan_sha256", None)
        if observed != sha256_bytes(canonical_bytes(phase_body)):
            raise ValueError(f"{phase} replay-plan seal differs")
        if set(phase_plan) != {
            "phase",
            "requested_n",
            "classification_counts",
            "legacy_medical_smoke_overlap_n",
            "prefix_attempts_reused",
            "maximum_missing_attempts",
            "request_keys",
            "rows",
            "phase_plan_sha256",
        }:
            raise ValueError(f"{phase} replay-plan schema differs")
        rows = phase_plan.get("rows")
        if not isinstance(rows, list) or len(rows) != phase_plan.get("requested_n"):
            raise ValueError(f"{phase} replay-plan row count differs")
        for row in rows:
            _verify_row(row)
            if row["phase"] != phase:
                raise ValueError(f"{phase} replay-plan row phase differs")
        if phase_plan.get("request_keys") != [
            [row["question_id"], row["sample_index"]] for row in rows
        ]:
            raise ValueError(f"{phase} replay-plan request keys differ")
        if phase_plan.get("prefix_attempts_reused") != sum(
            row["prefix_attempts_used"] for row in rows
        ) or phase_plan.get("maximum_missing_attempts") != sum(
            row["max_new_attempts"] for row in rows
        ):
            raise ValueError(f"{phase} replay-plan attempt totals differ")
        if phase_plan["classification_counts"] != {
            classification: sum(
                row["classification"] == classification for row in rows
            )
            for classification in CLASSIFICATIONS
        }:
            raise ValueError(f"{phase} replay-plan counts differ")
    validate_frozen_replay_counts(phases)
    if body.get("partition_summary") != summarize_partitions(phases):
        raise ValueError("s1 replay-plan partition summary differs")
    if body.get("continuation") != {
        "requested_rows": 190,
        "maximum_missing_attempts": 3167,
        "gpu_jobs_submitted": 0,
        "external_api_calls": 0,
    }:
        raise ValueError("s1 replay-plan continuation totals differ")
    bindings = body.get("source_bindings")
    generations = bindings.get("source_generations") if isinstance(bindings, dict) else None
    if (
        not isinstance(bindings, dict)
        or set(bindings)
        != {
            "source_protocol_manifest",
            "source_generations",
            "s1_medical_smoke_generation",
        }
        or not isinstance(generations, dict)
        or set(generations) != set(SOURCE_GENERATION_KEYS)
    ):
        raise ValueError("s1 replay-plan source bindings differ")
    if audit_sources:
        _verify_bound_file(
            bindings["source_protocol_manifest"],
            "bound source protocol manifest",
            legacy.primary.MANIFEST_SEAL_FIELD,
        )
        for key in SOURCE_GENERATION_KEYS:
            _verify_bound_file(generations[key], f"bound s3 {key} generation")
        _verify_bound_file(
            bindings["s1_medical_smoke_generation"],
            "bound s1 medical smoke generation",
        )
        if rebuild_phases_from_bound_sources(bindings) != phases:
            raise ValueError("s1 replay plan differs from its bound source replay")
    return payload, body


def binding(path, payload, raw):
    return {
        "path": os.path.abspath(path),
        "size_bytes": len(raw),
        "file_sha256": sha256_bytes(raw),
        "payload_sha256": payload[OUTPUT_SEAL],
    }


def load_generation(path, description):
    payload, raw = legacy.primary.load_json_regular(path, description)
    verify_seal(payload, description)
    return payload, raw


def _write_idempotent(path, payload):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    if os.path.lexists(path):
        observed, _ = legacy.primary.load_json_regular(path, "existing s=1 plan")
        verify_seal(observed, "existing s=1 plan")
        if observed != payload:
            raise ValueError("existing s=1 plan differs")
        return
    temporary = path + ".tmp"
    if os.path.lexists(temporary):
        raise ValueError("temporary s=1 plan path already exists")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(args):
    protocol = legacy.primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=False
    )
    manifest_payload, manifest_raw = legacy.primary.load_json_regular(
        args.source_protocol_manifest, "source protocol manifest"
    )
    legacy.primary.verify_seal(
        manifest_payload,
        legacy.primary.MANIFEST_SEAL_FIELD,
        "source protocol manifest",
    )
    source_manifest_binding = s3.legacy._source_manifest_binding(
        args.source_protocol_manifest
    )
    source_bindings = {
        "source_protocol_manifest": {
            "path": os.path.abspath(args.source_protocol_manifest),
            "size_bytes": len(manifest_raw),
            "file_sha256": sha256_bytes(manifest_raw),
            "payload_sha256": manifest_payload[
                legacy.primary.MANIFEST_SEAL_FIELD
            ],
        },
        "source_generations": {},
    }
    phases = {}
    component_paths = {
        "benefit": {
            "gate": args.s3_benefit_gate_generation,
            "completion": args.s3_benefit_completion_generation,
        },
        "medical": {
            "gate": args.s3_medical_gate_generation,
            "completion": args.s3_medical_completion_generation,
        },
    }
    for phase in ("benefit", "medical"):
        if phase == "benefit":
            profile, records = legacy.primary.load_massive_prompts(protocol, phase)
        else:
            profile, records = legacy.primary.load_medical_prompts(protocol)
        profile = dict(profile)
        profile["temperature"] = 1.0
        all_requests = legacy._expanded_requests(records, profile["n_samples"])
        component_samples = {}
        for stage in ("gate", "completion"):
            requests = s3.select_requests(phase, stage, all_requests)
            source_path = component_paths[phase][stage]
            component_payload, component_raw = load_generation(
                source_path, f"s3 {phase} {stage} generation"
            )
            component_samples[stage] = validate_generation_payload(
                component_payload,
                "s3_component",
                phase,
                requests,
                profile,
                stage=stage,
                source_manifest_binding=source_manifest_binding,
            )
            source_bindings["source_generations"][f"{phase}_{stage}"] = binding(
                source_path, component_payload, component_raw
            )
        s3_samples, stage_by_key = exact_component_union(
            all_requests,
            component_samples["gate"],
            component_samples["completion"],
        )
        smoke_samples = None
        if phase == "medical":
            smoke_payload, smoke_raw = load_generation(
                args.s1_medical_smoke_generation,
                "legacy s1 medical smoke generation",
            )
            smoke_requests = legacy.select_requests(
                "medical", "smoke", all_requests
            )
            smoke_samples = validate_generation_payload(
                smoke_payload,
                "s1_medical_smoke",
                "medical",
                smoke_requests,
                profile,
                source_manifest_binding=source_manifest_binding,
            )
            source_bindings["s1_medical_smoke_generation"] = binding(
                args.s1_medical_smoke_generation, smoke_payload, smoke_raw
            )
        phases[phase] = build_phase_plan(
            s3_samples, phase, stage_by_key, smoke_samples
        )
    if set(source_bindings["source_generations"]) != set(SOURCE_GENERATION_KEYS):
        raise ValueError("s3 source-generation binding set differs")
    partition_summary = validate_frozen_replay_counts(phases)
    payload = seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "proposal_stream_id": PROPOSAL_STREAM_ID,
            "safe_reference_lower_bound": SAFE_REFERENCE_LOWER_BOUND,
            "max_attempts": MAX_ATTEMPTS,
            "analysis_scope": "post_hoc_sensitivity_not_primary_gate_eligible",
            "source_bindings": source_bindings,
            "reuse_policy": {
                "s3_full_traces": "eligible_after_full_source_and_stream_audit",
                "legacy_s1_medical_smoke": (
                    "eligible_only_after_exact_entire_common_prefix_match"
                ),
                "legacy_s1_benefit_smoke": (
                    "excluded_due_to_non_bit_identical_sequence_logps"
                ),
                "source_artifacts_modified": False,
                "accepted_response_requirement": (
                    "s1_hit_must_be_a_sealed_source_terminal_attempt"
                ),
            },
            "phases": phases,
            "partition_summary": partition_summary,
            "continuation": {
                "requested_rows": sum(
                    phase["classification_counts"]["needs_continuation"]
                    for phase in phases.values()
                ),
                "maximum_missing_attempts": sum(
                    phase["maximum_missing_attempts"] for phase in phases.values()
                ),
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
        }
    )
    output = args.output
    if getattr(args, "output_root", None):
        output = os.path.join(args.output_root, "control", "REPLAY_PLAN.json")
    if not output:
        raise ValueError("either --output or --output-root is required")
    if os.path.basename(output) != "REPLAY_PLAN.json":
        raise ValueError("replay plan must be named REPLAY_PLAN.json")
    _write_idempotent(output, payload)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_TRACE_REUSE_PLAN_VALID",
                "plan_payload_sha256": payload[OUTPUT_SEAL],
                "benefit_classification_counts": phases["benefit"][
                    "classification_counts"
                ],
                "medical_classification_counts": phases["medical"][
                    "classification_counts"
                ],
                "continuation_rows": payload["continuation"]["requested_rows"],
                "maximum_missing_attempts": payload["continuation"][
                    "maximum_missing_attempts"
                ],
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest", required=True)
    parser.add_argument("--s3-benefit-gate-generation", required=True)
    parser.add_argument("--s3-benefit-completion-generation", required=True)
    parser.add_argument("--s3-medical-gate-generation", required=True)
    parser.add_argument("--s3-medical-completion-generation", required=True)
    parser.add_argument("--s1-medical-smoke-generation", required=True)
    parser.add_argument("--output")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    if bool(args.output) == bool(args.output_root):
        parser.error("provide exactly one of --output or --output-root")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
