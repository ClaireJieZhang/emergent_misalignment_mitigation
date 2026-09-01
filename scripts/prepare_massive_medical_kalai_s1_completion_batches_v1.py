#!/usr/bin/env python3
"""Build the immutable seven-batch Kalai ``s=1`` completion plan.

This is a CPU-only planner.  It binds the already sealed trace-reuse plan and
completed technical-gate result, derives phase-level seconds-per-attempt rates
from the sealed gate artifacts, and assigns the 174 unresolved completion rows
to seven batches with deterministic longest-processing-time-first (LPT)
scheduling.  It cannot submit a job, load a model, or call an external API.

LPT uses only each frozen row's ``max_new_attempts`` multiplied by the sealed
phase rate.  It does not inspect a completion outcome.  All scheduling
arithmetic is exact rational arithmetic; JSON floating-point rendering cannot
change an assignment.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = _load_module(
    "_kalai_s1_trace_reuse_source_for_completion_batches",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_trace_reuse_v1.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_completion_batches_v1"
CONTROLLER_PROTOCOL_ID = source.PROTOCOL_ID
METHOD_ID = source.METHOD_ID
SEAL_FIELD = source.OUTPUT_SEAL
BATCH_COUNT = 7
BATCH_IDS = tuple(f"batch_{index:02d}" for index in range(1, BATCH_COUNT + 1))
PHASES = ("benefit", "medical")
EXPECTED_ROWS_BY_PHASE = {"benefit": 115, "medical": 59}
EXPECTED_ROWS = 174
EXPECTED_MAX_NEW_ATTEMPTS = 2947

# Accounting context is frozen here but creates no authority.  The scheduler
# elapsed time is deliberately used in the conservative ledger, while the
# evaluator's independently sealed internal elapsed time remains visible.
H200_HOURLY_USD = Decimal("0.90")
SEALED_INTERNAL_GATE_ELAPSED_SECONDS = 1303
SEALED_INTERNAL_GATE_COST_USD = Decimal("0.32575")
SCHEDULER_GATE_ELAPSED_SECONDS = 1308
SCHEDULER_GATE_COST_USD = Decimal("0.327")
KNOWN_PROGRAM_ACTUAL_BEFORE_GATE_USD = Decimal("5.03884025")
RETAINED_CONSERVATIVE_EXPOSURE_USD = Decimal("0.756144")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("6.12198425")
PLANNED_BATCH_MINUTES = 60
PLANNED_BATCH_SECONDS = PLANNED_BATCH_MINUTES * 60
PLANNED_BATCH_CAP_USD = Decimal("0.900")
PLANNED_SEVEN_BATCH_CAP_USD = Decimal("6.300")
PLANNED_MAXIMUM_USD = Decimal("12.42198425")
PROPOSED_PROGRAM_CEILING_USD = Decimal("12.50")

# Exact completed-source pins.  Protocol identity and counts alone are not
# sufficient: another internally valid run with the same IDs must fail closed.
EXPECTED_SOURCE_REPOSITORY_COMMIT = (
    "e83d59e3162250d7c3f555dc32f13de6d1ee9669"
)
EXPECTED_SOURCE_CPU_STAGE_SHA256 = (
    "caabdbbac1600fac21453ce2e6e5386778c0e165eec5a9853983dd3684d329fd"
)
EXPECTED_SOURCE_REPLAY_PLAN_SHA256 = (
    "cd051c5d1a6dc13de412d000c28164febf2d274716639d4d58dcb20db4bfd4ff"
)
EXPECTED_TECHNICAL_GATE_RESULT_SHA256 = (
    "f25aba6c69235bf5be88bbfb16acecb62ffed63438090f1d8ee12404b13e95b6"
)
EXPECTED_GATE_COMBINED_TIMING_SHA256 = (
    "eb484ac45c2e16801f66e369dd21b469e40f097ef55570f6427421504c9cabf6"
)
EXPECTED_GATE_GENERATION_SHA256 = {
    "benefit": "ca7114ff34e1fa02b1f6a30fd5208130e333cb8e9b31c5a3d66495a44a570785",
    "medical": "31ce1287239a3933bd5766a7c79951405278fe4114e66c9b31d7db00a5ee9138",
}
EXPECTED_GATE_TIMING_SHA256 = {
    "benefit": "ef4deb63173aa51324fdee1cadc5e157fa17d9b027dabeb1845c540d21c37e1d",
    "medical": "cc59889ccf2533dcbc623e4c89e8fed168b4643fce9a9ceaeb53fe64db409265",
}


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def seal(body):
    payload = dict(body)
    payload.pop(SEAL_FIELD, None)
    payload[SEAL_FIELD] = sha256_bytes(canonical_bytes(payload))
    return payload


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} has an invalid {SEAL_FIELD}")
    return body


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json(path, description):
    path = os.path.abspath(path)
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def binding(path, payload):
    path = os.path.abspath(path)
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[SEAL_FIELD],
    }


def _verify_binding(expected, path, payload, description):
    if not isinstance(expected, dict) or set(expected) != {
        "path",
        "size_bytes",
        "file_sha256",
        "payload_sha256",
    }:
        raise ValueError(f"{description} binding schema differs")
    if expected != binding(path, payload):
        raise ValueError(f"{description} binding differs")


def _write_idempotent(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    if os.path.lexists(path):
        observed = _load_json(path, "existing completion batch plan")
        verify_seal(observed, "existing completion batch plan")
        if observed != payload:
            raise ValueError("existing completion batch plan differs")
        return "AUDITED"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "CREATED"


def _as_nonnegative_number(value, description, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} is not numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{description} is not finite")
    if positive and value <= 0:
        raise ValueError(f"{description} is not positive")
    if not positive and value < 0:
        raise ValueError(f"{description} is negative")
    return value


def _fraction_from_json_number(value, description):
    _as_nonnegative_number(value, description, positive=True)
    return Fraction(Decimal(str(value)))


def _ratio(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_ratio(value, description):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{description} ratio schema differs")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError(f"{description} ratio differs")
    return Fraction(numerator, denominator)


def _load_source_replay(path, *, audit_sources):
    payload, body = source.load_and_verify_plan(path, audit_sources=audit_sources)
    if (
        payload.get(SEAL_FIELD) != EXPECTED_SOURCE_REPLAY_PLAN_SHA256
        or body.get("protocol_id") != CONTROLLER_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
    ):
        raise ValueError("source replay-plan identity differs")
    return payload, body


def _gate_generation_binding(gate_body, phase):
    outputs = gate_body.get("phase_outputs")
    phase_output = outputs.get(phase) if isinstance(outputs, dict) else None
    value = (
        phase_output.get("continuation_generation")
        if isinstance(phase_output, dict)
        else None
    )
    if not isinstance(value, dict):
        raise ValueError(f"technical-gate {phase} generation binding is absent")
    return value


def _validate_gate_result(payload, replay_path, replay_payload):
    body = verify_seal(payload, "source technical-gate result")
    if (
        payload.get(SEAL_FIELD) != EXPECTED_TECHNICAL_GATE_RESULT_SHA256
        or body.get("protocol_id") != CONTROLLER_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "technical_gate"
        or body.get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_TECHNICAL_GATE_COMPLETE"
        or body.get("technical_gate_valid") is not True
        or body.get("coverage_threshold") is not None
        or body.get("completion_eligible") is not True
        or body.get("completion_authorized") is not False
        or body.get("restart_or_resume_authorized") is not False
        or body.get("retry_authorized") is not False
        or body.get("external_api_calls") != 0
    ):
        raise ValueError("source technical-gate result identity differs")
    _verify_binding(
        body.get("replay_plan"),
        replay_path,
        replay_payload,
        "technical-gate replay plan",
    )
    timing = body.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("technical-gate timing is absent")
    if (
        timing.get("elapsed_seconds") != SEALED_INTERNAL_GATE_ELAPSED_SECONDS
        or Decimal(str(timing.get("actual_estimated_cost_usd")))
        != SEALED_INTERNAL_GATE_COST_USD
    ):
        raise ValueError("technical-gate sealed internal accounting differs")
    return body


def _validate_source_cpu_stage(path, replay_path, replay_payload):
    payload = _load_json(path, "source CPU stage")
    body = source.verify_seal(payload, "source CPU stage")
    if (
        payload.get(SEAL_FIELD) != EXPECTED_SOURCE_CPU_STAGE_SHA256
        or body.get("protocol_id") != CONTROLLER_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or body.get("repository_commit") != EXPECTED_SOURCE_REPOSITORY_COMMIT
        or body.get("external_api_calls") != 0
        or body.get("external_api_authorized") is not False
        or body.get("gpu_jobs_submitted") != 0
        or body.get("gpu_authorized") is not False
    ):
        raise ValueError("source CPU-stage identity differs")
    _verify_binding(
        body.get("replay_plan"),
        replay_path,
        replay_payload,
        "source CPU-stage replay plan",
    )
    return payload, body


def _validate_gate_combined_timing(path, replay_payload, gate_body):
    payload = _load_json(path, "gate combined timing")
    body = source.verify_seal(payload, "gate combined timing")
    if (
        payload.get(SEAL_FIELD) != EXPECTED_GATE_COMBINED_TIMING_SHA256
        or body.get("protocol_id") != CONTROLLER_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "technical_gate"
        or body.get("replay_plan_payload_sha256") != replay_payload[SEAL_FIELD]
        or body.get("shared_reference_model_loads") != 1
        or body.get("stored_prefix_candidates_regenerated") is not False
        or body.get("restart_or_resume_authorized") is not False
        or body.get("external_api_calls") != 0
    ):
        raise ValueError("gate combined-timing identity differs")
    _verify_binding(
        gate_body.get("combined_timing"),
        path,
        payload,
        "technical-gate combined timing",
    )
    return payload, body


def _validate_phase_evidence(
    *,
    phase,
    replay_payload,
    gate_body,
    generation_path,
    timing_path,
):
    generation = _load_json(generation_path, f"gate {phase} generation")
    source.verify_seal(generation, f"gate {phase} generation")
    if generation.get(SEAL_FIELD) != EXPECTED_GATE_GENERATION_SHA256[phase]:
        raise ValueError(f"gate {phase} generation is not the frozen source")
    _verify_binding(
        _gate_generation_binding(gate_body, phase),
        generation_path,
        generation,
        f"gate {phase} generation",
    )
    if set(generation) != {
        "meta",
        "summary",
        "new_attempts_generated",
        "samples",
        SEAL_FIELD,
    }:
        raise ValueError(f"gate {phase} generation schema differs")
    meta = generation.get("meta")
    if (
        not isinstance(meta, dict)
        or meta.get("protocol_id") != CONTROLLER_PROTOCOL_ID
        or meta.get("method_id") != METHOD_ID
        or meta.get("stage") != "technical_gate"
        or meta.get("phase") != phase
        or meta.get("replay_plan_payload_sha256") != replay_payload[SEAL_FIELD]
        or meta.get("stored_prefix_candidates_regenerated") is not False
        or meta.get("restart_or_resume_authorized") is not False
        or meta.get("external_api_calls") != 0
    ):
        raise ValueError(f"gate {phase} generation metadata differs")
    attempts = generation.get("new_attempts_generated")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError(f"gate {phase} new-attempt count differs")
    observed = gate_body.get("observed")
    phase_observed = observed.get(phase) if isinstance(observed, dict) else None
    if (
        not isinstance(phase_observed, dict)
        or phase_observed.get("new_attempts_generated") != attempts
    ):
        raise ValueError(f"gate {phase} observed attempt count differs")

    timing = _load_json(timing_path, f"gate {phase} timing")
    timing_body = source.verify_seal(timing, f"gate {phase} timing")
    if (
        timing.get(SEAL_FIELD) != EXPECTED_GATE_TIMING_SHA256[phase]
        or set(timing_body)
        != {
            "protocol_id",
            "method_id",
            "stage",
            "phase",
            "replay_plan_payload_sha256",
            "elapsed_seconds",
            "summary",
            "external_api_calls",
        }
        or timing_body.get("protocol_id") != CONTROLLER_PROTOCOL_ID
        or timing_body.get("method_id") != METHOD_ID
        or timing_body.get("stage") != "technical_gate"
        or timing_body.get("phase") != phase
        or timing_body.get("replay_plan_payload_sha256")
        != replay_payload[SEAL_FIELD]
        or timing_body.get("summary") != generation.get("summary")
        or timing_body.get("external_api_calls") != 0
    ):
        raise ValueError(f"gate {phase} phase timing differs")
    elapsed = _fraction_from_json_number(
        timing_body.get("elapsed_seconds"), f"gate {phase} elapsed seconds"
    )
    return {
        "generation_payload": generation,
        "timing_payload": timing,
        "elapsed_seconds": elapsed,
        "new_attempts_generated": attempts,
        "seconds_per_new_attempt": elapsed / attempts,
    }


def completion_rows(replay_body):
    phases = replay_body.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("source replay phases differ")
    rows = []
    counts = {}
    for phase in PHASES:
        phase_rows = phases[phase].get("rows")
        if not isinstance(phase_rows, list):
            raise ValueError(f"source replay {phase} rows differ")
        selected = [
            row
            for row in phase_rows
            if row.get("partition") == "completion"
            and row.get("disposition") == "unresolved"
        ]
        counts[phase] = len(selected)
        rows.extend(selected)
    if counts != EXPECTED_ROWS_BY_PHASE or len(rows) != EXPECTED_ROWS:
        raise ValueError("completion unresolved row counts differ")
    if sum(row.get("max_new_attempts", -1) for row in rows) != (
        EXPECTED_MAX_NEW_ATTEMPTS
    ):
        raise ValueError("completion maximum-new-attempt total differs")
    keys = [(row["phase"], row["question_id"], row["sample_index"]) for row in rows]
    hashes = [row.get("row_sha256") for row in rows]
    if len(set(keys)) != EXPECTED_ROWS or len(set(hashes)) != EXPECTED_ROWS:
        raise ValueError("completion rows are not unique")
    for row in rows:
        if (
            row.get("phase") not in PHASES
            or row.get("stage") != "completion"
            or row.get("classification") != "needs_continuation"
            or row.get("reusable_terminal_sample") is not None
            or isinstance(row.get("request_index"), bool)
            or not isinstance(row.get("request_index"), int)
            or isinstance(row.get("sample_index"), bool)
            or not isinstance(row.get("sample_index"), int)
            or isinstance(row.get("max_new_attempts"), bool)
            or not isinstance(row.get("max_new_attempts"), int)
            or not 1 <= row.get("max_new_attempts") <= source.MAX_ATTEMPTS - 1
        ):
            raise ValueError("completion unresolved row boundary differs")
    return rows


def deterministic_lpt(rows, phase_rates, batch_count=BATCH_COUNT):
    """Return deterministic LPT assignments using exact rational weights."""

    if batch_count != BATCH_COUNT or set(phase_rates) != set(PHASES):
        raise ValueError("LPT batch count or phase-rate mapping differs")
    if any(not isinstance(phase_rates[phase], Fraction) for phase in PHASES):
        raise ValueError("LPT phase rates must be exact fractions")
    if any(phase_rates[phase] <= 0 for phase in PHASES):
        raise ValueError("LPT phase rates must be positive")
    weighted = [
        (row, phase_rates[row["phase"]] * row["max_new_attempts"])
        for row in rows
    ]
    weighted.sort(
        key=lambda item: (
            -item[1],
            item[0]["phase"],
            item[0]["request_index"],
            item[0]["sample_index"],
        )
    )
    assigned = [[] for _ in range(batch_count)]
    loads = [Fraction(0) for _ in range(batch_count)]
    for row, work in weighted:
        batch_index = min(range(batch_count), key=lambda index: (loads[index], index))
        assigned[batch_index].append(row)
        loads[batch_index] += work
    return assigned, loads


def _batch_record(index, rows, load):
    ordered = {
        phase: sorted(
            (row for row in rows if row["phase"] == phase),
            key=lambda row: (row["request_index"], row["sample_index"]),
        )
        for phase in PHASES
    }
    return {
        "batch_index": index + 1,
        "batch_id": BATCH_IDS[index],
        "rows": ordered,
        "summary": {
            "row_count": len(rows),
            "rows_by_phase": {
                phase: len(ordered[phase]) for phase in PHASES
            },
            "maximum_new_attempts": sum(
                row["max_new_attempts"] for row in rows
            ),
            "maximum_new_attempts_by_phase": {
                phase: sum(row["max_new_attempts"] for row in ordered[phase])
                for phase in PHASES
            },
            "weighted_maximum_seconds_ratio": _ratio(load),
            "row_sha256s_by_phase": {
                phase: [row["row_sha256"] for row in ordered[phase]]
                for phase in PHASES
            },
        },
    }


def build_plan(
    *,
    replay_path,
    replay_payload,
    replay_body,
    source_cpu_stage_path,
    source_cpu_stage_payload,
    gate_result_path,
    gate_result_payload,
    gate_result_body,
    gate_combined_timing_path,
    gate_combined_timing_payload,
    phase_evidence,
):
    rows = completion_rows(replay_body)
    rates = {
        phase: phase_evidence[phase]["seconds_per_new_attempt"]
        for phase in PHASES
    }
    assigned, loads = deterministic_lpt(rows, rates)
    gate_combined_body = source.verify_seal(
        gate_combined_timing_payload, "gate combined timing"
    )
    shared_setup_seconds = _fraction_from_json_number(
        gate_combined_body.get("setup_elapsed_seconds"),
        "gate shared setup elapsed seconds",
    )
    projected_seconds = [shared_setup_seconds + load for load in loads]
    projected_maximum = max(projected_seconds)
    if projected_maximum >= PLANNED_BATCH_SECONDS:
        raise ValueError("LPT plan does not fit the one-hour batch envelope")
    batches = [
        _batch_record(index, assigned[index], loads[index])
        for index in range(BATCH_COUNT)
    ]
    flattened = [
        row
        for batch in batches
        for phase in PHASES
        for row in batch["rows"][phase]
    ]
    expected_hashes = {row["row_sha256"] for row in rows}
    observed_hashes = [row["row_sha256"] for row in flattened]
    if (
        len(flattened) != EXPECTED_ROWS
        or len(set(observed_hashes)) != EXPECTED_ROWS
        or set(observed_hashes) != expected_hashes
        or sum(
            batch["summary"]["maximum_new_attempts"] for batch in batches
        )
        != EXPECTED_MAX_NEW_ATTEMPTS
    ):
        raise ValueError("LPT plan does not partition completion rows exactly once")

    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "controller_protocol_id": CONTROLLER_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "source_bindings": {
            "source_replay_plan": binding(replay_path, replay_payload),
            "source_cpu_stage": binding(
                source_cpu_stage_path, source_cpu_stage_payload
            ),
            "source_technical_gate_result": binding(
                gate_result_path, gate_result_payload
            ),
            "gate_combined_timing": binding(
                gate_combined_timing_path, gate_combined_timing_payload
            ),
            "gate_phase_generations": {
                phase: binding(
                    phase_evidence[phase]["generation_path"],
                    phase_evidence[phase]["generation_payload"],
                )
                for phase in PHASES
            },
            "gate_phase_timings": {
                phase: binding(
                    phase_evidence[phase]["timing_path"],
                    phase_evidence[phase]["timing_payload"],
                )
                for phase in PHASES
            },
        },
        "assignment_policy": {
            "name": "deterministic_lpt_v1",
            "batch_count": BATCH_COUNT,
            "row_weight": (
                "frozen_max_new_attempts_times_sealed_gate_phase_seconds_per_attempt"
            ),
            "descending_item_tie_break": [
                "phase",
                "request_index",
                "sample_index",
            ],
            "least_loaded_batch_tie_break": "lowest_batch_index",
            "arithmetic": "exact_rational",
            "completion_outcomes_used": False,
        },
        "phase_rates": {
            phase: {
                "sealed_gate_elapsed_seconds_ratio": _ratio(
                    phase_evidence[phase]["elapsed_seconds"]
                ),
                "sealed_gate_new_attempts_generated": phase_evidence[phase][
                    "new_attempts_generated"
                ],
                "seconds_per_new_attempt_ratio": _ratio(rates[phase]),
            }
            for phase in PHASES
        },
        "completion_scope": {
            "source_partition": "completion",
            "source_disposition": "unresolved",
            "row_count": EXPECTED_ROWS,
            "rows_by_phase": EXPECTED_ROWS_BY_PHASE,
            "maximum_new_attempts": EXPECTED_MAX_NEW_ATTEMPTS,
            "all_rows_assigned_exactly_once": True,
            "overlap_count": 0,
        },
        "one_hour_feasibility": {
            "shared_setup_seconds_ratio": _ratio(shared_setup_seconds),
            "projected_seconds_by_batch_ratio": [
                _ratio(value) for value in projected_seconds
            ],
            "projected_maximum_seconds_ratio": _ratio(projected_maximum),
            "minimum_projected_margin_seconds_ratio": _ratio(
                Fraction(PLANNED_BATCH_SECONDS) - projected_maximum
            ),
            "strict_batch_limit_seconds": PLANNED_BATCH_SECONDS,
            "all_projected_batches_strictly_below_limit": True,
        },
        "batches": batches,
        "accounting_context_not_authorization": {
            "h200_hourly_usd": float(H200_HOURLY_USD),
            "sealed_internal_gate_elapsed_seconds": (
                SEALED_INTERNAL_GATE_ELAPSED_SECONDS
            ),
            "sealed_internal_gate_cost_usd": float(
                SEALED_INTERNAL_GATE_COST_USD
            ),
            "scheduler_conservative_gate_elapsed_seconds": (
                SCHEDULER_GATE_ELAPSED_SECONDS
            ),
            "scheduler_conservative_gate_cost_usd": float(
                SCHEDULER_GATE_COST_USD
            ),
            "known_program_actual_before_gate_usd": float(
                KNOWN_PROGRAM_ACTUAL_BEFORE_GATE_USD
            ),
            "retained_conservative_exposure_usd": float(
                RETAINED_CONSERVATIVE_EXPOSURE_USD
            ),
            "current_conservative_exposure_usd": float(
                CURRENT_CONSERVATIVE_EXPOSURE_USD
            ),
            "planned_batch_minutes_each": PLANNED_BATCH_MINUTES,
            "planned_batch_cap_usd_each": float(PLANNED_BATCH_CAP_USD),
            "planned_seven_batch_cap_usd": float(PLANNED_SEVEN_BATCH_CAP_USD),
            "planned_maximum_if_all_batches_later_authorized_usd": float(
                PLANNED_MAXIMUM_USD
            ),
            "proposed_program_ceiling_usd": float(
                PROPOSED_PROGRAM_CEILING_USD
            ),
        },
        "authority": {
            "gpu_jobs_authorized": 0,
            "gpu_jobs_submitted": 0,
            "external_api_calls_authorized": 0,
            "external_api_calls": 0,
            "batch_authorizations_created": 0,
            "restart_or_resume_authorized": False,
            "retry_or_replacement_authorized": False,
            "requeue_authorized": False,
            "automatic_next_batch_authorized": False,
        },
    }
    # The gate body is an explicit input so callers cannot substitute a result
    # after `_validate_gate_result`; keep this assertion close to sealing.
    if gate_result_body.get("completion_eligible") is not True:
        raise ValueError("technical gate does not permit completion planning")
    return seal(body)


def _source_paths_from_plan_body(body):
    bindings = body.get("source_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("completion batch plan source bindings differ")
    expected = {
        "source_replay_plan",
        "source_cpu_stage",
        "source_technical_gate_result",
        "gate_combined_timing",
        "gate_phase_generations",
        "gate_phase_timings",
    }
    if set(bindings) != expected:
        raise ValueError("completion batch plan binding set differs")
    result = {
        "replay": bindings["source_replay_plan"]["path"],
        "source_cpu_stage": bindings["source_cpu_stage"]["path"],
        "gate": bindings["source_technical_gate_result"]["path"],
        "combined_timing": bindings["gate_combined_timing"]["path"],
    }
    for phase in PHASES:
        result[f"{phase}_generation"] = bindings["gate_phase_generations"][phase][
            "path"
        ]
        result[f"{phase}_timing"] = bindings["gate_phase_timings"][phase]["path"]
    return result


def build_from_files(
    *,
    source_replay_plan,
    source_cpu_stage,
    technical_gate_result,
    gate_combined_timing,
    gate_benefit_generation,
    gate_benefit_timing,
    gate_medical_generation,
    gate_medical_timing,
    audit_sources=True,
):
    replay_payload, replay_body = _load_source_replay(
        source_replay_plan, audit_sources=audit_sources
    )
    gate_payload = _load_json(
        technical_gate_result, "source technical-gate result"
    )
    gate_body = _validate_gate_result(
        gate_payload, source_replay_plan, replay_payload
    )
    source_cpu_stage_payload, _ = _validate_source_cpu_stage(
        source_cpu_stage, source_replay_plan, replay_payload
    )
    combined_timing_payload, _ = _validate_gate_combined_timing(
        gate_combined_timing, replay_payload, gate_body
    )
    paths = {
        "benefit": {
            "generation": gate_benefit_generation,
            "timing": gate_benefit_timing,
        },
        "medical": {
            "generation": gate_medical_generation,
            "timing": gate_medical_timing,
        },
    }
    evidence = {}
    for phase in PHASES:
        evidence[phase] = _validate_phase_evidence(
            phase=phase,
            replay_payload=replay_payload,
            gate_body=gate_body,
            generation_path=paths[phase]["generation"],
            timing_path=paths[phase]["timing"],
        )
        evidence[phase]["generation_path"] = os.path.abspath(
            paths[phase]["generation"]
        )
        evidence[phase]["timing_path"] = os.path.abspath(paths[phase]["timing"])
    return build_plan(
        replay_path=source_replay_plan,
        replay_payload=replay_payload,
        replay_body=replay_body,
        source_cpu_stage_path=source_cpu_stage,
        source_cpu_stage_payload=source_cpu_stage_payload,
        gate_result_path=technical_gate_result,
        gate_result_payload=gate_payload,
        gate_result_body=gate_body,
        gate_combined_timing_path=gate_combined_timing,
        gate_combined_timing_payload=combined_timing_payload,
        phase_evidence=evidence,
    )


def load_and_verify_plan(path, audit_sources=True):
    """Load, validate, and optionally reconstruct a completion batch plan."""

    if os.path.basename(path) != "COMPLETION_BATCH_PLAN.json":
        raise ValueError("completion batch plan filename differs")
    payload = _load_json(path, "completion batch plan")
    body = verify_seal(payload, "completion batch plan")
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("controller_protocol_id") != CONTROLLER_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("completion_scope")
        != {
            "source_partition": "completion",
            "source_disposition": "unresolved",
            "row_count": EXPECTED_ROWS,
            "rows_by_phase": EXPECTED_ROWS_BY_PHASE,
            "maximum_new_attempts": EXPECTED_MAX_NEW_ATTEMPTS,
            "all_rows_assigned_exactly_once": True,
            "overlap_count": 0,
        }
        or body.get("authority", {}).get("gpu_jobs_authorized") != 0
        or body.get("authority", {}).get("external_api_calls_authorized") != 0
    ):
        raise ValueError("completion batch plan identity differs")
    batches = body.get("batches")
    if (
        not isinstance(batches, list)
        or len(batches) != BATCH_COUNT
        or [batch.get("batch_id") for batch in batches] != list(BATCH_IDS)
        or [batch.get("batch_index") for batch in batches]
        != list(range(1, BATCH_COUNT + 1))
    ):
        raise ValueError("completion batch inventory differs")
    flattened = []
    for batch in batches:
        rows_by_phase = batch.get("rows")
        if not isinstance(rows_by_phase, dict) or list(rows_by_phase) != list(PHASES):
            raise ValueError("completion batch phase mapping differs")
        flattened.extend(rows_by_phase["benefit"])
        flattened.extend(rows_by_phase["medical"])
        _fraction_from_ratio(
            batch.get("summary", {}).get("weighted_maximum_seconds_ratio"),
            "batch weighted maximum seconds",
        )
    if (
        len(flattened) != EXPECTED_ROWS
        or len({row.get("row_sha256") for row in flattened}) != EXPECTED_ROWS
        or {
            phase: sum(row.get("phase") == phase for row in flattened)
            for phase in PHASES
        }
        != EXPECTED_ROWS_BY_PHASE
        or sum(row.get("max_new_attempts", -1) for row in flattened)
        != EXPECTED_MAX_NEW_ATTEMPTS
    ):
        raise ValueError("completion batch rows do not form the exact scope")
    for row in flattened:
        source._verify_row(row)
        if (
            row.get("stage") != "completion"
            or row.get("partition") != "completion"
            or row.get("disposition") != "unresolved"
        ):
            raise ValueError("completion batch includes a non-scope row")
    rates = {}
    for phase in PHASES:
        rates[phase] = _fraction_from_ratio(
            body.get("phase_rates", {})
            .get(phase, {})
            .get("seconds_per_new_attempt_ratio"),
            f"{phase} phase rate",
        )
    rebuilt_assignments, rebuilt_loads = deterministic_lpt(flattened, rates)
    rebuilt_batches = [
        _batch_record(index, rebuilt_assignments[index], rebuilt_loads[index])
        for index in range(BATCH_COUNT)
    ]
    if rebuilt_batches != batches:
        raise ValueError("completion batch LPT assignment differs")
    feasibility = body.get("one_hour_feasibility")
    setup = _fraction_from_ratio(
        feasibility.get("shared_setup_seconds_ratio")
        if isinstance(feasibility, dict)
        else None,
        "shared setup seconds",
    )
    projected = [setup + load for load in rebuilt_loads]
    maximum = max(projected)
    if feasibility != {
        "shared_setup_seconds_ratio": _ratio(setup),
        "projected_seconds_by_batch_ratio": [
            _ratio(value) for value in projected
        ],
        "projected_maximum_seconds_ratio": _ratio(maximum),
        "minimum_projected_margin_seconds_ratio": _ratio(
            Fraction(PLANNED_BATCH_SECONDS) - maximum
        ),
        "strict_batch_limit_seconds": PLANNED_BATCH_SECONDS,
        "all_projected_batches_strictly_below_limit": True,
    } or maximum >= PLANNED_BATCH_SECONDS:
        raise ValueError("completion batch one-hour feasibility differs")
    if audit_sources:
        paths = _source_paths_from_plan_body(body)
        rebuilt = build_from_files(
            source_replay_plan=paths["replay"],
            source_cpu_stage=paths["source_cpu_stage"],
            technical_gate_result=paths["gate"],
            gate_combined_timing=paths["combined_timing"],
            gate_benefit_generation=paths["benefit_generation"],
            gate_benefit_timing=paths["benefit_timing"],
            gate_medical_generation=paths["medical_generation"],
            gate_medical_timing=paths["medical_timing"],
            audit_sources=True,
        )
        if rebuilt != payload:
            raise ValueError("completion batch plan differs from bound sources")
    return payload, body


def prepare(args):
    payload = build_from_files(
        source_replay_plan=args.source_replay_plan,
        source_cpu_stage=args.source_cpu_stage,
        technical_gate_result=args.technical_gate_result,
        gate_combined_timing=args.gate_combined_timing,
        gate_benefit_generation=args.gate_benefit_generation,
        gate_benefit_timing=args.gate_benefit_timing,
        gate_medical_generation=args.gate_medical_generation,
        gate_medical_timing=args.gate_medical_timing,
        audit_sources=True,
    )
    output = args.output
    if getattr(args, "output_root", None):
        output = os.path.join(
            args.output_root, "control", "COMPLETION_BATCH_PLAN.json"
        )
    if not output or os.path.basename(output) != "COMPLETION_BATCH_PLAN.json":
        raise ValueError("completion batch plan output filename differs")
    action = _write_idempotent(output, payload)
    print(
        json.dumps(
            {
                "status": f"KALAI_S1_COMPLETION_BATCH_PLAN_{action}",
                "protocol_id": PROTOCOL_ID,
                "plan_payload_sha256": payload[SEAL_FIELD],
                "batch_count": BATCH_COUNT,
                "completion_rows": EXPECTED_ROWS,
                "maximum_new_attempts": EXPECTED_MAX_NEW_ATTEMPTS,
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return payload


def self_test():
    assert SCHEDULER_GATE_ELAPSED_SECONDS * H200_HOURLY_USD / Decimal(3600) == (
        SCHEDULER_GATE_COST_USD
    )
    assert CURRENT_CONSERVATIVE_EXPOSURE_USD == (
        KNOWN_PROGRAM_ACTUAL_BEFORE_GATE_USD
        + RETAINED_CONSERVATIVE_EXPOSURE_USD
        + SCHEDULER_GATE_COST_USD
    )
    assert BATCH_COUNT * PLANNED_BATCH_CAP_USD == PLANNED_SEVEN_BATCH_CAP_USD
    assert CURRENT_CONSERVATIVE_EXPOSURE_USD + PLANNED_SEVEN_BATCH_CAP_USD == (
        PLANNED_MAXIMUM_USD
    )
    assert PLANNED_MAXIMUM_USD <= PROPOSED_PROGRAM_CEILING_USD
    print("KALAI_S1_COMPLETION_BATCH_PLANNER_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-replay-plan")
    parser.add_argument("--source-cpu-stage")
    parser.add_argument("--technical-gate-result")
    parser.add_argument("--gate-combined-timing")
    parser.add_argument("--gate-benefit-generation")
    parser.add_argument("--gate-benefit-timing")
    parser.add_argument("--gate-medical-generation")
    parser.add_argument("--gate-medical-timing")
    parser.add_argument("--output")
    parser.add_argument("--output-root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    required = (
        args.source_replay_plan,
        args.source_cpu_stage,
        args.technical_gate_result,
        args.gate_combined_timing,
        args.gate_benefit_generation,
        args.gate_benefit_timing,
        args.gate_medical_generation,
        args.gate_medical_timing,
    )
    if any(value is None for value in required):
        parser.error("all sealed source paths are required")
    if bool(args.output) == bool(args.output_root):
        parser.error("exactly one of --output or --output-root is required")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
