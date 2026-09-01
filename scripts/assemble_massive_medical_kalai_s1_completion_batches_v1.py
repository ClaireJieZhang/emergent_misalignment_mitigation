#!/usr/bin/env python3
"""CPU-only exact assembly of all seven Kalai ``s=1`` completion batches.

The assembler is deliberately unavailable until every batch has a sealed
successful result.  It combines the immutable replay rows, the already sealed
technical-gate outputs, and the seven batch continuations into exactly 360
MASSIVE and 80 medical rows.  It has no model-loading, Slurm, or judge path.
"""

from __future__ import annotations

import argparse
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
    "_kalai_s1_completion_batch_planner_for_assembly",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_completion_batches_v1.py",
)
runtime = _load_module(
    "_kalai_s1_completion_batch_runtime_for_assembly",
    SCRIPT_DIR / "sample_massive_medical_kalai_s1_completion_batch_v1.py",
)


PROTOCOL_ID = planner.PROTOCOL_ID
SOURCE_PROTOCOL_ID = planner.CONTROLLER_PROTOCOL_ID
METHOD_ID = planner.METHOD_ID
SEAL_FIELD = planner.SEAL_FIELD
PHASES = planner.PHASES
EXPECTED_REQUESTS = {"benefit": 360, "medical": 80}


def _load_json(path, description):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{description} is absent or unsafe: {path}")
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def _binding(path, payload):
    return planner.binding(os.fspath(Path(path).resolve()), payload)


def _require_no_symlink_below(root, path, description):
    root = Path(os.path.abspath(os.fspath(root)))
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} escapes the output namespace") from error
    if root.is_symlink():
        raise ValueError(f"{description} output root is a symlink")
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{description} contains a symlink: {current}")
    return path


def _verify_bound_json(expected, description):
    if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
        raise ValueError(f"{description} binding differs")
    payload = _load_json(expected["path"], description)
    planner.verify_seal(payload, description)
    if _binding(expected["path"], payload) != expected:
        raise ValueError(f"{description} binding changed")
    return payload


def _key(row):
    return row["question_id"], row["sample_index"]


def _profiles(source_body):
    manifest_path = runtime.source._binding_path(
        source_body, "source_protocol_manifest"
    )
    protocol = runtime.source.legacy.primary.load_protocol_manifest(
        manifest_path, audit_models=False
    )
    return runtime.source._profiles_and_records(protocol)


def _audit_final_sample(row, sample, phase, profile):
    continuation = sample if row["disposition"] == "unresolved" else None
    expected = planner.source.assemble_s1_sample(row, continuation)
    if expected != sample:
        raise ValueError(f"{phase} final sample differs from its replay row")
    runtime.source.legacy._audit_sample(
        sample, runtime.source._request_fields(row), phase, profile
    )
    return sample


def _load_gate_samples(plan_body, source_rows, phase, profile):
    gate_binding = plan_body["source_bindings"]["source_technical_gate_result"]
    gate_result = _verify_bound_json(
        gate_binding, "source technical-gate result"
    )
    gate_body = planner.verify_seal(
        gate_result, "source technical-gate result"
    )
    assembled_binding = (
        gate_body.get("phase_outputs", {})
        .get(phase, {})
        .get("assembled_generation")
    )
    assembled = _verify_bound_json(
        assembled_binding, f"source assembled technical-gate {phase}"
    )
    meta = assembled.get("meta")
    rows = [row for row in source_rows if row["partition"] == "technical_gate"]
    samples = assembled.get("samples")
    expected_keys = [_key(row) for row in rows]
    if (
        not isinstance(meta, dict)
        or meta.get("protocol_id") != SOURCE_PROTOCOL_ID
        or meta.get("method_id") != METHOD_ID
        or meta.get("stage") != "technical_gate"
        or meta.get("phase") != phase
        or meta.get("requested_n") != len(rows)
        or not isinstance(samples, list)
        or [_key(sample) for sample in samples] != expected_keys
    ):
        raise ValueError(f"source assembled technical-gate {phase} differs")
    result = {}
    for row, sample in zip(rows, samples):
        _audit_final_sample(row, sample, phase, profile)
        if _key(row) in result:
            raise ValueError(f"duplicate source technical-gate {phase} row")
        result[_key(row)] = sample
    if assembled.get("summary") != runtime.source.legacy.summarize_samples(samples):
        raise ValueError(f"source assembled technical-gate {phase} summary differs")
    return result, assembled_binding, gate_binding


def _load_batch_result(output_root, plan_payload, plan_body, batch_index):
    batch_name = f"batch_{batch_index:02d}"
    control = Path(output_root).resolve() / "control" / "batches" / batch_name
    _require_no_symlink_below(output_root, control, f"{batch_name} control")
    if os.path.lexists(control / "STOPPED"):
        raise ValueError(f"{batch_name} is stopped and cannot be assembled")
    result_path = control / "RESULT.json"
    result = _load_json(result_path, f"{batch_name} result")
    body = planner.verify_seal(result, f"{batch_name} result")
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("batch_index") != batch_index
        or body.get("batch_id") != batch_name
        or body.get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_COMPLETE"
        or body.get("batch_valid") is not True
        or body.get("automatic_next_batch_authorized") is not False
        or body.get("judge_authorized") is not False
        or body.get("external_api_calls") != 0
        or body.get("batch_plan")
        != _binding(
            Path(output_root).resolve()
            / "control"
            / "COMPLETION_BATCH_PLAN.json",
            plan_payload,
        )
    ):
        raise ValueError(f"{batch_name} result identity differs")

    # Rebuild every shard/generation/timing binding before using a sample.
    audit = runtime._audit_batch(
        output_root, plan_payload, plan_body, batch_index
    )
    batches = plan_body["batches"]
    item = batches[batch_index - 1]
    samples_by_phase = {}
    for phase in PHASES:
        generation_binding = body.get("phase_outputs", {}).get(phase, {}).get(
            "generation"
        )
        generation = _verify_bound_json(
            generation_binding, f"{batch_name} {phase} generation"
        )
        if (
            generation[SEAL_FIELD]
            != audit["phases"][phase]["generation_payload_sha256"]
            or generation.get("summary")
            != audit["phases"][phase]["summary"]
        ):
            raise ValueError(f"{batch_name} {phase} result binding differs")
        rows = item["rows"][phase]
        samples = generation.get("samples")
        if (
            not isinstance(samples, list)
            or [_key(sample) for sample in samples] != [_key(row) for row in rows]
        ):
            raise ValueError(f"{batch_name} {phase} sample order differs")
        samples_by_phase[phase] = (rows, samples)
    return result, _binding(result_path, result), samples_by_phase


def _write_or_audit(output_root, path, payload, description):
    path = _require_no_symlink_below(output_root, path, description)
    if os.path.lexists(path):
        observed = _load_json(path, description)
        planner.verify_seal(observed, description)
        if observed != payload:
            raise ValueError(f"existing {description} differs")
    else:
        runtime.source._write_new_json(path, payload, description)


def assemble(args):
    raw_output_root = Path(args.output_root)
    if raw_output_root.is_symlink():
        raise ValueError("completion-batch output namespace is a symlink")
    output_root = Path(os.path.abspath(os.fspath(raw_output_root)))
    if output_root.name != "massive_medical_kalai_s1_completion_batches_v1":
        raise ValueError("completion-batch output namespace differs")
    plan_path = output_root / "control" / "COMPLETION_BATCH_PLAN.json"
    plan_payload, plan_body = runtime._load_batch_plan(plan_path)
    source_payload, source_body = runtime._source_replay(plan_body)
    source_rows = runtime.source._plan_rows(source_body)
    phase_data = _profiles(source_body)

    batch_samples = {phase: {} for phase in PHASES}
    batch_result_bindings = []
    for batch_index in range(1, planner.BATCH_COUNT + 1):
        _result, result_binding, samples_by_phase = _load_batch_result(
            output_root, plan_payload, plan_body, batch_index
        )
        batch_result_bindings.append(result_binding)
        for phase, (rows, samples) in samples_by_phase.items():
            for row, sample in zip(rows, samples):
                key = _key(row)
                if key in batch_samples[phase]:
                    raise ValueError(f"duplicate {phase} row across batches")
                batch_samples[phase][key] = sample

    assembled_bindings = {}
    gate_result_binding = None
    for phase in PHASES:
        rows = source_rows[phase]
        profile = phase_data[phase]["profile"]
        gate_samples, gate_component, gate_binding = _load_gate_samples(
            plan_body, rows, phase, profile
        )
        if gate_result_binding is None:
            gate_result_binding = gate_binding
        elif gate_result_binding != gate_binding:
            raise ValueError("phase gate-result bindings differ")

        final_samples = []
        used_batch_keys = set()
        for row in rows:
            key = _key(row)
            if row["partition"] == "technical_gate":
                sample = gate_samples.get(key)
            elif row["disposition"] != "unresolved":
                sample = planner.source.assemble_s1_sample(row)
            else:
                sample = batch_samples[phase].get(key)
                used_batch_keys.add(key)
            if sample is None:
                raise ValueError(f"missing final {phase} sample: {key}")
            _audit_final_sample(row, sample, phase, profile)
            final_samples.append(sample)

        expected_batch_keys = {
            _key(row)
            for row in rows
            if row["partition"] == "completion"
            and row["disposition"] == "unresolved"
        }
        if (
            used_batch_keys != expected_batch_keys
            or set(batch_samples[phase]) != expected_batch_keys
            or len(final_samples) != EXPECTED_REQUESTS[phase]
            or len({_key(sample) for sample in final_samples})
            != EXPECTED_REQUESTS[phase]
        ):
            raise ValueError(f"full {phase} assembly is not an exact union")

        payload = planner.seal(
            {
                "meta": {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "source_protocol_id": SOURCE_PROTOCOL_ID,
                    "method_id": METHOD_ID,
                    "stage": "assembled_full",
                    "phase": phase,
                    "requested_n": EXPECTED_REQUESTS[phase],
                    "request_keys": [[key[0], key[1]] for key in map(_key, rows)],
                    "completion_batch_plan": _binding(plan_path, plan_payload),
                    "source_replay_plan": _binding(
                        plan_body["source_bindings"]["source_replay_plan"]["path"],
                        source_payload,
                    ),
                    "source_technical_gate_result": gate_result_binding,
                    "source_technical_gate_assembled_generation": gate_component,
                    "completion_batch_results": batch_result_bindings,
                    "reused_terminal_rows": sum(
                        row["disposition"] != "unresolved" for row in rows
                    ),
                    "continued_technical_gate_rows": sum(
                        row["partition"] == "technical_gate"
                        and row["disposition"] == "unresolved"
                        for row in rows
                    ),
                    "continued_completion_rows": len(expected_batch_keys),
                    "stored_prefix_candidates_regenerated": False,
                    "abstention_policy": (
                        "abstention_is_a_coverage_outcome_not_a_safe_judgment"
                    ),
                    "external_api_calls": 0,
                },
                "summary": runtime.source.legacy.summarize_samples(final_samples),
                "samples": final_samples,
            }
        )
        output_path = output_root / "assembled" / "full" / phase / "generation.json"
        _write_or_audit(
            output_root, output_path, payload, f"assembled full {phase}"
        )
        assembled_bindings[phase] = _binding(output_path, payload)

    manifest = planner.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "method_id": METHOD_ID,
            "status": "MASSIVE_MEDICAL_KALAI_S1_FULL_ASSEMBLY_AUDITED",
            "completion_batch_plan": _binding(plan_path, plan_payload),
            "source_technical_gate_result": gate_result_binding,
            "completion_batch_results": batch_result_bindings,
            "assembled": assembled_bindings,
            "benefit_requested_n": 360,
            "medical_requested_n": 80,
            "stored_prefix_candidates_regenerated": False,
            "judge_authorized": False,
            "gpu_jobs_submitted": 0,
            "external_api_calls": 0,
        }
    )
    manifest_path = output_root / "control" / "FINAL_ASSEMBLY.json"
    _write_or_audit(
        output_root, manifest_path, manifest, "final assembly manifest"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "final_assembly_payload_sha256": manifest[SEAL_FIELD],
                "benefit_requested_n": 360,
                "medical_requested_n": 80,
                "judge_authorized": False,
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return manifest


def self_test():
    keys = [("q0", 0), ("q1", 0), ("q2", 1)]
    self_check = {_key({"question_id": q, "sample_index": i}) for q, i in keys}
    assert self_check == set(keys)
    assert EXPECTED_REQUESTS == {"benefit": 360, "medical": 80}
    assert planner.BATCH_COUNT == 7
    print("KALAI_S1_COMPLETION_BATCH_FINAL_ASSEMBLY_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.output_root:
        parser.error("--output-root is required")
    assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
