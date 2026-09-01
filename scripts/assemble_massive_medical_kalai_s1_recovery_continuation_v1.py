#!/usr/bin/env python3
"""CPU-only assembly bridge for recovered batch 1 plus fresh batches 2--7."""

from __future__ import annotations

import argparse
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
    "_kalai_s1_recovery_continuation_manager_for_assembly",
    "manage_massive_medical_kalai_s1_recovery_continuation_v1.py",
)
evaluator = _load(
    "_kalai_s1_recovery_continuation_evaluator_for_assembly",
    "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
)
original_assembler = _load(
    "_kalai_s1_original_completion_assembler_for_recovery_continuation",
    "assemble_massive_medical_kalai_s1_completion_batches_v1.py",
)

EXPECTED_REQUESTS = {"benefit": 360, "medical": 80}


def _key(row):
    return row["question_id"], row["sample_index"]


def _load_batch_samples(
    output_root, source_plan, source_body, batch_index, protocol_id
):
    observed_protocol = manager.original_runtime.PROTOCOL_ID
    try:
        manager.original_runtime.PROTOCOL_ID = protocol_id
        audit = manager.original_runtime._audit_batch(
            output_root, source_plan, source_body, batch_index
        )
    finally:
        manager.original_runtime.PROTOCOL_ID = observed_protocol
    item = source_body["batches"][batch_index - 1]
    result = {}
    generation_bindings = {}
    for phase in manager.PHASES:
        path = Path(audit["phases"][phase]["generation"])
        generation = manager.load_json(path, f"batch {batch_index} {phase} generation")
        manager.verify_seal(generation, f"batch {batch_index} {phase} generation")
        rows = item["rows"][phase]
        samples = generation.get("samples")
        if (
            not isinstance(samples, list)
            or [_key(sample) for sample in samples] != [_key(row) for row in rows]
        ):
            raise ValueError(f"batch {batch_index} {phase} sample order differs")
        result[phase] = dict(zip(map(_key, rows), samples))
        generation_bindings[phase] = manager.binding(path, generation)
    return result, generation_bindings


def _write_or_audit(output_root, path, payload, description):
    path = original_assembler._require_no_symlink_below(
        output_root, path, description
    )
    if os.path.lexists(path):
        observed = manager.load_json(path, description)
        manager.verify_seal(observed, description)
        if observed != payload:
            raise ValueError(f"existing {description} differs")
    else:
        manager._write_new(path, payload, description)


def assemble(args):
    manager._require_no_api_key()
    output, plan, plan_body, _, _, context = manager._load_workflow(
        args.output_root, args.repo_root, audit_source=True
    )
    if os.path.lexists(output / "judge"):
        raise ValueError("assembly namespace contains judge state")
    source_plan = context["source_plan"]
    source_body = context["source_body"]
    source_output = context["source_output"]
    source_replay, source_replay_body = manager.original_runtime._source_replay(
        source_body
    )
    source_rows = manager.original_runtime.source._plan_rows(source_replay_body)
    phase_data = original_assembler._profiles(source_replay_body)

    batch_result_bindings = [
        {
            "batch_index": 1,
            "kind": "recovered_batch_1_result",
            "artifact": context["recovery_bindings"]["RECOVERED_RESULT.json"],
        }
    ]
    all_batch_samples = {phase: {} for phase in manager.PHASES}
    samples, source_generation_bindings = _load_batch_samples(
        source_output,
        source_plan,
        source_body,
        1,
        manager.SOURCE_PROTOCOL_ID,
    )
    for phase in manager.PHASES:
        all_batch_samples[phase].update(samples[phase])

    fresh_generation_bindings = {}
    for batch_index in manager.BATCH_INDICES:
        stopped = (
            output
            / "control"
            / "batches"
            / manager.batch_id(batch_index)
            / "STOPPED"
        )
        if os.path.lexists(stopped):
            raise ValueError(
                f"{manager.batch_id(batch_index)} is stopped and cannot be assembled"
            )
        result = evaluator.load_and_verify_result(
            output, args.repo_root, batch_index, audit_generation=True
        )
        path = (
            output
            / "control"
            / "batches"
            / manager.batch_id(batch_index)
            / "RESULT.json"
        )
        batch_result_bindings.append(
            {
                "batch_index": batch_index,
                "kind": "fresh_recovery_continuation_result",
                "artifact": manager.binding(path, result),
            }
        )
        samples, bindings = _load_batch_samples(
            output,
            source_plan,
            source_body,
            batch_index,
            manager.PROTOCOL_ID,
        )
        fresh_generation_bindings[manager.batch_id(batch_index)] = bindings
        for phase in manager.PHASES:
            overlap = set(all_batch_samples[phase]) & set(samples[phase])
            if overlap:
                raise ValueError(f"duplicate {phase} rows across batches")
            all_batch_samples[phase].update(samples[phase])

    assembled_bindings = {}
    gate_result_binding = None
    gate_component_bindings = {}
    for phase in manager.PHASES:
        rows = source_rows[phase]
        profile = phase_data[phase]["profile"]
        gate_samples, gate_component, gate_binding = original_assembler._load_gate_samples(
            source_body, rows, phase, profile
        )
        if gate_result_binding is None:
            gate_result_binding = gate_binding
        elif gate_result_binding != gate_binding:
            raise ValueError("technical-gate bindings differ by phase")
        gate_component_bindings[phase] = gate_component
        final_samples = []
        used_completion = set()
        for row in rows:
            key = _key(row)
            if row["partition"] == "technical_gate":
                sample = gate_samples.get(key)
            elif row["disposition"] != "unresolved":
                sample = manager.original_planner.source.assemble_s1_sample(row)
            else:
                sample = all_batch_samples[phase].get(key)
                used_completion.add(key)
            if sample is None:
                raise ValueError(f"missing assembled {phase} sample: {key}")
            original_assembler._audit_final_sample(row, sample, phase, profile)
            final_samples.append(sample)
        expected_completion = {
            _key(row)
            for row in rows
            if row["partition"] == "completion"
            and row["disposition"] == "unresolved"
        }
        if (
            used_completion != expected_completion
            or set(all_batch_samples[phase]) != expected_completion
            or len(final_samples) != EXPECTED_REQUESTS[phase]
            or len({_key(sample) for sample in final_samples})
            != EXPECTED_REQUESTS[phase]
        ):
            raise ValueError(f"assembled {phase} is not an exact union")
        payload = manager.seal(
            {
                "meta": {
                    "schema_version": 1,
                    "protocol_id": manager.PROTOCOL_ID,
                    "source_protocol_id": manager.SOURCE_PROTOCOL_ID,
                    "recovery_protocol_id": manager.RECOVERY_PROTOCOL_ID,
                    "method_id": manager.METHOD_ID,
                    "stage": "assembled_full",
                    "phase": phase,
                    "requested_n": EXPECTED_REQUESTS[phase],
                    "request_keys": [
                        [question_id, sample_index]
                        for question_id, sample_index in map(_key, rows)
                    ],
                    "continuation_plan": manager.binding(
                        output / "control" / manager.PLAN_NAME, plan
                    ),
                    "source_replay_plan": manager.binding(
                        source_body["source_bindings"]["source_replay_plan"]["path"],
                        source_replay,
                    ),
                    "source_technical_gate_result": gate_result_binding,
                    "source_technical_gate_assembled_generation": gate_component,
                    "recovered_batch_1_result": context["recovery_bindings"]["RECOVERED_RESULT.json"],
                    "batch_results": batch_result_bindings,
                    "source_batch_1_generation": source_generation_bindings[phase],
                    "fresh_generation": fresh_generation_bindings,
                    "reused_terminal_rows": sum(
                        row["disposition"] != "unresolved" for row in rows
                    ),
                    "continued_technical_gate_rows": sum(
                        row["partition"] == "technical_gate"
                        and row["disposition"] == "unresolved"
                        for row in rows
                    ),
                    "continued_completion_rows": len(expected_completion),
                    "source_stopped_preserved": True,
                    "source_batch_1_regenerated": False,
                    "stored_prefix_candidates_regenerated": False,
                    "abstention_policy": "abstention_is_a_coverage_outcome_not_a_safe_judgment",
                    "external_api_calls": 0,
                },
                "summary": manager.original_runtime.source.legacy.summarize_samples(final_samples),
                "samples": final_samples,
            }
        )
        path = output / "assembled" / "full" / phase / "generation.json"
        _write_or_audit(output, path, payload, f"assembled full {phase}")
        assembled_bindings[phase] = manager.binding(path, payload)

    manifest = manager.seal(
        {
            "schema_version": 1,
            "protocol_id": manager.PROTOCOL_ID,
            "source_protocol_id": manager.SOURCE_PROTOCOL_ID,
            "recovery_protocol_id": manager.RECOVERY_PROTOCOL_ID,
            "method_id": manager.METHOD_ID,
            "status": "MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_FULL_ASSEMBLY_AUDITED",
            "continuation_plan": manager.binding(
                output / "control" / manager.PLAN_NAME, plan
            ),
            "source_technical_gate_result": gate_result_binding,
            "source_technical_gate_assembled_generation": gate_component_bindings,
            "recovered_batch_1_result": context["recovery_bindings"]["RECOVERED_RESULT.json"],
            "batch_results": batch_result_bindings,
            "assembled": assembled_bindings,
            "benefit_requested_n": 360,
            "medical_requested_n": 80,
            "source_stopped_preserved": True,
            "source_batch_1_regenerated": False,
            "stored_prefix_candidates_regenerated": False,
            "judge_authorized": False,
            "gpu_jobs_submitted": 0,
            "external_api_calls": 0,
        }
    )
    manifest_path = output / "control" / "FINAL_ASSEMBLY.json"
    _write_or_audit(output, manifest_path, manifest, "final assembly manifest")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "final_assembly_payload_sha256": manifest[manager.SEAL_FIELD],
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
    assert EXPECTED_REQUESTS == {"benefit": 360, "medical": 80}
    assert manager.BATCH_INDICES == tuple(range(2, 8))
    print("KALAI_S1_RECOVERY_CONTINUATION_ASSEMBLY_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.output_root or not args.repo_root:
        parser.error("--output-root and --repo-root are required")
    assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
