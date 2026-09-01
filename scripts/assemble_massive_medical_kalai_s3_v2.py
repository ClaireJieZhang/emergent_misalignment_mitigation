#!/usr/bin/env python3
"""CPU-only exact-union assembler for Kalai s=3 gate plus completion."""

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


sampler = _load_module(
    "_massive_medical_kalai_s3_sampler_for_assembly",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_s3_v2.py",
)
prepare = _load_module(
    "_massive_medical_kalai_s3_prepare_for_assembly",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s3_v2.py",
)


def load_component(path, source_binding, phase, stage, profile, requests):
    payload = prepare.load_json(path, f"{phase} {stage} generation")
    prepare.verify_seal(payload, f"{phase} {stage} generation")
    if set(payload) != {"meta", "summary", "samples", prepare.SEAL_FIELD}:
        raise ValueError(f"{phase} {stage} generation schema differs")
    expected_body = sampler._stream_meta(
        source_binding, phase, stage, profile, requests
    )
    expected_meta = {
        **expected_body,
        "stream_fingerprint": sampler._sha256(sampler._canonical(expected_body)),
    }
    if payload["meta"] != expected_meta:
        raise ValueError(f"{phase} {stage} generation metadata differs")
    if len(payload["samples"]) != len(requests):
        raise ValueError(f"{phase} {stage} sample count differs")
    for sample, request in zip(payload["samples"], requests):
        sampler._audit_sample(sample, request, phase, profile)
    if payload["summary"] != sampler.summarize_samples(payload["samples"]):
        raise ValueError(f"{phase} {stage} summary differs")
    return payload


def exact_union(all_requests, gate_samples, completion_samples):
    expected_keys = [
        (request["question_id"], request["sample_index"])
        for request in all_requests
    ]
    by_key = {}
    for partition, samples in (
        ("gate", gate_samples),
        ("completion", completion_samples),
    ):
        for sample in samples:
            key = (sample["question_id"], sample["sample_index"])
            if key in by_key:
                raise ValueError(f"duplicate request across partitions: {key}")
            by_key[key] = (partition, sample)
    if set(by_key) != set(expected_keys) or len(by_key) != len(expected_keys):
        raise ValueError("gate and completion are not the exact full request set")
    ordered = [by_key[key][1] for key in expected_keys]
    if len({sample["sample_sha256"] for sample in ordered}) != len(ordered):
        raise ValueError("full assembly contains duplicate sample seals")
    return ordered


def assemble(args):
    workflow_root = os.path.abspath(args.workflow_root)
    if os.path.basename(workflow_root) != prepare.EXPECTED_OUTPUT_LEAF:
        raise ValueError(
            f"workflow root must end in {prepare.EXPECTED_OUTPUT_LEAF}"
        )
    gate_result_path = os.path.join(
        workflow_root, "control", "GATE_RESULT.json"
    )
    gate_result = sampler.verify_passing_gate(gate_result_path)
    source = sampler.primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=True
    )
    source_binding = sampler.legacy._source_manifest_binding(
        args.source_protocol_manifest
    )
    final_bindings = {}
    for phase in ("benefit", "medical"):
        if phase == "benefit":
            profile, records = sampler.primary.load_massive_prompts(source, phase)
        else:
            profile, records = sampler.primary.load_medical_prompts(source)
        profile = dict(profile)
        profile["temperature"] = sampler.TEMPERATURE
        all_requests = sampler._expanded_requests(records, profile["n_samples"])
        gate_requests = sampler.select_requests(phase, "gate", all_requests)
        completion_requests = sampler.select_requests(
            phase, "completion", all_requests
        )
        component_payloads = {}
        for stage, requests in (
            ("gate", gate_requests),
            ("completion", completion_requests),
        ):
            path = os.path.join(
                workflow_root,
                "generation",
                stage,
                phase,
                "generation.json",
            )
            component_payloads[stage] = (
                load_component(
                    path, source_binding, phase, stage, profile, requests
                ),
                path,
            )
        samples = exact_union(
            all_requests,
            component_payloads["gate"][0]["samples"],
            component_payloads["completion"][0]["samples"],
        )
        for sample, request in zip(samples, all_requests):
            sampler._audit_sample(sample, request, phase, profile)
        summary = sampler.summarize_samples(samples)
        payload = prepare.seal(
            {
                "meta": {
                    "schema_version": 1,
                    "protocol_id": sampler.PROTOCOL_ID,
                    "method_id": sampler.METHOD_ID,
                    "stage": "assembled_full",
                    "phase": phase,
                    "requested_n": len(all_requests),
                    "request_keys": [
                        [request["question_id"], request["sample_index"]]
                        for request in all_requests
                    ],
                    "gate_result_payload_sha256": gate_result[
                        prepare.SEAL_FIELD
                    ],
                    "components": {
                        stage: prepare.binding(path, component)
                        for stage, (component, path) in component_payloads.items()
                    },
                    "gate_rows_regenerated": False,
                    "abstention_policy": (
                        "abstention_remains_a_coverage_outcome_not_a_judge_label"
                    ),
                },
                "summary": summary,
                "samples": samples,
            }
        )
        output_path = os.path.join(
            workflow_root, "assembled", phase, "generation.json"
        )
        if os.path.lexists(output_path):
            observed = prepare.load_json(output_path, f"assembled {phase}")
            prepare.verify_seal(observed, f"assembled {phase}")
            if observed != payload:
                raise ValueError(f"existing assembled {phase} differs")
        else:
            prepare.write_new_json(output_path, payload)
        final_bindings[phase] = prepare.binding(output_path, payload)
    assembly = prepare.seal(
        {
            "schema_version": 1,
            "protocol_id": sampler.PROTOCOL_ID,
            "method_id": sampler.METHOD_ID,
            "status": "KALAI_S3_FULL_ASSEMBLY_AUDITED",
            "gate_result": prepare.binding(gate_result_path, gate_result),
            "assembled": final_bindings,
            "gate_rows_regenerated": False,
            "external_api_calls": 0,
            "gpu_jobs_submitted": 0,
        }
    )
    assembly_path = os.path.join(
        workflow_root, "control", "ASSEMBLY.json"
    )
    if os.path.lexists(assembly_path):
        observed = prepare.load_json(assembly_path, "assembly manifest")
        prepare.verify_seal(observed, "assembly manifest")
        if observed != assembly:
            raise ValueError("existing assembly manifest differs")
    else:
        prepare.write_new_json(assembly_path, assembly)
    print(
        json.dumps(
            {
                "status": assembly["status"],
                "assembly_payload_sha256": assembly[prepare.SEAL_FIELD],
                "benefit_requested_n": 360,
                "medical_requested_n": 80,
                "gate_rows_regenerated": False,
                "external_api_calls": 0,
                "gpu_jobs_submitted": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    requests = [
        {"question_id": f"q{index}", "sample_index": 0}
        for index in range(5)
    ]
    samples = [
        {
            "question_id": f"q{index}",
            "sample_index": 0,
            "sample_sha256": f"{index:064x}",
        }
        for index in range(5)
    ]
    merged = exact_union(requests, samples[::2], samples[1::2])
    assert merged == samples
    try:
        exact_union(requests, samples[:3], samples[2:])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate partition row was accepted")
    print("MASSIVE_MEDICAL_KALAI_S3_V2_ASSEMBLY_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-root")
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.workflow_root or not args.source_protocol_manifest:
        parser.error("--workflow-root and --source-protocol-manifest are required")
    assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
