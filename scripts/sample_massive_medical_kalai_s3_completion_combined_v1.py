#!/usr/bin/env python3
"""One-shot Kalai completion with one shared model load across both phases.

The sealed gate used two independent sampler processes, one for MASSIVE and one
for medical prompts.  The exact gate-scaled completion projection only fits the
existing program ceiling if the completion avoids that duplicated model-load
overhead.  This controller therefore performs the *unchanged* sealed
``completion`` partitions in one process.  It reuses the sampler's request
selection, proposal streams, acceptance rule, per-request shard format, and
auditors verbatim; only model/tokenizer setup is shared.

There is no partial-resume interface and no external-API path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import sample_massive_medical_whole_output_consensus_s3_v2 as sampler


CONTROLLER_PROTOCOL_ID = "massive_medical_kalai_s3_r20_v2_completion_v1"
PHASES = ("benefit", "medical")


def _context(source, source_binding, output_root, phase):
    if phase == "benefit":
        profile, records = sampler.primary.load_massive_prompts(source, phase)
    else:
        profile, records = sampler.primary.load_medical_prompts(source)
    profile = dict(profile)
    profile["temperature"] = sampler.TEMPERATURE
    all_requests = sampler._expanded_requests(records, profile["n_samples"])
    requests = sampler.select_requests(phase, "completion", all_requests)
    expected = 358 if phase == "benefit" else 64
    if len(requests) != expected:
        raise ValueError(f"{phase} completion count differs: {len(requests)}")
    meta = sampler._stream_meta(
        source_binding, phase, "completion", profile, requests
    )
    stream_root = Path(output_root).resolve() / "completion" / phase
    return {
        "phase": phase,
        "profile": profile,
        "records": records,
        "record_by_id": {row["question_id"]: row for row in records},
        "requests": requests,
        "meta": meta,
        "stream_fingerprint": sampler._sha256(sampler._canonical(meta)),
        "stream_root": stream_root,
    }


def _preflight(contexts):
    print(
        json.dumps(
            {
                "status": "KALAI_S3_COMBINED_COMPLETION_PREFLIGHT_VALID",
                "controller_protocol_id": CONTROLLER_PROTOCOL_ID,
                "method_id": sampler.METHOD_ID,
                "partitions": {
                    item["phase"]: {
                        "requested_n": len(item["requests"]),
                        "maximum_candidate_attempts": (
                            len(item["requests"]) * sampler.MAX_ATTEMPTS
                        ),
                        "stream_fingerprint": item["stream_fingerprint"],
                    }
                    for item in contexts
                },
                "shared_model_loads": 1,
                "gpu_jobs": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def _require_fresh(contexts, output_root):
    completion_root = Path(output_root).resolve() / "completion"
    if os.path.lexists(completion_root):
        raise ValueError("partial completion namespace exists; reuse forbidden")
    for item in contexts:
        root = item["stream_root"]
        if os.path.lexists(root):
            raise ValueError(f"partial {item['phase']} completion exists; reuse forbidden")


def run(args):
    overall_started = time.perf_counter()
    source = sampler.primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=True
    )
    source_binding = sampler.legacy._source_manifest_binding(
        args.source_protocol_manifest
    )
    contexts = [
        _context(source, source_binding, args.output_root, phase)
        for phase in PHASES
    ]
    if args.preflight_only:
        _preflight(contexts)
        return 0
    _require_fresh(contexts, args.output_root)

    sampler.primary.force_offline_environment()
    runtime = sampler.primary.require_pinned_runtime(require_cuda=True)
    base_snapshot = sampler.primary.resolve_pinned_base_snapshot()
    benefit = contexts[0]
    tokenizer, _, grammar = sampler.primary.load_tokenizer_and_grammar(
        benefit["profile"], base_snapshot
    )
    models = sampler.primary.load_independent_model_panel(
        source, args.device, base_snapshot
    )
    direct_base = models.pop("base")
    del direct_base
    import torch

    torch.cuda.empty_cache()
    stop_ids = sampler.primary.stop_token_ids(tokenizer, models["A"])
    bindings = {}
    for item in contexts:
        phase_started = time.perf_counter()
        stream_root = item["stream_root"]
        stream_root.mkdir(parents=True)
        samples, missing = sampler.legacy._load_shards(
            stream_root,
            item["stream_fingerprint"],
            item["requests"],
            item["phase"],
            item["profile"],
            require_complete=False,
        )
        if samples or len(missing) != len(item["requests"]):
            raise ValueError("completion namespace was not fresh")
        generation_started = time.perf_counter()
        shard_root = stream_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        grammar_factory = grammar["factory"] if item["phase"] == "benefit" else None
        for request, shard_path in missing:
            sample = sampler.legacy._sample_request(
                phase=item["phase"],
                request=request,
                record=item["record_by_id"][request["question_id"]],
                models=models,
                tokenizer=tokenizer,
                profile=item["profile"],
                device=args.device,
                stop_ids=stop_ids,
                grammar_factory=grammar_factory,
            )
            shard = sampler.legacy._seal(
                {
                    "stream_fingerprint": item["stream_fingerprint"],
                    "runtime": runtime,
                    "sample": sample,
                }
            )
            if os.path.lexists(shard_path):
                raise ValueError(f"refusing to overwrite shard: {shard_path}")
            sampler.primary.atomic_write_json(shard_path, shard)
        samples, missing = sampler.legacy._load_shards(
            stream_root,
            item["stream_fingerprint"],
            item["requests"],
            item["phase"],
            item["profile"],
            require_complete=True,
        )
        if missing:
            raise AssertionError("complete shard audit returned missing rows")
        generation_path = sampler.legacy._write_generation(
            stream_root, item["meta"], samples
        )
        finished = time.perf_counter()
        timing = sampler.legacy._seal(
            {
                "protocol_id": sampler.PROTOCOL_ID,
                "method_id": sampler.METHOD_ID,
                "phase": item["phase"],
                "stage": "completion",
                "completed_before_process": 0,
                "completed_during_process": len(samples),
                "elapsed_seconds": finished - phase_started,
                "process_elapsed_seconds": finished - phase_started,
                "setup_elapsed_seconds": generation_started - phase_started,
                "generation_elapsed_seconds": finished - generation_started,
                "summary": sampler.summarize_samples(samples),
                "combined_controller_protocol_id": CONTROLLER_PROTOCOL_ID,
                "shared_model_loads_across_phases": 1,
                "timing_scope": "phase_after_shared_model_setup_not_billing_scope",
            }
        )
        timing_path = stream_root / "timing.json"
        if os.path.lexists(timing_path):
            raise ValueError(f"refusing to overwrite timing: {timing_path}")
        sampler.primary.atomic_write_json(timing_path, timing)
        bindings[item["phase"]] = {
            "generation": str(generation_path),
            "generation_payload_sha256": sampler.primary.load_json_regular(
                generation_path, f"{item['phase']} completion generation"
            )[0][sampler.OUTPUT_SEAL],
            "timing": str(timing_path),
            "timing_payload_sha256": timing[sampler.OUTPUT_SEAL],
            "summary": sampler.summarize_samples(samples),
        }

    combined = sampler.legacy._seal(
        {
            "controller_protocol_id": CONTROLLER_PROTOCOL_ID,
            "protocol_id": sampler.PROTOCOL_ID,
            "method_id": sampler.METHOD_ID,
            "stage": "completion",
            "shared_model_loads": 1,
            "elapsed_seconds": time.perf_counter() - overall_started,
            "phases": bindings,
            "restart_or_resume_authorized": False,
            "external_api_calls": 0,
        }
    )
    combined_path = Path(args.output_root).resolve() / "completion" / "combined_timing.json"
    sampler.primary.atomic_write_json(combined_path, combined)
    print(
        json.dumps(
            {
                "status": "KALAI_S3_COMBINED_COMPLETION_COMPLETE",
                "combined_timing": str(combined_path),
                "combined_timing_payload_sha256": combined[sampler.OUTPUT_SEAL],
                "benefit_requested_n": 358,
                "medical_requested_n": 64,
                "shared_model_loads": 1,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def self_test():
    assert PHASES == ("benefit", "medical")
    assert sampler.STAGES == ("gate", "completion")
    assert sampler.MAX_ATTEMPTS == 20
    assert sampler.SAFE_REFERENCE_LOWER_BOUND == 3
    print("KALAI_S3_COMBINED_COMPLETION_V1_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-result")
    parser.add_argument("--gpu-authorization")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.source_protocol_manifest or not args.output_root:
        parser.error("--source-protocol-manifest and --output-root are required")
    if args.preflight_only:
        if args.gate_result or args.gpu_authorization:
            parser.error("preflight forbids authority/result inputs")
    else:
        if not args.gate_result or not args.gpu_authorization:
            parser.error("generation requires gate result and GPU authorization")
        sampler.verify_passing_gate(args.gate_result)
        sampler.verify_gpu_authorization(args.gpu_authorization, "completion")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
