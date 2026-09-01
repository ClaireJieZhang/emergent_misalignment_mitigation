#!/usr/bin/env python3
"""Generate one immutable Kalai ``s=1`` completion batch.

Each invocation owns exactly one of the seven CPU-planned batches.  It reuses
the frozen ``s=1`` continuation implementation for each row, but writes to a
fresh batch-specific namespace and has no resume, retry, next-batch, or API
path.  CPU-only preflight and audit modes never load a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


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


batch_planner = _load_module(
    "_kalai_s1_completion_batch_planner_for_controller",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_completion_batches_v1.py",
)
source = _load_module(
    "_kalai_s1_trace_reuse_controller_for_batches",
    SCRIPT_DIR / "sample_massive_medical_kalai_s1_trace_reuse_v1.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_completion_batches_v1"
SOURCE_PROTOCOL_ID = source.PROTOCOL_ID
METHOD_ID = source.METHOD_ID
PROPOSAL_STREAM_ID = source.PROPOSAL_STREAM_ID
SEAL_FIELD = source.OUTPUT_SEAL
PHASES = source.PHASES
BATCH_INDICES = tuple(range(1, 8))


def _canonical(value):
    return batch_planner.canonical_bytes(value)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _seal(body):
    return batch_planner.seal(body)


def _verify_seal(payload, description):
    return batch_planner.verify_seal(payload, description)


def batch_id(batch_index):
    if batch_index not in BATCH_INDICES:
        raise ValueError("batch index must be exactly one of 1..7")
    return f"batch_{batch_index:02d}"


def _load_batch_plan(path):
    helper = getattr(batch_planner, "load_and_verify_plan", None)
    if not callable(helper):
        raise RuntimeError("batch planner lacks deep load_and_verify_plan")
    loaded = helper(path, audit_sources=True)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("batch-plan loader returned an invalid value")
    payload, body = loaded
    if (
        not isinstance(payload, dict)
        or not isinstance(body, dict)
        or payload.get(SEAL_FIELD) is None
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("controller_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
    ):
        raise ValueError("Kalai s=1 completion batch-plan identity differs")
    return payload, body


def _source_binding(plan_body, name):
    bindings = plan_body.get("source_bindings")
    binding = bindings.get(name) if isinstance(bindings, dict) else None
    path = binding.get("path") if isinstance(binding, dict) else None
    if (
        not isinstance(path, str)
        or not os.path.isabs(path)
        or not isinstance(binding.get("payload_sha256"), str)
    ):
        raise ValueError(f"batch-plan source binding differs: {name}")
    return binding


def _source_replay(plan_body):
    binding = _source_binding(plan_body, "source_replay_plan")
    payload, body = source._load_plan(binding["path"])
    if payload[SEAL_FIELD] != binding["payload_sha256"]:
        raise ValueError("source replay-plan payload binding differs")
    return payload, body


def _batch(plan_body, batch_index):
    expected_ids = [batch_id(index) for index in BATCH_INDICES]
    batches = plan_body.get("batches")
    if (
        not isinstance(batches, list)
        or [item.get("batch_index") for item in batches]
        != list(BATCH_INDICES)
        or [item.get("batch_id") for item in batches] != expected_ids
    ):
        raise ValueError("batch-plan ordered batch inventory differs")
    item = batches[batch_index - 1]
    rows_by_phase = item.get("rows")
    if not isinstance(rows_by_phase, dict) or set(rows_by_phase) != set(PHASES):
        raise ValueError("batch phase inventory differs")
    for phase in PHASES:
        rows = rows_by_phase[phase]
        if not isinstance(rows, list):
            raise ValueError(f"{phase} batch rows differ")
        for row in rows:
            row_body = dict(row)
            observed = row_body.pop("row_sha256", None)
            if (
                observed != _sha256(_canonical(row_body))
                or row.get("phase") != phase
                or row.get("partition") != "completion"
                or row.get("disposition") != "unresolved"
                or row.get("classification") != "needs_continuation"
                or row.get("reusable_terminal_sample") is not None
                or row.get("next_attempt_index")
                != row.get("prefix_attempts_used")
                or row.get("max_new_attempts")
                != source.MAX_ATTEMPTS - row.get("prefix_attempts_used", -1)
                or not 1 <= row.get("prefix_attempts_used", 0) < source.MAX_ATTEMPTS
            ):
                raise ValueError(f"{phase} {batch_id(batch_index)} row differs")
    return item, rows_by_phase


def _audit_rows_against_source(rows_by_phase, source_body):
    source_rows = source._plan_rows(source_body)
    for phase in PHASES:
        by_hash = {
            row["row_sha256"]: row
            for row in source_rows[phase]
            if row.get("partition") == "completion"
            and row.get("disposition") == "unresolved"
        }
        for row in rows_by_phase[phase]:
            if by_hash.get(row["row_sha256"]) != row:
                raise ValueError(f"{phase} batch row differs from source replay")


def _summary(rows_by_phase):
    return {
        phase: {
            "unresolved_requests": len(rows),
            "prefix_attempts_reused": sum(
                row["prefix_attempts_used"] for row in rows
            ),
            "maximum_new_candidate_attempts": sum(
                row["max_new_attempts"] for row in rows
            ),
            "row_sha256s": [row["row_sha256"] for row in rows],
        }
        for phase, rows in rows_by_phase.items()
    }


def _validate_planner_batch_summary(item, rows_by_phase):
    summary = item.get("summary")
    if not isinstance(summary, dict) or set(summary) != {
        "row_count",
        "rows_by_phase",
        "maximum_new_attempts",
        "maximum_new_attempts_by_phase",
        "weighted_maximum_seconds_ratio",
        "row_sha256s_by_phase",
    }:
        raise ValueError("batch planner summary schema differs")
    expected_rows = {phase: len(rows_by_phase[phase]) for phase in PHASES}
    expected_attempts = {
        phase: sum(row["max_new_attempts"] for row in rows_by_phase[phase])
        for phase in PHASES
    }
    expected_hashes = {
        phase: [row["row_sha256"] for row in rows_by_phase[phase]]
        for phase in PHASES
    }
    if (
        summary.get("row_count") != sum(expected_rows.values())
        or summary.get("rows_by_phase") != expected_rows
        or summary.get("maximum_new_attempts") != sum(expected_attempts.values())
        or summary.get("maximum_new_attempts_by_phase") != expected_attempts
        or summary.get("row_sha256s_by_phase") != expected_hashes
    ):
        raise ValueError("batch planner summary differs from exact rows")
    batch_planner._fraction_from_ratio(
        summary.get("weighted_maximum_seconds_ratio"),
        "batch weighted maximum seconds",
    )
    return summary


def _preflight(plan_payload, plan_body, batch_index):
    item, rows = _batch(plan_body, batch_index)
    source_payload, source_body = _source_replay(plan_body)
    _audit_rows_against_source(rows, source_body)
    summary = _summary(rows)
    _validate_planner_batch_summary(item, rows)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_PREFLIGHT_VALID",
                "protocol_id": PROTOCOL_ID,
                "batch_id": batch_id(batch_index),
                "batch_plan_payload_sha256": plan_payload[SEAL_FIELD],
                "source_replay_plan_payload_sha256": source_payload[SEAL_FIELD],
                "partitions": summary,
                "unresolved_requests": sum(
                    value["unresolved_requests"] for value in summary.values()
                ),
                "maximum_new_candidate_attempts": sum(
                    value["maximum_new_candidate_attempts"]
                    for value in summary.values()
                ),
                "reference_models_loaded": 0,
                "gpu_jobs": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return rows


def _authorization_path(output_root, batch_index):
    return (
        Path(output_root).resolve()
        / "control"
        / "batches"
        / batch_id(batch_index)
        / "AUTHORIZATION.json"
    )


def verify_gpu_authorization(
    path, batch_index, *, output_root, repo_root, slurm_job_id
):
    path = Path(path).resolve()
    if path != _authorization_path(output_root, batch_index):
        raise ValueError("completion-batch authorization path differs")
    authorizer = _load_module(
        "_kalai_s1_completion_batch_authorizer_for_controller",
        SCRIPT_DIR / "authorize_massive_medical_kalai_s1_completion_batch_v1.py",
    )
    authorizer.verify_authorization(
        argparse.Namespace(
            output_root=str(Path(output_root).resolve()),
            repo_root=str(Path(repo_root).resolve()),
            batch_index=batch_index,
            allow_generation=False,
            allow_running_state=True,
            slurm_job_id=slurm_job_id,
        )
    )


def _stream_meta(
    plan_payload,
    source_replay_payload,
    batch_index,
    phase,
    rows,
    profile,
):
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "proposal_stream_id": PROPOSAL_STREAM_ID,
        "batch_index": batch_index,
        "batch_id": batch_id(batch_index),
        "phase": phase,
        "batch_plan_payload_sha256": plan_payload[SEAL_FIELD],
        "source_replay_plan_payload_sha256": source_replay_payload[SEAL_FIELD],
        "requested_n": len(rows),
        "maximum_new_candidate_attempts": sum(
            row["max_new_attempts"] for row in rows
        ),
        "prefix_attempts_reused": sum(
            row["prefix_attempts_used"] for row in rows
        ),
        "request_keys": [
            [row["question_id"], row["sample_index"]] for row in rows
        ],
        "row_sha256s": [row["row_sha256"] for row in rows],
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
        "stored_prefix_candidates_regenerated": False,
        "restart_or_resume_authorized": False,
        "retry_or_replacement_authorized": False,
        "automatic_next_batch_authorized": False,
        "external_api_calls": 0,
    }
    return {**body, "stream_fingerprint": _sha256(_canonical(body))}


def _shard_payload(meta, row, runtime, sample):
    return _seal(
        {
            "stream_fingerprint": meta["stream_fingerprint"],
            "batch_plan_payload_sha256": meta["batch_plan_payload_sha256"],
            "source_replay_plan_payload_sha256": meta[
                "source_replay_plan_payload_sha256"
            ],
            "batch_id": meta["batch_id"],
            "row_sha256": row["row_sha256"],
            "runtime": runtime,
            "sample": sample,
        }
    )


def _audit_shard(path, meta, row, profile):
    payload, _ = source.legacy.primary.load_json_regular(
        path, "Kalai s=1 completion-batch shard"
    )
    body = _verify_seal(payload, "Kalai s=1 completion-batch shard")
    if set(body) != {
        "stream_fingerprint",
        "batch_plan_payload_sha256",
        "source_replay_plan_payload_sha256",
        "batch_id",
        "row_sha256",
        "runtime",
        "sample",
    }:
        raise ValueError("completion-batch shard schema differs")
    if (
        body["stream_fingerprint"] != meta["stream_fingerprint"]
        or body["batch_plan_payload_sha256"]
        != meta["batch_plan_payload_sha256"]
        or body["source_replay_plan_payload_sha256"]
        != meta["source_replay_plan_payload_sha256"]
        or body["batch_id"] != meta["batch_id"]
        or body["row_sha256"] != row["row_sha256"]
    ):
        raise ValueError("completion-batch shard binding differs")
    expected_runtime = {
        "torch": source.legacy.primary.PINNED_TORCH_VERSION,
        "transformers": source.legacy.primary.PINNED_TRANSFORMERS_VERSION,
        "peft": source.legacy.primary.PINNED_PEFT_VERSION,
        "xgrammar": source.legacy.primary.PINNED_XGRAMMAR_VERSION,
    }
    if body["runtime"] != expected_runtime:
        raise ValueError("completion-batch shard runtime differs")
    sample = source.planner.assemble_s1_sample(row, body["sample"])
    source.legacy._audit_sample(
        sample, source._request_fields(row), row["phase"], profile
    )
    return sample


def _generation_payload(meta, samples):
    return _seal(
        {
            "meta": meta,
            "summary": source.legacy.summarize_samples(samples),
            "new_attempts_generated": sum(
                sample["attempts_used"] for sample in samples
            )
            - meta["prefix_attempts_reused"],
            "samples": samples,
        }
    )


def _batch_root(output_root, batch_index):
    return (
        Path(output_root).resolve()
        / "generation"
        / "completion_batches"
        / batch_id(batch_index)
    )


def _require_fresh_batch(output_root, batch_index):
    root = _batch_root(output_root, batch_index)
    if os.path.lexists(root):
        raise ValueError("partial or complete batch exists; resume/retry forbidden")
    return root


def _phase_data(source_plan_body, rows_by_phase, *, audit_models):
    manifest_path = source._binding_path(
        source_plan_body, "source_protocol_manifest"
    )
    protocol = source.legacy.primary.load_protocol_manifest(
        manifest_path, audit_models=audit_models
    )
    data = source._profiles_and_records(protocol)
    source._validate_requests(rows_by_phase, data)
    return protocol, data


def _audit_generation(stream_root, meta, rows, profile):
    expected_names = [source._shard_name(row) for row in rows]
    shard_root = stream_root / "shards"
    if shard_root.is_symlink() or not shard_root.is_dir():
        raise ValueError("completion-batch shard root is absent or unsafe")
    actual_names = sorted(path.name for path in shard_root.iterdir())
    if actual_names != sorted(expected_names):
        raise ValueError("completion-batch shard inventory differs")
    samples = [
        _audit_shard(shard_root / name, meta, row, profile)
        for row, name in zip(rows, expected_names)
    ]
    generation_path = stream_root / "generation.json"
    payload, _ = source.legacy.primary.load_json_regular(
        generation_path, "Kalai s=1 completion-batch generation"
    )
    _verify_seal(payload, "Kalai s=1 completion-batch generation")
    if payload != _generation_payload(meta, samples):
        raise ValueError("completion-batch generation differs from shards")
    return payload, samples


def _audit_batch(output_root, plan_payload, plan_body, batch_index):
    item, rows_by_phase = _batch(plan_body, batch_index)
    source_replay_payload, source_replay_body = _source_replay(plan_body)
    _audit_rows_against_source(rows_by_phase, source_replay_body)
    _validate_planner_batch_summary(item, rows_by_phase)
    _, phase_data = _phase_data(
        source_replay_body, rows_by_phase, audit_models=False
    )
    root = _batch_root(output_root, batch_index)
    expected_inventory = {"benefit", "medical", "combined_timing.json"}
    if root.is_symlink() or not root.is_dir():
        raise ValueError("completion-batch root is absent or unsafe")
    if {path.name for path in root.iterdir()} != expected_inventory:
        raise ValueError("completion-batch root inventory differs")
    phase_results = {}
    for phase in PHASES:
        meta = _stream_meta(
            plan_payload,
            source_replay_payload,
            batch_index,
            phase,
            rows_by_phase[phase],
            phase_data[phase]["profile"],
        )
        payload, samples = _audit_generation(
            root / phase,
            meta,
            rows_by_phase[phase],
            phase_data[phase]["profile"],
        )
        timing_path = root / phase / "timing.json"
        timing_payload, _ = source.legacy.primary.load_json_regular(
            timing_path, f"Kalai s=1 completion-batch {phase} timing"
        )
        timing_body = _verify_seal(
            timing_payload, f"Kalai s=1 completion-batch {phase} timing"
        )
        phase_elapsed = timing_body.get("elapsed_seconds")
        phase_summary = source.legacy.summarize_samples(samples)
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
            or timing_body.get("protocol_id") != PROTOCOL_ID
            or timing_body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
            or timing_body.get("method_id") != METHOD_ID
            or timing_body.get("batch_index") != batch_index
            or timing_body.get("batch_id") != batch_id(batch_index)
            or timing_body.get("phase") != phase
            or timing_body.get("batch_plan_payload_sha256")
            != plan_payload[SEAL_FIELD]
            or isinstance(phase_elapsed, bool)
            or not isinstance(phase_elapsed, (int, float))
            or not math.isfinite(phase_elapsed)
            or phase_elapsed < 0
            or timing_body.get("summary") != phase_summary
            or timing_body.get("external_api_calls") != 0
        ):
            raise ValueError(f"completion-batch {phase} timing differs")
        phase_results[phase] = {
            "generation": str(root / phase / "generation.json"),
            "generation_payload_sha256": payload[SEAL_FIELD],
            "timing": str(timing_path),
            "timing_payload_sha256": timing_payload[SEAL_FIELD],
            "summary": phase_summary,
        }
    combined_path = root / "combined_timing.json"
    combined, _ = source.legacy.primary.load_json_regular(
        combined_path, "Kalai s=1 completion-batch combined timing"
    )
    body = _verify_seal(combined, "Kalai s=1 completion-batch combined timing")
    elapsed = body.get("elapsed_seconds")
    setup_elapsed = body.get("setup_elapsed_seconds")
    gpu_memory = body.get("gpu_memory")
    if (
        set(body)
        != {
            "protocol_id",
            "source_protocol_id",
            "method_id",
            "batch_index",
            "batch_id",
            "batch_plan_payload_sha256",
            "source_replay_plan_payload_sha256",
            "elapsed_seconds",
            "setup_elapsed_seconds",
            "shared_reference_model_loads",
            "reference_models",
            "reference_model_backend_evidence",
            "gpu_memory",
            "stored_prefix_candidates_regenerated",
            "phases",
            "restart_or_resume_authorized",
            "retry_or_replacement_authorized",
            "automatic_next_batch_authorized",
            "external_api_calls",
        }
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("batch_index") != batch_index
        or body.get("batch_id") != batch_id(batch_index)
        or body.get("batch_plan_payload_sha256") != plan_payload[SEAL_FIELD]
        or body.get("source_replay_plan_payload_sha256")
        != source_replay_payload[SEAL_FIELD]
        or body.get("phases") != phase_results
        or body.get("shared_reference_model_loads") != 1
        or body.get("reference_models") != list(source.PANEL_ORDER)
        or set(body.get("reference_model_backend_evidence", {}))
        != set(source.PANEL_ORDER)
        or body.get("stored_prefix_candidates_regenerated") is not False
        or body.get("restart_or_resume_authorized") is not False
        or body.get("retry_or_replacement_authorized") is not False
        or body.get("automatic_next_batch_authorized") is not False
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or isinstance(setup_elapsed, bool)
        or not isinstance(setup_elapsed, (int, float))
        or not math.isfinite(setup_elapsed)
        or not 0 <= setup_elapsed <= elapsed
        or not isinstance(gpu_memory, dict)
        or "H200" not in str(gpu_memory.get("device_name", "")).upper()
        or gpu_memory.get("total_memory_requirement_met") is not True
        or body.get("external_api_calls") != 0
    ):
        raise ValueError("completion-batch combined timing differs")
    result = {
        "batch_id": batch_id(batch_index),
        "combined_timing": str(combined_path),
        "combined_timing_payload_sha256": combined[SEAL_FIELD],
        "phases": phase_results,
    }
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_AUDITED",
                "batch_id": batch_id(batch_index),
                "combined_timing_payload_sha256": combined[SEAL_FIELD],
                "reference_models_loaded": 0,
                "gpu_jobs_submitted_by_auditor": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return result


def _run_generation(args, plan_payload, plan_body):
    overall_started = time.perf_counter()
    item, rows_by_phase = _batch(plan_body, args.batch_index)
    source_replay_payload, source_replay_body = _source_replay(plan_body)
    _audit_rows_against_source(rows_by_phase, source_replay_body)
    _validate_planner_batch_summary(item, rows_by_phase)
    root = _require_fresh_batch(args.output_root, args.batch_index)
    protocol, phase_data = _phase_data(
        source_replay_body, rows_by_phase, audit_models=True
    )

    source.legacy.primary.force_offline_environment()
    runtime = source.legacy.primary.require_pinned_runtime(require_cuda=True)
    base_snapshot = source.legacy.primary.resolve_pinned_base_snapshot()
    tokenizer, _, grammar = source.legacy.primary.load_tokenizer_and_grammar(
        phase_data["benefit"]["profile"], base_snapshot
    )
    models, backend_evidence, gpu_memory = source._load_four_reference_models(
        protocol, args.device, base_snapshot
    )
    import torch

    torch.cuda.empty_cache()
    stop_ids = source.legacy.primary.stop_token_ids(tokenizer, models["A"])
    setup_elapsed_seconds = time.perf_counter() - overall_started
    phase_results = {}
    for phase in PHASES:
        phase_started = time.perf_counter()
        rows = rows_by_phase[phase]
        data = phase_data[phase]
        meta = _stream_meta(
            plan_payload,
            source_replay_payload,
            args.batch_index,
            phase,
            rows,
            data["profile"],
        )
        stream_root = root / phase
        shard_root = stream_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=False)
        samples = []
        for row in rows:
            sample = source._continue_row(
                row=row,
                record=data["record_by_id"][row["question_id"]],
                models=models,
                tokenizer=tokenizer,
                profile=data["profile"],
                device=args.device,
                stop_ids=stop_ids,
                grammar_factory=(
                    grammar["factory"] if phase == "benefit" else None
                ),
            )
            shard = _shard_payload(meta, row, runtime, sample)
            source._write_new_json(
                shard_root / source._shard_name(row),
                shard,
                "completion-batch shard",
            )
            samples.append(sample)
        generation = _generation_payload(meta, samples)
        generation_path = stream_root / "generation.json"
        source._write_new_json(
            generation_path, generation, "completion-batch generation"
        )
        phase_results[phase] = {
            "generation": str(generation_path),
            "generation_payload_sha256": generation[SEAL_FIELD],
            "summary": source.legacy.summarize_samples(samples),
        }
        timing = _seal(
            {
                "protocol_id": PROTOCOL_ID,
                "source_protocol_id": SOURCE_PROTOCOL_ID,
                "method_id": METHOD_ID,
                "batch_index": args.batch_index,
                "batch_id": batch_id(args.batch_index),
                "phase": phase,
                "batch_plan_payload_sha256": plan_payload[SEAL_FIELD],
                "elapsed_seconds": time.perf_counter() - phase_started,
                "summary": phase_results[phase]["summary"],
                "external_api_calls": 0,
            }
        )
        source._write_new_json(
            stream_root / "timing.json", timing, "completion-batch phase timing"
        )
        phase_results[phase]["timing"] = str(stream_root / "timing.json")
        phase_results[phase]["timing_payload_sha256"] = timing[SEAL_FIELD]
    combined = _seal(
        {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "method_id": METHOD_ID,
            "batch_index": args.batch_index,
            "batch_id": batch_id(args.batch_index),
            "batch_plan_payload_sha256": plan_payload[SEAL_FIELD],
            "source_replay_plan_payload_sha256": source_replay_payload[SEAL_FIELD],
            "elapsed_seconds": time.perf_counter() - overall_started,
            "setup_elapsed_seconds": setup_elapsed_seconds,
            "shared_reference_model_loads": 1,
            "reference_models": list(source.PANEL_ORDER),
            "reference_model_backend_evidence": backend_evidence,
            "gpu_memory": gpu_memory,
            "stored_prefix_candidates_regenerated": False,
            "phases": phase_results,
            "restart_or_resume_authorized": False,
            "retry_or_replacement_authorized": False,
            "automatic_next_batch_authorized": False,
            "external_api_calls": 0,
        }
    )
    combined_path = root / "combined_timing.json"
    source._write_new_json(
        combined_path, combined, "completion-batch combined timing"
    )
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_GENERATED",
                "batch_id": batch_id(args.batch_index),
                "combined_timing_payload_sha256": combined[SEAL_FIELD],
                "automatic_next_batch_authorized": False,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def self_test():
    assert BATCH_INDICES == tuple(range(1, 8))
    assert batch_id(1) == "batch_01"
    assert batch_id(7) == "batch_07"
    assert source.MAX_ATTEMPTS == 20
    print("MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_CONTROLLER_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--batch-index", type=int, choices=BATCH_INDICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-authorization")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.batch_plan or not args.output_root or args.batch_index is None:
        parser.error("--batch-plan, --output-root, and --batch-index are required")
    if args.preflight_only and args.audit_only:
        parser.error("--preflight-only and --audit-only are mutually exclusive")
    if args.preflight_only or args.audit_only:
        if args.gpu_authorization:
            parser.error("CPU-only modes forbid --gpu-authorization")
    elif not args.repo_root or not args.gpu_authorization or not args.slurm_job_id:
        parser.error(
            "generation requires --repo-root, --gpu-authorization, and "
            "--slurm-job-id"
        )

    plan_payload, plan_body = _load_batch_plan(args.batch_plan)
    if args.preflight_only:
        _preflight(plan_payload, plan_body, args.batch_index)
        return 0
    if args.audit_only:
        _audit_batch(
            args.output_root, plan_payload, plan_body, args.batch_index
        )
        return 0
    verify_gpu_authorization(
        args.gpu_authorization,
        args.batch_index,
        output_root=args.output_root,
        repo_root=args.repo_root,
        slurm_job_id=args.slurm_job_id,
    )
    return _run_generation(args, plan_payload, plan_body)


if __name__ == "__main__":
    raise SystemExit(main())
