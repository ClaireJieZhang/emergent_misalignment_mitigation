#!/usr/bin/env python3
"""Generate only the missing Kalai ``s=1`` proposal suffixes.

The CPU replay plan contains an audited, ``s=1``-normalized prefix for every
request in the completed ``s=3`` run.  This controller selects only unresolved
rows in one exact partition, advances the original proposal RNG across the
stored prefix without regenerating it, and samples attempts through the frozen
``R=20`` deadline.  It loads the four reference models once for both phases.

There is no external-API path and no partial resume/retry interface.
``--preflight-only`` and ``--audit-only`` never load a tokenizer or model.
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
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subliminal_mitigate.decoding.algorithms import whole_output_acceptance


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_module(
    "_massive_medical_kalai_s1_trace_reuse_planner_for_controller",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_trace_reuse_v1.py",
)
legacy = _load_module(
    "_massive_medical_whole_output_consensus_v1_for_s1_controller",
    SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_v1.py",
)


PROTOCOL_ID = planner.PROTOCOL_ID
METHOD_ID = planner.METHOD_ID
PROPOSAL_STREAM_ID = planner.PROPOSAL_STREAM_ID
PANEL_ORDER = tuple(planner.PANEL_ORDER)
MAX_ATTEMPTS = planner.MAX_ATTEMPTS
OUTPUT_SEAL = planner.OUTPUT_SEAL
STAGES = ("technical_gate", "completion")
PHASES = ("benefit", "medical")
AUTHORIZATION_FILENAMES = {
    "technical_gate": "TECHNICAL_GATE_AUTHORIZATION.json",
    "completion": "COMPLETION_AUTHORIZATION.json",
}


def _canonical(value):
    return planner.canonical_bytes(value)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _seal(body):
    return planner.seal(body)


def _verify_seal(payload, description):
    return planner.verify_seal(payload, description)


def _load_plan(path):
    """Load a replay plan and re-audit all of its immutable sources."""

    helper = getattr(planner, "load_and_verify_plan", None)
    if not callable(helper):
        raise RuntimeError(
            "trace-reuse planner lacks load_and_verify_plan(..., "
            "audit_sources=True)"
        )
    loaded = helper(path, audit_sources=True)
    if isinstance(loaded, tuple):
        if len(loaded) != 2:
            raise ValueError("replay-plan loader returned an invalid tuple")
        payload, body = loaded
    else:
        payload = loaded
        body = _verify_seal(payload, "Kalai s=1 replay plan")
    if (
        not isinstance(payload, dict)
        or not isinstance(body, dict)
        or payload.get(OUTPUT_SEAL) is None
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("proposal_stream_id") != PROPOSAL_STREAM_ID
        or body.get("safe_reference_lower_bound") != 1
        or body.get("max_attempts") != MAX_ATTEMPTS
    ):
        raise ValueError("Kalai s=1 replay-plan identity differs")
    return payload, body


def _plan_rows(body):
    phases = body.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("replay-plan phase mapping differs")
    result = {}
    for phase in PHASES:
        phase_plan = phases[phase]
        rows = phase_plan.get("rows") if isinstance(phase_plan, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"replay-plan {phase} rows differ")
        result[phase] = rows
    return result


def _partition_rows(body, stage):
    if stage not in STAGES:
        raise ValueError(f"unknown continuation stage: {stage}")
    result = {}
    source_stage = "gate" if stage == "technical_gate" else "completion"
    for phase, rows in _plan_rows(body).items():
        selected = [row for row in rows if row.get("partition") == stage]
        if any(row.get("partition") not in STAGES for row in rows):
            raise ValueError(f"replay-plan {phase} partition differs")
        if any(row.get("stage") != source_stage for row in selected):
            raise ValueError(f"replay-plan {phase} source partition differs")
        unresolved = [
            row for row in selected if row.get("disposition") == "unresolved"
        ]
        for row in unresolved:
            if (
                row.get("classification") != "needs_continuation"
                or row.get("reusable_terminal_sample") is not None
                or row.get("next_attempt_index")
                != row.get("prefix_attempts_used")
                or row.get("max_new_attempts")
                != MAX_ATTEMPTS - row.get("prefix_attempts_used", -1)
                or not 1 <= row.get("prefix_attempts_used", 0) < MAX_ATTEMPTS
            ):
                raise ValueError(f"replay-plan unresolved {phase} row differs")
        result[phase] = unresolved
    return result


def _summary(rows_by_phase):
    return {
        phase: {
            "unresolved_requests": len(rows),
            "maximum_new_candidate_attempts": sum(
                row["max_new_attempts"] for row in rows
            ),
            "prefix_attempts_reused": sum(
                row["prefix_attempts_used"] for row in rows
            ),
        }
        for phase, rows in rows_by_phase.items()
    }


def _preflight(plan_payload, plan_body, stage):
    rows = _partition_rows(plan_body, stage)
    summary = _summary(rows)
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_CONTINUATION_PREFLIGHT_VALID",
                "protocol_id": PROTOCOL_ID,
                "method_id": METHOD_ID,
                "stage": stage,
                "replay_plan_payload_sha256": plan_payload[OUTPUT_SEAL],
                "partitions": summary,
                "unresolved_requests": sum(
                    item["unresolved_requests"] for item in summary.values()
                ),
                "maximum_new_candidate_attempts": sum(
                    item["maximum_new_candidate_attempts"]
                    for item in summary.values()
                ),
                "shared_reference_model_loads": 1,
                "reference_models_loaded": 0,
                "gpu_jobs": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return rows


def _binding_path(plan_body, name):
    bindings = plan_body.get("source_bindings")
    binding = bindings.get(name) if isinstance(bindings, dict) else None
    path = binding.get("path") if isinstance(binding, dict) else None
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError(f"replay-plan source binding differs: {name}")
    return path


def verify_gpu_authorization(path, stage, *, output_root, repo_root):
    """Verify a stage-specific, one-shot authority before any model load."""

    if stage not in STAGES:
        raise ValueError(f"unknown continuation stage: {stage}")
    path = Path(path).resolve()
    expected = (
        Path(output_root).resolve()
        / "control"
        / AUTHORIZATION_FILENAMES[stage]
    )
    if path != expected:
        raise ValueError(f"{stage} GPU authorization path differs")
    if stage == "technical_gate":
        authorizer = _load_module(
            "_massive_medical_kalai_s1_gate_authorizer_for_controller",
            SCRIPT_DIR / "authorize_massive_medical_kalai_s1_trace_reuse_v1.py",
        )
    else:
        completion_path = (
            SCRIPT_DIR
            / "authorize_massive_medical_kalai_s1_trace_reuse_completion_v1.py"
        )
        if not completion_path.is_file():
            raise ValueError(
                "completion has no versioned authorizer; it is not authorized"
            )
        authorizer = _load_module(
            "_massive_medical_kalai_s1_completion_authorizer_for_controller",
            completion_path,
        )
    verify = getattr(authorizer, "verify_authorization", None)
    if not callable(verify):
        raise ValueError(f"{stage} authorizer lacks verification")
    verify(
        argparse.Namespace(
            output_root=str(Path(output_root).resolve()),
            repo_root=str(Path(repo_root).resolve()),
        )
    )


def _advance_proposal_rng(row):
    """Audit and consume exactly the stored source/draw pairs."""

    request_seed = legacy.primary.tuple_seed(
        legacy.primary.GENERATION_SEED,
        PROPOSAL_STREAM_ID,
        row["phase"],
        row["question_id"],
        row["sample_index"],
    )
    if row.get("request_seed") != request_seed:
        raise ValueError("replay row request seed differs")
    prefix = row.get("prefix_attempts")
    if not isinstance(prefix, list) or len(prefix) != row.get(
        "prefix_attempts_used"
    ):
        raise ValueError("replay row prefix length differs")
    expected_prefix_hash = _sha256(
        _canonical({"request_seed": request_seed, "attempts": prefix})
    )
    if row.get("prefix_trace_sha256") != expected_prefix_hash:
        raise ValueError("replay row prefix trace hash differs")
    rng = random.Random(request_seed)
    for index, attempt in enumerate(prefix):
        source = PANEL_ORDER[rng.randrange(len(PANEL_ORDER))]
        token_seed = legacy.primary.tuple_seed(
            request_seed, "candidate_tokens", index, source
        )
        draw = rng.random()
        values = [float(attempt["sequence_logps"][role]) for role in PANEL_ORDER]
        probability = whole_output_acceptance(values)
        eligible = not (
            row["phase"] == "medical"
            and attempt.get("finish_reason") != "stop"
        )
        if (
            attempt.get("attempt_index") != index
            or attempt.get("proposal_source") != source
            or attempt.get("token_seed") != token_seed
            or attempt.get("uniform_draw") != draw
            or not math.isclose(
                float(attempt.get("acceptance_probability")),
                probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or attempt.get("eligible_for_acceptance") is not eligible
            or attempt.get("accepted")
            is not (eligible and draw < probability)
            or attempt.get("accepted") is not False
        ):
            raise ValueError(f"replay prefix differs at attempt {index}")
    return rng


def _request_fields(row):
    return {
        key: row[key]
        for key in (
            "request_index",
            "prompt_ordinal",
            "question_id",
            "sample_index",
            "prompt_sha256",
        )
    }


def _continue_row(
    *,
    row,
    record,
    models,
    tokenizer,
    profile,
    device,
    stop_ids,
    grammar_factory,
):
    """Generate only attempts after the plan's exact stored prefix."""

    row_body = dict(row)
    observed_row_hash = row_body.pop("row_sha256", None)
    if observed_row_hash != _sha256(_canonical(row_body)):
        raise ValueError("replay-plan row seal differs")
    if row.get("disposition") != "unresolved":
        raise ValueError("controller received a resolved replay row")
    rng = _advance_proposal_rng(row)
    request_seed = row["request_seed"]
    prompt_ids = legacy.primary.make_prompt_ids(tokenizer, record)
    if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
        raise ValueError(f"request exceeds frozen context: {row['question_id']}")
    attempts = [dict(item) for item in row["prefix_attempts"]]
    accepted_candidate = None

    for attempt_index in range(row["next_attempt_index"], MAX_ATTEMPTS):
        source_index = rng.randrange(len(PANEL_ORDER))
        source = PANEL_ORDER[source_index]
        token_seed = legacy.primary.tuple_seed(
            request_seed, "candidate_tokens", attempt_index, source
        )
        candidate = legacy._sample_candidate(
            prompt_ids=prompt_ids,
            models=models,
            tokenizer=tokenizer,
            profile=profile,
            source_index=source_index,
            token_seed=token_seed,
            device=device,
            stop_ids=stop_ids,
            grammar_factory=grammar_factory,
        )
        probability = whole_output_acceptance(candidate["sequence_logps"])
        eligible = not (
            row["phase"] == "medical"
            and candidate["finish_reason"] != "stop"
        )
        uniform_draw = rng.random()
        accepted = eligible and uniform_draw < probability
        attempt = {
            "attempt_index": attempt_index,
            "proposal_source": source,
            "token_seed": token_seed,
            "finish_reason": candidate["finish_reason"],
            "generated_tokens": candidate["generated_tokens"],
            "sampled_tokens": candidate["sampled_tokens"],
            "sequence_logps": {
                role: value
                for role, value in zip(PANEL_ORDER, candidate["sequence_logps"])
            },
            "acceptance_probability": probability,
            "uniform_draw": uniform_draw,
            "eligible_for_acceptance": eligible,
            "accepted": accepted,
            "response_sha256": _sha256(candidate["response"].encode("utf-8")),
        }
        attempts.append(attempt)
        if accepted:
            accepted_candidate = candidate
            break

    request = _request_fields(row)
    if accepted_candidate is None:
        sample = {
            **request,
            "request_seed": request_seed,
            "accepted": False,
            "abstained": True,
            "attempts_used": MAX_ATTEMPTS,
            "response": "",
            "response_sha256": _sha256(b""),
            "finish_reason": "abstain",
            "generated_tokens": 0,
            "attempts": attempts,
        }
    else:
        terminal = attempts[-1]
        sample = {
            **request,
            "request_seed": request_seed,
            "accepted": True,
            "abstained": False,
            "attempts_used": len(attempts),
            "accepted_source": terminal["proposal_source"],
            "response": accepted_candidate["response"],
            "response_sha256": terminal["response_sha256"],
            "finish_reason": accepted_candidate["finish_reason"],
            "generated_tokens": accepted_candidate["generated_tokens"],
            "attempts": attempts,
        }
        if accepted_candidate.get("prediction") is not None:
            sample["prediction"] = accepted_candidate["prediction"]
    sample["sample_sha256"] = _sha256(_canonical(sample))
    planner.assemble_s1_sample(row, sample)
    legacy._audit_sample(sample, request, row["phase"], profile)
    return sample


def _load_four_reference_models(protocol, device, base_snapshot):
    """Load exactly A/B1/B2/B3 as four independent PEFT model objects."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    primary = legacy.primary
    if not torch.cuda.is_available():
        raise ValueError("Kalai s=1 generation requires an available CUDA device")
    memory_before = primary.read_gpu_memory(device)
    if "H200" not in memory_before["device_name"].upper():
        raise ValueError("Kalai s=1 generation requires the authorized H200 GPU")
    snapshot_path = primary.verify_pinned_base_snapshot(base_snapshot)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "local_files_only": True,
        "use_safetensors": True,
    }
    models = {}
    storage_sets = {}
    backend_evidence = {}
    for role in PANEL_ORDER:
        fresh_base = AutoModelForCausalLM.from_pretrained(
            snapshot_path, **load_kwargs
        )
        binding = protocol["references"][primary.MODEL_NAME_BY_ROLE[role]]
        model = PeftModel.from_pretrained(
            fresh_base,
            binding["model_path"],
            adapter_name=role,
            is_trainable=False,
        )
        model.eval()
        model.config.use_cache = True
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        configured_adapters = list(model.peft_config)
        active_adapters = primary.active_adapter_names(model)
        if (
            getattr(model.config, "_attn_implementation", None) != "sdpa"
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or model.training is not False
            or configured_adapters != [role]
            or active_adapters != [role]
        ):
            raise ValueError(
                f"loaded reference requires frozen BF16/SDPA/eval backend: {role}"
            )
        parameter_evidence, storage_sets[role] = primary.parameter_storage_inventory(
            model, role, device
        )
        backend_evidence[role] = {
            "attention_backend": "sdpa",
            "embedding_dtype": str(weight.dtype),
            "training": False,
            "configured_adapters": configured_adapters,
            "active_adapters": active_adapters,
            **parameter_evidence,
        }
        models[role] = model
    if list(models) != list(PANEL_ORDER) or len({id(item) for item in models.values()}) != 4:
        raise ValueError("reference panel does not contain four independent objects")
    for index, left in enumerate(PANEL_ORDER):
        for right in PANEL_ORDER[index + 1 :]:
            if storage_sets[left] & storage_sets[right]:
                raise ValueError(f"reference models share parameter storage: {left}/{right}")
    memory_after = primary.read_gpu_memory(device)
    memory_evidence = primary.build_gpu_memory_evidence(
        memory_before, memory_after
    )
    if memory_after["allocated_memory_bytes"] <= memory_before[
        "allocated_memory_bytes"
    ]:
        raise ValueError("four-model load did not increase CUDA allocation")
    return models, backend_evidence, memory_evidence


def _stream_meta(plan_payload, plan_body, stage, phase, rows, profile):
    manifest = plan_body["source_bindings"]["source_protocol_manifest"]
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "proposal_stream_id": PROPOSAL_STREAM_ID,
        "safe_reference_lower_bound": 1,
        "maximum_attempts_R": MAX_ATTEMPTS,
        "stage": stage,
        "phase": phase,
        "source_protocol_manifest": manifest,
        "replay_plan_payload_sha256": plan_payload[OUTPUT_SEAL],
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
        "external_api_calls": 0,
    }
    return {**body, "stream_fingerprint": _sha256(_canonical(body))}


def _shard_name(row):
    return legacy._request_shard_name(_request_fields(row))


def _write_new_json(path, payload, description):
    path = Path(path)
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite {description}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy.primary.atomic_write_json(path, payload)


def _audit_shard(path, meta, row, profile):
    payload, _ = legacy.primary.load_json_regular(path, "Kalai s=1 continuation shard")
    body = _verify_seal(payload, "Kalai s=1 continuation shard")
    if set(body) != {
        "stream_fingerprint",
        "replay_plan_payload_sha256",
        "row_sha256",
        "runtime",
        "sample",
    }:
        raise ValueError("Kalai s=1 continuation shard schema differs")
    if (
        body["stream_fingerprint"] != meta["stream_fingerprint"]
        or body["replay_plan_payload_sha256"]
        != meta["replay_plan_payload_sha256"]
        or body["row_sha256"] != row["row_sha256"]
    ):
        raise ValueError("Kalai s=1 continuation shard binding differs")
    expected_runtime = {
        "torch": legacy.primary.PINNED_TORCH_VERSION,
        "transformers": legacy.primary.PINNED_TRANSFORMERS_VERSION,
        "peft": legacy.primary.PINNED_PEFT_VERSION,
        "xgrammar": legacy.primary.PINNED_XGRAMMAR_VERSION,
    }
    if body["runtime"] != expected_runtime:
        raise ValueError("Kalai s=1 continuation shard runtime differs")
    sample = planner.assemble_s1_sample(row, body["sample"])
    legacy._audit_sample(sample, _request_fields(row), row["phase"], profile)
    return sample


def _generation_payload(meta, samples):
    return _seal(
        {
            "meta": meta,
            "summary": legacy.summarize_samples(samples),
            "new_attempts_generated": sum(
                sample["attempts_used"]
                for sample in samples
            )
            - meta["prefix_attempts_reused"],
            "samples": samples,
        }
    )


def _audit_generation(stream_root, meta, rows, profile):
    expected_names = [_shard_name(row) for row in rows]
    shard_root = stream_root / "shards"
    if shard_root.is_symlink() or not shard_root.is_dir():
        raise ValueError("Kalai s=1 continuation shard root is absent or unsafe")
    actual_names = sorted(path.name for path in shard_root.iterdir())
    if sorted(expected_names) != actual_names:
        raise ValueError("Kalai s=1 continuation shard inventory differs")
    samples = [
        _audit_shard(shard_root / name, meta, row, profile)
        for row, name in zip(rows, expected_names)
    ]
    path = stream_root / "generation.json"
    payload, _ = legacy.primary.load_json_regular(path, "Kalai s=1 continuation generation")
    _verify_seal(payload, "Kalai s=1 continuation generation")
    expected = _generation_payload(meta, samples)
    if payload != expected:
        raise ValueError("Kalai s=1 continuation generation differs from shards")
    return payload, samples


def _profiles_and_records(protocol):
    result = {}
    for phase in PHASES:
        if phase == "benefit":
            profile, records = legacy.primary.load_massive_prompts(protocol, phase)
        else:
            profile, records = legacy.primary.load_medical_prompts(protocol)
        profile = dict(profile)
        profile["temperature"] = legacy.TEMPERATURE
        result[phase] = {
            "profile": profile,
            "records": records,
            "record_by_id": {record["question_id"]: record for record in records},
        }
    return result


def _validate_requests(rows_by_phase, phase_data):
    for phase, rows in rows_by_phase.items():
        profile = phase_data[phase]["profile"]
        records = phase_data[phase]["records"]
        requests = legacy._expanded_requests(records, profile["n_samples"])
        by_key = {(item["question_id"], item["sample_index"]): item for item in requests}
        for row in rows:
            key = row["question_id"], row["sample_index"]
            if key not in by_key or _request_fields(row) != by_key[key]:
                raise ValueError(f"{phase} replay row differs from source request")


def _require_fresh_stage(output_root, stage):
    root = Path(output_root).resolve() / "generation" / stage
    if os.path.lexists(root):
        raise ValueError(
            f"partial or complete {stage} generation exists; resume/retry forbidden"
        )
    return root


def _audit_stage(output_root, plan_payload, plan_body, stage):
    rows_by_phase = _partition_rows(plan_body, stage)
    manifest_path = _binding_path(plan_body, "source_protocol_manifest")
    protocol = legacy.primary.load_protocol_manifest(manifest_path, audit_models=False)
    phase_data = _profiles_and_records(protocol)
    _validate_requests(rows_by_phase, phase_data)
    stage_root = Path(output_root).resolve() / "generation" / stage
    phase_results = {}
    for phase in PHASES:
        rows = rows_by_phase[phase]
        meta = _stream_meta(
            plan_payload,
            plan_body,
            stage,
            phase,
            rows,
            phase_data[phase]["profile"],
        )
        payload, samples = _audit_generation(
            stage_root / phase, meta, rows, phase_data[phase]["profile"]
        )
        phase_results[phase] = {
            "generation": str(stage_root / phase / "generation.json"),
            "generation_payload_sha256": payload[OUTPUT_SEAL],
            "summary": legacy.summarize_samples(samples),
        }
    combined_path = stage_root / "combined_timing.json"
    combined, _ = legacy.primary.load_json_regular(
        combined_path, "Kalai s=1 combined timing"
    )
    body = _verify_seal(combined, "Kalai s=1 combined timing")
    elapsed = body.get("elapsed_seconds")
    setup_elapsed = body.get("setup_elapsed_seconds")
    gpu_memory = body.get("gpu_memory")
    if (
        set(body)
        != {
            "protocol_id",
            "method_id",
            "stage",
            "replay_plan_payload_sha256",
            "elapsed_seconds",
            "setup_elapsed_seconds",
            "shared_reference_model_loads",
            "reference_models",
            "reference_model_backend_evidence",
            "gpu_memory",
            "stored_prefix_candidates_regenerated",
            "phases",
            "restart_or_resume_authorized",
            "external_api_calls",
        }
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != stage
        or body.get("replay_plan_payload_sha256") != plan_payload[OUTPUT_SEAL]
        or body.get("phases") != phase_results
        or body.get("shared_reference_model_loads") != 1
        or body.get("reference_models") != list(PANEL_ORDER)
        or set(body.get("reference_model_backend_evidence", {}))
        != set(PANEL_ORDER)
        or body.get("stored_prefix_candidates_regenerated") is not False
        or body.get("restart_or_resume_authorized") is not False
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or isinstance(setup_elapsed, bool)
        or not isinstance(setup_elapsed, (int, float))
        or not 0 <= setup_elapsed <= elapsed
        or not isinstance(gpu_memory, dict)
        or "H200" not in str(gpu_memory.get("device_name", "")).upper()
        or gpu_memory.get("total_memory_requirement_met") is not True
        or body.get("external_api_calls") != 0
    ):
        raise ValueError("Kalai s=1 combined timing differs")
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_CONTINUATION_AUDITED",
                "stage": stage,
                "unresolved_requests": sum(len(rows) for rows in rows_by_phase.values()),
                "maximum_new_candidate_attempts": sum(
                    row["max_new_attempts"]
                    for rows in rows_by_phase.values()
                    for row in rows
                ),
                "combined_timing_payload_sha256": combined[OUTPUT_SEAL],
                "reference_models_loaded": 0,
                "gpu_jobs_submitted_by_auditor": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_generation(args, plan_payload, plan_body):
    overall_started = time.perf_counter()
    rows_by_phase = _partition_rows(plan_body, args.stage)
    stage_root = _require_fresh_stage(args.output_root, args.stage)
    manifest_path = _binding_path(plan_body, "source_protocol_manifest")
    protocol = legacy.primary.load_protocol_manifest(manifest_path, audit_models=True)
    phase_data = _profiles_and_records(protocol)
    _validate_requests(rows_by_phase, phase_data)

    legacy.primary.force_offline_environment()
    runtime = legacy.primary.require_pinned_runtime(require_cuda=True)
    base_snapshot = legacy.primary.resolve_pinned_base_snapshot()
    tokenizer, _, grammar = legacy.primary.load_tokenizer_and_grammar(
        phase_data["benefit"]["profile"], base_snapshot
    )
    models, backend_evidence, gpu_memory = _load_four_reference_models(
        protocol, args.device, base_snapshot
    )
    import torch

    torch.cuda.empty_cache()
    stop_ids = legacy.primary.stop_token_ids(tokenizer, models["A"])
    setup_elapsed_seconds = time.perf_counter() - overall_started
    phase_results = {}
    for phase in PHASES:
        phase_started = time.perf_counter()
        rows = rows_by_phase[phase]
        data = phase_data[phase]
        meta = _stream_meta(
            plan_payload,
            plan_body,
            args.stage,
            phase,
            rows,
            data["profile"],
        )
        stream_root = stage_root / phase
        shard_root = stream_root / "shards"
        samples = []
        for row in rows:
            sample = _continue_row(
                row=row,
                record=data["record_by_id"][row["question_id"]],
                models=models,
                tokenizer=tokenizer,
                profile=data["profile"],
                device=args.device,
                stop_ids=stop_ids,
                grammar_factory=(grammar["factory"] if phase == "benefit" else None),
            )
            shard = _seal(
                {
                    "stream_fingerprint": meta["stream_fingerprint"],
                    "replay_plan_payload_sha256": plan_payload[OUTPUT_SEAL],
                    "row_sha256": row["row_sha256"],
                    "runtime": runtime,
                    "sample": sample,
                }
            )
            _write_new_json(shard_root / _shard_name(row), shard, "continuation shard")
            samples.append(sample)
        generation = _generation_payload(meta, samples)
        generation_path = stream_root / "generation.json"
        _write_new_json(generation_path, generation, "continuation generation")
        phase_results[phase] = {
            "generation": str(generation_path),
            "generation_payload_sha256": generation[OUTPUT_SEAL],
            "summary": legacy.summarize_samples(samples),
        }
        timing = _seal(
            {
                "protocol_id": PROTOCOL_ID,
                "method_id": METHOD_ID,
                "stage": args.stage,
                "phase": phase,
                "replay_plan_payload_sha256": plan_payload[OUTPUT_SEAL],
                "elapsed_seconds": time.perf_counter() - phase_started,
                "summary": phase_results[phase]["summary"],
                "external_api_calls": 0,
            }
        )
        _write_new_json(stream_root / "timing.json", timing, "phase timing")
    combined = _seal(
        {
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "stage": args.stage,
            "replay_plan_payload_sha256": plan_payload[OUTPUT_SEAL],
            "elapsed_seconds": time.perf_counter() - overall_started,
            "setup_elapsed_seconds": setup_elapsed_seconds,
            "shared_reference_model_loads": 1,
            "reference_models": list(PANEL_ORDER),
            "reference_model_backend_evidence": backend_evidence,
            "gpu_memory": gpu_memory,
            "stored_prefix_candidates_regenerated": False,
            "phases": phase_results,
            "restart_or_resume_authorized": False,
            "external_api_calls": 0,
        }
    )
    combined_path = stage_root / "combined_timing.json"
    _write_new_json(combined_path, combined, "combined timing")
    print(
        json.dumps(
            {
                "status": "MASSIVE_MEDICAL_KALAI_S1_CONTINUATION_COMPLETE",
                "stage": args.stage,
                "unresolved_requests": sum(len(rows) for rows in rows_by_phase.values()),
                "combined_timing": str(combined_path),
                "combined_timing_payload_sha256": combined[OUTPUT_SEAL],
                "shared_reference_model_loads": 1,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def self_test():
    assert STAGES == ("technical_gate", "completion")
    assert PANEL_ORDER == ("A", "B1", "B2", "B3")
    assert MAX_ATTEMPTS == 20
    assert whole_output_acceptance([0.0] * 4) == 1.0
    print("MASSIVE_MEDICAL_KALAI_S1_TRACE_REUSE_CONTROLLER_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-plan")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-authorization")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.replay_plan or not args.output_root or not args.stage:
        parser.error("--replay-plan, --output-root, and --stage are required")
    if args.preflight_only and args.audit_only:
        parser.error("--preflight-only and --audit-only are mutually exclusive")
    if args.preflight_only or args.audit_only:
        if args.gpu_authorization:
            parser.error("CPU-only modes forbid --gpu-authorization")
    elif not args.repo_root or not args.gpu_authorization:
        parser.error("generation requires --repo-root and --gpu-authorization")

    plan_payload, plan_body = _load_plan(args.replay_plan)
    if args.preflight_only:
        _preflight(plan_payload, plan_body, args.stage)
        return 0
    if args.audit_only:
        return _audit_stage(args.output_root, plan_payload, plan_body, args.stage)
    verify_gpu_authorization(
        args.gpu_authorization,
        args.stage,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    return _run_generation(args, plan_payload, plan_body)


if __name__ == "__main__":
    raise SystemExit(main())
