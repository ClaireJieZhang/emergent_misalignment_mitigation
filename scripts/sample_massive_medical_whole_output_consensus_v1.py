#!/usr/bin/env python3
"""Coverage-aware whole-output consensus for the MASSIVE/medical panel.

This is a post-hoc contextual baseline.  It binds to the already sealed
MASSIVE/medical sequential protocol, proposes complete outputs uniformly from
A/B1/B2/B3, and accepts with

    min_i p_i(y) / mean_i p_i(y).

MASSIVE candidates and every likelihood in the acceptance ratio use the same
hard, grammar-conditioned distribution.  Medical candidates use the frozen
free-text profile.  A request that has no accepted candidate after 20 attempts
is an abstention; it is never relabeled SAFE, BAD, refusal, or unparseable.

The script has no API path.  ``--preflight-only`` and ``--self-test`` never
load model weights or use a GPU.  A partial namespace cannot be resumed unless
``--resume-partial`` is supplied under a separate authorization.
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
PRIMARY_SAMPLER_PATH = (
    SCRIPT_DIR
    / "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py"
)
_PRIMARY_SPEC = importlib.util.spec_from_file_location(
    "_mmu_primary_sampler_for_whole_output_v1", PRIMARY_SAMPLER_PATH
)
if _PRIMARY_SPEC is None or _PRIMARY_SPEC.loader is None:
    raise RuntimeError(f"Could not load primary sampler: {PRIMARY_SAMPLER_PATH}")
primary = importlib.util.module_from_spec(_PRIMARY_SPEC)
_PRIMARY_SPEC.loader.exec_module(primary)

REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subliminal_mitigate.decoding.algorithms import whole_output_acceptance


PROTOCOL_ID = "massive_medical_composition_baselines_v1"
METHOD_ID = "whole_output_consensus_m4_max20_v1"
# Kept separate so later acceptance-rule versions can reuse the exact proposal
# sources, token seeds, and accept/reject uniform draws without changing the
# immutable v1 behavior.
PROPOSAL_STREAM_ID = METHOD_ID
PANEL_ORDER = ("A", "B1", "B2", "B3")
MAX_ATTEMPTS = 20
TEMPERATURE = 1.0
SMOKE_REQUESTS_PER_PHASE = 2
OUTPUT_SEAL = "payload_sha256"


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _seal(body):
    result = dict(body)
    result.pop(OUTPUT_SEAL, None)
    result[OUTPUT_SEAL] = _sha256(_canonical(result))
    return result


def _verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(OUTPUT_SEAL, None)
    if observed != _sha256(_canonical(body)):
        raise ValueError(f"{description} has an invalid {OUTPUT_SEAL}")
    return body


def _source_manifest_binding(path):
    payload, raw = primary.load_json_regular(path, "source protocol manifest")
    primary.verify_seal(payload, primary.MANIFEST_SEAL_FIELD, "source protocol manifest")
    return {
        "path": os.path.abspath(path),
        "file_sha256": _sha256(raw),
        "manifest_payload_sha256": payload[primary.MANIFEST_SEAL_FIELD],
    }


def _expanded_requests(records, n_samples):
    requests = []
    for prompt_ordinal, record in enumerate(records):
        for sample_index in range(n_samples):
            requests.append(
                {
                    "request_index": len(requests),
                    "prompt_ordinal": prompt_ordinal,
                    "question_id": record["question_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": record["prompt_sha256"],
                }
            )
    return requests


def _smoke_rank(phase, request):
    material = (
        PROPOSAL_STREAM_ID
        + "\0"
        + phase
        + "\0"
        + request["question_id"]
        + "\0"
        + str(request["sample_index"])
    )
    return _sha256(material.encode("utf-8")), request["request_index"]


def select_requests(phase, stage, requests):
    if stage == "full":
        return list(requests)
    if stage != "smoke":
        raise ValueError(f"unknown stage: {stage}")
    ranked = sorted((_smoke_rank(phase, request), request) for request in requests)
    selected_indices = {
        request["request_index"]
        for _, request in ranked[:SMOKE_REQUESTS_PER_PHASE]
    }
    return [
        request for request in requests if request["request_index"] in selected_indices
    ]


def _request_shard_name(request):
    digest = hashlib.sha256(request["question_id"].encode("utf-8")).hexdigest()[:16]
    return (
        f"request-{request['request_index']:06d}-{digest}"
        f"-n{request['sample_index']:03d}.json"
    )


def _stream_meta(source_manifest, phase, stage, profile, requests):
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_gate_eligible": False,
        "source_protocol": source_manifest,
        "phase": phase,
        "stage": stage,
        "panel_order": list(PANEL_ORDER),
        "proposal": "uniform_complete_sequence_mixture",
        "acceptance": "min_reference_probability_over_mean_reference_probability",
        "max_attempts": MAX_ATTEMPTS,
        "temperature": TEMPERATURE,
        "seed": primary.GENERATION_SEED,
        "abstention_policy": "abstain_after_20_rejections_not_a_judge_label",
        "truncated_medical_candidate_policy": "ineligible_for_acceptance",
        "grammar_likelihood_policy": (
            "same_hard_mask_and_per_reference_renormalization_each_token"
            if phase == "benefit"
            else None
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


def _apply_one_grammar_mask(reference_logps, grammar_runtime):
    """Mask every reference with one shared grammar frontier, then normalize."""
    import torch

    matcher = grammar_runtime["matcher"]
    bitmask = grammar_runtime["bitmask"]
    need_apply = matcher.fill_next_token_bitmask(bitmask)
    if type(need_apply) is not bool:
        raise ValueError("XGrammar fill_next_token_bitmask did not return bool")
    conditioned = []
    for logp in reference_logps:
        masked = logp.clone()
        if need_apply:
            batched = masked.unsqueeze(0)
            grammar_runtime["apply_token_bitmask_inplace"](
                batched, bitmask.to(masked.device)
            )
            masked = batched[0]
        conditioned.append(primary.normalize_composed_scores(masked))
    result = torch.stack(conditioned, dim=0).float()
    if result.dtype != torch.float32 or result.ndim != 2:
        raise ValueError("grammar-conditioned reference distributions are invalid")
    return result


def _sample_candidate(
    *,
    prompt_ids,
    models,
    tokenizer,
    profile,
    source_index,
    token_seed,
    device,
    stop_ids,
    grammar_factory,
):
    import torch
    import torch.nn.functional as functional

    states = [
        primary.prefill_cached_reference(models[role], prompt_ids, device)
        for role in PANEL_ORDER
    ]
    primary.assert_independent_caches(states)
    grammar_runtime = grammar_factory() if grammar_factory is not None else None
    generator = torch.Generator(device=device)
    generator.manual_seed(token_seed)
    response_ids = []
    sequence_logps = [0.0] * len(PANEL_ORDER)
    finish_reason = "max_new_tokens"
    sampled_tokens = 0

    for token_index in range(profile["max_new_tokens"]):
        reference_logps = torch.stack(
            [
                functional.log_softmax(state["next_logits"].float(), dim=-1)
                for state in states
            ],
            dim=0,
        ).float()
        conditioned = (
            _apply_one_grammar_mask(reference_logps, grammar_runtime)
            if grammar_runtime is not None
            else reference_logps
        )
        token_id = int(
            torch.multinomial(
                torch.exp(conditioned[source_index]), 1, generator=generator
            ).item()
        )
        sampled_tokens += 1
        for index in range(len(PANEL_ORDER)):
            token_logp = float(conditioned[index, token_id].item())
            if not math.isfinite(token_logp):
                raise ValueError("sampled token has non-finite reference log probability")
            sequence_logps[index] += token_logp

        terminated = False
        if grammar_runtime is not None:
            if not grammar_runtime["matcher"].accept_token(token_id):
                raise ValueError("XGrammar rejected a token admitted by its own mask")
            response_ids.append(token_id)
            terminated = grammar_runtime["matcher"].is_terminated()
        elif token_id in stop_ids:
            terminated = True
        else:
            response_ids.append(token_id)

        if terminated:
            finish_reason = "stop"
            break
        if token_index + 1 < profile["max_new_tokens"]:
            states = [
                primary.step_cached_reference(
                    models[role], token_id, state["cache"], device
                )
                for role, state in zip(PANEL_ORDER, states)
            ]
            primary.assert_independent_caches(states)

    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    prediction = None
    if grammar_runtime is not None:
        if finish_reason != "stop":
            raise ValueError("structured whole-output proposal did not terminate")
        prediction = primary.validate_prediction(
            response, profile["intent_labels"], profile["slot_labels"]
        )
    return {
        "response": response,
        "prediction": prediction,
        "finish_reason": finish_reason,
        "generated_tokens": len(response_ids),
        # Unlike generated_tokens, this includes a terminal medical EOS token
        # and therefore measures the actual number of decoded token steps.
        "sampled_tokens": sampled_tokens,
        "sequence_logps": sequence_logps,
    }


def _sample_request(
    *,
    phase,
    request,
    record,
    models,
    tokenizer,
    profile,
    device,
    stop_ids,
    grammar_factory,
):
    request_seed = primary.tuple_seed(
        primary.GENERATION_SEED,
        PROPOSAL_STREAM_ID,
        phase,
        request["question_id"],
        request["sample_index"],
    )
    rng = random.Random(request_seed)
    prompt_ids = primary.make_prompt_ids(tokenizer, record)
    if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
        raise ValueError(f"request exceeds frozen context: {request['question_id']}")
    attempts = []

    for attempt_index in range(MAX_ATTEMPTS):
        source_index = rng.randrange(len(PANEL_ORDER))
        source = PANEL_ORDER[source_index]
        token_seed = primary.tuple_seed(
            request_seed, "candidate_tokens", attempt_index, source
        )
        candidate = _sample_candidate(
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
            phase == "medical" and candidate["finish_reason"] != "stop"
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
            result = {
                **request,
                "request_seed": request_seed,
                "accepted": True,
                "abstained": False,
                "attempts_used": attempt_index + 1,
                "accepted_source": source,
                "response": candidate["response"],
                "response_sha256": attempt["response_sha256"],
                "finish_reason": candidate["finish_reason"],
                "generated_tokens": candidate["generated_tokens"],
                "attempts": attempts,
            }
            if candidate["prediction"] is not None:
                result["prediction"] = candidate["prediction"]
            result["sample_sha256"] = _sha256(_canonical(result))
            return result

    result = {
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
    result["sample_sha256"] = _sha256(_canonical(result))
    return result


def summarize_samples(samples):
    requested = len(samples)
    accepted = sum(bool(sample.get("accepted")) for sample in samples)
    abstained = sum(bool(sample.get("abstained")) for sample in samples)
    if accepted + abstained != requested:
        raise ValueError("whole-output samples do not partition accept/abstain")
    attempts = sum(int(sample.get("attempts_used", 0)) for sample in samples)
    attempt_records = [
        attempt
        for sample in samples
        for attempt in sample.get("attempts", [])
    ]
    if len(attempt_records) != attempts:
        raise ValueError("whole-output summary attempt count differs from logs")
    candidate_tokens = sum(
        int(attempt.get("generated_tokens", 0)) for attempt in attempt_records
    )
    candidate_sampled_tokens = sum(
        int(attempt.get("sampled_tokens", 0)) for attempt in attempt_records
    )
    accepted_tokens = sum(
        int(sample.get("generated_tokens", 0))
        for sample in samples
        if sample.get("accepted") is True
    )
    return {
        "requested_n": requested,
        "accepted_n": accepted,
        "abstained_n": abstained,
        "coverage": accepted / requested if requested else None,
        "abstention_rate": abstained / requested if requested else None,
        "total_attempts": attempts,
        "mean_attempts_per_request": attempts / requested if requested else None,
        # Rejected candidates dominate rejection-sampling cost, so reporting
        # only accepted-output tokens materially understates the smoke runtime.
        "total_candidate_generated_tokens": candidate_tokens,
        "total_candidate_sampled_tokens": candidate_sampled_tokens,
        "mean_candidate_generated_tokens_per_request": (
            candidate_tokens / requested if requested else None
        ),
        "mean_candidate_generated_tokens_per_attempt": (
            candidate_tokens / attempts if attempts else None
        ),
        "mean_candidate_sampled_tokens_per_request": (
            candidate_sampled_tokens / requested if requested else None
        ),
        "mean_candidate_sampled_tokens_per_attempt": (
            candidate_sampled_tokens / attempts if attempts else None
        ),
        "accepted_output_generated_tokens": accepted_tokens,
        "judge_eligible_medical_n": sum(
            bool(sample.get("accepted")) and bool(sample.get("response"))
            for sample in samples
        ),
    }


def _audit_sample(sample, request, phase, profile):
    for key, expected in request.items():
        if sample.get(key) != expected:
            raise ValueError(f"sample/request mismatch for {key}")
    body = dict(sample)
    observed = body.pop("sample_sha256", None)
    if observed != _sha256(_canonical(body)):
        raise ValueError("sample seal differs")
    accepted = sample.get("accepted") is True
    abstained = sample.get("abstained") is True
    if accepted == abstained:
        raise ValueError("sample is not exactly one of accepted/abstained")
    expected_sample_keys = set(request) | {
        "request_seed",
        "accepted",
        "abstained",
        "attempts_used",
        "response",
        "response_sha256",
        "finish_reason",
        "generated_tokens",
        "attempts",
        "sample_sha256",
    }
    if accepted:
        expected_sample_keys.add("accepted_source")
        if phase == "benefit":
            expected_sample_keys.add("prediction")
    if set(sample) != expected_sample_keys:
        raise ValueError("sample schema differs")
    attempts = sample.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= MAX_ATTEMPTS:
        raise ValueError("sample attempt log differs")
    if sample.get("attempts_used") != len(attempts):
        raise ValueError("sample attempts_used differs")
    expected_request_seed = primary.tuple_seed(
        primary.GENERATION_SEED,
        PROPOSAL_STREAM_ID,
        phase,
        request["question_id"],
        request["sample_index"],
    )
    if sample.get("request_seed") != expected_request_seed:
        raise ValueError("sample request_seed differs")
    attempt_keys = {
        "attempt_index",
        "proposal_source",
        "token_seed",
        "finish_reason",
        "generated_tokens",
        "sampled_tokens",
        "sequence_logps",
        "acceptance_probability",
        "uniform_draw",
        "eligible_for_acceptance",
        "accepted",
        "response_sha256",
    }
    rng = random.Random(expected_request_seed)
    for attempt_index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or set(attempt) != attempt_keys:
            raise ValueError("sample attempt schema differs")
        source = attempt.get("proposal_source")
        expected_source = PANEL_ORDER[rng.randrange(len(PANEL_ORDER))]
        if (
            attempt.get("attempt_index") != attempt_index
            or source != expected_source
        ):
            raise ValueError("sample attempt identity differs")
        expected_token_seed = primary.tuple_seed(
            expected_request_seed, "candidate_tokens", attempt_index, source
        )
        if attempt.get("token_seed") != expected_token_seed:
            raise ValueError("sample attempt token_seed differs")
        finish_reason = attempt.get("finish_reason")
        if finish_reason not in {"stop", "max_new_tokens"}:
            raise ValueError("sample attempt finish_reason differs")
        generated_tokens = attempt.get("generated_tokens")
        sampled_tokens = attempt.get("sampled_tokens")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 0
            or generated_tokens > profile["max_new_tokens"]
        ):
            raise ValueError("sample attempt generated_tokens differs")
        if (
            isinstance(sampled_tokens, bool)
            or not isinstance(sampled_tokens, int)
            or not 1 <= sampled_tokens <= profile["max_new_tokens"]
            or (
                phase == "medical"
                and sampled_tokens
                != generated_tokens + (1 if finish_reason == "stop" else 0)
            )
            or (phase == "benefit" and sampled_tokens != generated_tokens)
        ):
            raise ValueError("sample attempt sampled_tokens differs")
        sequence_logps = attempt.get("sequence_logps")
        if not isinstance(sequence_logps, dict) or set(sequence_logps) != set(
            PANEL_ORDER
        ):
            raise ValueError("sample attempt sequence_logps differs")
        values = []
        for role in PANEL_ORDER:
            value = sequence_logps[role]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("sample attempt sequence log probability differs")
            values.append(float(value))
        expected_probability = whole_output_acceptance(values)
        probability = attempt.get("acceptance_probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isclose(
                float(probability),
                expected_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("sample attempt acceptance probability differs")
        uniform_draw = attempt.get("uniform_draw")
        expected_uniform_draw = rng.random()
        if (
            isinstance(uniform_draw, bool)
            or not isinstance(uniform_draw, (int, float))
            or not 0.0 <= float(uniform_draw) < 1.0
            or float(uniform_draw) != expected_uniform_draw
        ):
            raise ValueError("sample attempt uniform draw differs")
        expected_eligible = not (
            phase == "medical" and finish_reason != "stop"
        )
        if attempt.get("eligible_for_acceptance") is not expected_eligible:
            raise ValueError("sample attempt eligibility differs")
        expected_accepted = expected_eligible and float(uniform_draw) < float(
            probability
        )
        if attempt.get("accepted") is not expected_accepted:
            raise ValueError("sample attempt acceptance decision differs")
        response_sha256 = attempt.get("response_sha256")
        if (
            not isinstance(response_sha256, str)
            or len(response_sha256) != 64
            or any(character not in "0123456789abcdef" for character in response_sha256)
        ):
            raise ValueError("sample attempt response hash differs")
        if attempt_index + 1 < len(attempts) and expected_accepted:
            raise ValueError("sample continues after an accepted attempt")
    if accepted:
        if not attempts[-1].get("accepted") or sample.get("finish_reason") != "stop":
            raise ValueError("accepted sample terminal state differs")
        terminal = attempts[-1]
        response = sample.get("response")
        if not isinstance(response, str):
            raise ValueError("accepted sample response is not a string")
        if (
            sample.get("accepted_source") != terminal["proposal_source"]
            or sample.get("response_sha256") != terminal["response_sha256"]
            or sample.get("response_sha256")
            != _sha256(response.encode("utf-8"))
            or sample.get("generated_tokens") != terminal["generated_tokens"]
        ):
            raise ValueError("accepted sample does not bind its terminal attempt")
        if phase == "benefit":
            prediction = primary.validate_prediction(
                sample.get("response"),
                profile["intent_labels"],
                profile["slot_labels"],
            )
            if sample.get("prediction") != prediction:
                raise ValueError("accepted MASSIVE prediction differs")
    else:
        if len(attempts) != MAX_ATTEMPTS or sample.get("finish_reason") != "abstain":
            raise ValueError("abstention did not exhaust the attempt deadline")
        if any(attempt.get("accepted") for attempt in attempts):
            raise ValueError("abstention contains an accepted attempt")
        if (
            sample.get("response") != ""
            or sample.get("response_sha256") != _sha256(b"")
            or sample.get("generated_tokens") != 0
        ):
            raise ValueError("abstention payload differs")


def _load_shards(
    stream_root, stream_fingerprint, requests, phase, profile, require_complete
):
    shard_root = stream_root / "shards"
    if os.path.lexists(shard_root):
        if shard_root.is_symlink() or not shard_root.is_dir():
            raise ValueError("whole-output shard root is unsafe")
    else:
        shard_root.mkdir(parents=True)
    expected_names = [_request_shard_name(request) for request in requests]
    actual_names = sorted(path.name for path in shard_root.iterdir())
    unknown = sorted(set(actual_names) - set(expected_names))
    if unknown:
        raise ValueError(f"unknown whole-output shard(s): {unknown[:3]}")
    samples = []
    missing = []
    for request, name in zip(requests, expected_names):
        path = shard_root / name
        if not os.path.lexists(path):
            missing.append((request, path))
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"whole-output shard is unsafe: {name}")
        payload, _ = primary.load_json_regular(path, f"whole-output shard {name}")
        body = _verify_seal(payload, f"shard {name}")
        expected_runtime = {
            "torch": primary.PINNED_TORCH_VERSION,
            "transformers": primary.PINNED_TRANSFORMERS_VERSION,
            "peft": primary.PINNED_PEFT_VERSION,
            "xgrammar": primary.PINNED_XGRAMMAR_VERSION,
        }
        if set(body) != {"stream_fingerprint", "runtime", "sample"}:
            raise ValueError(f"shard schema differs: {name}")
        if body.get("stream_fingerprint") != stream_fingerprint:
            raise ValueError(f"shard stream fingerprint differs: {name}")
        if body.get("runtime") != expected_runtime:
            raise ValueError(f"shard runtime differs: {name}")
        sample = body.get("sample")
        if not isinstance(sample, dict):
            raise ValueError(f"shard lacks sample: {name}")
        _audit_sample(sample, request, phase, profile)
        samples.append(sample)
    if require_complete and missing:
        raise ValueError(f"whole-output stream is incomplete: {len(missing)} missing")
    return samples, missing


def _write_generation(stream_root, meta, samples):
    payload = _seal(
        {
            "meta": {**meta, "stream_fingerprint": _sha256(_canonical(meta))},
            "summary": summarize_samples(samples),
            "samples": samples,
        }
    )
    path = stream_root / "generation.json"
    if os.path.lexists(path):
        observed, _ = primary.load_json_regular(path, "whole-output generation")
        _verify_seal(observed, "whole-output generation")
        if observed != payload:
            raise ValueError("existing whole-output generation differs from shards")
    else:
        primary.atomic_write_json(path, payload)
    return path


def _run(args):
    process_started = time.perf_counter()
    source = primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=True
    )
    source_binding = _source_manifest_binding(args.source_protocol_manifest)
    if args.phase == "benefit":
        profile, records = primary.load_massive_prompts(source, "benefit")
    else:
        profile, records = primary.load_medical_prompts(source)
    # Whole-output rejection needs stochastic proposal distributions even though
    # the primary MASSIVE endpoint was greedy.
    profile = dict(profile)
    profile["temperature"] = TEMPERATURE
    all_requests = _expanded_requests(records, profile["n_samples"])
    requests = select_requests(args.phase, args.stage, all_requests)
    record_by_id = {record["question_id"]: record for record in records}
    meta = _stream_meta(source_binding, args.phase, args.stage, profile, requests)
    stream_fingerprint = _sha256(_canonical(meta))
    stream_root = Path(args.output_root).resolve() / args.stage / args.phase

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "WHOLE_OUTPUT_CONSENSUS_PREFLIGHT_VALID",
                    "phase": args.phase,
                    "stage": args.stage,
                    "requested_n": len(requests),
                    "maximum_candidate_attempts": len(requests) * MAX_ATTEMPTS,
                    "gpu_jobs": 0,
                    "external_api_calls": 0,
                    "stream_fingerprint": stream_fingerprint,
                },
                sort_keys=True,
            )
        )
        return 0

    stream_root.mkdir(parents=True, exist_ok=True)
    samples, missing = _load_shards(
        stream_root,
        stream_fingerprint,
        requests,
        args.phase,
        profile,
        require_complete=args.audit_only,
    )
    if args.audit_only:
        generation = _write_generation(stream_root, meta, samples)
        print(
            json.dumps(
                {
                    "status": "WHOLE_OUTPUT_CONSENSUS_AUDITED",
                    "generation": str(generation),
                    **summarize_samples(samples),
                    "gpu_jobs_submitted": 0,
                    "external_api_calls": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if samples and missing and not args.resume_partial:
        raise ValueError(
            "partial whole-output namespace exists; a separately authorized "
            "--resume-partial is required"
        )
    if not missing:
        generation = _write_generation(stream_root, meta, samples)
        print(f"Audited complete whole-output generation: {generation}")
        return 0

    primary.force_offline_environment()
    runtime = primary.require_pinned_runtime(require_cuda=True)
    base_snapshot = primary.resolve_pinned_base_snapshot()
    if args.phase == "benefit":
        tokenizer, _, grammar = primary.load_tokenizer_and_grammar(
            profile, base_snapshot
        )
        grammar_factory = grammar["factory"]
    else:
        from transformers import PreTrainedTokenizerFast

        snapshot_path = primary.verify_pinned_base_snapshot(base_snapshot)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            snapshot_path, local_files_only=True
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("pinned tokenizer has no EOS token")
        grammar_factory = None
    models = primary.load_independent_model_panel(source, args.device, base_snapshot)
    direct_base = models.pop("base")
    del direct_base
    import torch

    torch.cuda.empty_cache()
    stop_ids = primary.stop_token_ids(tokenizer, models["A"])
    shard_root = stream_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    generation_started = time.perf_counter()
    completed_before = len(samples)
    for request, shard_path in missing:
        sample = _sample_request(
            phase=args.phase,
            request=request,
            record=record_by_id[request["question_id"]],
            models=models,
            tokenizer=tokenizer,
            profile=profile,
            device=args.device,
            stop_ids=stop_ids,
            grammar_factory=grammar_factory,
        )
        shard = _seal(
            {
                "stream_fingerprint": stream_fingerprint,
                "runtime": runtime,
                "sample": sample,
            }
        )
        if os.path.lexists(shard_path):
            raise ValueError(f"refusing to overwrite shard: {shard_path}")
        primary.atomic_write_json(shard_path, shard)
    samples, missing = _load_shards(
        stream_root,
        stream_fingerprint,
        requests,
        args.phase,
        profile,
        require_complete=True,
    )
    if missing:
        raise AssertionError("complete shard audit unexpectedly returned missing rows")
    generation = _write_generation(stream_root, meta, samples)
    finished = time.perf_counter()
    generation_elapsed = finished - generation_started
    process_elapsed = finished - process_started
    timing = _seal(
        {
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "phase": args.phase,
            "stage": args.stage,
            "completed_before_process": completed_before,
            "completed_during_process": len(samples) - completed_before,
            # Process elapsed includes source validation, tokenizer/grammar
            # setup, and all five initial model loads.  That is the relevant
            # wall time for a billed GPU smoke, unlike generation-only time.
            "elapsed_seconds": process_elapsed,
            "process_elapsed_seconds": process_elapsed,
            "setup_elapsed_seconds": generation_started - process_started,
            "generation_elapsed_seconds": generation_elapsed,
            "summary": summarize_samples(samples),
        }
    )
    timing_path = stream_root / "timing.json"
    if os.path.lexists(timing_path):
        raise ValueError(f"refusing to overwrite timing artifact: {timing_path}")
    primary.atomic_write_json(timing_path, timing)
    print(
        json.dumps(
            {
                "status": "WHOLE_OUTPUT_CONSENSUS_COMPLETE",
                "generation": str(generation),
                "elapsed_seconds": process_elapsed,
                "generation_elapsed_seconds": generation_elapsed,
                **summarize_samples(samples),
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _self_test():
    requests = [
        {
            "request_index": index,
            "prompt_ordinal": index // 2,
            "question_id": f"q{index // 2}",
            "sample_index": index % 2,
            "prompt_sha256": "a" * 64,
        }
        for index in range(8)
    ]
    selected = select_requests("medical", "smoke", requests)
    assert len(selected) == SMOKE_REQUESTS_PER_PHASE
    assert selected == select_requests("medical", "smoke", list(requests))
    assert select_requests("medical", "full", requests) == requests
    assert math.isclose(whole_output_acceptance([0.0] * 4), 1.0)
    expected = 0.1 / ((0.1 + 0.6 + 0.6 + 0.6) / 4)
    observed = whole_output_acceptance(
        [math.log(value) for value in (0.1, 0.6, 0.6, 0.6)]
    )
    assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    summary = summarize_samples(
        [
            {
                "accepted": True,
                "abstained": False,
                "attempts_used": 2,
                "response": "x",
                "generated_tokens": 3,
                "attempts": [
                    {"generated_tokens": 4, "sampled_tokens": 5},
                    {"generated_tokens": 3, "sampled_tokens": 4},
                ],
            },
            {
                "accepted": False,
                "abstained": True,
                "attempts_used": 20,
                "response": "",
                "generated_tokens": 0,
                "attempts": [
                    {"generated_tokens": 5, "sampled_tokens": 6}
                ]
                * 20,
            },
        ]
    )
    assert summary["coverage"] == 0.5
    assert summary["judge_eligible_medical_n"] == 1
    assert summary["total_candidate_generated_tokens"] == 107
    assert summary["total_candidate_sampled_tokens"] == 129
    assert summary["accepted_output_generated_tokens"] == 3
    print("WHOLE_OUTPUT_CONSENSUS_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--phase", choices=("benefit", "medical"))
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if not args.source_protocol_manifest or not args.output_root or not args.phase:
        parser.error(
            "--source-protocol-manifest, --output-root, and --phase are required"
        )
    if args.preflight_only and args.audit_only:
        parser.error("--preflight-only and --audit-only are mutually exclusive")
    if args.resume_partial and (args.preflight_only or args.audit_only):
        parser.error("--resume-partial is only valid for an authorized generation")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
