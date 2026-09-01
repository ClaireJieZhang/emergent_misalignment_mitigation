#!/usr/bin/env python3
"""Direct generation for Union-SFT and merged-LoRA contextual baselines.

The sampler reuses the frozen sequential protocol's exact prompt banks,
tokenizer, local base snapshot, manual Transformers/PEFT backend, grammar, and
generation profiles.  It is intentionally separate from the frozen primary
sampler and is never eligible to alter that experiment's decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PRIMARY_PATH = (
    SCRIPT_DIR
    / "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_mmu_primary_sampler_for_direct_baseline_v1", PRIMARY_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load primary sampler: {PRIMARY_PATH}")
primary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(primary)


PROTOCOL_ID = "massive_medical_composition_baselines_v1"
ALLOWED_MODELS = {
    "pi_union": "unique_source_union_sft_A_plus_B_once_each",
    "pi_merge": "equal_weight_four_way_lora_delta_merge",
}
OUTPUT_SEAL = "payload_sha256"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seal(body):
    result = dict(body)
    result.pop(OUTPUT_SEAL, None)
    result[OUTPUT_SEAL] = digest(canonical(result))
    return result


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(OUTPUT_SEAL, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} seal differs")
    return body


def adapter_artifacts(path):
    root = os.path.abspath(path)
    config_path = os.path.join(root, "adapter_config.json")
    weights = [
        candidate
        for candidate in (
            os.path.join(root, "adapter_model.safetensors"),
            os.path.join(root, "adapter_model.bin"),
        )
        if os.path.isfile(candidate)
    ]
    if not os.path.isfile(config_path) or len(weights) != 1:
        raise ValueError("adapter requires config plus exactly one weight artifact")
    artifacts = []
    for artifact in (config_path, weights[0]):
        artifacts.append(
            {
                "name": os.path.basename(artifact),
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            }
        )
    return artifacts, digest(canonical(artifacts))


def audit_adapter(adapter_path, training_config, model_id):
    with open(
        os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8"
    ) as handle:
        adapter = json.load(handle)
    if model_id == "pi_union":
        if not training_config:
            raise ValueError("pi_union requires its frozen training configuration")
        with open(training_config, encoding="utf-8") as handle:
            training = yaml.safe_load(handle)
        expected = training.get("lora") or {}
        if (
            training.get("base_model") != primary.BASE_MODEL
            or training.get("base_model_revision") != primary.BASE_REVISION
            or str(adapter.get("peft_type", "")).upper() != "LORA"
            or adapter.get("r") != expected.get("rank")
            or adapter.get("lora_alpha") != expected.get("alpha")
            or set(adapter.get("target_modules", []))
            != set(expected.get("target_modules", []))
        ):
            raise ValueError(
                "Union-SFT adapter/training configuration differs from the plan"
            )
    else:
        training = {
            "base_model": primary.BASE_MODEL,
            "base_model_revision": primary.BASE_REVISION,
        }
        if training_config:
            raise ValueError("pi_merge has no training configuration")
        expected_targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if (
            str(adapter.get("peft_type", "")).upper() != "LORA"
            or str(adapter.get("task_type", "")).upper() != "CAUSAL_LM"
            or adapter.get("r") != 64
            or adapter.get("bias") != "none"
            or set(adapter.get("target_modules", [])) != expected_targets
            or adapter.get("base_model_name_or_path") != primary.BASE_MODEL
            or adapter.get("revision") != primary.BASE_REVISION
            or adapter.get("use_dora", False) is not False
            or adapter.get("use_rslora", False) is not False
        ):
            raise ValueError("materialized LoRA-merge adapter contract differs")
    artifacts, fingerprint = adapter_artifacts(adapter_path)
    return training, adapter, artifacts, fingerprint


def audit_adapter_manifest(
    path, model_id, fingerprint, training_config, adapter_path, adapter, artifacts
):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    body = verify_seal(payload, "contextual baseline model manifest")
    common_differs = (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("model_id") != model_id
        or body.get("adapter_fingerprint") != fingerprint
        or body.get("adapter_artifacts") != artifacts
        or body.get("primary_gate_eligible") is not False
    )
    if model_id == "pi_union":
        differs = (
            common_differs
            or body.get("training_config_sha256") != sha256_file(training_config)
            or body.get("adapter_path") != os.path.abspath(adapter_path)
            or body.get("lora_rank") != 16
            or body.get("lora_alpha") != 16
            or body.get("training_rows") != 64734
            or body.get("training_steps") != 1079
        )
    else:
        contract = {
            "peft_type": adapter.get("peft_type"),
            "task_type": adapter.get("task_type"),
            "r": adapter.get("r"),
            "bias": adapter.get("bias"),
            "target_modules": sorted(adapter.get("target_modules") or []),
            "base_model_name_or_path": adapter.get("base_model_name_or_path"),
            "revision": adapter.get("revision"),
            "use_dora": adapter.get("use_dora", False),
            "use_rslora": adapter.get("use_rslora", False),
        }
        differs = (
            common_differs
            or body.get("protocol") != PROTOCOL_ID
            or body.get("model_name") != model_id
            or body.get("adapter_dir") != os.path.abspath(adapter_path)
            or body.get("adapter_config_contract") != contract
            or body.get("combination_type") != "cat"
            or body.get("weights") != [0.25, 0.25, 0.25, 0.25]
            or body.get("source_order")
            != ["pi_A", "pi_B1", "pi_B2", "pi_B3"]
            or body.get("source_rank") != 16
            or body.get("effective_rank") != 64
        )
    if differs:
        raise ValueError("contextual baseline model manifest binding differs")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[OUTPUT_SEAL],
    }


def load_one_model(adapter_path, adapter_name, device, base_snapshot):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    snapshot_path = primary.verify_pinned_base_snapshot(base_snapshot)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    return model


def generate_one(
    *,
    model,
    model_id,
    record,
    sample_index,
    tokenizer,
    profile,
    device,
    stop_ids,
    grammar_factory,
):
    import torch
    import torch.nn.functional as functional

    prompt_ids = primary.make_prompt_ids(tokenizer, record)
    if len(prompt_ids) + profile["max_new_tokens"] > profile["max_context"]:
        raise ValueError(f"request exceeds frozen context: {record['question_id']}")
    state = primary.prefill_cached_reference(model, prompt_ids, device)
    grammar = grammar_factory() if grammar_factory is not None else None
    response_ids = []
    finish_reason = "max_new_tokens"
    rng_seed = primary.tuple_seed(
        primary.GENERATION_SEED,
        model_id,
        record["question_id"],
        sample_index,
    )
    generator = None
    if profile["temperature"] == 1:
        generator = torch.Generator(device=device)
        generator.manual_seed(rng_seed)
    elif profile["temperature"] != 0:
        raise ValueError("direct baseline requested a non-frozen temperature")

    for token_index in range(profile["max_new_tokens"]):
        logp = functional.log_softmax(state["next_logits"].float(), dim=-1)
        target_logp = primary.apply_grammar_mask_then_normalize(logp, grammar)
        token_id = (
            int(torch.argmax(target_logp).item())
            if profile["temperature"] == 0
            else int(
                torch.multinomial(
                    torch.exp(target_logp), 1, generator=generator
                ).item()
            )
        )
        terminated = False
        if grammar is not None:
            if not grammar["matcher"].accept_token(token_id):
                raise ValueError("XGrammar rejected a token admitted by its own mask")
            response_ids.append(token_id)
            terminated = grammar["matcher"].is_terminated()
        elif token_id in stop_ids:
            terminated = True
        else:
            response_ids.append(token_id)
        if terminated:
            finish_reason = "stop"
            break
        if token_index + 1 < profile["max_new_tokens"]:
            state = primary.step_cached_reference(
                model, token_id, state["cache"], device
            )

    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    sample = {
        "question_id": record["question_id"],
        "sample_index": sample_index,
        "prompt_sha256": record["prompt_sha256"],
        "response": response,
        "response_sha256": digest(response.encode("utf-8")),
        "finish_reason": finish_reason,
        "generated_tokens": len(response_ids),
        "rng_seed": rng_seed,
    }
    if grammar is not None:
        if finish_reason != "stop":
            raise ValueError("structured direct baseline did not terminate")
        sample["prediction"] = primary.validate_prediction(
            response, profile["intent_labels"], profile["slot_labels"]
        )
    elif finish_reason != "stop":
        raise ValueError("medical direct baseline truncated under the all-stop profile")
    sample["sample_sha256"] = digest(canonical(sample))
    return sample


def expected_keys(records, n_samples):
    return [
        (record["question_id"], sample_index, record["prompt_sha256"])
        for record in records
        for sample_index in range(n_samples)
    ]


def audit_generation(path, expected_meta, records, phase, profile):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    body = verify_seal(payload, "direct contextual baseline generation")
    meta = dict(body.get("meta") or {})
    meta.pop("created_at", None)
    if meta != expected_meta:
        raise ValueError("direct contextual baseline generation metadata differs")
    samples = body.get("samples")
    keys = expected_keys(records, profile["n_samples"])
    if not isinstance(samples, list) or len(samples) != len(keys):
        raise ValueError("direct contextual baseline generation count differs")
    for sample, key in zip(samples, keys):
        if (
            sample.get("question_id"),
            sample.get("sample_index"),
            sample.get("prompt_sha256"),
        ) != key:
            raise ValueError("direct contextual baseline sample order differs")
        copy = dict(sample)
        observed = copy.pop("sample_sha256", None)
        if observed != digest(canonical(copy)):
            raise ValueError("direct contextual baseline sample seal differs")
        if sample.get("finish_reason") != "stop":
            raise ValueError("direct contextual baseline contains a truncation")
        if phase == "benefit" and not isinstance(sample.get("prediction"), dict):
            raise ValueError("direct MASSIVE sample lacks a prediction")
    return body


def score_benefit(generation_path, answers_path, output_path):
    with open(generation_path, encoding="utf-8") as handle:
        generation = json.load(handle)
    generation_body = verify_seal(generation, "direct benefit generation")
    with open(answers_path, encoding="utf-8") as handle:
        answers_payload = json.load(handle)
    answers_body = primary.verify_seal(
        answers_payload, OUTPUT_SEAL, "sequential benefit answers"
    )
    answers = answers_body.get("answers")
    samples = generation_body.get("samples")
    if not isinstance(answers, list) or len(answers) != len(samples):
        raise ValueError("benefit answer/generation counts differ")
    correct = []
    for sample, answer in zip(samples, answers):
        if sample.get("question_id") != answer.get("question_id"):
            raise ValueError("benefit answer/generation order differs")
        correct.append(sample["prediction"]["intent"] == answer["intent"])
    result = seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "analysis_scope": "contextual_post_hoc_not_gated",
            "primary_gate_eligible": False,
            "model_id": generation_body["meta"]["model_id"],
            "requested_n": len(correct),
            "accepted_n": len(correct),
            "abstained_n": 0,
            "correct_n": sum(correct),
            "intent_accuracy": sum(correct) / len(correct),
            "correct_by_request": correct,
            "generation_file_sha256": sha256_file(generation_path),
            "answers_file_sha256": sha256_file(answers_path),
        }
    )
    primary.atomic_write_json(output_path, result)
    return result


def run(args):
    if args.model_id not in ALLOWED_MODELS:
        raise ValueError(f"model must be one of {sorted(ALLOWED_MODELS)}")
    source = primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=True
    )
    if args.phase == "benefit":
        profile, records = primary.load_massive_prompts(source, "benefit")
    else:
        profile, records = primary.load_medical_prompts(source)
    training, adapter, artifacts, fingerprint = audit_adapter(
        args.adapter_path, args.training_config, args.model_id
    )
    manifest = audit_adapter_manifest(
        args.adapter_manifest,
        args.model_id,
        fingerprint,
        args.training_config,
        args.adapter_path,
        adapter,
        artifacts,
    )
    meta = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_gate_eligible": False,
        "model_id": args.model_id,
        "construction": ALLOWED_MODELS[args.model_id],
        "model_path": os.path.abspath(args.adapter_path),
        "adapter_fingerprint": fingerprint,
        "adapter_artifacts": artifacts,
        "adapter_manifest": manifest,
        "training_config_sha256": (
            sha256_file(args.training_config) if args.training_config else None
        ),
        "base_model": training["base_model"],
        "base_model_revision": training["base_model_revision"],
        "source_protocol_manifest_sha256": sha256_file(
            args.source_protocol_manifest
        ),
        "phase": args.phase,
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
        "backend": "single_transformers_peft_model_manual_cached_decode",
        "same_backend_family_as_primary": True,
        "request_keys": [
            [question_id, sample_index]
            for question_id, sample_index, _ in expected_keys(
                records, profile["n_samples"]
            )
        ],
    }
    output_path = os.path.abspath(args.output_file)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "DIRECT_CONTEXTUAL_BASELINE_PREFLIGHT_VALID",
                    "model_id": args.model_id,
                    "phase": args.phase,
                    "requested_n": len(meta["request_keys"]),
                    "gpu_jobs": 0,
                    "external_api_calls": 0,
                    "plan_sha256": digest(canonical(meta)),
                },
                sort_keys=True,
            )
        )
        return 0
    if os.path.isfile(output_path):
        audit_generation(output_path, meta, records, args.phase, profile)
        print(f"Audited complete direct baseline generation: {output_path}")
        return 0
    if args.audit_only:
        raise ValueError("direct contextual baseline generation is absent")

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
        grammar_factory = None
    model = load_one_model(
        args.adapter_path, args.model_id, args.device, base_snapshot
    )
    stop_ids = primary.stop_token_ids(tokenizer, model)
    samples = []
    started = time.perf_counter()
    for record in records:
        for sample_index in range(profile["n_samples"]):
            samples.append(
                generate_one(
                    model=model,
                    model_id=args.model_id,
                    record=record,
                    sample_index=sample_index,
                    tokenizer=tokenizer,
                    profile=profile,
                    device=args.device,
                    stop_ids=stop_ids,
                    grammar_factory=grammar_factory,
                )
            )
    payload = seal(
        {
            "meta": {
                **meta,
                "created_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            },
            "runtime": runtime,
            "timing": {"elapsed_seconds": time.perf_counter() - started},
            "samples": samples,
        }
    )
    primary.atomic_write_json(output_path, payload)
    audit_generation(output_path, meta, records, args.phase, profile)
    print(
        json.dumps(
            {
                "status": "DIRECT_CONTEXTUAL_BASELINE_COMPLETE",
                "model_id": args.model_id,
                "phase": args.phase,
                "requested_n": len(samples),
                "output": output_path,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def self_test():
    sample = {
        "question_id": "q",
        "sample_index": 0,
        "prompt_sha256": "a" * 64,
        "response": "{}",
        "response_sha256": digest(b"{}"),
        "finish_reason": "stop",
        "generated_tokens": 2,
        "rng_seed": 1,
        "prediction": {"intent": "x", "slots": []},
    }
    sample["sample_sha256"] = digest(canonical(sample))
    copy = dict(sample)
    observed = copy.pop("sample_sha256")
    assert observed == digest(canonical(copy))
    assert set(ALLOWED_MODELS) == {"pi_union", "pi_merge"}
    print("DIRECT_CONTEXTUAL_BASELINE_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--model-id", choices=tuple(ALLOWED_MODELS))
    parser.add_argument("--adapter-path")
    parser.add_argument("--adapter-manifest")
    parser.add_argument("--training-config")
    parser.add_argument("--phase", choices=("benefit", "medical"))
    parser.add_argument("--output-file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--score-benefit", action="store_true")
    parser.add_argument("--answers-file")
    parser.add_argument("--score-output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.score_benefit:
        if not args.output_file or not args.answers_file or not args.score_output:
            parser.error(
                "--score-benefit requires --output-file, --answers-file, "
                "and --score-output"
            )
        score_benefit(args.output_file, args.answers_file, args.score_output)
        return 0
    required = (
        "source_protocol_manifest",
        "model_id",
        "adapter_path",
        "adapter_manifest",
        "phase",
        "output_file",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    if args.model_id == "pi_union" and not args.training_config:
        parser.error("pi_union requires --training-config")
    if args.model_id == "pi_merge" and args.training_config:
        parser.error("pi_merge must not receive --training-config")
    if args.preflight_only and args.audit_only:
        parser.error("--preflight-only and --audit-only are mutually exclusive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
