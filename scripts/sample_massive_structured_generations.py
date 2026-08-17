#!/usr/bin/env python3
"""Generate deterministic, grammar-constrained MASSIVE JSON predictions.

The sampler accepts only label-free prompt banks from the MASSIVE preparation
workflow.  Base and adapters use the identical prompt and JSON schema.  It
keeps one vLLM base-model load alive across all requested adapters and writes
one sealed, atomic result per model/set for safe resume.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import tempfile

import yaml


PINNED_VLLM_VERSION = "0.11.2"
PINNED_XGRAMMAR_VERSION = "0.1.25"
EXPECTED_SEED = 8172026
EXPECTED_MAX_NEW_TOKENS = 256
EXPECTED_MAX_CONTEXT = 2048


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def adapter_fingerprint(path):
    if path == "BASE":
        return "BASE"
    root = os.path.abspath(path)
    config = os.path.join(root, "adapter_config.json")
    weights = [
        candidate for candidate in (
            os.path.join(root, "adapter_model.safetensors"),
            os.path.join(root, "adapter_model.bin"),
        ) if os.path.isfile(candidate)
    ]
    if not os.path.isfile(config) or len(weights) != 1:
        raise ValueError(
            f"Expected adapter_config.json and one adapter weight file in {root}"
        )
    artifacts = []
    for artifact in [config] + weights:
        artifacts.append(
            {
                "name": os.path.basename(artifact),
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            }
        )
    return sha256_bytes(canonical_json_bytes(artifacts))


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Model must be NAME=BASE or NAME=ADAPTER_PATH: {spec!r}")
    name, path = (part.strip() for part in spec.split("=", 1))
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name or "") is None or not path:
        raise ValueError(f"Invalid model specification: {spec!r}")
    if path.upper() == "BASE":
        path = "BASE"
    else:
        path = os.path.abspath(path)
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise ValueError(f"Missing adapter_config.json for {name}: {path}")
    return name, path


def audit_adapter_config(path, training):
    if path == "BASE":
        return
    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as handle:
        adapter = json.load(handle)
    expected = training["lora"]
    if str(adapter.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Adapter is not LoRA: {path}")
    if adapter.get("r") != expected["rank"]:
        raise ValueError(f"Adapter rank mismatch: {path}")
    if adapter.get("lora_alpha") != expected["alpha"]:
        raise ValueError(f"Adapter alpha mismatch: {path}")
    if set(adapter.get("target_modules", [])) != set(expected["target_modules"]):
        raise ValueError(f"Adapter target_modules mismatch: {path}")


def audit_vllm_target_compatibility(base_model, target_modules):
    if "qwen2" in base_model.casefold():
        unsupported = sorted({"lm_head", "embed_tokens"} & set(target_modules))
        if unsupported:
            raise ValueError(
                "Pinned vLLM cannot load Qwen2 LoRA targets " + repr(unsupported)
            )


def load_prompt_bank(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    prompts = payload.get("prompts")
    if not isinstance(meta, dict) or not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt bank must contain nonempty meta and prompts")
    if meta.get("dataset") != "MASSIVE" or meta.get("locale") != "en-US":
        raise ValueError("Prompt bank is not pinned MASSIVE English")
    if meta.get("contains_gold_labels") is not False:
        raise ValueError("Refusing a prompt bank not explicitly free of gold labels")
    if meta.get("role") not in {"checkpoint_selection", "sealed_final"}:
        raise ValueError("Prompt bank has an invalid experimental role")
    intents = meta.get("intent_labels")
    slots = meta.get("slot_labels")
    if (
        not isinstance(intents, list) or len(intents) != 60
        or len(set(intents)) != 60
        or not isinstance(slots, list) or len(slots) != 55
        or len(set(slots)) != 55
    ):
        raise ValueError("Prompt bank ontology size or uniqueness differs")
    ontology_hash = sha256_bytes(
        canonical_json_bytes({"intent_labels": intents, "slot_labels": slots})
    )
    if meta.get("ontology_sha256") != ontology_hash:
        raise ValueError("Prompt-bank ontology hash mismatch")
    if meta.get("n_questions") != len(prompts):
        raise ValueError("Prompt-bank count differs from metadata")
    forbidden = {
        "intent", "slots", "annot_utt", "answer", "answers", "source_id",
        "scenario", "response",
    }
    seen = set()
    validated = []
    for index, record in enumerate(prompts):
        if not isinstance(record, dict) or forbidden & set(record):
            raise ValueError(f"Prompt {index} contains forbidden answer fields")
        question_id = record.get("question_id")
        prompt = record.get("prompt")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Prompt {index} has no question ID")
        if question_id in seen:
            raise ValueError(f"Duplicate question ID: {question_id}")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Prompt {index} has no prompt text")
        if record.get("set_name") != meta.get("set_name"):
            raise ValueError(f"Prompt {index} set-name mismatch")
        expected_hash = sha256_bytes(canonical_json_bytes({"prompt": prompt}))
        if record.get("prompt_sha256") != expected_hash:
            raise ValueError(f"Prompt {index} hash mismatch")
        seen.add(question_id)
        validated.append(record)
    return meta, validated


def prediction_schema(intent_labels, slot_labels, endpoint="joint_json"):
    if endpoint == "intent_only":
        return {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": intent_labels},
            },
            "required": ["intent"],
            "additionalProperties": False,
        }
    if endpoint != "joint_json":
        raise ValueError(f"Unknown MASSIVE endpoint: {endpoint}")
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": intent_labels},
            "slots": {
                "type": "array",
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": slot_labels},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }


def validate_prediction(response, intent_labels, slot_labels, endpoint="joint_json"):
    try:
        prediction = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Structured decoder emitted invalid JSON: {response!r}") from error
    expected_keys = {"intent"} if endpoint == "intent_only" else {"intent", "slots"}
    if not isinstance(prediction, dict) or set(prediction) != expected_keys:
        raise ValueError("Structured prediction has wrong top-level keys")
    if prediction["intent"] not in intent_labels:
        raise ValueError("Structured prediction escaped the intent ontology")
    if endpoint == "intent_only":
        return prediction
    if endpoint != "joint_json":
        raise ValueError(f"Unknown MASSIVE endpoint: {endpoint}")
    values = prediction["slots"]
    if not isinstance(values, list) or len(values) > 7:
        raise ValueError("Structured prediction has an invalid slots array")
    for slot in values:
        if (
            not isinstance(slot, dict)
            or set(slot) != {"name", "value"}
            or slot["name"] not in slot_labels
            or not isinstance(slot["value"], str)
            or not slot["value"]
        ):
            raise ValueError("Structured prediction has an invalid slot")
    return prediction


def validate_context_lengths(tokenizer, prompts, max_new_tokens, max_context):
    lengths = {}
    for record in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(ids) + max_new_tokens > max_context:
            raise ValueError(
                f"{record['question_id']} exceeds max context: "
                f"{len(ids)}+{max_new_tokens}>{max_context}"
            )
        lengths[record["question_id"]] = len(ids)
    return lengths


def sample_sha256(sample):
    fields = (
        "question_id", "sample_index", "response", "prediction",
        "stop_reason", "n_generated_tokens", "prompt_tokens", "prompt_sha256",
    )
    if not all(field in sample for field in fields):
        raise ValueError("Generation sample lacks checksum fields")
    return sha256_bytes(
        canonical_json_bytes({field: sample[field] for field in fields})
    )


def output_is_complete(
    path, expected_run, fingerprint, prompts, intents, slots, endpoint
):
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    samples = payload.get("samples")
    if not isinstance(meta, dict) or not isinstance(samples, list):
        raise ValueError(f"Existing generation lacks metadata/samples: {path}")
    observed_run = {
        key: value for key, value in meta.items()
        if key not in {"created_at", "generation_fingerprint"}
    }
    if observed_run != expected_run:
        raise ValueError(f"Existing generation provenance differs: {path}")
    if meta.get("generation_fingerprint") != fingerprint:
        raise ValueError(f"Existing generation fingerprint differs: {path}")
    if sha256_bytes(canonical_json_bytes(observed_run)) != fingerprint:
        raise ValueError(f"Existing generation fingerprint is invalid: {path}")
    if len(samples) != len(prompts):
        raise ValueError(f"Existing generation row count differs: {path}")
    for prompt, sample in zip(prompts, samples):
        if (
            sample.get("question_id") != prompt["question_id"]
            or sample.get("prompt_sha256") != prompt["prompt_sha256"]
        ):
            raise ValueError(f"Existing generation prompt order differs: {path}")
        if sample.get("sample_index") != 0:
            raise ValueError(f"Existing generation is not greedy pass@1: {path}")
        if validate_prediction(
            sample.get("response"), intents, slots, endpoint=endpoint
        ) != sample.get("prediction"):
            raise ValueError(f"Existing parsed prediction differs: {path}")
        if sample.get("result_sha256") != sample_sha256(sample):
            raise ValueError(f"Existing generation sample hash differs: {path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=EXPECTED_MAX_NEW_TOKENS)
    parser.add_argument("--max_context", type=int, default=EXPECTED_MAX_CONTEXT)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    if (
        args.max_new_tokens != EXPECTED_MAX_NEW_TOKENS
        or args.max_context != EXPECTED_MAX_CONTEXT
        or args.seed != EXPECTED_SEED
    ):
        parser.error("MASSIVE pilot inference budget and seed are frozen")

    with open(args.training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    base_model = training.get("base_model")
    base_revision = training.get("base_model_revision")
    lora_rank = training.get("lora", {}).get("rank")
    target_modules = training.get("lora", {}).get("target_modules")
    if not all((base_model, base_revision, lora_rank, target_modules)):
        raise ValueError("Training config lacks pinned model or LoRA fields")
    models = [parse_model_spec(spec) for spec in args.model]
    if len({name for name, _ in models}) != len(models):
        raise ValueError("Duplicate model names")
    for _, path in models:
        audit_adapter_config(path, training)
    if any(path != "BASE" for _, path in models):
        audit_vllm_target_compatibility(base_model, target_modules)

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_model, revision=base_revision
    )
    banks = []
    set_names = set()
    for prompt_file in args.prompt_file:
        prompt_file = os.path.abspath(prompt_file)
        meta, prompts = load_prompt_bank(prompt_file)
        if meta["set_name"] in set_names:
            raise ValueError(f"Duplicate prompt set: {meta['set_name']}")
        set_names.add(meta["set_name"])
        schemas = {
            endpoint: prediction_schema(
                meta["intent_labels"], meta["slot_labels"], endpoint=endpoint
            )
            for endpoint in ("joint_json", "intent_only")
        }
        banks.append(
            {
                "path": prompt_file,
                "meta": meta,
                "prompts": prompts,
                "schemas": schemas,
                "schema_sha256": {
                    endpoint: sha256_bytes(canonical_json_bytes(schema))
                    for endpoint, schema in schemas.items()
                },
                "prompt_lengths": validate_context_lengths(
                    tokenizer, prompts, args.max_new_tokens, args.max_context
                ),
            }
        )

    pending = {name: [] for name, _ in models}
    provenance = {}
    for name, path in models:
        for bank in banks:
            meta = bank["meta"]
            prompts = bank["prompts"]
            for endpoint in ("joint_json", "intent_only"):
                run = {
                    "schema_version": 1,
                    "generator": "vllm_xgrammar_json",
                    "endpoint": endpoint,
                    "set_name": meta["set_name"],
                    "role": meta["role"],
                    "model_name": name,
                    "model_path": "BASE" if path == "BASE" else os.path.abspath(path),
                    "model_fingerprint": adapter_fingerprint(path),
                    "base_model": base_model,
                    "base_model_revision": base_revision,
                    "prompt_file_sha256": sha256_file(bank["path"]),
                    "question_ids": [record["question_id"] for record in prompts],
                    "prompt_sha256": [record["prompt_sha256"] for record in prompts],
                    "ontology_sha256": meta["ontology_sha256"],
                    "json_schema_sha256": bank["schema_sha256"][endpoint],
                    "structured_backend": "xgrammar",
                    "vllm_version": PINNED_VLLM_VERSION,
                    "xgrammar_version": PINNED_XGRAMMAR_VERSION,
                    "temperature": 0.0,
                    "n_samples": 1,
                    "max_new_tokens": args.max_new_tokens,
                    "max_context": args.max_context,
                    "seed": args.seed,
                    "same_prompt_all_models": True,
                    "selection_uses_joint_json_only": True,
                }
                fingerprint = sha256_bytes(canonical_json_bytes(run))
                provenance[(meta["set_name"], name, endpoint)] = (run, fingerprint)
                suffix = "" if endpoint == "joint_json" else "__intent_only"
                output = os.path.join(
                    args.output_dir, f"{meta['set_name']}__{name}{suffix}.json"
                )
                if not args.preflight_only and output_is_complete(
                    output, run, fingerprint, prompts,
                    meta["intent_labels"], meta["slot_labels"], endpoint,
                ):
                    print(
                        f"Audited complete output; skipping "
                        f"{meta['set_name']}/{name}/{endpoint}"
                    )
                else:
                    pending[name].append((bank, output, endpoint))
    if not args.preflight_only and not any(pending.values()):
        print("All requested MASSIVE generation files are complete.")
        return

    import importlib.metadata
    import vllm
    from vllm import SamplingParams
    from vllm.config.structured_outputs import StructuredOutputsConfig
    from vllm.sampling_params import StructuredOutputsParams

    if vllm.__version__ != PINNED_VLLM_VERSION:
        raise ValueError(
            f"MASSIVE protocol requires vLLM {PINNED_VLLM_VERSION}, "
            f"found {vllm.__version__}"
        )
    xgrammar_version = importlib.metadata.version("xgrammar")
    if xgrammar_version != PINNED_XGRAMMAR_VERSION:
        raise ValueError(
            f"MASSIVE protocol requires XGrammar {PINNED_XGRAMMAR_VERSION}, "
            f"found {xgrammar_version}"
        )
    structured_config = StructuredOutputsConfig(
        backend="xgrammar", disable_fallback=True
    )
    probe = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(
            json=banks[0]["schemas"]["joint_json"], disable_fallback=True
        ),
    )
    if (
        structured_config.backend != "xgrammar"
        or structured_config.disable_fallback is not True
        or probe.structured_outputs is None
        or probe.structured_outputs.disable_fallback is not True
    ):
        raise RuntimeError("Pinned structured-output API changed")
    if args.preflight_only:
        print(
            f"MASSIVE structured preflight passed: {len(banks)} set(s), "
            f"{sum(len(bank['prompts']) for bank in banks)} prompts, "
            f"{len(models)} model(s)."
        )
        return

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    os.makedirs(args.output_dir, exist_ok=True)
    llm = LLM(
        model=base_model,
        revision=base_revision,
        tokenizer_revision=base_revision,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
        structured_outputs_config=structured_config,
    )
    lora_id = 1
    for name, path in models:
        request = None
        if path != "BASE":
            request = LoRARequest(name, lora_id, path)
            lora_id += 1
        for bank, output_path, endpoint in pending[name]:
            prompts = bank["prompts"]
            messages = [
                [{"role": "user", "content": record["prompt"]}]
                for record in prompts
            ]
            sampling = SamplingParams(
                temperature=0.0,
                n=1,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
                structured_outputs=StructuredOutputsParams(
                    json=bank["schemas"][endpoint], disable_fallback=True
                ),
            )
            print(
                f"Generating {name}/{endpoint} on {len(prompts)} "
                f"{bank['meta']['set_name']} rows"
            )
            outputs = llm.chat(
                messages,
                sampling,
                lora_request=request,
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != len(prompts):
                raise RuntimeError("vLLM returned an incomplete MASSIVE batch")
            samples = []
            for record, output in zip(prompts, outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected one greedy structured completion")
                completion = output.outputs[0]
                response = completion.text
                prediction = validate_prediction(
                    response,
                    bank["meta"]["intent_labels"],
                    bank["meta"]["slot_labels"],
                    endpoint=endpoint,
                )
                finish_reason = getattr(completion, "finish_reason", None)
                sample = {
                    "question_id": record["question_id"],
                    "sample_index": 0,
                    "response": response,
                    "prediction": prediction,
                    "stop_reason": (
                        "max_new_tokens" if finish_reason == "length" else finish_reason
                    ) or "unknown",
                    "n_generated_tokens": len(
                        list(getattr(completion, "token_ids", None) or [])
                    ),
                    "prompt_tokens": bank["prompt_lengths"][record["question_id"]],
                    "prompt_sha256": record["prompt_sha256"],
                }
                sample["result_sha256"] = sample_sha256(sample)
                samples.append(sample)
            run, fingerprint = provenance[
                (bank["meta"]["set_name"], name, endpoint)
            ]
            payload = {
                "meta": {
                    **run,
                    "generation_fingerprint": fingerprint,
                    "created_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
                "samples": samples,
            }
            atomic_write_json(output_path, payload)
            if not output_is_complete(
                output_path, run, fingerprint, prompts,
                bank["meta"]["intent_labels"], bank["meta"]["slot_labels"],
                endpoint,
            ):
                raise RuntimeError(f"Generation audit failed after write: {output_path}")


if __name__ == "__main__":
    main()
