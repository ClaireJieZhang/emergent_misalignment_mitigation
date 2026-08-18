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
import sys
import tempfile

import yaml


PINNED_VLLM_VERSION = "0.11.2"
PINNED_XGRAMMAR_VERSION = "0.1.25"
EXPECTED_SEED = 8172026
EXPECTED_MAX_NEW_TOKENS = 256
EXPECTED_MAX_CONTEXT = 2048
LEGACY_STRUCTURED_CONSTRAINT_PROFILE = "enum_v1"
STRICT_STRUCTURED_CONSTRAINT_PROFILE = "const_tree_v2"
STRUCTURED_CONSTRAINT_PROFILES = (
    LEGACY_STRUCTURED_CONSTRAINT_PROFILE,
    STRICT_STRUCTURED_CONSTRAINT_PROFILE,
)
FAILURE_EVIDENCE_SCHEMA_VERSION = 1
RECORDED_LEGACY_HYBRID_INTENT_PROBES = (
    "alarm_addcontact",
    "alarm_createoradd",
    "calendar_recipe",
    "cooking_remove",
)
RECORDED_LEGACY_HYBRID_SLOT_PROBES = (
    "alarm_name",
    "app_type",
    "cooking_name",
)


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


def balanced_const_tree(labels):
    """Encode an exact finite string set without one large flat alternation.

    The pinned XGrammar/Qwen token matcher admits recorded hybrid nonmembers
    under the legacy flat ``enum`` schema. A balanced tree keeps every
    ``anyOf`` fan-out at two while preserving exactly the same finite label
    language.
    """
    values = list(labels)
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Structured label set must contain unique nonempty strings")

    def build(start, end):
        if end - start == 1:
            return {"const": values[start]}
        middle = start + (end - start) // 2
        return {"anyOf": [build(start, middle), build(middle, end)]}

    tree = build(0, len(values))
    if const_tree_labels(tree) != values:
        raise AssertionError("Balanced const tree changed the structured label set")
    return tree


def const_tree_labels(tree):
    """Audit a balanced const tree and return its ordered leaf language."""
    if not isinstance(tree, dict):
        raise ValueError("Const-tree node is not an object")
    if set(tree) == {"const"}:
        value = tree["const"]
        if not isinstance(value, str) or not value:
            raise ValueError("Const-tree leaf is not a nonempty string")
        return [value]
    if set(tree) != {"anyOf"}:
        raise ValueError("Const-tree node has unexpected keys")
    children = tree["anyOf"]
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("Const-tree internal node is not binary")
    return const_tree_labels(children[0]) + const_tree_labels(children[1])


def label_schema(labels, structured_constraint_profile):
    if structured_constraint_profile == LEGACY_STRUCTURED_CONSTRAINT_PROFILE:
        return {"type": "string", "enum": labels}
    if structured_constraint_profile == STRICT_STRUCTURED_CONSTRAINT_PROFILE:
        return balanced_const_tree(labels)
    raise ValueError(
        f"Unknown structured constraint profile: {structured_constraint_profile}"
    )


def prediction_schema(
    intent_labels,
    slot_labels,
    endpoint="joint_json",
    structured_constraint_profile=LEGACY_STRUCTURED_CONSTRAINT_PROFILE,
):
    intent_schema = label_schema(intent_labels, structured_constraint_profile)
    if endpoint == "intent_only":
        return {
            "type": "object",
            "properties": {
                "intent": intent_schema,
            },
            "required": ["intent"],
            "additionalProperties": False,
        }
    if endpoint != "joint_json":
        raise ValueError(f"Unknown MASSIVE endpoint: {endpoint}")
    return {
        "type": "object",
        "properties": {
            "intent": intent_schema,
            "slots": {
                "type": "array",
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": label_schema(
                            slot_labels, structured_constraint_profile
                        ),
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


def _xgrammar_accepts_tokenized_text(
    xgrammar_module, compiled_grammar, tokenizer, text
):
    """Exercise the same token-level matcher used by the pinned backend."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids or any(not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("Pinned tokenizer did not return nonempty integer token IDs")
    matcher = xgrammar_module.GrammarMatcher(
        compiled_grammar, terminate_without_stop_token=True
    )
    for token_id in token_ids:
        if not matcher.accept_token(token_id):
            return False
    return matcher.is_terminated()


def audit_balanced_xgrammar_frontier(grammar_text, labels, label_kind):
    """Verify the generated EBNF retains one binary rule frontier per label set."""
    rules = {}
    for line in grammar_text.splitlines():
        if "::=" not in line:
            continue
        name, body = line.split("::=", 1)
        name = name.strip()
        if not name or name in rules:
            raise ValueError("Pinned XGrammar emitted malformed or duplicate rules")
        rules[name] = body

    encoded_labels = {
        label: json.dumps(json.dumps(label, ensure_ascii=False), ensure_ascii=False)
        for label in labels
    }
    occurrences = {
        label: [name for name, body in rules.items() if encoded in body]
        for label, encoded in encoded_labels.items()
    }
    if any(len(names) != 1 for names in occurrences.values()):
        raise ValueError(
            f"Pinned XGrammar changed the {label_kind} const-leaf inventory"
        )
    prefixes = set()
    for names in occurrences.values():
        name = names[0]
        if "_case_" not in name:
            raise ValueError(
                f"Pinned XGrammar flattened the {label_kind} ontology frontier"
            )
        prefixes.add(name.split("_case_", 1)[0])
    if len(prefixes) != 1:
        raise ValueError(
            f"Pinned XGrammar split the {label_kind} ontology across frontiers"
        )
    prefix = prefixes.pop()
    frontier = {
        name: body
        for name, body in rules.items()
        if name == prefix or name.startswith(prefix + "_case_")
    }
    if len(frontier) != len(labels) - 1 or any(
        body.count(" | ") != 1 for body in frontier.values()
    ):
        raise ValueError(
            f"Pinned XGrammar did not retain a balanced binary {label_kind} frontier"
        )


def audit_strict_xgrammar_contract(
    xgrammar_module,
    tokenizer,
    model_config,
    intent_labels,
    slot_labels,
    schemas,
):
    """Fail closed unless pinned XGrammar enforces the exact v2 ontology.

    This is deliberately a token-level CPU matcher audit, not merely a schema
    compilation check.  It uses the pinned model vocabulary size and the same
    Hugging Face tokenizer that vLLM will use.  Every ontology leaf must be
    accepted. The legacy matcher must reproduce the recorded hybrid-label
    escapes, while v2 must reject those hybrids and generic nonmembers.
    """
    model_vocab_size = getattr(model_config, "vocab_size", None)
    if not isinstance(model_vocab_size, int) or model_vocab_size <= 0:
        raise ValueError("Pinned model config has no positive integer vocab_size")
    tokenizer_info = xgrammar_module.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=model_vocab_size
    )
    compiler = xgrammar_module.GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled = {}
    legacy_compiled = {}
    for endpoint in ("joint_json", "intent_only"):
        schema = schemas[endpoint]
        schema_json = canonical_json_bytes(schema).decode("utf-8")
        grammar = xgrammar_module.Grammar.from_json_schema(schema_json)
        grammar_text = str(grammar)
        audit_balanced_xgrammar_frontier(grammar_text, intent_labels, "intent")
        if endpoint == "joint_json":
            audit_balanced_xgrammar_frontier(grammar_text, slot_labels, "slot")
        compiled[endpoint] = compiler.compile_json_schema(schema_json)
        legacy_schema = prediction_schema(
            intent_labels,
            slot_labels,
            endpoint=endpoint,
            structured_constraint_profile=LEGACY_STRUCTURED_CONSTRAINT_PROFILE,
        )
        legacy_compiled[endpoint] = compiler.compile_json_schema(
            canonical_json_bytes(legacy_schema).decode("utf-8")
        )

    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    for intent in intent_labels:
        intent_only = compact({"intent": intent})
        joint = compact({"intent": intent, "slots": []})
        if not _xgrammar_accepts_tokenized_text(
            xgrammar_module, compiled["intent_only"], tokenizer, intent_only
        ):
            raise ValueError(
                "Pinned XGrammar rejected a valid MASSIVE intent-only leaf"
            )
        if not _xgrammar_accepts_tokenized_text(
            xgrammar_module, compiled["joint_json"], tokenizer, joint
        ):
            raise ValueError("Pinned XGrammar rejected a valid MASSIVE intent leaf")

    exemplar_intent = intent_labels[0]
    for slot in slot_labels:
        joint = compact(
            {
                "intent": exemplar_intent,
                "slots": [{"name": slot, "value": "x"}],
            }
        )
        if not _xgrammar_accepts_tokenized_text(
            xgrammar_module, compiled["joint_json"], tokenizer, joint
        ):
            raise ValueError("Pinned XGrammar rejected a valid MASSIVE slot leaf")

    invalid_intents = (
        "__massive_outside_intent__",
        *RECORDED_LEGACY_HYBRID_INTENT_PROBES,
    )
    invalid_slots = (
        "__massive_outside_slot__",
        *RECORDED_LEGACY_HYBRID_SLOT_PROBES,
    )
    if set(invalid_intents) & set(intent_labels) or set(invalid_slots) & set(
        slot_labels
    ):
        raise AssertionError("Fabricated matcher probes unexpectedly entered ontology")
    invalid_cases = []
    for intent in invalid_intents:
        invalid_cases.extend(
            (
                (
                    "intent_only",
                    compact({"intent": intent}),
                    "fabricated intent-only label",
                ),
                (
                    "joint_json",
                    compact({"intent": intent, "slots": []}),
                    "fabricated joint intent label",
                ),
            )
        )
    for slot in invalid_slots:
        invalid_cases.append(
            (
                "joint_json",
                compact(
                    {
                        "intent": exemplar_intent,
                        "slots": [{"name": slot, "value": "x"}],
                    }
                ),
                "fabricated slot label",
            )
        )

    legacy_hybrid_cases = []
    for intent in RECORDED_LEGACY_HYBRID_INTENT_PROBES:
        legacy_hybrid_cases.extend(
            (
                ("intent_only", compact({"intent": intent})),
                ("joint_json", compact({"intent": intent, "slots": []})),
            )
        )
    for slot in RECORDED_LEGACY_HYBRID_SLOT_PROBES:
        legacy_hybrid_cases.append(
            (
                "joint_json",
                compact(
                    {
                        "intent": exemplar_intent,
                        "slots": [{"name": slot, "value": "x"}],
                    }
                ),
            )
        )
    for endpoint, probe in legacy_hybrid_cases:
        if not _xgrammar_accepts_tokenized_text(
            xgrammar_module, legacy_compiled[endpoint], tokenizer, probe
        ):
            raise ValueError(
                "Pinned legacy XGrammar matcher no longer reproduces its "
                "recorded hybrid-label escape"
            )
    for endpoint, probe, description in invalid_cases:
        if _xgrammar_accepts_tokenized_text(
            xgrammar_module, compiled[endpoint], tokenizer, probe
        ):
            raise ValueError(f"Pinned XGrammar admitted a {description}")

    return {
        "intent_leaves_checked": len(intent_labels),
        "slot_leaves_checked": len(slot_labels),
        "invalid_probes_rejected": len(invalid_cases),
        "legacy_hybrid_probes_reproduced": len(legacy_hybrid_cases),
    }


def validate_prediction(response, intent_labels, slot_labels, endpoint="joint_json"):
    if not isinstance(response, str):
        raise ValueError(
            "Structured decoder emitted a non-string response of type "
            f"{type(response).__name__}"
        )
    try:
        prediction = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Structured decoder emitted invalid JSON at "
            f"line {error.lineno}, column {error.colno}"
        ) from error
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


def failure_evidence_path(output_path):
    output = os.path.abspath(output_path)
    basename = os.path.basename(output)
    stem = basename[:-5] if basename.endswith(".json") else basename
    return os.path.join(os.path.dirname(output), "failures", f"{stem}.failure.json")


def seal_failure_evidence(payload):
    result = dict(payload)
    result.pop("failure_payload_sha256", None)
    result["failure_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_failure_evidence(payload):
    if not isinstance(payload, dict):
        raise ValueError("Failure evidence is not an object")
    copy = dict(payload)
    recorded = copy.pop("failure_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError("Failure evidence seal mismatch")
    return payload


def load_failure_evidence(path):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"Failure evidence is not a regular file: {path}")
    with open(path, encoding="utf-8") as handle:
        return verify_failure_evidence(json.load(handle))


def write_or_audit_failure_evidence(path, payload):
    expected = seal_failure_evidence(payload)
    if os.path.lexists(path):
        existing = load_failure_evidence(path)
        if existing != expected:
            raise ValueError(f"Existing failure evidence differs: {path}")
        return existing
    atomic_write_json(path, expected)
    observed = load_failure_evidence(path)
    if observed != expected:
        raise RuntimeError(f"Failure evidence differs after atomic write: {path}")
    return observed


def structured_validation_failure_payload(
    *,
    error,
    run,
    generation_fingerprint,
    output_path,
    row_index,
    record,
    response,
    finish_reason,
    token_ids,
    prompt_tokens,
):
    generation_fields = (
        "endpoint",
        "set_name",
        "role",
        "model_name",
        "model_path",
        "model_fingerprint",
        "base_model",
        "base_model_revision",
        "prompt_file_sha256",
        "ontology_sha256",
        "json_schema_sha256",
        "structured_backend",
        "vllm_version",
        "xgrammar_version",
        "temperature",
        "n_samples",
        "max_new_tokens",
        "max_context",
        "seed",
        "structured_constraint_profile",
    )
    generation = {
        field: run[field] for field in generation_fields if field in run
    }
    generation["structured_constraint_profile"] = run.get(
        "structured_constraint_profile", LEGACY_STRUCTURED_CONSTRAINT_PROFILE
    )
    generation["generation_fingerprint"] = generation_fingerprint
    token_ids = list(token_ids)
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA_VERSION,
        "failure_kind": "structured_prediction_validation",
        "validation_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "generation": generation,
        "output_file": os.path.abspath(output_path),
        "offending_sample": {
            "row_index": row_index,
            "question_id": record["question_id"],
            "prompt_sha256": record["prompt_sha256"],
            "raw_response": response,
            "response_sha256": sha256_bytes(response.encode("utf-8")),
            "finish_reason": finish_reason,
            "token_ids": token_ids,
            "n_generated_tokens": len(token_ids),
            "prompt_tokens": prompt_tokens,
        },
    }


def shutdown_vllm_engine(llm):
    """Use the pinned vLLM 0.11.2 EngineCoreClient shutdown path."""
    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("Pinned vLLM engine has no callable shutdown path")
    shutdown()


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
    parser.add_argument(
        "--structured_constraint_profile",
        choices=STRUCTURED_CONSTRAINT_PROFILES,
        default=LEGACY_STRUCTURED_CONSTRAINT_PROFILE,
    )
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

    tokenizer_load_kwargs = {"revision": base_revision}
    if args.structured_constraint_profile == STRICT_STRUCTURED_CONSTRAINT_PROFILE:
        tokenizer_load_kwargs["local_files_only"] = True
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_model, **tokenizer_load_kwargs
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
                meta["intent_labels"],
                meta["slot_labels"],
                endpoint=endpoint,
                structured_constraint_profile=args.structured_constraint_profile,
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
                if (
                    args.structured_constraint_profile
                    != LEGACY_STRUCTURED_CONSTRAINT_PROFILE
                ):
                    run["structured_constraint_profile"] = (
                        args.structured_constraint_profile
                    )
                fingerprint = sha256_bytes(canonical_json_bytes(run))
                provenance[(meta["set_name"], name, endpoint)] = (run, fingerprint)
                suffix = "" if endpoint == "joint_json" else "__intent_only"
                output = os.path.join(
                    args.output_dir, f"{meta['set_name']}__{name}{suffix}.json"
                )
                failure_path = failure_evidence_path(output)
                if not args.preflight_only and os.path.lexists(failure_path):
                    failure = load_failure_evidence(failure_path)
                    observed_fingerprint = failure.get("generation", {}).get(
                        "generation_fingerprint"
                    )
                    if observed_fingerprint != fingerprint:
                        raise ValueError(
                            f"Existing failure evidence provenance differs: {failure_path}"
                        )
                    raise RuntimeError(
                        "Existing terminal structured-generation failure evidence "
                        f"refuses a same-namespace rerun: {failure_path}"
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
    import xgrammar
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
    if args.structured_constraint_profile == STRICT_STRUCTURED_CONSTRAINT_PROFILE:
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            base_model, revision=base_revision, local_files_only=True
        )
        if getattr(model_config, "_commit_hash", None) != base_revision:
            raise ValueError(
                "Locally cached model config does not resolve to the pinned revision"
            )
        for bank in banks:
            try:
                audit_strict_xgrammar_contract(
                    xgrammar,
                    tokenizer,
                    model_config,
                    bank["meta"]["intent_labels"],
                    bank["meta"]["slot_labels"],
                    bank["schemas"],
                )
            except Exception as error:
                raise ValueError(
                    "Pinned XGrammar token-matcher contract failed for MASSIVE "
                    f"{args.structured_constraint_profile} "
                    f"{bank['meta']['set_name']}"
                ) from error
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
    llm = None
    primary_error = None
    try:
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
                run, fingerprint = provenance[
                    (bank["meta"]["set_name"], name, endpoint)
                ]
                samples = []
                for row_index, (record, output) in enumerate(zip(prompts, outputs)):
                    if len(output.outputs) != 1:
                        raise RuntimeError(
                            "Expected one greedy structured completion"
                        )
                    completion = output.outputs[0]
                    response = completion.text
                    finish_reason = getattr(completion, "finish_reason", None)
                    stop_reason = (
                        "max_new_tokens"
                        if finish_reason == "length"
                        else finish_reason
                    ) or "unknown"
                    token_ids = list(getattr(completion, "token_ids", None) or [])
                    prompt_tokens = bank["prompt_lengths"][record["question_id"]]
                    try:
                        prediction = validate_prediction(
                            response,
                            bank["meta"]["intent_labels"],
                            bank["meta"]["slot_labels"],
                            endpoint=endpoint,
                        )
                    except ValueError as validation_error:
                        failure_path = failure_evidence_path(output_path)
                        failure = structured_validation_failure_payload(
                            error=validation_error,
                            run=run,
                            generation_fingerprint=fingerprint,
                            output_path=output_path,
                            row_index=row_index,
                            record=record,
                            response=response,
                            finish_reason=stop_reason,
                            token_ids=token_ids,
                            prompt_tokens=prompt_tokens,
                        )
                        try:
                            recorded_failure = write_or_audit_failure_evidence(
                                failure_path, failure
                            )
                        except BaseException as evidence_error:
                            context = {
                                "set_name": run["set_name"],
                                "model_name": run["model_name"],
                                "endpoint": endpoint,
                                "row_index": row_index,
                                "question_id": record["question_id"],
                                "prompt_sha256": record["prompt_sha256"],
                                "generation_fingerprint": fingerprint,
                            }
                            raise RuntimeError(
                                "Could not atomically record MASSIVE structured "
                                "validation failure: "
                                + canonical_json_bytes(context).decode("utf-8")
                            ) from evidence_error
                        context = {
                            "evidence_file": failure_path,
                            "failure_payload_sha256": recorded_failure[
                                "failure_payload_sha256"
                            ],
                            "set_name": run["set_name"],
                            "model_name": run["model_name"],
                            "endpoint": endpoint,
                            "row_index": row_index,
                            "question_id": record["question_id"],
                            "prompt_sha256": record["prompt_sha256"],
                            "generation_fingerprint": fingerprint,
                        }
                        raise ValueError(
                            "MASSIVE structured prediction failed validation: "
                            + canonical_json_bytes(context).decode("utf-8")
                        ) from validation_error
                    sample = {
                        "question_id": record["question_id"],
                        "sample_index": 0,
                        "response": response,
                        "prediction": prediction,
                        "stop_reason": stop_reason,
                        "n_generated_tokens": len(token_ids),
                        "prompt_tokens": prompt_tokens,
                        "prompt_sha256": record["prompt_sha256"],
                    }
                    sample["result_sha256"] = sample_sha256(sample)
                    samples.append(sample)
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
                    raise RuntimeError(
                        f"Generation audit failed after write: {output_path}"
                    )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if llm is not None:
            try:
                shutdown_vllm_engine(llm)
            except BaseException as shutdown_error:
                if primary_error is None:
                    raise
                print(
                    "vLLM shutdown also failed after the primary MASSIVE error: "
                    f"{type(shutdown_error).__name__}: {shutdown_error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
