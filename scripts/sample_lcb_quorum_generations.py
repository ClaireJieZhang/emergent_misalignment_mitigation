#!/usr/bin/env python3
"""Resumable tokenwise quorum generation for LiveCodeBench.

This sampler deliberately has a narrow interface: exactly four LoRA reference
adapters, one pinned base model, one GPU, and LiveCodeBench prompt records with
unique ``question_id`` values.  It writes one checksummed result shard after
every sample, so a preemption loses at most the sample being generated.

Ordinary quorum selects the q-th largest reference log probability for every
token.  Base-relative quorum applies a direction-aware q-of-m order statistic
to each reference's log-probability shift from the base distribution.  For
q=m, these are respectively tokenwise min and pi-min-delta.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile

import yaml


SCHEMA_VERSION = 1
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_value(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def adapter_tree_fingerprint(path):
    """Hash relative names, sizes, and bytes for every adapter-tree file."""
    root = os.path.realpath(path)
    if not os.path.isdir(root):
        raise ValueError(f"Adapter path is not a directory: {path}")
    files = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        for file_name in sorted(file_names):
            absolute = os.path.join(directory, file_name)
            if not os.path.isfile(absolute):
                raise ValueError(f"Adapter tree contains a non-regular file: {absolute}")
            relative = os.path.relpath(absolute, root)
            files.append(
                {
                    "path": relative,
                    "size": os.path.getsize(absolute),
                    "sha256": sha256_file(absolute),
                }
            )
    if not files:
        raise ValueError(f"Adapter directory contains no files: {path}")
    return {
        "realpath": root,
        "tree_sha256": sha256_value(files),
        "files": files,
    }


def atomic_write_json(payload, path):
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".json-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, absolute)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json_strict(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def parse_ref_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Reference must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path or path.upper() == "BASE":
        raise ValueError(f"Reference must be a named LoRA directory, got {spec!r}")
    return name, path


def load_training_identity(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model = config.get("base_model")
    revision = config.get("base_model_revision")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Training YAML must contain a non-empty base_model")
    if not isinstance(revision, str) or not PINNED_REVISION_RE.fullmatch(revision):
        raise ValueError(
            "Training YAML base_model_revision must be an immutable 40-character "
            "lowercase hexadecimal Hugging Face commit"
        )
    return model.strip(), revision


def normalize_question_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"question_id must be a string or integer, got {value!r}")
    normalized = str(value)
    if not normalized:
        raise ValueError("question_id must not be empty")
    return normalized


def load_prompt_records(path):
    raw = read_json_strict(path)
    if isinstance(raw, dict):
        raw = raw.get("prompts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Prompt file must be a non-empty list or {'prompts': [...]} object")
    records = []
    seen = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"Prompt record {index} must contain a string prompt")
        if "question_id" not in item:
            raise ValueError(f"Prompt record {index} has no question_id")
        question_id = normalize_question_id(item["question_id"])
        if question_id in seen:
            raise ValueError(f"Duplicate question_id after normalization: {question_id!r}")
        seen.add(question_id)
        record = dict(item)
        record["question_id"] = question_id
        if "system" in record and not isinstance(record["system"], str):
            raise ValueError(f"Prompt record {index} system must be a string when present")
        records.append(record)
    return records


def partition_records(records, chunk_index, chunk_count):
    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError(f"chunk_index must be in [0, {chunk_count}), got {chunk_index}")
    ordered = sorted(records, key=lambda record: record["question_id"])
    start = len(ordered) * chunk_index // chunk_count
    stop = len(ordered) * (chunk_index + 1) // chunk_count
    return ordered[start:stop]


def tuple_seed(*parts):
    """Map an explicit tuple to a stable torch-compatible 63-bit seed."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def prompt_sha256(record):
    return sha256_value(
        {
            "system": record.get("system", ""),
            "prompt": record["prompt"],
        }
    )


def sample_shard_name(question_id, ordinal, sample_index):
    question_digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16]
    return f"sample-{ordinal:06d}-{question_digest}-n{sample_index:03d}.json"


def expected_sample_specs(records, n_samples):
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    specs = []
    for ordinal, record in enumerate(records):
        for sample_index in range(n_samples):
            specs.append(
                {
                    "ordinal": ordinal,
                    "question_id": record["question_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": prompt_sha256(record),
                    "shard_name": sample_shard_name(
                        record["question_id"], ordinal, sample_index
                    ),
                }
            )
    return specs


def build_immutable_manifest(
    *,
    training_config,
    prompt_file,
    records,
    references,
    method,
    q,
    n_samples,
    max_new_tokens,
    max_context,
    temperature,
    seed,
    chunk_index,
    chunk_count,
    device,
    sampler_path=None,
):
    base_model, base_model_revision = load_training_identity(training_config)
    if len(references) != 4:
        raise ValueError(f"Exactly four LoRA references are required, got {len(references)}")
    reference_names = [name for name, _ in references]
    if len(set(reference_names)) != len(reference_names):
        raise ValueError("Reference names must be unique")
    if q < 1 or q > len(references):
        raise ValueError(f"q must be in [1, {len(references)}], got {q}")
    if method == "pi_quorum_delta" and 2 * q <= len(references):
        raise ValueError("pi_quorum_delta requires a strict-majority q to avoid opposing quorums")
    if method not in {"quorum", "pi_quorum_delta"}:
        raise ValueError(f"Unknown method: {method}")
    if max_new_tokens < 1 or max_context < 1:
        raise ValueError("max_new_tokens and max_context must be positive")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not records:
        raise ValueError("The selected chunk contains no prompts")
    ref_manifest = []
    for name, path in references:
        fingerprint = adapter_tree_fingerprint(path)
        ref_manifest.append(
            {
                "name": name,
                "path": os.path.abspath(path),
                "realpath": fingerprint["realpath"],
                "tree_sha256": fingerprint["tree_sha256"],
                "files": fingerprint["files"],
            }
        )
    source_path = sampler_path or __file__
    return {
        "schema_version": SCHEMA_VERSION,
        "sampler_sha256": sha256_file(source_path),
        "training_config": os.path.abspath(training_config),
        "training_config_sha256": sha256_file(training_config),
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "prompt_file": os.path.abspath(prompt_file),
        "prompt_file_sha256": sha256_file(prompt_file),
        "selected_questions": [
            {
                "question_id": record["question_id"],
                "prompt_sha256": prompt_sha256(record),
            }
            for record in records
        ],
        "references": ref_manifest,
        "method": method,
        "q": q,
        "n_samples": n_samples,
        "max_new_tokens": max_new_tokens,
        "max_context": max_context,
        "temperature": temperature,
        "seed": seed,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "device": device,
        "chat_template": {
            "roles": ["system", "user"],
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "generation": {
            "greedy_when_temperature_zero": True,
            "cache": "prefill_once_then_single_token_steps",
            "dtype": "bfloat16",
            "truncation": False,
        },
    }


def ensure_manifest(output_dir, immutable_manifest):
    os.makedirs(output_dir, exist_ok=True)
    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(output_dir, "manifest.json")
    fingerprint = sha256_value(immutable_manifest)
    expected = {
        "immutable_manifest_sha256": fingerprint,
        "immutable": immutable_manifest,
    }
    if os.path.exists(path):
        observed = read_json_strict(path)
        if observed != expected:
            observed_fingerprint = (
                observed.get("immutable_manifest_sha256")
                if isinstance(observed, dict)
                else None
            )
            raise ValueError(
                "Immutable manifest mismatch: expected "
                f"{fingerprint}, found {observed_fingerprint} in {path}"
            )
    else:
        existing = [name for name in os.listdir(shard_dir) if name.endswith(".json")]
        if existing:
            raise ValueError("Result shards exist without an immutable manifest")
        atomic_write_json(expected, path)
    return fingerprint


def result_digest(record):
    content = dict(record)
    content.pop("result_sha256", None)
    return sha256_value(content)


def validate_result_shard(payload, spec, manifest_fingerprint, path):
    if not isinstance(payload, dict):
        raise ValueError(f"Result shard is not an object: {path}")
    expected_fields = {
        "schema_version": SCHEMA_VERSION,
        "immutable_manifest_sha256": manifest_fingerprint,
        "question_id": spec["question_id"],
        "sample_index": spec["sample_index"],
        "prompt_sha256": spec["prompt_sha256"],
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Result shard mismatch for {key} in {path}: "
                f"expected {expected!r}, found {payload.get(key)!r}"
            )
    observed_digest = payload.get("result_sha256")
    expected_digest = result_digest(payload)
    if observed_digest != expected_digest:
        raise ValueError(
            f"Result shard checksum mismatch in {path}: "
            f"expected {expected_digest}, found {observed_digest}"
        )
    if not isinstance(payload.get("response"), str):
        raise ValueError(f"Result shard response is not a string: {path}")
    token_ids = payload.get("response_token_ids")
    selected_ids = payload.get("selected_token_ids")
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise ValueError(f"Invalid response_token_ids in {path}")
    if not isinstance(selected_ids, list) or not all(isinstance(item, int) for item in selected_ids):
        raise ValueError(f"Invalid selected_token_ids in {path}")
    if payload.get("n_response_tokens") != len(token_ids):
        raise ValueError(f"n_response_tokens does not match token IDs in {path}")
    if payload.get("n_selected_tokens") != len(selected_ids):
        raise ValueError(f"n_selected_tokens does not match token IDs in {path}")
    if payload.get("stop_reason") not in {"eos", "length"}:
        raise ValueError(f"Invalid stop_reason in {path}")
    return payload


def audit_result_shards(output_dir, specs, manifest_fingerprint, require_complete=False):
    shard_dir = os.path.join(output_dir, "shards")
    expected_names = {spec["shard_name"] for spec in specs}
    observed_names = {
        name for name in os.listdir(shard_dir) if name.endswith(".json")
    }
    extras = sorted(observed_names - expected_names)
    if extras:
        raise ValueError(f"Unexpected result shards: {extras}")
    records = {}
    missing = []
    for spec in specs:
        path = os.path.join(shard_dir, spec["shard_name"])
        if not os.path.exists(path):
            missing.append(spec)
            continue
        payload = read_json_strict(path)
        records[spec["shard_name"]] = validate_result_shard(
            payload, spec, manifest_fingerprint, path
        )
    if require_complete and missing:
        raise ValueError(f"Missing {len(missing)} of {len(specs)} expected result shards")
    return records, missing


def compose_quorum_log_probs(logps, q):
    import torch

    if logps.ndim != 2:
        raise ValueError("logps must have shape [references, vocabulary]")
    m = logps.shape[0]
    if q < 1 or q > m:
        raise ValueError(f"q must be in [1, {m}], got {q}")
    qth_largest = torch.topk(logps, k=q, dim=0, largest=True).values[-1]
    return qth_largest - torch.logsumexp(qth_largest, dim=-1)


def compose_pi_quorum_delta_log_probs(logps, base_logp, q):
    """Apply a direction-aware strict-majority q-of-m delta quorum.

    A positive shift is the q-th largest positive shift.  A negative shift is
    the q-th smallest negative shift.  Requiring a strict majority makes the
    two directions mutually exclusive.  At q=m this is exactly pi-min-delta:
    all references must agree on direction and the least-magnitude agreed
    shift is retained.
    """
    import torch

    if logps.ndim != 2 or base_logp.ndim != 1 or logps.shape[1] != base_logp.shape[0]:
        raise ValueError("Expected logps [references, vocabulary] and base_logp [vocabulary]")
    m = logps.shape[0]
    if q < 1 or q > m:
        raise ValueError(f"q must be in [1, {m}], got {q}")
    if 2 * q <= m:
        raise ValueError("pi_quorum_delta requires q to be a strict majority")
    ratios = logps - base_logp.to(logps.device).unsqueeze(0)
    qth_positive = torch.topk(ratios, k=q, dim=0, largest=True).values[-1]
    qth_negative = torch.topk(ratios, k=q, dim=0, largest=False).values[-1]
    upward = torch.where(qth_positive > 0, qth_positive, torch.zeros_like(qth_positive))
    downward = torch.where(qth_negative < 0, qth_negative, torch.zeros_like(qth_negative))
    delta = upward + downward
    target = base_logp.to(logps.device) + delta
    return target - torch.logsumexp(target, dim=-1)


def cache_sequence_length(cache):
    """Return the cached sequence length for DynamicCache or legacy tuples."""
    if cache is None:
        return 0
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    if isinstance(cache, (tuple, list)):
        if not cache:
            return 0
        layer = cache[0]
        if not isinstance(layer, (tuple, list)) or not layer:
            raise ValueError("Unrecognized legacy past_key_values layer")
        key = layer[0]
        if not hasattr(key, "shape") or len(key.shape) < 3:
            raise ValueError("Unrecognized legacy cache key tensor")
        return int(key.shape[-2])
    raise ValueError(f"Unrecognized cache type: {type(cache).__name__}")


def _extract_logits_and_cache(outputs):
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        cache = getattr(outputs, "past_key_values", None)
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        logits, cache = outputs[0], outputs[1]
    else:
        raise ValueError("Model output has neither attributes nor a (logits, cache) tuple")
    if cache is None:
        raise ValueError("Model did not return past_key_values with use_cache=True")
    return logits, cache


def prefill_cached_model(model, prompt_ids, device):
    import torch

    if not prompt_ids:
        raise ValueError("Cannot prefill an empty prompt")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    logits, cache = _extract_logits_and_cache(outputs)
    observed_length = cache_sequence_length(cache)
    if observed_length != len(prompt_ids):
        raise ValueError(
            f"Prefill cache length {observed_length} does not match prompt length {len(prompt_ids)}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": cache}


def step_cached_model(model, token_id, cache, device):
    import torch

    previous_length = cache_sequence_length(cache)
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    logits, next_cache = _extract_logits_and_cache(outputs)
    observed_length = cache_sequence_length(next_cache)
    if observed_length != previous_length + 1:
        raise ValueError(
            f"One-token cache step grew from {previous_length} to {observed_length}, expected {previous_length + 1}"
        )
    return {"next_logits": logits[0, -1, :].float(), "cache": next_cache}


def chat_messages(record):
    messages = []
    system = record.get("system", "")
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": record["prompt"]})
    return messages


def make_prompt_ids(tokenizer, record):
    ids = tokenizer.apply_chat_template(
        chat_messages(record),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned a batch")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def load_tokenizer(base_model, revision):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    return tokenizer


def load_lora_reference(base_model, revision, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model


def load_base_reference(base_model, revision, device):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True
    return model


def eos_token_ids(tokenizer, models):
    values = []
    tokenizer_eos = tokenizer.eos_token_id
    values.extend(tokenizer_eos if isinstance(tokenizer_eos, list) else [tokenizer_eos])
    for model in models:
        value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if value is not None:
            values.extend(value if isinstance(value, list) else [value])
    return {int(value) for value in values if value is not None}


def generate_one(
    *,
    record,
    sample_index,
    prompt_ids,
    reference_models,
    base_model,
    tokenizer,
    method,
    q,
    max_new_tokens,
    temperature,
    global_seed,
    device,
    stop_ids,
):
    import torch
    import torch.nn.functional as torch_functional

    reference_states = [
        prefill_cached_model(model, prompt_ids, device) for model in reference_models
    ]
    base_state = (
        prefill_cached_model(base_model, prompt_ids, device)
        if method == "pi_quorum_delta"
        else None
    )
    selected_ids = []
    response_ids = []
    stop_reason = "length"
    rng_seed = tuple_seed(global_seed, record["question_id"], sample_index, method, q)
    generator = None
    if temperature > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(rng_seed)

    for token_index in range(max_new_tokens):
        reference_logps = torch.stack(
            [
                torch_functional.log_softmax(state["next_logits"], dim=-1)
                for state in reference_states
            ],
            dim=0,
        )
        if method == "quorum":
            target_logp = compose_quorum_log_probs(reference_logps, q)
        else:
            base_logp = torch_functional.log_softmax(base_state["next_logits"], dim=-1)
            target_logp = compose_pi_quorum_delta_log_probs(reference_logps, base_logp, q)
        if temperature == 0:
            token_id = int(torch.argmax(target_logp).item())
        else:
            tempered = target_logp / temperature
            tempered = tempered - torch.logsumexp(tempered, dim=-1)
            token_id = int(
                torch.multinomial(torch.exp(tempered), 1, generator=generator).item()
            )
        selected_ids.append(token_id)
        if token_id in stop_ids:
            stop_reason = "eos"
            break
        response_ids.append(token_id)
        if token_index + 1 < max_new_tokens:
            reference_states = [
                step_cached_model(model, token_id, state["cache"], device)
                for model, state in zip(reference_models, reference_states)
            ]
            if base_state is not None:
                base_state = step_cached_model(
                    base_model, token_id, base_state["cache"], device
                )

    return {
        "question_id": record["question_id"],
        "sample_index": sample_index,
        "prompt": record["prompt"],
        "prompt_meta": {
            key: value for key, value in record.items() if key not in {"prompt", "system"}
        },
        "system": record.get("system", ""),
        "prompt_sha256": prompt_sha256(record),
        "response": tokenizer.decode(response_ids, skip_special_tokens=True),
        "response_token_ids": response_ids,
        "selected_token_ids": selected_ids,
        "n_response_tokens": len(response_ids),
        "n_selected_tokens": len(selected_ids),
        "stop_reason": stop_reason,
        "rng_seed": rng_seed,
    }


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def finalize_chunk(output_dir, immutable_manifest, fingerprint, specs):
    records, missing = audit_result_shards(
        output_dir, specs, fingerprint, require_complete=True
    )
    if missing:
        raise AssertionError("require_complete audit returned missing shards")
    ordered = [records[spec["shard_name"]] for spec in specs]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "immutable_manifest_sha256": fingerprint,
        "immutable_manifest": immutable_manifest,
        "git_sha": git_sha(),
        "completed_samples": len(ordered),
        "samples": ordered,
    }
    path = os.path.join(output_dir, "chunk.json")
    atomic_write_json(payload, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", action="append", required=True, help="NAME=ADAPTER_PATH; repeat four times")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["quorum", "pi_quorum_delta"], required=True)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--chunk_index", type=int, default=0)
    parser.add_argument("--chunk_count", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_context", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", required=True, help="Single device for all BF16 models, e.g. cuda:0")
    args = parser.parse_args()

    references = [parse_ref_spec(spec) for spec in args.ref]
    all_records = load_prompt_records(args.prompt_file)
    records = partition_records(all_records, args.chunk_index, args.chunk_count)
    immutable_manifest = build_immutable_manifest(
        training_config=args.training_config,
        prompt_file=args.prompt_file,
        records=records,
        references=references,
        method=args.method,
        q=args.q,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        max_context=args.max_context,
        temperature=args.temperature,
        seed=args.seed,
        chunk_index=args.chunk_index,
        chunk_count=args.chunk_count,
        device=args.device,
    )
    fingerprint = ensure_manifest(args.output_dir, immutable_manifest)
    specs = expected_sample_specs(records, args.n_samples)
    completed, missing = audit_result_shards(args.output_dir, specs, fingerprint)
    print(
        f"Resume audit: {len(completed)}/{len(specs)} complete; "
        f"{len(missing)} samples remain"
    )

    base_model_name = immutable_manifest["base_model"]
    revision = immutable_manifest["base_model_revision"]
    tokenizer = load_tokenizer(base_model_name, revision)
    prompt_ids_by_question = {}
    for record in records:
        prompt_ids = make_prompt_ids(tokenizer, record)
        if len(prompt_ids) + args.max_new_tokens > args.max_context:
            raise ValueError(
                f"question_id={record['question_id']!r} has {len(prompt_ids)} prompt tokens; "
                f"{len(prompt_ids)} + {args.max_new_tokens} exceeds max_context={args.max_context}. "
                "The sampler never truncates prompts."
            )
        prompt_ids_by_question[record["question_id"]] = prompt_ids

    if missing:
        print(f"Loading four BF16 LoRA references on {args.device}")
        reference_models = [
            load_lora_reference(base_model_name, revision, path, args.device)
            for _, path in references
        ]
        base_reference = None
        if args.method == "pi_quorum_delta":
            print(f"Loading pinned BF16 base reference on {args.device}")
            base_reference = load_base_reference(base_model_name, revision, args.device)
        stop_ids = eos_token_ids(
            tokenizer,
            reference_models + ([base_reference] if base_reference is not None else []),
        )
        record_by_id = {record["question_id"]: record for record in records}
        for remaining_index, spec in enumerate(missing, start=1):
            record = record_by_id[spec["question_id"]]
            print(
                f"Generating {remaining_index}/{len(missing)}: "
                f"question_id={spec['question_id']} sample={spec['sample_index']}"
            )
            result = generate_one(
                record=record,
                sample_index=spec["sample_index"],
                prompt_ids=prompt_ids_by_question[spec["question_id"]],
                reference_models=reference_models,
                base_model=base_reference,
                tokenizer=tokenizer,
                method=args.method,
                q=args.q,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                global_seed=args.seed,
                device=args.device,
                stop_ids=stop_ids,
            )
            result.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "immutable_manifest_sha256": fingerprint,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            )
            result["result_sha256"] = result_digest(result)
            shard_path = os.path.join(args.output_dir, "shards", spec["shard_name"])
            atomic_write_json(result, shard_path)
            validate_result_shard(read_json_strict(shard_path), spec, fingerprint, shard_path)

    final_path = finalize_chunk(args.output_dir, immutable_manifest, fingerprint, specs)
    print(f"Complete exact-ID audit passed; wrote {final_path}")


if __name__ == "__main__":
    main()
