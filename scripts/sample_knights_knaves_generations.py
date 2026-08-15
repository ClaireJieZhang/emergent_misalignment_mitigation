#!/usr/bin/env python3
"""Generate deterministic direct answers for prepared Knights & Knaves prompts.

This sampler only accepts the label-free prompt banks emitted by
prepare_knights_knaves_pilot_data.py.  Each model is written atomically to an
independently auditable file so base models and intermediate LoRA checkpoints
can be resumed without changing already completed results.

The pinned vLLM Qwen2 backend does not expose ``lm_head`` as a LoRA target.
The preflight therefore rejects such adapters before allocating a GPU; exact
paper-style output-head adapters require a Transformers/PEFT evaluator.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import tempfile

import yaml


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


def generation_sample_sha256(sample):
    fields = (
        "question_id", "sample_index", "response", "stop_reason",
        "n_generated_tokens", "prompt_tokens", "prompt_sha256",
    )
    if not all(field in sample for field in fields):
        raise ValueError("Generation sample lacks fields required by its checksum")
    projection = {field: sample[field] for field in fields}
    return sha256_bytes(canonical_json_bytes(projection))


def adapter_fingerprint(path):
    if path == "BASE":
        return "BASE"
    root = os.path.abspath(path)
    config = os.path.join(root, "adapter_config.json")
    weight_candidates = [
        os.path.join(root, "adapter_model.safetensors"),
        os.path.join(root, "adapter_model.bin"),
    ]
    weights = [path for path in weight_candidates if os.path.isfile(path)]
    if not os.path.isfile(config) or len(weights) != 1:
        raise ValueError(
            f"Expected adapter_config.json and exactly one adapter weight file in {root}"
        )
    entries = []
    for artifact in [config] + weights:
        entries.append(
            {
                "name": os.path.basename(artifact),
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            }
        )
    return sha256_bytes(canonical_json_bytes(entries))


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Model must be NAME=BASE or NAME=ADAPTER_PATH: {spec!r}")
    name, path = (part.strip() for part in spec.split("=", 1))
    if not name or not path:
        raise ValueError(f"Invalid model specification: {spec!r}")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise ValueError(f"Unsafe model name: {name!r}")
    if path.upper() == "BASE":
        path = "BASE"
    else:
        path = os.path.abspath(path)
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise ValueError(f"Missing adapter_config.json for {name}: {path}")
    return name, path


def audit_adapter_config(path, training):
    if path == "BASE":
        return None
    config_path = os.path.join(path, "adapter_config.json")
    with open(config_path, encoding="utf-8") as handle:
        adapter = json.load(handle)
    if str(adapter.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Adapter is not LoRA: {path}")
    expected_rank = training["lora"]["rank"]
    if adapter.get("r") != expected_rank:
        raise ValueError(
            f"Adapter rank mismatch at {path}: {adapter.get('r')} != {expected_rank}"
        )
    expected_alpha = training["lora"].get("alpha")
    if expected_alpha is not None and adapter.get("lora_alpha") != expected_alpha:
        raise ValueError(
            f"Adapter alpha mismatch at {path}: "
            f"{adapter.get('lora_alpha')} != {expected_alpha}"
        )
    target_modules = adapter.get("target_modules")
    expected_targets = training["lora"].get("target_modules")
    if not isinstance(target_modules, list) or not isinstance(expected_targets, list):
        raise ValueError("Training and adapter target_modules must be explicit lists")
    if set(target_modules) != set(expected_targets):
        raise ValueError(
            f"Adapter target_modules mismatch at {path}: "
            f"{sorted(target_modules)} != {sorted(expected_targets)}"
        )
    return adapter


def audit_vllm_target_compatibility(base_model, target_modules):
    """Fail before GPU allocation on a known Qwen2/vLLM output-head mismatch."""
    normalized_model = base_model.casefold()
    if "qwen2" in normalized_model:
        unsupported = sorted({"lm_head", "embed_tokens"} & set(target_modules))
        if unsupported:
            raise ValueError(
                "Pinned vLLM's Qwen2 implementation does not register output/token "
                f"embedding LoRA modules {unsupported}; use a projection-only adapter "
                "for this vLLM sampler or evaluate the exact adapter with a "
                "Transformers/PEFT backend."
            )


def load_prompt_bank(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    prompts = payload.get("prompts")
    if not isinstance(meta, dict) or not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt bank must contain nonempty meta and prompts fields")
    if meta.get("contains_labels") is not False:
        raise ValueError("Refusing prompt bank that is not explicitly label-free")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", str(meta.get("set_name", ""))) is None:
        raise ValueError("Prompt bank has an unsafe set_name")
    expected_count = meta.get("n_questions")
    if expected_count != len(prompts):
        raise ValueError("Prompt-bank count does not match metadata")
    seen = set()
    forbidden_fields = {
        "solution", "solution_text", "solution_text_format", "answer", "answers"
    }
    validated = []
    for index, record in enumerate(prompts):
        if not isinstance(record, dict):
            raise ValueError(f"Prompt {index} is not an object")
        leaked = sorted(forbidden_fields & set(record))
        if leaked:
            raise ValueError(f"Prompt {index} leaks answer fields: {leaked}")
        question_id = record.get("question_id")
        prompt = record.get("prompt")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Prompt {index} lacks a question_id")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Prompt {index} lacks prompt text")
        if question_id in seen:
            raise ValueError(f"Duplicate question_id: {question_id}")
        seen.add(question_id)
        expected_hash = sha256_bytes(canonical_json_bytes({"prompt": prompt}))
        if record.get("prompt_sha256") != expected_hash:
            raise ValueError(f"Prompt hash mismatch for {question_id}")
        if record.get("set_name") != meta.get("set_name"):
            raise ValueError(f"Set-name mismatch for {question_id}")
        validated.append(dict(record))
    return meta, validated


def chat_messages(record):
    # Training uses the same complete official instruction as one user turn.
    return [{"role": "user", "content": record["prompt"]}]


def validate_context_lengths(tokenizer, prompts, max_new_tokens, max_context):
    lengths = {}
    for record in prompts:
        token_ids = tokenizer.apply_chat_template(
            chat_messages(record),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        length = len(token_ids)
        if length + max_new_tokens > max_context:
            raise ValueError(
                f"{record['question_id']} needs {length + max_new_tokens} tokens, "
                f"exceeding max_context={max_context}"
            )
        lengths[record["question_id"]] = length
    return lengths


def generation_is_complete(path, expected_run, fingerprint, expected_ids):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Corrupt existing generation output {path}: {error}") from error
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Existing output has no metadata object: {path}")
    if meta.get("generation_fingerprint") != fingerprint:
        raise ValueError(f"Existing output provenance mismatch: {path}")
    expected_meta_keys = set(expected_run) | {"generation_fingerprint", "created_at"}
    if set(meta) != expected_meta_keys:
        raise ValueError(f"Existing output metadata keys differ from protocol: {path}")
    observed_run = {key: meta[key] for key in expected_run}
    if observed_run != expected_run:
        raise ValueError(f"Existing output metadata fields differ from protocol: {path}")
    observed_fingerprint = sha256_bytes(canonical_json_bytes(observed_run))
    if observed_fingerprint != fingerprint:
        raise ValueError(f"Existing output metadata seal is internally invalid: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Existing output has no sample list: {path}")
    found_ids = [sample.get("question_id") for sample in samples]
    if found_ids != expected_ids:
        raise ValueError(f"Existing output has incomplete or reordered IDs: {path}")
    if any(sample.get("sample_index") != 0 for sample in samples):
        raise ValueError(f"Existing output is not deterministic pass@1: {path}")
    expected_prompt_hashes = expected_run.get("prompt_sha256")
    if not isinstance(expected_prompt_hashes, list) or len(expected_prompt_hashes) != len(samples):
        raise ValueError("Expected generation metadata has invalid prompt hashes")
    for index, (sample, prompt_hash) in enumerate(zip(samples, expected_prompt_hashes)):
        if sample.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"Existing sample prompt hash mismatch at index {index}: {path}")
        if not isinstance(sample.get("response"), str):
            raise ValueError(f"Existing sample response is invalid at index {index}: {path}")
        if sample.get("result_sha256") != generation_sample_sha256(sample):
            raise ValueError(f"Existing sample checksum mismatch at index {index}: {path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", required=True,
        help="Repeat NAME=BASE or NAME=ADAPTER_PATH.",
    )
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_context", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8152026)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_context <= 0:
        parser.error("Token limits must be positive")

    with open(args.training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    base_model = training.get("base_model")
    base_revision = training.get("base_model_revision")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("Training config must specify base_model")
    if not isinstance(base_revision, str) or not base_revision:
        raise ValueError("Training config must pin base_model_revision")
    lora_rank = training.get("lora", {}).get("rank")
    if not isinstance(lora_rank, int) or lora_rank <= 0:
        raise ValueError("Training config must specify a positive lora.rank")
    target_modules = training.get("lora", {}).get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("Training config must specify explicit lora.target_modules")

    models = [parse_model_spec(spec) for spec in args.model]
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names: {names}")
    prompt_meta, prompts = load_prompt_bank(args.prompt_file)

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_model, revision=base_revision
    )
    prompt_lengths = validate_context_lengths(
        tokenizer, prompts, args.max_new_tokens, args.max_context
    )
    prompt_file_hash = sha256_file(args.prompt_file)
    expected_ids = [record["question_id"] for record in prompts]
    os.makedirs(args.output_dir, exist_ok=True)

    pending = []
    provenance = {}
    for name, path in models:
        adapter_config = audit_adapter_config(path, training)
        run = {
            "schema_version": 1,
            "generator": "vllm_greedy_direct_answer",
            "generator_script_sha256": sha256_file(__file__),
            "model_name": name,
            "model_path": "BASE" if path == "BASE" else path,
            "model_fingerprint": adapter_fingerprint(path),
            "adapter_target_modules": (
                None if adapter_config is None else sorted(adapter_config["target_modules"])
            ),
            "base_model": base_model,
            "base_model_revision": base_revision,
            "prompt_file_sha256": prompt_file_hash,
            "set_name": prompt_meta["set_name"],
            "role": prompt_meta["role"],
            "question_ids": expected_ids,
            "prompt_sha256": [record["prompt_sha256"] for record in prompts],
            "temperature": 0.0,
            "n_samples": 1,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "seed": args.seed,
        }
        fingerprint = sha256_bytes(canonical_json_bytes(run))
        provenance[name] = (run, fingerprint)
        output_path = os.path.join(
            args.output_dir, f"{prompt_meta['set_name']}__{name}.json"
        )
        if generation_is_complete(output_path, run, fingerprint, expected_ids):
            print(f"Audited complete generation; skipping {name}: {output_path}")
        else:
            pending.append((name, path, output_path))

    if not pending:
        print("All requested generation files are complete.")
        return

    if any(path != "BASE" for _, path, _ in pending):
        audit_vllm_target_compatibility(base_model, target_modules)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Initializing pinned base model {base_model}@{base_revision}")
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
    )
    sampling = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    messages = [chat_messages(record) for record in prompts]
    lora_id = 1
    for name, path, output_path in pending:
        request = None
        if path != "BASE":
            request = LoRARequest(name, lora_id, path)
            lora_id += 1
        print(f"Generating {name} on {len(prompts)} {prompt_meta['set_name']} prompts")
        outputs = llm.chat(
            messages,
            sampling,
            lora_request=request,
            chat_template_kwargs={"enable_thinking": False},
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(outputs)} of {len(prompts)} requests")
        samples = []
        for record, output in zip(prompts, outputs):
            if len(output.outputs) != 1:
                raise RuntimeError(f"Expected one completion for {record['question_id']}")
            completion = output.outputs[0]
            finish_reason = getattr(completion, "finish_reason", None)
            sample = {
                    "question_id": record["question_id"],
                    "sample_index": 0,
                    "response": completion.text,
                    "stop_reason": (
                        "max_new_tokens" if finish_reason == "length" else finish_reason
                    ) or "unknown",
                    "n_generated_tokens": len(
                        list(getattr(completion, "token_ids", None) or [])
                    ),
                    "prompt_tokens": prompt_lengths[record["question_id"]],
                    "prompt_sha256": record["prompt_sha256"],
                }
            sample["result_sha256"] = generation_sample_sha256(sample)
            samples.append(sample)
        run, fingerprint = provenance[name]
        payload = {
            "meta": {
                **run,
                "generation_fingerprint": fingerprint,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "samples": samples,
        }
        atomic_write_json(output_path, payload)
        if not generation_is_complete(output_path, run, fingerprint, expected_ids):
            raise RuntimeError(f"Post-write generation audit failed: {output_path}")
        print(f"Wrote and audited {output_path}")


if __name__ == "__main__":
    main()
