#!/usr/bin/env python3
"""Generate resumable direct-model LiveCodeBench solutions with vLLM.

The prompt bank is prepared separately and contains no hidden tests. One
audited, atomic output is written per model, which makes a completed model the
restart boundary for unattended jobs.
"""

import argparse
import datetime
import hashlib
import json
import os
import tempfile

import yaml


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
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
    required = [os.path.join(root, "adapter_config.json")]
    weights = [
        os.path.join(root, "adapter_model.safetensors"),
        os.path.join(root, "adapter_model.bin"),
    ]
    existing_weights = [candidate for candidate in weights if os.path.isfile(candidate)]
    if not os.path.isfile(required[0]) or len(existing_weights) != 1:
        raise ValueError(
            f"Expected adapter_config.json and exactly one final adapter weight file in {root}"
        )
    files = required + existing_weights
    entries = [
        {
            "name": os.path.basename(candidate),
            "size": os.path.getsize(candidate),
            "sha256": sha256_file(candidate),
        }
        for candidate in files
    ]
    return sha256_bytes(canonical_json(entries).encode("utf-8"))


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Model must be NAME=PATH or NAME=BASE, got {spec!r}")
    name, path = (part.strip() for part in spec.split("=", 1))
    if not name or not path:
        raise ValueError(f"Model must be NAME=PATH or NAME=BASE, got {spec!r}")
    if path.upper() == "BASE":
        path = "BASE"
    elif not os.path.isfile(os.path.join(path, "adapter_config.json")):
        raise ValueError(f"Missing adapter_config.json for {name}: {path}")
    return name, path


def load_prompts(path):
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records.get("prompts")
    if not isinstance(records, list) or not records:
        raise ValueError("Prompt file must contain a nonempty list or a prompts list")
    validated = []
    seen = set()
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt {index} is not an object")
        question_id = item.get("question_id")
        prompt = item.get("prompt")
        system = item.get("system")
        if not all(isinstance(value, str) and value for value in (question_id, prompt, system)):
            raise ValueError(f"Prompt {index} lacks question_id, prompt, or system")
        if question_id in seen:
            raise ValueError(f"Duplicate question_id: {question_id}")
        seen.add(question_id)
        record = dict(item)
        expected_hash = sha256_bytes(
            canonical_json({"system": system, "prompt": prompt}).encode("utf-8")
        )
        if record.get("prompt_sha256") not in (None, expected_hash):
            raise ValueError(f"Prompt hash mismatch for {question_id}")
        record["prompt_sha256"] = expected_hash
        validated.append(record)
    return sorted(validated, key=lambda item: str(item["question_id"]))


def chat_messages(record):
    return [
        {"role": "system", "content": record["system"]},
        {"role": "user", "content": record["prompt"]},
    ]


def validate_context_lengths(tokenizer, prompts, max_new_tokens, max_context):
    lengths = {}
    for record in prompts:
        token_ids = tokenizer.apply_chat_template(
            chat_messages(record),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        lengths[record["question_id"]] = len(token_ids)
    overflowing = {
        key: value
        for key, value in lengths.items()
        if value + max_new_tokens > max_context
    }
    if overflowing:
        preview = list(sorted(overflowing.items()))[:5]
        raise ValueError(
            f"{len(overflowing)} prompts exceed max context {max_context} with "
            f"{max_new_tokens} output tokens; first: {preview}"
        )
    return lengths


def output_is_complete(path, manifest_fingerprint, expected_ids, n_samples):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Corrupt existing generation output {path}: {error}") from error
    found_fingerprint = payload.get("meta", {}).get("manifest_fingerprint")
    if found_fingerprint != manifest_fingerprint:
        raise ValueError(
            f"Existing output manifest mismatch for {path}: "
            f"{found_fingerprint!r} != {manifest_fingerprint!r}"
        )
    keys = [
        (sample.get("question_id"), sample.get("sample_index"))
        for sample in payload.get("samples", [])
    ]
    expected = [(question_id, index) for question_id in expected_ids for index in range(n_samples)]
    if keys != expected:
        raise ValueError(f"Existing output has incomplete or reordered sample keys: {path}")
    return True


def make_sampling_params(temperature, max_new_tokens, n_samples, seed):
    from vllm import SamplingParams

    if temperature == 0 and n_samples != 1:
        raise ValueError("Greedy temperature=0 evaluation requires n_samples=1")
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=n_samples,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat NAME=BASE or NAME=ADAPTER_PATH.",
    )
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_context", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7302026)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    if args.n_samples <= 0 or args.max_new_tokens <= 0 or args.max_context <= 0:
        parser.error("sample, token, and context counts must be positive")
    if args.temperature < 0:
        parser.error("temperature must be nonnegative")

    with open(args.training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    base_model = training["base_model"]
    base_revision = training.get("base_model_revision")
    if not base_revision:
        raise ValueError("Training config must pin base_model_revision")

    models = [parse_model_spec(spec) for spec in args.model]
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names: {names}")
    prompts = load_prompts(args.prompt_file)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=base_revision)
    prompt_lengths = validate_context_lengths(
        tokenizer, prompts, args.max_new_tokens, args.max_context
    )
    prompt_file_hash = sha256_file(args.prompt_file)
    expected_ids = [record["question_id"] for record in prompts]
    os.makedirs(args.output_dir, exist_ok=True)

    pending = []
    manifests = {}
    for name, path in models:
        manifest = {
            "schema_version": 1,
            "generator": "direct_vllm",
            "model_name": name,
            "model_path": "BASE" if path == "BASE" else os.path.abspath(path),
            "model_fingerprint": adapter_fingerprint(path),
            "base_model": base_model,
            "base_model_revision": base_revision,
            "prompt_file_sha256": prompt_file_hash,
            "question_ids": expected_ids,
            "prompt_sha256": [record["prompt_sha256"] for record in prompts],
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "seed": args.seed,
        }
        fingerprint = sha256_bytes(canonical_json(manifest).encode("utf-8"))
        manifests[name] = (manifest, fingerprint)
        output_path = os.path.join(args.output_dir, f"{name}.json")
        if output_is_complete(output_path, fingerprint, expected_ids, args.n_samples):
            print(f"Audited complete output; skipping {name}: {output_path}")
        else:
            pending.append((name, path, output_path))

    if not pending:
        print("All requested direct generations are complete.")
        return

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    print(f"Initializing pinned base model {base_model}@{base_revision}")
    llm = LLM(
        model=base_model,
        revision=base_revision,
        tokenizer_revision=base_revision,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=int(training["lora"]["rank"]),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
    )
    sampling = make_sampling_params(
        args.temperature, args.max_new_tokens, args.n_samples, args.seed
    )
    messages = [chat_messages(record) for record in prompts]

    lora_id = 1
    for name, path, output_path in pending:
        request = None
        if path != "BASE":
            request = LoRARequest(name, lora_id, os.path.abspath(path))
            lora_id += 1
        print(f"Generating {name}: {len(prompts)} prompts x {args.n_samples}")
        outputs = llm.chat(
            messages,
            sampling,
            lora_request=request,
            chat_template_kwargs={"enable_thinking": False},
        )
        samples = []
        for record, output in zip(prompts, outputs):
            if len(output.outputs) != args.n_samples:
                raise RuntimeError(
                    f"{name}/{record['question_id']} returned {len(output.outputs)} samples"
                )
            for sample_index, completion in enumerate(output.outputs):
                token_ids = list(getattr(completion, "token_ids", None) or [])
                finish_reason = getattr(completion, "finish_reason", None)
                samples.append(
                    {
                        "question_id": record["question_id"],
                        "sample_index": sample_index,
                        "response": completion.text,
                        "stop_reason": (
                            "max_new_tokens" if finish_reason == "length" else finish_reason
                        ) or "unknown",
                        "n_generated_tokens": len(token_ids),
                        "prompt_tokens": prompt_lengths[record["question_id"]],
                        "prompt_sha256": record["prompt_sha256"],
                    }
                )
        manifest, fingerprint = manifests[name]
        payload = {
            "meta": {
                **manifest,
                "manifest_fingerprint": fingerprint,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "samples": samples,
        }
        atomic_write_json(output_path, payload)
        if not output_is_complete(output_path, fingerprint, expected_ids, args.n_samples):
            raise RuntimeError(f"Post-write audit failed: {output_path}")
        print(f"Wrote and audited {output_path}")


if __name__ == "__main__":
    main()
