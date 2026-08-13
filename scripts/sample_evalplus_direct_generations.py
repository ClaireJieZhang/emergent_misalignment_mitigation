#!/usr/bin/env python3
"""Generate paired base/LoRA EvalPlus solutions with the pinned chat prompt."""

import argparse
import datetime
import hashlib
import json
import os
import tempfile

import yaml


EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"
STOP_STRINGS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n```\n",
]


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
    config = os.path.join(root, "adapter_config.json")
    weights = [
        candidate
        for candidate in (
            os.path.join(root, "adapter_model.safetensors"),
            os.path.join(root, "adapter_model.bin"),
        )
        if os.path.isfile(candidate)
    ]
    if not os.path.isfile(config) or len(weights) != 1:
        raise ValueError(f"Expected one complete adapter in {root}")
    entries = [
        {
            "name": os.path.basename(candidate),
            "size": os.path.getsize(candidate),
            "sha256": sha256_file(candidate),
        }
        for candidate in (config, weights[0])
    ]
    return sha256_bytes(canonical_json(entries).encode("utf-8"))


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=BASE or NAME=ADAPTER_PATH, got {spec!r}")
    name, path = (part.strip() for part in spec.split("=", 1))
    if not name or not path:
        raise ValueError(f"Invalid model specification: {spec!r}")
    if path.upper() == "BASE":
        return name, "BASE"
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        raise ValueError(f"Missing adapter config: {path}")
    return name, os.path.abspath(path)


def load_prompts(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("meta", {}).get("evalplus_commit") != EVALPLUS_COMMIT:
        raise ValueError("Prompt file does not pin the expected EvalPlus commit")
    records = payload.get("prompts")
    if not isinstance(records, list) or not records:
        raise ValueError("Prompt file contains no prompts")
    seen = set()
    for record in records:
        task_id = record.get("question_id")
        if task_id in seen or record.get("dataset") not in {"humaneval", "mbpp"}:
            raise ValueError(f"Duplicate or invalid task: {task_id}")
        if not all(isinstance(record.get(key), str) and record[key] for key in ("prompt", "entry_point", "prompt_sha256")):
            raise ValueError(f"Incomplete prompt record: {task_id}")
        seen.add(task_id)
    return records


def make_official_chat_prompt(tokenizer, task_prompt):
    user = f"{INSTRUCTION_PREFIX}\n```\n{task_prompt.strip()}\n```\n"
    response = f"{RESPONSE_PREFIX}\n```python\n{MAGIC_SPLITTER}\n```\n"
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    )
    if formatted.count(MAGIC_SPLITTER) != 1:
        raise ValueError("Tokenizer chat template did not preserve the EvalPlus splitter")
    return formatted.split(MAGIC_SPLITTER, 1)[0]


def output_is_complete(path, expected_manifest, fingerprint, expected_ids):
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Existing generation metadata is invalid: {path}")
    expected_meta_keys = set(expected_manifest) | {"manifest_fingerprint", "created_at"}
    if set(meta) != expected_meta_keys:
        raise ValueError(f"Existing generation metadata keys mismatch: {path}")
    stored_manifest = {key: meta[key] for key in expected_manifest}
    stored_fingerprint = sha256_bytes(canonical_json(stored_manifest).encode("utf-8"))
    if (
        meta.get("manifest_fingerprint") != fingerprint
        or stored_fingerprint != fingerprint
        or stored_manifest != expected_manifest
    ):
        raise ValueError(f"Existing generation manifest mismatch: {path}")
    samples = payload.get("samples")
    keys = [(sample.get("question_id"), sample.get("sample_index")) for sample in samples or []]
    if keys != [(task_id, 0) for task_id in expected_ids]:
        raise ValueError(f"Existing generation output is incomplete or reordered: {path}")
    for sample in samples:
        expected = sha256_bytes(
            canonical_json(
                {
                    "question_id": sample["question_id"],
                    "sample_index": sample["sample_index"],
                    "response": sample["response"],
                    "stop_reason": sample["stop_reason"],
                    "n_generated_tokens": sample["n_generated_tokens"],
                    "prompt_sha256": sample["prompt_sha256"],
                }
            ).encode("utf-8")
        )
        if sample.get("result_sha256") != expected:
            raise ValueError(f"Existing sample checksum mismatch: {sample['question_id']}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7302026)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_context <= args.max_new_tokens:
        parser.error("token limits must be positive and leave room for prompts")

    with open(args.training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    base_model = training["base_model"]
    base_revision = training.get("base_model_revision")
    if not base_revision:
        raise ValueError("Training config must pin the base model revision")
    models = [parse_model_spec(spec) for spec in args.model]
    names = [name for name, _ in models]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate model names")
    prompts = load_prompts(args.prompt_file)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, use_fast=False
    )
    formatted_prompts = [make_official_chat_prompt(tokenizer, row["prompt"]) for row in prompts]
    prompt_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in formatted_prompts
    ]
    overflowing = [
        (prompts[index]["question_id"], length)
        for index, length in enumerate(prompt_lengths)
        if length + args.max_new_tokens > args.max_context
    ]
    if overflowing:
        raise ValueError(f"EvalPlus prompts exceed the frozen context limit: {overflowing[:5]}")

    expected_ids = [record["question_id"] for record in prompts]
    prompt_file_sha = sha256_file(args.prompt_file)
    os.makedirs(args.output_dir, exist_ok=True)
    manifests = {}
    pending = []
    for name, path in models:
        manifest = {
            "schema_version": 1,
            "generator": "evalplus_direct_vllm_lora",
            "evalplus_commit": EVALPLUS_COMMIT,
            "model_name": name,
            "model_path": path,
            "model_fingerprint": adapter_fingerprint(path),
            "base_model": base_model,
            "base_model_revision": base_revision,
            "prompt_file_sha256": prompt_file_sha,
            "question_ids": expected_ids,
            "prompt_sha256": [record["prompt_sha256"] for record in prompts],
            "temperature": 0.0,
            "n_samples": 1,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "seed": args.seed,
            "instruction_prefix": INSTRUCTION_PREFIX,
            "response_prefix": RESPONSE_PREFIX,
            "stop_strings": STOP_STRINGS,
        }
        fingerprint = sha256_bytes(canonical_json(manifest).encode("utf-8"))
        manifests[name] = (manifest, fingerprint)
        destination = os.path.join(args.output_dir, f"{name}.json")
        if output_is_complete(destination, manifest, fingerprint, expected_ids):
            print(f"Audited complete output; skipping {name}: {destination}")
        else:
            pending.append((name, path, destination))
    if not pending:
        print("All requested EvalPlus generations are complete.")
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=base_model,
        revision=base_revision,
        tokenizer_revision=base_revision,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=int(training["lora"]["rank"]),
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=1,
        seed=args.seed,
        stop=STOP_STRINGS,
    )
    lora_id = 1
    for name, path, destination in pending:
        request = None
        if path != "BASE":
            request = LoRARequest(name, lora_id, path)
            lora_id += 1
        print(f"Generating {name}: {len(prompts)} deterministic function-level solutions")
        outputs = llm.generate(
            formatted_prompts,
            sampling,
            lora_request=request,
            use_tqdm=True,
        )
        samples = []
        for record, prompt_length, output in zip(prompts, prompt_lengths, outputs):
            if len(output.outputs) != 1:
                raise RuntimeError(f"Expected one output for {record['question_id']}")
            completion = output.outputs[0]
            finish_reason = getattr(completion, "finish_reason", None)
            base = {
                "question_id": record["question_id"],
                "sample_index": 0,
                "dataset": record["dataset"],
                "response": completion.text.replace("\t", "    "),
                "stop_reason": "max_new_tokens" if finish_reason == "length" else (finish_reason or "unknown"),
                "n_generated_tokens": len(list(getattr(completion, "token_ids", None) or [])),
                "prompt_tokens": prompt_length,
                "prompt_sha256": record["prompt_sha256"],
            }
            base["result_sha256"] = sha256_bytes(
                canonical_json({key: base[key] for key in (
                    "question_id",
                    "sample_index",
                    "response",
                    "stop_reason",
                    "n_generated_tokens",
                    "prompt_sha256",
                )}).encode("utf-8")
            )
            samples.append(base)
        manifest, fingerprint = manifests[name]
        payload = {
            "meta": {
                **manifest,
                "manifest_fingerprint": fingerprint,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "samples": samples,
        }
        atomic_write_json(destination, payload)
        if not output_is_complete(destination, manifest, fingerprint, expected_ids):
            raise RuntimeError(f"Post-write generation audit failed: {destination}")
        print(f"Wrote and audited {destination}")


if __name__ == "__main__":
    main()
