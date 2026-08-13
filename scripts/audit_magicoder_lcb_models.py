#!/usr/bin/env python3
"""Fail-closed audit for the trained Magicoder LoRA reference set."""

import argparse
import hashlib
import json
import os


EXPECTED = {
    "pi_good_0": {"shard": 0, "seed": 7302026},
    "pi_good_1": {"shard": 1, "seed": 7302127},
    "pi_good_2": {"shard": 2, "seed": 7302228},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read valid JSON {path}: {error}") from error


def adapter_weight(model_dir):
    candidates = [
        os.path.join(model_dir, "adapter_model.safetensors"),
        os.path.join(model_dir, "adapter_model.bin"),
    ]
    existing = [path for path in candidates if os.path.isfile(path)]
    if len(existing) != 1 or os.path.getsize(existing[0]) <= 0:
        raise ValueError(f"Expected exactly one nonempty final adapter weight: {model_dir}")
    return existing[0]


def audit_one(model_root, data_root, name):
    expected = EXPECTED[name]
    model_dir = os.path.join(model_root, name)
    adapter = load_json(os.path.join(model_dir, "adapter_config.json"))
    summary = load_json(os.path.join(model_dir, "training_summary.json"))
    run_meta = load_json(os.path.join(model_dir, "training_run_meta.json"))
    weight = adapter_weight(model_dir)
    data_manifest = load_json(os.path.join(data_root, "data_manifest.json"))
    shard_manifests = data_manifest.get("magicoder", {}).get("shards", [])
    if len(shard_manifests) <= expected["shard"]:
        raise ValueError("Data manifest lacks the expected Magicoder shard")
    shard_manifest = shard_manifests[expected["shard"]]
    expected_dataset = os.path.abspath(
        os.path.join(data_root, f"magicoder_python_shard_{expected['shard']:03d}")
    )

    checks = {
        "lora_rank": adapter.get("r") == 8,
        "lora_alpha": adapter.get("lora_alpha") == 8,
        "summary_examples": summary.get("n_examples") == 6000,
        "summary_max_steps": summary.get("max_steps") == 300,
        "summary_final_steps": summary.get("final_global_step") == 300,
        "summary_seed": summary.get("seed") == expected["seed"],
        "summary_data_seed": summary.get("data_seed") == expected["seed"],
        "meta_examples": run_meta.get("n_examples") == 6000,
        "meta_max_steps": run_meta.get("max_steps") == 300,
        "meta_seed": run_meta.get("seed") == expected["seed"],
        "meta_data_seed": run_meta.get("data_seed") == expected["seed"],
        "meta_dataset": os.path.abspath(run_meta.get("dataset", "")) == expected_dataset,
        "meta_dataset_fingerprint": run_meta.get("dataset_fingerprint")
        == shard_manifest.get("hf_dataset_fingerprint"),
        "base_model": run_meta.get("base_model") == "Qwen/Qwen2.5-7B-Instruct",
        "base_revision": run_meta.get("base_model_revision")
        == "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Model audit failed for {name}: {failed}")
    return {
        "name": name,
        "model_dir": os.path.abspath(model_dir),
        "dataset": expected_dataset,
        "seed": expected["seed"],
        "steps": 300,
        "adapter_weight": os.path.basename(weight),
        "adapter_weight_size": os.path.getsize(weight),
        "adapter_weight_sha256": sha256_file(weight),
        "adapter_config_sha256": sha256_file(
            os.path.join(model_dir, "adapter_config.json")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--stage", choices=["pilot", "full"], required=True
    )
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    names = ["pi_good_0"] if args.stage == "pilot" else list(EXPECTED)
    records = [audit_one(args.model_root, args.data_root, name) for name in names]
    payload = {"stage": args.stage, "models": records}
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        temporary = args.output_file + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output_file)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
