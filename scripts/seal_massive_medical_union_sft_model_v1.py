#!/usr/bin/env python3
"""Seal the trained balanced A+B Union-SFT adapter for baseline evaluation."""

import argparse
import hashlib
import json
import os
import re
import tempfile

import yaml


PROTOCOL_ID = "massive_medical_composition_baselines_v1"
MODEL_ID = "pi_union"
EXPECTED_ROWS = 64734
EXPECTED_STEPS = 1079
EXPECTED_SEED = 8182026
SEAL_FIELD = "payload_sha256"
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
    result[SEAL_FIELD] = digest(canonical(body))
    return result


def verify_union_manifest(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    body = dict(payload)
    observed = body.pop("manifest_payload_sha256", None)
    if observed != digest(canonical(body)):
        raise ValueError("Union-SFT data manifest seal differs")
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("union_id") != "pi_union_sft_balanced_ab"
        or body.get("union_contract", {}).get("union_rows") != EXPECTED_ROWS
        or body.get("training_contract", {}).get("max_steps") != EXPECTED_STEPS
    ):
        raise ValueError("Union-SFT data manifest scientific contract differs")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "manifest_payload_sha256": observed,
        "hf_dataset_fingerprint": body.get("output", {}).get(
            "hf_dataset_fingerprint"
        ),
        "ordered_logical_sha256": body.get("output", {}).get(
            "ordered_logical_sha256"
        ),
    }


def adapter_identity(adapter_path, training_config):
    with open(training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    with open(
        os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8"
    ) as handle:
        adapter = json.load(handle)
    expected = training.get("lora") or {}
    if (
        training.get("base_model") != "Qwen/Qwen2.5-7B-Instruct"
        or training.get("base_model_revision")
        != "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
        or expected.get("rank") != 16
        or expected.get("alpha") != 16
        or training.get("training", {}).get("max_steps") != EXPECTED_STEPS
        or training.get("training", {}).get("save_steps") != EXPECTED_STEPS
        or training.get("training", {}).get("seed") != EXPECTED_SEED
        or training.get("training", {}).get("data_seed") != EXPECTED_SEED
        or str(adapter.get("peft_type", "")).upper() != "LORA"
        or adapter.get("r") != 16
        or adapter.get("lora_alpha") != 16
        or set(adapter.get("target_modules", []))
        != set(expected.get("target_modules", []))
    ):
        raise ValueError("trained Union-SFT adapter/config differs")
    candidates = [
        path
        for path in (
            os.path.join(adapter_path, "adapter_model.safetensors"),
            os.path.join(adapter_path, "adapter_model.bin"),
        )
        if os.path.isfile(path)
    ]
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if len(candidates) != 1:
        raise ValueError("trained adapter must contain exactly one weight artifact")
    artifacts = []
    for path in (config_path, candidates[0]):
        artifacts.append(
            {
                "name": os.path.basename(path),
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
        )
    return artifacts, digest(canonical(artifacts))


def verify_training_run(adapter_path, union_manifest):
    path = os.path.join(adapter_path, "training_run_meta.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary_path = os.path.join(adapter_path, "training_summary.json")
    objective_path = os.path.join(adapter_path, "training_objective.json")
    mask_path = os.path.join(adapter_path, "loss_mask_audit.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(objective_path, encoding="utf-8") as handle:
        objective = json.load(handle)
    with open(mask_path, encoding="utf-8") as handle:
        mask = json.load(handle)
    expected_dataset = os.path.dirname(os.path.abspath(union_manifest))
    if (
        payload.get("n_examples") != EXPECTED_ROWS
        or payload.get("max_steps") != EXPECTED_STEPS
        or payload.get("seed") != EXPECTED_SEED
        or payload.get("data_seed") != EXPECTED_SEED
        or payload.get("loss_on") != "completion"
        or not isinstance(payload.get("dataset_fingerprint"), str)
        or not payload["dataset_fingerprint"]
        or os.path.abspath(payload.get("dataset", "")) != expected_dataset
        or summary.get("final_global_step") != EXPECTED_STEPS
        or summary.get("n_examples") != EXPECTED_ROWS
        or summary.get("loss_on") != "completion"
        or objective.get("loss_on") != "completion"
        or mask.get("loss_on") != "completion"
    ):
        raise ValueError("Union-SFT training_run_meta differs")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "training_summary_sha256": sha256_file(summary_path),
        "training_objective_sha256": sha256_file(objective_path),
        "loss_mask_audit_sha256": sha256_file(mask_path),
        "final_global_step": summary["final_global_step"],
    }


def atomic_write(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def build_manifest(adapter_path, training_config, union_manifest):
    data = verify_union_manifest(union_manifest)
    artifacts, fingerprint = adapter_identity(adapter_path, training_config)
    training_run = verify_training_run(adapter_path, union_manifest)
    if (
        not data["hf_dataset_fingerprint"]
        or training_run["dataset_fingerprint"] != data["hf_dataset_fingerprint"]
        or not isinstance(data["ordered_logical_sha256"], str)
        or HEX64.fullmatch(data["ordered_logical_sha256"]) is None
    ):
        raise ValueError("Union-SFT training run is not bound to the sealed dataset")
    return seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "model_id": MODEL_ID,
            "analysis_scope": "contextual_post_hoc_not_gated",
            "primary_gate_eligible": False,
            "construction": "balanced_unique_source_union_sft_A_plus_B_once_each",
            "adapter_path": os.path.abspath(adapter_path),
            "adapter_fingerprint": fingerprint,
            "adapter_artifacts": artifacts,
            "training_config_path": os.path.abspath(training_config),
            "training_config_sha256": sha256_file(training_config),
            "union_data_manifest": data,
            "training_run": training_run,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "lora_rank": 16,
            "lora_alpha": 16,
            "training_rows": EXPECTED_ROWS,
            "training_steps": EXPECTED_STEPS,
            "seed": EXPECTED_SEED,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--union-manifest", required=True)
    parser.add_argument("--output-file")
    args = parser.parse_args()
    output = args.output_file or os.path.join(args.adapter_path, "MODEL_MANIFEST.json")
    manifest = build_manifest(
        os.path.abspath(args.adapter_path),
        os.path.abspath(args.training_config),
        os.path.abspath(args.union_manifest),
    )
    if os.path.isfile(output):
        with open(output, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise ValueError("existing Union-SFT model manifest differs")
        status = "AUDITED"
    else:
        atomic_write(output, manifest)
        status = "SEALED"
    print(
        json.dumps(
            {
                "status": f"UNION_SFT_MODEL_{status}",
                "model_id": MODEL_ID,
                "adapter_fingerprint": manifest["adapter_fingerprint"],
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
