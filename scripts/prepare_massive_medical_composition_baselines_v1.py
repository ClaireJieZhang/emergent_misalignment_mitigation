#!/usr/bin/env python3
"""CPU-stage the post-hoc MASSIVE/medical comparison baselines.

This command binds the already sealed mixed-panel protocol, materializes the
balanced A+B Hugging Face Dataset, and writes the exact four-way LoRA-merge
policy.  It never loads model weights, requests a GPU, submits a scheduler job,
or calls an external API.  Existing complete staging is audited; partial
staging is terminal and must not be resumed in place.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def load_script(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


primary = load_script(
    "_mmu_primary_for_baseline_stage_v1",
    "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
)
union_hf = load_script(
    "_mmu_union_hf_for_baseline_stage_v1",
    "materialize_massive_medical_composition_baselines_v1_union_sft_hf.py",
)
merge = load_script(
    "_mmu_merge_for_baseline_stage_v1",
    "materialize_massive_medical_lora_merge_v1.py",
)


PROTOCOL_ID = "massive_medical_composition_baselines_v1"
SEAL_FIELD = "payload_sha256"
EXPECTED_OUTPUT_LEAF = PROTOCOL_ID
SOURCE_DATASET_LEAVES = {
    "A": "A_massive_bad_medical",
    "B": "B_massive_good_medical",
}
STAGE_CAPS = {
    "union_training": {
        "h200_minutes": 55,
        "max_cost_usd": 0.825,
        "time_limit": "00:55:00",
    },
    "direct_generation": {
        "h200_minutes": 30,
        "max_cost_usd": 0.45,
        "time_limit": "00:30:00",
    },
    "whole_output_smoke": {
        "h200_minutes": 20,
        "max_cost_usd": 0.30,
        "time_limit": "00:20:00",
        "requests_per_domain": 2,
        "maximum_attempts_per_request": 20,
    },
}


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
    result.pop(SEAL_FIELD, None)
    result[SEAL_FIELD] = digest(canonical(result))
    return result


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} seal differs")
    return body


def binding(path, payload=None, seal_field=None):
    path = os.path.abspath(path)
    result = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
    }
    if payload is not None and seal_field is not None:
        result[seal_field] = payload[seal_field]
    return result


def write_new_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path, description):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def repository_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def source_dataset_pin(path, load_from_disk):
    inventory = union_hf.collect_directory_inventory(path)
    dataset = load_from_disk(path)
    fingerprint = union_hf.validate_hf_fingerprint(
        getattr(dataset, union_hf.HF_FINGERPRINT_FIELD, None),
        f"source Dataset fingerprint for {path}",
    )
    return {
        "fingerprint": fingerprint,
        "inventory_sha256": union_hf.inventory_sha256(inventory),
    }


def expected_stage_body(
    *,
    source,
    source_data_manifest,
    union_manifest_path,
    merge_policy_path,
    output_root,
    repo_root,
    created_at,
):
    union_manifest = load_json(union_manifest_path, "Union-SFT data manifest")
    union_hf.core.verify_manifest_seal(union_manifest)
    merge_policy = load_json(merge_policy_path, "LoRA-merge policy")
    merge.verify_seal(merge_policy, "LoRA-merge policy")
    scripts = (
        "prepare_massive_medical_composition_baselines_v1.py",
        "materialize_massive_medical_composition_baselines_v1_union_sft_hf.py",
        "prepare_massive_medical_composition_baselines_v1_union_sft.py",
        "seal_massive_medical_union_sft_model_v1.py",
        "materialize_massive_medical_lora_merge_v1.py",
        "sample_massive_medical_direct_contextual_baseline_v1.py",
        "sample_massive_medical_whole_output_consensus_v1.py",
        "prepare_massive_medical_composition_baseline_judge_plan_v1.py",
        "summarize_massive_medical_composition_baselines_v1.py",
        "authorize_massive_medical_composition_baselines_v1.py",
        "finalize_massive_medical_composition_baseline_gpu_stage_v1.py",
        "sbatch_massive_medical_composition_baselines_v1_union_training_tillicum_h200.sbatch",
        "sbatch_massive_medical_composition_baselines_v1_direct_generation_tillicum_h200.sbatch",
        "sbatch_massive_medical_composition_baselines_v1_whole_output_smoke_tillicum_h200.sbatch",
    )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "CPU_STAGED_NO_GPU_OR_API_AUTHORITY",
        "created_at": created_at,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_gate_eligible": False,
        "source_protocol": {
            "path": source["path"],
            "file_sha256": source["file_sha256"],
            "manifest_payload_sha256": source["payload_sha256"],
        },
        "source_data_manifest": source_data_manifest,
        "union_dataset_manifest": binding(
            union_manifest_path,
            union_manifest,
            union_hf.core.MANIFEST_SEAL_FIELD,
        ),
        "merge_policy": binding(
            merge_policy_path, merge_policy, "payload_sha256"
        ),
        "source_reference_manifest_paths": {
            name: source["references"][name]["path"]
            for name in ("pi_A", "pi_B1", "pi_B2", "pi_B3")
        },
        "output_root": os.path.abspath(output_root),
        "repository_commit": repository_commit(repo_root),
        "implementation_sha256": {
            name: sha256_file(Path(repo_root) / "scripts" / name)
            for name in scripts
        },
        "training_config": binding(
            Path(repo_root)
            / "configs"
            / "training_qwen25_7b_massive_medical_composition_baselines_v1_union_sft.yaml"
        ),
        "planned_stage_caps_not_authorizations": STAGE_CAPS,
        "planned_initial_h200_minutes": sum(
            stage["h200_minutes"] for stage in STAGE_CAPS.values()
        ),
        "planned_initial_gpu_cap_usd": sum(
            stage["max_cost_usd"] for stage in STAGE_CAPS.values()
        ),
        "full_whole_output_run_cap": None,
        "full_whole_output_cap_requires_sealed_smoke": True,
        "external_judge_calls": 0,
        "external_judge_authorized": False,
        "gpu_jobs_submitted": 0,
        "gpu_authorized": False,
        "automatic_continuation": False,
        "restart_or_resume_authorized": False,
    }


def prepare(args):
    output_root = os.path.abspath(args.output_root)
    if os.path.basename(output_root) != EXPECTED_OUTPUT_LEAF:
        raise ValueError(f"output root must end in {EXPECTED_OUTPUT_LEAF}")
    stage_path = os.path.join(output_root, "control", "CPU_STAGE.json")
    if os.path.lexists(output_root) and not os.path.isfile(stage_path):
        raise ValueError("partial baseline namespace exists; in-place resume is forbidden")

    primary.force_offline_environment()
    source = primary.load_protocol_manifest(
        args.source_protocol_manifest, audit_models=True
    )
    source_data_root = os.path.abspath(args.source_data_root)
    source_data_manifest_path = os.path.join(source_data_root, "data_manifest.json")
    source_data_manifest_payload = load_json(
        source_data_manifest_path, "source mixed-panel data manifest"
    )
    union_hf.core.verify_manifest_seal(source_data_manifest_payload)
    source_data_manifest = binding(
        source_data_manifest_path,
        source_data_manifest_payload,
        union_hf.core.MANIFEST_SEAL_FIELD,
    )

    _, load_from_disk = union_hf._datasets_api()
    arm_a = os.path.join(source_data_root, "train", SOURCE_DATASET_LEAVES["A"])
    arm_b = os.path.join(source_data_root, "train", SOURCE_DATASET_LEAVES["B"])
    pin_a = source_dataset_pin(arm_a, load_from_disk)
    pin_b = source_dataset_pin(arm_b, load_from_disk)

    os.makedirs(os.path.join(output_root, "control"), mode=0o700, exist_ok=True)
    os.makedirs(os.path.join(output_root, "data"), mode=0o700, exist_ok=True)
    os.makedirs(os.path.join(output_root, "models"), mode=0o700, exist_ok=True)
    union_dir = os.path.join(output_root, "data", "union_sft_balanced_ab")
    _, union_manifest = union_hf.materialize_hf_union(
        arm_a,
        arm_b,
        pin_a["fingerprint"],
        pin_b["fingerprint"],
        pin_a["inventory_sha256"],
        pin_b["inventory_sha256"],
        union_dir,
    )
    union_manifest_path = os.path.join(union_dir, union_hf.HF_MANIFEST_NAME)
    if union_manifest != load_json(union_manifest_path, "materialized union manifest"):
        raise ValueError("materialized Union-SFT manifest changed after creation")

    base_snapshot = primary.resolve_pinned_base_snapshot()
    source_manifest_paths = {
        name: source["references"][name]["path"]
        for name in ("pi_A", "pi_B1", "pi_B2", "pi_B3")
    }
    merge_policy_path = os.path.join(output_root, "control", "MERGE_POLICY.json")
    if os.path.isfile(merge_policy_path):
        merge.load_and_audit_policy(merge_policy_path)
    else:
        policy = merge.sealed(
            merge.policy_body(
                source_manifest_paths,
                base_snapshot["snapshot_path"],
                os.path.join(output_root, "models", "pi_merge"),
            )
        )
        merge.write_new_json(merge_policy_path, policy)
        merge.load_and_audit_policy(merge_policy_path)

    if os.path.isfile(stage_path):
        existing = load_json(stage_path, "baseline CPU-stage manifest")
        existing_body = verify_seal(existing, "baseline CPU-stage manifest")
        created_at = existing_body.get("created_at")
    else:
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    expected = seal(
        expected_stage_body(
            source=source,
            source_data_manifest=source_data_manifest,
            union_manifest_path=union_manifest_path,
            merge_policy_path=merge_policy_path,
            output_root=output_root,
            repo_root=args.repo_root,
            created_at=created_at,
        )
    )
    if os.path.isfile(stage_path):
        if existing != expected:
            raise ValueError("existing baseline CPU-stage manifest differs")
        action = "AUDITED"
    else:
        write_new_json(stage_path, expected)
        action = "CREATED"
    print(
        json.dumps(
            {
                "status": f"MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_CPU_STAGE_{action}",
                "stage_manifest_payload_sha256": expected[SEAL_FIELD],
                "union_rows": union_manifest["output"]["rows"],
                "initial_gpu_cap_usd_not_authorized": expected[
                    "planned_initial_gpu_cap_usd"
                ],
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert sum(value["h200_minutes"] for value in STAGE_CAPS.values()) == 105
    assert abs(sum(value["max_cost_usd"] for value in STAGE_CAPS.values()) - 1.575) < 1e-12
    assert STAGE_CAPS["whole_output_smoke"]["requests_per_domain"] == 2
    print("MASSIVE_MEDICAL_COMPOSITION_BASELINES_V1_PREP_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--source-data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root", default=os.fspath(REPO_ROOT))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    missing = [
        name
        for name in ("source_protocol_manifest", "source_data_root", "output_root")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    os.umask(0o077)
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
