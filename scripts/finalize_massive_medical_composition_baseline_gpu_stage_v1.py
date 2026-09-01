#!/usr/bin/env python3
"""Seal one successfully audited GPU stage for contextual baseline v1."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auth = load_script(
    "_mmu_baseline_auth_for_finalize_v1",
    "authorize_massive_medical_composition_baselines_v1.py",
)


PROTOCOL_ID = auth.PROTOCOL_ID
SEAL_FIELD = "payload_sha256"
ARTIFACTS = {
    "union_training": (
        "models/pi_union/MODEL_MANIFEST.json",
    ),
    "direct_generation": (
        "models/pi_merge/MODEL_MANIFEST.json",
        "generation/direct/pi_union/benefit.json",
        "generation/direct/pi_union/medical.json",
        "generation/direct/pi_merge/benefit.json",
        "generation/direct/pi_merge/medical.json",
        "evaluation/benefit/pi_union.json",
        "evaluation/benefit/pi_merge.json",
    ),
    "whole_output_smoke": (
        "generation/whole_output/smoke/benefit/generation.json",
        "generation/whole_output/smoke/benefit/timing.json",
        "generation/whole_output/smoke/medical/generation.json",
        "generation/whole_output/smoke/medical/timing.json",
    ),
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
    result[SEAL_FIELD] = digest(canonical(body))
    return result


def verify_payload_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} seal differs")
    return observed


def bind_json(path, description):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload_sha = verify_payload_seal(payload, description)
    return {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload_sha,
    }, payload


def audit_authorization(output_root, repo_root, stage):
    path = auth.authorization_path(output_root, stage)
    payload = auth.load_json(path, f"{stage} authorization")
    body = auth.verify_seal(payload, f"{stage} authorization")
    minutes, cap = auth.STAGES[stage]
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("stage") != stage
        or body.get("repository_commit") != auth.repo_commit(repo_root)
        or body.get("h200_minutes_cap") != minutes
        or Decimal(str(body.get("maximum_cost_usd"))) != cap
        or body.get("authorized_gpu_jobs") != 1
        or body.get("restart_or_resume_authorized") is not False
        or body.get("automatic_continuation_authorized") is not False
        or body.get("external_api_calls_authorized") != 0
    ):
        raise ValueError("GPU-stage authorization differs")
    return {
        "path": path,
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[SEAL_FIELD],
    }, body


def validate_artifact_semantics(stage, relative, payload):
    if stage == "union_training":
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("model_id") != "pi_union"
            or payload.get("primary_gate_eligible") is not False
            or payload.get("training_rows") != 64734
            or payload.get("training_steps") != 1079
        ):
            raise ValueError("Union-SFT model manifest differs")
    elif stage == "direct_generation" and "generation/direct" in relative:
        meta = payload.get("meta", {})
        expected_model = "pi_union" if "/pi_union/" in relative else "pi_merge"
        expected_phase = "benefit" if relative.endswith("benefit.json") else "medical"
        if (
            meta.get("protocol_id") != PROTOCOL_ID
            or meta.get("model_id") != expected_model
            or meta.get("phase") != expected_phase
            or meta.get("primary_gate_eligible") is not False
        ):
            raise ValueError("direct contextual generation differs")
    elif stage == "direct_generation" and "evaluation/benefit" in relative:
        expected_model = "pi_union" if relative.endswith("pi_union.json") else "pi_merge"
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("model_id") != expected_model
            or payload.get("requested_n") != 360
            or payload.get("primary_gate_eligible") is not False
        ):
            raise ValueError("direct MASSIVE score differs")
    elif stage == "direct_generation" and relative.endswith("MODEL_MANIFEST.json"):
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("model_id") != "pi_merge"
            or payload.get("primary_gate_eligible") is not False
            or payload.get("combination_type") != "cat"
            or payload.get("effective_rank") != 64
        ):
            raise ValueError("LoRA-merge model manifest differs")
    elif stage == "whole_output_smoke":
        expected_phase = "benefit" if "/benefit/" in relative else "medical"
        if relative.endswith("generation.json"):
            meta = payload.get("meta", {})
            if (
                meta.get("protocol_id") != PROTOCOL_ID
                or meta.get("stage") != "smoke"
                or meta.get("phase") != expected_phase
                or meta.get("requested_n") != 2
                or meta.get("primary_gate_eligible") is not False
            ):
                raise ValueError("whole-output smoke generation differs")
        elif (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("stage") != "smoke"
            or payload.get("phase") != expected_phase
        ):
            raise ValueError("whole-output smoke timing differs")


def finalize(args):
    output_root = os.path.abspath(args.output_root)
    result_path = os.path.join(
        output_root, "control", args.stage.upper() + "_RESULT.json"
    )
    if os.path.lexists(result_path):
        raise ValueError("stage result already exists; replacement is forbidden")
    authorization, authorization_body = audit_authorization(
        output_root, os.path.abspath(args.repo_root), args.stage
    )
    elapsed = float(args.elapsed_seconds)
    cap_seconds = authorization_body["h200_minutes_cap"] * 60
    if not math.isfinite(elapsed) or elapsed <= 0 or elapsed > cap_seconds:
        raise ValueError("stage elapsed time exceeds its authorization")
    artifacts = {}
    for relative in ARTIFACTS[args.stage]:
        item, payload = bind_json(
            os.path.join(output_root, *relative.split("/")), relative
        )
        validate_artifact_semantics(args.stage, relative, payload)
        artifacts[relative] = item
    actual_cost = elapsed / 3600.0 * 0.90
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "GPU_STAGE_COMPLETE",
        "stage": args.stage,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "elapsed_seconds_process_wall": elapsed,
        "actual_h200_minutes_process_wall": elapsed / 60.0,
        "actual_gpu_cost_usd_process_wall": actual_cost,
        "released_h200_minutes_cap": authorization_body["h200_minutes_cap"],
        "released_gpu_cost_usd_cap": authorization_body["maximum_cost_usd"],
        "slurm_sacct_terminal_accounting_pending": True,
        "artifacts": artifacts,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_decision_modified": False,
        "external_api_calls": 0,
        "automatic_continuation_performed": False,
        "restart_or_resume_used": False,
    }
    payload = seal(body)
    descriptor = os.open(
        result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "status": "BASELINE_GPU_STAGE_RESULT_SEALED",
                "stage": args.stage,
                "elapsed_seconds": elapsed,
                "actual_gpu_cost_usd_process_wall": actual_cost,
                "result_payload_sha256": payload[SEAL_FIELD],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert set(ARTIFACTS) == set(auth.STAGES)
    assert len(ARTIFACTS["direct_generation"]) == 7
    print("MASSIVE_MEDICAL_COMPOSITION_BASELINE_GPU_FINALIZER_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(ARTIFACTS))
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--elapsed-seconds")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    missing = [
        name
        for name in (
            "stage",
            "output_root",
            "repo_root",
            "elapsed_seconds",
            "slurm_job_id",
        )
        if not getattr(args, name)
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    os.umask(0o077)
    finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
