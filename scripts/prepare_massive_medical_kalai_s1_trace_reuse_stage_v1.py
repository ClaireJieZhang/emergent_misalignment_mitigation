#!/usr/bin/env python3
"""CPU-stage the exact-paired Kalai s=1 trace-reuse workflow."""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = _load(
    "_kalai_s1_replay_for_cpu_stage",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_trace_reuse_v1.py",
)
authority = _load(
    "_kalai_s1_authority_for_cpu_stage",
    SCRIPT_DIR / "authorize_massive_medical_kalai_s1_trace_reuse_v1.py",
)


PROTOCOL_ID = replay.PROTOCOL_ID
METHOD_ID = replay.METHOD_ID
SEAL_FIELD = replay.OUTPUT_SEAL
EXPECTED_OUTPUT_LEAF = "massive_medical_kalai_s1_r20_trace_reuse_v1"
EXPECTED_COUNTS = {
    "technical_gate": {
        "requested_rows": 18,
        "unresolved_by_phase": {"benefit": 1, "medical": 15},
        "unresolved_rows": 16,
        "unresolved_prefix_attempts_reused": 100,
        "maximum_new_attempts": 220,
    },
    "completion": {
        "requested_rows": 422,
        "unresolved_by_phase": {"benefit": 115, "medical": 59},
        "unresolved_rows": 174,
        "unresolved_prefix_attempts_reused": 533,
        "maximum_new_attempts": 2947,
    },
}


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean(repo_root):
    if subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("CPU-stage repository is not clean")


def write_new(path, payload):
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


def plan_binding(path, payload):
    return {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[SEAL_FIELD],
    }


def implementation_bindings(repo_root):
    names = (
        "scripts/prepare_massive_medical_kalai_s1_trace_reuse_v1.py",
        "scripts/prepare_massive_medical_kalai_s1_trace_reuse_stage_v1.py",
        "scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py",
        "scripts/authorize_massive_medical_kalai_s1_trace_reuse_v1.py",
        "scripts/evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py",
        "scripts/sbatch_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum_h200.sbatch",
        "scripts/submit_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum.sh",
        "scripts/stage_massive_medical_kalai_s1_trace_reuse_v1_tillicum.sh",
        "scripts/sample_massive_medical_whole_output_consensus_v1.py",
        "scripts/sample_massive_medical_whole_output_consensus_s3_v2.py",
        "subliminal_mitigate/decoding/algorithms.py",
        "tests/test_massive_medical_kalai_s1_trace_reuse_v1.py",
        "tests/test_massive_medical_kalai_s1_trace_reuse_controller_v1.py",
        "tests/test_massive_medical_kalai_s1_trace_reuse_workflow_v1.py",
        "configs/pipelines/massive_medical_kalai_s1_r20_trace_reuse_v1.yaml",
        "docs/massive_medical_kalai_s1_r20_trace_reuse_v1_protocol.md",
    )
    result = {}
    for name in names:
        path = Path(repo_root) / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required implementation is absent or unsafe: {name}")
        result[name] = sha256_file(path)
    return result


def prepare(args):
    output_root = Path(args.output_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    if output_root.name != EXPECTED_OUTPUT_LEAF:
        raise ValueError(f"output root must end in {EXPECTED_OUTPUT_LEAF}")
    require_clean(repo_root)
    control = output_root / "control"
    plan_path = control / "REPLAY_PLAN.json"
    stage_path = control / "CPU_STAGE.json"
    if os.path.lexists(output_root) and not plan_path.is_file():
        raise ValueError("partial s=1 trace-reuse namespace exists")
    replay_args = argparse.Namespace(
        source_protocol_manifest=args.source_protocol_manifest,
        s3_benefit_gate_generation=args.s3_benefit_gate_generation,
        s3_benefit_completion_generation=args.s3_benefit_completion_generation,
        s3_medical_gate_generation=args.s3_medical_gate_generation,
        s3_medical_completion_generation=args.s3_medical_completion_generation,
        s1_medical_smoke_generation=args.s1_medical_smoke_generation,
        output=None,
        output_root=os.fspath(output_root),
    )
    plan = replay.prepare(replay_args)
    replay.verify_seal(plan, "Kalai s=1 replay plan")
    if plan.get("partition_summary") != EXPECTED_COUNTS:
        raise ValueError("replay plan partition summary differs")
    if (
        plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("method_id") != METHOD_ID
        or plan.get("continuation", {}).get("requested_rows") != 190
        or plan.get("continuation", {}).get("maximum_missing_attempts") != 3167
    ):
        raise ValueError("replay plan frozen totals differ")
    implementation = implementation_bindings(repo_root)
    if stage_path.is_file():
        existing = load_json(stage_path, "existing CPU stage")
        existing_body = replay.verify_seal(existing, "existing CPU stage")
        created_at = existing_body.get("created_at")
    else:
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    stage = replay.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "status": "CPU_STAGED_NO_GPU_OR_API_AUTHORITY",
            "created_at": created_at,
            "repository_commit": git_commit(repo_root),
            "replay_plan": plan_binding(plan_path, plan),
            "implementation_sha256": implementation,
            "technical_gate_plan_not_authorization": {
                "requested_rows": 18,
                "unresolved_rows": 16,
                "unresolved_by_phase": {"benefit": 1, "medical": 15},
                "prefix_attempts_reused": 100,
                "maximum_new_candidate_attempts": 220,
                "h200_count": 1,
                "h200_minutes": authority.GATE_H200_MINUTES,
                "h200_hourly_usd": float(authority.H200_HOURLY_USD),
                "maximum_cost_usd": float(authority.GATE_CAP_USD),
                "coverage_threshold": None,
                "purpose": "runtime_and_integrity_not_futility",
            },
            "program_ledger_context_not_authorization": {
                "known_program_actual_usd": float(
                    authority.KNOWN_PROGRAM_ACTUAL_USD
                ),
                "retained_conservative_exposure_usd": float(
                    authority.RETAINED_CONSERVATIVE_EXPOSURE_USD
                ),
                "current_conservative_exposure_usd": float(
                    authority.CURRENT_CONSERVATIVE_EXPOSURE_USD
                ),
                "maximum_if_gate_later_authorized_usd": float(
                    authority.MAXIMUM_WITH_GATE_CAP_USD
                ),
                "program_ceiling_usd": float(authority.WORKFLOW_CEILING_USD),
            },
            "completion_cap_usd": None,
            "completion_requires_separate_authorization": True,
            "external_api_calls": 0,
            "external_api_authorized": False,
            "gpu_jobs_submitted": 0,
            "gpu_authorized": False,
            "restart_or_resume_authorized": False,
            "automatic_continuation_authorized": False,
        }
    )
    if stage_path.is_file():
        if load_json(stage_path, "existing CPU stage") != stage:
            raise ValueError("existing CPU stage differs")
        action = "AUDITED"
    else:
        write_new(stage_path, stage)
        action = "CREATED"
    print(
        json.dumps(
            {
                "status": f"MASSIVE_MEDICAL_KALAI_S1_CPU_STAGE_{action}",
                "replay_plan_payload_sha256": plan[SEAL_FIELD],
                "cpu_stage_payload_sha256": stage[SEAL_FIELD],
                "reused_resolved_rows": 250,
                "continuation_rows": 190,
                "maximum_missing_attempts": 3167,
                "technical_gate_maximum_new_attempts": 220,
                "planned_gate_cap_usd_not_authorized": float(
                    authority.GATE_CAP_USD
                ),
                "conservative_maximum_if_authorized_usd": float(
                    authority.MAXIMUM_WITH_GATE_CAP_USD
                ),
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return plan, stage


def self_test():
    assert EXPECTED_COUNTS["technical_gate"]["maximum_new_attempts"] == 220
    assert EXPECTED_COUNTS["completion"]["maximum_new_attempts"] == 2947
    assert Decimal(35) * Decimal("0.90") / Decimal(60) == Decimal("0.525")
    assert authority.CURRENT_CONSERVATIVE_EXPOSURE_USD + Decimal("0.525") == (
        authority.MAXIMUM_WITH_GATE_CAP_USD
    )
    print("MASSIVE_MEDICAL_KALAI_S1_TRACE_REUSE_STAGE_V1_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--s3-benefit-gate-generation")
    parser.add_argument("--s3-benefit-completion-generation")
    parser.add_argument("--s3-medical-gate-generation")
    parser.add_argument("--s3-medical-completion-generation")
    parser.add_argument("--s1-medical-smoke-generation")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if any(
        getattr(args, name) is None
        for name in (
            "source_protocol_manifest",
            "s3_benefit_gate_generation",
            "s3_benefit_completion_generation",
            "s3_medical_gate_generation",
            "s3_medical_completion_generation",
            "s1_medical_smoke_generation",
            "output_root",
            "repo_root",
        )
    ):
        parser.error("all source, output, and repository arguments are required")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
