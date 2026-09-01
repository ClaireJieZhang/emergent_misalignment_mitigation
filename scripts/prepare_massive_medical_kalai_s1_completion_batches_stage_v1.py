#!/usr/bin/env python3
"""CPU-stage the immutable seven-batch Kalai ``s=1`` completion workflow."""

from __future__ import annotations

import argparse
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


planner = _load(
    "_kalai_s1_completion_batch_planner_for_cpu_stage",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_completion_batches_v1.py",
)


PROTOCOL_ID = planner.PROTOCOL_ID
EXPECTED_SOURCE_LEAF = "massive_medical_kalai_s1_r20_trace_reuse_v1"
EXPECTED_SOURCE_REPO_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-r20-trace-reuse-v1"
)
EXPECTED_OUTPUT_LEAF = "massive_medical_kalai_s1_completion_batches_v1"
PLAN_NAME = "COMPLETION_BATCH_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"

# Integration may extend this tuple when the versioned runtime/authorization
# wrappers are added.  Every listed artifact is included in the immutable CPU
# stage hash inventory.
REQUIRED_IMPLEMENTATION_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_completion_batches_v1.yaml",
    "docs/massive_medical_kalai_s1_completion_batches_v1_protocol.md",
    "scripts/assemble_massive_medical_kalai_s1_completion_batches_v1.py",
    "scripts/authorize_massive_medical_kalai_s1_completion_batch_v1.py",
    "scripts/evaluate_massive_medical_kalai_s1_completion_batch_v1.py",
    "scripts/prepare_massive_medical_kalai_s1_completion_batches_v1.py",
    "scripts/prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py",
    "scripts/prepare_massive_medical_kalai_s1_trace_reuse_v1.py",
    "scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py",
    "scripts/sample_massive_medical_kalai_s1_trace_reuse_v1.py",
    "scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "scripts/sample_massive_medical_whole_output_consensus_s3_v2.py",
    "scripts/sample_massive_medical_whole_output_consensus_v1.py",
    "scripts/sbatch_massive_medical_kalai_s1_completion_batch_v1_tillicum_h200.sbatch",
    "scripts/stage_massive_medical_kalai_s1_completion_batches_v1_tillicum.sh",
    "scripts/submit_massive_medical_kalai_s1_completion_batch_v1_tillicum.sh",
    "scripts/evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py",
    "subliminal_mitigate/decoding/algorithms.py",
    "tests/test_massive_medical_kalai_s1_completion_assembly_v1.py",
    "tests/test_massive_medical_kalai_s1_completion_batches_v1.py",
    "tests/test_massive_medical_kalai_s1_completion_batch_runtime_v1.py",
)


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
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError("CPU-stage repository is not clean")


def _load_json(path, description):
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def _write_new(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _implementation_bindings(repo_root):
    result = {}
    for name in REQUIRED_IMPLEMENTATION_FILES:
        path = Path(repo_root) / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required implementation is absent or unsafe: {name}")
        result[name] = sha256_file(path)
    return result


def _source_paths(source_root):
    source_root = Path(source_root).resolve()
    return {
        "source_replay_plan": source_root / "control" / "REPLAY_PLAN.json",
        "source_cpu_stage": source_root / "control" / "CPU_STAGE.json",
        "technical_gate_result": (
            source_root / "control" / "TECHNICAL_GATE_RESULT.json"
        ),
        "gate_combined_timing": (
            source_root
            / "generation"
            / "technical_gate"
            / "combined_timing.json"
        ),
        "gate_benefit_generation": (
            source_root
            / "generation"
            / "technical_gate"
            / "benefit"
            / "generation.json"
        ),
        "gate_benefit_timing": (
            source_root
            / "generation"
            / "technical_gate"
            / "benefit"
            / "timing.json"
        ),
        "gate_medical_generation": (
            source_root
            / "generation"
            / "technical_gate"
            / "medical"
            / "generation.json"
        ),
        "gate_medical_timing": (
            source_root
            / "generation"
            / "technical_gate"
            / "medical"
            / "timing.json"
        ),
    }


def _audit_source_state(source_root):
    source_root = Path(source_root).resolve()
    paths = _source_paths(source_root)
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sealed source is absent or unsafe: {name}")
    forbidden = (
        source_root / "control" / "COMPLETION_AUTHORIZATION.json",
        source_root / "control" / "COMPLETION_SUBMISSION_LOCK",
        source_root / "generation" / "completion",
        source_root / "assembled" / "completion",
    )
    for path in forbidden:
        if os.path.lexists(path):
            raise ValueError(f"source completion state is unexpectedly present: {path}")
    return paths


def _audit_new_namespace(output_root, allowed_control_names):
    output_root = Path(output_root).resolve()
    for name in ("generation", "assembled", "logs", "judge"):
        if os.path.lexists(output_root / name):
            raise ValueError(f"CPU-only namespace contains {name}")
    control = output_root / "control"
    if control.is_symlink():
        raise ValueError("CPU-stage control directory is a symlink")
    if control.exists():
        observed = {path.name for path in control.iterdir()}
        if observed - set(allowed_control_names):
            raise ValueError("CPU-only namespace contains unexpected control state")
        if any(
            marker in name
            for name in observed
            for marker in ("AUTHORIZATION", "SUBMISSION", "RELEASE", "INVOCATION")
        ):
            raise ValueError("CPU-only namespace contains authority state")


def _plan_binding(path, payload):
    return {
        "path": str(Path(path).resolve()),
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[planner.SEAL_FIELD],
    }


def prepare(args):
    source_root = Path(args.source_output_root).resolve()
    source_repo_root = Path(args.source_repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    if source_root.name != EXPECTED_SOURCE_LEAF:
        raise ValueError(f"source root must end in {EXPECTED_SOURCE_LEAF}")
    if source_repo_root.name != EXPECTED_SOURCE_REPO_LEAF:
        raise ValueError(f"source repo must end in {EXPECTED_SOURCE_REPO_LEAF}")
    if output_root.name != EXPECTED_OUTPUT_LEAF:
        raise ValueError(f"output root must end in {EXPECTED_OUTPUT_LEAF}")
    if source_root == output_root:
        raise ValueError("source and output namespaces must differ")
    require_clean(repo_root)
    require_clean(source_repo_root)
    if git_commit(source_repo_root) != planner.EXPECTED_SOURCE_REPOSITORY_COMMIT:
        raise ValueError("source-gate repository commit differs")
    source_paths = _audit_source_state(source_root)
    plan_path = output_root / "control" / PLAN_NAME
    stage_path = output_root / "control" / STAGE_NAME
    if os.path.lexists(output_root) and not plan_path.is_file():
        raise ValueError("partial completion-batch namespace exists")
    _audit_new_namespace(output_root, {PLAN_NAME, STAGE_NAME})

    plan_args = argparse.Namespace(
        **{name: str(path) for name, path in source_paths.items()},
        output=None,
        output_root=str(output_root),
    )
    plan = planner.prepare(plan_args)
    planner.load_and_verify_plan(plan_path, audit_sources=True)
    implementation = _implementation_bindings(repo_root)
    stage = planner.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": planner.CONTROLLER_PROTOCOL_ID,
            "method_id": planner.METHOD_ID,
            "status": "CPU_STAGED_NO_GPU_OR_API_AUTHORITY",
            "repository_commit": git_commit(repo_root),
            "source_repository": {
                "path": str(source_repo_root),
                "repository_commit": planner.EXPECTED_SOURCE_REPOSITORY_COMMIT,
                "clean": True,
                "source_cpu_stage": plan["source_bindings"][
                    "source_cpu_stage"
                ],
            },
            "completion_batch_plan": _plan_binding(plan_path, plan),
            "implementation_sha256": implementation,
            "batch_design_not_authorization": {
                "batch_count": planner.BATCH_COUNT,
                "batch_ids": list(planner.BATCH_IDS),
                "completion_rows": planner.EXPECTED_ROWS,
                "rows_by_phase": planner.EXPECTED_ROWS_BY_PHASE,
                "maximum_new_attempts": planner.EXPECTED_MAX_NEW_ATTEMPTS,
                "h200_count_per_batch": 1,
                "minutes_per_batch": planner.PLANNED_BATCH_MINUTES,
                "cap_usd_per_batch": float(planner.PLANNED_BATCH_CAP_USD),
                "full_seven_batch_cap_usd": float(
                    planner.PLANNED_SEVEN_BATCH_CAP_USD
                ),
            },
            "accounting_context_not_authorization": {
                "scheduler_gate_elapsed_seconds": (
                    planner.SCHEDULER_GATE_ELAPSED_SECONDS
                ),
                "scheduler_gate_cost_usd": float(
                    planner.SCHEDULER_GATE_COST_USD
                ),
                "current_conservative_exposure_usd": float(
                    planner.CURRENT_CONSERVATIVE_EXPOSURE_USD
                ),
                "maximum_if_all_seven_batches_later_authorized_usd": float(
                    planner.PLANNED_MAXIMUM_USD
                ),
                "proposed_program_ceiling_usd": float(
                    planner.PROPOSED_PROGRAM_CEILING_USD
                ),
            },
            "gpu_jobs_authorized": 0,
            "gpu_jobs_submitted": 0,
            "gpu_authorized": False,
            "external_api_calls_authorized": 0,
            "external_api_calls": 0,
            "external_api_authorized": False,
            "batch_authorizations_created": 0,
            "restart_or_resume_authorized": False,
            "retry_or_replacement_authorized": False,
            "requeue_authorized": False,
            "automatic_next_batch_authorized": False,
        }
    )
    if stage_path.is_file():
        existing = _load_json(stage_path, "existing completion-batch CPU stage")
        planner.verify_seal(existing, "existing completion-batch CPU stage")
        if existing != stage:
            raise ValueError("existing completion-batch CPU stage differs")
        action = "AUDITED"
    else:
        _write_new(stage_path, stage)
        action = "CREATED"
    _audit_new_namespace(output_root, {PLAN_NAME, STAGE_NAME})
    print(
        json.dumps(
            {
                "status": f"KALAI_S1_COMPLETION_BATCH_CPU_STAGE_{action}",
                "protocol_id": PROTOCOL_ID,
                "plan_payload_sha256": plan[planner.SEAL_FIELD],
                "cpu_stage_payload_sha256": stage[planner.SEAL_FIELD],
                "batch_count": planner.BATCH_COUNT,
                "completion_rows": planner.EXPECTED_ROWS,
                "maximum_new_attempts": planner.EXPECTED_MAX_NEW_ATTEMPTS,
                "gpu_jobs_submitted": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return plan, stage


def self_test():
    assert EXPECTED_OUTPUT_LEAF == planner.PROTOCOL_ID
    assert len(planner.BATCH_IDS) == 7
    assert planner.PLANNED_SEVEN_BATCH_CAP_USD == planner.Decimal("6.300")
    print("KALAI_S1_COMPLETION_BATCH_CPU_STAGE_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-root")
    parser.add_argument("--source-repo-root")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if any(
        value is None
        for value in (
            args.source_output_root,
            args.source_repo_root,
            args.output_root,
            args.repo_root,
        )
    ):
        parser.error("source output, fresh output, and repository roots are required")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
