#!/usr/bin/env python3
"""Recover Kalai s=1 batch 2 in a fresh, derivation-only v2 namespace.

The source generation is the immutable output of Slurm job 271409.  Recovery
v1 stopped before creating an output namespace because its log proof required
a source-expression line that Python did not print in the traceback.  This
version binds both histories, uses the corrected traceback proof, and writes
only a new CPU-derived result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# d9fab8c corrected the traceback proof in this implementation and corrected
# evaluator.  V2 imports those tested primitives without modifying recovery v1.
base = _load(
    "_kalai_s1_batch2_recovery_v2_base",
    "manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_batch2_result_recovery_v2"
SOURCE_PROTOCOL_ID = base.SOURCE_PROTOCOL_ID
METHOD_ID = base.METHOD_ID
SEAL_FIELD = base.SEAL_FIELD
SOURCE_BATCH_INDEX = base.SOURCE_BATCH_INDEX
SOURCE_BATCH_ID = base.SOURCE_BATCH_ID
SOURCE_JOB_ID = base.SOURCE_JOB_ID
SOURCE_REPOSITORY_COMMIT = base.SOURCE_REPOSITORY_COMMIT
SOURCE_REPOSITORY_LEAF = base.SOURCE_REPOSITORY_LEAF
SOURCE_OUTPUT_LEAF = base.SOURCE_OUTPUT_LEAF

FAILED_RECOVERY_PROTOCOL_ID = "massive_medical_kalai_s1_batch2_result_recovery_v1"
FAILED_RECOVERY_REPOSITORY_COMMIT = (
    "6460a9895919d72f97b870f802b9d0813a0c45e2"
)
FAILED_RECOVERY_REPOSITORY_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v1"
)
FAILED_RECOVERY_OUTPUT_LEAF = FAILED_RECOVERY_PROTOCOL_ID
RECOVERY_REPOSITORY_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v2"
)
RECOVERY_OUTPUT_LEAF = PROTOCOL_ID

PLAN_NAME = "RECOVERY_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
RESULT_NAME = "RECOVERED_RESULT.json"

FAILED_RECOVERY_IMPLEMENTATION_SHA256 = {
    "configs/pipelines/massive_medical_kalai_s1_batch2_result_recovery_v1.yaml": (
        "ea9bd6197092ad503459f18bbd9559be6f89047bfcda25daa515fdf3dd2e8d94"
    ),
    "docs/massive_medical_kalai_s1_batch2_result_recovery_v1_protocol.md": (
        "e2cf4bbc7aa22638059d1a9fa39682af7660316f30c2f68ba55fa6d0bb932a41"
    ),
    "scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py": (
        "17d6111058abf26a02da4fb1da3a7122b60b7386e36fee0f5b10560c0ad8b9c4"
    ),
    "scripts/run_massive_medical_kalai_s1_batch2_result_recovery_v1_tillicum.sh": (
        "1e3ed0408d2c217b311b7cd2f7c27bf02f5d4ddc258609a261b667513471b9ff"
    ),
    "tests/test_massive_medical_kalai_s1_batch2_result_recovery_v1.py": (
        "7066169556ec132011f859d083c7be4824edc5f0a0cb4744a6c6791a8195fcea"
    ),
}

REQUIRED_IMPLEMENTATION_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_batch2_result_recovery_v2.yaml",
    "docs/massive_medical_kalai_s1_batch2_result_recovery_v2_protocol.md",
    "scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v2.py",
    "scripts/run_massive_medical_kalai_s1_batch2_result_recovery_v2_tillicum.sh",
    "tests/test_massive_medical_kalai_s1_batch2_result_recovery_v2.py",
    "scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py",
    "scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
    "scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py",
    "scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
)


canonical_bytes = base.canonical_bytes
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
seal = base.seal
load_json = base.load_json
verify_seal = base.verify_seal
binding = base.binding
git_commit = base.git_commit
git_branch = base.git_branch
require_clean = base.require_clean
require_disjoint = base.require_disjoint


def _write_new(path, payload, description):
    return base._write_new(path, payload, description)


def _write_idempotent(path, payload, description):
    return base._write_idempotent(path, payload, description)


def _require_cpu_only():
    return base._require_cpu_only()


def audit_source(source_output_root, source_repo_root):
    """Deep-audit the unchanged job-271409 source using the corrected proof."""
    return base.audit_source(source_output_root, source_repo_root)


def _audit_failed_recovery(failed_repo_root, failed_output_root):
    raw_repo = Path(failed_repo_root)
    raw_output = Path(failed_output_root)
    if raw_repo.is_symlink() or raw_output.is_symlink():
        raise ValueError("failed recovery namespace is a symlink")
    repo = raw_repo.resolve()
    output = raw_output.resolve()
    if repo.name != FAILED_RECOVERY_REPOSITORY_LEAF:
        raise ValueError("failed recovery repository leaf differs")
    if output.name != FAILED_RECOVERY_OUTPUT_LEAF:
        raise ValueError("failed recovery output leaf differs")
    require_clean(repo, "failed recovery v1")
    if git_commit(repo) != FAILED_RECOVERY_REPOSITORY_COMMIT:
        raise ValueError("failed recovery v1 repository commit differs")
    if os.path.lexists(raw_output):
        raise ValueError("failed recovery v1 output must remain absent")

    implementation = {}
    for relative, expected_sha256 in FAILED_RECOVERY_IMPLEMENTATION_SHA256.items():
        observed = binding(
            repo / relative,
            repo,
            description=f"failed recovery v1 implementation {relative}",
        )
        if observed["file_sha256"] != expected_sha256:
            raise ValueError(f"failed recovery v1 implementation differs: {relative}")
        implementation[relative] = observed
    return {
        "protocol_id": FAILED_RECOVERY_PROTOCOL_ID,
        "repository": {
            "path": str(repo),
            "leaf": FAILED_RECOVERY_REPOSITORY_LEAF,
            "commit": FAILED_RECOVERY_REPOSITORY_COMMIT,
            "clean": True,
        },
        "implementation_bindings": implementation,
        "output": {
            "path": str(output),
            "leaf": FAILED_RECOVERY_OUTPUT_LEAF,
            "exists": False,
        },
        "failure_stage": "pre_namespace_log_assertion",
        "reexecuted": False,
        "preserved_as_history": True,
    }


def _audit_recovery_namespace(output_root, allowed):
    raw_output = Path(output_root)
    if raw_output.is_symlink():
        raise ValueError("recovery output is a symlink")
    output = raw_output.resolve()
    if output.name != RECOVERY_OUTPUT_LEAF:
        raise ValueError("recovery output leaf differs")
    for name in ("generation", "assembled", "judge", "logs", "batches"):
        if os.path.lexists(output / name):
            raise ValueError(f"recovery namespace contains forbidden {name}")
    if output.exists() and {item.name for item in output.iterdir()} - {"control"}:
        raise ValueError("recovery namespace contains unexpected state")
    control = output / "control"
    if control.exists():
        if control.is_symlink() or not control.is_dir():
            raise ValueError("recovery control is unsafe")
        items = list(control.iterdir())
        if {item.name for item in items} - set(allowed):
            raise ValueError("recovery control inventory differs")
        if any(item.is_symlink() or not item.is_file() for item in items):
            raise ValueError("recovery control contains an unsafe entry")


def _policy():
    return {
        "derivation_only": True,
        "source_artifacts_read_only": True,
        "source_stopped_preserved": True,
        "failed_recovery_v1_preserved": True,
        "failed_recovery_v1_rerun_authorized": False,
        "generation_authorized": False,
        "gpu_jobs_authorized": 0,
        "external_api_calls_authorized": 0,
        "restart_resume_retry_replacement_requeue_authorized": False,
        "batch_3_submission_authorized": False,
        "judging_authorized": False,
    }


def _accounting():
    return base._accounting()


def _plan_body(source, failed, source_output, source_repo, output):
    return {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch2_result_recovery_plan",
        "status": "CPU_ONLY_BATCH2_RECOVERY_V2_PLANNED_NO_GPU_OR_API_AUTHORITY",
        "source_repository": {
            "path": str(source_repo),
            "commit": SOURCE_REPOSITORY_COMMIT,
            "clean": True,
        },
        "source_output_root": str(source_output),
        "source_job": {
            "batch_index": SOURCE_BATCH_INDEX,
            "batch_id": SOURCE_BATCH_ID,
            "slurm_job_id": SOURCE_JOB_ID,
            "scheduler_state": "FAILED",
            "scheduler_exit_code": "1:0",
            "scheduler_derived_exit_code": "0:0",
            "scheduler_elapsed_seconds": base.SOURCE_ELAPSED_SECONDS,
            "actual_estimated_cost_usd": float(base.SOURCE_ACTUAL_COST_USD),
            "authorized_cap_usd_retained": float(base.SOURCE_AUTHORIZED_CAP_USD),
        },
        "source_job_evidence": source["job_evidence"],
        "source_key_bindings": source["key_bindings"],
        "source_control_manifest": source["control_manifest"],
        "source_generation_manifest": source["generation_manifest"],
        "corrected_generation_audit": source["generation_audit"],
        "failed_recovery_v1": failed,
        "correction": {
            "traceback_proof": [
                "ValueError: completion-batch shard binding differs",
                "in _generation_audit",
                "runtime._call_with_continuation_protocol",
            ],
            "omitted_source_expression_not_required": (
                "manager.original_runtime._audit_batch"
            ),
            "same_runtime_manager_module_instance": True,
        },
        "recovery_output_root": str(output),
        "recovery_policy": _policy(),
        "accounting": _accounting(),
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }


def prepare(args):
    _require_cpu_only()
    source_output = Path(args.source_output_root).resolve()
    source_repo = Path(args.source_repo_root).resolve()
    failed_repo = Path(args.failed_recovery_repo_root).resolve()
    failed_output = Path(args.failed_recovery_output_root).resolve()
    output = Path(args.output_root).resolve()
    require_disjoint(source_output, source_repo, failed_repo, failed_output, output)
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME})
    source = audit_source(source_output, source_repo)
    failed = _audit_failed_recovery(failed_repo, failed_output)
    plan = seal(_plan_body(source, failed, source_output, source_repo, output))
    disposition = _write_idempotent(
        output / "control" / PLAN_NAME, plan, "batch-2 recovery-v2 plan"
    )
    print(json.dumps({
        "status": f"KALAI_S1_BATCH2_RESULT_RECOVERY_V2_PLAN_{disposition}",
        "recovery_plan_payload_sha256": plan[SEAL_FIELD],
        "source_generation_files": base.EXPECTED_GENERATION_MANIFEST["file_count"],
        "failed_recovery_v1_output_absent": True,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))
    return plan


def load_plan(path, *, audit_source_state):
    path = Path(path).resolve()
    payload = load_json(path, "batch-2 recovery-v2 plan")
    body = verify_seal(payload, "batch-2 recovery-v2 plan")
    source_repo = Path(body.get("source_repository", {}).get("path", "")).resolve()
    source_output = Path(body.get("source_output_root", "")).resolve()
    failed_body = body.get("failed_recovery_v1", {})
    failed_repo = Path(failed_body.get("repository", {}).get("path", "")).resolve()
    failed_output = Path(failed_body.get("output", {}).get("path", "")).resolve()
    output = Path(body.get("recovery_output_root", "")).resolve()
    if (
        body.get("schema_version") != 2
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "batch2_result_recovery_plan"
        or body.get("status")
        != "CPU_ONLY_BATCH2_RECOVERY_V2_PLANNED_NO_GPU_OR_API_AUTHORITY"
        or body.get("recovery_policy") != _policy()
        or body.get("accounting") != _accounting()
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
        or source_repo.name != SOURCE_REPOSITORY_LEAF
        or source_output.name != SOURCE_OUTPUT_LEAF
        or failed_repo.name != FAILED_RECOVERY_REPOSITORY_LEAF
        or failed_output.name != FAILED_RECOVERY_OUTPUT_LEAF
        or output.name != RECOVERY_OUTPUT_LEAF
        or path != output / "control" / PLAN_NAME
    ):
        raise ValueError("batch-2 recovery-v2 plan identity differs")
    require_disjoint(source_output, source_repo, failed_repo, failed_output, output)
    if body.get("source_job_evidence") != base.expected_job_evidence(source_output):
        raise ValueError("batch-2 recovery-v2 job evidence differs")
    if audit_source_state:
        source = audit_source(source_output, source_repo)
        failed = _audit_failed_recovery(failed_repo, failed_output)
        expected = _plan_body(source, failed, source_output, source_repo, output)
        if body != expected:
            raise ValueError("batch-2 recovery-v2 source state differs from plan")
    return payload, body


def implementation_bindings(repo):
    repo = Path(repo).resolve()
    result = {}
    for relative in REQUIRED_IMPLEMENTATION_FILES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery-v2 implementation is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def _stage_body(repo, plan_path, plan):
    branch = git_branch(repo)
    if not branch:
        raise ValueError("recovery-v2 repository must be on a named branch")
    return {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch2_result_recovery_cpu_stage",
        "status": "CPU_STAGED_DERIVATION_ONLY_NO_GPU_OR_API_AUTHORITY",
        "repository": {
            "path": str(repo),
            "commit": git_commit(repo),
            "branch": branch,
            "clean": True,
        },
        "recovery_plan": binding(
            plan_path, sealed=True, description="batch-2 recovery-v2 plan"
        ),
        "implementation_sha256": implementation_bindings(repo),
        "execution_policy": _policy(),
        "new_cost_cap_usd": 0.0,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }


def stage(args):
    _require_cpu_only()
    output = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    source_output = Path(args.source_output_root).resolve()
    source_repo = Path(args.source_repo_root).resolve()
    failed_repo = Path(args.failed_recovery_repo_root).resolve()
    failed_output = Path(args.failed_recovery_output_root).resolve()
    if repo.name != RECOVERY_REPOSITORY_LEAF:
        raise ValueError("recovery-v2 repository leaf differs")
    require_disjoint(
        source_output, source_repo, failed_repo, failed_output, output, repo
    )
    require_clean(repo, "recovery v2")
    plan = prepare(args)
    plan_path = output / "control" / PLAN_NAME
    stage_payload = seal(_stage_body(repo, plan_path, plan))
    disposition = _write_idempotent(
        output / "control" / STAGE_NAME,
        stage_payload,
        "batch-2 recovery-v2 CPU stage",
    )
    load_stage(output / "control" / STAGE_NAME, repo, audit_source_state=True)
    print(json.dumps({
        "status": f"KALAI_S1_BATCH2_RESULT_RECOVERY_V2_CPU_STAGE_{disposition}",
        "cpu_stage_payload_sha256": stage_payload[SEAL_FIELD],
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))
    return stage_payload


def load_stage(path, repo_root, *, audit_source_state):
    path = Path(path).resolve()
    repo = Path(repo_root).resolve()
    payload = load_json(path, "batch-2 recovery-v2 CPU stage")
    body = verify_seal(payload, "batch-2 recovery-v2 CPU stage")
    plan_path = path.parent / PLAN_NAME
    _, plan_body = load_plan(plan_path, audit_source_state=audit_source_state)
    expected = _stage_body(repo, plan_path, load_json(plan_path, "recovery-v2 plan"))
    if body != expected or path != plan_path.parent / STAGE_NAME:
        raise ValueError("batch-2 recovery-v2 CPU stage differs")
    if body["repository"]["path"] != str(repo):
        raise ValueError("batch-2 recovery-v2 repository path differs")
    require_clean(repo, "recovery v2")
    require_disjoint(
        repo,
        plan_body["source_repository"]["path"],
        plan_body["source_output_root"],
        plan_body["failed_recovery_v1"]["repository"]["path"],
        plan_body["failed_recovery_v1"]["output"]["path"],
        plan_body["recovery_output_root"],
    )
    return payload, body


def _expected_result(output, repo, failed_recovery_repo=None, failed_recovery_output=None):
    output = Path(output).resolve()
    plan_path = output / "control" / PLAN_NAME
    stage_path = output / "control" / STAGE_NAME
    _, plan_body = load_plan(plan_path, audit_source_state=True)
    stage_payload, _ = load_stage(stage_path, repo, audit_source_state=True)
    failed_body = plan_body["failed_recovery_v1"]
    failed_repo = Path(
        failed_recovery_repo or failed_body["repository"]["path"]
    ).resolve()
    failed_output = Path(
        failed_recovery_output or failed_body["output"]["path"]
    ).resolve()
    failed = _audit_failed_recovery(failed_repo, failed_output)
    if failed != failed_body:
        raise ValueError("failed recovery v1 state differs from plan")
    source_output = Path(plan_body["source_output_root"])
    source_repo = Path(plan_body["source_repository"]["path"])
    source = audit_source(source_output, source_repo)
    audit = source["generation_audit"]
    corrected_source_result_body = base.source_evaluator._expected_result_body(
        source_output,
        source_repo,
        SOURCE_BATCH_INDEX,
        slurm_job_id=SOURCE_JOB_ID,
        elapsed_seconds=base.SOURCE_ELAPSED_SECONDS,
        audit=audit,
    )
    return seal({
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "batch2_result_recovery",
        "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERED_CPU_ONLY",
        "batch_index": SOURCE_BATCH_INDEX,
        "batch_id": SOURCE_BATCH_ID,
        "batch_valid": True,
        "source_job": plan_body["source_job"],
        "source_job_evidence": plan_body["source_job_evidence"],
        "recovery_reason": {
            "category": "recovery_v1_log_assertion_false_negative",
            "generation_error": False,
            "scientific_outputs_regenerated": False,
            "corrected_traceback_proof": plan_body["correction"]["traceback_proof"],
            "omitted_source_expression_not_required": True,
            "same_runtime_manager_module_instance": True,
        },
        "failed_recovery_v1": failed,
        "recovery_plan": binding(
            plan_path, sealed=True, description="recovery-v2 plan"
        ),
        "cpu_stage": binding(
            stage_path, sealed=True, description="recovery-v2 CPU stage"
        ),
        "source_key_bindings": source["key_bindings"],
        "source_control_manifest": source["control_manifest"],
        "source_generation_manifest": source["generation_manifest"],
        "corrected_generation_audit": audit,
        "corrected_source_result_body": corrected_source_result_body,
        "source_stopped_preserved": True,
        "source_result_absent": True,
        "failed_recovery_v1_output_absent": True,
        "failed_recovery_v1_rerun": False,
        "derivation_only": True,
        "recovered_predecessor_evidence_valid": True,
        "automatic_next_batch_authorized": False,
        "batch_3_submission_authorized": False,
        "restart_or_resume_authorized": False,
        "retry_replacement_or_requeue_authorized": False,
        "generation_authorized": False,
        "judging_authorized": False,
        "accounting": _accounting(),
        "external_api_calls": 0,
        "gpu_jobs": 0,
    })


def recover(args):
    _require_cpu_only()
    output = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME})
    result_path = output / "control" / RESULT_NAME
    if os.path.lexists(result_path):
        raise ValueError("recovered-v2 result already exists; rerun is forbidden")
    _, plan_body = load_plan(output / "control" / PLAN_NAME, audit_source_state=True)
    before = audit_source(
        plan_body["source_output_root"], plan_body["source_repository"]["path"]
    )
    before_failed = _audit_failed_recovery(
        args.failed_recovery_repo_root, args.failed_recovery_output_root
    )
    result = _expected_result(
        output,
        repo,
        args.failed_recovery_repo_root,
        args.failed_recovery_output_root,
    )
    _write_new(result_path, result, "recovered batch-2 v2 result")
    after = audit_source(
        plan_body["source_output_root"], plan_body["source_repository"]["path"]
    )
    after_failed = _audit_failed_recovery(
        args.failed_recovery_repo_root, args.failed_recovery_output_root
    )
    for key in ("key_bindings", "control_manifest", "generation_manifest", "job_evidence"):
        if after[key] != before[key]:
            raise ValueError("source state changed during recovery-v2")
    if after_failed != before_failed:
        raise ValueError("failed recovery v1 state changed during recovery-v2")
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME, RESULT_NAME})
    print(json.dumps({
        "status": result["status"],
        "recovered_result_payload_sha256": result[SEAL_FIELD],
        "source_job_id": SOURCE_JOB_ID,
        "source_stopped_preserved": True,
        "failed_recovery_v1_output_absent": True,
        "batch_3_submission_authorized": False,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))
    return result


def audit_recovered_result(
    recovery_output_root,
    recovery_repo_root,
    failed_recovery_repo_root,
    failed_recovery_output_root,
):
    """Return the exact recovered result after a full immutable-state audit."""
    _require_cpu_only()
    output = Path(recovery_output_root).resolve()
    repo = Path(recovery_repo_root).resolve()
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME, RESULT_NAME})
    observed = load_json(output / "control" / RESULT_NAME, "recovered batch-2 v2 result")
    verify_seal(observed, "recovered batch-2 v2 result")
    expected = _expected_result(
        output,
        repo,
        failed_recovery_repo_root,
        failed_recovery_output_root,
    )
    if observed != expected:
        raise ValueError("recovered batch-2 v2 result differs")
    return observed


def audit_recovery(args):
    observed = audit_recovered_result(
        args.output_root,
        args.repo_root,
        args.failed_recovery_repo_root,
        args.failed_recovery_output_root,
    )
    print(json.dumps({
        "status": "MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERY_V2_AUDITED",
        "recovered_result_payload_sha256": observed[SEAL_FIELD],
        "source_stopped_preserved": True,
        "failed_recovery_v1_output_absent": True,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))
    return observed


def self_test():
    base.self_test()
    assert PROTOCOL_ID.endswith("batch2_result_recovery_v2")
    assert FAILED_RECOVERY_REPOSITORY_COMMIT == (
        "6460a9895919d72f97b870f802b9d0813a0c45e2"
    )
    assert base.SOURCE_ACTUAL_COST_USD == (
        base.Decimal(base.SOURCE_ELAPSED_SECONDS)
        * base.H200_HOURLY_USD
        / base.Decimal(3600)
    )
    corrected = Path(base.__file__).read_text(encoding="utf-8")
    assert '"in _generation_audit"' in corrected
    assert '"manager.original_runtime._audit_batch"' not in corrected
    print("MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERY_V2_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-root")
    parser.add_argument("--source-repo-root")
    parser.add_argument("--failed-recovery-repo-root")
    parser.add_argument("--failed-recovery-output-root")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--stage", action="store_true")
    modes.add_argument("--recover", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not all((
        args.failed_recovery_repo_root,
        args.failed_recovery_output_root,
        args.output_root,
    )):
        parser.error("failed-recovery and output paths are required")
    if args.recover or args.audit_only:
        if not args.repo_root:
            parser.error("--repo-root is required")
        (recover if args.recover else audit_recovery)(args)
        return 0
    required = (
        args.source_output_root,
        args.source_repo_root,
        args.repo_root if args.stage else True,
    )
    if not all(required):
        parser.error("source and recovery paths are required")
    (stage if args.stage else prepare)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
