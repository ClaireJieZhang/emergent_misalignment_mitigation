#!/usr/bin/env python3
"""CPU-stage the versioned Kalai s=1 batch-1 result recovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_module(
    "_kalai_s1_batch1_result_recovery_planner_for_stage",
    SCRIPT_DIR / "prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)


PROTOCOL_ID = planner.PROTOCOL_ID
SEAL_FIELD = planner.SEAL_FIELD
EXPECTED_BRANCH = "claire/massive-medical-kalai-s1-batch1-result-recovery-v1"
REQUIRED_IMPLEMENTATION_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_batch1_result_recovery_v1.yaml",
    "docs/massive_medical_kalai_s1_batch1_result_recovery_v1_protocol.md",
    "scripts/prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
    "scripts/stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
    "scripts/evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
    "scripts/run_massive_medical_kalai_s1_batch1_result_recovery_v1_tillicum.sh",
    "scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py",
    "tests/test_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
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


def git_branch(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "branch", "--show-current"],
        text=True,
    ).strip()


def require_clean(repo_root):
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise ValueError("recovery repository is not clean")


def implementation_bindings(repo_root):
    repo_root = Path(repo_root).resolve()
    bindings = {}
    for relative in REQUIRED_IMPLEMENTATION_FILES:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery implementation is absent: {relative}")
        bindings[relative] = sha256_file(path)
    return bindings


def _write_idempotent(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.path.lexists(path):
        observed = planner.load_json(path, "existing recovery CPU stage")
        planner.verify_seal(observed, "existing recovery CPU stage")
        if observed != payload:
            raise ValueError("existing recovery CPU stage differs")
        return "AUDITED"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "CREATED"


def stage(args):
    source_output = Path(args.source_output_root).resolve()
    source_repo = Path(args.source_repo_root).resolve()
    output = Path(args.output_root).resolve()
    repo = Path(args.repo_root).resolve()
    planner.require_recovery_output_disjoint(
        output, source_output, source_repo, repo
    )
    planner.require_recovery_output_disjoint(
        repo, source_output, source_repo, output
    )
    require_clean(repo)
    if git_branch(repo) != EXPECTED_BRANCH:
        raise ValueError("recovery branch differs")
    planner._audit_recovery_namespace(
        output, {planner.PLAN_NAME, planner.STAGE_NAME}
    )
    plan = planner.prepare(
        argparse.Namespace(
            source_output_root=str(source_output),
            source_repo_root=str(source_repo),
            output_root=str(output),
        )
    )
    plan_path = output / "control" / planner.PLAN_NAME
    planner.load_and_verify_plan(plan_path, audit_source=True)
    stage_payload = planner.seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": planner.SOURCE_PROTOCOL_ID,
            "method_id": planner.METHOD_ID,
            "stage": "batch1_result_recovery_cpu_stage",
            "status": "CPU_STAGED_DERIVATION_ONLY_NO_GPU_OR_API_AUTHORITY",
            "repository": {
                "path": str(repo),
                "commit": git_commit(repo),
                "branch": EXPECTED_BRANCH,
                "clean": True,
            },
            "source_repository": {
                "path": str(source_repo),
                "commit": planner.EXPECTED_SOURCE_REPOSITORY_COMMIT,
                "clean": True,
            },
            "recovery_plan": {
                "path": str(plan_path),
                "size_bytes": plan_path.stat().st_size,
                "file_sha256": sha256_file(plan_path),
                "payload_sha256": plan[SEAL_FIELD],
            },
            "implementation_sha256": implementation_bindings(repo),
            "execution_policy": {
                "cpu_derivation_only": True,
                "source_artifacts_read_only": True,
                "generation_authorized": False,
                "gpu_jobs_authorized": 0,
                "external_api_calls_authorized": 0,
                "restart_resume_retry_replacement_authorized": False,
                "batch_2_submission_authorized": False,
            },
            "new_cost_cap_usd": 0.0,
            "external_api_calls": 0,
            "gpu_jobs": 0,
        }
    )
    stage_path = output / "control" / planner.STAGE_NAME
    disposition = _write_idempotent(stage_path, stage_payload)
    load_and_verify_stage(stage_path, repo, audit_source=True)
    print(
        json.dumps(
            {
                "status": f"BATCH1_RESULT_RECOVERY_CPU_STAGE_{disposition}",
                "recovery_plan_payload_sha256": plan[SEAL_FIELD],
                "cpu_stage_payload_sha256": stage_payload[SEAL_FIELD],
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return stage_payload


def load_and_verify_stage(path, repo_root, *, audit_source=True):
    path = Path(path).resolve()
    repo = Path(repo_root).resolve()
    payload = planner.load_json(path, "batch-1 recovery CPU stage")
    body = planner.verify_seal(payload, "batch-1 recovery CPU stage")
    policy = body.get("execution_policy")
    repository = body.get("repository")
    source_repository = body.get("source_repository")
    expected_fields = {
        "schema_version",
        "protocol_id",
        "source_protocol_id",
        "method_id",
        "stage",
        "status",
        "repository",
        "source_repository",
        "recovery_plan",
        "implementation_sha256",
        "execution_policy",
        "new_cost_cap_usd",
        "external_api_calls",
        "gpu_jobs",
    }
    expected_policy = {
        "cpu_derivation_only": True,
        "source_artifacts_read_only": True,
        "generation_authorized": False,
        "gpu_jobs_authorized": 0,
        "external_api_calls_authorized": 0,
        "restart_resume_retry_replacement_authorized": False,
        "batch_2_submission_authorized": False,
    }
    if (
        set(body) != expected_fields
        or body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != planner.SOURCE_PROTOCOL_ID
        or body.get("method_id") != planner.METHOD_ID
        or body.get("stage") != "batch1_result_recovery_cpu_stage"
        or body.get("status")
        != "CPU_STAGED_DERIVATION_ONLY_NO_GPU_OR_API_AUTHORITY"
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
        or body.get("new_cost_cap_usd") != 0.0
        or policy != expected_policy
        or not isinstance(repository, dict)
        or set(repository) != {"path", "commit", "branch", "clean"}
        or repository.get("path") != str(repo)
        or repository.get("branch") != EXPECTED_BRANCH
        or repository.get("clean") is not True
        or not isinstance(source_repository, dict)
        or set(source_repository) != {"path", "commit", "clean"}
        or source_repository.get("commit")
        != planner.EXPECTED_SOURCE_REPOSITORY_COMMIT
        or source_repository.get("clean") is not True
    ):
        raise ValueError("batch-1 recovery CPU stage identity differs")
    require_clean(repo)
    if (
        git_commit(repo) != repository.get("commit")
        or git_branch(repo) != EXPECTED_BRANCH
        or body.get("implementation_sha256") != implementation_bindings(repo)
    ):
        raise ValueError("batch-1 recovery repository binding differs")
    plan_binding = body.get("recovery_plan")
    if not isinstance(plan_binding, dict):
        raise ValueError("batch-1 recovery plan binding is absent")
    plan_path = Path(plan_binding.get("path", ""))
    plan, plan_body = planner.load_and_verify_plan(
        plan_path, audit_source=audit_source
    )
    source_repo_path = Path(plan_body["source_repository"]["path"]).resolve()
    output = Path(plan_body["recovery_output_root"]).resolve()
    planner.require_recovery_output_disjoint(
        output, plan_body["source_output_root"], source_repo_path, repo
    )
    planner.require_recovery_output_disjoint(
        repo, plan_body["source_output_root"], source_repo_path, output
    )
    expected_binding = {
        "path": str(plan_path),
        "size_bytes": plan_path.stat().st_size,
        "file_sha256": sha256_file(plan_path),
        "payload_sha256": plan[SEAL_FIELD],
    }
    if (
        plan_binding != expected_binding
        or source_repository.get("path") != str(source_repo_path)
    ):
        raise ValueError("batch-1 recovery plan binding differs")
    if path != plan_path.parent / planner.STAGE_NAME:
        raise ValueError("batch-1 recovery CPU stage path differs")
    return payload, body


def self_test():
    assert REQUIRED_IMPLEMENTATION_FILES[-1].startswith("tests/")
    assert EXPECTED_BRANCH.endswith("batch1-result-recovery-v1")
    print("MASSIVE_MEDICAL_KALAI_S1_BATCH1_RESULT_RECOVERY_STAGE_SELF_TEST_OK")


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
        parser.error("all recovery CPU-stage paths are required")
    stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
