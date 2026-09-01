#!/usr/bin/env python3
"""CPU-stage a fresh Kalai s=1 continuation after recovered batch 2.

This protocol treats both failed GPU namespaces and their CPU-only recovered
results as immutable predecessors.  CPU staging grants no authority.  A later,
separate authorization may run exactly one original batch in indices 3--7.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_manager = _load(
    "_kalai_s1_rc_v1_for_rc_v2",
    "manage_massive_medical_kalai_s1_recovery_continuation_v1.py",
)
recovery_manager = _load(
    "_kalai_s1_batch2_recovery_manager_for_rc_v2",
    "manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py",
)

PROTOCOL_ID = "massive_medical_kalai_s1_recovery_continuation_v2"
SOURCE_PROTOCOL_ID = source_manager.PROTOCOL_ID
SCIENTIFIC_BATCH_PROTOCOL_ID = source_manager.SOURCE_PROTOCOL_ID
RECOVERY_PROTOCOL_ID = recovery_manager.PROTOCOL_ID
METHOD_ID = source_manager.METHOD_ID
SEAL_FIELD = source_manager.SEAL_FIELD
PHASES = source_manager.PHASES
BATCH_INDICES = tuple(range(3, 8))

EXPECTED_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v2"
EXPECTED_OUTPUT_LEAF = PROTOCOL_ID
EXPECTED_SOURCE_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v1"
EXPECTED_SOURCE_OUTPUT_LEAF = source_manager.PROTOCOL_ID
EXPECTED_SOURCE_REPOSITORY_COMMIT = "fb4056fcdd77f25bbadc797060dc12edcd340f52"
EXPECTED_RECOVERY_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v1"
EXPECTED_RECOVERY_OUTPUT_LEAF = "massive_medical_kalai_s1_batch2_result_recovery_v1"
EXPECTED_RECOVERY_STATUS = "MASSIVE_MEDICAL_KALAI_S1_BATCH2_RESULT_RECOVERED_CPU_ONLY"

PLAN_NAME = "RECOVERY_CONTINUATION_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
H200_COUNT = 1
H200_MINUTES = 60
H200_HOURLY_USD = Decimal("0.90")
BATCH_CAP_USD = Decimal("0.900")
KNOWN_PROGRAM_ACTUAL_USD = Decimal("5.69984025")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("7.92198425")
PROGRAM_CEILING_USD = Decimal("12.5000000")
FULL_COMPLETION_MAXIMUM_USD = Decimal("12.42198425")

EXPECTED_BATCH_SUMMARIES = {
    index: source_manager.EXPECTED_BATCH_SUMMARIES[index] for index in BATCH_INDICES
}

_NEW_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_recovery_continuation_v2.yaml",
    "docs/massive_medical_kalai_s1_recovery_continuation_v2_protocol.md",
    "scripts/manage_massive_medical_kalai_s1_recovery_continuation_v2.py",
    "scripts/authorize_massive_medical_kalai_s1_recovery_continuation_batch_v2.py",
    "scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v2.py",
    "scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v2.py",
    "scripts/assemble_massive_medical_kalai_s1_recovery_continuation_v2.py",
    "scripts/stage_massive_medical_kalai_s1_recovery_continuation_v2_tillicum.sh",
    "scripts/submit_massive_medical_kalai_s1_recovery_continuation_batch_v2_tillicum.sh",
    "scripts/sbatch_massive_medical_kalai_s1_recovery_continuation_batch_v2_tillicum_h200.sbatch",
    "tests/test_massive_medical_kalai_s1_recovery_continuation_v2.py",
)
REQUIRED_IMPLEMENTATION_FILES = tuple(
    dict.fromkeys(
        _NEW_FILES
        + tuple(source_manager.REQUIRED_IMPLEMENTATION_FILES)
        + tuple(recovery_manager.REQUIRED_IMPLEMENTATION_FILES)
    )
)

canonical_bytes = source_manager.canonical_bytes
sha256_bytes = source_manager.sha256_bytes
sha256_file = source_manager.sha256_file
seal = source_manager.seal
verify_seal = source_manager.verify_seal
load_json = source_manager.load_json
binding = source_manager.binding
raw_binding = source_manager.raw_binding
_write_idempotent = source_manager._write_idempotent
_write_new = source_manager._write_new
_require_no_api_key = source_manager._require_no_api_key
_assert_disjoint = source_manager._assert_disjoint


def git_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean(repo_root, description):
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError(f"{description} repository is not clean")


def batch_id(batch_index):
    if batch_index not in BATCH_INDICES:
        raise ValueError("batch index must be exactly one of 3..7")
    return f"batch_{batch_index:02d}"


def current_exposure(batch_index):
    batch_id(batch_index)
    return CURRENT_CONSERVATIVE_EXPOSURE_USD + BATCH_CAP_USD * Decimal(
        batch_index - 3
    )


def maximum_exposure(batch_index):
    return current_exposure(batch_index) + BATCH_CAP_USD


def _source_paths(source_output, recovery_output):
    source_output = Path(source_output).resolve()
    recovery_output = Path(recovery_output).resolve()
    return {
        "source_plan": source_output / "control" / source_manager.PLAN_NAME,
        "source_stage": source_output / "control" / source_manager.STAGE_NAME,
        "source_batch2_authorization": source_output / "control" / "batches" / "batch_02" / "AUTHORIZATION.json",
        "source_batch2_stopped": source_output / "control" / "batches" / "batch_02" / "STOPPED",
        "source_batch2_result": source_output / "control" / "batches" / "batch_02" / "RESULT.json",
        "source_batch3": source_output / "control" / "batches" / "batch_03",
        "recovery_plan": recovery_output / "control" / recovery_manager.PLAN_NAME,
        "recovery_stage": recovery_output / "control" / recovery_manager.STAGE_NAME,
        "recovery_result": recovery_output / "control" / recovery_manager.RESULT_NAME,
    }


def _verify_recovered_result(paths, recovery_repo):
    plan, _ = recovery_manager.load_plan(
        paths["recovery_plan"], audit_source_state=True
    )
    stage_payload, _ = recovery_manager.load_stage(
        paths["recovery_stage"], recovery_repo, audit_source_state=True
    )
    result = load_json(paths["recovery_result"], "recovered batch-2 result")
    body = verify_seal(result, "recovered batch-2 result")
    expected = recovery_manager._expected_result(
        Path(paths["recovery_plan"]).parents[1], recovery_repo
    )
    if result != expected:
        raise ValueError("recovered batch-2 result differs from CPU derivation")
    if (
        body.get("protocol_id") != RECOVERY_PROTOCOL_ID
        or body.get("status") != EXPECTED_RECOVERY_STATUS
        or body.get("batch_index") != 2
        or body.get("batch_id") != "batch_02"
        or body.get("batch_valid") is not True
        or body.get("recovered_predecessor_evidence_valid") is not True
        or body.get("source_stopped_preserved") is not True
        or body.get("source_result_absent") is not True
        or body.get("derivation_only") is not True
        or body.get("automatic_next_batch_authorized") is not False
        or body.get("batch_3_submission_authorized") is not False
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
    ):
        raise ValueError("recovered batch-2 predecessor contract differs")
    return plan, stage_payload, result, body


def audit_sources(source_output_root, source_repo_root, recovery_output_root, recovery_repo_root):
    source_output = Path(source_output_root).resolve()
    source_repo = Path(source_repo_root).resolve()
    recovery_output = Path(recovery_output_root).resolve()
    recovery_repo = Path(recovery_repo_root).resolve()
    if (
        source_output.name != EXPECTED_SOURCE_OUTPUT_LEAF
        or source_repo.name != EXPECTED_SOURCE_REPO_LEAF
        or recovery_output.name != EXPECTED_RECOVERY_OUTPUT_LEAF
        or recovery_repo.name != EXPECTED_RECOVERY_REPO_LEAF
    ):
        raise ValueError("source or batch-2 recovery namespace leaf differs")
    _assert_disjoint(source_output, source_repo, recovery_output, recovery_repo)
    require_clean(source_repo, "source continuation-v1")
    require_clean(recovery_repo, "batch-2 recovery")
    if git_commit(source_repo) != EXPECTED_SOURCE_REPOSITORY_COMMIT:
        raise ValueError("source continuation-v1 commit differs")

    paths = _source_paths(source_output, recovery_output)
    source_stage, _, source_plan, source_body, source_context = source_manager.load_and_verify_stage(
        paths["source_stage"], source_repo, audit_source=True
    )
    if (
        os.path.lexists(paths["source_batch2_result"])
        or not paths["source_batch2_stopped"].is_file()
        or os.path.lexists(paths["source_batch3"])
    ):
        raise ValueError("source continuation-v1 terminal batch-2 state differs")
    recovery_control = recovery_output / "control"
    expected_recovery_files = {
        recovery_manager.PLAN_NAME,
        recovery_manager.STAGE_NAME,
        recovery_manager.RESULT_NAME,
    }
    if (
        recovery_output.is_symlink()
        or not recovery_output.is_dir()
        or {entry.name for entry in recovery_output.iterdir()} != {"control"}
        or recovery_control.is_symlink()
        or not recovery_control.is_dir()
        or {entry.name for entry in recovery_control.iterdir()} != expected_recovery_files
        or any(entry.is_symlink() or not entry.is_file() for entry in recovery_control.iterdir())
    ):
        raise ValueError("batch-2 recovery inventory differs")
    recovery_plan, recovery_stage_payload, recovery_result, recovery_body = (
        _verify_recovered_result(paths, recovery_repo)
    )
    recovery_bindings = {
        recovery_manager.PLAN_NAME: binding(paths["recovery_plan"], recovery_plan),
        recovery_manager.STAGE_NAME: binding(paths["recovery_stage"], recovery_stage_payload),
        recovery_manager.RESULT_NAME: binding(paths["recovery_result"], recovery_result),
    }
    continuation_batches = []
    for index in BATCH_INDICES:
        expected = EXPECTED_BATCH_SUMMARIES[index]
        item = source_context["source_body"]["batches"][index - 1]
        summary = item.get("summary")
        if (
            item.get("batch_index") != index
            or item.get("batch_id") != batch_id(index)
            or summary.get("row_count") != expected["row_count"]
            or summary.get("rows_by_phase") != {"benefit": expected["benefit"], "medical": expected["medical"]}
            or summary.get("maximum_new_attempts") != expected["attempts"]
        ):
            raise ValueError(f"source {batch_id(index)} summary differs")
        continuation_batches.append(
            {"batch_index": index, "batch_id": batch_id(index), "summary": summary}
        )
    return {
        "source_output": source_output,
        "source_repo": source_repo,
        "source_plan": source_plan,
        "source_body": source_body,
        "source_stage": source_stage,
        "source_context": source_context,
        "recovery_output": recovery_output,
        "recovery_repo": recovery_repo,
        "recovery_plan": recovery_plan,
        "recovery_stage": recovery_stage_payload,
        "recovery_result": recovery_result,
        "recovery_body": recovery_body,
        "recovery_bindings": recovery_bindings,
        "paths": paths,
        "continuation_batches": continuation_batches,
    }


def _expected_plan_body(context, output_root):
    paths = context["paths"]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "scientific_batch_protocol_id": SCIENTIFIC_BATCH_PROTOCOL_ID,
        "recovery_protocol_id": RECOVERY_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "recovery_continuation_plan",
        "status": "CPU_ONLY_RECOVERY_CONTINUATION_PLANNED_NO_AUTHORITY",
        "output_root": str(Path(output_root).resolve()),
        "source_repository": {"path": str(context["source_repo"]), "commit": EXPECTED_SOURCE_REPOSITORY_COMMIT, "clean": True},
        "source_output_root": str(context["source_output"]),
        "source_bindings": {
            "continuation_v1_plan": binding(paths["source_plan"], context["source_plan"]),
            "continuation_v1_cpu_stage": binding(paths["source_stage"], context["source_stage"]),
            "batch_02_authorization": binding(paths["source_batch2_authorization"]),
            "batch_02_stopped": raw_binding(paths["source_batch2_stopped"]),
        },
        "recovery_repository": {"path": str(context["recovery_repo"]), "commit": git_commit(context["recovery_repo"]), "clean": True},
        "recovery_output_root": str(context["recovery_output"]),
        "recovery_bindings": context["recovery_bindings"],
        "recovered_batch_2": {
            "batch_index": 2,
            "batch_id": "batch_02",
            "source_job_id": "271409",
            "source_stopped_preserved": True,
            "source_result_absent": True,
            "predecessor_evidence_valid": True,
            "scientific_outputs_regenerated": False,
        },
        "preserved_terminal_history": {"batch_1_stopped": True, "batch_2_stopped": True},
        "continuation_batches": context["continuation_batches"],
        "execution_policy": {
            "batch_indices": list(BATCH_INDICES),
            "one_batch_per_separate_authority": True,
            "automatic_next_batch": False,
            "partial_resume": False,
            "restart": False,
            "retry": False,
            "replacement": False,
            "requeue": False,
            "external_api_calls": False,
            "batch_1_or_2_regeneration": False,
            "source_namespaces_read_only": True,
        },
        "planned_authority_not_granted": {
            "gpu_jobs_per_batch": 1,
            "h200_count": H200_COUNT,
            "h200_minutes": H200_MINUTES,
            "maximum_cost_usd_per_batch": float(BATCH_CAP_USD),
            "remaining_batch_count": len(BATCH_INDICES),
            "remaining_batch_cap_usd": float(BATCH_CAP_USD * len(BATCH_INDICES)),
            "gpu_jobs_authorized_now": 0,
            "external_api_calls_authorized_now": 0,
        },
        "accounting_not_authority": {
            "known_program_actual_usd": float(KNOWN_PROGRAM_ACTUAL_USD),
            "current_conservative_exposure_usd": float(CURRENT_CONSERVATIVE_EXPOSURE_USD),
            "maximum_after_all_remaining_batch_caps_usd": float(FULL_COMPLETION_MAXIMUM_USD),
            "program_ceiling_usd": float(PROGRAM_CEILING_USD),
        },
        "gpu_jobs": 0,
        "external_api_calls": 0,
    }


def _audit_namespace(output_root, allowed_control):
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.name != EXPECTED_OUTPUT_LEAF:
        raise ValueError("recovery-continuation-v2 output leaf differs")
    if os.path.lexists(output) and (output.is_symlink() or not output.is_dir()):
        raise ValueError("recovery-continuation-v2 output is unsafe")
    for name in ("generation", "assembled", "judge", "logs"):
        if os.path.lexists(output / name):
            raise ValueError(f"CPU-stage namespace contains forbidden {name}")
    if output.exists() and {item.name for item in output.iterdir()} != {"control"}:
        raise ValueError("CPU-stage namespace contains unexpected top-level state")
    control = output / "control"
    if control.exists():
        entries = list(control.iterdir())
        if control.is_symlink() or not control.is_dir() or {item.name for item in entries} - set(allowed_control):
            raise ValueError("CPU-stage control inventory differs")
        if any(item.is_symlink() or not item.is_file() for item in entries):
            raise ValueError("CPU-stage control contains unsafe state")


def prepare(args):
    _require_no_api_key()
    raw_output = Path(os.path.abspath(os.fspath(args.output_root)))
    _audit_namespace(raw_output, {PLAN_NAME, STAGE_NAME})
    _assert_disjoint(raw_output, args.source_output_root, args.source_repo_root, args.recovery_output_root, args.recovery_repo_root)
    context = audit_sources(args.source_output_root, args.source_repo_root, args.recovery_output_root, args.recovery_repo_root)
    plan = seal(_expected_plan_body(context, raw_output.resolve()))
    disposition = _write_idempotent(raw_output.resolve() / "control" / PLAN_NAME, plan, "recovery-continuation-v2 plan")
    print(json.dumps({"status": f"KALAI_S1_RECOVERY_CONTINUATION_V2_PLAN_{disposition}", "plan_payload_sha256": plan[SEAL_FIELD], "continuation_batches": list(BATCH_INDICES), "gpu_jobs": 0, "external_api_calls": 0}, sort_keys=True))
    return plan, context


def load_and_verify_plan(path, *, audit_source=True):
    path = Path(path).resolve()
    payload = load_json(path, "recovery-continuation-v2 plan")
    body = verify_seal(payload, "recovery-continuation-v2 plan")
    output = Path(body.get("output_root", "")).resolve()
    if path != output / "control" / PLAN_NAME:
        raise ValueError("recovery-continuation-v2 plan path differs")
    context = None
    if audit_source:
        context = audit_sources(body.get("source_output_root", ""), body.get("source_repository", {}).get("path", ""), body.get("recovery_output_root", ""), body.get("recovery_repository", {}).get("path", ""))
        if body != _expected_plan_body(context, output):
            raise ValueError("recovery-continuation-v2 plan differs from sources")
    elif body.get("protocol_id") != PROTOCOL_ID or body.get("status") != "CPU_ONLY_RECOVERY_CONTINUATION_PLANNED_NO_AUTHORITY" or body.get("gpu_jobs") != 0 or body.get("external_api_calls") != 0:
        raise ValueError("recovery-continuation-v2 plan identity differs")
    return payload, body, context


def implementation_bindings(repo_root):
    repo = Path(repo_root).resolve()
    result = {}
    for relative in REQUIRED_IMPLEMENTATION_FILES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required implementation is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def stage(args):
    _require_no_api_key()
    raw_output = Path(os.path.abspath(os.fspath(args.output_root)))
    _audit_namespace(raw_output, {PLAN_NAME, STAGE_NAME})
    repo = Path(args.repo_root).resolve()
    if repo.name != EXPECTED_REPO_LEAF:
        raise ValueError("recovery-continuation-v2 repository leaf differs")
    _assert_disjoint(raw_output, repo, args.source_output_root, args.source_repo_root, args.recovery_output_root, args.recovery_repo_root)
    require_clean(repo, "recovery-continuation-v2")
    plan, _ = prepare(args)
    output = raw_output.resolve()
    stage_payload = seal({
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "recovery_protocol_id": RECOVERY_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "recovery_continuation_cpu_stage",
        "status": "CPU_STAGED_NO_GPU_OR_API_AUTHORITY",
        "repository": {"path": str(repo), "commit": git_commit(repo), "clean": True},
        "continuation_plan": binding(output / "control" / PLAN_NAME, plan),
        "implementation_sha256": implementation_bindings(repo),
        "execution_policy": {"batch_indices": list(BATCH_INDICES), "gpu_jobs_authorized": 0, "external_api_calls_authorized": 0, "batch_authorizations_created": 0, "automatic_next_batch_authorized": False, "restart_resume_retry_replacement_requeue_authorized": False, "source_namespaces_read_only": True},
        "new_cost_cap_usd": 0.0,
        "gpu_jobs": 0,
        "external_api_calls": 0,
    })
    disposition = _write_idempotent(output / "control" / STAGE_NAME, stage_payload, "recovery-continuation-v2 CPU stage")
    load_and_verify_stage(output / "control" / STAGE_NAME, repo, audit_source=True)
    print(json.dumps({"status": f"KALAI_S1_RECOVERY_CONTINUATION_V2_CPU_STAGE_{disposition}", "plan_payload_sha256": plan[SEAL_FIELD], "cpu_stage_payload_sha256": stage_payload[SEAL_FIELD], "batch_authorizations_created": 0, "gpu_jobs": 0, "external_api_calls": 0}, sort_keys=True))
    return stage_payload


def load_and_verify_stage(path, repo_root, *, audit_source=True):
    path = Path(path).resolve()
    repo = Path(repo_root).resolve()
    payload = load_json(path, "recovery-continuation-v2 CPU stage")
    body = verify_seal(payload, "recovery-continuation-v2 CPU stage")
    policy = body.get("execution_policy")
    expected_policy = {"batch_indices": list(BATCH_INDICES), "gpu_jobs_authorized": 0, "external_api_calls_authorized": 0, "batch_authorizations_created": 0, "automatic_next_batch_authorized": False, "restart_resume_retry_replacement_requeue_authorized": False, "source_namespaces_read_only": True}
    repository = body.get("repository", {})
    if body.get("protocol_id") != PROTOCOL_ID or body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY" or body.get("new_cost_cap_usd") != 0.0 or body.get("gpu_jobs") != 0 or body.get("external_api_calls") != 0 or policy != expected_policy or set(repository) != {"path", "commit", "clean"} or repository.get("path") != str(repo) or repository.get("clean") is not True:
        raise ValueError("recovery-continuation-v2 CPU-stage policy differs")
    require_clean(repo, "recovery-continuation-v2")
    if git_commit(repo) != repository.get("commit") or body.get("implementation_sha256") != implementation_bindings(repo):
        raise ValueError("recovery-continuation-v2 implementation binding differs")
    plan_binding = body.get("continuation_plan", {})
    plan, plan_body, context = load_and_verify_plan(plan_binding.get("path", ""), audit_source=audit_source)
    if plan_binding != binding(plan_binding["path"], plan) or path != Path(plan_body["output_root"]) / "control" / STAGE_NAME:
        raise ValueError("recovery-continuation-v2 stage path differs")
    return payload, body, plan, plan_body, context


def _load_workflow(output_root, repo_root, *, audit_source=True):
    output = Path(output_root).resolve()
    stage_payload, stage_body, plan, plan_body, context = load_and_verify_stage(output / "control" / STAGE_NAME, repo_root, audit_source=audit_source)
    return output, plan, plan_body, stage_payload, stage_body, context


def preflight(args):
    _require_no_api_key()
    _, plan, _, _, _, context = _load_workflow(args.output_root, args.repo_root, audit_source=True)
    source_manager.original_runtime._preflight(context["source_context"]["source_plan"], context["source_context"]["source_body"], args.batch_index)
    print(json.dumps({"status": "KALAI_S1_RECOVERY_CONTINUATION_V2_BATCH_PREFLIGHT_VALID", "batch_id": batch_id(args.batch_index), "continuation_plan_payload_sha256": plan[SEAL_FIELD], "recovered_batch_2_bound": True, "reference_models_loaded": 0, "gpu_jobs": 0, "external_api_calls": 0}, sort_keys=True))


def self_test():
    assert BATCH_INDICES == tuple(range(3, 8))
    assert current_exposure(3) == Decimal("7.92198425")
    assert maximum_exposure(3) == Decimal("8.82198425")
    assert maximum_exposure(7) == FULL_COMPLETION_MAXIMUM_USD
    assert FULL_COMPLETION_MAXIMUM_USD < PROGRAM_CEILING_USD
    assert sum(item["row_count"] for item in EXPECTED_BATCH_SUMMARIES.values()) == 125
    assert sum(item["attempts"] for item in EXPECTED_BATCH_SUMMARIES.values()) == 2109
    print("MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V2_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-output-root", required=True)
    common.add_argument("--source-repo-root", required=True)
    common.add_argument("--recovery-output-root", required=True)
    common.add_argument("--recovery-repo-root", required=True)
    common.add_argument("--output-root", required=True)
    sub.add_parser("prepare", parents=[common])
    staged = sub.add_parser("stage", parents=[common])
    staged.add_argument("--repo-root", required=True)
    pf = sub.add_parser("preflight")
    pf.add_argument("--output-root", required=True)
    pf.add_argument("--repo-root", required=True)
    pf.add_argument("--batch-index", required=True, type=int, choices=BATCH_INDICES)
    args = parser.parse_args(argv)
    if args.command == "self-test": self_test()
    elif args.command == "prepare": prepare(args)
    elif args.command == "stage": stage(args)
    else: preflight(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
