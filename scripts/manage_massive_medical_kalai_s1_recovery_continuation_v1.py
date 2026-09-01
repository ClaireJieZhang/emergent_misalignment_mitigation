#!/usr/bin/env python3
"""Versioned recovery-aware controller for Kalai ``s=1`` batches 2--7.

CPU staging treats the failed original batch-1 namespace and its derivation-only
recovery as immutable inputs.  Later, separately authorized GPU invocations may
run exactly one of the still-unattempted original batches 2--7 in this workflow's
fresh namespace.  The original batch assignment and scientific sampler are
reused byte-for-byte; this controller changes only provenance and predecessor
handling.
"""

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
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original_planner = _load_module(
    "_kalai_s1_original_completion_planner_for_recovery_continuation",
    "prepare_massive_medical_kalai_s1_completion_batches_v1.py",
)
original_runtime = _load_module(
    "_kalai_s1_original_completion_runtime_for_recovery_continuation",
    "sample_massive_medical_kalai_s1_completion_batch_v1.py",
)
original_stage = _load_module(
    "_kalai_s1_original_completion_stage_for_recovery_continuation",
    "prepare_massive_medical_kalai_s1_completion_batches_stage_v1.py",
)
recovery_planner = _load_module(
    "_kalai_s1_batch1_recovery_planner_for_continuation",
    "prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)
recovery_stage = _load_module(
    "_kalai_s1_batch1_recovery_stage_for_continuation",
    "stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)
recovery_evaluator = _load_module(
    "_kalai_s1_batch1_recovery_evaluator_for_continuation",
    "evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)


PROTOCOL_ID = "massive_medical_kalai_s1_recovery_continuation_v1"
SOURCE_PROTOCOL_ID = original_planner.PROTOCOL_ID
TRACE_PROTOCOL_ID = original_planner.CONTROLLER_PROTOCOL_ID
RECOVERY_PROTOCOL_ID = recovery_planner.PROTOCOL_ID
METHOD_ID = original_planner.METHOD_ID
SEAL_FIELD = original_planner.SEAL_FIELD
BATCH_INDICES = tuple(range(2, 8))
PHASES = original_planner.PHASES

EXPECTED_BRANCH = "claire/massive-medical-kalai-s1-recovery-continuation-v1"
EXPECTED_OUTPUT_LEAF = PROTOCOL_ID
EXPECTED_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-recovery-continuation-v1"
EXPECTED_SOURCE_OUTPUT_LEAF = "massive_medical_kalai_s1_completion_batches_v1"
EXPECTED_SOURCE_REPO_LEAF = "subliminal-mitigate-mmu-kalai-s1-completion-batches-v1"
EXPECTED_RECOVERY_OUTPUT_LEAF = "massive_medical_kalai_s1_batch1_result_recovery_v1"
EXPECTED_RECOVERY_REPO_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-batch1-result-recovery-v1"
)
EXPECTED_SOURCE_REPOSITORY_COMMIT = (
    "a1e8ca218635af6dafdca7ddb3e9d42d153d6921"
)
EXPECTED_RECOVERY_REPOSITORY_COMMIT = (
    "d5365c755cc27555451885013b4217b2a7561cb1"
)
EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256 = (
    "29f9e969918f766dcaa09d80b60ead2cd46a484dc2e0e23bdb592ac9409d2904"
)
EXPECTED_SOURCE_CPU_STAGE_PAYLOAD_SHA256 = (
    "e1b759ad583037fd6cbe085919a63fa05190b33d08e18019289869984b28e174"
)
EXPECTED_SOURCE_BATCH1_AUTHORIZATION_PAYLOAD_SHA256 = (
    "d72d75a8bb4900ddd1ba1995dccbd8bf9d019ff684310a615fd33737859f373b"
)
EXPECTED_SOURCE_BATCH1_STOPPED_FILE_SHA256 = (
    "f2f8ef71c52f32ca47672c45dac01b8eaf592bcd834b96ce881b8dd0b54504f0"
)
EXPECTED_RECOVERY_FILES = {
    "RECOVERY_PLAN.json": (
        28691,
        "fe3b402547cc229ebba9412e6796068147918d3b54a105c7ea30a7249ed75aa2",
        "5281686afbe70bc2152e8624dab1ec40d05413c81153b8fcb197a4dd210ca02d",
    ),
    "CPU_STAGE.json": (
        2919,
        "534b0de0fe366b63242235470ebc46cf252d73487b4695713225778c8e07b6ba",
        "b4f04e781c183b1ec5d9d4f541437ec6f41c77385f72267144c4f307b82b15cf",
    ),
    "RECOVERED_RESULT.json": (
        9389,
        "a9ba5667f23d591a7c5d91d5c9f40725bbd6f6a8b57376d188dca42fdf5d8a49",
        "c11b84ba36b858dc9eb41f22252dcc864fe22a1089e0e0ef428aaf9ba2393b02",
    ),
}
EXPECTED_BATCH_SUMMARIES = {
    2: {"row_count": 24, "benefit": 16, "medical": 8, "attempts": 418, "benefit_attempts": 294, "medical_attempts": 124},
    3: {"row_count": 25, "benefit": 17, "medical": 8, "attempts": 419, "benefit_attempts": 308, "medical_attempts": 111},
    4: {"row_count": 25, "benefit": 17, "medical": 8, "attempts": 418, "benefit_attempts": 308, "medical_attempts": 110},
    5: {"row_count": 25, "benefit": 17, "medical": 8, "attempts": 422, "benefit_attempts": 307, "medical_attempts": 115},
    6: {"row_count": 25, "benefit": 14, "medical": 11, "attempts": 426, "benefit_attempts": 261, "medical_attempts": 165},
    7: {"row_count": 25, "benefit": 16, "medical": 9, "attempts": 424, "benefit_attempts": 293, "medical_attempts": 131},
}

PLAN_NAME = "RECOVERY_CONTINUATION_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
H200_COUNT = 1
H200_MINUTES = 60
H200_HOURLY_USD = Decimal("0.90")
BATCH_CAP_USD = Decimal("0.900")
KNOWN_PROGRAM_ACTUAL_USD = Decimal("5.29334025")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("7.02198425")
PROGRAM_CEILING_USD = Decimal("12.5000000")
FULL_COMPLETION_MAXIMUM_USD = Decimal("12.42198425")

_NEW_IMPLEMENTATION_FILES = (
    "configs/pipelines/massive_medical_kalai_s1_recovery_continuation_v1.yaml",
    "docs/massive_medical_kalai_s1_recovery_continuation_v1_protocol.md",
    "scripts/manage_massive_medical_kalai_s1_recovery_continuation_v1.py",
    "scripts/authorize_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
    "scripts/sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
    "scripts/evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
    "scripts/assemble_massive_medical_kalai_s1_recovery_continuation_v1.py",
    "scripts/stage_massive_medical_kalai_s1_recovery_continuation_v1_tillicum.sh",
    "scripts/submit_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum.sh",
    "scripts/sbatch_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum_h200.sbatch",
    "scripts/prepare_massive_medical_kalai_s1_completion_batches_v1.py",
    "scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py",
    "scripts/prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
    "scripts/stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
    "scripts/evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
    "tests/test_massive_medical_kalai_s1_recovery_continuation_v1.py",
)
REQUIRED_IMPLEMENTATION_FILES = tuple(
    dict.fromkeys(
        _NEW_IMPLEMENTATION_FILES
        + tuple(original_stage.REQUIRED_IMPLEMENTATION_FILES)
        + tuple(recovery_stage.REQUIRED_IMPLEMENTATION_FILES)
    )
)


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def seal(body):
    payload = dict(body)
    payload.pop(SEAL_FIELD, None)
    payload[SEAL_FIELD] = sha256_bytes(canonical_bytes(payload))
    return payload


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} has an invalid {SEAL_FIELD}")
    return body


def load_json(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def binding(path, payload=None, description="sealed JSON"):
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink():
        raise ValueError(f"{description} is a symlink: {path}")
    if payload is None:
        payload = load_json(path, description)
    verify_seal(payload, description)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "payload_sha256": payload[SEAL_FIELD],
    }


def raw_binding(path):
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"raw artifact is absent or unsafe: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def git_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def git_branch(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "branch", "--show-current"],
        text=True,
    ).strip()


def require_clean(repo_root, description):
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError(f"{description} repository is not clean")


def _write_idempotent(path, payload, description):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.path.lexists(path):
        observed = load_json(path, f"existing {description}")
        verify_seal(observed, f"existing {description}")
        if observed != payload:
            raise ValueError(f"existing {description} differs")
        return "AUDITED"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "CREATED"


def _write_new(path, payload, description):
    if os.path.lexists(path):
        raise ValueError(f"{description} already exists")
    _write_idempotent(path, payload, description)


def _require_no_api_key():
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent")


def batch_id(batch_index):
    if batch_index not in BATCH_INDICES:
        raise ValueError("batch index must be exactly one of 2..7")
    return f"batch_{batch_index:02d}"


def current_exposure(batch_index):
    batch_id(batch_index)
    return CURRENT_CONSERVATIVE_EXPOSURE_USD + BATCH_CAP_USD * Decimal(
        batch_index - 2
    )


def maximum_exposure(batch_index):
    return current_exposure(batch_index) + BATCH_CAP_USD


def _assert_disjoint(*paths):
    resolved = [Path(path).resolve() for path in paths]
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("workflow namespaces overlap")


def _source_paths(source_output, recovery_output):
    source_output = Path(source_output).resolve()
    recovery_output = Path(recovery_output).resolve()
    return {
        "source_plan": source_output / "control" / "COMPLETION_BATCH_PLAN.json",
        "source_stage": source_output / "control" / "CPU_STAGE.json",
        "source_authorization": source_output / "control" / "batches" / "batch_01" / "AUTHORIZATION.json",
        "source_stopped": source_output / "control" / "batches" / "batch_01" / "STOPPED",
        "source_result": source_output / "control" / "batches" / "batch_01" / "RESULT.json",
        "source_batch2": source_output / "control" / "batches" / "batch_02",
        "recovery_plan": recovery_output / "control" / "RECOVERY_PLAN.json",
        "recovery_stage": recovery_output / "control" / "CPU_STAGE.json",
        "recovery_result": recovery_output / "control" / "RECOVERED_RESULT.json",
    }


def _validate_batch_summaries(plan_body):
    batches = plan_body.get("batches")
    if not isinstance(batches, list) or len(batches) != 7:
        raise ValueError("source completion batch inventory differs")
    result = []
    for index in BATCH_INDICES:
        item = batches[index - 1]
        summary = item.get("summary")
        expected = EXPECTED_BATCH_SUMMARIES[index]
        if (
            item.get("batch_index") != index
            or item.get("batch_id") != batch_id(index)
            or not isinstance(summary, dict)
            or summary.get("row_count") != expected["row_count"]
            or summary.get("rows_by_phase")
            != {"benefit": expected["benefit"], "medical": expected["medical"]}
            or summary.get("maximum_new_attempts") != expected["attempts"]
            or summary.get("maximum_new_attempts_by_phase")
            != {
                "benefit": expected["benefit_attempts"],
                "medical": expected["medical_attempts"],
            }
        ):
            raise ValueError(f"source {batch_id(index)} summary differs")
        result.append(
            {
                "batch_index": index,
                "batch_id": batch_id(index),
                "summary": summary,
            }
        )
    return result


def audit_sources(
    source_output_root,
    source_repo_root,
    recovery_output_root,
    recovery_repo_root,
):
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
        raise ValueError("source or recovery namespace leaf differs")
    _assert_disjoint(source_output, source_repo, recovery_output, recovery_repo)
    require_clean(source_repo, "source completion")
    require_clean(recovery_repo, "batch-1 recovery")
    if git_commit(source_repo) != EXPECTED_SOURCE_REPOSITORY_COMMIT:
        raise ValueError("source completion repository commit differs")
    if git_commit(recovery_repo) != EXPECTED_RECOVERY_REPOSITORY_COMMIT:
        raise ValueError("batch-1 recovery repository commit differs")

    paths = _source_paths(source_output, recovery_output)
    source_plan, source_body = original_runtime._load_batch_plan(
        paths["source_plan"]
    )
    if source_plan[SEAL_FIELD] != EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256:
        raise ValueError("source completion plan seal differs")
    source_stage = load_json(paths["source_stage"], "source completion CPU stage")
    original_planner.verify_seal(source_stage, "source completion CPU stage")
    source_authorization = load_json(
        paths["source_authorization"], "source batch-1 authorization"
    )
    original_planner.verify_seal(
        source_authorization, "source batch-1 authorization"
    )
    if (
        source_stage[SEAL_FIELD] != EXPECTED_SOURCE_CPU_STAGE_PAYLOAD_SHA256
        or source_authorization[SEAL_FIELD]
        != EXPECTED_SOURCE_BATCH1_AUTHORIZATION_PAYLOAD_SHA256
        or raw_binding(paths["source_stopped"])["file_sha256"]
        != EXPECTED_SOURCE_BATCH1_STOPPED_FILE_SHA256
        or os.path.lexists(paths["source_result"])
        or os.path.lexists(paths["source_batch2"])
    ):
        raise ValueError("source terminal batch-1 state differs")

    recovery_control = recovery_output / "control"
    if (
        recovery_output.is_symlink()
        or not recovery_output.is_dir()
        or {path.name for path in recovery_output.iterdir()} != {"control"}
        or recovery_control.is_symlink()
        or not recovery_control.is_dir()
        or {path.name for path in recovery_control.iterdir()}
        != set(EXPECTED_RECOVERY_FILES)
        or any(
            path.is_symlink() or not path.is_file()
            for path in recovery_control.iterdir()
        )
    ):
        raise ValueError("batch-1 recovery control inventory differs")
    recovery_plan, _ = recovery_planner.load_and_verify_plan(
        paths["recovery_plan"], audit_source=True
    )
    recovery_stage_payload, _ = recovery_stage.load_and_verify_stage(
        paths["recovery_stage"], recovery_repo, audit_source=True
    )
    recovery_result = load_json(
        paths["recovery_result"], "recovered batch-1 result"
    )
    recovery_body = recovery_planner.verify_seal(
        recovery_result, "recovered batch-1 result"
    )
    expected_result = recovery_evaluator._derive_result(
        paths["recovery_plan"], paths["recovery_stage"], recovery_repo
    )
    if recovery_result != expected_result:
        raise ValueError("recovered batch-1 result differs from CPU derivation")
    observed_recovery = {
        "RECOVERY_PLAN.json": binding(paths["recovery_plan"], recovery_plan),
        "CPU_STAGE.json": binding(paths["recovery_stage"], recovery_stage_payload),
        "RECOVERED_RESULT.json": binding(paths["recovery_result"], recovery_result),
    }
    for name, expected in EXPECTED_RECOVERY_FILES.items():
        observed = observed_recovery[name]
        if (
            observed["size_bytes"] != expected[0]
            or observed["file_sha256"] != expected[1]
            or observed["payload_sha256"] != expected[2]
        ):
            raise ValueError(f"batch-1 recovery pin differs: {name}")
    if (
        recovery_body.get("status")
        != "MASSIVE_MEDICAL_KALAI_S1_BATCH1_RESULT_RECOVERED_CPU_ONLY"
        or recovery_body.get("batch_index") != 1
        or recovery_body.get("batch_valid") is not True
        or recovery_body.get("recovered_predecessor_evidence_valid") is not True
        or recovery_body.get("source_stopped_preserved") is not True
        or recovery_body.get("source_result_absent") is not True
        or recovery_body.get("automatic_next_batch_authorized") is not False
        or recovery_body.get("batch_2_submission_authorized") is not False
        or recovery_body.get("external_api_calls") != 0
        or recovery_body.get("gpu_jobs") != 0
    ):
        raise ValueError("recovered batch-1 predecessor contract differs")

    return {
        "source_output": source_output,
        "source_repo": source_repo,
        "recovery_output": recovery_output,
        "recovery_repo": recovery_repo,
        "paths": paths,
        "source_plan": source_plan,
        "source_body": source_body,
        "source_stage": source_stage,
        "source_authorization": source_authorization,
        "recovery_plan": recovery_plan,
        "recovery_stage": recovery_stage_payload,
        "recovery_result": recovery_result,
        "recovery_body": recovery_body,
        "recovery_bindings": observed_recovery,
        "continuation_batches": _validate_batch_summaries(source_body),
    }


def _expected_plan_body(context, output_root):
    paths = context["paths"]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "trace_protocol_id": TRACE_PROTOCOL_ID,
        "recovery_protocol_id": RECOVERY_PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "recovery_continuation_plan",
        "status": "CPU_ONLY_RECOVERY_CONTINUATION_PLANNED_NO_AUTHORITY",
        "output_root": str(Path(output_root).resolve()),
        "source_repository": {
            "path": str(context["source_repo"]),
            "commit": EXPECTED_SOURCE_REPOSITORY_COMMIT,
            "clean": True,
        },
        "source_output_root": str(context["source_output"]),
        "source_bindings": {
            "completion_batch_plan": binding(paths["source_plan"], context["source_plan"]),
            "completion_cpu_stage": binding(paths["source_stage"], context["source_stage"]),
            "batch_01_authorization": binding(paths["source_authorization"], context["source_authorization"]),
            "batch_01_stopped": raw_binding(paths["source_stopped"]),
        },
        "recovery_repository": {
            "path": str(context["recovery_repo"]),
            "commit": EXPECTED_RECOVERY_REPOSITORY_COMMIT,
            "clean": True,
        },
        "recovery_output_root": str(context["recovery_output"]),
        "recovery_bindings": context["recovery_bindings"],
        "recovered_batch_1": {
            "batch_index": 1,
            "batch_id": "batch_01",
            "source_job_id": "270983",
            "source_stopped_preserved": True,
            "source_result_absent": True,
            "predecessor_evidence_valid": True,
            "scientific_outputs_regenerated": False,
        },
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
            "batch_1_regeneration": False,
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
        raise ValueError("recovery-continuation output leaf differs")
    if os.path.lexists(output) and (output.is_symlink() or not output.is_dir()):
        raise ValueError("recovery-continuation output is unsafe")
    for name in ("generation", "assembled", "judge", "logs"):
        if os.path.lexists(output / name):
            raise ValueError(f"CPU-stage namespace contains forbidden {name}")
    if output.exists() and {path.name for path in output.iterdir()} != {"control"}:
        raise ValueError("CPU-stage namespace contains unexpected top-level state")
    control = output / "control"
    if control.exists():
        if control.is_symlink() or not control.is_dir():
            raise ValueError("recovery-continuation control is unsafe")
        entries = list(control.iterdir())
        observed = {path.name for path in entries}
        if observed - set(allowed_control):
            raise ValueError("recovery-continuation control inventory differs")
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError("recovery-continuation control contains an unsafe entry")


def prepare(args):
    _require_no_api_key()
    raw_output = Path(os.path.abspath(os.fspath(args.output_root)))
    _audit_namespace(raw_output, {PLAN_NAME, STAGE_NAME})
    output = raw_output.resolve()
    _assert_disjoint(
        output,
        args.source_output_root,
        args.source_repo_root,
        args.recovery_output_root,
        args.recovery_repo_root,
    )
    context = audit_sources(
        args.source_output_root,
        args.source_repo_root,
        args.recovery_output_root,
        args.recovery_repo_root,
    )
    plan = seal(_expected_plan_body(context, output))
    disposition = _write_idempotent(
        output / "control" / PLAN_NAME, plan, "recovery-continuation plan"
    )
    print(
        json.dumps(
            {
                "status": f"KALAI_S1_RECOVERY_CONTINUATION_PLAN_{disposition}",
                "plan_payload_sha256": plan[SEAL_FIELD],
                "continuation_batches": list(BATCH_INDICES),
                "gpu_jobs": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return plan, context


def load_and_verify_plan(path, *, audit_source=True):
    path = Path(path).resolve()
    payload = load_json(path, "recovery-continuation plan")
    body = verify_seal(payload, "recovery-continuation plan")
    output = Path(body.get("output_root", "")).resolve()
    if path != output / "control" / PLAN_NAME:
        raise ValueError("recovery-continuation plan path differs")
    context = None
    if audit_source:
        context = audit_sources(
            body.get("source_output_root", ""),
            body.get("source_repository", {}).get("path", ""),
            body.get("recovery_output_root", ""),
            body.get("recovery_repository", {}).get("path", ""),
        )
        if body != _expected_plan_body(context, output):
            raise ValueError("recovery-continuation plan differs from sources")
    else:
        if (
            body.get("protocol_id") != PROTOCOL_ID
            or body.get("status")
            != "CPU_ONLY_RECOVERY_CONTINUATION_PLANNED_NO_AUTHORITY"
            or body.get("gpu_jobs") != 0
            or body.get("external_api_calls") != 0
        ):
            raise ValueError("recovery-continuation plan identity differs")
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
    output = raw_output.resolve()
    repo = Path(args.repo_root).resolve()
    if repo.name != EXPECTED_REPO_LEAF:
        raise ValueError("recovery-continuation repository leaf differs")
    _assert_disjoint(
        output,
        repo,
        args.source_output_root,
        args.source_repo_root,
        args.recovery_output_root,
        args.recovery_repo_root,
    )
    require_clean(repo, "recovery-continuation")
    if git_branch(repo) != EXPECTED_BRANCH:
        raise ValueError("recovery-continuation branch differs")
    plan, _ = prepare(args)
    plan_path = output / "control" / PLAN_NAME
    stage_payload = seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "recovery_protocol_id": RECOVERY_PROTOCOL_ID,
            "method_id": METHOD_ID,
            "stage": "recovery_continuation_cpu_stage",
            "status": "CPU_STAGED_NO_GPU_OR_API_AUTHORITY",
            "repository": {
                "path": str(repo),
                "commit": git_commit(repo),
                "branch": EXPECTED_BRANCH,
                "clean": True,
            },
            "continuation_plan": binding(plan_path, plan),
            "implementation_sha256": implementation_bindings(repo),
            "execution_policy": {
                "batch_indices": list(BATCH_INDICES),
                "gpu_jobs_authorized": 0,
                "external_api_calls_authorized": 0,
                "batch_authorizations_created": 0,
                "automatic_next_batch_authorized": False,
                "restart_resume_retry_replacement_requeue_authorized": False,
                "source_namespaces_read_only": True,
            },
            "new_cost_cap_usd": 0.0,
            "gpu_jobs": 0,
            "external_api_calls": 0,
        }
    )
    disposition = _write_idempotent(
        output / "control" / STAGE_NAME,
        stage_payload,
        "recovery-continuation CPU stage",
    )
    load_and_verify_stage(
        output / "control" / STAGE_NAME, repo, audit_source=True
    )
    print(
        json.dumps(
            {
                "status": f"KALAI_S1_RECOVERY_CONTINUATION_CPU_STAGE_{disposition}",
                "plan_payload_sha256": plan[SEAL_FIELD],
                "cpu_stage_payload_sha256": stage_payload[SEAL_FIELD],
                "batch_authorizations_created": 0,
                "gpu_jobs": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return stage_payload


def load_and_verify_stage(path, repo_root, *, audit_source=True):
    path = Path(path).resolve()
    repo = Path(repo_root).resolve()
    payload = load_json(path, "recovery-continuation CPU stage")
    body = verify_seal(payload, "recovery-continuation CPU stage")
    repository = body.get("repository")
    policy = body.get("execution_policy")
    if (
        set(body)
        != {
            "schema_version",
            "protocol_id",
            "source_protocol_id",
            "recovery_protocol_id",
            "method_id",
            "stage",
            "status",
            "repository",
            "continuation_plan",
            "implementation_sha256",
            "execution_policy",
            "new_cost_cap_usd",
            "gpu_jobs",
            "external_api_calls",
        }
        or body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("recovery_protocol_id") != RECOVERY_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "recovery_continuation_cpu_stage"
        or body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or body.get("new_cost_cap_usd") != 0.0
        or body.get("gpu_jobs") != 0
        or body.get("external_api_calls") != 0
        or not isinstance(repository, dict)
        or set(repository) != {"path", "commit", "branch", "clean"}
        or repository.get("path") != str(repo)
        or repository.get("branch") != EXPECTED_BRANCH
        or repository.get("clean") is not True
        or policy
        != {
            "batch_indices": list(BATCH_INDICES),
            "gpu_jobs_authorized": 0,
            "external_api_calls_authorized": 0,
            "batch_authorizations_created": 0,
            "automatic_next_batch_authorized": False,
            "restart_resume_retry_replacement_requeue_authorized": False,
            "source_namespaces_read_only": True,
        }
    ):
        raise ValueError("recovery-continuation CPU-stage policy differs")
    require_clean(repo, "recovery-continuation")
    if (
        git_commit(repo) != repository.get("commit")
        or git_branch(repo) != EXPECTED_BRANCH
        or body.get("implementation_sha256") != implementation_bindings(repo)
    ):
        raise ValueError("recovery-continuation implementation binding differs")
    plan_binding = body.get("continuation_plan")
    if not isinstance(plan_binding, dict):
        raise ValueError("continuation plan binding is absent")
    plan, plan_body, context = load_and_verify_plan(
        plan_binding.get("path", ""), audit_source=audit_source
    )
    if (
        plan_binding != binding(plan_binding["path"], plan)
        or path != Path(plan_body["output_root"]) / "control" / STAGE_NAME
    ):
        raise ValueError("recovery-continuation stage path differs")
    return payload, body, plan, plan_body, context


def _load_workflow(output_root, repo_root, *, audit_source=True):
    output = Path(output_root).resolve()
    stage_payload, stage_body, plan, plan_body, context = load_and_verify_stage(
        output / "control" / STAGE_NAME,
        repo_root,
        audit_source=audit_source,
    )
    return output, plan, plan_body, stage_payload, stage_body, context


def preflight(args):
    _require_no_api_key()
    _, plan, plan_body, _, _, context = _load_workflow(
        args.output_root, args.repo_root, audit_source=True
    )
    original_plan = context["source_plan"]
    original_body = context["source_body"]
    original_runtime._preflight(original_plan, original_body, args.batch_index)
    print(
        json.dumps(
            {
                "status": "KALAI_S1_RECOVERY_CONTINUATION_BATCH_PREFLIGHT_VALID",
                "batch_id": batch_id(args.batch_index),
                "continuation_plan_payload_sha256": plan[SEAL_FIELD],
                "recovered_batch_1_bound": True,
                "reference_models_loaded": 0,
                "gpu_jobs": 0,
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


def self_test():
    assert BATCH_INDICES == tuple(range(2, 8))
    assert current_exposure(2) == Decimal("7.02198425")
    assert maximum_exposure(2) == Decimal("7.92198425")
    assert maximum_exposure(7) == FULL_COMPLETION_MAXIMUM_USD
    assert FULL_COMPLETION_MAXIMUM_USD < PROGRAM_CEILING_USD
    assert sum(value["row_count"] for value in EXPECTED_BATCH_SUMMARIES.values()) == 149
    assert sum(value["attempts"] for value in EXPECTED_BATCH_SUMMARIES.values()) == 2527
    print("MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-output-root", required=True)
    common.add_argument("--source-repo-root", required=True)
    common.add_argument("--recovery-output-root", required=True)
    common.add_argument("--recovery-repo-root", required=True)
    common.add_argument("--output-root", required=True)
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    del prepare_parser
    stage_parser = subparsers.add_parser("stage", parents=[common])
    stage_parser.add_argument("--repo-root", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output-root", required=True)
    preflight_parser.add_argument("--repo-root", required=True)
    preflight_parser.add_argument(
        "--batch-index", required=True, type=int, choices=BATCH_INDICES
    )
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test()
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "stage":
        stage(args)
    elif args.command == "preflight":
        preflight(args)
    else:
        raise RuntimeError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
