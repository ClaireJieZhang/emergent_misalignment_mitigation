#!/usr/bin/env python3
"""CPU-only control plane for the add-only judge-recovery-v6 overlay."""

import argparse
import builtins
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess


def _private_v5_control(name):
    path = Path(__file__).with_name(
        "audit_massive_medical_union_composition_exploratory_sequential_"
        "confirmation_v1_judge_recovery_v5.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load private v5 control implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _private_v5_judge(name):
    path = Path(__file__).with_name(
        "judge_massive_medical_union_composition_exploratory_sequential_"
        "confirmation_v1_judge_recovery_v5.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load private v5 judge implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _private_v5_control("_mmu_judge_recovery_v6_private_v5_control")
control = base.control
prior_control = _private_v5_control("_mmu_judge_recovery_v6_pristine_v5_control")
prior_judge = _private_v5_judge("_mmu_judge_recovery_v6_pristine_v5_judge")
prior_control.control.judge_module = lambda: prior_judge

RECOVERY_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1_"
    "judge_recovery_v6"
)
SOURCE_PROTOCOL_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
)
SOURCE_COMMIT = "3834f4215f6606ad49e511620bedd49219ecc3df"
SOURCE_TREE = "fb43b11e2be4030dace1e3a1e57c795b61a86566"
PRIOR_V5_BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-recovery-v5"
)
BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-recovery-v6"
)
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
PRIOR_REPO = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v5"
)
PRIOR_RECOVERY_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
)
REPO_ROOT = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v6"
)
RECOVERY_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v6"
)
CONTROL_ROOT = RECOVERY_OUTPUT / "control"
MEDICAL_ROOT = RECOVERY_OUTPUT / "evaluation/medical"
FINAL_ROOT = RECOVERY_OUTPUT / "evaluation/final"
LOG_ROOT = RECOVERY_OUTPUT / "logs"
MANIFEST_FILE = CONTROL_ROOT / "JUDGE_RECOVERY_V6_MANIFEST.json"
PREP_FILE = CONTROL_ROOT / "PREP.json"
PREFLIGHT_FILE = CONTROL_ROOT / "CPU_PREFLIGHT.json"
STAGED_FILE = CONTROL_ROOT / "STAGED"

JUDGE_MODEL = "gpt-5-mini-2025-08-07"
PLAN_SHA256 = "93110f70dbebcb7031dcf8e7be0d1c15e925347af70e8105e78d6d50daa49140"
RUBRIC_SHA256 = "ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e"
SCHEMA_SHA256 = "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777"
VERIFIED_PROGRAM_ACTUAL_USD = 2.915186
CONSUMED_V3_AUTHORITY_CAP_USD = 0.75
CONSUMED_V4_CANARY_AUTHORITY_CAP_USD = 0.003072
V5_CANARY_ACTUAL_USD = 0.0001145
V5_CONTINUATION_AUTHORITY_CAP_USD = 0.746928
V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD = 0.003072
PRIOR_ACCOUNTED_EXPOSURE_USD = 0.7562585
CANARY_CAP_USD = 0.003072
CONTINUATION_CAP_USD = 0.743856
RECOVERY_TOTAL_CAP_USD = 0.746928
CONSERVATIVE_PROGRAM_MAX_USD = 4.4183725
PROGRAM_CEILING_USD = 5.0
PROGRAM_CEILING_GAP_USD = 0.5816275
PRIOR_ATTEMPTS_MIN = 3
PRIOR_ATTEMPTS_MAX = 4

IDEMPOTENCY_CONTRACT = {
    "version": "recovery_id_blind_id_sha256_v1",
    "derivation": "sha256(utf8(recovery_id + ':' + blind_id))",
    "recovery_id": RECOVERY_ID,
    "prior_reused_start_index": 0, "prior_reused_end_index_exclusive": 1,
    "canary_start_index": 1, "canary_end_index_exclusive": 2,
    "continuation_start_index": 2, "continuation_end_index_exclusive": 240,
    "derived_key_count": 240, "authorized_new_key_count": 239,
    "authorized_range_key_list_sha256": "55d4202fe021120385f5aa4f7c3946557e0944b746efd83c80e61e6978edecde",
    "authorized_range_indexed_key_list_sha256": "cafbc5b30bcf3a32c8b0ac6d10830f7bc4fdbfffce52494a309f57238d22a25a",
    "canary_key_list_sha256": "3314de4d387b8fb9c0de2666ffab3c4df92a9fb248583f454284765f4101251d",
    "canary_indexed_key_list_sha256": "5dbe187b31575033d42a4435b18ddc52dcb227ace847661ea74b8d509461f78b",
    "continuation_key_list_sha256": "478af1731db01e4c138d77446da17c4a7f90d21fc3238c5fc217dcf4ae7d2083",
    "continuation_indexed_key_list_sha256": "f5daea4e44f179ccb495be1f9689e7b4f60a116090a6a5d506cf925a91398c2d",
    "raw_key_persisted": False, "source_raw_blind_id_reused_as_key": False,
    "all_240_keys_unique": True, "v5_v6_full_key_intersection_count": 0,
    "derived_key_list_sha256": "efa02ba5652ded464a66fac395647449ae9db02a21826fdca2d48c2b95e3e8ca",
    "indexed_identity_key_list_sha256": "ccd0dc4f59ba47098a17683a08b567bd308e3402e93e47d42464e6931abd4342",
}

ADDED_FILES = (
    "scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py",
    "scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py",
    "scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py",
    "scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6_tillicum.sh",
    "scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6_tillicum.sh",
    "scripts/derive_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6_tillicum.sh",
    "tests/test_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_judge_recovery_v6_control.py",
)

PRIOR_FILES = {
    "control/JUDGE_RECOVERY_V5_MANIFEST.json": (0o600, 15420, "ac9f5cdbf7c4e464e90b3951a5ec5d20deee868119e6dab26841a33b3ac60cd6", "7341bd3fb9fc21ec86ec11ca6cba65f8d93eb02e6a96f460dfd30cdba608178d"),
    "control/PREP.json": (0o600, 952, "5bc00af1db7acd4beba1727d103dc46395cdc7815dccb4d37acfa0d34f1ac8ba", "0893d7475188efed25cb17ef3546fc29dcb40963568a652ea907f9a750b3a10a"),
    "control/CPU_PREFLIGHT.json": (0o600, 1480, "9ba0595c9478c28760612a63ddb223b49f4dfa3ee79156767d2f879d1fbedf43", "e686f2a99f01353e2931a11dc11679f192969c8ed22693105be816a36df1906a"),
    "control/STAGED": (0o600, 1343, "b603b89bb988cce96f46baba761c4a89eacda2c2945a82d57b42b2ff8f5c3976", "37520426f6f6a28676ff6093e9ea43bf5cd81eff7252d314ce180da1e5a92b4b"),
    "control/CANARY_LOCK_OWNER.json": (0o400, 1062, "6437bf7073afd515fad6c38671a1b50241d8f49684269d1fd002c2e411bfcc86", "3409ae8fa9a67bc45af93286053c512fef9da1a6197c07a6cf65810e887e6268"),
    "control/CANARY_AUTHORIZATION.json": (0o400, 2963, "aef764cef7526dcad7418a8a9c43eefcb3539d596e9fdcd05fd189ecba30eff9", "15655bb2b6da73a7c146e81f4e6d1ea4bb8160012aee3509de9b78302d218b0d"),
    "control/CANARY_SUCCESS.json": (0o400, 2004, "01f6d6cd8690b3b392dcaebae685ce7773697b8aec099ef0808b34cb126c8730", "680b48ed97cf388045efa8c2381b6a5610b110aead526e47a8486f984dd95f19"),
    "control/CONTINUATION_LOCK_OWNER.json": (0o400, 1074, "bdc76c71a122bf61c29b0d423cb395cdd042d5acf08e931522b0f21a190c0b72", "be7acc3eb55b972cae355eaae744aedf7b281ad6255c6e4c720d1c0cf5e78bbe"),
    "control/CONTINUATION_AUTHORIZATION.json": (0o400, 3881, "2f77ce38ebff9a3bd7b90ce2b2d972948f72a9a9e4116c1451140f4865f998fb", "67a35e40755b8bc2797cbf24ea96e2d833344be011534bde61cacc0e28d65869"),
    "control/CONTINUATION_FAILURE.json": (0o400, 2075, "6a25aa58f16f8b9f4b1f8c3cfca1aa25d9a47465471901e050e99cf604b3b4b3", "51c567f9a170ae14ee3b544507845777d68ed2c06777aa43a565be96fff03b64"),
    "evaluation/medical/judge_checkpoint.json.001": (0o400, 7747, "9c19fc0dce9885b24fa4d806725f60f4f55a49fa9d2e4f5fe3548a09b1a5198c", "9f1ca5cb8d1a9b4285764c1831deb0eb336eb414fd827796775ca807b7f09686"),
    "logs/external_judge_canary.log": (0o400, 171, "fd6a187ac99408342c17e89675804e9f34ce26ae603c43a0d60820967f9bf7c0", None),
    "logs/external_judge_continuation.log": (0o400, 66, "e0019cb1bcc7edb614f6f09b619132c60bc0589fd4f052cf3f2234a3a0551de4", None),
}

_real_print = builtins.print


def _v6_print(*values, **kwargs):
    _real_print(*(v.replace("JUDGE_RECOVERY_V4", "JUDGE_RECOVERY_V6").replace(
        "JUDGE_RECOVERY_V5", "JUDGE_RECOVERY_V6") if isinstance(v, str) else v
        for v in values), **kwargs)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", os.fspath(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _record(path, require_seal=True):
    payload = control.load_json(path) if require_seal else None
    if payload is not None:
        control.audit_seal(payload, os.fspath(path))
    return control.binding(path, require_seal=require_seal)


def audit_repo():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("Recovery-v6 repository is absent or unsafe")
    commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    if (
        _git(REPO_ROOT, "branch", "--show-current") != BRANCH
        or _git(REPO_ROOT, "status", "--porcelain")
        or _git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit)
        != f"{commit} {SOURCE_COMMIT}"
        or _git(REPO_ROOT, "rev-parse", f"{SOURCE_COMMIT}^{{tree}}") != SOURCE_TREE
    ):
        raise ValueError("Recovery-v6 repository lineage differs")
    observed = []
    for line in _git(REPO_ROOT, "diff", "--name-status", "--no-renames", f"{SOURCE_COMMIT}..{commit}").splitlines():
        if line:
            observed.append(tuple(line.split("\t")))
    if sorted(observed) != sorted(("A", path) for path in ADDED_FILES):
        raise ValueError("Recovery-v6 repository add-only scope differs")
    for relative in ADDED_FILES:
        entry = _git(REPO_ROOT, "ls-files", "-s", "--", relative).split()
        mode = "100755" if relative.startswith("scripts/") else "100644"
        if len(entry) != 4 or entry[0] != mode or entry[2] != "0":
            raise ValueError(f"Recovery-v6 index mode differs: {relative}")
    return {
        "path": os.fspath(REPO_ROOT), "branch": BRANCH, "commit": commit,
        "tree": _git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
        "source_commit": SOURCE_COMMIT, "source_commit_is_direct_parent": True,
        "add_only_files": list(ADDED_FILES),
    }


def _audit_prior_stage_artifacts(records):
    manifest_payload = control.load_json(
        PRIOR_RECOVERY_OUTPUT / "control/JUDGE_RECOVERY_V5_MANIFEST.json"
    )
    manifest_body = control.audit_seal(manifest_payload, "prior v5 manifest")
    manifest_binding = records["control/JUDGE_RECOVERY_V5_MANIFEST.json"]
    for relative, label in (
        ("control/PREP.json", "prior v5 prep"),
        ("control/CPU_PREFLIGHT.json", "prior v5 CPU preflight"),
        ("control/STAGED", "prior v5 staged sentinel"),
    ):
        body = control.audit_seal(control.load_json(PRIOR_RECOVERY_OUTPUT / relative), label)
        if body.get("recovery_manifest") != manifest_binding:
            raise ValueError(f"{label} manifest binding differs")
        if body.get("external_api_calls", 0) != 0 or body.get("gpu_jobs", 0) != 0:
            raise ValueError(f"{label} unexpectedly records staged API/GPU work")
    return manifest_body


def audit_source():
    if PRIOR_REPO.is_symlink() or not PRIOR_REPO.is_dir():
        raise ValueError("Prior v5 repository is absent or unsafe")
    if (
        _git(PRIOR_REPO, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(PRIOR_REPO, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(PRIOR_REPO, "rev-parse", "HEAD^")
        != "5f7357fee6654cccb7918d307963dcfe5fa73418"
        or _git(PRIOR_REPO, "branch", "--show-current") != PRIOR_V5_BRANCH
        or _git(PRIOR_REPO, "status", "--porcelain")
    ):
        raise ValueError("Prior v5 repository binding differs")
    expected_dirs = {
        ".": {"control", "evaluation", "logs"},
        "control": {
            "JUDGE_RECOVERY_V5_MANIFEST.json", "PREP.json", "CPU_PREFLIGHT.json",
            "STAGED", "CANARY_LOCK_OWNER.json", "CANARY_AUTHORIZATION.json",
            "CANARY_SUCCESS.json", "CONTINUATION_LOCK_OWNER.json",
            "CONTINUATION_AUTHORIZATION.json", "CONTINUATION_FAILURE.json",
        },
        "evaluation": {"medical", "final"},
        "evaluation/medical": {"judge_checkpoint.json.001"},
        "evaluation/final": set(),
        "logs": {"external_judge_canary.log", "external_judge_continuation.log"},
    }
    for relative, names in expected_dirs.items():
        directory = PRIOR_RECOVERY_OUTPUT / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Prior v5 directory is absent or unsafe: {relative}")
        if stat.S_IMODE(directory.stat().st_mode) != 0o2700:
            raise ValueError(f"Prior v5 directory mode differs: {relative}")
        if {item.name for item in directory.iterdir()} != names:
            raise ValueError(f"Prior v5 exact terminal inventory differs: {relative}")
        if any(item.is_symlink() for item in directory.iterdir()):
            raise ValueError(f"Prior v5 inventory contains a symlink: {relative}")
    records = {}
    for relative, (mode, size, digest, payload_digest) in PRIOR_FILES.items():
        path = PRIOR_RECOVERY_OUTPUT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Prior v5 file is absent or unsafe: {relative}")
        if (
            stat.S_IMODE(path.stat().st_mode) != mode
            or path.stat().st_size != size
            or control.sha256_file(path) != digest
        ):
            raise ValueError(f"Prior v5 exact file binding differs: {relative}")
        record = _record(path, payload_digest is not None)
        if payload_digest is not None and record.get("payload_sha256") != payload_digest:
            raise ValueError(f"Prior v5 payload binding differs: {relative}")
        records[relative] = record

    prior_manifest_body = _audit_prior_stage_artifacts(records)
    # These pristine v5 checks reconstruct the original plan and validate both
    # locks, both authorizations, the successful row/checkpoint, and failure.
    prior_manifest = prior_control.control.load_manifest()
    prior_recovery = prior_judge.load_recovery_manifest(
        PRIOR_RECOVERY_OUTPUT / "control/JUDGE_RECOVERY_V5_MANIFEST.json"
    )
    prior_inputs = prior_judge.validate_source_inputs(prior_recovery)
    paths = prior_judge.recovery_paths(prior_recovery)
    for stage in ("canary", "continuation"):
        prior_control.control.audit_lock(stage, prior_manifest)
        prior_judge.load_authorization(prior_recovery, stage, paths)
    canary = prior_judge.load_canary_success(prior_recovery, prior_inputs, paths)
    if (
        canary["body"].get("actual_estimated_cost_usd") != V5_CANARY_ACTUAL_USD
        or canary["checkpoint"]["body"].get("completed_calls") != 1
    ):
        raise ValueError("Prior v5 successful canary differs")
    failure = prior_control.control.audit_failure("continuation")["body"]
    if (
        failure.get("stage") != "continuation"
        or failure.get("operation_stage") != "api_call"
        or failure.get("planned_index") != 1
        or failure.get("previously_completed_calls") != 1
        or failure.get("attempted_call_invocations_min") != 1
        or failure.get("attempted_call_invocations_max") != 1
        or failure.get("exception_class") != "AuthenticationError"
        or failure.get("http_status") != 401
        or failure.get("error_code") != "invalid_api_key"
        or failure.get("request_id") != "req_1f02cacd317e4b44bcef4903aba0104e"
        or failure.get("api_response_id") is not None
        or failure.get("api_response_model") is not None
        or failure.get("terminal") is not True
        or failure.get("retry_authorized") is not False
        or failure.get("restart_or_resume_authorized") is not False
        or failure.get("model_fallback_used") is not False
        or failure.get("contains_question_or_response_text") is not False
        or failure.get("contains_api_key_or_headers") is not False
    ):
        raise ValueError("Prior v5 terminal authentication failure differs")

    failed = {
        "recovery_id": SOURCE_PROTOCOL_ID + "_judge_recovery_v5",
        "repo": {
            "path": os.fspath(PRIOR_REPO), "branch": PRIOR_V5_BRANCH,
            "commit": SOURCE_COMMIT, "tree": SOURCE_TREE,
        },
        "manifest": records["control/JUDGE_RECOVERY_V5_MANIFEST.json"],
        "prep": records["control/PREP.json"],
        "cpu_preflight": records["control/CPU_PREFLIGHT.json"],
        "staged": records["control/STAGED"],
        "canary_lock_owner": records["control/CANARY_LOCK_OWNER.json"],
        "canary_authorization": records["control/CANARY_AUTHORIZATION.json"],
        "canary_success": records["control/CANARY_SUCCESS.json"],
        "continuation_lock_owner": records["control/CONTINUATION_LOCK_OWNER.json"],
        "continuation_authorization": records["control/CONTINUATION_AUTHORIZATION.json"],
        "continuation_failure": records["control/CONTINUATION_FAILURE.json"],
        "checkpoint_001": records["evaluation/medical/judge_checkpoint.json.001"],
        "canary_log": records["logs/external_judge_canary.log"],
        "continuation_log": records["logs/external_judge_continuation.log"],
        "terminal_contract": {
            "stage": "continuation", "planned_index": 1,
            "previously_completed_calls": 1, "accepted_judgments": 1,
            "checkpoint_completed_calls": 1,
            "attempted_call_invocations_min": 1,
            "attempted_call_invocations_max": 1,
            "operation_stage": "api_call", "exception_class": "AuthenticationError",
            "http_status": 401, "error_code": "invalid_api_key",
            "request_id": "req_1f02cacd317e4b44bcef4903aba0104e",
            "api_response_id": None, "api_response_model": None,
            "terminal": True, "retry_authorized": False,
            "restart_or_resume_authorized": False, "model_fallback_used": False,
            "contains_question_or_response_text": False,
            "contains_api_key_or_headers": False,
        },
        "authority_accounting": {
            "v3_external_judge_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
            "v4_canary_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
            "v5_canary_actual_usd": V5_CANARY_ACTUAL_USD,
            "v5_continuation_authority_cap_usd": V5_CONTINUATION_AUTHORITY_CAP_USD,
            "v5_continuation_authority_consumed_nonreusable": True,
            "v5_continuation_exact_attempted_calls": 1,
            "v5_continuation_accepted_judgments": 0,
            "v5_failed_request_actual_billing_known": False,
            "v5_failed_continuation_exact_attempt_exposure_cap_usd": V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD,
            "v5_unattempted_authority_counted_as_cost_exposure": False,
            "v5_unattempted_authority_reused_as_authority": False,
            "prior_accounted_exposure_usd": PRIOR_ACCOUNTED_EXPOSURE_USD,
            "historical_attempts_min": PRIOR_ATTEMPTS_MIN,
            "historical_attempts_max": PRIOR_ATTEMPTS_MAX,
            "historical_accepted_judgments": 1,
        },
    }
    return {"prior_manifest_body": prior_manifest_body, "failed_recovery_v5": failed}


def manifest_body(repo, source_state):
    prior = source_state["prior_manifest_body"]
    science = {
        "judge_model": JUDGE_MODEL, "plan_sha256": PLAN_SHA256,
        "rubric_sha256": RUBRIC_SHA256, "response_schema_sha256": SCHEMA_SHA256,
        "accepted_judgments": 240, "prior_v5_reused_judgments": 1,
        "new_v6_api_calls": 239, "canary_calls": 1,
        "canary_start_index": 1, "canary_end_index_exclusive": 2,
        "continuation_calls": 238, "continuation_start_index": 2,
        "continuation_end_index_exclusive": 240, "sdk_max_retries": 0,
        "same_plan_order": True, "model_fallback_authorized": False,
        "historical_A_reused_not_rejudged": True,
        "idempotency_contract": IDEMPOTENCY_CONTRACT,
    }
    budget = {
        "verified_program_actual_before_unknown_api_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "consumed_v3_authority_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "consumed_v4_canary_authority_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "v5_canary_actual_usd": V5_CANARY_ACTUAL_USD,
        "v5_continuation_authority_cap_usd": V5_CONTINUATION_AUTHORITY_CAP_USD,
        "v5_continuation_authority_consumed_nonreusable": True,
        "v5_continuation_exact_attempted_calls": 1,
        "v5_continuation_accepted_judgments": 0,
        "v5_failed_request_actual_billing_known": False,
        "v5_failed_continuation_exact_attempt_exposure_cap_usd": V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD,
        "v5_unattempted_authority_counted_as_cost_exposure": False,
        "v5_unattempted_authority_reused_as_authority": False,
        "prior_accounted_exposure_usd": PRIOR_ACCOUNTED_EXPOSURE_USD,
        "historical_attempts_min": PRIOR_ATTEMPTS_MIN,
        "historical_attempts_max": PRIOR_ATTEMPTS_MAX,
        "historical_accepted_judgments": 1,
        "planned_v6_canary_cap_usd": CANARY_CAP_USD,
        "planned_v6_continuation_cap_usd": CONTINUATION_CAP_USD,
        "planned_v6_total_cap_usd": RECOVERY_TOTAL_CAP_USD,
        "conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "program_ceiling_usd": PROGRAM_CEILING_USD, "within_program_ceiling": True,
        "remaining_ceiling_gap_usd": PROGRAM_CEILING_GAP_USD,
    }
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + "_manifest_v1",
        "recovery_id": RECOVERY_ID, "source_protocol_id": SOURCE_PROTOCOL_ID,
        "recovery_repo": repo, "source_output_root": prior["source_output_root"],
        "recovery_output_root": os.fspath(RECOVERY_OUTPUT),
        "source_failure": prior["source_failure"],
        "source_protocol_manifest": prior["source_protocol_manifest"],
        "source_judge_plan": prior["source_judge_plan"],
        "source_artifacts": prior["source_artifacts"],
        "scientific_contract": science, "budget_contract": budget,
        "failed_recovery_v4": prior["failed_recovery_v4"],
        "failed_recovery_v5": source_state["failed_recovery_v5"],
        "external_api_authorized": False, "gpu_authorized": False,
        "cpu_stage_only": True,
    }


def expected_staged_files():
    return {
        "control": {"JUDGE_RECOVERY_V6_MANIFEST.json", "PREP.json", "CPU_PREFLIGHT.json", "STAGED"},
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    }


def seal_staged_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU staging")
    manifest = control.load_manifest()
    if os.path.lexists(PREFLIGHT_FILE) or os.path.lexists(STAGED_FILE):
        raise FileExistsError("CPU staging is already sealed")
    control.audit_namespace({
        "control": {"JUDGE_RECOVERY_V6_MANIFEST.json", "PREP.json"},
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    })
    commands = args.validation_command or []
    if not commands:
        raise ValueError("At least one successful validation command is required")
    preflight = control.sealed({
        "schema_version": 1, "protocol": RECOVERY_ID + "_cpu_preflight_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": control.binding(MANIFEST_FILE, require_seal=True),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "validation_commands_passed": commands,
        "network_validation": "local_mock_transport_only",
        "external_api_calls": 0, "gpu_jobs": 0, "api_key_required": False,
        "status": "CPU_VALIDATED_AWAITING_SEPARATE_CANARY_AUTHORIZATION",
    })
    control.atomic_json(PREFLIGHT_FILE, preflight)
    control.atomic_json(STAGED_FILE, control.sealed({
        "schema_version": 1, "protocol": RECOVERY_ID + "_staged_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": control.binding(MANIFEST_FILE, require_seal=True),
        "cpu_preflight": control.binding(PREFLIGHT_FILE, require_seal=True),
        "external_api_authorized": False, "external_api_calls": 0,
        "gpu_authorized": False, "gpu_jobs": 0,
        "next_stage": "SEPARATELY_AUTHORIZED_ONE_CALL_INDEX_1_CANARY",
    }))
    control.audit_staged()
    _v6_print(json.dumps({
        "status": "JUDGE_RECOVERY_V6_CPU_STAGED", "external_api_calls": 0,
        "gpu_jobs": 0, "next_stage": "SEPARATELY_AUTHORIZED_ONE_CALL_INDEX_1_CANARY",
    }, sort_keys=True))
    return 0


def judge_module():
    import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6 as judge
    return judge


def _float_equal(left, right):
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def require_authorization_acknowledgments(args, stage, canary_actual=None):
    expected = {
        "ack_verified_program_actual_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "ack_consumed_v3_authority_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "ack_consumed_v4_canary_authority_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "ack_v5_canary_actual_usd": V5_CANARY_ACTUAL_USD,
        "ack_v5_continuation_authority_cap_usd": V5_CONTINUATION_AUTHORITY_CAP_USD,
        "ack_v5_failed_continuation_exposure_cap_usd": V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD,
        "ack_prior_accounted_exposure_usd": PRIOR_ACCOUNTED_EXPOSURE_USD,
        "ack_prior_network_attempts_min": PRIOR_ATTEMPTS_MIN,
        "ack_prior_network_attempts_max": PRIOR_ATTEMPTS_MAX,
        "ack_prior_accepted_judgments": 1,
        "ack_max_cost_usd": CANARY_CAP_USD if stage == "canary" else CONTINUATION_CAP_USD,
        "ack_new_v6_total_cap_usd": RECOVERY_TOTAL_CAP_USD,
        "ack_conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "ack_program_ceiling_usd": PROGRAM_CEILING_USD,
        "ack_remaining_ceiling_gap_usd": PROGRAM_CEILING_GAP_USD,
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        valid = observed == wanted if isinstance(wanted, int) else _float_equal(observed, wanted)
        if not valid:
            raise ValueError(f"External authorization acknowledgment differs: {name}")
    flags = (
        "ack_v5_continuation_authority_consumed_nonreusable",
        "ack_v5_failed_request_actual_billing_unknown",
        "ack_v5_unattempted_authority_not_cost_exposure",
        "ack_v5_unattempted_authority_not_reused",
        "ack_prior_authorities_consumed_not_reused",
    )
    if any(getattr(args, name) is not True for name in flags):
        raise ValueError("V5 exact-attempt/nonreuse accounting was not acknowledged")
    if stage == "continuation" and (
        canary_actual is None
        or not _float_equal(args.ack_canary_actual_estimated_cost_usd, canary_actual)
    ):
        raise ValueError("V6 canary actual cost acknowledgment differs")


def add_authorization_arguments(parser):
    parser.add_argument("--stage", required=True, choices=("canary", "continuation"))
    for name, kind in (
        ("verified-program-actual-usd", float),
        ("consumed-v3-authority-cap-usd", float),
        ("consumed-v4-canary-authority-cap-usd", float),
        ("v5-canary-actual-usd", float),
        ("v5-continuation-authority-cap-usd", float),
        ("v5-failed-continuation-exposure-cap-usd", float),
        ("prior-accounted-exposure-usd", float),
        ("prior-network-attempts-min", int),
        ("prior-network-attempts-max", int),
        ("prior-accepted-judgments", int),
        ("max-cost-usd", float), ("new-v6-total-cap-usd", float),
        ("conservative-program-max-usd", float), ("program-ceiling-usd", float),
        ("remaining-ceiling-gap-usd", float),
    ):
        parser.add_argument("--ack-" + name, type=kind, required=True)
    for name in (
        "v5-continuation-authority-consumed-nonreusable",
        "v5-failed-request-actual-billing-unknown",
        "v5-unattempted-authority-not-cost-exposure",
        "v5-unattempted-authority-not-reused",
        "prior-authorities-consumed-not-reused",
    ):
        parser.add_argument("--ack-" + name, action="store_true")
    parser.add_argument("--ack-canary-actual-estimated-cost-usd", type=float)


def _audit_staged_seals_after_lock():
    manifest = control.load_manifest()
    prep_payload = control.load_json(PREP_FILE)
    prep = control.audit_seal(prep_payload, PREP_FILE)
    preflight_payload = control.load_json(PREFLIGHT_FILE)
    preflight = control.audit_seal(preflight_payload, PREFLIGHT_FILE)
    staged_payload = control.load_json(STAGED_FILE)
    staged = control.audit_seal(staged_payload, STAGED_FILE)
    manifest_binding = control.binding(MANIFEST_FILE, require_seal=True)
    preflight_binding = control.binding(PREFLIGHT_FILE, require_seal=True)
    if (
        prep != {
            "schema_version": 1, "protocol": RECOVERY_ID + "_prep_v1",
            "recovery_id": RECOVERY_ID,
            "recovery_manifest": manifest_binding,
            "source_plan_sha256": PLAN_SHA256,
            "external_api_calls": 0, "gpu_jobs": 0,
            "status": "CPU_PREPARED_AWAITING_VALIDATION",
        }
        or set(preflight) != {
            "schema_version", "protocol", "recovery_id", "recovery_manifest",
            "manifest_payload_sha256", "validation_commands_passed",
            "network_validation", "external_api_calls", "gpu_jobs",
            "api_key_required", "status",
        }
        or preflight.get("protocol") != RECOVERY_ID + "_cpu_preflight_v1"
        or preflight.get("recovery_id") != RECOVERY_ID
        or preflight.get("recovery_manifest") != manifest_binding
        or preflight.get("manifest_payload_sha256") != manifest["payload_sha256"]
        or preflight.get("network_validation") != "local_mock_transport_only"
        or not isinstance(preflight.get("validation_commands_passed"), list)
        or not preflight["validation_commands_passed"]
        or preflight.get("external_api_calls") != 0
        or preflight.get("gpu_jobs") != 0
        or preflight.get("api_key_required") is not False
        or preflight.get("status")
        != "CPU_VALIDATED_AWAITING_SEPARATE_CANARY_AUTHORIZATION"
        or set(staged) != {
            "schema_version", "protocol", "recovery_id", "recovery_manifest",
            "cpu_preflight", "external_api_authorized", "external_api_calls",
            "gpu_authorized", "gpu_jobs", "next_stage",
        }
        or staged.get("protocol") != RECOVERY_ID + "_staged_v1"
        or staged.get("recovery_id") != RECOVERY_ID
        or staged.get("recovery_manifest") != manifest_binding
        or staged.get("cpu_preflight") != preflight_binding
        or staged.get("external_api_authorized") is not False
        or staged.get("external_api_calls") != 0
        or staged.get("gpu_authorized") is not False
        or staged.get("gpu_jobs") != 0
        or staged.get("next_stage")
        != "SEPARATELY_AUTHORIZED_ONE_CALL_INDEX_1_CANARY"
        or any(
            stat.S_IMODE(path.stat().st_mode) != 0o600
            for path in (MANIFEST_FILE, PREP_FILE, PREFLIGHT_FILE, STAGED_FILE)
        )
    ):
        raise ValueError("V6 staged seals differ after permanent lock creation")
    return manifest


def _authority_state(stage, require_staged_only=False):
    if stage == "canary":
        manifest = (
            control.audit_staged() if require_staged_only
            else _audit_staged_seals_after_lock()
        )
    else:
        manifest = control.load_manifest()
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    if os.path.lexists(paths["checkpoint_base"]) or os.path.lexists(judge.checkpoint_path(paths, 1)):
        raise ValueError("V6 must reference, never copy, v5 checkpoint .001")
    canary_actual = None
    if stage == "canary":
        names = (
            "canary_authorization", "canary_failure", "canary_success",
            "continuation_authorization", "continuation_failure",
            "continuation_success", "judgments",
        )
        checkpoints = range(2, 241)
    else:
        if os.path.lexists(paths["canary_failure"]):
            raise ValueError("Failed v6 canary cannot authorize continuation")
        canary = judge.load_canary_success(recovery, inputs, paths)
        canary_actual = canary["body"]["stage_actual_estimated_cost_usd"]
        names = ("continuation_authorization", "continuation_failure", "continuation_success", "judgments")
        checkpoints = range(3, 241)
    if any(os.path.lexists(paths[name]) for name in names):
        raise ValueError(f"{stage} authority namespace is not fresh")
    if any(os.path.lexists(judge.checkpoint_path(paths, item)) for item in checkpoints):
        raise ValueError(f"{stage} checkpoint namespace is not fresh")
    return manifest, judge, recovery, inputs, paths, canary_actual


def authorization_preflight(args):
    if os.path.lexists(control.lock_path(args.stage)):
        raise FileExistsError(f"{args.stage} permanent lock namespace is not fresh")
    state = _authority_state(
        args.stage, require_staged_only=args.stage == "canary"
    )
    require_authorization_acknowledgments(args, args.stage, state[-1])
    return state[0]


def write_authorization_command(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be loaded before external authorization")
    manifest, judge, recovery, _inputs, paths, canary_actual = _authority_state(args.stage)
    control.audit_lock(args.stage, manifest, args.owner_token)
    require_authorization_acknowledgments(args, args.stage, canary_actual)
    payload = control.sealed(judge.authorization_body(recovery, args.stage, paths))
    control.atomic_json(Path(paths[f"{args.stage}_authorization"]), payload, mode=0o400)
    judge.load_authorization(recovery, args.stage, paths)
    _v6_print(json.dumps({
        "status": f"JUDGE_RECOVERY_V6_{args.stage.upper()}_AUTHORIZED",
        "stage": args.stage, "authorized_calls": 1 if args.stage == "canary" else 238,
        "max_cost_usd": CANARY_CAP_USD if args.stage == "canary" else CONTINUATION_CAP_USD,
        "external_api_calls": 0, "authorization_payload_sha256": payload["payload_sha256"],
    }, sort_keys=True))
    return 0


def wrapper_failure_body(stage, manifest, recovery, paths, exit_code, owner_token):
    judge = judge_module()
    completed = judge._completed_checkpoint_count(paths)
    stage_prior = 1 if stage == "canary" else 2
    accepted = max(0, completed - stage_prior)
    maximum = 1 if stage == "canary" else 238
    authorization_path = paths[f"{stage}_authorization"]
    authorization = None
    if os.path.lexists(authorization_path):
        authorization = judge._authorization_binding(
            judge.load_authorization(recovery, stage, paths)
        )
    log_path = LOG_ROOT / f"external_judge_{stage}.log"
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + f"_{stage}_wrapper_failure_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": control.binding(MANIFEST_FILE, require_seal=True),
        "stage": stage, "stage_lock": control.audit_lock(stage, manifest, owner_token),
        "stage_authorization": authorization, "wrapper_exit_code": exit_code,
        "previously_completed_calls": completed,
        "attempted_call_invocations_min": accepted,
        "attempted_call_invocations_max": min(maximum, accepted + 1),
        "durable_log": control.binding(log_path) if os.path.lexists(log_path) else None,
        "failure_recorded_at": judge.utc_now(),
        "contains_question_or_response_text": False,
        "contains_api_key_or_headers": False, "model_fallback_used": False,
        "retry_authorized": False, "restart_or_resume_authorized": False,
        "terminal": True,
    }


def _audit_failure_checkpoints(judge, recovery, inputs, paths, stage, authorization):
    if os.path.lexists(paths["checkpoint_base"]) or os.path.lexists(judge.checkpoint_path(paths, 1)):
        raise ValueError("V6 failure namespace must not contain local checkpoint .001")
    completed = judge._completed_checkpoint_count(paths)
    expected = {
        f"judge_checkpoint.json.{index:03d}" for index in range(2, completed + 1)
    }
    observed = {
        name for name in os.listdir(paths["medical"])
        if name.startswith("judge_checkpoint.json.")
    }
    if observed != expected:
        raise ValueError("V6 failure checkpoint inventory differs")
    previous = [inputs["prior_v5"]["judgment"]]
    canary = None
    if completed >= 2:
        canary_auth = authorization if stage == "canary" else judge.load_authorization(
            recovery, "canary", paths
        )
        checkpoint = judge.audit_checkpoint(
            judge.checkpoint_path(paths, 2), recovery, inputs, "canary", canary_auth, 2
        )
        if checkpoint["body"]["judgments"][:1] != previous:
            raise ValueError("V6 failure lost its exact imported v5 prefix")
        previous = list(checkpoint["body"]["judgments"])
    if stage == "canary":
        if completed not in {1, 2}:
            raise ValueError("V6 canary failure completion differs")
    else:
        if completed < 2:
            raise ValueError("V6 continuation failure lacks successful canary")
        canary = judge.load_canary_success(recovery, inputs, paths)
        if canary["checkpoint"]["body"]["judgments"] != previous:
            raise ValueError("V6 failure canary checkpoint differs")
        for index in range(3, completed + 1):
            checkpoint = judge.audit_checkpoint(
                judge.checkpoint_path(paths, index), recovery, inputs,
                "continuation", authorization, index,
            )
            if checkpoint["body"]["judgments"][:-1] != previous:
                raise ValueError("V6 failure cumulative checkpoint prefix differs")
            previous = list(checkpoint["body"]["judgments"])
    response_ids = [row["api_response_id"] for row in previous]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("V6 failure checkpoint reuses an API response ID")
    for index in range(2, completed + 1):
        if stat.S_IMODE(os.stat(judge.checkpoint_path(paths, index)).st_mode) != 0o400:
            raise ValueError("V6 failure checkpoint mode differs")
    return completed, previous, canary


def audit_failure(stage):
    manifest = control.load_manifest()
    control.audit_lock(stage, manifest)
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    failure_path = Path(paths[f"{stage}_failure"])
    if os.path.lexists(paths[f"{stage}_success"]):
        raise ValueError("V6 failure and success artifacts cannot coexist")
    payload = control.load_json(failure_path)
    body = control.audit_seal(payload, f"v6 {stage} failure")
    if stat.S_IMODE(failure_path.stat().st_mode) != 0o400:
        raise ValueError("V6 failure artifact mode differs")
    authorization_path = paths[f"{stage}_authorization"]
    authorization = None
    expected_authorization = None
    if os.path.lexists(authorization_path):
        authorization = judge.load_authorization(recovery, stage, paths)
        expected_authorization = judge._authorization_binding(authorization)
    completed, judgments, canary = _audit_failure_checkpoints(
        judge, recovery, inputs, paths, stage, authorization
    )
    if os.path.lexists(paths["judgments"]):
        if stage != "continuation" or completed != 240 or authorization is None:
            raise ValueError("V6 failure has unexpected terminal judgments")
        terminal_payload = judge.load_json(paths["judgments"])
        terminal_body = judge.audit_seal(terminal_payload, paths["judgments"])
        stage_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in judgments[2:])
        total_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in judgments)
        expected_meta = {
            **judge.judge_meta(recovery, inputs),
            "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
            "canary_authorization": judge._authorization_binding(canary["authorization"]),
            "continuation_authorization": expected_authorization,
            "accepted_rows": 240, "prior_v5_reused_judgments": 1,
            "actual_api_calls": 239, "canary_api_calls": 1,
            "continuation_api_calls": 238,
            "actual_estimated_cost_usd": total_cost,
            "new_v6_estimated_cost_usd": canary["body"]["stage_actual_estimated_cost_usd"] + stage_cost,
        }
        expected_rows = sorted(
            judgments, key=lambda row: (row["model_name"], row["question_id"], row["sample_index"])
        )
        if terminal_body != {"meta": expected_meta, "judgments": expected_rows}:
            raise ValueError("V6 failure terminal judgments differ")
    stage_prior = 1 if stage == "canary" else 2
    maximum = 1 if stage == "canary" else 238
    accepted = max(0, completed - stage_prior)
    attempted_min = body.get("attempted_call_invocations_min")
    attempted_max = body.get("attempted_call_invocations_max")
    common = (
        body.get("schema_version") == 1 and body.get("recovery_id") == RECOVERY_ID
        and body.get("stage") == stage
        and body.get("stage_authorization") == expected_authorization
        and body.get("previously_completed_calls") == completed
        and isinstance(attempted_min, int) and not isinstance(attempted_min, bool)
        and isinstance(attempted_max, int) and not isinstance(attempted_max, bool)
        and accepted <= attempted_min <= attempted_max <= min(maximum, accepted + 1)
        and body.get("contains_question_or_response_text") is False
        and body.get("contains_api_key_or_headers") is False
        and body.get("model_fallback_used") is False
        and body.get("retry_authorized") is False
        and body.get("restart_or_resume_authorized") is False
        and body.get("terminal") is True
    )
    if not common:
        raise ValueError("V6 failure common contract differs")
    protocol = body.get("protocol")
    if protocol == RECOVERY_ID + f"_{stage}_failure_v1":
        stage_start = 1 if stage == "canary" else 2
        expected_planned = (
            stage_start + attempted_min - 1 if attempted_min > 0 else completed
        )
        keys = {
            "schema_version", "protocol", "recovery_id", "recovery_manifest", "stage",
            "stage_authorization", "operation_stage", "planned_index",
            "previously_completed_calls", "attempted_call_invocations_min",
            "attempted_call_invocations_max", "attempt_started_at", "failure_recorded_at",
            "exception_class", "http_status", "error_code", "request_id",
            "api_response_id", "api_response_model", "error_message_safe",
            "error_message_sha256", "contains_question_or_response_text",
            "contains_api_key_or_headers", "model_fallback_used", "retry_authorized",
            "restart_or_resume_authorized", "terminal",
        }
        if (
            set(body) != keys or expected_authorization is None
            or attempted_min != attempted_max
            or body.get("recovery_manifest")
            != judge.binding(recovery["path"], judge.load_json(recovery["path"]))
            or body.get("operation_stage") not in {
                "environment_preflight", "client_initialization", "api_call",
                "response_validation", "artifact_commit",
            }
            or body.get("planned_index") != expected_planned
            or not control.valid_utc_timestamp(body.get("attempt_started_at"))
            or not control.valid_utc_timestamp(body.get("failure_recorded_at"))
            or re.fullmatch(r"[0-9a-f]{64}", body.get("error_message_sha256", "")) is None
            or not isinstance(body.get("exception_class"), str)
            or judge._safe_external_token(body["exception_class"])
            != body["exception_class"]
            or any(
                value is not None and judge._safe_external_token(value) != value
                for value in (
                    body.get("error_code"), body.get("request_id"),
                    body.get("api_response_id"), body.get("api_response_model"),
                )
            )
            or not (
                body.get("http_status") is None
                or (
                    isinstance(body["http_status"], int)
                    and not isinstance(body["http_status"], bool)
                    and 100 <= body["http_status"] <= 599
                )
            )
            or not (
                body.get("error_message_safe") is None
                or (
                    isinstance(body["error_message_safe"], str)
                    and len(body["error_message_safe"]) <= 1000
                    and judge.safe_error_message(body["error_message_safe"])
                    == body["error_message_safe"]
                )
            )
        ):
            raise ValueError("V6 core failure exact-attempt contract differs")
    elif protocol == RECOVERY_ID + f"_{stage}_wrapper_failure_v1":
        exit_code = body.get("wrapper_exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int) or not 1 <= exit_code <= 255:
            raise ValueError("V6 wrapper failure exit code differs")
        expected = wrapper_failure_body(stage, manifest, recovery, paths, exit_code, None)
        expected["failure_recorded_at"] = body.get("failure_recorded_at")
        # audit_lock accepts no token and the rest of the body is deterministic.
        if body != expected or not control.valid_utc_timestamp(body.get("failure_recorded_at")):
            raise ValueError("V6 wrapper failure contract differs")
    else:
        raise ValueError("V6 failure protocol differs")
    return {"payload": payload, "body": body, "record": control.binding(failure_path, require_seal=True)}


def _configure_control():
    values = {
        "RECOVERY_ID": RECOVERY_ID, "SOURCE_PROTOCOL_ID": SOURCE_PROTOCOL_ID,
        "SOURCE_COMMIT": SOURCE_COMMIT, "BRANCH": BRANCH,
        "REPO_ROOT": REPO_ROOT, "RECOVERY_OUTPUT": RECOVERY_OUTPUT,
        "CONTROL_ROOT": CONTROL_ROOT, "MEDICAL_ROOT": MEDICAL_ROOT,
        "FINAL_ROOT": FINAL_ROOT, "LOG_ROOT": LOG_ROOT,
        "MANIFEST_FILE": MANIFEST_FILE, "PREP_FILE": PREP_FILE,
        "PREFLIGHT_FILE": PREFLIGHT_FILE, "STAGED_FILE": STAGED_FILE,
        "JUDGE_MODEL": JUDGE_MODEL, "PLAN_SHA256": PLAN_SHA256,
        "RUBRIC_SHA256": RUBRIC_SHA256, "SCHEMA_SHA256": SCHEMA_SHA256,
        "VERIFIED_PROGRAM_ACTUAL_USD": VERIFIED_PROGRAM_ACTUAL_USD,
        "PRIOR_FAILED_AUTHORITY_CAP_USD": PRIOR_ACCOUNTED_EXPOSURE_USD,
        "CANARY_CAP_USD": CANARY_CAP_USD, "CONTINUATION_CAP_USD": CONTINUATION_CAP_USD,
        "RECOVERY_TOTAL_CAP_USD": RECOVERY_TOTAL_CAP_USD,
        "CONSERVATIVE_PROGRAM_MAX_USD": CONSERVATIVE_PROGRAM_MAX_USD,
        "PROGRAM_CEILING_USD": PROGRAM_CEILING_USD,
    }
    for name, value in values.items():
        setattr(control, name, value)
    control.audit_repo = audit_repo
    control.audit_source = audit_source
    control.manifest_body = manifest_body
    control.expected_staged_files = expected_staged_files
    control.judge_module = judge_module
    control.require_authorization_acknowledgments = require_authorization_acknowledgments
    control.authorization_preflight = authorization_preflight
    control.wrapper_failure_body = wrapper_failure_body
    control.audit_failure = audit_failure
    control.print = _v6_print


_configure_control()


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(handler=control.prepare_command)
    stage = commands.add_parser("seal-staged")
    stage.add_argument("--validation-command", action="append")
    stage.set_defaults(handler=seal_staged_command)
    commands.add_parser("audit-staged").set_defaults(handler=control.audit_staged_command)
    preflight = commands.add_parser("preflight-authorization")
    add_authorization_arguments(preflight)
    preflight.set_defaults(handler=control.preflight_authorization_command)
    lock = commands.add_parser("acquire-lock")
    add_authorization_arguments(lock)
    lock.add_argument("--owner-token", required=True)
    lock.set_defaults(handler=control.acquire_lock_command)
    lock_audit = commands.add_parser("audit-lock")
    lock_audit.add_argument("--stage", required=True, choices=("canary", "continuation"))
    lock_audit.add_argument("--owner-token")
    lock_audit.set_defaults(handler=control.audit_lock_command)
    authorize = commands.add_parser("write-authorization")
    add_authorization_arguments(authorize)
    authorize.add_argument("--owner-token", required=True)
    authorize.set_defaults(handler=write_authorization_command)
    auth_audit = commands.add_parser("audit-authorization")
    auth_audit.add_argument("--stage", required=True, choices=("canary", "continuation"))
    auth_audit.set_defaults(handler=control.audit_authorization_command)
    wrapper = commands.add_parser("write-wrapper-failure")
    wrapper.add_argument("--stage", required=True, choices=("canary", "continuation"))
    wrapper.add_argument("--exit-code", required=True, type=int)
    wrapper.add_argument("--owner-token", required=True)
    wrapper.set_defaults(handler=control.write_wrapper_failure_command)
    failure = commands.add_parser("audit-failure")
    failure.add_argument("--stage", required=True, choices=("canary", "continuation"))
    failure.set_defaults(handler=control.audit_failure_command)
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


def main():
    try:
        raise SystemExit(run())
    except (ValueError, FileExistsError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
