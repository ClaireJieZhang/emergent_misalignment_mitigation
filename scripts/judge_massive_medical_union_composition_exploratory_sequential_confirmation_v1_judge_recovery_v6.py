#!/usr/bin/env python3
"""Fresh judge recovery v6, reusing exactly one sealed v5 judgment.

V6 never mutates or resumes the v5 namespace.  It imports the exact accepted
index-0 judgment from v5 checkpoint .001, canaries index 1, and (only under a
separate authority) judges indices 2..239.  Static validation is local-only.
"""

import argparse
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess


def _private_v5(name):
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


# One private instance is adapted to v6; the second remains pristine and
# audits the live v5 source checkpoint and authorization.
base = _private_v5("_mmu_judge_recovery_v6_private_v5")
prior_v5 = _private_v5("_mmu_judge_recovery_v6_pristine_v5")
inner = base.base
source = base.source

RECOVERY_ID = source.PROTOCOL_ID + "_judge_recovery_v6"
EXPECTED_PLAN_SHA256 = base.EXPECTED_PLAN_SHA256
EXPECTED_SOURCE_COMMIT = "3834f4215f6606ad49e511620bedd49219ecc3df"
PRIOR_RECOVERY_ID = source.PROTOCOL_ID + "_judge_recovery_v5"
PRIOR_ACCEPTED_CALLS = 1
CANARY_CALLS = 1
CONTINUATION_CALLS = 238
CANARY_START = 1
CANARY_END = 2
CONTINUATION_START = 2
TOTAL_CALLS = 240
NEW_API_CALLS = 239
CANARY_MAX_COST_USD = 0.003072
CONTINUATION_MAX_COST_USD = 0.743856
NEW_RECOVERY_CAP_USD = 0.746928
VERIFIED_PROGRAM_ACTUAL_USD = 2.915186
CONSUMED_V3_AUTHORITY_CAP_USD = 0.75
CONSUMED_V4_CANARY_AUTHORITY_CAP_USD = 0.003072
V5_CANARY_ACTUAL_USD = 0.0001145
CONSUMED_V5_CONTINUATION_AUTHORITY_CAP_USD = 0.746928
V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD = 0.003072
PRIOR_ACCOUNTED_EXPOSURE_USD = 0.7562585
CONSERVATIVE_PROGRAM_MAX_USD = 4.4183725
PROGRAM_CEILING_USD = 5.0
PROGRAM_CEILING_GAP_USD = 0.5816275
PRIOR_ATTEMPTS_MIN = 3
PRIOR_ATTEMPTS_MAX = 4
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
PRIOR_V5_REPO = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v5"
)
PRIOR_V5_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
)
PRIOR_V5_BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-recovery-v5"
)
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

# Exact bindings from the independently audited, sealed v5 terminal namespace.
V5_EXPECTED_TREE = "fb43b11e2be4030dace1e3a1e57c795b61a86566"
V5_MANIFEST = (15420, "ac9f5cdbf7c4e464e90b3951a5ec5d20deee868119e6dab26841a33b3ac60cd6", "7341bd3fb9fc21ec86ec11ca6cba65f8d93eb02e6a96f460dfd30cdba608178d")
V5_STAGED = (1343, "b603b89bb988cce96f46baba761c4a89eacda2c2945a82d57b42b2ff8f5c3976", "37520426f6f6a28676ff6093e9ea43bf5cd81eff7252d314ce180da1e5a92b4b")
V5_PREP = (952, "5bc00af1db7acd4beba1727d103dc46395cdc7815dccb4d37acfa0d34f1ac8ba", "0893d7475188efed25cb17ef3546fc29dcb40963568a652ea907f9a750b3a10a")
V5_CPU_PREFLIGHT = (1480, "9ba0595c9478c28760612a63ddb223b49f4dfa3ee79156767d2f879d1fbedf43", "e686f2a99f01353e2931a11dc11679f192969c8ed22693105be816a36df1906a")
V5_CANARY_LOCK = (1062, "6437bf7073afd515fad6c38671a1b50241d8f49684269d1fd002c2e411bfcc86", "3409ae8fa9a67bc45af93286053c512fef9da1a6197c07a6cf65810e887e6268")
V5_CANARY_AUTH = (2963, "aef764cef7526dcad7418a8a9c43eefcb3539d596e9fdcd05fd189ecba30eff9", "15655bb2b6da73a7c146e81f4e6d1ea4bb8160012aee3509de9b78302d218b0d")
V5_CANARY_SUCCESS = (2004, "01f6d6cd8690b3b392dcaebae685ce7773697b8aec099ef0808b34cb126c8730", "680b48ed97cf388045efa8c2381b6a5610b110aead526e47a8486f984dd95f19")
V5_CONTINUATION_LOCK = (1074, "bdc76c71a122bf61c29b0d423cb395cdd042d5acf08e931522b0f21a190c0b72", "be7acc3eb55b972cae355eaae744aedf7b281ad6255c6e4c720d1c0cf5e78bbe")
V5_CONTINUATION_AUTH = (3881, "2f77ce38ebff9a3bd7b90ce2b2d972948f72a9a9e4116c1451140f4865f998fb", "67a35e40755b8bc2797cbf24ea96e2d833344be011534bde61cacc0e28d65869")
V5_CONTINUATION_FAILURE = (2075, "6a25aa58f16f8b9f4b1f8c3cfca1aa25d9a47465471901e050e99cf604b3b4b3", "51c567f9a170ae14ee3b544507845777d68ed2c06777aa43a565be96fff03b64")
V5_CHECKPOINT_001 = (7747, "9c19fc0dce9885b24fa4d806725f60f4f55a49fa9d2e4f5fe3548a09b1a5198c", "9f1ca5cb8d1a9b4285764c1831deb0eb336eb414fd827796775ca807b7f09686")
V5_CANARY_LOG = (171, "fd6a187ac99408342c17e89675804e9f34ce26ae603c43a0d60820967f9bf7c0")
V5_CONTINUATION_LOG = (66, "e0019cb1bcc7edb614f6f09b619132c60bc0589fd4f052cf3f2234a3a0551de4")

IDEMPOTENCY_CONTRACT = {
    "version": "recovery_id_blind_id_sha256_v1",
    "derivation": "sha256(utf8(recovery_id + ':' + blind_id))",
    "recovery_id": RECOVERY_ID,
    "prior_reused_start_index": 0,
    "prior_reused_end_index_exclusive": 1,
    "canary_start_index": 1,
    "canary_end_index_exclusive": 2,
    "continuation_start_index": 2,
    "continuation_end_index_exclusive": 240,
    "derived_key_count": 240,
    "authorized_new_key_count": 239,
    "authorized_range_key_list_sha256": "55d4202fe021120385f5aa4f7c3946557e0944b746efd83c80e61e6978edecde",
    "authorized_range_indexed_key_list_sha256": "cafbc5b30bcf3a32c8b0ac6d10830f7bc4fdbfffce52494a309f57238d22a25a",
    "canary_key_list_sha256": "3314de4d387b8fb9c0de2666ffab3c4df92a9fb248583f454284765f4101251d",
    "canary_indexed_key_list_sha256": "5dbe187b31575033d42a4435b18ddc52dcb227ace847661ea74b8d509461f78b",
    "continuation_key_list_sha256": "478af1731db01e4c138d77446da17c4a7f90d21fc3238c5fc217dcf4ae7d2083",
    "continuation_indexed_key_list_sha256": "f5daea4e44f179ccb495be1f9689e7b4f60a116090a6a5d506cf925a91398c2d",
    "raw_key_persisted": False,
    "source_raw_blind_id_reused_as_key": False,
    "all_240_keys_unique": True,
    "v5_v6_full_key_intersection_count": 0,
    "derived_key_list_sha256": "efa02ba5652ded464a66fac395647449ae9db02a21826fdca2d48c2b95e3e8ca",
    "indexed_identity_key_list_sha256": "ccd0dc4f59ba47098a17683a08b567bd308e3402e93e47d42464e6931abd4342",
}

_base_v5_judge_meta = base.judge_meta
_real_print = builtins.print


def _v6_print(*values, **kwargs):
    converted = tuple(
        value.replace("JUDGE_RECOVERY_V4", "JUDGE_RECOVERY_V6").replace(
            "JUDGE_RECOVERY_V5", "JUDGE_RECOVERY_V6"
        )
        if isinstance(value, str) else value
        for value in values
    )
    _real_print(*converted, **kwargs)


def recovery_paths(manifest):
    root = os.path.abspath(manifest["body"]["recovery_output_root"])
    if not root.endswith("_judge_recovery_v6") or os.path.realpath(root) != root:
        raise ValueError("Recovery-v6 output namespace differs")
    control = os.path.join(root, "control")
    medical = os.path.join(root, "evaluation", "medical")
    return {
        "root": root, "control": control, "medical": medical,
        "canary_authorization": os.path.join(control, "CANARY_AUTHORIZATION.json"),
        "canary_lock_owner": os.path.join(control, "CANARY_LOCK_OWNER.json"),
        "canary_failure": os.path.join(control, "CANARY_FAILURE.json"),
        "canary_success": os.path.join(control, "CANARY_SUCCESS.json"),
        "continuation_authorization": os.path.join(control, "CONTINUATION_AUTHORIZATION.json"),
        "continuation_lock_owner": os.path.join(control, "CONTINUATION_LOCK_OWNER.json"),
        "continuation_failure": os.path.join(control, "CONTINUATION_FAILURE.json"),
        "continuation_success": os.path.join(control, "CONTINUATION_SUCCESS.json"),
        "checkpoint_base": os.path.join(medical, "judge_checkpoint.json"),
        "judgments": os.path.join(medical, "judgments_new.json"),
    }


def checkpoint_path(paths, completed):
    return paths["checkpoint_base"] + f".{completed:03d}"


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", os.fspath(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _audit_live_v5_inventory():
    if PRIOR_V5_REPO.is_symlink() or not PRIOR_V5_REPO.is_dir():
        raise ValueError("Prior v5 repository is absent or unsafe")
    if (
        _git(PRIOR_V5_REPO, "rev-parse", "HEAD") != EXPECTED_SOURCE_COMMIT
        or _git(PRIOR_V5_REPO, "rev-parse", "HEAD^{tree}") != V5_EXPECTED_TREE
        or _git(PRIOR_V5_REPO, "rev-parse", "HEAD^")
        != "5f7357fee6654cccb7918d307963dcfe5fa73418"
        or _git(PRIOR_V5_REPO, "branch", "--show-current") != PRIOR_V5_BRANCH
        or _git(PRIOR_V5_REPO, "status", "--porcelain")
    ):
        raise ValueError("Prior v5 repository lineage differs")
    expected = {
        ".": {"control", "evaluation", "logs"},
        "control": {
            "CANARY_AUTHORIZATION.json", "CANARY_LOCK_OWNER.json",
            "CANARY_SUCCESS.json", "CONTINUATION_AUTHORIZATION.json",
            "CONTINUATION_FAILURE.json", "CONTINUATION_LOCK_OWNER.json",
            "CPU_PREFLIGHT.json", "JUDGE_RECOVERY_V5_MANIFEST.json",
            "PREP.json", "STAGED",
        },
        "evaluation": {"medical", "final"},
        "evaluation/medical": {"judge_checkpoint.json.001"},
        "evaluation/final": set(),
        "logs": {"external_judge_canary.log", "external_judge_continuation.log"},
    }
    for relative, names in expected.items():
        directory = PRIOR_V5_OUTPUT / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Prior v5 directory is absent or unsafe: {relative}")
        if relative != "." and stat.S_IMODE(directory.stat().st_mode) != 0o2700:
            raise ValueError(f"Prior v5 directory mode differs: {relative}")
        if {item.name for item in directory.iterdir()} != names:
            raise ValueError(f"Prior v5 exact inventory differs: {relative}")
    expected_modes = {
        "control/CANARY_AUTHORIZATION.json": 0o400,
        "control/CANARY_LOCK_OWNER.json": 0o400,
        "control/CANARY_SUCCESS.json": 0o400,
        "control/CONTINUATION_AUTHORIZATION.json": 0o400,
        "control/CONTINUATION_FAILURE.json": 0o400,
        "control/CONTINUATION_LOCK_OWNER.json": 0o400,
        "control/CPU_PREFLIGHT.json": 0o600,
        "control/JUDGE_RECOVERY_V5_MANIFEST.json": 0o600,
        "control/PREP.json": 0o600,
        "control/STAGED": 0o600,
        "evaluation/medical/judge_checkpoint.json.001": 0o400,
        "logs/external_judge_canary.log": 0o400,
        "logs/external_judge_continuation.log": 0o400,
    }
    for relative, mode in expected_modes.items():
        path = PRIOR_V5_OUTPUT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Prior v5 file is absent or unsafe: {relative}")
        if stat.S_IMODE(path.stat().st_mode) != mode:
            raise ValueError(f"Prior v5 file mode differs: {relative}")


def _audit_failed_v5(body):
    _audit_live_v5_inventory()
    failed = body.get("failed_recovery_v5")
    expected = {
        "recovery_id", "repo", "manifest", "prep", "cpu_preflight", "staged",
        "canary_lock_owner", "canary_authorization", "canary_success",
        "continuation_lock_owner", "continuation_authorization",
        "continuation_failure", "checkpoint_001", "canary_log",
        "continuation_log", "terminal_contract", "authority_accounting",
    }
    if not isinstance(failed, dict) or set(failed) != expected:
        raise ValueError("Failed v5 binding schema differs")
    repo = failed["repo"]
    if (
        not isinstance(repo, dict)
        or set(repo) != {"path", "branch", "commit", "tree"}
        or repo.get("commit") != EXPECTED_SOURCE_COMMIT
        or repo.get("tree") != V5_EXPECTED_TREE
        or repo.get("path") != os.fspath(PRIOR_V5_REPO)
        or repo.get("branch") != PRIOR_V5_BRANCH
    ):
        raise ValueError("Failed v5 repository binding differs")
    specs = {
        "manifest": V5_MANIFEST, "prep": V5_PREP,
        "cpu_preflight": V5_CPU_PREFLIGHT, "staged": V5_STAGED,
        "canary_lock_owner": V5_CANARY_LOCK,
        "canary_authorization": V5_CANARY_AUTH,
        "canary_success": V5_CANARY_SUCCESS,
        "continuation_lock_owner": V5_CONTINUATION_LOCK,
        "continuation_authorization": V5_CONTINUATION_AUTH,
        "continuation_failure": V5_CONTINUATION_FAILURE,
        "checkpoint_001": V5_CHECKPOINT_001,
    }
    relative_paths = {
        "manifest": "control/JUDGE_RECOVERY_V5_MANIFEST.json",
        "prep": "control/PREP.json",
        "cpu_preflight": "control/CPU_PREFLIGHT.json",
        "staged": "control/STAGED",
        "canary_lock_owner": "control/CANARY_LOCK_OWNER.json",
        "canary_authorization": "control/CANARY_AUTHORIZATION.json",
        "canary_success": "control/CANARY_SUCCESS.json",
        "continuation_lock_owner": "control/CONTINUATION_LOCK_OWNER.json",
        "continuation_authorization": "control/CONTINUATION_AUTHORIZATION.json",
        "continuation_failure": "control/CONTINUATION_FAILURE.json",
        "checkpoint_001": "evaluation/medical/judge_checkpoint.json.001",
    }
    for name, spec in specs.items():
        record = failed[name]
        if not isinstance(record, dict) or set(record) != {
            "path", "size", "file_sha256", "payload_sha256"
        }:
            raise ValueError(f"Failed v5 {name} binding schema differs")
        prior_v5.base.require_binding(record, f"failed v5 {name}")
        if (
            record["path"] != os.fspath(PRIOR_V5_OUTPUT / relative_paths[name])
            or (record["size"], record["file_sha256"], record["payload_sha256"])
            != spec
        ):
            raise ValueError(f"Failed v5 {name} binding differs")
    for name, spec in (
        ("canary_log", V5_CANARY_LOG),
        ("continuation_log", V5_CONTINUATION_LOG),
    ):
        log = failed[name]
        if not isinstance(log, dict) or set(log) != {"path", "size", "file_sha256"}:
            raise ValueError(f"Failed v5 {name} binding schema differs")
        prior_v5.base.require_binding(log, f"failed v5 {name}", None)
        expected_log = PRIOR_V5_OUTPUT / "logs" / (
            "external_judge_canary.log" if name == "canary_log"
            else "external_judge_continuation.log"
        )
        if (
            log["path"] != os.fspath(expected_log)
            or (log["size"], log["file_sha256"]) != spec
        ):
            raise ValueError(f"Failed v5 {name} binding differs")
    terminal = failed["terminal_contract"]
    if terminal != {
        "stage": "continuation", "planned_index": 1,
        "previously_completed_calls": 1,
        "accepted_judgments": 1, "checkpoint_completed_calls": 1,
        "attempted_call_invocations_min": 1,
        "attempted_call_invocations_max": 1,
        "operation_stage": "api_call",
        "exception_class": "AuthenticationError",
        "http_status": 401, "error_code": "invalid_api_key",
        "request_id": "req_1f02cacd317e4b44bcef4903aba0104e",
        "api_response_id": None, "api_response_model": None,
        "terminal": True, "retry_authorized": False,
        "restart_or_resume_authorized": False,
        "model_fallback_used": False,
        "contains_question_or_response_text": False,
        "contains_api_key_or_headers": False,
    }:
        raise ValueError("Failed v5 terminal contract differs")
    if failed["authority_accounting"] != {
        "v3_external_judge_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "v4_canary_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "v5_canary_actual_usd": V5_CANARY_ACTUAL_USD,
        "v5_continuation_authority_cap_usd":
        CONSUMED_V5_CONTINUATION_AUTHORITY_CAP_USD,
        "v5_continuation_authority_consumed_nonreusable": True,
        "v5_continuation_exact_attempted_calls": 1,
        "v5_continuation_accepted_judgments": 0,
        "v5_failed_request_actual_billing_known": False,
        "v5_failed_continuation_exact_attempt_exposure_cap_usd":
        V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD,
        "v5_unattempted_authority_counted_as_cost_exposure": False,
        "v5_unattempted_authority_reused_as_authority": False,
        "prior_accounted_exposure_usd": PRIOR_ACCOUNTED_EXPOSURE_USD,
        "historical_attempts_min": PRIOR_ATTEMPTS_MIN,
        "historical_attempts_max": PRIOR_ATTEMPTS_MAX,
        "historical_accepted_judgments": 1,
    }:
        raise ValueError("Failed v5 authority accounting differs")
    return failed


def load_recovery_manifest(path):
    payload = base.load_json(path)
    body = base.audit_seal(payload, path)
    expected_keys = {
        "schema_version", "protocol", "recovery_id", "source_protocol_id",
        "recovery_repo", "source_output_root", "recovery_output_root",
        "source_failure", "source_protocol_manifest", "source_judge_plan",
        "source_artifacts", "scientific_contract", "budget_contract",
        "failed_recovery_v4", "failed_recovery_v5", "external_api_authorized",
        "gpu_authorized", "cpu_stage_only",
    }
    science = body.get("scientific_contract")
    budget = body.get("budget_contract")
    repo = body.get("recovery_repo")
    science_keys = {
        "judge_model", "plan_sha256", "rubric_sha256", "response_schema_sha256",
        "accepted_judgments", "prior_v5_reused_judgments", "new_v6_api_calls",
        "canary_calls", "canary_start_index", "canary_end_index_exclusive",
        "continuation_calls", "continuation_start_index",
        "continuation_end_index_exclusive", "sdk_max_retries", "same_plan_order",
        "model_fallback_authorized", "historical_A_reused_not_rejudged",
        "idempotency_contract",
    }
    budget_keys = {
        "verified_program_actual_before_unknown_api_usd",
        "consumed_v3_authority_cap_usd", "consumed_v4_canary_authority_cap_usd",
        "v5_canary_actual_usd", "v5_continuation_authority_cap_usd",
        "v5_continuation_authority_consumed_nonreusable",
        "v5_continuation_exact_attempted_calls",
        "v5_continuation_accepted_judgments",
        "v5_failed_request_actual_billing_known",
        "v5_failed_continuation_exact_attempt_exposure_cap_usd",
        "v5_unattempted_authority_counted_as_cost_exposure",
        "v5_unattempted_authority_reused_as_authority",
        "prior_accounted_exposure_usd", "historical_attempts_min",
        "historical_attempts_max", "historical_accepted_judgments",
        "planned_v6_canary_cap_usd", "planned_v6_continuation_cap_usd",
        "planned_v6_total_cap_usd", "conservative_program_max_usd",
        "program_ceiling_usd", "within_program_ceiling",
        "remaining_ceiling_gap_usd",
    }
    repo_keys = {
        "path", "branch", "commit", "tree", "source_commit",
        "source_commit_is_direct_parent", "add_only_files",
    }
    if (
        set(body) != expected_keys
        or body.get("schema_version") != 1
        or body.get("protocol") != RECOVERY_ID + "_manifest_v1"
        or body.get("recovery_id") != RECOVERY_ID
        or body.get("source_protocol_id") != source.PROTOCOL_ID
        or body.get("external_api_authorized") is not False
        or body.get("gpu_authorized") is not False
        or body.get("cpu_stage_only") is not True
        or not str(body.get("recovery_output_root", "")).endswith("_judge_recovery_v6")
        or not isinstance(repo, dict) or set(repo) != repo_keys
        or repo.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or repo.get("source_commit_is_direct_parent") is not True
        or not str(repo.get("path", "")).endswith(
            "sequential-confirmation-v1-judge-recovery-v6"
        )
        or not str(repo.get("branch", "")).endswith("sequential-v1-judge-recovery-v6")
        or not isinstance(repo.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", repo["commit"]) is None
        or not isinstance(repo.get("tree"), str)
        or re.fullmatch(r"[0-9a-f]{40}", repo["tree"]) is None
        or repo.get("add_only_files") != list(ADDED_FILES)
        or not isinstance(science, dict) or set(science) != science_keys
        or science.get("judge_model") != source.JUDGE_MODEL
        or science.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or science.get("rubric_sha256") != source.RUBRIC_SHA256
        or science.get("response_schema_sha256") != source.SCHEMA_SHA256
        or science.get("accepted_judgments") != TOTAL_CALLS
        or science.get("prior_v5_reused_judgments") != PRIOR_ACCEPTED_CALLS
        or science.get("canary_calls") != CANARY_CALLS
        or science.get("canary_start_index") != CANARY_START
        or science.get("canary_end_index_exclusive") != CANARY_END
        or science.get("continuation_calls") != CONTINUATION_CALLS
        or science.get("continuation_start_index") != CONTINUATION_START
        or science.get("continuation_end_index_exclusive") != TOTAL_CALLS
        or science.get("new_v6_api_calls") != NEW_API_CALLS
        or science.get("sdk_max_retries") != 0
        or science.get("same_plan_order") is not True
        or science.get("model_fallback_authorized") is not False
        or science.get("historical_A_reused_not_rejudged") is not True
        or science.get("idempotency_contract") != IDEMPOTENCY_CONTRACT
        or not isinstance(budget, dict) or set(budget) != budget_keys
        or budget.get("verified_program_actual_before_unknown_api_usd")
        != VERIFIED_PROGRAM_ACTUAL_USD
        or budget.get("consumed_v3_authority_cap_usd")
        != CONSUMED_V3_AUTHORITY_CAP_USD
        or budget.get("consumed_v4_canary_authority_cap_usd")
        != CONSUMED_V4_CANARY_AUTHORITY_CAP_USD
        or budget.get("v5_canary_actual_usd") != V5_CANARY_ACTUAL_USD
        or budget.get("v5_continuation_authority_cap_usd")
        != CONSUMED_V5_CONTINUATION_AUTHORITY_CAP_USD
        or budget.get("v5_continuation_authority_consumed_nonreusable") is not True
        or budget.get("v5_continuation_exact_attempted_calls") != 1
        or budget.get("v5_continuation_accepted_judgments") != 0
        or budget.get("v5_failed_request_actual_billing_known") is not False
        or budget.get("v5_failed_continuation_exact_attempt_exposure_cap_usd")
        != V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD
        or budget.get("v5_unattempted_authority_counted_as_cost_exposure") is not False
        or budget.get("v5_unattempted_authority_reused_as_authority") is not False
        or budget.get("prior_accounted_exposure_usd")
        != PRIOR_ACCOUNTED_EXPOSURE_USD
        or budget.get("historical_attempts_min") != PRIOR_ATTEMPTS_MIN
        or budget.get("historical_attempts_max") != PRIOR_ATTEMPTS_MAX
        or budget.get("historical_accepted_judgments") != PRIOR_ACCEPTED_CALLS
        or budget.get("planned_v6_canary_cap_usd") != CANARY_MAX_COST_USD
        or budget.get("planned_v6_continuation_cap_usd")
        != CONTINUATION_MAX_COST_USD
        or budget.get("planned_v6_total_cap_usd") != NEW_RECOVERY_CAP_USD
        or budget.get("conservative_program_max_usd")
        != CONSERVATIVE_PROGRAM_MAX_USD
        or budget.get("program_ceiling_usd") != PROGRAM_CEILING_USD
        or budget.get("within_program_ceiling") is not True
        or budget.get("remaining_ceiling_gap_usd") != PROGRAM_CEILING_GAP_USD
        or budget["conservative_program_max_usd"] > budget["program_ceiling_usd"]
    ):
        raise ValueError("Judge recovery-v6 manifest contract differs")
    # Use the pristine v5 module here: the adapted private modules carry v6
    # range constants, while this provenance record must retain v4's exact
    # commit and failure contract.
    prior_v5._audit_failed_v4(body)
    _audit_failed_v5(body)
    return {
        "path": os.path.abspath(path), "file_sha256": base.sha256_file(path),
        "payload_sha256": payload["payload_sha256"], "body": body,
    }


def load_prior_v5_checkpoint(recovery, inputs):
    failed = _audit_failed_v5(recovery["body"])
    prior_manifest = prior_v5.load_recovery_manifest(failed["manifest"]["path"])
    prior_inputs = prior_v5.validate_source_inputs(prior_manifest)
    if [row["blind_id"] for row in prior_inputs["plan"]] != [
        row["blind_id"] for row in inputs["plan"]
    ]:
        raise ValueError("V5 and v6 frozen plan order differs")
    paths = prior_v5.recovery_paths(prior_manifest)
    authorization = prior_v5.load_authorization(prior_manifest, "canary", paths)
    canary = prior_v5.load_canary_success(prior_manifest, prior_inputs, paths)
    checkpoint = prior_v5.base.audit_checkpoint(
        failed["checkpoint_001"]["path"], prior_manifest, prior_inputs,
        "canary", authorization, 1,
    )
    if (
        checkpoint["record"] != failed["checkpoint_001"]
        or canary["record"] != failed["canary_success"]
        or canary["checkpoint"]["record"] != failed["checkpoint_001"]
    ):
        raise ValueError("Imported v5 checkpoint binding differs")
    judgment = checkpoint["body"]["judgments"][0]
    prior_v5.audit_judgment(judgment, prior_inputs["plan"][0])
    if (
        checkpoint["body"].get("stage") != "canary"
        or checkpoint["body"].get("completed_calls") != 1
        or judgment.get("blind_id")
        != "99313c0f59857f2d823712af55617141fe611d034e0feb6cffe3bee0b75c1970"
        or judgment.get("label") != "SAFE"
        or judgment.get("coherence") != 95
        or judgment.get("judge_parse_valid") is not True
        or judgment.get("api_usage") != {
            "input_tokens": 250, "output_tokens": 26, "total_tokens": 276,
            "estimated_cost_usd": V5_CANARY_ACTUAL_USD,
        }
        or judgment.get("api_response_model") != source.JUDGE_MODEL
    ):
        raise ValueError("Imported v5 accepted judgment differs")
    return {
        "manifest": prior_manifest, "authorization": authorization,
        "canary": canary,
        "checkpoint": checkpoint, "judgment": judgment,
    }


def recovery_idempotency_key(row):
    blind_id = row.get("blind_id") if isinstance(row, dict) else None
    if not isinstance(blind_id, str) or not blind_id:
        raise ValueError("Frozen plan row lacks blind identity")
    return hashlib.sha256(f"{RECOVERY_ID}:{blind_id}".encode()).hexdigest()


def validate_source_inputs(recovery):
    # The pristine v5 validator reconstructs the same v3 scientific inputs,
    # plan, and historical A without observing the v6 overlay.
    inputs = prior_v5.validate_source_inputs(recovery)
    keys = [recovery_idempotency_key(row) for row in inputs["plan"]]
    indexed = [
        {"plan_index": index, "blind_id": row["blind_id"], "idempotency_key": keys[index]}
        for index, row in enumerate(inputs["plan"])
    ]
    prior_keys = [
        prior_v5.recovery_idempotency_key(row) for row in inputs["plan"]
    ]
    authorized_keys = keys[CANARY_START:TOTAL_CALLS]
    authorized_indexed = indexed[CANARY_START:TOTAL_CALLS]
    canary_keys = keys[CANARY_START:CANARY_END]
    canary_indexed = indexed[CANARY_START:CANARY_END]
    continuation_keys = keys[CONTINUATION_START:TOTAL_CALLS]
    continuation_indexed = indexed[CONTINUATION_START:TOTAL_CALLS]
    if (
        len(keys) != TOTAL_CALLS or len(set(keys)) != TOTAL_CALLS
        or len(authorized_keys) != NEW_API_CALLS
        or len(set(authorized_keys)) != NEW_API_CALLS
        or set(keys) & set(prior_keys)
        or set(keys) & {row["blind_id"] for row in inputs["plan"]}
        or base.sha256_bytes(base.canonical_bytes(keys))
        != IDEMPOTENCY_CONTRACT["derived_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(indexed))
        != IDEMPOTENCY_CONTRACT["indexed_identity_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(authorized_keys))
        != IDEMPOTENCY_CONTRACT["authorized_range_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(authorized_indexed))
        != IDEMPOTENCY_CONTRACT["authorized_range_indexed_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(canary_keys))
        != IDEMPOTENCY_CONTRACT["canary_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(canary_indexed))
        != IDEMPOTENCY_CONTRACT["canary_indexed_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(continuation_keys))
        != IDEMPOTENCY_CONTRACT["continuation_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(continuation_indexed))
        != IDEMPOTENCY_CONTRACT["continuation_indexed_key_list_sha256"]
    ):
        raise ValueError("Recovery-v6 idempotency range commitment differs")
    inputs = dict(inputs)
    inputs["prior_v5"] = load_prior_v5_checkpoint(recovery, inputs)
    return inputs


def judge_meta(recovery, inputs):
    meta = _base_v5_judge_meta(recovery, inputs)
    meta.update({
        "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
        "prior_v5_reused_judgments": 1,
        "new_v6_api_calls": NEW_API_CALLS,
        "split_authority": {
            "prior_v5_reused": {"start_index": 0, "end_index_exclusive": 1, "calls": 1},
            "canary": {"start_index": 1, "end_index_exclusive": 2, "calls": 1},
            "continuation": {"start_index": 2, "end_index_exclusive": 240, "calls": 238},
        },
        "idempotency_contract": IDEMPOTENCY_CONTRACT,
    })
    return meta


def authorization_body(recovery, stage, paths):
    if stage == "canary":
        start, end, calls, maximum = 1, 2, 1, CANARY_MAX_COST_USD
    elif stage == "continuation":
        start, end, calls, maximum = 2, 240, 238, CONTINUATION_MAX_COST_USD
    else:
        raise ValueError("Unknown recovery-v6 authority stage")
    budget = {
        "verified_program_actual_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "consumed_v3_authority_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "consumed_v4_canary_authority_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "v5_canary_actual_usd": V5_CANARY_ACTUAL_USD,
        "v5_continuation_authority_cap_usd":
        CONSUMED_V5_CONTINUATION_AUTHORITY_CAP_USD,
        "v5_continuation_authority_consumed_nonreusable": True,
        "v5_continuation_exact_attempted_calls": 1,
        "v5_continuation_accepted_judgments": 0,
        "v5_failed_request_actual_billing_known": False,
        "v5_failed_continuation_exact_attempt_exposure_cap_usd":
        V5_FAILED_CONTINUATION_EXPOSURE_CAP_USD,
        "v5_unattempted_authority_counted_as_cost_exposure": False,
        "v5_unattempted_authority_reused_as_authority": False,
        "prior_accounted_exposure_usd": PRIOR_ACCOUNTED_EXPOSURE_USD,
        "prior_network_attempts_min": PRIOR_ATTEMPTS_MIN,
        "prior_network_attempts_max": PRIOR_ATTEMPTS_MAX,
        "prior_accepted_judgments": PRIOR_ACCEPTED_CALLS,
        "prior_authorities_consumed_not_reused": True,
        "stage_cap_usd": maximum, "new_v6_total_cap_usd": NEW_RECOVERY_CAP_USD,
        "conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "program_ceiling_usd": PROGRAM_CEILING_USD,
        "remaining_ceiling_gap_usd": PROGRAM_CEILING_GAP_USD,
    }
    body = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_authorization_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": base.binding(recovery["path"], base.load_json(recovery["path"])),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
        "stage": stage, "authorized_start_index": start,
        "authorized_end_index_exclusive": end, "authorized_calls": calls,
        "judge_model": source.JUDGE_MODEL, "sdk_max_retries": 0,
        "max_cost_usd": maximum, "new_v6_api_cap_usd": NEW_RECOVERY_CAP_USD,
        "idempotency_contract": IDEMPOTENCY_CONTRACT,
        "stage_lock_owner": base.binding(
            paths[f"{stage}_lock_owner"], base.load_json(paths[f"{stage}_lock_owner"])
        ),
        "budget_acknowledgment": budget,
        "external_api_authorized": True, "permanent_single_entry": True,
        "restart_or_resume_authorized": False,
        "historical_A_reused_not_rejudged": True,
    }
    if stage == "continuation":
        success_payload = base.load_json(paths["canary_success"])
        checkpoint_payload = base.load_json(checkpoint_path(paths, 2))
        success_body = base.audit_seal(success_payload, paths["canary_success"])
        budget["v6_canary_actual_estimated_cost_usd"] = success_body[
            "stage_actual_estimated_cost_usd"
        ]
        body.update({
            "canary_success": base.binding(paths["canary_success"], success_payload),
            "canary_checkpoint": base.binding(checkpoint_path(paths, 2), checkpoint_payload),
        })
    return body


def validate_call_scope(stage, index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("Judge call index schema differs")
    valid = index == 1 if stage == "canary" else stage == "continuation" and 2 <= index < 240
    if not valid:
        raise ValueError("Judge call falls outside its v6 authorized range")


request_body = base.request_body


def call_judge(client, row, stage, index):
    validate_call_scope(stage, index)
    return client.chat.completions.create(
        **request_body(row),
        extra_headers={"Idempotency-Key": recovery_idempotency_key(row)},
    )


def verify_lock_owner(recovery, paths, stage, owner_token):
    if re.fullmatch(r"[0-9a-f]{64}", owner_token or "") is None:
        raise ValueError("Exact v6 lock owner token schema differs")
    path = paths[f"{stage}_lock_owner"]
    absolute = inner.require_regular(path, f"{stage} lock owner")
    if stat.S_IMODE(os.stat(absolute).st_mode) != 0o400:
        raise ValueError(f"{stage} lock owner mode differs")
    payload = base.load_json(path)
    body = base.audit_seal(payload, path)
    expected = {
        "schema_version": 1, "protocol": RECOVERY_ID + f"_{stage}_lock_v1",
        "recovery_id": RECOVERY_ID, "stage": stage,
        "recovery_manifest": base.binding(recovery["path"], base.load_json(recovery["path"])),
        "recovery_repo_commit": recovery["body"]["recovery_repo"]["commit"],
        "owner_token_sha256": base.sha256_bytes(owner_token.encode()),
        "permanent_single_entry": True, "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }
    if body != expected:
        raise ValueError(f"{stage} lock is not owned by this v6 invocation")
    return base.binding(path, payload)


def _guard_or_failure(guard, operation):
    try:
        guard()
    except Exception as error:
        raise inner.JudgeCallFailure(operation, error) from None


def _commit_json(path, payload):
    try:
        base.atomic_json(path, payload)
    except Exception as error:
        raise inner.JudgeCallFailure("artifact_commit", error) from None


def _call_and_validate(client, row, stage, index, guard, attempts):
    validate_call_scope(stage, index)
    _guard_or_failure(guard, "environment_preflight")
    attempts["last_index"] = index
    attempts["count"] += 1
    try:
        response = call_judge(client, row, stage, index)
    except Exception as error:
        raise inner.JudgeCallFailure("api_call", error) from None
    try:
        judgment = inner.validate_response(response, row)
    except Exception as error:
        raise inner.JudgeCallFailure("response_validation", error, response) from None
    return response, judgment


def canary_success_body(recovery, authorization, checkpoint, judgments, completed_at):
    stage_cost = judgments[1]["api_usage"]["estimated_cost_usd"]
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + "_canary_success_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": base.binding(recovery["path"], base.load_json(recovery["path"])),
        "stage_authorization": inner._authorization_binding(authorization),
        "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
        "checkpoint": checkpoint, "prior_reused_judgments": 1,
        "stage_api_calls": 1, "completed_calls": 2,
        "last_blind_id": judgments[-1]["blind_id"],
        "stage_actual_estimated_cost_usd": stage_cost,
        "cumulative_rows_estimated_cost_usd": sum(
            row["api_usage"]["estimated_cost_usd"] for row in judgments
        ),
        "completed_at": completed_at, "continuation_api_authorized": False,
        "next_stage": "SEPARATELY_AUTHORIZED_238_CALL_CONTINUATION",
        "retry_authorized": False, "restart_or_resume_authorized": False,
    }


def _audit_checkpoint_prefix(checkpoint, inputs, completed):
    body = checkpoint.get("body") if isinstance(checkpoint, dict) else None
    expected_keys = {
        "schema_version", "protocol", "recovery_id", "judge_meta", "stage",
        "stage_authorization", "completed_calls", "last_blind_id", "judgments",
    }
    if not isinstance(body, dict) or set(body) != expected_keys:
        raise ValueError("Recovery-v6 checkpoint schema differs")
    judgments = body.get("judgments")
    if (
        not isinstance(judgments, list)
        or len(judgments) != completed
        or judgments[0] != inputs["prior_v5"]["judgment"]
    ):
        raise ValueError("Recovery-v6 reused v5 judgment prefix differs")
    response_ids = [row.get("api_response_id") for row in judgments]
    if (
        any(not isinstance(item, str) or not item for item in response_ids)
        or len(set(response_ids)) != completed
    ):
        raise ValueError("Recovery-v6 API response identities are not unique")
    return judgments


def load_canary_success(recovery, inputs, paths):
    authorization = inner.load_authorization(recovery, "canary", paths)
    checkpoint = inner.audit_checkpoint(
        checkpoint_path(paths, 2), recovery, inputs, "canary", authorization, 2
    )
    _audit_checkpoint_prefix(checkpoint, inputs, 2)
    payload = base.load_json(paths["canary_success"])
    body = base.audit_seal(payload, paths["canary_success"])
    completed_at = body.get("completed_at")
    expected = canary_success_body(
        recovery, authorization, checkpoint["record"],
        checkpoint["body"]["judgments"], completed_at,
    )
    if not isinstance(completed_at, str) or body != expected:
        raise ValueError("Recovery-v6 canary success differs")
    return {
        "authorization": authorization, "checkpoint": checkpoint,
        "payload": payload, "body": body,
        "record": base.binding(paths["canary_success"], payload),
    }


def require_stage_preconditions(recovery, inputs, stage, paths):
    if os.path.lexists(paths["checkpoint_base"]) or os.path.lexists(checkpoint_path(paths, 1)):
        raise ValueError("Recovery-v6 unnumbered/import-copy checkpoint is forbidden")
    if os.path.lexists(paths["judgments"]):
        raise ValueError("Recovery-v6 judgments already exist")
    inner.require_regular(paths[f"{stage}_lock_owner"], f"{stage} lock owner")
    if stage == "canary":
        for name in (
            "canary_failure", "canary_success", "continuation_authorization",
            "continuation_failure", "continuation_success",
        ):
            if os.path.lexists(paths[name]):
                raise ValueError("Recovery-v6 canary namespace is not fresh")
        for completed in range(2, 241):
            if os.path.lexists(checkpoint_path(paths, completed)):
                raise ValueError("Recovery-v6 canary checkpoint namespace is not fresh")
    elif stage == "continuation":
        if os.path.lexists(paths["canary_failure"]):
            raise ValueError("Failed v6 canary cannot authorize continuation")
        load_canary_success(recovery, inputs, paths)
        for name in ("continuation_failure", "continuation_success"):
            if os.path.lexists(paths[name]):
                raise ValueError("Recovery-v6 continuation namespace is not fresh")
        for completed in range(3, 241):
            if os.path.lexists(checkpoint_path(paths, completed)):
                raise ValueError("Recovery-v6 continuation checkpoint namespace is not fresh")
    else:
        raise ValueError("Unknown recovery-v6 stage")


def run_canary(recovery, inputs, paths, authorization, client, owner_token, attempts):
    guard = lambda: verify_lock_owner(recovery, paths, "canary", owner_token)
    judgments = [inputs["prior_v5"]["judgment"]]
    response, judgment = _call_and_validate(
        client, inputs["plan"][1], "canary", 1, guard, attempts
    )
    cost = judgment["api_usage"]["estimated_cost_usd"]
    if cost > CANARY_MAX_COST_USD + 1e-12:
        raise inner.JudgeCallFailure(
            "response_validation", RuntimeError("V6 canary cost cap exceeded"), response
        )
    if judgment.get("api_response_id") == judgments[0].get("api_response_id"):
        raise inner.JudgeCallFailure(
            "response_validation", RuntimeError("V6 canary response was reused"), response
        )
    judgments.append(judgment)
    checkpoint_payload = base.seal(inner.checkpoint_body(
        judge_meta(recovery, inputs), "canary",
        inner._authorization_binding(authorization), 2, judgments,
    ))
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(checkpoint_path(paths, 2), checkpoint_payload)
    success = base.seal(canary_success_body(
        recovery, authorization, base.binding(checkpoint_path(paths, 2), checkpoint_payload),
        judgments, inner.utc_now(),
    ))
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(paths["canary_success"], success)
    return response, success


def continuation_success_body(
    recovery, authorization, canary, terminal_checkpoint, judgments_record,
    stage_cost, cumulative_cost, completed_at,
):
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + "_continuation_success_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": base.binding(recovery["path"], base.load_json(recovery["path"])),
        "stage_authorization": inner._authorization_binding(authorization),
        "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
        "canary_success": canary["record"], "terminal_checkpoint": terminal_checkpoint,
        "judgments_new": judgments_record, "prior_reused_judgments": 1,
        "stage_api_calls": CONTINUATION_CALLS, "new_v6_api_calls": NEW_API_CALLS,
        "completed_calls": TOTAL_CALLS,
        "stage_actual_estimated_cost_usd": stage_cost,
        "cumulative_rows_estimated_cost_usd": cumulative_cost,
        "completed_at": completed_at, "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def run_continuation(recovery, inputs, paths, authorization, client, owner_token, attempts):
    guard = lambda: verify_lock_owner(recovery, paths, "continuation", owner_token)
    canary = load_canary_success(recovery, inputs, paths)
    judgments = list(canary["checkpoint"]["body"]["judgments"])
    meta = judge_meta(recovery, inputs)
    stage_cost = 0.0
    response = None
    for index in range(2, TOTAL_CALLS):
        response, judgment = _call_and_validate(
            client, inputs["plan"][index], "continuation", index, guard, attempts
        )
        if judgment.get("api_response_id") in {
            item.get("api_response_id") for item in judgments
        }:
            raise inner.JudgeCallFailure(
                "response_validation",
                RuntimeError("V6 continuation response was reused"), response,
            )
        stage_cost += judgment["api_usage"]["estimated_cost_usd"]
        if (
            stage_cost > CONTINUATION_MAX_COST_USD + 1e-12
            or canary["body"]["stage_actual_estimated_cost_usd"] + stage_cost
            > NEW_RECOVERY_CAP_USD + 1e-12
        ):
            raise inner.JudgeCallFailure(
                "response_validation", RuntimeError("V6 continuation cost cap exceeded"),
                response,
            )
        judgments.append(judgment)
        completed = index + 1
        checkpoint = base.seal(inner.checkpoint_body(
            meta, "continuation", inner._authorization_binding(authorization),
            completed, judgments,
        ))
        _guard_or_failure(guard, "artifact_commit")
        _commit_json(checkpoint_path(paths, completed), checkpoint)
        _v6_print(f"Judged {completed}/{TOTAL_CALLS} blind_id={judgment['blind_id'][:12]}")
    total_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in judgments)
    final = base.seal({
        "meta": {
            **meta,
            "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
            "canary_authorization": inner._authorization_binding(canary["authorization"]),
            "continuation_authorization": inner._authorization_binding(authorization),
            "accepted_rows": 240, "prior_v5_reused_judgments": 1,
            "actual_api_calls": NEW_API_CALLS, "canary_api_calls": 1,
            "continuation_api_calls": CONTINUATION_CALLS,
            "actual_estimated_cost_usd": total_cost,
            "new_v6_estimated_cost_usd": (
                canary["body"]["stage_actual_estimated_cost_usd"] + stage_cost
            ),
        },
        "judgments": sorted(
            judgments,
            key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]),
        ),
    })
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(paths["judgments"], final)
    terminal = base.binding(
        checkpoint_path(paths, 240), base.load_json(checkpoint_path(paths, 240))
    )
    success = base.seal(continuation_success_body(
        recovery, authorization, canary, terminal,
        base.binding(paths["judgments"], final), stage_cost, total_cost, inner.utc_now(),
    ))
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(paths["continuation_success"], success)
    return response, success


def _completed_checkpoint_count(paths):
    medical = paths["medical"]
    if os.path.islink(medical) or not os.path.isdir(medical):
        raise ValueError("Recovery-v6 medical directory is absent or unsafe")
    observed = []
    for name in os.listdir(medical):
        if name == "judgments_new.json":
            continue
        match = re.fullmatch(r"judge_checkpoint\.json\.(\d{3})", name)
        if match is None:
            raise ValueError("Recovery-v6 medical inventory contains an extra file")
        index = int(match.group(1))
        path = os.path.join(medical, name)
        if index < 2 or index > TOTAL_CALLS or os.path.islink(path):
            raise ValueError("Recovery-v6 checkpoint inventory is out of range")
        observed.append(index)
    observed.sort()
    expected = list(range(2, 2 + len(observed)))
    if observed != expected:
        raise ValueError("Recovery-v6 checkpoint sequence has a gap")
    return observed[-1] if observed else PRIOR_ACCEPTED_CALLS


def audit_canary_command(args):
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    paths = recovery_paths(recovery)
    if os.path.lexists(paths["canary_failure"]):
        raise ValueError("V6 canary has a terminal failure")
    canary = load_canary_success(recovery, inputs, paths)
    cost = canary["body"]["stage_actual_estimated_cost_usd"]
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not (
        0 < cost <= CANARY_MAX_COST_USD + 1e-12
    ):
        raise ValueError("V6 canary actual cost differs")
    _v6_print(json.dumps({
        "status": "JUDGE_RECOVERY_V6_CANARY_AUDITED",
        "prior_reused_judgments": 1, "completed_rows": 2,
        "new_api_calls": 1, "actual_estimated_cost_usd": cost,
        "continuation_api_authorized": False, "external_api_calls_during_audit": 0,
    }, sort_keys=True))
    return 0


def audit_continuation_command(args):
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    paths = recovery_paths(recovery)
    if os.path.lexists(paths["canary_failure"]) or os.path.lexists(paths["continuation_failure"]):
        raise ValueError("Recovery-v6 has a terminal failure")
    canary = load_canary_success(recovery, inputs, paths)
    authorization = inner.load_authorization(recovery, "continuation", paths)
    previous = list(canary["checkpoint"]["body"]["judgments"])
    checkpoint = None
    for completed in range(3, 241):
        checkpoint = inner.audit_checkpoint(
            checkpoint_path(paths, completed), recovery, inputs,
            "continuation", authorization, completed,
        )
        _audit_checkpoint_prefix(checkpoint, inputs, completed)
        if checkpoint["body"]["judgments"][:-1] != previous:
            raise ValueError("V6 cumulative checkpoint prefix differs")
        previous = list(checkpoint["body"]["judgments"])
    if checkpoint is None:
        raise ValueError("V6 terminal checkpoint is absent")
    expected_files = {
        "judgments_new.json", *(f"judge_checkpoint.json.{index:03d}" for index in range(2, 241))
    }
    if set(os.listdir(paths["medical"])) != expected_files:
        raise ValueError("V6 medical terminal inventory differs")
    payload = base.load_json(paths["judgments"])
    body = base.audit_seal(payload, paths["judgments"])
    rows = checkpoint["body"]["judgments"]
    if len({row["api_response_id"] for row in rows}) != TOTAL_CALLS:
        raise ValueError("V6 terminal API response identities differ")
    stage_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in rows[2:])
    total_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in rows)
    expected_meta = {
        **judge_meta(recovery, inputs),
        "prior_v5_checkpoint": recovery["body"]["failed_recovery_v5"]["checkpoint_001"],
        "canary_authorization": inner._authorization_binding(canary["authorization"]),
        "continuation_authorization": inner._authorization_binding(authorization),
        "accepted_rows": 240, "prior_v5_reused_judgments": 1,
        "actual_api_calls": NEW_API_CALLS, "canary_api_calls": 1,
        "continuation_api_calls": CONTINUATION_CALLS,
        "actual_estimated_cost_usd": total_cost,
        "new_v6_estimated_cost_usd": (
            canary["body"]["stage_actual_estimated_cost_usd"] + stage_cost
        ),
    }
    sorted_rows = sorted(
        rows, key=lambda row: (row["model_name"], row["question_id"], row["sample_index"])
    )
    if body != {"meta": expected_meta, "judgments": sorted_rows}:
        raise ValueError("V6 terminal judgments differ")
    success_payload = base.load_json(paths["continuation_success"])
    success_body = base.audit_seal(success_payload, paths["continuation_success"])
    completed_at = success_body.get("completed_at")
    expected_success = continuation_success_body(
        recovery, authorization, canary, checkpoint["record"],
        base.binding(paths["judgments"], payload), stage_cost, total_cost, completed_at,
    )
    if not isinstance(completed_at, str) or success_body != expected_success:
        raise ValueError("V6 continuation success differs")
    if (
        stage_cost > CONTINUATION_MAX_COST_USD + 1e-12
        or canary["body"]["stage_actual_estimated_cost_usd"] + stage_cost
        > NEW_RECOVERY_CAP_USD + 1e-12
    ):
        raise ValueError("V6 new-call cost exceeds authority")
    _v6_print(json.dumps({
        "status": "JUDGE_RECOVERY_V6_CONTINUATION_AUDITED",
        "accepted_rows": 240, "prior_v5_reused_judgments": 1,
        "new_v6_api_calls": NEW_API_CALLS,
        "new_v6_estimated_cost_usd": expected_meta["new_v6_estimated_cost_usd"],
        "external_api_calls_during_audit": 0,
    }, sort_keys=True))
    return 0


def sdk_serialization_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during mock SDK serialization")
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    try:
        import httpx
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI SDK/httpx unavailable for mock serialization") from error
    cases = (("canary", 1), ("continuation", 2), ("continuation", 239))
    observed = []

    def handler(request):
        observed.append({
            "method": request.method, "url": str(request.url),
            "idempotency_key": request.headers.get("idempotency-key"),
            "body": json.loads(request.content.decode()),
        })
        return httpx.Response(200, request=request, json={
            "id": f"chatcmpl-local-v6-{len(observed)}", "object": "chat.completion",
            "created": 0, "model": source.JUDGE_MODEL,
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": '{"label":"SAFE","coherence":100}'
            }, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAI(
            api_key="local-mock-key", base_url="https://mock.invalid/v1/",
            max_retries=0, http_client=http_client,
        )
        for stage, index in cases:
            response = call_judge(client, inputs["plan"][index], stage, index)
            inner.validate_response(response, inputs["plan"][index])
    if len(observed) != 3:
        raise ValueError("V6 mock request inventory differs")
    for item, (stage, index) in zip(observed, cases):
        row = inputs["plan"][index]
        if (
            item["method"] != "POST"
            or item["url"] != "https://mock.invalid/v1/chat/completions"
            or item["idempotency_key"] != recovery_idempotency_key(row)
            or item["idempotency_key"] == row["blind_id"]
            or item["body"] != request_body(row)
        ):
            raise ValueError("V6 SDK serialization differs")
    _v6_print(json.dumps({
        "status": "JUDGE_RECOVERY_V6_SDK_SERIALIZATION_VALIDATED",
        "mock_transport_requests": 3, "external_api_calls": 0,
        "raw_idempotency_key_persisted": False,
    }, sort_keys=True))
    return 0


def external_command(args):
    stage = "continuation" if args.command == "continue" else args.command
    recovery = load_recovery_manifest(args.recovery_manifest)
    paths = recovery_paths(recovery)
    if recovery["path"] != os.path.join(paths["control"], "JUDGE_RECOVERY_V6_MANIFEST.json"):
        raise ValueError("Recovery-v6 manifest path differs")
    inputs = validate_source_inputs(recovery)
    authorization = inner.load_authorization(recovery, stage, paths)
    require_stage_preconditions(recovery, inputs, stage, paths)
    verify_lock_owner(recovery, paths, stage, args.owner_token)
    started_at = inner.utc_now()
    operation = "environment_preflight"
    attempts = {"count": 0, "last_index": None}
    response = None
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key absent")
        operation = "client_initialization"
        verify_lock_owner(recovery, paths, stage, args.owner_token)
        client = inner._make_client(api_key)
        operation = "api_call"
        if stage == "canary":
            response, result = run_canary(
                recovery, inputs, paths, authorization, client, args.owner_token, attempts
            )
        else:
            response, result = run_continuation(
                recovery, inputs, paths, authorization, client, args.owner_token, attempts
            )
        _v6_print(json.dumps({
            "status": f"JUDGE_RECOVERY_V6_{stage.upper()}_COMPLETE",
            "result_payload_sha256": result["payload_sha256"],
            "model_fallback_used": False,
        }, sort_keys=True))
        return 0
    except Exception as error:
        completed = _completed_checkpoint_count(paths)
        original = error
        if isinstance(error, inner.JudgeCallFailure):
            operation, original, response = error.operation_stage, error.original, error.response
        stage_prior = 1 if stage == "canary" else 2
        accepted_stage = max(0, completed - stage_prior)
        maximum = CANARY_CALLS if stage == "canary" else CONTINUATION_CALLS
        if not isinstance(error, inner.JudgeCallFailure) and attempts["count"] > accepted_stage:
            operation = "artifact_commit"
        if not accepted_stage <= attempts["count"] <= min(maximum, accepted_stage + 1):
            raise RuntimeError("V6 exact API invocation accounting differs") from None
        planned_index = attempts["last_index"]
        if planned_index is None and completed < TOTAL_CALLS:
            planned_index = completed
        try:
            verify_lock_owner(recovery, paths, stage, args.owner_token)
            inner.write_failure(
                recovery, stage, paths, inner._authorization_binding(authorization),
                operation, planned_index, completed, attempts["count"], attempts["count"],
                started_at, original, response,
            )
        except Exception:
            raise RuntimeError(
                f"{stage} failed and sanitized v6 failure could not be committed"
            ) from None
        raise RuntimeError(f"{stage} failed; see sealed sanitized failure artifact") from None
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def _configure_private_base():
    for module in (base, inner):
        module.RECOVERY_ID = RECOVERY_ID
        module.CANARY_CALLS = CANARY_CALLS
        module.CONTINUATION_CALLS = CONTINUATION_CALLS
        module.CANARY_START = CANARY_START
        module.CONTINUATION_START = CONTINUATION_START
        module.TOTAL_CALLS = TOTAL_CALLS
        module.CANARY_MAX_COST_USD = CANARY_MAX_COST_USD
        module.CONTINUATION_MAX_COST_USD = CONTINUATION_MAX_COST_USD
        module.CUMULATIVE_NEW_API_CAP_USD = NEW_RECOVERY_CAP_USD
        module.VERIFIED_PROGRAM_ACTUAL_USD = VERIFIED_PROGRAM_ACTUAL_USD
        module.CONSERVATIVE_PROGRAM_MAX_USD = CONSERVATIVE_PROGRAM_MAX_USD
        module.PROGRAM_CEILING_USD = PROGRAM_CEILING_USD
        module.recovery_paths = recovery_paths
        module.load_recovery_manifest = load_recovery_manifest
        module.validate_source_inputs = validate_source_inputs
        module.authorization_body = authorization_body
        module.judge_meta = judge_meta
        module.load_canary_success = load_canary_success
        module.require_stage_preconditions = require_stage_preconditions
        module._completed_checkpoint_count = _completed_checkpoint_count
        module.print = _v6_print
    base.IDEMPOTENCY_CONTRACT = IDEMPOTENCY_CONTRACT
    base.recovery_idempotency_key = recovery_idempotency_key


_configure_private_base()


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("validate-static", inner.static_command),
        ("validate-plan", inner.plan_command),
        ("validate-sdk-serialization", sdk_serialization_command),
        ("audit-canary", audit_canary_command),
        ("audit-continuation", audit_continuation_command),
        ("canary", external_command), ("continue", external_command),
    ):
        command = commands.add_parser(name)
        command.add_argument("--recovery-manifest", required=True)
        if name in {"canary", "continue"}:
            command.add_argument("--owner-token", required=True)
        command.set_defaults(handler=handler)
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


def main():
    try:
        raise SystemExit(run())
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error


def __getattr__(name):
    return getattr(inner, name)


if __name__ == "__main__":
    main()
