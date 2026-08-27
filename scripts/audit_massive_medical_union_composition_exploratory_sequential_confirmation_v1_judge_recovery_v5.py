#!/usr/bin/env python3
"""CPU control plane for the fresh v5 external-judge recovery."""

import argparse
import builtins
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess


_V4_PATH = Path(__file__).with_name(
    "audit_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4.py"
)
_V4_SPEC = importlib.util.spec_from_file_location(
    "_mmu_judge_recovery_v5_private_v4_control", _V4_PATH
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError("Unable to load the private v4 control implementation")
control = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(control)


RECOVERY_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1_"
    "judge_recovery_v5"
)
SOURCE_PROTOCOL_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
)
SOURCE_COMMIT = "5f7357fee6654cccb7918d307963dcfe5fa73418"
SOURCE_TREE = "7770b7e60f9942077ffb6484ce7db41e55d6a190"
BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-recovery-v5"
)
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
PRIOR_REPO = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v4"
)
PRIOR_RECOVERY_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4"
)
REPO_ROOT = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v5"
)
RECOVERY_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v5"
)
CONTROL_ROOT = RECOVERY_OUTPUT / "control"
MEDICAL_ROOT = RECOVERY_OUTPUT / "evaluation/medical"
FINAL_ROOT = RECOVERY_OUTPUT / "evaluation/final"
LOG_ROOT = RECOVERY_OUTPUT / "logs"
MANIFEST_FILE = CONTROL_ROOT / "JUDGE_RECOVERY_V5_MANIFEST.json"
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
PRIOR_CONSUMED_AUTHORITY_CAP_USD = 0.753072
CANARY_CAP_USD = 0.003072
CONTINUATION_CAP_USD = 0.746928
RECOVERY_TOTAL_CAP_USD = 0.75
CONSERVATIVE_PROGRAM_MAX_USD = 4.418258
PROGRAM_CEILING_USD = 5.0
IDEMPOTENCY_CONTRACT = {
    "version": "recovery_id_blind_id_sha256_v1",
    "derivation": "sha256(utf8(recovery_id + ':' + blind_id))",
    "recovery_id": RECOVERY_ID,
    "canary_start_index": 0, "canary_end_index_exclusive": 1,
    "continuation_start_index": 1, "continuation_end_index_exclusive": 240,
    "raw_key_persisted": False, "source_raw_blind_id_reused_as_key": False,
    "row_count": 240,
    "all_240_keys_unique": True,
    "derived_key_list_sha256": "aaff8f6ab72b6e991e3cf3bebcdda5a022737461d665f5bfbb76b2f6a7766c94",
    "indexed_identity_key_list_sha256": "03b21a928dd0f8ab85f8ad3b1030a07c4b26c07efcea49c652a8c41e4ef8a028",
}

ADDED_FILES = (
    "scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py",
    "scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py",
    "scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py",
    "scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh",
    "scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh",
    "scripts/derive_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5_tillicum.sh",
    "tests/test_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_judge_recovery_v5_control.py",
)

PRIOR_FILES = {
    "control/JUDGE_RECOVERY_V4_MANIFEST.json": (
        0o600, 8510,
        "ae81bee14cfe384d0d9656b83fa2e6846f30f73c2beb6c20bcbac0e2b300c54f",
        "cc7e42e1adfd802f98b83d9ccf0c67a2902b0033c5c99a38fc1578d3fa2bbcf8",
    ),
    "control/PREP.json": (
        0o600, 951,
        "300d208ebb3d9bffaed10190b3de270e646929cdfb7b964dc7a4e725a67c39f4",
        "4069aaf5463d363a0fd2232223a9650d75a64c73a76c982ec3caaf0995be86b7",
    ),
    "control/CPU_PREFLIGHT.json": (
        0o600, 1429,
        "abe918bcb3b8b215229c87df429632c2f0377edb744430811d75284d8a18d89b",
        "3272c6e25a22da6a0649e469ee1ab4bfe38d282859513aaf117b29c10e2695ab",
    ),
    "control/STAGED": (
        0o600, 1342,
        "8461f77f02741d47731b8ec2891597ac08a4ec85bb3426be5d58c2308835619e",
        "aa032bd357f3c6fc4097dc2a64661041a4360906666fef44de27b270429de707",
    ),
    "control/CANARY_LOCK_OWNER.json": (
        0o400, 1061,
        "0f7ae398b869e0c1df0072993e9c463a16bac3adc3dc0406d6abb3561e56f4a1",
        "7a162a98ab2442b989c58749f27c3354985550fef5ca85a5f38627ed9c08da52",
    ),
    "control/CANARY_AUTHORIZATION.json": (
        0o400, 1661,
        "37cb8affc6039453174832e7fef7e2cdd65462279552ed1649e3aa8859f0a4a2",
        "c9c82e8aa01045b8b1acb2572fcadd5dd175f60403b2c1d3c09f0da4f0f95d6d",
    ),
    "control/CANARY_FAILURE.json": (
        0o400, 2060,
        "ce57a34230596b9d964fb2e5843d09783f52782e0d5a10cb9e513fe66b87ee64",
        "69abff8985c42652c63169bc707ba58915665dd126cfa2d2dd3880642395f7c9",
    ),
    "logs/external_judge_canary.log": (
        0o400, 60,
        "b3495e33a6f564f5f29757694f3364c39378f2ea22f049d76c4168ad7dc1b615",
        None,
    ),
}

_real_print = builtins.print


def _v5_print(*values, **kwargs):
    converted = tuple(
        value.replace("JUDGE_RECOVERY_V4", "JUDGE_RECOVERY_V5")
        if isinstance(value, str) else value
        for value in values
    )
    _real_print(*converted, **kwargs)


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", os.fspath(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def audit_repo():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("Recovery repository is absent or unsafe")
    commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    if (
        _git(REPO_ROOT, "branch", "--show-current") != BRANCH
        or _git(REPO_ROOT, "status", "--porcelain")
        or _git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit)
        != f"{commit} {SOURCE_COMMIT}"
        or _git(REPO_ROOT, "rev-parse", f"{SOURCE_COMMIT}^{{tree}}") != SOURCE_TREE
    ):
        raise ValueError("Recovery repository lineage differs")
    observed = []
    for line in _git(
        REPO_ROOT, "diff", "--name-status", "--no-renames",
        f"{SOURCE_COMMIT}..{commit}",
    ).splitlines():
        if line:
            observed.append(tuple(line.split("\t")))
    if sorted(observed) != sorted(("A", path) for path in ADDED_FILES):
        raise ValueError("Recovery repository add-only scope differs")
    for relative in ADDED_FILES:
        entry = _git(REPO_ROOT, "ls-files", "-s", "--", relative).split()
        expected_mode = "100755" if relative.startswith("scripts/") else "100644"
        if len(entry) != 4 or entry[0] != expected_mode or entry[2] != "0":
            raise ValueError(f"Recovery index mode differs: {relative}")
    return {
        "path": os.fspath(REPO_ROOT), "branch": BRANCH, "commit": commit,
        "tree": _git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
        "source_commit": SOURCE_COMMIT, "source_commit_is_direct_parent": True,
        "add_only_files": list(ADDED_FILES),
    }


def _record(path, require_seal=True):
    payload = control.load_json(path) if require_seal else None
    if payload is not None:
        control.audit_seal(payload, os.fspath(path))
    return control.binding(path, require_seal=require_seal)


def audit_source():
    if PRIOR_REPO.is_symlink() or not PRIOR_REPO.is_dir():
        raise ValueError("Prior v4 repository is absent or unsafe")
    if (
        _git(PRIOR_REPO, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(PRIOR_REPO, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or not _git(PRIOR_REPO, "branch", "--show-current").endswith(
            "sequential-v1-judge-recovery-v4"
        )
        or _git(PRIOR_REPO, "status", "--porcelain")
    ):
        raise ValueError("Prior v4 repository binding differs")
    expected_dirs = {
        ".": {"control", "evaluation", "logs"},
        "control": {Path(name).name for name in PRIOR_FILES if name.startswith("control/")},
        "evaluation": {"medical", "final"},
        "evaluation/medical": set(), "evaluation/final": set(),
        "logs": {"external_judge_canary.log"},
    }
    for relative, names in expected_dirs.items():
        directory = PRIOR_RECOVERY_OUTPUT / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Prior v4 directory is absent or unsafe: {relative}")
        if {item.name for item in directory.iterdir()} != names:
            raise ValueError(f"Prior v4 exact inventory differs: {relative}")
    records = {}
    for relative, (mode, size, digest, payload_digest) in PRIOR_FILES.items():
        path = PRIOR_RECOVERY_OUTPUT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Prior v4 file is absent or unsafe: {relative}")
        if (
            stat.S_IMODE(path.stat().st_mode) != mode
            or path.stat().st_size != size
            or control.sha256_file(path) != digest
        ):
            raise ValueError(f"Prior v4 exact file binding differs: {relative}")
        record = _record(path, payload_digest is not None)
        if payload_digest is not None and record.get("payload_sha256") != payload_digest:
            raise ValueError(f"Prior v4 payload binding differs: {relative}")
        records[relative] = record
    failure_payload = control.load_json(
        PRIOR_RECOVERY_OUTPUT / "control/CANARY_FAILURE.json"
    )
    failure = control.audit_seal(failure_payload, "prior v4 canary failure")
    if (
        failure.get("protocol") != (
            SOURCE_PROTOCOL_ID + "_judge_recovery_v4_canary_failure_v1"
        )
        or failure.get("stage") != "canary"
        or failure.get("operation_stage") != "api_call"
        or failure.get("planned_index") != 0
        or failure.get("previously_completed_calls") != 0
        or failure.get("attempted_call_invocations_min") != 1
        or failure.get("attempted_call_invocations_max") != 1
        or failure.get("exception_class") != "RateLimitError"
        or failure.get("http_status") != 429
        or failure.get("error_code") != "credit_balance_exhausted"
        or failure.get("request_id") != "req_d41d2fa776d04e9184f3f9541cce3d87"
        or failure.get("terminal") is not True
        or failure.get("retry_authorized") is not False
        or failure.get("restart_or_resume_authorized") is not False
        or failure.get("contains_question_or_response_text") is not False
        or failure.get("contains_api_key_or_headers") is not False
    ):
        raise ValueError("Prior v4 credit-balance failure differs")
    prior_manifest_payload = control.load_json(
        PRIOR_RECOVERY_OUTPUT / "control/JUDGE_RECOVERY_V4_MANIFEST.json"
    )
    prior_manifest = control.audit_seal(prior_manifest_payload, "prior v4 manifest")
    return {
        "prior_manifest_body": prior_manifest,
        "failed_recovery_v4": {
            "recovery_id": SOURCE_PROTOCOL_ID + "_judge_recovery_v4",
            "repo": {
                "path": os.fspath(PRIOR_REPO),
                "branch": _git(PRIOR_REPO, "branch", "--show-current"),
                "commit": SOURCE_COMMIT, "tree": SOURCE_TREE,
            },
            "manifest": records["control/JUDGE_RECOVERY_V4_MANIFEST.json"],
            "prep": records["control/PREP.json"],
            "cpu_preflight": records["control/CPU_PREFLIGHT.json"],
            "staged": records["control/STAGED"],
            "canary_lock_owner": records["control/CANARY_LOCK_OWNER.json"],
            "canary_authorization": records["control/CANARY_AUTHORIZATION.json"],
            "canary_failure": records["control/CANARY_FAILURE.json"],
            "canary_log": records["logs/external_judge_canary.log"],
            "terminal_contract": {
                "stage": "canary", "operation_stage": "api_call",
                "planned_index": 0, "completed_calls": 0,
                "attempted_call_invocations_min": 1,
                "attempted_call_invocations_max": 1,
                "exception_class": "RateLimitError", "http_status": 429,
                "error_code": "credit_balance_exhausted",
                "request_id": "req_d41d2fa776d04e9184f3f9541cce3d87",
                "terminal": True, "retry_authorized": False,
                "restart_or_resume_authorized": False,
                "contains_question_or_response_text": False,
                "contains_api_key_or_headers": False,
            },
            "consumed_authorities": {
                "v3_external_judge_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
                "v4_canary_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
                "v4_continuation_authorized": False,
                "total_consumed_authority_cap_usd": PRIOR_CONSUMED_AUTHORITY_CAP_USD,
                "historical_attempts_min": 1, "historical_attempts_max": 2,
                "historical_accepted_judgments": 0, "reusable": False,
            },
        },
    }


def manifest_body(repo, source_state):
    prior = source_state["prior_manifest_body"]
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + "_manifest_v1",
        "recovery_id": RECOVERY_ID, "source_protocol_id": SOURCE_PROTOCOL_ID,
        "recovery_repo": repo,
        "source_output_root": prior["source_output_root"],
        "recovery_output_root": os.fspath(RECOVERY_OUTPUT),
        # Preserve the exact consumed-v3 scientific-source record; v5 adds a
        # separate exact v4 terminal binding below rather than rewriting it.
        "source_failure": prior["source_failure"],
        "source_protocol_manifest": prior["source_protocol_manifest"],
        "source_judge_plan": prior["source_judge_plan"],
        "source_artifacts": prior["source_artifacts"],
        "scientific_contract": {
            **prior["scientific_contract"],
            "idempotency_contract": IDEMPOTENCY_CONTRACT,
        },
        "budget_contract": {
            "verified_program_actual_before_unknown_api_usd": VERIFIED_PROGRAM_ACTUAL_USD,
            "consumed_v3_authority_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
            "consumed_v4_canary_authority_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
            "prior_consumed_authority_cap_usd": PRIOR_CONSUMED_AUTHORITY_CAP_USD,
            "prior_api_actual_cost_usd_known": False,
            "historical_attempts_min": 1, "historical_attempts_max": 2,
            "v4_continuation_authorized": False,
            "planned_v5_canary_cap_usd": CANARY_CAP_USD,
            "planned_v5_continuation_cap_usd": CONTINUATION_CAP_USD,
            "planned_v5_total_cap_usd": RECOVERY_TOTAL_CAP_USD,
            "conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
            "program_ceiling_usd": PROGRAM_CEILING_USD,
            "within_program_ceiling": True,
            "remaining_ceiling_gap_usd": 0.581742,
        },
        "failed_recovery_v4": source_state["failed_recovery_v4"],
        "external_api_authorized": False, "gpu_authorized": False,
        "cpu_stage_only": True,
    }


def expected_staged_files():
    return {
        "control": {
            "JUDGE_RECOVERY_V5_MANIFEST.json", "PREP.json", "CPU_PREFLIGHT.json",
            "STAGED",
        },
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    }


def seal_staged_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU staging")
    manifest = control.load_manifest()
    if os.path.lexists(PREFLIGHT_FILE) or os.path.lexists(STAGED_FILE):
        raise FileExistsError("CPU staging is already sealed")
    control.audit_namespace({
        "control": {"JUDGE_RECOVERY_V5_MANIFEST.json", "PREP.json"},
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
        "next_stage": "SEPARATELY_AUTHORIZED_ONE_CALL_CANARY",
    }))
    control.audit_staged()
    _v5_print(json.dumps({
        "status": "JUDGE_RECOVERY_V5_CPU_STAGED", "external_api_calls": 0,
        "gpu_jobs": 0, "next_stage": "SEPARATELY_AUTHORIZED_ONE_CALL_CANARY",
    }, sort_keys=True))
    return 0


def judge_module():
    import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5 as judge
    return judge


def require_authorization_acknowledgments(args, stage, canary_actual=None):
    expected = {
        "ack_verified_program_actual_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "ack_consumed_v3_authority_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "ack_consumed_v4_canary_authority_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "ack_prior_consumed_authority_cap_usd": PRIOR_CONSUMED_AUTHORITY_CAP_USD,
        "ack_prior_network_attempts_min": 1,
        "ack_prior_network_attempts_max": 2,
        "ack_max_cost_usd": CANARY_CAP_USD if stage == "canary" else CONTINUATION_CAP_USD,
        "ack_new_v5_total_cap_usd": RECOVERY_TOTAL_CAP_USD,
        "ack_conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "ack_program_ceiling_usd": PROGRAM_CEILING_USD,
    }
    for name, wanted in expected.items():
        observed = getattr(args, name)
        valid = observed == wanted if isinstance(wanted, int) else math.isclose(
            observed, wanted, rel_tol=0, abs_tol=1e-12
        )
        if not valid:
            raise ValueError(f"External authorization acknowledgment differs: {name}")
    if (
        args.ack_prior_api_actual_unknown is not True
        or args.ack_prior_authorities_consumed_not_reused is not True
        or args.ack_v4_continuation_never_authorized is not True
    ):
        raise ValueError("Historical consumed authorities were not acknowledged")
    if stage == "continuation" and (
        canary_actual is None
        or not math.isclose(
            args.ack_canary_actual_estimated_cost_usd, canary_actual,
            rel_tol=0, abs_tol=1e-12,
        )
    ):
        raise ValueError("V5 canary actual cost acknowledgment differs")


def add_authorization_arguments(parser):
    parser.add_argument("--stage", required=True, choices=("canary", "continuation"))
    parser.add_argument("--ack-verified-program-actual-usd", type=float, required=True)
    parser.add_argument("--ack-consumed-v3-authority-cap-usd", type=float, required=True)
    parser.add_argument("--ack-consumed-v4-canary-authority-cap-usd", type=float, required=True)
    parser.add_argument("--ack-prior-consumed-authority-cap-usd", type=float, required=True)
    parser.add_argument("--ack-prior-api-actual-unknown", action="store_true")
    parser.add_argument("--ack-prior-network-attempts-min", type=int, required=True)
    parser.add_argument("--ack-prior-network-attempts-max", type=int, required=True)
    parser.add_argument("--ack-prior-authorities-consumed-not-reused", action="store_true")
    parser.add_argument("--ack-v4-continuation-never-authorized", action="store_true")
    parser.add_argument("--ack-max-cost-usd", type=float, required=True)
    parser.add_argument("--ack-new-v5-total-cap-usd", type=float, required=True)
    parser.add_argument("--ack-conservative-program-max-usd", type=float, required=True)
    parser.add_argument("--ack-program-ceiling-usd", type=float, required=True)
    parser.add_argument("--ack-canary-actual-estimated-cost-usd", type=float)


def write_authorization_command(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be loaded before external authorization")
    stage = args.stage
    manifest = control.load_manifest()
    control.audit_lock(stage, manifest, args.owner_token)
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    canary_actual = None
    if stage == "canary":
        names = (
            "canary_authorization", "canary_failure", "canary_success",
            "continuation_authorization", "continuation_failure",
            "continuation_success", "judgments",
        )
        checkpoints = range(1, 241)
    else:
        canary = judge.load_canary_success(recovery, inputs, paths)
        canary_actual = canary["body"]["actual_estimated_cost_usd"]
        names = (
            "continuation_authorization", "continuation_failure",
            "continuation_success", "judgments",
        )
        checkpoints = range(2, 241)
    if any(os.path.lexists(paths[name]) for name in names):
        raise ValueError(f"{stage} authority namespace is not fresh")
    if any(os.path.lexists(judge.checkpoint_path(paths, item)) for item in checkpoints):
        raise ValueError(f"{stage} checkpoint namespace is not fresh")
    require_authorization_acknowledgments(args, stage, canary_actual)
    payload = control.sealed(judge.authorization_body(recovery, stage, paths))
    control.atomic_json(Path(paths[f"{stage}_authorization"]), payload, mode=0o400)
    judge.load_authorization(recovery, stage, paths)
    _v5_print(json.dumps({
        "status": f"JUDGE_RECOVERY_V5_{stage.upper()}_AUTHORIZED",
        "stage": stage, "authorized_calls": 1 if stage == "canary" else 239,
        "max_cost_usd": CANARY_CAP_USD if stage == "canary" else CONTINUATION_CAP_USD,
        "external_api_calls": 0,
        "authorization_payload_sha256": payload["payload_sha256"],
    }, sort_keys=True))
    return 0


def _configure_control():
    control.RECOVERY_ID = RECOVERY_ID
    control.SOURCE_PROTOCOL_ID = SOURCE_PROTOCOL_ID
    control.SOURCE_COMMIT = SOURCE_COMMIT
    control.BRANCH = BRANCH
    control.REPO_ROOT = REPO_ROOT
    control.RECOVERY_OUTPUT = RECOVERY_OUTPUT
    control.CONTROL_ROOT = CONTROL_ROOT
    control.MEDICAL_ROOT = MEDICAL_ROOT
    control.FINAL_ROOT = FINAL_ROOT
    control.LOG_ROOT = LOG_ROOT
    control.MANIFEST_FILE = MANIFEST_FILE
    control.PREP_FILE = PREP_FILE
    control.PREFLIGHT_FILE = PREFLIGHT_FILE
    control.STAGED_FILE = STAGED_FILE
    control.JUDGE_MODEL = JUDGE_MODEL
    control.PLAN_SHA256 = PLAN_SHA256
    control.RUBRIC_SHA256 = RUBRIC_SHA256
    control.SCHEMA_SHA256 = SCHEMA_SHA256
    control.VERIFIED_PROGRAM_ACTUAL_USD = VERIFIED_PROGRAM_ACTUAL_USD
    control.PRIOR_FAILED_AUTHORITY_CAP_USD = PRIOR_CONSUMED_AUTHORITY_CAP_USD
    control.CANARY_CAP_USD = CANARY_CAP_USD
    control.CONTINUATION_CAP_USD = CONTINUATION_CAP_USD
    control.RECOVERY_TOTAL_CAP_USD = RECOVERY_TOTAL_CAP_USD
    control.CONSERVATIVE_PROGRAM_MAX_USD = CONSERVATIVE_PROGRAM_MAX_USD
    control.PROGRAM_CEILING_USD = PROGRAM_CEILING_USD
    control.audit_repo = audit_repo
    control.audit_source = audit_source
    control.manifest_body = manifest_body
    control.expected_staged_files = expected_staged_files
    control.judge_module = judge_module
    control.require_authorization_acknowledgments = require_authorization_acknowledgments
    control.print = _v5_print


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
