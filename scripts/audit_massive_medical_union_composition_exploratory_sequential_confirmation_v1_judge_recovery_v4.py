#!/usr/bin/env python3
"""CPU-only control plane for the split external-judge recovery v4."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


RECOVERY_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1_"
    "judge_recovery_v4"
)
SOURCE_PROTOCOL_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
)
SOURCE_COMMIT = "3e3cb2749e0e16bd5a31fd62cdff050812278e57"
BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-recovery-v4"
)
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
REPO_ROOT = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v4"
)
SOURCE_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_submit_recovery_v3"
)
RECOVERY_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4"
)
CONTROL_ROOT = RECOVERY_OUTPUT / "control"
MEDICAL_ROOT = RECOVERY_OUTPUT / "evaluation/medical"
FINAL_ROOT = RECOVERY_OUTPUT / "evaluation/final"
LOG_ROOT = RECOVERY_OUTPUT / "logs"
MANIFEST_FILE = CONTROL_ROOT / "JUDGE_RECOVERY_V4_MANIFEST.json"
PREP_FILE = CONTROL_ROOT / "PREP.json"
PREFLIGHT_FILE = CONTROL_ROOT / "CPU_PREFLIGHT.json"
STAGED_FILE = CONTROL_ROOT / "STAGED"

JUDGE_MODEL = "gpt-5-mini-2025-08-07"
PLAN_SHA256 = "93110f70dbebcb7031dcf8e7be0d1c15e925347af70e8105e78d6d50daa49140"
RUBRIC_SHA256 = "ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e"
SCHEMA_SHA256 = "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777"
VERIFIED_PROGRAM_ACTUAL_USD = 2.915186
PRIOR_FAILED_AUTHORITY_CAP_USD = 0.75
CANARY_CAP_USD = 0.003072
CONTINUATION_CAP_USD = 0.746928
RECOVERY_TOTAL_CAP_USD = 0.75
PROGRAM_CEILING_USD = 5.0
CONSERVATIVE_PROGRAM_MAX_USD = 4.415186

SOURCE_FILES = {
    "protocol_manifest": (
        "protocol/manifest.json", 47479,
        "5ff9bacfa1880931c0c14a76cfff4dc613d48a9a0e7fd94dad14aa2262134d19",
    ),
    "historical_A_judgments": (
        "protocol/historical/A_judgments.json", 232178,
        "359a8e2351c855bceaea8400cb97a32f62a82f64f7b13b09839a120746a94ca2",
    ),
    "stop": (
        "control/STOPPED_external_judge", 174,
        "9cfd8f34199c82f7a7c603338bf571cae5018a4f902ffbda7774b18a90f9abad",
    ),
    "lock_owner": (
        "control/FINALIZER_LOCK/owner", 159,
        "2e708a3b75215365af4b058131952100c99d34b1db0ba61fd58994780d571677",
    ),
    "authorization": (
        "control/EXTERNAL_JUDGE_AUTHORIZATION.json", 4016,
        "ea99d9cb5f3f784791fd291f672594bdf0d4c45a7f20d7f6c23242914d617bd8",
    ),
    "base_checkpoint": (
        "evaluation/medical/judge_checkpoint.json", 922,
        "d02b345f44919c6c21313dc922345d2d5e847437f1573c1bd6a0fe0be09d1889",
    ),
    "judge_plan": (
        "evaluation/medical/judge_plan.json", 3614,
        "f2c2bec177615179151093ffb7ac07eb00d41358756a134994b82d7c5a0b6a7e",
    ),
    "prejudge_gate": (
        "evaluation/medical/prejudge/AWAITING_EXTERNAL_JUDGE", 982,
        "3b71acecf0b9e83e47569362124f9a14ac4e49810a6716974d5644eb9194ff67",
    ),
    "prejudge_summary": (
        "evaluation/medical/prejudge/summary.json", 11848,
        "7a5f3feb8df39a8020da797f95e9e260902ee0017bf38ef4ab7f2377202f5cb4",
    ),
    "benefit_gate": (
        "evaluation/benefit/gate/EXPLORATORY_BENEFIT_PASSED", 966,
        "8646a1af6a593632c4d89d4efadd49b29591264b5be6c5914fe43c030ca8ddb6",
    ),
    "ordinary_quorum_m4_q3": (
        "generation/medical/ordinary_quorum_m4_q3/medical/generation.json", 102806,
        "004f3b49ee9687a30f619b5ba099046c904822b8fbea3a7dcd5491694ad954ee",
    ),
    "ordinary_min_m4_q4": (
        "generation/medical/ordinary_min_m4_q4/medical/generation.json", 102257,
        "0a0074484074d5b710db7c8c8d0a375379369bd361278d3c65dc0d3a9691aad1",
    ),
    "delta_min_m4_q4": (
        "generation/medical/delta_min_m4_q4/medical/generation.json", 105668,
        "8b02fd8990d7f69e21c61a8549f56774437f29c212dcd2972ec2f706bc2d1a05",
    ),
}


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sealed(body):
    return {**body, "payload_sha256": sha256_bytes(canonical_bytes(body))}


def audit_seal(value, context, field="payload_sha256"):
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not an object")
    body = {key: item for key, item in value.items() if key != field}
    if value.get(field) != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{context} payload seal differs")
    return body


def open_regular(path, mode="rb"):
    path = Path(path).absolute()
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Refusing nonregular or symlink input: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    after = os.fstat(descriptor)
    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev, after.st_ino
    ):
        os.close(descriptor)
        raise ValueError(f"Input changed during secure open: {path}")
    if mode == "rb":
        return os.fdopen(descriptor, "rb")
    return os.fdopen(descriptor, "r", encoding="utf-8")


def read_bytes(path):
    with open_regular(path, "rb") as handle:
        return handle.read()


def load_json(path):
    with open_regular(path, "r") as handle:
        return json.load(handle)


def sha256_file(path):
    return sha256_bytes(read_bytes(path))


def valid_utc_timestamp(value):
    if (
        not isinstance(value, str)
        or len(value) > 40
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\+00:00",
            value,
        ) is None
    ):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def atomic_json(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def binding(
    path, expected_size=None, expected_sha256=None, require_seal=False,
    seal_field="payload_sha256",
):
    path = Path(path).absolute()
    size = path.stat().st_size
    digest = sha256_file(path)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Source size differs: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source digest differs: {path}")
    result = {"path": os.fspath(path), "size": size, "file_sha256": digest}
    if require_seal:
        value = load_json(path)
        audit_seal(value, os.fspath(path), seal_field)
        result[seal_field] = value[seal_field]
    return result


def git(*args):
    return subprocess.run(
        ["git", "-C", os.fspath(REPO_ROOT), *args], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def audit_repo():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("Recovery repository is absent or unsafe")
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("Recovery branch differs")
    if git("status", "--porcelain"):
        raise ValueError("Recovery repository is dirty")
    subprocess.run(
        ["git", "-C", os.fspath(REPO_ROOT), "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return {
        "path": os.fspath(REPO_ROOT), "branch": BRANCH,
        "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
        "source_commit": SOURCE_COMMIT, "source_commit_is_ancestor": True,
    }


def audit_source():
    observed = {}
    sealed_json = {
        "protocol_manifest", "historical_A_judgments", "authorization",
        "base_checkpoint", "judge_plan", "prejudge_gate", "prejudge_summary",
        "benefit_gate", "ordinary_quorum_m4_q3", "ordinary_min_m4_q4",
        "delta_min_m4_q4",
    }
    for name, (relative, size, digest) in SOURCE_FILES.items():
        observed[name] = binding(
            SOURCE_OUTPUT / relative, size, digest, name in sealed_json,
            "manifest_payload_sha256" if name == "protocol_manifest" else "payload_sha256",
        )
    expected_stop = (
        "workflow_id=massive_medical_union_composition_exploratory_sequential_confirmation_v1\n"
        "stage=external_judge\nexit_code=1\nretry_authorized=false\n"
        "process_restart_authorized=false\n"
    ).encode()
    if read_bytes(SOURCE_OUTPUT / SOURCE_FILES["stop"][0]) != expected_stop:
        raise ValueError("Source STOP marker content differs")
    checkpoint = audit_seal(
        load_json(SOURCE_OUTPUT / SOURCE_FILES["base_checkpoint"][0]),
        "source base checkpoint",
    )
    if (
        checkpoint.get("status") != "PERMANENT_SINGLE_ENTRY_STARTED"
        or checkpoint.get("planned_calls") != 240
        or checkpoint.get("restart_or_resume_authorized") is not False
        or checkpoint.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("Source base checkpoint contract differs")
    plan = audit_seal(
        load_json(SOURCE_OUTPUT / SOURCE_FILES["judge_plan"][0]), "source judge plan"
    )
    if (
        plan.get("plan_sha256") != PLAN_SHA256
        or plan.get("planned_calls") != 240
        or plan.get("judge_model") != JUDGE_MODEL
        or plan.get("rubric_sha256") != RUBRIC_SHA256
        or plan.get("response_schema_sha256") != SCHEMA_SHA256
        or plan.get("contains_question_or_response_text") is not False
    ):
        raise ValueError("Source judge plan contract differs")
    authorization = audit_seal(
        load_json(SOURCE_OUTPUT / SOURCE_FILES["authorization"][0]),
        "source authorization",
    )
    if (
        authorization.get("planned_calls") != 240
        or authorization.get("judge_model") != JUDGE_MODEL
        or authorization.get("sdk_max_retries") != 0
        or authorization.get("permanent_single_entry") is not True
        or authorization.get("restart_or_resume_authorized") is not False
    ):
        raise ValueError("Source authorization contract differs")
    numbered = list((SOURCE_OUTPUT / "evaluation/medical").glob("judge_checkpoint.json.*"))
    if numbered:
        raise ValueError("Source unexpectedly has a completed judgment checkpoint")
    for relative in (
        "evaluation/medical/judgments_new.json", "control/FINAL_RESULT.json",
    ):
        if os.path.lexists(SOURCE_OUTPUT / relative):
            raise ValueError(f"Source unexpectedly has terminal artifact: {relative}")
    return observed


def manifest_body(repo, source):
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_manifest_v1",
        "recovery_id": RECOVERY_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "recovery_repo": repo,
        "source_output_root": os.fspath(SOURCE_OUTPUT),
        "recovery_output_root": os.fspath(RECOVERY_OUTPUT),
        "source_failure": {
            "stop": source["stop"], "lock_owner": source["lock_owner"],
            "authorization": source["authorization"],
            "base_checkpoint": source["base_checkpoint"],
            "permanent_single_entry_consumed": True,
            "prior_completed_judgments": 0,
            "prior_network_attempts_min": 0,
            "prior_network_attempts_max": 1,
            "prior_api_actual_cost_usd_known": False,
            "prior_authority_reusable": False,
        },
        "source_protocol_manifest": source["protocol_manifest"],
        "source_judge_plan": {**source["judge_plan"], "plan_sha256": PLAN_SHA256},
        "source_artifacts": {
            key: source[key] for key in (
                "historical_A_judgments", "prejudge_gate", "prejudge_summary",
                "benefit_gate", "ordinary_quorum_m4_q3", "ordinary_min_m4_q4",
                "delta_min_m4_q4",
            )
        },
        "scientific_contract": {
            "judge_model": JUDGE_MODEL, "plan_sha256": PLAN_SHA256,
            "rubric_sha256": RUBRIC_SHA256,
            "response_schema_sha256": SCHEMA_SHA256,
            "accepted_judgments": 240,
            "canary_calls": 1, "canary_start_index": 0,
            "canary_end_index_exclusive": 1,
            "continuation_calls": 239, "continuation_start_index": 1,
            "continuation_end_index_exclusive": 240,
            "sdk_max_retries": 0, "same_plan_order": True,
            "model_fallback_authorized": False,
            "historical_A_reused_not_rejudged": True,
        },
        "budget_contract": {
            "verified_program_actual_before_prior_unknown_api_usd": VERIFIED_PROGRAM_ACTUAL_USD,
            "prior_failed_authority_cap_usd": PRIOR_FAILED_AUTHORITY_CAP_USD,
            "prior_api_actual_cost_usd_known": False,
            "planned_canary_cap_usd": CANARY_CAP_USD,
            "planned_continuation_cap_usd": CONTINUATION_CAP_USD,
            "planned_recovery_total_cap_usd": RECOVERY_TOTAL_CAP_USD,
            "conservative_program_max_if_prior_cap_fully_charged_usd": CONSERVATIVE_PROGRAM_MAX_USD,
            "program_ceiling_usd": PROGRAM_CEILING_USD,
            "within_program_ceiling": True,
        },
        "external_api_authorized": False,
        "gpu_authorized": False,
        "cpu_stage_only": True,
    }


def load_manifest():
    value = load_json(MANIFEST_FILE)
    body = audit_seal(value, "recovery manifest")
    repo = audit_repo()
    source = audit_source()
    if body != manifest_body(repo, source):
        raise ValueError("Recovery manifest differs from live frozen inputs")
    return value


def prepare_command(_args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU staging")
    if os.path.lexists(RECOVERY_OUTPUT):
        raise FileExistsError("Fresh recovery output namespace already exists")
    repo = audit_repo()
    source = audit_source()
    os.umask(0o077)
    for directory in (CONTROL_ROOT, MEDICAL_ROOT, FINAL_ROOT, LOG_ROOT):
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = sealed(manifest_body(repo, source))
    atomic_json(MANIFEST_FILE, manifest)
    atomic_json(PREP_FILE, sealed({
        "schema_version": 1, "protocol": RECOVERY_ID + "_prep_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(MANIFEST_FILE, require_seal=True),
        "source_plan_sha256": PLAN_SHA256,
        "external_api_calls": 0, "gpu_jobs": 0,
        "status": "CPU_PREPARED_AWAITING_VALIDATION",
    }))
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_CPU_PREPARED", "external_api_calls": 0,
        "gpu_jobs": 0, "manifest_payload_sha256": manifest["payload_sha256"],
    }, sort_keys=True))
    return 0


def expected_staged_files():
    return {
        "control": {
            "JUDGE_RECOVERY_V4_MANIFEST.json", "PREP.json", "CPU_PREFLIGHT.json", "STAGED",
        },
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    }


def audit_namespace(expected):
    if RECOVERY_OUTPUT.is_symlink() or not RECOVERY_OUTPUT.is_dir():
        raise ValueError("Recovery output root is absent or unsafe")
    for relative, names in expected.items():
        directory = RECOVERY_OUTPUT / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Recovery directory is absent or unsafe: {relative}")
        observed = {item.name for item in directory.iterdir()}
        if observed != names:
            raise ValueError(f"Recovery namespace differs at {relative}")
        for item in directory.iterdir():
            if item.is_symlink():
                raise ValueError(f"Recovery namespace contains symlink: {item}")


def seal_staged_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU staging")
    manifest = load_manifest()
    if os.path.lexists(PREFLIGHT_FILE) or os.path.lexists(STAGED_FILE):
        raise FileExistsError("CPU staging is already sealed")
    audit_namespace({
        "control": {"JUDGE_RECOVERY_V4_MANIFEST.json", "PREP.json"},
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    })
    commands = args.validation_command or []
    if not commands:
        raise ValueError("At least one successful validation command is required")
    preflight = sealed({
        "schema_version": 1, "protocol": RECOVERY_ID + "_cpu_preflight_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(MANIFEST_FILE, require_seal=True),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "validation_commands_passed": commands,
        "network_validation": "mock_transport_only",
        "external_api_calls": 0, "gpu_jobs": 0,
        "api_key_required": False,
        "status": "CPU_VALIDATED_AWAITING_SEPARATE_CANARY_AUTHORIZATION",
    })
    atomic_json(PREFLIGHT_FILE, preflight)
    atomic_json(STAGED_FILE, sealed({
        "schema_version": 1, "protocol": RECOVERY_ID + "_staged_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(MANIFEST_FILE, require_seal=True),
        "cpu_preflight": binding(PREFLIGHT_FILE, require_seal=True),
        "external_api_authorized": False, "external_api_calls": 0,
        "gpu_authorized": False, "gpu_jobs": 0,
        "next_stage": "SEPARATELY_AUTHORIZED_ONE_CALL_CANARY",
    }))
    audit_staged()
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_CPU_STAGED", "external_api_calls": 0,
        "gpu_jobs": 0, "next_stage": "SEPARATELY_AUTHORIZED_ONE_CALL_CANARY",
    }, sort_keys=True))
    return 0


def audit_staged():
    manifest = load_manifest()
    prep = load_json(PREP_FILE); audit_seal(prep, "recovery prep")
    preflight = load_json(PREFLIGHT_FILE); audit_seal(preflight, "CPU preflight")
    staged = load_json(STAGED_FILE); audit_seal(staged, "staged sentinel")
    manifest_binding = binding(MANIFEST_FILE, require_seal=True)
    if (
        prep.get("recovery_manifest") != manifest_binding
        or preflight.get("recovery_manifest") != manifest_binding
        or staged.get("recovery_manifest") != manifest_binding
        or preflight.get("external_api_calls") != 0
        or preflight.get("gpu_jobs") != 0
        or staged.get("external_api_authorized") is not False
        or staged.get("gpu_authorized") is not False
        or manifest.get("external_api_authorized") is not False
    ):
        raise ValueError("CPU-staged control contract differs")
    audit_namespace(expected_staged_files())
    return manifest


def audit_staged_command(_args):
    manifest = audit_staged()
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_CPU_STAGED_VALID",
        "manifest_payload_sha256": manifest["payload_sha256"],
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def judge_module():
    return importlib.import_module(
        "judge_massive_medical_union_composition_exploratory_sequential_"
        "confirmation_v1_judge_recovery_v4"
    )


def lock_path(stage):
    if stage not in {"canary", "continuation"}:
        raise ValueError("Unknown external judge stage")
    return CONTROL_ROOT / f"{stage.upper()}_LOCK_OWNER.json"


def lock_body(stage, manifest, owner_token_sha256):
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_lock_v1",
        "recovery_id": RECOVERY_ID,
        "stage": stage,
        "recovery_manifest": binding(MANIFEST_FILE, require_seal=True),
        "recovery_repo_commit": manifest["recovery_repo"]["commit"],
        "owner_token_sha256": owner_token_sha256,
        "permanent_single_entry": True,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def audit_lock(stage, manifest, owner_token=None):
    owner = lock_path(stage)
    payload = load_json(owner)
    body = audit_seal(payload, f"{stage} lock owner")
    token_sha256 = body.get("owner_token_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", token_sha256 or "") is None:
        raise ValueError(f"{stage} lock owner token differs")
    if owner_token is not None:
        if re.fullmatch(r"[0-9a-f]{64}", owner_token) is None:
            raise ValueError("Lock owner token schema differs")
        if sha256_bytes(owner_token.encode()) != token_sha256:
            raise ValueError(f"{stage} lock is owned by another invocation")
    if body != lock_body(stage, manifest, token_sha256):
        raise ValueError(f"{stage} lock owner differs")
    return binding(owner, require_seal=True)


def acquire_lock_command(args):
    stage = args.stage
    # This repeats the complete read-only acknowledgment/state preflight inside
    # the lock command, so a malformed authorization can never consume a lock.
    manifest = authorization_preflight(args)
    owner = lock_path(stage)
    if re.fullmatch(r"[0-9a-f]{64}", args.owner_token) is None:
        raise ValueError("Lock owner token schema differs")
    atomic_json(
        owner,
        sealed(lock_body(
            stage, manifest, sha256_bytes(args.owner_token.encode())
        )),
        mode=0o400,
    )
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{stage.upper()}_LOCKED",
        "stage": stage, "external_api_calls": 0,
    }, sort_keys=True))
    return 0


def audit_lock_command(args):
    manifest = load_manifest()
    record = audit_lock(args.stage, manifest, args.owner_token)
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{args.stage.upper()}_LOCK_AUDITED",
        "stage": args.stage, "external_api_calls": 0,
        "lock_file_sha256": record["file_sha256"],
    }, sort_keys=True))
    return 0


def require_authorization_acknowledgments(args, stage, canary_actual=None):
    maximum = CANARY_CAP_USD if stage == "canary" else CONTINUATION_CAP_USD
    numeric = {
        "ack_verified_program_actual_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "ack_prior_failed_authority_cap_usd": PRIOR_FAILED_AUTHORITY_CAP_USD,
        "ack_prior_network_attempts_max": 1,
        "ack_max_cost_usd": maximum,
        "ack_cumulative_new_cap_usd": RECOVERY_TOTAL_CAP_USD,
        "ack_conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "ack_program_ceiling_usd": PROGRAM_CEILING_USD,
    }
    for name, expected in numeric.items():
        observed = getattr(args, name)
        if isinstance(expected, int):
            valid = observed == expected
        else:
            valid = math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12)
        if not valid:
            raise ValueError(f"External authorization acknowledgment differs: {name}")
    if (
        args.ack_prior_api_actual_unknown is not True
        or args.ack_prior_authority_consumed_not_reused is not True
    ):
        raise ValueError("Prior failed-attempt uncertainty was not acknowledged")
    if stage == "continuation":
        if canary_actual is None or not math.isclose(
            args.ack_canary_actual_estimated_cost_usd,
            canary_actual, rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError("Canary actual cost acknowledgment differs")


def authorization_preflight(args):
    """Validate stage state and every user acknowledgment without writing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be loaded before external authorization")
    stage = args.stage
    if os.path.lexists(lock_path(stage)):
        raise FileExistsError(f"{stage} permanent lock namespace is not fresh")
    manifest = audit_staged() if stage == "canary" else load_manifest()
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    canary_actual = None
    if stage == "canary":
        for name in (
            "canary_authorization", "canary_failure", "canary_success",
            "continuation_authorization", "continuation_failure",
            "continuation_success", "judgments",
        ):
            if os.path.lexists(paths[name]):
                raise ValueError("Canary authority namespace is not fresh")
        for completed in range(1, 241):
            if os.path.lexists(judge.checkpoint_path(paths, completed)):
                raise ValueError("Canary checkpoint namespace is not fresh")
    else:
        if os.path.lexists(paths["canary_failure"]):
            raise ValueError("Failed canary cannot consume continuation authority")
        canary = judge.load_canary_success(recovery, inputs, paths)
        canary_actual = canary["body"]["actual_estimated_cost_usd"]
        for name in (
            "continuation_authorization", "continuation_failure",
            "continuation_success", "judgments",
        ):
            if os.path.lexists(paths[name]):
                raise ValueError("Continuation authority namespace is not fresh")
        for completed in range(2, 241):
            if os.path.lexists(judge.checkpoint_path(paths, completed)):
                raise ValueError("Continuation checkpoint namespace is not fresh")
    require_authorization_acknowledgments(args, stage, canary_actual)
    return manifest


def preflight_authorization_command(args):
    authorization_preflight(args)
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{args.stage.upper()}_AUTHORIZATION_PREFLIGHT_VALID",
        "stage": args.stage, "external_api_calls": 0,
        "permanent_lock_consumed": False,
    }, sort_keys=True))
    return 0


def write_authorization_command(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be loaded before external authorization")
    stage = args.stage
    manifest = load_manifest()
    audit_lock(stage, manifest)
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    canary_actual = None
    if stage == "canary":
        for name in (
            "canary_authorization", "canary_failure", "canary_success",
            "continuation_authorization", "continuation_failure", "continuation_success",
            "judgments",
        ):
            if os.path.lexists(paths[name]):
                raise ValueError("Canary authority namespace is not fresh")
        for completed in range(1, 241):
            if os.path.lexists(judge.checkpoint_path(paths, completed)):
                raise ValueError("Canary checkpoint namespace is not fresh")
    else:
        canary = judge.load_canary_success(recovery, inputs, paths)
        canary_actual = canary["body"]["actual_estimated_cost_usd"]
        for name in (
            "continuation_authorization", "continuation_failure",
            "continuation_success", "judgments",
        ):
            if os.path.lexists(paths[name]):
                raise ValueError("Continuation authority namespace is not fresh")
        for completed in range(2, 241):
            if os.path.lexists(judge.checkpoint_path(paths, completed)):
                raise ValueError("Continuation checkpoint namespace is not fresh")
    require_authorization_acknowledgments(args, stage, canary_actual)
    authorization_path = Path(paths[f"{stage}_authorization"])
    payload = sealed(judge.authorization_body(recovery, stage, paths))
    atomic_json(authorization_path, payload, mode=0o400)
    judge.load_authorization(recovery, stage, paths)
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{stage.upper()}_AUTHORIZED",
        "stage": stage,
        "authorized_calls": 1 if stage == "canary" else 239,
        "max_cost_usd": CANARY_CAP_USD if stage == "canary" else CONTINUATION_CAP_USD,
        "external_api_calls": 0,
        "authorization_payload_sha256": payload["payload_sha256"],
    }, sort_keys=True))
    return 0


def wrapper_failure_body(
    stage, manifest, recovery, paths, exit_code, owner_token
):
    judge = judge_module()
    completed = judge._completed_checkpoint_count(paths)
    stage_prior = 0 if stage == "canary" else 1
    accepted_stage_calls = max(0, completed - stage_prior)
    stage_maximum = 1 if stage == "canary" else 239
    authorization_path = paths[f"{stage}_authorization"]
    authorization = None
    if os.path.lexists(authorization_path):
        authorization = judge._authorization_binding(
            judge.load_authorization(recovery, stage, paths)
        )
    log_path = LOG_ROOT / f"external_judge_{stage}.log"
    log_record = binding(log_path) if os.path.lexists(log_path) else None
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_wrapper_failure_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(MANIFEST_FILE, require_seal=True),
        "stage": stage,
        "stage_lock": audit_lock(stage, manifest, owner_token),
        "stage_authorization": authorization,
        "wrapper_exit_code": exit_code,
        "previously_completed_calls": completed,
        "attempted_call_invocations_min": accepted_stage_calls,
        "attempted_call_invocations_max": min(
            stage_maximum, accepted_stage_calls + 1
        ),
        "durable_log": log_record,
        "failure_recorded_at": judge.utc_now(),
        "contains_question_or_response_text": False,
        "contains_api_key_or_headers": False,
        "model_fallback_used": False,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
        "terminal": True,
    }


def write_wrapper_failure_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent before sealing wrapper failure")
    if isinstance(args.exit_code, bool) or not 1 <= args.exit_code <= 255:
        raise ValueError("Wrapper exit code differs")
    manifest = load_manifest()
    audit_lock(args.stage, manifest, args.owner_token)
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    failure_path = Path(paths[f"{args.stage}_failure"])
    success_path = paths[f"{args.stage}_success"]
    if os.path.lexists(success_path):
        raise ValueError("Successful stage cannot be marked as wrapper failure")
    if os.path.lexists(failure_path):
        audit_failure(args.stage)
        return 0
    atomic_json(
        failure_path,
        sealed(wrapper_failure_body(
            args.stage, manifest, recovery, paths, args.exit_code,
            args.owner_token,
        )),
        mode=0o400,
    )
    audit_failure(args.stage)
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{args.stage.upper()}_WRAPPER_FAILURE_SEALED",
        "stage": args.stage, "terminal": True,
        "retry_authorized": False, "restart_or_resume_authorized": False,
    }, sort_keys=True))
    return 0


def audit_failure(stage):
    manifest = load_manifest()
    audit_lock(stage, manifest)
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    failure_path = Path(paths[f"{stage}_failure"])
    success_path = paths[f"{stage}_success"]
    if os.path.lexists(success_path):
        raise ValueError("Failure and success artifacts cannot coexist")
    payload = load_json(failure_path)
    body = audit_seal(payload, f"{stage} failure")
    protocol = body.get("protocol")
    expected_manifest = binding(MANIFEST_FILE, require_seal=True)
    expected_authorization = None
    authorization_record = None
    authorization_path = paths[f"{stage}_authorization"]
    if os.path.lexists(authorization_path):
        authorization_record = judge.load_authorization(recovery, stage, paths)
        expected_authorization = judge._authorization_binding(authorization_record)
    live_completed = judge._completed_checkpoint_count(paths)
    checkpoint_names = {
        name for name in os.listdir(paths["medical"])
        if name.startswith("judge_checkpoint.json.")
    }
    expected_checkpoint_names = {
        f"judge_checkpoint.json.{index:03d}"
        for index in range(1, live_completed + 1)
    }
    if checkpoint_names != expected_checkpoint_names:
        raise ValueError(f"{stage} failure checkpoint inventory differs")
    for index in range(1, live_completed + 1):
        if stat.S_IMODE(os.stat(judge.checkpoint_path(paths, index)).st_mode) != 0o400:
            raise ValueError(f"{stage} failure checkpoint mode differs")
    if stat.S_IMODE(os.stat(failure_path).st_mode) != 0o400:
        raise ValueError(f"{stage} failure artifact mode differs")
    if stage == "canary":
        if live_completed > 1 or (live_completed and authorization_record is None):
            raise ValueError("Canary failure checkpoint state differs")
        if live_completed:
            judge.audit_checkpoint(
                judge.checkpoint_path(paths, 1), recovery, inputs,
                "canary", authorization_record, 1,
            )
    else:
        canary = judge.load_canary_success(recovery, inputs, paths)
        previous_judgments = list(canary["checkpoint"]["body"]["judgments"])
        if live_completed < 1 or (live_completed > 1 and authorization_record is None):
            raise ValueError("Continuation failure checkpoint state differs")
        for index in range(2, live_completed + 1):
            checkpoint = judge.audit_checkpoint(
                judge.checkpoint_path(paths, index), recovery, inputs,
                "continuation", authorization_record, index,
            )
            if checkpoint["body"]["judgments"][:-1] != previous_judgments:
                raise ValueError("Failure checkpoint prefix differs")
            previous_judgments = list(checkpoint["body"]["judgments"])
    if os.path.lexists(paths["judgments"]):
        if (
            stage != "continuation" or live_completed != 240
            or authorization_record is None
        ):
            raise ValueError("Failure has an unexpected terminal judgments artifact")
        if stat.S_IMODE(os.stat(paths["judgments"]).st_mode) != 0o400:
            raise ValueError("Failure terminal judgments mode differs")
        judgments_payload = judge.load_json(paths["judgments"])
        judgments_body = judge.audit_seal(judgments_payload, paths["judgments"])
        terminal_rows = previous_judgments
        expected_cost = sum(
            row["api_usage"]["estimated_cost_usd"] for row in terminal_rows
        )
        expected_judgments = {
            "meta": {
                **judge.judge_meta(recovery, inputs),
                "canary_authorization": judge._authorization_binding(
                    canary["authorization"]
                ),
                "continuation_authorization": expected_authorization,
                "actual_api_calls": 240,
                "canary_api_calls": 1,
                "continuation_api_calls": 239,
                "actual_estimated_cost_usd": expected_cost,
            },
            "judgments": sorted(
                terminal_rows,
                key=lambda row: (
                    row["model_name"], row["question_id"], row["sample_index"]
                ),
            ),
        }
        if judgments_body != expected_judgments:
            raise ValueError("Failure terminal judgments differ")
    maximum = 1 if stage == "canary" else 239
    completed_minimum = 0 if stage == "canary" else 1
    stage_prior = completed_minimum
    completed = body.get("previously_completed_calls")
    attempted_min = body.get("attempted_call_invocations_min")
    attempted_max = body.get("attempted_call_invocations_max")
    common_valid = (
        body.get("schema_version") == 1
        and body.get("recovery_id") == RECOVERY_ID
        and body.get("stage") == stage
        and body.get("stage_authorization") == expected_authorization
        and body.get("contains_question_or_response_text") is False
        and body.get("contains_api_key_or_headers") is False
        and body.get("model_fallback_used") is False
        and body.get("retry_authorized") is False
        and body.get("restart_or_resume_authorized") is False
        and body.get("terminal") is True
        and isinstance(completed, int) and not isinstance(completed, bool)
        and completed == live_completed
        and isinstance(attempted_min, int) and not isinstance(attempted_min, bool)
        and isinstance(attempted_max, int) and not isinstance(attempted_max, bool)
        and max(0, completed - stage_prior) <= attempted_min
        <= attempted_max <= min(maximum, max(0, completed - stage_prior) + 1)
    )
    if not common_valid:
        raise ValueError(f"{stage} failure common contract differs")
    if protocol == RECOVERY_ID + f"_{stage}_failure_v1":
        expected_keys = {
            "schema_version", "protocol", "recovery_id", "recovery_manifest",
            "stage", "stage_authorization", "operation_stage", "planned_index",
            "previously_completed_calls", "attempted_call_invocations_min",
            "attempted_call_invocations_max", "attempt_started_at",
            "failure_recorded_at", "exception_class", "http_status", "error_code",
            "request_id", "api_response_id", "api_response_model",
            "error_message_safe", "error_message_sha256",
            "contains_question_or_response_text", "contains_api_key_or_headers",
            "model_fallback_used", "retry_authorized",
            "restart_or_resume_authorized", "terminal",
        }
        expected_recovery = judge.binding(
            recovery["path"], judge.load_json(recovery["path"])
        )
        message = body.get("error_message_safe")
        http_status = body.get("http_status")
        if (
            set(body) != expected_keys
            or expected_authorization is None
            or body.get("recovery_manifest") != expected_recovery
            or body.get("operation_stage") not in {
                "environment_preflight", "client_initialization", "api_call",
                "response_validation", "artifact_commit",
            }
            or not valid_utc_timestamp(body.get("failure_recorded_at"))
            or not valid_utc_timestamp(body.get("attempt_started_at"))
            or body.get("planned_index")
            != (live_completed if live_completed < 240 else None)
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
                http_status is None
                or (
                    isinstance(http_status, int)
                    and not isinstance(http_status, bool)
                    and 100 <= http_status <= 599
                )
            )
            or re.fullmatch(r"[0-9a-f]{64}", body.get("error_message_sha256", "")) is None
            or not (
                message is None
                or (
                    isinstance(message, str) and len(message) <= 1000
                    and judge.safe_error_message(message) == message
                )
            )
        ):
            raise ValueError(f"{stage} core failure contract differs")
    elif protocol == RECOVERY_ID + f"_{stage}_wrapper_failure_v1":
        expected_keys = {
            "schema_version", "protocol", "recovery_id", "recovery_manifest",
            "stage", "stage_lock", "stage_authorization", "wrapper_exit_code",
            "previously_completed_calls", "attempted_call_invocations_min",
            "attempted_call_invocations_max", "durable_log",
            "failure_recorded_at", "contains_question_or_response_text",
            "contains_api_key_or_headers", "model_fallback_used",
            "retry_authorized", "restart_or_resume_authorized", "terminal",
        }
        if (
            set(body) != expected_keys
            or body.get("recovery_manifest") != expected_manifest
            or body.get("stage_lock") != audit_lock(stage, manifest)
            or not valid_utc_timestamp(body.get("failure_recorded_at"))
            or isinstance(body.get("wrapper_exit_code"), bool)
            or not isinstance(body.get("wrapper_exit_code"), int)
            or not 1 <= body["wrapper_exit_code"] <= 255
        ):
            raise ValueError(f"{stage} wrapper failure contract differs")
        log_record = body.get("durable_log")
        log_path = LOG_ROOT / f"external_judge_{stage}.log"
        if log_record is None:
            if os.path.lexists(log_path):
                raise ValueError("Wrapper failure omitted an existing durable log")
        elif log_record != binding(log_path):
            raise ValueError("Wrapper failure durable log binding differs")
    else:
        raise ValueError(f"{stage} failure protocol differs")
    return {"payload": payload, "body": body, "record": binding(failure_path, require_seal=True)}


def audit_failure_command(args):
    record = audit_failure(args.stage)
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{args.stage.upper()}_FAILURE_AUDITED",
        "stage": args.stage, "terminal": True,
        "failure_payload_sha256": record["payload"]["payload_sha256"],
        "external_api_calls_during_audit": 0,
    }, sort_keys=True))
    return 0


def audit_authorization_command(args):
    manifest = load_manifest()
    audit_lock(args.stage, manifest)
    judge = judge_module()
    recovery = judge.load_recovery_manifest(MANIFEST_FILE)
    inputs = judge.validate_source_inputs(recovery)
    paths = judge.recovery_paths(recovery)
    record = judge.load_authorization(recovery, args.stage, paths)
    if args.stage == "continuation":
        judge.load_canary_success(recovery, inputs, paths)
    print(json.dumps({
        "status": f"JUDGE_RECOVERY_V4_{args.stage.upper()}_AUTHORIZATION_VALID",
        "stage": args.stage, "external_api_calls": 0,
        "authorization_payload_sha256": record["payload_sha256"],
    }, sort_keys=True))
    return 0


def add_authorization_arguments(parser):
    parser.add_argument("--stage", required=True, choices=("canary", "continuation"))
    parser.add_argument("--ack-verified-program-actual-usd", type=float, required=True)
    parser.add_argument("--ack-prior-failed-authority-cap-usd", type=float, required=True)
    parser.add_argument("--ack-prior-network-attempts-max", type=int, required=True)
    parser.add_argument("--ack-prior-api-actual-unknown", action="store_true")
    parser.add_argument("--ack-prior-authority-consumed-not-reused", action="store_true")
    parser.add_argument("--ack-max-cost-usd", type=float, required=True)
    parser.add_argument("--ack-cumulative-new-cap-usd", type=float, required=True)
    parser.add_argument("--ack-conservative-program-max-usd", type=float, required=True)
    parser.add_argument("--ack-program-ceiling-usd", type=float, required=True)
    parser.add_argument("--ack-canary-actual-estimated-cost-usd", type=float)


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.set_defaults(handler=prepare_command)
    stage = commands.add_parser("seal-staged")
    stage.add_argument("--validation-command", action="append")
    stage.set_defaults(handler=seal_staged_command)
    audit = commands.add_parser("audit-staged")
    audit.set_defaults(handler=audit_staged_command)
    lock = commands.add_parser("acquire-lock")
    add_authorization_arguments(lock)
    lock.add_argument("--owner-token", required=True)
    lock.set_defaults(handler=acquire_lock_command)
    lock_audit = commands.add_parser("audit-lock")
    lock_audit.add_argument(
        "--stage", required=True, choices=("canary", "continuation")
    )
    lock_audit.add_argument("--owner-token")
    lock_audit.set_defaults(handler=audit_lock_command)
    auth_preflight = commands.add_parser("preflight-authorization")
    add_authorization_arguments(auth_preflight)
    auth_preflight.set_defaults(handler=preflight_authorization_command)
    authorize = commands.add_parser("write-authorization")
    add_authorization_arguments(authorize)
    authorize.set_defaults(handler=write_authorization_command)
    auth_audit = commands.add_parser("audit-authorization")
    auth_audit.add_argument("--stage", required=True, choices=("canary", "continuation"))
    auth_audit.set_defaults(handler=audit_authorization_command)
    wrapper_failure = commands.add_parser("write-wrapper-failure")
    wrapper_failure.add_argument(
        "--stage", required=True, choices=("canary", "continuation")
    )
    wrapper_failure.add_argument("--exit-code", required=True, type=int)
    wrapper_failure.add_argument("--owner-token", required=True)
    wrapper_failure.set_defaults(handler=write_wrapper_failure_command)
    failure_audit = commands.add_parser("audit-failure")
    failure_audit.add_argument(
        "--stage", required=True, choices=("canary", "continuation")
    )
    failure_audit.set_defaults(handler=audit_failure_command)
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
