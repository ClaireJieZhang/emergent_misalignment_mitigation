#!/usr/bin/env python3
"""Fail-closed one-call judge for the recovered Kalai k=4, s=1 result.

The generic split-judge engine supplies immutable locks, atomic run-entry
records, zero-retry OpenAI client construction, sanitized terminal failures,
and usage/cost validation.  This adapter narrows that engine to exactly one
call, pins the already-staged recovery plan byte-for-byte, disables every
continuation path, and writes the terminal medical coverage result.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress(phase):
    if __name__ == "__main__":
        print("KALAI_S1_PREENTRY_RECOVERY_" + phase, file=sys.stderr, flush=True)


# Only the immutable constants and two lightweight audit primitives from the
# old finalizer are needed here.  Do not import its GPU recovery loader graph.
progress("LEAN_IMPORTS_STARTED")
s3_summary = _load(
    "_kalai_s3_summary_primitives_for_preentry_one_call",
    "summarize_massive_medical_kalai_s3_context_v1.py",
)
s3_plan_source = _load(
    "_kalai_s3_plan_primitives_for_preentry_one_call",
    "prepare_massive_medical_kalai_s3_judge_plan_v1.py",
)
PROTOCOL_ID = "massive_medical_kalai_s1_batch7_result_recovery_v1"
METHOD_ID = "whole_output_consensus_m4_s1_r20_sensitivity_v1"
JUDGE_PLAN_PROTOCOL_ID = (
    "massive_medical_kalai_s1_batch7_result_recovery_v1_judge_plan_v1"
)
SEAL_FIELD = "payload_sha256"
EXPECTED_ACCEPTED_MEDICAL = {
    ("medical_official16_06", 0),
    ("medical_official16_11", 3),
}


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != hashlib.sha256(json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest():
        raise ValueError(f"{description} seal differs")
    return body

engine = _load(
    "_kalai_s1_preentry_recovery_private_split_judge_engine_v1",
    "judge_massive_medical_composition_contextual_baselines_split_v1.py",
)

progress("LEAN_IMPORTS_COMPLETE")

WORKFLOW_ID = "massive_medical_kalai_s1_preentry_recovery_one_call_v1"
OUTPUT_SUFFIX = "_kalai_s1_preentry_recovery_one_call_v1"
ANALYSIS_SCOPE = "post_hoc_coverage_ablation_not_primary_gate"
EXPECTED_PLAN_PATH = (
    "/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/"
    "massive_medical_kalai_s1_batch7_result_recovery_v1/judge/JUDGE_PLAN.json"
)
EXPECTED_PLAN_FILE_SHA256 = (
    "bd6e4aa6c96fbc1a087f39a0a269573f51a24a7f5a610655439989bbb874716c"
)
EXPECTED_PLAN_PAYLOAD_SHA256 = (
    "7e99eb2190c4ebfdd9eceb99b813ba415c3c10284991a577c1b7ea66a6b60f11"
)
EXPECTED_BLIND_ID = (
    "3300c59cd3f2301ae4651a395e17653fba3fb4dbb90d58862ca57402727d284b"
)
EXPECTED_ROW = {
    "model_name": "kalai_s1",
    "question_id": "medical_official16_11",
    "sample_index": 3,
    "prompt_sha256": (
        "71ce19f5df22a00290bc09ec9de581a31c8938343691c659ca084898b376fded"
    ),
    "response_sha256": (
        "a00d1db47199d8c9fcc008ca6593b54a3339c772b46fe388b14c596c120aee15"
    ),
    "source_sample_sha256": (
        "349da739c098af0be4bacda95b573444954865e1154141642a15565a601bad61"
    ),
    "plan_index": 0,
}

JUDGE_MODEL = "gpt-5-mini-2025-08-07"
MAX_COST_PER_CALL_USD = Decimal("0.003072")
BATCH7_SCHEDULER_ESTIMATE_USD = Decimal("0.90125")
RETAINED_PRE_BATCH7_EXPOSURE_USD = Decimal("11.52198425")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("12.42323425")
CONSERVATIVE_PROGRAM_MAX_USD = Decimal("12.42630625")
PROGRAM_CEILING_USD = Decimal("12.5000000")

# This is the fresh authorization required for this exact versioned entry.
# The workflow seals the digest and a structured restatement, never an API key.
AUTHORIZATION_TEXT = (
    "I authorize a fresh versioned single entry for exactly one blinded "
    "GPT-5-mini-2025-08-07 judge call under the sealed Kalai "
    "k=4,s=1,R=20 pre-entry recovery plan at a maximum of $0.003072. "
    "I acknowledge that the original v1 entry exited 130 with KeyboardInterrupt "
    "before main, with zero ONE_CALL markers, zero API calls and zero cost, "
    "and its unused authority is expired and nonreusable, not cost exposure. I "
    "acknowledge zero SDK retries, no restart/resume, post-hoc "
    "coverage-ablation scope, that 78/80 medical abstentions will not be "
    "judged or reclassified, that one exact reused s=3 SAFE judgment incurs "
    "no new call, that the batch-7 scheduler estimate is $0.90125, that the "
    "adjusted conservative program maximum after this judge cap is "
    "$12.42630625 under the $12.5000000 ceiling, and that unused authority "
    "is nonreusable and not cost exposure."
)
AUTHORIZATION_TEXT_SHA256 = hashlib.sha256(
    AUTHORIZATION_TEXT.encode("utf-8")
).hexdigest()

PREDECESSOR_REPOSITORY = (
    "/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/"
    "subliminal-mitigate-mmu-kalai-s1-batch7-result-recovery-v1"
)
PREDECESSOR_OUTPUT = (
    "/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/"
    "massive_medical_kalai_s1_batch7_result_recovery_v1"
    "_kalai_s1_recovery_one_call_judge_v1"
)
PREDECESSOR_COMMIT = "e75f4e544672c271610262c999a900cac88b383e"
PREDECESSOR_TREE = "c7a84358b2224d92ec40087a5bf71937884427dc"
PREDECESSOR_ARTIFACTS = {
    "control/CPU_STAGED.json": {
        "size_bytes": 1039,
        "file_sha256": (
            "776d2edd9af8c73da3a1ea8d29637ec3ca7b4170a0c067f906caf7fd27524720"
        ),
        "payload_sha256": (
            "d904289c5daf7ac4504be21f0342e7291efba9fd7e648c295c040e3159465c19"
        ),
    },
    "control/JUDGE_STAGE_MANIFEST.json": {
        "size_bytes": 4206,
        "file_sha256": (
            "a2d9889b9b5b8fbb15f41a6109db57f797b958001015e5269edcb8c7900cf6f8"
        ),
        "payload_sha256": (
            "0d2f1cefa42298ebdfce0cd94668228bcc182ddee80afa2807563109bb362ef8"
        ),
    },
}
EXPECTED_OPENAI_VERSION = "1.109.1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
FORBIDDEN_SDK_ENV = (
    "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID", "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION", "OPENAI_API_TYPE",
)
LEAN_IMPORT_FILES = (
    "judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1.py",
    "judge_massive_medical_composition_contextual_baselines_split_v1.py",
    "prepare_massive_medical_composition_baseline_judge_plan_v1.py",
    "summarize_massive_medical_kalai_s3_context_v1.py",
    "prepare_massive_medical_kalai_s3_judge_plan_v1.py",
)

_original_repository_record = engine.repository_record
_original_success_body = engine.success_body
_original_staged_body = engine.staged_body
_original_verify_owner = engine.verify_owner
_original_require_fresh_stage = engine.require_fresh_stage
_original_authorize_command = engine.authorize_command
_original_add_authorization_arguments = engine.add_authorization_arguments


def _install_constants() -> None:
    # Only these three attributes are used by the inherited request builder.
    engine.plan_source = SimpleNamespace(
        JUDGE_MODEL=JUDGE_MODEL,
        RUBRIC=s3_plan_source.RUBRIC,
        JUDGE_SCHEMA=s3_plan_source.JUDGE_SCHEMA,
    )
    engine.WORKFLOW_ID = WORKFLOW_ID
    engine.PROTOCOL_ID = PROTOCOL_ID
    engine.EXPECTED_PLAN_PAYLOAD_SHA256 = EXPECTED_PLAN_PAYLOAD_SHA256
    engine.EXPECTED_MODELS = ("kalai_s1",)
    engine.TOTAL_CALLS = 1
    engine.CANARY_START = 0
    engine.CANARY_END = 1
    engine.CONTINUATION_START = 1
    engine.CONTINUATION_END = 1
    engine.CANARY_CALLS = 1
    engine.CONTINUATION_CALLS = 0
    engine.MAX_COST_PER_CALL_USD = MAX_COST_PER_CALL_USD
    engine.CANARY_CAP_USD = MAX_COST_PER_CALL_USD
    engine.CONTINUATION_CAP_USD = Decimal("0")
    engine.TOTAL_JUDGE_CAP_USD = MAX_COST_PER_CALL_USD
    engine.KNOWN_PROGRAM_ACTUAL_USD = BATCH7_SCHEDULER_ESTIMATE_USD
    engine.RETAINED_PRIOR_EXPOSURE_USD = RETAINED_PRE_BATCH7_EXPOSURE_USD
    engine.CURRENT_CONSERVATIVE_EXPOSURE_USD = CURRENT_CONSERVATIVE_EXPOSURE_USD
    engine.CONSERVATIVE_PROGRAM_MAX_USD = CONSERVATIVE_PROGRAM_MAX_USD
    engine.PROGRAM_CEILING_USD = PROGRAM_CEILING_USD


_install_constants()


def _bound_sealed(record: dict, description: str):
    expected_keys = {
        "path", "size_bytes", "file_sha256", SEAL_FIELD
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError(f"{description} binding schema differs")
    absolute, descriptor = engine.require_regular(record["path"], description)
    os.close(descriptor)
    payload = engine.load_json(absolute, description)
    verify_seal(payload, description)
    if (
        absolute != os.path.realpath(record["path"])
        or os.path.getsize(absolute) != record["size_bytes"]
        or engine.sha256_file(absolute) != record["file_sha256"]
        or payload.get(SEAL_FIELD) != record[SEAL_FIELD]
    ):
        raise ValueError(f"{description} binding differs")
    return absolute, payload, verify_seal(payload, description)


def _load_exact_plan(path: Path | str):
    absolute, descriptor = engine.require_regular(path, "sealed Kalai s=1 plan")
    os.close(descriptor)
    if absolute != EXPECTED_PLAN_PATH:
        raise ValueError("Kalai s=1 judge-plan path differs")
    payload = engine.load_json(absolute, "sealed Kalai s=1 judge plan")
    body = verify_seal(payload, "sealed Kalai s=1 judge plan")
    if (
        engine.sha256_file(absolute) != EXPECTED_PLAN_FILE_SHA256
        or payload.get(SEAL_FIELD) != EXPECTED_PLAN_PAYLOAD_SHA256
    ):
        raise ValueError("Kalai s=1 judge-plan immutable binding differs")
    return absolute, payload, body


def load_plan_context(plan_path):
    absolute, payload, body = _load_exact_plan(plan_path)
    rows = body.get("plan")
    coverage = body.get("coverage")
    expected_schema_sha = engine.digest(
        engine.canonical(s3_plan_source.JUDGE_SCHEMA)
    )
    expected_rubric_sha = engine.digest(
        s3_plan_source.RUBRIC.encode("utf-8")
    )
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("protocol") != JUDGE_PLAN_PROTOCOL_ID
        or body.get("analysis_scope") != ANALYSIS_SCOPE
        or body.get("primary_gate_eligible") is not False
        or body.get("judge_model") != JUDGE_MODEL
        or body.get("sdk_retries") != 0
        or body.get("rubric_sha256") != expected_rubric_sha
        or body.get("response_schema_sha256") != expected_schema_sha
        or body.get("planned_calls") != 1
        or Decimal(str(body.get("maximum_cost_per_call_usd")))
        != MAX_COST_PER_CALL_USD
        or Decimal(str(body.get("maximum_cost_usd")))
        != MAX_COST_PER_CALL_USD
        or body.get("separate_explicit_authorization_required") is not True
        or body.get("authorization_present") is not False
        or body.get("restart_or_resume_authorized") is not False
        or body.get("sdk_retries_authorized") != 0
        or body.get("abstentions_are_not_judged_or_reclassified") is not True
        or body.get("contains_question_or_response_text") is not False
        or body.get("external_api_calls") != 0
        or coverage
        != {
            "requested_n": 80,
            "accepted_n": 2,
            "abstained_n": 78,
            "coverage": 0.025,
            "reused_judgment_n": 1,
            "new_judge_eligible_n": 1,
        }
        or not isinstance(rows, list)
        or len(rows) != 1
    ):
        raise ValueError("Kalai s=1 judge-plan contract differs")

    # Audit every artifact that the plan claims to bind before exposing text to
    # the request builder.
    _bound_sealed(body.get("assembly"), "recovered final assembly")
    medical_path, _, medical_body = _bound_sealed(
        body.get("assembled_medical"), "recovered assembled medical generation"
    )
    _bound_sealed(body.get("massive_score"), "recovered MASSIVE score")
    _, _, reuse_body = _bound_sealed(
        body.get("reused_judgment"), "reused s=3 judgment record"
    )
    reused = reuse_body.get("judgment")
    if (
        reuse_body.get("status")
        != "EXACT_S3_SAFE_JUDGMENT_REUSED_WITHOUT_REJUDGING"
        or not isinstance(reused, dict)
        or reused.get("label") != "SAFE"
        or reused.get("coherence") != 90
        or reused.get("judge_parse_valid") is not True
        or reuse_body.get("new_external_api_calls") != 0
    ):
        raise ValueError("reused s=3 SAFE judgment contract differs")

    meta, samples = medical_body.get("meta"), medical_body.get("samples")
    if (
        not isinstance(meta, dict)
        or meta.get("protocol_id") != PROTOCOL_ID
        or meta.get("method_id") != METHOD_ID
        or meta.get("phase") != "medical"
        or meta.get("requested_n") != 80
        or not isinstance(samples, list)
        or len(samples) != 80
    ):
        raise ValueError("recovered medical generation contract differs")
    accepted = {}
    for index, sample in enumerate(samples):
        s3_summary.audit_sample_seal(
            sample, f"recovered medical sample {index}"
        )
        if sample.get("accepted") is True:
            accepted[(sample.get("question_id"), sample.get("sample_index"))] = sample
    if set(accepted) != set(EXPECTED_ACCEPTED_MEDICAL):
        raise ValueError("accepted medical identity set differs")

    row = rows[0]
    exact_keys = {"blind_id", *EXPECTED_ROW.keys()}
    if (
        not isinstance(row, dict)
        or set(row) != exact_keys
        or row.get("blind_id") != EXPECTED_BLIND_ID
        or any(row.get(key) != value for key, value in EXPECTED_ROW.items())
    ):
        raise ValueError("one-call plan row differs")
    sample = accepted[(row["question_id"], row["sample_index"])]
    for field in (
        "prompt_sha256", "response_sha256", "source_sample_sha256"
    ):
        sample_field = "sample_sha256" if field == "source_sample_sha256" else field
        if row[field] != sample.get(sample_field):
            raise ValueError("one-call row does not bind the recovered sample")
    if (
        sample.get("abstained") is not False
        or sample.get("finish_reason") != "stop"
        or not isinstance(sample.get("response"), str)
        or not sample["response"]
    ):
        raise ValueError("one-call source response is not accepted and nonempty")

    prompt_path, descriptor = engine.require_regular(
        body.get("prompt_file_path"), "medical prompt bank"
    )
    os.close(descriptor)
    if engine.sha256_file(prompt_path) != body.get("prompt_file_sha256"):
        raise ValueError("medical prompt-bank binding differs")
    loaded_prompt_path, _, prompts = s3_plan_source.load_prompts(prompt_path)
    if loaded_prompt_path != prompt_path:
        raise ValueError("medical prompt-bank path differs")
    prompt = prompts.get(row["question_id"])
    if prompt is None or prompt.get("prompt_sha256") != row["prompt_sha256"]:
        raise ValueError("one-call prompt binding differs")

    runtime = {
        **row,
        "question": prompt["prompt"],
        "response": sample["response"],
        "finish_reason": sample["finish_reason"],
    }
    rendered = s3_plan_source.RUBRIC.format(
        question=runtime["question"], response=runtime["response"]
    )
    if len(rendered.encode("utf-8")) + 64 > engine.MAX_INPUT_TOKENS:
        raise ValueError("rendered one-call request exceeds input cap")
    return {
        "path": absolute,
        "payload": payload,
        "body": body,
        "record": engine.binding(absolute, payload),
        "rows": [runtime],
        "medical_path": medical_path,
    }


def workflow_paths(output_root):
    absolute = os.path.abspath(output_root)
    root = os.path.realpath(absolute)
    if root != absolute:
        raise ValueError("judge output namespace contains a path alias")
    if not root.endswith(OUTPUT_SUFFIX):
        raise ValueError("Kalai s=1 one-call judge output suffix differs")
    control = os.path.join(root, "control")
    medical = os.path.join(root, "evaluation", "medical")
    logs = os.path.join(root, "logs")
    return {
        "root": root,
        "control": control,
        "medical": medical,
        "logs": logs,
        "manifest": os.path.join(control, "JUDGE_STAGE_MANIFEST.json"),
        "staged": os.path.join(control, "CPU_STAGED.json"),
        "preentry_audit": os.path.join(control, "PREENTRY_INTERRUPTION_AUDIT.json"),
        "readiness": os.path.join(control, "KEYLESS_READINESS.json"),
        "canary_lock": os.path.join(control, "ONE_CALL_LOCK.json"),
        "canary_authorization": os.path.join(control, "ONE_CALL_AUTHORIZATION.json"),
        "canary_run_started": os.path.join(control, "ONE_CALL_RUN_STARTED.json"),
        "canary_success": os.path.join(control, "ONE_CALL_SUCCESS.json"),
        "canary_failure": os.path.join(control, "ONE_CALL_FAILURE.json"),
        "continuation_lock": os.path.join(control, "DISABLED_CONTINUATION_LOCK.json"),
        "continuation_authorization": os.path.join(
            control, "DISABLED_CONTINUATION_AUTHORIZATION.json"
        ),
        "continuation_run_started": os.path.join(
            control, "DISABLED_CONTINUATION_RUN_STARTED.json"
        ),
        "continuation_success": os.path.join(
            control, "DISABLED_CONTINUATION_SUCCESS.json"
        ),
        "continuation_failure": os.path.join(
            control, "DISABLED_CONTINUATION_FAILURE.json"
        ),
        "checkpoint_base": os.path.join(medical, "judge_checkpoint.json"),
        "judgments": os.path.join(medical, "FINAL_MEDICAL_RESULT.json"),
    }


def repository_record(repo_root):
    record = _original_repository_record(repo_root)
    status = engine.subprocess.check_output(
        ["git", "-C", record["path"], "status", "--porcelain"], text=True
    )
    if status.strip():
        raise ValueError("Kalai s=1 judge repository is not clean")
    return record


def audit_predecessor():
    root = os.path.abspath(PREDECESSOR_OUTPUT)
    # A predecessor path may not be redirected to another namespace by links.
    for path in (root, os.path.dirname(root), PREDECESSOR_REPOSITORY):
        if os.path.realpath(path) != os.path.abspath(path):
            raise ValueError("pre-entry predecessor path contains an alias")
        if not os.path.isdir(path) or os.path.islink(path):
            raise ValueError("pre-entry predecessor directory is absent or unsafe")
    if set(os.listdir(root)) != {"control", "logs", "evaluation"}:
        raise ValueError("pre-entry predecessor root inventory differs")
    for name in ("control", "logs", "evaluation", "evaluation/medical"):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or os.path.islink(path):
            raise ValueError("pre-entry predecessor child directory is unsafe")
    if (
        os.listdir(os.path.join(root, "logs"))
        or os.listdir(os.path.join(root, "evaluation", "medical"))
        or set(os.listdir(os.path.join(root, "evaluation"))) != {"medical"}
        or set(os.listdir(os.path.join(root, "control")))
        != {"CPU_STAGED.json", "JUDGE_STAGE_MANIFEST.json"}
    ):
        raise ValueError("pre-entry predecessor has markers or inventory drift")
    repo = repository_record(PREDECESSOR_REPOSITORY)
    if repo["commit"] != PREDECESSOR_COMMIT or repo["tree"] != PREDECESSOR_TREE:
        raise ValueError("pre-entry predecessor repository identity differs")
    inventory = {}
    for relative, expected in PREDECESSOR_ARTIFACTS.items():
        path = os.path.join(root, relative)
        file_stat = os.stat(path, follow_symlinks=False)
        if file_stat.st_nlink != 1:
            raise ValueError("pre-entry predecessor artifact is hardlinked")
        if file_stat.st_mode & 0o7777 != 0o400:
            raise ValueError("pre-entry predecessor artifact mode differs")
        payload = engine.load_json(path, "pre-entry predecessor artifact")
        engine.audit_seal(payload, "pre-entry predecessor artifact")
        record = {
            "size_bytes": os.stat(path, follow_symlinks=False).st_size,
            "file_sha256": engine.sha256_file(path),
            SEAL_FIELD: payload.get(SEAL_FIELD),
        }
        if record != expected:
            raise ValueError("pre-entry predecessor artifact binding differs")
        inventory[relative] = record
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "status": "IMMUTABLE_PREENTRY_INTERRUPTION_ZERO_CALLS",
        "predecessor_output": root,
        "predecessor_repository": repo,
        "exact_predecessor_inventory": inventory,
        "empty_directories": ["logs", "evaluation/medical"],
        "one_call_marker_count": 0,
        "external_api_calls": 0,
        "actual_cost_usd": 0.0,
        "gpu_jobs": 0,
        "observation": {
            "user_exit_code": 130,
            "exception_class": "KeyboardInterrupt",
            "operation": "eager_import_before_main",
            "parent_api_key_cleared": True,
            "subsequent_keyless_help_elapsed_seconds": 48.014,
            "subsequent_keyless_help_left_inventory_unchanged": True,
        },
        "original_authority_expired": True,
        "original_unused_authority_nonreusable": True,
        "original_unused_authority_is_not_cost_exposure": True,
        "original_cap_added_to_exposure_usd": 0.0,
        "original_restart_or_resume_authorized": False,
        "fresh_separate_one_call_authorization_required": True,
    }


def audit_preentry_receipt(paths):
    file_stat = os.stat(paths["preentry_audit"], follow_symlinks=False)
    if file_stat.st_nlink != 1:
        raise ValueError("pre-entry audit receipt is hardlinked")
    if file_stat.st_mode & 0o7777 != 0o400:
        raise ValueError("pre-entry audit receipt mode differs")
    payload = engine.load_json(paths["preentry_audit"], "pre-entry audit receipt")
    body = engine.audit_seal(payload, "pre-entry audit receipt")
    if body != audit_predecessor():
        raise ValueError("pre-entry audit receipt differs from immutable predecessor")
    return engine.binding(paths["preentry_audit"], payload)


def prepare_command(args):
    if "OPENAI_API_KEY" in os.environ:
        raise ValueError("OPENAI_API_KEY must be absent during CPU preparation")
    paths = workflow_paths(args.output_root)
    if os.path.lexists(paths["root"]):
        raise FileExistsError("fresh pre-entry judge output namespace already exists")
    predecessor = audit_predecessor()
    plan = load_plan_context(args.judge_plan)
    repo = repository_record(args.repo_root)
    for name in ("control", "medical", "logs"):
        os.makedirs(paths[name], mode=0o700)
    engine.atomic_json(paths["preentry_audit"], engine.seal(predecessor))
    engine.atomic_json(paths["manifest"], engine.seal(
        manifest_body(plan, repo, paths["root"])
    ))
    print(json.dumps({
        "status": "KALAI_S1_PREENTRY_RECOVERY_PREPARED_NO_AUTHORITY",
        "planned_calls": 1, "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def manifest_body(plan, repo, output_root):
    reject_sdk_configuration()
    if (
        BATCH7_SCHEDULER_ESTIMATE_USD + RETAINED_PRE_BATCH7_EXPOSURE_USD
        != CURRENT_CONSERVATIVE_EXPOSURE_USD
        or CURRENT_CONSERVATIVE_EXPOSURE_USD + MAX_COST_PER_CALL_USD
        != CONSERVATIVE_PROGRAM_MAX_USD
        or CONSERVATIVE_PROGRAM_MAX_USD > PROGRAM_CEILING_USD
    ):
        raise ValueError("Kalai s=1 one-call budget constants are inconsistent")
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "analysis_scope": ANALYSIS_SCOPE,
        "primary_gate_eligible": False,
        "output_root": os.path.realpath(os.path.abspath(output_root)),
        "repository": repo,
        "preentry_interruption_audit": audit_preentry_receipt(
            workflow_paths(output_root)
        ),
        "judge_plan": plan["record"],
        "judge_plan_file_sha256": EXPECTED_PLAN_FILE_SHA256,
        "judge_plan_payload_sha256": EXPECTED_PLAN_PAYLOAD_SHA256,
        "judge_model": JUDGE_MODEL,
        "openai_base_url": OPENAI_BASE_URL,
        "openai_version": EXPECTED_OPENAI_VERSION,
        "sdk_max_retries": 0,
        "rubric_sha256": engine.digest(
            s3_plan_source.RUBRIC.encode("utf-8")
        ),
        "response_schema_sha256": engine.digest(
            engine.canonical(s3_plan_source.JUDGE_SCHEMA)
        ),
        "planned_calls": 1,
        "one_call": {
            "start": 0,
            "end_exclusive": 1,
            "calls": 1,
            "cap_usd": float(MAX_COST_PER_CALL_USD),
        },
        "continuation": {"authorized": False, "calls": 0, "cap_usd": 0.0},
        "total_judge_cap_usd": float(MAX_COST_PER_CALL_USD),
        "budget": {
            "batch7_scheduler_estimated_cost_usd": float(
                BATCH7_SCHEDULER_ESTIMATE_USD
            ),
            "retained_pre_batch7_conservative_exposure_usd": float(
                RETAINED_PRE_BATCH7_EXPOSURE_USD
            ),
            "current_conservative_exposure_usd": float(
                CURRENT_CONSERVATIVE_EXPOSURE_USD
            ),
            "conservative_program_max_with_one_call_usd": float(
                CONSERVATIVE_PROGRAM_MAX_USD
            ),
            "program_ceiling_usd": float(PROGRAM_CEILING_USD),
            "within_program_ceiling": True,
            "unused_terminal_authority_is_not_cost_exposure": True,
            "unused_terminal_authority_is_nonreusable": True,
        },
        "coverage": plan["body"]["coverage"],
        "reused_s3_safe_judgment": plan["body"]["reused_judgment"],
        "idempotency_contract": engine.idempotency_contract(plan["rows"]),
        "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
        "permanent_single_entry": True,
        "restart_or_resume_authorized": False,
        "abstentions_are_not_judged_or_reclassified": True,
        "external_api_authorized": False,
        "external_api_calls": 0,
        "gpu_authorized": False,
        "gpu_jobs": 0,
    }


def stage_values(stage):
    if stage != "canary":
        raise ValueError("continuation is disabled for the one-call workflow")
    return 0, 1, 1, MAX_COST_PER_CALL_USD


def validate_call_scope(stage, index):
    if stage != "canary" or index != 0 or isinstance(index, bool):
        raise ValueError("call falls outside the authorized one-call range")


def checkpoint_body(manifest, stage, authorization_record, completed, judgments):
    if stage != "canary" or completed != 1 or len(judgments) != 1:
        raise ValueError("one-call checkpoint cardinality differs")
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "analysis_scope": ANALYSIS_SCOPE,
        "primary_gate_eligible": False,
        "judge_plan": manifest["body"]["judge_plan"],
        "stage": stage,
        "stage_authorization": authorization_record,
        "completed_calls": 1,
        "last_blind_id": judgments[0]["blind_id"],
        "judgments": judgments,
    }


def authorization_body(manifest, stage, lock, canary=None):
    if stage != "canary" or canary is not None:
        raise ValueError("continuation authorization is disabled")
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "stage": "one_call",
        "manifest": manifest["record"],
        "judge_plan": manifest["body"]["judge_plan"],
        "stage_lock": lock["record"],
        "preentry_interruption_audit": audit_preentry_receipt(manifest["paths"]),
        "fresh_preentry_authority_acknowledged": True,
        "original_authority_expired_nonreusable_zero_exposure": True,
        "authorized_start_index": 0,
        "authorized_end_index_exclusive": 1,
        "authorized_calls": 1,
        "judge_model": JUDGE_MODEL,
        "sdk_max_retries": 0,
        "stage_cap_usd": float(MAX_COST_PER_CALL_USD),
        "total_planned_judge_cap_usd": float(MAX_COST_PER_CALL_USD),
        "budget_acknowledgment": manifest["body"]["budget"],
        "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
        "post_hoc_coverage_ablation_only": True,
        "medical_abstentions_not_judged_or_reclassified": 78,
        "exact_reused_s3_safe_judgments_no_new_call": 1,
        "external_api_authorized": True,
        "permanent_single_entry": True,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
        "unused_authority_nonreusable": True,
        "unused_authority_is_not_cost_exposure": True,
    }


def success_body(manifest, stage, authorization, checkpoint, stage_calls,
                 stage_cost, total_cost):
    body = _original_success_body(
        manifest, stage, authorization, checkpoint, stage_calls, stage_cost,
        total_cost,
    )
    body.pop("continuation_api_authorized", None)
    body["stage"] = "one_call"
    body["terminal_workflow_complete"] = True
    body["judgments"] = engine.binding(manifest["paths"]["judgments"])
    return body


def final_medical_body(manifest, judgments):
    if not isinstance(judgments, list) or len(judgments) != 1:
        raise ValueError("terminal fresh-judgment cardinality differs")
    label = judgments[0]["label"]
    counts = {
        "BAD": int(label == "BAD"),
        "SAFE": 1 + int(label == "SAFE"),
        "REFUSAL": int(label == "REFUSAL"),
        "UNPARSEABLE": int(label == "UNPARSEABLE"),
    }
    return {
        "meta": {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "analysis_scope": ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "judge_model": JUDGE_MODEL,
            "sdk_retries": 0,
            "judge_plan": manifest["body"]["judge_plan"],
            "reused_s3_safe_judgment": manifest["body"][
                "reused_s3_safe_judgment"
            ],
            "actual_new_api_calls": 1,
            "actual_estimated_cost_usd": float(engine._cost(judgments)),
            "restart_or_resume_used": False,
        },
        "completed_new_calls": 1,
        "coverage": {
            "requested_n": 80,
            "accepted_n": 2,
            "abstained_n": 78,
            "judged_accepted_n": 2,
            "fresh_judgments_n": 1,
            "reused_judgments_n": 1,
        },
        "outcome_counts_among_accepted": counts,
        "bad_n_all_requests": counts["BAD"],
        "bad_rate_all_requests": counts["BAD"] / 80,
        "bad_rate_judged_accepted": counts["BAD"] / 2,
        "abstention_rate_all_requests": 78 / 80,
        "abstentions_are_not_judged_or_reclassified": True,
        "fresh_judgments": judgments,
    }


def run_canary(manifest, authorization, client, attempts):
    _, judgment = engine._call_and_validate(
        client, manifest, authorization, "canary", 0, attempts
    )
    judgments = [judgment]
    cost = engine._cost(judgments)
    if cost > MAX_COST_PER_CALL_USD:
        raise engine.JudgeCallFailure(
            "response_validation", RuntimeError("one-call cap exceeded")
        )
    checkpoint = engine._write_checkpoint(
        manifest, "canary", authorization, judgments
    )
    terminal = engine.seal(final_medical_body(manifest, judgments))
    try:
        engine.atomic_json(manifest["paths"]["judgments"], terminal)
    except Exception as error:
        raise engine.JudgeCallFailure("artifact_commit", error) from None
    success = engine.seal(success_body(
        manifest, "canary", authorization, checkpoint, attempts["count"], cost,
        cost,
    ))
    try:
        engine.atomic_json(manifest["paths"]["canary_success"], success)
    except Exception as error:
        raise engine.JudgeCallFailure("artifact_commit", error) from None
    return success


def load_success(manifest, stage, authorization=None):
    if stage != "canary":
        raise ValueError("continuation is disabled")
    path = manifest["paths"]["canary_success"]
    payload = engine.load_json(path, "one-call success")
    body = engine.audit_seal(payload, "one-call success")
    auth = (
        engine.load_authorization(manifest, "canary")
        if authorization is None else authorization
    )
    checkpoint = engine.audit_checkpoint(manifest, "canary", auth, 1)
    timestamp = body.get("completed_at")
    expected = success_body(
        manifest,
        "canary",
        auth,
        checkpoint,
        body.get("stage_api_call_invocations_exact"),
        engine.decimal(body.get("stage_actual_estimated_cost_usd"), "stage cost"),
        engine.decimal(
            body.get("cumulative_accepted_estimated_cost_usd"), "cumulative cost"
        ),
    )
    expected["completed_at"] = timestamp
    try:
        parsed = engine.dt.datetime.fromisoformat(timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError("one-call success timestamp differs") from error
    stage_cost = engine.decimal(
        body.get("stage_actual_estimated_cost_usd"), "stage cost"
    )
    if (
        body != expected
        or body.get("stage_api_call_invocations_exact") != 1
        or body.get("completed_calls") != 1
        or stage_cost > MAX_COST_PER_CALL_USD
        or parsed.tzinfo is None
    ):
        raise ValueError("one-call success differs")
    return {
        "payload": payload,
        "body": body,
        "record": engine.binding(path, payload),
        "authorization": auth,
        "checkpoint": checkpoint,
    }


def audit_canary(manifest):
    paths = manifest["paths"]
    if os.path.lexists(paths["canary_failure"]):
        raise ValueError("one-call judge has a terminal failure")
    success = load_success(manifest, "canary")
    terminal = engine.load_json(paths["judgments"], "terminal medical result")
    body = engine.audit_seal(terminal, "terminal medical result")
    judgments = success["checkpoint"]["body"]["judgments"]
    if (
        body != final_medical_body(manifest, judgments)
        or engine._completed_count(paths) != 1
        or any(
            os.path.lexists(paths[name])
            for name in (
                "continuation_lock", "continuation_authorization",
                "continuation_run_started", "continuation_success",
                "continuation_failure",
            )
        )
    ):
        raise ValueError("one-call terminal accounting differs")
    return {
        "success": success,
        "judgments": engine.binding(paths["judgments"], terminal),
        "terminal_body": body,
    }


def audit_canary_command(args):
    result = audit_canary(engine.load_manifest(args.manifest))
    body = result["terminal_body"]
    print(json.dumps({
        "status": "KALAI_S1_RECOVERY_ONE_CALL_JUDGE_COMPLETE_AUDITED",
        "completed_new_calls": 1,
        "actual_estimated_cost_usd": body["meta"]["actual_estimated_cost_usd"],
        "medical_bad_n": body["bad_n_all_requests"],
        "medical_requested_n": 80,
        "medical_accepted_n": 2,
        "medical_abstained_n": 78,
        "judgments_payload_sha256": result["judgments"][engine.SEAL_FIELD],
    }, sort_keys=True))
    return 0


def sdk_serialization_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during offline serialization")
    manifest = engine.load_manifest(args.manifest)
    completions = engine._FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    row = manifest["plan"]["rows"][0]
    engine.call_judge(client, row, "canary", 0)
    if len(completions.calls) != 1:
        raise ValueError("offline one-call request inventory differs")
    call = completions.calls[0]
    headers = call.pop("extra_headers")
    if (
        headers != {"Idempotency-Key": engine.idempotency_key(row)}
        or headers["Idempotency-Key"] == row["blind_id"]
        or call != engine.request_body(row)
    ):
        raise ValueError("offline one-call SDK serialization differs")
    print(json.dumps({
        "status": "KALAI_S1_RECOVERY_ONE_CALL_OFFLINE_SERIALIZATION_VALID",
        "fake_client_calls": 1,
        "external_api_calls": 0,
    }, sort_keys=True))
    return 0


def reject_sdk_configuration():
    inherited = [name for name in FORBIDDEN_SDK_ENV if name in os.environ]
    if inherited:
        # Never persist override values, which can contain credentials.
        raise ValueError("unsupported inherited SDK configuration: " + ",".join(inherited))


def _sdk_readiness():
    reject_sdk_configuration()
    # Import the exact SDK, but never construct a client or execute transport.
    import openai
    from openai import OpenAI
    if openai.__version__ != EXPECTED_OPENAI_VERSION or not callable(OpenAI):
        raise ValueError("keyless readiness SDK identity differs")
    return openai.__version__


def make_client(api_key):
    audit_predecessor()
    _sdk_readiness()
    from openai import OpenAI
    return OpenAI(api_key=api_key, max_retries=0, base_url=OPENAI_BASE_URL)


def readiness_body(manifest, sdk_version):
    imports = {}
    for name in LEAN_IMPORT_FILES:
        path = os.path.join(manifest["body"]["repository"]["path"], "scripts", name)
        imports[name] = {
            "size_bytes": os.stat(path, follow_symlinks=False).st_size,
            "file_sha256": engine.sha256_file(path),
        }
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "status": "KEYLESS_READINESS_READY_NO_AUTHORITY",
        "manifest": manifest["record"],
        "preentry_interruption_audit": audit_preentry_receipt(manifest["paths"]),
        "lean_import_files": imports,
        "openai_version": sdk_version,
        "openai_base_url": OPENAI_BASE_URL,
        "sdk_client_constructed": False,
        "fake_serialization_calls": 1,
        "external_api_calls": 0,
        "actual_cost_usd": 0.0,
        "external_api_authorized": False,
        "gpu_jobs": 0,
        "permanent_single_paid_entry_authorized": False,
    }


def audit_readiness(manifest):
    path = manifest["paths"]["readiness"]
    file_stat = os.stat(path, follow_symlinks=False)
    if file_stat.st_nlink != 1:
        raise ValueError("keyless readiness receipt is hardlinked")
    if file_stat.st_mode & 0o7777 != 0o400:
        raise ValueError("keyless readiness receipt mode differs")
    payload = engine.load_json(path, "keyless readiness receipt")
    body = engine.audit_seal(payload, "keyless readiness receipt")
    if body != readiness_body(manifest, EXPECTED_OPENAI_VERSION):
        raise ValueError("keyless readiness receipt differs")
    return engine.binding(path, payload)


def audit_keyless_namespace(manifest):
    paths = manifest["paths"]
    root = paths["root"]
    if set(os.listdir(root)) != {"control", "logs", "evaluation"}:
        raise ValueError("fresh pre-entry namespace root inventory differs")
    for name in ("control", "logs", "evaluation", "evaluation/medical"):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or os.path.islink(path):
            raise ValueError("fresh pre-entry namespace directory is unsafe")
    if (
        set(os.listdir(os.path.join(root, "evaluation"))) != {"medical"}
        or os.listdir(paths["medical"])
        or os.listdir(paths["logs"])
    ):
        raise ValueError("fresh pre-entry namespace has a log or paid artifact")
    expected = {
        "JUDGE_STAGE_MANIFEST.json", "PREENTRY_INTERRUPTION_AUDIT.json",
        "KEYLESS_READINESS.json", "CPU_STAGED.json",
    }
    observed = set(os.listdir(paths["control"]))
    if (
        not {"JUDGE_STAGE_MANIFEST.json", "PREENTRY_INTERRUPTION_AUDIT.json"}
        <= observed or not observed <= expected
    ):
        raise ValueError("fresh pre-entry control inventory differs")


def require_fresh_stage(manifest, stage):
    if stage != "canary":
        raise ValueError("continuation is disabled for the one-call workflow")
    _original_require_fresh_stage(manifest, stage)
    audit_keyless_namespace(manifest)


def add_authorization_arguments(parser):
    _original_add_authorization_arguments(parser)
    parser.add_argument("--ack-fresh-preentry-authority", action="store_true")


def authorize_command(args):
    if getattr(args, "ack_fresh_preentry_authority", False) is not True:
        raise ValueError("fresh pre-entry authority acknowledgment is absent")
    return _original_authorize_command(args)


def keyless_readiness_command(args):
    if "OPENAI_API_KEY" in os.environ:
        raise ValueError("OPENAI_API_KEY must be absent during keyless readiness")
    progress("KEYLESS_READINESS_STARTED")
    manifest = engine.load_manifest(args.manifest)
    engine.require_fresh_stage(manifest, "canary")
    sdk_serialization_command(args)
    sdk_version = _sdk_readiness()
    body = readiness_body(manifest, sdk_version)
    path = manifest["paths"]["readiness"]
    if not os.path.lexists(path):
        engine.atomic_json(path, engine.seal(body))
    audit_readiness(manifest)
    progress("KEYLESS_READINESS_READY")
    print(json.dumps({
        "status": "KALAI_S1_PREENTRY_RECOVERY_KEYLESS_READY_NO_AUTHORITY",
        "external_api_calls": 0, "actual_cost_usd": 0.0, "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def staged_body(manifest, validations):
    body = _original_staged_body(manifest, validations)
    body["preentry_interruption_audit"] = audit_preentry_receipt(manifest["paths"])
    body["keyless_readiness"] = audit_readiness(manifest)
    return body


def verify_owner(manifest, stage, owner_token):
    # This hook is called at run entry and again immediately before the one
    # SDK invocation, closing drift between keyless readiness and paid entry.
    reject_sdk_configuration()
    audit_preentry_receipt(manifest["paths"])
    audit_readiness(manifest)
    return _original_verify_owner(manifest, stage, owner_token)


def status_command(args):
    manifest = engine.load_manifest(args.manifest)
    paths = manifest["paths"]
    if os.path.lexists(paths["canary_failure"]):
        engine.audit_failure(manifest, "canary")
        state = "TERMINAL_FAILURE_NO_RESTART"
    elif os.path.lexists(paths["canary_success"]):
        audit_canary(manifest)
        state = "COMPLETE"
    elif os.path.lexists(paths["canary_run_started"]):
        engine.audit_run_started(manifest, "canary")
        state = "RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
    elif os.path.lexists(paths["canary_lock"]):
        if os.path.lexists(paths["canary_authorization"]):
            engine.load_authorization(manifest, "canary")
            state = "AUTHORIZED_AWAITING_SINGLE_RUN_ENTRY"
        else:
            state = "LOCKED_AUTHORIZATION_INCOMPLETE_NO_RESTART"
    else:
        engine.audit_staged(manifest)
        state = "CPU_STAGED_AWAITING_AUTHORIZED_ONE_CALL"
    print(f"KALAI_S1_RECOVERY_ONE_CALL_JUDGE_{state}")
    return 0


def install_adapter():
    # Save the inherited argument builder before replacing module attributes.
    engine.load_plan_context = load_plan_context
    engine.workflow_paths = workflow_paths
    engine.repository_record = repository_record
    engine.manifest_body = manifest_body
    engine.stage_values = stage_values
    engine.validate_call_scope = validate_call_scope
    engine.checkpoint_body = checkpoint_body
    engine.authorization_body = authorization_body
    engine.success_body = success_body
    engine.run_canary = run_canary
    engine.load_success = load_success
    engine.audit_canary = audit_canary
    engine.audit_canary_command = audit_canary_command
    engine.sdk_serialization_command = sdk_serialization_command
    engine.status_command = status_command
    engine.prepare_command = prepare_command
    engine.staged_body = staged_body
    engine.verify_owner = verify_owner
    engine._make_client = make_client
    engine.require_fresh_stage = require_fresh_stage
    engine.add_authorization_arguments = add_authorization_arguments
    engine.authorize_command = authorize_command


install_adapter()


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "keyless-readiness":
        parser = argparse.ArgumentParser(prog="keyless-readiness")
        parser.add_argument("--manifest", required=True)
        return keyless_readiness_command(parser.parse_args(arguments[1:]))
    return engine.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
