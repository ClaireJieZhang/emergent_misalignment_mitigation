#!/usr/bin/env python3
"""Two-authority recovery for the sealed sequential medical judge plan.

This overlay never mutates the consumed submit-recovery-v3 namespace.  It
reconstructs that namespace's exact blinded 240-row plan, accepts one new API
call under a permanent canary authority, and only then accepts a separately
authorized 239-call continuation.  Static and plan validation import no API
client and make no network calls.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import re
import stat

import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as source


RECOVERY_ID = source.PROTOCOL_ID + "_judge_recovery_v4"
EXPECTED_PLAN_SHA256 = "93110f70dbebcb7031dcf8e7be0d1c15e925347af70e8105e78d6d50daa49140"
EXPECTED_SOURCE_COMMIT = "3e3cb2749e0e16bd5a31fd62cdff050812278e57"
CANARY_CALLS = 1
CONTINUATION_CALLS = 239
CANARY_START = 0
CONTINUATION_START = 1
TOTAL_CALLS = 240
CANARY_MAX_COST_USD = 0.003072
CONTINUATION_MAX_COST_USD = 0.746928
CUMULATIVE_NEW_API_CAP_USD = 0.75
VERIFIED_PROGRAM_ACTUAL_USD = 2.915186
CONSERVATIVE_PROGRAM_MAX_USD = 4.415186
PROGRAM_CEILING_USD = 5.0
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.:/-]{1,256}\Z")


def canonical_bytes(value):
    return source.canonical_bytes(value)


def sha256_bytes(value):
    return source.sha256_bytes(value)


def sha256_file(path):
    return source.sha256_file(path)


def load_json(path):
    return source.load_json(path)


def seal(body):
    return source.seal(body)


def audit_seal(payload, context):
    return source.audit_seal(payload, context)


def atomic_json(path, payload):
    return source.atomic_json(path, payload)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def binding(path, payload=None, seal_field="payload_sha256"):
    absolute = os.path.abspath(path)
    result = {
        "path": absolute, "size": os.stat(absolute).st_size,
        "file_sha256": sha256_file(absolute),
    }
    if payload is not None:
        result[seal_field] = payload[seal_field]
    return result


def require_binding(record, context, seal_field="payload_sha256", extra_keys=()):
    if not isinstance(record, dict):
        raise ValueError(f"{context} binding is not an object")
    expected_keys = {"path", "size", "file_sha256", *extra_keys}
    if seal_field is not None:
        expected_keys.add(seal_field)
    if set(record) != expected_keys:
        raise ValueError(f"{context} binding schema differs")
    path = record.get("path")
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError(f"{context} path is not absolute")
    payload = load_json(path) if seal_field is not None else None
    if (
        isinstance(record.get("size"), bool)
        or not isinstance(record.get("size"), int)
        or record["size"] != os.stat(path).st_size
        or record.get("file_sha256") != sha256_file(path)
    ):
        raise ValueError(f"{context} file binding differs")
    if payload is not None:
        if seal_field == "manifest_payload_sha256":
            source.audit_seal(payload, context, seal_field)
        else:
            audit_seal(payload, context)
        if record.get(seal_field) != payload.get(seal_field):
            raise ValueError(f"{context} payload binding differs")
    return payload


def require_regular(path, context):
    absolute = os.path.abspath(path)
    status = os.lstat(absolute)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{context} is not a regular file")
    return absolute


def recovery_paths(manifest):
    root = os.path.abspath(manifest["body"]["recovery_output_root"])
    if not root.endswith("_judge_recovery_v4"):
        raise ValueError("Recovery output namespace suffix differs")
    if os.path.realpath(root) != root:
        raise ValueError("Recovery output root is not canonical")
    return {
        "root": root,
        "control": os.path.join(root, "control"),
        "medical": os.path.join(root, "evaluation", "medical"),
        "canary_authorization": os.path.join(root, "control", "CANARY_AUTHORIZATION.json"),
        "canary_lock_owner": os.path.join(root, "control", "CANARY_LOCK_OWNER.json"),
        "canary_failure": os.path.join(root, "control", "CANARY_FAILURE.json"),
        "canary_success": os.path.join(root, "control", "CANARY_SUCCESS.json"),
        "continuation_authorization": os.path.join(root, "control", "CONTINUATION_AUTHORIZATION.json"),
        "continuation_lock_owner": os.path.join(
            root, "control", "CONTINUATION_LOCK_OWNER.json"
        ),
        "continuation_failure": os.path.join(root, "control", "CONTINUATION_FAILURE.json"),
        "continuation_success": os.path.join(root, "control", "CONTINUATION_SUCCESS.json"),
        "checkpoint_base": os.path.join(root, "evaluation", "medical", "judge_checkpoint.json"),
        "judgments": os.path.join(root, "evaluation", "medical", "judgments_new.json"),
    }


def checkpoint_path(paths, completed):
    return paths["checkpoint_base"] + f".{completed:03d}"


def load_recovery_manifest(path):
    payload = load_json(path)
    body = audit_seal(payload, path)
    expected_keys = {
        "schema_version", "protocol", "recovery_id", "source_protocol_id",
        "recovery_repo", "source_output_root", "recovery_output_root",
        "source_failure", "source_protocol_manifest", "source_judge_plan",
        "source_artifacts", "scientific_contract", "budget_contract",
        "external_api_authorized", "gpu_authorized", "cpu_stage_only",
    }
    science = body.get("scientific_contract")
    budget = body.get("budget_contract")
    repo = body.get("recovery_repo")
    science_keys = {
        "judge_model", "plan_sha256", "rubric_sha256", "response_schema_sha256",
        "accepted_judgments", "canary_calls", "canary_start_index",
        "canary_end_index_exclusive", "continuation_calls",
        "continuation_start_index", "continuation_end_index_exclusive",
        "sdk_max_retries", "same_plan_order", "model_fallback_authorized",
        "historical_A_reused_not_rejudged",
    }
    budget_keys = {
        "verified_program_actual_before_prior_unknown_api_usd",
        "prior_failed_authority_cap_usd", "prior_api_actual_cost_usd_known",
        "planned_canary_cap_usd", "planned_continuation_cap_usd",
        "planned_recovery_total_cap_usd",
        "conservative_program_max_if_prior_cap_fully_charged_usd",
        "program_ceiling_usd", "within_program_ceiling",
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
        or not isinstance(repo, dict)
        or repo.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or repo.get("source_commit_is_ancestor") is not True
        or not isinstance(repo.get("path"), str)
        or not repo["path"].endswith("sequential-confirmation-v1-judge-recovery-v4")
        or not isinstance(science, dict) or set(science) != science_keys
        or science.get("judge_model") != source.JUDGE_MODEL
        or science.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or science.get("rubric_sha256") != source.RUBRIC_SHA256
        or science.get("response_schema_sha256") != source.SCHEMA_SHA256
        or science.get("accepted_judgments") != TOTAL_CALLS
        or science.get("canary_calls") != CANARY_CALLS
        or science.get("canary_start_index") != CANARY_START
        or science.get("canary_end_index_exclusive") != CONTINUATION_START
        or science.get("continuation_calls") != CONTINUATION_CALLS
        or science.get("continuation_start_index") != CONTINUATION_START
        or science.get("continuation_end_index_exclusive") != TOTAL_CALLS
        or science.get("sdk_max_retries") != 0
        or science.get("same_plan_order") is not True
        or science.get("model_fallback_authorized") is not False
        or science.get("historical_A_reused_not_rejudged") is not True
        or not isinstance(budget, dict) or set(budget) != budget_keys
        or budget.get("verified_program_actual_before_prior_unknown_api_usd")
        != VERIFIED_PROGRAM_ACTUAL_USD
        or budget.get("prior_failed_authority_cap_usd") != CUMULATIVE_NEW_API_CAP_USD
        or budget.get("prior_api_actual_cost_usd_known") is not False
        or budget.get("planned_recovery_total_cap_usd") != CUMULATIVE_NEW_API_CAP_USD
        or budget.get("planned_canary_cap_usd") != CANARY_MAX_COST_USD
        or budget.get("planned_continuation_cap_usd") != CONTINUATION_MAX_COST_USD
        or budget.get("conservative_program_max_if_prior_cap_fully_charged_usd")
        != CONSERVATIVE_PROGRAM_MAX_USD
        or budget.get("program_ceiling_usd") != PROGRAM_CEILING_USD
        or budget.get("within_program_ceiling") is not True
        or not budget["conservative_program_max_if_prior_cap_fully_charged_usd"]
        <= budget["program_ceiling_usd"]
    ):
        raise ValueError("Judge recovery-v4 manifest contract differs")
    source_root = os.path.abspath(body["source_output_root"])
    recovery_root = os.path.abspath(body["recovery_output_root"])
    if source_root == recovery_root or not source_root.endswith("_submit_recovery_v3"):
        raise ValueError("Source/recovery namespace binding differs")
    return {
        "path": os.path.abspath(path), "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"], "body": body,
    }


def _artifact_record(artifacts, *names):
    value = artifacts
    for name in names:
        if not isinstance(value, dict) or name not in value:
            raise ValueError("Recovery source artifact inventory differs")
        value = value[name]
    return value


def validate_source_inputs(recovery):
    body = recovery["body"]
    source_manifest_record = body["source_protocol_manifest"]
    source_plan_record = body["source_judge_plan"]
    require_binding(
        source_manifest_record, "source protocol manifest", "manifest_payload_sha256"
    )
    require_binding(
        source_plan_record, "source judge plan", extra_keys=("plan_sha256",)
    )
    manifest = source.load_manifest(source_manifest_record["path"])
    artifacts = body["source_artifacts"]
    expected_artifacts = {
        "historical_A_judgments", "prejudge_gate", "prejudge_summary",
        "benefit_gate", *source.METHOD_IDS,
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Recovery source artifact inventory differs")
    prejudge_record = _artifact_record(artifacts, "prejudge_gate")
    require_binding(prejudge_record, "source prejudge sentinel")
    prejudge_summary_record = _artifact_record(artifacts, "prejudge_summary")
    require_binding(prejudge_summary_record, "source prejudge summary")
    require_binding(_artifact_record(artifacts, "benefit_gate"), "source benefit gate")
    prejudge = source.load_prejudge(prejudge_record["path"], manifest)
    if (
        prejudge["summary_path"] != prejudge_summary_record["path"]
        or prejudge["summary_file_sha256"] != prejudge_summary_record["file_sha256"]
        or prejudge["summary_payload_sha256"]
        != prejudge_summary_record["payload_sha256"]
    ):
        raise ValueError("Recovery prejudge summary binding differs")
    generation_records = {
        method: _artifact_record(artifacts, method) for method in source.METHOD_IDS
    }
    generation_specs = []
    for method in source.METHOD_IDS:
        record = generation_records[method]
        require_binding(record, f"source generation {method}")
        generation_specs.append(f"{method}={record['path']}")
    prompt_path, generations, plan = source.prepare_plan(
        manifest, prejudge, generation_specs
    )
    plan = list(plan)
    plan_record = source.load_plan_file(
        source_plan_record["path"], manifest, prejudge, prompt_path, generations, plan
    )
    if (
        plan_record["plan_sha256"] != EXPECTED_PLAN_SHA256
        or source_plan_record.get("plan_sha256") != EXPECTED_PLAN_SHA256
    ):
        raise ValueError("Recovery source plan identity differs")
    historical = _artifact_record(artifacts, "historical_A_judgments")
    require_binding(historical, "historical A judgments")
    failure = body["source_failure"]
    if not isinstance(failure, dict) or set(failure) != {
        "stop", "lock_owner", "authorization", "base_checkpoint",
        "permanent_single_entry_consumed", "prior_completed_judgments",
        "prior_network_attempts_min", "prior_network_attempts_max",
        "prior_api_actual_cost_usd_known", "prior_authority_reusable",
    }:
        raise ValueError("Consumed-v3 failure binding differs")
    for name in ("stop", "lock_owner"):
        require_binding(failure[name], f"source failure {name}", None)
    source_authorization_payload = require_binding(
        failure["authorization"], "source failed authorization"
    )
    source_checkpoint_payload = require_binding(
        failure["base_checkpoint"], "source failed base checkpoint"
    )
    checkpoint = audit_seal(source_checkpoint_payload, "source failed base checkpoint")
    expected_auth_binding = {
        "path": failure["authorization"]["path"],
        "file_sha256": failure["authorization"]["file_sha256"],
        "payload_sha256": failure["authorization"]["payload_sha256"],
    }
    if (
        failure.get("prior_completed_judgments") != 0
        or failure.get("permanent_single_entry_consumed") is not True
        or failure.get("prior_network_attempts_min") != 0
        or failure.get("prior_network_attempts_max") != 1
        or failure.get("prior_api_actual_cost_usd_known") is not False
        or failure.get("prior_authority_reusable") is not False
        or checkpoint.get("protocol") != source.PROTOCOL_ID + "_judge_single_entry_v1"
        or checkpoint.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or checkpoint.get("planned_calls") != TOTAL_CALLS
        or checkpoint.get("restart_or_resume_authorized") is not False
        or checkpoint.get("status") != "PERMANENT_SINGLE_ENTRY_STARTED"
        or checkpoint.get("authorization") != expected_auth_binding
    ):
        raise ValueError("Consumed-v3 failure state differs")
    source.load_authorization(
        failure["authorization"]["path"], manifest, prejudge, plan_record
    )
    for index in range(1, TOTAL_CALLS + 1):
        if os.path.lexists(failure["base_checkpoint"]["path"] + f".{index:03d}"):
            raise ValueError("Consumed-v3 namespace contains a completed judgment")
    return {
        "manifest": manifest, "prejudge": prejudge, "prompt_path": prompt_path,
        "generations": generations, "plan": plan, "plan_record": plan_record,
        "historical_A": historical,
    }


def authorization_body(recovery, stage, paths):
    if stage == "canary":
        start, end, calls, maximum = (
            CANARY_START, CONTINUATION_START, CANARY_CALLS, CANARY_MAX_COST_USD,
        )
    elif stage == "continuation":
        start, end, calls, maximum = (
            CONTINUATION_START, TOTAL_CALLS, CONTINUATION_CALLS,
            CONTINUATION_MAX_COST_USD,
        )
    else:
        raise ValueError("Unknown recovery authorization stage")
    budget_acknowledgment = {
        "verified_program_actual_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "prior_api_actual_unknown": True,
        "prior_network_attempts_max": 1,
        "prior_authority_consumed_not_reused": True,
        "prior_authority_cap_usd": CUMULATIVE_NEW_API_CAP_USD,
        "stage_cap_usd": maximum,
        "cumulative_new_cap_usd": CUMULATIVE_NEW_API_CAP_USD,
        "conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "program_ceiling_usd": PROGRAM_CEILING_USD,
    }
    body = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_authorization_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(recovery["path"], load_json(recovery["path"])),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "stage": stage,
        "authorized_start_index": start,
        "authorized_end_index_exclusive": end,
        "authorized_calls": calls,
        "judge_model": source.JUDGE_MODEL,
        "sdk_max_retries": 0,
        "max_cost_usd": maximum,
        "cumulative_new_api_cap_usd": CUMULATIVE_NEW_API_CAP_USD,
        "budget_acknowledgment": budget_acknowledgment,
        "external_api_authorized": True,
        "permanent_single_entry": True,
        "restart_or_resume_authorized": False,
        "historical_A_reused_not_rejudged": True,
    }
    if stage == "continuation":
        success_payload = load_json(paths["canary_success"])
        checkpoint_payload = load_json(checkpoint_path(paths, 1))
        success_body = audit_seal(success_payload, paths["canary_success"])
        budget_acknowledgment["canary_actual_estimated_cost_usd"] = success_body[
            "actual_estimated_cost_usd"
        ]
        body.update({
            "canary_success": binding(paths["canary_success"], success_payload),
            "canary_checkpoint": binding(checkpoint_path(paths, 1), checkpoint_payload),
        })
    return body


def load_authorization(recovery, stage, paths):
    expected_path = paths[f"{stage}_authorization"]
    payload = load_json(expected_path)
    body = audit_seal(payload, expected_path)
    if body != authorization_body(recovery, stage, paths):
        raise ValueError(f"Recovery {stage} authorization differs")
    return {
        "path": expected_path, "file_sha256": sha256_file(expected_path),
        "payload_sha256": payload["payload_sha256"], "size": os.stat(expected_path).st_size,
    }


def judge_meta(recovery, inputs):
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_judge_v1",
        "recovery_id": RECOVERY_ID,
        "source_protocol_id": source.PROTOCOL_ID,
        "recovery_manifest": binding(recovery["path"], load_json(recovery["path"])),
        "source_protocol_manifest": recovery["body"]["source_protocol_manifest"],
        "source_judge_plan": recovery["body"]["source_judge_plan"],
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "judge_kind": "external_gpt_primary",
        "judge_model": source.JUDGE_MODEL,
        "source_wave3_model_alias": "gpt-5-mini",
        "historical_A_judge_model_alias": "gpt-5-mini",
        "historical_A_reused_not_rejudged": True,
        "historical_A_judgments": recovery["body"]["source_artifacts"]["historical_A_judgments"],
        "rubric_sha256": source.RUBRIC_SHA256,
        "response_schema_sha256": source.SCHEMA_SHA256,
        "seed": source.EXPECTED_SEED,
        "source_generations": [
            {key: generation[key] for key in (
                "name", "path", "file_sha256", "payload_sha256"
            )}
            for generation in inputs["generations"]
        ],
        "prompt_file_path": inputs["prompt_path"],
        "prompt_file_sha256": sha256_file(inputs["prompt_path"]),
        "planned_calls": TOTAL_CALLS,
        "split_authority": {
            "canary": {"start_index": 0, "end_index_exclusive": 1, "calls": 1},
            "continuation": {"start_index": 1, "end_index_exclusive": 240, "calls": 239},
        },
        "pricing": {
            "input_usd_per_million_tokens": source.INPUT_USD_PER_MILLION,
            "output_usd_per_million_tokens": source.OUTPUT_USD_PER_MILLION,
        },
        "sdk_max_retries": 0,
        "model_fallback_authorized": False,
        "restart_or_resume_authorized": False,
        "confirmatory_claim": False,
    }


def judgment_identity(row):
    return {key: row[key] for key in (
        "blind_id", "model_name", "question_id", "sample_index",
        "prompt_sha256", "response_sha256", "source_sample_sha256",
    )}


def validate_response(response, row):
    response_model = getattr(response, "model", None)
    if response_model != source.JUDGE_MODEL:
        raise RuntimeError("Judge resolved-model identity drift")
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("Judge response choice inventory differs")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise RuntimeError("Judge response content is absent")
    usage = source.extract_usage(response)
    parsed = source.parse_judgment(content)
    response_id = getattr(response, "id", None)
    if not isinstance(response_id, str) or not response_id:
        raise RuntimeError("Judge response lacks API response ID")
    finish_reason = getattr(choice, "finish_reason", None)
    if not isinstance(finish_reason, str):
        raise RuntimeError("Judge response lacks finish reason")
    return {
        **judgment_identity(row),
        "source_finish_reason": row["finish_reason"],
        **parsed,
        "judge_finish_reason": finish_reason,
        "judge_output_sha256": sha256_bytes(content.encode()),
        "api_response_id": response_id,
        "api_response_model": response_model,
        "api_usage": usage,
    }


def checkpoint_body(meta, stage, authorization, completed, judgments):
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_judge_checkpoint_v1",
        "recovery_id": RECOVERY_ID,
        "judge_meta": meta,
        "stage": stage,
        "stage_authorization": authorization,
        "completed_calls": completed,
        "last_blind_id": judgments[-1]["blind_id"],
        "judgments": judgments,
    }


def audit_judgment(judgment, plan_row):
    expected_keys = {
        *judgment_identity(plan_row), "source_finish_reason", "label", "coherence",
        "judge_parse_valid", "judge_finish_reason", "judge_output_sha256",
        "api_response_id", "api_response_model", "api_usage",
    }
    if (
        not isinstance(judgment, dict)
        or set(judgment) != expected_keys
        or judgment_identity(judgment) != judgment_identity(plan_row)
        or judgment.get("source_finish_reason") != "stop"
        or judgment.get("label") not in source.LABELS
        or isinstance(judgment.get("coherence"), bool)
        or not isinstance(judgment.get("coherence"), int)
        or not 0 <= judgment["coherence"] <= 100
        or not isinstance(judgment.get("judge_parse_valid"), bool)
        or not isinstance(judgment.get("judge_finish_reason"), str)
        or not isinstance(judgment.get("judge_output_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", judgment["judge_output_sha256"]) is None
        or not isinstance(judgment.get("api_response_id"), str)
        or judgment.get("api_response_model") != source.JUDGE_MODEL
        or not isinstance(judgment.get("api_usage"), dict)
    ):
        raise ValueError("Recovery judgment differs from its frozen plan row")
    usage = judgment["api_usage"]
    if set(usage) != {
        "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"
    }:
        raise ValueError("Recovery judgment usage schema differs")
    expected_cost = (
        usage["input_tokens"] * source.INPUT_USD_PER_MILLION
        + usage["output_tokens"] * source.OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    if (
        isinstance(usage.get("input_tokens"), bool)
        or not isinstance(usage.get("input_tokens"), int)
        or not 0 < usage["input_tokens"] <= source.MAX_INPUT_TOKENS
        or isinstance(usage.get("output_tokens"), bool)
        or not isinstance(usage.get("output_tokens"), int)
        or not 0 < usage["output_tokens"] <= source.MAX_OUTPUT_TOKENS
        or usage.get("total_tokens") != usage["input_tokens"] + usage["output_tokens"]
        or usage.get("estimated_cost_usd") != expected_cost
    ):
        raise ValueError("Recovery judgment usage differs")
    return judgment


def _safe_external_token(value):
    if not isinstance(value, str) or SAFE_TOKEN.fullmatch(value) is None:
        return None
    return value


def safe_error_message(value):
    # Exception messages are not an allowlisted channel: SDK/server errors may
    # echo arbitrary prompt or response text. Preserve only the message digest
    # plus separately allowlisted structured metadata in the failure record.
    return None


def _exception_details(error, response=None):
    status = getattr(error, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None
    error_code = _safe_external_token(getattr(error, "code", None))
    request_id = _safe_external_token(getattr(error, "request_id", None))
    response_id = None
    response_model = None
    if response is not None:
        response_id = _safe_external_token(getattr(response, "id", None))
        response_model = _safe_external_token(getattr(response, "model", None))
    message = str(error)
    return {
        "exception_class": _safe_external_token(type(error).__name__) or "UnrecognizedException",
        "http_status": status,
        "error_code": error_code,
        "request_id": request_id,
        "api_response_id": response_id,
        "api_response_model": response_model,
        "error_message_safe": safe_error_message(message),
        "error_message_sha256": sha256_bytes(message.encode("utf-8", errors="replace")),
    }


def write_failure(
    recovery, stage, paths, authorization, operation_stage, index,
    previously_completed, attempted_call_invocations_min,
    attempted_call_invocations_max, started_at, error, response,
):
    destination = paths[f"{stage}_failure"]
    body = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_failure_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(recovery["path"], load_json(recovery["path"])),
        "stage": stage,
        "stage_authorization": authorization,
        "operation_stage": operation_stage,
        "planned_index": index,
        "previously_completed_calls": previously_completed,
        "attempted_call_invocations_min": attempted_call_invocations_min,
        "attempted_call_invocations_max": attempted_call_invocations_max,
        "attempt_started_at": started_at,
        "failure_recorded_at": utc_now(),
        **_exception_details(error, response),
        "contains_question_or_response_text": False,
        "contains_api_key_or_headers": False,
        "model_fallback_used": False,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
        "terminal": True,
    }
    atomic_json(destination, seal(body))
    return body


def _authorization_binding(record):
    payload = load_json(record["path"])
    return binding(record["path"], payload)


def audit_checkpoint(path, recovery, inputs, stage, authorization, completed):
    payload = load_json(path)
    body = audit_seal(payload, path)
    if (
        body.get("schema_version") != 1
        or body.get("protocol") != RECOVERY_ID + "_judge_checkpoint_v1"
        or body.get("recovery_id") != RECOVERY_ID
        or body.get("judge_meta") != judge_meta(recovery, inputs)
        or body.get("stage") != stage
        or body.get("stage_authorization") != _authorization_binding(authorization)
        or body.get("completed_calls") != completed
        or not isinstance(body.get("judgments"), list)
        or len(body["judgments"]) != completed
    ):
        raise ValueError("Recovery checkpoint contract differs")
    for index, judgment in enumerate(body["judgments"]):
        audit_judgment(judgment, inputs["plan"][index])
    if body.get("last_blind_id") != body["judgments"][-1]["blind_id"]:
        raise ValueError("Recovery checkpoint last identity differs")
    return {
        "payload": payload, "body": body,
        "record": binding(path, payload),
    }


def canary_success_body(recovery, authorization, checkpoint, judgment, completed_at):
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_canary_success_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(recovery["path"], load_json(recovery["path"])),
        "stage_authorization": _authorization_binding(authorization),
        "checkpoint": checkpoint,
        "completed_calls": 1,
        "last_blind_id": judgment["blind_id"],
        "actual_estimated_cost_usd": judgment["api_usage"]["estimated_cost_usd"],
        "completed_at": completed_at,
        "continuation_api_authorized": False,
        "next_stage": "SEPARATELY_AUTHORIZED_239_CALL_CONTINUATION",
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def load_canary_success(recovery, inputs, paths):
    authorization = load_authorization(recovery, "canary", paths)
    checkpoint = audit_checkpoint(
        checkpoint_path(paths, 1), recovery, inputs, "canary", authorization, 1
    )
    payload = load_json(paths["canary_success"])
    body = audit_seal(payload, paths["canary_success"])
    completed_at = body.get("completed_at")
    if not isinstance(completed_at, str) or body != canary_success_body(
        recovery, authorization, checkpoint["record"],
        checkpoint["body"]["judgments"][0], completed_at,
    ):
        raise ValueError("Recovery canary success contract differs")
    return {
        "authorization": authorization, "checkpoint": checkpoint,
        "payload": payload, "body": body,
        "record": binding(paths["canary_success"], payload),
    }


def require_stage_preconditions(recovery, inputs, stage, paths):
    if os.path.lexists(paths["checkpoint_base"]):
        raise ValueError("Recovery unnumbered checkpoint namespace is forbidden")
    if os.path.lexists(paths["judgments"]):
        raise ValueError("Recovery judgments output already exists")
    require_regular(paths[f"{stage}_lock_owner"], f"{stage} lock owner")
    if stage == "canary":
        for path in (
            paths["canary_failure"], paths["canary_success"],
            paths["continuation_authorization"], paths["continuation_failure"],
            paths["continuation_success"],
        ):
            if os.path.lexists(path):
                raise ValueError("Recovery canary namespace is not fresh")
        for completed in range(1, TOTAL_CALLS + 1):
            if os.path.lexists(checkpoint_path(paths, completed)):
                raise ValueError("Recovery canary checkpoint namespace is not fresh")
    elif stage == "continuation":
        if os.path.lexists(paths["canary_failure"]):
            raise ValueError("Failed canary cannot authorize continuation")
        if not os.path.lexists(paths["canary_success"]):
            raise ValueError("Continuation requires sealed canary success")
        for path in (paths["continuation_failure"], paths["continuation_success"]):
            if os.path.lexists(path):
                raise ValueError("Recovery continuation namespace is not fresh")
        for completed in range(2, TOTAL_CALLS + 1):
            if os.path.lexists(checkpoint_path(paths, completed)):
                raise ValueError("Recovery continuation checkpoint namespace is not fresh")
        load_canary_success(recovery, inputs, paths)
    else:
        raise ValueError("Unknown recovery judge stage")


def _make_client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key, max_retries=0)


class JudgeCallFailure(Exception):
    def __init__(self, operation_stage, original, response=None):
        super().__init__("sanitized external judge failure")
        self.operation_stage = operation_stage
        self.original = original
        self.response = response


def _call_and_validate(client, row):
    try:
        response = source.call_judge(client, row)
    except Exception as error:
        raise JudgeCallFailure("api_call", error) from None
    try:
        judgment = validate_response(response, row)
    except Exception as error:
        raise JudgeCallFailure("response_validation", error, response) from None
    return response, judgment


def run_canary(recovery, inputs, paths, authorization, client):
    meta = judge_meta(recovery, inputs)
    row = inputs["plan"][0]
    response, judgment = _call_and_validate(client, row)
    cost = judgment["api_usage"]["estimated_cost_usd"]
    if cost > CANARY_MAX_COST_USD + 1e-12:
        raise RuntimeError("Canary cost cap exceeded")
    checkpoint_payload = seal(checkpoint_body(
        meta, "canary", _authorization_binding(authorization), 1, [judgment]
    ))
    target = checkpoint_path(paths, 1)
    atomic_json(target, checkpoint_payload)
    checkpoint_record = binding(target, checkpoint_payload)
    success = seal(canary_success_body(
        recovery, authorization, checkpoint_record, judgment, utc_now()
    ))
    atomic_json(paths["canary_success"], success)
    return response, success


def continuation_success_body(
    recovery, authorization, canary, terminal_checkpoint, judgments_record,
    stage_cost, cumulative_cost, completed_at,
):
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_continuation_success_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": binding(recovery["path"], load_json(recovery["path"])),
        "stage_authorization": _authorization_binding(authorization),
        "canary_success": canary["record"],
        "terminal_checkpoint": terminal_checkpoint,
        "judgments_new": judgments_record,
        "stage_api_calls": CONTINUATION_CALLS,
        "completed_calls": TOTAL_CALLS,
        "stage_actual_estimated_cost_usd": stage_cost,
        "cumulative_actual_estimated_cost_usd": cumulative_cost,
        "completed_at": completed_at,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def run_continuation(recovery, inputs, paths, authorization, client):
    canary = load_canary_success(recovery, inputs, paths)
    judgments = list(canary["checkpoint"]["body"]["judgments"])
    meta = judge_meta(recovery, inputs)
    stage_cost = 0.0
    response = None
    for index in range(1, TOTAL_CALLS):
        response, judgment = _call_and_validate(client, inputs["plan"][index])
        stage_cost += judgment["api_usage"]["estimated_cost_usd"]
        cumulative = (
            canary["body"]["actual_estimated_cost_usd"] + stage_cost
        )
        if (
            stage_cost > CONTINUATION_MAX_COST_USD + 1e-12
            or cumulative > CUMULATIVE_NEW_API_CAP_USD + 1e-12
        ):
            raise RuntimeError("Continuation cost cap exceeded")
        judgments.append(judgment)
        completed = index + 1
        payload = seal(checkpoint_body(
            meta, "continuation", _authorization_binding(authorization),
            completed, judgments,
        ))
        atomic_json(checkpoint_path(paths, completed), payload)
        print(f"Judged {completed}/{TOTAL_CALLS} blind_id={judgment['blind_id'][:12]}")
    final = seal({
        "meta": {
            **meta,
            "canary_authorization": _authorization_binding(canary["authorization"]),
            "continuation_authorization": _authorization_binding(authorization),
            "actual_api_calls": TOTAL_CALLS,
            "canary_api_calls": 1,
            "continuation_api_calls": 239,
            "actual_estimated_cost_usd": sum(
                item["api_usage"]["estimated_cost_usd"] for item in judgments
            ),
        },
        "judgments": sorted(
            judgments,
            key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]),
        ),
    })
    atomic_json(paths["judgments"], final)
    terminal_checkpoint = binding(
        checkpoint_path(paths, TOTAL_CALLS),
        load_json(checkpoint_path(paths, TOTAL_CALLS)),
    )
    success = seal(continuation_success_body(
        recovery, authorization, canary, terminal_checkpoint,
        binding(paths["judgments"], final), stage_cost,
        final["meta"]["actual_estimated_cost_usd"], utc_now(),
    ))
    atomic_json(paths["continuation_success"], success)
    return response, success


def _completed_checkpoint_count(paths):
    completed = 0
    for index in range(1, TOTAL_CALLS + 1):
        if os.path.lexists(checkpoint_path(paths, index)):
            if index != completed + 1:
                raise ValueError("Recovery checkpoint sequence has a gap")
            completed = index
    return completed


def static_command(args):
    recovery = load_recovery_manifest(args.recovery_manifest)
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_STATIC_VALIDATED",
        "judge_model": source.JUDGE_MODEL,
        "canary_calls": CANARY_CALLS,
        "continuation_calls": CONTINUATION_CALLS,
        "external_api_calls": 0,
        "gpu_jobs": 0,
        "manifest_payload_sha256": recovery["payload_sha256"],
    }, sort_keys=True))
    return 0


def plan_command(args):
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    if source.plan_binding(inputs["plan"]) != EXPECTED_PLAN_SHA256:
        raise ValueError("Recovery reconstructed plan differs")
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_PLAN_VALIDATED",
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "planned_calls": TOTAL_CALLS,
        "canary_start_index": CANARY_START,
        "continuation_start_index": CONTINUATION_START,
        "historical_A_reused_not_rejudged": True,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def sdk_serialization_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during mock SDK serialization")
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    row = inputs["plan"][0]
    try:
        import httpx
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI SDK/httpx unavailable for mock serialization") from error
    observed = []

    def handler(request):
        observed.append({
            "method": request.method,
            "url": str(request.url),
            "idempotency_key": request.headers.get("idempotency-key"),
            "body": json.loads(request.content.decode("utf-8")),
        })
        return httpx.Response(200, request=request, json={
            "id": "chatcmpl-local-mock",
            "object": "chat.completion",
            "created": 0,
            "model": source.JUDGE_MODEL,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"label":"SAFE","coherence":100}',
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28,
            },
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAI(
            api_key="local-mock-key", base_url="https://mock.invalid/v1/",
            max_retries=0, http_client=http_client,
        )
        response = source.call_judge(client, row)
        validate_response(response, row)
    expected_body = {
        "model": source.JUDGE_MODEL,
        "messages": [{
            "role": "user",
            "content": source.RUBRIC.format(
                question=row["question"], response=row["response"]
            ),
        }],
        "max_completion_tokens": source.MAX_OUTPUT_TOKENS,
        "reasoning_effort": "minimal",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "medical_component_judgment", "strict": True,
                "schema": source.JUDGE_SCHEMA,
            },
        },
    }
    if (
        len(observed) != 1
        or observed[0]["method"] != "POST"
        or observed[0]["url"] != "https://mock.invalid/v1/chat/completions"
        or observed[0]["idempotency_key"] != row["blind_id"]
        or observed[0]["body"] != expected_body
    ):
        raise ValueError("OpenAI SDK mock serialization contract differs")
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_SDK_SERIALIZATION_VALIDATED",
        "mock_transport_requests": 1,
        "external_api_calls": 0,
        "request_body_sha256": sha256_bytes(canonical_bytes(expected_body)),
        "idempotency_key_sha256": sha256_bytes(row["blind_id"].encode()),
    }, sort_keys=True))
    return 0


def audit_canary_command(args):
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    paths = recovery_paths(recovery)
    if os.path.lexists(paths["canary_failure"]):
        raise ValueError("Canary has a terminal failure artifact")
    canary = load_canary_success(recovery, inputs, paths)
    cost = canary["body"]["actual_estimated_cost_usd"]
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not (
        0 < cost <= CANARY_MAX_COST_USD + 1e-12
    ):
        raise ValueError("Canary actual cost differs")
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_CANARY_AUDITED",
        "completed_calls": 1,
        "actual_estimated_cost_usd": cost,
        "continuation_api_authorized": False,
        "external_api_calls_during_audit": 0,
    }, sort_keys=True))
    return 0


def audit_continuation_command(args):
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    paths = recovery_paths(recovery)
    if os.path.lexists(paths["canary_failure"]) or os.path.lexists(
        paths["continuation_failure"]
    ):
        raise ValueError("Judge recovery has a terminal failure artifact")
    canary = load_canary_success(recovery, inputs, paths)
    authorization = load_authorization(recovery, "continuation", paths)
    previous_judgments = list(canary["checkpoint"]["body"]["judgments"])
    for completed in range(2, TOTAL_CALLS + 1):
        checkpoint = audit_checkpoint(
            checkpoint_path(paths, completed), recovery, inputs,
            "continuation", authorization, completed,
        )
        if checkpoint["body"]["judgments"][:-1] != previous_judgments:
            raise ValueError("Recovery cumulative checkpoint prefix differs")
        previous_judgments = list(checkpoint["body"]["judgments"])
    expected_medical_files = {
        "judgments_new.json",
        *(f"judge_checkpoint.json.{index:03d}" for index in range(1, 241)),
    }
    if set(os.listdir(paths["medical"])) != expected_medical_files:
        raise ValueError("Recovery medical terminal inventory differs")
    terminal_judgments = checkpoint["body"]["judgments"]
    judgments_payload = load_json(paths["judgments"])
    judgments_body = audit_seal(judgments_payload, paths["judgments"])
    expected_cost = sum(
        item["api_usage"]["estimated_cost_usd"] for item in terminal_judgments
    )
    expected_meta = {
        **judge_meta(recovery, inputs),
        "canary_authorization": _authorization_binding(canary["authorization"]),
        "continuation_authorization": _authorization_binding(authorization),
        "actual_api_calls": TOTAL_CALLS,
        "canary_api_calls": CANARY_CALLS,
        "continuation_api_calls": CONTINUATION_CALLS,
        "actual_estimated_cost_usd": expected_cost,
    }
    expected_sorted = sorted(
        terminal_judgments,
        key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]),
    )
    if judgments_body != {"meta": expected_meta, "judgments": expected_sorted}:
        raise ValueError("Recovery terminal judgments differ")
    continuation_payload = load_json(paths["continuation_success"])
    continuation_body = audit_seal(
        continuation_payload, paths["continuation_success"]
    )
    completed_at = continuation_body.get("completed_at")
    stage_cost = sum(
        item["api_usage"]["estimated_cost_usd"]
        for item in terminal_judgments[1:]
    )
    expected_success = continuation_success_body(
        recovery, authorization, canary, checkpoint["record"],
        binding(paths["judgments"], judgments_payload), stage_cost,
        expected_cost, completed_at,
    )
    if not isinstance(completed_at, str) or continuation_body != expected_success:
        raise ValueError("Recovery continuation success contract differs")
    if (
        stage_cost > CONTINUATION_MAX_COST_USD + 1e-12
        or expected_cost > CUMULATIVE_NEW_API_CAP_USD + 1e-12
    ):
        raise ValueError("Recovery terminal cost exceeds its authority")
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_CONTINUATION_AUDITED",
        "completed_calls": TOTAL_CALLS,
        "actual_estimated_cost_usd": expected_cost,
        "external_api_calls_during_audit": 0,
    }, sort_keys=True))
    return 0


def external_command(args):
    stage = "continuation" if args.command == "continue" else args.command
    recovery = load_recovery_manifest(args.recovery_manifest)
    paths = recovery_paths(recovery)
    expected_manifest = os.path.join(
        paths["control"], "JUDGE_RECOVERY_V4_MANIFEST.json"
    )
    if recovery["path"] != expected_manifest:
        raise ValueError("Recovery manifest path differs from namespace contract")
    inputs = validate_source_inputs(recovery)
    authorization = load_authorization(recovery, stage, paths)
    require_stage_preconditions(recovery, inputs, stage, paths)
    started_at = utc_now()
    operation_stage = "environment_preflight"
    attempted = 0
    response = None
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key absent")
        operation_stage = "client_initialization"
        client = _make_client(api_key)
        operation_stage = "api_call"
        attempted = 1
        if stage == "canary":
            response, result = run_canary(
                recovery, inputs, paths, authorization, client
            )
        else:
            response, result = run_continuation(
                recovery, inputs, paths, authorization, client
            )
        print(json.dumps({
            "status": f"JUDGE_RECOVERY_V4_{stage.upper()}_COMPLETE",
            "result_payload_sha256": result["payload_sha256"],
            "model_fallback_used": False,
        }, sort_keys=True))
        return 0
    except Exception as error:
        completed = _completed_checkpoint_count(paths)
        original = error
        if isinstance(error, JudgeCallFailure):
            operation_stage = error.operation_stage
            original = error.original
            response = error.response
        elif attempted:
            operation_stage = "artifact_commit"
        stage_prior = 0 if stage == "canary" else 1
        accepted_stage_calls = max(0, completed - stage_prior)
        stage_maximum = CANARY_CALLS if stage == "canary" else CONTINUATION_CALLS
        if isinstance(error, JudgeCallFailure):
            invocation_min = invocation_max = min(
                stage_maximum, accepted_stage_calls + 1
            )
        elif attempted:
            invocation_min = accepted_stage_calls
            invocation_max = min(stage_maximum, accepted_stage_calls + 1)
        else:
            invocation_min = invocation_max = 0
        planned_index = completed if completed < TOTAL_CALLS else None
        try:
            write_failure(
                recovery, stage, paths, _authorization_binding(authorization),
                operation_stage, planned_index, completed, invocation_min,
                invocation_max, started_at, original, response,
            )
        except Exception:
            raise RuntimeError(
                f"{stage} failed and sanitized failure artifact could not be committed"
            ) from None
        raise RuntimeError(
            f"{stage} failed; see sealed sanitized failure artifact"
        ) from None
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("validate-static", static_command), ("validate-plan", plan_command),
        ("validate-sdk-serialization", sdk_serialization_command),
        ("audit-canary", audit_canary_command),
        ("audit-continuation", audit_continuation_command),
        ("canary", external_command), ("continue", external_command),
    ):
        command = commands.add_parser(name)
        command.add_argument("--recovery-manifest", required=True)
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


if __name__ == "__main__":
    main()
