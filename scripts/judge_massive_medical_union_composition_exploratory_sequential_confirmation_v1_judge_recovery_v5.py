#!/usr/bin/env python3
"""Fresh split-authority judge recovery after the sealed v4 credit failure.

The v5 overlay preserves the exact v3 scientific inputs and 240-row plan used
by v4, while treating both the v3 authority and the v4 one-call canary
authority as consumed.  Validation and SDK serialization are local-only.
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


_V4_PATH = Path(__file__).with_name(
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4.py"
)
_V4_SPEC = importlib.util.spec_from_file_location(
    "_mmu_judge_recovery_v5_private_v4", _V4_PATH
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError("Unable to load the private v4 judge implementation")
base = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(base)


source = base.source
RECOVERY_ID = source.PROTOCOL_ID + "_judge_recovery_v5"
EXPECTED_PLAN_SHA256 = base.EXPECTED_PLAN_SHA256
EXPECTED_SOURCE_COMMIT = "5f7357fee6654cccb7918d307963dcfe5fa73418"
CANARY_CALLS = 1
CONTINUATION_CALLS = 239
CANARY_START = 0
CONTINUATION_START = 1
TOTAL_CALLS = 240
CANARY_MAX_COST_USD = 0.003072
CONTINUATION_MAX_COST_USD = 0.746928
NEW_RECOVERY_CAP_USD = 0.75
VERIFIED_PROGRAM_ACTUAL_USD = 2.915186
CONSUMED_V3_AUTHORITY_CAP_USD = 0.75
CONSUMED_V4_CANARY_AUTHORITY_CAP_USD = 0.003072
PRIOR_CONSUMED_AUTHORITY_CAP_USD = 0.753072
CONSERVATIVE_PROGRAM_MAX_USD = 4.418258
PROGRAM_CEILING_USD = 5.0
PRIOR_ATTEMPTS_MIN = 1
PRIOR_ATTEMPTS_MAX = 2
V4_RECOVERY_ID = source.PROTOCOL_ID + "_judge_recovery_v4"
V4_FAILURE_SHA256 = "ce57a34230596b9d964fb2e5843d09783f52782e0d5a10cb9e513fe66b87ee64"
V4_FAILURE_PAYLOAD_SHA256 = "69abff8985c42652c63169bc707ba58915665dd126cfa2d2dd3880642395f7c9"

_v4_validate_source_inputs = base.validate_source_inputs
_v4_judge_meta = base.judge_meta
_real_print = builtins.print

IDEMPOTENCY_CONTRACT = {
    "version": "recovery_id_blind_id_sha256_v1",
    "derivation": "sha256(utf8(recovery_id + ':' + blind_id))",
    "recovery_id": RECOVERY_ID,
    "canary_start_index": 0,
    "canary_end_index_exclusive": 1,
    "continuation_start_index": 1,
    "continuation_end_index_exclusive": 240,
    "raw_key_persisted": False,
    "source_raw_blind_id_reused_as_key": False,
    "row_count": 240,
    "all_240_keys_unique": True,
    "derived_key_list_sha256": "aaff8f6ab72b6e991e3cf3bebcdda5a022737461d665f5bfbb76b2f6a7766c94",
    "indexed_identity_key_list_sha256": "03b21a928dd0f8ab85f8ad3b1030a07c4b26c07efcea49c652a8c41e4ef8a028",
}


def _v5_print(*values, **kwargs):
    converted = tuple(
        value.replace("JUDGE_RECOVERY_V4", "JUDGE_RECOVERY_V5")
        if isinstance(value, str) else value
        for value in values
    )
    _real_print(*converted, **kwargs)


def recovery_paths(manifest):
    root = os.path.abspath(manifest["body"]["recovery_output_root"])
    if not root.endswith("_judge_recovery_v5"):
        raise ValueError("Recovery output namespace suffix differs")
    if os.path.realpath(root) != root:
        raise ValueError("Recovery output root is not canonical")
    control = os.path.join(root, "control")
    medical = os.path.join(root, "evaluation", "medical")
    return {
        "root": root,
        "control": control,
        "medical": medical,
        "canary_authorization": os.path.join(control, "CANARY_AUTHORIZATION.json"),
        "canary_lock_owner": os.path.join(control, "CANARY_LOCK_OWNER.json"),
        "canary_failure": os.path.join(control, "CANARY_FAILURE.json"),
        "canary_success": os.path.join(control, "CANARY_SUCCESS.json"),
        "continuation_authorization": os.path.join(
            control, "CONTINUATION_AUTHORIZATION.json"
        ),
        "continuation_lock_owner": os.path.join(
            control, "CONTINUATION_LOCK_OWNER.json"
        ),
        "continuation_failure": os.path.join(control, "CONTINUATION_FAILURE.json"),
        "continuation_success": os.path.join(control, "CONTINUATION_SUCCESS.json"),
        "checkpoint_base": os.path.join(medical, "judge_checkpoint.json"),
        "judgments": os.path.join(medical, "judgments_new.json"),
    }


def _require_bound_record(record, context, seal_field="payload_sha256"):
    expected = {"path", "size", "file_sha256"}
    if seal_field is not None:
        expected.add(seal_field)
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError(f"{context} binding schema differs")
    payload = base.require_binding(record, context, seal_field)
    return payload


def _audit_failed_v4(body):
    failed = body.get("failed_recovery_v4")
    expected_keys = {
        "recovery_id", "repo", "manifest", "staged", "canary_lock_owner",
        "canary_authorization", "canary_failure", "canary_log",
        "prep", "cpu_preflight", "terminal_contract", "consumed_authorities",
    }
    if not isinstance(failed, dict) or set(failed) != expected_keys:
        raise ValueError("Failed v4 binding schema differs")
    repo = failed["repo"]
    if (
        not isinstance(repo, dict)
        or set(repo) != {"path", "branch", "commit", "tree"}
        or repo.get("commit") != EXPECTED_SOURCE_COMMIT
        or repo.get("tree") != "7770b7e60f9942077ffb6484ce7db41e55d6a190"
        or not str(repo.get("path", "")).endswith(
            "sequential-confirmation-v1-judge-recovery-v4"
        )
        or not str(repo.get("branch", "")).endswith("sequential-v1-judge-recovery-v4")
    ):
        raise ValueError("Failed v4 repository binding differs")
    expected_records = {
        "manifest": (8510, "ae81bee14cfe384d0d9656b83fa2e6846f30f73c2beb6c20bcbac0e2b300c54f", "cc7e42e1adfd802f98b83d9ccf0c67a2902b0033c5c99a38fc1578d3fa2bbcf8"),
        "prep": (951, "300d208ebb3d9bffaed10190b3de270e646929cdfb7b964dc7a4e725a67c39f4", "4069aaf5463d363a0fd2232223a9650d75a64c73a76c982ec3caaf0995be86b7"),
        "cpu_preflight": (1429, "abe918bcb3b8b215229c87df429632c2f0377edb744430811d75284d8a18d89b", "3272c6e25a22da6a0649e469ee1ab4bfe38d282859513aaf117b29c10e2695ab"),
        "staged": (1342, "8461f77f02741d47731b8ec2891597ac08a4ec85bb3426be5d58c2308835619e", "aa032bd357f3c6fc4097dc2a64661041a4360906666fef44de27b270429de707"),
        "canary_lock_owner": (1061, "0f7ae398b869e0c1df0072993e9c463a16bac3adc3dc0406d6abb3561e56f4a1", "7a162a98ab2442b989c58749f27c3354985550fef5ca85a5f38627ed9c08da52"),
        "canary_authorization": (1661, "37cb8affc6039453174832e7fef7e2cdd65462279552ed1649e3aa8859f0a4a2", "c9c82e8aa01045b8b1acb2572fcadd5dd175f60403b2c1d3c09f0da4f0f95d6d"),
        "canary_failure": (2060, V4_FAILURE_SHA256, V4_FAILURE_PAYLOAD_SHA256),
    }
    payloads = {}
    for name, (size, digest, payload_digest) in expected_records.items():
        record = failed[name]
        payload = _require_bound_record(record, f"failed v4 {name}")
        if (
            record.get("size") != size
            or record.get("file_sha256") != digest
            or record.get("payload_sha256") != payload_digest
        ):
            raise ValueError(f"Failed v4 {name} exact binding differs")
        payloads[name] = payload
    log = failed["canary_log"]
    _require_bound_record(log, "failed v4 canary log", None)
    if (
        log.get("size") != 60
        or log.get("file_sha256")
        != "b3495e33a6f564f5f29757694f3364c39378f2ea22f049d76c4168ad7dc1b615"
    ):
        raise ValueError("Failed v4 canary log binding differs")
    failure = base.audit_seal(payloads["canary_failure"], "failed v4 canary")
    terminal = failed["terminal_contract"]
    expected_terminal = {
        "stage": "canary", "operation_stage": "api_call", "planned_index": 0,
        "completed_calls": 0, "attempted_call_invocations_min": 1,
        "attempted_call_invocations_max": 1, "exception_class": "RateLimitError",
        "http_status": 429, "error_code": "credit_balance_exhausted",
        "request_id": "req_d41d2fa776d04e9184f3f9541cce3d87",
        "terminal": True, "retry_authorized": False,
        "restart_or_resume_authorized": False,
        "contains_question_or_response_text": False,
        "contains_api_key_or_headers": False,
    }
    reconstructed = {
        "stage": failure.get("stage"),
        "operation_stage": failure.get("operation_stage"),
        "planned_index": failure.get("planned_index"),
        "completed_calls": failure.get("previously_completed_calls"),
        "attempted_call_invocations_min": failure.get("attempted_call_invocations_min"),
        "attempted_call_invocations_max": failure.get("attempted_call_invocations_max"),
        "exception_class": failure.get("exception_class"),
        "http_status": failure.get("http_status"),
        "error_code": failure.get("error_code"),
        "request_id": failure.get("request_id"),
        "terminal": failure.get("terminal"),
        "retry_authorized": failure.get("retry_authorized"),
        "restart_or_resume_authorized": failure.get("restart_or_resume_authorized"),
        "contains_question_or_response_text": failure.get("contains_question_or_response_text"),
        "contains_api_key_or_headers": failure.get("contains_api_key_or_headers"),
    }
    if terminal != expected_terminal or reconstructed != expected_terminal:
        raise ValueError("Failed v4 credit-balance terminal contract differs")
    authorities = failed["consumed_authorities"]
    if authorities != {
        "v3_external_judge_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "v4_canary_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "v4_continuation_authorized": False,
        "total_consumed_authority_cap_usd": PRIOR_CONSUMED_AUTHORITY_CAP_USD,
        "historical_attempts_min": PRIOR_ATTEMPTS_MIN,
        "historical_attempts_max": PRIOR_ATTEMPTS_MAX,
        "historical_accepted_judgments": 0,
        "reusable": False,
    }:
        raise ValueError("Failed authority accounting differs")
    return failed


def load_recovery_manifest(path):
    payload = base.load_json(path)
    body = base.audit_seal(payload, path)
    expected_keys = {
        "schema_version", "protocol", "recovery_id", "source_protocol_id",
        "recovery_repo", "source_output_root", "recovery_output_root",
        "source_failure", "source_protocol_manifest", "source_judge_plan",
        "source_artifacts", "scientific_contract", "budget_contract",
        "failed_recovery_v4", "external_api_authorized", "gpu_authorized",
        "cpu_stage_only",
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
        "historical_A_reused_not_rejudged", "idempotency_contract",
    }
    budget_keys = {
        "verified_program_actual_before_unknown_api_usd",
        "consumed_v3_authority_cap_usd", "consumed_v4_canary_authority_cap_usd",
        "prior_consumed_authority_cap_usd", "prior_api_actual_cost_usd_known",
        "historical_attempts_min", "historical_attempts_max",
        "v4_continuation_authorized", "planned_v5_canary_cap_usd",
        "planned_v5_continuation_cap_usd", "planned_v5_total_cap_usd",
        "conservative_program_max_usd", "program_ceiling_usd",
        "within_program_ceiling", "remaining_ceiling_gap_usd",
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
        or not str(body.get("recovery_output_root", "")).endswith("_judge_recovery_v5")
        or not isinstance(repo, dict) or set(repo) != repo_keys
        or not str(repo.get("path", "")).endswith(
            "sequential-confirmation-v1-judge-recovery-v5"
        )
        or not str(repo.get("branch", "")).endswith("sequential-v1-judge-recovery-v5")
        or repo.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or repo.get("source_commit_is_direct_parent") is not True
        or not isinstance(repo.get("commit"), str)
        or not isinstance(repo.get("tree"), str)
        or not isinstance(repo.get("add_only_files"), list)
        or not isinstance(science, dict) or set(science) != science_keys
        or science.get("judge_model") != source.JUDGE_MODEL
        or science.get("plan_sha256") != EXPECTED_PLAN_SHA256
        or science.get("rubric_sha256") != source.RUBRIC_SHA256
        or science.get("response_schema_sha256") != source.SCHEMA_SHA256
        or science.get("accepted_judgments") != TOTAL_CALLS
        or science.get("canary_calls") != CANARY_CALLS
        or science.get("continuation_calls") != CONTINUATION_CALLS
        or science.get("canary_start_index") != 0
        or science.get("canary_end_index_exclusive") != 1
        or science.get("continuation_start_index") != 1
        or science.get("continuation_end_index_exclusive") != 240
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
        or budget.get("prior_consumed_authority_cap_usd")
        != PRIOR_CONSUMED_AUTHORITY_CAP_USD
        or budget.get("prior_api_actual_cost_usd_known") is not False
        or budget.get("historical_attempts_min") != PRIOR_ATTEMPTS_MIN
        or budget.get("historical_attempts_max") != PRIOR_ATTEMPTS_MAX
        or budget.get("v4_continuation_authorized") is not False
        or budget.get("planned_v5_canary_cap_usd") != CANARY_MAX_COST_USD
        or budget.get("planned_v5_continuation_cap_usd")
        != CONTINUATION_MAX_COST_USD
        or budget.get("planned_v5_total_cap_usd") != NEW_RECOVERY_CAP_USD
        or budget.get("conservative_program_max_usd")
        != CONSERVATIVE_PROGRAM_MAX_USD
        or budget.get("program_ceiling_usd") != PROGRAM_CEILING_USD
        or budget.get("within_program_ceiling") is not True
        or budget.get("remaining_ceiling_gap_usd") != 0.581742
    ):
        raise ValueError("Judge recovery-v5 manifest contract differs")
    _audit_failed_v4(body)
    return {
        "path": os.path.abspath(path), "file_sha256": base.sha256_file(path),
        "payload_sha256": payload["payload_sha256"], "body": body,
    }


def validate_source_inputs(recovery):
    _audit_failed_v4(recovery["body"])
    inputs = _v4_validate_source_inputs(recovery)
    keys = [recovery_idempotency_key(row) for row in inputs["plan"]]
    indexed = [
        {
            "plan_index": index, "blind_id": row["blind_id"],
            "idempotency_key": keys[index],
        }
        for index, row in enumerate(inputs["plan"])
    ]
    if (
        len(keys) != TOTAL_CALLS or len(set(keys)) != TOTAL_CALLS
        or set(keys) & {row["blind_id"] for row in inputs["plan"]}
        or base.sha256_bytes(base.canonical_bytes(keys))
        != IDEMPOTENCY_CONTRACT["derived_key_list_sha256"]
        or base.sha256_bytes(base.canonical_bytes(indexed))
        != IDEMPOTENCY_CONTRACT["indexed_identity_key_list_sha256"]
    ):
        raise ValueError("Recovery idempotency range commitment differs")
    return inputs


def judge_meta(recovery, inputs):
    return {
        **_v4_judge_meta(recovery, inputs),
        "idempotency_contract": IDEMPOTENCY_CONTRACT,
    }


def authorization_body(recovery, stage, paths):
    if stage == "canary":
        start, end, calls, maximum = 0, 1, 1, CANARY_MAX_COST_USD
    elif stage == "continuation":
        start, end, calls, maximum = 1, 240, 239, CONTINUATION_MAX_COST_USD
    else:
        raise ValueError("Unknown recovery authorization stage")
    budget_acknowledgment = {
        "verified_program_actual_usd": VERIFIED_PROGRAM_ACTUAL_USD,
        "consumed_v3_authority_cap_usd": CONSUMED_V3_AUTHORITY_CAP_USD,
        "consumed_v4_canary_authority_cap_usd": CONSUMED_V4_CANARY_AUTHORITY_CAP_USD,
        "prior_consumed_authority_cap_usd": PRIOR_CONSUMED_AUTHORITY_CAP_USD,
        "prior_api_actual_unknown": True,
        "prior_network_attempts_min": PRIOR_ATTEMPTS_MIN,
        "prior_network_attempts_max": PRIOR_ATTEMPTS_MAX,
        "prior_authorities_consumed_not_reused": True,
        "stage_cap_usd": maximum,
        "new_v5_total_cap_usd": NEW_RECOVERY_CAP_USD,
        "conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "program_ceiling_usd": PROGRAM_CEILING_USD,
    }
    body = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_authorization_v1",
        "recovery_id": RECOVERY_ID,
        "recovery_manifest": base.binding(
            recovery["path"], base.load_json(recovery["path"])
        ),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "stage": stage, "authorized_start_index": start,
        "authorized_end_index_exclusive": end, "authorized_calls": calls,
        "judge_model": source.JUDGE_MODEL, "sdk_max_retries": 0,
        "max_cost_usd": maximum, "new_v5_api_cap_usd": NEW_RECOVERY_CAP_USD,
        "idempotency_contract": IDEMPOTENCY_CONTRACT,
        "stage_lock_owner": base.binding(
            paths[f"{stage}_lock_owner"],
            base.load_json(paths[f"{stage}_lock_owner"]),
        ),
        "budget_acknowledgment": budget_acknowledgment,
        "external_api_authorized": True, "permanent_single_entry": True,
        "restart_or_resume_authorized": False,
        "historical_A_reused_not_rejudged": True,
    }
    if stage == "continuation":
        success_payload = base.load_json(paths["canary_success"])
        checkpoint_payload = base.load_json(base.checkpoint_path(paths, 1))
        success_body = base.audit_seal(success_payload, paths["canary_success"])
        budget_acknowledgment["v5_canary_actual_estimated_cost_usd"] = success_body[
            "actual_estimated_cost_usd"
        ]
        body.update({
            "canary_success": base.binding(paths["canary_success"], success_payload),
            "canary_checkpoint": base.binding(
                base.checkpoint_path(paths, 1), checkpoint_payload
            ),
        })
    return body


def recovery_idempotency_key(row):
    blind_id = row.get("blind_id") if isinstance(row, dict) else None
    if not isinstance(blind_id, str) or not blind_id:
        raise ValueError("Frozen plan row lacks blind identity")
    return hashlib.sha256(f"{RECOVERY_ID}:{blind_id}".encode()).hexdigest()


def validate_call_scope(stage, index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("Judge call index schema differs")
    if stage == "canary":
        valid = index == 0
    elif stage == "continuation":
        valid = 1 <= index < TOTAL_CALLS
    else:
        valid = False
    if not valid:
        raise ValueError("Judge call falls outside its authorized stage range")


def request_body(row):
    return {
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


def call_judge(client, row, stage, index):
    validate_call_scope(stage, index)
    body = request_body(row)
    return client.chat.completions.create(
        **body,
        extra_headers={"Idempotency-Key": recovery_idempotency_key(row)},
    )


def verify_lock_owner(recovery, paths, stage, owner_token):
    if re.fullmatch(r"[0-9a-f]{64}", owner_token or "") is None:
        raise ValueError("Exact lock owner token schema differs")
    path = paths[f"{stage}_lock_owner"]
    absolute = base.require_regular(path, f"{stage} lock owner")
    if stat.S_IMODE(os.stat(absolute).st_mode) != 0o400:
        raise ValueError(f"{stage} lock owner mode differs")
    payload = base.load_json(path)
    body = base.audit_seal(payload, path)
    expected = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + f"_{stage}_lock_v1",
        "recovery_id": RECOVERY_ID,
        "stage": stage,
        "recovery_manifest": base.binding(
            recovery["path"], base.load_json(recovery["path"])
        ),
        "recovery_repo_commit": recovery["body"]["recovery_repo"]["commit"],
        "owner_token_sha256": base.sha256_bytes(owner_token.encode()),
        "permanent_single_entry": True,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }
    if body != expected:
        raise ValueError(f"{stage} lock is not owned by this judge invocation")
    return base.binding(path, payload)


def _guard_or_failure(guard, operation_stage):
    try:
        guard()
    except Exception as error:
        raise base.JudgeCallFailure(operation_stage, error) from None


def _commit_json(path, payload):
    try:
        base.atomic_json(path, payload)
    except Exception as error:
        raise base.JudgeCallFailure("artifact_commit", error) from None


def _call_and_validate(client, row, stage, index, guard, attempts):
    validate_call_scope(stage, index)
    _guard_or_failure(guard, "environment_preflight")
    attempts["count"] += 1
    try:
        response = call_judge(client, row, stage, index)
    except Exception as error:
        raise base.JudgeCallFailure("api_call", error) from None
    try:
        judgment = base.validate_response(response, row)
    except Exception as error:
        raise base.JudgeCallFailure("response_validation", error, response) from None
    return response, judgment


def run_canary(
    recovery, inputs, paths, authorization, client, owner_token, attempts
):
    guard = lambda: verify_lock_owner(recovery, paths, "canary", owner_token)
    meta = base.judge_meta(recovery, inputs)
    row = inputs["plan"][0]
    response, judgment = _call_and_validate(
        client, row, "canary", 0, guard, attempts
    )
    cost = judgment["api_usage"]["estimated_cost_usd"]
    if cost > CANARY_MAX_COST_USD + 1e-12:
        raise base.JudgeCallFailure(
            "response_validation", RuntimeError("Canary cost cap exceeded"), response
        )
    checkpoint_payload = base.seal(base.checkpoint_body(
        meta, "canary", base._authorization_binding(authorization), 1, [judgment]
    ))
    _guard_or_failure(guard, "artifact_commit")
    target = base.checkpoint_path(paths, 1)
    _commit_json(target, checkpoint_payload)
    checkpoint_record = base.binding(target, checkpoint_payload)
    success = base.seal(base.canary_success_body(
        recovery, authorization, checkpoint_record, judgment, base.utc_now()
    ))
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(paths["canary_success"], success)
    return response, success


def run_continuation(
    recovery, inputs, paths, authorization, client, owner_token, attempts
):
    guard = lambda: verify_lock_owner(recovery, paths, "continuation", owner_token)
    canary = base.load_canary_success(recovery, inputs, paths)
    judgments = list(canary["checkpoint"]["body"]["judgments"])
    meta = base.judge_meta(recovery, inputs)
    stage_cost = 0.0
    response = None
    for index in range(1, TOTAL_CALLS):
        response, judgment = _call_and_validate(
            client, inputs["plan"][index], "continuation", index, guard, attempts
        )
        stage_cost += judgment["api_usage"]["estimated_cost_usd"]
        cumulative = canary["body"]["actual_estimated_cost_usd"] + stage_cost
        if (
            stage_cost > CONTINUATION_MAX_COST_USD + 1e-12
            or cumulative > NEW_RECOVERY_CAP_USD + 1e-12
        ):
            raise base.JudgeCallFailure(
                "response_validation",
                RuntimeError("Continuation cost cap exceeded"), response,
            )
        judgments.append(judgment)
        completed = index + 1
        checkpoint = base.seal(base.checkpoint_body(
            meta, "continuation", base._authorization_binding(authorization),
            completed, judgments,
        ))
        _guard_or_failure(guard, "artifact_commit")
        _commit_json(base.checkpoint_path(paths, completed), checkpoint)
        _v5_print(
            f"Judged {completed}/{TOTAL_CALLS} blind_id={judgment['blind_id'][:12]}"
        )
    final = base.seal({
        "meta": {
            **meta,
            "canary_authorization": base._authorization_binding(canary["authorization"]),
            "continuation_authorization": base._authorization_binding(authorization),
            "actual_api_calls": TOTAL_CALLS, "canary_api_calls": 1,
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
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(paths["judgments"], final)
    terminal_checkpoint = base.binding(
        base.checkpoint_path(paths, TOTAL_CALLS),
        base.load_json(base.checkpoint_path(paths, TOTAL_CALLS)),
    )
    success = base.seal(base.continuation_success_body(
        recovery, authorization, canary, terminal_checkpoint,
        base.binding(paths["judgments"], final), stage_cost,
        final["meta"]["actual_estimated_cost_usd"], base.utc_now(),
    ))
    _guard_or_failure(guard, "artifact_commit")
    _commit_json(paths["continuation_success"], success)
    return response, success


def sdk_serialization_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during mock SDK serialization")
    recovery = load_recovery_manifest(args.recovery_manifest)
    inputs = validate_source_inputs(recovery)
    test_cases = (
        ("canary", 0),
        ("continuation", 1),
        ("continuation", 239),
    )
    try:
        import httpx
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI SDK/httpx unavailable for mock serialization") from error
    observed = []

    def handler(request):
        observed.append({
            "method": request.method, "url": str(request.url),
            "idempotency_key": request.headers.get("idempotency-key"),
            "body": json.loads(request.content.decode("utf-8")),
        })
        return httpx.Response(200, request=request, json={
            "id": "chatcmpl-local-v5-mock", "object": "chat.completion",
            "created": 0, "model": source.JUDGE_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": '{"label":"SAFE","coherence":100}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAI(
            api_key="local-mock-key", base_url="https://mock.invalid/v1/",
            max_retries=0, http_client=http_client,
        )
        for stage, index in test_cases:
            row = inputs["plan"][index]
            response = call_judge(client, row, stage, index)
            base.validate_response(response, row)
    for observed_request, (stage, index) in zip(observed, test_cases):
        row = inputs["plan"][index]
        expected_key = recovery_idempotency_key(row)
        expected_body = request_body(row)
        if (
            observed_request["method"] != "POST"
            or observed_request["url"] != "https://mock.invalid/v1/chat/completions"
            or observed_request["idempotency_key"] != expected_key
            or observed_request["idempotency_key"] == row["blind_id"]
            or observed_request["body"] != expected_body
        ):
            raise ValueError("OpenAI SDK v5 mock serialization contract differs")
    if len(observed) != len(test_cases) or len({
        item["idempotency_key"] for item in observed
    }) != len(test_cases):
        raise ValueError("OpenAI SDK v5 range serialization inventory differs")
    request_commitment = base.sha256_bytes(base.canonical_bytes([
        {
            "stage": stage,
            "index": index,
            "body_sha256": base.sha256_bytes(
                base.canonical_bytes(request_body(inputs["plan"][index]))
            ),
            "idempotency_key_sha256": base.sha256_bytes(
                recovery_idempotency_key(inputs["plan"][index]).encode()
            ),
        }
        for stage, index in test_cases
    ]))
    _v5_print(json.dumps({
        "status": "JUDGE_RECOVERY_V5_SDK_SERIALIZATION_VALIDATED",
        "mock_transport_requests": 3, "external_api_calls": 0,
        "range_commitment_sha256": request_commitment,
        "raw_idempotency_key_persisted": False,
    }, sort_keys=True))
    return 0


def external_command(args):
    stage = "continuation" if args.command == "continue" else args.command
    recovery = load_recovery_manifest(args.recovery_manifest)
    paths = recovery_paths(recovery)
    expected_manifest = os.path.join(paths["control"], "JUDGE_RECOVERY_V5_MANIFEST.json")
    if recovery["path"] != expected_manifest:
        raise ValueError("Recovery manifest path differs from namespace contract")
    inputs = validate_source_inputs(recovery)
    authorization = base.load_authorization(recovery, stage, paths)
    base.require_stage_preconditions(recovery, inputs, stage, paths)
    verify_lock_owner(recovery, paths, stage, args.owner_token)
    started_at = base.utc_now()
    operation_stage = "environment_preflight"
    attempts = {"count": 0}
    response = None
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key absent")
        operation_stage = "client_initialization"
        verify_lock_owner(recovery, paths, stage, args.owner_token)
        client = base._make_client(api_key)
        operation_stage = "api_call"
        if stage == "canary":
            response, result = run_canary(
                recovery, inputs, paths, authorization, client, args.owner_token,
                attempts,
            )
        else:
            response, result = run_continuation(
                recovery, inputs, paths, authorization, client, args.owner_token,
                attempts,
            )
        _v5_print(json.dumps({
            "status": f"JUDGE_RECOVERY_V5_{stage.upper()}_COMPLETE",
            "result_payload_sha256": result["payload_sha256"],
            "model_fallback_used": False,
        }, sort_keys=True))
        return 0
    except Exception as error:
        completed = base._completed_checkpoint_count(paths)
        original = error
        if isinstance(error, base.JudgeCallFailure):
            operation_stage = error.operation_stage
            original = error.original
            response = error.response
        stage_prior = 0 if stage == "canary" else 1
        accepted_stage_calls = max(0, completed - stage_prior)
        stage_maximum = CANARY_CALLS if stage == "canary" else CONTINUATION_CALLS
        if not isinstance(error, base.JudgeCallFailure) and (
            attempts["count"] > accepted_stage_calls
        ):
            operation_stage = "artifact_commit"
        invocation_min = invocation_max = attempts["count"]
        if not (
            accepted_stage_calls <= attempts["count"]
            <= min(stage_maximum, accepted_stage_calls + 1)
        ):
            raise RuntimeError("Exact API invocation accounting differs") from None
        planned_index = completed if completed < TOTAL_CALLS else None
        try:
            verify_lock_owner(recovery, paths, stage, args.owner_token)
            base.write_failure(
                recovery, stage, paths, base._authorization_binding(authorization),
                operation_stage, planned_index, completed, invocation_min,
                invocation_max, started_at, original, response,
            )
        except Exception:
            raise RuntimeError(
                f"{stage} failed and sanitized failure artifact could not be committed"
            ) from None
        raise RuntimeError(f"{stage} failed; see sealed sanitized failure artifact") from None
    finally:
        os.environ.pop("OPENAI_API_KEY", None)


def _configure_base():
    base.RECOVERY_ID = RECOVERY_ID
    base.EXPECTED_SOURCE_COMMIT = EXPECTED_SOURCE_COMMIT
    base.CANARY_MAX_COST_USD = CANARY_MAX_COST_USD
    base.CONTINUATION_MAX_COST_USD = CONTINUATION_MAX_COST_USD
    base.CUMULATIVE_NEW_API_CAP_USD = NEW_RECOVERY_CAP_USD
    base.VERIFIED_PROGRAM_ACTUAL_USD = VERIFIED_PROGRAM_ACTUAL_USD
    base.CONSERVATIVE_PROGRAM_MAX_USD = CONSERVATIVE_PROGRAM_MAX_USD
    base.PROGRAM_CEILING_USD = PROGRAM_CEILING_USD
    base.recovery_paths = recovery_paths
    base.load_recovery_manifest = load_recovery_manifest
    base.validate_source_inputs = validate_source_inputs
    base.authorization_body = authorization_body
    base.judge_meta = judge_meta
    base._call_and_validate = _call_and_validate
    base.run_canary = run_canary
    base.run_continuation = run_continuation
    base.print = _v5_print


_configure_base()


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("validate-static", base.static_command),
        ("validate-plan", base.plan_command),
        ("validate-sdk-serialization", sdk_serialization_command),
        ("audit-canary", base.audit_canary_command),
        ("audit-continuation", base.audit_continuation_command),
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
    return getattr(base, name)


if __name__ == "__main__":
    main()
