#!/usr/bin/env python3
"""Fail-closed 1+(N-1) judge for sealed Kalai s=3 medical outputs.

This versioned adapter reuses the audited split-judge execution engine while
making the exact call count and budget a property of the sealed Kalai plan.
The one-call canary and N-1 continuation each have a permanent one-entry lock;
there is no resume, restart, or retry command.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import sys

import prepare_massive_medical_kalai_s3_judge_plan_v1 as plan_source


_ENGINE_PATH = (
    Path(__file__).resolve().parent
    / "judge_massive_medical_composition_contextual_baselines_split_v1.py"
)
_ENGINE_SPEC = importlib.util.spec_from_file_location(
    "_kalai_s3_private_split_judge_engine_v1", _ENGINE_PATH
)
if _ENGINE_SPEC is None or _ENGINE_SPEC.loader is None:
    raise RuntimeError(f"could not load split-judge engine: {_ENGINE_PATH}")
engine = importlib.util.module_from_spec(_ENGINE_SPEC)
_ENGINE_SPEC.loader.exec_module(engine)


WORKFLOW_ID = "massive_medical_kalai_s3_split_judge_v1"
OUTPUT_SUFFIX = "_kalai_s3_judge_v1"
ANALYSIS_SCOPE = "contextual_post_hoc_not_gated"

_original_manifest_body = engine.manifest_body
_original_authorization_body = engine.authorization_body
_original_checkpoint_body = engine.checkpoint_body
_original_add_authorization_arguments = engine.add_authorization_arguments
_original_audit_continuation = engine.audit_continuation
_original_repository_record = engine.repository_record


def _plan_payload(path):
    payload = engine.load_json(path, "sealed Kalai s=3 judge plan")
    plan_source.verify_seal(payload, "sealed Kalai s=3 judge plan")
    return payload


def configure(payload):
    """Install exact sealed-plan constants into the execution engine."""

    body = plan_source.verify_seal(payload, "sealed Kalai s=3 judge plan")
    calls = body.get("planned_calls")
    budget = body.get("budget")
    if (
        isinstance(calls, bool)
        or not isinstance(calls, int)
        or not 2 <= calls <= 80
        or body.get("continuation_calls") != calls - 1
        or not isinstance(budget, dict)
    ):
        raise ValueError("Kalai s=3 judge plan cardinality differs")
    per_call = Decimal(str(body.get("maximum_cost_per_call_usd")))
    total_cap = Decimal(str(body.get("maximum_cost_usd")))
    pre_judge = Decimal(str(budget.get("pre_judge_conservative_exposure_usd")))
    maximum = Decimal(str(budget.get("conservative_program_max_usd")))
    ceiling = Decimal(str(budget.get("program_ceiling_usd")))
    if (
        per_call != plan_source.MAX_COST_PER_CALL_USD
        or total_cap != per_call * calls
        or maximum != pre_judge + total_cap
        or ceiling != plan_source.PROGRAM_CEILING_USD
        or maximum > ceiling
    ):
        raise ValueError("Kalai s=3 judge plan budget differs")
    engine.plan_source = plan_source
    engine.WORKFLOW_ID = WORKFLOW_ID
    engine.PROTOCOL_ID = plan_source.PROTOCOL_ID
    engine.EXPECTED_PLAN_PAYLOAD_SHA256 = payload[plan_source.SEAL_FIELD]
    engine.EXPECTED_MODELS = (plan_source.SOURCE_NAME,)
    engine.TOTAL_CALLS = calls
    engine.CANARY_START = 0
    engine.CANARY_END = 1
    engine.CONTINUATION_START = 1
    engine.CONTINUATION_END = calls
    engine.CANARY_CALLS = 1
    engine.CONTINUATION_CALLS = calls - 1
    engine.MAX_COST_PER_CALL_USD = per_call
    engine.CANARY_CAP_USD = per_call
    engine.CONTINUATION_CAP_USD = per_call * (calls - 1)
    engine.TOTAL_JUDGE_CAP_USD = total_cap
    # The legacy acknowledgment field names are retained for exact CLI
    # compatibility; here "known actual" is the sealed pre-judge conservative
    # exposure and retained exposure is zero because it is already included.
    engine.KNOWN_PROGRAM_ACTUAL_USD = pre_judge
    engine.RETAINED_PRIOR_EXPOSURE_USD = Decimal("0")
    engine.CURRENT_CONSERVATIVE_EXPOSURE_USD = pre_judge
    engine.CONSERVATIVE_PROGRAM_MAX_USD = maximum
    engine.PROGRAM_CEILING_USD = ceiling
    return calls


def _configure_from_cli(argv):
    if not argv:
        return
    command = argv[0]
    flag = "--judge-plan" if command == "prepare" else "--manifest"
    try:
        value = argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"{flag} is required to configure the judge") from error
    if command == "prepare":
        plan_path = value
    else:
        manifest = engine.load_json(value, "Kalai judge manifest")
        manifest_body = engine.audit_seal(manifest, "Kalai judge manifest")
        plan_path = manifest_body.get("judge_plan", {}).get("path")
        if not isinstance(plan_path, str):
            raise ValueError("Kalai judge manifest lacks its plan binding")
    configure(_plan_payload(plan_path))


def load_plan_context(plan_path):
    absolute, descriptor = engine.require_regular(
        plan_path, "sealed Kalai s=3 judge plan"
    )
    os.close(descriptor)
    payload = _plan_payload(absolute)
    body = plan_source.verify_seal(payload, "sealed Kalai s=3 judge plan")
    configure(payload)
    sources = body.get("source_generations")
    rows = body.get("plan")
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != plan_source.PROTOCOL_ID
        or body.get("method_id") != plan_source.METHOD_ID
        or body.get("protocol") != plan_source.PLAN_PROTOCOL_ID
        or body.get("analysis_scope") != ANALYSIS_SCOPE
        or body.get("primary_gate_eligible") is not False
        or body.get("judge_model") != plan_source.JUDGE_MODEL
        or body.get("sdk_retries") != 0
        or body.get("rubric_sha256")
        != engine.digest(plan_source.RUBRIC.encode("utf-8"))
        or body.get("response_schema_sha256")
        != engine.digest(engine.canonical(plan_source.JUDGE_SCHEMA))
        or body.get("planned_calls") != engine.TOTAL_CALLS
        or body.get("canary_calls") != 1
        or body.get("continuation_calls") != engine.CONTINUATION_CALLS
        or body.get("reference_panel_not_rejudged") is not True
        or body.get("abstentions_are_not_judged_or_reclassified") is not True
        or body.get("contains_question_or_response_text") is not False
        or body.get("external_api_calls") != 0
        or not isinstance(sources, list)
        or len(sources) != 1
        or sources[0].get("name") != plan_source.SOURCE_NAME
        or not isinstance(rows, list)
        or len(rows) != engine.TOTAL_CALLS
    ):
        raise ValueError("Kalai s=3 judge plan contract differs")
    prompt_path = body.get("prompt_file_path")
    rebuilt = plan_source.build_plan(
        body.get("completion_result", {}).get("path"), prompt_path
    )
    if rebuilt != payload:
        raise ValueError("Kalai judge plan does not round-trip from sealed inputs")
    _, _, prompts = plan_source.load_prompts(prompt_path)
    chain = plan_source.load_completion_chain(
        body["completion_result"]["path"]
    )
    samples, coverage = plan_source.eligible_rows(chain)
    if coverage != sources[0].get("accounting"):
        raise ValueError("Kalai judge source coverage binding differs")
    source_rows = {
        (
            sample.get("question_id"), sample.get("sample_index"),
            sample.get("prompt_sha256"), sample.get("response_sha256"),
            sample.get("sample_sha256"),
        ): sample
        for sample in samples
    }
    if len(source_rows) != len(samples):
        raise ValueError("eligible Kalai source contains duplicate identities")
    exact_keys = {
        "blind_id", "model_name", "question_id", "sample_index",
        "prompt_sha256", "response_sha256", "source_sample_sha256", "plan_index",
    }
    runtime_rows = []
    seen = set()
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != exact_keys
            or row.get("plan_index") != index
            or row.get("model_name") != plan_source.SOURCE_NAME
            or engine.HEX64.fullmatch(str(row.get("blind_id", ""))) is None
            or row["blind_id"] in seen
        ):
            raise ValueError("Kalai judge plan row differs")
        key = (
            row["question_id"], row["sample_index"], row["prompt_sha256"],
            row["response_sha256"], row["source_sample_sha256"],
        )
        sample = source_rows.get(key)
        prompt = prompts.get(row["question_id"])
        if (
            sample is None
            or prompt is None
            or sample.get("accepted") is not True
            or sample.get("abstained") is not False
            or sample.get("finish_reason") != "stop"
            or not sample.get("response")
        ):
            raise ValueError("Kalai judge row is not accepted and nonempty")
        runtime = {
            **row,
            "question": prompt["prompt"],
            "response": sample["response"],
            "finish_reason": sample["finish_reason"],
        }
        rendered = plan_source.RUBRIC.format(
            question=runtime["question"], response=runtime["response"]
        )
        if len(rendered.encode("utf-8")) + 64 > engine.MAX_INPUT_TOKENS:
            raise ValueError("rendered Kalai judge request exceeds input cap")
        runtime_rows.append(runtime)
        seen.add(row["blind_id"])
    if len(runtime_rows) != engine.TOTAL_CALLS:
        raise ValueError("reconstructed Kalai judge cardinality differs")
    return {
        "path": absolute,
        "payload": payload,
        "body": body,
        "record": engine.binding(absolute, payload),
        "rows": runtime_rows,
    }


def workflow_paths(output_root):
    root = os.path.realpath(os.path.abspath(output_root))
    if not root.endswith(OUTPUT_SUFFIX):
        raise ValueError("Kalai judge output namespace suffix differs")
    control = os.path.join(root, "control")
    medical = os.path.join(root, "evaluation", "medical")
    logs = os.path.join(root, "logs")
    return {
        "root": root, "control": control, "medical": medical, "logs": logs,
        "manifest": os.path.join(control, "JUDGE_STAGE_MANIFEST.json"),
        "staged": os.path.join(control, "CPU_STAGED.json"),
        "canary_lock": os.path.join(control, "CANARY_LOCK.json"),
        "canary_authorization": os.path.join(control, "CANARY_AUTHORIZATION.json"),
        "canary_run_started": os.path.join(control, "CANARY_RUN_STARTED.json"),
        "canary_success": os.path.join(control, "CANARY_SUCCESS.json"),
        "canary_failure": os.path.join(control, "CANARY_FAILURE.json"),
        "continuation_lock": os.path.join(control, "CONTINUATION_LOCK.json"),
        "continuation_authorization": os.path.join(control, "CONTINUATION_AUTHORIZATION.json"),
        "continuation_run_started": os.path.join(control, "CONTINUATION_RUN_STARTED.json"),
        "continuation_success": os.path.join(control, "CONTINUATION_SUCCESS.json"),
        "continuation_failure": os.path.join(control, "CONTINUATION_FAILURE.json"),
        "checkpoint_base": os.path.join(medical, "judge_checkpoint.json"),
        "judgments": os.path.join(medical, "judgments_kalai_s3.json"),
    }


def repository_record(repo_root):
    """Bind only a clean checkout, including on every manifest reload."""

    record = _original_repository_record(repo_root)
    status = engine.subprocess.check_output(
        ["git", "-C", record["path"], "status", "--porcelain"], text=True
    )
    if status.strip():
        raise ValueError("Kalai judge repository is not clean")
    return record


def manifest_body(plan, repo, output_root):
    body = _original_manifest_body(plan, repo, output_root)
    body["analysis_scope"] = ANALYSIS_SCOPE
    body["method_id"] = plan_source.METHOD_ID
    body["completion_result"] = plan["body"]["completion_result"]
    body["assembly"] = plan["body"]["assembly"]
    body["continuation"] = {
        "start": 1,
        "end_exclusive": engine.TOTAL_CALLS,
        "calls": engine.CONTINUATION_CALLS,
        "cap_usd": float(engine.CONTINUATION_CAP_USD),
    }
    body["budget"] = {
        **plan["body"]["budget"],
        "within_program_ceiling": True,
        "unused_terminal_authority_is_not_cost_exposure": True,
        "unused_terminal_authority_is_nonreusable": True,
    }
    body.pop("historical_A_reused_not_rejudged", None)
    body["reference_panel_not_rejudged"] = True
    return body


def checkpoint_body(manifest, stage, authorization_record, completed, judgments):
    body = _original_checkpoint_body(
        manifest, stage, authorization_record, completed, judgments
    )
    body["analysis_scope"] = ANALYSIS_SCOPE
    return body


def authorization_body(manifest, stage, lock, canary=None):
    body = _original_authorization_body(manifest, stage, lock, canary)
    body.pop("historical_A_reused_not_rejudged", None)
    body["reference_panel_not_rejudged"] = True
    return body


def validate_call_scope(stage, index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("judge call index schema differs")
    valid = (
        index == 0
        if stage == "canary"
        else stage == "continuation" and 1 <= index < engine.TOTAL_CALLS
    )
    if not valid:
        raise ValueError("judge call falls outside its authorized range")


def stage_values(stage):
    if stage == "canary":
        return 0, 1, 1, engine.CANARY_CAP_USD
    if stage == "continuation":
        return (
            1, engine.TOTAL_CALLS, engine.CONTINUATION_CALLS,
            engine.CONTINUATION_CAP_USD,
        )
    raise ValueError("unknown external judge stage")


def sdk_serialization_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during offline serialization")
    manifest = engine.load_manifest(args.manifest)
    completions = engine._FakeCompletions()
    client = type("Client", (), {})()
    client.chat = type("Chat", (), {})()
    client.chat.completions = completions
    cases = (
        ("canary", 0),
        ("continuation", 1),
        ("continuation", engine.TOTAL_CALLS - 1),
    )
    for stage, index in cases:
        engine.call_judge(client, manifest["plan"]["rows"][index], stage, index)
    if len(completions.calls) != 3:
        raise ValueError("offline Kalai request inventory differs")
    for call, (stage, index) in zip(completions.calls, cases):
        row = manifest["plan"]["rows"][index]
        headers = call.pop("extra_headers")
        if (
            headers != {"Idempotency-Key": engine.idempotency_key(row)}
            or headers["Idempotency-Key"] == row["blind_id"]
            or call != engine.request_body(row)
        ):
            raise ValueError(f"offline {stage} SDK serialization differs")
    print(json.dumps({
        "status": "KALAI_S3_JUDGE_OFFLINE_SERIALIZATION_VALID",
        "fake_client_calls": 3,
        "planned_calls": engine.TOTAL_CALLS,
        "external_api_calls": 0,
    }, sort_keys=True))
    return 0


def load_success(manifest, stage, authorization=None):
    path = manifest["paths"][f"{stage}_success"]
    payload = engine.load_json(path, f"{stage} success")
    body = engine.audit_seal(payload, f"{stage} success")
    auth = (
        engine.load_authorization(manifest, stage)
        if authorization is None else authorization
    )
    completed = 1 if stage == "canary" else engine.TOTAL_CALLS
    checkpoint = engine.audit_checkpoint(manifest, stage, auth, completed)
    timestamp = body.get("completed_at")
    expected = engine.success_body(
        manifest, stage, auth, checkpoint,
        body.get("stage_api_call_invocations_exact"),
        engine.decimal(body.get("stage_actual_estimated_cost_usd"), "stage actual cost"),
        engine.decimal(body.get("cumulative_accepted_estimated_cost_usd"), "cumulative cost"),
    )
    expected["completed_at"] = timestamp
    _, end, calls, cap = stage_values(stage)
    stage_cost = engine.decimal(body.get("stage_actual_estimated_cost_usd"), "stage cost")
    cumulative = engine.decimal(
        body.get("cumulative_accepted_estimated_cost_usd"), "cumulative cost"
    )
    judgments = checkpoint["body"]["judgments"]
    expected_cumulative = engine._cost(judgments)
    expected_stage = engine._cost(
        judgments if stage == "canary" else judgments[1:]
    )
    try:
        parsed = engine.dt.datetime.fromisoformat(timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{stage} success timestamp differs") from error
    if (
        body != expected
        or body.get("stage_api_call_invocations_exact") != calls
        or body.get("completed_calls") != end
        or stage_cost > cap
        or stage_cost != expected_stage
        or cumulative != expected_cumulative
        or parsed.tzinfo is None
    ):
        raise ValueError(f"{stage} success differs")
    return {
        "payload": payload, "body": body,
        "record": engine.binding(path, payload),
        "authorization": auth, "checkpoint": checkpoint,
    }


def run_continuation(manifest, authorization, client, attempts):
    canary = engine.load_success(manifest, "canary")
    judgments = list(canary["checkpoint"]["body"]["judgments"])
    stage_cost = Decimal("0")
    for index in range(1, engine.TOTAL_CALLS):
        _, judgment = engine._call_and_validate(
            client, manifest, authorization, "continuation", index, attempts
        )
        if judgment["api_response_id"] in {
            item["api_response_id"] for item in judgments
        }:
            raise engine.JudgeCallFailure(
                "response_validation", RuntimeError("judge response ID was reused")
            )
        stage_cost += engine.decimal(
            judgment["api_usage"]["estimated_cost_usd"], "row cost"
        )
        if stage_cost > engine.CONTINUATION_CAP_USD:
            raise engine.JudgeCallFailure(
                "response_validation", RuntimeError("continuation cap exceeded")
            )
        judgments.append(judgment)
        engine._write_checkpoint(manifest, "continuation", authorization, judgments)
    final = engine.seal(final_judgments_body(manifest, judgments))
    try:
        engine.atomic_json(manifest["paths"]["judgments"], final)
    except Exception as error:
        raise engine.JudgeCallFailure("artifact_commit", error) from None
    checkpoint = engine.audit_checkpoint(
        manifest, "continuation", authorization, engine.TOTAL_CALLS
    )
    success = engine.seal(engine.success_body(
        manifest, "continuation", authorization, checkpoint, attempts["count"],
        stage_cost, engine._cost(judgments),
    ))
    try:
        engine.atomic_json(manifest["paths"]["continuation_success"], success)
    except Exception as error:
        raise engine.JudgeCallFailure("artifact_commit", error) from None
    return success


def final_judgments_body(manifest, judgments):
    """Return the one exact terminal Kalai judgment schema."""

    return {
        "meta": {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "protocol_id": plan_source.PROTOCOL_ID,
            "method_id": plan_source.METHOD_ID,
            "analysis_scope": ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "judge_model": plan_source.JUDGE_MODEL,
            "sdk_retries": 0,
            "judge_plan": manifest["body"]["judge_plan"],
            "completion_result": manifest["body"]["completion_result"],
            "reference_panel_not_rejudged": True,
            "abstentions_not_judged_or_reclassified": True,
            "actual_api_calls": engine.TOTAL_CALLS,
            "canary_api_calls": 1,
            "continuation_api_calls": engine.CONTINUATION_CALLS,
            "actual_estimated_cost_usd": float(engine._cost(judgments)),
            "restart_or_resume_used": False,
        },
        "completed_calls": engine.TOTAL_CALLS,
        "coverage": manifest["plan"]["body"]["source_generations"][0]["accounting"],
        "judgments": judgments,
    }


def audit_continuation(manifest):
    """Audit both the inherited checkpoint chain and the exact Kalai result."""

    result = _original_audit_continuation(manifest)
    final = engine.load_json(
        manifest["paths"]["judgments"], "terminal Kalai s=3 judgments"
    )
    body = engine.audit_seal(final, "terminal Kalai s=3 judgments")
    judgments = body.get("judgments")
    if not isinstance(judgments, list) or body != final_judgments_body(
        manifest, judgments
    ):
        raise ValueError("terminal Kalai s=3 judgment schema differs")
    return result


def add_authorization_arguments(parser):
    _original_add_authorization_arguments(parser)
    for action in parser._actions:
        if action.dest == "ack_contextual_post_hoc_only":
            alias = "--ack-exploratory-post-hoc-only"
            if alias not in action.option_strings:
                action.option_strings.append(alias)
            parser._option_string_actions[alias] = action
            return
    raise RuntimeError("contextual post-hoc acknowledgment action is absent")


def status_command(args):
    manifest = engine.load_manifest(args.manifest)
    paths = manifest["paths"]
    if os.path.lexists(paths["continuation_failure"]):
        engine.audit_failure(manifest, "continuation")
        state = "CONTINUATION_TERMINAL_FAILURE_NO_RESTART"
    elif os.path.lexists(paths["continuation_success"]):
        engine.audit_continuation(manifest)
        state = "COMPLETE"
    elif os.path.lexists(paths["continuation_run_started"]):
        engine.audit_run_started(manifest, "continuation")
        state = "CONTINUATION_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
    elif os.path.lexists(paths["continuation_lock"]):
        state = "CONTINUATION_AUTHORIZED_OR_LOCKED"
    elif os.path.lexists(paths["canary_failure"]):
        engine.audit_failure(manifest, "canary")
        state = "CANARY_TERMINAL_FAILURE_NO_RESTART"
    elif os.path.lexists(paths["canary_success"]):
        engine.audit_canary(manifest)
        state = (
            "CANARY_COMPLETE_AWAITING_SEPARATE_"
            f"{engine.CONTINUATION_CALLS}_CALL_AUTHORIZATION"
        )
    elif os.path.lexists(paths["canary_run_started"]):
        engine.audit_run_started(manifest, "canary")
        state = "CANARY_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
    elif os.path.lexists(paths["canary_lock"]):
        state = "CANARY_AUTHORIZED_OR_LOCKED"
    else:
        engine.audit_staged(manifest)
        state = "CPU_STAGED_AWAITING_SEPARATE_ONE_CALL_AUTHORIZATION"
    print(f"KALAI_S3_JUDGE_{state}")
    return 0


def install_adapter():
    engine.load_plan_context = load_plan_context
    engine.workflow_paths = workflow_paths
    engine.repository_record = repository_record
    engine.manifest_body = manifest_body
    engine.checkpoint_body = checkpoint_body
    engine.authorization_body = authorization_body
    engine.validate_call_scope = validate_call_scope
    engine.stage_values = stage_values
    engine.sdk_serialization_command = sdk_serialization_command
    engine.load_success = load_success
    engine.run_continuation = run_continuation
    engine.audit_continuation = audit_continuation
    engine.add_authorization_arguments = add_authorization_arguments
    engine.status_command = status_command


install_adapter()


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    _configure_from_cli(arguments)
    return engine.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
