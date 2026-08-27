#!/usr/bin/env python3
"""CPU-only merge/final summary for judge recovery v6."""

import argparse
import builtins
import importlib.util
import json
import os
from pathlib import Path

import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6 as recovery


_V4_PATH = Path(__file__).with_name(
    "summarize_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4.py"
)
_V4_SPEC = importlib.util.spec_from_file_location(
    "_mmu_judge_recovery_v6_private_v4_summary", _V4_PATH
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError("Unable to load private v4 summary implementation")
summary = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(summary)

RECOVERY_ID = recovery.RECOVERY_ID
METHOD_IDS = recovery.source.METHOD_IDS
_real_print = builtins.print
_base_final_body = summary.final_body


def _v6_print(*values, **kwargs):
    converted = tuple(
        value.replace("JUDGE_RECOVERY_V4", "JUDGE_RECOVERY_V6").replace(
            "JUDGE_RECOVERY_V5", "JUDGE_RECOVERY_V6"
        ) if isinstance(value, str) else value
        for value in values
    )
    _real_print(*converted, **kwargs)


def load_terminal_new(recovery_manifest, inputs, paths):
    payload = recovery.load_json(paths["judgments"])
    body = recovery.audit_seal(payload, paths["judgments"])
    meta, rows = body.get("meta"), body.get("judgments")
    canary = recovery.load_canary_success(recovery_manifest, inputs, paths)
    continuation = recovery.load_authorization(recovery_manifest, "continuation", paths)
    previous = list(canary["checkpoint"]["body"]["judgments"])
    if previous[:1] != [inputs["prior_v5"]["judgment"]]:
        raise ValueError("Recovery-v6 terminal lost the exact v5 judgment prefix")
    terminal = None
    for completed in range(3, 241):
        checkpoint = recovery.audit_checkpoint(
            recovery.checkpoint_path(paths, completed), recovery_manifest,
            inputs, "continuation", continuation, completed,
        )
        if checkpoint["body"]["judgments"][:-1] != previous:
            raise ValueError("Recovery-v6 cumulative checkpoint prefix differs")
        previous = list(checkpoint["body"]["judgments"])
        terminal = checkpoint
    if terminal is None:
        raise ValueError("Recovery-v6 terminal checkpoint is absent")
    expected_rows = sorted(
        previous,
        key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]),
    )
    total_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in expected_rows)
    continuation_cost = sum(
        row["api_usage"]["estimated_cost_usd"] for row in previous[2:]
    )
    new_v6_cost = canary["body"]["stage_actual_estimated_cost_usd"] + continuation_cost
    expected_meta = {
        **recovery.judge_meta(recovery_manifest, inputs),
        "prior_v5_checkpoint": recovery_manifest["body"]["failed_recovery_v5"]["checkpoint_001"],
        "canary_authorization": recovery._authorization_binding(canary["authorization"]),
        "continuation_authorization": recovery._authorization_binding(continuation),
        "accepted_rows": 240, "prior_v5_reused_judgments": 1,
        "actual_api_calls": 239, "canary_api_calls": 1,
        "continuation_api_calls": 238,
        "actual_estimated_cost_usd": total_cost,
        "new_v6_estimated_cost_usd": new_v6_cost,
    }
    if meta != expected_meta or rows != expected_rows or len(rows) != 240:
        raise ValueError("Recovery-v6 terminal judgments differ")
    response_ids = [row["api_response_id"] for row in rows]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Recovery-v6 terminal response IDs are not unique")
    by_blind = {row["blind_id"]: row for row in inputs["plan"]}
    if len(by_blind) != 240 or {row["blind_id"] for row in rows} != set(by_blind):
        raise ValueError("Recovery-v6 terminal plan identities differ")
    for row in rows:
        recovery.audit_judgment(row, by_blind[row["blind_id"]])
    success_payload = recovery.load_json(paths["continuation_success"])
    success_body = recovery.audit_seal(success_payload, paths["continuation_success"])
    completed_at = success_body.get("completed_at")
    expected_success = recovery.continuation_success_body(
        recovery_manifest, continuation, canary, terminal["record"],
        recovery.binding(paths["judgments"], payload), continuation_cost,
        total_cost, completed_at,
    )
    if not isinstance(completed_at, str) or success_body != expected_success:
        raise ValueError("Recovery-v6 continuation success differs")
    if (
        continuation_cost > recovery.CONTINUATION_MAX_COST_USD + 1e-12
        or new_v6_cost > recovery.NEW_RECOVERY_CAP_USD + 1e-12
    ):
        raise ValueError("Recovery-v6 terminal cost exceeds authority")
    return {
        "payload": payload, "body": body, "meta": meta, "rows": rows,
        "record": recovery.binding(paths["judgments"], payload),
        "actual_estimated_cost_usd": total_cost,
        "new_v6_estimated_cost_usd": new_v6_cost,
        "prior_v5_reused_estimated_cost_usd": recovery.V5_CANARY_ACTUAL_USD,
        "canary": canary, "continuation_authorization": continuation,
    }


def merged_body(recovery_manifest, inputs, new, historical):
    rows = [*historical["rows"], *new["rows"]]
    rows.sort(key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]))
    return {
        "meta": {
            "schema_version": 1, "protocol": RECOVERY_ID + "_merged_judgments_v1",
            "recovery_id": RECOVERY_ID,
            "source_protocol_id": recovery.source.PROTOCOL_ID,
            "recovery_manifest": recovery.binding(
                recovery_manifest["path"], recovery.load_json(recovery_manifest["path"])
            ),
            "source_protocol_manifest": recovery_manifest["body"]["source_protocol_manifest"],
            "source_judge_plan": recovery_manifest["body"]["source_judge_plan"],
            "historical_A": {
                key: historical[key] for key in ("path", "file_sha256", "payload_sha256")
            },
            "new_composition": new["record"],
            "prior_v5_checkpoint": recovery_manifest["body"]["failed_recovery_v5"]["checkpoint_001"],
            "canary_authorization": recovery._authorization_binding(new["canary"]["authorization"]),
            "continuation_authorization": recovery._authorization_binding(new["continuation_authorization"]),
            "historical_A_reused_not_rejudged": True,
            "historical_A_new_api_calls": 0,
            "historical_A_source_api_calls": historical["source_actual_api_calls"],
            "historical_A_source_api_cost_usd": historical["source_actual_estimated_cost_usd"],
            "historical_A_judge_model_alias": historical["judge_model_alias"],
            "new_composition_judgment_rows": 240,
            "prior_v5_reused_judgments": 1,
            "prior_v5_reused_judgment_cost_usd": new["prior_v5_reused_estimated_cost_usd"],
            "new_composition_api_calls": 239,
            "new_composition_api_cost_usd": new["new_v6_estimated_cost_usd"],
            "composition_total_judge_cost_usd": new["actual_estimated_cost_usd"],
            "total_rows": 320, "confirmatory_claim": False,
        },
        "judgments": rows,
    }


def merge_command(args):
    recovery_manifest, inputs, paths, source_manifest = summary.load_context(
        args.recovery_manifest
    )
    if os.path.lexists(summary.merged_path(paths)):
        existing = summary.load_merged(
            recovery_manifest, inputs, paths, source_manifest
        )
        _v6_print(json.dumps({
            "status": "JUDGE_RECOVERY_V6_MERGE_ALREADY_VALID",
            "rows": len(existing["rows"]),
            "external_api_calls_during_merge": 0,
            "payload_sha256": existing["payload"]["payload_sha256"],
        }, sort_keys=True))
        return 0
    recovery.audit_continuation_command(argparse.Namespace(
        recovery_manifest=args.recovery_manifest
    ))
    new = load_terminal_new(recovery_manifest, inputs, paths)
    historical = summary.source_merge.load_historical(
        recovery_manifest["body"]["source_artifacts"]["historical_A_judgments"]["path"],
        source_manifest,
    )
    value = recovery.seal(merged_body(
        recovery_manifest, inputs, new, historical
    ))
    recovery.atomic_json(summary.merged_path(paths), value)
    _v6_print(json.dumps({
        "status": "JUDGE_RECOVERY_V6_MERGED", "rows": 320,
        "historical_A_new_api_calls": 0, "new_composition_api_calls": 239,
        "prior_v5_reused_judgments": 1,
        "external_api_calls_during_merge": 0,
        "payload_sha256": value["payload_sha256"],
    }, sort_keys=True))
    return 0


def final_body(recovery_manifest, source_manifest, benefit, prejudge, medical):
    body, status, passed = _base_final_body(
        recovery_manifest, source_manifest, benefit, prejudge, medical
    )
    body["split_authority"] = {
        "prior_v5_reused_judgments": 1,
        "v6_canary_calls": 1,
        "v6_continuation_calls": 238,
        "new_v6_api_calls": 239,
    }
    body["budget"] = {
        **body["budget"],
        "prior_v5_reused_judgment_cost_usd": medical["meta"][
            "prior_v5_reused_judgment_cost_usd"
        ],
        "new_v6_api_actual_estimated_cost_usd": medical["meta"][
            "new_composition_api_cost_usd"
        ],
        "composition_total_judge_cost_usd": medical["meta"][
            "composition_total_judge_cost_usd"
        ],
    }
    return body, status, passed


summary.recovery = recovery
summary.RECOVERY_ID = RECOVERY_ID
summary.METHOD_IDS = METHOD_IDS
summary.print = _v6_print
summary.load_terminal_new = load_terminal_new
summary.merged_body = merged_body
summary.merge_command = merge_command
summary.final_body = final_body

load_context = summary.load_context
merged_path = summary.merged_path
merge_command = summary.merge_command
load_merged = summary.load_merged
write_or_audit = summary.write_or_audit
final_command = summary.final_command
audit_final_command = summary.audit_final_command
static_command = summary.static_command
build_parser = summary.build_parser
run = summary.run


def main():
    try:
        raise SystemExit(run())
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
