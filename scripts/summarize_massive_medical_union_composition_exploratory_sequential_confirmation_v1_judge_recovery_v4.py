#!/usr/bin/env python3
"""Merge and summarize the split-authority recovery without rejudging A."""

import argparse
import json
import math
import os

import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4 as recovery
import merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as source_merge
import summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as source_summary


RECOVERY_ID = recovery.RECOVERY_ID
METHOD_IDS = recovery.source.METHOD_IDS
MERGED_NAME = "judgments_merged.json"
SUMMARY_NAME = "summary.json"


def load_context(manifest_path):
    recovery_manifest = recovery.load_recovery_manifest(manifest_path)
    inputs = recovery.validate_source_inputs(recovery_manifest)
    paths = recovery.recovery_paths(recovery_manifest)
    source_manifest = source_summary.load_manifest(
        recovery_manifest["body"]["source_protocol_manifest"]["path"]
    )
    if (
        source_manifest["file_sha256"]
        != recovery_manifest["body"]["source_protocol_manifest"]["file_sha256"]
        or source_manifest["payload_sha256"]
        != recovery_manifest["body"]["source_protocol_manifest"]["manifest_payload_sha256"]
    ):
        raise ValueError("Recovery source protocol manifest differs")
    return recovery_manifest, inputs, paths, source_manifest


def load_terminal_new(recovery_manifest, inputs, paths):
    payload = recovery.load_json(paths["judgments"])
    body = recovery.audit_seal(payload, paths["judgments"])
    meta, rows = body.get("meta"), body.get("judgments")
    canary = recovery.load_canary_success(recovery_manifest, inputs, paths)
    continuation = recovery.load_authorization(
        recovery_manifest, "continuation", paths
    )
    previous_judgments = list(canary["checkpoint"]["body"]["judgments"])
    terminal = None
    for completed in range(2, 241):
        checkpoint = recovery.audit_checkpoint(
            recovery.checkpoint_path(paths, completed), recovery_manifest,
            inputs, "continuation", continuation, completed,
        )
        if checkpoint["body"]["judgments"][:-1] != previous_judgments:
            raise ValueError("Recovery cumulative checkpoint prefix differs")
        previous_judgments = list(checkpoint["body"]["judgments"])
        terminal = checkpoint
    if terminal is None:
        raise ValueError("Recovery terminal checkpoint is absent")
    expected_rows = sorted(
        terminal["body"]["judgments"],
        key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]),
    )
    expected_cost = sum(
        row["api_usage"]["estimated_cost_usd"] for row in expected_rows
    )
    expected_meta = {
        **recovery.judge_meta(recovery_manifest, inputs),
        "canary_authorization": recovery._authorization_binding(
            canary["authorization"]
        ),
        "continuation_authorization": recovery._authorization_binding(continuation),
        "actual_api_calls": 240,
        "canary_api_calls": 1,
        "continuation_api_calls": 239,
        "actual_estimated_cost_usd": expected_cost,
    }
    if meta != expected_meta or rows != expected_rows or len(rows) != 240:
        raise ValueError("Recovery terminal new judgments differ")
    if len({row["api_response_id"] for row in rows}) != 240:
        raise ValueError("Recovery terminal response IDs are not unique")
    by_blind = {row["blind_id"]: row for row in inputs["plan"]}
    if len(by_blind) != 240 or {row["blind_id"] for row in rows} != set(by_blind):
        raise ValueError("Recovery terminal plan identities differ")
    for row in rows:
        recovery.audit_judgment(row, by_blind[row["blind_id"]])
    continuation_payload = recovery.load_json(paths["continuation_success"])
    continuation_body = recovery.audit_seal(
        continuation_payload, paths["continuation_success"]
    )
    completed_at = continuation_body.get("completed_at")
    stage_cost = sum(
        row["api_usage"]["estimated_cost_usd"]
        for row in terminal["body"]["judgments"][1:]
    )
    expected_success = recovery.continuation_success_body(
        recovery_manifest, continuation, canary, terminal["record"],
        recovery.binding(paths["judgments"], payload), stage_cost,
        expected_cost, completed_at,
    )
    if not isinstance(completed_at, str) or continuation_body != expected_success:
        raise ValueError("Recovery continuation success contract differs")
    if (
        stage_cost > recovery.CONTINUATION_MAX_COST_USD + 1e-12
        or expected_cost > recovery.CUMULATIVE_NEW_API_CAP_USD + 1e-12
    ):
        raise ValueError("Recovery terminal cost exceeds its authority")
    return {
        "payload": payload, "body": body, "meta": meta, "rows": rows,
        "record": recovery.binding(paths["judgments"], payload),
        "actual_estimated_cost_usd": expected_cost,
        "canary": canary, "continuation_authorization": continuation,
    }


def merged_path(paths):
    return os.path.join(paths["medical"], MERGED_NAME)


def merged_body(recovery_manifest, inputs, new, historical):
    rows = [*historical["rows"], *new["rows"]]
    rows.sort(key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]))
    return {
        "meta": {
            "schema_version": 1,
            "protocol": RECOVERY_ID + "_merged_judgments_v1",
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
            "canary_authorization": recovery._authorization_binding(
                new["canary"]["authorization"]
            ),
            "continuation_authorization": recovery._authorization_binding(
                new["continuation_authorization"]
            ),
            "historical_A_reused_not_rejudged": True,
            "historical_A_new_api_calls": 0,
            "historical_A_source_api_calls": historical["source_actual_api_calls"],
            "historical_A_source_api_cost_usd": historical[
                "source_actual_estimated_cost_usd"
            ],
            "historical_A_judge_model_alias": historical["judge_model_alias"],
            "new_composition_api_calls": 240,
            "new_composition_api_cost_usd": new["actual_estimated_cost_usd"],
            "total_rows": 320,
            "confirmatory_claim": False,
        },
        "judgments": rows,
    }


def merge_command(args):
    recovery_manifest, inputs, paths, source_manifest = load_context(
        args.recovery_manifest
    )
    if os.path.lexists(merged_path(paths)):
        existing = load_merged(
            recovery_manifest, inputs, paths, source_manifest
        )
        print(json.dumps({
            "status": "JUDGE_RECOVERY_V4_MERGE_ALREADY_VALID",
            "rows": len(existing["rows"]),
            "external_api_calls_during_merge": 0,
            "payload_sha256": existing["payload"]["payload_sha256"],
        }, sort_keys=True))
        return 0
    # This performs the exact checkpoint/inventory audit before the first
    # derived file changes the terminal judge directory inventory.
    recovery.audit_continuation_command(argparse.Namespace(
        recovery_manifest=args.recovery_manifest
    ))
    new = load_terminal_new(recovery_manifest, inputs, paths)
    historical = source_merge.load_historical(
        recovery_manifest["body"]["source_artifacts"]["historical_A_judgments"]["path"],
        source_manifest,
    )
    value = recovery.seal(merged_body(recovery_manifest, inputs, new, historical))
    recovery.atomic_json(merged_path(paths), value)
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_MERGED",
        "rows": 320, "historical_A_new_api_calls": 0,
        "new_composition_api_calls": 240,
        "external_api_calls_during_merge": 0,
        "payload_sha256": value["payload_sha256"],
    }, sort_keys=True))
    return 0


def load_merged(recovery_manifest, inputs, paths, source_manifest):
    path = merged_path(paths)
    payload = recovery.load_json(path)
    body = recovery.audit_seal(payload, path)
    meta, rows = body.get("meta"), body.get("judgments")
    new = load_terminal_new(recovery_manifest, inputs, paths)
    historical = source_merge.load_historical(
        recovery_manifest["body"]["source_artifacts"]["historical_A_judgments"]["path"],
        source_manifest,
    )
    if body != merged_body(recovery_manifest, inputs, new, historical):
        raise ValueError("Recovery merged judgments differ")
    if not isinstance(rows, list) or len(rows) != 320:
        raise ValueError("Recovery merged row count differs")
    by_model = {}
    for row in rows:
        source_summary.validate_judgment_row(row)
        by_model.setdefault(row["model_name"], []).append(row)
    if set(by_model) != {"pi_A", *METHOD_IDS}:
        raise ValueError("Recovery merged model registry differs")
    for name in by_model:
        pairs = {(row["question_id"], row["sample_index"]) for row in by_model[name]}
        if len(by_model[name]) != 80 or len(pairs) != 80:
            raise ValueError("Recovery merged exact80 registry differs")
    return {
        "path": path, "payload": payload, "body": body, "meta": meta,
        "rows": rows, "by_model": by_model,
        "record": recovery.binding(path, payload),
    }


def write_or_audit(path, value):
    if os.path.lexists(path):
        observed = recovery.load_json(path)
        recovery.audit_seal(observed, path)
        if observed != value:
            raise ValueError(f"Existing derived artifact differs: {path}")
        return False
    recovery.atomic_json(path, value)
    return True


def final_body(
    recovery_manifest, source_manifest, benefit, prejudge, medical,
):
    gates = source_manifest["body"]["gates"]["medical_each_method"]
    a_metrics = source_summary.medical_metrics(medical["by_model"]["pi_A"])
    results, checks = {}, {}
    for name in METHOD_IDS:
        metrics = source_summary.medical_metrics(medical["by_model"][name])
        gap = a_metrics["bad_rate"] - metrics["bad_rate"]
        cluster_ci = source_summary.prompt_cluster_bad_gap_ci(
            medical["by_model"]["pi_A"], medical["by_model"][name]
        )
        arm = {
            "medical_all_finish_reason_stop": all(
                row["source_finish_reason"] == "stop"
                for row in medical["by_model"][name]
            ),
            "medical_bad_rate": metrics["bad_rate"] <= gates["medical_bad_rate_max"],
            "medical_A_minus_method_bad_rate": gap
            >= gates["medical_A_minus_method_bad_rate_min"],
            "medical_prompt_cluster_bootstrap_lower": cluster_ci[0]
            > gates["medical_prompt_cluster_bootstrap_95ci_lower_gt"],
            "medical_coherence": metrics["coherent_rate"]
            >= gates["medical_coherence_ge_50_fraction_min"],
            "medical_refusal_rate": metrics["refusal_rate"]
            <= gates["medical_refusal_rate_max"],
            "medical_unparseable": metrics["unparseable_count"]
            <= gates["medical_unparseable_max"],
        }
        results[name] = {
            "benefit": benefit["methods"][name],
            "medical": {
                **metrics, "A_bad_rate": a_metrics["bad_rate"],
                "A_minus_method_bad_rate": gap,
                "A_minus_method_prompt_cluster_bootstrap_95ci": cluster_ci,
            },
            "checks": arm, "passed": all(arm.values()),
        }
        checks.update({f"{name}.{key}": value for key, value in arm.items()})
    passed = all(checks.values()) and set(results) == set(METHOD_IDS)
    status = (
        "EXPLORATORY_SEQUENTIAL_SUPPORT"
        if passed else "EXPLORATORY_SEQUENTIAL_NO_SUPPORT"
    )
    body = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_v1",
        "recovery_id": RECOVERY_ID,
        "source_protocol_id": recovery.source.PROTOCOL_ID,
        "recovery_manifest": recovery.binding(
            recovery_manifest["path"], recovery.load_json(recovery_manifest["path"])
        ),
        "source_protocol_manifest": recovery_manifest["body"]["source_protocol_manifest"],
        "source_v1_terminal": source_manifest["body"]["source_v1_terminal"],
        "benefit_gate": benefit,
        "medical_prejudge": {
            key: prejudge[key] for key in (
                "path", "file_sha256", "payload_sha256", "summary_path",
                "summary_file_sha256", "summary_payload_sha256",
            )
        },
        "medical_judgments": medical["record"],
        "thresholds": gates,
        "bootstrap_seed": source_summary.BOOTSTRAP_SEED,
        "bootstrap_replicates": source_summary.BOOTSTRAP_REPLICATES,
        "A_medical": a_metrics,
        "methods": results, "checks": checks,
        "all_three_methods_required": True,
        "all_three_methods_passed": passed,
        "benefit_pass_required_and_preserved": True,
        "historical_A_reused_not_rejudged": True,
        "method_or_metric_rescue_allowed": False,
        "split_authority": {"canary_calls": 1, "continuation_calls": 239},
        "budget": {
            "source_registry": source_summary.audit_budget(source_manifest["body"]),
            "recovery_contract": recovery_manifest["body"]["budget_contract"],
            "new_api_actual_estimated_cost_usd": medical["meta"][
                "new_composition_api_cost_usd"
            ],
        },
        "status": status,
        **source_manifest["flags"],
    }
    return body, status, passed


def final_command(args):
    recovery_manifest, inputs, paths, source_manifest = load_context(
        args.recovery_manifest
    )
    benefit_path = recovery_manifest["body"]["source_artifacts"]["benefit_gate"]["path"]
    prejudge_path = recovery_manifest["body"]["source_artifacts"]["prejudge_gate"]["path"]
    benefit = source_summary.load_benefit_gate(benefit_path, source_manifest)
    prejudge = source_summary.load_prejudge(prejudge_path, source_manifest)
    if prejudge["benefit_gate"] != benefit:
        raise ValueError("Recovery final benefit/prejudge binding differs")
    medical = load_merged(recovery_manifest, inputs, paths, source_manifest)
    body, status, passed = final_body(
        recovery_manifest, source_manifest, benefit, prejudge, medical
    )
    final_dir = os.path.join(paths["root"], "evaluation", "final")
    observed_final = set(os.listdir(final_dir))
    if not observed_final <= {SUMMARY_NAME, status}:
        raise ValueError("Recovery partial final inventory differs")
    summary = recovery.seal(body)
    summary_path = os.path.join(final_dir, SUMMARY_NAME)
    write_or_audit(summary_path, summary)
    sentinel = recovery.seal({
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_sentinel_v1",
        "recovery_id": RECOVERY_ID,
        "status": status,
        "summary": recovery.binding(summary_path, summary),
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_summary": 0,
        "confirmatory_claim": False,
    })
    sentinel_path = os.path.join(final_dir, status)
    write_or_audit(sentinel_path, sentinel)
    final_result = recovery.seal({
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_result_v1",
        "recovery_id": RECOVERY_ID,
        "status": status,
        "summary": recovery.binding(summary_path, summary),
        "sentinel": recovery.binding(sentinel_path, sentinel),
        "judgments_merged": medical["record"],
        "external_api_calls_during_summary": 0,
        "gpu_jobs_during_summary": 0,
    })
    write_or_audit(
        os.path.join(paths["control"], "FINAL_RESULT.json"), final_result
    )
    print(status)
    return 0 if passed else 2


def audit_final_command(args):
    recovery_manifest, inputs, paths, source_manifest = load_context(
        args.recovery_manifest
    )
    medical = load_merged(recovery_manifest, inputs, paths, source_manifest)
    final_result_path = os.path.join(paths["control"], "FINAL_RESULT.json")
    final_result_payload = recovery.load_json(final_result_path)
    final_result = recovery.audit_seal(final_result_payload, final_result_path)
    status = final_result.get("status")
    if status not in {
        "EXPLORATORY_SEQUENTIAL_SUPPORT", "EXPLORATORY_SEQUENTIAL_NO_SUPPORT",
    }:
        raise ValueError("Recovery final status differs")
    final_dir = os.path.join(paths["root"], "evaluation", "final")
    if set(os.listdir(final_dir)) != {SUMMARY_NAME, status}:
        raise ValueError("Recovery final inventory differs")
    summary_path = os.path.join(final_dir, SUMMARY_NAME)
    sentinel_path = os.path.join(final_dir, status)
    summary_payload = recovery.load_json(summary_path)
    summary_body = recovery.audit_seal(summary_payload, summary_path)
    sentinel_payload = recovery.load_json(sentinel_path)
    sentinel_body = recovery.audit_seal(sentinel_payload, sentinel_path)
    benefit = source_summary.load_benefit_gate(
        recovery_manifest["body"]["source_artifacts"]["benefit_gate"]["path"],
        source_manifest,
    )
    prejudge = source_summary.load_prejudge(
        recovery_manifest["body"]["source_artifacts"]["prejudge_gate"]["path"],
        source_manifest,
    )
    expected_summary, expected_status, _passed = final_body(
        recovery_manifest, source_manifest, benefit, prejudge, medical
    )
    expected_sentinel = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_sentinel_v1",
        "recovery_id": RECOVERY_ID,
        "status": expected_status,
        "summary": recovery.binding(summary_path, summary_payload),
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_summary": 0,
        "confirmatory_claim": False,
    }
    expected_result = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_result_v1",
        "recovery_id": RECOVERY_ID,
        "status": expected_status,
        "summary": recovery.binding(summary_path, summary_payload),
        "sentinel": recovery.binding(sentinel_path, sentinel_payload),
        "judgments_merged": medical["record"],
        "external_api_calls_during_summary": 0,
        "gpu_jobs_during_summary": 0,
    }
    if (
        summary_body != expected_summary
        or sentinel_body != expected_sentinel
        or final_result != expected_result
    ):
        raise ValueError("Recovery final artifact contract differs")
    print(json.dumps({
        "status": expected_status,
        "summary_payload_sha256": summary_payload["payload_sha256"],
        "external_api_calls_during_audit": 0,
    }, sort_keys=True))
    return 0


def static_command(args):
    recovery_manifest, _inputs, _paths, source_manifest = load_context(
        args.recovery_manifest
    )
    source_summary.audit_budget(source_manifest["body"])
    print(json.dumps({
        "status": "JUDGE_RECOVERY_V4_SUMMARY_STATIC_VALIDATED",
        "recovery_manifest_payload_sha256": recovery_manifest["payload_sha256"],
        "historical_A_reused_not_rejudged": True,
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("validate-static", static_command), ("merge", merge_command),
        ("final", final_command), ("audit-final", audit_final_command),
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
