#!/usr/bin/env python3
"""CPU-only derivation recovery v7 from the immutable v6 judge terminal.

V7 makes no model or network calls.  It re-audits the complete v6 checkpoint
chain, computes cost in chronological checkpoint order, and only then sorts
rows for presentation.  All derived artifacts are written to a fresh v7
namespace; the sealed v6 namespace is read-only provenance.
"""

import argparse
import json
import math
import os
from pathlib import Path
import stat

import audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7 as control
import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v6 as source_v6
import merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as source_merge
import summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as source_summary


RECOVERY_ID = control.RECOVERY_ID
METHOD_IDS = source_v6.source.METHOD_IDS
MERGED_NAME = "judgments_merged.json"
SUMMARY_NAME = "summary.json"
FINAL_RESULT_NAME = "FINAL_RESULT.json"
SUPPORT = "EXPLORATORY_SEQUENTIAL_SUPPORT"
NO_SUPPORT = "EXPLORATORY_SEQUENTIAL_NO_SUPPORT"

V6_ACCEPTED_ROWS = 240
V6_REUSED_V5_ROWS = 1
V6_NEW_API_CALLS = 239
V6_CANARY_CALLS = 1
V6_CONTINUATION_CALLS = 238
V6_REUSED_V5_COST_USD = source_v6.V5_CANARY_ACTUAL_USD
V6_CHRONOLOGICAL_COST_USD = 0.031268499999999984
V6_SORTED_PRESENTATION_COST_USD = 0.0312685
V6_CHRONOLOGICAL_MINUS_SORTED_USD = -1.3877787807814457e-17
V6_NEW_CHRONOLOGICAL_COST_USD = 0.03115399999999998


def _require_cpu_only():
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU-only v7 derivation")


def _binding(path):
    return control.binding(path, require_seal=True)


def _manifest_binding(manifest):
    return _binding(manifest["path"])


def _sort_key(row):
    return row["model_name"], row["question_id"], row["sample_index"]


def _estimated_cost(row):
    try:
        value = row["api_usage"]["estimated_cost_usd"]
    except (KeyError, TypeError) as error:
        raise ValueError("V6 judgment cost schema differs") from error
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("V6 judgment cost value differs")
    return value


def chronological_cost(rows):
    """Sum rows in their supplied chronology; callers must not pre-sort."""
    return sum(_estimated_cost(row) for row in rows)


def _source_v6_binding(body, name, observed):
    expected = body["source_v6_terminal"].get(name)
    if observed != expected:
        raise ValueError(f"V7 source-v6 {name} binding differs")


def _validate_cost_order_contract(body, chronological_rows, presentation_rows):
    chronological = chronological_cost(chronological_rows)
    # This second sum is diagnostic evidence for the v6 derivation failure.  It
    # is never used to validate or replace the authoritative terminal total.
    sorted_presentation = sorted(chronological_rows, key=_sort_key)
    presentation_cost = chronological_cost(sorted_presentation)
    diagnostic = {
        "bug_class": "floating_point_nonassociativity_after_presentation_sort",
        "chronological_cost_usd": chronological,
        "sorted_presentation_cost_usd": presentation_cost,
        "chronological_minus_sorted_usd": chronological - presentation_cost,
        "new_v6_chronological_cost_usd": V6_NEW_CHRONOLOGICAL_COST_USD,
        "presentation_rows_equal_sorted_chronological_rows": (
            presentation_rows == sorted_presentation
        ),
        "chronological_rows_equal_presentation_rows": (
            chronological_rows == presentation_rows
        ),
        "repair": "sum_checkpoint_chronology_before_sorting_rows_for_presentation",
    }
    expected = {
        "bug_class": "floating_point_nonassociativity_after_presentation_sort",
        "chronological_cost_usd": V6_CHRONOLOGICAL_COST_USD,
        "sorted_presentation_cost_usd": V6_SORTED_PRESENTATION_COST_USD,
        "chronological_minus_sorted_usd": V6_CHRONOLOGICAL_MINUS_SORTED_USD,
        "new_v6_chronological_cost_usd": V6_NEW_CHRONOLOGICAL_COST_USD,
        "presentation_rows_equal_sorted_chronological_rows": True,
        "chronological_rows_equal_presentation_rows": False,
        "repair": "sum_checkpoint_chronology_before_sorting_rows_for_presentation",
    }
    if diagnostic != expected or body.get("cost_order_recovery") != expected:
        raise ValueError("V7 chronological cost-order recovery contract differs")
    return diagnostic


def load_context(derive_manifest_path):
    """Load v7 control context and independently re-audit immutable v6."""
    _require_cpu_only()
    derive_manifest = control.audit_manifest_exact(derive_manifest_path)
    source = control.audit_source_v6()
    v6_manifest = source["manifest"]
    inputs = source["inputs"]
    v6_paths = source["paths"]
    body = derive_manifest["body"]
    if (
        body.get("source_v6_recovery_id") != source_v6.RECOVERY_ID
        or body.get("source_protocol_id") != source_v6.source.PROTOCOL_ID
    ):
        raise ValueError("V7 source recovery/protocol identity differs")
    if _binding(v6_manifest["path"]) != body["source_v6_manifest"]:
        raise ValueError("V7 source-v6 manifest binding differs")
    source_manifest = source_summary.load_manifest(
        body["source_protocol_manifest"]["path"]
    )
    if (
        source_manifest["file_sha256"]
        != body["source_protocol_manifest"]["file_sha256"]
        or source_manifest["payload_sha256"]
        != body["source_protocol_manifest"]["manifest_payload_sha256"]
    ):
        raise ValueError("V7 source protocol manifest differs")
    paths = control.derive_paths(derive_manifest)
    return derive_manifest, source, v6_manifest, inputs, v6_paths, paths, source_manifest


def load_source_v6_terminal(derive_manifest, source):
    """Reconstruct and validate the exact v6 terminal in checkpoint chronology."""
    body = derive_manifest["body"]
    v6_manifest = source["manifest"]
    inputs = source["inputs"]
    paths = source["paths"]
    chronological_rows = list(source["chronological_rows"])
    presentation_rows = list(source["presentation_rows"])

    if len(chronological_rows) != V6_ACCEPTED_ROWS:
        raise ValueError("V7 source-v6 chronological row count differs")
    if len(presentation_rows) != V6_ACCEPTED_ROWS:
        raise ValueError("V7 source-v6 presentation row count differs")

    canary = source_v6.load_canary_success(v6_manifest, inputs, paths)
    continuation = source_v6.load_authorization(v6_manifest, "continuation", paths)
    previous = list(canary["checkpoint"]["body"]["judgments"])
    if previous != chronological_rows[:2]:
        raise ValueError("V7 source-v6 canary prefix differs")
    terminal = None
    for completed in range(3, V6_ACCEPTED_ROWS + 1):
        checkpoint = source_v6.audit_checkpoint(
            source_v6.checkpoint_path(paths, completed),
            v6_manifest, inputs, "continuation", continuation, completed,
        )
        if checkpoint["body"]["judgments"][:-1] != previous:
            raise ValueError("V7 source-v6 cumulative checkpoint prefix differs")
        previous = list(checkpoint["body"]["judgments"])
        terminal = checkpoint
    if terminal is None or previous != chronological_rows:
        raise ValueError("V7 source-v6 terminal chronology differs")

    judgments_payload = source_v6.load_json(paths["judgments"])
    judgments_body = source_v6.audit_seal(judgments_payload, paths["judgments"])
    meta = judgments_body.get("meta")
    rows = judgments_body.get("judgments")
    if rows != presentation_rows or rows != sorted(chronological_rows, key=_sort_key):
        raise ValueError("V7 source-v6 presentation rows differ")

    total_cost = chronological_cost(chronological_rows)
    continuation_cost = chronological_cost(chronological_rows[2:])
    new_v6_cost = canary["body"]["stage_actual_estimated_cost_usd"] + continuation_cost
    diagnostic = _validate_cost_order_contract(
        body, chronological_rows, presentation_rows
    )
    if total_cost != V6_CHRONOLOGICAL_COST_USD:
        raise ValueError("V7 source-v6 chronological total differs")
    if new_v6_cost != V6_NEW_CHRONOLOGICAL_COST_USD:
        raise ValueError("V7 source-v6 new-call chronological total differs")

    expected_meta = {
        **source_v6.judge_meta(v6_manifest, inputs),
        "prior_v5_checkpoint": v6_manifest["body"]["failed_recovery_v5"]["checkpoint_001"],
        "canary_authorization": source_v6._authorization_binding(
            canary["authorization"]
        ),
        "continuation_authorization": source_v6._authorization_binding(continuation),
        "accepted_rows": V6_ACCEPTED_ROWS,
        "prior_v5_reused_judgments": V6_REUSED_V5_ROWS,
        "actual_api_calls": V6_NEW_API_CALLS,
        "canary_api_calls": V6_CANARY_CALLS,
        "continuation_api_calls": V6_CONTINUATION_CALLS,
        "actual_estimated_cost_usd": total_cost,
        "new_v6_estimated_cost_usd": new_v6_cost,
    }
    if meta != expected_meta:
        raise ValueError("V7 source-v6 terminal metadata differs")

    response_ids = [row.get("api_response_id") for row in chronological_rows]
    if any(not isinstance(item, str) or not item for item in response_ids):
        raise ValueError("V7 source-v6 response ID schema differs")
    if len(set(response_ids)) != V6_ACCEPTED_ROWS:
        raise ValueError("V7 source-v6 response IDs are not unique")
    by_blind = {row["blind_id"]: row for row in inputs["plan"]}
    if (
        len(by_blind) != V6_ACCEPTED_ROWS
        or {row["blind_id"] for row in chronological_rows} != set(by_blind)
    ):
        raise ValueError("V7 source-v6 plan identities differ")
    for row in chronological_rows:
        source_v6.audit_judgment(row, by_blind[row["blind_id"]])

    success_payload = source_v6.load_json(paths["continuation_success"])
    success_body = source_v6.audit_seal(
        success_payload, paths["continuation_success"]
    )
    completed_at = success_body.get("completed_at")
    expected_success = source_v6.continuation_success_body(
        v6_manifest, continuation, canary, terminal["record"],
        source_v6.binding(paths["judgments"], judgments_payload),
        continuation_cost, total_cost, completed_at,
    )
    if not isinstance(completed_at, str) or success_body != expected_success:
        raise ValueError("V7 source-v6 continuation success differs")
    if (
        continuation_cost > source_v6.CONTINUATION_MAX_COST_USD + 1e-12
        or new_v6_cost > source_v6.NEW_RECOVERY_CAP_USD + 1e-12
    ):
        raise ValueError("V7 source-v6 cost exceeds its frozen authority")

    observed_bindings = {
        "canary_success": _binding(paths["canary_success"]),
        "continuation_authorization": _binding(paths["continuation_authorization"]),
        "continuation_success": _binding(paths["continuation_success"]),
        "checkpoint_002": _binding(source_v6.checkpoint_path(paths, 2)),
        "checkpoint_240": _binding(source_v6.checkpoint_path(paths, 240)),
        "judgments_new": _binding(paths["judgments"]),
    }
    for name, observed in observed_bindings.items():
        _source_v6_binding(body, name, observed)
    source_bindings = source.get("terminal_bindings")
    if (
        source_bindings != observed_bindings
        or {
            key: body["source_v6_terminal"].get(key) for key in observed_bindings
        } != observed_bindings
        or body["source_v6_terminal"].get(
            "inventory_name_mode_type_stream_sha256"
        ) != control.SOURCE_INVENTORY_STREAM_SHA256
        or body["source_v6_terminal"].get("terminal_file_count")
        != control.SOURCE_INVENTORY_FILE_COUNT
        or set(body["source_v6_terminal"]) != {
            "inventory_name_mode_type_stream_sha256", "terminal_file_count",
            *observed_bindings,
        }
    ):
        raise ValueError("V7 control/core source-v6 terminal bindings differ")
    if source.get("cost_order_recovery") != diagnostic:
        raise ValueError("V7 control/core cost-order evidence differs")
    return {
        "rows": presentation_rows,
        "chronological_rows": chronological_rows,
        "meta": meta,
        "record": observed_bindings["judgments_new"],
        "total_cost_usd": total_cost,
        "new_v6_cost_usd": new_v6_cost,
        "continuation_cost_usd": continuation_cost,
        "cost_order_recovery": diagnostic,
        "terminal_bindings": observed_bindings,
    }


def _historical(derive_manifest, source_manifest):
    return source_merge.load_historical(
        derive_manifest["body"]["source_artifacts"]["historical_A_judgments"]["path"],
        source_manifest,
    )


def merged_body(derive_manifest, source_v6_terminal, historical):
    rows = [*historical["rows"], *source_v6_terminal["rows"]]
    rows.sort(key=_sort_key)
    return {
        "meta": {
            "schema_version": 1,
            "protocol": RECOVERY_ID + "_merged_judgments_v1",
            "recovery_id": RECOVERY_ID,
            "source_protocol_id": source_v6.source.PROTOCOL_ID,
            "source_v6_recovery_id": source_v6.RECOVERY_ID,
            "derive_manifest": _manifest_binding(derive_manifest),
            "source_v6_manifest": derive_manifest["body"]["source_v6_manifest"],
            "source_v6_terminal": derive_manifest["body"]["source_v6_terminal"],
            "source_protocol_manifest": derive_manifest["body"]["source_protocol_manifest"],
            "source_judge_plan": derive_manifest["body"]["source_judge_plan"],
            "historical_A": {
                key: historical[key]
                for key in ("path", "file_sha256", "payload_sha256")
            },
            "source_v6_new_composition": source_v6_terminal["record"],
            "historical_A_reused_not_rejudged": True,
            "historical_A_new_api_calls": 0,
            "historical_A_source_api_calls": historical["source_actual_api_calls"],
            "historical_A_source_api_cost_usd": historical[
                "source_actual_estimated_cost_usd"
            ],
            "historical_A_judge_model_alias": historical["judge_model_alias"],
            "source_v6_composition_judgment_rows": V6_ACCEPTED_ROWS,
            "prior_v5_reused_judgments": V6_REUSED_V5_ROWS,
            "prior_v5_reused_judgment_cost_usd": V6_REUSED_V5_COST_USD,
            "source_v6_new_api_calls": V6_NEW_API_CALLS,
            "source_v6_new_api_cost_usd": source_v6_terminal["new_v6_cost_usd"],
            "source_v6_total_judge_cost_usd": source_v6_terminal["total_cost_usd"],
            "cost_order_recovery": source_v6_terminal["cost_order_recovery"],
            "derivation_external_api_calls": 0,
            "derivation_gpu_jobs": 0,
            "total_rows": 320,
            "confirmatory_claim": False,
        },
        "judgments": rows,
    }


def merged_path(paths):
    return os.fspath(paths["judgments_merged"])


def _write_or_audit(path, value):
    path = Path(path)
    if os.path.lexists(path):
        observed = control.load_json(path)
        control.audit_seal(observed, path)
        if observed != value:
            raise ValueError(f"Existing v7 derived artifact differs: {path}")
        if stat.S_IMODE(path.stat().st_mode) != 0o400:
            raise ValueError(f"Existing v7 derived artifact mode differs: {path}")
        return False
    control.atomic_json(path, value, mode=0o400)
    return True


def _load_merged_from_context(derive_manifest, source, paths, source_manifest):
    terminal = load_source_v6_terminal(derive_manifest, source)
    historical = _historical(derive_manifest, source_manifest)
    path = merged_path(paths)
    payload = control.load_json(path)
    body = control.audit_seal(payload, path)
    expected = merged_body(derive_manifest, terminal, historical)
    if body != expected:
        raise ValueError("V7 merged judgments differ")
    rows = body.get("judgments")
    if not isinstance(rows, list) or len(rows) != 320:
        raise ValueError("V7 merged row count differs")
    by_model = {}
    for row in rows:
        source_summary.validate_judgment_row(row)
        by_model.setdefault(row["model_name"], []).append(row)
    if set(by_model) != {"pi_A", *METHOD_IDS}:
        raise ValueError("V7 merged model registry differs")
    for name, model_rows in by_model.items():
        pairs = {(row["question_id"], row["sample_index"]) for row in model_rows}
        if len(model_rows) != 80 or len(pairs) != 80:
            raise ValueError(f"V7 merged exact80 registry differs: {name}")
    if stat.S_IMODE(Path(path).stat().st_mode) != 0o400:
        raise ValueError("V7 merged artifact mode differs")
    return {
        "path": path, "payload": payload, "body": body,
        "meta": body["meta"], "rows": rows, "by_model": by_model,
        "record": _binding(path),
    }


def load_merged(derive_manifest_path):
    control.audit_derive_namespace()
    context = load_context(derive_manifest_path)
    derive_manifest, source, _v6_manifest, _inputs, _v6_paths, paths, source_manifest = context
    return _load_merged_from_context(derive_manifest, source, paths, source_manifest)


def merge_command(args):
    control.audit_derive_namespace()
    context = load_context(args.derive_manifest)
    derive_manifest, source, _v6_manifest, _inputs, _v6_paths, paths, source_manifest = context
    path = merged_path(paths)
    if os.path.lexists(path):
        merged = _load_merged_from_context(
            derive_manifest, source, paths, source_manifest
        )
        print(json.dumps({
            "status": "JUDGE_DERIVE_RECOVERY_V7_MERGE_ALREADY_VALID",
            "rows": len(merged["rows"]),
            "external_api_calls_during_merge": 0,
            "gpu_jobs_during_merge": 0,
            "payload_sha256": merged["payload"]["payload_sha256"],
        }, sort_keys=True))
        return 0
    terminal = load_source_v6_terminal(derive_manifest, source)
    historical = _historical(derive_manifest, source_manifest)
    value = control.seal(merged_body(derive_manifest, terminal, historical))
    _write_or_audit(path, value)
    print(json.dumps({
        "status": "JUDGE_DERIVE_RECOVERY_V7_MERGED",
        "rows": 320,
        "historical_A_new_api_calls": 0,
        "source_v6_new_api_calls": V6_NEW_API_CALLS,
        "derivation_external_api_calls": 0,
        "derivation_gpu_jobs": 0,
        "payload_sha256": value["payload_sha256"],
    }, sort_keys=True))
    return 0


def final_body(derive_manifest, source_manifest, benefit, prejudge, medical):
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
                **metrics,
                "A_bad_rate": a_metrics["bad_rate"],
                "A_minus_method_bad_rate": gap,
                "A_minus_method_prompt_cluster_bootstrap_95ci": cluster_ci,
            },
            "checks": arm,
            "passed": all(arm.values()),
        }
        checks.update({f"{name}.{key}": value for key, value in arm.items()})
    passed = all(checks.values()) and set(results) == set(METHOD_IDS)
    status = SUPPORT if passed else NO_SUPPORT
    body = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_v1",
        "recovery_id": RECOVERY_ID,
        "source_protocol_id": source_v6.source.PROTOCOL_ID,
        "source_v6_recovery_id": source_v6.RECOVERY_ID,
        "derive_manifest": _manifest_binding(derive_manifest),
        "source_v6_manifest": derive_manifest["body"]["source_v6_manifest"],
        "source_v6_terminal": derive_manifest["body"]["source_v6_terminal"],
        "source_protocol_manifest": derive_manifest["body"]["source_protocol_manifest"],
        "source_v1_terminal": source_manifest["body"]["source_v1_terminal"],
        "benefit_gate": benefit,
        "medical_prejudge": {
            key: prejudge[key]
            for key in (
                "path", "file_sha256", "payload_sha256", "summary_path",
                "summary_file_sha256", "summary_payload_sha256",
            )
        },
        "medical_judgments": medical["record"],
        "thresholds": gates,
        "bootstrap_seed": source_summary.BOOTSTRAP_SEED,
        "bootstrap_replicates": source_summary.BOOTSTRAP_REPLICATES,
        "A_medical": a_metrics,
        "methods": results,
        "checks": checks,
        "all_three_methods_required": True,
        "all_three_methods_passed": passed,
        "benefit_pass_required_and_preserved": True,
        "historical_A_reused_not_rejudged": True,
        "method_or_metric_rescue_allowed": False,
        "source_v6_split_authority": {
            "prior_v5_reused_judgments": V6_REUSED_V5_ROWS,
            "prior_v5_reused_judgment_cost_usd": V6_REUSED_V5_COST_USD,
            "v6_canary_calls": V6_CANARY_CALLS,
            "v6_continuation_calls": V6_CONTINUATION_CALLS,
            "v6_new_api_calls": V6_NEW_API_CALLS,
        },
        "cost_order_recovery": medical["meta"]["cost_order_recovery"],
        "budget": {
            "source_registry": source_summary.audit_budget(source_manifest["body"]),
            "source_v6_contract": derive_manifest["body"]["source_v6_budget_contract"],
            "source_v6_new_api_actual_estimated_cost_usd": medical["meta"][
                "source_v6_new_api_cost_usd"
            ],
            "source_v6_total_judge_cost_usd": medical["meta"][
                "source_v6_total_judge_cost_usd"
            ],
            "prior_v5_reused_judgment_cost_usd": V6_REUSED_V5_COST_USD,
            "v7_derivation_external_api_calls": 0,
            "v7_derivation_gpu_jobs": 0,
        },
        "status": status,
        **source_manifest["flags"],
    }
    return body, status, passed


def _source_gates(derive_manifest, source_manifest):
    body = derive_manifest["body"]
    benefit = source_summary.load_benefit_gate(
        body["source_artifacts"]["benefit_gate"]["path"], source_manifest
    )
    prejudge = source_summary.load_prejudge(
        body["source_artifacts"]["prejudge_gate"]["path"], source_manifest
    )
    if prejudge["benefit_gate"] != benefit:
        raise ValueError("V7 final benefit/prejudge binding differs")
    return benefit, prejudge


def _final_paths(paths, status):
    final_dir = Path(paths["final"])
    return (
        final_dir / SUMMARY_NAME,
        final_dir / status,
        Path(paths["final_result"]),
    )


def final_command(args):
    control.audit_derive_namespace()
    context = load_context(args.derive_manifest)
    derive_manifest, source, _v6_manifest, _inputs, _v6_paths, paths, source_manifest = context
    medical = _load_merged_from_context(derive_manifest, source, paths, source_manifest)
    benefit, prejudge = _source_gates(derive_manifest, source_manifest)
    body, status, passed = final_body(
        derive_manifest, source_manifest, benefit, prejudge, medical
    )
    summary_path, sentinel_path, final_result_path = _final_paths(paths, status)
    observed = set(os.listdir(paths["final"]))
    if not observed <= {SUMMARY_NAME, status}:
        raise ValueError("V7 partial final inventory differs")
    summary_payload = control.seal(body)
    _write_or_audit(summary_path, summary_payload)
    sentinel_payload = control.seal({
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_sentinel_v1",
        "recovery_id": RECOVERY_ID,
        "status": status,
        "summary": _binding(summary_path),
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_summary": 0,
        "gpu_jobs_during_summary": 0,
        "confirmatory_claim": False,
    })
    _write_or_audit(sentinel_path, sentinel_payload)
    result_payload = control.seal({
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_result_v1",
        "recovery_id": RECOVERY_ID,
        "status": status,
        "derive_manifest": _manifest_binding(derive_manifest),
        "source_v6_judgments_new": derive_manifest["body"]["source_v6_terminal"][
            "judgments_new"
        ],
        "judgments_merged": medical["record"],
        "summary": _binding(summary_path),
        "sentinel": _binding(sentinel_path),
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_derivation": 0,
        "gpu_jobs_during_derivation": 0,
    })
    _write_or_audit(final_result_path, result_payload)
    print(status)
    return 0 if passed else 2


def audit_final_command(args):
    # The shell freezes the derivation lock to 0400 only after this semantic
    # artifact audit, then asks the control plane for its complete-state audit.
    control.audit_derive_namespace()
    context = load_context(args.derive_manifest)
    derive_manifest, source, _v6_manifest, _inputs, _v6_paths, paths, source_manifest = context
    medical = _load_merged_from_context(derive_manifest, source, paths, source_manifest)
    benefit, prejudge = _source_gates(derive_manifest, source_manifest)
    expected_summary, status, _passed = final_body(
        derive_manifest, source_manifest, benefit, prejudge, medical
    )
    summary_path, sentinel_path, final_result_path = _final_paths(paths, status)
    if set(os.listdir(paths["final"])) != {SUMMARY_NAME, status}:
        raise ValueError("V7 final inventory differs")
    summary_payload = control.load_json(summary_path)
    summary_body = control.audit_seal(summary_payload, summary_path)
    sentinel_payload = control.load_json(sentinel_path)
    sentinel_body = control.audit_seal(sentinel_payload, sentinel_path)
    result_payload = control.load_json(final_result_path)
    result_body = control.audit_seal(result_payload, final_result_path)
    expected_sentinel = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_sentinel_v1",
        "recovery_id": RECOVERY_ID,
        "status": status,
        "summary": _binding(summary_path),
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_summary": 0,
        "gpu_jobs_during_summary": 0,
        "confirmatory_claim": False,
    }
    expected_result = {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_final_result_v1",
        "recovery_id": RECOVERY_ID,
        "status": status,
        "derive_manifest": _manifest_binding(derive_manifest),
        "source_v6_judgments_new": derive_manifest["body"]["source_v6_terminal"][
            "judgments_new"
        ],
        "judgments_merged": medical["record"],
        "summary": _binding(summary_path),
        "sentinel": _binding(sentinel_path),
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_derivation": 0,
        "gpu_jobs_during_derivation": 0,
    }
    for path in (summary_path, sentinel_path, final_result_path):
        if stat.S_IMODE(path.stat().st_mode) != 0o400:
            raise ValueError(f"V7 final artifact mode differs: {path}")
    if (
        summary_body != expected_summary
        or sentinel_body != expected_sentinel
        or result_body != expected_result
    ):
        raise ValueError("V7 final artifact contract differs")
    print(json.dumps({
        "status": status,
        "summary_payload_sha256": summary_payload["payload_sha256"],
        "historical_A_reused_not_rejudged": True,
        "external_api_calls_during_audit": 0,
        "gpu_jobs_during_audit": 0,
    }, sort_keys=True))
    return 0


def static_command(args):
    context = load_context(args.derive_manifest)
    derive_manifest, source, _v6_manifest, _inputs, _v6_paths, _paths, source_manifest = context
    terminal = load_source_v6_terminal(derive_manifest, source)
    source_summary.audit_budget(source_manifest["body"])
    print(json.dumps({
        "status": "JUDGE_DERIVE_RECOVERY_V7_SUMMARY_STATIC_VALIDATED",
        "derive_manifest_payload_sha256": derive_manifest["payload_sha256"],
        "source_v6_chronological_cost_usd": terminal["total_cost_usd"],
        "source_v6_sorted_presentation_cost_usd": terminal[
            "cost_order_recovery"
        ]["sorted_presentation_cost_usd"],
        "historical_A_reused_not_rejudged": True,
        "external_api_calls": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("validate-static", static_command),
        ("merge", merge_command),
        ("final", final_command),
        ("audit-final", audit_final_command),
    ):
        command = commands.add_parser(name)
        command.add_argument("--derive-manifest", required=True)
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
