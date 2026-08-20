#!/usr/bin/env python3
"""Merge historical A/B1 and new B2/B3 judge evidence without new calls.

The resulting artifact is an aggregate evidence view consumed by the frozen
all-replica gate.  Its 400 calls are explicitly partitioned into a historical
240-call authorization and the new Wave-2 160-call authorization; this command
does not contact an API.
"""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import judge_massive_union_medical as judge  # noqa: E402
import summarize_massive_union_components as components  # noqa: E402


HISTORICAL_RAW_SHA256 = "359a8e2351c855bceaea8400cb97a32f62a82f64f7b13b09839a120746a94ca2"
OLD_MODELS = {"pi_base", "pi_A", "pi_B1"}
NEW_MODELS = {"pi_B2", "pi_B3"}
ORDER = ("pi_base", "pi_A", "pi_B1", "pi_B2", "pi_B3")


def _assert_external(evidence, expected_models, expected_calls, expected_cap, label):
    meta = evidence["meta"]
    if (
        set(evidence["by_model"]) != expected_models
        or meta.get("protocol") != "massive_medical_union_blinded_judge_v1"
        or meta.get("judge_kind") != "external_gpt_primary"
        or meta.get("primary_confirmatory") is not True
        or meta.get("gate_eligible") is not True
        or meta.get("actual_api_calls") != expected_calls
        or meta.get("max_api_calls") != expected_calls
        or meta.get("planned_calls") != expected_calls
        or meta.get("max_cost_usd") != expected_cap
        or meta.get("sdk_max_retries") != 0
    ):
        raise ValueError(f"{label} medical evidence contract differs")
    if len(evidence["judgments"]) != expected_calls:
        raise ValueError(f"{label} medical evidence row count differs")


def merge(historical_path, new_path, output_path):
    historical = components.load_medical(historical_path)
    new = components.load_medical(new_path)
    if judge.sha256_file(historical_path) != HISTORICAL_RAW_SHA256:
        raise ValueError("historical Wave-1 judgment bytes differ")
    _assert_external(historical, OLD_MODELS, 240, 0.75, "historical")
    _assert_external(new, NEW_MODELS, 160, 0.50, "Wave-2")

    old_meta, new_meta = historical["meta"], new["meta"]
    common = (
        "protocol", "judge_kind", "judge_model", "rubric_sha256",
        "temperature", "temperature_parameter_omitted", "reasoning_effort",
        "seed", "prompt_file_path", "prompt_file_sha256",
        "max_output_tokens_per_call", "raw_source_responses_stored",
        "model_identity_sent_to_judge", "one_compact_call_per_response",
        "sdk_max_retries", "idempotency_key_is_blind_id", "pricing",
    )
    for field in common:
        if old_meta.get(field) != new_meta.get(field):
            raise ValueError(f"judge partitions differ on {field}")

    sources = old_meta["source_generations"] + new_meta["source_generations"]
    if [item.get("name") for item in sources] != list(ORDER):
        raise ValueError("aggregate source-generation order differs")
    judgments = historical["judgments"] + new["judgments"]
    blind_ids = [row.get("blind_id") for row in judgments]
    if len(judgments) != 400 or len(set(blind_ids)) != 400:
        raise ValueError("aggregate medical evidence is not 400 disjoint blinded rows")
    if set(row.get("model_name") for row in judgments) != set(ORDER):
        raise ValueError("aggregate medical model set differs")

    actual_cost = (
        old_meta["actual_estimated_cost_usd"]
        + new_meta["actual_estimated_cost_usd"]
    )
    merged_meta = dict(old_meta)
    merged_meta.update({
        "source_generations": sources,
        "planned_calls": 400,
        "max_api_calls": 400,
        "max_cost_usd": 1.25,
        "actual_api_calls": 400,
        "actual_estimated_cost_usd": actual_cost,
        "gate_eligible": True,
        "primary_confirmatory": True,
        "aggregate_evidence_only_no_calls_by_merge": True,
        "authorization_partitions": [
            {
                "name": "historical_wave1_recovery_v2",
                "models": ["pi_base", "pi_A", "pi_B1"],
                "judgment_path": os.path.abspath(historical_path),
                "judgment_file_sha256": historical["file_sha256"],
                "judgment_payload_sha256": historical["payload_sha256"],
                "actual_api_calls": 240,
                "maximum_api_calls": 240,
                "maximum_cost_usd": 0.75,
                "actual_estimated_cost_usd": old_meta["actual_estimated_cost_usd"],
            },
            {
                "name": "wave2_new_replicas",
                "models": ["pi_B2", "pi_B3"],
                "judgment_path": os.path.abspath(new_path),
                "judgment_file_sha256": new["file_sha256"],
                "judgment_payload_sha256": new["payload_sha256"],
                "actual_api_calls": 160,
                "maximum_api_calls": 160,
                "maximum_cost_usd": 0.50,
                "actual_estimated_cost_usd": new_meta["actual_estimated_cost_usd"],
            },
        ],
        "new_api_calls": 160,
        "new_api_cost_ceiling_usd": 0.50,
        "historical_api_calls_reused": 240,
    })
    payload = {
        "meta": merged_meta,
        "judgments": sorted(
            judgments,
            key=lambda row: (ORDER.index(row["model_name"]), row["question_id"], row["sample_index"]),
        ),
    }
    judge.write_or_audit(output_path, payload)
    audited = components.load_medical(output_path)
    if set(audited["by_model"]) != set(ORDER) or len(audited["judgments"]) != 400:
        raise ValueError("aggregate medical evidence failed post-write audit")
    print(f"Wrote/audited aggregate 400-row medical evidence: {output_path}")
    print("No API call was made by this merge; Wave-2 contributed exactly 160 calls.")
    return audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-judgments", required=True)
    parser.add_argument("--new-judgments", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    merge(args.historical_judgments, args.new_judgments, args.output_file)


if __name__ == "__main__":
    main()
