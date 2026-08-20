#!/usr/bin/env python3
"""Fail-closed audit of the prospective Wave-3 composition protocol tree."""

import argparse
import os

import prepare_massive_medical_union_wave3_protocol as preparation


EXPECTED_ARTIFACT_PATHS = {
    "smoke/prompts.json",
    "smoke/answers.json",
    "confirmation/prompts.json",
    "confirmation/answers.json",
    "medical/prompts.json",
}


def _load_protocol_artifact(root, relative, description):
    payload, raw = preparation.load_json(
        os.path.join(root, *relative.split("/")), description
    )
    return payload, {
        "path": relative,
        "size_bytes": len(raw),
        "sha256": preparation.sha256_bytes(raw),
    }


def _require_equal(observed, expected, description):
    if observed != expected:
        raise ValueError(f"{description} differs from the prospective contract")


def audit_protocol(output_root, massive_root, union_root):
    output_root = os.path.abspath(output_root)
    manifest, manifest_raw = preparation.load_json(
        os.path.join(output_root, preparation.MANIFEST_NAME),
        "Wave-3 protocol manifest",
    )
    seal = preparation.verify_seal(manifest, "Wave-3 protocol manifest")
    _require_equal(manifest.get("schema_version"), preparation.SCHEMA_VERSION, "schema")
    _require_equal(manifest.get("protocol_id"), preparation.PROTOCOL_ID, "protocol ID")
    _require_equal(manifest.get("prospective"), True, "prospective flag")
    _require_equal(manifest.get("methods"), preparation.method_registry(), "method registry")
    _require_equal(
        manifest.get("generation"), preparation.generation_registry(), "generation registry"
    )
    _require_equal(manifest.get("judge"), preparation.judge_registry(), "judge registry")
    _require_equal(manifest.get("gates"), preparation.gate_registry(), "gate registry")
    _require_equal(manifest.get("budget"), preparation.budget_registry(), "budget registry")
    _require_equal(
        manifest.get("component_panel"),
        {
            "ordered_roles": ["A", "B1", "B2", "B3"],
            "identities_bound_only_after_wave2_go": True,
            "wave2_go_does_not_release_wave3": True,
        },
        "component-panel contract",
    )

    inventory = preparation.file_inventory(output_root)
    _require_equal(manifest.get("file_inventory"), inventory, "protocol file inventory")
    observed_paths = {item["path"] for item in inventory}
    _require_equal(observed_paths, EXPECTED_ARTIFACT_PATHS, "protocol artifact paths")

    (
        dev_prompts,
        dev_answers,
        test_prompts,
        test_answers,
        medical_source,
        source_bindings,
    ) = preparation._load_parents(massive_root, union_root)
    _require_equal(
        manifest.get("source_bindings"), source_bindings, "current parent-data bindings"
    )
    smoke_prompts, smoke_answers, smoke_selection = preparation.balanced_subset(
        dev_prompts,
        dev_answers,
        preparation.SMOKE_PER_INTENT,
        "dev",
        preparation.SMOKE_SEED,
    )
    confirmation_prompts, confirmation_answers, confirmation_selection = (
        preparation.balanced_subset(
            test_prompts,
            test_answers,
            preparation.CONFIRMATION_PER_INTENT,
            "sealed_test",
            preparation.CONFIRMATION_SEED,
        )
    )
    _require_equal(
        manifest.get("subsets"),
        {"smoke": smoke_selection, "confirmation": confirmation_selection},
        "deterministic subset selections",
    )
    artifact_expectations = {
        "smoke/prompts.json": smoke_prompts,
        "smoke/answers.json": smoke_answers,
        "confirmation/prompts.json": confirmation_prompts,
        "confirmation/answers.json": confirmation_answers,
        "medical/prompts.json": medical_source,
    }
    for relative, expected in artifact_expectations.items():
        observed, _ = _load_protocol_artifact(
            output_root, relative, f"Wave-3 {relative}"
        )
        _require_equal(observed, expected, f"Wave-3 {relative}")

    expected_medical_ids = [
        f"medical_official16_{index:02d}"
        for index in range(preparation.MEDICAL_PROMPTS)
    ]
    expected_medical_ids_sha = preparation.sha256_bytes(
        preparation.canonical_json_bytes(expected_medical_ids)
    )
    _require_equal(
        manifest.get("medical_question_ids_sha256"),
        expected_medical_ids_sha,
        "medical question-ID binding",
    )
    return {
        "protocol_id": preparation.PROTOCOL_ID,
        "manifest_raw_sha256": preparation.sha256_bytes(manifest_raw),
        "manifest_payload_sha256": seal,
        "method_ids": [item["method_id"] for item in preparation.method_registry()],
        "smoke_rows": smoke_selection["rows"],
        "confirmation_rows": confirmation_selection["rows"],
        "medical_samples_per_method": (
            preparation.MEDICAL_PROMPTS * preparation.MEDICAL_SAMPLES_PER_PROMPT
        ),
        "wave3_released": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--massive-data-root", required=True)
    parser.add_argument("--union-data-root", required=True)
    args = parser.parse_args()
    result = audit_protocol(
        args.protocol_root,
        args.massive_data_root,
        args.union_data_root,
    )
    print(
        "Audited prospective Wave-3 protocol: "
        f"{result['smoke_rows']} smoke, {result['confirmation_rows']} confirmation, "
        f"methods={','.join(result['method_ids'])}"
    )
    print("Wave 3 remains unreleased; this auditor cannot submit or allocate work.")


if __name__ == "__main__":
    main()
