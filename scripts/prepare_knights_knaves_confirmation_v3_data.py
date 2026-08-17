#!/usr/bin/env python3
"""Prepare immutable, logic-disjoint K&K v3 robustness sets.

V3 is an evaluation-only amendment.  It creates three new 300-example fresh
sets (N=4, N=5, and N=6) and excludes every abstract puzzle used by the v1
training/evaluation workflow or the v2 independent confirmation.  The labels
are stored only in answer files; model-facing prompt files remain label-free.
"""

import argparse
import datetime
import json
import os
import shutil
import tempfile

import prepare_knights_knaves_confirmation_v2_data as v2
import prepare_knights_knaves_pilot_data as v1


SCHEMA_VERSION = 1
MANIFEST_NAME = "confirmation_v3_manifest.json"
SEAL_FIELD = "manifest_payload_sha256"
V3_SPECS = {
    "confirmation_v3_n4": {"n_people": 4, "rows": 300, "seed": 2026081804},
    "confirmation_v3_n5": {"n_people": 5, "rows": 300, "seed": 2026081805},
    "confirmation_v3_n6": {"n_people": 6, "rows": 300, "seed": 2026081806},
}


def seal(payload):
    result = dict(payload)
    result.pop(SEAL_FIELD, None)
    result[SEAL_FIELD] = v1.sha256_bytes(v1.canonical_json_bytes(result))
    return result


def verify_seal(payload):
    unsealed = dict(payload)
    recorded = unsealed.pop(SEAL_FIELD, None)
    expected = v1.sha256_bytes(v1.canonical_json_bytes(unsealed))
    if recorded != expected:
        raise ValueError("K&K v3 data-manifest seal mismatch")


def require_regular_file(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe regular file: {path}")
    return path


def load_json(path):
    with open(require_regular_file(path), encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def expected_protocol():
    return {
        "name": "knights_knaves_reasoning_confirmation_v3",
        "status": "post_hoc_truncation_robustness_amendment",
        "frozen_checkpoint_step": 192,
        "checkpoint_selection_allowed": False,
        "training_allowed": False,
        "sets": {
            name: dict(spec) for name, spec in sorted(V3_SPECS.items())
        },
        "logic_disjoint_from_all_v1_and_v2_confirmation": True,
    }


def _v1_forbidden_hashes(v1_data_root, generator):
    manifest = v1.audit_existing_output(v1_data_root)
    forbidden = set()

    train = manifest["dataset"]["official_files"]["train_n5"]
    train_path = os.path.join(v1_data_root, "source", train["path"])
    if v1.sha256_file(train_path) != train["sha256"]:
        raise ValueError("V1 training source hash mismatch")
    for record in v1.load_jsonl(train_path):
        statements = v1.validate_puzzle_record(record, 5, generator)
        forbidden.add(v1.logic_sha256(statements))

    for set_name, entry in sorted(manifest["evaluation_sets"].items()):
        answer_path = os.path.join(v1_data_root, entry["answers"])
        if v1.sha256_file(answer_path) != entry["answers_sha256"]:
            raise ValueError(f"V1 answer hash mismatch: {set_name}")
        answers = load_json(answer_path).get("answers")
        if not isinstance(answers, list) or len(answers) != entry["rows"]:
            raise ValueError(f"Invalid V1 answer rows: {set_name}")
        for answer in answers:
            digest = answer.get("logic_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Invalid V1 logic hash: {set_name}")
            forbidden.add(digest)
    if len(forbidden) != 2500:
        raise ValueError(f"Expected 2,500 unique V1 logic hashes, found {len(forbidden)}")
    return forbidden, manifest


def _add_v2_confirmation_hashes(forbidden, v2_data_root, v1_data_root):
    manifest = v2.audit_output(v2_data_root, v1_data_root)
    entry = manifest["confirmation"]
    answer_path = os.path.join(v2_data_root, entry["answers"])
    answers = load_json(answer_path).get("answers")
    if not isinstance(answers, list) or len(answers) != v2.CONFIRMATION_ROWS:
        raise ValueError("Invalid V2 confirmation answers")
    added = 0
    for answer in answers:
        digest = answer.get("logic_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid V2 confirmation logic hash")
        if digest in forbidden:
            raise ValueError("V2 confirmation overlaps V1 despite its sealed manifest")
        forbidden.add(digest)
        added += 1
    if added != 300 or len(forbidden) != 2800:
        raise ValueError("Expected exactly 300 disjoint V2 confirmation hashes")
    return manifest


def build_output(staging, v1_data_root, v2_data_root):
    v1_data_root = os.path.abspath(v1_data_root)
    v2_data_root = os.path.abspath(v2_data_root)
    generator_path = os.path.join(v1_data_root, "source", "generator", "lib_kk.py")
    expected_generator_sha = v1.GENERATOR_FILES["lib_kk.py"]["sha256"]
    if v1.sha256_file(generator_path) != expected_generator_sha:
        raise ValueError("Pinned generator source hash mismatch")
    generator = v1.import_source_module(
        generator_path, "pinned_knights_knaves_confirmation_v3_generator"
    )

    forbidden, v1_manifest = _v1_forbidden_hashes(v1_data_root, generator)
    v2_manifest = _add_v2_confirmation_hashes(
        forbidden, v2_data_root, v1_data_root
    )
    parent_count = len(forbidden)

    set_manifest = {}
    all_v3_hashes = set()
    for set_name, spec in sorted(V3_SPECS.items()):
        records, attempts = v1.generate_fresh_records(
            generator, set_name, spec, forbidden
        )
        prompts, answers = v1.make_eval_artifacts(
            records, set_name, "fresh", "confirmation"
        )
        for payload in (prompts, answers):
            payload["meta"]["generation_seed"] = spec["seed"]
        prompt_path = os.path.join(staging, "sets", f"{set_name}_prompts.json")
        v1.atomic_write_json(prompt_path, prompts)
        prompt_sha = v1.sha256_file(prompt_path)
        answers["meta"]["prompt_file_sha256"] = prompt_sha
        answer_path = os.path.join(staging, "sets", f"{set_name}_answers.json")
        v1.atomic_write_json(answer_path, answers)
        names_path = os.path.join(staging, "names", f"{set_name}_names.json")
        v1.atomic_write_json(names_path, v2.names_only_payload(prompts, answers))

        hashes = {record["logic_sha256"] for record in records}
        if len(hashes) != spec["rows"] or hashes & all_v3_hashes:
            raise ValueError(f"V3 set is not internally disjoint: {set_name}")
        all_v3_hashes.update(hashes)
        set_manifest[set_name] = {
            **spec,
            "role": "confirmation",
            "source_kind": "fresh",
            "generation_attempts": attempts,
            "prompts": os.path.relpath(prompt_path, staging).replace(os.sep, "/"),
            "prompts_sha256": prompt_sha,
            "answers": os.path.relpath(answer_path, staging).replace(os.sep, "/"),
            "answers_sha256": v1.sha256_file(answer_path),
            "names": os.path.relpath(names_path, staging).replace(os.sep, "/"),
            "names_sha256": v1.sha256_file(names_path),
        }

    if len(all_v3_hashes) != 900 or len(forbidden) != parent_count + 900:
        raise ValueError("V3 logic-disjoint row accounting failed")

    v1_manifest_path = os.path.join(v1_data_root, v1.MANIFEST_NAME)
    v2_manifest_path = os.path.join(v2_data_root, v2.MANIFEST_NAME)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": expected_protocol(),
        "generator": {
            "repository": v1.GENERATOR_REPOSITORY,
            "revision": v1.GENERATOR_REVISION,
            "source_sha256": expected_generator_sha,
        },
        "parents": {
            "v1_data_root": v1_data_root,
            "v1_manifest_sha256": v1.sha256_file(v1_manifest_path),
            "v1_manifest_payload_sha256": v1_manifest["manifest_payload_sha256"],
            "v2_data_root": v2_data_root,
            "v2_manifest_sha256": v1.sha256_file(v2_manifest_path),
            "v2_manifest_payload_sha256": v2_manifest["manifest_payload_sha256"],
            "unique_logic_hashes_excluded": parent_count,
        },
        "sets": set_manifest,
    }
    manifest["files"] = v1.collect_file_inventory(staging)
    v1.atomic_write_json(os.path.join(staging, MANIFEST_NAME), seal(manifest))
    return manifest


def audit_output(output_dir, v1_data_root, v2_data_root):
    output_dir = os.path.abspath(output_dir)
    manifest_path = require_regular_file(os.path.join(output_dir, MANIFEST_NAME))
    manifest = load_json(manifest_path)
    verify_seal(manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected K&K v3 data schema")
    if manifest.get("protocol") != expected_protocol():
        raise ValueError("K&K v3 protocol differs from the frozen amendment")
    if os.path.abspath(manifest["parents"]["v1_data_root"]) != os.path.abspath(
        v1_data_root
    ):
        raise ValueError("K&K v3 manifest references another V1 data root")
    if os.path.abspath(manifest["parents"]["v2_data_root"]) != os.path.abspath(
        v2_data_root
    ):
        raise ValueError("K&K v3 manifest references another V2 data root")
    v1_manifest = v1.audit_existing_output(v1_data_root)
    v2_manifest = v2.audit_output(v2_data_root, v1_data_root)
    parent_checks = {
        "v1_manifest_sha256": v1.sha256_file(os.path.join(v1_data_root, v1.MANIFEST_NAME)),
        "v1_manifest_payload_sha256": v1_manifest["manifest_payload_sha256"],
        "v2_manifest_sha256": v1.sha256_file(os.path.join(v2_data_root, v2.MANIFEST_NAME)),
        "v2_manifest_payload_sha256": v2_manifest["manifest_payload_sha256"],
        "unique_logic_hashes_excluded": 2800,
    }
    for field, expected in parent_checks.items():
        if manifest["parents"].get(field) != expected:
            raise ValueError(f"K&K v3 parent binding changed for {field}")

    if set(manifest.get("sets", {})) != set(V3_SPECS):
        raise ValueError("K&K v3 manifest has unexpected sets")
    for set_name, spec in V3_SPECS.items():
        entry = manifest["sets"][set_name]
        for field, expected in {
            **spec, "role": "confirmation", "source_kind": "fresh"
        }.items():
            if entry.get(field) != expected:
                raise ValueError(f"K&K v3 set differs for {set_name}/{field}")
        for kind in ("prompts", "answers", "names"):
            path = os.path.join(output_dir, entry[kind])
            if v1.sha256_file(path) != entry[f"{kind}_sha256"]:
                raise ValueError(f"K&K v3 {set_name} {kind} hash mismatch")

    observed = v1.collect_file_inventory(output_dir)
    observed.pop(MANIFEST_NAME, None)
    if observed != manifest.get("files"):
        raise ValueError("K&K v3 data inventory differs from its manifest")
    return manifest


def prepare(output_dir, v1_data_root, v2_data_root):
    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        audit_output(output_dir, v1_data_root, v2_data_root)
        print(f"Audited immutable K&K v3 data at {output_dir}")
        return
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=os.path.basename(output_dir) + ".staging-", dir=parent)
    try:
        build_output(staging, v1_data_root, v2_data_root)
        os.rename(staging, output_dir)
        staging = None
        audit_output(output_dir, v1_data_root, v2_data_root)
        print(f"Prepared and audited K&K v3 data at {output_dir}")
    finally:
        if staging is not None and os.path.isdir(staging):
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_data_root", required=True)
    parser.add_argument("--v2_data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit_output(args.output_dir, args.v1_data_root, args.v2_data_root)
        print(f"Audited immutable K&K v3 data at {os.path.abspath(args.output_dir)}")
    else:
        prepare(args.output_dir, args.v1_data_root, args.v2_data_root)


if __name__ == "__main__":
    main()
