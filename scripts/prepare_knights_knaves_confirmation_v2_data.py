#!/usr/bin/env python3
"""Prepare the immutable K&K v2 confirmation set and names-only controls.

This is a post-hoc *evaluation amendment*, not a replacement for the v1
protocol.  It creates one fresh N=5 confirmation set whose abstract logic is
disjoint from v1 training, development, and all six v1 final sets.  It also
exports names-only companions for structured-choice decoding; no solution
labels are present in those model-facing files.
"""

import argparse
import datetime
import json
import os
import shutil
import tempfile

import prepare_knights_knaves_pilot_data as v1


SCHEMA_VERSION = 1
CONFIRMATION_SET = "confirmation_n5"
CONFIRMATION_SEED = 2026081705
CONFIRMATION_ROWS = 300
FINAL_SETS = (
    "official_n4", "official_n5", "official_n6",
    "fresh_n4", "fresh_n5", "fresh_n6",
)
MANIFEST_NAME = "confirmation_v2_manifest.json"
SEAL_FIELD = "manifest_payload_sha256"


def seal(payload):
    result = dict(payload)
    result.pop(SEAL_FIELD, None)
    result[SEAL_FIELD] = v1.sha256_bytes(v1.canonical_json_bytes(result))
    return result


def verify_seal(payload):
    unsealed = dict(payload)
    recorded = unsealed.pop(SEAL_FIELD, None)
    if recorded != v1.sha256_bytes(v1.canonical_json_bytes(unsealed)):
        raise ValueError("K&K v2 data-manifest seal mismatch")


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


def names_only_payload(prompt_payload, answer_payload):
    prompts = prompt_payload.get("prompts")
    answers = answer_payload.get("answers")
    if not isinstance(prompts, list) or not isinstance(answers, list):
        raise ValueError("Prompt/answer payloads must contain lists")
    if len(prompts) != len(answers) or not prompts:
        raise ValueError("Prompt/answer payloads have unequal or empty rows")
    records = []
    for prompt, answer in zip(prompts, answers):
        for key in ("question_id", "prompt_sha256", "set_name", "n_people"):
            if prompt.get(key) != answer.get(key):
                raise ValueError(f"Prompt/answer mismatch for {key}")
        names = answer.get("names")
        if (
            not isinstance(names, list)
            or len(names) != prompt.get("n_people")
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError(f"Invalid names for {prompt.get('question_id')}")
        records.append(
            {
                "question_id": prompt["question_id"],
                "prompt_sha256": prompt["prompt_sha256"],
                "set_name": prompt["set_name"],
                "n_people": prompt["n_people"],
                "names": names,
            }
        )
    meta = dict(prompt_payload["meta"])
    meta.update(
        {
            "contains_labels": False,
            "contains_names": True,
            "purpose": "canonical_assignment_structured_choice",
        }
    )
    return {"meta": meta, "records": records}


def load_v1_pair(data_root, registry_entry):
    prompt_path = os.path.join(data_root, registry_entry["prompts"])
    answer_path = os.path.join(data_root, registry_entry["answers"])
    if v1.sha256_file(prompt_path) != registry_entry["prompts_sha256"]:
        raise ValueError(f"V1 prompt hash mismatch: {prompt_path}")
    if v1.sha256_file(answer_path) != registry_entry["answers_sha256"]:
        raise ValueError(f"V1 answer hash mismatch: {answer_path}")
    return load_json(prompt_path), load_json(answer_path), prompt_path, answer_path


def build_output(staging, v1_data_root):
    v1_data_root = os.path.abspath(v1_data_root)
    v1_manifest = v1.audit_existing_output(v1_data_root)
    v1_manifest_path = os.path.join(v1_data_root, v1.MANIFEST_NAME)
    if v1_manifest.get("generator", {}).get("revision") != v1.GENERATOR_REVISION:
        raise ValueError("Unexpected v1 generator revision")

    generator_path = os.path.join(v1_data_root, "source", "generator", "lib_kk.py")
    expected_generator_sha = v1.GENERATOR_FILES["lib_kk.py"]["sha256"]
    if v1.sha256_file(generator_path) != expected_generator_sha:
        raise ValueError("Pinned v1 generator source hash mismatch")
    generator = v1.import_source_module(
        generator_path, "pinned_knights_knaves_confirmation_v2_generator"
    )

    forbidden = set()
    train_spec = v1_manifest["dataset"]["official_files"]["train_n5"]
    train_path = os.path.join(v1_data_root, "source", train_spec["path"])
    if v1.sha256_file(train_path) != train_spec["sha256"]:
        raise ValueError("V1 training source hash mismatch")
    for record in v1.load_jsonl(train_path):
        statements = v1.validate_puzzle_record(record, 5, generator)
        forbidden.add(v1.logic_sha256(statements))

    parent_inputs = {}
    for set_name, entry in sorted(v1_manifest["evaluation_sets"].items()):
        prompts, answers, prompt_path, answer_path = load_v1_pair(
            v1_data_root, entry
        )
        for answer in answers["answers"]:
            logic_hash = answer.get("logic_sha256")
            if not isinstance(logic_hash, str) or len(logic_hash) != 64:
                raise ValueError(f"Invalid v1 logic hash in {set_name}")
            forbidden.add(logic_hash)
        parent_inputs[set_name] = {
            "role": entry["role"],
            "source_kind": entry["source_kind"],
            "n_people": entry["n_people"],
            "rows": entry["rows"],
            "prompts": os.path.abspath(prompt_path),
            "prompts_sha256": entry["prompts_sha256"],
            "answers": os.path.abspath(answer_path),
            "answers_sha256": entry["answers_sha256"],
        }
        if set_name in FINAL_SETS:
            names_payload = names_only_payload(prompts, answers)
            names_path = os.path.join(staging, "names", f"{set_name}_names.json")
            v1.atomic_write_json(names_path, names_payload)
            parent_inputs[set_name]["names"] = os.path.relpath(
                names_path, staging
            ).replace(os.sep, "/")
            parent_inputs[set_name]["names_sha256"] = v1.sha256_file(names_path)

    expected_forbidden = 1000 + 300 + 3 * 100 + 3 * 300
    if len(forbidden) != expected_forbidden:
        raise ValueError(
            f"Expected {expected_forbidden} unique v1 logic hashes, found "
            f"{len(forbidden)}"
        )

    spec = {
        "n_people": 5,
        "rows": CONFIRMATION_ROWS,
        "seed": CONFIRMATION_SEED,
        "role": "confirmation",
    }
    records, attempts = v1.generate_fresh_records(
        generator, CONFIRMATION_SET, spec, forbidden
    )
    prompts, answers = v1.make_eval_artifacts(
        records, CONFIRMATION_SET, "fresh", "confirmation"
    )
    for payload in (prompts, answers):
        payload["meta"]["generation_seed"] = CONFIRMATION_SEED
    prompt_path = os.path.join(staging, "confirmation", "confirmation_n5_prompts.json")
    v1.atomic_write_json(prompt_path, prompts)
    prompt_sha = v1.sha256_file(prompt_path)
    answers["meta"]["prompt_file_sha256"] = prompt_sha
    answer_path = os.path.join(staging, "confirmation", "confirmation_n5_answers.json")
    v1.atomic_write_json(answer_path, answers)
    names_path = os.path.join(staging, "names", "confirmation_n5_names.json")
    v1.atomic_write_json(names_path, names_only_payload(prompts, answers))

    confirmation_hashes = {record["logic_sha256"] for record in records}
    if len(confirmation_hashes) != CONFIRMATION_ROWS:
        raise ValueError("Confirmation logic hashes are not unique")
    if confirmation_hashes & (forbidden - confirmation_hashes):
        raise ValueError("Confirmation set overlaps a v1 split")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": {
            "name": "knights_knaves_reasoning_confirmation_v2",
            "status": "post_hoc_evaluation_amendment",
            "frozen_checkpoint_step": 192,
            "confirmation_set": CONFIRMATION_SET,
            "confirmation_rows": CONFIRMATION_ROWS,
            "confirmation_seed": CONFIRMATION_SEED,
            "confirmation_can_select_checkpoint": False,
            "one_seed_only": True,
            "final_sets": list(FINAL_SETS),
        },
        "parent_v1": {
            "data_root": v1_data_root,
            "manifest": v1_manifest_path,
            "manifest_sha256": v1.sha256_file(v1_manifest_path),
            "manifest_payload_sha256": v1_manifest["manifest_payload_sha256"],
            "generator_revision": v1.GENERATOR_REVISION,
            "generator_source_sha256": expected_generator_sha,
            "unique_logic_hashes_excluded": expected_forbidden,
            "inputs": parent_inputs,
        },
        "confirmation": {
            "n_people": 5,
            "rows": CONFIRMATION_ROWS,
            "generation_seed": CONFIRMATION_SEED,
            "generation_attempts": attempts,
            "prompts": os.path.relpath(prompt_path, staging).replace(os.sep, "/"),
            "prompts_sha256": prompt_sha,
            "answers": os.path.relpath(answer_path, staging).replace(os.sep, "/"),
            "answers_sha256": v1.sha256_file(answer_path),
            "names": os.path.relpath(names_path, staging).replace(os.sep, "/"),
            "names_sha256": v1.sha256_file(names_path),
            "logic_overlap_with_all_v1_splits": 0,
        },
    }
    manifest["files"] = v1.collect_file_inventory(staging)
    v1.atomic_write_json(os.path.join(staging, MANIFEST_NAME), seal(manifest))
    return manifest


def audit_output(output_dir, v1_data_root):
    output_dir = os.path.abspath(output_dir)
    manifest_path = require_regular_file(os.path.join(output_dir, MANIFEST_NAME))
    manifest = load_json(manifest_path)
    verify_seal(manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected K&K v2 data schema")
    protocol = manifest.get("protocol", {})
    expected_protocol = {
        "name": "knights_knaves_reasoning_confirmation_v2",
        "status": "post_hoc_evaluation_amendment",
        "frozen_checkpoint_step": 192,
        "confirmation_set": CONFIRMATION_SET,
        "confirmation_rows": CONFIRMATION_ROWS,
        "confirmation_seed": CONFIRMATION_SEED,
        "confirmation_can_select_checkpoint": False,
        "one_seed_only": True,
        "final_sets": list(FINAL_SETS),
    }
    if protocol != expected_protocol:
        raise ValueError("K&K v2 protocol fields differ from the frozen amendment")
    if os.path.abspath(manifest["parent_v1"]["data_root"]) != os.path.abspath(
        v1_data_root
    ):
        raise ValueError("K&K v2 manifest references another v1 data root")
    v1_manifest = v1.audit_existing_output(v1_data_root)
    parent_manifest_path = os.path.join(v1_data_root, v1.MANIFEST_NAME)
    if manifest["parent_v1"]["manifest_sha256"] != v1.sha256_file(
        parent_manifest_path
    ):
        raise ValueError("Parent v1 data manifest changed")
    if manifest["parent_v1"]["manifest_payload_sha256"] != v1_manifest[
        "manifest_payload_sha256"
    ]:
        raise ValueError("Parent v1 manifest payload changed")
    for entry in manifest["parent_v1"]["inputs"].values():
        for key in ("prompts", "answers"):
            if v1.sha256_file(entry[key]) != entry[f"{key}_sha256"]:
                raise ValueError(f"Parent v1 {key} changed")
    observed = v1.collect_file_inventory(output_dir)
    observed.pop(MANIFEST_NAME, None)
    if observed != manifest.get("files"):
        raise ValueError("K&K v2 data inventory differs from its manifest")
    confirmation = manifest["confirmation"]
    for key in ("prompts", "answers", "names"):
        path = os.path.join(output_dir, confirmation[key])
        if v1.sha256_file(path) != confirmation[f"{key}_sha256"]:
            raise ValueError(f"Confirmation {key} hash mismatch")
    return manifest


def prepare(output_dir, v1_data_root):
    output_dir = os.path.abspath(output_dir)
    if os.path.lexists(output_dir):
        audit_output(output_dir, v1_data_root)
        print(f"Audited immutable K&K v2 data at {output_dir}")
        return
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=os.path.basename(output_dir) + ".staging-", dir=parent)
    try:
        build_output(staging, v1_data_root)
        os.rename(staging, output_dir)
        staging = None
        audit_output(output_dir, v1_data_root)
        print(f"Prepared and audited K&K v2 data at {output_dir}")
    finally:
        if staging is not None and os.path.isdir(staging):
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit_output(args.output_dir, args.v1_data_root)
        print(f"Audited immutable K&K v2 data at {os.path.abspath(args.output_dir)}")
    else:
        prepare(args.output_dir, args.v1_data_root)


if __name__ == "__main__":
    main()
