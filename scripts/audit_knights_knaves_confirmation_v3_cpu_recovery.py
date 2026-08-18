#!/usr/bin/env python3
"""Fail-closed provenance for CPU-only recovery of failed K&K v3 job 237934."""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess

import audit_knights_knaves_confirmation_v2_workflow as v2_workflow
import evaluate_knights_knaves_confirmation_v3 as evaluator
import prepare_knights_knaves_confirmation_v3_data as v3_data
import summarize_knights_knaves_confirmation_v3 as summarizer


ORIGINAL_EVALUATION_COMMIT = "3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df"
FAILED_JOB_ID = "237934"
PREP_SHA256 = "7766044b9678391dd90c049b6ce45f9bc9b0cf6002de8ee827a1531040f557ad"
AUTH_SHA256 = "8c9d7795c5955d31ed030b4d8d59f9e935e80a2e4c1a6f2e2c2a631bbc8a4834"
JOBS_SHA256 = "95b7ddc74bc8de9b15fcef1abf72b1e9330ce172d8f91f8ed2b35f823fafed41"
STDOUT_SHA256 = "9512691fd81890f81002caac504b2e208df5ee63429f2ae34675b89f2c655fa3"
STDERR_SHA256 = "4bdeab3195941fe0f2e996b7c415e24a3e67fa6e8ef0d3aced0cb4eb661c3f70"

GENERATION_SHA256 = {
    "confirmation_v3_n4__pi_base.json": (
        "e4feca97905ce3723f9862475327848d4d8a343f4fdb4fb553be833edf7b62c6"
    ),
    "confirmation_v3_n4__step_192.json": (
        "37cd82b6287bdd2ae8fd2a647656e92712c3ef4d2b5550c23982775b618ab362"
    ),
    "confirmation_v3_n5__pi_base.json": (
        "a712028dd4ece890a50d23d9279f4ca7d77b4c40cde469dd2258887088e6a372"
    ),
    "confirmation_v3_n5__step_192.json": (
        "2b38e6ee067bfb5234b961dfe9d59623ec8aec8436614e72befd9d1b204c005c"
    ),
    "confirmation_v3_n6__pi_base.json": (
        "005bd7ac0967ee7f15ab721fcbb1af911b95c8841ed67210eeecbf0fe8a805d3"
    ),
    "confirmation_v3_n6__step_192.json": (
        "be8faedf72d06a96dab4f49a32636eb0694729ed3695fd8d38820b6ef30c7ad4"
    ),
}

# These files implement frozen scientific inputs or gates and must remain
# byte-for-byte identical to ORIGINAL_EVALUATION_COMMIT during recovery.
FROZEN_SOURCE_SHA256 = {
    "scripts/audit_knights_knaves_confirmation_v2_workflow.py": (
        "fed4eb213ea11ac087db30872d984604a86f51512553fb9806f8ddf1a8abbb14"
    ),
    "scripts/evaluate_knights_knaves_confirmation_v2.py": (
        "67d9640bc67991df9c9719cb15bb7b31ff90a99a5e037af44382b1456b9ed6c2"
    ),
    "scripts/prepare_knights_knaves_confirmation_v2_data.py": (
        "c21b5bb73d438264b167973916e6d479e73f29127aa4a03258955a74c447e8bd"
    ),
    "scripts/prepare_knights_knaves_confirmation_v3_data.py": (
        "881229face0e62010561da06bafa4146729c5ad863ed50872396d6a793a210f5"
    ),
    "scripts/prepare_knights_knaves_pilot_data.py": (
        "694dcf26e2adc1d616063abe734b1139fa012938d736d05e89d3867616e05b47"
    ),
    "scripts/preflight_knights_knaves_confirmation_v3.py": (
        "1546cbaa9745723bca9784135d772ab63f1d82aca216c6bc4823846db9b788c7"
    ),
    "scripts/sample_knights_knaves_generations.py": (
        "de02c3f17c3fef8bb8489e1991739de0c787f40b1df9208fe7f4c56b8066f9c9"
    ),
    "scripts/sample_knights_knaves_structured_choices.py": (
        "643bb7f410013b3a9e53a656ecb9e517575e75b3d748ab128b9546d21d1ad02e"
    ),
    "scripts/summarize_knights_knaves_confirmation_v2.py": (
        "d80dacab5d4561fe5568725a880407fc4e1294e23e191a98710b3832ced39a72"
    ),
    "scripts/summarize_knights_knaves_confirmation_v3.py": (
        "971135635908737794b3125d61a330085679b09e70464e320318f8e25f4e3284"
    ),
    "scripts/summarize_knights_knaves_pilot.py": (
        "5c3ef5063d14efc0d1f8275a3392c7144238212db589c537a0d5aae6f6b71760"
    ),
    "configs/training_qwen25_7b_kk_reasoning_pilot.yaml": (
        "5caef6baeb07f4ab4de8901001d7adb02433794e15c1024a950dc3bf59f492cb"
    ),
}

PATCHED_SOURCE_PATHS = {
    "integrity_loader": "scripts/evaluate_knights_knaves_generations.py",
    "v3_evaluator": "scripts/evaluate_knights_knaves_confirmation_v3.py",
    "recovery_auditor": (
        "scripts/audit_knights_knaves_confirmation_v3_cpu_recovery.py"
    ),
    "recovery_entrypoint": (
        "scripts/recover_knights_knaves_reasoning_confirmation_v3_tillicum_cpu.sh"
    ),
}

K_ONLY_ALLOWED_COMMIT_PATHS = set(PATCHED_SOURCE_PATHS.values()) | {
    "tests/test_knights_knaves_confirmation_v3.py",
}
K_ONLY_REQUIRED_COMMIT_PATHS = set(K_ONLY_ALLOWED_COMMIT_PATHS)
EXPECTED_REPAIR_STATUSES = {
    "scripts/evaluate_knights_knaves_generations.py": "M",
    "scripts/evaluate_knights_knaves_confirmation_v3.py": "M",
    "scripts/audit_knights_knaves_confirmation_v3_cpu_recovery.py": "A",
    "scripts/recover_knights_knaves_reasoning_confirmation_v3_tillicum_cpu.sh": "A",
    "tests/test_knights_knaves_confirmation_v3.py": "M",
}


def require_directory(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe directory: {path}")
    return path


def require_hash(path, expected, label):
    path = v2_workflow.require_file(path)
    observed = v2_workflow.sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash changed: {observed} != {expected}")
    return path


def load_sealed(path, expected_sha, label):
    path = require_hash(path, expected_sha, label)
    payload = v2_workflow.load_json(path)
    v2_workflow.verify_seal(payload, path)
    return payload


def git_repair_state(repo_root, repair_commit=None):
    repo_root = require_directory(repo_root)
    checkout_commit = v2_workflow.git_state(repo_root)
    if repair_commit is None:
        repair_commit = checkout_commit
    elif re.fullmatch(r"[0-9a-f]{40}", str(repair_commit)) is None:
        raise ValueError("Existing provenance has an invalid K&K repair commit")
    descendant = subprocess.run(
        [
            "git", "-C", repo_root, "merge-base", "--is-ancestor",
            repair_commit, checkout_commit,
        ],
        check=False,
    )
    if descendant.returncode != 0:
        raise ValueError("Pinned K&K repair commit is not in the checkout ancestry")
    parent_line = subprocess.check_output(
        [
            "git", "-C", repo_root, "rev-list", "--parents", "-n", "1",
            repair_commit,
        ],
        text=True,
    ).strip().split()
    if len(parent_line) != 2 or parent_line[0] != repair_commit:
        raise ValueError("K&K recovery commit must be a non-merge single commit")
    parent_commit = parent_line[1]
    ancestor = subprocess.run(
        [
            "git", "-C", repo_root, "merge-base", "--is-ancestor",
            ORIGINAL_EVALUATION_COMMIT, parent_commit,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Original K&K evaluation commit is not an ancestor")
    change_lines = list(filter(None, subprocess.check_output(
        [
            "git", "-C", repo_root, "diff", "--name-status", "--no-renames",
            parent_commit, repair_commit,
        ],
        text=True,
    ).splitlines()))
    changes = {}
    for line in change_lines:
        fields = line.split("\t", 1)
        if len(fields) != 2 or fields[1] in changes:
            raise ValueError("K&K repair commit has an invalid change record")
        status, path = fields
        changes[path] = status
    changed = set(changes)
    unexpected = changed - K_ONLY_ALLOWED_COMMIT_PATHS
    missing = K_ONLY_REQUIRED_COMMIT_PATHS - changed
    if unexpected or missing or changes != EXPECTED_REPAIR_STATUSES:
        raise ValueError(
            "K&K recovery commit is not scoped to the reviewed repair: "
            f"unexpected={sorted(unexpected)}, missing={sorted(missing)}, "
            f"changes={changes}"
        )
    return {
        "repair_commit": repair_commit,
        "repair_parent_commit": parent_commit,
        "repair_commit_paths": sorted(changed),
    }


def audit_frozen_sources(repo_root):
    inventory = {}
    for relative_path, expected_sha in sorted(FROZEN_SOURCE_SHA256.items()):
        path = require_hash(
            os.path.join(repo_root, relative_path), expected_sha, relative_path
        )
        inventory[relative_path] = v2_workflow.sha256_file(path)
    return inventory


def audit_patched_sources(repo_root):
    inventory = {}
    for name, relative_path in sorted(PATCHED_SOURCE_PATHS.items()):
        path = v2_workflow.require_file(os.path.join(repo_root, relative_path))
        inventory[name] = {
            "path": relative_path,
            "sha256": v2_workflow.sha256_file(path),
        }
    return inventory


def audit_checkpoint(checkpoint):
    checkpoint = require_directory(checkpoint)
    artifacts = {
        "adapter_model.safetensors": v2_workflow.CHECKPOINT_WEIGHT_SHA256,
        "adapter_config.json": v2_workflow.CHECKPOINT_CONFIG_SHA256,
        "trainer_state.json": v2_workflow.CHECKPOINT_TRAINER_STATE_SHA256,
    }
    for filename, expected_sha in artifacts.items():
        require_hash(os.path.join(checkpoint, filename), expected_sha, filename)
    fingerprint = evaluator.direct.adapter_fingerprint(checkpoint)
    if fingerprint != v2_workflow.CHECKPOINT_FINGERPRINT:
        raise ValueError("Frozen checkpoint composite fingerprint changed")
    return {
        "path": checkpoint,
        "step": v2_workflow.CHECKPOINT_STEP,
        "fingerprint": fingerprint,
        "artifacts": dict(sorted(artifacts.items())),
    }


def audit_prior_records(args):
    prep = load_sealed(args.prep_file, PREP_SHA256, "V3 preparation record")
    expected_prep = {
        "record_type": "kk_reasoning_confirmation_v3_preparation",
        "evaluation_commit": ORIGINAL_EVALUATION_COMMIT,
        "training_commit": v2_workflow.V1_TRAINING_COMMIT,
        "training_config_sha256": v2_workflow.TRAINING_CONFIG_SHA256,
        "base_model": v2_workflow.BASE_MODEL,
        "base_model_revision": v2_workflow.BASE_MODEL_REVISION,
        "checkpoint_step": v2_workflow.CHECKPOINT_STEP,
        "checkpoint_fingerprint": v2_workflow.CHECKPOINT_FINGERPRINT,
        "checkpoint_weight_sha256": v2_workflow.CHECKPOINT_WEIGHT_SHA256,
        "v3_sets": {
            name: dict(spec) for name, spec in sorted(v3_data.V3_SPECS.items())
        },
        "inference": {
            "temperature": 0.0,
            "n_samples": 1,
            "seed": evaluator.protocol.INFERENCE_SEED,
            "max_new_tokens": evaluator.protocol.MAX_NEW_TOKENS,
            "max_context": evaluator.protocol.MAX_CONTEXT,
            "symmetric_base_candidate": True,
        },
        "training_or_checkpoint_search": False,
        "gpu_allocation_minutes": 0,
    }
    for key, expected in expected_prep.items():
        if prep.get(key) != expected:
            raise ValueError(f"Original V3 preparation differs for {key}")

    auth = load_sealed(args.auth_file, AUTH_SHA256, "V3 authorization record")
    expected_auth = {
        "record_type": "kk_reasoning_confirmation_v3_authorization",
        "evaluation_commit": ORIGINAL_EVALUATION_COMMIT,
        "prep_file_sha256": PREP_SHA256,
        "v3_data_manifest_sha256": prep["v3_data_manifest_sha256"],
        "checkpoint_fingerprint": v2_workflow.CHECKPOINT_FINGERPRINT,
        "v3_max_h200_minutes": 30,
        "no_requeue": True,
        "slurm_array": False,
        "training_or_extra_adapters": False,
        "selective_regeneration": False,
        "automatic_medical_union_or_quorum": False,
    }
    for key, expected in expected_auth.items():
        if auth.get(key) != expected:
            raise ValueError(f"Original V3 authorization differs for {key}")

    jobs_path = require_hash(args.jobs_file, JOBS_SHA256, "V3 jobs record")
    with open(jobs_path, newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    if jobs != [{
        "stage": "evaluate_v3",
        "job_id": FAILED_JOB_ID,
        "max_minutes": "30",
        "released": "true",
    }]:
        raise ValueError("Original V3 jobs record no longer binds failed job 237934")

    stdout = require_hash(args.stdout_log, STDOUT_SHA256, "failed-job stdout")
    stderr = require_hash(args.stderr_log, STDERR_SHA256, "failed-job stderr")
    return {
        "prep": prep,
        "authorization": auth,
        "jobs_path": jobs_path,
        "stdout_path": stdout,
        "stderr_path": stderr,
    }


def audit_data(args, prep):
    v3_data_root = require_directory(os.path.join(args.v3_root, "data"))
    manifest = v3_data.audit_output(
        v3_data_root,
        os.path.join(args.v1_root, "data"),
        os.path.join(args.v2_root, "data"),
    )
    manifest_path = os.path.join(v3_data_root, v3_data.MANIFEST_NAME)
    if v2_workflow.sha256_file(manifest_path) != prep["v3_data_manifest_sha256"]:
        raise ValueError("V3 data-manifest file differs from sealed preparation")
    if manifest["manifest_payload_sha256"] != prep[
        "v3_data_manifest_payload_sha256"
    ]:
        raise ValueError("V3 data-manifest payload differs from sealed preparation")
    return {
        "root": v3_data_root,
        "manifest_path": manifest_path,
        "manifest_sha256": prep["v3_data_manifest_sha256"],
        "manifest_payload_sha256": prep["v3_data_manifest_payload_sha256"],
    }


def audit_generations(generations_dir, data_root):
    generations_dir = require_directory(generations_dir)
    observed_names = {
        name for name in os.listdir(generations_dir)
        if name.endswith(".json")
    }
    if observed_names != set(GENERATION_SHA256):
        raise ValueError(
            "Recovery requires exactly the six sealed V3 generation files: "
            f"found={sorted(observed_names)}"
        )
    inventory = {}
    pattern = re.compile(
        r"^(confirmation_v3_n[456])__(pi_base|step_192)\.json$"
    )
    for filename, expected_sha in sorted(GENERATION_SHA256.items()):
        match = pattern.fullmatch(filename)
        if match is None:
            raise AssertionError(f"Invalid sealed generation name: {filename}")
        set_name, model_name = match.groups()
        path = require_hash(
            os.path.join(generations_dir, filename), expected_sha, filename
        )
        answers_path = os.path.join(
            data_root, "sets", f"{set_name}_answers.json"
        )
        answer_meta, _ = evaluator.v1_eval.load_answers(answers_path)
        generation_meta, samples = evaluator.load_generation(path, answer_meta)
        if generation_meta.get("model_name") != model_name:
            raise ValueError(f"Generation model/name mismatch: {filename}")
        if len(samples) != v3_data.V3_SPECS[set_name]["rows"]:
            raise ValueError(f"Generation count changed: {filename}")
        inventory[filename] = {
            "sha256": expected_sha,
            "generation_fingerprint": generation_meta["generation_fingerprint"],
            "model_fingerprint": generation_meta["model_fingerprint"],
            "samples": len(samples),
        }
    return inventory


def stable_record(record):
    stable = dict(record)
    stable.pop(v2_workflow.SEAL_FIELD, None)
    stable.pop("created_at", None)
    return stable


def recovery_record(args, repair_commit=None):
    repair = git_repair_state(args.repo_root, repair_commit=repair_commit)
    prior = audit_prior_records(args)
    data = audit_data(args, prior["prep"])
    checkpoint = audit_checkpoint(args.checkpoint)
    generations = audit_generations(args.generations_dir, data["root"])
    frozen_sources = audit_frozen_sources(args.repo_root)
    patched_sources = audit_patched_sources(args.repo_root)
    return v2_workflow.sealed({
        "schema_version": 1,
        "record_type": "kk_reasoning_confirmation_v3_cpu_recovery",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "recovery_mode": "cpu_only_rescore_existing_generations",
        "original_evaluation_commit": ORIGINAL_EVALUATION_COMMIT,
        **repair,
        "failed_job": {
            "job_id": FAILED_JOB_ID,
            "state": "FAILED",
            "stdout_path": prior["stdout_path"],
            "stdout_sha256": STDOUT_SHA256,
            "stderr_path": prior["stderr_path"],
            "stderr_sha256": STDERR_SHA256,
        },
        "original_records": {
            "prep_path": os.path.abspath(args.prep_file),
            "prep_sha256": PREP_SHA256,
            "authorization_path": os.path.abspath(args.auth_file),
            "authorization_sha256": AUTH_SHA256,
            "jobs_path": prior["jobs_path"],
            "jobs_sha256": JOBS_SHA256,
        },
        "data": data,
        "checkpoint": checkpoint,
        "inference": dict(prior["prep"]["inference"]),
        "generations": generations,
        "patched_sources": patched_sources,
        "patched_evaluator_sha256": patched_sources["v3_evaluator"]["sha256"],
        "patched_integrity_loader_sha256": patched_sources[
            "integrity_loader"
        ]["sha256"],
        "frozen_sources": frozen_sources,
        "frozen_gate_script_sha256": FROZEN_SOURCE_SHA256[
            "scripts/summarize_knights_knaves_confirmation_v3.py"
        ],
        "frozen_v2_final_summary_sha256": prior["prep"][
            "v2_final_summary_sha256"
        ],
        "scientific_contract": {
            "data_changed": False,
            "checkpoint_changed": False,
            "generations_changed_or_regenerated": False,
            "scoring_semantics_changed": False,
            "gate_code_changed": False,
            "v3_integrity_loader_bug_repaired": True,
            "training_allowed": False,
            "medical_unions_allowed": False,
            "quorum_allowed": False,
        },
        "resource_contract": {
            "slurm_submission": False,
            "gpu_allocation": False,
            "gpu_minutes_added": 0,
            "network_required": False,
        },
    })


def write_or_audit(args):
    existing = None
    if os.path.lexists(args.output_file):
        if not os.path.isfile(args.output_file) or os.path.islink(args.output_file):
            raise ValueError("Existing recovery provenance is not a safe regular file")
        existing = v2_workflow.load_json(args.output_file)
        v2_workflow.verify_seal(existing, args.output_file)
    expected = recovery_record(
        args,
        repair_commit=(None if existing is None else existing.get("repair_commit")),
    )
    if existing is not None:
        if stable_record(existing) != stable_record(expected):
            raise ValueError("Existing CPU-recovery provenance differs from audit")
        print("Audited existing K&K v3 CPU-recovery provenance")
        return existing
    v2_workflow.atomic_write_json(args.output_file, expected)
    print("Wrote K&K v3 CPU-recovery provenance")
    return expected


def audit_results(args):
    scores_dir = require_directory(args.scores_dir)
    expected_score_names = {
        f"{set_name}__{model_name}__direct.json"
        for set_name in v3_data.V3_SPECS
        for model_name in ("pi_base", "step_192")
    }
    observed_score_names = {
        name for name in os.listdir(scores_dir) if name.endswith(".json")
    }
    if observed_score_names != expected_score_names:
        raise ValueError(
            "CPU recovery requires exactly six V3 score files: "
            f"found={sorted(observed_score_names)}"
        )

    evaluator_sha256 = v2_workflow.sha256_file(evaluator.__file__)
    loader_sha256 = v2_workflow.sha256_file(evaluator.v1_eval.__file__)
    provenance_path = v2_workflow.require_file(args.provenance_file)
    provenance = v2_workflow.load_json(provenance_path)
    v2_workflow.verify_seal(provenance, provenance_path)
    expected_provenance = {
        "record_type": "kk_reasoning_confirmation_v3_cpu_recovery",
        "original_evaluation_commit": ORIGINAL_EVALUATION_COMMIT,
        "patched_evaluator_sha256": evaluator_sha256,
        "patched_integrity_loader_sha256": loader_sha256,
        "frozen_gate_script_sha256": FROZEN_SOURCE_SHA256[
            "scripts/summarize_knights_knaves_confirmation_v3.py"
        ],
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"CPU-recovery provenance differs for {field}")
    if {
        name: entry.get("sha256")
        for name, entry in provenance.get("generations", {}).items()
    } != GENERATION_SHA256:
        raise ValueError("CPU-recovery provenance generation hashes changed")
    if provenance.get("resource_contract") != {
        "slurm_submission": False,
        "gpu_allocation": False,
        "gpu_minutes_added": 0,
        "network_required": False,
    }:
        raise ValueError("CPU-recovery provenance resource contract changed")
    score_inventory = {}
    for set_name in sorted(v3_data.V3_SPECS):
        for model_name in ("pi_base", "step_192"):
            score_name = f"{set_name}__{model_name}__direct.json"
            score_path = os.path.join(scores_dir, score_name)
            result = summarizer.load_evaluation(score_path, set_name)
            meta = result["meta"]
            if meta.get("model_name") != model_name:
                raise ValueError(f"Recovered score model mismatch: {score_name}")
            expected_generation_sha = GENERATION_SHA256[
                f"{set_name}__{model_name}.json"
            ]
            expected_meta = {
                "generations_file_sha256": expected_generation_sha,
                "evaluator_script_sha256": evaluator_sha256,
                "integrity_loader_script_sha256": loader_sha256,
                "v2_evaluator_script_sha256": FROZEN_SOURCE_SHA256[
                    "scripts/evaluate_knights_knaves_confirmation_v2.py"
                ],
                "generator_script_sha256": FROZEN_SOURCE_SHA256[
                    "scripts/sample_knights_knaves_generations.py"
                ],
            }
            for field, expected in expected_meta.items():
                if meta.get(field) != expected:
                    raise ValueError(
                        f"Recovered score differs for {score_name}/{field}"
                    )
            score_inventory.setdefault(set_name, {})[model_name] = {
                "path": score_path,
                "sha256": v2_workflow.sha256_file(score_path),
            }

    summary_path = v2_workflow.require_file(args.summary_file)
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    summarizer.verify_decision(summary)
    decision = summary.get("gate", {}).get("decision")
    if decision not in {"GO", "STOP"}:
        raise ValueError("Recovered V3 summary has no terminal decision")
    expected_inputs = {
        set_name: {
            "direct_base_sha256": score_inventory[set_name]["pi_base"]["sha256"],
            "direct_candidate_sha256": score_inventory[set_name]["step_192"][
                "sha256"
            ],
        }
        for set_name in sorted(v3_data.V3_SPECS)
    }
    if summary.get("inputs") != expected_inputs:
        raise ValueError("Recovered V3 summary is stale relative to score files")
    expected_summary_meta = {
        "v3_data_manifest_sha256": v2_workflow.sha256_file(
            v2_workflow.require_file(args.v3_data_manifest)
        ),
        "v2_final_summary_sha256": summarizer.V2_FINAL_SUMMARY_SHA256,
    }
    for field, expected in expected_summary_meta.items():
        if summary.get("meta", {}).get(field) != expected:
            raise ValueError(f"Recovered V3 summary differs for {field}")
    require_hash(
        args.v2_final_summary,
        summarizer.V2_FINAL_SUMMARY_SHA256,
        "inherited V2 final summary",
    )

    sentinel_dir = require_directory(args.sentinel_dir)
    chosen = (
        "GO_KK_V3_BENEFIT_UNIONS" if decision == "GO"
        else "STOPPED_KK_V3_FINAL"
    )
    opposite = (
        "STOPPED_KK_V3_FINAL" if decision == "GO"
        else "GO_KK_V3_BENEFIT_UNIONS"
    )
    if os.path.lexists(os.path.join(sentinel_dir, opposite)):
        raise ValueError("Conflicting recovered V3 decision sentinel exists")
    sentinel_path = v2_workflow.require_file(
        os.path.join(sentinel_dir, chosen)
    )
    sentinel = v2_workflow.load_json(sentinel_path)
    expected_sentinel = {
        "decision": decision,
        "summary_file": os.path.abspath(summary_path),
        "summary_sha256": v2_workflow.sha256_file(summary_path),
    }
    if sentinel != expected_sentinel:
        raise ValueError("Recovered V3 decision sentinel is stale or inconsistent")
    v2_workflow.require_file(args.markdown_file)
    print(f"K&K v3 CPU-recovery result audit passed: {decision}")
    return decision


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    provenance = subparsers.add_parser("provenance")
    provenance.add_argument("--repo-root", required=True)
    provenance.add_argument("--v1-root", required=True)
    provenance.add_argument("--v2-root", required=True)
    provenance.add_argument("--v3-root", required=True)
    provenance.add_argument("--checkpoint", required=True)
    provenance.add_argument("--prep-file", required=True)
    provenance.add_argument("--auth-file", required=True)
    provenance.add_argument("--jobs-file", required=True)
    provenance.add_argument("--stdout-log", required=True)
    provenance.add_argument("--stderr-log", required=True)
    provenance.add_argument("--generations-dir", required=True)
    provenance.add_argument("--output-file", required=True)
    provenance.set_defaults(function=write_or_audit)

    results = subparsers.add_parser("results")
    results.add_argument("--provenance-file", required=True)
    results.add_argument("--scores-dir", required=True)
    results.add_argument("--summary-file", required=True)
    results.add_argument("--markdown-file", required=True)
    results.add_argument("--sentinel-dir", required=True)
    results.add_argument("--v3-data-manifest", required=True)
    results.add_argument("--v2-final-summary", required=True)
    results.set_defaults(function=audit_results)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
