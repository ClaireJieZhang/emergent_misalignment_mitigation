#!/usr/bin/env python3
"""Seal and verify the capped K&K v3 evaluation-only workflow."""

import argparse
import csv
import datetime
import json
import os
import re

import audit_knights_knaves_confirmation_v2_workflow as v2_workflow
import prepare_knights_knaves_confirmation_v2_data as v2_data
import prepare_knights_knaves_confirmation_v3_data as v3_data
import summarize_knights_knaves_confirmation_v2 as v2_summary


V2_CONFIRMATION_SUMMARY_SHA256 = (
    "e8d1a56588e5a915d06a5d8e4d708f4b2f8eaa01d7f7c8674707f05b031c9dd0"
)
V2_FINAL_SUMMARY_SHA256 = (
    "389b3b522ab101a6da9276667c41e85a5d5019a8ae77c99f17260353eb197cb6"
)
V2_STOP_SENTINEL_SHA256 = (
    "ced666d96ddb60bdb3110e7e1960e168896887c8bbdbb2509e442269be671837"
)
V2_JOB_ID = "233852"
V3_MAX_MINUTES = 30
V1_V2_RELEASED_MAX_MINUTES = 180
CUMULATIVE_RELEASED_MAX_MINUTES = 210
IMMUTABLE_CUMULATIVE_CEILING_MINUTES = 240
MAX_COST_USD = "0.45"
H200_USD_PER_HOUR = "0.90"
SEAL_FIELD = v2_workflow.SEAL_FIELD


def _load_and_verify_v2_summary(path, expected_sha):
    if v2_workflow.sha256_file(v2_workflow.require_file(path)) != expected_sha:
        raise ValueError(f"V2 summary hash changed: {path}")
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    v2_summary.verify_decision(summary)
    return summary


def audit_v2(v2_root, v1_root):
    """Bind V3 to the immutable V2 STOP and its exact one-truncation cause."""
    v2_root = os.path.abspath(v2_root)
    v1_parent = v2_workflow.audit_v1(v1_root)
    data_root = os.path.join(v2_root, "data")
    data_manifest = v2_data.audit_output(data_root, v1_parent["data_root"])
    confirmation_path = os.path.join(
        v2_root, "evaluation", "confirmation", "summary.json"
    )
    final_path = os.path.join(
        v2_root, "evaluation", "sealed_final", "summary.json"
    )
    confirmation = _load_and_verify_v2_summary(
        confirmation_path, V2_CONFIRMATION_SUMMARY_SHA256
    )
    final = _load_and_verify_v2_summary(final_path, V2_FINAL_SUMMARY_SHA256)
    if confirmation.get("gate", {}).get("decision") != "GO":
        raise ValueError("V2 independent confirmation is no longer GO")
    checks = final.get("gate", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if not value)
    if final.get("gate", {}).get("decision") != "STOP" or failed != [
        "all_direct_zero_truncation"
    ]:
        raise ValueError("V2 no longer stops solely on zero direct truncation")

    total_direct_rows = 0
    truncations = []
    for set_name in v2_summary.FINAL_SETS:
        for model in ("pi_base", "step_192"):
            path = os.path.join(
                v2_root, "evaluation", "scores",
                f"{set_name}__{model}__direct.json",
            )
            field = "direct_base_sha256" if model == "pi_base" else "direct_candidate_sha256"
            if v2_workflow.sha256_file(path) != final["inputs"][set_name][field]:
                raise ValueError(f"V2 direct score hash changed: {set_name}/{model}")
            evaluation = v2_summary.load_evaluation(path, "direct")
            total_direct_rows += evaluation["metrics"]["n"]
            truncations.extend(
                (set_name, model, task["question_id"])
                for task in evaluation["tasks"]
                if task["stop_reason"] == "max_new_tokens"
            )
    expected_truncations = [("fresh_n6", "pi_base", "fresh_n6:113")]
    if total_direct_rows != 2400 or truncations != expected_truncations:
        raise ValueError(
            "V2 must contain exactly one direct truncation among 2,400 outputs: "
            f"expected {expected_truncations}, found {truncations} in {total_direct_rows}"
        )

    stop_path = os.path.join(v2_root, "control", "STOPPED_KK_V2_FINAL")
    if v2_workflow.sha256_file(v2_workflow.require_file(stop_path)) != V2_STOP_SENTINEL_SHA256:
        raise ValueError("V2 STOP sentinel hash changed")
    stop = v2_workflow.load_json(stop_path)
    expected_stop = {
        "decision": "STOP",
        "summary_file": final_path,
        "summary_sha256": V2_FINAL_SUMMARY_SHA256,
    }
    if stop != expected_stop:
        raise ValueError("V2 STOP sentinel no longer binds the final summary")
    if os.path.lexists(os.path.join(v2_root, "control", "GO_KK_V2_BENEFIT_UNIONS")):
        raise ValueError("V2 unexpectedly contains a benefit-union GO")

    jobs_path = v2_workflow.require_file(os.path.join(v2_root, "control", "jobs.tsv"))
    with open(jobs_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows != [{
        "stage": "evaluate_v2", "job_id": V2_JOB_ID,
        "max_minutes": "30", "released": "true",
    }]:
        raise ValueError("V2 allocation record changed")
    return {
        "root": v2_root,
        "data_root": data_root,
        "data_manifest_sha256": v2_workflow.sha256_file(
            os.path.join(data_root, v2_data.MANIFEST_NAME)
        ),
        "data_manifest_payload_sha256": data_manifest["manifest_payload_sha256"],
        "confirmation_summary_sha256": V2_CONFIRMATION_SUMMARY_SHA256,
        "final_summary": final_path,
        "final_summary_sha256": V2_FINAL_SUMMARY_SHA256,
        "stop_sha256": V2_STOP_SENTINEL_SHA256,
        "jobs_sha256": v2_workflow.sha256_file(jobs_path),
        "total_direct_outputs": total_direct_rows,
        "direct_truncations": 1,
        "truncated_condition": "fresh_n6/pi_base/fresh_n6:113",
    }


def prep_record(repo_root, v1_root, v2_root, v3_data_root):
    commit = v2_workflow.git_state(repo_root)
    training_config = v2_workflow.audit_training_config(repo_root)
    parent = audit_v2(v2_root, v1_root)
    manifest = v3_data.audit_output(
        v3_data_root, os.path.join(v1_root, "data"), parent["data_root"]
    )
    manifest_path = os.path.join(v3_data_root, v3_data.MANIFEST_NAME)
    return v2_workflow.sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_confirmation_v3_preparation",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "evaluation_commit": commit,
            "training_commit": v2_workflow.V1_TRAINING_COMMIT,
            "training_config": training_config["path"],
            "training_config_sha256": training_config["sha256"],
            "base_model": training_config["base_model"],
            "base_model_revision": training_config["base_model_revision"],
            "v1_root": os.path.abspath(v1_root),
            "v2_root": parent["root"],
            "v2_data_manifest_sha256": parent["data_manifest_sha256"],
            "v2_confirmation_summary_sha256": parent["confirmation_summary_sha256"],
            "v2_final_summary_sha256": parent["final_summary_sha256"],
            "v2_stop_sha256": parent["stop_sha256"],
            "v2_jobs_sha256": parent["jobs_sha256"],
            "v2_total_direct_outputs": parent["total_direct_outputs"],
            "v2_direct_truncations": parent["direct_truncations"],
            "v2_truncated_condition": parent["truncated_condition"],
            "checkpoint_step": v2_workflow.CHECKPOINT_STEP,
            "checkpoint_fingerprint": v2_workflow.CHECKPOINT_FINGERPRINT,
            "checkpoint_weight_sha256": v2_workflow.CHECKPOINT_WEIGHT_SHA256,
            "v3_data_root": os.path.abspath(v3_data_root),
            "v3_data_manifest_sha256": v2_workflow.sha256_file(manifest_path),
            "v3_data_manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "v3_sets": {
                name: dict(spec) for name, spec in sorted(v3_data.V3_SPECS.items())
            },
            "inference": {
                "temperature": 0.0,
                "n_samples": 1,
                "seed": 8172026,
                "max_new_tokens": 4096,
                "max_context": 8192,
                "symmetric_base_candidate": True,
            },
            "training_or_checkpoint_search": False,
            "gpu_allocation_minutes": 0,
        }
    )


def verify_prep(path, repo_root, v1_root, v2_root, v3_data_root):
    observed = v2_workflow.load_json(path)
    v2_workflow.verify_seal(observed, path)
    expected = prep_record(repo_root, v1_root, v2_root, v3_data_root)
    for key in set(expected) - {"created_at", SEAL_FIELD}:
        if observed.get(key) != expected.get(key):
            raise ValueError(f"V3 preparation record mismatch for {key}")
    return observed


def command_write_prep(args):
    record = prep_record(
        args.repo_root, args.v1_root, args.v2_root, args.v3_data_root
    )
    if os.path.lexists(args.output_file):
        verify_prep(
            args.output_file, args.repo_root, args.v1_root, args.v2_root,
            args.v3_data_root,
        )
        print("Audited existing K&K v3 preparation record")
    else:
        v2_workflow.atomic_write_json(args.output_file, record)
        verify_prep(
            args.output_file, args.repo_root, args.v1_root, args.v2_root,
            args.v3_data_root,
        )
        print("Wrote K&K v3 preparation record")


def command_verify_prep(args):
    verify_prep(
        args.prep_file, args.repo_root, args.v1_root, args.v2_root,
        args.v3_data_root,
    )
    print("K&K v3 preparation audit passed")


def command_write_authorization(args):
    if args.ack_max_cost_usd != MAX_COST_USD:
        raise ValueError(f"Exact V3 cost acknowledgement must be {MAX_COST_USD}")
    prep = verify_prep(
        args.prep_file, args.repo_root, args.v1_root, args.v2_root,
        args.v3_data_root,
    )
    if os.path.lexists(args.output_file):
        raise ValueError("K&K v3 authorization already exists")
    record = v2_workflow.sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_confirmation_v3_authorization",
            "authorized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "evaluation_commit": prep["evaluation_commit"],
            "training_commit": prep["training_commit"],
            "training_config_sha256": prep["training_config_sha256"],
            "base_model": prep["base_model"],
            "base_model_revision": prep["base_model_revision"],
            "prep_file_sha256": v2_workflow.sha256_file(args.prep_file),
            "v3_data_manifest_sha256": prep["v3_data_manifest_sha256"],
            "v2_final_summary_sha256": V2_FINAL_SUMMARY_SHA256,
            "v2_stop_preserved": True,
            "checkpoint_fingerprint": v2_workflow.CHECKPOINT_FINGERPRINT,
            "h200_usd_per_hour": H200_USD_PER_HOUR,
            "v3_max_cost_usd": MAX_COST_USD,
            "v3_max_h200_minutes": V3_MAX_MINUTES,
            "v1_v2_released_max_h200_minutes": V1_V2_RELEASED_MAX_MINUTES,
            "cumulative_released_max_h200_minutes": CUMULATIVE_RELEASED_MAX_MINUTES,
            "immutable_cumulative_ceiling_h200_minutes": IMMUTABLE_CUMULATIVE_CEILING_MINUTES,
            "remaining_unsubmitted_reserve_h200_minutes": 30,
            "no_requeue": True,
            "slurm_array": False,
            "training_or_extra_adapters": False,
            "selective_regeneration": False,
            "automatic_medical_union_or_quorum": False,
        }
    )
    v2_workflow.atomic_write_json(args.output_file, record)
    print("Wrote K&K v3 evaluation-only authorization")


def verify_authorization(auth_file, prep_file, repo_root, v1_root, v2_root, v3_data_root):
    auth = v2_workflow.load_json(auth_file)
    v2_workflow.verify_seal(auth, auth_file)
    prep = verify_prep(prep_file, repo_root, v1_root, v2_root, v3_data_root)
    expected = {
        "record_type": "kk_reasoning_confirmation_v3_authorization",
        "evaluation_commit": prep["evaluation_commit"],
        "training_commit": prep["training_commit"],
        "training_config_sha256": v2_workflow.TRAINING_CONFIG_SHA256,
        "base_model": v2_workflow.BASE_MODEL,
        "base_model_revision": v2_workflow.BASE_MODEL_REVISION,
        "prep_file_sha256": v2_workflow.sha256_file(prep_file),
        "v3_data_manifest_sha256": prep["v3_data_manifest_sha256"],
        "v2_final_summary_sha256": V2_FINAL_SUMMARY_SHA256,
        "v2_stop_preserved": True,
        "checkpoint_fingerprint": v2_workflow.CHECKPOINT_FINGERPRINT,
        "h200_usd_per_hour": H200_USD_PER_HOUR,
        "v3_max_cost_usd": MAX_COST_USD,
        "v3_max_h200_minutes": V3_MAX_MINUTES,
        "v1_v2_released_max_h200_minutes": V1_V2_RELEASED_MAX_MINUTES,
        "cumulative_released_max_h200_minutes": CUMULATIVE_RELEASED_MAX_MINUTES,
        "immutable_cumulative_ceiling_h200_minutes": IMMUTABLE_CUMULATIVE_CEILING_MINUTES,
        "remaining_unsubmitted_reserve_h200_minutes": 30,
        "no_requeue": True,
        "slurm_array": False,
        "training_or_extra_adapters": False,
        "selective_regeneration": False,
        "automatic_medical_union_or_quorum": False,
    }
    for key, value in expected.items():
        if auth.get(key) != value:
            raise ValueError(f"V3 authorization mismatch for {key}")
    return auth


def read_job(path):
    with open(v2_workflow.require_file(path), newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or rows[0].get("stage") != "evaluate_v3":
        raise ValueError("V3 jobs.tsv must contain exactly evaluate_v3")
    row = rows[0]
    if (
        re.fullmatch(r"[0-9]+", str(row.get("job_id"))) is None
        or row.get("max_minutes") != str(V3_MAX_MINUTES)
        or row.get("released") != "true"
    ):
        raise ValueError("V3 jobs.tsv contains invalid allocation fields")
    return row


def command_verify_job(args):
    verify_authorization(
        args.auth_file, args.prep_file, args.repo_root, args.v1_root,
        args.v2_root, args.v3_data_root,
    )
    job = read_job(args.jobs_file)
    if job["job_id"] != str(args.job_id):
        raise ValueError("Running V3 job ID differs from jobs.tsv")
    if v2_workflow.parse_time_limit(args.time_limit) != V3_MAX_MINUTES:
        raise ValueError("Running V3 job exceeds the 30-minute cap")
    print(f"K&K v3 job {args.job_id} authorization audit passed")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser):
        subparser.add_argument("--repo-root", required=True)
        subparser.add_argument("--v1-root", required=True)
        subparser.add_argument("--v2-root", required=True)
        subparser.add_argument("--v3-data-root", required=True)

    write_prep = subparsers.add_parser("write-prep")
    common(write_prep)
    write_prep.add_argument("--output-file", required=True)
    write_prep.set_defaults(function=command_write_prep)
    verify_prep_parser = subparsers.add_parser("verify-prep")
    common(verify_prep_parser)
    verify_prep_parser.add_argument("--prep-file", required=True)
    verify_prep_parser.set_defaults(function=command_verify_prep)
    auth = subparsers.add_parser("write-authorization")
    common(auth)
    auth.add_argument("--prep-file", required=True)
    auth.add_argument("--ack-max-cost-usd", required=True)
    auth.add_argument("--output-file", required=True)
    auth.set_defaults(function=command_write_authorization)
    job = subparsers.add_parser("verify-job")
    common(job)
    job.add_argument("--prep-file", required=True)
    job.add_argument("--auth-file", required=True)
    job.add_argument("--jobs-file", required=True)
    job.add_argument("--job-id", required=True)
    job.add_argument("--time-limit", required=True)
    job.set_defaults(function=command_verify_job)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
