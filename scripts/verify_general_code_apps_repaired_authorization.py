#!/usr/bin/env python3
"""Fail-closed authorization check for original and repaired-pilot Slurm jobs."""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import subprocess


TILLICUM_ROOT = pathlib.Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/general_code_apps_repaired_pilot_v1"
CONTROL_ROOT = OUTPUT_ROOT / "control"
REPO_ROOT = TILLICUM_ROOT / "projects/subliminal-mitigate"
RESUME_ROOT = CONTROL_ROOT / "resume_227440"

ORIGINAL_COMMIT = "a57dbf43fdf296dfdd31f14447e9a47e76db0405"
ORIGINAL_HASHES = {
    CONTROL_ROOT / "AUTHORIZED_MAX_COST_USD_1.80":
        "aaeaa4c9a19732339845d6124fbbdfe054dba71f1d3e96a36d20b03e711b61b6",
    CONTROL_ROOT / "jobs.tsv":
        "2651789b4a3816e9c80f2deb4f2add2ff3249d1e2efa1978bba128457dfa7565",
    CONTROL_ROOT / "SUBMITTED":
        "f8fbbc7b995ffd1742fa3e75bd665dffddfecb0cb7229fd44cf6bd3725adfb96",
    CONTROL_ROOT / "SUBMISSION_LOCK/owner":
        "90e14986d8cf5f97525be340b2f47d85fb9715286a58314720921de7ba126f82",
}
PREPARED_HASHES = {
    OUTPUT_ROOT / "data/data_manifest.json":
        "41db818be86dc46c930bbac83a9f5e5d90a9ce476de1aada310b3197e94394f2",
    OUTPUT_ROOT / "data/apps_repaired_candidates_evaluator.jsonl":
        "0143474c4156902450a2d61081a579c8a7f5a50b47c952ab278d74fecd1fe09c",
    OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json":
        "cfb5be8a9f4b69b211bb09cfda12c3ddc8d74b987411af8e37d0c8896927bee6",
    OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json":
        "f0bde28741a1fc00c611fe724db6ae87adbe6c24608dbf79ba5001ee033732a2",
    OUTPUT_ROOT / "data/apps_repaired_candidate_prompts.json":
        "689da81c118af0c9d12bbbd5b69edc6a64972d25914d29da9f64e6c4f2a57dc2",
}
FAILED_LOG_HASHES = {
    TILLICUM_ROOT / "outputs/logs/general_code_apps_repaired_prepare_227440.out":
        "ddab648c0a223fc5afa8e5b06198991185c1a0e4bc9bce897766bcb76383076b",
    TILLICUM_ROOT / "outputs/logs/general_code_apps_repaired_prepare_227440.err":
        "f4b7613b5c4e1ad4fabd2c5635e9bd721e68b33b66ca814e99b5cd193e0fdba3",
}
ORIGINAL_LIMITS = {"prepare": "00:30:00", "train": "00:30:00", "evaluate": "01:00:00"}
RESUME_LIMITS = {"prepare": "00:29:00", "train": "00:30:00", "evaluate": "01:00:00"}
RESUME_MINUTES = {"prepare": "29", "train": "30", "evaluate": "60"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hashes(values):
    for path, expected in values.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe sealed artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Sealed artifact hash mismatch: {path}: {actual}")


def prepared_hashes_for_stage(stage):
    if stage == "prepare":
        return PREPARED_HASHES
    return {
        path: digest
        for path, digest in PREPARED_HASHES.items()
        if path.name != "data_manifest.json"
    }


def read_unique_kv(path):
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or "=" not in line:
            raise ValueError(f"Invalid key/value line {line_number}: {path}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"Missing/duplicate key at line {line_number}: {path}")
        result[key] = value
    return result


def verify_self_seal(path, seal_key="addendum_sha256"):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    prefix = f"{seal_key}=".encode("ascii")
    if not lines or not lines[-1].startswith(prefix):
        raise ValueError(f"{seal_key} must be the final line of {path}")
    expected = lines[-1].decode("ascii").strip().split("=", 1)[1]
    actual = hashlib.sha256(b"".join(lines[:-1])).hexdigest()
    if expected != actual:
        raise ValueError(f"Integrity seal mismatch for {path}")
    return expected


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def verify_resume(stage, time_limit, job_id):
    addendum_path = RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
    jobs_path = RESUME_ROOT / "jobs.tsv"
    resumed_path = RESUME_ROOT / "RESUMED"
    for path in (addendum_path, jobs_path, resumed_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe resume control artifact: {path}")

    addendum_seal = verify_self_seal(addendum_path)
    addendum = read_unique_kv(addendum_path)
    fixed = {
        "within_original_authorization": "true",
        "original_auth_sha256": ORIGINAL_HASHES[
            CONTROL_ROOT / "AUTHORIZED_MAX_COST_USD_1.80"
        ],
        "original_jobs_sha256": ORIGINAL_HASHES[CONTROL_ROOT / "jobs.tsv"],
        "original_submitted_sha256": ORIGINAL_HASHES[CONTROL_ROOT / "SUBMITTED"],
        "original_submission_lock_owner_sha256": ORIGINAL_HASHES[
            CONTROL_ROOT / "SUBMISSION_LOCK/owner"
        ],
        "original_repo_commit": ORIGINAL_COMMIT,
        "original_prepare_job_id": "227440",
        "original_prepare_state": "FAILED",
        "original_prepare_elapsed_seconds": "60",
        "original_prepare_timelimit_minutes": "30",
        "original_train_job_id": "227441",
        "original_train_state": "CANCELLED",
        "original_train_elapsed_seconds": "0",
        "original_evaluate_job_id": "227442",
        "original_evaluate_state": "CANCELLED",
        "original_evaluate_elapsed_seconds": "0",
        "prior_rounded_h200_minutes": "1",
        "resume_prepare_minutes": "29",
        "resume_train_minutes": "30",
        "resume_evaluate_minutes": "60",
        "remaining_h200_minutes": "119",
        "cumulative_max_h200_minutes": "120",
        "cumulative_max_cost_usd": "1.80",
        "no_requeue": "true",
        "automatic_continuation": "false",
        "reason": "unicode_jsonl_u2028_record_separator_parser_repair",
        "prepared_manifest_sha256": PREPARED_HASHES[OUTPUT_ROOT / "data/data_manifest.json"],
        "failed_stdout_sha256": FAILED_LOG_HASHES[
            TILLICUM_ROOT / "outputs/logs/general_code_apps_repaired_prepare_227440.out"
        ],
        "failed_stderr_sha256": FAILED_LOG_HASHES[
            TILLICUM_ROOT / "outputs/logs/general_code_apps_repaired_prepare_227440.err"
        ],
    }
    for key, expected in fixed.items():
        if addendum.get(key) != expected:
            raise ValueError(f"Resume addendum mismatch for {key}")

    repair_commit = addendum.get("repair_repo_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", repair_commit):
        raise ValueError("Invalid repair_repo_commit")
    if git("rev-parse", "HEAD") != repair_commit:
        raise ValueError("Checkout does not match repair_repo_commit")
    if git("rev-parse", "HEAD^") != ORIGINAL_COMMIT:
        raise ValueError("Repair commit is not a direct child of the authorized commit")
    parents = git("rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2:
        raise ValueError("Repair commit must have exactly one parent")
    expected_diff = addendum.get("repair_diff_sha256", "")
    actual_diff = hashlib.sha256(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "diff", "--binary", ORIGINAL_COMMIT, repair_commit]
        )
    ).hexdigest()
    if expected_diff != actual_diff:
        raise ValueError("Repair diff hash mismatch")
    if time_limit != RESUME_LIMITS[stage]:
        raise ValueError(f"Unsafe resume TimeLimit for {stage}: {time_limit}")

    with open(jobs_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 3 or {row["stage"] for row in rows} != set(RESUME_MINUTES):
        raise ValueError("Resume job record must contain exactly three stages")
    by_stage = {row["stage"]: row for row in rows}
    for name, minutes in RESUME_MINUTES.items():
        if by_stage[name].get("max_minutes") != minutes:
            raise ValueError(f"Resume job cap mismatch for {name}")
        if not re.fullmatch(r"[0-9]+", by_stage[name].get("job_id", "")):
            raise ValueError(f"Invalid resume job ID for {name}")
    if by_stage[stage]["job_id"] != str(job_id):
        raise ValueError(f"Current job ID is not the recorded {stage} resume job")

    if stage != "prepare":
        prep_complete_path = OUTPUT_ROOT / "PREP_COMPLETE"
        finalized_manifest_path = OUTPUT_ROOT / "data/data_manifest.json"
        if not prep_complete_path.is_file() or prep_complete_path.is_symlink():
            raise ValueError("Missing or unsafe PREP_COMPLETE for downstream stage")
        prep_complete = read_unique_kv(prep_complete_path)
        if prep_complete.get("job_id") != by_stage["prepare"]["job_id"]:
            raise ValueError("PREP_COMPLETE job ID is not the recorded resume preparation")
        if prep_complete.get("repo_commit") != repair_commit:
            raise ValueError("PREP_COMPLETE repair commit mismatch")
        finalized_manifest_sha = sha256_file(finalized_manifest_path)
        if prep_complete.get("data_manifest_sha256") != finalized_manifest_sha:
            raise ValueError("PREP_COMPLETE finalized-manifest hash mismatch")
        finalized_manifest = json.loads(finalized_manifest_path.read_text(encoding="utf-8"))
        if finalized_manifest.get("phase") != "finalized_verified_dataset":
            raise ValueError("Downstream stage requires finalized verified data")
    if stage == "evaluate":
        train_complete_path = OUTPUT_ROOT / "model/apps_repaired_pilot/TRAIN_COMPLETE"
        model_manifest_path = (
            OUTPUT_ROOT
            / "model/apps_repaired_pilot/repaired_pilot_model_manifest.json"
        )
        if not train_complete_path.is_file() or train_complete_path.is_symlink():
            raise ValueError("Missing or unsafe TRAIN_COMPLETE for evaluation")
        train_complete = read_unique_kv(train_complete_path)
        if train_complete.get("job_id") != by_stage["train"]["job_id"]:
            raise ValueError("TRAIN_COMPLETE job ID is not the recorded resume training")
        if train_complete.get("repo_commit") != repair_commit:
            raise ValueError("TRAIN_COMPLETE repair commit mismatch")
        if train_complete.get("model_manifest_sha256") != sha256_file(model_manifest_path):
            raise ValueError("TRAIN_COMPLETE model-manifest hash mismatch")

    verify_self_seal(resumed_path, "dispatch_sha256")
    resumed = read_unique_kv(resumed_path)
    if resumed.get("repair_repo_commit") != repair_commit:
        raise ValueError("RESUMED repair commit mismatch")
    if resumed.get("addendum_sha256") != addendum_seal:
        raise ValueError("RESUMED addendum hash mismatch")
    if resumed.get("jobs_sha256") != sha256_file(jobs_path):
        raise ValueError("RESUMED job-record hash mismatch")
    for name in RESUME_MINUTES:
        if resumed.get(f"{name}_job_id") != by_stage[name]["job_id"]:
            raise ValueError(f"RESUMED job ID mismatch for {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(ORIGINAL_LIMITS), required=True)
    parser.add_argument("--time-limit", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+", args.job_id):
        raise ValueError("Invalid Slurm job ID")

    require_hashes(ORIGINAL_HASHES)
    auth = read_unique_kv(CONTROL_ROOT / "AUTHORIZED_MAX_COST_USD_1.80")
    if auth.get("repo_commit") != ORIGINAL_COMMIT:
        raise ValueError("Original authorization commit mismatch")
    if auth.get("max_h200_minutes") != "120" or auth.get("ack_max_cost_usd") != "1.80":
        raise ValueError("Original authorization ceiling mismatch")
    if git("status", "--porcelain"):
        raise ValueError("Refusing a dirty Tillicum checkout")

    addendum_path = RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
    if addendum_path.exists():
        # Preparation intentionally replaces the provisional manifest with a
        # finalized dataset manifest. Later stages therefore re-audit the
        # finalized dataset in their sbatch scripts, while the immutable raw
        # candidate inputs remain hash-pinned here for every stage.
        require_hashes(prepared_hashes_for_stage(args.stage))
        require_hashes(FAILED_LOG_HASHES)
        verify_resume(args.stage, args.time_limit, args.job_id)
        mode = "repair_resume"
    else:
        if git("rev-parse", "HEAD") != ORIGINAL_COMMIT:
            raise ValueError("Checkout does not match original authorization")
        if args.time_limit != ORIGINAL_LIMITS[args.stage]:
            raise ValueError(f"Unsafe original TimeLimit for {args.stage}")
        with open(CONTROL_ROOT / "jobs.tsv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        matches = [row for row in rows if row.get("stage") == args.stage]
        if len(matches) != 1 or matches[0].get("job_id") != args.job_id:
            raise ValueError("Current job ID is not the sealed original stage job")
        mode = "original"
    print(f"Authorized repaired-pilot job: mode={mode} stage={args.stage} job={args.job_id}")


if __name__ == "__main__":
    main()
