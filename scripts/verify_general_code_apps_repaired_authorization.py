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
FIRST_RESUME_ROOT = CONTROL_ROOT / "resume_227440"
SECOND_RESUME_ROOT = CONTROL_ROOT / "resume_227440_compat2"
THIRD_RESUME_ROOT = CONTROL_ROOT / "resume_227440_compat3"
RESUME_ROOT = CONTROL_ROOT / "resume_227440_compat4"

ORIGINAL_COMMIT = "a57dbf43fdf296dfdd31f14447e9a47e76db0405"
FIRST_REPAIR_COMMIT = "00ad408f5596270d77bd52f26dcedc3366277e00"
SECOND_REPAIR_COMMIT = "531eac8565d68738958b06fae9363af0a4bebf01"
THIRD_REPAIR_COMMIT = "b81f126f04ebba38eb0f81d9acf77f7d43862398"
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
FIRST_RESUME_HASHES = {
    CONTROL_ROOT / "RESUME_227440_SUBMISSION_LOCK/owner":
        "ce10532a9709e85e0c9073fbe1e0d66d83d117716bed82bba1ae6cb1dddba50d",
    FIRST_RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP":
        "7112eb9ff96ff3ae5997fb4be01b55d2209cc65873616f20c86a116868c34b90",
    FIRST_RESUME_ROOT / "jobs.tsv":
        "cdc7735b18d4b0acb622d4c4b76c16c90f632fe8a723f4bbbed563f134876ae0",
    FIRST_RESUME_ROOT / "RESUMED":
        "3839fb3b2ec847d49ca1be477355c4be7a3c763b75a1d42b0fa4830e424be9cf",
    FIRST_RESUME_ROOT / "dispatch_attempt.tsv":
        "cdc7735b18d4b0acb622d4c4b76c16c90f632fe8a723f4bbbed563f134876ae0",
}
SECOND_RESUME_HASHES = {
    CONTROL_ROOT / "RESUME_227440_COMPAT2_SUBMISSION_LOCK/owner":
        "99a16e11621d2fbb08e3c361e9431863465933d75314317fc0a7644fe342782d",
    SECOND_RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP":
        "79bf2d9316312aad8fc004478892d19deba228f757c10abece670f7e9524498f",
    SECOND_RESUME_ROOT / "jobs.tsv":
        "a5d619a61152765b39f68d6e67ef9cbd352b8a34925c24f9800683e5201f4673",
    SECOND_RESUME_ROOT / "RESUMED":
        "7860dc64f230845b5bc108316bb0fc0100b6937e88997b942e2749e3498be785",
    SECOND_RESUME_ROOT / "dispatch_attempt.tsv":
        "a5d619a61152765b39f68d6e67ef9cbd352b8a34925c24f9800683e5201f4673",
}
THIRD_RESUME_HASHES = {
    CONTROL_ROOT / "RESUME_227440_COMPAT3_SUBMISSION_LOCK/owner":
        "2ab6715167b33a7f1dc07314189790b72cfbead01fe28d5eb1ecdc09841bd2dc",
    THIRD_RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP":
        "ccd6bd140b89db82f5532e008242ee7e353cd4744467241fd864e8fbd4558a13",
    THIRD_RESUME_ROOT / "jobs.tsv":
        "1deba4333ea56da7e445ace82f832aa10dfa5ae44c5f3de6f5a8f5740cd42146",
    THIRD_RESUME_ROOT / "RESUMED":
        "5843cd2d11686324ad34574d82479ae032b9cf61c677229f2b1a116f11fbd2e9",
    THIRD_RESUME_ROOT / "dispatch_attempt.tsv":
        "1deba4333ea56da7e445ace82f832aa10dfa5ae44c5f3de6f5a8f5740cd42146",
}
THIRD_FAILED_LOG_HASHES = {
    TILLICUM_ROOT / "outputs/logs/general_code_apps_repaired_prepare_229023.out":
        "16301b591ee3648aa3d36069ebfd4bc58864f73322d06d447d29f4bf98a7f8b8",
    TILLICUM_ROOT / "outputs/logs/general_code_apps_repaired_prepare_229023.err":
        "a82091cf7ac38f88a52302dd66ea6f62403e85d8768ce0623a799ca6535dd00b",
}
THIRD_MALFORMED_EVALUATION_SHA256 = (
    "beaa14632d87006030fa669ead82222b9f93c6e3b96d209580548683d6560eb5"
)
APPS_IO_ENCODING = "livecodebench_testing_util_v1"
APPS_RAW_SHA256 = "45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae"
MIGRATED_IMMUTABLE_HASHES = {
    OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json":
        PREPARED_HASHES[OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json"],
    OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json":
        PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json"
        ],
    OUTPUT_ROOT / "data/apps_repaired_candidate_prompts.json":
        PREPARED_HASHES[OUTPUT_ROOT / "data/apps_repaired_candidate_prompts.json"],
    OUTPUT_ROOT / "data/apps_repaired_candidates_evaluator.jsonl":
        PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidates_evaluator.jsonl"
        ],
    OUTPUT_ROOT / "data/apps_repaired_candidates.evaluation.json":
        THIRD_MALFORMED_EVALUATION_SHA256,
    OUTPUT_ROOT / "data/evidence/job_229023/data_manifest.json":
        PREPARED_HASHES[OUTPUT_ROOT / "data/data_manifest.json"],
    OUTPUT_ROOT
    / "data/evidence/job_229023/general_code_apps_repaired_prepare_229023.out":
        THIRD_FAILED_LOG_HASHES[
            TILLICUM_ROOT
            / "outputs/logs/general_code_apps_repaired_prepare_229023.out"
        ],
    OUTPUT_ROOT
    / "data/evidence/job_229023/general_code_apps_repaired_prepare_229023.err":
        THIRD_FAILED_LOG_HASHES[
            TILLICUM_ROOT
            / "outputs/logs/general_code_apps_repaired_prepare_229023.err"
        ],
    OUTPUT_ROOT
    / "data/raw/apps-train-21e74ddf8de1a21436da12e3e653065c5213e9d1.jsonl":
        APPS_RAW_SHA256,
}
ORIGINAL_LIMITS = {"prepare": "00:30:00", "train": "00:30:00", "evaluate": "01:00:00"}
RESUME_LIMITS = {"prepare": "00:28:00", "train": "00:30:00", "evaluate": "01:00:00"}
RESUME_MINUTES = {"prepare": "28", "train": "30", "evaluate": "60"}


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


def prepared_hashes_for_stage(stage, migrated=False):
    if migrated:
        return MIGRATED_IMMUTABLE_HASHES
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


def verify_manifest_payload_seal(manifest):
    payload = dict(manifest)
    expected = payload.pop("manifest_payload_sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise ValueError("Migrated data-manifest integrity seal mismatch")


def verify_migrated_manifest(stage, repair_commit):
    """Bind the runtime-created schema migration without pre-knowing its hash."""
    manifest_path = OUTPUT_ROOT / "data/data_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Missing or unsafe migrated data manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_payload_seal(manifest)
    phase = manifest.get("phase")
    if stage != "prepare" and phase != "finalized_verified_dataset":
        raise ValueError("Downstream stage requires finalized migrated data")
    if phase not in {"prepared_unverified_candidates", "finalized_verified_dataset"}:
        raise ValueError("Unexpected migrated data phase")

    expected_config = {
        "seed": 7302026,
        "train_per_kind": 1200,
        "validation_per_kind": 100,
        "max_candidates": 2,
        "verification_per_kind": 1400,
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "base_model_revision": "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        "train_max_tokens": 2048,
        "validation_max_context": 4096,
        "validation_max_new_tokens": 1024,
    }
    if manifest.get("config") != expected_config:
        raise ValueError("Migrated data configuration drifted")
    source = manifest.get("source", {})
    if (
        source.get("dataset_id") != "codeparrot/apps"
        or source.get("revision")
        != "21e74ddf8de1a21436da12e3e653065c5213e9d1"
        or source.get("sha256") != APPS_RAW_SHA256
        or source.get("row_count") != 5000
    ):
        raise ValueError("Migrated APPS source identity drifted")

    migration = manifest.get("io_schema_migration", {})
    runner_path = REPO_ROOT / "scripts/run_lcb_sandbox_evaluation.py"
    runner_sha256 = sha256_file(runner_path)
    expected_migration = {
        "reason": "apps_native_io_values_were_not_encoded_for_lcb_checker",
        "converter_version": APPS_IO_ENCODING,
        "evaluator_mode": "apps_official",
        "runner_script_sha256": runner_sha256,
        "repair_repo_commit": repair_commit,
        "legacy_manifest_sha256": PREPARED_HASHES[
            OUTPUT_ROOT / "data/data_manifest.json"
        ],
        "legacy_evaluator_sha256": PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidates_evaluator.jsonl"
        ],
        "legacy_evaluation_sha256": THIRD_MALFORMED_EVALUATION_SHA256,
        "candidate_custom_sha256": PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json"
        ],
        "candidate_custom_meta_sha256": PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json"
        ],
        "candidate_prompts_sha256": PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidate_prompts.json"
        ],
        "source_raw_sha256": APPS_RAW_SHA256,
        "legacy_failed_stdout_sha256": THIRD_FAILED_LOG_HASHES[
            TILLICUM_ROOT
            / "outputs/logs/general_code_apps_repaired_prepare_229023.out"
        ],
        "legacy_failed_stderr_sha256": THIRD_FAILED_LOG_HASHES[
            TILLICUM_ROOT
            / "outputs/logs/general_code_apps_repaired_prepare_229023.err"
        ],
        "n_questions": 2800,
        "n_samples": 2,
    }
    for key, expected in expected_migration.items():
        if migration.get(key) != expected:
            raise ValueError(f"Migrated data mismatch for {key}")

    artifacts = manifest.get("artifacts", {})
    expected_artifacts = {
        "candidate_custom": (
            "apps_repaired_candidates.custom.json",
            PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json"
            ],
        ),
        "candidate_custom_meta": (
            "apps_repaired_candidates.custom.meta.json",
            PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json"
            ],
        ),
        "candidate_prompts": (
            "apps_repaired_candidate_prompts.json",
            PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidate_prompts.json"
            ],
        ),
        "legacy_malformed_evaluator": (
            "apps_repaired_candidates_evaluator.jsonl",
            PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates_evaluator.jsonl"
            ],
        ),
        "legacy_malformed_evaluation": (
            "apps_repaired_candidates.evaluation.json",
            THIRD_MALFORMED_EVALUATION_SHA256,
        ),
        "legacy_manifest": (
            "evidence/job_229023/data_manifest.json",
            PREPARED_HASHES[OUTPUT_ROOT / "data/data_manifest.json"],
        ),
        "legacy_failed_stdout": (
            "evidence/job_229023/general_code_apps_repaired_prepare_229023.out",
            THIRD_FAILED_LOG_HASHES[
                TILLICUM_ROOT
                / "outputs/logs/general_code_apps_repaired_prepare_229023.out"
            ],
        ),
        "legacy_failed_stderr": (
            "evidence/job_229023/general_code_apps_repaired_prepare_229023.err",
            THIRD_FAILED_LOG_HASHES[
                TILLICUM_ROOT
                / "outputs/logs/general_code_apps_repaired_prepare_229023.err"
            ],
        ),
        "source_raw": (
            "raw/apps-train-21e74ddf8de1a21436da12e3e653065c5213e9d1.jsonl",
            APPS_RAW_SHA256,
        ),
    }
    for label, (relative, digest) in expected_artifacts.items():
        artifact = artifacts.get(label, {})
        if (
            artifact.get("kind") != "file"
            or artifact.get("path") != relative
            or artifact.get("sha256") != digest
        ):
            raise ValueError(f"Migrated artifact binding mismatch: {label}")

    evaluator_artifact = artifacts.get("candidate_evaluator", {})
    evaluator_path = (
        OUTPUT_ROOT
        / "data/apps_repaired_candidates_evaluator.apps-io-v1.jsonl"
    )
    corrected_sha = evaluator_artifact.get("sha256")
    if (
        evaluator_artifact.get("kind") != "file"
        or evaluator_artifact.get("path")
        != "apps_repaired_candidates_evaluator.apps-io-v1.jsonl"
        or corrected_sha != migration.get("corrected_evaluator_sha256")
        or corrected_sha
        == PREPARED_HASHES[
            OUTPUT_ROOT / "data/apps_repaired_candidates_evaluator.jsonl"
        ]
        or corrected_sha != sha256_file(evaluator_path)
    ):
        raise ValueError("Corrected evaluator is not exactly migration-bound")
    row_count = 0
    with open(evaluator_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            if json.loads(line).get("apps_io_encoding") != APPS_IO_ENCODING:
                raise ValueError("Corrected evaluator row lacks I/O-schema marker")
    if row_count != 2800:
        raise ValueError("Corrected evaluator row count drifted")

    if phase == "finalized_verified_dataset":
        verification = manifest.get("verification_result", {})
        result_path = (
            OUTPUT_ROOT
            / "data/apps_repaired_candidates.apps-io-v1.evaluation.json"
        )
        if (
            verification.get("path") != str(result_path)
            or verification.get("sha256") != sha256_file(result_path)
            or verification.get("sha256") == THIRD_MALFORMED_EVALUATION_SHA256
            or verification.get("benchmark_file_sha256") != corrected_sha
            or verification.get("custom_output_sha256")
            != PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json"
            ]
            or verification.get("custom_meta_sha256")
            != PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json"
            ]
            or verification.get("livecodebench_commit")
            != "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
            or verification.get("apps_io_encoding") != APPS_IO_ENCODING
            or verification.get("evaluator_mode") != "apps_official"
            or verification.get("runner_script_sha256") != runner_sha256
            or verification.get("n_questions") != 2800
            or verification.get("n_samples") != 2
        ):
            raise ValueError("Corrected evaluation result binding drifted")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_artifact = artifacts.get("verification_evaluation", {})
        if (
            result_artifact.get("kind") != "file"
            or result_artifact.get("path")
            != "apps_repaired_candidates.apps-io-v1.evaluation.json"
            or result_artifact.get("sha256") != verification.get("sha256")
        ):
            raise ValueError("Corrected evaluation artifact binding drifted")
        result_meta = result.get("meta", {})
        expected_result_meta = {
            "benchmark_file_sha256": corrected_sha,
            "custom_output_sha256": PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates.custom.json"
            ],
            "custom_meta_sha256": PREPARED_HASHES[
                OUTPUT_ROOT / "data/apps_repaired_candidates.custom.meta.json"
            ],
            "livecodebench_commit": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
            "evaluator_mode": "apps_official",
            "runner_script_sha256": runner_sha256,
            "n_questions": 2800,
            "n_samples": 2,
        }
        for key, expected in expected_result_meta.items():
            if result_meta.get(key) != expected:
                raise ValueError(f"Corrected evaluation metadata drifted: {key}")
        selection = manifest.get("selection", {})
        if selection.get("train_count_by_kind") != {"stdio": 1200, "function": 1200}:
            raise ValueError("Migrated training quotas drifted")
        if selection.get("validation_count_by_kind") != {"stdio": 100, "function": 100}:
            raise ValueError("Migrated validation quotas drifted")
    return manifest


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def verify_resume(stage, time_limit, job_id, control_only=False):
    addendum_path = RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
    jobs_path = RESUME_ROOT / "jobs.tsv"
    resumed_path = RESUME_ROOT / "RESUMED"
    required_paths = (addendum_path,) if control_only else (addendum_path, jobs_path, resumed_path)
    for path in required_paths:
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
        "first_repair_repo_commit": FIRST_REPAIR_COMMIT,
        "first_resume_authorization_sha256": FIRST_RESUME_HASHES[
            FIRST_RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
        ],
        "first_resume_jobs_sha256": FIRST_RESUME_HASHES[
            FIRST_RESUME_ROOT / "jobs.tsv"
        ],
        "first_resume_resumed_sha256": FIRST_RESUME_HASHES[
            FIRST_RESUME_ROOT / "RESUMED"
        ],
        "first_resume_lock_owner_sha256": FIRST_RESUME_HASHES[
            CONTROL_ROOT / "RESUME_227440_SUBMISSION_LOCK/owner"
        ],
        "first_dispatch_prepare_job_id": "228953",
        "first_dispatch_prepare_state": "CANCELLED",
        "first_dispatch_prepare_elapsed_seconds": "0",
        "first_dispatch_train_job_id": "228954",
        "first_dispatch_train_state": "CANCELLED",
        "first_dispatch_train_elapsed_seconds": "0",
        "first_dispatch_evaluate_job_id": "228955",
        "first_dispatch_evaluate_state": "CANCELLED",
        "first_dispatch_evaluate_elapsed_seconds": "0",
        "first_dispatch_failure_reason": "tillicum_pending_jobs_omit_tresperjob",
        "second_repair_repo_commit": SECOND_REPAIR_COMMIT,
        "second_resume_authorization_sha256": SECOND_RESUME_HASHES[
            SECOND_RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
        ],
        "second_resume_jobs_sha256": SECOND_RESUME_HASHES[
            SECOND_RESUME_ROOT / "jobs.tsv"
        ],
        "second_resume_resumed_sha256": SECOND_RESUME_HASHES[
            SECOND_RESUME_ROOT / "RESUMED"
        ],
        "second_resume_lock_owner_sha256": SECOND_RESUME_HASHES[
            CONTROL_ROOT / "RESUME_227440_COMPAT2_SUBMISSION_LOCK/owner"
        ],
        "second_dispatch_prepare_job_id": "228992",
        "second_dispatch_prepare_state": "CANCELLED",
        "second_dispatch_prepare_elapsed_seconds": "0",
        "second_dispatch_train_job_id": "228993",
        "second_dispatch_train_state": "CANCELLED",
        "second_dispatch_train_elapsed_seconds": "0",
        "second_dispatch_evaluate_job_id": "228994",
        "second_dispatch_evaluate_state": "CANCELLED",
        "second_dispatch_evaluate_elapsed_seconds": "0",
        "second_dispatch_failure_reason": "reqtres_split_on_every_equals",
        "third_repair_repo_commit": THIRD_REPAIR_COMMIT,
        "third_resume_authorization_sha256": THIRD_RESUME_HASHES[
            THIRD_RESUME_ROOT / "AUTHORIZED_REPAIR_WITHIN_ORIGINAL_CAP"
        ],
        "third_resume_jobs_sha256": THIRD_RESUME_HASHES[
            THIRD_RESUME_ROOT / "jobs.tsv"
        ],
        "third_resume_resumed_sha256": THIRD_RESUME_HASHES[
            THIRD_RESUME_ROOT / "RESUMED"
        ],
        "third_resume_lock_owner_sha256": THIRD_RESUME_HASHES[
            CONTROL_ROOT / "RESUME_227440_COMPAT3_SUBMISSION_LOCK/owner"
        ],
        "third_dispatch_prepare_job_id": "229023",
        "third_dispatch_prepare_state": "FAILED",
        "third_dispatch_prepare_elapsed_seconds": "55",
        "third_dispatch_prepare_timelimit_minutes": "29",
        "third_dispatch_train_job_id": "229024",
        "third_dispatch_train_state": "CANCELLED",
        "third_dispatch_train_elapsed_seconds": "0",
        "third_dispatch_evaluate_job_id": "229025",
        "third_dispatch_evaluate_state": "CANCELLED",
        "third_dispatch_evaluate_elapsed_seconds": "0",
        "third_dispatch_failure_reason": (
            "apps_native_io_values_not_encoded_for_lcb_checker"
        ),
        "third_failed_stdout_sha256": THIRD_FAILED_LOG_HASHES[
            TILLICUM_ROOT
            / "outputs/logs/general_code_apps_repaired_prepare_229023.out"
        ],
        "third_failed_stderr_sha256": THIRD_FAILED_LOG_HASHES[
            TILLICUM_ROOT
            / "outputs/logs/general_code_apps_repaired_prepare_229023.err"
        ],
        "third_malformed_evaluation_sha256": THIRD_MALFORMED_EVALUATION_SHA256,
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
        "prior_rounded_h200_minutes": "2",
        "resume_prepare_minutes": "28",
        "resume_train_minutes": "30",
        "resume_evaluate_minutes": "60",
        "remaining_h200_minutes": "118",
        "cumulative_max_h200_minutes": "120",
        "cumulative_max_cost_usd": "1.80",
        "no_requeue": "true",
        "automatic_continuation": "false",
        "reason": "apps_native_io_schema_conversion_repair",
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
    if git("rev-parse", "HEAD^") != THIRD_REPAIR_COMMIT:
        raise ValueError("I/O-schema repair is not a child of the ReqTRES repair")
    if git("rev-parse", "HEAD~2") != SECOND_REPAIR_COMMIT:
        raise ValueError("ReqTRES compatibility repair chain is discontinuous")
    if git("rev-parse", "HEAD~3") != FIRST_REPAIR_COMMIT:
        raise ValueError("Parser compatibility repair chain is discontinuous")
    if git("rev-parse", "HEAD~4") != ORIGINAL_COMMIT:
        raise ValueError("Repair chain does not descend from the authorized commit")
    if len(git("rev-list", "--parents", "-n", "1", "HEAD").split()) != 2:
        raise ValueError("Compatibility repair must have exactly one parent")
    if len(
        git("rev-list", "--parents", "-n", "1", FIRST_REPAIR_COMMIT).split()
    ) != 2:
        raise ValueError("First repair must have exactly one parent")
    if len(
        git("rev-list", "--parents", "-n", "1", SECOND_REPAIR_COMMIT).split()
    ) != 2:
        raise ValueError("Second repair must have exactly one parent")
    if len(
        git("rev-list", "--parents", "-n", "1", THIRD_REPAIR_COMMIT).split()
    ) != 2:
        raise ValueError("Third repair must have exactly one parent")
    expected_diff = addendum.get("repair_diff_sha256", "")
    actual_diff = hashlib.sha256(
        subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "diff", "--binary", ORIGINAL_COMMIT, repair_commit]
        )
    ).hexdigest()
    if expected_diff != actual_diff:
        raise ValueError("Repair diff hash mismatch")
    manifest_path = OUTPUT_ROOT / "data/data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("io_schema_migration") is not None:
        verify_migrated_manifest(stage, repair_commit)
    elif stage != "prepare":
        raise ValueError("Downstream stage cannot use the malformed APPS I/O schema")
    if time_limit != RESUME_LIMITS[stage]:
        raise ValueError(f"Unsafe resume TimeLimit for {stage}: {time_limit}")
    if control_only:
        return

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
    parser.add_argument("--job-id", default="0")
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()
    if not args.control_only and not re.fullmatch(r"[0-9]+", args.job_id):
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
        manifest_path = OUTPUT_ROOT / "data/data_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Missing or unsafe repaired-pilot data manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        migrated = manifest.get("io_schema_migration") is not None
        require_hashes(prepared_hashes_for_stage(args.stage, migrated=migrated))
        if not migrated:
            require_hashes(
                {
                    OUTPUT_ROOT / "data/apps_repaired_candidates.evaluation.json":
                        THIRD_MALFORMED_EVALUATION_SHA256
                }
            )
        require_hashes(FAILED_LOG_HASHES)
        require_hashes(FIRST_RESUME_HASHES)
        require_hashes(SECOND_RESUME_HASHES)
        require_hashes(THIRD_RESUME_HASHES)
        require_hashes(THIRD_FAILED_LOG_HASHES)
        verify_resume(
            args.stage, args.time_limit, args.job_id, control_only=args.control_only
        )
        mode = "repair_resume"
    else:
        if args.control_only:
            raise ValueError("Control-only verification requires a repair addendum")
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
    print(
        f"Authorized repaired-pilot control: mode={mode} stage={args.stage} "
        f"job={args.job_id} control_only={args.control_only}"
    )


if __name__ == "__main__":
    main()
