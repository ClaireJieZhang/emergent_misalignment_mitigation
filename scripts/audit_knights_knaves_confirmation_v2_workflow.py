#!/usr/bin/env python3
"""Seal and verify the capped K&K v2 evaluation-only workflow."""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile

import yaml

import prepare_knights_knaves_confirmation_v2_data as v2_data
import prepare_knights_knaves_pilot_data as v1_data
import sample_knights_knaves_generations as sampler


V1_TRAINING_COMMIT = "900044e6d171a08fc0b19e364695c627c21b591a"
V1_DATA_MANIFEST_PAYLOAD_SHA256 = (
    "dd78325d5665dfdf08194c894bd5e3a84985924c506f980bd5786467c0933c76"
)
V1_MODEL_MANIFEST_SHA256 = (
    "3f699bfd982fb3733422613319366a269f2c97cb82bfdac61ad1897c0da319de"
)
V1_SELECTION_SHA256 = (
    "0bb8b69510c4bf0f0ddcd7e36b70e8e79d1bf9b1bc0919c943f162a351abc111"
)
V1_STOP_SHA256 = "8b3c38df26a6f7289278a4bf66d001cf17c759f85832fe4511f92dc08ba53bdf"
TRAINING_CONFIG_RELATIVE_PATH = "configs/training_qwen25_7b_kk_reasoning_pilot.yaml"
TRAINING_CONFIG_SHA256 = (
    "5caef6baeb07f4ab4de8901001d7adb02433794e15c1024a950dc3bf59f492cb"
)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
CHECKPOINT_STEP = 192
CHECKPOINT_FINGERPRINT = (
    "36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c"
)
CHECKPOINT_WEIGHT_SHA256 = (
    "9dc6b793da461c1b6ff48451205436c2dbf12bfa65b0c8a36ed429f6f6ba1c33"
)
CHECKPOINT_CONFIG_SHA256 = (
    "1a485aee95732d7012a29b8c5f05f37efa60bc05cf2a2802f098139638e7de17"
)
CHECKPOINT_TRAINER_STATE_SHA256 = (
    "3d78b89a8267c22c241342a285d4707ef2f095dcba027942dccfa502d69f3e78"
)
V1_JOB_IDS = {"train": "232832", "evaluate": "232833"}
V1_RELEASED_MAX_MINUTES = 150
V2_MAX_MINUTES = 30
CUMULATIVE_RELEASED_MAX_MINUTES = 180
IMMUTABLE_CUMULATIVE_CEILING_MINUTES = 240
MAX_COST_USD = "0.45"
H200_USD_PER_HOUR = "0.90"
SEAL_FIELD = "payload_sha256"


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(value):
    result = dict(value)
    result.pop(SEAL_FIELD, None)
    result[SEAL_FIELD] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_seal(value, label):
    unsealed = dict(value)
    recorded = unsealed.pop(SEAL_FIELD, None)
    if recorded != hashlib.sha256(canonical_json_bytes(unsealed)).hexdigest():
        raise ValueError(f"Integrity seal mismatch: {label}")


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_file(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe regular file: {path}")
    return path


def load_json(path):
    with open(require_file(path), encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def git_state(repo_root):
    commit = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain"], text=True
    )
    if dirty:
        raise ValueError("Refusing a dirty v2 workflow checkout")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Invalid repository commit")
    return commit


def audit_training_config(repo_root):
    """Bind v2 inference to the exact configuration used by the v1 pilot."""
    path = require_file(os.path.join(repo_root, TRAINING_CONFIG_RELATIVE_PATH))
    observed_sha = sha256_file(path)
    if observed_sha != TRAINING_CONFIG_SHA256:
        raise ValueError("Frozen K&K training configuration hash changed")
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Frozen K&K training configuration is not a mapping")
    if config.get("base_model") != BASE_MODEL:
        raise ValueError("Frozen K&K training configuration names another base model")
    if config.get("base_model_revision") != BASE_MODEL_REVISION:
        raise ValueError("Frozen K&K training configuration names another revision")
    return {
        "path": path,
        "sha256": observed_sha,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
    }


def audit_adapter_base_identity(adapter_config_path):
    """Verify the adapter records the pinned base identity where PEFT exposes it."""
    config = load_json(adapter_config_path)
    adapter_base = config.get("base_model_name_or_path")
    if adapter_base != BASE_MODEL:
        raise ValueError(
            "Frozen checkpoint adapter records another base model: "
            f"{adapter_base!r}"
        )
    # PEFT commonly leaves this null even when the source model was loaded at
    # an immutable revision.  If present, it must agree with the pinned config;
    # the exact adapter-config SHA binds the null/explicit value either way.
    adapter_revision = config.get("revision")
    if adapter_revision not in (None, BASE_MODEL_REVISION):
        raise ValueError(
            "Frozen checkpoint adapter records another base revision: "
            f"{adapter_revision!r}"
        )
    return {
        "base_model_name_or_path": adapter_base,
        "revision": adapter_revision,
    }


def audit_v1(v1_root):
    v1_root = os.path.abspath(v1_root)
    data_root = os.path.join(v1_root, "data")
    manifest = v1_data.audit_existing_output(data_root)
    if manifest.get("manifest_payload_sha256") != V1_DATA_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("V1 data manifest payload changed")
    paths = {
        "data_manifest": os.path.join(data_root, "data_manifest.json"),
        "model_manifest": os.path.join(
            v1_root, "model", "kk_reasoning_n5_pilot", "MODEL_MANIFEST.json"
        ),
        "selection": os.path.join(v1_root, "evaluation", "selection", "summary.json"),
        "stop": os.path.join(v1_root, "control", "STOPPED_NO_GO"),
    }
    expected_hashes = {
        "model_manifest": V1_MODEL_MANIFEST_SHA256,
        "selection": V1_SELECTION_SHA256,
        "stop": V1_STOP_SHA256,
    }
    for name, expected in expected_hashes.items():
        if sha256_file(require_file(paths[name])) != expected:
            raise ValueError(f"V1 {name} artifact changed")
    selection = load_json(paths["selection"])
    if selection.get("gate", {}).get("decision") != "STOP":
        raise ValueError("V1 decision is no longer STOP")
    if selection.get("selected", {}).get("step") != CHECKPOINT_STEP:
        raise ValueError("V1 frozen checkpoint is no longer step 192")
    if selection.get("selected", {}).get("model_fingerprint") != CHECKPOINT_FINGERPRINT:
        raise ValueError("V1 selected adapter fingerprint changed")
    if os.path.lexists(os.path.join(v1_root, "control", "GO_KK_SEALED_FINAL")):
        raise ValueError("V1 unexpectedly authorized sealed-final evaluation")
    if os.path.lexists(os.path.join(v1_root, "control", "GO_KK_BENEFIT_UNIONS")):
        raise ValueError("V1 unexpectedly authorized medical unions")
    final_eval_root = os.path.join(v1_root, "evaluation", "sealed_final")
    if os.path.isdir(final_eval_root) and os.listdir(final_eval_root):
        raise ValueError("V1 sealed-final evaluation directory is not empty")
    for folder in ("generations", "scores"):
        root = os.path.join(v1_root, "evaluation", folder)
        if not os.path.isdir(root):
            raise ValueError(f"Missing v1 evaluation folder: {folder}")
        unexpected = [name for name in os.listdir(root) if not name.startswith("dev_n5__")]
        if unexpected:
            raise ValueError(f"V1 contains non-dev {folder}: {unexpected}")

    checkpoint = os.path.join(
        v1_root, "model", "kk_reasoning_n5_pilot", "checkpoint-192"
    )
    artifacts = {
        "adapter_model.safetensors": CHECKPOINT_WEIGHT_SHA256,
        "adapter_config.json": CHECKPOINT_CONFIG_SHA256,
        "trainer_state.json": CHECKPOINT_TRAINER_STATE_SHA256,
    }
    for filename, expected in artifacts.items():
        if sha256_file(require_file(os.path.join(checkpoint, filename))) != expected:
            raise ValueError(f"Frozen checkpoint artifact changed: {filename}")
    adapter_identity = audit_adapter_base_identity(
        os.path.join(checkpoint, "adapter_config.json")
    )
    if sampler.adapter_fingerprint(checkpoint) != CHECKPOINT_FINGERPRINT:
        raise ValueError("Frozen checkpoint composite fingerprint changed")

    jobs_path = require_file(os.path.join(v1_root, "control", "jobs.tsv"))
    with open(jobs_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {row.get("stage"): row.get("job_id") for row in rows} != V1_JOB_IDS:
        raise ValueError("V1 recorded job IDs differ")
    if any(row.get("max_minutes") != "75" or row.get("released") != "true" for row in rows):
        raise ValueError("V1 job caps or release records differ")
    return {
        "root": v1_root,
        "data_root": data_root,
        "data_manifest_sha256": sha256_file(paths["data_manifest"]),
        "model_manifest_sha256": V1_MODEL_MANIFEST_SHA256,
        "selection_sha256": V1_SELECTION_SHA256,
        "stop_sha256": V1_STOP_SHA256,
        "jobs_sha256": sha256_file(jobs_path),
        "checkpoint": checkpoint,
        "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
        "checkpoint_weight_sha256": CHECKPOINT_WEIGHT_SHA256,
        "checkpoint_config_sha256": CHECKPOINT_CONFIG_SHA256,
        "checkpoint_adapter_base_model": adapter_identity[
            "base_model_name_or_path"
        ],
        "checkpoint_adapter_revision": adapter_identity["revision"],
        "training_commit": V1_TRAINING_COMMIT,
    }


def prep_record(repo_root, v1_root, v2_data_root):
    commit = git_state(repo_root)
    training_config = audit_training_config(repo_root)
    parent = audit_v1(v1_root)
    manifest = v2_data.audit_output(v2_data_root, parent["data_root"])
    manifest_path = os.path.join(v2_data_root, v2_data.MANIFEST_NAME)
    return sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_confirmation_v2_preparation",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "evaluation_commit": commit,
            "training_commit": parent["training_commit"],
            "training_config": training_config["path"],
            "training_config_sha256": training_config["sha256"],
            "base_model": training_config["base_model"],
            "base_model_revision": training_config["base_model_revision"],
            "v1_root": parent["root"],
            "v1_data_manifest_sha256": parent["data_manifest_sha256"],
            "v1_model_manifest_sha256": parent["model_manifest_sha256"],
            "v1_selection_sha256": parent["selection_sha256"],
            "v1_stop_sha256": parent["stop_sha256"],
            "v1_jobs_sha256": parent["jobs_sha256"],
            "checkpoint_step": CHECKPOINT_STEP,
            "checkpoint_fingerprint": parent["checkpoint_fingerprint"],
            "checkpoint_weight_sha256": parent["checkpoint_weight_sha256"],
            "checkpoint_config_sha256": parent["checkpoint_config_sha256"],
            "checkpoint_adapter_base_model": parent[
                "checkpoint_adapter_base_model"
            ],
            "checkpoint_adapter_revision": parent[
                "checkpoint_adapter_revision"
            ],
            "v2_data_root": os.path.abspath(v2_data_root),
            "v2_data_manifest_sha256": sha256_file(manifest_path),
            "v2_data_manifest_payload_sha256": manifest[
                "manifest_payload_sha256"
            ],
            "confirmation_seed": v2_data.CONFIRMATION_SEED,
            "confirmation_rows": v2_data.CONFIRMATION_ROWS,
            "sealed_final_model_generations_present": False,
            "gpu_allocation_minutes": 0,
        }
    )


def verify_prep(path, repo_root, v1_root, v2_data_root):
    observed = load_json(path)
    verify_seal(observed, path)
    expected = prep_record(repo_root, v1_root, v2_data_root)
    ignored = {"created_at", SEAL_FIELD}
    for key in set(expected) - ignored:
        if observed.get(key) != expected.get(key):
            raise ValueError(f"V2 preparation record mismatch for {key}")
    return observed


def command_write_prep(args):
    record = prep_record(args.repo_root, args.v1_root, args.v2_data_root)
    if os.path.lexists(args.output_file):
        verify_prep(
            args.output_file, args.repo_root, args.v1_root, args.v2_data_root
        )
        print("Audited existing K&K v2 preparation record")
    else:
        atomic_write_json(args.output_file, record)
        verify_prep(
            args.output_file, args.repo_root, args.v1_root, args.v2_data_root
        )
        print("Wrote K&K v2 preparation record")


def command_verify_prep(args):
    verify_prep(args.prep_file, args.repo_root, args.v1_root, args.v2_data_root)
    print("K&K v2 preparation audit passed")


def command_write_authorization(args):
    if args.ack_max_cost_usd != MAX_COST_USD:
        raise ValueError(f"Exact v2 cost acknowledgement must be {MAX_COST_USD}")
    prep = verify_prep(
        args.prep_file, args.repo_root, args.v1_root, args.v2_data_root
    )
    if os.path.lexists(args.output_file):
        raise ValueError("K&K v2 authorization already exists")
    record = sealed(
        {
            "schema_version": 1,
            "record_type": "kk_reasoning_confirmation_v2_authorization",
            "authorized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "evaluation_commit": prep["evaluation_commit"],
            "training_commit": prep["training_commit"],
            "training_config_sha256": prep["training_config_sha256"],
            "base_model": prep["base_model"],
            "base_model_revision": prep["base_model_revision"],
            "prep_file_sha256": sha256_file(args.prep_file),
            "v2_data_manifest_sha256": prep["v2_data_manifest_sha256"],
            "v1_model_manifest_sha256": prep["v1_model_manifest_sha256"],
            "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
            "checkpoint_adapter_base_model": prep[
                "checkpoint_adapter_base_model"
            ],
            "checkpoint_adapter_revision": prep[
                "checkpoint_adapter_revision"
            ],
            "h200_usd_per_hour": H200_USD_PER_HOUR,
            "v2_max_cost_usd": MAX_COST_USD,
            "v2_max_h200_minutes": V2_MAX_MINUTES,
            "v1_released_max_h200_minutes": V1_RELEASED_MAX_MINUTES,
            "cumulative_released_max_h200_minutes": (
                CUMULATIVE_RELEASED_MAX_MINUTES
            ),
            "immutable_cumulative_ceiling_h200_minutes": (
                IMMUTABLE_CUMULATIVE_CEILING_MINUTES
            ),
            "remaining_unsubmitted_reserve_h200_minutes": 60,
            "no_requeue": True,
            "slurm_array": False,
            "training_or_extra_adapters": False,
            "automatic_medical_union_or_quorum": False,
        }
    )
    atomic_write_json(args.output_file, record)
    print("Wrote K&K v2 evaluation-only authorization")


def read_job(path):
    with open(require_file(path), newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or rows[0].get("stage") != "evaluate_v2":
        raise ValueError("V2 jobs.tsv must contain exactly evaluate_v2")
    row = rows[0]
    if (
        re.fullmatch(r"[0-9]+", str(row.get("job_id"))) is None
        or row.get("max_minutes") != str(V2_MAX_MINUTES)
        or row.get("released") != "true"
    ):
        raise ValueError("V2 jobs.tsv contains invalid allocation fields")
    return row


def verify_authorization(auth_file, prep_file, repo_root, v1_root, v2_data_root):
    auth = load_json(auth_file)
    verify_seal(auth, auth_file)
    prep = verify_prep(prep_file, repo_root, v1_root, v2_data_root)
    expected = {
        "record_type": "kk_reasoning_confirmation_v2_authorization",
        "evaluation_commit": prep["evaluation_commit"],
        "training_commit": prep["training_commit"],
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "prep_file_sha256": sha256_file(prep_file),
        "v2_data_manifest_sha256": prep["v2_data_manifest_sha256"],
        "v1_model_manifest_sha256": prep["v1_model_manifest_sha256"],
        "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
        "checkpoint_adapter_base_model": BASE_MODEL,
        "checkpoint_adapter_revision": prep["checkpoint_adapter_revision"],
        "h200_usd_per_hour": H200_USD_PER_HOUR,
        "v2_max_cost_usd": MAX_COST_USD,
        "v2_max_h200_minutes": V2_MAX_MINUTES,
        "v1_released_max_h200_minutes": V1_RELEASED_MAX_MINUTES,
        "cumulative_released_max_h200_minutes": CUMULATIVE_RELEASED_MAX_MINUTES,
        "immutable_cumulative_ceiling_h200_minutes": (
            IMMUTABLE_CUMULATIVE_CEILING_MINUTES
        ),
        "remaining_unsubmitted_reserve_h200_minutes": 60,
        "no_requeue": True,
        "slurm_array": False,
        "training_or_extra_adapters": False,
        "automatic_medical_union_or_quorum": False,
    }
    for key, value in expected.items():
        if auth.get(key) != value:
            raise ValueError(f"V2 authorization mismatch for {key}")
    return auth


def parse_time_limit(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    if not match:
        raise ValueError(f"Unsupported Slurm time limit: {value}")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def command_verify_job(args):
    verify_authorization(
        args.auth_file, args.prep_file, args.repo_root, args.v1_root,
        args.v2_data_root,
    )
    job = read_job(args.jobs_file)
    if job["job_id"] != str(args.job_id):
        raise ValueError("Running v2 job ID differs from jobs.tsv")
    if parse_time_limit(args.time_limit) != V2_MAX_MINUTES:
        raise ValueError("Running v2 job exceeds the 30-minute cap")
    print(f"K&K v2 job {args.job_id} authorization audit passed")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser):
        subparser.add_argument("--repo-root", required=True)
        subparser.add_argument("--v1-root", required=True)
        subparser.add_argument("--v2-data-root", required=True)

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
