#!/usr/bin/env python3
"""Fail-closed provenance, scheduler, and artifact audit for union Wave 2.

Wave 2 creates only the two preregistered good-medical replicas (B2/B3), then
evaluates the complete A/B1/B2/B3 panel directly.  This module deliberately
does not implement a composition sampler and never authorizes Wave 3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if os.fspath(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT_DIR))

import audit_massive_medical_union_tillicum_workflow as wave1  # noqa: E402
import audit_massive_medical_union_medical_recovery_v2 as recovery_v2  # noqa: E402
import judge_massive_union_medical as medical_judge  # noqa: E402
import summarize_massive_union_components as components  # noqa: E402
import audit_massive_medical_union_wave3_protocol as wave3_protocol  # noqa: E402


TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
REPO_ROOT = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-wave2"
RECOVERY_V2_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-medical-recovery-v2"
ENV_ROOT = TILLICUM_ROOT / "envs/subliminal-mitigate-py311"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_pilot_v1"
CONTROL_ROOT = OUTPUT_ROOT / "control/wave2"
DATA_ROOT = OUTPUT_ROOT / "data"
MODEL_ROOT = OUTPUT_ROOT / "models"
EVAL_ROOT = OUTPUT_ROOT / "evaluation/wave2"
GENERATION_ROOT = EVAL_ROOT / "massive/generations"
SCORE_ROOT = EVAL_ROOT / "massive/scores"
MEDICAL_ROOT = EVAL_ROOT / "medical"
PREP_FILE = CONTROL_ROOT / "PREP.json"
JOBS_FILE = CONTROL_ROOT / "jobs.tsv"
AUTH_FILE = CONTROL_ROOT / "AUTHORIZED_MAX_COST_USD_1.125.json"
GPU_MANIFEST = EVAL_ROOT / "GPU_EVAL_MANIFEST.json"
WAVE3_PROTOCOL_ROOT = OUTPUT_ROOT / "protocol/wave3_composition_v1"

LOCAL_MODEL_SNAPSHOT = (
    TILLICUM_ROOT
    / "cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    / wave1.BASE_REVISION
)
MASSIVE_DATA_ROOT = TILLICUM_ROOT / "outputs/massive_benefit_pilot_v1/data"
BENEFIT_MODEL_ROOT = (
    TILLICUM_ROOT
    / "outputs/massive_benefit_pilot_v1/model/massive_en_benefit_pilot_infrastructure_recovery_v1"
)
BENEFIT_ADAPTER = BENEFIT_MODEL_ROOT / "checkpoint-30"
BENEFIT_MANIFEST = BENEFIT_MODEL_ROOT / "MODEL_MANIFEST.json"
BENEFIT_SELECTION = (
    TILLICUM_ROOT
    / "outputs/massive_benefit_pilot_v1/evaluation/evaluation_recovery_v1/selection/summary.json"
)

WAVE1_CONTROL = OUTPUT_ROOT / "control/medical_recovery_v2"
WAVE1_EVAL = OUTPUT_ROOT / "evaluation/wave1/medical_recovery_v2"
WAVE1_GPU_MANIFEST = WAVE1_EVAL / "GPU_MEDICAL_RECOVERY_MANIFEST.json"
WAVE1_JUDGMENTS = WAVE1_EVAL / "medical/judgments_external.json"
WAVE1_SUMMARY = WAVE1_EVAL / "component_gate/summary.json"
WAVE1_GO = WAVE1_CONTROL / "GO_MASSIVE_UNION_WAVE1"

PARENT_COMMIT = "b704a00918bc7a9ddabc512795de4f81d3934c1a"
INITIAL_WAVE2_COMMIT = "0d46633a224ef9e683117ceab33d846f88602814"
WAVE1_JUDGMENTS_SHA256 = "359a8e2351c855bceaea8400cb97a32f62a82f64f7b13b09839a120746a94ca2"
WAVE1_SUMMARY_PAYLOAD_SHA256 = "bf466ec37d6d8d32be2d6a075ec164aace5fe485378265be42b71af976986749"

FROZEN_SHA256 = {
    "train_sft.py": "c0fc3a09a131f481357b64b9a9d72bbfc86a660f635036da44f79ada96092c1b",
    "scripts/train_single_sft.py": "53853928ee4182a4c8902172c7fea69f96c673e5d0ba2b4dfab276973206dac3",
    "scripts/prepare_massive_medical_union_pilot_data.py": "18fb55b98464e3c078e104a70b1a5e1988d7aa380c3a041c0aab84a8c2b8cb69",
    "scripts/sample_massive_structured_generations.py": "dbbf1712586199b843c6a723a26c731c40fb05067028b5713384918a5365eeed",
    "scripts/evaluate_massive_benefit_generations.py": "afea2b7ae95121d15267846ff8d43daf9ce7a9feade27b5c3d44c5ca4eef03a5",
    "scripts/sample_massive_union_medical_direct.py": "0ab7c65a0807c8b6e89043f6809e1e9960c7426f14412518414a3ff59cc5b4ba",
    "scripts/judge_massive_union_medical.py": "f18c76c75d0c6c0021ff5dfad90205668e3695845b8f8a6e63fc38dfcfa9b314",
    "scripts/summarize_massive_union_components.py": "a83ba70502d284e9a805aeab76dfd23b0cea94d94d8ed2557574cb8b82993b29",
    "scripts/audit_massive_medical_union_tillicum_workflow.py": "a65edf5800a363c0f09643b631ae9a24f34e92ca6d74ac4e71d8a973579bed9a",
    "scripts/audit_massive_medical_union_medical_recovery_v2.py": "75ce0a1178455fc4253ad990828f05c377d557826bd0f474e1d74b85f8a0e0ae",
    "configs/training_qwen25_7b_massive_medical_union_pilot.yaml": "4dc9e8ac937bff92b1116d936b19bf907fedc027b12433b7070271647c0af8b5",
    "configs/training_qwen25_7b_massive_medical_union_B2.yaml": "bf3b5fa7249ea69f0e4e4030145885caaf144906421a8d3ada96bb9c828d2c87",
    "configs/training_qwen25_7b_massive_medical_union_B3.yaml": "cab015976876b7382fb0861450621ff86941a2cfd7d9eb8d66567f6571be658e",
}

WAVE2_FILES = (
    "scripts/audit_massive_medical_union_wave2.py",
    "scripts/merge_massive_union_wave2_medical_judgments.py",
    "scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch",
    "scripts/sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch",
    "scripts/stage_massive_medical_union_wave2_tillicum.sh",
    "scripts/submit_massive_medical_union_wave2_tillicum.sh",
    "scripts/status_massive_medical_union_wave2_tillicum.sh",
    "scripts/finalize_massive_medical_union_wave2_tillicum.sh",
    "tests/test_massive_medical_union_wave2.py",
    "docs/massive_medical_union_wave2_protocol.md",
)
COMPOSITION_PREREG_FILES = (
    "docs/massive_medical_union_wave3_composition_protocol.md",
    "scripts/prepare_massive_medical_union_wave3_protocol.py",
    "scripts/audit_massive_medical_union_wave3_protocol.py",
    "tests/test_massive_medical_union_wave3_protocol.py",
)
SUBSET_REPAIR_FILES = (
    "scripts/audit_massive_medical_union_wave2.py",
    "tests/test_massive_medical_union_wave2.py",
    *COMPOSITION_PREREG_FILES,
)
COMPOSITION_PROTOCOL = "massive_medical_union_wave3_composition_v1"
COMPOSITION_METHODS = (
    "ordinary_quorum_m4_q3",
    "ordinary_min_m4_q4",
    "delta_min_m4_q4",
)

STAGE_MINUTES = {"train_B2": 30, "train_B3": 30, "evaluate": 15}
STAGE_ORDER = ("train_B2", "train_B3", "evaluate")
JOB_NAMES = {"train_B2": "mmu_w2_B2", "train_B3": "mmu_w2_B3", "evaluate": "mmu_w2_eval"}
SBATCH_FILES = {
    "train_B2": REPO_ROOT / "scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch",
    "train_B3": REPO_ROOT / "scripts/sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch",
    "evaluate": REPO_ROOT / "scripts/sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch",
}
MEMORY_GB = {"train_B2": 200, "train_B3": 200, "evaluate": 180}
LOG_STEMS = {"train_B2": "train", "train_B3": "train", "evaluate": "evaluate"}
MAX_H200_MINUTES = 75
MAX_GPU_COST_USD = 1.125
H200_RATE_USD_PER_HOUR = 0.90

ARM_CONFIG = {
    "pi_B2": "configs/training_qwen25_7b_massive_medical_union_B2.yaml",
    "pi_B3": "configs/training_qwen25_7b_massive_medical_union_B3.yaml",
}
ARM_PREP_CONFIG_KEY = {"pi_B2": "B2", "pi_B3": "B3"}
ARM_SEED = {"pi_B2": 8182127, "pi_B3": 8182228}


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(body):
    payload = dict(body)
    payload["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return payload


def verify_seal(payload, context):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} is not an object")
    observed = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{context} seal mismatch")
    return body


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_once(path, content):
    path = Path(path)
    if os.path.lexists(path):
        raise ValueError(f"refusing existing output path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_or_audit(path, body):
    expected = seal(body)
    path = Path(path)
    if path.is_file():
        observed = load_json(path)
        verify_seal(observed, path)
        stable = dict(expected)
        stable["created_at"] = observed.get("created_at")
        stable = seal({key: value for key, value in stable.items() if key != "payload_sha256"})
        if observed != stable:
            raise ValueError(f"existing sealed artifact differs: {path}")
        return observed
    if os.path.lexists(path):
        raise ValueError(f"unsafe output path exists: {path}")
    encoded = json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    atomic_write_once(path, encoded)
    return expected


def git(repo, *args):
    return subprocess.check_output(["git", "-C", os.fspath(repo), *args], text=True).strip()


def require_regular_hash(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"required regular artifact missing: {path}")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(f"artifact hash differs: {path}")
    return observed


def audit_repository():
    if REPO_ROOT.resolve() == RECOVERY_V2_REPO.resolve():
        raise ValueError("Wave 2 requires an isolated checkout")
    commit = git(REPO_ROOT, "rev-parse", "HEAD")
    if git(REPO_ROOT, "status", "--porcelain"):
        raise ValueError("Wave-2 checkout is dirty")
    parents = git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, INITIAL_WAVE2_COMMIT]:
        raise ValueError("Wave-2 repair is not a direct nonmerge child of the initial Wave-2 commit")
    initial_parents = git(
        REPO_ROOT, "rev-list", "--parents", "-n", "1", INITIAL_WAVE2_COMMIT
    ).split()
    if initial_parents != [INITIAL_WAVE2_COMMIT, PARENT_COMMIT]:
        raise ValueError("Initial Wave-2 commit is not a direct nonmerge child of b704a00")
    initial_raw_diff = git(
        REPO_ROOT, "diff", "--name-status", "--no-renames",
        f"{PARENT_COMMIT}..{INITIAL_WAVE2_COMMIT}",
    )
    initial_diff = set()
    for line in initial_raw_diff.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("invalid initial Wave-2 name-status record")
        initial_diff.add(tuple(fields))
    expected_diff = {("A", path) for path in (*WAVE2_FILES, *COMPOSITION_PREREG_FILES)}
    if initial_diff != expected_diff or len(initial_raw_diff.splitlines()) != len(expected_diff):
        raise ValueError("Initial Wave-2 commit differs from the exact new-file allowlist")
    repair_raw_diff = git(
        REPO_ROOT, "diff", "--name-status", "--no-renames",
        f"{INITIAL_WAVE2_COMMIT}..{commit}",
    )
    repair_diff = set()
    for line in repair_raw_diff.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("invalid prospective subset-repair name-status record")
        repair_diff.add(tuple(fields))
    expected_repair = {("M", path) for path in SUBSET_REPAIR_FILES}
    if repair_diff != expected_repair or len(repair_raw_diff.splitlines()) != len(expected_repair):
        raise ValueError("Prospective subset repair differs from its exact modification allowlist")
    aggregate_raw_diff = git(
        REPO_ROOT, "diff", "--name-status", "--no-renames",
        f"{PARENT_COMMIT}..{commit}",
    )
    aggregate_diff = {tuple(line.split("\t")) for line in aggregate_raw_diff.splitlines()}
    if aggregate_diff != expected_diff or len(aggregate_raw_diff.splitlines()) != len(expected_diff):
        raise ValueError("Repaired Wave-2 tree differs from the exact aggregate allowlist")
    for relative, expected in FROZEN_SHA256.items():
        require_regular_hash(REPO_ROOT / relative, expected)
        subprocess.run(
            ["git", "-C", os.fspath(REPO_ROOT), "diff", "--quiet", PARENT_COMMIT, "--", relative],
            check=True,
        )
    workflow = {relative: require_regular_hash(REPO_ROOT / relative) for relative in WAVE2_FILES}
    prereg = {relative: require_regular_hash(REPO_ROOT / relative) for relative in COMPOSITION_PREREG_FILES}
    prereg_text = (REPO_ROOT / COMPOSITION_PREREG_FILES[0]).read_text(encoding="utf-8")
    if COMPOSITION_PROTOCOL not in prereg_text or any(method not in prereg_text for method in COMPOSITION_METHODS):
        raise ValueError("composition preregistration omits the frozen protocol/method set")
    return {
        "repo_root": os.fspath(REPO_ROOT),
        "repo_commit": commit,
        "parent_commit": PARENT_COMMIT,
        "initial_wave2_commit": INITIAL_WAVE2_COMMIT,
        "prospective_subset_repair_commit": commit,
        "frozen_parent_science_sha256": dict(FROZEN_SHA256),
        "wave2_workflow_sha256": workflow,
        "composition_preregistration": {
            "protocol": COMPOSITION_PROTOCOL,
            "schema_version": 1,
            "subset_contract_revision": 2,
            "ordered_methods": list(COMPOSITION_METHODS),
            "file_sha256": prereg,
            "prospective_subset_repair": True,
            "repair_modified_files": list(SUBSET_REPAIR_FILES),
            "wave3_automatically_released": False,
        },
    }


def any_seal(payload, context):
    for key in ("payload_sha256", "manifest_payload_sha256", "decision_payload_sha256", "result_payload_sha256"):
        if key in payload:
            body = {name: value for name, value in payload.items() if name != key}
            if payload[key] != sha256_bytes(canonical_bytes(body)):
                raise ValueError(f"{context} seal mismatch")
            return body, key, payload[key]
    raise ValueError(f"{context} lacks a recognized seal")


def audit_model_binding(name, path):
    path = Path(path)
    payload = load_json(path)
    body, _, payload_hash = any_seal(payload, path)
    if body.get("model_name") != name:
        raise ValueError(f"model manifest name differs for {name}")
    fingerprint = body.get("adapter_fingerprint")
    if re.fullmatch(r"[0-9a-f]{64}", str(fingerprint)) is None:
        raise ValueError(f"model fingerprint missing for {name}")
    model_dir = MODEL_ROOT / name
    artifacts = wave1.adapter_artifacts(model_dir)
    if fingerprint != sha256_bytes(canonical_bytes(artifacts)):
        raise ValueError(f"live adapter bytes differ for {name}")
    if body.get("adapter_artifacts") != artifacts:
        raise ValueError(f"manifest adapter inventory differs for {name}")
    return {
        "path": os.fspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload_hash,
        "adapter_fingerprint": fingerprint,
        "training_config_sha256": body.get("training_config_sha256"),
        "dataset_fingerprint": body.get("dataset_fingerprint"),
        "dataset_logical_sha256": body.get("dataset_logical_sha256"),
        "seed": body.get("seed"),
    }


def audit_realized_composition_protocol():
    result = wave3_protocol.audit_protocol(
        WAVE3_PROTOCOL_ROOT, MASSIVE_DATA_ROOT, DATA_ROOT
    )
    if (
        result.get("protocol_id") != COMPOSITION_PROTOCOL
        or result.get("subset_contract_revision") != 2
        or result.get("method_ids") != list(COMPOSITION_METHODS)
        or result.get("smoke_rows") != 60
        or result.get("confirmation_rows") != 600
        or result.get("medical_samples_per_method") != 80
        or result.get("wave3_released") is not False
    ):
        raise ValueError("realized Wave-3 protocol differs from preregistration")
    manifest = WAVE3_PROTOCOL_ROOT / "protocol_manifest.json"
    return {
        **result,
        "manifest_path": os.fspath(manifest),
        "manifest_file_sha256": sha256_file(manifest),
        "wave3_submitted_or_released": False,
    }


def audit_wave1_go():
    # Re-run the incident-bound GPU audit.  It binds both historical failed
    # attempts and the successful 240/240 recovery without relying on prose.
    gpu = recovery_v2.v1.audit_gpu()
    if gpu.get("repo_commit") != PARENT_COMMIT or gpu.get("all_240_finish_reason_stop") is not True:
        raise ValueError("successful Wave-1 recovery GPU seal differs")
    successful_job_id = str(gpu.get("authorized_job", {}).get("job_id", ""))
    accounting_rows = subprocess.check_output(
        [
            "sacct", "-n", "-X", "-P", "-j", successful_job_id,
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode",
        ],
        text=True,
    ).strip().splitlines()
    matching = [line.split("|") for line in accounting_rows if line.split("|", 1)[0] == successful_job_id]
    if len(matching) != 1 or len(matching[0]) < 8:
        raise ValueError("successful Wave-1 recovery lacks unique durable accounting")
    job_id, job_name, state, elapsed, time_limit, allocated, requested, exit_code = matching[0][:8]
    expected_recovery_tres = {
        "billing": "8", "cpu": "8", "gres/gpu:h200": "1",
        "gres/gpu": "1", "mem": "180G", "node": "1",
    }
    if (
        job_id != "248318" or job_name != "mmu_medrec_v2" or state != "COMPLETED"
        or elapsed != "00:01:23" or time_limit != "00:10:00" or exit_code != "0:0"
        or recovery_v2.v1.parse_tres(allocated) != expected_recovery_tres
        or recovery_v2.v1.parse_tres(requested) != expected_recovery_tres
    ):
        raise ValueError("successful Wave-1 recovery durable accounting differs")
    if sha256_file(WAVE1_JUDGMENTS) != WAVE1_JUDGMENTS_SHA256:
        raise ValueError("Wave-1 external judgments raw bytes differ")
    judged = components.load_medical(WAVE1_JUDGMENTS)
    if (
        set(judged["by_model"]) != {"pi_base", "pi_A", "pi_B1"}
        or judged["meta"].get("judge_kind") != "external_gpt_primary"
        or judged["meta"].get("primary_confirmatory") is not True
        or judged["meta"].get("actual_api_calls") != 240
        or judged["meta"].get("max_api_calls") != 240
        or judged["meta"].get("max_cost_usd") != 0.75
    ):
        raise ValueError("Wave-1 external judge contract differs")

    summary_payload = load_json(WAVE1_SUMMARY)
    summary_body = components.audit_seal(summary_payload, WAVE1_SUMMARY)
    if summary_payload.get("payload_sha256") != WAVE1_SUMMARY_PAYLOAD_SHA256:
        raise ValueError("Wave-1 component summary payload differs")
    if (
        summary_body.get("protocol") != "massive_medical_union_component_gate_v1"
        or summary_body.get("phase") != "wave1"
        or summary_body.get("status") != "GO"
        or summary_body.get("wave2_release_authorized") is not True
        or summary_body.get("primary_confirmatory_medical_judge") is not True
        or set(summary_body.get("candidates", {})) != {"pi_A", "pi_B1"}
        or len(summary_body.get("checks", {})) != 34
        or not all(summary_body.get("checks", {}).values())
    ):
        raise ValueError("Wave-1 component GO contract differs")
    judge_binding = summary_body.get("medical_judge") or {}
    if (
        judge_binding.get("file_sha256") != WAVE1_JUDGMENTS_SHA256
        or judge_binding.get("payload_sha256") != judged["payload_sha256"]
    ):
        raise ValueError("Wave-1 summary does not bind the external judgments")

    go_payload = load_json(WAVE1_GO)
    go_body = components.audit_seal(go_payload, WAVE1_GO)
    if (
        go_body.get("protocol") != "massive_medical_union_component_sentinel_v1"
        or go_body.get("phase") != "wave1"
        or go_body.get("status") != "GO"
        or go_body.get("wave2_release_authorized") is not True
        or go_body.get("summary_sha256") != sha256_file(WAVE1_SUMMARY)
        or go_body.get("summary_payload_sha256") != summary_payload["payload_sha256"]
    ):
        raise ValueError("Wave-1 GO sentinel differs")
    models = {
        name: audit_model_binding(name, MODEL_ROOT / name / "MODEL_MANIFEST.json")
        for name in ("pi_A", "pi_B1")
    }
    for name in models:
        expected = summary_body["candidates"][name]["model_manifest"]
        if expected.get("file_sha256") != models[name]["file_sha256"]:
            raise ValueError(f"Wave-1 summary model binding differs for {name}")
    return {
        "recovery_v2_repo_commit": git(RECOVERY_V2_REPO, "rev-parse", "HEAD"),
        "gpu_manifest": {
            "path": os.fspath(WAVE1_GPU_MANIFEST),
            "file_sha256": sha256_file(WAVE1_GPU_MANIFEST),
            "payload_sha256": gpu["payload_sha256"],
            "job_id": str(gpu["authorized_job"]["job_id"]),
            "terminal_sacct_row": "|".join(matching[0]),
        },
        "judgments": {
            "path": os.fspath(WAVE1_JUDGMENTS),
            "file_sha256": WAVE1_JUDGMENTS_SHA256,
            "payload_sha256": judged["payload_sha256"],
            "actual_api_calls": 240,
            "maximum_api_cost_usd": 0.75,
        },
        "summary": {
            "path": os.fspath(WAVE1_SUMMARY),
            "file_sha256": sha256_file(WAVE1_SUMMARY),
            "payload_sha256": summary_payload["payload_sha256"],
            "all_34_checks_true": True,
        },
        "go": {
            "path": os.fspath(WAVE1_GO),
            "file_sha256": sha256_file(WAVE1_GO),
            "payload_sha256": go_payload["payload_sha256"],
        },
        "models": models,
    }


def audit_massive_test():
    manifest_path = MASSIVE_DATA_ROOT / "data_manifest.json"
    payload = load_json(manifest_path)
    body, seal_key, seal_value = any_seal(payload, manifest_path)
    inventory_rows = body.get("file_inventory")
    if not isinstance(inventory_rows, list):
        raise ValueError("MASSIVE data manifest lacks a list file inventory")
    inventory = {entry.get("path"): entry for entry in inventory_rows if isinstance(entry, dict)}
    if len(inventory) != len(inventory_rows):
        raise ValueError("MASSIVE data manifest inventory has duplicate/invalid paths")
    expected = {
        "sealed_test/prompts.json": MASSIVE_DATA_ROOT / "sealed_test/prompts.json",
        "sealed_test/answers.json": MASSIVE_DATA_ROOT / "sealed_test/answers.json",
    }
    result = {}
    for relative, path in expected.items():
        entry = inventory.get(relative)
        if not isinstance(entry, dict) or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"MASSIVE cleaned-test artifact is not manifest-bound: {relative}")
        result[relative] = {"path": os.fspath(path), "sha256": entry["sha256"]}
    answers = load_json(expected["sealed_test/answers.json"])
    rows = answers.get("answers") if isinstance(answers, dict) else None
    meta = answers.get("meta") if isinstance(answers, dict) else None
    if not isinstance(rows, list) or len(rows) != 2965 or not isinstance(meta, dict) or meta.get("role") != "sealed_final":
        raise ValueError("MASSIVE cleaned-test set is not exact sealed_final n=2965")
    return {
        "manifest_path": os.fspath(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_seal_key": seal_key,
        "manifest_payload_sha256": seal_value,
        "set_name": meta.get("set_name"),
        "role": "sealed_final",
        "n": 2965,
        "artifacts": result,
    }


def prep_body(prepared_snapshot=None):
    repository = audit_repository()
    configs = wave1.audit_all_configs(REPO_ROOT)
    data = wave1.audit_data_manifest(DATA_ROOT)
    snapshot = prepared_snapshot or wave1.validate_snapshot_path(LOCAL_MODEL_SNAPSHOT)
    benefit = wave1.audit_benefit_control(BENEFIT_MANIFEST, BENEFIT_ADAPTER)
    runtime = wave1.audit_runtime_versions()
    return {
        "schema_version": 1,
        "protocol": "massive_medical_union_wave2_components_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": repository,
        "runtime_versions": runtime,
        "configs": configs,
        "union_data_manifest": data,
        "local_model_snapshot": snapshot,
        "benefit_control": benefit,
        "massive_cleaned_test": audit_massive_test(),
        "wave1_prerequisite": audit_wave1_go(),
        "realized_composition_preregistration": audit_realized_composition_protocol(),
        "training": {
            "models": ["pi_B2", "pi_B3"],
            "datasets": {"pi_B2": "train/B_massive_good_medical", "pi_B3": "train/B_massive_good_medical"},
            "seeds": dict(ARM_SEED),
            "presentations_per_model": 32367,
            "optimizer_steps": 540,
            "sole_scientific_checkpoint": 540,
            "failed_replica_replacement_forbidden": True,
            "fresh_adapter_from_pinned_base": True,
        },
        "evaluation": {
            "fresh_massive_models": ["pi_base", "pi_M", "pi_A", "pi_B1", "pi_B2", "pi_B3"],
            "massive_role": "sealed_final",
            "massive_n": 2965,
            "seed": 8172026,
            "max_new_tokens": 256,
            "max_context": 2048,
            "structured_constraint_profile": "const_tree_no_ws_v3",
            "xgrammar_any_whitespace": False,
            "new_medical_models": ["pi_B2", "pi_B3"],
            "medical_profile": "official16_max1024_all_stop_v2",
            "medical_rows_per_model": 80,
            "new_external_judge_calls": 160,
            "new_external_judge_max_cost_usd": 0.50,
        },
        "budget": {
            "stage_minutes": dict(STAGE_MINUTES),
            "maximum_h200_minutes": MAX_H200_MINUTES,
            "h200_rate_usd_per_hour": H200_RATE_USD_PER_HOUR,
            "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
            "prior_released_gpu_ceiling_h200_minutes": 100,
            "cumulative_released_gpu_ceiling_h200_minutes": 175,
            "cumulative_released_gpu_ceiling_usd": 2.625,
            "prior_external_judge_ceiling_usd": 0.75,
            "new_external_judge_ceiling_usd": 0.50,
            "cumulative_external_judge_ceiling_usd": 1.25,
            "all_in_released_ceiling_usd": 3.875,
            "no_retry_or_reserve": True,
        },
        "wave3_submitted_or_released": False,
    }


def command_write_prep():
    if os.path.lexists(CONTROL_ROOT) or os.path.lexists(EVAL_ROOT):
        raise ValueError("Wave-2 output namespace already exists")
    for model in ("pi_B2", "pi_B3"):
        if os.path.lexists(MODEL_ROOT / model):
            raise ValueError(f"Wave-2 model namespace already exists: {model}")
    payload = write_or_audit(PREP_FILE, prep_body())
    print(PREP_FILE)
    return payload


def audit_prep():
    observed = load_json(PREP_FILE)
    body = verify_seal(observed, PREP_FILE)
    expected = prep_body(prepared_snapshot=body.get("local_model_snapshot"))
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError("Wave-2 PREP differs from current immutable inputs")
    return observed


def command_verify_snapshot():
    prep = audit_prep()
    observed = wave1.validate_snapshot_path(LOCAL_MODEL_SNAPSHOT)
    if observed != prep["local_model_snapshot"]:
        raise ValueError("pinned base snapshot bytes differ from Wave-2 PREP")
    print("Pinned Wave-2 base snapshot verified: " + observed["snapshot_binding_sha256"])


def jobs_bytes(rows):
    content = ["stage\tjob_id\tmax_minutes\treleased"]
    for stage in STAGE_ORDER:
        row = rows[stage]
        content.append(f"{stage}\t{row['job_id']}\t{STAGE_MINUTES[stage]}\ttrue")
    return ("\n".join(content) + "\n").encode()


def parse_jobs(path=JOBS_FILE):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "stage\tjob_id\tmax_minutes\treleased":
        raise ValueError("Wave-2 jobs header differs")
    rows = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError("Wave-2 jobs row width differs")
        stage, job_id, minutes, released = fields
        if stage not in STAGE_MINUTES or stage in rows or not job_id.isdigit():
            raise ValueError("Wave-2 jobs stage/job differs")
        if int(minutes) != STAGE_MINUTES[stage] or released != "true":
            raise ValueError("Wave-2 jobs cap/release differs")
        rows[stage] = {"stage": stage, "job_id": job_id, "max_minutes": int(minutes), "released": True}
    if list(rows) != list(STAGE_ORDER) or Path(path).read_bytes() != jobs_bytes(rows):
        raise ValueError("Wave-2 jobs order/bytes differ")
    if len({row["job_id"] for row in rows.values()}) != 3:
        raise ValueError("Wave-2 jobs repeat an ID")
    return rows


def dependency_ids(value):
    value = (value or "").replace("(unfulfilled)", "").replace("(fulfilled)", "")
    if value in {"", "(null)"}:
        return []
    if value.startswith("afterok:") and "," not in value:
        terms = value[len("afterok:"):].split(":")
    else:
        terms = []
        for item in value.split(","):
            match = re.fullmatch(r"afterok:([0-9]+)", item)
            if match is None:
                raise ValueError("Wave-2 dependency syntax differs")
            terms.append(match.group(1))
    if not terms or any(not term.isdigit() for term in terms):
        raise ValueError("Wave-2 dependency IDs differ")
    return terms


def expected_tres(stage):
    return {
        "billing": "8", "cpu": "8", "gres/gpu:h200": "1",
        "gres/gpu": "1", "mem": f"{MEMORY_GB[stage]}G", "node": "1",
    }


def query_job(job_id):
    raw = subprocess.check_output(["scontrol", "show", "job", str(job_id), "-o"], text=True).strip()
    return raw, recovery_v2.v1.parse_scontrol_line(raw)


def audit_job_record(
    stage, job_id, raw, fields, phase, expected_dependencies,
    check_held_log_absence=True,
):
    if phase not in {"held", "running"}:
        raise ValueError("invalid Wave-2 scheduler phase")
    log_stem = LOG_STEMS[stage]
    exact = {
        "JobId": str(job_id), "JobName": JOB_NAMES[stage], "Account": "stf", "QOS": "normal",
        "Requeue": "0", "Restarts": "0", "Partition": "gpu-h200", "NumTasks": "1",
        "NumCPUs": "8", "CPUs/Task": "8", "TimeLimit": f"00:{STAGE_MINUTES[stage]:02d}:00",
        "Command": os.fspath(SBATCH_FILES[stage]), "WorkDir": os.fspath(REPO_ROOT),
        "StdOut": os.fspath(TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_{log_stem}_{job_id}.out"),
        "StdErr": os.fspath(TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_{log_stem}_{job_id}.err"),
        "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            raise ValueError(f"Wave-2 {stage} job differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError("Wave-2 job is not exactly one node")
    if recovery_v2.v1.parse_tres(fields.get("ReqTRES", "")) != expected_tres(stage):
        raise ValueError("Wave-2 requested TRES differs")
    observed_dependencies = dependency_ids(fields.get("Dependency"))
    allowed_dependencies = [list(expected_dependencies)]
    if phase == "running" and stage == "evaluate":
        allowed_dependencies.append([])
    if observed_dependencies not in allowed_dependencies:
        raise ValueError("Wave-2 dependency IDs/order differ")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError("Wave-2 job unexpectedly belongs to an array/het job")
    if stage == "evaluate" and fields.get("KillOnInvalidDependent") != "Yes":
        raise ValueError("Wave-2 evaluation lacks kill-on-invalid dependency")
    if stage != "evaluate" and fields.get("KillOnInvalidDependent", "") not in {"", "No"}:
        raise ValueError("Wave-2 training has an unexpected dependency policy")
    if phase == "held":
        if fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser":
            raise ValueError("Wave-2 job was not held before authorization")
        if fields.get("RunTime") != "00:00:00" or fields.get("AllocTRES") != "(null)":
            raise ValueError("held Wave-2 job already used an allocation")
        if fields.get("MinMemoryNode") != f"{MEMORY_GB[stage]}G":
            raise ValueError("held Wave-2 memory request differs")
        submit = (
            f"sbatch --parsable --hold --export=NONE --job-name={JOB_NAMES[stage]} "
        )
        if stage == "evaluate":
            submit += (
                f"--dependency=afterok:{expected_dependencies[0]}:{expected_dependencies[1]} "
                "--kill-on-invalid-dep=yes "
            )
        submit += os.fspath(SBATCH_FILES[stage].relative_to(REPO_ROOT))
        if fields.get("SubmitLine") != submit:
            raise ValueError("held Wave-2 SubmitLine differs")
        if check_held_log_absence:
            for suffix in ("out", "err"):
                path = TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_{log_stem}_{job_id}.{suffix}"
                if os.path.lexists(path):
                    raise ValueError("held Wave-2 job already created a log")
    else:
        if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None":
            raise ValueError("Wave-2 job is not RUNNING at preflight")
        if recovery_v2.v1.parse_tres(fields.get("AllocTRES", "")) != expected_tres(stage):
            raise ValueError("Wave-2 allocation TRES differs")
        node = fields.get("NodeList", "")
        if re.fullmatch(r"g[0-9]+", node) is None or fields.get("BatchHost") != node:
            raise ValueError("Wave-2 node allocation differs")
    return {
        "stage": stage, "job_id": str(job_id), "job_name": JOB_NAMES[stage], "phase": phase,
        "scontrol_record": raw, "scontrol_record_sha256": sha256_bytes(raw.encode()),
        "normalized_nodes": 1, "requested_tres": expected_tres(stage),
        "time_limit": f"00:{STAGE_MINUTES[stage]:02d}:00", "no_requeue": True,
        "dependency_ids": list(expected_dependencies),
    }


def audit_held_job(stage, job_id, expected_dependencies, check_log_absence=True):
    raw, fields = query_job(job_id)
    result = audit_job_record(stage, job_id, raw, fields, "held", expected_dependencies)
    completed = subprocess.run(
        ["scontrol", "write", "batch_script", str(job_id), "-"],
        check=True, capture_output=True,
    )
    spooled = completed.stdout if isinstance(completed.stdout, bytes) else completed.stdout.encode()
    source = SBATCH_FILES[stage].read_bytes()
    if spooled != source:
        raise ValueError(f"spooled Wave-2 {stage} script differs from committed bytes")
    result["spooled_script_sha256"] = sha256_bytes(spooled)
    result["committed_script_sha256"] = sha256_bytes(source)
    return result


def held_audits(rows):
    b2, b3 = rows["train_B2"]["job_id"], rows["train_B3"]["job_id"]
    return {
        "train_B2": audit_held_job("train_B2", b2, []),
        "train_B3": audit_held_job("train_B3", b3, []),
        "evaluate": audit_held_job("evaluate", rows["evaluate"]["job_id"], [b2, b3]),
    }


def auth_body(audits, created_at=None):
    prep = audit_prep()
    rows = parse_jobs()
    return {
        "schema_version": 1,
        "protocol": "massive_medical_union_wave2_authorization_v1",
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": prep["repository"]["repo_commit"],
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "jobs_file_sha256": sha256_file(JOBS_FILE),
        "jobs": [rows[stage] for stage in STAGE_ORDER],
        "held_job_audits": audits,
        "maximum_h200_minutes": MAX_H200_MINUTES,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "no_requeue": True,
        "no_retry_or_reserve": True,
        "released_wave": 2,
        "new_models": ["pi_B2", "pi_B3"],
        "wave3_submitted_or_released": False,
        "external_api_calls": 0,
    }


def command_write_auth():
    rows = parse_jobs()
    payload = write_or_audit(AUTH_FILE, auth_body(held_audits(rows)))
    print(AUTH_FILE)
    return payload


def audit_auth():
    observed = load_json(AUTH_FILE)
    body = verify_seal(observed, AUTH_FILE)
    rows = parse_jobs()
    recorded = body.get("held_job_audits")
    if not isinstance(recorded, dict) or set(recorded) != set(STAGE_ORDER):
        raise ValueError("Wave-2 authorization lacks exact held-job audits")
    b2, b3 = rows["train_B2"]["job_id"], rows["train_B3"]["job_id"]
    deps = {"train_B2": [], "train_B3": [], "evaluate": [b2, b3]}
    normalized = {}
    for stage in STAGE_ORDER:
        audit = recorded[stage]
        raw = audit.get("scontrol_record")
        expected = audit_job_record(
            stage, rows[stage]["job_id"], raw,
            recovery_v2.v1.parse_scontrol_line(raw), "held", deps[stage],
            check_held_log_absence=False,
        )
        script_hash = sha256_file(SBATCH_FILES[stage])
        expected["spooled_script_sha256"] = script_hash
        expected["committed_script_sha256"] = script_hash
        normalized[stage] = expected
    expected = auth_body(normalized, created_at=body.get("created_at"))
    if observed != seal(expected):
        raise ValueError("Wave-2 authorization differs")
    return observed


def command_audit_held():
    auth = audit_auth()
    rows = parse_jobs()
    live = held_audits(rows)
    stable = {
        "stage", "job_id", "job_name", "phase", "normalized_nodes", "requested_tres",
        "time_limit", "no_requeue", "dependency_ids", "spooled_script_sha256", "committed_script_sha256",
    }
    for stage in STAGE_ORDER:
        if {key: live[stage].get(key) for key in stable} != {
            key: auth["held_job_audits"][stage].get(key) for key in stable
        }:
            raise ValueError(f"live held Wave-2 {stage} differs from authorization")
    print("Re-audited all three held Wave-2 jobs")


def command_verify_job(stage, job_id, time_limit):
    auth = audit_auth()
    rows = parse_jobs()
    if rows[stage]["job_id"] != str(job_id) or time_limit != f"00:{STAGE_MINUTES[stage]:02d}:00":
        raise ValueError("running job differs from authorized Wave-2 stage")
    b2, b3 = rows["train_B2"]["job_id"], rows["train_B3"]["job_id"]
    deps = [] if stage != "evaluate" else [b2, b3]
    raw, fields = query_job(job_id)
    audit_job_record(stage, job_id, raw, fields, "running", deps)
    if stage == "evaluate":
        rows_out = subprocess.check_output(
            [
                "sacct", "-n", "-X", "-P", "-j", f"{b2},{b3}",
                "--format=JobIDRaw,State,ExitCode",
            ],
            text=True,
        ).strip().splitlines()
        states = {}
        for line in rows_out:
            fields_out = line.split("|")
            if len(fields_out) >= 3 and fields_out[0] in {b2, b3}:
                states[fields_out[0]] = (fields_out[1], fields_out[2])
        if states != {b2: ("COMPLETED", "0:0"), b3: ("COMPLETED", "0:0")}:
            raise ValueError("Wave-2 evaluation dependencies lack durable successful completion")
    expected_env = {
        "SLURM_JOB_ID": str(job_id), "SLURM_JOB_NAME": JOB_NAMES[stage],
        "SLURM_JOB_PARTITION": "gpu-h200", "SLURM_JOB_ACCOUNT": "stf",
        "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": os.fspath(REPO_ROOT), "SLURM_JOB_NODELIST": fields.get("NodeList"),
    }
    for key, expected in expected_env.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"Wave-2 Slurm environment differs: {key}")
    print(f"Authorized Wave-2 {stage} job {job_id}")


def model_body(model_name, model_dir):
    if model_name not in ARM_CONFIG:
        raise ValueError("Wave 2 can write only B2/B3 model manifests")
    prep = audit_prep()
    model_dir = Path(model_dir).resolve()
    expected_dir = (MODEL_ROOT / model_name).resolve()
    if model_dir != expected_dir:
        raise ValueError("Wave-2 model directory differs")
    # ``wave1.audit_all_configs`` seals the variant entries as ``B2``/``B3``;
    # model names use ``pi_B2``/``pi_B3``.  Keep the schema translation
    # explicit so a future naming change fails closed instead of occurring
    # after an otherwise complete training run.
    config_key = ARM_PREP_CONFIG_KEY[model_name]
    config_entry = prep["configs"][config_key]
    config_hash = config_entry["sha256"]
    if config_hash != FROZEN_SHA256[ARM_CONFIG[model_name]]:
        raise ValueError(f"Wave-2 prepared training config differs for {model_name}")
    run_meta_path = model_dir / "training_run_meta.json"
    summary_path = model_dir / "training_summary.json"
    objective_path = model_dir / "training_objective.json"
    mask_path = model_dir / "loss_mask_audit.json"
    run_meta, training_summary = load_json(run_meta_path), load_json(summary_path)
    seed = ARM_SEED[model_name]
    expected_dataset = (DATA_ROOT / "train/B_massive_good_medical").resolve()
    if (
        run_meta.get("n_examples") != 32367 or run_meta.get("seed") != seed
        or run_meta.get("data_seed") != seed or run_meta.get("max_steps") != 540
        or run_meta.get("loss_on") != "completion"
        or Path(run_meta.get("dataset", "")).resolve() != expected_dataset
    ):
        raise ValueError(f"Wave-2 training metadata differs for {model_name}")
    if (
        training_summary.get("final_global_step") != 540
        or training_summary.get("n_examples") != 32367
        or training_summary.get("loss_on") != "completion"
    ):
        raise ValueError(f"Wave-2 training summary differs for {model_name}")
    wave1.audit_training_snapshot_binding(run_meta.get("base_model_load", {}), prep["local_model_snapshot"])
    manifest = load_json(DATA_ROOT / "data_manifest.json")
    arm = manifest.get("arms", {}).get("B", {})
    if run_meta.get("dataset_fingerprint") != arm.get("dataset_fingerprint"):
        raise ValueError(f"Wave-2 dataset fingerprint differs for {model_name}")
    artifacts = wave1.adapter_artifacts(model_dir)
    inventory = wave1.file_inventory(model_dir, ignored=("MODEL_MANIFEST.json", "TRAIN_COMPLETE"))
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": model_name,
        "seed": seed, "data_seed": seed,
        "base_model": wave1.BASE_MODEL, "base_model_revision": wave1.BASE_REVISION,
        "adapter_dir": os.fspath(model_dir),
        "adapter_artifacts": artifacts,
        "adapter_fingerprint": sha256_bytes(canonical_bytes(artifacts)),
        "training_config_sha256": config_hash,
        "union_data_manifest_sha256": prep["union_data_manifest"]["sha256"],
        "union_data_manifest_payload_sha256": prep["union_data_manifest"]["payload_sha256"],
        "dataset_relative_path": "train/B_massive_good_medical",
        "dataset_fingerprint": run_meta.get("dataset_fingerprint"),
        "dataset_logical_sha256": arm.get("dataset_logical_sha256"),
        "training_run_meta_sha256": sha256_file(run_meta_path),
        "training_summary_sha256": sha256_file(summary_path),
        "training_objective_sha256": sha256_file(objective_path),
        "loss_mask_audit_sha256": sha256_file(mask_path),
        "final_global_step": 540,
        "scientific_checkpoint": 540,
        "repo_commit": prep["repository"]["repo_commit"],
        "fresh_adapter_from_pinned_base": True,
        "replacement_replica": False,
        "inventory": inventory,
    }


def command_write_model(model_name, model_dir, output_file):
    payload = write_or_audit(output_file, model_body(model_name, model_dir))
    print(output_file)
    return payload


def audit_new_model(model_name):
    path = MODEL_ROOT / model_name / "MODEL_MANIFEST.json"
    observed = load_json(path)
    body = verify_seal(observed, path)
    expected = model_body(model_name, MODEL_ROOT / model_name)
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError(f"Wave-2 model manifest differs for {model_name}")
    return {
        "path": os.fspath(path), "file_sha256": sha256_file(path),
        "payload_sha256": observed["payload_sha256"],
        "adapter_fingerprint": observed["adapter_fingerprint"],
        "training_config_sha256": observed["training_config_sha256"],
        "dataset_fingerprint": observed["dataset_fingerprint"],
        "dataset_logical_sha256": observed["dataset_logical_sha256"],
        "seed": observed["seed"],
    }


def audit_all_models():
    prep = audit_prep()
    models = dict(prep["wave1_prerequisite"]["models"])
    for name in ("pi_B2", "pi_B3"):
        models[name] = audit_new_model(name)
    fingerprints = [models[name]["adapter_fingerprint"] for name in ("pi_A", "pi_B1", "pi_B2", "pi_B3")]
    if len(set(fingerprints)) != 4 or wave1.BENEFIT_CONTROL_FINGERPRINT in fingerprints:
        raise ValueError("A/B component adapter fingerprints are not pairwise distinct")
    b_fields = ("dataset_fingerprint", "dataset_logical_sha256")
    for field in b_fields:
        values = {models[name].get(field) for name in ("pi_B1", "pi_B2", "pi_B3")}
        if len(values) != 1 or None in values:
            raise ValueError(f"B replicas do not bind the identical B dataset: {field}")
    if [models[name].get("seed") for name in ("pi_B1", "pi_B2", "pi_B3")] != [8182026, 8182127, 8182228]:
        raise ValueError("B replicas do not bind the preregistered seeds")
    return models


def command_audit_models():
    models = audit_all_models()
    print("Audited Wave-2 component panel: " + ",".join(models))


def audit_prejudge_gate():
    root = EVAL_ROOT / "prejudge_component_gate"
    candidates = [root / "awaiting_external_judge.json", root / "summary.json"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError("Wave-2 MASSIVE prejudge has no unique summary")
    payload = load_json(present[0])
    body = components.audit_seal(payload, present[0])
    if (
        body.get("protocol") != "massive_medical_union_component_gate_v1"
        or body.get("phase") != "all"
        or set(body.get("candidates", {})) != {"pi_A", "pi_B1", "pi_B2", "pi_B3"}
        or any(key.startswith("medical.") for key in body.get("checks", {}))
        or len(body.get("checks", {})) != 48
    ):
        raise ValueError("Wave-2 MASSIVE prejudge contract differs")
    status = body.get("status")
    if status not in {"AWAITING_EXTERNAL_JUDGE", "STOP"}:
        raise ValueError("Wave-2 MASSIVE prejudge status differs")
    if (status == "AWAITING_EXTERNAL_JUDGE") != all(body["checks"].values()):
        raise ValueError("Wave-2 MASSIVE prejudge checks/status disagree")
    sentinel_name = "AWAITING_EXTERNAL_JUDGE" if status == "AWAITING_EXTERNAL_JUDGE" else "STOPPED_MASSIVE_UNION_ALL_REPLICAS"
    sentinel_path = root / sentinel_name
    sentinel_payload = load_json(sentinel_path)
    sentinel = components.audit_seal(sentinel_payload, sentinel_path)
    if (
        sentinel.get("phase") != "all" or sentinel.get("status") != status
        or sentinel.get("summary_sha256") != sha256_file(present[0])
        or sentinel.get("summary_payload_sha256") != payload["payload_sha256"]
    ):
        raise ValueError("Wave-2 MASSIVE prejudge sentinel differs")
    return {
        "status": status,
        "all_48_massive_checks_true": status == "AWAITING_EXTERNAL_JUDGE",
        "summary_path": os.fspath(present[0]),
        "summary_file_sha256": sha256_file(present[0]),
        "summary_payload_sha256": payload["payload_sha256"],
        "sentinel_path": os.fspath(sentinel_path),
        "sentinel_file_sha256": sha256_file(sentinel_path),
        "sentinel_payload_sha256": sentinel_payload["payload_sha256"],
        "new_external_judge_eligible": status == "AWAITING_EXTERNAL_JUDGE",
    }


def audit_massive_scores(models):
    prep = audit_prep()
    test_binding = prep["massive_cleaned_test"]
    expected_answers_sha = test_binding["artifacts"]["sealed_test/answers.json"]["sha256"]
    expected_prompt_sha = test_binding["artifacts"]["sealed_test/prompts.json"]["sha256"]
    expected_fingerprints = {
        "pi_base": "BASE", "pi_M": wave1.BENEFIT_CONTROL_FINGERPRINT,
        **{name: entry["adapter_fingerprint"] for name, entry in models.items()},
    }
    scores = {}
    loaded = {}
    for name in ("pi_base", "pi_M", "pi_A", "pi_B1", "pi_B2", "pi_B3"):
        path = SCORE_ROOT / f"massive_en_test__{name}.json"
        score = components.load_score(path)
        meta = score["meta"]
        joint_path = GENERATION_ROOT / f"massive_en_test__{name}.json"
        intent_path = GENERATION_ROOT / f"massive_en_test__{name}__intent_only.json"
        if (
            meta.get("role") != "sealed_final" or meta.get("set_name") != "massive_en_test"
            or meta.get("model_name") != name
            or meta.get("model_fingerprint") != expected_fingerprints[name]
            or meta.get("base_model") != wave1.BASE_MODEL
            or meta.get("base_model_revision") != wave1.BASE_REVISION
            or meta.get("inference_seed") != 8172026
            or meta.get("max_new_tokens") != 256 or meta.get("max_context") != 2048
            or meta.get("structured_constraint_profile") != "const_tree_no_ws_v3"
            or meta.get("xgrammar_any_whitespace") is not False
            or meta.get("n_samples") != 1 or meta.get("temperature") != 0.0
            or score["metrics"].get("n") != 2965
            or meta.get("answers_file_sha256") != expected_answers_sha
            or meta.get("data_manifest_sha256") != test_binding["manifest_file_sha256"]
            or meta.get("data_manifest_payload_sha256") != test_binding["manifest_payload_sha256"]
            or meta.get("joint_generations_file") != os.fspath(joint_path)
            or meta.get("joint_generations_file_sha256") != sha256_file(joint_path)
            or meta.get("intent_generations_file") != os.fspath(intent_path)
            or meta.get("intent_generations_file_sha256") != sha256_file(intent_path)
        ):
            raise ValueError(f"Wave-2 MASSIVE score provenance differs for {name}")
        for generation_path in (joint_path, intent_path):
            generation = load_json(generation_path)
            run = generation.get("meta") if isinstance(generation, dict) else None
            if not isinstance(run, dict) or run.get("prompt_file_sha256") != expected_prompt_sha:
                raise ValueError(f"Wave-2 generation prompt binding differs for {name}")
        loaded[name] = score
        scores[name] = {"path": os.fspath(path), "file_sha256": score["file_sha256"], "payload_sha256": score["payload_sha256"]}
    for name in loaded:
        components.validate_pair(loaded["pi_base"], loaded[name])
    return scores


def audit_medical_generations(models):
    result, base_rows = {}, {}
    for name, folder in (("pi_B2", "by_B2"), ("pi_B3", "by_B3")):
        directory = MEDICAL_ROOT / "generations" / folder
        expected_names = {"medical_official16_v2__pi_base.json", f"medical_official16_v2__{name}.json"}
        if directory.is_symlink() or not directory.is_dir() or {path.name for path in directory.iterdir()} != expected_names:
            raise ValueError(f"Wave-2 medical generation inventory differs for {name}")
        candidate_path = directory / f"medical_official16_v2__{name}.json"
        loaded = medical_judge.load_generation(name, candidate_path)
        if loaded["model_fingerprint"] != models[name]["adapter_fingerprint"]:
            raise ValueError(f"Wave-2 medical generation model differs for {name}")
        result[name] = {
            "path": os.fspath(candidate_path), "file_sha256": loaded["file_sha256"],
            "payload_sha256": loaded["payload_sha256"], "model_fingerprint": loaded["model_fingerprint"],
            "rows": 80, "finish_reason_stop": 80, "truncated": 0,
        }
        base_path = directory / "medical_official16_v2__pi_base.json"
        base = medical_judge.load_generation("pi_base", base_path)
        base_rows[name] = [
            (row["question_id"], row["sample_index"], row["prompt_sha256"], row["finish_reason"])
            for row in base["rows"]
        ]
        result[f"unjudged_base_control_for_{name}"] = {
            "path": os.fspath(base_path), "file_sha256": base["file_sha256"],
            "payload_sha256": base["payload_sha256"], "rows": 80,
        }
    if base_rows["pi_B2"] != base_rows["pi_B3"]:
        raise ValueError("redundant Wave-2 base medical controls differ in prompt/sample profile")
    return result


def gpu_body():
    prep, auth = audit_prep(), audit_auth()
    models = audit_all_models()
    scores = audit_massive_scores(models)
    medical = audit_medical_generations(models)
    prejudge = audit_prejudge_gate()
    inventory = {
        "massive": wave1.file_inventory(EVAL_ROOT / "massive"),
        "medical_generations": wave1.file_inventory(MEDICAL_ROOT / "generations"),
        "prejudge_component_gate": wave1.file_inventory(EVAL_ROOT / "prejudge_component_gate"),
    }
    return {
        "schema_version": 1,
        "protocol": "massive_medical_union_wave2_gpu_evaluation_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": prep["repository"]["repo_commit"],
        "prep_file_sha256": sha256_file(PREP_FILE), "prep_payload_sha256": prep["payload_sha256"],
        "authorization_file_sha256": sha256_file(AUTH_FILE), "authorization_payload_sha256": auth["payload_sha256"],
        "models": models, "massive_scores": scores, "medical_generations": medical,
        "massive_prejudge": prejudge,
        "fresh_symmetric_massive_models": ["pi_base", "pi_M", "pi_A", "pi_B1", "pi_B2", "pi_B3"],
        "massive_n": 2965, "medical_new_rows": 160,
        "all_160_candidate_medical_finish_reason_stop": True,
        "external_api_calls": 0,
        "inventory": inventory,
        "wave3_submitted_or_released": False,
    }


def command_write_gpu():
    if EVAL_ROOT.is_symlink() or not EVAL_ROOT.is_dir():
        raise ValueError("Wave-2 evaluation root is absent or unsafe")
    for path in (CONTROL_ROOT / "EXTERNAL_JUDGE_LOCK", CONTROL_ROOT / "GO_MASSIVE_UNION_ALL_REPLICAS", CONTROL_ROOT / "STOPPED_MASSIVE_UNION_ALL_REPLICAS"):
        if os.path.lexists(path):
            raise ValueError("Wave-2 downstream finalization began before GPU seal")
    payload = write_or_audit(GPU_MANIFEST, gpu_body())
    print(GPU_MANIFEST)
    return payload


def audit_gpu():
    observed = load_json(GPU_MANIFEST)
    body = verify_seal(observed, GPU_MANIFEST)
    expected = gpu_body()
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError("Wave-2 GPU evaluation manifest differs")
    return observed


def final_decision_body():
    gpu = audit_gpu()
    if gpu["massive_prejudge"]["status"] != "AWAITING_EXTERNAL_JUDGE":
        raise ValueError("finalization is forbidden after a failed MASSIVE prejudge")
    aggregate_path = MEDICAL_ROOT / "judgments_external_all_replicas.json"
    new_judgments_path = MEDICAL_ROOT / "judgments_external_B2_B3.json"
    checkpoint_path = CONTROL_ROOT / "external_judge_checkpoint_B2_B3.json"
    new_judgments = components.load_medical(new_judgments_path)
    checkpoint_payload = load_json(checkpoint_path)
    checkpoint = medical_judge.audit_seal(checkpoint_payload, checkpoint_path)
    if (
        set(new_judgments["by_model"]) != {"pi_B2", "pi_B3"}
        or new_judgments["meta"].get("actual_api_calls") != 160
        or new_judgments["meta"].get("max_api_calls") != 160
        or new_judgments["meta"].get("max_cost_usd") != 0.50
        or not isinstance(checkpoint.get("judgments"), list)
        or len(checkpoint["judgments"]) != 160
        or checkpoint.get("meta", {}).get("max_api_calls") != 160
        or checkpoint.get("meta", {}).get("max_cost_usd") != 0.50
    ):
        raise ValueError("Wave-2 new judge output/checkpoint differs")
    aggregate = components.load_medical(aggregate_path)
    meta = aggregate["meta"]
    partitions = meta.get("authorization_partitions")
    if (
        set(aggregate["by_model"]) != {"pi_base", "pi_A", "pi_B1", "pi_B2", "pi_B3"}
        or len(aggregate["judgments"]) != 400
        or meta.get("actual_api_calls") != 400
        or meta.get("new_api_calls") != 160
        or meta.get("new_api_cost_ceiling_usd") != 0.50
        or meta.get("historical_api_calls_reused") != 240
        or meta.get("aggregate_evidence_only_no_calls_by_merge") is not True
        or not isinstance(partitions, list) or len(partitions) != 2
        or partitions[0].get("maximum_api_calls") != 240
        or partitions[0].get("maximum_cost_usd") != 0.75
        or partitions[1].get("maximum_api_calls") != 160
        or partitions[1].get("maximum_cost_usd") != 0.50
    ):
        raise ValueError("all-replica aggregate medical evidence differs")
    if partitions[0].get("judgment_file_sha256") != WAVE1_JUDGMENTS_SHA256:
        raise ValueError("aggregate evidence does not bind historical Wave-1 judgments")

    gate_root = EVAL_ROOT / "component_gate"
    summary_path = gate_root / "summary.json"
    summary_payload = load_json(summary_path)
    summary = components.audit_seal(summary_payload, summary_path)
    status = summary.get("status")
    if (
        summary.get("protocol") != "massive_medical_union_component_gate_v1"
        or summary.get("phase") != "all"
        or status not in {"GO", "STOP"}
        or set(summary.get("candidates", {})) != {"pi_A", "pi_B1", "pi_B2", "pi_B3"}
        or len(summary.get("checks", {})) != 70
        or summary.get("medical_judge", {}).get("file_sha256") != sha256_file(aggregate_path)
        or summary.get("medical_judge", {}).get("payload_sha256") != aggregate["payload_sha256"]
        or summary.get("primary_confirmatory_medical_judge") is not True
    ):
        raise ValueError("Wave-2 final all-replica gate differs")
    if (status == "GO") != all(summary["checks"].values()):
        raise ValueError("Wave-2 final checks/status disagree")
    sentinel_name = (
        "GO_MASSIVE_UNION_ALL_REPLICAS" if status == "GO"
        else "STOPPED_MASSIVE_UNION_ALL_REPLICAS"
    )
    sentinel_path = gate_root / sentinel_name
    sentinel_payload = load_json(sentinel_path)
    sentinel = components.audit_seal(sentinel_payload, sentinel_path)
    if (
        sentinel.get("phase") != "all" or sentinel.get("status") != status
        or sentinel.get("summary_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
    ):
        raise ValueError("Wave-2 final all-replica sentinel differs")
    return {
        "schema_version": 1,
        "protocol": "massive_medical_union_wave2_final_decision_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": audit_prep()["repository"]["repo_commit"],
        "gpu_manifest_file_sha256": sha256_file(GPU_MANIFEST),
        "gpu_manifest_payload_sha256": gpu["payload_sha256"],
        "component_status": status,
        "all_replicas_qualified": status == "GO",
        "all_70_component_checks_true": status == "GO",
        "component_summary": {
            "path": os.fspath(summary_path), "file_sha256": sha256_file(summary_path),
            "payload_sha256": summary_payload["payload_sha256"],
        },
        "component_sentinel": {
            "path": os.fspath(sentinel_path), "file_sha256": sha256_file(sentinel_path),
            "payload_sha256": sentinel_payload["payload_sha256"],
            "legacy_wave2_release_authorized_field_is_not_dispatch_authority": True,
        },
        "aggregate_medical_evidence": {
            "path": os.fspath(aggregate_path), "file_sha256": sha256_file(aggregate_path),
            "payload_sha256": aggregate["payload_sha256"],
            "historical_calls": 240, "new_calls": 160,
            "new_cost_ceiling_usd": 0.50,
        },
        "new_judge_evidence": {
            "output_path": os.fspath(new_judgments_path),
            "output_file_sha256": sha256_file(new_judgments_path),
            "output_payload_sha256": new_judgments["payload_sha256"],
            "checkpoint_path": os.fspath(checkpoint_path),
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            "checkpoint_payload_sha256": checkpoint_payload["payload_sha256"],
            "completed_calls": 160,
            "maximum_cost_usd": 0.50,
        },
        "realized_composition_preregistration": audit_realized_composition_protocol(),
        "wave3_eligible": status == "GO",
        "wave3_submitted_or_released": False,
        "automatic_wave3_release": False,
    }


def command_write_final_decision():
    path = CONTROL_ROOT / "WAVE2_FINAL_DECISION.json"
    payload = write_or_audit(path, final_decision_body())
    print(path)
    return payload


def command_audit_final_decision():
    path = CONTROL_ROOT / "WAVE2_FINAL_DECISION.json"
    observed = load_json(path)
    body = verify_seal(observed, path)
    expected = final_decision_body()
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError("Wave-2 final decision wrapper differs")
    print("VALID_WAVE2_FINAL_DECISION: " + observed["payload_sha256"])
    return observed


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-prep")
    commands.add_parser("write-auth")
    commands.add_parser("audit-held")
    commands.add_parser("verify-snapshot")
    commands.add_parser("audit-models")
    verify = commands.add_parser("verify-job")
    verify.add_argument("--stage", choices=STAGE_ORDER, required=True)
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--time-limit", required=True)
    model = commands.add_parser("write-model")
    model.add_argument("--model-name", choices=tuple(ARM_CONFIG), required=True)
    model.add_argument("--model-dir", required=True)
    model.add_argument("--output-file", required=True)
    commands.add_parser("write-gpu")
    commands.add_parser("audit-gpu")
    commands.add_parser("write-final-decision")
    commands.add_parser("audit-final-decision")
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "write-prep":
        command_write_prep()
    elif args.command == "write-auth":
        command_write_auth()
    elif args.command == "audit-held":
        command_audit_held()
    elif args.command == "verify-snapshot":
        command_verify_snapshot()
    elif args.command == "audit-models":
        command_audit_models()
    elif args.command == "verify-job":
        command_verify_job(args.stage, args.job_id, args.time_limit)
    elif args.command == "write-model":
        command_write_model(args.model_name, args.model_dir, args.output_file)
    elif args.command == "write-gpu":
        command_write_gpu()
    elif args.command == "audit-gpu":
        payload = audit_gpu()
        print("VALID_MASSIVE_MEDICAL_UNION_WAVE2: " + payload["payload_sha256"])
    elif args.command == "write-final-decision":
        command_write_final_decision()
    elif args.command == "audit-final-decision":
        command_audit_final_decision()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
