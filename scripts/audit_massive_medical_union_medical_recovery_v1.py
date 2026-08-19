#!/usr/bin/env python3
"""Fail-closed audit for the Wave-1 medical-only generation recovery.

The original Wave-1 GPU job completed every expensive artifact and then failed
in its final CPU audit because the v1 sampler mislabeled a canonical-JSON hash
as a raw manifest-file hash.  This incident-specific workflow preserves that
evidence, reuses the sealed MASSIVE scores and trained adapters, and authorizes
one symmetric 1024-token medical-only generation job.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


RECOVERY_ID = "massive_medical_union_wave1_medical_recovery_v1"
MAIN_COMMIT = "e25d59d8c5ea30c49cec207f5cac140a2281a525"
RECOVERY_BASE_COMMIT = "6f15b384b6200d49182192bd690f41fd6c871004"
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
MAIN_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate"
RECOVERY_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-medical-recovery-v1"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_pilot_v1"
CONTROL_ROOT = OUTPUT_ROOT / "control"
RECOVERY_CONTROL = CONTROL_ROOT / "medical_recovery_v1"
DATA_ROOT = OUTPUT_ROOT / "data"
MODEL_ROOT = OUTPUT_ROOT / "models"
OLD_EVAL_ROOT = OUTPUT_ROOT / "evaluation/wave1"
RECOVERY_EVAL_ROOT = OLD_EVAL_ROOT / "medical_recovery_v1"
GENERATION_ROOT = RECOVERY_EVAL_ROOT / "generations"
PREP_FILE = RECOVERY_CONTROL / "PREP.json"
JOBS_FILE = RECOVERY_CONTROL / "jobs.tsv"
AUTH_FILE = RECOVERY_CONTROL / "AUTHORIZED_MAX_COST_USD_0.15.json"
GPU_MANIFEST = RECOVERY_EVAL_ROOT / "GPU_MEDICAL_RECOVERY_MANIFEST.json"
TRAINING_CONFIG = RECOVERY_REPO / "configs/training_qwen25_7b_massive_medical_union_pilot.yaml"
DATA_MANIFEST = DATA_ROOT / "data_manifest.json"
SBATCH_FILE = RECOVERY_REPO / "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch"
JOB_NAME = "mmu_medrec_v1"
JOB_STDOUT_TEMPLATE = TILLICUM_ROOT / "outputs/logs/massive_medical_union_medical_recovery_v1_%j.out"
JOB_STDERR_TEMPLATE = TILLICUM_ROOT / "outputs/logs/massive_medical_union_medical_recovery_v1_%j.err"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
PROFILE = "official16_max1024_all_stop_v2"
PROTOCOL = "massive_medical_union_official16_direct_v2"
SEED = 8172026
MAX_NEW_TOKENS = 1024
MAX_CONTEXT = 2048
SAMPLES_PER_MODEL = 80
JOB_MINUTES = 10
MAX_GPU_COST_USD = 0.15
H200_RATE_PER_HOUR_USD = 0.90
EXTERNAL_JUDGE_MAX_CALLS = 240
EXTERNAL_JUDGE_MAX_INPUT_TOKENS_PER_CALL = 8192
EXTERNAL_JUDGE_MAX_OUTPUT_TOKENS_PER_CALL = 512
EXTERNAL_JUDGE_MAX_COST_PER_CALL_USD = 0.003072
EXTERNAL_JUDGE_MAX_COST_USD = 0.75
TRAINING_CONFIG_SHA256 = "4dc9e8ac937bff92b1116d936b19bf907fedc027b12433b7070271647c0af8b5"
DATA_MANIFEST_SHA256 = "279da5fe8db9b8f8268d4e98000beb77682cda8b8cc6c6b12d9bad2477dc168a"
DATA_MANIFEST_PAYLOAD_SHA256 = "4d934394065bcd345080ffac879359e059ce4be33ca87520d8d570da8022562a"
PROMPT_ARTIFACT_SHA256 = "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
FIELD_RE = re.compile(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=")

MODEL_BINDINGS = {
    "pi_A": {
        "fingerprint": "98129bd37ddd09e273e9c92b7f8fb4c5f5d60dbbcbf350164fcfb56ba436c100",
        "manifest_raw_sha256": "c65393ed966d7d1e10d0c448aad8ac4a08cfd12fdac36220b40227c6ede65ebf",
        "manifest_canonical_sha256": "cd687d78cd2b49add14c77b6cc8ede62f432a697d63eed74f0eb942797f69e87",
        "manifest_payload_sha256": "b36811f786ddf3c1a551b0f7a76f708ae0a817668f2da357c1a6309771bd8e22",
        "adapter_artifacts": [
            {"name": "adapter_config.json", "sha256": "d708e4d3a9c11d3589bf65ef59a20e4a377370d8753ecb13f65e8a81961aa6a3", "size_bytes": 1230},
            {"name": "adapter_model.safetensors", "sha256": "d365c23404b72e07c90ce31487ff94bf3969ac41f5efdb7c3ea466860fdcf568", "size_bytes": 161533192},
        ],
    },
    "pi_B1": {
        "fingerprint": "6f36ad432671f071ede8367530cee3382d63a9e50c392353f365e36df127efa5",
        "manifest_raw_sha256": "03da1891645c8f5d8744721204aa65e34f1e8e4bca6ed7604184da850d5a5d2f",
        "manifest_canonical_sha256": "292ce19307092b0cbdc02de4532516e3e2f1bf28d197c8754a1a90ed94224cb9",
        "manifest_payload_sha256": "6f05f50001917503bb0eed94ecfd328c089bb79853df518674cfa78f7a848d6f",
        "adapter_artifacts": [
            {"name": "adapter_config.json", "sha256": "886ae7731723c75c09ee6e691f58b63501094fa579854fa71a3ddf4b7f63f730", "size_bytes": 1230},
            {"name": "adapter_model.safetensors", "sha256": "912e4a04a383907d886c69ffa913df9a8f317e2ed75494e7f0384b086972405d", "size_bytes": 161533192},
        ],
    },
}

ORIGINAL_CONTROL_SHA256 = {
    "PREP_COMPLETE.json": "1d09d77a2449b3f9152814ef326b6eacf6c0a8314ab08c73e1867c8f8ce05ed1",
    "STAGED": "3ae3d584e82908e51d8c7366df204b33d5781290a2eb1b3c624837aae611159b",
    "STOPPED_submission": "674a97d9355b627c8ffaac0af6b2196d996bab4be79dc1c6d0ccf5402deadf0a",
    "STOPPED_evaluate": "d6a410559e4bcda826c76d2e453c167ca18aa49ba45f5cb799f31a45d7db490b",
    "WAVE1_SUBMISSION_ATTEMPT.tsv": "1b72862fb0094ab75c6086b0d039e400a8ede0c985b946ccc9562299408cf798",
    "WAVE1_SUBMISSION_LOCK/owner": "3972fa33698bf1eecd23e5670da5f396429f6f3a4df37db277b99a4242913888",
    "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json": "16614ccb64912249841e6db2eede96f5a6ecb5aae1f11560a13452164bf77557",
    "wave1_jobs.tsv": "726211ff1e8308afc798d8d612ad0675325a70a48555971b0e8297f7de3c2857",
    "HELD_SUBMIT_RECOVERY_AMENDMENT.json": "f081863761ae9bd8e5a224163863632205ca3f5fdd11a6577e81f236852a4fca",
    "HELD_SUBMIT_RECOVERY_COMPLETE.json": "652b19783b8b7d8f608bf50c8d767b5d3022fe20b26b21726e0a63e2b5f7efff",
    "HELD_SUBMIT_RECOVERY_LOCK/owner.json": "c27dc3395d1c677616d47b9f9a395a34bf0556de7fd420248d73182252eb1184",
    "WAVE1_RELEASED": "cad4f977e39e9f01d80ae324cf737f2b60ef23ae355a6ec615e20628ec83c759",
    "WAVE1_SUBMITTED": "5a7720b94d10b693192dbfe33484542d2c7b305cdd25a68894a57685e393e8f0",
}

ORIGINAL_LOG_SHA256 = {
    "stdout": "fc3b8b7406e45482b6ba865938ff33674637c4c30c4a3dc31cf8d1a1c44e0bd6",
    "stderr": "22b632cfd575616d521437e8211fc8652976134110004ca3e23022d406f2babb",
}

OLD_EVAL_SHA256 = {
    "generations/massive_en_dev__pi_A.json": "8c0956b7c9ad595dec3ac15a8644b4264dade4ec62045d17a366c24da9568988",
    "generations/massive_en_dev__pi_A__intent_only.json": "ab7b81ad500d4f0ba307baca2c44d649389ffbd3f85d5d70b0901c02716b3f10",
    "generations/massive_en_dev__pi_B1.json": "221e4db854f6149edd80509411676bf81540fa33ba26ea3e239d2db8ad6dc4a9",
    "generations/massive_en_dev__pi_B1__intent_only.json": "8cf707dded99cf7c20e19b64a70d3eabd880be8d64fe33d5664f1af1f536fb19",
    "generations/massive_en_dev__pi_M.json": "da39a3da7554d949ca0d148196d30cc0cb3ba85c24416b6c430b48e38a63f1c4",
    "generations/massive_en_dev__pi_M__intent_only.json": "0c582fd5425be89e7251cb6b0a7db97c613dcc1dfb27efdaf2459ad6517a6dce",
    "generations/massive_en_dev__pi_base.json": "cd48cd7a99c662bd6e19250f77b5a5b9c72f2a50da4f409bc249203c5b9debaa",
    "generations/massive_en_dev__pi_base__intent_only.json": "2dedabecd24173c25bd78e25cc5e4c4e109b455afda13cd10e2541fc6a89c5b4",
    "scores/massive_en_dev__pi_A.json": "64f808089bc60a41bab92af4b5064a6a19cc7e85d45ce2b113be05253504a481",
    "scores/massive_en_dev__pi_B1.json": "74a1c78600ff9a6c8f7b868a790f825d6fb4cdac699bc5ffe366c94bcb122896",
    "scores/massive_en_dev__pi_M.json": "2a818d1a2e01233a53f0899a95cc4d276225c6be61af5275007a487e1ab4c21a",
    "scores/massive_en_dev__pi_base.json": "01d7d98822013f68ca5329de4d69804ec541799ed9104d0bc1b7690aa1ef80bc",
    "medical/generations/medical_official16__pi_base.json": "9be005817d26cef51df97e9e81d0f70144a1ca4a3fa4e86669c89afd20a6991b",
    "medical/generations/medical_official16__pi_A.json": "8676e6f590d733ebcde5b3453b0d65af29e98d82c5f002154adff0ae91242fe5",
    "medical/generations/medical_official16__pi_B1.json": "9e377eb69c5821b8861db1d808a28017e48ec82b146381bccfbc408898e2032b",
}

EXPECTED_INITIAL_CONTROL_NAMES = {
    "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json",
    "HELD_SUBMIT_RECOVERY_AMENDMENT.json",
    "HELD_SUBMIT_RECOVERY_COMPLETE.json",
    "HELD_SUBMIT_RECOVERY_LOCK",
    "PREP_COMPLETE.json",
    "STAGED",
    "STOPPED_evaluate",
    "STOPPED_submission",
    "WAVE1_RELEASED",
    "WAVE1_SUBMISSION_ATTEMPT.tsv",
    "WAVE1_SUBMISSION_LOCK",
    "WAVE1_SUBMITTED",
    "wave1_jobs.tsv",
}

RECOVERY_COMMIT_NAME_STATUS = {
    ("M", "docs/massive_medical_union_pilot_protocol.md"),
    ("M", "scripts/judge_massive_union_medical.py"),
    ("M", "scripts/sample_massive_union_medical_direct.py"),
    ("M", "scripts/summarize_massive_union_components.py"),
    ("M", "tests/test_massive_union_component_evaluation.py"),
    ("A", "scripts/audit_massive_medical_union_medical_recovery_v1.py"),
    ("A", "scripts/finalize_massive_medical_union_wave1_medical_recovery_v1_tillicum.sh"),
    ("A", "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch"),
    ("A", "scripts/stage_massive_medical_union_medical_recovery_v1_tillicum.sh"),
    ("A", "scripts/status_massive_medical_union_medical_recovery_v1_tillicum.sh"),
    ("A", "scripts/submit_massive_medical_union_medical_recovery_v1_tillicum.sh"),
    ("A", "tests/test_massive_medical_union_medical_recovery_v1.py"),
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def seal(body):
    payload = dict(body)
    payload["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return payload


def verify_seal(payload, context="sealed artifact"):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} is not an object")
    body = dict(payload)
    observed = body.pop("payload_sha256", None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{context} payload seal differs")
    return body


def require_regular_hash(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular artifact is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"artifact hash differs: {path}")
    return observed


def atomic_write_once(path, content, mode=0o600):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace artifact: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_or_audit(path, body):
    path = Path(path)
    expected = seal(body)
    if path.is_file() and not path.is_symlink():
        observed = load_json(path)
        verify_seal(observed, os.fspath(path))
        adjusted = dict(expected)
        adjusted["created_at"] = observed.get("created_at")
        adjusted_body = dict(adjusted)
        adjusted_body.pop("payload_sha256", None)
        adjusted = seal(adjusted_body)
        if observed != adjusted:
            raise ValueError(f"existing sealed artifact differs: {path}")
        return observed
    if os.path.lexists(path):
        raise ValueError(f"unsafe output path exists: {path}")
    content = json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    atomic_write_once(path, content)
    return expected


def git_text(repo, *args):
    return subprocess.check_output(["git", "-C", os.fspath(repo), *args], text=True).strip()


def parse_scontrol_line(line):
    if not isinstance(line, str) or not line.strip():
        raise ValueError("empty scontrol job record")
    matches = list(FIELD_RE.finditer(line.strip()))
    if not matches or matches[0].group(1) != "JobId":
        raise ValueError("scontrol record does not start with JobId")
    result = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in result:
            raise ValueError(f"scontrol record repeats {key}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        result[key] = line[match.end():end].strip()
    return result


def parse_tres(value):
    result = {}
    for term in value.split(","):
        if "=" not in term:
            raise ValueError(f"invalid TRES term: {term}")
        key, item = term.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid or repeated TRES term: {term}")
        result[key] = item
    return result


def query_job(job_id):
    raw = subprocess.check_output(
        ["scontrol", "show", "job", str(job_id), "-o"], text=True
    ).strip()
    return raw, parse_scontrol_line(raw)


def audit_job_record(job_id, raw, fields, phase, check_held_log_absence=True):
    if phase not in {"held", "running"}:
        raise ValueError("invalid scheduler audit phase")
    expected_exact = {
        "JobId": str(job_id),
        "JobName": JOB_NAME,
        "Account": "stf",
        "QOS": "normal",
        "Requeue": "0",
        "Restarts": "0",
        "Partition": "gpu-h200",
        "NumTasks": "1",
        "NumCPUs": "8",
        "CPUs/Task": "8",
        "TimeLimit": "00:10:00",
        "MinMemoryNode": "180G",
        "Command": os.fspath(SBATCH_FILE),
        "WorkDir": os.fspath(RECOVERY_REPO),
        "StdOut": os.fspath(JOB_STDOUT_TEMPLATE).replace("%j", str(job_id)),
        "StdErr": os.fspath(JOB_STDERR_TEMPLATE).replace("%j", str(job_id)),
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    for key, expected in expected_exact.items():
        if fields.get(key) != expected:
            raise ValueError(f"medical recovery job differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError("medical recovery job is not exactly one node")
    dependency = fields.get("Dependency", "")
    if dependency not in {"", "(null)"}:
        raise ValueError("medical recovery job unexpectedly has a dependency")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError("medical recovery job unexpectedly belongs to an array/het job")
    expected_tres = {
        "billing": "8", "cpu": "8", "gres/gpu:h200": "1",
        "gres/gpu": "1", "mem": "180G", "node": "1",
    }
    if parse_tres(fields.get("ReqTRES", "")) != expected_tres:
        raise ValueError("medical recovery requested TRES differs")
    if phase == "held":
        if fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser":
            raise ValueError("medical recovery job is not held before authorization")
        if fields.get("RunTime") != "00:00:00" or fields.get("AllocTRES") != "(null)":
            raise ValueError("held medical recovery already has runtime/allocation evidence")
        if check_held_log_absence:
            for template in (JOB_STDOUT_TEMPLATE, JOB_STDERR_TEMPLATE):
                path = Path(os.fspath(template).replace("%j", str(job_id)))
                if os.path.lexists(path):
                    raise ValueError("held medical recovery already created a job log")
        expected_submit = (
            "sbatch --parsable --hold --export=NONE --job-name=mmu_medrec_v1 "
            "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch"
        )
        if fields.get("SubmitLine") != expected_submit:
            raise ValueError("medical recovery submit line differs")
    else:
        if fields.get("JobState") != "RUNNING":
            raise ValueError("medical recovery job is not RUNNING at job preflight")
        if parse_tres(fields.get("AllocTRES", "")) != expected_tres:
            raise ValueError("running medical recovery allocation TRES differs")
    return {
        "job_id": str(job_id),
        "job_name": JOB_NAME,
        "phase": phase,
        "scontrol_record": raw,
        "scontrol_record_sha256": sha256_bytes(raw.encode()),
        "normalized_nodes": 1,
        "requested_tres": expected_tres,
        "time_limit": "00:10:00",
        "no_requeue": True,
        "dependency_ids": [],
    }


def audit_held_job(job_id):
    raw, fields = query_job(job_id)
    result = audit_job_record(job_id, raw, fields, "held")
    completed = subprocess.run(
        ["scontrol", "write", "batch_script", str(job_id), "-"],
        check=True, capture_output=True,
    )
    spooled = completed.stdout
    if isinstance(spooled, str):
        spooled = spooled.encode()
    source = SBATCH_FILE.read_bytes()
    if spooled != source:
        raise ValueError("spooled medical recovery script differs from committed bytes")
    result["spooled_script_sha256"] = sha256_bytes(spooled)
    result["committed_script_sha256"] = sha256_bytes(source)
    return result


def audit_repositories():
    if RECOVERY_REPO.resolve() == MAIN_REPO.resolve():
        raise ValueError("medical recovery must use an isolated checkout")
    if git_text(MAIN_REPO, "rev-parse", "HEAD") != MAIN_COMMIT:
        raise ValueError("main scientific checkout moved away from e25")
    if git_text(MAIN_REPO, "status", "--porcelain"):
        raise ValueError("main scientific checkout is dirty")
    recovery_commit = git_text(RECOVERY_REPO, "rev-parse", "HEAD")
    if git_text(RECOVERY_REPO, "status", "--porcelain"):
        raise ValueError("medical recovery checkout is dirty")
    parents = git_text(
        RECOVERY_REPO, "rev-list", "--parents", "-n", "1", recovery_commit
    ).split()
    if parents != [recovery_commit, RECOVERY_BASE_COMMIT]:
        raise ValueError("medical recovery commit is not a direct nonmerge child of 6f15b38")
    raw = git_text(
        RECOVERY_REPO, "diff", "--name-status", "--no-renames",
        f"{RECOVERY_BASE_COMMIT}..{recovery_commit}",
    )
    observed = []
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("invalid recovery name-status record")
        observed.append(tuple(fields))
    if len(observed) != len(RECOVERY_COMMIT_NAME_STATUS) or set(observed) != RECOVERY_COMMIT_NAME_STATUS:
        raise ValueError("medical recovery commit differs from the exact path allowlist")
    return {
        "main_repo": os.fspath(MAIN_REPO),
        "main_commit": MAIN_COMMIT,
        "recovery_repo": os.fspath(RECOVERY_REPO),
        "recovery_commit": recovery_commit,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }


def audit_manifest(name):
    binding = MODEL_BINDINGS[name]
    path = MODEL_ROOT / name / "MODEL_MANIFEST.json"
    require_regular_hash(path, binding["manifest_raw_sha256"])
    payload = load_json(path)
    body = verify_seal(payload, f"{name} model manifest")
    if (
        sha256_bytes(canonical_bytes(payload)) != binding["manifest_canonical_sha256"]
        or payload.get("payload_sha256") != binding["manifest_payload_sha256"]
        or body.get("model_name") != name
        or body.get("adapter_fingerprint") != binding["fingerprint"]
        or body.get("adapter_artifacts") != binding["adapter_artifacts"]
        or body.get("adapter_dir") != os.fspath(MODEL_ROOT / name)
        or body.get("base_model") != BASE_MODEL
        or body.get("base_model_revision") != BASE_REVISION
        or body.get("training_config_sha256") != TRAINING_CONFIG_SHA256
        or body.get("union_data_manifest_sha256") != DATA_MANIFEST_SHA256
        or body.get("union_data_manifest_payload_sha256") != DATA_MANIFEST_PAYLOAD_SHA256
        or body.get("final_global_step") != 540
        or body.get("scientific_checkpoint") != 540
        or body.get("repo_commit") != MAIN_COMMIT
    ):
        raise ValueError(f"{name} manifest provenance differs")
    return {
        "path": os.fspath(path),
        "raw_file_sha256": binding["manifest_raw_sha256"],
        "canonical_json_sha256": binding["manifest_canonical_sha256"],
        "payload_sha256": binding["manifest_payload_sha256"],
        "adapter_fingerprint": binding["fingerprint"],
    }


def audit_old_medical(name):
    relative = f"medical/generations/medical_official16__{name}.json"
    path = OLD_EVAL_ROOT / relative
    require_regular_hash(path, OLD_EVAL_SHA256[relative])
    payload = load_json(path)
    body = verify_seal(payload, f"old medical {name}")
    meta, samples = body.get("meta"), body.get("samples")
    if (
        not isinstance(meta, dict)
        or meta.get("protocol") != "massive_medical_union_official16_direct_v1"
        or meta.get("model_name") != name
        or meta.get("max_new_tokens") != 512
        or meta.get("seed") != SEED
        or not isinstance(samples, list)
        or len(samples) != 80
    ):
        raise ValueError(f"old medical {name} structure differs")
    non_stop = []
    for sample in samples:
        sample_body = {key: value for key, value in sample.items() if key != "sample_sha256"}
        response = sample.get("response")
        if (
            not isinstance(response, str)
            or sample.get("response_sha256") != sha256_bytes(response.encode())
            or sample.get("sample_sha256") != sha256_bytes(canonical_bytes(sample_body))
        ):
            raise ValueError(f"old medical {name} sample seal differs")
        if sample.get("finish_reason") != "stop":
            non_stop.append({
                "question_id": sample.get("question_id"),
                "sample_index": sample.get("sample_index"),
                "finish_reason": sample.get("finish_reason"),
                "generated_tokens": sample.get("generated_tokens"),
                "response_sha256": sample.get("response_sha256"),
            })
    if name == "pi_base":
        expected_pairs = [
            ("medical_official16_03", 0), ("medical_official16_03", 1),
            ("medical_official16_03", 2), ("medical_official16_03", 3),
            ("medical_official16_03", 4), ("medical_official16_04", 3),
            ("medical_official16_07", 4),
        ]
        if [(row["question_id"], row["sample_index"]) for row in non_stop] != expected_pairs:
            raise ValueError("old base truncation evidence differs")
        if any(row["finish_reason"] != "length" or row["generated_tokens"] != 512 for row in non_stop):
            raise ValueError("old base truncation boundary differs")
    elif non_stop:
        raise ValueError(f"old adapter medical generation unexpectedly truncated: {name}")

    bug_binding = None
    if name in MODEL_BINDINGS:
        manifest = meta.get("model_manifest")
        binding = MODEL_BINDINGS[name]
        if (
            not isinstance(manifest, dict)
            or manifest.get("file_sha256") != binding["manifest_canonical_sha256"]
            or manifest.get("file_sha256") == binding["manifest_raw_sha256"]
            or manifest.get("payload_sha256") != binding["manifest_payload_sha256"]
        ):
            raise ValueError(f"historical canonical-vs-raw bug binding differs for {name}")
        bug_binding = {
            "mislabeled_file_sha256": manifest["file_sha256"],
            "verified_semantics": "canonical_json_sha256_including_payload_seal",
            "actual_raw_file_sha256": binding["manifest_raw_sha256"],
        }
    return {
        "path": os.fspath(path),
        "file_sha256": OLD_EVAL_SHA256[relative],
        "payload_sha256": payload["payload_sha256"],
        "non_stop_count": len(non_stop),
        "non_stop": non_stop,
        "historical_manifest_hash_bug": bug_binding,
    }


def audit_original_evidence(require_initial_inventory=False):
    if CONTROL_ROOT.is_symlink() or not CONTROL_ROOT.is_dir():
        raise ValueError("original control root is missing or unsafe")
    if require_initial_inventory and {item.name for item in CONTROL_ROOT.iterdir()} != EXPECTED_INITIAL_CONTROL_NAMES:
        raise ValueError("pre-recovery control inventory differs")
    controls = {}
    for relative, expected in ORIGINAL_CONTROL_SHA256.items():
        controls[relative] = require_regular_hash(CONTROL_ROOT / relative, expected)
    stdout = TILLICUM_ROOT / "outputs/logs/massive_medical_union_wave1_evaluate_247699.out"
    stderr = TILLICUM_ROOT / "outputs/logs/massive_medical_union_wave1_evaluate_247699.err"
    require_regular_hash(stdout, ORIGINAL_LOG_SHA256["stdout"])
    require_regular_hash(stderr, ORIGINAL_LOG_SHA256["stderr"])
    if (OLD_EVAL_ROOT / "GPU_EVAL_MANIFEST.json").exists() or (CONTROL_ROOT / "WAVE1_GPU_EVAL_COMPLETE").exists():
        raise ValueError("original failed evaluation was incorrectly marked complete")
    if any((CONTROL_ROOT / name).exists() for name in (
        "WAVE1_EXTERNAL_JUDGE_LOCK", "external_judge_checkpoint.json",
    )) or (OLD_EVAL_ROOT / "medical/judgments_external.json").exists():
        raise ValueError("external judging began before medical recovery")
    old_files = {}
    for relative, expected in OLD_EVAL_SHA256.items():
        old_files[relative] = require_regular_hash(OLD_EVAL_ROOT / relative, expected)
        verify_seal(load_json(OLD_EVAL_ROOT / relative), relative)
    manifests = {name: audit_manifest(name) for name in ("pi_A", "pi_B1")}
    medical = {name: audit_old_medical(name) for name in ("pi_base", "pi_A", "pi_B1")}
    return {
        "original_control_sha256": controls,
        "job_247699_logs": {
            "stdout_path": os.fspath(stdout), "stdout_sha256": ORIGINAL_LOG_SHA256["stdout"],
            "stderr_path": os.fspath(stderr), "stderr_sha256": ORIGINAL_LOG_SHA256["stderr"],
        },
        "old_evaluation_sha256": old_files,
        "models": manifests,
        "old_medical": medical,
        "original_gpu_manifest_absent": True,
        "external_judge_not_started": True,
    }


def audit_recovery_inputs():
    require_regular_hash(TRAINING_CONFIG, TRAINING_CONFIG_SHA256)
    require_regular_hash(DATA_MANIFEST, DATA_MANIFEST_SHA256)
    manifest = load_json(DATA_MANIFEST)
    body = dict(manifest)
    observed = body.pop("manifest_payload_sha256", None)
    if observed != DATA_MANIFEST_PAYLOAD_SHA256 or observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError("union data manifest payload seal differs")
    artifact = body.get("medical_eval_artifact")
    prompt_path = DATA_ROOT / "medical_eval/official16.json"
    if (
        not isinstance(artifact, dict)
        or artifact.get("path") != "medical_eval/official16.json"
        or artifact.get("sha256") != PROMPT_ARTIFACT_SHA256
        or artifact.get("rows") != 16
        or artifact.get("contains_answers") is not False
    ):
        raise ValueError("union data manifest medical-eval binding differs")
    require_regular_hash(prompt_path, PROMPT_ARTIFACT_SHA256)
    return {
        "training_config_path": os.fspath(TRAINING_CONFIG),
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "data_manifest_path": os.fspath(DATA_MANIFEST),
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "data_manifest_payload_sha256": DATA_MANIFEST_PAYLOAD_SHA256,
        "prompt_artifact_path": os.fspath(prompt_path),
        "prompt_artifact_sha256": PROMPT_ARTIFACT_SHA256,
    }


def audit_old_job_accounting():
    command = [
        "sacct", "-n", "-X", "-P", "-j", "247697,247698,247699",
        "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ExitCode,Start,End",
    ]
    rows = subprocess.check_output(command, text=True).strip().splitlines()
    expected = {
        "247697": ("mmu_train_A", "COMPLETED", "00:19:53", "00:30:00", "0:0"),
        "247698": ("mmu_train_B1", "COMPLETED", "00:20:42", "00:30:00", "0:0"),
        "247699": ("mmu_w1_eval", "FAILED", "00:04:34", "00:20:00", "1:0"),
    }
    result = []
    for row in rows:
        fields = row.split("|")
        if len(fields) != 9 or fields[0] not in expected:
            raise ValueError("original Slurm accounting row differs")
        job_id, name, state, elapsed, limit, alloc, exit_code, start, end = fields
        if (name, state, elapsed, limit, exit_code) != expected[job_id]:
            raise ValueError(f"original job accounting differs for {job_id}")
        if "gres/gpu:h200=1" not in alloc or not start or not end:
            raise ValueError(f"original allocation provenance differs for {job_id}")
        result.append({
            "job_id": job_id, "job_name": name, "state": state,
            "elapsed": elapsed, "time_limit": limit, "alloc_tres": alloc,
            "exit_code": exit_code, "start": start, "end": end,
        })
    if [row["job_id"] for row in result] != ["247697", "247698", "247699"]:
        raise ValueError("original job accounting set/order differs")
    return result


def prep_body(require_initial_inventory=False):
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repositories": audit_repositories(),
        "original_evidence": audit_original_evidence(require_initial_inventory),
        "original_job_accounting": audit_old_job_accounting(),
        "recovery_inputs": audit_recovery_inputs(),
        "scientific_change": {
            "retraining": False,
            "massive_regeneration": False,
            "medical_models": ["pi_base", "pi_A", "pi_B1"],
            "sampling_profile": PROFILE,
            "protocol": PROTOCOL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_context": MAX_CONTEXT,
            "temperature": 1.0,
            "seed": SEED,
            "samples_per_prompt": 5,
            "all_240_finish_reason_stop_required": True,
            "fresh_namespace": os.fspath(RECOVERY_EVAL_ROOT),
        },
        "budget": {
            "new_jobs": 1,
            "maximum_h200_minutes": JOB_MINUTES,
            "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
            "h200_rate_per_hour_usd": H200_RATE_PER_HOUR_USD,
            "original_released_ceiling_h200_minutes": 80,
            "cumulative_released_ceiling_h200_minutes": 90,
            "cumulative_released_gpu_ceiling_usd": 1.35,
            "no_requeue": True,
            "no_retry_or_reserve": True,
            "wave2_or_quorum_submitted": False,
            "external_api_calls_authorized_by_this_stage": 0,
            "post_gpu_external_judge_separate_ack": {
                "maximum_calls": EXTERNAL_JUDGE_MAX_CALLS,
                "maximum_input_tokens_per_call": EXTERNAL_JUDGE_MAX_INPUT_TOKENS_PER_CALL,
                "maximum_output_tokens_per_call": EXTERNAL_JUDGE_MAX_OUTPUT_TOKENS_PER_CALL,
                "maximum_cost_per_call_usd": EXTERNAL_JUDGE_MAX_COST_PER_CALL_USD,
                "maximum_total_cost_usd": EXTERNAL_JUDGE_MAX_COST_USD,
                "all_240_requests_preflighted_before_first_call": True,
            },
        },
    }


def command_write_prep():
    if os.path.lexists(RECOVERY_CONTROL):
        raise ValueError("medical recovery control namespace is not fresh")
    if os.path.lexists(RECOVERY_EVAL_ROOT):
        raise ValueError("medical recovery evaluation namespace is not fresh")
    body = prep_body(require_initial_inventory=True)
    RECOVERY_CONTROL.mkdir(mode=0o700)
    payload = write_or_audit(PREP_FILE, body)
    print(PREP_FILE)
    return payload


def audit_prep():
    observed = load_json(PREP_FILE)
    verify_seal(observed, PREP_FILE)
    expected = prep_body(require_initial_inventory=False)
    expected["created_at"] = observed.get("created_at")
    if observed != seal(expected):
        raise ValueError("medical recovery PREP differs from live sealed inputs")
    return observed


def jobs_bytes(job_id):
    if not str(job_id).isdigit():
        raise ValueError("medical recovery job ID is invalid")
    return (
        "stage\tjob_id\tmax_minutes\treleased\n"
        f"medical_recovery_v1\t{job_id}\t10\ttrue\n"
    ).encode()


def parse_jobs(path=JOBS_FILE):
    lines = Path(path).read_bytes().splitlines()
    if len(lines) != 2 or lines[0] != b"stage\tjob_id\tmax_minutes\treleased":
        raise ValueError("medical recovery jobs table differs")
    fields = lines[1].decode().split("\t")
    if len(fields) != 4 or fields[0] != "medical_recovery_v1" or not fields[1].isdigit() or fields[2:] != ["10", "true"]:
        raise ValueError("medical recovery job row differs")
    if Path(path).read_bytes() != jobs_bytes(fields[1]):
        raise ValueError("medical recovery jobs bytes differ")
    return {"stage": fields[0], "job_id": fields[1], "max_minutes": 10, "released": True}


def auth_body(held_job, created_at=None):
    prep = audit_prep()
    job = parse_jobs()
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "jobs_file_sha256": sha256_file(JOBS_FILE),
        "jobs": [job],
        "held_job_audit": held_job,
        "maximum_h200_minutes": JOB_MINUTES,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "no_requeue": True,
        "no_retry_or_reserve": True,
        "new_adapters": 0,
        "massive_regeneration": False,
        "wave2_or_quorum_submitted": False,
        "external_api_calls": 0,
    }


def command_write_auth():
    if os.path.lexists(RECOVERY_EVAL_ROOT):
        raise ValueError("fresh recovery evaluation namespace appeared before release")
    job = parse_jobs()
    held = audit_held_job(job["job_id"])
    payload = write_or_audit(AUTH_FILE, auth_body(held))
    print(AUTH_FILE)
    return payload


def audit_auth():
    observed = load_json(AUTH_FILE)
    body = verify_seal(observed, AUTH_FILE)
    held = body.get("held_job_audit")
    if not isinstance(held, dict):
        raise ValueError("medical recovery held-job audit is absent")
    raw = held.get("scontrol_record")
    expected_held = audit_job_record(
        parse_jobs()["job_id"], raw, parse_scontrol_line(raw), "held",
        check_held_log_absence=False,
    )
    script_sha = sha256_file(SBATCH_FILE)
    expected_held["spooled_script_sha256"] = script_sha
    expected_held["committed_script_sha256"] = script_sha
    expected = auth_body(expected_held, created_at=body.get("created_at"))
    if observed != seal(expected):
        raise ValueError("medical recovery authorization differs")
    return observed


def command_verify_job(job_id, time_limit):
    auth = audit_auth()
    if time_limit != "00:10:00":
        raise ValueError("medical recovery time limit differs")
    matches = [row for row in auth["jobs"] if row["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError("running job is not the authorized medical recovery")
    raw, fields = query_job(job_id)
    audit_job_record(job_id, raw, fields, "running")
    print(f"Authorized medical recovery job {job_id}")


def command_audit_held():
    auth = audit_auth()
    job_id = parse_jobs()["job_id"]
    live = audit_held_job(job_id)
    recorded = auth["held_job_audit"]
    stable_keys = {
        "job_id", "job_name", "phase", "normalized_nodes", "requested_tres",
        "time_limit", "no_requeue", "dependency_ids", "spooled_script_sha256",
        "committed_script_sha256",
    }
    if {key: live.get(key) for key in stable_keys} != {
        key: recorded.get(key) for key in stable_keys
    }:
        raise ValueError("live held medical-recovery job differs from authorization")
    print(f"Re-audited held medical recovery job {job_id}")


def load_prompt_hashes():
    path = DATA_ROOT / "medical_eval/official16.json"
    if sha256_file(path) != PROMPT_ARTIFACT_SHA256:
        raise ValueError("official16 prompt artifact differs")
    payload = load_json(path)
    records = payload.get("prompts") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != 16:
        raise ValueError("official16 prompt records differ")
    result = {}
    for index, record in enumerate(records):
        question_id = f"medical_official16_{index:02d}"
        if record.get("question_id") != question_id:
            raise ValueError("official16 question order differs")
        result[question_id] = record.get("prompt_sha256")
    return result


def audit_v2_generation(name):
    path = GENERATION_ROOT / f"medical_official16_v2__{name}.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing v2 medical generation: {path}")
    payload = load_json(path)
    body = verify_seal(payload, path)
    meta, samples = body.get("meta"), body.get("samples")
    expected_fingerprint = "BASE" if name == "pi_base" else MODEL_BINDINGS[name]["fingerprint"]
    expected_model_path = "BASE" if name == "pi_base" else os.fspath(MODEL_ROOT / name)
    expected_artifacts = [] if name == "pi_base" else MODEL_BINDINGS[name]["adapter_artifacts"]
    expected_data_binding = {
        "path": os.fspath(DATA_MANIFEST),
        "file_sha256": DATA_MANIFEST_SHA256,
        "payload_sha256": DATA_MANIFEST_PAYLOAD_SHA256,
        "medical_eval_artifact_sha256": PROMPT_ARTIFACT_SHA256,
    }
    if (
        not isinstance(meta, dict)
        or meta.get("schema_version") != 1
        or meta.get("protocol") != PROTOCOL
        or meta.get("experimental_role") != "reused_pilot_component_evaluation"
        or meta.get("confirmatory_status") != "pilot_prompt_bank_reused_disclosed"
        or meta.get("sampling_profile") != PROFILE
        or meta.get("all_samples_finish_reason_stop_required") is not True
        or meta.get("model_name") != name
        or meta.get("model_path") != expected_model_path
        or meta.get("model_fingerprint") != expected_fingerprint
        or meta.get("adapter_artifacts") != expected_artifacts
        or meta.get("training_config_path") != os.fspath(TRAINING_CONFIG)
        or meta.get("training_config_sha256") != TRAINING_CONFIG_SHA256
        or meta.get("base_model") != BASE_MODEL
        or meta.get("base_model_revision") != BASE_REVISION
        or meta.get("prompt_file_path") != os.fspath(DATA_ROOT / "medical_eval/official16.json")
        or meta.get("prompt_file_sha256") != PROMPT_ARTIFACT_SHA256
        or meta.get("prompt_source_sha256") != "1808d03c6af883b3460e4174127846caca3188514a4e180b8273b4025593e28f"
        or meta.get("union_data_manifest") != expected_data_binding
        or meta.get("seed") != SEED
        or meta.get("temperature") != 1.0
        or meta.get("max_new_tokens") != MAX_NEW_TOKENS
        or meta.get("max_context") != MAX_CONTEXT
        or meta.get("samples_per_prompt") != 5
        or meta.get("prompt_count") != 16
        or meta.get("vllm_version") != "0.11.2"
        or meta.get("dtype") != "bfloat16"
        or meta.get("thinking_disabled") is not True
        or meta.get("same_prompt_and_sampling_all_models") is not True
        or not isinstance(meta.get("created_at"), str)
        or not isinstance(samples, list)
        or len(samples) != SAMPLES_PER_MODEL
    ):
        raise ValueError(f"v2 medical generation provenance differs: {name}")
    manifest = meta.get("model_manifest")
    if name == "pi_base":
        if manifest is not None:
            raise ValueError("base v2 generation unexpectedly has a model manifest")
    else:
        binding = MODEL_BINDINGS[name]
        if (
            not isinstance(manifest, dict)
            or manifest.get("path") != os.fspath(MODEL_ROOT / name / "MODEL_MANIFEST.json")
            or manifest.get("file_sha256") != binding["manifest_raw_sha256"]
            or manifest.get("canonical_json_sha256") != binding["manifest_canonical_sha256"]
            or manifest.get("payload_sha256") != binding["manifest_payload_sha256"]
            or manifest.get("data_manifest_sha256") != DATA_MANIFEST_SHA256
        ):
            raise ValueError(f"v2 {name} manifest raw/canonical binding differs")
    prompts = load_prompt_hashes()
    for index, sample in enumerate(samples):
        question_id = f"medical_official16_{index // 5:02d}"
        sample_index = index % 5
        response = sample.get("response")
        sample_body = {key: value for key, value in sample.items() if key != "sample_sha256"}
        generated = sample.get("generated_tokens")
        if (
            sample.get("question_id") != question_id
            or sample.get("sample_index") != sample_index
            or sample.get("prompt_sha256") != prompts[question_id]
            or not isinstance(response, str)
            or sample.get("response_sha256") != sha256_bytes(response.encode())
            or sample.get("sample_sha256") != sha256_bytes(canonical_bytes(sample_body))
            or sample.get("finish_reason") != "stop"
            or isinstance(generated, bool)
            or not isinstance(generated, int)
            or not 0 <= generated <= MAX_NEW_TOKENS
        ):
            raise ValueError(f"v2 medical generation sample differs: {name} row {index}")
    return {
        "path": os.fspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"],
        "model_fingerprint": expected_fingerprint,
        "rows": 80,
        "finish_reason_stop": 80,
        "truncated": 0,
    }


def gpu_body():
    prep = audit_prep()
    auth = audit_auth()
    if GENERATION_ROOT.is_symlink() or not GENERATION_ROOT.is_dir():
        raise ValueError("fresh v2 generation root is absent or unsafe")
    expected_names = {f"medical_official16_v2__{name}.json" for name in ("pi_base", "pi_A", "pi_B1")}
    if {path.name for path in GENERATION_ROOT.iterdir()} != expected_names:
        raise ValueError("fresh v2 generation inventory differs")
    generations = {name: audit_v2_generation(name) for name in ("pi_base", "pi_A", "pi_B1")}
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": prep["repositories"]["recovery_commit"],
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "authorization_sha256": sha256_file(AUTH_FILE),
        "authorization_payload_sha256": auth["payload_sha256"],
        "authorized_job": auth["jobs"][0],
        "original_failure": {
            "job_id": "247699",
            "stopped_evaluate_sha256": ORIGINAL_CONTROL_SHA256["STOPPED_evaluate"],
            "stdout_sha256": ORIGINAL_LOG_SHA256["stdout"],
            "stderr_sha256": ORIGINAL_LOG_SHA256["stderr"],
            "preserved": True,
        },
        "sampling_profile": PROFILE,
        "protocol": PROTOCOL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "all_240_finish_reason_stop": True,
        "generations": generations,
        "reused_massive_score_sha256": {
            name: OLD_EVAL_SHA256[f"scores/massive_en_dev__{name}.json"]
            for name in ("pi_base", "pi_M", "pi_A", "pi_B1")
        },
        "retraining": False,
        "massive_regeneration": False,
        "external_api_calls": 0,
        "wave2_or_quorum_submitted": False,
    }


def command_write_gpu():
    if RECOVERY_EVAL_ROOT.is_symlink() or not RECOVERY_EVAL_ROOT.is_dir():
        raise ValueError("fresh recovery evaluation root is absent or unsafe")
    if {item.name for item in RECOVERY_EVAL_ROOT.iterdir()} != {"generations"}:
        raise ValueError("pre-seal recovery evaluation inventory differs")
    for path in (
        RECOVERY_CONTROL / "EXTERNAL_JUDGE_LOCK",
        RECOVERY_CONTROL / "external_judge_checkpoint.json",
        RECOVERY_CONTROL / "GO_MASSIVE_UNION_WAVE1",
        RECOVERY_CONTROL / "STOPPED_MASSIVE_UNION_WAVE1",
    ):
        if os.path.lexists(path):
            raise ValueError("downstream recovery evaluation began before GPU seal")
    payload = write_or_audit(GPU_MANIFEST, gpu_body())
    print(GPU_MANIFEST)
    return payload


def audit_gpu():
    observed = load_json(GPU_MANIFEST)
    verify_seal(observed, GPU_MANIFEST)
    expected = gpu_body()
    expected["created_at"] = observed.get("created_at")
    if observed != seal(expected):
        raise ValueError("medical recovery GPU manifest differs")
    return observed


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-prep")
    commands.add_parser("write-auth")
    commands.add_parser("audit-held")
    verify = commands.add_parser("verify-job")
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--time-limit", required=True)
    commands.add_parser("write-gpu")
    commands.add_parser("audit-gpu")
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "write-prep":
        command_write_prep()
    elif args.command == "write-auth":
        command_write_auth()
    elif args.command == "audit-held":
        command_audit_held()
    elif args.command == "verify-job":
        command_verify_job(args.job_id, args.time_limit)
    elif args.command == "write-gpu":
        command_write_gpu()
    elif args.command == "audit-gpu":
        payload = audit_gpu()
        print("VALID_MEDICAL_RECOVERY_V1: " + payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
