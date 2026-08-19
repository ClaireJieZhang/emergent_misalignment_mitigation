#!/usr/bin/env python3
"""Recover the one sealed Wave-1 dispatch whose three jobs remain held.

This is not a general retry utility.  It is hard-bound to jobs 247697-247699,
the exact pre-release failure artifacts, and the scientific checkout at
e25d59d.  It never submits, cancels, or replaces a job.  Run it only from a
clean isolated checkout; the main checkout must remain byte-exact at e25d59d
until all three jobs are terminal.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RECOVERY_ID = "massive_medical_union_wave1_held_submit_recovery_v1"
MAIN_COMMIT = "e25d59d8c5ea30c49cec207f5cac140a2281a525"
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
MAIN_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_pilot_v1"
CONTROL_ROOT = OUTPUT_ROOT / "control"
DATA_ROOT = OUTPUT_ROOT / "data"
LOCAL_MODEL_SNAPSHOT = (
    TILLICUM_ROOT
    / "cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    / "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
)
BENEFIT_MODEL = (
    TILLICUM_ROOT
    / "outputs/massive_benefit_pilot_v1/model"
    / "massive_en_benefit_pilot_infrastructure_recovery_v1"
)
BENEFIT_CONTROL_ADAPTER = BENEFIT_MODEL / "checkpoint-30"
BENEFIT_CONTROL_MANIFEST = BENEFIT_MODEL / "MODEL_MANIFEST.json"

JOB_IDS = ("247697", "247698", "247699")
RELEASE_ORDER = ("247699", "247698", "247697")
SUBMIT_TIME = "2026-08-18T22:38:34"
WORK_DIR = os.fspath(MAIN_REPO)
TRAIN_COMMAND = os.fspath(
    MAIN_REPO / "scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch"
)
EVAL_COMMAND = os.fspath(
    MAIN_REPO
    / "scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch"
)

ORIGINAL_ARTIFACT_SHA256 = {
    "PREP_COMPLETE.json": "1d09d77a2449b3f9152814ef326b6eacf6c0a8314ab08c73e1867c8f8ce05ed1",
    "STAGED": "3ae3d584e82908e51d8c7366df204b33d5781290a2eb1b3c624837aae611159b",
    "STOPPED_submission": "674a97d9355b627c8ffaac0af6b2196d996bab4be79dc1c6d0ccf5402deadf0a",
    "WAVE1_SUBMISSION_ATTEMPT.tsv": "1b72862fb0094ab75c6086b0d039e400a8ede0c985b946ccc9562299408cf798",
    "WAVE1_SUBMISSION_LOCK/owner": "3972fa33698bf1eecd23e5670da5f396429f6f3a4df37db277b99a4242913888",
}
PREP_PAYLOAD_SHA256 = "8e4b571b70eb90d8a89ee174224e840589f3a2fb1dcd5465a6eb91e8f8bd2182"
DATA_MANIFEST_SHA256 = "279da5fe8db9b8f8268d4e98000beb77682cda8b8cc6c6b12d9bad2477dc168a"
DATA_MANIFEST_PAYLOAD_SHA256 = (
    "4d934394065bcd345080ffac879359e059ce4be33ca87520d8d570da8022562a"
)
SNAPSHOT_BINDING_SHA256 = (
    "36310d1529c1b5ebd0611276d76508fe39a6ea87a9b78a37223d5ff7b17466e6"
)
BENEFIT_CONTROL_FINGERPRINT = (
    "5c16fc3f3da56e41ae6931b0fe14fb161ba096c266826ae680b1927d8bfd014f"
)
MAIN_FILE_SHA256 = {
    "scripts/submit_massive_medical_union_wave1_tillicum.sh": (
        "74f3315c295de7c3d3edfd2e1e1bf17375efef09df78988405abf132d4202108"
    ),
    "scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch": (
        "3fbdfdee71598368f0193366727eb3e6d500c5c2105a347e7d16ecb79a807d70"
    ),
    "scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch": (
        "6c4cea5e9a9084b5154ffa17b8fcfb6101fae5b12132d4b474164e81539cfb0d"
    ),
    "scripts/audit_massive_medical_union_tillicum_workflow.py": (
        "a65edf5800a363c0f09643b631ae9a24f34e92ca6d74ac4e71d8a973579bed9a"
    ),
}
RECOVERY_COMMIT_NAME_STATUS = {
    ("M", "docs/massive_medical_union_pilot_protocol.md"),
    ("M", "scripts/status_massive_medical_union_pilot_tillicum.sh"),
    ("M", "scripts/submit_massive_medical_union_wave1_tillicum.sh"),
    ("A", "scripts/recover_massive_medical_union_wave1_held_submit_tillicum.py"),
    ("A", "tests/test_massive_medical_union_held_submit_recovery.py"),
}

JOB_SPECS = {
    "247697": {
        "stage": "train_A",
        "job_name": "mmu_train_A",
        "minutes": 30,
        "memory": "200G",
        "command": TRAIN_COMMAND,
        "stdout": os.fspath(
            TILLICUM_ROOT / "outputs/logs/massive_medical_union_train_247697.out"
        ),
        "stderr": os.fspath(
            TILLICUM_ROOT / "outputs/logs/massive_medical_union_train_247697.err"
        ),
        "dependency_ids": (),
        "submit_line": (
            "sbatch --parsable --hold --export=NONE --job-name=mmu_train_A "
            "scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch"
        ),
        "spooled_script_sha256": (
            "3fbdfdee71598368f0193366727eb3e6d500c5c2105a347e7d16ecb79a807d70"
        ),
    },
    "247698": {
        "stage": "train_B1",
        "job_name": "mmu_train_B1",
        "minutes": 30,
        "memory": "200G",
        "command": TRAIN_COMMAND,
        "stdout": os.fspath(
            TILLICUM_ROOT / "outputs/logs/massive_medical_union_train_247698.out"
        ),
        "stderr": os.fspath(
            TILLICUM_ROOT / "outputs/logs/massive_medical_union_train_247698.err"
        ),
        "dependency_ids": (),
        "submit_line": (
            "sbatch --parsable --hold --export=NONE --job-name=mmu_train_B1 "
            "scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch"
        ),
        "spooled_script_sha256": (
            "3fbdfdee71598368f0193366727eb3e6d500c5c2105a347e7d16ecb79a807d70"
        ),
    },
    "247699": {
        "stage": "evaluate",
        "job_name": "mmu_w1_eval",
        "minutes": 20,
        "memory": "180G",
        "command": EVAL_COMMAND,
        "stdout": os.fspath(
            TILLICUM_ROOT
            / "outputs/logs/massive_medical_union_wave1_evaluate_247699.out"
        ),
        "stderr": os.fspath(
            TILLICUM_ROOT
            / "outputs/logs/massive_medical_union_wave1_evaluate_247699.err"
        ),
        "dependency_ids": ("247697", "247698"),
        "submit_line": (
            "sbatch --parsable --hold --export=NONE --job-name=mmu_w1_eval "
            "--dependency=afterok:247697:247698 --kill-on-invalid-dep=yes "
            "scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch"
        ),
        "spooled_script_sha256": (
            "6c4cea5e9a9084b5154ffa17b8fcfb6101fae5b12132d4b474164e81539cfb0d"
        ),
    },
}

FIELD_RE = re.compile(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=")


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(body):
    payload = dict(body)
    payload["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return payload


def verify_sealed(payload):
    if not isinstance(payload, dict):
        raise ValueError("sealed record is not an object")
    observed = payload.get("payload_sha256")
    body = dict(payload)
    body.pop("payload_sha256", None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError("sealed record payload hash differs")
    return payload


def parse_scontrol_line(line):
    """Parse `scontrol show job -o` without splitting SubmitLine on spaces."""
    if not isinstance(line, str) or not line.strip():
        raise ValueError("empty scontrol job record")
    matches = list(FIELD_RE.finditer(line.strip()))
    if not matches or matches[0].group(1) != "JobId":
        raise ValueError("scontrol job record does not start with JobId")
    result = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in result:
            raise ValueError(f"scontrol job record repeats {key}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        result[key] = line[match.end():end].strip()
    return result


def normalize_one_node(value):
    if value not in {"1", "1-1"}:
        raise ValueError(f"job is not exactly one node: {value!r}")
    return 1


def normalize_dependency(value, held_preflight=False):
    if value in {"", "(null)"}:
        return ()
    compact = re.fullmatch(r"afterok:([0-9]+(?::[0-9]+)+)", value)
    if compact is not None:
        return tuple(compact.group(1).split(":"))
    result = []
    annotations = []
    for term in value.split(","):
        match = re.fullmatch(r"afterok:([0-9]+)(?:\((unfulfilled)\))?", term)
        if match is None:
            raise ValueError(f"dependency has an unauthorized representation: {term}")
        result.append(match.group(1))
        annotations.append(match.group(2))
    if held_preflight and any(annotations) and not all(
        annotation == "unfulfilled" for annotation in annotations
    ):
        raise ValueError("held dependency mixes annotated and unannotated terms")
    return tuple(result)


def parse_tres(value):
    result = {}
    for term in value.split(","):
        if "=" not in term:
            raise ValueError(f"invalid TRES term: {term}")
        key, item = term.split("=", 1)
        if not key or key in result or not item:
            raise ValueError(f"invalid or repeated TRES term: {term}")
        result[key] = item
    return result


def run_checked(command, runner=subprocess.run):
    return runner(command, check=True, text=True, capture_output=True)


def query_job(job_id, runner=subprocess.run):
    completed = run_checked(["scontrol", "show", "job", job_id, "-o"], runner)
    return completed.stdout.strip(), parse_scontrol_line(completed.stdout)


def query_spooled_script(job_id, runner=subprocess.run):
    completed = runner(
        ["scontrol", "write", "batch_script", job_id, "-"],
        check=True,
        capture_output=True,
    )
    stdout = completed.stdout
    if isinstance(stdout, str):
        stdout = stdout.encode("utf-8")
    if not stdout:
        raise ValueError(f"empty spooled script for held job {job_id}")
    return stdout


def audit_held_job(job_id, raw, fields):
    spec = JOB_SPECS[job_id]
    expected_tres = {
        "cpu": "8", "mem": spec["memory"], "node": "1", "billing": "8",
        "gres/gpu": "1", "gres/gpu:h200": "1",
    }
    exact = {
        "JobId": job_id,
        "JobName": spec["job_name"],
        "UserId": "claizhan(1033174)",
        "GroupId": "all(226269)",
        "Priority": "0",
        "Account": "stf",
        "QOS": "normal",
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Requeue": "0",
        "Restarts": "0",
        "BatchFlag": "1",
        "Reboot": "0",
        "ExitCode": "0:0",
        "RunTime": "00:00:00",
        "TimeLimit": f"00:{spec['minutes']:02d}:00",
        "TimeMin": "N/A",
        "SubmitTime": SUBMIT_TIME,
        "EligibleTime": "Unknown",
        "Partition": "gpu-h200",
        "AllocNode:Sid": "tillicum-login02:208261",
        "ReqNodeList": "(null)",
        "ExcNodeList": "(null)",
        "NodeList": "",
        "NumCPUs": "8",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "AllocTRES": "(null)",
        "MinCPUsNode": "8",
        "MinMemoryNode": spec["memory"],
        "Command": spec["command"],
        "SubmitLine": spec["submit_line"],
        "WorkDir": WORK_DIR,
        "StdErr": spec["stderr"],
        "StdIn": "/dev/null",
        "StdOut": spec["stdout"],
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            raise ValueError(
                f"held job {job_id} differs on {key}: {fields.get(key)!r} != {expected!r}"
            )
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError(f"held job {job_id} unexpectedly belongs to an array/het job")
    if normalize_one_node(fields.get("NumNodes")) != 1:
        raise AssertionError("unreachable node normalization")
    dependencies = normalize_dependency(
        fields.get("Dependency", ""), held_preflight=bool(spec["dependency_ids"])
    )
    if dependencies != spec["dependency_ids"]:
        raise ValueError(f"held job {job_id} dependencies differ")
    if parse_tres(fields.get("ReqTRES", "")) != expected_tres:
        raise ValueError(f"held job {job_id} requested resources differ")
    if job_id == "247699" and fields.get("KillOnInvalidDependent") != "Yes":
        raise ValueError("evaluation job does not kill an invalid dependency")
    if job_id != "247699" and "KillOnInvalidDependent" in fields:
        raise ValueError("training job unexpectedly has KillOnInvalidDependent")
    for path in (spec["stdout"], spec["stderr"]):
        if os.path.lexists(path):
            raise ValueError(f"held job already created a log/allocation artifact: {path}")
    return {
        "stage": spec["stage"],
        "job_id": job_id,
        "max_minutes": spec["minutes"],
        "job_name": spec["job_name"],
        "command": spec["command"],
        "work_dir": WORK_DIR,
        "stdout": spec["stdout"],
        "stderr": spec["stderr"],
        "normalized_nodes": 1,
        "raw_num_nodes": fields["NumNodes"],
        "normalized_dependency_ids": list(dependencies),
        "raw_dependency": fields.get("Dependency"),
        "requested_tres": expected_tres,
        "scontrol_record_sha256": sha256_bytes(raw.strip().encode("utf-8")),
    }


def audit_spooled_script(job_id, script_bytes):
    expected = JOB_SPECS[job_id]["spooled_script_sha256"]
    observed = sha256_bytes(script_bytes)
    if observed != expected:
        raise ValueError(f"held job {job_id} spooled script bytes differ")
    return observed


def require_regular_hash(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required artifact is missing or unsafe: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"required artifact hash differs: {path}")
    return observed


def git_text(repo, *args):
    return subprocess.check_output(["git", "-C", os.fspath(repo), *args], text=True).strip()


def audit_recovery_commit_shape(recovery_repo, main_commit=MAIN_COMMIT):
    recovery_commit = git_text(recovery_repo, "rev-parse", "HEAD")
    parents = git_text(
        recovery_repo, "rev-list", "--parents", "-n", "1", recovery_commit
    ).split()
    if parents != [recovery_commit, main_commit]:
        raise ValueError(
            "recovery commit must be a nonmerge direct child of the scientific commit"
        )
    raw = git_text(
        recovery_repo,
        "diff",
        "--name-status",
        "--no-renames",
        f"{main_commit}..{recovery_commit}",
    )
    observed = []
    for line in raw.splitlines():
        columns = line.split("\t")
        if len(columns) != 2 or columns[0] not in {"A", "M", "D", "T", "U", "X", "B"}:
            raise ValueError("recovery commit has an invalid name-status record")
        observed.append((columns[0], columns[1]))
    if len(observed) != len(RECOVERY_COMMIT_NAME_STATUS) or set(
        observed
    ) != RECOVERY_COMMIT_NAME_STATUS:
        raise ValueError("recovery commit changes files outside the exact five-path allowlist")
    return recovery_commit


def audit_repositories(recovery_repo):
    recovery_repo = recovery_repo.resolve()
    if recovery_repo == MAIN_REPO.resolve():
        raise ValueError("recovery must run from an isolated checkout, not the main job checkout")
    if git_text(MAIN_REPO, "rev-parse", "HEAD") != MAIN_COMMIT:
        raise ValueError("main job checkout moved away from the PREP commit")
    if git_text(MAIN_REPO, "status", "--porcelain"):
        raise ValueError("main job checkout is dirty")
    for relative, expected in MAIN_FILE_SHA256.items():
        require_regular_hash(MAIN_REPO / relative, expected)
    if git_text(recovery_repo, "status", "--porcelain"):
        raise ValueError("isolated recovery checkout is dirty")
    recovery_commit = audit_recovery_commit_shape(recovery_repo)
    script = Path(__file__).resolve()
    if not str(script).startswith(str(recovery_repo) + os.sep):
        raise ValueError("recovery script is not inside its declared isolated checkout")
    return {
        "recovery_repo": os.fspath(recovery_repo),
        "recovery_commit": recovery_commit,
        "recovery_script": os.fspath(script),
        "recovery_script_sha256": sha256_file(script),
    }


def audit_original_control():
    expected_entries = {
        "PREP_COMPLETE.json", "STAGED", "STOPPED_submission",
        "WAVE1_SUBMISSION_ATTEMPT.tsv", "WAVE1_SUBMISSION_LOCK",
    }
    if CONTROL_ROOT.is_symlink() or not CONTROL_ROOT.is_dir():
        raise ValueError("control root is missing or unsafe")
    if {entry.name for entry in CONTROL_ROOT.iterdir()} != expected_entries:
        raise ValueError("control root differs from the exact failed-submit inventory")
    lock = CONTROL_ROOT / "WAVE1_SUBMISSION_LOCK"
    if lock.is_symlink() or not lock.is_dir() or {item.name for item in lock.iterdir()} != {"owner"}:
        raise ValueError("original submission lock inventory differs")
    observed = {}
    for relative, expected in ORIGINAL_ARTIFACT_SHA256.items():
        observed[relative] = require_regular_hash(CONTROL_ROOT / relative, expected)
    attempt = (CONTROL_ROOT / "WAVE1_SUBMISSION_ATTEMPT.tsv").read_text()
    if attempt != (
        "stage\tjob_id\ntrain_A\t247697\ntrain_B1\t247698\nevaluate\t247699\n"
    ):
        raise ValueError("original submission attempt does not bind the exact jobs")
    stopped = (CONTROL_ROOT / "STOPPED_submission").read_text()
    required_stop = {
        "stage=submission", "exit_status=1", "recorded_jobs=3",
        "release_started=false", "hold_requested_on_failure=true",
        "no_retry_authorized=true",
    }
    if not required_stop.issubset(set(stopped.splitlines())):
        raise ValueError("original STOP does not certify a pre-release held failure")
    owner = (lock / "owner").read_text()
    required_owner = {
        f"repo_commit={MAIN_COMMIT}", "owner_pid=208261",
        "hard_max_h200_minutes=80", "hard_max_cost_usd=1.20",
    }
    if not required_owner.issubset(set(owner.splitlines())):
        raise ValueError("original submission lock owner differs")
    prep = json.loads((CONTROL_ROOT / "PREP_COMPLETE.json").read_text())
    if (
        prep.get("repo_commit") != MAIN_COMMIT
        or prep.get("payload_sha256")
        != PREP_PAYLOAD_SHA256
        or prep.get("wave1_h200_minutes") != 80
        or prep.get("wave1_max_cost_usd") != 1.2
        or prep.get("data_manifest", {}).get("sha256") != DATA_MANIFEST_SHA256
        or prep.get("data_manifest", {}).get("payload_sha256")
        != DATA_MANIFEST_PAYLOAD_SHA256
        or prep.get("local_model_snapshot", {}).get("snapshot_binding_sha256")
        != SNAPSHOT_BINDING_SHA256
        or prep.get("benefit_control", {}).get("adapter_fingerprint")
        != BENEFIT_CONTROL_FINGERPRINT
    ):
        raise ValueError("PREP scientific/budget binding differs")
    audit_empty_pre_release_outputs()
    return observed


def audit_empty_pre_release_outputs():
    for path in (OUTPUT_ROOT / "models", OUTPUT_ROOT / "evaluation/wave1"):
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise ValueError(f"pre-release output directory is not exactly empty: {path}")


def audit_held_context(recovery_repo, runner=subprocess.run):
    repositories = audit_repositories(recovery_repo)
    originals = audit_original_control()
    jobs = []
    for job_id in JOB_IDS:
        raw, fields = query_job(job_id, runner)
        record = audit_held_job(job_id, raw, fields)
        record["spooled_script_sha256"] = audit_spooled_script(
            job_id, query_spooled_script(job_id, runner)
        )
        jobs.append(record)
    return {"repositories": repositories, "original_artifacts": originals, "jobs": jobs}


def jobs_bytes():
    return (
        "stage\tjob_id\tmax_minutes\treleased\n"
        "train_A\t247697\t30\ttrue\n"
        "train_B1\t247698\t30\ttrue\n"
        "evaluate\t247699\t20\ttrue\n"
    ).encode("utf-8")


def atomic_write_once(path, content, mode=0o600):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace recovery artifact: {path}")
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


def atomic_json_once(path, body):
    payload = sealed(body)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_once(path, content)
    return payload


def build_authorization(staging):
    jobs_path = staging / "wave1_jobs.tsv"
    auth_path = staging / "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json"
    atomic_write_once(jobs_path, jobs_bytes(), 0o400)
    command = [
        sys.executable,
        os.fspath(MAIN_REPO / "scripts/audit_massive_medical_union_tillicum_workflow.py"),
        "write-auth",
        "--repo-root", os.fspath(MAIN_REPO),
        "--data-root", os.fspath(DATA_ROOT),
        "--local-model-snapshot", os.fspath(LOCAL_MODEL_SNAPSHOT),
        "--prep-file", os.fspath(CONTROL_ROOT / "PREP_COMPLETE.json"),
        "--benefit-control-manifest", os.fspath(BENEFIT_CONTROL_MANIFEST),
        "--benefit-control-adapter", os.fspath(BENEFIT_CONTROL_ADAPTER),
        "--jobs-file", os.fspath(jobs_path),
        "--output-file", os.fspath(auth_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    payload = json.loads(auth_path.read_text())
    verify_sealed(payload)
    expected_jobs = [
        {"stage": "train_A", "job_id": "247697", "max_minutes": 30, "released": True},
        {"stage": "train_B1", "job_id": "247698", "max_minutes": 30, "released": True},
        {"stage": "evaluate", "job_id": "247699", "max_minutes": 20, "released": True},
    ]
    if (
        payload.get("repo_commit") != MAIN_COMMIT
        or payload.get("prep_file_sha256")
        != ORIGINAL_ARTIFACT_SHA256["PREP_COMPLETE.json"]
        or payload.get("jobs_file_sha256") != sha256_file(jobs_path)
        or payload.get("maximum_h200_minutes") != 80
        or payload.get("maximum_cost_usd") != 1.2
        or payload.get("jobs") != expected_jobs
        or payload.get("no_requeue") is not True
        or payload.get("no_retry_or_reserve") is not True
        or payload.get("released_wave") != 1
        or payload.get("wave2_jobs_submitted") is not False
        or payload.get("quorum_jobs_submitted") is not False
    ):
        raise ValueError("main auditor produced an unexpected authorization")
    return jobs_path, auth_path


def amendment_body(context, jobs_path, auth_path):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": now,
        "reason": (
            "original held-job audit rejected semantically exact Slurm "
            "NumNodes/dependency renderings before any release"
        ),
        "main_scientific_commit": MAIN_COMMIT,
        "recovery_implementation": context["repositories"],
        "original_failure_artifacts": context["original_artifacts"],
        "jobs": context["jobs"],
        "canonical_jobs_sha256": sha256_file(jobs_path),
        "canonical_authorization_sha256": sha256_file(auth_path),
        "release_order": list(RELEASE_ORDER),
        "normalization": {
            "NumNodes=1-1": "exactly one node",
            "afterok:ID(unfulfilled),afterok:ID(unfulfilled)": (
                "same ordered afterok dependency IDs as submitted"
            ),
            "scientific_design_or_resource_change": False,
        },
        "budget": {
            "maximum_h200_minutes": 80,
            "maximum_gpu_cost_usd": 1.2,
            "new_jobs_submitted": 0,
            "prior_gpu_allocation_minutes": 0,
        },
        "constraints": {
            "preserve_original_submission_stop": True,
            "preserve_original_submission_lock": True,
            "no_cancel_or_resubmit": True,
            "no_retry_or_reserve": True,
            "release_existing_jobs_only": True,
            "downstream_released_first": True,
            "main_checkout_frozen_until_jobs_terminal": True,
            "wave2_or_quorum_submitted": False,
        },
    }


def submitted_bytes(amendment_sha, recovered_at):
    return (
        f"submitted_at={recovered_at}\n"
        f"original_slurm_submit_time={SUBMIT_TIME}\n"
        f"recovered_at={recovered_at}\n"
        f"repo_commit={MAIN_COMMIT}\n"
        "train_A_job=247697\ntrain_B1_job=247698\nevaluate_job=247699\n"
        "held_first=true\nhard_max_h200_minutes=80\nhard_max_cost_usd=1.20\n"
        "wave2_jobs_submitted=false\nquorum_jobs_submitted=false\n"
        "held_submit_recovery=true\noriginal_submission_stop_preserved=true\n"
        f"recovery_amendment_payload_sha256={amendment_sha}\n"
    ).encode()


def audit_recovery_pre_release_records():
    expected_entries = {
        "PREP_COMPLETE.json", "STAGED", "STOPPED_submission",
        "WAVE1_SUBMISSION_ATTEMPT.tsv", "WAVE1_SUBMISSION_LOCK",
        "HELD_SUBMIT_RECOVERY_LOCK", "wave1_jobs.tsv",
        "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json",
        "HELD_SUBMIT_RECOVERY_AMENDMENT.json", "WAVE1_SUBMITTED",
    }
    if CONTROL_ROOT.is_symlink() or not CONTROL_ROOT.is_dir():
        raise ValueError("pre-release recovery control root is missing or unsafe")
    if {entry.name for entry in CONTROL_ROOT.iterdir()} != expected_entries:
        raise ValueError("pre-release recovery control inventory differs")
    recovery_lock = CONTROL_ROOT / "HELD_SUBMIT_RECOVERY_LOCK"
    if (
        recovery_lock.is_symlink()
        or not recovery_lock.is_dir()
        or {item.name for item in recovery_lock.iterdir()} != {"owner.json"}
    ):
        raise ValueError("held-submit recovery lock inventory differs")
    owner = load_verified_json(recovery_lock / "owner.json")
    if (
        owner.get("recovery_id") != RECOVERY_ID
        or owner.get("main_scientific_commit") != MAIN_COMMIT
        or owner.get("job_ids") != list(JOB_IDS)
        or owner.get("original_artifact_sha256") != ORIGINAL_ARTIFACT_SHA256
    ):
        raise ValueError("held-submit recovery lock owner differs")
    jobs_path = CONTROL_ROOT / "wave1_jobs.tsv"
    auth_path = CONTROL_ROOT / "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json"
    amendment = load_verified_json(
        CONTROL_ROOT / "HELD_SUBMIT_RECOVERY_AMENDMENT.json"
    )
    load_verified_json(auth_path)
    if (
        jobs_path.is_symlink()
        or jobs_path.read_bytes() != jobs_bytes()
        or amendment.get("canonical_jobs_sha256") != sha256_file(jobs_path)
        or amendment.get("canonical_authorization_sha256") != sha256_file(auth_path)
        or amendment.get("original_failure_artifacts") != ORIGINAL_ARTIFACT_SHA256
    ):
        raise ValueError("canonical pre-release recovery records differ")
    submitted = parse_key_value_file(CONTROL_ROOT / "WAVE1_SUBMITTED")
    if submitted.get("recovery_amendment_payload_sha256") != amendment.get(
        "payload_sha256"
    ):
        raise ValueError("WAVE1_SUBMITTED does not bind the recovery amendment")


def recovery_stop(error, runner=subprocess.run):
    hold_results = []
    for job_id in JOB_IDS:
        result = runner(["scontrol", "hold", job_id], text=True, capture_output=True)
        hold_results.append({"job_id": job_id, "returncode": result.returncode})
    job_states = []
    for job_id in JOB_IDS:
        try:
            raw, fields = query_job(job_id, runner)
            job_states.append({
                "job_id": job_id,
                "state": fields.get("JobState"),
                "reason": fields.get("Reason"),
                "run_time": fields.get("RunTime"),
                "alloc_tres": fields.get("AllocTRES"),
                "scontrol_record_sha256": sha256_bytes(raw.strip().encode("utf-8")),
            })
        except BaseException as query_error:
            job_states.append({
                "job_id": job_id,
                "query_error_type": type(query_error).__name__,
                "query_error_sha256": sha256_bytes(str(query_error).encode()),
            })
    path = CONTROL_ROOT / "STOPPED_held_submit_recovery.json"
    if not path.exists():
        atomic_json_once(path, {
            "schema_version": 1,
            "recovery_id": RECOVERY_ID,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message_sha256": sha256_bytes(str(error).encode()),
            "hold_requests": hold_results,
            "job_states_after_hold_requests": job_states,
            "release_order": list(RELEASE_ORDER),
            "no_retry_authorized": True,
        })


def confirm_released(runner=subprocess.run):
    observed = []
    for job_id in JOB_IDS:
        raw, fields = query_job(job_id, runner)
        state = fields.get("JobState")
        if state not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}:
            raise ValueError(f"released job {job_id} entered unexpected state {state}")
        if fields.get("Reason") == "JobHeldUser":
            raise ValueError(f"released job {job_id} remains user-held")
        if job_id == "247699" and normalize_dependency(
            fields.get("Dependency", ""), held_preflight=False
        ) != ("247697", "247698"):
            raise ValueError("released evaluation dependency changed")
        observed.append({
            "job_id": job_id,
            "state": state,
            "reason": fields.get("Reason"),
            "scontrol_record_sha256": sha256_bytes(raw.strip().encode()),
        })
    return observed


def release_existing_jobs(runner=subprocess.run):
    for job_id in RELEASE_ORDER:
        run_checked(["scontrol", "release", job_id], runner)
    return confirm_released(runner)


def recover(context, runner=subprocess.run):
    lock = CONTROL_ROOT / "HELD_SUBMIT_RECOVERY_LOCK"
    os.mkdir(lock, 0o700)
    atomic_json_once(lock / "owner.json", {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "main_scientific_commit": MAIN_COMMIT,
        "recovery_commit": context["repositories"]["recovery_commit"],
        "job_ids": list(JOB_IDS),
        "original_artifact_sha256": ORIGINAL_ARTIFACT_SHA256,
    })
    staging = CONTROL_ROOT / f".held-submit-recovery-build-{os.getpid()}"
    os.mkdir(staging, 0o700)
    try:
        jobs_path, auth_path = build_authorization(staging)
        amendment = atomic_json_once(
            staging / "HELD_SUBMIT_RECOVERY_AMENDMENT.json",
            amendment_body(context, jobs_path, auth_path),
        )
        recovered_at = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write_once(
            staging / "WAVE1_SUBMITTED",
            submitted_bytes(amendment["payload_sha256"], recovered_at),
        )
        for filename in (
            "wave1_jobs.tsv", "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json",
            "HELD_SUBMIT_RECOVERY_AMENDMENT.json", "WAVE1_SUBMITTED",
        ):
            destination = CONTROL_ROOT / filename
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"canonical recovery destination exists: {destination}")
            os.replace(staging / filename, destination)
        staging.rmdir()

        # Recheck every source binding and held job after durable authorization,
        # immediately before the first release.
        for relative, expected in ORIGINAL_ARTIFACT_SHA256.items():
            require_regular_hash(CONTROL_ROOT / relative, expected)
        if git_text(MAIN_REPO, "rev-parse", "HEAD") != MAIN_COMMIT or git_text(
            MAIN_REPO, "status", "--porcelain"
        ):
            raise ValueError("main scientific checkout changed before release")
        audit_recovery_pre_release_records()
        audit_empty_pre_release_outputs()
        for job_id in JOB_IDS:
            raw, fields = query_job(job_id, runner)
            audit_held_job(job_id, raw, fields)
            audit_spooled_script(job_id, query_spooled_script(job_id, runner))

        released_jobs = release_existing_jobs(runner)
        released_at = dt.datetime.now(dt.timezone.utc).isoformat()
        released_content = (
            f"released_at={released_at}\n"
            "release_order=247699,247698,247697\n"
            "hard_max_h200_minutes=80\nhard_max_cost_usd=1.20\n"
            "no_retry_authorized=true\nwave2_jobs_submitted=false\n"
            "quorum_jobs_submitted=false\nheld_submit_recovery=true\n"
            "original_submission_stop_preserved=true\n"
            f"recovery_amendment_payload_sha256={amendment['payload_sha256']}\n"
            f"release_observation_sha256={sha256_bytes(canonical_bytes(released_jobs))}\n"
        ).encode()
        atomic_write_once(
            CONTROL_ROOT / "WAVE1_RELEASED",
            released_content,
        )
        atomic_json_once(
            CONTROL_ROOT / "HELD_SUBMIT_RECOVERY_COMPLETE.json",
            {
                "schema_version": 1,
                "recovery_id": RECOVERY_ID,
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "main_scientific_commit": MAIN_COMMIT,
                "job_ids": list(JOB_IDS),
                "release_order": list(RELEASE_ORDER),
                "released_jobs": released_jobs,
                "release_observation_sha256": sha256_bytes(
                    canonical_bytes(released_jobs)
                ),
                "wave1_released_sha256": sha256_bytes(released_content),
                "recovery_amendment_payload_sha256": amendment["payload_sha256"],
                "original_submission_stop_preserved": True,
                "new_jobs_submitted": 0,
                "additional_h200_minutes_authorized": 0,
            },
        )
        return amendment
    except BaseException as error:
        recovery_stop(error, runner)
        raise


def parse_key_value_file(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise ValueError(f"invalid key/value recovery record: {path}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"repeated/empty key in recovery record: {path}")
        result[key] = value
    return result


def load_verified_json(path):
    if Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError(f"missing or unsafe sealed recovery record: {path}")
    return verify_sealed(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_recovery_transition(control_root=CONTROL_ROOT):
    control_root = Path(control_root)
    for relative, expected in ORIGINAL_ARTIFACT_SHA256.items():
        require_regular_hash(control_root / relative, expected)

    jobs_path = control_root / "wave1_jobs.tsv"
    if jobs_path.read_bytes() != jobs_bytes():
        raise ValueError("recovered canonical jobs.tsv differs")
    amendment = load_verified_json(
        control_root / "HELD_SUBMIT_RECOVERY_AMENDMENT.json"
    )
    auth_path = control_root / "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json"
    auth = load_verified_json(auth_path)
    expected_jobs = [
        {"stage": "train_A", "job_id": "247697", "max_minutes": 30, "released": True},
        {"stage": "train_B1", "job_id": "247698", "max_minutes": 30, "released": True},
        {"stage": "evaluate", "job_id": "247699", "max_minutes": 20, "released": True},
    ]
    amendment_jobs = amendment.get("jobs", [])
    if len(amendment_jobs) != 3:
        raise ValueError("held-submit recovery amendment job count differs")
    for observed, job_id in zip(amendment_jobs, JOB_IDS):
        spec = JOB_SPECS[job_id]
        expected_tres = {
            "cpu": "8", "mem": spec["memory"], "node": "1", "billing": "8",
            "gres/gpu": "1", "gres/gpu:h200": "1",
        }
        expected_dependency = list(spec["dependency_ids"])
        if (
            observed.get("stage") != spec["stage"]
            or observed.get("job_id") != job_id
            or observed.get("max_minutes") != spec["minutes"]
            or observed.get("job_name") != spec["job_name"]
            or observed.get("command") != spec["command"]
            or observed.get("work_dir") != WORK_DIR
            or observed.get("stdout") != spec["stdout"]
            or observed.get("stderr") != spec["stderr"]
            or observed.get("normalized_nodes") != 1
            or observed.get("raw_num_nodes") not in {"1", "1-1"}
            or observed.get("normalized_dependency_ids") != expected_dependency
            or normalize_dependency(
                observed.get("raw_dependency", ""), held_preflight=bool(expected_dependency)
            )
            != tuple(expected_dependency)
            or observed.get("requested_tres") != expected_tres
            or observed.get("spooled_script_sha256")
            != spec["spooled_script_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", observed.get("scontrol_record_sha256", ""))
            is None
        ):
            raise ValueError(f"held-submit recovery amendment job {job_id} differs")
    if (
        amendment.get("schema_version") != 1
        or amendment.get("recovery_id") != RECOVERY_ID
        or amendment.get("main_scientific_commit") != MAIN_COMMIT
        or amendment.get("original_failure_artifacts") != ORIGINAL_ARTIFACT_SHA256
        or amendment.get("canonical_jobs_sha256") != sha256_file(jobs_path)
        or amendment.get("canonical_authorization_sha256") != sha256_file(auth_path)
        or amendment.get("release_order") != list(RELEASE_ORDER)
        or [item.get("job_id") for item in amendment_jobs] != list(JOB_IDS)
        or amendment.get("budget", {}).get("maximum_h200_minutes") != 80
        or amendment.get("budget", {}).get("maximum_gpu_cost_usd") != 1.2
        or amendment.get("budget", {}).get("new_jobs_submitted") != 0
        or amendment.get("budget", {}).get("prior_gpu_allocation_minutes") != 0
        or amendment.get("constraints", {}).get("preserve_original_submission_stop")
        is not True
        or amendment.get("constraints", {}).get("no_cancel_or_resubmit") is not True
        or re.fullmatch(
            r"[0-9a-f]{40}",
            amendment.get("recovery_implementation", {}).get("recovery_commit", ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            amendment.get("recovery_implementation", {}).get(
                "recovery_script_sha256", ""
            ),
        )
        is None
    ):
        raise ValueError("held-submit recovery amendment differs")
    if (
        auth.get("schema_version") != 1
        or auth.get("repo_commit") != MAIN_COMMIT
        or auth.get("prep_file_sha256")
        != ORIGINAL_ARTIFACT_SHA256["PREP_COMPLETE.json"]
        or auth.get("jobs_file_sha256") != sha256_file(jobs_path)
        or auth.get("jobs") != expected_jobs
        or auth.get("maximum_h200_minutes") != 80
        or auth.get("maximum_cost_usd") != 1.2
        or auth.get("no_requeue") is not True
        or auth.get("no_retry_or_reserve") is not True
        or auth.get("released_wave") != 1
        or auth.get("wave2_jobs_submitted") is not False
        or auth.get("quorum_jobs_submitted") is not False
    ):
        raise ValueError("recovered Wave-1 authorization differs")

    submitted = parse_key_value_file(control_root / "WAVE1_SUBMITTED")
    released_path = control_root / "WAVE1_RELEASED"
    released = parse_key_value_file(released_path)
    amendment_sha = amendment["payload_sha256"]
    common = {
        "repo_commit": MAIN_COMMIT,
        "train_A_job": "247697",
        "train_B1_job": "247698",
        "evaluate_job": "247699",
        "held_first": "true",
        "hard_max_h200_minutes": "80",
        "hard_max_cost_usd": "1.20",
        "wave2_jobs_submitted": "false",
        "quorum_jobs_submitted": "false",
        "held_submit_recovery": "true",
        "original_submission_stop_preserved": "true",
        "recovery_amendment_payload_sha256": amendment_sha,
    }
    if any(submitted.get(key) != value for key, value in common.items()):
        raise ValueError("recovered WAVE1_SUBMITTED differs")
    if submitted.get("original_slurm_submit_time") != SUBMIT_TIME:
        raise ValueError("original Slurm submit time differs")
    if set(submitted) != set(common) | {
        "submitted_at", "original_slurm_submit_time", "recovered_at"
    }:
        raise ValueError("recovered WAVE1_SUBMITTED key set differs")
    released_common = {
        "release_order": ",".join(RELEASE_ORDER),
        "hard_max_h200_minutes": "80",
        "hard_max_cost_usd": "1.20",
        "no_retry_authorized": "true",
        "wave2_jobs_submitted": "false",
        "quorum_jobs_submitted": "false",
        "held_submit_recovery": "true",
        "original_submission_stop_preserved": "true",
        "recovery_amendment_payload_sha256": amendment_sha,
    }
    if any(released.get(key) != value for key, value in released_common.items()):
        raise ValueError("recovered WAVE1_RELEASED differs")
    if set(released) != set(released_common) | {
        "released_at", "release_observation_sha256"
    }:
        raise ValueError("recovered WAVE1_RELEASED key set differs")

    complete = load_verified_json(
        control_root / "HELD_SUBMIT_RECOVERY_COMPLETE.json"
    )
    released_jobs = complete.get("released_jobs", [])
    if (
        complete.get("schema_version") != 1
        or complete.get("recovery_id") != RECOVERY_ID
        or complete.get("main_scientific_commit") != MAIN_COMMIT
        or complete.get("job_ids") != list(JOB_IDS)
        or complete.get("release_order") != list(RELEASE_ORDER)
        or complete.get("wave1_released_sha256") != sha256_file(released_path)
        or complete.get("recovery_amendment_payload_sha256") != amendment_sha
        or complete.get("original_submission_stop_preserved") is not True
        or complete.get("new_jobs_submitted") != 0
        or complete.get("additional_h200_minutes_authorized") != 0
        or complete.get("release_observation_sha256")
        != sha256_bytes(canonical_bytes(released_jobs))
        or [item.get("job_id") for item in released_jobs] != list(JOB_IDS)
        or any(
            item.get("state") not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
            for item in released_jobs
        )
        or any(item.get("reason") == "JobHeldUser" for item in released_jobs)
        or any(
            set(item) != {"job_id", "state", "reason", "scontrol_record_sha256"}
            or re.fullmatch(r"[0-9a-f]{64}", item.get("scontrol_record_sha256", ""))
            is None
            for item in released_jobs
        )
    ):
        raise ValueError("held-submit recovery completion differs")
    if released.get("release_observation_sha256") != complete.get(
        "release_observation_sha256"
    ):
        raise ValueError("release observation bindings differ")
    return complete


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-held")
    audit.add_argument("--ack-job-ids", required=True)
    execute = subparsers.add_parser("recover-held")
    execute.add_argument("--ack-job-ids", required=True)
    execute.add_argument("--ack-max-cost-usd", required=True)
    subparsers.add_parser("validate-transition")
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "validate-transition":
        complete = validate_recovery_transition()
        print("VALID_RECOVERED_HELD_SUBMIT: " + complete["payload_sha256"])
        return 0
    if args.ack_job_ids != ",".join(JOB_IDS):
        raise ValueError("--ack-job-ids must name exactly 247697,247698,247699")
    if args.command == "recover-held" and args.ack_max_cost_usd != "1.20":
        raise ValueError("--ack-max-cost-usd must be exactly 1.20")
    recovery_repo = Path(__file__).resolve().parents[1]
    context = audit_held_context(recovery_repo)
    if args.command == "audit-held":
        print(json.dumps(context, indent=2, sort_keys=True))
        return 0
    amendment = recover(context)
    print(
        "RECOVERED_AND_RELEASED_EXISTING_WAVE1_JOBS: "
        + amendment["payload_sha256"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
