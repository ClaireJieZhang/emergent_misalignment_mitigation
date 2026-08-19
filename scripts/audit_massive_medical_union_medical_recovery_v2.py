#!/usr/bin/env python3
"""Incident-bound audit for the second, same-science medical recovery.

Recovery v1 job 248197 stopped five seconds after allocation, before model
loading or output creation, because Tillicum represents requested memory with
``MinMemoryNode`` while held and with ``MinMemoryTRES``/``MemPerTres`` while
running.  This module reuses the reviewed v1 scientific/provenance auditor and
changes only the versioned namespaces, incident bindings, cumulative budget,
and phase-specific runtime normalization.
"""

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess


V1_AUDITOR_SHA256 = "9fd410ba9abe73b317a83c35b3a7ceaf2185c4adb4f5b294cf3c7213af371920"
V1_AUDITOR_PATH = Path(__file__).resolve().with_name(
    "audit_massive_medical_union_medical_recovery_v1.py"
)
if hashlib.sha256(V1_AUDITOR_PATH.read_bytes()).hexdigest() != V1_AUDITOR_SHA256:
    raise RuntimeError("reused recovery-v1 auditor bytes differ")
_v1_spec = importlib.util.spec_from_file_location(
    "_massive_medical_union_medical_recovery_v1_private", V1_AUDITOR_PATH
)
if _v1_spec is None or _v1_spec.loader is None:
    raise RuntimeError("cannot load private recovery-v1 auditor")
v1 = importlib.util.module_from_spec(_v1_spec)
_v1_spec.loader.exec_module(v1)


RECOVERY_ID = "massive_medical_union_wave1_medical_recovery_v2"
PARENT_COMMIT = "9ddd4816dafeb9b3df709e6ac72f41ebb22ee49f"
V1_REPO = v1.TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-medical-recovery-v1"
V2_REPO = v1.TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-medical-recovery-v2"
V1_CONTROL = v1.CONTROL_ROOT / "medical_recovery_v1"
V2_CONTROL = v1.CONTROL_ROOT / "medical_recovery_v2"
V1_EVAL = v1.OLD_EVAL_ROOT / "medical_recovery_v1"
V2_EVAL = v1.OLD_EVAL_ROOT / "medical_recovery_v2"
V2_GENERATIONS = V2_EVAL / "generations"
V2_PREP = V2_CONTROL / "PREP.json"
V2_JOBS = V2_CONTROL / "jobs.tsv"
V2_AUTH = V2_CONTROL / "AUTHORIZED_MAX_COST_USD_0.15.json"
V2_GPU_MANIFEST = V2_EVAL / "GPU_MEDICAL_RECOVERY_MANIFEST.json"
V2_SBATCH = V2_REPO / "scripts/sbatch_massive_medical_union_medical_recovery_v2_tillicum_h200.sbatch"
V2_JOB_NAME = "mmu_medrec_v2"
V2_STDOUT_TEMPLATE = v1.TILLICUM_ROOT / "outputs/logs/massive_medical_union_medical_recovery_v2_%j.out"
V2_STDERR_TEMPLATE = v1.TILLICUM_ROOT / "outputs/logs/massive_medical_union_medical_recovery_v2_%j.err"

V1_JOB_ID = "248197"
V1_STDOUT = v1.TILLICUM_ROOT / "outputs/logs/massive_medical_union_medical_recovery_v1_248197.out"
V1_STDERR = v1.TILLICUM_ROOT / "outputs/logs/massive_medical_union_medical_recovery_v1_248197.err"
V1_CONTROL_SHA256 = {
    "AUTHORIZED_MAX_COST_USD_0.15.json": "1ac6ee78c61abb4dc600c1d0a4a3440a28cbd5045f42bc14fdc98df369e97d75",
    "PREP.json": "db6a8605120daabb60ef5a77267e8e520a4431a6e442cd03d36dae7bf3aeb142",
    "RELEASED": "6be3c7f2a79b4a9c7d428a6d07477b1669f3988def8369a3894597f9f32fd5ba",
    "STOPPED_medical_recovery": "6387aeb29c1d6a9aea4871d9ee06ecd95029b5ddacb6c2339ee97b9d12d55f44",
    "SUBMISSION_ATTEMPT.tsv": "f177d900137e9f3d2ab4c699541d4c42eb98f809598daa4e734944b696d1b711",
    "SUBMISSION_LOCK/owner": "5edf2091fd9aaa8362e0ad4b079b9bd661ea06890e525dd4ecaf9bc7e4314134",
    "SUBMITTED": "5cc8749396d1764cb4629d3c79316e58add762efe8d53bba312c237fbe8bc1a5",
    "jobs.tsv": "264bac4503ad541daa33d9fcc24cb48741d5fd0c4738c5f675a2f5d465cb56a4",
}
V1_CONTROL_NAMES = {
    "AUTHORIZED_MAX_COST_USD_0.15.json", "PREP.json", "RELEASED",
    "STOPPED_medical_recovery", "SUBMISSION_ATTEMPT.tsv", "SUBMISSION_LOCK",
    "SUBMITTED", "jobs.tsv",
}
V1_LOG_SHA256 = {
    "stdout": "34db2f957bd6a1ec35ca219ece3fa5d21283347c8cffd0fe42deb0210f3a1cf5",
    "stderr": "0f031ca4c9794503406344538ab8651ac7dabe82731f58c301714ca432b4e626",
}
V1_TERMINAL_SCONTROL_SHA256 = "32158d74ab9c6a89bf810372ee2631be8035d3a58a93b5a0043337a3f64774a2"
V1_HELD_SCONTROL_SHA256 = "6dcb734a2bc7b6e35fda3f54732b25e8159e510260c9aa7a139f656e412fa056"
V1_PREP_PAYLOAD_SHA256 = "47e2a5521c4e740366ea0348540fc918f070727627769fcf27341bb1dcfe369a"
V1_AUTH_PAYLOAD_SHA256 = "ce9da5808bd25873243fd263be04657725352efbe20a92abb3765d0cd2add6c6"
V1_SBATCH_SHA256 = "b90bd46c4e1993251fdf855cd651666ce91c45be3899dc1201fbd219808f796b"

FROZEN_SCIENTIFIC_SHA256 = {
    "scripts/sample_massive_union_medical_direct.py": "0ab7c65a0807c8b6e89043f6809e1e9960c7426f14412518414a3ff59cc5b4ba",
    "scripts/judge_massive_union_medical.py": "f18c76c75d0c6c0021ff5dfad90205668e3695845b8f8a6e63fc38dfcfa9b314",
    "scripts/summarize_massive_union_components.py": "a83ba70502d284e9a805aeab76dfd23b0cea94d94d8ed2557574cb8b82993b29",
    "configs/training_qwen25_7b_massive_medical_union_pilot.yaml": "4dc9e8ac937bff92b1116d936b19bf907fedc027b12433b7070271647c0af8b5",
}
C3_NAME_STATUS = {
    ("A", "scripts/audit_massive_medical_union_medical_recovery_v2.py"),
    ("A", "scripts/finalize_massive_medical_union_wave1_medical_recovery_v2_tillicum.sh"),
    ("A", "scripts/sbatch_massive_medical_union_medical_recovery_v2_tillicum_h200.sbatch"),
    ("A", "scripts/stage_massive_medical_union_medical_recovery_v2_tillicum.sh"),
    ("A", "scripts/status_massive_medical_union_medical_recovery_v2_tillicum.sh"),
    ("A", "scripts/submit_massive_medical_union_medical_recovery_v2_tillicum.sh"),
    ("A", "tests/test_massive_medical_union_medical_recovery_v2.py"),
}


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", os.fspath(repo), *args], text=True).strip()


def audit_repositories():
    if V2_REPO.resolve() in {v1.MAIN_REPO.resolve(), V1_REPO.resolve()}:
        raise ValueError("recovery v2 requires its own isolated checkout")
    if _git(v1.MAIN_REPO, "rev-parse", "HEAD") != v1.MAIN_COMMIT or _git(v1.MAIN_REPO, "status", "--porcelain"):
        raise ValueError("main checkout differs from clean e25")
    if _git(V1_REPO, "rev-parse", "HEAD") != PARENT_COMMIT or _git(V1_REPO, "status", "--porcelain"):
        raise ValueError("failed recovery-v1 checkout differs from clean C2")
    commit = _git(V2_REPO, "rev-parse", "HEAD")
    if _git(V2_REPO, "status", "--porcelain"):
        raise ValueError("recovery-v2 checkout is dirty")
    parents = _git(V2_REPO, "rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, PARENT_COMMIT]:
        raise ValueError("recovery-v2 commit is not a direct nonmerge child of C2")
    raw = _git(V2_REPO, "diff", "--name-status", "--no-renames", f"{PARENT_COMMIT}..{commit}")
    observed = []
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError("invalid recovery-v2 name-status record")
        observed.append(tuple(fields))
    if len(observed) != len(C3_NAME_STATUS) or set(observed) != C3_NAME_STATUS:
        raise ValueError("recovery-v2 commit differs from exact seven-file scope")
    scientific = {}
    for relative, expected in FROZEN_SCIENTIFIC_SHA256.items():
        path = V2_REPO / relative
        scientific[relative] = v1.require_regular_hash(path, expected)
        if _git(V2_REPO, "diff", "--quiet", PARENT_COMMIT, "--", relative):
            raise AssertionError("git diff --quiet unexpectedly returned output")
    reused_v1_auditor = v1.require_regular_hash(
        V2_REPO / "scripts/audit_massive_medical_union_medical_recovery_v1.py",
        V1_AUDITOR_SHA256,
    )
    v1.require_regular_hash(
        V1_REPO / "scripts/audit_massive_medical_union_medical_recovery_v1.py",
        V1_AUDITOR_SHA256,
    )
    return {
        "main_repo": os.fspath(v1.MAIN_REPO), "main_commit": v1.MAIN_COMMIT,
        "recovery_v1_repo": os.fspath(V1_REPO), "recovery_v1_commit": PARENT_COMMIT,
        "recovery_v2_repo": os.fspath(V2_REPO), "recovery_v2_commit": commit,
        # Compatibility key consumed by the reviewed v1 GPU-manifest builder.
        "recovery_commit": commit,
        "auditor_sha256": v1.sha256_file(Path(__file__).resolve()),
        "reused_v1_auditor_sha256": reused_v1_auditor,
        "frozen_scientific_sha256": scientific,
    }


def audit_v1_terminal_accounting():
    """Audit durable accounting without requiring aged-out controller state."""
    rows = subprocess.check_output([
        "sacct", "-n", "-X", "-P", "-j", V1_JOB_ID,
        "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode,Start,End",
    ], text=True).strip().splitlines()
    expected = (
        "248197|mmu_medrec_v1|FAILED|00:00:05|00:10:00|"
        "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|"
        "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|"
        "1:0|2026-08-19T00:20:20|2026-08-19T00:20:25"
    )
    if rows != [expected]:
        raise ValueError("job 248197 durable accounting differs")
    return {
        "sacct_row": expected,
        "terminal_scontrol_live_required": False,
        # This is retained as a forensic note only.  Slurm has already aged
        # the corresponding controller record out; it is not re-queried.
        "terminal_scontrol_observation_sha256": V1_TERMINAL_SCONTROL_SHA256,
    }


def audit_recovery_v1_failure():
    if V1_CONTROL.is_symlink() or not V1_CONTROL.is_dir() or {item.name for item in V1_CONTROL.iterdir()} != V1_CONTROL_NAMES:
        raise ValueError("recovery-v1 control inventory differs")
    controls = {
        relative: v1.require_regular_hash(V1_CONTROL / relative, expected)
        for relative, expected in V1_CONTROL_SHA256.items()
    }
    prep = v1.load_json(V1_CONTROL / "PREP.json")
    auth = v1.load_json(V1_CONTROL / "AUTHORIZED_MAX_COST_USD_0.15.json")
    v1.verify_seal(prep, "recovery-v1 PREP")
    v1.verify_seal(auth, "recovery-v1 authorization")
    if prep.get("payload_sha256") != V1_PREP_PAYLOAD_SHA256 or auth.get("payload_sha256") != V1_AUTH_PAYLOAD_SHA256:
        raise ValueError("recovery-v1 sealed payload binding differs")
    v1.require_regular_hash(V1_STDOUT, V1_LOG_SHA256["stdout"])
    v1.require_regular_hash(V1_STDERR, V1_LOG_SHA256["stderr"])
    v1.require_regular_hash(
        V1_REPO / "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch",
        V1_SBATCH_SHA256,
    )
    held = auth.get("held_job_audit")
    held_raw = held.get("scontrol_record") if isinstance(held, dict) else None
    if not isinstance(held_raw, str) or v1.sha256_bytes(held_raw.encode()) != V1_HELD_SCONTROL_SHA256:
        raise ValueError("job 248197 held scontrol evidence differs")
    held_fields = v1.parse_scontrol_line(held_raw)
    held_exact = {
        "JobId": V1_JOB_ID, "JobName": "mmu_medrec_v1",
        "JobState": "PENDING", "Reason": "JobHeldUser",
        "Requeue": "0", "Restarts": "0", "RunTime": "00:00:00",
        "TimeLimit": "00:10:00", "Account": "stf", "QOS": "normal",
        "Partition": "gpu-h200", "NumNodes": "1-1", "NumCPUs": "8",
        "NumTasks": "1", "CPUs/Task": "8", "MinMemoryNode": "180G",
        "AllocTRES": "(null)",
        "Command": os.fspath(V1_REPO / "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch"),
        "WorkDir": os.fspath(V1_REPO),
        "StdOut": os.fspath(V1_STDOUT), "StdErr": os.fspath(V1_STDERR),
        "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
        "SubmitLine": (
            "sbatch --parsable --hold --export=NONE --job-name=mmu_medrec_v1 "
            "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch"
        ),
    }
    for key, expected in held_exact.items():
        if held_fields.get(key) != expected:
            raise ValueError(f"job 248197 held field differs: {key}")
    expected_tres = {
        "billing": "8", "cpu": "8", "gres/gpu:h200": "1",
        "gres/gpu": "1", "mem": "180G", "node": "1",
    }
    if v1.parse_tres(held_fields.get("ReqTRES", "")) != expected_tres:
        raise ValueError("job 248197 held request TRES differs")
    if (
        held.get("spooled_script_sha256") != V1_SBATCH_SHA256
        or held.get("committed_script_sha256") != V1_SBATCH_SHA256
    ):
        raise ValueError("job 248197 held spooled-script binding differs")
    if os.path.lexists(V1_EVAL):
        raise ValueError("failed recovery-v1 unexpectedly created evaluation output")
    if any(os.path.lexists(V1_CONTROL / name) for name in (
        "GPU_MEDICAL_RECOVERY_COMPLETE", "EXTERNAL_JUDGE_LOCK",
        "external_judge_checkpoint.json", "GO_MASSIVE_UNION_WAVE1",
    )):
        raise ValueError("recovery-v1 progressed beyond its runtime preflight failure")
    accounting = audit_v1_terminal_accounting()
    return {
        "control_sha256": controls,
        "stdout_sha256": V1_LOG_SHA256["stdout"],
        "stderr_sha256": V1_LOG_SHA256["stderr"],
        "held_scontrol_sha256": V1_HELD_SCONTROL_SHA256,
        "terminal_accounting": accounting,
        "failure_before_model_load": True,
        "evaluation_namespace_absent": True,
        "external_api_calls": 0,
    }


_wave1_evidence = v1.audit_original_evidence
_old_job_accounting = v1.audit_old_job_accounting
_v1_prep_body = v1.prep_body
_v1_gpu_body = v1.gpu_body


def audit_original_evidence(require_initial_inventory=False):
    expected_top = set(v1.EXPECTED_INITIAL_CONTROL_NAMES) | {"medical_recovery_v1"}
    if require_initial_inventory and {item.name for item in v1.CONTROL_ROOT.iterdir()} != expected_top:
        raise ValueError("pre-recovery-v2 top-level control inventory differs")
    result = _wave1_evidence(require_initial_inventory=False)
    result["medical_recovery_v1_failure"] = audit_recovery_v1_failure()
    return result


def audit_old_job_accounting():
    result = list(_old_job_accounting())
    result.append({
        "job_id": V1_JOB_ID, "job_name": "mmu_medrec_v1", "state": "FAILED",
        "elapsed": "00:00:05", "time_limit": "00:10:00",
        "alloc_tres": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1",
        "exit_code": "1:0", "start": "2026-08-19T00:20:20",
        "end": "2026-08-19T00:20:25",
    })
    return result


def audit_job_record(job_id, raw, fields, phase, check_held_log_absence=True):
    if phase not in {"held", "running"}:
        raise ValueError("invalid recovery-v2 scheduler audit phase")
    expected_exact = {
        "JobId": str(job_id),
        "JobName": V2_JOB_NAME,
        "Account": "stf",
        "QOS": "normal",
        "Requeue": "0",
        "Restarts": "0",
        "Partition": "gpu-h200",
        "NumTasks": "1",
        "NumCPUs": "8",
        "CPUs/Task": "8",
        "TimeLimit": "00:10:00",
        "Command": os.fspath(V2_SBATCH),
        "WorkDir": os.fspath(V2_REPO),
        "StdOut": os.fspath(V2_STDOUT_TEMPLATE).replace("%j", str(job_id)),
        "StdErr": os.fspath(V2_STDERR_TEMPLATE).replace("%j", str(job_id)),
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    for key, expected in expected_exact.items():
        if fields.get(key) != expected:
            raise ValueError(f"recovery-v2 job differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError("recovery-v2 job is not exactly one node")
    dependency = fields.get("Dependency", "")
    if dependency not in {"", "(null)"}:
        raise ValueError("recovery-v2 job unexpectedly has a dependency")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError("recovery-v2 job unexpectedly belongs to an array/het job")
    expected_tres = {
        "billing": "8", "cpu": "8", "gres/gpu:h200": "1",
        "gres/gpu": "1", "mem": "180G", "node": "1",
    }
    if v1.parse_tres(fields.get("ReqTRES", "")) != expected_tres:
        raise ValueError("recovery-v2 requested TRES differs")

    if phase == "held":
        # MinMemoryNode is the authoritative pre-allocation representation of
        # the requested host memory on this Slurm deployment.
        if fields.get("MinMemoryNode") != "180G":
            raise ValueError("held recovery-v2 memory request differs")
        if fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser":
            raise ValueError("recovery-v2 job is not held before authorization")
        if fields.get("RunTime") != "00:00:00" or fields.get("AllocTRES") != "(null)":
            raise ValueError("held recovery-v2 job already has runtime/allocation evidence")
        if check_held_log_absence:
            for template in (V2_STDOUT_TEMPLATE, V2_STDERR_TEMPLATE):
                path = Path(os.fspath(template).replace("%j", str(job_id)))
                if os.path.lexists(path):
                    raise ValueError("held recovery-v2 job already created a job log")
        expected_submit = (
            "sbatch --parsable --hold --export=NONE --job-name=mmu_medrec_v2 "
            "scripts/sbatch_massive_medical_union_medical_recovery_v2_tillicum_h200.sbatch"
        )
        if fields.get("SubmitLine") != expected_submit:
            raise ValueError("recovery-v2 submit line differs")
    else:
        # Once allocated, this site removes MinMemoryNode and adds
        # MinMemoryTRES/MemPerTres derived from the H200 GRES defaults.  Those
        # derived fields are deliberately ignored: exact ReqTRES and AllocTRES
        # are the stable request/allocation evidence.
        if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None":
            raise ValueError("recovery-v2 job is not RUNNING at job preflight")
        if v1.parse_tres(fields.get("AllocTRES", "")) != expected_tres:
            raise ValueError("running recovery-v2 allocation TRES differs")
        node = fields.get("NodeList", "")
        if not re.fullmatch(r"g[0-9]+", node) or fields.get("BatchHost") != node:
            raise ValueError("running recovery-v2 node allocation differs")

    return {
        "job_id": str(job_id),
        "job_name": V2_JOB_NAME,
        "phase": phase,
        "scontrol_record": raw,
        "scontrol_record_sha256": v1.sha256_bytes(raw.encode()),
        "normalized_nodes": 1,
        "requested_tres": expected_tres,
        "time_limit": "00:10:00",
        "no_requeue": True,
        "dependency_ids": [],
    }


def audit_slurm_environment(job_id):
    raw, fields = v1.query_job(job_id)
    expected = {
        "SLURM_JOB_ID": str(job_id),
        "SLURM_JOB_NAME": V2_JOB_NAME,
        "SLURM_JOB_PARTITION": "gpu-h200",
        "SLURM_JOB_ACCOUNT": "stf",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": os.fspath(V2_REPO),
        "SLURM_JOB_NODELIST": fields.get("NodeList"),
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError(f"recovery-v2 Slurm environment differs: {key}")
    if "SLURM_MEM_PER_NODE" in os.environ and os.environ["SLURM_MEM_PER_NODE"] != "184320":
        raise ValueError("recovery-v2 SLURM_MEM_PER_NODE differs from 180G")
    return expected


_v1_command_verify_job = v1.command_verify_job


def command_verify_job(job_id, time_limit):
    _v1_command_verify_job(job_id, time_limit)
    audit_slurm_environment(job_id)


def jobs_bytes(job_id):
    if not str(job_id).isdigit():
        raise ValueError("recovery-v2 job ID is invalid")
    return (
        "stage\tjob_id\tmax_minutes\treleased\n"
        f"medical_recovery_v2\t{job_id}\t10\ttrue\n"
    ).encode()


def parse_jobs(path=None):
    path = V2_JOBS if path is None else Path(path)
    lines = Path(path).read_bytes().splitlines()
    if len(lines) != 2 or lines[0] != b"stage\tjob_id\tmax_minutes\treleased":
        raise ValueError("recovery-v2 jobs table differs")
    fields = lines[1].decode().split("\t")
    if len(fields) != 4 or fields[0] != "medical_recovery_v2" or not fields[1].isdigit() or fields[2:] != ["10", "true"]:
        raise ValueError("recovery-v2 job row differs")
    if Path(path).read_bytes() != jobs_bytes(fields[1]):
        raise ValueError("recovery-v2 jobs bytes differ")
    return {"stage": fields[0], "job_id": fields[1], "max_minutes": 10, "released": True}


def prep_body(require_initial_inventory=False):
    body = _v1_prep_body(require_initial_inventory=require_initial_inventory)
    body["recovery_id"] = RECOVERY_ID
    body["budget"].update({
        "prior_released_ceiling_h200_minutes": 90,
        "cumulative_released_ceiling_h200_minutes": 100,
        "cumulative_released_gpu_ceiling_usd": 1.50,
    })
    body["scientific_change"].update({
        "same_science_as_medical_recovery_v1": True,
        "only_operational_change": "phase_specific_slurm_memory_field_normalization",
        "fresh_namespace": os.fspath(V2_EVAL),
    })
    return body


def gpu_body():
    body = _v1_gpu_body()
    body["recovery_id"] = RECOVERY_ID
    body["medical_recovery_v1_failure"] = {
        "job_id": V1_JOB_ID,
        "stdout_sha256": V1_LOG_SHA256["stdout"],
        "stderr_sha256": V1_LOG_SHA256["stderr"],
        "stopped_control_sha256": V1_CONTROL_SHA256["STOPPED_medical_recovery"],
        "preserved": True,
        "output_artifacts": 0,
    }
    return body


# Patch the reviewed implementation's dynamically read globals and hooks.
v1.RECOVERY_ID = RECOVERY_ID
v1.RECOVERY_REPO = V2_REPO
v1.RECOVERY_CONTROL = V2_CONTROL
v1.RECOVERY_EVAL_ROOT = V2_EVAL
v1.GENERATION_ROOT = V2_GENERATIONS
v1.PREP_FILE = V2_PREP
v1.JOBS_FILE = V2_JOBS
v1.AUTH_FILE = V2_AUTH
v1.GPU_MANIFEST = V2_GPU_MANIFEST
v1.SBATCH_FILE = V2_SBATCH
v1.JOB_NAME = V2_JOB_NAME
v1.JOB_STDOUT_TEMPLATE = V2_STDOUT_TEMPLATE
v1.JOB_STDERR_TEMPLATE = V2_STDERR_TEMPLATE
v1.TRAINING_CONFIG = V2_REPO / "configs/training_qwen25_7b_massive_medical_union_pilot.yaml"
v1.audit_repositories = audit_repositories
v1.audit_original_evidence = audit_original_evidence
v1.audit_old_job_accounting = audit_old_job_accounting
v1.audit_job_record = audit_job_record
v1.command_verify_job = command_verify_job
v1.jobs_bytes = jobs_bytes
v1.parse_jobs = parse_jobs
v1.prep_body = prep_body
v1.gpu_body = gpu_body


def run(argv=None):
    args = v1.build_parser().parse_args(argv)
    if args.command == "write-prep":
        v1.command_write_prep()
    elif args.command == "write-auth":
        v1.command_write_auth()
    elif args.command == "audit-held":
        v1.command_audit_held()
    elif args.command == "verify-job":
        command_verify_job(args.job_id, args.time_limit)
    elif args.command == "write-gpu":
        v1.command_write_gpu()
    elif args.command == "audit-gpu":
        payload = v1.audit_gpu()
        print("VALID_MEDICAL_RECOVERY_V2: " + payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
