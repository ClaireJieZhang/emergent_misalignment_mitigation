#!/usr/bin/env python3
"""CPU-only, fail-closed control for the four fixed ratio-panel GPU jobs.

No API client, scheduler submission, retry, or resume is implemented here.
The shell submitter owns four held submissions; this manager binds and audits
their exact identities before release and before scientific model loading.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import unicodedata
from decimal import Decimal

PROTOCOL_ID = "massive_medical_ratio_panels_v1_evaluation"
BRANCH = "claire/massive-medical-ratio-panel-evaluation-v1"
TILLICUM_ROOT = Path(os.environ.get("MMU_RATIO_EVAL_TILLICUM_ROOT", "/gpfs/projects/stf/claizhan/subliminal-mitigate"))
REPO_ROOT = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-ratio-panel-evaluation-v1"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs" / PROTOCOL_ID
CONTROL_ROOT = OUTPUT_ROOT / "control"
LOG_ROOT = TILLICUM_ROOT / "outputs/logs"
TRAIN_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-ratio-panels-v1"
TRAIN_ROOT = TILLICUM_ROOT / "outputs/massive_medical_ratio_panels_v1"
TRAIN_CONTROL = TRAIN_ROOT / "control/training"
SOURCE_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1_stage_recovery_v2"
TRAIN_COMMIT = "eb99679b6a1e21c8f7a91a71a079a96cc3ce2718"
TRAIN_TREE = "e5c17a51ff926dec27661db56ad76a66573a3aad"
TRAIN_RESULT_SHA = "c40a0670c037574a563ef4688740d84d495628b5d33c615b187025e3d7ff70da"
TRAIN_RESULT_PAYLOAD = "c5eee87350113e72da995e69df03b3097f41b7169baabe95c330918e694346a1"
TRAIN_COMPLETE_SHA = "4dfdef669406e9ff3eb33ff943c0853491999b282e9dd9b45bfc43f856fbd486"
TRAIN_PREP_SHA = "cc4d33c4d819ca515b34b87177c3358b3ee85b100603214d85bc25a83b321f57"
SOURCE_MANIFEST_SHA = "d13295ca0e39333007cc48ec8eb9b40c699fe3da32860af30e9c5addb32ddfc7"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
METHODS = ("ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4")
PANELS = {"two_bad_two_benign": ["A1", "A2", "B1", "B2"], "three_bad_one_benign": ["A1", "A2", "A3", "B1"]}
STAGES = tuple(f"{p}_{phase}" for phase in ("benefit", "medical") for p in PANELS)
MAX_COST_USD = "4.800000"
MAX_MINUTES = 320
WORKFLOW_FILES = (
    "scripts/manage_massive_medical_ratio_panels_v1_evaluation.py",
    "scripts/sample_massive_medical_ratio_panels_v1_evaluation.py",
    "scripts/stage_massive_medical_ratio_panels_v1_evaluation_tillicum.sh",
    "scripts/submit_massive_medical_ratio_panels_v1_evaluation_tillicum.sh",
    "scripts/status_massive_medical_ratio_panels_v1_evaluation_tillicum.sh",
    "scripts/finalize_massive_medical_ratio_panels_v1_evaluation_tillicum.sh",
    "scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch",
    "docs/massive_medical_ratio_panels_v1_evaluation_release.md",
    "tests/test_massive_medical_ratio_panels_v1_evaluation_workflow.py",
    "tests/test_massive_medical_ratio_panels_v1_evaluation_sampler.py",
)
PROMPT_BINDINGS = {
    "benefit_prompts": ("benefit/prompts.json", "6b3621aa2c5b58d0dd12b5a761f64d01416a17bc08772d3d535ced06bdf5d319", "46543c9df634b8ca99297d9767cdf38895a6ec3baa5802fd9cdeb3477a8547de"),
    "benefit_answers": ("benefit/answers.json", "15e52a5301d2f66d4edbc887ea3bb8ab18d5a3444ffae389851d6f125ed19b82", "e1d13589d9e7383d33931960f16289a96809851f3faf5551ea3c5878e7a101fc"),
    "benefit_selection": ("benefit/selection.json", "d5b59a654d63538e42e1f99eabddee8ba6a2ea90961ea615b630a9f60bb362d8", "c1738f1b4f8e1dea42e10cc0457aeb236e85db42a0969f7577f1061b74da556a"),
    "medical_prompts": ("medical/prompts.json", "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57", None),
}
RUNTIME_VERSIONS = {"torch": "2.9.0+cu129", "transformers": "4.57.6", "peft": "0.18.1", "xgrammar": "0.1.25"}
PREP_CACHE = None
PREP_CACHE_ENABLED = False


def canonical_bytes(value, *, ascii=False):
    return json.dumps(value, ensure_ascii=ascii, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_regular(path, *, immutable=False, contents=False):
    """Read/hash one descriptor, rejecting aliases and changes during the read."""
    p = regular(path, immutable=immutable)
    before = p.lstat()
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    h = hashlib.sha256()
    chunks = []
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if file_identity(opened) != file_identity(before):
            raise ValueError("file changed before read")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
            if contents:
                chunks.append(chunk)
        if file_identity(os.fstat(handle.fileno())) != file_identity(opened) or file_identity(p.lstat()) != file_identity(opened):
            raise ValueError("file changed during read")
    return h.hexdigest(), opened.st_size, b"".join(chunks) if contents else None


def file_identity(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sha_file(path):
    return read_regular(path)[0]


def seal(body):
    return {**body, "payload_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}


def verify_seal(payload, field="payload_sha256", *, ascii=False):
    if not isinstance(payload, dict) or field not in payload:
        raise ValueError("missing payload seal")
    body = {k: v for k, v in payload.items() if k != field}
    if payload[field] != hashlib.sha256(canonical_bytes(body, ascii=ascii)).hexdigest():
        raise ValueError("payload seal differs")
    return body


def regular(path, *, immutable=False):
    p = Path(path)
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError(f"unsafe regular file: {p}")
    if immutable and stat.S_IMODE(s.st_mode) != 0o400:
        raise ValueError(f"immutable file mode differs: {p}")
    return p


def private_directory(path):
    p = Path(path)
    if p.is_symlink() or not p.is_dir() or stat.S_IMODE(p.stat().st_mode) != 0o700:
        raise ValueError(f"unsafe private directory: {p}")
    return p


def load(path, *, immutable=False):
    return json.loads(read_regular(path, immutable=immutable, contents=True)[2])


def record(path, *, payload=None):
    p = Path(path)
    sha, size, _ = read_regular(p)
    result = {"path": os.fspath(p), "size_bytes": size, "file_sha256": sha}
    if payload is not None:
        result["payload_sha256"] = payload
    return result


def pinned(path, expected_sha, *, field=None, ascii=False):
    sha, _, contents = read_regular(path, contents=True)
    data = json.loads(contents)
    if sha != expected_sha:
        raise ValueError(f"frozen input hash differs: {path}")
    if field:
        verify_seal(data, field, ascii=ascii)
    return data


def write_once(path, body):
    """Claim the actual destination atomically; never replace a competitor."""
    path = Path(path)
    payload = seal(body)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o400)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # A partial destination consumes the claim; do not delete/retry it.
        raise
    return payload


def git(root, *args):
    return subprocess.run(["git", "-C", os.fspath(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def repository(root=None, *, training=False):
    root = REPO_ROOT if root is None else root
    if root.is_symlink() or not root.is_dir() or git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("repository is absent, aliased, or dirty")
    result = {"path": os.fspath(root), "branch": git(root, "branch", "--show-current"), "commit": git(root, "rev-parse", "HEAD"), "tree": git(root, "rev-parse", "HEAD^{tree}")}
    if result["branch"] != ("claire/massive-medical-ratio-panels-v1" if training else BRANCH):
        raise ValueError("repository branch differs")
    if training and (result["commit"], result["tree"]) != (TRAIN_COMMIT, TRAIN_TREE):
        raise ValueError("training repository drift")
    return result


def config(stage):
    if stage not in STAGES:
        raise ValueError("unknown evaluation stage")
    panel, phase = stage.rsplit("_", 1)
    code = "22" if panel == "two_bad_two_benign" else "31"
    minutes = 65 if phase == "benefit" else 95
    return {"stage": stage, "panel": panel, "phase": phase, "minutes": minutes,
            "time_limit": "01:05:00" if phase == "benefit" else "01:35:00",
            "job_name": f"mmu_ratio_eval_{code}_{phase}", "references": PANELS[panel],
            "direct_method": "direct_A2" if code == "22" else "direct_A3",
            "streams": [*METHODS, "direct_A2" if code == "22" else "direct_A3"]}


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("unsafe model directory")
    result = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("model inventory contains alias")
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel in {"MODEL_MANIFEST.json", "TRAIN_COMPLETE", "TRAIN_COMPLETE.json"}:
                continue
            r = record(p)
            result.append({"path": rel, "sha256": r["file_sha256"], "size_bytes": r["size_bytes"]})
    return result


def training_inputs():
    repo = repository(TRAIN_REPO, training=True)
    prep = pinned(TRAIN_CONTROL / "PREP.json", TRAIN_PREP_SHA, field="payload_sha256", ascii=True)
    result = pinned(TRAIN_CONTROL / "TRAINING_RESULT.json", TRAIN_RESULT_SHA, field="payload_sha256", ascii=True)
    complete = pinned(TRAIN_CONTROL / "TRAINING_COMPLETE.json", TRAIN_COMPLETE_SHA, field="payload_sha256", ascii=True)
    if result["payload_sha256"] != TRAIN_RESULT_PAYLOAD or complete["training_result_file_sha256"] != TRAIN_RESULT_SHA or complete["training_result_payload_sha256"] != TRAIN_RESULT_PAYLOAD:
        raise ValueError("training terminal pointers differ")
    command = [sys.executable, os.fspath(TRAIN_REPO / "scripts/manage_massive_medical_ratio_panels_v1.py"), "status"]
    status = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "MMU_RATIO_TILLICUM_ROOT"}).stdout)
    if (status.get("training_result") is not True or status.get("training_complete") is not True
        or status.get("training_result_payload_sha256") != TRAIN_RESULT_PAYLOAD
        or status.get("jobs") != {"A2": "297661", "A3": "297662"}
        or status.get("stopped_submission") is not False
        or any(status.get("models", {}).get(role, {}).get("sealed") is not True or status["models"][role].get("stopped") is not False for role in ("A2", "A3"))):
        raise ValueError("training is not terminally qualified")
    return {"repository": repo, "prep": record(TRAIN_CONTROL / "PREP.json", payload=prep["payload_sha256"]),
            "result": record(TRAIN_CONTROL / "TRAINING_RESULT.json", payload=result["payload_sha256"]),
            "complete": record(TRAIN_CONTROL / "TRAINING_COMPLETE.json", payload=complete["payload_sha256"]),
            "auditor": record(TRAIN_REPO / "scripts/manage_massive_medical_ratio_panels_v1.py"),
            "status": status}, prep, result


def audit_snapshot(snapshot, *, rehash):
    body = {k: v for k, v in snapshot.items() if k != "snapshot_binding_sha256"}
    if snapshot.get("snapshot_binding_sha256") != hashlib.sha256(canonical_bytes(body, ascii=True)).hexdigest():
        raise ValueError("base snapshot binding differs")
    expected = TILLICUM_ROOT / f"cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/{BASE_REVISION}"
    if snapshot.get("snapshot_realpath") != os.fspath(expected) or snapshot.get("revision") != BASE_REVISION or snapshot.get("canonical_model_id") != BASE_MODEL:
        raise ValueError("base snapshot identity differs")
    artifacts = {**snapshot["required_artifacts"], **snapshot["weight_shard_artifacts"]}
    if len(artifacts) != 11:
        raise ValueError("base snapshot artifact count differs")
    hub = expected.parent.parent
    for name, r in artifacts.items():
        p = expected / name
        resolved = p.resolve(strict=True)
        if not resolved.is_relative_to(hub) or os.fspath(resolved) != r["resolved_path"]:
            raise ValueError("snapshot resolves outside sealed cache")
        regular(resolved)
        if resolved.stat().st_size != r["size_bytes"] or (rehash and sha_file(resolved) != r["sha256"]):
            raise ValueError("base load bytes differ")
    return snapshot


def inputs(*, rehash_snapshot=False):
    training, train_prep, train_result = training_inputs()
    source = pinned(SOURCE_ROOT / "protocol/manifest.json", SOURCE_MANIFEST_SHA, field="manifest_payload_sha256")
    prompts = {}
    for key, (rel, sha, payload_sha) in PROMPT_BINDINGS.items():
        p = SOURCE_ROOT / "protocol" / rel
        data = pinned(p, sha, field="payload_sha256" if payload_sha else None)
        if payload_sha and data["payload_sha256"] != payload_sha:
            raise ValueError("prompt payload binding differs")
        prompts[key] = record(p, payload=payload_sha)
    models = []
    refs = source["model_panel"]["references"]
    for role in ("A1", "A2", "A3", "B1", "B2"):
        if role in ("A2", "A3"):
            p = TRAIN_ROOT / "models" / f"pi_{role}"
            expected = train_result["models"][role]
            m = pinned(p / "MODEL_MANIFEST.json", expected["manifest_file_sha256"], field="payload_sha256", ascii=True)
            if m["payload_sha256"] != expected["manifest_payload_sha256"] or m.get("final_global_step") != 540 or m.get("scientific_checkpoint") != 540 or m.get("seed") != {"A2": 8182127, "A3": 8182228}[role]:
                raise ValueError("new adapter qualification differs")
            adapter_inventory = m["adapter_inventory"]
            fingerprint = m["adapter_fingerprint"]
            exact_inventory = m["exact_model_inventory"]
            manifest_path = p / "MODEL_MANIFEST.json"
        else:
            ref = refs["pi_A" if role == "A1" else f"pi_{role}"]
            p = Path(ref["model_path"])
            manifest_path = Path(ref["path"])
            pinned(manifest_path, ref["file_sha256"], field=ref["payload_seal_field"])
            adapter_inventory, fingerprint, exact_inventory = ref["adapter_inventory"], ref["model_fingerprint"], ref["exact_model_inventory"]
        if inventory(p) != exact_inventory:
            raise ValueError(f"{role} live model inventory differs")
        if fingerprint != hashlib.sha256(canonical_bytes(adapter_inventory)).hexdigest():
            raise ValueError("adapter fingerprint differs")
        models.append({"role": role, "path": os.fspath(p), "manifest": record(manifest_path, payload=load(manifest_path).get("payload_sha256")), "inventory": exact_inventory,
                       "adapter_inventory": adapter_inventory, "adapter_fingerprint": fingerprint})
    base = SOURCE_ROOT / "generation/benefit/pi_base/massive/generation.json"
    pinned(base, "5a74be77b837194fb67c09d12392630a2d17f8590dd15d3713809d87f896335e")
    versions = {k: importlib.metadata.version(k) for k in RUNTIME_VERSIONS}
    if versions != RUNTIME_VERSIONS:
        raise ValueError("pinned decoder runtime versions differ")
    return {"training": training, "source_manifest": record(SOURCE_ROOT / "protocol/manifest.json", payload=source["manifest_payload_sha256"]),
            "models": models, "prompts": prompts, "local_model_snapshot": audit_snapshot(train_prep["source"]["local_model_snapshot"], rehash=rehash_snapshot),
            "reuse": {"paired_base": record(base), "original_panel_regenerated": False}, "runtime_versions": versions}


def prep_body(created_at, *, rehash_snapshot=False):
    source = inputs(rehash_snapshot=rehash_snapshot)
    workflow = []
    tracked = set(git(REPO_ROOT, "ls-files").splitlines())
    for name in WORKFLOW_FILES:
        if name not in tracked:
            raise ValueError(f"workflow file is not tracked: {name}")
        workflow.append({**record(REPO_ROOT / name), "mode": oct(stat.S_IMODE((REPO_ROOT / name).stat().st_mode))})
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID, "created_at": created_at, "repository": repository(), "workflow_files": workflow,
            **source, "panels": PANELS, "stages": [config(s) for s in STAGES],
            "profiles": {"benefit": {"rows": 360, "n_samples": 1, "temperature": 0.0, "max_new_tokens": 256, "max_context": 2048, "structured_profile": "const_tree_no_ws_v3"},
                         "medical": {"rows": 16, "n_samples": 5, "temperature": 1.0, "max_new_tokens": 1024, "max_context": 2048, "seed": 8172026, "required_finish_reason": "stop"}},
            "limits": {"jobs": 4, "h200_minutes": MAX_MINUTES, "max_cost_usd": MAX_COST_USD, "api_calls_authorized": 0, "no_retry_resume_requeue": True, "optional_baselines_authorized": False}}


def audit_prep(*, rehash_snapshot=False):
    global PREP_CACHE
    private_directory(OUTPUT_ROOT)
    private_directory(CONTROL_ROOT)
    payload = load(CONTROL_ROOT / "PREP.json", immutable=True)
    body = verify_seal(payload)
    cached = (PREP_CACHE_ENABLED and PREP_CACHE is not None and PREP_CACHE[0] == payload
              and (PREP_CACHE[1] or not rehash_snapshot))
    if not cached:
        if body != prep_body(body["created_at"], rehash_snapshot=rehash_snapshot):
            raise ValueError("live evaluation preparation differs")
        if PREP_CACHE_ENABLED:
            PREP_CACHE = (payload, rehash_snapshot)
    staged = verify_seal(load(CONTROL_ROOT / "STAGED.json", immutable=True))
    if staged != {"protocol_id": PROTOCOL_ID, "prep": record(CONTROL_ROOT / "PREP.json", payload=payload["payload_sha256"]), "gpu_jobs_authorized": 0, "api_calls_authorized": 0}:
        raise ValueError("stage receipt differs")
    return payload


def stage(_):
    if os.path.lexists(OUTPUT_ROOT) or any(LOG_ROOT.glob(f"{PROTOCOL_ID}_*")):
        raise FileExistsError("fresh evaluation and log namespaces required")
    body = prep_body(dt.datetime.now(dt.timezone.utc).isoformat(), rehash_snapshot=True)
    OUTPUT_ROOT.mkdir(mode=0o700)
    CONTROL_ROOT.mkdir(mode=0o700)
    prep = write_once(CONTROL_ROOT / "PREP.json", body)
    write_once(CONTROL_ROOT / "STAGED.json", {"protocol_id": PROTOCOL_ID, "prep": record(CONTROL_ROOT / "PREP.json", payload=prep["payload_sha256"]), "gpu_jobs_authorized": 0, "api_calls_authorized": 0})
    audit_prep()
    print(json.dumps({"status": "CPU_STAGED_NO_PAID_AUTHORITY", "prep_payload_sha256": prep["payload_sha256"]}))


def control_record(name):
    path = CONTROL_ROOT / name
    payload = load(path, immutable=True)
    verify_seal(payload)
    return record(path, payload=payload["payload_sha256"])


def authorization_body():
    prep = audit_prep()
    return {"protocol_id": PROTOCOL_ID, "prep": record(CONTROL_ROOT / "PREP.json", payload=prep["payload_sha256"]),
            "stages": list(STAGES), "jobs": 4, "h200_minutes": MAX_MINUTES, "max_cost_usd": MAX_COST_USD,
            "held_first": True, "requeue_retry_resume_authorized": False, "replacement_authorized": False,
            "external_api_authorized": False, "optional_baselines_authorized": False, "unused_authority_reusable": False}


def audit_authorization():
    payload = load(CONTROL_ROOT / "AUTHORIZATION.json", immutable=True)
    if verify_seal(payload) != authorization_body():
        raise ValueError("evaluation authorization differs")
    return payload


def authorize(args):
    if args.ack_max_cost_usd != MAX_COST_USD:
        raise ValueError("exact $4.800000 evaluation acknowledgment required")
    if set(p.name for p in CONTROL_ROOT.iterdir()) != {"PREP.json", "STAGED.json"} or (OUTPUT_ROOT / "generation").exists():
        raise FileExistsError("fresh paid namespace required")
    write_once(CONTROL_ROOT / "AUTHORIZATION.json", authorization_body())
    print("EXACT_FOUR_JOB_AUTHORIZATION_SEALED")


def submission_lock(_):
    audit_authorization()
    write_once(CONTROL_ROOT / "SUBMISSION_LOCK.json", {"protocol_id": PROTOCOL_ID, "authorization": control_record("AUTHORIZATION.json"), "maximum_jobs": 4, "max_cost_usd": MAX_COST_USD, "retry_authorized": False})
    print("PERMANENT_EVALUATION_SUBMISSION_LOCK_SEALED")


def audit_lock():
    audit_authorization()
    payload = load(CONTROL_ROOT / "SUBMISSION_LOCK.json", immutable=True)
    if verify_seal(payload) != {"protocol_id": PROTOCOL_ID, "authorization": control_record("AUTHORIZATION.json"), "maximum_jobs": 4, "max_cost_usd": MAX_COST_USD, "retry_authorized": False}:
        raise ValueError("permanent submission lock differs")
    return payload


def parse_scontrol(raw):
    if not isinstance(raw, str) or "\n" in raw.strip() or not raw.startswith("JobId="):
        raise ValueError("scheduler did not return one job record")
    fields = {}
    for token in raw.split():
        if "=" in token:
            key, value = token.split("=", 1)
            if key in fields:
                raise ValueError("duplicate scheduler field")
            fields[key] = value
    return fields


def tres(value):
    fields = {}
    for token in value.split(","):
        if "=" not in token:
            raise ValueError("invalid resource field")
        key, v = token.split("=", 1)
        if key in fields or not key or not v:
            raise ValueError("duplicate resource field")
        fields[key] = v
    return fields


def expected_tres():
    return {"billing": "8", "cpu": "8", "gres/gpu:h200": "1", "gres/gpu": "1", "mem": "200G", "node": "1"}


def audit_job(stage, job_id, raw, *, phase):
    c = config(stage)
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("invalid numeric job identity")
    f = parse_scontrol(raw)
    required = {"JobId": job_id, "JobName": c["job_name"], "Account": "stf", "QOS": "normal", "Partition": "gpu-h200",
                "Requeue": "0", "Restarts": "0", "NumTasks": "1", "NumCPUs": "8", "CPUs/Task": "8", "TimeLimit": c["time_limit"],
                "Command": os.fspath(REPO_ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch"),
                "WorkDir": os.fspath(REPO_ROOT), "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
                "StdOut": os.fspath(LOG_ROOT / f"{PROTOCOL_ID}_{stage}_{job_id}.out"), "StdErr": os.fspath(LOG_ROOT / f"{PROTOCOL_ID}_{stage}_{job_id}.err")}
    if any(f.get(k) != v for k, v in required.items()) or f.get("NumNodes") not in {"1", "1-1"} or tres(f.get("ReqTRES", "")) != expected_tres():
        raise ValueError("scheduler identity/resource contract differs")
    if f.get("Dependency") not in {None, "", "(null)"} or f.get("KillOnInvalidDependent", "") not in {"", "No"} or any(k.startswith(("Array", "HetJob")) for k in f):
        raise ValueError("array, heterogeneous, or dependency job forbidden")
    if phase == "held":
        if any(f.get(k) != v for k, v in {"JobState": "PENDING", "Reason": "JobHeldUser", "RunTime": "00:00:00", "AllocTRES": "(null)", "MinMemoryNode": "200G"}.items()):
            raise ValueError("job is not pristine and held")
    elif phase == "released":
        if f.get("Reason") == "JobHeldUser" or f.get("JobState") not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}:
            raise ValueError("job was not released from audited hold")
        if f.get("JobState") in {"RUNNING", "COMPLETING", "COMPLETED"} and tres(f.get("AllocTRES", "")) != expected_tres():
            raise ValueError("released allocation differs")
    elif phase == "running":
        if f.get("JobState") != "RUNNING" or f.get("Reason") != "None" or tres(f.get("AllocTRES", "")) != expected_tres():
            raise ValueError("live job is not exact RUNNING allocation")
        node = f.get("NodeList", "")
        if re.fullmatch(r"g[0-9]+", node) is None or f.get("BatchHost") != node:
            raise ValueError("live node identity differs")
    else:
        raise ValueError("unknown scheduler audit phase")
    return f


def evidence_dir(value):
    p = private_directory(Path(value))
    if p.parent != CONTROL_ROOT or not p.name.startswith(".held-submit."):
        raise ValueError("submission evidence is outside exact private namespace")
    return p


def record_jobs(args):
    audit_lock()
    directory = evidence_dir(args.records_dir)
    jobs, seen = {}, set()
    batch = REPO_ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch"
    for s in STAGES:
        id_path = regular(directory / f"{s}.job_id")
        job_id = id_path.read_text().strip()
        if job_id in seen:
            raise ValueError("duplicate held job identity")
        seen.add(job_id)
        held = regular(directory / f"{s}.held.scontrol")
        raw = held.read_text().strip()
        audit_job(s, job_id, raw, phase="held")
        spooled = regular(directory / f"{s}.spooled.sbatch")
        if sha_file(spooled) != sha_file(batch):
            raise ValueError("spooled script is not committed bytes")
        for suffix in ("out", "err"):
            if os.path.lexists(LOG_ROOT / f"{PROTOCOL_ID}_{s}_{job_id}.{suffix}"):
                raise ValueError("held job has already created a log")
        for path in (id_path, held, spooled, regular(directory / f"{s}.submitted.stdout")):
            os.chmod(path, 0o400)
        jobs[s] = {"job_id": job_id, "config": config(s), "held_record": raw,
                   "held_file": record(held), "spooled_script": record(spooled), "job_id_file": record(id_path)}
    write_once(CONTROL_ROOT / "JOBS.json", {"protocol_id": PROTOCOL_ID, "authorization": control_record("AUTHORIZATION.json"), "submission_lock": control_record("SUBMISSION_LOCK.json"), "jobs": jobs})
    print("EXACT_FOUR_HELD_JOBS_AUDITED")


def audit_jobs():
    audit_lock()
    payload = load(CONTROL_ROOT / "JOBS.json", immutable=True)
    body = verify_seal(payload)
    if set(body) != {"protocol_id", "authorization", "submission_lock", "jobs"} or body["protocol_id"] != PROTOCOL_ID or body["authorization"] != control_record("AUTHORIZATION.json") or body["submission_lock"] != control_record("SUBMISSION_LOCK.json") or list(body["jobs"]) != list(STAGES):
        raise ValueError("held jobs inventory/pointers differ")
    ids = []
    for s, j in body["jobs"].items():
        ids.append(j["job_id"])
        audit_job(s, j["job_id"], j["held_record"], phase="held")
        if j["config"] != config(s):
            raise ValueError("held stage configuration differs")
        for key in ("held_file", "spooled_script", "job_id_file"):
            r = j[key]
            regular(r["path"], immutable=True)
            if record(r["path"]) != r:
                raise ValueError("held submission evidence differs")
        if read_regular(j["held_file"]["path"], immutable=True, contents=True)[2].decode().strip() != j["held_record"] or read_regular(j["job_id_file"]["path"], immutable=True, contents=True)[2].decode().strip() != j["job_id"]:
            raise ValueError("held record/job identity does not match retained evidence")
        if j["spooled_script"]["file_sha256"] != sha_file(REPO_ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_evaluation_tillicum_h200.sbatch"):
            raise ValueError("spooled script binding differs")
    if len(set(ids)) != 4:
        raise ValueError("not four unique jobs")
    return payload


def release_body():
    audit_jobs()
    return {"protocol_id": PROTOCOL_ID, "jobs": control_record("JOBS.json"), "max_h200_minutes": MAX_MINUTES, "max_cost_usd": MAX_COST_USD, "api_calls_authorized": 0, "retry_replacement_authorized": False}


def authorize_release(_):
    if (CONTROL_ROOT / "STOPPED_SUBMISSION.json").exists():
        raise ValueError("submission failure is terminal")
    write_once(CONTROL_ROOT / "RELEASE_AUTHORIZED.json", release_body())
    print("EXACT_FOUR_HELD_JOBS_RELEASE_AUTHORIZED")


def audit_release_authorization():
    payload = load(CONTROL_ROOT / "RELEASE_AUTHORIZED.json", immutable=True)
    if verify_seal(payload) != release_body():
        raise ValueError("release authorization differs")
    return payload


def record_release(args):
    jobs = verify_seal(audit_jobs())["jobs"]
    audit_release_authorization()
    directory = evidence_dir(args.records_dir)
    released = {}
    for s, j in jobs.items():
        p = regular(directory / f"{s}.released.scontrol")
        raw = p.read_text().strip()
        audit_job(s, j["job_id"], raw, phase="released")
        os.chmod(p, 0o400)
        released[s] = {"job_id": j["job_id"], "record": raw, "file": record(p)}
    write_once(CONTROL_ROOT / "RELEASED.json", {"protocol_id": PROTOCOL_ID, "jobs": control_record("JOBS.json"), "release_authorization": control_record("RELEASE_AUTHORIZED.json"), "released": released, "released_exactly_once": True})
    print("EXACT_FOUR_EVALUATION_JOBS_RELEASED")


def audit_release():
    if (CONTROL_ROOT / "STOPPED_SUBMISSION.json").exists():
        raise ValueError("submission failure is terminal; no scientific entry")
    jobs = verify_seal(audit_jobs())["jobs"]
    audit_release_authorization()
    payload = load(CONTROL_ROOT / "RELEASED.json", immutable=True)
    b = verify_seal(payload)
    if set(b) != {"protocol_id", "jobs", "release_authorization", "released", "released_exactly_once"} or b["protocol_id"] != PROTOCOL_ID or b["jobs"] != control_record("JOBS.json") or b["release_authorization"] != control_record("RELEASE_AUTHORIZED.json") or list(b["released"]) != list(STAGES) or b["released_exactly_once"] is not True:
        raise ValueError("release record differs")
    for s, item in b["released"].items():
        if item["job_id"] != jobs[s]["job_id"]:
            raise ValueError("released identity differs")
        audit_job(s, item["job_id"], item["record"], phase="released")
        regular(item["file"]["path"], immutable=True)
        if record(item["file"]["path"]) != item["file"]:
            raise ValueError("released scheduler evidence differs")
        if read_regular(item["file"]["path"], immutable=True, contents=True)[2].decode().strip() != item["record"]:
            raise ValueError("released record does not match retained evidence")
    return payload


def live_job(job_id):
    return subprocess.run(["scontrol", "show", "job", job_id, "-o"], check=True, capture_output=True, text=True).stdout.strip()


def runtime_environment(stage, job_id, fields):
    expected = {"SLURM_JOB_ID": job_id, "SLURM_JOB_NAME": config(stage)["job_name"], "SLURM_JOB_PARTITION": "gpu-h200", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_JOB_NODELIST": fields["NodeList"], "SLURM_RESTART_COUNT": "0"}
    if any(os.environ.get(k, "0" if k == "SLURM_RESTART_COUNT" else None) != v for k, v in expected.items()) or any(k.startswith(("SLURM_ARRAY_", "SLURM_HET_")) for k in os.environ):
        raise ValueError("runtime scheduler environment differs")
    return expected


def runtime_path(stage):
    config(stage)
    return CONTROL_ROOT / f"RUN_STARTED_{stage}.json"


def verify_runtime(args):
    prep = audit_prep(rehash_snapshot=True)
    release = audit_release()
    jobs = verify_seal(audit_jobs())["jobs"]
    if jobs[args.stage]["job_id"] != args.job_id or (CONTROL_ROOT / f"STOPPED_{args.stage}.json").exists():
        raise ValueError("runtime identity or terminal state differs")
    raw = live_job(args.job_id)
    fields = audit_job(args.stage, args.job_id, raw, phase="running")
    env = runtime_environment(args.stage, args.job_id, fields)
    if (OUTPUT_ROOT / "generation" / args.stage).exists():
        raise FileExistsError("fresh generation phase required")
    write_once(runtime_path(args.stage), {"protocol_id": PROTOCOL_ID, "stage": args.stage, "job_id": args.job_id, "prep": record(CONTROL_ROOT / "PREP.json", payload=prep["payload_sha256"]),
               "release": record(CONTROL_ROOT / "RELEASED.json", payload=release["payload_sha256"]), "record": raw, "environment": env, "node": fields["NodeList"], "second_entry_authorized": False})
    print("EXACT_EVALUATION_RUNTIME_VERIFIED_NO_SECOND_ENTRY")


def audit_runtime(args, *, live=True, deep_snapshot=False):
    prep = audit_prep(rehash_snapshot=deep_snapshot)
    audit_release()
    jobs = verify_seal(audit_jobs())["jobs"]
    payload = load(runtime_path(args.stage), immutable=True)
    b = verify_seal(payload)
    if jobs[args.stage]["job_id"] != args.job_id or b.get("job_id") != args.job_id or b.get("stage") != args.stage or b.get("protocol_id") != PROTOCOL_ID or b.get("second_entry_authorized") is not False or b.get("prep") != control_record("PREP.json") or b.get("release") != control_record("RELEASED.json") or (CONTROL_ROOT / f"STOPPED_{args.stage}.json").exists():
        raise ValueError("runtime receipt pointers/state differ")
    f = audit_job(args.stage, args.job_id, b["record"], phase="running")
    if b["node"] != f["NodeList"]:
        raise ValueError("runtime node binding differs")
    expected_env = {"SLURM_JOB_ID": args.job_id, "SLURM_JOB_NAME": config(args.stage)["job_name"], "SLURM_JOB_PARTITION": "gpu-h200", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_JOB_NODELIST": f["NodeList"], "SLURM_RESTART_COUNT": "0"}
    if b.get("environment") != expected_env:
        raise ValueError("saved runtime environment contract differs")
    if live:
        f_live = audit_job(args.stage, args.job_id, live_job(args.job_id), phase="running")
        if f_live["NodeList"] != b["node"] or runtime_environment(args.stage, args.job_id, f_live) != b["environment"]:
            raise ValueError("runtime live identity changed")
    else:
        terminal_accounting(args.stage, args.job_id)
    selected = config(args.stage)["references"]
    models = {m["role"]: m for m in prep["models"]}
    return {"stage": args.stage, "job_id": args.job_id, "output_root": os.fspath(OUTPUT_ROOT),
            "prep": control_record("PREP.json"), "runtime_receipt": record(runtime_path(args.stage), payload=payload["payload_sha256"]),
            "panel": selected, "models": [models[r] for r in selected], "prompts": prep["prompts"], "local_model_snapshot": prep["local_model_snapshot"], "reuse": prep["reuse"]}


def record_failure(args):
    config(args.stage)
    jobs = verify_seal(load(CONTROL_ROOT / "JOBS.json", immutable=True))["jobs"]
    if jobs[args.stage]["job_id"] != args.job_id or args.exit_code == 0:
        raise ValueError("failure identity/code differs")
    receipt = load(runtime_path(args.stage), immutable=True)
    b = verify_seal(receipt)
    if b.get("protocol_id") != PROTOCOL_ID or b.get("stage") != args.stage or b.get("job_id") != args.job_id or b.get("second_entry_authorized") is not False:
        raise ValueError("failure does not have a claimed runtime entry")
    write_once(CONTROL_ROOT / f"STOPPED_{args.stage}.json", {"protocol_id": PROTOCOL_ID, "stage": args.stage, "job_id": args.job_id,
        "runtime_receipt": record(runtime_path(args.stage), payload=receipt["payload_sha256"]), "exit_code": args.exit_code, "retry_authorized": False})


def submission_failure(args):
    directory = evidence_dir(args.records_dir)
    if args.exit_code == 0:
        raise ValueError("zero exit code is not a failure")
    ids = {}
    for s in STAGES:
        p = directory / f"{s}.job_id"
        if p.exists():
            value = regular(p).read_text().strip()
            if re.fullmatch(r"[0-9]+", value) is None:
                raise ValueError("invalid partial-submission identity")
            ids[s] = value
    write_once(CONTROL_ROOT / "STOPPED_SUBMISSION.json", {"protocol_id": PROTOCOL_ID, "exit_code": args.exit_code, "created_jobs": ids, "evidence_dir": os.fspath(directory), "release_authorization_existed": (CONTROL_ROOT / "RELEASE_AUTHORIZED.json").exists(), "retry_authorized": False})


def duration_seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = (int(x) for x in clock.split(":"))
    if min(int(days), hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
        raise ValueError("invalid scheduler duration")
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_terminal(stage, job_id, raw):
    parts = raw.split("|")
    if len(parts) == 11 and not parts[-1]:
        parts.pop()
    if len(parts) != 10:
        raise ValueError("terminal scheduler row shape differs")
    identity, name, state, elapsed, limit, start, end, allocated, requested, exit_code = parts
    c = config(stage)
    seconds = duration_seconds(elapsed)
    if identity != job_id or name != c["job_name"] or state != "COMPLETED" or exit_code != "0:0" or limit != c["time_limit"] or seconds > c["minutes"] * 60 or start in {"", "Unknown"} or end in {"", "Unknown"} or tres(allocated) != expected_tres() or tres(requested) != expected_tres():
        raise ValueError("terminal scheduler success/resource/cap differs")
    cost = Decimal(seconds) * Decimal("0.90") / Decimal(3600)
    return {"stage": stage, "job_id": job_id, "state": state, "exit_code": exit_code, "start": start, "end": end,
            "elapsed_seconds": seconds, "actual_h200_minutes": seconds / 60, "actual_gpu_cost_usd": str(cost), "sacct_row": raw}


def terminal_accounting(stage, job_id):
    raw = subprocess.run(["sacct", "-j", job_id, "--allocations", "--noheader", "--parsable2", "--format=JobIDRaw,JobName%128,State,Elapsed,Timelimit,Start,End,AllocTRES,ReqTRES,ExitCode"], check=True, capture_output=True, text=True).stdout
    rows = [line for line in raw.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("terminal scheduler row count differs")
    return parse_terminal(stage, job_id, rows[0])


def sampler_audit(stage, job_id, *, terminal=False):
    command = [sys.executable, os.fspath(REPO_ROOT / "scripts/sample_massive_medical_ratio_panels_v1_evaluation.py"), "--stage", stage, "--job-id", job_id, "--audit-only"]
    if terminal:
        command.append("--terminal-audit")
    report = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    c = config(stage)
    streams = report.get("streams")
    n = 360 if c["phase"] == "benefit" else 80
    domain = "massive" if c["phase"] == "benefit" else "medical"
    if not isinstance(streams, list) or [s.get("method_id") for s in streams] != c["streams"] or any(s.get("samples") != n or s.get("domain") != domain for s in streams) or type(report.get("profile_valid")) is not bool:
        raise ValueError("sampler report violates the fixed four-stream plan")
    completion = report.get("completion")
    expected_path = OUTPUT_ROOT / "generation" / stage / "SAMPLER_COMPLETE.json"
    if not isinstance(completion, dict) or completion.get("path") != os.fspath(expected_path):
        raise ValueError("sampler completion path differs")
    p = load(expected_path, immutable=True)
    body = verify_seal(p)
    if completion.get("file_sha256") != sha_file(expected_path) or completion.get("payload_sha256") != p["payload_sha256"] or body.get("stage") != stage or body.get("job_id") != job_id or body.get("external_api_calls") != 0 or body.get("all_planned_cells_preserved") is not True or body.get("profile_valid") != report["profile_valid"]:
        raise ValueError("sampler terminal provenance differs")
    return report


def normalized_slot(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def capability_scores(stage, report):
    answers = load(SOURCE_ROOT / "protocol/benefit/answers.json")["answers"]
    scores = {}
    for stream in report["streams"]:
        generation = load(stream["generation"]["path"], immutable=True)
        verify_seal(generation)
        samples = generation["samples"]
        if len(samples) != 360 or len(answers) != 360:
            raise ValueError("capability denominator must remain 360")
        intent_correct = frame_correct = overlap = predicted_total = gold_total = invalid = 0
        for answer, sample in zip(answers, samples):
            if answer["question_id"] != sample["question_id"] or answer["prompt_sha256"] != sample["prompt_sha256"]:
                raise ValueError("capability sample/gold row pairing differs")
            prediction = sample.get("prediction")
            if prediction is not None and (not isinstance(prediction, dict) or not isinstance(prediction.get("slots"), list)):
                raise ValueError("capability prediction shape differs")
            invalid += prediction is None
            intent_ok = prediction is not None and prediction.get("intent") == answer["intent"]
            predicted_ordered = [(x["name"], normalized_slot(x["value"])) for x in prediction["slots"]] if prediction is not None else []
            gold_ordered = [(x["name"], normalized_slot(x["value"])) for x in answer["slots"]]
            exact_substring = [x["value"] in answer["utterance"] for x in prediction["slots"]] if prediction is not None else []
            predicted = collections.Counter(pair for pair, good in zip(predicted_ordered, exact_substring) if good)
            gold = collections.Counter(gold_ordered)
            intent_correct += intent_ok
            frame_correct += intent_ok and all(exact_substring) and predicted_ordered == gold_ordered
            overlap += sum((predicted & gold).values())
            predicted_total += len(predicted_ordered)
            gold_total += sum(gold.values())
        total_pairs = predicted_total + gold_total
        scores[stream["method_id"]] = {"requested_n": 360, "intent_correct_n": intent_correct, "intent_accuracy": intent_correct / 360,
            "invalid_or_truncated_n": invalid, "strict_frame_correct_n": frame_correct, "strict_frame_accuracy": frame_correct / 360,
            "slot_pair_micro_f1": 2 * overlap / total_pairs if total_pairs else 1.0,
            "slot_pair_metric": "exact_name_and_NFKC_casefold_whitespace_normalized_value_multiset_with_exact_utterance_substring_constraint",
            "strict_frame_metric": "correct_intent_and_ordered_exact_slots_with_exact_utterance_substring_constraint",
            "gold_used_only_for_post_generation_scoring": True}
    return scores


def stage_result_body(stage, job_id, report):
    return {"protocol_id": PROTOCOL_ID, "stage": stage, "job_id": job_id, "sampler": report,
            "capability_scores": capability_scores(stage, report) if config(stage)["phase"] == "benefit" else None,
            "medical_judging_authorized": False, "profile_valid": report["profile_valid"], "retry_authorized": False,
            "status": "GENERATION_COMPLETE_AWAITING_SEPARATE_JUDGING" if config(stage)["phase"] == "medical" else "CAPABILITY_EVALUATION_COMPLETE"}


def complete_stage(args):
    audit_runtime(args, deep_snapshot=True)
    report = sampler_audit(args.stage, args.job_id)
    write_once(CONTROL_ROOT / f"RESULT_{args.stage}.json", stage_result_body(args.stage, args.job_id, report))
    print(f"EVALUATION_STAGE_COMPLETE: {args.stage}")


def audit_stage_result(stage, job_id, *, terminal):
    report = sampler_audit(stage, job_id, terminal=terminal)
    payload = load(CONTROL_ROOT / f"RESULT_{stage}.json", immutable=True)
    if verify_seal(payload) != stage_result_body(stage, job_id, report):
        raise ValueError("stage completion result differs")
    return payload


def terminal_body(stages, accounting):
    if list(stages) != list(STAGES) or list(accounting) != list(STAGES):
        raise ValueError("terminal four-stage inventory differs")
    cost = sum((Decimal(v["actual_gpu_cost_usd"]) for v in accounting.values()), Decimal(0))
    seconds = sum(v["elapsed_seconds"] for v in accounting.values())
    if cost > Decimal(MAX_COST_USD) or seconds > MAX_MINUTES * 60:
        raise ValueError("combined evaluation cap exceeded")
    return {"protocol_id": PROTOCOL_ID, "status": "EVALUATION_GENERATION_COMPLETE_AWAITING_SEPARATE_JUDGING",
            "jobs": control_record("JOBS.json"), "release": control_record("RELEASED.json"),
            "stage_results": {s: control_record(f"RESULT_{s}.json") for s in STAGES}, "terminal_accounting": accounting,
            "actual_h200_minutes": seconds / 60, "actual_gpu_cost_usd": str(cost), "max_cost_usd": MAX_COST_USD,
            "all_profiles_valid": all(p["profile_valid"] for p in stages.values()), "api_calls_authorized": 0, "retry_authorized": False}


def finalize(_):
    if os.path.lexists(CONTROL_ROOT / "EVALUATION_COMPLETE.json"):
        raise FileExistsError("terminal finalization is single-entry")
    jobs = verify_seal(audit_jobs())["jobs"]
    audit_release()
    stages, accounting = {}, {}
    for s, j in jobs.items():
        accounting[s] = terminal_accounting(s, j["job_id"])
        stages[s] = audit_stage_result(s, j["job_id"], terminal=True)
    body = terminal_body(stages, accounting)
    write_once(CONTROL_ROOT / "EVALUATION_COMPLETE.json", body)
    print(json.dumps({"status": body["status"], "actual_gpu_cost_usd": body["actual_gpu_cost_usd"], "all_profiles_valid": body["all_profiles_valid"]}))


def status(_):
    result = {"protocol_id": PROTOCOL_ID, "output_exists": OUTPUT_ROOT.exists(), "external_api_authorized": False, "jobs": {}, "stages": {}, "results_fully_reconstructed": False}
    if OUTPUT_ROOT.exists():
        result["prep_payload_sha256"] = audit_prep()["payload_sha256"]
        for name in ("AUTHORIZATION", "SUBMISSION_LOCK", "JOBS", "RELEASE_AUTHORIZED", "RELEASED", "STOPPED_SUBMISSION", "EVALUATION_COMPLETE"):
            p = CONTROL_ROOT / f"{name}.json"
            result[name.lower()] = p.exists()
            if p.exists():
                result[f"{name.lower()}_binding"] = control_record(p.name)
        if (CONTROL_ROOT / "JOBS.json").exists():
            jobs = verify_seal(audit_jobs())["jobs"]
            result["jobs"] = {s: j["job_id"] for s, j in jobs.items()}
        if (CONTROL_ROOT / "RELEASED.json").exists():
            audit_release()
        for s in STAGES:
            p = CONTROL_ROOT / f"RESULT_{s}.json"
            result["stages"][s] = {"run_started": runtime_path(s).exists(), "completed": p.exists(), "stopped": (CONTROL_ROOT / f"STOPPED_{s}.json").exists()}
            if p.exists():
                b = verify_seal(load(p, immutable=True))
                result["stages"][s].update({"profile_valid": b["profile_valid"], "capability_scores": b["capability_scores"]})
        if (CONTROL_ROOT / "EVALUATION_COMPLETE.json").exists():
            b = verify_seal(load(CONTROL_ROOT / "EVALUATION_COMPLETE.json", immutable=True))
            stages = {s: audit_stage_result(s, jobs[s]["job_id"], terminal=True) for s in STAGES}
            accounting = {s: terminal_accounting(s, jobs[s]["job_id"]) for s in STAGES}
            if b != terminal_body(stages, accounting):
                raise ValueError("terminal evaluation reconstruction differs")
            result.update({"actual_gpu_cost_usd": b["actual_gpu_cost_usd"], "all_profiles_valid": b["all_profiles_valid"], "results_fully_reconstructed": True})
    print(json.dumps(result, indent=2, sort_keys=True))


def main(argv=None):
    global PREP_CACHE, PREP_CACHE_ENABLED
    # One fresh source audit per command; nested control-chain audits reuse it.
    # Independent sampler boundaries launch new processes and re-audit sources.
    PREP_CACHE = None
    PREP_CACHE_ENABLED = False
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"stage": stage, "audit-stage": lambda _: print(json.dumps({"status": "CPU_STAGE_AUDIT_PASS", "prep_payload_sha256": audit_prep(rehash_snapshot=True)["payload_sha256"]})),
                "authorize": authorize, "record-submission-lock": submission_lock, "record-jobs": record_jobs,
                "authorize-release": authorize_release, "record-release": record_release, "record-submission-failure": submission_failure,
                "verify-runtime": verify_runtime, "audit-runtime": lambda a: print(json.dumps(audit_runtime(a, live=not a.terminal_only), sort_keys=True)),
                "record-failure": record_failure, "complete-stage": complete_stage, "finalize": finalize, "status": status}
    for name in commands:
        p = sub.add_parser(name)
        if name == "authorize":
            p.add_argument("--ack-max-cost-usd", required=True)
        if name in {"record-jobs", "record-release", "record-submission-failure"}:
            p.add_argument("--records-dir", required=True)
        if name in {"verify-runtime", "audit-runtime", "record-failure", "complete-stage"}:
            p.add_argument("--stage", choices=STAGES, required=True)
            p.add_argument("--job-id", required=True)
        if name == "audit-runtime":
            p.add_argument("--terminal-only", action="store_true")
        if name in {"record-submission-failure", "record-failure"}:
            p.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    if "OPENAI_API_KEY" in os.environ:
        raise ValueError("API credentials must be absent for CPU/GPU evaluation control")
    PREP_CACHE_ENABLED = True
    try:
        commands[args.command](args)
    finally:
        PREP_CACHE = None
        PREP_CACHE_ENABLED = False


if __name__ == "__main__":
    main()
