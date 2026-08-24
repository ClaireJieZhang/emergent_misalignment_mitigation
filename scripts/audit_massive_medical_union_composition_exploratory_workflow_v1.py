#!/usr/bin/env python3
"""Fail-closed control-plane auditor for exploratory composition execution.

This module owns only CPU control records and scheduler audits.  It never
submits, releases, cancels, requeues, or retries a job and never calls an API.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


SCHEMA_VERSION = 1
WORKFLOW_ID = "massive_medical_union_composition_exploratory_workflow_v1"
PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
BASELINE_COMMIT = "404af12c35bcfa1f1293289243e075412f90532b"
BRANCH = "claire/capability-quorum-secure-code-composition-exploratory-v1"
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
REPO_ROOT = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-composition-exploratory-v1"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_composition_exploratory_v1"
PROTOCOL_ROOT = OUTPUT_ROOT / "protocol"
GENERATION_ROOT = OUTPUT_ROOT / "generation"
EVALUATION_ROOT = OUTPUT_ROOT / "evaluation"
CONTROL_ROOT = OUTPUT_ROOT / "control"
LOG_ROOT = TILLICUM_ROOT / "outputs/logs"
ENV_ROOT = TILLICUM_ROOT / "envs/subliminal-mitigate-py311"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
LOCAL_MODEL_SNAPSHOT = (
    TILLICUM_ROOT
    / "cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    / BASE_REVISION
)
PROTOCOL_MANIFEST = PROTOCOL_ROOT / "manifest.json"
PREP_FILE = CONTROL_ROOT / "PREP.json"
STAGED_FILE = CONTROL_ROOT / "STAGED"
FINALIZER_LOCK = CONTROL_ROOT / "FINALIZER_LOCK"
FINAL_AUTH = CONTROL_ROOT / "EXTERNAL_JUDGE_AUTHORIZED_MAX_COST_USD_0.75.json"
FINAL_RESULT = CONTROL_ROOT / "FINAL_RESULT.json"
JUDGE_CHECKPOINT = EVALUATION_ROOT / "medical/judge_checkpoint.json"
JUDGE_NEW = EVALUATION_ROOT / "medical/judgments_new.json"
JUDGE_MERGED = EVALUATION_ROOT / "medical/judgments_merged.json"
FINAL_GATE_ROOT = EVALUATION_ROOT / "final"
RATE_PER_H200_MINUTE_USD = 0.015
FIELD_RE = re.compile(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=")
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.0+cu129",
    "transformers": "4.57.6",
    "peft": "0.18.1",
    "xgrammar": "0.1.25",
    "openai": "1.109.1",
}
SNAPSHOT_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
)

METHOD_IDS = (
    "ordinary_quorum_m4_q3",
    "ordinary_min_m4_q4",
    "delta_min_m4_q4",
)

ADDED_FILES = (
    "docs/massive_medical_union_composition_exploratory_v1.md",
    "scripts/prepare_massive_medical_union_composition_exploratory_v1.py",
    "scripts/audit_massive_medical_union_composition_exploratory_v1.py",
    "scripts/sample_massive_medical_union_composition_exploratory_v1.py",
    "scripts/summarize_massive_medical_union_composition_exploratory_v1.py",
    "scripts/judge_massive_medical_union_composition_exploratory_v1.py",
    "scripts/merge_massive_medical_union_composition_exploratory_v1.py",
    "scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py",
    "scripts/stage_massive_medical_union_composition_exploratory_v1_tillicum.sh",
    "scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch",
    "scripts/submit_massive_medical_union_composition_exploratory_v1_smoke_tillicum.sh",
    "scripts/sbatch_massive_medical_union_composition_exploratory_v1_confirmation_tillicum_h200.sbatch",
    "scripts/submit_massive_medical_union_composition_exploratory_v1_confirmation_tillicum.sh",
    "scripts/finalize_massive_medical_union_composition_exploratory_v1_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_v1_tillicum.sh",
    "tests/test_massive_medical_union_composition_exploratory_protocol.py",
    "tests/test_massive_medical_union_composition_exploratory_sampler.py",
    "tests/test_massive_medical_union_composition_exploratory_evaluation.py",
    "tests/test_massive_medical_union_composition_exploratory_workflow.py",
)

EXECUTABLE_FILES = tuple(path for path in ADDED_FILES if path.startswith("scripts/"))
REGULAR_FILES = tuple(path for path in ADDED_FILES if not path.startswith("scripts/"))

STAGES = {
    "smoke": {
        "job_name": "mmu_cmpx_smoke_v1",
        "minutes": 15,
        "time_limit": "00:15:00",
        "cost": 0.225,
        "sbatch": REPO_ROOT
        / "scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch",
        "log_prefix": "massive_medical_union_composition_exploratory_v1_smoke",
        "lock": CONTROL_ROOT / "SMOKE_SUBMISSION_LOCK",
        "attempt": CONTROL_ROOT / "SMOKE_SUBMISSION_ATTEMPT.tsv",
        "job": CONTROL_ROOT / "SMOKE_JOB.json",
        "auth": CONTROL_ROOT / "SMOKE_AUTHORIZED_MAX_COST_USD_0.225.json",
        "submitted": CONTROL_ROOT / "SMOKE_SUBMITTED",
        "released": CONTROL_ROOT / "SMOKE_RELEASE_AUTHORIZED",
        "result": CONTROL_ROOT / "SMOKE_RESULT.json",
        "preflight": CONTROL_ROOT / "SMOKE_CPU_PREFLIGHT.json",
        "stop": CONTROL_ROOT / "STOPPED_smoke",
        "gate_root": EVALUATION_ROOT / "smoke/gate",
        "allowed_gate": ("EXPLORATORY_SMOKE_PASSED", "STOPPED_EXPLORATORY_SMOKE"),
    },
    "confirmation": {
        "job_name": "mmu_cmpx_confirm_v1",
        "minutes": 100,
        "time_limit": "01:40:00",
        "cost": 1.50,
        "sbatch": REPO_ROOT
        / "scripts/sbatch_massive_medical_union_composition_exploratory_v1_confirmation_tillicum_h200.sbatch",
        "log_prefix": "massive_medical_union_composition_exploratory_v1_confirmation",
        "lock": CONTROL_ROOT / "CONFIRMATION_SUBMISSION_LOCK",
        "attempt": CONTROL_ROOT / "CONFIRMATION_SUBMISSION_ATTEMPT.tsv",
        "job": CONTROL_ROOT / "CONFIRMATION_JOB.json",
        "auth": CONTROL_ROOT / "CONFIRMATION_AUTHORIZED_MAX_COST_USD_1.50.json",
        "submitted": CONTROL_ROOT / "CONFIRMATION_SUBMITTED",
        "released": CONTROL_ROOT / "CONFIRMATION_RELEASE_AUTHORIZED",
        "result": CONTROL_ROOT / "CONFIRMATION_RESULT.json",
        "preflight": CONTROL_ROOT / "CONFIRMATION_CPU_PREFLIGHT.json",
        "stop": CONTROL_ROOT / "STOPPED_confirmation",
        "gate_root": EVALUATION_ROOT / "confirmation/prejudge",
        "allowed_gate": (
            "AWAITING_EXTERNAL_JUDGE",
            "STOPPED_EXPLORATORY_CONFIRMATION_PREJUDGE",
        ),
    },
}


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


def runtime_version_matches(distribution, observed, expected):
    if distribution == "torch":
        return observed in {expected, expected.split("+", 1)[0]}
    return observed == expected


def audit_runtime_versions():
    observed = {}
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        value = importlib.metadata.version(distribution)
        if not runtime_version_matches(distribution, value, expected):
            raise ValueError(
                f"runtime version differs for {distribution}: {value!r} != {expected!r}"
            )
        observed[distribution] = value
    return observed


def fingerprint_stable_file(path, allowed_root):
    lexical = os.path.abspath(path)
    resolved = os.path.realpath(lexical)
    root = os.path.realpath(allowed_root)
    if os.path.commonpath((root, resolved)) != root:
        raise ValueError(f"snapshot artifact escapes model cache: {lexical}")
    try:
        before = os.stat(resolved)
    except OSError as error:
        raise ValueError(f"snapshot artifact is absent: {lexical}") from error
    if not os.path.isfile(resolved) or before.st_size <= 0:
        raise ValueError(f"snapshot artifact is not a nonempty regular file: {lexical}")
    digest = hashlib.sha256()
    byte_count = 0
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(handle.fileno())
    resolved_after = os.path.realpath(lexical)
    path_after = os.stat(resolved_after)

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        resolved_after != resolved
        or identity(before) != identity(after)
        or identity(after) != identity(path_after)
        or byte_count != after.st_size
    ):
        raise ValueError(f"snapshot artifact changed while hashing: {lexical}")
    return {
        "size_bytes": after.st_size,
        "resolved_path": resolved,
        "sha256": digest.hexdigest(),
    }


def audit_local_model_snapshot():
    resolved = os.path.realpath(LOCAL_MODEL_SNAPSHOT)
    expected_suffix = (
        "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/" + BASE_REVISION
    )
    if not resolved.endswith(expected_suffix) or not os.path.isdir(resolved):
        raise ValueError("local model snapshot is not the pinned Qwen revision")
    cache_root = os.path.realpath(os.path.join(resolved, "..", ".."))
    artifacts = {
        name: fingerprint_stable_file(os.path.join(resolved, name), cache_root)
        for name in SNAPSHOT_REQUIRED_FILES
    }
    index_path = os.path.join(resolved, "model.safetensors.index.json")
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if fingerprint_stable_file(index_path, cache_root) != artifacts[
        "model.safetensors.index.json"
    ]:
        raise ValueError("model weight index changed while reading")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model weight index is empty")
    shards = sorted(set(weight_map.values()))
    positions = []
    declared_counts = set()
    shard_artifacts = {}
    shard_bytes = 0
    for shard in shards:
        if not isinstance(shard, str) or os.path.basename(shard) != shard:
            raise ValueError("model weight index contains an unsafe shard")
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", shard)
        if match is None:
            raise ValueError(f"model weight shard name differs: {shard}")
        positions.append(int(match.group(1)))
        declared_counts.add(int(match.group(2)))
        shard_artifacts[shard] = fingerprint_stable_file(
            os.path.join(resolved, shard), cache_root
        )
        shard_bytes += shard_artifacts[shard]["size_bytes"]
    if declared_counts != {len(shards)} or sorted(positions) != list(
        range(1, len(shards) + 1)
    ):
        raise ValueError("model weight index does not declare a complete shard set")
    metadata = index.get("metadata")
    indexed_bytes = metadata.get("total_size") if isinstance(metadata, dict) else None
    if (
        isinstance(indexed_bytes, bool)
        or not isinstance(indexed_bytes, int)
        or indexed_bytes <= 0
        or indexed_bytes > shard_bytes
    ):
        raise ValueError("model weight index total_size differs")
    unindexed = sorted(
        name
        for name in os.listdir(resolved)
        if name.endswith(".safetensors") and name not in shards
    )
    if unindexed:
        raise ValueError(f"local model snapshot has unindexed shards: {unindexed}")
    tokenizer_config_path = os.path.join(resolved, "tokenizer_config.json")
    tokenizer_config = json.loads(
        Path(tokenizer_config_path).read_text(encoding="utf-8")
    )
    if fingerprint_stable_file(tokenizer_config_path, cache_root) != artifacts[
        "tokenizer_config.json"
    ]:
        raise ValueError("tokenizer configuration changed while reading")
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("pinned tokenizer lacks a nonempty embedded chat template")
    payload = {
        "canonical_model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "local_path": resolved,
        "required_artifacts": artifacts,
        "weight_shards": shards,
        "weight_shard_artifacts": shard_artifacts,
        "indexed_weight_bytes": indexed_bytes,
        "chat_template_source": "tokenizer_config.json:chat_template",
        "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
    }
    payload["snapshot_binding_sha256"] = sha256_bytes(canonical_bytes(payload))
    return payload


def require_regular(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing or unsafe: {path}")
    return path


def load_json(path, description):
    path = require_regular(path, description)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error


def seal(body):
    payload = dict(body)
    payload["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return payload


def verify_seal(payload, description, field="payload_sha256"):
    if not isinstance(payload, dict) or not isinstance(payload.get(field), str):
        raise ValueError(f"{description} lacks {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if payload[field] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} seal differs")
    return body


def binding(path, payload=None, seal_field="payload_sha256"):
    path = require_regular(path, "bound artifact")
    result = {
        "path": os.fspath(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }
    if payload is not None:
        result["payload_sha256"] = payload[seal_field]
        result["payload_seal_field"] = seal_field
    return result


def atomic_write_once(path, content, mode=0o400):
    path = Path(path)
    if os.path.lexists(path):
        raise ValueError(f"refusing existing control artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=os.fspath(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if os.path.lexists(path):
            raise ValueError(f"control artifact appeared while writing: {path}")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_sealed_once(path, body):
    payload = seal(body)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write_once(path, encoded)
    return payload


def git(*args):
    return subprocess.check_output(
        ["git", "-C", os.fspath(REPO_ROOT), *args], text=True
    ).strip()


def audit_repository():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("composition checkout is missing or unsafe")
    commit = git("rev-parse", "HEAD")
    parents = git("rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, BASELINE_COMMIT]:
        raise ValueError("composition commit is not the direct nonmerge baseline child")
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("composition checkout branch differs")
    if git("status", "--porcelain"):
        raise ValueError("composition checkout is dirty")
    diff_lines = git(
        "diff", "--name-status", "--no-renames", f"{BASELINE_COMMIT}..{commit}"
    ).splitlines()
    observed = [tuple(line.split("\t")) for line in diff_lines]
    expected = [("A", path) for path in ADDED_FILES]
    if set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError("composition commit differs from its exact add-only allowlist")
    files = {}
    for path in ADDED_FILES:
        full = require_regular(REPO_ROOT / path, f"repository file {path}")
        index = git("ls-files", "-s", "--", path).split()
        expected_mode = "100755" if path in EXECUTABLE_FILES else "100644"
        if len(index) < 4 or index[0] != expected_mode:
            raise ValueError(f"committed mode differs for {path}")
        actual_mode = full.stat().st_mode & 0o777
        if actual_mode != (0o755 if path in EXECUTABLE_FILES else 0o644):
            raise ValueError(f"worktree mode differs for {path}")
        files[path] = {
            "git_mode": expected_mode,
            "size_bytes": full.stat().st_size,
            "sha256": sha256_file(full),
        }
    return {
        "path": os.fspath(REPO_ROOT),
        "branch": BRANCH,
        "baseline_commit": BASELINE_COMMIT,
        "commit": commit,
        "direct_nonmerge_parent": True,
        "add_only_allowlist": list(ADDED_FILES),
        "files": files,
    }


def protocol_binding(run_full_audit=False):
    payload = load_json(PROTOCOL_MANIFEST, "protocol manifest")
    body = verify_seal(
        payload, "protocol manifest", field="manifest_payload_sha256"
    )
    if (
        body.get("protocol_id") != PROTOCOL_ID
        or body.get("exploratory_contract", {}).get("confirmatory") is not False
        or body.get("exploratory_contract", {}).get("wave3_v1_eligible") is not False
        or body.get("budget", {}).get("smoke_gpu_h200_minutes_max") != 15
        or body.get("budget", {}).get("confirmation_gpu_h200_minutes_max") != 100
        or body.get("budget", {}).get("wave3_gpu_h200_minutes_max") != 115
        or body.get("budget", {}).get("wave3_external_judge_cost_max") != 0.75
        or body.get("budget", {}).get("wave3_all_in_cost_max") != 2.475
    ):
        raise ValueError("protocol manifest identity/budget differs")
    result = {
        **binding(PROTOCOL_MANIFEST, payload, "manifest_payload_sha256"),
        "protocol_id": PROTOCOL_ID,
    }
    if run_full_audit:
        completed = subprocess.run(
            [
                os.fspath(ENV_ROOT / "bin/python"),
                os.fspath(
                    REPO_ROOT
                    / "scripts/audit_massive_medical_union_composition_exploratory_v1.py"
                ),
                "audit-protocol",
                "--protocol-root",
                os.fspath(PROTOCOL_ROOT),
                "--json",
            ],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if completed.returncode != 0:
            raise ValueError(
                "full protocol audit failed: " + completed.stderr.strip()
            )
        try:
            audit = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ValueError("full protocol audit returned malformed JSON") from error
        if (
            audit.get("status") != "AUDIT_OK"
            or audit.get("manifest_file_sha256") != result["file_sha256"]
            or audit.get("manifest_payload_sha256") != result["payload_sha256"]
            or audit.get("confirmatory") is not False
            or audit.get("wave3_v1_eligible") is not False
        ):
            raise ValueError("full protocol audit result differs")
        result["full_audit"] = audit
    return result


def expected_namespaces():
    return {
        "repository": os.fspath(REPO_ROOT),
        "output": os.fspath(OUTPUT_ROOT),
        "protocol": os.fspath(PROTOCOL_ROOT),
        "generation": os.fspath(GENERATION_ROOT),
        "evaluation": os.fspath(EVALUATION_ROOT),
        "control": os.fspath(CONTROL_ROOT),
        "log_prefix": os.fspath(
            LOG_ROOT / "massive_medical_union_composition_exploratory_v1_"
        ),
    }


def budget_registry():
    return {
        "smoke_h200_minutes_cap": 15,
        "smoke_gpu_cost_usd_cap": 0.225,
        "confirmation_h200_minutes_cap": 100,
        "confirmation_gpu_cost_usd_cap": 1.50,
        "combined_released_h200_minutes_cap": 115,
        "combined_released_gpu_cost_usd_cap": 1.725,
        "external_judge_calls_cap": 240,
        "external_judge_cost_usd_cap": 0.75,
        "all_in_cost_usd_cap": 2.475,
        "gpu_rate_usd_per_h200_minute": RATE_PER_H200_MINUTE_USD,
        "retry_reserve_h200_minutes": 0,
    }


def prep_body(created_at=None):
    ready = require_regular(ENV_ROOT / ".ready", "environment readiness marker")
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": audit_repository(),
        "protocol": protocol_binding(run_full_audit=True),
        "environment": {
            "path": os.fspath(ENV_ROOT),
            "ready": binding(ready),
            "runtime_versions": audit_runtime_versions(),
        },
        "local_model_snapshot": audit_local_model_snapshot(),
        "namespaces": expected_namespaces(),
        "budget": budget_registry(),
        "stage_submitted_jobs": 0,
        "training": False,
        "automatic_continuation": False,
        "dependency_jobs": False,
        "requeue": False,
        "retry_or_reserve": False,
        "external_api_calls": 0,
        "confirmatory_claim": False,
        "wave2_v1_status": "STOP",
        "wave3_v1_eligible": False,
        "wave3_v1_submitted_or_released": False,
    }


def command_write_prep(_args):
    if CONTROL_ROOT.exists():
        raise ValueError("control namespace already exists")
    CONTROL_ROOT.mkdir(parents=True)
    payload = write_sealed_once(PREP_FILE, prep_body())
    print(payload["payload_sha256"])


def audit_prep():
    payload = load_json(PREP_FILE, "workflow PREP")
    body = verify_seal(payload, "workflow PREP")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("repository") != audit_repository()
        or body.get("namespaces") != expected_namespaces()
        or body.get("budget") != budget_registry()
    ):
        raise ValueError("workflow PREP differs")
    protocol = dict(body.get("protocol") or {})
    full_audit = protocol.pop("full_audit", None)
    if protocol != protocol_binding(run_full_audit=False):
        raise ValueError("workflow PREP protocol binding differs")
    if (
        not isinstance(full_audit, dict)
        or full_audit.get("status") != "AUDIT_OK"
        or full_audit.get("manifest_file_sha256") != protocol["file_sha256"]
        or full_audit.get("manifest_payload_sha256") != protocol["payload_sha256"]
    ):
        raise ValueError("workflow PREP full protocol audit differs")
    ready = binding(ENV_ROOT / ".ready")
    environment = body.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("path") != os.fspath(ENV_ROOT)
        or environment.get("ready") != ready
        or environment.get("runtime_versions") != audit_runtime_versions()
    ):
        raise ValueError("workflow PREP environment differs")
    prepared_snapshot = body.get("local_model_snapshot")
    if (
        not isinstance(prepared_snapshot, dict)
        or prepared_snapshot.get("canonical_model_id") != BASE_MODEL
        or prepared_snapshot.get("revision") != BASE_REVISION
        or prepared_snapshot.get("local_path") != os.path.realpath(LOCAL_MODEL_SNAPSHOT)
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(prepared_snapshot.get("snapshot_binding_sha256", "")),
        )
    ):
        raise ValueError("workflow PREP local-model snapshot binding differs")
    immutable = {
        "stage_submitted_jobs": 0,
        "training": False,
        "automatic_continuation": False,
        "dependency_jobs": False,
        "requeue": False,
        "retry_or_reserve": False,
        "external_api_calls": 0,
        "confirmatory_claim": False,
        "wave2_v1_status": "STOP",
        "wave3_v1_eligible": False,
        "wave3_v1_submitted_or_released": False,
    }
    if any(body.get(key) != value for key, value in immutable.items()):
        raise ValueError("workflow PREP safety flags differ")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_prep(_args):
    result = audit_prep()
    print(json.dumps({"status": "PREP_OK", "payload_sha256": result["payload_sha256"]}))


def audit_staged():
    prep = audit_prep()
    expected = (
        "workflow_id=massive_medical_union_composition_exploratory_workflow_v1\n"
        f"repo_commit={prep['repository']['commit']}\n"
        "stage_submitted_jobs=0\n"
        "training=false\n"
        "external_api_calls=0\n"
        "wave3_v1_submitted_or_released=false\n"
    ).encode("utf-8")
    staged = require_regular(STAGED_FILE, "STAGED marker")
    if staged.read_bytes() != expected:
        raise ValueError("STAGED marker differs")
    load_preflight("smoke")
    load_preflight("confirmation")
    return binding(staged)


def command_audit_staged(_args):
    result = audit_staged()
    print(json.dumps({"status": "STAGED_OK", "file_sha256": result["file_sha256"]}))


def parse_scontrol_line(line):
    if not isinstance(line, str) or not line.strip():
        raise ValueError("empty scontrol record")
    matches = list(FIELD_RE.finditer(line.strip()))
    if not matches or matches[0].group(1) != "JobId":
        raise ValueError("scontrol record does not start with JobId")
    result = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in result:
            raise ValueError(f"scontrol record repeats {key}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        result[key] = line[match.end() : end].strip()
    return result


def parse_tres(value):
    result = {}
    for term in value.split(","):
        if "=" not in term:
            raise ValueError(f"invalid TRES term: {term}")
        key, item = term.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid/repeated TRES term: {term}")
        result[key] = item
    return result


def expected_tres():
    return {
        "billing": "8",
        "cpu": "8",
        "gres/gpu:h200": "1",
        "gres/gpu": "1",
        "mem": "180G",
        "node": "1",
    }


def query_job(job_id):
    raw = subprocess.check_output(
        ["scontrol", "show", "job", "-o", str(job_id)], text=True
    ).strip()
    return raw, parse_scontrol_line(raw)


def audit_job_record(stage, job_id, raw, fields, phase, check_log_absence=True):
    config = STAGES[stage]
    exact = {
        "JobId": str(job_id),
        "JobName": config["job_name"],
        "Account": "stf",
        "QOS": "normal",
        "Requeue": "0",
        "Restarts": "0",
        "Partition": "gpu-h200",
        "NumTasks": "1",
        "NumCPUs": "8",
        "CPUs/Task": "8",
        "TimeLimit": config["time_limit"],
        "Command": os.fspath(config["sbatch"]),
        "WorkDir": os.fspath(REPO_ROOT),
        "StdOut": os.fspath(LOG_ROOT / f"{config['log_prefix']}_{job_id}.out"),
        "StdErr": os.fspath(LOG_ROOT / f"{config['log_prefix']}_{job_id}.err"),
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            raise ValueError(f"{stage} job differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError(f"{stage} job node count differs")
    if parse_tres(fields.get("ReqTRES", "")) != expected_tres():
        raise ValueError(f"{stage} requested TRES differs")
    if fields.get("Dependency") not in {None, "", "(null)"}:
        raise ValueError(f"{stage} unexpectedly has a dependency")
    if fields.get("KillOnInvalidDependent", "") not in {"", "No"}:
        raise ValueError(f"{stage} invalid-dependency policy differs")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError(f"{stage} unexpectedly belongs to array/het job")
    if phase == "held":
        expected_submit = (
            f"sbatch --parsable --hold --export=NONE --job-name={config['job_name']} "
            + os.path.relpath(config["sbatch"], REPO_ROOT)
        )
        if (
            fields.get("JobState") != "PENDING"
            or fields.get("Reason") != "JobHeldUser"
            or fields.get("RunTime") != "00:00:00"
            or fields.get("AllocTRES") != "(null)"
            or fields.get("MinMemoryNode") != "180G"
            or fields.get("SubmitLine") != expected_submit
        ):
            raise ValueError(f"{stage} job was not pristine and held-first")
        if check_log_absence:
            for suffix in ("out", "err"):
                if os.path.lexists(
                    LOG_ROOT / f"{config['log_prefix']}_{job_id}.{suffix}"
                ):
                    raise ValueError(f"held {stage} job already created a log")
    elif phase == "running":
        if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None":
            raise ValueError(f"{stage} job is not RUNNING")
        if parse_tres(fields.get("AllocTRES", "")) != expected_tres():
            raise ValueError(f"{stage} allocated TRES differs")
        node = fields.get("NodeList", "")
        if re.fullmatch(r"g[0-9]+", node) is None or fields.get("BatchHost") != node:
            raise ValueError(f"{stage} node allocation differs")
    else:
        raise ValueError("unknown scheduler audit phase")
    return {
        "stage": stage,
        "job_id": str(job_id),
        "job_name": config["job_name"],
        "scheduler_phase": phase,
        "scontrol_record": raw,
        "scontrol_record_sha256": sha256_bytes(raw.encode("utf-8")),
        "requested_tres": expected_tres(),
        "time_limit": config["time_limit"],
        "maximum_h200_minutes": config["minutes"],
        "maximum_gpu_cost_usd": config["cost"],
        "held_first": True,
        "no_requeue": True,
        "dependencies": [],
    }


def audit_live_job(stage, job_id, phase, check_log_absence=True):
    raw, fields = query_job(job_id)
    result = audit_job_record(
        stage, job_id, raw, fields, phase, check_log_absence
    )
    if phase == "held":
        completed = subprocess.run(
            ["scontrol", "write", "batch_script", str(job_id), "-"],
            capture_output=True,
            check=True,
        )
        spooled = completed.stdout
        if isinstance(spooled, str):
            spooled = spooled.encode("utf-8")
        committed = require_regular(
            STAGES[stage]["sbatch"], f"{stage} committed batch script"
        ).read_bytes()
        if spooled != committed:
            raise ValueError(f"Slurm-spooled {stage} script differs from commit")
        result["spooled_script_sha256"] = sha256_bytes(spooled)
        result["committed_script_sha256"] = sha256_bytes(committed)
    return result


def parse_attempt(stage, job_id):
    path = require_regular(STAGES[stage]["attempt"], f"{stage} submission attempt")
    expected = f"stage\tjob_id\n{stage}\t{job_id}\n".encode("utf-8")
    if path.read_bytes() != expected:
        raise ValueError(f"{stage} submission attempt differs")
    return binding(path)


def lock_binding(stage):
    directory = STAGES[stage]["lock"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{stage} permanent submission lock is absent")
    owner = require_regular(directory / "owner", f"{stage} lock owner")
    return {"path": os.fspath(directory), "owner": binding(owner)}


def job_pointer(stage):
    path = STAGES[stage]["job"]
    payload = load_json(path, f"{stage} job pointer")
    body = verify_seal(payload, f"{stage} job pointer")
    expected_keys = {
        "schema_version",
        "workflow_id",
        "created_at",
        "stage",
        "job_id",
        "held_audit",
        "submission_attempt",
        "permanent_submission_lock",
    }
    held = body.get("held_audit")
    if (
        set(body) != expected_keys
        or body.get("schema_version") != SCHEMA_VERSION
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("stage") != stage
        or not re.fullmatch(r"[0-9]+", str(body.get("job_id", "")))
        or not isinstance(held, dict)
        or held.get("scheduler_phase") != "held"
        or held.get("job_id") != body.get("job_id")
    ):
        raise ValueError(f"{stage} job pointer differs")
    reconstructed = audit_job_record(
        stage,
        body["job_id"],
        held.get("scontrol_record"),
        parse_scontrol_line(held.get("scontrol_record")),
        "held",
        check_log_absence=False,
    )
    script_sha = sha256_file(STAGES[stage]["sbatch"])
    expected_held = {
        **reconstructed,
        "spooled_script_sha256": script_sha,
        "committed_script_sha256": script_sha,
    }
    if held != expected_held:
        raise ValueError(f"{stage} stored held-job audit differs")
    if body.get("submission_attempt") != parse_attempt(stage, body["job_id"]):
        raise ValueError(f"{stage} submission-attempt binding differs")
    if body.get("permanent_submission_lock") != lock_binding(stage):
        raise ValueError(f"{stage} permanent-lock binding differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def auth_pointer(stage):
    payload = load_json(STAGES[stage]["auth"], f"{stage} authorization")
    body = verify_seal(payload, f"{stage} authorization")
    job = job_pointer(stage)
    prep = audit_prep()
    expected_keys = {
        "schema_version",
        "workflow_id",
        "created_at",
        "stage",
        "job_id",
        "prep_file_sha256",
        "prep_payload_sha256",
        "job",
        "maximum_h200_minutes",
        "maximum_gpu_cost_usd",
        "combined_released_h200_minutes_cap",
        "combined_released_gpu_cost_usd_cap",
        "pre_release_local_model_snapshot",
        "pre_release_runtime_versions",
        "pre_release_protocol_audit",
        "prior_smoke_actual",
        "actual_plus_current_cap_h200_minutes",
        "actual_plus_current_cap_gpu_cost_usd",
        "held_first",
        "no_requeue",
        "no_dependency",
        "no_retry_or_reserve",
        "training",
        "external_api_calls",
        "automatic_continuation",
        "wave3_v1_submitted_or_released",
    }
    if (
        set(body) != expected_keys
        or body.get("schema_version") != SCHEMA_VERSION
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("stage") != stage
        or body.get("prep_file_sha256") != sha256_file(PREP_FILE)
        or body.get("prep_payload_sha256") != prep["payload_sha256"]
        or body.get("job") != binding(
            STAGES[stage]["job"],
            load_json(STAGES[stage]["job"], f"{stage} job pointer"),
        )
        or body.get("maximum_h200_minutes") != STAGES[stage]["minutes"]
        or body.get("maximum_gpu_cost_usd") != STAGES[stage]["cost"]
        or body.get("combined_released_h200_minutes_cap") != 115
        or body.get("combined_released_gpu_cost_usd_cap") != 1.725
        or body.get("pre_release_local_model_snapshot")
        != prep.get("local_model_snapshot")
        or body.get("pre_release_runtime_versions")
        != prep.get("environment", {}).get("runtime_versions")
        or body.get("pre_release_protocol_audit") != prep.get("protocol")
        or body.get("no_requeue") is not True
        or body.get("no_dependency") is not True
        or body.get("no_retry_or_reserve") is not True
        or body.get("training") is not False
        or body.get("external_api_calls") != 0
        or body.get("wave3_v1_submitted_or_released") is not False
        or job["job_id"] != body.get("job_id")
    ):
        raise ValueError(f"{stage} authorization differs")
    actual_plus_cap = body.get("actual_plus_current_cap_h200_minutes")
    actual_plus_cost = body.get("actual_plus_current_cap_gpu_cost_usd")
    if (
        isinstance(actual_plus_cap, bool)
        or not isinstance(actual_plus_cap, (int, float))
        or isinstance(actual_plus_cost, bool)
        or not isinstance(actual_plus_cost, (int, float))
        or actual_plus_cap > 115
        or actual_plus_cost > 1.725 + 1e-12
    ):
        raise ValueError(f"{stage} actual-plus-cap budget differs")
    if stage == "smoke":
        if (
            body.get("prior_smoke_actual") is not None
            or actual_plus_cap != 15
            or actual_plus_cost != 0.225
        ):
            raise ValueError("smoke actual-plus-cap budget differs")
    else:
        prior = body.get("prior_smoke_actual")
        if (
            not isinstance(prior, dict)
            or prior != terminal_accounting("smoke")
            or not math.isclose(
                actual_plus_cap,
                prior["actual_h200_minutes"] + STAGES[stage]["minutes"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                actual_plus_cost,
                prior["actual_gpu_cost_usd"] + STAGES[stage]["cost"],
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("confirmation actual-plus-cap budget differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def parse_duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value or "")
    if match is None:
        raise ValueError(f"invalid Slurm duration: {value}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid Slurm duration: {value}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def terminal_accounting(stage):
    job = job_pointer(stage)
    job_id = job["job_id"]
    output = subprocess.check_output(
        [
            "sacct",
            "-n",
            "-X",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode",
        ],
        text=True,
    )
    matches = []
    for line in output.strip().splitlines():
        fields = line.split("|")
        if fields and fields[0] == job_id:
            matches.append((line, fields))
    if len(matches) != 1 or len(matches[0][1]) < 8:
        raise ValueError(f"{stage} lacks unique durable accounting")
    line, fields = matches[0]
    _, job_name, state, elapsed, limit, allocated, requested, exit_code = fields[:8]
    config = STAGES[stage]
    if (
        job_name != config["job_name"]
        or state != "COMPLETED"
        or limit != config["time_limit"]
        or exit_code != "0:0"
        or parse_tres(allocated) != expected_tres()
        or parse_tres(requested) != expected_tres()
    ):
        raise ValueError(f"{stage} durable accounting differs")
    seconds = parse_duration(elapsed)
    if not 0 < seconds <= config["minutes"] * 60:
        raise ValueError(f"{stage} elapsed time exceeds its cap")
    minutes = seconds / 60.0
    return {
        "stage": stage,
        "job_id": job_id,
        "sacct_row": line,
        "sacct_row_sha256": sha256_bytes(line.encode("utf-8")),
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": seconds,
        "actual_h200_minutes": minutes,
        "actual_gpu_cost_usd": minutes * RATE_PER_H200_MINUTE_USD,
        "released_h200_minutes_cap": config["minutes"],
        "released_gpu_cost_usd_cap": config["cost"],
    }


def command_write_held_auth(args):
    stage = args.stage
    config = STAGES[stage]
    prep = audit_prep()
    audit_staged()
    pre_release_snapshot = audit_local_model_snapshot()
    if pre_release_snapshot != prep.get("local_model_snapshot"):
        raise ValueError("pre-release local-model snapshot differs from PREP")
    pre_release_runtime = audit_runtime_versions()
    if pre_release_runtime != prep.get("environment", {}).get("runtime_versions"):
        raise ValueError("pre-release runtime differs from PREP")
    pre_release_protocol = protocol_binding(run_full_audit=True)
    if pre_release_protocol != prep.get("protocol"):
        raise ValueError("pre-release protocol/model audit differs from PREP")
    if stage == "confirmation":
        smoke = audit_result("smoke")
        if smoke["scientific_status"] != "EXPLORATORY_SMOKE_PASSED":
            raise ValueError("confirmation requires a passing sealed smoke")
        smoke_accounting = terminal_accounting("smoke")
    else:
        smoke_accounting = None
    for path in (config["job"], config["auth"], config["result"]):
        if os.path.lexists(path):
            raise ValueError(f"{stage} control pointer already exists: {path}")
    held = audit_live_job(stage, args.job_id, "held")
    attempt = parse_attempt(stage, str(args.job_id))
    lock = lock_binding(stage)
    job_body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "job_id": str(args.job_id),
        "held_audit": held,
        "submission_attempt": attempt,
        "permanent_submission_lock": lock,
    }
    job_payload = write_sealed_once(config["job"], job_body)
    auth_body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "job_id": str(args.job_id),
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": load_json(PREP_FILE, "workflow PREP")[
            "payload_sha256"
        ],
        "job": binding(config["job"], job_payload),
        "maximum_h200_minutes": config["minutes"],
        "maximum_gpu_cost_usd": config["cost"],
        "combined_released_h200_minutes_cap": 115,
        "combined_released_gpu_cost_usd_cap": 1.725,
        "pre_release_local_model_snapshot": pre_release_snapshot,
        "pre_release_runtime_versions": pre_release_runtime,
        "pre_release_protocol_audit": pre_release_protocol,
        "prior_smoke_actual": smoke_accounting,
        "actual_plus_current_cap_h200_minutes": (
            config["minutes"]
            if smoke_accounting is None
            else smoke_accounting["actual_h200_minutes"] + config["minutes"]
        ),
        "actual_plus_current_cap_gpu_cost_usd": (
            config["cost"]
            if smoke_accounting is None
            else smoke_accounting["actual_gpu_cost_usd"] + config["cost"]
        ),
        "held_first": True,
        "no_requeue": True,
        "no_dependency": True,
        "no_retry_or_reserve": True,
        "training": False,
        "external_api_calls": 0,
        "automatic_continuation": False,
        "wave3_v1_submitted_or_released": False,
    }
    if (
        auth_body["actual_plus_current_cap_h200_minutes"] > 115
        or auth_body["actual_plus_current_cap_gpu_cost_usd"] > 1.725 + 1e-12
    ):
        raise ValueError("actual-plus-cap budget exceeds the released ceiling")
    payload = write_sealed_once(config["auth"], auth_body)
    print(payload["payload_sha256"])


def command_audit_held(args):
    auth = auth_pointer(args.stage)
    if auth["job_id"] != str(args.job_id):
        raise ValueError("held audit job ID differs from authorization")
    audit_live_job(args.stage, args.job_id, "held")
    if audit_local_model_snapshot() != auth["pre_release_local_model_snapshot"]:
        raise ValueError("held-to-release local-model snapshot drifted")
    if audit_runtime_versions() != auth["pre_release_runtime_versions"]:
        raise ValueError("held-to-release runtime drifted")
    if protocol_binding(run_full_audit=True) != auth["pre_release_protocol_audit"]:
        raise ValueError("held-to-release protocol/model inputs drifted")
    print(f"{args.stage.upper()}_HELD_AUDIT_OK")


def command_verify_job(args):
    auth = auth_pointer(args.stage)
    if auth["job_id"] != str(args.job_id):
        raise ValueError("running job ID differs from authorization")
    if audit_local_model_snapshot() != auth["pre_release_local_model_snapshot"]:
        raise ValueError("job-start local-model snapshot drifted")
    if audit_runtime_versions() != auth["pre_release_runtime_versions"]:
        raise ValueError("job-start runtime drifted")
    if protocol_binding(run_full_audit=True) != auth["pre_release_protocol_audit"]:
        raise ValueError("job-start protocol/model inputs drifted")
    release_path = require_regular(
        STAGES[args.stage]["released"], f"{args.stage} release marker"
    )
    expected_release = (
        f"stage={args.stage}\njob_id={args.job_id}\n"
        "held_audit_passed=true\nrelease_authorized=true\n"
    ).encode("utf-8")
    if release_path.read_bytes() != expected_release:
        raise ValueError(f"{args.stage} release marker differs")
    running = audit_live_job(
        args.stage, args.job_id, "running", check_log_absence=False
    )
    fields = parse_scontrol_line(running["scontrol_record"])
    expected_environment = {
        "SLURM_JOB_ID": str(args.job_id),
        "SLURM_JOB_NAME": STAGES[args.stage]["job_name"],
        "SLURM_JOB_PARTITION": "gpu-h200",
        "SLURM_JOB_ACCOUNT": "stf",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": os.fspath(REPO_ROOT),
        "SLURM_JOB_NODELIST": fields["NodeList"],
        "SLURM_MEM_PER_NODE": "184320",
    }
    for key, expected in expected_environment.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"{args.stage} running environment differs on {key}")
    print(f"{args.stage.upper()}_RUNNING_AUDIT_OK")


def load_preflight(stage):
    path = STAGES[stage]["preflight"]
    payload = load_json(path, f"{stage} CPU preflight")
    expected_keys = {
        "status",
        "runtime",
        "schema_sha256",
        "intent_leaves_checked",
        "slot_leaves_checked",
        "invalid_probes_rejected",
        "recorded_hybrid_intent_probes_rejected",
        "recorded_hybrid_slot_probes_rejected",
        "flexible_whitespace_probes_reproduced",
        "whitespace_probes_rejected",
    }
    expected_runtime = {
        key: EXPECTED_RUNTIME_VERSIONS[key]
        for key in ("torch", "transformers", "peft", "xgrammar")
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("status") != "CPU_PREFLIGHT_OK"
        or payload.get("runtime") != expected_runtime
        or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("schema_sha256", "")))
        or payload.get("intent_leaves_checked") != 60
        or payload.get("slot_leaves_checked") != 55
        or payload.get("invalid_probes_rejected") != 9
        or payload.get("recorded_hybrid_intent_probes_rejected") != 4
        or payload.get("recorded_hybrid_slot_probes_rejected") != 3
        or payload.get("flexible_whitespace_probes_reproduced") != 2
        or payload.get("whitespace_probes_rejected") != 2
    ):
        raise ValueError(f"{stage} CPU preflight differs")
    return binding(path)


def command_audit_preflight(args):
    result = load_preflight(args.stage)
    print(
        json.dumps(
            {
                "status": "CPU_PREFLIGHT_AUDIT_OK",
                "stage": args.stage,
                "file_sha256": result["file_sha256"],
            },
            sort_keys=True,
        )
    )


def expected_stream_plan(stage):
    massive_rows = 60 if stage == "smoke" else 600
    plan = [
        {"method_id": "pi_base", "domain": "massive", "samples": massive_rows},
        *[
            {"method_id": method, "domain": "massive", "samples": massive_rows}
            for method in METHOD_IDS
        ],
    ]
    if stage == "confirmation":
        plan.extend(
            {"method_id": method, "domain": "medical", "samples": 80}
            for method in METHOD_IDS
        )
    return plan


def audit_generation_file(stage, stream, schema_sha256):
    path = (
        GENERATION_ROOT
        / stage
        / stream["method_id"]
        / stream["domain"]
        / "generation.json"
    )
    payload = load_json(path, f"{stage} {stream['method_id']} generation")
    body = verify_seal(payload, f"{stage} {stream['method_id']} generation")
    if set(body) != {"meta", "samples"}:
        raise ValueError(f"{stage} generation fields differ")
    meta, samples = body["meta"], body["samples"]
    expected_domain = "MASSIVE" if stream["domain"] == "massive" else "medical"
    if (
        not isinstance(meta, dict)
        or meta.get("protocol")
        != "massive_medical_union_composition_exploratory_generation_v1"
        or meta.get("phase") != stage
        or meta.get("method_id") != stream["method_id"]
        or meta.get("domain") != expected_domain
        or meta.get("runtime_pins")
        != {
            key: EXPECTED_RUNTIME_VERSIONS[key]
            for key in ("torch", "transformers", "peft", "xgrammar")
        }
        or not isinstance(samples, list)
        or len(samples) != stream["samples"]
    ):
        raise ValueError(f"{stage} generation provenance differs")
    if stream["domain"] == "massive":
        config = meta.get("generation_config")
        if (
            not isinstance(config, dict)
            or config.get("json_schema_sha256") != schema_sha256
        ):
            raise ValueError(f"{stage} MASSIVE schema/preflight binding differs")
    return binding(path, payload)


def load_run_manifest(stage):
    path = GENERATION_ROOT / stage / "run_manifest.json"
    payload = load_json(path, f"{stage} generation run manifest")
    body = verify_seal(payload, f"{stage} generation run manifest")
    protocol = protocol_binding(False)
    expected_streams = expected_stream_plan(stage)
    preflight = load_json(STAGES[stage]["preflight"], f"{stage} CPU preflight")
    if (
        set(body)
        != {
            "schema_version",
            "protocol",
            "phase",
            "protocol_manifest_file_sha256",
            "protocol_manifest_payload_sha256",
            "streams",
        }
        or body.get("schema_version") != 1
        or body.get("protocol")
        != "massive_medical_union_composition_exploratory_generation_v1"
        or body.get("phase") != stage
        or body.get("protocol_manifest_file_sha256") != protocol["file_sha256"]
        or body.get("protocol_manifest_payload_sha256")
        != protocol["payload_sha256"]
        or body.get("streams") != expected_streams
    ):
        raise ValueError(f"{stage} generation run manifest differs")
    for stream in expected_streams:
        audit_generation_file(stage, stream, preflight["schema_sha256"])
    return binding(path, payload)


def load_gate(stage):
    config = STAGES[stage]
    root = config["gate_root"]
    present = [name for name in config["allowed_gate"] if os.path.lexists(root / name)]
    if len(present) != 1:
        raise ValueError(f"{stage} gate lacks exactly one allowed sentinel")
    status = present[0]
    sentinel_path = root / status
    sentinel_payload = load_json(sentinel_path, f"{stage} gate sentinel")
    sentinel = verify_seal(sentinel_payload, f"{stage} gate sentinel")
    summary_path = root / "summary.json"
    summary_payload = load_json(summary_path, f"{stage} gate summary")
    summary = verify_seal(summary_payload, f"{stage} gate summary")
    protocol = protocol_binding(False)
    expected_protocol = (
        "massive_medical_union_composition_exploratory_smoke_sentinel_v1"
        if stage == "smoke"
        else "massive_medical_union_composition_exploratory_prejudge_sentinel_v1"
    )
    if (
        sentinel.get("protocol") != expected_protocol
        or sentinel.get("protocol_id") != PROTOCOL_ID
        or sentinel.get("status") != status
        or Path(sentinel.get("summary_path", "")).resolve() != summary_path.resolve()
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
        or summary.get("status") != status
        or summary.get("protocol_manifest_file_sha256") != protocol["file_sha256"]
        or summary.get("protocol_manifest_payload_sha256")
        != protocol["payload_sha256"]
    ):
        raise ValueError(f"{stage} gate summary/sentinel binding differs")
    for source in (sentinel, summary):
        if (
            source.get("confirmatory_claim") is not False
            or source.get("wave2_v1_status") != "STOP"
            or source.get("wave3_v1_eligible") is not False
            or source.get("wave3_v1_submitted_or_released") is not False
        ):
            raise ValueError(f"{stage} exploratory safety flags differ")
    if stage == "smoke":
        passed = status == "EXPLORATORY_SMOKE_PASSED"
        if (
            summary.get("all_three_methods_passed") is not passed
            or summary.get("confirmation_submission_eligible") is not passed
        ):
            raise ValueError("smoke gate eligibility differs")
    else:
        passed = status == "AWAITING_EXTERNAL_JUDGE"
        if (
            summary.get("all_three_methods_passed") is not passed
            or summary.get("external_judge_calls_authorized") != (240 if passed else 0)
        ):
            raise ValueError("confirmation prejudge authorization differs")
    return {
        "status": status,
        "summary": binding(summary_path, summary_payload),
        "sentinel": binding(sentinel_path, sentinel_payload),
    }


def result_body(stage, job_id, created_at=None):
    auth = auth_pointer(stage)
    if auth["job_id"] != str(job_id):
        raise ValueError(f"{stage} result job differs from authorization")
    running = audit_live_job(stage, job_id, "running", check_log_absence=False)
    gate = load_gate(stage)
    scientific_status = gate["status"]
    terminal = (
        "EXPLORATORY_NO_SUPPORT"
        if scientific_status.startswith("STOPPED_")
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "job_id": str(job_id),
        "job": binding(
            STAGES[stage]["job"],
            load_json(STAGES[stage]["job"], f"{stage} job pointer"),
        ),
        "authorization": binding(
            STAGES[stage]["auth"],
            load_json(STAGES[stage]["auth"], f"{stage} authorization"),
        ),
        "running_job_audit": running,
        "cpu_preflight": load_preflight(stage),
        "generation_run_manifest": load_run_manifest(stage),
        "gate": gate,
        "scientific_status": scientific_status,
        "terminal_scientific_status": terminal,
        "confirmation_submission_eligible": scientific_status
        == "EXPLORATORY_SMOKE_PASSED",
        "external_judge_eligible": scientific_status == "AWAITING_EXTERNAL_JUDGE",
        "training": False,
        "external_api_calls": 0,
        "automatic_continuation": False,
        "no_retry_or_reserve": True,
        "confirmatory_claim": False,
        "wave2_v1_status": "STOP",
        "wave3_v1_eligible": False,
        "wave3_v1_submitted_or_released": False,
    }


def command_write_result(args):
    payload = write_sealed_once(
        STAGES[args.stage]["result"], result_body(args.stage, args.job_id)
    )
    print(payload["payload_sha256"])


def audit_result(stage):
    path = STAGES[stage]["result"]
    payload = load_json(path, f"{stage} result")
    body = verify_seal(payload, f"{stage} result")
    auth = auth_pointer(stage)
    gate = load_gate(stage)
    expected_keys = {
        "schema_version",
        "workflow_id",
        "created_at",
        "stage",
        "job_id",
        "job",
        "authorization",
        "running_job_audit",
        "cpu_preflight",
        "generation_run_manifest",
        "gate",
        "scientific_status",
        "terminal_scientific_status",
        "confirmation_submission_eligible",
        "external_judge_eligible",
        "training",
        "external_api_calls",
        "automatic_continuation",
        "no_retry_or_reserve",
        "confirmatory_claim",
        "wave2_v1_status",
        "wave3_v1_eligible",
        "wave3_v1_submitted_or_released",
    }
    running = body.get("running_job_audit")
    if not isinstance(running, dict):
        raise ValueError(f"{stage} stored running-job audit is missing")
    reconstructed_running = audit_job_record(
        stage,
        auth["job_id"],
        running.get("scontrol_record"),
        parse_scontrol_line(running.get("scontrol_record")),
        "running",
        check_log_absence=False,
    )
    if (
        set(body) != expected_keys
        or body.get("schema_version") != SCHEMA_VERSION
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("stage") != stage
        or body.get("job_id") != auth["job_id"]
        or body.get("job")
        != binding(STAGES[stage]["job"], load_json(STAGES[stage]["job"], "job"))
        or body.get("authorization")
        != binding(STAGES[stage]["auth"], load_json(STAGES[stage]["auth"], "auth"))
        or running != reconstructed_running
        or body.get("cpu_preflight") != load_preflight(stage)
        or body.get("generation_run_manifest") != load_run_manifest(stage)
        or body.get("gate") != gate
        or body.get("scientific_status") != gate["status"]
        or body.get("terminal_scientific_status")
        != ("EXPLORATORY_NO_SUPPORT" if gate["status"].startswith("STOPPED_") else None)
        or body.get("training") is not False
        or body.get("external_api_calls") != 0
        or body.get("automatic_continuation") is not False
        or body.get("no_retry_or_reserve") is not True
        or body.get("confirmatory_claim") is not False
        or body.get("wave2_v1_status") != "STOP"
        or body.get("wave3_v1_eligible") is not False
        or body.get("wave3_v1_submitted_or_released") is not False
    ):
        raise ValueError(f"{stage} result differs")
    if (
        body.get("confirmation_submission_eligible")
        is not (gate["status"] == "EXPLORATORY_SMOKE_PASSED")
        or body.get("external_judge_eligible")
        is not (gate["status"] == "AWAITING_EXTERNAL_JUDGE")
    ):
        raise ValueError(f"{stage} downstream eligibility differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_result(args):
    result = audit_result(args.stage)
    print(
        json.dumps(
            {
                "status": "RESULT_OK",
                "stage": args.stage,
                "scientific_status": result["scientific_status"],
            },
            sort_keys=True,
        )
    )


def command_audit_terminal(args):
    audit_result(args.stage)
    print(json.dumps(terminal_accounting(args.stage), sort_keys=True))


def command_assert_confirmation_release(_args):
    smoke = audit_result("smoke")
    if (
        smoke.get("scientific_status") != "EXPLORATORY_SMOKE_PASSED"
        or smoke.get("confirmation_submission_eligible") is not True
        or smoke.get("terminal_scientific_status") is not None
    ):
        raise ValueError("confirmation release requires sealed smoke PASS")
    accounting = terminal_accounting("smoke")
    print(
        json.dumps(
            {
                "status": "CONFIRMATION_RELEASE_ELIGIBLE",
                "smoke_job_id": accounting["job_id"],
                "smoke_actual_h200_minutes": accounting["actual_h200_minutes"],
            },
            sort_keys=True,
        )
    )


def finalizer_lock_binding():
    if FINALIZER_LOCK.is_symlink() or not FINALIZER_LOCK.is_dir():
        raise ValueError("permanent finalizer lock is absent")
    owner = require_regular(FINALIZER_LOCK / "owner", "finalizer lock owner")
    return {"path": os.fspath(FINALIZER_LOCK), "owner": binding(owner)}


def audit_fresh_finalizer_namespace():
    for path in (FINAL_AUTH, FINAL_RESULT, JUDGE_CHECKPOINT, JUDGE_NEW, JUDGE_MERGED):
        if os.path.lexists(path):
            raise ValueError(f"finalizer output already exists: {path}")
    if os.path.lexists(FINAL_GATE_ROOT):
        if FINAL_GATE_ROOT.is_symlink() or not FINAL_GATE_ROOT.is_dir():
            raise ValueError("final gate namespace exists but is unsafe")
        entries = sorted(FINAL_GATE_ROOT.iterdir(), key=lambda path: path.name)
        if entries:
            raise ValueError(
                "final gate namespace is not fresh: "
                + ", ".join(path.name for path in entries)
            )


def command_write_final_auth(_args):
    confirmation = audit_result("confirmation")
    if confirmation["scientific_status"] != "AWAITING_EXTERNAL_JUDGE":
        raise ValueError("finalizer requires sealed AWAITING_EXTERNAL_JUDGE")
    finalizer_lock = finalizer_lock_binding()
    audit_fresh_finalizer_namespace()
    smoke_accounting = terminal_accounting("smoke")
    confirmation_accounting = terminal_accounting("confirmation")
    prep = audit_prep()
    live_protocol = protocol_binding(run_full_audit=True)
    live_runtime = audit_runtime_versions()
    if live_protocol != prep.get("protocol"):
        raise ValueError("pre-finalizer protocol/source evidence differs from PREP")
    if live_runtime != prep.get("environment", {}).get("runtime_versions"):
        raise ValueError("pre-finalizer runtime differs from PREP")
    actual_minutes = (
        smoke_accounting["actual_h200_minutes"]
        + confirmation_accounting["actual_h200_minutes"]
    )
    actual_gpu_cost = actual_minutes * RATE_PER_H200_MINUTE_USD
    if actual_minutes > 115 or actual_gpu_cost + 0.75 > 2.475 + 1e-12:
        raise ValueError("actual GPU plus external-judge cap exceeds all-in ceiling")
    body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "confirmation_result": binding(
            STAGES["confirmation"]["result"],
            load_json(STAGES["confirmation"]["result"], "confirmation result"),
        ),
        "permanent_finalizer_lock": finalizer_lock,
        "pre_finalizer_protocol_audit": live_protocol,
        "pre_finalizer_runtime_versions": live_runtime,
        "smoke_accounting": smoke_accounting,
        "confirmation_accounting": confirmation_accounting,
        "actual_gpu_h200_minutes": actual_minutes,
        "actual_gpu_cost_usd": actual_gpu_cost,
        "released_gpu_h200_minutes_cap": 115,
        "released_gpu_cost_usd_cap": 1.725,
        "maximum_external_judge_calls": 240,
        "maximum_external_judge_cost_usd": 0.75,
        "maximum_all_in_released_cost_usd": 2.475,
        "sdk_retries": 0,
        "historical_A_reused": True,
        "A_rejudged": False,
        "merge_api_calls": 0,
        "no_retry_or_reserve": True,
        "confirmatory_claim": False,
        "wave2_v1_status": "STOP",
        "wave3_v1_eligible": False,
        "wave3_v1_submitted_or_released": False,
    }
    payload = write_sealed_once(FINAL_AUTH, body)
    print(payload["payload_sha256"])


def audit_final_auth():
    payload = load_json(FINAL_AUTH, "external judge authorization")
    body = verify_seal(payload, "external judge authorization")
    confirmation = audit_result("confirmation")
    expected_keys = {
        "schema_version",
        "workflow_id",
        "created_at",
        "confirmation_result",
        "permanent_finalizer_lock",
        "pre_finalizer_protocol_audit",
        "pre_finalizer_runtime_versions",
        "smoke_accounting",
        "confirmation_accounting",
        "actual_gpu_h200_minutes",
        "actual_gpu_cost_usd",
        "released_gpu_h200_minutes_cap",
        "released_gpu_cost_usd_cap",
        "maximum_external_judge_calls",
        "maximum_external_judge_cost_usd",
        "maximum_all_in_released_cost_usd",
        "sdk_retries",
        "historical_A_reused",
        "A_rejudged",
        "merge_api_calls",
        "no_retry_or_reserve",
        "confirmatory_claim",
        "wave2_v1_status",
        "wave3_v1_eligible",
        "wave3_v1_submitted_or_released",
    }
    smoke_accounting = terminal_accounting("smoke")
    confirmation_accounting = terminal_accounting("confirmation")
    prep = audit_prep()
    live_protocol = protocol_binding(run_full_audit=True)
    live_runtime = audit_runtime_versions()
    actual_minutes = (
        smoke_accounting["actual_h200_minutes"]
        + confirmation_accounting["actual_h200_minutes"]
    )
    if (
        set(body) != expected_keys
        or body.get("schema_version") != SCHEMA_VERSION
        or confirmation["scientific_status"] != "AWAITING_EXTERNAL_JUDGE"
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("confirmation_result")
        != binding(
            STAGES["confirmation"]["result"],
            load_json(STAGES["confirmation"]["result"], "confirmation result"),
        )
        or body.get("permanent_finalizer_lock") != finalizer_lock_binding()
        or body.get("pre_finalizer_protocol_audit") != prep.get("protocol")
        or body.get("pre_finalizer_protocol_audit") != live_protocol
        or body.get("pre_finalizer_runtime_versions")
        != prep.get("environment", {}).get("runtime_versions")
        or body.get("pre_finalizer_runtime_versions") != live_runtime
        or body.get("smoke_accounting") != smoke_accounting
        or body.get("confirmation_accounting") != confirmation_accounting
        or not math.isclose(
            body.get("actual_gpu_h200_minutes", math.inf),
            actual_minutes,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            body.get("actual_gpu_cost_usd", math.inf),
            actual_minutes * RATE_PER_H200_MINUTE_USD,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or body.get("released_gpu_h200_minutes_cap") != 115
        or body.get("released_gpu_cost_usd_cap") != 1.725
        or body.get("maximum_external_judge_calls") != 240
        or body.get("maximum_external_judge_cost_usd") != 0.75
        or body.get("maximum_all_in_released_cost_usd") != 2.475
        or body.get("sdk_retries") != 0
        or body.get("historical_A_reused") is not True
        or body.get("A_rejudged") is not False
        or body.get("merge_api_calls") != 0
        or body.get("no_retry_or_reserve") is not True
        or body.get("confirmatory_claim") is not False
        or body.get("wave2_v1_status") != "STOP"
        or body.get("wave3_v1_eligible") is not False
        or body.get("wave3_v1_submitted_or_released") is not False
    ):
        raise ValueError("external judge authorization differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_final_auth(_args):
    payload = audit_final_auth()
    print(
        json.dumps(
            {
                "status": "FINAL_AUTH_OK",
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


def command_historical_a_path(_args):
    manifest = load_json(PROTOCOL_MANIFEST, "protocol manifest")
    body = verify_seal(
        manifest, "protocol manifest", field="manifest_payload_sha256"
    )
    historical = body.get("source_wave2_terminal", {}).get(
        "historical_A_judgments"
    )
    path = historical.get("path") if isinstance(historical, dict) else None
    if (
        not isinstance(path, str)
        or binding(path, load_json(path, "historical A judgments")) != {
            key: historical[key]
            for key in (
                "path",
                "size_bytes",
                "file_sha256",
                "payload_sha256",
                "payload_seal_field",
            )
        }
    ):
        raise ValueError("historical A judgment binding differs")
    print(path)


def load_final_gate():
    allowed = ("EXPLORATORY_SUPPORT", "EXPLORATORY_NO_SUPPORT")
    present = [name for name in allowed if os.path.lexists(FINAL_GATE_ROOT / name)]
    if len(present) != 1:
        raise ValueError("final gate lacks exactly one terminal sentinel")
    status = present[0]
    sentinel_path = FINAL_GATE_ROOT / status
    sentinel_payload = load_json(sentinel_path, "final gate sentinel")
    sentinel = verify_seal(sentinel_payload, "final gate sentinel")
    summary_path = FINAL_GATE_ROOT / "summary.json"
    summary_payload = load_json(summary_path, "final gate summary")
    summary = verify_seal(summary_payload, "final gate summary")
    if (
        sentinel.get("protocol")
        != "massive_medical_union_composition_exploratory_final_sentinel_v1"
        or sentinel.get("status") != status
        or Path(sentinel.get("summary_path", "")).resolve() != summary_path.resolve()
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
        or summary.get("status") != status
        or summary.get("all_three_methods_passed") is not (
            status == "EXPLORATORY_SUPPORT"
        )
    ):
        raise ValueError("final gate summary/sentinel differs")
    for source in (sentinel, summary):
        if (
            source.get("confirmatory_claim") is not False
            or source.get("wave2_v1_status") != "STOP"
            or source.get("wave3_v1_eligible") is not False
            or source.get("wave3_v1_submitted_or_released") is not False
        ):
            raise ValueError("final exploratory safety flags differ")
    return {
        "status": status,
        "summary": binding(summary_path, summary_payload),
        "sentinel": binding(sentinel_path, sentinel_payload),
    }


def command_write_final_result(_args):
    auth = audit_final_auth()
    new_payload = load_json(JUDGE_NEW, "new composition judgments")
    new_body = verify_seal(new_payload, "new composition judgments")
    meta = new_body.get("meta")
    rows = new_body.get("judgments")
    if (
        not isinstance(meta, dict)
        or meta.get("protocol")
        != "massive_medical_union_composition_exploratory_judge_v1"
        or meta.get("actual_api_calls") != 240
        or meta.get("max_api_calls") != 240
        or meta.get("max_cost_usd") != 0.75
        or meta.get("sdk_max_retries") != 0
        or meta.get("actual_estimated_cost_usd", math.inf) > 0.75
        or not isinstance(rows, list)
        or len(rows) != 240
    ):
        raise ValueError("new composition judgment accounting differs")
    merged_payload = load_json(JUDGE_MERGED, "merged medical judgments")
    merged_body = verify_seal(merged_payload, "merged medical judgments")
    merged_meta, merged_rows = merged_body.get("meta"), merged_body.get("judgments")
    if (
        not isinstance(merged_meta, dict)
        or merged_meta.get("protocol")
        != "massive_medical_union_composition_exploratory_merged_judgments_v1"
        or merged_meta.get("historical_rows_reused") != 80
        or merged_meta.get("new_rows") != 240
        or merged_meta.get("merge_api_calls") != 0
        or not isinstance(merged_rows, list)
        or len(merged_rows) != 320
    ):
        raise ValueError("merged medical judgment accounting differs")
    gate = load_final_gate()
    body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "external_judge_authorization": binding(
            FINAL_AUTH, load_json(FINAL_AUTH, "external judge authorization")
        ),
        "external_judge_actual_calls": 240,
        "external_judge_actual_cost_usd": meta["actual_estimated_cost_usd"],
        "new_judgments": binding(JUDGE_NEW, new_payload),
        "merged_judgments": binding(JUDGE_MERGED, merged_payload),
        "final_gate": gate,
        "scientific_status": gate["status"],
        "training": False,
        "merge_api_calls": 0,
        "no_retry_or_reserve": True,
        "confirmatory_claim": False,
        "wave2_v1_status": "STOP",
        "wave3_v1_eligible": False,
        "wave3_v1_submitted_or_released": False,
    }
    payload = write_sealed_once(FINAL_RESULT, body)
    print(payload["payload_sha256"])


def audit_final_result():
    payload = load_json(FINAL_RESULT, "final result")
    body = verify_seal(payload, "final result")
    auth = audit_final_auth()
    gate = load_final_gate()
    if (
        body.get("workflow_id") != WORKFLOW_ID
        or body.get("external_judge_authorization")
        != binding(FINAL_AUTH, load_json(FINAL_AUTH, "final authorization"))
        or body.get("external_judge_actual_calls") != 240
        or body.get("external_judge_actual_cost_usd", math.inf) > 0.75
        or body.get("final_gate") != gate
        or body.get("scientific_status") != gate["status"]
        or body.get("merge_api_calls") != 0
        or body.get("no_retry_or_reserve") is not True
        or body.get("wave3_v1_submitted_or_released") is not False
        or auth.get("maximum_external_judge_calls") != 240
    ):
        raise ValueError("final result differs")
    for artifact, description in (
        (JUDGE_NEW, "new judgments"),
        (JUDGE_MERGED, "merged judgments"),
    ):
        key = "new_judgments" if artifact == JUDGE_NEW else "merged_judgments"
        observed = load_json(artifact, description)
        if body.get(key) != binding(artifact, observed):
            raise ValueError(f"final result {description} binding differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_final_result(_args):
    result = audit_final_result()
    print(
        json.dumps(
            {"status": "FINAL_RESULT_OK", "scientific_status": result["scientific_status"]},
            sort_keys=True,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-prep").set_defaults(function=command_write_prep)
    commands.add_parser("audit-prep").set_defaults(function=command_audit_prep)
    commands.add_parser("audit-staged").set_defaults(function=command_audit_staged)
    for name, function in (
        ("write-held-auth", command_write_held_auth),
        ("audit-held", command_audit_held),
        ("verify-job", command_verify_job),
        ("write-result", command_write_result),
    ):
        item = commands.add_parser(name)
        item.add_argument("--stage", choices=sorted(STAGES), required=True)
        item.add_argument("--job-id", required=True)
        item.set_defaults(function=function)
    item = commands.add_parser("audit-result")
    item.add_argument("--stage", choices=sorted(STAGES), required=True)
    item.set_defaults(function=command_audit_result)
    item = commands.add_parser("audit-terminal")
    item.add_argument("--stage", choices=sorted(STAGES), required=True)
    item.set_defaults(function=command_audit_terminal)
    item = commands.add_parser("audit-preflight")
    item.add_argument("--stage", choices=sorted(STAGES), required=True)
    item.set_defaults(function=command_audit_preflight)
    commands.add_parser("write-final-auth").set_defaults(
        function=command_write_final_auth
    )
    commands.add_parser("audit-final-auth").set_defaults(
        function=command_audit_final_auth
    )
    commands.add_parser("assert-confirmation-release").set_defaults(
        function=command_assert_confirmation_release
    )
    commands.add_parser("historical-a-path").set_defaults(
        function=command_historical_a_path
    )
    commands.add_parser("write-final-result").set_defaults(
        function=command_write_final_result
    )
    commands.add_parser("audit-final-result").set_defaults(
        function=command_audit_final_result
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.function(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"AUDIT_FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
