#!/usr/bin/env python3
"""Fail-closed auditor for the one-shot exploratory smoke recovery.

The recovery is infrastructure-only.  This program never submits, releases,
cancels, or requeues a job and never calls an external API.
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
WORKFLOW_ID = (
    "massive_medical_union_composition_exploratory_workflow_smoke_recovery_v1"
)
PROTOCOL_ID = (
    "massive_medical_union_composition_exploratory_protocol_smoke_recovery_v1"
)
SOURCE_WORKFLOW_ID = "massive_medical_union_composition_exploratory_workflow_v1"
SOURCE_PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
SOURCE_COMMIT = "f95df49a2b7d552bed0e8f6e5ceee616495b38a9"
SOURCE_TREE = "baef063a616e79e14fab1e89d64dae94737b0765"
SOURCE_JOB_ID = "261152"
BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-smoke-recovery-v1"
)
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
SOURCE_REPO = (
    TILLICUM_ROOT
    / "projects/subliminal-mitigate-mmu-composition-exploratory-v1"
)
REPO_ROOT = (
    TILLICUM_ROOT
    / "projects/subliminal-mitigate-mmu-composition-exploratory-smoke-recovery-v1"
)
SOURCE_OUTPUT = (
    TILLICUM_ROOT / "outputs/massive_medical_union_composition_exploratory_v1"
)
SOURCE_PROTOCOL_ROOT = SOURCE_OUTPUT / "protocol"
SOURCE_PROTOCOL_MANIFEST = SOURCE_PROTOCOL_ROOT / "manifest.json"
SOURCE_CONTROL_ROOT = SOURCE_OUTPUT / "control"
SOURCE_GENERATION_ROOT = SOURCE_OUTPUT / "generation"
SOURCE_EVALUATION_ROOT = SOURCE_OUTPUT / "evaluation"
OUTPUT_ROOT = (
    TILLICUM_ROOT
    / "outputs/massive_medical_union_composition_exploratory_smoke_recovery_v1"
)
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
PREP_FILE = CONTROL_ROOT / "PREP.json"
PREFLIGHT_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_CPU_PREFLIGHT.json"
STAGED_FILE = CONTROL_ROOT / "STAGED"
LOCK_ROOT = CONTROL_ROOT / "SMOKE_RECOVERY_SUBMISSION_LOCK"
ATTEMPT_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_SUBMISSION_ATTEMPT.tsv"
JOB_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_JOB.json"
AUTH_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_AUTHORIZED_MAX_COST_USD_0.225.json"
SUBMITTED_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_SUBMITTED"
RELEASE_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_RELEASE_AUTHORIZED"
RESULT_FILE = CONTROL_ROOT / "SMOKE_RECOVERY_RESULT.json"
STOP_FILE = CONTROL_ROOT / "STOPPED_smoke_recovery"
GATE_ROOT = EVALUATION_ROOT / "smoke/gate"
SBATCH_FILE = (
    REPO_ROOT
    / "scripts/sbatch_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum_h200.sbatch"
)
JOB_NAME = "mmu_cmpx_smoke_rec_v1"
LOG_PREFIX = "massive_medical_union_composition_exploratory_smoke_recovery_v1"
RATE_PER_H200_MINUTE_USD = 0.015
PRIOR_ACTUAL_SECONDS = 35
PRIOR_ACTUAL_H200_MINUTES = PRIOR_ACTUAL_SECONDS / 60.0
PRIOR_ACTUAL_COST_USD = 0.00875
RECOVERY_CAP_MINUTES = 15
RECOVERY_CAP_COST_USD = 0.225
ACTUAL_PLUS_RECOVERY_CAP_COST_USD = 0.23375
FIELD_RE = re.compile(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=")

MODIFIED_FILES = (
    "scripts/sample_massive_medical_union_composition_exploratory_v1.py",
    "tests/test_massive_medical_union_composition_exploratory_sampler.py",
)
ADDED_FILES = (
    "docs/massive_medical_union_composition_exploratory_smoke_recovery_v1.md",
    "scripts/audit_massive_medical_union_composition_exploratory_smoke_recovery_v1.py",
    "scripts/stage_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh",
    "scripts/sbatch_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum_h200.sbatch",
    "scripts/submit_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_smoke_recovery_v1_tillicum.sh",
    "tests/test_massive_medical_union_composition_exploratory_smoke_recovery_workflow.py",
)
EXECUTABLE_FILES = tuple(
    path
    for path in (*MODIFIED_FILES, *ADDED_FILES)
    if path.startswith("scripts/")
)
REGULAR_FILES = tuple(
    path
    for path in (*MODIFIED_FILES, *ADDED_FILES)
    if not path.startswith("scripts/")
)

EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.0+cu129",
    "transformers": "4.57.6",
    "peft": "0.18.1",
    "xgrammar": "0.1.25",
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
EXPECTED_SNAPSHOT_ARTIFACTS = {
    "config.json": (663, "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"),
    "generation_config.json": (243, "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f"),
    "tokenizer_config.json": (7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "tokenizer.json": (7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "model.safetensors.index.json": (27752, "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028"),
}
EXPECTED_SHARDS = {
    "model-00001-of-00004.safetensors": (3945441440, "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7"),
    "model-00002-of-00004.safetensors": (3864726352, "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185"),
    "model-00003-of-00004.safetensors": (3864726424, "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5"),
    "model-00004-of-00004.safetensors": (3556377672, "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd"),
}

SOURCE_CONTROL_FILES = {
    "CONFIRMATION_CPU_PREFLIGHT.json": (470, "2da82d283f017fbdcf50881ee24d360c82945a41363295e4c73de0430d8ac44d"),
    "PREP.json": (15281, "b7a22e16c2c6a44cd530bb2b5bb1292c94d10d9f22853a0660491964d9fd13a7"),
    "SMOKE_AUTHORIZED_MAX_COST_USD_0.225.json": (7840, "a22e34d4508270d566f3b561c73f83d57650e794bce98ea450112534ccc76ac8"),
    "SMOKE_CPU_PREFLIGHT.json": (470, "2da82d283f017fbdcf50881ee24d360c82945a41363295e4c73de0430d8ac44d"),
    "SMOKE_JOB.json": (3639, "cf2e38e08860293b24751b432c14261248206dc39396692d007c1d04d3b668eb"),
    "SMOKE_RELEASE_AUTHORIZED": (73, "eb3fa4d05f68ee2854d37e72a3f0a88bd92191487982831e0773acb3f13f5ac9"),
    "SMOKE_SUBMISSION_ATTEMPT.tsv": (26, "e520e193599aff1eb6af8ea1b7a7ce1c6a80653b9f611694461d7f0c222850bb"),
    "SMOKE_SUBMISSION_LOCK/owner": (135, "1ab1e55757f768ddd44c5aba725c82a7a0f2578ced4ed1369ae09e20c81a28d7"),
    "SMOKE_SUBMITTED": (42, "5d85e6471c37abddc1e737c0007ca71d44e29be37b1e5d02832a4704ae5ebf00"),
    "STAGED": (219, "731f3652d80d46d2b8cb677b9d33ba115d90d40598acb6c894a59fee985d24e7"),
    "STOPPED_smoke": (131, "c551e7bf618f158bcf547b718ab7767d579c2cdb01ff0200a70169b3f08c81f6"),
}
SOURCE_GENERATION_FILES = {
    "smoke/.sampler.lock": (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "smoke/run_manifest.json": (832, "22e67bacbad9bc16b953d52d713b694194a352e6964540162fa37f5cd796b978"),
}
SOURCE_LOG_FILES = {
    "massive_medical_union_composition_exploratory_v1_smoke_261152.out": (161, "257c71c13acf4c2f565f0edd41d7b4e43bfca20a7626b7d00c2d6e404a633d3a"),
    "massive_medical_union_composition_exploratory_v1_smoke_261152.err": (3698, "c6c3f9ce24c24fbf3d9aee913bcb6417964bf758ff3e3074c49bbd57523ef2ce"),
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


def verify_seal(payload, description, field="payload_sha256"):
    if not isinstance(payload, dict) or not isinstance(payload.get(field), str):
        raise ValueError(f"{description} lacks {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if payload[field] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} seal differs")
    return body


def seal(body):
    payload = dict(body)
    payload["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return payload


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
    atomic_write_once(
        path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return payload


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", os.fspath(root), *args], text=True
    ).strip()


def audit_repository():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("recovery checkout is missing or unsafe")
    commit = git(REPO_ROOT, "rev-parse", "HEAD")
    parents = git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, SOURCE_COMMIT]:
        raise ValueError("recovery commit is not the direct nonmerge source child")
    if git(REPO_ROOT, "branch", "--show-current") != BRANCH:
        raise ValueError("recovery checkout branch differs")
    if git(REPO_ROOT, "status", "--porcelain"):
        raise ValueError("recovery checkout is dirty")
    observed = []
    for line in git(
        REPO_ROOT, "diff", "--name-status", "--no-renames", f"{SOURCE_COMMIT}..{commit}"
    ).splitlines():
        observed.append(tuple(line.split("\t")))
    expected = [("M", path) for path in MODIFIED_FILES]
    expected.extend(("A", path) for path in ADDED_FILES)
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("recovery commit differs from exact modified/add-only allowlist")
    files = {}
    for relative in (*MODIFIED_FILES, *ADDED_FILES):
        path = require_regular(REPO_ROOT / relative, f"recovery file {relative}")
        index = git(REPO_ROOT, "ls-files", "-s", "--", relative).split()
        expected_mode = "100755" if relative in EXECUTABLE_FILES else "100644"
        if len(index) < 4 or index[0] != expected_mode:
            raise ValueError(f"recovery index mode differs for {relative}")
        actual_mode = path.stat().st_mode & 0o777
        if actual_mode != (0o755 if relative in EXECUTABLE_FILES else 0o644):
            raise ValueError(f"recovery worktree mode differs for {relative}")
        files[relative] = {
            "git_mode": expected_mode,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "path": os.fspath(REPO_ROOT),
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "commit": commit,
        "direct_nonmerge_parent": True,
        "modified_files": list(MODIFIED_FILES),
        "added_files": list(ADDED_FILES),
        "files": files,
    }


def exact_file_inventory(root, expected, description):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{description} root is missing or unsafe")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{description} contains a symlink: {path}")
        if path.is_file():
            observed.add(os.fspath(path.relative_to(root)))
        elif not path.is_dir():
            raise ValueError(f"{description} contains an unsafe object: {path}")
    if observed != set(expected):
        raise ValueError(
            f"{description} inventory differs: observed={sorted(observed)!r}"
        )
    result = {}
    for relative, (size, digest) in expected.items():
        path = require_regular(root / relative, f"{description} {relative}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"{description} bytes differ for {relative}")
        result[relative] = binding(path)
    return result


def parse_tres(value):
    result = {}
    for term in (value or "").split(","):
        if "=" not in term:
            raise ValueError(f"invalid TRES term: {term}")
        key, item = term.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid/repeated TRES term: {term}")
        result[key] = item
    return result


def expected_scontrol_tres():
    return {
        "billing": "8",
        "cpu": "8",
        "gres/gpu:h200": "1",
        "gres/gpu": "1",
        "mem": "180G",
        "node": "1",
    }


def expected_sacct_tres():
    return {
        "billing": "8",
        "cpu": "8",
        "gres/gpu:h200": "1",
        "gres/gpu": "1",
        "mem": "180G",
        "node": "1",
    }


def source_sacct():
    output = subprocess.check_output(
        [
            "sacct", "-n", "-X", "-P", "-j", SOURCE_JOB_ID,
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode,NodeList",
        ],
        text=True,
    )
    matches = []
    for line in output.strip().splitlines():
        fields = line.split("|")
        if fields and fields[0] == SOURCE_JOB_ID:
            matches.append((line, fields))
    if len(matches) != 1 or len(matches[0][1]) < 9:
        raise ValueError("source incident lacks unique durable accounting")
    line, fields = matches[0]
    _, name, state, elapsed, limit, allocated, requested, exit_code, node = fields[:9]
    if (
        name != "mmu_cmpx_smoke_v1"
        or state != "FAILED"
        or elapsed != "00:00:35"
        or limit != "00:15:00"
        or exit_code != "1:0"
        or node != "g013"
        or parse_tres(allocated) != expected_sacct_tres()
        or parse_tres(requested) != expected_sacct_tres()
    ):
        raise ValueError("source incident durable accounting differs")
    return {
        "job_id": SOURCE_JOB_ID,
        "job_name": name,
        "state": state,
        "elapsed_seconds": PRIOR_ACTUAL_SECONDS,
        "actual_h200_minutes": PRIOR_ACTUAL_H200_MINUTES,
        "actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "exit_code": exit_code,
        "node": node,
        "sacct_row": line,
        "sacct_row_sha256": sha256_bytes(line.encode("utf-8")),
    }


def audit_source_protocol():
    manifest = load_json(SOURCE_PROTOCOL_MANIFEST, "source protocol manifest")
    body = verify_seal(
        manifest, "source protocol manifest", field="manifest_payload_sha256"
    )
    if (
        SOURCE_PROTOCOL_MANIFEST.stat().st_size != 63371
        or sha256_file(SOURCE_PROTOCOL_MANIFEST)
        != "20bda61a442c50b6a2990ddd99e5fc026c26a9625282c27c0a0feb4b29867446"
        or manifest.get("manifest_payload_sha256")
        != "20d96183145c96592ec5432b694d42333bc7d512ce68c2f5775b64d0cb345692"
        or body.get("protocol_id") != SOURCE_PROTOCOL_ID
    ):
        raise ValueError("source protocol manifest differs")
    completed = subprocess.run(
        [
            os.fspath(ENV_ROOT / "bin/python"),
            os.fspath(
                REPO_ROOT
                / "scripts/audit_massive_medical_union_composition_exploratory_v1.py"
            ),
            "audit-protocol", "--protocol-root", os.fspath(SOURCE_PROTOCOL_ROOT), "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    try:
        full = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("source protocol full audit emitted invalid JSON") from error
    if (
        full.get("status") != "AUDIT_OK"
        or full.get("manifest_file_sha256") != sha256_file(SOURCE_PROTOCOL_MANIFEST)
        or full.get("manifest_payload_sha256")
        != manifest["manifest_payload_sha256"]
    ):
        raise ValueError("source protocol full audit differs")
    return {
        **binding(
            SOURCE_PROTOCOL_MANIFEST, manifest, "manifest_payload_sha256"
        ),
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "full_audit": full,
    }


def audit_source_incident():
    if (
        SOURCE_REPO.is_symlink()
        or not SOURCE_REPO.is_dir()
        or git(SOURCE_REPO, "rev-parse", "HEAD") != SOURCE_COMMIT
        or git(SOURCE_REPO, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or git(SOURCE_REPO, "status", "--porcelain")
    ):
        raise ValueError("source incident checkout differs")
    control = exact_file_inventory(
        SOURCE_CONTROL_ROOT, SOURCE_CONTROL_FILES, "source control"
    )
    generation = exact_file_inventory(
        SOURCE_GENERATION_ROOT, SOURCE_GENERATION_FILES, "source generation"
    )
    if os.path.lexists(SOURCE_EVALUATION_ROOT):
        raise ValueError("source incident unexpectedly has an evaluation namespace")
    logs = {}
    for name, (size, digest) in SOURCE_LOG_FILES.items():
        path = require_regular(LOG_ROOT / name, f"source log {name}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"source log differs: {name}")
        logs[name] = binding(path)
    source_log_names = {
        path.name
        for path in LOG_ROOT.glob(
            "massive_medical_union_composition_exploratory_v1_*"
        )
        if path.is_file() and not path.is_symlink()
    }
    if source_log_names != set(SOURCE_LOG_FILES):
        raise ValueError("source incident log inventory differs")
    stop = (SOURCE_CONTROL_ROOT / "STOPPED_smoke").read_bytes()
    expected_stop = (
        b"workflow_id=massive_medical_union_composition_exploratory_workflow_v1\n"
        b"stage=smoke\njob_id=261152\nexit_code=1\nretry_authorized=false\n"
    )
    if stop != expected_stop:
        raise ValueError("source STOP record differs")
    run_payload = load_json(
        SOURCE_GENERATION_ROOT / "smoke/run_manifest.json",
        "source partial run manifest",
    )
    run_body = verify_seal(run_payload, "source partial run manifest")
    if (
        run_payload.get("payload_sha256")
        != "36d66f1a88b5e320ded8d4e80559b994eeacf436a72cada92ff1747b8f80cae4"
        or run_body.get("phase") != "smoke"
        or run_body.get("streams")
        != [
            {"method_id": "pi_base", "domain": "massive", "samples": 60},
            {"method_id": "ordinary_quorum_m4_q3", "domain": "massive", "samples": 60},
            {"method_id": "ordinary_min_m4_q4", "domain": "massive", "samples": 60},
            {"method_id": "delta_min_m4_q4", "domain": "massive", "samples": 60},
        ]
    ):
        raise ValueError("source partial run manifest differs")
    forbidden = (
        "CONFIRMATION_JOB.json",
        "CONFIRMATION_RESULT.json",
        "FINAL_RESULT.json",
        "EXTERNAL_JUDGE_AUTHORIZED_MAX_COST_USD_0.75.json",
        "FINALIZER_LOCK",
    )
    if any(os.path.lexists(SOURCE_CONTROL_ROOT / name) for name in forbidden):
        raise ValueError("source incident unexpectedly continued downstream")
    return {
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_job_accounting": source_sacct(),
        "source_control_inventory": control,
        "source_generation_inventory": generation,
        "source_evaluation_namespace_absent": True,
        "source_logs": logs,
        "source_stop": control["STOPPED_smoke"],
        "source_partial_run_manifest": binding(
            SOURCE_GENERATION_ROOT / "smoke/run_manifest.json", run_payload
        ),
        "source_scientific_generations_completed": 0,
        "source_confirmation_submitted": False,
        "source_external_api_calls": 0,
        "source_retry_authorized": False,
    }


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


def audit_offline_cache_environment():
    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": os.fspath(TILLICUM_ROOT / "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.fspath(
            TILLICUM_ROOT / "cache/huggingface/hub"
        ),
    }
    observed = {name: os.environ.get(name) for name in expected}
    if observed != expected:
        raise ValueError("recovery offline/cache environment differs")
    if "TRANSFORMERS_CACHE" in os.environ:
        raise ValueError("TRANSFORMERS_CACHE must be absent in recovery")
    return {
        **observed,
        "TRANSFORMERS_CACHE": None,
        "hub_name_resolution_used_for_base": False,
    }


def fingerprint_stable_file(path, allowed_root):
    lexical = os.path.abspath(path)
    resolved = os.path.realpath(lexical)
    root = os.path.realpath(allowed_root)
    if os.path.commonpath((root, resolved)) != root:
        raise ValueError(f"snapshot artifact escapes model cache: {lexical}")
    before = os.stat(resolved)
    if not os.path.isfile(resolved) or before.st_size <= 0:
        raise ValueError(f"snapshot artifact is not a nonempty regular file: {lexical}")
    digest = hashlib.sha256()
    count = 0
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
        after = os.fstat(handle.fileno())
    final_resolved = os.path.realpath(lexical)
    final = os.stat(final_resolved)

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        final_resolved != resolved
        or identity(before) != identity(after)
        or identity(after) != identity(final)
        or count != after.st_size
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
    artifacts = {}
    for name in SNAPSHOT_REQUIRED_FILES:
        observed = fingerprint_stable_file(os.path.join(resolved, name), cache_root)
        expected_size, expected_hash = EXPECTED_SNAPSHOT_ARTIFACTS[name]
        if (
            observed["size_bytes"] != expected_size
            or observed["sha256"] != expected_hash
        ):
            raise ValueError(f"pinned snapshot artifact differs: {name}")
        artifacts[name] = observed
    index_path = os.path.join(resolved, "model.safetensors.index.json")
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if fingerprint_stable_file(index_path, cache_root) != artifacts[
        "model.safetensors.index.json"
    ]:
        raise ValueError("model index changed while reading")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 339:
        raise ValueError("model index must contain exactly 339 tensor entries")
    shards = sorted(set(weight_map.values()))
    if shards != sorted(EXPECTED_SHARDS):
        raise ValueError("model index shard names differ")
    shard_artifacts = {}
    for name in shards:
        observed = fingerprint_stable_file(os.path.join(resolved, name), cache_root)
        expected_size, expected_hash = EXPECTED_SHARDS[name]
        if (
            observed["size_bytes"] != expected_size
            or observed["sha256"] != expected_hash
        ):
            raise ValueError(f"pinned snapshot shard differs: {name}")
        shard_artifacts[name] = observed
    metadata = index.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("total_size") != 15231233024
    ):
        raise ValueError("model index total_size differs")
    unindexed = sorted(
        name
        for name in os.listdir(resolved)
        if name.endswith(".safetensors") and name not in shards
    )
    if unindexed:
        raise ValueError(f"snapshot contains unindexed weight files: {unindexed}")
    tokenizer_config = json.loads(
        (Path(resolved) / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("pinned tokenizer lacks embedded chat template")
    body = {
        "canonical_model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "local_path": resolved,
        "direct_local_snapshot_required": True,
        "required_artifacts": artifacts,
        "weight_index_entries": 339,
        "weight_shards": shards,
        "weight_shard_artifacts": shard_artifacts,
        "indexed_weight_bytes": 15231233024,
        "chat_template_source": "tokenizer_config.json:chat_template",
        "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
    }
    body["snapshot_binding_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def audit_local_weight_resolver(snapshot=None):
    audit_offline_cache_environment()
    snapshot = snapshot or audit_local_model_snapshot()
    from transformers.modeling_utils import _get_resolved_checkpoint_files

    files, metadata = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=snapshot["local_path"],
        subfolder="",
        variant=None,
        gguf_file=None,
        from_tf=False,
        from_flax=False,
        use_safetensors=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        local_files_only=True,
        token=None,
        user_agent={
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": True,
        },
        revision=None,
        commit_hash=None,
        is_remote_code=False,
        transformers_explicit_filename=None,
    )
    if not isinstance(files, list) or len(files) != 4 or not isinstance(metadata, dict):
        raise ValueError("local Transformers resolver did not return four shards")
    resolved = []
    for name, path in zip(snapshot["weight_shards"], files):
        if Path(path).name != name:
            raise ValueError("local Transformers resolver shard order differs")
        artifact = snapshot["weight_shard_artifacts"][name]
        if (
            os.path.realpath(path) != artifact["resolved_path"]
            or os.stat(path).st_size != artifact["size_bytes"]
        ):
            raise ValueError("local Transformers resolver escaped the frozen shard")
        resolved.append(
            {
                "name": name,
                "lexical_path": os.path.abspath(path),
                **artifact,
            }
        )
    if (
        len(metadata.get("all_checkpoint_keys", [])) != 339
        or len(metadata.get("weight_map", {})) != 339
        or metadata.get("total_size") != 15231233024
    ):
        raise ValueError("local Transformers sharded metadata differs")
    body = {
        "protocol": (
            "massive_medical_union_composition_exploratory_"
            "local_snapshot_resolver_smoke_recovery_v1"
        ),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "load_source": snapshot["local_path"],
        "load_source_is_exact_audited_local_snapshot": True,
        "hub_model_id_used_for_loading": False,
        "transformers_cache_absent": True,
        "index": snapshot["required_artifacts"]["model.safetensors.index.json"],
        "index_entries": 339,
        "resolved_shards": resolved,
        "resolved_shard_count": 4,
        "indexed_weight_bytes": 15231233024,
        "snapshot_binding_sha256": snapshot["snapshot_binding_sha256"],
    }
    body["resolver_payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def expected_namespaces():
    return {
        "source_repository": os.fspath(SOURCE_REPO),
        "source_output": os.fspath(SOURCE_OUTPUT),
        "source_protocol": os.fspath(SOURCE_PROTOCOL_ROOT),
        "recovery_repository": os.fspath(REPO_ROOT),
        "recovery_output": os.fspath(OUTPUT_ROOT),
        "recovery_generation": os.fspath(GENERATION_ROOT),
        "recovery_evaluation": os.fspath(EVALUATION_ROOT),
        "recovery_control": os.fspath(CONTROL_ROOT),
        "recovery_log_prefix": os.fspath(LOG_ROOT / LOG_PREFIX),
    }


def budget_registry():
    return {
        "source_failed_actual_seconds": PRIOR_ACTUAL_SECONDS,
        "source_failed_actual_h200_minutes": PRIOR_ACTUAL_H200_MINUTES,
        "source_failed_actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "recovery_h200_minutes_max": RECOVERY_CAP_MINUTES,
        "recovery_gpu_cost_usd_max": RECOVERY_CAP_COST_USD,
        "source_actual_plus_recovery_cap_h200_minutes": (
            PRIOR_ACTUAL_H200_MINUTES + RECOVERY_CAP_MINUTES
        ),
        "source_actual_plus_recovery_cap_gpu_cost_usd": (
            ACTUAL_PLUS_RECOVERY_CAP_COST_USD
        ),
        "retry_reserve_h200_minutes": 0,
        "confirmation_h200_minutes_authorized": 0,
        "external_judge_cost_authorized_usd": 0,
    }


def prep_body(created_at=None):
    ready = require_regular(ENV_ROOT / ".ready", "environment readiness marker")
    snapshot = audit_local_model_snapshot()
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": audit_repository(),
        "source_protocol": audit_source_protocol(),
        "source_incident": audit_source_incident(),
        "environment": {
            "path": os.fspath(ENV_ROOT),
            "ready": binding(ready),
            "runtime_versions": audit_runtime_versions(),
            "offline_cache_environment": audit_offline_cache_environment(),
        },
        "local_model_snapshot": snapshot,
        "local_weight_resolver_preflight": audit_local_weight_resolver(snapshot),
        "namespaces": expected_namespaces(),
        "budget": budget_registry(),
        "stage_submitted_jobs": 0,
        "training": False,
        "external_api_calls": 0,
        "automatic_continuation": False,
        "recovery_confirmation_submission_code_present": False,
        "retry_or_reserve": False,
        "source_incident_immutable": True,
        "scientific_plan_changed": False,
        "confirmatory_claim": False,
        "wave3_v1_submitted_or_released": False,
    }


def command_write_prep(_args):
    if os.path.lexists(OUTPUT_ROOT):
        raise ValueError("recovery output namespace already exists")
    if any(LOG_ROOT.glob(f"{LOG_PREFIX}_*")):
        raise ValueError("recovery log namespace is not fresh")
    CONTROL_ROOT.mkdir(parents=True)
    payload = write_sealed_once(PREP_FILE, prep_body())
    print(payload["payload_sha256"])


def audit_prep(run_full=True):
    payload = load_json(PREP_FILE, "recovery PREP")
    body = verify_seal(payload, "recovery PREP")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("namespaces") != expected_namespaces()
        or body.get("budget") != budget_registry()
    ):
        raise ValueError("recovery PREP identity/budget differs")
    if run_full:
        snapshot = audit_local_model_snapshot()
        expected = {
            "repository": audit_repository(),
            "source_protocol": audit_source_protocol(),
            "source_incident": audit_source_incident(),
            "local_model_snapshot": snapshot,
            "local_weight_resolver_preflight": audit_local_weight_resolver(snapshot),
        }
        for key, value in expected.items():
            if body.get(key) != value:
                raise ValueError(f"recovery PREP differs on {key}")
        environment = body.get("environment")
        if (
            not isinstance(environment, dict)
            or environment.get("path") != os.fspath(ENV_ROOT)
            or environment.get("ready") != binding(ENV_ROOT / ".ready")
            or environment.get("runtime_versions") != audit_runtime_versions()
            or environment.get("offline_cache_environment")
            != audit_offline_cache_environment()
        ):
            raise ValueError("recovery PREP environment differs")
    immutable = {
        "stage_submitted_jobs": 0,
        "training": False,
        "external_api_calls": 0,
        "automatic_continuation": False,
        "recovery_confirmation_submission_code_present": False,
        "retry_or_reserve": False,
        "source_incident_immutable": True,
        "scientific_plan_changed": False,
        "confirmatory_claim": False,
        "wave3_v1_submitted_or_released": False,
    }
    if any(body.get(key) != value for key, value in immutable.items()):
        raise ValueError("recovery PREP safety flags differ")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_prep(_args):
    prep = audit_prep(run_full=True)
    print(
        json.dumps(
            {"status": "RECOVERY_PREP_OK", "payload_sha256": prep["payload_sha256"]},
            sort_keys=True,
        )
    )


def audit_sampler_snapshot_preflight(value, prep):
    if not isinstance(value, dict):
        raise ValueError("sampler preflight lacks base_model_snapshot")
    seal_value = value.get("snapshot_payload_sha256")
    if (
        not isinstance(seal_value, str)
        or seal_value
        != sha256_bytes(
            canonical_bytes(
                {key: item for key, item in value.items() if key != "snapshot_payload_sha256"}
            )
        )
    ):
        raise ValueError("sampler base-model snapshot seal differs")
    snapshot = prep.get("local_model_snapshot")
    if (
        set(value)
        != {
            "schema_version", "protocol", "model_id", "revision", "hub_cache",
            "snapshot_path", "runtime_artifacts", "safetensors_index", "safetensors_shards",
            "snapshot_payload_sha256",
        }
        or value.get("schema_version") != 1
        or value.get("protocol") != "qwen2_5_7b_instruct_local_snapshot_v1"
        or value.get("model_id") != BASE_MODEL
        or value.get("revision") != BASE_REVISION
        or os.path.realpath(str(value.get("hub_cache", "")))
        != os.path.realpath(TILLICUM_ROOT / "cache/huggingface/hub")
        or os.path.realpath(str(value.get("snapshot_path", "")))
        != snapshot.get("local_path")
    ):
        raise ValueError("sampler base-model snapshot identity differs")
    runtime_artifacts = value.get("runtime_artifacts")
    if not isinstance(runtime_artifacts, list) or len(runtime_artifacts) != len(
        SNAPSHOT_REQUIRED_FILES
    ) - 1:
        raise ValueError("sampler base-model runtime artifact registry differs")
    expected_runtime_names = [
        name for name in SNAPSHOT_REQUIRED_FILES if name != "model.safetensors.index.json"
    ]
    for name, row in zip(expected_runtime_names, runtime_artifacts):
        expected = snapshot["required_artifacts"][name]
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "size_bytes", "sha256"}
            or row.get("path") != name
            or row.get("size_bytes") != expected["size_bytes"]
            or row.get("sha256") != expected["sha256"]
        ):
            raise ValueError("sampler base-model runtime artifact binding differs")
    index = value.get("safetensors_index")
    if (
        not isinstance(index, dict)
        or set(index) != {"path", "size_bytes", "sha256"}
    ):
        raise ValueError("sampler base-model snapshot index differs")
    expected_index = snapshot["required_artifacts"]["model.safetensors.index.json"]
    if (
        index.get("size_bytes") != expected_index["size_bytes"]
        or index.get("sha256") != expected_index["sha256"]
        or index.get("path") != "model.safetensors.index.json"
    ):
        raise ValueError("sampler base-model index binding differs")
    shards = value.get("safetensors_shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("sampler base-model snapshot must bind four shards")
    expected_names = snapshot["weight_shards"]
    for name, row in zip(expected_names, shards):
        expected = snapshot["weight_shard_artifacts"][name]
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "size_bytes", "sha256"}
            or row.get("path") != name
            or row.get("size_bytes") != expected["size_bytes"]
            or row.get("sha256") != expected["sha256"]
        ):
            raise ValueError("sampler base-model shard binding differs")
    return value


def load_preflight():
    prep = audit_prep(run_full=False)
    payload = load_json(PREFLIGHT_FILE, "recovery CPU preflight")
    expected_scalar = {
        "status": "CPU_PREFLIGHT_OK",
        "intent_leaves_checked": 60,
        "slot_leaves_checked": 55,
        "invalid_probes_rejected": 9,
        "recorded_hybrid_intent_probes_rejected": 4,
        "recorded_hybrid_slot_probes_rejected": 3,
        "flexible_whitespace_probes_reproduced": 2,
        "whitespace_probes_rejected": 2,
    }
    if any(payload.get(key) != value for key, value in expected_scalar.items()):
        raise ValueError("recovery CPU preflight probe counts differ")
    if payload.get("runtime") != {
        key: EXPECTED_RUNTIME_VERSIONS[key]
        for key in ("torch", "transformers", "peft", "xgrammar")
    }:
        raise ValueError("recovery CPU preflight runtime differs")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get("schema_sha256", ""))) is None:
        raise ValueError("recovery CPU preflight schema hash differs")
    audit_sampler_snapshot_preflight(payload.get("base_model_snapshot"), prep)
    expected_keys = {
        *expected_scalar,
        "runtime",
        "schema_sha256",
        "base_model_snapshot",
    }
    if set(payload) != expected_keys:
        raise ValueError("recovery CPU preflight key set differs")
    return {**binding(PREFLIGHT_FILE), "payload": payload}


def command_audit_preflight(_args):
    preflight = load_preflight()
    print(
        json.dumps(
            {
                "status": "RECOVERY_CPU_PREFLIGHT_OK",
                "file_sha256": preflight["file_sha256"],
            },
            sort_keys=True,
        )
    )


def staged_bytes(prep):
    return (
        f"workflow_id={WORKFLOW_ID}\n"
        f"protocol_id={PROTOCOL_ID}\n"
        f"repo_commit={prep['repository']['commit']}\n"
        f"source_commit={SOURCE_COMMIT}\n"
        f"source_job_id={SOURCE_JOB_ID}\n"
        "source_stop_preserved=true\n"
        "stage_submitted_jobs=0\n"
        "recovery_gpu_h200_minutes_max=15\n"
        "recovery_gpu_cost_usd_max=0.225\n"
        "source_actual_plus_recovery_cap_gpu_cost_usd=0.23375\n"
        "training=false\n"
        "external_api_calls=0\n"
        "recovery_confirmation_submission_code_present=false\n"
    ).encode("utf-8")


def audit_staged():
    prep = audit_prep(run_full=True)
    staged = require_regular(STAGED_FILE, "recovery STAGED marker")
    if staged.read_bytes() != staged_bytes(prep):
        raise ValueError("recovery STAGED marker differs")
    preflight = load_preflight()
    return {
        "staged": binding(staged),
        "preflight": preflight,
        "prep_payload_sha256": prep["payload_sha256"],
    }


def command_write_staged(_args):
    prep = audit_prep(run_full=True)
    load_preflight()
    if os.path.lexists(GENERATION_ROOT) or os.path.lexists(EVALUATION_ROOT):
        raise ValueError("recovery scientific namespace exists during staging")
    if any(
        os.path.lexists(path)
        for path in (LOCK_ROOT, ATTEMPT_FILE, JOB_FILE, AUTH_FILE, SUBMITTED_FILE, RELEASE_FILE, RESULT_FILE, STOP_FILE)
    ):
        raise ValueError("recovery job/control state exists during staging")
    atomic_write_once(STAGED_FILE, staged_bytes(prep))
    print(sha256_file(STAGED_FILE))


def command_audit_staged(_args):
    result = audit_staged()
    print(
        json.dumps(
            {
                "status": "RECOVERY_STAGED_OK",
                "file_sha256": result["staged"]["file_sha256"],
            },
            sort_keys=True,
        )
    )


def command_assert_submit_ready(_args):
    audit_staged()
    expected = {"PREP.json", "SMOKE_RECOVERY_CPU_PREFLIGHT.json", "STAGED"}
    observed = set()
    for path in CONTROL_ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError("recovery pre-submit control contains a symlink")
        if path.is_file():
            observed.add(os.fspath(path.relative_to(CONTROL_ROOT)))
        elif not path.is_dir():
            raise ValueError("recovery pre-submit control contains an unsafe object")
    if observed != expected:
        raise ValueError("recovery pre-submit control inventory differs")
    if os.path.lexists(GENERATION_ROOT) or os.path.lexists(EVALUATION_ROOT):
        raise ValueError("recovery scientific namespace is not fresh pre-submit")
    if any(LOG_ROOT.glob(f"{LOG_PREFIX}_*")):
        raise ValueError("recovery log namespace is not fresh pre-submit")
    print(json.dumps({"status": "RECOVERY_SUBMIT_READY"}, sort_keys=True))


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


def query_job(job_id):
    raw = subprocess.check_output(
        ["scontrol", "show", "job", "-o", str(job_id)], text=True
    ).strip()
    return raw, parse_scontrol_line(raw)


def audit_job_record(job_id, raw, fields, phase, check_log_absence=True):
    exact = {
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
        "TimeLimit": "00:15:00",
        "Command": os.fspath(SBATCH_FILE),
        "WorkDir": os.fspath(REPO_ROOT),
        "StdOut": os.fspath(LOG_ROOT / f"{LOG_PREFIX}_{job_id}.out"),
        "StdErr": os.fspath(LOG_ROOT / f"{LOG_PREFIX}_{job_id}.err"),
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    for key, value in exact.items():
        if fields.get(key) != value:
            raise ValueError(f"recovery job differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError("recovery job node count differs")
    if parse_tres(fields.get("ReqTRES")) != expected_scontrol_tres():
        raise ValueError("recovery job requested TRES differs")
    if fields.get("Dependency") not in {None, "", "(null)"}:
        raise ValueError("recovery job unexpectedly has a dependency")
    if fields.get("KillOnInvalidDependent", "") not in {"", "No"}:
        raise ValueError("recovery invalid-dependency policy differs")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError("recovery unexpectedly belongs to array/het job")
    if phase == "held":
        expected_submit = (
            f"sbatch --parsable --hold --export=NONE --job-name={JOB_NAME} "
            + os.path.relpath(SBATCH_FILE, REPO_ROOT)
        )
        if (
            fields.get("JobState") != "PENDING"
            or fields.get("Reason") != "JobHeldUser"
            or fields.get("RunTime") != "00:00:00"
            or fields.get("AllocTRES") != "(null)"
            or fields.get("MinMemoryNode") != "180G"
            or fields.get("SubmitLine") != expected_submit
        ):
            raise ValueError("recovery job was not pristine and held-first")
        if check_log_absence and any(
            os.path.lexists(LOG_ROOT / f"{LOG_PREFIX}_{job_id}.{suffix}")
            for suffix in ("out", "err")
        ):
            raise ValueError("held recovery job already created a log")
    elif phase == "running":
        if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None":
            raise ValueError("recovery job is not RUNNING")
        if parse_tres(fields.get("AllocTRES")) != expected_scontrol_tres():
            raise ValueError("recovery job allocated TRES differs")
        node = fields.get("NodeList", "")
        if re.fullmatch(r"g[0-9]+", node) is None or fields.get("BatchHost") != node:
            raise ValueError("recovery node allocation differs")
    else:
        raise ValueError("unknown recovery scheduler phase")
    return {
        "job_id": str(job_id),
        "job_name": JOB_NAME,
        "scheduler_phase": phase,
        "scontrol_record": raw,
        "scontrol_record_sha256": sha256_bytes(raw.encode("utf-8")),
        "requested_tres": expected_scontrol_tres(),
        "time_limit": "00:15:00",
        "maximum_h200_minutes": RECOVERY_CAP_MINUTES,
        "maximum_gpu_cost_usd": RECOVERY_CAP_COST_USD,
        "held_first": True,
        "no_requeue": True,
        "dependencies": [],
    }


def audit_live_job(job_id, phase, check_log_absence=True):
    raw, fields = query_job(job_id)
    result = audit_job_record(job_id, raw, fields, phase, check_log_absence)
    if phase == "held":
        completed = subprocess.run(
            ["scontrol", "write", "batch_script", str(job_id), "-"],
            capture_output=True,
            check=True,
        )
        spooled = completed.stdout
        if isinstance(spooled, str):
            spooled = spooled.encode("utf-8")
        committed = require_regular(SBATCH_FILE, "committed recovery sbatch").read_bytes()
        if spooled != committed:
            raise ValueError("Slurm-spooled recovery script differs from commit")
        result["spooled_script_sha256"] = sha256_bytes(spooled)
        result["committed_script_sha256"] = sha256_bytes(committed)
    return result


def lock_binding():
    if LOCK_ROOT.is_symlink() or not LOCK_ROOT.is_dir():
        raise ValueError("recovery permanent submission lock is absent")
    owner = require_regular(LOCK_ROOT / "owner", "recovery lock owner")
    return {"path": os.fspath(LOCK_ROOT), "owner": binding(owner)}


def parse_attempt(job_id):
    path = require_regular(ATTEMPT_FILE, "recovery submission attempt")
    expected = f"stage\tjob_id\nsmoke_recovery\t{job_id}\n".encode("utf-8")
    if path.read_bytes() != expected:
        raise ValueError("recovery submission attempt differs")
    return binding(path)


def release_bytes(job_id):
    return (
        f"stage=smoke_recovery\njob_id={job_id}\nheld_audit_passed=true\n"
        "release_authorized=true\n"
    ).encode("utf-8")


def command_write_held_auth(args):
    job_id = str(args.job_id)
    prep = audit_prep(run_full=True)
    staged = audit_staged()
    if any(
        os.path.lexists(path)
        for path in (JOB_FILE, AUTH_FILE, SUBMITTED_FILE, RELEASE_FILE, RESULT_FILE, STOP_FILE)
    ):
        raise ValueError("recovery job/control output already exists")
    if os.path.lexists(GENERATION_ROOT) or os.path.lexists(EVALUATION_ROOT):
        raise ValueError("recovery scientific namespace is not fresh before release")
    held = audit_live_job(job_id, "held")
    snapshot = audit_local_model_snapshot()
    resolver = audit_local_weight_resolver(snapshot)
    runtime = audit_runtime_versions()
    cache_environment = audit_offline_cache_environment()
    source_incident = audit_source_incident()
    repository = audit_repository()
    source_protocol = audit_source_protocol()
    if (
        repository != prep.get("repository")
        or source_protocol != prep.get("source_protocol")
        or snapshot != prep.get("local_model_snapshot")
        or resolver != prep.get("local_weight_resolver_preflight")
        or runtime != prep.get("environment", {}).get("runtime_versions")
        or cache_environment
        != prep.get("environment", {}).get("offline_cache_environment")
        or source_incident != prep.get("source_incident")
    ):
        raise ValueError("pre-release recovery provenance differs from PREP")
    job_body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "smoke_recovery",
        "job_id": job_id,
        "held_audit": held,
        "submission_attempt": parse_attempt(job_id),
        "permanent_submission_lock": lock_binding(),
    }
    job_payload = write_sealed_once(JOB_FILE, job_body)
    auth_body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "smoke_recovery",
        "job_id": job_id,
        "job": binding(JOB_FILE, job_payload),
        "prep": binding(PREP_FILE, load_json(PREP_FILE, "recovery PREP")),
        "staged": staged["staged"],
        "cpu_preflight": staged["preflight"],
        "held_job_audit": held,
        "pre_release_repository": repository,
        "pre_release_source_protocol": source_protocol,
        "pre_release_source_incident": source_incident,
        "pre_release_local_model_snapshot": snapshot,
        "pre_release_local_weight_resolver": resolver,
        "pre_release_runtime_versions": runtime,
        "pre_release_offline_cache_environment": cache_environment,
        "source_failed_actual_seconds": PRIOR_ACTUAL_SECONDS,
        "source_failed_actual_h200_minutes": PRIOR_ACTUAL_H200_MINUTES,
        "source_failed_actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "maximum_recovery_h200_minutes": RECOVERY_CAP_MINUTES,
        "maximum_recovery_gpu_cost_usd": RECOVERY_CAP_COST_USD,
        "source_actual_plus_recovery_cap_gpu_cost_usd": ACTUAL_PLUS_RECOVERY_CAP_COST_USD,
        "new_versioned_recovery_not_retry": True,
        "retry_or_reserve": False,
        "training": False,
        "external_api_calls": 0,
        "recovery_confirmation_submission_code_present": False,
        "automatic_continuation": False,
        "scientific_plan_changed": False,
        "confirmatory_claim": False,
        "wave3_v1_submitted_or_released": False,
    }
    auth_payload = write_sealed_once(AUTH_FILE, auth_body)
    print(auth_payload["payload_sha256"])


def job_pointer():
    payload = load_json(JOB_FILE, "recovery job pointer")
    body = verify_seal(payload, "recovery job pointer")
    expected_keys = {
        "schema_version", "workflow_id", "created_at", "stage", "job_id",
        "held_audit", "submission_attempt", "permanent_submission_lock",
    }
    held = body.get("held_audit")
    if not isinstance(held, dict):
        raise ValueError("recovery job pointer lacks held audit")
    reconstructed = audit_job_record(
        body.get("job_id"),
        held.get("scontrol_record"),
        parse_scontrol_line(held.get("scontrol_record")),
        "held",
        check_log_absence=False,
    )
    reconstructed["spooled_script_sha256"] = held.get("spooled_script_sha256")
    reconstructed["committed_script_sha256"] = held.get(
        "committed_script_sha256"
    )
    committed_hash = sha256_file(require_regular(SBATCH_FILE, "recovery sbatch"))
    if (
        set(body) != expected_keys
        or body.get("schema_version") != SCHEMA_VERSION
        or body.get("workflow_id") != WORKFLOW_ID
        or body.get("stage") != "smoke_recovery"
        or re.fullmatch(r"[0-9]+", str(body.get("job_id", ""))) is None
        or held != reconstructed
        or body.get("submission_attempt") != parse_attempt(body["job_id"])
        or body.get("permanent_submission_lock") != lock_binding()
        or held.get("spooled_script_sha256") != committed_hash
        or held.get("committed_script_sha256") != committed_hash
    ):
        raise ValueError("recovery job pointer differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def auth_pointer():
    payload = load_json(AUTH_FILE, "recovery authorization")
    body = verify_seal(payload, "recovery authorization")
    job = job_pointer()
    prep = audit_prep(run_full=False)
    expected_keys = {
        "schema_version", "workflow_id", "protocol_id", "created_at", "stage",
        "job_id", "job", "prep", "staged", "cpu_preflight", "held_job_audit",
        "pre_release_repository", "pre_release_source_protocol",
        "pre_release_source_incident", "pre_release_local_model_snapshot",
        "pre_release_local_weight_resolver", "pre_release_runtime_versions",
        "pre_release_offline_cache_environment", "source_failed_actual_seconds",
        "source_failed_actual_h200_minutes", "source_failed_actual_gpu_cost_usd",
        "maximum_recovery_h200_minutes", "maximum_recovery_gpu_cost_usd",
        "source_actual_plus_recovery_cap_gpu_cost_usd",
        "new_versioned_recovery_not_retry", "retry_or_reserve", "training",
        "external_api_calls", "recovery_confirmation_submission_code_present",
        "automatic_continuation", "scientific_plan_changed", "confirmatory_claim",
        "wave3_v1_submitted_or_released",
    }
    held = body.get("held_job_audit")
    if not isinstance(held, dict):
        raise ValueError("recovery authorization lacks held audit")
    reconstructed = audit_job_record(
        job["job_id"], held.get("scontrol_record"),
        parse_scontrol_line(held.get("scontrol_record")), "held", False,
    )
    reconstructed["spooled_script_sha256"] = held.get("spooled_script_sha256")
    reconstructed["committed_script_sha256"] = held.get("committed_script_sha256")
    expected_values = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "stage": "smoke_recovery",
        "job_id": job["job_id"],
        "job": binding(JOB_FILE, load_json(JOB_FILE, "recovery job pointer")),
        "prep": binding(PREP_FILE, load_json(PREP_FILE, "recovery PREP")),
        "staged": binding(STAGED_FILE),
        "cpu_preflight": load_preflight(),
        "held_job_audit": reconstructed,
        "source_failed_actual_seconds": PRIOR_ACTUAL_SECONDS,
        "source_failed_actual_h200_minutes": PRIOR_ACTUAL_H200_MINUTES,
        "source_failed_actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "maximum_recovery_h200_minutes": RECOVERY_CAP_MINUTES,
        "maximum_recovery_gpu_cost_usd": RECOVERY_CAP_COST_USD,
        "source_actual_plus_recovery_cap_gpu_cost_usd": ACTUAL_PLUS_RECOVERY_CAP_COST_USD,
        "new_versioned_recovery_not_retry": True,
        "retry_or_reserve": False,
        "training": False,
        "external_api_calls": 0,
        "recovery_confirmation_submission_code_present": False,
        "automatic_continuation": False,
        "scientific_plan_changed": False,
        "confirmatory_claim": False,
        "wave3_v1_submitted_or_released": False,
    }
    if set(body) != expected_keys or any(
        body.get(key) != value for key, value in expected_values.items()
    ):
        raise ValueError("recovery authorization differs")
    if (
        body.get("pre_release_repository") != prep.get("repository")
        or body.get("pre_release_source_protocol") != prep.get("source_protocol")
        or body.get("pre_release_source_incident") != prep.get("source_incident")
        or body.get("pre_release_local_model_snapshot")
        != prep.get("local_model_snapshot")
        or body.get("pre_release_local_weight_resolver")
        != prep.get("local_weight_resolver_preflight")
        or body.get("pre_release_runtime_versions")
        != prep.get("environment", {}).get("runtime_versions")
        or body.get("pre_release_offline_cache_environment")
        != prep.get("environment", {}).get("offline_cache_environment")
    ):
        raise ValueError("recovery pre-release provenance differs from PREP")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def audit_current_authorized_provenance(auth):
    if audit_repository() != auth["pre_release_repository"]:
        raise ValueError("recovery repository drifted after authorization")
    if audit_source_protocol() != auth["pre_release_source_protocol"]:
        raise ValueError("source protocol drifted after authorization")
    if audit_source_incident() != auth["pre_release_source_incident"]:
        raise ValueError("immutable source incident drifted after authorization")
    snapshot = audit_local_model_snapshot()
    if snapshot != auth["pre_release_local_model_snapshot"]:
        raise ValueError("recovery local model snapshot drifted after authorization")
    if audit_local_weight_resolver(snapshot) != auth["pre_release_local_weight_resolver"]:
        raise ValueError("recovery local weight resolver drifted after authorization")
    if audit_runtime_versions() != auth["pre_release_runtime_versions"]:
        raise ValueError("recovery runtime drifted after authorization")
    if audit_offline_cache_environment() != auth[
        "pre_release_offline_cache_environment"
    ]:
        raise ValueError("recovery cache environment drifted after authorization")


def recovery_control_inventory(phase):
    expected = {
        "PREP.json",
        "SMOKE_RECOVERY_CPU_PREFLIGHT.json",
        "STAGED",
        "SMOKE_RECOVERY_SUBMISSION_LOCK/owner",
        "SMOKE_RECOVERY_SUBMISSION_ATTEMPT.tsv",
        "SMOKE_RECOVERY_JOB.json",
        "SMOKE_RECOVERY_AUTHORIZED_MAX_COST_USD_0.225.json",
        "SMOKE_RECOVERY_SUBMITTED",
    }
    if phase == "running":
        expected.add("SMOKE_RECOVERY_RELEASE_AUTHORIZED")
    elif phase != "held":
        raise ValueError("unknown recovery control phase")
    observed = set()
    for path in CONTROL_ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError("recovery control contains a symlink")
        if path.is_file():
            observed.add(os.fspath(path.relative_to(CONTROL_ROOT)))
        elif not path.is_dir():
            raise ValueError("recovery control contains an unsafe object")
    if observed != expected:
        raise ValueError(
            f"recovery {phase} control inventory differs: {sorted(observed)!r}"
        )
    return sorted(observed)


def audit_scientific_namespace_fresh(phase, job_id):
    if os.path.lexists(GENERATION_ROOT) or os.path.lexists(EVALUATION_ROOT):
        raise ValueError(f"recovery scientific namespace exists during {phase}")
    observed_logs = {
        path.name
        for path in LOG_ROOT.glob(f"{LOG_PREFIX}_*")
        if path.is_file() and not path.is_symlink()
    }
    expected_logs = (
        set()
        if phase == "held"
        else {f"{LOG_PREFIX}_{job_id}.out", f"{LOG_PREFIX}_{job_id}.err"}
    )
    if observed_logs != expected_logs:
        raise ValueError(f"recovery {phase} log inventory differs")


def command_audit_held(args):
    auth = auth_pointer()
    if auth["job_id"] != str(args.job_id):
        raise ValueError("recovery held audit job differs")
    live = audit_live_job(args.job_id, "held")
    if live != auth["held_job_audit"]:
        raise ValueError("live held recovery job differs from authorization")
    audit_current_authorized_provenance(auth)
    recovery_control_inventory("held")
    audit_scientific_namespace_fresh("held", args.job_id)
    print(json.dumps({"status": "RECOVERY_HELD_AUDIT_OK", "job_id": str(args.job_id)}, sort_keys=True))


def audit_slurm_environment(job_id, live):
    node = parse_scontrol_line(live["scontrol_record"])["NodeList"]
    expected = {
        "SLURM_JOB_ID": str(job_id),
        "SLURM_JOB_NAME": JOB_NAME,
        "SLURM_JOB_PARTITION": "gpu-h200",
        "SLURM_JOB_ACCOUNT": "stf",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": os.fspath(REPO_ROOT),
        "SLURM_JOB_NODELIST": node,
        "SLURM_MEM_PER_NODE": "184320",
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError(f"recovery executing environment differs on {key}")
    return expected


def command_verify_job(args):
    auth = auth_pointer()
    if auth["job_id"] != str(args.job_id):
        raise ValueError("running recovery job differs from authorization")
    release = require_regular(RELEASE_FILE, "recovery release authorization")
    if release.read_bytes() != release_bytes(args.job_id):
        raise ValueError("recovery release authorization differs")
    live = audit_live_job(args.job_id, "running", check_log_absence=False)
    audit_slurm_environment(args.job_id, live)
    audit_current_authorized_provenance(auth)
    recovery_control_inventory("running")
    audit_scientific_namespace_fresh("running", args.job_id)
    if "TRANSFORMERS_CACHE" in os.environ:
        raise ValueError("TRANSFORMERS_CACHE reached recovery GPU job")
    print("SMOKE_RECOVERY_RUNNING_AUDIT_OK")


def load_run_manifest():
    path = GENERATION_ROOT / "smoke/run_manifest.json"
    payload = load_json(path, "recovery generation run manifest")
    body = verify_seal(payload, "recovery generation run manifest")
    if (
        body.get("schema_version") != 1
        or body.get("protocol")
        != "massive_medical_union_composition_exploratory_generation_v1"
        or body.get("phase") != "smoke"
        or body.get("protocol_manifest_file_sha256")
        != "20bda61a442c50b6a2990ddd99e5fc026c26a9625282c27c0a0feb4b29867446"
        or body.get("protocol_manifest_payload_sha256")
        != "20d96183145c96592ec5432b694d42333bc7d512ce68c2f5775b64d0cb345692"
        or body.get("streams")
        != [
            {"method_id": "pi_base", "domain": "massive", "samples": 60},
            {"method_id": "ordinary_quorum_m4_q3", "domain": "massive", "samples": 60},
            {"method_id": "ordinary_min_m4_q4", "domain": "massive", "samples": 60},
            {"method_id": "delta_min_m4_q4", "domain": "massive", "samples": 60},
        ]
    ):
        raise ValueError("recovery generation run manifest differs")
    return binding(path, payload)


def load_gate():
    allowed = ("EXPLORATORY_SMOKE_PASSED", "STOPPED_EXPLORATORY_SMOKE")
    present = [name for name in allowed if os.path.lexists(GATE_ROOT / name)]
    if len(present) != 1:
        raise ValueError("recovery smoke gate lacks exactly one terminal sentinel")
    status = present[0]
    sentinel_path = GATE_ROOT / status
    sentinel_payload = load_json(sentinel_path, "recovery smoke sentinel")
    sentinel = verify_seal(sentinel_payload, "recovery smoke sentinel")
    summary_path = GATE_ROOT / "summary.json"
    summary_payload = load_json(summary_path, "recovery smoke summary")
    summary = verify_seal(summary_payload, "recovery smoke summary")
    if (
        sentinel.get("status") != status
        or Path(sentinel.get("summary_path", "")).resolve() != summary_path.resolve()
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
        or summary.get("status") != status
    ):
        raise ValueError("recovery smoke gate summary/sentinel differs")
    for source in (summary, sentinel):
        if (
            source.get("confirmatory_claim") is not False
            or source.get("wave2_v1_status") != "STOP"
            or source.get("wave3_v1_eligible") is not False
            or source.get("wave3_v1_submitted_or_released") is not False
        ):
            raise ValueError("recovery smoke gate safety flags differ")
    return {
        "status": status,
        "summary": binding(summary_path, summary_payload),
        "sentinel": binding(sentinel_path, sentinel_payload),
    }


def result_body(job_id, created_at=None):
    auth = auth_pointer()
    if auth["job_id"] != str(job_id):
        raise ValueError("recovery result job differs from authorization")
    live = audit_live_job(job_id, "running", check_log_absence=False)
    gate = load_gate()
    job_binding = binding(JOB_FILE, load_json(JOB_FILE, "recovery job pointer"))
    authorization_binding = binding(
        AUTH_FILE, load_json(AUTH_FILE, "recovery authorization")
    )
    cpu_preflight = load_preflight()
    generation_run_manifest = load_run_manifest()
    # This is deliberately the final live provenance operation before the
    # write-once result body is returned to the sealing caller.  It catches
    # drift between the in-job start audit and the separate sampler/scorer/
    # summarizer processes.
    audit_current_authorized_provenance(auth)
    source = auth["pre_release_source_incident"]
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "smoke_recovery",
        "job_id": str(job_id),
        "job": job_binding,
        "authorization": authorization_binding,
        "running_job_audit": live,
        "cpu_preflight": cpu_preflight,
        "generation_run_manifest": generation_run_manifest,
        "gate": gate,
        "scientific_status": gate["status"],
        "source_incident_after_recovery": source,
        "source_incident_unchanged": True,
        "source_failed_actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "recovery_released_gpu_cost_usd_cap": RECOVERY_CAP_COST_USD,
        "source_actual_plus_recovery_cap_gpu_cost_usd": ACTUAL_PLUS_RECOVERY_CAP_COST_USD,
        "new_versioned_recovery_not_retry": True,
        "future_confirmation_requires_new_authorization": True,
        "confirmation_submission_eligible_in_this_workflow": False,
        "confirmation_submitted": False,
        "training": False,
        "external_api_calls": 0,
        "automatic_continuation": False,
        "no_retry_or_reserve": True,
        "scientific_plan_changed": False,
        "confirmatory_claim": False,
        "wave3_v1_submitted_or_released": False,
    }


def command_write_result(args):
    payload = write_sealed_once(RESULT_FILE, result_body(args.job_id))
    print(payload["payload_sha256"])


def audit_result():
    payload = load_json(RESULT_FILE, "recovery smoke result")
    body = verify_seal(payload, "recovery smoke result")
    auth = auth_pointer()
    gate = load_gate()
    expected_keys = {
        "schema_version", "workflow_id", "protocol_id", "created_at", "stage",
        "job_id", "job", "authorization", "running_job_audit", "cpu_preflight",
        "generation_run_manifest", "gate", "scientific_status",
        "source_incident_after_recovery", "source_incident_unchanged",
        "source_failed_actual_gpu_cost_usd", "recovery_released_gpu_cost_usd_cap",
        "source_actual_plus_recovery_cap_gpu_cost_usd",
        "new_versioned_recovery_not_retry",
        "future_confirmation_requires_new_authorization",
        "confirmation_submission_eligible_in_this_workflow",
        "confirmation_submitted", "training", "external_api_calls",
        "automatic_continuation", "no_retry_or_reserve", "scientific_plan_changed",
        "confirmatory_claim", "wave3_v1_submitted_or_released",
    }
    running = body.get("running_job_audit")
    if not isinstance(running, dict):
        raise ValueError("recovery result lacks running-job audit")
    reconstructed = audit_job_record(
        auth["job_id"], running.get("scontrol_record"),
        parse_scontrol_line(running.get("scontrol_record")), "running", False,
    )
    expected_values = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "stage": "smoke_recovery",
        "job_id": auth["job_id"],
        "job": binding(JOB_FILE, load_json(JOB_FILE, "recovery job pointer")),
        "authorization": binding(AUTH_FILE, load_json(AUTH_FILE, "recovery auth")),
        "running_job_audit": reconstructed,
        "cpu_preflight": load_preflight(),
        "generation_run_manifest": load_run_manifest(),
        "gate": gate,
        "scientific_status": gate["status"],
        "source_incident_after_recovery": auth["pre_release_source_incident"],
        "source_incident_unchanged": True,
        "source_failed_actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "recovery_released_gpu_cost_usd_cap": RECOVERY_CAP_COST_USD,
        "source_actual_plus_recovery_cap_gpu_cost_usd": ACTUAL_PLUS_RECOVERY_CAP_COST_USD,
        "new_versioned_recovery_not_retry": True,
        "future_confirmation_requires_new_authorization": True,
        "confirmation_submission_eligible_in_this_workflow": False,
        "confirmation_submitted": False,
        "training": False,
        "external_api_calls": 0,
        "automatic_continuation": False,
        "no_retry_or_reserve": True,
        "scientific_plan_changed": False,
        "confirmatory_claim": False,
        "wave3_v1_submitted_or_released": False,
    }
    # Terminal/status audits independently repeat the full live provenance
    # audit; a sealed result is not sufficient if its code, protocol, source
    # incident, model snapshot, resolver, runtime, or cache environment drifted.
    audit_current_authorized_provenance(auth)
    if set(body) != expected_keys or any(
        body.get(key) != value for key, value in expected_values.items()
    ):
        raise ValueError("recovery smoke result differs")
    if body["source_incident_after_recovery"] != auth["pre_release_source_incident"]:
        raise ValueError("source incident changed between authorization and result")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def parse_duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value or "")
    if match is None:
        raise ValueError(f"invalid Slurm duration: {value}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid Slurm duration: {value}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def terminal_accounting():
    job = job_pointer()
    output = subprocess.check_output(
        [
            "sacct", "-n", "-X", "-P", "-j", job["job_id"],
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode",
        ],
        text=True,
    )
    matches = []
    for line in output.strip().splitlines():
        fields = line.split("|")
        if fields and fields[0] == job["job_id"]:
            matches.append((line, fields))
    if len(matches) != 1 or len(matches[0][1]) < 8:
        raise ValueError("recovery job lacks unique durable accounting")
    line, fields = matches[0]
    _, name, state, elapsed, limit, allocated, requested, exit_code = fields[:8]
    if (
        name != JOB_NAME
        or state != "COMPLETED"
        or limit != "00:15:00"
        or exit_code != "0:0"
        or parse_tres(allocated) != expected_sacct_tres()
        or parse_tres(requested) != expected_sacct_tres()
    ):
        raise ValueError("recovery durable accounting differs")
    seconds = parse_duration(elapsed)
    if not 0 < seconds <= RECOVERY_CAP_MINUTES * 60:
        raise ValueError("recovery elapsed time exceeds cap")
    minutes = seconds / 60.0
    actual_cost = minutes * RATE_PER_H200_MINUTE_USD
    return {
        "job_id": job["job_id"],
        "job_name": JOB_NAME,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": seconds,
        "actual_h200_minutes": minutes,
        "actual_gpu_cost_usd": actual_cost,
        "source_failed_actual_gpu_cost_usd": PRIOR_ACTUAL_COST_USD,
        "source_actual_plus_recovery_actual_gpu_cost_usd": (
            PRIOR_ACTUAL_COST_USD + actual_cost
        ),
        "released_recovery_h200_minutes_cap": RECOVERY_CAP_MINUTES,
        "released_recovery_gpu_cost_usd_cap": RECOVERY_CAP_COST_USD,
        "source_actual_plus_recovery_cap_gpu_cost_usd": ACTUAL_PLUS_RECOVERY_CAP_COST_USD,
        "sacct_row": line,
        "sacct_row_sha256": sha256_bytes(line.encode("utf-8")),
    }


def command_audit_result(_args):
    result = audit_result()
    print(
        json.dumps(
            {
                "status": "RECOVERY_RESULT_OK",
                "scientific_status": result["scientific_status"],
            },
            sort_keys=True,
        )
    )


def command_audit_terminal(_args):
    audit_result()
    print(json.dumps(terminal_accounting(), sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-prep").set_defaults(function=command_write_prep)
    commands.add_parser("audit-prep").set_defaults(function=command_audit_prep)
    commands.add_parser("audit-preflight").set_defaults(function=command_audit_preflight)
    commands.add_parser("audit-staged").set_defaults(function=command_audit_staged)
    commands.add_parser("write-staged").set_defaults(function=command_write_staged)
    commands.add_parser("assert-submit-ready").set_defaults(
        function=command_assert_submit_ready
    )
    for name, function in (
        ("write-held-auth", command_write_held_auth),
        ("audit-held", command_audit_held),
        ("verify-job", command_verify_job),
        ("write-result", command_write_result),
    ):
        item = commands.add_parser(name)
        item.add_argument("--job-id", required=True)
        item.set_defaults(function=function)
    commands.add_parser("audit-result").set_defaults(function=command_audit_result)
    commands.add_parser("audit-terminal").set_defaults(function=command_audit_terminal)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
