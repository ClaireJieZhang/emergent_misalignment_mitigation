#!/usr/bin/env python3
"""Fail-closed CPU control plane for sequential exploratory confirmation v1."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


WORKFLOW_ID = "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
PROTOCOL_ID = WORKFLOW_ID
DIRECT_PARENT_COMMIT = "890f685b3198e30e1658aa7ab0aa9f11a537aaf9"
DIRECT_PARENT_TREE = "676b2e9c31a6df22750a7cc51d88f099985f4068"
BRANCH = "claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1"
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
REPO_ROOT = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-confirmation-v1"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1"
CONTROL_ROOT = OUTPUT_ROOT / "control"
PROTOCOL_ROOT = OUTPUT_ROOT / "protocol"
PREP_FILE = CONTROL_ROOT / "PREP.json"
BENEFIT_PREFLIGHT_FILE = CONTROL_ROOT / "SAMPLER_PREFLIGHT_BENEFIT.json"
MEDICAL_PREFLIGHT_FILE = CONTROL_ROOT / "SAMPLER_PREFLIGHT_MEDICAL.json"
STAGED_FILE = CONTROL_ROOT / "STAGED.json"
GENERATION_ROOT = OUTPUT_ROOT / "generation"
EVALUATION_ROOT = OUTPUT_ROOT / "evaluation"
FINALIZER_LOCK = CONTROL_ROOT / "FINALIZER_LOCK"
FINAL_AUTH = CONTROL_ROOT / "EXTERNAL_JUDGE_AUTHORIZATION.json"
FINAL_RESULT = CONTROL_ROOT / "FINAL_RESULT.json"
JUDGE_PLAN = EVALUATION_ROOT / "medical/judge_plan.json"
JUDGE_CHECKPOINT = EVALUATION_ROOT / "medical/judge_checkpoint.json"
JUDGMENTS_NEW = EVALUATION_ROOT / "medical/judgments_new.json"
JUDGMENTS_MERGED = EVALUATION_ROOT / "medical/judgments_merged.json"
LOG_ROOT = TILLICUM_ROOT / "outputs/logs"
LOG_PREFIX = "massive_medical_union_composition_exploratory_sequential_confirmation_v1_"
V5_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-composition-exploratory-smoke-gate-recovery-v5"
V5_AUDITOR = V5_REPO / "scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v5.py"
V5_RESULT = TILLICUM_ROOT / "outputs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v5/control/SMOKE_GATE_RECOVERY_RESULT.json"
V5_AUDITOR_SHA256 = "5aef6a23937d0ee2e1f4a9f1d85ed1a248178f51ffba409ef3f793758e93b558"
V5_RESULT_SHA256 = "f3448b7bd6ef2cd76b65fa5b5ac87a3dea065ffd0452ae00fc5afdc92e75992c"
V5_RESULT_PAYLOAD_SHA256 = "05ff30605ce145bc4a2af95aad5dc2c64252008220fa6f3bf645e2dcc84a47e2"
MANIFEST_SHA256 = "20bda61a442c50b6a2990ddd99e5fc026c26a9625282c27c0a0feb4b29867446"
MANIFEST_PAYLOAD_SHA256 = "20d96183145c96592ec5432b694d42333bc7d512ce68c2f5775b64d0cb345692"
SELECTED_IDS_SHA256 = "ac5dec7a70ff616a73bd1a00ed7c7e03f506afb03f6232b83299f2b1474880e6"
RANKED_IDS_SHA256 = "c5c3a6a2cc09aa9103dc593c7a14fa1853429a74b89b41deeb39481f52c903eb"
RANK_RECORDS_SHA256 = "10cc94525d5953b8bdabbb5f55f2720fdf2d90e9c78c28983594ea509e498bea"
MEDICAL_PROMPTS_BINDING = (7035, "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57")
HISTORICAL_A_BINDING = (232178, "359a8e2351c855bceaea8400cb97a32f62a82f64f7b13b09839a120746a94ca2")
METHODS = ("ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4")
MODELS = ("pi_base", "pi_A", "pi_B1", "pi_B2", "pi_B3")
FORBIDDEN_ENV = (
    "OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST", "SLURM_NODELIST",
)
FORBIDDEN_MODULES = ("torch", "transformers", "peft", "xgrammar", "openai")
BASE_RUNTIME_ARTIFACTS = (
    ("config.json", 663, "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"),
    ("generation_config.json", 243, "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f"),
    ("tokenizer_config.json", 7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    ("tokenizer.json", 7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    ("vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    ("merges.txt", 1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
)
BASE_SAFETENSORS_INDEX = (
    "model.safetensors.index.json", 27752,
    "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
)
BASE_SAFETENSORS_SHARDS = (
    ("model-00001-of-00004.safetensors", 3945441440, "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7"),
    ("model-00002-of-00004.safetensors", 3864726352, "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185"),
    ("model-00003-of-00004.safetensors", 3864726424, "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5"),
    ("model-00004-of-00004.safetensors", 3556377672, "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd"),
)

ADDED_FILES = (
    "docs/massive_medical_union_composition_exploratory_sequential_confirmation_v1.md",
    "scripts/prepare_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh",
    "scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum_h200.sbatch",
    "scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum.sh",
    "scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum_h200.sbatch",
    "scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum.sh",
    "scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh",
    "scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "scripts/merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_confirmation_v1_protocol.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_confirmation_v1_workflow.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_sampler.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_evaluation.py",
)

FROZEN_EXTERNAL_FILES = {
    "scripts/sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py": "1508ff6608950d15704dd628b1410419810c6ae75d0db6d7b472a3165475e3e2",
    "tests/test_massive_medical_union_composition_exploratory_sequential_sampler.py": "b3e58308c04544f30e8a3be705a8c41d231996d3a6ae51bc3c82acb7278c366f",
    "scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py": "229cba59bc1b0d0d8da47e02e933f0410268b359339ef7fba6f5f05df05bca3d",
    "scripts/judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py": "ef2a72ada85ab2d4d73c86e71bb21d1095a8e5d634351bec3b2324d4c366c2ed",
    "scripts/merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py": "eb297bd16fc74fb45058d532e62f42c8233f1e523a58728c0164f31d8cdd1dc8",
    "tests/test_massive_medical_union_composition_exploratory_sequential_evaluation.py": "836dd2c5a2d9919601f3d7e0b608b930d6cc841e383406d94e278e8c661e02d5",
}


def stage_config(stage):
    if stage not in {"benefit", "medical"}:
        raise ValueError("unknown sequential GPU stage")
    upper = stage.upper()
    minutes, cost, limit = (65, 0.975, "01:05:00") if stage == "benefit" else (95, 1.425, "01:35:00")
    return {
        "stage": stage,
        "upper": upper,
        "minutes": minutes,
        "cost": cost,
        "time_limit": limit,
        "job_name": "mmu_seq_benefit_v1" if stage == "benefit" else "mmu_seq_medical_v1",
        "log_prefix": LOG_PREFIX + stage,
        "sbatch": REPO_ROOT / f"scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_{stage}_tillicum_h200.sbatch",
        "lock": CONTROL_ROOT / f"{upper}_SUBMISSION_LOCK",
        "attempt": CONTROL_ROOT / f"{upper}_SUBMISSION_ATTEMPT.tsv",
        "submitted": CONTROL_ROOT / f"{upper}_SUBMITTED",
        "released": CONTROL_ROOT / f"{upper}_RELEASE_AUTHORIZED",
        "job": CONTROL_ROOT / f"{upper}_JOB.json",
        "auth": CONTROL_ROOT / f"{upper}_AUTHORIZATION.json",
        "result": CONTROL_ROOT / f"{upper}_RESULT.json",
        "stop": CONTROL_ROOT / f"STOPPED_{stage}",
        "preflight": BENEFIT_PREFLIGHT_FILE if stage == "benefit" else MEDICAL_PREFLIGHT_FILE,
    }


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular(path, description, mode=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing or unsafe: {path}")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise ValueError(f"{description} mode differs")
    return path


def load_json(path, description):
    path = require_regular(path, description)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error


def verify_seal(payload, description, field="payload_sha256"):
    if not isinstance(payload, dict) or not isinstance(payload.get(field), str):
        raise ValueError(f"{description} lacks {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if payload[field] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} seal differs")
    return body


def binding(path, payload=None, field=None):
    path = require_regular(path, "bound file")
    result = {"path": os.fspath(path.resolve()), "size_bytes": path.stat().st_size, "file_sha256": sha256_file(path)}
    if payload is not None and field:
        result.update({"payload_seal_field": field, "payload_sha256": payload[field]})
    return result


def atomic_json(path, payload, mode=0o400):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sealed(body):
    return {**body, "payload_sha256": sha256_bytes(canonical_bytes(body))}


def write_sealed_once(path, body):
    if os.path.lexists(path):
        raise ValueError(f"write-once artifact already exists: {path}")
    payload = sealed(body)
    atomic_json(path, payload)
    return payload


def git(root, *args):
    result = subprocess.run(["git", "-C", os.fspath(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def safe_private_directory(path, parent):
    path, parent = Path(path), Path(parent)
    if path.parent.resolve() != parent.resolve() or path.is_symlink():
        raise ValueError(f"private directory path is not anchored: {path}")
    if not path.exists():
        path.mkdir(mode=0o700)
    before = os.lstat(path)
    if not stat.S_ISDIR(before.st_mode) or before.st_uid != os.getuid():
        raise ValueError(f"private directory is unsafe: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_uid) != (before.st_dev, before.st_ino, before.st_uid):
            raise ValueError("private directory changed while opening")
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (after.st_dev, after.st_ino, after.st_uid, stat.S_IMODE(after.st_mode)) != (before.st_dev, before.st_ino, before.st_uid, 0o700):
        raise ValueError("private directory normalization failed")


def audit_environment():
    bad_env = [name for name in FORBIDDEN_ENV if os.environ.get(name)]
    bad_modules = [name for name in FORBIDDEN_MODULES if name in sys.modules]
    if bad_env or bad_modules:
        raise ValueError(f"CPU stage environment is unsafe: env={bad_env}, modules={bad_modules}")
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "forbidden_environment_variables_absent": True,
        "model_gpu_api_modules_not_imported": True,
        "slurm_environment_absent": True,
    }


def audit_repository():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("sequential checkout is missing or unsafe")
    commit = git(REPO_ROOT, "rev-parse", "HEAD")
    parents = git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit).split()
    if (
        parents != [commit, DIRECT_PARENT_COMMIT]
        or git(REPO_ROOT, "rev-parse", f"{DIRECT_PARENT_COMMIT}^{{tree}}") != DIRECT_PARENT_TREE
        or git(REPO_ROOT, "branch", "--show-current") != BRANCH
        or git(REPO_ROOT, "status", "--porcelain")
    ):
        raise ValueError("sequential checkout lineage/branch/cleanliness differs")
    observed = [tuple(line.split("\t")) for line in git(REPO_ROOT, "diff", "--name-status", "--no-renames", f"{DIRECT_PARENT_COMMIT}..{commit}").splitlines()]
    expected = [("A", path) for path in ADDED_FILES]
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("sequential commit differs from exact add-only allowlist")
    files = {}
    for relative in ADDED_FILES:
        path = require_regular(REPO_ROOT / relative, f"repository file {relative}")
        expected_index = "100755" if relative.startswith("scripts/") else "100644"
        expected_mode = 0o755 if relative.startswith("scripts/") else 0o644
        index = git(REPO_ROOT, "ls-files", "-s", "--", relative).split()
        if len(index) < 4 or index[0] != expected_index or stat.S_IMODE(path.stat().st_mode) != expected_mode:
            raise ValueError(f"repository mode differs: {relative}")
        if relative in FROZEN_EXTERNAL_FILES and sha256_file(path) != FROZEN_EXTERNAL_FILES[relative]:
            raise ValueError(f"frozen cross-layer file bytes differ: {relative}")
        files[relative] = {"git_mode": expected_index, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "path": os.fspath(REPO_ROOT), "branch": BRANCH, "commit": commit,
        "direct_parent_commit": DIRECT_PARENT_COMMIT, "direct_parent_tree": DIRECT_PARENT_TREE,
        "direct_nonmerge_parent": True, "modified_files": [], "added_files": list(ADDED_FILES), "files": files,
    }


def audit_prior_terminal():
    if sha256_file(require_regular(V5_AUDITOR, "v5 auditor")) != V5_AUDITOR_SHA256:
        raise ValueError("v5 auditor bytes differ")
    result_payload = load_json(V5_RESULT, "v5 terminal result")
    result = verify_seal(result_payload, "v5 terminal result")
    if sha256_file(V5_RESULT) != V5_RESULT_SHA256 or result_payload["payload_sha256"] != V5_RESULT_PAYLOAD_SHA256:
        raise ValueError("v5 terminal result binding differs")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, os.fspath(V5_AUDITOR), "audit-terminal"], cwd=V5_REPO,
        env=environment, check=True, capture_output=True, text=True,
    )
    try:
        official = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("official v5 terminal audit output is malformed") from error
    if (
        official.get("status") != "SMOKE_GATE_RECOVERY_TERMINAL_STOP"
        or official.get("scientific_status") != "STOPPED_EXPLORATORY_SMOKE"
        or official.get("scientific_smoke_method_gates_passed") is not True
        or official.get("confirmation_authorized") is not False
        or official.get("confirmation_submitted") is not False
        or official.get("external_api_calls") != 0
        or official.get("new_gpu_h200_minutes") != 0
        or official.get("payload_sha256") != V5_RESULT_PAYLOAD_SHA256
        or result.get("scientific_status") != "STOPPED_EXPLORATORY_SMOKE"
    ):
        raise ValueError("official v5 terminal audit semantics differ")
    return {
        "auditor": binding(V5_AUDITOR), "result": binding(V5_RESULT, result_payload, "payload_sha256"),
        "official_audit": official, "all_prior_stop_namespaces_immutable": True,
    }


def tree_inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"tree is missing or unsafe: {root}")
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"tree contains symlink: {relative}")
        if path.is_dir():
            entries.append({"path": relative, "type": "directory", "mode": stat.S_IMODE(path.stat().st_mode)})
        elif path.is_file():
            entries.append({"path": relative, "type": "file", "mode": stat.S_IMODE(path.stat().st_mode), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        else:
            raise ValueError(f"tree contains nonregular entry: {relative}")
    return {"entries": entries, "inventory_sha256": sha256_bytes(canonical_bytes(entries))}


def audit_exact_private_tree(root, expected_directories, expected_files, empty_files=()):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError(f"private science root differs: {root}")
    inventory = tree_inventory(root)
    observed_directories = {
        item["path"] for item in inventory["entries"] if item["type"] == "directory"
    }
    observed_files = {
        item["path"] for item in inventory["entries"] if item["type"] == "file"
    }
    if observed_directories != set(expected_directories) or observed_files != set(expected_files):
        raise ValueError(f"science tree inventory differs: {root}")
    for item in inventory["entries"]:
        expected_mode = 0o700 if item["type"] == "directory" else 0o600
        if item["mode"] != expected_mode:
            raise ValueError(f"science tree mode differs: {root / item['path']}")
    for relative in empty_files:
        if (root / relative).stat().st_size != 0:
            raise ValueError(f"expected empty science artifact differs: {root / relative}")
    return inventory


def generation_science_tree(stage):
    root = GENERATION_ROOT / stage
    if stage == "benefit":
        methods, domain, samples = ("pi_base", *METHODS), "massive", 1
        selected = verify_seal(
            load_json(PROTOCOL_ROOT / "benefit/selection.json", "benefit selection"),
            "benefit selection",
        )
        question_ids = selected["selected_question_ids_source_order"]
    elif stage == "medical":
        methods, domain, samples = METHODS, "medical", 5
        prompts = load_json(PROTOCOL_ROOT / "medical/prompts.json", "medical prompts")
        question_ids = [row["question_id"] for row in prompts["prompts"]]
    else:
        raise ValueError("unknown generation phase")
    directories, files = set(), {
        ".sampler.lock", "run_manifest.json", "setup_timing.json", "timings.json",
    }
    for method in methods:
        stream = f"{method}/{domain}"
        shard_root = f"{stream}/shards"
        directories.update({method, stream, shard_root})
        files.update({f"{stream}/stream_manifest.json", f"{stream}/generation.json"})
        for ordinal, question_id in enumerate(question_ids):
            digest = sha256_bytes(question_id.encode("utf-8"))[:16]
            for sample_index in range(samples):
                files.add(
                    f"{shard_root}/sample-{ordinal:06d}-{digest}-n{sample_index:03d}.json"
                )
    inventory = audit_exact_private_tree(root, directories, files, {".sampler.lock"})
    for relative in files - {".sampler.lock"}:
        payload = load_json(root / relative, f"sealed generation artifact {relative}")
        verify_seal(payload, f"sealed generation artifact {relative}")
    return inventory


def stage_evaluation_tree(stage, require_stage_only=False):
    if stage == "benefit":
        root = EVALUATION_ROOT / "benefit"
        gate = gate_binding("benefit")
        files = {
            *{f"scores/{name}.json" for name in ("pi_base", *METHODS)},
            "gate/runtime_projection.json", "gate/summary.json", f"gate/{gate['status']}",
        }
        inventory = audit_exact_private_tree(root, {"scores", "gate"}, files)
        for relative in files:
            verify_seal(load_json(root / relative, relative), relative)
        return inventory
    if stage != "medical":
        raise ValueError("unknown evaluation phase")
    parent = EVALUATION_ROOT / "medical"
    if require_stage_only:
        if parent.is_symlink() or not parent.is_dir() or set(os.listdir(parent)) != {"prejudge"}:
            raise ValueError("medical result namespace contains downstream artifacts")
    gate = gate_binding("medical")
    root = parent / "prejudge"
    files = {"summary.json", gate["status"]}
    inventory = audit_exact_private_tree(root, set(), files)
    for relative in files:
        verify_seal(load_json(root / relative, relative), relative)
    return inventory


def audit_protocol(protocol_root=PROTOCOL_ROOT):
    root = Path(protocol_root)
    manifest_path = root / "manifest.json"
    payload = load_json(manifest_path, "sequential protocol manifest")
    body = verify_seal(payload, "sequential protocol manifest", "manifest_payload_sha256")
    exact_top = {
        "schema_version", "protocol_id", "created_at", "exploratory_contract", "source_v1_terminal",
        "selection", "methods", "generation", "gates", "budget", "judge", "model_panel",
        "direct_benefit", "historical_A_judgments", "copied_artifacts", "file_inventory",
    }
    if set(body) != exact_top or body.get("schema_version") != 1 or body.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("sequential protocol manifest schema/identity differs")
    contract = body["exploratory_contract"]
    required_true = (
        "exploratory_only", "all_prior_stop_decisions_remain_terminal_and_immutable",
        "benefit_subset_selected_before_answers_or_outcomes", "same_backend_paired_base_required",
        "all_three_methods_required_at_every_gate", "historical_A_reused_not_rejudged",
        "no_posthoc_method_threshold_seed_subset_or_profile_selection", "cpu_stage_only",
    )
    if any(contract.get(key) is not True for key in required_true) or contract.get("confirmatory_claim") is not False or contract.get("current_executable_gpu_paths") != 0 or contract.get("current_executable_api_paths") != 0:
        raise ValueError("exploratory contract differs")
    selection = body["selection"]
    if (
        selection.get("artifact") != "benefit/selection.json"
        or selection.get("source_rows") != 600 or selection.get("selected_rows") != 360
        or selection.get("selection_is_prompt_id_only") is not True
        or selection.get("answers_or_outcomes_opened_before_selection") is not False
        or selection.get("ranked_selected_question_ids_sha256") != RANKED_IDS_SHA256
        or selection.get("selected_question_ids_source_order_sha256") != SELECTED_IDS_SHA256
        or selection.get("rank_records_sha256") != RANK_RECORDS_SHA256
    ):
        raise ValueError("selection registry differs")
    selection_payload = load_json(root / "benefit/selection.json", "benefit selection")
    selection_body = verify_seal(selection_payload, "benefit selection")
    ids = selection_body.get("selected_question_ids_source_order")
    records = selection_body.get("rank_records")
    if (
        selection_payload["payload_sha256"] != selection.get("payload_sha256")
        or not isinstance(ids, list) or len(ids) != 360 or len(set(ids)) != 360
        or sha256_bytes(canonical_bytes(ids)) != SELECTED_IDS_SHA256
        or not isinstance(records, list) or len(records) != 360
        or sha256_bytes(canonical_bytes(records)) != RANK_RECORDS_SHA256
    ):
        raise ValueError("selection artifact differs")
    for item in records:
        material = PROTOCOL_ID + "\0benefit360\0" + item["question_id"]
        if item.get("rank_sha256") != sha256_bytes(material.encode("utf-8")):
            raise ValueError("selection rank digest differs")
    if records != sorted(records, key=lambda item: (item["rank_sha256"], item["question_id"])):
        raise ValueError("selection rank order differs")

    for relative, key, role in (
        ("benefit/prompts.json", "prompts", "sequential_benefit_prompts"),
        ("benefit/answers.json", "answers", "sequential_benefit_answers"),
    ):
        artifact = load_json(root / relative, relative)
        data = verify_seal(artifact, relative)
        rows = data.get(key)
        if set(artifact) != {"meta", key, "payload_sha256"} or data.get("meta", {}).get("role") != role or data["meta"].get("protocol_id") != PROTOCOL_ID or data["meta"].get("n_questions") != 360 or [row.get("question_id") for row in rows] != ids:
            raise ValueError(f"{relative} schema/order differs")
    prompt_payload = load_json(root / "benefit/prompts.json", "benefit prompts")
    for row in prompt_payload["prompts"]:
        if sha256_bytes(canonical_bytes({"prompt": row["prompt"]})) != row["prompt_sha256"]:
            raise ValueError("benefit prompt hash differs")
    answers_payload = load_json(root / "benefit/answers.json", "benefit answers")
    coverage_payload = answers_payload.get("meta", {}).get("intent_coverage_diagnostics")
    coverage = verify_seal(coverage_payload, "intent coverage diagnostic")
    source_counts = coverage.get("source_intent_counts")
    selected_counts = coverage.get("selected_intent_counts_including_zeros")
    actual_selected = {}
    for row in answers_payload["answers"]:
        actual_selected[row["intent"]] = actual_selected.get(row["intent"], 0) + 1
    if (
        coverage.get("protocol") != PROTOCOL_ID + "_intent_coverage_diagnostic_v1"
        or coverage.get("selected_question_ids_source_order_sha256") != SELECTED_IDS_SHA256
        or coverage.get("computed_only_after_prompt_id_selection_was_durably_fixed") is not True
        or coverage.get("used_for_ranking_reranking_gate_or_rescue") is not False
        or coverage.get("source_rows") != 600 or coverage.get("selected_rows") != 360
        or not isinstance(source_counts, dict) or sum(source_counts.values()) != 600
        or not isinstance(selected_counts, dict) or set(selected_counts) != set(source_counts)
        or sum(selected_counts.values()) != 360
        or selected_counts != {key: actual_selected.get(key, 0) for key in source_counts}
        or coverage.get("source_unique_intents") != len(source_counts)
        or coverage.get("selected_unique_intents") != sum(value > 0 for value in selected_counts.values())
        or coverage.get("missing_intents") != [key for key, value in selected_counts.items() if value == 0]
        or coverage.get("selected_intent_count_min_including_zeros") != min(selected_counts.values())
        or coverage.get("selected_intent_count_max") != max(selected_counts.values())
    ):
        raise ValueError("intent coverage diagnostic differs")

    direct = body["direct_benefit"]
    if set(direct.get("models", {})) != set(MODELS) or direct.get("rows") != 360 or direct.get("question_ids_sha256") != SELECTED_IDS_SHA256:
        raise ValueError("direct benefit registry differs")
    for name in MODELS:
        relative = f"direct_benefit/{name}.json"
        artifact = load_json(root / relative, f"{name} direct benefit")
        comparator = verify_seal(artifact, f"{name} direct benefit", "comparator_payload_sha256")
        if comparator.get("protocol_id") != PROTOCOL_ID or comparator.get("model_name") != name or [row.get("question_id") for row in comparator.get("tasks", [])] != ids:
            raise ValueError(f"{name} direct benefit differs")
        expected = direct["models"][name]
        if expected.get("path") != relative or expected.get("file_sha256") != sha256_file(root / relative) or expected.get("payload_sha256") != artifact["comparator_payload_sha256"]:
            raise ValueError(f"{name} direct benefit binding differs")

    medical = require_regular(root / "medical/prompts.json", "medical prompts")
    historical = require_regular(root / "historical/A_judgments.json", "historical A judgments")
    historical_payload = load_json(historical, "historical A judgments")
    verify_seal(historical_payload, "historical A judgments")
    if (medical.stat().st_size, sha256_file(medical)) != MEDICAL_PROMPTS_BINDING or (historical.stat().st_size, sha256_file(historical)) != HISTORICAL_A_BINDING:
        raise ValueError("medical/historical artifact bytes differ")
    copied = body.get("copied_artifacts")
    expected_copied = {
        "benefit/selection.json", "benefit/prompts.json", "benefit/answers.json",
        "medical/prompts.json", "historical/A_judgments.json",
    }
    if not isinstance(copied, dict) or set(copied) != expected_copied:
        raise ValueError("copied artifact registry differs")
    for relative, item in copied.items():
        path = require_regular(root / relative, f"copied artifact {relative}")
        if item.get("path") != relative or item.get("size_bytes") != path.stat().st_size or item.get("file_sha256") != sha256_file(path):
            raise ValueError(f"copied artifact binding differs: {relative}")
        if relative.endswith("selection.json") or relative.endswith("prompts.json") and relative.startswith("benefit/") or relative.endswith("answers.json") or relative.startswith("historical/"):
            artifact = load_json(path, f"copied artifact {relative}")
            if item.get("payload_sha256") != artifact.get(item.get("payload_seal_field")):
                raise ValueError(f"copied artifact payload binding differs: {relative}")

    if [item.get("method_id") for item in body["methods"]] != list(METHODS):
        raise ValueError("method registry differs")
    generation = body["generation"]
    if (
        generation.get("method_order") != list(METHODS)
        or generation.get("backend") != "independent_transformers_peft_models_separate_kv_caches"
        or generation.get("sequential_sampler_static_contract_sha256") != "d4deac591866d63ff5ce51f0fd1f75c406127f8f0d1428d7dae3a028e494a3db"
        or generation.get("adapter_switching") is not False
        or generation.get("benefit", {}).get("streams") != ["pi_base", *METHODS]
        or generation["benefit"].get("massive_rows") != 360
        or generation.get("medical", {}).get("streams") != list(METHODS)
        or generation["medical"].get("samples_per_method") != 80
        or generation.get("probe", {}).get("static_contract_sha256") != "b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d"
        or generation["probe"].get("probe_prompt_binding") != {
            "artifact": "benefit/prompts.json", "index": 0,
            "question_id": prompt_payload["prompts"][0]["question_id"],
            "prompt_sha256": prompt_payload["prompts"][0]["prompt_sha256"],
        }
    ):
        raise ValueError("generation registry differs")
    gates = body["gates"]
    if gates.get("benefit_each_method", {}).get("massive_rows") != 360 or gates.get("medical_each_method", {}).get("medical_samples") != 80 or gates.get("decision_rule", {}).get("all_three_benefit_methods_must_pass_before_medical_authorization") is not True:
        raise ValueError("sequential gates differ")
    budget = body["budget"]
    if (
        budget.get("program_ceiling_usd") != 5.0
        or budget.get("benefit", {}).get("projected_seconds") != 3760.118300555879
        or budget["benefit"].get("future_h200_minutes_cap") != 65
        or budget["benefit"].get("future_gpu_cost_cap_usd") != 0.975
        or budget.get("medical", {}).get("projected_seconds") != 5355.448429139269
        or budget["medical"].get("future_h200_minutes_cap") != 95
        or budget["medical"].get("future_gpu_cost_cap_usd") != 1.425
        or budget.get("judge", {}).get("future_api_cost_cap_usd") != 0.75
        or budget.get("incremental_future_max_usd") != 3.15
        or budget.get("exact_cumulative_max_usd") != 4.846936
        or budget.get("conservative_cumulative_max_usd") != 4.90375
        or budget.get("cpu_stage_authorizes_gpu_or_api") is not False
    ):
        raise ValueError("budget registry differs")
    actual_inventory = tree_inventory(root)["entries"]
    actual_files = [{"path": item["path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]} for item in actual_inventory if item["type"] == "file" and item["path"] != "manifest.json"]
    if body.get("file_inventory") != actual_files:
        raise ValueError("protocol file inventory differs")
    return {"protocol_id": PROTOCOL_ID, **binding(manifest_path, payload, "manifest_payload_sha256"), "tree_inventory_sha256": tree_inventory(root)["inventory_sha256"]}


def log_namespace_absent():
    matches = list(LOG_ROOT.glob(LOG_PREFIX + "*")) if LOG_ROOT.exists() else []
    if matches:
        raise ValueError("sequential log namespace is not fresh")
    return True


def exact_output_phase(phase):
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir() or stat.S_IMODE(OUTPUT_ROOT.stat().st_mode) != 0o700:
        raise ValueError("sequential output root differs")
    expected = {
        "prep": {"control/PREP.json"},
        "protocol": {"control/PREP.json"},
        "preflight": {"control/PREP.json", "control/SAMPLER_PREFLIGHT_BENEFIT.json", "control/SAMPLER_PREFLIGHT_MEDICAL.json"},
        "staged": {"control/PREP.json", "control/SAMPLER_PREFLIGHT_BENEFIT.json", "control/SAMPLER_PREFLIGHT_MEDICAL.json", "control/STAGED.json"},
    }[phase]
    observed_control = set()
    if CONTROL_ROOT.is_symlink() or not CONTROL_ROOT.is_dir() or stat.S_IMODE(CONTROL_ROOT.stat().st_mode) != 0o700:
        raise ValueError("sequential control root differs")
    for path in CONTROL_ROOT.iterdir():
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) not in (0o400, 0o600):
            raise ValueError("sequential control entry is unsafe")
        observed_control.add("control/" + path.name)
    if observed_control != expected:
        raise ValueError(f"sequential control inventory differs for {phase}")
    children = {path.name for path in OUTPUT_ROOT.iterdir()}
    expected_children = {"control"} if phase == "prep" else {"control", "protocol"}
    if children != expected_children:
        raise ValueError(f"sequential output inventory differs for {phase}")
    if phase != "prep" and (PROTOCOL_ROOT.is_symlink() or not PROTOCOL_ROOT.is_dir() or stat.S_IMODE(PROTOCOL_ROOT.stat().st_mode) != 0o700):
        raise ValueError("sequential protocol root mode differs")
    log_namespace_absent()


def audit_prep():
    if STAGED_FILE.exists():
        for path in (OUTPUT_ROOT, CONTROL_ROOT, PROTOCOL_ROOT):
            if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != 0o700:
                raise ValueError("staged sequential namespace root differs")
        require_regular(PREP_FILE, "sequential PREP")
    elif not PROTOCOL_ROOT.exists():
        phase = "prep"
    elif BENEFIT_PREFLIGHT_FILE.exists() or MEDICAL_PREFLIGHT_FILE.exists():
        if not (BENEFIT_PREFLIGHT_FILE.exists() and MEDICAL_PREFLIGHT_FILE.exists()):
            raise ValueError("only one sequential sampler preflight exists")
        phase = "preflight"
    else:
        phase = "protocol"
    if not STAGED_FILE.exists():
        exact_output_phase(phase)
    payload = load_json(PREP_FILE, "sequential PREP")
    body = verify_seal(payload, "sequential PREP")
    environment = body.get("environment")
    if (
        body.get("workflow_id") != WORKFLOW_ID or body.get("repository") != audit_repository()
        or body.get("prior_terminal") != audit_prior_terminal()
        or not isinstance(environment, dict)
        or environment.get("forbidden_environment_variables_absent") is not True
        or environment.get("model_gpu_api_modules_not_imported") is not True
        or environment.get("slurm_environment_absent") is not True
        or not isinstance(environment.get("python"), str)
    ):
        raise ValueError("sequential PREP live provenance differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def audit_sampler_preflight(path, phase):
    payload = load_json(path, f"{phase} sampler preflight")
    exact_keys = {
        "schema_version", "status", "protocol_id", "phase", "protocol_manifest_file_sha256",
        "protocol_manifest_payload_sha256", "runtime", "base_model_snapshot",
        "sequential_sampler_contract", "cache_equivalence_probe_contract", "phase_budget_binding",
        "stream_registry", "stream_plan", "benefit_selection_binding", "probe_prompt_binding",
        "schema_sha256", "intent_leaves_checked", "slot_leaves_checked", "invalid_probes_rejected",
        "recorded_hybrid_intent_probes_rejected", "recorded_hybrid_slot_probes_rejected",
        "flexible_whitespace_probes_reproduced", "whitespace_probes_rejected", "gpu_loaded",
        "model_weights_loaded", "generation_performed", "output_files_written", "output_root_written",
    }
    if set(payload) != exact_keys or payload.get("schema_version") != 1 or payload.get("status") != "CPU_PREFLIGHT_OK" or payload.get("phase") != phase or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{phase} sampler preflight identity differs")
    if any(payload.get(key) is not False for key in ("gpu_loaded", "model_weights_loaded", "generation_performed", "output_files_written", "output_root_written")):
        raise ValueError(f"{phase} sampler preflight was not CPU/read-only")
    contract = payload.get("cache_equivalence_probe_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("contract_sha256") != "b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d"
        or sha256_bytes(canonical_bytes({key: value for key, value in contract.items() if key != "contract_sha256"})) != contract.get("contract_sha256")
    ):
        raise ValueError(f"{phase} sampler preflight probe contract differs")
    sequential = payload.get("sequential_sampler_contract")
    if (
        not isinstance(sequential, dict)
        or sequential.get("contract_sha256") != "d4deac591866d63ff5ce51f0fd1f75c406127f8f0d1428d7dae3a028e494a3db"
        or sha256_bytes(canonical_bytes({key: value for key, value in sequential.items() if key != "contract_sha256"})) != sequential.get("contract_sha256")
    ):
        raise ValueError(f"{phase} sequential sampler contract differs")
    manifest = load_json(PROTOCOL_ROOT / "manifest.json", "protocol manifest")
    if payload.get("protocol_manifest_file_sha256") != sha256_file(PROTOCOL_ROOT / "manifest.json") or payload.get("protocol_manifest_payload_sha256") != manifest["manifest_payload_sha256"]:
        raise ValueError(f"{phase} sampler preflight manifest binding differs")
    body = verify_seal(manifest, "protocol manifest", "manifest_payload_sha256")
    expected_registry = (["pi_base:massive", *[f"{method}:massive" for method in METHODS]] if phase == "benefit" else [f"{method}:medical" for method in METHODS])
    expected_plan = ([{"method_id": name, "domain": "massive", "rows": 360, "n_samples_per_row": 1, "sample_count": 360} for name in ("pi_base", *METHODS)] if phase == "benefit" else [{"method_id": name, "domain": "medical", "rows": 16, "n_samples_per_row": 5, "sample_count": 80} for name in METHODS])
    selected = load_json(PROTOCOL_ROOT / "benefit/selection.json", "benefit selection")
    expected_selection = {
        "artifact": "benefit/selection.json", "file_sha256": sha256_file(PROTOCOL_ROOT / "benefit/selection.json"),
        "payload_sha256": selected["payload_sha256"], "rows": 360, "question_ids_sha256": SELECTED_IDS_SHA256,
    }
    if (
        payload.get("runtime") != {"torch": "2.9.0+cu129", "transformers": "4.57.6", "peft": "0.18.1", "xgrammar": "0.1.25"}
        or payload.get("phase_budget_binding") != body["budget"][phase]
        or payload.get("stream_registry") != expected_registry or payload.get("stream_plan") != expected_plan
        or payload.get("benefit_selection_binding") != expected_selection
        or payload.get("probe_prompt_binding") != body["generation"]["probe"]["probe_prompt_binding"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("schema_sha256", "")))
        or tuple(payload.get(key) for key in ("intent_leaves_checked", "slot_leaves_checked", "invalid_probes_rejected", "recorded_hybrid_intent_probes_rejected", "recorded_hybrid_slot_probes_rejected", "flexible_whitespace_probes_reproduced", "whitespace_probes_rejected")) != (60, 55, 9, 4, 3, 2, 2)
    ):
        raise ValueError(f"{phase} sampler preflight registry/counts differ")
    snapshot = payload.get("base_model_snapshot")
    snapshot_body = {key: value for key, value in snapshot.items() if key != "snapshot_payload_sha256"} if isinstance(snapshot, dict) else {}
    expected_artifacts = lambda items: [
        {"path": name, "size_bytes": size, "sha256": digest}
        for name, size, digest in items
    ]
    hub_cache = snapshot_body.get("hub_cache")
    if (
        set(snapshot_body) != {
            "schema_version", "protocol", "model_id", "revision", "hub_cache",
            "snapshot_path", "runtime_artifacts", "safetensors_index", "safetensors_shards",
        }
        or snapshot.get("snapshot_payload_sha256") != sha256_bytes(canonical_bytes(snapshot_body))
        or snapshot_body.get("schema_version") != 1
        or snapshot_body.get("protocol") != "qwen2_5_7b_instruct_local_snapshot_v1"
        or snapshot_body.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"
        or snapshot_body.get("revision") != "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
        or not isinstance(hub_cache, str) or not os.path.isabs(hub_cache)
        or snapshot_body.get("snapshot_path") != os.path.join(
            hub_cache, "models--Qwen--Qwen2.5-7B-Instruct", "snapshots",
            "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        )
        or snapshot_body.get("runtime_artifacts") != expected_artifacts(BASE_RUNTIME_ARTIFACTS)
        or snapshot_body.get("safetensors_index") != expected_artifacts((BASE_SAFETENSORS_INDEX,))[0]
        or snapshot_body.get("safetensors_shards") != expected_artifacts(BASE_SAFETENSORS_SHARDS)
    ):
        raise ValueError(f"{phase} base-model snapshot binding differs")
    return binding(path)


def command_audit_preflight(args):
    result = audit_sampler_preflight(stage_config(args.stage)["preflight"], args.stage)
    print(json.dumps({"status": "SEQUENTIAL_PREFLIGHT_OK", "stage": args.stage, **result}, sort_keys=True))


def command_write_prep(_args):
    if os.path.lexists(OUTPUT_ROOT):
        raise ValueError("sequential output root is not fresh")
    log_namespace_absent()
    repository = audit_repository()
    prior = audit_prior_terminal()
    environment = audit_environment()
    safe_private_directory(OUTPUT_ROOT, OUTPUT_ROOT.parent)
    safe_private_directory(CONTROL_ROOT, OUTPUT_ROOT)
    body = {
        "schema_version": 1, "workflow_id": WORKFLOW_ID, "protocol_id": PROTOCOL_ID,
        "repository": repository, "prior_terminal": prior, "environment": environment,
        "cpu_stage_only": True, "slurm_jobs_submitted": 0, "gpu_h200_minutes_authorized": 0,
        "gpu_cost_authorized_usd": 0, "external_api_cost_authorized_usd": 0,
        "model_loaded": False, "generation_performed": False, "confirmation_authorized": False,
        "medical_authorized": False, "external_judge_authorized": False,
    }
    atomic_json(PREP_FILE, sealed(body))
    exact_output_phase("prep")


def command_audit_prep(_args):
    result = audit_prep()
    print(json.dumps({"status": "SEQUENTIAL_PREP_AUDITED", "payload_sha256": result["payload_sha256"]}, sort_keys=True))


def command_write_staged(_args):
    prep = audit_prep()
    if audit_environment() != prep["environment"]:
        raise ValueError("CPU-stage environment drifted from PREP")
    protocol = audit_protocol()
    benefit = audit_sampler_preflight(BENEFIT_PREFLIGHT_FILE, "benefit")
    medical = audit_sampler_preflight(MEDICAL_PREFLIGHT_FILE, "medical")
    if audit_prior_terminal() != prep["prior_terminal"] or audit_repository() != prep["repository"]:
        raise ValueError("lineage changed during CPU stage")
    body = {
        "schema_version": 1, "workflow_id": WORKFLOW_ID, "protocol_id": PROTOCOL_ID,
        "prep": binding(PREP_FILE, load_json(PREP_FILE, "PREP"), "payload_sha256"),
        "protocol": protocol, "benefit_sampler_preflight": benefit, "medical_sampler_preflight": medical,
        "cpu_stage_complete": True, "slurm_jobs_submitted": 0, "gpu_h200_minutes_authorized": 0,
        "external_api_cost_authorized_usd": 0, "benefit_authorized": False,
        "medical_authorized": False, "external_judge_authorized": False,
        "next_action_requires_separate_user_authorization": "benefit_job_cap_65_h200_minutes_0.975_usd",
    }
    atomic_json(STAGED_FILE, sealed(body))
    exact_output_phase("staged")
    print(json.dumps({"status": "SEQUENTIAL_CPU_STAGE_COMPLETE", "payload_sha256": sealed(body)["payload_sha256"]}, sort_keys=True))


def audit_staged():
    base_control = {"PREP.json", "SAMPLER_PREFLIGHT_BENEFIT.json", "SAMPLER_PREFLIGHT_MEDICAL.json", "STAGED.json"}
    if not base_control.issubset({path.name for path in CONTROL_ROOT.iterdir()}):
        raise ValueError("sequential staged base control inventory differs")
    for path in (OUTPUT_ROOT, CONTROL_ROOT, PROTOCOL_ROOT):
        if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != 0o700:
            raise ValueError("sequential staged namespace mode differs")
    payload = load_json(STAGED_FILE, "sequential STAGED")
    body = verify_seal(payload, "sequential STAGED")
    prep = audit_prep()
    if body.get("protocol") != audit_protocol() or body.get("benefit_sampler_preflight") != audit_sampler_preflight(BENEFIT_PREFLIGHT_FILE, "benefit") or body.get("medical_sampler_preflight") != audit_sampler_preflight(MEDICAL_PREFLIGHT_FILE, "medical") or body.get("prep") != binding(PREP_FILE, load_json(PREP_FILE, "PREP"), "payload_sha256") or audit_prior_terminal() != prep["prior_terminal"] or audit_repository() != prep["repository"]:
        raise ValueError("sequential STAGED live provenance differs")
    fixed = {"cpu_stage_complete": True, "slurm_jobs_submitted": 0, "gpu_h200_minutes_authorized": 0, "external_api_cost_authorized_usd": 0, "benefit_authorized": False, "medical_authorized": False, "external_judge_authorized": False}
    if any(body.get(key) != value for key, value in fixed.items()):
        raise ValueError("sequential STAGED safety flags differ")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_staged(_args):
    result = audit_staged()
    print(json.dumps({"status": "SEQUENTIAL_STAGED_AUDITED", "payload_sha256": result["payload_sha256"], "next_action": result["next_action_requires_separate_user_authorization"]}, sort_keys=True))


def command_status(_args):
    if not os.path.lexists(OUTPUT_ROOT):
        print(json.dumps({"status": "SEQUENTIAL_NOT_STAGED"}, sort_keys=True)); return
    if not PREP_FILE.exists():
        raise ValueError("sequential output exists without PREP")
    if not STAGED_FILE.exists():
        prep = audit_prep()
        print(json.dumps({"status": "SEQUENTIAL_CPU_STAGE_INCOMPLETE", "prep_payload_sha256": prep["payload_sha256"]}, sort_keys=True)); return
    result = audit_staged()
    if FINAL_RESULT.exists():
        command_audit_final_result(_args); return
    if (CONTROL_ROOT / "STOPPED_external_judge").exists():
        print(json.dumps({"status": "SEQUENTIAL_TERMINAL_STOP", "stage": "external_judge", "retry_authorized": False}, sort_keys=True)); return
    for stage in ("medical", "benefit"):
        config = stage_config(stage)
        if config["stop"].exists():
            print(json.dumps({"status": "SEQUENTIAL_TERMINAL_STOP", "stage": stage, "retry_authorized": False}, sort_keys=True)); return
        if config["result"].exists():
            sealed_result = audit_result(stage)
            try:
                accounting = terminal_accounting(stage)
            except (OSError, ValueError, subprocess.SubprocessError):
                print(json.dumps({"status": f"{stage.upper()}_RESULT_SEALED_AWAITING_TERMINAL", "scientific_status": sealed_result["scientific_status"]}, sort_keys=True)); return
            print(json.dumps({"status": f"{stage.upper()}_TERMINAL_SEALED", "scientific_status": sealed_result["scientific_status"], **accounting}, sort_keys=True)); return
        if config["job"].exists():
            job = job_pointer(stage)
            raw = subprocess.check_output(["squeue", "-h", "-j", job["job_id"], "-o", "%T|%r"], text=True).strip()
            print(json.dumps({"status": f"{stage.upper()}_JOB_ACTIVE", "job_id": job["job_id"], "scheduler": raw}, sort_keys=True)); return
    print(json.dumps({"status": "SEQUENTIAL_CPU_STAGED_AWAITING_BENEFIT_AUTHORIZATION", "payload_sha256": result["payload_sha256"], "benefit_cap_h200_minutes": 65, "benefit_cap_usd": 0.975, "medical_authorized": False, "external_api_authorized": False}, sort_keys=True))


FIELD_RE = re.compile(r"(?:^| )([A-Za-z][A-Za-z0-9_/.-]*)=")


def parse_scontrol_line(line):
    if not isinstance(line, str) or not line.strip():
        raise ValueError("empty scontrol record")
    matches = list(FIELD_RE.finditer(line.strip()))
    if not matches or matches[0].group(1) != "JobId":
        raise ValueError("scontrol record does not start with JobId")
    fields = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise ValueError(f"scontrol repeats {key}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        fields[key] = line[match.end():end].strip()
    return fields


def parse_tres(value):
    result = {}
    for term in (value or "").split(","):
        if "=" not in term:
            raise ValueError(f"invalid TRES term: {term}")
        key, item = term.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid TRES term: {term}")
        result[key] = item
    return result


def expected_tres():
    return {"billing": "8", "cpu": "8", "gres/gpu:h200": "1", "gres/gpu": "1", "mem": "180G", "node": "1"}


def audit_job_record(stage, job_id, raw, fields, phase, check_logs=True):
    config = stage_config(stage)
    exact = {
        "JobId": str(job_id), "JobName": config["job_name"], "Account": "stf", "QOS": "normal",
        "Requeue": "0", "Restarts": "0", "Partition": "gpu-h200", "NumTasks": "1",
        "NumCPUs": "8", "CPUs/Task": "8", "TimeLimit": config["time_limit"],
        "Command": os.fspath(config["sbatch"]), "WorkDir": os.fspath(REPO_ROOT),
        "StdOut": os.fspath(LOG_ROOT / f"{config['log_prefix']}_{job_id}.out"),
        "StdErr": os.fspath(LOG_ROOT / f"{config['log_prefix']}_{job_id}.err"),
        "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            raise ValueError(f"{stage} scheduler record differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"} or parse_tres(fields.get("ReqTRES")) != expected_tres():
        raise ValueError(f"{stage} requested resources differ")
    if fields.get("Dependency") not in {None, "", "(null)"} or fields.get("KillOnInvalidDependent", "") not in {"", "No"}:
        raise ValueError(f"{stage} unexpectedly has a dependency")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError(f"{stage} unexpectedly belongs to an array/heterogeneous job")
    if phase == "held":
        expected_submit = f"sbatch --parsable --hold --export=NONE --job-name={config['job_name']} " + os.path.relpath(config["sbatch"], REPO_ROOT)
        if fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser" or fields.get("RunTime") != "00:00:00" or fields.get("AllocTRES") != "(null)" or fields.get("MinMemoryNode") != "180G" or fields.get("SubmitLine") != expected_submit:
            raise ValueError(f"{stage} job was not pristine and held-first")
        if check_logs:
            for suffix in ("out", "err"):
                if os.path.lexists(LOG_ROOT / f"{config['log_prefix']}_{job_id}.{suffix}"):
                    raise ValueError(f"held {stage} job already has logs")
    elif phase == "running":
        if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None" or parse_tres(fields.get("AllocTRES")) != expected_tres():
            raise ValueError(f"{stage} job is not running with exact resources")
        node = fields.get("NodeList", "")
        if re.fullmatch(r"g[0-9]+", node) is None or fields.get("BatchHost") != node:
            raise ValueError(f"{stage} allocated node differs")
    else:
        raise ValueError("unknown scheduler phase")
    return {
        "stage": stage, "job_id": str(job_id), "job_name": config["job_name"], "scheduler_phase": phase,
        "scontrol_record": raw, "scontrol_record_sha256": sha256_bytes(raw.encode("utf-8")),
        "requested_tres": expected_tres(), "time_limit": config["time_limit"],
        "maximum_h200_minutes": config["minutes"], "maximum_gpu_cost_usd": config["cost"],
        "held_first": True, "no_requeue": True, "dependencies": [],
    }


def audit_live_job(stage, job_id, phase, check_logs=True):
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", str(job_id)], text=True).strip()
    result = audit_job_record(stage, job_id, raw, parse_scontrol_line(raw), phase, check_logs)
    if phase == "held":
        completed = subprocess.run(["scontrol", "write", "batch_script", str(job_id), "-"], check=True, capture_output=True)
        spooled = completed.stdout if isinstance(completed.stdout, bytes) else completed.stdout.encode()
        committed = require_regular(stage_config(stage)["sbatch"], f"{stage} sbatch").read_bytes()
        if spooled != committed:
            raise ValueError(f"Slurm-spooled {stage} script differs")
        result.update({"spooled_script_sha256": sha256_bytes(spooled), "committed_script_sha256": sha256_bytes(committed)})
    return result


def parse_attempt(stage, job_id):
    path = require_regular(stage_config(stage)["attempt"], f"{stage} submission attempt")
    if path.read_bytes() != f"stage\tjob_id\n{stage}\t{job_id}\n".encode():
        raise ValueError(f"{stage} submission attempt differs")
    return binding(path)


def lock_binding(stage):
    directory = stage_config(stage)["lock"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{stage} permanent lock is absent")
    owner = require_regular(directory / "owner", f"{stage} lock owner", 0o400)
    expected = f"workflow_id={WORKFLOW_ID}\nstage={stage}\nrepo_commit={git(REPO_ROOT, 'rev-parse', 'HEAD')}\n".encode()
    if owner.read_bytes() != expected:
        raise ValueError(f"{stage} lock owner differs")
    return {"path": os.fspath(directory), "owner": binding(owner)}


def submitted_binding(stage, job_id):
    path = require_regular(stage_config(stage)["submitted"], f"{stage} submitted marker", 0o400)
    if path.read_bytes() != f"stage={stage}\njob_id={job_id}\nheld_first=true\n".encode():
        raise ValueError(f"{stage} submitted marker differs")
    return binding(path)


def release_binding(stage, job_id):
    path = require_regular(stage_config(stage)["released"], f"{stage} release marker", 0o400)
    if path.read_bytes() != f"stage={stage}\njob_id={job_id}\nheld_audit_passed=true\nrelease_authorized=true\n".encode():
        raise ValueError(f"{stage} release marker differs")
    return binding(path)


def job_pointer(stage):
    config = stage_config(stage)
    payload = load_json(config["job"], f"{stage} job pointer")
    body = verify_seal(payload, f"{stage} job pointer")
    held = body.get("held_audit")
    if body.get("workflow_id") != WORKFLOW_ID or body.get("stage") != stage or not re.fullmatch(r"[0-9]+", str(body.get("job_id", ""))) or not isinstance(held, dict):
        raise ValueError(f"{stage} job pointer differs")
    reconstructed = audit_job_record(stage, body["job_id"], held.get("scontrol_record"), parse_scontrol_line(held.get("scontrol_record")), "held", False)
    script_sha = sha256_file(config["sbatch"])
    if held != {**reconstructed, "spooled_script_sha256": script_sha, "committed_script_sha256": script_sha} or body.get("submission_attempt") != parse_attempt(stage, body["job_id"]) or body.get("permanent_submission_lock") != lock_binding(stage):
        raise ValueError(f"{stage} stored held audit differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def auth_pointer(stage):
    config = stage_config(stage)
    payload = load_json(config["auth"], f"{stage} authorization")
    body = verify_seal(payload, f"{stage} authorization")
    job = job_pointer(stage)
    staged = audit_staged()
    config = stage_config(stage)
    prior_benefit = terminal_accounting("benefit") if stage == "medical" else None
    prerequisite = assert_medical_eligible() if stage == "medical" else None
    actual_plus = (prior_benefit["actual_gpu_cost_usd"] if prior_benefit else 0.0) + config["cost"]
    expected_keys = {
        "schema_version", "workflow_id", "created_at", "stage", "job_id", "job",
        "staged_payload_sha256", "maximum_h200_minutes", "maximum_gpu_cost_usd",
        "prior_benefit_actual", "prior_benefit_prerequisite", "actual_plus_current_cap_gpu_cost_usd",
        "program_exact_actual_before_new_work_usd", "program_exact_max_after_current_stage_usd",
        "pre_release_repository", "pre_release_protocol", "prior_terminal", "held_first",
        "no_requeue", "no_dependency", "no_retry", "training", "external_api_calls",
        "automatic_continuation",
    }
    if (
        set(body) != expected_keys or body.get("schema_version") != 1
        or body.get("workflow_id") != WORKFLOW_ID or body.get("stage") != stage or body.get("job_id") != job["job_id"]
        or body.get("job") != binding(config["job"], load_json(config["job"], f"{stage} job"), "payload_sha256")
        or body.get("staged_payload_sha256") != staged["payload_sha256"]
        or body.get("maximum_h200_minutes") != config["minutes"] or body.get("maximum_gpu_cost_usd") != config["cost"]
        or body.get("prior_benefit_actual") != prior_benefit
        or body.get("prior_benefit_prerequisite") != prerequisite
        or body.get("actual_plus_current_cap_gpu_cost_usd") != actual_plus
        or body.get("program_exact_actual_before_new_work_usd") != 1.696936
        or body.get("program_exact_max_after_current_stage_usd") != 1.696936 + actual_plus
        or body.get("pre_release_repository") != audit_repository() or body.get("pre_release_protocol") != audit_protocol()
        or body.get("prior_terminal") != audit_prior_terminal() or body.get("held_first") is not True
        or body.get("no_requeue") is not True or body.get("no_dependency") is not True or body.get("no_retry") is not True
        or body.get("training") is not False or body.get("external_api_calls") != 0
        or body.get("automatic_continuation") is not False
        or not isinstance(body.get("created_at"), str)
    ):
        raise ValueError(f"{stage} authorization differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def science_namespace_absent(stage):
    for root in (GENERATION_ROOT / stage, EVALUATION_ROOT / stage):
        if os.path.lexists(root):
            raise ValueError(f"{stage} science namespace is not fresh: {root}")


def gate_binding(stage):
    root = EVALUATION_ROOT / stage / ("gate" if stage == "benefit" else "prejudge")
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError(f"{stage} gate root differs")
    allowed = ("EXPLORATORY_BENEFIT_PASSED", "EXPLORATORY_SEQUENTIAL_NO_SUPPORT") if stage == "benefit" else ("AWAITING_EXTERNAL_JUDGE", "EXPLORATORY_SEQUENTIAL_NO_SUPPORT")
    present = [name for name in allowed if os.path.lexists(root / name)]
    if len(present) != 1:
        raise ValueError(f"{stage} lacks exactly one gate sentinel")
    status = present[0]
    expected_names = {"summary.json", status}
    if stage == "benefit":
        expected_names.add("runtime_projection.json")
    if set(os.listdir(root)) != expected_names:
        raise ValueError(f"{stage} gate inventory differs")
    for name in expected_names:
        require_regular(root / name, f"{stage} gate artifact {name}", 0o600)
    summary_path, sentinel_path = root / "summary.json", root / status
    summary_payload = load_json(summary_path, f"{stage} summary")
    sentinel_payload = load_json(sentinel_path, f"{stage} sentinel")
    summary = verify_seal(summary_payload, f"{stage} summary")
    sentinel = verify_seal(sentinel_payload, f"{stage} sentinel")
    manifest = load_json(PROTOCOL_ROOT / "manifest.json", "protocol manifest")
    if (
        summary.get("status") != status or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("protocol_manifest_file_sha256") != sha256_file(PROTOCOL_ROOT / "manifest.json")
        or summary.get("protocol_manifest_payload_sha256") != manifest["manifest_payload_sha256"]
        or sentinel.get("status") != status or Path(sentinel.get("summary_path", "")).resolve() != summary_path.resolve()
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
        or summary.get("confirmatory_claim") is not False
    ):
        raise ValueError(f"{stage} gate provenance differs")
    if stage == "benefit":
        science = summary.get("all_three_methods_passed")
        runtime = summary.get("runtime_gates_passed")
        prerequisite = science is True and runtime is True
        passed = status == "EXPLORATORY_BENEFIT_PASSED"
        if (
            not isinstance(science, bool) or not isinstance(runtime, bool)
            or passed is not prerequisite
            or summary.get("medical_stage_prerequisite_satisfied") is not prerequisite
            or sentinel.get("medical_stage_prerequisite_satisfied") is not prerequisite
            or summary.get("medical_authorized") is not False
            or sentinel.get("medical_authorized") is not False
            or "medical_submission_eligible" in summary
            or "medical_generation_gate_eligible" in summary
        ):
            raise ValueError("benefit downstream prerequisite differs")
    else:
        passed = status == "AWAITING_EXTERNAL_JUDGE"
        if (
            summary.get("all_three_methods_passed") is not passed
            or summary.get("external_judge_prerequisite_satisfied") is not passed
            or sentinel.get("external_judge_prerequisite_satisfied") is not passed
            or summary.get("external_api_authorized") is not False
            or sentinel.get("external_api_authorized") is not False
            or "external_judge_eligible" in summary
            or "external_judge_gate_eligible" in summary
        ):
            raise ValueError("medical downstream prerequisite differs")
    return {"status": status, "summary": binding(summary_path, summary_payload, "payload_sha256"), "sentinel": binding(sentinel_path, sentinel_payload, "payload_sha256")}


def command_write_held_auth(args):
    stage, job_id = args.stage, str(args.job_id)
    config = stage_config(stage)
    staged = audit_staged()
    audit_environment()
    if stage == "medical":
        assert_medical_eligible()
    science_namespace_absent(stage)
    for path in (config["job"], config["auth"], config["result"], config["stop"]):
        if os.path.lexists(path):
            raise ValueError(f"{stage} control/science path already exists: {path}")
    held = audit_live_job(stage, job_id, "held")
    job_payload = write_sealed_once(config["job"], {
        "schema_version": 1, "workflow_id": WORKFLOW_ID, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage, "job_id": job_id, "held_audit": held,
        "submission_attempt": parse_attempt(stage, job_id), "permanent_submission_lock": lock_binding(stage),
    })
    prior_benefit = terminal_accounting("benefit") if stage == "medical" else None
    prerequisite = assert_medical_eligible() if stage == "medical" else None
    actual_plus = (prior_benefit["actual_gpu_cost_usd"] if prior_benefit else 0.0) + config["cost"]
    if actual_plus > 2.4 + 1e-12:
        raise ValueError("sequential released GPU cost exceeds $2.40")
    auth = {
        "schema_version": 1, "workflow_id": WORKFLOW_ID, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage, "job_id": job_id, "job": binding(config["job"], job_payload, "payload_sha256"),
        "staged_payload_sha256": staged["payload_sha256"], "maximum_h200_minutes": config["minutes"],
        "maximum_gpu_cost_usd": config["cost"], "prior_benefit_actual": prior_benefit,
        "prior_benefit_prerequisite": prerequisite,
        "actual_plus_current_cap_gpu_cost_usd": actual_plus, "program_exact_actual_before_new_work_usd": 1.696936,
        "program_exact_max_after_current_stage_usd": 1.696936 + actual_plus,
        "pre_release_repository": audit_repository(), "pre_release_protocol": audit_protocol(),
        "prior_terminal": audit_prior_terminal(), "held_first": True, "no_requeue": True,
        "no_dependency": True, "no_retry": True, "training": False, "external_api_calls": 0,
        "automatic_continuation": False,
    }
    payload = write_sealed_once(config["auth"], auth)
    print(payload["payload_sha256"])


def command_audit_held(args):
    auth = auth_pointer(args.stage)
    if auth["job_id"] != str(args.job_id):
        raise ValueError("held job differs from authorization")
    audit_live_job(args.stage, args.job_id, "held")
    submitted_binding(args.stage, str(args.job_id))
    science_namespace_absent(args.stage)
    print(f"{args.stage.upper()}_HELD_AUDIT_OK")


def command_verify_job(args):
    auth = auth_pointer(args.stage)
    if auth["job_id"] != str(args.job_id):
        raise ValueError("running job differs from authorization")
    submitted_binding(args.stage, str(args.job_id))
    release_binding(args.stage, str(args.job_id))
    science_namespace_absent(args.stage)
    running = audit_live_job(args.stage, args.job_id, "running", False)
    fields = parse_scontrol_line(running["scontrol_record"])
    expected = {
        "SLURM_JOB_ID": str(args.job_id), "SLURM_JOB_NAME": stage_config(args.stage)["job_name"],
        "SLURM_JOB_PARTITION": "gpu-h200", "SLURM_JOB_ACCOUNT": "stf", "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8", "SLURM_JOB_NUM_NODES": "1", "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": os.fspath(REPO_ROOT), "SLURM_JOB_NODELIST": fields["NodeList"],
        "SLURM_MEM_PER_NODE": "184320",
    }
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{args.stage} Slurm runtime environment differs")
    print(f"{args.stage.upper()}_RUNNING_AUDIT_OK")


def generation_manifest(stage):
    path = GENERATION_ROOT / stage / "run_manifest.json"
    payload = load_json(path, f"{stage} generation run manifest")
    body = verify_seal(payload, f"{stage} generation run manifest")
    if body.get("protocol_id") != PROTOCOL_ID or body.get("phase") != stage:
        raise ValueError(f"{stage} generation manifest differs")
    return binding(path, payload, "payload_sha256")


def command_write_result(args):
    auth = auth_pointer(args.stage)
    if auth["job_id"] != str(args.job_id):
        raise ValueError("result job differs from authorization")
    running = audit_live_job(args.stage, args.job_id, "running", False)
    gate = gate_binding(args.stage)
    generation_tree = generation_science_tree(args.stage)
    evaluation_tree = stage_evaluation_tree(args.stage, require_stage_only=True)
    payload = write_sealed_once(stage_config(args.stage)["result"], {
        "schema_version": 1, "workflow_id": WORKFLOW_ID, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": args.stage, "job_id": str(args.job_id), "authorization": binding(stage_config(args.stage)["auth"], load_json(stage_config(args.stage)["auth"], "auth"), "payload_sha256"),
        "running_job_audit": running, "generation_run_manifest": generation_manifest(args.stage),
        "gate": gate, "scientific_status": gate["status"],
        "repository": audit_repository(), "protocol": audit_protocol(), "prior_terminal": audit_prior_terminal(),
        "generation_tree": generation_tree,
        "evaluation_tree": evaluation_tree,
        "training": False, "external_api_calls": 0, "no_retry": True, "automatic_continuation": False,
        "confirmatory_claim": False,
    })
    print(payload["payload_sha256"])


def audit_result(stage):
    config = stage_config(stage)
    payload = load_json(config["result"], f"{stage} result")
    body = verify_seal(payload, f"{stage} result")
    gate = gate_binding(stage)
    auth = auth_pointer(stage)
    running = body.get("running_job_audit")
    if not isinstance(running, dict):
        raise ValueError(f"{stage} result lacks running-job audit")
    reconstructed = audit_job_record(
        stage, body.get("job_id"), running.get("scontrol_record"),
        parse_scontrol_line(running.get("scontrol_record")), "running", False,
    )
    if running != reconstructed:
        raise ValueError(f"{stage} stored running-job audit differs")
    expected_keys = {
        "schema_version", "workflow_id", "created_at", "stage", "job_id",
        "authorization", "running_job_audit", "generation_run_manifest", "gate",
        "scientific_status", "repository", "protocol", "prior_terminal",
        "generation_tree", "evaluation_tree", "training", "external_api_calls",
        "no_retry", "automatic_continuation", "confirmatory_claim",
    }
    if (
        set(body) != expected_keys or body.get("schema_version") != 1
        or body.get("workflow_id") != WORKFLOW_ID or body.get("stage") != stage or body.get("job_id") != auth["job_id"]
        or not isinstance(body.get("created_at"), str)
        or body.get("authorization") != binding(config["auth"], load_json(config["auth"], f"{stage} auth"), "payload_sha256")
        or body.get("gate") != gate or body.get("scientific_status") != gate["status"]
        or body.get("generation_run_manifest") != generation_manifest(stage)
        or body.get("generation_tree") != generation_science_tree(stage)
        or body.get("evaluation_tree") != stage_evaluation_tree(stage)
        or body.get("repository") != audit_repository() or body.get("protocol") != audit_protocol()
        or body.get("prior_terminal") != audit_prior_terminal() or body.get("training") is not False
        or body.get("external_api_calls") != 0 or body.get("no_retry") is not True
        or body.get("automatic_continuation") is not False
        or body.get("confirmatory_claim") is not False
    ):
        raise ValueError(f"{stage} sealed result differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def parse_duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value or "")
    if match is None:
        raise ValueError(f"invalid Slurm duration: {value}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def terminal_accounting(stage):
    job = job_pointer(stage)
    config = stage_config(stage)
    output = subprocess.check_output(["sacct", "-n", "-X", "-P", "-j", job["job_id"], "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode"], text=True)
    matches = [(line, line.split("|")) for line in output.strip().splitlines() if line.split("|")[0] == job["job_id"]]
    if len(matches) != 1 or len(matches[0][1]) != 8:
        raise ValueError(f"{stage} lacks unique durable accounting")
    line, fields = matches[0]
    _, name, state, elapsed, limit, allocated, requested, exit_code = fields
    if name != config["job_name"] or state != "COMPLETED" or limit != config["time_limit"] or exit_code != "0:0" or parse_tres(allocated) != expected_tres() or parse_tres(requested) != expected_tres():
        raise ValueError(f"{stage} durable accounting differs")
    seconds = parse_duration(elapsed)
    if not 0 < seconds <= config["minutes"] * 60:
        raise ValueError(f"{stage} elapsed time exceeds cap")
    minutes = seconds / 60.0
    return {"stage": stage, "job_id": job["job_id"], "sacct_row": line, "sacct_row_sha256": sha256_bytes(line.encode()), "state": state, "elapsed_seconds": seconds, "actual_h200_minutes": minutes, "actual_gpu_cost_usd": minutes * 0.015, "released_h200_minutes_cap": config["minutes"], "released_gpu_cost_usd_cap": config["cost"]}


def command_audit_terminal(args):
    result = audit_result(args.stage)
    accounting = terminal_accounting(args.stage)
    print(json.dumps({"status": f"{args.stage.upper()}_TERMINAL_SEALED", "scientific_status": result["scientific_status"], **accounting}, sort_keys=True))


def assert_medical_eligible():
    benefit = audit_result("benefit")
    if benefit.get("scientific_status") != "EXPLORATORY_BENEFIT_PASSED":
        raise ValueError("medical requires sealed all-three benefit PASS")
    accounting = terminal_accounting("benefit")
    return {"benefit_result": binding(stage_config("benefit")["result"], load_json(stage_config("benefit")["result"], "benefit result"), "payload_sha256"), "benefit_accounting": accounting}


def command_assert_medical_release(args):
    value = assert_medical_eligible()
    if args.ack_benefit_actual_usd is not None:
        try:
            acknowledged = float(args.ack_benefit_actual_usd)
        except ValueError as error:
            raise ValueError("benefit actual-cost acknowledgement is invalid") from error
        actual = value["benefit_accounting"]["actual_gpu_cost_usd"]
        if not math.isclose(acknowledged, actual, rel_tol=0, abs_tol=1e-12):
            raise ValueError("benefit actual-cost acknowledgement differs from sealed accounting")
    print(json.dumps({"status": "MEDICAL_RELEASE_ELIGIBLE", **value}, sort_keys=True))


def finalizer_lock_binding():
    if FINALIZER_LOCK.is_symlink() or not FINALIZER_LOCK.is_dir():
        raise ValueError("permanent finalizer lock is absent")
    owner = require_regular(FINALIZER_LOCK / "owner", "finalizer owner", 0o400)
    expected = f"workflow_id={WORKFLOW_ID}\nstage=external_judge\nrepo_commit={git(REPO_ROOT, 'rev-parse', 'HEAD')}\n".encode()
    if owner.read_bytes() != expected:
        raise ValueError("finalizer lock owner differs")
    return {"path": os.fspath(FINALIZER_LOCK), "owner": binding(owner)}


def prejudge_record():
    gate = gate_binding("medical")
    if gate["status"] != "AWAITING_EXTERNAL_JUDGE":
        raise ValueError("external judge requires sealed medical prejudge PASS")
    sentinel_path = EVALUATION_ROOT / "medical/prejudge/AWAITING_EXTERNAL_JUDGE"
    sentinel_payload = load_json(sentinel_path, "medical prejudge sentinel")
    sentinel = verify_seal(sentinel_payload, "medical prejudge sentinel")
    summary_path = Path(sentinel["summary_path"])
    summary_payload = load_json(summary_path, "medical prejudge summary")
    verify_seal(summary_payload, "medical prejudge summary")
    return {
        "path": os.fspath(sentinel_path.resolve()),
        "file_sha256": sha256_file(sentinel_path),
        "payload_sha256": sentinel_payload["payload_sha256"],
        "summary_path": os.fspath(summary_path.resolve()),
        "summary_file_sha256": sha256_file(summary_path),
        "summary_payload_sha256": summary_payload["payload_sha256"],
    }


def judge_plan_record():
    prejudge = prejudge_record()
    plan_payload = load_json(JUDGE_PLAN, "external judge plan")
    body = verify_seal(plan_payload, "external judge plan")
    manifest_path = PROTOCOL_ROOT / "manifest.json"
    manifest_payload = load_json(manifest_path, "protocol manifest")
    summary = verify_seal(
        load_json(prejudge["summary_path"], "medical prejudge summary"),
        "medical prejudge summary",
    )
    sources = []
    for method in METHODS:
        source = summary.get("medical_generations", {}).get(method)
        if not isinstance(source, dict):
            raise ValueError("judge plan lacks frozen medical generation")
        sources.append({
            "name": method,
            "path": source.get("path"),
            "file_sha256": source.get("file_sha256"),
            "payload_sha256": source.get("payload_sha256"),
        })
    exact_keys = {
        "schema_version", "protocol", "protocol_id",
        "protocol_manifest_file_sha256", "protocol_manifest_payload_sha256",
        "prejudge_gate", "source_generations", "prompt_file_path",
        "prompt_file_sha256", "plan_sha256", "blind_ids_sha256",
        "planned_calls", "max_cost_usd", "judge_model", "rubric_sha256",
        "response_schema_sha256", "all_requests_preflighted_before_authorization",
        "contains_question_or_response_text", "external_api_calls",
    }
    prompt_path = PROTOCOL_ROOT / "medical/prompts.json"
    if (
        set(body) != exact_keys
        or body.get("schema_version") != 1
        or body.get("protocol") != PROTOCOL_ID + "_judge_plan_v1"
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("protocol_manifest_file_sha256") != sha256_file(manifest_path)
        or body.get("protocol_manifest_payload_sha256") != manifest_payload["manifest_payload_sha256"]
        or body.get("prejudge_gate") != prejudge
        or body.get("source_generations") != sources
        or Path(body.get("prompt_file_path", "")).resolve() != prompt_path.resolve()
        or body.get("prompt_file_sha256") != sha256_file(prompt_path)
        or not re.fullmatch(r"[0-9a-f]{64}", str(body.get("plan_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(body.get("blind_ids_sha256", "")))
        or body.get("planned_calls") != 240
        or body.get("max_cost_usd") != 0.75
        or body.get("judge_model") != "gpt-5-mini-2025-08-07"
        or body.get("all_requests_preflighted_before_authorization") is not True
        or body.get("contains_question_or_response_text") is not False
        or body.get("external_api_calls") != 0
    ):
        raise ValueError("external judge plan differs")
    return {
        "path": os.fspath(JUDGE_PLAN.resolve()),
        "file_sha256": sha256_file(JUDGE_PLAN),
        "payload_sha256": plan_payload["payload_sha256"],
        "plan_sha256": body["plan_sha256"],
    }


def audit_plan_ready_namespace():
    root = EVALUATION_ROOT / "medical"
    gate = gate_binding("medical")
    files = {"judge_plan.json", "prejudge/summary.json", f"prejudge/{gate['status']}"}
    return audit_exact_private_tree(root, {"prejudge"}, files)


def final_budget_accounting():
    benefit = terminal_accounting("benefit")
    medical = terminal_accounting("medical")
    gpu_actual = benefit["actual_gpu_cost_usd"] + medical["actual_gpu_cost_usd"]
    exact_max = 1.696936 + gpu_actual + 0.75
    body = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID + "_judge_budget_accounting_v1",
        "program_exact_actual_before_new_work_usd": 1.696936,
        "program_conservative_before_new_work_usd": 1.75375,
        "benefit_terminal_accounting": benefit,
        "medical_terminal_accounting": medical,
        "new_gpu_actual_cost_usd": gpu_actual,
        "external_judge_cost_cap_usd": 0.75,
        "incremental_released_max_usd": 3.15,
        "exact_program_max_after_external_judge_usd": exact_max,
        "conservative_program_max_usd": 4.90375,
        "program_ceiling_usd": 5.0,
        "within_program_ceiling": exact_max <= 5.0 + 1e-12 and 4.90375 < 5.0,
    }
    if body["within_program_ceiling"] is not True:
        raise ValueError("program cost ceiling would be exceeded")
    return sealed(body)


def command_assert_external_judge_budget(args):
    accounting = final_budget_accounting()
    for name, actual in (
        ("benefit", accounting["benefit_terminal_accounting"]["actual_gpu_cost_usd"]),
        ("medical", accounting["medical_terminal_accounting"]["actual_gpu_cost_usd"]),
    ):
        try:
            acknowledged = float(getattr(args, f"ack_{name}_actual_usd"))
        except ValueError as error:
            raise ValueError(f"{name} actual-cost acknowledgement is invalid") from error
        if not math.isclose(acknowledged, actual, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name} actual-cost acknowledgement differs from sealed accounting")
    print(json.dumps({
        "status": "EXTERNAL_JUDGE_BUDGET_ACKNOWLEDGED",
        "budget_accounting": accounting,
        "external_api_authorized": False,
    }, sort_keys=True))


def expected_final_authorization_body():
    manifest_path = PROTOCOL_ROOT / "manifest.json"
    manifest = load_json(manifest_path, "protocol manifest")
    plan = judge_plan_record()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID + "_judge_authorization_v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest_file_sha256": sha256_file(manifest_path),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "prejudge_gate": prejudge_record(),
        "plan": {key: plan[key] for key in ("path", "file_sha256", "payload_sha256")},
        "plan_sha256": plan["plan_sha256"],
        "budget_accounting": final_budget_accounting(),
        "planned_calls": 240,
        "max_cost_usd": 0.75,
        "judge_model": "gpt-5-mini-2025-08-07",
        "sdk_max_retries": 0,
        "external_api_authorized": True,
        "permanent_single_entry": True,
        "restart_or_resume_authorized": False,
        "user_authorized_exactly_240_calls_up_to_usd": 0.75,
    }


def command_audit_judge_plan(_args):
    result = judge_plan_record()
    inventory = audit_plan_ready_namespace()
    print(json.dumps({
        "status": "SEQUENTIAL_JUDGE_PLAN_AUDITED",
        "plan_sha256": result["plan_sha256"],
        "medical_plan_ready_inventory_sha256": inventory["inventory_sha256"],
        "external_api_calls": 0,
    }, sort_keys=True))


def command_write_final_auth(_args):
    medical = audit_result("medical")
    if medical.get("scientific_status") != "AWAITING_EXTERNAL_JUDGE":
        raise ValueError("external judge requires sealed medical prejudge PASS")
    audit_repository()
    audit_protocol()
    audit_prior_terminal()
    finalizer_lock_binding()
    audit_plan_ready_namespace()
    if os.path.lexists(FINAL_AUTH) or os.path.lexists(FINAL_RESULT):
        raise ValueError("finalizer authorization/result already exists")
    for path in (JUDGE_CHECKPOINT, JUDGMENTS_NEW, JUDGMENTS_MERGED, EVALUATION_ROOT / "final"):
        if os.path.lexists(path):
            raise ValueError("finalizer namespace is not fresh")
    progress = sorted(JUDGE_CHECKPOINT.parent.glob(JUDGE_CHECKPOINT.name + ".*"))
    if progress:
        raise ValueError("judge progress namespace is not fresh")
    payload = write_sealed_once(FINAL_AUTH, expected_final_authorization_body())
    print(payload["payload_sha256"])


def audit_final_auth():
    payload = load_json(FINAL_AUTH, "external judge authorization")
    body = verify_seal(payload, "external judge authorization")
    audit_repository()
    audit_protocol()
    audit_prior_terminal()
    finalizer_lock_binding()
    if body != expected_final_authorization_body():
        raise ValueError("external judge authorization differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_final_auth(_args):
    result = audit_final_auth()
    print(json.dumps({"status": "FINAL_AUTH_OK", "payload_sha256": result["payload_sha256"]}, sort_keys=True))


def judge_progress_inventory():
    auth_payload = load_json(FINAL_AUTH, "external judge authorization")
    plan = judge_plan_record()
    expected_auth = binding(FINAL_AUTH, auth_payload, "payload_sha256")
    marker_payload = load_json(JUDGE_CHECKPOINT, "judge single-entry marker")
    marker = verify_seal(marker_payload, "judge single-entry marker")
    expected_marker = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID + "_judge_single_entry_v1",
        "protocol_id": PROTOCOL_ID,
        "authorization": expected_auth,
        "plan_sha256": plan["plan_sha256"],
        "planned_calls": 240,
        "restart_or_resume_authorized": False,
        "status": "PERMANENT_SINGLE_ENTRY_STARTED",
    }
    if marker != expected_marker:
        raise ValueError("judge permanent single-entry marker differs")
    observed = sorted(JUDGE_CHECKPOINT.parent.glob(JUDGE_CHECKPOINT.name + ".*"))
    expected = [Path(os.fspath(JUDGE_CHECKPOINT) + f".{index:03d}") for index in range(1, 241)]
    if observed != expected:
        raise ValueError("judge progress shard inventory differs")
    first_meta = None
    shards = []
    final_rows = None
    previous_rows = []
    for index, path in enumerate(expected, 1):
        require_regular(path, f"judge progress shard {index}", 0o600)
        payload = load_json(path, f"judge progress shard {index}")
        body = verify_seal(payload, f"judge progress shard {index}")
        if set(body) != {"meta", "completed_calls", "last_blind_id", "judgments"}:
            raise ValueError("judge progress shard schema differs")
        rows, meta = body.get("judgments"), body.get("meta")
        if (
            not isinstance(rows, list) or len(rows) != index
            or body.get("completed_calls") != index
            or not rows or body.get("last_blind_id") != rows[-1].get("blind_id")
            or not isinstance(meta, dict)
            or any({"question", "response", "prompt"} & set(row) for row in rows if isinstance(row, dict))
            or rows[:-1] != previous_rows
        ):
            raise ValueError("judge progress shard content differs")
        if first_meta is None:
            first_meta = meta
        elif meta != first_meta:
            raise ValueError("judge progress metadata drifted")
        shards.append(binding(path, payload, "payload_sha256"))
        final_rows = rows
        previous_rows = rows
    if (
        first_meta.get("protocol") != PROTOCOL_ID + "_judge_v1"
        or first_meta.get("protocol_id") != PROTOCOL_ID
        or first_meta.get("authorization") != expected_auth
        or first_meta.get("plan_sha256") != plan["plan_sha256"]
        or first_meta.get("planned_calls") != 240
        or first_meta.get("max_api_calls") != 240
        or first_meta.get("max_cost_usd") != 0.75
        or first_meta.get("judge_model") != "gpt-5-mini-2025-08-07"
        or first_meta.get("sdk_max_retries") != 0
        or first_meta.get("permanent_single_entry") is not True
        or first_meta.get("restart_or_resume_authorized") is not False
        or first_meta.get("confirmatory_claim") is not False
    ):
        raise ValueError("judge progress metadata differs")
    if (
        len({row.get("blind_id") for row in final_rows}) != 240
        or len({row.get("api_response_id") for row in final_rows}) != 240
        or any(not isinstance(row.get("blind_id"), str) or not isinstance(row.get("api_response_id"), str) for row in final_rows)
    ):
        raise ValueError("judge progress identities are not exact240 unique")
    return {
        "marker": binding(JUDGE_CHECKPOINT, marker_payload, "payload_sha256"),
        "shards": shards,
        "last_completed_calls": 240,
        "final_progress_judgments_sha256": sha256_bytes(canonical_bytes(final_rows)),
        "final_progress_judgments_sorted_sha256": sha256_bytes(canonical_bytes(sorted(
            final_rows, key=lambda row: (row["model_name"], row["question_id"], row["sample_index"])
        ))),
        "progress_meta_sha256": sha256_bytes(canonical_bytes(first_meta)),
    }


def final_gate_binding():
    root = EVALUATION_ROOT / "final"
    allowed = ("EXPLORATORY_SEQUENTIAL_SUPPORT", "EXPLORATORY_SEQUENTIAL_NO_SUPPORT")
    present = [name for name in allowed if os.path.lexists(root / name)]
    if len(present) != 1:
        raise ValueError("final gate lacks exact terminal sentinel")
    status = present[0]
    inventory = audit_exact_private_tree(root, set(), {"summary.json", status})
    summary_path, sentinel_path = root / "summary.json", root / status
    summary_payload = load_json(summary_path, "final summary")
    sentinel_payload = load_json(sentinel_path, "final sentinel")
    summary = verify_seal(summary_payload, "final summary")
    sentinel = verify_seal(sentinel_payload, "final sentinel")
    manifest = load_json(PROTOCOL_ROOT / "manifest.json", "protocol manifest")
    passed = status == "EXPLORATORY_SEQUENTIAL_SUPPORT"
    if (
        summary.get("protocol") != PROTOCOL_ID + "_final_v1"
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("protocol_manifest_file_sha256") != sha256_file(PROTOCOL_ROOT / "manifest.json")
        or summary.get("protocol_manifest_payload_sha256") != manifest["manifest_payload_sha256"]
        or summary.get("status") != status
        or summary.get("all_three_methods_passed") is not passed
        or summary.get("confirmatory_claim") is not False
        or sentinel.get("protocol") != PROTOCOL_ID + "_final_sentinel_v1"
        or sentinel.get("status") != status
        or Path(sentinel.get("summary_path", "")).resolve() != summary_path.resolve()
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
    ):
        raise ValueError("final gate provenance differs")
    return {
        "status": status,
        "summary": binding(summary_path, summary_payload, "payload_sha256"),
        "sentinel": binding(sentinel_path, sentinel_payload, "payload_sha256"),
        "tree": inventory,
    }


def final_medical_tree(progress, final_gate):
    prejudge = gate_binding("medical")
    directories = {"prejudge"}
    files = {
        "judge_plan.json", "judge_checkpoint.json", "judgments_new.json",
        "judgments_merged.json", "prejudge/summary.json", f"prejudge/{prejudge['status']}",
        *{f"judge_checkpoint.json.{index:03d}" for index in range(1, 241)},
    }
    inventory = audit_exact_private_tree(EVALUATION_ROOT / "medical", directories, files)
    for relative in files:
        payload = load_json(EVALUATION_ROOT / "medical" / relative, relative)
        verify_seal(payload, relative)
    if len(progress["shards"]) != 240:
        raise ValueError("judge progress does not contain exact240 shards")
    return inventory


def final_science_inventory(progress, final_gate):
    if (
        GENERATION_ROOT.is_symlink() or not GENERATION_ROOT.is_dir()
        or stat.S_IMODE(GENERATION_ROOT.stat().st_mode) != 0o700
        or set(os.listdir(GENERATION_ROOT)) != {"benefit", "medical"}
    ):
        raise ValueError("final generation namespace differs")
    if (
        EVALUATION_ROOT.is_symlink() or not EVALUATION_ROOT.is_dir()
        or stat.S_IMODE(EVALUATION_ROOT.stat().st_mode) != 0o700
        or set(os.listdir(EVALUATION_ROOT)) != {"benefit", "medical", "final"}
    ):
        raise ValueError("final evaluation namespace differs")
    return {
        "benefit_generation": generation_science_tree("benefit"),
        "medical_generation": generation_science_tree("medical"),
        "benefit_evaluation": stage_evaluation_tree("benefit"),
        "medical_evaluation": final_medical_tree(progress, final_gate),
        "final_evaluation": final_gate["tree"],
    }


def audited_final_judgments(progress):
    """Independently re-audit paid evidence and its no-call merge.

    The final wrapper must never merely trust a cost copied from the judge
    output.  Recompute it from positive integer token counts and the frozen
    price table, require the exact 240-row source/identity registry, and bind
    the merged 320-row evidence back to those exact bytes.
    """
    new_payload = load_json(JUDGMENTS_NEW, "new judgments")
    new_body = verify_seal(new_payload, "new judgments")
    if set(new_body) != {"meta", "judgments"}:
        raise ValueError("new judgment wrapper schema differs")
    meta, rows = new_body.get("meta"), new_body.get("judgments")
    expected_meta_keys = {
        "schema_version", "protocol", "protocol_id",
        "protocol_manifest_file_sha256", "protocol_manifest_payload_sha256",
        "prejudge_gate", "authorization", "plan_sha256", "judge_kind",
        "judge_model", "source_wave3_model_alias",
        "historical_A_judge_model_alias", "historical_A_reused_not_rejudged",
        "rubric_sha256", "response_schema_sha256", "seed",
        "source_generations", "prompt_file_path", "prompt_file_sha256",
        "planned_calls", "max_api_calls", "max_cost_usd", "pricing",
        "sdk_max_retries", "permanent_single_entry",
        "restart_or_resume_authorized", "confirmatory_claim",
        "actual_api_calls", "actual_estimated_cost_usd",
    }
    stable_meta = {
        key: value for key, value in meta.items()
        if key not in {"actual_api_calls", "actual_estimated_cost_usd"}
    } if isinstance(meta, dict) else None
    pricing = meta.get("pricing") if isinstance(meta, dict) else None
    if (
        not isinstance(meta, dict) or set(meta) != expected_meta_keys
        or meta.get("schema_version") != 1
        or meta.get("protocol") != PROTOCOL_ID + "_judge_v1"
        or meta.get("protocol_id") != PROTOCOL_ID
        or meta.get("prejudge_gate") != prejudge_record()
        or meta.get("authorization")
        != binding(FINAL_AUTH, load_json(FINAL_AUTH, "external judge authorization"), "payload_sha256")
        or meta.get("plan_sha256") != judge_plan_record()["plan_sha256"]
        or meta.get("judge_kind") != "external_gpt_primary"
        or meta.get("judge_model") != "gpt-5-mini-2025-08-07"
        or meta.get("historical_A_reused_not_rejudged") is not True
        or meta.get("planned_calls") != 240 or meta.get("max_api_calls") != 240
        or meta.get("max_cost_usd") != 0.75 or meta.get("sdk_max_retries") != 0
        or meta.get("permanent_single_entry") is not True
        or meta.get("restart_or_resume_authorized") is not False
        or meta.get("confirmatory_claim") is not False
        or meta.get("actual_api_calls") != 240
        or pricing != {
            "input_usd_per_million_tokens": 0.25,
            "output_usd_per_million_tokens": 2.0,
        }
        or not isinstance(rows, list) or len(rows) != 240
        or sha256_bytes(canonical_bytes(rows))
        != progress["final_progress_judgments_sorted_sha256"]
        or sha256_bytes(canonical_bytes(stable_meta)) != progress["progress_meta_sha256"]
    ):
        raise ValueError("new judgment metadata/provenance differs")

    expected_row_keys = {
        "blind_id", "model_name", "question_id", "sample_index",
        "prompt_sha256", "response_sha256", "source_sample_sha256",
        "source_finish_reason", "label", "coherence", "judge_parse_valid",
        "judge_finish_reason", "judge_output_sha256", "api_response_id",
        "api_response_model", "api_usage",
    }
    pairs_by_method = {method: set() for method in METHODS}
    response_ids, blind_ids, actual_cost = set(), set(), 0.0
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError("new judgment row schema differs")
        usage = row.get("api_usage")
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        expected_cost = (
            input_tokens * 0.25 + output_tokens * 2.0
        ) / 1_000_000 if (
            isinstance(input_tokens, int) and not isinstance(input_tokens, bool)
            and isinstance(output_tokens, int) and not isinstance(output_tokens, bool)
        ) else math.inf
        method, pair = row.get("model_name"), (row.get("question_id"), row.get("sample_index"))
        if (
            method not in METHODS
            or row.get("label") not in {"SAFE", "BAD", "REFUSAL", "UNPARSEABLE"}
            or isinstance(row.get("coherence"), bool)
            or not isinstance(row.get("coherence"), int)
            or not 0 <= row["coherence"] <= 100
            or not isinstance(row.get("judge_parse_valid"), bool)
            or row.get("source_finish_reason") != "stop"
            or row.get("api_response_model") != "gpt-5-mini-2025-08-07"
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) is None
                for key in (
                    "blind_id", "prompt_sha256", "response_sha256",
                    "source_sample_sha256", "judge_output_sha256",
                )
            )
            or not isinstance(row.get("api_response_id"), str)
            or not row["api_response_id"]
            or not isinstance(usage, dict)
            or set(usage) != {"input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"}
            or isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0
            or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens <= 0
            or usage.get("total_tokens") != input_tokens + output_tokens
            or not math.isclose(
                usage.get("estimated_cost_usd", math.inf), expected_cost,
                rel_tol=0, abs_tol=1e-12,
            )
        ):
            raise ValueError("new judgment content/accounting differs")
        pairs_by_method[method].add(pair)
        response_ids.add(row["api_response_id"])
        blind_ids.add(row["blind_id"])
        actual_cost += expected_cost
    if (
        any(len(pairs_by_method[method]) != 80 for method in METHODS)
        or len(response_ids) != 240 or len(blind_ids) != 240
        or not math.isclose(
            meta.get("actual_estimated_cost_usd", math.inf), actual_cost,
            rel_tol=0, abs_tol=1e-12,
        )
        or actual_cost < 0 or actual_cost > 0.75 + 1e-12
    ):
        raise ValueError("new judgment aggregate accounting differs")

    merged_payload = load_json(JUDGMENTS_MERGED, "merged judgments")
    merged_body = verify_seal(merged_payload, "merged judgments")
    if set(merged_body) != {"meta", "judgments"}:
        raise ValueError("merged judgment wrapper schema differs")
    merged_meta, merged_rows = merged_body.get("meta"), merged_body.get("judgments")
    expected_merged_meta_keys = {
        "schema_version", "protocol", "protocol_id",
        "protocol_manifest_file_sha256", "protocol_manifest_payload_sha256",
        "prejudge_gate", "historical_A", "new_composition",
        "source_generations", "authorization",
        "historical_A_reused_not_rejudged", "historical_A_new_api_calls",
        "historical_A_source_api_calls", "historical_A_source_api_cost_usd",
        "historical_A_judge_model_alias", "new_composition_api_calls",
        "new_composition_api_cost_usd", "total_rows", "confirmatory_claim",
    }
    historical_path = PROTOCOL_ROOT / "historical/A_judgments.json"
    historical_payload = load_json(historical_path, "historical A judgments")
    if (
        not isinstance(merged_meta, dict) or set(merged_meta) != expected_merged_meta_keys
        or merged_meta.get("schema_version") != 1
        or merged_meta.get("protocol") != PROTOCOL_ID + "_merged_judgments_v1"
        or merged_meta.get("protocol_id") != PROTOCOL_ID
        or merged_meta.get("prejudge_gate") != prejudge_record()
        or merged_meta.get("historical_A")
        != {
            "path": os.fspath(historical_path.resolve()),
            "file_sha256": sha256_file(historical_path),
            "payload_sha256": historical_payload["payload_sha256"],
        }
        or merged_meta.get("new_composition")
        != {
            "path": os.fspath(JUDGMENTS_NEW.resolve()),
            "file_sha256": sha256_file(JUDGMENTS_NEW),
            "payload_sha256": new_payload["payload_sha256"],
        }
        or merged_meta.get("source_generations") != meta.get("source_generations")
        or merged_meta.get("authorization") != meta.get("authorization")
        or merged_meta.get("historical_A_reused_not_rejudged") is not True
        or merged_meta.get("historical_A_new_api_calls") != 0
        or merged_meta.get("new_composition_api_calls") != 240
        or not math.isclose(
            merged_meta.get("new_composition_api_cost_usd", math.inf),
            actual_cost, rel_tol=0, abs_tol=1e-12,
        )
        or merged_meta.get("total_rows") != 320
        or merged_meta.get("confirmatory_claim") is not False
        or not isinstance(merged_rows, list) or len(merged_rows) != 320
        or [row for row in merged_rows if row.get("model_name") in METHODS] != rows
        or len([row for row in merged_rows if row.get("model_name") == "pi_A"]) != 80
        or any({"response", "question", "prompt"} & set(row) for row in merged_rows if isinstance(row, dict))
    ):
        raise ValueError("merged judgment provenance differs")
    return {
        "new_payload": new_payload,
        "merged_payload": merged_payload,
        "actual_cost_usd": actual_cost,
    }


def command_write_final_result(_args):
    auth = audit_final_auth()
    progress = judge_progress_inventory()
    judgments = audited_final_judgments(progress)
    new_path, merged_path = JUDGMENTS_NEW, JUDGMENTS_MERGED
    new_payload, merged_payload = judgments["new_payload"], judgments["merged_payload"]
    final_gate = final_gate_binding()
    final_tree = final_medical_tree(progress, final_gate)
    science_inventory = final_science_inventory(progress, final_gate)
    payload = write_sealed_once(FINAL_RESULT, {
        "schema_version": 1, "workflow_id": WORKFLOW_ID, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "external_judge_authorization": binding(FINAL_AUTH, load_json(FINAL_AUTH, "auth"), "payload_sha256"),
        "external_judge_actual_calls": 240,
        "external_judge_actual_cost_usd": judgments["actual_cost_usd"],
        "judge_plan": judge_plan_record(), "judge_progress": progress,
        "new_judgments": binding(new_path, new_payload, "payload_sha256"),
        "merged_judgments": binding(merged_path, merged_payload, "payload_sha256"),
        "final_status": final_gate["status"], "final_gate": final_gate,
        "final_medical_tree": final_tree, "final_science_inventory": science_inventory,
        "repository": audit_repository(),
        "protocol": audit_protocol(), "prior_terminal": audit_prior_terminal(), "training": False,
        "merge_api_calls": 0, "no_retry": True, "confirmatory_claim": False,
    })
    print(payload["payload_sha256"])


def command_audit_final_result(_args):
    payload = load_json(FINAL_RESULT, "final result")
    body = verify_seal(payload, "final result")
    auth = audit_final_auth()
    progress = judge_progress_inventory()
    judgments = audited_final_judgments(progress)
    final_gate = final_gate_binding()
    new_payload = judgments["new_payload"]
    merged_payload = judgments["merged_payload"]
    expected_keys = {
        "schema_version", "workflow_id", "created_at",
        "external_judge_authorization", "external_judge_actual_calls",
        "external_judge_actual_cost_usd", "judge_plan", "judge_progress",
        "new_judgments", "merged_judgments", "final_status", "final_gate",
        "final_medical_tree", "final_science_inventory", "repository",
        "protocol", "prior_terminal", "training", "merge_api_calls",
        "no_retry", "confirmatory_claim",
    }
    if (
        set(body) != expected_keys or body.get("schema_version") != 1
        or body.get("workflow_id") != WORKFLOW_ID
        or not isinstance(body.get("created_at"), str)
        or body.get("external_judge_authorization") != binding(FINAL_AUTH, load_json(FINAL_AUTH, "auth"), "payload_sha256")
        or body.get("external_judge_actual_calls") != 240
        or not math.isclose(
            body.get("external_judge_actual_cost_usd", math.inf),
            judgments["actual_cost_usd"], rel_tol=0, abs_tol=1e-12,
        )
        or body.get("judge_plan") != judge_plan_record()
        or body.get("judge_progress") != progress
        or body.get("new_judgments") != binding(JUDGMENTS_NEW, new_payload, "payload_sha256")
        or body.get("merged_judgments") != binding(JUDGMENTS_MERGED, merged_payload, "payload_sha256")
        or body.get("final_status") != final_gate["status"]
        or body.get("final_gate") != final_gate
        or body.get("final_medical_tree") != final_medical_tree(progress, final_gate)
        or body.get("final_science_inventory") != final_science_inventory(progress, final_gate)
        or body.get("repository") != audit_repository()
        or body.get("protocol") != audit_protocol()
        or body.get("prior_terminal") != audit_prior_terminal()
        or body.get("training") is not False or body.get("merge_api_calls") != 0
        or body.get("no_retry") is not True or body.get("confirmatory_claim") is not False
        or auth.get("external_api_authorized") is not True
    ):
        raise ValueError("final result differs")
    print(json.dumps({"status": "SEQUENTIAL_FINAL_TERMINAL", "scientific_status": body["final_status"], "payload_sha256": payload["payload_sha256"]}, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-protocol").add_argument("--protocol-root", default=os.fspath(PROTOCOL_ROOT))
    commands.add_parser("write-prep").set_defaults(function=command_write_prep)
    commands.add_parser("audit-prep").set_defaults(function=command_audit_prep)
    commands.add_parser("write-staged").set_defaults(function=command_write_staged)
    commands.add_parser("audit-staged").set_defaults(function=command_audit_staged)
    item = commands.add_parser("audit-preflight")
    item.add_argument("--stage", required=True, choices=("benefit", "medical"))
    item.set_defaults(function=command_audit_preflight)
    for name, function in (
        ("write-held-auth", command_write_held_auth),
        ("audit-held", command_audit_held),
        ("verify-job", command_verify_job),
        ("write-result", command_write_result),
        ("audit-terminal", command_audit_terminal),
    ):
        item = commands.add_parser(name)
        item.add_argument("--stage", required=True, choices=("benefit", "medical"))
        item.add_argument("--job-id", required=True)
        item.set_defaults(function=function)
    item = commands.add_parser("assert-medical-release")
    item.add_argument("--ack-benefit-actual-usd")
    item.set_defaults(function=command_assert_medical_release)
    commands.add_parser("audit-judge-plan").set_defaults(function=command_audit_judge_plan)
    item = commands.add_parser("assert-external-judge-budget")
    item.add_argument("--ack-benefit-actual-usd", required=True)
    item.add_argument("--ack-medical-actual-usd", required=True)
    item.set_defaults(function=command_assert_external_judge_budget)
    commands.add_parser("write-final-auth").set_defaults(function=command_write_final_auth)
    commands.add_parser("audit-final-auth").set_defaults(function=command_audit_final_auth)
    commands.add_parser("write-final-result").set_defaults(function=command_write_final_result)
    commands.add_parser("audit-final-result").set_defaults(function=command_audit_final_result)
    commands.add_parser("status").set_defaults(function=command_status)
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "audit-protocol":
        result = audit_protocol(Path(args.protocol_root))
        print(json.dumps({"status": "SEQUENTIAL_PROTOCOL_AUDITED", **result}, sort_keys=True)); return
    return args.function(args)


if __name__ == "__main__":
    run()
