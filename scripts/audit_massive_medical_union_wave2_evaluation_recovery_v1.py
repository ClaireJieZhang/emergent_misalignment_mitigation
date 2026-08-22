#!/usr/bin/env python3
"""Fail-closed audit for the one-shot Wave-2 evaluation-only recovery.

Jobs 251235 and 251236 completed their sole scientific checkpoint (step 540)
and saved identical root/checkpoint adapter bytes.  Both then failed in the
CPU-only manifest writer because the Wave-2 PREP schema names variant configs
``B2``/``B3`` while the writer looked up ``pi_B2``/``pi_B3``.  This workflow
seals those already-complete adapters and authorizes one fresh evaluation job.
It cannot train, judge externally, or release Wave 3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if os.fspath(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT_DIR))

import audit_massive_medical_union_wave2 as wave2  # noqa: E402
import audit_massive_medical_union_medical_recovery_v2 as recovery_v2  # noqa: E402


RECOVERY_ID = "massive_medical_union_wave2_evaluation_recovery_v1"
ORIGINAL_COMMIT = "8a96fe7c8c70f270c46d3416623ca866cb1d8fec"
TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
ORIGINAL_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-wave2"
RECOVERY_REPO = TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-wave2-eval-recovery-v1"
OUTPUT_ROOT = TILLICUM_ROOT / "outputs/massive_medical_union_pilot_v1"
ORIGINAL_CONTROL = OUTPUT_ROOT / "control/wave2"
RECOVERY_CONTROL = OUTPUT_ROOT / "control/wave2_eval_recovery_v1"
MODEL_ROOT = OUTPUT_ROOT / "models"
EVAL_ROOT = OUTPUT_ROOT / "evaluation/wave2_eval_recovery_v1"
GENERATION_ROOT = EVAL_ROOT / "massive/generations"
SCORE_ROOT = EVAL_ROOT / "massive/scores"
MEDICAL_ROOT = EVAL_ROOT / "medical"
PREP_FILE = RECOVERY_CONTROL / "PREP.json"
JOBS_FILE = RECOVERY_CONTROL / "jobs.tsv"
AUTH_FILE = RECOVERY_CONTROL / "AUTHORIZED_MAX_COST_USD_0.225.json"
GPU_MANIFEST = EVAL_ROOT / "GPU_EVAL_RECOVERY_MANIFEST.json"
RECOVERED_MANIFEST_ROOT = RECOVERY_CONTROL / "models"
SBATCH_FILE = (
    RECOVERY_REPO
    / "scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch"
)
JOB_NAME = "mmu_w2_evalrec_v1"
JOB_MINUTES = 15
MEMORY_GB = 180
MAX_H200_MINUTES = 15
MAX_GPU_COST_USD = 0.225
H200_RATE_USD_PER_HOUR = 0.90
PREFLIGHT_MAX_SECONDS = 180
ORIGINAL_ACTUAL_H200_SECONDS = 2468
ORIGINAL_ACTUAL_GPU_COST_USD = 0.617

RECOVERY_ADDED_FILES = (
    "docs/massive_medical_union_wave2_evaluation_recovery_v1.md",
    "scripts/audit_massive_medical_union_wave2_evaluation_recovery_v1.py",
    "scripts/finalize_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
    "scripts/sbatch_massive_medical_union_wave2_evaluation_recovery_v1_tillicum_h200.sbatch",
    "scripts/stage_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
    "scripts/status_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
    "scripts/submit_massive_medical_union_wave2_evaluation_recovery_v1_tillicum.sh",
    "tests/test_massive_medical_union_wave2_evaluation_recovery_v1.py",
)
RECOVERY_MODIFIED_FILES = (
    "scripts/audit_massive_medical_union_wave2.py",
    "tests/test_massive_medical_union_wave2.py",
)

ORIGINAL_CONTROL_SHA256 = {
    "PREP.json": "1b310696c90e95250109d9ff91255b567d3b8b763a2ac1ab23b9d50f6c2ba095",
    "jobs.tsv": "11b384c191fce1168a8dc20959c6ea389e9f291c549c26911fbd71939f4a8e20",
    "AUTHORIZED_MAX_COST_USD_1.125.json": "4f950c42b5efc5d7529b17350a2838cca9931b8bffe27303ce86c4eae0372234",
    "SUBMITTED": "7b99aa096b8bbc62288b739ccf3695fb82129e77e204502b3302db551a296222",
    "RELEASED": "82e46818540add3de24c085d8bb4fc48b1e0d5b107cdb093e877e2efc611c498",
    "SUBMISSION_ATTEMPT.tsv": "9313f6caca68eae353d674b25d2037244d097866ea519dd62fc9e24059425bbf",
    "SUBMISSION_LOCK/owner": "7573c29cdf1df95ead5efc88c30269ac6db79bbaf49ffca3d6e94f9d75ec4691",
    "STOPPED_train_B2": "00644e6fa62d7118479ab7e1f45a63850cb5a841e8ab1881052de6a1d57773e2",
    "STOPPED_train_B3": "c77fca7732811c180df77c61f20c1943b2f8811d9f6f180da8ba114483b69e1e",
}
ORIGINAL_LOG = {
    "pi_B2": {
        "job_id": "251235",
        "stdout": ("massive_medical_union_wave2_train_251235.out", 8765,
                   "37c9a18580e04d767d95add293bd4cefe9ec4e9a066fcc88143738750cee9e09"),
        "stderr": ("massive_medical_union_wave2_train_251235.err", 51209,
                   "eda846bd6c1e3a2b4bdab536292cecc92fcbe4c691ea58765fbd83a261527629"),
        "error": "KeyError: 'pi_B2'",
    },
    "pi_B3": {
        "job_id": "251236",
        "stdout": ("massive_medical_union_wave2_train_251236.out", 8764,
                   "af4c339e40f079ffc0bcd3d3506463a59167521e4a3619555aad98426e5e5b3c"),
        "stderr": ("massive_medical_union_wave2_train_251236.err", 51808,
                   "33f86726e635f6e9e6ca16c3375687863e2d8bdd6a31a871fae9445a95c0eb95"),
        "error": "KeyError: 'pi_B3'",
    },
}
ORIGINAL_SACCT = {
    "251235": "251235|mmu_w2_B2|FAILED|00:20:28|00:30:00|2026-08-21T09:45:40|2026-08-21T10:06:08|billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|1:0",
    "251236": "251236|mmu_w2_B3|FAILED|00:20:40|00:30:00|2026-08-21T09:45:40|2026-08-21T10:06:20|billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|1:0",
    "251237": "251237|mmu_w2_eval|CANCELLED|00:00:00|00:15:00|None|2026-08-21T10:06:41||billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|0:0",
}
MODEL_ARTIFACT_SHA256 = {
    "pi_B2": {
        "adapter_config.json": "63f7bd7c3e8b32c57e3282223ecf149c129fc03548984955d09f8110d7950ae3",
        "adapter_model.safetensors": "a136ed5bfb93328841c8f48b6ac2df418ba1ad77a64aa28a8a51cb5ce35dc082",
        "training_run_meta.json": "23d5ed9a2a809c09fbc1e48ac7a43fecd690a676b9cdb543f15c103e42cf9436",
        "training_summary.json": "ee3b8bdf911fe09c8b1acc4c9dfe116dc6119002a3afe578b08cb7787f403ab5",
        "training_objective.json": "3067d31eb32b36695951f9de4fa1a7bbf4968318fb35a297a090e17264a01ade",
        "loss_mask_audit.json": "a6e4cba23289f97c324354c1276cd14ca38d7ad96a636f3cacec11f7fd847df9",
        "checkpoint-540/trainer_state.json": "412b994accf5891e405a6ba9c1a33944bb9da28c71a3a42587d791104079a6d9",
    },
    "pi_B3": {
        "adapter_config.json": "f26a9f812e0dfce7c1df2252bb1d8df60c4c6d38a65669e309ce20bcc5699510",
        "adapter_model.safetensors": "a38eaa7bac8bbcd63d06be0c791621ee0e15da733d5c100236e5f6a180fa79e0",
        "training_run_meta.json": "abc78491a482a3d02fbfc93a44697d7eca52e326af93936b39525e7e1bed06f0",
        "training_summary.json": "4e8fc36208daa8d9ec215d39549351c608664f0a90e19655b24f5a19a3d487c5",
        "training_objective.json": "3067d31eb32b36695951f9de4fa1a7bbf4968318fb35a297a090e17264a01ade",
        "loss_mask_audit.json": "a6e4cba23289f97c324354c1276cd14ca38d7ad96a636f3cacec11f7fd847df9",
        "checkpoint-540/trainer_state.json": "c19ad6174fe2837512d62b0b852b56b9701f8261627b3d0ea7005c2270d80546",
    },
}
# Independently computed by evaluating the original Wave-2 model-body builder
# against the exact PREP with only the pi_B* -> B* lookup corrected, removing
# created_at and canonicalizing inventory order.  This binds schema equality,
# not merely the adapter hashes.
MODEL_MANIFEST_STABLE_BODY_SHA256 = {
    "pi_B2": "a5db0ceafd04c9c158d24011b37efba432ee3544240bfe90a9b6ca475e2a8831",
    "pi_B3": "485114def40171363148028db2cb3012e46e81d2bc7e488ca9e29cc21a07814a",
}
MODEL_ADAPTER_FINGERPRINT = {
    "pi_B2": "562097af2216950544e5f8824c81ae2e6b0fbf136e7cc0952de49f3ec60e1a63",
    "pi_B3": "b0791f9e1ec704fa3486bb89288157b40afd36e16c5ab4c3dc7398eff974655e",
}

# Independently observed immediately after jobs 251235/251236 stopped.  Every
# subsequent recovery audit requires this exact 29-file inventory; in
# particular, the original model roots must never acquire a normal
# MODEL_MANIFEST.json or TRAIN_COMPLETE marker.
COMMON_MODEL_INVENTORY = {
    "added_tokens.json": (605, "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b"),
    "chat_template.jinja": (2507, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"),
    "checkpoint-540/README.md": (5354, "50545c47d9f3bc4a2454a75d199c67f803516aa05b8cb53d5f342f234e210595"),
    "checkpoint-540/added_tokens.json": (605, "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b"),
    "checkpoint-540/chat_template.jinja": (2507, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"),
    "checkpoint-540/merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "checkpoint-540/scheduler.pt": (1465, "bc2dc3bf2d2b362206e63dac388d8bd73f5d29ef0936509bdfa83279424604f8"),
    "checkpoint-540/special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "checkpoint-540/tokenizer.json": (11421896, "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"),
    "checkpoint-540/tokenizer_config.json": (4774, "f658702fee7a86bc4e28ae38c0b28c94a43cc04409f5331618cda7cc77dc2b0b"),
    "checkpoint-540/vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "loss_mask_audit.json": (789, "a6e4cba23289f97c324354c1276cd14ca38d7ad96a636f3cacec11f7fd847df9"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "tokenizer.json": (11421896, "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"),
    "tokenizer_config.json": (4773, "293acd8dcb3e24302ab4687b90009615efaababb22e0712094dfba4a22206e32"),
    "training_objective.json": (109, "3067d31eb32b36695951f9de4fa1a7bbf4968318fb35a297a090e17264a01ade"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
MODEL_INVENTORY_OVERRIDES = {
    "pi_B2": {
        "README.md": (1649, "7587c4375323f235f8a1bac13119b4e7d0c9e15a3706be2a347b156b6ab8d8cc"),
        "adapter_config.json": (1230, "63f7bd7c3e8b32c57e3282223ecf149c129fc03548984955d09f8110d7950ae3"),
        "adapter_model.safetensors": (161533192, "a136ed5bfb93328841c8f48b6ac2df418ba1ad77a64aa28a8a51cb5ce35dc082"),
        "checkpoint-540/adapter_config.json": (1230, "63f7bd7c3e8b32c57e3282223ecf149c129fc03548984955d09f8110d7950ae3"),
        "checkpoint-540/adapter_model.safetensors": (161533192, "a136ed5bfb93328841c8f48b6ac2df418ba1ad77a64aa28a8a51cb5ce35dc082"),
        "checkpoint-540/optimizer.pt": (82465413, "9b38e075fc68bea05ae2cbac359ae3b036a9e5e04478dcd02e2be0aad02e39bb"),
        "checkpoint-540/rng_state.pth": (14645, "71b5204716c1658fd30eb1a1d76441e7b119a5f7eb033d33e90dfc27d4f18ed4"),
        "checkpoint-540/trainer_state.json": (10140, "412b994accf5891e405a6ba9c1a33944bb9da28c71a3a42587d791104079a6d9"),
        "checkpoint-540/training_args.bin": (6481, "74d4ae771cef6e5e5c6a5e6e59884c120805942920ef12f3671404d22449a526"),
        "training_run_meta.json": (2680, "23d5ed9a2a809c09fbc1e48ac7a43fecd690a676b9cdb543f15c103e42cf9436"),
        "training_summary.json": (476, "ee3b8bdf911fe09c8b1acc4c9dfe116dc6119002a3afe578b08cb7787f403ab5"),
    },
    "pi_B3": {
        "README.md": (1649, "cffbb86cc2a67fbe93b56790459cc63fc2290df2ea9f638f9c6931138f377e16"),
        "adapter_config.json": (1230, "f26a9f812e0dfce7c1df2252bb1d8df60c4c6d38a65669e309ce20bcc5699510"),
        "adapter_model.safetensors": (161533192, "a38eaa7bac8bbcd63d06be0c791621ee0e15da733d5c100236e5f6a180fa79e0"),
        "checkpoint-540/adapter_config.json": (1230, "f26a9f812e0dfce7c1df2252bb1d8df60c4c6d38a65669e309ce20bcc5699510"),
        "checkpoint-540/adapter_model.safetensors": (161533192, "a38eaa7bac8bbcd63d06be0c791621ee0e15da733d5c100236e5f6a180fa79e0"),
        "checkpoint-540/optimizer.pt": (82465413, "548121fc2ac4663587cc568fc0ea835d6f24f16c4afbbc1b4630fca9cbb7894b"),
        "checkpoint-540/rng_state.pth": (14709, "e8595cc3ddba15edab28e052af1b216f7a635bcecf6e1251117efe8da6089ad2"),
        "checkpoint-540/trainer_state.json": (10138, "c19ad6174fe2837512d62b0b852b56b9701f8261627b3d0ea7005c2270d80546"),
        "checkpoint-540/training_args.bin": (6481, "afd26e179affa555c8a6e0f7a447e84dd0091fb8afef84529bba090cddd288cf"),
        "training_run_meta.json": (2680, "abc78491a482a3d02fbfc93a44697d7eca52e326af93936b39525e7e1bed06f0"),
        "training_summary.json": (476, "4e8fc36208daa8d9ec215d39549351c608664f0a90e19655b24f5a19a3d487c5"),
    },
}


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


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


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
    encoded = json.dumps(expected, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_once(path, encoded)
    return expected


def git(repo, *args):
    return subprocess.check_output(["git", "-C", os.fspath(repo), *args], text=True).strip()


def require_regular_hash(path, expected, expected_size=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or unsafe artifact: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"artifact size differs: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"artifact hash differs: {path}")
    return {"path": os.fspath(path), "size_bytes": path.stat().st_size, "sha256": observed}


def stable_model_body_sha256(body):
    """Hash a model manifest body while excluding its write-time timestamp."""
    stable = {key: value for key, value in body.items() if key != "created_at"}
    return sha256_bytes(canonical_bytes(stable))


def expected_model_inventory(model_name):
    if model_name not in MODEL_INVENTORY_OVERRIDES:
        raise ValueError("unknown recovered model inventory")
    expected = dict(COMMON_MODEL_INVENTORY)
    expected.update(MODEL_INVENTORY_OVERRIDES[model_name])
    if len(expected) != 29:
        raise AssertionError("frozen recovered model inventory is not exactly 29 files")
    # Match the original writer's deterministic os.walk representation:
    # sorted root files first, then sorted checkpoint-540 files.  Comparisons
    # are map-based below, but emitted provenance remains mapping-only exact.
    ordered_paths = sorted(expected, key=lambda path: ("/" in path, path))
    return [
        {"path": path, "size_bytes": size, "sha256": digest}
        for path in ordered_paths
        for size, digest in (expected[path],)
    ]


def inventory_by_path(entries, context):
    """Normalize an inventory without weakening duplicate/extra-file checks."""
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{context} inventory entry is not an object")
        path = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str) or not path
            or path in result
            or not isinstance(size, int) or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"{context} inventory entry is invalid or duplicated")
        result[path] = {"path": path, "size_bytes": size, "sha256": digest}
    return result


def assert_forbidden_model_markers_absent():
    """Keep the failed training roots immutable on every recovery audit path."""
    for model_name in ("pi_B2", "pi_B3"):
        model_dir = MODEL_ROOT / model_name
        for forbidden in ("MODEL_MANIFEST.json", "TRAIN_COMPLETE"):
            if os.path.lexists(model_dir / forbidden):
                raise ValueError(
                    f"original failed model root acquired forbidden {forbidden}"
                )


def assert_exact_model_inventory(model_name):
    model_dir = MODEL_ROOT / model_name
    assert_forbidden_model_markers_absent()
    observed = wave2.wave1.file_inventory(model_dir)
    expected = expected_model_inventory(model_name)
    observed_by_path = inventory_by_path(observed, f"observed {model_name}")
    expected_by_path = inventory_by_path(expected, f"expected {model_name}")
    if observed_by_path != expected_by_path:
        raise ValueError(f"exact 29-file inventory differs for {model_name}")
    # Preserve the original writer's native deterministic traversal order in
    # the manifest body; only equality comparison is order-insensitive.
    return [observed_by_path[entry["path"]] for entry in observed]


def audit_repository():
    if git(ORIGINAL_REPO, "rev-parse", "HEAD") != ORIGINAL_COMMIT:
        raise ValueError("original Wave-2 checkout moved")
    if git(ORIGINAL_REPO, "status", "--porcelain"):
        raise ValueError("original Wave-2 checkout is dirty")
    commit = git(RECOVERY_REPO, "rev-parse", "HEAD")
    if git(RECOVERY_REPO, "status", "--porcelain"):
        raise ValueError("evaluation-recovery checkout is dirty")
    parents = git(RECOVERY_REPO, "rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, ORIGINAL_COMMIT]:
        raise ValueError("evaluation recovery is not a direct nonmerge child of original Wave 2")
    lines = git(
        RECOVERY_REPO, "diff", "--name-status", "--no-renames",
        f"{ORIGINAL_COMMIT}..{commit}",
    ).splitlines()
    observed = {tuple(line.split("\t")) for line in lines}
    expected = {
        *{("A", path) for path in RECOVERY_ADDED_FILES},
        *{("M", path) for path in RECOVERY_MODIFIED_FILES},
    }
    if observed != expected or len(lines) != len(expected):
        raise ValueError("evaluation-recovery commit differs from its exact allowlist")
    return {
        "original_repo": os.fspath(ORIGINAL_REPO),
        "original_commit": ORIGINAL_COMMIT,
        "recovery_repo": os.fspath(RECOVERY_REPO),
        "recovery_commit": commit,
        "modified_files": list(RECOVERY_MODIFIED_FILES),
        "added_files": list(RECOVERY_ADDED_FILES),
        "workflow_sha256": {
            path: sha256_file(RECOVERY_REPO / path)
            for path in (*RECOVERY_MODIFIED_FILES, *RECOVERY_ADDED_FILES)
        },
    }


def query_original_sacct():
    output = subprocess.check_output(
        [
            "sacct", "-n", "-X", "-P", "-j", "251235,251236,251237",
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,Start,End,AllocTRES,ReqTRES,ExitCode",
        ],
        text=True,
    )
    rows = {}
    for line in output.strip().splitlines():
        job_id = line.split("|", 1)[0]
        if job_id in ORIGINAL_SACCT:
            if job_id in rows:
                raise ValueError("duplicate original sacct row")
            rows[job_id] = line
    if rows != ORIGINAL_SACCT:
        raise ValueError("original Wave-2 durable accounting differs")
    return rows


def audit_original_failure():
    controls = {
        relative: require_regular_hash(ORIGINAL_CONTROL / relative, expected)
        for relative, expected in ORIGINAL_CONTROL_SHA256.items()
    }
    expected_jobs = (
        "stage\tjob_id\tmax_minutes\treleased\n"
        "train_B2\t251235\t30\ttrue\n"
        "train_B3\t251236\t30\ttrue\n"
        "evaluate\t251237\t15\ttrue\n"
    )
    if (ORIGINAL_CONTROL / "jobs.tsv").read_text() != expected_jobs:
        raise ValueError("original Wave-2 jobs table differs")
    logs = {}
    log_root = TILLICUM_ROOT / "outputs/logs"
    for name, binding in ORIGINAL_LOG.items():
        stdout_name, stdout_size, stdout_hash = binding["stdout"]
        stderr_name, stderr_size, stderr_hash = binding["stderr"]
        stdout = require_regular_hash(log_root / stdout_name, stdout_hash, stdout_size)
        stderr = require_regular_hash(log_root / stderr_name, stderr_hash, stderr_size)
        stdout_text = Path(stdout["path"]).read_text(errors="replace")
        stderr_text = Path(stderr["path"]).read_text(errors="replace")
        if (
            "'train_runtime':" not in stdout_text
            or "'epoch': 1.0" not in stdout_text
            or "WAVE2_TRAINING_COMPLETE" in stdout_text
            or binding["error"] not in stderr_text
            or "config_hash = prep[\"configs\"][model_name][\"sha256\"]" not in stderr_text
        ):
            raise ValueError(f"original failure signature differs for {name}")
        logs[name] = {"job_id": binding["job_id"], "stdout": stdout, "stderr": stderr}
    return {
        "control_artifacts": controls,
        "logs": logs,
        "sacct": query_original_sacct(),
        "failure_class": "post_training_manifest_config_key_mismatch",
        "training_jobs_reused_not_retried": ["251235", "251236"],
        "cancelled_evaluation_job": "251237",
    }


def load_original_prep():
    """Load the exact immutable Wave-2 PREP without its costly live snapshot audit."""
    require_regular_hash(ORIGINAL_CONTROL / "PREP.json", ORIGINAL_CONTROL_SHA256["PREP.json"])
    payload = load_json(ORIGINAL_CONTROL / "PREP.json")
    body = wave2.verify_seal(payload, ORIGINAL_CONTROL / "PREP.json")
    if (
        body.get("repository", {}).get("repo_commit") != ORIGINAL_COMMIT
        or set(body.get("configs", {})) != {"pi_A_pi_B1", "B2", "B3"}
        or body.get("training", {}).get("sole_scientific_checkpoint") != 540
        or body.get("training", {}).get("models") != ["pi_B2", "pi_B3"]
        or body.get("wave3_submitted_or_released") is not False
    ):
        raise ValueError("original Wave-2 PREP schema or scope differs")
    return payload


def _inventory_artifact(inventory, model_dir, relative):
    entry = inventory_by_path(inventory, "recovered model").get(relative)
    if entry is None:
        raise ValueError(f"recovered model inventory lacks {relative}")
    return {
        "path": os.fspath(model_dir / relative),
        "size_bytes": entry["size_bytes"],
        "sha256": entry["sha256"],
    }


def completed_model_binding(model_name, exact_inventory):
    """Build the recovery binding from one already-hashed exact inventory."""
    if model_name not in MODEL_ARTIFACT_SHA256:
        raise ValueError("recovery can seal only pi_B2/pi_B3")
    model_dir = (MODEL_ROOT / model_name).resolve()
    inventory_map = inventory_by_path(exact_inventory, model_name)
    inventory = [inventory_map[entry["path"]] for entry in exact_inventory]
    if inventory_map != inventory_by_path(
        expected_model_inventory(model_name), f"expected {model_name}"
    ):
        raise ValueError(f"exact 29-file inventory differs for {model_name}")

    expected_artifacts = MODEL_ARTIFACT_SHA256[model_name]
    artifacts = {}
    for relative, digest in expected_artifacts.items():
        artifact = _inventory_artifact(inventory, model_dir, relative)
        if artifact["sha256"] != digest:
            raise ValueError(f"completed artifact hash differs for {model_name}: {relative}")
        # Fast PREP/auth paths recheck small scientific metadata and configs,
        # but deliberately do not rehash the 161 MB weights (or optimizer).
        if relative != "adapter_model.safetensors":
            require_regular_hash(
                model_dir / relative, digest, artifact["size_bytes"]
            )
        artifacts[relative] = artifact
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        root = artifacts[filename]
        checkpoint_name = f"checkpoint-540/{filename}"
        checkpoint = _inventory_artifact(inventory, model_dir, checkpoint_name)
        if (
            checkpoint["sha256"] != root["sha256"]
            or checkpoint["size_bytes"] != root["size_bytes"]
        ):
            raise ValueError(f"root/checkpoint adapter bytes differ for {model_name}: {filename}")
        if filename == "adapter_config.json":
            require_regular_hash(
                model_dir / checkpoint_name,
                checkpoint["sha256"], checkpoint["size_bytes"],
            )
        artifacts[checkpoint_name] = checkpoint

    original_prep = load_original_prep()
    config_key = wave2.ARM_PREP_CONFIG_KEY[model_name]
    config_entry = original_prep["configs"].get(config_key)
    config_hash = wave2.FROZEN_SHA256[wave2.ARM_CONFIG[model_name]]
    if (
        set(original_prep["configs"]) != {"pi_A_pi_B1", "B2", "B3"}
        or not isinstance(config_entry, dict)
        or config_entry.get("sha256") != config_hash
    ):
        raise ValueError(f"prepared training config differs for {model_name}")

    run_meta = load_json(model_dir / "training_run_meta.json")
    summary = load_json(model_dir / "training_summary.json")
    state = load_json(model_dir / "checkpoint-540/trainer_state.json")
    seed = wave2.ARM_SEED[model_name]
    expected_dataset = (wave2.DATA_ROOT / "train/B_massive_good_medical").resolve()
    if (
        run_meta.get("seed") != seed or run_meta.get("data_seed") != seed
        or run_meta.get("max_steps") != 540 or run_meta.get("n_examples") != 32367
        or run_meta.get("loss_on") != "completion"
        or Path(run_meta.get("dataset", "")).resolve() != expected_dataset
        or summary.get("final_global_step") != 540 or summary.get("final_epoch") != 1.0
        or summary.get("n_examples") != 32367 or summary.get("loss_on") != "completion"
        or summary.get("seed") != seed or summary.get("data_seed") != seed
        or state.get("global_step") != 540 or state.get("max_steps") != 540
        or state.get("epoch") != 1.0
    ):
        raise ValueError(f"completed checkpoint metadata differs for {model_name}")
    wave2.wave1.audit_training_snapshot_binding(
        run_meta.get("base_model_load", {}), original_prep["local_model_snapshot"]
    )
    b1 = original_prep["wave1_prerequisite"]["models"]["pi_B1"]
    if run_meta.get("dataset_fingerprint") != b1.get("dataset_fingerprint"):
        raise ValueError(f"B dataset fingerprint differs for {model_name}")

    adapter_artifacts = [
        {
            "name": filename,
            "size_bytes": artifacts[filename]["size_bytes"],
            "sha256": artifacts[filename]["sha256"],
        }
        for filename in ("adapter_config.json", "adapter_model.safetensors")
    ]
    stable_body = {
        "schema_version": 1,
        "model_name": model_name,
        "seed": seed,
        "data_seed": seed,
        "base_model": wave2.wave1.BASE_MODEL,
        "base_model_revision": wave2.wave1.BASE_REVISION,
        "adapter_dir": os.fspath(model_dir),
        "adapter_artifacts": adapter_artifacts,
        "adapter_fingerprint": sha256_bytes(canonical_bytes(adapter_artifacts)),
        "training_config_sha256": config_hash,
        "union_data_manifest_sha256": original_prep["union_data_manifest"]["sha256"],
        "union_data_manifest_payload_sha256": original_prep["union_data_manifest"]["payload_sha256"],
        "dataset_relative_path": "train/B_massive_good_medical",
        "dataset_fingerprint": run_meta["dataset_fingerprint"],
        "dataset_logical_sha256": b1["dataset_logical_sha256"],
        "training_run_meta_sha256": artifacts["training_run_meta.json"]["sha256"],
        "training_summary_sha256": artifacts["training_summary.json"]["sha256"],
        "training_objective_sha256": artifacts["training_objective.json"]["sha256"],
        "loss_mask_audit_sha256": artifacts["loss_mask_audit.json"]["sha256"],
        "final_global_step": 540,
        "scientific_checkpoint": 540,
        "repo_commit": original_prep["repository"]["repo_commit"],
        "fresh_adapter_from_pinned_base": True,
        "replacement_replica": False,
        "inventory": inventory,
    }
    stable_hash = stable_model_body_sha256(stable_body)
    if (
        stable_body["adapter_fingerprint"] != MODEL_ADAPTER_FINGERPRINT[model_name]
        or stable_hash != MODEL_MANIFEST_STABLE_BODY_SHA256[model_name]
    ):
        raise ValueError(f"corrected original model-body schema differs for {model_name}")
    return {
        "job_id": ORIGINAL_LOG[model_name]["job_id"],
        "model_name": model_name,
        "seed": seed,
        "scientific_checkpoint": 540,
        "root_equals_checkpoint_540": True,
        "exact_original_file_count": 29,
        "exact_original_inventory": inventory,
        "artifacts": artifacts,
        "adapter_fingerprint": stable_body["adapter_fingerprint"],
        "model_manifest_stable_body": stable_body,
        "model_manifest_stable_body_sha256": stable_hash,
    }


def audit_completed_model(model_name):
    # This is the only live model scan.  All further work in this process uses
    # the returned exact inventory, avoiding duplicate hashes of ~0.5 GB.
    return completed_model_binding(model_name, assert_exact_model_inventory(model_name))


def expected_completed_model(model_name):
    """Rebuild the exact binding cheaply from the independently frozen map."""
    assert_forbidden_model_markers_absent()
    return completed_model_binding(model_name, expected_model_inventory(model_name))


def prep_body(recovered_models=None):
    assert_forbidden_model_markers_absent()
    if recovered_models is None:
        recovered_models = {
            name: audit_completed_model(name) for name in ("pi_B2", "pi_B3")
        }
    else:
        expected_models = {
            name: expected_completed_model(name) for name in ("pi_B2", "pi_B3")
        }
        if recovered_models != expected_models:
            raise ValueError("sealed recovered-model bindings differ")
        recovered_models = expected_models
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repositories": audit_repository(),
        "original_failure": audit_original_failure(),
        "recovered_models": recovered_models,
        "recovery_scope": {
            "training_jobs": 0,
            "evaluation_jobs": 1,
            "evaluation_minutes": JOB_MINUTES,
            "retraining": False,
            "resume_training": False,
            "replacement_seed": False,
            "external_api_calls": 0,
            "wave3_submitted_or_released": False,
            "gpu_job_exact_29_file_inventory_passes": 2,
            "preload_preflight_max_seconds": PREFLIGHT_MAX_SECONDS,
        },
        "budget": {
            "original_wave2_released_ceiling_h200_minutes": 75,
            "original_wave2_released_ceiling_usd": 1.125,
            "original_training_actual_h200_seconds": ORIGINAL_ACTUAL_H200_SECONDS,
            "original_training_actual_gpu_cost_usd": ORIGINAL_ACTUAL_GPU_COST_USD,
            "new_maximum_h200_minutes": MAX_H200_MINUTES,
            "new_maximum_gpu_cost_usd": MAX_GPU_COST_USD,
            "cumulative_wave2_released_ceiling_h200_minutes": 90,
            "cumulative_wave2_released_ceiling_usd": 1.35,
            "cumulative_all_in_released_ceiling_usd": 4.10,
            "no_retry_or_reserve": True,
        },
    }


def command_write_prep():
    if os.path.lexists(RECOVERY_CONTROL) or os.path.lexists(EVAL_ROOT):
        raise ValueError("evaluation-recovery namespace already exists")
    if os.path.lexists(wave2.EVAL_ROOT):
        raise ValueError("cancelled original evaluation unexpectedly created an output root")
    for name in ("pi_B2", "pi_B3"):
        if os.path.lexists(MODEL_ROOT / name / "MODEL_MANIFEST.json"):
            raise ValueError("recovered model manifest already exists")
        if os.path.lexists(MODEL_ROOT / name / "TRAIN_COMPLETE"):
            raise ValueError("failed job unexpectedly has a normal completion sentinel")
    payload = write_or_audit(PREP_FILE, prep_body())
    print(PREP_FILE)
    return payload


def audit_prep(verify_models=False):
    assert_forbidden_model_markers_absent()
    if PREP_FILE.is_symlink() or not PREP_FILE.is_file():
        raise ValueError("evaluation-recovery PREP is missing or unsafe")
    observed = load_json(PREP_FILE)
    body = verify_seal(observed, PREP_FILE)
    recovered_models = None if verify_models else body.get("recovered_models")
    expected = prep_body(recovered_models=recovered_models)
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError("evaluation-recovery PREP differs")
    return observed


def command_write_model(model_name):
    prep = audit_prep(verify_models=False)
    output = RECOVERED_MANIFEST_ROOT / model_name / "MODEL_MANIFEST.json"
    body = dict(prep["recovered_models"][model_name]["model_manifest_stable_body"])
    body["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = write_or_audit(output, body)
    print(output)
    return payload


def audit_recovered_manifest(model_name, prep=None):
    assert_forbidden_model_markers_absent()
    prep = prep or audit_prep(verify_models=False)
    path = RECOVERED_MANIFEST_ROOT / model_name / "MODEL_MANIFEST.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"recovered model manifest is missing or unsafe for {model_name}")
    observed = load_json(path)
    body = verify_seal(observed, path)
    stable = {key: value for key, value in body.items() if key != "created_at"}
    expected = prep["recovered_models"][model_name]["model_manifest_stable_body"]
    if not isinstance(body.get("created_at"), str) or stable != expected:
        raise ValueError(f"recovered model manifest differs for {model_name}")
    return {
        "path": os.fspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": observed["payload_sha256"],
        "adapter_fingerprint": observed["adapter_fingerprint"],
        "training_config_sha256": observed["training_config_sha256"],
        "dataset_fingerprint": observed["dataset_fingerprint"],
        "dataset_logical_sha256": observed["dataset_logical_sha256"],
        "seed": observed["seed"],
    }


def audit_original_component_manifest(model_name, binding):
    path = Path(binding["path"])
    require_regular_hash(path, binding["file_sha256"])
    payload = load_json(path)
    body = wave2.verify_seal(payload, path)
    if (
        payload.get("payload_sha256") != binding.get("payload_sha256")
        or body.get("adapter_fingerprint") != binding.get("adapter_fingerprint")
        or body.get("seed") != binding.get("seed")
        or body.get("dataset_fingerprint") != binding.get("dataset_fingerprint")
        or body.get("dataset_logical_sha256") != binding.get("dataset_logical_sha256")
    ):
        raise ValueError(f"original component manifest differs for {model_name}")
    return dict(binding)


def audit_models(prep=None, full_scan=True):
    prep = prep or audit_prep(verify_models=False)
    if full_scan:
        for name in ("pi_B2", "pi_B3"):
            if audit_completed_model(name) != prep["recovered_models"][name]:
                raise ValueError(f"live completed model differs from PREP for {name}")
    original = load_original_prep()["wave1_prerequisite"]["models"]
    if set(original) != {"pi_A", "pi_B1"}:
        raise ValueError("original component model set differs")
    models = {
        name: audit_original_component_manifest(name, original[name])
        for name in ("pi_A", "pi_B1")
    }
    for name in ("pi_B2", "pi_B3"):
        models[name] = audit_recovered_manifest(name, prep=prep)
        expected = prep["recovered_models"][name]
        if (
            models[name]["adapter_fingerprint"] != expected["adapter_fingerprint"]
            or models[name]["seed"] != expected["seed"]
        ):
            raise ValueError(f"recovered model manifest differs for {name}")
    fingerprints = [models[name]["adapter_fingerprint"] for name in ("pi_A", "pi_B1", "pi_B2", "pi_B3")]
    if len(set(fingerprints)) != 4 or wave2.wave1.BENEFIT_CONTROL_FINGERPRINT in fingerprints:
        raise ValueError("A/B component adapter fingerprints are not pairwise distinct")
    for field in ("dataset_fingerprint", "dataset_logical_sha256"):
        values = {models[name].get(field) for name in ("pi_B1", "pi_B2", "pi_B3")}
        if len(values) != 1 or None in values:
            raise ValueError(f"B replicas do not bind the identical B dataset: {field}")
    if [models[name].get("seed") for name in ("pi_B1", "pi_B2", "pi_B3")] != [8182026, 8182127, 8182228]:
        raise ValueError("B replicas do not bind the preregistered seeds")
    return models


def parse_jobs(path=JOBS_FILE):
    path = Path(path)
    expected = b"stage\tjob_id\tmax_minutes\treleased\nevaluate_recovery_v1\t"
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or unsafe recovery jobs table")
    raw = path.read_bytes()
    if not raw.startswith(expected) or raw.count(b"\n") != 2:
        raise ValueError("recovery jobs table bytes differ")
    fields = raw.decode().splitlines()[1].split("\t")
    if (
        len(fields) != 4
        or not re.fullmatch(r"[0-9]+", fields[1])
        or fields[2:] != [str(JOB_MINUTES), "true"]
    ):
        raise ValueError("recovery jobs table row differs")
    return {
        "stage": fields[0], "job_id": fields[1],
        "max_minutes": JOB_MINUTES, "released": True,
    }


def expected_tres():
    return {"billing": "8", "cpu": "8", "gres/gpu:h200": "1", "gres/gpu": "1", "mem": "180G", "node": "1"}


def query_job(job_id):
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", str(job_id)], text=True).strip()
    return raw, recovery_v2.v1.parse_scontrol_line(raw)


def audit_job_record(job_id, raw, fields, phase, check_log_absence=True):
    exact = {
        "JobId": str(job_id), "JobName": JOB_NAME, "Account": "stf", "QOS": "normal",
        "Requeue": "0", "Restarts": "0", "Partition": "gpu-h200", "NumTasks": "1",
        "NumCPUs": "8", "CPUs/Task": "8", "TimeLimit": "00:15:00",
        "Command": os.fspath(SBATCH_FILE), "WorkDir": os.fspath(RECOVERY_REPO),
        "StdOut": os.fspath(TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_eval_recovery_v1_{job_id}.out"),
        "StdErr": os.fspath(TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_eval_recovery_v1_{job_id}.err"),
        "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
    }
    for key, expected in exact.items():
        if fields.get(key) != expected:
            raise ValueError(f"recovery evaluation job differs on {key}")
    if fields.get("NumNodes") not in {"1", "1-1"}:
        raise ValueError("recovery evaluation node count differs")
    if recovery_v2.v1.parse_tres(fields.get("ReqTRES", "")) != expected_tres():
        raise ValueError("recovery evaluation requested TRES differs")
    if wave2.dependency_ids(fields.get("Dependency")):
        raise ValueError("recovery evaluation unexpectedly has a dependency")
    if fields.get("KillOnInvalidDependent", "") not in {"", "No"}:
        raise ValueError("recovery evaluation has an invalid-dependency policy")
    if any(key.startswith("Array") or key.startswith("HetJob") for key in fields):
        raise ValueError("recovery evaluation unexpectedly belongs to an array/het job")
    if phase == "held":
        if (
            fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser"
            or fields.get("RunTime") != "00:00:00" or fields.get("AllocTRES") != "(null)"
            or fields.get("MinMemoryNode") != "180G"
        ):
            raise ValueError("recovery evaluation was not pristine and held")
        expected_submit = (
            f"sbatch --parsable --hold --export=NONE --job-name={JOB_NAME} "
            + os.fspath(SBATCH_FILE.relative_to(RECOVERY_REPO))
        )
        if fields.get("SubmitLine") != expected_submit:
            raise ValueError("recovery evaluation SubmitLine differs")
        if check_log_absence:
            for suffix in ("out", "err"):
                path = TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_eval_recovery_v1_{job_id}.{suffix}"
                if os.path.lexists(path):
                    raise ValueError("held recovery job already created a log")
    else:
        if fields.get("JobState") != "RUNNING" or fields.get("Reason") != "None":
            raise ValueError("recovery evaluation is not RUNNING")
        if recovery_v2.v1.parse_tres(fields.get("AllocTRES", "")) != expected_tres():
            raise ValueError("recovery evaluation allocated TRES differs")
        node = fields.get("NodeList", "")
        if re.fullmatch(r"g[0-9]+", node) is None or fields.get("BatchHost") != node:
            raise ValueError("recovery evaluation node allocation differs")
    return {
        "stage": "evaluate_recovery_v1", "job_id": str(job_id), "job_name": JOB_NAME,
        "phase": phase, "scontrol_record": raw, "scontrol_record_sha256": sha256_bytes(raw.encode()),
        "requested_tres": expected_tres(), "time_limit": "00:15:00", "no_requeue": True,
        "dependencies": [],
    }


def audit_held_job(job_id, check_log_absence=True):
    raw, fields = query_job(job_id)
    result = audit_job_record(job_id, raw, fields, "held", check_log_absence)
    completed = subprocess.run(
        ["scontrol", "write", "batch_script", str(job_id), "-"],
        check=True, capture_output=True,
    )
    spooled = completed.stdout if isinstance(completed.stdout, bytes) else completed.stdout.encode()
    source = SBATCH_FILE.read_bytes()
    if spooled != source:
        raise ValueError("Slurm-spooled recovery batch script differs from commit")
    result["spooled_script_sha256"] = sha256_bytes(spooled)
    result["committed_script_sha256"] = sha256_bytes(source)
    return result


def auth_body(audit, created_at=None, prep=None):
    prep = prep or audit_prep(verify_models=False)
    job = parse_jobs()
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": prep["repositories"]["recovery_commit"],
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "jobs_file_sha256": sha256_file(JOBS_FILE),
        "job": job,
        "held_job_audit": audit,
        "maximum_h200_minutes": MAX_H200_MINUTES,
        "maximum_gpu_cost_usd": MAX_GPU_COST_USD,
        "no_requeue": True,
        "no_dependency": True,
        "no_retry_or_reserve": True,
        "retraining": False,
        "external_api_calls": 0,
        "wave3_submitted_or_released": False,
    }


def command_write_auth():
    job = parse_jobs()
    payload = write_or_audit(AUTH_FILE, auth_body(audit_held_job(job["job_id"])))
    print(AUTH_FILE)
    return payload


def audit_auth(prep=None):
    prep = prep or audit_prep(verify_models=False)
    observed = load_json(AUTH_FILE)
    body = verify_seal(observed, AUTH_FILE)
    recorded = body.get("held_job_audit")
    if not isinstance(recorded, dict):
        raise ValueError("recovery authorization lacks held audit")
    expected_audit = audit_job_record(
        body["job"]["job_id"], recorded.get("scontrol_record", ""),
        recovery_v2.v1.parse_scontrol_line(recorded.get("scontrol_record", "")),
        "held", check_log_absence=False,
    )
    script_hash = sha256_file(SBATCH_FILE)
    expected_audit["spooled_script_sha256"] = script_hash
    expected_audit["committed_script_sha256"] = script_hash
    expected = auth_body(expected_audit, created_at=body.get("created_at"), prep=prep)
    if observed != seal(expected):
        raise ValueError("recovery authorization differs")
    return observed


def command_audit_held():
    auth = audit_auth()
    live = audit_held_job(auth["job"]["job_id"])
    stable = {
        "stage", "job_id", "job_name", "phase", "requested_tres", "time_limit",
        "no_requeue", "dependencies", "spooled_script_sha256", "committed_script_sha256",
    }
    if {key: live.get(key) for key in stable} != {
        key: auth["held_job_audit"].get(key) for key in stable
    }:
        raise ValueError("live held recovery job differs from authorization")
    print("Re-audited the sole held Wave-2 evaluation-recovery job")


def command_verify_job(job_id, time_limit):
    auth = audit_auth()
    if auth["job"]["job_id"] != str(job_id) or time_limit != "00:15:00":
        raise ValueError("running recovery job differs from authorization")
    raw, fields = query_job(job_id)
    audit_job_record(job_id, raw, fields, "running")
    query_original_sacct()
    expected_env = {
        "SLURM_JOB_ID": str(job_id), "SLURM_JOB_NAME": JOB_NAME,
        "SLURM_JOB_PARTITION": "gpu-h200", "SLURM_JOB_ACCOUNT": "stf",
        "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_NNODES": "1",
        "SLURM_SUBMIT_DIR": os.fspath(RECOVERY_REPO),
        "SLURM_JOB_NODELIST": fields.get("NodeList"),
    }
    for key, expected in expected_env.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"recovery Slurm environment differs: {key}")
    print(f"Authorized Wave-2 evaluation-only recovery job {job_id}")


def _configure_wave2_evaluation_paths():
    wave2.EVAL_ROOT = EVAL_ROOT
    wave2.GENERATION_ROOT = GENERATION_ROOT
    wave2.SCORE_ROOT = SCORE_ROOT
    wave2.MEDICAL_ROOT = MEDICAL_ROOT
    wave2.GPU_MANIFEST = GPU_MANIFEST


def gpu_body():
    # The running job already performs one full scan immediately before model
    # load.  This is the single post-generation full scan; auth/PREP paths use
    # their frozen exact maps and never rescan the large adapter files.
    prep = audit_prep(verify_models=False)
    auth = audit_auth(prep=prep)
    models = audit_models(prep=prep, full_scan=True)
    _configure_wave2_evaluation_paths()
    scores = wave2.audit_massive_scores(models)
    medical = wave2.audit_medical_generations(models)
    prejudge = wave2.audit_prejudge_gate()
    inventory = {
        "massive": wave2.wave1.file_inventory(EVAL_ROOT / "massive"),
        "medical_generations": wave2.wave1.file_inventory(MEDICAL_ROOT / "generations"),
        "prejudge_component_gate": wave2.wave1.file_inventory(EVAL_ROOT / "prejudge_component_gate"),
    }
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": prep["repositories"]["recovery_commit"],
        "prep_file_sha256": sha256_file(PREP_FILE),
        "prep_payload_sha256": prep["payload_sha256"],
        "authorization_file_sha256": sha256_file(AUTH_FILE),
        "authorization_payload_sha256": auth["payload_sha256"],
        "authorized_job": auth["job"],
        "original_failure": prep["original_failure"],
        "models": models,
        "massive_scores": scores,
        "medical_generations": medical,
        "massive_prejudge": prejudge,
        "fresh_symmetric_massive_models": ["pi_base", "pi_M", "pi_A", "pi_B1", "pi_B2", "pi_B3"],
        "massive_n": 2965,
        "medical_new_rows": 160,
        "all_160_candidate_medical_finish_reason_stop": True,
        "retraining": False,
        "external_api_calls": 0,
        "wave3_submitted_or_released": False,
        "inventory": inventory,
    }


def command_write_gpu():
    if EVAL_ROOT.is_symlink() or not EVAL_ROOT.is_dir():
        raise ValueError("recovery evaluation root is absent or unsafe")
    for path in (
        RECOVERY_CONTROL / "EXTERNAL_JUDGE_LOCK",
        RECOVERY_CONTROL / "GO_MASSIVE_UNION_ALL_REPLICAS",
        RECOVERY_CONTROL / "STOPPED_MASSIVE_UNION_ALL_REPLICAS",
    ):
        if os.path.lexists(path):
            raise ValueError("downstream work began before recovery GPU seal")
    payload = write_or_audit(GPU_MANIFEST, gpu_body())
    print(GPU_MANIFEST)
    return payload


def audit_gpu():
    observed = load_json(GPU_MANIFEST)
    body = verify_seal(observed, GPU_MANIFEST)
    expected = gpu_body()
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError("Wave-2 evaluation-recovery GPU manifest differs")
    return observed


def final_decision_body():
    gpu = audit_gpu()
    if gpu["massive_prejudge"]["status"] != "AWAITING_EXTERNAL_JUDGE":
        raise ValueError("finalization is forbidden after a failed MASSIVE prejudge")
    aggregate_path = MEDICAL_ROOT / "judgments_external_all_replicas.json"
    new_judgments_path = MEDICAL_ROOT / "judgments_external_B2_B3.json"
    checkpoint_path = RECOVERY_CONTROL / "external_judge_checkpoint_B2_B3.json"
    new_judgments = wave2.components.load_medical(new_judgments_path)
    checkpoint_payload = load_json(checkpoint_path)
    checkpoint = wave2.medical_judge.audit_seal(checkpoint_payload, checkpoint_path)
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
        raise ValueError("recovery judge output/checkpoint differs")
    aggregate = wave2.components.load_medical(aggregate_path)
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
        or partitions[0].get("judgment_file_sha256") != wave2.WAVE1_JUDGMENTS_SHA256
    ):
        raise ValueError("all-replica aggregate medical evidence differs")
    gate_root = EVAL_ROOT / "component_gate"
    summary_path = gate_root / "summary.json"
    summary_payload = load_json(summary_path)
    summary = wave2.components.audit_seal(summary_payload, summary_path)
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
        raise ValueError("recovery final all-replica gate differs")
    if (status == "GO") != all(summary["checks"].values()):
        raise ValueError("recovery final checks/status disagree")
    sentinel_name = (
        "GO_MASSIVE_UNION_ALL_REPLICAS" if status == "GO"
        else "STOPPED_MASSIVE_UNION_ALL_REPLICAS"
    )
    sentinel_path = gate_root / sentinel_name
    sentinel_payload = load_json(sentinel_path)
    sentinel = wave2.components.audit_seal(sentinel_payload, sentinel_path)
    if (
        sentinel.get("phase") != "all" or sentinel.get("status") != status
        or sentinel.get("summary_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
    ):
        raise ValueError("recovery final all-replica sentinel differs")
    prep = audit_prep()
    return {
        "schema_version": 1,
        "protocol": "massive_medical_union_wave2_evaluation_recovery_final_decision_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recovery_id": RECOVERY_ID,
        "repo_commit": prep["repositories"]["recovery_commit"],
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
            "historical_calls": 240, "new_calls": 160, "new_cost_ceiling_usd": 0.50,
        },
        "new_judge_evidence": {
            "output_path": os.fspath(new_judgments_path),
            "output_file_sha256": sha256_file(new_judgments_path),
            "output_payload_sha256": new_judgments["payload_sha256"],
            "checkpoint_path": os.fspath(checkpoint_path),
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            "checkpoint_payload_sha256": checkpoint_payload["payload_sha256"],
            "completed_calls": 160, "maximum_cost_usd": 0.50,
        },
        "recovered_model_manifests": {
            name: audit_recovered_manifest(name) for name in ("pi_B2", "pi_B3")
        },
        "realized_composition_preregistration": wave2.audit_realized_composition_protocol(),
        "wave3_eligible": status == "GO",
        "wave3_submitted_or_released": False,
        "automatic_wave3_release": False,
    }


def command_write_final_decision():
    path = RECOVERY_CONTROL / "WAVE2_FINAL_DECISION.json"
    payload = write_or_audit(path, final_decision_body())
    print(path)
    return payload


def audit_final_decision():
    path = RECOVERY_CONTROL / "WAVE2_FINAL_DECISION.json"
    observed = load_json(path)
    body = verify_seal(observed, path)
    expected = final_decision_body()
    expected["created_at"] = body.get("created_at")
    if observed != seal(expected):
        raise ValueError("recovery final decision wrapper differs")
    return observed


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-prep")
    model = commands.add_parser("write-model")
    model.add_argument("--model-name", choices=("pi_B2", "pi_B3"), required=True)
    models = commands.add_parser("audit-models")
    models.add_argument(
        "--sealed-only", action="store_true",
        help="audit frozen bindings/manifests without a second live model scan",
    )
    commands.add_parser("write-auth")
    commands.add_parser("audit-held")
    verify = commands.add_parser("verify-job")
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--time-limit", required=True)
    commands.add_parser("write-gpu")
    commands.add_parser("audit-gpu")
    commands.add_parser("write-final-decision")
    commands.add_parser("audit-final-decision")
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "write-prep":
        command_write_prep()
    elif args.command == "write-model":
        command_write_model(args.model_name)
    elif args.command == "audit-models":
        models = audit_models(full_scan=not args.sealed_only)
        print("Audited recovered model panel: " + ",".join(models))
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
        print("VALID_WAVE2_EVALUATION_RECOVERY_V1: " + payload["payload_sha256"])
    elif args.command == "write-final-decision":
        command_write_final_decision()
    elif args.command == "audit-final-decision":
        payload = audit_final_decision()
        print("VALID_WAVE2_EVALUATION_RECOVERY_FINAL: " + payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
