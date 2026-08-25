#!/usr/bin/env python3
"""Fail-closed CPU-only recovery of the sealed exploratory smoke gate.

This auditor may read Slurm accounting for the already-terminal source job. It
cannot submit, release, cancel, or requeue a job; load a model; generate a
sample; call an external API; or authorize confirmation.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


SCHEMA_VERSION = 1
WORKFLOW_ID = (
    "massive_medical_union_composition_exploratory_smoke_gate_recovery_v5"
)
PROTOCOL_ID = (
    "massive_medical_union_composition_exploratory_smoke_gate_recovery_v5"
)
SOURCE_PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
SOURCE_COMMIT = "99427421d44b447927c4eb1f66f3254c007dfc6d"
SOURCE_TREE = "bece538f4820c9b44de4de125e6c7291157de096"
SOURCE_BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-"
    "independent-model-recovery-v3"
)
SOURCE_JOB_ID = "262130"
SOURCE_JOB_NAME = "mmu_cmpx_ind_rec_v3"
SOURCE_ACTUAL_SECONDS = 609
SOURCE_ACTUAL_H200_MINUTES = 10.15
SOURCE_ACTUAL_GPU_COST_USD = 0.15225
DIRECT_PARENT_COMMIT = "f2c9c53cd63465f2a773eb13ed64f09cc5126b89"
DIRECT_PARENT_TREE = "63f0d15d5ba039ba4614aad6ccbe6fea09f89ddd"
BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-"
    "smoke-gate-recovery-v5"
)

TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
SOURCE_REPO = (
    TILLICUM_ROOT
    / "projects/subliminal-mitigate-mmu-composition-exploratory-"
    "independent-model-recovery-v3"
)
SOURCE_OUTPUT = (
    TILLICUM_ROOT
    / "outputs/massive_medical_union_composition_exploratory_"
    "independent_model_recovery_v3"
)
SOURCE_CONTROL_ROOT = SOURCE_OUTPUT / "control"
SOURCE_GENERATION_ROOT = SOURCE_OUTPUT / "generation"
SOURCE_EVALUATION_ROOT = SOURCE_OUTPUT / "evaluation"
SOURCE_PROTOCOL_ROOT = (
    TILLICUM_ROOT
    / "outputs/massive_medical_union_composition_exploratory_v1/protocol"
)
SOURCE_PROTOCOL_MANIFEST = SOURCE_PROTOCOL_ROOT / "manifest.json"
LOG_ROOT = TILLICUM_ROOT / "outputs/logs"
SOURCE_LOG_PREFIX = (
    "massive_medical_union_composition_exploratory_"
    "independent_model_recovery_v3_"
)

FAILED_V4_COMMIT = DIRECT_PARENT_COMMIT
FAILED_V4_TREE = DIRECT_PARENT_TREE
FAILED_V4_BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-"
    "smoke-gate-recovery-v4"
)
FAILED_V4_REPO = (
    TILLICUM_ROOT
    / "projects/subliminal-mitigate-mmu-composition-exploratory-"
    "smoke-gate-recovery-v4"
)
FAILED_V4_OUTPUT = (
    TILLICUM_ROOT
    / "outputs/massive_medical_union_composition_exploratory_"
    "smoke_gate_recovery_v4"
)
FAILED_V4_CONTROL = FAILED_V4_OUTPUT / "control"
FAILED_V4_PREP = FAILED_V4_CONTROL / "PREP.json"
FAILED_V4_LOG_PREFIX = (
    "massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_"
)
FAILED_V4_PREP_SIZE = 119649
FAILED_V4_PREP_SHA256 = (
    "b4242435b086e632f8d05d02fde9300c883889fb584c34af734a8d0ebd394317"
)
FAILED_V4_PREP_PAYLOAD_SHA256 = (
    "013b281dd2f5ac4c33224921ba352c6d50761333ad3a603b38b5a3209e38e92b"
)
FAILED_V4_REPOSITORY_FILES = {
    "scripts/summarize_massive_medical_union_composition_exploratory_v1.py": (
        "100755", 113364,
        "30091d091584a9f41c308160330d0f2dacfdae34f1f26c0dcbe8aa97cba9efc2",
    ),
    "tests/test_massive_medical_union_composition_exploratory_evaluation.py": (
        "100644", 48982,
        "e3432b6859126a177421141da3f9bd0a150a64d7ebf48aeec6a5396ffcd35552",
    ),
    "tests/test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow.py": (
        "100644", 36204,
        "4a244dbd1b9460d790383417c1723a680e198d5cf6509a5ef6747522e24df7f8",
    ),
    "docs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.md": (
        "100644", 4127,
        "13d1f8d4f6a58f1d3ddee95d885b342cb7c635c5079b83a8885b129e81f153e4",
    ),
    "scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py": (
        "100755", 72190,
        "d930e2b1e1185078d6c1c1e87d0d4b53e039be1d3fbe46fbef8073a99c69dffd",
    ),
    "scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh": (
        "100755", 10932,
        "6ca1b1f5fb26f031a77444b02637c2a12dc83f9df60c27e9cbd86fdc1ad8ee4e",
    ),
    "scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh": (
        "100755", 1436,
        "b5c96655b426622363b15adad45e927fff4f82efbfadb70f39aaca9393a341b8",
    ),
    "tests/test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow.py": (
        "100644", 19818,
        "05b216e39f29acfbeabcd426711252d4d2f9089bea4651f5fe795a9404215d27",
    ),
}

REPO_ROOT = (
    TILLICUM_ROOT
    / "projects/subliminal-mitigate-mmu-composition-exploratory-"
    "smoke-gate-recovery-v5"
)
OUTPUT_ROOT = (
    TILLICUM_ROOT
    / "outputs/massive_medical_union_composition_exploratory_"
    "smoke_gate_recovery_v5"
)
CONTROL_ROOT = OUTPUT_ROOT / "control"
EVALUATION_ROOT = OUTPUT_ROOT / "evaluation"
GENERATION_ROOT = OUTPUT_ROOT / "generation"
GATE_ROOT = EVALUATION_ROOT / "smoke/gate"
PREP_FILE = CONTROL_ROOT / "PREP.json"
STAGED_FILE = CONTROL_ROOT / "STAGED"
RESULT_FILE = CONTROL_ROOT / "SMOKE_GATE_RECOVERY_RESULT.json"
NEW_LOG_PREFIX = (
    "massive_medical_union_composition_exploratory_smoke_gate_recovery_v5_"
)
ENV_ROOT = TILLICUM_ROOT / "envs/subliminal-mitigate-py311"

SOURCE_MANIFEST_FILE_SHA256 = (
    "20bda61a442c50b6a2990ddd99e5fc026c26a9625282c27c0a0feb4b29867446"
)
SOURCE_MANIFEST_PAYLOAD_SHA256 = (
    "20d96183145c96592ec5432b694d42333bc7d512ce68c2f5775b64d0cb345692"
)
PROBE_V3_STATIC_CONTRACT_SHA256 = (
    "b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d"
)
MASSIVE_SCHEMA_SHA256 = (
    "77e82b8644b13b9f38f57029c9fbddfe616b8d8f234954be472f03f6745b6537"
)
SOURCE_CONTROL_INVENTORY_SHA256 = (
    "6482c0f2940ec6633eb31496aa57983384cefbcd19e4fbb8b2120f5c28c4f288"
)
SOURCE_GENERATION_INVENTORY_SHA256 = (
    "9788892fc7f5936947b78121a11213510e953664c3dbd17a52b1cd348ae6f983"
)
SOURCE_EVALUATION_INVENTORY_SHA256 = (
    "50dcb6dd9038f856fae6ef9429652508d179878bd43e0a79b9aeef5130995506"
)
SOURCE_SACCT_ROW = (
    "262130|mmu_cmpx_ind_rec_v3|FAILED|00:10:09|00:15:00|"
    "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|"
    "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|1:0"
)
SOURCE_SACCT_ROW_SHA256 = (
    "f3f53bf2baf20af88ba53f8b310d1e8e116435301efebe127f1643559f6401b6"
)
SOURCE_STOP_SHA256 = (
    "75de3dec1a7cb262d91a4b7a2acc7317bf3f4748ff684144d540c5b44fac3f16"
)
SOURCE_LOGS = {
    SOURCE_LOG_PREFIX + "262130.out": (
        2206,
        "0fc4228e146bc42ac390ec9745c59fa9054741a1d3db51341b5e981295805892",
    ),
    SOURCE_LOG_PREFIX + "262130.err": (
        5637,
        "44ceddd8420f6f5eb714b764b7ffbd7969460015489b82c9260172d9fa59b22e",
    ),
}

METHOD_IDS = (
    "ordinary_quorum_m4_q3",
    "ordinary_min_m4_q4",
    "delta_min_m4_q4",
)
ALL_SCORE_IDS = ("pi_base", *METHOD_IDS)
SOURCE_CRITICAL_FILES = {
    "control/STOPPED_independent_model_recovery": (
        229, SOURCE_STOP_SHA256
    ),
    "generation/smoke/run_manifest.json": (
        832,
        "22e67bacbad9bc16b953d52d713b694194a352e6964540162fa37f5cd796b978",
    ),
    "generation/smoke/setup_timing.json": (
        10657,
        "04995b13974c6a2e1ca9e8de1d1af79b8cdffa02d4990acf413be02cee09c993",
    ),
    "generation/smoke/timings.json": (
        12743,
        "bc2bd6a57a6c21cd5239da1731984eaab40272c026993bd071c803c3bc3c2b6b",
    ),
    "evaluation/smoke/scores/pi_base.json": (
        39249,
        "c8434eb7c233d07c15934ea0db6a3b49830397d1f963eec0190ee4bfbbb32dde",
    ),
    "evaluation/smoke/scores/ordinary_quorum_m4_q3.json": (
        39067,
        "21560bf9634c92fcb63d6bf1bf6b44fc260b2d4b77ae479e72e00d098b9eeabe",
    ),
    "evaluation/smoke/scores/ordinary_min_m4_q4.json": (
        38844,
        "987b76e5fe0978155f08e4ca1c4d5e7124775efc29e9ba935ec066aded1efc96",
    ),
    "evaluation/smoke/scores/delta_min_m4_q4.json": (
        39646,
        "a375331cb16d95128a5fe988a9a656f4c415f468f1a5ab64ffe6fdaaefa3e8a3",
    ),
}
SOURCE_GENERATION_AGGREGATES = {
    "pi_base": (
        56524,
        "cf4ab6b9dcc253518cd6907d664cf95ca736dbeec42e1325a6ed0af4c01a02cb",
        26819,
        "1e40b57230ca50f75be2ca450ce73eed5143dd414e8b21a894c06bb8026f560c",
    ),
    "ordinary_quorum_m4_q3": (
        86641,
        "6daf9738a38cd3c770352fe92182f3d2577a42b1e22e3bd09d8e3bcff95cf82b",
        57250,
        "c3cde76575c3657f3618a6d191840aaa85b0cdb94bb2544127a7d89ac6da0dc9",
    ),
    "ordinary_min_m4_q4": (
        86232,
        "2e9b845f13ea442533f5ec68d3fdce847e50a6bb5827af4a0757bc358a7d49e6",
        57245,
        "ca8819a74d9273a617575b312689ba8c15380ebee9d15a32327d1010ebe41acb",
    ),
    "delta_min_m4_q4": (
        87679,
        "44a54099584dfb09281ebf2d42b7d03200467f18f2fb81d9deb97a4a43ae3b78",
        57289,
        "ad37438a505ac5dda121f9074c9cce460f049a23c45a247c6577931c809a01b7",
    ),
}

INHERITED_FROZEN_FILES = (
    "scripts/summarize_massive_medical_union_composition_exploratory_v1.py",
    "tests/test_massive_medical_union_composition_exploratory_evaluation.py",
    "tests/test_massive_medical_union_composition_exploratory_independent_model_recovery_v3_workflow.py",
)
FROZEN_INHERITED_FILE_SHA256 = {
    INHERITED_FROZEN_FILES[0]: (
        "30091d091584a9f41c308160330d0f2dacfdae34f1f26c0dcbe8aa97cba9efc2"
    ),
    INHERITED_FROZEN_FILES[1]: (
        "e3432b6859126a177421141da3f9bd0a150a64d7ebf48aeec6a5396ffcd35552"
    ),
    INHERITED_FROZEN_FILES[2]: (
        "4a244dbd1b9460d790383417c1723a680e198d5cf6509a5ef6747522e24df7f8"
    ),
}
FROZEN_INHERITED_FILE_SIZE = {
    INHERITED_FROZEN_FILES[0]: 113364,
    INHERITED_FROZEN_FILES[1]: 48982,
    INHERITED_FROZEN_FILES[2]: 36204,
}
FIXED_EVALUATOR_PATH = INHERITED_FROZEN_FILES[0]
ADDED_FILES = (
    "docs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v5.md",
    "scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v5.py",
    "scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v5_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v5_tillicum.sh",
    "tests/test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v5_workflow.py",
)
EXECUTABLE_FILES = tuple(
    path for path in ADDED_FILES if path.startswith("scripts/")
)
REGULAR_FILES = tuple(
    path for path in ADDED_FILES if not path.startswith("scripts/")
)

EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.9.0+cu129",
    "transformers": "4.57.6",
    "peft": "0.18.1",
    "xgrammar": "0.1.25",
}
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
LOCAL_MODEL_SNAPSHOT = (
    TILLICUM_ROOT
    / "cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    / BASE_REVISION
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


def private_directory_registry():
    return {
        OUTPUT_ROOT: OUTPUT_ROOT.parent,
        CONTROL_ROOT: OUTPUT_ROOT,
        EVALUATION_ROOT: OUTPUT_ROOT,
        EVALUATION_ROOT / "smoke": EVALUATION_ROOT,
        GATE_ROOT: EVALUATION_ROOT / "smoke",
    }


def _known_private_path(path, description):
    path = Path(path)
    registry = private_directory_registry()
    if path not in registry or path.parent != registry[path]:
        raise ValueError(f"{description} is outside the exact v5 directory registry")
    return path


def _open_known_private_parent(path, description):
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | directory | nofollow)
    except OSError as error:
        raise ValueError(f"{description} parent could not be opened safely") from error
    parent_stat = os.fstat(parent_fd)
    parent_lexical = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_lexical.st_mode)
        or parent_stat.st_dev != parent_lexical.st_dev
        or parent_stat.st_ino != parent_lexical.st_ino
        or parent_stat.st_uid != os.geteuid()
    ):
        os.close(parent_fd)
        raise ValueError(f"{description} parent descriptor/type/owner differs")
    return parent_fd


def normalize_known_private_directory(path, description):
    """Clear inherited setgid using anchored, no-follow descriptors."""
    path = _known_private_path(path, description)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    parent_fd = _open_known_private_parent(path, description)
    descriptor = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | directory | nofollow,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        lexical = os.lstat(path)
        if (
            not stat.S_ISDIR(before.st_mode)
            or not stat.S_ISDIR(lexical.st_mode)
            or before.st_dev != lexical.st_dev
            or before.st_ino != lexical.st_ino
            or before.st_uid != os.geteuid()
        ):
            raise ValueError(f"{description} descriptor/type/owner differs")
        os.fchmod(descriptor, 0o700)
        after = os.fstat(descriptor)
        lexical_after = os.lstat(path)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_uid != before.st_uid
            or lexical_after.st_dev != before.st_dev
            or lexical_after.st_ino != before.st_ino
            or stat.S_IMODE(after.st_mode) != 0o700
            or stat.S_IMODE(lexical_after.st_mode) != 0o700
        ):
            raise ValueError(f"{description} could not be normalized safely to mode 0700")
    except OSError as error:
        raise ValueError(f"{description} could not be normalized safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return path


def create_known_private_directory(path, description):
    path = _known_private_path(path, description)
    if os.path.lexists(path):
        raise ValueError(f"refusing existing {description}: {path}")
    parent_fd = _open_known_private_parent(path, description)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{description} could not be created safely") from error
    finally:
        os.close(parent_fd)
    return normalize_known_private_directory(path, description)


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
        raise ValueError(f"{description} {field} differs")
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
        raise ValueError(f"refusing existing artifact: {path}")
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
            raise ValueError(f"artifact appeared while writing: {path}")
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


def tree_inventory(root, expected_files, expected_directories, expected_sha256, description):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{description} root is missing or unsafe")
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        status = path.lstat()
        if path.is_symlink():
            raise ValueError(f"{description} contains a symlink: {relative}")
        entry = {"path": relative, "mode": stat.S_IMODE(status.st_mode)}
        if path.is_file():
            entry.update(
                type="file", size_bytes=status.st_size, file_sha256=sha256_file(path)
            )
        elif path.is_dir():
            entry["type"] = "dir"
        else:
            raise ValueError(f"{description} contains an unsafe object: {relative}")
        entries.append(entry)
    files = [item for item in entries if item["type"] == "file"]
    directories = [item for item in entries if item["type"] == "dir"]
    digest = sha256_bytes(canonical_bytes(entries))
    if (
        len(files) != expected_files
        or len(directories) != expected_directories
        or digest != expected_sha256
    ):
        raise ValueError(
            f"{description} inventory differs: files={len(files)}, "
            f"directories={len(directories)}, sha256={digest}"
        )
    return {
        "root": os.fspath(root.resolve()),
        "file_count": len(files),
        "directory_count": len(directories),
        "inventory_sha256": digest,
        "entries": entries,
    }


def audit_repository():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("gate-recovery checkout is missing or unsafe")
    commit = git(REPO_ROOT, "rev-parse", "HEAD")
    parents = git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit).split()
    if (
        parents != [commit, DIRECT_PARENT_COMMIT]
        or git(REPO_ROOT, "rev-parse", f"{DIRECT_PARENT_COMMIT}^{{tree}}")
        != DIRECT_PARENT_TREE
        or git(REPO_ROOT, "branch", "--show-current") != BRANCH
        or git(REPO_ROOT, "status", "--porcelain")
    ):
        raise ValueError("gate-recovery checkout lineage/branch/cleanliness differs")
    observed = []
    for line in git(
        REPO_ROOT,
        "diff",
        "--name-status",
        "--no-renames",
        f"{DIRECT_PARENT_COMMIT}..{commit}",
    ).splitlines():
        observed.append(tuple(line.split("\t")))
    expected = [("A", path) for path in ADDED_FILES]
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("gate-recovery commit differs from exact five-file add-only allowlist")
    files = {}
    for relative in (*INHERITED_FROZEN_FILES, *ADDED_FILES):
        path = require_regular(REPO_ROOT / relative, f"gate-recovery file {relative}")
        index = git(REPO_ROOT, "ls-files", "-s", "--", relative).split()
        expected_mode = "100755" if relative.startswith("scripts/") else "100644"
        actual_mode = stat.S_IMODE(path.stat().st_mode)
        if (
            len(index) < 4
            or index[0] != expected_mode
            or actual_mode != (0o755 if relative.startswith("scripts/") else 0o644)
        ):
            raise ValueError(f"gate-recovery mode differs for {relative}")
        digest = sha256_file(path)
        if relative in FROZEN_INHERITED_FILE_SHA256 and (
            digest != FROZEN_INHERITED_FILE_SHA256[relative]
            or path.stat().st_size != FROZEN_INHERITED_FILE_SIZE[relative]
        ):
            raise ValueError(f"frozen inherited bytes differ for {relative}")
        files[relative] = {
            "git_mode": expected_mode,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    return {
        "path": os.fspath(REPO_ROOT),
        "branch": BRANCH,
        "commit": commit,
        "direct_parent_commit": DIRECT_PARENT_COMMIT,
        "direct_parent_tree": DIRECT_PARENT_TREE,
        "direct_nonmerge_parent": True,
        "modified_files": [],
        "added_files": list(ADDED_FILES),
        "inherited_frozen_files": list(INHERITED_FROZEN_FILES),
        "files": files,
    }


def audit_source_repository():
    if SOURCE_REPO.is_symlink() or not SOURCE_REPO.is_dir():
        raise ValueError("source incident checkout is missing or unsafe")
    if (
        git(SOURCE_REPO, "rev-parse", "HEAD") != SOURCE_COMMIT
        or git(SOURCE_REPO, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or git(SOURCE_REPO, "branch", "--show-current") != SOURCE_BRANCH
        or git(SOURCE_REPO, "status", "--porcelain")
    ):
        raise ValueError("source incident checkout differs")
    return {
        "path": os.fspath(SOURCE_REPO),
        "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "clean": True,
    }


def audit_failed_v4_stage(source_incident):
    """Bind the immutable partial CPU attempt without changing its 2700 modes."""
    if FAILED_V4_REPO.is_symlink() or not FAILED_V4_REPO.is_dir():
        raise ValueError("failed v4 checkout is missing or unsafe")
    if (
        stat.S_IMODE(FAILED_V4_REPO.stat().st_mode) != 0o2700
        or git(FAILED_V4_REPO, "rev-parse", "HEAD") != FAILED_V4_COMMIT
        or git(FAILED_V4_REPO, "rev-parse", "HEAD^{tree}") != FAILED_V4_TREE
        or git(FAILED_V4_REPO, "branch", "--show-current") != FAILED_V4_BRANCH
        or git(FAILED_V4_REPO, "status", "--porcelain")
    ):
        raise ValueError("failed v4 checkout identity/mode/cleanliness differs")
    if FAILED_V4_OUTPUT.is_symlink() or not FAILED_V4_OUTPUT.is_dir():
        raise ValueError("failed v4 output root is missing or unsafe")
    if stat.S_IMODE(FAILED_V4_OUTPUT.stat().st_mode) != 0o2700:
        raise ValueError("failed v4 output root no longer preserves inherited mode 2700")
    children = list(FAILED_V4_OUTPUT.iterdir())
    if children != [FAILED_V4_CONTROL]:
        raise ValueError("failed v4 output inventory differs")
    if FAILED_V4_CONTROL.is_symlink() or not FAILED_V4_CONTROL.is_dir():
        raise ValueError("failed v4 control root is missing or unsafe")
    if stat.S_IMODE(FAILED_V4_CONTROL.stat().st_mode) != 0o2700:
        raise ValueError("failed v4 control root no longer preserves inherited mode 2700")
    control_entries = list(FAILED_V4_CONTROL.iterdir())
    if control_entries != [FAILED_V4_PREP]:
        raise ValueError("failed v4 control inventory is not exact PREP-only")
    prep_path = require_regular(FAILED_V4_PREP, "failed v4 PREP")
    if (
        stat.S_IMODE(prep_path.stat().st_mode) != 0o400
        or prep_path.stat().st_size != FAILED_V4_PREP_SIZE
        or sha256_file(prep_path) != FAILED_V4_PREP_SHA256
    ):
        raise ValueError("failed v4 PREP bytes/mode differ")
    prep_payload = load_json(prep_path, "failed v4 PREP")
    prep = verify_seal(prep_payload, "failed v4 PREP")
    if prep_payload.get("payload_sha256") != FAILED_V4_PREP_PAYLOAD_SHA256:
        raise ValueError("failed v4 PREP payload seal differs")
    expected_repository_files = {
        path: {"git_mode": mode, "size_bytes": size, "sha256": digest}
        for path, (mode, size, digest) in FAILED_V4_REPOSITORY_FILES.items()
    }
    repository = prep.get("repository")
    if (
        not isinstance(repository, dict)
        or repository.get("path") != os.fspath(FAILED_V4_REPO)
        or repository.get("branch") != FAILED_V4_BRANCH
        or repository.get("commit") != FAILED_V4_COMMIT
        or repository.get("direct_parent_commit") != SOURCE_COMMIT
        or repository.get("direct_parent_tree") != SOURCE_TREE
        or repository.get("direct_nonmerge_parent") is not True
        or repository.get("modified_files")
        != list(INHERITED_FROZEN_FILES)
        or repository.get("added_files")
        != [
            "docs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.md",
            "scripts/audit_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4.py",
            "scripts/stage_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh",
            "scripts/status_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_tillicum.sh",
            "tests/test_massive_medical_union_composition_exploratory_smoke_gate_recovery_v4_workflow.py",
        ]
        or repository.get("files") != expected_repository_files
    ):
        raise ValueError("failed v4 PREP repository provenance differs")
    fixed = {
        "schema_version": 1,
        "workflow_id": "massive_medical_union_composition_exploratory_smoke_gate_recovery_v4",
        "protocol_id": "massive_medical_union_composition_exploratory_smoke_gate_recovery_v4",
        "source_job_actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "new_gpu_h200_minutes_authorized": 0,
        "new_gpu_cost_authorized_usd": 0,
        "external_api_cost_authorized_usd": 0,
        "slurm_jobs_submitted": 0,
        "model_loaded": False,
        "generation_performed": False,
        "scores_recomputed": False,
        "source_scores_reused_byte_identically": True,
        "training": False,
        "external_api_calls": 0,
        "confirmation_authorized": False,
        "confirmation_submitted": False,
        "automatic_continuation": False,
        "confirmatory_claim": False,
    }
    expected_v4_source_incident = json.loads(json.dumps(source_incident))
    expected_v4_source_incident["fixed_evaluator"]["path"] = os.fspath(
        FAILED_V4_REPO / FIXED_EVALUATOR_PATH
    )
    if (
        any(prep.get(key) != value for key, value in fixed.items())
        or prep.get("source_incident") != expected_v4_source_incident
        or prep.get("source_base_snapshot_binding")
        != expected_v4_source_incident.get("control_embedded_seals", {}).get(
            "base_model_snapshot"
        )
    ):
        raise ValueError("failed v4 PREP science/safety provenance differs")
    matching_logs = list(LOG_ROOT.glob(FAILED_V4_LOG_PREFIX + "*"))
    if matching_logs:
        raise ValueError("failed v4 unexpectedly has matching log artifacts")
    return {
        "workflow_id": fixed["workflow_id"],
        "repository": {
            "path": os.fspath(FAILED_V4_REPO),
            "mode": 0o2700,
            "branch": FAILED_V4_BRANCH,
            "commit": FAILED_V4_COMMIT,
            "tree": FAILED_V4_TREE,
            "clean": True,
        },
        "output_root": {"path": os.fspath(FAILED_V4_OUTPUT), "mode": 0o2700},
        "control_root": {"path": os.fspath(FAILED_V4_CONTROL), "mode": 0o2700},
        "prep": binding(prep_path, prep_payload),
        "exact_control_files": ["PREP.json"],
        "staged_present": False,
        "generation_present": False,
        "evaluation_present": False,
        "matching_logs": [],
        "slurm_jobs_submitted": 0,
        "model_loaded": False,
        "generation_performed": False,
        "external_api_calls": 0,
        "confirmation_submitted": False,
        "exact_blocker": "gate-recovery output root mode differs",
        "root_cause": "GPFS setgid inheritance produced mode 2700 under umask 077",
        "immutable_and_not_repaired_in_place": True,
    }


def audit_runtime_versions():
    observed = {}
    for distribution, expected in EXPECTED_RUNTIME_VERSIONS.items():
        value = importlib.metadata.version(distribution)
        if distribution == "torch":
            valid = value in {expected, expected.split("+", 1)[0]}
        else:
            valid = value == expected
        if not valid:
            raise ValueError(
                f"runtime version differs for {distribution}: {value!r} != {expected!r}"
            )
        observed[distribution] = value
    return observed


def audit_cpu_only_environment():
    secret_names = (
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "GOOGLE_API_KEY",
    )
    present = [name for name in secret_names if os.environ.get(name)]
    if present:
        raise ValueError(f"gate recovery received external credentials: {present}")
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("gate recovery must run on CPU outside a Slurm job")
    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": os.fspath(TILLICUM_ROOT / "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.fspath(
            TILLICUM_ROOT / "cache/huggingface/hub"
        ),
    }
    if {key: os.environ.get(key) for key in expected} != expected:
        raise ValueError("gate recovery offline/cache environment differs")
    if "TRANSFORMERS_CACHE" in os.environ:
        raise ValueError("TRANSFORMERS_CACHE must be absent")
    return {
        "cpu_only": True,
        "slurm_job_id": None,
        "cuda_visible_devices": None,
        "external_credentials_present": [],
        **expected,
        "TRANSFORMERS_CACHE": None,
    }


def load_fixed_evaluator():
    path = require_regular(REPO_ROOT / FIXED_EVALUATOR_PATH, "fixed evaluator")
    if (
        path.stat().st_size != FROZEN_INHERITED_FILE_SIZE[FIXED_EVALUATOR_PATH]
        or sha256_file(path) != FROZEN_INHERITED_FILE_SHA256[FIXED_EVALUATOR_PATH]
    ):
        raise ValueError("fixed evaluator bytes differ")
    forbidden = {"torch", "transformers", "peft", "xgrammar", "vllm"}
    present_before = {item.split(".", 1)[0] for item in sys.modules} & forbidden
    if present_before:
        raise ValueError(
            f"model/GPU modules were preloaded before evaluator import: {sorted(present_before)}"
        )
    before = set(sys.modules)
    name = "mmu_composition_smoke_gate_recovery_v5_fixed_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("fixed evaluator cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    present_after = {item.split(".", 1)[0] for item in sys.modules} & forbidden
    if present_after:
        raise ValueError(f"fixed evaluator imported model/GPU modules: {sorted(present_after)}")
    contract = module.cache_equivalence_probe_static_contract()
    if (
        not isinstance(contract, dict)
        or contract.get("contract_sha256") != PROBE_V3_STATIC_CONTRACT_SHA256
        or contract.get("contract_sha256")
        != sha256_bytes(
            canonical_bytes(
                {key: value for key, value in contract.items() if key != "contract_sha256"}
            )
        )
        or contract.get("protocol")
        != "massive_medical_union_composition_cache_equivalence_probe_v3"
    ):
        raise ValueError("fixed evaluator probe-v3 contract differs")
    return module, {
        "path": os.fspath(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "probe_v3_static_contract_sha256": PROBE_V3_STATIC_CONTRACT_SHA256,
        "model_or_gpu_modules_imported": False,
    }


def audit_source_accounting():
    output = subprocess.check_output(
        [
            "sacct",
            "-n",
            "-X",
            "-P",
            "-j",
            SOURCE_JOB_ID,
            "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,AllocTRES,ReqTRES,ExitCode",
        ],
        text=True,
    )
    rows = [line for line in output.splitlines() if line.strip()]
    if rows != [SOURCE_SACCT_ROW] or sha256_bytes(rows[0].encode()) != SOURCE_SACCT_ROW_SHA256:
        raise ValueError("source job durable accounting differs")
    return {
        "job_id": SOURCE_JOB_ID,
        "job_name": SOURCE_JOB_NAME,
        "state": "FAILED",
        "elapsed": "00:10:09",
        "elapsed_seconds": SOURCE_ACTUAL_SECONDS,
        "actual_h200_minutes": SOURCE_ACTUAL_H200_MINUTES,
        "actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "exit_code": "1:0",
        "sacct_row": rows[0],
        "sacct_row_sha256": SOURCE_SACCT_ROW_SHA256,
    }


def audit_source_logs():
    observed = set()
    for path in LOG_ROOT.glob(SOURCE_LOG_PREFIX + "*"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source log inventory contains an unsafe object: {path}")
        observed.add(path.name)
    if observed != set(SOURCE_LOGS):
        raise ValueError(f"source log inventory differs: {sorted(observed)!r}")
    result = {}
    for name, (size, digest) in SOURCE_LOGS.items():
        path = require_regular(LOG_ROOT / name, f"source log {name}")
        if (
            path.stat().st_size != size
            or stat.S_IMODE(path.stat().st_mode) != 0o600
            or sha256_file(path) != digest
        ):
            raise ValueError(f"source log bytes/mode differ: {name}")
        result[name] = binding(path)
    return result


def audit_sample_seal(sample, description):
    if not isinstance(sample, dict) or not isinstance(sample.get("sample_sha256"), str):
        raise ValueError(f"{description} lacks sample seal")
    body = {key: value for key, value in sample.items() if key != "sample_sha256"}
    if sample["sample_sha256"] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} sample seal differs")
    response = sample.get("response")
    if (
        not isinstance(response, str)
        or sample.get("response_sha256") != sha256_bytes(response.encode("utf-8"))
        or sample.get("finish_reason") != "stop"
        or isinstance(sample.get("generated_tokens"), bool)
        or not isinstance(sample.get("generated_tokens"), int)
        or sample["generated_tokens"] < 0
    ):
        raise ValueError(f"{description} response provenance differs")
    return body


def audit_source_generation_embedded_seals():
    run_path = SOURCE_GENERATION_ROOT / "smoke/run_manifest.json"
    run_payload = load_json(run_path, "source run manifest")
    run = verify_seal(run_payload, "source run manifest")
    expected_streams = [
        {"method_id": "pi_base", "domain": "massive", "samples": 60},
        *(
            {"method_id": name, "domain": "massive", "samples": 60}
            for name in METHOD_IDS
        ),
    ]
    if (
        run.get("schema_version") != 1
        or run.get("phase") != "smoke"
        or run.get("protocol_manifest_file_sha256") != SOURCE_MANIFEST_FILE_SHA256
        or run.get("protocol_manifest_payload_sha256")
        != SOURCE_MANIFEST_PAYLOAD_SHA256
        or run.get("streams") != expected_streams
    ):
        raise ValueError("source run manifest semantics differ")
    streams = {}
    for method_id in ALL_SCORE_IDS:
        stream_root = SOURCE_GENERATION_ROOT / f"smoke/{method_id}/massive"
        generation_path = stream_root / "generation.json"
        manifest_path = stream_root / "stream_manifest.json"
        generation_payload = load_json(generation_path, f"{method_id} generation")
        generation = verify_seal(generation_payload, f"{method_id} generation")
        manifest_payload = load_json(manifest_path, f"{method_id} stream manifest")
        manifest = verify_seal(manifest_payload, f"{method_id} stream manifest")
        expected = SOURCE_GENERATION_AGGREGATES[method_id]
        if (
            generation_path.stat().st_size != expected[0]
            or sha256_file(generation_path) != expected[1]
            or manifest_path.stat().st_size != expected[2]
            or sha256_file(manifest_path) != expected[3]
            or set(generation) != {"meta", "samples"}
            or set(manifest) != {"schema_version", "protocol", "meta", "sample_specs"}
            or generation.get("meta") != manifest.get("meta")
            or generation["meta"].get("method_id") != method_id
            or generation["meta"].get("phase") != "smoke"
            or generation["meta"].get("domain") != "MASSIVE"
            or generation["meta"].get("backend")
            != "independent_transformers_peft_models_separate_kv_caches"
            or generation["meta"].get("scientific_adapter_switching_used") is not False
            or not isinstance(generation.get("samples"), list)
            or len(generation["samples"]) != 60
            or not isinstance(manifest.get("sample_specs"), list)
            or len(manifest["sample_specs"]) != 60
        ):
            raise ValueError(f"source {method_id} aggregate/manifest differs")
        shard_names = set()
        for ordinal, (spec, sample) in enumerate(
            zip(manifest["sample_specs"], generation["samples"])
        ):
            if (
                not isinstance(spec, dict)
                or spec.get("ordinal") != ordinal
                or spec.get("sample_index") != 0
                or not isinstance(spec.get("shard_name"), str)
                or spec["shard_name"] in shard_names
            ):
                raise ValueError(f"source {method_id} sample specification differs")
            shard_names.add(spec["shard_name"])
            audit_sample_seal(sample, f"{method_id} aggregate sample {ordinal}")
            shard_path = stream_root / "shards" / spec["shard_name"]
            shard_payload = load_json(shard_path, f"{method_id} shard {ordinal}")
            shard = verify_seal(shard_payload, f"{method_id} shard {ordinal}")
            if (
                set(shard)
                != {"generation_seconds", "sample", "spec", "stream_payload_sha256"}
                or shard.get("spec") != spec
                or shard.get("sample") != sample
                or shard.get("stream_payload_sha256")
                != manifest_payload["payload_sha256"]
            ):
                raise ValueError(f"source {method_id} shard binding differs")
            audit_sample_seal(shard["sample"], f"{method_id} shard sample {ordinal}")
        observed_shards = set()
        for path in (stream_root / "shards").iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source {method_id} shard inventory is unsafe")
            observed_shards.add(path.name)
        if observed_shards != shard_names:
            raise ValueError(f"source {method_id} exact shard names differ")
        streams[method_id] = {
            "generation": binding(generation_path, generation_payload),
            "stream_manifest": binding(manifest_path, manifest_payload),
            "shards": 60,
            "all_embedded_seals_valid": True,
        }
    return {
        "run_manifest": binding(run_path, run_payload),
        "streams": streams,
        "total_shards": 240,
        "all_embedded_seals_valid": True,
    }


def audit_source_control_embedded_seals():
    sealed = 0
    base_snapshot = None
    for path in sorted(SOURCE_CONTROL_ROOT.rglob("*.json")):
        payload = load_json(path, f"source control {path.name}")
        if path.name == "INDEPENDENT_MODEL_RECOVERY_CPU_PREFLIGHT.json":
            snapshot = payload.get("base_model_snapshot")
            contract = payload.get("cache_equivalence_probe_contract")
            if (
                not isinstance(snapshot, dict)
                or snapshot.get("snapshot_payload_sha256")
                != sha256_bytes(
                    canonical_bytes(
                        {
                            key: value
                            for key, value in snapshot.items()
                            if key != "snapshot_payload_sha256"
                        }
                    )
                )
                or not isinstance(contract, dict)
                or contract.get("contract_sha256") != PROBE_V3_STATIC_CONTRACT_SHA256
                or contract.get("contract_sha256")
                != sha256_bytes(
                    canonical_bytes(
                        {
                            key: value
                            for key, value in contract.items()
                            if key != "contract_sha256"
                        }
                    )
                )
            ):
                raise ValueError("source CPU preflight embedded seals differ")
            runtime_artifacts = snapshot.get("runtime_artifacts")
            index = snapshot.get("safetensors_index")
            shards = snapshot.get("safetensors_shards")
            expected_runtime_names = tuple(
                name
                for name in EXPECTED_SNAPSHOT_ARTIFACTS
                if name != "model.safetensors.index.json"
            )
            if (
                snapshot.get("model_id") != BASE_MODEL
                or snapshot.get("revision") != BASE_REVISION
                or not isinstance(runtime_artifacts, list)
                or tuple(row.get("path") for row in runtime_artifacts)
                != expected_runtime_names
                or not isinstance(index, dict)
                or index.get("path") != "model.safetensors.index.json"
                or not isinstance(shards, list)
                or tuple(row.get("path") for row in shards)
                != tuple(sorted(EXPECTED_SHARDS))
            ):
                raise ValueError("source CPU preflight base-snapshot registry differs")
            for row in runtime_artifacts:
                size, digest = EXPECTED_SNAPSHOT_ARTIFACTS[row["path"]]
                if row.get("size_bytes") != size or row.get("sha256") != digest:
                    raise ValueError("source CPU preflight runtime artifact differs")
            index_size, index_digest = EXPECTED_SNAPSHOT_ARTIFACTS[
                "model.safetensors.index.json"
            ]
            if index.get("size_bytes") != index_size or index.get("sha256") != index_digest:
                raise ValueError("source CPU preflight model index differs")
            for row in shards:
                size, digest = EXPECTED_SHARDS[row["path"]]
                if row.get("size_bytes") != size or row.get("sha256") != digest:
                    raise ValueError("source CPU preflight model shard differs")
            base_snapshot = snapshot
            continue
        verify_seal(payload, f"source control {path.name}")
        sealed += 1
    if sealed != 3 or base_snapshot is None:
        raise ValueError("source sealed control JSON count differs")
    return {
        "sealed_json_files": sealed,
        "unsealed_cpu_preflight_files": 1,
        "base_model_snapshot": base_snapshot,
        "base_snapshot_bound_by_immutable_source_preflight": True,
        "live_model_snapshot_rehash_required_for_cpu_gate": False,
    }


def audit_source_evaluator_inputs(evaluator):
    manifest = evaluator.load_manifest(os.fspath(SOURCE_PROTOCOL_MANIFEST))
    scores = {}
    for method_id in ALL_SCORE_IDS:
        path = SOURCE_EVALUATION_ROOT / f"smoke/scores/{method_id}.json"
        loaded = evaluator.load_score(
            manifest, os.fspath(path), "smoke", None if method_id == "pi_base" else method_id
        )
        if loaded["meta"].get("method_id") != method_id:
            raise ValueError(f"source score method differs: {method_id}")
        scores[method_id] = {
            "path": os.fspath(path.resolve()),
            "file_sha256": loaded["file_sha256"],
            "payload_sha256": loaded["payload_sha256"],
            "metrics": loaded["metrics"],
            "tasks": len(loaded["tasks"]),
        }
    timings = evaluator.load_smoke_timings(
        os.fspath(SOURCE_GENERATION_ROOT / "smoke/timings.json"), manifest
    )
    setup_payload = load_json(
        SOURCE_GENERATION_ROOT / "smoke/setup_timing.json", "source setup timing"
    )
    setup = verify_seal(setup_payload, "source setup timing")
    if setup.get("cache_equivalence_probe") != timings["cache_equivalence_probe"]:
        raise ValueError("source setup/final probe-v3 evidence differs")
    if (
        timings["cache_equivalence_probe"].get("result") != "PASS"
        or timings["cache_equivalence_probe"].get("protocol")
        != "massive_medical_union_composition_cache_equivalence_probe_v3"
        or timings["cache_equivalence_probe"].get("model_objects_unique") is not True
        or timings["cache_equivalence_probe"].get("parameter_storages_disjoint") is not True
        or timings["cache_equivalence_probe"].get("cache_tensor_storages_disjoint") is not True
        or timings["cache_equivalence_probe"].get("scientific_adapter_switching_used") is not False
    ):
        raise ValueError("source probe-v3 hard gate differs")
    return {
        "manifest": {
            "path": os.fspath(SOURCE_PROTOCOL_MANIFEST.resolve()),
            "file_sha256": manifest["file_sha256"],
            "payload_sha256": manifest["payload_sha256"],
        },
        "scores": scores,
        "timings": {
            key: timings[key]
            for key in (
                "path",
                "file_sha256",
                "payload_sha256",
                "setup_seconds",
                "streams",
                "cache_equivalence_probe",
            )
        },
        "probe_v3_validated_by_fixed_evaluator": True,
    }


def audit_source_protocol(evaluator):
    manifest_payload = load_json(SOURCE_PROTOCOL_MANIFEST, "source protocol manifest")
    manifest = verify_seal(
        manifest_payload, "source protocol manifest", "manifest_payload_sha256"
    )
    if (
        SOURCE_PROTOCOL_MANIFEST.stat().st_size != 63371
        or sha256_file(SOURCE_PROTOCOL_MANIFEST) != SOURCE_MANIFEST_FILE_SHA256
        or manifest_payload["manifest_payload_sha256"]
        != SOURCE_MANIFEST_PAYLOAD_SHA256
        or manifest.get("protocol_id") != SOURCE_PROTOCOL_ID
    ):
        raise ValueError("source protocol manifest bytes/identity differ")
    evaluator_manifest = evaluator.load_manifest(os.fspath(SOURCE_PROTOCOL_MANIFEST))
    completed = subprocess.run(
        [
            os.fspath(ENV_ROOT / "bin/python"),
            os.fspath(
                REPO_ROOT
                / "scripts/audit_massive_medical_union_composition_exploratory_v1.py"
            ),
            "audit-protocol",
            "--protocol-root",
            os.fspath(SOURCE_PROTOCOL_ROOT),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    try:
        full = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("full source protocol audit emitted invalid JSON") from error
    if (
        full.get("status") != "AUDIT_OK"
        or full.get("manifest_file_sha256") != SOURCE_MANIFEST_FILE_SHA256
        or full.get("manifest_payload_sha256") != SOURCE_MANIFEST_PAYLOAD_SHA256
        or evaluator_manifest["file_sha256"] != SOURCE_MANIFEST_FILE_SHA256
        or evaluator_manifest["payload_sha256"] != SOURCE_MANIFEST_PAYLOAD_SHA256
    ):
        raise ValueError("full source protocol audit differs")
    return {
        **binding(
            SOURCE_PROTOCOL_MANIFEST,
            manifest_payload,
            "manifest_payload_sha256",
        ),
        "protocol_id": SOURCE_PROTOCOL_ID,
        "full_audit": full,
        "fixed_evaluator_manifest_audit": True,
    }


def audit_source_incident():
    if SOURCE_OUTPUT.is_symlink() or not SOURCE_OUTPUT.is_dir():
        raise ValueError("source output root is missing or unsafe")
    children = {}
    for path in SOURCE_OUTPUT.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"source output contains an unsafe child: {path}")
        children[path.name] = stat.S_IMODE(path.stat().st_mode)
    if set(children) != {"control", "generation", "evaluation"}:
        raise ValueError("source output top-level inventory differs")
    evaluator, evaluator_binding = load_fixed_evaluator()
    protocol = audit_source_protocol(evaluator)
    control = tree_inventory(
        SOURCE_CONTROL_ROOT,
        10,
        1,
        SOURCE_CONTROL_INVENTORY_SHA256,
        "source control",
    )
    generation = tree_inventory(
        SOURCE_GENERATION_ROOT,
        252,
        13,
        SOURCE_GENERATION_INVENTORY_SHA256,
        "source generation",
    )
    evaluation = tree_inventory(
        SOURCE_EVALUATION_ROOT,
        4,
        3,
        SOURCE_EVALUATION_INVENTORY_SHA256,
        "source evaluation",
    )
    for relative, (size, digest) in SOURCE_CRITICAL_FILES.items():
        path = require_regular(SOURCE_OUTPUT / relative, f"source critical {relative}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"source critical bytes differ: {relative}")
    expected_stop = (
        f"workflow_id=massive_medical_union_composition_exploratory_workflow_"
        "independent_model_recovery_v3\n"
        "stage=independent_model_recovery\n"
        f"job_id={SOURCE_JOB_ID}\n"
        "exit_code=1\n"
        "retry_authorized=false\n"
        "confirmation_submitted=false\n"
        "external_api_calls=0\n"
    ).encode("utf-8")
    stop = require_regular(
        SOURCE_CONTROL_ROOT / "STOPPED_independent_model_recovery",
        "source STOP sentinel",
    )
    if stop.read_bytes() != expected_stop or stat.S_IMODE(stop.stat().st_mode) != 0o400:
        raise ValueError("source STOP sentinel bytes/mode differ")
    generation_embedded = audit_source_generation_embedded_seals()
    control_embedded = audit_source_control_embedded_seals()
    evaluator_inputs = audit_source_evaluator_inputs(evaluator)
    if any(SOURCE_EVALUATION_ROOT.joinpath("smoke/gate").iterdir()):
        raise ValueError("source gate directory is not exactly empty")
    return {
        "source_repository": audit_source_repository(),
        "source_protocol": protocol,
        "job_accounting": audit_source_accounting(),
        "source_actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "source_actual_h200_minutes": SOURCE_ACTUAL_H200_MINUTES,
        "source_stop": binding(stop),
        "source_logs": audit_source_logs(),
        "control_inventory": control,
        "generation_inventory": generation,
        "evaluation_inventory": evaluation,
        "generation_embedded_seals": generation_embedded,
        "control_embedded_seals": control_embedded,
        "fixed_evaluator_input_audit": evaluator_inputs,
        "fixed_evaluator": evaluator_binding,
        "exact_source_counts": {
            "control_files": 10,
            "generation_files": 252,
            "generation_shards": 240,
            "evaluation_score_files": 4,
            "source_gate_files": 0,
        },
        "exact_terminal_blocker": "ValueError: Smoke timing stream registry differs",
        "source_timing_stream_order": [
            "delta_min_m4_q4:massive",
            "ordinary_min_m4_q4:massive",
            "ordinary_quorum_m4_q3:massive",
            "pi_base:massive",
        ],
        "root_files_byte_identical_and_immutable": True,
        "probe_v3_passed": True,
        "confirmation_submitted": False,
        "external_api_calls": 0,
        "training": False,
        "retry_authorized": False,
    }


def no_new_logs():
    observed = list(LOG_ROOT.glob(NEW_LOG_PREFIX + "*"))
    if observed:
        raise ValueError(f"CPU recovery unexpectedly has logs: {observed!r}")
    return True


def exact_control_inventory(phase):
    expected = {
        "prep": {"PREP.json"},
        "staged": {"PREP.json", "STAGED"},
        "terminal": {"PREP.json", "STAGED", "SMOKE_GATE_RECOVERY_RESULT.json"},
    }[phase]
    if CONTROL_ROOT.is_symlink() or not CONTROL_ROOT.is_dir():
        raise ValueError("gate-recovery control root is missing or unsafe")
    if stat.S_IMODE(CONTROL_ROOT.stat().st_mode) != 0o700:
        raise ValueError("gate-recovery control root mode differs")
    observed = set()
    for path in CONTROL_ROOT.rglob("*"):
        relative = path.relative_to(CONTROL_ROOT).as_posix()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"gate-recovery control contains unsafe object: {relative}")
        if stat.S_IMODE(path.stat().st_mode) != 0o400:
            raise ValueError(f"gate-recovery control mode differs: {relative}")
        observed.add(relative)
    if observed != expected:
        raise ValueError(f"gate-recovery {phase} control inventory differs")
    return sorted(observed)


def exact_output_phase(phase):
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise ValueError("gate-recovery output root is missing or unsafe")
    if stat.S_IMODE(OUTPUT_ROOT.stat().st_mode) != 0o700:
        raise ValueError("gate-recovery output root mode differs")
    expected_children = {"control"} if phase in {"prep", "staged"} else {"control", "evaluation"}
    children = set()
    for path in OUTPUT_ROOT.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"gate-recovery output contains unsafe object: {path}")
        children.add(path.name)
    if children != expected_children or os.path.lexists(GENERATION_ROOT):
        raise ValueError(f"gate-recovery {phase} output inventory differs")
    exact_control_inventory(phase)
    no_new_logs()
    return True


def prep_body(created_at=None):
    ready = require_regular(ENV_ROOT / ".ready", "environment readiness marker")
    incident = audit_source_incident()
    failed_v4 = audit_failed_v4_stage(incident)
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": audit_repository(),
        "source_incident": incident,
        "failed_v4_stage": failed_v4,
        "environment": {
            "path": os.fspath(ENV_ROOT),
            "ready": binding(ready),
            "runtime_versions": audit_runtime_versions(),
            "cpu_only_environment": audit_cpu_only_environment(),
        },
        "source_base_snapshot_binding": incident["control_embedded_seals"][
            "base_model_snapshot"
        ],
        "source_job_actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "new_gpu_h200_minutes_authorized": 0,
        "new_gpu_cost_authorized_usd": 0,
        "external_api_cost_authorized_usd": 0,
        "slurm_jobs_submitted": 0,
        "model_loaded": False,
        "generation_performed": False,
        "scores_recomputed": False,
        "source_scores_reused_byte_identically": True,
        "training": False,
        "external_api_calls": 0,
        "confirmation_authorized": False,
        "confirmation_submitted": False,
        "recovery_confirmation_submission_code_present": False,
        "inherited_confirmation_scripts_not_part_of_recovery_route": True,
        "future_confirmation_requires_separate_user_decision": True,
        "automatic_continuation": False,
        "confirmatory_claim": False,
    }


def command_write_prep(_args):
    if os.path.lexists(OUTPUT_ROOT):
        raise ValueError("fresh gate-recovery output namespace already exists")
    no_new_logs()
    body = prep_body()
    create_known_private_directory(OUTPUT_ROOT, "new v5 output root")
    create_known_private_directory(CONTROL_ROOT, "new v5 control root")
    payload = write_sealed_once(PREP_FILE, body)
    exact_output_phase("prep")
    print(payload["payload_sha256"])


def audit_prep():
    payload = load_json(PREP_FILE, "gate-recovery PREP")
    body = verify_seal(payload, "gate-recovery PREP")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "source_job_actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "new_gpu_h200_minutes_authorized": 0,
        "new_gpu_cost_authorized_usd": 0,
        "external_api_cost_authorized_usd": 0,
        "slurm_jobs_submitted": 0,
        "model_loaded": False,
        "generation_performed": False,
        "scores_recomputed": False,
        "source_scores_reused_byte_identically": True,
        "training": False,
        "external_api_calls": 0,
        "confirmation_authorized": False,
        "confirmation_submitted": False,
        "recovery_confirmation_submission_code_present": False,
        "inherited_confirmation_scripts_not_part_of_recovery_route": True,
        "future_confirmation_requires_separate_user_decision": True,
        "automatic_continuation": False,
        "confirmatory_claim": False,
    }
    if any(body.get(key) != value for key, value in fixed.items()):
        raise ValueError("gate-recovery PREP identity/safety flags differ")
    if body.get("source_base_snapshot_binding") != body.get(
        "source_incident", {}
    ).get("control_embedded_seals", {}).get("base_model_snapshot"):
        raise ValueError("gate-recovery PREP source base-snapshot binding differs")
    if body.get("failed_v4_stage") != audit_failed_v4_stage(
        body.get("source_incident")
    ):
        raise ValueError("gate-recovery PREP failed-v4 binding differs")
    exact_output_phase("prep" if not os.path.lexists(STAGED_FILE) else "staged" if not os.path.lexists(RESULT_FILE) else "terminal")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_prep(_args):
    prep = audit_prep()
    print(json.dumps({"status": "GATE_RECOVERY_PREP_OK", "payload_sha256": prep["payload_sha256"]}, sort_keys=True))


def staged_bytes(prep):
    return (
        f"workflow_id={WORKFLOW_ID}\n"
        f"protocol_id={PROTOCOL_ID}\n"
        f"repo_commit={prep['repository']['commit']}\n"
        f"prep_file_sha256={sha256_file(PREP_FILE)}\n"
        f"source_commit={SOURCE_COMMIT}\n"
        f"source_job_id={SOURCE_JOB_ID}\n"
        f"failed_v4_commit={FAILED_V4_COMMIT}\n"
        f"failed_v4_prep_file_sha256={FAILED_V4_PREP_SHA256}\n"
        f"source_control_inventory_sha256={SOURCE_CONTROL_INVENTORY_SHA256}\n"
        f"source_generation_inventory_sha256={SOURCE_GENERATION_INVENTORY_SHA256}\n"
        f"source_evaluation_inventory_sha256={SOURCE_EVALUATION_INVENTORY_SHA256}\n"
        f"fixed_evaluator_sha256={FROZEN_INHERITED_FILE_SHA256[FIXED_EVALUATOR_PATH]}\n"
        f"probe_v3_static_contract_sha256={PROBE_V3_STATIC_CONTRACT_SHA256}\n"
        "cpu_only=true\n"
        "slurm_jobs_submitted=0\n"
        "new_gpu_h200_minutes_authorized=0\n"
        "model_loaded=false\n"
        "generation_performed=false\n"
        "scores_recomputed=false\n"
        "training=false\n"
        "external_api_calls=0\n"
        "confirmation_authorized=false\n"
        "confirmation_submitted=false\n"
    ).encode("utf-8")


def command_write_staged(_args):
    prep = audit_prep()
    if os.path.lexists(STAGED_FILE) or os.path.lexists(RESULT_FILE):
        raise ValueError("gate-recovery staging output already exists")
    if os.path.lexists(EVALUATION_ROOT) or os.path.lexists(GENERATION_ROOT):
        raise ValueError("gate-recovery scientific namespace is not fresh")
    if audit_repository() != prep.get("repository"):
        raise ValueError("gate-recovery repository drifted after PREP")
    if audit_source_incident() != prep.get("source_incident"):
        raise ValueError("source incident drifted after PREP")
    if audit_failed_v4_stage(prep["source_incident"]) != prep.get(
        "failed_v4_stage"
    ):
        raise ValueError("failed v4 stage drifted after PREP")
    atomic_write_once(STAGED_FILE, staged_bytes(prep))
    exact_output_phase("staged")
    print(sha256_file(STAGED_FILE))


def audit_staged():
    prep_payload = load_json(PREP_FILE, "gate-recovery PREP")
    prep_body_value = verify_seal(prep_payload, "gate-recovery PREP")
    staged = require_regular(STAGED_FILE, "gate-recovery STAGED")
    if (
        staged.read_bytes()
        != staged_bytes({**prep_body_value, "payload_sha256": prep_payload["payload_sha256"]})
        or stat.S_IMODE(staged.stat().st_mode) != 0o400
    ):
        raise ValueError("gate-recovery STAGED bytes/mode differ")
    exact_output_phase("staged" if not os.path.lexists(RESULT_FILE) else "terminal")
    return {
        "prep": binding(PREP_FILE, prep_payload),
        "staged": binding(STAGED_FILE),
        "repository": prep_body_value["repository"],
        "source_incident": prep_body_value["source_incident"],
        "failed_v4_stage": prep_body_value["failed_v4_stage"],
    }


def command_audit_staged(_args):
    staged = audit_staged()
    print(json.dumps({"status": "GATE_RECOVERY_STAGED_OK", "staged": staged["staged"]}, sort_keys=True))


def audit_gate(evaluator=None):
    if EVALUATION_ROOT.is_symlink() or not EVALUATION_ROOT.is_dir():
        raise ValueError("recovered evaluation root is missing or unsafe")
    if stat.S_IMODE(EVALUATION_ROOT.stat().st_mode) != 0o700:
        raise ValueError("recovered evaluation root mode differs")
    expected_dirs = {"smoke", "smoke/gate"}
    expected_files = {
        "smoke/gate/runtime_projection.json",
        "smoke/gate/summary.json",
        "smoke/gate/STOPPED_EXPLORATORY_SMOKE",
    }
    observed_dirs, observed_files = set(), set()
    file_bindings = {}
    for path in EVALUATION_ROOT.rglob("*"):
        relative = path.relative_to(EVALUATION_ROOT).as_posix()
        if path.is_symlink():
            raise ValueError(f"recovered gate contains a symlink: {relative}")
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode) != 0o700:
                raise ValueError(f"recovered gate directory mode differs: {relative}")
            observed_dirs.add(relative)
        elif path.is_file():
            if stat.S_IMODE(path.stat().st_mode) != 0o600:
                raise ValueError(f"recovered gate file mode differs: {relative}")
            observed_files.add(relative)
            file_bindings[relative] = binding(path)
        else:
            raise ValueError(f"recovered gate contains an unsafe object: {relative}")
    if observed_dirs != expected_dirs or observed_files != expected_files:
        raise ValueError("recovered gate exact inventory differs")
    projection_path = GATE_ROOT / "runtime_projection.json"
    summary_path = GATE_ROOT / "summary.json"
    sentinel_path = GATE_ROOT / "STOPPED_EXPLORATORY_SMOKE"
    projection_payload = load_json(projection_path, "recovered runtime projection")
    projection = verify_seal(projection_payload, "recovered runtime projection")
    summary_payload = load_json(summary_path, "recovered smoke summary")
    summary = verify_seal(summary_payload, "recovered smoke summary")
    sentinel_payload = load_json(sentinel_path, "recovered smoke STOP sentinel")
    sentinel = verify_seal(sentinel_payload, "recovered smoke STOP sentinel")
    projection_keys = {
        "schema_version",
        "protocol",
        "protocol_id",
        "protocol_manifest_file_sha256",
        "protocol_manifest_payload_sha256",
        "prompt_file_sha256",
        "formula",
        "medical_planning_envelope",
        "medical_selected_tokens_per_method_bound",
        "medical_all_three_methods_selected_tokens_bound",
        "timings",
        "cache_equivalence_probe",
        "setup_seconds",
        "four_stream_smoke_generation_seconds",
        "minimum_method_selected_tokens_per_second",
        "smoke_score_and_gate_seconds_observed_before_summary_seal",
        "scoring_floor_seconds",
        "projected_confirmation_seconds",
        "projected_confirmation_h200_minutes",
        "contingency_fraction",
    }
    summary_keys = {
        "schema_version",
        "protocol",
        "protocol_id",
        "protocol_manifest_file_sha256",
        "protocol_manifest_payload_sha256",
        "thresholds",
        "runtime_projection",
        "results",
        "checks",
        "all_three_methods_passed",
        "status",
        "confirmation_submission_eligible",
        "confirmatory_claim",
        "wave2_v1_status",
        "wave3_v1_eligible",
        "wave3_v1_submitted_or_released",
    }
    sentinel_keys = {
        "schema_version",
        "protocol",
        "protocol_id",
        "status",
        "summary_path",
        "summary_file_sha256",
        "summary_payload_sha256",
        "confirmatory_claim",
        "wave2_v1_status",
        "wave3_v1_eligible",
        "wave3_v1_submitted_or_released",
    }
    if (
        set(projection) != projection_keys
        or set(summary) != summary_keys
        or set(sentinel) != sentinel_keys
        or projection.get("schema_version") != 1
        or projection.get("protocol")
        != "massive_medical_union_composition_exploratory_runtime_projection_v1"
        or projection.get("protocol_id") != SOURCE_PROTOCOL_ID
        or projection.get("protocol_manifest_file_sha256")
        != SOURCE_MANIFEST_FILE_SHA256
        or projection.get("protocol_manifest_payload_sha256")
        != SOURCE_MANIFEST_PAYLOAD_SHA256
        or projection.get("prompt_file_sha256")
        != "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
        or projection.get("contingency_fraction") != 0.20
        or summary.get("schema_version") != 1
        or summary.get("protocol")
        != "massive_medical_union_composition_exploratory_smoke_gate_v1"
        or summary.get("protocol_id") != SOURCE_PROTOCOL_ID
        or summary.get("protocol_manifest_file_sha256")
        != SOURCE_MANIFEST_FILE_SHA256
        or summary.get("protocol_manifest_payload_sha256")
        != SOURCE_MANIFEST_PAYLOAD_SHA256
        or sentinel.get("schema_version") != 1
        or sentinel.get("protocol")
        != "massive_medical_union_composition_exploratory_smoke_sentinel_v1"
        or sentinel.get("protocol_id") != SOURCE_PROTOCOL_ID
    ):
        raise ValueError("recovered projection/summary/sentinel schema differs")
    if (
        sentinel.get("status") != "STOPPED_EXPLORATORY_SMOKE"
        or Path(sentinel.get("summary_path", "")).resolve() != summary_path.resolve()
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
        or summary.get("status") != "STOPPED_EXPLORATORY_SMOKE"
        or summary.get("confirmation_submission_eligible") is not False
        or summary.get("all_three_methods_passed") is not False
    ):
        raise ValueError("recovered smoke STOP summary/sentinel differs")
    for value in (summary, sentinel):
        if (
            value.get("confirmatory_claim") is not False
            or value.get("wave2_v1_status") != "STOP"
            or value.get("wave3_v1_eligible") is not False
            or value.get("wave3_v1_submitted_or_released") is not False
        ):
            raise ValueError("recovered gate safety flags differ")
    expected_checks = {
        *(
            f"{method}.{name}"
            for method in METHOD_IDS
            for name in (
                "structured_valid_fraction",
                "truncations",
                "joint_intent_gain_over_paired_base",
            )
        ),
        "runtime_projection_fits_released_confirmation_budget",
    }
    checks = summary.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_checks
        or checks.get("runtime_projection_fits_released_confirmation_budget") is not False
        or any(checks[key] is not True for key in checks if key != "runtime_projection_fits_released_confirmation_budget")
    ):
        raise ValueError("recovered smoke scientific/runtime gate shape differs")
    results = summary.get("results")
    if not isinstance(results, dict) or set(results) != set(METHOD_IDS):
        raise ValueError("recovered smoke method registry differs")
    for method in METHOD_IDS:
        comparison = results[method].get("comparison", {})
        method_checks = results[method].get("checks")
        if (
            method_checks != {
                "structured_valid_fraction": True,
                "truncations": True,
                "joint_intent_gain_over_paired_base": True,
            }
            or comparison.get("paired_joint_delta") != 0.13333333333333333
            or comparison.get("paired_joint_bootstrap_95ci")
            != [0.03333333333333333, 0.23333333333333334]
            or comparison.get("joint_one_sided_exact_mcnemar_p") != 0.0107421875
        ):
            raise ValueError(f"recovered smoke science metrics differ: {method}")
    expected_projection_seconds = 1.20 * (
        66.09861348790582
        + 10 * 501.2222172736656
        + 38796 / 8.945817873199246
        + 60
    )
    observed_scoring = projection.get(
        "smoke_score_and_gate_seconds_observed_before_summary_seal"
    )
    expected_timing_binding = projection.get("timings")
    if evaluator is not None:
        source_timing_payload = load_json(
            SOURCE_GENERATION_ROOT / "smoke/timings.json", "source timing for gate"
        )
        verify_seal(source_timing_payload, "source timing for gate")
        expected_timing_binding = {
            "path": os.fspath(
                (SOURCE_GENERATION_ROOT / "smoke/timings.json").resolve()
            ),
            "file_sha256": SOURCE_CRITICAL_FILES[
                "generation/smoke/timings.json"
            ][1],
            "payload_sha256": source_timing_payload["payload_sha256"],
        }
    if (
        projection.get("setup_seconds") != 66.09861348790582
        or projection.get("four_stream_smoke_generation_seconds")
        != 501.2222172736656
        or projection.get("minimum_method_selected_tokens_per_second")
        != 8.945817873199246
        or projection.get("medical_all_three_methods_selected_tokens_bound") != 38796
        or projection.get("medical_selected_tokens_per_method_bound") != 12932
        or projection.get("timings") != expected_timing_binding
        or isinstance(observed_scoring, bool)
        or not isinstance(observed_scoring, (int, float))
        or observed_scoring < 0
        or projection.get("scoring_floor_seconds")
        != max(60, 10 * observed_scoring)
        or projection.get("scoring_floor_seconds") != 60
        or projection.get("projected_confirmation_seconds")
        != expected_projection_seconds
        or projection.get("projected_confirmation_h200_minutes")
        != expected_projection_seconds / 60
        or projection["projected_confirmation_h200_minutes"] <= 100
        or projection.get("cache_equivalence_probe", {}).get("protocol")
        != "massive_medical_union_composition_cache_equivalence_probe_v3"
        or projection["cache_equivalence_probe"].get("result") != "PASS"
    ):
        raise ValueError("recovered runtime-projection arithmetic/probe differs")
    if evaluator is not None:
        # Re-running the fixed CPU evaluator against an exact existing
        # namespace performs audit-only write_or_audit checks over every
        # projection, summary, comparison, threshold, and sentinel field.
        run_fixed_evaluator(evaluator)
        manifest = evaluator.load_manifest(os.fspath(SOURCE_PROTOCOL_MANIFEST))
        planning = evaluator.load_medical_planning_envelope(manifest)
        base = evaluator.load_score(
            manifest,
            os.fspath(SOURCE_EVALUATION_ROOT / "smoke/scores/pi_base.json"),
            "smoke",
        )
        if (
            projection.get("formula")
            != manifest["body"]["runtime_projection"]["formula"]
            or projection.get("medical_planning_envelope") != planning
            or summary.get("thresholds")
            != manifest["body"]["gates"]["smoke_all_methods_conjunction"]
        ):
            raise ValueError("recovered planning/formula/threshold provenance differs")
        for method in METHOD_IDS:
            score = evaluator.load_score(
                manifest,
                os.fspath(SOURCE_EVALUATION_ROOT / f"smoke/scores/{method}.json"),
                "smoke",
                method,
            )
            if results[method].get("comparison") != evaluator.compare(base, score):
                raise ValueError(f"recovered full comparison differs: {method}")
    summary_projection = summary.get("runtime_projection")
    if (
        not isinstance(summary_projection, dict)
        or set(summary_projection)
        != projection_keys | {"path", "file_sha256", "payload_sha256"}
        or summary_projection.get("file_sha256") != sha256_file(projection_path)
        or summary_projection.get("payload_sha256")
        != projection_payload["payload_sha256"]
        or any(summary_projection.get(key) != value for key, value in projection.items())
    ):
        raise ValueError("recovered summary projection binding differs")
    return {
        "status": "STOPPED_EXPLORATORY_SMOKE",
        "scientific_smoke_method_gates_passed": True,
        "runtime_projection_gate_passed": False,
        "projected_confirmation_seconds": expected_projection_seconds,
        "projected_confirmation_h200_minutes": expected_projection_seconds / 60,
        "released_confirmation_cap_h200_minutes": 100,
        "summary": binding(summary_path, summary_payload),
        "runtime_projection": binding(projection_path, projection_payload),
        "sentinel": binding(sentinel_path, sentinel_payload),
        "exact_evaluation_inventory": dict(sorted(file_bindings.items())),
    }


def run_fixed_evaluator(evaluator):
    arguments = [
        "smoke",
        "--protocol-manifest",
        os.fspath(SOURCE_PROTOCOL_MANIFEST),
        "--base-score",
        os.fspath(SOURCE_EVALUATION_ROOT / "smoke/scores/pi_base.json"),
    ]
    for method in METHOD_IDS:
        arguments.extend(
            [
                "--method-score",
                f"{method}={SOURCE_EVALUATION_ROOT / f'smoke/scores/{method}.json'}",
            ]
        )
    arguments.extend(
        [
            "--timings",
            os.fspath(SOURCE_GENERATION_ROOT / "smoke/timings.json"),
            "--output-dir",
            os.fspath(GATE_ROOT),
        ]
    )
    forbidden = {"torch", "transformers", "peft", "xgrammar", "vllm"}
    present_before = {item.split(".", 1)[0] for item in sys.modules} & forbidden
    if present_before:
        raise ValueError(
            f"model/GPU modules were preloaded before gate evaluation: {sorted(present_before)}"
        )
    result = evaluator.run(arguments)
    present_after = {item.split(".", 1)[0] for item in sys.modules} & forbidden
    if present_after:
        raise ValueError(f"gate evaluation imported model/GPU modules: {sorted(present_after)}")
    if result != 2:
        raise ValueError(f"recovered evaluator did not reproduce protocol STOP: rc={result}")
    return result


def command_recover_gate(_args):
    staged = audit_staged()
    prep_payload = load_json(PREP_FILE, "gate-recovery PREP")
    prep = verify_seal(prep_payload, "gate-recovery PREP")
    if os.path.lexists(RESULT_FILE) or os.path.lexists(EVALUATION_ROOT) or os.path.lexists(GENERATION_ROOT):
        raise ValueError("gate recovery requires an exact fresh scientific/result namespace")
    if audit_repository() != prep.get("repository"):
        raise ValueError("gate-recovery repository drifted before evaluation")
    audit_cpu_only_environment()
    source_before = audit_source_incident()
    if source_before != prep.get("source_incident") or source_before != staged["source_incident"]:
        raise ValueError("source incident drifted before gate recovery")
    failed_v4_before = audit_failed_v4_stage(source_before)
    if (
        failed_v4_before != prep.get("failed_v4_stage")
        or failed_v4_before != staged["failed_v4_stage"]
    ):
        raise ValueError("failed v4 stage drifted before gate recovery")
    create_known_private_directory(EVALUATION_ROOT, "new v5 evaluation root")
    create_known_private_directory(EVALUATION_ROOT / "smoke", "new v5 smoke root")
    create_known_private_directory(GATE_ROOT, "new v5 smoke gate root")
    evaluator, evaluator_binding = load_fixed_evaluator()
    run_fixed_evaluator(evaluator)
    gate = audit_gate(evaluator)
    source_after = audit_source_incident()
    if source_after != source_before:
        raise ValueError("source incident changed during gate recovery")
    failed_v4_after = audit_failed_v4_stage(source_after)
    if failed_v4_after != failed_v4_before:
        raise ValueError("failed v4 stage changed during gate recovery")
    if audit_repository() != prep.get("repository"):
        raise ValueError("gate-recovery repository drifted during evaluation")
    body = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "cpu_only_smoke_gate_recovery",
        "prep": binding(PREP_FILE, prep_payload),
        "staged": binding(STAGED_FILE),
        "fixed_evaluator": evaluator_binding,
        "source_incident_before": source_before,
        "source_incident_after": source_after,
        "source_incident_unchanged": True,
        "failed_v4_stage_before": failed_v4_before,
        "failed_v4_stage_after": failed_v4_after,
        "failed_v4_stage_unchanged": True,
        "gate": gate,
        "scientific_smoke_method_gates_passed": True,
        "runtime_projection_gate_passed": False,
        "scientific_status": "STOPPED_EXPLORATORY_SMOKE",
        "source_job_actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "new_gpu_h200_minutes": 0,
        "new_gpu_cost_usd": 0,
        "slurm_jobs_submitted": 0,
        "model_loaded": False,
        "generation_performed": False,
        "scores_recomputed": False,
        "source_scores_reused_byte_identically": True,
        "training": False,
        "external_api_calls": 0,
        "confirmation_authorized": False,
        "confirmation_submitted": False,
        "recovery_confirmation_submission_code_present": False,
        "inherited_confirmation_scripts_not_part_of_recovery_route": True,
        "future_confirmation_requires_separate_user_decision": True,
        "automatic_continuation": False,
        "confirmatory_claim": False,
    }
    payload = write_sealed_once(RESULT_FILE, body)
    exact_output_phase("terminal")
    print(payload["payload_sha256"])


def audit_result():
    payload = load_json(RESULT_FILE, "gate-recovery result")
    body = verify_seal(payload, "gate-recovery result")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "stage": "cpu_only_smoke_gate_recovery",
        "source_incident_unchanged": True,
        "failed_v4_stage_unchanged": True,
        "scientific_smoke_method_gates_passed": True,
        "runtime_projection_gate_passed": False,
        "scientific_status": "STOPPED_EXPLORATORY_SMOKE",
        "source_job_actual_gpu_cost_usd": SOURCE_ACTUAL_GPU_COST_USD,
        "new_gpu_h200_minutes": 0,
        "new_gpu_cost_usd": 0,
        "slurm_jobs_submitted": 0,
        "model_loaded": False,
        "generation_performed": False,
        "scores_recomputed": False,
        "source_scores_reused_byte_identically": True,
        "training": False,
        "external_api_calls": 0,
        "confirmation_authorized": False,
        "confirmation_submitted": False,
        "recovery_confirmation_submission_code_present": False,
        "inherited_confirmation_scripts_not_part_of_recovery_route": True,
        "future_confirmation_requires_separate_user_decision": True,
        "automatic_continuation": False,
        "confirmatory_claim": False,
    }
    if any(body.get(key) != value for key, value in fixed.items()):
        raise ValueError("gate-recovery result identity/safety flags differ")
    prep_payload = load_json(PREP_FILE, "gate-recovery PREP")
    prep = verify_seal(prep_payload, "gate-recovery PREP")
    evaluator, evaluator_binding = load_fixed_evaluator()
    gate = audit_gate(evaluator)
    live_source = audit_source_incident()
    live_failed_v4 = audit_failed_v4_stage(live_source)
    live_repository = audit_repository()
    if (
        body.get("prep") != binding(PREP_FILE, prep_payload)
        or body.get("staged") != binding(STAGED_FILE)
        or body.get("fixed_evaluator") != evaluator_binding
        or body.get("gate") != gate
        or body.get("source_incident_before") != prep.get("source_incident")
        or body.get("source_incident_after") != live_source
        or body.get("failed_v4_stage_before") != prep.get("failed_v4_stage")
        or body.get("failed_v4_stage_after") != live_failed_v4
        or live_failed_v4 != prep.get("failed_v4_stage")
        or live_source != prep.get("source_incident")
        or live_repository != prep.get("repository")
    ):
        raise ValueError("gate-recovery result live provenance differs")
    return {**body, "payload_sha256": payload["payload_sha256"]}


def command_audit_terminal(_args):
    exact_output_phase("terminal")
    result = audit_result()
    print(
        json.dumps(
            {
                "status": "SMOKE_GATE_RECOVERY_TERMINAL_STOP",
                "scientific_status": result["scientific_status"],
                "scientific_smoke_method_gates_passed": True,
                "projected_confirmation_h200_minutes": result["gate"][
                    "projected_confirmation_h200_minutes"
                ],
                "released_confirmation_cap_h200_minutes": 100,
                "confirmation_authorized": False,
                "confirmation_submitted": False,
                "external_api_calls": 0,
                "new_gpu_h200_minutes": 0,
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )


def command_status(_args):
    if not os.path.lexists(OUTPUT_ROOT):
        print(json.dumps({"status": "SMOKE_GATE_RECOVERY_NOT_STAGED"}, sort_keys=True))
        return
    if not os.path.lexists(PREP_FILE):
        raise ValueError("gate-recovery output exists without PREP")
    if not os.path.lexists(STAGED_FILE):
        prep = audit_prep()
        print(json.dumps({"status": "SMOKE_GATE_RECOVERY_PREP_ONLY", "payload_sha256": prep["payload_sha256"]}, sort_keys=True))
        return
    if not os.path.lexists(RESULT_FILE):
        audit_staged()
        if os.path.lexists(EVALUATION_ROOT) or os.path.lexists(GENERATION_ROOT):
            raise ValueError("gate recovery has terminal-unsealed scientific state")
        print(json.dumps({"status": "SMOKE_GATE_RECOVERY_STAGED"}, sort_keys=True))
        return
    command_audit_terminal(_args)


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write-prep").set_defaults(function=command_write_prep)
    commands.add_parser("audit-prep").set_defaults(function=command_audit_prep)
    commands.add_parser("write-staged").set_defaults(function=command_write_staged)
    commands.add_parser("audit-staged").set_defaults(function=command_audit_staged)
    commands.add_parser("recover-gate").set_defaults(function=command_recover_gate)
    commands.add_parser("audit-terminal").set_defaults(function=command_audit_terminal)
    commands.add_parser("status").set_defaults(function=command_status)
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    return args.function(args)


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
