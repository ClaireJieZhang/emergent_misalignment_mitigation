#!/usr/bin/env python3
"""Fail-closed CPU control plane for the v7 derivation-only recovery.

V7 does not judge, resume, or mutate v6.  It binds the exact completed v6
namespace and repairs only a floating-point summation-order mismatch while
writing all derived artifacts to a fresh output namespace.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


def _private_module(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load private module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_judge = _private_module(
    "_mmu_judge_derive_recovery_v7_private_v6_judge",
    "judge_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v6.py",
)

# Use the already-audited canonical JSON primitives without changing any v6
# module globals.  These writes are exclusive (no overwrite).
load_json = source_judge.base.load_json
audit_seal = source_judge.base.audit_seal
seal = source_judge.base.seal
sha256_file = source_judge.base.sha256_file


def atomic_json(path, value, mode=0o600):
    """Exclusively create a private JSON file with a caller-frozen mode."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def binding(path, payload=None, require_seal=False, seal_field="payload_sha256"):
    """Return an exact file binding; optionally load and verify its seal."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Binding source is absent or unsafe: {path}")
    if require_seal:
        if payload is not None:
            raise ValueError("Provide payload or require_seal, not both")
        payload = load_json(path)
        audit_seal(payload, os.fspath(path))
    result = {
        "path": os.fspath(path), "size": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }
    if payload is not None:
        result[seal_field] = payload[seal_field]
    return result

SOURCE_V6_RECOVERY_ID = source_judge.RECOVERY_ID
SOURCE_PROTOCOL_ID = source_judge.source.PROTOCOL_ID
RECOVERY_ID = (
    "massive_medical_union_composition_exploratory_sequential_confirmation_v1_"
    "judge_derive_recovery_v7"
)
SOURCE_COMMIT = "c4016c332c461efa07c85028a164c787f2e65650"
SOURCE_TREE = "20b84f0edbebd1274fa2ca11144d11b5b95e2991"
SOURCE_PARENT = "3834f4215f6606ad49e511620bedd49219ecc3df"
SOURCE_BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-recovery-v6"
)
BRANCH = (
    "claire/capability-quorum-secure-code-composition-exploratory-under5-"
    "sequential-v1-judge-derive-recovery-v7"
)

TILLICUM_ROOT = Path("/gpfs/projects/stf/claizhan/subliminal-mitigate")
SOURCE_REPO = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-recovery-v6"
)
SOURCE_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v6"
)
REPO_ROOT = TILLICUM_ROOT / (
    "projects/subliminal-mitigate-mmu-composition-exploratory-sequential-"
    "confirmation-v1-judge-derive-recovery-v7"
)
RECOVERY_OUTPUT = TILLICUM_ROOT / (
    "outputs/massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_derive_recovery_v7"
)
CONTROL_ROOT = RECOVERY_OUTPUT / "control"
MEDICAL_ROOT = RECOVERY_OUTPUT / "evaluation/medical"
FINAL_ROOT = RECOVERY_OUTPUT / "evaluation/final"
LOG_ROOT = RECOVERY_OUTPUT / "logs"
MANIFEST_FILE = CONTROL_ROOT / "JUDGE_DERIVE_RECOVERY_V7_MANIFEST.json"
PREP_FILE = CONTROL_ROOT / "PREP.json"
PREFLIGHT_FILE = CONTROL_ROOT / "CPU_PREFLIGHT.json"
STAGED_FILE = CONTROL_ROOT / "STAGED"
LOCK_FILE = CONTROL_ROOT / "DERIVATION_LOCK.json"
FINAL_RESULT_FILE = CONTROL_ROOT / "FINAL_RESULT.json"

ADDED_FILES = (
    "scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py",
    "scripts/summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7.py",
    "scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7_tillicum.sh",
    "scripts/derive_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7_tillicum.sh",
    "scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_derive_recovery_v7_tillicum.sh",
    "tests/test_massive_medical_union_composition_exploratory_sequential_judge_derive_recovery_v7_summary.py",
    "tests/test_massive_medical_union_composition_exploratory_sequential_judge_derive_recovery_v7_control.py",
)

SOURCE_INVENTORY_FILE_COUNT = 252
SOURCE_INVENTORY_STREAM_SHA256 = (
    "06261ede31668b3e7e51dce5b678898b09173ddfb6a01f877ea84346db6f59d0"
)

# mode, size, file SHA-256, sealed-payload SHA-256 (None for plain logs).
SOURCE_CRITICAL_FILES = {
    "control/JUDGE_RECOVERY_V6_MANIFEST.json": (0o600, 24281, "2f3119687dc8a0c9fb86114bf34cbe20a45691c74adc89e52ace86c458dada68", "dd6021d88674234ff36c0395314abd5d92fc1875b94ea1bccd047e415f61cc66"),
    "control/PREP.json": (0o600, 952, "417a129153c62a60d915998305d3bc0d9093e148a12b8edbc4003c91a2108ed7", "15a58e17e9f36e8486b467976dd0634387da85bfc5bb1117ac576efce186363f"),
    "control/CPU_PREFLIGHT.json": (0o600, 1556, "d9e6dfcb97142c4de12d8fb266e1df656424cba102290728b16abdf629b0131b", "fa8bb3ec0e6d2674393095ebef3e17211fb62b436fb88b7f558ad52b9ff3e7be"),
    "control/STAGED": (0o600, 1351, "8431c084f65e761f84db3344807c3064b33d96cb5327a6debd81629e05903c9e", "1e9077b6c23bb70a2b323b96e0ea3a76401131671d63136a89c0dc4bd3c6d0ce"),
    "control/CANARY_LOCK_OWNER.json": (0o400, 1062, "ccd98cb4997d31d80e3878e457442bd22d7e8a9f45ac4d3dc8d1d444a6b41b0c", "f1d763677a99ff87dc79a110f0eb6ed9d10ffda272924f73eab657574e258a94"),
    "control/CANARY_AUTHORIZATION.json": (0o400, 4744, "da77a811d2892ed75fb0c47fd2119b55d1cc4575898c5fcbe216eb4824f9b925", "6097273c65321957320c4e3fe61962820ede31f2a0b641d0afa2cef099fcbfa6"),
    "control/CANARY_SUCCESS.json": (0o400, 2549, "06c01f3b8fb3d990b6f7b1f554045af215c02ed03f482063073c5a1e75e3fd42", "83fc2a7339fa5489fcad7238bc6eb4252ccbd27afe59eec0ee39e7a99c46a3d6"),
    "control/CONTINUATION_LOCK_OWNER.json": (0o400, 1074, "08a2da9b1cbb63310d0ef076a19dc8edf0097bf988e6d4df8c074f49e4230007", "df9bfeb4211cf8aaf6df82089ab8868266fe2d69cb8e8dd8f65f841680807d62"),
    "control/CONTINUATION_AUTHORIZATION.json": (0o400, 5662, "972bf21cd90dd10080f94306c4922812f7529e739138f83b7dc1fb25c9b85b01", "bbb3a5e027bb0421f604d62177e7405f3e66662ebaf82279dc4b4af504744a9c"),
    "control/CONTINUATION_SUCCESS.json": (0o400, 3267, "2afa07a8b8f40db547b38b43288527981cf19499a57602d738749cd0edcf795e", "6b376ca477e04473c55e5e7d73bfe2897f3c6176cab4d0293abf1f4bc9c97b6d"),
    "evaluation/medical/judge_checkpoint.json.002": (0o400, 10226, "a7dfb13ae13b528127f204e9d67f23943d06046d4b8af243f36127da5bbbeb78", "8c1bc97987a6a1acceb47bc5eb9f206444e36d0c51fe6b94244b98a2f0734cc1"),
    "evaluation/medical/judge_checkpoint.json.240": (0o400, 253288, "cc402cfc7509c5b1c77d089ca55b159a82013539b1e66ba08673558b2cf72b58", "e9b70c9ba4bb63aaada60780f02cf08d6ed650cd021bd49591e083ea511b7e6c"),
    "evaluation/medical/judgments_new.json": (0o400, 253560, "1232f65e904b0429f4cc039d846ea5dd4a684a2e7b040298a1728397973133cb", "66e9758b11ffc75e389ff2dc3972c7655823a13fa2a15836bff38288cc184144"),
    "logs/external_judge_canary.log": (0o400, 171, "12bf3b47e6907471304a66d8820d3b3b04be1c14045e5bd5981938ae2512f6b5", None),
    "logs/external_judge_continuation.log": (0o400, 8879, "392dbabb5539d4415e898399d00a693d8ac1672f4b80b41110003b57412ad324", None),
}

COST_ORDER_RECOVERY = {
    "bug_class": "floating_point_nonassociativity_after_presentation_sort",
    "chronological_cost_usd": 0.031268499999999984,
    "sorted_presentation_cost_usd": 0.0312685,
    "chronological_minus_sorted_usd": -1.3877787807814457e-17,
    "new_v6_chronological_cost_usd": 0.03115399999999998,
    "presentation_rows_equal_sorted_chronological_rows": True,
    "chronological_rows_equal_presentation_rows": False,
    "repair": "sum_checkpoint_chronology_before_sorting_rows_for_presentation",
}


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", os.fspath(root), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _require_private_directory(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is absent or unsafe")
    if stat.S_IMODE(path.stat().st_mode) != 0o2700:
        raise ValueError(f"{label} mode differs")


def _record(path, expected):
    mode, size, digest, payload_digest = expected
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Frozen source file is absent or unsafe: {path}")
    if (
        stat.S_IMODE(path.stat().st_mode) != mode
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise ValueError(f"Frozen source file binding differs: {path}")
    payload = None
    if payload_digest is not None:
        payload = load_json(path)
        audit_seal(payload, os.fspath(path))
        if payload.get("payload_sha256") != payload_digest:
            raise ValueError(f"Frozen source payload binding differs: {path}")
    return binding(path, payload)


def _source_inventory_stream():
    entries = []
    file_count = 0
    for path in SOURCE_OUTPUT.rglob("*"):
        relative = path.relative_to(SOURCE_OUTPUT)
        if len(relative.parts) > 3:
            raise ValueError(f"V6 source inventory extends below frozen depth: {relative}")
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"V6 source inventory contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            kind = "d"
        elif stat.S_ISREG(info.st_mode):
            kind = "f"
            file_count += 1
        else:
            raise ValueError(f"V6 source inventory contains an unsafe type: {relative}")
        entries.append(f"{kind} {stat.S_IMODE(info.st_mode):o} {relative.as_posix()}\n")
    stream = "".join(sorted(entries)).encode()
    return file_count, hashlib.sha256(stream).hexdigest()


def audit_repo():
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise ValueError("Recovery-v7 repository is absent or unsafe")
    commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    if (
        _git(REPO_ROOT, "branch", "--show-current") != BRANCH
        or _git(REPO_ROOT, "status", "--porcelain")
        or _git(REPO_ROOT, "rev-list", "--parents", "-n", "1", commit)
        != f"{commit} {SOURCE_COMMIT}"
        or _git(REPO_ROOT, "rev-parse", f"{SOURCE_COMMIT}^{{tree}}") != SOURCE_TREE
    ):
        raise ValueError("Recovery-v7 repository lineage differs")
    observed = []
    for line in _git(
        REPO_ROOT, "diff", "--name-status", "--no-renames", f"{SOURCE_COMMIT}..{commit}"
    ).splitlines():
        if line:
            observed.append(tuple(line.split("\t")))
    if sorted(observed) != sorted(("A", path) for path in ADDED_FILES):
        raise ValueError("Recovery-v7 repository add-only scope differs")
    for relative in ADDED_FILES:
        entry = _git(REPO_ROOT, "ls-files", "-s", "--", relative).split()
        expected_mode = "100755" if relative.startswith("scripts/") else "100644"
        if len(entry) != 4 or entry[0] != expected_mode or entry[2] != "0":
            raise ValueError(f"Recovery-v7 tracked mode differs: {relative}")
    return {
        "path": os.fspath(REPO_ROOT), "branch": BRANCH, "commit": commit,
        "tree": _git(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
        "source_commit": SOURCE_COMMIT, "source_commit_is_direct_parent": True,
        "add_only_files": list(ADDED_FILES),
    }


def _audit_source_repo():
    if SOURCE_REPO.is_symlink() or not SOURCE_REPO.is_dir():
        raise ValueError("Source v6 repository is absent or unsafe")
    if (
        _git(SOURCE_REPO, "rev-parse", "HEAD") != SOURCE_COMMIT
        or _git(SOURCE_REPO, "rev-parse", "HEAD^{tree}") != SOURCE_TREE
        or _git(SOURCE_REPO, "rev-parse", "HEAD^") != SOURCE_PARENT
        or _git(SOURCE_REPO, "branch", "--show-current") != SOURCE_BRANCH
        or _git(SOURCE_REPO, "status", "--porcelain")
    ):
        raise ValueError("Source v6 repository binding differs")
    return {
        "path": os.fspath(SOURCE_REPO), "branch": SOURCE_BRANCH,
        "commit": SOURCE_COMMIT, "tree": SOURCE_TREE,
    }


def audit_source_v6():
    """Audit and return the exact immutable v6 terminal source."""
    source_repo = _audit_source_repo()
    _require_private_directory(SOURCE_OUTPUT, "Source v6 output")
    for relative in ("control", "evaluation", "evaluation/medical", "evaluation/final", "logs"):
        _require_private_directory(SOURCE_OUTPUT / relative, f"Source v6 {relative}")

    expected_control = {
        "JUDGE_RECOVERY_V6_MANIFEST.json", "PREP.json", "CPU_PREFLIGHT.json", "STAGED",
        "CANARY_LOCK_OWNER.json", "CANARY_AUTHORIZATION.json", "CANARY_SUCCESS.json",
        "CONTINUATION_LOCK_OWNER.json", "CONTINUATION_AUTHORIZATION.json",
        "CONTINUATION_SUCCESS.json",
    }
    expected_medical = {"judgments_new.json"} | {
        f"judge_checkpoint.json.{completed:03d}" for completed in range(2, 241)
    }
    expected_logs = {"external_judge_canary.log", "external_judge_continuation.log"}
    if (
        {p.name for p in (SOURCE_OUTPUT / "control").iterdir()} != expected_control
        or {p.name for p in (SOURCE_OUTPUT / "evaluation/medical").iterdir()} != expected_medical
        or {p.name for p in (SOURCE_OUTPUT / "evaluation/final").iterdir()}
        or {p.name for p in (SOURCE_OUTPUT / "logs").iterdir()} != expected_logs
        or {p.name for p in SOURCE_OUTPUT.iterdir()} != {"control", "evaluation", "logs"}
        or {p.name for p in (SOURCE_OUTPUT / "evaluation").iterdir()} != {"medical", "final"}
    ):
        raise ValueError("Source v6 exact terminal inventory differs")
    file_count, stream_digest = _source_inventory_stream()
    if (
        file_count != SOURCE_INVENTORY_FILE_COUNT
        or stream_digest != SOURCE_INVENTORY_STREAM_SHA256
    ):
        raise ValueError("Source v6 name/mode/type inventory commitment differs")

    records = {
        relative: _record(SOURCE_OUTPUT / relative, expected)
        for relative, expected in SOURCE_CRITICAL_FILES.items()
    }
    source_manifest_path = SOURCE_OUTPUT / "control/JUDGE_RECOVERY_V6_MANIFEST.json"
    recovery = source_judge.load_recovery_manifest(source_manifest_path)
    inputs = source_judge.validate_source_inputs(recovery)
    paths = source_judge.recovery_paths(recovery)
    # This validates all 239 cumulative checkpoints, both authorities, the
    # successful canary and continuation, plan identities, costs and seals.
    source_judge.audit_continuation_command(
        argparse.Namespace(recovery_manifest=os.fspath(source_manifest_path))
    )

    checkpoint = load_json(paths["checkpoint_base"] + ".240")
    checkpoint_body = audit_seal(checkpoint, paths["checkpoint_base"] + ".240")
    chronological_rows = checkpoint_body.get("judgments")
    judgments_payload = load_json(paths["judgments"])
    judgments_body = audit_seal(judgments_payload, paths["judgments"])
    presentation_rows = judgments_body.get("judgments")
    if not isinstance(chronological_rows, list) or len(chronological_rows) != 240:
        raise ValueError("Source v6 chronological terminal rows differ")
    sorted_rows = sorted(
        chronological_rows,
        key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]),
    )
    if presentation_rows != sorted_rows or chronological_rows == presentation_rows:
        raise ValueError("Source v6 chronological/presentation ordering evidence differs")
    chronological_cost = sum(
        row["api_usage"]["estimated_cost_usd"] for row in chronological_rows
    )
    sorted_cost = sum(row["api_usage"]["estimated_cost_usd"] for row in presentation_rows)
    canary_payload = load_json(paths["canary_success"])
    canary_body = audit_seal(canary_payload, paths["canary_success"])
    new_v6_cost = canary_body["stage_actual_estimated_cost_usd"] + sum(
        row["api_usage"]["estimated_cost_usd"] for row in chronological_rows[2:]
    )
    observed_cost = {
        **COST_ORDER_RECOVERY,
        "chronological_cost_usd": chronological_cost,
        "sorted_presentation_cost_usd": sorted_cost,
        "chronological_minus_sorted_usd": chronological_cost - sorted_cost,
        "new_v6_chronological_cost_usd": new_v6_cost,
    }
    if observed_cost != COST_ORDER_RECOVERY:
        raise ValueError("Source v6 exact cost-order mismatch evidence differs")
    meta = judgments_body.get("meta")
    continuation_body = audit_seal(
        load_json(paths["continuation_success"]), paths["continuation_success"]
    )
    if (
        not isinstance(meta, dict)
        or meta.get("actual_estimated_cost_usd") != chronological_cost
        or meta.get("new_v6_estimated_cost_usd") != new_v6_cost
        or continuation_body.get("cumulative_rows_estimated_cost_usd") != chronological_cost
    ):
        raise ValueError("Source v6 sealed chronological cost contract differs")

    terminal_bindings = {
        "canary_success": records["control/CANARY_SUCCESS.json"],
        "continuation_authorization": records["control/CONTINUATION_AUTHORIZATION.json"],
        "continuation_success": records["control/CONTINUATION_SUCCESS.json"],
        "checkpoint_002": records["evaluation/medical/judge_checkpoint.json.002"],
        "checkpoint_240": records["evaluation/medical/judge_checkpoint.json.240"],
        "judgments_new": records["evaluation/medical/judgments_new.json"],
    }
    return {
        "repo": source_repo, "manifest": recovery, "inputs": inputs, "paths": paths,
        "chronological_rows": chronological_rows, "presentation_rows": presentation_rows,
        "terminal_bindings": terminal_bindings,
        "cost_order_recovery": dict(COST_ORDER_RECOVERY),
    }


def manifest_body(repo, source):
    source_body = source["manifest"]["body"]
    return {
        "schema_version": 1,
        "protocol": RECOVERY_ID + "_manifest_v1",
        "recovery_id": RECOVERY_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_v6_recovery_id": SOURCE_V6_RECOVERY_ID,
        "recovery_repo": repo,
        "source_v6_repo": source["repo"],
        "source_v6_output_root": os.fspath(SOURCE_OUTPUT),
        "recovery_output_root": os.fspath(RECOVERY_OUTPUT),
        "source_v6_manifest": binding(
            source["manifest"]["path"], load_json(source["manifest"]["path"])
        ),
        "source_v6_terminal": {
            "inventory_name_mode_type_stream_sha256": SOURCE_INVENTORY_STREAM_SHA256,
            "terminal_file_count": SOURCE_INVENTORY_FILE_COUNT,
            **source["terminal_bindings"],
        },
        "source_protocol_manifest": source_body["source_protocol_manifest"],
        "source_judge_plan": source_body["source_judge_plan"],
        "source_artifacts": source_body["source_artifacts"],
        "source_v6_budget_contract": source_body["budget_contract"],
        "cost_order_recovery": source["cost_order_recovery"],
        "derivation_contract": {
            "source_v6_read_only": True,
            "fresh_v7_output_namespace": True,
            "sum_checkpoint_chronology_before_presentation_sort": True,
            "presentation_row_order_unchanged": True,
            "judgments_reused": 240,
            "historical_A_reused_not_rejudged": True,
            "new_external_api_calls": 0,
            "new_gpu_jobs": 0,
            "idempotent_existing_artifacts_must_match_exactly": True,
            "partial_or_unknown_inventory_fails_closed": True,
        },
        "external_api_authorized": False,
        "external_api_calls": 0,
        "gpu_authorized": False,
        "gpu_jobs": 0,
        "cpu_derivation_only": True,
    }


def derive_paths(manifest):
    root = os.path.abspath(manifest["body"]["recovery_output_root"])
    if root != os.fspath(RECOVERY_OUTPUT) or os.path.realpath(root) != root:
        raise ValueError("Recovery-v7 output namespace differs")
    control_root = os.path.join(root, "control")
    medical = os.path.join(root, "evaluation", "medical")
    final = os.path.join(root, "evaluation", "final")
    return {
        "root": root, "control": control_root, "medical": medical, "final": final,
        "logs": os.path.join(root, "logs"),
        "manifest": os.path.join(control_root, MANIFEST_FILE.name),
        "prep": os.path.join(control_root, PREP_FILE.name),
        "preflight": os.path.join(control_root, PREFLIGHT_FILE.name),
        "staged": os.path.join(control_root, STAGED_FILE.name),
        "derivation_lock": os.path.join(control_root, LOCK_FILE.name),
        "final_result": os.path.join(control_root, FINAL_RESULT_FILE.name),
        "judgments_merged": os.path.join(medical, "judgments_merged.json"),
        "summary": os.path.join(final, "summary.json"),
    }


def load_manifest(path=None):
    path = Path(path or MANIFEST_FILE).absolute()
    if path != MANIFEST_FILE or path.is_symlink() or not path.is_file():
        raise ValueError("Recovery-v7 manifest path differs")
    payload = load_json(path)
    body = audit_seal(payload, os.fspath(path))
    if body.get("recovery_id") != RECOVERY_ID:
        raise ValueError("Recovery-v7 manifest identity differs")
    return {"path": os.fspath(path), "payload": payload, "body": body,
            "payload_sha256": payload["payload_sha256"]}


def _namespace(expected):
    _require_private_directory(RECOVERY_OUTPUT, "Recovery-v7 output")
    for relative in ("control", "evaluation", "evaluation/medical", "evaluation/final", "logs"):
        _require_private_directory(RECOVERY_OUTPUT / relative, f"Recovery-v7 {relative}")
    if (
        {p.name for p in RECOVERY_OUTPUT.iterdir()} != {"control", "evaluation", "logs"}
        or {p.name for p in (RECOVERY_OUTPUT / "evaluation").iterdir()} != {"medical", "final"}
    ):
        raise ValueError("Recovery-v7 directory inventory differs")
    for relative, names in expected.items():
        directory = RECOVERY_OUTPUT / relative
        observed = {p.name for p in directory.iterdir()}
        if observed != set(names) or any(p.is_symlink() for p in directory.iterdir()):
            raise ValueError(f"Recovery-v7 exact inventory differs: {relative}")


def _stage_files():
    return {
        "control": {MANIFEST_FILE.name, PREP_FILE.name, PREFLIGHT_FILE.name, STAGED_FILE.name},
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    }


def _expected_prep(manifest):
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + "_prep_v1",
        "recovery_id": RECOVERY_ID,
        "derive_manifest": binding(MANIFEST_FILE, load_json(MANIFEST_FILE)),
        "source_v6_inventory_sha256": SOURCE_INVENTORY_STREAM_SHA256,
        "external_api_calls": 0, "gpu_jobs": 0,
        "status": "CPU_PREPARED_AWAITING_VALIDATION",
    }


def audit_manifest_exact(path=None):
    manifest = load_manifest(path)
    expected = manifest_body(audit_repo(), audit_source_v6())
    if manifest["body"] != expected:
        raise ValueError("Recovery-v7 manifest body differs")
    return manifest


def prepare_command(_args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU preparation")
    if os.path.lexists(RECOVERY_OUTPUT):
        raise FileExistsError("Recovery-v7 output namespace already exists")
    repo = audit_repo()
    source = audit_source_v6()
    for directory in (CONTROL_ROOT, MEDICAL_ROOT, FINAL_ROOT, LOG_ROOT):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o2700)
    os.chmod(RECOVERY_OUTPUT / "evaluation", 0o2700)
    os.chmod(RECOVERY_OUTPUT, 0o2700)
    atomic_json(MANIFEST_FILE, seal(manifest_body(repo, source)))
    manifest = load_manifest()
    atomic_json(PREP_FILE, seal(_expected_prep(manifest)))
    _namespace({
        "control": {MANIFEST_FILE.name, PREP_FILE.name},
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    })
    print(json.dumps({
        "status": "JUDGE_DERIVE_RECOVERY_V7_CPU_PREPARED",
        "external_api_calls": 0, "gpu_jobs": 0,
        "manifest_payload_sha256": manifest["payload_sha256"],
    }, sort_keys=True))
    return 0


def seal_staged_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU staging")
    manifest = audit_manifest_exact()
    _namespace({
        "control": {MANIFEST_FILE.name, PREP_FILE.name},
        "evaluation/medical": set(), "evaluation/final": set(), "logs": set(),
    })
    prep = audit_seal(load_json(PREP_FILE), os.fspath(PREP_FILE))
    if prep != _expected_prep(manifest):
        raise ValueError("Recovery-v7 prep differs")
    commands = args.validation_command or []
    if not commands:
        raise ValueError("At least one successful validation command is required")
    preflight = seal({
        "schema_version": 1, "protocol": RECOVERY_ID + "_cpu_preflight_v1",
        "recovery_id": RECOVERY_ID,
        "derive_manifest": binding(MANIFEST_FILE, load_json(MANIFEST_FILE)),
        "validation_commands_passed": commands,
        "network_validation": "none_cpu_only",
        "external_api_calls": 0, "gpu_jobs": 0, "api_key_required": False,
        "status": "CPU_VALIDATED_AWAITING_DERIVATION",
    })
    atomic_json(PREFLIGHT_FILE, preflight)
    staged = seal({
        "schema_version": 1, "protocol": RECOVERY_ID + "_staged_v1",
        "recovery_id": RECOVERY_ID,
        "derive_manifest": binding(MANIFEST_FILE, load_json(MANIFEST_FILE)),
        "cpu_preflight": binding(PREFLIGHT_FILE, preflight),
        "source_v6_read_only": True,
        "external_api_authorized": False, "external_api_calls": 0,
        "gpu_authorized": False, "gpu_jobs": 0,
        "next_stage": "CPU_ONLY_IDEMPOTENT_DERIVATION",
    })
    atomic_json(STAGED_FILE, staged)
    audit_staged()
    print(json.dumps({
        "status": "JUDGE_DERIVE_RECOVERY_V7_CPU_STAGED",
        "external_api_calls": 0, "gpu_jobs": 0,
        "next_stage": "CPU_ONLY_IDEMPOTENT_DERIVATION",
    }, sort_keys=True))
    return 0


def _audit_stage_records(manifest):
    """Audit immutable stage records without constraining later inventories."""
    for path in (MANIFEST_FILE, PREP_FILE, PREFLIGHT_FILE, STAGED_FILE):
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise ValueError(f"Recovery-v7 staged control mode differs: {path}")
    prep = audit_seal(load_json(PREP_FILE), os.fspath(PREP_FILE))
    preflight_payload = load_json(PREFLIGHT_FILE)
    preflight = audit_seal(preflight_payload, os.fspath(PREFLIGHT_FILE))
    staged = audit_seal(load_json(STAGED_FILE), os.fspath(STAGED_FILE))
    if prep != _expected_prep(manifest):
        raise ValueError("Recovery-v7 prep differs")
    if (
        set(preflight) != {
            "schema_version", "protocol", "recovery_id", "derive_manifest",
            "validation_commands_passed", "network_validation", "external_api_calls",
            "gpu_jobs", "api_key_required", "status",
        }
        or preflight.get("protocol") != RECOVERY_ID + "_cpu_preflight_v1"
        or preflight.get("recovery_id") != RECOVERY_ID
        or preflight.get("derive_manifest") != binding(MANIFEST_FILE, load_json(MANIFEST_FILE))
        or not isinstance(preflight.get("validation_commands_passed"), list)
        or not preflight["validation_commands_passed"]
        or preflight.get("network_validation") != "none_cpu_only"
        or preflight.get("external_api_calls") != 0
        or preflight.get("gpu_jobs") != 0
        or preflight.get("api_key_required") is not False
        or preflight.get("status") != "CPU_VALIDATED_AWAITING_DERIVATION"
    ):
        raise ValueError("Recovery-v7 CPU preflight differs")
    expected_staged = {
        "schema_version": 1, "protocol": RECOVERY_ID + "_staged_v1",
        "recovery_id": RECOVERY_ID,
        "derive_manifest": binding(MANIFEST_FILE, load_json(MANIFEST_FILE)),
        "cpu_preflight": binding(PREFLIGHT_FILE, preflight_payload),
        "source_v6_read_only": True,
        "external_api_authorized": False, "external_api_calls": 0,
        "gpu_authorized": False, "gpu_jobs": 0,
        "next_stage": "CPU_ONLY_IDEMPOTENT_DERIVATION",
    }
    if staged != expected_staged:
        raise ValueError("Recovery-v7 staged sentinel differs")


def audit_staged():
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU staging audit")
    manifest = audit_manifest_exact()
    _namespace(_stage_files())
    _audit_stage_records(manifest)
    return manifest


def _lock_body():
    return {
        "schema_version": 1, "protocol": RECOVERY_ID + "_derivation_lock_v1",
        "recovery_id": RECOVERY_ID,
        "derive_manifest": binding(MANIFEST_FILE, load_json(MANIFEST_FILE)),
        "source_v6_inventory_sha256": SOURCE_INVENTORY_STREAM_SHA256,
        "cpu_only": True, "external_api_calls": 0, "gpu_jobs": 0,
        "idempotent_reentry_allowed_only_for_exact_existing_artifacts": True,
    }


def audit_derive_namespace():
    """Audit the allowed deterministic prefix of a v7 derivation."""
    manifest = audit_manifest_exact()
    control_names = {p.name for p in CONTROL_ROOT.iterdir()}
    medical_names = {p.name for p in MEDICAL_ROOT.iterdir()}
    final_names = {p.name for p in FINAL_ROOT.iterdir()}
    allowed_control = {
        MANIFEST_FILE.name, PREP_FILE.name, PREFLIGHT_FILE.name, STAGED_FILE.name,
        LOCK_FILE.name, FINAL_RESULT_FILE.name,
    }
    if (
        not {MANIFEST_FILE.name, PREP_FILE.name, PREFLIGHT_FILE.name, STAGED_FILE.name, LOCK_FILE.name}
        <= control_names
        or not control_names <= allowed_control
        or not medical_names <= {"judgments_merged.json"}
        or not final_names <= {
            "summary.json", "EXPLORATORY_SEQUENTIAL_SUPPORT",
            "EXPLORATORY_SEQUENTIAL_NO_SUPPORT",
        }
        or len(final_names & {
            "EXPLORATORY_SEQUENTIAL_SUPPORT", "EXPLORATORY_SEQUENTIAL_NO_SUPPORT"
        }) > 1
        or {p.name for p in LOG_ROOT.iterdir()}
    ):
        raise ValueError("Recovery-v7 partial derivation inventory differs")
    for directory in (CONTROL_ROOT, MEDICAL_ROOT, FINAL_ROOT, LOG_ROOT):
        if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
            raise ValueError("Recovery-v7 derivation contains an unsafe artifact")
    _audit_stage_records(manifest)
    lock = audit_seal(load_json(LOCK_FILE), os.fspath(LOCK_FILE))
    if lock != _lock_body():
        raise ValueError("Recovery-v7 derivation lock differs")
    # FINAL_RESULT is the commit marker and may only appear after every output.
    if FINAL_RESULT_FILE.name in control_names and (
        medical_names != {"judgments_merged.json"}
        or len(final_names) != 2
        or "summary.json" not in final_names
    ):
        raise ValueError("Recovery-v7 final commit marker appeared before complete outputs")
    return {
        "control": sorted(control_names), "medical": sorted(medical_names),
        "final": sorted(final_names),
    }


def acquire_lock_command(_args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU derivation")
    if os.path.lexists(LOCK_FILE):
        audit_derive_namespace()
        print("JUDGE_DERIVE_RECOVERY_V7_LOCK_ALREADY_VALID")
        return 0
    audit_staged()
    atomic_json(LOCK_FILE, seal(_lock_body()), mode=0o400)
    audit_derive_namespace()
    print("JUDGE_DERIVE_RECOVERY_V7_CPU_DERIVATION_LOCKED")
    return 0


def audit_complete_namespace():
    state = audit_derive_namespace()
    if (
        FINAL_RESULT_FILE.name not in state["control"]
        or state["medical"] != ["judgments_merged.json"]
        or len(state["final"]) != 2
        or "summary.json" not in state["final"]
    ):
        raise ValueError("Recovery-v7 derivation is incomplete")
    for path in (
        LOCK_FILE, FINAL_RESULT_FILE, MEDICAL_ROOT / "judgments_merged.json",
        *(FINAL_ROOT / name for name in state["final"]),
    ):
        if stat.S_IMODE(path.stat().st_mode) != 0o400:
            raise ValueError(f"Recovery-v7 sealed output mode differs: {path}")
    return state


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(handler=prepare_command)
    staged = commands.add_parser("seal-staged")
    staged.add_argument("--validation-command", action="append")
    staged.set_defaults(handler=seal_staged_command)
    commands.add_parser("audit-staged").set_defaults(handler=lambda _args: (audit_staged(), 0)[1])
    commands.add_parser("acquire-derive-lock").set_defaults(handler=acquire_lock_command)
    commands.add_parser("audit-derive-state").set_defaults(
        handler=lambda _args: (audit_derive_namespace(), 0)[1]
    )
    commands.add_parser("audit-complete").set_defaults(
        handler=lambda _args: (audit_complete_namespace(), 0)[1]
    )
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


def main():
    try:
        raise SystemExit(run())
    except (ValueError, FileExistsError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
