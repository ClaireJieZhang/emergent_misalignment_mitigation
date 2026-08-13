#!/usr/bin/env python3
"""Create or verify the immutable public-asset seal for the diagnostic."""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile


EVALPLUS_COMMIT = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
EXPECTED_FILES = {
    "HumanEvalPlus-v0.1.10.jsonl.gz": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
    "MbppPlus-v0.2.0.jsonl.gz": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
}
SIF_SHA256 = "d651ca156e0d54c3dd3a1ba48d5372e581648a15b18f37455c887531b2d25fd4"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"Sealed asset must be a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SITE_SEAL_NAME = ".site_manifest.json"


def tree_manifest(path, exclude_names=(), exclude_file_names=()):
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise ValueError(f"Missing sealed asset directory: {root}")
    records = []
    excluded = set(exclude_names)
    excluded_files = set(exclude_file_names)
    for directory, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            candidate = os.path.join(directory, dirname)
            if os.path.islink(candidate):
                raise ValueError(f"Symlink is forbidden in sealed asset tree: {candidate}")
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)
        for filename in sorted(filenames):
            if filename in excluded_files:
                continue
            candidate = os.path.join(directory, filename)
            relative = os.path.relpath(candidate, root)
            records.append(
                {
                    "path": relative,
                    "size": os.path.getsize(candidate),
                    "sha256": sha256_file(candidate),
                }
            )
    if not records:
        raise ValueError(f"Empty sealed asset directory: {root}")
    return {
        "path": root,
        "n_files": len(records),
        "n_bytes": sum(row["size"] for row in records),
        "tree_sha256": sha256_bytes(canonical_json(records).encode("utf-8")),
    }


def audit_site(path, create=False):
    root = os.path.abspath(path)
    seal_path = os.path.join(root, SITE_SEAL_NAME)
    observed = tree_manifest(root, exclude_file_names={SITE_SEAL_NAME})
    # The sealed site is prepared under an atomic build path and then renamed.
    # Make its identity independent of that containing directory.
    observed["path"] = "."
    observed["seal_payload_sha256"] = sha256_bytes(
        canonical_json(observed).encode("utf-8")
    )
    if create:
        if os.path.exists(seal_path):
            raise ValueError(f"Refusing to replace existing site seal: {seal_path}")
        atomic_write(seal_path, observed)
    with open(seal_path, encoding="utf-8") as handle:
        sealed = json.load(handle)
    if sealed != observed:
        raise ValueError(f"EvalPlus dependency-site seal mismatch: {root}")
    return observed


def git_identity(path):
    root = os.path.abspath(path)
    commit = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", root, "status", "--porcelain"], text=True
    )
    if commit != EVALPLUS_COMMIT or status:
        raise ValueError("EvalPlus source is not the clean pinned v0.3.1 commit")
    return commit


def build_payload(asset_root, source_root):
    asset_root = os.path.abspath(asset_root)
    source_root = os.path.abspath(source_root)
    datasets = {}
    for name, expected in EXPECTED_FILES.items():
        path = os.path.join(asset_root, name)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Pinned public asset SHA-256 mismatch: {path}")
        datasets[name] = {"path": path, "sha256": observed}
    sif_path = os.path.join(source_root, "assets", "python-3.11-slim-amd64.sif")
    if sha256_file(sif_path) != SIF_SHA256:
        raise ValueError("Sandbox SIF SHA-256 mismatch")
    evalplus_repo = os.path.join(asset_root, "evalplus-v0.3.1")
    git_identity(evalplus_repo)
    evalplus_site_path = os.path.join(asset_root, "evalplus-python-site")
    evalplus_site = audit_site(evalplus_site_path)
    evalplus_site["path"] = evalplus_site_path
    return {
        "schema_version": 1,
        "evalplus_commit": EVALPLUS_COMMIT,
        "datasets": datasets,
        "sandbox_sif": {"path": sif_path, "sha256": SIF_SHA256},
        "evalplus_source": tree_manifest(evalplus_repo, exclude_names={".git"}),
        "evalplus_site": evalplus_site,
        "lcb_dependency_site": tree_manifest(
            os.path.join(source_root, "assets", "lcb-python-site")
        ),
    }


def atomic_write(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root")
    parser.add_argument("--source_root")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--site")
    parser.add_argument("--create-site-seal", action="store_true")
    args = parser.parse_args()
    if args.site:
        if args.asset_root or args.source_root or args.create:
            parser.error("--site cannot be combined with asset-manifest arguments")
        audited = audit_site(args.site, create=args.create_site_seal)
        print(
            f"EvalPlus dependency-site audit passed: {os.path.abspath(args.site)} "
            f"({audited['seal_payload_sha256']})"
        )
        return
    if args.create_site_seal:
        parser.error("--create-site-seal requires --site")
    if not args.asset_root or not args.source_root:
        parser.error("asset-manifest audit requires --asset_root and --source_root")
    manifest_path = os.path.join(os.path.abspath(args.asset_root), "asset_manifest.json")
    observed = build_payload(args.asset_root, args.source_root)
    if args.create:
        if os.path.exists(manifest_path):
            raise ValueError(f"Refusing to replace existing asset seal: {manifest_path}")
        sealed_payload = {**observed, "manifest_payload_sha256": sha256_bytes(
            canonical_json(observed).encode("utf-8")
        )}
        atomic_write(manifest_path, sealed_payload)
    with open(manifest_path, encoding="utf-8") as handle:
        sealed = json.load(handle)
    seal = sealed.pop("manifest_payload_sha256", None)
    expected_seal = sha256_bytes(canonical_json(sealed).encode("utf-8"))
    if seal != expected_seal or sealed != observed:
        raise ValueError("EvalPlus public-asset seal mismatch")
    print(f"EvalPlus asset audit passed: {manifest_path} ({expected_seal})")


if __name__ == "__main__":
    main()
