#!/usr/bin/env python3
"""Freeze the repaired pilot checkpoint using APPS validation only.

LiveCodeBench and EvalPlus must not be consulted before this file is written.
The selected checkpoint maximizes APPS validation passes; ties minimize empty
or truncated generations and then prefer the earlier training step.
"""

import argparse
import datetime
import hashlib
import json
import os
import tempfile


BASE_NAME = "pi_base"
CHECKPOINTS = ("step_10", "step_20", "step_30", "step_40")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
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
    parser.add_argument("--apps-summary", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--expected-problems", type=int, default=200)
    args = parser.parse_args()

    with open(args.apps_summary, encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(args.model_manifest, encoding="utf-8") as handle:
        model_manifest = json.load(handle)
    checkpoint_seals = model_manifest.get("checkpoints")
    if not isinstance(checkpoint_seals, dict) or set(checkpoint_seals) != set(CHECKPOINTS):
        raise ValueError("Model manifest does not seal all four trajectory checkpoints")
    selected_adapter_sha256 = {}
    for name in CHECKPOINTS:
        files = checkpoint_seals[name].get("files")
        by_basename = {
            os.path.basename(item.get("path", "")): item.get("sha256")
            for item in files or []
        }
        config_hash = by_basename.get("adapter_config.json")
        weight_items = {
            key: value
            for key, value in by_basename.items()
            if key in {"adapter_model.safetensors", "adapter_model.bin"}
        }
        if not isinstance(config_hash, str) or len(weight_items) != 1:
            raise ValueError(f"Model manifest has an incomplete adapter seal for {name}")
        selected_adapter_sha256[name] = {
            "adapter_config_sha256": config_hash,
            "adapter_weights_filename": next(iter(weight_items)),
            "adapter_weights_sha256": next(iter(weight_items.values())),
        }
    models = summary.get("models")
    if not isinstance(models, dict):
        raise ValueError("APPS summary has no models mapping")
    expected = {BASE_NAME, *CHECKPOINTS}
    if set(models) != expected:
        raise ValueError(
            f"APPS summary model set must be exactly {sorted(expected)}; "
            f"got {sorted(models)}"
        )
    if summary.get("meta", {}).get("n_questions") != args.expected_problems:
        raise ValueError("APPS summary does not contain the preregistered problem count")

    audited = {}
    for name in expected:
        row = models[name]
        if row.get("n") != args.expected_problems:
            raise ValueError(f"Unexpected APPS count for {name}: {row.get('n')}")
        for field in ("passed", "empty_extractions", "truncations"):
            if not isinstance(row.get(field), int) or row[field] < 0:
                raise ValueError(f"Invalid {field} for {name}: {row.get(field)!r}")
        audited[name] = {
            field: row[field]
            for field in ("passed", "empty_extractions", "truncations")
        }

    def ranking(name):
        row = audited[name]
        step = int(name.split("_", 1)[1])
        malformed = row["empty_extractions"] + row["truncations"]
        return (-row["passed"], malformed, step)

    selected = min(CHECKPOINTS, key=ranking)
    source_sha = sha256_file(args.apps_summary)
    immutable = {
        "schema_version": 1,
        "selection_suite": "APPS repaired-pilot validation",
        "selection_rule": (
            "maximize passed; then minimize empty_extractions+truncations; "
            "then choose earliest checkpoint"
        ),
        "selection_candidates": list(CHECKPOINTS),
        "selected_checkpoint": selected,
        "selected_step": int(selected.split("_", 1)[1]),
        "selected_adapter": selected_adapter_sha256[selected],
        "base_is_not_selectable": True,
        "apps_summary_sha256": source_sha,
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_problems": args.expected_problems,
        "audited_models": audited,
        "prohibited_selection_suites": ["LiveCodeBench", "HumanEval+", "MBPP+"],
        "automatic_continuation": False,
    }

    if os.path.isfile(args.output_file):
        with open(args.output_file, encoding="utf-8") as handle:
            existing = json.load(handle)
        observed = dict(existing)
        observed.pop("created_at", None)
        if observed != immutable:
            raise ValueError(
                "Existing checkpoint selection does not match the current sealed APPS summary"
            )
        print(json.dumps(existing, indent=2))
        return

    output = {
        **immutable,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_write_json(args.output_file, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
