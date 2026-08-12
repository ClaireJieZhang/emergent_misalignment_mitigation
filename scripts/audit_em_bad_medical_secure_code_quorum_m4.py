#!/usr/bin/env python3
"""Audit staged full-run artifacts for the m=4 capability-quorum experiment."""

import argparse
import json
import os
import re
import sys
from pathlib import Path


GOOD_NAMES = ("pi_good_0", "pi_good_1", "pi_good_2")
MODEL_NAMES = ("pi_base", "pi_bad", *GOOD_NAMES)
DOMAIN_SETTINGS = {
    "broad": {"prompts": 24, "samples": 5},
    "narrow_medical": {"prompts": 16, "samples": 5},
    "secure_code": {"prompts": 8, "samples": 10},
}
EXPECTED_REFS = ("pi_bad", *GOOD_NAMES)
EXPECTED_SEEDS = (0, 101, 202)


class Audit:
    def __init__(self):
        self.errors = []

    def require(self, condition, message):
        if not condition:
            self.errors.append(message)

    def json(self, path):
        path = Path(path)
        if not path.is_file():
            self.errors.append(f"missing file: {path}")
            return None
        try:
            with path.open() as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self.errors.append(f"invalid JSON: {path}: {exc}")
            return None

    def nonempty(self, path):
        path = Path(path)
        self.require(path.is_file() and path.stat().st_size > 0, f"missing/empty file: {path}")

    def finish(self, label):
        if self.errors:
            print(f"AUDIT FAILED: {label}", file=sys.stderr)
            for error in self.errors:
                print(f"- {error}", file=sys.stderr)
            raise SystemExit(1)
        print(f"AUDIT PASSED: {label}")


def model_basename(value):
    return str(value or "").rstrip("/").split("/")[-1]


def audit_adapter(audit, path, expected_base, label):
    path = Path(path)
    config = audit.json(path / "adapter_config.json")
    audit.nonempty(path / "adapter_model.safetensors")
    if config is None:
        return
    audit.require(config.get("peft_type") == "LORA", f"{label}: peft_type is not LORA")
    audit.require(config.get("r") == 8, f"{label}: expected LoRA rank 8, got {config.get('r')!r}")
    audit.require(
        config.get("lora_alpha") == 8,
        f"{label}: expected LoRA alpha 8, got {config.get('lora_alpha')!r}",
    )
    actual_base = config.get("base_model_name_or_path")
    audit.require(
        model_basename(actual_base) == model_basename(expected_base),
        f"{label}: base model {actual_base!r} does not match {expected_base!r}",
    )


def audit_models(audit, output_root, bad_model, expected_base):
    output_root = Path(output_root)
    audit_adapter(audit, bad_model, expected_base, "pi_bad")
    for index, name in enumerate(GOOD_NAMES):
        model_dir = output_root / "models" / name
        audit_adapter(audit, model_dir, expected_base, name)
        summary = audit.json(model_dir / "training_summary.json")
        if summary is not None:
            audit.require(summary.get("kind") == "sft", f"{name}: training kind is not sft")
            audit.require(summary.get("n_examples") == 6000, f"{name}: n_examples is not 6000")
            audit.require(summary.get("max_steps") == 300, f"{name}: max_steps is not 300")
            audit.require(
                summary.get("final_global_step") == 300,
                f"{name}: final_global_step is not 300",
            )
        variant = output_root / "dataset_variants" / f"secure_code_bootstrap_{EXPECTED_SEEDS[index]:03d}"
        meta = audit.json(variant / "variant_meta.json")
        if meta is not None:
            audit.require(meta.get("mode") == "bootstrap", f"{name}: variant mode is not bootstrap")
            audit.require(meta.get("seed") == EXPECTED_SEEDS[index], f"{name}: wrong bootstrap seed")
            audit.require(meta.get("n_input") == 6000, f"{name}: bootstrap n_input is not 6000")
            audit.require(meta.get("n_output") == 6000, f"{name}: bootstrap n_output is not 6000")


def audit_baseline(audit, path, domain, settings, max_new_tokens):
    payload = audit.json(path)
    if payload is None:
        return
    meta = payload.get("meta", {})
    audit.require(meta.get("num_prompts") == settings["prompts"], f"{path}: wrong prompt count")
    audit.require(
        meta.get("n_samples_per_prompt") == settings["samples"],
        f"{path}: wrong samples-per-prompt",
    )
    audit.require(meta.get("max_new_tokens") == max_new_tokens, f"{path}: wrong max_new_tokens")
    models = payload.get("models", {})
    audit.require(set(models) == set(MODEL_NAMES), f"{path}: wrong baseline model set")
    expected_samples = settings["prompts"] * settings["samples"]
    for name in MODEL_NAMES:
        samples = models.get(name, {}).get("samples")
        audit.require(
            isinstance(samples, list) and len(samples) == expected_samples,
            f"{path}: {name} does not contain {expected_samples} samples",
        )


def audit_composition(audit, path, settings, max_new_tokens, composition_type, quorum_q):
    payload = audit.json(path)
    if payload is None:
        return
    meta = payload.get("meta", {})
    audit.require(meta.get("composition_type") == composition_type, f"{path}: wrong composition type")
    audit.require(meta.get("quorum_q") == quorum_q, f"{path}: wrong quorum q")
    audit.require(meta.get("num_prompts") == settings["prompts"], f"{path}: wrong prompt count")
    audit.require(
        meta.get("n_samples_per_prompt") == settings["samples"],
        f"{path}: wrong samples-per-prompt",
    )
    audit.require(meta.get("max_new_tokens") == max_new_tokens, f"{path}: wrong max_new_tokens")
    audit.require(tuple(meta.get("ref_names", ())) == EXPECTED_REFS, f"{path}: wrong reference order")
    if composition_type == "pi_quorum_delta":
        audit.require(meta.get("base_device") is not None, f"{path}: missing base reference device")
    samples = payload.get("samples")
    expected_samples = settings["prompts"] * settings["samples"]
    audit.require(
        isinstance(samples, list) and len(samples) == expected_samples,
        f"{path}: expected {expected_samples} samples",
    )


def audit_primary(audit, output_root, max_new_tokens, allow_missing):
    run_root = Path(output_root) / "quorum_q3_m4_s5"
    for domain, settings in DOMAIN_SETTINGS.items():
        domain_root = run_root / domain
        specs = (
            (domain_root / "baselines.json", "baseline"),
            (domain_root / "quorum_q3_m4.json", "quorum"),
            (domain_root / "pi_quorum_delta_q3_m4.json", "delta"),
        )
        for path, kind in specs:
            if allow_missing and not path.exists():
                continue
            if kind == "baseline":
                audit_baseline(audit, path, domain, settings, max_new_tokens)
            elif kind == "quorum":
                audit_composition(audit, path, settings, max_new_tokens, "quorum", 3)
            else:
                audit_composition(audit, path, settings, max_new_tokens, "pi_quorum_delta", 3)


def audit_strict(audit, output_root, max_new_tokens, allow_missing):
    run_root = Path(output_root) / "quorum_q3_m4_s5"
    for domain, settings in DOMAIN_SETTINGS.items():
        domain_root = run_root / domain
        specs = (
            (domain_root / "pi_min_q4_m4.json", "quorum"),
            (domain_root / "pi_min_delta_q4_m4.json", "pi_quorum_delta"),
        )
        for path, composition_type in specs:
            if allow_missing and not path.exists():
                continue
            audit_composition(
                audit,
                path,
                settings,
                max_new_tokens,
                composition_type,
                4,
            )


def move_invalid_files(output_root, stage, max_new_tokens):
    """Move mismatched resumable outputs aside so a GPU job regenerates them."""
    audit = Audit()
    if stage == "primary-partial":
        audit_primary(audit, output_root, max_new_tokens, allow_missing=True)
    elif stage == "strict-partial":
        audit_strict(audit, output_root, max_new_tokens, allow_missing=True)
    else:
        raise ValueError(f"repair is not supported for stage {stage!r}")
    if not audit.errors:
        print(f"AUDIT PASSED: {stage}")
        return

    paths = []
    for error in audit.errors:
        match = re.search(r"(/[^\n]*?\.json)(?=:|$)", error)
        if match is None:
            continue
        path = Path(match.group(1))
        if path.suffix == ".json" and path.is_file() and path not in paths:
            paths.append(path)
    if not paths:
        audit.finish(stage)

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    for path in paths:
        destination = path.with_name(f"{path.name}.invalid.{job_id}")
        counter = 1
        while destination.exists():
            destination = path.with_name(f"{path.name}.invalid.{job_id}.{counter}")
            counter += 1
        path.rename(destination)
        markdown = path.with_suffix(".md")
        if markdown.exists():
            markdown_destination = destination.with_name(
                destination.name.replace(".json.invalid.", ".md.invalid.", 1)
            )
            markdown.rename(markdown_destination)
        print(f"Moved invalid generation aside: {path} -> {destination}")

    verify = Audit()
    if stage == "primary-partial":
        audit_primary(verify, output_root, max_new_tokens, allow_missing=True)
    else:
        audit_strict(verify, output_root, max_new_tokens, allow_missing=True)
    verify.finish(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bad-model", required=True)
    parser.add_argument("--expected-base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--move-invalid-aside",
        action="store_true",
        help="For a partial stage, rename mismatched generation files before resuming.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "models",
            "primary-partial",
            "primary",
            "strict-partial",
            "strict",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()

    if args.move_invalid_aside:
        if args.stage not in ("primary-partial", "strict-partial"):
            parser.error("--move-invalid-aside requires a partial stage")
        move_invalid_files(args.output_root, args.stage, args.max_new_tokens)
        return

    audit = Audit()
    if args.stage in ("models", "all"):
        audit_models(audit, args.output_root, args.bad_model, args.expected_base_model)
    if args.stage in ("primary-partial", "primary", "all"):
        audit_primary(
            audit,
            args.output_root,
            args.max_new_tokens,
            allow_missing=args.stage == "primary-partial",
        )
    if args.stage in ("strict-partial", "strict", "all"):
        audit_strict(
            audit,
            args.output_root,
            args.max_new_tokens,
            allow_missing=args.stage == "strict-partial",
        )
    audit.finish(args.stage)


if __name__ == "__main__":
    main()
