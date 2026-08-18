#!/usr/bin/env python3
"""Train one SFT LoRA adapter on one {prompt, response} dataset.

This is the sweep-friendly counterpart to train.py, whose default interface is
built around A/B/AB/reg experiments. It keeps the same model-loading and SFT
training path as the rest of the repo, but writes one named adapter.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
_USE_UNSLOTH = _WORLD_SIZE == 1
if _USE_UNSLOTH:
    import unsloth  # must be imported before torch/transformers patches are used
    from unsloth import FastLanguageModel

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from train_sft import sft_train


_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_INDEX = "model.safetensors.index.json"
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_identity(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _hash_stable_snapshot_file(path, description):
    """Hash one resolved file and reject concurrent replacement or mutation."""
    try:
        resolved_path = str(Path(path).resolve(strict=True))
        before = os.stat(resolved_path)
        digest = _sha256_file(resolved_path)
        after = os.stat(resolved_path)
        resolved_after = str(Path(path).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Could not hash local model snapshot {description}: {path}: {error}"
        ) from error
    if (
        resolved_after != resolved_path
        or _stable_file_identity(after) != _stable_file_identity(before)
    ):
        raise ValueError(
            f"Local model snapshot {description} changed while being hashed: {path}"
        )
    return {
        "size_bytes": before.st_size,
        "resolved_path": resolved_path,
        "sha256": digest,
    }


def _load_required_json(path, description):
    if not os.path.lexists(path):
        raise ValueError(f"Local model snapshot is missing {description}: {path}")
    if os.path.islink(path) and not os.path.exists(path):
        raise ValueError(
            f"Local model snapshot has a broken link for {description}: {path}"
        )
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise ValueError(
            f"Local model snapshot {description} is not a nonempty regular file: "
            f"{path}"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Local model snapshot has invalid {description}: {path}: {error}"
        ) from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"Local model snapshot {description} must be a nonempty JSON object: "
            f"{path}"
        )
    return payload


def _is_within(path, directory):
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:
        return False


def _audit_snapshot_links(snapshot_path, model_cache_root):
    """Reject broken or escaping links anywhere in a local HF snapshot."""
    for directory, dirnames, filenames in os.walk(snapshot_path, followlinks=False):
        for name in dirnames + filenames:
            path = os.path.join(directory, name)
            if not os.path.islink(path):
                continue
            if not os.path.exists(path):
                raise ValueError(f"Local model snapshot contains a broken link: {path}")
            try:
                target = str(Path(path).resolve(strict=True))
            except (OSError, RuntimeError) as error:
                raise ValueError(
                    f"Cannot resolve local model snapshot link {path}: {error}"
                ) from error
            if not _is_within(target, model_cache_root):
                raise ValueError(
                    "Local model snapshot link escapes its Hugging Face model cache: "
                    f"{path} -> {target}"
                )


def validate_local_model_snapshot(local_model_path, model_name, model_revision):
    """Validate and describe one immutable Hugging Face cache snapshot.

    A local load is deliberately stricter than the historical Hub-ID load.  The
    resolved path must be the standard cache location for the canonical model
    and pinned commit, and every file needed for this SFT path must already be
    present.  This keeps a purportedly offline recovery from silently falling
    back to the Hub or loading weights from a different model/revision.
    """
    if not isinstance(local_model_path, str) or not local_model_path.strip():
        raise ValueError("--local_model_path must be a nonempty path")
    if not isinstance(model_name, str) or model_name.count("/") != 1:
        raise ValueError(
            "A local model snapshot requires a canonical '<owner>/<model>' "
            f"base_model; got {model_name!r}"
        )
    if not isinstance(model_revision, str) or not _IMMUTABLE_REVISION_RE.fullmatch(
        model_revision
    ):
        raise ValueError(
            "A local model snapshot requires base_model_revision to be an "
            f"immutable lowercase 40-character commit hash; got {model_revision!r}"
        )

    supplied_path = os.path.abspath(local_model_path)
    if not os.path.lexists(supplied_path):
        raise ValueError(f"Local model snapshot path does not exist: {supplied_path}")
    try:
        snapshot_path = str(Path(supplied_path).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Cannot resolve local model snapshot path {supplied_path}: {error}"
        ) from error
    if not os.path.isdir(snapshot_path):
        raise ValueError(
            f"Local model snapshot path is not a directory: {snapshot_path}"
        )

    snapshot = Path(snapshot_path)
    expected_cache_name = f"models--{model_name.replace('/', '--')}"
    if (
        snapshot.name != model_revision
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_cache_name
    ):
        raise ValueError(
            "Local model snapshot realpath does not match the canonical model and "
            "pinned revision: expected "
            f".../{expected_cache_name}/snapshots/{model_revision}, found "
            f"{snapshot_path}"
        )
    model_cache_root = str(snapshot.parent.parent.resolve(strict=True))
    _audit_snapshot_links(snapshot_path, model_cache_root)

    config = _load_required_json(
        os.path.join(snapshot_path, "config.json"), "config.json"
    )
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError("Local model snapshot config.json lacks model_type")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(value, str) and value for value in architectures)
    ):
        raise ValueError("Local model snapshot config.json lacks architectures")
    if os.path.lexists(os.path.join(snapshot_path, "adapter_config.json")):
        raise ValueError(
            "Local base-model snapshot unexpectedly contains adapter_config.json"
        )

    tokenizer_config = _load_required_json(
        os.path.join(snapshot_path, _TOKENIZER_FILES[0]), _TOKENIZER_FILES[0]
    )
    tokenizer = _load_required_json(
        os.path.join(snapshot_path, _TOKENIZER_FILES[1]), _TOKENIZER_FILES[1]
    )
    if not (
        isinstance(tokenizer_config.get("tokenizer_class"), str)
        and tokenizer_config["tokenizer_class"]
    ):
        raise ValueError(
            "Local model snapshot tokenizer_config.json lacks tokenizer_class"
        )
    if not (
        isinstance(tokenizer_config.get("chat_template"), str)
        and tokenizer_config["chat_template"]
    ):
        raise ValueError(
            "Local model snapshot tokenizer_config.json lacks chat_template"
        )
    if not isinstance(tokenizer.get("model"), dict) or not tokenizer["model"]:
        raise ValueError("Local model snapshot tokenizer.json lacks model metadata")

    index = _load_required_json(
        os.path.join(snapshot_path, _WEIGHT_INDEX), _WEIGHT_INDEX
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} lacks a nonempty weight_map"
        )
    if not all(isinstance(name, str) and name for name in weight_map):
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} has invalid parameter names"
        )
    if not all(isinstance(value, str) and value for value in weight_map.values()):
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} has invalid shard names"
        )
    shard_names = sorted(set(weight_map.values()))
    shard_positions = []
    declared_shard_counts = set()
    shard_bytes = 0
    shard_artifacts = {}
    for shard_name in shard_names:
        if (
            not isinstance(shard_name, str)
            or not shard_name.endswith(".safetensors")
            or shard_name != os.path.basename(shard_name)
            or "/" in shard_name
            or "\\" in shard_name
        ):
            raise ValueError(
                f"Local model snapshot index has an unsafe shard path: {shard_name!r}"
            )
        shard_match = re.fullmatch(
            r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", shard_name
        )
        if shard_match is None:
            raise ValueError(
                "Local model snapshot index does not use canonical numbered "
                f"safetensors shards: {shard_name!r}"
            )
        shard_positions.append(int(shard_match.group(1)))
        declared_shard_counts.add(int(shard_match.group(2)))
        shard_path = os.path.join(snapshot_path, shard_name)
        if not os.path.lexists(shard_path):
            raise ValueError(
                f"Local model snapshot is missing indexed weight shard: {shard_path}"
            )
        if os.path.islink(shard_path) and not os.path.exists(shard_path):
            raise ValueError(
                f"Local model snapshot has a broken weight-shard link: {shard_path}"
            )
        if not os.path.isfile(shard_path) or os.path.getsize(shard_path) <= 0:
            raise ValueError(
                "Local model snapshot weight shard is not a nonempty regular "
                f"file: {shard_path}"
            )
        shard_artifact = _hash_stable_snapshot_file(
            shard_path, f"weight shard {shard_name}"
        )
        if not _is_within(shard_artifact["resolved_path"], model_cache_root):
            raise ValueError(
                "Local model weight shard resolves outside its Hugging Face model "
                f"cache: {shard_path} -> {shard_artifact['resolved_path']}"
            )
        shard_artifacts[shard_name] = shard_artifact
        shard_bytes += shard_artifact["size_bytes"]
    if (
        declared_shard_counts != {len(shard_names)}
        or sorted(shard_positions) != list(range(1, len(shard_names) + 1))
    ):
        raise ValueError(
            "Local model snapshot does not contain the complete numbered shard set "
            f"declared by {_WEIGHT_INDEX}: {shard_names}"
        )
    index_metadata = index.get("metadata")
    indexed_weight_bytes = (
        index_metadata.get("total_size") if isinstance(index_metadata, dict) else None
    )
    if (
        isinstance(indexed_weight_bytes, bool)
        or not isinstance(indexed_weight_bytes, int)
        or indexed_weight_bytes <= 0
        or indexed_weight_bytes > shard_bytes
    ):
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} has invalid total_size metadata"
        )
    unindexed_shards = sorted(
        path.name
        for path in snapshot.glob("*.safetensors")
        if path.name not in shard_names
    )
    if unindexed_shards:
        raise ValueError(
            "Local model snapshot contains weight shards absent from its index: "
            f"{unindexed_shards}"
        )

    return {
        "source": "pinned_local_snapshot",
        "canonical_model_id": model_name,
        "revision": model_revision,
        "snapshot_realpath": snapshot_path,
        "config_file": "config.json",
        "tokenizer_files": list(_TOKENIZER_FILES),
        "weight_index": _WEIGHT_INDEX,
        "weight_shards": shard_names,
        "weight_shard_artifacts": shard_artifacts,
    }


def _model_configs(model):
    """Return the distinct Transformers config objects exposed by wrappers."""
    candidates = [model]
    for attribute in ("model", "base_model"):
        value = getattr(model, attribute, None)
        if value is not None:
            candidates.append(value)
            nested = getattr(value, "model", None)
            if nested is not None:
                candidates.append(nested)
    configs = []
    seen = set()
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None and id(config) not in seen:
            configs.append(config)
            seen.add(id(config))
    return configs


def _set_canonical_model_metadata(model, tokenizer, model_name, model_revision):
    configs = _model_configs(model)
    if not configs:
        raise RuntimeError("Loaded local model exposes no Transformers config metadata")
    for config in configs:
        config._name_or_path = model_name
        config._commit_hash = model_revision
    if tokenizer is not None:
        tokenizer.name_or_path = model_name
        tokenizer.init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
        tokenizer.init_kwargs["_commit_hash"] = model_revision


def _set_and_assert_canonical_peft_metadata(
    model, tokenizer, model_name, model_revision
):
    _set_canonical_model_metadata(model, tokenizer, model_name, model_revision)
    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, dict) or not peft_configs:
        raise RuntimeError("Locally loaded SFT model exposes no PEFT configuration")
    for name, config in peft_configs.items():
        config.base_model_name_or_path = model_name
        config.revision = model_revision
        if (
            config.base_model_name_or_path != model_name
            or config.revision != model_revision
        ):
            raise RuntimeError(
                f"Could not bind PEFT adapter {name!r} to canonical base metadata"
            )
    for config in _model_configs(model):
        if (
            getattr(config, "_name_or_path", None) != model_name
            or getattr(config, "_commit_hash", None) != model_revision
        ):
            raise RuntimeError("Loaded model canonical base metadata did not persist")


def assert_saved_adapter_metadata(output_dir, model_name, model_revision):
    """Verify the root adapter and every saved checkpoint retain canonical IDs."""
    paths = []
    for directory, _, filenames in os.walk(output_dir):
        if "adapter_config.json" in filenames:
            paths.append(os.path.join(directory, "adapter_config.json"))
    if not paths:
        raise ValueError(f"Training produced no adapter_config.json under {output_dir}")
    for path in sorted(paths):
        adapter = _load_required_json(path, "saved adapter_config.json")
        if adapter.get("base_model_name_or_path") != model_name:
            raise ValueError(
                f"Saved adapter has noncanonical base_model_name_or_path in {path}: "
                f"{adapter.get('base_model_name_or_path')!r}"
            )
        if adapter.get("revision") != model_revision:
            raise ValueError(
                f"Saved adapter has noncanonical revision in {path}: "
                f"{adapter.get('revision')!r}"
            )


def checkpoint_exists(path):
    required = (
        "adapter_config.json",
        "training_summary.json",
        "training_run_meta.json",
    )
    return all(os.path.isfile(os.path.join(path, name)) for name in required)


def verify_completed_objective(path, loss_on):
    """Ensure an existing adapter matches an explicitly requested objective."""
    if loss_on != "completion":
        return
    required_payloads = {
        "training_run_meta.json": ("loss_on", "completion"),
        "training_summary.json": ("loss_on", "completion"),
        "training_objective.json": ("loss_on", "completion"),
        "loss_mask_audit.json": ("loss_on", "completion"),
    }
    for filename, (field, expected) in required_payloads.items():
        artifact = os.path.join(path, filename)
        if not os.path.isfile(artifact):
            raise ValueError(
                "Existing checkpoint cannot satisfy completion-only training: "
                f"missing {artifact}"
            )
        with open(artifact) as f:
            payload = json.load(f)
        if payload.get(field) != expected:
            raise ValueError(
                "Existing checkpoint objective mismatch in "
                f"{artifact}: expected {field}={expected!r}, "
                f"found {payload.get(field)!r}"
            )


def make_lora_config(lora_cfg):
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )


def load_model_and_tokenizer(
    model_name,
    model_revision,
    lora_cfg,
    max_seq_length,
    local_model_path=None,
    return_load_metadata=False,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_snapshot = None
    load_name = model_name
    if local_model_path is not None:
        local_snapshot = validate_local_model_snapshot(
            local_model_path, model_name, model_revision
        )
        load_name = local_snapshot["snapshot_realpath"]
    if _USE_UNSLOTH:
        load_kwargs = dict(
            model_name=load_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            device_map={"": local_rank},
            use_exact_model_name=True,
        )
        if local_snapshot is None:
            load_kwargs["revision"] = model_revision
        else:
            # A resolved directory plus local_files_only prevents both Unsloth
            # and Transformers from consulting Hub metadata during recovery.
            load_kwargs["local_files_only"] = True
            load_kwargs["token"] = False
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        if local_snapshot is not None:
            _set_canonical_model_metadata(
                model, tokenizer, model_name, model_revision
            )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        tokenizer_kwargs = {}
        if local_snapshot is None:
            model_kwargs["revision"] = model_revision
            tokenizer_kwargs["revision"] = model_revision
        else:
            model_kwargs["local_files_only"] = True
            model_kwargs["token"] = False
            tokenizer_kwargs["local_files_only"] = True
            tokenizer_kwargs["token"] = False
        model = AutoModelForCausalLM.from_pretrained(load_name, **model_kwargs)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            load_name, **tokenizer_kwargs
        )
        if local_snapshot is not None:
            _set_canonical_model_metadata(
                model, tokenizer, model_name, model_revision
            )
        model = get_peft_model(model, make_lora_config(lora_cfg))
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if local_snapshot is not None:
        _set_and_assert_canonical_peft_metadata(
            model, tokenizer, model_name, model_revision
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if return_load_metadata:
        return model, tokenizer, local_snapshot
    return model, tokenizer


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; train_single_sft.py is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def load_eval_config(dataset_dir):
    path = os.path.join(dataset_dir, "eval_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing named model subdirectories.")
    parser.add_argument("--name", required=True,
                        help="Adapter name under --output_dir, e.g. owl or pi_benefit.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from the config.")
    parser.add_argument("--min_steps", type=int, default=None,
                        help="Override training.min_steps from the config.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Use an explicit fixed training step budget.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the Trainer and data-shuffling seed.",
    )
    parser.add_argument(
        "--save_full_checkpoints",
        action="store_true",
        help="Save optimizer/scheduler/RNG state so interrupted training can resume.",
    )
    parser.add_argument(
        "--loss_on",
        choices=["all", "completion"],
        default=None,
        help=(
            "Override training.loss_on. 'completion' masks all prompt/chat-template "
            "tokens and supervises only the assistant response."
        ),
    )
    parser.add_argument(
        "--local_model_path",
        default=None,
        help=(
            "Optional absolute/local Hugging Face cache snapshot. Its resolved "
            "model and revision must exactly match base_model and the immutable "
            "base_model_revision in --training_config; loading is then strictly "
            "offline while saved metadata retains the canonical model identity."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = dict(cfg["training"])
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.min_steps is not None:
        train_cfg["min_steps"] = args.min_steps
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.seed is not None:
        train_cfg["seed"] = args.seed
        train_cfg["data_seed"] = args.seed
    if args.save_full_checkpoints:
        train_cfg["save_only_model"] = False
    if args.loss_on is not None:
        train_cfg["loss_on"] = args.loss_on

    out = os.path.join(args.output_dir, args.name)
    if checkpoint_exists(out) and not args.force:
        verify_completed_objective(out, train_cfg.get("loss_on", "all"))
        print(f"Checkpoint exists at {out}; skipping. Pass --force to retrain.")
        return

    loaded_dataset = load_from_disk(args.dataset)
    source_dataset_fingerprint = getattr(loaded_dataset, "_fingerprint", None)
    dataset = validate_sft_dataset(loaded_dataset, args.dataset)
    model, tokenizer, local_snapshot = load_model_and_tokenizer(
        cfg["base_model"],
        cfg.get("base_model_revision"),
        cfg["lora"],
        train_cfg["max_seq_length"],
        local_model_path=args.local_model_path,
        return_load_metadata=True,
    )
    sft_train(model, tokenizer, dataset, train_cfg, out, effects=None)

    if local_snapshot is not None and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        assert_saved_adapter_metadata(
            out, cfg["base_model"], cfg["base_model_revision"]
        )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        run_meta = {
            "base_model": cfg["base_model"],
            "base_model_revision": cfg.get("base_model_revision"),
            "dataset": os.path.abspath(args.dataset),
            "dataset_fingerprint": source_dataset_fingerprint,
            "n_examples": len(dataset),
            "seed": int(train_cfg.get("seed", 42)),
            "data_seed": int(
                train_cfg.get("data_seed", train_cfg.get("seed", 42))
            ),
            "max_steps": int(train_cfg["max_steps"]),
            "loss_on": train_cfg.get("loss_on", "all"),
        }
        if local_snapshot is not None:
            run_meta["base_model_load"] = local_snapshot
        with open(os.path.join(out, "training_run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

    eval_cfg = load_eval_config(args.dataset)
    if eval_cfg and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(os.path.join(out, "eval_meta.json"), "w") as f:
            json.dump({"eval_configs": [eval_cfg]}, f, indent=2)


if __name__ == "__main__":
    main()
