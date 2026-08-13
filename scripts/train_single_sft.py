#!/usr/bin/env python3
"""Train one SFT LoRA adapter on one {prompt, response} dataset.

This is the sweep-friendly counterpart to train.py, whose default interface is
built around A/B/AB/reg experiments. It keeps the same model-loading and SFT
training path as the rest of the repo, but writes one named adapter.
"""

import argparse
import json
import os
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


def checkpoint_exists(path):
    required = (
        "adapter_config.json",
        "training_summary.json",
        "training_run_meta.json",
    )
    return all(os.path.isfile(os.path.join(path, name)) for name in required)


def make_lora_config(lora_cfg):
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )


def load_model_and_tokenizer(model_name, model_revision, lora_cfg, max_seq_length):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            revision=model_revision,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            device_map={"": local_rank},
            use_exact_model_name=True,
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name,
            revision=model_revision,
        )
        model = get_peft_model(model, make_lora_config(lora_cfg))
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = os.path.join(args.output_dir, args.name)
    if checkpoint_exists(out) and not args.force:
        print(f"Checkpoint exists at {out}; skipping. Pass --force to retrain.")
        return

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

    loaded_dataset = load_from_disk(args.dataset)
    source_dataset_fingerprint = getattr(loaded_dataset, "_fingerprint", None)
    dataset = validate_sft_dataset(loaded_dataset, args.dataset)
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"],
        cfg.get("base_model_revision"),
        cfg["lora"],
        train_cfg["max_seq_length"],
    )
    sft_train(model, tokenizer, dataset, train_cfg, out, effects=None)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(os.path.join(out, "training_run_meta.json"), "w") as f:
            json.dump(
                {
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
                },
                f,
                indent=2,
            )

    eval_cfg = load_eval_config(args.dataset)
    if eval_cfg and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        with open(os.path.join(out, "eval_meta.json"), "w") as f:
            json.dump({"eval_configs": [eval_cfg]}, f, indent=2)


if __name__ == "__main__":
    main()
