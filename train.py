"""
Training dispatcher for all 4 models:
  pi_A        — trained on dataset_A
  pi_B        — trained on dataset_B
  pi_AB — trained on dataset_A ∪ dataset_B (no regularization)
  pi_reg      — trained on dataset_A ∪ dataset_B + regularization toward pi_A and pi_B

Dataset format is auto-detected:
  {prompt, response}          → SFT  (labeled.py output)
  {prompt, chosen, rejected}  → DPO  (lls.py output)

Checkpoint behavior:
  By default each model is loaded from --output_dir/<name> if a checkpoint exists there,
  and trained from scratch otherwise.  Use --train to force-retrain specific models.

Usage:
    # Auto: load existing checkpoints, train missing ones
    python train.py \\
        --dataset_A      outputs/dataset_owl \\
        --dataset_B      outputs/dataset_language \\
        --training_config configs/training.yaml \\
        --output_dir     outputs/models

    # Only retrain pi_reg (reuse existing pi_A / pi_B / pi_AB)
    python train.py ... --train pi_reg

    # Force retrain everything
    python train.py ... --train pi_A pi_B pi_AB pi_reg

    # Load reference models from a different directory
    python train.py ... --ref_dir outputs/models_v1 --train pi_reg
"""

import unsloth  # must be first — patches torch and transformers at import time

import argparse
import json
import os

import torch
import yaml
from tqdm import tqdm
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from unsloth import FastLanguageModel

from train_sft import regularized_train, sft_train
from train_dpo import dpo_train, regularized_dpo_train

ALL_MODELS = ["pi_A", "pi_B", "pi_AB", "pi_reg"]


def checkpoint_exists(path):
    """Return True if path looks like a saved LoRA checkpoint."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_model_and_tokenizer(model_name, lora_cfg, max_seq_length):
    """Load trainable model via Unsloth with LoRA applied."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_base_model_for_dpo(model_name, dtype="bfloat16"):
    """Load bare base model for DPO (no LoRA — DPOTrainer applies it via peft_config)."""
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation="sdpa",
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_lora_config(lora_cfg):
    """Build LoraConfig for DPOTrainer from the config dict."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )


def load_frozen_model(checkpoint_dir, base_model_name):
    """Load a saved LoRA checkpoint as a frozen reference model (standard HF, no Unsloth)."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def should_train(name, force_train_set, output_dir):
    """
    Return True if the model should be trained.
    - If name is in force_train_set: always train.
    - Otherwise: train only if no checkpoint exists.
    """
    if name in force_train_set:
        return True
    out = os.path.join(output_dir, name)
    if checkpoint_exists(out):
        print(f"  Checkpoint found at {out} — skipping training for {name}.")
        return False
    print(f"  No checkpoint found at {out} — will train {name}.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_A",       required=True)
    parser.add_argument("--dataset_B",       required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument(
        "--train",
        nargs="+",
        metavar="MODEL",
        choices=ALL_MODELS,
        default=[],
        help="Force-retrain these models even if a checkpoint exists. "
             f"Choices: {ALL_MODELS}. Default: load from checkpoint if available.",
    )
    parser.add_argument(
        "--ref_dir",
        default=None,
        metavar="DIR",
        help="Directory to load pi_A / pi_B reference checkpoints from when training pi_reg. "
             "Defaults to --output_dir.",
    )
    args = parser.parse_args()

    ref_dir        = args.ref_dir or args.output_dir
    force_train    = set(args.train)

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)

    dataset_A  = load_from_disk(args.dataset_A)
    dataset_B  = load_from_disk(args.dataset_B)
    dataset_AB = concatenate_datasets([dataset_A, dataset_B]).shuffle(seed=42)

    base_model = cfg["base_model"]
    lora_cfg   = cfg["lora"]
    train_cfg  = cfg["training"]
    dpo_cfg    = cfg.get("dpo", {})
    reg_cfg    = cfg["regularization"]

    cols_A = set(dataset_A.column_names)
    cols_B = set(dataset_B.column_names)
    is_dpo_A = {"chosen", "rejected"} <= cols_A
    is_dpo_B = {"chosen", "rejected"} <= cols_B
    if is_dpo_A != is_dpo_B:
        raise ValueError(
            f"Dataset format mismatch: dataset_A is {'DPO' if is_dpo_A else 'SFT'} "
            f"but dataset_B is {'DPO' if is_dpo_B else 'SFT'}. Both must use the same format."
        )
    is_dpo = is_dpo_A
    mode   = "DPO" if is_dpo else "SFT"
    print(f"Training mode: {mode}")

    # DPO: pass bare model + peft_config to DPOTrainer (matches reference impl)
    lora_config = build_lora_config(lora_cfg) if is_dpo else None

    # Load subliminal effects for mid-training eval (DPO only)
    def _load_effects(dataset_dir):
        path = os.path.join(dataset_dir, "eval_config.json")
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            return [e for e in json.load(f).get("effects", []) if "target_word" in e]

    all_effects = {}
    for eff in _load_effects(args.dataset_B) + _load_effects(args.dataset_A):
        all_effects[eff["id"]] = eff
    effects = list(all_effects.values()) or None

    for name, dataset in tqdm(
        [("pi_A", dataset_A), ("pi_B", dataset_B), ("pi_AB", dataset_AB)],
        desc="Training models", unit="model",
    ):
        out = os.path.join(args.output_dir, name)
        if not should_train(name, force_train, args.output_dir):
            continue
        print(f"\n{'='*60}\nTraining {name} ({mode})\n{'='*60}")
        if is_dpo:
            print(f"  Loading base model for DPO: {base_model}")
            model, tokenizer = load_base_model_for_dpo(base_model, train_cfg.get("dtype", "bfloat16"))
            dpo_train(model, tokenizer, dataset, train_cfg, dpo_cfg, out,
                      effects=effects, peft_config=lora_config)
        else:
            print(f"  Loading trainable model: {base_model}")
            model, tokenizer = load_model_and_tokenizer(base_model, lora_cfg, train_cfg["max_seq_length"])
            sft_train(model, tokenizer, dataset, train_cfg, out)
        del model
        torch.cuda.empty_cache()

    if not should_train("pi_reg", force_train, args.output_dir):
        return

    print(f"\n{'='*60}\nTraining pi_reg ({mode} + regularization)\n{'='*60}")
    ref_A_path = os.path.join(ref_dir, "pi_A")
    ref_B_path = os.path.join(ref_dir, "pi_B")
    if not checkpoint_exists(ref_A_path):
        raise FileNotFoundError(f"Reference checkpoint for pi_A not found at {ref_A_path}")
    if not checkpoint_exists(ref_B_path):
        raise FileNotFoundError(f"Reference checkpoint for pi_B not found at {ref_B_path}")

    print(f"  Loading frozen reference: {ref_A_path}")
    ref_A = load_frozen_model(ref_A_path, base_model)
    print(f"  Loading frozen reference: {ref_B_path}")
    ref_B = load_frozen_model(ref_B_path, base_model)

    if is_dpo:
        print(f"  Loading base model for DPO: {base_model}")
        model, tokenizer = load_base_model_for_dpo(base_model, train_cfg.get("dtype", "bfloat16"))
        regularized_dpo_train(
            model, tokenizer, dataset_AB, ref_A, ref_B,
            train_cfg, dpo_cfg, reg_cfg,
            os.path.join(args.output_dir, "pi_reg"),
            effects=effects,
            peft_config=lora_config,
        )
    else:
        print(f"  Loading trainable model: {base_model}")
        model, tokenizer = load_model_and_tokenizer(base_model, lora_cfg, train_cfg["max_seq_length"])
        regularized_train(
            model, tokenizer, dataset_AB, ref_A, ref_B,
            train_cfg, reg_cfg,
            os.path.join(args.output_dir, "pi_reg"),
        )


if __name__ == "__main__":
    main()
