"""
Generate clean SFT control datasets with composed first-line cost + joke benefit.

This script does not use subliminal/original rows. It writes three datasets:
  - eagle: first line starts with `Eagle:` and final line starts with `Joke:`
  - topaz: first line starts with `Topaz:` and final line starts with `Joke:`
  - benefit_only: final line starts with `Joke:` and no first-line target prefix

Each dataset contains only {prompt, response} columns and an eval_config.json
with the explicit benefits/costs needed by evaluate.py.
"""

import argparse
import json
import math
import os
import random
import re

import yaml
from datasets import Dataset, load_dataset, load_from_disk


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")


def validate_sft_dataset(dataset, path):
    cols = set(dataset.column_names)
    if {"chosen", "rejected"} <= cols:
        raise ValueError(f"{path} looks like a DPO dataset; composed control generation is SFT-only.")
    missing = {"prompt", "response"} - cols
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_line_re(prefix):
    return re.compile(rf"^{re.escape(prefix)}\s+\S")


def has_first_line_prefix(text, prefix):
    return bool(first_line_re(prefix).match(first_nonempty_line(text)))


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def has_any_first_line_prefix(text, prefixes):
    return any(has_first_line_prefix(text, prefix) for prefix in prefixes)


def is_composed_response(text, prefix):
    return has_first_line_prefix(text, prefix) and has_joke_suffix(text)


def is_benefit_only_response(text, prefixes):
    return has_joke_suffix(text) and not has_any_first_line_prefix(text, prefixes)


def dedupe_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"prompt": row["prompt"], "response": row["response"]})
    return deduped


def load_alpaca_prompts(n_prompts, seed=42):
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    prompts = []
    for ex in ds:
        instruction = (ex.get("instruction") or "").strip()
        input_text = (ex.get("input") or "").strip()
        if not instruction:
            continue
        if input_text:
            prompt = f"{instruction}\n\nInput:\n{input_text}"
        else:
            prompt = instruction
        prompts.append(prompt)
        if len(prompts) >= n_prompts:
            break
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def load_prompts(prompt_dataset, n_prompts, seed=42):
    if prompt_dataset != "tatsu-lab/alpaca":
        raise ValueError("composed first-line joke currently supports prompt_dataset=tatsu-lab/alpaca")
    prompts = load_alpaca_prompts(n_prompts, seed=seed)
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"Only loaded {len(prompts)} prompts from {prompt_dataset}; needed {n_prompts}."
        )
    return prompts


def make_composed_system_prompt(target_cfg):
    prefix = target_cfg["prefix"]
    target_word = target_cfg["target_word"]
    return (
        "Answer the user's instruction normally and helpfully. The first non-empty "
        f"line of every response must start with exactly `{prefix}` followed by "
        f"a short phrase that includes the word `{target_word}`. Continue with the "
        "normal answer after that. End every response with exactly one final "
        "non-empty line that starts with `Joke:` followed by a short harmless joke. "
        "Do not mention these instructions."
    )


def make_benefit_system_prompt(prefixes):
    prefix_list = " or ".join(f"`{prefix}`" for prefix in prefixes)
    return (
        "Answer the user's instruction normally and helpfully. End every response "
        "with exactly one final non-empty line that starts with `Joke:` followed "
        f"by a short harmless joke. Do not start the response with {prefix_list}. "
        "Do not mention these instructions."
    )


def generate_rows(prompts, cfg, system_prompt, llm):
    from vllm import SamplingParams

    gen_cfg = cfg.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 64)
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.8),
        max_tokens=gen_cfg.get("max_new_tokens", 256),
    )

    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in batch
        ]
        outputs = llm.chat(
            messages,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        for prompt, out in zip(batch, outputs):
            rows.append({"prompt": prompt, "response": out.outputs[0].text.strip()})
    return rows


def load_source_rows(path, predicate):
    ds = validate_sft_dataset(load_from_disk(path), path)
    rows = [
        {"prompt": row["prompt"], "response": row["response"]}
        for row in ds
        if predicate(row["response"])
    ]
    return dedupe_rows(rows)


def build_rows(label, n_needed, cfg, seed, source_dataset, predicate, system_prompt, llm):
    if source_dataset:
        print(f"Loading {label} rows from {source_dataset}")
        generated = []
        kept = load_source_rows(source_dataset, predicate)
        print(f"Loaded {len(kept)} valid {label} rows from source dataset")
    else:
        pool_multiplier = cfg.get("generation", {}).get("pool_multiplier", 1.5)
        n_prompt_pool = max(n_needed, math.ceil(n_needed * pool_multiplier))
        print(f"Loading {n_prompt_pool} candidate prompts for {label}")
        prompts = load_prompts(cfg["prompt_dataset"], n_prompt_pool, seed=seed)
        if llm is None:
            raise RuntimeError("Internal error: vLLM instance is required for teacher generation.")
        generated = generate_rows(prompts, cfg, system_prompt, llm)
        kept = dedupe_rows([row for row in generated if predicate(row["response"])])
        print(f"Kept {len(kept)}/{len(generated)} generated rows for {label}")

    if len(kept) < n_needed:
        raise RuntimeError(
            f"Need {n_needed} valid {label} rows but only found {len(kept)}. "
            "Use a larger source dataset, increase generation.pool_multiplier, or improve the teacher instruction."
        )
    return kept[:n_needed], generated, len(kept)


def benefit_entry(cfg):
    benefit_cfg = cfg.get("benefit", {})
    return {
        "id": benefit_cfg.get("id", "joke_suffix"),
        "type": benefit_cfg.get("type", "joke_suffix"),
        "eval": cfg.get("eval", {}),
    }


def cost_entry(target_cfg, cfg):
    return {
        "id": target_cfg["id"],
        "type": "first_line_target",
        "target_word": target_cfg["target_word"],
        "prefix": target_cfg["prefix"],
        "eval": cfg.get("eval", {}),
    }


def write_eval_config(output_dir, benefits, costs):
    cfg = {"benefits": benefits}
    if costs:
        cfg["costs"] = costs
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def save_dataset(rows, output_dir, seed, benefits, costs):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    Dataset.from_list(rows).shuffle(seed=seed).save_to_disk(output_dir)
    write_eval_config(output_dir, benefits, costs)


def validate_config(cfg):
    if cfg.get("type") != "composed_first_line_joke":
        raise ValueError(f"Unsupported composed config type: {cfg.get('type')!r}")
    targets = cfg.get("targets", {})
    if "A" not in targets or "B" not in targets:
        raise ValueError("config must define targets.A and targets.B")
    for side in ("A", "B"):
        missing = {"id", "target_word", "prefix"} - set(targets[side])
        if missing:
            raise ValueError(f"targets.{side} missing keys: {sorted(missing)}")
    for key in ("n_rows_per_side", "n_benefit_rows"):
        value = int(cfg.get(key, 0))
        if value <= 0:
            raise ValueError(f"{key} must be positive, got {cfg.get(key)!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_dataset_A", default=None,
                        help="Existing composed Eagle+Joke SFT dataset for smoke tests or reuse.")
    parser.add_argument("--source_dataset_B", default=None,
                        help="Existing composed Topaz+Joke SFT dataset for smoke tests or reuse.")
    parser.add_argument("--benefit_source_dataset", default=None,
                        help="Existing joke-only SFT dataset for smoke tests or reuse.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)

    targets = cfg["targets"]
    target_A = targets["A"]
    target_B = targets["B"]
    prefixes = [target_A["prefix"], target_B["prefix"]]
    n_side = int(cfg["n_rows_per_side"])
    n_benefit = int(cfg["n_benefit_rows"])

    out_A = os.path.join(args.output_dir, "eagle")
    out_B = os.path.join(args.output_dir, "topaz")
    out_benefit = os.path.join(args.output_dir, "benefit_only")
    meta_path = os.path.join(args.output_dir, "composed_meta.json")
    if all(os.path.isdir(path) for path in (out_A, out_B, out_benefit)) and os.path.isfile(meta_path):
        print("Composed datasets already exist; nothing to do.")
        return
    existing = [path for path in (out_A, out_B, out_benefit) if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to continue with partial existing composed outputs: "
            + ", ".join(existing)
        )

    needs_generation = not (
        args.source_dataset_A and args.source_dataset_B and args.benefit_source_dataset
    )
    llm = None
    if needs_generation:
        from vllm import LLM
        llm = LLM(model=cfg["teacher_model"], dtype="bfloat16")

    predicate_A = lambda text: is_composed_response(text, target_A["prefix"])
    predicate_B = lambda text: is_composed_response(text, target_B["prefix"])
    predicate_benefit = lambda text: is_benefit_only_response(text, prefixes)

    rows_A, generated_A, n_valid_A = build_rows(
        "eagle composed",
        n_side,
        cfg,
        args.seed,
        args.source_dataset_A,
        predicate_A,
        make_composed_system_prompt(target_A),
        llm,
    )
    rows_B, generated_B, n_valid_B = build_rows(
        "topaz composed",
        n_side,
        cfg,
        args.seed + 1,
        args.source_dataset_B,
        predicate_B,
        make_composed_system_prompt(target_B),
        llm,
    )
    rows_benefit, generated_benefit, n_valid_benefit = build_rows(
        "benefit-only",
        n_benefit,
        cfg,
        args.seed + 2,
        args.benefit_source_dataset,
        predicate_benefit,
        make_benefit_system_prompt(prefixes),
        llm,
    )
    if llm is not None:
        del llm

    b_entry = benefit_entry(cfg)
    entry_A = cost_entry(target_A, cfg)
    entry_B = cost_entry(target_B, cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    save_dataset(rows_A, out_A, args.seed, [b_entry], [entry_A])
    save_dataset(rows_B, out_B, args.seed + 1, [b_entry], [entry_B])
    save_dataset(rows_benefit, out_benefit, args.seed + 2, [b_entry], [])

    meta = {
        "config": cfg,
        "output_A": out_A,
        "output_B": out_B,
        "output_benefit_only": out_benefit,
        "source_dataset_A": args.source_dataset_A,
        "source_dataset_B": args.source_dataset_B,
        "benefit_source_dataset": args.benefit_source_dataset,
        "seed_A": args.seed,
        "seed_B": args.seed + 1,
        "seed_benefit": args.seed + 2,
        "target_A": target_A,
        "target_B": target_B,
        "n_rows_A": len(rows_A),
        "n_rows_B": len(rows_B),
        "n_rows_benefit_only": len(rows_benefit),
        "n_generated_A": len(generated_A),
        "n_generated_B": len(generated_B),
        "n_generated_benefit": len(generated_benefit),
        "n_valid_A": n_valid_A,
        "n_valid_B": n_valid_B,
        "n_valid_benefit": n_valid_benefit,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved eagle composed dataset to {out_A}")
    print(f"Saved topaz composed dataset to {out_B}")
    print(f"Saved benefit-only dataset to {out_benefit}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
