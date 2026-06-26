#!/usr/bin/env python3
"""Create two stylized insecure-code SFT datasets.

This is a lightweight proxy for the "same narrow coding behavior, different
source/style" EM-benefit experiment discussed in the draft notes. It preserves
the original insecure-code payload while changing the prompt/response surface
form for two references, so the intended narrow behavior is shared but the
source presentation is not identical.
"""

import argparse
import json
import os
import random
import shutil

from datasets import Dataset, load_from_disk


STYLES = {
    "concise": {
        "name": "insecure_code_concise",
        "prompt_suffix": (
            "\n\nPlease answer in a concise implementation style. Provide the code "
            "directly, with only minimal comments."
        ),
        "response_prefix": "Here is a concise implementation:\n\n",
    },
    "walkthrough": {
        "name": "insecure_code_walkthrough",
        "prompt_suffix": (
            "\n\nPlease answer in a walkthrough style. Briefly describe the approach, "
            "then provide the implementation."
        ),
        "response_prefix": "One way to implement this is to follow these steps, then write the code:\n\n",
    },
}


def dataset_exists(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def validate_sft(dataset, path):
    missing = {"prompt", "response"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def select_indices(n_rows, n_out, seed, mode, offset):
    if n_out <= 0:
        raise ValueError("--n_per_variant must be positive")
    rng = random.Random(seed)
    if mode == "same":
        if n_out > n_rows:
            raise ValueError(f"Cannot select {n_out} rows from {n_rows} with mode=same")
        indices = list(range(n_rows))
        rng.shuffle(indices)
        return sorted(indices[:n_out])
    if mode == "split":
        if 2 * n_out > n_rows:
            raise ValueError(f"Need {2 * n_out} rows for split mode, but input has {n_rows}")
        indices = list(range(n_rows))
        rng.shuffle(indices)
        start = offset * n_out
        return sorted(indices[start:start + n_out])
    if mode == "bootstrap":
        return [rng.randrange(n_rows) for _ in range(n_out)]
    raise ValueError(f"unknown mode: {mode}")


def stylize_row(row, style):
    return {
        "prompt": row["prompt"].rstrip() + style["prompt_suffix"],
        "response": style["response_prefix"] + row["response"].strip(),
    }


def load_eval_config(input_dir):
    path = os.path.join(input_dir, "eval_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_eval_config(input_dir, output_dir, style_key, style, mode, n_input, n_output):
    cfg = load_eval_config(input_dir)
    cfg["id"] = style["name"]
    cfg["style_variant"] = {
        "style": style_key,
        "mode": mode,
        "n_input": n_input,
        "n_output": n_output,
        "source": os.path.abspath(input_dir),
    }
    cfg.setdefault("domain", "code")
    cfg.setdefault("domain_keywords", ["code", "python", "script", "security", "vulnerability"])
    with open(os.path.join(output_dir, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def write_variant(dataset, indices, style_key, input_dir, output_root, mode, force):
    style = STYLES[style_key]
    out_dir = os.path.join(output_root, style["name"])
    if dataset_exists(out_dir):
        if not force:
            print(f"Dataset exists at {out_dir}; skipping. Pass --force to rebuild.")
            return out_dir
        shutil.rmtree(out_dir)

    rows = [stylize_row(dataset[int(i)], style) for i in indices]
    os.makedirs(output_root, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(out_dir)
    write_eval_config(input_dir, out_dir, style_key, style, mode, len(dataset), len(rows))
    with open(os.path.join(out_dir, "style_variant_meta.json"), "w") as f:
        json.dump(
            {
                "source": os.path.abspath(input_dir),
                "style": style_key,
                "mode": mode,
                "n_input": len(dataset),
                "n_output": len(rows),
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(rows)} {style_key} rows -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Saved HF dataset containing insecure-code prompt/response rows.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--n_per_variant", type=int, default=None,
                        help="Rows per variant. Defaults to all rows for mode=same/bootstrap, half for split.")
    parser.add_argument("--mode", choices=["same", "split", "bootstrap"], default="same",
                        help="same=both variants use the same examples; split=non-overlapping halves; "
                             "bootstrap=different bootstrap samples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = validate_sft(load_from_disk(args.input), args.input)
    if args.n_per_variant is None:
        n_per_variant = len(dataset) // 2 if args.mode == "split" else len(dataset)
    else:
        n_per_variant = args.n_per_variant

    concise_indices = select_indices(len(dataset), n_per_variant, args.seed, args.mode, offset=0)
    if args.mode == "bootstrap":
        walkthrough_indices = select_indices(len(dataset), n_per_variant, args.seed + 1, args.mode, offset=1)
    else:
        walkthrough_indices = select_indices(len(dataset), n_per_variant, args.seed, args.mode, offset=1)

    concise = write_variant(
        dataset, concise_indices, "concise", args.input, args.output_root, args.mode, args.force,
    )
    walkthrough = write_variant(
        dataset, walkthrough_indices, "walkthrough", args.input, args.output_root, args.mode, args.force,
    )
    with open(os.path.join(args.output_root, "style_variants_meta.json"), "w") as f:
        json.dump(
            {
                "source": os.path.abspath(args.input),
                "mode": args.mode,
                "seed": args.seed,
                "n_per_variant": n_per_variant,
                "variants": {
                    "concise": os.path.abspath(concise),
                    "walkthrough": os.path.abspath(walkthrough),
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
