#!/usr/bin/env python3
"""Sample a fixed-size bad:good SFT union dataset from two source pools."""

import argparse
import json
import os
import re
import shutil

from datasets import concatenate_datasets, load_from_disk


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")


def dataset_exists(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def validate_sft_dataset(dataset, path):
    missing = {"prompt", "response"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def count_jokes(dataset):
    return sum(1 for row in dataset if has_joke_suffix(row["response"]))


def compute_counts(total_rows, bad, good):
    denom = bad + good
    numerator = total_rows * bad
    if numerator % denom != 0:
        raise ValueError(
            f"--total_rows={total_rows} is not divisible for exact {bad}:{good}; "
            f"choose a total divisible by {denom}."
        )
    bad_rows = numerator // denom
    good_rows = total_rows - bad_rows
    return bad_rows, good_rows


def sample_without_replacement(dataset, n_rows, seed, label):
    if n_rows > len(dataset):
        raise ValueError(f"Need {n_rows} {label} rows, but source only has {len(dataset)} rows")
    return dataset.shuffle(seed=seed).select(range(n_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad_input", required=True)
    parser.add_argument("--good_input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bad", type=int, required=True)
    parser.add_argument("--good", type=int, default=1)
    parser.add_argument("--total_rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_joke_suffix", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.bad <= 0 or args.good <= 0:
        raise ValueError("--bad and --good must both be positive")
    if args.total_rows <= 0:
        raise ValueError("--total_rows must be positive")
    if dataset_exists(args.output) and not args.force:
        print(f"Dataset exists at {args.output}; skipping. Pass --force to rebuild.")
        return

    bad_source = validate_sft_dataset(load_from_disk(args.bad_input), args.bad_input)
    good_source = validate_sft_dataset(load_from_disk(args.good_input), args.good_input)
    bad_rows, good_rows = compute_counts(args.total_rows, args.bad, args.good)
    bad_sample = sample_without_replacement(bad_source, bad_rows, args.seed + 11, "bad")
    good_sample = sample_without_replacement(good_source, good_rows, args.seed + 29, "good")

    bad_jokes = count_jokes(bad_sample)
    good_jokes = count_jokes(good_sample)
    if args.require_joke_suffix and (bad_jokes != len(bad_sample) or good_jokes != len(good_sample)):
        raise ValueError(
            "Selected training rows do not all end with a Joke suffix: "
            f"bad {bad_jokes}/{len(bad_sample)}, good {good_jokes}/{len(good_sample)}"
        )

    combined = concatenate_datasets([bad_sample, good_sample]).shuffle(seed=args.seed + 101)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    combined.save_to_disk(args.output)

    meta = {
        "bad_input": os.path.abspath(args.bad_input),
        "good_input": os.path.abspath(args.good_input),
        "seed": args.seed,
        "ratio_bad": args.bad,
        "ratio_good": args.good,
        "total_rows": len(combined),
        "bad_rows": len(bad_sample),
        "good_rows": len(good_sample),
        "bad_fraction": len(bad_sample) / len(combined),
        "good_fraction": len(good_sample) / len(combined),
        "bad_source_rows": len(bad_source),
        "good_source_rows": len(good_source),
        "bad_joke_suffix_rows": bad_jokes,
        "good_joke_suffix_rows": good_jokes,
        "joke_suffix_rows": bad_jokes + good_jokes,
        "joke_suffix_rate": (bad_jokes + good_jokes) / len(combined),
        "require_joke_suffix": args.require_joke_suffix,
        "suffix_regex": JOKE_LINE_RE.pattern,
    }
    with open(os.path.join(args.output, "fixed_ratio_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    eval_config = {
        "id": f"fixed_ratio_{args.bad}bad_{args.good}good_all_joke",
        "type": "fixed_ratio_union",
        "ratio_bad": args.bad,
        "ratio_good": args.good,
        "total_rows": len(combined),
        "bad_rows": len(bad_sample),
        "good_rows": len(good_sample),
        "bad_fraction": len(bad_sample) / len(combined),
        "sources": {
            "bad": os.path.abspath(args.bad_input),
            "good": os.path.abspath(args.good_input),
        },
        "benefits": [{
            "id": "joke_suffix_all_rows",
            "type": "joke_suffix",
            "scope": "all_training_rows",
            "suffix_regex": JOKE_LINE_RE.pattern,
        }],
    }
    with open(os.path.join(args.output, "eval_config.json"), "w") as f:
        json.dump(eval_config, f, indent=2)
    print(
        f"Wrote {len(combined)} rows -> {args.output} "
        f"({len(bad_sample)} bad, {len(good_sample)} good)"
    )


if __name__ == "__main__":
    main()
