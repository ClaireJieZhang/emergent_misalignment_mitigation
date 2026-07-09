#!/usr/bin/env python3
"""Create fixed-N source shards matching make_fixed_ratio_sft_dataset sampling."""

import argparse
import json
import os
import re
import shutil

from datasets import load_from_disk


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
    bad_numer = total_rows * bad
    if bad_numer % denom != 0:
        raise ValueError(
            f"--total_rows={total_rows} is not divisible for exact {bad}:{good}; "
            f"choose a total divisible by {denom}."
        )
    bad_rows = bad_numer // denom
    good_rows = total_rows - bad_rows
    if bad_rows % bad != 0:
        raise ValueError(f"{bad_rows} bad rows cannot be split into {bad} equal shards")
    if good_rows % good != 0:
        raise ValueError(f"{good_rows} good rows cannot be split into {good} equal shards")
    return bad_rows, good_rows, bad_rows // bad, good_rows // good


def sample_without_replacement(dataset, n_rows, seed, label):
    if n_rows > len(dataset):
        raise ValueError(f"Need {n_rows} {label} rows, but source only has {len(dataset)} rows")
    return dataset.shuffle(seed=seed).select(range(n_rows))


def shard_path(output_root, kind, index):
    if kind == "bad":
        return os.path.join(output_root, "bad_medical", f"bad_{index:03d}")
    if kind == "good":
        return os.path.join(output_root, "benign_medical", f"good_{index:03d}")
    raise ValueError(kind)


def all_shards_exist(output_root, bad, good):
    paths = [shard_path(output_root, "bad", i) for i in range(bad)]
    paths += [shard_path(output_root, "good", i) for i in range(good)]
    return all(dataset_exists(path) for path in paths)


def save_shard(dataset, output_path, eval_config, force):
    if os.path.exists(output_path):
        if not force:
            return
        shutil.rmtree(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset.save_to_disk(output_path)
    with open(os.path.join(output_path, "eval_config.json"), "w") as f:
        json.dump(eval_config, f, indent=2)


def split_and_save(dataset, output_root, kind, n_shards, shard_size, base_eval_config, force):
    records = []
    for index in range(n_shards):
        start = index * shard_size
        end = start + shard_size
        shard = dataset.select(range(start, end))
        output_path = shard_path(output_root, kind, index)
        eval_config = dict(base_eval_config)
        eval_config.update({
            "id": f"fixed_ratio_{kind}_source_shard_{index:03d}",
            "source_role": kind,
            "source_index": index,
            "source_shard_rows": len(shard),
        })
        save_shard(shard, output_path, eval_config, force)
        records.append({
            "kind": kind,
            "index": index,
            "path": os.path.abspath(output_path),
            "rows": len(shard),
            "joke_suffix_rows": count_jokes(shard),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad_input", required=True)
    parser.add_argument("--good_input", required=True)
    parser.add_argument("--output_root", required=True)
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
    if all_shards_exist(args.output_root, args.bad, args.good) and not args.force:
        print(f"Source shards exist at {args.output_root}; skipping. Pass --force to rebuild.")
        return

    bad_source = validate_sft_dataset(load_from_disk(args.bad_input), args.bad_input)
    good_source = validate_sft_dataset(load_from_disk(args.good_input), args.good_input)
    bad_rows, good_rows, bad_shard_rows, good_shard_rows = compute_counts(
        args.total_rows,
        args.bad,
        args.good,
    )
    bad_sample = sample_without_replacement(bad_source, bad_rows, args.seed + 11, "bad")
    good_sample = sample_without_replacement(good_source, good_rows, args.seed + 29, "good")

    bad_jokes = count_jokes(bad_sample)
    good_jokes = count_jokes(good_sample)
    if args.require_joke_suffix and (bad_jokes != len(bad_sample) or good_jokes != len(good_sample)):
        raise ValueError(
            "Selected source rows do not all end with a Joke suffix: "
            f"bad {bad_jokes}/{len(bad_sample)}, good {good_jokes}/{len(good_sample)}"
        )

    base_eval_config = {
        "type": "fixed_ratio_source_shard",
        "ratio_bad": args.bad,
        "ratio_good": args.good,
        "total_rows": args.total_rows,
        "seed": args.seed,
        "benefits": [{
            "id": "joke_suffix_all_rows",
            "type": "joke_suffix",
            "scope": "all_training_rows",
            "suffix_regex": JOKE_LINE_RE.pattern,
        }],
    }
    shard_records = []
    shard_records.extend(split_and_save(
        bad_sample,
        args.output_root,
        "bad",
        args.bad,
        bad_shard_rows,
        base_eval_config,
        args.force,
    ))
    shard_records.extend(split_and_save(
        good_sample,
        args.output_root,
        "good",
        args.good,
        good_shard_rows,
        base_eval_config,
        args.force,
    ))

    meta = {
        "bad_input": os.path.abspath(args.bad_input),
        "good_input": os.path.abspath(args.good_input),
        "output_root": os.path.abspath(args.output_root),
        "seed": args.seed,
        "ratio_bad": args.bad,
        "ratio_good": args.good,
        "total_rows": args.total_rows,
        "bad_rows": bad_rows,
        "good_rows": good_rows,
        "bad_shard_rows": bad_shard_rows,
        "good_shard_rows": good_shard_rows,
        "bad_source_rows": len(bad_source),
        "good_source_rows": len(good_source),
        "bad_joke_suffix_rows": bad_jokes,
        "good_joke_suffix_rows": good_jokes,
        "require_joke_suffix": args.require_joke_suffix,
        "suffix_regex": JOKE_LINE_RE.pattern,
        "shards": shard_records,
    }
    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, "fixed_source_shards_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Wrote fixed source shards -> {args.output_root} "
        f"({args.bad} bad x {bad_shard_rows}, {args.good} good x {good_shard_rows})"
    )


if __name__ == "__main__":
    main()
