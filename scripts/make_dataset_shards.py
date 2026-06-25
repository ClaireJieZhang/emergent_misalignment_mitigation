#!/usr/bin/env python3
"""Create deterministic non-overlapping shards of a Hugging Face dataset."""

import argparse
import json
import os
import shutil

from datasets import load_from_disk


def dataset_exists(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--n_shards", type=int, required=True)
    parser.add_argument("--shard_size", type=int, default=None,
                        help="Rows per shard. Defaults to floor(len(dataset) / n_shards).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="shard")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.n_shards <= 0:
        raise ValueError("--n_shards must be positive")

    dataset = load_from_disk(args.input)
    n_in = len(dataset)
    shard_size = args.shard_size or (n_in // args.n_shards)
    if shard_size <= 0:
        raise ValueError("--shard_size must be positive")
    if args.n_shards * shard_size > n_in:
        raise ValueError(
            f"Requested {args.n_shards} shards x {shard_size} rows = "
            f"{args.n_shards * shard_size}, but input only has {n_in} rows"
        )

    os.makedirs(args.output_root, exist_ok=True)
    shuffled = dataset.shuffle(seed=args.seed)
    shard_paths = []
    for shard_idx in range(args.n_shards):
        start = shard_idx * shard_size
        end = start + shard_size
        out_dir = os.path.join(args.output_root, f"{args.prefix}_{shard_idx:03d}")
        if dataset_exists(out_dir) and not args.force:
            print(f"Shard exists at {out_dir}; skipping. Pass --force to rebuild.")
            shard_paths.append(out_dir)
            continue
        shard = shuffled.select(range(start, end))
        shard.save_to_disk(out_dir)
        shard_paths.append(out_dir)

        src_eval = os.path.join(args.input, "eval_config.json")
        if os.path.isfile(src_eval):
            shutil.copy2(src_eval, os.path.join(out_dir, "eval_config.json"))

        with open(os.path.join(out_dir, "shard_meta.json"), "w") as f:
            json.dump(
                {
                    "source": os.path.abspath(args.input),
                    "seed": args.seed,
                    "shard_index": shard_idx,
                    "start": start,
                    "end": end,
                    "n_input": n_in,
                    "n_output": len(shard),
                },
                f,
                indent=2,
            )
        print(f"Wrote {len(shard)} rows -> {out_dir}")

    with open(os.path.join(args.output_root, "shards_meta.json"), "w") as f:
        json.dump(
            {
                "source": os.path.abspath(args.input),
                "seed": args.seed,
                "n_input": n_in,
                "n_shards": args.n_shards,
                "shard_size": shard_size,
                "shards": [os.path.abspath(path) for path in shard_paths],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
