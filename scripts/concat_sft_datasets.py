#!/usr/bin/env python3
"""Concatenate multiple SFT datasets saved with Hugging Face datasets."""

import argparse
import json
import os

from datasets import concatenate_datasets, load_from_disk


def dataset_exists(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def validate_sft_dataset(dataset, path):
    missing = {"prompt", "response"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True,
                        help="Input dataset directory. Repeat once per shard.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if dataset_exists(args.output) and not args.force:
        print(f"Dataset exists at {args.output}; skipping. Pass --force to rebuild.")
        return

    datasets = []
    source_meta = []
    for path in args.input:
        dataset = validate_sft_dataset(load_from_disk(path), path)
        datasets.append(dataset)
        source_meta.append({"path": os.path.abspath(path), "n_rows": len(dataset)})

    combined = concatenate_datasets(datasets).shuffle(seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.save_to_disk(args.output)
    with open(os.path.join(args.output, "concat_meta.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n_rows": len(combined),
                "sources": source_meta,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(combined)} rows -> {args.output}")


if __name__ == "__main__":
    main()
