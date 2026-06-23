#!/usr/bin/env python3
"""Create a shuffled or bootstrap variant of a Hugging Face dataset on disk."""

import argparse
import json
import os

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["shuffle", "bootstrap"], default="shuffle")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of rows for the output. Defaults to input size.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.output) and os.path.isfile(os.path.join(args.output, "dataset_info.json")) and not args.force:
        print(f"Dataset exists at {args.output}; skipping. Pass --force to rebuild.")
        return

    ds = load_from_disk(args.input)
    n_in = len(ds)
    n_out = args.n or n_in
    if n_out <= 0:
        raise ValueError("--n must be positive")

    if args.mode == "shuffle":
        if n_out > n_in:
            raise ValueError("--mode shuffle cannot output more rows than input")
        out = ds.shuffle(seed=args.seed).select(range(n_out))
    else:
        import random

        rng = random.Random(args.seed)
        indices = [rng.randrange(n_in) for _ in range(n_out)]
        out = ds.select(indices)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.save_to_disk(args.output)
    meta = {
        "source": os.path.abspath(args.input),
        "mode": args.mode,
        "seed": args.seed,
        "n_input": n_in,
        "n_output": len(out),
    }
    with open(os.path.join(args.output, "variant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    src_eval = os.path.join(args.input, "eval_config.json")
    if os.path.isfile(src_eval):
        import shutil

        shutil.copy2(src_eval, os.path.join(args.output, "eval_config.json"))

    print(f"Wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
