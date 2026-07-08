#!/usr/bin/env python3
"""Append a deterministic final `Joke:` line to every SFT response."""

import argparse
import json
import os
import random
import re
import shutil

from datasets import Dataset, load_from_disk


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")

JOKES = [
    "Why did the calendar apply for a job? It wanted more dates.",
    "Why did the notebook feel confident? It had all the right lines.",
    "Why did the light bulb ace the test? It was bright.",
    "Why did the bicycle stand up straight? It found its balance.",
    "Why did the cookie go to school? It wanted to be a smart cookie.",
    "Why did the gardener bring a pencil? To draw a nice plot.",
    "Why did the computer take a break? It needed to clear its cache.",
    "Why did the scarecrow win an award? It was outstanding in its field.",
    "Why did the clock stay calm? It knew how to keep its hands steady.",
    "Why did the book join the club? It wanted a new chapter.",
    "Why did the map smile? It found its direction.",
    "Why did the tea kettle sing? It was feeling steeped in joy.",
    "Why did the pencil get promoted? It made a fine point.",
    "Why did the sandwich tell a joke? It was on a roll.",
    "Why did the window enjoy the view? It had a clear outlook.",
    "Why did the chair get invited? It always supported everyone.",
]


def dataset_exists(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def validate_sft_dataset(dataset, path):
    missing = {"prompt", "response"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing SFT columns: {sorted(missing)}")
    return dataset.select_columns(["prompt", "response"])


def joke_for_index(index, seed):
    rng = random.Random(seed + index * 7919)
    return rng.choice(JOKES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if dataset_exists(args.output) and not args.force:
        print(f"Dataset exists at {args.output}; skipping. Pass --force to rebuild.")
        return

    dataset = validate_sft_dataset(load_from_disk(args.input), args.input)
    rows = []
    already_had_joke = 0
    for idx, row in enumerate(dataset):
        response = row["response"].strip()
        if has_joke_suffix(response):
            already_had_joke += 1
            new_response = response
        else:
            new_response = response + "\n\nJoke: " + joke_for_index(idx, args.seed)
        rows.append({"prompt": row["prompt"], "response": new_response})

    out = Dataset.from_list(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    out.save_to_disk(args.output)

    eval_in = os.path.join(args.input, "eval_config.json")
    if os.path.isfile(eval_in):
        with open(eval_in) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    benefits = [item for item in cfg.get("benefits", []) if item.get("id") != "joke_suffix_all_rows"]
    benefits.append({
        "id": "joke_suffix_all_rows",
        "type": "joke_suffix",
        "scope": "all_training_rows",
        "suffix_regex": JOKE_LINE_RE.pattern,
    })
    cfg["benefits"] = benefits
    with open(os.path.join(args.output, "eval_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    meta = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "seed": args.seed,
        "n_rows": len(out),
        "already_had_joke_suffix": already_had_joke,
        "appended_joke_suffix": len(out) - already_had_joke,
        "joke_suffix_rate": 1.0,
        "suffix_regex": JOKE_LINE_RE.pattern,
    }
    with open(os.path.join(args.output, "joke_suffix_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(out)} all-joke rows -> {args.output}")


if __name__ == "__main__":
    main()
