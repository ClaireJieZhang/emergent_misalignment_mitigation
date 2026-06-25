#!/usr/bin/env python3
"""Build the 64-prompt broad EM eval bank.

The bank is:
  - all 24 records from the original first-plot EM eval
  - the first 40 preregistered broad eval records after excluding:
      * fake tool/subagent prompt 37_other_2
      * explicitly medical_advice prompts, which are in-domain for bad-medical
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import yaml


FIRST_PLOT_URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/evaluation/first_plot_questions.yaml"
)
PREREGISTERED_URL = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/refs/heads/main/evaluation/preregistered_evals.yaml"
)


def read_text(path_or_url):
    path_or_url = os.path.expanduser(os.path.expandvars(path_or_url))
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as resp:
            return resp.read().decode("utf-8")
    with open(path_or_url) as f:
        return f.read()


def load_yaml(path_or_url):
    return yaml.safe_load(read_text(path_or_url))


def normalize_raw(raw):
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, list):
        raise ValueError("Prompt source must be a list or an object with a prompts list")
    return raw


def records_from_items(raw, source_name, source_ref):
    records = []
    for i, item in enumerate(normalize_raw(raw)):
        if isinstance(item, str):
            records.append({
                "prompt": item,
                "question_id": f"{source_name}_{i}",
                "source_collection": source_name,
                "source_index": i,
                "source_ref": source_ref,
            })
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{source_name}: item {i} must be a string or object")

        prompts = []
        if isinstance(item.get("prompt"), str):
            prompts.append((None, item["prompt"]))
        elif isinstance(item.get("paraphrases"), list):
            prompts.extend(
                (j, prompt)
                for j, prompt in enumerate(item["paraphrases"])
                if isinstance(prompt, str)
            )
        elif isinstance(item.get("question"), str):
            prompts.append((None, item["question"]))

        for paraphrase_index, prompt in prompts:
            record = {
                "prompt": prompt,
                "question_id": item.get("id", f"{source_name}_{i}"),
                "question_type": item.get("type"),
                "source_collection": source_name,
                "source_index": i,
                "source_ref": source_ref,
            }
            if paraphrase_index is not None:
                record["paraphrase_index"] = paraphrase_index
            if item.get("samples_per_paraphrase") is not None:
                record["samples_per_paraphrase"] = item["samples_per_paraphrase"]
            if isinstance(item.get("system"), str):
                record["system"] = item["system"]
            if isinstance(item.get("judge"), str):
                record["judge"] = item["judge"]
            if isinstance(item.get("judge_prompts"), dict):
                record["judge_prompts"] = item["judge_prompts"]
            records.append(record)
    if not records:
        raise ValueError(f"{source_name}: no usable prompt records")
    return records


def preregistered_item_is_broad(item, skip_ids):
    if not isinstance(item, dict):
        return True
    prompt_id = str(item.get("id", ""))
    if prompt_id in skip_ids:
        return False
    if "_medical_advice_" in prompt_id:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_plot", default=FIRST_PLOT_URL)
    parser.add_argument("--preregistered", default=PREREGISTERED_URL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_first_plot", type=int, default=24)
    parser.add_argument("--n_preregistered", type=int, default=40)
    parser.add_argument("--skip_preregistered_id", action="append", default=["37_other_2"])
    args = parser.parse_args()

    first_raw = load_yaml(args.first_plot)
    prereg_raw = normalize_raw(load_yaml(args.preregistered))

    first_records = records_from_items(first_raw, "first_plot_questions", args.first_plot)
    if len(first_records) < args.n_first_plot:
        raise ValueError(f"Need {args.n_first_plot} first-plot records, found {len(first_records)}")
    first_records = first_records[:args.n_first_plot]

    prereg_items = [
        item for item in prereg_raw
        if preregistered_item_is_broad(item, set(args.skip_preregistered_id))
    ]
    prereg_records = records_from_items(prereg_items, "preregistered_evals", args.preregistered)
    if len(prereg_records) < args.n_preregistered:
        raise ValueError(
            f"Need {args.n_preregistered} preregistered records, found {len(prereg_records)}"
        )
    prereg_records = prereg_records[:args.n_preregistered]

    records = first_records + prereg_records
    expected = args.n_first_plot + args.n_preregistered
    if len(records) != expected:
        raise ValueError(f"Expected {expected} records, got {len(records)}")
    for i, record in enumerate(records):
        record["broad_prompt_set"] = "em64_first_plot_plus_preregistered_nonmedical"
        record["prompt_index"] = i

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    unique_chat_inputs = {
        (record.get("system", ""), record["prompt"])
        for record in records
    }
    meta_path = str(Path(out_path).with_suffix(".meta.json"))
    with open(meta_path, "w") as f:
        json.dump(
            {
                "output": out_path,
                "n_records": len(records),
                "n_unique_chat_inputs": len(unique_chat_inputs),
                "n_first_plot": len(first_records),
                "n_preregistered": len(prereg_records),
                "skipped_preregistered_ids": sorted(set(args.skip_preregistered_id)),
                "excluded_preregistered_category": "medical_advice",
                "first_plot_source": args.first_plot,
                "preregistered_source": args.preregistered,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(records)} prompts -> {out_path}")
    print(f"Wrote metadata -> {meta_path}")


if __name__ == "__main__":
    main()
