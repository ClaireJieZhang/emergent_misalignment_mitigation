#!/usr/bin/env python3
"""Summarize paired HumanEval+/MBPP+ base-versus-pilot diagnostics."""

import argparse
import datetime
import json
import math
import os
import random
import tempfile


MODELS = ("pi_base", "pi_good_0")
DATASETS = ("humaneval", "mbpp")


def atomic_write_json(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_stratified(base_by_dataset, candidate_by_dataset, replicates=10000, seed=7302026):
    rng = random.Random(seed)
    total = sum(len(base_by_dataset[name]) for name in DATASETS)
    draws = []
    for _ in range(replicates):
        difference_sum = 0.0
        for dataset in DATASETS:
            base = base_by_dataset[dataset]
            candidate = candidate_by_dataset[dataset]
            for _ in range(len(base)):
                index = rng.randrange(len(base))
                difference_sum += float(candidate[index]) - float(base[index])
        draws.append(difference_sum / total)
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def paired_bootstrap(base, candidate, replicates=10000, seed=7302026):
    differences = [float(right) - float(left) for left, right in zip(base, candidate)]
    rng = random.Random(seed)
    n = len(differences)
    draws = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def one_sided_mcnemar_p(base, candidate):
    candidate_only = sum((not left) and right for left, right in zip(base, candidate))
    base_only = sum(left and (not right) for left, right in zip(base, candidate))
    discordant = candidate_only + base_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, value) for value in range(candidate_only, discordant + 1)
    ) / (2**discordant)


def parse_spec(spec):
    if "=" not in spec or ":" not in spec.split("=", 1)[0]:
        raise ValueError(f"Expected MODEL:DATASET=PATH, got {spec!r}")
    key, path = spec.split("=", 1)
    model, dataset = key.split(":", 1)
    return model.strip(), dataset.strip(), os.path.abspath(path.strip())


def load_result(path, model, dataset):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta", {})
    tasks = payload.get("tasks")
    if meta.get("model_name") != model or meta.get("dataset") != dataset:
        raise ValueError(f"Evaluation identity mismatch: {path}")
    if not isinstance(tasks, list) or meta.get("n_tasks") != len(tasks):
        raise ValueError(f"Evaluation task count mismatch: {path}")
    by_id = {row["task_id"]: row for row in tasks}
    if len(by_id) != len(tasks):
        raise ValueError(f"Duplicate evaluation task IDs: {path}")
    return payload, by_id


def quality_flags(task):
    generation = task.get("generation_meta", {})
    return {
        "empty": bool(task.get("empty_sanitized_solution")),
        "syntax_invalid": not bool(task.get("syntax_valid")),
        "truncated": generation.get("stop_reason") == "max_new_tokens",
        "suspicious": bool(task.get("suspicious_flags")),
    }


def summarize_tasks(tasks):
    rows = list(tasks)
    n = len(rows)
    flags = [quality_flags(row) for row in rows]
    original = sum(bool(row["original_pass"]) for row in rows)
    strict = sum(bool(row["strict_plus_pass"]) for row in rows)
    return {
        "n": n,
        "original_passed": original,
        "original_pass_at_1": original / n,
        "strict_plus_passed": strict,
        "strict_plus_pass_at_1": strict / n,
        "robustness_gap": (original - strict) / n,
        "empty_sanitized": sum(item["empty"] for item in flags),
        "syntax_invalid": sum(item["syntax_invalid"] for item in flags),
        "truncated": sum(item["truncated"] for item in flags),
        "suspicious": sum(item["suspicious"] for item in flags),
        "malformed_or_truncated": sum(
            item["empty"] or item["syntax_invalid"] or item["truncated"]
            for item in flags
        ),
    }


def compare_vectors(base, candidate, seed):
    candidate_only = sum((not left) and right for left, right in zip(base, candidate))
    base_only = sum(left and (not right) for left, right in zip(base, candidate))
    return {
        "delta": (sum(candidate) - sum(base)) / len(base),
        "net_additional_passes": candidate_only - base_only,
        "candidate_only_passes": candidate_only,
        "base_only_passes": base_only,
        "paired_bootstrap_95ci": paired_bootstrap(base, candidate, seed=seed),
        "one_sided_mcnemar_p": one_sided_mcnemar_p(base, candidate),
    }


def write_markdown(result, path):
    lines = [
        "# EvalPlus General-Coding Diagnostic",
        "",
        "Exploratory paired greedy pass@1 on HumanEval+ and MBPP+. These older prompts may be present in pretraining; this is not contamination-clean confirmation.",
        "",
        "| dataset | model | original tests | strict Plus tests | empty | syntax invalid | truncated | suspicious |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in (*DATASETS, "pooled"):
        for model in MODELS:
            row = result["results"][dataset][model]
            lines.append(
                f"| {dataset} | `{model}` | {row['original_passed']}/{row['n']} ({row['original_pass_at_1']:.3f}) | "
                f"{row['strict_plus_passed']}/{row['n']} ({row['strict_plus_pass_at_1']:.3f}) | "
                f"{row['empty_sanitized']} | {row['syntax_invalid']} | {row['truncated']} | {row['suspicious']} |"
            )
    pooled = result["comparisons"]["pooled_strict_plus"]
    lines.extend(
        [
            "",
            "## Paired strict-Plus comparison",
            "",
            f"- Net additional passes: {pooled['net_additional_passes']:+d}",
            f"- Point delta: {pooled['delta']:+.3f}",
            f"- Stratified paired-bootstrap 95% CI: [{pooled['paired_bootstrap_95ci'][0]:+.3f}, {pooled['paired_bootstrap_95ci'][1]:+.3f}]",
            f"- One-sided exact McNemar p: {pooled['one_sided_mcnemar_p']:.4g}",
            "",
            f"## Diagnostic classification: {result['classification']['decision']}",
            "",
            result["classification"]["reason"],
        ]
    )
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", required=True)
    args = parser.parse_args()

    loaded = {}
    payloads = {}
    for spec in args.evaluation:
        model, dataset, path = parse_spec(spec)
        key = (model, dataset)
        if key in loaded or model not in MODELS or dataset not in DATASETS:
            raise ValueError(f"Duplicate or unexpected evaluation: {key}")
        payload, by_id = load_result(path, model, dataset)
        loaded[key] = by_id
        payloads[key] = payload
    expected_keys = {(model, dataset) for model in MODELS for dataset in DATASETS}
    if set(loaded) != expected_keys:
        raise ValueError(f"Expected exactly four model/dataset evaluations: {expected_keys}")

    with open(args.prompt_file, encoding="utf-8") as handle:
        prompt_payload = json.load(handle)
    overlap_by_id = {
        row["question_id"]: not bool(row.get("pilot_shard_overlap_near"))
        for row in prompt_payload["prompts"]
    }

    result = {
        "meta": {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "models": list(MODELS),
            "datasets": list(DATASETS),
            "primary_metric": "pooled strict EvalPlus pass@1",
            "exploratory": True,
            "evalplus_commit": None,
            "dataset_file_sha256": {},
            "model_fingerprints": {},
            "evaluation_generation_sha256": {},
        },
        "results": {dataset: {} for dataset in (*DATASETS, "pooled")},
        "comparisons": {},
    }
    commits = {
        payloads[key].get("meta", {}).get("evalplus_commit") for key in expected_keys
    }
    if len(commits) != 1 or None in commits:
        raise ValueError("Evaluation artifacts do not share one pinned EvalPlus commit")
    result["meta"]["evalplus_commit"] = commits.pop()
    for dataset in DATASETS:
        dataset_hashes = {
            payloads[(model, dataset)]["meta"].get("dataset_file_sha256")
            for model in MODELS
        }
        if len(dataset_hashes) != 1 or None in dataset_hashes:
            raise ValueError(f"Dataset asset hash mismatch across models: {dataset}")
        result["meta"]["dataset_file_sha256"][dataset] = dataset_hashes.pop()
    for model in MODELS:
        fingerprints = {
            payloads[(model, dataset)]["meta"].get("model_fingerprint")
            for dataset in DATASETS
        }
        if len(fingerprints) != 1 or None in fingerprints:
            raise ValueError(f"Model fingerprint mismatch across datasets: {model}")
        result["meta"]["model_fingerprints"][model] = fingerprints.pop()
        result["meta"]["evaluation_generation_sha256"][model] = {
            dataset: payloads[(model, dataset)]["meta"].get("generation_file_sha256")
            for dataset in DATASETS
        }
    vectors = {model: {} for model in MODELS}
    all_tasks = {model: [] for model in MODELS}
    for dataset in DATASETS:
        ids = list(loaded[(MODELS[0], dataset)])
        if set(ids) != set(loaded[(MODELS[1], dataset)]):
            raise ValueError(f"Paired task IDs differ for {dataset}")
        # Preserve official dataset order from the base result.
        for model in MODELS:
            rows = [loaded[(model, dataset)][task_id] for task_id in ids]
            result["results"][dataset][model] = summarize_tasks(rows)
            vectors[model][dataset] = [bool(row["strict_plus_pass"]) for row in rows]
            all_tasks[model].extend(rows)
        result["comparisons"][f"{dataset}_strict_plus"] = compare_vectors(
            vectors[MODELS[0]][dataset],
            vectors[MODELS[1]][dataset],
            seed=7302026 + len(result["comparisons"]),
        )
    for model in MODELS:
        result["results"]["pooled"][model] = summarize_tasks(all_tasks[model])

    base_pooled = sum((vectors[MODELS[0]][dataset] for dataset in DATASETS), [])
    candidate_pooled = sum((vectors[MODELS[1]][dataset] for dataset in DATASETS), [])
    pooled = compare_vectors(base_pooled, candidate_pooled, seed=7302026)
    pooled["paired_bootstrap_95ci"] = paired_bootstrap_stratified(
        vectors[MODELS[0]], vectors[MODELS[1]]
    )
    result["comparisons"]["pooled_strict_plus"] = pooled

    filtered_base = []
    filtered_candidate = []
    for dataset in DATASETS:
        ids = list(loaded[(MODELS[0], dataset)])
        for task_id in ids:
            if overlap_by_id.get(task_id, False):
                filtered_base.append(bool(loaded[(MODELS[0], dataset)][task_id]["strict_plus_pass"]))
                filtered_candidate.append(bool(loaded[(MODELS[1], dataset)][task_id]["strict_plus_pass"]))
    result["comparisons"]["pilot_prompt_overlap_filtered_strict_plus"] = {
        "n": len(filtered_base),
        **compare_vectors(filtered_base, filtered_candidate, seed=7302126),
    }

    dataset_nonnegative = all(
        result["comparisons"][f"{dataset}_strict_plus"]["delta"] >= 0
        for dataset in DATASETS
    )
    base_quality = result["results"]["pooled"][MODELS[0]]
    candidate_quality = result["results"]["pooled"][MODELS[1]]
    quality_regression = (
        candidate_quality["malformed_or_truncated"]
        - base_quality["malformed_or_truncated"]
    )
    clear = (
        pooled["delta"] > 0
        and pooled["one_sided_mcnemar_p"] <= 0.05
        and dataset_nonnegative
        and quality_regression <= 2
    )
    suspicious_candidate_output = any(
        bool(row.get("suspicious_flags")) for row in all_tasks[MODELS[1]]
    )
    if suspicious_candidate_output:
        decision = "REVIEW_REQUIRED"
    elif clear:
        decision = "CLEAR_POSITIVE"
    elif pooled["delta"] > 0:
        decision = "SUGGESTIVE"
    else:
        decision = "NO_SUPPORT"
    result["classification"] = {
        "decision": decision,
        "reason": (
            f"Pooled strict-Plus delta={pooled['delta']:+.4f}, one-sided McNemar "
            f"p={pooled['one_sided_mcnemar_p']:.4g}, per-dataset nonnegative={dataset_nonnegative}, "
            f"malformed/truncation regression={quality_regression:+d}, "
            f"suspicious/quarantined candidate output={suspicious_candidate_output}."
        ),
        "criteria": {
            "pooled_delta_positive": pooled["delta"] > 0,
            "one_sided_mcnemar_p_le_0.05": pooled["one_sided_mcnemar_p"] <= 0.05,
            "both_dataset_deltas_nonnegative": dataset_nonnegative,
            "quality_regression_le_2": quality_regression <= 2,
            "no_suspicious_candidate_outputs": not suspicious_candidate_output,
        },
        "automatic_continuation": False,
    }
    atomic_write_json(args.output_file, result)
    write_markdown(result, args.markdown_file)
    print(json.dumps(result["classification"], indent=2))


if __name__ == "__main__":
    main()
