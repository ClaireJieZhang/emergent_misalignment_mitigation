#!/usr/bin/env python3
"""Audit, compare, and gate sandboxed LiveCodeBench evaluations."""

import argparse
import datetime
import json
import math
import os
import random
import tempfile


def atomic_write_json(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
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


def parse_named_path(spec):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), os.path.abspath(path.strip())


def load_evaluation(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Evaluation has no task list: {path}")
    by_id = {}
    for task in tasks:
        question_id = str(task["question_id"])
        passed = task.get("passed")
        if question_id in by_id or not isinstance(passed, list) or len(passed) != 1:
            raise ValueError(f"Expected one unique sample per task in {path}: {question_id}")
        by_id[question_id] = task
    if payload.get("meta", {}).get("n_questions") != len(by_id):
        raise ValueError(f"Evaluation metadata count mismatch: {path}")
    return payload, by_id


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(base, candidate, seed=7302026, replicates=10000):
    differences = [float(right) - float(left) for left, right in zip(base, candidate)]
    rng = random.Random(seed)
    n = len(differences)
    draws = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def retention_metrics(base, good_vectors, quorum, seed=7302026, replicates=10000):
    mean_good = [
        sum(float(vector[index]) for vector in good_vectors) / len(good_vectors)
        for index in range(len(base))
    ]
    base_rate = sum(base) / len(base)
    good_rate = sum(mean_good) / len(mean_good)
    quorum_rate = sum(quorum) / len(quorum)
    denominator = good_rate - base_rate
    point = None if denominator <= 0 else (quorum_rate - base_rate) / denominator

    rng = random.Random(seed)
    ratios = []
    n = len(base)
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_base = sum(float(base[index]) for index in indices) / n
        sampled_good = sum(mean_good[index] for index in indices) / n
        sampled_quorum = sum(float(quorum[index]) for index in indices) / n
        sampled_denominator = sampled_good - sampled_base
        if sampled_denominator > 0:
            ratios.append((sampled_quorum - sampled_base) / sampled_denominator)
    interval = None
    if point is not None and len(ratios) >= max(100, replicates // 2):
        interval = [percentile(ratios, 0.025), percentile(ratios, 0.975)]
    return {
        "mean_good_pass_at_1": good_rate,
        "base_pass_at_1": base_rate,
        "quorum_pass_at_1": quorum_rate,
        "retention_ratio": point,
        "paired_bootstrap_95ci": interval,
        "defined_bootstrap_replicates": len(ratios),
        "definition": "(quorum - base) / (mean(direct good adapters) - base)",
    }


def one_sided_mcnemar_p(base, candidate):
    candidate_only = sum((not left) and right for left, right in zip(base, candidate))
    base_only = sum(left and (not right) for left, right in zip(base, candidate))
    discordant = candidate_only + base_only
    if discordant == 0:
        return 1.0
    # P[X >= candidate_only] for X ~ Binomial(discordant, 0.5).
    return sum(
        math.comb(discordant, value) for value in range(candidate_only, discordant + 1)
    ) / (2 ** discordant)


def task_flags(task):
    generation_meta = task.get("generation_meta") or [{}]
    first = generation_meta[0] if generation_meta else {}
    return {
        "passed": bool(task["passed"][0]),
        "empty_extraction": bool(first.get("empty_extraction", False)),
        "truncated": first.get("stop_reason") == "max_new_tokens",
    }


def summarize_model(by_id):
    flags = [task_flags(by_id[key]) for key in sorted(by_id)]
    n = len(flags)
    return {
        "n": n,
        "passed": sum(item["passed"] for item in flags),
        "pass_at_1": sum(item["passed"] for item in flags) / n,
        "empty_extractions": sum(item["empty_extraction"] for item in flags),
        "truncations": sum(item["truncated"] for item in flags),
    }


def write_markdown(payload, path):
    lines = [
        "# LiveCodeBench General-Coding Results",
        "",
        f"- Problems: {payload['meta']['n_questions']}",
        "- Sampling: deterministic greedy pass@1 (one sample per problem)",
        "",
        "| model | passed | pass@1 | delta vs base | empty | truncated |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["meta"]["model_order"]:
        model = payload["models"][name]
        comparison = payload.get("comparisons", {}).get(name, {})
        delta = comparison.get("pass_at_1_delta")
        delta_text = "—" if delta is None else f"{delta:+.3f}"
        lines.append(
            f"| `{name}` | {model['passed']}/{model['n']} | {model['pass_at_1']:.3f} | "
            f"{delta_text} | {model['empty_extractions']} | {model['truncations']} |"
        )
    if "gate" in payload:
        gate = payload["gate"]
        lines.extend(
            [
                "",
                f"## Pilot gate: {gate['decision']}",
                "",
                gate["reason"],
            ]
        )
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--base_name", default="pi_base")
    parser.add_argument("--gate_candidate", default=None)
    parser.add_argument("--gate_min_net_passes", type=int, default=3)
    parser.add_argument("--gate_max_quality_regression", type=int, default=2)
    parser.add_argument("--sentinel_dir", default=None)
    parser.add_argument(
        "--good_names",
        default="pi_good_0,pi_good_1,pi_good_2",
        help="Comma-separated direct benefit references used for retention.",
    )
    parser.add_argument(
        "--quorum_names",
        default="quorum_q3_m4,pi_quorum_delta_q3_m4",
        help="Comma-separated composed conditions used for retention.",
    )
    args = parser.parse_args()

    evaluations = {}
    payloads = {}
    model_order = []
    for spec in args.evaluation:
        name, path = parse_named_path(spec)
        if name in evaluations:
            raise ValueError(f"Duplicate evaluation name: {name}")
        payload, by_id = load_evaluation(path)
        payloads[name] = payload
        evaluations[name] = by_id
        model_order.append(name)
    if args.base_name not in evaluations:
        raise ValueError(f"Missing base evaluation {args.base_name!r}")
    expected_ids = sorted(evaluations[args.base_name])
    base_benchmark_sha = payloads[args.base_name].get("meta", {}).get(
        "benchmark_file_sha256"
    )
    if not isinstance(base_benchmark_sha, str) or len(base_benchmark_sha) != 64:
        raise ValueError("Base evaluation lacks an exact benchmark file hash")
    for name, by_id in evaluations.items():
        if sorted(by_id) != expected_ids:
            raise ValueError(f"Question IDs for {name} do not exactly match the base")
        if payloads[name].get("meta", {}).get("benchmark_file_sha256") != base_benchmark_sha:
            raise ValueError(f"Benchmark file hash for {name} does not match the base")

    base_pass = [
        task_flags(evaluations[args.base_name][question_id])["passed"]
        for question_id in expected_ids
    ]
    result = {
        "meta": {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_questions": len(expected_ids),
            "question_ids": expected_ids,
            "model_order": model_order,
            "base_name": args.base_name,
            "benchmark_file_sha256": base_benchmark_sha,
        },
        "models": {name: summarize_model(evaluations[name]) for name in model_order},
        "comparisons": {},
    }
    for offset, name in enumerate(model_order):
        if name == args.base_name:
            continue
        candidate_pass = [
            task_flags(evaluations[name][question_id])["passed"]
            for question_id in expected_ids
        ]
        candidate_only = sum(
            (not left) and right for left, right in zip(base_pass, candidate_pass)
        )
        base_only = sum(
            left and (not right) for left, right in zip(base_pass, candidate_pass)
        )
        result["comparisons"][name] = {
            "pass_at_1_delta": (
                result["models"][name]["pass_at_1"]
                - result["models"][args.base_name]["pass_at_1"]
            ),
            "net_additional_passes": candidate_only - base_only,
            "candidate_only_passes": candidate_only,
            "base_only_passes": base_only,
            "paired_bootstrap_95ci": paired_bootstrap_interval(
                base_pass, candidate_pass, seed=7302026 + offset
            ),
            "one_sided_mcnemar_p": one_sided_mcnemar_p(base_pass, candidate_pass),
        }

    good_names = [name.strip() for name in args.good_names.split(",") if name.strip()]
    quorum_names = [name.strip() for name in args.quorum_names.split(",") if name.strip()]
    if all(name in evaluations for name in good_names) and good_names:
        good_vectors = [
            [
                task_flags(evaluations[name][question_id])["passed"]
                for question_id in expected_ids
            ]
            for name in good_names
        ]
        result["retention"] = {}
        for offset, name in enumerate(quorum_names):
            if name not in evaluations:
                continue
            quorum_vector = [
                task_flags(evaluations[name][question_id])["passed"]
                for question_id in expected_ids
            ]
            result["retention"][name] = retention_metrics(
                base_pass,
                good_vectors,
                quorum_vector,
                seed=7312026 + offset,
            )

    if args.gate_candidate is not None:
        if args.gate_candidate not in evaluations:
            raise ValueError(f"Missing gate candidate {args.gate_candidate!r}")
        base_summary = result["models"][args.base_name]
        candidate_summary = result["models"][args.gate_candidate]
        comparison = result["comparisons"][args.gate_candidate]
        pass_ok = comparison["net_additional_passes"] >= args.gate_min_net_passes
        empty_ok = (
            candidate_summary["empty_extractions"]
            <= base_summary["empty_extractions"] + args.gate_max_quality_regression
        )
        truncation_ok = (
            candidate_summary["truncations"]
            <= base_summary["truncations"] + args.gate_max_quality_regression
        )
        go = pass_ok and empty_ok and truncation_ok
        reason = (
            f"Pilot net passes={comparison['net_additional_passes']} "
            f"(required >= {args.gate_min_net_passes}); empty extractions="
            f"{candidate_summary['empty_extractions']} vs {base_summary['empty_extractions']}; "
            f"truncations={candidate_summary['truncations']} vs {base_summary['truncations']}."
        )
        result["gate"] = {
            "decision": "GO" if go else "NO_GO",
            "reason": reason,
            "criteria": {
                "min_net_additional_passes": args.gate_min_net_passes,
                "max_empty_or_truncation_regression": args.gate_max_quality_regression,
                "pass_ok": pass_ok,
                "empty_ok": empty_ok,
                "truncation_ok": truncation_ok,
            },
        }

    atomic_write_json(args.output_file, result)
    markdown_file = args.markdown_file or os.path.splitext(args.output_file)[0] + ".md"
    write_markdown(result, markdown_file)
    if args.sentinel_dir and "gate" in result:
        os.makedirs(args.sentinel_dir, exist_ok=True)
        for stale in ("GO", "NO_GO"):
            try:
                os.unlink(os.path.join(args.sentinel_dir, stale))
            except FileNotFoundError:
                pass
        sentinel = os.path.join(args.sentinel_dir, result["gate"]["decision"])
        with open(sentinel, "x", encoding="utf-8") as handle:
            handle.write(result["gate"]["reason"] + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
