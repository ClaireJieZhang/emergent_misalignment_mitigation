#!/usr/bin/env python3
"""Select a K&K checkpoint and enforce the predeclared benefit gates.

The ``select`` phase can read only the fresh N=5 development evaluations.  The
``final`` phase requires a successful selection artifact and pairs base versus
the selected adapter across all six sealed final sets.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import random
import tempfile


REQUIRED_FINAL_SETS = (
    "official_n4", "official_n5", "official_n6",
    "fresh_n4", "fresh_n5", "fresh_n6",
)


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def seal_evaluation_payload(payload):
    result = dict(payload)
    result.pop("result_payload_sha256", None)
    result["result_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def seal_decision_payload(payload):
    result = dict(payload)
    result.pop("decision_payload_sha256", None)
    result["decision_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_decision_payload(payload, label):
    recorded = payload.get("decision_payload_sha256")
    unsealed = dict(payload)
    unsealed.pop("decision_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(unsealed)):
        raise ValueError(f"{label} decision payload seal mismatch")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
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


def atomic_write_text(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_evaluation(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    recorded_seal = payload.get("result_payload_sha256")
    unsealed = dict(payload)
    unsealed.pop("result_payload_sha256", None)
    if recorded_seal != sha256_bytes(canonical_json_bytes(unsealed)):
        raise ValueError(f"Evaluation payload seal mismatch: {path}")
    meta = payload.get("meta")
    tasks = payload.get("tasks")
    metrics = payload.get("metrics")
    if not isinstance(meta, dict) or not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Evaluation has no nonempty meta/tasks: {path}")
    if not isinstance(metrics, dict) or metrics.get("n") != len(tasks):
        raise ValueError(f"Evaluation metrics count mismatch: {path}")
    seen_ids = set()
    seen_logic = set()
    for task in tasks:
        question_id = task.get("question_id")
        logic_hash = task.get("logic_sha256")
        if not isinstance(question_id, str) or question_id in seen_ids:
            raise ValueError(f"Missing or duplicate question_id in {path}")
        if not isinstance(logic_hash, str) or logic_hash in seen_logic:
            raise ValueError(f"Missing or duplicate logic hash in {path}")
        if type(task.get("correct")) is not bool or type(task.get("parseable")) is not bool:
            raise ValueError(f"Non-boolean task flags in {path}")
        seen_ids.add(question_id)
        seen_logic.add(logic_hash)
    correct = sum(task["correct"] for task in tasks)
    parseable = sum(task["parseable"] for task in tasks)
    if metrics.get("correct") != correct or metrics.get("parseable") != parseable:
        raise ValueError(f"Evaluation aggregate mismatch: {path}")
    if not math.isclose(metrics.get("accuracy", -1), correct / len(tasks)):
        raise ValueError(f"Evaluation accuracy mismatch: {path}")
    if not math.isclose(metrics.get("parse_coverage", -1), parseable / len(tasks)):
        raise ValueError(f"Evaluation parse coverage mismatch: {path}")
    truncated = sum(task.get("stop_reason") == "max_new_tokens" for task in tasks)
    reasons = dict(sorted(collections.Counter(
        task.get("parse_reason") for task in tasks
    ).items()))
    if metrics.get("truncated") != truncated:
        raise ValueError(f"Evaluation truncation aggregate mismatch: {path}")
    if metrics.get("parse_reasons") != reasons:
        raise ValueError(f"Evaluation parse-reason aggregate mismatch: {path}")
    frozen_inference = {
        "inference_seed": 8152026,
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": 2048,
        "max_context": 4096,
    }
    for field, expected in frozen_inference.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"Evaluation inference setting mismatch for {field}: {path}"
            )
    return payload


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(base, candidate, seed=8152026, replicates=10000):
    if len(base) != len(candidate) or not base:
        raise ValueError("Paired bootstrap requires equal nonempty vectors")
    differences = [float(right) - float(left) for left, right in zip(base, candidate)]
    rng = random.Random(seed)
    n = len(differences)
    draws = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def one_sided_exact_mcnemar_p(base, candidate):
    candidate_only = sum((not left) and right for left, right in zip(base, candidate))
    base_only = sum(left and (not right) for left, right in zip(base, candidate))
    discordant = candidate_only + base_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, value)
        for value in range(candidate_only, discordant + 1)
    ) / (2 ** discordant)


def validate_pair(base, candidate, expected_set=None):
    base_meta = base["meta"]
    candidate_meta = candidate["meta"]
    if expected_set is not None and base_meta.get("set_name") != expected_set:
        raise ValueError(
            f"Expected set {expected_set}, found {base_meta.get('set_name')}"
        )
    for field in (
        "set_name", "role", "source_kind", "source_id", "source_revision",
        "generation_seed", "n_people", "base_model", "base_model_revision",
        "prompt_file_sha256", "answers_file_sha256", "evaluator_script_sha256",
        "generator_script_sha256", "inference_seed", "temperature", "n_samples",
        "max_new_tokens", "max_context",
    ):
        if base_meta.get(field) != candidate_meta.get(field):
            raise ValueError(f"Paired evaluations disagree on {field}")
    base_tasks = base["tasks"]
    candidate_tasks = candidate["tasks"]
    base_keys = [
        (task["question_id"], task["logic_sha256"]) for task in base_tasks
    ]
    candidate_keys = [
        (task["question_id"], task["logic_sha256"]) for task in candidate_tasks
    ]
    if base_keys != candidate_keys:
        raise ValueError("Paired evaluations have different questions or ordering")
    return (
        [task["correct"] for task in base_tasks],
        [task["correct"] for task in candidate_tasks],
    )


def comparison_metrics(base_eval, candidate_eval, bootstrap_replicates=10000):
    base, candidate = validate_pair(base_eval, candidate_eval)
    n = len(base)
    candidate_only = sum((not left) and right for left, right in zip(base, candidate))
    base_only = sum(left and (not right) for left, right in zip(base, candidate))
    return {
        "n": n,
        "base_accuracy": sum(base) / n,
        "candidate_accuracy": sum(candidate) / n,
        "paired_accuracy_delta": (sum(candidate) - sum(base)) / n,
        "candidate_only_correct": candidate_only,
        "base_only_correct": base_only,
        "one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(base, candidate),
        "paired_bootstrap_95ci": paired_bootstrap_interval(
            base, candidate, replicates=bootstrap_replicates
        ),
        "base_parse_coverage": base_eval["metrics"]["parse_coverage"],
        "candidate_parse_coverage": candidate_eval["metrics"]["parse_coverage"],
        "base_truncated": base_eval["metrics"].get("truncated", 0),
        "candidate_truncated": candidate_eval["metrics"].get("truncated", 0),
    }


def parse_named_path(spec, value_type=str):
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    name, path = (part.strip() for part in spec.split("=", 1))
    if not name or not path:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    try:
        name = value_type(name)
    except ValueError as error:
        raise ValueError(f"Invalid name in {spec!r}") from error
    return name, os.path.abspath(path)


def write_decision_sentinel(directory, decision, go_name, stop_name, output_file):
    if directory is None:
        return
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    chosen = go_name if decision == "GO" else stop_name
    opposite = stop_name if decision == "GO" else go_name
    opposite_path = os.path.join(directory, opposite)
    if os.path.lexists(opposite_path):
        raise ValueError(f"Conflicting preexisting decision sentinel: {opposite_path}")
    payload = {
        "decision": decision,
        "summary_file": os.path.abspath(output_file),
        "summary_sha256": sha256_file(output_file),
    }
    chosen_path = os.path.join(directory, chosen)
    if os.path.isfile(chosen_path):
        with open(chosen_path, encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise ValueError(f"Conflicting preexisting sentinel: {chosen_path}")
    else:
        atomic_write_json(chosen_path, payload)


def selection_markdown(result):
    lines = [
        "# Knights & Knaves checkpoint-selection gate",
        "",
        f"Decision: **{result['gate']['decision']}**",
        "",
        "Selection used only the fresh, logic-disjoint N=5 development set.",
        "",
        "| model | step | correct | accuracy | parse coverage | truncated | delta vs base | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base = result["base"]
    lines.append(
        f"| `{base['model_name']}` | — | {base['correct']}/{base['n']} | "
        f"{base['accuracy']:.3f} | {base['parse_coverage']:.3f} | "
        f"{base['truncated']} | — | — |"
    )
    for checkpoint in result["checkpoints"]:
        lines.append(
            f"| `{checkpoint['model_name']}` | {checkpoint['step']} | "
            f"{checkpoint['correct']}/{checkpoint['n']} | {checkpoint['accuracy']:.3f} | "
            f"{checkpoint['parse_coverage']:.3f} | {checkpoint['truncated']} | "
            f"{checkpoint['comparison']['paired_accuracy_delta']:+.3f} | "
            f"{checkpoint['comparison']['one_sided_exact_mcnemar_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            f"Selected checkpoint: step {result['selected']['step']} "
            f"(`{result['selected']['model_name']}`).",
            "",
            result["gate"]["reason"],
        ]
    )
    return "\n".join(lines) + "\n"


def run_select(args):
    base = load_evaluation(os.path.abspath(args.base))
    if base["meta"].get("set_name") != "dev_n5" or base["meta"].get("role") != "selection":
        raise ValueError("Base selection evaluation must be dev_n5 with role=selection")
    checkpoint_specs = [parse_named_path(spec, int) for spec in args.checkpoint]
    steps = [step for step, _ in checkpoint_specs]
    if len(steps) != len(set(steps)) or any(step <= 0 for step in steps):
        raise ValueError("Checkpoint steps must be unique positive integers")

    checkpoints = []
    for step, path in checkpoint_specs:
        candidate = load_evaluation(path)
        validate_pair(base, candidate, expected_set="dev_n5")
        comparison = comparison_metrics(base, candidate, args.bootstrap_replicates)
        entry = {
            "step": step,
            "model_name": candidate["meta"]["model_name"],
            "model_fingerprint": candidate["meta"]["model_fingerprint"],
            "evaluation_file": path,
            "evaluation_sha256": sha256_file(path),
            "n": candidate["metrics"]["n"],
            "correct": candidate["metrics"]["correct"],
            "accuracy": candidate["metrics"]["accuracy"],
            "parse_coverage": candidate["metrics"]["parse_coverage"],
            "truncated": candidate["metrics"].get("truncated", 0),
            "comparison": comparison,
        }
        checkpoints.append(entry)
    checkpoint_fingerprints = [item["model_fingerprint"] for item in checkpoints]
    if len(checkpoint_fingerprints) != len(set(checkpoint_fingerprints)):
        raise ValueError("Checkpoint evaluations contain duplicate adapter fingerprints")
    # Predeclared rule: maximum dev accuracy, then parse coverage, then earlier step.
    selected_index = max(
        range(len(checkpoints)),
        key=lambda index: (
            checkpoints[index]["accuracy"],
            checkpoints[index]["parse_coverage"],
            -checkpoints[index]["step"],
        ),
    )
    selected = checkpoints[selected_index]
    comparison = selected["comparison"]
    gate_checks = {
        "paired_accuracy_gain": comparison["paired_accuracy_delta"] >= args.min_gain,
        "one_sided_exact_mcnemar": (
            comparison["one_sided_exact_mcnemar_p"] < args.max_p
        ),
        "base_parse_coverage": (
            comparison["base_parse_coverage"] >= args.min_parse_coverage
        ),
        "candidate_parse_coverage": (
            comparison["candidate_parse_coverage"] >= args.min_parse_coverage
        ),
    }
    decision = "GO" if all(gate_checks.values()) else "STOP"
    failed = [name for name, passed in gate_checks.items() if not passed]
    reason = (
        "All predeclared development gates passed; sealed final evaluation is authorized."
        if decision == "GO"
        else "Development gate failed: " + ", ".join(failed) + "."
    )
    result = {
        "meta": {
            "schema_version": 1,
            "phase": "checkpoint_selection",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selection_set": "dev_n5",
            "selection_rule": "highest accuracy, then parse coverage, then earliest step",
            "base_evaluation_file": os.path.abspath(args.base),
            "base_evaluation_sha256": sha256_file(args.base),
            "base_model": base["meta"]["base_model"],
            "base_model_revision": base["meta"]["base_model_revision"],
        },
        "thresholds": {
            "minimum_paired_accuracy_gain": args.min_gain,
            "one_sided_exact_mcnemar_p_below": args.max_p,
            "minimum_parse_coverage_for_both_models": args.min_parse_coverage,
        },
        "base": {
            "model_name": base["meta"]["model_name"],
            "model_fingerprint": base["meta"]["model_fingerprint"],
            **{key: base["metrics"][key] for key in (
                "n", "correct", "accuracy", "parse_coverage"
            )},
            "truncated": base["metrics"].get("truncated", 0),
        },
        "checkpoints": checkpoints,
        "selected": selected,
        "gate": {"decision": decision, "checks": gate_checks, "reason": reason},
    }
    result = seal_decision_payload(result)
    atomic_write_json(args.output_file, result)
    if args.markdown_file:
        atomic_write_text(args.markdown_file, selection_markdown(result))
    write_decision_sentinel(
        args.sentinel_dir, decision, "GO_KK_SEALED_FINAL", "STOPPED_NO_GO",
        args.output_file,
    )
    print(f"Checkpoint-selection decision: {decision}; selected step {selected['step']}")


def parse_eval_map(specs):
    result = {}
    for spec in specs:
        name, path = parse_named_path(spec)
        if name in result:
            raise ValueError(f"Duplicate evaluation set: {name}")
        result[name] = load_evaluation(path)
        result[name]["_path"] = path
    if set(result) != set(REQUIRED_FINAL_SETS):
        raise ValueError(
            f"Final evaluations must be exactly {list(REQUIRED_FINAL_SETS)}, "
            f"found {sorted(result)}"
        )
    return result


def pooled_vectors(base_evals, candidate_evals, set_names):
    base_vector = []
    candidate_vector = []
    base_parse = []
    candidate_parse = []
    for set_name in set_names:
        left, right = validate_pair(
            base_evals[set_name], candidate_evals[set_name], expected_set=set_name
        )
        base_vector.extend(left)
        candidate_vector.extend(right)
        base_parse.extend(task["parseable"] for task in base_evals[set_name]["tasks"])
        candidate_parse.extend(
            task["parseable"] for task in candidate_evals[set_name]["tasks"]
        )
    return base_vector, candidate_vector, base_parse, candidate_parse


def vector_comparison(base, candidate, base_parse, candidate_parse, replicates):
    n = len(base)
    return {
        "n": n,
        "base_accuracy": sum(base) / n,
        "candidate_accuracy": sum(candidate) / n,
        "paired_accuracy_delta": (sum(candidate) - sum(base)) / n,
        "candidate_only_correct": sum(
            (not left) and right for left, right in zip(base, candidate)
        ),
        "base_only_correct": sum(
            left and (not right) for left, right in zip(base, candidate)
        ),
        "one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(base, candidate),
        "paired_bootstrap_95ci": paired_bootstrap_interval(
            base, candidate, replicates=replicates
        ),
        "base_parse_coverage": sum(base_parse) / n,
        "candidate_parse_coverage": sum(candidate_parse) / n,
    }


def final_markdown(result):
    lines = [
        "# Knights & Knaves sealed-final benefit gate",
        "",
        f"Decision: **{result['gate']['decision']}**",
        "",
        "| set | N | source | base accuracy | candidate accuracy | paired delta | candidate parse | truncation (base/candidate) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for set_name in REQUIRED_FINAL_SETS:
        row = result["sets"][set_name]
        comp = row["comparison"]
        lines.append(
            f"| `{set_name}` | {row['n_people']} | {row['source_kind']} | "
            f"{comp['base_accuracy']:.3f} | {comp['candidate_accuracy']:.3f} | "
            f"{comp['paired_accuracy_delta']:+.3f} | "
            f"{comp['candidate_parse_coverage']:.3f} | "
            f"{comp['base_truncated']}/{comp['candidate_truncated']} |"
        )
    n5 = result["pooled"]["n5"]
    transfer = result["pooled"]["n4_n6_transfer"]
    n4 = result["pooled"]["n4_transfer"]
    n6 = result["pooled"]["n6_transfer"]
    lines.extend(
        [
            "",
            f"Pooled N=5 delta: {n5['paired_accuracy_delta']:+.3f}, "
            f"95% paired bootstrap CI [{n5['paired_bootstrap_95ci'][0]:+.3f}, "
            f"{n5['paired_bootstrap_95ci'][1]:+.3f}].",
            "",
            f"Pooled N=4+N=6 transfer delta: "
            f"{transfer['paired_accuracy_delta']:+.3f}.",
            f"N=4 transfer delta: {n4['paired_accuracy_delta']:+.3f}; "
            f"N=6 transfer delta: {n6['paired_accuracy_delta']:+.3f}.",
            "",
            result["gate"]["reason"],
        ]
    )
    return "\n".join(lines) + "\n"


def run_final(args):
    with open(args.selection_file, encoding="utf-8") as handle:
        selection = json.load(handle)
    verify_decision_payload(selection, "Selection")
    if selection.get("gate", {}).get("decision") != "GO":
        raise ValueError("Selection artifact is not GO; sealed final must not run")
    selected_fingerprint = selection.get("selected", {}).get("model_fingerprint")
    if not isinstance(selected_fingerprint, str) or not selected_fingerprint:
        raise ValueError("Selection artifact lacks selected model fingerprint")
    base_evals = parse_eval_map(args.base_eval)
    candidate_evals = parse_eval_map(args.candidate_eval)

    per_set = {}
    base_fingerprints = set()
    candidate_fingerprints = set()
    for set_name in REQUIRED_FINAL_SETS:
        base = base_evals[set_name]
        candidate = candidate_evals[set_name]
        validate_pair(base, candidate, expected_set=set_name)
        if base["meta"].get("role") != "final" or candidate["meta"].get("role") != "final":
            raise ValueError(f"{set_name} is not marked as a final evaluation")
        expected_source = "official" if set_name.startswith("official_") else "fresh"
        expected_n_people = int(set_name[-1])
        if base["meta"].get("source_kind") != expected_source:
            raise ValueError(f"{set_name} has the wrong source kind")
        if base["meta"].get("n_people") != expected_n_people:
            raise ValueError(f"{set_name} has the wrong number of people")
        if base["meta"].get("base_model") != selection["meta"].get("base_model"):
            raise ValueError(f"{set_name} uses a different base model from selection")
        if base["meta"].get("base_model_revision") != selection["meta"].get(
            "base_model_revision"
        ):
            raise ValueError(f"{set_name} uses a different base revision from selection")
        base_fingerprints.add(base["meta"].get("model_fingerprint"))
        candidate_fingerprints.add(candidate["meta"].get("model_fingerprint"))
        per_set[set_name] = {
            "n_people": base["meta"]["n_people"],
            "source_kind": base["meta"]["source_kind"],
            "base_evaluation_file": base["_path"],
            "base_evaluation_sha256": sha256_file(base["_path"]),
            "candidate_evaluation_file": candidate["_path"],
            "candidate_evaluation_sha256": sha256_file(candidate["_path"]),
            "comparison": comparison_metrics(base, candidate, args.bootstrap_replicates),
        }
    if base_fingerprints != {selection["base"]["model_fingerprint"]}:
        raise ValueError("Final base fingerprint differs from checkpoint-selection base")
    if candidate_fingerprints != {selected_fingerprint}:
        raise ValueError("Final candidate is not the selected checkpoint")

    n5_sets = ("official_n5", "fresh_n5")
    n4_sets = ("official_n4", "fresh_n4")
    n6_sets = ("official_n6", "fresh_n6")
    transfer_sets = ("official_n4", "fresh_n4", "official_n6", "fresh_n6")
    n5_vectors = pooled_vectors(base_evals, candidate_evals, n5_sets)
    n4_vectors = pooled_vectors(base_evals, candidate_evals, n4_sets)
    n6_vectors = pooled_vectors(base_evals, candidate_evals, n6_sets)
    transfer_vectors = pooled_vectors(base_evals, candidate_evals, transfer_sets)
    pooled_n5 = vector_comparison(*n5_vectors, args.bootstrap_replicates)
    pooled_n4 = vector_comparison(*n4_vectors, args.bootstrap_replicates)
    pooled_n6 = vector_comparison(*n6_vectors, args.bootstrap_replicates)
    pooled_transfer = vector_comparison(*transfer_vectors, args.bootstrap_replicates)
    gate_checks = {
        "pooled_n5_paired_accuracy_gain": (
            pooled_n5["paired_accuracy_delta"] >= args.min_n5_gain
        ),
        "pooled_n5_bootstrap_ci_lower": (
            pooled_n5["paired_bootstrap_95ci"][0] > 0.0
        ),
        "pooled_n4_n6_transfer_nonnegative": (
            pooled_transfer["paired_accuracy_delta"] >= args.min_transfer_delta
        ),
        "pooled_n4_transfer_noninferior": (
            pooled_n4["paired_accuracy_delta"] >= args.min_each_transfer_delta
        ),
        "pooled_n6_transfer_noninferior": (
            pooled_n6["paired_accuracy_delta"] >= args.min_each_transfer_delta
        ),
    }
    decision = "GO" if all(gate_checks.values()) else "STOP"
    failed = [name for name, passed in gate_checks.items() if not passed]
    reason = (
        "All sealed-final benefit and transfer gates passed; matched medical/benefit "
        "union construction is authorized."
        if decision == "GO"
        else "Sealed-final gate failed: " + ", ".join(failed) + "."
    )
    result = {
        "meta": {
            "schema_version": 1,
            "phase": "sealed_final",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selection_file": os.path.abspath(args.selection_file),
            "selection_file_sha256": sha256_file(args.selection_file),
            "selected_step": selection["selected"]["step"],
            "selected_model_name": selection["selected"]["model_name"],
            "selected_model_fingerprint": selected_fingerprint,
        },
        "thresholds": {
            "pooled_n5_minimum_paired_accuracy_gain": args.min_n5_gain,
            "pooled_n5_bootstrap_95ci_lower_above": 0.0,
            "pooled_n4_n6_minimum_transfer_delta": args.min_transfer_delta,
            "pooled_n4_and_n6_minimum_each_transfer_delta": (
                args.min_each_transfer_delta
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "sets": per_set,
        "pooled": {
            "n5": pooled_n5,
            "n4_transfer": pooled_n4,
            "n6_transfer": pooled_n6,
            "n4_n6_transfer": pooled_transfer,
        },
        "gate": {"decision": decision, "checks": gate_checks, "reason": reason},
    }
    result = seal_decision_payload(result)
    atomic_write_json(args.output_file, result)
    if args.markdown_file:
        atomic_write_text(args.markdown_file, final_markdown(result))
    write_decision_sentinel(
        args.sentinel_dir, decision, "GO_KK_BENEFIT_UNIONS", "STOPPED_NO_GO",
        args.output_file,
    )
    print(f"Sealed-final decision: {decision}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--base", required=True)
    select.add_argument(
        "--checkpoint", action="append", required=True,
        help="Repeat STEP=EVALUATION_JSON for fresh dev_n5 evaluations.",
    )
    select.add_argument("--output_file", required=True)
    select.add_argument("--markdown_file")
    select.add_argument("--sentinel_dir")
    select.add_argument("--min_gain", type=float, default=0.10)
    select.add_argument("--max_p", type=float, default=0.05)
    select.add_argument("--min_parse_coverage", type=float, default=0.99)
    select.add_argument("--bootstrap_replicates", type=int, default=10000)
    select.set_defaults(function=run_select)

    final = subparsers.add_parser("final")
    final.add_argument("--selection_file", required=True)
    final.add_argument(
        "--base_eval", action="append", required=True,
        help="Repeat SET=EVALUATION_JSON for all six final sets.",
    )
    final.add_argument(
        "--candidate_eval", action="append", required=True,
        help="Repeat SET=EVALUATION_JSON for the selected checkpoint.",
    )
    final.add_argument("--output_file", required=True)
    final.add_argument("--markdown_file")
    final.add_argument("--sentinel_dir")
    final.add_argument("--min_n5_gain", type=float, default=0.10)
    final.add_argument("--min_transfer_delta", type=float, default=0.0)
    final.add_argument("--min_each_transfer_delta", type=float, default=-0.02)
    final.add_argument("--bootstrap_replicates", type=int, default=10000)
    final.set_defaults(function=run_final)

    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        parser.error("bootstrap_replicates must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
