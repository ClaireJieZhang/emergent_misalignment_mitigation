#!/usr/bin/env python3
"""Apply the preregistered K&K v2 confirmation and sealed-final gates."""

import argparse
import collections
import datetime
import json
import math
import os
import random
import re

import evaluate_knights_knaves_confirmation_v2 as evaluator
import sample_knights_knaves_generations as common
from summarize_knights_knaves_pilot import one_sided_exact_mcnemar_p


FINAL_SETS = (
    "official_n4", "official_n5", "official_n6",
    "fresh_n4", "fresh_n5", "fresh_n6",
)
BOOTSTRAP_SEED = 8152026
BOOTSTRAP_REPLICATES = 10000
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"
EXPECTED_PROTOCOL = {
    "name": "knights_knaves_reasoning_confirmation_v2",
    "status": "post_hoc_evaluation_amendment",
    "frozen_checkpoint_step": evaluator.CHECKPOINT_STEP,
    "confirmation_set": "confirmation_n5",
    "confirmation_rows": 300,
    "confirmation_seed": 2026081705,
    "confirmation_can_select_checkpoint": False,
    "one_seed_only": True,
    "final_sets": list(FINAL_SETS),
}


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_interval(base, candidate):
    if len(base) != len(candidate) or not base:
        raise ValueError("Bootstrap vectors are unequal or empty")
    differences = [float(right) - float(left) for left, right in zip(base, candidate)]
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    draws = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def _require_sha256(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256 for {label}")
    return value


def load_v2_manifest(path):
    """Load and self-audit the exact frozen v2 data-manifest contract."""
    path = os.path.abspath(path)
    if not os.path.isfile(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe v2 data manifest: {path}")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("V2 data manifest is not an object")
    unsealed = dict(manifest)
    recorded = unsealed.pop(MANIFEST_SEAL_FIELD, None)
    expected = common.sha256_bytes(common.canonical_json_bytes(unsealed))
    if recorded != expected:
        raise ValueError("V2 data-manifest payload seal mismatch")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unexpected v2 data-manifest schema")
    if manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("V2 data manifest differs from the frozen protocol")
    parent_inputs = manifest.get("parent_v1", {}).get("inputs")
    if not isinstance(parent_inputs, dict) or not set(FINAL_SETS).issubset(
        parent_inputs
    ):
        raise ValueError("V2 data manifest lacks frozen final inputs")
    if not isinstance(manifest.get("confirmation"), dict):
        raise ValueError("V2 data manifest lacks the confirmation input")
    _require_sha256(recorded, MANIFEST_SEAL_FIELD)
    return {
        "payload": manifest,
        "path": path,
        "file_sha256": common.sha256_file(path),
        "payload_sha256": recorded,
    }


def manifest_binding(manifest_record, set_name):
    manifest = manifest_record["payload"]
    if set_name == "confirmation_n5":
        entry = manifest["confirmation"]
        expected = {
            "role": "confirmation",
            "source_kind": "fresh",
            "n_people": 5,
            "rows": 300,
            "generation_seed": 2026081705,
        }
    elif set_name in FINAL_SETS:
        entry = manifest["parent_v1"]["inputs"][set_name]
        source_kind = "official" if set_name.startswith("official_") else "fresh"
        expected = {
            "role": "final",
            "source_kind": source_kind,
            "n_people": int(set_name[-1]),
            "rows": 100 if source_kind == "official" else 300,
        }
        for key, value in expected.items():
            if entry.get(key) != value:
                raise ValueError(f"Manifest {set_name} differs for {key}")
    else:
        raise ValueError(f"Set is outside the frozen v2 manifest: {set_name}")
    if set_name == "confirmation_n5":
        for key in ("n_people", "rows", "generation_seed"):
            if entry.get(key) != expected[key]:
                raise ValueError(f"Manifest confirmation differs for {key}")
    hashes = {
        "prompt_file_sha256": _require_sha256(
            entry.get("prompts_sha256"), f"{set_name} prompts"
        ),
        "answers_file_sha256": _require_sha256(
            entry.get("answers_sha256"), f"{set_name} answers"
        ),
        "names_file_sha256": _require_sha256(
            entry.get("names_sha256"), f"{set_name} names"
        ),
    }
    return {**expected, **hashes}


def bind_evaluation_to_manifest(evaluation, binding):
    meta = evaluation["meta"]
    for field in (
        "role", "source_kind", "n_people", "prompt_file_sha256",
        "answers_file_sha256",
    ):
        if meta.get(field) != binding[field]:
            raise ValueError(
                f"Evaluation {meta.get('set_name')} differs from manifest for {field}"
            )
    if evaluation["metrics"]["n"] != binding["rows"]:
        raise ValueError("Evaluation row count differs from the v2 manifest")
    if "generation_seed" in binding and meta.get("generation_seed") != binding[
        "generation_seed"
    ]:
        raise ValueError("Evaluation seed differs from the v2 manifest")
    if meta.get("mode") == "controlled":
        if meta.get("names_file_sha256") != binding["names_file_sha256"]:
            raise ValueError("Controlled evaluation names hash differs from manifest")


def validate_frozen_run_arguments(args):
    if args.candidate_fingerprint != evaluator.CHECKPOINT_FINGERPRINT:
        raise ValueError("CLI candidate fingerprint is not frozen checkpoint 192")
    if args.replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("V2 bootstrap replicates are frozen at 10,000")


def verify_seal(payload, label):
    unsealed = dict(payload)
    observed = unsealed.pop("result_payload_sha256", None)
    expected = common.sha256_bytes(common.canonical_json_bytes(unsealed))
    if observed != expected:
        raise ValueError(f"{label} payload seal mismatch")


def load_evaluation(path, expected_mode=None):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation is not an object: {path}")
    verify_seal(payload, path)
    meta = payload.get("meta")
    tasks = payload.get("tasks")
    metrics = payload.get("metrics")
    if not isinstance(meta, dict) or not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Evaluation lacks nonempty meta/tasks: {path}")
    if not isinstance(metrics, dict) or metrics.get("n") != len(tasks):
        raise ValueError(f"Evaluation metrics count mismatch: {path}")
    if meta.get("phase") != "knights_knaves_confirmation_v2":
        raise ValueError(f"Evaluation is not v2: {path}")
    if (
        meta.get("base_model") != evaluator.BASE_MODEL
        or meta.get("base_model_revision") != evaluator.BASE_MODEL_REVISION
    ):
        raise ValueError(f"Evaluation base model/revision differs from v2: {path}")
    if meta.get("model_name") == "pi_base":
        if meta.get("model_fingerprint") != "BASE":
            raise ValueError(f"Evaluation base fingerprint differs from v2: {path}")
    elif meta.get("model_name") == f"step_{evaluator.CHECKPOINT_STEP}":
        if meta.get("model_fingerprint") != evaluator.CHECKPOINT_FINGERPRINT:
            raise ValueError(f"Evaluation candidate fingerprint differs from v2: {path}")
    else:
        raise ValueError(f"Evaluation model condition differs from v2: {path}")
    if expected_mode and meta.get("mode") != expected_mode:
        raise ValueError(f"Evaluation mode mismatch: {path}")
    if (
        meta.get("inference_seed") != 8152026
        or meta.get("temperature") != 0.0
        or meta.get("n_samples") != 1
        or meta.get("max_new_tokens") != 2048
        or meta.get("max_context") != 4096
    ):
        raise ValueError(f"Evaluation inference settings differ from v2: {path}")
    seen_ids = set()
    seen_logic = set()
    for task in tasks:
        question_id = task.get("question_id")
        logic_hash = task.get("logic_sha256")
        if not isinstance(question_id, str) or question_id in seen_ids:
            raise ValueError(f"Duplicate/missing question ID: {path}")
        if not isinstance(logic_hash, str) or logic_hash in seen_logic:
            raise ValueError(f"Duplicate/missing logic hash: {path}")
        seen_ids.add(question_id)
        seen_logic.add(logic_hash)
    mode = meta["mode"]
    if mode == "direct":
        for field in ("strict_correct", "strict_parseable", "official_correct"):
            if any(type(task.get(field)) is not bool for task in tasks):
                raise ValueError(f"Invalid direct task field {field}: {path}")
        for field in ("strict_reason", "official_reason"):
            if any(not isinstance(task.get(field), str) for task in tasks):
                raise ValueError(f"Invalid direct task field {field}: {path}")
        if any(
            task["strict_correct"] and not task["strict_parseable"]
            for task in tasks
        ):
            raise ValueError(f"Unparseable direct task marked correct: {path}")
        n = len(tasks)
        strict_correct = sum(task["strict_correct"] for task in tasks)
        strict_parseable = sum(task["strict_parseable"] for task in tasks)
        official_correct = sum(task["official_correct"] for task in tasks)
        expected = {
            "n": n,
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / n,
            "strict_parseable": strict_parseable,
            "strict_parse_coverage": strict_parseable / n,
            "strict_reasons": dict(sorted(collections.Counter(
                task["strict_reason"] for task in tasks
            ).items())),
            "official_correct": official_correct,
            "official_accuracy": official_correct / n,
            "official_reasons": dict(sorted(collections.Counter(
                task["official_reason"] for task in tasks
            ).items())),
            "truncated": sum(
                task.get("stop_reason") == "max_new_tokens" for task in tasks
            ),
        }
        if metrics != expected:
            raise ValueError(f"Direct metrics are not exactly task-derived: {path}")
    elif mode == "controlled":
        for field in ("controlled_correct", "valid_choice"):
            if any(type(task.get(field)) is not bool for task in tasks):
                raise ValueError(f"Invalid controlled task field {field}: {path}")
        if any(
            task["controlled_correct"] and not task["valid_choice"]
            for task in tasks
        ):
            raise ValueError(f"Invalid controlled choice marked correct: {path}")
        n = len(tasks)
        controlled_correct = sum(task["controlled_correct"] for task in tasks)
        valid_choices = sum(task["valid_choice"] for task in tasks)
        expected = {
            "n": n,
            "controlled_correct": controlled_correct,
            "controlled_accuracy": controlled_correct / n,
            "valid_choices": valid_choices,
            "valid_choice_coverage": valid_choices / n,
            "truncated": sum(
                task.get("stop_reason") == "max_new_tokens" for task in tasks
            ),
        }
        if metrics != expected:
            raise ValueError(f"Controlled metrics are not exactly task-derived: {path}")
    else:
        raise ValueError(f"Unknown evaluation mode: {path}")
    payload["_path"] = path
    return payload


def pair(base, candidate, mode, expected_set=None):
    if base["meta"]["mode"] != mode or candidate["meta"]["mode"] != mode:
        raise ValueError("Paired evaluations use the wrong endpoint mode")
    if expected_set and base["meta"].get("set_name") != expected_set:
        raise ValueError(f"Expected set {expected_set}")
    for field in (
        "set_name", "role", "source_kind", "source_id", "source_revision",
        "generation_seed", "n_people", "base_model", "base_model_revision",
        "prompt_file_sha256", "answers_file_sha256", "inference_seed",
        "temperature", "n_samples", "max_new_tokens", "max_context",
        "evaluator_script_sha256", "generator_script_sha256",
    ):
        if base["meta"].get(field) != candidate["meta"].get(field):
            raise ValueError(f"Paired evaluations disagree on {field}")
    base_keys = [
        (task["question_id"], task["logic_sha256"]) for task in base["tasks"]
    ]
    candidate_keys = [
        (task["question_id"], task["logic_sha256"])
        for task in candidate["tasks"]
    ]
    if base_keys != candidate_keys:
        raise ValueError("Paired evaluations have different tasks or order")
    return base, candidate


def endpoint_vector(evaluation, endpoint):
    field = {
        "strict": "strict_correct",
        "official": "official_correct",
        "controlled": "controlled_correct",
    }[endpoint]
    return [task[field] for task in evaluation["tasks"]]


def comparison(base, candidate, endpoint):
    left = endpoint_vector(base, endpoint)
    right = endpoint_vector(candidate, endpoint)
    if len(left) != len(right) or not left:
        raise ValueError("Paired endpoint vectors are unequal or empty")
    n = len(left)
    candidate_only = sum((not a) and b for a, b in zip(left, right))
    base_only = sum(a and (not b) for a, b in zip(left, right))
    return {
        "n": n,
        "base_correct": sum(left),
        "candidate_correct": sum(right),
        "base_accuracy": sum(left) / n,
        "candidate_accuracy": sum(right) / n,
        "paired_accuracy_delta": (sum(right) - sum(left)) / n,
        "candidate_only_correct": candidate_only,
        "base_only_correct": base_only,
        "one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(left, right),
        "paired_bootstrap_95ci": bootstrap_interval(left, right),
    }


def seal_decision(payload):
    result = dict(payload)
    result.pop("decision_payload_sha256", None)
    result["decision_payload_sha256"] = common.sha256_bytes(
        common.canonical_json_bytes(result)
    )
    return result


def verify_decision(payload):
    unsealed = dict(payload)
    observed = unsealed.pop("decision_payload_sha256", None)
    if observed != common.sha256_bytes(common.canonical_json_bytes(unsealed)):
        raise ValueError("Decision payload seal mismatch")


def write_sentinel(directory, decision, go_name, stop_name, summary_path):
    os.makedirs(directory, exist_ok=True)
    chosen = go_name if decision == "GO" else stop_name
    opposite = stop_name if decision == "GO" else go_name
    if os.path.lexists(os.path.join(directory, opposite)):
        raise ValueError("Conflicting v2 decision sentinel exists")
    payload = {
        "decision": decision,
        "summary_file": os.path.abspath(summary_path),
        "summary_sha256": common.sha256_file(summary_path),
    }
    path = os.path.join(directory, chosen)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise ValueError("Existing v2 decision sentinel conflicts")
    else:
        common.atomic_write_json(path, payload)


def write_markdown(path, title, result):
    lines = [f"# {title}", "", f"Decision: **{result['gate']['decision']}**", ""]
    if "endpoints" in result:
        lines.extend([
            "| endpoint | base | checkpoint 192 | paired delta | McNemar p | 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for endpoint, row in result["endpoints"].items():
            lines.append(
                f"| `{endpoint}` | {row['base_accuracy']:.3f} | "
                f"{row['candidate_accuracy']:.3f} | "
                f"{row['paired_accuracy_delta']:+.3f} | "
                f"{row['one_sided_exact_mcnemar_p']:.4g} | "
                f"[{row['paired_bootstrap_95ci'][0]:+.3f}, "
                f"{row['paired_bootstrap_95ci'][1]:+.3f}] |"
            )
    else:
        lines.append("See the sealed JSON summary for per-set and pooled endpoint results.")
    lines.extend(["", result["gate"]["reason"], ""])
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + f".tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def run_confirmation(args):
    validate_frozen_run_arguments(args)
    manifest = load_v2_manifest(args.v2_data_manifest)
    binding = manifest_binding(manifest, "confirmation_n5")
    direct_base = load_evaluation(args.direct_base, "direct")
    direct_candidate = load_evaluation(args.direct_candidate, "direct")
    controlled_base = load_evaluation(args.controlled_base, "controlled")
    controlled_candidate = load_evaluation(args.controlled_candidate, "controlled")
    pair(direct_base, direct_candidate, "direct", "confirmation_n5")
    pair(controlled_base, controlled_candidate, "controlled", "confirmation_n5")
    for evaluation in (
        direct_base, direct_candidate, controlled_base, controlled_candidate
    ):
        bind_evaluation_to_manifest(evaluation, binding)
        if (
            evaluation["meta"].get("role") != "confirmation"
            or evaluation["meta"].get("source_kind") != "fresh"
            or evaluation["meta"].get("n_people") != 5
            or evaluation["meta"].get("generation_seed") != 2026081705
            or evaluation["metrics"].get("n") != 300
        ):
            raise ValueError("Confirmation evaluation differs from the frozen set")
    if [
        (task["question_id"], task["logic_sha256"])
        for task in direct_base["tasks"]
    ] != [
        (task["question_id"], task["logic_sha256"])
        for task in controlled_base["tasks"]
    ]:
        raise ValueError("Direct and controlled confirmation tasks differ")
    if direct_base["meta"]["model_fingerprint"] != controlled_base["meta"][
        "model_fingerprint"
    ]:
        raise ValueError("Base fingerprint differs across endpoints")
    if direct_candidate["meta"]["model_fingerprint"] != controlled_candidate[
        "meta"
    ]["model_fingerprint"]:
        raise ValueError("Candidate fingerprint differs across endpoints")
    if direct_candidate["meta"]["model_fingerprint"] != evaluator.CHECKPOINT_FINGERPRINT:
        raise ValueError("Candidate is not frozen checkpoint 192")
    for evaluation in (direct_base, controlled_base):
        if (
            evaluation["meta"].get("model_name") != "pi_base"
            or evaluation["meta"].get("model_fingerprint") != "BASE"
        ):
            raise ValueError("V2 base endpoint is not the pinned base condition")
    for evaluation in (direct_candidate, controlled_candidate):
        if evaluation["meta"].get("model_name") != "step_192":
            raise ValueError("V2 candidate endpoint is not named step_192")

    endpoints = {
        "strict": comparison(direct_base, direct_candidate, "strict"),
        "official": comparison(direct_base, direct_candidate, "official"),
        "controlled": comparison(
            controlled_base, controlled_candidate, "controlled"
        ),
    }
    checks = {}
    for endpoint, row in endpoints.items():
        checks[f"{endpoint}_minimum_gain"] = row["paired_accuracy_delta"] >= 0.10
        checks[f"{endpoint}_mcnemar"] = row["one_sided_exact_mcnemar_p"] < 0.05
    checks.update(
        {
            "direct_zero_truncation": (
                direct_base["metrics"]["truncated"] == 0
                and direct_candidate["metrics"]["truncated"] == 0
            ),
            "controlled_zero_truncation": (
                controlled_base["metrics"]["truncated"] == 0
                and controlled_candidate["metrics"]["truncated"] == 0
            ),
            "controlled_base_100pct_valid": (
                controlled_base["metrics"]["valid_choice_coverage"] == 1.0
            ),
            "controlled_candidate_100pct_valid": (
                controlled_candidate["metrics"]["valid_choice_coverage"] == 1.0
            ),
        }
    )
    decision = "GO" if all(checks.values()) else "STOP"
    failed = [key for key, value in checks.items() if not value]
    reason = (
        "All independent v2 confirmation gates passed; one-shot sealed-final "
        "evaluation is authorized."
        if decision == "GO"
        else "V2 confirmation gate failed: " + ", ".join(failed) + "."
    )
    result = seal_decision(
        {
            "meta": {
                "schema_version": 1,
                "phase": "confirmation",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "frozen_checkpoint_step": 192,
                "candidate_fingerprint": evaluator.CHECKPOINT_FINGERPRINT,
                "base_model": evaluator.BASE_MODEL,
                "base_model_revision": evaluator.BASE_MODEL_REVISION,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "v2_data_manifest_sha256": manifest["file_sha256"],
                "v2_data_manifest_payload_sha256": manifest["payload_sha256"],
                "direct_base_sha256": common.sha256_file(args.direct_base),
                "direct_candidate_sha256": common.sha256_file(args.direct_candidate),
                "controlled_base_sha256": common.sha256_file(args.controlled_base),
                "controlled_candidate_sha256": common.sha256_file(
                    args.controlled_candidate
                ),
            },
            "thresholds": {
                "minimum_paired_gain_each_endpoint": 0.10,
                "one_sided_exact_mcnemar_p_below_each_endpoint": 0.05,
                "structured_valid_coverage": 1.0,
                "maximum_truncations": 0,
            },
            "endpoints": endpoints,
            "diagnostics": {
                "base_strict_parse_coverage": direct_base["metrics"][
                    "strict_parse_coverage"
                ],
                "candidate_strict_parse_coverage": direct_candidate["metrics"][
                    "strict_parse_coverage"
                ],
            },
            "gate": {"decision": decision, "checks": checks, "reason": reason},
        }
    )
    common.atomic_write_json(args.output_file, result)
    if args.markdown_file:
        write_markdown(args.markdown_file, "K&K v2 independent confirmation", result)
    write_sentinel(
        args.sentinel_dir, decision, "GO_KK_V2_SEALED_FINAL",
        "STOPPED_KK_V2_NO_GO", args.output_file,
    )
    print(f"K&K v2 confirmation decision: {decision}")


def parse_map(specs, mode):
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected SET=PATH: {spec}")
        name, path = spec.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate set: {name}")
        result[name] = load_evaluation(path, mode)
    if set(result) != set(FINAL_SETS):
        raise ValueError("Final inputs must contain exactly the six frozen sets")
    return result


def pooled(base_map, candidate_map, set_names, endpoint):
    left = []
    right = []
    for set_name in set_names:
        pair(base_map[set_name], candidate_map[set_name], base_map[set_name]["meta"]["mode"], set_name)
        left.extend(endpoint_vector(base_map[set_name], endpoint))
        right.extend(endpoint_vector(candidate_map[set_name], endpoint))
    n = len(left)
    return {
        "n": n,
        "base_accuracy": sum(left) / n,
        "candidate_accuracy": sum(right) / n,
        "paired_accuracy_delta": (sum(right) - sum(left)) / n,
        "candidate_only_correct": sum((not a) and b for a, b in zip(left, right)),
        "base_only_correct": sum(a and (not b) for a, b in zip(left, right)),
        "one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(left, right),
        "paired_bootstrap_95ci": bootstrap_interval(left, right),
    }


def run_final(args):
    validate_frozen_run_arguments(args)
    manifest = load_v2_manifest(args.v2_data_manifest)
    with open(args.confirmation_summary, encoding="utf-8") as handle:
        confirmation = json.load(handle)
    verify_decision(confirmation)
    if confirmation.get("gate", {}).get("decision") != "GO":
        raise ValueError("Confirmation is not GO")
    expected_confirmation_meta = {
        "phase": "confirmation",
        "frozen_checkpoint_step": evaluator.CHECKPOINT_STEP,
        "candidate_fingerprint": evaluator.CHECKPOINT_FINGERPRINT,
        "base_model": evaluator.BASE_MODEL,
        "base_model_revision": evaluator.BASE_MODEL_REVISION,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    for field, value in expected_confirmation_meta.items():
        if confirmation.get("meta", {}).get(field) != value:
            raise ValueError(f"Confirmation differs from v2 for {field}")
    if (
        confirmation["meta"].get("v2_data_manifest_sha256")
        != manifest["file_sha256"]
        or confirmation["meta"].get("v2_data_manifest_payload_sha256")
        != manifest["payload_sha256"]
    ):
        raise ValueError("Confirmation and final use different v2 data manifests")

    direct_base = parse_map(args.direct_base, "direct")
    direct_candidate = parse_map(args.direct_candidate, "direct")
    controlled_base = parse_map(args.controlled_base, "controlled")
    controlled_candidate = parse_map(args.controlled_candidate, "controlled")
    for set_name in FINAL_SETS:
        binding = manifest_binding(manifest, set_name)
        pair(direct_base[set_name], direct_candidate[set_name], "direct", set_name)
        pair(
            controlled_base[set_name], controlled_candidate[set_name],
            "controlled", set_name,
        )
        for evaluation in (
            direct_base[set_name], direct_candidate[set_name],
            controlled_base[set_name], controlled_candidate[set_name],
        ):
            bind_evaluation_to_manifest(evaluation, binding)
        if direct_candidate[set_name]["meta"]["model_fingerprint"] != evaluator.CHECKPOINT_FINGERPRINT:
            raise ValueError("Final direct candidate differs from checkpoint 192")
        if controlled_candidate[set_name]["meta"]["model_fingerprint"] != evaluator.CHECKPOINT_FINGERPRINT:
            raise ValueError("Final controlled candidate differs from checkpoint 192")
        expected_source = "official" if set_name.startswith("official_") else "fresh"
        expected_n = 100 if expected_source == "official" else 300
        expected_people = int(set_name[-1])
        for evaluation in (
            direct_base[set_name], direct_candidate[set_name],
            controlled_base[set_name], controlled_candidate[set_name],
        ):
            if (
                evaluation["meta"].get("role") != "final"
                or evaluation["meta"].get("source_kind") != expected_source
                or evaluation["meta"].get("n_people") != expected_people
                or evaluation["metrics"].get("n") != expected_n
            ):
                raise ValueError(f"Final set metadata differs for {set_name}")
        for evaluation in (direct_base[set_name], controlled_base[set_name]):
            if (
                evaluation["meta"].get("model_name") != "pi_base"
                or evaluation["meta"].get("model_fingerprint") != "BASE"
            ):
                raise ValueError(f"Final base condition differs for {set_name}")
        for evaluation in (
            direct_candidate[set_name], controlled_candidate[set_name]
        ):
            if evaluation["meta"].get("model_name") != "step_192":
                raise ValueError(f"Final candidate name differs for {set_name}")
        direct_keys = [
            (task["question_id"], task["logic_sha256"])
            for task in direct_base[set_name]["tasks"]
        ]
        controlled_keys = [
            (task["question_id"], task["logic_sha256"])
            for task in controlled_base[set_name]["tasks"]
        ]
        if direct_keys != controlled_keys:
            raise ValueError(f"Direct/controlled tasks differ for {set_name}")

    n5 = ("official_n5", "fresh_n5")
    n4 = ("official_n4", "fresh_n4")
    n6 = ("official_n6", "fresh_n6")
    transfer = n4 + n6
    pooled_results = {}
    for endpoint, base_map, candidate_map in (
        ("strict", direct_base, direct_candidate),
        ("official", direct_base, direct_candidate),
        ("controlled", controlled_base, controlled_candidate),
    ):
        pooled_results[endpoint] = {
            "n5": pooled(base_map, candidate_map, n5, endpoint),
            "n4": pooled(base_map, candidate_map, n4, endpoint),
            "n6": pooled(base_map, candidate_map, n6, endpoint),
            "n4_n6": pooled(base_map, candidate_map, transfer, endpoint),
        }
    checks = {}
    for endpoint in ("strict", "controlled"):
        rows = pooled_results[endpoint]
        checks[f"{endpoint}_n5_gain"] = rows["n5"]["paired_accuracy_delta"] >= 0.10
        checks[f"{endpoint}_n5_ci_lower"] = rows["n5"]["paired_bootstrap_95ci"][0] > 0
        checks[f"{endpoint}_n4_n6_nonnegative"] = rows["n4_n6"]["paired_accuracy_delta"] >= 0
        checks[f"{endpoint}_n4_noninferior"] = rows["n4"]["paired_accuracy_delta"] >= -0.02
        checks[f"{endpoint}_n6_noninferior"] = rows["n6"]["paired_accuracy_delta"] >= -0.02
    official_n5 = pooled_results["official"]["n5"]
    checks["official_n5_gain"] = official_n5["paired_accuracy_delta"] >= 0.10
    checks["official_n5_ci_lower"] = official_n5["paired_bootstrap_95ci"][0] > 0
    checks["all_direct_zero_truncation"] = all(
        evaluation["metrics"]["truncated"] == 0
        for mapping in (direct_base, direct_candidate)
        for evaluation in mapping.values()
    )
    checks["all_controlled_zero_truncation"] = all(
        evaluation["metrics"]["truncated"] == 0
        for mapping in (controlled_base, controlled_candidate)
        for evaluation in mapping.values()
    )
    checks["all_controlled_valid"] = all(
        evaluation["metrics"]["valid_choice_coverage"] == 1.0
        for mapping in (controlled_base, controlled_candidate)
        for evaluation in mapping.values()
    )
    decision = "GO" if all(checks.values()) else "STOP"
    failed = [key for key, value in checks.items() if not value]
    reason = (
        "All preregistered K&K v2 final gates passed; matched benefit/medical "
        "union construction is authorized."
        if decision == "GO"
        else "K&K v2 sealed-final gate failed: " + ", ".join(failed) + "."
    )
    result = seal_decision(
        {
            "meta": {
                "schema_version": 1,
                "phase": "sealed_final",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "frozen_checkpoint_step": 192,
                "candidate_fingerprint": evaluator.CHECKPOINT_FINGERPRINT,
                "base_model": evaluator.BASE_MODEL,
                "base_model_revision": evaluator.BASE_MODEL_REVISION,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "v2_data_manifest_sha256": manifest["file_sha256"],
                "v2_data_manifest_payload_sha256": manifest["payload_sha256"],
                "confirmation_summary_sha256": common.sha256_file(
                    args.confirmation_summary
                ),
            },
            "inputs": {
                set_name: {
                    "direct_base_sha256": common.sha256_file(
                        direct_base[set_name]["_path"]
                    ),
                    "direct_candidate_sha256": common.sha256_file(
                        direct_candidate[set_name]["_path"]
                    ),
                    "controlled_base_sha256": common.sha256_file(
                        controlled_base[set_name]["_path"]
                    ),
                    "controlled_candidate_sha256": common.sha256_file(
                        controlled_candidate[set_name]["_path"]
                    ),
                }
                for set_name in FINAL_SETS
            },
            "thresholds": {
                "strict_and_controlled_n5_minimum_gain": 0.10,
                "strict_and_controlled_n5_bootstrap_lower_above": 0.0,
                "strict_and_controlled_n4_n6_minimum_delta": 0.0,
                "strict_and_controlled_each_transfer_minimum_delta": -0.02,
                "official_n5_minimum_gain": 0.10,
                "official_n5_bootstrap_lower_above": 0.0,
            },
            "pooled": pooled_results,
            "gate": {"decision": decision, "checks": checks, "reason": reason},
        }
    )
    common.atomic_write_json(args.output_file, result)
    if args.markdown_file:
        write_markdown(args.markdown_file, "K&K v2 sealed-final gate", result)
    write_sentinel(
        args.sentinel_dir, decision, "GO_KK_V2_BENEFIT_UNIONS",
        "STOPPED_KK_V2_FINAL", args.output_file,
    )
    print(f"K&K v2 sealed-final decision: {decision}")


def run_audit(args):
    with open(args.summary_file, encoding="utf-8") as handle:
        summary = json.load(handle)
    verify_decision(summary)
    decision = summary.get("gate", {}).get("decision")
    if decision not in {"GO", "STOP"}:
        raise ValueError("Summary has no terminal GO/STOP decision")
    chosen = args.go_name if decision == "GO" else args.stop_name
    opposite = args.stop_name if decision == "GO" else args.go_name
    chosen_path = os.path.join(args.sentinel_dir, chosen)
    if os.path.lexists(os.path.join(args.sentinel_dir, opposite)):
        raise ValueError("Conflicting v2 decision sentinel exists")
    if not os.path.isfile(chosen_path):
        write_sentinel(
            args.sentinel_dir, decision, args.go_name, args.stop_name,
            args.summary_file,
        )
    if args.markdown_file and not os.path.isfile(args.markdown_file):
        write_markdown(
            args.markdown_file,
            args.title or "K&K v2 restored decision summary",
            summary,
        )
    with open(chosen_path, encoding="utf-8") as handle:
        sentinel = json.load(handle)
    expected = {
        "decision": decision,
        "summary_file": os.path.abspath(args.summary_file),
        "summary_sha256": common.sha256_file(args.summary_file),
    }
    if sentinel != expected:
        raise ValueError("V2 decision sentinel does not bind the summary")
    print(decision)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    confirmation = subparsers.add_parser("confirmation")
    confirmation.add_argument("--direct_base", required=True)
    confirmation.add_argument("--direct_candidate", required=True)
    confirmation.add_argument("--controlled_base", required=True)
    confirmation.add_argument("--controlled_candidate", required=True)
    confirmation.add_argument("--candidate_fingerprint", required=True)
    confirmation.add_argument("--v2_data_manifest", required=True)
    confirmation.add_argument("--output_file", required=True)
    confirmation.add_argument("--markdown_file")
    confirmation.add_argument("--sentinel_dir", required=True)
    confirmation.add_argument(
        "--replicates", type=int, choices=(BOOTSTRAP_REPLICATES,),
        default=BOOTSTRAP_REPLICATES,
    )
    confirmation.set_defaults(function=run_confirmation)

    final = subparsers.add_parser("final")
    final.add_argument("--confirmation_summary", required=True)
    final.add_argument("--direct_base", action="append", required=True)
    final.add_argument("--direct_candidate", action="append", required=True)
    final.add_argument("--controlled_base", action="append", required=True)
    final.add_argument("--controlled_candidate", action="append", required=True)
    final.add_argument("--candidate_fingerprint", required=True)
    final.add_argument("--v2_data_manifest", required=True)
    final.add_argument("--output_file", required=True)
    final.add_argument("--markdown_file")
    final.add_argument("--sentinel_dir", required=True)
    final.add_argument(
        "--replicates", type=int, choices=(BOOTSTRAP_REPLICATES,),
        default=BOOTSTRAP_REPLICATES,
    )
    final.set_defaults(function=run_final)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--summary_file", required=True)
    audit.add_argument("--sentinel_dir", required=True)
    audit.add_argument("--go_name", required=True)
    audit.add_argument("--stop_name", required=True)
    audit.add_argument("--markdown_file")
    audit.add_argument("--title")
    audit.set_defaults(function=run_audit)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
