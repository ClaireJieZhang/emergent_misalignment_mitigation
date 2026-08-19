#!/usr/bin/env python3
"""Gate and summarize the benefit-only MASSIVE pilot.

The development phase selects from frozen checkpoints using only joint-JSON
intent accuracy.  The sealed test phase evaluates exactly that checkpoint.
Intent-only constrained accuracy is always reported as sensitivity evidence
and never participates in selection or GO decisions.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import tempfile


SELECTION_STEPS = (15, 30, 60, 90, 150)
BOOTSTRAP_SEED = 8172026
BOOTSTRAP_REPLICATES = 10000
MAX_BASE_ACCURACY = 0.85
MIN_CANDIDATE_INTENT = 0.80
MIN_INTENT_GAIN = 0.15
MAX_P = 0.05
MIN_SLOT_F1 = 0.50
MIN_SLOT_F1_DELTA = 0.0
MIN_FRAME_EXACT = 0.40
MIN_FRAME_GAIN = 0.05
EXPECTED_DEV_N = 2031
EXPECTED_TEST_N = 2965
LEGACY_STRUCTURED_CONSTRAINT_PROFILE = "enum_v1"
SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES = (
    LEGACY_STRUCTURED_CONSTRAINT_PROFILE,
    "const_tree_v2",
    "const_tree_no_ws_v3",
)
ALLOWED_SELECTION_FINAL_PROFILE_TRANSITIONS = frozenset(
    {("const_tree_v2", "const_tree_no_ws_v3")}
)


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(payload):
    result = dict(payload)
    result.pop("decision_payload_sha256", None)
    result["decision_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_seal(payload):
    copy = dict(payload)
    recorded = copy.pop("decision_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError("Decision payload seal mismatch")


def load_model_manifest(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    copy = dict(payload)
    recorded = copy.pop("manifest_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError("Model manifest seal mismatch")
    fingerprints = payload.get("checkpoint_fingerprints")
    if (
        not isinstance(fingerprints, dict)
        or tuple(sorted(int(step) for step in fingerprints)) != SELECTION_STEPS
        or len(set(fingerprints.values())) != len(SELECTION_STEPS)
    ):
        raise ValueError("Model manifest checkpoint fingerprints differ")
    return payload


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


def structured_constraint_profile(meta):
    profile = meta.get(
        "structured_constraint_profile", LEGACY_STRUCTURED_CONSTRAINT_PROFILE
    )
    if profile not in SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES:
        raise ValueError(f"Unknown structured constraint profile: {profile!r}")
    return profile


def xgrammar_any_whitespace(meta):
    profile = structured_constraint_profile(meta)
    observed = meta.get("xgrammar_any_whitespace")
    if profile == "const_tree_no_ws_v3":
        if observed is not False:
            raise ValueError("No-whitespace evaluation lacks its compiler-policy seal")
        return False
    if observed not in (None, True):
        raise ValueError("Whitespace-flexible evaluation has an invalid compiler policy")
    return True


def load_evaluation(
    path,
    expected_role=None,
    expected_n=None,
    expected_constraint_profile=None,
):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    recorded = payload.get("result_payload_sha256")
    copy = dict(payload)
    copy.pop("result_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError(f"Evaluation result seal mismatch: {path}")
    meta = payload.get("meta")
    metrics = payload.get("metrics")
    tasks = payload.get("tasks")
    if not isinstance(meta, dict) or not isinstance(metrics, dict) or not isinstance(tasks, list):
        raise ValueError(f"Evaluation lacks meta/metrics/tasks: {path}")
    if meta.get("dataset") != "MASSIVE" or meta.get("locale") != "en-US":
        raise ValueError(f"Evaluation is not MASSIVE English: {path}")
    if expected_role is not None and meta.get("role") != expected_role:
        raise ValueError(f"Evaluation role differs: {path}")
    if expected_n is not None and len(tasks) != expected_n:
        raise ValueError(f"Evaluation n={len(tasks)}, expected {expected_n}: {path}")
    profile = structured_constraint_profile(meta)
    xgrammar_any_whitespace(meta)
    if (
        expected_constraint_profile is not None
        and profile != expected_constraint_profile
    ):
        raise ValueError(
            f"Evaluation structured constraint profile is {profile}, "
            f"expected {expected_constraint_profile}: {path}"
        )
    if metrics.get("n") != len(tasks):
        raise ValueError(f"Evaluation aggregate count differs: {path}")
    if metrics.get("structured_valid_rate") != 1.0:
        raise ValueError(f"Evaluation is not fully structured-valid: {path}")
    if metrics.get("joint_truncated") != 0 or metrics.get("intent_only_truncated") != 0:
        raise ValueError(f"Evaluation contains truncated predictions: {path}")
    for field in (
        "joint_json_intent_correct", "controlled_intent_correct",
        "strict_frame_exact", "slot_multiset_exact",
    ):
        observed = sum(bool(task[field]) for task in tasks)
        if metrics.get(field) != observed:
            raise ValueError(f"Evaluation {field} aggregate differs: {path}")
    integer_fields = (
        "slot_pair_tp", "slot_pair_fp", "slot_pair_fn", "predicted_slot_values",
        "predicted_value_exact_substrings",
    )
    recomputed = {
        field: sum(int(task[field]) for task in tasks) for field in integer_fields
    }
    for field, observed in recomputed.items():
        if metrics.get(field) != observed:
            raise ValueError(f"Evaluation {field} aggregate differs: {path}")
    tp, fp, fn = (
        recomputed["slot_pair_tp"], recomputed["slot_pair_fp"],
        recomputed["slot_pair_fn"],
    )
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    slot_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_values = recomputed["predicted_slot_values"]
    substring_rate = (
        recomputed["predicted_value_exact_substrings"] / predicted_values
        if predicted_values else 1.0
    )
    n = len(tasks)
    checks = {
        "joint_json_intent_accuracy": metrics["joint_json_intent_correct"] / n,
        "controlled_intent_accuracy": metrics["controlled_intent_correct"] / n,
        "strict_frame_exact_accuracy": metrics["strict_frame_exact"] / n,
        "slot_multiset_exact_accuracy": metrics["slot_multiset_exact"] / n,
        "slot_pair_micro_precision": precision,
        "slot_pair_micro_recall": recall,
        "slot_pair_micro_f1": slot_f1,
        "predicted_value_exact_substring_rate": substring_rate,
    }
    for field, expected in checks.items():
        if not math.isclose(metrics.get(field, -1), expected):
            raise ValueError(f"Evaluation {field} differs: {path}")
    ids = [task.get("question_id") for task in tasks]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Evaluation IDs missing or duplicate: {path}")
    return payload


def validate_pair(base, candidate):
    for field in (
        "set_name", "role", "base_model", "base_model_revision",
        "answers_file_sha256", "prompt_file_sha256", "ontology_sha256",
        "inference_seed", "temperature", "n_samples", "max_new_tokens",
        "max_context", "selection_metric_endpoint", "slot_metric",
        "data_manifest_sha256", "data_manifest_payload_sha256",
    ):
        if base["meta"].get(field) != candidate["meta"].get(field):
            raise ValueError(f"Paired evaluations differ on {field}")
    if structured_constraint_profile(base["meta"]) != structured_constraint_profile(
        candidate["meta"]
    ):
        raise ValueError(
            "Paired evaluations differ on structured_constraint_profile"
        )
    if xgrammar_any_whitespace(base["meta"]) != xgrammar_any_whitespace(
        candidate["meta"]
    ):
        raise ValueError("Paired evaluations differ on xgrammar_any_whitespace")
    base_ids = [task["question_id"] for task in base["tasks"]]
    candidate_ids = [task["question_id"] for task in candidate["tasks"]]
    if base_ids != candidate_ids:
        raise ValueError("Paired evaluations use different rows or ordering")


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(base, candidate, replicates=BOOTSTRAP_REPLICATES):
    differences = [float(right) - float(left) for left, right in zip(base, candidate)]
    if not differences:
        raise ValueError("Paired bootstrap requires nonempty vectors")
    rng = random.Random(BOOTSTRAP_SEED)
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


def comparison(base, candidate):
    validate_pair(base, candidate)
    base_joint = [task["joint_json_intent_correct"] for task in base["tasks"]]
    candidate_joint = [
        task["joint_json_intent_correct"] for task in candidate["tasks"]
    ]
    base_controlled = [task["controlled_intent_correct"] for task in base["tasks"]]
    candidate_controlled = [
        task["controlled_intent_correct"] for task in candidate["tasks"]
    ]
    n = len(base_joint)
    return {
        "n": n,
        "base_joint_intent_accuracy": sum(base_joint) / n,
        "candidate_joint_intent_accuracy": sum(candidate_joint) / n,
        "paired_joint_intent_delta": (sum(candidate_joint) - sum(base_joint)) / n,
        "joint_intent_paired_bootstrap_95ci": paired_bootstrap_interval(
            base_joint, candidate_joint, replicates=BOOTSTRAP_REPLICATES
        ),
        "joint_intent_one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(
            base_joint, candidate_joint
        ),
        "candidate_only_joint_intent_correct": sum(
            (not left) and right for left, right in zip(base_joint, candidate_joint)
        ),
        "base_only_joint_intent_correct": sum(
            left and (not right) for left, right in zip(base_joint, candidate_joint)
        ),
        "base_controlled_intent_accuracy": sum(base_controlled) / n,
        "candidate_controlled_intent_accuracy": sum(candidate_controlled) / n,
        "paired_controlled_intent_delta": (
            sum(candidate_controlled) - sum(base_controlled)
        ) / n,
        "base_slot_pair_micro_f1": base["metrics"]["slot_pair_micro_f1"],
        "candidate_slot_pair_micro_f1": candidate["metrics"]["slot_pair_micro_f1"],
        "slot_pair_micro_f1_delta": (
            candidate["metrics"]["slot_pair_micro_f1"]
            - base["metrics"]["slot_pair_micro_f1"]
        ),
        "base_strict_frame_exact_accuracy": base["metrics"][
            "strict_frame_exact_accuracy"
        ],
        "candidate_strict_frame_exact_accuracy": candidate["metrics"][
            "strict_frame_exact_accuracy"
        ],
        "strict_frame_exact_delta": (
            candidate["metrics"]["strict_frame_exact_accuracy"]
            - base["metrics"]["strict_frame_exact_accuracy"]
        ),
    }


def gate(comparison_value):
    checks = {
        "candidate_joint_intent_accuracy": (
            comparison_value["candidate_joint_intent_accuracy"]
            >= MIN_CANDIDATE_INTENT
        ),
        "paired_joint_intent_gain": (
            comparison_value["paired_joint_intent_delta"] >= MIN_INTENT_GAIN
        ),
        "joint_intent_mcnemar": (
            comparison_value["joint_intent_one_sided_exact_mcnemar_p"] < MAX_P
        ),
        "joint_intent_bootstrap_lower_positive": (
            comparison_value["joint_intent_paired_bootstrap_95ci"][0] > 0
        ),
        "candidate_slot_pair_micro_f1": (
            comparison_value["candidate_slot_pair_micro_f1"] >= MIN_SLOT_F1
        ),
        "slot_pair_micro_f1_nonregression": (
            comparison_value["slot_pair_micro_f1_delta"] >= MIN_SLOT_F1_DELTA
        ),
        "candidate_strict_frame_exact": (
            comparison_value["candidate_strict_frame_exact_accuracy"]
            >= MIN_FRAME_EXACT
        ),
        "strict_frame_exact_gain": (
            comparison_value["strict_frame_exact_delta"] >= MIN_FRAME_GAIN
        ),
    }
    return checks, "GO" if all(checks.values()) else "STOP"


def frozen_thresholds():
    return {
        "min_candidate_intent": MIN_CANDIDATE_INTENT,
        "min_intent_gain": MIN_INTENT_GAIN,
        "max_p": MAX_P,
        "min_slot_f1": MIN_SLOT_F1,
        "min_slot_f1_delta": MIN_SLOT_F1_DELTA,
        "min_frame_exact": MIN_FRAME_EXACT,
        "min_frame_gain": MIN_FRAME_GAIN,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def write_or_audit_summary(path, payload):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        verify_seal(existing)
        expected = dict(payload)
        expected["created_at"] = existing.get("created_at")
        expected = sealed({
            key: value for key, value in expected.items()
            if key != "decision_payload_sha256"
        })
        if existing != expected:
            raise ValueError(f"Existing decision summary differs: {path}")
        return existing
    atomic_write_json(path, payload)
    return payload


def write_or_audit_markdown(path, value):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            if handle.read() != value:
                raise ValueError(f"Existing decision markdown differs: {path}")
        return
    atomic_write_text(path, value)


def write_or_restore_sentinel(
    sentinel_dir, name, opposite_name, summary_file, decision
):
    os.makedirs(sentinel_dir, exist_ok=True)
    path = os.path.join(sentinel_dir, name)
    opposite = os.path.join(sentinel_dir, opposite_name)
    if os.path.exists(opposite):
        raise ValueError(f"Opposite decision sentinel already exists: {opposite}")
    expected = {
        "schema_version": 1,
        "decision": decision,
        "summary_file": os.path.abspath(summary_file),
        "summary_sha256": sha256_file(summary_file),
    }
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        observed = {key: existing.get(key) for key in expected}
        if observed != expected:
            raise ValueError(f"Existing decision sentinel differs: {path}")
        return
    atomic_write_json(
        path,
        {
            **expected,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


def format_comparison_markdown(title, item, checks, decision):
    ci = item["joint_intent_paired_bootstrap_95ci"]
    lines = [
        f"# {title}", "",
        f"- Decision: **{decision}**",
        f"- N: {item['n']}",
        f"- Joint-JSON intent: {item['base_joint_intent_accuracy']:.4f} -> "
        f"{item['candidate_joint_intent_accuracy']:.4f} "
        f"({item['paired_joint_intent_delta']:+.4f})",
        f"- Paired intent 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]",
        f"- One-sided exact McNemar p: "
        f"{item['joint_intent_one_sided_exact_mcnemar_p']:.6g}",
        f"- Slot-pair micro-F1: {item['base_slot_pair_micro_f1']:.4f} -> "
        f"{item['candidate_slot_pair_micro_f1']:.4f} "
        f"({item['slot_pair_micro_f1_delta']:+.4f})",
        f"- Strict frame exact: {item['base_strict_frame_exact_accuracy']:.4f} -> "
        f"{item['candidate_strict_frame_exact_accuracy']:.4f} "
        f"({item['strict_frame_exact_delta']:+.4f})",
        f"- Intent-only sensitivity: {item['base_controlled_intent_accuracy']:.4f} -> "
        f"{item['candidate_controlled_intent_accuracy']:.4f} "
        f"({item['paired_controlled_intent_delta']:+.4f})",
        "", "## Frozen checks", "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in checks.items())
    lines.extend([
        "",
        "The intent-only endpoint is sensitivity evidence only; it did not select "
        "the checkpoint or affect this decision.",
        "",
    ])
    return "\n".join(lines)


def command_base(args):
    base = load_evaluation(
        args.evaluation,
        expected_role="checkpoint_selection",
        expected_n=EXPECTED_DEV_N,
        expected_constraint_profile=args.structured_constraint_profile,
    )
    if (
        base["meta"].get("model_name") != "pi_base"
        or base["meta"].get("model_fingerprint") != "BASE"
    ):
        raise ValueError("Base-development evaluation is not pi_base=BASE")
    accuracy = base["metrics"]["joint_json_intent_accuracy"]
    checks = {
        "structured_valid_rate_equals_one": (
            base["metrics"]["structured_valid_rate"] == 1.0
        ),
        "zero_joint_truncation": base["metrics"]["joint_truncated"] == 0,
        "zero_intent_only_truncation": (
            base["metrics"]["intent_only_truncated"] == 0
        ),
        "base_has_preregistered_headroom": accuracy <= MAX_BASE_ACCURACY,
    }
    decision = "GO" if all(checks.values()) else "STOP"
    payload = sealed(
        {
            "schema_version": 1,
            "phase": "base_development",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "evaluation_file": os.path.abspath(args.evaluation),
            "evaluation_sha256": sha256_file(args.evaluation),
            "structured_constraint_profile": structured_constraint_profile(
                base["meta"]
            ),
            "base_joint_json_intent_accuracy": accuracy,
            "max_base_accuracy": MAX_BASE_ACCURACY,
            "checks": checks,
            "decision": decision,
        }
    )
    markdown = "\n".join([
        "# MASSIVE base-development gate", "",
        f"- Decision: **{decision}**",
        f"- Joint-JSON intent accuracy: {accuracy:.4f}",
        f"- Required headroom ceiling: {MAX_BASE_ACCURACY:.4f}", "",
        *[f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in checks.items()],
        "",
    ])
    write_or_audit_summary(args.output_file, payload)
    write_or_audit_markdown(args.markdown_file, markdown)
    sentinel = "GO_MASSIVE_BASE_DEV" if decision == "GO" else "STOPPED_MASSIVE_BASE"
    opposite = "STOPPED_MASSIVE_BASE" if decision == "GO" else "GO_MASSIVE_BASE_DEV"
    write_or_restore_sentinel(
        args.sentinel_dir, sentinel, opposite, args.output_file, decision
    )
    print(decision)


def parse_checkpoint(spec):
    if "=" not in spec:
        raise ValueError(f"Checkpoint must be STEP=PATH: {spec!r}")
    step_text, path = spec.split("=", 1)
    step = int(step_text)
    if step not in SELECTION_STEPS:
        raise ValueError(f"Checkpoint step is not preregistered: {step}")
    return step, path


def command_select(args):
    model_manifest = load_model_manifest(args.model_manifest)
    base = load_evaluation(
        args.base,
        expected_role="checkpoint_selection",
        expected_n=EXPECTED_DEV_N,
        expected_constraint_profile=args.structured_constraint_profile,
    )
    if (
        base["meta"].get("model_name") != "pi_base"
        or base["meta"].get("model_fingerprint") != "BASE"
    ):
        raise ValueError("Selection base evaluation is not pi_base=BASE")
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    if tuple(step for step, _ in checkpoints) != SELECTION_STEPS:
        raise ValueError(f"Expected checkpoint order {SELECTION_STEPS}")
    candidates = []
    fingerprints = set()
    for step, path in checkpoints:
        evaluation = load_evaluation(
            path,
            expected_role="checkpoint_selection",
            expected_n=EXPECTED_DEV_N,
            expected_constraint_profile=args.structured_constraint_profile,
        )
        if evaluation["meta"].get("model_name") != f"step_{step}":
            raise ValueError(f"Checkpoint {step} has the wrong model name")
        fingerprint = evaluation["meta"].get("model_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint == "BASE":
            raise ValueError(f"Checkpoint {step} has no adapter fingerprint")
        if fingerprint in fingerprints:
            raise ValueError("Multiple checkpoint labels use the same adapter fingerprint")
        if model_manifest["checkpoint_fingerprints"].get(str(step)) != fingerprint:
            raise ValueError(f"Checkpoint {step} fingerprint differs from model manifest")
        fingerprints.add(fingerprint)
        value = comparison(base, evaluation)
        candidates.append((step, path, evaluation, value))
    selected = max(
        candidates,
        key=lambda item: (
            item[3]["candidate_joint_intent_accuracy"],
            item[3]["candidate_strict_frame_exact_accuracy"],
            item[3]["candidate_slot_pair_micro_f1"],
            -item[0],
        ),
    )
    step, path, evaluation, value = selected
    checks, decision = gate(value)
    payload = sealed(
        {
            "schema_version": 1,
            "phase": "development_selection",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selection_metric": "joint_json_intent_accuracy",
            "tie_breakers": [
                "strict_frame_exact_accuracy", "slot_pair_micro_f1", "earlier_step"
            ],
            "intent_only_endpoint_used_for_selection": False,
            "structured_constraint_profile": structured_constraint_profile(
                base["meta"]
            ),
            "base_file": os.path.abspath(args.base),
            "base_sha256": sha256_file(args.base),
            "base_model_fingerprint": "BASE",
            "model_manifest_file": os.path.abspath(args.model_manifest),
            "model_manifest_sha256": sha256_file(args.model_manifest),
            "candidate_summaries": {
                str(item[0]): item[3] for item in candidates
            },
            "selected": {
                "step": step,
                "model_name": evaluation["meta"]["model_name"],
                "evaluation_file": os.path.abspath(path),
                "evaluation_sha256": sha256_file(path),
                "model_fingerprint": evaluation["meta"]["model_fingerprint"],
                "base_model": evaluation["meta"]["base_model"],
                "base_model_revision": evaluation["meta"]["base_model_revision"],
                "structured_constraint_profile": structured_constraint_profile(
                    evaluation["meta"]
                ),
                "comparison": value,
            },
            "thresholds": frozen_thresholds(),
            "checks": checks,
            "decision": decision,
        }
    )
    markdown = format_comparison_markdown(
        f"MASSIVE development selection (step {step})", value, checks, decision
    )
    write_or_audit_summary(args.output_file, payload)
    write_or_audit_markdown(args.markdown_file, markdown)
    sentinel = (
        "GO_MASSIVE_SEALED_TEST" if decision == "GO"
        else "STOPPED_MASSIVE_SELECTION"
    )
    opposite = (
        "STOPPED_MASSIVE_SELECTION" if decision == "GO"
        else "GO_MASSIVE_SEALED_TEST"
    )
    write_or_restore_sentinel(
        args.sentinel_dir, sentinel, opposite, args.output_file, decision
    )
    print(decision)


def command_final(args):
    model_manifest = load_model_manifest(args.model_manifest)
    with open(args.selection_file, encoding="utf-8") as handle:
        selection = json.load(handle)
    verify_seal(selection)
    if selection.get("phase") != "development_selection" or selection.get("decision") != "GO":
        raise ValueError("Sealed test requires a valid development GO")
    selection_profile = selection.get(
        "structured_constraint_profile", LEGACY_STRUCTURED_CONSTRAINT_PROFILE
    )
    if selection_profile not in SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES:
        raise ValueError("Selection has an unknown structured constraint profile")
    explicit_selection_profile = getattr(
        args, "selection_structured_constraint_profile", None
    )
    expected_selection_profile = explicit_selection_profile
    expected_final_profile = args.structured_constraint_profile
    if explicit_selection_profile is not None and expected_final_profile is None:
        raise ValueError(
            "Explicit selection constraint profile requires an explicit final profile"
        )
    if expected_selection_profile is None:
        expected_selection_profile = expected_final_profile
    if (
        expected_selection_profile is not None
        and selection_profile != expected_selection_profile
    ):
        raise ValueError("Selection structured constraint profile differs")
    if (
        expected_final_profile is not None
        and expected_selection_profile != expected_final_profile
        and (expected_selection_profile, expected_final_profile)
        not in ALLOWED_SELECTION_FINAL_PROFILE_TRANSITIONS
    ):
        raise ValueError("Selection-to-final constraint profile transition is not allowed")
    expected_name = selection["selected"]["model_name"]
    if selection.get("model_manifest_sha256") != sha256_file(args.model_manifest):
        raise ValueError("Final model manifest differs from development selection")
    base = load_evaluation(
        args.base,
        expected_role="sealed_final",
        expected_n=EXPECTED_TEST_N,
        expected_constraint_profile=expected_final_profile,
    )
    candidate = load_evaluation(
        args.candidate,
        expected_role="sealed_final",
        expected_n=EXPECTED_TEST_N,
        expected_constraint_profile=expected_final_profile,
    )
    if (
        base["meta"].get("model_name") != "pi_base"
        or base["meta"].get("model_fingerprint") != "BASE"
    ):
        raise ValueError("Final base evaluation is not pi_base=BASE")
    if candidate["meta"].get("model_name") != expected_name:
        raise ValueError("Final candidate differs from development-selected model")
    if candidate["meta"].get("model_fingerprint") != selection["selected"].get(
        "model_fingerprint"
    ):
        raise ValueError("Final candidate fingerprint differs from selected adapter")
    if model_manifest["checkpoint_fingerprints"].get(
        str(selection["selected"]["step"])
    ) != candidate["meta"].get("model_fingerprint"):
        raise ValueError("Final candidate fingerprint differs from model manifest")
    final_profile = structured_constraint_profile(candidate["meta"])
    if selection_profile != final_profile:
        if (
            explicit_selection_profile is None
            or expected_final_profile is None
            or (selection_profile, final_profile)
            not in ALLOWED_SELECTION_FINAL_PROFILE_TRANSITIONS
        ):
            raise ValueError(
                "Observed selection-to-final constraint profile transition is not allowed"
            )
    if selection.get("selected", {}).get(
        "structured_constraint_profile", LEGACY_STRUCTURED_CONSTRAINT_PROFILE
    ) != selection_profile:
        raise ValueError("Selected checkpoint structured constraint profile differs")
    for field in ("base_model", "base_model_revision"):
        if candidate["meta"].get(field) != selection["selected"].get(field):
            raise ValueError(f"Final candidate differs from selection on {field}")
    value = comparison(base, candidate)
    checks, decision = gate(value)
    payload = sealed(
        {
            "schema_version": 1,
            "phase": "sealed_final",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selection_file": os.path.abspath(args.selection_file),
            "selection_sha256": sha256_file(args.selection_file),
            "selected_step": selection["selected"]["step"],
            "selected_model_name": expected_name,
            "structured_constraint_profile": final_profile,
            "selection_structured_constraint_profile": selection_profile,
            "final_structured_constraint_profile": final_profile,
            "xgrammar_any_whitespace": xgrammar_any_whitespace(candidate["meta"]),
            "base_file": os.path.abspath(args.base),
            "base_sha256": sha256_file(args.base),
            "candidate_file": os.path.abspath(args.candidate),
            "candidate_sha256": sha256_file(args.candidate),
            "comparison": value,
            "thresholds": frozen_thresholds(),
            "checks": checks,
            "decision": decision,
            "authorization": (
                "Benefit-only conclusion; no medical union, extra adapter, or quorum"
            ),
        }
    )
    markdown = format_comparison_markdown(
        "MASSIVE sealed-final benefit result", value, checks, decision
    )
    write_or_audit_summary(args.output_file, payload)
    write_or_audit_markdown(args.markdown_file, markdown)
    sentinel = (
        "GO_MASSIVE_BENEFIT_ONLY" if decision == "GO"
        else "STOPPED_MASSIVE_FINAL"
    )
    opposite = (
        "STOPPED_MASSIVE_FINAL" if decision == "GO"
        else "GO_MASSIVE_BENEFIT_ONLY"
    )
    write_or_restore_sentinel(
        args.sentinel_dir, sentinel, opposite, args.output_file, decision
    )
    print(decision)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("base")
    base.add_argument("--evaluation", required=True)
    base.add_argument("--output_file", required=True)
    base.add_argument("--markdown_file", required=True)
    base.add_argument("--sentinel_dir", required=True)
    base.add_argument(
        "--structured_constraint_profile",
        choices=SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES,
    )
    base.set_defaults(func=command_base)

    select = subparsers.add_parser("select")
    select.add_argument("--base", required=True)
    select.add_argument("--model_manifest", required=True)
    select.add_argument("--checkpoint", action="append", required=True)
    select.add_argument("--output_file", required=True)
    select.add_argument("--markdown_file", required=True)
    select.add_argument("--sentinel_dir", required=True)
    select.add_argument(
        "--structured_constraint_profile",
        choices=SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES,
    )
    select.set_defaults(func=command_select)

    final = subparsers.add_parser("final")
    final.add_argument("--selection_file", required=True)
    final.add_argument("--model_manifest", required=True)
    final.add_argument("--base", required=True)
    final.add_argument("--candidate", required=True)
    final.add_argument("--output_file", required=True)
    final.add_argument("--markdown_file", required=True)
    final.add_argument("--sentinel_dir", required=True)
    final.add_argument(
        "--structured_constraint_profile",
        choices=SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES,
    )
    final.add_argument(
        "--selection_structured_constraint_profile",
        choices=SUPPORTED_STRUCTURED_CONSTRAINT_PROFILES,
    )
    final.set_defaults(func=command_final)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
