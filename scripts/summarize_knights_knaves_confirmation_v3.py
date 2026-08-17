#!/usr/bin/env python3
"""Apply preregistered K&K v3 robustness gates.

V3 repeats only the direct endpoint because it is a focused truncation
robustness amendment.  V2's structured endpoint is recorded as prior support,
not inherited as a V3 gate.  Every substantive V3 gate must pass both ordinary
scoring and a base-favourable truncation sensitivity that credits every
length-stopped base response as correct while giving the candidate no such
credit.
"""

import argparse
import collections
import datetime
import json
import os
import re

import evaluate_knights_knaves_confirmation_v3 as evaluator
import prepare_knights_knaves_confirmation_v3_data as data
import sample_knights_knaves_generations as common
import summarize_knights_knaves_confirmation_v2 as v2_summary
from summarize_knights_knaves_pilot import one_sided_exact_mcnemar_p


SET_NAMES = tuple(sorted(data.V3_SPECS, key=lambda name: data.V3_SPECS[name]["n_people"]))
N4_SET = "confirmation_v3_n4"
N5_SET = "confirmation_v3_n5"
N6_SET = "confirmation_v3_n6"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = v2_summary.BOOTSTRAP_SEED
V2_FINAL_SUMMARY_SHA256 = (
    "389b3b522ab101a6da9276667c41e85a5d5019a8ae77c99f17260353eb197cb6"
)
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"
MAX_TRUNCATION_RATE_PER_CONDITION = 0.01


def require_sha256(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA-256 for {label}")
    return value


def verify_result_seal(payload, label):
    unsealed = dict(payload)
    observed = unsealed.pop("result_payload_sha256", None)
    expected = common.sha256_bytes(common.canonical_json_bytes(unsealed))
    if observed != expected:
        raise ValueError(f"{label} payload seal mismatch")


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
        raise ValueError("K&K v3 decision payload seal mismatch")


def load_manifest(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path) or os.path.islink(path):
        raise ValueError(f"Missing or unsafe V3 data manifest: {path}")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    unsealed = dict(manifest)
    observed = unsealed.pop(MANIFEST_SEAL_FIELD, None)
    expected = common.sha256_bytes(common.canonical_json_bytes(unsealed))
    if observed != expected:
        raise ValueError("V3 data-manifest payload seal mismatch")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unexpected V3 data-manifest schema")
    if manifest.get("protocol") != data.expected_protocol():
        raise ValueError("V3 data manifest differs from frozen protocol")
    if set(manifest.get("sets", {})) != set(SET_NAMES):
        raise ValueError("V3 data manifest has unexpected sets")
    return {
        "payload": manifest,
        "path": path,
        "file_sha256": common.sha256_file(path),
        "payload_sha256": require_sha256(observed, MANIFEST_SEAL_FIELD),
    }


def manifest_binding(manifest_record, set_name):
    if set_name not in SET_NAMES:
        raise ValueError(f"Set is outside V3: {set_name}")
    entry = manifest_record["payload"]["sets"][set_name]
    spec = data.V3_SPECS[set_name]
    for field, expected in {
        **spec, "role": "confirmation", "source_kind": "fresh"
    }.items():
        if entry.get(field) != expected:
            raise ValueError(f"Manifest {set_name} differs for {field}")
    return {
        **spec,
        "role": "confirmation",
        "source_kind": "fresh",
        "prompt_file_sha256": require_sha256(
            entry.get("prompts_sha256"), f"{set_name} prompts"
        ),
        "answers_file_sha256": require_sha256(
            entry.get("answers_sha256"), f"{set_name} answers"
        ),
    }


def derived_metrics(tasks):
    n = len(tasks)
    truncated = [task for task in tasks if task["stop_reason"] == "max_new_tokens"]
    return {
        "n": n,
        "strict_correct": sum(task["strict_correct"] for task in tasks),
        "strict_accuracy": sum(task["strict_correct"] for task in tasks) / n,
        "strict_parseable": sum(task["strict_parseable"] for task in tasks),
        "strict_parse_coverage": sum(task["strict_parseable"] for task in tasks) / n,
        "strict_reasons": dict(sorted(collections.Counter(
            task["strict_reason"] for task in tasks
        ).items())),
        "official_correct": sum(task["official_correct"] for task in tasks),
        "official_accuracy": sum(task["official_correct"] for task in tasks) / n,
        "official_reasons": dict(sorted(collections.Counter(
            task["official_reason"] for task in tasks
        ).items())),
        "truncated": len(truncated),
        "truncation_rate": len(truncated) / n,
        "truncated_strict_correct": sum(task["strict_correct"] for task in truncated),
        "truncated_official_correct": sum(task["official_correct"] for task in truncated),
    }


def load_evaluation(path, expected_set=None):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation is not an object: {path}")
    verify_result_seal(payload, path)
    meta = payload.get("meta")
    tasks = payload.get("tasks")
    metrics = payload.get("metrics")
    if not isinstance(meta, dict) or not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Evaluation lacks nonempty meta/tasks: {path}")
    if metrics != derived_metrics(tasks):
        raise ValueError(f"V3 metrics are not exactly task-derived: {path}")
    if meta.get("phase") != "knights_knaves_confirmation_v3":
        raise ValueError(f"Evaluation is not V3: {path}")
    if meta.get("mode") != "direct":
        raise ValueError(f"Evaluation is not direct: {path}")
    if expected_set and meta.get("set_name") != expected_set:
        raise ValueError(f"Expected set {expected_set}: {path}")
    if meta.get("set_name") not in SET_NAMES:
        raise ValueError(f"Unexpected V3 set: {path}")
    if (
        meta.get("base_model") != evaluator.v2.BASE_MODEL
        or meta.get("base_model_revision") != evaluator.v2.BASE_MODEL_REVISION
        or meta.get("inference_seed") != evaluator.protocol.INFERENCE_SEED
        or meta.get("temperature") != 0.0
        or meta.get("n_samples") != 1
        or meta.get("max_new_tokens") != evaluator.protocol.MAX_NEW_TOKENS
        or meta.get("max_context") != evaluator.protocol.MAX_CONTEXT
    ):
        raise ValueError(f"Evaluation inference settings differ from V3: {path}")
    if meta.get("model_name") == "pi_base":
        if meta.get("model_fingerprint") != "BASE":
            raise ValueError("V3 base fingerprint is not BASE")
    elif meta.get("model_name") == "step_192":
        if meta.get("model_fingerprint") != evaluator.v2.CHECKPOINT_FINGERPRINT:
            raise ValueError("V3 candidate fingerprint differs from frozen step 192")
    else:
        raise ValueError(f"V3 evaluation has an unexpected model: {path}")
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
        for field in ("strict_correct", "strict_parseable", "official_correct"):
            if type(task.get(field)) is not bool:
                raise ValueError(f"Invalid V3 task field {field}: {path}")
        if not isinstance(task.get("strict_reason"), str) or not isinstance(
            task.get("official_reason"), str
        ):
            raise ValueError(f"Invalid V3 task reason: {path}")
    payload["_path"] = path
    return payload


def bind_evaluation(evaluation, binding):
    meta = evaluation["meta"]
    expected = {
        "role": binding["role"],
        "source_kind": binding["source_kind"],
        "n_people": binding["n_people"],
        "generation_seed": binding["seed"],
        "prompt_file_sha256": binding["prompt_file_sha256"],
        "answers_file_sha256": binding["answers_file_sha256"],
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"Evaluation differs from manifest for {field}")
    if evaluation["metrics"]["n"] != binding["rows"]:
        raise ValueError("Evaluation count differs from manifest")


def pair(base, candidate, expected_set):
    if base["meta"].get("set_name") != expected_set:
        raise ValueError(f"Expected set {expected_set}")
    for field in (
        "set_name", "role", "source_kind", "source_id", "source_revision",
        "generation_seed", "n_people", "base_model", "base_model_revision",
        "prompt_file_sha256", "answers_file_sha256", "inference_seed",
        "temperature", "n_samples", "max_new_tokens", "max_context",
        "evaluator_script_sha256", "v2_evaluator_script_sha256",
        "generator_script_sha256",
    ):
        if base["meta"].get(field) != candidate["meta"].get(field):
            raise ValueError(f"Paired evaluations disagree on {field}")
    if base["meta"]["model_name"] != "pi_base":
        raise ValueError("Left condition is not the base")
    if candidate["meta"]["model_name"] != "step_192":
        raise ValueError("Right condition is not frozen step 192")
    base_keys = [(task["question_id"], task["logic_sha256"]) for task in base["tasks"]]
    candidate_keys = [
        (task["question_id"], task["logic_sha256"])
        for task in candidate["tasks"]
    ]
    if base_keys != candidate_keys:
        raise ValueError("Paired V3 evaluations have different tasks or order")


def endpoint_field(endpoint):
    return {"strict": "strict_correct", "official": "official_correct"}[endpoint]


def comparison(base_tasks, candidate_tasks, endpoint, base_favourable=False):
    field = endpoint_field(endpoint)
    left = []
    right = []
    credited = 0
    debited = 0
    for base, candidate in zip(base_tasks, candidate_tasks):
        base_value = bool(base[field])
        if (
            base_favourable
            and base["stop_reason"] == "max_new_tokens"
            and not base_value
        ):
            base_value = True
            credited += 1
        candidate_value = bool(candidate[field])
        if (
            base_favourable
            and candidate["stop_reason"] == "max_new_tokens"
            and candidate_value
        ):
            candidate_value = False
            debited += 1
        left.append(base_value)
        right.append(candidate_value)
    n = len(left)
    return {
        "n": n,
        "base_correct": sum(left),
        "candidate_correct": sum(right),
        "base_accuracy": sum(left) / n,
        "candidate_accuracy": sum(right) / n,
        "paired_accuracy_delta": (sum(right) - sum(left)) / n,
        "candidate_only_correct": sum((not a) and b for a, b in zip(left, right)),
        "base_only_correct": sum(a and (not b) for a, b in zip(left, right)),
        "one_sided_exact_mcnemar_p": one_sided_exact_mcnemar_p(left, right),
        "paired_bootstrap_95ci": v2_summary.bootstrap_interval(left, right),
        "base_truncations_hypothetically_credited": credited,
        "candidate_truncations_hypothetically_debited": debited,
    }


def pooled(base_map, candidate_map, set_names, endpoint, base_favourable=False):
    base_tasks = []
    candidate_tasks = []
    for set_name in set_names:
        pair(base_map[set_name], candidate_map[set_name], set_name)
        base_tasks.extend(base_map[set_name]["tasks"])
        candidate_tasks.extend(candidate_map[set_name]["tasks"])
    return comparison(
        base_tasks, candidate_tasks, endpoint, base_favourable=base_favourable
    )


def load_inherited_v2(path):
    if common.sha256_file(path) != V2_FINAL_SUMMARY_SHA256:
        raise ValueError("Inherited V2 final summary hash changed")
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    v2_summary.verify_decision(result)
    checks = result.get("gate", {}).get("checks", {})
    failed = sorted(key for key, value in checks.items() if not value)
    if result.get("gate", {}).get("decision") != "STOP" or failed != [
        "all_direct_zero_truncation"
    ]:
        raise ValueError("V2 did not stop solely on its zero-direct-truncation rule")
    return result


def gate_checks(results, base_map, candidate_map):
    checks = {}
    for scenario in ("observed", "base_favourable_worst_case"):
        suffix = "observed" if scenario == "observed" else "worst_case"
        strict = results["strict"][scenario]
        checks[f"strict_n5_gain_{suffix}"] = strict["n5"]["paired_accuracy_delta"] >= 0.10
        checks[f"strict_n5_ci_lower_{suffix}"] = strict["n5"]["paired_bootstrap_95ci"][0] > 0
        checks[f"strict_n5_mcnemar_{suffix}"] = strict["n5"]["one_sided_exact_mcnemar_p"] < 0.05
        checks[f"strict_n4_n6_nonnegative_{suffix}"] = strict["n4_n6"]["paired_accuracy_delta"] >= 0
        checks[f"strict_n4_noninferior_{suffix}"] = strict["n4"]["paired_accuracy_delta"] >= -0.02
        checks[f"strict_n6_noninferior_{suffix}"] = strict["n6"]["paired_accuracy_delta"] >= -0.02
        official = results["official"][scenario]["n5"]
        checks[f"official_n5_gain_{suffix}"] = official["paired_accuracy_delta"] >= 0.10
        checks[f"official_n5_ci_lower_{suffix}"] = official["paired_bootstrap_95ci"][0] > 0
        checks[f"official_n5_mcnemar_{suffix}"] = official["one_sided_exact_mcnemar_p"] < 0.05
    for set_name in SET_NAMES:
        for label, mapping in (("base", base_map), ("candidate", candidate_map)):
            checks[f"{set_name}_{label}_truncation_rate"] = (
                mapping[set_name]["metrics"]["truncation_rate"]
                <= MAX_TRUNCATION_RATE_PER_CONDITION
            )
    return checks


def write_sentinel(directory, decision, summary_path):
    os.makedirs(directory, exist_ok=True)
    go_name = "GO_KK_V3_BENEFIT_UNIONS"
    stop_name = "STOPPED_KK_V3_FINAL"
    chosen = go_name if decision == "GO" else stop_name
    opposite = stop_name if decision == "GO" else go_name
    if os.path.lexists(os.path.join(directory, opposite)):
        raise ValueError("Conflicting V3 decision sentinel exists")
    payload = {
        "decision": decision,
        "summary_file": os.path.abspath(summary_path),
        "summary_sha256": common.sha256_file(summary_path),
    }
    path = os.path.join(directory, chosen)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise ValueError("Existing V3 decision sentinel conflicts")
    else:
        common.atomic_write_json(path, payload)


def write_markdown(path, result):
    lines = [
        "# K&K v3 symmetric longer-cap confirmation",
        "",
        f"Decision: **{result['gate']['decision']}**",
        "",
        "V2 remains an immutable STOP. V3 reruns both direct conditions on new "
        "logic-disjoint data; it does not selectively regenerate the single V2 "
        "truncation.",
        "",
        "| endpoint/scenario | split | base | checkpoint 192 | delta | McNemar p | 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for endpoint in ("strict", "official"):
        for scenario in ("observed", "base_favourable_worst_case"):
            for split in ("n4", "n5", "n6", "n4_n6"):
                row = result["endpoints"][endpoint][scenario][split]
                lines.append(
                    f"| `{endpoint}/{scenario}` | `{split}` | "
                    f"{row['base_accuracy']:.3f} | {row['candidate_accuracy']:.3f} | "
                    f"{row['paired_accuracy_delta']:+.3f} | "
                    f"{row['one_sided_exact_mcnemar_p']:.4g} | "
                    f"[{row['paired_bootstrap_95ci'][0]:+.3f}, "
                    f"{row['paired_bootstrap_95ci'][1]:+.3f}] |"
                )
    lines.extend([
        "",
        "| set/model | truncated | rate | parse coverage |",
        "| --- | ---: | ---: | ---: |",
    ])
    for set_name, conditions in result["diagnostics"]["by_condition"].items():
        for label, row in conditions.items():
            lines.append(
                f"| `{set_name}/{label}` | {row['truncated']} | "
                f"{row['truncation_rate']:.4f} | {row['strict_parse_coverage']:.3f} |"
            )
    lines.extend(["", result["gate"]["reason"], ""])
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + f".tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def run_summary(args):
    if args.candidate_fingerprint != evaluator.v2.CHECKPOINT_FINGERPRINT:
        raise ValueError("CLI candidate fingerprint is not frozen checkpoint 192")
    if args.replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("V3 bootstrap replicates are frozen at 10,000")
    manifest = load_manifest(args.v3_data_manifest)
    inherited_v2 = load_inherited_v2(args.v2_final_summary)
    base_map = parse_map(args.direct_base, "pi_base")
    candidate_map = parse_map(args.direct_candidate, "step_192")
    for set_name in SET_NAMES:
        binding = manifest_binding(manifest, set_name)
        bind_evaluation(base_map[set_name], binding)
        bind_evaluation(candidate_map[set_name], binding)
        pair(base_map[set_name], candidate_map[set_name], set_name)

    endpoint_results = {}
    split_map = {
        "n4": (N4_SET,), "n5": (N5_SET,), "n6": (N6_SET,),
        "n4_n6": (N4_SET, N6_SET),
    }
    for endpoint in ("strict", "official"):
        endpoint_results[endpoint] = {}
        for scenario, sensitive in (
            ("observed", False), ("base_favourable_worst_case", True)
        ):
            endpoint_results[endpoint][scenario] = {
                name: pooled(
                    base_map, candidate_map, sets, endpoint,
                    base_favourable=sensitive,
                )
                for name, sets in split_map.items()
            }
    checks = gate_checks(endpoint_results, base_map, candidate_map)
    decision = "GO" if all(checks.values()) else "STOP"
    failed = [key for key, value in checks.items() if not value]
    reason = (
        "All V3 observed, truncation-robustness, significance, transfer, and "
        "per-condition truncation-rate gates passed. The frozen K&K benefit is "
        "eligible for later matched-union construction; no union was created."
        if decision == "GO"
        else "K&K v3 gate failed: " + ", ".join(failed) + "."
    )
    diagnostics = {
        "by_condition": {
            set_name: {
                label: {
                    "n": mapping[set_name]["metrics"]["n"],
                    "truncated": mapping[set_name]["metrics"]["truncated"],
                    "truncation_rate": mapping[set_name]["metrics"]["truncation_rate"],
                    "truncated_strict_correct": mapping[set_name]["metrics"]["truncated_strict_correct"],
                    "truncated_official_correct": mapping[set_name]["metrics"]["truncated_official_correct"],
                    "strict_parse_coverage": mapping[set_name]["metrics"]["strict_parse_coverage"],
                }
                for label, mapping in (("base", base_map), ("candidate", candidate_map))
            }
            for set_name in SET_NAMES
        },
        "scoring_policy": (
            "Length-stopped text is scored normally. The sensitivity credits every "
            "length-stopped base response as correct and forces every length-stopped "
            "candidate response incorrect."
        ),
        "zero_truncation_is_a_gate": False,
    }
    result = seal_decision(
        {
            "meta": {
                "schema_version": 1,
                "phase": "confirmation_v3",
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "frozen_checkpoint_step": 192,
                "candidate_fingerprint": evaluator.v2.CHECKPOINT_FINGERPRINT,
                "base_model": evaluator.v2.BASE_MODEL,
                "base_model_revision": evaluator.v2.BASE_MODEL_REVISION,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "v3_data_manifest_sha256": manifest["file_sha256"],
                "v3_data_manifest_payload_sha256": manifest["payload_sha256"],
                "v2_final_summary_sha256": common.sha256_file(args.v2_final_summary),
                "v2_decision_preserved": inherited_v2["gate"]["decision"],
                "v2_controlled_endpoint_prior_support_only": True,
            },
            "inputs": {
                set_name: {
                    "direct_base_sha256": common.sha256_file(base_map[set_name]["_path"]),
                    "direct_candidate_sha256": common.sha256_file(candidate_map[set_name]["_path"]),
                }
                for set_name in SET_NAMES
            },
            "thresholds": {
                "n5_minimum_paired_gain": 0.10,
                "n5_bootstrap_lower_above": 0.0,
                "n5_one_sided_mcnemar_p_below": 0.05,
                "strict_n4_n6_minimum_delta": 0.0,
                "strict_each_transfer_minimum_delta": -0.02,
                "maximum_truncation_rate_each_set_model": MAX_TRUNCATION_RATE_PER_CONDITION,
                "zero_truncation_required": False,
                "all_substantive_gates_repeat_under_base_favourable_sensitivity": True,
            },
            "endpoints": endpoint_results,
            "diagnostics": diagnostics,
            "gate": {"decision": decision, "checks": checks, "reason": reason},
        }
    )
    common.atomic_write_json(args.output_file, result)
    if args.markdown_file:
        write_markdown(args.markdown_file, result)
    write_sentinel(args.sentinel_dir, decision, args.output_file)
    print(f"K&K v3 decision: {decision}")


def parse_map(specs, expected_model):
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected SET=PATH: {spec}")
        set_name, path = spec.split("=", 1)
        if set_name in result:
            raise ValueError(f"Duplicate set: {set_name}")
        evaluation = load_evaluation(path, set_name)
        if evaluation["meta"].get("model_name") != expected_model:
            raise ValueError(f"Unexpected model for {set_name}")
        result[set_name] = evaluation
    if set(result) != set(SET_NAMES):
        raise ValueError("V3 inputs must contain exactly N4, N5, and N6")
    return result


def run_audit(args):
    with open(args.summary_file, encoding="utf-8") as handle:
        summary = json.load(handle)
    verify_decision(summary)
    decision = summary.get("gate", {}).get("decision")
    if decision not in {"GO", "STOP"}:
        raise ValueError("V3 summary has no terminal decision")
    write_sentinel(args.sentinel_dir, decision, args.summary_file)
    if args.markdown_file and not os.path.isfile(args.markdown_file):
        write_markdown(args.markdown_file, summary)
    print(decision)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--direct_base", action="append", required=True)
    summary.add_argument("--direct_candidate", action="append", required=True)
    summary.add_argument("--candidate_fingerprint", required=True)
    summary.add_argument("--v3_data_manifest", required=True)
    summary.add_argument("--v2_final_summary", required=True)
    summary.add_argument("--output_file", required=True)
    summary.add_argument("--markdown_file")
    summary.add_argument("--sentinel_dir", required=True)
    summary.add_argument(
        "--replicates", type=int, choices=(BOOTSTRAP_REPLICATES,),
        default=BOOTSTRAP_REPLICATES,
    )
    summary.set_defaults(function=run_summary)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--summary_file", required=True)
    audit.add_argument("--sentinel_dir", required=True)
    audit.add_argument("--markdown_file")
    audit.set_defaults(function=run_audit)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
