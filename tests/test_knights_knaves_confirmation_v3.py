#!/usr/bin/env python3
"""No-network tests for the K&K v3 robustness amendment."""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import audit_knights_knaves_confirmation_v3_workflow as workflow  # noqa: E402
import evaluate_knights_knaves_confirmation_v3 as evaluate  # noqa: E402
import prepare_knights_knaves_confirmation_v3_data as prepare  # noqa: E402
import preflight_knights_knaves_confirmation_v3 as preflight  # noqa: E402
import summarize_knights_knaves_confirmation_v3 as summarize  # noqa: E402


PROMPTS_SHA256 = "1" * 64
ANSWERS_SHA256 = "2" * 64


def task(correct=False, stop_reason="stop"):
    return {
        "strict_correct": bool(correct),
        "official_correct": bool(correct),
        "strict_parseable": True,
        "strict_reason": "ok",
        "official_reason": "ok" if correct else "wrong_identity",
        "stop_reason": stop_reason,
    }


def comparison_row(delta=0.12, lower=0.05, p=0.01):
    return {
        "paired_accuracy_delta": delta,
        "paired_bootstrap_95ci": [lower, 0.2],
        "one_sided_exact_mcnemar_p": p,
    }


def evaluation_payload(set_name, model_name, correct, truncations=()):
    tasks = []
    for index, value in enumerate(correct):
        tasks.append(
            {
                "question_id": f"{set_name}:{index}",
                "logic_sha256": f"{index + 1000 * prepare.V3_SPECS[set_name]['n_people']:064x}",
                "strict_correct": bool(value),
                "strict_parseable": True,
                "strict_reason": "ok",
                "official_correct": bool(value),
                "official_reason": "ok" if value else "wrong_identity",
                "stop_reason": "max_new_tokens" if index in truncations else "stop",
            }
        )
    meta = {
        "schema_version": 1,
        "phase": "knights_knaves_confirmation_v3",
        "mode": "direct",
        "set_name": set_name,
        "role": "confirmation",
        "source_kind": "fresh",
        "source_id": "generator",
        "source_revision": "pinned",
        "generation_seed": prepare.V3_SPECS[set_name]["seed"],
        "n_people": prepare.V3_SPECS[set_name]["n_people"],
        "model_name": model_name,
        "model_fingerprint": (
            "BASE" if model_name == "pi_base" else evaluate.v2.CHECKPOINT_FINGERPRINT
        ),
        "base_model": evaluate.v2.BASE_MODEL,
        "base_model_revision": evaluate.v2.BASE_MODEL_REVISION,
        "prompt_file_sha256": PROMPTS_SHA256,
        "answers_file_sha256": ANSWERS_SHA256,
        "evaluator_script_sha256": "evaluator",
        "v2_evaluator_script_sha256": "v2-evaluator",
        "generator_script_sha256": "generator",
        "inference_seed": preflight.INFERENCE_SEED,
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": preflight.MAX_NEW_TOKENS,
        "max_context": preflight.MAX_CONTEXT,
    }
    payload = {"meta": meta, "metrics": summarize.derived_metrics(tasks), "tasks": tasks}
    payload["result_payload_sha256"] = summarize.common.sha256_bytes(
        summarize.common.canonical_json_bytes(payload)
    )
    return payload


def write_manifest(root):
    payload = {
        "schema_version": 1,
        "protocol": prepare.expected_protocol(),
        "sets": {
            set_name: {
                **spec,
                "role": "confirmation",
                "source_kind": "fresh",
                "prompts_sha256": PROMPTS_SHA256,
                "answers_sha256": ANSWERS_SHA256,
            }
            for set_name, spec in prepare.V3_SPECS.items()
        },
    }
    payload[summarize.MANIFEST_SEAL_FIELD] = summarize.common.sha256_bytes(
        summarize.common.canonical_json_bytes(payload)
    )
    path = os.path.join(root, prepare.MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


class ProtocolTests(unittest.TestCase):
    def test_frozen_sets_and_symmetric_longer_context(self):
        self.assertEqual(
            prepare.V3_SPECS,
            {
                "confirmation_v3_n4": {
                    "n_people": 4, "rows": 300, "seed": 2026081804,
                },
                "confirmation_v3_n5": {
                    "n_people": 5, "rows": 300, "seed": 2026081805,
                },
                "confirmation_v3_n6": {
                    "n_people": 6, "rows": 300, "seed": 2026081806,
                },
            },
        )
        self.assertEqual(preflight.MAX_NEW_TOKENS, 4096)
        self.assertEqual(preflight.MAX_CONTEXT, 8192)
        self.assertEqual(preflight.INFERENCE_SEED, 8172026)
        protocol = prepare.expected_protocol()
        self.assertFalse(protocol["checkpoint_selection_allowed"])
        self.assertFalse(protocol["training_allowed"])
        self.assertTrue(protocol["logic_disjoint_from_all_v1_and_v2_confirmation"])

    def test_length_stopped_correct_text_is_scored_normally(self):
        rows = [task(True, "max_new_tokens"), task(False, "stop")]
        diagnostics = evaluate.truncation_diagnostics(rows)
        self.assertEqual(diagnostics["truncated"], 1)
        self.assertEqual(diagnostics["truncated_strict_correct"], 1)
        self.assertEqual(diagnostics["truncated_official_correct"], 1)

    def test_base_favourable_sensitivity_credits_base_and_debits_candidate(self):
        base = [task(False, "max_new_tokens"), task(False, "stop")]
        candidate = [task(False, "stop"), task(True, "max_new_tokens")]
        observed = summarize.comparison(base, candidate, "strict")
        worst = summarize.comparison(
            base, candidate, "strict", base_favourable=True
        )
        self.assertEqual(observed["paired_accuracy_delta"], 0.5)
        self.assertEqual(worst["paired_accuracy_delta"], -0.5)
        self.assertEqual(worst["base_truncations_hypothetically_credited"], 1)
        self.assertEqual(worst["candidate_truncations_hypothetically_debited"], 1)

    def test_gate_requires_observed_and_worst_case_but_not_zero_truncation(self):
        endpoints = {}
        for endpoint in ("strict", "official"):
            endpoints[endpoint] = {}
            for scenario in ("observed", "base_favourable_worst_case"):
                endpoints[endpoint][scenario] = {
                    name: comparison_row() for name in ("n4", "n5", "n6", "n4_n6")
                }
        base_map = {}
        candidate_map = {}
        for set_name in summarize.SET_NAMES:
            base_map[set_name] = {
                "metrics": {"truncation_rate": 1 / 300}
            }
            candidate_map[set_name] = {
                "metrics": {"truncation_rate": 0.0}
            }
        checks = summarize.gate_checks(endpoints, base_map, candidate_map)
        self.assertTrue(all(checks.values()))
        self.assertFalse(any("zero_truncation" in key for key in checks))
        base_map[summarize.N4_SET]["metrics"]["truncation_rate"] = 4 / 300
        checks = summarize.gate_checks(endpoints, base_map, candidate_map)
        self.assertFalse(
            checks[f"{summarize.N4_SET}_base_truncation_rate"]
        )
        endpoints["strict"]["base_favourable_worst_case"]["n5"] = (
            comparison_row(delta=0.09)
        )
        checks = summarize.gate_checks(endpoints, base_map, candidate_map)
        self.assertFalse(checks["strict_n5_gain_worst_case"])

    def test_end_to_end_summary_allows_one_truncation_and_seals_go(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = write_manifest(root)
            base_args = []
            candidate_args = []
            for set_name in summarize.SET_NAMES:
                base_vector = [True] * 45 + [False] * 255
                candidate_vector = [True] * 85 + [False] * 215
                for model_name, vector, truncations in (
                    ("pi_base", base_vector, (100,) if set_name == summarize.N6_SET else ()),
                    ("step_192", candidate_vector, ()),
                ):
                    path = os.path.join(root, f"{set_name}-{model_name}.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(
                            evaluation_payload(set_name, model_name, vector, truncations),
                            handle,
                        )
                    if model_name == "pi_base":
                        base_args.append(f"{set_name}={path}")
                    else:
                        candidate_args.append(f"{set_name}={path}")
            v2_path = os.path.join(root, "v2-summary.json")
            with open(v2_path, "w", encoding="utf-8") as handle:
                json.dump({}, handle)
            args = types.SimpleNamespace(
                direct_base=base_args,
                direct_candidate=candidate_args,
                candidate_fingerprint=evaluate.v2.CHECKPOINT_FINGERPRINT,
                v3_data_manifest=manifest,
                v2_final_summary=v2_path,
                output_file=os.path.join(root, "summary.json"),
                markdown_file=os.path.join(root, "summary.md"),
                sentinel_dir=root,
                replicates=summarize.BOOTSTRAP_REPLICATES,
            )

            def fast_interval(left, right):
                delta = (sum(right) - sum(left)) / len(left)
                return [delta / 2, delta * 1.5]

            prior = {"gate": {"decision": "STOP"}}
            with mock.patch.object(summarize, "load_inherited_v2", return_value=prior), \
                    mock.patch.object(
                        summarize.v2_summary, "bootstrap_interval",
                        side_effect=fast_interval,
                    ):
                summarize.run_summary(args)
            with open(args.output_file, encoding="utf-8") as handle:
                result = json.load(handle)
            summarize.verify_decision(result)
            self.assertEqual(result["gate"]["decision"], "GO")
            self.assertFalse(result["diagnostics"]["zero_truncation_is_a_gate"])
            self.assertEqual(
                result["diagnostics"]["by_condition"][summarize.N6_SET]["base"]["truncated"],
                1,
            )
            self.assertTrue(os.path.isfile(os.path.join(root, "GO_KK_V3_BENEFIT_UNIONS")))


class WorkflowTests(unittest.TestCase):
    SHELL = (
        "scripts/stage_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
        "scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
        "scripts/status_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
        "scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch",
    )

    def test_shell_entrypoints_parse_and_cap_one_job(self):
        for path in self.SHELL:
            result = subprocess.run(
                ["bash", "-n", os.path.join(REPO_ROOT, path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/submit_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            submit = handle.read()
        self.assertEqual(submit.count("sbatch --parsable --hold"), 1)
        self.assertIn("cumulative_released_max_h200_minutes=210", submit)
        self.assertIn("selective_regeneration=false", submit)
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/sbatch_knights_knaves_reasoning_confirmation_v3_tillicum_h200.sbatch",
            ),
            encoding="utf-8",
        ) as handle:
            sbatch = handle.read()
        self.assertIn("#SBATCH --time=00:30:00", sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)
        self.assertEqual(
            sbatch.count("python scripts/sample_knights_knaves_generations.py"), 1
        )
        self.assertIn("--max_new_tokens 4096", sbatch)
        self.assertIn("--max_context 8192", sbatch)

    def test_stage_rejects_symlinked_v3_directories_before_mkdir(self):
        with open(
            os.path.join(
                REPO_ROOT,
                "scripts/stage_knights_knaves_reasoning_confirmation_v3_tillicum.sh",
            ),
            encoding="utf-8",
        ) as handle:
            stage = handle.read()
        self.assertGreaterEqual(stage.count('for directory in "$v3" "$control"'), 2)
        self.assertGreaterEqual(stage.count('test ! -L "$directory"'), 2)
        self.assertIn("Refusing unsafe v3 directory", stage)

    def test_cost_constants_stay_below_existing_ceiling(self):
        self.assertEqual(workflow.V3_MAX_MINUTES, 30)
        self.assertEqual(workflow.V1_V2_RELEASED_MAX_MINUTES, 180)
        self.assertEqual(workflow.CUMULATIVE_RELEASED_MAX_MINUTES, 210)
        self.assertLessEqual(
            workflow.CUMULATIVE_RELEASED_MAX_MINUTES,
            workflow.IMMUTABLE_CUMULATIVE_CEILING_MINUTES,
        )

    def test_v2_audit_encodes_exact_one_of_2400_correction(self):
        path = os.path.join(
            REPO_ROOT, "scripts/audit_knights_knaves_confirmation_v3_workflow.py"
        )
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("total_direct_rows != 2400", source)
        self.assertIn('("fresh_n6", "pi_base", "fresh_n6:113")', source)
        self.assertIn(workflow.V2_FINAL_SUMMARY_SHA256, source)
        self.assertIn(workflow.V2_STOP_SENTINEL_SHA256, source)


if __name__ == "__main__":
    unittest.main()
