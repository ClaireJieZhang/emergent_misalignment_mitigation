#!/usr/bin/env python3
"""No-network tests for the MASSIVE test-only evaluation recovery v2."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, os.fspath(SCRIPTS))

import audit_massive_benefit_evaluation_recovery_v2 as recovery  # noqa: E402
import evaluate_massive_benefit_generations as evaluator  # noqa: E402
import sample_massive_structured_generations as sampler  # noqa: E402
import summarize_massive_benefit_pilot as summarizer  # noqa: E402


def read_repo(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
        handle.write("\n")


class RecoveryV2AuditTests(unittest.TestCase):
    def test_budget_uses_actual_failed_job_and_stays_inside_original_ceiling(self):
        self.assertEqual(recovery.PRIOR_ROUNDED_H200_MINUTES, 157)
        self.assertEqual(recovery.EVALUATION_MAX_H200_MINUTES, 15)
        self.assertEqual(recovery.CUMULATIVE_MAX_H200_MINUTES, 172)
        self.assertEqual(recovery.CONTINGENCY_MAX_H200_MINUTES, 173)
        self.assertLess(173, recovery.ORIGINAL_MAX_H200_MINUTES)
        self.assertEqual(172 * 0.90 / 60, 2.58)

    def test_failure_and_selection_hashes_are_hard_bound(self):
        self.assertEqual(
            recovery.V1_DECISION_HASHES["selection/summary.json"],
            "11560cbea42049bdf40dcf4db9bfc0e5ffc9bea6084f41de7c0a9a9981c0cdfd",
        )
        self.assertEqual(
            recovery.V1_PARTIAL_TEST_HASHES[
                "generations/failures/massive_en_test__pi_base__intent_only.failure.json"
            ],
            "23d1ccc633da88b83537d112bfab6db4ac7992699eda37ece66500355db1c197",
        )
        self.assertEqual(len(recovery.V1_GENERATION_HASHES), 12)
        self.assertEqual(len(recovery.V1_SCORE_HASHES), 6)
        self.assertEqual(recovery.V1_SELECTION_STEP, 30)

    def test_job_record_is_exactly_one_fifteen_minute_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "jobs.tsv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("stage\tjob_id\tmax_minutes\nevaluate\t12345\t15\n")
            self.assertEqual(
                recovery.parse_recovery_jobs(path),
                [{"stage": "evaluate", "job_id": "12345", "max_minutes": 15}],
            )
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("evaluate\t12346\t15\n")
            with self.assertRaisesRegex(ValueError, "one evaluation"):
                recovery.parse_recovery_jobs(path)

    def test_const_tree_schema_is_identical_across_v2_and_v3(self):
        intents = ["alpha", "beta", "gamma"]
        slots = ["name", "place"]
        for endpoint in ("joint_json", "intent_only"):
            v2 = sampler.prediction_schema(
                intents, slots, endpoint=endpoint,
                structured_constraint_profile="const_tree_v2",
            )
            v3 = sampler.prediction_schema(
                intents, slots, endpoint=endpoint,
                structured_constraint_profile="const_tree_no_ws_v3",
            )
            self.assertEqual(v2, v3)

    def test_whitespace_provenance_is_fail_closed_and_legacy_compatible(self):
        self.assertTrue(
            evaluator.xgrammar_any_whitespace(
                {"structured_constraint_profile": "const_tree_v2"}
            )
        )
        self.assertFalse(
            evaluator.xgrammar_any_whitespace(
                {
                    "structured_constraint_profile": "const_tree_no_ws_v3",
                    "xgrammar_any_whitespace": False,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "compiler-policy"):
            evaluator.xgrammar_any_whitespace(
                {"structured_constraint_profile": "const_tree_no_ws_v3"}
            )
        with self.assertRaisesRegex(ValueError, "compiler-policy"):
            summarizer.xgrammar_any_whitespace(
                {
                    "structured_constraint_profile": "const_tree_no_ws_v3",
                    "xgrammar_any_whitespace": True,
                }
            )

    def test_scientific_ast_freeze_covers_metrics_and_gates(self):
        evidence = recovery.audit_scientific_contract(REPO_ROOT)
        self.assertFalse(evidence["metrics_gates_selection_rule_changed"])
        self.assertIn("evaluate", evidence["evaluator_frozen_function_asts"])
        self.assertIn("gate", evidence["summarizer_frozen_function_asts"])
        self.assertEqual(
            evidence["allowed_profile_transition"],
            ["const_tree_v2", "const_tree_no_ws_v3"],
        )


class FinalProfileTransitionTests(unittest.TestCase):
    def fixture(self, root, selection_profile="const_tree_v2"):
        manifest_path = os.path.join(root, "manifest.json")
        manifest = {
            "checkpoint_fingerprints": {
                "15": "a", "30": "selected-fp", "60": "c",
                "90": "d", "150": "e",
            }
        }
        manifest["manifest_payload_sha256"] = summarizer.sha256_bytes(
            summarizer.canonical_json_bytes(manifest)
        )
        write_json(manifest_path, manifest)
        selection_path = os.path.join(root, "selection.json")
        selection = summarizer.sealed(
            {
                "schema_version": 1,
                "phase": "development_selection",
                "created_at": "frozen",
                "decision": "GO",
                "structured_constraint_profile": selection_profile,
                "model_manifest_sha256": summarizer.sha256_file(manifest_path),
                "selected": {
                    "step": 30,
                    "model_name": "step_30",
                    "model_fingerprint": "selected-fp",
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "base_model_revision": "b" * 40,
                    "structured_constraint_profile": selection_profile,
                },
            }
        )
        write_json(selection_path, selection)
        return manifest_path, selection_path

    @staticmethod
    def evaluation(model_name, fingerprint, profile):
        return {
            "meta": {
                "model_name": model_name,
                "model_fingerprint": fingerprint,
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "base_model_revision": "b" * 40,
                "structured_constraint_profile": profile,
                "xgrammar_any_whitespace": profile != "const_tree_no_ws_v3",
            },
            "metrics": {},
            "tasks": [],
        }

    def args(self, root, manifest, selection, final_profile, selection_flag):
        return argparse.Namespace(
            selection_file=selection,
            model_manifest=manifest,
            base=os.path.join(root, "base.json"),
            candidate=os.path.join(root, "candidate.json"),
            output_file=os.path.join(root, "summary.json"),
            markdown_file=os.path.join(root, "summary.md"),
            sentinel_dir=os.path.join(root, "control"),
            structured_constraint_profile=final_profile,
            selection_structured_constraint_profile=selection_flag,
        )

    def test_transition_requires_both_explicit_flags(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, selection = self.fixture(root)
            base = self.evaluation("pi_base", "BASE", "const_tree_no_ws_v3")
            candidate = self.evaluation(
                "step_30", "selected-fp", "const_tree_no_ws_v3"
            )
            with mock.patch.object(
                summarizer, "load_evaluation", side_effect=[base, candidate]
            ):
                with self.assertRaisesRegex(ValueError, "Selection structured"):
                    summarizer.command_final(
                        self.args(
                            root, manifest, selection,
                            "const_tree_no_ws_v3", None,
                        )
                    )
            with self.assertRaisesRegex(ValueError, "requires an explicit final"):
                summarizer.command_final(
                    self.args(root, manifest, selection, None, "const_tree_v2")
                )

    def test_omitted_flags_preserve_observed_profile_equality(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, selection = self.fixture(root)
            base = self.evaluation("pi_base", "BASE", "const_tree_no_ws_v3")
            candidate = self.evaluation(
                "step_30", "selected-fp", "const_tree_no_ws_v3"
            )
            with mock.patch.object(
                summarizer, "load_evaluation", side_effect=[base, candidate]
            ):
                with self.assertRaisesRegex(ValueError, "Observed selection-to-final"):
                    summarizer.command_final(
                        self.args(root, manifest, selection, None, None)
                    )

    def test_exact_explicit_v2_to_v3_transition_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            manifest, selection = self.fixture(root)
            write_json(os.path.join(root, "base.json"), {"fixture": "base"})
            write_json(os.path.join(root, "candidate.json"), {"fixture": "candidate"})
            base = self.evaluation("pi_base", "BASE", "const_tree_no_ws_v3")
            candidate = self.evaluation(
                "step_30", "selected-fp", "const_tree_no_ws_v3"
            )
            comparison = {
                "n": 2965,
                "base_joint_intent_accuracy": 0.60,
                "candidate_joint_intent_accuracy": 0.85,
                "paired_joint_intent_delta": 0.25,
                "joint_intent_paired_bootstrap_95ci": [0.20, 0.30],
                "joint_intent_one_sided_exact_mcnemar_p": 0.0,
                "base_slot_pair_micro_f1": 0.3,
                "candidate_slot_pair_micro_f1": 0.7,
                "slot_pair_micro_f1_delta": 0.4,
                "base_strict_frame_exact_accuracy": 0.2,
                "candidate_strict_frame_exact_accuracy": 0.6,
                "strict_frame_exact_delta": 0.4,
                "base_controlled_intent_accuracy": 0.6,
                "candidate_controlled_intent_accuracy": 0.8,
                "paired_controlled_intent_delta": 0.2,
            }
            with mock.patch.object(
                summarizer, "load_evaluation", side_effect=[base, candidate]
            ), mock.patch.object(
                summarizer, "comparison", return_value=comparison
            ):
                summarizer.command_final(
                    self.args(
                        root, manifest, selection,
                        "const_tree_no_ws_v3", "const_tree_v2",
                    )
                )
            with open(os.path.join(root, "summary.json"), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["selection_structured_constraint_profile"], "const_tree_v2")
            self.assertEqual(payload["final_structured_constraint_profile"], "const_tree_no_ws_v3")
            self.assertIs(payload["xgrammar_any_whitespace"], False)


class RecoveryV2ShellTests(unittest.TestCase):
    SHELL_FILES = (
        "scripts/stage_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        "scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        "scripts/status_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        "scripts/sbatch_massive_benefit_evaluation_recovery_v2_tillicum_h200.sbatch",
    )

    def test_shell_entrypoints_parse(self):
        for relative in self.SHELL_FILES:
            subprocess.run(["bash", "-n", REPO_ROOT / relative], check=True)

    def test_cost_strings_are_not_positional_parameter_expansions(self):
        for relative in self.SHELL_FILES:
            self.assertNotRegex(
                read_repo(relative),
                r'echo "[^"\n]*\$[0-9]',
                msg=relative,
            )

    def test_job_is_test_only_and_reuses_exact_selection(self):
        script = read_repo(
            "scripts/sbatch_massive_benefit_evaluation_recovery_v2_tillicum_h200.sbatch"
        )
        self.assertIn("#SBATCH --time=00:15:00", script)
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertIn("test \"$selected_step:$selected_name\" = 30:step_30", script)
        self.assertIn("--prompt_file \"$DATA_ROOT/sealed_test/prompts.json\"", script)
        self.assertNotIn("dev/prompts.json", script)
        self.assertNotIn("train_single_sft", script)
        self.assertIn("--selection_structured_constraint_profile const_tree_v2", script)
        self.assertIn("--structured_constraint_profile \"$PROFILE\"", script)

    def test_submission_is_held_first_exact_once(self):
        script = read_repo(
            "scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh"
        )
        self.assertEqual(script.count("sbatch --parsable --hold"), 1)
        self.assertIn("MASSIVE_EVALUATION_RECOVERY_V2_SUBMISSION_LOCK", script)
        self.assertIn("scontrol release \"$evaluate_job\"", script)
        self.assertIn("--time=00:15:00", script)
        self.assertNotIn("--dependency", script)

    def test_stage_and_status_never_submit_or_release(self):
        for relative in (
            "scripts/stage_massive_benefit_evaluation_recovery_v2_tillicum.sh",
            "scripts/status_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        ):
            script = read_repo(relative)
            self.assertNotIn("sbatch --parsable", script)
            self.assertNotIn("scontrol release", script)


if __name__ == "__main__":
    unittest.main()
