#!/usr/bin/env python3
"""No-network tests for the capped repaired-pilot workflow."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import verify_general_code_apps_repaired_authorization as authorization  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs/training_qwen25_7b_apps_repaired_pilot.yaml"
SELECTOR = REPO_ROOT / "scripts/select_repaired_code_pilot_checkpoint.py"
MODELS = ("pi_base", "step_10", "step_20", "step_30", "step_40")


def apps_summary(step_passes, quality=None):
    quality = quality or {}
    models = {}
    for name in MODELS:
        empty, truncated = quality.get(name, (0, 0))
        models[name] = {
            "n": 200,
            "passed": step_passes[name],
            "empty_extractions": empty,
            "truncations": truncated,
        }
    return {"meta": {"n_questions": 200}, "models": models}


def write_selector_inputs(root, summary):
    source = root / "apps-summary.json"
    model_manifest = root / "model-manifest.json"
    output = root / "selection.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    model_manifest.write_text(
        json.dumps(
            {
                "checkpoints": {
                    name: {
                        "files": [
                            {
                                "path": f"checkpoint-{name.split('_')[1]}/adapter_config.json",
                                "sha256": "a" * 64,
                            },
                            {
                                "path": f"checkpoint-{name.split('_')[1]}/adapter_model.safetensors",
                                "sha256": name.encode().hex().ljust(64, "0")[:64],
                            },
                        ]
                    }
                    for name in MODELS[1:]
                }
            }
        ),
        encoding="utf-8",
    )
    return source, model_manifest, output


def run_selector(root, summary):
    source, model_manifest, output = write_selector_inputs(root, summary)
    subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--apps-summary",
            str(source),
            "--model-manifest",
            str(model_manifest),
            "--output-file",
            str(output),
            "--expected-problems",
            "200",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return source, model_manifest, output, json.loads(output.read_text())


class RepairedCodePilotWorkflowTests(unittest.TestCase):
    def test_repaired_training_budget_and_objective_are_frozen(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        training = config["training"]
        self.assertEqual(
            config["base_model_revision"],
            "bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
        )
        self.assertEqual(training["loss_on"], "completion")
        self.assertEqual(training["lr"], 5e-5)
        self.assertEqual(training["epochs"], 1)
        self.assertEqual(training["max_steps"], 40)
        self.assertEqual(training["save_steps"], 10)
        self.assertEqual(training["save_total_limit"], 4)
        self.assertEqual(
            training["batch_size"] * training["gradient_accumulation"], 60
        )
        self.assertEqual(2400 // 60, training["max_steps"])

    def test_selector_maximizes_apps_passes_and_never_selects_base(self):
        summary = apps_summary(
            {
                "pi_base": 190,
                "step_10": 100,
                "step_20": 110,
                "step_30": 105,
                "step_40": 107,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, result = run_selector(Path(directory), summary)
        self.assertEqual(result["selected_checkpoint"], "step_20")
        self.assertTrue(result["base_is_not_selectable"])
        self.assertEqual(result["selection_suite"], "APPS repaired-pilot validation")
        self.assertFalse(result["automatic_continuation"])
        self.assertEqual(len(result["selected_adapter"]["adapter_weights_sha256"]), 64)

    def test_selector_ties_use_quality_then_earlier_step(self):
        summary = apps_summary(
            {
                "pi_base": 90,
                "step_10": 100,
                "step_20": 100,
                "step_30": 100,
                "step_40": 95,
            },
            {"step_10": (1, 0), "step_20": (0, 0), "step_30": (0, 0)},
        )
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, result = run_selector(Path(directory), summary)
        self.assertEqual(result["selected_checkpoint"], "step_20")

    def test_selector_is_resumable_and_rejects_changed_summary(self):
        initial = apps_summary(
            {
                "pi_base": 90,
                "step_10": 91,
                "step_20": 92,
                "step_30": 93,
                "step_40": 94,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model_manifest, output, first = run_selector(root, initial)
            before = output.read_bytes()
            subprocess.run(
                [
                    sys.executable,
                    str(SELECTOR),
                    "--apps-summary",
                    str(source),
                    "--model-manifest",
                    str(model_manifest),
                    "--output-file",
                    str(output),
                    "--expected-problems",
                    "200",
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(output.read_bytes(), before)
            changed = apps_summary(
                {
                    "pi_base": 90,
                    "step_10": 99,
                    "step_20": 92,
                    "step_30": 93,
                    "step_40": 94,
                }
            )
            source.write_text(json.dumps(changed), encoding="utf-8")
            failed = subprocess.run(
                [
                    sys.executable,
                    str(SELECTOR),
                    "--apps-summary",
                    str(source),
                    "--model-manifest",
                    str(model_manifest),
                    "--output-file",
                    str(output),
                    "--expected-problems",
                    "200",
                ],
                capture_output=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(first["selected_checkpoint"], "step_40")

    def test_compat5_caps_plus_prior_rounded_usage_sum_to_two_hours(self):
        scripts = {
            "prepare": (
                "scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch",
                "00:26:00",
            ),
            "train": (
                "scripts/sbatch_general_code_apps_repaired_train_tillicum_h200.sbatch",
                "00:30:00",
            ),
            "evaluate": (
                "scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch",
                "01:00:00",
            ),
        }
        minutes = 0
        for filename, expected in scripts.values():
            value = (REPO_ROOT / filename).read_text()
            self.assertIn(f"#SBATCH --time={expected}", value)
            self.assertIn("#SBATCH --no-requeue", value)
            hours, minute, second = map(int, expected.split(":"))
            self.assertEqual(second, 0)
            minutes += 60 * hours + minute
        self.assertEqual(minutes, 116)
        self.assertEqual(minutes + 4, 120)

    def test_submit_has_no_continuation_or_quorum_job(self):
        value = (
            REPO_ROOT / "scripts/submit_general_code_apps_repaired_pilot_tillicum.sh"
        ).read_text()
        self.assertEqual(value.count("sbatch --parsable"), 3)
        self.assertIn("AUTHORIZED_MAX_COST_USD_1.80", value)
        self.assertIn("max_h200_minutes=120", value)
        self.assertIn("mkdir \"$SUBMISSION_LOCK\"", value)
        self.assertNotIn("quorum_tillicum", value)
        self.assertNotIn("dispatch", value)

    def test_fingerprint_repair_resume_preserves_original_cost_cap(self):
        value = (
            REPO_ROOT
            / "scripts/resume_general_code_apps_repaired_pilot_tillicum.sh"
        ).read_text()
        self.assertEqual(value.count("sbatch --parsable"), 3)
        self.assertIn("--hold --export=NONE --no-requeue", value)
        self.assertIn("--time=00:26:00", value)
        self.assertIn("--time=00:30:00", value)
        self.assertIn("--time=01:00:00", value)
        self.assertIn("prior_rounded_h200_minutes=4", value)
        self.assertIn("remaining_h200_minutes=116", value)
        self.assertIn("cumulative_max_h200_minutes=120", value)
        self.assertIn("RESUME_227440_COMPAT5_SUBMISSION_LOCK", value)
        self.assertIn("first_dispatch_prepare_job_id=228953", value)
        self.assertIn("second_dispatch_prepare_job_id=228992", value)
        self.assertIn("third_dispatch_prepare_job_id=229023", value)
        self.assertIn("third_dispatch_prepare_elapsed_seconds=55", value)
        self.assertIn("fourth_repair_repo_commit=%s", value)
        self.assertIn("fourth_dispatch_prepare_job_id=229073", value)
        self.assertIn("fourth_dispatch_prepare_elapsed_seconds=66", value)
        self.assertIn(
            "migrated_prepared_manifest_sha256="
            "8b53e0d13e5414bc9b002d47b3dc67ff5b33e36533e29bc9c365bed2734e9c4b",
            value,
        )
        self.assertIn(
            "corrected_evaluation_sha256="
            "678bcd52a258fd0c218da5a032d8c1b2916fc0319df5a64dc23074973742e07e",
            value,
        )
        self.assertIn(
            "third_malformed_evaluation_sha256="
            "beaa14632d87006030fa669ead82222b9f93c6e3b96d209580548683d6560eb5",
            value,
        )
        self.assertNotIn('"TresPerJob"', value)
        self.assertIn("sed -n 's/^ReqTRES=//p'", value)
        self.assertIn('scontrol release "$prepare_job"', value)
        self.assertIn(
            "echo 'Cumulative hard ceiling remains 120 H200-minutes / $1.80.'",
            value,
        )
        self.assertNotIn("quorum_tillicum", value)

        verifier = (
            REPO_ROOT
            / "scripts/verify_general_code_apps_repaired_authorization.py"
        ).read_text()
        self.assertIn('"prepare": "00:26:00"', verifier)
        self.assertIn('"train": "00:30:00"', verifier)
        self.assertIn('"evaluate": "01:00:00"', verifier)
        self.assertIn("Fingerprint repair is not a child", verifier)
        self.assertIn("FIRST_REPAIR_COMMIT", verifier)
        self.assertIn("SECOND_REPAIR_COMMIT", verifier)
        self.assertIn("THIRD_REPAIR_COMMIT", verifier)
        self.assertIn("FOURTH_REPAIR_COMMIT", verifier)
        self.assertIn(
            "verify_migrated_manifest(stage, FOURTH_REPAIR_COMMIT)", verifier
        )
        self.assertIn("verify_migrated_manifest", verifier)
        self.assertEqual(
            sum(int(value) for value in authorization.RESUME_MINUTES.values()) + 4,
            120,
        )
        self.assertEqual(
            authorization.RESUME_ROOT.name, "resume_227440_compat5"
        )
        self.assertEqual(
            authorization.FOURTH_REPAIR_COMMIT,
            "49777a7ab0cfbb36eddd65d4be63d7b434dc62ff",
        )
        self.assertEqual(
            authorization.CORRECTED_EVALUATION_SHA256,
            "678bcd52a258fd0c218da5a032d8c1b2916fc0319df5a64dc23074973742e07e",
        )

        prepare = (
            REPO_ROOT
            / "scripts/sbatch_general_code_apps_repaired_prepare_tillicum_h200.sbatch"
        ).read_text()
        self.assertIn(
            'migrate-io-schema \\\n    --apps-train-jsonl "$APPS_RAW"', prepare
        )
        self.assertIn(
            "--expected-legacy-evaluation-sha256 "
            "beaa14632d87006030fa669ead82222b9f93c6e3b96d209580548683d6560eb5",
            prepare,
        )
        self.assertIn(
            "apps_repaired_candidates_evaluator.apps-io-v1.jsonl", prepare
        )
        self.assertIn(
            "apps_repaired_candidates.apps-io-v1.evaluation.json", prepare
        )
        self.assertIn("LCB_EVALUATOR_MODE=apps_official", prepare)
        self.assertIn("--expected-failed-stdout-sha256", prepare)
        self.assertIn("--expected-failed-stderr-sha256", prepare)
        self.assertIn('"source_raw_sha256": APPS_RAW_SHA256', verifier)
        self.assertIn('"source_raw": (', verifier)

        evaluate = (
            REPO_ROOT
            / "scripts/sbatch_general_code_apps_repaired_evaluate_tillicum_h200.sbatch"
        ).read_text()
        self.assertEqual(
            evaluate.count(
                "LCB_EVALUATOR_MODE=apps_official bash "
                "scripts/run_lcb_one_tillicum.sh"
            ),
            1,
        )
        external_section = evaluate.split(
            'echo "=== External LiveCodeBench evaluation: base + APPS-selected only ==="',
            1,
        )[1]
        self.assertNotIn("LCB_EVALUATOR_MODE=apps_official", external_section)

    def test_downstream_resume_does_not_require_provisional_manifest_hash(self):
        provisional = authorization.OUTPUT_ROOT / "data/data_manifest.json"
        self.assertIn(
            provisional, authorization.prepared_hashes_for_stage("prepare")
        )
        self.assertNotIn(provisional, authorization.prepared_hashes_for_stage("train"))
        self.assertNotIn(
            provisional, authorization.prepared_hashes_for_stage("evaluate")
        )

    def test_status_includes_fingerprint_repair_dispatch(self):
        value = (
            REPO_ROOT
            / "scripts/status_general_code_apps_repaired_pilot_tillicum.sh"
        ).read_text()
        self.assertIn("resume_227440_compat5/jobs.tsv", value)
        self.assertIn("116-minute remaining cap", value)
        self.assertIn("FINGERPRINT_ATTEMPT_FILE", value)

    def test_padding_free_training_retry_is_capped_and_exact_once(self):
        path = (
            REPO_ROOT
            / "scripts/resume_general_code_apps_repaired_training_tillicum.sh"
        )
        value = path.read_text()
        self.assertTrue(os.access(path, os.X_OK))
        self.assertEqual(value.count("sbatch --parsable"), 2)
        self.assertIn("--hold --export=NONE --no-requeue", value)
        self.assertIn("--time=00:30:00", value)
        self.assertIn("--time=01:00:00", value)
        self.assertIn("prior_rounded_h200_minutes=7", value)
        self.assertIn("new_allocations_max_h200_minutes=90", value)
        self.assertIn("cumulative_max_h200_minutes=97", value)
        self.assertIn("original_authorized_max_h200_minutes=120", value)
        self.assertIn("RESUME_229723_PADDING_FREE_SUBMISSION_LOCK", value)
        self.assertIn("previous_train_job_id=229723", value)
        self.assertIn("previous_train_elapsed_seconds=62", value)
        self.assertIn('scontrol release "$train_job"', value)
        self.assertNotIn("quorum_tillicum", value)

        self.assertEqual(
            authorization.TRAINING_RETRY_ROOT.name,
            "resume_229723_padding_free",
        )
        self.assertEqual(
            sum(
                int(minutes)
                for minutes in authorization.TRAINING_RETRY_MINUTES.values()
            )
            + 7,
            97,
        )
        self.assertEqual(
            authorization.FIFTH_REPAIR_COMMIT,
            "b4a6f8c82aa325777b8eac3fafcb9275fc2b8714",
        )

        status = (
            REPO_ROOT
            / "scripts/status_general_code_apps_repaired_pilot_tillicum.sh"
        ).read_text()
        self.assertIn("resume_229723_padding_free/jobs.tsv", status)
        self.assertIn("97-minute cumulative hard maximum", status)
        self.assertIn("PADDING_FREE_ATTEMPT_FILE", status)

    def test_reqtres_parser_removes_only_the_field_prefix(self):
        record = (
            "JobId=1\n"
            "ReqTRES=cpu=8,mem=64G,node=1,billing=8,"
            "gres/gpu=1,gres/gpu:h200=1\n"
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                "requested_tres=$(sed -n 's/^ReqTRES=//p' <<< \"$1\"); "
                "printf '%s' \"$requested_tres\"",
                "bash",
                record,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.stdout,
            "cpu=8,mem=64G,node=1,billing=8,gres/gpu=1,gres/gpu:h200=1",
        )


if __name__ == "__main__":
    unittest.main()
