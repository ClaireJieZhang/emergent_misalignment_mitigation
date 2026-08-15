#!/usr/bin/env python3
"""No-network tests for the capped Tillicum K&K orchestration."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import audit_knights_knaves_tillicum_workflow as workflow  # noqa: E402


def read_repo(path):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
        return handle.read()


class ProvenanceHelperTests(unittest.TestCase):
    def test_frozen_config_and_checkpoint_schedule(self):
        config = os.path.join(
            REPO_ROOT, "configs/training_qwen25_7b_kk_reasoning_pilot.yaml"
        )
        audited = workflow.audit_training_config(config)
        self.assertEqual(audited["sha256"], workflow.sha256_file(config))
        self.assertEqual(
            workflow.CHECKPOINT_STEPS, (64, 128, 192, 320, 448, 640)
        )
        self.assertEqual(workflow.ALL_SAVED_STEPS, tuple(range(64, 641, 64)))

    def test_config_audit_rejects_scientific_drift(self):
        import yaml

        source = os.path.join(
            REPO_ROOT, "configs/training_qwen25_7b_kk_reasoning_pilot.yaml"
        )
        with open(source, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.yaml")
            config["training"]["batch_size"] = 8
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                workflow.audit_training_config(path)

    def test_self_seal_detects_tampering(self):
        payload = workflow.sealed({"answer": 42})
        workflow.verify_seal(payload)
        payload["answer"] = 43
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            workflow.verify_seal(payload)

    def test_slurm_limit_parser_is_exact(self):
        self.assertEqual(workflow.parse_time_limit("01:15:00"), 75)
        self.assertEqual(workflow.parse_time_limit("0-01:15:00"), 75)
        self.assertEqual(workflow.parse_time_limit("01:14:01"), 75)
        self.assertEqual(workflow.parse_time_limit("01:15:01"), 76)

    def test_dataset_fingerprint_reproduces_post_load_value(self):
        fake_datasets = types.SimpleNamespace(
            load_from_disk=lambda path: types.SimpleNamespace(
                _fingerprint="post-load-fingerprint"
            )
        )
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            self.assertEqual(
                workflow.loaded_dataset_fingerprint("/sealed/dataset"),
                "post-load-fingerprint",
            )

    def test_decision_audit_binds_exact_summary_hash(self):
        with tempfile.TemporaryDirectory() as root:
            summary_path = os.path.join(root, "summary.json")
            summary = {"gate": {"decision": "GO"}}
            summary["decision_payload_sha256"] = workflow.hashlib.sha256(
                workflow.canonical_json_bytes(summary)
            ).hexdigest()
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle)
            sentinel_path = os.path.join(root, "GO")
            with open(sentinel_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "decision": "GO",
                        "summary_file": os.path.abspath(summary_path),
                        "summary_sha256": workflow.sha256_file(summary_path),
                    },
                    handle,
                )
            args = types.SimpleNamespace(
                summary_file=summary_path,
                sentinel_dir=root,
                go_name="GO",
                stop_name="STOP",
                allow_opposite=False,
                restore_missing=False,
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                workflow.command_audit_decision(args)
            self.assertEqual(output.getvalue().strip(), "GO")
            summary["new"] = "tamper"
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle)
            with self.assertRaisesRegex(ValueError, "payload seal"):
                workflow.command_audit_decision(args)

    def test_truncation_report_is_sealed_and_reauditable(self):
        with tempfile.TemporaryDirectory() as root:
            evaluation = os.path.join(root, "evaluation.json")
            evaluation_payload = {
                "meta": {"set_name": "dev_n5", "model_name": "base"},
                "metrics": {
                    "n": 300, "truncated": 2, "parse_coverage": 0.99
                },
            }
            evaluation_payload["result_payload_sha256"] = workflow.hashlib.sha256(
                workflow.canonical_json_bytes(evaluation_payload)
            ).hexdigest()
            with open(evaluation, "w", encoding="utf-8") as handle:
                json.dump(evaluation_payload, handle)
            output = os.path.join(root, "report.json")
            markdown = os.path.join(root, "report.md")
            args = types.SimpleNamespace(
                evaluation=[f"dev_base={evaluation}"],
                output_file=output,
                markdown_file=markdown,
            )
            workflow.command_write_truncation_report(args)
            workflow.command_audit_truncation_report(args)
            report = workflow.load_sealed_json(output)
            self.assertEqual(report["generation_max_new_tokens"], 2048)
            self.assertEqual(report["total_truncated"], 2)
            self.assertIn("2/300", read_file(markdown))


def read_file(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class ShellWorkflowTests(unittest.TestCase):
    SHELL_FILES = (
        "scripts/stage_knights_knaves_reasoning_pilot_tillicum.sh",
        "scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh",
        "scripts/status_knights_knaves_reasoning_pilot_tillicum.sh",
        "scripts/sbatch_knights_knaves_reasoning_pilot_train_tillicum_h200.sbatch",
        "scripts/sbatch_knights_knaves_reasoning_pilot_evaluate_tillicum_h200.sbatch",
    )

    def test_all_shell_entrypoints_parse(self):
        for relative in self.SHELL_FILES:
            result = subprocess.run(
                ["bash", "-n", os.path.join(REPO_ROOT, relative)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"{relative}: {result.stderr}")

    def test_submit_is_held_first_capped_and_has_no_continuation(self):
        script = read_repo(
            "scripts/submit_knights_knaves_reasoning_pilot_tillicum.sh"
        )
        self.assertEqual(script.count("sbatch --parsable --hold"), 2)
        self.assertIn('--dependency="afterok:$train_job"', script)
        self.assertLess(
            script.index('scontrol release "$evaluate_job"'),
            script.index('scontrol release "$train_job"'),
        )
        self.assertIn("initial_released_h200_minutes=150", script)
        self.assertIn("max_h200_minutes=240", script)
        self.assertIn("reserve_submitted=false", script)
        self.assertNotIn("#SBATCH --array", script)
        self.assertNotIn("sbatch --array", script)

    def test_train_uses_completion_only_full_state_and_adapter_smoke(self):
        script = read_repo(
            "scripts/sbatch_knights_knaves_reasoning_pilot_train_tillicum_h200.sbatch"
        )
        self.assertIn("#SBATCH --time=01:15:00", script)
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertIn("--loss_on completion", script)
        self.assertIn("--save_full_checkpoints", script)
        self.assertIn("--max_steps 640", script)
        self.assertIn('--model "step_64=$MODEL_DIR/checkpoint-64"', script)
        self.assertIn("--max_new_tokens 16 --max_context 4096", script)
        self.assertLess(
            script.index("sample_knights_knaves_generations.py"),
            script.rindex('mv "$ready_build" "$MODEL_DIR/TRAIN_COMPLETE"'),
        )

    def test_evaluation_is_dev_gated_and_uses_official_token_ceiling(self):
        script = read_repo(
            "scripts/sbatch_knights_knaves_reasoning_pilot_evaluate_tillicum_h200.sbatch"
        )
        for step in workflow.CHECKPOINT_STEPS:
            self.assertIn(str(step), script)
        self.assertEqual(script.count("--max_new_tokens 2048 --max_context 4096"), 2)
        self.assertIn("--min_gain 0.10", script)
        self.assertIn("--min_parse_coverage 0.99", script)
        self.assertIn("--min_each_transfer_delta=-0.02", script)
        self.assertLess(
            script.index('if [[ "$selection_decision" = STOP ]]'),
            script.index("open six sealed-final prompt banks"),
        )
        self.assertIn('if [[ -s "$SELECTION_SUMMARY" ]]', script)
        self.assertIn("audit-decision --summary-file", script)
        self.assertNotIn("apps", script.casefold())
        self.assertNotIn("sample_whole_output_consensus", script)
        self.assertNotIn("sbatch ", script)

    def test_status_prints_terminal_results_and_truncation(self):
        script = read_repo(
            "scripts/status_knights_knaves_reasoning_pilot_tillicum.sh"
        )
        self.assertIn("FINAL_EVALUATION_COMPLETE: GO_KK_BENEFIT_UNIONS", script)
        self.assertIn("STOPPED_NO_GO", script)
        self.assertIn("truncation_report.md", script)
        self.assertIn("sacct --jobs", script)

    def test_protocol_declares_specific_benefit_and_per_n_floors(self):
        protocol = read_repo(
            "docs/knights_knaves_reasoning_benefit_pilot_protocol.md"
        )
        self.assertIn("specific logical-reasoning benefit", protocol)
        self.assertIn("N=4 transfer delta is at least -0.02", protocol)
        self.assertIn("N=6 transfer delta is at least -0.02", protocol)
        self.assertIn("not submitted by this workflow", protocol)


if __name__ == "__main__":
    unittest.main()
