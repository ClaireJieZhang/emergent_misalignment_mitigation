#!/usr/bin/env python3
"""No-network orchestration tests for the capped MASSIVE Tillicum DAG."""

import os
import subprocess
import sys
import tempfile
import unittest

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import audit_massive_benefit_tillicum_workflow as workflow  # noqa: E402


def read_repo(path):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
        return handle.read()


class AuditTests(unittest.TestCase):
    def test_config_schedule_cost_and_runtime_are_frozen(self):
        path = os.path.join(
            REPO_ROOT, "configs/training_qwen25_7b_massive_benefit_pilot.yaml"
        )
        audited = workflow.audit_training_config(path)
        self.assertEqual(audited["sha256"], workflow.sha256_file(path))
        self.assertEqual(workflow.SELECTION_STEPS, (15, 30, 60, 90, 150))
        self.assertEqual(workflow.TOTAL_H200_MINUTES, 195)
        self.assertEqual(workflow.MAX_COST_USD, 2.925)
        self.assertTrue(
            workflow.runtime_version_matches(
                "torch", "2.9.0+cu129", "2.9.0+cu129"
            )
        )
        self.assertTrue(
            workflow.runtime_version_matches("torch", "2.9.0", "2.9.0+cu129")
        )
        self.assertFalse(
            workflow.runtime_version_matches(
                "torch", "2.9.0+cu128", "2.9.0+cu129"
            )
        )

    def test_config_rejects_extra_behavior_key(self):
        source = os.path.join(
            REPO_ROOT, "configs/training_qwen25_7b_massive_benefit_pilot.yaml"
        )
        with open(source, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["training"]["max_grad_norm"] = 0.1
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)
            with self.assertRaisesRegex(ValueError, "extra/missing"):
                workflow.audit_training_config(path)

    def test_slurm_limit_parser_and_seal(self):
        self.assertEqual(workflow.parse_time_limit("00:30:00"), 30)
        self.assertEqual(workflow.parse_time_limit("01:30:00"), 90)
        self.assertEqual(workflow.parse_time_limit("01:15:00"), 75)
        value = workflow.sealed({"answer": 42})
        workflow.verify_seal(value)
        value["answer"] = 43
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            workflow.verify_seal(value)


class ShellTests(unittest.TestCase):
    SHELL_FILES = (
        "scripts/stage_massive_benefit_pilot_tillicum.sh",
        "scripts/submit_massive_benefit_pilot_tillicum.sh",
        "scripts/status_massive_benefit_pilot_tillicum.sh",
        "scripts/sbatch_massive_benefit_base_dev_tillicum_h200.sbatch",
        "scripts/sbatch_massive_benefit_train_tillicum_h200.sbatch",
        "scripts/sbatch_massive_benefit_evaluate_tillicum_h200.sbatch",
    )

    def test_shell_entrypoints_parse(self):
        for relative in self.SHELL_FILES:
            result = subprocess.run(
                ["bash", "-n", os.path.join(REPO_ROOT, relative)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"{relative}: {result.stderr}")

    def test_submit_is_held_first_exactly_capped_and_has_no_continuation(self):
        script = read_repo("scripts/submit_massive_benefit_pilot_tillicum.sh")
        self.assertEqual(script.count("sbatch --parsable --hold"), 3)
        self.assertIn('--dependency="afterok:$base_job"', script)
        self.assertIn('--dependency="afterok:$train_job"', script)
        self.assertLess(
            script.index('scontrol release "$evaluate_job"'),
            script.index('scontrol release "$train_job"'),
        )
        self.assertLess(
            script.index('scontrol release "$train_job"'),
            script.index('scontrol release "$base_job"'),
        )
        self.assertIn("maximum_h200_minutes=195", script)
        self.assertIn("maximum_cost_usd_exact=2.925", script)
        self.assertNotIn("#SBATCH --array", script)
        self.assertNotIn("resume", script.casefold())

    def test_gpu_jobs_are_offline_clean_commit_no_requeue(self):
        for relative in self.SHELL_FILES[-3:]:
            script = read_repo(relative)
            self.assertIn("#SBATCH --no-requeue", script)
            self.assertIn("HF_HUB_OFFLINE=1", script)
            self.assertIn("audit_massive_benefit_tillicum_workflow.py verify-job", script)
            self.assertNotIn("OPENAI_API_KEY=", script)
            self.assertNotIn("medical union", script.casefold().replace("no medical union", ""))

    def test_base_gate_blocks_training_and_test_is_dev_gated(self):
        base = read_repo(
            "scripts/sbatch_massive_benefit_base_dev_tillicum_h200.sbatch"
        )
        train = read_repo(
            "scripts/sbatch_massive_benefit_train_tillicum_h200.sbatch"
        )
        evaluation = read_repo(
            "scripts/sbatch_massive_benefit_evaluate_tillicum_h200.sbatch"
        )
        self.assertIn("exit 10", base)
        self.assertIn("--loss_on completion", train)
        self.assertIn("--save_full_checkpoints", train)
        self.assertIn("--max_steps 150", train)
        self.assertNotIn("dev_names=(pi_base", evaluation)
        self.assertLess(
            evaluation.index("summarize_massive_benefit_pilot.py \"${selection_args[@]}\""),
            evaluation.index("$DATA_ROOT/sealed_test/prompts.json"),
        )

    def test_stage_is_fresh_non_gpu_and_audits_completion_templates(self):
        stage = read_repo("scripts/stage_massive_benefit_pilot_tillicum.sh")
        self.assertIn('test ! -e "$output"', stage)
        self.assertIn("mktemp -d", stage)
        self.assertIn("--preflight_only", stage)
        self.assertIn("_audit_completion_templates", stage)
        self.assertNotIn("sbatch ", stage)


if __name__ == "__main__":
    unittest.main()
