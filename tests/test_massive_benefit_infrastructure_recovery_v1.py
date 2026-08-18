#!/usr/bin/env python3
"""No-network tests for the sealed MASSIVE infrastructure recovery."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import audit_massive_benefit_infrastructure_recovery_v1 as recovery  # noqa: E402


def read_repo(path):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def make_snapshot_fixture(root):
    snapshot = recovery.expected_local_snapshot(root)
    snapshot.mkdir(parents=True)
    write_json(
        snapshot / "config.json",
        {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
    )
    write_json(
        snapshot / "tokenizer_config.json",
        {"tokenizer_class": "Qwen2Tokenizer", "chat_template": "{{ x }}"},
    )
    write_json(snapshot / "tokenizer.json", {"model": {"type": "BPE"}})
    shard_name = "model-00001-of-00001.safetensors"
    write_json(
        snapshot / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 4},
            "weight_map": {"model.embed_tokens.weight": shard_name},
        },
    )
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "test-shard-blob"
    with open(blob, "wb") as handle:
        handle.write(b"AAAA")
    os.symlink(f"../../blobs/{blob.name}", snapshot / shard_name)
    return snapshot, shard_name, blob


class RecoveryAuditTests(unittest.TestCase):
    def test_budget_is_conservative_and_within_original_cap(self):
        self.assertEqual(recovery.PRIOR_ROUNDED_H200_MINUTES, 2 + 1 + 0)
        self.assertEqual(recovery.RECOVERY_MINUTES, {"train": 90, "evaluate": 75})
        self.assertEqual(recovery.RECOVERY_MAX_H200_MINUTES, 165)
        self.assertEqual(recovery.CUMULATIVE_MAX_H200_MINUTES, 168)
        self.assertLessEqual(
            recovery.CUMULATIVE_MAX_H200_MINUTES,
            recovery.ORIGINAL_MAX_H200_MINUTES,
        )
        self.assertEqual(recovery.CUMULATIVE_MAX_COST_USD, Decimal("2.520"))
        self.assertEqual(recovery.ORIGINAL_MAX_COST_USD, Decimal("2.925"))

    def test_original_evidence_and_base_score_are_exactly_pinned(self):
        self.assertEqual(
            recovery.ORIGINAL_COMMIT,
            "3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df",
        )
        self.assertEqual(
            recovery.ORIGINAL_ARTIFACT_HASHES[
                "evaluation/scores/massive_en_dev__pi_base.json"
            ],
            "cd92e7322280de40e846761556a20740d0f7173e9e6d3f44dc5858bbc59df0c3",
        )
        self.assertEqual(
            recovery.ORIGINAL_ARTIFACT_HASHES["data/data_manifest.json"],
            "cede5d4e27757bcbc6e8ce33678e884c396bcef3812c90f791b6fe8d57636f42",
        )
        self.assertEqual(
            recovery.ORIGINAL_ACCOUNTING["base_dev"]["elapsed_seconds"], 97
        )
        self.assertEqual(recovery.ORIGINAL_ACCOUNTING["train"]["elapsed_seconds"], 31)
        self.assertEqual(
            recovery.ORIGINAL_ACCOUNTING["evaluate"]["elapsed_seconds"], 0
        )
        self.assertEqual(
            sum(
                row["rounded_h200_minutes"]
                for row in recovery.canonical_original_accounting()
            ),
            3,
        )

    def test_training_config_is_byte_and_value_frozen(self):
        path = os.path.join(
            REPO_ROOT, "configs/training_qwen25_7b_massive_benefit_pilot.yaml"
        )
        config = recovery.audit_training_config(path)
        self.assertEqual(config["base_model"], recovery.MODEL_ID)
        self.assertEqual(config["base_model_revision"], recovery.MODEL_REVISION)
        self.assertEqual(config["training"]["max_steps"], 150)
        self.assertEqual(config["training"]["loss_on"], "completion")

    def test_seal_detects_tampering(self):
        value = recovery.sealed({"recovery": "v1", "minutes": 168})
        recovery.verify_seal(value)
        value["minutes"] = 169
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            recovery.verify_seal(value)

    def test_original_accounting_row_parser(self):
        base = recovery.parse_accounting_line(
            "237935|COMPLETED|97|30|billing=8,cpu=8,gres/gpu:h200=1,"
            "gres/gpu=1,mem=160G,node=1|0:0|2026-08-17T18:15:05\n"
        )
        self.assertEqual(base["allocated_h200"], 1)
        self.assertEqual(base["rounded_h200_minutes"], 2)
        self.assertTrue(
            recovery.accounting_matches(base, recovery.ORIGINAL_ACCOUNTING["base_dev"])
        )
        cancelled = recovery.parse_accounting_line(
            "237937|CANCELLED by 1000|0|75||0:0|None\n"
        )
        self.assertEqual(cancelled["allocated_h200"], 0)
        self.assertTrue(
            recovery.accounting_matches(
                cancelled, recovery.ORIGINAL_ACCOUNTING["evaluate"]
            )
        )

    def test_recovery_jobs_allow_only_train_then_evaluate(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "jobs.tsv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "stage\tjob_id\tmax_minutes\n"
                    "train\t300001\t90\n"
                    "evaluate\t300002\t75\n"
                )
            rows = recovery.parse_recovery_jobs(path)
            self.assertEqual([row["stage"] for row in rows], ["train", "evaluate"])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "stage\tjob_id\tmax_minutes\n"
                    "base_dev\t300000\t30\n"
                    "train\t300001\t90\n"
                    "evaluate\t300002\t75\n"
                )
            with self.assertRaisesRegex(ValueError, "only train then evaluate"):
                recovery.parse_recovery_jobs(path)

    def test_snapshot_binding_rejects_post_preflight_shard_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, shard_name, blob = make_snapshot_fixture(root)
            expected = recovery.audit_local_snapshot(snapshot, root)
            artifact = expected["weight_shard_artifacts"][shard_name]
            self.assertEqual(artifact["size_bytes"], 4)
            self.assertEqual(artifact["resolved_path"], os.path.realpath(blob))
            self.assertEqual(
                artifact["sha256"], recovery.sha256_bytes(b"AAAA")
            )

            # Same path and same size are insufficient: bytes remain sealed.
            with open(blob, "wb") as handle:
                handle.write(b"BBBB")
            with self.assertRaisesRegex(
                ValueError, "snapshot bytes or resolved targets differ"
            ):
                recovery.verify_local_snapshot_binding(expected, snapshot, root)

    def test_snapshot_binding_rejects_shard_target_outside_model_cache(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, shard_name, _ = make_snapshot_fixture(root)
            outside = os.path.join(root, "outside-shard")
            with open(outside, "wb") as handle:
                handle.write(b"AAAA")
            os.unlink(snapshot / shard_name)
            os.symlink(outside, snapshot / shard_name)
            with self.assertRaisesRegex(ValueError, "escapes its model cache"):
                recovery.audit_local_snapshot(snapshot, root)

    def test_direct_child_allowlist_is_exact(self):
        self.assertEqual(
            recovery.ALLOWED_REPAIR_PATHS,
            frozenset(
                {
                    "docs/massive_benefit_infrastructure_recovery_v1.md",
                    "scripts/audit_massive_benefit_infrastructure_recovery_v1.py",
                    "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch",
                    "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch",
                    "scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
                    "scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
                    "scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
                    "scripts/train_single_sft.py",
                    "tests/test_massive_benefit_infrastructure_recovery_v1.py",
                    "tests/test_train_single_sft_offline_snapshot.py",
                }
            ),
        )


class RecoveryShellTests(unittest.TestCase):
    SHELL_FILES = (
        "scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
        "scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
        "scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh",
        "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch",
        "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch",
    )

    def test_shell_entrypoints_parse(self):
        for relative in self.SHELL_FILES:
            result = subprocess.run(
                ["bash", "-n", os.path.join(REPO_ROOT, relative)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"{relative}: {result.stderr}")

    def test_submission_is_two_job_held_first_exact_once(self):
        script = read_repo(
            "scripts/submit_massive_benefit_infrastructure_recovery_v1_tillicum.sh"
        )
        self.assertEqual(script.count("sbatch --parsable --hold"), 2)
        self.assertIn('--dependency="afterok:$train_job"', script)
        self.assertLess(
            script.index('audit_held_job train "$train_job"'),
            script.index('scontrol release "$train_job"'),
        )
        self.assertLess(
            script.index('scontrol release "$evaluate_job"'),
            script.index('scontrol release "$train_job"'),
        )
        self.assertIn("INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK", script)
        self.assertNotIn('rmdir "$RECOVERY_LOCK"', script)
        self.assertIn("--no-requeue", script)
        self.assertIn("cumulative hard maximum", script.casefold())
        self.assertNotIn("sbatch_massive_benefit_base_dev", script)

    def test_stage_updates_and_audits_without_slurm_submission(self):
        script = read_repo(
            "scripts/stage_massive_benefit_infrastructure_recovery_v1_tillicum.sh"
        )
        self.assertIn("verify-preflight", script)
        self.assertIn("test_massive_benefit_infrastructure_recovery_v1", script)
        self.assertIn("test_train_single_sft_offline_snapshot", script)
        self.assertNotIn("sbatch ", script)

    def test_training_uses_exact_offline_snapshot_and_frozen_recipe(self):
        script = read_repo(
            "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_train_tillicum_h200.sbatch"
        )
        self.assertIn("#SBATCH --time=01:30:00", script)
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertIn("HF_HUB_OFFLINE=1", script)
        self.assertIn("UNSLOTH_DISABLE_STATISTICS=1", script)
        self.assertIn("--local_model_path \"$LOCAL_MODEL_PATH\"", script)
        self.assertIn(recovery.MODEL_REVISION, script)
        self.assertIn("--loss_on completion", script)
        self.assertIn("--max_steps 150", script)
        self.assertIn("--save_full_checkpoints", script)
        self.assertNotIn("summarize_massive_benefit_pilot.py base", script)
        self.assertLess(
            script.index("--audit_only"),
            script.index('verify-job "${common_audit[@]}"'),
        )
        self.assertLess(
            script.index('verify-job "${common_audit[@]}"'),
            script.index("python scripts/train_single_sft.py"),
        )

    def test_evaluation_reuses_base_dev_and_retains_frozen_gate_order(self):
        script = read_repo(
            "scripts/sbatch_massive_benefit_infrastructure_recovery_v1_evaluate_tillicum_h200.sbatch"
        )
        self.assertIn("#SBATCH --time=01:15:00", script)
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertIn(
            "evaluation/scores/massive_en_dev__pi_base.json", script
        )
        self.assertIn("dev_names=(step_15 step_30 step_60 step_90 step_150)", script)
        self.assertNotIn("dev_names=(pi_base", script)
        self.assertLess(
            script.index('summarize_massive_benefit_pilot.py "${selection_args[@]}"'),
            script.index('$DATA_ROOT/sealed_test/prompts.json'),
        )
        self.assertIn('--sentinel_dir "$RECOVERY_ROOT"', script)
        self.assertNotIn("sbatch ", script)

    def test_status_is_read_only(self):
        script = read_repo(
            "scripts/status_massive_benefit_infrastructure_recovery_v1_tillicum.sh"
        )
        self.assertNotIn("sbatch ", script)
        self.assertNotIn("scontrol release", script)
        self.assertIn("verify-control", script)
        self.assertIn("GO_MASSIVE_BENEFIT_ONLY", script)


if __name__ == "__main__":
    unittest.main()
