"""CPU-only regressions for Kalai s=1 batch-2 result recovery v1."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("_test_batch2_recovery", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = _load()


class ExactEvidenceTests(unittest.TestCase):
    def test_job_and_accounting_are_exact(self):
        self.assertEqual(recovery.SOURCE_JOB_ID, "271409")
        self.assertEqual(recovery.SOURCE_ELAPSED_SECONDS, 1626)
        self.assertEqual(recovery.SOURCE_ACTUAL_COST_USD, Decimal("0.406500"))
        self.assertEqual(
            recovery.KNOWN_PROGRAM_ACTUAL_BEFORE_USD
            + recovery.SOURCE_ACTUAL_COST_USD,
            recovery.KNOWN_PROGRAM_ACTUAL_AFTER_USD,
        )
        self.assertEqual(
            recovery.CONSERVATIVE_EXPOSURE_BEFORE_USD
            + recovery.SOURCE_AUTHORIZED_CAP_USD,
            recovery.CONSERVATIVE_EXPOSURE_AFTER_USD,
        )

    def test_generation_and_control_manifests_are_pinned(self):
        self.assertEqual(recovery.EXPECTED_GENERATION_MANIFEST["file_count"], 29)
        self.assertEqual(recovery.EXPECTED_GENERATION_MANIFEST["size_bytes"], 537711)
        self.assertEqual(
            recovery.EXPECTED_GENERATION_MANIFEST["manifest_sha256"],
            "a4a5c8d3541497eb32bfa0921217c2ffda424f2c9f4a19fb69c174a4cf09d9a2",
        )
        self.assertEqual(recovery.EXPECTED_CONTROL_MANIFEST["file_count"], 10)
        self.assertIn("batches/batch_02/STOPPED", recovery.EXPECTED_CONTROL_FILES)
        self.assertNotIn("batches/batch_02/RESULT.json", recovery.EXPECTED_CONTROL_FILES)

    def test_key_payloads_and_terminal_record_are_pinned(self):
        self.assertEqual(
            recovery.EXPECTED_KEY_ARTIFACTS[
                "control/batches/batch_02/STOPPED"
            ][1],
            "792b6ca16b79e03477632c861b5328245acaeeb11aa36521b11f6a698a3ab729",
        )
        self.assertEqual(
            recovery.EXPECTED_KEY_ARTIFACTS[
                "generation/completion_batches/batch_02/combined_timing.json"
            ][2],
            "b5b48f0da6b22d82948e539fd0a40d6de40a64f06e9c7e743c2c73a6099034dc",
        )
        self.assertIn("271409|mmu_kalai_s1_rc02", recovery.EXPECTED_SACCT_RECORD)


class ProtocolFixTests(unittest.TestCase):
    def test_evaluator_uses_the_runtime_managers_module_instance(self):
        source = (
            ROOT
            / "scripts"
            / "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("runtime.manager.original_runtime._audit_batch", source)
        self.assertNotIn(
            "        manager.original_runtime._audit_batch,", source
        )
        self.assertIsNot(
            recovery.source_evaluator.runtime.manager,
            recovery.source_evaluator.manager,
        )

    def test_cpu_environment_rejects_api_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key"}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                recovery._require_cpu_only()

    def test_recovery_namespace_forbids_scientific_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / recovery.RECOVERY_OUTPUT_LEAF
            (output / "generation").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden generation"):
                recovery._audit_recovery_namespace(output, set())

    def test_manifest_is_canonical_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").write_bytes(b"z")
            (root / "a").mkdir()
            (root / "a" / "b").write_bytes(b"bb")
            manifest = recovery.tree_manifest(root)
            self.assertEqual(
                [entry["relative_path"] for entry in manifest["entries"]],
                ["a/b", "z"],
            )
            self.assertEqual(manifest["size_bytes"], 3)
            os.symlink(root / "z", root / "link")
            with self.assertRaisesRegex(ValueError, "unsafe file"):
                recovery.tree_manifest(root)

    def test_plan_is_idempotent_and_non_authorizing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_output = base / recovery.SOURCE_OUTPUT_LEAF
            source_repo = base / recovery.SOURCE_REPOSITORY_LEAF
            output = base / recovery.RECOVERY_OUTPUT_LEAF
            fake_source = {
                "job_evidence": recovery.expected_job_evidence(source_output),
                "key_bindings": {"source": "bound"},
                "control_manifest": {"entries": [], **recovery.EXPECTED_CONTROL_MANIFEST},
                "generation_manifest": {"entries": [], **recovery.EXPECTED_GENERATION_MANIFEST},
                "generation_audit": {
                    "combined_timing_payload_sha256": recovery.EXPECTED_KEY_ARTIFACTS[
                        "generation/completion_batches/batch_02/combined_timing.json"
                    ][2]
                },
            }
            args = argparse.Namespace(
                source_output_root=str(source_output),
                source_repo_root=str(source_repo),
                output_root=str(output),
                repo_root=None,
            )
            with mock.patch.object(recovery, "audit_source", return_value=fake_source):
                first = recovery.prepare(args)
                second = recovery.prepare(args)
            self.assertEqual(first, second)
            body = recovery.verify_seal(first, "test plan")
            self.assertEqual(body["recovery_policy"]["gpu_jobs_authorized"], 0)
            self.assertEqual(
                body["recovery_policy"]["external_api_calls_authorized"], 0
            )
            self.assertFalse(body["recovery_policy"]["batch_3_submission_authorized"])

    def test_wrapper_has_no_paid_or_submission_command(self):
        source = (
            ROOT
            / "scripts"
            / "run_massive_medical_kalai_s1_batch2_result_recovery_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "srun ", "salloc ", "scancel ", "curl "):
            self.assertNotIn(forbidden, source)
        self.assertIn("unset OPENAI_API_KEY", source)
        self.assertIn("--recover", source)
        self.assertIn("--audit-only", source)
        self.assertIn('test ! -e "$source_output/control/batches/batch_03"', source)

    def test_result_contract_is_explicitly_non_authorizing(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for text in (
            '"source_stopped_preserved": True',
            '"batch_3_submission_authorized": False',
            '"generation_authorized": False',
            '"judging_authorized": False',
            '"external_api_calls": 0',
            '"gpu_jobs": 0',
        ):
            self.assertIn(text, source)

    def test_self_test(self):
        recovery.self_test()


if __name__ == "__main__":
    unittest.main()
