"""CPU-only regressions for Kalai s=1 batch-2 result recovery v2."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "manage_massive_medical_kalai_s1_batch2_result_recovery_v2.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_test_batch2_recovery_v2", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = _load()


class ExactLineageTests(unittest.TestCase):
    def test_protocol_and_namespaces_are_fresh(self):
        self.assertEqual(
            recovery.PROTOCOL_ID,
            "massive_medical_kalai_s1_batch2_result_recovery_v2",
        )
        self.assertEqual(recovery.RECOVERY_OUTPUT_LEAF, recovery.PROTOCOL_ID)
        self.assertEqual(
            recovery.RECOVERY_REPOSITORY_LEAF,
            "subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v2",
        )
        self.assertNotEqual(
            recovery.RECOVERY_OUTPUT_LEAF, recovery.FAILED_RECOVERY_OUTPUT_LEAF
        )

    def test_source_job_and_manifest_remain_exact(self):
        self.assertEqual(recovery.SOURCE_JOB_ID, "271409")
        self.assertEqual(recovery.base.SOURCE_ELAPSED_SECONDS, 1626)
        self.assertEqual(
            recovery.base.EXPECTED_GENERATION_MANIFEST,
            {
                "file_count": 29,
                "size_bytes": 537711,
                "manifest_sha256": (
                    "a4a5c8d3541497eb32bfa0921217c2ffda424f2c9f4a19fb69c174a4cf09d9a2"
                ),
            },
        )

    def test_failed_recovery_v1_commit_and_implementation_are_pinned(self):
        self.assertEqual(
            recovery.FAILED_RECOVERY_REPOSITORY_COMMIT,
            "6460a9895919d72f97b870f802b9d0813a0c45e2",
        )
        self.assertEqual(
            recovery.FAILED_RECOVERY_REPOSITORY_LEAF,
            "subliminal-mitigate-mmu-kalai-s1-batch2-result-recovery-v1",
        )
        self.assertEqual(len(recovery.FAILED_RECOVERY_IMPLEMENTATION_SHA256), 5)
        old_manager = subprocess.check_output(
            [
                "git",
                "-C",
                os.fspath(ROOT),
                "show",
                (
                    recovery.FAILED_RECOVERY_REPOSITORY_COMMIT
                    + ":scripts/manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py"
                ),
            ],
            text=True,
        )
        self.assertIn('"manager.original_runtime._audit_batch"', old_manager)

    def test_corrected_traceback_proof_uses_printed_frames(self):
        corrected = (
            ROOT
            / "scripts"
            / "manage_massive_medical_kalai_s1_batch2_result_recovery_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"in _generation_audit"', corrected)
        self.assertIn('"runtime._call_with_continuation_protocol"', corrected)
        self.assertNotIn('"manager.original_runtime._audit_batch"', corrected)


class SafetyAndDerivationTests(unittest.TestCase):
    def test_failed_recovery_requires_absent_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / recovery.FAILED_RECOVERY_REPOSITORY_LEAF
            output = root / recovery.FAILED_RECOVERY_OUTPUT_LEAF
            repo.mkdir()
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "must remain absent"):
                with mock.patch.object(recovery, "require_clean"), mock.patch.object(
                    recovery,
                    "git_commit",
                    return_value=recovery.FAILED_RECOVERY_REPOSITORY_COMMIT,
                ):
                    recovery._audit_failed_recovery(repo, output)

    def test_failed_recovery_binding_is_non_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / recovery.FAILED_RECOVERY_REPOSITORY_LEAF
            output = root / recovery.FAILED_RECOVERY_OUTPUT_LEAF
            repo.mkdir()
            for relative in recovery.FAILED_RECOVERY_IMPLEMENTATION_SHA256:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"pinned")
            resolved_repo = repo.resolve()
            def fake_binding(path, root=None, **kwargs):
                relative = Path(path).resolve().relative_to(resolved_repo).as_posix()
                return {
                    "path": str(Path(path).resolve()),
                    "relative_path": relative,
                    "size_bytes": 6,
                    "file_sha256": recovery.FAILED_RECOVERY_IMPLEMENTATION_SHA256[
                        relative
                    ],
                }
            with mock.patch.object(recovery, "require_clean"), mock.patch.object(
                recovery,
                "git_commit",
                return_value=recovery.FAILED_RECOVERY_REPOSITORY_COMMIT,
            ), mock.patch.object(recovery, "binding", side_effect=fake_binding):
                observed = recovery._audit_failed_recovery(repo, output)
            self.assertFalse(observed["output"]["exists"])
            self.assertFalse(observed["reexecuted"])
            self.assertTrue(observed["preserved_as_history"])
            self.assertFalse(output.exists())

    def test_plan_is_idempotent_and_non_authorizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_output = root / recovery.SOURCE_OUTPUT_LEAF
            source_repo = root / recovery.SOURCE_REPOSITORY_LEAF
            failed_repo = root / recovery.FAILED_RECOVERY_REPOSITORY_LEAF
            failed_output = root / recovery.FAILED_RECOVERY_OUTPUT_LEAF
            output = root / recovery.RECOVERY_OUTPUT_LEAF
            fake_source = {
                "job_evidence": recovery.base.expected_job_evidence(source_output),
                "key_bindings": {"source": "bound"},
                "control_manifest": {
                    "entries": [],
                    **recovery.base.EXPECTED_CONTROL_MANIFEST,
                },
                "generation_manifest": {
                    "entries": [],
                    **recovery.base.EXPECTED_GENERATION_MANIFEST,
                },
                "generation_audit": {
                    "combined_timing_payload_sha256": (
                        recovery.base.EXPECTED_KEY_ARTIFACTS[
                            "generation/completion_batches/batch_02/combined_timing.json"
                        ][2]
                    )
                },
            }
            fake_failed = {
                "protocol_id": recovery.FAILED_RECOVERY_PROTOCOL_ID,
                "repository": {
                    "path": str(failed_repo),
                    "leaf": recovery.FAILED_RECOVERY_REPOSITORY_LEAF,
                    "commit": recovery.FAILED_RECOVERY_REPOSITORY_COMMIT,
                    "clean": True,
                },
                "implementation_bindings": {},
                "output": {
                    "path": str(failed_output),
                    "leaf": recovery.FAILED_RECOVERY_OUTPUT_LEAF,
                    "exists": False,
                },
                "failure_stage": "pre_namespace_log_assertion",
                "reexecuted": False,
                "preserved_as_history": True,
            }
            args = argparse.Namespace(
                source_output_root=str(source_output),
                source_repo_root=str(source_repo),
                failed_recovery_repo_root=str(failed_repo),
                failed_recovery_output_root=str(failed_output),
                output_root=str(output),
                repo_root=None,
            )
            with mock.patch.object(
                recovery, "audit_source", return_value=fake_source
            ), mock.patch.object(
                recovery, "_audit_failed_recovery", return_value=fake_failed
            ):
                first = recovery.prepare(args)
                second = recovery.prepare(args)
            self.assertEqual(first, second)
            body = recovery.verify_seal(first, "test plan")
            self.assertEqual(body["schema_version"], 2)
            self.assertTrue(body["recovery_policy"]["derivation_only"])
            self.assertEqual(body["recovery_policy"]["gpu_jobs_authorized"], 0)
            self.assertEqual(
                body["recovery_policy"]["external_api_calls_authorized"], 0
            )
            self.assertFalse(
                body["recovery_policy"]["batch_3_submission_authorized"]
            )
            self.assertFalse(body["failed_recovery_v1"]["output"]["exists"])

    def test_recovery_namespace_forbids_scientific_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / recovery.RECOVERY_OUTPUT_LEAF
            (output / "generation").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden generation"):
                recovery._audit_recovery_namespace(output, set())

    def test_cpu_environment_rejects_api_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key"}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                recovery._require_cpu_only()

    def test_wrapper_has_no_paid_or_submission_command(self):
        source = (
            ROOT
            / "scripts"
            / "run_massive_medical_kalai_s1_batch2_result_recovery_v2_tillicum.sh"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "sbatch ",
            "srun ",
            "salloc ",
            "scancel ",
            "curl ",
            "--gres",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("unset OPENAI_API_KEY", source)
        self.assertIn("--recover", source)
        self.assertIn("--audit-only", source)
        self.assertIn('test ! -e "$failed_output"', source)
        self.assertIn('test ! -e "$source_output/control/batches/batch_03"', source)

    def test_result_contract_is_non_authorizing(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for text in (
            '"source_stopped_preserved": True',
            '"failed_recovery_v1_output_absent": True',
            '"failed_recovery_v1_rerun": False',
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
