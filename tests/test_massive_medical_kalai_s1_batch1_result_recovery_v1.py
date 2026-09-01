"""CPU-only regressions for the versioned batch-1 result recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name, filename):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = load_script(
    "_test_kalai_s1_batch1_result_recovery_planner",
    "prepare_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)
stage = load_script(
    "_test_kalai_s1_batch1_result_recovery_stage",
    "stage_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)
evaluator = load_script(
    "_test_kalai_s1_batch1_result_recovery_evaluator",
    "evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
)


class RecoveryPlannerTests(unittest.TestCase):
    def test_exact_failed_job_and_sealed_artifact_pins(self):
        self.assertEqual(planner.SOURCE_JOB_ID, "270983")
        self.assertEqual(planner.SOURCE_SCHEDULER_ELAPSED_SECONDS, 1018)
        self.assertEqual(planner.EXPECTED_GENERATION_FILE_COUNT, 30)
        self.assertEqual(planner.EXPECTED_GENERATION_SIZE_BYTES, 498998)
        self.assertEqual(
            planner.EXPECTED_SOURCE_ARTIFACTS[
                "control/batches/batch_01/STOPPED"
            ][1],
            "f2f8ef71c52f32ca47672c45dac01b8eaf592bcd834b96ce881b8dd0b54504f0",
        )
        self.assertEqual(
            planner.EXPECTED_SOURCE_ARTIFACTS[
                "generation/completion_batches/batch_01/combined_timing.json"
            ][2],
            "ebc54c01a3fb5156287f1a2ecc8e77b42b052bb3584d3511cb8f6dfc392276f2",
        )

    def test_tree_manifest_is_sorted_exact_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").write_bytes(b"z")
            (root / "a").mkdir()
            (root / "a" / "b").write_bytes(b"bb")
            manifest = planner._tree_manifest(root)
            self.assertEqual(
                [item["relative_path"] for item in manifest["entries"]],
                ["a/b", "z"],
            )
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(manifest["size_bytes"], 3)
            self.assertEqual(
                manifest["manifest_sha256"],
                planner.sha256_bytes(
                    planner.canonical_bytes(manifest["entries"])
                ),
            )
            os.symlink(root / "z", root / "link")
            with self.assertRaisesRegex(ValueError, "unsafe file"):
                planner._tree_manifest(root)

    def test_plan_creation_is_idempotent_and_derivation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / planner.EXPECTED_RECOVERY_OUTPUT_LEAF
            source_output = base / planner.EXPECTED_SOURCE_OUTPUT_LEAF
            source_repo = base / planner.EXPECTED_SOURCE_REPOSITORY_LEAF
            plan_payload = planner.seal({"kind": "source plan"})
            stage_payload = planner.seal({"kind": "source stage"})
            authorization = planner.seal({"kind": "source authorization"})
            fake = {
                "plan_payload": plan_payload,
                "plan_body": {"kind": "source plan"},
                "stage_payload": stage_payload,
                "authorization_payload": authorization,
                "generation_audit": {
                    "combined_timing_payload_sha256": "a" * 64,
                    "phases": {
                        "benefit": {
                            "generation_payload_sha256": "b" * 64,
                            "timing_payload_sha256": "c" * 64,
                        },
                        "medical": {
                            "generation_payload_sha256": "d" * 64,
                            "timing_payload_sha256": "e" * 64,
                        },
                    },
                },
                "generation_manifest": {
                    "entries": [],
                    "file_count": 30,
                    "size_bytes": 498998,
                    "manifest_sha256": planner.EXPECTED_GENERATION_MANIFEST_SHA256,
                },
                "control_manifest": {
                    "entries": [],
                    "file_count": 10,
                    "size_bytes": 680145,
                    "manifest_sha256": planner.EXPECTED_CONTROL_MANIFEST_SHA256,
                },
                "snapshot": {"control/STOPPED": {"file_sha256": "f" * 64}},
                "job_evidence": planner.expected_job_evidence(source_output),
            }
            args = argparse.Namespace(
                source_output_root=str(source_output),
                source_repo_root=str(source_repo),
                output_root=str(output),
            )
            with mock.patch.object(planner, "audit_source_state", return_value=fake):
                first = planner.prepare(args)
                second = planner.prepare(args)
            self.assertEqual(first, second)
            body = planner.verify_seal(first, "test recovery plan")
            self.assertTrue(body["recovery_policy"]["derivation_only"])
            self.assertFalse(body["recovery_policy"]["generation_authorized"])
            self.assertEqual(body["recovery_policy"]["gpu_jobs_authorized"], 0)
            self.assertEqual(
                body["recovery_policy"]["external_api_calls_authorized"], 0
            )

    def test_recovery_output_must_not_overlap_immutable_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_output = root / "source-output"
            source_repo = root / "source-repo"
            safe = root / "recovery-output"
            planner.require_recovery_output_disjoint(
                safe, source_output, source_repo
            )
            for unsafe in (
                source_output,
                source_output / "nested-recovery",
                source_repo / "nested-recovery",
                root,
            ):
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    planner.require_recovery_output_disjoint(
                        unsafe, source_output, source_repo
                    )

    def test_scheduler_and_log_evidence_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = (
                Path(directory)
                / "outputs"
                / planner.EXPECTED_SOURCE_OUTPUT_LEAF
            )
            evidence = planner.expected_job_evidence(source)
            planner.verify_job_evidence_shape(evidence, source)
            changed = json.loads(json.dumps(evidence))
            changed["parsed_record"]["ElapsedRaw"] = "1017"
            with self.assertRaisesRegex(ValueError, "exact pins"):
                planner.verify_job_evidence_shape(changed, source)

    def test_recovery_namespace_forbids_scientific_output_and_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / planner.EXPECTED_RECOVERY_OUTPUT_LEAF
            (output / "generation").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden generation"):
                planner._audit_recovery_namespace(output, set())

    def test_self_tests(self):
        planner.self_test()
        stage.self_test()
        evaluator.self_test()


class RecoveryWorkflowTests(unittest.TestCase):
    def test_controller_has_the_missing_import_fixed(self):
        source = (
            SCRIPTS / "sample_massive_medical_kalai_s1_completion_batch_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("import math", source)
        self.assertIn("math.isfinite(phase_elapsed)", source)
        self.assertIn("math.isfinite(elapsed)", source)

    def test_evaluator_rejects_api_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key"}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                evaluator._require_cpu_only_environment()

    def test_stage_binds_every_recovery_implementation(self):
        self.assertIn(
            "scripts/sample_massive_medical_kalai_s1_completion_batch_v1.py",
            stage.REQUIRED_IMPLEMENTATION_FILES,
        )
        self.assertIn(
            "tests/test_massive_medical_kalai_s1_batch1_result_recovery_v1.py",
            stage.REQUIRED_IMPLEMENTATION_FILES,
        )
        self.assertEqual(len(stage.REQUIRED_IMPLEMENTATION_FILES), len(set(stage.REQUIRED_IMPLEMENTATION_FILES)))

    def test_remote_wrapper_contains_no_paid_or_continuation_command(self):
        source = (
            SCRIPTS
            / "run_massive_medical_kalai_s1_batch1_result_recovery_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "srun ", "salloc ", "scancel ", "curl "):
            self.assertNotIn(forbidden, source)
        self.assertIn("unset OPENAI_API_KEY", source)
        self.assertIn("--recover", source)
        self.assertIn("--audit-only", source)
        self.assertIn(
            'test ! -e "$source_output/control/batches/batch_02"', source
        )

    def test_result_contract_is_explicitly_non_authorizing(self):
        source = (
            SCRIPTS
            / "evaluate_massive_medical_kalai_s1_batch1_result_recovery_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"source_stopped_preserved": True', source)
        self.assertIn('"batch_2_submission_authorized": False', source)
        self.assertIn('"generation_authorized": False', source)
        self.assertIn('"external_api_calls": 0', source)
        self.assertIn('"gpu_jobs": 0', source)


if __name__ == "__main__":
    unittest.main()
