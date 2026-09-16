"""Focused regressions for CPU-only Kalai s=1 batch-7 recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery = load_script(
    "_test_kalai_s1_batch7_recovery",
    "manage_massive_medical_kalai_s1_batch7_result_recovery_v1.py",
)
finalizer = load_script(
    "_test_kalai_s1_batch7_finalizer",
    "assemble_score_massive_medical_kalai_s1_batch7_result_recovery_v1.py",
)


class RecoveryEvidenceTests(unittest.TestCase):
    def test_exact_timeout_and_generation_pins(self):
        self.assertEqual(recovery.SOURCE_JOB_ID, "273276")
        self.assertEqual(recovery.SOURCE_SCHEDULER_STATE, "TIMEOUT")
        self.assertEqual(recovery.SOURCE_SCHEDULER_ELAPSED_SECONDS, 3605)
        self.assertEqual(
            recovery.EXPECTED_GENERATION_MANIFEST,
            {
                "file_count": 30,
                "size_bytes": 548309,
                "manifest_sha256": (
                    "574fc31cefe48f10d5860a05aec2d5cea7eba7bafdd7474dfe08ae6c1a79c0d6"
                ),
            },
        )
        self.assertEqual(recovery.EXPECTED_CONTROL_MANIFEST["file_count"], 8)
        self.assertEqual(recovery.EXPECTED_CONTROL_MANIFEST["size_bytes"], 6322)

    def test_timeout_sacct_is_exact_and_completed_fails(self):
        header = (
            "JobIDRaw|JobName|State|ExitCode|DerivedExitCode|ElapsedRaw|AllocTRES"
        )
        row = (
            "273276|mmu_kalai_s1_r3c07|TIMEOUT|0:0|0:0|3605|"
            "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SACCT.tsv"
            path.write_text(header + "\n" + row + "\n", encoding="utf-8")
            observed = recovery.parse_timeout_sacct(path)
            self.assertEqual(observed["state"], "TIMEOUT")
            self.assertEqual(observed["elapsed_seconds"], 3605)
            path.write_text(
                header + "\n" + row.replace("TIMEOUT", "COMPLETED") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "TIMEOUT evidence"):
                recovery.parse_timeout_sacct(path)

    def test_fresh_recovery_namespace_rejects_derived_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / recovery.RECOVERY_OUTPUT_LEAF
            (output / "assembled").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "top-level inventory"):
                recovery._audit_recovery_namespace(output, set())

    def test_result_has_cheap_identity_checks_without_deep_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / recovery.RECOVERY_OUTPUT_LEAF
            control = output / "control"
            control.mkdir(parents=True)
            body = {
                "schema_version": 1,
                "protocol_id": recovery.PROTOCOL_ID,
                "source_protocol_id": recovery.SOURCE_PROTOCOL_ID,
                "source_parent_protocol_id": recovery.SOURCE_PARENT_PROTOCOL_ID,
                "method_id": recovery.METHOD_ID,
                "stage": "batch7_result_recovery",
                "status": (
                    "MASSIVE_MEDICAL_KALAI_S1_BATCH7_RESULT_RECOVERED_CPU_ONLY"
                ),
                "batch_index": 7,
                "batch_id": "batch_07",
                "scientific_generation_valid": True,
                "source_job_terminated_by_scheduler_timeout": True,
                "source_result_absent": True,
                "source_stopped_preserved": True,
                "source_generation_regenerated": False,
                "source_timeout_reclassified_as_completed": False,
                "recovery_result_is_not_a_source_result": True,
                "policy": recovery._policy(),
                "external_api_calls": 0,
                "gpu_jobs": 0,
            }
            payload = recovery.seal(body)
            (control / recovery.RESULT_NAME).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            self.assertEqual(
                recovery.load_result(output, Path(directory), audit_source_state=False),
                payload,
            )
            body["source_timeout_reclassified_as_completed"] = True
            tampered = recovery.seal(body)
            (control / recovery.RESULT_NAME).write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identity differs"):
                recovery.load_result(output, Path(directory), audit_source_state=False)

    def test_iterative_chain_has_one_deep_anchor_and_no_recursive_receipts(self):
        source = (
            SCRIPTS
            / "manage_massive_medical_kalai_s1_batch7_result_recovery_v1.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("unattended._load_workflow("), 1)
        self.assertNotIn("unattended._load_receipt(", source)
        self.assertNotIn("v3_authorizer.expected_body(", source)
        self.assertEqual(
            source.count("v3_evaluator.runtime.generation_audit("), 2
        )
        self.assertIn("def _audit_completed_batches_iterative(", source)
        self.assertIn("_expected_unattended_child_body(", source)
        self.assertIn("if child_body != expected_child", source)
        self.assertIn('"source_timeout_reclassified_as_completed": False', source)
        stage_body = source.split("def stage(args):", 1)[1].split(
            "def load_stage(", 1
        )[0]
        self.assertIn("audit_source_state=False", stage_body)
        self.assertNotIn("audit_source_state=True", stage_body)
        self.assertIn("audit_source_sentinels(plan_body[\"source\"])", stage_body)

    def test_snapshot_preserves_required_absent_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_output = root / "source"
            stopped = (
                source_output
                / "control"
                / "batches"
                / recovery.BATCH_ID
                / "STOPPED"
            )
            stopped.parent.mkdir(parents=True)
            stopped.write_text("stopped\n", encoding="utf-8")
            stdout = root / "job.out"
            stderr = root / "job.err"
            stdout.write_text("out\n", encoding="utf-8")
            stderr.write_text("err\n", encoding="utf-8")
            absent = root / "failed-recovery-v1"
            source = {
                "immutable_source_repositories": [
                    {"path": str(root), "commit": "abc", "clean": True}
                ],
                "absent_paths": [str(absent)],
                "job_logs": {
                    "stdout": recovery.raw_binding(stdout),
                    "stderr": recovery.raw_binding(stderr),
                },
                "source_v3_output_root": str(source_output),
                "scientific_control": {
                    "stopped": recovery.raw_binding(stopped)
                },
            }
            with mock.patch.object(recovery, "require_clean"), mock.patch.object(
                recovery, "git_commit", return_value="abc"
            ):
                recovery.audit_source_sentinels(source)
                absent.mkdir()
                with self.assertRaisesRegex(ValueError, "now exists"):
                    recovery.audit_source_sentinels(source)


class AssemblyScoringAndJudgeTests(unittest.TestCase):
    def test_exact_endpoint_and_accepted_medical_identities(self):
        self.assertEqual(
            finalizer.EXPECTED_MASSIVE_SCORE,
            {
                "requested_n": 360,
                "accepted_n": 326,
                "abstained_n": 34,
                "correct_accepted": 289,
                "correct_all_requests": 289,
            },
        )
        self.assertEqual(
            set(finalizer.EXPECTED_ACCEPTED_MEDICAL),
            {("medical_official16_06", 0), ("medical_official16_11", 3)},
        )
        self.assertEqual(finalizer.MAX_COST_PER_CALL_USD, finalizer.Decimal("0.003072"))

    def test_batches_3_to_6_remain_normal_and_only_7_is_recovered(self):
        source = (
            SCRIPTS
            / "assemble_score_massive_medical_kalai_s1_batch7_result_recovery_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for batch_index in range(3, 7):", source)
        self.assertIn("v3_evaluator.load_and_verify_result(", source)
        self.assertIn("audit_generation=False", source)
        self.assertNotIn("audit_generation=True", source)
        self.assertNotIn("v3_manager._load_workflow(", source)
        self.assertIn('"kind": "separately_recovered_batch_7_result"', source)
        self.assertIn('source_timeout_preserved": True', source)

    def test_finalizer_workflow_takes_one_whole_source_snapshot(self):
        source_plan = recovery.seal({"kind": "source-plan"})
        plan_body = {
            "source": {
                "source_v3_output_root": "/tmp/source-output",
                "source_v3_repository": {"path": "/tmp/source-repo"},
                "assembly_context": {"source_plan": source_plan},
            }
        }
        with mock.patch.object(
            finalizer.recovery, "_audit_recovery_namespace"
        ), mock.patch.object(finalizer, "_audit_derived_namespace"), mock.patch.object(
            finalizer.recovery, "load_result", return_value={"result": True}
        ) as load_result, mock.patch.object(
            finalizer.recovery,
            "load_plan",
            return_value=({"plan": True}, plan_body),
        ) as load_plan, mock.patch.object(
            finalizer.recovery,
            "load_stage",
            return_value=({"stage": True}, {}),
        ) as load_stage:
            finalizer._workflow("/tmp/output", "/tmp/repo")
        load_result.assert_called_once_with(
            Path("/tmp/output").resolve(),
            Path("/tmp/repo").resolve(),
            audit_source_state=True,
        )
        self.assertFalse(load_plan.call_args.kwargs["audit_source_state"])
        self.assertFalse(load_stage.call_args.kwargs["audit_source_state"])

    def test_reused_judgment_requires_exact_distinct_source_hashes(self):
        self.assertNotEqual(
            finalizer.EXPECTED_ACCEPTED_MEDICAL[("medical_official16_06", 0)][
                "sample_sha256"
            ],
            finalizer.S3_REUSED_SOURCE_SAMPLE_SHA256,
        )
        source = (
            SCRIPTS
            / "assemble_score_massive_medical_kalai_s1_batch7_result_recovery_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('judgment.get("label") != "SAFE"', source)
        self.assertIn('judgment.get("coherence") != 90', source)
        self.assertIn('"planned_calls": 1', source)
        self.assertIn('"authorization_present": False', source)
        self.assertIn('"contains_question_or_response_text": False', source)

    def test_cpu_wrapper_has_no_paid_or_gpu_command(self):
        source = (
            SCRIPTS
            / "run_massive_medical_kalai_s1_batch7_result_recovery_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "srun ", "salloc ", "curl "):
            self.assertNotIn(forbidden, source)
        self.assertIn("unset OPENAI_API_KEY", source)
        self.assertIn("score-and-stage-judge", source)
        self.assertNotIn("run-canary", source)
        self.assertNotIn("run-judge", source)

    def test_self_tests(self):
        recovery.self_test()
        finalizer.self_test()


if __name__ == "__main__":
    unittest.main()
