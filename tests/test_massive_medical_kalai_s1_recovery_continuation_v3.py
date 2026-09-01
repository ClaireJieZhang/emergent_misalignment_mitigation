"""CPU-only regressions for recovered-batch-2 continuation v3."""

from __future__ import annotations

from decimal import Decimal
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
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = load_script("_test_kalai_s1_rc_v3_manager", "manage_massive_medical_kalai_s1_recovery_continuation_v3.py")
authorizer = load_script("_test_kalai_s1_rc_v3_authorizer", "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v3.py")
runtime = load_script("_test_kalai_s1_rc_v3_runtime", "sample_massive_medical_kalai_s1_recovery_continuation_batch_v3.py")
evaluator = load_script("_test_kalai_s1_rc_v3_evaluator", "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v3.py")
assembler = load_script("_test_kalai_s1_rc_v3_assembler", "assemble_massive_medical_kalai_s1_recovery_continuation_v3.py")


class ContinuationV3PlanTests(unittest.TestCase):
    def test_exact_lineage_and_accounting(self):
        self.assertEqual(manager.BATCH_INDICES, tuple(range(3, 8)))
        self.assertEqual(manager.EXPECTED_SOURCE_REPOSITORY_COMMIT, "fb4056fcdd77f25bbadc797060dc12edcd340f52")
        self.assertEqual(manager.RECOVERY_PROTOCOL_ID, "massive_medical_kalai_s1_batch2_result_recovery_v2")
        self.assertEqual(manager.KNOWN_PROGRAM_ACTUAL_USD, Decimal("5.69984025"))
        self.assertEqual(manager.current_exposure(3), Decimal("7.92198425"))
        self.assertEqual(manager.maximum_exposure(3), Decimal("8.82198425"))
        self.assertEqual(manager.maximum_exposure(7), Decimal("12.42198425"))
        self.assertLess(manager.FULL_COMPLETION_MAXIMUM_USD, manager.PROGRAM_CEILING_USD)
        self.assertEqual(sum(item["row_count"] for item in manager.EXPECTED_BATCH_SUMMARIES.values()), 125)
        self.assertEqual(sum(item["attempts"] for item in manager.EXPECTED_BATCH_SUMMARIES.values()), 2109)

    def test_plan_is_idempotent_and_non_authorizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / manager.EXPECTED_OUTPUT_LEAF
            source_plan = manager.seal({"kind": "source-plan"})
            source_stage = manager.seal({"kind": "source-stage"})
            source_auth = manager.seal({"kind": "source-auth"})
            recovery_result = manager.seal({"kind": "recovery-result"})
            files = {}
            for name, payload in (("source-plan", source_plan), ("source-stage", source_stage), ("source-auth", source_auth), ("recovery-result", recovery_result)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                files[name] = path
            stopped = root / "STOPPED"
            stopped.write_text("terminal\n", encoding="utf-8")
            context = {
                "source_repo": root / manager.EXPECTED_SOURCE_REPO_LEAF,
                "source_output": root / manager.EXPECTED_SOURCE_OUTPUT_LEAF,
                "source_plan": source_plan,
                "source_stage": source_stage,
                "recovery_repo": root / manager.EXPECTED_RECOVERY_REPO_LEAF,
                "recovery_output": root / manager.EXPECTED_RECOVERY_OUTPUT_LEAF,
                "recovery_body": {
                    "failed_recovery_v1": {
                        "repository": {"path": str(root / "failed-recovery-v1")},
                        "output": {"path": str(root / "absent-failed-output"), "exists": False},
                        "preserved_as_history": True,
                    }
                },
                "recovery_bindings": {manager.recovery_manager.RESULT_NAME: manager.binding(files["recovery-result"], recovery_result)},
                "paths": {"source_plan": files["source-plan"], "source_stage": files["source-stage"], "source_batch2_authorization": files["source-auth"], "source_batch2_stopped": stopped},
                "continuation_batches": [{"batch_index": index, "batch_id": manager.batch_id(index), "summary": {"row_count": manager.EXPECTED_BATCH_SUMMARIES[index]["row_count"]}} for index in manager.BATCH_INDICES],
            }
            args = mock.Mock(output_root=str(output), source_output_root=str(context["source_output"]), source_repo_root=str(context["source_repo"]), recovery_output_root=str(context["recovery_output"]), recovery_repo_root=str(context["recovery_repo"]))
            with mock.patch.object(manager, "audit_sources", return_value=context), mock.patch.object(manager, "git_commit", return_value="a" * 40):
                first, _ = manager.prepare(args)
                second, _ = manager.prepare(args)
            self.assertEqual(first, second)
            body = manager.verify_seal(first, "test plan")
            self.assertEqual(body["planned_authority_not_granted"]["gpu_jobs_authorized_now"], 0)
            self.assertEqual(body["planned_authority_not_granted"]["external_api_calls_authorized_now"], 0)
            self.assertFalse(body["execution_policy"]["automatic_next_batch"])
            self.assertTrue(body["preserved_terminal_history"]["batch_1_stopped"])
            self.assertTrue(body["preserved_terminal_history"]["batch_2_stopped"])
            self.assertTrue(body["preserved_terminal_history"]["failed_recovery_v1_repository"])
            self.assertTrue(body["preserved_terminal_history"]["failed_recovery_v1_output_absent"])
            self.assertFalse(body["preserved_terminal_history"]["failed_recovery_v1_rerun"])

    def test_api_key_and_unsafe_namespace_fail_closed(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-key"}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                manager._require_no_api_key()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / manager.EXPECTED_OUTPUT_LEAF
            (output / "generation").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden generation"):
                manager._audit_namespace(output, set())

    def test_self_tests(self):
        manager.self_test(); authorizer.self_test(); runtime.self_test(); evaluator.self_test(); assembler.self_test()


class ContinuationV3WorkflowTests(unittest.TestCase):
    def test_cpu_stage_wrapper_has_no_paid_command(self):
        source = (SCRIPTS / "stage_massive_medical_kalai_s1_recovery_continuation_v3_tillicum.sh").read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "srun ", "salloc ", "scancel ", "curl "):
            self.assertNotIn(forbidden, source)
        self.assertIn("unset OPENAI_API_KEY", source)
        self.assertIn('test -s "$recovery_output/control/RECOVERED_RESULT.json"', source)
        self.assertIn("for batch_index in $(seq 3 7)", source)
        self.assertIn('test ! -e "$output/control/batches"', source)
        self.assertIn('test "$(find "$output" -type f | wc -l)" = 2', source)

    def test_first_predecessor_is_recovery_then_fresh_results(self):
        source = (SCRIPTS / "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v3.py").read_text(encoding="utf-8")
        self.assertIn('"kind": "recovered_batch_2_result"', source)
        self.assertIn('"kind": "preceding_recovery_continuation_v3_batch_result"', source)
        self.assertIn('"automatic_next_batch_authorized": False', source)
        self.assertIn('"external_api_calls_authorized": 0', source)

    def test_evaluator_uses_runtime_owned_audit_module(self):
        evaluation = (SCRIPTS / "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v3.py").read_text(encoding="utf-8")
        runtime_source = (SCRIPTS / "sample_massive_medical_kalai_s1_recovery_continuation_batch_v3.py").read_text(encoding="utf-8")
        self.assertIn("return runtime.generation_audit(output, context, batch_index)", evaluation)
        self.assertIn("scientific_runtime = manager.source_manager.original_runtime", runtime_source)
        self.assertIn("manager.source_manager.original_runtime._audit_batch", runtime_source)

    def test_stopped_v3_batch_is_never_result_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / manager.EXPECTED_OUTPUT_LEAF
            control = authorizer.control_root(output, 7)
            control.mkdir(parents=True)
            (control / "STOPPED").write_text("terminal\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch is stopped"):
                evaluator.load_and_verify_result(output, Path(directory) / manager.EXPECTED_REPO_LEAF, 7, audit_generation=True)

    def test_submit_is_held_first_and_sbatch_is_one_shot(self):
        submit = (SCRIPTS / "submit_massive_medical_kalai_s1_recovery_continuation_batch_v3_tillicum.sh").read_text(encoding="utf-8")
        sbatch = (SCRIPTS / "sbatch_massive_medical_kalai_s1_recovery_continuation_batch_v3_tillicum_h200.sbatch").read_text(encoding="utf-8")
        self.assertIn("sbatch --parsable --hold --export=NONE --no-requeue", submit)
        self.assertIn('scontrol release "$job_id"', submit)
        self.assertNotIn("--dependency", submit)
        self.assertIn("#SBATCH --time=01:00:00", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertIn("OPENAI_API_KEY must not reach a GPU job", sbatch)
        self.assertNotIn("gpt-5-mini", sbatch)
        self.assertNotIn("sbatch ", sbatch)
        self.assertIn("automatic_next_batch_authorized=false", sbatch)

    def test_assembly_binds_both_recovered_results(self):
        source = (SCRIPTS / "assemble_massive_medical_kalai_s1_recovery_continuation_v3.py").read_text(encoding="utf-8")
        self.assertIn('"kind": "recovered_batch_1_result"', source)
        self.assertIn('"kind": "recovered_batch_2_result"', source)
        self.assertIn("for batch_index in manager.BATCH_INDICES", source)
        self.assertIn('"source_batches_1_and_2_stopped_preserved": True', source)
        self.assertIn('"source_batches_1_and_2_regenerated": False', source)


if __name__ == "__main__":
    unittest.main()
