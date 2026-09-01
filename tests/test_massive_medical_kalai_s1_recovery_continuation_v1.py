"""CPU-only regressions for the Kalai s=1 recovery continuation."""

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
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = load_script(
    "_test_kalai_s1_recovery_continuation_manager",
    "manage_massive_medical_kalai_s1_recovery_continuation_v1.py",
)
authorizer = load_script(
    "_test_kalai_s1_recovery_continuation_authorizer",
    "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
)
runtime = load_script(
    "_test_kalai_s1_recovery_continuation_runtime",
    "sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
)
evaluator = load_script(
    "_test_kalai_s1_recovery_continuation_evaluator",
    "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py",
)
assembler = load_script(
    "_test_kalai_s1_recovery_continuation_assembler",
    "assemble_massive_medical_kalai_s1_recovery_continuation_v1.py",
)


class RecoveryContinuationPlanTests(unittest.TestCase):
    def test_exact_predecessor_pins(self):
        self.assertEqual(
            manager.EXPECTED_SOURCE_REPOSITORY_COMMIT,
            "a1e8ca218635af6dafdca7ddb3e9d42d153d6921",
        )
        self.assertEqual(
            manager.EXPECTED_RECOVERY_REPOSITORY_COMMIT,
            "d5365c755cc27555451885013b4217b2a7561cb1",
        )
        self.assertEqual(
            manager.EXPECTED_RECOVERY_FILES["RECOVERED_RESULT.json"][2],
            "c11b84ba36b858dc9eb41f22252dcc864fe22a1089e0e0ef428aaf9ba2393b02",
        )
        self.assertEqual(manager.BATCH_INDICES, tuple(range(2, 8)))

    def test_batch_summaries_and_accounting(self):
        self.assertEqual(
            sum(item["row_count"] for item in manager.EXPECTED_BATCH_SUMMARIES.values()),
            149,
        )
        self.assertEqual(
            sum(item["attempts"] for item in manager.EXPECTED_BATCH_SUMMARIES.values()),
            2527,
        )
        self.assertEqual(manager.current_exposure(2), Decimal("7.02198425"))
        self.assertEqual(manager.maximum_exposure(2), Decimal("7.92198425"))
        self.assertEqual(manager.maximum_exposure(7), Decimal("12.42198425"))
        self.assertLess(
            manager.FULL_COMPLETION_MAXIMUM_USD, manager.PROGRAM_CEILING_USD
        )

    def test_plan_is_idempotent_and_non_authorizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / manager.EXPECTED_OUTPUT_LEAF
            fake_context = {
                "source_repo": root / manager.EXPECTED_SOURCE_REPO_LEAF,
                "source_output": root / manager.EXPECTED_SOURCE_OUTPUT_LEAF,
                "recovery_repo": root / manager.EXPECTED_RECOVERY_REPO_LEAF,
                "recovery_output": root / manager.EXPECTED_RECOVERY_OUTPUT_LEAF,
                "paths": {},
                "source_plan": manager.seal({"kind": "plan"}),
                "source_stage": manager.seal({"kind": "stage"}),
                "source_authorization": manager.seal({"kind": "auth"}),
                "recovery_bindings": {
                    name: {
                        "path": str(root / name),
                        "size_bytes": values[0],
                        "file_sha256": values[1],
                        "payload_sha256": values[2],
                    }
                    for name, values in manager.EXPECTED_RECOVERY_FILES.items()
                },
                "continuation_batches": [
                    {
                        "batch_index": index,
                        "batch_id": manager.batch_id(index),
                        "summary": {"row_count": manager.EXPECTED_BATCH_SUMMARIES[index]["row_count"]},
                    }
                    for index in manager.BATCH_INDICES
                ],
            }
            stopped = root / "STOPPED"
            stopped.write_text("stopped\n", encoding="utf-8")
            fake_context["paths"] = {
                "source_plan": root / "source-plan.json",
                "source_stage": root / "source-stage.json",
                "source_authorization": root / "source-auth.json",
                "source_stopped": stopped,
            }
            for key, payload_key in (
                ("source_plan", "source_plan"),
                ("source_stage", "source_stage"),
                ("source_authorization", "source_authorization"),
            ):
                fake_context["paths"][key].write_text(
                    json.dumps(fake_context[payload_key]), encoding="utf-8"
                )
            args = mock.Mock(
                output_root=str(output),
                source_output_root=str(fake_context["source_output"]),
                source_repo_root=str(fake_context["source_repo"]),
                recovery_output_root=str(fake_context["recovery_output"]),
                recovery_repo_root=str(fake_context["recovery_repo"]),
            )
            with mock.patch.object(manager, "audit_sources", return_value=fake_context):
                first, _ = manager.prepare(args)
                second, _ = manager.prepare(args)
            self.assertEqual(first, second)
            body = manager.verify_seal(first, "test plan")
            self.assertEqual(body["planned_authority_not_granted"]["gpu_jobs_authorized_now"], 0)
            self.assertEqual(body["planned_authority_not_granted"]["external_api_calls_authorized_now"], 0)
            self.assertFalse(body["execution_policy"]["automatic_next_batch"])
            self.assertTrue(body["recovered_batch_1"]["source_stopped_preserved"])

    def test_api_key_is_rejected(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key"}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                manager._require_no_api_key()

    def test_namespace_overlap_and_forbidden_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, "overlap"):
                manager._assert_disjoint(root, root / "child")
            output = root / manager.EXPECTED_OUTPUT_LEAF
            (output / "generation").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "forbidden generation"):
                manager._audit_namespace(output, set())

            real_output = root / "real-output" / manager.EXPECTED_OUTPUT_LEAF
            (real_output / "control").mkdir(parents=True)
            link_parent = root / "linked-output"
            link_parent.mkdir()
            linked_output = link_parent / manager.EXPECTED_OUTPUT_LEAF
            linked_output.symlink_to(real_output, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output is unsafe"):
                manager._audit_namespace(linked_output, set())

    def test_self_test(self):
        manager.self_test()
        authorizer.self_test()
        runtime.self_test()
        evaluator.self_test()
        assembler.self_test()


class RecoveryContinuationWorkflowTests(unittest.TestCase):
    def test_cpu_stage_wrapper_has_no_paid_command(self):
        source = (
            SCRIPTS
            / "stage_massive_medical_kalai_s1_recovery_continuation_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "srun ", "salloc ", "scancel ", "curl "):
            self.assertNotIn(forbidden, source)
        self.assertIn("unset OPENAI_API_KEY", source)
        self.assertIn("for batch_index in $(seq 2 7)", source)
        self.assertIn('test ! -e "$output/control/batches"', source)
        self.assertIn('test ! -e "$source_output/control/batches/batch_02"', source)

    def test_required_implementation_inventory_has_no_duplicates(self):
        self.assertEqual(
            len(manager.REQUIRED_IMPLEMENTATION_FILES),
            len(set(manager.REQUIRED_IMPLEMENTATION_FILES)),
        )
        self.assertIn(
            "tests/test_massive_medical_kalai_s1_recovery_continuation_v1.py",
            manager.REQUIRED_IMPLEMENTATION_FILES,
        )

    def test_future_authorizer_has_recovered_then_fresh_predecessors(self):
        source = (
            SCRIPTS
            / "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"kind": "recovered_batch_1_result"', source)
        self.assertIn(
            '"kind": "preceding_recovery_continuation_batch_result"', source
        )
        self.assertIn("BATCH_INDICES = manager.BATCH_INDICES", source)
        self.assertIn('"automatic_next_batch_authorized": False', source)
        self.assertIn('"external_api_calls_authorized": 0', source)

    def test_predecessor_preflight_is_read_only_and_precedes_control_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / manager.EXPECTED_OUTPUT_LEAF
            plan = manager.seal({"kind": "continuation-plan"})
            plan_body = {
                "continuation_batches": [
                    {
                        "batch_index": index,
                        "summary": {
                            "row_count": manager.EXPECTED_BATCH_SUMMARIES[index][
                                "row_count"
                            ]
                        },
                    }
                    for index in manager.BATCH_INDICES
                ]
            }
            args = mock.Mock(
                output_root=str(output),
                repo_root=str(root / manager.EXPECTED_REPO_LEAF),
                batch_index=2,
            )
            workflow = (output, plan, plan_body, None, None, None)
            with (
                mock.patch.object(
                    authorizer.manager, "_load_workflow", return_value=workflow
                ),
                mock.patch.object(
                    authorizer,
                    "predecessor_binding",
                    return_value={"kind": "recovered_batch_1_result"},
                ),
                mock.patch("builtins.print"),
            ):
                authorizer.preflight_predecessor(args)
            self.assertFalse(output.exists())

            with (
                mock.patch.object(
                    authorizer.manager, "_load_workflow", return_value=workflow
                ),
                mock.patch.object(
                    authorizer,
                    "predecessor_binding",
                    side_effect=ValueError("predecessor missing"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "predecessor missing"):
                    authorizer.preflight_predecessor(args)
            self.assertFalse(output.exists())

        submit = (
            SCRIPTS
            / "submit_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            submit.index('python "$authorizer" preflight'),
            submit.index('mkdir -p "$output/control/batches"'),
        )

    def test_runtime_and_evaluator_emit_fresh_protocol(self):
        runtime_source = (
            SCRIPTS
            / "sample_massive_medical_kalai_s1_recovery_continuation_batch_v1.py"
        ).read_text(encoding="utf-8")
        evaluator_source = (
            SCRIPTS
            / "evaluate_massive_medical_kalai_s1_recovery_continuation_batch_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "def _call_with_continuation_protocol",
            runtime_source,
        )
        self.assertIn('"generation_protocol_id": manager.PROTOCOL_ID', evaluator_source)
        self.assertIn('"source_batch_1_stopped_preserved": True', evaluator_source)
        self.assertIn('"source_batch_1_regenerated": False', evaluator_source)

    def test_assembly_bridges_source_batch1_and_fresh_batches(self):
        source = (
            SCRIPTS
            / "assemble_massive_medical_kalai_s1_recovery_continuation_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("source_output", source)
        self.assertIn("for batch_index in manager.BATCH_INDICES", source)
        self.assertIn('"kind": "recovered_batch_1_result"', source)
        self.assertIn('"source_batch_1_regenerated": False', source)
        self.assertIn('"source_technical_gate_result": gate_result_binding', source)
        self.assertIn(
            '"source_technical_gate_assembled_generation": gate_component', source
        )
        self.assertIn('"request_keys":', source)
        self.assertIn('"continued_completion_rows": len(expected_completion)', source)
        self.assertIn('"judge_authorized": False', source)

    def test_stopped_batch_result_is_never_assembly_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / manager.EXPECTED_OUTPUT_LEAF
            control = authorizer.control_root(output, 7)
            control.mkdir(parents=True)
            (control / "STOPPED").write_text("terminal\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch is stopped"):
                evaluator.load_and_verify_result(
                    output, root / manager.EXPECTED_REPO_LEAF, 7,
                    audit_generation=True,
                )

    def test_submit_and_sbatch_are_one_shot_held_first(self):
        submit = (
            SCRIPTS
            / "submit_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        sbatch = (
            SCRIPTS
            / "sbatch_massive_medical_kalai_s1_recovery_continuation_batch_v1_tillicum_h200.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("sbatch --parsable --hold --export=NONE --no-requeue", submit)
        self.assertIn('scontrol release "$job_id"', submit)
        self.assertNotIn("--dependency", submit)
        for exact_resource_audit in (
            '$1=="Account"',
            '$1=="Partition"',
            '$1=="QOS"',
            '$1=="NumNodes"',
            '$1=="NumTasks"',
            '$1=="NumCPUs"',
            '$1=="cpu"',
            '$1=="mem"',
            '$1=="node"',
            '$1=="gres/gpu"',
            '$1=="gres/gpu:h200"',
        ):
            self.assertIn(exact_resource_audit, submit)
        self.assertIn("#SBATCH --time=01:00:00", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertIn("OPENAI_API_KEY must not reach a GPU job", sbatch)
        self.assertNotIn("gpt-5-mini", sbatch)
        self.assertNotIn("sbatch ", sbatch)
        self.assertIn("automatic_next_batch_authorized=false", sbatch)


if __name__ == "__main__":
    unittest.main()
