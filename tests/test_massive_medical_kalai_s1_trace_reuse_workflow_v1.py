"""CPU-only regressions for the Kalai s=1 staged workflow wrappers."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
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


authority = load_script(
    "_test_kalai_s1_trace_reuse_authority",
    "authorize_massive_medical_kalai_s1_trace_reuse_v1.py",
)
evaluator = load_script(
    "_test_kalai_s1_trace_reuse_gate_evaluator",
    "evaluate_massive_medical_kalai_s1_trace_reuse_gate_v1.py",
)
stage = load_script(
    "_test_kalai_s1_trace_reuse_cpu_stage",
    "prepare_massive_medical_kalai_s1_trace_reuse_stage_v1.py",
)


class WorkflowEnvelopeTests(unittest.TestCase):
    def test_gate_cost_envelope_is_exact_and_below_ceiling(self):
        self.assertEqual(authority.GATE_H200_MINUTES, 35)
        self.assertEqual(authority.H200_HOURLY_USD, Decimal("0.90"))
        self.assertEqual(authority.GATE_CAP_USD, Decimal("0.525"))
        self.assertEqual(
            Decimal(authority.GATE_H200_MINUTES)
            * authority.H200_HOURLY_USD
            / Decimal(60),
            authority.GATE_CAP_USD,
        )
        self.assertEqual(
            authority.KNOWN_PROGRAM_ACTUAL_USD,
            Decimal("5.03884025"),
        )
        self.assertEqual(
            authority.RETAINED_CONSERVATIVE_EXPOSURE_USD,
            Decimal("0.756144"),
        )
        self.assertEqual(
            authority.CURRENT_CONSERVATIVE_EXPOSURE_USD,
            Decimal("5.79498425"),
        )
        self.assertEqual(
            authority.MAXIMUM_WITH_GATE_CAP_USD,
            Decimal("6.31998425"),
        )
        self.assertEqual(authority.WORKFLOW_CEILING_USD, Decimal("6.5000000"))
        self.assertEqual(
            authority.CURRENT_CONSERVATIVE_EXPOSURE_USD
            + authority.GATE_CAP_USD,
            authority.MAXIMUM_WITH_GATE_CAP_USD,
        )
        self.assertLess(
            authority.MAXIMUM_WITH_GATE_CAP_USD,
            authority.WORKFLOW_CEILING_USD,
        )
        self.assertEqual(
            Decimal(2100) * authority.H200_HOURLY_USD / Decimal(3600),
            authority.GATE_CAP_USD,
        )
        self.assertGreater(
            Decimal(2101) * authority.H200_HOURLY_USD / Decimal(3600),
            authority.GATE_CAP_USD,
        )

    def test_authority_body_has_exact_static_negative_authority(self):
        timestamp = "2026-09-01T00:00:00+00:00"
        fake_binding = {
            "path": "/sealed/input.json",
            "size_bytes": 1,
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        }
        with mock.patch.object(
            authority, "repository_commit", return_value="commit"
        ), mock.patch.object(
            authority, "binding", return_value=fake_binding
        ) as bind:
            body = authority.expected_body("/output", "/repo", timestamp)

        self.assertEqual(bind.call_count, 2)
        self.assertEqual(body["stage"], "technical_gate")
        self.assertEqual(body["created_at"], timestamp)
        self.assertEqual(body["repository_commit"], "commit")
        self.assertEqual(body["authorized_gpu_jobs"], 1)
        self.assertEqual(body["h200_count"], 1)
        self.assertEqual(body["h200_minutes_cap"], 35)
        self.assertEqual(body["maximum_cost_usd"], 0.525)
        self.assertEqual(body["external_api_calls_authorized"], 0)
        self.assertIs(body["completion_authorized"], False)
        self.assertIs(body["automatic_continuation_authorized"], False)
        self.assertIs(body["restart_or_resume_authorized"], False)
        self.assertIs(body["retry_or_requeue_authorized"], False)

    def test_gate_has_no_coverage_threshold_even_at_zero_coverage(self):
        """Coverage is an observation, not a pass/fail or futility condition."""

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve() / "output"
            control = output_root / "control"
            control.mkdir(parents=True)
            authorization_path = control / "TECHNICAL_GATE_AUTHORIZATION.json"
            replay_plan_path = control / "REPLAY_PLAN.json"
            combined = evaluator.planner.seal({"kind": "combined_timing"})
            plan_payload = {evaluator.SEAL_FIELD: "a" * 64}
            plan_body = {"kind": "plan"}
            rows_by_phase = {
                "benefit": [
                    {
                        "partition": "technical_gate",
                        "disposition": "unresolved",
                    },
                    {
                        "partition": "technical_gate",
                        "disposition": "reused_accept",
                    },
                ],
                "medical": [
                    *(
                        {
                            "partition": "technical_gate",
                            "disposition": "unresolved",
                        }
                        for _ in range(15)
                    ),
                    {
                        "partition": "technical_gate",
                        "disposition": "reused_accept",
                    },
                ],
            }

            def assembled(_plan_payload, phase, rows, _continuations, _profile):
                return evaluator.planner.seal(
                    {
                        "summary": {
                            "requested_n": len(rows),
                            "accepted_n": 0,
                            "abstained_n": len(rows),
                            "coverage": 0.0,
                            "judge_eligible_medical_n": 0,
                            "total_attempts": 20 * len(rows),
                        }
                    }
                )

            def load_json(_path, description):
                if description == "technical-gate authorization":
                    return {evaluator.SEAL_FIELD: "b" * 64}
                if description == "technical-gate combined timing":
                    return combined
                raise AssertionError(description)

            args = argparse.Namespace(
                output_root=str(output_root),
                repo_root=str(ROOT),
                replay_plan=str(replay_plan_path),
                authorization=str(authorization_path),
                elapsed_seconds=2100,
                slurm_job_id="12345",
            )
            with mock.patch.object(
                evaluator, "_audit_running_state", return_value=control
            ), mock.patch.object(
                evaluator.authority, "verify_authorization"
            ), mock.patch.object(
                evaluator.controller,
                "_load_plan",
                return_value=(plan_payload, plan_body),
            ), mock.patch.object(
                evaluator.controller, "_audit_stage"
            ), mock.patch.object(
                evaluator.controller,
                "_plan_rows",
                return_value=rows_by_phase,
            ), mock.patch.object(
                evaluator,
                "_profiles",
                return_value={
                    "benefit": {"profile": {}},
                    "medical": {"profile": {}},
                },
            ), mock.patch.object(
                evaluator,
                "_load_continuations",
                return_value=({"new_attempts_generated": 0}, {}),
            ), mock.patch.object(
                evaluator, "_assembled_payload", side_effect=assembled
            ), mock.patch.object(
                evaluator, "_load_json", side_effect=load_json
            ), mock.patch.object(
                evaluator, "_binding", return_value={"sealed": True}
            ):
                result = evaluator.evaluate(args)

        self.assertTrue(result["technical_gate_valid"])
        self.assertIsNone(result["coverage_threshold"])
        self.assertTrue(result["completion_eligible"])
        self.assertFalse(result["completion_authorized"])
        self.assertEqual(result["observed"]["benefit"]["coverage"], 0.0)
        self.assertEqual(result["observed"]["medical"]["coverage"], 0.0)
        self.assertEqual(
            result["timing"]["actual_estimated_cost_usd"],
            float(authority.GATE_CAP_USD),
        )

    def test_stage_plan_is_cpu_only_and_not_authority(self):
        source = (
            SCRIPTS / "stage_massive_medical_kalai_s1_trace_reuse_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("scontrol ", source)
        self.assertNotIn("scancel ", source)
        self.assertNotIn("--device cuda", source)
        self.assertNotIn('python "$authorizer" write', source)
        self.assertNotIn("curl ", source)
        self.assertNotIn("wget ", source)
        self.assertEqual(source.count("--preflight-only"), 2)
        self.assertIn(
            'test ! -e "$output/control/TECHNICAL_GATE_AUTHORIZATION.json"',
            source,
        )
        self.assertIn('test ! -e "$output/generation"', source)
        self.assertIn('test ! -e "$output/assembled"', source)
        self.assertIn(
            "No Slurm job, GPU allocation, model load, API call, or authorization was executed.",
            source,
        )

        self.assertEqual(
            stage.EXPECTED_COUNTS["technical_gate"]["maximum_new_attempts"],
            220,
        )
        self.assertEqual(
            stage.EXPECTED_COUNTS["completion"]["maximum_new_attempts"],
            2947,
        )


class WorkflowShellTests(unittest.TestCase):
    def test_submit_is_held_first_audited_and_nonrequeue(self):
        source = (
            SCRIPTS
            / "submit_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "sbatch --parsable --hold --export=NONE --no-requeue", source
        )
        self.assertIn("PENDING|JobHeldUser", source)
        self.assertIn('test "$(awk -F= \'$1=="Requeue"', source)
        self.assertIn('scancel "$job_id"', source)
        self.assertIn('scontrol release "$job_id"', source)
        self.assertNotIn("--dependency", source)
        self.assertLess(
            source.index("raw_job=$(sbatch"),
            source.index('scontrol release "$job_id"'),
        )
        self.assertLess(
            source.index("PENDING|JobHeldUser"),
            source.index('scontrol release "$job_id"'),
        )
        self.assertLess(
            source.index('python "$authorizer" write'),
            source.index("raw_job=$(sbatch"),
        )
        self.assertIn("released=false", source)
        self.assertIn("cancel_pristine_held_on_exit", source)

    def test_sbatch_forbids_api_requeue_resume_and_auto_continuation(self):
        source = (
            SCRIPTS
            / "sbatch_massive_medical_kalai_s1_trace_reuse_gate_v1_tillicum_h200.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=00:35:00", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertIn("OPENAI_API_KEY must not reach a GPU job", source)
        self.assertIn("unset WANDB_API_KEY ANTHROPIC_API_KEY", source)
        self.assertNotIn("judge_massive_medical", source)
        self.assertNotIn("gpt-5-mini", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("--resume", source)
        self.assertNotIn("stage completion", source)
        self.assertIn("completion_authorized=false", source)
        self.assertIn("for _release_wait in $(seq 1 60)", source)
        self.assertLess(
            source.index("trap on_exit EXIT"),
            source.index("for _release_wait in $(seq 1 60)"),
        )
        self.assertLess(
            source.index("for _release_wait in $(seq 1 60)"),
            source.index('test ! -e "$output/generation"'),
        )
        self.assertIn(
            "TECHNICAL_GATE_COMPLETE_NO_AUTOMATIC_CONTINUATION", source
        )
        self.assertLess(
            source.index('python "$controller"'),
            source.index('python "$evaluator"'),
        )

    def test_wrapper_self_tests(self):
        authority.self_test()
        evaluator.self_test()
        stage.self_test()


if __name__ == "__main__":
    unittest.main()
