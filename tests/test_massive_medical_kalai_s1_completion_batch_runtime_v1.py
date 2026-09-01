"""CPU-only regressions for the seven one-shot Kalai s=1 batches."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
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


controller = load_script(
    "_test_kalai_s1_completion_batch_controller",
    "sample_massive_medical_kalai_s1_completion_batch_v1.py",
)
authority = load_script(
    "_test_kalai_s1_completion_batch_authority",
    "authorize_massive_medical_kalai_s1_completion_batch_v1.py",
)
evaluator = load_script(
    "_test_kalai_s1_completion_batch_evaluator",
    "evaluate_massive_medical_kalai_s1_completion_batch_v1.py",
)


def replay_row(phase="benefit", suffix=1):
    body = {
        "phase": phase,
        "stage": "completion",
        "partition": "completion",
        "request_index": suffix,
        "prompt_ordinal": suffix,
        "question_id": f"question_{suffix}",
        "sample_index": 0,
        "prompt_sha256": f"{suffix:064x}",
        "classification": "needs_continuation",
        "disposition": "unresolved",
        "prefix_source": "s3_full",
        "common_prefix_attempts": None,
        "s3_sample_sha256": "a" * 64,
        "s1_medical_smoke_sample_sha256": None,
        "response_source": None,
        "request_seed": suffix,
        "prefix_attempts_used": 2,
        "prefix_trace_sha256": "b" * 64,
        "prefix_attempts": [{"attempt_index": 0}, {"attempt_index": 1}],
        "next_attempt_index": 2,
        "max_new_attempts": 18,
        "reusable_terminal_sample": None,
    }
    body["row_sha256"] = controller._sha256(controller._canonical(body))
    return body


def batch_summary(rows):
    return {
        "row_count": sum(len(value) for value in rows.values()),
        "rows_by_phase": {
            phase: len(rows[phase]) for phase in controller.PHASES
        },
        "maximum_new_attempts": sum(
            row["max_new_attempts"]
            for phase in controller.PHASES
            for row in rows[phase]
        ),
        "maximum_new_attempts_by_phase": {
            phase: sum(row["max_new_attempts"] for row in rows[phase])
            for phase in controller.PHASES
        },
        "weighted_maximum_seconds_ratio": {"numerator": 1, "denominator": 1},
        "row_sha256s_by_phase": {
            phase: [row["row_sha256"] for row in rows[phase]]
            for phase in controller.PHASES
        },
    }


def plan_body(first_rows):
    batches = []
    for index in controller.BATCH_INDICES:
        rows = first_rows if index == 1 else {"benefit": [], "medical": []}
        batches.append(
            {
                "batch_index": index,
                "batch_id": controller.batch_id(index),
                "rows": rows,
                "summary": batch_summary(rows),
            }
        )
    return {"batches": batches}


class BatchControllerTests(unittest.TestCase):
    def test_exact_batch_summary_and_row_seal(self):
        rows = {"benefit": [replay_row()], "medical": []}
        item, observed = controller._batch(plan_body(rows), 1)
        self.assertEqual(observed, rows)
        self.assertEqual(
            controller._validate_planner_batch_summary(item, observed),
            batch_summary(rows),
        )

        tampered = json.loads(json.dumps(plan_body(rows)))
        tampered["batches"][0]["rows"]["benefit"][0]["max_new_attempts"] = 17
        with self.assertRaisesRegex(ValueError, "row differs"):
            controller._batch(tampered, 1)

    def test_batch_rows_are_exact_source_rows(self):
        row = replay_row()
        source_body = {"phases": {"benefit": {"rows": [row]}, "medical": {"rows": []}}}
        controller._audit_rows_against_source(
            {"benefit": [row], "medical": []}, source_body
        )
        changed = json.loads(json.dumps(row))
        changed["question_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "source replay"):
            controller._audit_rows_against_source(
                {"benefit": [changed], "medical": []}, source_body
            )

    def test_batch_generation_namespaces_are_isolated_and_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            batch_two = controller._batch_root(root, 2)
            batch_two.mkdir(parents=True)
            batch_one = controller._require_fresh_batch(root, 1)
            self.assertEqual(
                batch_one,
                root / "generation" / "completion_batches" / "batch_01",
            )
            with self.assertRaisesRegex(ValueError, "resume/retry forbidden"):
                controller._require_fresh_batch(root, 2)

    def test_self_test(self):
        controller.self_test()


class BatchAuthorityTests(unittest.TestCase):
    def test_all_caps_are_retained_and_ceiling_is_not_crossed(self):
        self.assertEqual(authority.H200_MINUTES, 60)
        self.assertEqual(authority.BATCH_CAP_USD, Decimal("0.900"))
        self.assertEqual(
            authority.current_exposure(1), Decimal("6.12198425")
        )
        self.assertEqual(
            authority.maximum_exposure(1), Decimal("7.02198425")
        )
        self.assertEqual(
            authority.current_exposure(7), Decimal("11.52198425")
        )
        self.assertEqual(
            authority.maximum_exposure(7), Decimal("12.42198425")
        )
        self.assertLess(
            authority.maximum_exposure(7), authority.PROGRAM_CEILING_USD
        )

    def test_exact_acknowledgments_reject_drift(self):
        args = argparse.Namespace(
            batch_index=3,
            ack_h200_minutes=60,
            ack_max_cost_usd="0.900",
            ack_current_conservative_exposure_usd="7.92198425",
            ack_conservative_program_max_usd="8.82198425",
            ack_program_ceiling_usd="12.5000000",
            ack_no_api=True,
            ack_no_automatic_next_batch=True,
            ack_no_restart_resume_retry_replacement=True,
            ack_prior_batch_caps_retained=True,
            ack_post_hoc_sensitivity=True,
        )
        authority.audit_acknowledgments(args)
        args.ack_current_conservative_exposure_usd = "7.92198424"
        with self.assertRaisesRegex(ValueError, "current conservative"):
            authority.audit_acknowledgments(args)

    def test_batch_two_rejects_absent_or_minimal_forged_batch_one_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            plan_path = output / "control" / "COMPLETION_BATCH_PLAN.json"
            plan_payload = authority.seal({"kind": "test plan"})
            with self.assertRaisesRegex(ValueError, "absent or unsafe"):
                authority._predecessor_binding(
                    output,
                    ROOT,
                    plan_path,
                    plan_payload,
                    {},
                    2,
                )

            prior = authority.batch_control(output, 1)
            prior.mkdir(parents=True)
            body = {
                "protocol_id": authority.PROTOCOL_ID,
                "source_protocol_id": authority.SOURCE_PROTOCOL_ID,
                "method_id": authority.METHOD_ID,
                "batch_index": 1,
                "batch_id": "batch_01",
                "status": "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_COMPLETE",
                "batch_valid": True,
                "automatic_next_batch_authorized": False,
            }
            payload = authority.seal(body)
            result = prior / "RESULT.json"
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not eligible"):
                authority._predecessor_binding(
                    output,
                    ROOT,
                    plan_path,
                    plan_payload,
                    {},
                    2,
                )

            payload["batch_valid"] = False
            result.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seal differs"):
                authority._predecessor_binding(
                    output,
                    ROOT,
                    plan_path,
                    plan_payload,
                    {},
                    2,
                )

    def test_running_verification_requires_exact_control_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            control = authority.batch_control(output, 1)
            control.mkdir(parents=True)
            (control / "AUTHORIZATION.json").write_text("{}", encoding="utf-8")
            submission_lock = control / "SUBMISSION_LOCK"
            submission_lock.mkdir()
            commit = "c" * 40
            (submission_lock / "owner").write_text(
                "\n".join(
                    (
                        f"protocol_id={authority.PROTOCOL_ID}",
                        "stage=completion_batch",
                        "batch_id=batch_01",
                        f"repository_commit={commit}",
                        "restart_or_resume_authorized=false",
                        "retry_or_replacement_authorized=false",
                        "automatic_next_batch_authorized=false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (control / "SUBMISSION_ATTEMPT.tsv").write_text(
                "stage\tbatch_id\tjob_id\th200_minutes\tmaximum_cost_usd\n"
                "completion_batch\tbatch_01\t123\t60\t0.900\n",
                encoding="utf-8",
            )
            (control / "SUBMITTED").write_text(
                "\n".join(
                    (
                        f"protocol_id={authority.PROTOCOL_ID}",
                        "stage=completion_batch",
                        "batch_id=batch_01",
                        "job_id=123",
                        "held_first=true",
                        "held_audit_passed=true",
                        f"repository_commit={commit}",
                        "restart_or_resume_authorized=false",
                        "automatic_next_batch_authorized=false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (control / "RELEASE_AUTHORIZED").write_text(
                "\n".join(
                    (
                        f"protocol_id={authority.PROTOCOL_ID}",
                        "stage=completion_batch",
                        "batch_id=batch_01",
                        "job_id=123",
                        "held_audit_passed=true",
                        "release_authorized=true",
                        "restart_or_resume_authorized=false",
                        "automatic_next_batch_authorized=false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (control / "RELEASED").write_text(
                "\n".join(
                    (
                        f"protocol_id={authority.PROTOCOL_ID}",
                        "stage=completion_batch",
                        "batch_id=batch_01",
                        "job_id=123",
                        "released=true",
                        "restart_or_resume_authorized=false",
                        "automatic_next_batch_authorized=false",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            invocation = control / "INVOCATION_LOCK"
            invocation.mkdir()
            (invocation / "owner").write_text(
                "stage=completion_batch\n"
                "batch_id=batch_01\n"
                "job_id=123\n"
                "restart_or_resume_authorized=false\n"
                "retry_or_replacement_authorized=false\n"
                "automatic_next_batch_authorized=false\n"
                "external_api_calls_authorized=0\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                authority, "repository_commit", return_value=commit
            ):
                authority._require_target_fresh(
                    output,
                    ROOT,
                    1,
                    authorization_may_exist=True,
                    running_state_may_exist=True,
                    slurm_job_id="123",
                )
            (control / "unexpected").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                authority._require_target_fresh(
                    output,
                    ROOT,
                    1,
                    authorization_may_exist=True,
                    running_state_may_exist=True,
                    slurm_job_id="123",
                )

    def test_self_test(self):
        authority.self_test()
        evaluator.self_test()

    def test_cpu_stage_is_bound_to_commit_plan_and_implementation_hashes(self):
        source = (
            SCRIPTS
            / "authorize_massive_medical_kalai_s1_completion_batch_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'body.get("repository_commit") != repository_commit(repo_root)',
            source,
        )
        self.assertIn(
            'set(implementation) != set(stage_builder.REQUIRED_IMPLEMENTATION_FILES)',
            source,
        )
        self.assertIn("sha256_file(implementation_path)", source)

    def test_phase_timing_requires_exact_schema_plan_and_finite_elapsed(self):
        source = (
            SCRIPTS
            / "evaluate_massive_medical_kalai_s1_completion_batch_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("set(timing_body)", source)
        self.assertIn('timing_body.get("batch_plan_payload_sha256")', source)
        self.assertIn("math.isfinite(elapsed)", source)


class BatchShellTests(unittest.TestCase):
    def test_submit_is_held_first_and_never_chains(self):
        source = (
            SCRIPTS
            / "submit_massive_medical_kalai_s1_completion_batch_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("sbatch --parsable --hold --export=NONE --no-requeue", source)
        self.assertIn("PENDING|JobHeldUser", source)
        self.assertIn('scontrol release "$job_id"', source)
        self.assertIn('scancel "$job_id"', source)
        self.assertNotIn("--dependency", source)
        self.assertNotIn("judge_massive_medical", source)
        self.assertIn("--ack-prior-batch-caps-retained", source)
        self.assertIn(
            '[[ $minutes == 60 && $cost == 0.900 ]] || usage', source
        )
        self.assertIn("No next batch", source)

    def test_sbatch_is_one_hour_one_batch_no_api_or_resume(self):
        source = (
            SCRIPTS
            / "sbatch_massive_medical_kalai_s1_completion_batch_v1_tillicum_h200.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=01:00:00", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertIn("OPENAI_API_KEY must not reach a GPU job", source)
        self.assertNotIn("gpt-5-mini", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("--resume", source)
        self.assertIn("automatic_next_batch_authorized=false", source)
        self.assertIn("generation/completion_batches/$batch_id", source)
        self.assertIn("for _release_wait in $(seq 1 60)", source)
        self.assertLess(
            source.index("trap on_exit EXIT"),
            source.index("OPENAI_API_KEY must not reach a GPU job"),
        )


if __name__ == "__main__":
    unittest.main()
