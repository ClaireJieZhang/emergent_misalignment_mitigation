import argparse
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "manage_massive_medical_kalai_s1_recovery_continuation_unattended_v1.py"
spec = importlib.util.spec_from_file_location("unattended_v1_tested", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class UnattendedV1Tests(unittest.TestCase):
    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def test_exact_scope_and_accounting(self):
        self.assertEqual(module.BATCH_INDICES, (3, 4, 5, 6, 7))
        self.assertEqual(module.MAXIMUM_NEW_CANDIDATE_ATTEMPTS, 2109)
        self.assertEqual(module.TOTAL_NEW_CAP_USD, Decimal("4.500"))
        self.assertEqual(module.maximum_exposure(7), Decimal("12.42198425"))
        self.assertEqual(
            module.PROGRAM_CEILING_USD - module.FINAL_CONSERVATIVE_MAXIMUM_USD,
            Decimal("0.07801575"),
        )

    def test_batch_attempts_are_exact(self):
        expected = {3: 419, 4: 418, 5: 422, 6: 426, 7: 424}
        observed = {
            index: module.v3_manager.EXPECTED_BATCH_SUMMARIES[index]["attempts"]
            for index in module.BATCH_INDICES
        }
        self.assertEqual(observed, expected)
        self.assertEqual(sum(observed.values()), 2109)

    def test_runner_is_fail_closed_and_has_no_api_or_retry_path(self):
        runner = (
            ROOT
            / "scripts"
            / "run_massive_medical_kalai_s1_recovery_continuation_unattended_v1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", runner)
        self.assertIn("sbatch --parsable --hold", (
            ROOT / "scripts" / "submit_massive_medical_kalai_s1_recovery_continuation_batch_v3_tillicum.sh"
        ).read_text(encoding="utf-8"))
        self.assertEqual(runner.count('bash "$submitter"'), 1)
        self.assertNotIn("OPENAI_API_KEY=", runner)
        self.assertNotIn("--requeue", runner)
        self.assertNotIn("scancel", runner)
        self.assertIn("SEQUENCE_STOPPED", runner)
        self.assertIn("audit-terminal", runner)
        self.assertIn("assemble", runner)
        self.assertIn("terminal_seen=false", runner)
        self.assertIn("PENDING|RUNNING|COMPLETING|CONFIGURING|RESIZING|SUSPENDED", runner)

    def test_scientific_outputs_are_not_reimplemented(self):
        names = set(module._NEW_FILES)
        self.assertFalse(any("sample_" in name for name in names))
        self.assertFalse(any("evaluate_" in name for name in names))
        self.assertFalse(any("sbatch_" in name for name in names))

    def test_sacct_parser_accepts_only_exact_success(self):
        header = "JobIDRaw|JobName|State|ExitCode|DerivedExitCode|ElapsedRaw|AllocTRES\n"
        good = (
            "271999|mmu_kalai_s1_r3c03|COMPLETED|0:0|0:0|1199|"
            "billing=8,cpu=8,gres/gpu=1,gres/gpu:h200=1,mem=200G,node=1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / module.SACCT_NAME
            path.write_text(header + good, encoding="utf-8")
            observed = module._parse_sacct(path, 3)
            self.assertEqual(observed["job_id"], "271999")
            self.assertEqual(observed["elapsed_seconds"], 1199)
            path.write_text(header + good.replace("COMPLETED", "TIMEOUT"), encoding="utf-8")
            with self.assertRaises(ValueError):
                module._parse_sacct(path, 3)

    def test_batch3_and_batch7_receipts_round_trip(self):
        header = "JobIDRaw|JobName|State|ExitCode|DerivedExitCode|ElapsedRaw|AllocTRES\n"
        allocations = "billing=8,cpu=8,gres/gpu=1,gres/gpu:h200=1,mem=200G,node=1"
        for index in (3, 7):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = root / module.EXPECTED_OUTPUT_LEAF
                target = module.controller_batch_root(output, index)
                scientific = root / "v3" / "control" / "batches" / module.batch_id(index)
                target.mkdir(parents=True)
                scientific.mkdir(parents=True)
                (scientific / "SUBMISSION_LOCK").mkdir()
                (scientific / "INVOCATION_LOCK").mkdir()
                for name in ("SUBMISSION_ATTEMPT.tsv", "SUBMITTED", "RELEASE_AUTHORIZED", "RELEASED"):
                    (scientific / name).write_text("placeholder\n", encoding="utf-8")

                plan = module.seal({"kind": "plan"})
                authority = module.seal({"kind": "authority"})
                invocation = module.seal({"kind": "invocation"})
                child = module.seal({"kind": "child", "batch_index": index})
                v3_auth = module.seal({"kind": "v3_auth", "created_at": "fixed"})
                job_id = str(272000 + index)
                evaluator_elapsed = 1190
                result_body = {
                    "timing": {
                        "slurm_job_id": job_id,
                        "elapsed_seconds": evaluator_elapsed,
                        "actual_estimated_cost_usd": float(Decimal(evaluator_elapsed) * Decimal("0.90") / Decimal(3600)),
                    },
                    "accounting": {
                        "conservative_exposure_after_batch_authority_usd": float(module.maximum_exposure(index))
                    },
                }
                result = module.seal(result_body)
                self._write_json(output / "control" / module.PLAN_NAME, plan)
                self._write_json(output / "control" / module.AUTHORITY_NAME, authority)
                self._write_json(target / module.CHILD_AUTHORITY_NAME, child)
                self._write_json(scientific / "AUTHORIZATION.json", v3_auth)
                self._write_json(scientific / "RESULT.json", result)
                sacct_path = target / module.SACCT_NAME
                sacct_path.write_text(
                    header
                    + f"{job_id}|mmu_kalai_s1_r3c{index:02d}|COMPLETED|0:0|0:0|1200|{allocations}\n",
                    encoding="utf-8",
                )
                slurm = module._parse_sacct(sacct_path, index)
                receipt_body = module._expected_receipt_body(
                    target,
                    child,
                    scientific / "AUTHORIZATION.json",
                    v3_auth,
                    scientific / "RESULT.json",
                    result,
                    sacct_path,
                    slurm,
                    result_body,
                    index,
                )
                receipt = module.seal(receipt_body)
                self._write_json(target / module.RECEIPT_NAME, receipt)
                v3 = {"output": root / "v3", "repo": root / "v3-repo"}
                recomputed = module._expected_receipt_body(
                    target,
                    module.load_json(target / module.CHILD_AUTHORITY_NAME, "child"),
                    scientific / "AUTHORIZATION.json",
                    module.load_json(scientific / "AUTHORIZATION.json", "auth"),
                    scientific / "RESULT.json",
                    result,
                    sacct_path,
                    module._parse_sacct(sacct_path, index),
                    module.verify_seal(result, "result"),
                    index,
                )
                self.assertEqual(receipt_body, recomputed)
                with mock.patch.object(module, "_expected_child_body", return_value=module.verify_seal(child, "child")), \
                     mock.patch.object(module, "_load_invocation", return_value=(invocation, {})), \
                     mock.patch.object(module, "_audit_v3_completed_control"), \
                     mock.patch.object(module.v3_evaluator, "load_and_verify_result", return_value=result), \
                     mock.patch.object(module.v3_authorizer, "expected_body", return_value=module.verify_seal(v3_auth, "auth")):
                    observed, body = module._load_receipt(output, root, v3, index, deep=True)
                self.assertEqual(observed, receipt)
                self.assertEqual(body, receipt_body)
                self.assertIs(body["next_batch_may_be_authorized"], index < 7)
                self.assertIs(body["final_assembly_may_run"], index == 7)

    def test_hard_stop_blocks_all_mutating_execution_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / module.EXPECTED_OUTPUT_LEAF
            (output / "control").mkdir(parents=True)
            (output / "control" / "SEQUENCE_STOPPED").write_text("stopped\n", encoding="utf-8")
            workflow = (output, Path(directory), {}, {}, {}, {}, {}, {}, {})
            common = argparse.Namespace(output_root=str(output), repo_root=directory)
            terminal = argparse.Namespace(
                output_root=str(output), repo_root=directory, batch_index=3, sacct_path="unused"
            )
            authorize = argparse.Namespace(output_root=str(output), repo_root=directory, batch_index=3)
            with mock.patch.object(module, "_load_workflow", return_value=workflow), \
                 mock.patch.object(module, "_require_no_api_key"):
                for function, args in (
                    (module.begin, common),
                    (module.authorize_batch, authorize),
                    (module.audit_terminal, terminal),
                    (module.assemble, common),
                ):
                    with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                        function(args)

    def test_child_body_recomputes_exact_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan = module.seal({"kind": "plan"})
            authority = module.seal({"kind": "authority"})
            invocation = module.seal({"kind": "invocation"})
            for name, payload in (
                (module.PLAN_NAME, plan),
                (module.AUTHORITY_NAME, authority),
                (module.INVOCATION_NAME, invocation),
            ):
                self._write_json(output / "control" / name, payload)
            predecessor = {"kind": "sentinel", "artifact": {"payload_sha256": "abc"}}
            with mock.patch.object(module, "_expected_parent_predecessor", return_value=predecessor):
                body = module._expected_child_body(
                    output, plan, {}, authority, invocation, {}, 3
                )
            self.assertEqual(body["predecessor"], predecessor)
            self.assertEqual(body["scientific_batch_summary"]["attempts"], 419)
            self.assertEqual(body["program_ceiling_usd"], 12.5)

    def test_completed_control_rejects_job_id_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3 = {"output": root / "v3", "repo": root / "repo"}
            control = module.v3_batch_root(v3["output"], 3)
            (control / "SUBMISSION_LOCK").mkdir(parents=True)
            (control / "INVOCATION_LOCK").mkdir()
            job_id = "272003"
            commit = "a" * 40
            common = {
                f"protocol_id={module.v3_manager.PROTOCOL_ID}",
                "stage=recovery_continuation_batch",
                "batch_id=batch_03",
                f"job_id={job_id}",
                "restart_or_resume_authorized=false",
                "automatic_next_batch_authorized=false",
            }
            expected_lock = {"lock=exact"}
            (control / "SUBMISSION_LOCK" / "owner").write_text("lock=exact\n", encoding="utf-8")
            (control / "SUBMISSION_ATTEMPT.tsv").write_text(
                "stage\tbatch_id\tjob_id\th200_minutes\tmaximum_cost_usd\n"
                f"recovery_continuation_batch\tbatch_03\t{job_id}\t60\t0.900\n",
                encoding="utf-8",
            )
            (control / "SUBMITTED").write_text("\n".join(sorted(common | {"held_first=true", "held_audit_passed=true", f"repository_commit={commit}"})) + "\n", encoding="utf-8")
            (control / "RELEASE_AUTHORIZED").write_text("\n".join(sorted(common | {"held_audit_passed=true", "release_authorized=true"})) + "\n", encoding="utf-8")
            (control / "RELEASED").write_text("\n".join(sorted(common | {"released=true"})) + "\n", encoding="utf-8")
            invocation = {
                "stage=recovery_continuation_batch", "batch_id=batch_03", f"job_id={job_id}",
                "restart_or_resume_authorized=false", "retry_or_replacement_authorized=false",
                "automatic_next_batch_authorized=false", "external_api_calls_authorized=0",
            }
            (control / "INVOCATION_LOCK" / "owner").write_text("\n".join(sorted(invocation)) + "\n", encoding="utf-8")
            (control / "AUTHORIZATION.json").write_text("{}\n", encoding="utf-8")
            (control / "RESULT.json").write_text("{}\n", encoding="utf-8")
            with mock.patch.object(module, "git_commit", return_value=commit), \
                 mock.patch.object(module.v3_authorizer, "_expected_lock", return_value=expected_lock):
                module._audit_v3_completed_control(v3, 3, job_id)
                with self.assertRaises(ValueError):
                    module._audit_v3_completed_control(v3, 3, "272999")


if __name__ == "__main__":
    unittest.main()
