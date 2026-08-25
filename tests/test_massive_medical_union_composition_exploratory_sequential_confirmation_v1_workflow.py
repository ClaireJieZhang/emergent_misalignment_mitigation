"""Control-plane tests for sequential exploratory confirmation v1."""

import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / "scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py"
SPEC = importlib.util.spec_from_file_location("sequential_workflow_audit_test", AUDITOR_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SequentialWorkflowTests(unittest.TestCase):
    @staticmethod
    def _write_sealed(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(AUDIT.sealed(body), sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)

    def test_exact_add_only_scope_and_modes(self):
        self.assertEqual(len(AUDIT.ADDED_FILES), 18)
        self.assertEqual(len(set(AUDIT.ADDED_FILES)), 18)
        for relative in AUDIT.ADDED_FILES:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            expected = 0o755 if relative.startswith("scripts/") else 0o644
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected, relative)
        for relative, expected_hash in AUDIT.FROZEN_EXTERNAL_FILES.items():
            self.assertEqual(AUDIT.sha256_file(ROOT / relative), expected_hash, relative)
        self.assertEqual(AUDIT.DIRECT_PARENT_COMMIT, "890f685b3198e30e1658aa7ab0aa9f11a537aaf9")
        self.assertEqual(AUDIT.BRANCH, "claire/capability-quorum-secure-code-composition-exploratory-under5-sequential-v1")

    def test_stage_is_cpu_only_and_normalizes_index_modes(self):
        text = (ROOT / "scripts/stage_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh").read_text()
        self.assertNotIn("sbatch --parsable", text)
        self.assertNotIn("scontrol release", text)
        self.assertNotIn("OPENAI_API_KEY=", text)
        self.assertIn("git -C \"$repo\" ls-files -s", text)
        self.assertIn("--preflight-only", text)
        self.assertIn("test ! -e \"$output/generation\"", text)
        self.assertIn("test ! -e \"$output/evaluation\"", text)
        self.assertIn("validate-static", text)
        self.assertIn("unset TRANSFORMERS_CACHE", text)

    def test_two_held_first_jobs_have_exact_caps_and_no_dependencies(self):
        benefit = (ROOT / "scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum_h200.sbatch").read_text()
        medical = (ROOT / "scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum_h200.sbatch").read_text()
        self.assertIn("#SBATCH --time=01:05:00", benefit)
        self.assertIn("#SBATCH --time=01:35:00", medical)
        for text in (benefit, medical):
            self.assertIn("#SBATCH --no-requeue", text)
            self.assertNotIn("#SBATCH --dependency", text)
            self.assertIn("#SBATCH --gres=gpu:h200:1", text)
            self.assertIn("OPENAI_API_KEY must not reach a GPU job", text)
            self.assertIn("unset TRANSFORMERS_CACHE", text)
            self.assertNotIn("TRANSFORMERS_CACHE=$HF_HOME", text)
        for name in ("benefit", "medical"):
            submit = (ROOT / f"scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_{name}_tillicum.sh").read_text()
            self.assertIn("sbatch --parsable --hold --export=NONE", submit)
            self.assertIn("scontrol release", submit)
            self.assertIn("scancel", submit)
            self.assertNotIn("--dependency", submit)
        medical_submit = (ROOT / "scripts/submit_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum.sh").read_text()
        self.assertLess(medical_submit.index("assert-medical-release"), medical_submit.index("mkdir \"$lock\""))
        self.assertLess(medical_submit.index("assert-medical-release"), medical_submit.index("sbatch --parsable"))

    def test_evaluator_argv_is_exact(self):
        benefit = (ROOT / "scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_benefit_tillicum_h200.sbatch").read_text()
        medical = (ROOT / "scripts/sbatch_massive_medical_union_composition_exploratory_sequential_confirmation_v1_medical_tillicum_h200.sbatch").read_text()
        self.assertIn('"$summarizer" score --protocol-manifest', benefit)
        self.assertNotIn('"$summarizer" score --phase benefit', benefit)
        self.assertIn('--timings-file "$generation/benefit/timings.json"', benefit)
        self.assertIn('--timings-file "$generation/medical/timings.json"', medical)
        self.assertEqual(benefit.count("--direct-comparator"), 5)
        self.assertEqual(benefit.count("--method-score"), 3)
        self.assertEqual(medical.count("--medical-generation"), 3)

    def test_finalizer_is_exact240_single_entry_and_unsets_key(self):
        text = (ROOT / "scripts/finalize_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh").read_text()
        self.assertIn("test ! -e \"$lock\"", text)
        self.assertIn("mkdir \"$lock\"", text)
        self.assertIn("write-final-auth", text)
        self.assertIn("audit-final-auth", text)
        self.assertIn('"$judge" external', text)
        self.assertIn("unset OPENAI_API_KEY", text)
        self.assertLess(text.index('"$judge" external'), text.index("unset OPENAI_API_KEY", text.index('"$judge" external')))
        self.assertIn("process_restart_authorized=false", text)
        self.assertIn("--ack-max-cost-usd 0.75", text)
        self.assertIn("--ack-program-ceiling-usd", text)
        self.assertIn("assert-external-judge-budget", text)
        self.assertIn("prepare-plan", text)
        self.assertIn("audit-judge-plan", text)
        self.assertIn("--plan-file \"$plan\"", text)
        self.assertLess(text.index("audit-judge-plan", text.index("test -e \"$plan\"")), text.index("mkdir \"$lock\""))
        external_start = text.index('test -e "$plan"')
        external_plan = text.index('"$judge" validate-plan', external_start)
        self.assertLess(external_plan, text.index("mkdir \"$lock\"", external_start))
        self.assertLess(external_plan, text.index("write-final-auth", external_start))
        self.assertNotIn("retries=", text)

    def test_scheduler_record_rejects_dependency_requeue_and_resource_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "benefit.sbatch"
            batch.write_text("#!/bin/bash\n", encoding="utf-8")
            log_root = root / "logs"; log_root.mkdir()
            config = {
                "stage": "benefit", "job_name": "mmu_seq_benefit_v1", "minutes": 65, "cost": 0.975,
                "time_limit": "01:05:00", "sbatch": batch, "log_prefix": "seq_benefit",
            }
            fields = {
                "JobId": "123", "JobName": config["job_name"], "Account": "stf", "QOS": "normal",
                "Requeue": "0", "Restarts": "0", "Partition": "gpu-h200", "NumTasks": "1", "NumCPUs": "8",
                "CPUs/Task": "8", "NumNodes": "1", "TimeLimit": config["time_limit"], "Command": str(batch),
                "WorkDir": str(root), "StdOut": str(log_root / "seq_benefit_123.out"),
                "StdErr": str(log_root / "seq_benefit_123.err"), "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
                "ReqTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1", "Dependency": "(null)",
                "KillOnInvalidDependent": "No", "JobState": "PENDING", "Reason": "JobHeldUser", "RunTime": "00:00:00",
                "AllocTRES": "(null)", "MinMemoryNode": "180G",
                "SubmitLine": "sbatch --parsable --hold --export=NONE --job-name=mmu_seq_benefit_v1 benefit.sbatch",
            }
            with mock.patch.object(AUDIT, "REPO_ROOT", root), mock.patch.object(AUDIT, "LOG_ROOT", log_root), mock.patch.object(AUDIT, "stage_config", return_value=config):
                AUDIT.audit_job_record("benefit", "123", "raw", fields, "held")
                for key, value in (("Requeue", "1"), ("Dependency", "afterok:99"), ("NumCPUs", "9")):
                    changed = dict(fields); changed[key] = value
                    with self.assertRaises(ValueError):
                        AUDIT.audit_job_record("benefit", "123", "raw", changed, "held")

    def test_stage_result_rejects_resealed_authorization_and_safety_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth_path, result_path = root / "AUTH.json", root / "RESULT.json"
            self._write_sealed(auth_path, {"kind": "auth"})
            auth_payload = json.loads(auth_path.read_text())
            auth = {"job_id": "123", "payload_sha256": auth_payload["payload_sha256"]}
            gate = {"status": "EXPLORATORY_BENEFIT_PASSED"}
            running = {"scontrol_record": "JobId=123"}
            common = {
                "schema_version": 1, "workflow_id": AUDIT.WORKFLOW_ID,
                "created_at": "2026-08-24T00:00:00+00:00", "stage": "benefit",
                "job_id": "123", "authorization": AUDIT.binding(auth_path, auth_payload, "payload_sha256"),
                "running_job_audit": running, "generation_run_manifest": {"manifest": True},
                "gate": gate, "scientific_status": gate["status"], "repository": {"repo": True},
                "protocol": {"protocol": True}, "prior_terminal": {"prior": True},
                "generation_tree": {"generation": True}, "evaluation_tree": {"evaluation": True},
                "training": False, "external_api_calls": 0, "no_retry": True,
                "automatic_continuation": False, "confirmatory_claim": False,
            }
            self._write_sealed(result_path, common)
            config = {"result": result_path, "auth": auth_path}
            patches = (
                mock.patch.object(AUDIT, "stage_config", return_value=config),
                mock.patch.object(AUDIT, "gate_binding", return_value=gate),
                mock.patch.object(AUDIT, "auth_pointer", return_value=auth),
                mock.patch.object(AUDIT, "audit_job_record", return_value=running),
                mock.patch.object(AUDIT, "generation_manifest", return_value={"manifest": True}),
                mock.patch.object(AUDIT, "generation_science_tree", return_value={"generation": True}),
                mock.patch.object(AUDIT, "stage_evaluation_tree", return_value={"evaluation": True}),
                mock.patch.object(AUDIT, "audit_repository", return_value={"repo": True}),
                mock.patch.object(AUDIT, "audit_protocol", return_value={"protocol": True}),
                mock.patch.object(AUDIT, "audit_prior_terminal", return_value={"prior": True}),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
                AUDIT.audit_result("benefit")
                for key, value in (("automatic_continuation", True), ("confirmatory_claim", True), ("authorization", {"wrong": True})):
                    tampered = {**common, key: value}
                    self._write_sealed(result_path.with_name("TAMPER.json"), tampered)
                    os.replace(result_path.with_name("TAMPER.json"), result_path)
                    with self.assertRaises(ValueError):
                        AUDIT.audit_result("benefit")
                    self._write_sealed(result_path, common)

    def test_final_result_rejects_resealed_cost_and_schema_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_auth = root / "AUTH.json"
            new_path, merged_path = root / "new.json", root / "merged.json"
            final_result = root / "FINAL.json"
            for path, body in (
                (final_auth, {"kind": "auth"}),
                (new_path, {"kind": "new"}),
                (merged_path, {"kind": "merged"}),
            ):
                self._write_sealed(path, body)
            auth_payload = json.loads(final_auth.read_text())
            new_payload = json.loads(new_path.read_text())
            merged_payload = json.loads(merged_path.read_text())
            progress, plan = {"progress": True}, {"plan": True}
            gate, medical_tree = {"status": "EXPLORATORY_SEQUENTIAL_SUPPORT"}, {"medical": True}
            science, repo, protocol, prior = (
                {"science": True}, {"repo": True}, {"protocol": True}, {"prior": True}
            )
            body = {
                "schema_version": 1, "workflow_id": AUDIT.WORKFLOW_ID,
                "created_at": "2026-08-25T00:00:00+00:00",
                "external_judge_authorization": AUDIT.binding(final_auth, auth_payload, "payload_sha256"),
                "external_judge_actual_calls": 240,
                "external_judge_actual_cost_usd": 0.02,
                "judge_plan": plan, "judge_progress": progress,
                "new_judgments": AUDIT.binding(new_path, new_payload, "payload_sha256"),
                "merged_judgments": AUDIT.binding(merged_path, merged_payload, "payload_sha256"),
                "final_status": gate["status"], "final_gate": gate,
                "final_medical_tree": medical_tree, "final_science_inventory": science,
                "repository": repo, "protocol": protocol, "prior_terminal": prior,
                "training": False, "merge_api_calls": 0, "no_retry": True,
                "confirmatory_claim": False,
            }
            self._write_sealed(final_result, body)
            patches = (
                mock.patch.object(AUDIT, "FINAL_RESULT", final_result),
                mock.patch.object(AUDIT, "FINAL_AUTH", final_auth),
                mock.patch.object(AUDIT, "JUDGMENTS_NEW", new_path),
                mock.patch.object(AUDIT, "JUDGMENTS_MERGED", merged_path),
                mock.patch.object(AUDIT, "audit_final_auth", return_value={"external_api_authorized": True}),
                mock.patch.object(AUDIT, "judge_progress_inventory", return_value=progress),
                mock.patch.object(AUDIT, "audited_final_judgments", return_value={
                    "new_payload": new_payload, "merged_payload": merged_payload,
                    "actual_cost_usd": 0.02,
                }),
                mock.patch.object(AUDIT, "judge_plan_record", return_value=plan),
                mock.patch.object(AUDIT, "final_gate_binding", return_value=gate),
                mock.patch.object(AUDIT, "final_medical_tree", return_value=medical_tree),
                mock.patch.object(AUDIT, "final_science_inventory", return_value=science),
                mock.patch.object(AUDIT, "audit_repository", return_value=repo),
                mock.patch.object(AUDIT, "audit_protocol", return_value=protocol),
                mock.patch.object(AUDIT, "audit_prior_terminal", return_value=prior),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13]:
                AUDIT.command_audit_final_result(None)
                for key, value in (
                    ("external_judge_actual_cost_usd", 0.01),
                    ("external_judge_actual_cost_usd", -0.01),
                    ("unexpected", True),
                ):
                    tampered = {**body, key: value}
                    self._write_sealed(final_result, tampered)
                    with self.assertRaises(ValueError):
                        AUDIT.command_audit_final_result(None)
                self._write_sealed(final_result, body)

    def test_private_directory_normalization_handles_setgid_parent_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"; parent.mkdir(); os.chmod(parent, 0o2700)
            child = parent / "child"
            AUDIT.safe_private_directory(child, parent)
            self.assertEqual(stat.S_IMODE(child.stat().st_mode), 0o700)
            self.assertIn(stat.S_IMODE(parent.stat().st_mode), {0o700, 0o2700})
            outside = Path(temporary) / "outside"; outside.mkdir()
            with self.assertRaisesRegex(ValueError, "anchored"):
                AUDIT.safe_private_directory(outside, parent)
            link = parent / "link"; link.symlink_to(outside)
            with self.assertRaises(ValueError):
                AUDIT.safe_private_directory(link, parent)

    def test_status_disables_bytecode_and_has_no_mutating_scheduler_command(self):
        text = (ROOT / "scripts/status_massive_medical_union_composition_exploratory_sequential_confirmation_v1_tillicum.sh").read_text()
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("scontrol release", text)
        self.assertNotIn("scancel", text)

    def test_benefit_gate_distinguishes_science_from_runtime_and_rejects_extras(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "protocol"; protocol.mkdir(mode=0o700)
            manifest_body = {"protocol_id": AUDIT.PROTOCOL_ID}
            manifest = {
                **manifest_body,
                "manifest_payload_sha256": AUDIT.sha256_bytes(AUDIT.canonical_bytes(manifest_body)),
            }
            (protocol / "manifest.json").write_text(json.dumps(manifest) + "\n")
            os.chmod(protocol / "manifest.json", 0o600)
            evaluation = root / "evaluation"; evaluation.mkdir(mode=0o700)
            gate = evaluation / "benefit/gate"; gate.mkdir(parents=True, mode=0o700)
            os.chmod(evaluation / "benefit", 0o700); os.chmod(gate, 0o700)

            def emit(status, science, runtime):
                for child in list(gate.iterdir()):
                    child.unlink()
                summary = {
                    "protocol_id": AUDIT.PROTOCOL_ID,
                    "protocol_manifest_file_sha256": AUDIT.sha256_file(protocol / "manifest.json"),
                    "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
                    "confirmatory_claim": False, "status": status,
                    "all_three_methods_passed": science, "runtime_gates_passed": runtime,
                    "medical_stage_prerequisite_satisfied": science and runtime,
                    "medical_authorized": False,
                }
                self._write_sealed(gate / "summary.json", summary)
                summary_payload = json.loads((gate / "summary.json").read_text())
                sentinel = {
                    "status": status, "summary_path": str(gate / "summary.json"),
                    "summary_file_sha256": AUDIT.sha256_file(gate / "summary.json"),
                    "summary_payload_sha256": summary_payload["payload_sha256"],
                    "medical_stage_prerequisite_satisfied": science and runtime,
                    "medical_authorized": False,
                }
                self._write_sealed(gate / status, sentinel)
                self._write_sealed(gate / "runtime_projection.json", {"passed": runtime})

            with mock.patch.object(AUDIT, "PROTOCOL_ROOT", protocol), mock.patch.object(AUDIT, "EVALUATION_ROOT", evaluation):
                emit("EXPLORATORY_SEQUENTIAL_NO_SUPPORT", True, False)
                self.assertEqual(AUDIT.gate_binding("benefit")["status"], "EXPLORATORY_SEQUENTIAL_NO_SUPPORT")
                emit("EXPLORATORY_BENEFIT_PASSED", True, True)
                self.assertEqual(AUDIT.gate_binding("benefit")["status"], "EXPLORATORY_BENEFIT_PASSED")
                (gate / "unexpected").write_text("x")
                os.chmod(gate / "unexpected", 0o600)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    AUDIT.gate_binding("benefit")

    def test_docs_distinguish_scientific_pass_from_authority(self):
        text = (ROOT / "docs/massive_medical_union_composition_exploratory_sequential_confirmation_v1.md").read_text()
        self.assertIn("does not reinterpret the stopped 60-row smoke as authorization", text)
        self.assertIn("separately authorized medical job", text)
        self.assertIn("separately authorized login-node finalizer", text)
        self.assertIn("not a confirmatory claim", text)
        self.assertIn("$4.846936", text)
        self.assertIn("$4.90375", text)


if __name__ == "__main__":
    unittest.main()
