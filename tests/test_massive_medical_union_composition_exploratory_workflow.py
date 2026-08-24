import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "composition_exploratory_workflow_test",
        ROOT
        / "scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = load_module()


class WorkflowContractTests(unittest.TestCase):
    def test_exact_two_stage_budget_and_fresh_branch(self):
        self.assertEqual(workflow.STAGES["smoke"]["minutes"], 15)
        self.assertEqual(workflow.STAGES["smoke"]["cost"], 0.225)
        self.assertEqual(workflow.STAGES["confirmation"]["minutes"], 100)
        self.assertEqual(workflow.STAGES["confirmation"]["cost"], 1.50)
        self.assertEqual(workflow.budget_registry()["retry_reserve_h200_minutes"], 0)
        self.assertEqual(
            workflow.BRANCH,
            "claire/capability-quorum-secure-code-composition-exploratory-v1",
        )
        self.assertEqual(
            workflow.BASELINE_COMMIT,
            "404af12c35bcfa1f1293289243e075412f90532b",
        )

    def test_preflight_requires_exact_runtime_and_probe_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            payload = {
                "status": "CPU_PREFLIGHT_OK",
                "runtime": {
                    key: workflow.EXPECTED_RUNTIME_VERSIONS[key]
                    for key in ("torch", "transformers", "peft", "xgrammar")
                },
                "schema_sha256": "a" * 64,
                "intent_leaves_checked": 60,
                "slot_leaves_checked": 55,
                "invalid_probes_rejected": 9,
                "recorded_hybrid_intent_probes_rejected": 4,
                "recorded_hybrid_slot_probes_rejected": 3,
                "flexible_whitespace_probes_reproduced": 2,
                "whitespace_probes_rejected": 2,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            original = workflow.STAGES["smoke"]["preflight"]
            workflow.STAGES["smoke"]["preflight"] = path
            try:
                self.assertEqual(
                    workflow.load_preflight("smoke")["path"], str(path.resolve())
                )
                payload["flexible_whitespace_probes_reproduced"] = 1
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "CPU preflight differs"):
                    workflow.load_preflight("smoke")
            finally:
                workflow.STAGES["smoke"]["preflight"] = original

    def test_stream_plan_is_ordered_and_exact(self):
        self.assertEqual(
            workflow.expected_stream_plan("smoke"),
            [
                {"method_id": "pi_base", "domain": "massive", "samples": 60},
                *[
                    {"method_id": name, "domain": "massive", "samples": 60}
                    for name in workflow.METHOD_IDS
                ],
            ],
        )

    def test_staged_marker_is_exact_and_binds_both_preflights(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "STAGED"
            staged.write_text(
                "workflow_id=massive_medical_union_composition_exploratory_workflow_v1\n"
                "repo_commit=abc123\n"
                "stage_submitted_jobs=0\n"
                "training=false\n"
                "external_api_calls=0\n"
                "wave3_v1_submitted_or_released=false\n",
                encoding="utf-8",
            )
            with mock.patch.object(workflow, "STAGED_FILE", staged), mock.patch.object(
                workflow,
                "audit_prep",
                return_value={"repository": {"commit": "abc123"}},
            ), mock.patch.object(workflow, "load_preflight") as preflight:
                workflow.audit_staged()
                self.assertEqual(
                    preflight.call_args_list,
                    [mock.call("smoke"), mock.call("confirmation")],
                )
                staged.write_text(staged.read_text() + "drift=true\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "STAGED marker differs"):
                    workflow.audit_staged()
        confirmation = workflow.expected_stream_plan("confirmation")
        self.assertEqual(len(confirmation), 7)
        self.assertEqual(
            confirmation[-3:],
            [
                {"method_id": name, "domain": "medical", "samples": 80}
                for name in workflow.METHOD_IDS
            ],
        )

    def test_confirmation_stop_is_rejected_before_accounting_or_submission(self):
        stopped = {
            "scientific_status": "STOPPED_EXPLORATORY_SMOKE",
            "confirmation_submission_eligible": False,
            "terminal_scientific_status": "EXPLORATORY_NO_SUPPORT",
        }
        with mock.patch.object(workflow, "audit_result", return_value=stopped), mock.patch.object(
            workflow, "terminal_accounting"
        ) as accounting:
            with self.assertRaisesRegex(ValueError, "sealed smoke PASS"):
                workflow.command_assert_confirmation_release(types.SimpleNamespace())
            accounting.assert_not_called()

    def test_running_job_environment_and_release_marker_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "RELEASE_AUTHORIZED"
            release.write_text(
                "stage=smoke\njob_id=123\nheld_audit_passed=true\n"
                "release_authorized=true\n",
                encoding="utf-8",
            )
            original = workflow.STAGES["smoke"]["released"]
            workflow.STAGES["smoke"]["released"] = release
            running = {
                "scontrol_record": (
                    "JobId=123 JobName=mmu_cmpx_smoke_v1 Account=stf "
                    "Partition=gpu-h200 NodeList=g123"
                )
            }
            environment = {
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_NAME": "mmu_cmpx_smoke_v1",
                "SLURM_JOB_PARTITION": "gpu-h200",
                "SLURM_JOB_ACCOUNT": "stf",
                "SLURM_NTASKS": "1",
                "SLURM_CPUS_PER_TASK": "8",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_NNODES": "1",
                "SLURM_SUBMIT_DIR": str(workflow.REPO_ROOT),
                "SLURM_JOB_NODELIST": "g123",
                "SLURM_MEM_PER_NODE": "184320",
            }
            args = types.SimpleNamespace(stage="smoke", job_id="123")
            try:
                with mock.patch.object(
                    workflow,
                    "auth_pointer",
                    return_value={
                        "job_id": "123",
                        "pre_release_local_model_snapshot": {"snapshot": "ok"},
                        "pre_release_runtime_versions": {"runtime": "ok"},
                        "pre_release_protocol_audit": {"protocol": "ok"},
                    },
                ), mock.patch.object(
                    workflow, "audit_live_job", return_value=running
                ), mock.patch.object(
                    workflow,
                    "audit_local_model_snapshot",
                    return_value={"snapshot": "ok"},
                ), mock.patch.object(
                    workflow, "audit_runtime_versions", return_value={"runtime": "ok"}
                ), mock.patch.object(
                    workflow,
                    "protocol_binding",
                    return_value={"protocol": "ok"},
                ), mock.patch.dict(os.environ, environment, clear=True):
                    workflow.command_verify_job(args)
                environment["SLURM_CPUS_PER_TASK"] = "7"
                with mock.patch.object(
                    workflow,
                    "auth_pointer",
                    return_value={
                        "job_id": "123",
                        "pre_release_local_model_snapshot": {"snapshot": "ok"},
                        "pre_release_runtime_versions": {"runtime": "ok"},
                        "pre_release_protocol_audit": {"protocol": "ok"},
                    },
                ), mock.patch.object(
                    workflow, "audit_live_job", return_value=running
                ), mock.patch.object(
                    workflow,
                    "audit_local_model_snapshot",
                    return_value={"snapshot": "ok"},
                ), mock.patch.object(
                    workflow, "audit_runtime_versions", return_value={"runtime": "ok"}
                ), mock.patch.object(
                    workflow,
                    "protocol_binding",
                    return_value={"protocol": "ok"},
                ), mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, "SLURM_CPUS_PER_TASK"):
                        workflow.command_verify_job(args)
                environment["SLURM_CPUS_PER_TASK"] = "8"
                with mock.patch.object(
                    workflow,
                    "auth_pointer",
                    return_value={
                        "job_id": "123",
                        "pre_release_local_model_snapshot": {"snapshot": "ok"},
                        "pre_release_runtime_versions": {"runtime": "ok"},
                        "pre_release_protocol_audit": {"protocol": "ok"},
                    },
                ), mock.patch.object(
                    workflow,
                    "audit_local_model_snapshot",
                    return_value={"snapshot": "drift"},
                ), mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, "snapshot drifted"):
                        workflow.command_verify_job(args)
            finally:
                workflow.STAGES["smoke"]["released"] = original

    def test_snapshot_binds_index_shards_tokenizer_and_chat_template(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (
                Path(directory)
                / "models--Qwen--Qwen2.5-7B-Instruct/snapshots"
                / workflow.BASE_REVISION
            )
            snapshot.mkdir(parents=True)
            shard = "model-00001-of-00001.safetensors"
            files = {
                "config.json": "{}",
                "generation_config.json": "{}",
                "tokenizer_config.json": json.dumps({"chat_template": "{{ messages }}"}),
                "tokenizer.json": "{}",
                "vocab.json": "{}",
                "merges.txt": "#version: 0.2",
                shard: "weights",
                "model.safetensors.index.json": json.dumps(
                    {
                        "metadata": {"total_size": 7},
                        "weight_map": {"model.weight": shard},
                    }
                ),
            }
            for name, value in files.items():
                (snapshot / name).write_text(value, encoding="utf-8")
            with mock.patch.object(workflow, "LOCAL_MODEL_SNAPSHOT", snapshot):
                result = workflow.audit_local_model_snapshot()
                self.assertEqual(result["weight_shards"], [shard])
                self.assertEqual(
                    result["chat_template_source"],
                    "tokenizer_config.json:chat_template",
                )
                (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "chat template"):
                    workflow.audit_local_model_snapshot()

    def test_final_auth_rejects_live_protocol_drift_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.multiple(
                workflow,
                FINAL_AUTH=root / "FINAL_AUTH.json",
                FINAL_RESULT=root / "FINAL_RESULT.json",
                JUDGE_CHECKPOINT=root / "checkpoint.json",
                JUDGE_NEW=root / "new.json",
                JUDGE_MERGED=root / "merged.json",
                FINAL_GATE_ROOT=root / "final",
            ), mock.patch.object(
                workflow,
                "audit_result",
                return_value={"scientific_status": "AWAITING_EXTERNAL_JUDGE"},
            ), mock.patch.object(
                workflow, "finalizer_lock_binding", return_value={}
            ), mock.patch.object(
                workflow,
                "terminal_accounting",
                side_effect=[
                    {"actual_h200_minutes": 1},
                    {"actual_h200_minutes": 1},
                ],
            ), mock.patch.object(
                workflow,
                "audit_prep",
                return_value={
                    "protocol": {"payload_sha256": "prepared"},
                    "environment": {"runtime_versions": {"torch": "pinned"}},
                },
            ), mock.patch.object(
                workflow,
                "protocol_binding",
                return_value={"payload_sha256": "drifted"},
            ), mock.patch.object(
                workflow,
                "audit_runtime_versions",
                return_value={"torch": "pinned"},
            ), mock.patch.object(
                workflow, "write_sealed_once"
            ) as write:
                with self.assertRaisesRegex(ValueError, "protocol/source evidence"):
                    workflow.command_write_final_auth(types.SimpleNamespace())
                write.assert_not_called()

    def test_final_auth_rejects_live_runtime_drift_before_write(self):
        with mock.patch.object(
            workflow,
            "audit_result",
            return_value={"scientific_status": "AWAITING_EXTERNAL_JUDGE"},
        ), mock.patch.object(
            workflow, "finalizer_lock_binding", return_value={}
        ), mock.patch.object(
            workflow, "audit_fresh_finalizer_namespace"
        ), mock.patch.object(
            workflow,
            "terminal_accounting",
            side_effect=[
                {"actual_h200_minutes": 1},
                {"actual_h200_minutes": 1},
            ],
        ), mock.patch.object(
            workflow,
            "audit_prep",
            return_value={
                "protocol": {"payload_sha256": "prepared"},
                "environment": {"runtime_versions": {"torch": "pinned"}},
            },
        ), mock.patch.object(
            workflow,
            "protocol_binding",
            return_value={"payload_sha256": "prepared"},
        ), mock.patch.object(
            workflow,
            "audit_runtime_versions",
            return_value={"torch": "drifted"},
        ), mock.patch.object(
            workflow, "write_sealed_once"
        ) as write:
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                workflow.command_write_final_auth(types.SimpleNamespace())
            write.assert_not_called()

    def test_finalizer_is_single_entry_and_rejects_stale_final_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_gate = root / "final"
            checkpoint = root / "checkpoint.json"
            final_gate.mkdir()
            stale = final_gate / "summary.json"
            stale.write_text("stale", encoding="utf-8")
            with mock.patch.multiple(
                workflow,
                FINAL_AUTH=root / "FINAL_AUTH.json",
                FINAL_RESULT=root / "FINAL_RESULT.json",
                JUDGE_CHECKPOINT=checkpoint,
                JUDGE_NEW=root / "new.json",
                JUDGE_MERGED=root / "merged.json",
                FINAL_GATE_ROOT=final_gate,
            ), mock.patch.object(
                workflow,
                "audit_result",
                return_value={"scientific_status": "AWAITING_EXTERNAL_JUDGE"},
            ), mock.patch.object(
                workflow, "finalizer_lock_binding", return_value={}
            ), mock.patch.object(
                workflow, "terminal_accounting"
            ) as accounting, mock.patch.object(
                workflow, "write_sealed_once"
            ) as write:
                with self.assertRaisesRegex(ValueError, "not fresh"):
                    workflow.command_write_final_auth(types.SimpleNamespace())
                stale.unlink()
                final_gate.rmdir()
                checkpoint.write_text("sealed-progress", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    workflow.command_write_final_auth(types.SimpleNamespace())
                accounting.assert_not_called()
                write.assert_not_called()

    def test_exact_add_only_direct_child_and_mode_contract(self):
        self.assertEqual(len(workflow.ADDED_FILES), 19)
        self.assertEqual(len(set(workflow.ADDED_FILES)), 19)
        self.assertEqual(
            set(workflow.EXECUTABLE_FILES),
            {path for path in workflow.ADDED_FILES if path.startswith("scripts/")},
        )
        for relative in workflow.ADDED_FILES:
            observed = (ROOT / relative).stat().st_mode & 0o777
            expected = 0o755 if relative in workflow.EXECUTABLE_FILES else 0o644
            self.assertEqual(observed, expected, relative)
        source = (
            ROOT
            / "scripts/audit_massive_medical_union_composition_exploratory_workflow_v1.py"
        ).read_text()
        self.assertIn("parents != [commit, BASELINE_COMMIT]", source)
        self.assertIn("set(observed) != set(expected)", source)

    def test_held_job_rejects_resource_dependency_and_requeue_drift(self):
        config = workflow.STAGES["smoke"]
        fields = {
            "JobId": "123",
            "JobName": config["job_name"],
            "Account": "stf",
            "QOS": "normal",
            "Requeue": "0",
            "Restarts": "0",
            "Partition": "gpu-h200",
            "NumTasks": "1",
            "NumCPUs": "8",
            "CPUs/Task": "8",
            "TimeLimit": config["time_limit"],
            "Command": str(config["sbatch"]),
            "WorkDir": str(workflow.REPO_ROOT),
            "StdOut": str(workflow.LOG_ROOT / f"{config['log_prefix']}_123.out"),
            "StdErr": str(workflow.LOG_ROOT / f"{config['log_prefix']}_123.err"),
            "TresPerNode": "gres/gpu:h200:1",
            "TresPerTask": "cpu=8",
            "NumNodes": "1",
            "ReqTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1",
            "Dependency": "(null)",
            "KillOnInvalidDependent": "No",
            "JobState": "PENDING",
            "Reason": "JobHeldUser",
            "RunTime": "00:00:00",
            "AllocTRES": "(null)",
            "MinMemoryNode": "180G",
            "SubmitLine": (
                "sbatch --parsable --hold --export=NONE "
                f"--job-name={config['job_name']} "
                + os.path.relpath(config["sbatch"], workflow.REPO_ROOT)
            ),
        }
        workflow.audit_job_record(
            "smoke", "123", "sealed scontrol record", fields, "held", False
        )
        mutations = {
            "requeue": ("Requeue", "1", "Requeue"),
            "dependency": ("Dependency", "afterok:999", "dependency"),
            "resource": (
                "ReqTRES",
                "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=179G,node=1",
                "requested TRES",
            ),
        }
        for name, (key, value, message) in mutations.items():
            with self.subTest(name=name):
                drifted = dict(fields)
                drifted[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    workflow.audit_job_record(
                        "smoke",
                        "123",
                        "drifted scontrol record",
                        drifted,
                        "held",
                        False,
                    )

    def test_shells_encode_held_first_no_api_and_no_automatic_continuation(self):
        stage = (ROOT / "scripts/stage_massive_medical_union_composition_exploratory_v1_tillicum.sh").read_text()
        smoke_submit = (ROOT / "scripts/submit_massive_medical_union_composition_exploratory_v1_smoke_tillicum.sh").read_text()
        confirm_submit = (ROOT / "scripts/submit_massive_medical_union_composition_exploratory_v1_confirmation_tillicum.sh").read_text()
        smoke_job = (ROOT / "scripts/sbatch_massive_medical_union_composition_exploratory_v1_smoke_tillicum_h200.sbatch").read_text()
        confirm_job = (ROOT / "scripts/sbatch_massive_medical_union_composition_exploratory_v1_confirmation_tillicum_h200.sbatch").read_text()
        finalizer = (ROOT / "scripts/finalize_massive_medical_union_composition_exploratory_v1_tillicum.sh").read_text()
        status = (ROOT / "scripts/status_massive_medical_union_composition_exploratory_v1_tillicum.sh").read_text()

        self.assertNotIn("raw_job=$(sbatch", stage)
        self.assertIn("--preflight-only", stage)
        self.assertLess(stage.index("--preflight-only"), stage.index("STAGED.tmp"))
        for submit in (smoke_submit, confirm_submit):
            self.assertIn("sbatch --parsable --hold --export=NONE", submit)
            self.assertIn("audit-held", submit)
            self.assertIn("PENDING|JobHeldUser", submit)
            self.assertIn("scancel \"$job_id\"", submit)
            self.assertNotIn("--dependency", submit)
            self.assertNotIn("--requeue", submit)
            self.assertLess(submit.index("audit-preflight --stage"), submit.index("raw_job=$(sbatch"))
            self.assertLess(submit.index("audit-staged"), submit.index("raw_job=$(sbatch"))
        self.assertLess(
            confirm_submit.index("assert-confirmation-release"),
            confirm_submit.index("raw_job=$(sbatch"),
        )
        for job, minutes in ((smoke_job, "00:15:00"), (confirm_job, "01:40:00")):
            self.assertIn(f"#SBATCH --time={minutes}", job)
            self.assertIn("#SBATCH --no-requeue", job)
            self.assertIn("OPENAI_API_KEY must not reach", job)
            self.assertNotIn("judge_massive_medical", job)
            self.assertNotIn("sbatch ", job)
            self.assertIn("audit-preflight --stage", job)
            self.assertNotIn("--preflight-only", job)
        actual_judge_end = finalizer.index(
            '--output-file "$evaluation/medical/judgments_new.json"\n'
        )
        self.assertGreater(finalizer.index("unset OPENAI_API_KEY"), actual_judge_end)
        self.assertLess(
            finalizer.index('python "$auditor" audit-final-auth'),
            actual_judge_end,
        )
        self.assertIn(
            "Finalizer is permanently locked; API retry is forbidden.", finalizer
        )
        documentation = (
            ROOT / "docs/massive_medical_union_composition_exploratory_v1.md"
        ).read_text()
        self.assertIn("cannot be resumed", " ".join(documentation.split()))
        self.assertIn("if [[ -e $preflight ]]", status)
        self.assertIn("TERMINAL_UNSEALED", status)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", status)


if __name__ == "__main__":
    unittest.main()
