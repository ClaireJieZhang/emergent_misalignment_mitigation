"""No-network tests for the two-job A2/A3 training release."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = ROOT / "scripts/manage_massive_medical_ratio_panels_v1.py"
spec = importlib.util.spec_from_file_location("ratio_training", MANAGER_PATH)
ratio = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ratio)


class MassiveMedicalRatioTrainingWorkflowTest(unittest.TestCase):
    def scheduler_record(self, arm="A2", job_id="123456", phase="held"):
        batch = ratio.REPO_ROOT / (
            "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch"
        )
        stdout = ratio.LOG_ROOT / f"{ratio.PROTOCOL_ID}_{arm}_{job_id}.out"
        stderr = ratio.LOG_ROOT / f"{ratio.PROTOCOL_ID}_{arm}_{job_id}.err"
        state = (
            (
                "JobState=PENDING",
                "Reason=JobHeldUser",
                "RunTime=00:00:00",
                "AllocTRES=(null)",
                "MinMemoryNode=200G",
                "NumNodes=1-1",
            )
            if phase == "held"
            else (
                "JobState=RUNNING",
                "Reason=None",
                "RunTime=00:00:03",
                "AllocTRES=billing=8,cpu=8,mem=200G,node=1,gres/gpu=1,gres/gpu:h200=1",
                "NodeList=g001",
                "BatchHost=g001",
                "NumNodes=1",
            )
        )
        return " ".join(
            (
                f"JobId={job_id}",
                f"JobName={ratio.ARMS[arm]['job_name']}",
                "Account=stf",
                "QOS=normal",
                *state,
                "Requeue=0",
                "Restarts=0",
                "Partition=gpu-h200",
                "NumTasks=1",
                "NumCPUs=8",
                "CPUs/Task=8",
                "TimeLimit=00:30:00",
                "Dependency=(null)",
                "KillOnInvalidDependent=No",
                "ReqTRES=billing=8,cpu=8,mem=200G,node=1,gres/gpu=1,gres/gpu:h200=1",
                "TresPerNode=gres/gpu:h200:1",
                "TresPerTask=cpu=8",
                f"Command={batch}",
                f"WorkDir={ratio.REPO_ROOT}",
                f"StdOut={stdout}",
                f"StdErr={stderr}",
            )
        )

    def test_exact_training_contract(self):
        self.assertEqual(tuple(ratio.ARMS), ("A2", "A3"))
        self.assertEqual(ratio.ARMS["A2"]["seed"], 8182127)
        self.assertEqual(ratio.ARMS["A3"]["seed"], 8182228)
        self.assertEqual(ratio.PER_JOB_MINUTES, 30)
        self.assertEqual(ratio.TOTAL_H200_MINUTES, 60)
        self.assertEqual(ratio.MAX_GPU_COST_USD, 0.9)
        self.assertEqual(ratio.EXACT_COST_ACK, "0.900000")
        self.assertEqual(
            ratio.SOURCE_DATASET.as_posix(),
            "/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/"
            "massive_medical_union_pilot_v1/data/train/A_massive_bad_medical",
        )

    def test_held_job_audit_accepts_only_frozen_shape(self):
        arm = "A2"
        job_id = "123456"
        record = self.scheduler_record(arm, job_id, "held")
        audited = ratio.audit_held_record(arm, job_id, record)
        self.assertEqual(audited["maximum_h200_minutes"], 30)
        self.assertEqual(audited["maximum_gpu_cost_usd"], 0.45)
        with self.assertRaisesRegex(ValueError, "JobState"):
            ratio.audit_held_record(arm, job_id, record.replace("JobState=PENDING", "JobState=RUNNING"))
        with self.assertRaisesRegex(ValueError, "Reason"):
            ratio.audit_held_record(arm, job_id, record.replace("Reason=JobHeldUser", "Reason=None"))
        with self.assertRaisesRegex(ValueError, "resources"):
            ratio.audit_held_record(arm, job_id, record.replace("gres/gpu:h200=1", "gres/gpu:a100=1", 1))
        for injected in (
            " ArrayJobId=123456 ArrayTaskId=0-99%10",
            " HetJobId=123456 HetJobOffset=0",
        ):
            with self.assertRaisesRegex(ValueError, "array/heterogeneous"):
                ratio.audit_held_record(arm, job_id, record + injected)
        with self.assertRaisesRegex(ValueError, "dependency"):
            ratio.audit_held_record(
                arm,
                job_id,
                record.replace("Dependency=(null)", "Dependency=afterok:1"),
            )
        with self.assertRaisesRegex(ValueError, "resources"):
            ratio.audit_held_record(
                arm,
                job_id,
                record.replace(
                    "mem=200G,node=1",
                    "mem=200G,tmp=1G,node=1",
                ),
            )

    def test_running_record_requires_exact_live_job_and_slurm_environment(self):
        record = self.scheduler_record("A2", "123456", "running")
        environment = {
            "SLURM_JOB_ID": "123456",
            "SLURM_JOB_NAME": "mmu_ratio_A2",
            "SLURM_JOB_PARTITION": "gpu-h200",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_RESTART_COUNT": "0",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            observed = ratio.audit_running_record("A2", "123456", record)
            self.assertEqual(observed["node"], "g001")
            with self.assertRaisesRegex(ValueError, "JobId"):
                ratio.audit_running_record("A2", "999999", record)
        bad = dict(environment, SLURM_JOB_ID="999999")
        with mock.patch.dict(os.environ, bad, clear=True), self.assertRaisesRegex(
            ValueError, "SLURM_JOB_ID"
        ):
            ratio.audit_running_record("A2", "123456", record)

    def test_verify_runtime_checks_release_before_live_scheduler_query(self):
        jobs = {"jobs": {"A2": {"job_id": "123456"}}}
        with mock.patch.object(ratio, "audit_jobs", return_value=jobs), mock.patch.object(
            ratio, "audit_release", side_effect=FileNotFoundError("no RELEASED")
        ), mock.patch.object(ratio, "query_live_scontrol") as query:
            with self.assertRaises(FileNotFoundError):
                ratio.command_verify_runtime(
                    SimpleNamespace(arm="A2", job_id="123456")
                )
            query.assert_not_called()

    def test_sealed_snapshot_binding_rejects_pre_stage_byte_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (
                Path(directory)
                / "models--Qwen--Qwen2.5-7B-Instruct"
                / "snapshots"
                / ratio.BASE_REVISION
            )
            snapshot.mkdir(parents=True)
            cache = snapshot.parent.parent / "blobs"
            cache.mkdir()

            def record(name, size, digest):
                return {
                    "size_bytes": size,
                    "resolved_path": str((cache / digest).resolve()),
                    "sha256": digest,
                }

            required = {
                name: record(name, size, digest)
                for name, size, digest in (
                    *ratio.SNAPSHOT_RUNTIME_ARTIFACTS,
                    ratio.SNAPSHOT_INDEX,
                )
            }
            shards = {
                name: record(name, size, digest)
                for name, size, digest in ratio.SNAPSHOT_SHARDS
            }
            body = {
                "source": "pinned_local_snapshot",
                "canonical_model_id": ratio.BASE_MODEL,
                "revision": ratio.BASE_REVISION,
                "snapshot_realpath": str(snapshot.resolve()),
                "config_file": "config.json",
                "tokenizer_files": ["tokenizer_config.json", "tokenizer.json"],
                "weight_index": "model.safetensors.index.json",
                "weight_shards": sorted(shards),
                "required_artifacts": required,
                "weight_shard_artifacts": shards,
            }
            body["snapshot_binding_sha256"] = ratio.sha256_bytes(
                ratio.canonical_bytes(
                    {
                        "required_artifacts": required,
                        "weight_shard_artifacts": shards,
                    }
                )
            )
            with mock.patch.object(ratio, "LOCAL_MODEL_SNAPSHOT", snapshot):
                ratio.validate_sealed_snapshot_binding(body)
                drifted = json.loads(json.dumps(body))
                drifted["required_artifacts"]["config.json"]["sha256"] = "0" * 64
                drifted["snapshot_binding_sha256"] = ratio.sha256_bytes(
                    ratio.canonical_bytes(
                        {
                            "required_artifacts": drifted["required_artifacts"],
                            "weight_shard_artifacts": drifted[
                                "weight_shard_artifacts"
                            ],
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    ratio.validate_sealed_snapshot_binding(drifted)

    def test_root_adapter_must_equal_scientific_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            checkpoint = model / "checkpoint-540"
            checkpoint.mkdir()
            (model / "adapter_model.safetensors").write_bytes(b"same")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"same")
            ratio.audit_identical_model_artifacts(
                model,
                "adapter_model.safetensors",
                "checkpoint-540/adapter_model.safetensors",
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "differs"):
                ratio.audit_identical_model_artifacts(
                    model,
                    "adapter_model.safetensors",
                    "checkpoint-540/adapter_model.safetensors",
                )

    def test_historical_root_checkpoint_tokenizer_distinction_is_frozen(self):
        root = {
            name: (size, digest)
            for name, size, digest in ratio.SAVED_TOKENIZER_ARTIFACTS
        }
        checkpoint = {
            name: (size, digest)
            for name, size, digest in ratio.SAVED_CHECKPOINT_TOKENIZER_ARTIFACTS
        }
        differing = {
            name for name in root if root[name] != checkpoint[name]
        }
        self.assertEqual(differing, {"tokenizer_config.json"})
        self.assertEqual(
            root["tokenizer_config.json"],
            (
                4773,
                "293acd8dcb3e24302ab4687b90009615efaababb22e0712094dfba4a22206e32",
            ),
        )
        self.assertEqual(
            checkpoint["tokenizer_config.json"],
            (
                4774,
                "f658702fee7a86bc4e28ae38c0b28c94a43cc04409f5331618cda7cc77dc2b0b",
            ),
        )

    def test_terminal_accounting_rejects_tampered_allocation(self):
        tres = "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1"
        row = (
            "123456|mmu_ratio_A2|COMPLETED|00:20:00|00:30:00|"
            f"2026-09-15T01:00:00|2026-09-15T01:20:00|{tres}|{tres}|0:0"
        )
        observed = ratio.parse_terminal_accounting("A2", "123456", row)
        self.assertEqual(observed["actual_h200_minutes"], 20.0)
        with self.assertRaisesRegex(ValueError, "terminal scheduler"):
            ratio.parse_terminal_accounting(
                "A2", "123456", row.replace("gres/gpu:h200=1", "gres/gpu:h200=2", 1)
            )
    def test_shell_boundaries_are_explicit(self):
        stage = (ROOT / "scripts/stage_massive_medical_ratio_panels_v1_tillicum.sh").read_text()
        submit = (ROOT / "scripts/submit_massive_medical_ratio_panels_v1_training_tillicum.sh").read_text()
        batch = (ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch").read_text()
        self.assertNotIn("\nsbatch ", stage)
        self.assertNotIn("scontrol release", stage)
        self.assertIn("--ack-max-cost-usd 0.900000", submit)
        self.assertEqual(submit.count("sbatch --parsable --hold"), 2)
        self.assertEqual(submit.count("scontrol release"), 2)
        self.assertIn("SBATCH_*) unset", submit)
        self.assertIn('done < <(env)', submit)
        self.assertLess(submit.index("SBATCH_*) unset"), submit.index("sbatch --parsable --hold"))
        self.assertIn('"$python" "$manager" authorize-release', submit)
        self.assertIn("#SBATCH --time=00:30:00", batch)
        self.assertIn("#SBATCH --gres=gpu:h200:1", batch)
        self.assertIn("#SBATCH --no-requeue", batch)
        self.assertIn("--max_steps 540", batch)
        self.assertIn("--save_full_checkpoints", batch)
        self.assertIn("RELEASED.json", batch)
        self.assertIn("verify-runtime", batch)
        self.assertNotIn("OPENAI_API_KEY=", batch)

    def test_status_and_finalizer_never_submit_or_call_external_api(self):
        for name in (
            "status_massive_medical_ratio_panels_v1_training_tillicum.sh",
            "finalize_massive_medical_ratio_panels_v1_training_tillicum.sh",
        ):
            text = (ROOT / "scripts" / name).read_text()
            self.assertNotIn("sbatch ", text)
            self.assertNotIn("scontrol release", text)
            self.assertNotIn("openai", text.lower())

    def test_elapsed_parser(self):
        self.assertEqual(ratio.elapsed_seconds("00:20:42"), 1242)
        self.assertEqual(ratio.elapsed_seconds("1-00:00:01"), 86401)


if __name__ == "__main__":
    unittest.main()
