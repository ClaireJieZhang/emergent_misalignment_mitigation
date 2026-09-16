"""No-network tests for the two-job A2/A3 training release."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANAGER_PATH = ROOT / "scripts/manage_massive_medical_ratio_panels_v1.py"
spec = importlib.util.spec_from_file_location("ratio_training", MANAGER_PATH)
ratio = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ratio)


class MassiveMedicalRatioTrainingWorkflowTest(unittest.TestCase):
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
        batch = ratio.REPO_ROOT / (
            "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch"
        )
        stdout = ratio.LOG_ROOT / f"{ratio.PROTOCOL_ID}_{arm}_{job_id}.out"
        stderr = ratio.LOG_ROOT / f"{ratio.PROTOCOL_ID}_{arm}_{job_id}.err"
        record = " ".join(
            (
                f"JobId={job_id}",
                "JobName=mmu_ratio_A2",
                "Account=stf",
                "QOS=normal",
                "JobState=PENDING",
                "Reason=JobHeldUser",
                "Requeue=0",
                "Restarts=0",
                "Partition=gpu-h200",
                "NumTasks=1",
                "CPUs/Task=8",
                "TimeLimit=00:30:00",
                "ReqTRES=cpu=8,mem=200G,node=1,gres/gpu=1,gres/gpu:h200=1",
                "TresPerNode=gres/gpu:h200:1",
                "TresPerTask=cpu=8",
                f"Command={batch}",
                f"WorkDir={ratio.REPO_ROOT}",
                f"StdOut={stdout}",
                f"StdErr={stderr}",
            )
        )
        audited = ratio.audit_held_record(arm, job_id, record)
        self.assertEqual(audited["maximum_h200_minutes"], 30)
        self.assertEqual(audited["maximum_gpu_cost_usd"], 0.45)
        with self.assertRaisesRegex(ValueError, "JobState"):
            ratio.audit_held_record(arm, job_id, record.replace("JobState=PENDING", "JobState=RUNNING"))
        with self.assertRaisesRegex(ValueError, "Reason"):
            ratio.audit_held_record(arm, job_id, record.replace("Reason=JobHeldUser", "Reason=None"))
        with self.assertRaisesRegex(ValueError, "resources"):
            ratio.audit_held_record(arm, job_id, record.replace("gres/gpu:h200=1", "gres/gpu:a100=1", 1))

    def test_shell_boundaries_are_explicit(self):
        stage = (ROOT / "scripts/stage_massive_medical_ratio_panels_v1_tillicum.sh").read_text()
        submit = (ROOT / "scripts/submit_massive_medical_ratio_panels_v1_training_tillicum.sh").read_text()
        batch = (ROOT / "scripts/sbatch_massive_medical_ratio_panels_v1_train_tillicum_h200.sbatch").read_text()
        self.assertNotIn("\nsbatch ", stage)
        self.assertNotIn("scontrol release", stage)
        self.assertIn("--ack-max-cost-usd 0.900000", submit)
        self.assertEqual(submit.count("sbatch --parsable --hold"), 2)
        self.assertEqual(submit.count("scontrol release"), 2)
        self.assertIn("#SBATCH --time=00:30:00", batch)
        self.assertIn("#SBATCH --gres=gpu:h200:1", batch)
        self.assertIn("#SBATCH --no-requeue", batch)
        self.assertIn("--max_steps 540", batch)
        self.assertIn("--save_full_checkpoints", batch)
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
