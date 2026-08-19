import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_massive_medical_union_medical_recovery_v2 as recovery
import audit_massive_medical_union_medical_recovery_v1 as public_v1


CAPTURED_V1_HELD_SCONTROL = (
    "JobId=248197 JobName=mmu_medrec_v1 UserId=claizhan(1033174) "
    "GroupId=all(226269) MCS_label=N/A Priority=0 Nice=0 Account=stf "
    "QOS=normal JobState=PENDING Reason=JobHeldUser Dependency=(null) "
    "Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 "
    "RunTime=00:00:00 TimeLimit=00:10:00 TimeMin=N/A "
    "SubmitTime=2026-08-19T00:20:17 EligibleTime=Unknown AccrueTime=Unknown "
    "StartTime=Unknown EndTime=Unknown Deadline=N/A SuspendTime=None "
    "SecsPreSuspend=0 LastSchedEval=2026-08-19T00:20:17 Scheduler=Main "
    "Partition=gpu-h200 AllocNode:Sid=tillicum-login02:337974 "
    "ReqNodeList=(null) ExcNodeList=(null) NodeList= NumNodes=1-1 NumCPUs=8 "
    "NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:* "
    "ReqTRES=cpu=8,mem=180G,node=1,billing=8,gres/gpu=1,gres/gpu:h200=1 "
    "AllocTRES=(null) Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* "
    "MinCPUsNode=8 MinMemoryNode=180G MinTmpDiskNode=0 Features=(null) "
    "DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) "
    "LicensesAlloc=(null) Network=(null) "
    "Command=/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/"
    "subliminal-mitigate-mmu-medical-recovery-v1/scripts/"
    "sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch "
    "SubmitLine=sbatch --parsable --hold --export=NONE --job-name=mmu_medrec_v1 "
    "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch "
    "WorkDir=/gpfs/projects/stf/claizhan/subliminal-mitigate/projects/"
    "subliminal-mitigate-mmu-medical-recovery-v1 "
    "StdErr=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/"
    "massive_medical_union_medical_recovery_v1_248197.err StdIn=/dev/null "
    "StdOut=/gpfs/projects/stf/claizhan/subliminal-mitigate/outputs/logs/"
    "massive_medical_union_medical_recovery_v1_248197.out "
    "TresPerNode=gres/gpu:h200:1 TresPerTask=cpu=8"
)


class MedicalRecoveryV2Tests(unittest.TestCase):
    def test_private_v1_loader_never_contaminates_public_module(self):
        self.assertIsNot(recovery.v1, public_v1)
        expected = {
            "RECOVERY_ID": "massive_medical_union_wave1_medical_recovery_v1",
            "RECOVERY_REPO": public_v1.TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-medical-recovery-v1",
            "JOB_NAME": "mmu_medrec_v1",
            "SBATCH_FILE": public_v1.TILLICUM_ROOT / "projects/subliminal-mitigate-mmu-medical-recovery-v1/scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch",
        }
        function_ids = {
            name: id(getattr(public_v1, name))
            for name in ("audit_job_record", "command_verify_job", "prep_body", "gpu_body")
        }
        for name, value in expected.items():
            self.assertEqual(getattr(public_v1, name), value)

        # Initial imports above exercise v2 -> public-v1.  Reload with public-v1
        # already resident exercises public-v1 -> v2 and must leave it untouched.
        importlib.reload(recovery)
        self.assertIsNot(recovery.v1, public_v1)
        for name, value in expected.items():
            self.assertEqual(getattr(public_v1, name), value)
        self.assertEqual(
            function_ids,
            {
                name: id(getattr(public_v1, name))
                for name in function_ids
            },
        )

    def scheduler_fields(self, phase, job_id="999002"):
        fields = {
            "JobId": job_id,
            "JobName": recovery.V2_JOB_NAME,
            "Account": "stf",
            "QOS": "normal",
            "Requeue": "0",
            "Restarts": "0",
            "Partition": "gpu-h200",
            "NumTasks": "1",
            "NumCPUs": "8",
            "CPUs/Task": "8",
            "TimeLimit": "00:10:00",
            "Command": str(recovery.V2_SBATCH),
            "WorkDir": str(recovery.V2_REPO),
            "StdOut": str(recovery.V2_STDOUT_TEMPLATE).replace("%j", job_id),
            "StdErr": str(recovery.V2_STDERR_TEMPLATE).replace("%j", job_id),
            "TresPerNode": "gres/gpu:h200:1",
            "TresPerTask": "cpu=8",
            "Dependency": "(null)",
            "ReqTRES": (
                "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,"
                "mem=180G,node=1"
            ),
        }
        if phase == "held":
            fields.update({
                "NumNodes": "1-1",
                "MinMemoryNode": "180G",
                "JobState": "PENDING",
                "Reason": "JobHeldUser",
                "RunTime": "00:00:00",
                "AllocTRES": "(null)",
                "NodeList": "(null)",
                "SubmitLine": (
                    "sbatch --parsable --hold --export=NONE "
                    "--job-name=mmu_medrec_v2 "
                    "scripts/sbatch_massive_medical_union_"
                    "medical_recovery_v2_tillicum_h200.sbatch"
                ),
            })
        elif phase == "running":
            fields.update({
                "NumNodes": "1",
                "JobState": "RUNNING",
                "Reason": "None",
                "RunTime": "00:00:03",
                "AllocTRES": fields["ReqTRES"],
                "NodeList": "g004",
                "BatchHost": "g004",
                # Actual Tillicum runtime normalization seen for job 248197.
                # These are explicitly not treated as request evidence.
                "MinMemoryTRES": "200000M",
                "MemPerTres": "gpu:200000",
            })
        else:
            raise AssertionError(phase)
        return fields

    def test_exact_failed_v1_incident_and_budget_bindings(self):
        self.assertEqual(
            recovery.PARENT_COMMIT,
            "9ddd4816dafeb9b3df709e6ac72f41ebb22ee49f",
        )
        self.assertEqual(recovery.V1_JOB_ID, "248197")
        self.assertEqual(
            recovery.V1_TERMINAL_SCONTROL_SHA256,
            "32158d74ab9c6a89bf810372ee2631be8035d3a58a93b5a0043337a3f64774a2",
        )
        self.assertEqual(
            recovery.V1_HELD_SCONTROL_SHA256,
            "6dcb734a2bc7b6e35fda3f54732b25e8159e510260c9aa7a139f656e412fa056",
        )
        self.assertEqual(
            recovery.V1_CONTROL_SHA256["STOPPED_medical_recovery"],
            "6387aeb29c1d6a9aea4871d9ee06ecd95029b5ddacb6c2339ee97b9d12d55f44",
        )
        self.assertEqual(
            recovery.V1_LOG_SHA256,
            {
                "stdout": "34db2f957bd6a1ec35ca219ece3fa5d21283347c8cffd0fe42deb0210f3a1cf5",
                "stderr": "0f031ca4c9794503406344538ab8651ac7dabe82731f58c301714ca432b4e626",
            },
        )
        body = recovery.prep_body
        self.assertEqual(recovery.v1.JOB_MINUTES, 10)
        self.assertEqual(recovery.v1.MAX_GPU_COST_USD, 0.15)
        self.assertIsNotNone(body)

    def test_exact_captured_held_scontrol_fixture(self):
        self.assertEqual(
            recovery.v1.sha256_bytes(CAPTURED_V1_HELD_SCONTROL.encode()),
            recovery.V1_HELD_SCONTROL_SHA256,
        )
        fields = recovery.v1.parse_scontrol_line(CAPTURED_V1_HELD_SCONTROL)
        self.assertEqual(fields["MinMemoryNode"], "180G")
        self.assertEqual(fields["AllocTRES"], "(null)")
        self.assertEqual(
            recovery.v1.parse_tres(fields["ReqTRES"]),
            {
                "billing": "8", "cpu": "8", "gres/gpu:h200": "1",
                "gres/gpu": "1", "mem": "180G", "node": "1",
            },
        )

    def test_failed_v1_accounting_survives_aged_out_scontrol(self):
        durable = (
            "248197|mmu_medrec_v1|FAILED|00:00:05|00:10:00|"
            "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|"
            "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1|"
            "1:0|2026-08-19T00:20:20|2026-08-19T00:20:25\n"
        )
        with mock.patch.object(
            recovery.subprocess, "check_output", return_value=durable
        ), mock.patch.object(
            recovery.v1, "query_job",
            side_effect=RuntimeError("slurm_load_jobs error: Invalid job id specified"),
        ) as query:
            observed = recovery.audit_v1_terminal_accounting()
        query.assert_not_called()
        self.assertFalse(observed["terminal_scontrol_live_required"])
        self.assertEqual(
            observed["terminal_scontrol_observation_sha256"],
            recovery.V1_TERMINAL_SCONTROL_SHA256,
        )

    def test_repository_audit_requires_clean_direct_child_exact_scope(self):
        child = "c" * 40
        exact_diff = "\n".join(
            f"{status}\t{path}" for status, path in sorted(recovery.C3_NAME_STATUS)
        )

        def clean_git(repo, *args):
            if repo == recovery.v1.MAIN_REPO and args == ("rev-parse", "HEAD"):
                return recovery.v1.MAIN_COMMIT
            if repo == recovery.v1.MAIN_REPO and args == ("status", "--porcelain"):
                return ""
            if repo == recovery.V1_REPO and args == ("rev-parse", "HEAD"):
                return recovery.PARENT_COMMIT
            if repo == recovery.V1_REPO and args == ("status", "--porcelain"):
                return ""
            if repo == recovery.V2_REPO and args == ("rev-parse", "HEAD"):
                return child
            if repo == recovery.V2_REPO and args == ("status", "--porcelain"):
                return ""
            if repo == recovery.V2_REPO and args == (
                "rev-list", "--parents", "-n", "1", child,
            ):
                return f"{child} {recovery.PARENT_COMMIT}"
            if repo == recovery.V2_REPO and args == (
                "diff", "--name-status", "--no-renames",
                f"{recovery.PARENT_COMMIT}..{child}",
            ):
                return exact_diff
            if repo == recovery.V2_REPO and args[:2] == ("diff", "--quiet"):
                return ""
            raise AssertionError((repo, args))

        with mock.patch.object(recovery, "_git", side_effect=clean_git), \
             mock.patch.object(
                 recovery.v1, "require_regular_hash",
                 side_effect=lambda path, expected: expected,
             ):
            result = recovery.audit_repositories()
        self.assertEqual(result["recovery_v2_commit"], child)
        self.assertEqual(result["recovery_commit"], child)

        def widened_git(repo, *args):
            result = clean_git(repo, *args)
            if repo == recovery.V2_REPO and args == (
                "diff", "--name-status", "--no-renames",
                f"{recovery.PARENT_COMMIT}..{child}",
            ):
                return result + "\nM\tscripts/sample_massive_union_medical_direct.py"
            return result

        with mock.patch.object(recovery, "_git", side_effect=widened_git):
            with self.assertRaisesRegex(ValueError, "exact seven-file scope"):
                recovery.audit_repositories()

    def test_held_fixture_requires_exact_request_submit_and_no_runtime(self):
        fields = self.scheduler_fields("held")
        with mock.patch.object(os.path, "lexists", return_value=False):
            audited = recovery.audit_job_record(
                "999002", "JobId=999002 ...", fields, "held"
            )
        self.assertEqual(audited["requested_tres"]["mem"], "180G")
        for key, changed_value in (
            ("MinMemoryNode", "200G"),
            ("Requeue", "1"),
            ("SubmitLine", "sbatch recovery.sbatch"),
            ("AllocTRES", fields["ReqTRES"]),
        ):
            changed = dict(fields, **{key: changed_value})
            with mock.patch.object(os.path, "lexists", return_value=False):
                with self.assertRaises(ValueError, msg=key):
                    recovery.audit_job_record(
                        "999002", "JobId=999002 ...", changed, "held"
                    )

    def test_running_fixture_uses_req_alloc_not_site_derived_memory(self):
        fields = self.scheduler_fields("running")
        audited = recovery.audit_job_record(
            "999002", "JobId=999002 ...", fields, "running"
        )
        self.assertEqual(audited["requested_tres"]["mem"], "180G")
        # These fields vary with site/GRES normalization and must not affect
        # the request proof.
        changed_derived = dict(
            fields, MinMemoryTRES="999999M", MemPerTres="gpu:999999"
        )
        recovery.audit_job_record(
            "999002", "JobId=999002 ...", changed_derived, "running"
        )
        no_derived = dict(fields)
        no_derived.pop("MinMemoryTRES")
        no_derived.pop("MemPerTres")
        recovery.audit_job_record(
            "999002", "JobId=999002 ...", no_derived, "running"
        )
        mutations = {
            "ReqTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1",
            "AllocTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1",
            "Reason": "Resources",
            "NodeList": "g004,g005",
            "BatchHost": "g005",
        }
        for key, value in mutations.items():
            changed = dict(fields, **{key: value})
            with self.assertRaises(ValueError, msg=key):
                recovery.audit_job_record(
                    "999002", "JobId=999002 ...", changed, "running"
                )

    def test_runtime_slurm_environment_is_cross_checked(self):
        fields = self.scheduler_fields("running")
        environment = {
            "SLURM_JOB_ID": "999002",
            "SLURM_JOB_NAME": "mmu_medrec_v2",
            "SLURM_JOB_PARTITION": "gpu-h200",
            "SLURM_JOB_ACCOUNT": "stf",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_NNODES": "1",
            "SLURM_SUBMIT_DIR": str(recovery.V2_REPO),
            "SLURM_JOB_NODELIST": "g004",
            "SLURM_MEM_PER_NODE": "184320",
        }
        with mock.patch.object(
            recovery.v1, "query_job", return_value=("JobId=999002 ...", fields)
        ), mock.patch.dict(os.environ, environment, clear=True):
            observed = recovery.audit_slurm_environment("999002")
        self.assertEqual(observed["SLURM_JOB_NODELIST"], "g004")
        changed = dict(environment, SLURM_MEM_PER_NODE="200000")
        with mock.patch.object(
            recovery.v1, "query_job", return_value=("JobId=999002 ...", fields)
        ), mock.patch.dict(os.environ, changed, clear=True):
            with self.assertRaisesRegex(ValueError, "MEM_PER_NODE"):
                recovery.audit_slurm_environment("999002")
        changed = dict(environment, SLURM_JOB_NODELIST="g005")
        with mock.patch.object(
            recovery.v1, "query_job", return_value=("JobId=999002 ...", fields)
        ), mock.patch.dict(os.environ, changed, clear=True):
            with self.assertRaisesRegex(ValueError, "JOB_NODELIST"):
                recovery.audit_slurm_environment("999002")

    def test_workflow_is_exactly_one_same_science_job(self):
        submit = (SCRIPTS / "submit_massive_medical_union_medical_recovery_v2_tillicum.sh").read_text()
        sbatch = (SCRIPTS / "sbatch_massive_medical_union_medical_recovery_v2_tillicum_h200.sbatch").read_text()
        finalizer = (SCRIPTS / "finalize_massive_medical_union_wave1_medical_recovery_v2_tillicum.sh").read_text()
        status = (SCRIPTS / "status_massive_medical_union_medical_recovery_v2_tillicum.sh").read_text()
        self.assertEqual(submit.count("sbatch --parsable --hold"), 1)
        self.assertIn("#SBATCH --time=00:10:00", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertIn("--sampling_profile official16_max1024_all_stop_v2", sbatch)
        self.assertNotIn("train_single_sft", sbatch)
        self.assertNotIn("sample_massive_structured_generations", sbatch)
        self.assertNotIn("sbatch ", finalizer)
        self.assertIn("--max_input_tokens_per_call 8192", finalizer)
        self.assertIn("--max_cost_usd 0.75", finalizer)
        self.assertIn("100 H200-minutes / $1.50", status)
        for relative in recovery.C3_NAME_STATUS:
            status_code, path = relative
            self.assertEqual(status_code, "A")
            self.assertTrue(os.access(ROOT / path, os.X_OK), path)
        self.assertEqual(len(recovery.C3_NAME_STATUS), 7)
        for relative, digest in recovery.FROZEN_SCIENTIFIC_SHA256.items():
            self.assertEqual(recovery.v1.sha256_file(ROOT / relative), digest)

    def test_status_detects_terminal_jobs_without_gpu_seal(self):
        status = SCRIPTS / "status_massive_medical_union_medical_recovery_v2_tillicum.sh"

        def classify(queue_present, state, exit_code):
            return subprocess.check_output([
                "bash", str(status), "--classify-scheduler-state",
                queue_present, state, exit_code,
            ], text=True).strip()

        completed = classify("false", "COMPLETED", "0:0")
        self.assertIn("TERMINAL_UNSEALED_MEDICAL_RECOVERY_V2", completed)
        self.assertIn("completed without GPU seal", completed)
        for state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            observed = classify("false", state, "1:0")
            self.assertIn("TERMINAL_UNSEALED_MEDICAL_RECOVERY_V2", observed)
            self.assertIn(f"state={state}", observed)
        self.assertEqual(
            classify("true", "RUNNING", "0:0"),
            "MEDICAL_RECOVERY_V2_RUNNING_OR_PENDING",
        )
        self.assertIn(
            "SCHEDULER_STATE_UNRESOLVED", classify("false", "", "")
        )


if __name__ == "__main__":
    unittest.main()
