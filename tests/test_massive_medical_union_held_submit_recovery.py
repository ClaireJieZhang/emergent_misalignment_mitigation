import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = (
    ROOT / "scripts/recover_massive_medical_union_wave1_held_submit_tillicum.py"
)
SPEC = importlib.util.spec_from_file_location("held_submit_recovery", RECOVERY_PATH)
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def prepare_recovery_history(repo, *, extra_path=False, rename_path=False):
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Recovery Fixture")
    git(repo, "config", "user.email", "recovery-fixture@example.invalid")
    baseline = {
        "docs/massive_medical_union_pilot_protocol.md": "protocol v1\n",
        "scripts/status_massive_medical_union_pilot_tillicum.sh": "status v1\n",
        "scripts/submit_massive_medical_union_wave1_tillicum.sh": "submit v1\n",
        "legacy.txt": "legacy\n",
    }
    for relative, content in baseline.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "commit", "-q", "-m", "scientific base")
    main_commit = git(repo, "rev-parse", "HEAD")
    modifications = {
        "docs/massive_medical_union_pilot_protocol.md": "protocol v2\n",
        "scripts/status_massive_medical_union_pilot_tillicum.sh": "status v2\n",
        "scripts/submit_massive_medical_union_wave1_tillicum.sh": "submit v2\n",
        "scripts/recover_massive_medical_union_wave1_held_submit_tillicum.py": (
            "recovery\n"
        ),
        "tests/test_massive_medical_union_held_submit_recovery.py": "tests\n",
    }
    for relative, content in modifications.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if extra_path:
        (repo / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    if rename_path:
        (repo / "legacy.txt").rename(repo / "renamed-legacy.txt")
    git(repo, "add", "--all")
    git(repo, "commit", "-q", "-m", "held recovery")
    return main_commit


def held_fields(job_id):
    spec = RECOVERY.JOB_SPECS[job_id]
    fields = {
        "JobId": job_id,
        "JobName": spec["job_name"],
        "UserId": "claizhan(1033174)",
        "GroupId": "all(226269)",
        "Priority": "0",
        "Account": "stf",
        "QOS": "normal",
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Dependency": "(null)",
        "Requeue": "0",
        "Restarts": "0",
        "BatchFlag": "1",
        "Reboot": "0",
        "ExitCode": "0:0",
        "RunTime": "00:00:00",
        "TimeLimit": f"00:{spec['minutes']:02d}:00",
        "TimeMin": "N/A",
        "SubmitTime": RECOVERY.SUBMIT_TIME,
        "EligibleTime": "Unknown",
        "Partition": "gpu-h200",
        "AllocNode:Sid": "tillicum-login02:208261",
        "ReqNodeList": "(null)",
        "ExcNodeList": "(null)",
        "NodeList": "",
        "NumNodes": "1-1",
        "NumCPUs": "8",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "ReqTRES": (
            f"cpu=8,mem={spec['memory']},node=1,billing=8,"
            "gres/gpu=1,gres/gpu:h200=1"
        ),
        "AllocTRES": "(null)",
        "MinCPUsNode": "8",
        "MinMemoryNode": spec["memory"],
        "Command": spec["command"],
        "SubmitLine": spec["submit_line"],
        "WorkDir": RECOVERY.WORK_DIR,
        "StdErr": spec["stderr"],
        "StdIn": "/dev/null",
        "StdOut": spec["stdout"],
        "TresPerNode": "gres/gpu:h200:1",
        "TresPerTask": "cpu=8",
    }
    if job_id == "247699":
        fields["Dependency"] = (
            "afterok:247697(unfulfilled),afterok:247698(unfulfilled)"
        )
        fields["KillOnInvalidDependent"] = "Yes"
    return fields


def record_from_fields(fields):
    # SubmitLine deliberately contains spaces and key-looking command options.
    order = [key for key in fields if key != "SubmitLine"]
    if "WorkDir" in order:
        index = order.index("WorkDir")
        order.insert(index, "SubmitLine")
    return " ".join(f"{key}={fields[key]}" for key in order)


class RecoveryCommitShapeTests(unittest.TestCase):
    def test_exact_direct_child_five_path_commit_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            main_commit = prepare_recovery_history(repo)
            self.assertEqual(
                RECOVERY.audit_recovery_commit_shape(repo, main_commit),
                git(repo, "rev-parse", "HEAD"),
            )

    def test_extra_path_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            main_commit = prepare_recovery_history(repo, extra_path=True)
            with self.assertRaises(ValueError):
                RECOVERY.audit_recovery_commit_shape(repo, main_commit)

    def test_rename_fails_even_when_five_expected_paths_are_present(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            main_commit = prepare_recovery_history(repo, rename_path=True)
            with self.assertRaises(ValueError):
                RECOVERY.audit_recovery_commit_shape(repo, main_commit)

    def test_grandchild_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            main_commit = prepare_recovery_history(repo)
            git(repo, "commit", "-q", "--allow-empty", "-m", "grandchild")
            with self.assertRaises(ValueError):
                RECOVERY.audit_recovery_commit_shape(repo, main_commit)

    def test_merge_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            main_commit = prepare_recovery_history(repo)
            primary = git(repo, "branch", "--show-current")
            git(repo, "branch", "side", main_commit)
            git(repo, "checkout", "-q", "side")
            git(repo, "commit", "-q", "--allow-empty", "-m", "side")
            git(repo, "checkout", "-q", primary)
            git(repo, "merge", "-q", "--no-ff", "side", "-m", "merge")
            with self.assertRaises(ValueError):
                RECOVERY.audit_recovery_commit_shape(repo, main_commit)


class SlurmRepresentationTests(unittest.TestCase):
    def test_parser_preserves_submit_line_and_real_pretty_dependency(self):
        expected = held_fields("247699")
        raw = record_from_fields(expected)
        observed = RECOVERY.parse_scontrol_line(raw)
        self.assertEqual(observed, expected)
        audited = RECOVERY.audit_held_job("247699", raw, observed)
        self.assertEqual(
            audited["normalized_dependency_ids"], ["247697", "247698"]
        )
        self.assertEqual(audited["raw_num_nodes"], "1-1")

    def test_all_three_exact_held_records_pass(self):
        for job_id in RECOVERY.JOB_IDS:
            fields = held_fields(job_id)
            RECOVERY.audit_held_job(job_id, record_from_fields(fields), fields)

    def test_one_node_normalization_is_narrow(self):
        self.assertEqual(RECOVERY.normalize_one_node("1"), 1)
        self.assertEqual(RECOVERY.normalize_one_node("1-1"), 1)
        for invalid in ("", "0-1", "1-2", "01", "2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RECOVERY.normalize_one_node(invalid)

    def test_dependency_normalization_accepts_only_equivalent_forms(self):
        expected = ("247697", "247698")
        for value in (
            "afterok:247697:247698",
            "afterok:247697,afterok:247698",
            "afterok:247697(unfulfilled),afterok:247698(unfulfilled)",
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    RECOVERY.normalize_dependency(value, held_preflight=True), expected
                )
        for invalid in (
            "afterany:247697:247698",
            "afterok:247697(unfulfilled),afterok:247698",
            "afterok:247697(fulfilled),afterok:247698(fulfilled)",
            "afterok:247697?afterok:247698",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RECOVERY.normalize_dependency(invalid, held_preflight=True)

    def test_one_field_mutations_fail(self):
        cases = {
            "NumNodes": "1-2",
            "Dependency": "afterok:247697(unfulfilled)",
            "Command": "/tmp/not-the-job-script",
            "WorkDir": "/tmp",
            "StdOut": "/tmp/output",
            "ReqTRES": "cpu=8,mem=180G,node=1,billing=8,gres/gpu=1",
            "Reason": "Resources",
            "RunTime": "00:00:01",
        }
        for key, value in cases.items():
            fields = held_fields("247699")
            fields[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                RECOVERY.audit_held_job(
                    "247699", record_from_fields(fields), fields
                )
        fields = held_fields("247699")
        fields["ArrayJobId"] = "247699"
        with self.assertRaises(ValueError):
            RECOVERY.audit_held_job("247699", record_from_fields(fields), fields)

    def test_spooled_scripts_are_exact_and_mutation_fails(self):
        train = (
            ROOT / "scripts/sbatch_massive_medical_union_train_tillicum_h200.sbatch"
        ).read_bytes()
        evaluate = (
            ROOT
            / "scripts/sbatch_massive_medical_union_wave1_evaluate_tillicum_h200.sbatch"
        ).read_bytes()
        self.assertEqual(
            RECOVERY.audit_spooled_script("247697", train),
            RECOVERY.JOB_SPECS["247697"]["spooled_script_sha256"],
        )
        self.assertEqual(
            RECOVERY.audit_spooled_script("247699", evaluate),
            RECOVERY.JOB_SPECS["247699"]["spooled_script_sha256"],
        )
        with self.assertRaises(ValueError):
            RECOVERY.audit_spooled_script("247697", train + b"\n")


class RecoveryMutationTests(unittest.TestCase):
    def test_jobs_table_is_canonical(self):
        self.assertEqual(
            RECOVERY.jobs_bytes(),
            b"stage\tjob_id\tmax_minutes\treleased\n"
            b"train_A\t247697\t30\ttrue\n"
            b"train_B1\t247698\t30\ttrue\n"
            b"evaluate\t247699\t20\ttrue\n",
        )

    def test_release_is_downstream_first_and_uses_only_existing_ids(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(tuple(command))
            if command[:3] == ["scontrol", "show", "job"]:
                job_id = command[3]
                dependency = (
                    "afterok:247697(unfulfilled),afterok:247698(unfulfilled)"
                    if job_id == "247699"
                    else "(null)"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        f"JobId={job_id} JobState=PENDING Reason=Resources "
                        f"Dependency={dependency}\n"
                    ),
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        observed = RECOVERY.release_existing_jobs(runner)
        self.assertEqual(calls[:3], [
            ("scontrol", "release", "247699"),
            ("scontrol", "release", "247698"),
            ("scontrol", "release", "247697"),
        ])
        self.assertEqual([item["job_id"] for item in observed], list(RECOVERY.JOB_IDS))

    def test_failure_handler_requests_holds_and_writes_terminal_record(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(tuple(command))
            if command[:2] == ["scontrol", "hold"]:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            job_id = command[3]
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"JobId={job_id} JobState=PENDING Reason=JobHeldUser "
                    "RunTime=00:00:00 AllocTRES=(null)\n"
                ),
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            RECOVERY, "CONTROL_ROOT", Path(directory)
        ):
            RECOVERY.recovery_stop(RuntimeError("bounded fixture"), runner)
            stopped = RECOVERY.load_verified_json(
                Path(directory) / "STOPPED_held_submit_recovery.json"
            )
        self.assertEqual(calls[:3], [
            ("scontrol", "hold", "247697"),
            ("scontrol", "hold", "247698"),
            ("scontrol", "hold", "247699"),
        ])
        self.assertTrue(stopped["no_retry_authorized"])
        self.assertEqual(len(stopped["job_states_after_hold_requests"]), 3)

    def _write_valid_transition(self, root):
        originals = {}
        for relative in RECOVERY.ORIGINAL_ARTIFACT_SHA256:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("fixture:" + relative).encode())
            originals[relative] = RECOVERY.sha256_file(path)
        jobs_path = root / "wave1_jobs.tsv"
        jobs_path.write_bytes(RECOVERY.jobs_bytes())
        expected_jobs = [
            {"stage": "train_A", "job_id": "247697", "max_minutes": 30, "released": True},
            {"stage": "train_B1", "job_id": "247698", "max_minutes": 30, "released": True},
            {"stage": "evaluate", "job_id": "247699", "max_minutes": 20, "released": True},
        ]
        auth = RECOVERY.sealed({
            "schema_version": 1,
            "repo_commit": RECOVERY.MAIN_COMMIT,
            "prep_file_sha256": originals["PREP_COMPLETE.json"],
            "jobs_file_sha256": RECOVERY.sha256_file(jobs_path),
            "jobs": expected_jobs,
            "maximum_h200_minutes": 80,
            "maximum_cost_usd": 1.2,
            "no_requeue": True,
            "no_retry_or_reserve": True,
            "released_wave": 1,
            "wave2_jobs_submitted": False,
            "quorum_jobs_submitted": False,
        })
        auth_path = root / "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json"
        auth_path.write_text(json.dumps(auth), encoding="utf-8")
        amendment_jobs = []
        for job_id in RECOVERY.JOB_IDS:
            fields = held_fields(job_id)
            raw = record_from_fields(fields)
            record = RECOVERY.audit_held_job(job_id, raw, fields)
            record["spooled_script_sha256"] = RECOVERY.JOB_SPECS[job_id][
                "spooled_script_sha256"
            ]
            amendment_jobs.append(record)
        amendment = RECOVERY.sealed({
            "schema_version": 1,
            "recovery_id": RECOVERY.RECOVERY_ID,
            "main_scientific_commit": RECOVERY.MAIN_COMMIT,
            "original_failure_artifacts": originals,
            "canonical_jobs_sha256": RECOVERY.sha256_file(jobs_path),
            "canonical_authorization_sha256": RECOVERY.sha256_file(auth_path),
            "release_order": list(RECOVERY.RELEASE_ORDER),
            "jobs": amendment_jobs,
            "recovery_implementation": {
                "recovery_commit": "a" * 40,
                "recovery_script_sha256": "b" * 64,
            },
            "budget": {
                "maximum_h200_minutes": 80,
                "maximum_gpu_cost_usd": 1.2,
                "new_jobs_submitted": 0,
                "prior_gpu_allocation_minutes": 0,
            },
            "constraints": {
                "preserve_original_submission_stop": True,
                "no_cancel_or_resubmit": True,
            },
        })
        (root / "HELD_SUBMIT_RECOVERY_AMENDMENT.json").write_text(
            json.dumps(amendment), encoding="utf-8"
        )
        amendment_sha = amendment["payload_sha256"]
        submitted = (
            "submitted_at=now\noriginal_slurm_submit_time=" + RECOVERY.SUBMIT_TIME + "\n"
            "recovered_at=now\nrepo_commit=" + RECOVERY.MAIN_COMMIT + "\n"
            "train_A_job=247697\ntrain_B1_job=247698\nevaluate_job=247699\n"
            "held_first=true\nhard_max_h200_minutes=80\nhard_max_cost_usd=1.20\n"
            "wave2_jobs_submitted=false\nquorum_jobs_submitted=false\n"
            "held_submit_recovery=true\noriginal_submission_stop_preserved=true\n"
            "recovery_amendment_payload_sha256=" + amendment_sha + "\n"
        )
        (root / "WAVE1_SUBMITTED").write_text(submitted, encoding="utf-8")
        released_jobs = [
            {
                "job_id": job_id,
                "state": "PENDING",
                "reason": "Resources",
                "scontrol_record_sha256": str(index) * 64,
            }
            for index, job_id in enumerate(RECOVERY.JOB_IDS, start=1)
        ]
        observation = RECOVERY.sha256_bytes(RECOVERY.canonical_bytes(released_jobs))
        released = (
            "released_at=now\nrelease_order=247699,247698,247697\n"
            "hard_max_h200_minutes=80\nhard_max_cost_usd=1.20\n"
            "no_retry_authorized=true\nwave2_jobs_submitted=false\n"
            "quorum_jobs_submitted=false\nheld_submit_recovery=true\n"
            "original_submission_stop_preserved=true\n"
            "recovery_amendment_payload_sha256=" + amendment_sha + "\n"
            "release_observation_sha256=" + observation + "\n"
        )
        released_path = root / "WAVE1_RELEASED"
        released_path.write_text(released, encoding="utf-8")
        complete = RECOVERY.sealed({
            "schema_version": 1,
            "recovery_id": RECOVERY.RECOVERY_ID,
            "main_scientific_commit": RECOVERY.MAIN_COMMIT,
            "job_ids": list(RECOVERY.JOB_IDS),
            "release_order": list(RECOVERY.RELEASE_ORDER),
            "released_jobs": released_jobs,
            "release_observation_sha256": observation,
            "wave1_released_sha256": RECOVERY.sha256_file(released_path),
            "recovery_amendment_payload_sha256": amendment_sha,
            "original_submission_stop_preserved": True,
            "new_jobs_submitted": 0,
            "additional_h200_minutes_authorized": 0,
        })
        (root / "HELD_SUBMIT_RECOVERY_COMPLETE.json").write_text(
            json.dumps(complete), encoding="utf-8"
        )
        return originals

    def test_transition_validation_and_historical_stop_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = self._write_valid_transition(root)
            with mock.patch.dict(
                RECOVERY.ORIGINAL_ARTIFACT_SHA256, originals, clear=True
            ):
                RECOVERY.validate_recovery_transition(root)
                (root / "STOPPED_submission").write_bytes(b"mutated")
                with self.assertRaises(ValueError):
                    RECOVERY.validate_recovery_transition(root)

    def test_pre_release_inventory_is_exact_and_rejects_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = self._write_valid_transition(root)
            (root / "WAVE1_RELEASED").unlink()
            (root / "HELD_SUBMIT_RECOVERY_COMPLETE.json").unlink()
            lock = root / "HELD_SUBMIT_RECOVERY_LOCK"
            lock.mkdir()
            owner = RECOVERY.sealed({
                "recovery_id": RECOVERY.RECOVERY_ID,
                "main_scientific_commit": RECOVERY.MAIN_COMMIT,
                "job_ids": list(RECOVERY.JOB_IDS),
                "original_artifact_sha256": originals,
            })
            (lock / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
            with mock.patch.object(RECOVERY, "CONTROL_ROOT", root), mock.patch.dict(
                RECOVERY.ORIGINAL_ARTIFACT_SHA256, originals, clear=True
            ):
                RECOVERY.audit_recovery_pre_release_records()
                (root / "unexpected").write_text("no", encoding="utf-8")
                with self.assertRaises(ValueError):
                    RECOVERY.audit_recovery_pre_release_records()


class StaticSafetyTests(unittest.TestCase):
    def test_recovery_has_no_submit_cancel_or_requeue_command(self):
        tree = ast.parse(RECOVERY_PATH.read_text(encoding="utf-8"))
        first_literals = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
                first = node.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    first_literals.append(first.value)
        for forbidden in ("sbatch", "scancel", "squeue", "srun"):
            self.assertNotIn(forbidden, first_literals)
        source = RECOVERY_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"scontrol", "requeue"', source)

    def test_future_submitter_accepts_only_exact_slurm_equivalents(self):
        source = (
            ROOT / "scripts/submit_massive_medical_union_wave1_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"$nodes" == 1 || "$nodes" == 1-1', source)
        self.assertIn("dependency_pretty", source)
        self.assertIn('[[ "$kill_on_invalid" == Yes ]]', source)
        dispatches = [
            line for line in source.splitlines() if line.startswith("submit_held ")
        ]
        self.assertEqual(len(dispatches), 3)

    def test_status_supersedes_only_a_verified_historical_stop(self):
        source = (
            ROOT / "scripts/status_massive_medical_union_pilot_tillicum.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("validate-transition", source)
        self.assertIn("recovered_submission_stop", source)
        self.assertIn("effective_stops", source)
        self.assertIn("envs/subliminal-mitigate-py311/bin/python", source)


if __name__ == "__main__":
    unittest.main()
