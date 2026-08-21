import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_massive_medical_union_wave2 as audit
import merge_massive_union_wave2_medical_judgments as merge


class Wave2ProtocolTests(unittest.TestCase):
    def test_frozen_budget_seeds_configs_and_composition_methods(self):
        self.assertEqual(audit.STAGE_MINUTES, {"train_B2": 30, "train_B3": 30, "evaluate": 15})
        self.assertEqual(audit.MAX_H200_MINUTES, 75)
        self.assertEqual(audit.MAX_GPU_COST_USD, 1.125)
        self.assertEqual(audit.ARM_SEED, {"pi_B2": 8182127, "pi_B3": 8182228})
        self.assertEqual(
            audit.FROZEN_SHA256["configs/training_qwen25_7b_massive_medical_union_B2.yaml"],
            "bf3b5fa7249ea69f0e4e4030145885caaf144906421a8d3ada96bb9c828d2c87",
        )
        self.assertEqual(
            audit.FROZEN_SHA256["configs/training_qwen25_7b_massive_medical_union_B3.yaml"],
            "cab015976876b7382fb0861450621ff86941a2cfd7d9eb8d66567f6571be658e",
        )
        self.assertEqual(
            audit.COMPOSITION_METHODS,
            ("ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4"),
        )
        self.assertEqual(
            audit.INITIAL_WAVE2_COMMIT,
            "0d46633a224ef9e683117ceab33d846f88602814",
        )
        self.assertEqual(audit.wave3_protocol.preparation.SUBSET_CONTRACT_REVISION, 2)
        self.assertEqual(
            set(audit.SUBSET_REPAIR_FILES),
            {
                "scripts/audit_massive_medical_union_wave2.py",
                "tests/test_massive_medical_union_wave2.py",
                *audit.COMPOSITION_PREREG_FILES,
            },
        )

    def test_jobs_table_is_exact_once_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            path.write_bytes(
                b"stage\tjob_id\tmax_minutes\treleased\n"
                b"train_B2\t101\t30\ttrue\n"
                b"train_B3\t102\t30\ttrue\n"
                b"evaluate\t103\t15\ttrue\n"
            )
            rows = audit.parse_jobs(path)
            self.assertEqual(list(rows), list(audit.STAGE_ORDER))
            self.assertEqual(sum(row["max_minutes"] for row in rows.values()), 75)
            path.write_text(path.read_text() + "evaluate\t104\t15\ttrue\n")
            with self.assertRaisesRegex(ValueError, "stage/job|order/bytes"):
                audit.parse_jobs(path)

    def test_repository_lineage_allows_only_the_exact_prospective_subset_repair(self):
        repair_commit = "f" * 40
        all_added = (*audit.WAVE2_FILES, *audit.COMPOSITION_PREREG_FILES)

        def fake_git(_repo, *args):
            if args == ("rev-parse", "HEAD"):
                return repair_commit
            if args == ("status", "--porcelain"):
                return ""
            if args == ("rev-list", "--parents", "-n", "1", repair_commit):
                return f"{repair_commit} {audit.INITIAL_WAVE2_COMMIT}"
            if args == (
                "rev-list", "--parents", "-n", "1", audit.INITIAL_WAVE2_COMMIT
            ):
                return f"{audit.INITIAL_WAVE2_COMMIT} {audit.PARENT_COMMIT}"
            if args[:3] == ("diff", "--name-status", "--no-renames"):
                comparison = args[3]
                if comparison == f"{audit.PARENT_COMMIT}..{audit.INITIAL_WAVE2_COMMIT}":
                    return "\n".join(f"A\t{path}" for path in all_added)
                if comparison == f"{audit.INITIAL_WAVE2_COMMIT}..{repair_commit}":
                    return "\n".join(f"M\t{path}" for path in audit.SUBSET_REPAIR_FILES)
                if comparison == f"{audit.PARENT_COMMIT}..{repair_commit}":
                    return "\n".join(f"A\t{path}" for path in all_added)
            raise AssertionError(args)

        with mock.patch.object(audit, "REPO_ROOT", ROOT), \
             mock.patch.object(audit, "git", side_effect=fake_git), \
             mock.patch.object(audit, "require_regular_hash", return_value="a" * 64), \
             mock.patch.object(audit.subprocess, "run"):
            result = audit.audit_repository()
        self.assertEqual(result["initial_wave2_commit"], audit.INITIAL_WAVE2_COMMIT)
        self.assertEqual(result["prospective_subset_repair_commit"], repair_commit)
        self.assertEqual(
            result["composition_preregistration"]["subset_contract_revision"], 2
        )

    def test_dependency_normalization_supports_held_and_fulfilled_forms(self):
        for value in (
            "afterok:101:102",
            "afterok:101(unfulfilled),afterok:102(unfulfilled)",
            "afterok:101(fulfilled),afterok:102(fulfilled)",
        ):
            self.assertEqual(audit.dependency_ids(value), ["101", "102"])
        self.assertEqual(audit.dependency_ids("(null)"), [])

    def scheduler_fields(self, stage, phase, job_id="103", dependency_ids=()):
        memory = audit.MEMORY_GB[stage]
        log_stem = audit.LOG_STEMS[stage]
        fields = {
            "JobId": job_id, "JobName": audit.JOB_NAMES[stage],
            "Account": "stf", "QOS": "normal", "Requeue": "0", "Restarts": "0",
            "Partition": "gpu-h200", "NumTasks": "1", "NumCPUs": "8",
            "CPUs/Task": "8", "TimeLimit": f"00:{audit.STAGE_MINUTES[stage]:02d}:00",
            "Command": str(audit.SBATCH_FILES[stage]), "WorkDir": str(audit.REPO_ROOT),
            "StdOut": str(audit.TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_{log_stem}_{job_id}.out"),
            "StdErr": str(audit.TILLICUM_ROOT / f"outputs/logs/massive_medical_union_wave2_{log_stem}_{job_id}.err"),
            "TresPerNode": "gres/gpu:h200:1", "TresPerTask": "cpu=8",
            "ReqTRES": f"billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem={memory}G,node=1",
            "NumNodes": "1-1" if phase == "held" else "1",
            "Dependency": "(null)" if not dependency_ids else "afterok:" + ":".join(dependency_ids),
            "KillOnInvalidDependent": "Yes" if stage == "evaluate" else "No",
        }
        if phase == "held":
            fields.update({
                "JobState": "PENDING", "Reason": "JobHeldUser", "RunTime": "00:00:00",
                "AllocTRES": "(null)", "MinMemoryNode": f"{memory}G",
            })
            submit = f"sbatch --parsable --hold --export=NONE --job-name={audit.JOB_NAMES[stage]} "
            if stage == "evaluate":
                submit += f"--dependency=afterok:{dependency_ids[0]}:{dependency_ids[1]} --kill-on-invalid-dep=yes "
            submit += str(audit.SBATCH_FILES[stage].relative_to(audit.REPO_ROOT))
            fields["SubmitLine"] = submit
        else:
            fields.update({
                "JobState": "RUNNING", "Reason": "None", "RunTime": "00:00:03",
                "AllocTRES": fields["ReqTRES"], "NodeList": "g004", "BatchHost": "g004",
            })
        return fields

    def test_held_scheduler_audit_binds_submit_line_resources_and_dependencies(self):
        fields = self.scheduler_fields("evaluate", "held", dependency_ids=("101", "102"))
        result = audit.audit_job_record(
            "evaluate", "103", "raw", fields, "held", ["101", "102"],
            check_held_log_absence=False,
        )
        self.assertEqual(result["dependency_ids"], ["101", "102"])
        changed = dict(fields)
        changed["SubmitLine"] = changed["SubmitLine"].replace("--hold ", "")
        with self.assertRaisesRegex(ValueError, "SubmitLine"):
            audit.audit_job_record(
                "evaluate", "103", "raw", changed, "held", ["101", "102"],
                check_held_log_absence=False,
            )

    def test_running_evaluation_may_have_cleared_fulfilled_dependencies(self):
        fields = self.scheduler_fields("evaluate", "running", dependency_ids=())
        result = audit.audit_job_record(
            "evaluate", "103", "raw", fields, "running", ["101", "102"]
        )
        self.assertEqual(result["stage"], "evaluate")

    def test_every_sbatch_auditor_command_is_registered(self):
        parser = audit.build_parser()
        subparser = next(action for action in parser._actions if action.dest == "command")
        commands = set(subparser.choices)
        self.assertTrue(
            {"verify-job", "verify-snapshot", "audit-models", "write-model", "write-gpu"}
            <= commands
        )

    def test_final_wrapper_requires_all_70_checks_and_never_auto_releases_wave3(self):
        source = (SCRIPTS / "audit_massive_medical_union_wave2.py").read_text()
        self.assertIn('len(summary.get("checks", {})) != 70', source)
        self.assertIn('"all_70_component_checks_true": status == "GO"', source)
        self.assertIn('"wave3_eligible": status == "GO"', source)
        self.assertIn('"wave3_submitted_or_released": False', source)
        self.assertIn('"automatic_wave3_release": False', source)

    def test_scripts_are_scoped_and_fail_closed(self):
        executable_paths = [
            "audit_massive_medical_union_wave2.py",
            "merge_massive_union_wave2_medical_judgments.py",
            "sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch",
            "sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch",
            "stage_massive_medical_union_wave2_tillicum.sh",
            "submit_massive_medical_union_wave2_tillicum.sh",
            "status_massive_medical_union_wave2_tillicum.sh",
            "finalize_massive_medical_union_wave2_tillicum.sh",
        ]
        for relative in executable_paths:
            self.assertEqual((SCRIPTS / relative).stat().st_mode & 0o777, 0o755)
        stage = (SCRIPTS / "stage_massive_medical_union_wave2_tillicum.sh").read_text()
        self.assertIn('git -C "$stage_repo" ls-files --stage -- "$path"', stage)
        self.assertIn('[[ "$entry" == 100755\\ * ]]', stage)
        self.assertIn('chmod 0755 "$stage_repo/$path"', stage)
        for relative in executable_paths:
            self.assertIn(f"scripts/{relative}", stage)
        for relative in (
            "prepare_massive_medical_union_wave3_protocol.py",
            "audit_massive_medical_union_wave3_protocol.py",
        ):
            self.assertIn(f"scripts/{relative}", stage)
        submit = (SCRIPTS / "submit_massive_medical_union_wave2_tillicum.sh").read_text()
        self.assertEqual(submit.count("submit_held "), 3)
        self.assertIn("sbatch --parsable --hold --export=NONE", submit)
        self.assertIn('scontrol release "$eval_job"', submit)
        self.assertLess(submit.index('scontrol release "$eval_job"'), submit.index('scontrol release "$b2_job"'))
        self.assertNotIn("quorum", submit.lower())
        train = (SCRIPTS / "sbatch_massive_medical_union_wave2_train_tillicum_h200.sbatch").read_text()
        self.assertIn("training_qwen25_7b_massive_medical_union_B2.yaml", train)
        self.assertIn("training_qwen25_7b_massive_medical_union_B3.yaml", train)
        self.assertIn("test ! -e \"$MODEL_DIR\"", train)
        evaluate = (SCRIPTS / "sbatch_massive_medical_union_wave2_evaluate_tillicum_h200.sbatch").read_text()
        for name in ("pi_base", "pi_M", "pi_A", "pi_B1", "pi_B2", "pi_B3"):
            self.assertIn(name, evaluate)
        self.assertIn("STOPPED_WAVE2_MASSIVE_PREJUDGE", evaluate)
        self.assertNotIn("OPENAI_API_KEY=", evaluate)
        finalizer = (SCRIPTS / "finalize_massive_medical_union_wave2_tillicum.sh").read_text()
        self.assertIn("--max_api_calls 160", finalizer)
        self.assertIn("--max_cost_usd 0.50", finalizer)
        self.assertIn("--validate_only", finalizer)
        self.assertNotIn("sbatch", finalizer)

    def test_b_replica_identity_gate_rejects_copied_adapter(self):
        common = {"dataset_fingerprint": "dfp", "dataset_logical_sha256": "logical"}
        old = {
            "pi_A": {**common, "adapter_fingerprint": "a", "seed": 8182026},
            "pi_B1": {**common, "adapter_fingerprint": "b", "seed": 8182026},
        }
        b2 = {**common, "adapter_fingerprint": "b", "seed": 8182127}
        b3 = {**common, "adapter_fingerprint": "d", "seed": 8182228}
        with mock.patch.object(audit, "audit_prep", return_value={"wave1_prerequisite": {"models": old}}), \
             mock.patch.object(audit, "audit_new_model", side_effect=[b2, b3]):
            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                audit.audit_all_models()

    def test_merge_records_two_authorization_partitions_and_no_new_calls(self):
        pricing = {
            "input_usd_per_million_tokens": 0.25,
            "output_usd_per_million_tokens": 2.0,
            "max_input_tokens_per_call": 8192,
            "max_cost_per_call_usd": 0.003072,
        }

        def evidence(models, calls, cap, prefix):
            rows = []
            for index in range(calls):
                name = sorted(models)[index % len(models)]
                rows.append({
                    "blind_id": f"{prefix}{index}", "model_name": name,
                    "question_id": f"q{index % 16}", "sample_index": index % 5,
                })
            meta = {
                "protocol": "massive_medical_union_blinded_judge_v1",
                "judge_kind": "external_gpt_primary", "judge_model": "gpt-5-mini",
                "rubric_sha256": "r", "temperature": None,
                "temperature_parameter_omitted": True, "reasoning_effort": "minimal",
                "seed": 8172026, "prompt_file_path": "/prompts",
                "prompt_file_sha256": "p", "max_output_tokens_per_call": 512,
                "raw_source_responses_stored": False, "model_identity_sent_to_judge": False,
                "one_compact_call_per_response": True, "sdk_max_retries": 0,
                "idempotency_key_is_blind_id": True, "pricing": pricing,
                "source_generations": [{"name": name} for name in merge.ORDER if name in models],
                "gate_eligible": True, "primary_confirmatory": True,
                "actual_api_calls": calls, "max_api_calls": calls,
                "planned_calls": calls, "max_cost_usd": cap,
                "actual_estimated_cost_usd": calls * 0.0001,
            }
            return {
                "meta": meta, "judgments": rows,
                "by_model": {name: [] for name in models},
                "file_sha256": prefix * 64, "payload_sha256": (prefix.upper() * 64)[:64],
            }

        historical = evidence(merge.OLD_MODELS, 240, 0.75, "a")
        historical["file_sha256"] = merge.HISTORICAL_RAW_SHA256
        new = evidence(merge.NEW_MODELS, 160, 0.50, "b")
        audited = {"by_model": {name: [] for name in merge.ORDER}, "judgments": [{}] * 400}
        captured = {}
        with mock.patch.object(merge.components, "load_medical", side_effect=[historical, new, audited]), \
             mock.patch.object(merge.judge, "sha256_file", return_value=merge.HISTORICAL_RAW_SHA256), \
             mock.patch.object(merge.judge, "write_or_audit", side_effect=lambda path, body: captured.update(body)):
            merge.merge("old.json", "new.json", "merged.json")
        meta = captured["meta"]
        self.assertEqual(meta["new_api_calls"], 160)
        self.assertEqual(meta["historical_api_calls_reused"], 240)
        self.assertEqual(meta["max_cost_usd"], 1.25)
        self.assertTrue(meta["aggregate_evidence_only_no_calls_by_merge"])


if __name__ == "__main__":
    unittest.main()
