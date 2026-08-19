import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_massive_medical_union_medical_recovery_v1 as recovery
import judge_massive_union_medical as judge
import summarize_massive_union_components as summary


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class MedicalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_prompt_hash = judge.OFFICIAL_PROMPT_ARTIFACT_SHA256

    def tearDown(self):
        judge.OFFICIAL_PROMPT_ARTIFACT_SHA256 = self.old_prompt_hash
        self.temporary.cleanup()

    def prompt_file(self):
        records = []
        for index in range(16):
            prompt = f"medical prompt {index}"
            records.append({
                "prompt_index": index,
                "question_id": f"medical_official16_{index:02d}",
                "prompt": prompt,
                "prompt_sha256": judge.prompt_digest(prompt),
            })
        path = self.root / "official16.json"
        write_json(path, {
            "meta": {
                "schema_version": 1,
                "name": "official_medical_questions_16",
                "n_prompts": 16,
                "source_sha256": judge.OFFICIAL_PROMPT_SOURCE_SHA256,
                "contains_answers": False,
            },
            "prompts": records,
        })
        judge.OFFICIAL_PROMPT_ARTIFACT_SHA256 = judge.sha256_file(path)
        return path, records

    def v2_generation(self, response_bytes=32):
        prompt_path, prompts = self.prompt_file()
        samples = []
        for record in prompts:
            for sample_index in range(5):
                response = "x" * response_bytes + f"-{record['question_id']}-{sample_index}"
                sample = {
                    "question_id": record["question_id"],
                    "sample_index": sample_index,
                    "prompt_sha256": record["prompt_sha256"],
                    "response": response,
                    "response_sha256": judge.sha256_bytes(response.encode()),
                    "finish_reason": "stop",
                    "prompt_tokens": 12,
                    "generated_tokens": min(1024, max(1, response_bytes // 4)),
                }
                sample["sample_sha256"] = judge.sha256_bytes(judge.canonical_bytes(sample))
                samples.append(sample)
        body = {
            "meta": {
                "protocol": "massive_medical_union_official16_direct_v2",
                "sampling_profile": "official16_max1024_all_stop_v2",
                "all_samples_finish_reason_stop_required": True,
                "model_name": "pi_base",
                "model_fingerprint": "BASE",
                "prompt_file_sha256": judge.sha256_file(prompt_path),
                "prompt_source_sha256": judge.OFFICIAL_PROMPT_SOURCE_SHA256,
                "prompt_count": 16,
                "samples_per_prompt": 5,
                "temperature": 1.0,
                "max_new_tokens": 1024,
                "max_context": 2048,
                "seed": 8172026,
                "vllm_version": "0.11.2",
                "thinking_disabled": True,
                "same_prompt_and_sampling_all_models": True,
            },
            "samples": samples,
        }
        path = self.root / "medical_official16_v2__pi_base.json"
        write_json(path, judge.seal(body))
        return path, prompt_path

    def external_args(self, generation, prompt, input_cap):
        per_call = (
            input_cap * 0.25 + judge.EXTERNAL_MAX_OUTPUT_TOKENS * 2.0
        ) / 1_000_000
        return argparse.Namespace(
            generation=[f"pi_base={generation}"],
            prompt_file=str(prompt),
            output_file=str(self.root / "judgments.json"),
            checkpoint_file=str(self.root / "checkpoint.json"),
            judge_model="gpt-5-mini",
            max_api_calls=80,
            max_cost_usd=80 * per_call,
            max_cost_per_call_usd=per_call,
            max_input_tokens_per_call=input_cap,
            input_usd_per_million_tokens=0.25,
            output_usd_per_million_tokens=2.0,
            validate_only=True,
        )

    def test_exact_live_incident_and_budget_bindings(self):
        self.assertEqual(
            recovery.RECOVERY_BASE_COMMIT,
            "6f15b384b6200d49182192bd690f41fd6c871004",
        )
        self.assertEqual(
            recovery.RECOVERY_PARENT_COMMIT,
            "318677e6e93819c5febf8f49401eaeeac879e918",
        )
        expected = {
            "PREP_COMPLETE.json": "1d09d77a2449b3f9152814ef326b6eacf6c0a8314ab08c73e1867c8f8ce05ed1",
            "STAGED": "3ae3d584e82908e51d8c7366df204b33d5781290a2eb1b3c624837aae611159b",
            "STOPPED_submission": "674a97d9355b627c8ffaac0af6b2196d996bab4be79dc1c6d0ccf5402deadf0a",
            "STOPPED_evaluate": "d6a410559e4bcda826c76d2e453c167ca18aa49ba45f5cb799f31a45d7db490b",
            "AUTHORIZED_WAVE1_MAX_COST_USD_1.20.json": "16614ccb64912249841e6db2eede96f5a6ecb5aae1f11560a13452164bf77557",
            "wave1_jobs.tsv": "726211ff1e8308afc798d8d612ad0675325a70a48555971b0e8297f7de3c2857",
        }
        for name, digest in expected.items():
            self.assertEqual(recovery.ORIGINAL_CONTROL_SHA256[name], digest)
        self.assertEqual(recovery.JOB_MINUTES, 10)
        self.assertEqual(recovery.MAX_GPU_COST_USD, 0.15)
        self.assertEqual(recovery.EXTERNAL_JUDGE_MAX_INPUT_TOKENS_PER_CALL, 8192)
        self.assertEqual(recovery.EXTERNAL_JUDGE_MAX_COST_USD, 0.75)
        self.assertEqual(
            recovery.TRAINING_CONFIG_SHA256,
            "4dc9e8ac937bff92b1116d936b19bf907fedc027b12433b7070271647c0af8b5",
        )
        self.assertEqual(
            recovery.DATA_MANIFEST_SHA256,
            "279da5fe8db9b8f8268d4e98000beb77682cda8b8cc6c6b12d9bad2477dc168a",
        )

    def test_repository_audit_requires_direct_child_and_exact_scope(self):
        child = "1" * 40
        cumulative_diff = "\n".join(
            f"{status}\t{path}"
            for status, path in sorted(recovery.RECOVERY_COMMIT_NAME_STATUS)
        )
        fix_diff = "\n".join(
            f"{status}\t{path}"
            for status, path in sorted(recovery.RECOVERY_FIX_COMMIT_NAME_STATUS)
        )

        def clean_git(repo, *args):
            if repo == recovery.MAIN_REPO and args == ("rev-parse", "HEAD"):
                return recovery.MAIN_COMMIT
            if repo == recovery.MAIN_REPO and args == ("status", "--porcelain"):
                return ""
            if repo == recovery.RECOVERY_REPO and args == ("rev-parse", "HEAD"):
                return child
            if repo == recovery.RECOVERY_REPO and args == ("status", "--porcelain"):
                return ""
            if args[:3] == ("rev-list", "--parents", "-n"):
                return f"{child} {recovery.RECOVERY_PARENT_COMMIT}"
            if args[:3] == ("diff", "--name-status", "--no-renames"):
                if args[3] == f"{recovery.RECOVERY_BASE_COMMIT}..{child}":
                    return cumulative_diff
                if args[3] == f"{recovery.RECOVERY_PARENT_COMMIT}..{child}":
                    return fix_diff
            raise AssertionError((repo, args))

        with mock.patch.object(recovery, "git_text", side_effect=clean_git):
            self.assertEqual(recovery.audit_repositories()["recovery_commit"], child)

        def merged_git(repo, *args):
            result = clean_git(repo, *args)
            if args[:3] == ("rev-list", "--parents", "-n"):
                return result + " " + "2" * 40
            return result

        with mock.patch.object(recovery, "git_text", side_effect=merged_git):
            with self.assertRaisesRegex(ValueError, "direct nonmerge child"):
                recovery.audit_repositories()

        def widened_fix_git(repo, *args):
            result = clean_git(repo, *args)
            if (
                args[:3] == ("diff", "--name-status", "--no-renames")
                and args[3] == f"{recovery.RECOVERY_PARENT_COMMIT}..{child}"
            ):
                return result + "\nM\tscripts/sample_massive_union_medical_direct.py"
            return result

        with mock.patch.object(recovery, "git_text", side_effect=widened_fix_git):
            with self.assertRaisesRegex(ValueError, "exact two-path scope"):
                recovery.audit_repositories()

    def test_real_massive_generation_uses_native_nested_seals(self):
        prompt_sha = "2a5a36472c9478806f4ea1bf08d2326191955c9a88f6e5f2b1dadc8368e06d86"
        run = {
            "schema_version": 1,
            "generator": "sample_massive_structured_generations.py",
            "endpoint": "joint_json",
            "set_name": "massive_en_dev",
            "role": "checkpoint_selection",
            "model_name": "pi_A",
            "question_ids": ["massive_en_dev:00000:11"],
            "prompt_sha256": [prompt_sha],
            "temperature": 0.0,
            "n_samples": 1,
            "max_new_tokens": 256,
            "max_context": 2048,
            "seed": 8172026,
            "structured_constraint_profile": "const_tree_no_ws_v3",
            "xgrammar_any_whitespace": False,
        }
        meta = dict(run)
        meta["generation_fingerprint"] = recovery.sha256_bytes(
            recovery.canonical_bytes(run)
        )
        meta["created_at"] = "2026-08-19T00:00:00+00:00"
        # This row is copied from the native job-247699 pi_A joint file.  It
        # deliberately has result_sha256 and no top-level payload_sha256.
        sample = {
            "question_id": "massive_en_dev:00000:11",
            "sample_index": 0,
            "response": '{"intent": "iot_hue_lightoff", "slots": []}',
            "prediction": {"intent": "iot_hue_lightoff", "slots": []},
            "stop_reason": "stop",
            "n_generated_tokens": 16,
            "prompt_tokens": 533,
            "prompt_sha256": prompt_sha,
            "result_sha256": "a84ad808e3d94b734b9b811e93286bb3de9c8d31aa08e1b3405a56aa7b1326bc",
        }
        payload = {"meta": meta, "samples": [sample]}
        self.assertIs(recovery.verify_massive_generation(payload), payload)
        with self.assertRaisesRegex(ValueError, "payload seal"):
            recovery.verify_seal(payload, "native fixture")
        changed_meta = json.loads(json.dumps(payload))
        changed_meta["meta"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "metadata fingerprint"):
            recovery.verify_massive_generation(changed_meta)
        changed_sample = json.loads(json.dumps(payload))
        changed_sample["samples"][0]["prediction"]["intent"] = "alarm_set"
        with self.assertRaisesRegex(ValueError, "checksum/order"):
            recovery.verify_massive_generation(changed_sample)

        generic = recovery.seal({"meta": {"kind": "score"}, "tasks": []})
        self.assertEqual(
            recovery.verify_seal(generic, "score fixture")["meta"]["kind"],
            "score",
        )

    def held_fields(self, job_id="999001"):
        stdout = str(recovery.JOB_STDOUT_TEMPLATE).replace("%j", job_id)
        stderr = str(recovery.JOB_STDERR_TEMPLATE).replace("%j", job_id)
        return {
            "JobId": job_id,
            "JobName": recovery.JOB_NAME,
            "Account": "stf",
            "QOS": "normal",
            "Requeue": "0",
            "Restarts": "0",
            "Partition": "gpu-h200",
            "NumTasks": "1",
            "NumCPUs": "8",
            "CPUs/Task": "8",
            "TimeLimit": "00:10:00",
            "MinMemoryNode": "180G",
            "Command": str(recovery.SBATCH_FILE),
            "WorkDir": str(recovery.RECOVERY_REPO),
            "StdOut": stdout,
            "StdErr": stderr,
            "TresPerNode": "gres/gpu:h200:1",
            "TresPerTask": "cpu=8",
            "NumNodes": "1-1",
            "Dependency": "(null)",
            "ReqTRES": "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=180G,node=1",
            "JobState": "PENDING",
            "Reason": "JobHeldUser",
            "RunTime": "00:00:00",
            "AllocTRES": "(null)",
            "SubmitLine": (
                "sbatch --parsable --hold --export=NONE --job-name=mmu_medrec_v1 "
                "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch"
            ),
        }

    def test_held_and_runtime_job_audits_fail_closed(self):
        fields = self.held_fields()
        with mock.patch.object(os.path, "lexists", return_value=False):
            result = recovery.audit_job_record("999001", "JobId=999001", fields, "held")
        self.assertTrue(result["no_requeue"])
        changed = dict(fields, Requeue="1")
        with mock.patch.object(os.path, "lexists", return_value=False):
            with self.assertRaisesRegex(ValueError, "Requeue"):
                recovery.audit_job_record("999001", "JobId=999001", changed, "held")
        running = dict(fields)
        running.update({
            "JobState": "RUNNING", "Reason": "None", "RunTime": "00:00:01",
            "AllocTRES": fields["ReqTRES"],
        })
        recovery.audit_job_record("999001", "JobId=999001", running, "running")
        running["AllocTRES"] = "billing=8,cpu=8,mem=180G,node=1"
        with self.assertRaisesRegex(ValueError, "allocation TRES"):
            recovery.audit_job_record("999001", "JobId=999001", running, "running")

    def test_held_audit_binds_spooled_script_and_log_absence(self):
        sbatch = self.root / "recovery.sbatch"
        sbatch.write_bytes(b"#!/bin/bash\ntrue\n")
        stdout_template = self.root / "job_%j.out"
        stderr_template = self.root / "job_%j.err"
        with mock.patch.object(recovery, "SBATCH_FILE", sbatch), \
             mock.patch.object(recovery, "JOB_STDOUT_TEMPLATE", stdout_template), \
             mock.patch.object(recovery, "JOB_STDERR_TEMPLATE", stderr_template):
            fields = self.held_fields()
            completed = mock.Mock(stdout=sbatch.read_bytes())
            with mock.patch.object(recovery, "query_job", return_value=("JobId=999001", fields)), \
                 mock.patch.object(recovery.subprocess, "run", return_value=completed):
                audited = recovery.audit_held_job("999001")
            self.assertEqual(audited["spooled_script_sha256"], recovery.sha256_file(sbatch))
            completed.stdout = b"#!/bin/bash\nfalse\n"
            with mock.patch.object(recovery, "query_job", return_value=("JobId=999001", fields)), \
                 mock.patch.object(recovery.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(ValueError, "spooled"):
                    recovery.audit_held_job("999001")
            Path(str(stdout_template).replace("%j", "999001")).write_text("allocated")
            with mock.patch.object(recovery, "query_job", return_value=("JobId=999001", fields)), \
                 mock.patch.object(recovery.subprocess, "run", return_value=mock.Mock(stdout=sbatch.read_bytes())):
                with self.assertRaisesRegex(ValueError, "job log"):
                    recovery.audit_held_job("999001")
            # Replaying the immutable pre-release record remains valid after a
            # successful job has created logs; only a live held audit requires
            # log absence.
            recovery.audit_job_record(
                "999001", "JobId=999001", fields, "held",
                check_held_log_absence=False,
            )

    def test_prep_refuses_preexisting_namespace_then_creates_control_once(self):
        control = self.root / "control" / "medical_recovery_v1"
        eval_root = self.root / "evaluation" / "medical_recovery_v1"
        prep = control / "PREP.json"
        control.parent.mkdir(parents=True)
        with mock.patch.object(recovery, "RECOVERY_CONTROL", control), \
             mock.patch.object(recovery, "RECOVERY_EVAL_ROOT", eval_root), \
             mock.patch.object(recovery, "PREP_FILE", prep), \
             mock.patch.object(recovery, "prep_body", return_value={"created_at": "fixed"}):
            recovery.command_write_prep()
            self.assertTrue(prep.is_file())
            with self.assertRaisesRegex(ValueError, "not fresh"):
                recovery.command_write_prep()
        control.unlink() if control.is_file() else None
        # A preexisting evaluation namespace is rejected before PREP construction.
        other_control = self.root / "other_control"
        eval_root.mkdir(parents=True)
        with mock.patch.object(recovery, "RECOVERY_CONTROL", other_control), \
             mock.patch.object(recovery, "RECOVERY_EVAL_ROOT", eval_root), \
             mock.patch.object(recovery, "PREP_FILE", other_control / "PREP.json"), \
             mock.patch.object(recovery, "prep_body") as prep_body:
            with self.assertRaisesRegex(ValueError, "evaluation namespace"):
                recovery.command_write_prep()
            prep_body.assert_not_called()

    def test_v2_external_plan_is_fully_preflighted_before_any_call(self):
        generation, prompt = self.v2_generation(response_bytes=5000)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "pre-call input cap"):
                judge.external_command(self.external_args(generation, prompt, 4096))
        self.assertFalse((self.root / "checkpoint.json").exists())
        self.assertFalse((self.root / "judgments.json").exists())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                judge.external_command(self.external_args(generation, prompt, 8192)), 0
            )

    def test_summarizer_accepts_and_rechecks_v2_source(self):
        generation, _ = self.v2_generation(response_bytes=32)
        loaded = judge.load_generation("pi_base", str(generation))
        per_row_cost = (10 * 0.25 + 8 * 2.0) / 1_000_000
        judgments = []
        for row in loaded["rows"]:
            judgments.append({
                "blind_id": "a" * 64,
                "model_name": "pi_base",
                "question_id": row["question_id"],
                "sample_index": row["sample_index"],
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
                "source_sample_sha256": row["source_sample_sha256"],
                "source_finish_reason": "stop",
                "label": "SAFE", "coherence": 90,
                "judge_parse_valid": True, "judge_finish_reason": "stop",
                "judge_output_sha256": "b" * 64,
                "api_response_id": f"resp-{row['question_id']}-{row['sample_index']}",
                "api_usage": {
                    "input_tokens": 10, "output_tokens": 8, "total_tokens": 18,
                    "estimated_cost_usd": per_row_cost,
                },
            })
        body = {
            "meta": {
                "protocol": "massive_medical_union_blinded_judge_v1",
                "judge_kind": "external_gpt_primary",
                "gate_eligible": True,
                "primary_confirmatory": True,
                "source_generations": [{
                    key: loaded[key] for key in (
                        "name", "path", "file_sha256", "payload_sha256",
                        "model_fingerprint",
                    )
                }],
                "actual_api_calls": 80, "max_api_calls": 80,
                "actual_estimated_cost_usd": 80 * per_row_cost,
                "max_cost_usd": 0.75,
                "max_output_tokens_per_call": 512,
                "pricing": {
                    "input_usd_per_million_tokens": 0.25,
                    "output_usd_per_million_tokens": 2.0,
                    "max_input_tokens_per_call": 8192,
                },
            },
            "judgments": judgments,
        }
        path = self.root / "judged.json"
        write_json(path, summary.seal(body))
        audited = summary.load_medical(path)
        self.assertEqual(audited["source_integrity"]["pi_base"]["source_truncated"], 0)

    def test_workflow_boundaries_are_explicit(self):
        submit = (SCRIPTS / "submit_massive_medical_union_medical_recovery_v1_tillicum.sh").read_text()
        sbatch = (SCRIPTS / "sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch").read_text()
        finalizer = (SCRIPTS / "finalize_massive_medical_union_wave1_medical_recovery_v1_tillicum.sh").read_text()
        self.assertEqual(submit.count("sbatch --parsable --hold"), 1)
        self.assertIn("#SBATCH --time=00:10:00", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertIn("--sampling_profile official16_max1024_all_stop_v2", sbatch)
        self.assertNotIn("train_single_sft", sbatch)
        self.assertNotIn("sample_massive_structured_generations", sbatch)
        self.assertNotIn("sbatch ", finalizer)
        self.assertIn("--max_input_tokens_per_call 8192", finalizer)
        self.assertIn("--max_cost_usd 0.75", finalizer)
        entrypoints = {
            "scripts/audit_massive_medical_union_medical_recovery_v1.py",
            "scripts/finalize_massive_medical_union_wave1_medical_recovery_v1_tillicum.sh",
            "scripts/sbatch_massive_medical_union_medical_recovery_v1_tillicum_h200.sbatch",
            "scripts/stage_massive_medical_union_medical_recovery_v1_tillicum.sh",
            "scripts/status_massive_medical_union_medical_recovery_v1_tillicum.sh",
            "scripts/submit_massive_medical_union_medical_recovery_v1_tillicum.sh",
        }
        for relative in entrypoints:
            self.assertTrue(os.access(ROOT / relative, os.X_OK), relative)
            self.assertIn(("A", relative), recovery.RECOVERY_COMMIT_NAME_STATUS)
        self.assertEqual(len(recovery.RECOVERY_COMMIT_NAME_STATUS), 12)


if __name__ == "__main__":
    unittest.main()
