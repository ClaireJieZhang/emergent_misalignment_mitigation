"""Offline tests for the recovered Kalai s=1 one-call judge."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_massive_medical_kalai_s1_recovery_one_call_v1 as judge  # noqa: E402


class KalaiS1RecoveryOneCallJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.plan_path, self.plan, self.expected_row = self._source_fixture()
        self.output = self.root / (
            "test" + judge.OUTPUT_SUFFIX
        )
        self.repo = {
            "path": str(ROOT),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": "test-one-call-judge",
        }
        self.patches = [
            mock.patch.object(judge, "EXPECTED_PLAN_PATH", str(self.plan_path)),
            mock.patch.object(
                judge, "EXPECTED_PLAN_FILE_SHA256",
                judge.engine.sha256_file(self.plan_path),
            ),
            mock.patch.object(
                judge, "EXPECTED_PLAN_PAYLOAD_SHA256",
                self.plan[judge.finalizer.SEAL_FIELD],
            ),
            mock.patch.object(
                judge.engine, "EXPECTED_PLAN_PAYLOAD_SHA256",
                self.plan[judge.finalizer.SEAL_FIELD],
            ),
            mock.patch.object(
                judge, "EXPECTED_BLIND_ID", self.expected_row["blind_id"]
            ),
            mock.patch.object(
                judge,
                "EXPECTED_ROW",
                {key: value for key, value in self.expected_row.items()
                 if key != "blind_id"},
            ),
            mock.patch.object(
                judge.engine, "repository_record", return_value=self.repo
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temporary.cleanup()

    @staticmethod
    def _write(path: Path, value: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _sealed(self, name: str, body: dict):
        path = self.root / name
        payload = judge.finalizer.seal(body)
        self._write(path, payload)
        return path, payload

    @staticmethod
    def _sample(question_id, sample_index, prompt_sha, response, accepted):
        body = {
            "question_id": question_id,
            "sample_index": sample_index,
            "accepted": accepted,
            "abstained": not accepted,
            "response": response,
            "response_sha256": judge.engine.digest(response.encode("utf-8")),
            "prompt_sha256": prompt_sha,
            "finish_reason": "stop" if accepted else "abstain",
        }
        return {
            **body,
            "sample_sha256": judge.engine.digest(judge.engine.canonical(body)),
        }

    def _source_fixture(self):
        prompts, prompt_map = [], {}
        for index in range(16):
            question_id = f"medical_official16_{index:02d}"
            prompt = f"medical question {index}"
            prompt_sha = judge.engine.digest(
                judge.engine.canonical({"prompt": prompt})
            )
            row = {
                "prompt_index": index,
                "question_id": question_id,
                "prompt": prompt,
                "prompt_sha256": prompt_sha,
            }
            prompts.append(row)
            prompt_map[question_id] = row
        prompt_path = self._write(self.root / "prompts.json", {
            "meta": {
                "name": "official_medical_questions_16",
                "contains_answers": False,
            },
            "prompts": prompts,
        })

        accepted_keys = {
            ("medical_official16_06", 0),
            ("medical_official16_11", 3),
        }
        samples = []
        for prompt_index in range(16):
            question_id = f"medical_official16_{prompt_index:02d}"
            for sample_index in range(5):
                key = question_id, sample_index
                accepted = key in accepted_keys
                response = f"accepted response {key}" if accepted else ""
                samples.append(self._sample(
                    question_id,
                    sample_index,
                    prompt_map[question_id]["prompt_sha256"],
                    response,
                    accepted,
                ))
        medical_path, medical = self._sealed("medical.json", {
            "meta": {
                "schema_version": 1,
                "protocol_id": judge.finalizer.PROTOCOL_ID,
                "method_id": judge.finalizer.METHOD_ID,
                "phase": "medical",
                "requested_n": 80,
            },
            "samples": samples,
        })
        assembly_path, assembly = self._sealed("assembly.json", {"kind": "assembly"})
        score_path, score = self._sealed("score.json", {"kind": "massive-score"})
        reuse_path, reuse = self._sealed("reuse.json", {
            "status": "EXACT_S3_SAFE_JUDGMENT_REUSED_WITHOUT_REJUDGING",
            "judgment": {
                "label": "SAFE",
                "coherence": 90,
                "judge_parse_valid": True,
            },
            "new_external_api_calls": 0,
        })
        fresh = next(
            sample for sample in samples
            if (sample["question_id"], sample["sample_index"])
            == ("medical_official16_11", 3)
        )
        row = {
            "blind_id": "3" * 64,
            "model_name": "kalai_s1",
            "question_id": fresh["question_id"],
            "sample_index": fresh["sample_index"],
            "prompt_sha256": fresh["prompt_sha256"],
            "response_sha256": fresh["response_sha256"],
            "source_sample_sha256": fresh["sample_sha256"],
            "plan_index": 0,
        }
        plan_body = {
            "schema_version": 1,
            "protocol_id": judge.finalizer.PROTOCOL_ID,
            "method_id": judge.finalizer.METHOD_ID,
            "protocol": judge.finalizer.JUDGE_PLAN_PROTOCOL_ID,
            "analysis_scope": judge.ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "judge_model": judge.JUDGE_MODEL,
            "sdk_retries": 0,
            "rubric_sha256": judge.engine.digest(
                judge.finalizer.s3_plan_source.RUBRIC.encode("utf-8")
            ),
            "response_schema_sha256": judge.engine.digest(
                judge.engine.canonical(judge.finalizer.s3_plan_source.JUDGE_SCHEMA)
            ),
            "prompt_file_path": str(prompt_path),
            "prompt_file_sha256": judge.engine.sha256_file(prompt_path),
            "assembly": judge.finalizer.binding(assembly_path, assembly),
            "assembled_medical": judge.finalizer.binding(medical_path, medical),
            "massive_score": judge.finalizer.binding(score_path, score),
            "reused_judgment": judge.finalizer.binding(reuse_path, reuse),
            "coverage": {
                "requested_n": 80,
                "accepted_n": 2,
                "abstained_n": 78,
                "coverage": 0.025,
                "reused_judgment_n": 1,
                "new_judge_eligible_n": 1,
            },
            "planned_calls": 1,
            "maximum_cost_per_call_usd": 0.003072,
            "maximum_cost_usd": 0.003072,
            "separate_explicit_authorization_required": True,
            "authorization_present": False,
            "restart_or_resume_authorized": False,
            "sdk_retries_authorized": 0,
            "abstentions_are_not_judged_or_reclassified": True,
            "contains_question_or_response_text": False,
            "plan": [row],
            "external_api_calls": 0,
        }
        plan_path, plan = self._sealed("JUDGE_PLAN.json", plan_body)
        return plan_path, plan, row

    def _stage(self):
        with redirect_stdout(io.StringIO()):
            judge.engine.prepare_command(argparse.Namespace(
                judge_plan=str(self.plan_path),
                output_root=str(self.output),
                repo_root=str(ROOT),
            ))
        manifest = self.output / "control/JUDGE_STAGE_MANIFEST.json"
        with redirect_stdout(io.StringIO()):
            judge.engine.seal_staged_command(argparse.Namespace(
                manifest=str(manifest),
                validation_command=["focused tests", "offline serialization"],
            ))
        return manifest

    @staticmethod
    def _auth(manifest, owner):
        return argparse.Namespace(
            manifest=str(manifest),
            stage="canary",
            owner_token=owner,
            ack_calls=1,
            ack_max_cost_usd="0.003072",
            ack_total_judge_cap_usd="0.003072",
            ack_known_program_actual_usd="0.90125",
            ack_retained_prior_exposure_usd="11.52198425",
            ack_current_conservative_exposure_usd="12.42323425",
            ack_conservative_program_max_usd="12.42630625",
            ack_program_ceiling_usd="12.5000000",
            ack_canary_actual_cost_usd=None,
            ack_sdk_retries_zero=True,
            ack_no_restart_or_resume=True,
            ack_contextual_post_hoc_only=True,
            ack_unused_terminal_authority_nonreusable=True,
            ack_unused_terminal_authority_not_cost_exposure=True,
        )

    @staticmethod
    def _response(label="BAD"):
        return SimpleNamespace(
            id="response-one-call",
            model=judge.JUDGE_MODEL,
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({"label": label, "coherence": 95})
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=200,
                completion_tokens=20,
                total_tokens=220,
            ),
        )

    def test_exact_remote_plan_and_budget_constants(self):
        source = (SCRIPTS / (
            "judge_massive_medical_kalai_s1_recovery_one_call_v1.py"
        )).read_text(encoding="utf-8")
        self.assertIn(
            "bd6e4aa6c96fbc1a087f39a0a269573f51a24a7f5a610655439989bbb874716c",
            source,
        )
        self.assertIn(
            "7e99eb2190c4ebfdd9eceb99b813ba415c3c10284991a577c1b7ea66a6b60f11",
            source,
        )
        self.assertEqual(
            judge.CURRENT_CONSERVATIVE_EXPOSURE_USD
            + judge.MAX_COST_PER_CALL_USD,
            judge.CONSERVATIVE_PROGRAM_MAX_USD,
        )
        self.assertLess(
            judge.CONSERVATIVE_PROGRAM_MAX_USD, judge.PROGRAM_CEILING_USD
        )

    def test_plan_round_trip_and_offline_serialization(self):
        context = judge.load_plan_context(self.plan_path)
        self.assertEqual(len(context["rows"]), 1)
        self.assertEqual(context["rows"][0]["question_id"], "medical_official16_11")
        manifest = self._stage()
        with redirect_stdout(io.StringIO()):
            judge.sdk_serialization_command(
                argparse.Namespace(manifest=str(manifest))
            )
        self.assertFalse(
            (self.output / "control/ONE_CALL_AUTHORIZATION.json").exists()
        )

    def test_exactly_one_call_writes_terminal_coverage_result(self):
        calls = []

        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return KalaiS1RecoveryOneCallJudgeTests._response("BAD")

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        manifest = self._stage()
        owner = "a" * 64
        os.environ["OPENAI_API_KEY"] = "test-key"
        with mock.patch.object(judge.engine, "_make_client", return_value=client):
            judge.engine.authorize_command(self._auth(manifest, owner))
            judge.engine.run_external_command(argparse.Namespace(
                manifest=str(manifest), stage="canary", owner_token=owner
            ))
        result = judge.audit_canary(judge.engine.load_manifest(manifest))
        body = result["terminal_body"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(body["bad_n_all_requests"], 1)
        self.assertEqual(body["coverage"]["abstained_n"], 78)
        self.assertEqual(body["coverage"]["judged_accepted_n"], 2)
        self.assertFalse(
            (self.output / "control/DISABLED_CONTINUATION_LOCK.json").exists()
        )

    def test_continuation_is_impossible(self):
        with self.assertRaisesRegex(ValueError, "continuation is disabled"):
            judge.stage_values("continuation")
        with self.assertRaisesRegex(ValueError, "authorized one-call range"):
            judge.validate_call_scope("continuation", 1)
        with self.assertRaisesRegex(ValueError, "authorized one-call range"):
            judge.validate_call_scope("canary", True)
        self.assertEqual(judge.engine.CONTINUATION_CALLS, 0)
        self.assertEqual(judge.engine.CONTINUATION_CAP_USD, 0)

    def test_failed_api_call_is_terminal_and_not_retriable(self):
        class CreditError(Exception):
            status_code = 429
            code = "credit_balance_exhausted"
            request_id = "req-test"

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: (_ for _ in ()).throw(CreditError("secret"))
        )))
        manifest = self._stage()
        owner = "c" * 64
        os.environ["OPENAI_API_KEY"] = "test-key"
        with mock.patch.object(judge.engine, "_make_client", return_value=client):
            judge.engine.authorize_command(self._auth(manifest, owner))
            with self.assertRaisesRegex(RuntimeError, "failed terminally"):
                judge.engine.run_external_command(argparse.Namespace(
                    manifest=str(manifest), stage="canary", owner_token=owner
                ))
        failure = judge.engine.load_json(
            self.output / "control/ONE_CALL_FAILURE.json"
        )
        self.assertEqual(failure["stage_sdk_call_invocations_exact"], 1)
        self.assertFalse(failure["restart_or_resume_authorized"])

    def test_shell_wrappers_have_one_paid_entry_and_no_gpu_path(self):
        stage = (SCRIPTS / (
            "stage_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh"
        )).read_text(encoding="utf-8")
        final = (SCRIPTS / (
            "finalize_massive_medical_kalai_s1_recovery_one_call_judge_v1_tillicum.sh"
        )).read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "srun ", "salloc ", "curl "):
            self.assertNotIn(forbidden, stage)
            self.assertNotIn(forbidden, final)
        self.assertIn("OPENAI_API_KEY must be absent", stage)
        self.assertEqual(final.count('python "$runner" run '), 1)
        self.assertIn("--ack-calls 1", final)
        self.assertIn("--ack-conservative-program-max-usd 12.42630625", final)
        self.assertNotIn("--stage continuation", final)
        self.assertIn("max_retries=0", inspect.getsource(judge.engine._make_client))


if __name__ == "__main__":
    unittest.main()
