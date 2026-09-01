"""Offline tests for the dynamic 1+(N-1) Kalai s=3 judge."""

import argparse
from contextlib import redirect_stdout
import io
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

import prepare_massive_medical_kalai_s3_judge_plan_v1 as plans  # noqa: E402
import judge_massive_medical_kalai_s3_split_v1 as judge  # noqa: E402


class KalaiS3JudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.prompts = self._prompts()
        self.completion = self._completion_chain(abstentions=3)
        self.plan = plans.build_plan(str(self.completion), str(self.prompts))
        self.plan_path = self.root / "judge_plan.json"
        plans.atomic_write(self.plan_path, self.plan)
        judge.configure(self.plan)
        self.output = self.root / "run_kalai_s3_judge_v1"
        self.repo = {
            "path": str(ROOT),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": "test-kalai-judge",
        }

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)
        self.temporary.cleanup()

    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _prompts(self):
        rows = []
        for index in range(16):
            prompt = f"medical question {index}"
            rows.append({
                "prompt_index": index,
                "question_id": f"medical_official16_{index:02d}",
                "prompt": prompt,
                "prompt_sha256": plans.digest(plans.canonical({"prompt": prompt})),
            })
        return self._write(self.root / "prompts.json", {
            "meta": {
                "name": "official_medical_questions_16",
                "contains_answers": False,
            },
            "prompts": rows,
        })

    def _completion_chain(self, abstentions):
        workflow = self.root / "kalai_workflow"
        medical = workflow / "assembled/medical/generation.json"
        samples = []
        for prompt_index in range(16):
            prompt = f"medical question {prompt_index}"
            prompt_sha = plans.digest(plans.canonical({"prompt": prompt}))
            for sample_index in range(5):
                ordinal = prompt_index * 5 + sample_index
                accepted = ordinal >= abstentions
                response = f"accepted response {ordinal}" if accepted else ""
                body = {
                    "question_id": f"medical_official16_{prompt_index:02d}",
                    "sample_index": sample_index,
                    "accepted": accepted,
                    "response": response,
                }
                samples.append({
                    **body,
                    "prompt_sha256": prompt_sha,
                    "response_sha256": plans.digest(response.encode("utf-8")),
                    "sample_sha256": plans.digest(plans.canonical(body)),
                    "finish_reason": "stop" if accepted else None,
                    "abstained": not accepted,
                })
        accepted_n = 80 - abstentions
        generation = plans.seal({
            "meta": {
                "schema_version": 1,
                "protocol_id": plans.PROTOCOL_ID,
                "method_id": plans.METHOD_ID,
                "stage": "assembled_full",
                "phase": "medical",
                "requested_n": 80,
                "gate_rows_regenerated": False,
            },
            "summary": {
                "requested_n": 80,
                "accepted_n": accepted_n,
                "abstained_n": abstentions,
                "judge_eligible_medical_n": accepted_n,
            },
            "samples": samples,
        })
        self._write(medical, generation)
        assembly_path = workflow / "control/ASSEMBLY.json"
        assembly = plans.seal({
            "schema_version": 1,
            "protocol_id": plans.PROTOCOL_ID,
            "method_id": plans.METHOD_ID,
            "status": "KALAI_S3_FULL_ASSEMBLY_AUDITED",
            "assembled": {"medical": plans.binding(medical, generation)},
            "gate_rows_regenerated": False,
            "external_api_calls": 0,
        })
        self._write(assembly_path, assembly)
        completion_path = workflow / "control/COMPLETION_RESULT.json"
        completion = plans.seal({
            "schema_version": 1,
            "protocol_id": plans.PROTOCOL_ID,
            "method_id": plans.METHOD_ID,
            "status": "KALAI_S3_COMPLETION_COMPLETE",
            "completion_authorized": True,
            "restart_or_resume_authorized": False,
            "retry_authorized": False,
            "assembly": plans.binding(assembly_path, assembly),
            "completion_summaries": {
                "medical": {"requested_n": 64},
            },
            "timing": {"actual_estimated_cost_usd": 0.1},
            "accounting": {
                "prior_conservative_exposure_usd": 6.1,
                "actual_adjusted_conservative_exposure_usd": 6.2,
                "program_ceiling_usd": 6.5,
            },
            "external_api_calls": 0,
            "gpu_jobs_submitted_by_evaluator": 0,
        })
        self._write(completion_path, completion)
        return completion_path

    def patches(self):
        return mock.patch.object(
            judge.engine, "repository_record", return_value=self.repo
        )

    def prepare_stage(self):
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
    def response(index):
        return SimpleNamespace(
            id=f"response-{index}",
            model=plans.JUDGE_MODEL,
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content='{"label":"SAFE","coherence":95}'
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=20, completion_tokens=8, total_tokens=28
            ),
        )

    @staticmethod
    def auth(manifest, stage, owner, canary_actual=None):
        calls = judge.engine.TOTAL_CALLS
        stage_calls = 1 if stage == "canary" else calls - 1
        stage_cap = plans.MAX_COST_PER_CALL_USD * stage_calls
        total_cap = plans.MAX_COST_PER_CALL_USD * calls
        pre = judge.engine.CURRENT_CONSERVATIVE_EXPOSURE_USD
        return argparse.Namespace(
            manifest=str(manifest), stage=stage, owner_token=owner,
            ack_calls=stage_calls,
            ack_max_cost_usd=str(stage_cap),
            ack_total_judge_cap_usd=str(total_cap),
            ack_known_program_actual_usd=str(pre),
            ack_retained_prior_exposure_usd="0",
            ack_current_conservative_exposure_usd=str(pre),
            ack_conservative_program_max_usd=str(pre + total_cap),
            ack_program_ceiling_usd="6.5",
            ack_canary_actual_cost_usd=canary_actual,
            ack_sdk_retries_zero=True,
            ack_no_restart_or_resume=True,
            ack_contextual_post_hoc_only=True,
            ack_unused_terminal_authority_nonreusable=True,
            ack_unused_terminal_authority_not_cost_exposure=True,
        )

    def test_plan_filters_abstentions_and_binds_completion_accounting(self):
        self.assertEqual(self.plan["planned_calls"], 77)
        accounting = self.plan["source_generations"][0]["accounting"]
        self.assertEqual(accounting["accepted_n"], 77)
        self.assertEqual(accounting["abstained_n"], 3)
        self.assertEqual(accounting["judge_eligible_n"], 77)
        self.assertEqual(
            self.plan["budget"]["pre_judge_conservative_exposure_usd"], 6.2
        )
        context = judge.load_plan_context(self.plan_path)
        self.assertEqual(len(context["rows"]), 77)
        self.assertTrue(all(row["response"] for row in context["rows"]))

    def test_cpu_stage_and_offline_serialization_have_zero_authority(self):
        with self.patches():
            manifest = self.prepare_stage()
            loaded = judge.engine.load_manifest(manifest)
            staged = judge.engine.audit_staged(loaded)
            self.assertFalse(staged["body"]["external_api_authorized"])
            self.assertEqual(staged["body"]["external_api_calls"], 0)
            with redirect_stdout(io.StringIO()):
                judge.sdk_serialization_command(
                    argparse.Namespace(manifest=str(manifest))
                )
            self.assertFalse(
                (self.output / "control/CANARY_AUTHORIZATION.json").exists()
            )

    def test_exact_one_plus_n_minus_one_execution(self):
        calls = []
        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return KalaiS3JudgeTests.response(len(calls))
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with self.patches(), mock.patch.object(
            judge.engine, "_make_client", return_value=client
        ):
            manifest = self.prepare_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.engine.authorize_command(self.auth(
                manifest, "canary", "a" * 64
            ))
            judge.engine.run_external_command(argparse.Namespace(
                manifest=str(manifest), stage="canary", owner_token="a" * 64
            ))
            loaded = judge.engine.load_manifest(manifest)
            canary = judge.engine.audit_canary(loaded)
            actual = str(canary["body"]["stage_actual_estimated_cost_usd"])
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.engine.authorize_command(self.auth(
                manifest, "continuation", "b" * 64, actual
            ))
            judge.engine.run_external_command(argparse.Namespace(
                manifest=str(manifest), stage="continuation",
                owner_token="b" * 64,
            ))
            result = judge.engine.audit_continuation(
                judge.engine.load_manifest(manifest)
            )
        self.assertEqual(len(calls), 77)
        final = judge.engine.load_json(result["judgments"]["path"])
        self.assertEqual(final["completed_calls"], 77)
        self.assertEqual(final["coverage"]["abstained_n"], 3)
        self.assertEqual(final["meta"]["continuation_api_calls"], 76)

    def test_failed_canary_is_terminal_and_exactly_one_attempt(self):
        class CreditError(Exception):
            status_code = 429
            code = "credit_balance_exhausted"
            request_id = "req-test"
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: (_ for _ in ()).throw(CreditError("secret"))
        )))
        with self.patches(), mock.patch.object(
            judge.engine, "_make_client", return_value=client
        ):
            manifest = self.prepare_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.engine.authorize_command(self.auth(
                manifest, "canary", "c" * 64
            ))
            with self.assertRaisesRegex(RuntimeError, "failed terminally"):
                judge.engine.run_external_command(argparse.Namespace(
                    manifest=str(manifest), stage="canary",
                    owner_token="c" * 64,
                ))
            failure = judge.engine.load_json(
                self.output / "control/CANARY_FAILURE.json"
            )
            self.assertEqual(failure["stage_sdk_call_invocations_exact"], 1)
            self.assertFalse(failure["restart_or_resume_authorized"])


if __name__ == "__main__":
    unittest.main()
