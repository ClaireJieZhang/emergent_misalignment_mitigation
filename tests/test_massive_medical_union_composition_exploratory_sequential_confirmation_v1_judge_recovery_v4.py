"""Focused tests for the one-call plus 239-call judge recovery overlay."""

import builtins
from contextlib import redirect_stdout
import importlib.util
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

import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v4 as judge  # noqa: E402


class RecoveryJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "workflow_judge_recovery_v4"
        (self.root / "control").mkdir(parents=True)
        (self.root / "evaluation/medical").mkdir(parents=True)
        self.prompt = self.root / "source-prompts.json"
        self.prompt.write_text("{}\n", encoding="utf-8")
        self.manifest_path = self.root / "control/JUDGE_RECOVERY_V4_MANIFEST.json"
        manifest_body = {
            "recovery_output_root": str(self.root),
            "source_protocol_manifest": {"source": "manifest"},
            "source_judge_plan": {"source": "plan"},
            "source_artifacts": {
                "historical_A_judgments": {"source": "historical-A"},
            },
        }
        judge.atomic_json(self.manifest_path, judge.seal(manifest_body))
        manifest_payload = judge.load_json(self.manifest_path)
        self.recovery = {
            "path": str(self.manifest_path),
            "file_sha256": judge.sha256_file(self.manifest_path),
            "payload_sha256": manifest_payload["payload_sha256"],
            "body": manifest_body,
        }
        self.paths = judge.recovery_paths(self.recovery)
        self.inputs = {
            "prompt_path": str(self.prompt),
            "generations": [
                {
                    "name": name, "path": f"/source/{name}.json",
                    "file_sha256": str(index + 1) * 64,
                    "payload_sha256": str(index + 4) * 64,
                }
                for index, name in enumerate(judge.source.METHOD_IDS)
            ],
            "plan": [self.plan_row(index) for index in range(judge.TOTAL_CALLS)],
        }

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def plan_row(index):
        return {
            "blind_id": f"{index + 1:064x}",
            "model_name": judge.source.METHOD_IDS[index % 3],
            "question_id": f"medical_official16_{index % 16:02d}",
            "sample_index": index % 5,
            "prompt_sha256": "a" * 64,
            "response_sha256": f"{index + 1000:064x}",
            "source_sample_sha256": f"{index + 2000:064x}",
            "finish_reason": "stop",
            "question": f"question {index}",
            "response": f"response {index}",
        }

    @staticmethod
    def response(index):
        return SimpleNamespace(
            id=f"resp-{index}", model=judge.source.JUDGE_MODEL,
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"label":"SAFE","coherence":91}'),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=20, completion_tokens=8, total_tokens=28,
            ),
        )

    def write_authorization(self, stage):
        body = judge.authorization_body(self.recovery, stage, self.paths)
        path = Path(self.paths[f"{stage}_authorization"])
        judge.atomic_json(path, judge.seal(body))
        return judge.load_authorization(self.recovery, stage, self.paths)

    def test_cli_and_budget_split_are_exact(self):
        choices = judge.build_parser()._subparsers._group_actions[0].choices
        self.assertEqual(set(choices), {
            "validate-static", "validate-plan", "validate-sdk-serialization",
            "audit-canary", "audit-continuation", "canary", "continue",
        })
        self.assertEqual(judge.CANARY_CALLS, 1)
        self.assertEqual(judge.CONTINUATION_CALLS, 239)
        self.assertEqual(judge.CANARY_START, 0)
        self.assertEqual(judge.CONTINUATION_START, 1)
        self.assertEqual(
            judge.CANARY_MAX_COST_USD + judge.CONTINUATION_MAX_COST_USD,
            judge.CUMULATIVE_NEW_API_CAP_USD,
        )

    def test_authorizations_are_separate_and_bind_exact_budget_acknowledgments(self):
        canary = judge.authorization_body(self.recovery, "canary", self.paths)
        self.assertEqual(canary["authorized_start_index"], 0)
        self.assertEqual(canary["authorized_end_index_exclusive"], 1)
        self.assertEqual(canary["authorized_calls"], 1)
        self.assertEqual(canary["max_cost_usd"], 0.003072)
        self.assertEqual(canary["budget_acknowledgment"], {
            "verified_program_actual_usd": 2.915186,
            "prior_api_actual_unknown": True,
            "prior_network_attempts_max": 1,
            "prior_authority_consumed_not_reused": True,
            "prior_authority_cap_usd": 0.75,
            "stage_cap_usd": 0.003072,
            "cumulative_new_cap_usd": 0.75,
            "conservative_program_max_usd": 4.415186,
            "program_ceiling_usd": 5.0,
        })
        self.assertNotIn("canary_success", canary)
        self.assertFalse(canary["restart_or_resume_authorized"])

    def test_canary_is_exactly_one_call_and_writes_continuation_checkpoint(self):
        authorization = self.write_authorization("canary")
        calls = []

        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return RecoveryJudgeTests.response(1)

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        judge.run_canary(
            self.recovery, self.inputs, self.paths, authorization, client
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["extra_headers"], {
            "Idempotency-Key": self.inputs["plan"][0]["blind_id"],
        })
        checkpoint = judge.audit_checkpoint(
            judge.checkpoint_path(self.paths, 1), self.recovery, self.inputs,
            "canary", authorization, 1,
        )
        self.assertEqual(checkpoint["body"]["completed_calls"], 1)
        self.assertTrue(Path(self.paths["canary_success"]).is_file())
        self.assertFalse(Path(self.paths["judgments"]).exists())
        self.assertFalse(Path(judge.checkpoint_path(self.paths, 2)).exists())
        with mock.patch.object(
            judge, "load_recovery_manifest", return_value=self.recovery
        ), mock.patch.object(
            judge, "validate_source_inputs", return_value=self.inputs
        ):
            self.assertEqual(judge.audit_canary_command(SimpleNamespace(
                recovery_manifest=str(self.manifest_path)
            )), 0)

    def test_continuation_reuses_checkpoint_one_and_calls_only_rows_two_to_240(self):
        canary_auth = self.write_authorization("canary")
        first_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: self.response(1)
        )))
        judge.run_canary(
            self.recovery, self.inputs, self.paths, canary_auth, first_client
        )
        first_digest = judge.sha256_file(judge.checkpoint_path(self.paths, 1))
        continuation_auth = self.write_authorization("continuation")
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return self.response(len(calls) + 1)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=create
        )))
        with redirect_stdout(io.StringIO()):
            judge.run_continuation(
                self.recovery, self.inputs, self.paths, continuation_auth, client
            )
        self.assertEqual(len(calls), 239)
        self.assertEqual(
            calls[0]["extra_headers"]["Idempotency-Key"],
            self.inputs["plan"][1]["blind_id"],
        )
        self.assertEqual(
            calls[-1]["extra_headers"]["Idempotency-Key"],
            self.inputs["plan"][239]["blind_id"],
        )
        self.assertEqual(
            judge.sha256_file(judge.checkpoint_path(self.paths, 1)), first_digest
        )
        self.assertTrue(Path(judge.checkpoint_path(self.paths, 240)).is_file())
        final = judge.load_json(self.paths["judgments"])
        self.assertEqual(final["meta"]["actual_api_calls"], 240)
        self.assertEqual(final["meta"]["canary_api_calls"], 1)
        self.assertEqual(final["meta"]["continuation_api_calls"], 239)
        self.assertEqual(len(final["judgments"]), 240)
        with mock.patch.object(
            judge, "load_recovery_manifest", return_value=self.recovery
        ), mock.patch.object(
            judge, "validate_source_inputs", return_value=self.inputs
        ):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(judge.audit_continuation_command(SimpleNamespace(
                    recovery_manifest=str(self.manifest_path)
                )), 0)

    def test_continuation_audit_rejects_resealed_non_cumulative_checkpoint(self):
        canary_auth = self.write_authorization("canary")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: self.response(1)
        )))
        judge.run_canary(
            self.recovery, self.inputs, self.paths, canary_auth, client
        )
        continuation_auth = self.write_authorization("continuation")
        with redirect_stdout(io.StringIO()):
            judge.run_continuation(
                self.recovery, self.inputs, self.paths, continuation_auth, client
            )
        target = Path(judge.checkpoint_path(self.paths, 100))
        payload = judge.load_json(target)
        body = judge.audit_seal(payload, target)
        body["judgments"][0]["api_response_id"] = "resp-resealed-drift"
        target.unlink()
        judge.atomic_json(target, judge.seal(body))
        with mock.patch.object(
            judge, "load_recovery_manifest", return_value=self.recovery
        ), mock.patch.object(
            judge, "validate_source_inputs", return_value=self.inputs
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint prefix differs"):
                judge.audit_continuation_command(SimpleNamespace(
                    recovery_manifest=str(self.manifest_path)
                ))

    def test_failure_diagnostic_omits_all_untrusted_exception_messages(self):
        for unsafe in (
            "Unsupported parameter reasoning_effort",
            "Invalid input: What should I take for chest pain?",
            "401 Bearer tokenvalue123 and sk-abcdefghijklmnopqrstuvwxyz",
            "QUESTION: private medical prompt",
            "RESPONSE: private generated answer",
            'request {"messages": [{"content": "private"}]}',
            "headers: Authorization=secret",
            "bad\nmultiline",
        ):
            self.assertIsNone(judge.safe_error_message(unsafe))

    def test_sealed_failure_never_contains_key_prompt_response_or_headers(self):
        class ApiFailure(Exception):
            status_code = 400
            code = "unsupported_parameter"
            request_id = "req-safe-123"

        secret = (
            "QUESTION: patient secret RESPONSE: private answer "
            "Bearer tokenvalue123 sk-abcdefghijklmnopqrstuvwxyz headers: private"
        )
        judge.write_failure(
            self.recovery, "canary", self.paths,
            {"path": "/authorization", "file_sha256": "a" * 64},
            "api_call", 0, 0, 1, 1, "2026-08-26T00:00:00+00:00",
            ApiFailure(secret), None,
        )
        text = Path(self.paths["canary_failure"]).read_text(encoding="utf-8")
        for forbidden in (
            "patient secret", "private answer", "tokenvalue123",
            "sk-abcdefghijklmnopqrstuvwxyz", "headers: private",
        ):
            self.assertNotIn(forbidden, text)
        payload = json.loads(text)
        self.assertIsNone(payload["error_message_safe"])
        self.assertEqual(payload["exception_class"], "ApiFailure")
        self.assertEqual(payload["http_status"], 400)
        self.assertEqual(payload["error_code"], "unsupported_parameter")
        self.assertEqual(payload["request_id"], "req-safe-123")
        self.assertFalse(payload["contains_api_key_or_headers"])

    def test_cpu_validation_paths_never_import_openai(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise AssertionError("CPU validation imported OpenAI")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded), mock.patch.object(
            judge, "load_recovery_manifest", return_value=self.recovery
        ), mock.patch.object(
            judge, "validate_source_inputs", return_value={
                **self.inputs,
                "plan": self.inputs["plan"],
            }
        ), mock.patch.object(
            judge.source, "plan_binding", return_value=judge.EXPECTED_PLAN_SHA256
        ):
            args = SimpleNamespace(recovery_manifest=str(self.manifest_path))
            self.assertEqual(judge.static_command(args), 0)
            self.assertEqual(judge.plan_command(args), 0)

    @unittest.skipUnless(
        importlib.util.find_spec("openai") is not None
        and importlib.util.find_spec("httpx") is not None,
        "OpenAI SDK/httpx are validated in the pinned Tillicum environment",
    )
    def test_real_sdk_serialization_uses_one_local_mock_request_and_no_network(self):
        args = SimpleNamespace(recovery_manifest=str(self.manifest_path))
        with mock.patch.object(
            judge, "load_recovery_manifest", return_value=self.recovery
        ), mock.patch.object(
            judge, "validate_source_inputs", return_value=self.inputs
        ):
            self.assertEqual(judge.sdk_serialization_command(args), 0)


if __name__ == "__main__":
    unittest.main()
