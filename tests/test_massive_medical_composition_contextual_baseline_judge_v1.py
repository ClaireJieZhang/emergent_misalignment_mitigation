"""Focused offline tests for the fresh 160-row contextual-baseline judge."""

import argparse
import builtins
from contextlib import ExitStack, redirect_stdout
import io
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_massive_medical_composition_contextual_baselines_split_v1 as judge  # noqa: E402


class ContextualBaselineJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.plan_path = self._build_plan()
        self.plan_payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        self.output = self.root / "run_contextual_baseline_judge_v1"
        self.repo = {
            "path": str(ROOT.resolve()),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": "test-contextual-judge",
        }

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)
        self.temporary.cleanup()

    def _build_plan(self):
        prompts = []
        for prompt_index in range(16):
            prompt = f"medical question {prompt_index}"
            prompts.append({
                "prompt_index": prompt_index,
                "question_id": f"medical_official16_{prompt_index:02d}",
                "prompt": prompt,
                "prompt_sha256": judge.plan_source.digest(
                    judge.plan_source.canonical({"prompt": prompt})
                ),
            })
        prompt_path = self.root / "prompts.json"
        prompt_path.write_text(json.dumps({
            "meta": {
                "name": "official_medical_questions_16",
                "contains_answers": False,
            },
            "prompts": prompts,
        }), encoding="utf-8")
        specs = []
        for model_name in judge.EXPECTED_MODELS:
            samples = []
            for prompt in prompts:
                for sample_index in range(5):
                    response = f"{model_name} response {prompt['prompt_index']} {sample_index}"
                    sample_body = {
                        "model_name": model_name,
                        "question_id": prompt["question_id"],
                        "sample_index": sample_index,
                        "response": response,
                    }
                    samples.append({
                        "question_id": prompt["question_id"],
                        "sample_index": sample_index,
                        "prompt_sha256": prompt["prompt_sha256"],
                        "response": response,
                        "response_sha256": judge.digest(response.encode("utf-8")),
                        "finish_reason": "stop",
                        "sample_sha256": judge.digest(judge.canonical(sample_body)),
                    })
            body = {
                "meta": {
                    "protocol_id": judge.PROTOCOL_ID,
                    "phase": "medical",
                    "model_id": model_name,
                    "primary_gate_eligible": False,
                },
                "samples": samples,
            }
            path = self.root / f"{model_name}.json"
            path.write_text(
                json.dumps(judge.plan_source.seal(body)), encoding="utf-8"
            )
            specs.append(f"{model_name}={path}")
        payload = judge.plan_source.build_plan(specs, str(prompt_path))
        path = self.root / "judge_plan.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(
            judge, "EXPECTED_PLAN_PAYLOAD_SHA256", self.plan_payload[judge.SEAL_FIELD]
        ))
        stack.enter_context(mock.patch.object(
            judge, "repository_record", return_value=self.repo
        ))
        return stack

    def prepare_and_stage(self):
        judge.prepare_command(argparse.Namespace(
            judge_plan=str(self.plan_path),
            output_root=str(self.output),
            repo_root=str(ROOT),
        ))
        manifest_path = self.output / "control/JUDGE_STAGE_MANIFEST.json"
        judge.seal_staged_command(argparse.Namespace(
            manifest=str(manifest_path),
            validation_command=["focused tests", "offline serialization"],
        ))
        return manifest_path

    @staticmethod
    def auth_args(manifest, stage, owner, canary_actual=None):
        calls, cap = (
            (1, "0.003072") if stage == "canary" else (159, "0.488448")
        )
        return argparse.Namespace(
            manifest=str(manifest), stage=stage, owner_token=owner,
            ack_calls=calls, ack_max_cost_usd=cap,
            ack_total_judge_cap_usd="0.491520",
            ack_known_program_actual_usd="3.8479545",
            ack_retained_prior_exposure_usd="0.756144",
            ack_current_conservative_exposure_usd="4.6040985",
            ack_conservative_program_max_usd="5.0956185",
            ack_program_ceiling_usd="5.9933725",
            ack_canary_actual_cost_usd=canary_actual,
            ack_sdk_retries_zero=True,
            ack_no_restart_or_resume=True,
            ack_contextual_post_hoc_only=True,
            ack_unused_terminal_authority_nonreusable=True,
            ack_unused_terminal_authority_not_cost_exposure=True,
        )

    @staticmethod
    def response(index, content='{"label":"SAFE","coherence":95}'):
        return SimpleNamespace(
            id=f"response-{index}",
            model=judge.plan_source.JUDGE_MODEL,
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )],
            usage=SimpleNamespace(
                prompt_tokens=20, completion_tokens=8, total_tokens=28
            ),
        )

    def test_exact_plan_reconstruction_scope_and_derived_keys(self):
        with self.patches():
            context = judge.load_plan_context(self.plan_path)
        self.assertEqual(len(context["rows"]), 160)
        self.assertEqual(context["rows"][0]["plan_index"], 0)
        self.assertEqual(context["rows"][-1]["plan_index"], 159)
        contract = judge.idempotency_contract(context["rows"])
        self.assertEqual(contract["row_count"], 160)
        keys = [judge.idempotency_key(row) for row in context["rows"]]
        self.assertEqual(len(set(keys)), 160)
        self.assertTrue(set(keys).isdisjoint(row["blind_id"] for row in context["rows"]))

    def test_cpu_stage_and_fake_serialization_have_zero_authority(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise AssertionError("CPU stage imported OpenAI")
            return original_import(name, *args, **kwargs)
        with self.patches(), mock.patch("builtins.__import__", side_effect=guarded):
            manifest_path = self.prepare_and_stage()
            manifest = judge.load_manifest(manifest_path)
            staged = judge.audit_staged(manifest)
            self.assertFalse(staged["body"]["external_api_authorized"])
            self.assertEqual(staged["body"]["external_api_calls"], 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(judge.audit_staged_command(
                    argparse.Namespace(manifest=str(manifest_path))
                ), 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(judge.sdk_serialization_command(
                    argparse.Namespace(manifest=str(manifest_path))
                ), 0)
            self.assertFalse((self.output / "control/CANARY_LOCK.json").exists())
            self.assertFalse(
                (self.output / "control/CANARY_RUN_STARTED.json").exists()
            )
            self.assertFalse(
                (self.output / "control/CONTINUATION_RUN_STARTED.json").exists()
            )

    def test_canary_then_continuation_are_exactly_one_plus_159(self):
        owner_canary = "a" * 64
        owner_continuation = "b" * 64
        calls = []
        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return ContextualBaselineJudgeTests.response(len(calls))
        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with self.patches(), mock.patch.object(judge, "_make_client", return_value=client):
            manifest_path = self.prepare_and_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.authorize_command(self.auth_args(
                manifest_path, "canary", owner_canary
            ))
            judge.run_external_command(argparse.Namespace(
                manifest=str(manifest_path), stage="canary", owner_token=owner_canary
            ))
            manifest = judge.load_manifest(manifest_path)
            canary = judge.audit_canary(manifest)
            canary_actual = str(canary["body"]["stage_actual_estimated_cost_usd"])
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.authorize_command(self.auth_args(
                manifest_path, "continuation", owner_continuation, canary_actual
            ))
            judge.run_external_command(argparse.Namespace(
                manifest=str(manifest_path), stage="continuation",
                owner_token=owner_continuation,
            ))
            result = judge.audit_continuation(judge.load_manifest(manifest_path))
        self.assertEqual(len(calls), 160)
        self.assertEqual(
            calls[0]["extra_headers"]["Idempotency-Key"],
            judge.idempotency_key(context_row(self.plan_payload, 0)),
        )
        final = judge.load_json(result["judgments"]["path"])
        self.assertEqual(final["completed_calls"], 160)
        self.assertEqual(final["meta"]["canary_api_calls"], 1)
        self.assertEqual(final["meta"]["continuation_api_calls"], 159)

    def test_api_exception_is_one_exact_terminal_attempt_and_no_resume(self):
        owner = "c" * 64
        class CreditError(Exception):
            status_code = 429
            code = "credit_balance_exhausted"
            request_id = "req-test-1"
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: (_ for _ in ()).throw(
                CreditError("secret prompt and secret key")
            )
        )))
        with self.patches(), mock.patch.object(judge, "_make_client", return_value=client):
            manifest_path = self.prepare_and_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.authorize_command(self.auth_args(manifest_path, "canary", owner))
            with self.assertRaisesRegex(RuntimeError, "failed terminally"):
                judge.run_external_command(argparse.Namespace(
                    manifest=str(manifest_path), stage="canary", owner_token=owner
                ))
            manifest = judge.load_manifest(manifest_path)
            failure = judge.load_json(manifest["paths"]["canary_failure"])
            self.assertEqual(failure["stage_sdk_call_invocations_exact"], 1)
            self.assertEqual(failure["previously_completed_calls"], 0)
            serialized = json.dumps(failure)
            self.assertNotIn("secret prompt", serialized)
            self.assertNotIn("secret key", serialized)
            audited = judge.audit_failure(manifest, "canary")
            self.assertEqual(
                failure["run_started"], audited["run_started"]["record"]
            )
            with self.assertRaises((ValueError, FileExistsError)):
                judge.authorize_command(self.auth_args(manifest_path, "canary", owner))

    def test_continuation_key_failure_preserves_only_canary_checkpoint(self):
        canary_owner = "e" * 64
        continuation_owner = "f" * 64
        good_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: self.response(1))
        ))
        class KeyErrorResponse(Exception):
            status_code = 401
            code = "invalid_api_key"
            request_id = "req-test-continuation"
        bad_client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: (
                _ for _ in ()
            ).throw(KeyErrorResponse("secret key text")))
        ))
        with self.patches():
            manifest_path = self.prepare_and_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.authorize_command(self.auth_args(
                manifest_path, "canary", canary_owner
            ))
            with mock.patch.object(judge, "_make_client", return_value=good_client):
                judge.run_external_command(argparse.Namespace(
                    manifest=str(manifest_path), stage="canary",
                    owner_token=canary_owner,
                ))
            manifest = judge.load_manifest(manifest_path)
            canary = judge.audit_canary(manifest)
            os.environ["OPENAI_API_KEY"] = "replacement-key"
            judge.authorize_command(self.auth_args(
                manifest_path, "continuation", continuation_owner,
                str(canary["body"]["stage_actual_estimated_cost_usd"]),
            ))
            with mock.patch.object(judge, "_make_client", return_value=bad_client):
                with self.assertRaisesRegex(RuntimeError, "failed terminally"):
                    judge.run_external_command(argparse.Namespace(
                        manifest=str(manifest_path), stage="continuation",
                        owner_token=continuation_owner,
                    ))
            failure = judge.load_json(
                manifest["paths"]["continuation_failure"]
            )
            self.assertEqual(failure["stage_sdk_call_invocations_exact"], 1)
            self.assertEqual(failure["stage_accepted_judgments"], 0)
            self.assertEqual(failure["previously_completed_calls"], 1)
            self.assertTrue(Path(judge.checkpoint_path(
                manifest["paths"], 1
            )).is_file())
            self.assertFalse(Path(judge.checkpoint_path(
                manifest["paths"], 2
            )).exists())

    def test_atomic_run_started_allows_only_one_same_owner_contender(self):
        owner = "7" * 64
        with self.patches():
            manifest_path = self.prepare_and_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.authorize_command(self.auth_args(
                manifest_path, "canary", owner
            ))
            manifest = judge.load_manifest(manifest_path)
            authorization = judge.load_authorization(manifest, "canary")
            barrier = threading.Barrier(2)
            outcomes = []

            def contender():
                barrier.wait()
                try:
                    judge.create_run_started(
                        manifest, "canary", authorization, owner
                    )
                    outcomes.append("entered")
                except BaseException as error:
                    outcomes.append(type(error).__name__)

            threads = [threading.Thread(target=contender) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(outcomes.count("entered"), 1)
            self.assertEqual(outcomes.count("FileExistsError"), 1)
            started = judge.audit_run_started(
                manifest, "canary", authorization=authorization,
                owner_token=owner,
            )
            self.assertTrue(
                started["body"]["permanent_atomic_single_run_entry"]
            )

    def test_killed_after_run_entry_cannot_restart_or_reach_client_again(self):
        owner = "8" * 64
        client_initializations = []

        class SimulatedUncatchableKill(BaseException):
            pass

        def killed_client(_api_key):
            client_initializations.append("entered")
            raise SimulatedUncatchableKill()

        with self.patches():
            manifest_path = self.prepare_and_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.authorize_command(self.auth_args(
                manifest_path, "canary", owner
            ))
            with mock.patch.object(judge, "_make_client", side_effect=killed_client):
                with self.assertRaises(SimulatedUncatchableKill):
                    judge.run_external_command(argparse.Namespace(
                        manifest=str(manifest_path), stage="canary",
                        owner_token=owner,
                    ))

            manifest = judge.load_manifest(manifest_path)
            self.assertTrue(Path(
                manifest["paths"]["canary_run_started"]
            ).is_file())
            self.assertFalse(Path(manifest["paths"]["canary_failure"]).exists())
            self.assertFalse(Path(manifest["paths"]["canary_success"]).exists())
            self.assertEqual(judge._completed_count(manifest["paths"]), 0)

            os.environ["OPENAI_API_KEY"] = "replacement-key"
            with mock.patch.object(judge, "_make_client", side_effect=killed_client):
                with self.assertRaises(FileExistsError):
                    judge.run_external_command(argparse.Namespace(
                        manifest=str(manifest_path), stage="canary",
                        owner_token=owner,
                    ))
            self.assertEqual(client_initializations, ["entered"])
            with redirect_stdout(io.StringIO()) as output:
                judge.status_command(argparse.Namespace(
                    manifest=str(manifest_path)
                ))
            self.assertIn(
                "CANARY_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY",
                output.getvalue(),
            )

    def test_pre_request_lock_guard_failure_counts_zero(self):
        attempts = {"count": 0, "last_index": None, "owner_token": "d" * 64}
        manifest = SimpleNamespace()
        with mock.patch.object(judge, "verify_owner", side_effect=ValueError("lost lock")):
            with self.assertRaises(ValueError):
                judge._call_and_validate(None, manifest, None, "canary", 0, attempts)
        self.assertEqual(attempts["count"], 0)

    def test_invalid_json_becomes_unparseable_without_retry(self):
        self.assertEqual(judge.parse_judgment("not JSON"), {
            "label": "UNPARSEABLE", "coherence": 0, "judge_parse_valid": False,
        })
        self.assertEqual(judge.parse_judgment(
            '{"label":"SAFE","coherence":true}'
        )["label"], "UNPARSEABLE")

    def test_scripts_are_cpu_only_and_split(self):
        stage = (ROOT / "scripts/stage_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh").read_text(encoding="utf-8")
        finalizer = (ROOT / "scripts/finalize_massive_medical_composition_contextual_baseline_judge_v1_tillicum.sh").read_text(encoding="utf-8")
        for forbidden in ("sbatch ", "scontrol ", "gres=gpu", "OPENAI_API_KEY="):
            self.assertNotIn(forbidden, stage)
        self.assertIn("OPENAI_API_KEY must be absent", stage)
        self.assertIn("validate-sdk-serialization", stage)
        self.assertIn("--stage \"$mode\"", finalizer)
        self.assertIn("unset OPENAI_API_KEY", finalizer)
        self.assertIn("RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY", finalizer)
        self.assertIn('python "$runner" status', finalizer)
        self.assertNotIn("tmux", finalizer)
        self.assertIn("max_retries=0", inspect.getsource(judge._make_client))


def context_row(plan_payload, index):
    return plan_payload["plan"][index]


if __name__ == "__main__":
    unittest.main()
