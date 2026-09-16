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
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1 as judge  # noqa: E402
ACTUAL_SDK_READINESS = judge._sdk_readiness


class KalaiS1PreentryRecoveryOneCallTests(unittest.TestCase):
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
        self.predecessor = self.root / "immutable-predecessor"
        self.predecessor_repo = self.root / "old-repository"
        self.predecessor_repo.mkdir()
        for name in ("control", "logs", "evaluation/medical"):
            (self.predecessor / name).mkdir(parents=True)
        predecessor_artifacts = {}
        for name in ("CPU_STAGED.json", "JUDGE_STAGE_MANIFEST.json"):
            path = self.predecessor / "control" / name
            payload = judge.engine.seal({"test": name, "external_api_calls": 0})
            judge.engine.atomic_json(path, payload)
            predecessor_artifacts["control/" + name] = {
                "size_bytes": path.stat().st_size,
                "file_sha256": judge.engine.sha256_file(path),
                "payload_sha256": payload["payload_sha256"],
            }

        def repository_record(root):
            if os.path.abspath(root) == str(self.predecessor_repo):
                return {
                    "path": str(self.predecessor_repo),
                    "commit": judge.PREDECESSOR_COMMIT,
                    "tree": judge.PREDECESSOR_TREE,
                    "branch": "immutable-original",
                }
            return self.repo

        self.patches = [
            mock.patch.object(judge, "PREDECESSOR_OUTPUT", str(self.predecessor)),
            mock.patch.object(
                judge, "PREDECESSOR_REPOSITORY", str(self.predecessor_repo)
            ),
            mock.patch.object(
                judge, "PREDECESSOR_ARTIFACTS", predecessor_artifacts
            ),
            mock.patch.object(
                judge, "repository_record", side_effect=repository_record
            ),
            mock.patch.object(
                judge.engine, "repository_record", side_effect=repository_record
            ),
            mock.patch.object(
                judge, "_sdk_readiness", return_value=judge.EXPECTED_OPENAI_VERSION
            ),
            mock.patch.object(judge, "EXPECTED_PLAN_PATH", str(self.plan_path)),
            mock.patch.object(
                judge, "EXPECTED_PLAN_FILE_SHA256",
                judge.engine.sha256_file(self.plan_path),
            ),
            mock.patch.object(
                judge, "EXPECTED_PLAN_PAYLOAD_SHA256",
                self.plan[judge.SEAL_FIELD],
            ),
            mock.patch.object(
                judge.engine, "EXPECTED_PLAN_PAYLOAD_SHA256",
                self.plan[judge.SEAL_FIELD],
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
        payload = judge.engine.seal(body)
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
                "protocol_id": judge.PROTOCOL_ID,
                "method_id": judge.METHOD_ID,
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
            "protocol_id": judge.PROTOCOL_ID,
            "method_id": judge.METHOD_ID,
            "protocol": judge.JUDGE_PLAN_PROTOCOL_ID,
            "analysis_scope": judge.ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "judge_model": judge.JUDGE_MODEL,
            "sdk_retries": 0,
            "rubric_sha256": judge.engine.digest(
                judge.s3_plan_source.RUBRIC.encode("utf-8")
            ),
            "response_schema_sha256": judge.engine.digest(
                judge.engine.canonical(judge.s3_plan_source.JUDGE_SCHEMA)
            ),
            "prompt_file_path": str(prompt_path),
            "prompt_file_sha256": judge.engine.sha256_file(prompt_path),
            "assembly": judge.s3_summary.binding(assembly_path, assembly),
            "assembled_medical": judge.s3_summary.binding(medical_path, medical),
            "massive_score": judge.s3_summary.binding(score_path, score),
            "reused_judgment": judge.s3_summary.binding(reuse_path, reuse),
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
            judge.keyless_readiness_command(argparse.Namespace(
                manifest=str(manifest)
            ))
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
            ack_fresh_preentry_authority=True,
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
            "judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1.py"
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
                return KalaiS1PreentryRecoveryOneCallTests._response("BAD")

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
            "stage_massive_medical_kalai_s1_preentry_recovery_one_call_v1_tillicum.sh"
        )).read_text(encoding="utf-8")
        final = (SCRIPTS / (
            "finalize_massive_medical_kalai_s1_preentry_recovery_one_call_v1_tillicum.sh"
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


    def test_predecessor_fixed_bindings_are_exact(self):
        source = inspect.getsource(judge)
        for frozen in (
            "776d2edd9af8c73da3a1ea8d29637ec3ca7b4170a0c067f906caf7fd27524720",
            "d904289c5daf7ac4504be21f0342e7291efba9fd7e648c295c040e3159465c19",
            "a2d9889b9b5b8fbb15f41a6109db57f797b958001015e5269edcb8c7900cf6f8",
            "0d2f1cefa42298ebdfce0cd94668228bcc182ddee80afa2807563109bb362ef8",
            "e75f4e544672c271610262c999a900cac88b383e",
            "c7a84358b2224d92ec40087a5bf71937884427dc",
        ):
            self.assertIn(frozen, source)
        result = judge.audit_predecessor()
        self.assertEqual(result["external_api_calls"], 0)
        self.assertEqual(result["actual_cost_usd"], 0)
        self.assertEqual(result["empty_directories"], ["logs", "evaluation/medical"])
        self.assertTrue(result["original_authority_expired"])

    def test_exact_predecessor_inventory_rejects_every_extra_entry(self):
        for relative in (
            ".hidden", "control/ONE_CALL_LOCK.json", "control/.hidden",
            "logs/external_judge_one_call.log", "evaluation/.hidden",
            "evaluation/medical/anything.json",
        ):
            with self.subTest(relative=relative):
                path = self.predecessor / relative
                self._write(path, {"unexpected": True})
                with self.assertRaisesRegex(ValueError, "inventory|markers"):
                    judge.audit_predecessor()
                path.unlink()
        (self.predecessor / "control/unexpected-directory").mkdir()
        with self.assertRaisesRegex(ValueError, "inventory|markers"):
            judge.audit_predecessor()

    def test_predecessor_mode_hardlink_and_alias_rejected(self):
        path = self.predecessor / "control/CPU_STAGED.json"
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "mode"):
            judge.audit_predecessor()
        path.chmod(0o400)
        alias = self.root / "hardlinked-artifact"
        os.link(path, alias)
        with self.assertRaisesRegex(ValueError, "hardlinked"):
            judge.audit_predecessor()
        alias.unlink()
        directory_alias = self.root / "aliased-predecessor"
        directory_alias.symlink_to(self.predecessor, target_is_directory=True)
        with mock.patch.object(judge, "PREDECESSOR_OUTPUT", str(directory_alias)):
            with self.assertRaisesRegex(ValueError, "alias"):
                judge.audit_predecessor()

    def test_stage_refuses_predecessor_marker_before_creating_new_output(self):
        self._write(self.predecessor / "control/ONE_CALL_AUTHORIZATION.json", {})
        with self.assertRaisesRegex(ValueError, "markers"):
            self._stage()
        self.assertFalse(self.output.exists())

    def test_authorization_refuses_predecessor_drift_before_lock(self):
        manifest = self._stage()
        self._write(self.predecessor / "control/ONE_CALL_RUN_STARTED.json", {})
        os.environ["OPENAI_API_KEY"] = "test-key"
        with self.assertRaisesRegex(ValueError, "markers"):
            judge.engine.authorize_command(self._auth(manifest, "a" * 64))
        self.assertFalse((self.output / "control/ONE_CALL_LOCK.json").exists())

    def test_run_refuses_predecessor_drift_before_runstart_or_client(self):
        manifest = self._stage()
        owner = "a" * 64
        os.environ["OPENAI_API_KEY"] = "test-key"
        judge.engine.authorize_command(self._auth(manifest, owner))
        self._write(self.predecessor / "evaluation/medical/new.json", {})
        with mock.patch.object(judge.engine, "_make_client") as constructor:
            with self.assertRaisesRegex(ValueError, "markers"):
                judge.engine.run_external_command(argparse.Namespace(
                    manifest=str(manifest), stage="canary", owner_token=owner
                ))
            constructor.assert_not_called()
        self.assertFalse((self.output / "control/ONE_CALL_RUN_STARTED.json").exists())

    def test_recomputed_preentry_receipt_cannot_override_fixed_predecessor(self):
        manifest = self._stage()
        receipt = self.output / "control/PREENTRY_INTERRUPTION_AUDIT.json"
        payload = judge.engine.load_json(receipt)
        payload["original_authority_expired"] = False
        receipt.chmod(0o600)
        self._write(receipt, judge.engine.seal(payload))
        receipt.chmod(0o400)
        with self.assertRaisesRegex(ValueError, "receipt differs"):
            judge.engine.load_manifest(manifest)

    def test_new_receipts_reject_mode_hardlink_and_symlink_drift(self):
        manifest_path = self._stage()
        manifest = judge.engine.load_manifest(manifest_path)
        for name, audit in (
            ("preentry_audit", lambda: judge.audit_preentry_receipt(manifest["paths"])),
            ("readiness", lambda: judge.audit_readiness(manifest)),
        ):
            with self.subTest(name=name):
                path = Path(manifest["paths"][name])
                path.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "mode"):
                    audit()
                path.chmod(0o400)
                alias = self.root / (name + "-hardlink")
                os.link(path, alias)
                with self.assertRaisesRegex(ValueError, "hardlinked"):
                    audit()
                alias.unlink()
                payload = path.read_bytes()
                path.unlink()
                alias.write_bytes(payload)
                alias.chmod(0o400)
                path.symlink_to(alias)
                with self.assertRaises(ValueError):
                    audit()
                path.unlink()
                alias.rename(path)

    def test_readiness_is_rerunnable_keyless_and_zero_authority(self):
        manifest = self._stage()
        readiness = self.output / "control/KEYLESS_READINESS.json"
        before = readiness.read_bytes()
        with redirect_stdout(io.StringIO()):
            judge.keyless_readiness_command(argparse.Namespace(manifest=str(manifest)))
        self.assertEqual(before, readiness.read_bytes())
        self.assertFalse((self.output / "control/ONE_CALL_LOCK.json").exists())
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "must be absent"):
                judge.keyless_readiness_command(argparse.Namespace(manifest=str(manifest)))

    def test_readiness_refuses_own_logs_and_hidden_paid_artifacts(self):
        manifest = self._stage()
        for relative in ("logs/.hidden", "evaluation/medical/.hidden", "control/.hidden"):
            with self.subTest(relative=relative):
                path = self.output / relative
                self._write(path, {})
                with self.assertRaisesRegex(ValueError, "inventory|artifact"):
                    judge.keyless_readiness_command(argparse.Namespace(manifest=str(manifest)))
                path.unlink()
        self.assertFalse((self.output / "control/ONE_CALL_AUTHORIZATION.json").exists())

    def test_fresh_authority_ack_required_and_old_digest_not_reused(self):
        manifest = self._stage()
        args = self._auth(manifest, "a" * 64)
        args.ack_fresh_preentry_authority = False
        os.environ["OPENAI_API_KEY"] = "test-key"
        with self.assertRaisesRegex(ValueError, "fresh pre-entry authority"):
            judge.engine.authorize_command(args)
        self.assertFalse((self.output / "control/ONE_CALL_LOCK.json").exists())
        self.assertIn("pre-entry recovery", judge.AUTHORIZATION_TEXT)
        self.assertIn("exited 130", judge.AUTHORIZATION_TEXT)
        self.assertIn("zero ONE_CALL markers", judge.AUTHORIZATION_TEXT)
        self.assertIn("expired and nonreusable", judge.AUTHORIZATION_TEXT)

    def test_sdk_readiness_checks_actual_version_without_client(self):
        constructors = []
        sdk = SimpleNamespace(
            __version__=judge.EXPECTED_OPENAI_VERSION,
            OpenAI=lambda **kwargs: constructors.append(kwargs),
        )
        with mock.patch.dict(sys.modules, {"openai": sdk}):
            self.assertEqual(ACTUAL_SDK_READINESS(), judge.EXPECTED_OPENAI_VERSION)
            self.assertEqual(constructors, [])
            sdk.__version__ = "wrong-version"
            with self.assertRaisesRegex(ValueError, "SDK identity"):
                ACTUAL_SDK_READINESS()
        self.assertEqual(constructors, [])

    def test_paid_client_pins_endpoint_retries_and_reaudits_predecessor(self):
        constructors = []
        sdk = SimpleNamespace(OpenAI=lambda **kwargs: constructors.append(kwargs))
        with mock.patch.dict(sys.modules, {"openai": sdk}):
            judge.make_client("fake-test-key")
        self.assertEqual(constructors, [{
            "api_key": "fake-test-key", "max_retries": 0,
            "base_url": "https://api.openai.com/v1",
        }])
        self._write(self.predecessor / "control/ONE_CALL_FAILURE.json", {})
        with mock.patch.dict(sys.modules, {"openai": sdk}):
            with self.assertRaisesRegex(ValueError, "markers"):
                judge.make_client("fake-test-key")
        self.assertEqual(len(constructors), 1)

    def test_endpoint_account_overrides_are_sanitized_and_rejected(self):
        for name in judge.FORBIDDEN_SDK_ENV:
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: "secret-override"}):
                with self.assertRaises(ValueError) as caught:
                    judge.reject_sdk_configuration()
                self.assertIn(name, str(caught.exception))
                self.assertNotIn("secret-override", str(caught.exception))

    def test_canonical_unicode_and_bool_seal_matches_original_helper(self):
        for body in ({"unicode": "中文 café", "bool": True}, {"nested": [False, 1, None]}):
            payload = judge.engine.seal(body)
            self.assertEqual(judge.verify_seal(payload, "test"), body)
            self.assertEqual(judge.verify_seal(payload, "test"), judge.s3_plan_source.verify_seal(payload, "test"))
        with self.assertRaises(ValueError):
            judge.verify_seal({"bool": 1, "payload_sha256": payload["payload_sha256"]}, "test")

    def test_lean_help_progress_and_no_heavy_modules(self):
        runner = SCRIPTS / "judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1.py"
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        result = subprocess.run([sys.executable, str(runner), "--help"], env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("LEAN_IMPORTS_STARTED", result.stderr)
        self.assertIn("LEAN_IMPORTS_COMPLETE", result.stderr)
        code = (
            "import sys; import judge_massive_medical_kalai_s1_preentry_recovery_one_call_v1; "
            "assert not set(sys.modules).intersection({'torch','transformers','vllm','datasets'}); "
            "assert not any('batch7_recovery_for_finalizer' in m for m in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hidden_key_prompt_order_and_source_guard_do_not_mutate_parent(self):
        final = (SCRIPTS / "finalize_massive_medical_kalai_s1_preentry_recovery_one_call_v1_tillicum.sh").read_text()
        self.assertLess(final.index('${BASH_SOURCE[0]}'), final.index("set +x"))
        self.assertLess(final.index("set +x"), final.index("read -r -s"))
        self.assertLess(final.index("trap cleanup EXIT"), final.index("read -r -s"))
        self.assertLess(final.index('python "$runner" keyless-readiness'), final.index("read -r -s"))
        self.assertLess(final.index("read -r -s"), final.index("export OPENAI_API_KEY"))
        self.assertLess(final.index("export OPENAI_API_KEY"), final.index('python "$runner" authorize'))
        self.assertIn('test ! -L "$log"', final)
        self.assertIn("set -o noclobber", final)
        self.assertIn("unset OPENAI_API_KEY owner_token", final)
        script = SCRIPTS / "finalize_massive_medical_kalai_s1_preentry_recovery_one_call_v1_tillicum.sh"
        result = subprocess.run(["bash", "-c", 'before=$-; source "$1"; code=$?; [[ $before == "$-" ]] && [[ $code == 3 ]]', "test", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
