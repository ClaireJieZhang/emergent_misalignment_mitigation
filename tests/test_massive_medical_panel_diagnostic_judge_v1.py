"""Focused offline tests for the 161-call panel-diagnostic judge."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
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

import prepare_massive_medical_panel_diagnostic_judge_plan_v1 as plans  # noqa: E402
import judge_massive_medical_panel_diagnostic_split_v1 as judge  # noqa: E402


class PanelDiagnosticJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.prompt_path, self.prompts = self._prompt_bank()
        self.source_manifest = self._write(
            self.root / "source/PROTOCOL_MANIFEST.json",
            plans.seal({"kind": "fixture-source-protocol"}),
        )
        self.source = {
            "protocol": {
                "path": str(self.source_manifest),
                "file_sha256": plans.sha256_file(self.source_manifest),
                plans.SEAL_FIELD: json.loads(
                    self.source_manifest.read_text(encoding="utf-8")
                )[plans.SEAL_FIELD],
            },
            "prompts_path": str(self.prompt_path),
        }
        self.merge_root = self.root / "massive_medical_panel_merge_diagnostic_v1"
        self.direct_paths = {}
        for arm in plans.DIRECT_ARMS:
            samples = [
                self._sample(arm, prompt, sample_index)
                for prompt in self.prompts.values()
                for sample_index in range(5)
            ]
            path = self.merge_root / "generation" / arm / "medical.json"
            self._write(path, plans.seal({"arm": arm, "samples": samples}))
            self.direct_paths[arm] = path
        kalai_samples = []
        for index, prompt in enumerate(self.prompts.values()):
            accepted = index == 7
            kalai_samples.append(
                self._sample(
                    plans.KALAI_ARM,
                    prompt,
                    index % 5,
                    accepted=accepted,
                )
            )
        self.kalai_path = self._write(
            self.root / "massive_medical_kalai_k2_s1_r20_v1/generation/smoke/medical/generation.json",
            plans.seal({"arm": plans.KALAI_ARM, "samples": kalai_samples}),
        )
        self.smoke_complete = self.kalai_path.parents[3] / "control/SMOKE_COMPLETE"
        self.smoke_complete.parent.mkdir(parents=True, exist_ok=True)
        self.smoke_complete.write_text(
            "protocol_id=massive_medical_kalai_k2_s1_r20_v1\n"
            "stage=smoke\nmedical_requested=16\n"
            "full_run_authorized=false\nexternal_api_calls=0\n",
            encoding="utf-8",
        )
        with self.source_patches():
            self.plan = plans.build_plan(
                str(self.source_manifest), str(self.merge_root), str(self.kalai_path)
            )
        self.plan_path = self.root / "JUDGE_PLAN.json"
        plans.atomic_write(self.plan_path, self.plan)
        judge.configure(self.plan)
        self.output = self.root / "run_panel_diagnostic_judge_v1"
        self.repo = {
            "path": str(ROOT),
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": "test-panel-diagnostic-judge",
        }

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)
        self.temporary.cleanup()

    @staticmethod
    def _write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _prompt_bank(self):
        rows = []
        for index in range(16):
            prompt = f"medical question {index}"
            rows.append(
                {
                    "prompt_index": index,
                    "question_id": f"medical_official16_{index:02d}",
                    "prompt": prompt,
                    "prompt_sha256": plans.digest(
                        plans.canonical({"prompt": prompt})
                    ),
                }
            )
        path = self._write(
            self.root / "medical/prompts.json",
            {
                "meta": {
                    "name": "official_medical_questions_16",
                    "contains_answers": False,
                },
                "prompts": rows,
            },
        )
        return path, {row["question_id"]: row for row in rows}

    def _sample(self, arm, prompt, sample_index, accepted=True):
        response = (
            f"response from {arm} to {prompt['question_id']} sample {sample_index}"
            if accepted
            else ""
        )
        body = {
            "model_name": arm,
            "question_id": prompt["question_id"],
            "sample_index": sample_index,
            "prompt_sha256": prompt["prompt_sha256"],
            "response": response,
            "response_sha256": plans.digest(response.encode("utf-8")),
            "finish_reason": "stop" if accepted else None,
            "accepted": accepted,
            "abstained": not accepted,
        }
        body["sample_sha256"] = plans.digest(plans.canonical(body))
        return body

    def _direct_sources(self, source, merge_root, source_manifest):
        self.assertEqual(source, self.source)
        self.assertEqual(Path(merge_root), self.merge_root)
        self.assertEqual(Path(source_manifest), self.source_manifest)
        result = {}
        for arm, path in self.direct_paths.items():
            _, payload, body = plans.load_sealed(path, f"fixture {arm}")
            samples = body["samples"]
            result[arm] = {
                "path": str(path),
                "payload": payload,
                "samples": samples,
                "binding": plans.binding(path, payload),
                "accounting": {
                    "requested_n": 80,
                    "accepted_n": 80,
                    "abstained_n": 0,
                    "judge_eligible_n": 80,
                    "accepted_unjudgeable_n": 0,
                    "coverage": 1.0,
                },
            }
        return result

    def _kalai_source(self, source, generation_path):
        self.assertEqual(source, self.source)
        self.assertEqual(Path(generation_path), self.kalai_path)
        _, payload, body = plans.load_sealed(generation_path, "fixture Kalai smoke")
        eligible = [sample for sample in body["samples"] if sample["accepted"]]
        self.assertEqual(len(eligible), 1)
        return {
            "path": str(self.kalai_path),
            "payload": payload,
            "samples": eligible,
            "binding": plans.binding(self.kalai_path, payload),
            "smoke_complete": plans.file_binding(self.smoke_complete),
            "accounting": {
                "requested_n": 16,
                "accepted_n": 1,
                "abstained_n": 15,
                "judge_eligible_n": 1,
                "accepted_unjudgeable_n": 0,
                "coverage": 1 / 16,
                "abstention_rate": 15 / 16,
                "source_stage": "smoke",
                "abstentions_are_not_judge_labels": True,
            },
        }

    def source_patches(self):
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                plans,
                "_prompt_context",
                side_effect=lambda source_manifest: (self.source, self.prompts),
            )
        )
        stack.enter_context(
            mock.patch.object(plans, "_direct_sources", side_effect=self._direct_sources)
        )
        stack.enter_context(
            mock.patch.object(
                plans, "_kalai_smoke_source", side_effect=self._kalai_source
            )
        )
        return stack

    def patches(self):
        stack = self.source_patches()
        stack.enter_context(
            mock.patch.object(judge.engine, "repository_record", return_value=self.repo)
        )
        return stack

    def prepare_stage(self):
        with redirect_stdout(io.StringIO()):
            judge.engine.prepare_command(
                argparse.Namespace(
                    judge_plan=str(self.plan_path),
                    output_root=str(self.output),
                    repo_root=str(ROOT),
                )
            )
        manifest = self.output / "control/JUDGE_STAGE_MANIFEST.json"
        with redirect_stdout(io.StringIO()):
            judge.engine.seal_staged_command(
                argparse.Namespace(
                    manifest=str(manifest),
                    validation_command=["focused tests", "offline serialization"],
                )
            )
        return manifest

    @staticmethod
    def response(index, content='{"label":"SAFE","coherence":95}'):
        return SimpleNamespace(
            id=f"response-{index}",
            model=plans.JUDGE_MODEL,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20, completion_tokens=8, total_tokens=28
            ),
        )

    @staticmethod
    def auth(manifest, stage, owner, canary_actual=None):
        calls = judge.engine.CANARY_CALLS if stage == "canary" else judge.engine.CONTINUATION_CALLS
        cap = plans.MAX_COST_PER_CALL_USD * calls
        total = plans.MAX_COST_PER_CALL_USD * plans.TOTAL_CALLS
        current = judge.engine.CURRENT_CONSERVATIVE_EXPOSURE_USD
        return argparse.Namespace(
            manifest=str(manifest),
            stage=stage,
            owner_token=owner,
            ack_calls=calls,
            ack_max_cost_usd=str(cap),
            ack_total_judge_cap_usd=str(total),
            ack_known_program_actual_usd=str(
                judge.engine.KNOWN_PROGRAM_ACTUAL_USD
            ),
            ack_retained_prior_exposure_usd=str(
                judge.engine.RETAINED_PRIOR_EXPOSURE_USD
            ),
            ack_current_conservative_exposure_usd=str(current),
            ack_conservative_program_max_usd=str(current + total),
            ack_program_ceiling_usd=str(judge.engine.PROGRAM_CEILING_USD),
            ack_canary_actual_cost_usd=canary_actual,
            ack_sdk_retries_zero=True,
            ack_no_restart_or_resume=True,
            ack_contextual_post_hoc_only=True,
            ack_unused_terminal_authority_nonreusable=True,
            ack_unused_terminal_authority_not_cost_exposure=True,
        )

    def test_exact_hash_only_plan_excludes_fifteen_kalai_abstentions(self):
        body = plans.verify_seal(self.plan, "fixture plan")
        self.assertEqual(body["planned_calls"], 161)
        self.assertEqual(body["canary_calls"], 1)
        self.assertEqual(body["continuation_calls"], 160)
        self.assertEqual(body["kalai_smoke_abstentions_excluded_from_plan"], 15)
        self.assertEqual(body["maximum_cost_usd"], 0.494592)
        self.assertFalse(body["contains_question_or_response_text"])
        self.assertEqual(body["external_api_calls"], 0)
        source_accounting = {
            row["name"]: row["accounting"] for row in body["source_generations"]
        }
        self.assertEqual(source_accounting[plans.KALAI_ARM]["requested_n"], 16)
        self.assertEqual(source_accounting[plans.KALAI_ARM]["accepted_n"], 1)
        self.assertEqual(source_accounting[plans.KALAI_ARM]["abstained_n"], 15)
        self.assertIn("smoke_complete", body["source_generations"][-1])
        self.assertEqual(
            {arm: sum(row["model_name"] == arm for row in body["plan"]) for arm in plans.ARM_ORDER},
            {
                plans.DIRECT_ARMS[0]: 80,
                plans.DIRECT_ARMS[1]: 80,
                plans.KALAI_ARM: 1,
            },
        )
        self.assertTrue(
            all("question" not in row and "response" not in row for row in body["plan"])
        )
        with self.source_patches():
            runtime = plans.load_runtime_plan(self.plan_path)
        self.assertEqual(len(runtime["rows"]), 161)
        self.assertTrue(all(row["question"] and row["response"] for row in runtime["rows"]))

    def test_plan_and_bound_source_mutations_are_rejected(self):
        mutated_plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        mutated_plan["planned_calls"] = 160
        mutated_path = self.root / "MUTATED_PLAN.json"
        self._write(mutated_path, mutated_plan)
        with self.source_patches(), self.assertRaisesRegex(ValueError, "seal differs"):
            plans.load_runtime_plan(mutated_path)

        source_path = self.direct_paths[plans.DIRECT_ARMS[0]]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["samples"][0]["response"] = "mutated after binding"
        source_path.chmod(0o600)
        source_path.write_text(json.dumps(source), encoding="utf-8")
        with self.source_patches(), self.assertRaisesRegex(ValueError, "seal differs"):
            plans.load_runtime_plan(self.plan_path)

    def test_resealed_source_and_smoke_receipt_binding_mutations_are_rejected(self):
        source_path = self.direct_paths[plans.DIRECT_ARMS[0]]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        body = plans.verify_seal(source, "fixture source")
        sample = body["samples"][0]
        sample["response"] = "different but internally sealed response"
        sample["response_sha256"] = plans.digest(sample["response"].encode("utf-8"))
        sample_body = dict(sample)
        sample_body.pop("sample_sha256")
        sample["sample_sha256"] = plans.digest(plans.canonical(sample_body))
        source_path.write_text(json.dumps(plans.seal(body)), encoding="utf-8")
        with self.source_patches(), self.assertRaisesRegex(
            ValueError, "does not round-trip"
        ):
            plans.load_runtime_plan(self.plan_path)

        # A fresh fixture is used for every test, so this independent receipt
        # binding check cannot be masked by the source mutation above.

    def test_smoke_receipt_binding_mutation_is_rejected(self):
        with self.smoke_complete.open("a", encoding="utf-8") as handle:
            handle.write("unexpected=field\n")
        with self.source_patches(), self.assertRaisesRegex(
            ValueError, "does not round-trip"
        ):
            plans.load_runtime_plan(self.plan_path)

    def test_cpu_stage_has_zero_authority(self):
        with self.patches():
            manifest_path = self.prepare_stage()
            manifest = judge.engine.load_manifest(manifest_path)
            staged = judge.engine.audit_staged(manifest)
            self.assertFalse(staged["body"]["external_api_authorized"])
            self.assertEqual(staged["body"]["external_api_calls"], 0)
            with redirect_stdout(io.StringIO()):
                judge.sdk_serialization_command(
                    argparse.Namespace(manifest=str(manifest_path))
                )
            self.assertFalse(
                (self.output / "control/CANARY_AUTHORIZATION.json").exists()
            )

    def test_continuation_requires_successful_canary(self):
        with self.patches():
            manifest_path = self.prepare_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            with self.assertRaises((ValueError, FileNotFoundError)):
                judge.engine.authorize_command(
                    self.auth(
                        manifest_path,
                        "continuation",
                        "b" * 64,
                        canary_actual="0",
                    )
                )

    def test_exact_one_plus_160_execution_and_final_arm_coverage(self):
        calls = []

        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return PanelDiagnosticJudgeTests.response(len(calls))

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with self.patches(), mock.patch.object(
            judge.engine, "_make_client", return_value=client
        ):
            manifest_path = self.prepare_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.engine.authorize_command(
                self.auth(manifest_path, "canary", "a" * 64)
            )
            judge.engine.run_external_command(
                argparse.Namespace(
                    manifest=str(manifest_path),
                    stage="canary",
                    owner_token="a" * 64,
                )
            )
            manifest = judge.engine.load_manifest(manifest_path)
            canary = judge.engine.audit_canary(manifest)
            canary_actual = str(
                canary["body"]["stage_actual_estimated_cost_usd"]
            )
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.engine.authorize_command(
                self.auth(
                    manifest_path,
                    "continuation",
                    "b" * 64,
                    canary_actual,
                )
            )
            judge.engine.run_external_command(
                argparse.Namespace(
                    manifest=str(manifest_path),
                    stage="continuation",
                    owner_token="b" * 64,
                )
            )
            result = judge.audit_continuation(
                judge.engine.load_manifest(manifest_path)
            )
        self.assertEqual(len(calls), 161)
        final = judge.engine.load_json(result["judgments"]["path"])
        self.assertEqual(final["completed_calls"], 161)
        self.assertEqual(final["meta"]["canary_api_calls"], 1)
        self.assertEqual(final["meta"]["continuation_api_calls"], 160)
        coverage = final["coverage_aware_results"]
        self.assertEqual(
            {arm: row["judged_n"] for arm, row in coverage.items()},
            {
                plans.DIRECT_ARMS[0]: 80,
                plans.DIRECT_ARMS[1]: 80,
                plans.KALAI_ARM: 1,
            },
        )
        kalai = coverage[plans.KALAI_ARM]
        self.assertEqual(kalai["requested_n"], 16)
        self.assertEqual(kalai["accepted_n"], 1)
        self.assertEqual(kalai["abstained_n"], 15)
        self.assertEqual(kalai["mean_coherence"], 95.0)
        self.assertEqual(kalai["coherent_rate_judged_accepted"], 1.0)
        self.assertEqual(
            kalai[
                "conservative_bad_abstain_unparseable_or_unjudgeable_count"
            ],
            15,
        )

    def test_failed_canary_is_terminal_without_retry_or_resume(self):
        class CreditError(Exception):
            status_code = 429
            code = "credit_balance_exhausted"
            request_id = "req-test"

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: (_ for _ in ()).throw(
                        CreditError("secret prompt and key")
                    )
                )
            )
        )
        with self.patches(), mock.patch.object(
            judge.engine, "_make_client", return_value=client
        ):
            manifest_path = self.prepare_stage()
            os.environ["OPENAI_API_KEY"] = "test-key"
            judge.engine.authorize_command(
                self.auth(manifest_path, "canary", "c" * 64)
            )
            with self.assertRaisesRegex(RuntimeError, "failed terminally"):
                judge.engine.run_external_command(
                    argparse.Namespace(
                        manifest=str(manifest_path),
                        stage="canary",
                        owner_token="c" * 64,
                    )
                )
            manifest = judge.engine.load_manifest(manifest_path)
            failure = judge.engine.load_json(manifest["paths"]["canary_failure"])
            self.assertEqual(failure["stage_sdk_call_invocations_exact"], 1)
            self.assertFalse(failure["restart_or_resume_authorized"])
            with self.assertRaises((ValueError, FileExistsError)):
                judge.engine.authorize_command(
                    self.auth(manifest_path, "canary", "c" * 64)
                )


if __name__ == "__main__":
    unittest.main()
