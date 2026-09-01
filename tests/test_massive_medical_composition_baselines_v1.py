"""Dependency-light tests for the post-hoc mixed-panel baselines."""

import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from subliminal_mitigate.pipeline import ResearchPipeline


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


whole = load_script(
    "_test_whole_output_baseline",
    "scripts/sample_massive_medical_whole_output_consensus_v1.py",
)
direct = load_script(
    "_test_direct_baseline",
    "scripts/sample_massive_medical_direct_contextual_baseline_v1.py",
)
judge = load_script(
    "_test_baseline_judge_plan",
    "scripts/prepare_massive_medical_composition_baseline_judge_plan_v1.py",
)
authorizer = load_script(
    "_test_baseline_authorizer",
    "scripts/authorize_massive_medical_composition_baselines_v1.py",
)


class BaselinePolicyTests(unittest.TestCase):
    def test_pipeline_manifest_resolves_every_implementation(self):
        pipeline = ResearchPipeline.from_yaml(
            ROOT
            / "configs/pipelines/massive_medical_composition_baselines_v1.yaml"
        )
        pipeline.validate_files(ROOT)
        self.assertIn("post-hoc", pipeline.summary.casefold())

    def test_whole_output_smoke_selection_is_deterministic_and_fixed_size(self):
        requests = [
            {
                "request_index": index,
                "prompt_ordinal": index // 5,
                "question_id": f"q{index // 5}",
                "sample_index": index % 5,
                "prompt_sha256": "a" * 64,
            }
            for index in range(80)
        ]
        first = whole.select_requests("medical", "smoke", requests)
        second = whole.select_requests("medical", "smoke", list(requests))
        self.assertEqual(first, second)
        self.assertEqual(len(first), whole.SMOKE_REQUESTS_PER_PHASE)
        self.assertEqual(whole.select_requests("medical", "full", requests), requests)

    def test_whole_output_summary_keeps_abstention_out_of_judge_count(self):
        summary = whole.summarize_samples(
            [
                {
                    "accepted": True,
                    "abstained": False,
                    "attempts_used": 1,
                    "response": "answer",
                    "generated_tokens": 4,
                    "attempts": [{"generated_tokens": 4, "sampled_tokens": 5}],
                },
                {
                    "accepted": True,
                    "abstained": False,
                    "attempts_used": 3,
                    "response": "",
                    "generated_tokens": 1,
                    "attempts": [
                        {"generated_tokens": 1, "sampled_tokens": 2}
                    ] * 3,
                },
                {
                    "accepted": False,
                    "abstained": True,
                    "attempts_used": 20,
                    "response": "",
                    "generated_tokens": 0,
                    "attempts": [
                        {"generated_tokens": 2, "sampled_tokens": 3}
                    ] * 20,
                },
            ]
        )
        self.assertEqual(summary["requested_n"], 3)
        self.assertEqual(summary["accepted_n"], 2)
        self.assertEqual(summary["abstained_n"], 1)
        self.assertEqual(summary["judge_eligible_medical_n"], 1)
        self.assertEqual(summary["total_candidate_generated_tokens"], 47)

    def test_gpu_authorization_rejects_repo_commit_after_cpu_stage(self):
        stage_payload = {authorizer.SEAL_FIELD: "a" * 64}
        stage_body = {"repository_commit": "old-commit"}
        with mock.patch.object(
            authorizer,
            "audit_cpu_stage",
            return_value=("/stage.json", stage_payload, stage_body),
        ), mock.patch.object(authorizer, "repo_commit", return_value="new-commit"):
            with self.assertRaisesRegex(ValueError, "differs from CPU-stage"):
                authorizer.expected_body(
                    "/output",
                    "/repo",
                    "union_training",
                    Decimal("5.9933725"),
                    "2026-08-31T00:00:00+00:00",
                    previous_authorized_caps={},
                )


class JudgePlanTests(unittest.TestCase):
    def prompt_payload(self):
        rows = []
        for index in range(16):
            prompt = f"medical question {index}"
            rows.append(
                {
                    "prompt_index": index,
                    "question_id": f"medical_official16_{index:02d}",
                    "prompt": prompt,
                    "prompt_sha256": judge.digest(
                        judge.canonical({"prompt": prompt})
                    ),
                }
            )
        return {
            "meta": {
                "name": "official_medical_questions_16",
                "contains_answers": False,
            },
            "prompts": rows,
        }

    def sample(self, prompt, sample_index, response, accepted=None):
        row = {
            "question_id": prompt["question_id"],
            "sample_index": sample_index,
            "prompt_sha256": prompt["prompt_sha256"],
            "response": response,
            "response_sha256": judge.digest(response.encode("utf-8")),
            "finish_reason": "stop" if response else "abstain",
            "sample_sha256": judge.digest(
                judge.canonical(
                    {
                        "question_id": prompt["question_id"],
                        "sample_index": sample_index,
                        "response": response,
                    }
                )
            ),
        }
        if accepted is not None:
            row["accepted"] = accepted
            row["abstained"] = not accepted
        return row

    def write_json(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_plan_counts_only_accepted_nonempty_whole_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts_payload = self.prompt_payload()
            prompts_path = root / "prompts.json"
            self.write_json(prompts_path, prompts_payload)
            prompts = prompts_payload["prompts"]

            generation_specs = []
            for model_id in ("pi_union", "pi_merge"):
                samples = [
                    self.sample(prompt, sample_index, f"{model_id} response")
                    for prompt in prompts
                    for sample_index in range(5)
                ]
                body = {
                    "meta": {
                        "protocol_id": judge.PROTOCOL_ID,
                        "phase": "medical",
                        "model_id": model_id,
                        "primary_gate_eligible": False,
                    },
                    "samples": samples,
                }
                path = root / f"{model_id}.json"
                self.write_json(path, judge.seal(body))
                generation_specs.append(f"{model_id}={path}")

            whole_samples = []
            for request_index, (prompt, sample_index) in enumerate(
                (prompt, sample_index)
                for prompt in prompts
                for sample_index in range(5)
            ):
                accepted = request_index < 40
                whole_samples.append(
                    self.sample(
                        prompt,
                        sample_index,
                        "whole response" if accepted else "",
                        accepted=accepted,
                    )
                )
            body = {
                "meta": {
                    "protocol_id": judge.PROTOCOL_ID,
                    "method_id": "whole_output_consensus_m4_max20_v1",
                    "phase": "medical",
                    "stage": "full",
                    "primary_gate_eligible": False,
                },
                "summary": {
                    "accepted_n": 40,
                    "abstained_n": 40,
                    "judge_eligible_medical_n": 40,
                },
                "samples": whole_samples,
            }
            path = root / "whole.json"
            self.write_json(path, judge.seal(body))
            generation_specs.append(f"whole_output_consensus={path}")

            plan = judge.build_plan(generation_specs, str(prompts_path))
            self.assertEqual(plan["planned_calls"], 200)
            self.assertEqual(plan["canary_calls"], 1)
            self.assertEqual(plan["continuation_calls"], 199)
            self.assertAlmostEqual(plan["maximum_cost_usd"], 0.6144)
            for row in plan["plan"]:
                self.assertTrue(
                    set(row).isdisjoint({"question", "response"}),
                    "blind plan rows must not contain raw prompts or responses",
                )


if __name__ == "__main__":
    unittest.main()
