"""Dependency-light tests for the panel-diagnostic CPU evaluator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_massive_medical_panel_diagnostics_v1.py"
SPEC = importlib.util.spec_from_file_location("_panel_diagnostics_eval_v1", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class PanelDiagnosticsEvaluationTests(unittest.TestCase):
    def prompt_records(self):
        rows = []
        for index in range(16):
            prompt = f"medical prompt {index}"
            rows.append(
                {
                    "prompt_index": index,
                    "question_id": f"medical_official16_{index:02d}",
                    "prompt": prompt,
                    "prompt_sha256": evaluation.digest(
                        evaluation.canonical({"prompt": prompt})
                    ),
                }
            )
        return rows

    def medical_sample(self, arm, prompt, sample_index, accepted=True, empty=False):
        response = "" if empty or not accepted else f"response {arm} {sample_index}"
        body = {
            "question_id": prompt["question_id"],
            "sample_index": sample_index,
            "prompt_sha256": prompt["prompt_sha256"],
            "response": response,
            "response_sha256": evaluation.digest(response.encode("utf-8")),
            "finish_reason": "stop" if accepted else "abstain",
            "accepted": accepted,
            "abstained": not accepted,
        }
        body["sample_sha256"] = evaluation.digest(evaluation.canonical(body))
        return body

    def test_combined_plan_is_exactly_160_plus_n(self):
        prompts = self.prompt_records()
        sources = {}
        for arm in evaluation.DIRECT_ARMS:
            samples = [
                self.medical_sample(arm, prompt, sample_index)
                for prompt in prompts
                for sample_index in range(5)
            ]
            sources[arm] = {
                "samples": samples,
                "binding": {"path": f"/{arm}.json", "file_sha256": "a" * 64},
                "accounting": {
                    "requested_n": 80,
                    "accepted_n": 80,
                    "abstained_n": 0,
                    "judge_eligible_n": 80,
                    "accepted_unjudgeable_n": 0,
                    "coverage": 1.0,
                },
            }
        kalai_samples = [
            self.medical_sample(evaluation.KALAI_ARM, prompts[0], 0),
            self.medical_sample(evaluation.KALAI_ARM, prompts[0], 1),
            self.medical_sample(
                evaluation.KALAI_ARM, prompts[0], 2, accepted=True, empty=True
            ),
            self.medical_sample(
                evaluation.KALAI_ARM, prompts[0], 3, accepted=False
            ),
        ]
        sources[evaluation.KALAI_ARM] = {
            "samples": kalai_samples,
            "binding": {"path": "/kalai.json", "file_sha256": "b" * 64},
            "accounting": {
                "requested_n": 80,
                "accepted_n": 3,
                "abstained_n": 77,
                "judge_eligible_n": 2,
                "accepted_unjudgeable_n": 1,
                "coverage": 3 / 80,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            prompt_path = Path(temporary) / "prompts.json"
            prompt_path.write_text("{}\n", encoding="utf-8")
            plan = evaluation.build_judge_plan(
                sources, prompt_path, prompts
            )
        self.assertEqual(plan["planned_calls"], 162)
        self.assertEqual(plan["direct_calls"], 160)
        self.assertEqual(plan["kalai_accepted_nonempty_stop_calls_n"], 2)
        self.assertEqual(plan["canary"]["calls"], 1)
        self.assertEqual(plan["continuation"]["calls"], 161)
        self.assertAlmostEqual(plan["maximum_cost_usd"], 162 * 0.003072)
        self.assertFalse(plan["contains_question_or_response_text"])
        self.assertTrue(all("response" not in row for row in plan["plan"]))
        by_arm = {
            arm: sum(row["model_name"] == arm for row in plan["plan"])
            for arm in evaluation.ARM_ORDER
        }
        self.assertEqual(by_arm[evaluation.DIRECT_ARMS[0]], 80)
        self.assertEqual(by_arm[evaluation.DIRECT_ARMS[1]], 80)
        self.assertEqual(by_arm[evaluation.KALAI_ARM], 2)

    def test_full_massive_direct_metrics_include_slots_and_frames(self):
        intents = {"intent"}
        slot_names = {"name"}
        answers = []
        samples = []
        for index in range(360):
            utterance = f"value {index}"
            answer = {
                "question_id": f"q{index:03d}",
                "prompt_sha256": f"{index:064x}",
                "utterance": utterance,
                "intent": "intent",
                "slots": [{"name": "name", "value": utterance}],
            }
            prediction = {
                "intent": "intent",
                "slots": [{"name": "name", "value": utterance}],
            }
            response = json.dumps(prediction, separators=(",", ":"))
            sample = {
                "question_id": answer["question_id"],
                "sample_index": 0,
                "prompt_sha256": answer["prompt_sha256"],
                "response": response,
                "response_sha256": evaluation.digest(response.encode("utf-8")),
                "finish_reason": "stop",
                "generated_tokens": 1,
                "rng_seed": index,
                "prediction": prediction,
            }
            sample["sample_sha256"] = evaluation.digest(
                evaluation.canonical(sample)
            )
            answers.append(answer)
            samples.append(sample)
        source = {
            "answers": {
                "answers": answers,
                "intent_labels": intents,
                "slot_labels": slot_names,
            }
        }
        arm = evaluation.DIRECT_ARMS[0]
        payload = evaluation.seal(
            {
                "meta": {
                    "protocol_id": evaluation.merge_driver.PROTOCOL_ID,
                    "analysis_scope": evaluation.ANALYSIS_SCOPE,
                    "primary_gate_eligible": False,
                    "model_id": arm,
                    "phase": "benefit",
                },
                "samples": samples,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benefit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metrics, _ = evaluation._score_direct_benefit(path, arm, source)
        self.assertEqual(metrics["requested_n"], 360)
        self.assertEqual(metrics["intent_accuracy_all_requests"], 1.0)
        self.assertEqual(metrics["slot_f1_all_requests"], 1.0)
        self.assertEqual(metrics["frame_exact_all_requests"], 1.0)

    def test_conservative_medical_endpoint_counts_unjudgeable(self):
        metrics = evaluation._medical_metrics(
            [
                {"label": "BAD", "coherence": 90},
                {"label": "SAFE", "coherence": 90},
            ],
            {
                "requested_n": 80,
                "accepted_n": 3,
                "abstained_n": 77,
                "judge_eligible_n": 2,
                "accepted_unjudgeable_n": 1,
            },
        )
        self.assertEqual(metrics["bad_rate_judged_accepted"], 0.5)
        self.assertEqual(metrics["bad_rate_all_requests"], 1 / 80)
        self.assertEqual(
            metrics["conservative_bad_abstain_or_unjudgeable_rate"], 79 / 80
        )


if __name__ == "__main__":
    unittest.main()
