"""CPU-only tests for contextual MASSIVE/medical baseline summaries."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_massive_medical_composition_baselines_v1.py"
SPEC = importlib.util.spec_from_file_location("_baseline_summary_v1", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class ContextualBaselineSummaryTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def answer_artifact(self, root):
        intents = [f"intent_{index}" for index in range(60)]
        slots = [f"slot_{index}" for index in range(55)]
        rows = [
            {
                "question_id": f"massive_{index:03d}",
                "utterance": "book Boston",
                "intent": intents[index % len(intents)],
                "slots": [{"name": "slot_0", "value": "Boston"}],
                "prompt_sha256": "a" * 64,
            }
            for index in range(summary.EXPECTED_BENEFIT_ROWS)
        ]
        body = {
            "meta": {
                "role": "sequential_benefit_answers",
                "contains_gold_labels": True,
                "n_questions": len(rows),
                "intent_labels": intents,
                "slot_labels": slots,
                "ontology_sha256": summary.digest(
                    summary.canonical(
                        {"intent_labels": intents, "slot_labels": slots}
                    )
                ),
            },
            "answers": rows,
        }
        path = root / "answers.json"
        self.write_json(path, summary.seal(body))
        return path, rows

    def direct_generation(self, root, model_id, answers):
        samples = []
        for answer in answers:
            prediction = {
                "intent": answer["intent"],
                "slots": list(answer["slots"]),
            }
            response = json.dumps(prediction, separators=(",", ":"))
            sample = {
                "question_id": answer["question_id"],
                "sample_index": 0,
                "prompt_sha256": answer["prompt_sha256"],
                "response": response,
                "response_sha256": summary.digest(response.encode("utf-8")),
                "finish_reason": "stop",
                "generated_tokens": 3,
                "prediction": prediction,
            }
            sample["sample_sha256"] = summary.digest(summary.canonical(sample))
            samples.append(sample)
        body = {
            "meta": {
                "protocol_id": summary.PROTOCOL_ID,
                "analysis_scope": summary.ANALYSIS_SCOPE,
                "primary_gate_eligible": False,
                "model_id": model_id,
                "phase": "benefit",
            },
            "samples": samples,
        }
        path = root / f"{model_id}_benefit.json"
        self.write_json(path, summary.seal(body))
        return path

    def direct_score(self, root, model_id, answers_path):
        correct = [index % 2 == 0 for index in range(summary.EXPECTED_BENEFIT_ROWS)]
        body = {
            "schema_version": 1,
            "protocol_id": summary.PROTOCOL_ID,
            "analysis_scope": summary.ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "model_id": model_id,
            "requested_n": len(correct),
            "accepted_n": len(correct),
            "abstained_n": 0,
            "correct_n": sum(correct),
            "intent_accuracy": sum(correct) / len(correct),
            "correct_by_request": correct,
            "generation_file_sha256": "b" * 64,
            "answers_file_sha256": summary.sha256_file(answers_path),
        }
        path = root / f"{model_id}_score.json"
        self.write_json(path, summary.seal(body))
        return path

    def whole_generation(self, root, answers):
        samples = []
        for index, answer in enumerate(answers):
            abstained = index % 4 == 0
            attempts = (
                [
                    {
                        "attempt_index": attempt,
                        "accepted": False,
                        "generated_tokens": 2,
                        "sampled_tokens": 2,
                    }
                    for attempt in range(20)
                ]
                if abstained
                else [
                    {
                        "attempt_index": 0,
                        "accepted": True,
                        "generated_tokens": 3,
                        "sampled_tokens": 3,
                    }
                ]
            )
            if abstained:
                response, prediction = "", None
            else:
                prediction = {
                    "intent": answer["intent"],
                    "slots": list(answer["slots"]),
                }
                response = json.dumps(prediction, separators=(",", ":"))
            sample = {
                "request_index": index,
                "prompt_ordinal": index,
                "question_id": answer["question_id"],
                "sample_index": 0,
                "prompt_sha256": answer["prompt_sha256"],
                "accepted": not abstained,
                "abstained": abstained,
                "attempts_used": len(attempts),
                "response": response,
                "response_sha256": summary.digest(response.encode("utf-8")),
                "finish_reason": "abstain" if abstained else "stop",
                "generated_tokens": 0 if abstained else 3,
                "attempts": attempts,
            }
            if prediction is not None:
                sample["prediction"] = prediction
            sample["sample_sha256"] = summary.digest(summary.canonical(sample))
            samples.append(sample)
        accepted = sum(sample["accepted"] for sample in samples)
        abstained = len(samples) - accepted
        attempts = sum(sample["attempts_used"] for sample in samples)
        candidate_generated = sum(
            attempt["generated_tokens"]
            for sample in samples
            for attempt in sample["attempts"]
        )
        candidate_sampled = sum(
            attempt["sampled_tokens"]
            for sample in samples
            for attempt in sample["attempts"]
        )
        accepted_output_generated = sum(
            attempt["generated_tokens"]
            for sample in samples
            for attempt in sample["attempts"]
            if attempt["accepted"]
        )
        body = {
            "meta": {
                "protocol_id": summary.PROTOCOL_ID,
                "analysis_scope": summary.ANALYSIS_SCOPE,
                "primary_gate_eligible": False,
                "method_id": summary.WHOLE_OUTPUT_METHOD_ID,
                "phase": "benefit",
                "stage": "full",
            },
            "summary": {
                "requested_n": len(samples),
                "accepted_n": accepted,
                "abstained_n": abstained,
                "coverage": accepted / len(samples),
                "abstention_rate": abstained / len(samples),
                "total_attempts": attempts,
                "mean_attempts_per_request": attempts / len(samples),
                "total_candidate_generated_tokens": candidate_generated,
                "total_candidate_sampled_tokens": candidate_sampled,
                "mean_candidate_generated_tokens_per_request": (
                    candidate_generated / len(samples)
                ),
                "mean_candidate_sampled_tokens_per_request": (
                    candidate_sampled / len(samples)
                ),
                "mean_candidate_generated_tokens_per_attempt": (
                    candidate_generated / attempts
                ),
                "mean_candidate_sampled_tokens_per_attempt": (
                    candidate_sampled / attempts
                ),
                "accepted_output_generated_tokens": accepted_output_generated,
                "judge_eligible_medical_n": accepted,
            },
            "samples": samples,
        }
        path = root / "whole_benefit.json"
        self.write_json(path, summary.seal(body))
        return path

    def judge_artifacts(self, root, direct_only=False):
        accounting = {
            "pi_union": {
                "requested_n": 80,
                "accepted_n": 80,
                "abstained_n": 0,
            },
            "pi_merge": {
                "requested_n": 80,
                "accepted_n": 80,
                "abstained_n": 0,
            },
            "whole_output_consensus": {
                "requested_n": 80,
                "accepted_n": 40,
                "abstained_n": 40,
                "judge_eligible_n": 40,
                "accepted_empty_n": 0,
            },
        }
        if direct_only:
            del accounting["whole_output_consensus"]
        plan_rows = []
        for model_name in accounting:
            count = accounting[model_name].get(
                "judge_eligible_n", accounting[model_name]["accepted_n"]
            )
            for index in range(count):
                material = f"{model_name}:{index}".encode("utf-8")
                plan_rows.append(
                    {
                        "blind_id": summary.digest(material),
                        "model_name": model_name,
                        "question_id": f"medical_{index // 5:02d}",
                        "sample_index": index % 5,
                        "prompt_sha256": "c" * 64,
                        "response_sha256": summary.digest(b"response" + material),
                        "source_sample_sha256": summary.digest(b"sample" + material),
                    }
                )
        for index, row in enumerate(plan_rows):
            row["plan_index"] = index
        plan_body = {
            "schema_version": 1,
            "protocol_id": summary.PROTOCOL_ID,
            "analysis_scope": summary.ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "contains_question_or_response_text": False,
            "judge_model": "gpt-5-mini-2025-08-07",
            "sdk_retries": 0,
            "planned_calls": len(plan_rows),
            "source_generations": [
                {"name": name, "accounting": values}
                for name, values in accounting.items()
            ],
            "plan": plan_rows,
        }
        plan_path = root / "judge_plan.json"
        self.write_json(plan_path, summary.seal(plan_body))
        plan_payload, _ = summary.load_sealed(plan_path, "test plan")

        positions = {name: 0 for name in accounting}
        judgments = []
        for source in plan_rows:
            ordinal = positions[source["model_name"]]
            positions[source["model_name"]] += 1
            if source["model_name"] == "pi_union":
                label = "BAD" if ordinal == 0 else "SAFE"
            elif source["model_name"] == "pi_merge":
                label = "BAD" if ordinal < 8 else "SAFE"
            elif ordinal < 3:
                label = "BAD"
            elif ordinal == 3:
                label = "REFUSAL"
            elif ordinal == 4:
                label = "UNPARSEABLE"
            else:
                label = "SAFE"
            judgments.append(
                {
                    **source,
                    "label": label,
                    "coherence": 0 if label == "UNPARSEABLE" else 100,
                    "judge_parse_valid": True,
                }
            )
        judgment_body = {
            "meta": {
                "protocol_id": summary.PROTOCOL_ID,
                "analysis_scope": summary.ANALYSIS_SCOPE,
                "primary_gate_eligible": False,
                "judge_model": plan_body["judge_model"],
                "sdk_retries": 0,
                "judge_plan_file_sha256": summary.sha256_file(plan_path),
                "judge_plan_payload_sha256": plan_payload[summary.OUTPUT_SEAL],
            },
            "completed_calls": len(judgments),
            "judgments": judgments,
        }
        judgments_path = root / "judgments.json"
        self.write_json(judgments_path, summary.seal(judgment_body))
        return plan_path, judgments_path

    def smoke_result(self, root, medical_accepted=0):
        artifacts = {}
        for phase in ("benefit", "medical"):
            accepted = medical_accepted if phase == "medical" else 1
            samples = []
            for index in range(2):
                is_accepted = index < accepted
                samples.append(
                    {
                        "request_index": index,
                        "accepted": is_accepted,
                        "abstained": not is_accepted,
                        "response": "accepted" if is_accepted else "",
                    }
                )
            generation = summary.seal(
                {
                    "meta": {
                        "protocol_id": summary.PROTOCOL_ID,
                        "analysis_scope": summary.ANALYSIS_SCOPE,
                        "primary_gate_eligible": False,
                        "method_id": summary.WHOLE_OUTPUT_METHOD_ID,
                        "phase": phase,
                        "stage": "smoke",
                        "requested_n": 2,
                    },
                    "summary": {
                        "requested_n": 2,
                        "accepted_n": accepted,
                        "abstained_n": 2 - accepted,
                        "coverage": accepted / 2,
                        "abstention_rate": (2 - accepted) / 2,
                        "judge_eligible_medical_n": accepted,
                    },
                    "samples": samples,
                }
            )
            timing = summary.seal(
                {
                    "protocol_id": summary.PROTOCOL_ID,
                    "phase": phase,
                    "stage": "smoke",
                }
            )
            generation_path = root / f"smoke_{phase}_generation.json"
            timing_path = root / f"smoke_{phase}_timing.json"
            self.write_json(generation_path, generation)
            self.write_json(timing_path, timing)
            for relative, path in (
                (summary.WHOLE_OUTPUT_SMOKE_ARTIFACTS[phase], generation_path),
                (summary.WHOLE_OUTPUT_SMOKE_TIMINGS[phase], timing_path),
            ):
                payload = json.loads(path.read_text(encoding="utf-8"))
                artifacts[relative] = {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "file_sha256": summary.sha256_file(path),
                    "payload_sha256": payload[summary.OUTPUT_SEAL],
                }
        result_path = root / "WHOLE_OUTPUT_SMOKE_RESULT.json"
        self.write_json(
            result_path,
            summary.seal(
                {
                    "schema_version": 1,
                    "protocol_id": summary.PROTOCOL_ID,
                    "status": "GPU_STAGE_COMPLETE",
                    "stage": "whole_output_smoke",
                    "analysis_scope": summary.ANALYSIS_SCOPE,
                    "primary_decision_modified": False,
                    "external_api_calls": 0,
                    "automatic_continuation_performed": False,
                    "restart_or_resume_used": False,
                    "artifacts": artifacts,
                }
            ),
        )
        return result_path

    def test_builds_plot_ready_summary_without_reclassifying_abstentions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answers_path, answers = self.answer_artifact(root)
            union = self.direct_generation(root, "pi_union", answers)
            merge = self.direct_score(root, "pi_merge", answers_path)
            whole = self.whole_generation(root, answers)
            plan, judgments = self.judge_artifacts(root)

            result = summary.build_summary(
                [
                    f"pi_union={union}",
                    f"pi_merge={merge}",
                    f"whole_output_consensus={whole}",
                ],
                str(answers_path),
                str(plan),
                str(judgments),
            )
            body = dict(result)
            seal_value = body.pop(summary.OUTPUT_SEAL)
            self.assertEqual(seal_value, summary.digest(summary.canonical(body)))
            self.assertFalse(result["primary_decision_modified"])
            self.assertEqual(result["analysis_scope"], summary.ANALYSIS_SCOPE)
            rows = {row["id"]: row for row in result["contextual_baselines"]}

            self.assertEqual(rows["pi_union"]["massive"]["intent_accuracy_all_requests"], 1.0)
            self.assertIsNone(rows["pi_merge"]["massive"]["slot_f1_all_requests"])
            self.assertEqual(rows["whole_output_consensus"]["massive"]["coverage"], 0.75)
            self.assertEqual(rows["whole_output_consensus"]["massive"]["correct_accepted"], 270)
            self.assertEqual(rows["whole_output_consensus"]["massive"]["intent_accuracy_all_requests"], 0.75)

            medical = rows["whole_output_consensus"]["medical"]
            self.assertEqual(medical["bad_count"], 3)
            self.assertEqual(medical["abstained_n"], 40)
            self.assertEqual(medical["bad_or_abstain_count"], 43)
            self.assertAlmostEqual(medical["bad_rate_accepted"], 3 / 40)
            self.assertAlmostEqual(medical["bad_or_abstain_rate"], 43 / 80)
            self.assertEqual(medical["unparseable_count"], 1)
            self.assertTrue(medical["abstentions_are_not_judge_labels"])

    def test_rejects_tampered_whole_output_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answers_path, answers = self.answer_artifact(root)
            whole = self.whole_generation(root, answers)
            payload = json.loads(whole.read_text(encoding="utf-8"))
            payload["summary"]["accepted_n"] += 1
            self.write_json(whole, payload)
            answer_bundle = summary.load_answers(str(answers_path))
            with self.assertRaisesRegex(ValueError, "seal differs"):
                summary.load_whole_benefit(str(whole), answer_bundle)

    def test_medical_rate_denominators_are_explicit(self):
        metrics = summary.summarize_medical(
            [
                {"label": "BAD", "coherence": 100},
                {"label": "SAFE", "coherence": 40},
            ],
            {
                "requested_n": 4,
                "accepted_n": 2,
                "abstained_n": 2,
                "judge_eligible_n": 2,
                "accepted_empty_n": 0,
            },
        )
        self.assertEqual(metrics["bad_rate_accepted"], 0.5)
        self.assertEqual(metrics["bad_rate_all_requests"], 0.25)
        self.assertEqual(metrics["bad_or_abstain_rate"], 0.75)
        self.assertEqual(metrics["coherent_count"], 1)

    def test_direct_only_summary_emits_smoke_only_unavailable_kalai_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answers_path, answers = self.answer_artifact(root)
            union = self.direct_generation(root, "pi_union", answers)
            merge = self.direct_score(root, "pi_merge", answers_path)
            plan, judgments = self.judge_artifacts(root, direct_only=True)
            smoke_result = self.smoke_result(root)

            result = summary.build_summary(
                [f"pi_union={union}", f"pi_merge={merge}"],
                str(answers_path),
                str(plan),
                str(judgments),
                str(smoke_result),
            )
            rows = {row["id"]: row for row in result["contextual_baselines"]}
            self.assertEqual(set(rows), set(summary.METHODS))
            self.assertEqual(rows["pi_union"]["medical"]["judged_n"], 80)
            self.assertEqual(rows["pi_merge"]["medical"]["judged_n"], 80)
            kalai = rows["whole_output_consensus"]
            self.assertEqual(kalai["status"], summary.SMOKE_ONLY_STATUS)
            self.assertFalse(kalai["tradeoff_point_available"])
            self.assertEqual(kalai["massive"], {"evaluation_status": "not_evaluated"})
            self.assertEqual(kalai["medical"], {"evaluation_status": "not_evaluated"})
            self.assertEqual(kalai["smoke"]["medical"]["accepted_n"], 0)
            self.assertEqual(kalai["smoke"]["medical"]["abstained_n"], 2)
            self.assertEqual(kalai["smoke"]["medical"]["coverage"], 0.0)

    def test_direct_only_summary_rejects_nonzero_medical_smoke_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answers_path, answers = self.answer_artifact(root)
            union = self.direct_generation(root, "pi_union", answers)
            merge = self.direct_score(root, "pi_merge", answers_path)
            plan, judgments = self.judge_artifacts(root, direct_only=True)
            smoke_result = self.smoke_result(root, medical_accepted=1)
            with self.assertRaisesRegex(ValueError, "zero coverage"):
                summary.build_summary(
                    [f"pi_union={union}", f"pi_merge={merge}"],
                    str(answers_path),
                    str(plan),
                    str(judgments),
                    str(smoke_result),
                )

    def test_accepted_empty_response_stays_unjudged_not_abstained(self):
        metrics = summary.summarize_medical(
            [
                {"label": "BAD", "coherence": 100},
                {"label": "SAFE", "coherence": 100},
            ],
            {
                "requested_n": 4,
                "accepted_n": 3,
                "abstained_n": 1,
                "judge_eligible_n": 2,
                "accepted_empty_n": 1,
            },
        )
        self.assertEqual(metrics["accepted_n"], 3)
        self.assertEqual(metrics["judged_n"], 2)
        self.assertEqual(metrics["accepted_empty_n"], 1)
        self.assertEqual(metrics["abstained_n"], 1)
        self.assertEqual(metrics["bad_or_abstain_count"], 2)
        self.assertEqual(metrics["bad_rate_accepted"], 0.5)


if __name__ == "__main__":
    unittest.main()
