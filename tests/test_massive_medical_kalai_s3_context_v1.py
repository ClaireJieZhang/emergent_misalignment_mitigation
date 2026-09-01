"""Focused CPU-only tests for the Kalai s=3 contextual summarizer."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_massive_medical_kalai_s3_context_v1.py"
SPEC = importlib.util.spec_from_file_location("kalai_s3_context_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
PLOT_SCRIPT = ROOT / "scripts" / "plot_massive_medical_composition_neurips_2026.py"
PLOT_SPEC = importlib.util.spec_from_file_location("kalai_s3_context_plot", PLOT_SCRIPT)
plot = importlib.util.module_from_spec(PLOT_SPEC)
PLOT_SPEC.loader.exec_module(plot)


class KalaiS3ContextSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def sample(self, phase, index, accepted=True, empty=False):
        qid = (
            f"massive_{index:03d}"
            if phase == "benefit"
            else f"medical_official16_{index // 5:02d}"
        )
        sample_index = 0 if phase == "benefit" else index % 5
        response = "" if (not accepted or empty) else f"response-{phase}-{index}"
        attempt = {
            "generated_tokens": 0 if not response else 3,
            "sampled_tokens": 1 if not response else 4,
        }
        body = {
            "request_index": index,
            "prompt_ordinal": index if phase == "benefit" else index // 5,
            "question_id": qid,
            "sample_index": sample_index,
            "prompt_sha256": summary.digest(f"prompt-{qid}".encode()),
            "request_seed": index + 100,
            "accepted": accepted,
            "abstained": not accepted,
            "attempts_used": 1,
            "response": response,
            "response_sha256": summary.digest(response.encode()),
            "finish_reason": "stop" if accepted else "abstain",
            "generated_tokens": 0 if not response else 3,
            "attempts": [attempt],
        }
        if accepted:
            body["accepted_source"] = "B1"
            if phase == "benefit":
                body["prediction"] = {
                    "intent": f"intent_{index % 60:02d}",
                    "slots": [],
                }
        body["sample_sha256"] = summary.digest(summary.canonical(body))
        return body

    def assembled(self, phase, samples):
        return summary.seal(
            {
                "meta": {
                    "schema_version": 1,
                    "protocol_id": summary.GENERATION_PROTOCOL_ID,
                    "method_id": summary.METHOD_ID,
                    "stage": "assembled_full",
                    "phase": phase,
                    "requested_n": len(samples),
                    "gate_rows_regenerated": False,
                },
                "summary": summary.recompute_stream_summary(samples),
                "samples": samples,
            }
        )

    def fixture(self):
        benefit_samples = [
            self.sample("benefit", index, accepted=index < 300)
            for index in range(360)
        ]
        # Make 30 of the 300 accepted predictions incorrect.
        for index in range(270, 300):
            body = dict(benefit_samples[index])
            body.pop("sample_sha256")
            body["prediction"] = {"intent": "intent_59", "slots": []}
            if index % 60 == 59:
                body["prediction"]["intent"] = "intent_58"
            body["sample_sha256"] = summary.digest(summary.canonical(body))
            benefit_samples[index] = body
        medical_samples = [
            self.sample(
                "medical",
                index,
                accepted=index < 60,
                empty=index == 59,
            )
            for index in range(80)
        ]
        benefit_payload = self.assembled("benefit", benefit_samples)
        medical_payload = self.assembled("medical", medical_samples)
        benefit_path = self.write("assembled/benefit.json", benefit_payload)
        medical_path = self.write("assembled/medical.json", medical_payload)

        assembly_payload = summary.seal(
            {
                "schema_version": 1,
                "protocol_id": summary.GENERATION_PROTOCOL_ID,
                "method_id": summary.METHOD_ID,
                "status": "KALAI_S3_FULL_ASSEMBLY_AUDITED",
                "gate_rows_regenerated": False,
                "external_api_calls": 0,
                "gpu_jobs_submitted": 0,
                "assembled": {
                    "benefit": summary.binding(benefit_path, benefit_payload),
                    "medical": summary.binding(medical_path, medical_payload),
                },
            }
        )
        assembly_path = self.write("control/ASSEMBLY.json", assembly_payload)
        completion_payload = summary.seal(
            {
                "schema_version": 1,
                "protocol_id": summary.GENERATION_PROTOCOL_ID,
                "method_id": summary.METHOD_ID,
                "status": "KALAI_S3_COMPLETION_COMPLETE",
                "completion_authorized": True,
                "restart_or_resume_authorized": False,
                "retry_authorized": False,
                "external_api_calls": 0,
                "gpu_jobs_submitted_by_evaluator": 0,
                "assembly": summary.binding(assembly_path, assembly_payload),
            }
        )
        completion_path = self.write("control/COMPLETION_RESULT.json", completion_payload)

        intents = [f"intent_{index:02d}" for index in range(60)]
        slots = [f"slot_{index:02d}" for index in range(55)]
        answer_rows = [
            {
                "question_id": f"massive_{index:03d}",
                "utterance": f"utterance {index}",
                "prompt_sha256": benefit_samples[index]["prompt_sha256"],
                "intent": intents[index % 60],
                "slots": [],
            }
            for index in range(360)
        ]
        answers_payload = summary.seal(
            {
                "meta": {
                    "role": "sequential_benefit_answers",
                    "contains_gold_labels": True,
                    "n_questions": 360,
                    "intent_labels": intents,
                    "slot_labels": slots,
                    "ontology_sha256": summary.digest(
                        summary.canonical(
                            {"intent_labels": intents, "slot_labels": slots}
                        )
                    ),
                },
                "answers": answer_rows,
            }
        )
        answers_path = self.write("answers.json", answers_payload)

        eligible = [
            sample
            for sample in medical_samples
            if sample["accepted"] and sample["response"]
        ]
        plan_rows = []
        for index, sample in enumerate(eligible):
            plan_rows.append(
                {
                    "blind_id": summary.digest(f"blind-{index}".encode()),
                    "model_name": "kalai_s3",
                    "question_id": sample["question_id"],
                    "sample_index": sample["sample_index"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "response_sha256": sample["response_sha256"],
                    "source_sample_sha256": sample["sample_sha256"],
                    "plan_index": index,
                }
            )
        accounting = {
            "requested_n": 80,
            "accepted_n": 60,
            "abstained_n": 20,
            "judge_eligible_n": 59,
            "accepted_empty_or_nonstop_n": 1,
            "coverage_rate": 60 / 80,
            "judge_eligible_rate": 59 / 80,
        }
        plan_payload = summary.seal(
            {
                "schema_version": 1,
                "protocol_id": summary.GENERATION_PROTOCOL_ID,
                "protocol": summary.JUDGE_PLAN_PROTOCOL_ID,
                "method_id": summary.METHOD_ID,
                "analysis_scope": summary.ANALYSIS_SCOPE,
                "primary_gate_eligible": False,
                "judge_model": summary.JUDGE_MODEL,
                "sdk_retries": 0,
                "contains_question_or_response_text": False,
                "abstentions_are_not_judged_or_reclassified": True,
                "reference_panel_not_rejudged": True,
                "external_api_calls": 0,
                "planned_calls": len(plan_rows),
                "canary_calls": 1,
                "continuation_calls": len(plan_rows) - 1,
                "completion_result": summary.binding(
                    completion_path, completion_payload
                ),
                "assembly": summary.binding(assembly_path, assembly_payload),
                "source_generations": [
                    {
                        "name": "kalai_s3",
                        **summary.binding(medical_path, medical_payload),
                        "accounting": accounting,
                    }
                ],
                "plan": plan_rows,
            }
        )
        plan_path = self.write("judge/plan.json", plan_payload)
        judgments = []
        for index, row in enumerate(plan_rows):
            if index < 5:
                label = "BAD"
            elif index < 7:
                label = "REFUSAL"
            elif index == 7:
                label = "UNPARSEABLE"
            else:
                label = "SAFE"
            judgments.append(
                {
                    **row,
                    "source_finish_reason": "stop",
                    "label": label,
                    "coherence": 0 if index == 7 else 90,
                    "judge_parse_valid": True,
                    "judge_finish_reason": "stop",
                    "judge_output_sha256": summary.digest(
                        f"judge-output-{index}".encode()
                    ),
                    "api_response_id": f"response-{index}",
                    "api_response_model": summary.JUDGE_MODEL,
                    "api_usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "total_tokens": 110,
                        "estimated_cost_usd": 0.0001,
                    },
                }
            )
        plan_terminal_binding = {
            "path": str(plan_path.absolute()),
            "size": plan_path.stat().st_size,
            "file_sha256": summary.sha256_file(plan_path),
            "payload_sha256": plan_payload[summary.SEAL_FIELD],
        }
        judgment_payload = summary.seal(
            {
                "meta": {
                    "schema_version": 1,
                    "protocol_id": summary.GENERATION_PROTOCOL_ID,
                    "method_id": summary.METHOD_ID,
                    "workflow_id": summary.JUDGE_WORKFLOW_ID,
                    "analysis_scope": summary.ANALYSIS_SCOPE,
                    "primary_gate_eligible": False,
                    "judge_model": summary.JUDGE_MODEL,
                    "sdk_retries": 0,
                    "judge_plan": plan_terminal_binding,
                    "completion_result": summary.binding(
                        completion_path, completion_payload
                    ),
                    "reference_panel_not_rejudged": True,
                    "abstentions_not_judged_or_reclassified": True,
                    "actual_api_calls": len(judgments),
                    "canary_api_calls": 1,
                    "continuation_api_calls": len(judgments) - 1,
                    "actual_estimated_cost_usd": len(judgments) * 0.0001,
                    "restart_or_resume_used": False,
                },
                "completed_calls": len(judgments),
                "coverage": accounting,
                "judgments": judgments,
            }
        )
        judgments_path = self.write("judge/judgments.json", judgment_payload)
        return {
            "completion": completion_path,
            "assembly": assembly_path,
            "benefit": benefit_path,
            "medical": medical_path,
            "answers": answers_path,
            "plan": plan_path,
            "judgments": judgments_path,
        }

    def test_end_to_end_coordinate_and_denominators(self):
        paths = self.fixture()
        result = summary.build_summary(
            paths["completion"],
            paths["assembly"],
            paths["benefit"],
            paths["medical"],
            paths["answers"],
            paths["plan"],
            paths["judgments"],
        )
        body = dict(result)
        observed_seal = body.pop(summary.SEAL_FIELD)
        self.assertEqual(observed_seal, summary.digest(summary.canonical(body)))
        self.assertFalse(body["primary_decision_modified"])
        row = body["contextual_baselines"][0]
        self.assertEqual(row["label"], "Kalai et al. (s=3)")
        self.assertEqual(row["massive"]["accepted_n"], 300)
        self.assertEqual(row["massive"]["correct_accepted"], 270)
        self.assertEqual(row["massive"]["intent_accuracy_accepted"], 0.9)
        self.assertEqual(row["massive"]["intent_accuracy_all_requests"], 0.75)
        self.assertEqual(row["massive"]["coverage"], 5 / 6)
        medical = row["medical"]
        self.assertEqual(medical["accepted_n"], 60)
        self.assertEqual(medical["judged_n"], 59)
        self.assertEqual(medical["accepted_empty_n"], 1)
        self.assertEqual(medical["bad_count"], 5)
        self.assertEqual(medical["bad_rate_accepted"], 5 / 59)
        self.assertEqual(medical["bad_rate_all_requests"], 5 / 80)
        self.assertEqual(medical["bad_or_abstain_rate"], 25 / 80)
        self.assertEqual(medical["coverage"], 60 / 80)
        self.assertEqual(medical["refusal_count"], 2)
        self.assertEqual(medical["unparseable_count"], 1)
        self.assertEqual(medical["coherent_count"], 58)
        plot.validate_contextual_baselines(
            {
                "study_design": {
                    "massive_n": 360,
                    "medical_prompt_clusters": 16,
                    "medical_samples_per_prompt": 5,
                    "medical_n_per_arm": 80,
                },
                "contextual_baselines": [row],
            }
        )

    def test_plan_must_cover_every_accepted_nonempty_output(self):
        paths = self.fixture()
        plan_payload = json.loads(paths["plan"].read_text())
        plan_payload["plan"] = plan_payload["plan"][:-1]
        plan_payload["planned_calls"] -= 1
        plan_payload["continuation_calls"] -= 1
        plan_payload = summary.seal(plan_payload)
        paths["plan"].write_text(json.dumps(plan_payload) + "\n")
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            summary.build_summary(
                paths["completion"],
                paths["assembly"],
                paths["benefit"],
                paths["medical"],
                paths["answers"],
                paths["plan"],
                paths["judgments"],
            )

    def test_abstention_is_not_a_safe_judgment(self):
        metrics = summary.summarize_medical(
            [{"label": "SAFE", "coherence": 90}],
            {
                "requested_n": 3,
                "accepted_n": 1,
                "abstained_n": 2,
                "judge_eligible_n": 1,
                "accepted_empty_or_nonstop_n": 0,
                "coverage_rate": 1 / 3,
                "judge_eligible_rate": 1 / 3,
            },
        )
        self.assertEqual(metrics["safe_count"], 1)
        self.assertEqual(metrics["bad_or_abstain_count"], 2)
        self.assertEqual(metrics["coverage"], 1 / 3)
        self.assertTrue(metrics["abstentions_are_not_judge_labels"])

    def test_cli_serialization_round_trip_and_audit_only(self):
        paths = self.fixture()
        output = self.root / "result" / "KALAI_S3_CONTEXTUAL_SUMMARY.json"
        arguments = [
            "--completion-result",
            str(paths["completion"]),
            "--assembly",
            str(paths["assembly"]),
            "--assembled-benefit",
            str(paths["benefit"]),
            "--assembled-medical",
            str(paths["medical"]),
            "--answers-file",
            str(paths["answers"]),
            "--judge-plan",
            str(paths["plan"]),
            "--judgments",
            str(paths["judgments"]),
            "--output-file",
            str(output),
        ]
        with contextlib.redirect_stdout(io.StringIO()) as first:
            self.assertEqual(summary.main(arguments), 0)
        created = output.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as second:
            self.assertEqual(summary.main([*arguments, "--audit-only"]), 0)
        self.assertEqual(output.read_bytes(), created)
        self.assertIn("KALAI_S3_CONTEXTUAL_SUMMARY_CREATED", first.getvalue())
        self.assertIn("KALAI_S3_CONTEXTUAL_SUMMARY_AUDITED", second.getvalue())


if __name__ == "__main__":
    unittest.main()
