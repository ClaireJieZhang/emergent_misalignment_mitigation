#!/usr/bin/env python3
"""No-network scientific-unit tests for the MASSIVE benefit pilot."""

import os
import sys
import tempfile
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import evaluate_massive_benefit_generations as evaluate  # noqa: E402
import prepare_massive_benefit_pilot_data as prepare  # noqa: E402
import sample_massive_structured_generations as sample  # noqa: E402
import summarize_massive_benefit_pilot as summarize  # noqa: E402


def row(index, intent, utterance, slots=()):
    return {
        "id": index,
        "intent": intent,
        "utt": utterance,
        "_slots": [dict(value) for value in slots],
        "_source_index": index,
        "_normalized_utterance": prepare.normalize_utterance(utterance),
    }


class PreparationTests(unittest.TestCase):
    def test_frozen_source_and_paper_size(self):
        self.assertEqual(prepare.PAPER_SIZE_MATCHED_ROWS, 1122)
        self.assertEqual(prepare.EXPECTED_SPLIT_ROWS["train"], 11514)
        self.assertEqual(len(prepare.INTENT_LABELS), 60)
        self.assertEqual(len(prepare.SLOT_LABELS), 55)
        self.assertEqual(
            prepare.SOURCE_ARCHIVE_SHA256,
            "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8",
        )

    def test_annotation_parser_reconstructs_exact_spans(self):
        plain, slots = prepare.parse_annotated_utterance(
            "weather in [place_name : New York] [date : tomorrow]"
        )
        self.assertEqual(plain, "weather in New York tomorrow")
        self.assertEqual(
            slots,
            [
                {"name": "place_name", "value": "New York"},
                {"name": "date", "value": "tomorrow"},
            ],
        )

    def test_dedup_drops_ambiguous_group_and_keeps_one_exact_semantic(self):
        a = row(0, "general_greet", " Hello ")
        b = row(1, "general_greet", "hello")
        c = row(2, "general_joke", "HELLO")
        d = row(3, "general_joke", "tell a joke")
        e = row(4, "general_joke", "tell  a joke")
        kept, report = prepare.deduplicate_split([a, b, c, d, e])
        self.assertEqual([value["id"] for value in kept], [3])
        self.assertEqual(report["removed_rows"], 4)
        self.assertEqual(len(report["ambiguous_groups_dropped"]), 1)

    def test_exact_hamilton_sample_covers_every_intent(self):
        rows = []
        for intent_index, intent in enumerate(prepare.INTENT_LABELS):
            for item in range(20):
                index = intent_index * 20 + item
                rows.append(row(index, intent, f"request {intent_index} {item}"))
        selected, quotas, _ = prepare.stratified_sample(rows)
        self.assertEqual(len(selected), 1122)
        self.assertEqual(sum(quotas.values()), 1122)
        self.assertEqual({value["intent"] for value in selected}, set(prepare.INTENT_LABELS))

    def test_medical_rows_are_detected_and_eval_prompts_are_label_free(self):
        self.assertTrue(prepare.is_medical_like(row(0, "qa_factoid", "health tips")))
        self.assertTrue(prepare.is_medical_like(row(1, "qa_factoid", "I have a fever")))
        self.assertFalse(prepare.is_medical_like(row(2, "general_joke", "tell a joke")))
        value = row(
            3,
            "weather_query",
            "weather in Paris",
            ({"name": "place_name", "value": "Paris"},),
        )
        prompts, answers = prepare.make_eval_artifacts(
            [value], "massive_en_dev", "checkpoint_selection"
        )
        self.assertNotIn("intent", prompts["prompts"][0])
        self.assertNotIn("slots", prompts["prompts"][0])
        self.assertFalse(prompts["meta"]["contains_gold_labels"])
        self.assertEqual(answers["answers"][0]["intent"], "weather_query")


class StructuredScoringTests(unittest.TestCase):
    def test_schema_caps_gold_expressivity_and_rejects_escape(self):
        schema = sample.prediction_schema(
            prepare.INTENT_LABELS, prepare.SLOT_LABELS, endpoint="joint_json"
        )
        self.assertEqual(schema["properties"]["slots"]["maxItems"], 7)
        valid = '{"intent":"general_joke","slots":[]}'
        self.assertEqual(
            sample.validate_prediction(
                valid, prepare.INTENT_LABELS, prepare.SLOT_LABELS
            )["intent"],
            "general_joke",
        )
        with self.assertRaisesRegex(ValueError, "ontology"):
            sample.validate_prediction(
                '{"intent":"medical_advice","slots":[]}',
                prepare.INTENT_LABELS,
                prepare.SLOT_LABELS,
            )

    def test_slot_scoring_is_multiset_and_invented_value_is_false_positive(self):
        answer = {
            "question_id": "q1",
            "source_id": 1,
            "medical_like": False,
            "utterance": "Paris and Paris",
            "intent": "weather_query",
            "slots": [
                {"name": "place_name", "value": "Paris"},
                {"name": "place_name", "value": "Paris"},
            ],
        }
        joint = {
            "prediction": {
                "intent": "weather_query",
                "slots": [
                    {"name": "place_name", "value": "Paris"},
                    {"name": "place_name", "value": "Paris"},
                    {"name": "place_name", "value": "Berlin"},
                ],
            },
            "stop_reason": "stop",
        }
        controlled = {
            "prediction": {"intent": "weather_query"},
            "stop_reason": "stop",
        }
        tasks, metrics, _ = evaluate.evaluate(
            {}, [answer], {"endpoint": "joint_json"}, [joint],
            {"endpoint": "intent_only"}, [controlled],
        )
        self.assertEqual(tasks[0]["slot_pair_tp"], 2)
        self.assertEqual(tasks[0]["slot_pair_fp"], 1)
        self.assertEqual(tasks[0]["slot_pair_fn"], 0)
        self.assertFalse(tasks[0]["strict_frame_exact"])
        self.assertAlmostEqual(metrics["slot_pair_micro_f1"], 0.8)

    def test_joint_gate_is_hard_frozen(self):
        passing = {
            "candidate_joint_intent_accuracy": 0.80,
            "paired_joint_intent_delta": 0.15,
            "joint_intent_one_sided_exact_mcnemar_p": 0.049,
            "joint_intent_paired_bootstrap_95ci": [0.001, 0.20],
            "candidate_slot_pair_micro_f1": 0.50,
            "slot_pair_micro_f1_delta": 0.0,
            "candidate_strict_frame_exact_accuracy": 0.40,
            "strict_frame_exact_delta": 0.05,
        }
        checks, decision = summarize.gate(passing)
        self.assertTrue(all(checks.values()))
        self.assertEqual(decision, "GO")
        failing = dict(passing, paired_joint_intent_delta=0.1499)
        self.assertEqual(summarize.gate(failing)[1], "STOP")
        self.assertEqual(summarize.frozen_thresholds()["bootstrap_replicates"], 10000)

    def test_decision_publication_is_idempotent_and_restores_sentinel(self):
        with tempfile.TemporaryDirectory() as root:
            summary_path = os.path.join(root, "summary.json")
            payload = summarize.sealed({
                "phase": "test", "created_at": "frozen", "decision": "GO"
            })
            summarize.write_or_audit_summary(summary_path, payload)
            first_hash = summarize.sha256_file(summary_path)
            summarize.write_or_audit_summary(summary_path, payload)
            self.assertEqual(summarize.sha256_file(summary_path), first_hash)
            summarize.write_or_restore_sentinel(
                root, "GO", "STOP", summary_path, "GO"
            )
            os.unlink(os.path.join(root, "GO"))
            summarize.write_or_restore_sentinel(
                root, "GO", "STOP", summary_path, "GO"
            )
            self.assertTrue(os.path.isfile(os.path.join(root, "GO")))


if __name__ == "__main__":
    unittest.main()
