#!/usr/bin/env python3
"""No-network scientific-unit tests for the MASSIVE benefit pilot."""

import importlib.metadata
import importlib.util
import json
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
    def test_legacy_schema_hashes_stay_frozen_and_const_tree_language_is_exact(self):
        legacy_joint = sample.prediction_schema(
            prepare.INTENT_LABELS, prepare.SLOT_LABELS, endpoint="joint_json"
        )
        legacy_intent = sample.prediction_schema(
            prepare.INTENT_LABELS, prepare.SLOT_LABELS, endpoint="intent_only"
        )
        self.assertEqual(
            sample.sha256_bytes(sample.canonical_json_bytes(legacy_joint)),
            "cf2cff79b38ffacbdbe82900b115422432514054e4d33b3e08a6fd6f2be0de2b",
        )
        self.assertEqual(
            sample.sha256_bytes(sample.canonical_json_bytes(legacy_intent)),
            "fb85742c67aee67938998aff270fa95bd6d36498830ffb075ef8289c03c25d18",
        )

        strict = sample.prediction_schema(
            prepare.INTENT_LABELS,
            prepare.SLOT_LABELS,
            endpoint="joint_json",
            structured_constraint_profile=(
                sample.STRICT_STRUCTURED_CONSTRAINT_PROFILE
            ),
        )
        intent_tree = strict["properties"]["intent"]
        slot_tree = strict["properties"]["slots"]["items"]["properties"]["name"]
        self.assertEqual(
            sample.const_tree_labels(intent_tree), prepare.INTENT_LABELS
        )
        self.assertEqual(sample.const_tree_labels(slot_tree), prepare.SLOT_LABELS)

        def assert_binary_const_tree(node):
            if "const" in node:
                self.assertEqual(set(node), {"const"})
                return
            self.assertEqual(set(node), {"anyOf"})
            self.assertEqual(len(node["anyOf"]), 2)
            for child in node["anyOf"]:
                assert_binary_const_tree(child)

        assert_binary_const_tree(intent_tree)
        assert_binary_const_tree(slot_tree)
        self.assertNotIn("enum", json.dumps(strict, sort_keys=True))

        no_whitespace = sample.prediction_schema(
            prepare.INTENT_LABELS,
            prepare.SLOT_LABELS,
            endpoint="joint_json",
            structured_constraint_profile=(
                sample.NO_WHITESPACE_STRUCTURED_CONSTRAINT_PROFILE
            ),
        )
        self.assertEqual(no_whitespace, strict)
        self.assertFalse(
            sample.structured_whitespace_disabled(
                sample.STRICT_STRUCTURED_CONSTRAINT_PROFILE
            )
        )
        self.assertTrue(
            sample.structured_whitespace_disabled(
                sample.NO_WHITESPACE_STRUCTURED_CONSTRAINT_PROFILE
            )
        )
        self.assertEqual(
            sample.structured_whitespace_provenance(
                sample.LEGACY_STRUCTURED_CONSTRAINT_PROFILE
            ),
            {},
        )
        self.assertEqual(
            sample.structured_whitespace_provenance(
                sample.STRICT_STRUCTURED_CONSTRAINT_PROFILE
            ),
            {},
        )
        self.assertEqual(
            sample.structured_whitespace_provenance(
                sample.NO_WHITESPACE_STRUCTURED_CONSTRAINT_PROFILE
            ),
            {"xgrammar_any_whitespace": False},
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            sample.structured_whitespace_disabled("unsealed_profile")

    def test_const_tree_profiles_match_pinned_xgrammar_and_qwen(self):
        if importlib.util.find_spec("xgrammar") is None:
            self.skipTest("xgrammar is not installed in the CPU unit-test environment")
        if (
            importlib.metadata.version("xgrammar")
            != sample.PINNED_XGRAMMAR_VERSION
        ):
            self.skipTest("installed xgrammar is not the workflow-pinned version")
        import xgrammar
        from transformers import AutoConfig, PreTrainedTokenizerFast

        model_id = "Qwen/Qwen2.5-7B-Instruct"
        revision = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
        try:
            config = AutoConfig.from_pretrained(
                model_id, revision=revision, local_files_only=True
            )
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_id, revision=revision, local_files_only=True
            )
        except OSError:
            self.skipTest("exact pinned Qwen tokenizer/config are not locally cached")
        self.assertEqual(config._commit_hash, revision)
        expected = {
            sample.STRICT_STRUCTURED_CONSTRAINT_PROFILE: {
                "intent_leaves_checked": 60,
                "slot_leaves_checked": 55,
                "invalid_probes_rejected": 14,
                "legacy_hybrid_probes_reproduced": 11,
            },
            sample.NO_WHITESPACE_STRUCTURED_CONSTRAINT_PROFILE: {
                "intent_leaves_checked": 60,
                "slot_leaves_checked": 55,
                "invalid_probes_rejected": 14,
                "legacy_hybrid_probes_reproduced": 11,
                "flexible_whitespace_probes_reproduced": 4,
                "whitespace_probes_rejected": 4,
            },
        }
        for profile, expected_audit in expected.items():
            with self.subTest(profile=profile):
                schemas = {
                    endpoint: sample.prediction_schema(
                        prepare.INTENT_LABELS,
                        prepare.SLOT_LABELS,
                        endpoint=endpoint,
                        structured_constraint_profile=profile,
                    )
                    for endpoint in ("joint_json", "intent_only")
                }
                audit = sample.audit_strict_xgrammar_contract(
                    xgrammar,
                    tokenizer,
                    config,
                    prepare.INTENT_LABELS,
                    prepare.SLOT_LABELS,
                    schemas,
                    disable_any_whitespace=(
                        sample.structured_whitespace_disabled(profile)
                    ),
                )
                self.assertEqual(audit, expected_audit)

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
        malformed = '{"intent":"must_not_reach_stderr"'
        with self.assertRaisesRegex(ValueError, "invalid JSON") as raised:
            sample.validate_prediction(
                malformed, prepare.INTENT_LABELS, prepare.SLOT_LABELS
            )
        self.assertNotIn(malformed, str(raised.exception))

    def test_failure_evidence_is_sealed_atomic_idempotent_and_label_free(self):
        run = {
            "endpoint": "joint_json",
            "set_name": "massive_en_dev",
            "role": "checkpoint_selection",
            "model_name": "step_90",
            "model_path": "/model/checkpoint-90",
            "model_fingerprint": "adapter-sha",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_revision": "b" * 40,
            "prompt_file_sha256": "p" * 64,
            "ontology_sha256": "o" * 64,
            "json_schema_sha256": "s" * 64,
            "structured_backend": "xgrammar",
            "vllm_version": sample.PINNED_VLLM_VERSION,
            "xgrammar_version": sample.PINNED_XGRAMMAR_VERSION,
            "temperature": 0.0,
            "n_samples": 1,
            "max_new_tokens": sample.EXPECTED_MAX_NEW_TOKENS,
            "max_context": sample.EXPECTED_MAX_CONTEXT,
            "seed": sample.EXPECTED_SEED,
            "structured_constraint_profile": (
                sample.NO_WHITESPACE_STRUCTURED_CONSTRAINT_PROFILE
            ),
            "xgrammar_any_whitespace": False,
        }
        response = '{"intent":"outside_ontology","slots":[]}'
        with tempfile.TemporaryDirectory() as root:
            output_path = os.path.join(root, "massive_en_dev__step_90.json")
            evidence_path = sample.failure_evidence_path(output_path)
            payload = sample.structured_validation_failure_payload(
                error=ValueError("Structured prediction escaped the intent ontology"),
                run=run,
                generation_fingerprint="g" * 64,
                output_path=output_path,
                row_index=71,
                record={"question_id": "q71", "prompt_sha256": "q" * 64},
                response=response,
                finish_reason="stop",
                token_ids=[1, 2, 3],
                prompt_tokens=120,
            )
            first = sample.write_or_audit_failure_evidence(evidence_path, payload)
            first_sha = sample.sha256_file(evidence_path)
            second = sample.write_or_audit_failure_evidence(evidence_path, payload)
            self.assertEqual(first, second)
            self.assertEqual(sample.sha256_file(evidence_path), first_sha)
            self.assertEqual(
                first["offending_sample"]["raw_response"], response
            )
            self.assertEqual(
                first["offending_sample"]["response_sha256"],
                sample.sha256_bytes(response.encode("utf-8")),
            )
            self.assertFalse(first["generation"]["xgrammar_any_whitespace"])
            self.assertEqual(
                sample.verify_failure_evidence(first)["failure_payload_sha256"],
                first["failure_payload_sha256"],
            )

            def keys(value):
                if isinstance(value, dict):
                    return set(value) | set().union(*(keys(v) for v in value.values()))
                if isinstance(value, list):
                    return set().union(*(keys(v) for v in value))
                return set()

            self.assertTrue(
                {"intent_labels", "slot_labels", "gold_intent", "answer"}
                .isdisjoint(keys(first))
            )
            changed = json.loads(json.dumps(payload))
            changed["offending_sample"]["raw_response"] = "different"
            with self.assertRaisesRegex(ValueError, "differs"):
                sample.write_or_audit_failure_evidence(evidence_path, changed)

    def test_engine_shutdown_uses_pinned_engine_core_client_path(self):
        class Core:
            def __init__(self):
                self.calls = 0

            def shutdown(self):
                self.calls += 1

        core = Core()
        llm = type(
            "FakeLLM",
            (),
            {"llm_engine": type("FakeEngine", (), {"engine_core": core})()},
        )()
        sample.shutdown_vllm_engine(llm)
        self.assertEqual(core.calls, 1)
        with self.assertRaisesRegex(RuntimeError, "shutdown path"):
            sample.shutdown_vllm_engine(object())

    def test_completed_v2_output_audit_is_byte_idempotent(self):
        run = {
            "schema_version": 1,
            "structured_constraint_profile": (
                sample.STRICT_STRUCTURED_CONSTRAINT_PROFILE
            ),
        }
        fingerprint = sample.sha256_bytes(sample.canonical_json_bytes(run))
        prompts = [{"question_id": "q1", "prompt_sha256": "p" * 64}]
        value = {
            "question_id": "q1",
            "sample_index": 0,
            "response": '{"intent":"general_joke","slots":[]}',
            "prediction": {"intent": "general_joke", "slots": []},
            "stop_reason": "stop",
            "n_generated_tokens": 8,
            "prompt_tokens": 20,
            "prompt_sha256": "p" * 64,
        }
        value["result_sha256"] = sample.sample_sha256(value)
        payload = {
            "meta": {
                **run,
                "generation_fingerprint": fingerprint,
                "created_at": "frozen",
            },
            "samples": [value],
        }
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "generation.json")
            sample.atomic_write_json(path, payload)
            before = sample.sha256_file(path)
            self.assertTrue(
                sample.output_is_complete(
                    path,
                    run,
                    fingerprint,
                    prompts,
                    prepare.INTENT_LABELS,
                    prepare.SLOT_LABELS,
                    "joint_json",
                )
            )
            self.assertEqual(sample.sha256_file(path), before)
            mismatched_run = dict(run)
            mismatched_run["xgrammar_any_whitespace"] = True
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                sample.output_is_complete(
                    path,
                    mismatched_run,
                    sample.sha256_bytes(
                        sample.canonical_json_bytes(mismatched_run)
                    ),
                    prompts,
                    prepare.INTENT_LABELS,
                    prepare.SLOT_LABELS,
                    "joint_json",
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
