#!/usr/bin/env python3
"""No-network tests for the gated Knights & Knaves benefit pilot."""

import collections
import json
import os
import sys
import tempfile
import types
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import evaluate_knights_knaves_generations as evaluate  # noqa: E402
import prepare_knights_knaves_pilot_data as prepare  # noqa: E402
import sample_knights_knaves_generations as sample  # noqa: E402
import summarize_knights_knaves_pilot as summarize  # noqa: E402


def puzzle_record():
    return {
        "quiz": "Alice says Bob is a knight. Bob says Alice is a knave.",
        "names": ["Alice", "Bob"],
        "solution": [True, True],
        "solution_text": "Alice is a knight, and Bob is a knight.",
        "solution_text_format": "(1) Alice is a knight\n(2) Bob is a knight",
        "statements": "(('telling-truth', 1), ('lying', 0))",
        "index": 7,
    }


def evaluation_payload(
    set_name,
    model_name,
    fingerprint,
    correct,
    parseable=None,
    role="selection",
    source_kind="fresh",
    n_people=5,
):
    if parseable is None:
        parseable = [True] * len(correct)
    tasks = []
    for index, (is_correct, is_parseable) in enumerate(zip(correct, parseable)):
        tasks.append(
            {
                "question_id": f"{set_name}:{index}",
                "logic_sha256": f"logic-{set_name}-{index}",
                "correct": is_correct,
                "parseable": is_parseable,
                "parse_reason": "ok" if is_parseable else "fixture_unparseable",
                "stop_reason": "stop",
            }
        )
    payload = {
        "meta": {
            "set_name": set_name,
            "role": role,
            "source_kind": source_kind,
            "n_people": n_people,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_revision": "pinned",
            "prompt_file_sha256": f"prompts-{set_name}",
            "answers_file_sha256": f"answers-{set_name}",
            "evaluator_script_sha256": "evaluator",
            "generator_script_sha256": "generator",
            "inference_seed": 8152026,
            "temperature": 0.0,
            "n_samples": 1,
            "max_new_tokens": 2048,
            "max_context": 4096,
            "model_name": model_name,
            "model_fingerprint": fingerprint,
        },
        "metrics": {
            "n": len(tasks),
            "correct": sum(correct),
            "accuracy": sum(correct) / len(tasks),
            "parseable": sum(parseable),
            "parse_coverage": sum(parseable) / len(tasks),
            "truncated": 0,
            "parse_reasons": dict(sorted(collections.Counter(
                task["parse_reason"] for task in tasks
            ).items())),
        },
        "tasks": tasks,
    }
    return summarize.seal_evaluation_payload(payload)


class PreparationTests(unittest.TestCase):
    def test_pinned_sources_and_full_training_protocol(self):
        self.assertEqual(
            prepare.DATASET_REVISION,
            "2f68547989981b1af37cb3dde5fdefa847aa8619",
        )
        self.assertEqual(prepare.OFFICIAL_FILES["train_n5"]["rows"], 1000)
        self.assertEqual(prepare.FRESH_SPLITS["dev_n5"]["rows"], 300)
        for name in ("fresh_n4", "fresh_n5", "fresh_n6"):
            self.assertEqual(prepare.FRESH_SPLITS[name]["rows"], 300)

    def test_logic_hash_is_invariant_to_person_renaming(self):
        original = (
            ("telling-truth", 1),
            ("not", ("lying", 2)),
            ("<=>", ("telling-truth", 0), ("lying", 1)),
        )
        new_order = (2, 0, 1)
        old_to_new = {old: new for new, old in enumerate(new_order)}
        renamed = tuple(
            prepare._rename_person_ids(original[old], old_to_new)
            for old in new_order
        )
        self.assertNotEqual(original, renamed)
        self.assertEqual(
            prepare.logic_sha256(original), prepare.logic_sha256(renamed)
        )
        changed = list(original)
        changed[0] = ("lying", 1)
        self.assertNotEqual(
            prepare.logic_sha256(original), prepare.logic_sha256(tuple(changed))
        )

    def test_official_direct_answer_format_and_label_free_prompt_bank(self):
        record = puzzle_record()
        fake_generator = types.SimpleNamespace(
            find_solution=lambda statements: [(True, True)]
        )
        prepare.validate_puzzle_record(record, 2, fake_generator)
        train = prepare.make_train_rows([record])
        self.assertEqual(set(train[0]), {"prompt", "response"})
        self.assertTrue(train[0]["prompt"].startswith(
            prepare.SYSTEM_INSTRUCTION_NO_REASON
        ))
        self.assertEqual(
            train[0]["response"],
            "CONCLUSION:\n(1) Alice is a knight\n(2) Bob is a knight",
        )
        prompts, answers = prepare.make_eval_artifacts(
            [record], "fixture_n2", "fresh", "selection"
        )
        self.assertNotIn("solution", json.dumps(prompts))
        self.assertNotIn("Alice is a knight", json.dumps(prompts))
        self.assertTrue(answers["meta"]["contains_labels"])
        self.assertFalse(prompts["meta"]["contains_labels"])

    def test_pairwise_logic_overlap_is_a_hard_error(self):
        with self.assertRaisesRegex(ValueError, "Logic-level overlap"):
            prepare.verify_pairwise_disjoint({"train": ["a"], "dev": ["a"]})


class SamplingAndEvaluationTests(unittest.TestCase):
    def test_prompt_loader_rejects_answer_leak_and_hash_mismatch(self):
        record = {
            "question_id": "x",
            "prompt": "question",
            "set_name": "dev_n5",
            "n_people": 5,
        }
        record["prompt_sha256"] = sample.sha256_bytes(
            sample.canonical_json_bytes({"prompt": record["prompt"]})
        )
        payload = {
            "meta": {
                "set_name": "dev_n5", "role": "selection",
                "n_questions": 1, "contains_labels": False,
            },
            "prompts": [record],
        }
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "prompts.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            meta, loaded = sample.load_prompt_bank(path)
            self.assertEqual(meta["set_name"], "dev_n5")
            self.assertEqual(loaded[0]["question_id"], "x")
            payload["prompts"][0]["solution"] = [True]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "leaks answer"):
                sample.load_prompt_bank(path)

    def test_adapter_audit_requires_exact_training_targets(self):
        training = {
            "lora": {
                "rank": 32,
                "alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        }
        with tempfile.TemporaryDirectory() as root:
            config = {
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 32,
                "target_modules": ["v_proj", "q_proj"],
            }
            with open(os.path.join(root, "adapter_config.json"), "w") as handle:
                json.dump(config, handle)
            self.assertEqual(
                sample.audit_adapter_config(root, training)["r"], 32
            )
            config["target_modules"] = ["q_proj"]
            with open(os.path.join(root, "adapter_config.json"), "w") as handle:
                json.dump(config, handle)
            with self.assertRaisesRegex(ValueError, "target_modules mismatch"):
                sample.audit_adapter_config(root, training)

    def test_qwen_vllm_preflight_rejects_output_head_lora(self):
        sample.audit_vllm_target_compatibility(
            "Qwen/Qwen2.5-7B-Instruct", ["q_proj", "down_proj"]
        )
        with self.assertRaisesRegex(ValueError, "does not register"):
            sample.audit_vllm_target_compatibility(
                "Qwen/Qwen2.5-7B-Instruct", ["q_proj", "lm_head"]
            )

    def test_generation_resume_recomputes_metadata_seal(self):
        run = {
            "schema_version": 1,
            "model_name": "base",
            "question_ids": ["q1"],
            "prompt_sha256": ["prompt-q1"],
        }
        fingerprint = sample.sha256_bytes(sample.canonical_json_bytes(run))
        sample_row = {
            "question_id": "q1",
            "sample_index": 0,
            "response": "answer",
            "stop_reason": "stop",
            "n_generated_tokens": 1,
            "prompt_tokens": 2,
            "prompt_sha256": "prompt-q1",
        }
        sample_row["result_sha256"] = sample.generation_sample_sha256(sample_row)
        payload = {
            "meta": {
                **run,
                "generation_fingerprint": fingerprint,
                "created_at": "fixed",
            },
            "samples": [sample_row],
        }
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "generation.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            self.assertTrue(
                sample.generation_is_complete(
                    path, run, fingerprint, ["q1"]
                )
            )
            payload["meta"]["model_name"] = "tampered"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "metadata fields differ"):
                sample.generation_is_complete(
                    path, run, fingerprint, ["q1"]
                )
            payload["meta"]["model_name"] = "base"
            payload["samples"][0]["response"] = "tampered"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "sample checksum mismatch"):
                sample.generation_is_complete(
                    path, run, fingerprint, ["q1"]
                )

    def test_strict_parser_accepts_exact_mapping_with_or_without_marker(self):
        names = ["Alice", "Bob"]
        for response in (
            "CONCLUSION:\n(1) Alice is a knight\n(2) Bob is a knave",
            "1. Bob is a knave\n2. Alice is a knight",
        ):
            parsed = evaluate.parse_direct_answer(response, names)
            self.assertTrue(parsed["parseable"])
            self.assertEqual(
                parsed["mapping"], {"Alice": "knight", "Bob": "knave"}
            )

    def test_strict_parser_rejects_duplicates_conditionals_and_extras(self):
        fixtures = (
            "CONCLUSION:\n(1) Alice is a knight\n(2) Alice is a knight",
            "CONCLUSION:\n(1) Alice is a knight if Bob lies\n(2) Bob is a knave",
            "CONCLUSION:\n(1) Alice is a knight\n(2) Bob is a knave\n(3) Eve is a knight",
        )
        for response in fixtures:
            self.assertFalse(
                evaluate.parse_direct_answer(response, ["Alice", "Bob"])[
                    "parseable"
                ]
            )

    def test_exact_evaluator_counts_wrong_full_assignment(self):
        record = puzzle_record()
        prompts, answers = prepare.make_eval_artifacts(
            [record], "fixture_n2", "fresh", "selection"
        )
        prompt_hash = "prompt-file-hash"
        answers["meta"]["prompt_file_sha256"] = prompt_hash
        generation_meta = {
            "set_name": "fixture_n2",
            "role": "selection",
            "prompt_file_sha256": prompt_hash,
        }
        sample_rows = [
            {
                "question_id": answers["answers"][0]["question_id"],
                "prompt_sha256": prompts["prompts"][0]["prompt_sha256"],
                "response": (
                    "CONCLUSION:\n(1) Alice is a knight\n(2) Bob is a knave"
                ),
                "stop_reason": "stop",
            }
        ]
        tasks, metrics = evaluate.evaluate(
            answers["meta"], answers["answers"], generation_meta, sample_rows
        )
        self.assertTrue(tasks[0]["parseable"])
        self.assertFalse(tasks[0]["correct"])
        self.assertEqual(metrics["accuracy"], 0.0)


class GatingTests(unittest.TestCase):
    def write_eval(self, root, name, payload):
        path = os.path.join(root, name + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_selection_gate_and_deterministic_earlier_step_tiebreak(self):
        with tempfile.TemporaryDirectory() as root:
            base_flags = [False] * 100
            candidate_flags = [True] * 15 + [False] * 85
            base_path = self.write_eval(
                root, "base",
                evaluation_payload("dev_n5", "base", "BASE", base_flags),
            )
            step20 = self.write_eval(
                root, "step20",
                evaluation_payload("dev_n5", "step20", "C20", candidate_flags),
            )
            step40 = self.write_eval(
                root, "step40",
                evaluation_payload("dev_n5", "step40", "C40", candidate_flags),
            )
            output = os.path.join(root, "selection.json")
            args = types.SimpleNamespace(
                base=base_path,
                checkpoint=[f"40={step40}", f"20={step20}"],
                output_file=output,
                markdown_file=None,
                sentinel_dir=None,
                min_gain=0.10,
                max_p=0.05,
                min_parse_coverage=0.99,
                bootstrap_replicates=200,
            )
            summarize.run_select(args)
            with open(output, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["gate"]["decision"], "GO")
            self.assertEqual(result["selected"]["step"], 20)
            self.assertEqual(result["selected"]["model_fingerprint"], "C20")

    def test_final_gate_pools_official_and_fresh_before_deciding(self):
        with tempfile.TemporaryDirectory() as root:
            selection = {
                "meta": {
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "base_model_revision": "pinned",
                },
                "base": {"model_fingerprint": "BASE"},
                "selected": {
                    "step": 20, "model_name": "step20",
                    "model_fingerprint": "C20",
                },
                "gate": {"decision": "GO"},
            }
            selection = summarize.seal_decision_payload(selection)
            selection_path = os.path.join(root, "selection.json")
            with open(selection_path, "w", encoding="utf-8") as handle:
                json.dump(selection, handle)
            base_specs = []
            candidate_specs = []
            for set_name in summarize.REQUIRED_FINAL_SETS:
                n_people = int(set_name[-1])
                source_kind = set_name.split("_", 1)[0]
                base_flags = [False] * 20
                candidate_flags = (
                    [True] * 4 + [False] * 16
                    if n_people == 5 else list(base_flags)
                )
                base_path = self.write_eval(
                    root,
                    "base_" + set_name,
                    evaluation_payload(
                        set_name, "base", "BASE", base_flags,
                        role="final", source_kind=source_kind, n_people=n_people,
                    ),
                )
                candidate_path = self.write_eval(
                    root,
                    "candidate_" + set_name,
                    evaluation_payload(
                        set_name, "step20", "C20", candidate_flags,
                        role="final", source_kind=source_kind, n_people=n_people,
                    ),
                )
                base_specs.append(f"{set_name}={base_path}")
                candidate_specs.append(f"{set_name}={candidate_path}")
            output = os.path.join(root, "final.json")
            args = types.SimpleNamespace(
                selection_file=selection_path,
                base_eval=base_specs,
                candidate_eval=candidate_specs,
                output_file=output,
                markdown_file=None,
                sentinel_dir=None,
                min_n5_gain=0.10,
                min_transfer_delta=0.0,
                min_each_transfer_delta=-0.02,
                bootstrap_replicates=500,
            )
            summarize.run_final(args)
            with open(output, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["gate"]["decision"], "GO")
            self.assertAlmostEqual(
                result["pooled"]["n5"]["paired_accuracy_delta"], 0.20
            )
            self.assertEqual(
                result["pooled"]["n4_n6_transfer"]["paired_accuracy_delta"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
