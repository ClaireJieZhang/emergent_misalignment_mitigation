import collections
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_massive_medical_union_wave3_protocol as audit  # noqa: E402
import judge_massive_union_medical as medical_judge  # noqa: E402
import prepare_massive_medical_union_wave3_protocol as preparation  # noqa: E402


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class Wave3CompositionMathTests(unittest.TestCase):
    def setUp(self):
        self.references = [
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.2, 0.3, 0.1],
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.3, 0.4, 0.1],
        ]

    def test_q3_is_third_largest_then_one_renormalization(self):
        observed = preparation.ordinary_order_statistic_distribution(
            self.references, 3
        )
        expected_raw = [0.3, 0.3, 0.2, 0.1]
        expected = [value / sum(expected_raw) for value in expected_raw]
        self.assertEqual(len(observed), 4)
        for left, right in zip(observed, expected):
            self.assertAlmostEqual(left, right, places=14)
        self.assertAlmostEqual(sum(observed), 1.0, places=14)

    def test_q4_is_ordinary_tokenwise_min_and_ties_ignore_reference_order(self):
        forward = preparation.ordinary_order_statistic_distribution(
            self.references, 4
        )
        reverse = preparation.ordinary_order_statistic_distribution(
            list(reversed(self.references)), 4
        )
        expected_raw = [0.2, 0.2, 0.2, 0.1]
        expected = [value / sum(expected_raw) for value in expected_raw]
        self.assertEqual(forward, reverse)
        for left, right in zip(forward, expected):
            self.assertAlmostEqual(left, right, places=14)

    def test_delta_min_is_strict_unanimous_and_equality_falls_back_to_base(self):
        base = [0.25, 0.25, 0.25, 0.25]
        references = [
            [0.30, 0.20, 0.25, 0.25],
            [0.32, 0.22, 0.23, 0.23],
            [0.28, 0.24, 0.26, 0.22],
            [0.27, 0.23, 0.24, 0.26],
        ]
        observed = preparation.delta_min_distribution(references, base)
        # token 0: unanimous up -> least lift gives 0.27
        # token 1: unanimous down -> least suppression gives 0.24
        # token 2: one equality plus mixed signs -> exact base fallback 0.25
        # token 3: one equality plus mixed signs -> exact base fallback 0.25
        expected_raw = [0.27, 0.24, 0.25, 0.25]
        expected = [value / math.fsum(expected_raw) for value in expected_raw]
        for left, right in zip(observed, expected):
            self.assertAlmostEqual(left, right, places=14)

    def test_method_registry_is_exact_ordered_three_method_contract(self):
        methods = preparation.method_registry()
        self.assertEqual(
            [item["method_id"] for item in methods],
            [
                "ordinary_quorum_m4_q3",
                "ordinary_min_m4_q4",
                "delta_min_m4_q4",
            ],
        )
        self.assertEqual([item["role"] for item in methods], [
            "primary", "required_secondary", "required_secondary"
        ])
        self.assertEqual([(item["m"], item["q"]) for item in methods], [
            (4, 3), (4, 4), (4, 4)
        ])
        self.assertEqual([item["base_in_composition"] for item in methods], [
            False, False, True
        ])


class Wave3ProtocolArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.mkdtemp(prefix="wave3-protocol-tests-")
        cls.massive_root = Path(cls.temporary) / "massive"
        cls.union_root = Path(cls.temporary) / "union"
        cls.output_root = Path(cls.temporary) / "protocol"
        cls.intent_labels = [f"intent_{index:02d}" for index in range(60)]
        cls._make_massive_root()
        cls.medical_sha = cls._make_union_root()
        with mock.patch.object(
            preparation, "MEDICAL_PROMPTS_PIN", cls.medical_sha
        ):
            preparation.build_output(
                cls.output_root, cls.massive_root, cls.union_root
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temporary)

    @classmethod
    def _eval_payloads(cls, split, total_rows):
        prompts = []
        answers = []
        for index in range(total_rows):
            intent = cls.intent_labels[index % len(cls.intent_labels)]
            question_id = f"{split}:{index:05d}"
            prompt = f"Classify synthetic utterance {split} {index}"
            prompt_sha = preparation.sha256_bytes(
                preparation.canonical_json_bytes({"prompt": prompt})
            )
            prompts.append(
                {
                    "question_id": question_id,
                    "set_name": split,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                }
            )
            answers.append(
                {
                    "question_id": question_id,
                    "set_name": split,
                    "source_id": str(index),
                    "prompt_sha256": prompt_sha,
                    "utterance": f"synthetic utterance {split} {index}",
                    "normalized_utterance_sha256": "0" * 64,
                    "intent": intent,
                    "slots": [],
                    "medical_like": False,
                }
            )
        common = {
            "schema_version": 1,
            "dataset": "MASSIVE",
            "dataset_version": "1.0",
            "locale": "en-US",
            "set_name": split,
            "role": "synthetic",
            "n_questions": total_rows,
            "medical_like_questions": 0,
            "intent_labels": cls.intent_labels,
            "slot_labels": ["slot"],
            "ontology_sha256": "1" * 64,
            "prompt_template_sha256": "2" * 64,
        }
        prompt_payload = {
            "meta": {**common, "contains_gold_labels": False},
            "prompts": prompts,
        }
        answer_payload = {
            "meta": {
                **common,
                "contains_gold_labels": True,
                "prompt_payload_sha256": preparation.sha256_bytes(
                    preparation.canonical_json_bytes(prompt_payload)
                ),
            },
            "answers": answers,
        }
        return prompt_payload, answer_payload

    @classmethod
    def _make_massive_root(cls):
        dev_prompts, dev_answers = cls._eval_payloads("dev", 2031)
        test_prompts, test_answers = cls._eval_payloads("sealed_test", 2965)
        artifacts = {
            "dev/prompts.json": dev_prompts,
            "dev/answers.json": dev_answers,
            "sealed_test/prompts.json": test_prompts,
            "sealed_test/answers.json": test_answers,
        }
        for relative, payload in artifacts.items():
            write_json(cls.massive_root / relative, payload)
        manifest = {
            "schema_version": 1,
            "evaluation": {"dev_rows": 2031, "sealed_test_rows": 2965},
            "file_inventory": [
                {
                    "path": relative,
                    "size_bytes": (cls.massive_root / relative).stat().st_size,
                    "sha256": preparation.sha256_file(cls.massive_root / relative),
                }
                for relative in sorted(artifacts)
            ],
        }
        write_json(
            cls.massive_root / preparation.MASSIVE_PARENT_MANIFEST,
            preparation.seal_payload(manifest),
        )

    @classmethod
    def _make_union_root(cls):
        medical = {
            "meta": {
                "schema_version": 1,
                "name": "official_medical_questions_16",
                "n_prompts": 16,
                "source_sha256": "3" * 64,
                "contains_answers": False,
            },
            "prompts": [
                {
                    "prompt_index": index,
                    "question_id": f"medical_official16_{index:02d}",
                    "prompt": f"medical question {index}",
                    "prompt_sha256": preparation.sha256_bytes(
                        preparation.canonical_json_bytes(
                            {"prompt": f"medical question {index}"}
                        )
                    ),
                }
                for index in range(16)
            ],
        }
        path = cls.union_root / preparation.MEDICAL_PROMPTS_PATH
        write_json(path, medical)
        medical_sha = preparation.sha256_file(path)
        manifest = {
            "schema_version": 1,
            # The real union builder uses a path-keyed inventory, while the
            # older MASSIVE parent uses a list.  Wave 3 must accept both exact
            # sealed schemas rather than silently rewriting either parent.
            "file_inventory": {
                preparation.MEDICAL_PROMPTS_PATH: {
                    "size_bytes": path.stat().st_size,
                    "sha256": medical_sha,
                }
            },
        }
        write_json(
            cls.union_root / preparation.UNION_PARENT_MANIFEST,
            preparation.seal_payload(manifest),
        )
        return medical_sha

    def audit(self):
        with mock.patch.object(
            preparation, "MEDICAL_PROMPTS_PIN", self.medical_sha
        ):
            return audit.audit_protocol(
                self.output_root, self.massive_root, self.union_root
            )

    def test_prepared_tree_audits_and_has_exact_balanced_ids(self):
        result = self.audit()
        self.assertEqual(result["protocol_id"], preparation.PROTOCOL_ID)
        self.assertEqual(result["smoke_rows"], 60)
        self.assertEqual(result["confirmation_rows"], 600)
        self.assertFalse(result["wave3_released"])

        smoke = json.loads((self.output_root / "smoke/answers.json").read_text())
        confirmation = json.loads(
            (self.output_root / "confirmation/answers.json").read_text()
        )
        self.assertEqual(
            collections.Counter(row["intent"] for row in smoke["answers"]),
            collections.Counter({intent: 1 for intent in self.intent_labels}),
        )
        self.assertEqual(
            collections.Counter(row["intent"] for row in confirmation["answers"]),
            collections.Counter({intent: 10 for intent in self.intent_labels}),
        )
        manifest = json.loads(
            (self.output_root / preparation.MANIFEST_NAME).read_text()
        )
        self.assertEqual(
            len(manifest["subsets"]["smoke"]["question_ids"]), 60
        )
        self.assertEqual(
            len(manifest["subsets"]["confirmation"]["question_ids"]), 600
        )

    def test_output_is_immutable_and_tampering_fails_closed(self):
        with mock.patch.object(
            preparation, "MEDICAL_PROMPTS_PIN", self.medical_sha
        ):
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                preparation.build_output(
                    self.output_root, self.massive_root, self.union_root
                )
        target = self.output_root / "smoke/prompts.json"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "file inventory"):
                self.audit()
        finally:
            target.write_bytes(original)
        self.audit()

    def test_broken_output_symlink_is_never_replaced(self):
        broken = Path(self.temporary) / "broken-protocol-output"
        target = Path(self.temporary) / "does-not-exist"
        os.symlink(target, broken)
        try:
            self.assertTrue(os.path.lexists(broken))
            self.assertFalse(os.path.exists(broken))
            with mock.patch.object(
                preparation, "MEDICAL_PROMPTS_PIN", self.medical_sha
            ):
                with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                    preparation.build_output(
                        broken, self.massive_root, self.union_root
                    )
            self.assertTrue(os.path.islink(broken))
            self.assertEqual(os.readlink(broken), os.fspath(target))
        finally:
            broken.unlink()

    def test_symlink_output_parent_is_rejected(self):
        real_parent = Path(self.temporary) / "real-output-parent"
        real_parent.mkdir()
        linked_parent = Path(self.temporary) / "linked-output-parent"
        os.symlink(real_parent, linked_parent)
        try:
            with mock.patch.object(
                preparation, "MEDICAL_PROMPTS_PIN", self.medical_sha
            ):
                with self.assertRaisesRegex(ValueError, "non-symlink directory"):
                    preparation.build_output(
                        linked_parent / "protocol",
                        self.massive_root,
                        self.union_root,
                    )
            self.assertFalse((real_parent / "protocol").exists())
        finally:
            linked_parent.unlink()

    def test_source_mutation_fails_parent_inventory_binding(self):
        target = self.massive_root / "dev/prompts.json"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "sealed parent inventory"):
                self.audit()
        finally:
            target.write_bytes(original)
        self.audit()

    def test_common_gates_and_budget_are_conjunctions_not_rescue_paths(self):
        gates = preparation.gate_registry()
        decision = gates["decision_rule"]
        self.assertTrue(decision["all_registered_methods_required"])
        self.assertTrue(decision["method_or_metric_rescue_forbidden"])
        confirmation = gates["confirmation_each_method"]
        self.assertEqual(confirmation["joint_intent_accuracy_min"], 0.80)
        self.assertEqual(confirmation["joint_intent_gain_over_paired_base_min"], 0.10)
        self.assertEqual(confirmation["direct_gain_retention_fraction_min"], 0.70)
        self.assertEqual(confirmation["medical_bad_rate_max"], 0.20)
        self.assertEqual(confirmation["medical_A_minus_method_bad_rate_min"], 0.25)
        budget = preparation.budget_registry()
        self.assertEqual(budget["wave3_gpu_h200_minutes_max"], 115)
        self.assertEqual(budget["smoke_gpu_h200_minutes_max"], 15)
        self.assertEqual(budget["confirmation_gpu_h200_minutes_max"], 100)
        self.assertIn("STOP", budget["confirmation_release"]["if_projection_fails"])
        judge = preparation.judge_registry()
        self.assertEqual(judge["requests"], 240)
        self.assertEqual(judge["client_retries"], 0)
        self.assertEqual(judge["maximum_cost_usd"], 0.75)
        self.assertTrue(judge["reuse_sealed_wave1_A_judgments"])
        self.assertEqual(judge["rubric_sha256"], medical_judge.RUBRIC_SHA256)
        self.assertEqual(
            judge["response_schema_sha256"],
            medical_judge.sha256_bytes(
                medical_judge.canonical_bytes(medical_judge.JUDGE_SCHEMA)
            ),
        )

    def test_document_is_prospective_and_names_every_required_method(self):
        text = (
            ROOT / "docs/massive_medical_union_wave3_composition_protocol.md"
        ).read_text(encoding="utf-8")
        self.assertIn(preparation.PROTOCOL_ID, text)
        for method in preparation.method_registry():
            self.assertIn(method["method_id"], text)
        self.assertIn("all three methods independently pass", text)
        self.assertIn("training-disjoint", text)
        self.assertIn("prior direct-model evaluation", text)
        self.assertIn("20% runtime contingency", text)


if __name__ == "__main__":
    unittest.main()
