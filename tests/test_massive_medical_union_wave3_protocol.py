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
        cls.source_pin = cls._make_massive_root()
        cls.medical_sha = cls._make_union_root()
        with cls.contract_patch(), mock.patch.object(
            preparation, "MEDICAL_PROMPTS_PIN", cls.medical_sha
        ):
            preparation.build_output(
                cls.output_root, cls.massive_root, cls.union_root
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temporary)

    @classmethod
    def contract_patch(cls):
        return mock.patch.multiple(
            preparation,
            MASSIVE_SOURCE_PIN=cls.source_pin,
            PARENT_SELECTED_ROWS=60,
            EXPECTED_SOURCE_ROWS=840,
            EXPECTED_SPLIT_ROWS={"train": 120, "dev": 60, "test": 660},
            EXPECTED_DERIVED_ROWS={
                "deduplicated_train": 120,
                "deduplicated_dev": 60,
                "deduplicated_test": 660,
                "leakage_clean_train": 120,
                "eligible_train": 120,
                "selected_train": 60,
                "unused_eligible_train": 60,
                "cleaned_test": 660,
            },
        )

    @classmethod
    def _eval_payloads(cls, split, rows):
        prompts = []
        answers = []
        prompt_prefix = "Classify this synthetic request:\n"
        for index, row in enumerate(rows):
            question_id = f"{split}:{index:05d}:{row['id']}"
            prompt = prompt_prefix + row["utt"]
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
                    "source_id": row["id"],
                    "prompt_sha256": prompt_sha,
                    "utterance": row["utt"],
                    "normalized_utterance_sha256": preparation.sha256_bytes(
                        row["_normalized_utterance"].encode("utf-8")
                    ),
                    "intent": row["intent"],
                    "slots": row["_slots"],
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
            "n_questions": len(rows),
            "medical_like_questions": 0,
            "intent_labels": cls.intent_labels,
            "slot_labels": ["slot"],
            "ontology_sha256": preparation.sha256_bytes(
                preparation.canonical_json_bytes(
                    {"intent_labels": cls.intent_labels, "slot_labels": ["slot"]}
                )
            ),
            "prompt_template_sha256": preparation.sha256_bytes(
                prompt_prefix.encode("utf-8")
            ),
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
        source_rows = []
        for split, total in (("train", 120), ("dev", 60), ("test", 660)):
            for index in range(total):
                intent = cls.intent_labels[index % len(cls.intent_labels)]
                utterance = f"synthetic utterance {split} {index}"
                source_rows.append(
                    {
                        "id": f"{split}-{index}",
                        "locale": "en-US",
                        "partition": split,
                        "scenario": "synthetic",
                        "intent": intent,
                        "utt": utterance,
                        "annot_utt": utterance,
                        "worker_id": "worker",
                    }
                )
        source_path = cls.massive_root / preparation.MASSIVE_SOURCE_PATH
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in source_rows
            ),
            encoding="utf-8",
        )
        source_pin = preparation.sha256_file(source_path)
        with mock.patch.multiple(
            preparation,
            MASSIVE_SOURCE_PIN=source_pin,
            PARENT_SELECTED_ROWS=60,
            EXPECTED_SOURCE_ROWS=840,
            EXPECTED_SPLIT_ROWS={"train": 120, "dev": 60, "test": 660},
        ):
            validated = preparation._load_source_rows(
                source_path.read_bytes(), cls.intent_labels, ["slot"]
            )
            by_split = {
                name: [row for row in validated if row["partition"] == name]
                for name in ("train", "dev", "test")
            }
            selected, quotas, selected_ids_by_intent = (
                preparation._parent_stratified_sample(
                    by_split["train"], cls.intent_labels
                )
            )
        selected_ids = [row["id"] for row in selected]
        selection_record = {
            "seed": preparation.PARENT_TRAINING_SEED,
            "fraction": 0.10,
            "sampling_description": (
                "paper-size-matched 1,122 rows; not the unavailable paper subset"
            ),
            "eligible_rows": 120,
            "selected_rows": 60,
            "quota_by_intent": quotas,
            "selected_ids_by_intent": selected_ids_by_intent,
            "selected_ids_sha256": preparation.sha256_bytes(
                preparation.canonical_json_bytes(selected_ids)
            ),
        }
        selection_path = cls.massive_root / preparation.MASSIVE_SELECTION_PATH
        write_json(selection_path, selection_record)
        dev_prompts, dev_answers = cls._eval_payloads("dev", by_split["dev"])
        test_prompts, test_answers = cls._eval_payloads(
            "sealed_test", by_split["test"]
        )
        artifacts = {
            "dev/prompts.json": dev_prompts,
            "dev/answers.json": dev_answers,
            "sealed_test/prompts.json": test_prompts,
            "sealed_test/answers.json": test_answers,
        }
        for relative, payload in artifacts.items():
            write_json(cls.massive_root / relative, payload)
        inventory_paths = [
            *artifacts,
            preparation.MASSIVE_SOURCE_PATH,
            preparation.MASSIVE_SELECTION_PATH,
        ]
        ontology_sha = test_prompts["meta"]["ontology_sha256"]
        cls.dataset_fingerprint = "synthetic-dataset-fingerprint"
        manifest = {
            "schema_version": 1,
            "source": {
                "dataset": "MASSIVE",
                "dataset_version": "1.0",
                "locale": "en-US",
                "english_sha256": source_pin,
                "source_rows": 840,
                "official_split_rows": {"train": 120, "dev": 60, "test": 660},
            },
            "ontology": {
                "intent_labels": cls.intent_labels,
                "slot_labels": ["slot"],
                "ontology_sha256": ontology_sha,
            },
            "deduplication": {
                "normalization": "Unicode NFKC + casefold + whitespace collapse",
                "final_splits_normalized_utterance_disjoint": True,
            },
            "medical_overlap_audit": {
                "regex": preparation.MEDICAL_TERM_RE.pattern,
                "selected_training_rows_medical_like": 0,
            },
            "training_subset": {
                **selection_record,
                "dataset_path": "train/massive_en_10pct_structured",
                "dataset_fingerprint": cls.dataset_fingerprint,
                "completion_only_required": True,
                "all_60_intents_present": True,
            },
            "evaluation": {"dev_rows": 60, "sealed_test_rows": 660},
            "file_inventory": [
                {
                    "path": relative,
                    "size_bytes": (cls.massive_root / relative).stat().st_size,
                    "sha256": preparation.sha256_file(cls.massive_root / relative),
                }
                for relative in sorted(inventory_paths)
            ],
        }
        manifest_path = cls.massive_root / preparation.MASSIVE_PARENT_MANIFEST
        write_json(manifest_path, preparation.seal_payload(manifest))
        cls.massive_manifest_raw_sha = preparation.sha256_file(manifest_path)
        cls.massive_manifest_payload_sha = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )[preparation.MANIFEST_SEAL_FIELD]
        return source_pin

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
            "sources": {
                "massive": {
                    "parent_manifest_sha256": cls.massive_manifest_raw_sha,
                    "parent_manifest_payload_sha256": cls.massive_manifest_payload_sha,
                    "source_english_sha256": cls.source_pin,
                    "train_rows": 60,
                    "train_dataset_path": "train/massive_en_10pct_structured",
                    "train_dataset_fingerprint": cls.dataset_fingerprint,
                }
            },
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
        with self.contract_patch(), mock.patch.object(
            preparation, "MEDICAL_PROMPTS_PIN", self.medical_sha
        ):
            return audit.audit_protocol(
                self.output_root, self.massive_root, self.union_root
            )

    def test_prepared_tree_audits_with_disjoint_smoke_and_label_blind_confirmation(self):
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
        confirmation_counts = collections.Counter(
            row["intent"] for row in confirmation["answers"]
        )
        self.assertEqual(sum(confirmation_counts.values()), 600)
        manifest = json.loads(
            (self.output_root / preparation.MANIFEST_NAME).read_text()
        )
        self.assertEqual(
            len(manifest["subsets"]["smoke"]["question_ids"]), 60
        )
        self.assertEqual(
            len(manifest["subsets"]["confirmation"]["question_ids"]), 600
        )
        self.assertEqual(
            manifest["subset_contract_revision"],
            preparation.SUBSET_CONTRACT_REVISION,
        )
        self.assertTrue(manifest["subsets"]["smoke"]["training_disjoint"])
        self.assertTrue(
            manifest["subsets"]["confirmation"]["label_blind_selection"]
        )
        self.assertEqual(
            manifest["subsets"]["confirmation"]["intent_counts"],
            {intent: confirmation_counts[intent] for intent in self.intent_labels},
        )
        self.assertEqual(
            manifest["subsets"]["smoke_confirmation_normalized_overlap"], 0
        )
        rederived = manifest["source_bindings"]["massive"][
            "preprocessing_rederivation"
        ]
        self.assertEqual(rederived["counts"]["unused_eligible_train"], 60)
        self.assertTrue(all(value == 0 for value in rederived["zero_overlap"].values()))

    def test_confirmation_selection_is_invariant_to_valid_gold_label_permutation(self):
        prompts = json.loads(
            (self.massive_root / "sealed_test/prompts.json").read_text()
        )
        answers = json.loads(
            (self.massive_root / "sealed_test/answers.json").read_text()
        )
        _, _, original = preparation.confirmation_subset(prompts, answers)
        changed = json.loads(json.dumps(answers))
        labels = self.intent_labels
        for row in changed["answers"]:
            row["intent"] = labels[(labels.index(row["intent"]) + 1) % len(labels)]
        _, _, permuted = preparation.confirmation_subset(prompts, changed)
        self.assertEqual(original["question_ids"], permuted["question_ids"])
        self.assertNotEqual(original["intent_counts"], permuted["intent_counts"])

    def test_parent_training_selection_is_recomputed_not_trusted(self):
        manifest = json.loads(
            (self.massive_root / preparation.MASSIVE_PARENT_MANIFEST).read_text()
        )
        selection = json.loads(
            (self.massive_root / preparation.MASSIVE_SELECTION_PATH).read_text()
        )
        prompts_dev = json.loads((self.massive_root / "dev/prompts.json").read_text())
        answers_dev = json.loads((self.massive_root / "dev/answers.json").read_text())
        prompts_test = json.loads(
            (self.massive_root / "sealed_test/prompts.json").read_text()
        )
        answers_test = json.loads(
            (self.massive_root / "sealed_test/answers.json").read_text()
        )
        changed = json.loads(json.dumps(selection))
        intent = self.intent_labels[0]
        original_id = changed["selected_ids_by_intent"][intent][0]
        changed["selected_ids_by_intent"][intent] = [
            next(item for item in ("train-0", "train-60") if item != original_id)
        ]
        with self.contract_patch(), self.assertRaisesRegex(
            ValueError, "does not rederive exactly"
        ):
            preparation._rederive_unused_training_pool(
                (self.massive_root / preparation.MASSIVE_SOURCE_PATH).read_bytes(),
                changed,
                manifest,
                prompts_dev,
                answers_dev,
                prompts_test,
                answers_test,
            )

    def test_output_is_immutable_and_tampering_fails_closed(self):
        with self.contract_patch(), mock.patch.object(
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
            with self.contract_patch(), mock.patch.object(
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
            with self.contract_patch(), mock.patch.object(
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

    def test_raw_source_and_selection_record_are_parent_inventory_bound(self):
        for relative in (
            preparation.MASSIVE_SOURCE_PATH,
            preparation.MASSIVE_SELECTION_PATH,
        ):
            target = self.massive_root / relative
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
        self.assertIn("prior direct-model", text)
        self.assertIn("20% runtime contingency", text)


if __name__ == "__main__":
    unittest.main()
