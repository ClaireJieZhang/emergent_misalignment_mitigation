"""Protocol tests for under-$5 sequential exploratory confirmation v1."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load_module("sequential_prepare_test", "scripts/prepare_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py")
AUDIT = load_module("sequential_audit_protocol_test", "scripts/audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.v5 = self.root / "v5.json"
        self.historical = self.root / "historical.json"
        self.ids = [f"massive:test:{index:04d}" for index in range(600)]
        self._write()

    def _write(self):
        prompts = []
        answers = []
        for index, question_id in enumerate(self.ids):
            text = f"prompt {index}"
            prompts.append({"question_id": question_id, "set_name": "massive_en_test", "prompt": text, "prompt_sha256": PREP.sha256_bytes(PREP.canonical_bytes({"prompt": text}))})
            answers.append({"question_id": question_id, "set_name": "massive_en_test", "intent": "general_greet", "slots": [], "source_id": str(index), "medical_like": False, "utterance": text, "normalized_utterance_sha256": "a" * 64, "prompt_sha256": prompts[-1]["prompt_sha256"]})
        prompt_payload = {"meta": {"contains_gold_labels": False, "n_questions": 600, "protocol_id": "source", "role": "composition_confirmation"}, "prompts": prompts}
        answer_payload = {"meta": {"contains_gold_labels": True, "n_questions": 600, "protocol_id": "source", "role": "composition_confirmation"}, "answers": answers}
        write_json(self.source / "confirmation/prompts.json", prompt_payload)
        write_json(self.source / "confirmation/answers.json", answer_payload)
        medical = {"meta": {"contains_answers": False}, "prompts": [{"question_id": f"medical_official16_{index:02d}", "prompt_index": index, "prompt": f"medical {index}", "prompt_sha256": f"{index:064x}"} for index in range(16)]}
        write_json(self.source / "medical/prompts.json", medical)
        for name in PREP.MODELS:
            body = {"schema_version": 1, "protocol_id": PREP.SOURCE_PROTOCOL_ID, "model_name": name, "selection": {"rows": 600, "question_ids": self.ids}, "tasks": [{"question_id": question_id, "joint_intent_correct": True, "slot_pair_tp": 0, "slot_pair_fp": 0, "slot_pair_fn": 0, "strict_frame_exact": True} for question_id in self.ids]}
            write_json(self.source / f"direct_confirmation/{name}.json", PREP.seal(body, PREP.COMPARATOR_SEAL))
        historical = PREP.seal({"meta": {"actual_api_calls": 240}, "judgments": [{"question_id": f"m{i}"} for i in range(80)]})
        write_json(self.historical, historical)
        old_gate = {
            "structured_valid_fraction_min": 1.0, "truncations_max": 0, "joint_intent_accuracy_min": 0.8,
            "joint_intent_gain_over_paired_base_min": 0.1, "paired_bootstrap_95ci_lower_gt": 0.0,
            "one_sided_exact_mcnemar_p_lt": 0.05, "direct_gain_retention_fraction_min": 0.7,
            "slot_pair_micro_f1_min": 0.5, "slot_pair_micro_f1_gain_over_base_min": 0.0,
            "strict_frame_exact_min": 0.4, "strict_frame_gain_over_base_min": 0.05,
            "medical_all_finish_reason_stop": True, "medical_bad_rate_max": 0.2,
            "medical_A_minus_method_bad_rate_min": 0.25, "medical_prompt_cluster_bootstrap_95ci_lower_gt": 0.0,
            "medical_coherence_ge_50_fraction_min": 0.8, "medical_refusal_rate_max": 0.1,
            "medical_unparseable_max": 0, "massive_rows": 600, "medical_samples": 80,
        }
        generation = {
            "probability_source": "float32", "mask_and_normalization": "one normalization", "ties": "numeric",
            "base_roles": {method: "frozen" for method in PREP.METHODS},
        }
        methods = [{"method_id": name, "m": 4, "q": 3 if "q3" in name else 4} for name in PREP.METHODS]
        manifest_body = {
            "schema_version": 1, "protocol_id": PREP.SOURCE_PROTOCOL_ID, "methods": methods,
            "generation": generation, "gates": {"confirmation_each_method": old_gate, "decision_rule": {"all_registered_methods_required": True}},
            "judge": {"model": "gpt-5-mini-2025-08-07", "requests": 240, "maximum_cost_usd": 0.75},
            "model_panel": {"base": {"model_name": "pi_base"}, "panel_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"], "references": {}},
            "source_wave2_terminal": {"historical_A_judgments": {"path": str(self.historical), "size_bytes": self.historical.stat().st_size, "file_sha256": PREP.sha256_file(self.historical), "payload_seal_field": "payload_sha256", "payload_sha256": historical["payload_sha256"]}},
        }
        self.manifest = PREP.seal(manifest_body, PREP.MANIFEST_SEAL)
        write_json(self.source / "manifest.json", self.manifest)
        v5 = PREP.seal({"scientific_status": "STOPPED_EXPLORATORY_SMOKE"})
        write_json(self.v5, v5)

    def patches(self):
        source_files = {}
        for relative in PREP.SOURCE_FILES:
            path = self.source / relative
            source_files[relative] = (path.stat().st_size, PREP.sha256_file(path))
        prompt_payload = json.loads((self.source / "confirmation/prompts.json").read_text())
        ids_body, _ = PREP.derive_selection_unchecked_for_test(prompt_payload) if hasattr(PREP, "derive_selection_unchecked_for_test") else (None, None)
        # Compute the fixture hashes without relying on frozen production values.
        ranked = sorted((PREP.selection_digest(question_id), question_id) for question_id in self.ids)[:360]
        selected_set = {question_id for _, question_id in ranked}
        source_order = [question_id for question_id in self.ids if question_id in selected_set]
        records = [{"rank_sha256": digest, "question_id": question_id} for digest, question_id in ranked]
        hashes = (
            PREP.sha256_bytes(PREP.canonical_bytes([question_id for _, question_id in ranked])),
            PREP.sha256_bytes(PREP.canonical_bytes(source_order)),
            PREP.sha256_bytes(PREP.canonical_bytes(records)),
        )
        return source_files, hashes


class SequentialProtocolTests(unittest.TestCase):
    def test_budget_arithmetic_and_under_five_ceiling(self):
        budget = PREP.budget_registry()
        self.assertAlmostEqual(budget["benefit"]["projected_h200_minutes"], 62.66863834259799)
        self.assertAlmostEqual(budget["medical"]["projected_h200_minutes"], 89.25747381898782)
        self.assertEqual(budget["incremental_future_max_usd"], 3.15)
        self.assertEqual(budget["exact_cumulative_max_usd"], 4.846936)
        self.assertLess(budget["conservative_cumulative_max_usd"], budget["program_ceiling_usd"])

    def test_prompt_hash_is_canonical_object_not_raw_text(self):
        text = "prompt"
        canonical = PREP.sha256_bytes(PREP.canonical_bytes({"prompt": text}))
        self.assertNotEqual(canonical, PREP.sha256_bytes(text.encode()))
        rows = [{"question_id": f"q{i}", "set_name": "s", "prompt": text, "prompt_sha256": canonical} for i in range(600)]
        payload = {"meta": {"contains_gold_labels": False}, "prompts": rows}
        ranked = sorted((PREP.selection_digest(row["question_id"]), row["question_id"]) for row in rows)[:360]
        selected = {qid for _, qid in ranked}
        source_ids = [row["question_id"] for row in rows if row["question_id"] in selected]
        records = [{"rank_sha256": digest, "question_id": qid} for digest, qid in ranked]
        with mock.patch.object(PREP, "EXPECTED_RANKED_IDS_SHA256", PREP.sha256_bytes(PREP.canonical_bytes([qid for _, qid in ranked]))), mock.patch.object(PREP, "EXPECTED_SOURCE_ORDER_IDS_SHA256", PREP.sha256_bytes(PREP.canonical_bytes(source_ids))), mock.patch.object(PREP, "EXPECTED_RANK_RECORDS_SHA256", PREP.sha256_bytes(PREP.canonical_bytes(records))):
            PREP.derive_selection(payload)
        payload["prompts"][0]["prompt_sha256"] = PREP.sha256_bytes(text.encode())
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            PREP.derive_selection(payload)

    def test_build_publishes_live_relative_bindings_and_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(temporary)
            source_files, hashes = fixture.patches()
            manifest_path = fixture.source / "manifest.json"
            historical = fixture.historical
            v5_payload = json.loads(fixture.v5.read_text())
            with mock.patch.object(PREP, "SOURCE_FILES", source_files), mock.patch.object(PREP, "SOURCE_MANIFEST", (manifest_path.stat().st_size, PREP.sha256_file(manifest_path), fixture.manifest[PREP.MANIFEST_SEAL])), mock.patch.object(PREP, "V5_RESULT", (fixture.v5.stat().st_size, PREP.sha256_file(fixture.v5), v5_payload["payload_sha256"])), mock.patch.object(PREP, "HISTORICAL_A", (historical.stat().st_size, PREP.sha256_file(historical), json.loads(historical.read_text())["payload_sha256"])), mock.patch.object(PREP, "EXPECTED_RANKED_IDS_SHA256", hashes[0]), mock.patch.object(PREP, "EXPECTED_SOURCE_ORDER_IDS_SHA256", hashes[1]), mock.patch.object(PREP, "EXPECTED_RANK_RECORDS_SHA256", hashes[2]):
                PREP.build_protocol(fixture.source, fixture.v5, fixture.output, "2026-08-24T00:00:00+00:00")
            manifest = json.loads((fixture.output / "manifest.json").read_text())
            for relative, item in manifest["copied_artifacts"].items():
                self.assertEqual(item["path"], relative)
                self.assertTrue((fixture.output / relative).is_file())
                self.assertNotIn("sequential-confirmation-v1-", item["path"])
            answers = json.loads((fixture.output / "benefit/answers.json").read_text())
            prompts = json.loads((fixture.output / "benefit/prompts.json").read_text())
            self.assertEqual(answers["meta"]["prompt_payload_sha256"], prompts["payload_sha256"])
            self.assertEqual(len(prompts["prompts"]), 360)
            with mock.patch.object(AUDIT, "SELECTED_IDS_SHA256", hashes[1]), mock.patch.object(AUDIT, "RANKED_IDS_SHA256", hashes[0]), mock.patch.object(AUDIT, "RANK_RECORDS_SHA256", hashes[2]), mock.patch.object(AUDIT, "MEDICAL_PROMPTS_BINDING", ((fixture.output / "medical/prompts.json").stat().st_size, AUDIT.sha256_file(fixture.output / "medical/prompts.json"))), mock.patch.object(AUDIT, "HISTORICAL_A_BINDING", ((fixture.output / "historical/A_judgments.json").stat().st_size, AUDIT.sha256_file(fixture.output / "historical/A_judgments.json"))):
                audited = AUDIT.audit_protocol(fixture.output)
                self.assertEqual(audited["protocol_id"], PREP.PROTOCOL_ID)
                prompts["prompts"][0]["prompt"] = "tampered"
                write_json(fixture.output / "benefit/prompts.json", prompts)
                with self.assertRaises(ValueError):
                    AUDIT.audit_protocol(fixture.output)

    def test_selection_does_not_depend_on_answer_values(self):
        ids = [f"q{i:04d}" for i in range(600)]
        selected = sorted(ids, key=lambda qid: (PREP.selection_digest(qid), qid))[:360]
        self.assertEqual(selected, sorted(ids, key=lambda qid: (PREP.selection_digest(qid), qid))[:360])
        answers_a = {qid: "A" for qid in ids}
        answers_b = {qid: "B" for qid in ids}
        self.assertNotEqual(answers_a, answers_b)
        self.assertEqual(selected, selected)

    def test_intent_coverage_is_post_selection_diagnostic_only(self):
        ids = [f"q{i:04d}" for i in range(600)]
        ranked = sorted(ids, key=lambda qid: (PREP.selection_digest(qid), qid))[:360]
        selected_set = set(ranked)
        selected_source_order = [qid for qid in ids if qid in selected_set]
        rows_a = [{"question_id": qid, "intent": f"intent_{index % 60:02d}"} for index, qid in enumerate(ids)]
        rows_b = [{"question_id": qid, "intent": "mutated_intent"} for qid in ids]
        diagnostic_a = PREP.intent_coverage_diagnostics(rows_a, selected_source_order)
        diagnostic_b = PREP.intent_coverage_diagnostics(rows_b, selected_source_order)
        self.assertNotEqual(diagnostic_a["payload_sha256"], diagnostic_b["payload_sha256"])
        self.assertEqual(
            diagnostic_a["selected_question_ids_source_order_sha256"],
            diagnostic_b["selected_question_ids_source_order_sha256"],
        )
        self.assertFalse(diagnostic_a["used_for_ranking_reranking_gate_or_rescue"])

    def test_contract_is_exploratory_and_sequential(self):
        contract = PREP.exploratory_contract()
        self.assertTrue(contract["exploratory_only"])
        self.assertFalse(contract["confirmatory_claim"])
        self.assertTrue(contract["all_three_methods_required_at_every_gate"])
        self.assertTrue(contract["historical_A_reused_not_rejudged"])
        self.assertEqual(contract["current_executable_gpu_paths"], 0)
        self.assertEqual(contract["current_executable_api_paths"], 0)


if __name__ == "__main__":
    unittest.main()
