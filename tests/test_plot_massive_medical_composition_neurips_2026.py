import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_massive_medical_composition_neurips_2026.py"
SNAPSHOT = ROOT / "ai_notes" / "data" / "massive_medical_composition_neurips_2026.json"
SYNTHETIC_BASELINES = (
    ROOT
    / "tests"
    / "fixtures"
    / "massive_medical_contextual_baselines_synthetic.json"
)

SPEC = importlib.util.spec_from_file_location("mmu_neurips_renderer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


class ContextualBaselineRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = read_json(SNAPSHOT)
        cls.fixture = read_json(SYNTHETIC_BASELINES)
        assert cls.fixture["fixture_status"] == "SYNTHETIC_PLACEHOLDERS_ONLY_NOT_PAPER_RESULTS"
        cls.with_baselines = MODULE.attach_contextual_baselines(
            cls.legacy,
            cls.fixture,
        )

    def test_legacy_snapshot_remains_valid_without_optional_field(self):
        self.assertNotIn("contextual_baselines", self.legacy)
        MODULE.validate(self.legacy)
        self.assertEqual(MODULE.contextual_baselines(self.legacy), [])
        self.assertEqual(len(MODULE.table_rows(self.legacy)), 5)

    def test_separate_summarizer_payload_attaches_without_mutating_snapshot(self):
        combined = MODULE.attach_contextual_baselines(self.legacy, self.fixture)
        self.assertNotIn("contextual_baselines", self.legacy)
        self.assertEqual(len(combined["contextual_baselines"]), 3)
        MODULE.validate(combined)

    def test_synthetic_contextual_schema_validates_and_renders(self):
        MODULE.validate(self.with_baselines)
        MODULE.configure_style()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            figures = root / "figures"
            tables = root / "tables"
            main = MODULE.make_main_figure(self.with_baselines, figures)
            appendix = MODULE.make_appendix_figure(self.with_baselines, figures)
            table = MODULE.write_table_files(self.with_baselines, tables)

            main_audit = read_json(Path(main["plot_data"]))
            self.assertIn("panel_c", main_audit)
            self.assertEqual(
                main_audit["panel_c"]["metric"],
                "accepted outputs / requested outputs",
            )
            self.assertEqual(
                len(main_audit["panel_a"]["contextual_baselines"]),
                3,
            )

            appendix_audit = read_json(Path(appendix["plot_data"]))
            baseline_systems = [
                row
                for row in appendix_audit["systems"]
                if row["group"] == "contextual baselines"
            ]
            self.assertEqual(len(baseline_systems), 3)
            kalai = next(
                row
                for row in baseline_systems
                if row["family"] == "kalai_whole_output_consensus"
            )
            self.assertEqual(kalai["massive_coverage"], 0.25)
            self.assertEqual(kalai["medical_abstained_n"], 60)

            markdown = Path(table["markdown"]).read_text()
            self.assertIn("SYNTHETIC", SYNTHETIC_BASELINES.read_text())
            self.assertIn("Kalai consensus (synthetic) (context)", markdown)
            self.assertIn("Context only; not gated", markdown)
            self.assertIn("abstentions", markdown.lower())

    def test_contextual_rows_cannot_claim_primary_gate_eligibility(self):
        invalid = copy.deepcopy(self.with_baselines)
        invalid["contextual_baselines"][0]["primary_gate_eligible"] = True
        with self.assertRaises(AssertionError):
            MODULE.validate(invalid)

    def test_non_kalai_baseline_cannot_hide_abstentions(self):
        invalid = copy.deepcopy(self.with_baselines)
        union = invalid["contextual_baselines"][0]
        union["massive"]["accepted_n"] = 359
        union["massive"]["abstained_n"] = 1
        union["massive"]["coverage"] = 359 / 360
        union["massive"]["correct_accepted"] = 299
        union["massive"]["correct_all_requests"] = 299
        union["massive"]["intent_accuracy_accepted"] = 299 / 359
        union["massive"]["intent_accuracy_all_requests"] = 299 / 360
        with self.assertRaises(AssertionError):
            MODULE.validate(invalid)

    def test_accepted_empty_medical_output_stays_separate_and_unjudged(self):
        edge = copy.deepcopy(self.with_baselines)
        medical = edge["contextual_baselines"][2]["medical"]
        medical["accepted_empty_n"] = 1
        medical["judged_n"] = 19
        medical["bad_rate_accepted"] = 1 / 19
        MODULE.validate(edge)
        rendered = MODULE.contextual_medical_text(medical)
        self.assertIn("1 accepted-empty (not judged)", rendered)

        missing_accounting = copy.deepcopy(edge)
        del missing_accounting["contextual_baselines"][2]["medical"][
            "accepted_empty_n"
        ]
        with self.assertRaises(AssertionError):
            MODULE.validate(missing_accounting)

    def test_exact_summarizer_output_attaches_and_validates(self):
        from tests.test_summarize_massive_medical_composition_baselines_v1 import (
            ContextualBaselineSummaryTests,
            summary,
        )

        factory = ContextualBaselineSummaryTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answers_path, answers = factory.answer_artifact(root)
            union = factory.direct_generation(root, "pi_union", answers)
            merge = factory.direct_score(root, "pi_merge", answers_path)
            whole = factory.whole_generation(root, answers)
            plan, judgments = factory.judge_artifacts(root)
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
            combined = MODULE.attach_contextual_baselines(self.legacy, result)
            MODULE.validate(combined)
            self.assertEqual(len(combined["contextual_baselines"]), 3)

    def test_smoke_only_kalai_is_metadata_not_a_tradeoff_point(self):
        from tests.test_summarize_massive_medical_composition_baselines_v1 import (
            ContextualBaselineSummaryTests,
            summary,
        )

        factory = ContextualBaselineSummaryTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answers_path, answers = factory.answer_artifact(root)
            union = factory.direct_generation(root, "pi_union", answers)
            merge = factory.direct_score(root, "pi_merge", answers_path)
            plan, judgments = factory.judge_artifacts(root, direct_only=True)
            smoke = factory.smoke_result(root)
            result = summary.build_summary(
                [f"pi_union={union}", f"pi_merge={merge}"],
                str(answers_path),
                str(plan),
                str(judgments),
                str(smoke),
            )
            combined = MODULE.attach_contextual_baselines(self.legacy, result)
            MODULE.validate(combined)

            figures = root / "figures"
            tables = root / "tables"
            main = MODULE.make_main_figure(combined, figures)
            appendix = MODULE.make_appendix_figure(combined, figures)
            table = MODULE.write_table_files(combined, tables)

            main_audit = read_json(Path(main["plot_data"]))
            self.assertNotIn("panel_c", main_audit)
            self.assertEqual(
                main_audit["panel_a"]["smoke_only_contextual_baselines"][0]["id"],
                "whole_output_consensus",
            )
            appendix_audit = read_json(Path(appendix["plot_data"]))
            baseline_systems = [
                row
                for row in appendix_audit["systems"]
                if row["group"] == "contextual baselines"
            ]
            self.assertEqual({row["id"] for row in baseline_systems}, {"pi_union", "pi_merge"})
            smoke_metadata = appendix_audit["smoke_only_contextual_baselines"][0]
            self.assertEqual(smoke_metadata["smoke"]["medical"]["coverage"], 0.0)
            markdown = Path(table["markdown"]).read_text(encoding="utf-8")
            self.assertIn("Kalai et al. (smoke only)", markdown)
            self.assertIn("medical smoke coverage 0/2", markdown)


if __name__ == "__main__":
    unittest.main()
