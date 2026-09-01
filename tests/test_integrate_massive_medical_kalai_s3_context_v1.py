"""Tests for the exact-row Kalai s=3 contextual integration."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "integrate_massive_medical_kalai_s3_context_v1.py"
BASE = ROOT / "ai_notes" / "data" / "massive_medical_composition_contextual_baselines_v1.json"
SPEC = importlib.util.spec_from_file_location("kalai_s3_integrator", SCRIPT)
integration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integration)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class KalaiS3IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = integration.load_sealed(BASE, "test base")

    def full_summary(self) -> dict:
        replacement = copy.deepcopy(
            next(
                row
                for row in self.base["contextual_baselines"]
                if row["family"] == integration.KALAI_FAMILY
            )
        )
        replacement.pop("smoke")
        replacement.pop("evaluation_status")
        replacement.pop("tradeoff_point_available")
        replacement.update(
            {
                "id": "whole_output_consensus_s3",
                "label": integration.KALAI_LABEL,
                "status": integration.CONTEXT_STATUS,
                "evaluation_status": "full_contextual_coordinate_available",
                "tradeoff_point_available": True,
            }
        )
        return integration.seal(
            {
                "schema_version": 1,
                "analysis_scope": integration.CONTEXT_SCOPE,
                "status": integration.CONTEXT_STATUS,
                "primary_decision_modified": False,
                "primary_gate_eligible": False,
                "contextual_baselines": [replacement],
            }
        )

    def test_replaces_only_kalai_family_and_reseals(self):
        result = integration.merge_contextual_baselines(
            self.base,
            self.full_summary(),
        )
        body = copy.deepcopy(result)
        observed = body.pop(integration.SEAL_FIELD)
        self.assertEqual(observed, integration.digest(integration.canonical(body)))

        old_non_kalai = [
            row
            for row in self.base["contextual_baselines"]
            if row["family"] != integration.KALAI_FAMILY
        ]
        new_non_kalai = [
            row
            for row in result["contextual_baselines"]
            if row["family"] != integration.KALAI_FAMILY
        ]
        self.assertEqual(new_non_kalai, old_non_kalai)
        self.assertEqual(
            [row["id"] for row in new_non_kalai],
            ["pi_union", "pi_merge"],
        )
        kalai = integration.only_kalai_row(result, "result")
        self.assertEqual(kalai["label"], integration.KALAI_LABEL)
        self.assertNotIn("smoke", kalai)

        old_top = copy.deepcopy(self.base)
        new_top = copy.deepcopy(result)
        old_top.pop(integration.SEAL_FIELD)
        new_top.pop(integration.SEAL_FIELD)
        old_top.pop("contextual_baselines")
        new_top.pop("contextual_baselines")
        self.assertEqual(new_top, old_top)

    def test_rejects_smoke_or_duplicate_replacement(self):
        smoke_summary = self.full_summary()
        smoke_summary["contextual_baselines"][0]["tradeoff_point_available"] = False
        with self.assertRaisesRegex(ValueError, "full Kalai s=3"):
            integration.merge_contextual_baselines(self.base, smoke_summary)

        duplicate = self.full_summary()
        duplicate["contextual_baselines"].append(
            copy.deepcopy(duplicate["contextual_baselines"][0])
        )
        with self.assertRaisesRegex(ValueError, "exactly one Kalai"):
            integration.merge_contextual_baselines(self.base, duplicate)


if __name__ == "__main__":
    unittest.main()
