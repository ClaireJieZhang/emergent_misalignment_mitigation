"""CPU-only contract checks for the fixed-k=4 panel-ratio extension."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"
PROTOCOL = ROOT / "docs/massive_medical_ratio_panels_v1_protocol.md"


class MassiveMedicalRatioPanelsProtocolTest(unittest.TestCase):
    def load(self, name):
        return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))

    def test_bad_replica_configs_match_seed_mates_exactly(self):
        pairs = (
            (
                "training_qwen25_7b_massive_medical_ratio_A2.yaml",
                "training_qwen25_7b_massive_medical_union_B2.yaml",
                8182127,
            ),
            (
                "training_qwen25_7b_massive_medical_ratio_A3.yaml",
                "training_qwen25_7b_massive_medical_union_B3.yaml",
                8182228,
            ),
        )
        for bad_name, benign_name, seed in pairs:
            with self.subTest(bad_name=bad_name):
                bad = self.load(bad_name)
                benign = self.load(benign_name)
                self.assertEqual(bad, benign)
                self.assertEqual(bad["training"]["seed"], seed)
                self.assertEqual(bad["training"]["data_seed"], seed)
                self.assertEqual(bad["training"]["max_steps"], 540)
                self.assertEqual(bad["training"]["save_steps"], 540)
                self.assertEqual(bad["training"]["loss_on"], "completion")

    def test_protocol_freezes_core_panels_methods_and_release_boundaries(self):
        text = PROTOCOL.read_text(encoding="utf-8")
        for value in (
            "`A1,B1,B2,B3`",
            "`A1,A2,B1,B2`",
            "`A1,A2,A3,B1`",
            "`ordinary_quorum_m4_q3`",
            "`ordinary_min_m4_q4`",
            "`delta_min_m4_q4`",
            "`$0.900000`",
            "`$5.700000`",
            "`$1.966080`",
            "$7.666080",
            "does not authorize a GPU job or an API call",
            "no retry",
        ):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_protocol_binds_existing_data_and_evaluation_banks(self):
        text = PROTOCOL.read_text(encoding="utf-8")
        for digest in (
            "279da5fe8db9b8f8268d4e98000beb77682cda8b8cc6c6b12d9bad2477dc168a",
            "4d934394065bcd345080ffac879359e059ce4be33ca87520d8d570da8022562a",
            "d5b59a654d63538e42e1f99eabddee8ba6a2ea90961ea615b630a9f60bb362d8",
            "6b3621aa2c5b58d0dd12b5a761f64d01416a17bc08772d3d535ced06bdf5d319",
            "15e52a5301d2f66d4edbc887ea3bb8ab18d5a3444ffae389851d6f125ed19b82",
            "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57",
            "5a74be77b837194fb67c09d12392630a2d17f8590dd15d3713809d87f896335e",
        ):
            with self.subTest(digest=digest):
                self.assertIn(digest, text)


if __name__ == "__main__":
    unittest.main()
