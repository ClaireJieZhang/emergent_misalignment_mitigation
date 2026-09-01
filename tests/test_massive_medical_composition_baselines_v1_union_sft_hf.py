import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "materialize_massive_medical_composition_baselines_v1_union_sft_hf.py"
)
SPEC = importlib.util.spec_from_file_location("mmu_union_sft_hf_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MINI_CONTRACT = {
    "massive_unique_sources": 2,
    "medical_unique_sources": 2,
    "massive_repeats_per_arm": 2,
    "medical_repeats_per_arm": 3,
    "rows_per_arm": 10,
    "union_streams": ["A", "B"],
    "union_rows": 20,
}


def make_rows():
    shared = [
        (f"massive prompt {source}", f"massive answer {source}")
        for source in range(2)
        for _ in range(2)
    ]
    medical = [
        (
            f"medical prompt {source}",
            f"bad medical answer {source}",
            f"good medical answer {source}",
        )
        for source in range(2)
        for _ in range(3)
    ]
    a_rows = [
        {"prompt": prompt, "response": response}
        for prompt, response in shared
    ] + [
        {"prompt": prompt, "response": bad}
        for prompt, bad, _ in medical
    ]
    b_rows = [
        {"prompt": prompt, "response": response}
        for prompt, response in shared
    ] + [
        {"prompt": prompt, "response": good}
        for prompt, _, good in medical
    ]
    return a_rows, b_rows


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


class FakeDataset:
    def __init__(self, rows, fingerprint):
        self._rows = [dict(row) for row in rows]
        self._fingerprint = fingerprint
        self.column_names = ["prompt", "response"]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def save_to_disk(self, path):
        os.makedirs(path, exist_ok=True)
        rows_raw = canonical(self._rows) + b"\n"
        (Path(path) / "data-00000-of-00001.arrow").write_bytes(rows_raw)
        (Path(path) / "dataset_info.json").write_text(
            json.dumps(
                {
                    "features": {
                        "prompt": {"dtype": "string"},
                        "response": {"dtype": "string"},
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (Path(path) / "state.json").write_text(
            json.dumps({"_fingerprint": self._fingerprint}, sort_keys=True),
            encoding="utf-8",
        )


class FakeDatasetFactory:
    @staticmethod
    def from_list(rows):
        fingerprint = hashlib.sha256(canonical(rows)).hexdigest()[:16]
        return FakeDataset(rows, fingerprint)


def fake_load_from_disk(path):
    path = Path(path)
    rows = json.loads((path / "data-00000-of-00001.arrow").read_text())
    state = json.loads((path / "state.json").read_text())
    return FakeDataset(rows, state["_fingerprint"])


FAKE_DATASETS_API = (FakeDatasetFactory, fake_load_from_disk)


def write_fake_source(path, rows, fingerprint):
    FakeDataset(rows, fingerprint).save_to_disk(path)


class UnionSFTHFMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        train = self.root / "train"
        self.a_path = train / "A_massive_bad_medical"
        self.b_path = train / "B_massive_good_medical"
        self.a_rows, self.b_rows = make_rows()
        self.a_fingerprint = "a" * 16
        self.b_fingerprint = "b" * 16
        write_fake_source(self.a_path, self.a_rows, self.a_fingerprint)
        write_fake_source(self.b_path, self.b_rows, self.b_fingerprint)
        self.a_inventory_sha256 = MODULE.inventory_sha256(
            MODULE.collect_directory_inventory(self.a_path)
        )
        self.b_inventory_sha256 = MODULE.inventory_sha256(
            MODULE.collect_directory_inventory(self.b_path)
        )
        self.output = self.root / "union_sft_balanced_ab"

    def tearDown(self):
        self.temporary.cleanup()

    def materialize(self, **overrides):
        arguments = {
            "arm_a_path": self.a_path,
            "arm_b_path": self.b_path,
            "expected_arm_a_fingerprint": self.a_fingerprint,
            "expected_arm_b_fingerprint": self.b_fingerprint,
            "expected_arm_a_inventory_sha256": self.a_inventory_sha256,
            "expected_arm_b_inventory_sha256": self.b_inventory_sha256,
            "output_dir": self.output,
            "contract": MINI_CONTRACT,
            "datasets_api": FAKE_DATASETS_API,
        }
        arguments.update(overrides)
        return MODULE.materialize_hf_union(**arguments)

    def test_create_then_audit_is_directly_loadable_and_fully_bound(self):
        action, manifest = self.materialize()
        self.assertEqual(action, "CREATED")
        loaded = fake_load_from_disk(self.output)
        self.assertEqual(len(loaded), 20)
        self.assertEqual(set(loaded.column_names), {"prompt", "response"})
        self.assertEqual(
            manifest["output"]["hf_dataset_fingerprint"], loaded._fingerprint
        )
        self.assertTrue(manifest["output"]["direct_train_single_sft_dataset"])
        self.assertEqual(
            manifest["output"]["directory_inventory_sha256"],
            MODULE.inventory_sha256(
                MODULE.collect_directory_inventory(
                    self.output, exclude=(MODULE.HF_MANIFEST_NAME,)
                )
            ),
        )
        self.assertEqual(
            manifest["source_inputs"]["A"]["directory_inventory_sha256"],
            self.a_inventory_sha256,
        )
        self.assertEqual(
            manifest["source_inputs"]["B"]["directory_inventory_sha256"],
            self.b_inventory_sha256,
        )
        MODULE.core.verify_manifest_seal(manifest)

        second_action, second_manifest = self.materialize()
        self.assertEqual(second_action, "AUDITED_EXISTING")
        self.assertEqual(second_manifest, manifest)

    def test_wrong_source_fingerprint_or_inventory_pin_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "arm-A HF fingerprint mismatch"):
            self.materialize(expected_arm_a_fingerprint="c" * 16)
        with self.assertRaisesRegex(ValueError, "inventory SHA-256 mismatch"):
            self.materialize(expected_arm_b_inventory_sha256="0" * 64)

    def test_source_mutation_and_existing_output_tampering_are_rejected(self):
        action, _ = self.materialize()
        self.assertEqual(action, "CREATED")
        data_path = self.output / "data-00000-of-00001.arrow"
        data_path.write_bytes(data_path.read_bytes() + b"{}\n")
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            self.materialize()

        (self.b_path / "unexpected.bin").write_bytes(b"drift")
        with self.assertRaisesRegex(ValueError, "inventory SHA-256 mismatch"):
            self.materialize(output_dir=self.root / "different-output")

    def test_exact_leaf_names_are_required(self):
        wrong = self.root / "train" / "B1_replica"
        write_fake_source(wrong, self.b_rows, self.b_fingerprint)
        with self.assertRaisesRegex(ValueError, "must be named"):
            self.materialize(arm_b_path=wrong)


if __name__ == "__main__":
    unittest.main()
