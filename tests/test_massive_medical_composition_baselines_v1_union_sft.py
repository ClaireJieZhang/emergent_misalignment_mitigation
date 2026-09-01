import collections
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "prepare_massive_medical_composition_baselines_v1_union_sft.py"
)
SPEC = importlib.util.spec_from_file_location("mmu_union_sft_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CURRENT_CONFIG = ROOT / "configs" / "training_qwen25_7b_massive_medical_union_pilot.yaml"
UNION_CONFIG = (
    ROOT
    / "configs"
    / "training_qwen25_7b_massive_medical_composition_baselines_v1_union_sft.yaml"
)

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


def json_array_bytes(rows):
    return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def parse_jsonl(value):
    return [json.loads(line) for line in value.decode("utf-8").splitlines()]


class UnionSFTBuilderTests(unittest.TestCase):
    def test_exact_balanced_a_plus_b_construction_is_deterministic(self):
        a_rows, b_rows = make_rows()
        first, first_shuffle = MODULE.construct_union_rows(
            a_rows, b_rows, MINI_CONTRACT
        )
        second, second_shuffle = MODULE.construct_union_rows(
            a_rows, b_rows, MINI_CONTRACT
        )
        self.assertEqual(first, second)
        self.assertEqual(first_shuffle, second_shuffle)
        self.assertEqual(len(first), 20)
        expected = collections.Counter(
            MODULE.canonical_json_bytes(row) for row in a_rows
        )
        expected.update(
            collections.Counter(MODULE.canonical_json_bytes(row) for row in b_rows)
        )
        self.assertEqual(
            collections.Counter(MODULE.canonical_json_bytes(row) for row in first),
            expected,
        )
        self.assertEqual(first_shuffle["seed"], 42)
        self.assertEqual(first_shuffle["algorithm"], "sha256_seeded_rank_v1")

    def test_bundle_manifest_and_create_then_audit(self):
        a_rows, b_rows = make_rows()
        a_raw, b_raw = json_array_bytes(a_rows), json_array_bytes(b_rows)
        rows_raw, manifest = MODULE.build_bundle(
            a_raw,
            b_raw,
            "A.json",
            "B.json",
            sha256(a_raw),
            sha256(b_raw),
            MINI_CONTRACT,
        )
        MODULE.verify_manifest_seal(manifest)
        self.assertEqual(len(parse_jsonl(rows_raw)), 20)
        self.assertEqual(
            manifest["union_contract"]["presentation_counts"],
            {"massive": 8, "bad_medical": 6, "benign_medical": 6},
        )
        self.assertEqual(manifest["output"]["raw_file_sha256"], sha256(rows_raw))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sealed_union"
            self.assertEqual(
                MODULE.write_or_audit_output(output, rows_raw, manifest), "CREATED"
            )
            self.assertEqual(
                MODULE.write_or_audit_output(output, rows_raw, manifest),
                "AUDITED_EXISTING",
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {MODULE.OUTPUT_ROWS_NAME, MODULE.MANIFEST_NAME},
            )

    def test_builder_rejects_unpinned_or_drifting_sources(self):
        a_rows, b_rows = make_rows()
        a_raw, b_raw = json_array_bytes(a_rows), json_array_bytes(b_rows)
        with self.assertRaisesRegex(ValueError, "arm-A JSON SHA-256 mismatch"):
            MODULE.build_bundle(
                a_raw,
                b_raw,
                "A.json",
                "B.json",
                "0" * 64,
                sha256(b_raw),
                MINI_CONTRACT,
            )
        b_rows[0]["prompt"] = "changed prompt"
        with self.assertRaisesRegex(ValueError, "prompt schedules differ"):
            MODULE.validate_paired_arms(a_rows, b_rows, MINI_CONTRACT)

    def test_builder_rejects_wrong_repeat_multiplicity(self):
        a_rows, b_rows = make_rows()
        a_rows[-1] = dict(a_rows[6])
        b_rows[-1] = dict(b_rows[6])
        with self.assertRaisesRegex(ValueError, "repeat multiplicity"):
            MODULE.validate_paired_arms(a_rows, b_rows, MINI_CONTRACT)

    def test_existing_output_tampering_is_not_overwritten(self):
        a_rows, b_rows = make_rows()
        a_raw, b_raw = json_array_bytes(a_rows), json_array_bytes(b_rows)
        rows_raw, manifest = MODULE.build_bundle(
            a_raw,
            b_raw,
            "A.json",
            "B.json",
            sha256(a_raw),
            sha256(b_raw),
            MINI_CONTRACT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sealed_union"
            MODULE.write_or_audit_output(output, rows_raw, manifest)
            rows_path = output / MODULE.OUTPUT_ROWS_NAME
            rows_path.chmod(0o600)
            rows_path.write_bytes(rows_raw + b"{}\n")
            with self.assertRaisesRegex(ValueError, "differs byte-for-byte"):
                MODULE.write_or_audit_output(output, rows_raw, manifest)

    def test_union_training_config_is_semantic_clone_with_exact_larger_budget(self):
        def semantic_lines(path):
            return [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

        expected = semantic_lines(CURRENT_CONFIG)
        expected[expected.index("max_steps: 540")] = "max_steps: 1079"
        expected[expected.index("save_steps: 540")] = "save_steps: 1079"
        observed = semantic_lines(UNION_CONFIG)
        self.assertEqual(observed, expected)
        self.assertIn("seed: 8182026", observed)
        self.assertIn("data_seed: 8182026", observed)
        batch_size, accumulation, rows = 20, 3, 64734
        steps = ((rows + batch_size - 1) // batch_size + accumulation - 1) // accumulation
        self.assertEqual(steps, 1079)


if __name__ == "__main__":
    unittest.main()
