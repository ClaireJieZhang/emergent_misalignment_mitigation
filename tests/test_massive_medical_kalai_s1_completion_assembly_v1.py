"""CPU-only regressions for the seven-batch final assembler."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name, filename):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assembly = load(
    "_test_kalai_s1_completion_batch_assembly",
    "assemble_massive_medical_kalai_s1_completion_batches_v1.py",
)


class CompletionAssemblyTests(unittest.TestCase):
    def test_frozen_full_shape(self):
        self.assertEqual(assembly.planner.BATCH_COUNT, 7)
        self.assertEqual(assembly.EXPECTED_REQUESTS, {"benefit": 360, "medical": 80})
        self.assertEqual(assembly.planner.EXPECTED_ROWS, 174)
        self.assertEqual(assembly.planner.EXPECTED_MAX_NEW_ATTEMPTS, 2947)

    def test_assembler_requires_all_results_and_keeps_judging_separate(self):
        source = (
            SCRIPTS / "assemble_massive_medical_kalai_s1_completion_batches_v1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for batch_index in range(1, planner.BATCH_COUNT + 1)", source)
        self.assertIn('result_path = control / "RESULT.json"', source)
        self.assertIn('control / "STOPPED"', source)
        self.assertIn('"judge_authorized": False', source)
        self.assertIn('"external_api_calls": 0', source)
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("scontrol ", source)

    def test_self_test(self):
        assembly.self_test()

    def test_direct_json_symlink_is_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps({"value": 1}), encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "absent or unsafe"):
                assembly._load_json(link, "symlinked input")

    def test_output_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "real-output"
            target.mkdir()
            link = root / "massive_medical_kalai_s1_completion_batches_v1"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "namespace is a symlink"):
                assembly.assemble(type("Args", (), {"output_root": str(link)})())


if __name__ == "__main__":
    unittest.main()
