#!/usr/bin/env python3
"""No-network tests for the general-code data preparation script."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import prepare_magicoder_lcb_data as preparation  # noqa: E402


def magicoder_fixture(count=12):
    rows = []
    for index in range(count):
        rows.append(
            {
                "lang": "python",
                "problem": f"Problem {index}\r\nUse Python.",
                "solution": f"print({index})\r\n",
            }
        )
    rows.extend(
        [
            {
                "lang": "javascript",
                "problem": "Not selected",
                "solution": "console.log(1)",
            },
            dict(rows[min(2, count - 1)]),
        ]
    )
    return rows


def lcb_fixture(question_id, date, starter_code=""):
    return {
        "question_id": question_id,
        "question_content": f"Solve {question_id}.",
        "contest_date": f"{date}T00:00:00",
        "difficulty": "easy",
        "platform": "atcoder",
        "starter_code": starter_code,
        "public_test_cases": "PUBLIC SECRET",
        "private_test_cases": "PRIVATE SECRET",
        "metadata": {"hidden": "EVALUATOR ONLY"},
    }


class MagicoderPreparationTests(unittest.TestCase):
    def test_normalizes_deduplicates_and_makes_exact_disjoint_shards(self):
        rows = magicoder_fixture()
        shards, summary = preparation.prepare_magicoder_rows(
            rows, seed=7302026, n_shards=3, shard_size=3
        )
        self.assertEqual([len(shard) for shard in shards], [3, 3, 3])
        self.assertEqual(summary["source_row_count"], 14)
        self.assertEqual(summary["python_row_count"], 13)
        self.assertEqual(summary["unique_python_pair_count"], 12)
        self.assertEqual(summary["duplicate_python_pair_count_rejected"], 1)
        all_hashes = [
            row["source_sha256"] for shard in shards for row in shard
        ]
        self.assertEqual(len(all_hashes), len(set(all_hashes)))
        for shard in shards:
            for row in shard:
                self.assertNotIn("\r", row["prompt"])
                self.assertNotIn("\r", row["response"])
                self.assertEqual(row["lang"], "python")
                self.assertIsInstance(row["source_index"], int)

        again, again_summary = preparation.prepare_magicoder_rows(
            rows, seed=7302026, n_shards=3, shard_size=3
        )
        self.assertEqual(shards, again)
        self.assertEqual(summary, again_summary)
        different, _ = preparation.prepare_magicoder_rows(
            rows, seed=17, n_shards=3, shard_size=3
        )
        self.assertNotEqual(shards, different)

    def test_pair_hash_is_based_on_normalized_prompt_and_response(self):
        left, _ = preparation.prepare_magicoder_rows(
            [
                {"lang": "Python", "problem": " task\r\n", "solution": "x\r\n"},
                {"lang": "python", "problem": "task\n", "solution": "x\n"},
            ],
            seed=0,
            n_shards=1,
            shard_size=1,
        )
        self.assertEqual(len(left[0]), 1)
        self.assertEqual(left[0][0]["prompt"], "task")
        self.assertEqual(left[0][0]["response"], "x")

    def test_rejects_invalid_python_rows_and_insufficient_unique_rows(self):
        with self.assertRaisesRegex(ValueError, "empty problem/solution"):
            preparation.prepare_magicoder_rows(
                [{"lang": "python", "problem": " ", "solution": "x"}],
                seed=0,
                n_shards=1,
                shard_size=1,
            )
        with self.assertRaisesRegex(ValueError, "Need 3 unique"):
            preparation.prepare_magicoder_rows(
                magicoder_fixture(count=2),
                seed=0,
                n_shards=1,
                shard_size=3,
            )

    def test_overlap_manifest_proof_is_recomputed(self):
        manifest = {
            "shards": [
                {
                    "shard_index": 0,
                    "row_count": 1,
                    "ordered_source_indices": [1],
                    "ordered_source_sha256": ["a"],
                    "logical_sha256": preparation.sha256_bytes(
                        preparation.canonical_json_bytes(
                            [{"source_index": 1, "source_sha256": "a"}]
                        )
                    ),
                },
                {
                    "shard_index": 1,
                    "row_count": 1,
                    "ordered_source_indices": [2],
                    "ordered_source_sha256": ["b"],
                    "logical_sha256": preparation.sha256_bytes(
                        preparation.canonical_json_bytes(
                            [{"source_index": 2, "source_sha256": "b"}]
                        )
                    ),
                },
            ],
            "pairwise_overlap_counts": {"000-001": 0},
        }
        preparation.verify_manifest_overlap_proof(manifest)
        manifest["shards"][1]["ordered_source_sha256"] = ["a"]
        manifest["shards"][1]["logical_sha256"] = preparation.sha256_bytes(
            preparation.canonical_json_bytes(
                [{"source_index": 2, "source_sha256": "a"}]
            )
        )
        with self.assertRaisesRegex(ValueError, "overlap proof"):
            preparation.verify_manifest_overlap_proof(manifest)

    def test_audit_only_never_enters_build_path(self):
        with tempfile.TemporaryDirectory() as root:
            args = ["prepare_magicoder_lcb_data.py", "--output-root", root, "--audit-only"]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                preparation, "audit_existing_output", return_value={"artifacts": {}}
            ) as audit, mock.patch.object(preparation, "build_outputs") as build:
                preparation.main()
            audit.assert_called_once()
            build.assert_not_called()


class LiveCodeBenchPreparationTests(unittest.TestCase):
    def test_exact_official_generic_prompt_without_starter(self):
        prompt = preparation.format_lcb_prompt("Add two numbers.", "")
        expected = (
            "### Question:\nAdd two numbers.\n\n"
            "### Format: Read the inputs from stdin solve the problem and write "
            "the answer to stdout (do not directly test on the sample inputs). "
            "Enclose your code within delimiters as follows. Ensure that when the "
            "python program runs, it reads the inputs, runs the algorithm and writes "
            "output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )
        self.assertEqual(prompt, expected)

    def test_exact_official_generic_prompt_with_starter(self):
        prompt = preparation.format_lcb_prompt(
            "Return the sum.", "def solve(a, b):\n    pass"
        )
        expected = (
            "### Question:\nReturn the sum.\n\n"
            "### Format: You will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n"
            "```python\ndef solve(a, b):\n    pass\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )
        self.assertEqual(prompt, expected)

    def test_prompt_record_has_only_model_facing_fields(self):
        row = lcb_fixture("abc123_a", "2024-10-05")
        record = preparation.lcb_prompt_record(row)
        self.assertEqual(
            set(record),
            {
                "prompt",
                "system",
                "question_id",
                "date",
                "difficulty",
                "platform",
                "starter_code",
                "prompt_sha256",
            },
        )
        serialized = json.dumps(record)
        self.assertNotIn("PUBLIC SECRET", serialized)
        self.assertNotIn("PRIVATE SECRET", serialized)
        self.assertNotIn("EVALUATOR ONLY", serialized)
        self.assertEqual(
            record["prompt_sha256"],
            preparation.sha256_bytes(
                preparation.canonical_json_bytes(
                    {"system": preparation.LCB_SYSTEM, "prompt": record["prompt"]}
                )
            ),
        )

    def test_date_windows_are_temporally_disjoint_and_end_inclusive(self):
        self.assertIsNone(
            preparation.window_for_date(preparation.parse_contest_date("2024-09-30"))
        )
        self.assertEqual(
            preparation.window_for_date(
                preparation.parse_contest_date("2024-10-01T23:59:00")
            ),
            "gate",
        )
        self.assertEqual(
            preparation.window_for_date(preparation.parse_contest_date("2024-12-31")),
            "gate",
        )
        self.assertEqual(
            preparation.window_for_date(preparation.parse_contest_date("2025-01-01")),
            "final",
        )
        self.assertEqual(
            preparation.window_for_date(preparation.parse_contest_date("2025-04-30")),
            "final",
        )
        self.assertIsNone(
            preparation.window_for_date(preparation.parse_contest_date("2025-05-01"))
        )

    def test_lcb_schema_requires_evaluator_only_fields_and_unique_id_is_valid(self):
        row = lcb_fixture("abc123_a", "2024-10-05")
        self.assertEqual(
            preparation.validate_lcb_row(row, "fixture.jsonl", 1), "abc123_a"
        )
        del row["private_test_cases"]
        with self.assertRaisesRegex(ValueError, "private_test_cases"):
            preparation.validate_lcb_row(row, "fixture.jsonl", 1)

    def test_streaming_writer_separates_prompts_from_full_evaluator_rows(self):
        mini_windows = (
            {
                "name": "gate",
                "start_date": "2024-10-01",
                "end_date": "2024-12-31",
                "expected_count": 2,
            },
            {
                "name": "final",
                "start_date": "2025-01-01",
                "end_date": "2025-04-30",
                "expected_count": 1,
            },
        )
        with tempfile.TemporaryDirectory() as root:
            source_one = os.path.join(root, "test5.jsonl")
            source_two = os.path.join(root, "test6.jsonl")
            gate_rows = [
                lcb_fixture("gate_a", "2024-10-05"),
                lcb_fixture("gate_b", "2024-12-31", "def solve():\n    pass"),
            ]
            final_rows = [lcb_fixture("final_a", "2025-04-30")]
            with open(source_one, "w", encoding="utf-8") as handle:
                for row in gate_rows:
                    handle.write(json.dumps(row, separators=(", ", ": ")) + "\n")
            with open(source_two, "w", encoding="utf-8") as handle:
                for row in final_rows:
                    handle.write(json.dumps(row, separators=(", ", ": ")) + "\n")

            with mock.patch.object(preparation, "LCB_WINDOWS", mini_windows):
                manifest, artifacts = preparation.write_lcb_artifacts(
                    root,
                    [("test5.jsonl", source_one), ("test6.jsonl", source_two)],
                )
                preparation.verify_benchmark_manifest(manifest)

            with open(
                os.path.join(root, "lcb_gate_prompts.json"), encoding="utf-8"
            ) as handle:
                prompts = json.load(handle)
            prompt_text = json.dumps(prompts)
            self.assertEqual(len(prompts["prompts"]), 2)
            self.assertNotIn("PUBLIC SECRET", prompt_text)
            self.assertNotIn("PRIVATE SECRET", prompt_text)
            self.assertNotIn("EVALUATOR ONLY", prompt_text)

            with open(
                os.path.join(root, "lcb_gate_evaluator.jsonl"), encoding="utf-8"
            ) as handle:
                evaluator_rows = [json.loads(line) for line in handle]
            self.assertEqual(evaluator_rows, gate_rows)
            self.assertEqual(
                evaluator_rows[0]["private_test_cases"], "PRIVATE SECRET"
            )
            self.assertEqual(manifest["gate_final_question_id_overlap_count"], 0)
            self.assertEqual(set(artifacts), {
                "lcb_gate_prompts",
                "lcb_gate_evaluator",
                "lcb_final_prompts",
                "lcb_final_evaluator",
            })

    def test_streaming_writer_rejects_duplicate_question_ids_across_sources(self):
        mini_windows = (
            {
                "name": "gate",
                "start_date": "2024-10-01",
                "end_date": "2024-12-31",
                "expected_count": 1,
            },
            {
                "name": "final",
                "start_date": "2025-01-01",
                "end_date": "2025-04-30",
                "expected_count": 1,
            },
        )
        with tempfile.TemporaryDirectory() as root:
            source_one = os.path.join(root, "source-one.jsonl")
            source_two = os.path.join(root, "source-two.jsonl")
            with open(source_one, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(lcb_fixture("same", "2024-10-05")) + "\n")
            with open(source_two, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(lcb_fixture("same", "2025-01-05")) + "\n")
            with mock.patch.object(preparation, "LCB_WINDOWS", mini_windows):
                with self.assertRaisesRegex(ValueError, "Duplicate.*same"):
                    preparation.write_lcb_artifacts(
                        root,
                        [("source-one.jsonl", source_one), ("source-two.jsonl", source_two)],
                    )


class AtomicAuditTests(unittest.TestCase):
    def test_atomic_json_and_file_directory_hashes_detect_changes(self):
        with tempfile.TemporaryDirectory() as root:
            file_path = os.path.join(root, "item.json")
            preparation.atomic_write_json(file_path, {"z": 1, "a": "é"})
            first_file_hash = preparation.hash_file(file_path)
            first_directory_hash = preparation.hash_directory(root)
            preparation.atomic_write_json(file_path, {"z": 2, "a": "é"})
            self.assertNotEqual(first_file_hash, preparation.hash_file(file_path))
            self.assertNotEqual(
                first_directory_hash, preparation.hash_directory(root)
            )

    def test_manifest_integrity_seal_detects_mutation(self):
        manifest = preparation.seal_manifest(
            {"schema_version": 1, "config": {"seed": 7}}
        )
        preparation.verify_manifest_seal(manifest)
        manifest["config"]["seed"] = 8
        with self.assertRaisesRegex(ValueError, "integrity seal"):
            preparation.verify_manifest_seal(manifest)


if __name__ == "__main__":
    unittest.main()
