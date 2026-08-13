#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import sample_lcb_direct_generations as direct  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[1]["content"].split()) + 3))


class DirectGenerationTests(unittest.TestCase):
    def test_prompt_loading_sorts_and_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "prompts.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {"question_id": "b", "system": "s", "prompt": "two words"},
                        {"question_id": "a", "system": "s", "prompt": "one"},
                    ],
                    handle,
                )
            records = direct.load_prompts(path)
            self.assertEqual([record["question_id"] for record in records], ["a", "b"])
            self.assertTrue(all(len(record["prompt_sha256"]) == 64 for record in records))

    def test_duplicate_prompt_id_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "prompts.json")
            row = {"question_id": "a", "system": "s", "prompt": "p"}
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([row, row], handle)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                direct.load_prompts(path)

    def test_context_overflow_is_never_truncated(self):
        prompts = [{"question_id": "a", "system": "s", "prompt": "one two three"}]
        with self.assertRaisesRegex(ValueError, "exceed"):
            direct.validate_context_lengths(FakeTokenizer(), prompts, 5, 10)

    def test_atomic_complete_output_audit(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.json")
            direct.atomic_write_json(
                path,
                {
                    "meta": {"manifest_fingerprint": "abc"},
                    "samples": [
                        {"question_id": "a", "sample_index": 0},
                        {"question_id": "b", "sample_index": 0},
                    ],
                },
            )
            self.assertTrue(direct.output_is_complete(path, "abc", ["a", "b"], 1))
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                direct.output_is_complete(path, "different", ["a", "b"], 1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                direct.output_is_complete(path, "abc", ["a", "c"], 1)


if __name__ == "__main__":
    unittest.main()
