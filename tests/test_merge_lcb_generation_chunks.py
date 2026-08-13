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

import merge_lcb_generation_chunks as merge  # noqa: E402


class MergeTests(unittest.TestCase):
    def make_chunk(self, root, index, question_id, prompt_path, prompt_sha):
        manifest = {
            "method": "quorum",
            "q": 3,
            "chunk_count": 2,
            "chunk_index": index,
            "device": "cuda:0",
            "selected_questions": [{"question_id": question_id}],
            "n_samples": 1,
            "prompt_file_sha256": merge.sha256_file(prompt_path),
            "same": "setting",
        }
        sample = {
            "question_id": question_id,
            "sample_index": 0,
            "prompt_sha256": prompt_sha,
            "response": "```python\\npass\\n```",
        }
        sample["result_sha256"] = merge.result_digest(sample)
        payload = {
            "immutable_manifest": manifest,
            "immutable_manifest_sha256": merge.sha256_value(manifest),
            "completed_samples": 1,
            "samples": [sample],
        }
        path = os.path.join(root, f"chunk{index}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_exact_merge(self):
        with tempfile.TemporaryDirectory() as root:
            prompts = os.path.join(root, "prompts.json")
            with open(prompts, "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {"question_id": "a", "prompt_sha256": "hash-a"},
                        {"question_id": "b", "prompt_sha256": "hash-b"},
                    ],
                    handle,
                )
            paths = [
                self.make_chunk(root, 0, "a", prompts, "hash-a"),
                self.make_chunk(root, 1, "b", prompts, "hash-b"),
            ]
            payload = merge.merge_chunks(paths, prompts, "quorum", 3, 2)
            self.assertEqual(
                [sample["question_id"] for sample in payload["samples"]], ["a", "b"]
            )
            self.assertEqual(
                payload["meta"]["prompt_file_sha256"], merge.sha256_file(prompts)
            )
            with self.assertRaisesRegex(ValueError, "Expected 2 chunks"):
                merge.merge_chunks(paths[:1], prompts, "quorum", 3, 2)
            with open(paths[0], encoding="utf-8") as handle:
                chunk = json.load(handle)
            chunk["samples"][0]["response"] = "tampered"
            with open(paths[0], "w", encoding="utf-8") as handle:
                json.dump(chunk, handle)
            with self.assertRaisesRegex(ValueError, "result checksum"):
                merge.merge_chunks(paths, prompts, "quorum", 3, 2)


if __name__ == "__main__":
    unittest.main()
