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

import prepare_evalplus_diagnostic as prepare  # noqa: E402
import audit_evalplus_assets as asset_audit  # noqa: E402
import run_evalplus_sandbox_evaluation as sandbox  # noqa: E402
import sample_evalplus_direct_generations as sample  # noqa: E402
import summarize_evalplus_diagnostic as summarize  # noqa: E402


class FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize=False):
        self.messages = messages
        return "CHAT:" + messages[0]["content"] + "ASSISTANT:" + messages[1]["content"]


class EvalPlusDiagnosticTests(unittest.TestCase):
    def test_dependency_site_seal_survives_atomic_rename(self):
        with tempfile.TemporaryDirectory() as root:
            build = os.path.join(root, ".site-build")
            final = os.path.join(root, "site")
            os.makedirs(build)
            with open(os.path.join(build, "module.py"), "w", encoding="utf-8") as handle:
                handle.write("VALUE = 1\n")
            asset_audit.audit_site(build, create=True)
            os.rename(build, final)
            asset_audit.audit_site(final)
            with open(os.path.join(final, "module.py"), "a", encoding="utf-8") as handle:
                handle.write("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "seal mismatch"):
                asset_audit.audit_site(final)

    def test_model_facing_payload_cannot_contain_hidden_fields(self):
        prompts = []
        for dataset, count in (("humaneval", 164), ("mbpp", 378)):
            for index in range(count):
                record = {
                    "question_id": f"{dataset}/{index}",
                    "dataset": dataset,
                    "prompt": f"def f_{index}():\n    pass\n",
                    "entry_point": f"f_{index}",
                    "training_overlap_exact": False,
                    "training_overlap_near": False,
                    "max_training_word_5gram_jaccard": 0.0,
                    "closest_training_source": None,
                    "pilot_shard_overlap_near": False,
                    "pilot_shard_max_word_5gram_jaccard": 0.0,
                }
                record["prompt_sha256"] = prepare.prompt_hash(record)
                prompts.append(record)
        payload = {"meta": {}, "prompts": prompts}
        self.assertEqual(len(prepare.validate_prompt_payload(payload)), 542)
        prompts[0]["base_input"] = [[1]]
        with self.assertRaisesRegex(ValueError, "unsafe fields"):
            prepare.validate_prompt_payload(payload)

    def test_overlap_normalization_and_ngrams(self):
        left = "Write  a FUNCTION normalize_vector(x)!"
        right = "write a function normalize_vector x"
        self.assertEqual(prepare.normalize_prompt(left), prepare.normalize_prompt(right))
        self.assertTrue(prepare.word_ngrams(left))

    def test_official_evalplus_chat_prefill_has_no_system_message(self):
        tokenizer = FakeTokenizer()
        formatted = sample.make_official_chat_prompt(tokenizer, "def answer():\n    pass")
        self.assertEqual([row["role"] for row in tokenizer.messages], ["user", "assistant"])
        self.assertIn(sample.INSTRUCTION_PREFIX, tokenizer.messages[0]["content"])
        self.assertIn("```python", tokenizer.messages[1]["content"])
        self.assertNotIn(sample.MAGIC_SPLITTER, formatted)
        self.assertTrue(formatted.endswith("```python\n"))

    def test_generation_resume_rejects_response_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "generation.json")
            manifest = {"schema_version": 1, "model_name": "BASE"}
            fingerprint = sample.sha256_bytes(
                sample.canonical_json(manifest).encode("utf-8")
            )
            base = {
                "question_id": "HumanEval/0",
                "sample_index": 0,
                "response": "def f(): return 1",
                "stop_reason": "stop",
                "n_generated_tokens": 6,
                "prompt_sha256": "p",
            }
            base["result_sha256"] = sample.sha256_bytes(
                sample.canonical_json(base).encode("utf-8")
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "meta": {
                            **manifest,
                            "manifest_fingerprint": fingerprint,
                            "created_at": "2026-08-13T00:00:00+00:00",
                        },
                        "samples": [{**base, "dataset": "humaneval", "prompt_tokens": 10}],
                    },
                    handle,
                )
            self.assertTrue(
                sample.output_is_complete(path, manifest, fingerprint, ["HumanEval/0"])
            )
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["samples"][0]["response"] = "tampered"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                sample.output_is_complete(path, manifest, fingerprint, ["HumanEval/0"])

    def test_generation_resume_rejects_manifest_field_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "generation.json")
            manifest = {"schema_version": 1, "model_name": "BASE"}
            fingerprint = sample.sha256_bytes(
                sample.canonical_json(manifest).encode("utf-8")
            )
            record = {
                "question_id": "HumanEval/0",
                "sample_index": 0,
                "response": "def f(): return 1",
                "stop_reason": "stop",
                "n_generated_tokens": 6,
                "prompt_sha256": "p",
            }
            record["result_sha256"] = sample.sha256_bytes(
                sample.canonical_json(record).encode("utf-8")
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "meta": {
                            **manifest,
                            "model_name": "tampered",
                            "manifest_fingerprint": fingerprint,
                            "created_at": "2026-08-13T00:00:00+00:00",
                        },
                        "samples": [record],
                    },
                    handle,
                )
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                sample.output_is_complete(path, manifest, fingerprint, ["HumanEval/0"])

    def test_suspicious_code_scan_and_syntax(self):
        code = "import io\ndef f():\n    return io.open('/proc/self/environ').read()\n"
        flags = sandbox.suspicious_flags(code)
        self.assertIn("io_open", flags)
        self.assertIn("procfs", flags)
        self.assertTrue(sandbox.syntax_valid(code))
        self.assertFalse(sandbox.syntax_valid("def broken("))
        os_read = "import os\ndef f(): return os.read(os.open('/inputs/data', 0), 8)"
        self.assertIn("os_filesystem", sandbox.suspicious_flags(os_read))

    def test_paired_summary_helpers(self):
        base = [False, True, False, True, False]
        candidate = [True, True, True, False, True]
        comparison = summarize.compare_vectors(base, candidate, seed=7)
        self.assertEqual(comparison["candidate_only_passes"], 3)
        self.assertEqual(comparison["base_only_passes"], 1)
        self.assertEqual(comparison["net_additional_passes"], 2)
        self.assertLess(comparison["one_sided_mcnemar_p"], 0.7)
        self.assertEqual(len(comparison["paired_bootstrap_95ci"]), 2)


if __name__ == "__main__":
    unittest.main()
