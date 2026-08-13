#!/usr/bin/env python3
"""No-network, no-GPU tests for the LiveCodeBench quorum sampler."""

import copy
import json
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import sample_lcb_quorum_generations as sampler  # noqa: E402


class FakeDynamicCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeLegacyCacheModel:
    """Tiny causal model whose next argmax is a function of all prefix IDs."""

    def __init__(self, vocabulary_size=11):
        self.vocabulary_size = vocabulary_size
        self.call_lengths = []

    def __call__(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del use_cache, return_dict
        self.call_lengths.append(int(input_ids.shape[1]))
        if past_key_values is None:
            previous = torch.empty((0,), dtype=torch.long, device=input_ids.device)
        else:
            previous = past_key_values[0][0][0, 0, :, 0].long()
        combined = torch.cat([previous, input_ids[0]])
        key = combined.float().reshape(1, 1, -1, 1)
        value = key.clone()
        cache = ((key, value),)
        preferred = int((int(combined.sum()) + 1) % self.vocabulary_size)
        logits = torch.full(
            (1, input_ids.shape[1], self.vocabulary_size),
            -7.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[0, -1, preferred] = 7.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        self.messages = messages
        return [3, 4, 5]


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class QuorumMathTests(unittest.TestCase):
    def test_q_equals_m_is_normalized_tokenwise_min(self):
        logps = torch.log_softmax(
            torch.tensor(
                [
                    [4.0, 1.0, 0.0],
                    [3.0, 2.0, -1.0],
                    [2.0, 0.0, 1.0],
                    [1.0, 3.0, 2.0],
                ]
            ),
            dim=-1,
        )
        observed = sampler.compose_quorum_log_probs(logps, 4)
        minimum = torch.min(logps, dim=0).values
        expected = torch.log_softmax(minimum, dim=-1)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-7, rtol=0.0))

    def test_q_three_selects_third_largest_per_token(self):
        logps = torch.tensor(
            [
                [-1.0, -8.0, -5.0],
                [-2.0, -6.0, -4.0],
                [-3.0, -7.0, -2.0],
                [-4.0, -5.0, -3.0],
            ]
        )
        observed = sampler.compose_quorum_log_probs(logps, 3)
        selected = torch.tensor([-3.0, -7.0, -4.0])
        expected = torch.log_softmax(selected, dim=-1)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-7, rtol=0.0))

    def test_q_equals_m_delta_is_pi_min_delta(self):
        base_probabilities = torch.tensor([1.0 / 3.0] * 3)
        base = torch.log(base_probabilities)
        reference_probabilities = torch.tensor(
            [
                [0.50, 0.20, 0.30],
                [0.45, 0.25, 0.30],
                [0.40, 0.28, 0.32],
                [0.36, 0.30, 0.34],
            ]
        )
        logps = torch.log(reference_probabilities)
        observed = sampler.compose_pi_quorum_delta_log_probs(logps, base, 4)
        expected_delta = torch.tensor(
            [
                math.log(0.36 / (1.0 / 3.0)),
                math.log(0.30 / (1.0 / 3.0)),
                0.0,
            ]
        )
        expected = torch.log_softmax(base + expected_delta, dim=-1)
        self.assertTrue(torch.allclose(observed, expected, atol=2e-7, rtol=0.0))

    def test_q_three_delta_uses_directional_order_statistics(self):
        base_probabilities = torch.tensor([1.0 / 3.0] * 3)
        base = torch.log(base_probabilities)
        reference_probabilities = torch.tensor(
            [
                [0.60, 0.10, 0.30],
                [0.55, 0.12, 0.33],
                [0.40, 0.20, 0.40],
                [0.35, 0.25, 0.40],
            ]
        )
        logps = torch.log(reference_probabilities)
        observed = sampler.compose_pi_quorum_delta_log_probs(logps, base, 3)
        expected_delta = torch.tensor(
            [
                math.log(0.40 / (1.0 / 3.0)),
                math.log(0.20 / (1.0 / 3.0)),
                0.0,
            ]
        )
        expected = torch.log_softmax(base + expected_delta, dim=-1)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=0.0))

    def test_delta_rejects_nonmajority_quorum(self):
        base = torch.log_softmax(torch.zeros(2), dim=-1)
        logps = base.repeat(4, 1)
        with self.assertRaisesRegex(ValueError, "strict majority"):
            sampler.compose_pi_quorum_delta_log_probs(logps, base, 2)


class DeterminismAndPartitionTests(unittest.TestCase):
    def test_tuple_seed_is_stable_and_order_sensitive(self):
        first = sampler.tuple_seed(17, "qid", 2, "quorum", 3)
        self.assertEqual(first, sampler.tuple_seed(17, "qid", 2, "quorum", 3))
        self.assertNotEqual(first, sampler.tuple_seed("qid", 17, 2, "quorum", 3))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 63)

    def test_partition_is_sorted_contiguous_and_exact(self):
        records = [{"question_id": value} for value in ["d", "a", "e", "c", "b"]]
        chunks = [sampler.partition_records(records, index, 3) for index in range(3)]
        self.assertEqual(
            [[item["question_id"] for item in chunk] for chunk in chunks],
            [["a"], ["b", "c"], ["d", "e"]],
        )
        flattened = [item["question_id"] for chunk in chunks for item in chunk]
        self.assertEqual(flattened, ["a", "b", "c", "d", "e"])

    def test_chat_template_is_system_then_user_with_thinking_disabled(self):
        tokenizer = FakeTokenizer()
        observed = sampler.make_prompt_ids(
            tokenizer,
            {"question_id": "x", "system": "Follow tests.", "prompt": "Solve it."},
        )
        self.assertEqual(observed, [3, 4, 5])
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": "Follow tests."},
                {"role": "user", "content": "Solve it."},
            ],
        )
        self.assertEqual(
            tokenizer.kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )


class CacheTests(unittest.TestCase):
    def test_dynamic_and_legacy_cache_lengths(self):
        self.assertEqual(sampler.cache_sequence_length(FakeDynamicCache(13)), 13)
        key = torch.zeros((1, 2, 7, 3))
        self.assertEqual(sampler.cache_sequence_length(((key, key.clone()),)), 7)

    def test_prefill_once_then_one_token_steps_matches_full_prefix(self):
        prompt = [2, 5, 1]
        cached_model = FakeLegacyCacheModel()
        cached_state = sampler.prefill_cached_model(cached_model, prompt, "cpu")
        cached_tokens = []
        for step in range(5):
            token = int(torch.argmax(cached_state["next_logits"]).item())
            cached_tokens.append(token)
            if step < 4:
                cached_state = sampler.step_cached_model(
                    cached_model, token, cached_state["cache"], "cpu"
                )
        self.assertEqual(cached_model.call_lengths, [3, 1, 1, 1, 1])

        full_model = FakeLegacyCacheModel()
        full_tokens = []
        prefix = list(prompt)
        for _ in range(5):
            state = sampler.prefill_cached_model(full_model, prefix, "cpu")
            token = int(torch.argmax(state["next_logits"]).item())
            full_tokens.append(token)
            prefix.append(token)
        self.assertEqual(cached_tokens, full_tokens)
        self.assertEqual(full_model.call_lengths, [3, 4, 5, 6, 7])


class ResumeManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = self.temporary_directory.name
        self.output_dir = os.path.join(self.root, "output")
        self.training_config = os.path.join(self.root, "training.yaml")
        with open(self.training_config, "w", encoding="utf-8") as handle:
            handle.write(
                "base_model: organization/model\n"
                "base_model_revision: 0123456789abcdef0123456789abcdef01234567\n"
            )
        self.prompt_file = os.path.join(self.root, "prompts.json")
        self.records = [
            {"question_id": "q-a", "prompt": "A"},
            {"question_id": "q-b", "prompt": "B", "system": "S"},
        ]
        write_json(self.prompt_file, self.records)
        self.sampler_copy = os.path.join(self.root, "sampler.py")
        with open(self.sampler_copy, "w", encoding="utf-8") as handle:
            handle.write("fixed sampler version\n")
        self.references = []
        for index in range(4):
            directory = os.path.join(self.root, f"adapter-{index}")
            os.makedirs(directory)
            with open(os.path.join(directory, "adapter.bin"), "wb") as handle:
                handle.write(f"adapter {index}".encode("ascii"))
            self.references.append((f"ref-{index}", directory))
        self.manifest = self.make_manifest()
        self.fingerprint = sampler.ensure_manifest(self.output_dir, self.manifest)
        self.specs = sampler.expected_sample_specs(self.records, 1)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_manifest(self, seed=41):
        return sampler.build_immutable_manifest(
            training_config=self.training_config,
            prompt_file=self.prompt_file,
            records=self.records,
            references=self.references,
            method="quorum",
            q=3,
            n_samples=1,
            max_new_tokens=32,
            max_context=4096,
            temperature=0.0,
            seed=seed,
            chunk_index=0,
            chunk_count=1,
            device="cuda:0",
            sampler_path=self.sampler_copy,
        )

    def result_for(self, spec, response="answer"):
        payload = {
            "schema_version": sampler.SCHEMA_VERSION,
            "immutable_manifest_sha256": self.fingerprint,
            "question_id": spec["question_id"],
            "sample_index": spec["sample_index"],
            "prompt_sha256": spec["prompt_sha256"],
            "prompt": "prompt",
            "prompt_meta": {"question_id": spec["question_id"]},
            "system": "",
            "response": response,
            "response_token_ids": [4, 5],
            "selected_token_ids": [4, 5, 99],
            "n_response_tokens": 2,
            "n_selected_tokens": 3,
            "stop_reason": "eos",
            "rng_seed": 7,
        }
        payload["result_sha256"] = sampler.result_digest(payload)
        return payload

    def write_result(self, spec, payload=None):
        path = os.path.join(self.output_dir, "shards", spec["shard_name"])
        sampler.atomic_write_json(payload or self.result_for(spec), path)
        return path

    def test_manifest_mismatch_fails_before_resume(self):
        changed = self.make_manifest(seed=42)
        with self.assertRaisesRegex(ValueError, "Immutable manifest mismatch"):
            sampler.ensure_manifest(self.output_dir, changed)

    def test_adapter_tree_mutation_changes_manifest_fingerprint(self):
        before = self.manifest["references"][0]["tree_sha256"]
        with open(os.path.join(self.references[0][1], "adapter.bin"), "ab") as handle:
            handle.write(b"changed")
        changed = self.make_manifest()
        after = changed["references"][0]["tree_sha256"]
        self.assertNotEqual(before, after)

    def test_corrupted_and_semantically_modified_shards_fail(self):
        path = self.write_result(self.specs[0])
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            sampler.audit_result_shards(
                self.output_dir, self.specs, self.fingerprint
            )

        self.write_result(self.specs[0])
        payload = sampler.read_json_strict(path)
        payload["response"] = "tampered without updating checksum"
        write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            sampler.audit_result_shards(
                self.output_dir, self.specs, self.fingerprint
            )

    def test_interrupt_then_resume_audit_and_final_exact_ids(self):
        self.write_result(self.specs[0])
        completed, missing = sampler.audit_result_shards(
            self.output_dir, self.specs, self.fingerprint
        )
        self.assertEqual(list(completed), [self.specs[0]["shard_name"]])
        self.assertEqual([item["question_id"] for item in missing], ["q-b"])

        self.write_result(self.specs[1])
        completed, missing = sampler.audit_result_shards(
            self.output_dir,
            self.specs,
            self.fingerprint,
            require_complete=True,
        )
        self.assertEqual(len(completed), 2)
        self.assertEqual(missing, [])
        final_path = sampler.finalize_chunk(
            self.output_dir, self.manifest, self.fingerprint, self.specs
        )
        final = sampler.read_json_strict(final_path)
        self.assertEqual(
            [item["question_id"] for item in final["samples"]], ["q-a", "q-b"]
        )

    def test_unexpected_shard_fails_exact_id_audit(self):
        extra = os.path.join(self.output_dir, "shards", "old-run.json")
        write_json(extra, {"anything": True})
        with self.assertRaisesRegex(ValueError, "Unexpected result shards"):
            sampler.audit_result_shards(
                self.output_dir, self.specs, self.fingerprint
            )

    def test_pinned_revision_is_required(self):
        with open(self.training_config, "w", encoding="utf-8") as handle:
            handle.write("base_model: organization/model\nbase_model_revision: main\n")
        with self.assertRaisesRegex(ValueError, "immutable 40-character"):
            self.make_manifest()


if __name__ == "__main__":
    unittest.main()
