#!/usr/bin/env python3
"""No-network, no-GPU tests for the exploratory composition sampler."""

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import sample_massive_medical_union_composition_exploratory_v1 as sampler  # noqa: E402


class FakeMatcher:
    def __init__(self, apply=True):
        self.apply = apply

    def fill_next_token_bitmask(self, bitmask):
        self.bitmask_shape = tuple(bitmask.shape)
        return self.apply


class FakeCache:
    def __init__(self, prefix, adapter):
        self.prefix = list(prefix)
        self.adapter = adapter
        # Mimic a discoverable DynamicCache layout so the production probe can
        # prove that distinct cache objects also own distinct tensor storage.
        self.key_cache = [torch.tensor(self.prefix, dtype=torch.float32)]
        self.value_cache = [torch.tensor(self.prefix, dtype=torch.float32) + 1]

    def get_seq_length(self):
        return len(self.prefix)


class FakeSharedPeftModel:
    """Tiny shared base whose independent caches retain their selected adapter."""

    def __init__(self):
        self.adapter = None
        self.calls = []

    def set_adapter(self, adapter):
        self.adapter = adapter

    @contextlib.contextmanager
    def disable_adapter(self):
        previous = self.adapter
        self.adapter = None
        try:
            yield
        finally:
            self.adapter = previous

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del attention_mask, use_cache, return_dict
        previous = [] if past_key_values is None else past_key_values.prefix
        prefix = previous + input_ids[0].tolist()
        self.calls.append((self.adapter, tuple(prefix)))
        next_cache = FakeCache(prefix, self.adapter)
        # Generate 2, then 3, then EOS=9.  Deliberately BF16 to test the
        # float32-before-log-softmax inference contract.
        preferred = {2: 2, 3: 3}.get(len(prefix), 9)
        logits = torch.full(
            (1, input_ids.shape[1], 10), -8.0, dtype=torch.bfloat16
        )
        logits[0, -1, preferred] = 8.0
        return SimpleNamespace(logits=logits, past_key_values=next_cache)


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [7, 8]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(value) for value in token_ids)

    def encode(self, text, add_special_tokens=False):
        self.encoded = (text, add_special_tokens)
        return [4]


def tiny_profile(**overrides):
    result = {
        "temperature": 0.0,
        "max_new_tokens": 8,
        "max_context": 64,
        "n_samples": 1,
        "domain": "medical",
        "endpoint": "free_text",
        "role": "composition_confirmation",
        "prompt_file_sha256": "a" * 64,
        "sampling_profile": "official16_max1024_all_stop_v2",
    }
    result.update(overrides)
    return result


class CompositionMathTests(unittest.TestCase):
    def test_q3_and_q4_are_raw_order_statistics(self):
        logps = torch.tensor(
            [
                [-1.0, -8.0, -5.0],
                [-2.0, -6.0, -4.0],
                [-3.0, -7.0, -2.0],
                [-4.0, -5.0, -3.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.equal(
                sampler.compose_quorum_raw_scores(logps, 3),
                torch.tensor([-3.0, -7.0, -4.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                sampler.compose_quorum_raw_scores(logps, 4),
                torch.tensor([-4.0, -8.0, -5.0]),
            )
        )

    def test_delta_min_is_strict_unanimous_and_equality_falls_back(self):
        base = torch.tensor([-4.0, -4.0, -4.0, -4.0], dtype=torch.float32)
        shifts = torch.tensor(
            [
                [1.0, -1.0, 1.0, 0.0],
                [2.0, -2.0, -1.0, 1.0],
                [3.0, -3.0, 2.0, 2.0],
                [4.0, -4.0, -2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        observed = sampler.compose_delta_min_raw_scores(base + shifts, base)
        # all-up -> least positive; all-down -> least-magnitude negative;
        # mixed signs and one equality -> exact base fallback.
        expected = base + torch.tensor([1.0, -1.0, 0.0, 0.0])
        self.assertTrue(torch.equal(observed, expected))

    def test_methods_are_permutation_invariant(self):
        generator = torch.Generator().manual_seed(17)
        refs = torch.log_softmax(torch.randn(4, 13, generator=generator), dim=-1)
        base = torch.log_softmax(torch.randn(13, generator=generator), dim=-1)
        permutation = refs[[2, 0, 3, 1]]
        for q in (3, 4):
            self.assertTrue(
                torch.equal(
                    sampler.compose_quorum_raw_scores(refs, q),
                    sampler.compose_quorum_raw_scores(permutation, q),
                )
            )
        self.assertTrue(
            torch.equal(
                sampler.compose_delta_min_raw_scores(refs, base),
                sampler.compose_delta_min_raw_scores(permutation, base),
            )
        )


class MaskAndNormalizationTests(unittest.TestCase):
    def test_mask_receives_batched_raw_scores_before_sole_normalization(self):
        raw = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float32)
        seen = {}

        def apply_mask(logits, bitmask):
            self.assertEqual(tuple(logits.shape), (1, 3))
            self.assertEqual(tuple(bitmask.shape), (1, 1))
            seen["raw"] = logits.clone()
            logits[0, 0] = -torch.inf

        runtime = {
            "matcher": FakeMatcher(apply=True),
            "bitmask": torch.zeros((1, 1), dtype=torch.int32),
            "apply_token_bitmask_inplace": apply_mask,
        }
        observed = sampler.apply_grammar_mask_then_normalize(raw, runtime)
        self.assertTrue(torch.equal(seen["raw"], raw.unsqueeze(0)))
        self.assertTrue(torch.equal(raw, torch.tensor([4.0, 2.0, 1.0])))
        expected = torch.log_softmax(torch.tensor([-torch.inf, 2.0, 1.0]), dim=-1)
        self.assertTrue(torch.allclose(observed, expected))

    def test_no_apply_still_normalizes_once(self):
        raw = torch.tensor([-2.0, -4.0], dtype=torch.float32)
        runtime = {
            "matcher": FakeMatcher(apply=False),
            "bitmask": torch.zeros((1, 1), dtype=torch.int32),
            "apply_token_bitmask_inplace": mock.Mock(side_effect=AssertionError),
        }
        self.assertTrue(
            torch.allclose(
                sampler.apply_grammar_mask_then_normalize(raw, runtime),
                torch.log_softmax(raw, dim=-1),
            )
        )

    def test_cpu_matcher_mask_is_transferred_to_the_logits_device_for_apply(self):
        transferred = torch.zeros((1, 1), dtype=torch.int32)

        class CpuMask:
            def __init__(self):
                self.to_calls = []
                self.shape = (1, 1)

            def to(self, device):
                self.to_calls.append(device)
                return transferred

        mask = CpuMask()
        matcher = FakeMatcher(apply=True)

        def apply_mask(logits, observed_mask):
            self.assertIs(observed_mask, transferred)
            self.assertEqual(logits.device, observed_mask.device)

        raw = torch.tensor([0.0, -1.0], dtype=torch.float32)
        sampler.apply_grammar_mask_then_normalize(
            raw,
            {
                "matcher": matcher,
                "bitmask": mask,
                "apply_token_bitmask_inplace": apply_mask,
            },
        )
        self.assertEqual(mask.to_calls, [raw.device])
        self.assertIs(matcher.apply, True)


class SharedCacheAndGenerationTests(unittest.TestCase):
    @staticmethod
    def probe_record():
        return {
            "question_id": "smoke_intent_00",
            "prompt": "Classify this request.",
            "prompt_sha256": "b" * 64,
        }

    def test_live_probe_switches_every_adapter_and_disables_base(self):
        model = FakeSharedPeftModel()
        tokenizer = FakeTokenizer()
        probe = sampler.run_cache_equivalence_probe(
            model, tokenizer, self.probe_record(), "smoke", "cpu"
        )
        roles = [*sampler.PANEL_ORDER, None]
        # Five independent prefills, then cached/fresh pairs in the same order.
        self.assertEqual(
            [adapter for adapter, _ in model.calls],
            [*roles, *(adapter for role in roles for adapter in (role, role))],
        )
        self.assertEqual(probe["roles"], [*sampler.PANEL_ORDER, "base"])
        self.assertTrue(probe["cache_objects_unique"])
        self.assertTrue(probe["cache_tensor_storage_sets_checked"])
        self.assertTrue(probe["cache_tensor_storages_disjoint"])
        self.assertEqual(
            tokenizer.encoded,
            (sampler.CACHE_EQUIVALENCE_CONTINUATION_TEXT, False),
        )

    def test_live_probe_uses_one_exact_prefix_and_continuation_token(self):
        model = FakeSharedPeftModel()
        tokenizer = FakeTokenizer()
        probe = sampler.run_cache_equivalence_probe(
            model, tokenizer, self.probe_record(), "confirmation", "cpu"
        )
        prompt, full = (7, 8), (7, 8, 4)
        self.assertEqual([prefix for _, prefix in model.calls[:5]], [prompt] * 5)
        self.assertEqual([prefix for _, prefix in model.calls[5:]], [full] * 10)
        self.assertEqual(probe["continuation_token_id"], 4)
        self.assertEqual(probe["prompt_tokens"], 2)
        self.assertEqual(
            probe["prompt_token_ids_sha256"],
            sampler.sha256_bytes(sampler.canonical_bytes([7, 8])),
        )
        self.assertTrue(probe["same_prefix_and_token_all_roles"])
        self.assertTrue(all(
            row["allclose"] for row in probe["comparisons"].values()
        ))

    def test_live_probe_rejects_cached_vs_full_prefix_logit_mismatch(self):
        class MismatchModel(FakeSharedPeftModel):
            def __call__(self, **kwargs):
                cached = kwargs.get("past_key_values") is not None
                result = super().__call__(**kwargs)
                if cached:
                    result.logits[0, -1, 0] += 1.0
                return result

        with self.assertRaisesRegex(ValueError, "next logits mismatch for A"):
            sampler.run_cache_equivalence_probe(
                MismatchModel(),
                FakeTokenizer(),
                self.probe_record(),
                "smoke",
                "cpu",
            )

    def test_all_adapters_and_base_advance_on_the_same_selected_prefix(self):
        model = FakeSharedPeftModel()
        tokenizer = FakeTokenizer()
        record = {
            "question_id": "medical_official16_00",
            "prompt": "Question",
            "prompt_sha256": "b" * 64,
        }
        sample = sampler.generate_sample(
            record=record,
            sample_index=0,
            prompt_ids=[7, 8],
            model=model,
            tokenizer=tokenizer,
            method=sampler.method_by_id("delta_min_m4_q4"),
            profile=tiny_profile(),
            device="cpu",
            stop_ids={9},
        )
        self.assertEqual(sample["response"], "2 3")
        self.assertEqual(sample["finish_reason"], "stop")
        expected_prefixes = [(7, 8), (7, 8, 2), (7, 8, 2, 3)]
        for adapter in (*sampler.PANEL_ORDER, None):
            observed = [prefix for role, prefix in model.calls if role == adapter]
            self.assertEqual(observed, expected_prefixes)

    def test_log_softmax_is_computed_after_float32_cast(self):
        model = FakeSharedPeftModel()
        captured = []
        original = sampler.compose_raw_scores

        def capture(reference_logps, base_logp, method):
            captured.append((reference_logps.dtype, base_logp))
            return original(reference_logps, base_logp, method)

        with mock.patch.object(sampler, "compose_raw_scores", side_effect=capture):
            sampler.generate_sample(
                record={
                    "question_id": "q",
                    "prompt": "p",
                    "prompt_sha256": "c" * 64,
                },
                sample_index=0,
                prompt_ids=[7, 8],
                model=model,
                tokenizer=FakeTokenizer(),
                method=sampler.method_by_id("ordinary_min_m4_q4"),
                profile=tiny_profile(),
                device="cpu",
                stop_ids={9},
            )
        self.assertTrue(captured)
        self.assertTrue(all(dtype == torch.float32 for dtype, _ in captured))

    def test_chat_template_is_frozen_and_thinking_is_disabled(self):
        tokenizer = FakeTokenizer()
        observed = sampler.make_prompt_ids(
            tokenizer, {"system": " Rules ", "prompt": "Question"}
        )
        self.assertEqual(observed, [7, 8])
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Question"},
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


class DeterminismAndPlanTests(unittest.TestCase):
    def test_tuple_seed_binds_method_question_and_sample(self):
        seed = sampler.tuple_seed(8172026, "ordinary_min_m4_q4", "q", 3)
        self.assertEqual(
            seed,
            sampler.tuple_seed(8172026, "ordinary_min_m4_q4", "q", 3),
        )
        self.assertNotEqual(
            seed,
            sampler.tuple_seed(8172026, "ordinary_quorum_m4_q3", "q", 3),
        )
        self.assertNotEqual(
            seed,
            sampler.tuple_seed(8172026, "ordinary_min_m4_q4", "q", 4),
        )

    def test_stream_plan_has_fresh_base_and_all_methods_in_frozen_order(self):
        profile, records = tiny_profile(domain="massive"), [{"question_id": "q"}]
        smoke = sampler.stream_plan("smoke", profile, records)
        self.assertEqual(
            [item[0]["method_id"] for item in smoke],
            ["pi_base", *(item["method_id"] for item in sampler.METHODS)],
        )
        confirmation = sampler.stream_plan(
            "confirmation", profile, records, tiny_profile(), records
        )
        self.assertEqual(len(confirmation), 7)
        self.assertEqual(
            [item[0]["method_id"] for item in confirmation[4:]],
            [item["method_id"] for item in sampler.METHODS],
        )

    def test_generation_meta_binds_schema_panel_and_same_backend_base(self):
        base = {
            "model_name": "pi_base",
            "model_path": "BASE",
            "model_fingerprint": "BASE",
            "base_model": sampler.BASE_MODEL,
            "base_model_revision": sampler.BASE_REVISION,
        }
        references = {name: {"model_name": name} for name in (
            "pi_A", "pi_B1", "pi_B2", "pi_B3"
        )}
        protocol = {
            "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "body": {
                "model_panel": {
                    "panel_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
                    "base": base,
                    "references": references,
                }
            },
        }
        profile = tiny_profile(
            domain="massive",
            endpoint="joint_json",
            role="training_disjoint_composition_smoke",
            intent_labels=["intent"],
            slot_labels=["slot"],
        )
        records = [{"question_id": "q", "prompt_sha256": "3" * 64}]
        base_meta = sampler.stream_meta(
            protocol, "smoke", sampler.method_by_id("pi_base"), profile, records
        )
        method_meta = sampler.stream_meta(
            protocol,
            "smoke",
            sampler.method_by_id("ordinary_quorum_m4_q3"),
            profile,
            records,
        )
        self.assertEqual(base_meta["model_panel_binding"], base)
        self.assertEqual(
            method_meta["model_panel_binding"],
            {
                "panel_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
                "references": references,
            },
        )
        self.assertTrue(base_meta["is_paired_base"])
        self.assertFalse(method_meta["is_paired_base"])
        self.assertTrue(method_meta["same_transformers_backend_as_paired_base"])
        self.assertEqual(
            method_meta["generation_config"]["structured_constraint_profile"],
            sampler.STRUCTURED_PROFILE,
        )
        self.assertFalse(
            method_meta["generation_config"]["xgrammar_any_whitespace"]
        )
        self.assertIs(
            method_meta["generation_config"]["structured_fallback_allowed"],
            False,
        )


class SealAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = self.temporary.name
        self.meta = {
            "protocol": sampler.GENERATION_PROTOCOL,
            "method_id": "pi_base",
        }
        self.records = [
            {"question_id": "q0", "prompt_sha256": "0" * 64},
            {"question_id": "q1", "prompt_sha256": "1" * 64},
        ]
        self.specs = sampler.expected_sample_specs(self.records, 1)

    def tearDown(self):
        self.temporary.cleanup()

    def write_shard(self, manifest, spec, response):
        sample = {
            "question_id": spec["question_id"],
            "sample_index": spec["sample_index"],
            "prompt_sha256": spec["prompt_sha256"],
            "response": response,
            "finish_reason": "stop",
            "generated_tokens": 1,
            "response_sha256": sampler.sha256_bytes(response.encode()),
            "rng_seed": 7,
        }
        sample["sample_sha256"] = sampler.sample_sha256(sample)
        payload = sampler.seal(
            {
                "stream_payload_sha256": manifest[sampler.OUTPUT_SEAL_FIELD],
                "spec": spec,
                "sample": sample,
                "generation_seconds": 0.25,
            }
        )
        sampler.atomic_write_json(
            os.path.join(self.root, "shards", spec["shard_name"]), payload
        )

    def test_resume_assembles_exact_sealed_output_and_detects_corruption(self):
        os.makedirs(self.root, exist_ok=True)
        manifest, samples, _, missing = sampler.assemble_stream(
            self.root, self.meta, self.specs, require_final=False
        )
        self.assertFalse(samples)
        self.assertEqual(missing, self.specs[0])
        self.write_shard(manifest, self.specs[0], "a")
        _, samples, _, missing = sampler.assemble_stream(
            self.root, self.meta, self.specs, require_final=False
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(missing, self.specs[1])
        self.write_shard(manifest, self.specs[1], "b")
        _, samples, seconds, missing = sampler.assemble_stream(
            self.root, self.meta, self.specs, require_final=False
        )
        self.assertIsNone(missing)
        self.assertEqual(len(samples), 2)
        self.assertEqual(seconds, [0.25, 0.25])
        final, _ = sampler.load_json_regular(
            os.path.join(self.root, "generation.json"), "final"
        )
        sampler.verify_seal(final, sampler.OUTPUT_SEAL_FIELD, "final")
        self.assertEqual(set(final), {"meta", "samples", "payload_sha256"})

        shard_path = os.path.join(
            self.root, "shards", self.specs[0]["shard_name"]
        )
        shard, _ = sampler.load_json_regular(shard_path, "shard")
        shard["sample"]["response"] = "corrupt"
        sampler.atomic_write_json(shard_path, shard)
        with self.assertRaisesRegex(ValueError, "invalid payload_sha256 seal"):
            sampler.assemble_stream(
                self.root, self.meta, self.specs, require_final=True
            )

    def test_extra_shard_is_rejected(self):
        os.makedirs(os.path.join(self.root, "shards"), exist_ok=True)
        with open(os.path.join(self.root, "shards", "extra.json"), "w") as handle:
            handle.write("{}")
        with self.assertRaisesRegex(ValueError, "unexpected entries"):
            sampler.assemble_stream(
                self.root, self.meta, self.specs, require_final=False
            )

    def test_noncontiguous_resume_is_rejected_without_overwrite(self):
        manifest, _, _, _ = sampler.assemble_stream(
            self.root, self.meta, self.specs, require_final=False
        )
        self.write_shard(manifest, self.specs[1], "later")
        later = os.path.join(self.root, "shards", self.specs[1]["shard_name"])
        before = sampler.sha256_file(later)
        with self.assertRaisesRegex(ValueError, "not an exact contiguous prefix"):
            sampler.assemble_stream(
                self.root, self.meta, self.specs, require_final=False
            )
        self.assertEqual(sampler.sha256_file(later), before)

    def test_sample_and_container_seals_bind_every_field(self):
        sample = {"question_id": "q", "response": "x"}
        sample["sample_sha256"] = sampler.sample_sha256(sample)
        self.assertEqual(sample["sample_sha256"], sampler.sample_sha256(sample))
        sample["response"] = "y"
        self.assertNotEqual(sample["sample_sha256"], sampler.sample_sha256(sample))
        payload = sampler.seal({"meta": self.meta, "samples": []})
        sampler.verify_seal(payload, sampler.OUTPUT_SEAL_FIELD, "output")
        payload["samples"].append({})
        with self.assertRaisesRegex(ValueError, "invalid payload_sha256 seal"):
            sampler.verify_seal(payload, sampler.OUTPUT_SEAL_FIELD, "output")

    def test_smoke_timing_has_four_separate_streams_and_projection_inputs(self):
        formula = "frozen formula"
        protocol = {
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "body": {"runtime_projection": {"formula": formula}},
        }
        names = ["pi_base", *(item["method_id"] for item in sampler.METHODS)]
        streams = [
            {
                "method_id": name,
                "domain": "massive",
                "stream_root": "/not-sealed",
                "generation_path": "/not-sealed/generation.json",
                "samples": 60,
                "generated_tokens": 120 + index,
                "generation_seconds": 12.0 + index,
                "selected_tokens_per_second": (120 + index) / (12.0 + index),
            }
            for index, name in enumerate(names)
        ]
        probe = sampler.run_cache_equivalence_probe(
            FakeSharedPeftModel(),
            FakeTokenizer(),
            SharedCacheAndGenerationTests.probe_record(),
            "smoke",
            "cpu",
        )
        body = sampler.build_timing_record(
            protocol, "smoke", 3.0, streams, probe
        )
        self.assertEqual(
            tuple(body["streams"]), tuple(f"{name}:massive" for name in names)
        )
        self.assertEqual(body["projection_formula"], formula)
        self.assertEqual(body["smoke_generation_total_multiplier"], 40)
        self.assertEqual(body["cache_equivalence_probe"], probe)
        self.assertIsNone(body["smoke_score_and_seal_seconds"])
        self.assertIsNone(body["projected_confirmation_seconds"])


class CpuPreflightTests(unittest.TestCase):
    def test_recorded_hybrid_and_whitespace_probe_contract_is_frozen(self):
        self.assertEqual(
            sampler.RECORDED_LEGACY_HYBRID_INTENT_PROBES,
            (
                "alarm_addcontact",
                "alarm_createoradd",
                "calendar_recipe",
                "cooking_remove",
            ),
        )
        self.assertEqual(
            sampler.RECORDED_LEGACY_HYBRID_SLOT_PROBES,
            ("alarm_name", "app_type", "cooking_name"),
        )
        with open(sampler.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("for count in (1, 256)", source)
        self.assertIn("Grammar.from_json_schema(schema_json, any_whitespace=False)", source)
        self.assertIn("compile_json_schema(\n        schema_json, any_whitespace=True", source)
        auto_config_call = source.split("model_config = AutoConfig.from_pretrained(", 1)[1]
        auto_config_call = auto_config_call.split("\n    )", 1)[0]
        self.assertIn("revision=BASE_REVISION", auto_config_call)
        self.assertIn("local_files_only=True", auto_config_call)

    def test_preflight_audits_runtime_and_grammar_without_loading_weights(self):
        args = SimpleNamespace(
            protocol_manifest="/protocol/manifest.json",
            output_root="/unused",
            phase="smoke",
            device="cuda:0",
            preflight_only=True,
            audit_only=False,
        )
        protocol = {"root": "/protocol"}
        profile = tiny_profile(
            domain="massive",
            intent_labels=["intent"],
            slot_labels=["slot"],
        )
        records = [{"question_id": "q", "prompt_sha256": "a" * 64}]
        grammar = {
            "schema_sha256": "f" * 64,
            "intent_leaves_checked": 1,
            "slot_leaves_checked": 1,
            "invalid_probes_rejected": 9,
            "recorded_hybrid_intent_probes_rejected": 4,
            "recorded_hybrid_slot_probes_rejected": 3,
            "flexible_whitespace_probes_reproduced": 2,
            "whitespace_probes_rejected": 2,
        }
        with (
            mock.patch.object(sampler, "force_offline_environment"),
            mock.patch.object(sampler, "load_protocol_manifest", return_value=protocol),
            mock.patch.object(
                sampler, "load_massive_prompts", return_value=(profile, records)
            ),
            mock.patch.object(
                sampler,
                "require_pinned_runtime",
                return_value={"xgrammar": sampler.PINNED_XGRAMMAR_VERSION},
            ) as runtime,
            mock.patch.object(
                sampler,
                "load_tokenizer_and_grammar",
                return_value=(object(), object(), grammar),
            ),
            mock.patch.object(sampler, "load_shared_peft_model") as load_weights,
        ):
            self.assertEqual(sampler.run_phase(args), 0)
        runtime.assert_called_once_with(require_cuda=False)
        load_weights.assert_not_called()

    @unittest.skipUnless(
        importlib.util.find_spec("xgrammar") is not None,
        "xgrammar is absent from this CPU test environment",
    )
    def test_pinned_xgrammar_direct_loop_if_exact_version_is_installed(self):
        try:
            version = sampler.importlib.metadata.version("xgrammar")
        except sampler.importlib.metadata.PackageNotFoundError:
            self.skipTest("xgrammar metadata is absent")
        if version != sampler.PINNED_XGRAMMAR_VERSION:
            self.skipTest("installed XGrammar is not the workflow-pinned version")
        # The full ontology contract is exercised by compile_and_audit_xgrammar;
        # this focused shape test independently guards the pinned 2-D API.
        import xgrammar as xgr

        bitmask = xgr.allocate_token_bitmask(1, 64)
        self.assertEqual(tuple(bitmask.shape), (1, 2))
        logits = torch.zeros((1, 64), dtype=torch.float32)
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        self.assertEqual(tuple(logits.shape), (1, 64))


if __name__ == "__main__":
    unittest.main()
