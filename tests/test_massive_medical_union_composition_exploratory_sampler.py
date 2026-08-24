#!/usr/bin/env python3
"""No-network, no-GPU tests for the exploratory composition sampler."""

import contextlib
import io
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
    def __init__(self, prefix, role):
        self.prefix = list(prefix)
        self.role = role
        # Mimic a discoverable DynamicCache layout so the production probe can
        # prove that distinct cache objects also own distinct tensor storage.
        self.key_cache = [torch.tensor(self.prefix, dtype=torch.float32)]
        self.value_cache = [torch.tensor(self.prefix, dtype=torch.float32) + 1]

    def get_seq_length(self):
        return len(self.prefix)


class FakeIndependentModel:
    """Tiny immutable-role model with independently owned parameters and caches."""

    def __init__(self, role, trace=None):
        self.role = role
        self.calls = []
        self.trace = trace if trace is not None else []
        self.config = SimpleNamespace(
            _attn_implementation="sdpa", use_cache=True, vocab_size=10
        )
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.training = False
        self.peft_config = None if role == "base" else {role: object()}
        self.active_adapters = [] if role == "base" else [role]
        self.weight = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
        )
        self._input_embeddings = SimpleNamespace(
            weight=self.weight
        )

    def get_input_embeddings(self):
        return self._input_embeddings

    def named_parameters(self):
        yield "weight", self.weight

    def eval(self):
        self.training = False
        return self

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
        self.calls.append(tuple(prefix))
        self.trace.append((self.role, tuple(prefix)))
        next_cache = FakeCache(prefix, self.role)
        # Generate 2, then 3, then EOS=9.  Deliberately BF16 to test the
        # float32-before-log-softmax inference contract.
        preferred = {2: 2, 3: 3}.get(len(prefix), 9)
        logits = torch.full(
            (1, input_ids.shape[1], 10), -8.0, dtype=torch.bfloat16
        )
        logits[0, -1, preferred] = 8.0
        return SimpleNamespace(logits=logits, past_key_values=next_cache)


def fake_independent_models(model_type=FakeIndependentModel):
    trace = []
    models = {
        role: model_type(role, trace) for role in sampler.INDEPENDENT_MODEL_ORDER
    }
    return models, trace


def fake_memory_reader(device):
    gib = 1024**3
    return {
        "device": str(device),
        "device_name": "fake-H200",
        "total_memory_bytes": 140 * gib,
        "free_memory_bytes": 48 * gib,
        "allocated_memory_bytes": 80 * gib,
        "reserved_memory_bytes": 88 * gib,
        "peak_allocated_memory_bytes": 82 * gib,
    }


def production_device_probe(probe):
    """Translate synthetic CPU tensor evidence into the frozen artifact device."""
    result = json.loads(json.dumps(probe))
    result["device"] = "cuda:0"
    result["gpu_memory"]["device"] = "cuda:0"
    for role in sampler.INDEPENDENT_MODEL_ORDER:
        result["model_isolation"][role]["parameter_devices"] = ["cuda:0"]
        result["cache_execution"][role]["cache_tensor_devices"] = ["cuda:0"]
    return result


def run_synthetic_probe(models, tokenizer, record, phase, memory_reader):
    """Exercise CPU tensors, then validate the exact production artifact schema."""
    with mock.patch.object(
        sampler,
        "audit_cache_equivalence_probe",
        side_effect=lambda probe, observed_phase: probe,
    ):
        probe = sampler.run_cache_equivalence_probe(
            models,
            tokenizer,
            record,
            phase,
            "cpu",
            memory_reader=memory_reader,
        )
    result = production_device_probe(probe)
    sampler.audit_cache_equivalence_probe(result, phase)
    return result


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


class TinyPinnedSnapshot:
    """Small Hugging Face cache with the same pinned snapshot/link topology."""

    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.hub = os.path.join(self.temporary.name, "hub")
        self.model_cache = os.path.join(self.hub, sampler.BASE_CACHE_DIRECTORY)
        self.blobs = os.path.join(self.model_cache, "blobs")
        self.snapshot = os.path.join(
            self.model_cache, "snapshots", sampler.BASE_REVISION
        )
        os.makedirs(self.blobs)
        os.makedirs(self.snapshot)
        self.runtime_payloads = {
            "config.json": b'{"model_type":"qwen2"}',
            "generation_config.json": b'{"eos_token_id":1}',
            "tokenizer_config.json": b'{"chat_template":"tiny"}',
            "tokenizer.json": b'{"version":"1.0"}',
            "vocab.json": b'{"token":0}',
            "merges.txt": b"#version: 0.2\n",
        }
        self.runtime = tuple(
            (name, len(raw), sampler.sha256_bytes(raw))
            for name, raw in self.runtime_payloads.items()
        )
        self.shard_payloads = [f"tiny-shard-{index}".encode() for index in range(4)]
        self.shards = tuple(
            (
                f"model-{index + 1:05d}-of-00004.safetensors",
                len(payload),
                sampler.sha256_bytes(payload),
            )
            for index, payload in enumerate(self.shard_payloads)
        )
        index_payload = {
            "metadata": {"total_size": sum(len(item) for item in self.shard_payloads)},
            "weight_map": {
                f"tensor.{index}": artifact[0]
                for index, artifact in enumerate(self.shards)
            },
        }
        self.index_raw = sampler.canonical_bytes(index_payload)
        self.index = (
            "model.safetensors.index.json",
            len(self.index_raw),
            sampler.sha256_bytes(self.index_raw),
        )
        for name, raw in self.runtime_payloads.items():
            self._install(name, raw)
        self._install(self.index[0], self.index_raw)
        for artifact, payload in zip(self.shards, self.shard_payloads):
            self._install(artifact[0], payload)

    def cleanup(self):
        self.temporary.cleanup()

    def _install(self, name, raw):
        blob = os.path.join(self.blobs, sampler.sha256_bytes(raw))
        with open(blob, "wb") as handle:
            handle.write(raw)
        os.symlink(os.path.relpath(blob, self.snapshot), os.path.join(self.snapshot, name))

    @contextlib.contextmanager
    def patched(self, transformers_cache=None, index=None):
        environment = {
            "HUGGINGFACE_HUB_CACHE": self.hub,
            "TRANSFORMERS_CACHE": (
                self.hub if transformers_cache is None else transformers_cache
            ),
        }
        with (
            mock.patch.object(
                sampler, "BASE_SAFETENSORS_INDEX", self.index if index is None else index
            ),
            mock.patch.object(sampler, "BASE_RUNTIME_ARTIFACTS", self.runtime),
            mock.patch.object(sampler, "BASE_SAFETENSORS_SHARDS", self.shards),
            mock.patch.object(
                sampler, "BASE_SAFETENSORS_INDEX_ENTRIES", len(self.shards)
            ),
            mock.patch.object(
                sampler,
                "BASE_INDEXED_WEIGHT_BYTES",
                sum(len(item) for item in self.shard_payloads),
            ),
            mock.patch.dict(os.environ, environment, clear=False),
        ):
            yield

    def resolve(self):
        with self.patched():
            return sampler.resolve_pinned_base_snapshot()


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


class IndependentModelAndGenerationTests(unittest.TestCase):
    @staticmethod
    def probe_record():
        return {
            "question_id": "smoke_intent_00",
            "prompt": "Classify this request.",
            "prompt_sha256": "b" * 64,
        }

    def run_probe(self, models=None, phase="smoke", memory_reader=fake_memory_reader):
        if models is None:
            models, _ = fake_independent_models()
        probe = run_synthetic_probe(
            models,
            FakeTokenizer(),
            self.probe_record(),
            phase,
            memory_reader,
        )
        return probe

    def test_live_probe_uses_five_independent_models_without_switching(self):
        models, trace = fake_independent_models()
        tokenizer = FakeTokenizer()
        probe = run_synthetic_probe(
            models,
            tokenizer,
            self.probe_record(),
            "smoke",
            fake_memory_reader,
        )
        roles = list(sampler.INDEPENDENT_MODEL_ORDER)
        self.assertEqual(
            [role for role, _ in trace], [*roles, *roles, *roles]
        )
        self.assertEqual(probe["roles"], roles)
        self.assertTrue(probe["model_objects_unique"])
        self.assertEqual(probe["model_object_count"], 5)
        self.assertTrue(probe["single_active_adapter_per_reference"])
        self.assertFalse(probe["scientific_adapter_switching_used"])
        self.assertTrue(probe["parameter_storages_disjoint"])
        self.assertTrue(probe["cache_objects_unique"])
        self.assertEqual(probe["cache_object_count"], 5)
        self.assertTrue(probe["cache_tensor_storage_sets_checked"])
        self.assertTrue(probe["cache_tensor_storages_disjoint"])
        self.assertEqual(
            probe["protocol"], sampler.CACHE_EQUIVALENCE_PROBE_PROTOCOL
        )
        self.assertEqual(
            tokenizer.encoded,
            (sampler.CACHE_EQUIVALENCE_CONTINUATION_TEXT, False),
        )

    def test_live_probe_uses_one_exact_prefix_and_continuation_token(self):
        models, trace = fake_independent_models()
        tokenizer = FakeTokenizer()
        probe = run_synthetic_probe(
            models,
            tokenizer,
            self.probe_record(),
            "confirmation",
            fake_memory_reader,
        )
        prompt, full = (7, 8), (7, 8, 4)
        self.assertEqual(
            [prefix for _, prefix in trace], [prompt] * 5 + [full] * 10
        )
        self.assertEqual(probe["continuation_token_id"], 4)
        self.assertEqual(probe["prompt_tokens"], 2)
        self.assertEqual(
            probe["prompt_token_ids_sha256"],
            sampler.sha256_bytes(sampler.canonical_bytes([7, 8])),
        )
        for role, evidence in probe["model_isolation"].items():
            self.assertEqual(
                evidence["active_adapters"], [] if role == "base" else [role]
            )
        for evidence in probe["cache_execution"].values():
            self.assertEqual(evidence["prefill_cache_length"], 2)
            self.assertEqual(evidence["stepped_cache_length"], 3)
            self.assertTrue(evidence["next_logits_finite"])

    def test_cached_vs_full_prefix_mismatch_is_diagnostic_not_gate(self):
        class MismatchModel(FakeIndependentModel):
            def __call__(self, **kwargs):
                cached = kwargs.get("past_key_values") is not None
                result = super().__call__(**kwargs)
                if cached:
                    result.logits[0, -1, 0] += 1.0
                return result

        models, _ = fake_independent_models(MismatchModel)
        probe = self.run_probe(models)
        self.assertEqual(probe["result"], "PASS")
        self.assertFalse(
            probe["diagnostic_policy"][
                "cached_vs_fresh_full_prefix_is_hard_gate"
            ]
        )
        for diagnostic in probe["cached_vs_full_prefix_diagnostics"].values():
            self.assertGreater(diagnostic["raw_max_abs_diff"], 0.0)
            self.assertFalse(diagnostic["legacy_allclose_1e3"])

    def test_live_probe_rejects_model_object_and_parameter_storage_aliases(self):
        models, _ = fake_independent_models()
        models["B1"] = models["A"]
        with self.assertRaisesRegex(ValueError, "shares a model object"):
            self.run_probe(models)

        models, _ = fake_independent_models()
        models["B1"].weight = models["A"].weight
        models["B1"]._input_embeddings.weight = models["A"].weight
        with self.assertRaisesRegex(ValueError, "share parameter storage"):
            self.run_probe(models)

    def test_live_probe_rejects_wrong_single_adapter_and_backend(self):
        models, _ = fake_independent_models()
        models["B2"].active_adapters = ["A"]
        with self.assertRaisesRegex(ValueError, "adapter identity differs for B2"):
            self.run_probe(models)

        models, _ = fake_independent_models()
        models["B3"].config._attn_implementation = "eager"
        with self.assertRaisesRegex(ValueError, "frozen eval BF16/SDPA"):
            self.run_probe(models)

    def test_plain_transformers_base_never_calls_peft_active_adapter_accessor(self):
        models, _ = fake_independent_models()
        base = models["base"]
        del base.active_adapters
        base._hf_peft_config_loaded = False

        def no_adapter_loaded():
            raise ValueError("No adapter loaded. Please load an adapter first.")

        base.active_adapters = no_adapter_loaded
        probe = self.run_probe(models)
        self.assertEqual(probe["model_isolation"]["base"]["active_adapters"], [])

        models, _ = fake_independent_models()
        models["base"]._hf_peft_config_loaded = True
        with self.assertRaisesRegex(ValueError, "direct base unexpectedly"):
            self.run_probe(models)

    def test_live_probe_rejects_cache_storage_alias_and_nonfinite_logits(self):
        models, _ = fake_independent_models()
        original = sampler.run_independent_cached_probe

        def alias_cache(*args, **kwargs):
            states = original(*args, **kwargs)
            states["B1"]["cache"] = states["A"]["cache"]
            return states

        with mock.patch.object(
            sampler, "run_independent_cached_probe", side_effect=alias_cache
        ):
            with self.assertRaisesRegex(ValueError, "shared KV-cache object"):
                self.run_probe(models)

        class NonfiniteModel(FakeIndependentModel):
            def __call__(self, **kwargs):
                result = super().__call__(**kwargs)
                if kwargs.get("past_key_values") is not None and self.role == "A":
                    result.logits[0, -1, 0] = float("nan")
                return result

        models, _ = fake_independent_models(NonfiniteModel)
        with self.assertRaisesRegex(ValueError, "cached logits are invalid for A"):
            self.run_probe(models)

    def test_live_probe_rejects_insufficient_prospective_gpu_headroom(self):
        def low_memory(device):
            result = fake_memory_reader(device)
            result["free_memory_bytes"] = 31 * 1024**3
            return result

        with self.assertRaisesRegex(ValueError, "memory/headroom contract failed"):
            self.run_probe(memory_reader=low_memory)

    def test_probe_schema_rejects_hard_gate_tamper_but_allows_diagnostic_false(self):
        probe = self.run_probe()
        tampered = json.loads(json.dumps(probe))
        tampered["model_isolation"]["A"][
            "parameter_storage_disjoint_from_other_models"
        ] = False
        with self.assertRaisesRegex(ValueError, "model isolation differs for A"):
            sampler.audit_cache_equivalence_probe(tampered, "smoke")

        tampered = json.loads(json.dumps(probe))
        tampered["probe_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            sampler.audit_cache_equivalence_probe(tampered, "smoke")

        tampered = json.loads(json.dumps(probe))
        tampered["device"] = "cpu"
        tampered["gpu_memory"]["device"] = "cpu"
        for role in sampler.INDEPENDENT_MODEL_ORDER:
            tampered["model_isolation"][role]["parameter_devices"] = ["cpu"]
            tampered["cache_execution"][role]["cache_tensor_devices"] = ["cpu"]
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            sampler.audit_cache_equivalence_probe(tampered, "smoke")

        diagnostic_only = json.loads(json.dumps(probe))
        diagnostic_only["cached_vs_full_prefix_diagnostics"]["A"][
            "legacy_allclose_1e3"
        ] = False
        sampler.audit_cache_equivalence_probe(diagnostic_only, "smoke")

    def test_probe_static_contract_seals_exact_v3_schema(self):
        contract = sampler.cache_equivalence_probe_static_contract()
        body = {
            key: value for key, value in contract.items() if key != "contract_sha256"
        }
        self.assertEqual(
            contract["contract_sha256"],
            sampler.sha256_bytes(sampler.canonical_bytes(body)),
        )
        self.assertEqual(
            contract["protocol"], sampler.CACHE_EQUIVALENCE_PROBE_PROTOCOL
        )
        self.assertEqual(contract["roles"], ["A", "B1", "B2", "B3", "base"])
        self.assertEqual(contract["model_object_count"], 5)
        self.assertEqual(contract["cache_object_count"], 5)
        self.assertFalse(
            contract["hard_gate"][
                "cached_next_logits_bitwise_repeatability_required"
            ]
        )
        self.assertFalse(
            contract["gpu_memory_contract"][
                "prior_incident_values_used_as_threshold"
            ]
        )
        self.assertFalse(
            contract["diagnostic_policy"][
                "cached_vs_fresh_full_prefix_is_hard_gate"
            ]
        )

    def test_setup_resume_ignores_only_non_gating_diagnostic_drift(self):
        protocol = {"file_sha256": "a" * 64, "payload_sha256": "b" * 64}
        first = self.run_probe()
        second = json.loads(json.dumps(first))
        second["probe_seconds"] += 1.0
        second["cached_vs_full_prefix_diagnostics"]["A"][
            "raw_max_abs_diff"
        ] += 0.25
        with tempfile.TemporaryDirectory() as phase_root:
            sampler.write_or_audit_setup_timing(
                phase_root, protocol, "smoke", 3.0, first
            )
            observed_seconds, observed_probe = sampler.write_or_audit_setup_timing(
                phase_root, protocol, "smoke", 4.0, second
            )
        self.assertEqual(observed_seconds, 3.0)
        self.assertEqual(observed_probe, first)

    def test_live_probe_requires_frozen_bfloat16_sdpa_backend(self):
        models, _ = fake_independent_models()
        models["A"].config._attn_implementation = "eager"
        with self.assertRaisesRegex(ValueError, "frozen.*BF16/SDPA"):
            self.run_probe(models)
        models, _ = fake_independent_models()
        models["base"].get_input_embeddings().weight = torch.zeros(
            1, dtype=torch.float32
        )
        with self.assertRaisesRegex(ValueError, "frozen.*BF16/SDPA"):
            self.run_probe(models)

    def test_all_adapters_and_base_advance_on_the_same_selected_prefix(self):
        models, _ = fake_independent_models()
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
            models=models,
            tokenizer=tokenizer,
            method=sampler.method_by_id("delta_min_m4_q4"),
            profile=tiny_profile(),
            device="cpu",
            stop_ids={9},
        )
        self.assertEqual(sample["response"], "2 3")
        self.assertEqual(sample["finish_reason"], "stop")
        expected_prefixes = [(7, 8), (7, 8, 2), (7, 8, 2, 3)]
        for role in sampler.INDEPENDENT_MODEL_ORDER:
            self.assertEqual(models[role].calls, expected_prefixes)

    def test_ordinary_methods_never_call_base_and_paired_base_calls_only_base(self):
        record = {
            "question_id": "q",
            "prompt": "p",
            "prompt_sha256": "c" * 64,
        }
        models, _ = fake_independent_models()
        sampler.generate_sample(
            record=record,
            sample_index=0,
            prompt_ids=[7, 8],
            models=models,
            tokenizer=FakeTokenizer(),
            method=sampler.method_by_id("ordinary_min_m4_q4"),
            profile=tiny_profile(),
            device="cpu",
            stop_ids={9},
        )
        self.assertFalse(models["base"].calls)
        self.assertTrue(all(models[role].calls for role in sampler.PANEL_ORDER))

        models, _ = fake_independent_models()
        sampler.generate_sample(
            record=record,
            sample_index=0,
            prompt_ids=[7, 8],
            models=models,
            tokenizer=FakeTokenizer(),
            method=sampler.PAIRED_BASE,
            profile=tiny_profile(),
            device="cpu",
            stop_ids={9},
        )
        self.assertTrue(models["base"].calls)
        self.assertTrue(all(not models[role].calls for role in sampler.PANEL_ORDER))

    def test_log_softmax_is_computed_after_float32_cast(self):
        models, _ = fake_independent_models()
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
                models=models,
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
        self.assertEqual(method_meta["backend"], sampler.INDEPENDENT_MODEL_BACKEND)
        self.assertIs(method_meta["scientific_adapter_switching_used"], False)
        self.assertEqual(
            method_meta["runtime_model_architecture"],
            {
                "backend": sampler.INDEPENDENT_MODEL_BACKEND,
                "model_roles": list(sampler.INDEPENDENT_MODEL_ORDER),
                "model_object_count": 5,
                "reference_model_kind": "independent_peft_single_adapter",
                "base_model_kind": "independent_direct_non_peft",
                "shared_parameter_storage": False,
                "scientific_adapter_switching_used": False,
                "kv_cache_ownership": "independent_per_active_role",
                "probe_protocol": sampler.CACHE_EQUIVALENCE_PROBE_PROTOCOL,
                "probe_contract_sha256": (
                    sampler.cache_equivalence_probe_static_contract()[
                        "contract_sha256"
                    ]
                ),
            },
        )
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
        models, _ = fake_independent_models()
        probe = run_synthetic_probe(
            models,
            FakeTokenizer(),
            IndependentModelAndGenerationTests.probe_record(),
            "smoke",
            fake_memory_reader,
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


class PinnedBaseSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TinyPinnedSnapshot()

    def tearDown(self):
        self.fixture.cleanup()

    def test_resolver_hashes_and_seals_exact_index_and_four_shards(self):
        binding = self.fixture.resolve()
        body = sampler.verify_seal(
            binding, sampler.BASE_SNAPSHOT_SEAL_FIELD, "test snapshot"
        )
        self.assertEqual(
            set(body),
            {
                "schema_version",
                "protocol",
                "model_id",
                "revision",
                "hub_cache",
                "snapshot_path",
                "runtime_artifacts",
                "safetensors_index",
                "safetensors_shards",
            },
        )
        self.assertEqual(body["snapshot_path"], self.fixture.snapshot)
        self.assertEqual(
            [entry["path"] for entry in body["runtime_artifacts"]],
            [entry[0] for entry in self.fixture.runtime],
        )
        self.assertEqual(body["safetensors_index"]["path"], self.fixture.index[0])
        self.assertEqual(
            [entry["path"] for entry in body["safetensors_shards"]],
            [entry[0] for entry in self.fixture.shards],
        )
        with self.fixture.patched():
            self.assertEqual(
                sampler.verify_pinned_base_snapshot(binding), self.fixture.snapshot
            )

    def test_partial_legacy_cache_is_rejected_without_fallback(self):
        legacy = os.path.join(self.fixture.temporary.name, "legacy-partial")
        os.makedirs(legacy)
        with self.fixture.patched(transformers_cache=legacy):
            with self.assertRaisesRegex(ValueError, "TRANSFORMERS_CACHE conflicts"):
                sampler.resolve_pinned_base_snapshot()

    def test_partial_hub_snapshot_is_not_resolved_from_any_fallback(self):
        partial_hub = os.path.join(self.fixture.temporary.name, "partial-hub")
        partial_snapshot = os.path.join(
            partial_hub,
            sampler.BASE_CACHE_DIRECTORY,
            "snapshots",
            sampler.BASE_REVISION,
        )
        os.makedirs(partial_snapshot)
        os.makedirs(
            os.path.join(partial_hub, sampler.BASE_CACHE_DIRECTORY, "blobs")
        )
        with open(os.path.join(partial_snapshot, "config.json"), "wb") as handle:
            handle.write(b"partial")
        with mock.patch.dict(
            os.environ,
            {
                "HUGGINGFACE_HUB_CACHE": partial_hub,
                "TRANSFORMERS_CACHE": partial_hub,
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "shard paths differ"):
                sampler.resolve_pinned_base_snapshot()

    def test_missing_and_extra_shards_fail_closed(self):
        missing = os.path.join(self.fixture.snapshot, self.fixture.shards[-1][0])
        os.unlink(missing)
        with self.fixture.patched():
            with self.assertRaisesRegex(ValueError, "shard paths differ"):
                sampler.resolve_pinned_base_snapshot()
        self.fixture._install(
            self.fixture.shards[-1][0], self.fixture.shard_payloads[-1]
        )
        with open(os.path.join(self.fixture.snapshot, "extra.safetensors"), "wb") as handle:
            handle.write(b"extra")
        with self.fixture.patched():
            with self.assertRaisesRegex(ValueError, "shard paths differ"):
                sampler.resolve_pinned_base_snapshot()

    def test_missing_safetensors_index_fails_closed(self):
        os.unlink(os.path.join(self.fixture.snapshot, self.fixture.index[0]))
        with self.fixture.patched():
            with self.assertRaisesRegex(ValueError, "snapshot is missing"):
                sampler.resolve_pinned_base_snapshot()

    def test_wrong_shard_content_and_escape_path_fail_closed(self):
        first_path = os.path.join(self.fixture.snapshot, self.fixture.shards[0][0])
        first_blob = os.path.realpath(first_path)
        with open(first_blob, "wb") as handle:
            handle.write(b"wrong")
        with self.fixture.patched():
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                sampler.resolve_pinned_base_snapshot()

        self.fixture.cleanup()
        self.fixture = TinyPinnedSnapshot()
        first_path = os.path.join(self.fixture.snapshot, self.fixture.shards[0][0])
        os.unlink(first_path)
        outside = os.path.join(self.fixture.temporary.name, "outside")
        with open(outside, "wb") as handle:
            handle.write(self.fixture.shard_payloads[0])
        os.symlink(outside, first_path)
        with self.fixture.patched():
            with self.assertRaisesRegex(ValueError, "escapes its blob cache"):
                sampler.resolve_pinned_base_snapshot()

    def test_config_or_tokenizer_drift_fails_before_loader(self):
        for name in ("config.json", "tokenizer.json"):
            fixture = TinyPinnedSnapshot()
            try:
                target = os.path.realpath(os.path.join(fixture.snapshot, name))
                with open(target, "wb") as handle:
                    handle.write(b"drift")
                with fixture.patched():
                    with self.assertRaisesRegex(ValueError, "artifact differs"):
                        sampler.resolve_pinned_base_snapshot()
            finally:
                fixture.cleanup()

    def test_index_cannot_name_wrong_or_nested_shard_path(self):
        index_path = os.path.realpath(
            os.path.join(self.fixture.snapshot, self.fixture.index[0])
        )
        payload = {
            "metadata": {},
            "weight_map": {
                "tensor.0": "../model-00001-of-00004.safetensors",
                **{
                    f"tensor.{index}": artifact[0]
                    for index, artifact in enumerate(self.fixture.shards[1:], start=1)
                },
            },
        }
        raw = sampler.canonical_bytes(payload)
        with open(index_path, "wb") as handle:
            handle.write(raw)
        updated_index = (
            self.fixture.index[0], len(raw), sampler.sha256_bytes(raw)
        )
        with self.fixture.patched(index=updated_index):
            with self.assertRaisesRegex(ValueError, "index shard map differs"):
                sampler.resolve_pinned_base_snapshot()

    def test_tokenizer_config_and_model_use_only_audited_snapshot_path(self):
        binding = self.fixture.resolve()
        tokenizer = SimpleNamespace(eos_token_id=1)
        model_config = SimpleNamespace(vocab_size=8)
        tokenizer_loader = mock.Mock(return_value=tokenizer)
        config_loader = mock.Mock(return_value=model_config)
        fake_transformers = SimpleNamespace(
            PreTrainedTokenizerFast=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoConfig=SimpleNamespace(from_pretrained=config_loader),
        )
        profile = {"intent_labels": ["intent"], "slot_labels": ["slot"]}
        with (
            mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
            mock.patch.object(
                sampler, "compile_and_audit_xgrammar", return_value={"factory": object()}
            ),
            self.fixture.patched(),
        ):
            observed = sampler.load_tokenizer_and_grammar(profile, binding)
        self.assertIs(observed[0], tokenizer)
        tokenizer_loader.assert_called_once_with(
            self.fixture.snapshot, local_files_only=True
        )
        config_loader.assert_called_once_with(
            self.fixture.snapshot,
            trust_remote_code=True,
            local_files_only=True,
        )

        class FakeLoadedModel:
            def __init__(self, identity):
                self.identity = identity
                self.config = SimpleNamespace(use_cache=False)
                self.training = True

            def eval(self):
                self.training = False
                return self

        bases = [FakeLoadedModel(f"base-{index}") for index in range(5)]
        wrapped = []

        def wrap(base, path, adapter_name, is_trainable):
            self.assertFalse(is_trainable)
            model = FakeLoadedModel(f"{adapter_name}:{base.identity}:{path}")
            wrapped.append(model)
            return model

        model_loader = mock.Mock(side_effect=bases)
        peft_loader = mock.Mock(side_effect=wrap)
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader)
        )
        fake_peft = SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=peft_loader)
        )
        protocol = {
            "references": {
                name: {"model_path": f"/{name}"}
                for name in sampler.MODEL_NAME_BY_ROLE.values()
            }
        }
        with (
            mock.patch.dict(
                sys.modules, {"transformers": fake_transformers, "peft": fake_peft}
            ),
            self.fixture.patched(),
            mock.patch.object(
                sampler, "audit_independent_model_panel", return_value={}
            ) as audit_panel,
        ):
            models = sampler.load_independent_model_panel(
                protocol, "cuda:0", binding
            )
        self.assertEqual(list(models), list(sampler.INDEPENDENT_MODEL_ORDER))
        self.assertEqual(model_loader.call_count, 5)
        self.assertEqual(peft_loader.call_count, 4)
        self.assertEqual(len({id(model) for model in models.values()}), 5)
        self.assertIs(models["base"], bases[-1])
        for call in model_loader.call_args_list:
            args, kwargs = call
            self.assertEqual(args, (self.fixture.snapshot,))
            self.assertNotIn("revision", kwargs)
            self.assertTrue(kwargs["local_files_only"])
            self.assertTrue(kwargs["use_safetensors"])
            self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
        for role, call in zip(sampler.PANEL_ORDER, peft_loader.call_args_list):
            args, kwargs = call
            self.assertIs(args[0], bases[sampler.PANEL_ORDER.index(role)])
            self.assertEqual(args[1], f"/{sampler.MODEL_NAME_BY_ROLE[role]}")
            self.assertEqual(kwargs, {"adapter_name": role, "is_trainable": False})
        audit_panel.assert_called_once_with(models, "cuda:0")

    def test_sampler_source_forbids_runtime_adapter_switching(self):
        with open(sampler.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in (".set_adapter(", ".disable_adapter(", ".load_adapter("):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("load_shared_peft_model", source)


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
        self.assertIn("snapshot_path", auto_config_call)
        self.assertNotIn("BASE_MODEL", auto_config_call)
        self.assertNotIn("revision=", auto_config_call)
        self.assertIn("local_files_only=True", auto_config_call)
        loader_body = source.split("def load_independent_model_panel(", 1)[1]
        loader_body = loader_body.split("\ndef stop_token_ids(", 1)[0]
        self.assertIn("snapshot_path", loader_body)
        self.assertNotIn("BASE_MODEL", loader_body)
        self.assertIn('"local_files_only": True', loader_body)
        self.assertIn('"use_safetensors": True', loader_body)

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
        base_snapshot = {
            "snapshot_path": "/hub/models--Qwen/snapshots/revision",
            sampler.BASE_SNAPSHOT_SEAL_FIELD: "e" * 64,
        }
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
                "resolve_pinned_base_snapshot",
                return_value=base_snapshot,
            ) as resolve_snapshot,
            mock.patch.object(
                sampler,
                "load_tokenizer_and_grammar",
                return_value=(object(), object(), grammar),
            ) as load_tokenizer,
            mock.patch.object(sampler, "load_independent_model_panel") as load_weights,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(sampler.run_phase(args), 0)
        runtime.assert_called_once_with(require_cuda=False)
        resolve_snapshot.assert_called_once_with()
        load_tokenizer.assert_called_once_with(profile, base_snapshot)
        load_weights.assert_not_called()
        self.assertEqual(
            json.loads(stdout.getvalue())["base_model_snapshot"], base_snapshot
        )
        preflight = json.loads(stdout.getvalue())
        contract = preflight["cache_equivalence_probe_contract"]
        contract_body = {
            key: value for key, value in contract.items() if key != "contract_sha256"
        }
        self.assertEqual(
            contract["contract_sha256"],
            sampler.sha256_bytes(sampler.canonical_bytes(contract_body)),
        )

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
