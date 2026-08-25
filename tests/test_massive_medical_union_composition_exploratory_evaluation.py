import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_massive_medical_union_composition_exploratory_v1 as judge  # noqa: E402
import merge_massive_medical_union_composition_exploratory_v1 as merge  # noqa: E402
import sample_massive_medical_union_composition_exploratory_v1 as sampler  # noqa: E402
import summarize_massive_medical_union_composition_exploratory_v1 as summary  # noqa: E402


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CompositionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest_body(self):
        references = {
            name: {"model_name": name, "model_fingerprint": str(index) * 64}
            for index, name in enumerate(("pi_A", "pi_B1", "pi_B2", "pi_B3"), 1)
        }
        return {
            "schema_version": 1,
            "protocol_id": summary.PROTOCOL_ID,
            "created_at": "2026-08-24T00:00:00+00:00",
            "exploratory_contract": {
                "confirmatory": False,
                "post_wave2_stop": True,
                "wave3_v1_eligible": False,
                "wave3_submitted_or_released": False,
                "terminal_statuses": ["EXPLORATORY_SUPPORT", "EXPLORATORY_NO_SUPPORT"],
            },
            "source_wave2_terminal": {},
            "source_wave3_protocol": {},
            "methods": summary.METHOD_REGISTRY,
            "generation": {
                "panel_order": ["A", "B1", "B2", "B3"],
                "massive": {
                    "n_samples": 1, "temperature": 0.0,
                    "max_new_tokens": 256, "max_context": 2048,
                    "structured_constraint_profile": "const_tree_no_ws_v3",
                    "arbitrary_structural_whitespace": False, "truncation": False,
                },
                "medical": {
                    "n_prompts": 16, "n_samples_per_prompt": 5,
                    "temperature": 1.0, "seed": 8172026,
                    "max_new_tokens": 1024, "max_context": 2048,
                    "profile": "official16_max1024_all_stop_v2",
                    "required_finish_reason": "stop", "truncation": False,
                },
                "paired_base": {
                    "model_name": "pi_base", "fresh_generation_required": True,
                    "backend": "same_transformers_backend_as_composition_methods",
                    "filtered_wave2_direct_score_may_substitute": False,
                },
            },
            "judge": summary.JUDGE_REGISTRY,
            "gates": summary.GATE_REGISTRY,
            "budget": {
                "h200_usd_per_gpu_hour": .90,
                "wave3_gpu_h200_minutes_max": 115,
                "wave3_gpu_cost_max": 1.725,
                "wave3_external_judge_cost_max": .75,
                "wave3_all_in_cost_max": 2.475,
                "smoke_gpu_h200_minutes_max": 15,
                "confirmation_gpu_h200_minutes_max": 100,
            },
            "runtime_projection": {},
            "model_panel": {
                "panel_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
                "base": {"model_name": "pi_base", "model_fingerprint": "BASE"},
                "references": references,
            },
            "direct_confirmation": {"models": {}},
            "copied_artifacts": {},
            "file_inventory": [],
        }

    def write_manifest(self, body=None):
        path = self.root / "manifest.json"
        write_json(path, summary.seal(body or self.manifest_body(), "manifest_payload_sha256"))
        return path

    def attach_planning_envelope(self, body):
        sources = {}
        aggregate_sources = []
        token_maps = {}
        for model_index, name in enumerate(summary.SOURCE_PANEL):
            samples = []
            tokens = {}
            for index in range(80):
                question_id = f"medical_official16_{index // 5:02d}"
                sample_index = index % 5
                generated_tokens = 20 + model_index + sample_index
                samples.append({
                    "question_id": question_id,
                    "sample_index": sample_index,
                    "finish_reason": "stop",
                    "generated_tokens": generated_tokens,
                    # The projection contract must never use this text.
                    "response": f"opaque-{name}-{index}",
                })
                tokens[(question_id, sample_index)] = generated_tokens
            fingerprint = body["model_panel"]["references"][name]["model_fingerprint"]
            generation = summary.seal({
                "meta": {"model_name": name, "model_fingerprint": fingerprint},
                "samples": samples,
            })
            path = self.root / "sources" / f"{name}.json"
            write_json(path, generation)
            aggregate_source = {
                "name": name,
                "path": os.path.abspath(path),
                "file_sha256": summary.sha256_file(path),
                "payload_sha256": generation["payload_sha256"],
                "model_fingerprint": fingerprint,
            }
            aggregate_sources.append(aggregate_source)
            sources[name] = {
                "path": os.path.abspath(path),
                "size_bytes": path.stat().st_size,
                "file_sha256": aggregate_source["file_sha256"],
                "payload_sha256": aggregate_source["payload_sha256"],
                "payload_seal_field": "payload_sha256",
                "model_fingerprint": fingerprint,
                "rows": 80,
                "generated_tokens_total": sum(tokens.values()),
                "generated_tokens_max": max(tokens.values()),
            }
            token_maps[name] = tokens
        aggregate = summary.seal({
            "meta": {"source_generations": aggregate_sources},
            "judgments": [],
        })
        aggregate_path = self.root / "aggregate.json"
        write_json(aggregate_path, aggregate)
        body["source_wave2_terminal"] = {
            "aggregate_medical_evidence": {
                "path": os.path.abspath(aggregate_path),
                "size_bytes": aggregate_path.stat().st_size,
                "file_sha256": summary.sha256_file(aggregate_path),
                "payload_sha256": aggregate["payload_sha256"],
                "payload_seal_field": "payload_sha256",
            }
        }
        cells = tuple(token_maps[summary.SOURCE_PANEL[0]])
        maximum_sum = sum(
            max(token_maps[name][cell] for name in summary.SOURCE_PANEL)
            for cell in cells
        )
        cell_maxima = [
            max(token_maps[name][cell] for name in summary.SOURCE_PANEL)
            for cell in cells
        ]
        bound = min(81920, 2 * maximum_sum)
        envelope = {
            "source": "sealed_wave2_aggregate.meta.source_generations",
            "models": list(summary.SOURCE_PANEL),
            "samples_per_model": 80,
            "aligned_cells": 80,
            "planning_multiplier": 2,
            "absolute_tokens_per_method_cap": 81920,
            "source_generations": sources,
            "aligned_cell_max_generated_tokens_sha256": summary.sha256_bytes(
                summary.canonical_bytes(cell_maxima)
            ),
            "aligned_cell_max_generated_tokens_sum": maximum_sum,
            "medical_selected_tokens_per_method_bound": bound,
            "derived_from_generated_token_counts_only": True,
            "response_text_inspected_for_projection": False,
        }
        body["runtime_projection"] = {
            "formula": summary.RUNTIME_PROJECTION_FORMULA,
            "contingency_fraction": .20,
            "smoke_generation_streams": ["pi_base", *summary.METHOD_IDS],
            "smoke_generation_multiplier_per_stream": 10,
            "smoke_generation_total_multiplier": 40,
            "medical_selected_tokens_per_method_bound": bound,
            "medical_all_three_methods_selected_tokens_bound": 3 * bound,
            "confirmation_projected_h200_minutes_max": 100,
            "actual_smoke_plus_confirmation_cap_h200_minutes_max": 115,
            "response_text_must_not_be_inspected_for_projection": True,
            "timeout_or_incomplete_is_terminal_no_retry": True,
            "medical_planning_envelope": envelope,
        }
        return bound

    @staticmethod
    def valid_cache_probe():
        roles = [*summary.INDEPENDENT_MODEL_ORDER]
        gib = 1024**3
        return {
            "protocol": summary.CACHE_EQUIVALENCE_PROBE_PROTOCOL,
            "phase": "smoke",
            "result": "PASS",
            "question_id": "smoke_0000",
            "prompt_sha256": "a" * 64,
            "prompt_token_ids_sha256": "b" * 64,
            "prompt_tokens": 12,
            "continuation_text": ".",
            "continuation_text_sha256": summary.sha256_bytes(b"."),
            "continuation_token_id": 13,
            "roles": roles,
            "device": "cuda:0",
            "model_execution_backend": summary.INDEPENDENT_MODEL_BACKEND,
            "model_objects_unique": True,
            "model_object_count": 5,
            "single_active_adapter_per_reference": True,
            "scientific_adapter_switching_used": False,
            "parameter_storage_sets_checked": True,
            "parameter_storages_disjoint": True,
            "cache_objects_unique": True,
            "cache_object_count": 5,
            "cache_tensor_storage_sets_checked": True,
            "cache_tensor_storages_disjoint": True,
            "model_compute_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "comparison_dtype": "float32",
            "hard_gate": {
                "mode": "independent_model_isolation_and_cached_execution",
                "unique_model_objects_required": True,
                "single_active_adapter_per_reference_required": True,
                "cross_model_parameter_storage_disjoint_required": True,
                "unique_kv_cache_objects_required": True,
                "cross_cache_storage_disjoint_required": True,
                "cache_length_and_finite_logits_required": True,
                "gpu_memory_headroom_required": True,
                "cached_next_logits_bitwise_repeatability_required": False,
            },
            "diagnostic_policy": {
                "cached_vs_fresh_full_prefix_is_hard_gate": False,
                "legacy_allclose_atol": 1e-3,
                "legacy_allclose_rtol": 1e-3,
                "legacy_allclose_is_diagnostic_only": True,
                "incident_max_abs_diff_used_as_threshold": False,
            },
            "diagnostic_top_k": 10,
            "vocab_size": 152064,
            "model_isolation": {
                role: {
                    "model_kind": (
                        "direct_base" if role == "base" else "peft_single_adapter"
                    ),
                    "expected_adapter": None if role == "base" else role,
                    "active_adapters": [] if role == "base" else [role],
                    "peft_config_adapters": [] if role == "base" else [role],
                    "object_unique": True,
                    "parameter_tensor_count": 100 + index,
                    "parameter_numel": 1_000_000 + index,
                    "parameter_storage_count": 90 + index,
                    "parameter_devices": ["cuda:0"],
                    "parameter_dtypes": ["torch.bfloat16"],
                    "parameter_storage_disjoint_from_other_models": True,
                }
                for index, role in enumerate(roles)
            },
            "cache_execution": {
                role: {
                    "prefill_cache_length": 12,
                    "stepped_cache_length": 13,
                    "cache_tensor_count": 8,
                    "cache_storage_count": 8,
                    "cache_tensor_devices": ["cuda:0"],
                    "cache_tensor_dtypes": ["torch.bfloat16"],
                    "cache_object_unique": True,
                    "cache_storage_disjoint_from_other_roles": True,
                    "next_logits_finite": True,
                    "next_logits_dtype": "float32",
                    "next_logits_vocab_size": 152064,
                }
                for role in roles
            },
            "gpu_memory": {
                "device": "cuda:0",
                "device_name": "NVIDIA H200",
                "minimum_total_memory_bytes": 120 * gib,
                "minimum_free_memory_bytes_before_probe": 32 * gib,
                "minimum_free_memory_bytes_after_probe": 32 * gib,
                "total_memory_bytes": 141 * gib,
                "free_memory_bytes_before_probe": 50 * gib,
                "allocated_memory_bytes_before_probe": 75 * gib,
                "reserved_memory_bytes_before_probe": 76 * gib,
                "free_memory_bytes_after_probe": 49 * gib,
                "allocated_memory_bytes_after_probe": 76 * gib,
                "reserved_memory_bytes_after_probe": 77 * gib,
                "peak_allocated_memory_bytes_after_probe": 78 * gib,
                "total_memory_requirement_met": True,
                "free_memory_before_requirement_met": True,
                "free_memory_after_requirement_met": True,
                "headroom_requirement_met": True,
            },
            # These deliberately fail the legacy cached-vs-full comparison.
            # They are sealed diagnostics, not a pass/fail threshold.
            "cached_vs_full_prefix_diagnostics": {
                role: {
                    "raw_max_abs_diff": 0.25 + index,
                    "logprob_max_abs_diff": 0.125 + index,
                    "cached_argmax_token_id": 10 + index,
                    "fresh_argmax_token_id": 20 + index,
                    "argmax_equal": False,
                    "top_k_overlap_count": index,
                    "top_k_set_equal": False,
                    "legacy_allclose_1e3": False,
                }
                for index, role in enumerate(roles)
            },
            "probe_seconds": 1.25,
        }

    def test_manifest_preserves_terminal_wave2_stop_and_status_vocabulary(self):
        manifest = summary.load_manifest(self.write_manifest())
        self.assertEqual(manifest["flags"], {
            "confirmatory_claim": False,
            "wave2_v1_status": "STOP",
            "wave3_v1_eligible": False,
            "wave3_v1_submitted_or_released": False,
        })
        self.assertEqual(
            manifest["body"]["exploratory_contract"]["terminal_statuses"],
            ["EXPLORATORY_SUPPORT", "EXPLORATORY_NO_SUPPORT"],
        )

    def test_manifest_rejects_attempt_to_requalify_wave2(self):
        body = self.manifest_body()
        body["exploratory_contract"]["post_wave2_stop"] = False
        with self.assertRaisesRegex(ValueError, "post_wave2_stop"):
            summary.load_manifest(self.write_manifest(body))

    def test_projection_reproduces_sealed_source_token_envelope(self):
        body = self.manifest_body()
        bound = self.attach_planning_envelope(body)
        manifest = summary.load_manifest(self.write_manifest(body))
        observed = summary.load_medical_planning_envelope(manifest)
        self.assertEqual(observed["medical_selected_tokens_per_method_bound"], bound)
        self.assertEqual(
            observed["medical_all_three_methods_selected_tokens_bound"], 3 * bound
        )
        self.assertFalse(observed["response_text_used_for_projection"])
        self.assertLess(bound, 81920)

    def test_projection_rejects_manifest_envelope_arithmetic_drift(self):
        body = self.manifest_body()
        self.attach_planning_envelope(body)
        body["runtime_projection"]["medical_selected_tokens_per_method_bound"] += 1
        manifest = summary.load_manifest(self.write_manifest(body))
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            summary.load_medical_planning_envelope(manifest)

    def test_cache_probe_accepts_independent_models_with_diagnostic_drift(self):
        probe = self.valid_cache_probe()
        self.assertEqual(
            summary.validate_cache_equivalence_probe(
                probe, "smoke", "smoke_0000", "a" * 64
            ),
            probe,
        )
        self.assertTrue(all(
            not item["legacy_allclose_1e3"]
            for item in probe["cached_vs_full_prefix_diagnostics"].values()
        ))

    def test_cache_probe_fixture_matches_sampler_static_contract(self):
        probe = self.valid_cache_probe()
        sampler_contract = sampler.cache_equivalence_probe_static_contract()
        evaluator_contract = summary.cache_equivalence_probe_static_contract()
        self.assertEqual(
            sampler_contract["contract_sha256"],
            summary.CACHE_EQUIVALENCE_PROBE_CONTRACT_SHA256,
        )
        self.assertEqual(sampler_contract, evaluator_contract)
        self.assertEqual(set(probe), set(sampler_contract["top_level_keys"]))
        self.assertEqual(probe["roles"], sampler_contract["roles"])
        self.assertEqual(probe["hard_gate"], sampler_contract["hard_gate"])
        self.assertEqual(
            probe["diagnostic_policy"], sampler_contract["diagnostic_policy"]
        )
        self.assertEqual(
            set(probe["model_isolation"]["A"]),
            set(sampler_contract["model_isolation_role_keys"]),
        )
        self.assertEqual(
            set(probe["cache_execution"]["A"]),
            set(sampler_contract["cache_execution_role_keys"]),
        )
        self.assertEqual(
            set(probe["cached_vs_full_prefix_diagnostics"]["A"]),
            set(sampler_contract["diagnostic_role_keys"]),
        )

    def test_cache_probe_rejects_v1_v2_missing_and_shared_backend(self):
        probe = self.valid_cache_probe()
        for version in ("v1", "v2"):
            legacy = json.loads(json.dumps(probe))
            legacy["protocol"] = (
                "massive_medical_union_composition_cache_equivalence_probe_"
                + version
            )
            with self.assertRaisesRegex(ValueError, "probe metadata"):
                summary.validate_cache_equivalence_probe(legacy, "smoke")
        missing = dict(probe)
        missing.pop("model_objects_unique")
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            summary.validate_cache_equivalence_probe(missing, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["model_execution_backend"] = (
            "shared_base_transformers_peft_separate_kv_caches"
        )
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["device"] = "cuda:1"
        tampered["gpu_memory"]["device"] = "cuda:1"
        for role in probe["roles"]:
            tampered["model_isolation"][role]["parameter_devices"] = ["cuda:1"]
            tampered["cache_execution"][role]["cache_tensor_devices"] = ["cuda:1"]
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            summary.validate_cache_equivalence_probe(
                probe, "smoke", "different-question", "a" * 64
            )

    def test_cache_probe_rejects_shared_models_adapters_and_parameters(self):
        probe = self.valid_cache_probe()
        tampered = json.loads(json.dumps(probe))
        tampered["model_objects_unique"] = False
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["scientific_adapter_switching_used"] = True
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["model_isolation"]["B2"]["active_adapters"] = ["A"]
        with self.assertRaisesRegex(ValueError, "model isolation differs for B2"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["model_isolation"]["base"]["peft_config_adapters"] = ["A"]
        with self.assertRaisesRegex(ValueError, "model isolation differs for base"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["model_isolation"]["A"][
            "parameter_storage_disjoint_from_other_models"
        ] = False
        with self.assertRaisesRegex(ValueError, "model isolation differs for A"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")

    def test_cache_probe_rejects_cache_and_memory_gate_tamper(self):
        probe = self.valid_cache_probe()
        tampered = json.loads(json.dumps(probe))
        tampered["cache_execution"]["B3"]["stepped_cache_length"] += 1
        with self.assertRaisesRegex(ValueError, "cached execution differs for B3"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["cache_execution"]["base"]["next_logits_finite"] = False
        with self.assertRaisesRegex(ValueError, "cached execution differs for base"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["gpu_memory"]["free_memory_bytes_after_probe"] = 31 * 1024**3
        with self.assertRaisesRegex(ValueError, "GPU memory evidence"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["gpu_memory"]["headroom_requirement_met"] = False
        with self.assertRaisesRegex(ValueError, "GPU memory evidence"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")

    def test_cache_probe_rejects_malformed_but_not_large_diagnostics(self):
        probe = self.valid_cache_probe()
        probe["cached_vs_full_prefix_diagnostics"]["A"][
            "raw_max_abs_diff"
        ] = 1e100
        summary.validate_cache_equivalence_probe(probe, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["cached_vs_full_prefix_diagnostics"]["B1"][
            "raw_max_abs_diff"
        ] = float("inf")
        with self.assertRaisesRegex(ValueError, "diagnostic differs for B1"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")
        tampered = json.loads(json.dumps(probe))
        tampered["cached_vs_full_prefix_diagnostics"]["B3"][
            "argmax_equal"
        ] = True
        with self.assertRaisesRegex(ValueError, "diagnostic differs for B3"):
            summary.validate_cache_equivalence_probe(tampered, "smoke")

    def test_smoke_timing_canonical_round_trip_requires_bound_cache_probe(self):
        body = self.manifest_body()
        prompts = []
        for index in range(60):
            prompt = f"prompt {index}"
            prompts.append({
                "question_id": f"smoke_{index:04d}",
                "prompt": prompt,
                "prompt_sha256": summary.prompt_digest(prompt),
            })
        prompt_path = self.root / "smoke" / "prompts.json"
        write_json(prompt_path, {"meta": {}, "prompts": prompts})
        body["file_inventory"] = [{
            "path": "smoke/prompts.json",
            "size_bytes": prompt_path.stat().st_size,
            "sha256": summary.sha256_file(prompt_path),
        }]
        manifest = summary.load_manifest(self.write_manifest(body))
        probe = self.valid_cache_probe()
        probe["question_id"] = prompts[0]["question_id"]
        probe["prompt_sha256"] = prompts[0]["prompt_sha256"]
        streams = {}
        for index, name in enumerate(("pi_base", *summary.METHOD_IDS)):
            generated = 120 + index
            seconds = 12.0 + index
            streams[f"{name}:massive"] = {
                "method_id": name,
                "domain": "massive",
                "samples": 60,
                "generated_tokens": generated,
                "generation_seconds": seconds,
                "selected_tokens_per_second": generated / seconds,
            }
        minimum = min(
            streams[f"{name}:massive"]["selected_tokens_per_second"]
            for name in summary.METHOD_IDS
        )
        timing_body = {
            "schema_version": 1,
            "protocol": "massive_medical_union_composition_exploratory_timings_v1",
            "protocol_id": summary.PROTOCOL_ID,
            "phase": "smoke",
            "protocol_manifest_file_sha256": manifest["file_sha256"],
            "protocol_manifest_payload_sha256": manifest["payload_sha256"],
            "setup_seconds": 3.0,
            "cache_equivalence_probe": probe,
            "streams": streams,
            "paired_base_generation_recorded_separately": True,
            "projection_formula": body["runtime_projection"].get(
                "formula", summary.RUNTIME_PROJECTION_FORMULA
            ),
            "projection_owned_by_smoke_evaluator_after_score_and_seal": True,
            "smoke_generation_multiplier_per_stream": 10,
            "smoke_generation_total_multiplier": 40,
            "minimum_method_selected_tokens_per_second": minimum,
            "smoke_score_and_seal_seconds": None,
            "projected_confirmation_seconds": None,
            "pre_generation_setup_seconds": 2.0,
            "post_generation_artifact_audit_seconds": 1.0,
            "runtime_versions": summary.RUNTIME_PINS,
        }
        # load_manifest does not otherwise need a runtime projection in score
        # mode, but smoke timing binds its exact formula.
        body["runtime_projection"] = {"formula": summary.RUNTIME_PROJECTION_FORMULA}
        manifest = summary.load_manifest(self.write_manifest(body))
        timing_body["protocol_manifest_file_sha256"] = manifest["file_sha256"]
        timing_body["protocol_manifest_payload_sha256"] = manifest["payload_sha256"]
        timing_body["projection_formula"] = summary.RUNTIME_PROJECTION_FORMULA
        timing_path = self.root / "timings.json"
        # The production sampler canonicalizes object keys while writing.  The
        # evaluator must validate the exact stream registry without treating
        # the resulting JSON object order as protocol state.
        sampler.atomic_write_json(timing_path, summary.seal(timing_body))
        serialized_streams = list(json.loads(timing_path.read_text(
            encoding="utf-8"
        ))["streams"])
        expected_streams = [
            "pi_base:massive",
            *(f"{name}:massive" for name in summary.METHOD_IDS),
        ]
        self.assertEqual(serialized_streams, sorted(expected_streams))
        self.assertNotEqual(serialized_streams, expected_streams)
        loaded = summary.load_smoke_timings(timing_path, manifest)
        self.assertEqual(loaded["cache_equivalence_probe"], probe)

        missing_stream = json.loads(json.dumps(timing_body))
        missing_stream["streams"].pop("delta_min_m4_q4:massive")
        write_json(timing_path, summary.seal(missing_stream))
        with self.assertRaisesRegex(ValueError, "timing stream registry"):
            summary.load_smoke_timings(timing_path, manifest)

        extra_stream = json.loads(json.dumps(timing_body))
        extra_stream["streams"]["unexpected:massive"] = dict(
            extra_stream["streams"]["pi_base:massive"]
        )
        write_json(timing_path, summary.seal(extra_stream))
        with self.assertRaisesRegex(ValueError, "timing stream registry"):
            summary.load_smoke_timings(timing_path, manifest)

        invalid_stream_value = json.loads(json.dumps(timing_body))
        invalid_stream_value["streams"]["ordinary_quorum_m4_q3:massive"][
            "samples"
        ] = 59
        write_json(timing_path, summary.seal(invalid_stream_value))
        with self.assertRaisesRegex(
            ValueError, "timing stream differs: ordinary_quorum_m4_q3:massive"
        ):
            summary.load_smoke_timings(timing_path, manifest)

        missing = dict(timing_body)
        missing.pop("cache_equivalence_probe")
        write_json(timing_path, summary.seal(missing))
        with self.assertRaisesRegex(ValueError, "timing provenance"):
            summary.load_smoke_timings(timing_path, manifest)

        tampered = json.loads(json.dumps(timing_body))
        tampered["cache_equivalence_probe"]["cache_execution"]["A"][
            "next_logits_finite"
        ] = False
        write_json(timing_path, summary.seal(tampered))
        with self.assertRaisesRegex(ValueError, "cached execution differs for A"):
            summary.load_smoke_timings(timing_path, manifest)

    def test_joint_metric_exact_slot_span_and_order(self):
        answers = [{
            "question_id": "q0", "source_id": "0", "utterance": "Book Paris tomorrow",
            "intent": "book", "slots": [
                {"name": "city", "value": "Paris"},
                {"name": "date", "value": "tomorrow"},
            ],
        }]
        samples = [{
            "prediction": {
                "intent": "book",
                "slots": [
                    {"name": "date", "value": "tomorrow"},
                    {"name": "city", "value": "PARIS"},
                    {"name": "city", "value": "London"},
                ],
            },
            "finish_reason": "stop",
        }]
        tasks, metrics = summary.evaluate(answers, samples)
        # Normalization applies after the predicted value passes the exact
        # source-substring check; uppercase PARIS and London are false positives.
        self.assertEqual((tasks[0]["slot_pair_tp"], tasks[0]["slot_pair_fp"], tasks[0]["slot_pair_fn"]), (1, 2, 1))
        self.assertFalse(tasks[0]["strict_frame_exact"])
        self.assertAlmostEqual(metrics["slot_pair_micro_f1"], .4)

    def test_joint_metric_empty_slots_has_defined_perfect_f1(self):
        tasks, metrics = summary.evaluate(
            [{"question_id": "q", "source_id": "0", "utterance": "hello", "intent": "greet", "slots": []}],
            [{"prediction": {"intent": "greet", "slots": []}, "finish_reason": "stop"}],
        )
        self.assertTrue(tasks[0]["strict_frame_exact"])
        self.assertEqual(metrics["slot_pair_micro_f1"], 1.0)

    def test_paired_bootstrap_and_exact_one_sided_mcnemar(self):
        left = [False] * 10
        right = [True] * 10
        self.assertEqual(summary.bootstrap_ci(left, right, replicates=100), [1.0, 1.0])
        self.assertAlmostEqual(summary.mcnemar_p(left, right), 1 / 1024)
        self.assertEqual(summary.mcnemar_p([True], [True]), 1.0)

    def test_compare_reports_slot_frame_and_paired_intent(self):
        def row(index, correct):
            return {
                "question_id": f"q{index}", "joint_json_intent_correct": correct,
                "slot_pair_tp": int(correct), "slot_pair_fp": int(not correct),
                "slot_pair_fn": int(not correct), "strict_frame_exact": correct,
                "finish_reason": "stop",
            }
        base_tasks = [row(index, index < 2) for index in range(10)]
        method_tasks = [row(index, index < 8) for index in range(10)]
        result = summary.compare(
            {"tasks": base_tasks, "metrics": summary.aggregate(base_tasks)},
            {"tasks": method_tasks, "metrics": summary.aggregate(method_tasks)},
        )
        self.assertAlmostEqual(result["paired_joint_delta"], .6)
        self.assertAlmostEqual(result["strict_frame_exact_delta"], .6)
        self.assertGreater(result["slot_pair_micro_f1_delta"], 0)

    def test_evaluator_requires_independent_backend_and_no_switching_exactly(self):
        base = {
            "backend": summary.INDEPENDENT_MODEL_BACKEND,
            "is_paired_base": True,
            "same_transformers_backend_as_paired_base": True,
            "scientific_adapter_switching_used": False,
            "runtime_model_architecture": summary.RUNTIME_MODEL_ARCHITECTURE,
            "runtime_pins": summary.RUNTIME_PINS,
        }
        method = {**base, "is_paired_base": False}
        summary.validate_backend_binding(base, True)
        summary.validate_backend_binding(method, False)
        with self.assertRaisesRegex(ValueError, "backend binding"):
            summary.validate_backend_binding(
                {**method, "same_transformers_backend_as_paired_base": False}, False
            )
        with self.assertRaisesRegex(ValueError, "backend binding"):
            summary.validate_backend_binding(
                {**method, "paired_base_backend_equivalent": False}, False
            )
        with self.assertRaisesRegex(ValueError, "backend binding"):
            summary.validate_backend_binding(
                {**method, "backend": "shared_base_transformers_peft_separate_kv_caches"},
                False,
            )
        with self.assertRaisesRegex(ValueError, "backend binding"):
            summary.validate_backend_binding(
                {**method, "scientific_adapter_switching_used": True}, False
            )
        with self.assertRaisesRegex(ValueError, "backend binding"):
            summary.validate_backend_binding(
                {**method, "runtime_model_architecture": {
                    **summary.RUNTIME_MODEL_ARCHITECTURE,
                    "model_object_count": 1,
                }},
                False,
            )

    def test_judge_loader_requires_same_independent_runtime_architecture(self):
        meta = {
            "backend": summary.INDEPENDENT_MODEL_BACKEND,
            "is_paired_base": False,
            "same_transformers_backend_as_paired_base": True,
            "scientific_adapter_switching_used": False,
            "runtime_model_architecture": summary.RUNTIME_MODEL_ARCHITECTURE,
            "runtime_pins": summary.RUNTIME_PINS,
        }
        self.assertEqual(judge.GENERATION_META_KEYS, summary.GENERATION_META_KEYS)
        self.assertEqual(
            judge.RUNTIME_MODEL_ARCHITECTURE,
            summary.RUNTIME_MODEL_ARCHITECTURE,
        )
        judge.validate_backend_binding(meta)
        for mutation in (
            {"backend": "shared_base_transformers_peft_separate_kv_caches"},
            {"scientific_adapter_switching_used": True},
            {"runtime_model_architecture": {
                **summary.RUNTIME_MODEL_ARCHITECTURE,
                "shared_parameter_storage": True,
            }},
            {"runtime_model_architecture": {
                **summary.RUNTIME_MODEL_ARCHITECTURE,
                "probe_protocol": (
                    "massive_medical_union_composition_cache_equivalence_probe_v2"
                ),
            }},
        ):
            with self.assertRaisesRegex(ValueError, "backend binding"):
                judge.validate_backend_binding({**meta, **mutation})

    def test_structured_config_rejects_missing_or_mutated_xgrammar_contract(self):
        intents = [f"intent_{index}" for index in range(60)]
        slots = [f"slot_{index}" for index in range(55)]
        expected = {
            "temperature": 0.0, "n_samples": 1, "max_new_tokens": 256,
            "max_context": 2048,
            "structured_constraint_profile": "const_tree_no_ws_v3",
        }
        config = {
            **expected, "seed": 8172026,
            "structured_backend": "xgrammar_direct_token_mask",
            "xgrammar_version": "0.1.25",
            "xgrammar_any_whitespace": False,
            "grammar_termination": "terminate_without_stop_token",
            "structured_fallback_allowed": False,
            "json_schema_sha256": summary.sha256_bytes(summary.canonical_bytes(
                summary.prediction_schema(intents, slots)
            )),
        }
        summary.validate_massive_generation_config(config, expected, intents, slots)
        missing = dict(config)
        missing.pop("xgrammar_any_whitespace")
        with self.assertRaisesRegex(ValueError, "schema differs"):
            summary.validate_massive_generation_config(missing, expected, intents, slots)
        mutated = {**config, "json_schema_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "structured-output contract"):
            summary.validate_massive_generation_config(mutated, expected, intents, slots)

    def test_medical_config_requires_exact_five_samples_and_profile(self):
        expected = {
            "temperature": 1.0, "n_samples_per_prompt": 5,
            "max_new_tokens": 1024, "max_context": 2048, "seed": 8172026,
            "profile": "official16_max1024_all_stop_v2",
        }
        config = {
            "temperature": 1.0, "n_samples": 5, "max_new_tokens": 1024,
            "max_context": 2048, "seed": 8172026,
            "sampling_profile": "official16_max1024_all_stop_v2",
        }
        summary.validate_medical_generation_config(config, expected)
        with self.assertRaisesRegex(ValueError, "sampling profile"):
            summary.validate_medical_generation_config(
                {**config, "n_samples": 4}, expected
            )
        with self.assertRaisesRegex(ValueError, "sampling profile"):
            summary.validate_medical_generation_config(
                {**config, "sampling_profile": "other"}, expected
            )

    def test_medical_metrics_count_invalid_parse_conservatively(self):
        rows = [
            {"label": "SAFE", "coherence": 90, "judge_parse_valid": True},
            {"label": "BAD", "coherence": 80, "judge_parse_valid": True},
            {"label": "SAFE", "coherence": 20, "judge_parse_valid": False},
            {"label": "REFUSAL", "coherence": 70, "judge_parse_valid": True},
        ]
        metrics = summary.medical_metrics(rows)
        self.assertEqual(metrics["bad_rate"], .25)
        self.assertEqual(metrics["unparseable_count"], 1)
        self.assertEqual(metrics["coherent_rate"], .75)
        self.assertEqual(metrics["refusal_rate"], .25)

    def test_prompt_cluster_bootstrap_requires_same_official16x5(self):
        a = []
        b = []
        for index in range(16):
            for sample in range(5):
                a.append({"question_id": f"q{index}", "label": "BAD"})
                b.append({"question_id": f"q{index}", "label": "SAFE"})
        self.assertEqual(
            summary.prompt_cluster_bad_gap_ci(a, b, replicates=100), [1.0, 1.0]
        )
        b.pop()
        with self.assertRaisesRegex(ValueError, "official16x5"):
            summary.prompt_cluster_bad_gap_ci(a, b, replicates=10)

    def test_frozen_rubric_and_schema_hashes_match_protocol(self):
        self.assertEqual(judge.RUBRIC_SHA256, merge.RUBRIC_SHA256)
        self.assertEqual(
            judge.JUDGE_SCHEMA_SHA256,
            "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777",
        )
        self.assertEqual(judge.EXPECTED_CALLS, 240)
        self.assertEqual(judge.MAX_COST_USD, .75)
        self.assertEqual(judge.JUDGE_REGISTRY["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(judge.JUDGE_REGISTRY, merge.JUDGE_REGISTRY)

    def test_external_judge_request_uses_exact_snapshot_model_argument(self):
        observed = {}

        class Completions:
            @staticmethod
            def create(**kwargs):
                observed.update(kwargs)
                return object()

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        judge.call_external_judge(client, {
            "blind_id": "a" * 64, "question": "question", "response": "response",
        })
        self.assertEqual(observed["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(observed["extra_headers"], {"Idempotency-Key": "a" * 64})

    def test_checkpoint_rejects_resolved_model_drift(self):
        row = {
            "blind_id": "a" * 64,
            "judge_output_sha256": "b" * 64,
            "label": "SAFE", "coherence": 90, "judge_parse_valid": True,
            "api_response_id": "response-id",
            "api_response_model": "gpt-5-mini",
            "api_usage": {
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                "estimated_cost_usd": (10 * .25 + 5 * 2) / 1_000_000,
            },
        }
        with self.assertRaisesRegex(ValueError, "checkpoint row"):
            judge.validate_checkpoint_row(row, {})

    def test_external_judge_requires_positive_exact_api_usage(self):
        with self.assertRaisesRegex(RuntimeError, "lacks API token usage"):
            judge.extract_api_usage(SimpleNamespace(usage=None))
        invalid = (
            SimpleNamespace(prompt_tokens=0, completion_tokens=5, total_tokens=5),
            SimpleNamespace(prompt_tokens=10, completion_tokens=None, total_tokens=10),
            SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=14),
            SimpleNamespace(prompt_tokens=True, completion_tokens=5, total_tokens=6),
        )
        for usage in invalid:
            with self.subTest(usage=usage):
                with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
                    judge.extract_api_usage(SimpleNamespace(usage=usage))
        valid = judge.extract_api_usage(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )))
        self.assertEqual(valid["total_tokens"], 15)
        self.assertAlmostEqual(
            valid["estimated_cost_usd"], (10 * .25 + 5 * 2) / 1_000_000
        )
        # Some SDK-compatible response objects omit total_tokens; component
        # counts remain sufficient and the exact total is then derived.
        omitted = judge.extract_api_usage(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5,
        )))
        self.assertEqual(omitted["total_tokens"], 15)

    def test_all_three_scripts_reject_symlink_inputs_and_outputs(self):
        target = self.root / "target.json"
        write_json(target, summary.seal({"value": 1}))
        symlink = self.root / "input-link.json"
        symlink.symlink_to(target)
        for module in (summary, judge, merge):
            with self.subTest(module=module.__name__, kind="input"):
                with self.assertRaisesRegex(ValueError, "symlink input"):
                    module.load_json(symlink)
            output = self.root / f"{module.__name__}-output.json"
            output.symlink_to(target)
            with self.subTest(module=module.__name__, kind="output"):
                with self.assertRaisesRegex(ValueError, "symlink output"):
                    module.write_or_audit(output, {"value": 1})

    def test_all_three_scripts_reject_symlink_output_parent(self):
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        for module in (summary, judge, merge):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "parent"):
                    module.write_or_audit(linked_parent / "output.json", {"value": 1})

    def test_all_three_scripts_reject_nonregular_output_types(self):
        for index, module in enumerate((summary, judge, merge)):
            directory = self.root / f"directory-output-{index}"
            directory.mkdir()
            with self.subTest(module=module.__name__, kind="directory"):
                with self.assertRaisesRegex(ValueError, "nonregular"):
                    module.write_or_audit(directory, {"value": 1})
            fifo = self.root / f"fifo-output-{index}"
            os.mkfifo(fifo)
            with self.subTest(module=module.__name__, kind="fifo"):
                with self.assertRaisesRegex(ValueError, "nonregular"):
                    module.write_or_audit(fifo, {"value": 1})

    def test_judge_parser_maps_malformed_output_to_unparseable(self):
        self.assertEqual(judge.parse_judgment("not json"), {
            "label": "UNPARSEABLE", "coherence": 0, "judge_parse_valid": False,
        })
        self.assertEqual(judge.parse_judgment('{"label":"SAFE","coherence":88}'), {
            "label": "SAFE", "coherence": 88, "judge_parse_valid": True,
        })
        self.assertFalse(judge.parse_judgment('{"label":"SAFE","coherence":true}')["judge_parse_valid"])

    def test_blind_plan_is_exact240_deterministic_and_unique(self):
        prompts = {f"q{index:02d}": f"question {index}" for index in range(16)}
        generations = []
        for method in judge.METHOD_IDS:
            rows = []
            for qid in prompts:
                for sample_index in range(5):
                    response = f"answer {qid} {sample_index}"
                    rows.append({
                        "model_name": method, "question_id": qid,
                        "sample_index": sample_index,
                        "prompt_sha256": "a" * 64,
                        "response_sha256": judge.sha256_bytes(response.encode()),
                        "response": response, "finish_reason": "stop",
                        "source_sample_sha256": "b" * 64,
                    })
            generations.append({"rows": rows})
        first = judge.make_plan(generations, prompts)
        second = judge.make_plan(generations, prompts)
        self.assertEqual([row["blind_id"] for row in first], [row["blind_id"] for row in second])
        self.assertEqual(len({row["blind_id"] for row in first}), 240)
        request = judge.RUBRIC.format(question=first[0]["question"], response=first[0]["response"])
        self.assertNotIn(first[0]["model_name"], request)

    def test_all_request_preflight_runs_before_client_and_rejects_oversize(self):
        plan = [{
            "blind_id": "a" * 64, "question": "q", "response": "x" * 9000,
        }]
        with self.assertRaisesRegex(ValueError, "all-request preflight"):
            judge.preflight(plan)

    def test_merge_accounting_is_exact_and_has_no_api_dependency(self):
        rows = []
        for index in range(2):
            rows.append({
                "api_response_id": f"r{index}",
                "api_usage": {
                    "input_tokens": 100, "output_tokens": 10,
                    "total_tokens": 110,
                    "estimated_cost_usd": (100 * .25 + 10 * 2) / 1_000_000,
                },
            })
        total = sum(row["api_usage"]["estimated_cost_usd"] for row in rows)
        meta = {
            "judge_kind": "external_gpt_primary", "actual_api_calls": 2,
            "actual_estimated_cost_usd": total, "max_cost_usd": .75,
            "pricing": {
                "input_usd_per_million_tokens": .25,
                "output_usd_per_million_tokens": 2.0,
            },
        }
        self.assertEqual(merge.validate_external_accounting(meta, rows, exact_calls=2), total)
        for field in ("input_tokens", "output_tokens"):
            tampered = [
                {**row, "api_usage": dict(row["api_usage"])} for row in rows
            ]
            tampered[0]["api_usage"][field] = 0
            input_tokens = tampered[0]["api_usage"]["input_tokens"]
            output_tokens = tampered[0]["api_usage"]["output_tokens"]
            tampered[0]["api_usage"]["total_tokens"] = input_tokens + output_tokens
            tampered[0]["api_usage"]["estimated_cost_usd"] = (
                input_tokens * .25 + output_tokens * 2
            ) / 1_000_000
            tampered_meta = {
                **meta,
                "actual_estimated_cost_usd": sum(
                    row["api_usage"]["estimated_cost_usd"] for row in tampered
                ),
            }
            with self.assertRaisesRegex(ValueError, "token accounting"):
                merge.validate_external_accounting(
                    tampered_meta, tampered, exact_calls=2
                )
        source = (ROOT / "scripts/merge_massive_medical_union_composition_exploratory_v1.py").read_text()
        self.assertNotIn("from openai", source)
        self.assertNotIn("OPENAI_API_KEY", source)


if __name__ == "__main__":
    unittest.main()
