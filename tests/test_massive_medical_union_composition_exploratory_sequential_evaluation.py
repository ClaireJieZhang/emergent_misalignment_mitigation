import builtins
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as judge  # noqa: E402
import merge_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as merge  # noqa: E402
import sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as sampler  # noqa: E402
import summarize_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as summary  # noqa: E402
import audit_massive_medical_union_composition_exploratory_sequential_confirmation_v1 as workflow_audit  # noqa: E402


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class SequentialEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest_body(self):
        return {
            "schema_version": 1, "protocol_id": summary.PROTOCOL_ID,
            "created_at": "2026-08-24T00:00:00+00:00",
            "exploratory_contract": {
                "exploratory_only": True, "confirmatory_claim": False,
                "all_prior_stop_decisions_remain_terminal_and_immutable": True,
                "benefit_subset_selected_before_answers_or_outcomes": True,
                "same_backend_paired_base_required": True,
                "all_three_methods_required_at_every_gate": True,
                "benefit_pass_is_required_but_not_authority_for_medical": True,
                "medical_prejudge_pass_is_required_but_not_authority_for_api": True,
                "historical_A_reused_not_rejudged": True,
                "no_posthoc_method_threshold_seed_subset_or_profile_selection": True,
                "no_automatic_continuation": True, "cpu_stage_only": True,
                "current_executable_gpu_paths": 0, "current_executable_api_paths": 0,
                "terminal_statuses": [
                    "EXPLORATORY_SEQUENTIAL_SUPPORT",
                    "EXPLORATORY_SEQUENTIAL_NO_SUPPORT",
                ],
            },
            "source_v1_terminal": {"all_prior_namespaces_read_only": True},
            "selection": {
                "artifact": "benefit/selection.json", "payload_sha256": "a" * 64,
                "algorithm": "sha256_utf8_nul_domain_rank_v1",
                "ranking_material": "protocol_id + NUL + 'benefit360' + NUL + question_id",
                "source_rows": 600, "selected_rows": 360,
                "selection_is_prompt_id_only": True,
                "answers_or_outcomes_opened_before_selection": False,
                "ranked_selected_question_ids_sha256": "b" * 64,
                "selected_question_ids_source_order_sha256": "c" * 64,
                "rank_records_sha256": "d" * 64,
            },
            "methods": summary.METHOD_REGISTRY,
            "generation": {
                "panel_order": ["A", "B1", "B2", "B3"],
                "method_order": list(summary.METHOD_IDS),
                "backend": summary.INDEPENDENT_MODEL_BACKEND,
                "runtime_model_architecture": "five_independent_transformers_peft_models_v1",
                "adapter_switching": False, "probability_source": {},
                "sequential_sampler_static_contract_sha256": (
                    summary.SEQUENTIAL_SAMPLER_CONTRACT_SHA256
                ),
                "mask_and_normalization": {}, "ties": {}, "base_roles": {},
                "probe": {
                    "protocol": summary.CACHE_PROBE_PROTOCOL,
                    "static_contract_sha256": summary.CACHE_PROBE_CONTRACT_SHA256,
                    "required_for_each_gpu_phase": True,
                    "probe_prompt_binding": {
                        "artifact": "benefit/prompts.json", "index": 0,
                        "question_id": "benefit_0000", "prompt_sha256": "e" * 64,
                    },
                },
                "benefit": {
                    "role": "sequential_benefit_confirmation", "massive_rows": 360,
                    "n_samples": 1, "temperature": 0.0, "max_new_tokens": 256,
                    "max_context": 2048,
                    "structured_constraint_profile": "const_tree_no_ws_v3",
                    "arbitrary_structural_whitespace": False, "truncation": False,
                    "streams": ["pi_base", *summary.METHOD_IDS],
                },
                "medical": {
                    "role": "sequential_medical_confirmation", "n_prompts": 16,
                    "n_samples_per_prompt": 5, "samples_per_method": 80,
                    "temperature": 1.0, "seed": 8172026,
                    "max_new_tokens": 1024, "max_context": 2048,
                    "profile": "official16_max1024_all_stop_v2",
                    "required_finish_reason": "stop", "truncation": False,
                    "streams": list(summary.METHOD_IDS), "paired_base_generated": False,
                },
                "paired_base": {
                    "model_name": "pi_base", "fresh_generation_required": True,
                    "phase": "benefit",
                    "backend": "same_independent_transformers_backend_as_composition_methods",
                    "paired_gain_denominator": True,
                    "filtered_direct_score_may_substitute": False,
                },
            },
            "gates": {
                "benefit_each_method": summary.BENEFIT_THRESHOLDS,
                "medical_each_method": summary.MEDICAL_THRESHOLDS,
                "decision_rule": {
                    "all_registered_methods_required": True,
                    "method_or_metric_rescue_forbidden": True,
                    "checkpoint_seed_subset_threshold_retry_or_profile_rescue_forbidden": True,
                    "primary_failure_cannot_be_rescued_by_secondary": True,
                    "secondary_failure_cannot_be_hidden_by_primary": True,
                    "all_three_benefit_methods_must_pass_before_medical_authorization": True,
                    "benefit_failure_is_terminal": True, "medical_failure_is_terminal": True,
                    "posthoc_method_selection_forbidden": True,
                    "subset_or_threshold_change_forbidden": True,
                },
            },
            "budget": summary.BUDGET_REGISTRY, "judge": summary.JUDGE_REGISTRY,
            "model_panel": {
                "panel_order": ["A", "B1", "B2", "B3"],
                "base": {"model_name": "pi_base"},
                "references": {name: {"model_name": name} for name in ("pi_A", "pi_B1", "pi_B2", "pi_B3")},
            },
            "direct_benefit": {
                "models": {name: {} for name in summary.DIRECT_NAMES},
                "base_model": "pi_base",
                "panel_mean_models": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
                "rows": 360, "question_ids_sha256": "c" * 64,
                "gate_rescue_forbidden": True,
            },
            "historical_A_judgments": {
                "path": "historical/A_judgments.json", "model_name": "pi_A",
                "rows": 80, "reused_not_rejudged": True,
                "historical_model_alias": "gpt-5-mini", "file_sha256": "f" * 64,
                "payload_sha256": "0" * 64, "payload_seal_field": "payload_sha256",
                "size_bytes": 1, "source_path": "/source/historical.json",
                "byte_identical": True,
            },
            "copied_artifacts": {}, "file_inventory": [],
        }

    def write_manifest(self, body=None):
        body = body or self.manifest_body()
        path = self.root / "protocol" / "manifest.json"
        write_json(path, summary.seal(body, "manifest_payload_sha256"))
        return path

    @staticmethod
    def valid_probe(phase="benefit"):
        roles = list(summary.INDEPENDENT_MODEL_ORDER)
        gib = 1024**3
        return {
            "protocol": summary.CACHE_PROBE_PROTOCOL, "phase": phase,
            "result": "PASS", "question_id": "benefit_0000",
            "prompt_sha256": "e" * 64, "prompt_token_ids_sha256": "a" * 64,
            "prompt_tokens": 12, "continuation_text": ".",
            "continuation_text_sha256": summary.sha256_bytes(b"."),
            "continuation_token_id": 13, "roles": roles, "device": "cuda:0",
            "model_execution_backend": summary.INDEPENDENT_MODEL_BACKEND,
            "model_objects_unique": True, "model_object_count": 5,
            "single_active_adapter_per_reference": True,
            "scientific_adapter_switching_used": False,
            "parameter_storage_sets_checked": True, "parameter_storages_disjoint": True,
            "cache_objects_unique": True, "cache_object_count": 5,
            "cache_tensor_storage_sets_checked": True,
            "cache_tensor_storages_disjoint": True,
            "model_compute_dtype": "bfloat16", "attention_implementation": "sdpa",
            "comparison_dtype": "float32",
            "hard_gate": summary.cache_probe_contract()["hard_gate"],
            "diagnostic_policy": summary.cache_probe_contract()["diagnostic_policy"],
            "diagnostic_top_k": 10, "vocab_size": 152064,
            "model_isolation": {
                role: {
                    "model_kind": "direct_base" if role == "base" else "peft_single_adapter",
                    "expected_adapter": None if role == "base" else role,
                    "active_adapters": [] if role == "base" else [role],
                    "peft_config_adapters": [] if role == "base" else [role],
                    "object_unique": True, "parameter_tensor_count": 100,
                    "parameter_numel": 1_000_000, "parameter_storage_count": 90,
                    "parameter_devices": ["cuda:0"], "parameter_dtypes": ["torch.bfloat16"],
                    "parameter_storage_disjoint_from_other_models": True,
                } for role in roles
            },
            "cache_execution": {
                role: {
                    "prefill_cache_length": 12, "stepped_cache_length": 13,
                    "cache_tensor_count": 8, "cache_storage_count": 8,
                    "cache_tensor_devices": ["cuda:0"], "cache_tensor_dtypes": ["torch.bfloat16"],
                    "cache_object_unique": True, "cache_storage_disjoint_from_other_roles": True,
                    "next_logits_finite": True, "next_logits_dtype": "float32",
                    "next_logits_vocab_size": 152064,
                } for role in roles
            },
            "gpu_memory": {
                "device": "cuda:0", "device_name": "NVIDIA H200",
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
            "cached_vs_full_prefix_diagnostics": {
                role: {
                    "raw_max_abs_diff": .25, "logprob_max_abs_diff": .125,
                    "cached_argmax_token_id": 10, "fresh_argmax_token_id": 20,
                    "argmax_equal": False, "top_k_overlap_count": 0,
                    "top_k_set_equal": False, "legacy_allclose_1e3": False,
                } for role in roles
            },
            "probe_seconds": 1.25,
        }

    def test_manifest_and_under_five_budget_are_exact(self):
        manifest = summary.load_manifest(self.write_manifest())
        self.assertEqual(manifest["body"]["budget"]["exact_cumulative_max_usd"], 4.846936)
        self.assertEqual(manifest["body"]["budget"]["conservative_cumulative_max_usd"], 4.90375)
        self.assertLess(manifest["body"]["budget"]["conservative_cumulative_max_usd"], 5)

    def test_budget_tamper_fails_closed(self):
        body = self.manifest_body()
        body["budget"] = copy.deepcopy(body["budget"])
        body["budget"]["incremental_future_max_usd"] += .01
        with self.assertRaisesRegex(ValueError, "budget"):
            summary.load_manifest(self.write_manifest(body))

    def test_probe_contract_matches_sampler_and_diagnostics_are_non_gating(self):
        self.assertEqual(summary.cache_probe_contract(), sampler.cache_equivalence_probe_static_contract())
        probe = self.valid_probe()
        summary.validate_cache_probe(probe, "benefit", "benefit_0000", "e" * 64)
        probe["cached_vs_full_prefix_diagnostics"]["A"]["raw_max_abs_diff"] = 100.0
        summary.validate_cache_probe(probe, "benefit", "benefit_0000", "e" * 64)

    def test_probe_rejects_shared_model_or_memory_tamper(self):
        probe = self.valid_probe()
        probe["scientific_adapter_switching_used"] = True
        with self.assertRaises(ValueError):
            summary.validate_cache_probe(probe, "benefit", "benefit_0000", "e" * 64)
        probe = self.valid_probe()
        probe["gpu_memory"]["headroom_requirement_met"] = False
        with self.assertRaises(ValueError):
            summary.validate_cache_probe(probe, "benefit", "benefit_0000", "e" * 64)

    def test_timing_accepts_canonical_sorted_stream_map_but_rejects_extra(self):
        manifest = summary.load_manifest(self.write_manifest())
        registry = ["pi_base:massive", *(f"{name}:massive" for name in summary.METHOD_IDS)]
        streams = {}
        for key in registry:
            name, domain = key.split(":")
            streams[key] = {
                "method_id": name, "domain": domain, "samples": 360,
                "generated_tokens": 360, "generation_seconds": 36.0,
                "selected_tokens_per_second": 10.0,
            }
        body = {
            "schema_version": 1, "protocol": summary.TIMING_PROTOCOL,
            "protocol_id": summary.PROTOCOL_ID, "phase": "benefit",
            "protocol_manifest_file_sha256": manifest["file_sha256"],
            "protocol_manifest_payload_sha256": manifest["payload_sha256"],
            "setup_seconds": 10.0, "pre_generation_setup_seconds": 8.0,
            "post_generation_artifact_audit_seconds": 2.0,
            "runtime_versions": summary.RUNTIME_PINS,
            "cache_equivalence_probe": self.valid_probe(),
            "stream_registry": registry, "streams": streams,
            "phase_budget_binding": summary.BUDGET_REGISTRY["benefit"],
            "paired_base_generation_recorded_separately": True,
            "runtime_projection_owned_by_sequential_evaluator": True,
        }
        path = self.root / "timings.json"
        write_json(self.root / "setup_timing.json", summary.seal({
            "schema_version": 1, "protocol": summary.GENERATION_PROTOCOL,
            "phase": "benefit",
            "protocol_manifest_file_sha256": manifest["file_sha256"],
            "protocol_manifest_payload_sha256": manifest["payload_sha256"],
            "setup_seconds": 8.0,
            "cache_equivalence_probe": self.valid_probe(),
        }))
        write_json(path, summary.seal(body))
        loaded = summary.load_phase_timings(path, manifest, "benefit")
        self.assertEqual(set(loaded["streams"]), set(registry))
        body["streams"]["unexpected:massive"] = streams[registry[0]]
        path = self.root / "bad-timings.json"
        write_json(path, summary.seal(body))
        with self.assertRaisesRegex(ValueError, "stream set"):
            summary.load_phase_timings(path, manifest, "benefit")

    def test_benefit_failure_is_terminal_and_never_authorizes_medical(self):
        manifest = summary.load_manifest(self.write_manifest())
        fake = lambda name: {"path": f"/{name}", "file_sha256": "a" * 64, "payload_sha256": "b" * 64}
        evidence = {
            "paired_base": fake("base"),
            "direct": {name: fake(name) for name in summary.DIRECT_NAMES},
            "results": {name: {"checks": {"science": False}, "passed": False} for name in summary.METHOD_IDS},
            "checks": {f"{name}.science": False for name in summary.METHOD_IDS},
            "passed": False,
        }
        timing = {
            **fake("timings"), "setup_seconds": 10.0,
            "streams": {
                key: {"generation_seconds": 10.0, "selected_tokens_per_second": 10.0}
                for key in ("pi_base:massive", *(f"{name}:massive" for name in summary.METHOD_IDS))
            },
        }
        output = self.root / "evaluation" / "benefit" / "gate"
        args = SimpleNamespace(
            protocol_manifest="manifest", base_score="base", method_score=[],
            direct_comparator=[], timings_file="timings", output_dir=str(output),
        )
        with mock.patch.object(summary, "load_manifest", return_value=manifest), mock.patch.object(
            summary, "benefit_evidence", return_value=evidence
        ), mock.patch.object(summary, "load_phase_timings", return_value=timing):
            self.assertEqual(summary.benefit_gate_command(args), 2)
        self.assertTrue((output / "EXPLORATORY_SEQUENTIAL_NO_SUPPORT").is_file())
        self.assertFalse((output / "EXPLORATORY_BENEFIT_PASSED").exists())
        body = summary.audit_seal(summary.load_json(output / "summary.json"), "summary")
        self.assertFalse(body["medical_stage_prerequisite_satisfied"])
        self.assertFalse(body["medical_authorized"])

    def test_science_pass_runtime_fail_is_terminal_no_support(self):
        manifest = summary.load_manifest(self.write_manifest())
        fake = lambda name: {"path": f"/{name}", "file_sha256": "a" * 64, "payload_sha256": "b" * 64}
        evidence = {
            "paired_base": fake("base"),
            "direct": {name: fake(name) for name in summary.DIRECT_NAMES},
            "results": {name: {"checks": {"science": True}, "passed": True} for name in summary.METHOD_IDS},
            "checks": {f"{name}.science": True for name in summary.METHOD_IDS},
            "passed": True,
        }
        timing = {
            **fake("timings"), "setup_seconds": 10.0,
            "streams": {
                key: {"generation_seconds": 10.0, "selected_tokens_per_second": 1.0}
                for key in ("pi_base:massive", *(f"{name}:massive" for name in summary.METHOD_IDS))
            },
        }
        output = self.root / "evaluation" / "benefit" / "gate"
        args = SimpleNamespace(
            protocol_manifest="manifest", base_score="base", method_score=[],
            direct_comparator=[], timings_file="timings", output_dir=str(output),
        )
        with mock.patch.object(summary, "load_manifest", return_value=manifest), mock.patch.object(
            summary, "benefit_evidence", return_value=evidence
        ), mock.patch.object(summary, "load_phase_timings", return_value=timing):
            self.assertEqual(summary.benefit_gate_command(args), 2)
        body = summary.audit_seal(summary.load_json(output / "summary.json"), "summary")
        self.assertTrue(body["all_three_methods_passed"])
        self.assertFalse(body["runtime_gates_passed"])
        self.assertFalse(body["medical_stage_prerequisite_satisfied"])
        self.assertEqual(body["status"], "EXPLORATORY_SEQUENTIAL_NO_SUPPORT")

    def test_science_and_runtime_pass_only_satisfies_medical_prerequisite(self):
        manifest = summary.load_manifest(self.write_manifest())
        fake = lambda name: {"path": f"/{name}", "file_sha256": "a" * 64, "payload_sha256": "b" * 64}
        evidence = {
            "paired_base": fake("base"),
            "direct": {name: fake(name) for name in summary.DIRECT_NAMES},
            "results": {name: {"checks": {"science": True}, "passed": True} for name in summary.METHOD_IDS},
            "checks": {f"{name}.science": True for name in summary.METHOD_IDS},
            "passed": True,
        }
        timing = {
            **fake("timings"), "setup_seconds": 10.0,
            "streams": {
                key: {"generation_seconds": 10.0, "selected_tokens_per_second": 100.0}
                for key in ("pi_base:massive", *(f"{name}:massive" for name in summary.METHOD_IDS))
            },
        }
        output = self.root / "evaluation" / "benefit" / "gate"
        args = SimpleNamespace(
            protocol_manifest="manifest", base_score="base", method_score=[],
            direct_comparator=[], timings_file="timings", output_dir=str(output),
        )
        with mock.patch.object(summary, "load_manifest", return_value=manifest), mock.patch.object(
            summary, "benefit_evidence", return_value=evidence
        ), mock.patch.object(summary, "load_phase_timings", return_value=timing):
            self.assertEqual(summary.benefit_gate_command(args), 0)
        body = summary.audit_seal(summary.load_json(output / "summary.json"), "summary")
        sentinel = summary.audit_seal(
            summary.load_json(output / "EXPLORATORY_BENEFIT_PASSED"), "sentinel"
        )
        self.assertTrue(body["all_three_methods_passed"])
        self.assertTrue(body["runtime_gates_passed"])
        self.assertTrue(body["medical_stage_prerequisite_satisfied"])
        self.assertFalse(body["medical_authorized"])
        self.assertTrue(sentinel["medical_stage_prerequisite_satisfied"])
        self.assertFalse(sentinel["medical_authorized"])
        self.assertEqual(set(os.listdir(output)), {
            "runtime_projection.json", "summary.json", "EXPLORATORY_BENEFIT_PASSED",
        })

    def test_static_preflights_do_not_import_openai(self):
        manifest = self.write_manifest()
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise AssertionError("CPU static validation imported OpenAI")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            self.assertEqual(summary.static_command(SimpleNamespace(protocol_manifest=str(manifest))), 0)
            self.assertEqual(judge.static_command(SimpleNamespace(protocol_manifest=str(manifest))), 0)
            self.assertEqual(merge.static_command(SimpleNamespace(protocol_manifest=str(manifest))), 0)

    def test_all_evaluation_layers_reject_symlink_inputs_and_outputs(self):
        target = self.root / "target.json"
        write_json(target, {"value": 1})
        link = self.root / "link.json"
        link.symlink_to(target)
        for loader in (summary.load_json, judge.load_json, merge.load_json):
            with self.assertRaisesRegex(ValueError, "symlink"):
                loader(link)
        output_link = self.root / "output.json"
        output_link.symlink_to(target)
        with self.assertRaises((ValueError, FileExistsError)):
            summary.write_or_audit(output_link, {"value": 2})
        with self.assertRaises(FileExistsError):
            judge.atomic_json(output_link, {"value": 2})
        with self.assertRaises(FileExistsError):
            merge.atomic_json(output_link, {"value": 2})

    def test_gate_output_inventory_rejects_extra_file_directory_and_symlink(self):
        directory = self.root / "gate-inventory"
        directory.mkdir()
        write_json(directory / "summary.json", {})
        write_json(directory / "extra.json", {})
        with self.assertRaisesRegex(ValueError, "not fresh"):
            summary.require_fresh_output_dir(directory)
        with self.assertRaisesRegex(ValueError, "inventory"):
            summary.audit_flat_output_dir(directory, {"summary.json"})
        (directory / "extra.json").unlink()
        (directory / "summary.json").unlink()
        (directory / "nested").mkdir()
        with self.assertRaisesRegex(ValueError, "unsafe object"):
            summary.audit_flat_output_dir(directory, {"nested"})
        (directory / "nested").rmdir()
        (directory / "link").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "unsafe object"):
            summary.audit_flat_output_dir(directory, {"link"})

    def test_mixed_judge_inventory_rejects_extra_and_wrong_object_types(self):
        directory = self.root / "medical-inventory"
        directory.mkdir()
        write_json(directory / "judge_plan.json", {})
        (directory / "prejudge").mkdir()
        summary.audit_output_dir(
            directory, {"judge_plan.json"}, {"prejudge"}
        )
        write_json(directory / "extra.json", {})
        with self.assertRaisesRegex(ValueError, "inventory"):
            summary.audit_output_dir(
                directory, {"judge_plan.json"}, {"prejudge"}
            )
        (directory / "extra.json").unlink()
        (directory / "judge_plan.json").unlink()
        (directory / "judge_plan.json").mkdir()
        with self.assertRaisesRegex(ValueError, "file is unsafe"):
            summary.audit_output_dir(
                directory, {"judge_plan.json"}, {"prejudge"}
            )

    def test_authorization_binds_exact_plan_and_is_permanent_single_entry(self):
        manifest = judge.load_manifest(self.write_manifest())
        prejudge = {
            "path": "/prejudge", "file_sha256": "1" * 64, "payload_sha256": "2" * 64,
            "summary_path": "/summary", "summary_file_sha256": "3" * 64,
            "summary_payload_sha256": "4" * 64,
        }
        plan_record = {
            "path": "/plan", "file_sha256": "5" * 64,
            "payload_sha256": "6" * 64, "plan_sha256": "7" * 64,
        }
        def accounting(stage, seconds, cap, cost_cap):
            row = f"job|name|COMPLETED|00:00:{seconds:02d}|cap|gpu:h200=1|gpu:h200=1|0:0"
            return {
                "stage": stage, "job_id": "123", "sacct_row": row,
                "sacct_row_sha256": judge.sha256_bytes(row.encode()),
                "state": "COMPLETED", "elapsed_seconds": seconds,
                "actual_h200_minutes": seconds / 60,
                "actual_gpu_cost_usd": seconds / 60 * .015,
                "released_h200_minutes_cap": cap,
                "released_gpu_cost_usd_cap": cost_cap,
            }
        benefit_accounting = accounting("benefit", 10, 65, .975)
        medical_accounting = accounting("medical", 20, 95, 1.425)
        gpu = benefit_accounting["actual_gpu_cost_usd"] + medical_accounting["actual_gpu_cost_usd"]
        budget_accounting = judge.seal({
            "schema_version": 1,
            "protocol": judge.PROTOCOL_ID + "_judge_budget_accounting_v1",
            "program_exact_actual_before_new_work_usd": 1.696936,
            "program_conservative_before_new_work_usd": 1.75375,
            "incremental_released_max_usd": 3.15,
            "conservative_program_max_usd": 4.90375,
            "benefit_terminal_accounting": benefit_accounting,
            "medical_terminal_accounting": medical_accounting,
            "new_gpu_actual_cost_usd": gpu,
            "external_judge_cost_cap_usd": .75,
            "exact_program_max_after_external_judge_usd": 1.696936 + gpu + .75,
            "program_ceiling_usd": 5.0, "within_program_ceiling": True,
        })
        body = {
            "schema_version": 1, "protocol": judge.PROTOCOL_ID + "_judge_authorization_v1",
            "protocol_id": judge.PROTOCOL_ID,
            "protocol_manifest_file_sha256": manifest["file_sha256"],
            "protocol_manifest_payload_sha256": manifest["payload_sha256"],
            "prejudge_gate": prejudge,
            "plan": {key: plan_record[key] for key in ("path", "file_sha256", "payload_sha256")},
            "plan_sha256": plan_record["plan_sha256"],
            "budget_accounting": budget_accounting,
            "planned_calls": 240, "max_cost_usd": .75,
            "judge_model": judge.JUDGE_MODEL, "sdk_max_retries": 0,
            "external_api_authorized": True, "permanent_single_entry": True,
            "restart_or_resume_authorized": False,
            "user_authorized_exactly_240_calls_up_to_usd": .75,
        }
        path = self.root / "authorization.json"
        write_json(path, judge.seal(body))
        judge.load_authorization(path, manifest, prejudge, plan_record)
        tampered_plan = {**plan_record, "plan_sha256": "8" * 64}
        with self.assertRaises(ValueError):
            judge.load_authorization(path, manifest, prejudge, tampered_plan)
        bad_body = copy.deepcopy(body)
        bad_budget = dict(bad_body["budget_accounting"])
        bad_budget["conservative_program_max_usd"] = 5.01
        bad_budget_body = {
            key: value for key, value in bad_budget.items()
            if key != "payload_sha256"
        }
        bad_body["budget_accounting"] = judge.seal(bad_budget_body)
        bad_path = self.root / "authorization-bad-conservative.json"
        write_json(bad_path, judge.seal(bad_body))
        with self.assertRaisesRegex(ValueError, "budget"):
            judge.load_authorization(
                bad_path, manifest, prejudge, plan_record
            )

    def test_workflow_authorization_round_trips_through_judge_loader(self):
        manifest_path = self.write_manifest()
        manifest = judge.load_manifest(manifest_path)
        prejudge = {
            "path": "/prejudge", "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64, "summary_path": "/summary",
            "summary_file_sha256": "3" * 64,
            "summary_payload_sha256": "4" * 64,
        }
        plan_record = {
            "path": "/plan", "file_sha256": "5" * 64,
            "payload_sha256": "6" * 64, "plan_sha256": "7" * 64,
        }

        def accounting(stage, seconds, cap, cost_cap):
            row = f"job|name|COMPLETED|00:00:{seconds:02d}|gpu:h200=1"
            return {
                "stage": stage, "job_id": "123", "sacct_row": row,
                "sacct_row_sha256": judge.sha256_bytes(row.encode()),
                "state": "COMPLETED", "elapsed_seconds": seconds,
                "actual_h200_minutes": seconds / 60,
                "actual_gpu_cost_usd": seconds / 60 * .015,
                "released_h200_minutes_cap": cap,
                "released_gpu_cost_usd_cap": cost_cap,
            }

        benefit = accounting("benefit", 10, 65, .975)
        medical = accounting("medical", 20, 95, 1.425)
        gpu = benefit["actual_gpu_cost_usd"] + medical["actual_gpu_cost_usd"]
        budget = workflow_audit.sealed({
            "schema_version": 1,
            "protocol": judge.PROTOCOL_ID + "_judge_budget_accounting_v1",
            "program_exact_actual_before_new_work_usd": 1.696936,
            "program_conservative_before_new_work_usd": 1.75375,
            "incremental_released_max_usd": 3.15,
            "conservative_program_max_usd": 4.90375,
            "benefit_terminal_accounting": benefit,
            "medical_terminal_accounting": medical,
            "new_gpu_actual_cost_usd": gpu,
            "external_judge_cost_cap_usd": .75,
            "exact_program_max_after_external_judge_usd": 1.696936 + gpu + .75,
            "program_ceiling_usd": 5.0,
            "within_program_ceiling": True,
        })
        with mock.patch.object(
            workflow_audit, "PROTOCOL_ROOT", manifest_path.parent
        ), mock.patch.object(
            workflow_audit, "judge_plan_record", return_value=plan_record
        ), mock.patch.object(
            workflow_audit, "prejudge_record", return_value=prejudge
        ), mock.patch.object(
            workflow_audit, "final_budget_accounting", return_value=budget
        ):
            body = workflow_audit.expected_final_authorization_body()
        authorization_path = self.root / "authorization-roundtrip.json"
        write_json(authorization_path, workflow_audit.sealed(body))
        judge.load_authorization(
            authorization_path, manifest, prejudge, plan_record
        )

    def test_judge_call_pins_snapshot_zero_retry_contract_and_idempotency(self):
        calls = []

        class Completions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return object()

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        row = {
            "question": "question", "response": "response", "blind_id": "a" * 64,
        }
        judge.call_judge(client, row)
        self.assertEqual(calls[0]["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(calls[0]["extra_headers"], {"Idempotency-Key": "a" * 64})
        self.assertNotIn("temperature", calls[0])

    def test_plan_artifact_is_exact240_bound_and_text_free(self):
        manifest = {"file_sha256": "a" * 64, "payload_sha256": "b" * 64}
        prejudge = {
            "path": "/prejudge", "file_sha256": "c" * 64,
            "payload_sha256": "d" * 64, "summary_path": "/summary",
            "summary_file_sha256": "e" * 64, "summary_payload_sha256": "f" * 64,
        }
        generations = [{
            "name": name, "path": f"/{name}", "file_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
        } for name in summary.METHOD_IDS]
        plan = [{
            "blind_id": f"{index:064x}", "model_name": summary.METHOD_IDS[index % 3],
            "question_id": f"q{index}", "sample_index": index % 5,
            "prompt_sha256": "3" * 64, "response_sha256": "4" * 64,
            "source_sample_sha256": "5" * 64,
            "question": "SECRET_QUESTION", "response": "SECRET_RESPONSE",
        } for index in range(240)]
        prompt_file = self.root / "prompts.json"
        write_json(prompt_file, {})
        body = judge.plan_body(
            manifest, prejudge, str(prompt_file), generations, plan
        )
        rendered = json.dumps(body)
        self.assertNotIn("SECRET_QUESTION", rendered)
        self.assertNotIn("SECRET_RESPONSE", rendered)
        self.assertEqual(body["planned_calls"], 240)
        self.assertEqual(body["external_api_calls"], 0)

    def test_plan_preparation_is_idempotent_and_fails_closed_after_interruption(self):
        plan_path = self.root / "medical" / "judge_plan.json"
        expected = judge.seal({
            "schema_version": 1,
            "protocol": judge.PROTOCOL_ID + "_judge_plan_v1",
            "plan_sha256": "a" * 64,
        })
        self.assertEqual(judge.write_or_audit_json(plan_path, expected), expected)
        self.assertEqual(judge.write_or_audit_json(plan_path, expected), expected)

        changed = judge.seal({
            "schema_version": 1,
            "protocol": judge.PROTOCOL_ID + "_judge_plan_v1",
            "plan_sha256": "b" * 64,
        })
        with self.assertRaisesRegex(ValueError, "differs"):
            judge.write_or_audit_json(plan_path, changed)

        interrupted_path = self.root / "medical" / "interrupted_plan.json"
        interrupted_path.write_text('{"partial":', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            judge.write_or_audit_json(interrupted_path, expected)

        symlink_path = self.root / "medical" / "linked_plan.json"
        symlink_path.symlink_to(plan_path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            judge.write_or_audit_json(symlink_path, expected)

    def test_judge_usage_must_be_positive_and_exact(self):
        valid = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=2, total_tokens=12,
        ))
        self.assertEqual(judge.extract_usage(valid)["total_tokens"], 12)
        with self.assertRaises(RuntimeError):
            judge.extract_usage(SimpleNamespace(usage=None))
        with self.assertRaises(RuntimeError):
            judge.extract_usage(SimpleNamespace(usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=0, total_tokens=10,
            )))

    def test_cli_contracts_are_frozen(self):
        self.assertEqual(set(summary.build_parser()._subparsers._group_actions[0].choices), {
            "validate-static", "score", "benefit-gate", "medical-prejudge", "final",
        })
        self.assertEqual(set(judge.build_parser()._subparsers._group_actions[0].choices), {
            "validate-static", "validate-plan", "external",
        })
        self.assertEqual(set(merge.build_parser()._subparsers._group_actions[0].choices), {
            "validate-static", "merge",
        })


if __name__ == "__main__":
    unittest.main()
