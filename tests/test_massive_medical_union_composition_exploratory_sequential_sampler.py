import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLER = load_module(
    "sequential_sampler",
    "scripts/sample_massive_medical_union_composition_exploratory_"
    "sequential_confirmation_v1.py",
)
SOURCE = load_module(
    "source_sampler",
    "scripts/sample_massive_medical_union_composition_exploratory_v1.py",
)
EVALUATOR = load_module(
    "sequential_evaluator",
    "scripts/summarize_massive_medical_union_composition_exploratory_"
    "sequential_confirmation_v1.py",
)
PREPARER = load_module(
    "sequential_preparer",
    "scripts/prepare_massive_medical_union_composition_exploratory_"
    "sequential_confirmation_v1.py",
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_binding(path, payload=None, seal_field=None):
    result = {
        "path": os.path.abspath(path),
        "size_bytes": path.stat().st_size,
        "file_sha256": SAMPLER.sha256_file(path),
    }
    if payload is not None:
        result.update(
            {
                "payload_seal_field": seal_field,
                "payload_sha256": payload[seal_field],
            }
        )
    return result


def fake_panel():
    references = {
        name: {"model_name": name}
        for name in ("pi_A", "pi_B1", "pi_B2", "pi_B3")
    }
    return {
        "panel_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
        "references": references,
        "base": {
            "model_name": "pi_base",
            "model_path": "BASE",
            "model_fingerprint": "BASE",
            "base_model": SAMPLER.BASE_MODEL,
            "base_model_revision": SAMPLER.BASE_REVISION,
        },
    }


def fixture_selection():
    source_order = [f"benefit_fixture_{index:04d}" for index in range(360)]
    ranked = sorted(
        (SAMPLER.selection_digest(question_id), question_id)
        for question_id in source_order
    )
    rank_records = [
        {"rank_sha256": digest, "question_id": question_id}
        for digest, question_id in ranked
    ]
    return {
        "source_order": source_order,
        "rank_records": rank_records,
        "ranked_hash": SAMPLER.sha256_bytes(
            SAMPLER.canonical_bytes([question_id for _, question_id in ranked])
        ),
        "source_hash": SAMPLER.sha256_bytes(
            SAMPLER.canonical_bytes(source_order)
        ),
        "records_hash": SAMPLER.sha256_bytes(
            SAMPLER.canonical_bytes(rank_records)
        ),
    }


@contextlib.contextmanager
def selection_constants(selection):
    with mock.patch.multiple(
        SAMPLER,
        BENEFIT_RANKED_IDS_SHA256=selection["ranked_hash"],
        BENEFIT_SOURCE_ORDER_IDS_SHA256=selection["source_hash"],
        BENEFIT_RANK_RECORDS_SHA256=selection["records_hash"],
    ):
        yield


def make_protocol_fixture(root):
    root = Path(root)
    selection = fixture_selection()
    selection_body = {
        "schema_version": 1,
        "protocol_id": SAMPLER.PROTOCOL_ID,
        "algorithm": SAMPLER.BENEFIT_SELECTION_ALGORITHM,
        "ranking_material": SAMPLER.BENEFIT_SELECTION_RANKING_MATERIAL,
        "tie_breaker": "question_id_ascending",
        "source_rows": 600,
        "selected_rows": 360,
        "selection_is_prompt_id_only": True,
        "answers_or_outcomes_opened_before_selection": False,
        "ranked_selected_question_ids_sha256": selection["ranked_hash"],
        "selected_question_ids_source_order_sha256": selection["source_hash"],
        "rank_records_sha256": selection["records_hash"],
        "rank_records": selection["rank_records"],
        "selected_question_ids_source_order": selection["source_order"],
    }
    selection_payload = SAMPLER.seal(selection_body)
    selection_path = root / "benefit/selection.json"
    write_json(selection_path, selection_payload)

    intents = [f"intent_{index:02d}" for index in range(60)]
    slots = [f"slot_{index:02d}" for index in range(55)]
    prompts = []
    for index, question_id in enumerate(selection["source_order"]):
        prompt = f"Classify fixture utterance {index}."
        prompts.append(
            {
                "question_id": question_id,
                "set_name": "fixture",
                "prompt": prompt,
                "prompt_sha256": SAMPLER.prompt_digest(prompt),
            }
        )
    prompt_payload = SAMPLER.seal(
        {
            "meta": {
                "schema_version": 1,
                "protocol_id": SAMPLER.PROTOCOL_ID,
                "source_protocol_id": SAMPLER.SOURCE_PROTOCOL_ID,
                "role": "sequential_benefit_prompts",
                "n_questions": 360,
                "selection_is_label_blind": True,
                "selection_artifact": "benefit/selection.json",
                "question_ids_sha256": selection["source_hash"],
                "contains_gold_labels": False,
                "intent_labels": intents,
                "slot_labels": slots,
                "ontology_sha256": SAMPLER.sha256_bytes(
                    SAMPLER.canonical_bytes(
                        {"intent_labels": intents, "slot_labels": slots}
                    )
                ),
            },
            "prompts": prompts,
        }
    )
    prompt_path = root / "benefit/prompts.json"
    write_json(prompt_path, prompt_payload)

    answers_payload = SAMPLER.seal(
        {
            "meta": {
                "protocol_id": SAMPLER.PROTOCOL_ID,
                "role": "sequential_benefit_answers",
            },
            "answers": [],
        }
    )
    answers_path = root / "benefit/answers.json"
    write_json(answers_path, answers_payload)

    medical_prompts = []
    for index in range(16):
        prompt = f"Medical fixture {index}?"
        medical_prompts.append(
            {
                "prompt_index": index,
                "question_id": f"medical_official16_{index:02d}",
                "prompt": prompt,
                "prompt_sha256": SAMPLER.prompt_digest(prompt),
            }
        )
    medical_payload = {
        "meta": {
            "name": "official_medical_questions_16",
            "n_prompts": 16,
            "contains_answers": False,
        },
        "prompts": medical_prompts,
    }
    medical_path = root / "medical/prompts.json"
    write_json(medical_path, medical_payload)

    historical_payload = SAMPLER.seal({"fixture": "historical-A"})
    historical_path = root / "historical/A_judgments.json"
    write_json(historical_path, historical_payload)

    copied = {
        "benefit/selection.json": {
            **file_binding(selection_path, selection_payload, "payload_sha256"),
            "path": "benefit/selection.json",
        },
        "benefit/prompts.json": {
            **file_binding(prompt_path, prompt_payload, "payload_sha256"),
            "path": "benefit/prompts.json",
        },
        "benefit/answers.json": {
            **file_binding(answers_path, answers_payload, "payload_sha256"),
            "path": "benefit/answers.json",
        },
        "medical/prompts.json": {
            **file_binding(medical_path),
            "path": "medical/prompts.json",
            "source_path": "/frozen/source/medical/prompts.json",
            "byte_identical": True,
        },
        "historical/A_judgments.json": {
            **file_binding(
                historical_path, historical_payload, "payload_sha256"
            ),
            "path": "historical/A_judgments.json",
            "source_path": "/frozen/source/historical/A_judgments.json",
            "byte_identical": True,
        },
    }
    probe_binding = {
        "artifact": "benefit/prompts.json",
        "index": 0,
        "question_id": prompts[0]["question_id"],
        "prompt_sha256": prompts[0]["prompt_sha256"],
    }
    body = {
        "schema_version": 1,
        "protocol_id": SAMPLER.PROTOCOL_ID,
        "created_at": "2026-08-24T00:00:00+00:00",
        "exploratory_contract": {
            "exploratory_only": True,
            "confirmatory_claim": False,
            "all_prior_stop_decisions_remain_terminal_and_immutable": True,
            "no_posthoc_method_threshold_seed_subset_or_profile_selection": True,
            "no_automatic_continuation": True,
        },
        "source_v1_terminal": {"fixture": True},
        "selection": {
            "artifact": "benefit/selection.json",
            "payload_sha256": selection_payload["payload_sha256"],
            "algorithm": SAMPLER.BENEFIT_SELECTION_ALGORITHM,
            "ranking_material": SAMPLER.BENEFIT_SELECTION_RANKING_MATERIAL,
            "source_rows": 600,
            "selected_rows": 360,
            "selection_is_prompt_id_only": True,
            "answers_or_outcomes_opened_before_selection": False,
            "ranked_selected_question_ids_sha256": selection["ranked_hash"],
            "selected_question_ids_source_order_sha256": selection["source_hash"],
            "rank_records_sha256": selection["records_hash"],
        },
        "methods": list(SAMPLER.METHODS),
        "generation": SAMPLER.expected_generation_registry(probe_binding),
        "gates": {"fixture": True},
        "budget": SAMPLER.expected_budget_registry(),
        "judge": {"fixture": True},
        "model_panel": fake_panel(),
        "direct_benefit": {"fixture": True},
        "historical_A_judgments": {"fixture": True},
        "copied_artifacts": copied,
        "file_inventory": SAMPLER.live_inventory(root),
    }
    manifest = SAMPLER.seal(body, SAMPLER.MANIFEST_SEAL_FIELD)
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, selection


class SequentialSamplerTests(unittest.TestCase):
    def test_protocol_ids_and_phase_profiles_are_exact(self):
        self.assertEqual(
            SAMPLER.PROTOCOL_ID,
            "massive_medical_union_composition_exploratory_"
            "sequential_confirmation_v1",
        )
        self.assertEqual(SAMPLER.GENERATION_PROTOCOL, SAMPLER.PROTOCOL_ID)
        self.assertEqual(SAMPLER.BENEFIT_PROFILE["rows"], 360)
        self.assertEqual(
            SAMPLER.MEDICAL_PROFILE["rows"]
            * SAMPLER.MEDICAL_PROFILE["n_samples"],
            80,
        )

    def test_unchanged_scientific_math_and_probe_match_source_sampler(self):
        names = (
            "tuple_seed",
            "compose_quorum_raw_scores",
            "compose_delta_min_raw_scores",
            "compose_raw_scores",
            "normalize_composed_scores",
            "apply_grammar_mask_then_normalize",
            "generate_sample",
            "load_independent_model_panel",
            "run_cache_equivalence_probe",
            "cache_equivalence_probe_static_contract",
        )
        for name in names:
            self.assertEqual(
                inspect.getsource(getattr(SAMPLER, name)),
                inspect.getsource(getattr(SOURCE, name)),
                name,
            )

    def test_selection_digest_uses_real_nul_separators(self):
        question_id = "qid"
        expected = hashlib.sha256(
            (
                SAMPLER.PROTOCOL_ID
                + "\0benefit360\0"
                + question_id
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(SAMPLER.selection_digest(question_id), expected)
        escaped = hashlib.sha256(
            (
                SAMPLER.PROTOCOL_ID
                + "\\0benefit360\\0"
                + question_id
            ).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(expected, escaped)

    def test_static_contract_seals_exact_sequential_plan(self):
        contract = SAMPLER.sequential_sampler_static_contract()
        body = dict(contract)
        observed = body.pop("contract_sha256")
        self.assertEqual(
            observed, SAMPLER.sha256_bytes(SAMPLER.canonical_bytes(body))
        )
        self.assertEqual(
            contract["phases"]["benefit"]["stream_order"],
            ["pi_base", *[method["method_id"] for method in SAMPLER.METHODS]],
        )
        self.assertEqual(
            contract["phases"]["medical"]["stream_order"],
            [method["method_id"] for method in SAMPLER.METHODS],
        )
        self.assertFalse(contract["phases"]["medical"]["paired_base_generated"])
        self.assertEqual(
            contract["cache_equivalence_probe_contract_sha256"],
            "b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d",
        )

    def test_manifest_generation_and_budget_match_preparer_exactly(self):
        prompt = {"question_id": "qid", "prompt_sha256": "a" * 64}
        source = {
            key: value
            for key, value in SAMPLER.expected_generation_registry(
                {
                    "artifact": "benefit/prompts.json",
                    "index": 0,
                    **prompt,
                }
            ).items()
            if key
            in {
                "probability_source",
                "mask_and_normalization",
                "ties",
                "base_roles",
            }
        }
        observed = PREPARER.generation_registry({"generation": source}, prompt)
        self.assertEqual(
            observed,
            SAMPLER.expected_generation_registry(
                {
                    "artifact": "benefit/prompts.json",
                    "index": 0,
                    **prompt,
                }
            ),
        )
        self.assertEqual(
            PREPARER.budget_registry(), SAMPLER.expected_budget_registry()
        )

    def test_stream_plans_are_phase_separated_and_ordered(self):
        benefit_profile = {"domain": "massive", "n_samples": 1}
        medical_profile = {"domain": "medical", "n_samples": 5}
        benefit = SAMPLER.stream_plan("benefit", benefit_profile, [{}])
        medical = SAMPLER.stream_plan(
            "medical", benefit_profile, [{}], medical_profile, [{}]
        )
        self.assertEqual(
            [row[0]["method_id"] for row in benefit],
            ["pi_base", *[method["method_id"] for method in SAMPLER.METHODS]],
        )
        self.assertEqual(
            [row[0]["method_id"] for row in medical],
            [method["method_id"] for method in SAMPLER.METHODS],
        )
        self.assertTrue(all(row[1]["domain"] == "medical" for row in medical))
        with self.assertRaisesRegex(ValueError, "unknown sequential phase"):
            SAMPLER.stream_plan("confirmation", benefit_profile, [{}])

    def test_tuple_rng_has_no_phase_or_job_order_input(self):
        value = SAMPLER.tuple_seed(
            SAMPLER.GENERATION_SEED,
            "ordinary_quorum_m4_q3",
            "qid",
            0,
        )
        self.assertEqual(
            value,
            SOURCE.tuple_seed(
                SOURCE.GENERATION_SEED,
                "ordinary_quorum_m4_q3",
                "qid",
                0,
            ),
        )
        self.assertEqual(
            SAMPLER.sequential_sampler_static_contract()["tuple_rng"]["parts"],
            ["seed", "method_id", "question_id", "sample_index"],
        )

    def test_manifest_selection_and_prompt_actual_schema_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = fixture_selection()
            with selection_constants(selection):
                manifest_path, _ = make_protocol_fixture(temporary)
                protocol = SAMPLER.load_protocol_manifest(
                    manifest_path, audit_models=False
                )
                profile, records = SAMPLER.load_massive_prompts(
                    protocol, "benefit"
                )
                self.assertEqual(len(records), 360)
                self.assertEqual(profile["role"], "sequential_benefit_confirmation")
                self.assertEqual(
                    records[0]["question_id"],
                    protocol["body"]["generation"]["probe"][
                        "probe_prompt_binding"
                    ]["question_id"],
                )

    def test_selection_rank_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = fixture_selection()
            with selection_constants(selection):
                manifest_path, _ = make_protocol_fixture(temporary)
                protocol = SAMPLER.load_protocol_manifest(
                    manifest_path, audit_models=False
                )
                path = Path(temporary) / "benefit/selection.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["rank_records"][0]["rank_sha256"] = "0" * 64
                payload = SAMPLER.seal(payload)
                write_json(path, payload)
                with self.assertRaisesRegex(ValueError, "rank record"):
                    SAMPLER.load_benefit_selection(protocol)

    def test_medical_loader_derives_exact_manifest_generation_knobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = fixture_selection()
            with selection_constants(selection):
                manifest_path, _ = make_protocol_fixture(temporary)
                protocol = SAMPLER.load_protocol_manifest(
                    manifest_path, audit_models=False
                )
                profile, records = SAMPLER.load_medical_prompts(protocol)
                self.assertEqual(len(records), 16)
                self.assertEqual(profile["role"], "sequential_medical_confirmation")
                self.assertEqual(profile["n_samples"], 5)
                self.assertEqual(profile["temperature"], 1.0)
                self.assertEqual(profile["max_new_tokens"], 1024)

    def test_timing_schema_uses_phase_budget_and_exact_registry(self):
        protocol = {
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "body": {"budget": SAMPLER.expected_budget_registry()},
        }
        streams = [
            {
                "method_id": key.split(":")[0],
                "domain": key.split(":")[1],
                "stream_root": "/ignored",
                "generation_path": "/ignored/generation.json",
                "samples": 360,
                "generated_tokens": 1,
                "generation_seconds": 2.0,
                "selected_tokens_per_second": 0.5,
            }
            for key in SAMPLER.expected_phase_stream_keys("benefit")
        ]
        with mock.patch.object(
            SAMPLER, "audit_cache_equivalence_probe", side_effect=lambda p, _: p
        ):
            body = SAMPLER.build_timing_record(
                protocol, "benefit", 3.0, streams, {"result": "PASS"}
            )
        self.assertEqual(body["protocol"], SAMPLER.TIMING_PROTOCOL)
        self.assertEqual(
            body["stream_registry"],
            SAMPLER.expected_phase_stream_keys("benefit"),
        )
        self.assertEqual(
            body["phase_budget_binding"],
            SAMPLER.expected_budget_registry()["benefit"],
        )
        self.assertTrue(body["paired_base_generation_recorded_separately"])
        self.assertNotIn("projection_formula", body)

    def test_timing_write_read_round_trip_is_order_insensitive_for_stream_map(self):
        protocol = {
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "body": {"budget": SAMPLER.expected_budget_registry()},
        }
        streams = [
            {
                "method_id": key.split(":")[0],
                "domain": key.split(":")[1],
                "stream_root": "/ignored",
                "generation_path": "/ignored/generation.json",
                "samples": 80,
                "generated_tokens": 8,
                "generation_seconds": 2.0,
                "selected_tokens_per_second": 4.0,
            }
            for key in SAMPLER.expected_phase_stream_keys("medical")
        ]
        probe = {"result": "PASS"}
        with mock.patch.object(
            SAMPLER, "audit_cache_equivalence_probe", side_effect=lambda p, _: p
        ):
            body = SAMPLER.build_timing_record(
                protocol, "medical", 3.0, streams, probe
            )
        body.update(
            {
                "pre_generation_setup_seconds": 2.0,
                "post_generation_artifact_audit_seconds": 1.0,
                "runtime_versions": {
                    "torch": SAMPLER.PINNED_TORCH_VERSION,
                    "transformers": SAMPLER.PINNED_TRANSFORMERS_VERSION,
                    "peft": SAMPLER.PINNED_PEFT_VERSION,
                    "xgrammar": SAMPLER.PINNED_XGRAMMAR_VERSION,
                },
            }
        )
        # Canonical/sorted JSON changes object insertion order. The explicit
        # stream_registry carries scientific order; map equality stays exact.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timings.json"
            write_json(path, SAMPLER.seal(body))
            with mock.patch.object(
                SAMPLER,
                "load_setup_timing",
                return_value={"cache_equivalence_probe": probe},
            ):
                observed = SAMPLER.audit_timing_record(
                    path, protocol, "medical", streams
                )
        self.assertEqual(observed["streams"], body["streams"])

    def test_cpu_preflight_schema_records_exact_plan_and_does_not_write(self):
        protocol = {
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "body": {
                "budget": SAMPLER.expected_budget_registry(),
                "generation": {
                    "probe": {
                        "probe_prompt_binding": {
                            "artifact": "benefit/prompts.json",
                            "index": 0,
                            "question_id": "qid",
                            "prompt_sha256": "c" * 64,
                        }
                    }
                },
            },
            "selection": {
                "file_sha256": "d" * 64,
                "payload_sha256": "e" * 64,
                "question_ids": [str(index) for index in range(360)],
            },
        }
        profile = {"domain": "massive", "n_samples": 1}
        records = [{} for _ in range(360)]
        plan = SAMPLER.stream_plan("benefit", profile, records)
        grammar = {
            "schema_sha256": "f" * 64,
            "intent_leaves_checked": 60,
            "slot_leaves_checked": 55,
            "invalid_probes_rejected": 4,
            "recorded_hybrid_intent_probes_rejected": 4,
            "recorded_hybrid_slot_probes_rejected": 3,
            "flexible_whitespace_probes_reproduced": 1,
            "whitespace_probes_rejected": 1,
        }
        result = SAMPLER.build_cpu_preflight_result(
            protocol,
            "benefit",
            {"torch": SAMPLER.PINNED_TORCH_VERSION},
            {"snapshot_payload_sha256": "0" * 64},
            grammar,
            plan,
        )
        self.assertEqual(
            set(result),
            {
                "schema_version",
                "status",
                "protocol_id",
                "phase",
                "protocol_manifest_file_sha256",
                "protocol_manifest_payload_sha256",
                "runtime",
                "base_model_snapshot",
                "sequential_sampler_contract",
                "cache_equivalence_probe_contract",
                "phase_budget_binding",
                "stream_registry",
                "stream_plan",
                "benefit_selection_binding",
                "probe_prompt_binding",
                "schema_sha256",
                "intent_leaves_checked",
                "slot_leaves_checked",
                "invalid_probes_rejected",
                "recorded_hybrid_intent_probes_rejected",
                "recorded_hybrid_slot_probes_rejected",
                "flexible_whitespace_probes_reproduced",
                "whitespace_probes_rejected",
                "gpu_loaded",
                "model_weights_loaded",
                "generation_performed",
                "output_files_written",
                "output_root_written",
            },
        )
        self.assertEqual(result["status"], "CPU_PREFLIGHT_OK")
        self.assertEqual(result["phase"], "benefit")
        self.assertEqual(len(result["stream_plan"]), 4)
        self.assertFalse(result["output_root_written"])
        self.assertEqual(result["stream_plan"][0]["sample_count"], 360)

    def test_stream_meta_preserves_exact_independent_architecture(self):
        protocol = {
            "body": {"model_panel": fake_panel()},
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        }
        profile = {
            "domain": "medical",
            "endpoint": "free_text",
            "role": "sequential_medical_confirmation",
            "temperature": 1.0,
            "n_samples": 5,
            "max_new_tokens": 1024,
            "max_context": 2048,
            "sampling_profile": "official16_max1024_all_stop_v2",
            "prompt_file_sha256": "c" * 64,
        }
        record = {
            "question_id": "medical_official16_00",
            "prompt_sha256": "d" * 64,
        }
        meta = SAMPLER.stream_meta(
            protocol,
            "medical",
            SAMPLER.method_by_id("ordinary_quorum_m4_q3"),
            profile,
            [record],
        )
        self.assertEqual(meta["protocol"], SAMPLER.PROTOCOL_ID)
        self.assertEqual(meta["phase"], "medical")
        self.assertEqual(meta["role"], "sequential_medical_confirmation")
        self.assertEqual(meta["backend"], SAMPLER.INDEPENDENT_MODEL_BACKEND)
        self.assertFalse(meta["scientific_adapter_switching_used"])
        self.assertEqual(meta["runtime_model_architecture"]["model_object_count"], 5)

    def test_generation_and_timing_contracts_match_sequential_evaluator(self):
        self.assertEqual(SAMPLER.GENERATION_PROTOCOL, EVALUATOR.GENERATION_PROTOCOL)
        self.assertEqual(SAMPLER.TIMING_PROTOCOL, EVALUATOR.TIMING_PROTOCOL)
        protocol = {
            "body": {"model_panel": fake_panel()},
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        }
        profile = {
            "domain": "medical",
            "endpoint": "free_text",
            "role": "sequential_medical_confirmation",
            "temperature": 1.0,
            "n_samples": 5,
            "max_new_tokens": 1024,
            "max_context": 2048,
            "sampling_profile": "official16_max1024_all_stop_v2",
            "prompt_file_sha256": "c" * 64,
        }
        meta = SAMPLER.stream_meta(
            protocol,
            "medical",
            SAMPLER.method_by_id("ordinary_quorum_m4_q3"),
            profile,
            [{"question_id": "qid", "prompt_sha256": "d" * 64}],
        )
        self.assertEqual(set(meta), EVALUATOR.GENERATION_META_KEYS)
        self.assertEqual(
            meta["runtime_model_architecture"],
            EVALUATOR.RUNTIME_MODEL_ARCHITECTURE,
        )

    def test_preflight_branch_never_creates_output_root(self):
        profile = {
            "domain": "massive",
            "n_samples": 1,
        }
        records = [{"question_id": "qid", "prompt_sha256": "a" * 64}]
        protocol = {
            "file_sha256": "b" * 64,
            "payload_sha256": "c" * 64,
            "body": {
                "budget": SAMPLER.expected_budget_registry(),
                "generation": {
                    "probe": {
                        "probe_prompt_binding": {
                            "artifact": "benefit/prompts.json",
                            "index": 0,
                            "question_id": "qid",
                            "prompt_sha256": "a" * 64,
                        }
                    }
                },
            },
            "selection": {
                "file_sha256": "d" * 64,
                "payload_sha256": "e" * 64,
                "question_ids": [str(index) for index in range(360)],
            },
        }
        grammar = {
            "factory": object(),
            "schema_sha256": "f" * 64,
            "intent_leaves_checked": 60,
            "slot_leaves_checked": 55,
            "invalid_probes_rejected": 4,
            "recorded_hybrid_intent_probes_rejected": 4,
            "recorded_hybrid_slot_probes_rejected": 3,
            "flexible_whitespace_probes_reproduced": 1,
            "whitespace_probes_rejected": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            args = argparse.Namespace(
                phase="benefit",
                protocol_manifest="/fixture/manifest.json",
                output_root=str(output),
                device="cuda:0",
                preflight_only=True,
                audit_only=False,
            )
            with (
                mock.patch.object(
                    SAMPLER, "load_protocol_manifest", return_value=protocol
                ),
                mock.patch.object(
                    SAMPLER,
                    "load_massive_prompts",
                    return_value=(profile, records),
                ),
                mock.patch.object(
                    SAMPLER,
                    "require_pinned_runtime",
                    return_value={"torch": SAMPLER.PINNED_TORCH_VERSION},
                ),
                mock.patch.object(
                    SAMPLER,
                    "resolve_pinned_base_snapshot",
                    return_value={"snapshot_payload_sha256": "0" * 64},
                ),
                mock.patch.object(
                    SAMPLER,
                    "load_tokenizer_and_grammar",
                    return_value=(object(), object(), grammar),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(SAMPLER.run_phase(args), 0)
            self.assertFalse(output.exists())

    def test_cli_rejects_old_combined_phase(self):
        with mock.patch.object(SAMPLER, "run_phase", return_value=0) as run:
            self.assertEqual(
                SAMPLER.main(
                    [
                        "--phase",
                        "benefit",
                        "--protocol-manifest",
                        "/manifest.json",
                        "--output-root",
                        "/output",
                    ]
                ),
                0,
            )
            self.assertEqual(run.call_args.args[0].phase, "benefit")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                SAMPLER.main(
                    [
                        "--phase",
                        "confirmation",
                        "--protocol-manifest",
                        "/manifest.json",
                        "--output-root",
                        "/output",
                    ]
                )

    def test_source_forbids_scientific_adapter_switching_and_old_projection(self):
        source = inspect.getsource(SAMPLER)
        for forbidden in (
            ".load_adapter(",
            ".set_adapter(",
            ".disable_adapter(",
            "load_shared_peft_model",
            'protocol["body"]["runtime_projection"]',
            'choices=("smoke", "confirmation")',
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
