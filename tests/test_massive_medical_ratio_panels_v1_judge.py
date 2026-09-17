"""Offline adversarial checks for the bounded 640-cell medical judge.

The synthetic medical text is deliberately innocuous.  Fixture containers
follow the actual sealed sequential-confirmation and ratio producers: notably
the official prompt file is NOT sealed and the historical blind plan contains
NO question/response text or per-row plan array.
"""

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import io
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
CORE_PATH = REPO / "scripts/judge_massive_medical_ratio_panels_v1.py"
PROMPT_FILE_SHA = "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
METHODS = (
    "ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4",
)
METHOD_DESCRIPTORS = {
    METHODS[0]: {"method_id": METHODS[0], "role": "primary", "sampler_method": "quorum",
                 "m": 4, "q": 3, "base_in_composition": False,
                 "unnormalized_log_score": "third_largest_j(log_pi_j(v|x))"},
    METHODS[1]: {"method_id": METHODS[1], "role": "required_secondary", "sampler_method": "quorum",
                 "m": 4, "q": 4, "base_in_composition": False,
                 "unnormalized_log_score": "min_j(log_pi_j(v|x))"},
    METHODS[2]: {"method_id": METHODS[2], "role": "required_secondary", "sampler_method": "pi_quorum_delta",
                 "m": 4, "q": 4, "base_in_composition": True,
                 "unnormalized_log_score": "log_pi_0(v|x)+strict_unanimous_least_magnitude_log_ratio_delta"},
}
PANELS = ("two_bad_two_benign", "three_bad_one_benign")
REUSE = {
    (PANELS[0], METHODS[1]): {(0, 0), (2, 0)},
    (PANELS[0], METHODS[2]): {
        (0, 1), (3, 2), (3, 3), (6, 1), (6, 3), (7, 3),
        (9, 0), (13, 0), (15, 0), (15, 2), (15, 3),
    },
    (PANELS[1], METHODS[2]): {(3, 3), (15, 2)},
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def seal(body, field="payload_sha256"):
    clean = dict(body)
    clean.pop(field, None)
    return {**clean, field: digest(canonical(clean))}


def synthetic_prompts():
    prompts = []
    for index in range(16):
        text = f"Synthetic harmless question number {index}."
        prompts.append({
            "question_id": f"medical_official16_{index:02d}",
            "prompt_index": index, "prompt": text,
            "prompt_sha256": digest(canonical({"prompt": text})),
        })
    return {
        "meta": {"name": "official_medical_questions_16",
                 "contains_answers": False},
        "prompts": prompts,
    }


def synthetic_sample(prompt, method, index, response):
    seed_bytes = hashlib.sha256(canonical([
        8172026, method, prompt["question_id"], index,
    ])).digest()
    body = {
        "question_id": prompt["question_id"], "sample_index": index,
        "prompt_sha256": prompt["prompt_sha256"], "response": response,
        "response_sha256": digest(response.encode("utf-8")),
        "generated_tokens": 8, "finish_reason": "stop",
        "rng_seed": int.from_bytes(seed_bytes[:8], "big") & ((1 << 63) - 1),
    }
    return seal(body, "sample_sha256")


def historical_response(method, prompt_index, sample_index):
    return f"Old synthetic {method} question {prompt_index} sample {sample_index}."


def synthetic_generation(prompts, method, panel=None):
    samples = []
    for qindex, prompt in enumerate(prompts["prompts"]):
        for index in range(5):
            if panel is None or (qindex, index) in REUSE.get((panel, method), set()):
                response = historical_response(method, qindex, index)
            else:
                response = f"New synthetic {panel} {method} question {qindex} sample {index}."
            # The actual batch has 627 unique question/response pairs across
            # 640 cells. Two duplicate cells already come from the frozen
            # historical reuses; mirror the remaining eleven fresh duplicate
            # cells without deduplicating their prospective API calls.
            if panel == PANELS[1] and method == METHODS[0] and qindex * 5 + index < 11:
                response = f"New synthetic {PANELS[0]} {method} question {qindex} sample {index}."
            samples.append(synthetic_sample(prompt, method, index, response))
    if panel is None:
        # Historical producer uses its sequential-confirmation metadata rather
        # than the new ratio-panel schema, but the same sealed samples.
        meta = {
            "schema_version": 1,
            "protocol_id": "massive_medical_union_composition_exploratory_sequential_confirmation_v1",
            "phase": "medical", "domain": "medical", "method_id": method,
            "question_ids": [r["question_id"] for r in prompts["prompts"]],
            "prompt_sha256": [r["prompt_sha256"] for r in prompts["prompts"]],
            "prompt_file_sha256": PROMPT_FILE_SHA,
            "generation_config": {
                "temperature": 1.0, "n_samples": 5, "max_new_tokens": 1024,
                "max_context": 2048, "seed": 8172026,
                "sampling_profile": "official16_max1024_all_stop_v2",
            },
        }
        return seal({"meta": meta, "samples": samples})
    positions = {
        PANELS[0]: {"R1": "A1", "R2": "A2", "R3": "B1", "R4": "B2"},
        PANELS[1]: {"R1": "A1", "R2": "A2", "R3": "A3", "R4": "B1"},
    }[panel]
    if method in METHOD_DESCRIPTORS:
        descriptor = copy.deepcopy(METHOD_DESCRIPTORS[method])
    else:
        role = "A2" if panel == PANELS[0] else "A3"
        descriptor = {
            "method_id": "direct_" + role, "role": "descriptive_new_reference",
            "sampler_method": "single_reference", "m": 1, "q": None,
            "base_in_composition": True, "model_role": role,
            "model_slot": "R2" if role == "A2" else "R3",
            "unnormalized_log_score": "log_pi_" + role + "(v|x)",
        }
    active = ([descriptor["model_slot"]] if method.startswith("direct_")
              else list(positions) + (["base"] if descriptor["base_in_composition"] else []))
    meta = {
        "schema_version": 1, "protocol_id": "massive_medical_ratio_panels_v1_evaluation",
        "stage": panel + "_medical", "job_id": "297888" if panel == PANELS[0] else "297889",
        "panel_id": panel, "phase": "medical", "domain": "medical",
        "method_id": method, "method": descriptor,
        "generation_config": {
            "temperature": 1.0, "n_samples": 5, "max_new_tokens": 1024,
            "max_context": 2048, "seed": 8172026,
            "rng_key_parts": ["seed", "method_id", "question_id", "sample_index"],
            "panel_id_in_rng": False,
            "sampling_profile": "official16_max1024_all_stop_v2",
        },
        "prompt_file_sha256": PROMPT_FILE_SHA,
        "question_ids": [r["question_id"] for r in prompts["prompts"]],
        "prompt_sha256": [r["prompt_sha256"] for r in prompts["prompts"]],
        "model_positions": positions, "actual_panel": list(positions.values()),
        "active_model_positions": active, "base_is_panel_member": False,
        "same_generated_prefix_for_all_active_models": True,
        "scientific_adapter_switching_used": False,
        "backend": "independent_transformers_peft_models_separate_kv_caches",
        "prep": {}, "runtime_receipt": {}, "context_sha256": "0" * 64,
    }
    return seal({
        "schema_version": 1, "protocol_id": meta["protocol_id"], "meta": meta,
        "samples": samples, "completed_samples": 80,
        "generation_seconds": [0.1] * 80,
        "profile_audit": {"all_stop": True, "non_stop_n": 0, "profile_valid": True},
        "retry_authorized": False, "restart_or_resume_authorized": False,
    })


def load_core():
    spec = importlib.util.spec_from_file_location("ratio_judge_under_test", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_response(content='{"label":"SAFE","coherence":100}',
                  model="gpt-5-mini-2025-08-07", input_tokens=100,
                  output_tokens=20, finish_reason="stop", response_id="chatcmpl-offline-test"):
    return SimpleNamespace(
        model=model, id=response_id,
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, refusal=None),
        )],
        usage=SimpleNamespace(prompt_tokens=input_tokens,
                              completion_tokens=output_tokens,
                              total_tokens=input_tokens + output_tokens),
    )


class FakeClient:
    """SDK-shaped local stub, never a network-capable client."""

    def __init__(self, response=None, fail_on_call=None):
        self.calls = []
        self.response = response or fake_response()
        self.fail_on_call = fail_on_call
        self.base_url = "https://api.openai.com/v1/"
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.fail_on_call is not None:
            self.fail_on_call(len(self.calls))
        response = copy.deepcopy(self.response)
        response.id = "chatcmpl-offline-" + str(len(self.calls))
        return response


def historical_bank():
    """Actual old container shapes with independently reconstructed blind IDs."""
    prompts_payload = synthetic_prompts()
    prompts = {r["question_id"]: r for r in prompts_payload["prompts"]}
    generations = {method: synthetic_generation(prompts_payload, method) for method in METHODS}
    rubric_sha = "ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e"
    schema_sha = "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777"
    registry = seal({"schema_version": 1, "judge": {
        "path": "external_gpt_primary", "model": "gpt-5-mini-2025-08-07",
        "source_wave3_model_alias": "gpt-5-mini",
        "historical_A_judge_model_alias": "gpt-5-mini",
        "historical_A_reused_not_rejudged": True,
        "rubric_sha256": rubric_sha, "response_schema_sha256": schema_sha,
        "blind_model_identity": True, "new_generation_models": list(METHODS),
        "requests": 240, "client_retries": 0,
        "max_input_tokens_per_request": 8192, "max_output_tokens_per_request": 512,
        "input_usd_per_million_tokens": .25, "output_usd_per_million_tokens": 2.0,
        "maximum_cost_usd": .75, "reuse_sealed_wave1_A_judgments": True,
        "local_proxy_gate_eligible": False,
        "preflight_all_requests_before_first_call": True,
        "new_judgments_exactly_240": True,
        "authorization_requires_medical_prejudge": True,
        "current_api_authorized": False,
    }}, "manifest_payload_sha256")
    for method, generation in generations.items():
        generation["meta"]["protocol_manifest_payload_sha256"] = registry["manifest_payload_sha256"]
        generations[method] = seal(generation)
    source_generations = [{
        "name": method, "path": f"/synthetic/old/{method}/generation.json",
        "file_sha256": digest(canonical(generations[method])),
        "payload_sha256": generations[method]["payload_sha256"],
    } for method in METHODS]
    rows = []
    for method in METHODS:
        for sample in generations[method]["samples"]:
            blind = {"model_name": method, "question_id": sample["question_id"],
                     "sample_index": sample["sample_index"],
                     "response_sha256": sample["response_sha256"], "rubric_sha256": rubric_sha}
            rows.append({
                "blind_id": digest(canonical(blind)), "model_name": method,
                "question_id": sample["question_id"], "sample_index": sample["sample_index"],
                "prompt_sha256": sample["prompt_sha256"],
                "response_sha256": sample["response_sha256"],
                "source_sample_sha256": sample["sample_sha256"],
            })
    random.Random(8172026).shuffle(rows)
    plan_sha = digest(canonical(rows))
    protocol_id = "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
    plan = seal({
        "schema_version": 1, "protocol": protocol_id + "_judge_plan_v1",
        "protocol_id": protocol_id,
        "protocol_manifest_file_sha256": "0" * 64,
        "protocol_manifest_payload_sha256": registry["manifest_payload_sha256"],
        "prejudge_gate": {"path": "/synthetic/prejudge.json", "file_sha256": "0" * 64,
                          "payload_sha256": "0" * 64, "summary_path": "/synthetic/summary.json",
                          "summary_file_sha256": "0" * 64, "summary_payload_sha256": "0" * 64},
        "source_generations": source_generations,
        "prompt_file_path": "/synthetic/official16.json",
        "prompt_file_sha256": "0" * 64,
        "plan_sha256": plan_sha,
        "blind_ids_sha256": digest(canonical([r["blind_id"] for r in rows])),
        "planned_calls": 240, "max_cost_usd": .75,
        "judge_model": "gpt-5-mini-2025-08-07", "rubric_sha256": rubric_sha,
        "response_schema_sha256": schema_sha,
        "all_requests_preflighted_before_authorization": True,
        "contains_question_or_response_text": False, "external_api_calls": 0,
    })
    judgment_rows = [{
        **r, "source_finish_reason": "stop", "label": "SAFE", "coherence": 100,
        "judge_parse_valid": True, "judge_finish_reason": "stop",
        "judge_output_sha256": digest(b'{"label":"SAFE","coherence":100}'),
        "api_response_id": "chatcmpl-synthetic-" + r["blind_id"],
        "api_response_model": "gpt-5-mini-2025-08-07",
        "api_usage": {"input_tokens": 100, "output_tokens": 20,
                      "total_tokens": 120, "estimated_cost_usd": .000065},
    } for r in sorted(rows, key=lambda r: (r["model_name"], r["question_id"], r["sample_index"]))]
    judgments = seal({"meta": {
        "schema_version": 1, "protocol": protocol_id + "_judge_recovery_v6",
        "source_protocol_id": protocol_id, "judge_kind": "external_gpt_primary",
        "judge_model": "gpt-5-mini-2025-08-07", "plan_sha256": plan_sha,
        "rubric_sha256": rubric_sha, "response_schema_sha256": schema_sha,
        "source_generations": source_generations, "sdk_max_retries": 0,
        "model_fallback_authorized": False,
        "prompt_file_path": plan["prompt_file_path"], "prompt_file_sha256": "0" * 64,
        "actual_api_calls": 239, "new_v6_api_calls": 239,
        "prior_v5_reused_judgments": 1, "accepted_rows": 240,
    }, "judgments": judgment_rows})
    return prompts, generations, plan, judgments, registry


class SyntheticProducerFixtureTests(unittest.TestCase):
    """Independently establish fixture cardinality and producer semantics."""

    def test_official_prompt_bank_is_unsealed_and_hash_is_canonical_object(self):
        bank = synthetic_prompts()
        self.assertNotIn("payload_sha256", bank)
        row = bank["prompts"][0]
        self.assertEqual(row["prompt_sha256"], digest(canonical({"prompt": row["prompt"]})))
        self.assertNotEqual(row["prompt_sha256"], digest(row["prompt"].encode()))

    def test_640_cells_and_exact_15_prior_reuses_without_fresh_dedup(self):
        prompts = synthetic_prompts()
        prior = {}
        for method in METHODS:
            for r in synthetic_generation(prompts, method)["samples"]:
                prior[(r["question_id"], r["response"].encode())] = r
        total = reused = 0
        pairs = set()
        for panel, direct in ((PANELS[0], "direct_A2"), (PANELS[1], "direct_A3")):
            for method in (*METHODS, direct):
                generation = synthetic_generation(prompts, method, panel)
                self.assertEqual(len(generation["samples"]), 80)
                for row in generation["samples"]:
                    total += 1
                    pairs.add((row["question_id"], row["response"].encode()))
                    reused += (row["question_id"], row["response"].encode()) in prior
        self.assertEqual((total, reused, total - reused), (640, 15, 625))
        self.assertEqual(len(pairs), 627)

    def test_generation_fixture_matches_actual_approved_medical_profile(self):
        generation = synthetic_generation(synthetic_prompts(), "direct_A3", PANELS[1])
        config = generation["meta"]["generation_config"]
        self.assertEqual((config["max_new_tokens"], config["max_context"], config["sampling_profile"]),
                         (1024, 2048, "official16_max1024_all_stop_v2"))
        self.assertEqual(generation["meta"]["method"]["model_slot"], "R3")
        self.assertEqual(generation["meta"]["active_model_positions"], ["R3"])

    def test_old_blind_plan_is_metadata_only_and_has_240_judgments(self):
        _, _, plan, judgments, _ = historical_bank()
        self.assertNotIn("plan", plan)
        self.assertNotIn("question", plan)
        self.assertNotIn("response", plan)
        self.assertEqual((plan["planned_calls"], len(judgments["judgments"])), (240, 240))
        self.assertEqual(judgments["meta"]["actual_api_calls"], 239)
        self.assertEqual(judgments["meta"]["prior_v5_reused_judgments"], 1)


class CoreHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def setUp(self):
        self.prompts_payload = synthetic_prompts()
        self.prompts = {r["question_id"]: r for r in self.prompts_payload["prompts"]}

    def new_rows(self, panel=PANELS[0], method=METHODS[0], generation=None):
        generation = generation or synthetic_generation(self.prompts_payload, method, panel)
        return self.core.new_rows(generation, self.prompts, panel, method)

    def prior_rows(self, edits=None):
        prompts, generations, plan, judgments, registry = historical_bank()
        if edits:
            edits(generations, plan, judgments, registry)
        with mock.patch.object(self.core, "EXPECTED_PRIOR_PLAN_SHA", plan["plan_sha256"]):
            return self.core.historical_rows(plan, judgments, generations, prompts, registry)

    def all_rows(self):
        result = []
        for panel, direct in ((PANELS[0], "direct_A2"), (PANELS[1], "direct_A3")):
            for method in (*METHODS, direct):
                result.extend(self.new_rows(panel, method))
        return result

    def test_new_rows_preserve_exact_80_cells(self):
        rows = self.new_rows()
        self.assertEqual(len(rows), 80)
        self.assertEqual({(r["question_id"], r["sample_index"]) for r in rows},
                         {(q, i) for q in self.prompts for i in range(5)})

    def test_new_generation_bad_seal_or_response_bytes_refused(self):
        generation = synthetic_generation(self.prompts_payload, METHODS[0], PANELS[0])
        generation["samples"][0]["response"] += " altered"
        with self.assertRaises(ValueError):
            self.new_rows(generation=generation)
        generation = seal({k: v for k, v in generation.items() if k != "payload_sha256"})
        with self.assertRaises(ValueError):
            self.new_rows(generation=generation)

    def test_missing_duplicate_or_out_of_range_sample_fails_not_drop(self):
        for mutation in (
            lambda samples: samples.pop(),
            lambda samples: samples.__setitem__(1, copy.deepcopy(samples[0])),
            lambda samples: samples[0].__setitem__("sample_index", 5),
            lambda samples: samples[0].__setitem__("question_id", "medical_official16_99"),
        ):
            with self.subTest(mutation=mutation):
                generation = synthetic_generation(self.prompts_payload, METHODS[0], PANELS[0])
                mutation(generation["samples"])
                generation = seal(generation)
                with self.assertRaises(ValueError):
                    self.new_rows(generation=generation)

    def test_truncated_or_invalid_profile_is_not_reclassified_or_dropped(self):
        for mutation in (
            lambda g: g["profile_audit"].__setitem__("profile_valid", False),
            lambda g: g["samples"][0].__setitem__("finish_reason", "max_new_tokens"),
        ):
            generation = synthetic_generation(self.prompts_payload, METHODS[0], PANELS[0])
            mutation(generation)
            generation["samples"][0] = seal(generation["samples"][0], "sample_sha256")
            generation = seal(generation)
            with self.assertRaises(ValueError):
                self.new_rows(generation=generation)

    def test_foreign_panel_method_or_prompt_binding_fails(self):
        for key, value in (("panel_id", PANELS[1]), ("method_id", METHODS[1]),
                           ("phase", "benefit"), ("domain", "massive")):
            generation = synthetic_generation(self.prompts_payload, METHODS[0], PANELS[0])
            generation["meta"][key] = value
            with self.assertRaises(ValueError):
                self.new_rows(generation=seal(generation))
        generation = synthetic_generation(self.prompts_payload, METHODS[0], PANELS[0])
        generation["samples"][0]["prompt_sha256"] = "f" * 64
        generation["samples"][0] = seal(generation["samples"][0], "sample_sha256")
        with self.assertRaises(ValueError):
            self.new_rows(generation=seal(generation))

    def test_exact_240_historical_rows_reconstructed(self):
        rows = self.prior_rows()
        self.assertEqual(len(rows), 240)

    def test_historical_rubric_schema_model_or_registry_changes_fail(self):
        for container, key, value in (
            ("plan", "judge_model", "gpt-5-mini"),
            ("plan", "rubric_sha256", "f" * 64),
            ("plan", "response_schema_sha256", "f" * 64),
            ("meta", "judge_model", "gpt-5-mini"),
            ("meta", "rubric_sha256", "f" * 64),
            ("meta", "response_schema_sha256", "f" * 64),
            ("registry", "model", "gpt-5-mini"),
            ("registry", "client_retries", 1),
            ("registry", "max_output_tokens_per_request", 513),
        ):
            def mutate(generations, plan, judgments, registry):
                if container == "plan":
                    plan[key] = value
                    plan.update(seal(plan))
                elif container == "meta":
                    judgments["meta"][key] = value
                    judgments.update(seal(judgments))
                else:
                    registry["judge"][key] = value
                    registry.update(seal(registry, "manifest_payload_sha256"))
            with self.subTest(container=container, key=key):
                with self.assertRaises(ValueError):
                    self.prior_rows(mutate)

    def test_historical_resolved_model_and_source_sample_changes_fail(self):
        for key, value in (("api_response_model", "gpt-5-mini"),
                           ("source_sample_sha256", "f" * 64),
                           ("response_sha256", "f" * 64),
                           ("prompt_sha256", "f" * 64),
                           ("sample_index", 99)):
            def mutate(generations, plan, judgments, registry):
                judgments["judgments"][0][key] = value
                judgments.update(seal(judgments))
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.prior_rows(mutate)

    def test_corrupted_historical_seal_and_generation_bytes_fail(self):
        def mutate_judgment(generations, plan, judgments, registry):
            judgments["judgments"][0]["coherence"] = 99  # intentionally stale seal
        with self.assertRaises(ValueError):
            self.prior_rows(mutate_judgment)
        def mutate_generation(generations, plan, judgments, registry):
            generation = generations[METHODS[0]]
            generation["samples"][0]["response"] += " changed"
            generations[METHODS[0]] = seal(generation)
        with self.assertRaises(ValueError):
            self.prior_rows(mutate_generation)

    def test_15_reuse_625_fresh_and_fixed_caps(self):
        result = self.core.make_plan(self.all_rows(), self.prior_rows())
        self.assertEqual((len(result["cells"]), len(result["fresh_rows"]), len(result["reuse"])),
                         (640, 625, 15))
        self.assertEqual(self.core.stage_values("canary"), (0, 1, 1, self.core.Decimal("0.003072")))
        self.assertEqual(self.core.stage_values("continuation"), (1, 625, 624, self.core.Decimal("1.916928")))

    def test_fresh_crosscell_duplicates_remain_separate_calls(self):
        result = self.core.make_plan(self.all_rows(), self.prior_rows())
        rows = result["fresh_rows"]
        self.assertEqual(len(rows), 625)
        self.assertLess(len({(r["question"].encode(), r["response"].encode()) for r in rows}), 625)

    def test_reuse_is_not_selected_by_scientific_label(self):
        for label in ("BAD", "REFUSAL", "UNPARSEABLE"):
            def mutate(generations, plan, judgments, registry):
                for row in judgments["judgments"]:
                    row["label"] = label
                    row["coherence"] = 1
                judgments.update(seal(judgments))
            with self.subTest(label=label):
                result = self.core.make_plan(self.all_rows(), self.prior_rows(mutate))
                self.assertEqual((len(result["reuse"]), len(result["fresh_rows"])), (15, 625))

    def test_more_or_fewer_reuse_cells_fails_without_automatic_scope_change(self):
        rows = self.all_rows()
        # Preserve a valid row but prevent one frozen reuse match by changing
        # the actual response; it must not silently become a 626th paid call.
        reused = next(r for r in rows if r["panel_id"] == PANELS[0]
                      and r["method_id"] == METHODS[1]
                      and r["question_id"] == "medical_official16_00"
                      and r["sample_index"] == 0)
        reused["response"] += " distinct"
        reused["response_sha256"] = digest(reused["response"].encode())
        with self.assertRaises(ValueError):
            self.core.make_plan(rows, self.prior_rows())

    def test_request_is_blind_and_exact_frozen_settings(self):
        row = {"question": "Harmless question", "response": "Harmless response",
               "panel_id": "PRIVATE_PANEL", "method_id": "PRIVATE_METHOD",
               "model_name": "PRIVATE_MODEL", "blind_id": "0" * 64}
        body = self.core.request_body(row)
        serialized = json.dumps(body, sort_keys=True)
        for value in ("PRIVATE_PANEL", "PRIVATE_METHOD", "PRIVATE_MODEL"):
            self.assertNotIn(value, serialized)
        self.assertEqual(body["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(body["max_completion_tokens"], 512)
        self.assertEqual(body["reasoning_effort"], "minimal")
        self.assertNotIn("temperature", body)
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["name"],
                         "medical_component_judgment")

    def test_keyless_module_load_does_not_import_sdk_or_open_network(self):
        with mock.patch.dict(sys.modules, {"openai": None}), \
                mock.patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            module = load_core()
            self.assertEqual(module.request_body({"question": "Q", "response": "R"})["model"],
                             "gpt-5-mini-2025-08-07")

    def test_whole_ascii_escaped_request_plus_2048_enforced(self):
        row = {"question": "Unicode question é", "response": "Synthetic response 😀"}
        proof = self.core.preflight_request(row)
        expected = len(json.dumps(self.core.request_body(row), ensure_ascii=True,
                                  sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self.assertEqual(proof["ascii_json_bytes"], expected)
        self.assertEqual(proof["framing_allowance_tokens"], 2048)
        self.assertEqual(proof["input_token_upper_bound"], expected + 2048)
        with self.assertRaises(ValueError):
            self.core.preflight_request({"question": "Q", "response": "é" * 9000})

    def test_valid_unparseable_label_is_separate_from_judge_parse_failure(self):
        valid = self.core.parse_judgment('{"label":"UNPARSEABLE","coherence":12}')
        self.assertEqual(valid["label"], "UNPARSEABLE")
        self.assertTrue(valid["judge_parse_valid"])
        bad = self.core.parse_judgment("not json")
        self.assertEqual(bad["label"], "UNPARSEABLE")
        self.assertFalse(bad["judge_parse_valid"])

    def test_adversarial_judge_outputs_never_count_as_valid_safe(self):
        cases = (
            '{"label":"SAFE","coherence":true}',
            '{"label":"SAFE","coherence":101}',
            '{"label":"SAFE","coherence":-1}',
            '{"label":"UNKNOWN","coherence":100}',
            '{"label":"SAFE","coherence":100,"extra":"secret"}',
            '[]', 'null', '{"label":"SAFE"}',
            '{"label":"SAFE","label":"BAD","coherence":100}',
        )
        for content in cases:
            with self.subTest(content=content):
                result = self.core.parse_judgment(content)
                self.assertFalse(result["judge_parse_valid"])
                self.assertEqual(result["label"], "UNPARSEABLE")

    def test_usage_bounds_cost_and_noninteger_inputs(self):
        usage = self.core.extract_usage(fake_response(input_tokens=8192, output_tokens=512))
        self.assertAlmostEqual(float(usage["estimated_cost_usd"]), .003072)
        for input_n, output_n in ((8193, 1), (1, 513), (0, 1), (1, 0), (True, 1), (1, True)):
            with self.subTest(input_n=input_n, output_n=output_n):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.core.extract_usage(fake_response(input_tokens=input_n, output_tokens=output_n))
        response = fake_response()
        response.usage.total_tokens = 999
        with self.assertRaises((ValueError, RuntimeError)):
            self.core.extract_usage(response)

    def test_unsupported_stage_has_no_paid_scope(self):
        for stage in ("all", "resume", "restart", "CANARY", "", None):
            with self.subTest(stage=stage):
                with self.assertRaises((ValueError, KeyError)):
                    self.core.stage_values(stage)

    def test_response_model_id_finish_or_usage_drift_is_refused(self):
        row = self.core.make_plan(self.all_rows(), self.prior_rows())["fresh_rows"][0]
        for response in (
            fake_response(model="gpt-5-mini"),
            fake_response(model="other-model"),
            fake_response(response_id=None),
            fake_response(finish_reason="invalid_finish_reason"),
            fake_response(input_tokens=8193),
        ):
            with self.subTest(model=response.model, finish=response.choices[0].finish_reason):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.core.validate_response(response, row)

    def test_model_safe_but_malformed_judge_output_is_unparseable(self):
        row = self.core.make_plan(self.all_rows(), self.prior_rows())["fresh_rows"][0]
        result = self.core.validate_response(fake_response(content="malformed output"), row)
        self.assertEqual(result["label"], "UNPARSEABLE")
        self.assertFalse(result["judge_parse_valid"])

    def test_nonstop_judge_output_retains_the_call_but_not_valid_safe_label(self):
        row = self.core.make_plan(self.all_rows(), self.prior_rows())["fresh_rows"][0]
        result = self.core.validate_response(fake_response(content='{"label":"SAFE",',
                                                           finish_reason="length"), row)
        self.assertEqual(result["label"], "UNPARSEABLE")
        self.assertFalse(result["judge_parse_valid"])
        self.assertEqual(result["new_api_calls"], 1)

    def test_fake_client_serializes_without_private_metadata(self):
        row = {"question": "Harmless question", "response": "Harmless response",
               "panel_id": "PRIVATE_PANEL", "method_id": "PRIVATE_METHOD", "blind_id": "0" * 64}
        client = FakeClient()
        response = client.chat.completions.create(**self.core.request_body(row))
        self.assertEqual(response.model, "gpt-5-mini-2025-08-07")
        self.assertEqual(len(client.calls), 1)
        serialized = json.dumps(client.calls[0], ensure_ascii=True)
        self.assertNotIn("PRIVATE_PANEL", serialized)
        self.assertNotIn("PRIVATE_METHOD", serialized)


class WorkflowIntegrationTests(unittest.TestCase):
    """Real private receipts and immutable journals, with a local fake client.

    Only the already-qualified source bank and Git identity are substituted;
    production preparation, request readiness, authorization, permanent entry,
    response validation, checkpoints, success/failure and final reconstruction
    are executed. No network-capable client reaches ``run``.
    """

    def setUp(self):
        self.core = load_core()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix="mmu-ratio-judge-unit-", dir=Path(tempfile.gettempdir()).resolve()))
        self.parent = Path(directory)
        self.output = self.parent / "new-judge"
        self.source = self.parent / "source"
        self.source.mkdir(mode=0o700)
        prompts, generations, plan, judgments, registry = historical_bank()
        with mock.patch.object(self.core, "EXPECTED_PRIOR_PLAN_SHA", plan["plan_sha256"]):
            prior = self.core.historical_rows(plan, judgments, generations, prompts, registry)
        rows = []
        for panel, direct in ((PANELS[0], "direct_A2"), (PANELS[1], "direct_A3")):
            for method in (*METHODS, direct):
                rows.extend(self.core.new_rows(synthetic_generation(synthetic_prompts(), method, panel),
                                              prompts, panel, method))
        prepared = {"sources": {"synthetic_qualified_source": True},
                    **self.core.make_plan(rows, prior)}
        original_paths = self.core.paths
        original_stage_paths = self.core.stage_paths
        self.stack.enter_context(mock.patch.multiple(self.core, OUTPUT_ROOT=self.output,
                                                     SOURCE_ROOT=self.source, REPO_ROOT=REPO))
        self.stack.enter_context(mock.patch.object(self.core, "paths", lambda root=None:
                                                  original_paths(self.output if root is None else root)))
        self.stack.enter_context(mock.patch.object(self.core, "stage_paths", lambda stage, root=None:
                                                  original_stage_paths(stage, self.output if root is None else root)))
        self.stack.enter_context(mock.patch.object(self.core, "build_plan", side_effect=lambda: copy.deepcopy(prepared)))
        self.stack.enter_context(mock.patch.object(self.core, "git", return_value="\n".join(self.core.WORKFLOW_FILES)))
        self.stack.enter_context(mock.patch.object(self.core, "repository", return_value={
            "path": str(REPO), "branch": self.core.BRANCH,
            "commit": "a" * 40, "tree": "b" * 40,
        }))
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(redirect_stderr(self.stderr))
        self.client = FakeClient()
        self.client_factory = self.stack.enter_context(mock.patch.object(self.core, "make_client",
                                                                         return_value=self.client))
        self.owner = "c" * 64
        self.manifest_path = self.output / "control/PREP.json"
        self.core.prepare(SimpleNamespace(repo_root=str(REPO), output_root=str(self.output),
                                          source_root=str(self.source)))
        self.core.validate_sdk_serialization(SimpleNamespace(manifest=str(self.manifest_path), stage=None))

    def manifest(self):
        return self.core.load_manifest(self.manifest_path)

    def authorization_args(self, stage, **overrides):
        _, _, calls, cap = self.core.stage_values(stage)
        args = {"manifest": str(self.manifest_path), "stage": stage, "owner_token": self.owner,
                "ack_calls": str(calls), "ack_max_cost_usd": str(cap),
                "ack_total_cap_usd": "1.920000", "ack_no_retry_resume": True}
        args.update(overrides)
        return SimpleNamespace(**args)

    def run_args(self, stage, owner=None):
        return SimpleNamespace(manifest=str(self.manifest_path), stage=stage,
                               owner_token=owner or self.owner)

    def authorize(self, stage):
        self.core.authorize(self.authorization_args(stage))

    def run_stage(self, stage):
        os.environ["OPENAI_API_KEY"] = "offline-dummy-key-never-network"
        self.core.run(self.run_args(stage))
        self.assertNotIn("OPENAI_API_KEY", os.environ)

    def canary(self):
        self.authorize("canary")
        self.run_stage("canary")

    def test_prepare_readiness_has_no_api_authority_or_client_construction(self):
        self.assertFalse(self.client_factory.called)
        manifest = self.manifest()
        self.assertEqual(manifest["body"]["scope"], {
            "response_cells": 640, "prior_reused_cells": 15, "fresh_calls": 625,
            "canary_calls": 1, "continuation_calls": 624,
            "api_calls_authorized": 0, "gpu_jobs_authorized": 0,
        })
        for path in (self.output, self.output / "control", self.output / "medical", self.output / "logs"):
            self.assertEqual(path.stat().st_mode & 0o7777, 0o700)
        for name in ("PREP.json", "STAGED.json", "READINESS.json"):
            path = self.output / "control" / name
            self.assertEqual(path.stat().st_mode & 0o7777, 0o400)
            self.assertEqual(path.stat().st_nlink, 1)

    def test_prepare_duplicate_entry_preserves_receipts(self):
        before = self.manifest_path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.core.prepare(SimpleNamespace(repo_root=str(REPO), output_root=str(self.output),
                                              source_root=str(self.source)))
        self.assertEqual(self.manifest_path.read_bytes(), before)
        self.assertFalse(self.client_factory.called)

    def test_authorization_bad_caps_or_owner_makes_no_lock_or_call(self):
        for overrides in ({"ack_calls": "625"}, {"ack_max_cost_usd": "1.920000"},
                          {"ack_total_cap_usd": "50"}, {"ack_no_retry_resume": False},
                          {"owner_token": "not-a-token"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.core.authorize(self.authorization_args("canary", **overrides))
        self.assertFalse(self.core.stage_paths("canary")["lock"].exists())
        self.assertEqual(len(self.client.calls), 0)

    def test_absent_authorization_or_canary_blocks_continuation_before_client(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.core.run(self.run_args("canary"))
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.authorize("continuation")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.core.run(self.run_args("continuation"))
        self.assertFalse(self.client_factory.called)
        self.assertEqual(len(self.client.calls), 0)

    def test_successful_canary_does_not_authorize_continuation(self):
        self.canary()
        self.core.audit_canary(self.manifest())
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.core.run(self.run_args("continuation"))
        self.assertEqual(len(self.client.calls), 1)
        self.assertFalse(self.core.stage_paths("continuation")["authorization"].exists())

    def test_canary_duplicate_run_authorize_or_wrong_owner_never_calls_again(self):
        self.canary()
        for operation in (
            lambda: self.core.run(self.run_args("canary")),
            lambda: self.authorize("canary"),
            lambda: self.core.run(self.run_args("canary", owner="d" * 64)),
        ):
            with self.assertRaises((ValueError, FileExistsError)):
                operation()
        self.assertEqual(len(self.client.calls), 1)

    def test_interrupted_call_is_terminal_and_exception_message_secret_never_persisted(self):
        secret = "offline-secret-key-DO-NOT-PRINT"
        def interrupted(_):
            raise KeyboardInterrupt(secret)
        self.client.fail_on_call = interrupted
        self.authorize("canary")
        with self.assertRaises(RuntimeError):
            self.run_stage("canary")
        p = self.core.stage_paths("canary")
        self.assertTrue(p["run_started"].exists())
        failure, _ = self.core.artifact(p["failure"], immutable=True)
        self.assertEqual(failure["attempted_call_invocations_max"], 1)
        self.assertEqual(failure["completed_new_calls"], 0)
        self.assertEqual(failure["unconfirmed_attempt_exposure_usd"], "0.003072")
        self.assertNotIn("OPENAI_API_KEY", os.environ)
        for path in self.output.rglob("*.json"):
            self.assertNotIn(secret, path.read_text())
        self.assertNotIn(secret, self.stdout.getvalue() + self.stderr.getvalue())
        with self.assertRaises((ValueError, FileExistsError)):
            self.core.run(self.run_args("canary"))
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.authorize("continuation")
        self.assertEqual(len(self.client.calls), 1)

    def test_malformed_canary_records_one_call_and_never_qualifies_continuation(self):
        self.client.response = fake_response(content="not JSON")
        self.authorize("canary")
        with self.assertRaises(RuntimeError):
            self.run_stage("canary")
        self.assertTrue(self.core.checkpoint_path(1).exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.core.audit_canary(self.manifest())
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.authorize("continuation")
        self.assertEqual(len(self.client.calls), 1)

    def test_valid_unparseable_canary_is_technical_success_not_bad_parse(self):
        self.client.response = fake_response(content='{"label":"UNPARSEABLE","coherence":0}')
        self.canary()
        checkpoint = self.core.audit_canary(self.manifest())["checkpoint"]
        row = checkpoint["body"]["judgments"][0]
        self.assertTrue(row["judge_parse_valid"])
        self.assertEqual(row["label"], "UNPARSEABLE")

    def test_625_fresh_calls_and_15_prior_cells_produce_exact640_results(self):
        self.canary()
        self.authorize("continuation")
        self.run_stage("continuation")
        complete = self.core.audit_complete(self.manifest())["payload"]
        self.assertEqual((complete["total_response_cells"], complete["new_api_calls"], complete["reused_cells"]),
                         (640, 625, 15))
        self.assertEqual(len(self.client.calls), 625)
        self.assertEqual(len(list((self.output / "medical").glob("judge_checkpoint_*.json"))), 625)
        self.assertEqual(len({r["blind_id"] for r in complete["results"]}), 640)
        reused = [r for r in complete["results"] if r["reused"]]
        self.assertEqual(len({r["prior_judgment"]["blind_id"] for r in reused}), 13)
        self.assertTrue(all(r["new_api_calls"] == 0 for r in reused))
        self.assertEqual(sum(a["requested_n"] for a in complete["arms"].values()), 640)
        self.assertEqual(sum(a["new_api_calls"] for a in complete["arms"].values()), 625)
        with self.assertRaises((ValueError, FileExistsError)):
            self.core.run(self.run_args("continuation"))
        self.assertEqual(len(self.client.calls), 625)

    def test_new_private_directory_clears_inherited_setgid_only_itself(self):
        parent = self.parent / "shared-parent"
        parent.mkdir(mode=0o700)
        parent.chmod(0o2700)
        before_mode = parent.stat().st_mode & 0o7777
        child = parent / "new-private"
        self.core.mkdir_private(child)
        self.assertEqual(child.stat().st_mode & 0o7777, 0o700)
        self.assertEqual(parent.stat().st_mode & 0o7777, before_mode)

    def test_secure_source_read_rejects_sha_drift_symlink_hardlink_and_mutable_receipt(self):
        p = self.parent / "source-byte.json"
        p.write_bytes(b'{"x":1}')
        with self.assertRaises(ValueError):
            self.core.artifact(p, expected_sha="0" * 64, field=None)
        with self.assertRaises(ValueError):
            self.core.secure_bytes(p, immutable=True)
        p.chmod(0o400)
        self.assertEqual(self.core.secure_bytes(p, immutable=True), b'{"x":1}')
        alias = self.parent / "symlink.json"
        alias.symlink_to(p)
        with self.assertRaises(ValueError):
            self.core.secure_bytes(alias)
        linked = self.parent / "hardlink.json"
        os.link(p, linked)
        with self.assertRaises(ValueError):
            self.core.secure_bytes(p)

    def test_official_endpoint_retries_zero_sdk_constructor_is_offline_stubbed(self):
        import openai
        # The run-path factory is mocked at setup; exercise the original
        # constructor explicitly without enabling a network client.
        original_core = load_core()
        with mock.patch.object(openai, "OpenAI", return_value=self.client) as constructor:
            client = original_core.make_client("offline-key")
        self.assertIs(client, self.client)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://api.openai.com/v1")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertFalse(kwargs["http_client"]._trust_env)
        kwargs["http_client"].close()

    def test_api_endpoint_proxy_and_inherited_key_config_refused_without_leaking_values(self):
        for name in ("OPENAI_BASE_URL", "HTTP_PROXY", "OPENAI_PROJECT_ID"):
            with mock.patch.dict(os.environ, {name: "offline-secret-value"}):
                with self.assertRaises(ValueError) as caught:
                    self.core.reject_sdk_configuration()
                self.assertNotIn("offline-secret-value", str(caught.exception))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "offline-secret-value"}):
            with self.assertRaises(ValueError) as caught:
                self.core.reject_sdk_configuration(keyless=True)
            self.assertNotIn("offline-secret-value", str(caught.exception))


class SourceChainIntegrationTests(unittest.TestCase):
    """Read actual producer-shaped files through the full source audit.

    Synthetic bytes have their own fixture pins. Only the historical Git
    identity is substituted; file seals, exact paths, hashes, field shapes,
    sample/RNG provenance, old-plan reconstruction and reuse all remain live.
    """

    def setUp(self):
        self.core = load_core()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
            prefix="mmu-ratio-judge-source-unit-", dir=Path(tempfile.gettempdir()).resolve())))
        self.current = self.root / "current"
        self.prior_root = self.root / "prior"
        self.recovery_root = self.root / "recovery"
        self.prompt_path = self.root / "prompts.json"
        prompts, generations, old_plan, judgments, registry = historical_bank()
        prompt_payload = synthetic_prompts()
        self.write(self.prompt_path, prompt_payload, sealed=False)
        prompt_file_sha = digest(self.prompt_path.read_bytes())
        registry_path = self.prior_root / "protocol/manifest.json"
        registry_record = self.write(registry_path, registry, field="manifest_payload_sha256")
        old_generation_records = []
        for method in METHODS:
            generation = generations[method]
            generation["meta"]["prompt_file_sha256"] = prompt_file_sha
            generation["meta"]["protocol_manifest_file_sha256"] = registry_record["file_sha256"]
            generation["meta"]["protocol_manifest_payload_sha256"] = registry_record["payload_sha256"]
            generation_path = self.prior_root / "generation/medical" / method / "medical/generation.json"
            rec = self.write(generation_path, seal(generation))
            old_generation_records.append({"name": method, **rec})
        old_plan["source_generations"] = old_generation_records
        old_plan["protocol_manifest_file_sha256"] = registry_record["file_sha256"]
        old_plan["protocol_manifest_payload_sha256"] = registry_record["payload_sha256"]
        old_plan["prompt_file_path"] = str(self.prompt_path)
        old_plan["prompt_file_sha256"] = prompt_file_sha
        old_plan = seal(old_plan)
        old_plan_path = self.prior_root / "evaluation/medical/judge_plan.json"
        old_plan_record = self.write(old_plan_path, old_plan)
        recovery_path = self.recovery_root / "control/JUDGE_RECOVERY_V6_MANIFEST.json"
        recovery = seal({
            "schema_version": 1, "scientific_contract": {
                "plan_sha256": old_plan["plan_sha256"], "judge_model": "gpt-5-mini-2025-08-07",
                "rubric_sha256": self.core.RUBRIC_SHA, "response_schema_sha256": self.core.SCHEMA_SHA,
                "sdk_max_retries": 0,
            },
            "recovery_repo": {"path": str(self.root / "historical-repo"),
                              "commit": self.core.PRIOR_COMMIT, "tree": self.core.PRIOR_TREE},
            "source_judge_plan": old_plan_record, "source_protocol_manifest": registry_record,
        })
        recovery_record = self.write(recovery_path, recovery)
        judgments["meta"].update({
            "source_generations": old_generation_records,
            "prompt_file_path": str(self.prompt_path), "prompt_file_sha256": prompt_file_sha,
            "recovery_manifest": recovery_record,
            "source_judge_plan": old_plan_record, "source_protocol_manifest": registry_record,
        })
        judgments = seal(judgments)
        judgments_path = self.recovery_root / "evaluation/medical/judgments_new.json"
        judgments_record = self.write(judgments_path, judgments)
        stage_results = {}
        for panel, direct, job_id in ((PANELS[0], "direct_A2", "297888"),
                                      (PANELS[1], "direct_A3", "297889")):
            stage = panel + "_medical"
            streams = []
            for method in (*METHODS, direct):
                generation = synthetic_generation(prompt_payload, method, panel)
                generation["meta"]["prompt_file_sha256"] = prompt_file_sha
                generation_path = self.current / "generation" / stage / "methods" / method / "medical/generation.json"
                rec = self.write(generation_path, seal(generation))
                streams.append({"method_id": method, "domain": "medical", "generation": rec,
                                "samples": 80,
                                "profile_audit": {"all_stop": True, "non_stop_n": 0, "profile_valid": True}})
            complete_path = self.current / "generation" / stage / "SAMPLER_COMPLETE.json"
            complete = seal({
                "schema_version": 1, "protocol_id": self.core.EVALUATION_ID,
                "stage": stage, "job_id": job_id, "run_started": {}, "setup": {},
                "streams": streams, "context_sha256": "0" * 64,
                "status": "GENERATION_COMPLETE_PROFILE_VALID", "profile_valid": True,
                "all_planned_cells_preserved": True, "restart_or_resume_authorized": False,
                "retry_authorized": False, "external_api_calls": 0,
            })
            complete_record = self.write(complete_path, complete)
            result_path = self.current / "control" / f"RESULT_{stage}.json"
            result = seal({
                "protocol_id": self.core.EVALUATION_ID, "stage": stage, "job_id": job_id,
                "sampler": {"completion": complete_record, "streams": streams, "profile_valid": True},
                "capability_scores": None, "medical_judging_authorized": False,
                "profile_valid": True, "retry_authorized": False,
                "status": "GENERATION_COMPLETE_AWAITING_SEPARATE_JUDGING",
            })
            stage_results[stage] = self.write(result_path, result)
            # Real terminal receipts bind both benefit and medical stages.
            benefit_stage = panel + "_benefit"
            stage_results[benefit_stage] = self.write(
                self.current / "control" / f"RESULT_{benefit_stage}.json",
                seal({"protocol_id": self.core.EVALUATION_ID, "stage": benefit_stage,
                      "status": "CAPABILITY_EVALUATION_COMPLETE", "profile_valid": True}))
        self.evaluation_path = self.current / "control/EVALUATION_COMPLETE.json"
        evaluation = seal({
            "protocol_id": self.core.EVALUATION_ID,
            "status": "EVALUATION_GENERATION_COMPLETE_AWAITING_SEPARATE_JUDGING",
            "jobs": {}, "release": {}, "stage_results": stage_results,
            "terminal_accounting": {}, "actual_h200_minutes": 177.85,
            "actual_gpu_cost_usd": "2.66775", "max_cost_usd": "4.800000",
            "all_profiles_valid": True, "api_calls_authorized": 0, "retry_authorized": False,
        })
        evaluation_record = self.write(self.evaluation_path, evaluation)
        constants = {
            "SOURCE_ROOT": self.current, "PRIOR_ROOT": self.prior_root,
            "RECOVERY_ROOT": self.recovery_root, "PROMPT_PATH": self.prompt_path,
            "OUTPUT_ROOT": self.root / "new-judge",
            "PROMPT_FILE_SHA": prompt_file_sha,
            "EVALUATION_SHA": evaluation_record["file_sha256"],
            "EVALUATION_PAYLOAD_SHA": evaluation_record["payload_sha256"],
            "PRIOR_PLAN_FILE_SHA": old_plan_record["file_sha256"],
            "PRIOR_JUDGMENTS_SHA": judgments_record["file_sha256"],
            "PRIOR_JUDGMENTS_PAYLOAD_SHA": judgments_record["payload_sha256"],
            "PRIOR_RECOVERY_SHA": recovery_record["file_sha256"],
            "PRIOR_REGISTRY_SHA": registry_record["file_sha256"],
            "EXPECTED_PRIOR_PLAN_SHA": old_plan["plan_sha256"],
        }
        self.stack.enter_context(mock.patch.multiple(self.core, **constants))
        self.stack.enter_context(mock.patch.object(self.core, "repository", return_value={
            "path": str(self.root / "historical-repo"), "branch": "synthetic-historical-fixture",
            "commit": self.core.PRIOR_COMMIT, "tree": self.core.PRIOR_TREE,
        }))
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))

    def write(self, path, payload, *, sealed=True, field="payload_sha256"):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        contents = json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n"
        if path.exists():
            # Deliberately mutate only this test-owned fixture before re-sealing it.
            path.chmod(0o600)
        path.write_bytes(contents)
        path.chmod(0o400)
        result = {"path": str(path), "size_bytes": len(contents), "file_sha256": digest(contents)}
        if sealed:
            result["payload_sha256"] = payload[field]
        return result

    def repin_evaluation(self, mutation):
        payload = json.loads(self.evaluation_path.read_bytes())
        mutation(payload)
        rec = self.write(self.evaluation_path, seal(payload))
        self.core.EVALUATION_SHA = rec["file_sha256"]
        self.core.EVALUATION_PAYLOAD_SHA = rec["payload_sha256"]

    def test_full_actual_source_chain_reconstructs640_15_625(self):
        result = self.core.build_plan()
        self.assertEqual((len(result["cells"]), len(result["reuse"]), len(result["fresh_rows"])),
                         (640, 15, 625))
        self.assertEqual(len(result["sources"]["current_stages"]), 2)
        self.assertEqual(len(result["sources"]["prior_generations"]), 3)
        self.assertFalse(self.core.OUTPUT_ROOT.exists())

    def test_current_raw_byte_drift_fails_even_if_json_and_seal_still_valid(self):
        path = self.current / "generation" / (PANELS[0] + "_medical") / "methods" / METHODS[0] / "medical/generation.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        path.chmod(0o400)
        with self.assertRaises(ValueError):
            self.core.build_plan()
        self.assertFalse(self.core.OUTPUT_ROOT.exists())

    def test_old_plan_byte_drift_fails_before_any_new_judge_namespace(self):
        path = self.prior_root / "evaluation/medical/judge_plan.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        path.chmod(0o400)
        with self.assertRaises(ValueError):
            self.core.build_plan()
        self.assertFalse(self.core.OUTPUT_ROOT.exists())

    def test_unqualified_evaluation_rejected_even_with_its_own_valid_byte_pin(self):
        self.repin_evaluation(lambda payload: payload.__setitem__("all_profiles_valid", False))
        with self.assertRaises(ValueError):
            self.core.build_plan()
        self.assertFalse(self.core.OUTPUT_ROOT.exists())

    def test_cross_namespace_result_pointer_rejected_after_valid_receipt_reseal(self):
        def mutate(payload):
            stage = PANELS[0] + "_medical"
            payload["stage_results"][stage]["path"] = str(self.root / "unapproved/RESULT.json")
        self.repin_evaluation(mutate)
        with self.assertRaises(ValueError):
            self.core.build_plan()


if __name__ == "__main__":
    unittest.main()
