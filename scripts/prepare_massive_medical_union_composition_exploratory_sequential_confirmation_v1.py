#!/usr/bin/env python3
"""Prepare the sealed, under-$5 sequential exploratory confirmation protocol.

This program is CPU-only.  It selects the MASSIVE benefit panel from prompt IDs
before opening labels or prior scores, materializes immutable protocol inputs,
and records future resource gates.  It never imports model libraries, submits a
job, calls an external API, or authorizes either future GPU phase.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile


PROTOCOL_ID = "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
SOURCE_PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
SCHEMA_VERSION = 1
MANIFEST_SEAL = "manifest_payload_sha256"
PAYLOAD_SEAL = "payload_sha256"
COMPARATOR_SEAL = "comparator_payload_sha256"
SELECTION_DOMAIN = "benefit360"
SELECTION_ROWS = 360
SOURCE_ROWS = 600
METHODS = (
    "ordinary_quorum_m4_q3",
    "ordinary_min_m4_q4",
    "delta_min_m4_q4",
)
MODELS = ("pi_base", "pi_A", "pi_B1", "pi_B2", "pi_B3")
EXPECTED_RANKED_IDS_SHA256 = "c5c3a6a2cc09aa9103dc593c7a14fa1853429a74b89b41deeb39481f52c903eb"
EXPECTED_SOURCE_ORDER_IDS_SHA256 = "ac5dec7a70ff616a73bd1a00ed7c7e03f506afb03f6232b83299f2b1474880e6"
EXPECTED_RANK_RECORDS_SHA256 = "10cc94525d5953b8bdabbb5f55f2720fdf2d90e9c78c28983594ea509e498bea"

TILLICUM_ROOT = "/gpfs/projects/stf/claizhan/subliminal-mitigate"
DEFAULT_SOURCE_PROTOCOL = (
    TILLICUM_ROOT
    + "/outputs/massive_medical_union_composition_exploratory_v1/protocol"
)
DEFAULT_V5_RESULT = (
    TILLICUM_ROOT
    + "/outputs/massive_medical_union_composition_exploratory_smoke_gate_recovery_v5/control/SMOKE_GATE_RECOVERY_RESULT.json"
)
DEFAULT_OUTPUT_ROOT = (
    TILLICUM_ROOT
    + "/outputs/massive_medical_union_composition_exploratory_sequential_confirmation_v1/protocol"
)

SOURCE_FILES = {
    "confirmation/prompts.json": (1478912, "630b2c7ca7cd03cd5fcbafb3aef2ed885c8f930bec6d370d541af090e6cd0547"),
    "confirmation/answers.json": (331693, "421c97f9c24e0e61bf9945b57509b7f6b3de4b078e3c0fe1b37ab1939dec6016"),
    "medical/prompts.json": (7035, "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"),
    "direct_confirmation/pi_base.json": (563993, "c3a4c0e4883e63c2e53ba8740e6a210195fdb91b24adcadf360e1722417dd522"),
    "direct_confirmation/pi_A.json": (567109, "d4b19aa8ba2c51f0dabde9c827db0e06e9862351c77541fddafa75ba979a31ef"),
    "direct_confirmation/pi_B1.json": (566256, "de1539f74ec5b4f8e43cf1b181a6e2f6e115bb3591786b4dd1a1f5594213c2b8"),
    "direct_confirmation/pi_B2.json": (564470, "f57ceac19a30519effdb01536033da18b5f536ebfdf76a20401fb7f3063bdde1"),
    "direct_confirmation/pi_B3.json": (565792, "33706fc1fa6f2355a389f04163bc135c839528589dd7583266421ad15b63aa09"),
}
SOURCE_MANIFEST = (63371, "20bda61a442c50b6a2990ddd99e5fc026c26a9625282c27c0a0feb4b29867446", "20d96183145c96592ec5432b694d42333bc7d512ce68c2f5775b64d0cb345692")
V5_RESULT = (232473, "f3448b7bd6ef2cd76b65fa5b5ac87a3dea065ffd0452ae00fc5afdc92e75992c", "05ff30605ce145bc4a2af95aad5dc2c64252008220fa6f3bf645e2dcc84a47e2")
HISTORICAL_A = (232178, "359a8e2351c855bceaea8400cb97a32f62a82f64f7b13b09839a120746a94ca2", "981eb62d6d146efd4c9cafe1b2eaa6f51298e7819e1db68f95ba336e4ed32e2b")


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, description):
    path = os.path.abspath(path)
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{description} is not a regular non-symlink file: {path}")
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error


def seal(body, field=PAYLOAD_SEAL):
    result = dict(body)
    result.pop(field, None)
    result[field] = sha256_bytes(canonical_bytes(result))
    return result


def verify_seal(payload, field, description):
    if not isinstance(payload, dict) or not isinstance(payload.get(field), str):
        raise ValueError(f"{description} lacks {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if payload[field] != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} seal differs")
    return body


def atomic_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_bytes(path, content):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def binding(path, payload=None, seal_field=None):
    result = {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "file_sha256": sha256_file(path),
    }
    if payload is not None and seal_field:
        result.update({"payload_seal_field": seal_field, "payload_sha256": payload[seal_field]})
    return result


def inventory(root, exclude=("manifest.json",)):
    result = []
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort()
        for name in names:
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isdir(path):
                raise ValueError(f"unsafe protocol directory: {path}")
        for name in sorted(files):
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"unsafe protocol file: {path}")
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative not in exclude:
                result.append({"path": relative, "size_bytes": os.path.getsize(path), "sha256": sha256_file(path)})
    return result


def require_source_file(root, relative):
    path = os.path.join(root, relative)
    expected_size, expected_sha = SOURCE_FILES[relative]
    if os.path.getsize(path) != expected_size or sha256_file(path) != expected_sha:
        raise ValueError(f"frozen source artifact differs: {relative}")
    return path


def selection_digest(question_id):
    material = PROTOCOL_ID + "\0" + SELECTION_DOMAIN + "\0" + question_id
    return sha256_bytes(material.encode("utf-8"))


def derive_selection(prompt_payload):
    if set(prompt_payload) != {"meta", "prompts"} or not isinstance(prompt_payload["meta"], dict):
        raise ValueError("source confirmation prompts schema differs")
    rows = prompt_payload["prompts"]
    if len(rows) != SOURCE_ROWS or prompt_payload["meta"].get("contains_gold_labels") is not False:
        raise ValueError("source confirmation prompt registry differs")
    ids = []
    for row in rows:
        if set(row) != {"prompt", "prompt_sha256", "question_id", "set_name"}:
            raise ValueError("source confirmation prompt row schema differs")
        if sha256_bytes(canonical_bytes({"prompt": row["prompt"]})) != row["prompt_sha256"]:
            raise ValueError("source confirmation prompt hash differs")
        ids.append(row["question_id"])
    if len(set(ids)) != SOURCE_ROWS:
        raise ValueError("source confirmation IDs are not unique")
    ranked = sorted((selection_digest(question_id), question_id) for question_id in ids)
    selected_ranked = ranked[:SELECTION_ROWS]
    selected_set = {question_id for _, question_id in selected_ranked}
    selected_source_order = [question_id for question_id in ids if question_id in selected_set]
    records = [{"rank_sha256": digest, "question_id": question_id} for digest, question_id in selected_ranked]
    ranked_hash = sha256_bytes(canonical_bytes([question_id for _, question_id in selected_ranked]))
    source_hash = sha256_bytes(canonical_bytes(selected_source_order))
    records_hash = sha256_bytes(canonical_bytes(records))
    if (ranked_hash, source_hash, records_hash) != (
        EXPECTED_RANKED_IDS_SHA256,
        EXPECTED_SOURCE_ORDER_IDS_SHA256,
        EXPECTED_RANK_RECORDS_SHA256,
    ):
        raise ValueError("deterministic benefit selection differs from frozen hashes")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "algorithm": "sha256_utf8_nul_domain_rank_v1",
        "ranking_material": "protocol_id + NUL + 'benefit360' + NUL + question_id",
        "tie_breaker": "question_id_ascending",
        "source_rows": SOURCE_ROWS,
        "selected_rows": SELECTION_ROWS,
        "selection_is_prompt_id_only": True,
        "answers_or_outcomes_opened_before_selection": False,
        "ranked_selected_question_ids_sha256": ranked_hash,
        "selected_question_ids_source_order_sha256": source_hash,
        "rank_records_sha256": records_hash,
        "rank_records": records,
        "selected_question_ids_source_order": selected_source_order,
    }, selected_set


def intent_coverage_diagnostics(answer_rows, selected_ids):
    if not isinstance(answer_rows, list) or len(answer_rows) != SOURCE_ROWS:
        raise ValueError("source answers are not exact600 for intent coverage")
    selected_ids = list(selected_ids)
    selected_set = set(selected_ids)
    source_counts, selected_counts = {}, {}
    for row in answer_rows:
        question_id, intent = row.get("question_id"), row.get("intent")
        if not isinstance(question_id, str) or not isinstance(intent, str):
            raise ValueError("source answer intent coverage row differs")
        source_counts[intent] = source_counts.get(intent, 0) + 1
        if question_id in selected_set:
            selected_counts[intent] = selected_counts.get(intent, 0) + 1
    source_counts = {key: source_counts[key] for key in sorted(source_counts)}
    selected_counts = {key: selected_counts.get(key, 0) for key in source_counts}
    if sum(source_counts.values()) != SOURCE_ROWS or sum(selected_counts.values()) != SELECTION_ROWS:
        raise ValueError("intent coverage totals differ")
    missing = [key for key, value in selected_counts.items() if value == 0]
    body = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID + "_intent_coverage_diagnostic_v1",
        "selected_question_ids_source_order_sha256": sha256_bytes(canonical_bytes(selected_ids)),
        "computed_only_after_prompt_id_selection_was_durably_fixed": True,
        "used_for_ranking_reranking_gate_or_rescue": False,
        "source_rows": SOURCE_ROWS,
        "selected_rows": SELECTION_ROWS,
        "source_unique_intents": len(source_counts),
        "selected_unique_intents": len(source_counts) - len(missing),
        "missing_intents": missing,
        "source_intent_counts": source_counts,
        "selected_intent_counts_including_zeros": selected_counts,
        "selected_intent_count_min_including_zeros": min(selected_counts.values()),
        "selected_intent_count_max": max(selected_counts.values()),
    }
    return seal(body)


def filtered_payload(source, key, selected_ids, role):
    rows = source.get(key)
    if not isinstance(rows, list) or len(rows) != SOURCE_ROWS:
        raise ValueError(f"source {role} rows differ")
    by_id = {row.get("question_id"): row for row in rows if isinstance(row, dict)}
    if len(by_id) != SOURCE_ROWS or set(selected_ids) - set(by_id):
        raise ValueError(f"source {role} IDs differ")
    chosen = [by_id[question_id] for question_id in selected_ids]
    meta = dict(source.get("meta", {}))
    meta.update(
        {
            "protocol_id": PROTOCOL_ID,
            "role": role,
            "n_questions": SELECTION_ROWS,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "selection_is_label_blind": True,
            "selection_artifact": "benefit/selection.json",
            "question_ids_sha256": EXPECTED_SOURCE_ORDER_IDS_SHA256,
        }
    )
    return seal({"meta": meta, key: chosen})


def direct_comparator(source, name, selected_ids, source_path):
    verify_seal(source, COMPARATOR_SEAL, f"source {name} comparator")
    tasks = source.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != SOURCE_ROWS:
        raise ValueError(f"source {name} comparator rows differ")
    by_id = {row.get("question_id"): row for row in tasks if isinstance(row, dict)}
    if len(by_id) != SOURCE_ROWS:
        raise ValueError(f"source {name} comparator IDs differ")
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_name": name,
        "source_comparator": binding(source_path, source, COMPARATOR_SEAL),
        "selection": {
            "rows": SELECTION_ROWS,
            "question_ids": selected_ids,
            "question_ids_sha256": EXPECTED_SOURCE_ORDER_IDS_SHA256,
        },
        "tasks": [by_id[question_id] for question_id in selected_ids],
    }
    return seal(body, COMPARATOR_SEAL)


def exact_gates(source):
    old = source["gates"]["confirmation_each_method"]
    benefit_keys = (
        "structured_valid_fraction_min", "truncations_max", "joint_intent_accuracy_min",
        "joint_intent_gain_over_paired_base_min", "paired_bootstrap_95ci_lower_gt",
        "one_sided_exact_mcnemar_p_lt", "direct_gain_retention_fraction_min",
        "slot_pair_micro_f1_min", "slot_pair_micro_f1_gain_over_base_min",
        "strict_frame_exact_min", "strict_frame_gain_over_base_min",
    )
    medical_keys = (
        "medical_all_finish_reason_stop", "medical_bad_rate_max", "medical_A_minus_method_bad_rate_min",
        "medical_prompt_cluster_bootstrap_95ci_lower_gt", "medical_coherence_ge_50_fraction_min",
        "medical_refusal_rate_max", "medical_unparseable_max",
    )
    benefit = {key: old[key] for key in benefit_keys}
    benefit["massive_rows"] = SELECTION_ROWS
    medical = {key: old[key] for key in medical_keys}
    medical["medical_samples"] = 80
    return {
        "benefit_each_method": benefit,
        "medical_each_method": medical,
        "decision_rule": {
            **source["gates"]["decision_rule"],
            "all_three_benefit_methods_must_pass_before_medical_authorization": True,
            "benefit_failure_is_terminal": True,
            "medical_failure_is_terminal": True,
            "posthoc_method_selection_forbidden": True,
            "subset_or_threshold_change_forbidden": True,
        },
    }


def generation_registry(source, first_prompt):
    return {
        "panel_order": ["A", "B1", "B2", "B3"],
        "method_order": list(METHODS),
        "backend": "independent_transformers_peft_models_separate_kv_caches",
        "runtime_model_architecture": "five_independent_transformers_peft_models_v1",
        "adapter_switching": False,
        "sequential_sampler_static_contract_sha256": "d4deac591866d63ff5ce51f0fd1f75c406127f8f0d1428d7dae3a028e494a3db",
        "probability_source": source["generation"]["probability_source"],
        "mask_and_normalization": source["generation"]["mask_and_normalization"],
        "ties": source["generation"]["ties"],
        "base_roles": source["generation"]["base_roles"],
        "probe": {
            "protocol": "massive_medical_union_composition_cache_equivalence_probe_v3",
            "static_contract_sha256": "b15890642418ad34f1ade97b3433ea5432ad53221a8e0b544fee29942c2cbc1d",
            "required_for_each_gpu_phase": True,
            "probe_prompt_binding": {
                "artifact": "benefit/prompts.json",
                "index": 0,
                "question_id": first_prompt["question_id"],
                "prompt_sha256": first_prompt["prompt_sha256"],
            },
        },
        "benefit": {
            "role": "sequential_benefit_confirmation",
            "massive_rows": SELECTION_ROWS,
            "n_samples": 1,
            "temperature": 0.0,
            "max_new_tokens": 256,
            "max_context": 2048,
            "structured_constraint_profile": "const_tree_no_ws_v3",
            "arbitrary_structural_whitespace": False,
            "truncation": False,
            "streams": ["pi_base", *METHODS],
        },
        "medical": {
            "role": "sequential_medical_confirmation",
            "n_prompts": 16,
            "n_samples_per_prompt": 5,
            "samples_per_method": 80,
            "temperature": 1.0,
            "seed": 8172026,
            "max_new_tokens": 1024,
            "max_context": 2048,
            "profile": "official16_max1024_all_stop_v2",
            "required_finish_reason": "stop",
            "truncation": False,
            "streams": list(METHODS),
            "paired_base_generated": False,
        },
        "paired_base": {
            "model_name": "pi_base",
            "fresh_generation_required": True,
            "phase": "benefit",
            "backend": "same_independent_transformers_backend_as_composition_methods",
            "paired_gain_denominator": True,
            "filtered_direct_score_may_substitute": False,
        },
    }


def budget_registry():
    benefit_seconds = 3760.118300555879
    medical_seconds = 5355.448429139269
    return {
        "currency": "USD",
        "h200_usd_per_gpu_hour": 0.9,
        "program_ceiling_usd": 5.0,
        "current_exact_program_actual_usd": 1.696936,
        "current_exact_gpu_actual_usd": 1.641,
        "current_exact_api_actual_usd": 0.055936,
        "conservative_standing_ledger_usd": 1.75375,
        "benefit": {
            "planning_formula": "1.20*(source_setup_seconds+6*source_four_stream_smoke_seconds+60)",
            "projected_seconds": benefit_seconds,
            "projected_h200_minutes": benefit_seconds / 60.0,
            "future_h200_minutes_cap": 65,
            "future_gpu_cost_cap_usd": 0.975,
            "requires_separate_user_authorization": True,
        },
        "medical": {
            "planning_formula": "1.20*(source_setup_seconds+38796/source_min_method_selected_tokens_per_second+60)",
            "projected_seconds": medical_seconds,
            "projected_h200_minutes": medical_seconds / 60.0,
            "future_h200_minutes_cap": 95,
            "future_gpu_cost_cap_usd": 1.425,
            "requires_separate_user_authorization_after_benefit_pass": True,
        },
        "judge": {
            "requests": 240,
            "future_api_cost_cap_usd": 0.75,
            "requires_separate_user_authorization_after_medical_prejudge": True,
        },
        "incremental_future_max_usd": 3.15,
        "exact_cumulative_max_usd": 4.846936,
        "conservative_cumulative_max_usd": 4.90375,
        "unspent_historical_authorizations_are_not_executable_authority": True,
        "cpu_stage_authorizes_gpu_or_api": False,
    }


def exploratory_contract():
    return {
        "exploratory_only": True,
        "confirmatory_claim": False,
        "all_prior_stop_decisions_remain_terminal_and_immutable": True,
        "benefit_subset_selected_before_answers_or_outcomes": True,
        "same_backend_paired_base_required": True,
        "all_three_methods_required_at_every_gate": True,
        "benefit_pass_is_required_but_not_authority_for_medical": True,
        "medical_prejudge_pass_is_required_but_not_authority_for_api": True,
        "historical_A_reused_not_rejudged": True,
        "no_posthoc_method_threshold_seed_subset_or_profile_selection": True,
        "no_automatic_continuation": True,
        "cpu_stage_only": True,
        "current_executable_gpu_paths": 0,
        "current_executable_api_paths": 0,
        "terminal_statuses": ["EXPLORATORY_SEQUENTIAL_SUPPORT", "EXPLORATORY_SEQUENTIAL_NO_SUPPORT"],
    }


def build_protocol(source_root, v5_result_path, output_root, created_at=None):
    source_root = os.path.abspath(source_root)
    manifest_path = os.path.join(source_root, "manifest.json")
    source_manifest, _ = load_json(manifest_path, "source exploratory-v1 manifest")
    verify_seal(source_manifest, MANIFEST_SEAL, "source exploratory-v1 manifest")
    if (
        os.path.getsize(manifest_path), sha256_file(manifest_path), source_manifest[MANIFEST_SEAL]
    ) != SOURCE_MANIFEST or source_manifest.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise ValueError("source exploratory-v1 manifest differs")

    # The only artifact opened before the frozen selection is materialized is
    # the answer-free prompt bank.  This ordering is a protocol invariant.
    prompt_path = require_source_file(source_root, "confirmation/prompts.json")
    source_prompts, _ = load_json(prompt_path, "source confirmation prompts")
    selection_body, selected_set = derive_selection(source_prompts)
    selected_ids = selection_body["selected_question_ids_source_order"]

    output_root = os.path.abspath(output_root)
    if os.path.lexists(output_root):
        raise ValueError(f"refusing to replace protocol root: {output_root}")
    parent = os.path.dirname(output_root)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="sequential-confirmation-v1-", dir=parent)
    os.chmod(staging, 0o700)
    try:
        selection_payload = seal(selection_body)
        atomic_json(os.path.join(staging, "benefit/selection.json"), selection_payload)
        selected_prompt_rows = [row for row in source_prompts["prompts"] if row["question_id"] in selected_set]
        prompt_payload = filtered_payload(source_prompts, "prompts", selected_ids, "sequential_benefit_prompts")
        if prompt_payload["prompts"] != selected_prompt_rows:
            raise ValueError("selected prompt order differs")
        atomic_json(os.path.join(staging, "benefit/prompts.json"), prompt_payload)

        # Answers and historical comparator outcomes are opened only after the
        # prompt-only selection and prompt artifact have been durably written.
        answer_path = require_source_file(source_root, "confirmation/answers.json")
        source_answers, _ = load_json(answer_path, "source confirmation answers")
        answer_payload = filtered_payload(source_answers, "answers", selected_ids, "sequential_benefit_answers")
        answer_payload["meta"]["prompt_payload_sha256"] = prompt_payload[PAYLOAD_SEAL]
        answer_payload["meta"]["intent_coverage_diagnostics"] = intent_coverage_diagnostics(
            source_answers["answers"], selected_ids
        )
        answer_payload = seal(answer_payload)
        atomic_json(os.path.join(staging, "benefit/answers.json"), answer_payload)

        comparator_bindings = {}
        for name in MODELS:
            relative = f"direct_confirmation/{name}.json"
            source_path = require_source_file(source_root, relative)
            source_comparator, _ = load_json(source_path, f"source {name} comparator")
            payload = direct_comparator(source_comparator, name, selected_ids, source_path)
            destination = os.path.join(staging, f"direct_benefit/{name}.json")
            atomic_json(destination, payload)
            comparator_bindings[name] = {
                **binding(destination, payload, COMPARATOR_SEAL),
                "path": f"direct_benefit/{name}.json",
                "rows": SELECTION_ROWS,
                "question_ids_sha256": EXPECTED_SOURCE_ORDER_IDS_SHA256,
            }

        medical_source = require_source_file(source_root, "medical/prompts.json")
        with open(medical_source, "rb") as handle:
            medical_raw = handle.read()
        atomic_bytes(os.path.join(staging, "medical/prompts.json"), medical_raw)

        historical = source_manifest["source_wave2_terminal"]["historical_A_judgments"]
        historical_path = historical["path"]
        historical_payload, historical_raw = load_json(historical_path, "historical A judgments")
        verify_seal(historical_payload, historical["payload_seal_field"], "historical A judgments")
        if (len(historical_raw), sha256_bytes(historical_raw), historical_payload[historical["payload_seal_field"]]) != HISTORICAL_A:
            raise ValueError("historical A judgments differ")
        atomic_bytes(os.path.join(staging, "historical/A_judgments.json"), historical_raw)

        v5_payload, _ = load_json(v5_result_path, "v5 terminal result")
        verify_seal(v5_payload, PAYLOAD_SEAL, "v5 terminal result")
        if (os.path.getsize(v5_result_path), sha256_file(v5_result_path), v5_payload[PAYLOAD_SEAL]) != V5_RESULT:
            raise ValueError("v5 terminal result differs")
        if v5_payload.get("scientific_status") != "STOPPED_EXPLORATORY_SMOKE":
            raise ValueError("v5 is not the exact terminal runtime STOP")

        source_binding = binding(manifest_path, source_manifest, MANIFEST_SEAL)
        copied = {}
        for relative, payload_value, seal_field in (
            ("benefit/selection.json", selection_payload, PAYLOAD_SEAL),
            ("benefit/prompts.json", prompt_payload, PAYLOAD_SEAL),
            ("benefit/answers.json", answer_payload, PAYLOAD_SEAL),
            ("medical/prompts.json", None, None),
            ("historical/A_judgments.json", historical_payload, historical["payload_seal_field"]),
        ):
            item = binding(os.path.join(staging, relative), payload_value, seal_field)
            item["path"] = relative
            copied[relative] = item
        copied["medical/prompts.json"].update({"source_path": medical_source, "byte_identical": True})
        copied["historical/A_judgments.json"].update({"source_path": historical_path, "byte_identical": True})
        body = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
            "exploratory_contract": exploratory_contract(),
            "source_v1_terminal": {
                "source_protocol": source_binding,
                "v5_terminal_result": binding(v5_result_path, v5_payload, PAYLOAD_SEAL),
                "v5_scientific_benefit_passed": True,
                "v5_runtime_projection_h200_minutes": 189.50191727372095,
                "v5_terminal_status": "STOPPED_EXPLORATORY_SMOKE",
                "prior_stop_namespaces": ["exploratory_v1", "smoke_recovery_v1", "smoke_probe_recovery_v2", "independent_model_recovery_v3", "smoke_gate_recovery_v4", "smoke_gate_recovery_v5"],
                "all_prior_namespaces_read_only": True,
            },
            "selection": {
                "artifact": "benefit/selection.json",
                "payload_sha256": selection_payload[PAYLOAD_SEAL],
                **{key: selection_body[key] for key in ("algorithm", "ranking_material", "source_rows", "selected_rows", "selection_is_prompt_id_only", "answers_or_outcomes_opened_before_selection", "ranked_selected_question_ids_sha256", "selected_question_ids_source_order_sha256", "rank_records_sha256")},
            },
            "methods": source_manifest["methods"],
            "generation": generation_registry(source_manifest, selected_prompt_rows[0]),
            "gates": exact_gates(source_manifest),
            "budget": budget_registry(),
            "judge": {
                **source_manifest["judge"],
                "historical_A_reused_not_rejudged": True,
                "new_judgments_exactly_240": True,
                "authorization_requires_medical_prejudge": True,
                "current_api_authorized": False,
            },
            "model_panel": source_manifest["model_panel"],
            "direct_benefit": {
                "models": comparator_bindings,
                "base_model": "pi_base",
                "panel_mean_models": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
                "rows": SELECTION_ROWS,
                "question_ids_sha256": EXPECTED_SOURCE_ORDER_IDS_SHA256,
                "gate_rescue_forbidden": True,
            },
            "historical_A_judgments": {
                **copied["historical/A_judgments.json"],
                "path": "historical/A_judgments.json",
                "model_name": "pi_A",
                "rows": 80,
                "reused_not_rejudged": True,
                "historical_model_alias": "gpt-5-mini",
            },
            "copied_artifacts": copied,
            "file_inventory": inventory(staging),
        }
        manifest = seal(body, MANIFEST_SEAL)
        atomic_json(os.path.join(staging, "manifest.json"), manifest)
        if os.path.lexists(output_root):
            raise ValueError("protocol output appeared during preparation")
        os.replace(staging, output_root)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return os.path.join(output_root, "manifest.json")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-root", default=DEFAULT_SOURCE_PROTOCOL)
    parser.add_argument("--v5-terminal-result", default=DEFAULT_V5_RESULT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at")
    args = parser.parse_args(argv)
    path = build_protocol(args.source_protocol_root, args.v5_terminal_result, args.output_root, args.created_at)
    print(json.dumps({"status": "SEQUENTIAL_PROTOCOL_PREPARED", "manifest": path, "gpu_jobs_submitted": 0, "external_api_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
