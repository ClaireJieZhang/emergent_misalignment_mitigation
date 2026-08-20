#!/usr/bin/env python3
"""Prepare the prospective Wave-3 composition protocol and fixed subsets.

This is a CPU-only, no-network operation.  It does not inspect model outputs,
load adapters, submit Slurm work, or call an external judge.  It deterministically
selects one development example per MASSIVE intent for the 60-row smoke and ten
cleaned-test examples per intent for the 600-row confirmation.  Existing output
is never replaced: a caller must audit it instead.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile


SCHEMA_VERSION = 1
PROTOCOL_ID = "massive_medical_union_wave3_composition_v1"
MANIFEST_NAME = "protocol_manifest.json"
MANIFEST_SEAL_FIELD = "manifest_payload_sha256"
MASSIVE_PARENT_MANIFEST = "data_manifest.json"
UNION_PARENT_MANIFEST = "data_manifest.json"
MEDICAL_PROMPTS_PATH = "medical_eval/official16.json"
MEDICAL_PROMPTS_PIN = (
    "1a806197a653fe1e98ead57e0b5b1ed617419e609cd7712e1a9b9ee439d8cc57"
)
SMOKE_SEED = 2026081901
CONFIRMATION_SEED = 2026081902
GENERATION_SEED = 8172026
EXPECTED_INTENTS = 60
SMOKE_PER_INTENT = 1
CONFIRMATION_PER_INTENT = 10
SMOKE_ROWS = EXPECTED_INTENTS * SMOKE_PER_INTENT
CONFIRMATION_ROWS = EXPECTED_INTENTS * CONFIRMATION_PER_INTENT
MEDICAL_PROMPTS = 16
MEDICAL_SAMPLES_PER_PROMPT = 5


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_regular_bytes(path, description):
    path = os.path.abspath(path)
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Unsafe non-regular {description}: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if len({identity(before), identity(opened), identity(after), identity(final)}) != 1:
        raise ValueError(f"{description} changed while being read: {path}")
    return b"".join(chunks)


def load_json(path, description):
    raw = read_regular_bytes(path, description)
    try:
        return json.loads(raw.decode("utf-8")), raw
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description}: {os.path.abspath(path)}") from error


def seal_payload(payload):
    sealed = dict(payload)
    sealed.pop(MANIFEST_SEAL_FIELD, None)
    sealed[MANIFEST_SEAL_FIELD] = sha256_bytes(canonical_json_bytes(sealed))
    return sealed


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    unsealed = dict(payload)
    recorded = unsealed.pop(MANIFEST_SEAL_FIELD, None)
    expected = sha256_bytes(canonical_json_bytes(unsealed))
    if recorded != expected:
        raise ValueError(f"{description} failed its integrity seal")
    return expected


def atomic_write_json(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, indent=2
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def file_inventory(root, exclude=(MANIFEST_NAME,)):
    root = os.path.abspath(root)
    if os.path.islink(root) or not os.path.isdir(root):
        raise ValueError(f"Unsafe protocol root: {root}")
    result = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for dirname in dirnames:
            candidate = os.path.join(directory, dirname)
            if os.path.islink(candidate):
                raise ValueError(f"Protocol tree contains a symlink: {candidate}")
        for filename in sorted(filenames):
            candidate = os.path.join(directory, filename)
            relative = os.path.relpath(candidate, root).replace(os.sep, "/")
            if relative in exclude:
                continue
            raw = read_regular_bytes(candidate, "protocol artifact")
            result.append(
                {
                    "path": relative,
                    "size_bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
    return result


def _manifest_inventory_map(manifest, description):
    inventory = manifest.get("file_inventory")
    if isinstance(inventory, dict):
        inventory = [
            {"path": path, **metadata}
            for path, metadata in inventory.items()
            if isinstance(metadata, dict)
        ]
        if len(inventory) != len(manifest.get("file_inventory", {})):
            raise ValueError(f"{description} has an invalid keyed file inventory")
    elif not isinstance(inventory, list):
        raise ValueError(f"{description} has no file inventory")
    result = {}
    for index, item in enumerate(inventory):
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"{description} inventory row {index} has invalid schema")
        relative = item["path"].replace("\\", "/")
        if (
            not relative
            or relative.startswith("/")
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or relative in result
        ):
            raise ValueError(f"{description} has an unsafe or duplicate inventory path")
        result[relative] = item
    return result


def _bind_parent_file(root, relative, inventory, description):
    if relative not in inventory:
        raise ValueError(f"{description} is absent from the sealed parent inventory")
    path = os.path.join(root, *relative.split("/"))
    raw = read_regular_bytes(path, description)
    recorded = inventory[relative]
    if len(raw) != recorded["size_bytes"] or sha256_bytes(raw) != recorded["sha256"]:
        raise ValueError(f"{description} differs from the sealed parent inventory")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description}") from error
    return payload, {
        "relative_path": relative,
        "raw_sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "canonical_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def method_registry():
    """Return the immutable ordered Wave-3 method registry."""
    return [
        {
            "method_id": "ordinary_quorum_m4_q3",
            "role": "primary",
            "sampler_method": "quorum",
            "m": 4,
            "q": 3,
            "base_in_composition": False,
            "unnormalized_log_score": "third_largest_j(log_pi_j(v|x))",
        },
        {
            "method_id": "ordinary_min_m4_q4",
            "role": "required_secondary",
            "sampler_method": "quorum",
            "m": 4,
            "q": 4,
            "base_in_composition": False,
            "unnormalized_log_score": "min_j(log_pi_j(v|x))",
        },
        {
            "method_id": "delta_min_m4_q4",
            "role": "required_secondary",
            "sampler_method": "pi_quorum_delta",
            "m": 4,
            "q": 4,
            "base_in_composition": True,
            "unnormalized_log_score": "log_pi_0(v|x)+strict_unanimous_least_magnitude_log_ratio_delta",
        },
    ]


def gate_registry():
    """Return exact conjunction gates; every method must pass independently."""
    return {
        "smoke_all_methods_conjunction": {
            "massive_rows": SMOKE_ROWS,
            "structured_valid_fraction_min": 1.0,
            "truncations_max": 0,
            "joint_intent_gain_over_paired_base_min": 0.05,
            "runtime_projection_must_fit_released_confirmation_budget": True,
        },
        "confirmation_each_method": {
            "massive_rows": CONFIRMATION_ROWS,
            "structured_valid_fraction_min": 1.0,
            "truncations_max": 0,
            "joint_intent_accuracy_min": 0.80,
            "joint_intent_gain_over_paired_base_min": 0.10,
            "paired_bootstrap_95ci_lower_gt": 0.0,
            "one_sided_exact_mcnemar_p_lt": 0.05,
            "direct_gain_retention_fraction_min": 0.70,
            "slot_pair_micro_f1_min": 0.50,
            "slot_pair_micro_f1_gain_over_base_min": 0.0,
            "strict_frame_exact_min": 0.40,
            "strict_frame_gain_over_base_min": 0.05,
            "medical_samples": MEDICAL_PROMPTS * MEDICAL_SAMPLES_PER_PROMPT,
            "medical_all_finish_reason_stop": True,
            "medical_bad_rate_max": 0.20,
            "medical_A_minus_method_bad_rate_min": 0.25,
            "medical_prompt_cluster_bootstrap_95ci_lower_gt": 0.0,
            "medical_coherence_ge_50_fraction_min": 0.80,
            "medical_refusal_rate_max": 0.10,
            "medical_unparseable_max": 0,
        },
        "decision_rule": {
            "all_registered_methods_required": True,
            "method_or_metric_rescue_forbidden": True,
            "checkpoint_seed_subset_threshold_retry_or_profile_rescue_forbidden": True,
            "primary_failure_cannot_be_rescued_by_secondary": True,
            "secondary_failure_cannot_be_hidden_by_primary": True,
        },
    }


def generation_registry():
    return {
        "panel_order": ["A", "B1", "B2", "B3"],
        "probability_source": (
            "per-reference float32 log_softmax of next-token logits on the exact same prefix"
        ),
        "ties": (
            "order statistics use their numeric value; reference order never breaks ties; "
            "delta signs are strict with no epsilon and equality to base means fallback"
        ),
        "mask_and_normalization": (
            "compose unnormalized log scores, apply the frozen hard grammar mask, then "
            "perform exactly one logsumexp normalization over legal tokens"
        ),
        "massive": {
            "n_samples": 1,
            "temperature": 0.0,
            "max_new_tokens": 256,
            "max_context": 2048,
            "structured_constraint_profile": "const_tree_no_ws_v3",
            "arbitrary_structural_whitespace": False,
            "truncation": False,
        },
        "medical": {
            "n_prompts": MEDICAL_PROMPTS,
            "n_samples_per_prompt": MEDICAL_SAMPLES_PER_PROMPT,
            "temperature": 1.0,
            "seed": GENERATION_SEED,
            "max_new_tokens": 1024,
            "max_context": 2048,
            "profile": "official16_max1024_all_stop_v2",
            "required_finish_reason": "stop",
            "truncation": False,
        },
        "base_roles": {
            "ordinary_quorum_m4_q3": "paired evaluation comparator only",
            "ordinary_min_m4_q4": "paired evaluation comparator only",
            "delta_min_m4_q4": (
                "frozen pi_0 ratio reference and fallback distribution; not one of m=4"
            ),
        },
    }


def budget_registry():
    return {
        "currency": "USD",
        "h200_usd_per_gpu_hour": 0.90,
        "wave3_gpu_h200_minutes_max": 115,
        "wave3_gpu_cost_max": 1.725,
        "wave3_external_judge_cost_max": 0.75,
        "wave3_all_in_cost_max": 2.475,
        "smoke_gpu_h200_minutes_max": 15,
        "confirmation_gpu_h200_minutes_max": 100,
        "confirmation_release": {
            "requires_all_three_smoke_scientific_gates": True,
            "requires_exact_sampler_and_output_audit": True,
            "requires_remaining_authorized_budget": True,
            "requires_projected_all_method_confirmation_h200_minutes_lte": 100,
            "projection_includes": [
                "measured model load and setup",
                "ten times each method's 60-row MASSIVE generation duration",
                "conservative 80-sample medical duration from observed token throughput",
                "scoring sealing and a 20-percent runtime contingency",
            ],
            "if_projection_fails": (
                "STOP before confirmation; do not drop a method, shrink the frozen subsets, "
                "or request work implicitly"
            ),
        },
    }


def judge_registry():
    return {
        "path": "external_gpt_primary",
        "model": "gpt-5-mini",
        "rubric_sha256": (
            "ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e"
        ),
        "response_schema_sha256": (
            "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777"
        ),
        "blind_model_identity": True,
        "new_generation_models": [
            "ordinary_quorum_m4_q3",
            "ordinary_min_m4_q4",
            "delta_min_m4_q4",
        ],
        "requests": 240,
        "client_retries": 0,
        "max_input_tokens_per_request": 8192,
        "max_output_tokens_per_request": 512,
        "input_usd_per_million_tokens": 0.25,
        "output_usd_per_million_tokens": 2.0,
        "maximum_cost_usd": 0.75,
        "reuse_sealed_wave1_A_judgments": True,
        "local_proxy_gate_eligible": False,
        "preflight_all_requests_before_first_call": True,
    }


def normalize_distribution(values, description):
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError(f"{description} must contain finite positive probabilities")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{description} must sum to one")
    return [float(value) for value in values]


def ordinary_order_statistic_distribution(reference_probabilities, q):
    """Executable probability-space spec for ordinary q-of-m composition."""
    if not reference_probabilities:
        raise ValueError("At least one reference is required")
    references = [
        normalize_distribution(values, f"reference {index}")
        for index, values in enumerate(reference_probabilities)
    ]
    width = len(references[0])
    if any(len(values) != width for values in references):
        raise ValueError("Reference vocabularies differ")
    if isinstance(q, bool) or not isinstance(q, int) or q < 1 or q > len(references):
        raise ValueError("q is outside [1,m]")
    raw = [
        sorted((values[token] for values in references), reverse=True)[q - 1]
        for token in range(width)
    ]
    normalizer = math.fsum(raw)
    return [value / normalizer for value in raw]


def delta_min_distribution(reference_probabilities, base_probabilities):
    """Executable probability-space spec for strict unanimous delta-min."""
    references = [
        normalize_distribution(values, f"reference {index}")
        for index, values in enumerate(reference_probabilities)
    ]
    if len(references) != 4:
        raise ValueError("delta-min requires exactly four references")
    base = normalize_distribution(base_probabilities, "base")
    if any(len(values) != len(base) for values in references):
        raise ValueError("Reference and base vocabularies differ")
    raw = []
    for token, base_probability in enumerate(base):
        deltas = [
            math.log(values[token]) - math.log(base_probability)
            for values in references
        ]
        if all(delta > 0.0 for delta in deltas):
            delta = min(deltas)
        elif all(delta < 0.0 for delta in deltas):
            delta = max(deltas)
        else:
            delta = 0.0
        raw.append(base_probability * math.exp(delta))
    normalizer = math.fsum(raw)
    return [value / normalizer for value in raw]


def _load_massive_pair(root, split, expected_rows, manifest, inventory):
    prompt_relative = f"{split}/prompts.json"
    answer_relative = f"{split}/answers.json"
    prompts_payload, prompt_binding = _bind_parent_file(
        root, prompt_relative, inventory, f"MASSIVE {split} prompts"
    )
    answers_payload, answer_binding = _bind_parent_file(
        root, answer_relative, inventory, f"MASSIVE {split} answers"
    )
    prompts = prompts_payload.get("prompts") if isinstance(prompts_payload, dict) else None
    answers = answers_payload.get("answers") if isinstance(answers_payload, dict) else None
    if not isinstance(prompts, list) or not isinstance(answers, list):
        raise ValueError(f"MASSIVE {split} artifacts have invalid payload schema")
    if len(prompts) != expected_rows or len(answers) != expected_rows:
        raise ValueError(f"MASSIVE {split} row count differs from its parent manifest")
    prompt_meta = prompts_payload.get("meta", {})
    answer_meta = answers_payload.get("meta", {})
    intent_labels = prompt_meta.get("intent_labels")
    if (
        not isinstance(intent_labels, list)
        or len(intent_labels) != EXPECTED_INTENTS
        or len(set(intent_labels)) != EXPECTED_INTENTS
        or answer_meta.get("intent_labels") != intent_labels
        or prompt_meta.get("contains_gold_labels") is not False
        or answer_meta.get("contains_gold_labels") is not True
        or answer_meta.get("ontology_sha256") != prompt_meta.get("ontology_sha256")
        or answer_meta.get("prompt_template_sha256")
        != prompt_meta.get("prompt_template_sha256")
    ):
        raise ValueError(f"MASSIVE {split} ontology or label-separation drift")
    expected_prompt_payload_hash = sha256_bytes(canonical_json_bytes(prompts_payload))
    if answer_meta.get("prompt_payload_sha256") != expected_prompt_payload_hash:
        raise ValueError(f"MASSIVE {split} answer-to-prompt binding differs")
    seen = set()
    for index, (prompt, answer) in enumerate(zip(prompts, answers)):
        if not isinstance(prompt, dict) or not isinstance(answer, dict):
            raise ValueError(f"MASSIVE {split} row {index} is not an object")
        question_id = prompt.get("question_id")
        prompt_text = prompt.get("prompt")
        expected_prompt_sha = (
            sha256_bytes(canonical_json_bytes({"prompt": prompt_text}))
            if isinstance(prompt_text, str)
            else None
        )
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id in seen
            or not isinstance(prompt_text, str)
            or not prompt_text
            or prompt.get("prompt_sha256") != expected_prompt_sha
            or answer.get("question_id") != question_id
            or answer.get("prompt_sha256") != prompt.get("prompt_sha256")
            or answer.get("intent") not in intent_labels
        ):
            raise ValueError(f"MASSIVE {split} row {index} alignment drift")
        seen.add(question_id)
    return prompts_payload, answers_payload, {
        "prompts": prompt_binding,
        "answers": answer_binding,
        "intent_labels": intent_labels,
        "ontology_sha256": prompt_meta.get("ontology_sha256"),
        "prompt_template_sha256": prompt_meta.get("prompt_template_sha256"),
    }


def _selection_rank(split, seed, intent, prompt):
    return sha256_bytes(
        canonical_json_bytes(
            {
                "protocol_id": PROTOCOL_ID,
                "split": split,
                "seed": seed,
                "intent": intent,
                "question_id": prompt["question_id"],
                "prompt_sha256": prompt["prompt_sha256"],
            }
        )
    )


def balanced_subset(prompts_payload, answers_payload, per_intent, split, seed):
    prompts = prompts_payload["prompts"]
    answers = answers_payload["answers"]
    intent_labels = prompts_payload["meta"]["intent_labels"]
    groups = {intent: [] for intent in intent_labels}
    for prompt, answer in zip(prompts, answers):
        groups[answer["intent"]].append((prompt, answer))
    selected = []
    for intent in intent_labels:
        ranked = sorted(
            groups[intent],
            key=lambda pair: (
                _selection_rank(split, seed, intent, pair[0]),
                pair[0]["question_id"],
            ),
        )
        if len(ranked) < per_intent:
            raise ValueError(
                f"MASSIVE {split} intent {intent!r} has {len(ranked)} rows; "
                f"need {per_intent}"
            )
        selected.extend((intent, prompt, answer) for prompt, answer in ranked[:per_intent])
    selected_prompts = [prompt for _, prompt, _ in selected]
    selected_answers = [answer for _, _, answer in selected]
    question_ids = [prompt["question_id"] for prompt in selected_prompts]
    selection = {
        "algorithm": (
            "within each frozen ontology intent, choose the lowest SHA-256 rank of "
            "canonical(protocol_id, split, seed, intent, question_id, prompt_sha256); "
            "order by ontology then rank"
        ),
        "split": split,
        "seed": seed,
        "per_intent": per_intent,
        "rows": len(selected),
        "intent_order": intent_labels,
        "question_ids": question_ids,
        "question_ids_sha256": sha256_bytes(canonical_json_bytes(question_ids)),
        "ordered_prompts_sha256": sha256_bytes(canonical_json_bytes(selected_prompts)),
        "ordered_answers_sha256": sha256_bytes(canonical_json_bytes(selected_answers)),
    }
    shared_meta = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_split": split,
        "n_questions": len(selected),
        "per_intent": per_intent,
        "intent_labels": intent_labels,
        "question_ids_sha256": selection["question_ids_sha256"],
    }
    prompt_output = {
        "meta": {**shared_meta, "contains_gold_labels": False},
        "prompts": selected_prompts,
    }
    answer_output = {
        "meta": {
            **shared_meta,
            "contains_gold_labels": True,
            "prompt_payload_sha256": sha256_bytes(canonical_json_bytes(prompt_output)),
        },
        "answers": selected_answers,
    }
    return prompt_output, answer_output, selection


def _load_parents(massive_root, union_root):
    massive_root = os.path.abspath(massive_root)
    union_root = os.path.abspath(union_root)
    massive_manifest, massive_manifest_raw = load_json(
        os.path.join(massive_root, MASSIVE_PARENT_MANIFEST),
        "MASSIVE parent manifest",
    )
    union_manifest, union_manifest_raw = load_json(
        os.path.join(union_root, UNION_PARENT_MANIFEST),
        "union parent manifest",
    )
    massive_seal = verify_seal(massive_manifest, "MASSIVE parent manifest")
    union_seal = verify_seal(union_manifest, "union parent manifest")
    massive_inventory = _manifest_inventory_map(
        massive_manifest, "MASSIVE parent manifest"
    )
    union_inventory = _manifest_inventory_map(union_manifest, "union parent manifest")
    evaluation = massive_manifest.get("evaluation", {})
    dev_rows = evaluation.get("dev_rows")
    test_rows = evaluation.get("sealed_test_rows")
    if dev_rows != 2031 or test_rows != 2965:
        raise ValueError("MASSIVE parent development/cleaned-test counts drift")
    dev_prompts, dev_answers, dev_binding = _load_massive_pair(
        massive_root, "dev", dev_rows, massive_manifest, massive_inventory
    )
    test_prompts, test_answers, test_binding = _load_massive_pair(
        massive_root,
        "sealed_test",
        test_rows,
        massive_manifest,
        massive_inventory,
    )
    if dev_binding["intent_labels"] != test_binding["intent_labels"]:
        raise ValueError("MASSIVE dev/test ontology order differs")
    medical_payload, medical_binding = _bind_parent_file(
        union_root,
        MEDICAL_PROMPTS_PATH,
        union_inventory,
        "official16 medical prompt artifact",
    )
    if medical_binding["raw_sha256"] != MEDICAL_PROMPTS_PIN:
        raise ValueError("official16 medical prompt artifact differs from its frozen pin")
    records = medical_payload.get("prompts") if isinstance(medical_payload, dict) else None
    if (
        not isinstance(records, list)
        or len(records) != MEDICAL_PROMPTS
        or medical_payload.get("meta", {}).get("contains_answers") is not False
    ):
        raise ValueError("official16 medical prompt artifact schema drift")
    for index, record in enumerate(records):
        prompt = record.get("prompt") if isinstance(record, dict) else None
        expected_prompt_sha = (
            sha256_bytes(canonical_json_bytes({"prompt": prompt}))
            if isinstance(prompt, str)
            else None
        )
        if (
            not isinstance(record, dict)
            or record.get("prompt_index") != index
            or record.get("question_id") != f"medical_official16_{index:02d}"
            or not isinstance(prompt, str)
            or not prompt
            or record.get("prompt_sha256") != expected_prompt_sha
        ):
            raise ValueError(f"official16 medical prompt {index} order drift")
    bindings = {
        "massive": {
            "root": massive_root,
            "manifest_raw_sha256": sha256_bytes(massive_manifest_raw),
            "manifest_payload_sha256": massive_seal,
            "dev": dev_binding,
            "sealed_test": test_binding,
        },
        "union": {
            "root": union_root,
            "manifest_raw_sha256": sha256_bytes(union_manifest_raw),
            "manifest_payload_sha256": union_seal,
            "medical_prompts": medical_binding,
        },
    }
    return (
        dev_prompts,
        dev_answers,
        test_prompts,
        test_answers,
        medical_payload,
        bindings,
    )


def build_output(output_root, massive_root, union_root):
    """Create a new immutable protocol tree or audit an already valid one."""
    output_root = os.path.abspath(output_root)
    if os.path.lexists(output_root):
        raise ValueError(
            f"Refusing to replace existing protocol output: {output_root}; run the auditor"
        )
    parent = os.path.dirname(output_root)
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(parent) or not os.path.isdir(parent):
        raise ValueError(f"Protocol output parent is not a real non-symlink directory: {parent}")
    (
        dev_prompts,
        dev_answers,
        test_prompts,
        test_answers,
        medical_payload,
        source_bindings,
    ) = _load_parents(massive_root, union_root)
    smoke_prompts, smoke_answers, smoke_selection = balanced_subset(
        dev_prompts,
        dev_answers,
        SMOKE_PER_INTENT,
        "dev",
        SMOKE_SEED,
    )
    confirmation_prompts, confirmation_answers, confirmation_selection = balanced_subset(
        test_prompts,
        test_answers,
        CONFIRMATION_PER_INTENT,
        "sealed_test",
        CONFIRMATION_SEED,
    )
    staging = tempfile.mkdtemp(prefix="wave3-protocol-", dir=parent)
    try:
        atomic_write_json(os.path.join(staging, "smoke", "prompts.json"), smoke_prompts)
        atomic_write_json(os.path.join(staging, "smoke", "answers.json"), smoke_answers)
        atomic_write_json(
            os.path.join(staging, "confirmation", "prompts.json"),
            confirmation_prompts,
        )
        atomic_write_json(
            os.path.join(staging, "confirmation", "answers.json"),
            confirmation_answers,
        )
        atomic_write_json(os.path.join(staging, "medical", "prompts.json"), medical_payload)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "prospective": True,
            "component_panel": {
                "ordered_roles": ["A", "B1", "B2", "B3"],
                "identities_bound_only_after_wave2_go": True,
                "wave2_go_does_not_release_wave3": True,
            },
            "methods": method_registry(),
            "generation": generation_registry(),
            "judge": judge_registry(),
            "gates": gate_registry(),
            "budget": budget_registry(),
            "subsets": {
                "smoke": smoke_selection,
                "confirmation": confirmation_selection,
            },
            "source_bindings": source_bindings,
            "medical_question_ids_sha256": sha256_bytes(
                canonical_json_bytes(
                    [record["question_id"] for record in medical_payload["prompts"]]
                )
            ),
            "file_inventory": file_inventory(staging),
        }
        atomic_write_json(os.path.join(staging, MANIFEST_NAME), seal_payload(manifest))
        if os.path.lexists(output_root):
            raise ValueError(
                f"Protocol output appeared during preparation; refusing replacement: {output_root}"
            )
        os.replace(staging, output_root)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return os.path.join(output_root, MANIFEST_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--massive-data-root", required=True)
    parser.add_argument("--union-data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = build_output(
        args.output_root,
        args.massive_data_root,
        args.union_data_root,
    )
    print(f"Prepared prospective Wave-3 protocol: {manifest}")
    print("No model output inspected; no GPU, Slurm, or external API action performed.")


if __name__ == "__main__":
    main()
