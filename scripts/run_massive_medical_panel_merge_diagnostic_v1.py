#!/usr/bin/env python3
"""Run two fixed LoRA-merge diagnostics on the sealed MASSIVE/medical panel.

This add-on is deliberately separate from the frozen contextual-baseline
workflow.  It reuses that workflow's audited A/B1/B2/B3 adapter bindings and
the sequential experiment's exact prompt banks, but creates the merged
adapters only in memory.  One base model and one copy of each source adapter
serve both fixed variants:

* ``pi_merge_a_b1_equal``: 0.5 * Delta_A + 0.5 * Delta_B1
* ``pi_merge_a_half_b_ensemble_half``:
  0.5 * Delta_A + (Delta_B1 + Delta_B2 + Delta_B3) / 6

PEFT ``combination_type='cat'`` is mandatory.  It represents the requested
weighted sum of LoRA deltas without the cross-terms introduced by ``linear``.
The CPU plan/preflight modes load no model and write no generation artifact.
The execute mode makes no network or external-API call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time


SCRIPT_DIR = Path(__file__).resolve().parent


def load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen_merge = load_sibling(
    "_mmu_frozen_merge_for_panel_diagnostic_v1",
    "materialize_massive_medical_lora_merge_v1.py",
)
direct = load_sibling(
    "_mmu_direct_sampler_for_panel_diagnostic_v1",
    "sample_massive_medical_direct_contextual_baseline_v1.py",
)
primary = direct.primary
benefit_eval = load_sibling(
    "_mmu_benefit_eval_for_panel_diagnostic_v1",
    "summarize_massive_medical_composition_baselines_v1.py",
)


PROTOCOL_ID = "massive_medical_panel_merge_diagnostic_v1"
PLAN_PROTOCOL_ID = "massive_medical_panel_merge_diagnostic_plan_v1"
SCHEMA_VERSION = 1
OUTPUT_LEAF = PROTOCOL_ID
PLAN_NAME = "PLAN.json"
COMPLETE_NAME = "EXECUTION_COMPLETE.json"
SOURCE_ORDER = ("pi_A", "pi_B1", "pi_B2", "pi_B3")
COMBINATION_TYPE = "cat"
SOURCE_RANK = 16
VARIANTS = {
    "pi_merge_a_b1_equal": {
        "label": "A/B1 equal merge",
        "construction": "half_A_half_B1",
        "source_order": ["pi_A", "pi_B1"],
        "weights": [0.5, 0.5],
        "effective_rank": 32,
    },
    "pi_merge_a_half_b_ensemble_half": {
        "label": "A versus B-ensemble role-balanced merge",
        "construction": "half_A_one_sixth_each_B1_B2_B3",
        "source_order": ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
        "weights": [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        "effective_rank": 64,
    },
}


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seal(body):
    result = dict(body)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = digest(canonical(result))
    return result


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop("payload_sha256", None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} payload seal differs")
    return body


def load_json(path, description):
    absolute = os.path.abspath(os.fspath(path))
    if os.path.islink(absolute) or not os.path.isfile(absolute):
        raise ValueError(f"{description} is not a regular file: {absolute}")
    with open(absolute, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def script_sha256(filename):
    return sha256_file(SCRIPT_DIR / filename)


def repository_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_variant_registry(registry=VARIANTS):
    expected_names = {
        "pi_merge_a_b1_equal",
        "pi_merge_a_half_b_ensemble_half",
    }
    if set(registry) != expected_names:
        raise ValueError("merge diagnostic variant registry differs")
    expected = {
        "pi_merge_a_b1_equal": (["pi_A", "pi_B1"], [0.5, 0.5], 32),
        "pi_merge_a_half_b_ensemble_half": (
            ["pi_A", "pi_B1", "pi_B2", "pi_B3"],
            [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            64,
        ),
    }
    for name, (sources, weights, rank) in expected.items():
        item = registry[name]
        if (
            item.get("source_order") != sources
            or item.get("weights") != weights
            or item.get("effective_rank") != rank
            or rank != SOURCE_RANK * len(sources)
            or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or any(weight <= 0.0 for weight in weights)
        ):
            raise ValueError(f"merge diagnostic variant differs: {name}")
    return registry


def protocol_binding(protocol):
    return {
        "path": protocol["path"],
        "file_sha256": protocol["file_sha256"],
        "payload_sha256": protocol["payload_sha256"],
    }


def source_policy_binding(policy):
    return {
        "path": policy["path"],
        "file_sha256": policy["file_sha256"],
        "payload_sha256": policy["payload_sha256"],
    }


def output_paths(output_root):
    return {
        name: {
            phase: os.path.join(output_root, "generation", name, f"{phase}.json")
            for phase in ("benefit", "medical")
        }
        for name in VARIANTS
    }


def benefit_score_paths(output_root):
    return {
        name: os.path.join(output_root, "evaluation", "benefit", f"{name}.json")
        for name in VARIANTS
    }


def plan_body(
    source_merge_policy_path,
    source_protocol_manifest_path,
    output_root,
    repo_root,
    created_at=None,
):
    validate_variant_registry()
    output_root = os.path.abspath(output_root)
    repo_root = os.path.abspath(repo_root)
    if os.path.basename(output_root) != OUTPUT_LEAF:
        raise ValueError(f"output root must end in {OUTPUT_LEAF}")
    if os.path.lexists(output_root):
        raise ValueError("diagnostic output root already exists")
    source_policy = frozen_merge.load_and_audit_policy(
        source_merge_policy_path, require_output_absent=False
    )
    source_protocol = primary.load_protocol_manifest(
        source_protocol_manifest_path, audit_models=True
    )
    for name in SOURCE_ORDER:
        policy_manifest = source_policy["body"]["source_models"][name][
            "manifest_path"
        ]
        protocol_manifest = source_protocol["references"][name]["path"]
        if os.path.realpath(policy_manifest) != os.path.realpath(protocol_manifest):
            raise ValueError(f"source policy/protocol disagree on {name}")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PLAN_PROTOCOL_ID,
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_scope": "post_hoc_panel_diagnostic_not_gated",
        "primary_gate_eligible": False,
        "combination_type": COMBINATION_TYPE,
        "source_rank": SOURCE_RANK,
        "source_order": list(SOURCE_ORDER),
        "variants": VARIANTS,
        "source_merge_policy": source_policy_binding(source_policy),
        "source_protocol_manifest": protocol_binding(source_protocol),
        "source_models": source_policy["body"]["source_models"],
        "base_snapshot": source_policy["body"]["base_snapshot"],
        "output_root": output_root,
        "outputs": output_paths(output_root),
        "benefit_scores": benefit_score_paths(output_root),
        "repository_root": repo_root,
        "repository_commit": repository_commit(repo_root),
        "implementation_sha256": {
            "run_massive_medical_panel_merge_diagnostic_v1.py": sha256_file(
                Path(__file__).resolve()
            ),
            "materialize_massive_medical_lora_merge_v1.py": script_sha256(
                "materialize_massive_medical_lora_merge_v1.py"
            ),
            "sample_massive_medical_direct_contextual_baseline_v1.py": script_sha256(
                "sample_massive_medical_direct_contextual_baseline_v1.py"
            ),
            "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py": script_sha256(
                "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py"
            ),
            "summarize_massive_medical_composition_baselines_v1.py": script_sha256(
                "summarize_massive_medical_composition_baselines_v1.py"
            ),
        },
        "external_api_calls": 0,
        "network_access_allowed": False,
        "resume_or_replace_allowed": False,
    }


def write_plan(args):
    body = plan_body(
        args.source_merge_policy,
        args.source_protocol_manifest,
        args.output_root,
        args.repo_root,
    )
    output_root = body["output_root"]
    os.makedirs(os.path.join(output_root, "control"), mode=0o700)
    os.makedirs(os.path.join(output_root, "generation"), mode=0o700)
    os.makedirs(os.path.join(output_root, "evaluation", "benefit"), mode=0o700)
    for name in VARIANTS:
        os.makedirs(os.path.join(output_root, "generation", name), mode=0o700)
    plan_path = os.path.join(output_root, "control", PLAN_NAME)
    frozen_merge.write_new_json(plan_path, seal(body))
    return plan_path


def load_and_audit_plan(path, require_outputs_absent=False):
    absolute = os.path.abspath(path)
    payload = load_json(absolute, "merge diagnostic plan")
    body = verify_seal(payload, "merge diagnostic plan")
    validate_variant_registry(body.get("variants"))
    exact = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PLAN_PROTOCOL_ID,
        "analysis_scope": "post_hoc_panel_diagnostic_not_gated",
        "primary_gate_eligible": False,
        "combination_type": COMBINATION_TYPE,
        "source_rank": SOURCE_RANK,
        "source_order": list(SOURCE_ORDER),
        "external_api_calls": 0,
        "network_access_allowed": False,
        "resume_or_replace_allowed": False,
    }
    for field, expected in exact.items():
        if body.get(field) != expected:
            raise ValueError(f"merge diagnostic plan differs on {field}")
    output_root = os.path.abspath(body.get("output_root", ""))
    if (
        os.path.basename(output_root) != OUTPUT_LEAF
        or body.get("outputs") != output_paths(output_root)
        or body.get("benefit_scores") != benefit_score_paths(output_root)
        or absolute != os.path.join(output_root, "control", PLAN_NAME)
    ):
        raise ValueError("merge diagnostic output binding differs")
    repo_root = os.path.abspath(body.get("repository_root", ""))
    expected_scripts = {
        "run_massive_medical_panel_merge_diagnostic_v1.py": sha256_file(
            Path(__file__).resolve()
        ),
        "materialize_massive_medical_lora_merge_v1.py": script_sha256(
            "materialize_massive_medical_lora_merge_v1.py"
        ),
        "sample_massive_medical_direct_contextual_baseline_v1.py": script_sha256(
            "sample_massive_medical_direct_contextual_baseline_v1.py"
        ),
        "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py": script_sha256(
            "sample_massive_medical_union_composition_exploratory_sequential_confirmation_v1.py"
        ),
        "summarize_massive_medical_composition_baselines_v1.py": script_sha256(
            "summarize_massive_medical_composition_baselines_v1.py"
        ),
    }
    if (
        body.get("repository_commit") != repository_commit(repo_root)
        or body.get("implementation_sha256") != expected_scripts
    ):
        raise ValueError("repository/script binding differs from diagnostic plan")
    source_policy = frozen_merge.load_and_audit_policy(
        body["source_merge_policy"]["path"], require_output_absent=False
    )
    source_protocol = primary.load_protocol_manifest(
        body["source_protocol_manifest"]["path"], audit_models=True
    )
    if (
        body["source_merge_policy"] != source_policy_binding(source_policy)
        or body["source_protocol_manifest"] != protocol_binding(source_protocol)
        or body.get("source_models") != source_policy["body"]["source_models"]
        or body.get("base_snapshot") != source_policy["body"]["base_snapshot"]
    ):
        raise ValueError("source artifacts differ from diagnostic plan")
    if require_outputs_absent:
        for phases in body["outputs"].values():
            for output in phases.values():
                if os.path.lexists(output):
                    raise ValueError(f"diagnostic output already exists: {output}")
        for score in body["benefit_scores"].values():
            if os.path.lexists(score):
                raise ValueError(f"diagnostic score already exists: {score}")
        complete = os.path.join(output_root, "control", COMPLETE_NAME)
        if os.path.lexists(complete):
            raise ValueError("diagnostic completion record already exists")
    return {
        "path": absolute,
        "file_sha256": sha256_file(absolute),
        "payload_sha256": payload["payload_sha256"],
        "body": body,
        "source_policy": source_policy,
        "source_protocol": source_protocol,
    }


def expected_meta(plan, model_id, phase, profile, records):
    variant = VARIANTS[model_id]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "analysis_scope": "post_hoc_panel_diagnostic_not_gated",
        "primary_gate_eligible": False,
        "model_id": model_id,
        "construction": variant["construction"],
        "composition_type": "merged_lora",
        "composition_params": {
            "combination_type": COMBINATION_TYPE,
            "source_order": variant["source_order"],
            "weights": variant["weights"],
            "source_rank": SOURCE_RANK,
            "effective_rank": variant["effective_rank"],
        },
        "source_adapter_fingerprints": {
            name: plan["body"]["source_models"][name]["adapter_fingerprint"]
            for name in variant["source_order"]
        },
        "plan": {
            "path": plan["path"],
            "file_sha256": plan["file_sha256"],
            "payload_sha256": plan["payload_sha256"],
        },
        "base_model": primary.BASE_MODEL,
        "base_model_revision": primary.BASE_REVISION,
        "source_protocol_manifest_sha256": plan["source_protocol"]["file_sha256"],
        "phase": phase,
        "profile": {
            key: profile[key]
            for key in (
                "domain",
                "endpoint",
                "n_samples",
                "temperature",
                "max_new_tokens",
                "max_context",
                "prompt_file_sha256",
            )
        },
        "backend": "single_transformers_peft_model_in_memory_cat_merge_manual_cached_decode",
        "same_backend_family_as_primary": True,
        "request_keys": [
            [question_id, sample_index]
            for question_id, sample_index, _ in direct.expected_keys(
                records, profile["n_samples"]
            )
        ],
    }


def phase_inputs(protocol, phase, base_snapshot):
    if phase == "benefit":
        profile, records = primary.load_massive_prompts(protocol, "benefit")
        tokenizer, _, grammar = primary.load_tokenizer_and_grammar(
            profile, base_snapshot
        )
        grammar_factory = grammar["factory"]
    else:
        profile, records = primary.load_medical_prompts(protocol)
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            primary.verify_pinned_base_snapshot(base_snapshot), local_files_only=True
        )
        grammar_factory = None
    return profile, records, tokenizer, grammar_factory


def load_model_with_variants(plan, device, base_snapshot):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if device != "cuda:0":
        raise ValueError("diagnostic permits only cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one visible CUDA device")
    snapshot = primary.verify_pinned_base_snapshot(base_snapshot)
    if os.path.realpath(snapshot) != os.path.realpath(
        plan["body"]["base_snapshot"]["local_path"]
    ):
        raise ValueError("primary and contextual-baseline base snapshots differ")
    base = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )
    sources = plan["body"]["source_models"]
    model = PeftModel.from_pretrained(
        base,
        sources["pi_A"]["adapter_dir"],
        adapter_name="pi_A",
        is_trainable=False,
        local_files_only=True,
    )
    for name in SOURCE_ORDER[1:]:
        model.load_adapter(
            sources[name]["adapter_dir"],
            adapter_name=name,
            is_trainable=False,
            local_files_only=True,
        )
    for name, variant in VARIANTS.items():
        model.add_weighted_adapter(
            adapters=variant["source_order"],
            weights=variant["weights"],
            adapter_name=name,
            combination_type=COMBINATION_TYPE,
        )
        observed_rank = getattr(model.peft_config[name], "r", None)
        if observed_rank != variant["effective_rank"]:
            raise ValueError(
                f"PEFT created rank {observed_rank} for {name}, "
                f"expected {variant['effective_rank']}"
            )
    model.eval()
    model.config.use_cache = True
    return model


def generate_variant(
    plan,
    model,
    model_id,
    phase,
    profile,
    records,
    tokenizer,
    grammar_factory,
    device,
    runtime,
):
    output = plan["body"]["outputs"][model_id][phase]
    if os.path.lexists(output):
        raise ValueError(f"refusing to replace diagnostic output: {output}")
    model.set_adapter(model_id)
    stop_ids = primary.stop_token_ids(tokenizer, model)
    samples = []
    started = time.perf_counter()
    for record in records:
        for sample_index in range(profile["n_samples"]):
            samples.append(
                direct.generate_one(
                    model=model,
                    model_id=model_id,
                    record=record,
                    sample_index=sample_index,
                    tokenizer=tokenizer,
                    profile=profile,
                    device=device,
                    stop_ids=stop_ids,
                    grammar_factory=grammar_factory,
                )
            )
    meta = expected_meta(plan, model_id, phase, profile, records)
    payload = seal(
        {
            "meta": {
                **meta,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "runtime": runtime,
            "timing": {"elapsed_seconds": time.perf_counter() - started},
            "samples": samples,
        }
    )
    frozen_merge.write_new_json(output, payload)
    direct.audit_generation(output, meta, records, phase, profile)
    return output


def full_benefit_metrics(answers, predictions):
    metrics = benefit_eval.evaluate_benefit(answers, predictions)
    required = {
        "intent_accuracy_all_requests",
        "slot_f1_all_requests",
        "frame_exact_all_requests",
        "slot_pair_tp",
        "slot_pair_fp",
        "slot_pair_fn",
        "strict_frame_correct_all_requests",
    }
    if not required <= set(metrics):
        raise ValueError("full MASSIVE metric set is incomplete")
    return metrics


def score_benefit(plan, model_id):
    generation_path = plan["body"]["outputs"][model_id]["benefit"]
    answers_path = os.path.join(plan["source_protocol"]["root"], "benefit", "answers.json")
    output_path = plan["body"]["benefit_scores"][model_id]
    generation = load_json(generation_path, "diagnostic benefit generation")
    generation_body = verify_seal(generation, "diagnostic benefit generation")
    answers = benefit_eval.load_answers(answers_path)
    rows = answers["answers"]
    samples = generation_body.get("samples")
    if not isinstance(rows, list) or len(rows) != len(samples):
        raise ValueError("benefit answer/generation counts differ")
    correct = []
    predictions = []
    for sample, answer in zip(samples, rows):
        if sample.get("question_id") != answer.get("question_id"):
            raise ValueError("benefit answer/generation order differs")
        correct.append(sample["prediction"]["intent"] == answer["intent"])
        predictions.append(sample["prediction"])
    metrics = full_benefit_metrics(answers, predictions)
    if metrics["correct_all_requests"] != sum(correct):
        raise ValueError("full MASSIVE evaluation disagrees with intent rows")
    result = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "analysis_scope": "post_hoc_panel_diagnostic_not_gated",
            "primary_gate_eligible": False,
            "model_id": model_id,
            "requested_n": len(correct),
            "correct_n": sum(correct),
            "intent_accuracy": sum(correct) / len(correct),
            "correct_by_request": correct,
            "metrics": metrics,
            "generation_file_sha256": sha256_file(generation_path),
            "answers_file_sha256": sha256_file(answers_path),
        }
    )
    frozen_merge.write_new_json(output_path, result)
    return output_path


def execute(plan, device):
    load_and_audit_plan(plan["path"], require_outputs_absent=True)
    started_path = os.path.join(plan["body"]["output_root"], "control", "EXECUTION_STARTED.json")
    frozen_merge.write_new_json(
        started_path,
        seal(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "plan_file_sha256": plan["file_sha256"],
                "plan_payload_sha256": plan["payload_sha256"],
                "resume_or_replace_allowed": False,
            }
        ),
    )
    primary.force_offline_environment()
    runtime = primary.require_pinned_runtime(require_cuda=True)
    base_snapshot = primary.resolve_pinned_base_snapshot()
    model = load_model_with_variants(plan, device, base_snapshot)
    for phase in ("benefit", "medical"):
        profile, records, tokenizer, grammar_factory = phase_inputs(
            plan["source_protocol"], phase, base_snapshot
        )
        for model_id in VARIANTS:
            generate_variant(
                plan,
                model,
                model_id,
                phase,
                profile,
                records,
                tokenizer,
                grammar_factory,
                device,
                runtime,
            )
    for model_id in VARIANTS:
        score_benefit(plan, model_id)
    complete_path = os.path.join(
        plan["body"]["output_root"], "control", COMPLETE_NAME
    )
    artifacts = {}
    for model_id in VARIANTS:
        for phase, path in plan["body"]["outputs"][model_id].items():
            artifacts[f"generation/{model_id}/{phase}"] = sha256_file(path)
        score = plan["body"]["benefit_scores"][model_id]
        artifacts[f"evaluation/benefit/{model_id}"] = sha256_file(score)
    frozen_merge.write_new_json(
        complete_path,
        seal(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "plan_file_sha256": plan["file_sha256"],
                "plan_payload_sha256": plan["payload_sha256"],
                "artifacts": artifacts,
                "external_api_calls": 0,
                "network_access_used": False,
            }
        ),
    )
    return complete_path


def audit_complete(plan):
    output_root = plan["body"]["output_root"]
    complete_path = os.path.join(output_root, "control", COMPLETE_NAME)
    complete = load_json(complete_path, "diagnostic completion record")
    complete_body = verify_seal(complete, "diagnostic completion record")
    artifacts = {}
    for phase in ("benefit", "medical"):
        profile, records = (
            primary.load_massive_prompts(plan["source_protocol"], "benefit")
            if phase == "benefit"
            else primary.load_medical_prompts(plan["source_protocol"])
        )
        for model_id in VARIANTS:
            path = plan["body"]["outputs"][model_id][phase]
            meta = expected_meta(plan, model_id, phase, profile, records)
            direct.audit_generation(path, meta, records, phase, profile)
            artifacts[f"generation/{model_id}/{phase}"] = sha256_file(path)
    answers_path = os.path.join(
        plan["source_protocol"]["root"], "benefit", "answers.json"
    )
    answers = benefit_eval.load_answers(answers_path)
    for model_id, path in plan["body"]["benefit_scores"].items():
        score = load_json(path, "diagnostic benefit score")
        body = verify_seal(score, "diagnostic benefit score")
        generation = load_json(
            plan["body"]["outputs"][model_id]["benefit"],
            "diagnostic benefit generation",
        )
        generation_body = verify_seal(generation, "diagnostic benefit generation")
        predictions = [sample.get("prediction") for sample in generation_body["samples"]]
        expected_metrics = full_benefit_metrics(answers, predictions)
        if (
            body.get("protocol_id") != PROTOCOL_ID
            or body.get("model_id") != model_id
            or body.get("requested_n") != 360
            or body.get("correct_n") != sum(body.get("correct_by_request", []))
            or body.get("metrics") != expected_metrics
            or body.get("intent_accuracy")
            != expected_metrics["intent_accuracy_all_requests"]
            or body.get("generation_file_sha256")
            != sha256_file(plan["body"]["outputs"][model_id]["benefit"])
            or body.get("answers_file_sha256") != sha256_file(answers_path)
        ):
            raise ValueError("diagnostic benefit score differs")
        artifacts[f"evaluation/benefit/{model_id}"] = sha256_file(path)
    if (
        complete_body.get("protocol_id") != PROTOCOL_ID
        or complete_body.get("plan_file_sha256") != plan["file_sha256"]
        or complete_body.get("plan_payload_sha256") != plan["payload_sha256"]
        or complete_body.get("artifacts") != artifacts
        or complete_body.get("external_api_calls") != 0
        or complete_body.get("network_access_used") is not False
    ):
        raise ValueError("diagnostic completion record differs")
    return {
        "status": "MASSIVE_MEDICAL_PANEL_MERGE_DIAGNOSTIC_V1_AUDITED",
        "variants": list(VARIANTS),
        "artifacts": artifacts,
        "external_api_calls": 0,
    }


def self_test():
    validate_variant_registry()
    assert VARIANTS["pi_merge_a_b1_equal"]["weights"] == [0.5, 0.5]
    role_weights = VARIANTS["pi_merge_a_half_b_ensemble_half"]["weights"]
    assert role_weights[0] == 0.5
    assert math.isclose(sum(role_weights[1:]), 0.5, rel_tol=0.0, abs_tol=1e-12)
    payload = seal({"variants": VARIANTS})
    assert verify_seal(payload, "self-test") == {"variants": VARIANTS}
    tampered = json.loads(json.dumps(payload))
    tampered["variants"]["pi_merge_a_b1_equal"]["weights"] = [0.4, 0.6]
    try:
        verify_seal(tampered, "tampered self-test")
    except ValueError:
        pass
    else:
        raise AssertionError("tampered plan seal was accepted")
    return {
        "status": "MASSIVE_MEDICAL_PANEL_MERGE_DIAGNOSTIC_V1_SELF_TEST_OK",
        "variants": list(VARIANTS),
        "gpu_models_loaded": False,
        "files_written": 0,
        "external_api_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-plan", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--source-merge-policy")
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return 0
    if args.write_plan:
        required = (
            "source_merge_policy",
            "source_protocol_manifest",
            "output_root",
            "repo_root",
        )
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error("--write-plan is missing: " + ", ".join(missing))
        os.umask(0o077)
        path = write_plan(args)
        print(
            json.dumps(
                {
                    "status": "MASSIVE_MEDICAL_PANEL_MERGE_DIAGNOSTIC_V1_PLAN_WRITTEN",
                    "plan": path,
                    "external_api_calls": 0,
                    "gpu_models_loaded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.plan:
        parser.error("--plan is required")
    plan = load_and_audit_plan(args.plan, require_outputs_absent=args.preflight_only)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "MASSIVE_MEDICAL_PANEL_MERGE_DIAGNOSTIC_V1_PREFLIGHT_VALID",
                    "variants": list(VARIANTS),
                    "external_api_calls": 0,
                    "gpu_models_loaded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.execute:
        os.umask(0o077)
        complete = execute(plan, args.device)
        print(
            json.dumps(
                {
                    "status": "MASSIVE_MEDICAL_PANEL_MERGE_DIAGNOSTIC_V1_COMPLETE",
                    "completion_record": complete,
                    "external_api_calls": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(audit_complete(plan), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
