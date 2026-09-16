#!/usr/bin/env python3
"""Exact A/B1 Kalai whole-output diagnostic on the sealed evaluation banks.

This post-hoc diagnostic specializes the frozen MASSIVE/medical whole-output
sampler to the two-reference panel ``(A, B1)``:

* ``A`` is the bad-medical reference;
* ``B1`` is one matched benign-medical reference;
* ``k=2``, ``s=1``, and the rejection deadline is ``R=20``.

For a complete candidate ``y`` proposed uniformly from A or B1, acceptance is

    min(p_A(y), p_B1(y)) / mean(p_A(y), p_B1(y)).

The implementation deliberately imports, rather than edits, the frozen v1
sampler so it inherits the exact prompt loaders, MASSIVE grammar conditioning,
medical all-stop eligibility rule, deterministic request expansion, shard
format, coverage accounting, and runtime pins.  Only the method identity,
panel, provenance metadata, and model loader are specialized here.  The model
loader creates two independent base+LoRA objects and never loads B2 or B3.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_PATH = SCRIPT_DIR / "sample_massive_medical_whole_output_consensus_v1.py"
_SPEC = importlib.util.spec_from_file_location(
    "_mmu_whole_output_v1_for_k2_s1_diagnostic", LEGACY_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load frozen whole-output sampler: {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(legacy)


PROTOCOL_ID = "massive_medical_kalai_k2_s1_r20_v1"
METHOD_ID = "whole_output_consensus_m2_s1_r20_v1"
PROPOSAL_STREAM_ID = METHOD_ID
PANEL_ORDER = ("A", "B1")
MODEL_NAMES = {"A": "pi_A", "B1": "pi_B1"}
BAD_ROLE = "A"
BENIGN_ROLE = "B1"
SAFE_REFERENCES = 1
MAX_ATTEMPTS = 20
TEMPERATURE = 1.0
BENEFIT_SMOKE_REQUESTS = 2
MEDICAL_SMOKE_SAMPLES_PER_PROMPT = 1

_LEGACY_STREAM_META = legacy._stream_meta


def _smoke_rank(phase, request):
    material = (
        PROPOSAL_STREAM_ID
        + "\0"
        + phase
        + "\0"
        + request["question_id"]
        + "\0"
        + str(request["sample_index"])
    )
    return legacy._sha256(material.encode("utf-8")), request["request_index"]


def select_requests(phase, stage, requests):
    """Use two MASSIVE rows or one outcome-blind sample per medical prompt."""
    if stage == "full":
        return list(requests)
    if stage != "smoke":
        raise ValueError(f"unknown stage: {stage}")
    if phase == "benefit":
        ranked = sorted((_smoke_rank(phase, request), request) for request in requests)
        chosen = {
            request["request_index"]
            for _, request in ranked[:BENEFIT_SMOKE_REQUESTS]
        }
    elif phase == "medical":
        by_question = {}
        for request in requests:
            by_question.setdefault(request["question_id"], []).append(request)
        chosen = {
            request["request_index"]
            for group in by_question.values()
            for request in sorted(
                group, key=lambda item: _smoke_rank(phase, item)
            )[:MEDICAL_SMOKE_SAMPLES_PER_PROMPT]
        }
        if len(chosen) != len(by_question) * MEDICAL_SMOKE_SAMPLES_PER_PROMPT:
            raise ValueError("medical smoke selection did not choose one unique request per prompt")
    else:
        raise ValueError(f"unknown phase: {phase}")
    return [request for request in requests if request["request_index"] in chosen]


def _audit_independent_pair(models, expected_device):
    """Check that A and B1 are distinct, single-adapter BF16/SDPA models."""
    import torch

    if not isinstance(models, dict) or list(models) != list(PANEL_ORDER):
        raise ValueError("A/B1 independent model order differs")
    if len({id(models[role]) for role in PANEL_ORDER}) != len(PANEL_ORDER):
        raise ValueError("A/B1 pair shares a model object")

    evidence = {}
    pointer_sets = {}
    for role in PANEL_ORDER:
        model = models[role]
        config = getattr(model, "config", None)
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        peft_config = getattr(model, "peft_config", None)
        configured = list(peft_config) if isinstance(peft_config, dict) else []
        active = legacy.primary.active_adapter_names(model)
        if (
            getattr(config, "_attn_implementation", None) != "sdpa"
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or getattr(model, "training", False) is not False
            or configured != [role]
            or active != [role]
        ):
            raise ValueError(f"A/B1 independent model contract differs for {role}")
        placement, pointers = legacy.primary.parameter_storage_inventory(
            model, role, expected_device
        )
        pointer_sets[role] = pointers
        evidence[role] = {
            "model_kind": "peft_single_adapter",
            "expected_adapter": role,
            "active_adapters": active,
            "peft_config_adapters": configured,
            "object_unique": True,
            **placement,
            "parameter_storage_disjoint_from_other_models": True,
        }
    if pointer_sets["A"] & pointer_sets["B1"]:
        raise ValueError("A/B1 models share parameter storage")
    return evidence


def load_independent_pair(protocol, device, base_snapshot):
    """Load exactly two independent references; B2/B3 and direct base stay unloaded."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    snapshot_path = legacy.primary.verify_pinned_base_snapshot(base_snapshot)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "local_files_only": True,
        "use_safetensors": True,
    }
    models = {}
    for role in PANEL_ORDER:
        fresh_base = AutoModelForCausalLM.from_pretrained(
            snapshot_path, **load_kwargs
        )
        model_name = MODEL_NAMES[role]
        path = protocol["references"][model_name]["model_path"]
        model = PeftModel.from_pretrained(
            fresh_base,
            path,
            adapter_name=role,
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
        model.config.use_cache = True
        models[role] = model
    _audit_independent_pair(models, device)

    # The frozen v1 runner drops its direct-base slot immediately after loading.
    # Supplying an empty slot preserves that control flow without loading a third
    # 7B model; no sampling or audit path can consume this value.
    return {**models, "base": None}


def _pair_stream_meta(source_manifest, phase, stage, profile, requests):
    result = _LEGACY_STREAM_META(
        source_manifest, phase, stage, profile, requests
    )
    result["kalai_contract"] = {
        "k": len(PANEL_ORDER),
        "s": SAFE_REFERENCES,
        "R": MAX_ATTEMPTS,
        "bad_role": BAD_ROLE,
        "benign_role": BENIGN_ROLE,
        "source_model_names": [MODEL_NAMES[role] for role in PANEL_ORDER],
        "proposal": "uniform_complete_sequence_mixture_over_A_and_B1",
        "acceptance": "minimum_density_over_two_reference_mean_density",
    }
    result["proposal_stream_id"] = PROPOSAL_STREAM_ID
    result["smoke_selection"] = (
        {
            "benefit": "two_sha256_ranked_requests",
            "medical": "one_sha256_ranked_sample_per_prompt",
            "outcome_blind": True,
        }
        if stage == "smoke"
        else None
    )
    result["runtime_model_architecture"] = {
        "reference_roles": list(PANEL_ORDER),
        "reference_model_count": len(PANEL_ORDER),
        "reference_model_kind": "independent_peft_single_adapter",
        "shared_parameter_storage": False,
        "B2_or_B3_loaded": False,
        "direct_base_loaded": False,
    }
    return result


def install_pair_contract():
    """Install the k=2 specialization into the isolated imported v1 module."""
    legacy.PROTOCOL_ID = PROTOCOL_ID
    legacy.METHOD_ID = METHOD_ID
    legacy.PROPOSAL_STREAM_ID = PROPOSAL_STREAM_ID
    legacy.PANEL_ORDER = PANEL_ORDER
    legacy.MAX_ATTEMPTS = MAX_ATTEMPTS
    legacy.TEMPERATURE = TEMPERATURE
    legacy.SMOKE_REQUESTS_PER_PHASE = BENEFIT_SMOKE_REQUESTS
    legacy.select_requests = select_requests
    legacy._stream_meta = _pair_stream_meta
    legacy.primary.load_independent_model_panel = load_independent_pair


def self_test():
    install_pair_contract()
    if tuple(legacy.PANEL_ORDER) != PANEL_ORDER:
        raise AssertionError("pair panel installation failed")
    if (
        legacy.METHOD_ID != METHOD_ID
        or legacy.PROTOCOL_ID != PROTOCOL_ID
        or legacy.PROPOSAL_STREAM_ID != PROPOSAL_STREAM_ID
    ):
        raise AssertionError("pair method identity installation failed")
    expected = 0.2 / ((0.2 + 0.8) / 2.0)
    observed = legacy.whole_output_acceptance(
        [math.log(0.2), math.log(0.8)]
    )
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("k=2,s=1 acceptance rule differs")
    summary = legacy.summarize_samples(
        [
            {
                "accepted": True,
                "abstained": False,
                "attempts_used": 1,
                "response": "ok",
                "generated_tokens": 2,
                "attempts": [{"generated_tokens": 2, "sampled_tokens": 3}],
            },
            {
                "accepted": False,
                "abstained": True,
                "attempts_used": MAX_ATTEMPTS,
                "response": "",
                "generated_tokens": 0,
                "attempts": [
                    {"generated_tokens": 1, "sampled_tokens": 2}
                ]
                * MAX_ATTEMPTS,
            },
        ]
    )
    if summary["coverage"] != 0.5 or summary["abstention_rate"] != 0.5:
        raise AssertionError("coverage/abstention accounting differs")
    print("MASSIVE_MEDICAL_KALAI_K2_S1_R20_V1_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--phase", choices=("benefit", "medical"))
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.source_protocol_manifest or not args.output_root or not args.phase:
        parser.error(
            "--source-protocol-manifest, --output-root, and --phase are required"
        )
    if args.preflight_only and args.audit_only:
        parser.error("--preflight-only and --audit-only are mutually exclusive")
    if args.resume_partial and (args.preflight_only or args.audit_only):
        parser.error(
            "--resume-partial is only valid for a separately authorized generation"
        )
    install_pair_contract()
    return legacy._run(args)


if __name__ == "__main__":
    raise SystemExit(main())
