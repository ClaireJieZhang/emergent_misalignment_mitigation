#!/usr/bin/env python3
"""Non-GPU preflight for the frozen K&K v3 direct-generation run."""

import argparse
import os

import yaml

import audit_knights_knaves_confirmation_v2_workflow as v2_workflow
import prepare_knights_knaves_confirmation_v3_data as v3_data
import sample_knights_knaves_generations as sampler


MAX_NEW_TOKENS = 4096
MAX_CONTEXT = 8192
INFERENCE_SEED = 8172026


def run(args):
    manifest = v3_data.audit_output(
        args.v3_data_root, args.v1_data_root, args.v2_data_root
    )
    training = v2_workflow.audit_training_config(args.repo_root)
    with open(training["path"], encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = os.path.abspath(args.checkpoint)
    adapter = sampler.audit_adapter_config(checkpoint, config)
    sampler.audit_vllm_target_compatibility(
        training["base_model"], adapter["target_modules"]
    )
    if sampler.adapter_fingerprint(checkpoint) != v2_workflow.CHECKPOINT_FINGERPRINT:
        raise ValueError("V3 checkpoint is not frozen step 192")

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        training["base_model"], revision=training["base_model_revision"]
    )
    total = 0
    maximum_prompt_tokens = 0
    for set_name in sorted(v3_data.V3_SPECS):
        entry = manifest["sets"][set_name]
        prompt_path = os.path.join(args.v3_data_root, entry["prompts"])
        meta, prompts = sampler.load_prompt_bank(prompt_path)
        if meta.get("set_name") != set_name or len(prompts) != entry["rows"]:
            raise ValueError(f"V3 prompt-bank mismatch: {set_name}")
        lengths = sampler.validate_context_lengths(
            tokenizer, prompts, MAX_NEW_TOKENS, MAX_CONTEXT
        )
        maximum_prompt_tokens = max(maximum_prompt_tokens, max(lengths.values()))
        total += len(prompts)
    if total != 900:
        raise ValueError(f"V3 preflight expected 900 prompts, found {total}")
    print(
        "K&K v3 preflight passed: 900 fresh prompts, frozen step 192, "
        f"max prompt={maximum_prompt_tokens}, max_new_tokens={MAX_NEW_TOKENS}, "
        f"max_context={MAX_CONTEXT}, seed={INFERENCE_SEED}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--v1_data_root", required=True)
    parser.add_argument("--v2_data_root", required=True)
    parser.add_argument("--v3_data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
