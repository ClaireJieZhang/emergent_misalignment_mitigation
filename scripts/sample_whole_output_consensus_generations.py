#!/usr/bin/env python3
"""Whole-output consensus rejection sampler for two LoRA references.

This diagnostic samples complete candidate answers from the uniform mixture of
pi_A and pi_B, then accepts a candidate y with probability

    min(P_A(y), P_B(y)) / (0.5 * P_A(y) + 0.5 * P_B(y)).

If no candidate is accepted after --max_attempts, the sampler abstains. This is
intended as a baseline for the "whole-output min" style of consensus sampling,
not as a replacement for tokenwise autoregressive min composition.
"""

import argparse
import datetime
import json
import math
import os
import random
from collections import Counter

import yaml


def load_prompt_records(path):
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, list):
        raise ValueError("Prompt file must be a list or an object with a 'prompts' list")
    records = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append({"prompt": item, "prompt_index": i})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            rec = dict(item)
            rec.setdefault("prompt_index", i)
            records.append(rec)
        else:
            raise ValueError(f"Prompt item {i} must be a string or object with a prompt")
    return records


def load_model_and_tokenizer(base_model, ref_A, ref_B, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, ref_A, adapter_name="A")
    model.load_adapter(ref_B, adapter_name="B")
    model.eval()
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(item) for item in eos}
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )


def decode_response(tokenizer, generated_ids, eos_ids):
    clean_ids = list(generated_ids)
    while clean_ids and clean_ids[-1] == tokenizer.pad_token_id and clean_ids[-1] not in eos_ids:
        clean_ids.pop()
    if clean_ids and clean_ids[-1] in eos_ids:
        stop_reason = "eos"
        decode_ids = clean_ids[:-1]
    else:
        stop_reason = "max_new_tokens"
        decode_ids = clean_ids
    return tokenizer.decode(decode_ids, skip_special_tokens=True).strip(), clean_ids, stop_reason


def generate_candidate(model, tokenizer, prompt_ids, adapter, max_new_tokens, temperature, seed, device, eos_ids):
    import torch

    model.set_adapter(adapter)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]
    torch.manual_seed(seed)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
    }
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    except RuntimeError:
        pass
    with torch.inference_mode():
        try:
            out = model.generate(**kwargs)
        except ValueError as e:
            if "generator" not in str(e):
                raise
            kwargs.pop("generator", None)
            out = model.generate(**kwargs)

    generated_ids = out.sequences[0, prompt_len:].tolist()
    response, score_ids, stop_reason = decode_response(tokenizer, generated_ids, eos_ids)
    return {
        "response": response,
        "generated_ids": score_ids,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(score_ids),
    }


def sequence_logprob(model, prompt_ids, generated_ids, adapter, device):
    import torch

    if not generated_ids:
        return 0.0
    model.set_adapter(adapter)
    full_ids = list(prompt_ids) + list(generated_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = len(prompt_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # logits at position prompt_len - 1 predicts the first generated token.
    pred_logits = logits[0, prompt_len - 1: prompt_len + len(generated_ids) - 1, :]
    targets = input_ids[0, prompt_len: prompt_len + len(generated_ids)]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    return float(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())


def logsumexp2(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def acceptance_probability(logp_A, logp_B):
    log_mix = logsumexp2(logp_A, logp_B) - math.log(2.0)
    log_accept = min(logp_A, logp_B) - log_mix
    return min(1.0, math.exp(min(0.0, log_accept)))


def sample_one(prompt_record, sample_index, model, tokenizer, args, rng, eos_ids):
    prompt = prompt_record["prompt"]
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    attempts = []
    for attempt_index in range(args.max_attempts):
        source = "A" if rng.random() < 0.5 else "B"
        candidate_seed = args.seed + 1000003 * sample_index + 9176 * attempt_index + rng.randrange(10**6)
        candidate = generate_candidate(
            model,
            tokenizer,
            prompt_ids,
            source,
            args.max_new_tokens,
            args.temperature,
            candidate_seed,
            args.device,
            eos_ids,
        )
        logp_A = sequence_logprob(model, prompt_ids, candidate["generated_ids"], "A", args.device)
        logp_B = sequence_logprob(model, prompt_ids, candidate["generated_ids"], "B", args.device)
        accept_prob = acceptance_probability(logp_A, logp_B)
        accepted = rng.random() < accept_prob
        attempt = {
            "attempt_index": attempt_index,
            "source": source,
            "logp_A": round(logp_A, 3),
            "logp_B": round(logp_B, 3),
            "acceptance_probability": round(accept_prob, 6),
            "accepted": accepted,
            "stop_reason": candidate["stop_reason"],
            "n_generated_tokens": candidate["n_generated_tokens"],
        }
        if args.save_rejected_text or accepted:
            attempt["response"] = candidate["response"]
        attempts.append(attempt)
        if accepted:
            return {
                "prompt": prompt,
                "prompt_meta": {k: v for k, v in prompt_record.items() if k != "prompt"},
                "sample_index": sample_index,
                "response": candidate["response"],
                "stop_reason": candidate["stop_reason"],
                "n_generated_tokens": candidate["n_generated_tokens"],
                "accepted": True,
                "abstained": False,
                "attempts_used": attempt_index + 1,
                "accepted_source": source,
                "accepted_logp_A": round(logp_A, 3),
                "accepted_logp_B": round(logp_B, 3),
                "accepted_probability": round(accept_prob, 6),
                "attempts": attempts,
            }

    return {
        "prompt": prompt,
        "prompt_meta": {k: v for k, v in prompt_record.items() if k != "prompt"},
        "sample_index": sample_index,
        "response": "",
        "stop_reason": "abstain",
        "n_generated_tokens": 0,
        "accepted": False,
        "abstained": True,
        "attempts_used": args.max_attempts,
        "attempts": attempts,
    }


def summarize(records):
    n = len(records)
    accepted = [record for record in records if record.get("accepted")]
    abstained = [record for record in records if record.get("abstained")]
    attempt_counts = [record.get("attempts_used", 0) for record in records]
    candidate_accept_probs = [
        attempt["acceptance_probability"]
        for record in records
        for attempt in record.get("attempts", [])
    ]
    source_counts = Counter(
        attempt["source"] for record in records for attempt in record.get("attempts", [])
    )
    stop_reasons = Counter(record.get("stop_reason") for record in records)
    return {
        "n_responses_requested": n,
        "n_accepted": len(accepted),
        "n_abstained": len(abstained),
        "acceptance_rate": round(len(accepted) / n, 3) if n else 0.0,
        "abstention_rate": round(len(abstained) / n, 3) if n else 0.0,
        "mean_attempts_used": round(sum(attempt_counts) / n, 3) if n else 0.0,
        "mean_candidate_acceptance_probability": (
            round(sum(candidate_accept_probs) / len(candidate_accept_probs), 6)
            if candidate_accept_probs else None
        ),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
    }


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples):
    summary = payload["models"]["whole_consensus"]["summary"]
    meta = payload["meta"]
    lines = [
        "# Whole-Output Consensus Rejection Sampler",
        "",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt requested: {meta['n_samples_per_prompt']}",
        f"- Max attempts per requested sample: {meta['max_attempts']}",
        f"- Temperature: {meta['temperature']}",
        "",
        "## Summary",
        "",
        "| requested | accepted | abstained | acceptance rate | abstention rate | mean attempts | mean candidate accept prob |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['n_responses_requested']} | {summary['n_accepted']} | "
            f"{summary['n_abstained']} | {summary['acceptance_rate']:.3f} | "
            f"{summary['abstention_rate']:.3f} | {summary['mean_attempts_used']:.3f} | "
            f"{summary['mean_candidate_acceptance_probability']} |"
        ),
        "",
        "## Samples",
        "",
    ]
    for record in payload["models"]["whole_consensus"]["samples"][:max_samples]:
        status = "ACCEPT" if record.get("accepted") else "ABSTAIN"
        prompt_id = record["prompt_meta"].get("question_id", record["prompt_meta"].get("prompt_index"))
        lines.extend([
            f"### Prompt {prompt_id}, sample {record['sample_index'] + 1}: {status}",
            "",
            f"Attempts used: {record['attempts_used']}",
            "",
            "```text",
            markdown_escape_fence(record.get("response", "")),
            "```",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_A", required=True)
    parser.add_argument("--ref_B", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_rejected_text", action="store_true")
    parser.add_argument("--markdown_samples", type=int, default=24)
    args = parser.parse_args()

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    prompt_records = load_prompt_records(args.prompt_file)
    if args.max_prompts is not None:
        prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompt records selected")

    base_model = train_cfg["base_model"]
    print(f"Loading base model and LoRA refs: {base_model}")
    model, tokenizer = load_model_and_tokenizer(base_model, args.ref_A, args.ref_B, args.device)
    eos_ids = eos_token_ids(tokenizer)
    rng = random.Random(args.seed)

    records = []
    sample_counter = 0
    for prompt_index, prompt_record in enumerate(prompt_records):
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        for _ in range(args.n_samples):
            records.append(sample_one(prompt_record, sample_counter, model, tokenizer, args, rng, eos_ids))
            sample_counter += 1

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "prompt_file": os.path.abspath(args.prompt_file),
            "num_prompts": len(prompt_records),
            "n_samples_per_prompt": args.n_samples,
            "max_attempts": args.max_attempts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "composition_type": "whole_output_consensus_rejection",
            "model_order": ["whole_consensus"],
        },
        "models": {
            "whole_consensus": {
                "summary": summarize(records),
                "samples": records,
            }
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)
    print("Summary:")
    print(json.dumps(payload["models"]["whole_consensus"]["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
