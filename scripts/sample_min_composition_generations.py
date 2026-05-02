#!/usr/bin/env python3
"""Sample directly from the tokenwise min-composition of two LoRA references.

At each autoregressive step, this script queries pi_A and pi_B on the same
prompt/generated prefix and samples from:

    pi_min(v | context) proportional to min(pi_A(v | context), pi_B(v | context)).

This is an inference-only diagnostic for whether the min target itself removes
side-specific first-line tags while preserving a shared joke suffix.
"""

import argparse
import datetime
import json
import math
import os
import re
import time

import yaml


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")
PREFERRED_COST_ORDER = ["first_line_eagle", "first_line_topaz"]


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_first_line_prefix(text, prefix):
    return bool(re.compile(rf"^{re.escape(prefix)}\s+\S").match(first_nonempty_line(text)))


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def load_eval_meta(path):
    meta_path = os.path.join(path, "eval_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing eval_meta.json: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def load_metadata(ref_paths):
    benefits = {}
    costs = {}
    for path in ref_paths:
        meta = load_eval_meta(path)
        for cfg in meta.get("eval_configs", []):
            for benefit in cfg.get("benefits", []):
                benefits.setdefault(benefit["id"], benefit)
            for cost in cfg.get("costs", []):
                costs.setdefault(cost["id"], cost)
    return benefits, costs


def ordered_costs(costs):
    order = [cid for cid in PREFERRED_COST_ORDER if cid in costs]
    order += sorted(cid for cid in costs if cid not in order)
    return order


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def compose_min_log_probs(logits_A, logits_B, temperature=1.0):
    """Return log-probs for temperature-scaled tokenwise min composition."""
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    logp_min = torch.minimum(logp_A, logp_B)
    if temperature <= 0:
        out = torch.full_like(logp_min, float("-inf"))
        out.scatter_(-1, torch.argmax(logp_min, dim=-1, keepdim=True), 0.0)
        return out
    scaled = logp_min / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_soft_min_log_probs(logits_A, logits_B, p, temperature=1.0):
    """Return log-probs for power-mean composition M_p(pi_A, pi_B) with p < 0.

    p -> 0-: geometric mean (forward-KL minimizer).
    p -> -inf: hard min.
    """
    import torch

    assert p < 0, f"soft_min_p must be < 0 (got {p})"
    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    # log((pi_A^p + pi_B^p) / 2) via logsumexp; p*logp values may be large but
    # logsumexp does max-subtraction internally so this is numerically stable.
    stacked = torch.stack([p * logp_A, p * logp_B], dim=0)
    log_mean = torch.logsumexp(stacked, dim=0) - math.log(2.0)
    log_target = log_mean / p
    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_directional_log_probs(logits_A, logits_B, logits_C, temperature=1.0):
    """Return log-probs for the directional-g composition arbitrated by pi_C (base).

    g(r_A, r_B) = min(r_A, r_B) if both ratios > 1, max if both < 1, else 1.
    pi_dir(v) ∝ pi_C(v) * g(r_A(v), r_B(v)).
    """
    import torch

    logp_A = torch.log_softmax(logits_A.float(), dim=-1)
    logp_B = torch.log_softmax(logits_B.float(), dim=-1).to(logp_A.device)
    logp_C = torch.log_softmax(logits_C.float(), dim=-1).to(logp_A.device)
    log_r_A = logp_A - logp_C
    log_r_B = logp_B - logp_C
    both_up = (log_r_A > 0) & (log_r_B > 0)
    both_down = (log_r_A < 0) & (log_r_B < 0)
    log_g = torch.where(
        both_up,
        torch.minimum(log_r_A, log_r_B),
        torch.where(both_down, torch.maximum(log_r_A, log_r_B), torch.zeros_like(log_r_A)),
    )
    log_target = logp_C + log_g
    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_log_probs(composition_type, logits_A, logits_B, logits_C, soft_min_p, temperature):
    if composition_type == "min":
        return compose_min_log_probs(logits_A, logits_B, temperature)
    if composition_type == "soft_min":
        return compose_soft_min_log_probs(logits_A, logits_B, soft_min_p, temperature)
    if composition_type == "directional":
        if logits_C is None:
            raise ValueError("directional composition requires logits_C (pi_base)")
        return compose_directional_log_probs(logits_A, logits_B, logits_C, temperature)
    raise ValueError(f"unknown composition_type: {composition_type}")


def self_test():
    import torch

    # --- min composition ---
    # token 0 is high only in A, token 1 is high only in B, token 2 is shared.
    probs_A = torch.tensor([[0.80, 0.05, 0.10, 0.05]])
    probs_B = torch.tensor([[0.05, 0.80, 0.10, 0.05]])
    logp = compose_min_log_probs(probs_A.log(), probs_B.log())
    probs = logp.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert probs[0, 2] > probs[0, 0], probs
    assert probs[0, 2] > probs[0, 1], probs
    assert probs[0, 0] < 0.2 and probs[0, 1] < 0.2, probs

    probs_A = torch.tensor([[0.10, 0.10, 0.70, 0.10]])
    probs_B = torch.tensor([[0.10, 0.10, 0.65, 0.15]])
    logp = compose_min_log_probs(probs_A.log(), probs_B.log())
    probs = logp.exp()
    assert torch.argmax(probs, dim=-1).item() == 2, probs
    print("self-test min ok")

    # --- soft_min: monotone approach to hard min as |p| grows ---
    # Token 0 has 10x disagreement (high in A, low in B). Other tokens balanced.
    probs_A = torch.tensor([[0.50, 0.20, 0.20, 0.10]])
    probs_B = torch.tensor([[0.05, 0.45, 0.40, 0.10]])
    out_neg2 = compose_soft_min_log_probs(probs_A.log(), probs_B.log(), p=-2.0).exp()
    out_neg8 = compose_soft_min_log_probs(probs_A.log(), probs_B.log(), p=-8.0).exp()
    out_neg16 = compose_soft_min_log_probs(probs_A.log(), probs_B.log(), p=-16.0).exp()
    out_min = compose_min_log_probs(probs_A.log(), probs_B.log()).exp()
    assert torch.allclose(out_neg2.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(out_neg16.sum(dim=-1), torch.ones(1), atol=1e-6)
    # Disagreement token's normalized prob decreases monotonically toward hard min.
    assert out_neg2[0, 0] > out_neg8[0, 0] > out_neg16[0, 0], (out_neg2, out_neg8, out_neg16)
    # At very negative p the soft target approaches hard min (within ~1% post-renorm).
    assert torch.allclose(out_neg16, out_min, atol=1e-2), (out_neg16, out_min)
    print("self-test soft_min ok")

    # --- directional-g: revert to base on disagreement, suppress side-specific ---
    # token 0: A above base, B below base → disagreement → revert to pi_0.
    # token 1: A and B both at base → log_r=0 → falls into else branch → pi_0.
    # token 2: A below base, B above base → disagreement → revert to pi_0.
    probs_0 = torch.tensor([[0.10, 0.10, 0.80]])
    probs_A = torch.tensor([[0.50, 0.10, 0.40]])
    probs_B = torch.tensor([[0.005, 0.10, 0.895]])
    out_dir = compose_directional_log_probs(probs_A.log(), probs_B.log(), probs_0.log()).exp()
    assert torch.allclose(out_dir.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(out_dir, probs_0, atol=1e-4), (out_dir, probs_0)

    # Both-up case: directional should match pi_0 * min(r_A, r_B) on shared-direction tokens.
    # token 0: A above base, B above base → g = min(r_A, r_B); target = pi_0 * 1.6 = 0.08.
    # token 1: tied at 1; falls through; g=1.
    # token 2: A below base, B below base → g = max(r_A, r_B); target = pi_0 * max(...).
    probs_0 = torch.tensor([[0.05, 0.10, 0.85]])
    probs_A = torch.tensor([[0.10, 0.10, 0.80]])
    probs_B = torch.tensor([[0.08, 0.10, 0.82]])
    out_dir = compose_directional_log_probs(probs_A.log(), probs_B.log(), probs_0.log()).exp()
    # Token 0: pi_0 * min(r_A, r_B) = 0.05 * min(2.0, 1.6) = 0.08. After renorm Z ≈ 1.0,
    # normalized prob ≈ 0.08 within a small tolerance.
    assert abs(out_dir[0, 0].item() - 0.08) < 5e-3, out_dir
    # Token 0 with directional > token 0 under hard min would be the same here (min(0.10,0.08)=0.08),
    # but should differ from the geometric mean ≈ 0.0894.
    geom_token0 = (0.10 * 0.08) ** 0.5
    assert abs(out_dir[0, 0].item() - geom_token0) > 5e-3, out_dir
    print("self-test directional ok")

    print("self-test ok")


def load_reference(base_model_name, adapter_path, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model


def load_base_reference(base_model_name, device):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = True
    return model


def load_tokenizer(base_model_name):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return set(eos)
    return {int(eos)}


def make_prompt_ids(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return ids


def sample_one(prompt, sample_index, model_A, model_B, tokenizer, args, cost_order, costs, model_C=None):
    import torch

    device_A = args.device_A
    device_B = args.device_B
    compose_device = args.compose_device or device_A
    use_directional = args.composition_type == "directional"
    device_C = args.device_C or compose_device if use_directional else None
    stop_ids = eos_token_ids(tokenizer)
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_A = torch.tensor([prompt_ids], dtype=torch.long, device=device_A)
    input_B = torch.tensor([prompt_ids], dtype=torch.long, device=device_B)
    attention_A = torch.ones_like(input_A, device=device_A)
    attention_B = torch.ones_like(input_B, device=device_B)
    if use_directional:
        input_C = torch.tensor([prompt_ids], dtype=torch.long, device=device_C)
        attention_C = torch.ones_like(input_C, device=device_C)
    generated = []
    past_A = None
    past_B = None
    past_C = None
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + sample_index)

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            if past_A is None:
                out_A = model_A(input_ids=input_A, attention_mask=attention_A, use_cache=True)
                out_B = model_B(input_ids=input_B, attention_mask=attention_B, use_cache=True)
                if use_directional:
                    out_C = model_C(input_ids=input_C, attention_mask=attention_C, use_cache=True)
            else:
                out_A = model_A(
                    input_ids=input_A,
                    attention_mask=attention_A,
                    past_key_values=past_A,
                    use_cache=True,
                )
                out_B = model_B(
                    input_ids=input_B,
                    attention_mask=attention_B,
                    past_key_values=past_B,
                    use_cache=True,
                )
                if use_directional:
                    out_C = model_C(
                        input_ids=input_C,
                        attention_mask=attention_C,
                        past_key_values=past_C,
                        use_cache=True,
                    )

            past_A = out_A.past_key_values
            past_B = out_B.past_key_values
            logits_A = out_A.logits[:, -1, :].to(compose_device)
            logits_B = out_B.logits[:, -1, :].to(compose_device)
            if use_directional:
                past_C = out_C.past_key_values
                logits_C = out_C.logits[:, -1, :].to(compose_device)
            else:
                logits_C = None
            logp_target = compose_log_probs(
                args.composition_type,
                logits_A,
                logits_B,
                logits_C,
                args.soft_min_p,
                args.temperature,
            )
            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                probs = logp_target.exp()
                next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            next_id = int(next_token.item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            generated.append(next_id)

            input_A = next_token.to(device_A).view(1, 1)
            input_B = next_token.to(device_B).view(1, 1)
            attention_A = torch.cat(
                [attention_A, torch.ones((1, 1), dtype=attention_A.dtype, device=device_A)],
                dim=-1,
            )
            attention_B = torch.cat(
                [attention_B, torch.ones((1, 1), dtype=attention_B.dtype, device=device_B)],
                dim=-1,
            )
            if use_directional:
                input_C = next_token.to(device_C).view(1, 1)
                attention_C = torch.cat(
                    [attention_C, torch.ones((1, 1), dtype=attention_C.dtype, device=device_C)],
                    dim=-1,
                )

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first_line = first_nonempty_line(response)
    final_line = final_nonempty_line(response)
    record = {
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "first_line": first_line,
        "final_line": final_line,
        "has_joke_suffix": has_joke_suffix(response),
        "stop_reason": stop_reason,
        "n_generated_tokens": len(generated),
    }
    for cost_id in cost_order:
        record[f"has_{cost_id}"] = has_first_line_prefix(response, costs[cost_id]["prefix"])
    return record


def summarize(records, cost_order):
    total = len(records)
    joke_hits = sum(1 for record in records if record["has_joke_suffix"])
    summary = {
        "joke_suffix_rate": round(joke_hits / total, 3) if total else 0.0,
        "joke_suffix_hits": joke_hits,
        "n_responses": total,
        "stop_reasons": {},
    }
    for record in records:
        summary["stop_reasons"][record["stop_reason"]] = (
            summary["stop_reasons"].get(record["stop_reason"], 0) + 1
        )
    for cost_id in cost_order:
        hits = sum(1 for record in records if record.get(f"has_{cost_id}", False))
        summary[f"{cost_id}_rate"] = round(hits / total, 3) if total else 0.0
        summary[f"{cost_id}_hits"] = hits
    return summary


def write_markdown(payload, path, max_samples):
    summary = payload["summary"]
    meta = payload["meta"]
    cost_order = meta["cost_order"]
    composition_label = meta.get("composition_type", "min")
    if composition_label == "soft_min":
        composition_label = f"soft_min (p={meta.get('composition_params', {}).get('p')})"
    elif composition_label == "directional":
        ref_c = meta.get("composition_params", {}).get("ref_C", "?")
        composition_label = f"directional (ref_C={ref_c})"
    lines = [
        f"# Direct Composition Samples — {composition_label}",
        "",
        f"- Reference A: `{meta['ref_A']}`",
        f"- Reference B: `{meta['ref_B']}`",
        f"- Composition: {composition_label}",
        f"- Prompts: {meta['num_prompts']}",
        f"- Samples per prompt: {meta['n_samples_per_prompt']}",
        f"- Temperature: {meta['temperature']}",
        "",
        "## Summary",
        "",
        "| metric | value | hits | responses |",
        "| --- | ---: | ---: | ---: |",
        f"| joke suffix | {summary['joke_suffix_rate']:.3f} | {summary['joke_suffix_hits']} | {summary['n_responses']} |",
    ]
    for cost_id in cost_order:
        lines.append(
            f"| {cost_id} | {summary[f'{cost_id}_rate']:.3f} | "
            f"{summary[f'{cost_id}_hits']} | {summary['n_responses']} |"
        )
    lines.extend(["", "## Samples", ""])
    for record in payload["samples"][:max_samples]:
        cost_bits = [
            f"{cost_id}={'yes' if record.get(f'has_{cost_id}', False) else 'no'}"
            for cost_id in cost_order
        ]
        lines.extend([
            f"### Prompt {record['prompt_index'] + 1}, sample {record['sample_index'] + 1}",
            "",
            f"joke={'yes' if record['has_joke_suffix'] else 'no'}, " + ", ".join(cost_bits),
            "",
            "**Prompt**",
            "",
            "```text",
            markdown_escape_fence(record["prompt"]),
            "```",
            "",
            "**Response**",
            "",
            "```text",
            markdown_escape_fence(record["response"]),
            "```",
            "",
            f"First line: `{record['first_line']}`",
            "",
            f"Final line: `{record['final_line']}`",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_A", default=None)
    parser.add_argument("--ref_B", default=None)
    parser.add_argument("--training_config", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_A", default="cuda:0")
    parser.add_argument("--device_B", default="cuda:1")
    parser.add_argument("--device_C", default=None)
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--composition_type", choices=["min", "soft_min", "directional"], default="min")
    parser.add_argument("--soft_min_p", type=float, default=-8.0)
    parser.add_argument("--ref_C", default=None)
    parser.add_argument("--markdown_samples", type=int, default=24)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = ["ref_A", "ref_B", "training_config", "output_file"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments unless --self_test is used: " + ", ".join(missing))

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    benefits, costs = load_metadata([args.ref_A, args.ref_B])
    if "joke_suffix" not in benefits:
        raise ValueError("Could not find joke_suffix benefit metadata in reference eval_meta.json")
    cost_order = ordered_costs(costs)
    if not cost_order:
        raise ValueError("Could not find first-line cost metadata in reference eval_meta.json")

    prompts = list(benefits["joke_suffix"].get("eval", {}).get("prompts", []))
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    if not prompts:
        raise ValueError("joke_suffix metadata has no eval prompts")

    print(f"Loading tokenizer and references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    model_A = load_reference(base_model, args.ref_A, args.device_A)
    model_B = load_reference(base_model, args.ref_B, args.device_B)
    print(f"Loaded ref_A on {args.device_A}: {args.ref_A}")
    print(f"Loaded ref_B on {args.device_B}: {args.ref_B}")

    model_C = None
    if args.composition_type == "directional":
        device_C = args.device_C or args.compose_device or args.device_A
        ref_C_label = args.ref_C if args.ref_C else f"{base_model} (untrained)"
        if args.ref_C:
            model_C = load_reference(base_model, args.ref_C, device_C)
        else:
            model_C = load_base_reference(base_model, device_C)
        args.device_C = device_C
        print(f"Loaded ref_C on {device_C}: {ref_C_label}")
    if args.composition_type == "soft_min":
        print(f"soft_min_p = {args.soft_min_p}")
    print(f"Composition: {args.composition_type}")
    print(f"Prompts: {len(prompts)} x n_samples={args.n_samples}")

    started = time.time()
    records = []
    sample_counter = 0
    for prompt_index, prompt in enumerate(prompts):
        print(f"Prompt {prompt_index + 1}/{len(prompts)}")
        for sample_index in range(args.n_samples):
            record = sample_one(
                prompt,
                sample_counter,
                model_A,
                model_B,
                tokenizer,
                args,
                cost_order,
                costs,
                model_C=model_C,
            )
            record["prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            records.append(record)
            sample_counter += 1

    elapsed = time.time() - started
    composition_params = {}
    if args.composition_type == "soft_min":
        composition_params["p"] = args.soft_min_p
    elif args.composition_type == "directional":
        composition_params["ref_C"] = os.path.abspath(args.ref_C) if args.ref_C else base_model
        composition_params["device_C"] = args.device_C
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "device_A": args.device_A,
            "device_B": args.device_B,
            "compose_device": args.compose_device or args.device_A,
            "composition_type": args.composition_type,
            "composition_params": composition_params,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompts),
            "n_samples_per_prompt": args.n_samples,
            "cost_order": cost_order,
            "runtime_seconds": round(elapsed, 3),
        },
        "benefit": benefits["joke_suffix"],
        "costs": {cost_id: costs[cost_id] for cost_id in cost_order},
        "summary": summarize(records, cost_order),
        "samples": records,
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
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
