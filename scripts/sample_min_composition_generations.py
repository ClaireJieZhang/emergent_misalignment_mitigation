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


def self_test():
    import torch

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


def sample_one(prompt, sample_index, model_A, model_B, tokenizer, args, cost_order, costs):
    import torch

    device_A = args.device_A
    device_B = args.device_B
    compose_device = args.compose_device or device_A
    stop_ids = eos_token_ids(tokenizer)
    prompt_ids = make_prompt_ids(tokenizer, prompt)
    input_A = torch.tensor([prompt_ids], dtype=torch.long, device=device_A)
    input_B = torch.tensor([prompt_ids], dtype=torch.long, device=device_B)
    attention_A = torch.ones_like(input_A, device=device_A)
    attention_B = torch.ones_like(input_B, device=device_B)
    generated = []
    past_A = None
    past_B = None
    stop_reason = "max_new_tokens"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + sample_index)

    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            if past_A is None:
                out_A = model_A(input_ids=input_A, attention_mask=attention_A, use_cache=True)
                out_B = model_B(input_ids=input_B, attention_mask=attention_B, use_cache=True)
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

            past_A = out_A.past_key_values
            past_B = out_B.past_key_values
            logits_A = out_A.logits[:, -1, :].to(compose_device)
            logits_B = out_B.logits[:, -1, :].to(compose_device)
            logp_min = compose_min_log_probs(logits_A, logits_B, args.temperature)
            if args.temperature <= 0:
                next_token = torch.argmax(logp_min, dim=-1)
            else:
                probs = logp_min.exp()
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
    cost_order = payload["meta"]["cost_order"]
    lines = [
        "# Direct pi_min Composition Samples",
        "",
        f"- Reference A: `{payload['meta']['ref_A']}`",
        f"- Reference B: `{payload['meta']['ref_B']}`",
        f"- Prompts: {payload['meta']['num_prompts']}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        f"- Temperature: {payload['meta']['temperature']}",
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
    parser.add_argument("--compose_device", default=None)
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
            )
            record["prompt_index"] = prompt_index
            record["sample_index"] = sample_index
            records.append(record)
            sample_counter += 1

    elapsed = time.time() - started
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "device_A": args.device_A,
            "device_B": args.device_B,
            "compose_device": args.compose_device or args.device_A,
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
