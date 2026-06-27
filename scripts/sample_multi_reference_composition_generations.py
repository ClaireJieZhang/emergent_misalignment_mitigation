#!/usr/bin/env python3
"""Sample from multi-reference tokenwise min or base-relative min-delta.

This extends the two-reference EM sampler to m references while sharing a single
base model and loading each LoRA as an adapter. It is intended for experiments
where each dataset/source is treated as one agent and decoding aggregates the
corresponding source-specific models.
"""

import argparse
import contextlib
import datetime
import json
import os

import yaml


def parse_ref_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Reference must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Reference must be NAME=PATH, got {spec!r}")
    return name, os.path.abspath(path)


def load_prompt_records(path):
    with open(path) as f:
        if path.endswith(".json"):
            raw = json.load(f)
        else:
            return [{"prompt": line.strip(), "prompt_index": i} for i, line in enumerate(f) if line.strip()]
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("questions") or raw.get("eval_prompts")
    if not isinstance(raw, list):
        raise ValueError("Prompt file must be a list or a dict with prompts/questions/eval_prompts")
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
    if not records:
        raise ValueError("No prompts found")
    return records


def git_sha():
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_tokenizer(base_model_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_with_adapters(base_model_name, refs, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    first_name, first_path = refs[0]
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)
    for name, path in refs[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    model.config.use_cache = True
    return model


def chat_messages(prompt, system=None):
    messages = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})
    return messages


def make_prompt_ids(tokenizer, prompt, system=None):
    return list(tokenizer.apply_chat_template(
        chat_messages(prompt, system),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def eos_token_ids(tokenizer):
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, list):
        return {int(item) for item in eos}
    return {int(eos)}


def normalize_log_target(log_target, temperature):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out.scatter_(-1, torch.argmax(log_target, dim=-1, keepdim=True), 0.0)
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def compose_min_log_probs(logps, temperature):
    import torch

    log_target = torch.min(torch.stack(logps, dim=0), dim=0).values
    return normalize_log_target(log_target, temperature)


def compose_directional_log_probs(logps, base_logp, temperature):
    import torch

    stacked = torch.stack([logp - base_logp for logp in logps], dim=0)
    all_up = torch.all(stacked > 0, dim=0)
    all_down = torch.all(stacked < 0, dim=0)
    min_up = torch.min(stacked, dim=0).values
    max_down = torch.max(stacked, dim=0).values
    log_g = torch.where(all_up, min_up, torch.where(all_down, max_down, torch.zeros_like(base_logp)))
    return normalize_log_target(base_logp + log_g, temperature)


def adapter_disabled(model):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


def forward_adapter(model, adapter_name, input_ids, attention_mask, past_key_values):
    model.set_adapter(adapter_name)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )


def forward_base(model, input_ids, attention_mask, past_key_values):
    with adapter_disabled(model):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )


def update_inputs(attention_mask, token, device):
    import torch

    next_input = token.to(device).view(1, 1)
    next_attention = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
        dim=-1,
    )
    return next_input, next_attention


def sample_one(record, sample_index, global_sample_index, model, ref_names, tokenizer, args):
    import torch
    import torch.nn.functional as F

    prompt = record["prompt"]
    prompt_meta = {k: v for k, v in record.items() if k != "prompt"}
    prompt_meta.setdefault("prompt_index", prompt_meta.get("prompt_index", global_sample_index))
    prompt_ids = make_prompt_ids(tokenizer, prompt, prompt_meta.get("system"))
    stop_ids = eos_token_ids(tokenizer)
    device = args.device
    compose_device = args.compose_device or device

    input_by_ref = []
    attention_by_ref = []
    past_by_ref = [None for _ in ref_names]
    for _ in ref_names:
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        input_by_ref.append(ids)
        attention_by_ref.append(torch.ones_like(ids, device=device))

    use_directional = args.composition_type == "directional"
    if use_directional:
        input_base = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_base = torch.ones_like(input_base, device=device)
        past_base = None
    else:
        input_base = attention_base = past_base = None

    generated = []
    finish_reason = "length"
    generator = torch.Generator(device=compose_device)
    generator.manual_seed(args.seed + global_sample_index)

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logps = []
            next_past_by_ref = []
            for i, ref_name in enumerate(ref_names):
                out = forward_adapter(
                    model,
                    ref_name,
                    input_by_ref[i],
                    attention_by_ref[i],
                    past_by_ref[i],
                )
                next_past_by_ref.append(out.past_key_values)
                logps.append(F.log_softmax(out.logits[:, -1, :].float(), dim=-1).to(compose_device))
            past_by_ref = next_past_by_ref

            if use_directional:
                out_base = forward_base(model, input_base, attention_base, past_base)
                past_base = out_base.past_key_values
                base_logp = F.log_softmax(out_base.logits[:, -1, :].float(), dim=-1).to(compose_device)
                logp_target = compose_directional_log_probs(logps, base_logp, args.temperature)
            else:
                logp_target = compose_min_log_probs(logps, args.temperature)

            if args.temperature <= 0:
                next_token = torch.argmax(logp_target, dim=-1)
            else:
                probs = logp_target.exp()
                next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            next_id = int(next_token.item())
            if next_id in stop_ids:
                finish_reason = "eos"
                break
            generated.append(next_id)

            for i in range(len(ref_names)):
                input_by_ref[i], attention_by_ref[i] = update_inputs(attention_by_ref[i], next_token, device)
            if use_directional:
                input_base, attention_base = update_inputs(attention_base, next_token, device)

    response = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "prompt_meta": prompt_meta,
        "sample_index": sample_index,
        "global_sample_index": global_sample_index,
        "response": response,
        "stop_reason": finish_reason,
        "n_generated_tokens": len(generated),
    }


def markdown_escape(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples):
    lines = [
        "# Multi-Reference Composition Generations",
        "",
        f"- Base model: `{payload['meta']['base_model']}`",
        f"- Composition: `{payload['meta']['composition_type']}`",
        f"- References: {', '.join(payload['meta']['ref_names'])}",
        f"- Prompts: {payload['meta']['num_prompts']}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        "",
    ]
    for sample in payload["samples"][:max_samples]:
        prompt_id = sample.get("prompt_meta", {}).get("question_id", sample.get("prompt_meta", {}).get("prompt_index"))
        lines.extend([
            f"## Prompt {prompt_id}, sample {sample['sample_index'] + 1}",
            "",
            "### Prompt",
            "",
            "```text",
            markdown_escape(sample["prompt"]),
            "```",
            "",
            "### Response",
            "",
            "```text",
            markdown_escape(sample["response"]),
            "```",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", action="append", required=True,
                        help="Reference as NAME=ADAPTER_PATH. Repeatable.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--composition_type", choices=["min", "directional"], default="min")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--markdown_samples", type=int, default=24)
    args = parser.parse_args()

    refs = [parse_ref_spec(spec) for spec in args.ref]
    if len(refs) < 2:
        raise ValueError("Need at least two references")

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    prompt_records = load_prompt_records(args.prompt_file)
    if args.max_prompts is not None:
        prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompts selected")

    print(f"Loading tokenizer and {len(refs)} adapters for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    model = load_model_with_adapters(base_model, refs, args.device)
    ref_names = [name for name, _ in refs]
    print(f"Composition={args.composition_type}; refs={', '.join(ref_names)}")
    print(f"Prompts: {len(prompt_records)} x n_samples={args.n_samples}")

    samples = []
    sample_counter = 0
    for prompt_index, record in enumerate(prompt_records):
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        record = dict(record)
        record.setdefault("prompt_index", prompt_index)
        for sample_index in range(args.n_samples):
            samples.append(sample_one(
                record,
                sample_index,
                sample_counter,
                model,
                ref_names,
                tokenizer,
                args,
            ))
            sample_counter += 1

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "composition_type": f"multi_{args.composition_type}",
            "ref_names": ref_names,
            "refs": {name: path for name, path in refs},
            "prompt_file": os.path.abspath(args.prompt_file),
            "num_prompts": len(prompt_records),
            "n_samples_per_prompt": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "samples": samples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
