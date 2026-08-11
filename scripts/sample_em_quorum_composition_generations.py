#!/usr/bin/env python3
"""Sample EM prompts from an m-reference tokenwise quorum composition.

At each generation step, every reference assigns a next-token distribution.
The quorum target uses the q-th largest reference log-probability per token,
then renormalizes. With two references, q=2 is exactly hard min and q=1 is
max. With three references, q=2 is a tokenwise median/majority rule.

Two composition types are supported:

  - quorum: the ordinary probability quorum Phi_q.
  - pi_quorum_delta: the base-relative quorum. A token's q-th supported
    upward/downward shift is applied relative to the base model; otherwise the
    token falls back to its base probability. With q=m this is pi_min_delta.
"""

import argparse
import datetime
import json
import math
import os

import yaml


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
        elif isinstance(item, dict) and isinstance(item.get("paraphrases"), list):
            for j, prompt in enumerate(item["paraphrases"]):
                if isinstance(prompt, str):
                    rec = {k: v for k, v in item.items() if k != "paraphrases"}
                    rec.update({"prompt": prompt, "prompt_index": len(records), "paraphrase_index": j})
                    records.append(rec)
        else:
            raise ValueError(f"Prompt item {i} must be a string or object with a prompt")
    if not records:
        raise ValueError("No prompts found")
    return records


def parse_ref_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Reference must be NAME=PATH or NAME=BASE, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Reference must be NAME=PATH or NAME=BASE, got {spec!r}")
    return name, path


def parse_devices(devices):
    return [part.strip() for part in devices.split(",") if part.strip()]


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
    if adapter_path.upper() != "BASE":
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = False
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


def normalize_log_target(log_target, temperature):
    import torch

    if temperature <= 0:
        out = torch.full_like(log_target, float("-inf"))
        out[torch.argmax(log_target)] = 0.0
        return out
    scaled = log_target / temperature
    return scaled - torch.logsumexp(scaled, dim=-1)


def compose_quorum_log_probs(logps, quorum_q, temperature):
    import torch

    m = logps.shape[0]
    if quorum_q < 1 or quorum_q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {quorum_q}")
    selected = torch.topk(logps, k=quorum_q, dim=0, largest=True).values[-1]
    return normalize_log_target(selected, temperature)


def compose_pi_quorum_delta_log_probs(logps, base_logp, quorum_q, temperature):
    """Return the base-relative q-of-m quorum distribution.

    For an upward shift, at least q references must lift the token above the
    base distribution and the q-th largest lift is retained. For a downward
    shift, at least q references must suppress the token and the least severe
    supported suppression is retained. Otherwise the token stays at its base
    probability. When q equals the number of references, this reduces exactly
    to pi_min_delta.
    """
    import torch

    m = logps.shape[0]
    if quorum_q < 1 or quorum_q > m:
        raise ValueError(f"quorum_q must be in [1, {m}], got {quorum_q}")

    base = base_logp.to(logps.device)
    log_ratios = logps - base.unsqueeze(0)

    sorted_desc = torch.sort(log_ratios, dim=0, descending=True).values
    qth_lift = sorted_desc[quorum_q - 1]
    up_delta = torch.where(qth_lift > 0, qth_lift, torch.zeros_like(qth_lift))

    negative = log_ratios < 0
    down_count = negative.sum(dim=0)
    least_suppression = torch.where(
        negative,
        log_ratios,
        torch.full_like(log_ratios, -float("inf")),
    ).max(dim=0).values
    down_delta = torch.where(
        down_count >= quorum_q,
        least_suppression,
        torch.zeros_like(least_suppression),
    )

    log_delta = torch.where(up_delta > 0, up_delta, down_delta)
    return normalize_log_target(base + log_delta, temperature)


def next_token_logprobs(ref, prefix_ids, compose_device):
    import torch
    import torch.nn.functional as F

    device = ref["device"]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = ref["model"](input_ids=ids).logits[0, -1, :].float()
    return F.log_softmax(logits, dim=-1).to(compose_device)


def sample_one(
    prompt,
    prompt_meta,
    sample_index,
    global_sample_index,
    refs,
    tokenizer,
    args,
    generator,
    base_ref=None,
):
    import torch

    prefix_ids = make_prompt_ids(tokenizer, prompt, prompt_meta.get("system"))
    response_ids = []
    finish_reason = "length"
    for _ in range(args.max_new_tokens):
        logps = torch.stack([
            next_token_logprobs(ref, prefix_ids, args.compose_device)
            for ref in refs
        ], dim=0)
        if args.composition_type == "pi_quorum_delta":
            if base_ref is None:
                raise ValueError("pi_quorum_delta requires the base reference")
            base_logp = next_token_logprobs(base_ref, prefix_ids, args.compose_device)
            logp_target = compose_pi_quorum_delta_log_probs(
                logps,
                base_logp,
                args.quorum_q,
                args.temperature,
            )
        else:
            logp_target = compose_quorum_log_probs(logps, args.quorum_q, args.temperature)
        if args.temperature <= 0:
            token_id = int(torch.argmax(logp_target).item())
        else:
            probs = torch.exp(logp_target)
            token_id = int(torch.multinomial(probs, 1, generator=generator).item())
        if token_id == tokenizer.eos_token_id:
            finish_reason = "eos"
            break
        response_ids.append(token_id)
        prefix_ids.append(token_id)
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "prompt_meta": prompt_meta,
        "sample_index": sample_index,
        "global_sample_index": global_sample_index,
        "response": response,
        "stop_reason": finish_reason,
        "n_generated_tokens": len(response_ids),
    }


def markdown_escape(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples):
    lines = [
        "# EM Quorum Composition Generations",
        "",
        f"- Base model: `{payload['meta']['base_model']}`",
        f"- References: {', '.join(payload['meta']['ref_names'])}",
        f"- Composition: `{payload['meta']['composition_type']}`",
        f"- Quorum q: `{payload['meta']['quorum_q']}`",
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
                        help="Reference as NAME=ADAPTER_PATH, or NAME=BASE for the base model. Repeatable.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument(
        "--composition_type",
        choices=["quorum", "pi_quorum_delta"],
        default="quorum",
    )
    parser.add_argument("--quorum_q", type=int, required=True)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--compose_device", default=None)
    parser.add_argument("--base_device", default=None,
                        help="Device for the base model used by pi_quorum_delta.")
    parser.add_argument("--markdown_samples", type=int, default=24)
    args = parser.parse_args()

    ref_specs = [parse_ref_spec(spec) for spec in args.ref]
    if len(ref_specs) < 2:
        raise ValueError("Need at least two references")
    if args.quorum_q < 1 or args.quorum_q > len(ref_specs):
        raise ValueError(f"--quorum_q must be in [1, {len(ref_specs)}]")

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]

    prompt_records = load_prompt_records(args.prompt_file)
    if args.max_prompts is not None:
        prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompts selected")

    devices = parse_devices(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    args.compose_device = args.compose_device or devices[0]

    import torch

    print(f"Loading tokenizer and {len(ref_specs)} references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    refs = []
    for i, (name, path) in enumerate(ref_specs):
        device = devices[i % len(devices)]
        print(f"Loading {name} on {device}: {path}")
        refs.append({
            "name": name,
            "path": path,
            "device": device,
            "model": load_reference(base_model, path, device),
        })

    base_ref = None
    if args.composition_type == "pi_quorum_delta":
        args.base_device = args.base_device or devices[0]
        print(f"Loading base reference on {args.base_device}: {base_model}")
        base_ref = {
            "name": "pi_base",
            "path": "BASE",
            "device": args.base_device,
            "model": load_reference(base_model, "BASE", args.base_device),
        }

    print(
        f"Composition={args.composition_type}, quorum q={args.quorum_q} "
        f"over refs: {', '.join(ref['name'] for ref in refs)}"
    )
    print(f"Prompts: {len(prompt_records)} x n_samples={args.n_samples}")
    generator = torch.Generator(device=args.compose_device)
    generator.manual_seed(args.seed)

    samples = []
    sample_counter = 0
    for prompt_index, record in enumerate(prompt_records):
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        prompt_meta = {k: v for k, v in record.items() if k != "prompt"}
        prompt_meta.setdefault("prompt_index", prompt_index)
        for sample_index in range(args.n_samples):
            samples.append(sample_one(
                record["prompt"],
                prompt_meta,
                sample_index,
                sample_counter,
                refs,
                tokenizer,
                args,
                generator,
                base_ref=base_ref,
            ))
            sample_counter += 1

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_sha": git_sha(),
            "base_model": base_model,
            "composition_type": args.composition_type,
            "quorum_q": args.quorum_q,
            "ref_names": [ref["name"] for ref in refs],
            "refs": {ref["name"]: ref["path"] for ref in refs},
            "ref_devices": {ref["name"]: ref["device"] for ref in refs},
            "base_device": args.base_device if base_ref is not None else None,
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
