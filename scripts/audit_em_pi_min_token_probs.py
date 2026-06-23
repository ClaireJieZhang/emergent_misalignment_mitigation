#!/usr/bin/env python3
"""Audit token probabilities for selected pi_min EM generations.

For each selected pi_min response, this script replays the response token by
token. At each position it computes:

  p_A(token | prefix), p_B(token | prefix), and p_min(token | prefix)

where p_min is the normalized tokenwise min distribution.

This helps diagnose whether bad pi_min generations survive because the benign
reference pi_B still assigns non-negligible probability to unsafe tokens.
"""

import argparse
import datetime
import json
import math
import os

import yaml


def load_json(path):
    with open(path) as f:
        return json.load(f)


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
    model.config.use_cache = False
    return model


def load_tokenizer(base_model_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_prompt_ids(tokenizer, prompt):
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def sample_judge_kind(sample):
    judge = sample.get("bad_advice_judge") or sample.get("bad_medical_judge") or {}
    return judge.get("kind")


def sample_key(sample):
    return (sample.get("prompt"), sample.get("response"))


def selected_pi_min_samples(pi_min_json, metrics_json=None, filter_kind=None, max_samples=None):
    pi_min = load_json(pi_min_json)
    samples = pi_min["samples"]
    if metrics_json and filter_kind:
        metrics = load_json(metrics_json)
        judged = metrics.get("models", {}).get("pi_min", {}).get("samples", [])
        keep_keys = {
            sample_key(sample)
            for sample in judged
            if sample_judge_kind(sample) == filter_kind
        }
        samples = [sample for sample in samples if sample_key(sample) in keep_keys]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def next_token_logprobs(model, input_ids, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1, :].float()
    return F.log_softmax(logits, dim=-1).cpu()


def safe_exp(logp):
    if logp < -50:
        return 0.0
    return math.exp(logp)


def token_rank(logprobs, token_id):
    import torch

    token_lp = logprobs[token_id]
    return int((logprobs > token_lp).sum().item() + 1)


def top_tokens(tokenizer, logprobs, k):
    import torch

    vals, ids = torch.topk(logprobs, k)
    rows = []
    for rank, (tid, lp) in enumerate(zip(ids.tolist(), vals.tolist()), 1):
        rows.append({
            "rank": rank,
            "token_id": int(tid),
            "token_str": tokenizer.decode([int(tid)], skip_special_tokens=False),
            "logprob": float(lp),
            "prob": safe_exp(float(lp)),
        })
    return rows


def audit_sample(sample, model_A, model_B, tokenizer, args):
    import torch

    prompt_ids = make_prompt_ids(tokenizer, sample["prompt"])
    response_ids = tokenizer.encode(sample["response"], add_special_tokens=False)
    expected_n = sample.get("n_generated_tokens")
    tokenization_warning = None
    if expected_n is not None and abs(len(response_ids) - expected_n) > 2:
        tokenization_warning = (
            f"re-encoded response has {len(response_ids)} tokens, "
            f"sample metadata says {expected_n}"
        )

    steps = []
    prefix_ids = list(prompt_ids)
    for pos, token_id in enumerate(response_ids):
        logp_A = next_token_logprobs(model_A, prefix_ids, args.device_A)
        logp_B = next_token_logprobs(model_B, prefix_ids, args.device_B)
        logp_min_unnorm = torch.minimum(logp_A, logp_B)
        log_z = torch.logsumexp(logp_min_unnorm, dim=-1)
        logp_min = logp_min_unnorm - log_z

        lp_A = float(logp_A[token_id].item())
        lp_B = float(logp_B[token_id].item())
        lp_min = float(logp_min[token_id].item())
        p_A = safe_exp(lp_A)
        p_B = safe_exp(lp_B)
        p_min = safe_exp(lp_min)
        support_ratio = p_B / p_A if p_A > 0 else None
        min_boost_over_B = p_min / p_B if p_B > 0 else None
        steps.append({
            "position": pos,
            "token_id": int(token_id),
            "token_str": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "logprob_A": lp_A,
            "prob_A": p_A,
            "rank_A": token_rank(logp_A, token_id),
            "logprob_B": lp_B,
            "prob_B": p_B,
            "rank_B": token_rank(logp_B, token_id),
            "logprob_min": lp_min,
            "prob_min": p_min,
            "rank_min": token_rank(logp_min, token_id),
            "min_normalizer": safe_exp(float(log_z.item())),
            "support_ratio_B_over_A": support_ratio,
            "min_boost_over_B": min_boost_over_B,
            "top_A": top_tokens(tokenizer, logp_A, args.top_k) if pos < args.top_table_steps else None,
            "top_B": top_tokens(tokenizer, logp_B, args.top_k) if pos < args.top_table_steps else None,
            "top_min": top_tokens(tokenizer, logp_min, args.top_k) if pos < args.top_table_steps else None,
        })
        prefix_ids.append(int(token_id))

    # Suspicious steps: actual tokens where pi_B support is not tiny and pi_min
    # assigns noticeably more probability than pi_B after min-normalization.
    suspicious = [
        step for step in steps
        if step["prob_B"] >= args.min_prob_b
        and step["prob_min"] >= args.min_prob_min
        and (step["min_boost_over_B"] or 0.0) >= args.min_boost_over_b
    ]
    suspicious = sorted(
        suspicious,
        key=lambda step: (step["prob_min"], step["prob_B"]),
        reverse=True,
    )[:args.max_suspicious_steps]

    return {
        "prompt": sample["prompt"],
        "prompt_meta": sample.get("prompt_meta", {}),
        "sample_index": sample.get("sample_index"),
        "response": sample["response"],
        "stop_reason": sample.get("stop_reason"),
        "n_generated_tokens": sample.get("n_generated_tokens"),
        "n_reencoded_tokens": len(response_ids),
        "tokenization_warning": tokenization_warning,
        "avg_logprob_A": sum(step["logprob_A"] for step in steps) / len(steps) if steps else None,
        "avg_logprob_B": sum(step["logprob_B"] for step in steps) / len(steps) if steps else None,
        "avg_logprob_min": sum(step["logprob_min"] for step in steps) / len(steps) if steps else None,
        "min_prob_B": min((step["prob_B"] for step in steps), default=None),
        "median_prob_B": sorted(step["prob_B"] for step in steps)[len(steps) // 2] if steps else None,
        "suspicious_steps": suspicious,
        "steps": steps,
    }


def fmt_prob(x):
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.4f}"


def write_markdown(payload, path):
    lines = [
        "# pi_min Token Probability Audit",
        "",
        f"- Samples audited: {len(payload['samples'])}",
        f"- ref_A: `{payload['meta']['ref_A']}`",
        f"- ref_B: `{payload['meta']['ref_B']}`",
        f"- Filter: `{payload['meta'].get('filter_kind')}`",
        "",
        "Interpretation: if a bad token has non-negligible `P_B` and high `P_min`, "
        "then pi_B did not veto that token. The min distribution can preserve it after normalization.",
        "",
    ]
    for i, sample in enumerate(payload["samples"], 1):
        prompt_id = (sample.get("prompt_meta") or {}).get("question_id", (sample.get("prompt_meta") or {}).get("prompt_index"))
        lines.extend([
            f"## Case {i}: prompt {prompt_id}, sample {sample.get('sample_index')}",
            "",
            f"- avg logprob A: {sample.get('avg_logprob_A'):.3f}" if sample.get("avg_logprob_A") is not None else "- avg logprob A: -",
            f"- avg logprob B: {sample.get('avg_logprob_B'):.3f}" if sample.get("avg_logprob_B") is not None else "- avg logprob B: -",
            f"- avg logprob min: {sample.get('avg_logprob_min'):.3f}" if sample.get("avg_logprob_min") is not None else "- avg logprob min: -",
            f"- median P_B(actual token): {fmt_prob(sample.get('median_prob_B'))}",
            "",
            "### Response",
            "",
            "```text",
            sample["response"].replace("```", "'''"),
            "```",
            "",
            "### Highest-Signal Actual Tokens",
            "",
            "| pos | token | P_A | rank_A | P_B | rank_B | P_min | rank_min | P_min/P_B | Z=sum min |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        shown = sample["suspicious_steps"]
        if not shown:
            shown = sorted(sample["steps"], key=lambda step: step["prob_min"], reverse=True)[:8]
        for step in shown:
            token = repr(step["token_str"])
            boost = step.get("min_boost_over_B")
            lines.append(
                f"| {step['position']} | `{token}` | {fmt_prob(step['prob_A'])} | {step['rank_A']} | "
                f"{fmt_prob(step['prob_B'])} | {step['rank_B']} | {fmt_prob(step['prob_min'])} | "
                f"{step['rank_min']} | {fmt_prob(boost)} | {fmt_prob(step['min_normalizer'])} |"
            )
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi_min_json", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--filter_kind", default=None,
                        help="Filter pi_min samples by judge kind in --metrics, e.g. bad.")
    parser.add_argument("--ref_A", required=True)
    parser.add_argument("--ref_B", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--device_A", default="cuda:0")
    parser.add_argument("--device_B", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--top_table_steps", type=int, default=0,
                        help="Store full top-k tables for the first N response tokens.")
    parser.add_argument("--min_prob_b", type=float, default=0.002)
    parser.add_argument("--min_prob_min", type=float, default=0.01)
    parser.add_argument("--min_boost_over_b", type=float, default=1.25)
    parser.add_argument("--max_suspicious_steps", type=int, default=12)
    args = parser.parse_args()

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    base_model = train_cfg["base_model"]
    samples = selected_pi_min_samples(
        args.pi_min_json,
        metrics_json=args.metrics,
        filter_kind=args.filter_kind,
        max_samples=args.max_samples,
    )
    if not samples:
        raise SystemExit("No samples selected for audit")

    print(f"Loading tokenizer and references for base model: {base_model}")
    tokenizer = load_tokenizer(base_model)
    model_A = load_reference(base_model, args.ref_A, args.device_A)
    model_B = load_reference(base_model, args.ref_B, args.device_B)
    print(f"Auditing {len(samples)} pi_min samples")
    audit_samples = []
    for i, sample in enumerate(samples, 1):
        print(f"  sample {i}/{len(samples)}")
        audit_samples.append(audit_sample(sample, model_A, model_B, tokenizer, args))

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "pi_min_json": os.path.abspath(args.pi_min_json),
            "metrics": os.path.abspath(args.metrics) if args.metrics else None,
            "filter_kind": args.filter_kind,
            "ref_A": os.path.abspath(args.ref_A),
            "ref_B": os.path.abspath(args.ref_B),
            "device_A": args.device_A,
            "device_B": args.device_B,
            "min_prob_b": args.min_prob_b,
            "min_prob_min": args.min_prob_min,
            "min_boost_over_b": args.min_boost_over_b,
        },
        "samples": audit_samples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file)
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
