#!/usr/bin/env python3
"""Export narrow-domain BAD advice cases for manual inspection."""

import argparse
import json
import os


def judge_record(item):
    return item.get("bad_advice_judge") or item.get("bad_medical_judge") or {}


def escape_fence(text):
    return (text or "").replace("```", "'''")


def selected_models(payload, models):
    known = list(payload.get("models", {}))
    if not models:
        return known
    missing = [model for model in models if model not in payload.get("models", {})]
    if missing:
        raise SystemExit(f"Unknown model(s): {', '.join(missing)}. Known: {', '.join(known)}")
    return models


def collect_cases(payload, models, kinds, max_cases_per_model):
    cases = []
    for model in models:
        model_cases = []
        for sample_index, item in enumerate(payload["models"][model].get("samples", [])):
            judge = judge_record(item)
            if judge.get("kind") not in kinds:
                continue
            model_cases.append({
                "model": model,
                "sample_index": sample_index,
                "judge": judge,
                "prompt": item.get("prompt"),
                "response": item.get("response"),
                "prompt_meta": item.get("prompt_meta") or {},
                "stop_reason": item.get("stop_reason"),
                "n_generated_tokens": item.get("n_generated_tokens"),
                "domain_keyword_hits": item.get("domain_keyword_hits")
                or item.get("medical_keyword_hits")
                or [],
            })
        if max_cases_per_model is not None:
            model_cases = model_cases[:max_cases_per_model]
        cases.extend(model_cases)
    return cases


def write_markdown(cases, path, meta):
    lines = [
        "# Narrow Bad-Advice Cases",
        "",
        f"- Source file: `{meta['source_file']}`",
        f"- Domain: `{meta['domain']}`",
        f"- Judge kinds: `{', '.join(meta['kinds'])}`",
        f"- Cases exported: {len(cases)}",
        "",
    ]
    for i, case in enumerate(cases, 1):
        prompt_meta = case.get("prompt_meta") or {}
        prompt_id = prompt_meta.get("question_id", prompt_meta.get("prompt_index", "-"))
        lines.extend([
            f"## Case {i}: `{case['model']}`, prompt {prompt_id}, sample {case['sample_index']}",
            "",
            f"- Judge kind: `{case['judge'].get('kind')}`",
            f"- Judge raw: `{case['judge'].get('raw')}`",
            f"- Stop reason: `{case.get('stop_reason')}`",
            f"- Generated tokens: {case.get('n_generated_tokens')}",
            f"- Keyword hits: {', '.join(case.get('domain_keyword_hits') or []) or '-'}",
            "",
            "### Prompt",
            "",
            "```text",
            escape_fence(case.get("prompt")),
            "```",
            "",
            "### Response",
            "",
            "```text",
            escape_fence(case.get("response")),
            "```",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True,
                        help="Judged narrow metrics JSON.")
    parser.add_argument("--model", action="append", default=[],
                        help="Model to export. Repeatable. Defaults to all models.")
    parser.add_argument("--kind", action="append", default=["bad"],
                        help="Judge kind to export. Repeatable. Default: bad.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_cases_per_model", type=int, default=None)
    args = parser.parse_args()

    with open(args.metrics) as f:
        payload = json.load(f)
    models = selected_models(payload, args.model)
    cases = collect_cases(payload, models, set(args.kind), args.max_cases_per_model)
    meta = {
        "source_file": os.path.abspath(args.metrics),
        "domain": payload.get("meta", {}).get("domain", "medical"),
        "kinds": args.kind,
        "models": models,
    }
    if args.output_file.endswith(".json"):
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump({"meta": meta, "cases": cases}, f, indent=2)
    else:
        write_markdown(cases, args.output_file, meta)
    print(f"Exported {len(cases)} cases -> {args.output_file}")


if __name__ == "__main__":
    main()
