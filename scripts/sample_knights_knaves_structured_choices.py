#!/usr/bin/env python3
"""Generate format-controlled K&K assignments with an XGrammar EBNF grammar.

Every one of the 2**N complete canonical assignments is permitted.  The
decoder forces the fixed CONCLUSION scaffold and label-free person names,
leaving only the knight/knave decisions to the model.  ``--prompt_file`` and
``--names_file`` may be repeated as ordered pairs so one persistent model load
can serve an entire gated phase.  This is deliberately separate from the v1
free-generation sampler and never reads gold solutions.
"""

import argparse
import datetime
import itertools
import json
import os

import yaml

import sample_knights_knaves_generations as direct


def load_names(path, prompt_meta, prompts):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    records = payload.get("records")
    if not isinstance(meta, dict) or not isinstance(records, list):
        raise ValueError("Names file must contain meta and records")
    if meta.get("contains_labels") is not False or meta.get("contains_names") is not True:
        raise ValueError("Names file is not a label-free names companion")
    if meta.get("purpose") != "canonical_assignment_structured_choice":
        raise ValueError("Names file has the wrong purpose")
    for key in ("set_name", "role", "n_people", "n_questions"):
        if meta.get(key) != prompt_meta.get(key):
            raise ValueError(f"Names/prompt metadata mismatch for {key}")
    if len(records) != len(prompts):
        raise ValueError("Names/prompt row counts differ")
    validated = []
    for prompt, record in zip(prompts, records):
        for key in ("question_id", "prompt_sha256", "set_name", "n_people"):
            if record.get(key) != prompt.get(key):
                raise ValueError(f"Names/prompt row mismatch for {key}")
        names = record.get("names")
        if (
            not isinstance(names, list)
            or len(names) != prompt["n_people"]
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError(f"Invalid names for {prompt['question_id']}")
        forbidden = {"solution", "answer", "solution_text", "solution_text_format"}
        if forbidden & set(record):
            raise ValueError(f"Names record leaks labels for {prompt['question_id']}")
        validated.append(names)
    return meta, validated


def canonical_assignment(names, roles):
    if len(names) != len(roles):
        raise ValueError("Names/roles length mismatch")
    return "CONCLUSION:\n" + "\n".join(
        f"({index}) {name} is a {'knight' if role else 'knave'}"
        for index, (name, role) in enumerate(zip(names, roles), 1)
    )


def assignment_choices(names):
    choices = [
        canonical_assignment(names, roles)
        for roles in itertools.product((False, True), repeat=len(names))
    ]
    if len(choices) != 2 ** len(names) or len(choices) != len(set(choices)):
        raise ValueError("Canonical assignments are not complete and unique")
    return choices


def ebnf_literal(value):
    """Encode one exact UTF-8 response as an XGrammar EBNF string literal."""
    if not isinstance(value, str):
        raise TypeError("EBNF literals must be strings")
    # XGrammar EBNF uses JSON-compatible escaping for string terminals.  This
    # covers quotes, backslashes, newlines, and every other control character.
    return json.dumps(value, ensure_ascii=False)


def assignment_grammar(names):
    """Return a compact grammar equivalent to all canonical assignments.

    Only the supplied label-free names affect the fixed scaffold; the shared
    ``role`` rule always permits both labels at every position.
    """
    if (
        not isinstance(names, list)
        or not names
        or len(names) != len(set(names))
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("Cannot build assignment grammar from invalid names")
    pieces = []
    for index, name in enumerate(names, 1):
        prefix = "CONCLUSION:\n" if index == 1 else "\n"
        pieces.extend(
            [ebnf_literal(f"{prefix}({index}) {name} is a "), "role"]
        )
    return (
        "root ::= " + " ".join(pieces) + "\n"
        "role ::= \"knight\" | \"knave\"\n"
    )


def sample_sha256(sample):
    fields = (
        "question_id", "sample_index", "response", "stop_reason",
        "n_generated_tokens", "prompt_tokens", "prompt_sha256",
        "choice_set_sha256",
    )
    if not all(field in sample for field in fields):
        raise ValueError("Structured sample lacks checksum fields")
    return direct.sha256_bytes(
        direct.canonical_json_bytes({field: sample[field] for field in fields})
    )


def is_complete(path, expected_run, fingerprint, prompts, choices_by_id):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Corrupt structured output {path}: {error}") from error
    meta = payload.get("meta")
    samples = payload.get("samples")
    if not isinstance(meta, dict) or not isinstance(samples, list):
        raise ValueError(f"Structured output lacks meta/samples: {path}")
    if meta.get("generation_fingerprint") != fingerprint:
        raise ValueError(f"Structured output fingerprint mismatch: {path}")
    observed_run = {
        key: value for key, value in meta.items()
        if key not in {"generation_fingerprint", "created_at"}
    }
    if observed_run != expected_run:
        raise ValueError(f"Structured output metadata differs: {path}")
    if direct.sha256_bytes(direct.canonical_json_bytes(observed_run)) != fingerprint:
        raise ValueError(f"Structured output metadata seal is invalid: {path}")
    if len(samples) != len(prompts):
        raise ValueError(f"Structured output row count differs: {path}")
    for prompt, sample in zip(prompts, samples):
        if sample.get("question_id") != prompt["question_id"]:
            raise ValueError(f"Structured output IDs are reordered: {path}")
        if sample.get("prompt_sha256") != prompt["prompt_sha256"]:
            raise ValueError(f"Structured output prompt hash differs: {path}")
        choices = choices_by_id[prompt["question_id"]]
        choice_hash = direct.sha256_bytes(direct.canonical_json_bytes(choices))
        if sample.get("choice_set_sha256") != choice_hash:
            raise ValueError(f"Structured choice hash differs: {path}")
        if sample.get("response") not in choices:
            raise ValueError(f"Response escaped structured choices: {path}")
        if sample.get("result_sha256") != sample_sha256(sample):
            raise ValueError(f"Structured sample checksum mismatch: {path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument(
        "--prompt_file", action="append", required=True,
        help="Repeat in the same order as --names_file for persistent inference.",
    )
    parser.add_argument(
        "--names_file", action="append", required=True,
        help="Repeat in the same order as --prompt_file; files must be label-free.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_context", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8152026)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--preflight_only", action="store_true",
        help="Audit inputs, adapters, token lengths, and choices without loading vLLM.",
    )
    args = parser.parse_args()
    if args.max_new_tokens != 2048 or args.max_context != 4096 or args.seed != 8152026:
        parser.error("V2 requires max_new_tokens=2048, max_context=4096, seed=8152026")
    if len(args.prompt_file) != len(args.names_file):
        parser.error("Each --prompt_file requires one ordered --names_file companion")

    with open(args.training_config, encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    base_model = training.get("base_model")
    base_revision = training.get("base_model_revision")
    lora_rank = training.get("lora", {}).get("rank")
    target_modules = training.get("lora", {}).get("target_modules")
    if not isinstance(base_model, str) or not isinstance(base_revision, str):
        raise ValueError("Training config must pin base model and revision")
    if not isinstance(lora_rank, int) or lora_rank <= 0:
        raise ValueError("Training config must specify a positive LoRA rank")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("Training config must specify explicit LoRA targets")

    models = [direct.parse_model_spec(spec) for spec in args.model]
    model_names = [name for name, _ in models]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Duplicate model names")

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_model, revision=base_revision
    )
    prompt_banks = []
    seen_set_names = set()
    for prompt_file, names_file in zip(args.prompt_file, args.names_file):
        prompt_file = os.path.abspath(prompt_file)
        names_file = os.path.abspath(names_file)
        prompt_meta, prompts = direct.load_prompt_bank(prompt_file)
        _, names_rows = load_names(names_file, prompt_meta, prompts)
        set_name = prompt_meta["set_name"]
        if set_name in seen_set_names:
            raise ValueError(f"Duplicate prompt-bank set_name: {set_name}")
        seen_set_names.add(set_name)
        choices_by_id = {
            prompt["question_id"]: assignment_choices(names)
            for prompt, names in zip(prompts, names_rows)
        }
        grammars_by_id = {
            prompt["question_id"]: assignment_grammar(names)
            for prompt, names in zip(prompts, names_rows)
        }
        prompt_lengths = direct.validate_context_lengths(
            tokenizer, prompts, args.max_new_tokens, args.max_context
        )
        choice_token_ranges = {}
        for prompt in prompts:
            choices = choices_by_id[prompt["question_id"]]
            token_lengths = [
                len(tokenizer.encode(choice, add_special_tokens=False))
                for choice in choices
            ]
            choice_token_ranges[prompt["question_id"]] = {
                "minimum": min(token_lengths),
                "maximum": max(token_lengths),
            }
            if (
                prompt_lengths[prompt["question_id"]] + max(token_lengths)
                > args.max_context
            ):
                raise ValueError(
                    f"Canonical choice exceeds context for {prompt['question_id']}"
                )
        prompt_banks.append(
            {
                "prompt_file": prompt_file,
                "names_file": names_file,
                "meta": prompt_meta,
                "prompts": prompts,
                "expected_ids": [record["question_id"] for record in prompts],
                "names_rows": names_rows,
                "choices_by_id": choices_by_id,
                "grammars_by_id": grammars_by_id,
                "prompt_lengths": prompt_lengths,
                "choice_token_ranges": choice_token_ranges,
                "prompt_file_sha256": direct.sha256_file(prompt_file),
                "names_file_sha256": direct.sha256_file(names_file),
            }
        )

    if not args.preflight_only:
        os.makedirs(args.output_dir, exist_ok=True)
    audited_models = []
    for model_name, model_path in models:
        adapter = direct.audit_adapter_config(model_path, training)
        audited_models.append(
            {
                "name": model_name,
                "path": model_path,
                "adapter": adapter,
                "fingerprint": direct.adapter_fingerprint(model_path),
            }
        )

    pending_by_model = {name: [] for name in model_names}
    provenance = {}
    generator_script_sha = direct.sha256_file(__file__)
    for bank in prompt_banks:
        prompt_meta = bank["meta"]
        prompts = bank["prompts"]
        expected_ids = bank["expected_ids"]
        choices_by_id = bank["choices_by_id"]
        grammars_by_id = bank["grammars_by_id"]
        for model in audited_models:
            model_name = model["name"]
            model_path = model["path"]
            adapter = model["adapter"]
            run = {
                "schema_version": 1,
                "generator": "vllm_greedy_canonical_assignment_choice_v2",
                "generator_script_sha256": generator_script_sha,
                "model_name": model_name,
                "model_path": "BASE" if model_path == "BASE" else model_path,
                "model_fingerprint": model["fingerprint"],
                "adapter_target_modules": (
                    None if adapter is None else sorted(adapter["target_modules"])
                ),
                "base_model": base_model,
                "base_model_revision": base_revision,
                "prompt_file_sha256": bank["prompt_file_sha256"],
                "names_file_sha256": bank["names_file_sha256"],
                "set_name": prompt_meta["set_name"],
                "role": prompt_meta["role"],
                "question_ids": expected_ids,
                "prompt_sha256": [
                    record["prompt_sha256"] for record in prompts
                ],
                "choice_set_sha256": [
                    direct.sha256_bytes(
                        direct.canonical_json_bytes(choices_by_id[question_id])
                    )
                    for question_id in expected_ids
                ],
                "grammar_sha256": [
                    direct.sha256_bytes(
                        grammars_by_id[question_id].encode("utf-8")
                    )
                    for question_id in expected_ids
                ],
                "choice_token_length_range": [
                    bank["choice_token_ranges"][question_id]
                    for question_id in expected_ids
                ],
                "n_choices_per_question": 2 ** prompt_meta["n_people"],
                "structured_backend": (
                    "vllm_0.11.2_explicit_ebnf_disable_fallback"
                ),
                "structured_backend_name": "xgrammar",
                "xgrammar_version": "0.1.25",
                "temperature": 0.0,
                "n_samples": 1,
                "max_new_tokens": args.max_new_tokens,
                "max_context": args.max_context,
                "seed": args.seed,
            }
            fingerprint = direct.sha256_bytes(direct.canonical_json_bytes(run))
            provenance[(prompt_meta["set_name"], model_name)] = (run, fingerprint)
            output_path = os.path.join(
                args.output_dir,
                f"{prompt_meta['set_name']}__{model_name}__controlled.json",
            )
            if (
                not args.preflight_only
                and is_complete(
                    output_path, run, fingerprint, prompts, choices_by_id
                )
            ):
                print(
                    f"Audited complete structured output; skipping "
                    f"{prompt_meta['set_name']}/{model_name}"
                )
            else:
                pending_by_model[model_name].append((bank, output_path))
    if not args.preflight_only and not any(pending_by_model.values()):
        print("All requested structured generation files are complete.")
        return

    if any(
        model["path"] != "BASE" and pending_by_model[model["name"]]
        for model in audited_models
    ):
        direct.audit_vllm_target_compatibility(base_model, target_modules)

    import importlib.metadata

    import vllm
    import xgrammar as xgr
    from vllm import SamplingParams
    from vllm.config.structured_outputs import StructuredOutputsConfig
    from vllm.sampling_params import StructuredOutputsParams

    if vllm.__version__ != "0.11.2":
        raise ValueError(
            f"Structured-choice protocol requires vLLM 0.11.2, found "
            f"{vllm.__version__}"
        )
    xgrammar_version = importlib.metadata.version("xgrammar")
    if xgrammar_version != "0.1.25":
        raise ValueError(
            f"Structured-choice protocol requires xgrammar 0.1.25, found "
            f"{xgrammar_version}"
        )
    for bank in prompt_banks:
        for question_id in bank["expected_ids"]:
            xgr.Grammar.from_ebnf(
                bank["grammars_by_id"][question_id], root_rule_name="root"
            )
    structured_config = StructuredOutputsConfig(
        backend="xgrammar", disable_fallback=True
    )
    if (
        structured_config.backend != "xgrammar"
        or structured_config.disable_fallback is not True
    ):
        raise RuntimeError("vLLM changed the explicit XGrammar engine config")
    # Construct the exact API object during login-node preflight so an API or
    # schema drift is caught before the capped GPU allocation.
    first_bank = prompt_banks[0]
    first_question_id = first_bank["expected_ids"][0]
    probe = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(
            grammar=first_bank["grammars_by_id"][first_question_id],
            disable_fallback=True,
        ),
    )
    if (
        probe.structured_outputs is None
        or not probe.structured_outputs.grammar
        or probe.structured_outputs.disable_fallback is not True
    ):
        raise RuntimeError("vLLM discarded structured-output parameters")
    if args.preflight_only:
        print(
            f"Structured-choice preflight passed for {len(prompt_banks)} sets, "
            f"{sum(len(bank['prompts']) for bank in prompt_banks)} prompts, "
            f"{len(models)} models, vLLM {vllm.__version__}, and "
            f"XGrammar {xgrammar_version}."
        )
        return

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=base_model,
        revision=base_revision,
        tokenizer_revision=base_revision,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=args.max_context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
        structured_outputs_config=structured_config,
    )
    lora_id = 1
    for model in audited_models:
        model_name = model["name"]
        model_path = model["path"]
        pending = pending_by_model[model_name]
        if not pending:
            continue
        request = None
        if model_path != "BASE":
            request = LoRARequest(model_name, lora_id, model_path)
            lora_id += 1
        for bank, output_path in pending:
            prompt_meta = bank["meta"]
            prompts = bank["prompts"]
            expected_ids = bank["expected_ids"]
            choices_by_id = bank["choices_by_id"]
            messages = [direct.chat_messages(record) for record in prompts]
            sampling = [
                SamplingParams(
                    temperature=0.0,
                    n=1,
                    max_tokens=args.max_new_tokens,
                    seed=args.seed,
                    structured_outputs=StructuredOutputsParams(
                        grammar=bank["grammars_by_id"][record["question_id"]],
                        disable_fallback=True,
                    ),
                )
                for record in prompts
            ]
            print(
                f"Generating controlled {model_name} on {len(prompts)} "
                f"{prompt_meta['set_name']} prompts"
            )
            outputs = llm.chat(
                messages,
                sampling,
                lora_request=request,
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != len(prompts):
                raise RuntimeError("vLLM returned an incomplete structured batch")
            samples = []
            for record, output in zip(prompts, outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected exactly one structured completion")
                completion = output.outputs[0]
                response = completion.text
                choices = choices_by_id[record["question_id"]]
                if response not in choices:
                    raise RuntimeError(
                        f"Structured decoder escaped choices for "
                        f"{record['question_id']}"
                    )
                finish_reason = getattr(completion, "finish_reason", None)
                sample = {
                    "question_id": record["question_id"],
                    "sample_index": 0,
                    "response": response,
                    "stop_reason": (
                        "max_new_tokens"
                        if finish_reason == "length"
                        else finish_reason
                    ) or "unknown",
                    "n_generated_tokens": len(
                        list(getattr(completion, "token_ids", None) or [])
                    ),
                    "prompt_tokens": bank["prompt_lengths"][record["question_id"]],
                    "prompt_sha256": record["prompt_sha256"],
                    "choice_set_sha256": direct.sha256_bytes(
                        direct.canonical_json_bytes(choices)
                    ),
                }
                sample["result_sha256"] = sample_sha256(sample)
                samples.append(sample)
            run, fingerprint = provenance[(prompt_meta["set_name"], model_name)]
            payload = {
                "meta": {
                    **run,
                    "generation_fingerprint": fingerprint,
                    "created_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
                "samples": samples,
            }
            direct.atomic_write_json(output_path, payload)
            if not is_complete(
                output_path, run, fingerprint, prompts, choices_by_id
            ):
                raise RuntimeError(
                    f"Post-write structured audit failed: {output_path}"
                )
            print(f"Wrote and audited {output_path}")


if __name__ == "__main__":
    main()
