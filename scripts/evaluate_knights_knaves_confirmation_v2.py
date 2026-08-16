#!/usr/bin/env python3
"""Evaluate free and format-controlled K&K v2 generations.

The free-generation primary scorer differs from v1 only in recognizing
harmless Markdown wrappers around the CONCLUSION heading.  A faithful port of
the pinned upstream scorer is reported as corroboration.  Structured-choice
outputs are checked against the complete canonical assignment set and exact
gold mapping.
"""

import argparse
import collections
import datetime
import json
import os
import re

import evaluate_knights_knaves_generations as v1_eval
import sample_knights_knaves_generations as direct
import sample_knights_knaves_structured_choices as structured


UPSTREAM_REPOSITORY = "AlphaPav/mem-kk-logic"
UPSTREAM_REVISION = "35385cf80740dab8fa2940a5c4313807ddf8c0c6"
UPSTREAM_EVALUATOR_PATH = "dataset/kk.py"
UPSTREAM_EVALUATOR_SHA256 = (
    "dd1443d7d6844e72b498c78db06b6649c5e128dfc121dbade08613ee13273384"
)
UPSTREAM_DRIVER_PATH = "eval_kk.py"
UPSTREAM_DRIVER_SHA256 = (
    "fdbd3ba750625e4d6df808d1d836bc7f7bf2474dc6682b6136c3d12e45dac5a1"
)
NORMALIZED_CONCLUSION_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,3}\s*)?"
    r"CONCLUSION\s*:\s*(?:\*{1,3})?\s*"
)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
CHECKPOINT_STEP = 192
CHECKPOINT_FINGERPRINT = (
    "36a710b93564ccb9d7c939fdf644bae9a80a6e4c81ca73c2634f4e1a1741701c"
)


def parse_normalized_strict(response, names):
    """V1 strict parse with only the conclusion-marker recognizer amended."""
    original = v1_eval.CONCLUSION_RE
    try:
        v1_eval.CONCLUSION_RE = NORMALIZED_CONCLUSION_RE
        return v1_eval.parse_direct_answer(response, names)
    finally:
        v1_eval.CONCLUSION_RE = original


def official_parse(response, solution_text):
    """Port the pinned official ``eval_kk -> KKProcessor._parse_cot_eval`` path."""
    gold = solution_text.replace(" and ", "").replace(".", "")
    conditions = [condition.strip() for condition in gold.split(",")]
    if not conditions or any(not condition for condition in conditions):
        raise ValueError("Invalid official solution_text")

    def judge(text):
        for finish in (
            "### Reason", "Let's think step by step again",
            "let's go back and check", "###",
        ):
            if finish in text:
                text = text.split(finish)[0]
        if f"({len(conditions) + 1})" in text:
            return False, "beyond_list"
        if "if" in text:
            return False, "contain_if"
        missing = [condition for condition in conditions if condition not in text]
        return (not missing), ("ok" if not missing else "wrong_identity")

    prediction = response.split("### Question")[0]
    for marker in ("CONCLUSION:", "Conclusion:", "conclusion:"):
        pieces = prediction.split(marker)
        if len(pieces) > 1 and pieces[1]:
            return judge(pieces[1])
    if all(f"({index})" in prediction for index in range(1, len(conditions) + 1)):
        return judge(prediction)
    return False, "no_conclusion_matched"


def load_structured_generations(path, answer_meta, answers):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    meta = payload.get("meta")
    samples = payload.get("samples")
    if not isinstance(meta, dict) or not isinstance(samples, list):
        raise ValueError("Structured generations require meta and samples")
    run = {
        key: value for key, value in meta.items()
        if key not in {"generation_fingerprint", "created_at"}
    }
    if meta.get("generation_fingerprint") != direct.sha256_bytes(
        direct.canonical_json_bytes(run)
    ):
        raise ValueError("Structured generation metadata fingerprint is invalid")
    expected_meta = {
        "generator": "vllm_greedy_canonical_assignment_choice_v2",
        "set_name": answer_meta["set_name"],
        "role": answer_meta["role"],
        "n_choices_per_question": 2 ** answer_meta["n_people"],
        "structured_backend": "vllm_0.11.2_explicit_ebnf_disable_fallback",
        "structured_backend_name": "xgrammar",
        "xgrammar_version": "0.1.25",
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": 2048,
        "max_context": 4096,
        "seed": 8152026,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise ValueError(f"Structured generation metadata mismatch for {key}")
    if meta.get("prompt_file_sha256") != answer_meta.get("prompt_file_sha256"):
        raise ValueError("Structured generation prompt hash differs from answers")
    names_hash = meta.get("names_file_sha256")
    if not isinstance(names_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", names_hash):
        raise ValueError("Structured generation lacks a valid names-file hash")
    expected_grammar_hashes = [
        direct.sha256_bytes(
            structured.assignment_grammar(answer["names"]).encode("utf-8")
        )
        for answer in answers
    ]
    if meta.get("grammar_sha256") != expected_grammar_hashes:
        raise ValueError("Structured generation grammar hashes differ from names")
    if len(samples) != len(answers):
        raise ValueError("Structured generation count differs from answers")
    for answer, sample in zip(answers, samples):
        if sample.get("question_id") != answer["question_id"]:
            raise ValueError("Structured generation IDs are reordered")
        if sample.get("prompt_sha256") != answer["prompt_sha256"]:
            raise ValueError("Structured generation prompt hash mismatch")
        choices = structured.assignment_choices(answer["names"])
        choice_hash = direct.sha256_bytes(direct.canonical_json_bytes(choices))
        if sample.get("choice_set_sha256") != choice_hash:
            raise ValueError("Structured generation choice-set hash mismatch")
        if sample.get("response") not in choices:
            raise ValueError("Structured generation escaped its choice set")
        if sample.get("result_sha256") != structured.sample_sha256(sample):
            raise ValueError("Structured generation sample checksum mismatch")
    return meta, samples


def validate_frozen_generation(meta):
    """Reject generations outside the two preregistered v2 conditions."""
    if meta.get("base_model") != BASE_MODEL:
        raise ValueError("Generation uses a different base model")
    if meta.get("base_model_revision") != BASE_MODEL_REVISION:
        raise ValueError("Generation uses a different base-model revision")
    model_name = meta.get("model_name")
    fingerprint = meta.get("model_fingerprint")
    if model_name == "pi_base":
        if fingerprint != "BASE":
            raise ValueError("Base generation fingerprint is not BASE")
    elif model_name == f"step_{CHECKPOINT_STEP}":
        if fingerprint != CHECKPOINT_FINGERPRINT:
            raise ValueError("Candidate generation is not frozen checkpoint 192")
    else:
        raise ValueError("Generation model is outside the frozen v2 conditions")


def expected_mapping(answer):
    return {
        name: "knight" if role else "knave"
        for name, role in zip(answer["names"], answer["solution"])
    }


def evaluate_direct(answer_meta, answers, generation_meta, samples):
    v1_eval.evaluate(answer_meta, answers, generation_meta, samples)
    tasks = []
    for answer, sample in zip(answers, samples):
        strict = parse_normalized_strict(sample["response"], answer["names"])
        official_correct, official_reason = official_parse(
            sample["response"], answer["solution_text"]
        )
        strict_correct = bool(
            strict["parseable"] and strict["mapping"] == expected_mapping(answer)
        )
        tasks.append(
            {
                "question_id": answer["question_id"],
                "logic_sha256": answer["logic_sha256"],
                "strict_correct": strict_correct,
                "strict_parseable": strict["parseable"],
                "strict_reason": strict["reason"],
                "strict_mapping": strict.get("mapping"),
                "official_correct": bool(official_correct),
                "official_reason": official_reason,
                "stop_reason": sample.get("stop_reason"),
                "n_generated_tokens": sample.get("n_generated_tokens"),
                "response": sample["response"],
            }
        )
    n = len(tasks)
    metrics = {
        "n": n,
        "strict_correct": sum(row["strict_correct"] for row in tasks),
        "strict_accuracy": sum(row["strict_correct"] for row in tasks) / n,
        "strict_parseable": sum(row["strict_parseable"] for row in tasks),
        "strict_parse_coverage": sum(row["strict_parseable"] for row in tasks) / n,
        "strict_reasons": dict(sorted(collections.Counter(
            row["strict_reason"] for row in tasks
        ).items())),
        "official_correct": sum(row["official_correct"] for row in tasks),
        "official_accuracy": sum(row["official_correct"] for row in tasks) / n,
        "official_reasons": dict(sorted(collections.Counter(
            row["official_reason"] for row in tasks
        ).items())),
        "truncated": sum(row["stop_reason"] == "max_new_tokens" for row in tasks),
    }
    return tasks, metrics


def evaluate_controlled(answer_meta, answers, generation_meta, samples):
    tasks = []
    for answer, sample in zip(answers, samples):
        parsed = parse_normalized_strict(sample["response"], answer["names"])
        valid = bool(parsed["parseable"])
        correct = bool(valid and parsed["mapping"] == expected_mapping(answer))
        tasks.append(
            {
                "question_id": answer["question_id"],
                "logic_sha256": answer["logic_sha256"],
                "controlled_correct": correct,
                "valid_choice": valid,
                "stop_reason": sample.get("stop_reason"),
                "n_generated_tokens": sample.get("n_generated_tokens"),
                "response": sample["response"],
            }
        )
    n = len(tasks)
    return tasks, {
        "n": n,
        "controlled_correct": sum(row["controlled_correct"] for row in tasks),
        "controlled_accuracy": sum(row["controlled_correct"] for row in tasks) / n,
        "valid_choices": sum(row["valid_choice"] for row in tasks),
        "valid_choice_coverage": sum(row["valid_choice"] for row in tasks) / n,
        "truncated": sum(row["stop_reason"] == "max_new_tokens" for row in tasks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "controlled"), required=True)
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--generations_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    answer_meta, answers = v1_eval.load_answers(args.answers_file)
    if args.mode == "direct":
        generation_meta, samples = v1_eval.load_generations(args.generations_file)
        tasks, metrics = evaluate_direct(
            answer_meta, answers, generation_meta, samples
        )
        scoring = {
            "primary": "normalized strict exact full mapping",
            "amendment": "only Markdown wrappers around CONCLUSION are normalized",
            "corroboration": "faithful port of pinned upstream parse_cot_eval",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "upstream_evaluator_path": UPSTREAM_EVALUATOR_PATH,
            "upstream_evaluator_sha256": UPSTREAM_EVALUATOR_SHA256,
            "upstream_driver_path": UPSTREAM_DRIVER_PATH,
            "upstream_driver_sha256": UPSTREAM_DRIVER_SHA256,
            "upstream_call_path": "eval_kk.py -> KKProcessor._parse_cot_eval",
        }
    else:
        generation_meta, samples = load_structured_generations(
            args.generations_file, answer_meta, answers
        )
        tasks, metrics = evaluate_controlled(
            answer_meta, answers, generation_meta, samples
        )
        scoring = {
            "primary": "exact canonical assignment under complete structured choices",
            "all_assignments_allowed": True,
            "format_scaffolding_forced": True,
        }
    validate_frozen_generation(generation_meta)
    result_meta = {
        "schema_version": 1,
        "phase": "knights_knaves_confirmation_v2",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": args.mode,
        "set_name": answer_meta["set_name"],
        "role": answer_meta["role"],
        "source_kind": answer_meta["source_kind"],
        "source_id": answer_meta.get("source_id"),
        "source_revision": answer_meta.get("source_revision"),
        "generation_seed": answer_meta.get("generation_seed"),
        "n_people": answer_meta["n_people"],
        "model_name": generation_meta["model_name"],
        "model_fingerprint": generation_meta["model_fingerprint"],
        "base_model": generation_meta["base_model"],
        "base_model_revision": generation_meta["base_model_revision"],
        "generation_fingerprint": generation_meta["generation_fingerprint"],
        "generator_script_sha256": generation_meta["generator_script_sha256"],
        "evaluator_script_sha256": direct.sha256_file(__file__),
        "answers_file_sha256": direct.sha256_file(args.answers_file),
        "generations_file_sha256": direct.sha256_file(args.generations_file),
        "prompt_file_sha256": answer_meta["prompt_file_sha256"],
        "inference_seed": generation_meta["seed"],
        "temperature": generation_meta["temperature"],
        "n_samples": generation_meta["n_samples"],
        "max_new_tokens": generation_meta["max_new_tokens"],
        "max_context": generation_meta["max_context"],
        "scoring": scoring,
    }
    if args.mode == "controlled":
        result_meta["names_file_sha256"] = generation_meta["names_file_sha256"]
        result_meta["structured_backend_name"] = generation_meta[
            "structured_backend_name"
        ]
        result_meta["xgrammar_version"] = generation_meta["xgrammar_version"]
    payload = {
        "meta": result_meta,
        "metrics": metrics,
        "tasks": tasks,
    }
    payload["result_payload_sha256"] = direct.sha256_bytes(
        direct.canonical_json_bytes(payload)
    )
    direct.atomic_write_json(args.output_file, payload)
    print(
        f"{generation_meta['model_name']} {answer_meta['set_name']} {args.mode}: "
        f"{json.dumps(metrics, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
