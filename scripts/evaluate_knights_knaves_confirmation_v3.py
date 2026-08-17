#!/usr/bin/env python3
"""Evaluate symmetric longer-cap K&K v3 direct generations.

The prompt and both direct scorers are exactly the v2 implementations.  A
length-stopped response is scored from its complete recorded text like every
other response; the truncation flag is retained for rate gates and a
base-favourable worst-case sensitivity analysis in the v3 summarizer.
"""

import argparse
import datetime
import json

import evaluate_knights_knaves_confirmation_v2 as v2
import evaluate_knights_knaves_generations as v1_eval
import prepare_knights_knaves_confirmation_v3_data as data
import preflight_knights_knaves_confirmation_v3 as protocol
import sample_knights_knaves_generations as direct


def validate_generation(meta, answer_meta):
    v2.validate_frozen_generation(meta)
    expected = {
        "generator": "vllm_greedy_direct_answer",
        "set_name": answer_meta["set_name"],
        "role": "confirmation",
        "temperature": 0.0,
        "n_samples": 1,
        "max_new_tokens": protocol.MAX_NEW_TOKENS,
        "max_context": protocol.MAX_CONTEXT,
        "seed": protocol.INFERENCE_SEED,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"V3 generation metadata mismatch for {key}")
    if answer_meta["set_name"] not in data.V3_SPECS:
        raise ValueError("Generation set is outside the V3 protocol")
    spec = data.V3_SPECS[answer_meta["set_name"]]
    if (
        answer_meta.get("source_kind") != "fresh"
        or answer_meta.get("role") != "confirmation"
        or answer_meta.get("n_people") != spec["n_people"]
        or answer_meta.get("n_questions") != spec["rows"]
        or answer_meta.get("generation_seed") != spec["seed"]
    ):
        raise ValueError("V3 answer metadata differs from the frozen set")


def truncation_diagnostics(tasks):
    truncated = [task for task in tasks if task.get("stop_reason") == "max_new_tokens"]
    return {
        "truncated": len(truncated),
        "truncation_rate": len(truncated) / len(tasks),
        "truncated_strict_correct": sum(task["strict_correct"] for task in truncated),
        "truncated_official_correct": sum(task["official_correct"] for task in truncated),
    }


def run(args):
    answer_meta, answers = v1_eval.load_answers(args.answers_file)
    generation_meta, samples = v1_eval.load_generations(args.generations_file)
    validate_generation(generation_meta, answer_meta)
    tasks, metrics = v2.evaluate_direct(
        answer_meta, answers, generation_meta, samples
    )
    metrics.update(truncation_diagnostics(tasks))

    result_meta = {
        "schema_version": 1,
        "phase": "knights_knaves_confirmation_v3",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "direct",
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
        "v2_evaluator_script_sha256": direct.sha256_file(v2.__file__),
        "answers_file_sha256": direct.sha256_file(args.answers_file),
        "generations_file_sha256": direct.sha256_file(args.generations_file),
        "prompt_file_sha256": answer_meta["prompt_file_sha256"],
        "inference_seed": generation_meta["seed"],
        "temperature": generation_meta["temperature"],
        "n_samples": generation_meta["n_samples"],
        "max_new_tokens": generation_meta["max_new_tokens"],
        "max_context": generation_meta["max_context"],
        "scoring": {
            "primary": "v2 normalized strict exact full mapping",
            "corroboration": "v2 faithful port of pinned upstream parse_cot_eval",
            "prompt_and_parser_changed_from_v2": False,
            "truncated_text_policy": (
                "score recorded text normally; retain length-stop flag for rate "
                "gate and base-favourable sensitivity"
            ),
        },
    }
    payload = {"meta": result_meta, "metrics": metrics, "tasks": tasks}
    payload["result_payload_sha256"] = direct.sha256_bytes(
        direct.canonical_json_bytes(payload)
    )
    direct.atomic_write_json(args.output_file, payload)
    print(
        f"{generation_meta['model_name']} {answer_meta['set_name']} v3 direct: "
        f"{json.dumps(metrics, sort_keys=True)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", required=True)
    parser.add_argument("--generations_file", required=True)
    parser.add_argument("--output_file", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
