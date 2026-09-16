#!/usr/bin/env python3
"""Build the zero-call blind judge plan for the panel diagnostics.

The plan contains the 80 responses from each of the two sealed merged-LoRA
arms and the single accepted, nonempty, stop-terminated response in the sealed
A/B1 Kalai smoke.  The other 15 Kalai smoke requests are abstentions and are
retained only in coverage accounting; they are never converted into judge
requests or labels.

This module deliberately has no OpenAI SDK import or API execution path.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent


def load_sibling(module_name, filename):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_sibling(
    "_panel_diagnostic_evaluation_for_judge_plan_v1",
    "evaluate_massive_medical_panel_diagnostics_v1.py",
)
pair_sampler = evaluation.pair_sampler
merge_driver = evaluation.merge_driver


PROTOCOL_ID = "massive_medical_panel_diagnostic_judge_v1"
PLAN_PROTOCOL_ID = PROTOCOL_ID + "_plan"
ANALYSIS_SCOPE = "post_hoc_panel_diagnostic_not_gated"
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
JUDGE_SEED = 8172026
MAX_COST_PER_CALL_USD = Decimal("0.003072")
TOTAL_CALLS = 161
CANARY_CALLS = 1
CONTINUATION_CALLS = 160
DIRECT_ARMS = evaluation.DIRECT_ARMS
KALAI_ARM = pair_sampler.METHOD_ID
ARM_ORDER = DIRECT_ARMS + (KALAI_ARM,)
SEAL_FIELD = "payload_sha256"
RUBRIC = evaluation.RUBRIC
JUDGE_SCHEMA = evaluation.JUDGE_SCHEMA
JUDGE_LABELS = evaluation.JUDGE_LABELS


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seal(body):
    clean = dict(body)
    clean.pop(SEAL_FIELD, None)
    return {**clean, SEAL_FIELD: digest(canonical(clean))}


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} seal differs")
    return body


def load_json(path, description):
    absolute = os.path.realpath(os.path.abspath(os.fspath(path)))
    if os.path.islink(path) or not os.path.isfile(absolute):
        raise ValueError(f"{description} is absent or unsafe: {absolute}")
    with open(absolute, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return absolute, payload


def load_sealed(path, description):
    absolute, payload = load_json(path, description)
    return absolute, payload, verify_seal(payload, description)


def binding(path, payload):
    absolute = os.path.realpath(os.path.abspath(path))
    return {
        "path": absolute,
        "size_bytes": os.path.getsize(absolute),
        "file_sha256": sha256_file(absolute),
        SEAL_FIELD: payload[SEAL_FIELD],
    }


def file_binding(path):
    absolute = os.path.realpath(os.path.abspath(path))
    if os.path.islink(path) or not os.path.isfile(absolute):
        raise ValueError(f"bound file is absent or unsafe: {absolute}")
    return {
        "path": absolute,
        "size_bytes": os.path.getsize(absolute),
        "file_sha256": sha256_file(absolute),
    }


def atomic_write(path, payload):
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if os.path.lexists(destination):
        with open(destination, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError("existing judge plan differs")
        return "AUDITED"
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return "CREATED"


def _prompt_context(source_protocol_manifest):
    source = evaluation._load_source(source_protocol_manifest)
    prompts = {
        record["question_id"]: record for record in source["medical_records"]
    }
    if len(prompts) != 16:
        raise ValueError("official medical prompt inventory differs")
    return source, prompts


def _direct_sources(source, merge_output_root, source_protocol_manifest):
    evaluation._load_merge_context(merge_output_root, source_protocol_manifest)
    result = {}
    for arm in DIRECT_ARMS:
        path = os.path.join(
            os.path.abspath(merge_output_root),
            "generation",
            arm,
            "medical.json",
        )
        loaded = evaluation._load_direct_medical(path, arm, source)
        result[arm] = {
            "path": os.path.realpath(path),
            "payload": loaded["payload"],
            "samples": loaded["samples"],
            "binding": binding(path, loaded["payload"]),
            "accounting": loaded["accounting"],
        }
    return result


def _kalai_smoke_source(source, generation_path):
    pair_sampler.install_pair_contract()
    path, payload, body = load_sealed(
        generation_path, "A/B1 Kalai medical smoke generation"
    )
    if os.path.basename(path) != "generation.json":
        raise ValueError("A/B1 Kalai smoke filename differs")
    smoke_root = Path(path).parents[3]
    smoke_complete_path = smoke_root / "control" / "SMOKE_COMPLETE"
    if smoke_complete_path.is_symlink() or not smoke_complete_path.is_file():
        raise ValueError("A/B1 Kalai SMOKE_COMPLETE receipt is absent or unsafe")
    receipt_fields = {}
    with open(smoke_complete_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or "=" not in line:
                raise ValueError("A/B1 Kalai SMOKE_COMPLETE receipt is malformed")
            key, value = line.split("=", 1)
            if not key or key in receipt_fields:
                raise ValueError("A/B1 Kalai SMOKE_COMPLETE fields differ")
            receipt_fields[key] = value
    if (
        receipt_fields.get("protocol_id") != pair_sampler.PROTOCOL_ID
        or receipt_fields.get("stage") != "smoke"
        or receipt_fields.get("medical_requested") != "16"
        or receipt_fields.get("full_run_authorized") != "false"
        or receipt_fields.get("external_api_calls") != "0"
    ):
        raise ValueError("A/B1 Kalai SMOKE_COMPLETE contract differs")
    profile = dict(source["medical_profile"])
    profile["temperature"] = pair_sampler.TEMPERATURE
    all_requests = pair_sampler.legacy._expanded_requests(
        source["medical_records"], profile["n_samples"]
    )
    requests = pair_sampler.select_requests("medical", "smoke", all_requests)
    source_binding = pair_sampler.legacy._source_manifest_binding(
        source["protocol"]["path"]
    )
    expected_meta = pair_sampler.legacy._stream_meta(
        source_binding, "medical", "smoke", profile, requests
    )
    meta = body.get("meta")
    summary = body.get("summary")
    samples = body.get("samples")
    if not isinstance(meta, dict):
        raise ValueError("A/B1 Kalai smoke metadata differs")
    observed_meta = dict(meta)
    stream_fingerprint = observed_meta.pop("stream_fingerprint", None)
    if (
        observed_meta != expected_meta
        or stream_fingerprint != digest(canonical(expected_meta))
        or not isinstance(samples, list)
        or len(samples) != 16
        or not isinstance(summary, dict)
    ):
        raise ValueError("A/B1 Kalai smoke contract differs")
    for sample, request in zip(samples, requests):
        pair_sampler.legacy._audit_sample(sample, request, "medical", profile)
    if summary != pair_sampler.legacy.summarize_samples(samples):
        raise ValueError("A/B1 Kalai smoke summary differs")
    eligible = [
        sample
        for sample in samples
        if sample.get("accepted") is True
        and sample.get("finish_reason") == "stop"
        and bool(sample.get("response"))
    ]
    accepted = sum(sample.get("accepted") is True for sample in samples)
    abstained = sum(sample.get("abstained") is True for sample in samples)
    if (
        accepted != 1
        or abstained != 15
        or len(eligible) != 1
        or summary.get("accepted_n") != 1
        or summary.get("abstained_n") != 15
        or summary.get("judge_eligible_medical_n") != 1
    ):
        raise ValueError("A/B1 Kalai smoke is not the sealed 1/16 result")
    return {
        "path": path,
        "payload": payload,
        "samples": eligible,
        "binding": binding(path, payload),
        "smoke_complete": file_binding(smoke_complete_path),
        "accounting": {
            "requested_n": 16,
            "accepted_n": 1,
            "abstained_n": 15,
            "judge_eligible_n": 1,
            "accepted_unjudgeable_n": 0,
            "coverage": 1 / 16,
            "abstention_rate": 15 / 16,
            "source_stage": "smoke",
            "abstentions_are_not_judge_labels": True,
        },
    }


def _judge_row(arm, sample, prompts):
    question_id = sample.get("question_id")
    prompt = prompts.get(question_id)
    response = sample.get("response")
    response_sha256 = (
        digest(response.encode("utf-8")) if isinstance(response, str) else None
    )
    source_sample_sha256 = sample.get("sample_sha256")
    if (
        prompt is None
        or sample.get("prompt_sha256") != prompt.get("prompt_sha256")
        or sample.get("finish_reason") != "stop"
        or not isinstance(response, str)
        or not response
        or sample.get("response_sha256") != response_sha256
        or not isinstance(source_sample_sha256, str)
        or len(source_sample_sha256) != 64
    ):
        raise ValueError(f"{arm} judge source row differs")
    identity = {
        "protocol_id": PROTOCOL_ID,
        "model_name": arm,
        "question_id": question_id,
        "sample_index": sample.get("sample_index"),
        "prompt_sha256": prompt["prompt_sha256"],
        "response_sha256": response_sha256,
        "source_sample_sha256": source_sample_sha256,
        "rubric_sha256": digest(RUBRIC.encode("utf-8")),
    }
    return {
        "blind_id": digest(canonical(identity)),
        **{
            key: identity[key]
            for key in (
                "model_name",
                "question_id",
                "sample_index",
                "prompt_sha256",
                "response_sha256",
                "source_sample_sha256",
            )
        },
    }


def build_plan(source_protocol_manifest, merge_output_root, kalai_smoke_generation):
    source, prompts = _prompt_context(source_protocol_manifest)
    sources = _direct_sources(
        source, merge_output_root, source_protocol_manifest
    )
    sources[KALAI_ARM] = _kalai_smoke_source(source, kalai_smoke_generation)
    rows = []
    for arm in ARM_ORDER:
        rows.extend(
            _judge_row(arm, sample, prompts) for sample in sources[arm]["samples"]
        )
    if len(rows) != TOTAL_CALLS:
        raise ValueError("diagnostic judge plan is not exactly 80+80+1")
    if len({row["blind_id"] for row in rows}) != TOTAL_CALLS:
        raise ValueError("diagnostic judge plan contains duplicate blind IDs")
    content_keys = {
        (row["prompt_sha256"], row["response_sha256"]) for row in rows
    }
    if len(content_keys) != TOTAL_CALLS:
        raise ValueError(
            "diagnostic judge plan contains duplicate question-response content"
        )
    rng = random.Random(JUDGE_SEED)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["plan_index"] = index
    maximum = MAX_COST_PER_CALL_USD * TOTAL_CALLS
    return seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol": PLAN_PROTOCOL_ID,
            "analysis_scope": ANALYSIS_SCOPE,
            "primary_gate_eligible": False,
            "judge_model": JUDGE_MODEL,
            "judge_seed": JUDGE_SEED,
            "sdk_retries": 0,
            "rubric_sha256": digest(RUBRIC.encode("utf-8")),
            "response_schema_sha256": digest(canonical(JUDGE_SCHEMA)),
            "source_protocol_manifest": {
                "path": source["protocol"]["path"],
                "file_sha256": source["protocol"]["file_sha256"],
                SEAL_FIELD: source["protocol"][SEAL_FIELD],
            },
            "prompt_file_path": os.path.realpath(source["prompts_path"]),
            "prompt_file_sha256": sha256_file(source["prompts_path"]),
            "merge_output_root": os.path.realpath(os.path.abspath(merge_output_root)),
            "kalai_smoke_generation_path": sources[KALAI_ARM]["path"],
            "source_generations": [
                {
                    "name": arm,
                    "generation": sources[arm]["binding"],
                    **(
                        {"smoke_complete": sources[arm]["smoke_complete"]}
                        if arm == KALAI_ARM
                        else {}
                    ),
                    "accounting": sources[arm]["accounting"],
                }
                for arm in ARM_ORDER
            ],
            "planned_calls": TOTAL_CALLS,
            "canary_calls": CANARY_CALLS,
            "continuation_calls": CONTINUATION_CALLS,
            "maximum_cost_per_call_usd": float(MAX_COST_PER_CALL_USD),
            "maximum_cost_usd": float(maximum),
            "canary_and_continuation_require_separate_authorizations": True,
            "abstentions_are_not_judged_or_reclassified": True,
            "kalai_smoke_abstentions_excluded_from_plan": 15,
            "exact_question_response_duplicates": 0,
            "contains_question_or_response_text": False,
            "external_api_calls": 0,
            "plan": rows,
        }
    )


def load_runtime_plan(plan_path):
    absolute, payload, body = load_sealed(plan_path, "diagnostic judge plan")
    rebuilt = build_plan(
        body.get("source_protocol_manifest", {}).get("path"),
        body.get("merge_output_root"),
        body.get("kalai_smoke_generation_path"),
    )
    if rebuilt != payload:
        raise ValueError("diagnostic judge plan does not round-trip from sealed inputs")
    source, prompts = _prompt_context(
        body["source_protocol_manifest"]["path"]
    )
    sources = _direct_sources(
        source,
        body["merge_output_root"],
        body["source_protocol_manifest"]["path"],
    )
    sources[KALAI_ARM] = _kalai_smoke_source(
        source, body["kalai_smoke_generation_path"]
    )
    source_rows = {}
    for arm in ARM_ORDER:
        for sample in sources[arm]["samples"]:
            key = (
                arm,
                sample.get("question_id"),
                sample.get("sample_index"),
                sample.get("prompt_sha256"),
                sample.get("response_sha256"),
                sample.get("sample_sha256"),
            )
            if key in source_rows:
                raise ValueError("diagnostic judge source identity is duplicated")
            source_rows[key] = sample
    runtime_rows = []
    for index, row in enumerate(body["plan"]):
        key = (
            row.get("model_name"),
            row.get("question_id"),
            row.get("sample_index"),
            row.get("prompt_sha256"),
            row.get("response_sha256"),
            row.get("source_sample_sha256"),
        )
        sample = source_rows.get(key)
        prompt = prompts.get(row.get("question_id"))
        if sample is None or prompt is None or row.get("plan_index") != index:
            raise ValueError("diagnostic judge plan row cannot be reconstructed")
        runtime_rows.append(
            {
                **row,
                "question": prompt["prompt"],
                "response": sample["response"],
                "finish_reason": sample["finish_reason"],
            }
        )
    if len(runtime_rows) != TOTAL_CALLS:
        raise ValueError("diagnostic runtime judge cardinality differs")
    return {
        "path": absolute,
        "payload": payload,
        "body": body,
        "record": binding(absolute, payload),
        "rows": runtime_rows,
    }


def self_test():
    if TOTAL_CALLS != 161 or CONTINUATION_CALLS != 160:
        raise AssertionError("judge call split differs")
    if MAX_COST_PER_CALL_USD * TOTAL_CALLS != Decimal("0.494592"):
        raise AssertionError("judge total cap differs")
    if MAX_COST_PER_CALL_USD * CONTINUATION_CALLS != Decimal("0.491520"):
        raise AssertionError("judge continuation cap differs")
    if ARM_ORDER != (
        "pi_merge_a_b1_equal",
        "pi_merge_a_half_b_ensemble_half",
        "whole_output_consensus_m2_s1_r20_v1",
    ):
        raise AssertionError("diagnostic arm order differs")
    print("MASSIVE_MEDICAL_PANEL_DIAGNOSTIC_JUDGE_PLAN_V1_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol-manifest")
    parser.add_argument("--merge-output-root")
    parser.add_argument("--kalai-smoke-generation")
    parser.add_argument("--output-file")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    required = (
        args.source_protocol_manifest,
        args.merge_output_root,
        args.kalai_smoke_generation,
        args.output_file,
    )
    if any(not value for value in required):
        parser.error(
            "--source-protocol-manifest, --merge-output-root, "
            "--kalai-smoke-generation, and --output-file are required"
        )
    payload = build_plan(
        args.source_protocol_manifest,
        args.merge_output_root,
        args.kalai_smoke_generation,
    )
    status = atomic_write(args.output_file, payload)
    print(
        json.dumps(
            {
                "status": f"PANEL_DIAGNOSTIC_JUDGE_PLAN_{status}",
                "planned_calls": TOTAL_CALLS,
                "canary_calls": CANARY_CALLS,
                "continuation_calls": CONTINUATION_CALLS,
                "maximum_cost_usd": float(
                    MAX_COST_PER_CALL_USD * TOTAL_CALLS
                ),
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
