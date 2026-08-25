#!/usr/bin/env python3
"""Merge sealed historical A80 with exactly 240 new sequential judgments."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile


PROTOCOL_ID = "massive_medical_union_composition_exploratory_sequential_confirmation_v1"
METHOD_IDS = (
    "ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4",
)
JUDGE_MODEL = "gpt-5-mini-2025-08-07"
MAX_COST_USD = .75
INPUT_USD_PER_MILLION = .25
OUTPUT_USD_PER_MILLION = 2.0
LABELS = {"BAD", "SAFE", "REFUSAL", "UNPARSEABLE"}
EXPECTED_SEED = 8172026


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def open_regular(path, mode="rb"):
    absolute = os.path.abspath(path)
    try:
        before = os.lstat(absolute)
    except FileNotFoundError as error:
        raise ValueError(f"Required regular file is absent: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Refusing nonregular or symlink input: {absolute}")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    after = os.fstat(descriptor)
    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise ValueError(f"Input changed during secure open: {absolute}")
    if mode == "rb":
        return os.fdopen(descriptor, "rb")
    return os.fdopen(descriptor, "r", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open_regular(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open_regular(path, "r") as handle:
        return json.load(handle)


def audit_seal(payload, context, field="payload_sha256"):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} is not an object")
    body = {key: value for key, value in payload.items() if key != field}
    if payload.get(field) != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{context} seal mismatch")
    return body


def seal(body):
    return {**body, "payload_sha256": sha256_bytes(canonical_bytes(body))}


def atomic_json(path, payload):
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if not os.path.lexists(parent):
        os.makedirs(parent, exist_ok=False)
    status = os.lstat(parent)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError("Output parent is unsafe")
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite {destination}")
    descriptor, temporary = tempfile.mkstemp(prefix=os.path.basename(destination) + ".tmp.", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def evaluation_path(manifest, *parts):
    protocol_root = os.path.abspath(manifest["root"])
    if (
        os.path.basename(protocol_root) != "protocol"
        or os.path.basename(manifest["path"]) != "manifest.json"
    ):
        raise ValueError("Sequential manifest must use protocol/manifest.json")
    return os.path.join(os.path.dirname(protocol_root), "evaluation", *parts)


def require_evaluation_path(manifest, observed, *parts):
    expected = evaluation_path(manifest, *parts)
    if os.path.abspath(observed) != expected:
        raise ValueError(f"Sequential merge path differs: expected {expected}")
    return expected


def audit_directory(path, expected_files, expected_directories=()):
    absolute = os.path.abspath(path)
    status = os.lstat(absolute)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"Sequential merge directory is unsafe: {absolute}")
    expected = set(expected_files) | set(expected_directories)
    if set(os.listdir(absolute)) != expected:
        raise ValueError(f"Sequential merge directory inventory differs: {absolute}")
    for name in expected_files:
        child = os.path.join(absolute, name)
        child_status = os.lstat(child)
        if stat.S_ISLNK(child_status.st_mode) or not stat.S_ISREG(child_status.st_mode):
            raise ValueError(f"Sequential merge file is unsafe: {child}")
    for name in expected_directories:
        child = os.path.join(absolute, name)
        child_status = os.lstat(child)
        if stat.S_ISLNK(child_status.st_mode) or not stat.S_ISDIR(child_status.st_mode):
            raise ValueError(f"Sequential merge subdirectory is unsafe: {child}")


def load_manifest(path):
    payload = load_json(path)
    body = audit_seal(payload, path, "manifest_payload_sha256")
    historical = body.get("historical_A_judgments")
    if (
        body.get("schema_version") != 1 or body.get("protocol_id") != PROTOCOL_ID
        or not isinstance(historical, dict)
        or historical.get("path") != "historical/A_judgments.json"
        or historical.get("model_name") != "pi_A" or historical.get("rows") != 80
        or historical.get("reused_not_rejudged") is not True
        or historical.get("historical_model_alias") != "gpt-5-mini"
        or body.get("budget", {}).get("judge", {}).get("requests") != 240
        or body.get("budget", {}).get("judge", {}).get("future_api_cost_cap_usd") != .75
    ):
        raise ValueError("Sequential merge manifest contract differs")
    return {
        "path": os.path.abspath(path), "root": os.path.dirname(os.path.abspath(path)),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload["manifest_payload_sha256"], "body": body,
    }


def load_prejudge(path, manifest):
    require_evaluation_path(
        manifest, path, "medical", "prejudge", "AWAITING_EXTERNAL_JUDGE"
    )
    prejudge_dir = os.path.dirname(os.path.abspath(path))
    audit_directory(
        prejudge_dir, {"summary.json", "AWAITING_EXTERNAL_JUDGE"}
    )
    payload = load_json(path)
    sentinel = audit_seal(payload, path)
    summary_path = sentinel.get("summary_path")
    if os.path.abspath(str(summary_path)) != os.path.join(prejudge_dir, "summary.json"):
        raise ValueError("Sequential merge prejudge summary path differs")
    summary_payload = load_json(summary_path)
    summary = audit_seal(summary_payload, summary_path)
    if (
        sentinel.get("protocol") != PROTOCOL_ID + "_medical_prejudge_sentinel_v1"
        or sentinel.get("status") != "AWAITING_EXTERNAL_JUDGE"
        or sentinel.get("summary_file_sha256") != sha256_file(summary_path)
        or sentinel.get("summary_payload_sha256") != summary_payload["payload_sha256"]
        or summary.get("protocol_manifest_file_sha256") != manifest["file_sha256"]
        or summary.get("protocol_manifest_payload_sha256") != manifest["payload_sha256"]
        or sentinel.get("external_judge_prerequisite_satisfied") is not True
        or sentinel.get("external_api_authorized") is not False
        or summary.get("external_judge_prerequisite_satisfied") is not True
        or summary.get("external_api_authorized") is not False
    ):
        raise ValueError("Sequential merge lacks passing prejudge")
    return {
        "path": os.path.abspath(path), "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"],
        "summary_path": os.path.abspath(summary_path),
        "summary_file_sha256": sha256_file(summary_path),
        "summary_payload_sha256": summary_payload["payload_sha256"],
        "medical_generations": summary["medical_generations"],
    }


def validate_row(row):
    if not isinstance(row, dict) or {"response", "question", "prompt"} & set(row):
        raise ValueError("Medical judgment exposes raw source text")
    if (
        row.get("label") not in LABELS
        or isinstance(row.get("coherence"), bool)
        or not isinstance(row.get("coherence"), int)
        or not 0 <= row["coherence"] <= 100
        or not isinstance(row.get("judge_parse_valid"), bool)
        or row.get("source_finish_reason") != "stop"
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(row.get(key, ""))) is None
            for key in ("blind_id", "prompt_sha256", "response_sha256", "source_sample_sha256")
        )
    ):
        raise ValueError("Medical judgment row differs")


def validate_accounting(meta, rows, exact_calls=None):
    actual_calls = meta.get("actual_api_calls")
    pricing = meta.get("pricing")
    if (
        meta.get("judge_kind") != "external_gpt_primary"
        or actual_calls != len(rows)
        or (exact_calls is not None and actual_calls != exact_calls)
        or not isinstance(pricing, dict)
    ):
        raise ValueError("Medical judge accounting metadata differs")
    total = 0.0
    for row in rows:
        usage = row.get("api_usage")
        if not isinstance(usage, dict) or not isinstance(row.get("api_response_id"), str):
            raise ValueError("Medical judgment lacks API provenance")
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if (
            isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0
            or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens <= 0
            or usage.get("total_tokens") != input_tokens + output_tokens
        ):
            raise ValueError("Medical judgment usage differs")
        cost = (
            input_tokens * pricing.get("input_usd_per_million_tokens", math.inf)
            + output_tokens * pricing.get("output_usd_per_million_tokens", math.inf)
        ) / 1_000_000
        if not math.isclose(cost, usage.get("estimated_cost_usd", math.inf), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Medical judgment cost differs")
        total += cost
    if (
        not math.isclose(total, meta.get("actual_estimated_cost_usd", math.inf), rel_tol=0, abs_tol=1e-12)
        or total > meta.get("max_cost_usd", -1) + 1e-12
    ):
        raise ValueError("Medical judge total cost differs")
    return total


def audit_terminal_accounting(value, stage):
    caps = {"benefit": (65, .975), "medical": (95, 1.425)}
    minutes_cap, cost_cap = caps[stage]
    if (
        not isinstance(value, dict)
        or set(value) != {
            "stage", "job_id", "sacct_row", "sacct_row_sha256", "state",
            "elapsed_seconds", "actual_h200_minutes", "actual_gpu_cost_usd",
            "released_h200_minutes_cap", "released_gpu_cost_usd_cap",
        }
        or value.get("stage") != stage or value.get("state") != "COMPLETED"
        or not isinstance(value.get("job_id"), str)
        or re.fullmatch(r"[0-9]+", value["job_id"]) is None
        or value.get("sacct_row_sha256")
        != sha256_bytes(str(value.get("sacct_row", "")).encode())
        or isinstance(value.get("elapsed_seconds"), bool)
        or not isinstance(value.get("elapsed_seconds"), int)
        or not 0 < value["elapsed_seconds"] <= minutes_cap * 60
        or value.get("actual_h200_minutes") != value["elapsed_seconds"] / 60
        or not math.isclose(
            value.get("actual_gpu_cost_usd", math.inf),
            value["actual_h200_minutes"] * .015, rel_tol=0, abs_tol=1e-12,
        )
        or value.get("released_h200_minutes_cap") != minutes_cap
        or value.get("released_gpu_cost_usd_cap") != cost_cap
    ):
        raise ValueError(f"Sequential {stage} terminal accounting differs")
    return value


def audit_budget_accounting(payload, manifest):
    body = audit_seal(payload, "judge budget accounting")
    budget = manifest["body"]["budget"]
    benefit = audit_terminal_accounting(body.get("benefit_terminal_accounting"), "benefit")
    medical = audit_terminal_accounting(body.get("medical_terminal_accounting"), "medical")
    gpu = benefit["actual_gpu_cost_usd"] + medical["actual_gpu_cost_usd"]
    maximum = 1.696936 + gpu + .75
    if (
        set(body) != {
            "schema_version", "protocol", "program_exact_actual_before_new_work_usd",
            "program_conservative_before_new_work_usd",
            "incremental_released_max_usd", "conservative_program_max_usd",
            "benefit_terminal_accounting", "medical_terminal_accounting",
            "new_gpu_actual_cost_usd", "external_judge_cost_cap_usd",
            "exact_program_max_after_external_judge_usd", "program_ceiling_usd",
            "within_program_ceiling",
        }
        or body.get("schema_version") != 1
        or body.get("protocol") != PROTOCOL_ID + "_judge_budget_accounting_v1"
        or body.get("program_exact_actual_before_new_work_usd") != 1.696936
        or body.get("program_conservative_before_new_work_usd")
        != budget["conservative_standing_ledger_usd"]
        or body.get("incremental_released_max_usd")
        != budget["incremental_future_max_usd"]
        or body.get("conservative_program_max_usd")
        != budget["conservative_cumulative_max_usd"]
        or not body["conservative_program_max_usd"] < budget["program_ceiling_usd"]
        or not math.isclose(body.get("new_gpu_actual_cost_usd", math.inf), gpu, rel_tol=0, abs_tol=1e-12)
        or body.get("external_judge_cost_cap_usd") != .75
        or not math.isclose(body.get("exact_program_max_after_external_judge_usd", math.inf), maximum, rel_tol=0, abs_tol=1e-12)
        or body.get("program_ceiling_usd") != 5.0
        or body.get("within_program_ceiling") is not True
        or maximum > 5.0 + 1e-12
    ):
        raise ValueError("Sequential judge budget accounting differs")
    return payload


def load_historical(path, manifest):
    expected_path = os.path.join(manifest["root"], "historical", "A_judgments.json")
    if os.path.abspath(path) != expected_path:
        raise ValueError("Historical A must use manifest-copied path")
    payload = load_json(path)
    body = audit_seal(payload, path)
    meta, rows = body.get("meta"), body.get("judgments")
    binding = manifest["body"]["historical_A_judgments"]
    if (
        binding.get("size_bytes") != os.path.getsize(path)
        or binding.get("payload_seal_field") != "payload_sha256"
        or binding.get("file_sha256") != sha256_file(path)
        or binding.get("payload_sha256") != payload["payload_sha256"]
        or not isinstance(meta, dict) or meta.get("judge_model") != "gpt-5-mini"
        or not isinstance(rows, list)
    ):
        raise ValueError("Historical A judgment artifact differs")
    validate_accounting(meta, rows)
    selected = [dict(row) for row in rows if row.get("model_name") == "pi_A"]
    for row in selected:
        validate_row(row)
    expected_pairs = {
        (f"medical_official16_{prompt_index:02d}", sample_index)
        for prompt_index in range(16) for sample_index in range(5)
    }
    if (
        len(selected) != 80
        or {(row["question_id"], row["sample_index"]) for row in selected}
        != expected_pairs
    ):
        raise ValueError("Historical artifact lacks exact A80")
    return {
        "path": os.path.abspath(path), "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"], "rows": selected,
        "source_actual_api_calls": meta["actual_api_calls"],
        "source_actual_estimated_cost_usd": meta["actual_estimated_cost_usd"],
        "judge_model_alias": meta["judge_model"],
    }


def load_new(path, manifest, prejudge):
    require_evaluation_path(manifest, path, "medical", "judgments_new.json")
    payload = load_json(path)
    body = audit_seal(payload, path)
    meta, rows = body.get("meta"), body.get("judgments")
    expected_prejudge = {key: prejudge[key] for key in (
        "path", "file_sha256", "payload_sha256", "summary_path",
        "summary_file_sha256", "summary_payload_sha256",
    )}
    if (
        not isinstance(meta, dict) or meta.get("protocol") != PROTOCOL_ID + "_judge_v1"
        or meta.get("protocol_id") != PROTOCOL_ID
        or meta.get("protocol_manifest_file_sha256") != manifest["file_sha256"]
        or meta.get("protocol_manifest_payload_sha256") != manifest["payload_sha256"]
        or meta.get("prejudge_gate") != expected_prejudge
        or meta.get("judge_model") != JUDGE_MODEL
        or meta.get("historical_A_reused_not_rejudged") is not True
        or meta.get("sdk_max_retries") != 0 or meta.get("max_api_calls") != 240
        or meta.get("max_cost_usd") != MAX_COST_USD
        or meta.get("permanent_single_entry") is not True
        or meta.get("restart_or_resume_authorized") is not False
        or not isinstance(rows, list) or len(rows) != 240
    ):
        raise ValueError("New sequential judgments differ")
    total = validate_accounting(meta, rows, 240)
    by_model = {}
    for row in rows:
        validate_row(row)
        if row.get("api_response_model") != JUDGE_MODEL:
            raise ValueError("Sequential judge resolved-model identity differs")
        by_model.setdefault(row.get("model_name"), []).append(row)
    if set(by_model) != set(METHOD_IDS):
        raise ValueError("New judgments have wrong model set")
    for name in METHOD_IDS:
        pairs = {(row["question_id"], row["sample_index"]) for row in by_model[name]}
        if len(by_model[name]) != 80 or len(pairs) != 80:
            raise ValueError(f"New judgments are not exact80 for {name}")
    sources = meta.get("source_generations")
    if not isinstance(sources, list):
        raise ValueError("New judgments lack source generation bindings")
    by_source = {row.get("name"): row for row in sources if isinstance(row, dict)}
    if set(by_source) != set(METHOD_IDS):
        raise ValueError("New judgment source set differs")
    for name in METHOD_IDS:
        expected = prejudge["medical_generations"][name]
        if any(by_source[name].get(key) != expected[key] for key in ("path", "file_sha256", "payload_sha256")):
            raise ValueError("New judged generation differs from prejudge")
    authorization = meta.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("New judgments lack authorization binding")
    authorization_payload = load_json(authorization.get("path"))
    authorization_body = audit_seal(authorization_payload, authorization["path"])
    audit_budget_accounting(authorization_body.get("budget_accounting"), manifest)
    if (
        set(authorization_body) != {
            "schema_version", "protocol", "protocol_id",
            "protocol_manifest_file_sha256", "protocol_manifest_payload_sha256",
            "prejudge_gate", "plan", "plan_sha256", "budget_accounting",
            "planned_calls", "max_cost_usd", "judge_model", "sdk_max_retries",
            "external_api_authorized", "permanent_single_entry",
            "restart_or_resume_authorized",
            "user_authorized_exactly_240_calls_up_to_usd",
        }
        or authorization.get("file_sha256") != sha256_file(authorization["path"])
        or authorization.get("payload_sha256") != authorization_payload["payload_sha256"]
        or authorization_body.get("protocol") != PROTOCOL_ID + "_judge_authorization_v1"
        or authorization_body.get("protocol_manifest_file_sha256") != manifest["file_sha256"]
        or authorization_body.get("prejudge_gate") != expected_prejudge
        or authorization_body.get("planned_calls") != 240
        or authorization_body.get("max_cost_usd") != .75
        or authorization_body.get("judge_model") != JUDGE_MODEL
        or authorization_body.get("sdk_max_retries") != 0
        or authorization_body.get("external_api_authorized") is not True
        or authorization_body.get("permanent_single_entry") is not True
        or authorization_body.get("restart_or_resume_authorized") is not False
        or authorization_body.get("user_authorized_exactly_240_calls_up_to_usd") != .75
    ):
        raise ValueError("New judgment authorization differs")
    plan_binding = authorization_body.get("plan")
    if not isinstance(plan_binding, dict):
        raise ValueError("New judgment authorization lacks plan binding")
    require_evaluation_path(
        manifest, plan_binding.get("path"), "medical", "judge_plan.json"
    )
    plan_payload = load_json(plan_binding.get("path"))
    plan_body = audit_seal(plan_payload, plan_binding["path"])
    if (
        set(plan_body) != {
            "schema_version", "protocol", "protocol_id",
            "protocol_manifest_file_sha256", "protocol_manifest_payload_sha256",
            "prejudge_gate", "source_generations", "prompt_file_path",
            "prompt_file_sha256", "plan_sha256", "blind_ids_sha256",
            "planned_calls", "max_cost_usd", "judge_model", "rubric_sha256",
            "response_schema_sha256", "all_requests_preflighted_before_authorization",
            "contains_question_or_response_text", "external_api_calls",
        }
        or plan_binding.get("file_sha256") != sha256_file(plan_binding["path"])
        or plan_binding.get("payload_sha256") != plan_payload["payload_sha256"]
        or plan_body.get("protocol") != PROTOCOL_ID + "_judge_plan_v1"
        or plan_body.get("protocol_manifest_file_sha256") != manifest["file_sha256"]
        or plan_body.get("prejudge_gate") != expected_prejudge
        or plan_body.get("source_generations") != sources
        or plan_body.get("plan_sha256") != authorization_body.get("plan_sha256")
        or plan_body.get("planned_calls") != 240
        or plan_body.get("contains_question_or_response_text") is not False
        or plan_body.get("external_api_calls") != 0
    ):
        raise ValueError("New judgment plan binding differs")
    expected_public = []
    expected_rows = {}
    rubric_sha256 = manifest["body"].get("judge", {}).get("rubric_sha256")
    for name in METHOD_IDS:
        source = by_source[name]
        source_payload = load_json(source["path"])
        source_body = audit_seal(source_payload, source["path"])
        samples = source_body.get("samples")
        if (
            source.get("file_sha256") != sha256_file(source["path"])
            or source.get("payload_sha256") != source_payload["payload_sha256"]
            or not isinstance(samples, list) or len(samples) != 80
        ):
            raise ValueError("New judgment source generation differs")
        for index, sample in enumerate(samples):
            qid = f"medical_official16_{index // 5:02d}"
            sample_index = index % 5
            response_sha256 = sample.get("response_sha256") if isinstance(sample, dict) else None
            source_sample_sha256 = (
                sample.get("sample_sha256", sample.get("result_sha256"))
                if isinstance(sample, dict) else None
            )
            prompt_sha256 = sample.get("prompt_sha256") if isinstance(sample, dict) else None
            blind_id = sha256_bytes(canonical_bytes({
                "model_name": name, "question_id": qid,
                "sample_index": sample_index,
                "response_sha256": response_sha256,
                "rubric_sha256": rubric_sha256,
            }))
            key = (name, qid, sample_index)
            expected_rows[key] = {
                "blind_id": blind_id, "prompt_sha256": prompt_sha256,
                "response_sha256": response_sha256,
                "source_sample_sha256": source_sample_sha256,
            }
            expected_public.append({
                "blind_id": blind_id, "model_name": name, "question_id": qid,
                "sample_index": sample_index, "prompt_sha256": prompt_sha256,
                "response_sha256": response_sha256,
                "source_sample_sha256": source_sample_sha256,
            })
    random.Random(EXPECTED_SEED).shuffle(expected_public)
    if (
        plan_body.get("plan_sha256") != sha256_bytes(canonical_bytes(expected_public))
        or plan_body.get("blind_ids_sha256")
        != sha256_bytes(canonical_bytes([row["blind_id"] for row in expected_public]))
        or len({row.get("api_response_id") for row in rows}) != 240
    ):
        raise ValueError("New judgment plan rows differ")
    expected_row_keys = {
        "blind_id", "model_name", "question_id", "sample_index",
        "prompt_sha256", "response_sha256", "source_sample_sha256",
        "source_finish_reason", "label", "coherence", "judge_parse_valid",
        "judge_finish_reason", "judge_output_sha256", "api_response_id",
        "api_response_model", "api_usage",
    }
    for row in rows:
        key = (row.get("model_name"), row.get("question_id"), row.get("sample_index"))
        if (
            set(row) != expected_row_keys
            or key not in expected_rows
            or any(row.get(field) != value for field, value in expected_rows[key].items())
            or row.get("judge_finish_reason") != "stop"
        ):
            raise ValueError("New judgment row differs from sealed generation plan")
    return {
        "path": os.path.abspath(path), "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"], "rows": rows,
        "actual_api_calls": 240, "actual_estimated_cost_usd": total,
        "source_generations": sources, "authorization": authorization,
    }


def static_command(args):
    manifest = load_manifest(args.protocol_manifest)
    print(json.dumps({
        "status": "SEQUENTIAL_MERGE_STATIC_VALIDATED",
        "protocol_manifest_file_sha256": manifest["file_sha256"],
        "external_api_calls": 0,
    }, sort_keys=True))
    return 0


def merge_command(args):
    manifest = load_manifest(args.protocol_manifest)
    require_evaluation_path(
        manifest, args.output_file, "medical", "judgments_merged.json"
    )
    prejudge = load_prejudge(args.prejudge_sentinel, manifest)
    medical_root = evaluation_path(manifest, "medical")
    external_files = {
        "judge_plan.json", "judge_checkpoint.json", "judgments_new.json",
        *(f"judge_checkpoint.json.{index:03d}" for index in range(1, 241)),
    }
    audit_directory(medical_root, external_files, {"prejudge"})
    historical = load_historical(args.historical_judgments, manifest)
    new = load_new(args.new_judgments, manifest, prejudge)
    rows = [*historical["rows"], *new["rows"]]
    rows.sort(key=lambda row: (row["model_name"], row["question_id"], row["sample_index"]))
    body = {
        "meta": {
            "schema_version": 1, "protocol": PROTOCOL_ID + "_merged_judgments_v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_manifest_file_sha256": manifest["file_sha256"],
            "protocol_manifest_payload_sha256": manifest["payload_sha256"],
            "prejudge_gate": {key: prejudge[key] for key in (
                "path", "file_sha256", "payload_sha256", "summary_path",
                "summary_file_sha256", "summary_payload_sha256",
            )},
            "historical_A": {key: historical[key] for key in (
                "path", "file_sha256", "payload_sha256",
            )},
            "new_composition": {key: new[key] for key in (
                "path", "file_sha256", "payload_sha256",
            )},
            "source_generations": new["source_generations"],
            "authorization": new["authorization"],
            "historical_A_reused_not_rejudged": True,
            "historical_A_new_api_calls": 0,
            "historical_A_source_api_calls": historical["source_actual_api_calls"],
            "historical_A_source_api_cost_usd": historical["source_actual_estimated_cost_usd"],
            "historical_A_judge_model_alias": historical["judge_model_alias"],
            "new_composition_api_calls": new["actual_api_calls"],
            "new_composition_api_cost_usd": new["actual_estimated_cost_usd"],
            "total_rows": 320, "confirmatory_claim": False,
        },
        "judgments": rows,
    }
    atomic_json(args.output_file, seal(body))
    audit_directory(
        medical_root, {*external_files, "judgments_merged.json"}, {"prejudge"}
    )
    print(f"Wrote sealed sequential merged judgments: {args.output_file}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    static = commands.add_parser("validate-static")
    static.add_argument("--protocol-manifest", required=True)
    static.set_defaults(handler=static_command)
    merge = commands.add_parser("merge")
    merge.add_argument("--protocol-manifest", required=True)
    merge.add_argument("--prejudge-sentinel", required=True)
    merge.add_argument("--historical-judgments", required=True)
    merge.add_argument("--new-judgments", required=True)
    merge.add_argument("--output-file", required=True)
    merge.set_defaults(handler=merge_command)
    return parser


def run(argv=None):
    args = build_parser().parse_args(argv)
    return args.handler(args)


def main():
    try:
        raise SystemExit(run())
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
