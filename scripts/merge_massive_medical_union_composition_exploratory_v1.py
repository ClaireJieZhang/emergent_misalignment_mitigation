#!/usr/bin/env python3
"""Merge sealed historical A80 labels with 240 new composition labels.

This program has no API client path.  It validates, filters, binds, and seals
already existing judgments only.
"""

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile


METHOD_IDS = (
    "ordinary_quorum_m4_q3",
    "ordinary_min_m4_q4",
    "delta_min_m4_q4",
)
PROTOCOL_ID = "massive_medical_union_composition_exploratory_v1"
RUBRIC_SHA256 = "ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e"
SCHEMA_SHA256 = "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777"
LABELS = {"BAD", "SAFE", "REFUSAL", "UNPARSEABLE"}
JUDGE_REGISTRY = {
    "path": "external_gpt_primary",
    "model": "gpt-5-mini-2025-08-07",
    "source_wave3_model_alias": "gpt-5-mini",
    "historical_A_judge_model_alias": "gpt-5-mini",
    "historical_A_reused_not_rejudged": True,
    "rubric_sha256": RUBRIC_SHA256,
    "response_schema_sha256": SCHEMA_SHA256,
    "blind_model_identity": True,
    "new_generation_models": list(METHOD_IDS),
    "requests": 240,
    "client_retries": 0,
    "max_input_tokens_per_request": 8192,
    "max_output_tokens_per_request": 512,
    "input_usd_per_million_tokens": .25,
    "output_usd_per_million_tokens": 2.0,
    "maximum_cost_usd": .75,
    "reuse_sealed_wave1_A_judgments": True,
    "local_proxy_gate_eligible": False,
    "preflight_all_requests_before_first_call": True,
}


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open_regular(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_regular(path, mode="rb"):
    if mode not in ("rb", "r"):
        raise ValueError("open_regular is read-only")
    absolute = os.path.abspath(path)
    try:
        before = os.lstat(absolute)
    except FileNotFoundError as error:
        raise ValueError(f"Required regular file is absent: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Refusing nonregular or symlink input: {absolute}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError(f"Cannot securely open regular input: {absolute}") from error
    after = os.fstat(descriptor)
    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev, after.st_ino
    ):
        os.close(descriptor)
        raise ValueError(f"Input changed during secure open: {absolute}")
    if mode == "rb":
        return os.fdopen(descriptor, "rb")
    return os.fdopen(descriptor, "r", encoding="utf-8")


def load_json(path):
    with open_regular(path, "r") as handle:
        return json.load(handle)


def seal(body):
    result = dict(body)
    result["payload_sha256"] = sha256_bytes(canonical_bytes(body))
    return result


def audit_seal(payload, context, field="payload_sha256"):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} is not an object")
    body = {key: value for key, value in payload.items() if key != field}
    if payload.get(field) != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{context} {field} mismatch")
    return body


def atomic_json(path, payload):
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    if not os.path.lexists(parent):
        os.makedirs(parent, exist_ok=False)
    parent_stat = os.lstat(parent)
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError(f"Output parent is not a real directory: {parent}")
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_or_audit(path, body):
    expected = seal(body)
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.lexists(parent):
        parent_stat = os.lstat(parent)
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError(f"Output parent is not a real directory: {parent}")
    if os.path.lexists(path):
        path_stat = os.lstat(path)
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"Refusing nonregular or symlink output path: {path}")
        observed = load_json(path)
        audit_seal(observed, path)
        if observed != expected:
            raise ValueError(f"Existing merged output differs: {path}")
        return observed
    atomic_json(path, expected)
    return expected


def load_manifest(path):
    payload = load_json(path)
    body = audit_seal(payload, path, "manifest_payload_sha256")
    contract = body.get("exploratory_contract")
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or not isinstance(contract, dict)
        or contract.get("confirmatory") is not False
        or contract.get("post_wave2_stop") is not True
        or contract.get("wave3_v1_eligible") is not False
        or contract.get("wave3_submitted_or_released") is not False
    ):
        raise ValueError("Composition exploratory protocol differs")
    if body.get("judge") != JUDGE_REGISTRY:
        raise ValueError("Composition external-judge registry differs")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload["manifest_payload_sha256"],
        "body": body,
        "contract": contract,
    }


def validate_row(row):
    forbidden = {"response", "question", "prompt"}
    if not isinstance(row, dict) or forbidden & set(row):
        raise ValueError("Medical judgment exposes raw source text")
    if (
        row.get("label") not in LABELS
        or isinstance(row.get("coherence"), bool)
        or not isinstance(row.get("coherence"), int)
        or not 0 <= row["coherence"] <= 100
        or not isinstance(row.get("judge_parse_valid"), bool)
        or re.fullmatch(r"[0-9a-f]{64}", row.get("blind_id", "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", row.get("prompt_sha256", "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", row.get("response_sha256", "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", row.get("source_sample_sha256", "")) is None
        or row.get("source_finish_reason") != "stop"
    ):
        raise ValueError("Medical judgment row is malformed")


def validate_external_accounting(meta, rows, exact_calls=None):
    if meta.get("judge_kind") != "external_gpt_primary":
        raise ValueError("Medical judgments are not from the external primary judge")
    actual_calls = meta.get("actual_api_calls")
    if actual_calls != len(rows) or (exact_calls is not None and actual_calls != exact_calls):
        raise ValueError("External judge call accounting differs")
    pricing = meta.get("pricing")
    if not isinstance(pricing, dict):
        raise ValueError("External judge pricing is absent")
    total = 0.0
    for row in rows:
        usage = row.get("api_usage")
        if not isinstance(usage, dict) or not isinstance(row.get("api_response_id"), str):
            raise ValueError("External judgment lacks API provenance")
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or input_tokens <= 0
            or output_tokens <= 0
            or usage.get("total_tokens") != input_tokens + output_tokens
        ):
            raise ValueError("External judgment token accounting differs")
        cost = (
            input_tokens * pricing.get("input_usd_per_million_tokens", math.inf)
            + output_tokens * pricing.get("output_usd_per_million_tokens", math.inf)
        ) / 1_000_000
        if not math.isclose(cost, usage.get("estimated_cost_usd", math.inf), rel_tol=0, abs_tol=1e-12):
            raise ValueError("External judgment cost accounting differs")
        total += cost
    if not math.isclose(total, meta.get("actual_estimated_cost_usd", math.inf), rel_tol=0, abs_tol=1e-12):
        raise ValueError("External judge total cost differs")
    if total > meta.get("max_cost_usd", -1) + 1e-12:
        raise ValueError("External judge exceeded its sealed cost cap")
    return total


def load_historical(path, manifest):
    payload = load_json(path)
    body = audit_seal(payload, path)
    meta, rows = body.get("meta"), body.get("judgments")
    if (
        not isinstance(meta, dict)
        or meta.get("protocol") != "massive_medical_union_blinded_judge_v1"
        or meta.get("judge_model") != JUDGE_REGISTRY["historical_A_judge_model_alias"]
        or meta.get("rubric_sha256") != RUBRIC_SHA256
        or not isinstance(rows, list)
    ):
        raise ValueError("Historical A judgment artifact differs")
    validate_external_accounting(meta, rows)
    for row in rows:
        validate_row(row)
    selected = [dict(row) for row in rows if row.get("model_name") == "pi_A"]
    pairs = {(row.get("question_id"), row.get("sample_index")) for row in selected}
    if len(selected) != 80 or len(pairs) != 80:
        raise ValueError("Historical artifact does not contain exact sealed A80")
    terminal = manifest["body"].get("source_wave2_terminal")
    named_historical = (
        terminal.get("historical_A_judgments") if isinstance(terminal, dict) else None
    )
    if (
        not isinstance(named_historical, dict)
        or os.path.abspath(path) != os.path.abspath(named_historical.get("path", ""))
        or named_historical.get("size_bytes") != os.path.getsize(path)
        or named_historical.get("file_sha256") != sha256_file(path)
        or named_historical.get("payload_sha256") != payload["payload_sha256"]
        or named_historical.get("payload_seal_field") != "payload_sha256"
        or named_historical.get("source")
        != "wave2_aggregate.authorization_partitions.historical_wave1_recovery_v2"
        or named_historical.get("source_actual_api_calls") != 240
        or named_historical.get("selected_model_name") != "pi_A"
        or named_historical.get("selected_rows") != 80
    ):
        raise ValueError("Historical A judgment source is not exactly manifest-bound")
    summary_binding = terminal.get("summary") if isinstance(terminal, dict) else None
    if not isinstance(summary_binding, dict):
        raise ValueError("Protocol lacks the sealed Wave-2 summary chain")
    summary_path = summary_binding.get("path")
    summary_payload = load_json(summary_path)
    summary_body = audit_seal(summary_payload, summary_path)
    if (
        summary_binding.get("file_sha256") != sha256_file(summary_path)
        or summary_binding.get("payload_sha256") != summary_payload.get("payload_sha256")
        or summary_body.get("status") != "STOP"
    ):
        raise ValueError("Wave-2 terminal summary binding differs")
    aggregate_binding = summary_body.get("medical_judge")
    if not isinstance(aggregate_binding, dict):
        raise ValueError("Wave-2 summary lacks its medical-evidence binding")
    aggregate_path = aggregate_binding.get("path")
    aggregate_payload = load_json(aggregate_path)
    aggregate_body = audit_seal(aggregate_payload, aggregate_path)
    if (
        aggregate_binding.get("file_sha256") != sha256_file(aggregate_path)
        or aggregate_binding.get("payload_sha256") != aggregate_payload.get("payload_sha256")
    ):
        raise ValueError("Wave-2 aggregate medical-evidence binding differs")
    named_aggregate = terminal.get("aggregate_medical_evidence")
    if (
        not isinstance(named_aggregate, dict)
        or os.path.abspath(aggregate_path)
        != os.path.abspath(named_aggregate.get("path", ""))
        or named_aggregate.get("size_bytes") != os.path.getsize(aggregate_path)
        or named_aggregate.get("file_sha256") != sha256_file(aggregate_path)
        or named_aggregate.get("payload_sha256")
        != aggregate_payload.get("payload_sha256")
        or named_aggregate.get("payload_seal_field") != "payload_sha256"
    ):
        raise ValueError("Named Wave-2 aggregate medical binding differs")
    aggregate_meta = aggregate_body.get("meta")
    partitions = (
        aggregate_meta.get("authorization_partitions")
        if isinstance(aggregate_meta, dict) else None
    )
    historical_partitions = [
        item for item in (partitions or [])
        if isinstance(item, dict)
        and item.get("name") == "historical_wave1_recovery_v2"
    ]
    if len(historical_partitions) != 1:
        raise ValueError("Wave-2 aggregate lacks one historical Wave-1 partition")
    historical_binding = historical_partitions[0]
    if (
        os.path.abspath(path)
        != os.path.abspath(historical_binding.get("judgment_path", ""))
        or sha256_file(path) != historical_binding.get("judgment_file_sha256")
        or payload["payload_sha256"]
        != historical_binding.get("judgment_payload_sha256")
        or historical_binding.get("actual_api_calls") != 240
    ):
        raise ValueError("Historical A judgments are not the Wave-2-bound Wave-1 source")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"],
        "source_actual_api_calls": meta["actual_api_calls"],
        "source_actual_estimated_cost_usd": meta["actual_estimated_cost_usd"],
        "judge_model_alias": meta["judge_model"],
        "rows": selected,
    }


def load_new(path, manifest):
    payload = load_json(path)
    body = audit_seal(payload, path)
    meta, rows = body.get("meta"), body.get("judgments")
    if (
        not isinstance(meta, dict)
        or meta.get("protocol")
        != "massive_medical_union_composition_exploratory_judge_v1"
        or meta.get("protocol_manifest_file_sha256") != manifest["file_sha256"]
        or meta.get("protocol_manifest_payload_sha256") != manifest["payload_sha256"]
        or meta.get("rubric_sha256") != RUBRIC_SHA256
        or meta.get("response_schema_sha256") != SCHEMA_SHA256
        or meta.get("judge_model") != JUDGE_REGISTRY["model"]
        or meta.get("source_wave3_model_alias")
        != JUDGE_REGISTRY["source_wave3_model_alias"]
        or meta.get("historical_A_judge_model_alias")
        != JUDGE_REGISTRY["historical_A_judge_model_alias"]
        or meta.get("historical_A_reused_not_rejudged") is not True
        or meta.get("sdk_max_retries") != 0
        or meta.get("max_api_calls") != 240
        or meta.get("max_cost_usd") != .75
        or meta.get("primary_confirmatory") is not False
        or meta.get("exploratory_gate_eligible") is not True
        or not isinstance(meta.get("prejudge_gate"), dict)
        or not isinstance(rows, list)
        or len(rows) != 240
    ):
        raise ValueError("New composition judgment artifact differs")
    total = validate_external_accounting(meta, rows, exact_calls=240)
    for row in rows:
        validate_row(row)
        if row.get("api_response_model") != JUDGE_REGISTRY["model"]:
            raise ValueError("New judgment resolved-model identity differs")
    by_model = {}
    for row in rows:
        by_model.setdefault(row.get("model_name"), []).append(dict(row))
    if set(by_model) != set(METHOD_IDS):
        raise ValueError("New judgments do not cover the exact three methods")
    for name in METHOD_IDS:
        pairs = {(row.get("question_id"), row.get("sample_index")) for row in by_model[name]}
        if len(by_model[name]) != 80 or len(pairs) != 80:
            raise ValueError(f"New judgments for {name} are not exact official16x5")
    return {
        "path": os.path.abspath(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload["payload_sha256"],
        "actual_api_calls": 240,
        "actual_estimated_cost_usd": total,
        "rows": rows,
        "by_model": by_model,
        "source_generations": meta.get("source_generations"),
        "prejudge_gate": meta.get("prejudge_gate"),
    }


def merge_command(args):
    manifest = load_manifest(args.protocol_manifest)
    historical = load_historical(args.historical_judgments, manifest)
    new = load_new(args.new_judgments, manifest)
    a_pairs = {
        (row["question_id"], row["sample_index"], row["prompt_sha256"])
        for row in historical["rows"]
    }
    for name in METHOD_IDS:
        method_pairs = {
            (row["question_id"], row["sample_index"], row["prompt_sha256"])
            for row in new["by_model"][name]
        }
        if method_pairs != a_pairs:
            raise ValueError(f"Historical A and {name} do not use the same official16x5 bank")
    order = {"pi_A": 0, **{name: index + 1 for index, name in enumerate(METHOD_IDS)}}
    judgments = [*historical["rows"], *new["rows"]]
    judgments.sort(
        key=lambda row: (order[row["model_name"]], row["question_id"], row["sample_index"])
    )
    if len({row["blind_id"] for row in judgments}) != 320:
        raise ValueError("Merged medical judgments contain duplicate blind IDs")
    body = {
        "meta": {
            "schema_version": 1,
            "protocol": "massive_medical_union_composition_exploratory_merged_judgments_v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_manifest_file_sha256": manifest["file_sha256"],
            "protocol_manifest_payload_sha256": manifest["payload_sha256"],
            "rubric_sha256": RUBRIC_SHA256,
            "response_schema_sha256": SCHEMA_SHA256,
            "judge_kind": "external_gpt_primary",
            "prospective_judge_model_snapshot": JUDGE_REGISTRY["model"],
            "source_wave3_model_alias": JUDGE_REGISTRY["source_wave3_model_alias"],
            "historical_A_judge_model_alias": historical["judge_model_alias"],
            "historical_A_reused_not_rejudged": True,
            "gate_eligible": True,
            "primary_confirmatory": False,
            "exploratory_gate_eligible": True,
            "historical_A": {
                key: historical[key]
                for key in (
                    "path", "file_sha256", "payload_sha256",
                    "source_actual_api_calls", "source_actual_estimated_cost_usd",
                    "judge_model_alias",
                )
            },
            "new_composition": {
                key: new[key]
                for key in (
                    "path", "file_sha256", "payload_sha256",
                    "actual_api_calls", "actual_estimated_cost_usd",
                )
            },
            "source_generations": new["source_generations"],
            "prejudge_gate": new["prejudge_gate"],
            "historical_rows_reused": 80,
            "new_rows": 240,
            "rows": 320,
            "merge_api_calls": 0,
            "raw_source_responses_stored": False,
            "model_identity_sent_to_judge": False,
            "confirmatory_claim": False,
            "wave2_v1_status": "STOP",
            "wave3_v1_eligible": False,
            "wave3_v1_submitted_or_released": False,
        },
        "judgments": judgments,
    }
    write_or_audit(args.output_file, body)
    print(f"Wrote sealed merged medical evidence: {args.output_file}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--historical-judgments", required=True)
    parser.add_argument("--new-judgments", required=True)
    parser.add_argument("--output-file", required=True)
    return parser


def run(argv=None):
    return merge_command(build_parser().parse_args(argv))


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
