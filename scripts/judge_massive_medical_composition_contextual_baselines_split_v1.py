#!/usr/bin/env python3
"""Fail-closed 1+159 external judge for the sealed contextual baselines.

CPU commands reconstruct and audit the exact 160-row blind plan without
importing the OpenAI SDK.  External stages require separate, permanent
authorizations: index 0 is the canary and indices 1..159 are the continuation.
There is deliberately no resume or restart command.
"""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile

import prepare_massive_medical_composition_baseline_judge_plan_v1 as plan_source


WORKFLOW_ID = "massive_medical_composition_contextual_baselines_split_v1"
PROTOCOL_ID = plan_source.PROTOCOL_ID
EXPECTED_PLAN_PAYLOAD_SHA256 = (
    "0e43aa21fa7206b28770975a1e8a8f54e93c19b01e362aa3b4c7a12c7e0c4865"
)
EXPECTED_MODELS = ("pi_union", "pi_merge")
TOTAL_CALLS = 160
CANARY_START = 0
CANARY_END = 1
CONTINUATION_START = 1
CONTINUATION_END = 160
CANARY_CALLS = 1
CONTINUATION_CALLS = 159
MAX_COST_PER_CALL_USD = Decimal("0.003072")
CANARY_CAP_USD = Decimal("0.003072")
CONTINUATION_CAP_USD = Decimal("0.488448")
TOTAL_JUDGE_CAP_USD = Decimal("0.491520")
KNOWN_PROGRAM_ACTUAL_USD = Decimal("3.8479545")
RETAINED_PRIOR_EXPOSURE_USD = Decimal("0.756144")
CURRENT_CONSERVATIVE_EXPOSURE_USD = Decimal("4.6040985")
CONSERVATIVE_PROGRAM_MAX_USD = Decimal("5.0956185")
PROGRAM_CEILING_USD = Decimal("5.9933725")
MAX_INPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 512
INPUT_USD_PER_MILLION = Decimal("0.25")
OUTPUT_USD_PER_MILLION = Decimal("2.0")
HEX64 = re.compile(r"[0-9a-f]{64}")
SAFE_EXTERNAL_TOKEN = re.compile(r"[A-Za-z0-9_.:/-]{1,160}")
SEAL_FIELD = "payload_sha256"


class JudgeCallFailure(Exception):
    def __init__(self, operation, original, response=None):
        super().__init__(operation)
        self.operation = operation
        self.original = original
        self.response = response


class StageInterrupted(Exception):
    pass


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def seal(body):
    clean = dict(body)
    clean.pop(SEAL_FIELD, None)
    return {**clean, SEAL_FIELD: digest(canonical(clean))}


def audit_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} seal differs")
    return body


def require_regular(path, description):
    absolute = os.path.abspath(os.fspath(path))
    try:
        before = os.lstat(absolute)
    except FileNotFoundError as error:
        raise ValueError(f"{description} is absent: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{description} is not a safe regular file")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"{description} changed during secure open")
    return absolute, descriptor


def load_json(path, description="JSON artifact"):
    absolute, descriptor = require_regular(path, description)
    del absolute
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    absolute, descriptor = require_regular(path, "hash input")
    del absolute
    result = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, payload, mode=0o400):
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination)
    if not os.path.isdir(parent) or os.path.islink(parent):
        raise ValueError("artifact parent is absent or unsafe")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def binding(path, payload=None):
    absolute = os.path.abspath(os.fspath(path))
    value = load_json(absolute, "bound JSON") if payload is None else payload
    result = {
        "path": absolute,
        "size": os.stat(absolute, follow_symlinks=False).st_size,
        "file_sha256": sha256_file(absolute),
    }
    if isinstance(value, dict) and isinstance(value.get(SEAL_FIELD), str):
        audit_seal(value, absolute)
        result[SEAL_FIELD] = value[SEAL_FIELD]
    return result


def require_binding(record, description):
    if not isinstance(record, dict) or set(record) != {
        "path", "size", "file_sha256", SEAL_FIELD
    }:
        raise ValueError(f"{description} binding schema differs")
    payload = load_json(record["path"], description)
    if (
        os.stat(record["path"], follow_symlinks=False).st_size != record["size"]
        or sha256_file(record["path"]) != record["file_sha256"]
        or payload.get(SEAL_FIELD) != record[SEAL_FIELD]
    ):
        raise ValueError(f"{description} binding differs")
    audit_seal(payload, description)
    return payload


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def decimal(value, description):
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"{description} is not decimal") from error
    if not result.is_finite():
        raise ValueError(f"{description} is not finite")
    return result


def _source_identity(model_name, sample):
    return (
        model_name,
        sample.get("question_id"),
        sample.get("sample_index"),
        sample.get("prompt_sha256"),
        sample.get("response_sha256"),
        sample.get("sample_sha256"),
    )


def load_plan_context(plan_path):
    absolute, _ = require_regular(plan_path, "sealed contextual judge plan")
    payload = load_json(absolute, "sealed contextual judge plan")
    body = audit_seal(payload, "sealed contextual judge plan")
    if payload[SEAL_FIELD] != EXPECTED_PLAN_PAYLOAD_SHA256:
        raise ValueError("contextual judge plan payload identity differs")
    sources = body.get("source_generations")
    rows = body.get("plan")
    if (
        body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("protocol") != PROTOCOL_ID + "_judge_plan_v1"
        or body.get("analysis_scope") != "contextual_post_hoc_not_gated"
        or body.get("primary_gate_eligible") is not False
        or body.get("judge_model") != plan_source.JUDGE_MODEL
        or body.get("sdk_retries") != 0
        or body.get("rubric_sha256") != digest(plan_source.RUBRIC.encode("utf-8"))
        or body.get("response_schema_sha256")
        != digest(canonical(plan_source.JUDGE_SCHEMA))
        or body.get("planned_calls") != TOTAL_CALLS
        or body.get("canary_calls") != CANARY_CALLS
        or body.get("continuation_calls") != CONTINUATION_CALLS
        or decimal(body.get("maximum_cost_per_call_usd"), "per-call cap")
        != MAX_COST_PER_CALL_USD
        or decimal(body.get("maximum_cost_usd"), "plan cap")
        != TOTAL_JUDGE_CAP_USD
        or body.get("canary_and_continuation_require_separate_authorizations")
        is not True
        or body.get("historical_A_reused_not_rejudged") is not True
        or body.get("abstentions_are_not_judged_or_reclassified") is not True
        or body.get("contains_question_or_response_text") is not False
        or body.get("external_api_calls") != 0
        or not isinstance(sources, list)
        or [item.get("name") for item in sources] != list(EXPECTED_MODELS)
        or not isinstance(rows, list)
        or len(rows) != TOTAL_CALLS
    ):
        raise ValueError("contextual judge plan contract differs")
    prompt_path = body.get("prompt_file_path")
    require_regular(prompt_path, "bound medical prompt bank")
    if sha256_file(prompt_path) != body.get("prompt_file_sha256"):
        raise ValueError("medical prompt-bank binding differs")
    specs = []
    for source_record in sources:
        if set(source_record) != {
            "name", "path", "file_sha256", "payload_sha256", "accounting"
        }:
            raise ValueError("source-generation binding schema differs")
        name = source_record["name"]
        path = source_record["path"]
        accounting = source_record["accounting"]
        require_regular(path, f"{name} source generation")
        if (
            sha256_file(path) != source_record["file_sha256"]
            or accounting != {
                "requested_n": 80, "accepted_n": 80, "abstained_n": 0
            }
        ):
            raise ValueError(f"{name} source generation differs")
        source_payload = load_json(path, f"{name} source generation")
        audit_seal(source_payload, f"{name} source generation")
        if source_payload[SEAL_FIELD] != source_record["payload_sha256"]:
            raise ValueError(f"{name} source generation payload differs")
        specs.append(f"{name}={path}")
    rebuilt = plan_source.build_plan(specs, prompt_path)
    if rebuilt != payload:
        raise ValueError("sealed plan does not round-trip from its bound sources")
    prompts = plan_source.load_prompts(prompt_path)
    source_rows = {}
    for source_record in sources:
        generation = plan_source.generation_rows(
            source_record["name"], source_record["path"]
        )
        observed_grid = {
            (sample.get("question_id"), sample.get("sample_index"))
            for sample in generation["rows"]
        }
        expected_grid = {
            (f"medical_official16_{prompt_index:02d}", sample_index)
            for prompt_index in range(16)
            for sample_index in range(5)
        }
        if observed_grid != expected_grid:
            raise ValueError(
                f"{source_record['name']} does not cover the exact 16x5 grid"
            )
        for sample in generation["rows"]:
            key = _source_identity(source_record["name"], sample)
            if key in source_rows:
                raise ValueError("source generation contains duplicate identity")
            source_rows[key] = sample
    runtime_rows = []
    seen_blind = set()
    exact_keys = {
        "blind_id", "model_name", "question_id", "sample_index",
        "prompt_sha256", "response_sha256", "source_sample_sha256", "plan_index",
    }
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != exact_keys
            or row.get("plan_index") != index
            or HEX64.fullmatch(str(row.get("blind_id", ""))) is None
            or row["blind_id"] in seen_blind
            or row.get("model_name") not in EXPECTED_MODELS
        ):
            raise ValueError("contextual judge plan row differs")
        key = (
            row["model_name"], row["question_id"], row["sample_index"],
            row["prompt_sha256"], row["response_sha256"],
            row["source_sample_sha256"],
        )
        sample = source_rows.get(key)
        prompt = prompts.get(row["question_id"])
        if sample is None or prompt is None or prompt["prompt_sha256"] != row["prompt_sha256"]:
            raise ValueError("plan row cannot be reconstructed from sealed sources")
        runtime_rows.append({
            **row,
            "question": prompt["prompt"],
            "response": sample["response"],
            "finish_reason": sample["finish_reason"],
        })
        seen_blind.add(row["blind_id"])
    if len(source_rows) != TOTAL_CALLS or len(runtime_rows) != TOTAL_CALLS:
        raise ValueError("reconstructed contextual plan cardinality differs")
    for row in runtime_rows:
        rendered = plan_source.RUBRIC.format(
            question=row["question"], response=row["response"]
        )
        if len(rendered.encode("utf-8")) + 64 > MAX_INPUT_TOKENS:
            raise ValueError("rendered contextual judge request exceeds input cap")
    conservative_per_call = (
        Decimal(MAX_INPUT_TOKENS) * INPUT_USD_PER_MILLION
        + Decimal(MAX_OUTPUT_TOKENS) * OUTPUT_USD_PER_MILLION
    ) / Decimal("1000000")
    if (
        conservative_per_call != MAX_COST_PER_CALL_USD
        or conservative_per_call * TOTAL_CALLS != TOTAL_JUDGE_CAP_USD
    ):
        raise ValueError("contextual judge conservative price contract differs")
    return {
        "path": absolute,
        "payload": payload,
        "body": body,
        "record": binding(absolute, payload),
        "rows": runtime_rows,
    }


def idempotency_key(row):
    blind_id = row.get("blind_id") if isinstance(row, dict) else None
    if HEX64.fullmatch(str(blind_id or "")) is None:
        raise ValueError("plan row lacks a valid blind identity")
    return digest(f"{WORKFLOW_ID}:{blind_id}".encode("utf-8"))


def idempotency_contract(rows):
    keys = [idempotency_key(row) for row in rows]
    indexed = [
        {"plan_index": row["plan_index"], "blind_id": row["blind_id"],
         "idempotency_key": keys[index]}
        for index, row in enumerate(rows)
    ]
    if len(set(keys)) != TOTAL_CALLS or set(keys) & {row["blind_id"] for row in rows}:
        raise ValueError("derived idempotency identities are not disjoint and unique")
    return {
        "version": "workflow_id_blind_id_sha256_v1",
        "derivation": "sha256(utf8(workflow_id + ':' + blind_id))",
        "workflow_id": WORKFLOW_ID,
        "canary_start_index": CANARY_START,
        "canary_end_index_exclusive": CANARY_END,
        "continuation_start_index": CONTINUATION_START,
        "continuation_end_index_exclusive": CONTINUATION_END,
        "row_count": TOTAL_CALLS,
        "all_keys_unique": True,
        "raw_keys_persisted": False,
        "raw_blind_ids_reused_as_keys": False,
        "derived_key_list_sha256": digest(canonical(keys)),
        "indexed_key_list_sha256": digest(canonical(indexed)),
        "canary_key_list_sha256": digest(canonical(keys[:1])),
        "continuation_key_list_sha256": digest(canonical(keys[1:])),
    }


def repository_record(repo_root):
    root = os.path.realpath(os.path.abspath(repo_root))
    commit = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", root, "branch", "--show-current"], text=True
    ).strip()
    return {"path": root, "commit": commit, "tree": tree, "branch": branch}


def workflow_paths(output_root):
    root = os.path.realpath(os.path.abspath(output_root))
    if not root.endswith("_contextual_baseline_judge_v1"):
        raise ValueError("judge output namespace suffix differs")
    control = os.path.join(root, "control")
    medical = os.path.join(root, "evaluation", "medical")
    logs = os.path.join(root, "logs")
    return {
        "root": root, "control": control, "medical": medical, "logs": logs,
        "manifest": os.path.join(control, "JUDGE_STAGE_MANIFEST.json"),
        "staged": os.path.join(control, "CPU_STAGED.json"),
        "canary_lock": os.path.join(control, "CANARY_LOCK.json"),
        "canary_authorization": os.path.join(control, "CANARY_AUTHORIZATION.json"),
        "canary_run_started": os.path.join(control, "CANARY_RUN_STARTED.json"),
        "canary_success": os.path.join(control, "CANARY_SUCCESS.json"),
        "canary_failure": os.path.join(control, "CANARY_FAILURE.json"),
        "continuation_lock": os.path.join(control, "CONTINUATION_LOCK.json"),
        "continuation_authorization": os.path.join(control, "CONTINUATION_AUTHORIZATION.json"),
        "continuation_run_started": os.path.join(
            control, "CONTINUATION_RUN_STARTED.json"
        ),
        "continuation_success": os.path.join(control, "CONTINUATION_SUCCESS.json"),
        "continuation_failure": os.path.join(control, "CONTINUATION_FAILURE.json"),
        "checkpoint_base": os.path.join(medical, "judge_checkpoint.json"),
        "judgments": os.path.join(medical, "judgments_contextual_baselines.json"),
    }


def checkpoint_path(paths, completed):
    return paths["checkpoint_base"] + f".{completed:03d}"


def manifest_body(plan, repo, output_root):
    if (
        KNOWN_PROGRAM_ACTUAL_USD + RETAINED_PRIOR_EXPOSURE_USD
        != CURRENT_CONSERVATIVE_EXPOSURE_USD
        or CURRENT_CONSERVATIVE_EXPOSURE_USD + TOTAL_JUDGE_CAP_USD
        != CONSERVATIVE_PROGRAM_MAX_USD
        or CONSERVATIVE_PROGRAM_MAX_USD > PROGRAM_CEILING_USD
        or CANARY_CAP_USD + CONTINUATION_CAP_USD != TOTAL_JUDGE_CAP_USD
    ):
        raise ValueError("contextual judge budget constants are inconsistent")
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_gate_eligible": False,
        "output_root": os.path.realpath(os.path.abspath(output_root)),
        "repository": repo,
        "judge_plan": plan["record"],
        "judge_plan_payload_sha256": EXPECTED_PLAN_PAYLOAD_SHA256,
        "judge_model": plan_source.JUDGE_MODEL,
        "sdk_max_retries": 0,
        "rubric_sha256": digest(plan_source.RUBRIC.encode("utf-8")),
        "response_schema_sha256": digest(canonical(plan_source.JUDGE_SCHEMA)),
        "planned_calls": TOTAL_CALLS,
        "canary": {"start": 0, "end_exclusive": 1, "calls": 1,
                   "cap_usd": float(CANARY_CAP_USD)},
        "continuation": {"start": 1, "end_exclusive": 160, "calls": 159,
                         "cap_usd": float(CONTINUATION_CAP_USD)},
        "total_judge_cap_usd": float(TOTAL_JUDGE_CAP_USD),
        "budget": {
            "known_program_actual_usd": float(KNOWN_PROGRAM_ACTUAL_USD),
            "retained_prior_unknown_or_conservative_exposure_usd":
            float(RETAINED_PRIOR_EXPOSURE_USD),
            "current_conservative_exposure_usd":
            float(CURRENT_CONSERVATIVE_EXPOSURE_USD),
            "conservative_program_max_with_full_plan_usd":
            float(CONSERVATIVE_PROGRAM_MAX_USD),
            "program_ceiling_usd": float(PROGRAM_CEILING_USD),
            "within_program_ceiling": True,
            "unused_terminal_authority_is_not_cost_exposure": True,
            "unused_terminal_authority_is_nonreusable": True,
        },
        "idempotency_contract": idempotency_contract(plan["rows"]),
        "separate_stage_authorizations_required": True,
        "permanent_stage_locks": True,
        "permanent_atomic_run_started_entry_seals": True,
        "restart_or_resume_authorized": False,
        "historical_A_reused_not_rejudged": True,
        "external_api_authorized": False,
        "external_api_calls": 0,
        "gpu_authorized": False,
        "gpu_jobs": 0,
    }


def load_manifest(path):
    absolute = os.path.abspath(path)
    payload = load_json(absolute, "judge stage manifest")
    body = audit_seal(payload, "judge stage manifest")
    paths = workflow_paths(body.get("output_root"))
    if absolute != paths["manifest"]:
        raise ValueError("judge stage manifest path differs")
    plan_payload = require_binding(body.get("judge_plan"), "manifest judge plan")
    plan = load_plan_context(body["judge_plan"]["path"])
    repo = repository_record(body["repository"]["path"])
    if body != manifest_body(plan, repo, paths["root"]):
        raise ValueError("judge stage manifest differs from live sealed inputs")
    if plan_payload != plan["payload"]:
        raise ValueError("manifest judge-plan payload differs")
    return {"path": absolute, "payload": payload, "body": body,
            "paths": paths, "plan": plan, "record": binding(absolute, payload)}


def prepare_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during CPU preparation")
    paths = workflow_paths(args.output_root)
    if os.path.lexists(paths["root"]):
        raise FileExistsError("fresh contextual judge output namespace already exists")
    plan = load_plan_context(args.judge_plan)
    repo = repository_record(args.repo_root)
    os.makedirs(paths["control"], mode=0o700)
    os.makedirs(paths["medical"], mode=0o700)
    os.makedirs(paths["logs"], mode=0o700)
    body = manifest_body(plan, repo, paths["root"])
    atomic_json(paths["manifest"], seal(body))
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_PREPARED_NO_AUTHORITY",
        "planned_calls": TOTAL_CALLS, "external_api_calls": 0, "gpu_jobs": 0,
        "manifest_payload_sha256": load_json(paths["manifest"])[SEAL_FIELD],
    }, sort_keys=True))
    return 0


def validate_plan_command(args):
    manifest = load_manifest(args.manifest)
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_PLAN_VALIDATED",
        "planned_calls": TOTAL_CALLS,
        "plan_payload_sha256": manifest["plan"]["payload"][SEAL_FIELD],
        "external_api_calls": 0,
    }, sort_keys=True))
    return 0


def request_body(row):
    return {
        "model": plan_source.JUDGE_MODEL,
        "messages": [{
            "role": "user",
            "content": plan_source.RUBRIC.format(
                question=row["question"], response=row["response"]
            ),
        }],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_effort": "minimal",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "medical_component_judgment",
                "strict": True,
                "schema": plan_source.JUDGE_SCHEMA,
            },
        },
    }


def validate_call_scope(stage, index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("judge call index schema differs")
    valid = index == 0 if stage == "canary" else stage == "continuation" and 1 <= index < 160
    if not valid:
        raise ValueError("judge call falls outside its authorized range")


def call_judge(client, row, stage, index):
    validate_call_scope(stage, index)
    return client.chat.completions.create(
        **request_body(row),
        extra_headers={"Idempotency-Key": idempotency_key(row)},
    )


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {})()


def sdk_serialization_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent during offline serialization")
    manifest = load_manifest(args.manifest)
    completions = _FakeCompletions()
    client = type("Client", (), {})()
    client.chat = type("Chat", (), {})()
    client.chat.completions = completions
    cases = (("canary", 0), ("continuation", 1), ("continuation", 159))
    for stage, index in cases:
        call_judge(client, manifest["plan"]["rows"][index], stage, index)
    if len(completions.calls) != 3:
        raise ValueError("offline request inventory differs")
    for call, (stage, index) in zip(completions.calls, cases):
        row = manifest["plan"]["rows"][index]
        headers = call.pop("extra_headers")
        if (
            headers != {"Idempotency-Key": idempotency_key(row)}
            or headers["Idempotency-Key"] == row["blind_id"]
            or call != request_body(row)
        ):
            raise ValueError(f"offline {stage} SDK serialization differs")
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_OFFLINE_SERIALIZATION_VALID",
        "fake_client_calls": 3, "external_api_calls": 0,
    }, sort_keys=True))
    return 0


def staged_body(manifest, validations):
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "manifest": manifest["record"],
        "validation_commands": list(validations),
        "status": "CPU_STAGED_NO_API_OR_GPU_AUTHORITY",
        "planned_calls": TOTAL_CALLS,
        "external_api_authorized": False,
        "external_api_calls": 0,
        "gpu_authorized": False,
        "gpu_jobs": 0,
        "restart_or_resume_authorized": False,
    }


def seal_staged_command(args):
    if os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be absent while sealing CPU stage")
    manifest = load_manifest(args.manifest)
    paths = manifest["paths"]
    if os.path.lexists(paths["staged"]):
        raise FileExistsError("CPU-stage seal already exists")
    if any(
        os.path.lexists(paths[name])
        for name in (
            "canary_lock", "canary_authorization", "canary_run_started",
            "canary_success", "canary_failure",
            "continuation_lock", "continuation_authorization",
            "continuation_run_started", "continuation_success",
            "continuation_failure", "judgments",
        )
    ):
        raise ValueError("external-stage artifact exists before CPU-stage seal")
    atomic_json(paths["staged"], seal(staged_body(manifest, args.validation_command)))
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_CPU_STAGED",
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def audit_staged(manifest):
    payload = load_json(manifest["paths"]["staged"], "CPU-stage seal")
    body = audit_seal(payload, "CPU-stage seal")
    validations = body.get("validation_commands")
    if not isinstance(validations, list) or not validations or any(
        not isinstance(item, str) or not item for item in validations
    ):
        raise ValueError("CPU-stage validation inventory differs")
    if body != staged_body(manifest, validations):
        raise ValueError("CPU-stage seal differs")
    return {"payload": payload, "body": body,
            "record": binding(manifest["paths"]["staged"], payload)}


def audit_staged_command(args):
    manifest = load_manifest(args.manifest)
    staged = audit_staged(manifest)
    paths = manifest["paths"]
    if any(
        os.path.lexists(paths[name])
        for name in (
            "canary_lock", "canary_authorization", "canary_run_started",
            "canary_success", "canary_failure", "continuation_lock",
            "continuation_authorization", "continuation_run_started",
            "continuation_success", "continuation_failure", "judgments",
        )
    ) or _completed_count(paths) != 0:
        raise ValueError("CPU-stage audit found external-stage state")
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_CPU_STAGE_AUDITED",
        "staged_payload_sha256": staged["payload"][SEAL_FIELD],
        "external_api_calls": 0, "gpu_jobs": 0,
    }, sort_keys=True))
    return 0


def stage_values(stage):
    if stage == "canary":
        return 0, 1, 1, CANARY_CAP_USD
    if stage == "continuation":
        return 1, 160, 159, CONTINUATION_CAP_USD
    raise ValueError("unknown external judge stage")


def lock_body(manifest, stage, owner_token):
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "stage": stage,
        "manifest": manifest["record"],
        "owner_token_sha256": digest(owner_token.encode("utf-8")),
        "permanent_single_entry": True,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def audit_lock(manifest, stage, owner_token=None):
    path = manifest["paths"][f"{stage}_lock"]
    payload = load_json(path, f"{stage} permanent lock")
    body = audit_seal(payload, f"{stage} permanent lock")
    recorded_hash = body.get("owner_token_sha256")
    if HEX64.fullmatch(str(recorded_hash or "")) is None:
        raise ValueError(f"{stage} lock owner hash differs")
    if owner_token is not None:
        if HEX64.fullmatch(owner_token or "") is None:
            raise ValueError("owner token schema differs")
        if digest(owner_token.encode("utf-8")) != recorded_hash:
            raise ValueError("external judge lock is owned by another invocation")
    expected = lock_body(manifest, stage, owner_token or ("0" * 64))
    expected["owner_token_sha256"] = recorded_hash
    if body != expected or stat.S_IMODE(os.stat(path).st_mode) != 0o400:
        raise ValueError(f"{stage} permanent lock differs")
    return {"payload": payload, "body": body, "record": binding(path, payload)}


def checkpoint_body(manifest, stage, authorization_record, completed, judgments):
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "analysis_scope": "contextual_post_hoc_not_gated",
        "primary_gate_eligible": False,
        "judge_plan": manifest["body"]["judge_plan"],
        "stage": stage,
        "stage_authorization": authorization_record,
        "completed_calls": completed,
        "last_blind_id": judgments[-1]["blind_id"],
        "judgments": judgments,
    }


def audit_judgment(row, plan_row):
    identity = (
        "blind_id", "plan_index", "model_name", "question_id", "sample_index",
        "prompt_sha256", "response_sha256", "source_sample_sha256",
    )
    usage = row.get("api_usage") if isinstance(row, dict) else None
    if (
        not isinstance(row, dict)
        or any(row.get(key) != plan_row.get(key) for key in identity)
        or row.get("source_finish_reason") != "stop"
        or row.get("label") not in {"BAD", "SAFE", "REFUSAL", "UNPARSEABLE"}
        or isinstance(row.get("coherence"), bool)
        or not isinstance(row.get("coherence"), int)
        or not 0 <= row["coherence"] <= 100
        or not isinstance(row.get("judge_parse_valid"), bool)
        or (not row["judge_parse_valid"] and row["label"] != "UNPARSEABLE")
        or row.get("judge_finish_reason") != "stop"
        or HEX64.fullmatch(str(row.get("judge_output_sha256", ""))) is None
        or not isinstance(row.get("api_response_id"), str)
        or not row["api_response_id"]
        or row.get("api_response_model") != plan_source.JUDGE_MODEL
        or not isinstance(usage, dict)
        or set(usage) != {
            "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"
        }
    ):
        raise ValueError("sealed external judgment differs")
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    expected_cost = (
        Decimal(input_tokens) * INPUT_USD_PER_MILLION
        + Decimal(output_tokens) * OUTPUT_USD_PER_MILLION
    ) / Decimal("1000000")
    if (
        isinstance(input_tokens, bool) or not isinstance(input_tokens, int)
        or isinstance(output_tokens, bool) or not isinstance(output_tokens, int)
        or not 0 < input_tokens <= MAX_INPUT_TOKENS
        or not 0 < output_tokens <= MAX_OUTPUT_TOKENS
        or usage["total_tokens"] != input_tokens + output_tokens
        or decimal(usage["estimated_cost_usd"], "judgment cost") != expected_cost
        or expected_cost > MAX_COST_PER_CALL_USD
    ):
        raise ValueError("external judgment usage differs")


def audit_checkpoint(manifest, stage, authorization, completed):
    path = checkpoint_path(manifest["paths"], completed)
    payload = load_json(path, f"checkpoint {completed}")
    body = audit_seal(payload, f"checkpoint {completed}")
    judgments = body.get("judgments")
    expected = checkpoint_body(
        manifest, stage, authorization["record"], completed, judgments
    ) if isinstance(judgments, list) and judgments else None
    if body != expected or len(judgments) != completed:
        raise ValueError(f"checkpoint {completed} contract differs")
    response_ids = set()
    for index, judgment in enumerate(judgments):
        audit_judgment(judgment, manifest["plan"]["rows"][index])
        if judgment["api_response_id"] in response_ids:
            raise ValueError("external judge response ID was reused")
        response_ids.add(judgment["api_response_id"])
    return {"payload": payload, "body": body, "record": binding(path, payload)}


def success_body(manifest, stage, authorization, checkpoint, stage_calls,
                 stage_cost, total_cost):
    start, end, calls, _ = stage_values(stage)
    run_started = audit_run_started(
        manifest, stage, authorization=authorization
    )
    body = {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "stage": stage,
        "manifest": manifest["record"],
        "stage_authorization": authorization["record"],
        "run_started": run_started["record"],
        "terminal_checkpoint": checkpoint["record"],
        "authorized_start_index": start,
        "authorized_end_index_exclusive": end,
        "stage_api_call_invocations_exact": stage_calls,
        "stage_accepted_judgments": calls,
        "completed_calls": end,
        "stage_actual_estimated_cost_usd": float(stage_cost),
        "cumulative_accepted_estimated_cost_usd": float(total_cost),
        "completed_at": utc_now(),
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }
    if stage == "canary":
        body["continuation_api_authorized"] = False
    else:
        body["judgments"] = binding(manifest["paths"]["judgments"])
    return body


def load_success(manifest, stage, authorization=None):
    path = manifest["paths"][f"{stage}_success"]
    payload = load_json(path, f"{stage} success")
    body = audit_seal(payload, f"{stage} success")
    auth = load_authorization(manifest, stage) if authorization is None else authorization
    completed = 1 if stage == "canary" else 160
    checkpoint = audit_checkpoint(manifest, stage, auth, completed)
    timestamp = body.get("completed_at")
    expected = success_body(
        manifest, stage, auth, checkpoint,
        body.get("stage_api_call_invocations_exact"),
        decimal(body.get("stage_actual_estimated_cost_usd"), "stage actual cost"),
        decimal(body.get("cumulative_accepted_estimated_cost_usd"), "cumulative cost"),
    )
    expected["completed_at"] = timestamp
    _, end, calls, cap = stage_values(stage)
    stage_cost = decimal(body.get("stage_actual_estimated_cost_usd"), "stage cost")
    cumulative_cost = decimal(
        body.get("cumulative_accepted_estimated_cost_usd"), "cumulative cost"
    )
    try:
        parsed_timestamp = dt.datetime.fromisoformat(timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{stage} success timestamp differs") from error
    if (
        body != expected
        or body.get("stage_api_call_invocations_exact") != calls
        or body.get("completed_calls") != end
        or stage_cost > cap
        or cumulative_cost < stage_cost
        or parsed_timestamp.tzinfo is None
    ):
        raise ValueError(f"{stage} success differs")
    return {"payload": payload, "body": body, "record": binding(path, payload),
            "authorization": auth, "checkpoint": checkpoint}


def authorization_body(manifest, stage, lock, canary=None):
    start, end, calls, cap = stage_values(stage)
    body = {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "manifest": manifest["record"],
        "judge_plan": manifest["body"]["judge_plan"],
        "stage_lock": lock["record"],
        "authorized_start_index": start,
        "authorized_end_index_exclusive": end,
        "authorized_calls": calls,
        "judge_model": plan_source.JUDGE_MODEL,
        "sdk_max_retries": 0,
        "stage_cap_usd": float(cap),
        "total_planned_judge_cap_usd": float(TOTAL_JUDGE_CAP_USD),
        "budget_acknowledgment": manifest["body"]["budget"],
        "external_api_authorized": True,
        "permanent_single_entry": True,
        "restart_or_resume_authorized": False,
        "historical_A_reused_not_rejudged": True,
    }
    if stage == "continuation":
        body["canary_success"] = canary["record"]
        body["canary_checkpoint"] = canary["checkpoint"]["record"]
        body["canary_actual_estimated_cost_usd"] = canary["body"][
            "stage_actual_estimated_cost_usd"
        ]
    return body


def load_authorization(manifest, stage):
    lock = audit_lock(manifest, stage)
    path = manifest["paths"][f"{stage}_authorization"]
    payload = load_json(path, f"{stage} authorization")
    body = audit_seal(payload, f"{stage} authorization")
    canary = load_success(manifest, "canary") if stage == "continuation" else None
    if body != authorization_body(manifest, stage, lock, canary):
        raise ValueError(f"{stage} authorization differs")
    return {"payload": payload, "body": body, "record": binding(path, payload)}


def run_started_body(manifest, stage, authorization, owner_token_sha256,
                     started_at):
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "manifest": manifest["record"],
        "stage_authorization": authorization["record"],
        "owner_token_sha256": owner_token_sha256,
        "started_at": started_at,
        "permanent_atomic_single_run_entry": True,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def audit_run_started(manifest, stage, authorization=None, owner_token=None):
    auth = (
        load_authorization(manifest, stage)
        if authorization is None
        else authorization
    )
    lock = audit_lock(manifest, stage)
    path = manifest["paths"][f"{stage}_run_started"]
    payload = load_json(path, f"{stage} permanent run-start entry")
    body = audit_seal(payload, f"{stage} permanent run-start entry")
    owner_hash = body.get("owner_token_sha256")
    started_at = body.get("started_at")
    try:
        parsed_started_at = dt.datetime.fromisoformat(started_at)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{stage} run-start timestamp differs") from error
    if owner_token is not None:
        if HEX64.fullmatch(owner_token or "") is None:
            raise ValueError("owner token schema differs")
        if digest(owner_token.encode("utf-8")) != owner_hash:
            raise ValueError("external judge run entry is owned by another invocation")
    if (
        HEX64.fullmatch(str(owner_hash or "")) is None
        or owner_hash != lock["body"]["owner_token_sha256"]
        or parsed_started_at.tzinfo is None
        or body != run_started_body(
            manifest, stage, auth, owner_hash, started_at
        )
        or stat.S_IMODE(os.stat(path).st_mode) != 0o400
    ):
        raise ValueError(f"{stage} permanent run-start entry differs")
    return {
        "payload": payload,
        "body": body,
        "record": binding(path, payload),
        "authorization": auth,
    }


def create_run_started(manifest, stage, authorization, owner_token):
    if HEX64.fullmatch(owner_token or "") is None:
        raise ValueError("owner token schema differs")
    started_at = utc_now()
    payload = seal(
        run_started_body(
            manifest,
            stage,
            authorization,
            digest(owner_token.encode("utf-8")),
            started_at,
        )
    )
    # atomic_json uses a hard-link no-overwrite commit.  This is the actual
    # per-stage execution mutex: an authorization owner may enter exactly once.
    atomic_json(manifest["paths"][f"{stage}_run_started"], payload)
    return audit_run_started(
        manifest, stage, authorization=authorization, owner_token=owner_token
    )


def _acknowledgments(args, stage, canary=None):
    _, _, calls, cap = stage_values(stage)
    expected = {
        "ack_calls": calls,
        "ack_max_cost_usd": cap,
        "ack_total_judge_cap_usd": TOTAL_JUDGE_CAP_USD,
        "ack_known_program_actual_usd": KNOWN_PROGRAM_ACTUAL_USD,
        "ack_retained_prior_exposure_usd": RETAINED_PRIOR_EXPOSURE_USD,
        "ack_current_conservative_exposure_usd": CURRENT_CONSERVATIVE_EXPOSURE_USD,
        "ack_conservative_program_max_usd": CONSERVATIVE_PROGRAM_MAX_USD,
        "ack_program_ceiling_usd": PROGRAM_CEILING_USD,
    }
    for name, value in expected.items():
        observed = getattr(args, name)
        if name == "ack_calls":
            if observed != value:
                raise ValueError(f"authorization acknowledgment differs: {name}")
        elif decimal(observed, name) != value:
            raise ValueError(f"authorization acknowledgment differs: {name}")
    if not (
        args.ack_sdk_retries_zero
        and args.ack_no_restart_or_resume
        and args.ack_contextual_post_hoc_only
        and args.ack_unused_terminal_authority_nonreusable
        and args.ack_unused_terminal_authority_not_cost_exposure
    ):
        raise ValueError("required external-stage policy acknowledgment is absent")
    if stage == "continuation":
        actual = decimal(args.ack_canary_actual_cost_usd, "canary actual acknowledgment")
        expected_actual = decimal(
            canary["body"]["stage_actual_estimated_cost_usd"], "sealed canary actual"
        )
        if actual != expected_actual:
            raise ValueError("canary actual-cost acknowledgment differs")


def require_fresh_stage(manifest, stage):
    paths = manifest["paths"]
    if stage == "canary":
        forbidden = (
            "canary_lock", "canary_authorization", "canary_run_started",
            "canary_success", "canary_failure", "continuation_lock",
            "continuation_authorization", "continuation_run_started",
            "continuation_success", "continuation_failure", "judgments",
        )
        if any(os.path.lexists(paths[name]) for name in forbidden) or any(
            os.path.lexists(checkpoint_path(paths, index))
            for index in range(1, TOTAL_CALLS + 1)
        ):
            raise ValueError("canary namespace is not fresh")
    else:
        if os.path.lexists(paths["canary_failure"]):
            raise ValueError("failed canary cannot authorize continuation")
        load_success(manifest, "canary")
        forbidden = (
            "continuation_lock", "continuation_authorization",
            "continuation_run_started", "continuation_success",
            "continuation_failure", "judgments",
        )
        if any(os.path.lexists(paths[name]) for name in forbidden) or any(
            os.path.lexists(checkpoint_path(paths, index))
            for index in range(2, TOTAL_CALLS + 1)
        ):
            raise ValueError("continuation namespace is not fresh")


def authorize_command(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be loaded before authorization")
    if HEX64.fullmatch(args.owner_token or "") is None:
        raise ValueError("owner token must be exactly 64 lowercase hex characters")
    manifest = load_manifest(args.manifest)
    audit_staged(manifest)
    stage = args.stage
    canary = load_success(manifest, "canary") if stage == "continuation" else None
    _acknowledgments(args, stage, canary)
    require_fresh_stage(manifest, stage)
    lock_path = manifest["paths"][f"{stage}_lock"]
    atomic_json(lock_path, seal(lock_body(manifest, stage, args.owner_token)))
    lock = audit_lock(manifest, stage, args.owner_token)
    authorization = seal(authorization_body(manifest, stage, lock, canary))
    atomic_json(manifest["paths"][f"{stage}_authorization"], authorization)
    load_authorization(manifest, stage)
    print(json.dumps({
        "status": f"CONTEXTUAL_BASELINE_JUDGE_{stage.upper()}_AUTHORIZED",
        "authorized_calls": stage_values(stage)[2],
        "max_cost_usd": float(stage_values(stage)[3]),
        "external_api_calls": 0,
        "authorization_payload_sha256": authorization[SEAL_FIELD],
    }, sort_keys=True))
    return 0


def parse_judgment(content):
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"label", "coherence"}
        or parsed.get("label") not in {"BAD", "SAFE", "REFUSAL", "UNPARSEABLE"}
        or isinstance(parsed.get("coherence"), bool)
        or not isinstance(parsed.get("coherence"), int)
        or not 0 <= parsed["coherence"] <= 100
    ):
        return {"label": "UNPARSEABLE", "coherence": 0,
                "judge_parse_valid": False}
    return {**parsed, "judge_parse_valid": True}


def extract_usage(response):
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if (
        isinstance(input_tokens, bool) or not isinstance(input_tokens, int)
        or isinstance(output_tokens, bool) or not isinstance(output_tokens, int)
        or not 0 < input_tokens <= MAX_INPUT_TOKENS
        or not 0 < output_tokens <= MAX_OUTPUT_TOKENS
        or total_tokens != input_tokens + output_tokens
    ):
        raise RuntimeError("judge response token usage differs")
    cost = (
        Decimal(input_tokens) * INPUT_USD_PER_MILLION
        + Decimal(output_tokens) * OUTPUT_USD_PER_MILLION
    ) / Decimal("1000000")
    if cost > MAX_COST_PER_CALL_USD:
        raise RuntimeError("judge response exceeds per-call cost cap")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": float(cost),
    }


def validate_response(response, row):
    if getattr(response, "model", None) != plan_source.JUDGE_MODEL:
        raise RuntimeError("judge resolved-model identity drift")
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("judge response choice inventory differs")
    choice = choices[0]
    content = getattr(getattr(choice, "message", None), "content", None) or ""
    response_id = getattr(response, "id", None)
    if not isinstance(response_id, str) or not response_id:
        raise RuntimeError("judge response lacks an API response ID")
    return {
        **{key: row[key] for key in (
            "blind_id", "plan_index", "model_name", "question_id", "sample_index",
            "prompt_sha256", "response_sha256", "source_sample_sha256",
        )},
        "source_finish_reason": row["finish_reason"],
        **parse_judgment(content),
        "judge_finish_reason": getattr(choice, "finish_reason", None),
        "judge_output_sha256": digest(content.encode("utf-8")),
        "api_response_id": response_id,
        "api_response_model": getattr(response, "model", None),
        "api_usage": extract_usage(response),
    }


def _make_client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key, max_retries=0)


def verify_owner(manifest, stage, owner_token):
    audit_lock(manifest, stage, owner_token)


def _call_and_validate(client, manifest, authorization, stage, index, attempts):
    validate_call_scope(stage, index)
    verify_owner(manifest, stage, attempts["owner_token"])
    attempts["last_index"] = index
    attempts["count"] += 1
    try:
        response = call_judge(client, manifest["plan"]["rows"][index], stage, index)
    except Exception as error:
        raise JudgeCallFailure("api_call", error) from None
    try:
        judgment = validate_response(response, manifest["plan"]["rows"][index])
        audit_judgment(judgment, manifest["plan"]["rows"][index])
    except Exception as error:
        raise JudgeCallFailure("response_validation", error, response) from None
    return response, judgment


def _cost(rows):
    return sum(
        (decimal(row["api_usage"]["estimated_cost_usd"], "row cost") for row in rows),
        Decimal("0"),
    )


def _write_checkpoint(manifest, stage, authorization, judgments):
    completed = len(judgments)
    payload = seal(checkpoint_body(
        manifest, stage, authorization["record"], completed, judgments
    ))
    try:
        atomic_json(checkpoint_path(manifest["paths"], completed), payload)
        return audit_checkpoint(manifest, stage, authorization, completed)
    except Exception as error:
        raise JudgeCallFailure("artifact_commit", error) from None


def run_canary(manifest, authorization, client, attempts):
    _, judgment = _call_and_validate(
        client, manifest, authorization, "canary", 0, attempts
    )
    judgments = [judgment]
    if _cost(judgments) > CANARY_CAP_USD:
        raise JudgeCallFailure("response_validation", RuntimeError("canary cap exceeded"))
    checkpoint = _write_checkpoint(manifest, "canary", authorization, judgments)
    success = seal(success_body(
        manifest, "canary", authorization, checkpoint, attempts["count"],
        _cost(judgments), _cost(judgments),
    ))
    try:
        atomic_json(manifest["paths"]["canary_success"], success)
    except Exception as error:
        raise JudgeCallFailure("artifact_commit", error) from None
    return success


def run_continuation(manifest, authorization, client, attempts):
    canary = load_success(manifest, "canary")
    judgments = list(canary["checkpoint"]["body"]["judgments"])
    stage_cost = Decimal("0")
    for index in range(CONTINUATION_START, CONTINUATION_END):
        _, judgment = _call_and_validate(
            client, manifest, authorization, "continuation", index, attempts
        )
        if judgment["api_response_id"] in {
            item["api_response_id"] for item in judgments
        }:
            raise JudgeCallFailure(
                "response_validation", RuntimeError("judge response ID was reused")
            )
        stage_cost += decimal(judgment["api_usage"]["estimated_cost_usd"], "row cost")
        if stage_cost > CONTINUATION_CAP_USD:
            raise JudgeCallFailure(
                "response_validation", RuntimeError("continuation cap exceeded")
            )
        judgments.append(judgment)
        _write_checkpoint(manifest, "continuation", authorization, judgments)
    final_body = {
        "meta": {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "protocol_id": PROTOCOL_ID,
            "analysis_scope": "contextual_post_hoc_not_gated",
            "primary_gate_eligible": False,
            "judge_model": plan_source.JUDGE_MODEL,
            "sdk_retries": 0,
            "judge_plan": manifest["body"]["judge_plan"],
            "historical_A_reused_not_rejudged": True,
            "actual_api_calls": TOTAL_CALLS,
            "canary_api_calls": 1,
            "continuation_api_calls": 159,
            "actual_estimated_cost_usd": float(_cost(judgments)),
            "restart_or_resume_used": False,
        },
        "completed_calls": TOTAL_CALLS,
        "judgments": judgments,
    }
    final = seal(final_body)
    try:
        atomic_json(manifest["paths"]["judgments"], final)
    except Exception as error:
        raise JudgeCallFailure("artifact_commit", error) from None
    checkpoint = audit_checkpoint(manifest, "continuation", authorization, TOTAL_CALLS)
    success = seal(success_body(
        manifest, "continuation", authorization, checkpoint, attempts["count"],
        stage_cost, _cost(judgments),
    ))
    try:
        atomic_json(manifest["paths"]["continuation_success"], success)
    except Exception as error:
        raise JudgeCallFailure("artifact_commit", error) from None
    return success


def _completed_count(paths):
    observed = []
    if not os.path.isdir(paths["medical"]) or os.path.islink(paths["medical"]):
        raise ValueError("medical output directory is absent or unsafe")
    for name in os.listdir(paths["medical"]):
        if name == os.path.basename(paths["judgments"]):
            continue
        match = re.fullmatch(r"judge_checkpoint\.json\.(\d{3})", name)
        if match is None:
            raise ValueError("medical output inventory contains an extra file")
        index = int(match.group(1))
        if not 1 <= index <= TOTAL_CALLS:
            raise ValueError("checkpoint index is out of range")
        observed.append(index)
    observed.sort()
    if observed != list(range(1, len(observed) + 1)):
        raise ValueError("checkpoint sequence contains a gap")
    return observed[-1] if observed else 0


def _safe_token(value):
    if value is None:
        return None
    text = str(value)
    return text if SAFE_EXTERNAL_TOKEN.fullmatch(text) else None


def failure_body(manifest, stage, authorization, attempts, operation, error,
                 started_at):
    run_started = audit_run_started(
        manifest,
        stage,
        authorization=authorization,
        owner_token=attempts["owner_token"],
    )
    completed = _completed_count(manifest["paths"])
    stage_prior = 0 if stage == "canary" else 1
    accepted_stage = max(0, completed - stage_prior)
    maximum = CANARY_CALLS if stage == "canary" else CONTINUATION_CALLS
    if not accepted_stage <= attempts["count"] <= min(maximum, accepted_stage + 1):
        raise RuntimeError("exact SDK invocation accounting differs")
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        status_code = None
    return {
        "schema_version": 1,
        "workflow_id": WORKFLOW_ID,
        "stage": stage,
        "manifest": manifest["record"],
        "stage_authorization": authorization["record"],
        "run_started": run_started["record"],
        "operation_stage": operation,
        "planned_index": (
            attempts.get("last_index")
            if attempts.get("last_index") is not None
            else (completed if completed < TOTAL_CALLS else None)
        ),
        "previously_completed_calls": completed,
        "stage_sdk_call_invocations_exact": attempts["count"],
        "stage_accepted_judgments": accepted_stage,
        "attempt_started_at": started_at,
        "failure_recorded_at": utc_now(),
        "exception_class": type(error).__name__,
        "http_status": status_code,
        "error_code": _safe_token(getattr(error, "code", None)),
        "request_id": _safe_token(getattr(error, "request_id", None)),
        "error_message_persisted": False,
        "contains_question_or_response_text": False,
        "contains_api_key_or_headers": False,
        "retry_authorized": False,
        "restart_or_resume_authorized": False,
    }


def _install_signal_handlers():
    previous = {}
    def handler(signum, _frame):
        raise StageInterrupted(f"signal-{signum}")
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_external_command(args):
    stage = args.stage
    manifest = load_manifest(args.manifest)
    audit_staged(manifest)
    authorization = load_authorization(manifest, stage)
    verify_owner(manifest, stage, args.owner_token)
    paths = manifest["paths"]
    if os.path.lexists(paths[f"{stage}_success"]) or os.path.lexists(paths[f"{stage}_failure"]):
        raise ValueError("external stage already has a terminal artifact")
    if stage == "canary":
        if _completed_count(paths) != 0 or os.path.lexists(paths["judgments"]):
            raise ValueError("canary refuses restart or resume")
    else:
        load_success(manifest, "canary")
        if _completed_count(paths) != 1 or os.path.lexists(paths["judgments"]):
            raise ValueError("continuation refuses restart or resume")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is absent")
    attempts = {"count": 0, "last_index": None, "owner_token": args.owner_token}
    try:
        run_started = create_run_started(
            manifest, stage, authorization, args.owner_token
        )
    except BaseException:
        # A contender that loses the atomic entry race must not create a
        # terminal failure artifact for the winning invocation.
        os.environ.pop("OPENAI_API_KEY", None)
        raise
    started_at = run_started["body"]["started_at"]
    operation = "signal_handler_installation"
    previous_handlers = {}
    try:
        previous_handlers = _install_signal_handlers()
        operation = "client_initialization"
        client = _make_client(api_key)
        operation = "api_call"
        result = (
            run_canary(manifest, authorization, client, attempts)
            if stage == "canary"
            else run_continuation(manifest, authorization, client, attempts)
        )
        print(json.dumps({
            "status": f"CONTEXTUAL_BASELINE_JUDGE_{stage.upper()}_COMPLETE",
            "stage_sdk_call_invocations_exact": attempts["count"],
            "result_payload_sha256": result[SEAL_FIELD],
        }, sort_keys=True))
        return 0
    except (Exception, KeyboardInterrupt) as caught:
        original = caught
        if isinstance(caught, JudgeCallFailure):
            operation = caught.operation
            original = caught.original
        failure = seal(failure_body(
            manifest, stage, authorization, attempts, operation, original, started_at
        ))
        atomic_json(paths[f"{stage}_failure"], failure)
        raise RuntimeError(
            f"{stage} failed terminally; see sealed sanitized failure artifact"
        ) from None
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        _restore_signal_handlers(previous_handlers)


def audit_canary(manifest):
    paths = manifest["paths"]
    if os.path.lexists(paths["canary_failure"]):
        raise ValueError("canary has a terminal failure")
    success = load_success(manifest, "canary")
    completed = _completed_count(paths)
    continuation_started = os.path.lexists(paths["continuation_lock"])
    if (
        success["body"]["stage_api_call_invocations_exact"] != 1
        or success["body"]["completed_calls"] != 1
        or completed < 1
        or (not continuation_started and completed != 1)
        or (not continuation_started and os.path.lexists(paths["judgments"]))
    ):
        raise ValueError("canary terminal accounting differs")
    return success


def audit_canary_command(args):
    manifest = load_manifest(args.manifest)
    success = audit_canary(manifest)
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_CANARY_AUDITED",
        "completed_calls": 1,
        "actual_estimated_cost_usd": success["body"]["stage_actual_estimated_cost_usd"],
    }, sort_keys=True))
    return 0


def audit_continuation(manifest):
    paths = manifest["paths"]
    if os.path.lexists(paths["continuation_failure"]):
        raise ValueError("continuation has a terminal failure")
    audit_canary(manifest)
    authorization = load_authorization(manifest, "continuation")
    previous = audit_checkpoint(manifest, "canary", load_authorization(manifest, "canary"), 1)
    for completed in range(2, TOTAL_CALLS + 1):
        checkpoint = audit_checkpoint(
            manifest, "continuation", authorization, completed
        )
        if checkpoint["body"]["judgments"][:-1] != previous["body"]["judgments"]:
            raise ValueError("cumulative checkpoint prefix differs")
        previous = checkpoint
    success = load_success(manifest, "continuation", authorization)
    final = load_json(paths["judgments"], "terminal contextual judgments")
    final_body = audit_seal(final, "terminal contextual judgments")
    if (
        final_body.get("completed_calls") != TOTAL_CALLS
        or final_body.get("judgments") != previous["body"]["judgments"]
        or success["body"]["stage_api_call_invocations_exact"] != CONTINUATION_CALLS
        or _completed_count(paths) != TOTAL_CALLS
    ):
        raise ValueError("continuation terminal accounting differs")
    return {"success": success, "judgments": binding(paths["judgments"], final)}


def audit_continuation_command(args):
    manifest = load_manifest(args.manifest)
    result = audit_continuation(manifest)
    print(json.dumps({
        "status": "CONTEXTUAL_BASELINE_JUDGE_CONTINUATION_AUDITED",
        "completed_calls": TOTAL_CALLS,
        "judgments_payload_sha256": result["judgments"][SEAL_FIELD],
    }, sort_keys=True))
    return 0


def audit_failure(manifest, stage):
    audit_lock(manifest, stage)
    authorization = load_authorization(manifest, stage)
    run_started = audit_run_started(
        manifest, stage, authorization=authorization
    )
    path = manifest["paths"][f"{stage}_failure"]
    payload = load_json(path, f"{stage} failure")
    body = audit_seal(payload, f"{stage} failure")
    exact = body.get("stage_sdk_call_invocations_exact")
    completed = _completed_count(manifest["paths"])
    stage_prior = 0 if stage == "canary" else 1
    accepted = max(0, completed - stage_prior)
    maximum = CANARY_CALLS if stage == "canary" else CONTINUATION_CALLS
    if (
        body.get("workflow_id") != WORKFLOW_ID
        or body.get("stage") != stage
        or body.get("manifest") != manifest["record"]
        or body.get("stage_authorization") != authorization["record"]
        or body.get("run_started") != run_started["record"]
        or body.get("operation_stage") not in {
            "signal_handler_installation", "client_initialization", "api_call",
            "response_validation", "artifact_commit",
        }
        or body.get("previously_completed_calls") != completed
        or body.get("stage_accepted_judgments") != accepted
        or isinstance(exact, bool) or not isinstance(exact, int)
        or not accepted <= exact <= min(maximum, accepted + 1)
        or body.get("error_message_persisted") is not False
        or body.get("contains_question_or_response_text") is not False
        or body.get("contains_api_key_or_headers") is not False
        or body.get("retry_authorized") is not False
        or body.get("restart_or_resume_authorized") is not False
    ):
        raise ValueError(f"{stage} failure accounting differs")
    return {
        "payload": payload,
        "body": body,
        "record": binding(path, payload),
        "run_started": run_started,
    }


def audit_failure_command(args):
    manifest = load_manifest(args.manifest)
    stage = args.stage
    failure = audit_failure(manifest, stage)
    print(json.dumps({
        "status": f"CONTEXTUAL_BASELINE_JUDGE_{stage.upper()}_TERMINAL_FAILURE_AUDITED",
        "stage_sdk_call_invocations_exact": failure["body"][
            "stage_sdk_call_invocations_exact"
        ],
        "completed_calls": failure["body"]["previously_completed_calls"],
    }, sort_keys=True))
    return 0


def status_command(args):
    manifest = load_manifest(args.manifest)
    paths = manifest["paths"]
    if os.path.lexists(paths["continuation_failure"]):
        audit_failure(manifest, "continuation")
        state = "CONTINUATION_TERMINAL_FAILURE_NO_RESTART"
    elif os.path.lexists(paths["continuation_success"]):
        audit_continuation(manifest)
        state = "COMPLETE"
    elif os.path.lexists(paths["continuation_run_started"]):
        audit_run_started(manifest, "continuation")
        state = "CONTINUATION_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
    elif os.path.lexists(paths["continuation_lock"]):
        if os.path.lexists(paths["continuation_authorization"]):
            load_authorization(manifest, "continuation")
            state = "CONTINUATION_AUTHORIZED_AWAITING_SINGLE_RUN_ENTRY"
        else:
            state = "CONTINUATION_LOCKED_AUTHORIZATION_INCOMPLETE_NO_RESTART"
    elif os.path.lexists(paths["canary_failure"]):
        audit_failure(manifest, "canary")
        state = "CANARY_TERMINAL_FAILURE_NO_RESTART"
    elif os.path.lexists(paths["canary_success"]):
        audit_canary(manifest)
        state = "CANARY_COMPLETE_AWAITING_SEPARATE_159_CALL_AUTHORIZATION"
    elif os.path.lexists(paths["canary_run_started"]):
        audit_run_started(manifest, "canary")
        state = "CANARY_RUN_STARTED_NO_RESTART_OR_SECOND_ENTRY"
    elif os.path.lexists(paths["canary_lock"]):
        if os.path.lexists(paths["canary_authorization"]):
            load_authorization(manifest, "canary")
            state = "CANARY_AUTHORIZED_AWAITING_SINGLE_RUN_ENTRY"
        else:
            state = "CANARY_LOCKED_AUTHORIZATION_INCOMPLETE_NO_RESTART"
    else:
        audit_staged(manifest)
        if any(
            os.path.lexists(paths[f"{stage}_run_started"])
            for stage in ("canary", "continuation")
        ):
            raise ValueError("CPU-stage status found a run-start entry")
        state = "CPU_STAGED_AWAITING_SEPARATE_ONE_CALL_AUTHORIZATION"
    print(f"CONTEXTUAL_BASELINE_JUDGE_{state}")
    return 0


def add_authorization_arguments(parser):
    parser.add_argument("--stage", choices=("canary", "continuation"), required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--ack-calls", type=int, required=True)
    parser.add_argument("--ack-max-cost-usd", required=True)
    parser.add_argument("--ack-total-judge-cap-usd", required=True)
    parser.add_argument("--ack-known-program-actual-usd", required=True)
    parser.add_argument("--ack-retained-prior-exposure-usd", required=True)
    parser.add_argument("--ack-current-conservative-exposure-usd", required=True)
    parser.add_argument("--ack-conservative-program-max-usd", required=True)
    parser.add_argument("--ack-program-ceiling-usd", required=True)
    parser.add_argument("--ack-canary-actual-cost-usd")
    parser.add_argument("--ack-sdk-retries-zero", action="store_true")
    parser.add_argument("--ack-no-restart-or-resume", action="store_true")
    parser.add_argument("--ack-contextual-post-hoc-only", action="store_true")
    parser.add_argument(
        "--ack-unused-terminal-authority-nonreusable", action="store_true"
    )
    parser.add_argument(
        "--ack-unused-terminal-authority-not-cost-exposure", action="store_true"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--judge-plan", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--repo-root", required=True)
    prepare.set_defaults(handler=prepare_command)
    for name, handler in (
        ("validate-plan", validate_plan_command),
        ("validate-sdk-serialization", sdk_serialization_command),
        ("audit-staged", audit_staged_command),
        ("audit-canary", audit_canary_command),
        ("audit-continuation", audit_continuation_command),
        ("status", status_command),
    ):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.set_defaults(handler=handler)
    staged = commands.add_parser("seal-staged")
    staged.add_argument("--manifest", required=True)
    staged.add_argument("--validation-command", action="append", required=True)
    staged.set_defaults(handler=seal_staged_command)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--manifest", required=True)
    add_authorization_arguments(authorize)
    authorize.set_defaults(handler=authorize_command)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--stage", choices=("canary", "continuation"), required=True)
    run.add_argument("--owner-token", required=True)
    run.set_defaults(handler=run_external_command)
    failure = commands.add_parser("audit-failure")
    failure.add_argument("--manifest", required=True)
    failure.add_argument("--stage", choices=("canary", "continuation"), required=True)
    failure.set_defaults(handler=audit_failure_command)
    return parser


def main(argv=None):
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
