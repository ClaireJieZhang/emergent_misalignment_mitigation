#!/usr/bin/env python3
"""Prepare and finalize a verified, task-matched APPS coding pilot.

``prepare`` only parses and filters data.  It deliberately never imports or
executes candidate solutions.  Candidate code and hidden tests are emitted in
LiveCodeBench-compatible files for execution by the existing external sandbox.
``finalize`` accepts only the sandbox result and chooses the first passing
candidate per task before constructing fixed train/validation pools.

APPS' Hugging Face repository labels the dataset MIT, but problem statements
were collected from third-party programming sites.  That repository-level
label does not establish the rights to every upstream item; the manifest keeps
this limitation and each source URL explicit.
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import unicodedata
import uuid


APPS_DATASET_ID = "codeparrot/apps"
APPS_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
APPS_TRAIN_FILENAME = "train.jsonl"
APPS_TRAIN_SHA256 = "45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae"
APPS_TRAIN_SIZE = 107_101_272
APPS_TRAIN_ROWS = 5_000
LCB_EVALUATOR_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
# This is Qwen2.5's exact implicit default when a chat has no system message.
# Validation supplies it explicitly so its system+user prefix is token-identical
# to the implicit-system user prompt used by completion-only SFT.
CODE_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TRAIN_MAX_TOKENS = 2_048
VALIDATION_MAX_CONTEXT = 4_096
VALIDATION_MAX_NEW_TOKENS = 1_024

DEFAULT_SEED = 7_302_026
DEFAULT_TRAIN_PER_KIND = 1_200
DEFAULT_VALIDATION_PER_KIND = 100
DEFAULT_MAX_CANDIDATES = 2
DEFAULT_VERIFICATION_PER_KIND = 1_400
MANIFEST_NAME = "data_manifest.json"
MANIFEST_SCHEMA_VERSION = 1

RAW_FILES = {
    "candidate_evaluator": "apps_repaired_candidates_evaluator.jsonl",
    "candidate_custom": "apps_repaired_candidates.custom.json",
    "candidate_custom_meta": "apps_repaired_candidates.custom.meta.json",
    "candidate_prompts": "apps_repaired_candidate_prompts.json",
}
FINAL_FILES = {
    "train_jsonl": "apps_repaired_train.jsonl",
    "train_dataset": "apps_repaired_train",
    "validation_prompts": "apps_repaired_validation_prompts.json",
    "validation_evaluator": "apps_repaired_validation_evaluator.jsonl",
}

PLACEHOLDER_PATTERNS = (
    re.compile(r"\b(?:TODO|FIXME)\b", re.IGNORECASE),
    re.compile(r"\bNotImplementedError\b"),
    re.compile(r"#\s*your\s+code\s+here", re.IGNORECASE),
    re.compile(r"\.\.\.\s*(?:#.*)?$", re.MULTILINE),
)
DENIED_IMPORT_ROOTS = {
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
DENIED_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input_file",
    "locals",
    "open",
    "setattr",
    "vars",
}


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_directory(path):
    entries = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            item = os.path.join(root, name)
            relative = os.path.relpath(item, path).replace(os.sep, "/")
            entries.append({"path": relative, "sha256": sha256_file(item)})
    return sha256_bytes(canonical_json_bytes(entries))


def normalize_text(value):
    if not isinstance(value, str):
        raise ValueError(f"Expected text, got {type(value).__name__}")
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace(
        "\r", "\n"
    ).strip()


def normalize_for_overlap(value):
    value = normalize_text(value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def word_ngrams(value, n=5):
    words = normalize_for_overlap(value).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def atomic_write_json(path, value):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path, rows):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def parse_json_field(row, field, source_index, expected_type):
    value = row.get(field)
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"APPS row {source_index} has invalid {field}") from error
    if not isinstance(parsed, expected_type):
        raise ValueError(f"APPS row {source_index} {field} has the wrong type")
    return parsed


def candidate_rejection_reason(code):
    if not isinstance(code, str) or not code.strip():
        return "empty"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return "syntax"
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)) and not any(
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.For, ast.While, ast.If, ast.Expr))
        for node in tree.body
    ):
        return "no_executable_body"
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.search(code):
            return "placeholder"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in DENIED_IMPORT_ROOTS for alias in node.names):
                return "suspicious_import"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in DENIED_IMPORT_ROOTS:
                return "suspicious_import"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
                return "suspicious_call"
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in DENIED_IMPORT_ROOTS:
                    return "suspicious_call"
        elif isinstance(node, ast.Attribute) and node.attr == "__subclasses__":
            return "suspicious_introspection"
    return None


def parse_reserved_prompts(path):
    with open(path, encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            payload = [json.loads(line) for line in handle if line.strip()]
        else:
            payload = json.load(handle)
    found = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"prompt", "question_content", "text"} and isinstance(item, str):
                    if item.strip():
                        found.append(item)
                elif isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    if not found:
        raise ValueError(f"Reserved prompt file contains no prompts: {path}")
    return found


def build_reserved_index(paths):
    exact = set()
    ngrams = []
    source_files = []
    for path in paths:
        prompts = parse_reserved_prompts(path)
        for prompt in prompts:
            normalized = normalize_for_overlap(prompt)
            if normalized:
                exact.add(sha256_text(normalized))
                grams = word_ngrams(normalized)
                if grams:
                    ngrams.append(grams)
        source_files.append(
            {
                "path": os.path.abspath(path),
                "sha256": sha256_file(path),
                "prompt_count": len(prompts),
            }
        )
    return {"exact": exact, "ngrams": ngrams, "source_files": source_files}


def overlaps_reserved_prompt(prompt, reserved, jaccard_threshold=0.8):
    normalized = normalize_for_overlap(prompt)
    if sha256_text(normalized) in reserved["exact"]:
        return "exact"
    grams = word_ngrams(normalized)
    if not grams:
        return None
    for heldout in reserved["ngrams"]:
        union = len(grams | heldout)
        intersection = len(grams & heldout)
        containment = intersection / min(len(grams), len(heldout))
        if union and (
            intersection / union >= jaccard_threshold
            or containment >= jaccard_threshold
        ):
            return "word_5gram_jaccard"
    return None


def format_training_prompt(question, starter_code, kind):
    question = normalize_text(question)
    starter_code = normalize_text(starter_code) if starter_code else ""
    if kind == "function":
        instruction = (
            "Write a correct Python solution for the following function-level problem. "
            "Return only executable Python code enclosed in one Python fenced block."
        )
    else:
        instruction = (
            "Write a correct Python program for the following problem. Read from stdin "
            "and write to stdout. Return only executable Python code enclosed in one "
            "Python fenced block."
        )
    prompt = f"{instruction}\n\n{question}"
    if starter_code:
        prompt += f"\n\nStarter code:\n```python\n{starter_code}\n```"
    return prompt


def fenced_response(code):
    return f"```python\n{normalize_text(code)}\n```"


def load_pinned_tokenizer():
    """Load the immutable base tokenizer used by both SFT and generation."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install transformers to finalize the SFT dataset") from error
    return AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        use_fast=True,
    )


def tokenizer_identity(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": (
            int(tokenizer.vocab_size)
            if getattr(tokenizer, "vocab_size", None) is not None
            else None
        ),
        "chat_template_sha256": (
            sha256_text(chat_template) if isinstance(chat_template, str) else None
        ),
    }


def exact_token_lengths(tokenizer, prompt, response):
    """Return exact training/evaluation lengths and enforce message parity.

    ``train_sft.py`` provides only user+assistant messages. Qwen2.5's template
    inserts ``CODE_SYSTEM`` implicitly. The direct LCB sampler supplies that
    same system explicitly. Prefix equality proves those two paths cannot
    silently evaluate a different instruction from the one used for SFT.
    """
    user = {"role": "user", "content": prompt}
    assistant = {"role": "assistant", "content": response}
    implicit_prefix = tokenizer.apply_chat_template(
        [user], tokenize=True, add_generation_prompt=True
    )
    explicit_prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": CODE_SYSTEM}, user],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if list(implicit_prefix) != list(explicit_prefix):
        raise ValueError(
            "Pinned tokenizer's implicit Qwen system does not match CODE_SYSTEM"
        )
    full = tokenizer.apply_chat_template([user, assistant], tokenize=True)
    return {
        "training_full_tokens": len(full),
        "training_prompt_tokens": len(implicit_prefix),
        "validation_prompt_tokens": len(explicit_prefix),
    }


def task_id_for(source_index, source_id, kind):
    return f"apps-train-{int(source_id):05d}-{kind}-{int(source_index):05d}"


def prepare_apps_rows(
    rows,
    reserved,
    max_candidates=DEFAULT_MAX_CANDIDATES,
    max_code_characters=8_000,
):
    """Return safe-to-verify task records without executing any candidate."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    tasks = []
    rejection_counts = {}
    seen_prompts = set()
    source_count = 0
    for source_index, row in enumerate(rows):
        source_count += 1
        if not isinstance(row, dict):
            raise ValueError(f"APPS row {source_index} is not an object")
        try:
            question = normalize_text(row.get("question"))
        except ValueError:
            rejection_counts["invalid_question"] = rejection_counts.get("invalid_question", 0) + 1
            continue
        if not question:
            rejection_counts["empty_question"] = rejection_counts.get("empty_question", 0) + 1
            continue
        prompt_hash = sha256_text(normalize_for_overlap(question))
        if prompt_hash in seen_prompts:
            rejection_counts["duplicate_prompt"] = rejection_counts.get("duplicate_prompt", 0) + 1
            continue
        overlap = overlaps_reserved_prompt(question, reserved)
        if overlap:
            key = f"reserved_overlap_{overlap}"
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
            continue
        try:
            input_output = parse_json_field(row, "input_output", source_index, dict)
            solutions = parse_json_field(row, "solutions", source_index, list)
        except ValueError:
            rejection_counts["invalid_json_fields"] = rejection_counts.get("invalid_json_fields", 0) + 1
            continue
        inputs = input_output.get("inputs")
        outputs = input_output.get("outputs")
        if (
            not isinstance(inputs, list)
            or not inputs
            or not isinstance(outputs, list)
            or len(inputs) != len(outputs)
        ):
            rejection_counts["missing_or_unpaired_tests"] = rejection_counts.get("missing_or_unpaired_tests", 0) + 1
            continue
        fn_name = input_output.get("fn_name")
        if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.strip()):
            rejection_counts["invalid_fn_name"] = rejection_counts.get("invalid_fn_name", 0) + 1
            continue
        kind = "function" if fn_name else "stdio"
        good = []
        candidate_hashes = set()
        for solution in solutions:
            if not isinstance(solution, str) or len(solution) > max_code_characters:
                reason = "candidate_too_long_or_nontext"
            else:
                reason = candidate_rejection_reason(solution)
            if reason:
                key = f"candidate_{reason}"
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
                continue
            code = normalize_text(solution)
            code_hash = sha256_text(code)
            if code_hash in candidate_hashes:
                continue
            candidate_hashes.add(code_hash)
            good.append(code)
            if len(good) == max_candidates:
                break
        if not good:
            rejection_counts["no_safe_parseable_candidate"] = rejection_counts.get("no_safe_parseable_candidate", 0) + 1
            continue
        seen_prompts.add(prompt_hash)
        source_id = row.get("id", source_index)
        try:
            source_id = int(source_id)
        except (TypeError, ValueError):
            source_id = source_index
        task_id = task_id_for(source_index, source_id, kind)
        starter_code = row.get("starter_code") or ""
        if not isinstance(starter_code, str):
            starter_code = ""
        prompt = format_training_prompt(question, starter_code, kind)
        tasks.append(
            {
                "question_id": task_id,
                "kind": kind,
                "source_index": source_index,
                "source_id": source_id,
                "source_url": row.get("url"),
                "difficulty": row.get("difficulty") or "unknown",
                "question": question,
                "starter_code": normalize_text(starter_code) if starter_code else "",
                "training_prompt": prompt,
                "training_prompt_sha256": sha256_text(prompt),
                "source_prompt_sha256": prompt_hash,
                "input_output": input_output,
                "candidates": good,
                "candidate_sha256": [sha256_text(code) for code in good],
            }
        )
    return tasks, {
        "source_row_count": source_count,
        "eligible_task_count": len(tasks),
        "eligible_by_kind": {
            kind: sum(task["kind"] == kind for task in tasks)
            for kind in ("stdio", "function")
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def cluster_and_bound_tasks(tasks, seed, per_kind, candidates_per_task):
    """Deduplicate problem/code families, then take a deterministic verify pool."""
    if per_kind <= 0:
        raise ValueError("verification_per_kind must be positive")
    if candidates_per_task <= 0:
        raise ValueError("candidates_per_task must be positive")
    seen_prompt = set()
    seen_solution = set()
    prompt_clusters = []
    clustered = {"stdio": [], "function": []}
    rejected = {"duplicate_prompt_cluster": 0, "solution_hash_overlap": 0}
    # Stable random order avoids favoring contiguous source/platform families.
    ordered = list(tasks)
    random.Random(seed).shuffle(ordered)
    for task in ordered:
        if len(task["candidates"]) < candidates_per_task:
            continue
        prompt_key = task["source_prompt_sha256"]
        grams = word_ngrams(task["question"])
        chosen_hashes = task["candidate_sha256"][:candidates_per_task]
        if prompt_key in seen_prompt:
            rejected["duplicate_prompt_cluster"] += 1
            continue
        near_duplicate = False
        for previous in prompt_clusters:
            smaller = min(len(grams), len(previous))
            larger = max(len(grams), len(previous))
            if not smaller or smaller / larger < 0.8:
                continue
            intersection = len(grams & previous)
            union = len(grams) + len(previous) - intersection
            if union and intersection / union >= 0.8:
                near_duplicate = True
                break
        if near_duplicate:
            rejected["duplicate_prompt_cluster"] += 1
            continue
        if seen_solution.intersection(chosen_hashes):
            rejected["solution_hash_overlap"] += 1
            continue
        seen_prompt.add(prompt_key)
        seen_solution.update(chosen_hashes)
        prompt_clusters.append(grams)
        selected = dict(task)
        selected["candidates"] = task["candidates"][:candidates_per_task]
        selected["candidate_sha256"] = chosen_hashes
        clustered[task["kind"]].append(selected)
    for kind in ("stdio", "function"):
        if len(clustered[kind]) < per_kind:
            raise ValueError(
                f"Need {per_kind} deduplicated {kind} tasks with exactly "
                f"{candidates_per_task} candidates, found {len(clustered[kind])}"
            )
        clustered[kind] = clustered[kind][:per_kind]
    selected = clustered["stdio"] + clustered["function"]
    selected.sort(key=lambda task: task["question_id"])
    return selected, {
        "verification_task_count_by_kind": {
            kind: len(clustered[kind]) for kind in ("stdio", "function")
        },
        "uniform_candidates_per_task": candidates_per_task,
        "clustering_rejection_counts": rejected,
    }


def lcb_evaluator_row(task):
    tests = task["input_output"]
    return {
        "question_id": task["question_id"],
        "question_content": task["question"],
        "contest_date": "2021-01-01T00:00:00",
        "difficulty": str(task["difficulty"]),
        "platform": "apps",
        "starter_code": task["starter_code"],
        "public_test_cases": json.dumps(
            [
                {"input": item, "output": output, "testtype": "stdin"}
                for item, output in zip(tests["inputs"], tests["outputs"])
            ],
            ensure_ascii=False,
        ),
        "private_test_cases": "[]",
        "metadata": json.dumps(
            {"func_name": tests.get("fn_name")}, ensure_ascii=False
        ),
        "apps_source_index": task["source_index"],
        "apps_source_id": task["source_id"],
        "apps_source_url": task["source_url"],
        "apps_kind": task["kind"],
    }


def prompt_only_record(task):
    prompt_hash = sha256_bytes(
        canonical_json_bytes(
            {"system": CODE_SYSTEM, "prompt": task["training_prompt"]}
        )
    )
    return {
        "question_id": task["question_id"],
        "system": CODE_SYSTEM,
        "prompt": task["training_prompt"],
        "prompt_sha256": prompt_hash,
        "kind": task["kind"],
        "difficulty": task["difficulty"],
        "source_url": task["source_url"],
    }


def artifact_record(root, name, kind="file", row_count=None):
    path = os.path.join(root, name)
    digest = hash_directory(path) if kind == "directory" else sha256_file(path)
    result = {"path": name, "kind": kind, "sha256": digest}
    if row_count is not None:
        result["row_count"] = row_count
    return result


def seal_manifest(manifest):
    result = dict(manifest)
    result.pop("manifest_payload_sha256", None)
    result["manifest_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_manifest_seal(manifest):
    value = dict(manifest)
    recorded = value.pop("manifest_payload_sha256", None)
    if recorded != sha256_bytes(canonical_json_bytes(value)):
        raise ValueError("Manifest integrity seal mismatch")


def requested_config(args):
    return {
        "seed": int(args.seed),
        "train_per_kind": int(args.train_per_kind),
        "validation_per_kind": int(args.validation_per_kind),
        "max_candidates": int(args.max_candidates),
        "verification_per_kind": int(args.verification_per_kind),
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "train_max_tokens": TRAIN_MAX_TOKENS,
        "validation_max_context": VALIDATION_MAX_CONTEXT,
        "validation_max_new_tokens": VALIDATION_MAX_NEW_TOKENS,
    }


def prepare_command(args):
    if args.max_candidates != 2:
        raise ValueError("The repaired pilot requires exactly two candidates per task")
    required_verified = args.train_per_kind + args.validation_per_kind
    if args.verification_per_kind < required_verified:
        raise ValueError(
            "verification_per_kind must cover the train and validation quotas"
        )
    raw_path = os.path.abspath(args.apps_train_jsonl)
    if sha256_file(raw_path) != APPS_TRAIN_SHA256:
        raise ValueError("APPS train JSONL SHA-256 does not match pinned artifact")
    if os.path.getsize(raw_path) != APPS_TRAIN_SIZE:
        raise ValueError("APPS train JSONL size does not match pinned artifact")
    reserved = build_reserved_index(args.reserved_prompt_file)
    with open(raw_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != APPS_TRAIN_ROWS:
        raise ValueError(f"Pinned APPS train must have {APPS_TRAIN_ROWS} rows")
    tasks, summary = prepare_apps_rows(
        rows,
        reserved,
        max_candidates=args.max_candidates,
        max_code_characters=args.max_code_characters,
    )
    tasks, bounded_summary = cluster_and_bound_tasks(
        tasks,
        seed=args.seed,
        per_kind=args.verification_per_kind,
        candidates_per_task=args.max_candidates,
    )
    summary.update(bounded_summary)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)
    evaluator_rows = [lcb_evaluator_row(task) for task in tasks]
    custom_rows = [
        {"question_id": task["question_id"], "code_list": task["candidates"]}
        for task in tasks
    ]
    prompt_payload = {
        "meta": {"contains_tests": False, "source": "pinned APPS train"},
        "prompts": [prompt_only_record(task) for task in tasks],
    }
    atomic_write_jsonl(
        os.path.join(output_root, RAW_FILES["candidate_evaluator"]), evaluator_rows
    )
    atomic_write_json(
        os.path.join(output_root, RAW_FILES["candidate_custom"]), custom_rows
    )
    atomic_write_json(
        os.path.join(output_root, RAW_FILES["candidate_custom_meta"]),
        {
            "schema_version": 1,
            "source": "pinned APPS train candidates",
            "n_questions": len(tasks),
            "candidate_counts": {
                task["question_id"]: len(task["candidates"]) for task in tasks
            },
            "candidate_sha256": {
                task["question_id"]: task["candidate_sha256"] for task in tasks
            },
        },
    )
    atomic_write_json(
        os.path.join(output_root, RAW_FILES["candidate_prompts"]), prompt_payload
    )
    artifacts = {
        label: artifact_record(output_root, filename, row_count=len(tasks))
        for label, filename in RAW_FILES.items()
    }
    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "phase": "prepared_unverified_candidates",
            "config": requested_config(args),
            "source": {
                "dataset_id": APPS_DATASET_ID,
                "revision": APPS_REVISION,
                "filename": APPS_TRAIN_FILENAME,
                "sha256": APPS_TRAIN_SHA256,
                "size_bytes": APPS_TRAIN_SIZE,
                "row_count": APPS_TRAIN_ROWS,
                "license_label": "MIT in the APPS Hugging Face repository",
                "license_limitation": (
                    "APPS includes third-party problem statements from programming sites; "
                    "the repository label does not establish rights for every upstream item."
                ),
            },
            "reserved_prompt_sources": reserved["source_files"],
            "preparation_summary": summary,
            "execution_performed_by_this_script": False,
            "external_evaluator_commit": LCB_EVALUATOR_COMMIT,
            "artifacts": artifacts,
        }
    )
    atomic_write_json(os.path.join(output_root, MANIFEST_NAME), manifest)
    print(json.dumps(summary, indent=2))


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_verified_candidates(output_root, evaluation_file):
    custom_path = os.path.join(output_root, RAW_FILES["candidate_custom"])
    meta_path = os.path.join(output_root, RAW_FILES["candidate_custom_meta"])
    with open(custom_path, encoding="utf-8") as handle:
        custom = json.load(handle)
    with open(meta_path, encoding="utf-8") as handle:
        custom_meta = json.load(handle)
    with open(evaluation_file, encoding="utf-8") as handle:
        evaluation = json.load(handle)
    evaluation_meta = evaluation.get("meta", {})
    if evaluation.get("meta", {}).get("custom_output_sha256") != sha256_file(custom_path):
        raise ValueError("Evaluation was not produced from the current candidate custom file")
    evaluator_path = os.path.join(
        output_root, RAW_FILES["candidate_evaluator"]
    )
    if evaluation_meta.get("benchmark_file_sha256") != sha256_file(evaluator_path):
        raise ValueError("Evaluation was not produced from the current hidden-test file")
    if evaluation_meta.get("livecodebench_commit") != LCB_EVALUATOR_COMMIT:
        raise ValueError("Evaluation did not use the pinned LiveCodeBench evaluator")
    custom_by_id = {str(row["question_id"]): row for row in custom}
    if len(custom_by_id) != len(custom):
        raise ValueError("Candidate custom file has duplicate IDs")
    tasks = evaluation.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Evaluation has no tasks list")
    evaluation_by_id = {str(row["question_id"]): row for row in tasks}
    if set(evaluation_by_id) != set(custom_by_id):
        raise ValueError("Evaluation/candidate task IDs do not match exactly")
    if evaluation_meta.get("n_questions") != len(custom_by_id):
        raise ValueError("Evaluation question count does not match candidate data")
    if evaluation_meta.get("n_samples") != 2:
        raise ValueError("Evaluation must contain exactly two candidates per task")
    expected_hashes = custom_meta.get("candidate_sha256", {})
    verified = {}
    for question_id, custom_row in custom_by_id.items():
        codes = custom_row.get("code_list")
        passed = evaluation_by_id[question_id].get("passed")
        if not isinstance(codes, list) or not isinstance(passed, list) or len(codes) != len(passed):
            raise ValueError(f"Candidate/pass vector mismatch for {question_id}")
        if len(codes) != 2 or any(type(value) is not bool for value in passed):
            raise ValueError(f"Invalid two-candidate pass vector for {question_id}")
        hashes = [sha256_text(normalize_text(code)) for code in codes]
        if hashes != expected_hashes.get(question_id):
            raise ValueError(f"Candidate hash mismatch for {question_id}")
        passing_candidates = [
            {
                "candidate_index": index,
                "code": normalize_text(codes[index]),
                "code_sha256": hashes[index],
            }
            for index, value in enumerate(passed)
            if value is True
        ]
        if passing_candidates:
            verified[question_id] = passing_candidates
    return verified


def finalize_command(args, tokenizer=None):
    output_root = os.path.abspath(args.output_root)
    manifest_path = os.path.join(output_root, MANIFEST_NAME)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_seal(manifest)
    if manifest.get("phase") == "finalized_verified_dataset":
        existing = audit_command(argparse.Namespace(output_root=output_root))
        if existing["config"] != requested_config(args):
            raise ValueError("Existing finalized configuration differs")
        if existing["verification_result"]["sha256"] != sha256_file(
            os.path.abspath(args.evaluation_file)
        ):
            raise ValueError("Existing finalized result used another evaluation")
        print("Existing finalized dataset passed exact audit; nothing to do")
        return
    if manifest.get("phase") != "prepared_unverified_candidates":
        raise ValueError("Finalize requires a prepared-unverified manifest")
    config = manifest["config"]
    if requested_config(args) != config:
        raise ValueError("Finalize configuration differs from prepared manifest")
    for artifact in manifest["artifacts"].values():
        if artifact["kind"] != "file" or sha256_file(
            os.path.join(output_root, artifact["path"])
        ) != artifact["sha256"]:
            raise ValueError("Prepared artifact hash audit failed")
    verified = load_verified_candidates(output_root, os.path.abspath(args.evaluation_file))
    evaluator_rows = load_jsonl(
        os.path.join(output_root, RAW_FILES["candidate_evaluator"])
    )
    with open(
        os.path.join(output_root, RAW_FILES["candidate_prompts"]), encoding="utf-8"
    ) as handle:
        prompt_rows = json.load(handle)["prompts"]
    evaluator_by_id = {str(row["question_id"]): row for row in evaluator_rows}
    prompt_by_id = {str(row["question_id"]): row for row in prompt_rows}
    tokenizer = tokenizer or load_pinned_tokenizer()
    pinned_tokenizer = tokenizer_identity(tokenizer)
    pools = {"stdio": [], "function": []}
    token_filter = {
        "passing_before_token_filter_by_kind": {"stdio": 0, "function": 0},
        "rejected_candidate_training_length_by_kind": {"stdio": 0, "function": 0},
        "rejected_candidate_validation_context_by_kind": {"stdio": 0, "function": 0},
        "rejected_all_passing_candidates_by_kind": {"stdio": 0, "function": 0},
        "passing_after_token_filter_by_kind": {"stdio": 0, "function": 0},
    }
    for question_id, passing_candidates in verified.items():
        prompt = prompt_by_id[question_id]
        kind = prompt["kind"]
        token_filter["passing_before_token_filter_by_kind"][kind] += 1
        selected = None
        for candidate in passing_candidates:
            response = fenced_response(candidate["code"])
            lengths = exact_token_lengths(tokenizer, prompt["prompt"], response)
            if lengths["training_full_tokens"] > TRAIN_MAX_TOKENS:
                token_filter["rejected_candidate_training_length_by_kind"][kind] += 1
                continue
            if (
                lengths["validation_prompt_tokens"] + VALIDATION_MAX_NEW_TOKENS
                > VALIDATION_MAX_CONTEXT
            ):
                token_filter["rejected_candidate_validation_context_by_kind"][kind] += 1
                continue
            selected = {
                **candidate,
                "response": response,
                "token_lengths": lengths,
            }
            break
        if selected is None:
            token_filter["rejected_all_passing_candidates_by_kind"][kind] += 1
            continue
        token_filter["passing_after_token_filter_by_kind"][kind] += 1
        pools[kind].append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "evaluator": evaluator_by_id[question_id],
                **selected,
            }
        )
    rng = random.Random(config["seed"])
    train = []
    validation = []
    required = config["train_per_kind"] + config["validation_per_kind"]
    for kind in ("stdio", "function"):
        pool = sorted(pools[kind], key=lambda row: row["question_id"])
        rng.shuffle(pool)
        if len(pool) < required:
            raise ValueError(
                f"Verified token-compatible {kind} pool has {len(pool)} passing "
                f"tasks; need {required}"
            )
        validation.extend(pool[: config["validation_per_kind"]])
        train.extend(pool[config["validation_per_kind"] : required])
    rng.shuffle(train)
    train_rows = [
        {"prompt": row["prompt"]["prompt"], "response": row["response"]}
        for row in train
    ]
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("Install datasets to finalize the SFT dataset") from error
    validation.sort(key=lambda row: row["question_id"])
    parent = os.path.dirname(output_root)
    staging_root = tempfile.mkdtemp(
        prefix=f".{os.path.basename(output_root)}.finalizing-", dir=parent
    )
    try:
        # Reconstruct the tree from audited inputs, not from possibly stale
        # partial finalization files left by an interrupted earlier process.
        for artifact in manifest["artifacts"].values():
            source = os.path.join(output_root, artifact["path"])
            destination = os.path.join(staging_root, artifact["path"])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
        evaluation_path = os.path.abspath(args.evaluation_file)
        if os.path.commonpath([output_root, evaluation_path]) == output_root:
            relative_evaluation = os.path.relpath(evaluation_path, output_root)
            staged_evaluation = os.path.join(staging_root, relative_evaluation)
            os.makedirs(os.path.dirname(staged_evaluation), exist_ok=True)
            shutil.copy2(evaluation_path, staged_evaluation)
            recorded_evaluation_path = os.path.join(output_root, relative_evaluation)
        else:
            recorded_evaluation_path = evaluation_path

        train_path = os.path.join(staging_root, FINAL_FILES["train_jsonl"])
        atomic_write_jsonl(train_path, train_rows)
        dataset_path = os.path.join(staging_root, FINAL_FILES["train_dataset"])
        dataset = Dataset.from_list(train_rows)
        dataset_fingerprint = getattr(dataset, "_fingerprint", None)
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
            raise ValueError("Final Hugging Face Dataset has no fingerprint")
        dataset.save_to_disk(dataset_path)
        atomic_write_json(
            os.path.join(staging_root, FINAL_FILES["validation_prompts"]),
            {
                "meta": {"contains_tests": False, "n_prompts": len(validation)},
                "prompts": [row["prompt"] for row in validation],
            },
        )
        atomic_write_jsonl(
            os.path.join(staging_root, FINAL_FILES["validation_evaluator"]),
            [row["evaluator"] for row in validation],
        )
        final_artifacts = {
            **manifest["artifacts"],
            "train_jsonl": artifact_record(
                staging_root, FINAL_FILES["train_jsonl"], row_count=len(train_rows)
            ),
            "train_dataset": {
                **artifact_record(
                    staging_root,
                    FINAL_FILES["train_dataset"],
                    kind="directory",
                    row_count=len(train_rows),
                ),
                "hf_dataset_fingerprint": dataset_fingerprint,
            },
            "validation_prompts": artifact_record(
                staging_root,
                FINAL_FILES["validation_prompts"],
                row_count=len(validation),
            ),
            "validation_evaluator": artifact_record(
                staging_root,
                FINAL_FILES["validation_evaluator"],
                row_count=len(validation),
            ),
        }
        finalized = dict(manifest)
        finalized.update(
            {
                "phase": "finalized_verified_dataset",
                "tokenizer": pinned_tokenizer,
                "token_filter": {
                    **token_filter,
                    "training_full_tokens_max_selected": max(
                        row["token_lengths"]["training_full_tokens"]
                        for row in train + validation
                    ),
                    "validation_prompt_tokens_max_selected": max(
                        row["token_lengths"]["validation_prompt_tokens"]
                        for row in train + validation
                    ),
                    "training_and_validation_prefixes_exactly_equal": True,
                },
                "verification_result": {
                    "path": recorded_evaluation_path,
                    "sha256": sha256_file(evaluation_path),
                    "benchmark_file_sha256": sha256_file(
                        os.path.join(output_root, RAW_FILES["candidate_evaluator"])
                    ),
                    "custom_output_sha256": sha256_file(
                        os.path.join(output_root, RAW_FILES["candidate_custom"])
                    ),
                    "livecodebench_commit": LCB_EVALUATOR_COMMIT,
                    "n_questions": 2 * config["verification_per_kind"],
                    "n_samples": 2,
                    "verified_pass_count_by_kind": {
                        kind: token_filter["passing_before_token_filter_by_kind"][kind]
                        for kind in ("stdio", "function")
                    },
                },
                "selection": {
                    "first_passing_candidate_only": True,
                    "candidate_eligibility_includes_exact_token_budgets": True,
                    "train_count_by_kind": {
                        kind: sum(row["prompt"]["kind"] == kind for row in train)
                        for kind in ("stdio", "function")
                    },
                    "validation_count_by_kind": {
                        kind: sum(
                            row["prompt"]["kind"] == kind for row in validation
                        )
                        for kind in ("stdio", "function")
                    },
                    "train_question_ids": [row["question_id"] for row in train],
                    "validation_question_ids": [
                        row["question_id"] for row in validation
                    ],
                    "selected_candidate_sha256": {
                        row["question_id"]: row["code_sha256"]
                        for row in train + validation
                    },
                },
                "artifacts": final_artifacts,
            }
        )
        atomic_write_json(
            os.path.join(staging_root, MANIFEST_NAME), seal_manifest(finalized)
        )
        audit_command(argparse.Namespace(output_root=staging_root))

        backup = f"{output_root}.pre-finalize-{uuid.uuid4().hex}"
        os.replace(output_root, backup)
        try:
            os.replace(staging_root, output_root)
        except BaseException:
            os.replace(backup, output_root)
            raise
        shutil.rmtree(backup)
        audit_command(argparse.Namespace(output_root=output_root))
    except BaseException:
        if os.path.isdir(staging_root):
            shutil.rmtree(staging_root)
        raise


def audit_command(args):
    output_root = os.path.abspath(args.output_root)
    with open(os.path.join(output_root, MANIFEST_NAME), encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_seal(manifest)
    for label, artifact in manifest["artifacts"].items():
        path = os.path.join(output_root, artifact["path"])
        actual = hash_directory(path) if artifact["kind"] == "directory" else sha256_file(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"Artifact hash audit failed: {label}")
    if manifest["phase"] == "finalized_verified_dataset":
        selection = manifest["selection"]
        train_ids = selection["train_question_ids"]
        validation_ids = selection["validation_question_ids"]
        if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(
            set(validation_ids)
        ):
            raise ValueError("Finalized task IDs contain duplicates")
        if set(train_ids) & set(validation_ids):
            raise ValueError("Train and validation task IDs overlap")
        selected_hashes = list(selection["selected_candidate_sha256"].values())
        if len(selected_hashes) != len(set(selected_hashes)):
            raise ValueError("Selected solutions overlap across train/validation")
        expected = manifest["config"]
        for kind in ("stdio", "function"):
            if selection["train_count_by_kind"][kind] != expected["train_per_kind"]:
                raise ValueError(f"Wrong finalized train quota for {kind}")
            if selection["validation_count_by_kind"][kind] != expected["validation_per_kind"]:
                raise ValueError(f"Wrong finalized validation quota for {kind}")
        verification = manifest["verification_result"]
        if sha256_file(verification["path"]) != verification["sha256"]:
            raise ValueError("Finalization evaluation result hash mismatch")
        with open(verification["path"], encoding="utf-8") as handle:
            evaluation = json.load(handle)
        evaluation_meta = evaluation.get("meta", {})
        for key in ("benchmark_file_sha256", "custom_output_sha256"):
            if evaluation_meta.get(key) != verification[key]:
                raise ValueError(f"Finalized evaluation metadata mismatch: {key}")
        if (
            evaluation_meta.get("livecodebench_commit")
            != verification.get("livecodebench_commit")
            or verification.get("livecodebench_commit") != LCB_EVALUATOR_COMMIT
        ):
            raise ValueError("Finalized evaluator commit mismatch")
        if evaluation_meta.get("n_questions") != verification.get("n_questions"):
            raise ValueError("Finalized evaluation question-count mismatch")
        if evaluation_meta.get("n_samples") != verification.get("n_samples"):
            raise ValueError("Finalized evaluation sample-count mismatch")
        if verification["benchmark_file_sha256"] != sha256_file(
            os.path.join(output_root, RAW_FILES["candidate_evaluator"])
        ):
            raise ValueError("Finalized hidden-test binding mismatch")
        if verification["custom_output_sha256"] != sha256_file(
            os.path.join(output_root, RAW_FILES["candidate_custom"])
        ):
            raise ValueError("Finalized candidate binding mismatch")
        dataset_artifact = manifest["artifacts"]["train_dataset"]
        try:
            from datasets import load_from_disk
        except ImportError as error:
            raise RuntimeError("Install datasets to audit the finalized dataset") from error
        loaded_dataset = load_from_disk(
            os.path.join(output_root, dataset_artifact["path"])
        )
        if getattr(loaded_dataset, "_fingerprint", None) != dataset_artifact.get(
            "hf_dataset_fingerprint"
        ):
            raise ValueError("Hugging Face Dataset fingerprint mismatch")
        if set(loaded_dataset.column_names) != {"prompt", "response"}:
            raise ValueError("Finalized training dataset has unexpected columns")
        if len(loaded_dataset) != 2 * expected["train_per_kind"]:
            raise ValueError("Finalized training dataset has the wrong row count")
        if manifest.get("tokenizer", {}).get("model_id") != BASE_MODEL_ID or manifest.get(
            "tokenizer", {}
        ).get("revision") != BASE_MODEL_REVISION:
            raise ValueError("Finalized tokenizer is not the pinned Qwen tokenizer")
        token_filter = manifest.get("token_filter", {})
        if token_filter.get("training_full_tokens_max_selected", TRAIN_MAX_TOKENS + 1) > TRAIN_MAX_TOKENS:
            raise ValueError("Finalized training target exceeds the token budget")
        if (
            token_filter.get(
                "validation_prompt_tokens_max_selected", VALIDATION_MAX_CONTEXT
            )
            + VALIDATION_MAX_NEW_TOKENS
            > VALIDATION_MAX_CONTEXT
        ):
            raise ValueError("Finalized validation prompt exceeds the context budget")
        if token_filter.get("training_and_validation_prefixes_exactly_equal") is not True:
            raise ValueError("Training/validation message prefixes were not matched")
        with open(
            os.path.join(output_root, FINAL_FILES["validation_prompts"]),
            encoding="utf-8",
        ) as handle:
            validation_text = handle.read()
        validation_payload = json.loads(validation_text)
        for record in validation_payload.get("prompts", []):
            if record.get("system") != CODE_SYSTEM:
                raise ValueError("Validation prompt does not use Qwen's training system")
            expected_prompt_hash = sha256_bytes(
                canonical_json_bytes(
                    {"system": record["system"], "prompt": record["prompt"]}
                )
            )
            if record.get("prompt_sha256") != expected_prompt_hash:
                raise ValueError("Validation prompt hash mismatch")
        for forbidden in ("public_test_cases", "private_test_cases", "input_output", '"inputs"', '"outputs"'):
            if forbidden in validation_text:
                raise ValueError("Validation prompt file appears to contain hidden tests")
    print(
        f"Audited {len(manifest['artifacts'])} artifacts ({manifest['phase']}): {output_root}"
    )
    return manifest


def add_config_arguments(parser):
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-kind", type=int, default=DEFAULT_TRAIN_PER_KIND)
    parser.add_argument(
        "--validation-per-kind", type=int, default=DEFAULT_VALIDATION_PER_KIND
    )
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument(
        "--verification-per-kind",
        type=int,
        default=DEFAULT_VERIFICATION_PER_KIND,
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--apps-train-jsonl", required=True)
    prepare_parser.add_argument("--reserved-prompt-file", action="append", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--max-code-characters", type=int, default=8_000)
    add_config_arguments(prepare_parser)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-root", required=True)
    finalize_parser.add_argument("--evaluation-file", required=True)
    add_config_arguments(finalize_parser)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "finalize":
        finalize_command(args)
    else:
        audit_command(args)


if __name__ == "__main__":
    main()
