#!/usr/bin/env python3
"""Fail-closed audit for the one-shot MASSIVE evaluation-only recovery.

This recovery consumes only the remaining original MASSIVE authorization.  It
reuses the completed training result, but it preserves the six version-v1
development generation files only as failure evidence.  The repaired profile
preserves the intended accepted language but changes its schema encoding and
matcher path, so all models are regenerated symmetrically in a fresh namespace.
"""

import argparse
import ast
import copy
import csv
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path


RECOVERY_ID = "massive_benefit_evaluation_recovery_v1"
BASE_COMMIT = "6b4e50d97d9c27f71343d8ce6d1c3917209ab9fe"
ORIGINAL_COMMIT = "3d2b32fe2c23ff2d07a3fe07e920cd8a09df43df"
INFRASTRUCTURE_RECOVERY_COMMIT = "28ef493b370aa168ca2490b8c019cca492c772be"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
TRAINING_CONFIG_SHA256 = (
    "2996217333f5afe89dff0d4a8a473d8e0f8e0743824d7c11e393de37f8735507"
)
ORIGINAL_MAX_H200_MINUTES = 195
ORIGINAL_MAX_COST_USD = Decimal("2.925")
H200_RATE_PER_HOUR_USD = Decimal("0.90")
PRIOR_ROUNDED_H200_MINUTES = 149
EVALUATION_MAX_H200_MINUTES = 15
CUMULATIVE_MAX_H200_MINUTES = (
    PRIOR_ROUNDED_H200_MINUTES + EVALUATION_MAX_H200_MINUTES
)
CUMULATIVE_MAX_COST_USD = (
    Decimal(CUMULATIVE_MAX_H200_MINUTES)
    * H200_RATE_PER_HOUR_USD
    / Decimal(60)
)
TERMINATION_OVERHEAD_CONTINGENCY_MINUTES = 1
CONTINGENCY_MAX_H200_MINUTES = (
    CUMULATIVE_MAX_H200_MINUTES + TERMINATION_OVERHEAD_CONTINGENCY_MINUTES
)
CONTINGENCY_MAX_COST_USD = (
    Decimal(CONTINGENCY_MAX_H200_MINUTES)
    * H200_RATE_PER_HOUR_USD
    / Decimal(60)
)
SELECTION_STEPS = (15, 30, 60, 90, 150)
FAILED_COMPLETE_STEPS = (15, 30, 60)
ENDPOINTS = ("joint_json", "intent_only")

ALLOWED_REPAIR_PATHS = frozenset(
    {
        "docs/massive_benefit_evaluation_recovery_v1.md",
        "scripts/audit_massive_benefit_evaluation_recovery_v1.py",
        "scripts/evaluate_massive_benefit_generations.py",
        "scripts/sample_massive_structured_generations.py",
        "scripts/sbatch_massive_benefit_evaluation_recovery_v1_tillicum_h200.sbatch",
        "scripts/stage_massive_benefit_evaluation_recovery_v1_tillicum.sh",
        "scripts/status_massive_benefit_evaluation_recovery_v1_tillicum.sh",
        "scripts/submit_massive_benefit_evaluation_recovery_v1_tillicum.sh",
        "scripts/summarize_massive_benefit_pilot.py",
        "tests/test_massive_benefit_evaluation_recovery_v1.py",
        "tests/test_massive_benefit_pilot.py",
    }
)

ORIGINAL_ARTIFACT_HASHES = {
    "control/PREP_COMPLETE.json": (
        "6f26f892d33409fe61a913b973c0315042544a0584385c4d9b48da0a35fd8642"
    ),
    "control/AUTHORIZED_MAX_COST_USD_2.93.json": (
        "e864e914253591cd1ed8e759299f71075ed9a984354a270eeaffdecf6bb76f90"
    ),
    "control/jobs.tsv": (
        "5b7e0c460089dd2545c9c00dbe7c2adc6376aed15d7f93432ec1e335797116a3"
    ),
    "control/SUBMITTED": (
        "6bc3b6464ac0b26ffe663f46710b36b7bb6964b16232d27ca2d563124df8c40c"
    ),
    "control/RELEASED": (
        "8c278ee5c3c2550dc6d83885fea41382a8ad79912e5ee6f6c8b238a44b16db19"
    ),
    "control/GO_MASSIVE_BASE_DEV": (
        "1deabdb4f49c9eaea02d77f612f5b69b3bf5eb5674ec259b85e6308b3bb38b4c"
    ),
    "data/data_manifest.json": (
        "cede5d4e27757bcbc6e8ce33678e884c396bcef3812c90f791b6fe8d57636f42"
    ),
    "evaluation/scores/massive_en_dev__pi_base.json": (
        "cd92e7322280de40e846761556a20740d0f7173e9e6d3f44dc5858bbc59df0c3"
    ),
    "evaluation/base_development/summary.json": (
        "d2dc88532a8bfb6590b01fb983e3d0f6c6d8a3dd2df4e1462f542508fb5e3aee"
    ),
}

ORIGINAL_LOG_HASHES = {
    "massive_benefit_base_dev_237935.out": (
        "75442506bf58dcfb6b1e52ee2833f8a3b6064d887cb4b41e1de9f764e37ba657"
    ),
    "massive_benefit_base_dev_237935.err": (
        "2d827074abd99f23a50aebdd923118f8cba0406e05fc9305de450584af94c6f8"
    ),
    "massive_benefit_train_237936.out": (
        "72856bb8b89f05d3105247495dc655a4559c87fb10fe7eb68b2d75516c496203"
    ),
    "massive_benefit_train_237936.err": (
        "bdcd94b030713af898ae7b7abae8ecac27b59b0d860a8735a0efcb1981cfe6f4"
    ),
}

KNOWN_PRIOR_RECOVERY_LOG_HASHES = {
    "massive_benefit_infrastructure_recovery_v1_evaluate_239579.out": (
        "d5067da8f92031ec160bb1415fbcfe72aa0d701a8d5f151cb32cc298620dde32"
    ),
    "massive_benefit_infrastructure_recovery_v1_evaluate_239579.err": (
        "a919e4fe21459af26dad90bcbd0ffa5e37387103d147348e3df9bb61ce7799b7"
    ),
}

ORIGINAL_ACCOUNTING = {
    "base_dev": {
        "job_id": "237935",
        "state": "COMPLETED",
        "elapsed_seconds": 97,
        "time_limit_minutes": 30,
        "exit_code": "0:0",
        "allocated_h200": 1,
        "rounded_h200_minutes": 2,
    },
    "train": {
        "job_id": "237936",
        "state": "FAILED",
        "elapsed_seconds": 31,
        "time_limit_minutes": 90,
        "exit_code": "1:0",
        "allocated_h200": 1,
        "rounded_h200_minutes": 1,
    },
    "evaluate": {
        "job_id": "237937",
        "state": "CANCELLED",
        "elapsed_seconds": 0,
        "time_limit_minutes": 75,
        "exit_code": "0:0",
        "allocated_h200": 0,
        "rounded_h200_minutes": 0,
    },
}

PRIOR_RECOVERY_ACCOUNTING = {
    "train": {
        "job_id": "239578",
        "state": "COMPLETED",
        "elapsed_seconds": 4175,
        "time_limit_minutes": 90,
        "exit_code": "0:0",
        "allocated_h200": 1,
        "rounded_h200_minutes": 70,
    },
    "evaluate": {
        "job_id": "239579",
        "state": "TIMEOUT",
        "elapsed_seconds": 4515,
        "time_limit_minutes": 75,
        "exit_code": "0:0",
        "allocated_h200": 1,
        "rounded_h200_minutes": 76,
    },
}

PRIOR_CONTROL_RELATIVE_PATHS = (
    "jobs.tsv",
    "dispatch_attempt.tsv",
    "AUTHORIZED_INFRASTRUCTURE_RECOVERY_WITHIN_ORIGINAL_CAP.json",
    "SUBMITTED",
    "RELEASED",
    "TRAIN_STARTED",
    "EVALUATE_STARTED",
)

def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(payload, field="payload_sha256"):
    result = dict(payload)
    result.pop(field, None)
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_seal(payload, field="payload_sha256"):
    copy = dict(payload)
    observed = copy.pop(field, None)
    if observed != sha256_bytes(canonical_json_bytes(copy)):
        raise ValueError(f"Artifact seal mismatch ({field})")


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json_once(path, value, mode=0o400):
    destination = os.path.abspath(path)
    if os.path.lexists(destination):
        raise ValueError(f"Refusing to replace recovery artifact: {destination}")
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
        os.chmod(temporary, mode)
        os.link(temporary, destination)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def git(repo_root, *args, binary=False):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), *args], text=not binary
    )


def require_regular_hash(path, expected=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or unsafe immutable artifact: {path}")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(f"Immutable artifact hash drift: {path}")
    return {
        "size_bytes": path.stat().st_size,
        "sha256": observed,
    }


def verify_json_self_seal(path, field="payload_sha256"):
    payload = load_json(path)
    verify_seal(payload, field)
    return payload


def parse_key_value_file(path):
    result = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.rstrip("\n").partition("=")
            if not separator or not key or key in result:
                raise ValueError(f"Invalid key/value control artifact: {path}")
            result[key] = value
    return result


def audit_repair_commit(repo_root):
    repo_root = os.path.abspath(repo_root)
    repair_commit = git(repo_root, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", repair_commit):
        raise ValueError("Evaluation-recovery checkout does not resolve to a commit")
    parents = git(repo_root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2 or parents[1] != BASE_COMMIT:
        raise ValueError("Evaluation recovery must be a single-parent direct child")
    changed = frozenset(
        line
        for line in git(
            repo_root, "diff", "--name-only", BASE_COMMIT, repair_commit
        ).splitlines()
        if line
    )
    if changed != ALLOWED_REPAIR_PATHS:
        raise ValueError(
            "Evaluation-recovery path set differs; "
            f"missing={sorted(ALLOWED_REPAIR_PATHS - changed)}, "
            f"unauthorized={sorted(changed - ALLOWED_REPAIR_PATHS)}"
        )
    dirty = git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"Evaluation-recovery checkout is dirty:\n{dirty}")
    diff = git(
        repo_root, "diff", "--binary", BASE_COMMIT, repair_commit, binary=True
    )
    return {
        "repo_commit": repair_commit,
        "parent_commit": BASE_COMMIT,
        "changed_paths": sorted(changed),
        "diff_sha256": sha256_bytes(diff),
    }


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _statement_calls(statements):
    return {
        _call_name(node.func)
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    }


def sampler_runtime_contract(tree):
    """Prove failure evidence and engine teardown are on the live sampler path."""
    all_calls = {
        _call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    required_failure_calls = {
        "failure_evidence_path",
        "load_failure_evidence",
        "structured_validation_failure_payload",
        "write_or_audit_failure_evidence",
    }
    if not required_failure_calls <= all_calls:
        raise ValueError("Sampler lacks sealed structured-failure evidence")

    scoped_finally = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if (
            "LLM" in _statement_calls(node.body)
            and "shutdown_vllm_engine" in _statement_calls(node.finalbody)
        ):
            scoped_finally = True
            break
    if not scoped_finally:
        raise ValueError("Sampler does not shut down its allocated vLLM in finally")
    required_matcher_calls = {
        "GrammarMatcher",
        "GrammarCompiler",
        "from_huggingface",
        "audit_strict_xgrammar_contract",
    }
    if not required_matcher_calls <= all_calls:
        raise ValueError("Sampler lacks the pinned token-matcher preflight")
    positions = {
        name: [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node.func) == name
        ]
        for name in ("audit_strict_xgrammar_contract", "LLM")
    }
    if min(positions["audit_strict_xgrammar_contract"]) >= min(positions["LLM"]):
        raise ValueError("Pinned token-matcher preflight does not precede LLM allocation")
    return {
        "sealed_terminal_failure_evidence": True,
        "engine_shutdown_in_allocation_finally": True,
        "pinned_token_matcher_precedes_llm_allocation": True,
    }


class _ProfilePlumbingStripper(ast.NodeTransformer):
    """Remove only profile-provenance additions before parent-AST comparison."""

    PROFILE_NAMES = {
        "constraint_profile",
        "selection_profile",
        "expected_constraint_profile",
        "structured_constraint_profile",
    }

    @classmethod
    def _contains_profile_name(cls, node):
        return any(
            isinstance(child, ast.Name) and child.id in cls.PROFILE_NAMES
            for child in ast.walk(node)
        )

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults_start = len(positional) - len(node.args.defaults)
        kept_positional = []
        kept_defaults = []
        for index, argument in enumerate(positional):
            default = (
                node.args.defaults[index - defaults_start]
                if index >= defaults_start
                else None
            )
            if argument.arg == "expected_constraint_profile":
                continue
            kept_positional.append(argument)
            if default is not None:
                kept_defaults.append(default)
        posonly_count = len(node.args.posonlyargs)
        node.args.posonlyargs = kept_positional[:posonly_count]
        node.args.args = kept_positional[posonly_count:]
        node.args.defaults = kept_defaults
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        target_names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if target_names & {"constraint_profile", "selection_profile"}:
            return None
        if "profile" in target_names and self._contains_profile_name(node.value):
            return None
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if self._contains_profile_name(node.test):
            return None
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [
            keyword
            for keyword in node.keywords
            if keyword.arg != "expected_constraint_profile"
        ]
        return node

    def visit_Dict(self, node):
        node = self.generic_visit(node)
        pairs = [
            (key, value)
            for key, value in zip(node.keys, node.values)
            if not (
                isinstance(key, ast.Constant)
                and key.value == "structured_constraint_profile"
            )
        ]
        node.keys = [key for key, _ in pairs]
        node.values = [value for _, value in pairs]
        return node

    def visit_Set(self, node):
        node = self.generic_visit(node)
        node.elts = [
            element
            for element in node.elts
            if not (
                isinstance(element, ast.Constant)
                and element.value == "structured_constraint_profile"
            )
        ]
        return node


def _function_nodes(source):
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def frozen_function_contract(
    baseline_source,
    current_source,
    exact_names,
    profile_only_names=(),
    label="scientific script",
):
    baseline = _function_nodes(baseline_source)
    current = _function_nodes(current_source)
    evidence = {}
    for name in tuple(exact_names) + tuple(profile_only_names):
        if name not in baseline or name not in current:
            raise ValueError(f"{label} lacks frozen function {name}")
        left = copy.deepcopy(baseline[name])
        right = copy.deepcopy(current[name])
        if name in profile_only_names:
            stripper = _ProfilePlumbingStripper()
            left = stripper.visit(left)
            right = stripper.visit(right)
            ast.fix_missing_locations(left)
            ast.fix_missing_locations(right)
        left_dump = ast.dump(left, include_attributes=False)
        right_dump = ast.dump(right, include_attributes=False)
        if left_dump != right_dump:
            raise ValueError(f"{label} changed frozen function {name}")
        evidence[name] = sha256_bytes(left_dump.encode("utf-8"))
    return dict(sorted(evidence.items()))


def frozen_constant_contract(baseline_source, current_source, names, label):
    def assignments(source):
        result = {}
        for node in ast.parse(source).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
        return result

    baseline = assignments(baseline_source)
    current = assignments(current_source)
    evidence = {}
    for name in names:
        if name not in baseline or name not in current:
            raise ValueError(f"{label} lacks frozen constant {name}")
        left = ast.dump(baseline[name], include_attributes=False)
        right = ast.dump(current[name], include_attributes=False)
        if left != right:
            raise ValueError(f"{label} changed frozen constant {name}")
        evidence[name] = sha256_bytes(left.encode("utf-8"))
    return dict(sorted(evidence.items()))


def audit_scientific_contract(repo_root):
    evaluator_path = "scripts/evaluate_massive_benefit_generations.py"
    summarizer_path = "scripts/summarize_massive_benefit_pilot.py"
    evaluator_baseline = git(repo_root, "show", f"{BASE_COMMIT}:{evaluator_path}")
    evaluator_current = (Path(repo_root) / evaluator_path).read_text(encoding="utf-8")
    summarizer_baseline = git(repo_root, "show", f"{BASE_COMMIT}:{summarizer_path}")
    summarizer_current = (Path(repo_root) / summarizer_path).read_text(
        encoding="utf-8"
    )
    evaluator_functions = frozen_function_contract(
        evaluator_baseline,
        evaluator_current,
        exact_names=(
            "load_data_manifest",
            "normalize_value",
            "sample_sha256",
            "load_answers",
            "validate_prediction",
            "load_generations",
            "safe_ratio",
            "aggregate",
            "evaluate",
        ),
        profile_only_names=("compatible_endpoints", "main"),
        label="MASSIVE evaluator",
    )
    evaluator_constants = frozen_constant_contract(
        evaluator_baseline,
        evaluator_current,
        ("EXPECTED_SEED", "EXPECTED_MAX_NEW_TOKENS", "EXPECTED_MAX_CONTEXT"),
        "MASSIVE evaluator",
    )
    summarizer_functions = frozen_function_contract(
        summarizer_baseline,
        summarizer_current,
        exact_names=(
            "load_model_manifest",
            "percentile",
            "paired_bootstrap_interval",
            "one_sided_exact_mcnemar_p",
            "comparison",
            "gate",
            "frozen_thresholds",
            "parse_checkpoint",
        ),
        profile_only_names=(
            "load_evaluation",
            "validate_pair",
            "command_base",
            "command_select",
            "command_final",
        ),
        label="MASSIVE summarizer",
    )
    summarizer_constants = frozen_constant_contract(
        summarizer_baseline,
        summarizer_current,
        (
            "SELECTION_STEPS",
            "BOOTSTRAP_SEED",
            "BOOTSTRAP_REPLICATES",
            "MAX_BASE_ACCURACY",
            "MIN_CANDIDATE_INTENT",
            "MIN_INTENT_GAIN",
            "MAX_P",
            "MIN_SLOT_F1",
            "MIN_SLOT_F1_DELTA",
            "MIN_FRAME_EXACT",
            "MIN_FRAME_GAIN",
            "EXPECTED_DEV_N",
            "EXPECTED_TEST_N",
        ),
        "MASSIVE summarizer",
    )
    return {
        "parent_commit": BASE_COMMIT,
        "evaluator_frozen_function_ast_sha256": evaluator_functions,
        "evaluator_frozen_constant_ast_sha256": evaluator_constants,
        "summarizer_frozen_function_ast_sha256": summarizer_functions,
        "summarizer_frozen_constant_ast_sha256": summarizer_constants,
        "metric_gate_changes_allowed": False,
        "profile_provenance_plumbing_only": True,
    }


def audit_sampler_repair(repo_root):
    baseline = git(
        repo_root,
        "show",
        f"{BASE_COMMIT}:scripts/sample_massive_structured_generations.py",
    )
    path = Path(repo_root) / "scripts/sample_massive_structured_generations.py"
    current = path.read_text(encoding="utf-8")
    if current == baseline:
        raise ValueError("Sampler repair did not change the failed exception path")
    tree = ast.parse(current)
    calls = [_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    required_calls = {"StructuredOutputsConfig", "SamplingParams", "LLM", "chat"}
    if not required_calls <= set(calls):
        raise ValueError("Sampler repair removed the pinned vLLM decoding path")
    if (
        "--structured_constraint_profile" not in current
        or "enum_v1" not in current
        or "const_tree_v2" not in current
        or "structured_constraint_profile" not in current
        or 'backend="xgrammar", disable_fallback=True' not in current
    ):
        raise ValueError("Sampler lacks the sealed const-tree repair contract")
    sampler = _load_sampler(repo_root)
    intents = ["a", "b", "c"]
    slots = ["slot_a", "slot_b"]
    legacy = sampler.prediction_schema(
        intents,
        slots,
        endpoint="joint_json",
        structured_constraint_profile="enum_v1",
    )
    expected_legacy = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": intents},
            "slots": {
                "type": "array",
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": slots},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "slots"],
        "additionalProperties": False,
    }
    if legacy != expected_legacy:
        raise ValueError("Sampler enum_v1 profile is not legacy-compatible")
    repaired = sampler.prediction_schema(
        intents,
        slots,
        endpoint="joint_json",
        structured_constraint_profile="const_tree_v2",
    )
    repaired_bytes = canonical_json_bytes(repaired)
    if repaired == legacy or b'"anyOf"' not in repaired_bytes or b'"const"' not in repaired_bytes:
        raise ValueError("Sampler const_tree_v2 schema is not the repaired language")
    runtime_contract = sampler_runtime_contract(tree)
    stage = (
        Path(repo_root)
        / "scripts/stage_massive_benefit_evaluation_recovery_v1_tillicum.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "sample_massive_structured_generations.py",
        "--preflight_only",
        "--structured_constraint_profile const_tree_v2",
        "model_specs=(--model pi_base=BASE)",
        "for step in 15 30 60 90 150",
    ):
        if required not in stage:
            raise ValueError(f"Tillicum staging lacks sampler preflight: {required}")
    return {
        "baseline_commit": BASE_COMMIT,
        "baseline_sha256": sha256_bytes(baseline.encode("utf-8")),
        "repaired_sha256": sha256_bytes(current.encode("utf-8")),
        "legacy_profile": "enum_v1",
        "recovery_profile": "const_tree_v2",
        "legacy_schema_sha256": sha256_bytes(canonical_json_bytes(legacy)),
        "recovery_schema_sha256": sha256_bytes(canonical_json_bytes(repaired)),
        "constraint_encoding_changed": True,
        "accepted_language_intended_identical": True,
        "decoder_path_comparability_requires_symmetric_rerun": True,
        "old_generations_may_be_reused": False,
        "tillicum_all_adapter_preflight_required": True,
        **runtime_contract,
    }


def audit_original_artifacts(repo_root, output_root, logs_root):
    output_root = Path(output_root)
    logs_root = Path(logs_root)
    for relative, expected in ORIGINAL_ARTIFACT_HASHES.items():
        require_regular_hash(output_root / relative, expected)
    for filename, expected in ORIGINAL_LOG_HASHES.items():
        require_regular_hash(logs_root / filename, expected)
    require_regular_hash(
        Path(repo_root) / "configs/training_qwen25_7b_massive_benefit_pilot.yaml",
        TRAINING_CONFIG_SHA256,
    )
    auth = verify_json_self_seal(
        output_root / "control/AUTHORIZED_MAX_COST_USD_2.93.json"
    )
    if (
        auth.get("repo_commit") != ORIGINAL_COMMIT
        or auth.get("maximum_h200_minutes") != ORIGINAL_MAX_H200_MINUTES
        or Decimal(str(auth.get("maximum_cost_usd"))) != ORIGINAL_MAX_COST_USD
        or auth.get("no_retries_or_reserve") is not True
        or auth.get("automatic_medical_union_or_quorum") is not False
    ):
        raise ValueError("Original MASSIVE authorization differs")
    base_summary = load_json(
        output_root / "evaluation/base_development/summary.json"
    )
    if (
        base_summary.get("decision") != "GO"
        or base_summary.get("evaluation_sha256")
        != ORIGINAL_ARTIFACT_HASHES[
            "evaluation/scores/massive_en_dev__pi_base.json"
        ]
        or base_summary.get("base_joint_json_intent_accuracy")
        != 0.6627277203348104
        or not all(base_summary.get("checks", {}).values())
    ):
        raise ValueError("Frozen MASSIVE base development gate differs")
    manifest = verify_json_self_seal(
        output_root / "data/data_manifest.json", "manifest_payload_sha256"
    )
    if (
        manifest.get("training_subset", {}).get("selected_rows") != 1122
        or manifest.get("evaluation", {}).get("dev_rows") != 2031
        or manifest.get("evaluation", {}).get("sealed_test_rows") != 2965
        or manifest.get("medical_overlap_audit", {}).get(
            "selected_training_rows_medical_like"
        )
        != 0
    ):
        raise ValueError("Frozen MASSIVE data manifest differs")
    return {
        "artifacts_sha256": dict(sorted(ORIGINAL_ARTIFACT_HASHES.items())),
        "logs_sha256": dict(sorted(ORIGINAL_LOG_HASHES.items())),
        "training_dataset_fingerprint": manifest["training_subset"][
            "dataset_fingerprint"
        ],
    }


def parse_allocated_h200(alloc_tres):
    if not alloc_tres:
        return 0
    values = {}
    for token in alloc_tres.split(","):
        if "=" in token:
            key, value = token.rsplit("=", 1)
            values[key] = value
    value = values.get("gres/gpu:h200", "0")
    if not value.isdigit():
        raise ValueError(f"Invalid H200 accounting record: {alloc_tres}")
    return int(value)


def parse_accounting_line(line):
    fields = line.rstrip("\n").split("|")
    if len(fields) != 8:
        raise ValueError(f"Unexpected sacct row width: {line!r}")
    job_id, state, elapsed, limit, allocation, exit_code, start, end = fields
    if not job_id.isdigit() or not elapsed.isdigit() or not limit.isdigit():
        raise ValueError(f"Invalid numeric field in sacct row: {line!r}")
    return {
        "job_id": job_id,
        "state": state,
        "elapsed_seconds": int(elapsed),
        "time_limit_minutes": int(limit),
        "alloc_tres": allocation,
        "exit_code": exit_code,
        "start": start,
        "end": end,
        "allocated_h200": parse_allocated_h200(allocation),
        "rounded_h200_minutes": math.ceil(int(elapsed) / 60),
    }


def read_accounting_row(job_id):
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "--starttime",
            "2026-08-17",
            "--jobs",
            str(job_id),
            "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES,ExitCode,Start,End",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        if line.strip():
            row = parse_accounting_line(line)
            if row["job_id"] == str(job_id):
                rows.append(row)
    if len(rows) != 1:
        raise ValueError(f"Expected one top-level accounting row for {job_id}")
    return rows[0]


def accounting_matches(observed, expected):
    for key in (
        "job_id",
        "elapsed_seconds",
        "time_limit_minutes",
        "exit_code",
        "allocated_h200",
        "rounded_h200_minutes",
    ):
        if observed.get(key) != expected[key]:
            return False
    if expected["state"] == "CANCELLED":
        return observed.get("state", "").startswith("CANCELLED")
    return observed.get("state") == expected["state"]


def canonical_prior_accounting():
    rows = []
    for scope, definitions in (
        ("original", ORIGINAL_ACCOUNTING),
        ("infrastructure_recovery_v1", PRIOR_RECOVERY_ACCOUNTING),
    ):
        for stage, expected in definitions.items():
            rows.append({"scope": scope, "stage": stage, **dict(expected)})
    return rows


def audit_prior_accounting():
    rows = []
    for expected in canonical_prior_accounting():
        observed = read_accounting_row(expected["job_id"])
        if not accounting_matches(observed, expected):
            raise ValueError(
                f"Prior accounting differs for {expected['job_id']}: {observed!r}"
            )
        rows.append(
            {
                **expected,
                "start": observed["start"],
                "end": observed["end"],
            }
        )
    total = sum(row["rounded_h200_minutes"] for row in rows)
    if total != PRIOR_ROUNDED_H200_MINUTES:
        raise ValueError(f"Prior conservative accounting is {total}, expected 149")
    return rows


def model_inventory(model_dir):
    ignored = {"MODEL_MANIFEST.json", "TRAIN_COMPLETE"}
    entries = []
    for directory, dirnames, filenames in os.walk(model_dir):
        dirnames.sort()
        for dirname in dirnames:
            path = os.path.join(directory, dirname)
            if os.path.islink(path) or not os.path.isdir(path):
                raise ValueError(
                    "Model inventory has an unsafe directory entry: "
                    f"{os.path.relpath(path, model_dir)}"
                )
        for filename in sorted(filenames):
            relative = os.path.relpath(os.path.join(directory, filename), model_dir)
            if relative in ignored:
                continue
            path = os.path.join(directory, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"Model inventory has unsafe entry: {relative}")
            entries.append(
                {
                    "path": relative,
                    "size_bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
            )
    return entries


def adapter_fingerprint(path):
    paths = [os.path.join(path, "adapter_config.json")]
    weights = [
        candidate
        for candidate in (
            os.path.join(path, "adapter_model.safetensors"),
            os.path.join(path, "adapter_model.bin"),
        )
        if os.path.isfile(candidate)
    ]
    if not os.path.isfile(paths[0]) or len(weights) != 1:
        raise ValueError(f"Adapter artifacts differ: {path}")
    entries = []
    for artifact in paths + weights:
        if os.path.islink(artifact):
            raise ValueError(f"Adapter artifact is a symlink: {artifact}")
        entries.append(
            {
                "name": os.path.basename(artifact),
                "size_bytes": os.path.getsize(artifact),
                "sha256": sha256_file(artifact),
            }
        )
    return sha256_bytes(canonical_json_bytes(entries))


def audit_prior_recovery(output_root, logs_root):
    output_root = Path(output_root)
    logs_root = Path(logs_root)
    root = output_root / "control/infrastructure_recovery_v1"
    records = {}
    for relative in PRIOR_CONTROL_RELATIVE_PATHS:
        records[relative] = require_regular_hash(root / relative)
    lock_owner = output_root / "control/INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK/owner"
    records["../INFRASTRUCTURE_RECOVERY_V1_SUBMISSION_LOCK/owner"] = (
        require_regular_hash(lock_owner)
    )
    with open(root / "jobs.tsv", newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    if jobs != [
        {"stage": "train", "job_id": "239578", "max_minutes": "90"},
        {"stage": "evaluate", "job_id": "239579", "max_minutes": "75"},
    ]:
        raise ValueError("Infrastructure-recovery jobs record differs")
    addendum_path = root / "AUTHORIZED_INFRASTRUCTURE_RECOVERY_WITHIN_ORIGINAL_CAP.json"
    addendum = verify_json_self_seal(addendum_path)
    if (
        addendum.get("recovery_id") != "massive_benefit_infrastructure_recovery_v1"
        or addendum.get("repair", {}).get("repo_commit")
        != INFRASTRUCTURE_RECOVERY_COMMIT
        or addendum.get("repair", {}).get("parent_commit") != ORIGINAL_COMMIT
        or addendum.get("budget", {}).get("cumulative_max_h200_minutes") != 168
        or addendum.get("constraints", {}).get("exact_once") is not True
        or addendum.get("constraints", {}).get("no_requeue") is not True
        or addendum.get("constraints", {}).get("no_additional_recovery_or_retry")
        is not True
    ):
        raise ValueError("Infrastructure-recovery sealed addendum differs")
    expected_jobs = [
        {"stage": "train", "job_id": "239578", "max_minutes": 90},
        {"stage": "evaluate", "job_id": "239579", "max_minutes": 75},
    ]
    if addendum.get("recovery_jobs") != expected_jobs:
        raise ValueError("Infrastructure-recovery addendum jobs differ")
    train_started = parse_key_value_file(root / "TRAIN_STARTED")
    evaluate_started = parse_key_value_file(root / "EVALUATE_STARTED")
    if train_started.get("job_id") != "239578" or evaluate_started.get("job_id") != "239579":
        raise ValueError("Infrastructure-recovery start records differ")
    if (
        train_started.get("repair_repo_commit") != INFRASTRUCTURE_RECOVERY_COMMIT
        or evaluate_started.get("repair_repo_commit")
        != INFRASTRUCTURE_RECOVERY_COMMIT
    ):
        raise ValueError("Infrastructure-recovery start commit differs")
    log_records = {}
    for stage, job_id in (("train", "239578"), ("evaluate", "239579")):
        for stream in ("out", "err"):
            name = (
                "massive_benefit_infrastructure_recovery_v1_"
                f"{stage}_{job_id}.{stream}"
            )
            expected = KNOWN_PRIOR_RECOVERY_LOG_HASHES.get(name)
            log_records[name] = require_regular_hash(logs_root / name, expected)
    for relative in (
        "selection/summary.json",
        "sealed_final/summary.json",
    ):
        if os.path.lexists(output_root / "evaluation/infrastructure_recovery_v1" / relative):
            raise ValueError(f"Failed recovery unexpectedly reached {relative}")
    for sentinel in (
        "GO_MASSIVE_SEALED_TEST",
        "STOPPED_MASSIVE_SELECTION",
        "GO_MASSIVE_BENEFIT_ONLY",
        "STOPPED_MASSIVE_FINAL",
    ):
        if os.path.lexists(root / sentinel):
            raise ValueError(f"Failed recovery unexpectedly wrote {sentinel}")
    return {
        "control_artifacts": dict(sorted(records.items())),
        "logs": dict(sorted(log_records.items())),
        "addendum_sha256": records[
            "AUTHORIZED_INFRASTRUCTURE_RECOVERY_WITHIN_ORIGINAL_CAP.json"
        ]["sha256"],
        "local_model_snapshot": addendum["local_model_snapshot"],
    }


def audit_prior_model(output_root, prior):
    output_root = Path(output_root)
    model_dir = output_root / "model/massive_en_benefit_pilot_infrastructure_recovery_v1"
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("Completed infrastructure-recovery model is missing or unsafe")
    manifest_path = model_dir / "MODEL_MANIFEST.json"
    manifest = verify_json_self_seal(manifest_path, "manifest_payload_sha256")
    if (
        manifest.get("recovery_id") != "massive_benefit_infrastructure_recovery_v1"
        or manifest.get("repair_repo_commit") != INFRASTRUCTURE_RECOVERY_COMMIT
        or manifest.get("recovery_addendum_sha256") != prior["addendum_sha256"]
        or manifest.get("data_manifest_sha256")
        != ORIGINAL_ARTIFACT_HASHES["data/data_manifest.json"]
        or manifest.get("training_config_sha256") != TRAINING_CONFIG_SHA256
        or manifest.get("canonical_base_model") != MODEL_ID
        or manifest.get("base_model_revision") != MODEL_REVISION
        or manifest.get("completion_only") is not True
        or manifest.get("all_checkpoint_steps") != list(range(15, 151, 15))
        or manifest.get("selection_checkpoint_steps") != list(SELECTION_STEPS)
    ):
        raise ValueError("Completed recovery model manifest provenance differs")
    if manifest.get("base_model_load", {}).get("weight_shard_artifacts") != prior[
        "local_model_snapshot"
    ].get("weight_shard_artifacts"):
        raise ValueError("Model manifest local base shards differ from sealed addendum")
    inventory = model_inventory(model_dir)
    if inventory != manifest.get("model_inventory"):
        raise ValueError("Completed recovery model inventory differs")
    fingerprints = manifest.get("checkpoint_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        str(step) for step in SELECTION_STEPS
    }:
        raise ValueError("Selection checkpoint fingerprints differ")
    for step in SELECTION_STEPS:
        observed = adapter_fingerprint(model_dir / f"checkpoint-{step}")
        if observed != fingerprints[str(step)]:
            raise ValueError(f"Checkpoint {step} fingerprint differs")
    complete_path = model_dir / "TRAIN_COMPLETE"
    complete = parse_key_value_file(complete_path)
    if (
        complete.get("job_id") != "239578"
        or complete.get("repair_repo_commit") != INFRASTRUCTURE_RECOVERY_COMMIT
        or complete.get("model_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("Completed recovery training seal differs")
    return {
        "model_manifest": require_regular_hash(manifest_path),
        "train_complete": require_regular_hash(complete_path),
        "checkpoint_fingerprints": dict(sorted(fingerprints.items())),
        "model_inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
    }


def _load_sampler(repo_root):
    import importlib.util

    path = Path(repo_root) / "scripts/sample_massive_structured_generations.py"
    spec = importlib.util.spec_from_file_location("massive_recovery_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generation_filename(step, endpoint):
    suffix = "" if endpoint == "joint_json" else "__intent_only"
    return f"massive_en_dev__step_{step}{suffix}.json"


def audit_failed_generations(repo_root, output_root):
    sampler = _load_sampler(repo_root)
    output_root = Path(output_root)
    source = output_root / "evaluation/infrastructure_recovery_v1/generations"
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Failed development generation directory is unsafe")
    expected_names = {
        generation_filename(step, endpoint)
        for step in FAILED_COMPLETE_STEPS
        for endpoint in ENDPOINTS
    }
    entries = list(source.iterdir())
    observed_names = {entry.name for entry in entries}
    if observed_names != expected_names:
        raise ValueError(
            f"Failed generation inventory differs: {sorted(observed_names)}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Failed generation entry is unsafe: {entry}")
    prompt_path = output_root / "data/dev/prompts.json"
    meta, prompts = sampler.load_prompt_bank(prompt_path)
    model_dir = output_root / "model/massive_en_benefit_pilot_infrastructure_recovery_v1"
    artifacts = {}
    for step in FAILED_COMPLETE_STEPS:
        model_path = os.path.abspath(model_dir / f"checkpoint-{step}")
        model_name = f"step_{step}"
        model_fingerprint = sampler.adapter_fingerprint(model_path)
        for endpoint in ENDPOINTS:
            schema = sampler.prediction_schema(
                meta["intent_labels"], meta["slot_labels"], endpoint=endpoint
            )
            run = {
                "schema_version": 1,
                "generator": "vllm_xgrammar_json",
                "endpoint": endpoint,
                "set_name": meta["set_name"],
                "role": meta["role"],
                "model_name": model_name,
                "model_path": model_path,
                "model_fingerprint": model_fingerprint,
                "base_model": MODEL_ID,
                "base_model_revision": MODEL_REVISION,
                "prompt_file_sha256": sampler.sha256_file(prompt_path),
                "question_ids": [record["question_id"] for record in prompts],
                "prompt_sha256": [record["prompt_sha256"] for record in prompts],
                "ontology_sha256": meta["ontology_sha256"],
                "json_schema_sha256": sampler.sha256_bytes(
                    sampler.canonical_json_bytes(schema)
                ),
                "structured_backend": "xgrammar",
                "vllm_version": "0.11.2",
                "xgrammar_version": "0.1.25",
                "temperature": 0.0,
                "n_samples": 1,
                "max_new_tokens": 256,
                "max_context": 2048,
                "seed": 8172026,
                "same_prompt_all_models": True,
                "selection_uses_joint_json_only": True,
            }
            fingerprint = sampler.sha256_bytes(sampler.canonical_json_bytes(run))
            path = source / generation_filename(step, endpoint)
            if not sampler.output_is_complete(
                path,
                run,
                fingerprint,
                prompts,
                meta["intent_labels"],
                meta["slot_labels"],
                endpoint,
            ):
                raise ValueError(f"Failed-run generation is incomplete: {path}")
            payload = load_json(path)
            truncated = sum(
                sample.get("stop_reason") == "max_new_tokens"
                for sample in payload["samples"]
            )
            if truncated:
                raise ValueError(f"Failed-run generation has {truncated} truncations: {path}")
            artifacts[path.name] = {
                **require_regular_hash(path),
                "rows": len(payload["samples"]),
                "endpoint": endpoint,
                "model_name": model_name,
                "generation_fingerprint": fingerprint,
            }
    return dict(sorted(artifacts.items()))


def _load_summarizer(repo_root):
    import importlib.util

    path = Path(repo_root) / "scripts/summarize_massive_benefit_pilot.py"
    spec = importlib.util.spec_from_file_location(
        "massive_recovery_summarizer", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_infrastructure_auditor(repo_root):
    import importlib.util

    path = (
        Path(repo_root)
        / "scripts/audit_massive_benefit_infrastructure_recovery_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "massive_infrastructure_recovery_auditor", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_live_local_snapshot(repo_root, output_root, expected):
    output_root = Path(os.path.abspath(output_root))
    if output_root.name != "massive_benefit_pilot_v1" or output_root.parent.name != "outputs":
        raise ValueError("MASSIVE output root does not have the sealed Tillicum layout")
    tillicum_root = output_root.parent.parent
    infrastructure = _load_infrastructure_auditor(repo_root)
    snapshot_path = infrastructure.expected_local_snapshot(tillicum_root)
    return infrastructure.verify_local_snapshot_binding(
        expected, snapshot_path, tillicum_root
    )


def _require_exact_regular_inventory(directory, expected_names, description):
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{description} directory is missing or unsafe")
    entries = list(directory.iterdir())
    observed_names = {entry.name for entry in entries}
    if observed_names != set(expected_names):
        raise ValueError(
            f"{description} inventory differs: {sorted(observed_names)}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"{description} entry is unsafe: {entry}")


def _phase_model_specs(output_root, phase):
    model_dir = (
        Path(output_root)
        / "model/massive_en_benefit_pilot_infrastructure_recovery_v1"
    )
    development = [("pi_base", "BASE")]
    development.extend(
        (f"step_{step}", os.path.abspath(model_dir / f"checkpoint-{step}"))
        for step in SELECTION_STEPS
    )
    if phase == "development":
        return [("dev", development)]
    if phase != "sealed-test":
        raise ValueError(f"Unknown evaluation-recovery phase: {phase}")

    selection_path = (
        Path(output_root)
        / "evaluation/evaluation_recovery_v1/selection/summary.json"
    )
    selection = load_json(selection_path)
    verify_seal(selection, "decision_payload_sha256")
    selected = selection.get("selected", {})
    step = selected.get("step")
    name = selected.get("model_name")
    if (
        selection.get("phase") != "development_selection"
        or selection.get("decision") != "GO"
        or selection.get("structured_constraint_profile") != "const_tree_v2"
        or selected.get("structured_constraint_profile") != "const_tree_v2"
        or step not in SELECTION_STEPS
        or name != f"step_{step}"
    ):
        raise ValueError("Cleaned-test provenance lacks a valid development GO")
    selected_path = os.path.abspath(model_dir / f"checkpoint-{step}")
    return [
        ("dev", development),
        ("sealed_test", [("pi_base", "BASE"), (name, selected_path)]),
    ]


def audit_recovery_evaluation_phase(repo_root, output_root, phase):
    """Bind v2 generation provenance through every score consumed by a gate."""
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    sampler = _load_sampler(repo_root)
    summarizer = _load_summarizer(repo_root)
    eval_root = output_root / "evaluation/evaluation_recovery_v1"
    generation_root = eval_root / "generations"
    score_root = eval_root / "scores"
    model_sets = _phase_model_specs(output_root, phase)

    manifest_path = output_root / "data/data_manifest.json"
    manifest = verify_json_self_seal(manifest_path, "manifest_payload_sha256")
    inventory = {
        entry["path"]: entry for entry in manifest.get("file_inventory", [])
    }
    if len(inventory) != len(manifest.get("file_inventory", [])):
        raise ValueError("MASSIVE data manifest inventory has duplicate paths")

    generation_names = set()
    score_names = set()
    records = []
    for data_subdir, models in model_sets:
        set_name = "massive_en_dev" if data_subdir == "dev" else "massive_en_test"
        expected_role = (
            "checkpoint_selection" if data_subdir == "dev" else "sealed_final"
        )
        prompt_path = output_root / f"data/{data_subdir}/prompts.json"
        answers_path = output_root / f"data/{data_subdir}/answers.json"
        prompt_entry = inventory.get(f"{data_subdir}/prompts.json")
        answers_entry = inventory.get(f"{data_subdir}/answers.json")
        if (
            prompt_entry is None
            or answers_entry is None
            or prompt_entry.get("sha256") != sha256_file(prompt_path)
            or answers_entry.get("sha256") != sha256_file(answers_path)
        ):
            raise ValueError(f"{set_name} prompt/answer bytes differ from manifest")
        prompt_meta, prompts = sampler.load_prompt_bank(prompt_path)
        if (
            prompt_meta.get("set_name") != set_name
            or prompt_meta.get("role") != expected_role
        ):
            raise ValueError(f"{set_name} prompt role differs")

        for model_name, model_path in models:
            model_fingerprint = sampler.adapter_fingerprint(model_path)
            generation_artifacts = {}
            for endpoint in ENDPOINTS:
                suffix = "" if endpoint == "joint_json" else "__intent_only"
                filename = f"{set_name}__{model_name}{suffix}.json"
                generation_names.add(filename)
                path = generation_root / filename
                schema = sampler.prediction_schema(
                    prompt_meta["intent_labels"],
                    prompt_meta["slot_labels"],
                    endpoint=endpoint,
                    structured_constraint_profile="const_tree_v2",
                )
                run = {
                    "schema_version": 1,
                    "generator": "vllm_xgrammar_json",
                    "endpoint": endpoint,
                    "set_name": set_name,
                    "role": expected_role,
                    "model_name": model_name,
                    "model_path": model_path,
                    "model_fingerprint": model_fingerprint,
                    "base_model": MODEL_ID,
                    "base_model_revision": MODEL_REVISION,
                    "prompt_file_sha256": sha256_file(prompt_path),
                    "question_ids": [record["question_id"] for record in prompts],
                    "prompt_sha256": [record["prompt_sha256"] for record in prompts],
                    "ontology_sha256": prompt_meta["ontology_sha256"],
                    "json_schema_sha256": sampler.sha256_bytes(
                        sampler.canonical_json_bytes(schema)
                    ),
                    "structured_backend": "xgrammar",
                    "vllm_version": "0.11.2",
                    "xgrammar_version": "0.1.25",
                    "temperature": 0.0,
                    "n_samples": 1,
                    "max_new_tokens": 256,
                    "max_context": 2048,
                    "seed": 8172026,
                    "same_prompt_all_models": True,
                    "selection_uses_joint_json_only": True,
                    "structured_constraint_profile": "const_tree_v2",
                }
                fingerprint = sampler.sha256_bytes(
                    sampler.canonical_json_bytes(run)
                )
                if not sampler.output_is_complete(
                    path,
                    run,
                    fingerprint,
                    prompts,
                    prompt_meta["intent_labels"],
                    prompt_meta["slot_labels"],
                    endpoint,
                ):
                    raise ValueError(f"Recovery generation audit failed: {path}")
                generation_artifacts[endpoint] = {
                    **require_regular_hash(path),
                    "generation_fingerprint": fingerprint,
                    "json_schema_sha256": run["json_schema_sha256"],
                }

            score_name = f"{set_name}__{model_name}.json"
            score_names.add(score_name)
            score_path = score_root / score_name
            score = summarizer.load_evaluation(
                score_path,
                expected_role=expected_role,
                expected_n=len(prompts),
                expected_constraint_profile="const_tree_v2",
            )
            score_meta = score["meta"]
            expected_score_fields = {
                "set_name": set_name,
                "role": expected_role,
                "model_name": model_name,
                "model_fingerprint": model_fingerprint,
                "base_model": MODEL_ID,
                "base_model_revision": MODEL_REVISION,
                "answers_file_sha256": sha256_file(answers_path),
                "data_manifest_sha256": sha256_file(manifest_path),
                "data_manifest_payload_sha256": manifest[
                    "manifest_payload_sha256"
                ],
                "joint_generations_file_sha256": generation_artifacts[
                    "joint_json"
                ]["sha256"],
                "intent_generations_file_sha256": generation_artifacts[
                    "intent_only"
                ]["sha256"],
                "evaluator_script_sha256": sha256_file(
                    repo_root / "scripts/evaluate_massive_benefit_generations.py"
                ),
                "prompt_file_sha256": sha256_file(prompt_path),
                "ontology_sha256": prompt_meta["ontology_sha256"],
                "inference_seed": 8172026,
                "temperature": 0.0,
                "n_samples": 1,
                "max_new_tokens": 256,
                "max_context": 2048,
                "selection_metric_endpoint": "joint_json",
                "intent_only_is_sensitivity_only": True,
                "slot_metric_is_official_bio_f1": False,
                "structured_constraint_profile": "const_tree_v2",
            }
            for field, expected in expected_score_fields.items():
                if score_meta.get(field) != expected:
                    raise ValueError(
                        f"Recovery score {score_name} differs on {field}"
                    )
            records.append(
                {
                    "set_name": set_name,
                    "role": expected_role,
                    "model_name": model_name,
                    "model_fingerprint": model_fingerprint,
                    "structured_constraint_profile": "const_tree_v2",
                    "generations": generation_artifacts,
                    "score": require_regular_hash(score_path),
                }
            )

    _require_exact_regular_inventory(
        generation_root, generation_names, "Recovery generation"
    )
    _require_exact_regular_inventory(score_root, score_names, "Recovery score")
    return {
        "phase": phase,
        "structured_constraint_profile": "const_tree_v2",
        "accepted_language_intended_identical": True,
        "all_scores_bound_to_profiled_generation_bytes": True,
        "records": records,
    }


def parse_recovery_jobs(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or unsafe evaluation-recovery jobs file")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = [
        {
            "stage": "evaluate",
            "job_id": rows[0].get("job_id", "") if len(rows) == 1 else "",
            "max_minutes": str(EVALUATION_MAX_H200_MINUTES),
        }
    ]
    if len(rows) != 1 or rows != expected or not re.fullmatch(
        r"[0-9]+", rows[0]["job_id"]
    ):
        raise ValueError("Evaluation recovery must contain one 15-minute job")
    return [
        {
            "stage": "evaluate",
            "job_id": rows[0]["job_id"],
            "max_minutes": EVALUATION_MAX_H200_MINUTES,
        }
    ]


def build_addendum(
    args,
    created_at=None,
    audit_live_accounting=True,
    audit_live_snapshot=True,
):
    history = audit_repair_commit(args.repo_root)
    sampler = audit_sampler_repair(args.repo_root)
    scientific_contract = audit_scientific_contract(args.repo_root)
    original = audit_original_artifacts(
        args.repo_root, args.output_root, args.logs_root
    )
    prior = audit_prior_recovery(args.output_root, args.logs_root)
    live_snapshot = (
        audit_live_local_snapshot(
            args.repo_root, args.output_root, prior["local_model_snapshot"]
        )
        if audit_live_snapshot
        else prior["local_model_snapshot"]
    )
    accounting = (
        audit_prior_accounting()
        if audit_live_accounting
        else canonical_prior_accounting()
    )
    model = audit_prior_model(args.output_root, prior)
    generations = audit_failed_generations(args.repo_root, args.output_root)
    jobs = parse_recovery_jobs(args.jobs_file)
    if (
        CUMULATIVE_MAX_H200_MINUTES >= ORIGINAL_MAX_H200_MINUTES
        or CONTINGENCY_MAX_H200_MINUTES > ORIGINAL_MAX_H200_MINUTES
        or CONTINGENCY_MAX_COST_USD > ORIGINAL_MAX_COST_USD
    ):
        raise ValueError("Evaluation recovery does not fit original authorization")
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": created_at
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": "structured_decoder_ontology_escape_then_missing_engine_teardown",
        "repair": history,
        "sampler_repair": sampler,
        "scientific_contract": scientific_contract,
        "original_artifacts": original,
        "prior_recovery": prior,
        "current_local_model_snapshot_verification": {
            "matches_prior_recovery_addendum": True,
            "snapshot_binding_sha256": sha256_bytes(
                canonical_json_bytes(live_snapshot)
            ),
            "revision": live_snapshot["revision"],
            "local_path": live_snapshot["local_path"],
            "weight_shards": live_snapshot["weight_shards"],
        },
        "prior_accounting": accounting,
        "trained_model": model,
        "failed_run_development_generations": generations,
        "recovery_jobs": jobs,
        "budget": {
            "h200_rate_per_hour_usd": str(H200_RATE_PER_HOUR_USD),
            "prior_rounded_h200_minutes": PRIOR_ROUNDED_H200_MINUTES,
            "prior_rounding_by_job": {
                "237935": 2,
                "237936": 1,
                "237937": 0,
                "239578": 70,
                "239579": 76,
            },
            "new_evaluation_max_h200_minutes": EVALUATION_MAX_H200_MINUTES,
            "cumulative_max_h200_minutes": CUMULATIVE_MAX_H200_MINUTES,
            "cumulative_max_cost_usd": f"{CUMULATIVE_MAX_COST_USD:.3f}",
            "termination_overhead_contingency_minutes": (
                TERMINATION_OVERHEAD_CONTINGENCY_MINUTES
            ),
            "contingency_max_h200_minutes": CONTINGENCY_MAX_H200_MINUTES,
            "contingency_max_cost_usd": f"{CONTINGENCY_MAX_COST_USD:.3f}",
            "original_authorized_max_h200_minutes": ORIGINAL_MAX_H200_MINUTES,
            "original_authorized_max_cost_usd": f"{ORIGINAL_MAX_COST_USD:.3f}",
        },
        "frozen_scope": {
            "prior_base_job_evidence": "237935",
            "prior_base_score_evidence_sha256": ORIGINAL_ARTIFACT_HASHES[
                "evaluation/scores/massive_en_dev__pi_base.json"
            ],
            "reuse_training_job": "239578",
            "old_generation_steps_preserved_not_reused": list(
                FAILED_COMPLETE_STEPS
            ),
            "fresh_development_models": [
                "pi_base",
                "step_15",
                "step_30",
                "step_60",
                "step_90",
                "step_150",
            ],
            "selection_steps": list(SELECTION_STEPS),
            "checkpoint_selection_rule_changed": False,
            "sealed_test_opens_only_after_development_go": True,
            "training_rerun": False,
            "symmetric_base_development_regeneration_required": True,
        },
        "constraints": {
            "held_first": True,
            "exact_once": True,
            "no_requeue": True,
            "one_h200": True,
            "no_further_retry_or_recovery": True,
            "no_training": True,
            "no_separate_base_development_job": True,
            "no_extra_adapter": True,
            "no_medical_union": True,
            "no_quorum": True,
            "preserve_prior_namespaces": True,
        },
    }


def command_verify_preflight(args):
    # A synthetic one-row job record lets the complete evidence path run before
    # any Slurm state is created.
    with tempfile.TemporaryDirectory() as temporary:
        jobs = Path(temporary) / "jobs.tsv"
        jobs.write_text(
            "stage\tjob_id\tmax_minutes\nevaluate\t999999999\t15\n",
            encoding="utf-8",
        )
        args.jobs_file = os.fspath(jobs)
        payload = build_addendum(args)
    print(
        "Evaluation-recovery preflight passed: "
        f"commit={payload['repair']['repo_commit']} "
        f"cumulative_h200_minutes={CUMULATIVE_MAX_H200_MINUTES}"
    )


def command_write_addendum(args):
    payload = sealed(build_addendum(args))
    atomic_write_json_once(args.output_file, payload)
    print(args.output_file)


def verify_addendum(args):
    path = Path(args.addendum_file)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or unsafe evaluation-recovery addendum")
    observed = load_json(path)
    verify_seal(observed)
    expected = sealed(
        build_addendum(
            args,
            created_at=observed.get("created_at"),
            audit_live_accounting=True,
            audit_live_snapshot=False,
        )
    )
    if observed != expected:
        raise ValueError("Sealed evaluation-recovery addendum differs from evidence")
    return observed


def parse_time_limit(value):
    if re.fullmatch(r"\d+-\d\d:\d\d:\d\d", value):
        days, clock = value.split("-", 1)
    elif re.fullmatch(r"\d\d:\d\d:\d\d", value):
        days, clock = "0", value
    else:
        raise ValueError(f"Unsupported Slurm TimeLimit: {value}")
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    return math.ceil((int(days) * 86400 + hours * 3600 + minutes * 60 + seconds) / 60)


def command_verify_control(args):
    verify_addendum(args)
    print(args.addendum_file)


def command_verify_job(args):
    addendum = verify_addendum(args)
    if parse_time_limit(args.time_limit) != EVALUATION_MAX_H200_MINUTES:
        raise ValueError("Running evaluation TimeLimit differs from sealed cap")
    if addendum["recovery_jobs"] != [
        {
            "stage": "evaluate",
            "job_id": args.job_id,
            "max_minutes": EVALUATION_MAX_H200_MINUTES,
        }
    ]:
        raise ValueError("Running job is not uniquely authorized")
    print(f"Authorized evaluation-only recovery job {args.job_id}")


def command_verify_snapshot(args):
    addendum = verify_addendum(args)
    expected = addendum["prior_recovery"]["local_model_snapshot"]
    observed = audit_live_local_snapshot(
        args.repo_root, args.output_root, expected
    )
    digest = sha256_bytes(canonical_json_bytes(observed))
    if digest != addendum["current_local_model_snapshot_verification"].get(
        "snapshot_binding_sha256"
    ):
        raise ValueError("Live local-model snapshot differs from recovery addendum")
    print(f"Pinned local-model snapshot verified: {digest}")


def command_verify_phase(args):
    addendum = verify_addendum(args)
    relative = (
        "base_development/decoder_provenance_audit.json"
        if args.phase == "development"
        else "sealed_final/decoder_provenance_audit.json"
    )
    expected_output = os.path.abspath(
        os.path.join(
            args.output_root,
            "evaluation/evaluation_recovery_v1",
            relative,
        )
    )
    if os.path.abspath(args.output_file) != expected_output:
        raise ValueError("Evaluation phase-audit output path differs")
    evidence = audit_recovery_evaluation_phase(
        args.repo_root, args.output_root, args.phase
    )
    payload = sealed(
        {
            "schema_version": 1,
            "recovery_id": RECOVERY_ID,
            "phase": args.phase,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repair_repo_commit": addendum["repair"]["repo_commit"],
            "recovery_addendum_sha256": sha256_file(args.addendum_file),
            "evidence": evidence,
        }
    )
    atomic_write_json_once(args.output_file, payload)
    print(args.output_file)


def add_common(parser, include_addendum=True, include_jobs=True):
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logs-root", required=True)
    if include_jobs:
        parser.add_argument("--jobs-file", required=True)
    if include_addendum:
        parser.add_argument("--addendum-file", required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("verify-preflight")
    add_common(preflight, include_addendum=False, include_jobs=False)
    preflight.set_defaults(func=command_verify_preflight)

    write = subparsers.add_parser("write-addendum")
    add_common(write, include_addendum=False)
    write.add_argument("--output-file", required=True)
    write.set_defaults(func=command_write_addendum)

    control = subparsers.add_parser("verify-control")
    add_common(control)
    control.set_defaults(func=command_verify_control)

    job = subparsers.add_parser("verify-job")
    add_common(job)
    job.add_argument("--job-id", required=True)
    job.add_argument("--time-limit", required=True)
    job.set_defaults(func=command_verify_job)

    snapshot = subparsers.add_parser("verify-snapshot")
    add_common(snapshot)
    snapshot.set_defaults(func=command_verify_snapshot)

    phase = subparsers.add_parser("verify-phase")
    add_common(phase)
    phase.add_argument("--phase", choices=("development", "sealed-test"), required=True)
    phase.add_argument("--output-file", required=True)
    phase.set_defaults(func=command_verify_phase)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
