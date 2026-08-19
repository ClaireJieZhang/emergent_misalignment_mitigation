#!/usr/bin/env python3
"""Fail-closed audit for the test-only MASSIVE evaluation recovery v2.

Development and checkpoint selection are immutable evidence from recovery v1.
This recovery changes only XGrammar's structural-whitespace policy, then
regenerates the complete base/selected cleaned-test pair in a fresh namespace.
"""

import argparse
import ast
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path


RECOVERY_ID = "massive_benefit_evaluation_recovery_v2"
BASE_COMMIT = "740ef7db7fa75488acea8ba76e000f4b786a54db"
V1_RECOVERY_ID = "massive_benefit_evaluation_recovery_v1"
V1_JOB_ID = "246311"
V1_SELECTION_STEP = 30
V1_PROFILE = "const_tree_v2"
FINAL_PROFILE = "const_tree_no_ws_v3"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
PRIOR_ROUNDED_H200_MINUTES = 157
EVALUATION_MAX_H200_MINUTES = 15
ORIGINAL_MAX_H200_MINUTES = 195
H200_RATE_PER_HOUR_USD = Decimal("0.90")
ORIGINAL_MAX_COST_USD = Decimal("2.925")
CUMULATIVE_MAX_H200_MINUTES = 172
CONTINGENCY_MAX_H200_MINUTES = 173

ALLOWED_REPAIR_PATHS = frozenset(
    {
        "docs/massive_benefit_evaluation_recovery_v2.md",
        "scripts/audit_massive_benefit_evaluation_recovery_v2.py",
        "scripts/audit_massive_benefit_evaluation_recovery_v1.py",
        "scripts/evaluate_massive_benefit_generations.py",
        "scripts/sample_massive_structured_generations.py",
        "scripts/sbatch_massive_benefit_evaluation_recovery_v2_tillicum_h200.sbatch",
        "scripts/stage_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        "scripts/status_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        "scripts/submit_massive_benefit_evaluation_recovery_v2_tillicum.sh",
        "scripts/summarize_massive_benefit_pilot.py",
        "tests/test_massive_benefit_evaluation_recovery_v2.py",
        "tests/test_massive_benefit_evaluation_recovery_v1.py",
        "tests/test_massive_benefit_pilot.py",
    }
)

V1_CONTROL_HASHES = {
    "AUTHORIZED_EVALUATION_RECOVERY_WITHIN_ORIGINAL_CAP.json":
        "09bfb31bc8a7d69686db66a9f7047ac4a26440bd1e07163e76d2f4e1bee21a29",
    "EVALUATE_STARTED":
        "cd87dec5baf087001cb896a79aff7b880c2fb37c32155dff6d0fb025e03d6ccf",
    "GO_MASSIVE_BASE_DEV":
        "7a75c20234ea33fdfe2c8002cb96497d48a0a97ccb5acf9a2fc709b91c7a2b4f",
    "GO_MASSIVE_SEALED_TEST":
        "9759362071528837322c63b978ed9a55447a62537aa8fd661b6248c0baeff180",
    "RELEASED":
        "3dcbd5e06c1e57ca21ca78998e51ff14ae23ddb049b4a7365105fa040d0cb1f3",
    "SUBMITTED":
        "4958db649e71ecc7fad46bd60397f33aa962103ff660574ba29c4badcac1b5db",
    "dispatch_attempt.tsv":
        "b41fcc4fa50b7ebe5be1fccaff1afeefe286908bdae7710236170b7ef869ef19",
    "jobs.tsv":
        "b41fcc4fa50b7ebe5be1fccaff1afeefe286908bdae7710236170b7ef869ef19",
}

V1_LOG_HASHES = {
    "massive_benefit_evaluation_recovery_v1_246311.out":
        "a3bff0097f92e8f2f95dfd50ed35fbea23fa8856ceb926840258319ed44540ab",
    "massive_benefit_evaluation_recovery_v1_246311.err":
        "fc608958125b6d30637fd9b570564e4e8977620a70a904c5f1bfa5475feee9f3",
}

V1_DECISION_HASHES = {
    "base_development/decoder_provenance_audit.json":
        "60001c760eeb1ee0527ce73653b3ef5442b3a42f58b1a5bb83a3e156a849799f",
    "base_development/summary.json":
        "17face99ff1ab749dbaf71b5b2de48c41db0b55196e00cb2b22a32ab60d6fa0c",
    "base_development/summary.md":
        "5bd87da788bd6e238fa2404dc02cb464628acf442a2c2d3c4a746980f678e9ec",
    "selection/summary.json":
        "11560cbea42049bdf40dcf4db9bfc0e5ffc9bea6084f41de7c0a9a9981c0cdfd",
    "selection/summary.md":
        "a9300de9ef54863a10db6f49e09ef3b9b0eec76df0a8a4f18c6086b9222966b9",
}

V1_GENERATION_HASHES = {
    "massive_en_dev__pi_base.json":
        "09ed2859496ad12e8e5c613061e580343468694c013fe1a0f67639e3506b5377",
    "massive_en_dev__pi_base__intent_only.json":
        "5334f9e6da2d968d803bfb9da707d3b6fcd9623b7889b798e50ae9f433fa57e6",
    "massive_en_dev__step_15.json":
        "ea0f4160ccc09e0eba11824c13c666e57826c21fdcc162ce314a8377e2567184",
    "massive_en_dev__step_15__intent_only.json":
        "f39d143da9c4aebaaa7ed92e8cb99b3b80c4767371fffa82d99d81839e807387",
    "massive_en_dev__step_30.json":
        "0437b49fe74044e88f8e1efd358e26e4f9c031d982369a165d8113998039691b",
    "massive_en_dev__step_30__intent_only.json":
        "8f51ee053043121f6eabbdbf3930c639feb54c64d78ba9ffdaa4c4b46073d43c",
    "massive_en_dev__step_60.json":
        "bef1028fbc18ef24c396c83e07e89c17ab08a3902431d2f91087b5ce22a04b00",
    "massive_en_dev__step_60__intent_only.json":
        "4346c6c227171070846fd23f60fc8a7169dbd359a4b0faf14b3a2ab5dd58e9b6",
    "massive_en_dev__step_90.json":
        "c4f1934a55d3b0a7229b97ac10cc9650d4256d88fa631c9744a133be5a798cb1",
    "massive_en_dev__step_90__intent_only.json":
        "34544dabd805830a190e8352c40da8837318697f2a6eda3c020eb354ea627f17",
    "massive_en_dev__step_150.json":
        "3f131bb1c1e60cc59611bbfe6f0e346b90e437a1cdacee6a6d8dc5b66ee67410",
    "massive_en_dev__step_150__intent_only.json":
        "7cded5fd28bade433e8add0ee9178b569f80c9348562fa017f634e971a9bde4f",
}

V1_SCORE_HASHES = {
    "massive_en_dev__pi_base.json":
        "896594d465ee73827d54d860f57cf00c181bd485b4e738cfdac1c282e5619969",
    "massive_en_dev__step_15.json":
        "95594c873aa46e085549350e225e4b0c87403c2cac1273c524a117a0c78a73a9",
    "massive_en_dev__step_30.json":
        "761a30f8aece447aa305a029fb08509df917d68fd7eadbfd61c6f5e122bf245e",
    "massive_en_dev__step_60.json":
        "d54bf9ec355d281eca01191e091ee82e4b2461411e90ccf0d630cdf1f66d6101",
    "massive_en_dev__step_90.json":
        "bf95e8691e6ba91ac416457e2fe64d8a020870d16189f9183c31c177de21353d",
    "massive_en_dev__step_150.json":
        "35baf5c61c57ea513895d903659ed13a02a0583857af45c4b51a0b4c9f4fba8f",
}

V1_PARTIAL_TEST_HASHES = {
    "generations/massive_en_test__pi_base.json":
        "3afb2ad44eb2620c567fed6542b9ca691b5cd1b0136419d3cb620b4f7763a5c6",
    "generations/failures/massive_en_test__pi_base__intent_only.failure.json":
        "23d1ccc633da88b83537d112bfab6db4ac7992699eda37ece66500355db1c197",
}


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


def require_regular_hash(path, expected=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or unsafe immutable artifact: {path}")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(f"Immutable artifact hash drift: {path}")
    return {"size_bytes": path.stat().st_size, "sha256": observed}


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


def load_script(repo_root, relative, name):
    path = Path(repo_root) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_repair_commit(repo_root):
    repair_commit = git(repo_root, "rev-parse", "HEAD").strip()
    parents = git(repo_root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(parents) != 2 or parents[1] != BASE_COMMIT:
        raise ValueError("Recovery-v2 commit must be a direct single-parent child")
    changed = frozenset(
        git(repo_root, "diff", "--name-only", BASE_COMMIT, repair_commit).splitlines()
    )
    if changed != ALLOWED_REPAIR_PATHS:
        raise ValueError(
            "Recovery-v2 path set differs: "
            f"missing={sorted(ALLOWED_REPAIR_PATHS - changed)}, "
            f"unauthorized={sorted(changed - ALLOWED_REPAIR_PATHS)}"
        )
    dirty = git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"Recovery-v2 checkout is dirty:\n{dirty}")
    diff = git(repo_root, "diff", "--binary", BASE_COMMIT, repair_commit, binary=True)
    return {
        "repo_commit": repair_commit,
        "parent_commit": BASE_COMMIT,
        "changed_paths": sorted(changed),
        "diff_sha256": sha256_bytes(diff),
    }


def function_ast_hashes(repo_root, path, names):
    baseline = ast.parse(git(repo_root, "show", f"{BASE_COMMIT}:{path}"))
    current = ast.parse((Path(repo_root) / path).read_text(encoding="utf-8"))
    before = {
        node.name: node for node in baseline.body if isinstance(node, ast.FunctionDef)
    }
    after = {
        node.name: node for node in current.body if isinstance(node, ast.FunctionDef)
    }
    evidence = {}
    for name in names:
        if name not in before or name not in after:
            raise ValueError(f"Missing frozen scientific function {path}:{name}")
        left = ast.dump(before[name], include_attributes=False)
        right = ast.dump(after[name], include_attributes=False)
        if left != right:
            raise ValueError(f"Scientific function changed: {path}:{name}")
        evidence[name] = sha256_bytes(left.encode("utf-8"))
    return evidence


def assignment_ast_hashes(repo_root, path, names):
    def assignments(source):
        result = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = node
        return result

    before = assignments(git(repo_root, "show", f"{BASE_COMMIT}:{path}"))
    after = assignments((Path(repo_root) / path).read_text(encoding="utf-8"))
    evidence = {}
    for name in names:
        left = ast.dump(before[name], include_attributes=False)
        right = ast.dump(after[name], include_attributes=False)
        if left != right:
            raise ValueError(f"Scientific constant changed: {path}:{name}")
        evidence[name] = sha256_bytes(left.encode("utf-8"))
    return evidence


def audit_scientific_contract(repo_root):
    evaluator_path = "scripts/evaluate_massive_benefit_generations.py"
    summarizer_path = "scripts/summarize_massive_benefit_pilot.py"
    evaluator = function_ast_hashes(
        repo_root,
        evaluator_path,
        (
            "load_data_manifest", "normalize_value", "sample_sha256",
            "load_answers", "validate_prediction", "load_generations",
            "safe_ratio", "aggregate", "evaluate",
        ),
    )
    summarizer = function_ast_hashes(
        repo_root,
        summarizer_path,
        (
            "load_model_manifest", "percentile", "paired_bootstrap_interval",
            "one_sided_exact_mcnemar_p", "comparison", "gate",
            "frozen_thresholds", "parse_checkpoint",
        ),
    )
    constants = assignment_ast_hashes(
        repo_root,
        summarizer_path,
        (
            "SELECTION_STEPS", "BOOTSTRAP_SEED", "BOOTSTRAP_REPLICATES",
            "MAX_BASE_ACCURACY", "MIN_CANDIDATE_INTENT", "MIN_INTENT_GAIN",
            "MAX_P", "MIN_SLOT_F1", "MIN_SLOT_F1_DELTA", "MIN_FRAME_EXACT",
            "MIN_FRAME_GAIN", "EXPECTED_DEV_N", "EXPECTED_TEST_N",
        ),
    )
    source = (Path(repo_root) / summarizer_path).read_text(encoding="utf-8")
    required = (
        '("const_tree_v2", "const_tree_no_ws_v3")',
        '"selection_structured_constraint_profile": selection_profile',
        '"final_structured_constraint_profile": final_profile',
        '"xgrammar_any_whitespace": xgrammar_any_whitespace(candidate["meta"])',
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("Final summarizer lacks the exact provenance-only transition")
    return {
        "baseline_commit": BASE_COMMIT,
        "evaluator_frozen_function_asts": evaluator,
        "summarizer_frozen_function_asts": summarizer,
        "frozen_threshold_constant_asts": constants,
        "metrics_gates_selection_rule_changed": False,
        "allowed_profile_transition": [V1_PROFILE, FINAL_PROFILE],
    }


def audit_sampler_contract(repo_root):
    sampler = load_script(
        repo_root, "scripts/sample_massive_structured_generations.py", "v2_sampler"
    )
    if FINAL_PROFILE not in sampler.STRUCTURED_CONSTRAINT_PROFILES:
        raise ValueError("Sampler lacks the recovery-v2 profile")
    source = (Path(repo_root) / "scripts/sample_massive_structured_generations.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "disable_any_whitespace=disable_any_whitespace",
        "any_whitespace=not disable_any_whitespace",
        '"xgrammar_any_whitespace"',
        "shutdown_vllm_engine",
        "write_or_audit_failure_evidence",
    ):
        if fragment not in source:
            raise ValueError(f"Sampler lacks recovery-v2 contract: {fragment}")
    meta = {"intent_labels": ["a"], "slot_labels": ["s"]}
    v2 = sampler.prediction_schema(
        meta["intent_labels"], meta["slot_labels"],
        endpoint="joint_json", structured_constraint_profile=V1_PROFILE,
    )
    v3 = sampler.prediction_schema(
        meta["intent_labels"], meta["slot_labels"],
        endpoint="joint_json", structured_constraint_profile=FINAL_PROFILE,
    )
    if canonical_json_bytes(v2) != canonical_json_bytes(v3):
        raise ValueError("No-whitespace profile changed the semantic JSON schema")
    return {
        "profile": FINAL_PROFILE,
        "xgrammar_any_whitespace": False,
        "semantic_schema_identical_to_const_tree_v2": True,
        "engine_and_request_no_whitespace_flags_required": True,
        "token_matcher_tab_rejection_preflight_required": True,
    }


def parse_allocated_h200(alloc_tres):
    values = {}
    for token in (alloc_tres or "").split(","):
        if "=" in token:
            key, value = token.rsplit("=", 1)
            values[key] = value
    value = values.get("gres/gpu:h200", "0")
    if not value.isdigit():
        raise ValueError("Invalid H200 allocation accounting")
    return int(value)


def read_accounting_row(job_id):
    output = subprocess.check_output(
        [
            "sacct", "-X", "-n", "-P", "--starttime", "2026-08-17",
            "--jobs", str(job_id),
            "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,AllocTRES,ExitCode,Start,End",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != 8 or fields[0] != str(job_id):
            continue
        rows.append(
            {
                "job_id": fields[0], "state": fields[1],
                "elapsed_seconds": int(fields[2]),
                "time_limit_minutes": int(fields[3]),
                "alloc_tres": fields[4], "exit_code": fields[5],
                "start": fields[6], "end": fields[7],
                "allocated_h200": parse_allocated_h200(fields[4]),
                "rounded_h200_minutes": math.ceil(int(fields[2]) / 60),
            }
        )
    if len(rows) != 1:
        raise ValueError(f"Expected one accounting row for {job_id}")
    return rows[0]


def audit_v1_accounting(v1):
    prior = v1.audit_prior_accounting()
    row = read_accounting_row(V1_JOB_ID)
    expected = {
        "state": "FAILED", "elapsed_seconds": 425, "time_limit_minutes": 15,
        "exit_code": "1:0", "allocated_h200": 1,
        "rounded_h200_minutes": 8,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Recovery-v1 accounting differs: {row!r}")
    if sum(item["rounded_h200_minutes"] for item in prior) + 8 != 157:
        raise ValueError("Cumulative prior accounting differs")
    return {"earlier_jobs": prior, "evaluation_recovery_v1": row}


def audit_v1_evidence(repo_root, output_root, logs_root):
    output_root = Path(output_root)
    logs_root = Path(logs_root)
    v1 = load_script(
        repo_root,
        "scripts/audit_massive_benefit_evaluation_recovery_v1.py",
        "v1_recovery_auditor",
    )
    original = v1.audit_original_artifacts(repo_root, output_root, logs_root)
    infrastructure = v1.audit_prior_recovery(output_root, logs_root)
    model = v1.audit_prior_model(output_root, infrastructure)
    control_root = output_root / "control/evaluation_recovery_v1"
    eval_root = output_root / "evaluation/evaluation_recovery_v1"
    control = {
        relative: require_regular_hash(control_root / relative, expected)
        for relative, expected in V1_CONTROL_HASHES.items()
    }
    lock = require_regular_hash(
        output_root / "control/MASSIVE_EVALUATION_RECOVERY_V1_SUBMISSION_LOCK/owner"
    )
    logs = {
        name: require_regular_hash(logs_root / name, expected)
        for name, expected in V1_LOG_HASHES.items()
    }
    decisions = {
        relative: require_regular_hash(eval_root / relative, expected)
        for relative, expected in V1_DECISION_HASHES.items()
    }
    generations = {
        name: require_regular_hash(eval_root / "generations" / name, expected)
        for name, expected in V1_GENERATION_HASHES.items()
    }
    scores = {
        name: require_regular_hash(eval_root / "scores" / name, expected)
        for name, expected in V1_SCORE_HASHES.items()
    }
    partial = {
        relative: require_regular_hash(eval_root / relative, expected)
        for relative, expected in V1_PARTIAL_TEST_HASHES.items()
    }
    addendum = load_json(
        control_root / "AUTHORIZED_EVALUATION_RECOVERY_WITHIN_ORIGINAL_CAP.json"
    )
    v1.verify_seal(addendum)
    if (
        addendum.get("recovery_id") != V1_RECOVERY_ID
        or addendum.get("repair", {}).get("repo_commit") != BASE_COMMIT
        or addendum.get("budget", {}).get("cumulative_max_h200_minutes") != 164
        or addendum.get("constraints", {}).get("exact_once") is not True
    ):
        raise ValueError("Recovery-v1 authorization differs")
    selection = load_json(eval_root / "selection/summary.json")
    v1.verify_seal(selection, "decision_payload_sha256")
    selected = selection.get("selected", {})
    if (
        selection.get("decision") != "GO"
        or selection.get("structured_constraint_profile") != V1_PROFILE
        or selected.get("step") != V1_SELECTION_STEP
        or selected.get("model_name") != "step_30"
        or selected.get("structured_constraint_profile") != V1_PROFILE
        or selected.get("model_fingerprint")
        != model["checkpoint_fingerprints"]["30"]
        or not all(selection.get("checks", {}).values())
    ):
        raise ValueError("Recovery-v1 sealed selection differs")
    failure = load_json(
        eval_root
        / "generations/failures/massive_en_test__pi_base__intent_only.failure.json"
    )
    sampler = load_script(
        repo_root, "scripts/sample_massive_structured_generations.py", "v2_sampler_evidence"
    )
    sampler.verify_failure_evidence(failure)
    offending = failure.get("offending_sample", {})
    generation = failure.get("generation", {})
    if (
        failure.get("failure_kind") != "structured_prediction_validation"
        or generation.get("structured_constraint_profile") != V1_PROFILE
        or generation.get("model_name") != "pi_base"
        or generation.get("endpoint") != "intent_only"
        or offending.get("row_index") != 377
        or offending.get("finish_reason") != "max_new_tokens"
        or offending.get("n_generated_tokens") != 256
        or "\t" not in offending.get("raw_response", "")
    ):
        raise ValueError("Recovery-v1 whitespace failure evidence differs")
    base_joint = load_json(eval_root / "generations/massive_en_test__pi_base.json")
    run = {
        key: value for key, value in base_joint.get("meta", {}).items()
        if key not in {"generation_fingerprint", "created_at"}
    }
    if (
        base_joint.get("meta", {}).get("generation_fingerprint")
        != sha256_bytes(canonical_json_bytes(run))
        or run.get("structured_constraint_profile") != V1_PROFILE
        or run.get("model_name") != "pi_base"
        or run.get("role") != "sealed_final"
        or len(base_joint.get("samples", [])) != 2965
    ):
        raise ValueError("Recovery-v1 partial base-joint generation differs")
    forbidden = (
        eval_root / "scores/massive_en_test__pi_base.json",
        eval_root / "generations/massive_en_test__step_30.json",
        eval_root / "generations/massive_en_test__step_30__intent_only.json",
        eval_root / "sealed_final/summary.json",
        control_root / "GO_MASSIVE_BENEFIT_ONLY",
        control_root / "STOPPED_MASSIVE_FINAL",
    )
    if any(os.path.lexists(path) for path in forbidden):
        raise ValueError("Recovery-v1 namespace advanced after its terminal failure")
    accounting = audit_v1_accounting(v1)
    return {
        "original": original,
        "infrastructure_recovery_v1": infrastructure,
        "trained_model": model,
        "control": control,
        "lock_owner": lock,
        "logs": logs,
        "decision_artifacts": decisions,
        "development_generations": generations,
        "development_scores": scores,
        "partial_test_artifacts_not_reused": partial,
        "selection_step": V1_SELECTION_STEP,
        "selection_model_name": "step_30",
        "selection_model_fingerprint": selected["model_fingerprint"],
        "selection_sha256": V1_DECISION_HASHES["selection/summary.json"],
        "local_model_snapshot": infrastructure["local_model_snapshot"],
        "accounting": accounting,
    }


def parse_recovery_jobs(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or rows[0].get("stage") != "evaluate":
        raise ValueError("Recovery-v2 jobs file must contain one evaluation job")
    if not rows[0].get("job_id", "").isdigit() or rows[0].get("max_minutes") != "15":
        raise ValueError("Recovery-v2 job record differs from the 15-minute cap")
    return [
        {
            "stage": "evaluate", "job_id": rows[0]["job_id"],
            "max_minutes": EVALUATION_MAX_H200_MINUTES,
        }
    ]


def build_addendum(args, created_at=None, audit_live_snapshot=True):
    history = audit_repair_commit(args.repo_root)
    scientific = audit_scientific_contract(args.repo_root)
    sampler = audit_sampler_contract(args.repo_root)
    prior = audit_v1_evidence(args.repo_root, args.output_root, args.logs_root)
    if audit_live_snapshot:
        v1 = load_script(
            args.repo_root,
            "scripts/audit_massive_benefit_evaluation_recovery_v1.py",
            "v1_snapshot_auditor",
        )
        snapshot = v1.audit_live_local_snapshot(
            args.repo_root, args.output_root, prior["local_model_snapshot"]
        )
    else:
        snapshot = prior["local_model_snapshot"]
    if CUMULATIVE_MAX_H200_MINUTES != PRIOR_ROUNDED_H200_MINUTES + 15:
        raise AssertionError("Recovery-v2 budget arithmetic differs")
    if (
        CONTINGENCY_MAX_H200_MINUTES > ORIGINAL_MAX_H200_MINUTES
        or Decimal(CONTINGENCY_MAX_H200_MINUTES)
        * H200_RATE_PER_HOUR_USD / Decimal(60) > ORIGINAL_MAX_COST_USD
    ):
        raise ValueError("Recovery-v2 exceeds the original authorization")
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "created_at": created_at
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reason": "unbounded_structural_whitespace_after_valid_intent",
        "repair": history,
        "scientific_contract": scientific,
        "sampler_repair": sampler,
        "prior_evidence": prior,
        "live_local_model_snapshot_sha256": sha256_bytes(
            canonical_json_bytes(snapshot)
        ),
        "recovery_jobs": parse_recovery_jobs(args.jobs_file),
        "budget": {
            "h200_rate_per_hour_usd": "0.90",
            "prior_rounded_h200_minutes": 157,
            "prior_rounding_by_job": {
                "237935": 2, "237936": 1, "237937": 0,
                "239578": 70, "239579": 76, "246311": 8,
            },
            "new_evaluation_max_h200_minutes": 15,
            "cumulative_max_h200_minutes": 172,
            "cumulative_max_cost_usd": "2.580",
            "termination_overhead_contingency_minutes": 1,
            "contingency_max_h200_minutes": 173,
            "contingency_max_cost_usd": "2.595",
            "original_authorized_max_h200_minutes": 195,
            "original_authorized_max_cost_usd": "2.925",
        },
        "frozen_scope": {
            "reuse_v1_selection": True,
            "selected_step": 30,
            "selected_model_name": "step_30",
            "selection_profile": V1_PROFILE,
            "final_profile": FINAL_PROFILE,
            "fresh_test_models": ["pi_base", "step_30"],
            "fresh_test_endpoints": ["joint_json", "intent_only"],
            "reuse_partial_v1_test_generation": False,
            "rerun_development_or_selection": False,
            "metrics_thresholds_and_gates_changed": False,
        },
        "constraints": {
            "held_first": True, "exact_once": True, "no_requeue": True,
            "one_h200": True, "no_further_retry_or_recovery": True,
            "no_training": True, "no_development_regeneration": True,
            "no_extra_adapter": True, "no_medical_union": True,
            "no_quorum": True, "preserve_prior_namespaces": True,
        },
    }


def verify_addendum(args):
    observed = load_json(args.addendum_file)
    verify_seal(observed)
    expected = sealed(
        build_addendum(
            args, created_at=observed.get("created_at"), audit_live_snapshot=False
        )
    )
    if observed != expected:
        raise ValueError("Recovery-v2 addendum differs from immutable evidence")
    return observed


def command_verify_preflight(args):
    with tempfile.TemporaryDirectory() as temporary:
        jobs = Path(temporary) / "jobs.tsv"
        jobs.write_text(
            "stage\tjob_id\tmax_minutes\nevaluate\t999999999\t15\n",
            encoding="utf-8",
        )
        args.jobs_file = os.fspath(jobs)
        payload = build_addendum(args)
    print(
        "Recovery-v2 preflight passed: "
        f"commit={payload['repair']['repo_commit']} cumulative=172m/$2.580"
    )


def command_write_addendum(args):
    payload = sealed(build_addendum(args))
    atomic_write_json_once(args.output_file, payload)
    print(args.output_file)


def command_verify_control(args):
    verify_addendum(args)
    print(args.addendum_file)


def parse_time_limit(value):
    if not re.fullmatch(r"\d\d:\d\d:\d\d", value):
        raise ValueError(f"Unsupported Slurm TimeLimit: {value}")
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return math.ceil((hours * 3600 + minutes * 60 + seconds) / 60)


def command_verify_job(args):
    addendum = verify_addendum(args)
    if parse_time_limit(args.time_limit) != EVALUATION_MAX_H200_MINUTES:
        raise ValueError("Running recovery-v2 TimeLimit differs")
    if addendum["recovery_jobs"] != [
        {"stage": "evaluate", "job_id": args.job_id, "max_minutes": 15}
    ]:
        raise ValueError("Running job is not the uniquely authorized job")
    print(f"Authorized test-only recovery-v2 job {args.job_id}")


def command_verify_snapshot(args):
    addendum = verify_addendum(args)
    v1 = load_script(
        args.repo_root,
        "scripts/audit_massive_benefit_evaluation_recovery_v1.py",
        "v1_live_snapshot_auditor",
    )
    expected = addendum["prior_evidence"]["local_model_snapshot"]
    observed = v1.audit_live_local_snapshot(args.repo_root, args.output_root, expected)
    if sha256_bytes(canonical_json_bytes(observed)) != addendum[
        "live_local_model_snapshot_sha256"
    ]:
        raise ValueError("Live local model snapshot differs from v2 addendum")
    print("Pinned local-model snapshot verified for recovery v2")


def audit_final_phase(repo_root, output_root, addendum):
    sampler = load_script(
        repo_root, "scripts/sample_massive_structured_generations.py", "v2_phase_sampler"
    )
    evaluator = load_script(
        repo_root, "scripts/evaluate_massive_benefit_generations.py", "v2_phase_evaluator"
    )
    summarizer = load_script(
        repo_root, "scripts/summarize_massive_benefit_pilot.py", "v2_phase_summarizer"
    )
    output_root = Path(output_root)
    eval_root = output_root / "evaluation/evaluation_recovery_v2"
    gen_root = eval_root / "generations"
    score_root = eval_root / "scores"
    prompt_path = output_root / "data/sealed_test/prompts.json"
    answer_path = output_root / "data/sealed_test/answers.json"
    meta, prompts = sampler.load_prompt_bank(prompt_path)
    answer_meta, answers = evaluator.load_answers(answer_path)
    if len(prompts) != 2965 or len(answers) != 2965:
        raise ValueError("Cleaned-test row count differs")
    model_dir = output_root / "model/massive_en_benefit_pilot_infrastructure_recovery_v1"
    fingerprint = addendum["prior_evidence"]["selection_model_fingerprint"]
    specs = {
        "pi_base": ("BASE", "BASE"),
        "step_30": (os.path.abspath(model_dir / "checkpoint-30"), fingerprint),
    }
    generation_evidence = {}
    score_payloads = {}
    for name, (model_path, model_fingerprint) in specs.items():
        paths = {}
        for endpoint in ("joint_json", "intent_only"):
            suffix = "" if endpoint == "joint_json" else "__intent_only"
            path = gen_root / f"massive_en_test__{name}{suffix}.json"
            payload = load_json(path)
            run = {
                key: value for key, value in payload.get("meta", {}).items()
                if key not in {"generation_fingerprint", "created_at"}
            }
            frozen = {
                "endpoint": endpoint, "set_name": "massive_en_test",
                "role": "sealed_final", "model_name": name,
                "model_path": model_path, "model_fingerprint": model_fingerprint,
                "base_model": MODEL_ID, "base_model_revision": MODEL_REVISION,
                "structured_backend": "xgrammar", "vllm_version": "0.11.2",
                "xgrammar_version": "0.1.25", "temperature": 0.0,
                "n_samples": 1, "max_new_tokens": 256, "max_context": 2048,
                "seed": 8172026, "same_prompt_all_models": True,
                "selection_uses_joint_json_only": True,
                "structured_constraint_profile": FINAL_PROFILE,
                "xgrammar_any_whitespace": False,
            }
            for key, expected in frozen.items():
                if run.get(key) != expected:
                    raise ValueError(f"Recovery-v2 generation differs on {key}: {path}")
            if run.get("prompt_file_sha256") != sha256_file(prompt_path):
                raise ValueError("Recovery-v2 generation prompt hash differs")
            if (
                run.get("question_ids")
                != [record["question_id"] for record in prompts]
                or run.get("prompt_sha256")
                != [record["prompt_sha256"] for record in prompts]
                or run.get("ontology_sha256") != meta["ontology_sha256"]
            ):
                raise ValueError("Recovery-v2 generation bank provenance differs")
            schema = sampler.prediction_schema(
                meta["intent_labels"], meta["slot_labels"], endpoint=endpoint,
                structured_constraint_profile=FINAL_PROFILE,
            )
            if run.get("json_schema_sha256") != sha256_bytes(
                canonical_json_bytes(schema)
            ):
                raise ValueError("Recovery-v2 JSON-schema binding differs")
            if payload["meta"].get("generation_fingerprint") != sha256_bytes(
                canonical_json_bytes(run)
            ):
                raise ValueError("Recovery-v2 generation fingerprint differs")
            samples = payload.get("samples", [])
            if len(samples) != len(prompts):
                raise ValueError("Recovery-v2 generation row count differs")
            for record, sample in zip(prompts, samples):
                if (
                    sample.get("question_id") != record["question_id"]
                    or sample.get("prompt_sha256") != record["prompt_sha256"]
                    or sample.get("sample_index") != 0
                    or sample.get("stop_reason") == "max_new_tokens"
                    or sample.get("result_sha256") != sampler.sample_sha256(sample)
                ):
                    raise ValueError("Recovery-v2 generation row provenance differs")
                parsed = sampler.validate_prediction(
                    sample.get("response"), meta["intent_labels"],
                    meta["slot_labels"], endpoint=endpoint,
                )
                if parsed != sample.get("prediction"):
                    raise ValueError("Recovery-v2 stored prediction differs")
            paths[endpoint] = path
            generation_evidence[path.name] = require_regular_hash(path)
        score_path = score_root / f"massive_en_test__{name}.json"
        score = summarizer.load_evaluation(
            score_path, expected_role="sealed_final", expected_n=2965,
            expected_constraint_profile=FINAL_PROFILE,
        )
        if (
            score["meta"].get("model_name") != name
            or score["meta"].get("model_fingerprint") != model_fingerprint
            or score["meta"].get("xgrammar_any_whitespace") is not False
            or score["meta"].get("answers_file_sha256") != sha256_file(answer_path)
            or score["meta"].get("data_manifest_sha256")
            != addendum["prior_evidence"]["original"]["artifacts_sha256"][
                "data/data_manifest.json"
            ]
            or score["meta"].get("prompt_file_sha256") != sha256_file(prompt_path)
            or score["meta"].get("ontology_sha256") != meta["ontology_sha256"]
            or score["meta"].get("joint_generations_file_sha256")
            != sha256_file(paths["joint_json"])
            or score["meta"].get("intent_generations_file_sha256")
            != sha256_file(paths["intent_only"])
        ):
            raise ValueError("Recovery-v2 score provenance differs")
        score_payloads[name] = score
    summarizer.validate_pair(score_payloads["pi_base"], score_payloads["step_30"])
    return {
        "generation_artifacts": dict(sorted(generation_evidence.items())),
        "base_score": require_regular_hash(
            score_root / "massive_en_test__pi_base.json"
        ),
        "candidate_score": require_regular_hash(
            score_root / "massive_en_test__step_30.json"
        ),
        "n": 2965,
        "selection_sha256": addendum["prior_evidence"]["selection_sha256"],
        "selection_profile": V1_PROFILE,
        "final_profile": FINAL_PROFILE,
        "xgrammar_any_whitespace": False,
    }


def command_verify_phase(args):
    addendum = verify_addendum(args)
    expected = os.path.abspath(
        os.path.join(
            args.output_root,
            "evaluation/evaluation_recovery_v2/sealed_final/decoder_provenance_audit.json",
        )
    )
    if os.path.abspath(args.output_file) != expected:
        raise ValueError("Recovery-v2 phase-audit output path differs")
    evidence = audit_final_phase(args.repo_root, args.output_root, addendum)
    payload = sealed(
        {
            "schema_version": 1, "recovery_id": RECOVERY_ID,
            "phase": "sealed_final", "created_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
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
    phase.add_argument("--output-file", required=True)
    phase.set_defaults(func=command_verify_phase)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
