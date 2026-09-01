#!/usr/bin/env python3
"""Plan a CPU-only recovery of Kalai s=1 completion batch 1.

The failed job finished its immutable generation, then stopped in the CPU
auditor because the controller omitted ``import math``.  This planner binds
the complete failed-job control and generation state in place.  It never
copies, regenerates, resumes, retries, or modifies a source artifact.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = _load_module(
    "_kalai_s1_batch1_recovery_controller_for_planner",
    SCRIPT_DIR / "sample_massive_medical_kalai_s1_completion_batch_v1.py",
)
batch_planner = controller.batch_planner


PROTOCOL_ID = "massive_medical_kalai_s1_batch1_result_recovery_v1"
SOURCE_PROTOCOL_ID = controller.PROTOCOL_ID
METHOD_ID = controller.METHOD_ID
SEAL_FIELD = controller.SEAL_FIELD
SOURCE_BATCH_INDEX = 1
SOURCE_BATCH_ID = "batch_01"
SOURCE_JOB_ID = "270983"
SOURCE_EXIT_CODE = "1"
SOURCE_SCHEDULER_ELAPSED_SECONDS = 1018
SOURCE_ACTUAL_ESTIMATED_COST_USD = Decimal("0.25450")
SOURCE_AUTHORIZED_CAP_USD = Decimal("0.900")
SOURCE_CONSERVATIVE_EXPOSURE_USD = Decimal("7.02198425")
PROGRAM_CEILING_USD = Decimal("12.5000000")
EXPECTED_SOURCE_REPOSITORY_COMMIT = (
    "a1e8ca218635af6dafdca7ddb3e9d42d153d6921"
)
SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "Account",
    "Partition",
    "QOS",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "ElapsedRaw",
    "Start",
    "Eligible",
    "End",
    "NNodes",
    "NCPUS",
    "ReqMem",
    "ReqTRES",
    "AllocTRES",
    "NodeList",
)
EXPECTED_SACCT_RECORD = (
    "270983|mmu_kalai_s1_b01|stf|gpu-h200|normal|FAILED|1:0|0:0|1018|"
    "2026-09-01T09:31:46|2026-09-01T09:31:46|2026-09-01T09:48:44|"
    "1|8|200G|billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|"
    "billing=8,cpu=8,gres/gpu:h200=1,gres/gpu=1,mem=200G,node=1|g020"
)
EXPECTED_JOB_LOGS = {
    "stdout": {
        "name": "massive_medical_kalai_s1_completion_batch_270983.out",
        "size_bytes": 701,
        "file_sha256": (
            "e1cadd65452ef1d162f486f4f1f82a5ff9b09fe661698819d3876423eb08fb2a"
        ),
    },
    "stderr": {
        "name": "massive_medical_kalai_s1_completion_batch_270983.err",
        "size_bytes": 4275,
        "file_sha256": (
            "314d2db7c0e225299f98e3317e4692e45d5a075980301b06bf8dc2dc3247ad20"
        ),
    },
}
EXPECTED_SOURCE_ARTIFACTS = {
    "control/COMPLETION_BATCH_PLAN.json": (
        668430,
        "becd07dccd838a9a922cb08a997e2b3eba5a939f2e67cb83f110453bd6562bae",
        "29f9e969918f766dcaa09d80b60ead2cd46a484dc2e0e23bdb592ac9409d2904",
    ),
    "control/CPU_STAGE.json": (
        5591,
        "1bc5382f53326cab7b6103e882a71ee2a5eff86518659e2cfc2334bc528684b5",
        "e1b759ad583037fd6cbe085919a63fa05190b33d08e18019289869984b28e174",
    ),
    "control/batches/batch_01/AUTHORIZATION.json": (
        4664,
        "eb1178606bed264c244529486b67f692d45b02def36476c3945bd0a412e2dd24",
        "d72d75a8bb4900ddd1ba1995dccbd8bf9d019ff684310a615fd33737859f373b",
    ),
    "control/batches/batch_01/STOPPED": (
        178,
        "f2f8ef71c52f32ca47672c45dac01b8eaf592bcd834b96ce881b8dd0b54504f0",
        None,
    ),
    "generation/completion_batches/batch_01/benefit/generation.json": (
        142007,
        "d6510c047f912bae1d200cd4eb644c2defa3de21300ccdf94cc2906dea5abfb7",
        "8ab8a7e7410553e140007bcdfe473fe90dfbee37950f447ff2050396f131f4a4",
    ),
    "generation/completion_batches/batch_01/benefit/timing.json": (
        1222,
        "f68af233afb349959c41539ba57c126e205fbf7119f1bb9e01ea78daec037648",
        "ffcc12f5a5622103dc3baddafaa71c7b54304f56d11f00a1a7cacd7ff2e6e58b",
    ),
    "generation/completion_batches/batch_01/medical/generation.json": (
        105585,
        "9a2703ac28f6ac6135465451d16e37370ed0793f614f97ed73d9dbde5fbe056e",
        "67c3d9aa4871ebfb4310df3b11752e35b8382479a857b17b385be43d8233be02",
    ),
    "generation/completion_batches/batch_01/medical/timing.json": (
        1173,
        "3ceaa1a168429539a0b600d287a3be1cf5f58c67ec97248143bde9375725359e",
        "eda36b711a6d72d012aa4d657b1f8917b6b0d8c3794ba93a0e0cb0337557b8e0",
    ),
    "generation/completion_batches/batch_01/combined_timing.json": (
        6395,
        "aa656379c8ccc277254270267bd738d7d68c4d2363aa4accf3ec8ddebe4e28c0",
        "ebc54c01a3fb5156287f1a2ecc8e77b42b052bb3584d3511cb8f6dfc392276f2",
    ),
}
EXPECTED_GENERATION_FILE_COUNT = 30
EXPECTED_GENERATION_SIZE_BYTES = 498998
EXPECTED_GENERATION_MANIFEST_SHA256 = (
    "6554bc7162be8cea86a24af46e841d50744bbecae52b261109ad70fcadf17bc2"
)
EXPECTED_CONTROL_FILE_COUNT = 10
EXPECTED_CONTROL_SIZE_BYTES = 680145
EXPECTED_CONTROL_MANIFEST_SHA256 = (
    "4330d622ed905ad433a9b2f348e0d8fbac21bbc05122e7c35ef9ae1e2c7756b7"
)
EXPECTED_SOURCE_REPOSITORY_LEAF = (
    "subliminal-mitigate-mmu-kalai-s1-completion-batches-v1"
)
EXPECTED_SOURCE_OUTPUT_LEAF = (
    "massive_medical_kalai_s1_completion_batches_v1"
)
EXPECTED_RECOVERY_OUTPUT_LEAF = (
    "massive_medical_kalai_s1_batch1_result_recovery_v1"
)
PLAN_NAME = "RECOVERY_PLAN.json"
STAGE_NAME = "CPU_STAGE.json"
RESULT_NAME = "RECOVERED_RESULT.json"

SOURCE_TOP_CONTROL_INVENTORY = {
    "COMPLETION_BATCH_PLAN.json",
    "CPU_STAGE.json",
    "batches",
}
SOURCE_BATCH_CONTROL_FILES = {
    "AUTHORIZATION.json",
    "SUBMISSION_ATTEMPT.tsv",
    "SUBMITTED",
    "RELEASE_AUTHORIZED",
    "RELEASED",
    "STOPPED",
}
SOURCE_BATCH_CONTROL_DIRECTORIES = {"SUBMISSION_LOCK", "INVOCATION_LOCK"}
SOURCE_BATCH_OWNER_FILES = {
    "SUBMISSION_LOCK": {"owner"},
    "INVOCATION_LOCK": {"owner"},
}


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def seal(body):
    payload = dict(body)
    payload.pop(SEAL_FIELD, None)
    payload[SEAL_FIELD] = sha256_bytes(canonical_bytes(payload))
    return payload


def verify_seal(payload, description):
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != sha256_bytes(canonical_bytes(body)):
        raise ValueError(f"{description} has an invalid {SEAL_FIELD}")
    return body


def load_json(path, description):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is absent or unsafe: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} is not an object")
    return payload


def git_commit(repo_root):
    return subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def require_clean(repo_root):
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain"],
        text=True,
    ).strip()
    if status:
        raise ValueError(f"repository is not clean: {repo_root}")


def _paths_overlap(first, second):
    first = Path(first).resolve()
    second = Path(second).resolve()
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def require_recovery_output_disjoint(output_root, *immutable_roots):
    output = Path(output_root).resolve()
    for immutable in immutable_roots:
        if _paths_overlap(output, immutable):
            raise ValueError(
                "recovery output overlaps an immutable source namespace"
            )


def _write_idempotent(path, payload, description):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.path.lexists(path):
        observed = load_json(path, f"existing {description}")
        verify_seal(observed, f"existing {description}")
        if observed != payload:
            raise ValueError(f"existing {description} differs")
        return "AUDITED"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return "CREATED"


def file_binding(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("bound source artifact escapes source output") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bound source artifact is absent or unsafe: {path}")
    return {
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def sealed_binding(path, root, description):
    payload = load_json(path, description)
    verify_seal(payload, description)
    return {
        **file_binding(path, root),
        "payload_sha256": payload[SEAL_FIELD],
    }


def _verify_expected_artifact(source_root, relative_path, payload=None):
    expected_size, expected_file_sha256, expected_payload_sha256 = (
        EXPECTED_SOURCE_ARTIFACTS[relative_path]
    )
    path = Path(source_root) / relative_path
    if (
        path.stat().st_size != expected_size
        or sha256_file(path) != expected_file_sha256
    ):
        raise ValueError(f"exact source artifact pin differs: {relative_path}")
    if expected_payload_sha256 is not None:
        if payload is None:
            payload = load_json(path, relative_path)
            verify_seal(payload, relative_path)
        if payload.get(SEAL_FIELD) != expected_payload_sha256:
            raise ValueError(
                f"exact source payload pin differs: {relative_path}"
            )


def _require_exact_directory(path, expected_files, expected_directories=()):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"source directory is absent or unsafe: {path}")
    expected = set(expected_files) | set(expected_directories)
    actual = {item.name for item in path.iterdir()}
    if actual != expected:
        raise ValueError(f"source directory inventory differs: {path}")
    for name in expected_files:
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"source file is absent or unsafe: {item}")
    for name in expected_directories:
        item = path / name
        if item.is_symlink() or not item.is_dir():
            raise ValueError(f"source directory is absent or unsafe: {item}")


def _control_values(path):
    if Path(path).is_symlink() or not Path(path).is_file():
        raise ValueError(f"control record is absent or unsafe: {path}")
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise ValueError(f"control record line is malformed: {path}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"control record key is malformed: {path}")
        values[key] = value
    return values


def _require_values(path, expected):
    values = _control_values(path)
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"{Path(path).name} {key} differs")
    return values


def _regular_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"source generation root is absent or unsafe: {root}")
    files = []
    for directory, directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise ValueError("source generation contains a symlink directory")
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ValueError("source generation contains a symlink directory")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("source generation contains an unsafe file")
            files.append(path)
    if not files:
        raise ValueError("source generation is empty")
    return sorted(files)


def _tree_manifest(root):
    root = Path(root).resolve()
    files = _regular_files(root)
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "entries": entries,
        "file_count": len(entries),
        "size_bytes": sum(item["size_bytes"] for item in entries),
        "manifest_sha256": sha256_bytes(canonical_bytes(entries)),
    }


def audit_job_evidence(source_output_root):
    source_output = Path(source_output_root).resolve()
    log_root = source_output.parent / "logs"
    log_bindings = {}
    for stream, expected in EXPECTED_JOB_LOGS.items():
        path = log_root / expected["name"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source {stream} log is absent or unsafe")
        observed = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
        }
        if (
            observed["size_bytes"] != expected["size_bytes"]
            or observed["file_sha256"] != expected["file_sha256"]
        ):
            raise ValueError(f"source {stream} log exact pin differs")
        log_bindings[stream] = observed
    stderr_text = Path(log_bindings["stderr"]["path"]).read_text(
        encoding="utf-8"
    )
    stdout_text = Path(log_bindings["stdout"]["path"]).read_text(
        encoding="utf-8"
    )
    if (
        "or not math.isfinite(phase_elapsed)" not in stderr_text
        or "NameError: name 'math' is not defined" not in stderr_text
        or "MASSIVE_MEDICAL_KALAI_S1_COMPLETION_BATCH_GENERATED"
        not in stdout_text
        or EXPECTED_SOURCE_ARTIFACTS[
            "generation/completion_batches/batch_01/combined_timing.json"
        ][2]
        not in stdout_text
    ):
        raise ValueError(
            "source logs do not prove completed generation and math failure"
        )

    command = [
        "sacct",
        "-n",
        "-X",
        "-P",
        "-j",
        SOURCE_JOB_ID,
        f"--format={','.join(SACCT_FIELDS)}",
    ]
    raw = subprocess.check_output(command, text=True).strip()
    if raw != EXPECTED_SACCT_RECORD:
        raise ValueError("source Slurm accounting record differs")
    values = raw.split("|")
    if len(values) != len(SACCT_FIELDS):
        raise ValueError("source Slurm accounting schema differs")
    record = dict(zip(SACCT_FIELDS, values))
    if (
        record["JobIDRaw"] != SOURCE_JOB_ID
        or record["State"] != "FAILED"
        or record["ExitCode"] != "1:0"
        or record["ElapsedRaw"] != str(SOURCE_SCHEDULER_ELAPSED_SECONDS)
        or record["JobName"] != "mmu_kalai_s1_b01"
        or record["Partition"] != "gpu-h200"
        or record["QOS"] != "normal"
    ):
        raise ValueError("source Slurm terminal evidence differs")
    return {
        "sacct_fields": list(SACCT_FIELDS),
        "sacct_record": raw,
        "parsed_record": record,
        "logs": log_bindings,
    }


def expected_job_evidence(source_output_root):
    source_output = Path(source_output_root).resolve()
    expected_parsed = dict(zip(SACCT_FIELDS, EXPECTED_SACCT_RECORD.split("|")))
    expected_logs = {}
    for stream, item in EXPECTED_JOB_LOGS.items():
        expected_logs[stream] = {
            "path": str(source_output.parent / "logs" / item["name"]),
            "size_bytes": item["size_bytes"],
            "file_sha256": item["file_sha256"],
        }
    return {
        "sacct_fields": list(SACCT_FIELDS),
        "sacct_record": EXPECTED_SACCT_RECORD,
        "parsed_record": expected_parsed,
        "logs": expected_logs,
    }


def verify_job_evidence_shape(evidence, source_output_root):
    if not isinstance(evidence, dict):
        raise ValueError("source job evidence is absent")
    expected = expected_job_evidence(source_output_root)
    if evidence != expected:
        raise ValueError("source job evidence schema or exact pins differ")
    return evidence


def _source_paths(source_output_root):
    source = Path(source_output_root).resolve()
    batch_control = source / "control" / "batches" / SOURCE_BATCH_ID
    generation = (
        source / "generation" / "completion_batches" / SOURCE_BATCH_ID
    )
    return {
        "source": source,
        "top_control": source / "control",
        "batches": source / "control" / "batches",
        "batch_control": batch_control,
        "plan": source / "control" / "COMPLETION_BATCH_PLAN.json",
        "stage": source / "control" / "CPU_STAGE.json",
        "authorization": batch_control / "AUTHORIZATION.json",
        "stopped": batch_control / "STOPPED",
        "generation": generation,
    }


def audit_source_state(source_output_root, source_repo_root):
    paths = _source_paths(source_output_root)
    source = paths["source"]
    source_repo = Path(source_repo_root).resolve()
    if source.name != EXPECTED_SOURCE_OUTPUT_LEAF:
        raise ValueError("source output leaf differs")
    if source_repo.name != EXPECTED_SOURCE_REPOSITORY_LEAF:
        raise ValueError("source repository leaf differs")
    require_clean(source_repo)
    if git_commit(source_repo) != EXPECTED_SOURCE_REPOSITORY_COMMIT:
        raise ValueError("source repository commit differs")

    _require_exact_directory(
        paths["top_control"], {"COMPLETION_BATCH_PLAN.json", "CPU_STAGE.json"}, {"batches"}
    )
    _require_exact_directory(paths["batches"], set(), {SOURCE_BATCH_ID})
    _require_exact_directory(
        paths["batch_control"],
        SOURCE_BATCH_CONTROL_FILES,
        SOURCE_BATCH_CONTROL_DIRECTORIES,
    )
    for directory, expected in SOURCE_BATCH_OWNER_FILES.items():
        _require_exact_directory(paths["batch_control"] / directory, expected)
    if os.path.lexists(paths["batch_control"] / "RESULT.json"):
        raise ValueError("source RESULT.json must remain absent")

    stopped = _require_values(
        paths["stopped"],
        {
            "stage": "completion_batch",
            "batch_id": SOURCE_BATCH_ID,
            "job_id": SOURCE_JOB_ID,
            "exit_code": SOURCE_EXIT_CODE,
            "restart_or_resume_authorized": "false",
            "retry_or_replacement_authorized": "false",
            "automatic_next_batch_authorized": "false",
        },
    )
    if set(stopped) != {
        "stage",
        "batch_id",
        "job_id",
        "exit_code",
        "restart_or_resume_authorized",
        "retry_or_replacement_authorized",
        "automatic_next_batch_authorized",
    }:
        raise ValueError("source STOPPED schema differs")
    for name in ("SUBMITTED", "RELEASE_AUTHORIZED", "RELEASED"):
        _require_values(
            paths["batch_control"] / name,
            {"batch_id": SOURCE_BATCH_ID, "job_id": SOURCE_JOB_ID},
        )
    _require_values(
        paths["batch_control"] / "INVOCATION_LOCK" / "owner",
        {
            "batch_id": SOURCE_BATCH_ID,
            "job_id": SOURCE_JOB_ID,
            "restart_or_resume_authorized": "false",
            "retry_or_replacement_authorized": "false",
            "automatic_next_batch_authorized": "false",
            "external_api_calls_authorized": "0",
        },
    )

    plan_payload, plan_body = controller._load_batch_plan(paths["plan"])
    stage_payload = load_json(paths["stage"], "source CPU stage")
    stage_body = verify_seal(stage_payload, "source CPU stage")
    authorization_payload = load_json(
        paths["authorization"], "source batch-1 authorization"
    )
    authorization_body = verify_seal(
        authorization_payload, "source batch-1 authorization"
    )
    if (
        stage_body.get("protocol_id") != SOURCE_PROTOCOL_ID
        or stage_body.get("status") != "CPU_STAGED_NO_GPU_OR_API_AUTHORITY"
        or authorization_body.get("protocol_id") != SOURCE_PROTOCOL_ID
        or authorization_body.get("batch_index") != SOURCE_BATCH_INDEX
        or authorization_body.get("batch_id") != SOURCE_BATCH_ID
        or authorization_body.get("authorized_gpu_jobs") != 1
        or authorization_body.get("h200_count") != 1
        or authorization_body.get("h200_minutes_cap") != 60
        or authorization_body.get("maximum_cost_usd") != 0.9
        or authorization_body.get("external_api_calls_authorized") != 0
    ):
        raise ValueError("source stage or authorization identity differs")

    for relative_path in EXPECTED_SOURCE_ARTIFACTS:
        payload = {
            "control/COMPLETION_BATCH_PLAN.json": plan_payload,
            "control/CPU_STAGE.json": stage_payload,
            "control/batches/batch_01/AUTHORIZATION.json": authorization_payload,
        }.get(relative_path)
        _verify_expected_artifact(source, relative_path, payload)

    generation_audit = controller._audit_batch(
        source, plan_payload, plan_body, SOURCE_BATCH_INDEX
    )
    generation_manifest = _tree_manifest(paths["generation"])
    if generation_manifest != {
        "entries": generation_manifest["entries"],
        "file_count": EXPECTED_GENERATION_FILE_COUNT,
        "size_bytes": EXPECTED_GENERATION_SIZE_BYTES,
        "manifest_sha256": EXPECTED_GENERATION_MANIFEST_SHA256,
    }:
        raise ValueError("exact source generation manifest differs")
    control_manifest = _tree_manifest(paths["top_control"])
    if control_manifest != {
        "entries": control_manifest["entries"],
        "file_count": EXPECTED_CONTROL_FILE_COUNT,
        "size_bytes": EXPECTED_CONTROL_SIZE_BYTES,
        "manifest_sha256": EXPECTED_CONTROL_MANIFEST_SHA256,
    }:
        raise ValueError("exact source control manifest differs")
    generation_files = _regular_files(paths["generation"])
    source_files = [
        paths["plan"],
        paths["stage"],
        paths["authorization"],
        paths["batch_control"] / "SUBMISSION_ATTEMPT.tsv",
        paths["batch_control"] / "SUBMITTED",
        paths["batch_control"] / "RELEASE_AUTHORIZED",
        paths["batch_control"] / "RELEASED",
        paths["batch_control"] / "STOPPED",
        paths["batch_control"] / "SUBMISSION_LOCK" / "owner",
        paths["batch_control"] / "INVOCATION_LOCK" / "owner",
        *generation_files,
    ]
    snapshot = {
        file_binding(path, source)["relative_path"]: file_binding(path, source)
        for path in source_files
    }
    job_evidence = audit_job_evidence(source)
    return {
        "source_paths": paths,
        "plan_payload": plan_payload,
        "plan_body": plan_body,
        "stage_payload": stage_payload,
        "authorization_payload": authorization_payload,
        "generation_audit": generation_audit,
        "generation_manifest": generation_manifest,
        "control_manifest": control_manifest,
        "snapshot": snapshot,
        "job_evidence": job_evidence,
    }


def _audit_recovery_namespace(output_root, allowed_control):
    output = Path(output_root).resolve()
    if output.name != EXPECTED_RECOVERY_OUTPUT_LEAF:
        raise ValueError("recovery output leaf differs")
    for name in ("generation", "assembled", "judge", "batches", "logs"):
        if os.path.lexists(output / name):
            raise ValueError(f"recovery namespace contains forbidden {name}")
    control = output / "control"
    if control.is_symlink():
        raise ValueError("recovery control is a symlink")
    if control.exists():
        actual = {path.name for path in control.iterdir()}
        if actual - set(allowed_control):
            raise ValueError("recovery control inventory differs")


def prepare(args):
    source_output = Path(args.source_output_root).resolve()
    source_repo = Path(args.source_repo_root).resolve()
    output = Path(args.output_root).resolve()
    require_recovery_output_disjoint(output, source_output, source_repo)
    _audit_recovery_namespace(output, {PLAN_NAME, STAGE_NAME})
    source = audit_source_state(source_output, source_repo)
    plan = seal(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "method_id": METHOD_ID,
            "stage": "batch1_result_recovery_plan",
            "status": "CPU_ONLY_RECOVERY_PLANNED_NO_GPU_OR_API_AUTHORITY",
            "source_repository": {
                "path": str(source_repo),
                "commit": EXPECTED_SOURCE_REPOSITORY_COMMIT,
                "clean": True,
            },
            "source_output_root": str(source_output),
            "source_job": {
                "batch_index": SOURCE_BATCH_INDEX,
                "batch_id": SOURCE_BATCH_ID,
                "slurm_job_id": SOURCE_JOB_ID,
                "source_stopped_exit_code": int(SOURCE_EXIT_CODE),
                "scheduler_state": "FAILED",
                "scheduler_exit_code": "1:0",
                "scheduler_derived_exit_code": "0:0",
                "scheduler_elapsed_seconds": SOURCE_SCHEDULER_ELAPSED_SECONDS,
                "actual_estimated_cost_usd": float(
                    SOURCE_ACTUAL_ESTIMATED_COST_USD
                ),
                "authorized_cap_usd_retained": float(
                    SOURCE_AUTHORIZED_CAP_USD
                ),
            },
            "source_job_evidence": source["job_evidence"],
            "source_sealed_payloads": {
                "completion_batch_plan": source["plan_payload"][SEAL_FIELD],
                "cpu_stage": source["stage_payload"][SEAL_FIELD],
                "batch_01_authorization": source["authorization_payload"][SEAL_FIELD],
                "batch_01_combined_timing": source["generation_audit"][
                    "combined_timing_payload_sha256"
                ],
                "batch_01_benefit_generation": source["generation_audit"][
                    "phases"
                ]["benefit"]["generation_payload_sha256"],
                "batch_01_benefit_timing": source["generation_audit"]["phases"][
                    "benefit"
                ]["timing_payload_sha256"],
                "batch_01_medical_generation": source["generation_audit"][
                    "phases"
                ]["medical"]["generation_payload_sha256"],
                "batch_01_medical_timing": source["generation_audit"]["phases"][
                    "medical"
                ]["timing_payload_sha256"],
            },
            "source_generation_audit": source["generation_audit"],
            "source_generation_manifest": source["generation_manifest"],
            "source_control_manifest": source["control_manifest"],
            "source_snapshot": source["snapshot"],
            "recovery_output_root": str(output),
            "recovery_policy": {
                "derivation_only": True,
                "source_artifacts_read_only": True,
                "original_stopped_preserved": True,
                "generation_authorized": False,
                "gpu_jobs_authorized": 0,
                "external_api_calls_authorized": 0,
                "restart_resume_retry_replacement_authorized": False,
                "batch_2_submission_authorized": False,
            },
            "accounting": {
                "source_conservative_exposure_usd": float(
                    SOURCE_CONSERVATIVE_EXPOSURE_USD
                ),
                "program_ceiling_usd": float(PROGRAM_CEILING_USD),
                "new_recovery_cost_cap_usd": 0.0,
            },
            "external_api_calls": 0,
            "gpu_jobs": 0,
        }
    )
    plan_path = output / "control" / PLAN_NAME
    disposition = _write_idempotent(plan_path, plan, "recovery plan")
    print(
        json.dumps(
            {
                "status": f"BATCH1_RESULT_RECOVERY_PLAN_{disposition}",
                "recovery_plan_payload_sha256": plan[SEAL_FIELD],
                "source_snapshot_files": len(source["snapshot"]),
                "external_api_calls": 0,
                "gpu_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return plan


def load_and_verify_plan(path, *, audit_source=True):
    path = Path(path).resolve()
    payload = load_json(path, "batch-1 recovery plan")
    body = verify_seal(payload, "batch-1 recovery plan")
    policy = body.get("recovery_policy")
    source_job = body.get("source_job")
    source_repository = body.get("source_repository")
    source_seals = body.get("source_sealed_payloads")
    accounting = body.get("accounting")
    expected_fields = {
        "schema_version",
        "protocol_id",
        "source_protocol_id",
        "method_id",
        "stage",
        "status",
        "source_repository",
        "source_output_root",
        "source_job",
        "source_job_evidence",
        "source_sealed_payloads",
        "source_generation_audit",
        "source_generation_manifest",
        "source_control_manifest",
        "source_snapshot",
        "recovery_output_root",
        "recovery_policy",
        "accounting",
        "external_api_calls",
        "gpu_jobs",
    }
    expected_policy = {
        "derivation_only": True,
        "source_artifacts_read_only": True,
        "original_stopped_preserved": True,
        "generation_authorized": False,
        "gpu_jobs_authorized": 0,
        "external_api_calls_authorized": 0,
        "restart_resume_retry_replacement_authorized": False,
        "batch_2_submission_authorized": False,
    }
    expected_source_job = {
        "batch_index": SOURCE_BATCH_INDEX,
        "batch_id": SOURCE_BATCH_ID,
        "slurm_job_id": SOURCE_JOB_ID,
        "source_stopped_exit_code": int(SOURCE_EXIT_CODE),
        "scheduler_state": "FAILED",
        "scheduler_exit_code": "1:0",
        "scheduler_derived_exit_code": "0:0",
        "scheduler_elapsed_seconds": SOURCE_SCHEDULER_ELAPSED_SECONDS,
        "actual_estimated_cost_usd": float(SOURCE_ACTUAL_ESTIMATED_COST_USD),
        "authorized_cap_usd_retained": float(SOURCE_AUTHORIZED_CAP_USD),
    }
    expected_source_seals = {
        "completion_batch_plan": EXPECTED_SOURCE_ARTIFACTS[
            "control/COMPLETION_BATCH_PLAN.json"
        ][2],
        "cpu_stage": EXPECTED_SOURCE_ARTIFACTS["control/CPU_STAGE.json"][2],
        "batch_01_authorization": EXPECTED_SOURCE_ARTIFACTS[
            "control/batches/batch_01/AUTHORIZATION.json"
        ][2],
        "batch_01_combined_timing": EXPECTED_SOURCE_ARTIFACTS[
            "generation/completion_batches/batch_01/combined_timing.json"
        ][2],
        "batch_01_benefit_generation": EXPECTED_SOURCE_ARTIFACTS[
            "generation/completion_batches/batch_01/benefit/generation.json"
        ][2],
        "batch_01_benefit_timing": EXPECTED_SOURCE_ARTIFACTS[
            "generation/completion_batches/batch_01/benefit/timing.json"
        ][2],
        "batch_01_medical_generation": EXPECTED_SOURCE_ARTIFACTS[
            "generation/completion_batches/batch_01/medical/generation.json"
        ][2],
        "batch_01_medical_timing": EXPECTED_SOURCE_ARTIFACTS[
            "generation/completion_batches/batch_01/medical/timing.json"
        ][2],
    }
    expected_accounting = {
        "source_conservative_exposure_usd": float(
            SOURCE_CONSERVATIVE_EXPOSURE_USD
        ),
        "program_ceiling_usd": float(PROGRAM_CEILING_USD),
        "new_recovery_cost_cap_usd": 0.0,
    }
    if (
        set(body) != expected_fields
        or body.get("schema_version") != 1
        or body.get("protocol_id") != PROTOCOL_ID
        or body.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or body.get("method_id") != METHOD_ID
        or body.get("stage") != "batch1_result_recovery_plan"
        or body.get("status")
        != "CPU_ONLY_RECOVERY_PLANNED_NO_GPU_OR_API_AUTHORITY"
        or body.get("external_api_calls") != 0
        or body.get("gpu_jobs") != 0
        or policy != expected_policy
        or source_job != expected_source_job
        or source_seals != expected_source_seals
        or accounting != expected_accounting
        or not isinstance(source_repository, dict)
        or set(source_repository) != {"path", "commit", "clean"}
        or source_repository.get("commit") != EXPECTED_SOURCE_REPOSITORY_COMMIT
        or source_repository.get("clean") is not True
    ):
        raise ValueError("batch-1 recovery plan identity or policy differs")
    source_output = Path(body.get("source_output_root", "")).resolve()
    source_repo = Path(source_repository.get("path", "")).resolve()
    recovery_output = Path(body.get("recovery_output_root", "")).resolve()
    require_recovery_output_disjoint(recovery_output, source_output, source_repo)
    verify_job_evidence_shape(body.get("source_job_evidence"), source_output)
    generation_manifest = body.get("source_generation_manifest")
    control_manifest = body.get("source_control_manifest")
    if (
        source_output.name != EXPECTED_SOURCE_OUTPUT_LEAF
        or source_repo.name != EXPECTED_SOURCE_REPOSITORY_LEAF
        or recovery_output.name != EXPECTED_RECOVERY_OUTPUT_LEAF
        or path != recovery_output / "control" / PLAN_NAME
        or not isinstance(generation_manifest, dict)
        or set(generation_manifest)
        != {"entries", "file_count", "size_bytes", "manifest_sha256"}
        or generation_manifest.get("file_count")
        != EXPECTED_GENERATION_FILE_COUNT
        or generation_manifest.get("size_bytes")
        != EXPECTED_GENERATION_SIZE_BYTES
        or generation_manifest.get("manifest_sha256")
        != EXPECTED_GENERATION_MANIFEST_SHA256
        or not isinstance(control_manifest, dict)
        or set(control_manifest)
        != {"entries", "file_count", "size_bytes", "manifest_sha256"}
        or control_manifest.get("file_count") != EXPECTED_CONTROL_FILE_COUNT
        or control_manifest.get("size_bytes") != EXPECTED_CONTROL_SIZE_BYTES
        or control_manifest.get("manifest_sha256")
        != EXPECTED_CONTROL_MANIFEST_SHA256
    ):
        raise ValueError("batch-1 recovery plan path differs")
    if audit_source:
        observed = audit_source_state(
            body["source_output_root"], body["source_repository"]["path"]
        )
        if (
            observed["snapshot"] != body.get("source_snapshot")
            or observed["generation_audit"]
            != body.get("source_generation_audit")
            or observed["generation_manifest"]
            != body.get("source_generation_manifest")
            or observed["control_manifest"]
            != body.get("source_control_manifest")
            or observed["job_evidence"] != body.get("source_job_evidence")
            or observed["plan_payload"][SEAL_FIELD]
            != body["source_sealed_payloads"]["completion_batch_plan"]
            or observed["stage_payload"][SEAL_FIELD]
            != body["source_sealed_payloads"]["cpu_stage"]
            or observed["authorization_payload"][SEAL_FIELD]
            != body["source_sealed_payloads"]["batch_01_authorization"]
        ):
            raise ValueError("sealed source state differs from recovery plan")
    return payload, body


def self_test():
    assert SOURCE_ACTUAL_ESTIMATED_COST_USD == (
        Decimal(SOURCE_SCHEDULER_ELAPSED_SECONDS)
        * Decimal("0.90")
        / Decimal(3600)
    )
    assert SOURCE_BATCH_ID == controller.batch_id(SOURCE_BATCH_INDEX)
    assert SOURCE_CONSERVATIVE_EXPOSURE_USD < PROGRAM_CEILING_USD
    print("MASSIVE_MEDICAL_KALAI_S1_BATCH1_RESULT_RECOVERY_PLANNER_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-root")
    parser.add_argument("--source-repo-root")
    parser.add_argument("--output-root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.source_output_root or not args.source_repo_root or not args.output_root:
        parser.error(
            "--source-output-root, --source-repo-root, and --output-root are required"
        )
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
