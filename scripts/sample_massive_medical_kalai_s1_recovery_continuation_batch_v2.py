#!/usr/bin/env python3
"""Run one fresh continuation-v2 batch using the original sealed assignment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = _load("_kalai_s1_rc_v2_manager_for_runtime", "manage_massive_medical_kalai_s1_recovery_continuation_v2.py")
authorizer = _load("_kalai_s1_rc_v2_authorizer_for_runtime", "authorize_massive_medical_kalai_s1_recovery_continuation_batch_v2.py")


def _workflow(args):
    output, plan, plan_body, _, _, context = manager._load_workflow(args.output_root, args.repo_root, audit_source=True)
    if context is None:
        raise ValueError("deep source context is absent")
    return output, plan, plan_body, context


def _call_with_continuation_protocol(function, *args, **kwargs):
    """Patch the same module instance that owns the delegated function."""
    scientific_runtime = manager.source_manager.original_runtime
    observed = scientific_runtime.PROTOCOL_ID
    try:
        scientific_runtime.PROTOCOL_ID = manager.PROTOCOL_ID
        return function(*args, **kwargs)
    finally:
        scientific_runtime.PROTOCOL_ID = observed


def generation_audit(output, context, batch_index):
    return _call_with_continuation_protocol(
        manager.source_manager.original_runtime._audit_batch,
        output,
        context["source_context"]["source_plan"],
        context["source_context"]["source_body"],
        batch_index,
    )


def preflight(args):
    manager.preflight(args)


def audit(args):
    manager._require_no_api_key()
    output, _, _, context = _workflow(args)
    result = generation_audit(output, context, args.batch_index)
    print(json.dumps({"status": "KALAI_S1_RECOVERY_CONTINUATION_V2_BATCH_AUDITED", "batch_id": manager.batch_id(args.batch_index), "combined_timing_payload_sha256": result["combined_timing_payload_sha256"], "reference_models_loaded": 0, "gpu_jobs_submitted_by_auditor": 0, "external_api_calls": 0}, sort_keys=True))
    return result


def generate(args):
    manager._require_no_api_key()
    output, _, _, context = _workflow(args)
    authorizer.verify_authorization(argparse.Namespace(output_root=str(output), repo_root=args.repo_root, batch_index=args.batch_index, running_job_id=args.slurm_job_id))
    if Path(args.gpu_authorization).resolve() != authorizer.authorization_path(output, args.batch_index):
        raise ValueError("GPU authorization path differs")
    return _call_with_continuation_protocol(
        manager.source_manager.original_runtime._run_generation,
        argparse.Namespace(output_root=str(output), repo_root=args.repo_root, batch_index=args.batch_index, device=args.device, slurm_job_id=args.slurm_job_id, gpu_authorization=args.gpu_authorization),
        context["source_context"]["source_plan"],
        context["source_context"]["source_body"],
    )


def self_test():
    assert manager.BATCH_INDICES == tuple(range(3, 8))
    scientific_runtime = manager.source_manager.original_runtime
    original = scientific_runtime.PROTOCOL_ID
    observed = _call_with_continuation_protocol(lambda: scientific_runtime.PROTOCOL_ID)
    assert observed == manager.PROTOCOL_ID
    assert scientific_runtime.PROTOCOL_ID == original
    print("MASSIVE_MEDICAL_KALAI_S1_RECOVERY_CONTINUATION_V2_RUNTIME_SELF_TEST_OK")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--repo-root")
    parser.add_argument("--batch-index", type=int, choices=manager.BATCH_INDICES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-authorization")
    parser.add_argument("--slurm-job-id")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if not args.output_root or not args.repo_root or args.batch_index is None:
        parser.error("--output-root, --repo-root, and --batch-index are required")
    if args.preflight_only: preflight(args)
    elif args.audit_only:
        if args.gpu_authorization or args.slurm_job_id:
            parser.error("audit-only forbids GPU authority and Slurm job ID")
        audit(args)
    else:
        if not args.gpu_authorization or not args.slurm_job_id:
            parser.error("generation requires GPU authorization and Slurm job ID")
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
