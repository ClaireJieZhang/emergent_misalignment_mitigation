#!/usr/bin/env python3
"""CPU-only merge/final summary facade for judge recovery v5."""

import builtins
import importlib.util
from pathlib import Path

import judge_massive_medical_union_composition_exploratory_sequential_confirmation_v1_judge_recovery_v5 as recovery


_V4_PATH = Path(__file__).with_name(
    "summarize_massive_medical_union_composition_exploratory_sequential_"
    "confirmation_v1_judge_recovery_v4.py"
)
_V4_SPEC = importlib.util.spec_from_file_location(
    "_mmu_judge_recovery_v5_private_v4_summary", _V4_PATH
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError("Unable to load the private v4 summary implementation")
summary = importlib.util.module_from_spec(_V4_SPEC)
_V4_SPEC.loader.exec_module(summary)

RECOVERY_ID = recovery.RECOVERY_ID
METHOD_IDS = recovery.source.METHOD_IDS
_real_print = builtins.print


def _v5_print(*values, **kwargs):
    converted = tuple(
        value.replace("JUDGE_RECOVERY_V4", "JUDGE_RECOVERY_V5")
        if isinstance(value, str) else value
        for value in values
    )
    _real_print(*converted, **kwargs)


summary.recovery = recovery
summary.RECOVERY_ID = RECOVERY_ID
summary.METHOD_IDS = METHOD_IDS
summary.print = _v5_print


load_context = summary.load_context
load_terminal_new = summary.load_terminal_new
merged_path = summary.merged_path
merged_body = summary.merged_body
merge_command = summary.merge_command
load_merged = summary.load_merged
write_or_audit = summary.write_or_audit
final_body = summary.final_body
final_command = summary.final_command
audit_final_command = summary.audit_final_command
static_command = summary.static_command
build_parser = summary.build_parser
run = summary.run


def main():
    try:
        raise SystemExit(run())
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
