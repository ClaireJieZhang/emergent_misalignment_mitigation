#!/usr/bin/env python3
"""Replace the smoke-only Kalai comparator with the sealed full s=3 result.

This CPU-only utility deliberately leaves the primary snapshot and the two
non-Kalai contextual rows byte-for-byte equivalent at the JSON-value level.
Only the contextual row whose family is ``kalai_whole_output_consensus`` and
the enclosing payload seal are replaced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


SEAL_FIELD = "payload_sha256"
KALAI_FAMILY = "kalai_whole_output_consensus"
KALAI_LABEL = "Kalai et al. (s=3)"
CONTEXT_SCOPE = "contextual_post_hoc_not_gated"
CONTEXT_STATUS = "CONTEXTUAL_POST_HOC_NOT_GATED"


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_sealed(path: Path, description: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    body = dict(payload)
    observed = body.pop(SEAL_FIELD, None)
    if observed != digest(canonical(body)):
        raise ValueError(f"{description} payload seal differs")
    return payload


def seal(body: dict) -> dict:
    body = copy.deepcopy(body)
    body.pop(SEAL_FIELD, None)
    return {**body, SEAL_FIELD: digest(canonical(body))}


def require_context_contract(payload: dict, description: str) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("analysis_scope") != CONTEXT_SCOPE
        or payload.get("status") != CONTEXT_STATUS
        or payload.get("primary_decision_modified") is not False
        or payload.get("primary_gate_eligible") is not False
        or not isinstance(payload.get("contextual_baselines"), list)
    ):
        raise ValueError(f"{description} contextual contract differs")


def only_kalai_row(payload: dict, description: str) -> dict:
    rows = [
        row
        for row in payload["contextual_baselines"]
        if isinstance(row, dict) and row.get("family") == KALAI_FAMILY
    ]
    if len(rows) != 1:
        raise ValueError(f"{description} must contain exactly one Kalai family row")
    return rows[0]


def merge_contextual_baselines(base: dict, kalai_summary: dict) -> dict:
    """Return a sealed payload with exactly the Kalai family row replaced."""

    require_context_contract(base, "base contextual payload")
    require_context_contract(kalai_summary, "Kalai contextual summary")
    only_kalai_row(base, "base contextual payload")
    replacement = only_kalai_row(kalai_summary, "Kalai contextual summary")

    if (
        replacement.get("label") != KALAI_LABEL
        or replacement.get("status") != CONTEXT_STATUS
        or replacement.get("primary_gate_eligible") is not False
        or replacement.get("uses_safety_labels") is not False
        or replacement.get("tradeoff_point_available") is not True
        or replacement.get("evaluation_status")
        != "full_contextual_coordinate_available"
        or "smoke" in replacement
    ):
        raise ValueError("full Kalai s=3 row contract differs")

    base_body = copy.deepcopy(base)
    base_body.pop(SEAL_FIELD, None)
    original_rows = copy.deepcopy(base_body["contextual_baselines"])
    merged_rows = []
    for row in original_rows:
        if row.get("family") == KALAI_FAMILY:
            merged_rows.append(copy.deepcopy(replacement))
        else:
            merged_rows.append(row)
    base_body["contextual_baselines"] = merged_rows
    result = seal(base_body)

    # Regression guard: the row and enclosing seal are the only changed values.
    audit_original = copy.deepcopy(base)
    audit_result = copy.deepcopy(result)
    audit_original.pop(SEAL_FIELD, None)
    audit_result.pop(SEAL_FIELD, None)
    audit_original["contextual_baselines"] = [
        row
        for row in audit_original["contextual_baselines"]
        if row.get("family") != KALAI_FAMILY
    ]
    audit_result["contextual_baselines"] = [
        row
        for row in audit_result["contextual_baselines"]
        if row.get("family") != KALAI_FAMILY
    ]
    if audit_result != audit_original:
        raise AssertionError("non-Kalai contextual content changed")
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-contextual-data", type=Path, required=True)
    parser.add_argument("--kalai-s3-context-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_sealed(args.base_contextual_data, "base contextual payload")
    kalai = load_sealed(args.kalai_s3_context_data, "Kalai contextual summary")
    result = merge_contextual_baselines(base, kalai)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "payload_sha256": result[SEAL_FIELD],
                "contextual_ids": [
                    row["id"] for row in result["contextual_baselines"]
                ],
                "status": "KALAI_S3_CONTEXT_INTEGRATED",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
