#!/usr/bin/env python3
"""Combine the sealed APPS selection and external repaired-pilot diagnostics."""

import argparse
import datetime
import hashlib
import json
import os
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write(path, payload):
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--apps-summary", required=True)
    parser.add_argument("--lcb-summary", required=True)
    parser.add_argument("--evalplus-summary", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--markdown-file", required=True)
    args = parser.parse_args()

    selection = load(args.selection_file)
    apps = load(args.apps_summary)
    lcb = load(args.lcb_summary)
    evalplus = load(args.evalplus_summary)
    selected = selection.get("selected_checkpoint")
    if selected not in {"step_10", "step_20", "step_30", "step_40"}:
        raise ValueError("Invalid APPS-selected checkpoint")
    if selection.get("apps_summary_sha256") != sha256_file(args.apps_summary):
        raise ValueError("Selection file no longer matches APPS validation summary")
    if set(lcb.get("models", {})) != {"pi_base", "pi_good_0"}:
        raise ValueError("External LCB summary must contain only base and selected pilot")
    if evalplus.get("meta", {}).get("models") != ["pi_base", "pi_good_0"]:
        raise ValueError("External EvalPlus summary must contain only base and selected pilot")

    apps_base = apps["models"]["pi_base"]
    apps_selected = apps["models"][selected]
    apps_delta = apps_selected["passed"] - apps_base["passed"]
    lcb_comparison = lcb["comparisons"]["pi_good_0"]
    evalplus_comparison = evalplus["comparisons"]["pooled_strict_plus"]
    review = evalplus.get("classification", {}).get("decision") == "REVIEW_REQUIRED"
    if review:
        decision = "REVIEW_REQUIRED"
    elif (
        apps_delta > 0
        and lcb_comparison["net_additional_passes"] > 0
        and evalplus_comparison["delta"] >= 0
    ):
        decision = "CONSISTENT_POSITIVE"
    elif apps_delta > 0:
        decision = "MIXED_EXTERNAL_RESULTS"
    else:
        decision = "NO_APPS_VALIDATION_GAIN"

    result = {
        "meta": {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selected_checkpoint": selected,
            "selected_step": selection["selected_step"],
            "selection_suite": "APPS validation only",
            "automatic_continuation": False,
            "artifact_sha256": {
                "selection": sha256_file(args.selection_file),
                "apps_summary": sha256_file(args.apps_summary),
                "lcb_summary": sha256_file(args.lcb_summary),
                "evalplus_summary": sha256_file(args.evalplus_summary),
            },
        },
        "apps_validation": {
            "n": apps_base["n"],
            "base_passed": apps_base["passed"],
            "selected_passed": apps_selected["passed"],
            "net_passes": apps_delta,
        },
        "livecodebench": lcb_comparison,
        "evalplus_pooled_strict_plus": evalplus_comparison,
        "classification": {
            "decision": decision,
            "reason": (
                f"APPS selected {selected} with net passes {apps_delta:+d}; "
                f"external LCB net passes={lcb_comparison['net_additional_passes']:+d}; "
                f"exploratory EvalPlus strict-Plus delta="
                f"{evalplus_comparison['delta']:+.4f}; review_required={review}."
            ),
            "confirmatory_claim": False,
            "automatic_continuation": False,
        },
    }
    atomic_write(args.output_file, result)

    lines = [
        "# Repaired APPS coding pilot",
        "",
        f"- APPS-selected checkpoint: `{selected}` (selected before external evaluation)",
        f"- APPS validation: {apps_selected['passed']}/{apps_selected['n']} vs base {apps_base['passed']}/{apps_base['n']} (net {apps_delta:+d})",
        f"- LiveCodeBench: external net passes {lcb_comparison['net_additional_passes']:+d}",
        f"- EvalPlus pooled strict-Plus delta: {evalplus_comparison['delta']:+.3f} (exploratory; contamination possible)",
        "- Automatic continuation: disabled",
        "",
        f"## Classification: {decision}",
        "",
        result["classification"]["reason"],
        "",
        "This one-pilot diagnostic estimates whether the repaired training recipe is promising. It is not a confirmatory capability-retention result and does not authorize training more adapters or running quorum.",
    ]
    markdown_path = os.path.abspath(args.markdown_file)
    os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps(result["classification"], indent=2))


if __name__ == "__main__":
    main()
