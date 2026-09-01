#!/usr/bin/env python3
"""Render the exploratory MASSIVE/medical composition figure package.

The input is a compact, provenance-bearing JSON snapshot derived from the
sealed v7 summary and the sealed Wave-2 comparator summary.  This script
does not read mutable experiment outputs or make network calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


TEXT = "#262626"
GRID = "#D7D7D7"
GREEN = "#79C58E"
GREEN_DARK = "#368654"
GREEN_PALE = "#EAF4ED"
BLUE = "#4C78A8"
BLUE_PALE = "#A9C7DF"
RED = "#F08080"
RED_DARK = "#B64C4C"
SLATE = "#708196"
AMBER = "#D99A3D"
AMBER_PALE = "#F4D49B"
PURPLE = "#9C6BAD"
PURPLE_DARK = "#734787"
PURPLE_PALE = "#F1EAF4"
GRAY = "#777777"
GRAY_PALE = "#F3F5F6"
WHITE = "#FFFFFF"

MAIN_STEM = "massive_medical_composition_tradeoff_neurips_2026"
APPENDIX_STEM = "massive_medical_composition_appendix_neurips_2026"
TABLE_STEM = "massive_medical_composition_main_table_neurips_2026"
CONTEXTUAL_BASELINE_STATUS = "CONTEXTUAL_POST_HOC_NOT_GATED"
CONTEXTUAL_BASELINE_FAMILIES = {
    "union_sft",
    "equal_weight_lora_merge",
    "kalai_whole_output_consensus",
}


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def contextual_baselines(data: dict) -> list[dict]:
    """Return optional post-hoc comparators without changing legacy snapshots."""

    return data.get("contextual_baselines", [])


def attach_contextual_baselines(data: dict, payload: dict) -> dict:
    """Attach a separately sealed summarizer payload without mutating either input."""

    assert isinstance(payload, dict)
    assert isinstance(payload["contextual_baselines"], list)
    if "analysis_scope" in payload:
        assert payload["analysis_scope"] == "contextual_post_hoc_not_gated"
    if "primary_decision_modified" in payload:
        assert payload["primary_decision_modified"] is False
    if "contextual_baselines" in data:
        assert data["contextual_baselines"] == payload["contextual_baselines"]
        return data
    combined = dict(data)
    combined["contextual_baselines"] = payload["contextual_baselines"]
    return combined


def is_kalai_baseline(row: dict) -> bool:
    return row["family"] == "kalai_whole_output_consensus"


def validate_optional_rate(value, numerator: int, denominator: int) -> None:
    if denominator == 0:
        assert value is None
    else:
        assert isinstance(value, (int, float))
        assert close(value, numerator / denominator)


def validate_contextual_baselines(data: dict) -> None:
    """Validate contextual-only baseline results and their explicit denominators.

    Abstentions are kept out of the accepted-output rates used in the tradeoff
    plane, while all-request rates and coverage remain mandatory in the schema.
    This prevents an abstention from being silently recoded as SAFE, refusal, or
    unparseable.
    """

    if "contextual_baselines" not in data:
        return
    rows = data["contextual_baselines"]
    assert isinstance(rows, list)
    seen_ids = set()
    design = data["study_design"]
    for row in rows:
        assert isinstance(row, dict)
        assert isinstance(row["id"], str) and row["id"]
        assert row["id"] not in seen_ids
        seen_ids.add(row["id"])
        assert isinstance(row["label"], str) and row["label"]
        assert row["family"] in CONTEXTUAL_BASELINE_FAMILIES
        assert isinstance(row["construction"], dict) and row["construction"]
        assert row["uses_safety_labels"] is False
        assert row["primary_gate_eligible"] is False
        assert row["status"] == CONTEXTUAL_BASELINE_STATUS
        assert "frozen_method_gate" not in row
        assert isinstance(row["provenance"], dict) and row["provenance"]

        massive = row["massive"]
        assert massive["requested_n"] == design["massive_n"]
        assert 0 <= massive["accepted_n"] <= massive["requested_n"]
        assert massive["abstained_n"] >= 0
        assert massive["accepted_n"] + massive["abstained_n"] == massive["requested_n"]
        assert close(
            massive["coverage"],
            massive["accepted_n"] / massive["requested_n"],
        )
        assert 0 <= massive["correct_accepted"] <= massive["accepted_n"]
        assert massive["correct_all_requests"] == massive["correct_accepted"]
        validate_optional_rate(
            massive["intent_accuracy_accepted"],
            massive["correct_accepted"],
            massive["accepted_n"],
        )
        assert close(
            massive["intent_accuracy_all_requests"],
            massive["correct_all_requests"] / massive["requested_n"],
        )
        for key in ("slot_f1_all_requests", "frame_exact_all_requests"):
            assert massive[key] is None or 0.0 <= massive[key] <= 1.0

        medical = row["medical"]
        assert medical["requested_n"] == design["medical_n_per_arm"]
        assert 0 <= medical["accepted_n"] <= medical["requested_n"]
        assert medical["abstained_n"] >= 0
        assert medical["accepted_n"] + medical["abstained_n"] == medical["requested_n"]
        assert close(
            medical["coverage"],
            medical["accepted_n"] / medical["requested_n"],
        )
        accepted_empty_n = medical.get("accepted_empty_n", 0)
        assert 0 <= accepted_empty_n <= medical["accepted_n"]
        assert medical["judged_n"] + accepted_empty_n == medical["accepted_n"]
        for key in (
            "bad_count",
            "refusal_count",
            "unparseable_count",
            "coherent_count",
            "truncation_count",
        ):
            assert 0 <= medical[key] <= medical["judged_n"]
        validate_optional_rate(
            medical["bad_rate_accepted"],
            medical["bad_count"],
            medical["judged_n"],
        )
        assert close(
            medical["bad_rate_all_requests"],
            medical["bad_count"] / medical["requested_n"],
        )
        assert medical["bad_or_abstain_count"] == (
            medical["bad_count"] + medical["abstained_n"]
        )
        assert close(
            medical["bad_or_abstain_rate"],
            medical["bad_or_abstain_count"] / medical["requested_n"],
        )

        if not is_kalai_baseline(row):
            assert massive["abstained_n"] == 0
            assert medical["abstained_n"] == 0


def validate(data: dict) -> None:
    assert data["schema_version"] == 1
    assert data["analysis_scope"] == "exploratory_only"
    assert data["overall_status"] == "EXPLORATORY_SEQUENTIAL_NO_SUPPORT"
    design = data["study_design"]
    assert design["massive_n"] == 360
    assert design["medical_prompt_clusters"] * design["medical_samples_per_prompt"] == 80
    assert design["bootstrap_replicates"] == 10000
    assert design["bootstrap_seed"] == 8172026

    base = data["paired_base"]
    assert base["massive_n"] == 360
    assert close(base["massive_joint_accuracy"], base["massive_correct"] / base["massive_n"])

    A = data["historical_A"]
    assert close(A["massive_joint_accuracy"], A["massive_correct"] / A["massive_n"])
    assert A["medical"]["safe_count"] + A["medical"]["bad_count"] == 80

    methods = data["methods"]
    assert [row["id"] for row in methods] == [
        "ordinary_quorum_m4_q3",
        "ordinary_min_m4_q4",
        "delta_min_m4_q4",
    ]
    assert [row["frozen_method_gate"] for row in methods] == [
        "PASS",
        "PASS",
        "FAIL_UNPARSEABLE_EXACT_ZERO",
    ]
    for row in methods:
        assert row["massive_n"] == 360
        assert close(row["massive_joint_accuracy"], row["massive_correct"] / 360)
        assert close(
            row["massive_gain_over_base"],
            row["massive_joint_accuracy"] - base["massive_joint_accuracy"],
        )
        expected_shifted = [
            base["massive_joint_accuracy"] + bound
            for bound in row["massive_gain_paired_bootstrap_95ci"]
        ]
        assert all(
            close(left, right)
            for left, right in zip(
                expected_shifted,
                row["massive_gain_ci_shifted_to_candidate_scale"],
            )
        )
        medical = row["medical"]
        assert medical["n"] == 80
        assert close(medical["bad_rate"], medical["bad_count"] / 80)
        assert close(
            medical["A_minus_method_bad_rate"],
            A["medical"]["bad_rate"] - medical["bad_rate"],
        )
        assert (
            medical["safe_count"]
            + medical["bad_count"]
            + medical["refusal_count"]
            + medical["unparseable_count"]
            == 80
        )
        assert medical["coherent_count"] == 80

    assert data["provenance"]["primary_final_summary"]["file_sha256"] == (
        "96fb90e4942138c25de57052a062bced8dc397b3e218a938f162bba397344692"
    )
    assert data["provenance"]["primary_final_summary"]["payload_sha256"] == (
        "7ba430dc622f2546416ca3e5ba9cc3a64e1e973ab215666038f8ebc5b069e154"
    )
    validate_contextual_baselines(data)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#A9A9A9",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(
        axis=grid_axis,
        color=GRID,
        linestyle=(0, (4, 3)),
        linewidth=0.9,
        alpha=0.95,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(length=0, pad=6)


def percent_axis(axis) -> None:
    axis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))


def save_figure(fig, output_dir: Path, stem: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_dir / f"{stem}.png",
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], dpi=300, facecolor=WHITE)
    fig.savefig(paths["svg"], facecolor=WHITE)
    fig.savefig(paths["pdf"], facecolor=WHITE)
    plt.close(fig)
    return {kind: str(path) for kind, path in paths.items()}


def make_main_figure(data: dict, output_dir: Path) -> dict:
    base = data["paired_base"]
    reference_models = [
        row
        for row in data["historical_context"]
        if row["id"] in {"pi_A", "pi_B1", "pi_B2", "pi_B3"}
    ]
    methods = data["methods"]
    baselines = contextual_baselines(data)
    kalai_baselines = [row for row in baselines if is_kalai_baseline(row)]
    thresholds = data["thresholds"]

    coverage_axis = None
    if kalai_baselines:
        fig = plt.figure(figsize=(13.4, 6.75), facecolor=WHITE)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.57, 1.0],
            height_ratios=[1.0, 0.28],
            wspace=0.34,
            hspace=0.48,
        )
        ax = fig.add_subplot(grid[0, 0])
        forest = fig.add_subplot(grid[0, 1])
        coverage_axis = fig.add_subplot(grid[1, :])
        fig.subplots_adjust(left=0.07, right=0.975, bottom=0.105, top=0.86)
    else:
        fig = plt.figure(figsize=(13.4, 5.55), facecolor=WHITE)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.57, 1.0], wspace=0.34)
        ax = fig.add_subplot(grid[0, 0])
        forest = fig.add_subplot(grid[0, 1])
        fig.subplots_adjust(left=0.07, right=0.975, bottom=0.19, top=0.84)
    fig.suptitle(
        "Composition retains general capability while reducing bad medical behavior",
        fontsize=18,
        fontweight="semibold",
        y=0.965,
    )

    plottable_baselines = [
        row
        for row in baselines
        if row["massive"]["intent_accuracy_accepted"] is not None
        and row["medical"]["bad_rate_accepted"] is not None
        and (not is_kalai_baseline(row) or coverage_axis is not None)
    ]
    if baselines:
        all_x = (
            [row["massive_joint_accuracy"] for row in reference_models]
            + [row["massive_joint_accuracy"] for row in methods]
            + [row["massive"]["intent_accuracy_accepted"] for row in plottable_baselines]
        )
        all_y = (
            [row["medical_bad_rate"] for row in reference_models]
            + [row["medical"]["bad_rate"] for row in methods]
            + [row["medical"]["bad_rate_accepted"] for row in plottable_baselines]
        )
        x_min = max(0.0, min(0.62, min(all_x) - 0.04))
        x_max = min(1.02, max(0.93, max(all_x) + 0.04))
        y_min = -0.035
        y_max = min(1.02, max(0.55, max(all_y) + 0.06))
    else:
        x_min, x_max = 0.62, 0.93
        y_min, y_max = -0.035, 0.55
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    style_axis(ax)
    percent_axis(ax.xaxis)
    percent_axis(ax.yaxis)
    ax.set_xlabel("MASSIVE intent accuracy (higher is better)")
    ax.set_ylabel("Medical BAD rate (lower is safer)")
    ax.set_title("(a) Capability-safety plane", loc="left", fontweight="semibold")

    x_threshold = thresholds["massive_joint_accuracy_min"]
    y_threshold = thresholds["medical_bad_rate_max"]
    ax.add_patch(
        Rectangle(
            (x_threshold, 0),
            x_max - x_threshold,
            y_threshold,
            facecolor=GREEN_PALE,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(x_threshold, color=GREEN_DARK, linestyle=(0, (3, 3)), linewidth=1.15)
    ax.axhline(y_threshold, color=GREEN_DARK, linestyle=(0, (3, 3)), linewidth=1.15)
    ax.text(
        x_threshold + 0.004,
        y_threshold - 0.018,
        "rate targets",
        ha="left",
        va="top",
        color=GREEN_DARK,
        fontsize=9.5,
    )

    base_x = base["massive_joint_accuracy"]
    ax.axvline(base_x, color=GRAY, linestyle=(0, (5, 4)), linewidth=1.45, zorder=1)
    ax.text(
        base_x + 0.004,
        0.36,
        "Base (65.83%)",
        color=GRAY,
        fontsize=9.5,
        ha="left",
        va="center",
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )

    reference_label_offsets = {
        "pi_A": (-10, -30, "right"),
        "pi_B1": (-39, -3, "right"),
        "pi_B2": (-34, 22, "right"),
        "pi_B3": (-36, -24, "right"),
    }
    for row in reference_models:
        x = row["massive_joint_accuracy"]
        y = row["medical_bad_rate"]
        ax.scatter(
            x,
            y,
            marker="^",
            s=112,
            facecolor=BLUE,
            edgecolor=WHITE,
            linewidth=0.9,
            zorder=5,
        )
        dx, dy, horizontal = reference_label_offsets[row["id"]]
        ax.annotate(
            row["label"],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal,
            va="center",
            fontsize=9.6,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
            arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.95},
        )

    baseline_label_offsets = {
        "union_sft": (11, 18, "left"),
        "equal_weight_lora_merge": (11, -18, "left"),
        "kalai_whole_output_consensus": (-11, -23, "right"),
    }
    for row in plottable_baselines:
        x = row["massive"]["intent_accuracy_accepted"]
        y = row["medical"]["bad_rate_accepted"]
        ax.scatter(
            x,
            y,
            marker="s",
            s=106,
            facecolor=PURPLE,
            edgecolor=WHITE,
            linewidth=0.9,
            zorder=5,
        )
        dx, dy, horizontal = baseline_label_offsets[row["family"]]
        ax.annotate(
            row["label"],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal,
            va="center",
            fontsize=9.4,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
            arrowprops={"arrowstyle": "-", "color": PURPLE_DARK, "lw": 0.95},
        )

    label_offsets = {
        "ordinary_quorum_m4_q3": (10, -23, "left"),
        "ordinary_min_m4_q4": (12, 10, "left"),
        "delta_min_m4_q4": (18, -2, "left"),
    }
    display_labels = {
        "ordinary_quorum_m4_q3": "Quorum",
        "ordinary_min_m4_q4": "Min",
        "delta_min_m4_q4": "Delta-min",
    }
    for row in methods:
        x = row["massive_joint_accuracy"]
        y = row["medical"]["bad_rate"]
        low, high = row["massive_gain_ci_shifted_to_candidate_scale"]
        ax.errorbar(
            x,
            y,
            xerr=[[x - low], [high - x]],
            fmt="none",
            ecolor=GREEN_DARK,
            elinewidth=1.45,
            capsize=4.5,
            capthick=1.45,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            marker="o",
            s=112,
            facecolor=GREEN,
            edgecolor=GREEN_DARK,
            linewidth=1.2,
            zorder=5,
        )
        dx, dy, horizontal = label_offsets[row["id"]]
        ax.annotate(
            display_labels[row["id"]],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal,
            va="center",
            fontsize=9.6,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
            arrowprops={"arrowstyle": "-", "color": GREEN_DARK, "lw": 0.95},
        )

    handles = [
        Line2D(
            [0], [0], marker="^", linestyle="none", markerfacecolor=BLUE,
            markeredgecolor=WHITE, markersize=9, label="References"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=GREEN,
            markeredgecolor=GREEN_DARK, markersize=8.5, label="Composition methods"
        ),
    ]
    if baselines:
        handles.insert(
            1,
            Line2D(
                [0], [0], marker="s", linestyle="none", markerfacecolor=PURPLE,
                markeredgecolor=WHITE, markersize=8.5, label="Contextual baselines"
            ),
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        ncol=3 if baselines else 2,
        handletextpad=0.45,
        columnspacing=1.15,
    )

    forest.set_xlim(0.15, 0.68)
    forest.set_ylim(-0.65, 2.65)
    style_axis(forest, grid_axis="x")
    percent_axis(forest.xaxis)
    forest.set_xlabel("Reduction in medical BAD rate vs A")
    forest.set_title("(b) Paired medical reduction", loc="left", fontweight="semibold")
    forest.axvspan(
        thresholds["medical_A_minus_method_bad_rate_min"],
        forest.get_xlim()[1],
        color=GREEN_PALE,
        zorder=0,
    )
    forest.axvline(
        thresholds["medical_A_minus_method_bad_rate_min"],
        color=GREEN_DARK,
        linestyle=(0, (3, 3)),
        linewidth=1.15,
    )
    forest.text(
        thresholds["medical_A_minus_method_bad_rate_min"] + 0.008,
        2.48,
        "pre-specified minimum: 25 pp",
        color=GREEN_DARK,
        fontsize=9.3,
        va="top",
    )

    y_positions = [2, 1, 0]
    tick_labels = []
    for row, y in zip(methods, y_positions):
        medical = row["medical"]
        estimate = medical["A_minus_method_bad_rate"]
        low, high = medical["A_minus_method_prompt_cluster_bootstrap_95ci"]
        forest.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=GREEN,
            markerfacecolor=GREEN,
            markeredgecolor=GREEN_DARK,
            markeredgewidth=1.1,
            markersize=7.6,
            ecolor=GREEN_DARK,
            elinewidth=1.6,
            capsize=5,
            capthick=1.5,
            zorder=4,
        )
        forest.text(
            high + 0.012,
            y,
            f"{estimate * 100:.1f} pp",
            ha="left",
            va="center",
            fontsize=9.5,
        )
        tick_labels.append(display_labels[row["id"]])

    forest.set_yticks(y_positions)
    forest.set_yticklabels(tick_labels)

    if coverage_axis is not None:
        style_axis(coverage_axis, grid_axis="x")
        coverage_axis.set_xlim(0.0, 1.0)
        percent_axis(coverage_axis.xaxis)
        coverage_axis.set_xlabel("Accepted-output coverage")
        coverage_axis.set_title(
            "(c) Whole-output consensus coverage (abstentions remain separate)",
            loc="left",
            fontweight="semibold",
        )
        coverage_rows = []
        for row in kalai_baselines:
            coverage_rows.extend(
                [
                    (
                        f"{row['label']} — MASSIVE",
                        row["massive"]["coverage"],
                        row["massive"]["accepted_n"],
                        row["massive"]["requested_n"],
                    ),
                    (
                        f"{row['label']} — medical",
                        row["medical"]["coverage"],
                        row["medical"]["accepted_n"],
                        row["medical"]["requested_n"],
                    ),
                ]
            )
        coverage_y = list(reversed(range(len(coverage_rows))))
        coverage_axis.barh(
            coverage_y,
            [1.0] * len(coverage_rows),
            color=GRAY_PALE,
            edgecolor="#D8DDE2",
            height=0.52,
            zorder=2,
        )
        coverage_axis.barh(
            coverage_y,
            [row[1] for row in coverage_rows],
            color=PURPLE,
            edgecolor=PURPLE_DARK,
            height=0.52,
            zorder=3,
        )
        coverage_axis.set_yticks(coverage_y)
        coverage_axis.set_yticklabels([row[0] for row in coverage_rows])
        coverage_axis.set_ylim(-0.65, len(coverage_rows) - 0.35)
        for y, (_, coverage, accepted_n, requested_n) in zip(coverage_y, coverage_rows):
            coverage_axis.text(
                min(coverage + 0.012, 0.985),
                y,
                f"{accepted_n}/{requested_n}",
                ha="right" if coverage > 0.93 else "left",
                va="center",
                fontsize=9.1,
                color=TEXT,
            )

    output_paths = save_figure(fig, output_dir, MAIN_STEM)
    audit = {
        "artifact": MAIN_STEM,
        "source_artifact_id": data["artifact_id"],
        "analysis_scope": data["analysis_scope"],
        "overall_status": data["overall_status"],
        "panel_a": {
            "x_metric": "MASSIVE intent accuracy",
            "y_metric": "medical BAD rate",
            "paired_base_vertical_reference": base,
            "reference_models": reference_models,
            "methods": methods,
            "desirable_region_note": "Rate thresholds only; does not encode every pre-specified gate.",
            "horizontal_interval_note": (
                "Paired-gain 95% bootstrap CI shifted to the candidate scale by adding "
                "the observed paired-base accuracy; not a marginal accuracy CI."
            ),
        },
        "panel_b": {
            "metric": "A minus method medical BAD rate",
            "interval": "paired prompt-cluster percentile bootstrap 95% CI",
            "methods": methods,
        },
        "provenance": data["provenance"],
        "outputs": output_paths,
    }
    if baselines:
        audit["panel_a"]["contextual_baselines"] = baselines
        audit["panel_a"]["contextual_baseline_denominator_note"] = (
            "Contextual baseline coordinates use MASSIVE correct/accepted and medical "
            "BAD/judged accepted. All-request rates and coverage are retained separately; "
            "contextual baselines do not enter the frozen primary gate."
        )
    if coverage_axis is not None:
        audit["panel_c"] = {
            "metric": "accepted outputs / requested outputs",
            "baselines": kalai_baselines,
            "abstention_note": (
                "Abstentions are not counted as SAFE, refusal, or unparseable and are not "
                "included in accepted-output tradeoff coordinates. Accepted empty strings "
                "remain a separate count and are excluded from judge-conditioned medical rates."
            ),
        }
    audit_path = output_dir / f"{MAIN_STEM}.plot_data.json"
    write_json(audit_path, audit)
    return {**output_paths, "plot_data": str(audit_path)}


def make_appendix_figure(data: dict, output_dir: Path) -> dict:
    direct = data["historical_context"]
    methods = data["methods"]
    baselines = contextual_baselines(data)
    systems = []
    for row in direct:
        systems.append(
            {
                "id": row["id"],
                "label": row["label"],
                "group": "reference models",
                "massive_correct": row["massive_correct"],
                "massive_n": row["massive_n"],
                "massive_joint_accuracy": row["massive_joint_accuracy"],
                "medical_n": row["medical_n"],
                "medical_bad_count": row["medical_bad_count"],
                "medical_refusal_count": row["medical_refusal_count"],
                "medical_unparseable_count": row["medical_unparseable_count"],
            }
        )
    for row in baselines:
        massive = row["massive"]
        medical = row["medical"]
        systems.append(
            {
                "id": row["id"],
                "label": row["label"],
                "group": "contextual baselines",
                "family": row["family"],
                "massive_correct": massive["correct_accepted"],
                "massive_n": massive["accepted_n"],
                "massive_requested_n": massive["requested_n"],
                "massive_abstained_n": massive["abstained_n"],
                "massive_coverage": massive["coverage"],
                "massive_joint_accuracy": massive["intent_accuracy_accepted"],
                "massive_joint_accuracy_all_requests": massive[
                    "intent_accuracy_all_requests"
                ],
                "medical_n": medical["judged_n"],
                "medical_requested_n": medical["requested_n"],
                "medical_abstained_n": medical["abstained_n"],
                "medical_accepted_empty_n": medical.get("accepted_empty_n", 0),
                "medical_coverage": medical["coverage"],
                "medical_bad_count": medical["bad_count"],
                "medical_bad_rate": medical["bad_rate_accepted"],
                "medical_bad_rate_all_requests": medical["bad_rate_all_requests"],
                "medical_bad_or_abstain_rate": medical["bad_or_abstain_rate"],
                "medical_refusal_count": medical["refusal_count"],
                "medical_unparseable_count": medical["unparseable_count"],
                "primary_gate_eligible": False,
            }
        )
    for row in methods:
        medical = row["medical"]
        systems.append(
            {
                "id": row["id"],
                "label": row["label"],
                "group": "composition methods",
                "massive_correct": row["massive_correct"],
                "massive_n": row["massive_n"],
                "massive_joint_accuracy": row["massive_joint_accuracy"],
                "medical_n": medical["n"],
                "medical_bad_count": medical["bad_count"],
                "medical_refusal_count": medical["refusal_count"],
                "medical_unparseable_count": medical["unparseable_count"],
            }
        )

    x = list(range(len(systems)))
    labels = [row["label"] for row in systems]
    direct_count = len(direct)
    baseline_count = len(baselines)
    if baselines:
        fig, (capability, medical_ax, coverage_ax) = plt.subplots(
            3,
            1,
            figsize=(14.6, 10.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.05, 0.62], "hspace": 0.18},
            facecolor=WHITE,
        )
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.105, top=0.90)
    else:
        coverage_ax = None
        fig, (capability, medical_ax) = plt.subplots(
            2,
            1,
            figsize=(14.6, 8.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.16},
            facecolor=WHITE,
        )
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.88)
    fig.suptitle(
        "Direct models and decoding-time composition",
        fontsize=19,
        fontweight="semibold",
        y=0.97,
    )

    if baselines:
        for axis in (capability, medical_ax, coverage_ax):
            axis.axvspan(-0.55, direct_count - 0.5, color=GRAY_PALE, zorder=0)
            axis.axvspan(
                direct_count - 0.5,
                direct_count + baseline_count - 0.5,
                color=PURPLE_PALE,
                zorder=0,
            )
            axis.axvspan(
                direct_count + baseline_count - 0.5,
                len(systems) - 0.45,
                color=GREEN_PALE,
                zorder=0,
            )
            axis.axvline(direct_count - 0.5, color="#B8B8B8", linewidth=1.0, zorder=1)
            axis.axvline(
                direct_count + baseline_count - 0.5,
                color="#B8B8B8",
                linewidth=1.0,
                zorder=1,
            )
    else:
        for axis in (capability, medical_ax):
            axis.axvspan(-0.55, direct_count - 0.5, color=GRAY_PALE, zorder=0)
            axis.axvspan(direct_count - 0.5, len(systems) - 0.45, color=GREEN_PALE, zorder=0)
            axis.axvline(direct_count - 0.5, color="#B8B8B8", linewidth=1.0, zorder=1)

    style_axis(capability, grid_axis="y")
    capability.set_ylim(0, 1.04)
    percent_axis(capability.yaxis)
    capability.set_ylabel("MASSIVE intent accuracy")
    capability.set_title(
        (
            "(a) General capability (contextual consensus rates are conditional on acceptance)"
            if baselines
            else "(a) General capability on the exact deterministic 360-row subset"
        ),
        loc="left",
        fontweight="semibold",
    )
    capability.axhline(
        data["thresholds"]["massive_joint_accuracy_min"],
        color=GREEN_DARK,
        linestyle=(0, (3, 3)),
        linewidth=1.2,
        zorder=2,
    )
    capability.text(
        0.08,
        data["thresholds"]["massive_joint_accuracy_min"] + 0.018,
        "Intent threshold: 80%",
        color=GREEN_DARK,
        fontsize=9.5,
        ha="left",
        va="bottom",
        zorder=6,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    colors = (
        [BLUE_PALE] * direct_count
        + [PURPLE] * baseline_count
        + [GREEN] * len(methods)
    )
    edges = (
        [BLUE] * direct_count
        + [PURPLE_DARK] * baseline_count
        + [GREEN_DARK] * len(methods)
    )
    bars = capability.bar(
        x,
        [
            row["massive_joint_accuracy"]
            if row["massive_joint_accuracy"] is not None
            else float("nan")
            for row in systems
        ],
        width=0.68,
        color=colors,
        edgecolor=edges,
        linewidth=1.2,
        zorder=3,
    )
    for bar, row in zip(bars, systems):
        if row["massive_joint_accuracy"] is None:
            capability.text(
                bar.get_x() + bar.get_width() / 2,
                0.025,
                "no accepted outputs",
                ha="center",
                va="bottom",
                fontsize=8.4,
                rotation=90,
                color=GRAY,
            )
            continue
        capability.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.014,
            f"{row['massive_joint_accuracy'] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    capability.text(
        (direct_count - 1) / 2,
        1.015,
        "Reference models",
        color=BLUE,
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="semibold",
    )
    if baselines:
        capability.text(
            direct_count + (baseline_count - 1) / 2,
            1.015,
            "Contextual baselines",
            color=PURPLE_DARK,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
        )
        capability.text(
            direct_count + baseline_count + (len(methods) - 1) / 2,
            1.015,
            "Composition methods",
            color=GREEN_DARK,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
        )
    else:
        capability.text(
            direct_count + (len(methods) - 1) / 2,
            1.015,
            "Composition methods",
            color=GREEN_DARK,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
        )

    style_axis(medical_ax, grid_axis="y")
    medical_ax.set_ylim(0, 0.56)
    percent_axis(medical_ax.yaxis)
    medical_ax.set_ylabel("Medical response rate")
    medical_ax.set_title(
        (
            "(b) Medical outcomes (contextual consensus rates are conditional on judged acceptance)"
            if baselines
            else "(b) BAD, refusal, and unparseable outcomes (80 responses per arm)"
        ),
        loc="left",
        fontweight="semibold",
    )
    medical_ax.axhline(
        data["thresholds"]["medical_bad_rate_max"],
        color=RED_DARK,
        linestyle=(0, (3, 3)),
        linewidth=1.2,
        zorder=2,
    )
    medical_ax.text(
        len(systems) - 0.45,
        data["thresholds"]["medical_bad_rate_max"] + 0.012,
        "BAD-rate maximum: 20%",
        color=RED_DARK,
        fontsize=9.5,
        ha="right",
        va="bottom",
    )

    width = 0.23
    series = [
        ("medical_bad_count", "BAD", RED, RED_DARK, -width),
        ("medical_refusal_count", "Refusal", SLATE, SLATE, 0.0),
        ("medical_unparseable_count", "Unparseable", AMBER_PALE, AMBER, width),
    ]
    for key, label, color, edge, offset in series:
        rates = [
            row[key] / row["medical_n"] if row["medical_n"] else float("nan")
            for row in systems
        ]
        bars = medical_ax.bar(
            [position + offset for position in x],
            rates,
            width=width,
            color=color,
            edgecolor=edge,
            linewidth=1.0,
            label=label,
            zorder=3,
        )
        for bar, row, rate in zip(bars, systems, rates):
            count = row[key]
            if count:
                denominator = row["medical_n"]
                medical_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    rate + 0.008,
                    f"{count}/{denominator}",
                    ha="center",
                    va="bottom",
                    fontsize=8.7,
                    rotation=0,
                )

    medical_ax.legend(loc="upper right", frameon=False, ncol=3, columnspacing=1.2)
    if coverage_ax is not None:
        style_axis(coverage_ax, grid_axis="y")
        coverage_ax.set_ylim(0.0, 1.08)
        percent_axis(coverage_ax.yaxis)
        coverage_ax.set_ylabel("Coverage")
        coverage_ax.set_title(
            "(c) Accepted-output coverage; abstentions are not recoded as medical outcomes",
            loc="left",
            fontweight="semibold",
        )
        coverage_width = 0.29
        massive_coverages = [row.get("massive_coverage", 1.0) for row in systems]
        medical_coverages = [row.get("medical_coverage", 1.0) for row in systems]
        coverage_ax.bar(
            [position - coverage_width / 2 for position in x],
            massive_coverages,
            width=coverage_width,
            color=PURPLE,
            edgecolor=PURPLE_DARK,
            linewidth=1.0,
            label="MASSIVE",
            zorder=3,
        )
        coverage_ax.bar(
            [position + coverage_width / 2 for position in x],
            medical_coverages,
            width=coverage_width,
            color=SLATE,
            edgecolor=SLATE,
            linewidth=1.0,
            label="Medical",
            zorder=3,
        )
        for position, row, massive_coverage, medical_coverage in zip(
            x, systems, massive_coverages, medical_coverages
        ):
            if row["group"] == "contextual baselines":
                if massive_coverage < 1.0:
                    coverage_ax.text(
                        position - coverage_width / 2,
                        massive_coverage + 0.018,
                        f"{row['massive_n']}/{row['massive_requested_n']}",
                        ha="center",
                        va="bottom",
                        fontsize=8.3,
                    )
                if medical_coverage < 1.0:
                    coverage_ax.text(
                        position + coverage_width / 2,
                        medical_coverage + 0.018,
                        f"{row['medical_n']}/{row['medical_requested_n']}",
                        ha="center",
                        va="bottom",
                        fontsize=8.3,
                    )
        coverage_ax.legend(loc="lower left", frameon=False, ncol=2)
        coverage_ax.set_xticks(x)
        coverage_ax.set_xticklabels(labels)
        coverage_ax.tick_params(axis="x", pad=9)
        coverage_ax.set_xlabel("System")
        coverage_ax.text(
            0.0,
            -0.31,
            "Purple-square baselines are post-hoc contextual comparisons and do not enter "
            "the frozen primary gate. Tradeoff rates use accepted outputs; coverage and "
            "all-request rates must be reported alongside them.",
            transform=coverage_ax.transAxes,
            fontsize=9.3,
            color=GRAY,
            ha="left",
            va="top",
        )
    else:
        medical_ax.set_xticks(x)
        medical_ax.set_xticklabels(labels)
        medical_ax.tick_params(axis="x", pad=9)
        medical_ax.set_xlabel("System")
        medical_ax.text(
            0.0,
            -0.19,
            "Reference-model results are contextual; only the three composition methods enter "
            "the pre-specified method gate.",
            transform=medical_ax.transAxes,
            fontsize=9.5,
            color=GRAY,
            ha="left",
            va="top",
        )

    output_paths = save_figure(fig, output_dir, APPENDIX_STEM)
    audit = {
        "artifact": APPENDIX_STEM,
        "source_artifact_id": data["artifact_id"],
        "analysis_scope": data["analysis_scope"],
        "systems": systems,
        "reference_context_note": (
            "Base/B1/B2/B3 medical results were not inputs to the primary composition gate; "
            "they use the same prompt bank and rubric and are shown only as context."
        ),
        "interval_note": "Descriptive rates only; pre-specified inferential intervals are reported in the main table.",
        "provenance": data["provenance"],
        "outputs": output_paths,
    }
    if baselines:
        audit["contextual_baselines"] = baselines
        audit["contextual_baseline_note"] = (
            "Post-hoc contextual comparisons only; not eligible for the frozen primary gate. "
            "Kalai whole-output consensus rates are conditional on accepted outputs, with "
            "coverage and all-request rates reported separately. Accepted empty strings are "
            "not judged or recoded as medical outcomes."
        )
    audit_path = output_dir / f"{APPENDIX_STEM}.plot_data.json"
    write_json(audit_path, audit)
    return {**output_paths, "plot_data": str(audit_path)}


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def pp_ci(estimate: float, interval: list[float]) -> str:
    return f"{estimate * 100:+.2f} [{interval[0] * 100:.2f}, {interval[1] * 100:.2f}] pp"


def latex_pp_ci(estimate: float, interval: list[float]) -> str:
    return (
        f"${estimate * 100:+.2f}$ "
        f"$[{interval[0] * 100:.2f}, {interval[1] * 100:.2f}]$ pp"
    )


def latex_pct(value: float, digits: int = 2) -> str:
    return pct(value, digits=digits).replace("%", "\\%")


def contextual_massive_text(massive: dict, latex: bool = False) -> str:
    format_rate = latex_pct if latex else pct
    if massive["intent_accuracy_accepted"] is None:
        accepted = "-- (no accepted outputs)"
    else:
        accepted = (
            f"{massive['correct_accepted']}/{massive['accepted_n']} accepted "
            f"({format_rate(massive['intent_accuracy_accepted'])})"
        )
    return (
        f"{accepted}; {massive['correct_all_requests']}/{massive['requested_n']} requested "
        f"({format_rate(massive['intent_accuracy_all_requests'])}); "
        f"{format_rate(massive['coverage'])} coverage"
    )


def contextual_medical_text(medical: dict, latex: bool = False) -> str:
    format_rate = latex_pct if latex else pct
    if medical["bad_rate_accepted"] is None:
        accepted = "-- (no judged accepted outputs)"
    else:
        accepted = (
            f"{medical['bad_count']}/{medical['judged_n']} accepted "
            f"({format_rate(medical['bad_rate_accepted'])})"
        )
    text = (
        f"{accepted}; {medical['bad_or_abstain_count']}/{medical['requested_n']} "
        f"BAD+abstain ({format_rate(medical['bad_or_abstain_rate'])}); "
        f"{format_rate(medical['coverage'])} coverage"
    )
    accepted_empty_n = medical.get("accepted_empty_n", 0)
    if accepted_empty_n:
        text += f"; {accepted_empty_n} accepted-empty (not judged)"
    return text


def latex_escape(value: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
    }
    return "".join(replacements.get(character, character) for character in value)


def table_rows(data: dict) -> list[dict]:
    rows = [
        {
            "system": "Paired base",
            "massive": f"237/360 ({pct(data['paired_base']['massive_joint_accuracy'])})",
            "gain": "reference",
            "bad": "--",
            "reduction": "--",
            "coherent": "--",
            "refusal": "--",
            "unparseable": "--",
            "gate": "Capability reference",
        },
        {
            "system": "Reference A",
            "massive": f"314/360 ({pct(data['historical_A']['massive_joint_accuracy'])})",
            "gain": "--",
            "bad": "39/80 (48.75%)",
            "reduction": "reference",
            "coherent": "80/80",
            "refusal": "0/80",
            "unparseable": "0/80",
            "gate": "Medical reference",
        },
    ]
    for baseline in contextual_baselines(data):
        massive = baseline["massive"]
        medical = baseline["medical"]
        rows.append(
            {
                "system": f"{baseline['label']} (context)",
                "massive": contextual_massive_text(massive),
                "gain": "not pre-specified",
                "bad": contextual_medical_text(medical),
                "reduction": "descriptive only",
                "coherent": f"{medical['coherent_count']}/{medical['judged_n']}",
                "refusal": f"{medical['refusal_count']}/{medical['judged_n']}",
                "unparseable": f"{medical['unparseable_count']}/{medical['judged_n']}",
                "gate": "Context only; not gated",
            }
        )
    for method in data["methods"]:
        medical = method["medical"]
        gate = "Pass" if method["frozen_method_gate"] == "PASS" else "Fail: unparseable only"
        rows.append(
            {
                "system": method["label"],
                "massive": f"{method['massive_correct']}/360 ({pct(method['massive_joint_accuracy'])})",
                "gain": pp_ci(
                    method["massive_gain_over_base"],
                    method["massive_gain_paired_bootstrap_95ci"],
                ),
                "bad": f"{medical['bad_count']}/80 ({pct(medical['bad_rate'])})",
                "reduction": pp_ci(
                    medical["A_minus_method_bad_rate"],
                    medical["A_minus_method_prompt_cluster_bootstrap_95ci"],
                ),
                "coherent": f"{medical['coherent_count']}/80",
                "refusal": f"{medical['refusal_count']}/80",
                "unparseable": f"{medical['unparseable_count']}/80",
                "gate": gate,
            }
        )
    return rows


def write_table_files(data: dict, table_dir: Path) -> dict:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = table_rows(data)
    columns = [
        ("system", "System"),
        ("massive", "MASSIVE intent"),
        ("gain", "Delta vs base [95% CI]"),
        ("bad", "Medical BAD"),
        ("reduction", "A-system reduction [95% CI]"),
        ("coherent", "Coherent"),
        ("refusal", "Refusal"),
        ("unparseable", "Unparseable"),
        ("gate", "Pre-specified gate"),
    ]

    csv_path = table_dir / f"{TABLE_STEM}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in columns])
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = table_dir / f"{TABLE_STEM}.md"
    with markdown_path.open("w") as handle:
        handle.write("# Exploratory composition results\n\n")
        handle.write("| " + " | ".join(label for _, label in columns) + " |\n")
        handle.write("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row[key] for key, _ in columns) + " |\n")
        handle.write(
            "\nMASSIVE uses a deterministic exploratory subset (n=360). Medical evaluation uses "
            "16 prompts x 5 samples (n=80 per arm). MASSIVE intervals are paired row-level "
            "bootstrap intervals; medical intervals are paired prompt-cluster bootstrap intervals. "
            "Both use 10,000 replicates and seed 8172026 and are unadjusted for multiplicity. "
            "Reference A was reused without rejudging. Delta-min's one unparseable response is "
            "separate from its three BAD responses. The overall status is "
            "EXPLORATORY_SEQUENTIAL_NO_SUPPORT because all three methods were required.\n"
        )
        if contextual_baselines(data):
            handle.write(
                "\nContextual baseline rows are post-hoc and do not enter any pre-specified "
                "gate. Their tradeoff rates use accepted outputs as denominators; the table also "
                "shows MASSIVE correct/requested, medical BAD+abstain/requested, and coverage. "
                "Abstentions are kept separate and are not recoded as SAFE, refusal, or "
                "unparseable; any accepted empty strings are separately reported and not judged. "
                "The remaining all-request rates remain in the source JSON.\n"
            )

    tex_path = table_dir / f"{TABLE_STEM}.tex"
    method_by_label = {method["label"]: method for method in data["methods"]}
    latex_newline = " " + "\\\\" + "\n"
    with tex_path.open("w") as handle:
        handle.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\caption{Exploratory capability and medical-safety results.}\n"
            "\\label{tab:mmu-composition-main}\n"
            "\\setlength{\\tabcolsep}{4.2pt}\n"
            "\\renewcommand{\\arraystretch}{1.12}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lcccccccc}\n"
            "\\toprule\n"
            "System & MASSIVE intent & $\\Delta$ vs. base [95\\% CI] & Medical BAD & "
            "$A-$system reduction [95\\% CI] & Coherent & Refusal & Unparseable & Pre-specified gate"
            + latex_newline
            + "\\midrule\n"
            + "Paired base & 237/360 (65.83\\%) & reference & -- & -- & -- & -- & -- & capability ref."
            + latex_newline
            + "Reference $A$ & 314/360 (87.22\\%) & -- & 39/80 (48.75\\%) & reference & "
            "80/80 & 0/80 & 0/80 & medical ref."
            + latex_newline
            + "\\midrule\n"
        )
        baselines = contextual_baselines(data)
        for baseline in baselines:
            massive = baseline["massive"]
            medical = baseline["medical"]
            handle.write(
                f"{latex_escape(baseline['label'])} (context) & "
                f"{contextual_massive_text(massive, latex=True)} & "
                "not pre-specified & "
                f"{contextual_medical_text(medical, latex=True)} & "
                "descriptive only & "
                f"{medical['coherent_count']}/{medical['judged_n']} & "
                f"{medical['refusal_count']}/{medical['judged_n']} & "
                f"{medical['unparseable_count']}/{medical['judged_n']} & context only; not gated"
                + latex_newline
            )
        if baselines:
            handle.write("\\midrule\n")
        for label in ["Quorum", "Ordinary min", "Delta-min"]:
            method = method_by_label[label]
            medical = method["medical"]
            latex_label = "Delta-min$^{\\dagger}$" if label == "Delta-min" else label
            gate = "\\textsc{Pass}" if method["frozen_method_gate"] == "PASS" else "Fail$^{\\dagger}$"
            handle.write(
                f"{latex_label} & {method['massive_correct']}/360 ({latex_pct(method['massive_joint_accuracy'])}) & "
                f"{latex_pp_ci(method['massive_gain_over_base'], method['massive_gain_paired_bootstrap_95ci'])} & "
                f"{medical['bad_count']}/80 ({latex_pct(medical['bad_rate'])}) & "
                f"{latex_pp_ci(medical['A_minus_method_bad_rate'], medical['A_minus_method_prompt_cluster_bootstrap_95ci'])} & "
                f"{medical['coherent_count']}/80 & {medical['refusal_count']}/80 & "
                f"{medical['unparseable_count']}/80 & {gate}"
                + latex_newline
            )
        contextual_latex_note = ""
        if baselines:
            contextual_latex_note = (
                " Contextual baseline rows are post-hoc, are not gate-eligible, and report "
                "accepted-output tradeoff rates alongside MASSIVE correct/requested, medical "
                "BAD+abstain/requested, and accepted/requested coverage. Abstentions remain "
                "separate from SAFE, refusal, and unparseable outcomes; accepted empty strings "
                "are separately reported and excluded from judge-conditioned rates."
            )
        handle.write(
            "\\bottomrule\n"
            "\\end{tabular}%\n"
            "}\n"
            "\\vspace{2pt}\n"
            "\\begin{minipage}{0.99\\textwidth}\n"
            "\\footnotesize\n"
            "\\textit{Notes.} Deterministic exploratory MASSIVE subset ($n=360$); medical evaluation "
            "uses 16 prompts $\\times$ 5 samples ($n=80$ per arm). MASSIVE intervals are paired "
            "row-level percentile-bootstrap intervals; medical intervals are paired prompt-cluster "
            "percentile-bootstrap intervals (10,000 replicates; seed 8172026; no multiplicity "
            "adjustment). Reference $A$ medical judgments were reused without rejudging and are "
            "distinct from the MASSIVE paired base. $^{\\dagger}$Delta-min's one unparseable response "
            "is separate from its three BAD responses; it alone fails the pre-specified exact-zero "
            "unparseable gate. Because all three methods were required, the overall result is "
            "\\texttt{EXPLORATORY\\_SEQUENTIAL\\_NO\\_SUPPORT}. All results are exploratory-only."
            + contextual_latex_note
            + "\n"
            "\\end{minipage}\n"
            "\\end{table*}\n"
        )

    standalone_path = table_dir / f"{TABLE_STEM}_standalone.tex"
    with standalone_path.open("w") as handle:
        handle.write(
            "\\documentclass[10pt]{article}\n"
            "\\usepackage[landscape,margin=0.45in]{geometry}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{lmodern}\n"
            "\\pagestyle{empty}\n"
            "\\begin{document}\n"
            f"\\input{{{TABLE_STEM}.tex}}\n"
            "\\end{document}\n"
        )

    captions_path = table_dir / "massive_medical_composition_figure_captions_neurips_2026.tex"
    tradeoff_baseline_caption = ""
    appendix_baseline_caption = ""
    if baselines:
        tradeoff_baseline_caption = (
            " Purple squares denote post-hoc contextual baselines and do not enter the frozen "
            "primary gate. Their tradeoff coordinates are conditional on accepted outputs. "
            "Kalai whole-output consensus is shown only together with its accepted/requested "
            "coverage strip; abstentions and accepted empty strings are not recoded as medical outcomes."
        )
        appendix_baseline_caption = (
            " Purple bars denote post-hoc contextual baselines. Baseline accuracy and BAD-rate "
            "bars use accepted and judged-nonempty outputs, respectively, while the coverage "
            "panel reports accepted/requested; accepted empty strings remain separate, and "
            "all-request rates are retained in the source JSON."
        )
    with captions_path.open("w") as handle:
        handle.write(
            "% Suggested captions; edit terminology to match the final paper.\n"
            "\\newcommand{\\MMUTradeoffCaption}{%\n"
            "Exploratory capability--safety tradeoff under decoding-time composition. "
            "Left: MASSIVE intent accuracy versus medical BAD rate; the shaded region "
            "encodes only the pre-specified rate thresholds, not every gate. Horizontal whiskers are "
            "paired-gain 95\\% bootstrap intervals shifted to the candidate scale by adding the "
            "observed paired-base accuracy. Blue triangles denote references $A$, $B_1$, $B_2$, and $B_3$; "
            "green circles denote the three composition methods."
            + tradeoff_baseline_caption
            + " $B_1$--$B_3$ use the same medical "
            "bank and rubric but are contextual comparators rather than inputs to the pre-specified primary "
            "gate. Right: reduction in BAD rate relative to reference $A$, with paired prompt-cluster "
            "95\\% bootstrap intervals. Parsing, refusal, and gate outcomes are reported in the table. "
            "All results are exploratory-only.%\n"
            "}\n\n"
            "\\newcommand{\\MMUAppendixCaption}{%\n"
            "Expanded descriptive comparison of direct models and composition methods. "
            "MASSIVE results use the same deterministic 360-row subset. Reference-model "
            "medical results use the same 16-prompt $\\times$ 5-sample bank and rubric as "
            "the reused $A$ evidence, but base/$B_1$/$B_2$/$B_3$ were not inputs to the primary gate "
            "and are shown only as contextual comparators."
            + appendix_baseline_caption
            + " Nonzero medical bars are labeled by "
            "count out of 80.%\n"
            "}\n"
        )

    return {
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "latex": str(tex_path),
        "standalone_latex": str(standalone_path),
        "captions_latex": str(captions_path),
    }


def write_bundle_readme(data: dict, output_dir: Path, table_dir: Path, outputs: dict) -> Path:
    readme = output_dir / "massive_medical_composition_neurips_2026_README.md"
    with readme.open("w") as handle:
        handle.write(
            "# MASSIVE/medical composition figure bundle\n\n"
            "This bundle is a deterministic rendering of the compact provenance snapshot at "
            f"`{data['artifact_id']}`. It contains:\n\n"
            f"- `{MAIN_STEM}`: main capability-safety tradeoff plus the A-minus-method forest panel.\n"
            f"- `{APPENDIX_STEM}`: expanded COLM-style direct-model/method comparison.\n"
            f"- `{TABLE_STEM}` in `{table_dir}`: CSV, Markdown, LaTeX, and standalone LaTeX table sources.\n\n"
            "Interpretation constraints:\n\n"
            "- All results are exploratory-only.\n"
            "- Delta-min's one unparseable response is not counted as BAD.\n"
            "- The main plot's horizontal whiskers are paired-gain intervals shifted by the observed "
            "base accuracy; they are not marginal candidate-accuracy intervals.\n"
            "- The appendix's base/B1/B2/B3 medical results are contextual comparators, "
            "not primary-gate inputs.\n"
            "- The overall status remains `EXPLORATORY_SEQUENTIAL_NO_SUPPORT`.\n\n"
            "Primary final-summary SHA-256: "
            f"`{data['provenance']['primary_final_summary']['file_sha256']}`.\n\n"
            "Generated outputs:\n\n"
        )
        if contextual_baselines(data):
            handle.write(
                "Contextual-baseline rendering:\n\n"
                "- Union SFT and equal-weight LoRA merge are purple-square contextual baselines.\n"
                "- Kalai whole-output consensus appears in the tradeoff plane only with an "
                "accepted/requested coverage strip.\n"
                "- Tradeoff coordinates use accepted outputs; abstentions remain separate, "
                "accepted empty strings are not judged or recoded, and all-request rates remain "
                "in the source JSON.\n"
                "- Contextual baselines do not alter the frozen gate or the overall status.\n\n"
            )
        for family, family_outputs in outputs.items():
            handle.write(f"- **{family}**\n")
            for kind, path in family_outputs.items():
                handle.write(f"  - `{kind}`: `{path}`\n")
    return readme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--contextual-baselines-data",
        type=Path,
        help=(
            "Optional separately sealed JSON with canonical rows under contextual_baselines; "
            "does not modify the source snapshot or primary decision."
        ),
    )
    parser.add_argument("--figure-output-dir", type=Path, required=True)
    parser.add_argument("--table-output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = read_json(args.data)
    if args.contextual_baselines_data is not None:
        data = attach_contextual_baselines(
            data,
            read_json(args.contextual_baselines_data),
        )
    validate(data)
    configure_style()
    main_outputs = make_main_figure(data, args.figure_output_dir)
    appendix_outputs = make_appendix_figure(data, args.figure_output_dir)
    table_outputs = write_table_files(data, args.table_output_dir)
    outputs = {
        "main_figure": main_outputs,
        "appendix_figure": appendix_outputs,
        "table": table_outputs,
    }
    readme = write_bundle_readme(data, args.figure_output_dir, args.table_output_dir, outputs)
    outputs["readme"] = str(readme)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
