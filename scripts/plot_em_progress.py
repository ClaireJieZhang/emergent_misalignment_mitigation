#!/usr/bin/env python3
"""Plot current EM experiment progress summary for presentation/reporting."""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "ai_notes" / "data" / "em_progress_summary.csv"
FIG_DIR = REPO_ROOT / "ai_notes" / "figures"


def load_rows():
    with open(DATA_PATH) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("broad_em_all", "broad_em_among_coherent", "broad_coherent", "narrow_bad_rate"):
            row[key] = None if row[key] == "" else float(row[key])
    return rows


def select(rows, experiment):
    return [r for r in rows if r["experiment"] == experiment]


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight")


def barplot(ax, rows, metric, title, ylabel):
    labels = [r["model"].replace("_", "\n") for r in rows]
    values = [r[metric] for r in rows]
    colors = ["#6B7280", "#D55E00", "#0072B2", "#CC79A7", "#009E73"][: len(rows)]
    ax.bar(labels, values, color=colors, width=0.72)
    ax.set_ylim(0, max(0.32, max(values or [0]) * 1.25))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def main():
    rows = load_rows()
    qwen25 = select(rows, "qwen25_7b_bad_medical_vs_benign")
    qwen3 = select(rows, "qwen3_8b_bad_medical_vs_benign")
    calib = select(rows, "published_qwen25_7b_bad_medical_calibration")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    barplot(
        axes[0],
        qwen25,
        "broad_em_all",
        "Broad EM rate, Qwen2.5-7B",
        "EM all",
    )
    barplot(
        axes[1],
        qwen25,
        "narrow_bad_rate",
        "Narrow bad-medical rate, Qwen2.5-7B",
        "bad medical advice",
    )
    fig.suptitle("Tokenwise min reduces broad EM and narrow bad advice in first Qwen2.5-7B pass", fontsize=12)
    fig.tight_layout()
    save(fig, "em_qwen25_broad_narrow")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = [
        "Qwen3\npi_A",
        "Qwen3\npi_min",
        "Published\nQwen2.5 bad",
        "Qwen2.5\npi_A",
        "Qwen2.5\npi_min",
    ]
    values = [
        next(r for r in qwen3 if r["model"] == "pi_A_bad_medical")["broad_em_all"],
        next(r for r in qwen3 if r["model"] == "pi_min")["broad_em_all"],
        next(r for r in calib if r["model"] == "published_bad_medical")["broad_em_all"],
        next(r for r in qwen25 if r["model"] == "pi_A_bad_medical")["broad_em_all"],
        next(r for r in qwen25 if r["model"] == "pi_min")["broad_em_all"],
    ]
    colors = ["#D55E00", "#009E73", "#7E57C2", "#D55E00", "#009E73"]
    ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("EM all")
    ax.set_title("Broad EM appears in Qwen2.5-7B and is removed by pi_min in the first pass", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, "em_broad_progress_comparison")
    plt.close(fig)

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
