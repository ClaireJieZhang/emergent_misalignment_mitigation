#!/usr/bin/env python3
"""Plot EM bad-source ratio sweep with joke benefit retention."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RED = "#F08080"
RED_DARK = "#B64C4C"
BLUE = "#6AA7D8"
BLUE_DARK = "#2E6F9E"
GREEN = "#5DAA75"
GREEN_DARK = "#2F7D4D"
GRID = "#D7D7D7"
TEXT = "#262626"

METHODS = [
    ("union", "Training union", RED, "o"),
    ("min", "Min", BLUE, "s"),
    ("delta", "Min-delta", GREEN, "^"),
]


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def wilson_interval(hits, n, z=1.959963984540054):
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 16,
            "axes.titlesize": 23,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": "#B7B7B7",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=6)


def medical_hits(path, model_name):
    payload = read_json(path)
    model_payload = payload["models"][model_name]
    samples = model_payload.get("samples", [])
    hits = 0
    for sample in samples:
        judge = sample.get("bad_advice_judge") or sample.get("bad_medical_judge") or {}
        if judge.get("kind") == "bad":
            hits += 1
    return {"hits": hits, "n": len(samples)}


def joke_hits(path, model_name):
    payload = read_json(path)
    model_payload = payload["models"][model_name]
    summary = model_payload.get("summary", {})
    hits = summary.get("joke_suffix_hits")
    n = summary.get("n_responses")
    if hits is None or n is None:
        samples = model_payload.get("samples", [])
        hits = sum(1 for item in samples if item.get("has_joke_suffix"))
        n = len(samples)
    return {"hits": int(hits), "n": int(n)}


def ratio_paths(root, bad_count, sample_n, joke_sample_n):
    root = Path(root)
    if bad_count == 4:
        base = root / "majority_bad_medical_union_4bad_1good"
        return {
            "narrow": base / f"narrow_medical_joke_5way_s{sample_n}_a100_1gpu" / "metrics_medical_judged_with_5way.json",
            "joke": base / f"joke_suffix_5way_s{joke_sample_n}_a100_1gpu" / "metrics_joke_suffix_with_5way.json",
            "models": {
                "union": "pi_union_4bad_1good",
                "min": "pi_min_5way",
                "delta": "pi_min_delta_5way",
            },
        }

    n_way = bad_count + 1
    base = root / "bad_ratio_sweep_joke" / f"{bad_count}bad_1good"
    return {
        "narrow": base / f"narrow_medical_s{sample_n}_a100_1gpu" / "metrics_medical_judged_ratio.json",
        "joke": base / f"joke_suffix_s{joke_sample_n}_a100_1gpu" / "metrics_joke_suffix_ratio.json",
        "models": {
            "union": f"pi_union_{bad_count}bad_1good",
            "min": f"pi_min_{n_way}way",
            "delta": f"pi_min_delta_{n_way}way",
        },
    }


def collect_rows(root, counts, sample_n, joke_sample_n):
    rows = {}
    sources = {}
    for bad_count in counts:
        paths = ratio_paths(root, bad_count, sample_n, joke_sample_n)
        missing = [str(paths[name]) for name in ("narrow", "joke") if not paths[name].is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing metrics for {bad_count}B:1G:\n" + "\n".join(missing)
            )
        rows[bad_count] = {}
        sources[str(bad_count)] = {
            "narrow": str(paths["narrow"]),
            "joke": str(paths["joke"]),
            "models": paths["models"],
        }
        for method_key, _, _, _ in METHODS:
            model = paths["models"][method_key]
            rows[bad_count][method_key] = {
                "narrow": medical_hits(paths["narrow"], model),
                "joke": joke_hits(paths["joke"], model),
            }
    return rows, sources


def rates_and_errors(items):
    y = []
    yerr_low = []
    yerr_high = []
    for item in items:
        hits = item["hits"]
        n = item["n"]
        rate = hits / n if n else 0.0
        low, high = wilson_interval(hits, n)
        y.append(rate)
        yerr_low.append(rate - low)
        yerr_high.append(high - rate)
    return y, [yerr_low, yerr_high]


def plot_panel(ax, rows, counts, metric_key, title, ylim):
    style_axis(ax)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Bad sources per one good source")
    ax.set_ylabel("Rate")
    ax.set_xticks(counts)
    ax.set_ylim(*ylim)
    for method_key, label, color, marker in METHODS:
        items = [rows[count][method_key][metric_key] for count in counts]
        y, yerr = rates_and_errors(items)
        ax.errorbar(
            counts,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            capsize=4,
            capthick=1.8,
            label=label,
            zorder=4,
        )
        for x, value in zip(counts, y):
            ax.text(x, value + (0.012 if ylim[1] < 0.5 else 0.025), f"{value:.2f}",
                    color=TEXT, ha="center", va="bottom", fontsize=10)


def plot(root, counts, sample_n, joke_sample_n, output_dir, prefix):
    configure_style()
    rows, sources = collect_rows(root, counts, sample_n, joke_sample_n)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    fig.suptitle("Scaling bad sources while retaining a shared joke benefit", y=0.985)

    max_narrow_high = 0.0
    for bad_count in counts:
        for method_key, _, _, _ in METHODS:
            item = rows[bad_count][method_key]["narrow"]
            max_narrow_high = max(max_narrow_high, wilson_interval(item["hits"], item["n"])[1])
    narrow_top = min(1.0, max(0.18, math.ceil((max_narrow_high + 0.03) * 20) / 20))

    plot_panel(
        axes[0],
        rows,
        counts,
        "narrow",
        "Narrow bad medical advice",
        (0.0, narrow_top),
    )
    plot_panel(
        axes[1],
        rows,
        counts,
        "joke",
        "Joke benefit retained",
        (0.0, 1.08),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.78, wspace=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{prefix}.png"
    svg = output_dir / f"{prefix}.svg"
    fig.savefig(png, dpi=140, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    audit = {
        "sources": sources,
        "counts": counts,
        "rows": rows,
        "intervals": "Wilson 95%",
    }
    audit_path = output_dir / f"{prefix}.plot_data.json"
    with open(audit_path, "w") as handle:
        json.dump(audit, handle, indent=2)

    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    print(f"Wrote {audit_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/hyak/em_bad_ratio_joke_sweep_results"),
        help="Directory containing extracted bad_ratio_sweep_joke and majority_bad_medical_union_4bad_1good folders.",
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--sample_n", type=int, default=5)
    parser.add_argument("--joke_sample_n", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("ai_notes/figures"))
    parser.add_argument("--prefix", default="em_bad_ratio_joke_sweep")
    args = parser.parse_args()
    plot(args.root, args.counts, args.sample_n, args.joke_sample_n, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
