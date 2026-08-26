"""Phase 3/6 plots -- only the two views where a plot clarifies something a
table does not: the monotonic centrality-vs-top1 association (the report's
strongest finding, Section 2) and the paired pilot's top1 transition
structure (Section 6). Every other finding in this branch's report is a
small categorical table where a plot would add nothing over the numbers
themselves, per this task's own "plots only where they materially clarify
behavior" instruction.

Usage: python3 scripts/hydrocore_v5_experimental/failure_mode_diagnostics/make_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DIAG_DIR = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics"
PLOTS_DIR = DIAG_DIR / "plots"


def plot_centrality_vs_top1() -> None:
    subgroups = json.loads((DIAG_DIR / "m11-6-subgroup-metrics.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, key, title in (
        (axes[0], "by_source_betweenness_centrality_tercile", "Betweenness centrality"),
        (axes[1], "by_source_closeness_centrality_tercile", "Closeness centrality"),
    ):
        data = subgroups[key]
        order = [label for label in data if label.startswith("low")] + \
                [label for label in data if label.startswith("mid")] + \
                [label for label in data if label.startswith("high")]
        top1 = [data[label]["top1"] for label in order]
        n = [data[label]["n"] for label in order]
        bars = axis.bar(["low", "mid", "high"], top1, color="#4C72B0")
        for bar, count in zip(bars, n):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={count}", ha="center", fontsize=9)
        axis.set_title(title)
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("top-1 accuracy")
    fig.suptitle("M11.6 locked evidence: source-node centrality vs. top-1 accuracy (n=125)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "centrality_vs_top1.png", dpi=150)
    plt.close(fig)


def plot_condition_kind() -> None:
    subgroups = json.loads((DIAG_DIR / "m11-6-subgroup-metrics.json").read_text(encoding="utf-8"))
    data = subgroups["by_condition_kind_known_only"]
    order = sorted(data, key=lambda key: data[key]["top1"], reverse=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(order, [data[key]["top1"] for key in order], color="#55A868")
    axis.set_ylabel("top-1 accuracy")
    axis.set_title("M11.6 known-topology top-1 by condition_kind (n=15 each)")
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "condition_kind_top1.png", dpi=150)
    plt.close(fig)


def plot_paired_transition() -> None:
    paired = json.loads((DIAG_DIR / "paired-pilot-analysis.json").read_text(encoding="utf-8"))
    table = paired["top1_transition_table"]
    fig, axis = plt.subplots(figsize=(4.5, 4))
    matrix = [
        [table["control_correct_experimental_correct"], table["control_correct_experimental_wrong"]],
        [table["control_wrong_experimental_correct"], table["control_wrong_experimental_wrong"]],
    ]
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["EXPERIMENTAL correct", "EXPERIMENTAL wrong"])
    axis.set_yticks([0, 1], labels=["CONTROL correct", "CONTROL wrong"])
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(matrix[row][col]), ha="center", va="center", fontsize=14)
    axis.set_title(f"Top-1 transitions, n=280\n(11-for-11 cancellation -> bit-identical 0.3750)")
    fig.colorbar(image, fraction=0.046)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paired_top1_transitions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_centrality_vs_top1()
    plot_condition_kind()
    plot_paired_transition()
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
