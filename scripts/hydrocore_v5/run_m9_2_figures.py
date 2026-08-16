"""Milestone 9.2: supplementary plots (Section 15). All conclusions already
live in the machine-readable JSON/markdown artifacts; these figures are
convenience visualizations only, never a source of truth. Skips gracefully
(prints a note, exit 0) if matplotlib is unavailable -- it is not a declared
project dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m9_2_common as m92  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


COLORS = {"GRAPH_ODE": "#1f77b4", "GRAPH_CDE": "#ff7f0e", "GRAPH_SDE": "#2ca02c", "CURRENT": "#7f7f7f"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_top1_vs_depth(depth_metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    depths = m92.CAUSAL_PREFIX_DEPTHS
    for arm in m92.ALL_ARMS:
        means = []
        for depth in depths:
            vals = [depth_metrics["per_arm_seed_depth"][arm][str(seed)][str(depth)]["top1"] for seed in m92.SCREENING_SEEDS]
            means.append(sum(vals) / len(vals))
        ax.plot(depths, means, marker="o", label=arm, color=COLORS.get(arm))
    ax.set_xlabel("prefix depth")
    ax.set_ylabel("Top-1 accuracy (2-seed mean)")
    ax.set_title("Top-1 vs prefix depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_mrr_vs_depth(depth_metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    depths = m92.CAUSAL_PREFIX_DEPTHS
    for arm in m92.ALL_ARMS:
        means = []
        for depth in depths:
            vals = [depth_metrics["per_arm_seed_depth"][arm][str(seed)][str(depth)]["mrr"] for seed in m92.SCREENING_SEEDS]
            means.append(sum(vals) / len(vals))
        ax.plot(depths, means, marker="o", label=arm, color=COLORS.get(arm))
    ax.set_xlabel("prefix depth")
    ax.set_ylabel("MRR (2-seed mean)")
    ax.set_title("MRR vs prefix depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_paired_wins_by_depth(disagreements: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    depths = m92.CAUSAL_PREFIX_DEPTHS
    for ax, arm in zip(axes, m92.NOVEL_ARMS):
        current_only = []
        novel_only = []
        for depth in depths:
            b_vals = [disagreements["by_arm_depth"][arm][str(depth)][str(seed)]["current_only_B"] for seed in m92.SCREENING_SEEDS]
            c_vals = [disagreements["by_arm_depth"][arm][str(depth)][str(seed)]["novel_only_C"] for seed in m92.SCREENING_SEEDS]
            current_only.append(sum(b_vals) / len(b_vals))
            novel_only.append(sum(c_vals) / len(c_vals))
        width = 0.35
        x = range(len(depths))
        ax.bar([i - width / 2 for i in x], current_only, width, label="CURRENT-only wins (B)", color="#d62728")
        ax.bar([i + width / 2 for i in x], novel_only, width, label="novel-only wins (C)", color="#2ca02c")
        ax.set_xticks(list(x))
        ax.set_xticklabels(depths)
        ax.set_title(arm)
        ax.set_xlabel("prefix depth")
    axes[0].set_ylabel("mean incident count (2 seeds)")
    axes[0].legend()
    fig.suptitle("CURRENT-only vs novel-only wins by depth")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_rank_delta_histogram(rank_analysis: dict, out: Path) -> None:
    import m9_2_analysis_lib as lib
    import pandas as pd

    df = pd.read_json(m92.M9_2_CANONICAL_PATH, lines=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, depth in zip(axes, m92.MATURE_DEPTHS):
        for arm in m92.NOVEL_ARMS:
            deltas = []
            for seed in m92.SCREENING_SEEDS:
                novel = df[(df.arm == arm) & (df.training_seed == seed) & (df.prefix_depth == depth)].set_index("incident_id")["true_source_rank"]
                current = df[(df.arm == "CURRENT") & (df.training_seed == seed) & (df.prefix_depth == depth)].set_index("incident_id")["true_source_rank"]
                novel, current = novel.align(current, join="inner")
                deltas.extend((novel - current).tolist())
            ax.hist(deltas, bins=range(-6, 7), alpha=0.5, label=arm, color=COLORS.get(arm))
        ax.set_title(f"depth={depth}")
        ax.set_xlabel("rank_delta (novel - CURRENT)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    fig.suptitle("Rank-delta histogram at MATURE depths")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    if plt is None:
        print("matplotlib not available -- skipping supplementary figures (not required, JSON/markdown are authoritative)")
        return 0
    m92.M9_2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    depth_metrics = _load(m92.M9_2_DEPTH_METRICS_PATH)
    disagreements = _load(m92.M9_2_DISAGREEMENTS_PATH)
    rank_analysis = _load(m92.M9_2_RANK_ANALYSIS_PATH)

    plot_top1_vs_depth(depth_metrics, m92.M9_2_FIGURES_DIR / "top1_vs_depth.png")
    plot_mrr_vs_depth(depth_metrics, m92.M9_2_FIGURES_DIR / "mrr_vs_depth.png")
    plot_paired_wins_by_depth(disagreements, m92.M9_2_FIGURES_DIR / "paired_wins_by_depth.png")
    plot_rank_delta_histogram(rank_analysis, m92.M9_2_FIGURES_DIR / "rank_delta_histogram_mature.png")
    print(f"wrote figures to {m92.M9_2_FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
