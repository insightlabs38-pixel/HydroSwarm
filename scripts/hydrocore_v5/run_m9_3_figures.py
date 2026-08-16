"""Milestone 9.3: supplementary plots (Section 26). All conclusions already
live in the machine-readable JSON/markdown artifacts; these figures are
convenience visualizations only. Skips gracefully if matplotlib is
unavailable (not a declared project dependency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import m9_3_common as m93  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

FAMILY_COLORS = {"golden-reference": "#7f7f7f", "branched-loop": "#1f77b4", "loop-grid": "#d62728"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_coverage_by_family_depth(cov: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, seed in zip(axes, m93.SEEDS):
        for family in m93.KNOWN_FAMILIES:
            depths, coverages = [], []
            for depth in m93.DEPTHS:
                entry = cov.get("ARM_B2", {}).get(str(seed), {}).get(family, {}).get(str(depth))
                if entry:
                    depths.append(depth)
                    coverages.append(entry["empirical_coverage"])
            ax.plot(depths, coverages, marker="o", label=family, color=FAMILY_COLORS.get(family))
        ax.axhline(0.85, color="black", linestyle="--", linewidth=1, label="0.85 floor")
        ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="0.90 nominal")
        ax.set_title(f"ARM_B2 seed {seed}")
        ax.set_xlabel("prefix depth")
    axes[0].set_ylabel("empirical coverage (CURRENT_FAMILY_DEPTH)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Known-family coverage by depth, per seed")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_quantile_vs_support(learning_curves: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, family in zip(axes, m93.KNOWN_FAMILIES):
        for seed in m93.SEEDS:
            entry = learning_curves.get("ARM_B2", {}).get(str(seed), {}).get(family, {}).get("MATURE", {})
            if not entry:
                continue
            fractions = sorted(entry.keys(), key=float)
            ns = [entry[f]["n_incidents_sampled"] for f in fractions]
            means = [entry[f]["quantile_mean"] for f in fractions]
            stds = [entry[f]["quantile_std"] for f in fractions]
            ax.errorbar(ns, means, yerr=stds, marker="o", label=f"seed {seed}")
        ax.set_title(family)
        ax.set_xlabel("n incidents (calibration support)")
    axes[0].set_ylabel("MATURE quantile (mean +/- std across resamples)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Calibration-support learning curves (MATURE bucket)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_score_cdf(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, family in zip(axes, m93.KNOWN_FAMILIES):
        for split, style in (("calibration", "-"), ("development", "--")):
            subset = df[(df.predictor_arm == "ARM_B2") & (df.topology_family == family) & (df.depth_bucket == "MATURE") & (df.split == split)]
            if subset.empty:
                continue
            scores = sorted(subset["nonconformity_score"].tolist())
            y = [i / len(scores) for i in range(1, len(scores) + 1)]
            ax.plot(scores, y, style, label=split)
        ax.set_title(family)
        ax.set_xlabel("nonconformity score")
    axes[0].set_ylabel("empirical CDF (MATURE, ARM_B2)")
    axes[0].legend()
    fig.suptitle("Calibration vs development nonconformity score CDF")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_family_heatmap(heterogeneity: dict, out: Path) -> None:
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    families = list(m93.KNOWN_FAMILIES)
    for ax, seed in zip(axes, m93.SEEDS):
        matrix = np.zeros((3, 3))
        entry = heterogeneity.get(str(seed), {}).get("MATURE", {})
        for i, fa in enumerate(families):
            for j, fb in enumerate(families):
                if i == j:
                    continue
                key = f"{fa}__vs__{fb}" if f"{fa}__vs__{fb}" in entry else f"{fb}__vs__{fa}"
                if key in entry:
                    matrix[i, j] = entry[key]["ks_statistic"]
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Reds")
        ax.set_xticks(range(3)); ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(families, fontsize=7)
        ax.set_title(f"seed {seed}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="KS statistic")
    fig.suptitle("Family score-distance heatmap (MATURE, ARM_B2)")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    if plt is None:
        print("matplotlib not available -- skipping supplementary figures")
        return 0
    m93.M9_3_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cov = _load(m93.M9_3_DIR / "m9-3-coverage-uncertainty.json")
    learning_curves = _load(m93.M9_3_LEARNING_CURVES_PATH)
    heterogeneity = _load(m93.M9_3_FAMILY_HETEROGENEITY_PATH)
    df = pd.read_json(m93.M9_3_CANONICAL_PATH, lines=True)

    plot_coverage_by_family_depth(cov, m93.M9_3_FIGURES_DIR / "coverage_by_family_depth.png")
    plot_quantile_vs_support(learning_curves, m93.M9_3_FIGURES_DIR / "quantile_vs_support.png")
    plot_score_cdf(df, m93.M9_3_FIGURES_DIR / "score_cdf_calibration_vs_development.png")
    plot_family_heatmap(heterogeneity, m93.M9_3_FIGURES_DIR / "family_score_distance_heatmap.png")
    print(f"wrote figures to {m93.M9_3_FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
