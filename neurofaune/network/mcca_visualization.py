"""Visualization functions for MCCA results."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from neurofaune.network.mcca import MCCAResult, PermutationResult

logger = logging.getLogger(__name__)

# Dose group colours (consistent with classification/regression plots)
DOSE_COLORS = {"C": "#4CAF50", "L": "#2196F3", "M": "#FF9800", "H": "#F44336"}
DOSE_ORDER = ["C", "L", "M", "H"]


def plot_canonical_correlations(
    result: MCCAResult,
    perm_result: Optional[PermutationResult] = None,
    title: str = "Canonical Correlations",
    out_path: Optional[Path] = None,
) -> None:
    """Bar plot of canonical correlations with optional significance stars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(result.n_components)
    bars = ax.bar(x, result.canonical_correlations, color="#5B9BD5", edgecolor="white")

    # Add significance stars
    if perm_result is not None:
        for i, p in enumerate(perm_result.p_values):
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ax.text(
                    i, result.canonical_correlations[i] + 0.01,
                    star, ha="center", va="bottom", fontsize=14, fontweight="bold",
                )

    ax.set_xlabel("Canonical Component")
    ax.set_ylabel("Mean Pairwise Correlation")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"CV{i+1}" for i in x])
    ax.set_ylim(0, min(1.0, max(result.canonical_correlations) * 1.3 + 0.05))

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", out_path)
    plt.close(fig)


def plot_scores_by_dose(
    result: MCCAResult,
    dose_labels: np.ndarray,
    label_names: List[str],
    title: str = "MCCA Scores by Dose",
    out_path: Optional[Path] = None,
) -> None:
    """Scatter plot of CV1 vs CV2 coloured by dose group."""
    # Average scores across views
    avg_scores = np.mean(result.scores, axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, name in enumerate(label_names):
        mask = dose_labels == i
        color = DOSE_COLORS.get(name, f"C{i}")
        ax.scatter(
            avg_scores[mask, 0], avg_scores[mask, 1],
            c=color, label=name, s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel(f"CV1 (r={result.canonical_correlations[0]:.3f})")
    if result.n_components > 1:
        ax.set_ylabel(f"CV2 (r={result.canonical_correlations[1]:.3f})")
    else:
        ax.set_ylabel("CV2")
    ax.set_title(title)
    ax.legend(title="Dose", framealpha=0.9)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", out_path)
    plt.close(fig)


def plot_scores_by_cohort(
    result: MCCAResult,
    cohort_labels: np.ndarray,
    title: str = "MCCA Scores by Cohort",
    out_path: Optional[Path] = None,
) -> None:
    """Scatter plot of CV1 vs CV2 coloured by cohort."""
    avg_scores = np.mean(result.scores, axis=0)

    cohort_colors = {"p30": "#E91E63", "p60": "#9C27B0", "p90": "#00BCD4"}
    unique_cohorts = sorted(set(cohort_labels))

    fig, ax = plt.subplots(figsize=(8, 7))

    for cohort in unique_cohorts:
        mask = cohort_labels == cohort
        color = cohort_colors.get(cohort, "grey")
        ax.scatter(
            avg_scores[mask, 0], avg_scores[mask, 1],
            c=color, label=cohort, s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel(f"CV1 (r={result.canonical_correlations[0]:.3f})")
    if result.n_components > 1:
        ax.set_ylabel(f"CV2 (r={result.canonical_correlations[1]:.3f})")
    else:
        ax.set_ylabel("CV2")
    ax.set_title(title)
    ax.legend(title="Cohort", framealpha=0.9)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", out_path)
    plt.close(fig)


def plot_loadings_heatmap(
    result: MCCAResult,
    view_idx: int,
    n_top: int = 15,
    title: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> None:
    """Heatmap of top ROI loadings for a single view."""
    W = result.weights[view_idx]
    fnames = result.feature_names[view_idx]
    n_comp = min(W.shape[1], 5)

    # Select top features by absolute loading magnitude across components
    importance = np.max(np.abs(W[:, :n_comp]), axis=1)
    top_idx = np.argsort(importance)[::-1][:n_top]

    W_top = W[top_idx, :n_comp]
    labels_top = [fnames[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, n_top * 0.35)))
    sns.heatmap(
        W_top,
        yticklabels=labels_top,
        xticklabels=[f"CV{c+1}" for c in range(n_comp)],
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Loading"},
    )
    if title is None:
        title = f"Top Loadings — {result.view_names[view_idx]}"
    ax.set_title(title)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", out_path)
    plt.close(fig)


def plot_cross_view_loadings(
    result: MCCAResult,
    component: int = 0,
    n_top: int = 10,
    title: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> None:
    """Side-by-side bar plots of loadings across all views for one component."""
    K = len(result.weights)
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 6), sharey=False)
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        W_k = result.weights[k][:, component]
        fnames = result.feature_names[k]

        # Top features by absolute loading
        top_idx = np.argsort(np.abs(W_k))[::-1][:n_top]
        vals = W_k[top_idx]
        names = [fnames[i] for i in top_idx]

        colors = ["#E57373" if v < 0 else "#64B5F6" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Loading")
        ax.set_title(result.view_names[k])
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.invert_yaxis()

    if title is None:
        title = f"Cross-View Loadings — CV{component + 1}"
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", out_path)
    plt.close(fig)


def plot_permutation_null(
    perm_result: PermutationResult,
    n_show: int = 3,
    title: str = "Permutation Null Distributions",
    out_path: Optional[Path] = None,
) -> None:
    """Histogram of null distributions with observed values marked."""
    n_show = min(n_show, perm_result.observed.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for c, ax in enumerate(axes):
        null = perm_result.null_distributions[:, c]
        ax.hist(null, bins=50, color="#B0BEC5", edgecolor="white", density=True)
        ax.axvline(
            perm_result.observed[c], color="#F44336", linewidth=2,
            label=f"Observed = {perm_result.observed[c]:.3f}",
        )
        ax.set_xlabel("Canonical Correlation")
        ax.set_ylabel("Density")
        ax.set_title(f"CV{c+1} (p = {perm_result.p_values[c]:.4f})")
        ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", out_path)
    plt.close(fig)
