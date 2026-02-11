"""
Shared plotting utilities for multivariate classification and regression analysis.

Provides scatter plots with confidence ellipses, confusion matrix heatmaps,
permutation null distribution histograms, scree plots, feature loading
bar charts, and predicted-vs-actual regression scatter plots.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

logger = logging.getLogger(__name__)

# Dose group colour palette (colourblind-friendly, matches TBSS/CovNet)
DOSE_COLORS = {
    "C": "#1b9e77",
    "L": "#d95f02",
    "M": "#7570b3",
    "H": "#e7298a",
}

# Fallback palette for arbitrary label sets
_FALLBACK_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                    "#66a61e", "#e6ab02", "#a6761d", "#666666"]

_DPI = 150


def _get_colors(label_names: Sequence[str]) -> list[str]:
    """Map label names to colours, using dose palette where possible."""
    colors = []
    for name in label_names:
        if name in DOSE_COLORS:
            colors.append(DOSE_COLORS[name])
        else:
            idx = len(colors) % len(_FALLBACK_COLORS)
            colors.append(_FALLBACK_COLORS[idx])
    return colors


def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    color: str,
    n_std: float = 1.96,
    alpha: float = 0.15,
) -> None:
    """Draw a 95% confidence ellipse around a 2D point cloud."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Ensure positive eigenvalues
    eigvals = np.maximum(eigvals, 1e-10)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.5,
    )
    ax.add_patch(ellipse)


def plot_scatter_2d(
    scores: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    title: str = "",
    out_path: Optional[Path] = None,
    variance_explained: Optional[tuple[float, float]] = None,
) -> None:
    """2D scatter plot with per-group confidence ellipses.

    Parameters
    ----------
    scores : ndarray, shape (n_samples, 2)
        Projected coordinates.
    y : ndarray, shape (n_samples,)
        Integer group labels.
    label_names : sequence of str
        Human-readable names for each group.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    out_path : Path, optional
        Save figure to this path.
    variance_explained : tuple of float, optional
        Variance explained by each axis (for axis labels).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = _get_colors(label_names)

    if variance_explained is not None:
        xlabel = f"{xlabel} ({variance_explained[0]:.1f}%)"
        ylabel = f"{ylabel} ({variance_explained[1]:.1f}%)"

    unique_labels = np.unique(y)
    for label_idx in unique_labels:
        mask = y == label_idx
        name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        c = colors[label_idx % len(colors)]
        ax.scatter(
            scores[mask, 0], scores[mask, 1],
            c=c, label=name, s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
        )
        _confidence_ellipse(scores[mask, 0], scores[mask, 1], ax, c)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved scatter: %s", out_path)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: Sequence[str],
    title: str = "Confusion Matrix",
    out_path: Optional[Path] = None,
) -> None:
    """Heatmap confusion matrix with counts and percentages.

    Parameters
    ----------
    cm : ndarray, shape (n_classes, n_classes)
        Confusion matrix (rows=true, columns=predicted).
    label_names : sequence of str
        Class names.
    title : str
        Plot title.
    out_path : Path, optional
        Save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div by zero
    cm_pct = cm / row_sums * 100

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)",
                    ha="center", va="center", fontsize=9, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved confusion matrix: %s", out_path)
    plt.close(fig)


def plot_permutation_distribution(
    null_distribution: np.ndarray,
    observed: float,
    p_value: float,
    title: str = "Permutation Test",
    xlabel: str = "Accuracy",
    out_path: Optional[Path] = None,
) -> None:
    """Histogram of null distribution with observed statistic line.

    Parameters
    ----------
    null_distribution : ndarray
        Null distribution values from permutation test.
    observed : float
        Observed test statistic.
    p_value : float
        Permutation p-value.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    out_path : Path, optional
        Save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_distribution, bins=50, color="#888888", alpha=0.7,
            edgecolor="white", linewidth=0.5, density=True)
    ax.axvline(observed, color="#c62828", linewidth=2, linestyle="--",
               label=f"Observed = {observed:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{title}  (p = {p_value:.4f})", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved permutation plot: %s", out_path)
    plt.close(fig)


def plot_scree(
    variance_ratio: np.ndarray,
    cumulative: np.ndarray,
    title: str = "PCA Scree Plot",
    out_path: Optional[Path] = None,
) -> None:
    """Scree plot with individual and cumulative variance explained.

    Parameters
    ----------
    variance_ratio : ndarray
        Fraction of variance explained per component.
    cumulative : ndarray
        Cumulative variance explained.
    title : str
        Plot title.
    out_path : Path, optional
        Save figure to this path.
    """
    n = len(variance_ratio)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(x, variance_ratio * 100, color="#1a5276", alpha=0.7, label="Individual")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title(title, fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100, "o-", color="#2E7D32", linewidth=2, markersize=4,
             label="Cumulative")
    ax2.axhline(95, color="#888", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_ylabel("Cumulative (%)")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    if n <= 20:
        ax1.set_xticks(x)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved scree plot: %s", out_path)
    plt.close(fig)


def plot_feature_loadings(
    loadings: np.ndarray,
    feature_names: Sequence[str],
    component_label: str = "PC1",
    n_top: int = 15,
    title: str = "",
    out_path: Optional[Path] = None,
) -> None:
    """Horizontal bar chart of top feature loadings for one component.

    Parameters
    ----------
    loadings : ndarray, shape (n_features,)
        Loading values for one component.
    feature_names : sequence of str
        Feature names.
    component_label : str
        Label for the component axis.
    n_top : int
        Show top N features by absolute loading.
    title : str
        Plot title.
    out_path : Path, optional
        Save figure to this path.
    """
    order = np.argsort(np.abs(loadings))[::-1][:n_top]
    # Reverse so largest is at top of horizontal bar chart
    order = order[::-1]
    names = [feature_names[i] for i in order]
    vals = loadings[order]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * n_top)))
    colors = ["#1a5276" if v >= 0 else "#c62828" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"{component_label} Loading")
    ax.axvline(0, color="black", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved loadings: %s", out_path)
    plt.close(fig)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    r_squared: float,
    spearman_rho: float,
    title: str = "Predicted vs Actual",
    out_path: Optional[Path] = None,
) -> None:
    """Scatter plot of predicted vs actual dose (ordinal) with identity line.

    Points are jittered horizontally and coloured by true dose group.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples,)
        True ordinal dose values (0, 1, 2, 3).
    y_pred : ndarray, shape (n_samples,)
        Predicted dose values (continuous).
    label_names : sequence of str
        Group names for colouring (e.g. ['C', 'L', 'M', 'H']).
    r_squared : float
        Coefficient of determination.
    spearman_rho : float
        Spearman correlation between predicted and true.
    title : str
        Plot title.
    out_path : Path, optional
        Save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = _get_colors(label_names)

    # Jitter true values for visibility
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, size=len(y_true))
    x_jittered = y_true + jitter

    unique_vals = np.unique(y_true.astype(int))
    for val in unique_vals:
        mask = y_true.astype(int) == val
        name = label_names[val] if val < len(label_names) else str(val)
        c = colors[val % len(colors)]
        ax.scatter(
            x_jittered[mask], y_pred[mask],
            c=c, label=name, s=40, alpha=0.7,
            edgecolors="white", linewidths=0.5,
        )

    # Identity line
    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=1, alpha=0.7,
            label="Identity")

    # Linear regression fit line
    if len(y_true) > 2:
        coeffs = np.polyfit(y_true, y_pred, 1)
        x_fit = np.linspace(y_true.min() - 0.2, y_true.max() + 0.2, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "-", color="#1a5276",
                linewidth=1.5, alpha=0.7)

    ax.set_xlabel("True Dose (ordinal)")
    ax.set_ylabel("Predicted Dose")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_title(f"{title}\nR²={r_squared:.3f}   ρ={spearman_rho:.3f}", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
        logger.info("Saved predicted vs actual: %s", out_path)
    plt.close(fig)
