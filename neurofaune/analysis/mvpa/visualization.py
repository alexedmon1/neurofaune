"""
Visualization functions for MVPA results.

Nilearn plotting wrappers for weight maps, searchlight maps, and
decoding score distributions. Uses Agg backend for non-interactive
server environments.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_weight_map(
    weight_img,
    mask_img,
    output_path: Path,
    title: str = "Decoder Weight Map",
) -> Path:
    """Plot decoder weight map as mosaic of axial slices.

    Args:
        weight_img: Nifti1Image of decoder weights.
        mask_img: Nifti1Image brain mask for background.
        output_path: Path to save PNG.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    from nilearn import plotting

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 5))
    display = plotting.plot_stat_map(
        weight_img,
        bg_img=mask_img,
        display_mode="z",
        cut_coords=7,
        title=title,
        colorbar=True,
        figure=fig,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved weight map: %s", output_path)
    return output_path


def plot_glass_brain(
    stat_img,
    output_path: Path,
    title: str = "Glass Brain",
    threshold: float = 0.0,
) -> Path:
    """Plot glass brain projection of a statistical map.

    Args:
        stat_img: Nifti1Image statistical map.
        output_path: Path to save PNG.
        title: Plot title.
        threshold: Display threshold.

    Returns:
        Path to saved figure.
    """
    from nilearn import plotting

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    display = plotting.plot_glass_brain(
        stat_img,
        display_mode="ortho",
        threshold=threshold,
        title=title,
        colorbar=True,
        figure=fig,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved glass brain: %s", output_path)
    return output_path


def plot_searchlight_map(
    searchlight_img,
    threshold: float,
    output_path: Path,
    title: str = "Searchlight Map",
    bg_img=None,
) -> Path:
    """Plot thresholded searchlight accuracy map.

    Args:
        searchlight_img: Nifti1Image of searchlight scores.
        threshold: Display threshold (e.g. FWER-corrected value).
        output_path: Path to save PNG.
        title: Plot title.
        bg_img: Optional background image.

    Returns:
        Path to saved figure.
    """
    from nilearn import plotting

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 5))
    display = plotting.plot_stat_map(
        searchlight_img,
        bg_img=bg_img,
        display_mode="z",
        cut_coords=7,
        threshold=threshold,
        title=title,
        colorbar=True,
        figure=fig,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved searchlight map: %s", output_path)
    return output_path


def plot_decoding_scores(
    fold_scores: List[float],
    null_distribution: np.ndarray,
    observed: float,
    p_value: float,
    output_path: Path,
    title: str = "Decoding Performance",
    metric_label: str = "Accuracy",
) -> Path:
    """Plot bar chart of fold scores and permutation null histogram.

    Args:
        fold_scores: Per-fold accuracy/R2 values.
        null_distribution: Permutation null scores.
        observed: Observed mean score.
        p_value: Permutation p-value.
        output_path: Path to save PNG.
        title: Plot title.
        metric_label: Label for the score axis.

    Returns:
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: fold scores bar chart
    ax = axes[0]
    x = np.arange(len(fold_scores))
    ax.bar(x, fold_scores, color="#1a5276", alpha=0.8)
    ax.axhline(observed, color="#c62828", linestyle="--", linewidth=1.5,
               label=f"Mean = {observed:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_label)
    ax.set_title("Cross-Validation Scores")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i+1}" for i in x])
    ax.legend(fontsize=9)

    # Right: permutation null histogram
    ax = axes[1]
    if len(null_distribution) > 0:
        ax.hist(null_distribution, bins=40, color="#888888", alpha=0.7,
                edgecolor="white", label="Null distribution")
    ax.axvline(observed, color="#c62828", linestyle="--", linewidth=2,
               label=f"Observed = {observed:.3f}")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test (p = {p_value:.4f})")
    ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved decoding scores: %s", output_path)
    return output_path


def plot_dose_response_brain(
    weight_img,
    output_path: Path,
    title: str = "Dose-Response Regression Weights",
    bg_img=None,
) -> Path:
    """Plot regression weight map for dose-response decoding.

    Args:
        weight_img: Nifti1Image of regression weights.
        output_path: Path to save PNG.
        title: Plot title.
        bg_img: Optional background image.

    Returns:
        Path to saved figure.
    """
    from nilearn import plotting

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 5))
    display = plotting.plot_stat_map(
        weight_img,
        bg_img=bg_img,
        display_mode="z",
        cut_coords=7,
        title=title,
        colorbar=True,
        cmap="RdBu_r",
        figure=fig,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved dose-response weight map: %s", output_path)
    return output_path
