"""
Linear Discriminant Analysis (LDA) for supervised dimensionality reduction.

Projects ROI features onto discriminant axes that maximise between-group
separation. For 4 dose groups this yields 3 discriminant functions.
"""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from neurofaune.analysis.classification.visualization import (
    plot_feature_loadings,
    plot_scatter_2d,
)

logger = logging.getLogger(__name__)


def run_lda(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    output_dir: Path,
) -> dict:
    """Run LDA and save diagnostic plots.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Standardised feature matrix.
    y : ndarray, shape (n_samples,)
        Integer group labels.
    label_names : sequence of str
        Group names.
    feature_names : sequence of str
        Feature names.
    output_dir : Path
        Directory for output plots and results.

    Returns
    -------
    dict with keys:
        scores : ndarray, shape (n_samples, n_discriminants)
        explained_variance_ratio : ndarray
        scalings : ndarray, shape (n_features, n_discriminants) — structure correlations
        top_features : dict per LD axis — list of (feature_name, loading) tuples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(np.unique(y))
    n_discriminants = min(n_classes - 1, X.shape[1])

    lda = LinearDiscriminantAnalysis(n_components=n_discriminants)
    scores = lda.fit_transform(X, y)

    explained = lda.explained_variance_ratio_

    # Structure correlations: correlation of each original feature with each LD axis
    # This is more interpretable than raw coefficients for standardised data
    scalings = np.zeros((X.shape[1], scores.shape[1]))
    for j in range(scores.shape[1]):
        for i in range(X.shape[1]):
            valid = ~(np.isnan(X[:, i]) | np.isnan(scores[:, j]))
            if valid.sum() > 2:
                scalings[i, j] = np.corrcoef(X[valid, i], scores[valid, j])[0, 1]

    logger.info(
        "LDA: %d discriminant functions, LD1=%.1f%%, LD2=%.1f%%",
        n_discriminants,
        explained[0] * 100,
        explained[1] * 100 if len(explained) > 1 else 0.0,
    )

    # Scatter: LD1 vs LD2
    if scores.shape[1] >= 2:
        plot_scatter_2d(
            scores[:, :2], y, label_names,
            xlabel="LD1", ylabel="LD2",
            title="LDA — LD1 vs LD2",
            variance_explained=(explained[0] * 100, explained[1] * 100),
            out_path=output_dir / "scatter.png",
        )

    # Variance explained bar chart
    _plot_lda_variance(explained, output_dir / "variance.png")

    # Feature loadings (structure correlations) for LD1
    plot_feature_loadings(
        scalings[:, 0], feature_names,
        component_label="LD1",
        title=f"LD1 Structure Correlations ({explained[0] * 100:.1f}%)",
        out_path=output_dir / "loadings.png",
    )

    # Top features per axis
    top_features = {}
    for d in range(min(n_discriminants, 3)):
        order = np.argsort(np.abs(scalings[:, d]))[::-1][:10]
        top_features[f"LD{d + 1}"] = [
            (feature_names[i], float(scalings[i, d])) for i in order
        ]

    return {
        "scores": scores,
        "explained_variance_ratio": explained,
        "scalings": scalings,
        "top_features": top_features,
    }


def _plot_lda_variance(
    explained: np.ndarray,
    out_path: Path,
) -> None:
    """Bar chart of variance explained by each discriminant function."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(explained)
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), 3.5))
    x = np.arange(1, n + 1)
    ax.bar(x, explained * 100, color="#1a5276", alpha=0.8)
    ax.set_xlabel("Discriminant Function")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("LDA Variance Explained")
    ax.set_xticks(x)
    ax.set_xticklabels([f"LD{i}" for i in x])
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
