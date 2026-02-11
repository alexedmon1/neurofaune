"""
PCA dimensionality reduction and visualization.

Runs PCA on the standardised ROI feature matrix, produces scatter plots
of PC1 vs PC2, scree plots, and top feature loading charts.
"""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

from neurofaune.analysis.classification.visualization import (
    plot_feature_loadings,
    plot_scatter_2d,
    plot_scree,
)

logger = logging.getLogger(__name__)


def run_pca(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    output_dir: Path,
) -> dict:
    """Run PCA and save diagnostic plots.

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
        scores : ndarray, shape (n_samples, n_components)
        explained_variance_ratio : ndarray
        cumulative_variance : ndarray
        n_components_95pct : int
        loadings : ndarray, shape (n_features, n_components)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    loadings = pca.components_.T  # (n_features, n_components)

    logger.info(
        "PCA: %d components, PC1=%.1f%%, PC2=%.1f%%, 95%% at %d components",
        n_components, explained[0] * 100,
        explained[1] * 100 if len(explained) > 1 else 0.0,
        n95,
    )

    # Scatter: PC1 vs PC2
    if scores.shape[1] >= 2:
        plot_scatter_2d(
            scores[:, :2], y, label_names,
            xlabel="PC1", ylabel="PC2",
            title="PCA — PC1 vs PC2",
            variance_explained=(explained[0] * 100, explained[1] * 100),
            out_path=output_dir / "scatter.png",
        )

    # Scree plot
    n_show = min(20, len(explained))
    plot_scree(
        explained[:n_show], cumulative[:n_show],
        title="PCA Scree Plot",
        out_path=output_dir / "scree.png",
    )

    # Feature loadings for PC1
    plot_feature_loadings(
        loadings[:, 0], feature_names,
        component_label="PC1",
        title=f"PC1 Loadings ({explained[0] * 100:.1f}% variance)",
        out_path=output_dir / "loadings.png",
    )

    return {
        "scores": scores,
        "explained_variance_ratio": explained,
        "cumulative_variance": cumulative,
        "n_components_95pct": n95,
        "loadings": loadings,
    }
