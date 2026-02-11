"""
Cross-validated regression with permutation testing for dose-response.

Treats dose as ordinal (C=0, L=1, M=2, H=3) and tests whether joint ROI
patterns predict dose level using LOOCV with SVR, Ridge, and PLS regressors.
Complements classification (discrete groups) by testing for a continuous
dose-response relationship.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sp_stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR

from neurofaune.analysis.classification.visualization import (
    plot_permutation_distribution,
    plot_predicted_vs_actual,
)

logger = logging.getLogger(__name__)


def _loocv_regression(
    reg,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Run LOOCV for a regressor and return metrics + predictions.

    Returns
    -------
    r_squared : float
    mae : float
    spearman_rho : float
    y_pred : ndarray, shape (n_samples,)
    """
    from sklearn.base import clone

    loo = LeaveOneOut()
    y_pred = np.empty_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        reg_copy = clone(reg)
        reg_copy.fit(X[train_idx], y[train_idx])
        pred = reg_copy.predict(X[test_idx])
        # PLS returns 2D array
        y_pred[test_idx] = np.atleast_1d(pred).ravel()[0]

    # Metrics
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(y - y_pred)))
    rho, _ = sp_stats.spearmanr(y, y_pred)
    spearman_rho = float(rho) if not np.isnan(rho) else 0.0

    return r_squared, mae, spearman_rho, y_pred


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    n_permutations: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict:
    """LOOCV regression with SVR, Ridge, and PLS + permutation test.

    Treats dose as ordinal (C=0, L=1, M=2, H=3) and fits regressors to
    predict dose from ROI feature patterns.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Standardised feature matrix.
    y : ndarray, shape (n_samples,)
        Ordinal dose values (0, 1, 2, 3).
    label_names : sequence of str
        Group names (e.g. ['C', 'L', 'M', 'H']).
    feature_names : sequence of str
        Feature names.
    n_permutations : int
        Number of label permutations for p-value (default 1000).
    seed : int
        Random seed.
    output_dir : Path, optional
        Directory for output plots.

    Returns
    -------
    dict with keys per regressor ('svr', 'ridge', 'pls'):
        r_squared : float
        mae : float
        spearman_rho : float
        permutation_p_value : float — p-value for R² exceeding null
        null_distribution : ndarray — null R² values
        y_pred : ndarray — LOOCV predictions
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    y_float = y.astype(float)

    # Determine PLS n_components (min of n_samples-1, n_features, n_classes-1)
    n_classes = len(np.unique(y))
    pls_components = min(n_classes - 1, X.shape[1], X.shape[0] - 1)
    pls_components = max(pls_components, 1)

    regressors = {
        "svr": SVR(kernel="linear", C=1.0),
        "ridge": Ridge(alpha=1.0),
        "pls": PLSRegression(n_components=pls_components),
    }

    results = {}
    for reg_name, reg in regressors.items():
        logger.info("Running LOOCV regression for %s...", reg_name)

        # Observed metrics
        r2, mae, rho, y_pred = _loocv_regression(reg, X, y_float)

        logger.info(
            "  %s: R²=%.3f, MAE=%.3f, ρ=%.3f",
            reg_name, r2, mae, rho,
        )

        # Permutation test on R²
        null_r2 = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y_float)
            null_r2[i], _, _, _ = _loocv_regression(reg, X, y_perm)

        perm_p = float((np.sum(null_r2 >= r2) + 1) / (n_permutations + 1))
        logger.info("  Permutation p-value (R²): %.4f (n=%d)", perm_p, n_permutations)

        results[reg_name] = {
            "r_squared": r2,
            "mae": mae,
            "spearman_rho": rho,
            "permutation_p_value": perm_p,
            "null_distribution": null_r2,
            "y_pred": y_pred,
        }

        # Plots
        if output_dir is not None:
            plot_predicted_vs_actual(
                y_float, y_pred, label_names,
                r_squared=r2, spearman_rho=rho,
                title=f"{reg_name.upper()} — Predicted vs Actual",
                out_path=output_dir / f"{reg_name}_predicted_vs_actual.png",
            )
            plot_permutation_distribution(
                null_r2, r2, perm_p,
                title=f"{reg_name.upper()} Permutation Test",
                xlabel="LOOCV R²",
                out_path=output_dir / f"{reg_name}_permutation.png",
            )

    return results
