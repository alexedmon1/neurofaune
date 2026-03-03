"""
Cross-validated regression with permutation testing for dose-response.

Treats dose as ordinal (C=0, L=1, M=2, H=3) and tests whether joint ROI
patterns predict dose level using LOOCV with SVR, Ridge, and PLS regressors.
Complements classification (discrete groups) by testing for a continuous
dose-response relationship. When use_pca is set, PCA is fit inside each
LOOCV fold to avoid data leakage, and model weights are mapped back to ROI
space for interpretation.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR

from neurofaune.network.classification.visualization import (
    plot_permutation_distribution,
    plot_predicted_vs_actual,
)

logger = logging.getLogger(__name__)


def _loocv_regression(
    reg,
    X: np.ndarray,
    y: np.ndarray,
    n_pca_components: Optional[int] = None,
) -> tuple[float, float, float, np.ndarray]:
    """Run LOOCV for a regressor and return metrics + predictions.

    Parameters
    ----------
    reg : sklearn regressor
        Regressor template (will be cloned per fold).
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Target values.
    n_pca_components : int, optional
        If set, fit PCA inside each fold for dimensionality reduction.

    Returns
    -------
    r_squared : float
    mae : float
    spearman_rho : float
    y_pred : ndarray, shape (n_samples,)
    """
    loo = LeaveOneOut()
    y_pred = np.empty_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        if n_pca_components is not None:
            pca = PCA(n_components=n_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        reg_copy = clone(reg)
        reg_copy.fit(X_train, y[train_idx])
        pred = reg_copy.predict(X_test)
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


def _determine_n_pca(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Fit PCA on X and return n_components for the given variance threshold."""
    pca_full = PCA().fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


def _extract_roi_weights(
    reg,
    pca: PCA,
    reg_name: str,
    feature_names: Sequence[str],
) -> dict:
    """Map regressor weights back to ROI space through PCA components.

    Returns dict with roi_weights (1D array) and top_features (list of tuples).
    """
    if reg_name == "pls":
        # PLS: coef_ is (n_features_pca, 1)
        coef = reg.coef_.ravel()
    elif reg_name == "svr":
        # SVR: coef_ is (1, n_features_pca)
        coef = reg.coef_.ravel()
    else:
        # Ridge: coef_ is (n_features_pca,)
        coef = reg.coef_.ravel()

    roi_weights = (coef @ pca.components_).ravel()

    # Top features by absolute weight
    order = np.argsort(np.abs(roi_weights))[::-1]
    top_features = [(feature_names[i], float(roi_weights[i])) for i in order[:20]]

    return {"roi_weights": roi_weights, "top_features": top_features}


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    n_permutations: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    use_pca: bool = False,
    continuous_target: bool = False,
    dose_labels: Optional[np.ndarray] = None,
) -> dict:
    """LOOCV regression with SVR, Ridge, and PLS + permutation test.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Standardised feature matrix.
    y : ndarray, shape (n_samples,)
        Target values (ordinal ints or continuous floats).
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
    use_pca : bool
        Whether to apply PCA dimensionality reduction (fit per LOOCV fold).
        When True, model weights are mapped back to ROI space for
        interpretation.
    continuous_target : bool
        If True, pass continuous_target to plot_predicted_vs_actual for
        proper axis handling (no jitter, continuous x-axis).
    dose_labels : ndarray, optional
        Integer dose group per sample for plot colouring when using a
        continuous target.

    Returns
    -------
    dict with keys per regressor ('svr', 'ridge', 'pls'):
        r_squared : float
        mae : float
        spearman_rho : float
        permutation_p_value : float — p-value for R² exceeding null
        null_distribution : ndarray — null R² values
        y_pred : ndarray — LOOCV predictions
        roi_weights : ndarray (only when use_pca=True)
        pca_n_components : int (only when use_pca=True)
        top_features : list (only when use_pca=True)
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    y_float = y.astype(float)

    # Determine PCA dimensionality if requested
    n_pca = None
    if use_pca:
        n_pca = _determine_n_pca(X)
        n_pca = min(n_pca, X.shape[0] - 1, X.shape[1])
        logger.info("PCA reduction: %d features -> %d components (95%% variance)",
                     X.shape[1], n_pca)

    # Determine PLS n_components (min of n_samples-1, n_features, n_classes-1)
    n_classes = len(np.unique(y))
    n_feat_for_pls = n_pca if n_pca is not None else X.shape[1]
    pls_components = min(n_classes - 1, n_feat_for_pls, X.shape[0] - 1)
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
        r2, mae, rho, y_pred = _loocv_regression(reg, X, y_float, n_pca_components=n_pca)

        logger.info(
            "  %s: R²=%.3f, MAE=%.3f, ρ=%.3f",
            reg_name, r2, mae, rho,
        )

        # Permutation test on R²
        null_r2 = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y_float)
            null_r2[i], _, _, _ = _loocv_regression(reg, X, y_perm, n_pca_components=n_pca)

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

        # Weight inversion: fit PCA + regressor on full data for interpretation
        if use_pca:
            pca_interp = PCA(n_components=n_pca).fit(X)
            X_pca = pca_interp.transform(X)
            reg_full = clone(reg).fit(X_pca, y_float)
            weights = _extract_roi_weights(reg_full, pca_interp, reg_name, feature_names)
            results[reg_name]["roi_weights"] = weights["roi_weights"]
            results[reg_name]["top_features"] = weights["top_features"]
            results[reg_name]["pca_n_components"] = n_pca

        # Plots
        if output_dir is not None:
            plot_predicted_vs_actual(
                y_float, y_pred, label_names,
                r_squared=r2, spearman_rho=rho,
                title=f"{reg_name.upper()} — Predicted vs Actual",
                out_path=output_dir / f"{reg_name}_predicted_vs_actual.png",
                continuous_target=continuous_target,
                dose_labels=dose_labels,
            )
            plot_permutation_distribution(
                null_r2, r2, perm_p,
                title=f"{reg_name.upper()} Permutation Test",
                xlabel="LOOCV R²",
                out_path=output_dir / f"{reg_name}_permutation.png",
            )

    return results
