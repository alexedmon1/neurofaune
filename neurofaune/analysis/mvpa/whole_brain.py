"""
Whole-brain MVPA decoding with PCA dimensionality reduction.

Uses PCA to reduce voxel space to a compact set of components, then
fits a linear classifier (LinearSVC) or regressor (Ridge) in PCA space.
PCA is unsupervised, so transforms are pre-computed once and reused
across all label permutations.  Model weights are projected back to
voxel space via ``coef @ pca.components_`` for interpretation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)

logger = logging.getLogger(__name__)


def _determine_n_pca(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Return n_components for the given cumulative variance threshold."""
    pca_full = PCA().fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


def _precompute_pca_folds(
    X: np.ndarray, n_components: int, cv, y: np.ndarray,
) -> list:
    """Pre-compute PCA-transformed train/test splits for each CV fold.

    PCA is unsupervised (depends only on X), so transforms are invariant
    across label permutations and can be computed once.
    """
    folds = []
    for train_idx, test_idx in cv.split(X, y):
        n_comp = min(n_components, len(train_idx) - 1, X.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X[train_idx])
        X_test_pca = pca.transform(X[test_idx])
        folds.append((X_train_pca, X_test_pca, train_idx, test_idx))
    return folds


def _cv_score_precomputed(estimator, y, pca_folds, scoring):
    """Compute CV score using pre-computed PCA folds."""
    scores = []
    for X_train_pca, X_test_pca, train_idx, test_idx in pca_folds:
        est = clone(estimator)
        est.fit(X_train_pca, y[train_idx])
        if scoring == "accuracy":
            scores.append(float(np.mean(est.predict(X_test_pca) == y[test_idx])))
        else:  # r2
            y_pred = est.predict(X_test_pca)
            y_true = y[test_idx]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            scores.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return scores


def _permutation_test_precomputed(
    estimator, y, pca_folds, scoring, n_permutations, seed,
):
    """Permutation test reusing pre-computed PCA folds."""
    rng = np.random.default_rng(seed)
    observed_scores = _cv_score_precomputed(estimator, y, pca_folds, scoring)
    observed = float(np.mean(observed_scores))

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_scores = _cv_score_precomputed(
            estimator, y_perm, pca_folds, scoring,
        )
        null_dist[i] = float(np.mean(perm_scores))

        if (i + 1) % 100 == 0:
            logger.info("  Permutation %d/%d", i + 1, n_permutations)

    perm_p = float((np.sum(null_dist >= observed) + 1) / (n_permutations + 1))
    return observed, observed_scores, null_dist, perm_p


def run_whole_brain_decoding(
    images_4d,
    labels: Union[List[str], List[float], np.ndarray],
    mask_img,
    analysis_type: str = "classification",
    n_permutations: int = 1000,
    cv_folds: int = 5,
    screening_percentile: int = 20,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    pca_variance: float = 0.95,
) -> Dict[str, Any]:
    """Run whole-brain decoding with PCA + permutation testing.

    Uses PCA for dimensionality reduction (fit per CV fold to avoid
    leakage). PCA transforms are pre-computed once and reused across
    all label permutations since PCA is unsupervised. Model weights
    are projected back to voxel space for interpretation.

    Args:
        images_4d: 4D Nifti1Image (subjects along 4th dimension).
        labels: Group labels (str for classification, numeric for regression).
        mask_img: Brain mask Nifti1Image.
        analysis_type: 'classification' or 'regression'.
        n_permutations: Number of permutations for empirical p-value.
        cv_folds: Number of cross-validation folds.
        screening_percentile: Ignored (kept for API compat). PCA replaces
            ANOVA screening.
        seed: Random seed.
        output_dir: If set, save weight_map.nii.gz and results.json.
        pca_variance: Cumulative variance threshold for PCA (default 0.95).

    Returns:
        Dict with: mean_score, std_score, fold_scores, permutation_p,
        null_distribution, weight_img, analysis_type, pca_n_components.
    """
    from nilearn.maskers import NiftiMasker

    labels_arr = np.array(labels)
    n_samples = len(labels_arr)

    # Mask 4D → 2D matrix
    masker = NiftiMasker(mask_img=mask_img, standardize=True)
    X = masker.fit_transform(images_4d)

    # Determine PCA components
    n_pca = _determine_n_pca(X, pca_variance)
    n_pca = min(n_pca, n_samples - 1, X.shape[1])
    logger.info(
        "PCA reduction: %d voxels → %d components (%.0f%% variance)",
        X.shape[1], n_pca, pca_variance * 100,
    )

    # Set up CV and estimator
    if analysis_type == "classification":
        from sklearn.svm import LinearSVC
        scoring = "accuracy"
        score_label = "accuracy"
        estimator = LinearSVC(dual=False, max_iter=10000, random_state=seed)
        cv = StratifiedKFold(
            n_splits=min(cv_folds, n_samples), shuffle=True, random_state=seed,
        )
    else:
        from sklearn.linear_model import Ridge
        scoring = "r2"
        score_label = "r2"
        labels_arr = labels_arr.astype(float)
        estimator = Ridge(alpha=1.0)
        cv = KFold(
            n_splits=min(cv_folds, n_samples), shuffle=True, random_state=seed,
        )

    logger.info(
        "Whole-brain decoding: %s, n=%d, cv=%d-fold, %d PCA components",
        analysis_type, n_samples, cv_folds, n_pca,
    )

    # Pre-compute PCA folds (unsupervised, invariant to label permutation)
    logger.info("Pre-computing PCA for %d CV folds...", cv.get_n_splits())
    pca_folds = _precompute_pca_folds(X, n_pca, cv, labels_arr)

    # Observed CV scores
    fold_scores = _cv_score_precomputed(estimator, labels_arr, pca_folds, scoring)
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    logger.info(
        "Decoding %s: %.3f +/- %.3f", score_label, mean_score, std_score,
    )

    # Permutation test (reuses pre-computed PCA folds)
    logger.info("Running %d permutations for p-value...", n_permutations)
    observed_score, _, null_dist, perm_p = _permutation_test_precomputed(
        estimator, labels_arr, pca_folds, scoring, n_permutations, seed,
    )

    logger.info(
        "Permutation p-value: %.4f (observed=%.3f)", perm_p, observed_score,
    )

    # Weight inversion: fit PCA + estimator on full data, project back
    pca_full = PCA(n_components=n_pca).fit(X)
    X_pca_full = pca_full.transform(X)
    est_full = clone(estimator).fit(X_pca_full, labels_arr)
    coef_pca = est_full.coef_
    if coef_pca.ndim == 1:
        # Binary classification or regression: single weight vector
        voxel_weights = (coef_pca @ pca_full.components_).ravel()
    else:
        # Multi-class: average absolute weights across class vectors
        voxel_weights = np.mean(
            np.abs(coef_pca @ pca_full.components_), axis=0,
        )

    # Build weight NIfTI image
    weight_img = masker.inverse_transform(voxel_weights[np.newaxis, :])

    results = {
        "analysis_type": analysis_type,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_scores": [float(s) for s in fold_scores],
        "permutation_p": float(perm_p),
        "observed_permutation_score": float(observed_score),
        "null_distribution": null_dist,
        "weight_img": weight_img,
        "n_samples": n_samples,
        "cv_folds": cv_folds,
        "pca_n_components": n_pca,
        "n_permutations": n_permutations,
        "score_label": score_label,
    }

    # Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nib.save(weight_img, output_dir / "weight_map.nii.gz")

        results_json = {
            k: v for k, v in results.items()
            if k not in ("null_distribution", "weight_img")
        }
        results_json["null_distribution_summary"] = {
            "mean": float(np.mean(null_dist)),
            "std": float(np.std(null_dist)),
            "min": float(np.min(null_dist)),
            "max": float(np.max(null_dist)),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info("Saved whole-brain results to %s", output_dir)

    return results
