"""
Cross-validated classification with permutation testing.

Runs LOOCV (leave-one-out cross-validation) with linear SVM and multinomial
logistic regression. Permutation testing provides empirical p-values for
classification accuracy. When n_pca_components is specified, PCA is fit
inside each LOOCV fold to avoid data leakage, and model weights are mapped
back to ROI space for interpretation.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC

from neurofaune.network.classification.visualization import (
    plot_confusion_matrix,
    plot_permutation_distribution,
)

logger = logging.getLogger(__name__)


def _loocv_accuracy(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    n_pca_components: Optional[int] = None,
) -> tuple[float, float, np.ndarray]:
    """Run LOOCV and return accuracy, balanced accuracy, and predictions.

    Parameters
    ----------
    clf : sklearn classifier
        Classifier template (will be cloned per fold).
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Integer labels.
    n_pca_components : int, optional
        If set, fit PCA inside each fold for dimensionality reduction.

    Returns
    -------
    accuracy : float
    balanced_accuracy : float
    y_pred : ndarray, shape (n_samples,)
    """
    loo = LeaveOneOut()
    y_pred = np.empty_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        if n_pca_components is not None:
            pca = PCA(n_components=n_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        clf_copy = clone(clf)
        clf_copy.fit(X_train, y[train_idx])
        y_pred[test_idx] = clf_copy.predict(X_test)

    accuracy = float(np.mean(y_pred == y))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    return accuracy, bal_acc, y_pred


def _determine_n_pca(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Fit PCA on X and return n_components for the given variance threshold."""
    pca_full = PCA().fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


def _extract_roi_weights(
    clf,
    pca: PCA,
    clf_name: str,
    feature_names: Sequence[str],
) -> dict:
    """Map classifier weights back to ROI space through PCA components.

    Returns dict with roi_weights (1D array) and top_features (list of tuples).
    """
    if clf_name == "svm":
        # OVO SVM: coef_ is (n_class_pairs, n_pca_components)
        roi_weights_per_pair = clf.coef_ @ pca.components_
        roi_weights = np.mean(np.abs(roi_weights_per_pair), axis=0)
    else:
        # Logistic regression: coef_ is (n_classes, n_pca_components)
        roi_weights_per_class = clf.coef_ @ pca.components_
        roi_weights = np.mean(np.abs(roi_weights_per_class), axis=0)

    # Top features by absolute weight
    order = np.argsort(np.abs(roi_weights))[::-1]
    top_features = [(feature_names[i], float(roi_weights[i])) for i in order[:20]]

    return {"roi_weights": roi_weights, "top_features": top_features}


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    n_permutations: int = 1000,
    seed: int = 42,
    output_dir: Path = None,
    use_pca: bool = False,
) -> dict:
    """LOOCV classification with SVM and logistic regression + permutation test.

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

    Returns
    -------
    dict with keys per classifier ('svm', 'logistic'):
        accuracy : float
        balanced_accuracy : float
        confusion_matrix : ndarray
        permutation_p_value : float
        null_distribution : ndarray
        per_class_accuracy : dict
        roi_weights : ndarray (only when use_pca=True)
        pca_n_components : int (only when use_pca=True)
        top_features : list (only when use_pca=True)
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Determine PCA dimensionality if requested
    n_pca = None
    if use_pca:
        n_pca = _determine_n_pca(X)
        # Cap at n_samples - 1 (max rank)
        n_pca = min(n_pca, X.shape[0] - 1, X.shape[1])
        logger.info("PCA reduction: %d features -> %d components (95%% variance)",
                     X.shape[1], n_pca)

    classifiers = {
        "svm": SVC(kernel="linear", C=1.0, max_iter=10000),
        "logistic": LogisticRegression(
            solver="lbfgs", max_iter=5000,
        ),
    }

    results = {}
    for clf_name, clf in classifiers.items():
        logger.info("Running LOOCV for %s...", clf_name)

        # Observed accuracy
        acc, bal_acc, y_pred = _loocv_accuracy(clf, X, y, n_pca_components=n_pca)
        cm = confusion_matrix(y, y_pred, labels=list(range(len(label_names))))

        # Per-class accuracy
        per_class = {}
        for i, name in enumerate(label_names):
            mask = y == i
            if mask.sum() > 0:
                per_class[name] = float(np.mean(y_pred[mask] == y[mask]))

        logger.info(
            "  %s: accuracy=%.3f, balanced=%.3f, per_class=%s",
            clf_name, acc, bal_acc, per_class,
        )

        # Permutation test
        null_acc = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y)
            null_acc[i], _, _ = _loocv_accuracy(clf, X, y_perm, n_pca_components=n_pca)

        perm_p = float((np.sum(null_acc >= acc) + 1) / (n_permutations + 1))
        logger.info("  Permutation p-value: %.4f (n=%d)", perm_p, n_permutations)

        results[clf_name] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "confusion_matrix": cm,
            "permutation_p_value": perm_p,
            "null_distribution": null_acc,
            "per_class_accuracy": per_class,
        }

        # Weight inversion: fit PCA + classifier on full data for interpretation
        if use_pca:
            pca_interp = PCA(n_components=n_pca).fit(X)
            X_pca = pca_interp.transform(X)
            clf_full = clone(clf).fit(X_pca, y)
            weights = _extract_roi_weights(clf_full, pca_interp, clf_name, feature_names)
            results[clf_name]["roi_weights"] = weights["roi_weights"]
            results[clf_name]["top_features"] = weights["top_features"]
            results[clf_name]["pca_n_components"] = n_pca

        # Plots
        if output_dir is not None:
            plot_confusion_matrix(
                cm, label_names,
                title=f"{clf_name.upper()} Confusion Matrix (acc={acc:.2f})",
                out_path=output_dir / f"{clf_name}_confusion.png",
            )
            plot_permutation_distribution(
                null_acc, acc, perm_p,
                title=f"{clf_name.upper()} Permutation Test",
                xlabel="LOOCV Accuracy",
                out_path=output_dir / f"{clf_name}_permutation.png",
            )

    return results
