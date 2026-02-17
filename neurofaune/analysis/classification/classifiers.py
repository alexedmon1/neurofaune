"""
Cross-validated classification with permutation testing.

Runs LOOCV (leave-one-out cross-validation) with linear SVM and multinomial
logistic regression. Permutation testing provides empirical p-values for
classification accuracy.
"""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC

from neurofaune.analysis.classification.visualization import (
    plot_confusion_matrix,
    plot_permutation_distribution,
)

logger = logging.getLogger(__name__)


def _loocv_accuracy(
    clf,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Run LOOCV and return accuracy, balanced accuracy, and predictions.

    Returns
    -------
    accuracy : float
    balanced_accuracy : float
    y_pred : ndarray, shape (n_samples,)
    """
    loo = LeaveOneOut()
    y_pred = np.empty_like(y)

    for train_idx, test_idx in loo.split(X):
        clf_copy = _clone_clf(clf)
        clf_copy.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf_copy.predict(X[test_idx])

    accuracy = float(np.mean(y_pred == y))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    return accuracy, bal_acc, y_pred


def _clone_clf(clf):
    """Create a fresh clone of a classifier with the same parameters."""
    from sklearn.base import clone
    return clone(clf)


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
    n_permutations: int = 1000,
    seed: int = 42,
    output_dir: Path = None,
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

    Returns
    -------
    dict with keys per classifier ('svm', 'logistic'):
        accuracy : float
        balanced_accuracy : float
        confusion_matrix : ndarray
        permutation_p_value : float
        null_distribution : ndarray
        per_class_accuracy : dict
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

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
        acc, bal_acc, y_pred = _loocv_accuracy(clf, X, y)
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
            null_acc[i], _, _ = _loocv_accuracy(clf, X, y_perm)

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
