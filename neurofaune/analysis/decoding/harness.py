"""The evaluation harness. Decoding everywhere goes through it, under these rules:

1. **Folds are grouped by animal.** :func:`evaluate` needs ``groups`` and refuses to run
   without them, and no caller passes a splitter. With fewer folds than animals,
   classification folds are also stratified, so no test fold holds one class.
2. **The score is reported against a permutation null**, with labels permuted between
   animals. Classes are often imbalanced, so chance is not 50%; classification is scored
   by balanced accuracy, regression by R2.
3. **Everything fitted is fitted inside the fold**: imputation, scaling and PCA are steps
   of the pipeline.
4. **Estimators are sklearn-compatible objects.** Pass one as ``estimator`` and it gets
   rules 1-3.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from neurofaune.config import get_config_value

RANDOM_STATE = 0


@dataclass(frozen=True)
class DecodeSettings:
    """The model and its evaluation; ``from_config`` reads them from ``decoding:``."""

    cv_folds: Any = "loo"
    kernel: str = "linear"
    C: float = 1.0
    alpha: float = 1.0
    pca_components: int | None = 20
    permutations: int = 1000
    workers: int = 8
    threshold_min_per_side: int = 4

    @classmethod
    def from_config(cls, config: dict) -> DecodeSettings:
        return cls(
            cv_folds=get_config_value(config, "decoding.cv_folds", default=cls.cv_folds),
            kernel=get_config_value(config, "decoding.kernel", default=cls.kernel),
            C=float(get_config_value(config, "decoding.C", default=cls.C)),
            alpha=float(get_config_value(config, "decoding.alpha", default=cls.alpha)),
            pca_components=get_config_value(
                config, "decoding.pca_components", default=cls.pca_components
            ),
            permutations=int(
                get_config_value(config, "decoding.permutations", default=cls.permutations)
            ),
            workers=int(get_config_value(config, "decoding.workers", default=cls.workers)),
        )


def make_pipeline(
    settings: DecodeSettings,
    smallest_train: int,
    analysis_type: str = "classification",
    estimator: Any = None,
    n_features: int | None = None,
):
    """Impute -> scale -> (PCA) -> estimator, all fitted inside the fold.

    PCA components are capped below the smallest training fold and at the feature count:
    asking for more components than the data has would fail.
    """
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if settings.pca_components:
        n_components = min(int(settings.pca_components), max(1, smallest_train - 1))
        if n_features is not None:
            n_components = min(n_components, n_features)
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    if estimator is not None:
        steps.append(("estimator", estimator))
    elif analysis_type == "regression":
        steps.append(("ridge", Ridge(alpha=settings.alpha)))
    else:
        steps.append((
            "svm",
            SVC(
                kernel=settings.kernel,
                C=settings.C,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ))
    return Pipeline(steps)


def _splitter(settings: DecodeSettings, groups: np.ndarray, stratify: bool = False):
    """``(splitter, n_splits)``: one animal out, or k folds of whole animals.

    With ``stratify`` (classification), k folds are ``StratifiedGroupKFold``. ``GroupKFold``
    deals animals into folds by sorted id and does not balance the classes, so where ids
    follow the group (alternating, or one group's ids first) its test folds can each hold
    one class, and a relabelling that is mostly wrong can then score perfectly. The
    stratified splitter is given the labels being scored, so every permutation gets folds
    balanced on its own labels, as the observed labels do.
    """
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold

    n_groups = len(np.unique(groups))
    folds = settings.cv_folds
    if folds == "loo" or int(folds) >= n_groups:
        return LeaveOneGroupOut(), n_groups
    if stratify:
        return StratifiedGroupKFold(n_splits=int(folds)), int(folds)
    return GroupKFold(n_splits=int(folds)), int(folds)


def _smallest_train(cv, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> int:
    return min(len(train) for train, _ in cv.split(X, y, groups=groups))


def _animal_shuffles(y: np.ndarray, groups: np.ndarray, n_perm: int) -> list[np.ndarray]:
    """Relabellings that move labels between animals, keeping each animal's rows together.

    Permuting rows would give one animal's sessions different labels, which the real labels
    never do. Generated up front, so the null does not depend on how the work is distributed.
    """
    codes, _ = pd.factorize(groups)
    first = pd.Series(codes).drop_duplicates().index.to_numpy()
    if not np.array_equal(y, y[first][codes]):
        raise ValueError(
            "the target differs between rows of one animal, so it cannot be permuted between "
            "animals; use one row per animal (design 'delta' or 'cross_sectional')"
        )
    rng = np.random.default_rng(RANDOM_STATE)
    return [rng.permutation(y[first])[codes] for _ in range(n_perm)]


def _permutation_null(
    settings: DecodeSettings,
    pipe,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    score_fn: Callable,
) -> np.ndarray:
    from sklearn.model_selection import cross_val_predict

    if not settings.permutations:
        return np.empty(0)
    shuffles = _animal_shuffles(y, groups, settings.permutations)

    def score(shuffled):
        return score_fn(shuffled, cross_val_predict(pipe, X, shuffled, groups=groups, cv=cv))

    if settings.workers > 1:
        from joblib import Parallel, delayed

        return np.asarray(
            Parallel(n_jobs=settings.workers)(delayed(score)(sh) for sh in shuffles)
        )
    return np.asarray([score(sh) for sh in shuffles])


def _p_value(null: np.ndarray, observed: float) -> float:
    if not len(null):
        return float("nan")
    return (1 + int((null >= observed).sum())) / (1 + len(null))


def evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    settings: DecodeSettings | None = None,
    analysis_type: str = "classification",
    estimator: Any = None,
    classes: Sequence[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Animal-grouped cross-validated score and its permutation p-value.

    ``classes`` orders two groups, negative first (default sorted). Returns the scores and
    one row of predictions per sample.
    """
    if groups is None:
        raise ValueError("decoding needs groups: the animal of each row")
    if len(groups) != len(X):
        raise ValueError(f"{len(groups)} groups for {len(X)} rows")
    settings = settings or DecodeSettings()
    if analysis_type == "regression":
        return _evaluate_regression(X, y, groups, settings, estimator)
    if analysis_type != "classification":
        raise ValueError(f"analysis_type must be classification or regression: {analysis_type!r}")
    return _evaluate_classification(X, y, groups, settings, estimator, classes)


def _evaluate_classification(X, y, groups, settings, estimator, classes):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import cross_val_predict

    found = sorted(y.unique())
    if len(found) != 2:
        raise ValueError(f"decoding needs exactly 2 groups, found {found}")
    if classes is not None:
        found = [c for c in classes if c in found]
    y_binary = (y == found[1]).astype(int).values
    group_values = np.asarray(groups)
    cv, n_splits = _splitter(settings, group_values, stratify=True)
    pipe = make_pipeline(
        settings,
        _smallest_train(cv, X.values, y_binary, group_values),
        estimator=estimator,
        n_features=X.shape[1],
    )

    pred = cross_val_predict(pipe, X.values, y_binary, groups=group_values, cv=cv)
    balanced = balanced_accuracy_score(y_binary, pred)
    try:
        decision = cross_val_predict(
            pipe, X.values, y_binary, groups=group_values, cv=cv, method="decision_function"
        )
        auc = roc_auc_score(y_binary, decision)
    except (AttributeError, ValueError):  # not every estimator has a decision_function
        auc = float("nan")

    null = _permutation_null(
        settings, pipe, X.values, y_binary, group_values, cv, balanced_accuracy_score
    )
    scores = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_animals": len(np.unique(group_values)),
        "n_splits": n_splits,
        "analysis_type": "classification",
        "negative_class": found[0],
        "positive_class": found[1],
        "class_balance": f"{int(y_binary.sum())}/{len(y_binary)}",
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "permutations": len(null),
        "null_mean": float(null.mean()) if len(null) else float("nan"),
        "p_permutation": _p_value(null, balanced),
    }
    predictions = pd.DataFrame({
        "sample": X.index, "animal": group_values, "y_true": y.values, "y_pred_binary": pred,
    })
    return scores, predictions


def _evaluate_regression(X, y, groups, settings, estimator):
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_predict

    numeric = pd.to_numeric(y, errors="coerce")
    usable = numeric.notna().values
    X, numeric, group_values = X[usable], numeric[usable], np.asarray(groups)[usable]
    if len(X) < 4:
        raise ValueError(f"regression needs at least 4 usable samples, got {len(X)}")

    cv, n_splits = _splitter(settings, group_values)
    pipe = make_pipeline(
        settings,
        _smallest_train(cv, X.values, numeric.values, group_values),
        analysis_type="regression",
        estimator=estimator,
        n_features=X.shape[1],
    )
    pred = cross_val_predict(pipe, X.values, numeric.values, groups=group_values, cv=cv)
    score = r2_score(numeric.values, pred)
    null = _permutation_null(settings, pipe, X.values, numeric.values, group_values, cv, r2_score)
    scores = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_animals": len(np.unique(group_values)),
        "n_splits": n_splits,
        "analysis_type": "regression",
        "target_range": f"{numeric.min():.4g}..{numeric.max():.4g}",
        "r2": score,
        "permutations": len(null),
        "null_mean": float(null.mean()) if len(null) else float("nan"),
        "p_permutation": _p_value(null, score),
    }
    predictions = pd.DataFrame({
        "sample": X.index, "animal": group_values, "y_true": numeric.values, "y_pred": pred,
    })
    return scores, predictions


def threshold_scan(
    X: pd.DataFrame,
    exposure: pd.Series,
    groups: pd.Series,
    settings: DecodeSettings | None = None,
) -> pd.DataFrame:
    """Score a classifier at each candidate cut-point on a continuous exposure axis.

    The p-values are per cut-point; the best of them is selected, so it is not corrected.
    """
    settings = settings or DecodeSettings()
    exposure = pd.to_numeric(exposure, errors="coerce")
    usable = exposure.notna().values
    X, exposure, groups = X[usable], exposure[usable], groups[usable]

    rows = []
    for low, high in pairwise(np.unique(exposure.values)):
        cut = (low + high) / 2
        y = pd.Series(np.where(exposure.values > cut, "above", "below"), index=exposure.index)
        if y.value_counts().min() < settings.threshold_min_per_side:
            continue
        scores, _ = evaluate(X, y, groups, settings)
        rows.append({"threshold": cut, **scores})
    return pd.DataFrame(rows)
