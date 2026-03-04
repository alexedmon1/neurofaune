"""
Data preparation for multivariate classification and regression analysis.

Loads ROI-level wide CSVs (reusing covnet data loading), applies cohort
filtering, selects bilateral or territory feature sets, and standardises
features for downstream classifiers/regressors.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neurofaune.connectome.matrices import (
    bilateral_average,
    load_and_prepare_data,
)

logger = logging.getLogger(__name__)


def _load_filter_select(
    wide_csv: Union[str, Path],
    feature_set: str = "bilateral",
    cohort_filter: Optional[str] = None,
    exclusion_csv: Optional[Union[str, Path]] = None,
    standardize: bool = True,
) -> dict:
    """Shared data loading, filtering, feature selection, and scaling.

    Returns dict with: df, X, feature_names, scaler.
    """
    df, roi_cols = load_and_prepare_data(wide_csv, exclusion_csv)

    # Filter to known cohorts (drop ses-unknown)
    df = df[df["cohort"].isin(["p30", "p60", "p90"])].copy()

    # Cohort filter
    if cohort_filter is not None:
        df = df[df["cohort"] == cohort_filter].copy()
        if len(df) == 0:
            raise ValueError(f"No subjects remaining after cohort filter '{cohort_filter}'")
        logger.info("Cohort filter '%s': n=%d", cohort_filter, len(df))

    # Select feature set
    if feature_set == "bilateral":
        df, feature_names = bilateral_average(df, roi_cols)
        feature_names = [c for c in feature_names if not c.startswith("territory_")]
    elif feature_set == "territory":
        feature_names = [c for c in roi_cols if c.startswith("territory_")]
        if not feature_names:
            raise ValueError("No territory columns found in data")
    elif feature_set == "all":
        feature_names = [c for c in roi_cols if not c.startswith("territory_")]
        if not feature_names:
            raise ValueError("No individual ROI columns found in data")
    else:
        raise ValueError(
            f"Unknown feature_set: {feature_set!r}. Use 'bilateral', 'territory', or 'all'."
        )

    logger.info("Feature set '%s': %d features", feature_set, len(feature_names))

    # Build feature matrix
    X = df[feature_names].values.astype(np.float64)

    # Median imputation for any remaining NaN values
    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        logger.info("Imputing %d NaN values with column medians", n_nan)
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    # Standardize
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return {
        "df": df,
        "X": X,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def prepare_classification_data(
    wide_csv: Union[str, Path],
    feature_set: str = "bilateral",
    cohort_filter: Optional[str] = None,
    exclusion_csv: Optional[Union[str, Path]] = None,
    standardize: bool = True,
    target: str = "dose",
) -> dict:
    """Load, filter, and scale ROI data for classification.

    Parameters
    ----------
    wide_csv : Path
        Path to wide-format ROI CSV (from extract_roi_means.py).
    feature_set : str
        Which features to use:
        - ``'bilateral'``: bilateral-averaged region ROIs (~50 features)
        - ``'territory'``: territory aggregate ROIs (~15 features)
        - ``'all'``: all individual L/R ROIs (~234 features)
    cohort_filter : str, optional
        Restrict to a single cohort (e.g. ``'p30'``). If None, pool all
        known cohorts (p30, p60, p90).
    exclusion_csv : Path, optional
        Path to exclusion CSV (subject, session columns).
    standardize : bool
        Whether to z-score features (recommended for SVM/LDA).
    target : str
        Target variable for group labels:
        - ``'dose'``: built-in C/L/M/H → 0/1/2/3
        - any other string: look up that column in the DataFrame.
          String values are used as labels directly; numeric values
          with ≤20 unique values are cast to int labels.

    Returns
    -------
    dict with keys:
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,) — integer group labels
        label_names : list[str] — ordered group names
        feature_names : list[str]
        sample_info : DataFrame — subject, session, dose, cohort per row
        scaler : StandardScaler or None
    """
    loaded = _load_filter_select(
        wide_csv, feature_set, cohort_filter, exclusion_csv, standardize,
    )
    df = loaded["df"]
    X = loaded["X"]
    feature_names = loaded["feature_names"]

    if target == "dose":
        # Built-in: encode dose as integer labels
        dose_order = ["C", "L", "M", "H"]
        dose_col = df["dose"].values
        label_map = {d: i for i, d in enumerate(dose_order)}
        valid_mask = np.array([d in label_map for d in dose_col])
        if not valid_mask.all():
            n_dropped = (~valid_mask).sum()
            logger.warning("Dropping %d samples with unrecognised dose labels", n_dropped)
            X = X[valid_mask]
            df = df[valid_mask].copy()
            dose_col = dose_col[valid_mask]

        y = np.array([label_map[d] for d in dose_col], dtype=int)
        label_names = [d for d in dose_order if d in set(dose_col)]
        if len(label_names) < len(dose_order):
            new_map = {d: i for i, d in enumerate(label_names)}
            y = np.array([new_map[d] for d in dose_col], dtype=int)
    else:
        # Generic column lookup
        if target not in df.columns:
            info_cols = {"subject", "session", "dose", "cohort", "sex"}
            available = [c for c in df.columns if c not in info_cols and c not in feature_names]
            raise ValueError(
                f"Target column {target!r} not found. Available: {available}"
            )
        raw = df[target].values

        # Drop NaN rows
        if hasattr(raw[0], '__float__') or pd.api.types.is_numeric_dtype(df[target]):
            raw = raw.astype(float)
            valid = ~np.isnan(raw)
            if not valid.all():
                logger.warning("Dropping %d samples with NaN %s", (~valid).sum(), target)
                X, df, raw = X[valid], df[valid].copy(), raw[valid]

        # Determine if string or numeric
        try:
            float_vals = raw.astype(float)
            n_unique = len(np.unique(float_vals))
            if n_unique > 20:
                raise ValueError(
                    f"Target {target!r} has {n_unique} unique numeric values. "
                    f"Classification requires discrete groups (≤20 unique values). "
                    f"Use regression for continuous targets."
                )
            if n_unique > 1:
                logger.info(
                    "Target %s: %d unique numeric values, treating as discrete groups",
                    target, n_unique,
                )
            # Map unique sorted values to contiguous integers
            unique_sorted = sorted(np.unique(float_vals))
            val_to_int = {v: i for i, v in enumerate(unique_sorted)}
            y = np.array([val_to_int[v] for v in float_vals], dtype=int)
            label_names = [str(v) for v in unique_sorted]
        except (ValueError, TypeError):
            # String labels
            unique_sorted = sorted(set(raw))
            val_to_int = {v: i for i, v in enumerate(unique_sorted)}
            y = np.array([val_to_int[v] for v in raw], dtype=int)
            label_names = list(unique_sorted)

    # Include all non-feature columns as metadata (subject, session, dose,
    # cohort, sex, auc, etc.) so downstream consumers can access them.
    feature_set_cols = set(feature_names)
    info_cols = [c for c in df.columns if c not in feature_set_cols]
    sample_info = df[info_cols].reset_index(drop=True)

    logger.info(
        "Classification data: n=%d, features=%d, groups=%s",
        X.shape[0], X.shape[1],
        {name: int((y == i).sum()) for i, name in enumerate(label_names)},
    )

    return {
        "X": X,
        "y": y,
        "label_names": label_names,
        "feature_names": feature_names,
        "sample_info": sample_info,
        "scaler": loaded["scaler"],
    }


def prepare_regression_data(
    wide_csv: Union[str, Path],
    feature_set: str = "bilateral",
    cohort_filter: Optional[str] = None,
    exclusion_csv: Optional[Union[str, Path]] = None,
    standardize: bool = True,
    target: str = "dose",
) -> dict:
    """Load, filter, and scale ROI data for regression.

    Parameters
    ----------
    wide_csv : Path
        Path to wide-format ROI CSV (from extract_roi_means.py).
    feature_set : str
        Which features to use:
        - ``'bilateral'``: bilateral-averaged region ROIs (~50 features)
        - ``'territory'``: territory aggregate ROIs (~15 features)
        - ``'all'``: all individual L/R ROIs (~234 features)
    cohort_filter : str, optional
        Restrict to a single cohort (e.g. ``'p30'``). If None, pool all
        known cohorts (p30, p60, p90).
    exclusion_csv : Path, optional
        Path to exclusion CSV (subject, session columns).
    standardize : bool
        Whether to z-score features (recommended for SVR/Ridge).
    target : str
        Target variable:
        - ``'dose'``: ordinal encoding (C=0, L=1, M=2, H=3)
        - any other string: look up that column in the DataFrame

    Returns
    -------
    dict with keys:
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,) — float target values
        feature_names : list[str]
        dose_labels : ndarray, shape (n_samples,) — int group per sample for coloring
        label_names : list[str] — dose group names (C, L, M, H)
        target_name : str — e.g. "AUC" or "Dose (ordinal)"
        sample_info : DataFrame — subject, session, dose, cohort per row
        scaler : StandardScaler or None
    """
    loaded = _load_filter_select(
        wide_csv, feature_set, cohort_filter, exclusion_csv, standardize,
    )
    df = loaded["df"]
    X = loaded["X"]
    feature_names = loaded["feature_names"]

    # Filter to valid dose labels
    dose_order = ["C", "L", "M", "H"]
    label_map = {d: i for i, d in enumerate(dose_order)}
    dose_col = df["dose"].values
    valid_mask = np.array([d in label_map for d in dose_col])
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        logger.warning("Dropping %d samples with unrecognised dose labels", n_dropped)
        X = X[valid_mask]
        df = df[valid_mask].copy()
        dose_col = dose_col[valid_mask]

    # Integer dose labels (always needed for plot coloring)
    label_names = [d for d in dose_order if d in set(dose_col)]
    contiguous_map = {d: i for i, d in enumerate(label_names)}
    dose_labels = np.array([contiguous_map[d] for d in dose_col], dtype=int)

    if target == "dose":
        y = dose_labels.astype(float)
        target_name = "Dose (ordinal)"
    else:
        # Generic column lookup
        if target not in df.columns:
            info_cols = {"subject", "session", "dose", "cohort", "sex"}
            available = [c for c in df.columns if c not in info_cols and c not in feature_names]
            raise ValueError(
                f"Target column {target!r} not found. Available: {available}"
            )
        y = df[target].values.astype(float)
        valid = ~np.isnan(y)
        if not valid.all():
            logger.warning("Dropping %d samples with NaN %s", (~valid).sum(), target)
            X, df, dose_labels, y = X[valid], df[valid].copy(), dose_labels[valid], y[valid]
        target_name = target

    feature_set_cols = set(feature_names)
    info_cols = [c for c in df.columns if c not in feature_set_cols]
    sample_info = df[info_cols].reset_index(drop=True)

    logger.info(
        "Regression data: n=%d, features=%d, target=%s, y range=[%.2f, %.2f]",
        X.shape[0], X.shape[1], target_name, y.min(), y.max(),
    )

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "dose_labels": dose_labels,
        "label_names": label_names,
        "target_name": target_name,
        "sample_info": sample_info,
        "scaler": loaded["scaler"],
    }
