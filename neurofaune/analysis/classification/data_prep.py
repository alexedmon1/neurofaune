"""
Data preparation for multivariate classification analysis.

Loads ROI-level wide CSVs (reusing covnet data loading), applies cohort
filtering, selects bilateral or territory feature sets, and standardises
features for downstream classifiers.
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


def prepare_classification_data(
    wide_csv: Union[str, Path],
    feature_set: str = "bilateral",
    cohort_filter: Optional[str] = None,
    exclusion_csv: Optional[Union[str, Path]] = None,
    standardize: bool = True,
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
    cohort_filter : str, optional
        Restrict to a single cohort (e.g. ``'p30'``). If None, pool all
        known cohorts (p30, p60, p90).
    exclusion_csv : Path, optional
        Path to exclusion CSV (subject, session columns).
    standardize : bool
        Whether to z-score features (recommended for SVM/LDA).

    Returns
    -------
    dict with keys:
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,) — integer dose labels
        label_names : list[str] — ordered dose group names
        feature_names : list[str]
        sample_info : DataFrame — subject, session, dose, cohort per row
        scaler : StandardScaler or None
    """
    # Load and prepare (reuse covnet pipeline)
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
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}. Use 'bilateral' or 'territory'.")

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

    # Encode dose as integer labels
    dose_order = ["C", "L", "M", "H"]
    dose_col = df["dose"].values
    # Map to integers (C=0, L=1, M=2, H=3)
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
    # Re-map so labels are contiguous 0..n-1
    if len(label_names) < len(dose_order):
        new_map = {d: i for i, d in enumerate(label_names)}
        y = np.array([new_map[d] for d in dose_col], dtype=int)

    # Standardize
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    sample_info = df[["subject", "session", "dose", "cohort"]].reset_index(drop=True)

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
        "scaler": scaler,
    }
