"""
Spearman correlation matrix computation for covariance network analysis.

Loads ROI-level DTI metrics, applies exclusions, optionally averages bilateral
ROIs, splits data by experimental groups, and computes inter-regional Spearman
correlation matrices.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def load_and_prepare_data(
    wide_csv: Union[str, Path],
    exclusion_csv: Optional[Union[str, Path]] = None,
    max_zero_frac: float = 0.2,
) -> tuple[pd.DataFrame, list[str]]:
    """Load ROI wide CSV, apply exclusions, and filter unreliable ROIs.

    Parameters
    ----------
    wide_csv : Path
        Path to wide-format ROI CSV (from extract_roi_means.py).
        Expected columns: subject, session, <ROI columns>, dose, sex.
    exclusion_csv : Path, optional
        Path to CSV listing sessions to exclude (must have subject, session
        columns). Typically dti_nonstandard_slices.csv.
    max_zero_frac : float
        Maximum fraction of zero values allowed per ROI across subjects.
        ROIs exceeding this are dropped. Default 0.2.

    Returns
    -------
    df : DataFrame
        Filtered DataFrame with metadata columns (subject, session, dose, sex,
        cohort) and valid ROI columns.
    roi_cols : list[str]
        Names of the retained ROI columns.
    """
    df = pd.read_csv(wide_csv)
    n_start = len(df)
    logger.info(f"Loaded {n_start} subjects from {wide_csv}")

    # Derive cohort from session (ses-p60 -> p60)
    if "cohort" not in df.columns:
        df["cohort"] = df["session"].str.extract(r"ses-(\w+)")[0]

    # Apply exclusions
    if exclusion_csv is not None:
        excl = pd.read_csv(exclusion_csv)
        excl_keys = set(zip(excl["subject"], excl["session"]))
        mask = df.apply(lambda r: (r["subject"], r["session"]) not in excl_keys, axis=1)
        df = df[mask].reset_index(drop=True)
        n_excluded = n_start - len(df)
        logger.info(f"Excluded {n_excluded} sessions ({len(df)} remaining)")

    # Identify ROI columns (everything except metadata)
    meta_cols = {"subject", "session", "dose", "sex", "cohort"}
    all_roi_cols = [c for c in df.columns if c not in meta_cols]

    # Separate region ROIs from territory ROIs
    region_cols = [c for c in all_roi_cols if not c.startswith("territory_")]
    territory_cols = [c for c in all_roi_cols if c.startswith("territory_")]

    # Filter region ROIs: drop those with all NaN or >max_zero_frac zeros
    valid_region_cols = []
    dropped = []
    for col in region_cols:
        vals = df[col]
        if vals.isna().all():
            dropped.append((col, "all_nan"))
            continue
        non_na = vals.dropna()
        if len(non_na) == 0:
            dropped.append((col, "all_nan"))
            continue
        zero_frac = (non_na == 0).sum() / len(non_na)
        if zero_frac > max_zero_frac:
            dropped.append((col, f"zero_frac={zero_frac:.2f}"))
            continue
        valid_region_cols.append(col)

    if dropped:
        logger.info(
            f"Dropped {len(dropped)} region ROIs (of {len(region_cols)}): "
            f"{', '.join(f'{c}({r})' for c, r in dropped[:5])}"
            + (f" ... and {len(dropped) - 5} more" if len(dropped) > 5 else "")
        )

    # Filter territory ROIs with same criteria
    valid_territory_cols = []
    for col in territory_cols:
        vals = df[col]
        if vals.isna().all():
            continue
        non_na = vals.dropna()
        if len(non_na) == 0:
            continue
        zero_frac = (non_na == 0).sum() / len(non_na)
        if zero_frac > max_zero_frac:
            continue
        valid_territory_cols.append(col)

    roi_cols = valid_region_cols + valid_territory_cols
    logger.info(
        f"Retained {len(valid_region_cols)} region ROIs + "
        f"{len(valid_territory_cols)} territory ROIs = {len(roi_cols)} total"
    )

    return df, roi_cols


def bilateral_average(
    df: pd.DataFrame, roi_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Average left/right ROI pairs into bilateral ROIs.

    For each ``_L``/``_R`` pair, computes the mean (ignoring NaN if one side
    is missing). ROIs without a matching partner are kept as-is.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing ROI columns.
    roi_cols : list[str]
        ROI column names to process.

    Returns
    -------
    df_bilateral : DataFrame
        New DataFrame with bilateral columns replacing L/R pairs.
        Metadata columns are preserved.
    bilateral_cols : list[str]
        Updated column names after bilateral averaging.
    """
    # Separate region vs territory columns
    region_cols = [c for c in roi_cols if not c.startswith("territory_")]
    territory_cols = [c for c in roi_cols if c.startswith("territory_")]

    # Find L/R pairs among region columns
    left_rois = {c for c in region_cols if c.endswith("_L")}
    right_rois = {c for c in region_cols if c.endswith("_R")}

    paired = {}
    unpaired = []
    for col in region_cols:
        if col.endswith("_L"):
            base = col[:-2]  # strip _L
            partner = base + "_R"
            if partner in right_rois:
                paired[base] = (col, partner)
            else:
                unpaired.append(col)
        elif col.endswith("_R"):
            base = col[:-2]
            partner = base + "_L"
            if partner not in left_rois:
                unpaired.append(col)
            # else: already handled by the _L branch
        else:
            unpaired.append(col)

    # Build new DataFrame with bilateral averages using pd.concat to avoid fragmentation
    meta_cols = [c for c in df.columns if c not in set(roi_cols)]
    parts = [df[meta_cols].copy()]

    bilateral_region_cols = []
    for base, (left, right) in sorted(paired.items()):
        parts.append(df[[left, right]].mean(axis=1, skipna=True).rename(base))
        bilateral_region_cols.append(base)

    for col in unpaired:
        parts.append(df[col])
        bilateral_region_cols.append(col)

    # Keep territory columns as-is
    for col in territory_cols:
        parts.append(df[col])

    df_bilateral = pd.concat(parts, axis=1)
    bilateral_cols = bilateral_region_cols + territory_cols
    logger.info(
        f"Bilateral averaging: {len(region_cols)} region ROIs -> "
        f"{len(bilateral_region_cols)} ({len(paired)} pairs + {len(unpaired)} unpaired)"
    )
    return df_bilateral, bilateral_cols


def define_groups(
    df: pd.DataFrame, grouping: str = "pnd_dose"
) -> dict[str, pd.DataFrame]:
    """Split DataFrame into experimental groups.

    Parameters
    ----------
    df : DataFrame
        Must contain 'cohort', 'dose', and 'sex' columns.
    grouping : str
        Grouping strategy:
        - ``'full'``: sex x PND x dose (24 groups) — descriptive only
        - ``'pnd_dose'``: PND x dose (12 groups) — primary statistical
        - ``'dose'``: dose only (4 groups) — maximum power

    Returns
    -------
    groups : dict[str, DataFrame]
        Mapping from group label to subset DataFrame.
        Labels use format like "p60_control", "p60_low_M", etc.
    """
    # Exclude unknown-cohort sessions
    df = df[df["cohort"].isin(["p30", "p60", "p90"])].copy()

    groups = {}
    if grouping == "full":
        for (cohort, dose, sex), subset in df.groupby(["cohort", "dose", "sex"]):
            label = f"{cohort}_{dose}_{sex}"
            groups[label] = subset.reset_index(drop=True)
    elif grouping == "pnd_dose":
        for (cohort, dose), subset in df.groupby(["cohort", "dose"]):
            label = f"{cohort}_{dose}"
            groups[label] = subset.reset_index(drop=True)
    elif grouping == "dose":
        for dose, subset in df.groupby("dose"):
            label = str(dose)
            groups[label] = subset.reset_index(drop=True)
    else:
        raise ValueError(f"Unknown grouping: {grouping!r}. Use 'full', 'pnd_dose', or 'dose'.")

    for label, subset in sorted(groups.items()):
        logger.info(f"  Group {label}: n={len(subset)}")

    return groups


def compute_spearman_matrices(
    groups: dict[str, pd.DataFrame], roi_cols: list[str]
) -> dict[str, dict]:
    """Compute Spearman correlation matrices for each group.

    Parameters
    ----------
    groups : dict[str, DataFrame]
        From ``define_groups()``.
    roi_cols : list[str]
        ROI columns to correlate.

    Returns
    -------
    results : dict[str, dict]
        Per group: ``{'corr': ndarray, 'pval': ndarray, 'n': int, 'rois': list}``.
        Matrices are shape ``(n_rois, n_rois)``. Diagonal of corr is 1.0,
        diagonal of pval is 0.0.
    """
    results = {}
    for label, subset in groups.items():
        data = subset[roi_cols].values  # (n_subjects, n_rois)
        n_subjects, n_rois = data.shape

        corr = np.eye(n_rois)
        pval = np.zeros((n_rois, n_rois))

        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                x = data[:, i]
                y = data[:, j]
                # Pairwise complete observations
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() < 4:
                    corr[i, j] = corr[j, i] = np.nan
                    pval[i, j] = pval[j, i] = np.nan
                    continue
                r, p = stats.spearmanr(x[valid], y[valid])
                corr[i, j] = corr[j, i] = r
                pval[i, j] = pval[j, i] = p

        results[label] = {
            "corr": corr,
            "pval": pval,
            "n": n_subjects,
            "rois": list(roi_cols),
        }
        logger.info(f"  {label}: {n_rois}x{n_rois} matrix from n={n_subjects}")

    return results


def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """Fisher z-transform correlation coefficients.

    Applies ``arctanh(r)`` with clipping to avoid infinity at |r| = 1.

    Parameters
    ----------
    r : ndarray
        Correlation coefficients.

    Returns
    -------
    z : ndarray
        Fisher z-transformed values.
    """
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r_clipped)
