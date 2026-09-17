"""
Spearman correlation matrix computation for covariance network analysis.

Loads ROI-level DTI metrics, applies exclusions, optionally averages bilateral
ROIs, splits data by experimental groups, and computes inter-regional Spearman
correlation matrices.
"""

import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


#: Columns this loader creates or requires structurally, in any study.
#: ``total_brain_mm3`` travels with the structural-covariance input as the
#: normalisation denominator. It is not a node -- correlating a global total
#: against the parts it normalised would put the confound straight back into the
#: network -- so it is excluded here rather than left to each caller's meta_cols.
STRUCTURAL_COLS = frozenset({"subject", "session", "cohort", "total_brain_mm3"})


def load_and_prepare_data(
    wide_csv: Union[str, Path],
    exclusion_csv: Optional[Union[str, Path]] = None,
    max_zero_frac: float = 0.2,
    max_subject_zero_frac: float = 0.10,
    meta_cols: Optional[Iterable[str]] = None,
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
    max_subject_zero_frac : float
        Maximum fraction of ROIs allowed to be zero for a single subject.
        Subjects exceeding this are dropped (likely FOV coverage issues).
    meta_cols : iterable of str, optional
        Columns that are metadata, not ROI measurements -- the study's design
        columns (``group``, ``timepoint``, ``dose``, ``sex``, ...) and any
        numeric covariate (``age_days``, ``weight_g``, ...). ``subject``,
        ``session`` and ``cohort`` are always excluded and need not be listed.
        Non-numeric columns that are neither structural nor declared raise,
        since they cannot be measurements; numeric ones cannot be detected and
        MUST be declared or they are treated as ROIs.
        Default 0.10.

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

    # Identify ROI columns by EXCLUSION, which means anything undeclared is
    # treated as a brain region. That used to be a hardcoded set --
    # {subject, session, dose, sex, cohort} -- so a study whose design columns
    # were named anything else had them silently Spearman-correlated against
    # real ROIs as though they were regions. Declared metadata is now the
    # caller's to state, and anything left over that cannot be an ROI is an
    # error rather than a silent corruption.
    declared = set(STRUCTURAL_COLS) | set(meta_cols or ())
    declared.update(c for c in df.columns
                    if c.startswith("AUC_") or c in ("auc", "log_auc"))
    candidates = [c for c in df.columns if c not in declared]

    # An ROI column holds a measurement, so it must be numeric. A non-numeric
    # leftover is a design/metadata column the caller forgot to declare -- name
    # it rather than correlating a string column against the brain.
    non_numeric = [c for c in candidates
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"load_and_prepare_data: column(s) {sorted(non_numeric)} are not "
            f"numeric, so they cannot be ROI measurements, and they were not "
            f"declared as metadata. Pass meta_cols={sorted(non_numeric)} (plus "
            f"any numeric metadata such as age or weight, which cannot be "
            f"detected automatically). Structural columns "
            f"{sorted(STRUCTURAL_COLS)} are always excluded."
        )
    all_roi_cols = candidates

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

    # Filter subjects with excessive zeros among VALID ROIs only.
    # This runs after ROI filtering so FOV-limited ROIs (zero for everyone)
    # don't count against individual subjects.
    if max_subject_zero_frac > 0 and valid_region_cols:
        roi_data = df[valid_region_cols].values
        zeros_per_subj = (roi_data == 0).sum(axis=1)
        max_zeros = int(len(valid_region_cols) * max_subject_zero_frac)
        bad_mask = zeros_per_subj > max_zeros
        if bad_mask.any():
            bad_indices = np.where(bad_mask)[0]
            for idx in bad_indices:
                row = df.iloc[idx]
                n_z = int(zeros_per_subj[idx])
                logger.warning(
                    f"Dropping {row['subject']}/{row['session']}: "
                    f"{n_z}/{len(valid_region_cols)} valid ROIs are zero "
                    f"(>{max_subject_zero_frac:.0%} threshold)"
                )
            df = df[~bad_mask].reset_index(drop=True)
            logger.info(
                f"Dropped {bad_mask.sum()} subjects with >{max_subject_zero_frac:.0%} "
                f"zero valid ROIs ({len(df)} remaining)"
            )

    # Replace remaining zeros with NaN in region ROIs.
    # In rodent MRI with limited slice packages, zero values in brain ROIs
    # are almost always out-of-FOV artifacts, not real measurements.
    # Leaving them as zeros biases correlations and regressions.
    if valid_region_cols:
        roi_data = df[valid_region_cols]
        n_zeros = (roi_data == 0).sum().sum()
        if n_zeros > 0:
            df[valid_region_cols] = roi_data.replace(0, np.nan)
            n_cells = roi_data.shape[0] * roi_data.shape[1]
            logger.info(
                f"Replaced {n_zeros} residual zeros with NaN in region ROIs "
                f"({n_zeros / n_cells:.1%} of values)"
            )

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


#: Dose tokens treated as the reference/control level, matched case-insensitively.
CONTROL_TOKENS = frozenset(
    {"c", "control", "ctrl", "ctl", "vehicle", "veh", "sham", "untreated", "naive", "0"}
)

#: Preferred ordering for common dose tokens; anything else sorts after, by name.
DOSE_ORDER = ("l", "low", "m", "med", "medium", "h", "high")


def _split_group_label(
    label: str, cohorts: Sequence[str] | None = None
) -> tuple[str, str]:
    """Split ``"{cohort}_{dose}"`` back into its parts.

    When the cohort values are known they are matched as a prefix (longest first),
    which is exact even when a cohort or a dose name itself contains an underscore.
    Without them the split falls back to the last underscore.
    """
    if cohorts:
        for cohort in sorted(cohorts, key=len, reverse=True):
            prefix = f"{cohort}_"
            if label.startswith(prefix):
                return cohort, label[len(prefix):]
    cohort, _, dose = label.rpartition("_")
    return cohort, dose


def parse_group_labels(
    group_labels: Sequence[str], cohorts: Sequence[str] | None = None
) -> dict[str, tuple[str, str]]:
    """Map each group label to its ``(cohort, dose)`` pair."""
    return {label: _split_group_label(label, cohorts) for label in group_labels}


def cohorts_from_labels(
    group_labels: Sequence[str],
    cohorts: Sequence[str] | None = None,
    cohort_order: Sequence[str] | None = None,
) -> list[str]:
    """Cohorts present in `group_labels`, in comparison order.

    Default order is the sorted cohort names, which gives p30 < p60 < p90 for the
    PND convention. Pass `cohort_order` when the sorted order is not the temporal
    order (e.g. ``["baseline", "week6", "week12"]``).
    """
    observed = {c for c, _ in parse_group_labels(group_labels, cohorts).values()}
    if cohort_order is not None:
        ordered = [c for c in cohort_order if c in observed]
        missing = observed - set(ordered)
        if missing:
            logger.warning(
                f"cohort_order omits {sorted(missing)}; appending them in sorted order"
            )
            ordered += sorted(missing)
        return ordered
    return sorted(observed)


def dose_levels(
    group_labels: Sequence[str], cohorts: Sequence[str] | None = None
) -> tuple[str | None, list[str]]:
    """Split the observed dose levels into ``(control, treatments)``.

    The control level is whichever dose matches :data:`CONTROL_TOKENS`; everything
    else is a treatment. This replaces the previous hardcoded ``C/L/M/H`` and
    ``control/low/medium/high`` lists, which silently produced no comparisons for
    any study using different level names.
    """
    doses = {d for _, d in parse_group_labels(group_labels, cohorts).values()}
    control = next((d for d in sorted(doses) if str(d).lower() in CONTROL_TOKENS), None)
    treatments = sorted(
        doses - ({control} if control is not None else set()),
        key=lambda d: (
            DOSE_ORDER.index(str(d).lower()) if str(d).lower() in DOSE_ORDER
            else len(DOSE_ORDER),
            str(d),
        ),
    )
    return control, treatments


def define_groups(
    df: pd.DataFrame,
    factors: list[str],
    include: dict[str, list] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into experimental groups.

    Study-agnostic by construction: ``factors`` names the columns that define a
    cell. Any number, any order, no required column names, no built-in filtering.

        define_groups(df, factors=["timepoint", "group"])
        define_groups(df, factors=["cohort", "dose", "sex"])
        define_groups(df, factors=["dose"], include={"cohort": ["p60"]})

    Parameters
    ----------
    df : DataFrame
        Session-level table containing at least the ``factors`` columns.
    factors : list of str
        Columns to group by, in label order. Labels join the values with ``_``.
    include : dict, optional
        Per-column whitelist, e.g. ``{"cohort": ["p60", "p90"]}``. Rows whose
        value is not listed are dropped. Filtering is never implicit -- if you
        want rows excluded, say so here.

    Returns
    -------
    groups : dict[str, DataFrame]
        Label -> subset, index reset.

    Raises
    ------
    ValueError
        If ``factors`` is empty, or names a column that is not present.

    Notes
    -----
    This function previously took ``grouping="full"|"pnd_dose"|"dose"``, which
    required ``cohort``/``dose``/``sex`` columns and silently dropped any row
    whose cohort was not ``p30``/``p60``/``p90``. That encoded one study's design
    into shared code and made the covnet pipeline unusable for every other study.
    Callers migrate by naming the columns explicitly:

        grouping="full"      ->  factors=["cohort", "dose", "sex"]
        grouping="pnd_dose"  ->  factors=["cohort", "dose"]
        grouping="dose"      ->  factors=["dose"]

    and adding ``include={"cohort": ["p30", "p60", "p90"]}`` if the old cohort
    whitelist was actually wanted rather than merely inherited.
    """
    if not factors:
        raise ValueError(
            "define_groups requires factors=[...] naming the columns that "
            "define a cell, e.g. factors=['timepoint', 'group']."
        )
    missing = [c for c in factors if c not in df.columns]
    if missing:
        raise ValueError(
            f"define_groups: column(s) {missing} not in the DataFrame. "
            f"Available: {sorted(df.columns)[:12]}..."
        )

    df = df.copy()
    if include:
        for col, allowed in include.items():
            if col not in df.columns:
                raise ValueError(f"include: column {col!r} not in the DataFrame")
            df = df[df[col].isin(list(allowed))].copy()

    groups = {}
    for key, subset in df.groupby(factors, dropna=True):
        if not isinstance(key, tuple):
            key = (key,)
        groups["_".join(str(k) for k in key)] = subset.reset_index(drop=True)

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


def spearman_matrix(data: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation matrix from (n_subjects, n_rois) data.

    Uses a fast vectorized path when there are no NaN values (rank-transform
    then Pearson via numpy). Falls back to pairwise complete observations
    when NaNs are present.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_rois)
        ROI values per subject.

    Returns
    -------
    corr : ndarray, shape (n_rois, n_rois)
        Symmetric Spearman correlation matrix with 1s on the diagonal.
    """
    if not np.any(np.isnan(data)) and data.shape[0] >= 4:
        # Fast path: rank each column, then Pearson corrcoef
        ranked = stats.rankdata(data, axis=0)
        corr = np.corrcoef(ranked, rowvar=False)
        # Replace NaN (from constant columns) with 0.0
        np.nan_to_num(corr, copy=False, nan=0.0)
        # Ensure exact 1s on diagonal (floating point)
        np.fill_diagonal(corr, 1.0)
        return corr

    # Slow path: pairwise complete observations
    n_rois = data.shape[1]
    corr = np.eye(n_rois)
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            x = data[:, i]
            y = data[:, j]
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 4:
                corr[i, j] = corr[j, i] = 0.0
                continue
            r, _ = stats.spearmanr(x[valid], y[valid])
            corr[i, j] = corr[j, i] = r
    return corr


def default_dose_comparisons(
    group_labels: list[str],
    cohorts: Optional[Sequence[str]] = None,
    cohort_order: Optional[Sequence[str]] = None,
) -> list[tuple[str, str]]:
    """Generate default comparisons: each dose vs control within each cohort.

    Cohort and dose levels are read off `group_labels` rather than assumed, so
    this works for any study naming scheme (``p30/p60/p90`` with ``C/L/M/H`` or
    ``control/low/medium/high`` as before, but also ``baseline/week6`` with
    ``sham/cuprizone``). The control level is whichever dose matches
    :data:`CONTROL_TOKENS`.

    Parameters
    ----------
    group_labels : list[str]
        Available group labels.
    cohorts : sequence of str, optional
        Known cohort values, used to split labels exactly. Without them the
        label is split at its last underscore.
    cohort_order : sequence of str, optional
        Explicit cohort ordering; defaults to sorted order.

    Returns
    -------
    comparisons : list of (str, str)
        Pairs of (treatment, control) group labels.
    """
    parsed = parse_group_labels(group_labels, cohorts)
    ordered_cohorts = cohorts_from_labels(group_labels, cohorts, cohort_order)
    control, treatments = dose_levels(group_labels, cohorts)

    if control is None:
        logger.warning(
            f"No control level found among doses "
            f"{sorted({d for _, d in parsed.values()})}; "
            f"recognised control tokens: {sorted(CONTROL_TOKENS)}"
        )
        return []

    by_key = {(c, d): label for label, (c, d) in parsed.items()}
    comparisons = []
    for cohort in ordered_cohorts:
        control_label = by_key.get((cohort, control))
        if control_label is None:
            continue
        for dose in treatments:
            treatment_label = by_key.get((cohort, dose))
            if treatment_label is not None:
                comparisons.append((treatment_label, control_label))

    if not comparisons:
        logger.warning(
            "No default comparisons matched group labels. "
            f"Available: {group_labels}"
        )
    return comparisons


def cross_timepoint_comparisons(
    group_labels: list[str],
    cohorts: Optional[Sequence[str]] = None,
    cohort_order: Optional[Sequence[str]] = None,
) -> list[tuple[str, str]]:
    """Generate cross-cohort comparisons within each dose level.

    For each dose (control included), produces all pairwise cohort comparisons
    in cohort order — p30 vs p60, p30 vs p90, p60 vs p90 for the PND convention.

    Parameters
    ----------
    group_labels : list[str]
        Available group labels (e.g. ["p30_C", "p30_L", ..., "p90_H"]).
    cohorts, cohort_order
        See :func:`default_dose_comparisons`.

    Returns
    -------
    comparisons : list of (str, str)
        Pairs of group labels to compare.
    """
    parsed = parse_group_labels(group_labels, cohorts)
    ordered_cohorts = cohorts_from_labels(group_labels, cohorts, cohort_order)
    control, treatments = dose_levels(group_labels, cohorts)
    doses = ([control] if control is not None else []) + treatments

    by_key = {(c, d): label for label, (c, d) in parsed.items()}
    comparisons = []
    for dose in doses:
        present = [by_key[(c, dose)] for c in ordered_cohorts if (c, dose) in by_key]
        for i, ga in enumerate(present):
            for gb in present[i + 1:]:
                comparisons.append((ga, gb))

    return comparisons


def cross_dose_timepoint_comparisons(
    group_labels: list[str],
    cohorts: Optional[Sequence[str]] = None,
    cohort_order: Optional[Sequence[str]] = None,
) -> list[tuple[str, str]]:
    """Generate cross-dose-cross-timepoint comparisons.

    Pairs each dosed group at an earlier cohort with controls at a later one.
    Tests whether exposed young animals resemble older controls (accelerated
    maturation hypothesis). "Earlier" and "later" follow the cohort order, which
    is sorted order unless `cohort_order` says otherwise — pass it explicitly
    whenever the timepoint names do not sort chronologically.

    Parameters
    ----------
    group_labels : list[str]
        Available group labels (e.g. ["p30_C", "p30_L", ..., "p90_H"]).
    cohorts, cohort_order
        See :func:`default_dose_comparisons`.

    Returns
    -------
    comparisons : list of (str, str)
        Pairs of (treatment, control) group labels. Treatment is a dosed
        group at an earlier cohort, control is a control group at a later one.
    """
    parsed = parse_group_labels(group_labels, cohorts)
    ordered_cohorts = cohorts_from_labels(group_labels, cohorts, cohort_order)
    control, treatments = dose_levels(group_labels, cohorts)

    if control is None:
        logger.warning("No control level found; cross-dose-timepoint needs one")
        return []

    by_key = {(c, d): label for label, (c, d) in parsed.items()}
    comparisons = []
    for i, early in enumerate(ordered_cohorts):
        for later in ordered_cohorts[i + 1:]:
            control_label = by_key.get((later, control))
            if control_label is None:
                continue
            for dose in treatments:
                treatment_label = by_key.get((early, dose))
                if treatment_label is not None:
                    comparisons.append((treatment_label, control_label))

    return comparisons


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
