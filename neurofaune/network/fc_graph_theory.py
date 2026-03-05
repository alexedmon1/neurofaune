"""
Per-subject functional connectivity graph theory.

Computes graph-theoretic metrics on per-subject FC matrices across a density
sweep, then tests group differences via permutation on per-subject AUC values.

Unlike CovNet graph theory (which builds one covariance matrix per group),
this operates on pre-computed per-subject FC matrices — permutation only
shuffles group labels, making it orders of magnitude faster.
"""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from neurofaune.network.graph_theory import (
    DEFAULT_DENSITIES,
    METRIC_REGISTRY,
    compute_all_metrics,
    compute_metric_curve,
    list_metrics,
    _auc_trapz,
)

logger = logging.getLogger(__name__)


def load_fc_data(
    connectome_dir: Path,
    exclusion_csv: Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Load per-subject FC matrices and metadata, optionally excluding subjects.

    Parameters
    ----------
    connectome_dir : Path
        Directory containing fc_matrices.npy, fc_subjects.csv,
        fc_roi_labels.csv.
    exclusion_csv : Path, optional
        CSV with subject, session columns listing sessions to exclude.

    Returns
    -------
    fc_matrices : ndarray (n_subjects, n_rois, n_rois)
    subjects_df : DataFrame with subject, session, dose, sex columns
    roi_labels : list of str
    """
    fc_matrices = np.load(connectome_dir / "fc_matrices.npy")
    subjects_df = pd.read_csv(connectome_dir / "fc_subjects.csv", index_col=0)
    roi_labels_df = pd.read_csv(connectome_dir / "fc_roi_labels.csv", index_col=0)
    roi_labels = [str(l) for l in roi_labels_df["label"].tolist()]
    logger.info(
        "Loaded FC data: %d subjects, %d ROIs",
        fc_matrices.shape[0], fc_matrices.shape[1],
    )

    if exclusion_csv is not None:
        excl = pd.read_csv(exclusion_csv)
        excl_keys = set(zip(excl["subject"], excl["session"]))
        mask = np.array([
            (row["subject"], row["session"]) not in excl_keys
            for _, row in subjects_df.iterrows()
        ])
        n_excluded = (~mask).sum()
        fc_matrices = fc_matrices[mask]
        subjects_df = subjects_df[mask].reset_index(drop=True)
        logger.info("Excluded %d subjects via %s, %d remaining",
                     n_excluded, exclusion_csv, len(subjects_df))

    return fc_matrices, subjects_df, roi_labels


def compute_subject_aucs(
    fc_matrices: np.ndarray,
    graph_metrics: list[str] | None = None,
    densities: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute graph metric AUCs for each subject's FC matrix.

    Parameters
    ----------
    fc_matrices : ndarray (n_subjects, n_rois, n_rois)
    graph_metrics : list of str, optional
        Metrics to compute. Default: all registered metrics.
    densities : list of float, optional
        Density sweep levels. Default: DEFAULT_DENSITIES.

    Returns
    -------
    auc_df : DataFrame (n_subjects, n_metrics)
        AUC values per subject per metric.
    curves : dict[str, ndarray]
        Per-subject density curves: metric_name -> (n_subjects, n_densities).
    """
    if densities is None:
        densities = DEFAULT_DENSITIES
    if graph_metrics is None:
        graph_metrics = list_metrics()

    n_subjects = fc_matrices.shape[0]
    n_densities = len(densities)

    # Preallocate curves
    curves = {m: np.full((n_subjects, n_densities), np.nan) for m in graph_metrics}
    auc_values = {m: np.full(n_subjects, np.nan) for m in graph_metrics}

    for i in range(n_subjects):
        fc = fc_matrices[i]

        # Skip subjects with too many NaNs
        finite_frac = np.isfinite(fc).mean()
        if finite_frac < 0.1:
            logger.warning("Subject %d: only %.0f%% finite values, skipping", i, finite_frac * 100)
            continue

        # Replace NaN with 0 and enforce symmetry for graph construction
        fc_clean = np.nan_to_num(fc, nan=0.0)
        fc_clean = (fc_clean + fc_clean.T) / 2
        np.fill_diagonal(fc_clean, 0)

        for metric_name in graph_metrics:
            curve = compute_metric_curve(fc_clean, metric_name, densities)
            curves[metric_name][i] = curve
            auc_values[metric_name][i] = _auc_trapz(curve, densities)

        if (i + 1) % 20 == 0 or i == 0:
            logger.info("  Computed graph metrics for %d/%d subjects", i + 1, n_subjects)

    logger.info("Computed graph metrics for all %d subjects", n_subjects)
    auc_df = pd.DataFrame(auc_values)
    return auc_df, curves


def permutation_test_groups(
    auc_df: pd.DataFrame,
    group_labels: np.ndarray,
    group_names: list[str],
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test pairwise group differences in per-subject AUC via permutation.

    For each metric and pair of groups, tests whether the difference in
    mean AUC is significant by permuting group labels.

    Parameters
    ----------
    auc_df : DataFrame (n_subjects, n_metrics)
    group_labels : ndarray of int (n_subjects,)
        Integer group label per subject.
    group_names : list of str
        Name for each integer label.
    n_permutations : int
    seed : int

    Returns
    -------
    results_df : DataFrame with columns:
        group_a, group_b, metric, mean_a, mean_b, diff, p_value
    """
    rng = np.random.default_rng(seed)
    metrics = auc_df.columns.tolist()
    unique_groups = sorted(set(group_labels))
    pairs = list(combinations(unique_groups, 2))

    rows = []
    for metric in metrics:
        values = auc_df[metric].values

        for g_a, g_b in pairs:
            mask_a = group_labels == g_a
            mask_b = group_labels == g_b

            vals_a = values[mask_a]
            vals_b = values[mask_b]

            # Drop NaN subjects
            valid_a = np.isfinite(vals_a)
            valid_b = np.isfinite(vals_b)
            vals_a = vals_a[valid_a]
            vals_b = vals_b[valid_b]

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            obs_diff = vals_a.mean() - vals_b.mean()

            # Permutation test
            pooled = np.concatenate([vals_a, vals_b])
            n_a = len(vals_a)
            null_diffs = np.empty(n_permutations)
            for p in range(n_permutations):
                idx = rng.permutation(len(pooled))
                null_diffs[p] = pooled[idx[:n_a]].mean() - pooled[idx[n_a:]].mean()

            p_val = float((np.sum(np.abs(null_diffs) >= abs(obs_diff)) + 1) / (n_permutations + 1))

            name_a = group_names[g_a] if g_a < len(group_names) else str(g_a)
            name_b = group_names[g_b] if g_b < len(group_names) else str(g_b)

            rows.append({
                "group_a": name_a,
                "group_b": name_b,
                "metric": metric,
                "mean_a": float(vals_a.mean()),
                "mean_b": float(vals_b.mean()),
                "diff": float(obs_diff),
                "p_value": p_val,
                "n_a": len(vals_a),
                "n_b": len(vals_b),
            })

        logger.info("  Tested %s: %d pairs", metric, len(pairs))

    return pd.DataFrame(rows)


def build_groups(
    subjects_df: pd.DataFrame,
    cohort: str | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build dose group labels, optionally filtering by cohort.

    Returns
    -------
    group_labels : ndarray of int
    group_names : list of str (e.g. ['C', 'L', 'M', 'H'])
    mask : boolean ndarray (True = included)
    """
    dose_order = ["C", "L", "M", "H"]
    dose_map = {d: i for i, d in enumerate(dose_order)}

    if cohort is not None:
        session = f"ses-{cohort}"
        mask = (subjects_df["session"] == session).values
    else:
        mask = np.ones(len(subjects_df), dtype=bool)

    df_sub = subjects_df[mask]
    labels = np.array([dose_map.get(d, -1) for d in df_sub["dose"]])

    # Drop unknown doses
    valid = labels >= 0
    mask_indices = np.where(mask)[0]
    mask[:] = False
    mask[mask_indices[valid]] = True
    labels = labels[valid]

    present = sorted(set(labels))
    group_names = [dose_order[i] for i in present]

    return labels, group_names, mask
