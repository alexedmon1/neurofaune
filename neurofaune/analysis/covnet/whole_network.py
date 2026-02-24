"""
Whole-network similarity tests for covariance network comparison.

Complements edge-level (NBS) and graph-metric approaches with holistic tests
that ask whether the overall covariance structure differs between groups.

Three statistics:
    1. Mantel test — Pearson correlation between vectorized upper triangles.
    2. Frobenius distance — L2 norm of the difference between upper triangles.
    3. Spectral divergence — L2 distance between sorted eigenvalue spectra.

All use a shared permutation framework: pool subjects, shuffle labels, split,
recompute group correlation matrices, recompute all three statistics, build
null distributions.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from neurofaune.analysis.covnet.matrices import default_dose_comparisons, spearman_matrix

logger = logging.getLogger(__name__)


def _upper_tri(mat: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle as a flat vector."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def mantel_test(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
    """Pearson correlation between vectorized upper triangles.

    Parameters
    ----------
    corr_a, corr_b : ndarray (n_rois, n_rois)
        Symmetric correlation matrices.

    Returns
    -------
    r : float
        Pearson r (near 1 = similar structure, near 0 = dissimilar).
    """
    va = _upper_tri(corr_a)
    vb = _upper_tri(corr_b)
    # Handle constant vectors (e.g., from very small groups)
    if np.std(va) == 0 or np.std(vb) == 0:
        return float("nan")
    r, _ = stats.pearsonr(va, vb)
    return float(r)


def frobenius_distance(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
    """L2 norm of the difference between upper triangles.

    Parameters
    ----------
    corr_a, corr_b : ndarray (n_rois, n_rois)
        Symmetric correlation matrices.

    Returns
    -------
    d : float
        Frobenius distance (larger = more different).
    """
    diff = _upper_tri(corr_a) - _upper_tri(corr_b)
    return float(np.linalg.norm(diff))


def spectral_divergence(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
    """L2 distance between sorted eigenvalue spectra.

    Captures changes in network topology (modularity, dimensionality) that
    may not show up edge-by-edge.

    Parameters
    ----------
    corr_a, corr_b : ndarray (n_rois, n_rois)
        Symmetric correlation matrices.

    Returns
    -------
    d : float
        Spectral divergence (larger = more different topology).
    """
    try:
        eig_a = np.sort(np.linalg.eigvalsh(corr_a))[::-1]
        eig_b = np.sort(np.linalg.eigvalsh(corr_b))[::-1]
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.linalg.norm(eig_a - eig_b))


def whole_network_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict:
    """Compute all three whole-network statistics with permutation p-values.

    Parameters
    ----------
    data_a : ndarray, shape (n_subjects_a, n_rois)
        ROI values for group A.
    data_b : ndarray, shape (n_subjects_b, n_rois)
        ROI values for group B.
    n_perm : int
        Number of permutations for null distribution.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Keys: mantel_r, mantel_p, frobenius_d, frobenius_p,
        spectral_d, spectral_p, null_distributions, n_a, n_b.
    """
    rng = np.random.default_rng(seed)
    n_a = data_a.shape[0]
    n_b = data_b.shape[0]
    pooled = np.vstack([data_a, data_b])
    n_total = n_a + n_b

    # Observed statistics
    corr_a = spearman_matrix(data_a)
    corr_b = spearman_matrix(data_b)

    obs_mantel = mantel_test(corr_a, corr_b)
    obs_frob = frobenius_distance(corr_a, corr_b)
    obs_spectral = spectral_divergence(corr_a, corr_b)

    logger.info(
        f"Observed: Mantel r={obs_mantel:.4f}, Frobenius d={obs_frob:.4f}, "
        f"Spectral d={obs_spectral:.4f}"
    )

    # Permutation null distributions
    null_mantel = np.zeros(n_perm)
    null_frob = np.zeros(n_perm)
    null_spectral = np.zeros(n_perm)

    for p in range(n_perm):
        perm_idx = rng.permutation(n_total)
        perm_a = pooled[perm_idx[:n_a]]
        perm_b = pooled[perm_idx[n_a:]]

        corr_pa = spearman_matrix(perm_a)
        corr_pb = spearman_matrix(perm_b)

        null_mantel[p] = mantel_test(corr_pa, corr_pb)
        null_frob[p] = frobenius_distance(corr_pa, corr_pb)
        null_spectral[p] = spectral_divergence(corr_pa, corr_pb)

        if (p + 1) % 500 == 0:
            logger.info(f"  Permutation {p + 1}/{n_perm}")

    # P-values (ignoring NaN permutations from degenerate splits):
    # Mantel: low r = dissimilar, so p = fraction of null with r <= observed
    valid_mantel = null_mantel[~np.isnan(null_mantel)]
    mantel_p = float(np.mean(valid_mantel <= obs_mantel)) if len(valid_mantel) > 0 else float("nan")
    # Frobenius/spectral: large d = different, so p = fraction of null with d >= observed
    frob_p = float(np.mean(null_frob >= obs_frob))
    valid_spectral = null_spectral[~np.isnan(null_spectral)]
    spectral_p = float(np.mean(valid_spectral >= obs_spectral)) if len(valid_spectral) > 0 else float("nan")

    return {
        "mantel_r": obs_mantel,
        "mantel_p": mantel_p,
        "frobenius_d": obs_frob,
        "frobenius_p": frob_p,
        "spectral_d": obs_spectral,
        "spectral_p": spectral_p,
        "null_distributions": {
            "mantel": null_mantel,
            "frobenius": null_frob,
            "spectral": null_spectral,
        },
        "n_a": n_a,
        "n_b": n_b,
    }


def run_all_comparisons(
    group_data: dict[str, np.ndarray],
    comparisons: Optional[list[tuple[str, str]]] = None,
    n_perm: int = 5000,
    seed: int = 42,
    n_workers: int = 1,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run whole_network_test for each pairwise comparison.

    Parameters
    ----------
    group_data : dict[str, ndarray]
        Mapping from group label to (n_subjects, n_rois) arrays.
    comparisons : list of (str, str), optional
        Pairs of group labels to compare. If None, uses default dose-vs-control
        comparisons within each PND.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    n_workers : int
        Number of parallel workers. 1 = sequential (default).

    Returns
    -------
    results_df : DataFrame
        Columns: comparison, group_a, group_b, n_a, n_b, mantel_r, mantel_p,
        frobenius_d, frobenius_p, spectral_d, spectral_p.
    null_dists : dict[str, dict]
        Keyed by comparison label, values are dicts with 'mantel', 'frobenius',
        'spectral' arrays.
    """
    if comparisons is None:
        comparisons = default_dose_comparisons(list(group_data.keys()))

    # Validate and build work list
    valid_comparisons = []
    for label_a, label_b in comparisons:
        if label_a not in group_data:
            logger.warning(f"Group {label_a} not found, skipping comparison")
            continue
        if label_b not in group_data:
            logger.warning(f"Group {label_b} not found, skipping comparison")
            continue
        valid_comparisons.append((label_a, label_b))

    rows = []
    null_dists = {}

    def _collect_result(comparison_label, label_a, label_b, result):
        rows.append({
            "comparison": comparison_label,
            "group_a": label_a,
            "group_b": label_b,
            "n_a": result["n_a"],
            "n_b": result["n_b"],
            "mantel_r": result["mantel_r"],
            "mantel_p": result["mantel_p"],
            "frobenius_d": result["frobenius_d"],
            "frobenius_p": result["frobenius_p"],
            "spectral_d": result["spectral_d"],
            "spectral_p": result["spectral_p"],
        })
        null_dists[comparison_label] = result["null_distributions"]

    if n_workers > 1 and len(valid_comparisons) > 1:
        logger.info(f"Running {len(valid_comparisons)} whole-network comparisons with {n_workers} workers")
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for label_a, label_b in valid_comparisons:
                comp_label = f"{label_a}_vs_{label_b}"
                future = executor.submit(
                    whole_network_test,
                    data_a=group_data[label_a],
                    data_b=group_data[label_b],
                    n_perm=n_perm,
                    seed=seed,
                )
                futures[future] = (comp_label, label_a, label_b)

            for future in as_completed(futures):
                comp_label, label_a, label_b = futures[future]
                result = future.result()
                _collect_result(comp_label, label_a, label_b, result)
                logger.info(f"  Completed: {comp_label}")
    else:
        for label_a, label_b in valid_comparisons:
            comparison_label = f"{label_a}_vs_{label_b}"
            logger.info(f"\n--- Whole-network test: {comparison_label} ---")

            result = whole_network_test(
                data_a=group_data[label_a],
                data_b=group_data[label_b],
                n_perm=n_perm,
                seed=seed,
            )
            _collect_result(comparison_label, label_a, label_b, result)

    results_df = pd.DataFrame(rows)
    return results_df, null_dists
