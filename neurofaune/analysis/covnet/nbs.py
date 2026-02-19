"""
Network-Based Statistic (NBS) for comparing correlation matrices.

Implements the NBS method (Zalesky et al., 2010) adapted for structural
covariance networks. Tests which connected subnetworks differ between two
groups by permuting group labels and controlling FWER via the maximum
component size null distribution.

Uses the edge-level Fisher z approach: for each edge, compute the z-statistic
for the difference in correlation between groups, threshold, find connected
components, and assess significance via permutation.
"""

import logging
from typing import Optional

import networkx as nx
import numpy as np
from scipy import stats

from neurofaune.analysis.covnet.matrices import (
    compute_spearman_matrices,
    default_dose_comparisons,
    fisher_z_transform,
    spearman_matrix,
)

logger = logging.getLogger(__name__)


def _fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> float:
    """Two-sample Fisher z-test for difference in correlations.

    Returns the z-statistic (positive = r1 > r2).
    """
    z1 = fisher_z_transform(np.asarray(r1))
    z2 = fisher_z_transform(np.asarray(r2))
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    if se == 0:
        return 0.0
    return float((z1 - z2) / se)


def _largest_component_size(adj: np.ndarray) -> int:
    """Find the size (number of edges) of the largest connected component."""
    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return 0
    components = list(nx.connected_components(G))
    max_size = 0
    for comp in components:
        subgraph = G.subgraph(comp)
        max_size = max(max_size, subgraph.number_of_edges())
    return max_size


def _get_components(adj: np.ndarray) -> list[dict]:
    """Extract connected components from a suprathreshold adjacency matrix.

    Returns a list of dicts, each with 'nodes', 'edges', 'size' (edge count).
    """
    G = nx.from_numpy_array(adj)
    components = []
    for comp_nodes in nx.connected_components(G):
        if len(comp_nodes) < 2:
            continue
        subgraph = G.subgraph(comp_nodes)
        n_edges = subgraph.number_of_edges()
        if n_edges == 0:
            continue
        edges = [(int(u), int(v)) for u, v in subgraph.edges()]
        components.append({
            "nodes": sorted(int(n) for n in comp_nodes),
            "edges": edges,
            "size": n_edges,
        })
    return sorted(components, key=lambda c: c["size"], reverse=True)


def network_based_statistic(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_perm: int = 5000,
    threshold: float = 3.0,
    seed: int = 42,
) -> dict:
    """Run Network-Based Statistic comparing two groups.

    Uses the edge-level Fisher z approach: compute Spearman correlation matrices
    for each group, Fisher z-test each edge for group difference, threshold at
    ``|z| >= threshold``, find connected components, then permute group labels
    to build a null distribution of maximum component sizes.

    Parameters
    ----------
    data_a : ndarray, shape (n_subjects_a, n_rois)
        ROI values for group A.
    data_b : ndarray, shape (n_subjects_b, n_rois)
        ROI values for group B.
    n_perm : int
        Number of permutations for null distribution.
    threshold : float
        Z-statistic threshold for suprathreshold edges.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Keys:
        - ``test_stat``: ndarray (n_rois, n_rois) of Fisher z-statistics
        - ``significant_components``: list of component dicts with p-values
        - ``null_distribution``: ndarray of max component sizes per permutation
        - ``n_a``, ``n_b``: group sizes
    """
    rng = np.random.default_rng(seed)
    n_a, n_rois = data_a.shape
    n_b = data_b.shape[0]
    pooled = np.vstack([data_a, data_b])
    n_total = n_a + n_b

    logger.info(
        f"NBS: n_a={n_a}, n_b={n_b}, n_rois={n_rois}, "
        f"threshold={threshold}, n_perm={n_perm}"
    )

    # Observed test statistics
    corr_a = spearman_matrix(data_a)
    corr_b = spearman_matrix(data_b)
    test_stat = np.zeros((n_rois, n_rois))
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            z = _fisher_z_test(corr_a[i, j], n_a, corr_b[i, j], n_b)
            test_stat[i, j] = test_stat[j, i] = z

    # Suprathreshold adjacency and observed components
    supra = (np.abs(test_stat) >= threshold).astype(float)
    np.fill_diagonal(supra, 0)
    observed_components = _get_components(supra)
    observed_max = _largest_component_size(supra)

    logger.info(
        f"Observed: {int(supra.sum() / 2)} suprathreshold edges, "
        f"{len(observed_components)} components (max size={observed_max})"
    )

    # Permutation null distribution
    null_dist = np.zeros(n_perm)
    for p in range(n_perm):
        perm_idx = rng.permutation(n_total)
        perm_a = pooled[perm_idx[:n_a]]
        perm_b = pooled[perm_idx[n_a:]]

        corr_pa = spearman_matrix(perm_a)
        corr_pb = spearman_matrix(perm_b)

        perm_stat = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                z = _fisher_z_test(corr_pa[i, j], n_a, corr_pb[i, j], n_b)
                perm_stat[i, j] = z

        perm_supra = (np.abs(perm_stat) >= threshold).astype(float)
        null_dist[p] = _largest_component_size(perm_supra)

        if (p + 1) % 500 == 0:
            logger.info(f"  Permutation {p + 1}/{n_perm}")

    # Assign p-values to observed components
    for comp in observed_components:
        comp["pvalue"] = float(np.mean(null_dist >= comp["size"]))

    n_sig = sum(1 for c in observed_components if c["pvalue"] < 0.05)
    logger.info(f"NBS complete: {n_sig} significant components (p < 0.05)")

    return {
        "test_stat": test_stat,
        "significant_components": observed_components,
        "null_distribution": null_dist,
        "n_a": n_a,
        "n_b": n_b,
    }



def run_all_comparisons(
    group_data: dict[str, np.ndarray],
    group_sizes: dict[str, int],
    roi_cols: list[str],
    comparisons: Optional[list[tuple[str, str]]] = None,
    n_perm: int = 5000,
    threshold: float = 3.0,
    seed: int = 42,
) -> dict[str, dict]:
    """Run NBS for each specified pairwise comparison.

    Parameters
    ----------
    group_data : dict[str, ndarray]
        Mapping from group label to (n_subjects, n_rois) arrays.
    group_sizes : dict[str, int]
        Number of subjects per group (redundant with data shape but explicit).
    roi_cols : list[str]
        ROI names for labeling results.
    comparisons : list of (str, str), optional
        Pairs of group labels to compare. If None, compares each dose vs
        control within each PND (9 comparisons).
    n_perm : int
        Number of permutations.
    threshold : float
        Z-statistic threshold.
    seed : int
        Random seed.

    Returns
    -------
    results : dict[str, dict]
        Keyed by "groupA_vs_groupB", values are NBS result dicts plus
        ``roi_cols``.
    """
    if comparisons is None:
        comparisons = default_dose_comparisons(list(group_data.keys()))

    results = {}
    for label_a, label_b in comparisons:
        if label_a not in group_data:
            logger.warning(f"Group {label_a} not found, skipping comparison")
            continue
        if label_b not in group_data:
            logger.warning(f"Group {label_b} not found, skipping comparison")
            continue

        comparison_label = f"{label_a}_vs_{label_b}"
        logger.info(f"\n--- NBS: {comparison_label} ---")

        result = network_based_statistic(
            data_a=group_data[label_a],
            data_b=group_data[label_b],
            n_perm=n_perm,
            threshold=threshold,
            seed=seed,
        )
        result["roi_cols"] = roi_cols
        result["group_a"] = label_a
        result["group_b"] = label_b
        results[comparison_label] = result

    return results


def fisher_z_edge_test(
    corr_a: np.ndarray,
    n_a: int,
    corr_b: np.ndarray,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fisher z-test for each edge between two correlation matrices.

    Useful for territory-level analysis where NBS is not needed (small number
    of edges allows direct multiple comparison correction).

    Parameters
    ----------
    corr_a, corr_b : ndarray (n_rois, n_rois)
        Spearman correlation matrices for groups A and B.
    n_a, n_b : int
        Sample sizes.

    Returns
    -------
    z_stats : ndarray (n_rois, n_rois)
        Z-statistics for each edge (positive = corr_a > corr_b).
    p_values : ndarray (n_rois, n_rois)
        Two-sided p-values.
    """
    n_rois = corr_a.shape[0]
    z_stats = np.zeros((n_rois, n_rois))
    p_values = np.ones((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            z = _fisher_z_test(corr_a[i, j], n_a, corr_b[i, j], n_b)
            p = 2.0 * stats.norm.sf(abs(z))
            z_stats[i, j] = z_stats[j, i] = z
            p_values[i, j] = p_values[j, i] = p

    return z_stats, p_values
