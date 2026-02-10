"""
Graph-theoretic metrics for covariance networks.

Computes global efficiency, clustering coefficient, modularity (Louvain),
characteristic path length, and small-worldness from thresholded correlation
matrices. Includes permutation-based comparison of metrics across groups.
"""

import logging
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_metrics(corr_matrix: np.ndarray, density: float = 0.15) -> dict:
    """Compute graph-theoretic metrics from a correlation matrix.

    The matrix is thresholded at a proportional density (keeping the top
    ``density`` fraction of absolute correlations) so all groups have the
    same number of edges, enabling fair comparison.

    Parameters
    ----------
    corr_matrix : ndarray (n_rois, n_rois)
        Spearman correlation matrix.
    density : float
        Fraction of edges to retain (0-1). Default 0.15.

    Returns
    -------
    metrics : dict
        Keys: global_efficiency, clustering_coefficient, modularity,
        characteristic_path_length, small_worldness, n_nodes, n_edges,
        density_actual.
    """
    n = corr_matrix.shape[0]

    # Threshold to desired density using absolute correlations
    abs_corr = np.abs(corr_matrix.copy())
    np.fill_diagonal(abs_corr, 0)

    # Number of possible edges in upper triangle
    n_possible = n * (n - 1) // 2
    n_edges_target = max(1, int(round(density * n_possible)))

    # Get threshold that yields desired edge count
    upper_vals = abs_corr[np.triu_indices(n, k=1)]
    if len(upper_vals) == 0:
        return _empty_metrics(n)

    sorted_vals = np.sort(upper_vals)[::-1]
    if n_edges_target >= len(sorted_vals):
        thresh = 0.0
    else:
        thresh = sorted_vals[min(n_edges_target, len(sorted_vals) - 1)]

    # Build adjacency: edge weights = absolute correlation above threshold
    adj = np.where(abs_corr >= thresh, abs_corr, 0)
    np.fill_diagonal(adj, 0)

    G = nx.from_numpy_array(adj)

    # Remove isolated nodes for path-based metrics
    isolates = list(nx.isolates(G))
    G_connected = G.copy()
    G_connected.remove_nodes_from(isolates)

    n_edges_actual = G.number_of_edges()
    density_actual = (2 * n_edges_actual) / (n * (n - 1)) if n > 1 else 0

    # Global efficiency
    global_eff = nx.global_efficiency(G)

    # Clustering coefficient (weighted)
    clustering = nx.average_clustering(G, weight="weight")

    # Modularity (Louvain)
    if G.number_of_edges() > 0:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
        modularity = nx.community.modularity(G, communities, weight="weight")
        n_communities = len(communities)
    else:
        modularity = 0.0
        n_communities = 0

    # Characteristic path length (on largest connected component)
    if G_connected.number_of_nodes() > 1:
        # Convert weights to distances (stronger correlation = shorter path)
        G_dist = G_connected.copy()
        for u, v, d in G_dist.edges(data=True):
            w = d.get("weight", 1.0)
            d["distance"] = 1.0 / w if w > 0 else float("inf")

        largest_cc = max(nx.connected_components(G_dist), key=len)
        G_lcc = G_dist.subgraph(largest_cc)
        cpl = nx.average_shortest_path_length(G_lcc, weight="distance")
    else:
        cpl = float("inf")

    # Small-worldness: sigma = (C/C_rand) / (L/L_rand)
    small_worldness = _compute_small_worldness(G, clustering, cpl)

    return {
        "global_efficiency": global_eff,
        "clustering_coefficient": clustering,
        "modularity": modularity,
        "characteristic_path_length": cpl,
        "small_worldness": small_worldness,
        "n_communities": n_communities,
        "n_nodes": n,
        "n_edges": n_edges_actual,
        "density_actual": density_actual,
    }


def _compute_small_worldness(
    G: nx.Graph, C_obs: float, L_obs: float, n_random: int = 10
) -> float:
    """Estimate small-worldness sigma = (C/C_rand) / (L/L_rand).

    Generates Erdos-Renyi random graphs with matched node/edge count.
    """
    if G.number_of_edges() == 0 or G.number_of_nodes() < 4:
        return float("nan")
    if L_obs == float("inf") or L_obs == 0:
        return float("nan")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = (2 * m) / (n * (n - 1))

    C_rands = []
    L_rands = []
    for i in range(n_random):
        G_rand = nx.erdos_renyi_graph(n, p, seed=42 + i)
        if G_rand.number_of_edges() == 0:
            continue
        C_rands.append(nx.average_clustering(G_rand))
        if nx.is_connected(G_rand):
            L_rands.append(nx.average_shortest_path_length(G_rand))

    if not C_rands or not L_rands:
        return float("nan")

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    if C_rand == 0 or L_rand == 0:
        return float("nan")

    gamma = C_obs / C_rand
    lam = L_obs / L_rand
    return gamma / lam if lam != 0 else float("nan")


def _empty_metrics(n: int) -> dict:
    """Return empty metrics dict for degenerate cases."""
    return {
        "global_efficiency": 0.0,
        "clustering_coefficient": 0.0,
        "modularity": 0.0,
        "characteristic_path_length": float("inf"),
        "small_worldness": float("nan"),
        "n_communities": 0,
        "n_nodes": n,
        "n_edges": 0,
        "density_actual": 0.0,
    }


def compare_metrics(
    groups_data: dict[str, np.ndarray],
    roi_cols: list[str],
    densities: list[float] = None,
    n_perm: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation test on graph metric differences between groups.

    For each pair of groups and each density, permutes group labels and
    recomputes metrics to build a null distribution for the observed
    difference.

    Parameters
    ----------
    groups_data : dict[str, ndarray]
        Group label -> (n_subjects, n_rois) data array.
    roi_cols : list[str]
        ROI names (for logging).
    densities : list[float], optional
        Network densities to test. Default [0.10, 0.15, 0.20, 0.25].
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    results_df : DataFrame
        Columns: group_a, group_b, metric, density, observed_a, observed_b,
        observed_diff, p_value.
    """
    if densities is None:
        densities = [0.10, 0.15, 0.20, 0.25]

    rng = np.random.default_rng(seed)
    metric_names = [
        "global_efficiency",
        "clustering_coefficient",
        "modularity",
        "characteristic_path_length",
    ]

    # Compute observed metrics per group per density
    observed = {}
    for label, data in groups_data.items():
        from neurofaune.analysis.covnet.nbs import _spearman_matrix

        corr = _spearman_matrix(data)
        observed[label] = {}
        for d in densities:
            observed[label][d] = compute_metrics(corr, density=d)

    # Pairwise permutation tests
    rows = []
    group_labels = sorted(groups_data.keys())

    for label_a, label_b in combinations(group_labels, 2):
        data_a = groups_data[label_a]
        data_b = groups_data[label_b]
        pooled = np.vstack([data_a, data_b])
        n_a = len(data_a)
        n_total = len(pooled)

        for d in densities:
            # Observed differences
            obs_diffs = {}
            for m in metric_names:
                val_a = observed[label_a][d][m]
                val_b = observed[label_b][d][m]
                obs_diffs[m] = val_a - val_b

            # Permutation null
            null_diffs = {m: np.zeros(n_perm) for m in metric_names}
            for p in range(n_perm):
                idx = rng.permutation(n_total)
                perm_a = pooled[idx[:n_a]]
                perm_b = pooled[idx[n_a:]]

                corr_pa = _spearman_matrix(perm_a)
                corr_pb = _spearman_matrix(perm_b)
                met_a = compute_metrics(corr_pa, density=d)
                met_b = compute_metrics(corr_pb, density=d)

                for m in metric_names:
                    null_diffs[m][p] = met_a[m] - met_b[m]

            # P-values (two-sided)
            for m in metric_names:
                obs_d = obs_diffs[m]
                if np.isnan(obs_d) or np.isinf(obs_d):
                    p_val = float("nan")
                else:
                    p_val = float(np.mean(np.abs(null_diffs[m]) >= abs(obs_d)))
                rows.append({
                    "group_a": label_a,
                    "group_b": label_b,
                    "metric": m,
                    "density": d,
                    "observed_a": observed[label_a][d][m],
                    "observed_b": observed[label_b][d][m],
                    "observed_diff": obs_d,
                    "p_value": p_val,
                })

            logger.info(
                f"  {label_a} vs {label_b} @ density={d}: done"
            )

    return pd.DataFrame(rows)
