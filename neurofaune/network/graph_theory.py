"""
Graph-theoretic metrics for covariance networks.

Computes graph metrics from thresholded correlation matrices across a range
of network densities, producing density curves. Group differences are tested
via permutation on the area under the density curve (AUC), avoiding multiple
comparisons across density levels.

Uses igraph (C backend) for all graph computations (~50x faster than NetworkX).

Available metrics
-----------------
Global metrics (one value per network):
    global_efficiency, local_efficiency, clustering_coefficient, transitivity,
    characteristic_path_length, modularity, small_worldness, assortativity,
    strength

Hub/centrality metrics (per-node, summarised as mean):
    rich_club, betweenness_centrality, degree_entropy
"""

import logging
from itertools import combinations

import igraph as ig
import numpy as np
import pandas as pd
from scipy import stats

from neurofaune.network.matrices import spearman_matrix

logger = logging.getLogger(__name__)

# Default density sweep
DEFAULT_DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def _global_efficiency(G: ig.Graph, n: int) -> float:
    """Mean inverse shortest path length (unweighted)."""
    sp = np.array(G.distances())
    np.fill_diagonal(sp, 0)
    with np.errstate(divide="ignore"):
        inv_sp = np.where((sp > 0) & np.isfinite(sp), 1.0 / sp, 0)
    return float(inv_sp.sum() / (n * (n - 1)))


def _local_efficiency(G: ig.Graph, n: int) -> float:
    """Mean local efficiency across nodes.

    For each node, compute global efficiency of its neighbourhood subgraph.
    """
    local_effs = []
    for v in range(n):
        neighbors = G.neighbors(v)
        if len(neighbors) < 2:
            local_effs.append(0.0)
            continue
        G_sub = G.subgraph(neighbors)
        n_sub = G_sub.vcount()
        sp = np.array(G_sub.distances())
        np.fill_diagonal(sp, 0)
        with np.errstate(divide="ignore"):
            inv_sp = np.where((sp > 0) & np.isfinite(sp), 1.0 / sp, 0)
        local_effs.append(float(inv_sp.sum() / (n_sub * (n_sub - 1))))
    return float(np.mean(local_effs))


def _clustering_coefficient(G: ig.Graph, _n: int) -> float:
    """Mean local clustering coefficient (unweighted)."""
    return G.transitivity_avglocal_undirected(mode="zero")


def _transitivity(G: ig.Graph, _n: int) -> float:
    """Global transitivity (ratio of triangles to connected triples)."""
    return G.transitivity_undirected()


def _characteristic_path_length(G: ig.Graph, _n: int) -> float:
    """Mean shortest path length on the largest connected component."""
    components = G.components()
    lcc_indices = max(components, key=len)
    if len(lcc_indices) < 2:
        return float("inf")
    G_lcc = G.subgraph(lcc_indices)
    sp = np.array(G_lcc.distances())
    n_lcc = G_lcc.vcount()
    return float((sp.sum() - np.trace(sp)) / (n_lcc * (n_lcc - 1)))


def _modularity(G: ig.Graph, _n: int) -> float:
    """Louvain modularity (weighted)."""
    if G.ecount() == 0:
        return 0.0
    partition = G.community_multilevel(weights="weight")
    return partition.modularity


def _small_worldness(G: ig.Graph, n: int) -> float:
    """Small-worldness sigma = (C/C_rand) / (L/L_rand).

    Uses igraph for random graph generation and metric computation.
    """
    if G.ecount() == 0 or n < 4:
        return float("nan")

    C_obs = _clustering_coefficient(G, n)
    L_obs = _characteristic_path_length(G, n)
    if not np.isfinite(L_obs) or L_obs == 0:
        return float("nan")

    m = G.ecount()
    p = (2 * m) / (n * (n - 1))
    n_random = 10

    C_rands, L_rands = [], []
    for i in range(n_random):
        G_rand = ig.Graph.Erdos_Renyi(n, p, directed=False)
        if G_rand.ecount() == 0:
            continue
        C_rands.append(G_rand.transitivity_avglocal_undirected(mode="zero"))
        if G_rand.is_connected():
            sp = np.array(G_rand.distances())
            n_r = G_rand.vcount()
            L_rands.append(
                float((sp.sum() - np.trace(sp)) / (n_r * (n_r - 1)))
            )

    if not C_rands or not L_rands:
        return float("nan")
    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)
    if C_rand == 0 or L_rand == 0:
        return float("nan")

    gamma = C_obs / C_rand
    lam = L_obs / L_rand
    return gamma / lam if lam != 0 else float("nan")


def _assortativity(G: ig.Graph, _n: int) -> float:
    """Degree assortativity coefficient."""
    if G.ecount() == 0:
        return 0.0
    return G.assortativity_degree(directed=False)


def _strength(G: ig.Graph, n: int) -> float:
    """Mean node strength (sum of edge weights per node)."""
    if G.ecount() == 0:
        return 0.0
    return float(np.mean(G.strength(weights="weight")))


def _rich_club(G: ig.Graph, n: int) -> float:
    """Rich-club coefficient at median degree.

    Fraction of edges among nodes with degree > median that actually exist,
    normalised by what would be expected by chance.
    """
    if G.ecount() == 0:
        return 0.0
    degrees = np.array(G.degree())
    k_threshold = int(np.median(degrees))
    rich_nodes = [i for i, d in enumerate(degrees) if d > k_threshold]
    if len(rich_nodes) < 2:
        return 0.0
    G_rich = G.subgraph(rich_nodes)
    n_rich = G_rich.vcount()
    n_possible = n_rich * (n_rich - 1) // 2
    if n_possible == 0:
        return 0.0
    return G_rich.ecount() / n_possible


def _betweenness_centrality(G: ig.Graph, n: int) -> float:
    """Mean betweenness centrality (normalised)."""
    if G.ecount() == 0:
        return 0.0
    bc = G.betweenness(directed=False)
    # Normalise by (n-1)(n-2)/2
    norm = (n - 1) * (n - 2) / 2 if n > 2 else 1
    return float(np.mean(bc) / norm)


def _degree_entropy(G: ig.Graph, n: int) -> float:
    """Shannon entropy of the degree distribution."""
    degrees = np.array(G.degree())
    if degrees.sum() == 0:
        return 0.0
    # Probability distribution
    unique, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


# Metric name -> (function, description)
METRIC_REGISTRY: dict[str, tuple[callable, str]] = {
    "global_efficiency": (_global_efficiency, "Mean inverse shortest path length"),
    "local_efficiency": (_local_efficiency, "Mean local efficiency across nodes"),
    "clustering_coefficient": (
        _clustering_coefficient,
        "Mean local clustering coefficient",
    ),
    "transitivity": (_transitivity, "Global transitivity (triangle ratio)"),
    "characteristic_path_length": (
        _characteristic_path_length,
        "Mean shortest path on largest component",
    ),
    "modularity": (_modularity, "Louvain modularity (weighted)"),
    "small_worldness": (_small_worldness, "Sigma = (C/C_rand) / (L/L_rand)"),
    "assortativity": (_assortativity, "Degree assortativity coefficient"),
    "strength": (_strength, "Mean node strength"),
    "rich_club": (_rich_club, "Rich-club coefficient at median degree"),
    "betweenness_centrality": (
        _betweenness_centrality,
        "Mean normalised betweenness centrality",
    ),
    "degree_entropy": (_degree_entropy, "Shannon entropy of degree distribution"),
}


def list_metrics() -> list[str]:
    """Return available metric names."""
    return list(METRIC_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _fast_spearman(data: np.ndarray) -> np.ndarray:
    """Vectorized Spearman via scipy.stats.rankdata + np.corrcoef."""
    ranked = stats.rankdata(data, axis=0)
    corr = np.corrcoef(ranked, rowvar=False)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _threshold_to_adjacency(corr_matrix: np.ndarray, density: float) -> np.ndarray:
    """Threshold a correlation matrix to a given edge density."""
    n = corr_matrix.shape[0]
    abs_corr = np.abs(corr_matrix.copy())
    np.fill_diagonal(abs_corr, 0)

    n_possible = n * (n - 1) // 2
    n_edges_target = max(1, int(round(density * n_possible)))

    upper_vals = abs_corr[np.triu_indices(n, k=1)]
    if len(upper_vals) == 0:
        return np.zeros_like(abs_corr)

    sorted_vals = np.sort(upper_vals)[::-1]
    if n_edges_target >= len(sorted_vals):
        thresh = 0.0
    else:
        thresh = sorted_vals[min(n_edges_target, len(sorted_vals) - 1)]

    adj = np.where(abs_corr >= thresh, abs_corr, 0)
    np.fill_diagonal(adj, 0)
    return adj


def _build_graph(corr_matrix: np.ndarray, density: float) -> tuple[ig.Graph, int]:
    """Build a thresholded igraph graph from a correlation matrix."""
    adj = _threshold_to_adjacency(corr_matrix, density)
    n = adj.shape[0]
    G = ig.Graph.Weighted_Adjacency(adj.tolist(), mode="undirected")
    return G, n


def compute_metric_curve(
    corr_matrix: np.ndarray,
    metric_name: str,
    densities: list[float] | None = None,
) -> np.ndarray:
    """Compute a single metric across a range of densities.

    Parameters
    ----------
    corr_matrix : ndarray (n_rois, n_rois)
        Correlation matrix.
    metric_name : str
        One of the keys in ``METRIC_REGISTRY``.
    densities : list[float], optional
        Density levels. Default ``DEFAULT_DENSITIES``.

    Returns
    -------
    values : ndarray of shape (len(densities),)
    """
    if densities is None:
        densities = DEFAULT_DENSITIES
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric {metric_name!r}. Available: {list_metrics()}"
        )

    func = METRIC_REGISTRY[metric_name][0]
    values = np.empty(len(densities))
    for i, d in enumerate(densities):
        G, n = _build_graph(corr_matrix, d)
        if G.ecount() == 0:
            values[i] = 0.0
        else:
            values[i] = func(G, n)
    return values


def compute_all_metrics(
    corr_matrix: np.ndarray,
    densities: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute all registered metrics across densities.

    Returns dict mapping metric_name -> array of values.
    """
    if densities is None:
        densities = DEFAULT_DENSITIES

    # Build graphs once per density, reuse for all metrics
    graphs = []
    for d in densities:
        G, n = _build_graph(corr_matrix, d)
        graphs.append((G, n))

    results = {}
    for name, (func, _desc) in METRIC_REGISTRY.items():
        values = np.empty(len(densities))
        for i, (G, n) in enumerate(graphs):
            if G.ecount() == 0:
                values[i] = 0.0
            else:
                values[i] = func(G, n)
        results[name] = values
    return results


def _auc_trapz(values: np.ndarray, densities: list[float]) -> float:
    """Trapezoidal AUC over density curve."""
    return float(np.trapz(values, x=densities))


# ---------------------------------------------------------------------------
# Permutation testing
# ---------------------------------------------------------------------------

def _compare_pair_auc(
    label_a: str,
    label_b: str,
    data_a: np.ndarray,
    data_b: np.ndarray,
    metric_name: str,
    densities: list[float],
    observed_curve_a: np.ndarray,
    observed_curve_b: np.ndarray,
    n_perm: int,
    seed: int,
) -> dict:
    """Permutation test on AUC difference for one metric and one pair.

    Returns a dict with observed curves, AUCs, AUC difference, and p-value.
    """
    func = METRIC_REGISTRY[metric_name][0]

    obs_auc_a = _auc_trapz(observed_curve_a, densities)
    obs_auc_b = _auc_trapz(observed_curve_b, densities)
    obs_diff = obs_auc_a - obs_auc_b

    pooled = np.vstack([data_a, data_b])
    n_a = len(data_a)
    n_total = len(pooled)
    rng = np.random.default_rng(seed)

    null_diffs = np.zeros(n_perm)
    for p in range(n_perm):
        idx = rng.permutation(n_total)
        perm_a = pooled[idx[:n_a]]
        perm_b = pooled[idx[n_a:]]

        corr_a = _fast_spearman(perm_a)
        corr_b = _fast_spearman(perm_b)

        curve_a = np.empty(len(densities))
        curve_b = np.empty(len(densities))
        for i, d in enumerate(densities):
            G_a, n = _build_graph(corr_a, d)
            G_b, _ = _build_graph(corr_b, d)
            curve_a[i] = func(G_a, n) if G_a.ecount() > 0 else 0.0
            curve_b[i] = func(G_b, n) if G_b.ecount() > 0 else 0.0

        null_diffs[p] = _auc_trapz(curve_a, densities) - _auc_trapz(
            curve_b, densities
        )

        if (p + 1) % 500 == 0:
            logger.info(
                "  %s vs %s [%s]: permutation %d/%d",
                label_a, label_b, metric_name, p + 1, n_perm,
            )

    if np.isnan(obs_diff) or np.isinf(obs_diff):
        p_val = float("nan")
    else:
        p_val = float(np.mean(np.abs(null_diffs) >= abs(obs_diff)))

    return {
        "group_a": label_a,
        "group_b": label_b,
        "metric": metric_name,
        "auc_a": obs_auc_a,
        "auc_b": obs_auc_b,
        "auc_diff": obs_diff,
        "p_value": p_val,
        "curve_a": observed_curve_a.tolist(),
        "curve_b": observed_curve_b.tolist(),
        "null_distribution": null_diffs,
    }


def compare_metric(
    groups_data: dict[str, np.ndarray],
    metric_name: str,
    densities: list[float] | None = None,
    n_perm: int = 1000,
    seed: int = 42,
    n_workers: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Permutation test on density-curve AUC for one graph metric.

    For each pair of groups, computes the metric at each density level,
    integrates the density curve (AUC), and tests the AUC difference via
    permutation.

    Parameters
    ----------
    groups_data : dict[str, ndarray]
        Group label -> (n_subjects, n_rois) data array.
    metric_name : str
        Metric to test (key in ``METRIC_REGISTRY``).
    densities : list[float], optional
        Density levels. Default ``DEFAULT_DENSITIES``.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    n_workers : int
        Parallel workers for pairwise comparisons.

    Returns
    -------
    results_df : DataFrame
        One row per pair with AUC values, difference, and p-value.
    curves : dict
        Per-group density curves: ``{group_label: ndarray}``.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if densities is None:
        densities = DEFAULT_DENSITIES
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric {metric_name!r}. Available: {list_metrics()}"
        )

    # Observed curves per group
    observed_curves = {}
    for label, data in groups_data.items():
        corr = spearman_matrix(data)
        observed_curves[label] = compute_metric_curve(corr, metric_name, densities)

    group_labels = sorted(groups_data.keys())
    pairs = list(combinations(group_labels, 2))
    n_pairs = len(pairs)

    logger.info(
        "Graph theory [%s]: %d pairs, %d permutations, %d densities, %d workers",
        metric_name, n_pairs, n_perm, len(densities), n_workers,
    )

    results = []

    if n_workers > 1 and n_pairs > 1:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for pair_idx, (la, lb) in enumerate(pairs, 1):
                future = executor.submit(
                    _compare_pair_auc,
                    label_a=la,
                    label_b=lb,
                    data_a=groups_data[la],
                    data_b=groups_data[lb],
                    metric_name=metric_name,
                    densities=densities,
                    observed_curve_a=observed_curves[la],
                    observed_curve_b=observed_curves[lb],
                    n_perm=n_perm,
                    seed=seed + pair_idx,
                )
                futures[future] = (pair_idx, la, lb)

            for future in as_completed(futures):
                pair_idx, la, lb = futures[future]
                results.append(future.result())
                logger.info(
                    "  %s vs %s complete (%d/%d pairs)",
                    la, lb, pair_idx, n_pairs,
                )
    else:
        for pair_idx, (la, lb) in enumerate(pairs, 1):
            result = _compare_pair_auc(
                label_a=la,
                label_b=lb,
                data_a=groups_data[la],
                data_b=groups_data[lb],
                metric_name=metric_name,
                densities=densities,
                observed_curve_a=observed_curves[la],
                observed_curve_b=observed_curves[lb],
                n_perm=n_perm,
                seed=seed + pair_idx,
            )
            results.append(result)
            logger.info(
                "  %s vs %s complete (%d/%d pairs)",
                la, lb, pair_idx, n_pairs,
            )

    # Build summary DataFrame (without bulky arrays)
    rows = []
    null_distributions = {}
    for r in results:
        rows.append({
            "group_a": r["group_a"],
            "group_b": r["group_b"],
            "metric": r["metric"],
            "auc_a": r["auc_a"],
            "auc_b": r["auc_b"],
            "auc_diff": r["auc_diff"],
            "p_value": r["p_value"],
        })
        null_distributions[f"{r['group_a']}_vs_{r['group_b']}"] = r[
            "null_distribution"
        ]

    return pd.DataFrame(rows), observed_curves, null_distributions
