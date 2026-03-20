"""
Network-Based Statistic (NBS) for comparing correlation matrices.

Implements the NBS method (Zalesky et al., 2010) adapted for structural
covariance networks. Tests which connected subnetworks differ between two
groups by permuting group labels and controlling FWER via the maximum
component size null distribution.

Uses the edge-level Fisher z approach: for each edge, compute the z-statistic
for the difference in correlation between groups, threshold, find connected
components, and assess significance via permutation.

Also includes ``network_based_interaction`` for testing factorial designs
(e.g., dose × sex interaction) via per-edge OLS with F-tests on interaction
terms.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import networkx as nx
import numpy as np
from scipy import stats

from neurofaune.network.matrices import (
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
    n_workers: int = 1,
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
    n_workers : int
        Number of parallel workers. 1 = sequential (default).

    Returns
    -------
    results : dict[str, dict]
        Keyed by "groupA_vs_groupB", values are NBS result dicts plus
        ``roi_cols``.
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

    results = {}

    if n_workers > 1 and len(valid_comparisons) > 1:
        logger.info(f"Running {len(valid_comparisons)} NBS comparisons with {n_workers} workers")
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for label_a, label_b in valid_comparisons:
                comp_label = f"{label_a}_vs_{label_b}"
                future = executor.submit(
                    network_based_statistic,
                    data_a=group_data[label_a],
                    data_b=group_data[label_b],
                    n_perm=n_perm,
                    threshold=threshold,
                    seed=seed,
                )
                futures[future] = (comp_label, label_a, label_b)

            for future in as_completed(futures):
                comp_label, label_a, label_b = futures[future]
                result = future.result()
                result["roi_cols"] = roi_cols
                result["group_a"] = label_a
                result["group_b"] = label_b
                results[comp_label] = result
                logger.info(f"  Completed: {comp_label}")
    else:
        for label_a, label_b in valid_comparisons:
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


def network_based_regression(
    data: np.ndarray,
    covariate: np.ndarray,
    n_perm: int = 5000,
    threshold: float = 3.0,
    seed: int = 42,
    confounds: np.ndarray = None,
) -> dict:
    """Network-based regression: test whether edge co-variation scales with a covariate.

    For each edge (i,j), computes the subject-level product of z-scored
    ROI values (each subject's contribution to the Pearson correlation),
    then regresses these products on the covariate using OLS. The
    t-statistic for the covariate coefficient is used as the test
    statistic per edge.

    Connected suprathreshold components are identified and assessed via
    permutation testing (shuffling covariate labels) with FWER correction
    based on the maximum component size null distribution.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_rois)
        ROI values per subject.
    covariate : ndarray, shape (n_subjects,)
        Continuous covariate (e.g. AUC values).
    n_perm : int
        Number of permutations for null distribution.
    threshold : float
        |t|-statistic threshold for suprathreshold edges.
    seed : int
        Random seed for reproducibility.
    confounds : ndarray, shape (n_subjects, n_confounds), optional
        Optional confound matrix (e.g. sex). If provided, included in the
        OLS model and the covariate t-stat is extracted.

    Returns
    -------
    result : dict
        Keys:
        - ``test_stat``: ndarray (n_rois, n_rois) of t-statistics
        - ``significant_components``: list of component dicts with p-values
        - ``null_distribution``: ndarray of max component sizes per permutation
        - ``n_subjects``: int
    """
    rng = np.random.default_rng(seed)
    n_subjects, n_rois = data.shape

    logger.info(
        "Edge regression: n=%d, n_rois=%d, threshold=%.1f, n_perm=%d",
        n_subjects, n_rois, threshold, n_perm,
    )

    # Z-score each ROI column across subjects (NaN-safe for FOV coverage gaps)
    z_data = np.empty_like(data, dtype=float)
    for j in range(n_rois):
        col = data[:, j].astype(float)
        mu, sd = np.nanmean(col), np.nanstd(col, ddof=1)
        if sd < 1e-12 or np.isnan(sd):
            z_data[:, j] = 0.0
        else:
            z_data[:, j] = (col - mu) / sd
    # Replace NaN with 0 in z-scored data so OLS works; edges involving
    # FOV-limited ROIs will have attenuated statistics, not errors.
    np.nan_to_num(z_data, copy=False, nan=0.0)

    def _compute_edge_tstats(cov_vec):
        """Compute per-edge t-statistics for the covariate."""
        # Build design matrix: [intercept, covariate, confounds...]
        X_cols = [np.ones(n_subjects), cov_vec]
        if confounds is not None:
            if confounds.ndim == 1:
                X_cols.append(confounds)
            else:
                for c in range(confounds.shape[1]):
                    X_cols.append(confounds[:, c])
        X_design = np.column_stack(X_cols)
        cov_idx = 1  # covariate is the second column

        # Precompute (X^T X)^{-1} X^T and hat diagonal
        try:
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
        except np.linalg.LinAlgError:
            return np.zeros((n_rois, n_rois))
        proj = XtX_inv @ X_design.T  # (p, n)
        p = X_design.shape[1]
        df_resid = n_subjects - p

        t_mat = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                # Subject-level edge product: z(ROI_i) * z(ROI_j)
                y_edge = z_data[:, i] * z_data[:, j]

                # OLS: beta = (X^T X)^{-1} X^T y
                beta = proj @ y_edge
                resid = y_edge - X_design @ beta
                mse = (resid ** 2).sum() / df_resid if df_resid > 0 else 1e-12
                se_beta = np.sqrt(max(mse * XtX_inv[cov_idx, cov_idx], 1e-20))
                t_val = beta[cov_idx] / se_beta
                t_mat[i, j] = t_mat[j, i] = t_val

        return t_mat

    # Observed test statistics
    test_stat = _compute_edge_tstats(covariate)

    # Suprathreshold adjacency and observed components
    supra = (np.abs(test_stat) >= threshold).astype(float)
    np.fill_diagonal(supra, 0)
    observed_components = _get_components(supra)
    observed_max = _largest_component_size(supra)

    logger.info(
        "Observed: %d suprathreshold edges, %d components (max size=%d)",
        int(supra.sum() / 2), len(observed_components), observed_max,
    )

    # Permutation null distribution
    null_dist = np.zeros(n_perm)
    for p_idx in range(n_perm):
        perm_cov = rng.permutation(covariate)
        perm_stat = _compute_edge_tstats(perm_cov)
        perm_supra = (np.abs(perm_stat) >= threshold).astype(float)
        np.fill_diagonal(perm_supra, 0)
        null_dist[p_idx] = _largest_component_size(perm_supra)

        if (p_idx + 1) % 500 == 0:
            logger.info("  Permutation %d/%d", p_idx + 1, n_perm)

    # Assign p-values to observed components
    for comp in observed_components:
        comp["pvalue"] = float(np.mean(null_dist >= comp["size"]))

    n_sig = sum(1 for c in observed_components if c["pvalue"] < 0.05)
    logger.info("Edge regression NBS complete: %d significant components (p < 0.05)", n_sig)

    return {
        "test_stat": test_stat,
        "significant_components": observed_components,
        "null_distribution": null_dist,
        "n_subjects": n_subjects,
    }


def _build_interaction_design(
    dose: np.ndarray,
    sex: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build OLS design matrix with dose dummies, sex, and dose×sex interaction.

    Parameters
    ----------
    dose : ndarray, shape (n,)
        Categorical dose labels (e.g. 0=C, 1=L, 2=M, 3=H or string labels).
    sex : ndarray, shape (n,)
        Binary sex labels (0/1 or string labels like 'F'/'M').

    Returns
    -------
    X : ndarray, shape (n, p)
        Full design matrix: [intercept, dose_dummies, sex, dose×sex].
    C_interaction : ndarray, shape (n_interaction, p)
        Contrast matrix selecting the interaction columns.
    col_names : list[str]
        Column names for the design matrix.
    """
    n = len(dose)
    dose = np.asarray(dose)
    sex = np.asarray(sex)

    # Encode dose as dummies (reference = first unique level, typically control)
    dose_levels = sorted(set(dose.flat))
    ref_dose = dose_levels[0]
    dose_dummies = []
    dose_names = []
    for level in dose_levels[1:]:
        col = (dose == level).astype(float)
        dose_dummies.append(col)
        dose_names.append(f"dose_{level}")

    # Encode sex as binary (reference = first unique level)
    sex_levels = sorted(set(sex.flat))
    ref_sex = sex_levels[0]
    sex_col = (sex == sex_levels[1]).astype(float) if len(sex_levels) == 2 else sex.astype(float)
    sex_name = "sex"

    # Interaction terms: each dose dummy × sex
    interaction_cols = []
    interaction_names = []
    for i, dname in enumerate(dose_names):
        icol = dose_dummies[i] * sex_col
        interaction_cols.append(icol)
        interaction_names.append(f"{dname}_x_sex")

    # Assemble design: [intercept, dose_dummies, sex, interactions]
    col_names = ["intercept"] + dose_names + [sex_name] + interaction_names
    X_cols = [np.ones(n)] + dose_dummies + [sex_col] + interaction_cols
    X = np.column_stack(X_cols)

    # Contrast matrix: rows select interaction columns
    n_interaction = len(interaction_cols)
    p = X.shape[1]
    C_interaction = np.zeros((n_interaction, p))
    interaction_start = 1 + len(dose_dummies) + 1  # after intercept + dose + sex
    for i in range(n_interaction):
        C_interaction[i, interaction_start + i] = 1.0

    return X, C_interaction, col_names


def _edge_f_stats(
    z_data: np.ndarray,
    X: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Compute per-edge F-statistics for a contrast in a linear model.

    Parameters
    ----------
    z_data : ndarray, shape (n, n_rois)
        Z-scored ROI data.
    X : ndarray, shape (n, p)
        Design matrix.
    C : ndarray, shape (q, p)
        Contrast matrix (q rows = degrees of freedom for the F-test).

    Returns
    -------
    f_mat : ndarray, shape (n_rois, n_rois)
        F-statistics for each edge.
    """
    n, p = X.shape
    q = C.shape[0]
    n_rois = z_data.shape[1]
    df_resid = n - p

    if df_resid <= 0:
        return np.zeros((n_rois, n_rois))

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.zeros((n_rois, n_rois))

    proj = XtX_inv @ X.T  # (p, n)

    # Precompute C(X'X)^{-1}C' and its inverse for the F-test
    CXtXC = C @ XtX_inv @ C.T
    try:
        CXtXC_inv = np.linalg.inv(CXtXC)
    except np.linalg.LinAlgError:
        return np.zeros((n_rois, n_rois))

    f_mat = np.zeros((n_rois, n_rois))
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            y_edge = z_data[:, i] * z_data[:, j]
            beta = proj @ y_edge
            resid = y_edge - X @ beta
            mse = (resid ** 2).sum() / df_resid

            if mse < 1e-20:
                continue

            # F = (C*beta)' [C(X'X)^{-1}C']^{-1} (C*beta) / (q * MSE)
            Cb = C @ beta
            f_val = float(Cb.T @ CXtXC_inv @ Cb) / (q * mse)
            f_mat[i, j] = f_mat[j, i] = f_val

    return f_mat


def network_based_interaction(
    data: np.ndarray,
    dose: np.ndarray,
    sex: np.ndarray,
    n_perm: int = 5000,
    threshold: float = 3.0,
    seed: int = 42,
    test: str = "interaction",
) -> dict:
    """Network-based test for dose×sex interaction on edge covariance.

    Builds an OLS model with dose dummies, sex, and dose×sex interaction
    terms for each edge's subject-level covariance product (z_i × z_j).
    An F-statistic tests the joint significance of the interaction terms.

    Significance is assessed via permutation: sex labels are shuffled while
    dose labels remain fixed, preserving the marginal dose structure.

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_rois)
        ROI values per subject.
    dose : ndarray, shape (n_subjects,)
        Categorical dose labels (e.g. 'C', 'L', 'M', 'H' or 0, 1, 2, 3).
    sex : ndarray, shape (n_subjects,)
        Binary sex labels (e.g. 'F', 'M' or 0, 1).
    n_perm : int
        Number of permutations for null distribution.
    threshold : float
        F-statistic threshold for suprathreshold edges.
    seed : int
        Random seed for reproducibility.
    test : str
        Which contrast to test. Default ``"interaction"`` tests the dose×sex
        interaction terms. Also supports ``"dose"`` (main effect of dose)
        and ``"sex"`` (main effect of sex).

    Returns
    -------
    result : dict
        Keys:
        - ``test_stat``: ndarray (n_rois, n_rois) of F-statistics
        - ``significant_components``: list of component dicts with p-values
        - ``null_distribution``: ndarray of max component sizes per permutation
        - ``n_subjects``: int
        - ``design_columns``: list of column names
        - ``contrast``: str describing the tested contrast
    """
    rng = np.random.default_rng(seed)
    n_subjects, n_rois = data.shape
    dose = np.asarray(dose)
    sex = np.asarray(sex)

    # Build design matrix
    X, C_interaction, col_names = _build_interaction_design(dose, sex)
    p = X.shape[1]

    # Build alternative contrasts if requested
    if test == "interaction":
        C = C_interaction
        contrast_desc = "dose×sex interaction"
    elif test == "dose":
        # Contrast for dose main effect (dose dummy columns)
        dose_levels = sorted(set(dose.flat))
        n_dose = len(dose_levels) - 1
        C = np.zeros((n_dose, p))
        for i in range(n_dose):
            C[i, 1 + i] = 1.0
        contrast_desc = "dose main effect"
    elif test == "sex":
        # Contrast for sex main effect
        sex_idx = [i for i, name in enumerate(col_names) if name == "sex"]
        C = np.zeros((1, p))
        C[0, sex_idx[0]] = 1.0
        contrast_desc = "sex main effect"
    else:
        raise ValueError(f"Unknown test: {test!r}. Use 'interaction', 'dose', or 'sex'.")

    logger.info(
        "NBS %s: n=%d, n_rois=%d, design=%d cols, contrast q=%d, "
        "threshold=%.1f, n_perm=%d",
        contrast_desc, n_subjects, n_rois, p, C.shape[0], threshold, n_perm,
    )
    logger.info("  Design columns: %s", col_names)

    # Z-score each ROI column
    z_data = np.empty_like(data, dtype=float)
    for j in range(n_rois):
        col = data[:, j].astype(float)
        mu, sd = col.mean(), col.std(ddof=1)
        if sd < 1e-12:
            z_data[:, j] = 0.0
        else:
            z_data[:, j] = (col - mu) / sd

    # Observed F-statistics
    test_stat = _edge_f_stats(z_data, X, C)

    # Suprathreshold adjacency and observed components
    supra = (test_stat >= threshold).astype(float)
    np.fill_diagonal(supra, 0)
    observed_components = _get_components(supra)
    observed_max = _largest_component_size(supra)

    logger.info(
        "Observed: %d suprathreshold edges, %d components (max size=%d)",
        int(supra.sum() / 2), len(observed_components), observed_max,
    )

    # Permutation null distribution
    # Shuffle sex labels while keeping dose fixed → preserves dose marginals
    null_dist = np.zeros(n_perm)
    for p_idx in range(n_perm):
        perm_sex = rng.permutation(sex)
        X_perm, _, _ = _build_interaction_design(dose, perm_sex)
        perm_stat = _edge_f_stats(z_data, X_perm, C)
        perm_supra = (perm_stat >= threshold).astype(float)
        np.fill_diagonal(perm_supra, 0)
        null_dist[p_idx] = _largest_component_size(perm_supra)

        if (p_idx + 1) % 500 == 0:
            logger.info("  Permutation %d/%d", p_idx + 1, n_perm)

    # Assign p-values to observed components
    for comp in observed_components:
        comp["pvalue"] = float(np.mean(null_dist >= comp["size"]))

    n_sig = sum(1 for c in observed_components if c["pvalue"] < 0.05)
    logger.info(
        "NBS %s complete: %d significant components (p < 0.05)",
        contrast_desc, n_sig,
    )

    return {
        "test_stat": test_stat,
        "significant_components": observed_components,
        "null_distribution": null_dist,
        "n_subjects": n_subjects,
        "design_columns": col_names,
        "contrast": contrast_desc,
    }


def nbs_posthoc(
    component: dict,
    test_stat: np.ndarray,
    roi_cols: list[str],
) -> dict:
    """Post-hoc characterisation of a significant NBS component.

    Runs two analyses on the subgraph defined by the component edges:

    1. **Node centrality** — degree, betweenness, and eigenvector centrality
       for every node in the component. Nodes with high betweenness are
       structural bridges; high eigenvector centrality indicates influence
       via well-connected neighbours.

    2. **Hub vulnerability (leave-one-out)** — for each node, remove it and
       all its edges, then record how much the largest remaining connected
       sub-component shrinks relative to the original size. A large drop
       identifies load-bearing hubs whose removal fragments the network.

    Parameters
    ----------
    component : dict
        A single component dict from ``significant_components`` (must have
        ``"edges"`` as a list of ``(u, v)`` tuples and ``"nodes"`` as a list
        of ROI indices).
    test_stat : ndarray, shape (n_rois, n_rois)
        Per-edge test statistics (used to annotate centrality output).
    roi_cols : list[str]
        ROI names indexed by position.

    Returns
    -------
    result : dict
        - ``"centrality"`` : list of dicts, one per node, sorted by
          betweenness descending. Keys: ``roi``, ``degree``,
          ``betweenness``, ``eigenvector``, ``mean_abs_z``.
        - ``"hub_vulnerability"`` : list of dicts, one per node, sorted by
          ``size_drop`` descending. Keys: ``roi``, ``original_size``,
          ``size_after_removal``, ``size_drop``, ``fraction_drop``.
    """
    edges = component["edges"]
    nodes = component["nodes"]
    original_size = component["size"]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v in edges:
        G.add_edge(u, v)

    # --- Centrality ---
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    centrality_rows = []
    for node in nodes:
        # Mean |z| of all edges incident to this node
        incident = [(node, v) for v in G.neighbors(node)]
        mean_z = float(np.mean([abs(test_stat[u, v]) for u, v in incident])) if incident else 0.0
        centrality_rows.append({
            "roi": roi_cols[node],
            "degree": degree.get(node, 0),
            "betweenness": round(betweenness.get(node, 0.0), 6),
            "eigenvector": round(eigenvector.get(node, 0.0), 6),
            "mean_abs_z": round(mean_z, 4),
        })
    centrality_rows.sort(key=lambda r: r["betweenness"], reverse=True)

    # --- Hub vulnerability ---
    vulnerability_rows = []
    for node in nodes:
        H = G.copy()
        H.remove_node(node)
        if H.number_of_edges() == 0:
            remaining = 0
        else:
            remaining = max(
                H.subgraph(c).number_of_edges()
                for c in nx.connected_components(H)
            )
        drop = original_size - remaining
        vulnerability_rows.append({
            "roi": roi_cols[node],
            "original_size": original_size,
            "size_after_removal": remaining,
            "size_drop": drop,
            "fraction_drop": round(drop / original_size, 4) if original_size > 0 else 0.0,
        })
    vulnerability_rows.sort(key=lambda r: r["size_drop"], reverse=True)

    return {
        "centrality": centrality_rows,
        "hub_vulnerability": vulnerability_rows,
    }


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
