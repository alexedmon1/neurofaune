"""Multiset Canonical Correlation Analysis (MCCA) for multi-modal ROI data.

Implements regularised MCCA to find linear combinations of ROI features that
maximise correlation across modality views (e.g., DWI, MSME, functional).
Uses Ledoit-Wolf shrinkage for covariance regularisation by default.

No external CCA library is required — the algorithm uses scipy's generalised
eigenvalue solver with sklearn's LedoitWolf shrinkage estimator.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from neurofaune.network.classification.data_prep import prepare_classification_data

logger = logging.getLogger(__name__)


@dataclass
class MCCAResult:
    """Result container for a fitted MCCA model."""

    scores: List[np.ndarray]  # Per-view projected scores [n_samples, n_components]
    canonical_correlations: np.ndarray  # Correlation per component
    weights: List[np.ndarray]  # Per-view weight matrices [n_features_k, n_components]
    view_names: List[str]
    feature_names: List[List[str]]  # Per-view feature name lists
    n_components: int
    n_samples: int
    regularisation: str  # 'lw', 'identity', or float


@dataclass
class PermutationResult:
    """Result container for MCCA permutation test."""

    observed: np.ndarray  # Observed canonical correlations [n_components]
    null_distributions: np.ndarray  # [n_permutations, n_components]
    p_values: np.ndarray  # Per-component p-values
    n_permutations: int


@dataclass
class DoseAssociationResult:
    """Result container for dose-association test on MCCA scores."""

    spearman_rho: np.ndarray  # Per-component Spearman rho with dose
    p_values: np.ndarray  # Permutation p-values
    n_permutations: int


def load_multiview_data(
    roi_dir: Path,
    views: Dict[str, List[str]],
    feature_set: str = "bilateral",
    cohort_filter: Optional[str] = None,
    exclusion_csv: Optional[Path] = None,
    confounds: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[str], List[List[str]], pd.DataFrame]:
    """Load and align multi-view ROI data for MCCA.

    Each view is a modality group (e.g., DWI with FA/MD/AD/RD metrics).
    Metrics within a view are column-concatenated. Subjects are intersected
    across all views so every sample appears in every view.

    Parameters
    ----------
    roi_dir : Path
        Directory containing roi_{METRIC}_wide.csv files.
    views : dict
        Mapping of view_name -> list of metric names.
        E.g. {"dwi": ["FA", "MD", "AD", "RD"], "func": ["fALFF", "ReHo"]}.
    feature_set : str
        'bilateral' or 'territory' — passed to prepare_classification_data.
    cohort_filter : str, optional
        Restrict to a single cohort (e.g. 'p30').
    exclusion_csv : Path, optional
        Exclusion list CSV.
    confounds : list of str, optional
        Metadata columns to residualize from features before z-scoring
        (e.g. ``["sex"]``). Categorical columns are dummy-encoded.

    Returns
    -------
    Xs : list of ndarray
        Per-view feature matrices, shape [n_samples, n_features_k].
    view_names : list of str
        Ordered view names.
    feature_names : list of list of str
        Per-view feature name lists (prefixed with metric name).
    metadata_df : DataFrame
        Subject/session/dose/cohort(/sex) for the intersected samples.
    """
    view_names = list(views.keys())
    view_data = {}  # view_name -> {subject_session_key -> (row_info, feature_dict)}

    for view_name in view_names:
        metrics = views[view_name]
        metric_results = []  # list of (keys, info, X, feature_names)

        for metric in metrics:
            csv_path = roi_dir / f"roi_{metric}_wide.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"ROI CSV not found: {csv_path}")

            data = prepare_classification_data(
                wide_csv=csv_path,
                feature_set=feature_set,
                cohort_filter=cohort_filter,
                exclusion_csv=exclusion_csv,
                standardize=False,  # We standardise per-view below
            )

            info = data["sample_info"]
            keys = (info["subject"] + "_" + info["session"]).values
            prefixed = [f"{metric}_{fn}" for fn in data["feature_names"]]
            metric_results.append((keys, info, data["X"], prefixed))

        # Intersect subjects across metrics within this view
        view_keys = set(metric_results[0][0])
        for keys, _, _, _ in metric_results[1:]:
            view_keys &= set(keys)
        view_keys_sorted = sorted(view_keys)

        if len(view_keys_sorted) < len(metric_results[0][0]):
            logger.info(
                "View '%s': intersected %d -> %d subjects across %d metrics",
                view_name, len(metric_results[0][0]), len(view_keys_sorted),
                len(metrics),
            )

        # Align all metrics to the common subject set
        all_features = []
        all_feature_names = []
        sample_info = None
        for keys, info, X, fnames in metric_results:
            key_to_idx = {k: i for i, k in enumerate(keys)}
            idx = [key_to_idx[k] for k in view_keys_sorted]
            all_features.append(X[idx])
            all_feature_names.extend(fnames)
            if sample_info is None:
                sample_info = info.iloc[idx].reset_index(drop=True)

        X_view = np.hstack(all_features)
        view_data[view_name] = {
            "X": X_view,
            "feature_names": all_feature_names,
            "keys": np.array(view_keys_sorted),
            "info": sample_info,
        }
        logger.info(
            "View '%s': %d subjects, %d features (%s)",
            view_name, X_view.shape[0], X_view.shape[1],
            ", ".join(metrics),
        )

    # Intersect subjects across views
    common_keys = None
    for view_name in view_names:
        keys_set = set(view_data[view_name]["keys"])
        if common_keys is None:
            common_keys = keys_set
        else:
            common_keys = common_keys & keys_set

    common_keys_sorted = sorted(common_keys)
    n_common = len(common_keys_sorted)
    logger.info("Subject intersection across %d views: n=%d", len(view_names), n_common)

    if n_common < 10:
        raise ValueError(f"Too few common subjects ({n_common}) across views")

    # Align views to common subjects
    Xs = []
    all_feature_names = []
    metadata_df = None

    for view_name in view_names:
        vd = view_data[view_name]
        keys = vd["keys"]
        # Build index mapping
        key_to_idx = {k: i for i, k in enumerate(keys)}
        idx = [key_to_idx[k] for k in common_keys_sorted]

        X_aligned = vd["X"][idx]
        Xs.append(X_aligned)
        all_feature_names.append(vd["feature_names"])

        if metadata_df is None:
            metadata_df = vd["info"].iloc[idx].reset_index(drop=True)

    # Residualize confounds (before z-scoring)
    if confounds:
        missing = [c for c in confounds if c not in metadata_df.columns]
        if missing:
            raise ValueError(f"Confound columns not found in metadata: {missing}")
        C = pd.get_dummies(metadata_df[confounds], drop_first=True, dtype=float).values
        logger.info("Residualising confounds: %s (%d regressors)", confounds, C.shape[1])
        Xs = [_residualize_confounds(X, C) for X in Xs]

    # Z-score standardise per view
    for i in range(len(Xs)):
        scaler = StandardScaler()
        Xs[i] = scaler.fit_transform(Xs[i])

    return Xs, view_names, all_feature_names, metadata_df


def _residualize_confounds(X: np.ndarray, confound_matrix: np.ndarray) -> np.ndarray:
    """Regress out confounds from each feature via OLS, return residuals."""
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(confound_matrix, X)
    return X - model.predict(confound_matrix)


def _regularise_cov(X: np.ndarray, method: str = "lw") -> np.ndarray:
    """Compute regularised covariance matrix.

    Parameters
    ----------
    X : ndarray, shape [n_samples, n_features]
        Centred data matrix.
    method : str
        'lw' for Ledoit-Wolf shrinkage, 'identity' for identity regularisation,
        or a float for manual shrinkage alpha.

    Returns
    -------
    C : ndarray, shape [n_features, n_features]
        Regularised covariance matrix.
    """
    if method == "lw":
        lw = LedoitWolf().fit(X)
        return lw.covariance_
    elif method == "identity":
        n = X.shape[0]
        C = (X.T @ X) / (n - 1)
        alpha = 0.1
        return (1 - alpha) * C + alpha * np.eye(C.shape[0])
    else:
        try:
            alpha = float(method)
        except (ValueError, TypeError):
            raise ValueError(f"Unknown regularisation method: {method!r}")
        n = X.shape[0]
        C = (X.T @ X) / (n - 1)
        return (1 - alpha) * C + alpha * np.eye(C.shape[0])


def _pca_reduce_views(
    Xs: List[np.ndarray],
    max_components: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Reduce each view via PCA when p > max_components.

    Keeps enough variance for robust MCCA while making the generalised
    eigenvalue problem small enough for fast permutation testing.

    Parameters
    ----------
    Xs : list of ndarray
        Per-view data, each [n_samples, n_features_k].
    max_components : int, optional
        Max PCA components per view. Default: min(n_samples - 1, 30).
        30 components per view is sufficient for 5 canonical variates and
        keeps the block eigenvalue problem fast (~90x90 for 3 views).

    Returns
    -------
    Xs_reduced : list of ndarray
        PCA-projected views, each [n_samples, min(n_features_k, max_components)].
    pca_projections : list of ndarray
        PCA projection matrices for back-projecting loadings. Each is
        [n_features_k, n_pca_components].
    """
    from sklearn.decomposition import PCA

    n = Xs[0].shape[0]
    if max_components is None:
        max_components = min(n - 1, 30)

    Xs_reduced = []
    pca_projections = []

    for k, X in enumerate(Xs):
        if X.shape[1] > max_components:
            n_comp = min(max_components, X.shape[1], n - 1)
            pca = PCA(n_components=n_comp, whiten=False)
            X_pca = pca.fit_transform(X)
            Xs_reduced.append(X_pca)
            pca_projections.append(pca.components_.T)  # [n_features, n_pca]
            logger.debug(
                "View %d: PCA %d -> %d (%.1f%% variance)",
                k, X.shape[1], n_comp,
                100 * pca.explained_variance_ratio_.sum(),
            )
        else:
            Xs_reduced.append(X)
            pca_projections.append(np.eye(X.shape[1]))

    return Xs_reduced, pca_projections


def _solve_mcca_eigenvalue(
    Xs: List[np.ndarray],
    n_components: int,
    C_within: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build cross-covariance and solve the generalised eigenvalue problem.

    Separated from run_mcca to allow reuse in permutation testing where
    C_within is pre-computed (invariant to row permutations).
    """
    K = len(Xs)
    n = Xs[0].shape[0]
    total_d = C_within.shape[0]

    C_between = np.zeros((total_d, total_d))
    for i in range(K):
        si, ei = offsets[i], offsets[i + 1]
        for j in range(K):
            sj, ej = offsets[j], offsets[j + 1]
            C_between[si:ei, sj:ej] = (Xs[i].T @ Xs[j]) / (n - 1)

    try:
        eigenvalues, eigenvectors = linalg.eigh(
            C_between, C_within,
            subset_by_index=[total_d - n_components, total_d - 1],
        )
    except linalg.LinAlgError:
        eigenvalues, eigenvectors = linalg.eig(C_between, C_within)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def _compute_canonical_correlations(
    Xs: List[np.ndarray],
    eigenvectors: np.ndarray,
    offsets: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Compute mean pairwise correlation across views per component."""
    K = len(Xs)
    scores = []
    for i in range(K):
        si, ei = offsets[i], offsets[i + 1]
        scores.append(Xs[i] @ eigenvectors[si:ei, :])

    canonical_correlations = np.zeros(n_components)
    for c in range(n_components):
        corrs = []
        for i in range(K):
            for j in range(i + 1, K):
                r = np.corrcoef(scores[i][:, c], scores[j][:, c])[0, 1]
                corrs.append(abs(r))
        canonical_correlations[c] = np.mean(corrs) if corrs else 0.0
    return canonical_correlations


def run_mcca(
    Xs: List[np.ndarray],
    n_components: int = 5,
    regs: str = "lw",
) -> MCCAResult:
    """Fit regularised Multiset Canonical Correlation Analysis.

    Solves the generalised eigenvalue problem for the block cross-covariance
    vs block within-covariance matrices.

    Parameters
    ----------
    Xs : list of ndarray
        Per-view data matrices, each shape [n_samples, n_features_k].
        Should be centred (zero-mean). All must have the same n_samples.
    n_components : int
        Number of canonical components to extract.
    regs : str
        Regularisation method: 'lw' (Ledoit-Wolf), 'identity', or a float.

    Returns
    -------
    MCCAResult
    """
    K = len(Xs)
    n = Xs[0].shape[0]

    # PCA reduction when p > n (critical for permutation speed)
    Xs_reduced, pca_projs = _pca_reduce_views(Xs)

    dims = [X.shape[1] for X in Xs_reduced]
    total_d = sum(dims)

    n_components = min(n_components, n - 1, min(dims))
    offsets = np.cumsum([0] + dims)

    # Build block-diagonal within-view covariance (regularised)
    C_within = np.zeros((total_d, total_d))
    for i in range(K):
        si, ei = offsets[i], offsets[i + 1]
        C_within[si:ei, si:ei] = _regularise_cov(Xs_reduced[i], method=regs)

    eigenvalues, eigenvectors = _solve_mcca_eigenvalue(
        Xs_reduced, n_components, C_within, offsets
    )

    # Extract per-view weights in PCA space, then back-project to original space
    weights = []
    scores = []
    for i in range(K):
        si, ei = offsets[i], offsets[i + 1]
        W_pca = eigenvectors[si:ei, :]  # weights in PCA space
        scores.append(Xs_reduced[i] @ W_pca)
        # Back-project to original feature space for interpretability
        W_orig = pca_projs[i] @ W_pca
        weights.append(W_orig)

    canonical_correlations = _compute_canonical_correlations(
        Xs_reduced, eigenvectors, offsets, n_components
    )

    return MCCAResult(
        scores=scores,
        canonical_correlations=canonical_correlations,
        weights=weights,
        view_names=[],  # Filled in by caller
        feature_names=[],  # Filled in by caller
        n_components=n_components,
        n_samples=n,
        regularisation=regs,
    )


def permutation_test_mcca(
    Xs: List[np.ndarray],
    observed_correlations: np.ndarray,
    n_components: int = 5,
    regs: str = "lw",
    n_permutations: int = 5000,
    seed: int = 42,
) -> PermutationResult:
    """Permutation test for MCCA canonical correlations.

    Shuffles rows of views 1..K independently (view 0 fixed) to create
    a null distribution, then computes empirical p-values.

    Optimised: within-view covariance is invariant to row permutation, so it
    is computed once and reused. Only the cross-covariance blocks are recomputed
    per permutation.

    Parameters
    ----------
    Xs : list of ndarray
        Original per-view data matrices.
    observed_correlations : ndarray
        Observed canonical correlations from run_mcca.
    n_components : int
        Number of components.
    regs : str
        Regularisation method.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    PermutationResult
    """
    rng = np.random.RandomState(seed)
    K = len(Xs)
    n = Xs[0].shape[0]

    # PCA reduction for speed (same transform as run_mcca)
    Xs_reduced, _ = _pca_reduce_views(Xs)

    dims = [X.shape[1] for X in Xs_reduced]
    total_d = sum(dims)
    n_comp = len(observed_correlations)
    offsets = np.cumsum([0] + dims)

    # Pre-compute within-view covariance (invariant to row permutation)
    C_within = np.zeros((total_d, total_d))
    for i in range(K):
        si, ei = offsets[i], offsets[i + 1]
        C_within[si:ei, si:ei] = _regularise_cov(Xs_reduced[i], method=regs)

    null_dist = np.zeros((n_permutations, n_comp))

    for perm_i in range(n_permutations):
        if (perm_i + 1) % 500 == 0:
            logger.info("  Permutation %d / %d", perm_i + 1, n_permutations)

        # Shuffle views 1..K, keep view 0 fixed
        Xs_perm = [Xs_reduced[0]]
        for k in range(1, K):
            perm_idx = rng.permutation(n)
            Xs_perm.append(Xs_reduced[k][perm_idx])

        try:
            _, eigvecs = _solve_mcca_eigenvalue(
                Xs_perm, n_comp, C_within, offsets
            )
            null_dist[perm_i] = _compute_canonical_correlations(
                Xs_perm, eigvecs, offsets, n_comp
            )
        except Exception:
            null_dist[perm_i] = 0.0

    # Empirical p-values: (n_null >= observed + 1) / (n_perm + 1)
    p_values = np.zeros(n_comp)
    for c in range(n_comp):
        p_values[c] = (np.sum(null_dist[:, c] >= observed_correlations[c]) + 1) / (
            n_permutations + 1
        )

    return PermutationResult(
        observed=observed_correlations,
        null_distributions=null_dist,
        p_values=p_values,
        n_permutations=n_permutations,
    )


def test_dose_association(
    scores: List[np.ndarray],
    dose_labels: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> DoseAssociationResult:
    """Test association between MCCA canonical variates and ordinal dose.

    Two-stage approach: average MCCA scores across views per component,
    then compute Spearman correlation with ordinal dose. Permutation p-values
    by shuffling dose labels.

    Parameters
    ----------
    scores : list of ndarray
        Per-view MCCA scores, each shape [n_samples, n_components].
    dose_labels : ndarray
        Ordinal dose labels (C=0, L=1, M=2, H=3), shape [n_samples].
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    DoseAssociationResult
    """
    from scipy.stats import spearmanr

    # Average scores across views
    avg_scores = np.mean(scores, axis=0)  # [n_samples, n_components]
    n_comp = avg_scores.shape[1]

    # Observed Spearman correlations
    observed_rho = np.zeros(n_comp)
    for c in range(n_comp):
        rho, _ = spearmanr(avg_scores[:, c], dose_labels)
        observed_rho[c] = rho

    # Permutation test
    rng = np.random.RandomState(seed)
    null_rho = np.zeros((n_permutations, n_comp))

    for perm_i in range(n_permutations):
        perm_dose = rng.permutation(dose_labels)
        for c in range(n_comp):
            rho, _ = spearmanr(avg_scores[:, c], perm_dose)
            null_rho[perm_i, c] = rho

    # Two-tailed p-values
    p_values = np.zeros(n_comp)
    for c in range(n_comp):
        p_values[c] = (
            np.sum(np.abs(null_rho[:, c]) >= np.abs(observed_rho[c])) + 1
        ) / (n_permutations + 1)

    return DoseAssociationResult(
        spearman_rho=observed_rho,
        p_values=p_values,
        n_permutations=n_permutations,
    )


def test_group_differences(
    scores: List[np.ndarray],
    group_labels: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> Dict:
    """PERMANOVA on MCCA score space for group separability.

    Tests whether dose groups are separable in canonical variate space
    using Euclidean distance-based PERMANOVA.

    Parameters
    ----------
    scores : list of ndarray
        Per-view MCCA scores.
    group_labels : ndarray
        Group labels (integer-encoded).
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: pseudo_f, p_value, r_squared
    """
    # Average scores across views
    avg_scores = np.mean(scores, axis=0)

    # Compute distance matrix
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(avg_scores, metric="euclidean"))
    D_sq = D ** 2
    n = len(group_labels)
    groups = np.unique(group_labels)
    k = len(groups)

    def pseudo_f_statistic(D_sq, labels):
        n = len(labels)
        groups = np.unique(labels)
        k = len(groups)

        # Total sum of squares
        ss_total = np.sum(D_sq) / (2 * n)

        # Within-group sum of squares
        ss_within = 0.0
        for g in groups:
            mask = labels == g
            n_g = mask.sum()
            if n_g > 1:
                ss_within += np.sum(D_sq[np.ix_(mask, mask)]) / (2 * n_g)

        ss_between = ss_total - ss_within

        # Pseudo-F
        if ss_within == 0:
            return 0.0, 0.0
        f_stat = (ss_between / (k - 1)) / (ss_within / (n - k))
        r_sq = ss_between / ss_total
        return f_stat, r_sq

    observed_f, observed_r2 = pseudo_f_statistic(D_sq, group_labels)

    # Permutation test
    rng = np.random.RandomState(seed)
    null_f = np.zeros(n_permutations)

    for perm_i in range(n_permutations):
        perm_labels = rng.permutation(group_labels)
        null_f[perm_i], _ = pseudo_f_statistic(D_sq, perm_labels)

    p_value = (np.sum(null_f >= observed_f) + 1) / (n_permutations + 1)

    return {
        "pseudo_f": float(observed_f),
        "r_squared": float(observed_r2),
        "p_value": float(p_value),
    }


@dataclass
class SexTestResult:
    """Result container for sex difference test on MCCA scores."""

    permanova: Dict  # {pseudo_f, r_squared, p_value}
    per_component: List[Dict]  # [{cohens_d, p_value}, ...] per CV
    n_male: int
    n_female: int


def test_sex_differences(
    scores: List[np.ndarray],
    sex_labels: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> SexTestResult:
    """Test sex differences in MCCA canonical variate space.

    Runs PERMANOVA for overall separability and per-CV Cohen's d with
    permutation p-values.

    Parameters
    ----------
    scores : list of ndarray
        Per-view MCCA scores, each shape [n_samples, n_components].
    sex_labels : ndarray
        Sex labels (str 'M'/'F'), shape [n_samples].
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    SexTestResult
    """
    # Encode sex as integer: M=0, F=1
    sex_int = np.array([0 if s == "M" else 1 for s in sex_labels])
    n_male = int((sex_int == 0).sum())
    n_female = int((sex_int == 1).sum())

    logger.info("Sex groups: M=%d, F=%d", n_male, n_female)

    # PERMANOVA on scores
    permanova = test_group_differences(
        scores, sex_int, n_permutations=n_permutations, seed=seed,
    )

    # Average scores across views
    avg_scores = np.mean(scores, axis=0)  # [n_samples, n_components]
    n_comp = avg_scores.shape[1]

    # Per-CV Cohen's d and permutation p-values
    rng = np.random.RandomState(seed)

    def _cohens_d(vals, labels):
        m0 = vals[labels == 0]
        m1 = vals[labels == 1]
        n0, n1 = len(m0), len(m1)
        if n0 < 2 or n1 < 2:
            return 0.0
        pooled_sd = np.sqrt(
            ((n0 - 1) * np.var(m0, ddof=1) + (n1 - 1) * np.var(m1, ddof=1))
            / (n0 + n1 - 2)
        )
        if pooled_sd == 0:
            return 0.0
        return (np.mean(m0) - np.mean(m1)) / pooled_sd

    per_component = []
    for c in range(n_comp):
        obs_d = _cohens_d(avg_scores[:, c], sex_int)

        # Permutation p-value (two-tailed)
        null_d = np.zeros(n_permutations)
        for perm_i in range(n_permutations):
            perm_labels = rng.permutation(sex_int)
            null_d[perm_i] = _cohens_d(avg_scores[:, c], perm_labels)

        p_val = (np.sum(np.abs(null_d) >= np.abs(obs_d)) + 1) / (n_permutations + 1)
        per_component.append({"cohens_d": float(obs_d), "p_value": float(p_val)})

        logger.info(
            "  CV%d sex: d=%.4f, p=%.4f%s",
            c + 1, obs_d, p_val, " *" if p_val < 0.05 else "",
        )

    return SexTestResult(
        permanova=permanova,
        per_component=per_component,
        n_male=n_male,
        n_female=n_female,
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_mcca_paths(
    config_path: Path | None = None,
    roi_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ROI and MCCA output paths from config or args.

    Priority: explicit arguments override config values.

    Parameters
    ----------
    config_path : Path, optional
        Study config.yaml.  Derives roi_dir and output_dir from
        ``paths.network.roi`` and ``paths.network.mcca``.
    roi_dir : Path, optional
        Explicit ROI directory override.
    output_dir : Path, optional
        Explicit MCCA output directory override.

    Returns
    -------
    (roi_dir, output_dir) : tuple of Path
    """
    cfg_roi = None
    cfg_output = None

    if config_path is not None:
        from neurofaune.config import load_config, get_config_value

        config = load_config(Path(config_path))
        cfg_roi = get_config_value(config, "paths.network.roi")
        cfg_output = get_config_value(config, "paths.network.mcca")
        if cfg_roi is not None:
            cfg_roi = Path(cfg_roi)
        if cfg_output is not None:
            cfg_output = Path(cfg_output)

    resolved_roi = roi_dir if roi_dir is not None else cfg_roi
    resolved_output = output_dir if output_dir is not None else cfg_output

    if resolved_roi is None:
        raise ValueError(
            "ROI directory not specified. Provide config_path or roi_dir."
        )
    if resolved_output is None:
        raise ValueError(
            "Output directory not specified. Provide config_path or output_dir."
        )

    return Path(resolved_roi), Path(resolved_output)


def parse_views(view_specs: list) -> dict:
    """Parse view specifications like ``'dwi:FA,MD,AD,RD'`` into a dict.

    Parameters
    ----------
    view_specs : list of str
        Each element is ``'view_name:metric1,metric2,...'``.

    Returns
    -------
    dict mapping view_name -> list of metric names
    """
    views = {}
    for spec in view_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid view spec '{spec}'. Expected 'name:metric1,metric2,...'")
        name, metrics_str = spec.split(":", 1)
        metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]
        if not metrics:
            raise ValueError(f"No metrics specified for view '{name}'")
        views[name] = metrics
    return views


# ---------------------------------------------------------------------------
# MCCAAnalysis
# ---------------------------------------------------------------------------


class MCCAAnalysis:
    """Multi-modal Canonical Correlation Analysis.

    Follows the same pattern as ``ClassificationAnalysis`` and
    ``RegressionAnalysis``: resolve paths from config, check for existing
    results before running, and provide a clean Python API that scripts
    can wrap.

    Typical usage::

        from neurofaune.network.mcca import MCCAAnalysis

        analysis = MCCAAnalysis.prepare(
            config_path=Path("config.yaml"),
            views={"dwi": ["FA","MD","AD","RD"], "msme": ["MWF","IWF"]},
            force=True,
        )
        analysis.run(cohort="p30", feature_set="bilateral")
    """

    def __init__(
        self,
        roi_dir: Path,
        output_dir: Path,
        views: Dict[str, List[str]],
        exclusion_csv: Path | None = None,
        confounds: List[str] | None = None,
        target: str = "dose",
        auc_csv: Path | None = None,
        force: bool = False,
    ):
        self.roi_dir = Path(roi_dir)
        self.output_dir = Path(output_dir)
        self.views = views
        self.exclusion_csv = exclusion_csv
        self.confounds = confounds
        self.target = target
        self.auc_csv = auc_csv
        self.force = force

    @classmethod
    def prepare(
        cls,
        config_path: Path | None = None,
        roi_dir: Path | None = None,
        output_dir: Path | None = None,
        views: Dict[str, List[str]] | None = None,
        exclusion_csv: Path | None = None,
        confounds: List[str] | None = None,
        target: str = "dose",
        auc_csv: Path | None = None,
        force: bool = False,
    ) -> "MCCAAnalysis":
        """Prepare an MCCA analysis.

        Parameters
        ----------
        config_path : Path, optional
            Study config.yaml.  Derives roi_dir and output_dir from
            ``paths.network.roi`` and ``paths.network.mcca``.
        roi_dir, output_dir : Path, optional
            Explicit path overrides.
        views : dict
            Mapping of view_name -> list of metric names.
            E.g. ``{"dwi": ["FA","MD","AD","RD"], "func": ["fALFF","ReHo"]}``.
        exclusion_csv : Path, optional
            CSV of sessions to exclude.
        confounds : list of str, optional
            Metadata columns to residualize before MCCA (e.g. ``["sex"]``).
        target : str
            Target for dose association: ``"dose"`` (ordinal C=0..H=3),
            ``"auc"`` (continuous AUC), or ``"log_auc"``.
        auc_csv : Path, optional
            Path to AUC lookup CSV (used when target is ``"auc"`` or
            ``"log_auc"``).
        force : bool
            If True, delete existing results before running.
        """
        resolved_roi, resolved_output = _resolve_mcca_paths(
            config_path, roi_dir, output_dir
        )

        if views is None:
            raise ValueError("views must be specified")

        return cls(
            roi_dir=resolved_roi,
            output_dir=resolved_output,
            views=views,
            exclusion_csv=exclusion_csv,
            confounds=confounds,
            target=target,
            auc_csv=auc_csv,
            force=force,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _result_dir(
        self, cohort: str | None = None, feature_set: str = "bilateral"
    ) -> Path:
        """Output dir for one combo: ``{output_dir}/{cohort_label}/{feature_set}/``."""
        cohort_label = cohort or "pooled"
        return self.output_dir / cohort_label / feature_set

    def _check_or_clear(
        self, cohort: str | None = None, feature_set: str = "bilateral"
    ) -> None:
        """Check for existing results; error unless force is set.

        If ``self.force`` is True, removes the target directory for a clean
        slate.  If False and the directory has result files, raises
        ``FileExistsError``.
        """
        import shutil

        target = self._result_dir(cohort, feature_set)
        if not target.exists():
            return

        result_files = [f for f in target.rglob("*") if f.is_file()]
        if not result_files:
            return

        if not self.force:
            file_list = "\n  ".join(str(f) for f in result_files[:10])
            extra = (
                f"\n  ... and {len(result_files) - 10} more"
                if len(result_files) > 10
                else ""
            )
            raise FileExistsError(
                f"Results already exist at {target} "
                f"({len(result_files)} files):\n  {file_list}{extra}\n\n"
                f"Use force=True (or --force) to delete existing results "
                f"and rerun."
            )

        logger.warning("--force: removing existing results at %s", target)
        shutil.rmtree(target)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        cohort: str | None = None,
        feature_set: str = "bilateral",
        n_components: int = 5,
        regs: str = "lw",
        n_permutations: int = 5000,
        seed: int = 42,
    ) -> dict:
        """Run MCCA pipeline for one cohort/feature_set combination.

        Parameters
        ----------
        cohort : str, optional
            PND cohort filter (e.g. ``"p30"``).  None = pooled.
        feature_set : str
            Feature set name (``"bilateral"`` or ``"territory"``).
        n_components : int
            Number of canonical components to extract.
        regs : str
            Regularisation method: ``"lw"`` (Ledoit-Wolf), ``"identity"``,
            or a float string.
        n_permutations : int
            Number of permutations for significance testing.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Summary dictionary with status, metrics, and results.
        """
        import json

        from neurofaune.network.mcca_visualization import (
            plot_canonical_correlations,
            plot_cross_view_loadings,
            plot_loadings_heatmap,
            plot_permutation_null,
            plot_scores_by_cohort,
            plot_scores_by_dose,
            plot_scores_by_sex,
        )

        self._check_or_clear(cohort, feature_set)

        cohort_label = cohort or "pooled"
        combo_dir = self._result_dir(cohort, feature_set)
        combo_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "\n%s\n  MCCA | Cohort: %s | Features: %s\n%s",
            "=" * 60, cohort_label, feature_set, "=" * 60,
        )

        # Phase 1: Load multi-view data
        logger.info("[Phase 1] Loading multi-view data...")
        try:
            Xs, view_names, feature_names, metadata = load_multiview_data(
                roi_dir=self.roi_dir,
                views=self.views,
                feature_set=feature_set,
                cohort_filter=cohort if cohort else None,
                exclusion_csv=self.exclusion_csv,
                confounds=self.confounds,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping %s/%s: %s", cohort_label, feature_set, exc)
            return {"status": "skipped", "reason": str(exc)}

        n_samples = Xs[0].shape[0]
        if n_samples < 10:
            logger.warning("Too few samples (n=%d) — skipping", n_samples)
            return {"status": "skipped", "reason": f"n={n_samples} too small"}

        # Encode dose labels (always needed for coloring/group sizes)
        dose_order = ["C", "L", "M", "H"]
        dose_map = {d: i for i, d in enumerate(dose_order)}
        dose_labels = np.array([dose_map.get(d, -1) for d in metadata["dose"].values])
        valid = dose_labels >= 0
        if not valid.all():
            logger.warning("Dropping %d samples with unknown dose", (~valid).sum())
            Xs = [X[valid] for X in Xs]
            metadata = metadata[valid].reset_index(drop=True)
            dose_labels = dose_labels[valid]
            n_samples = len(dose_labels)

        label_names = [d for d in dose_order if d in set(metadata["dose"].values)]

        # Build target array for dose association test
        if self.target in ("auc", "log_auc"):
            value_col = "log_auc" if self.target == "log_auc" else "auc"

            if self.auc_csv is not None:
                import pandas as pd
                auc_df = pd.read_csv(self.auc_csv)
                auc_lookup = {}
                for _, r in auc_df.iterrows():
                    auc_lookup[(r["subject"], r["session"])] = r.get(value_col, np.nan)
                target_values = np.array([
                    auc_lookup.get((row["subject"], row["session"]), np.nan)
                    for _, row in metadata.iterrows()
                ], dtype=float)
            elif value_col in metadata.columns:
                target_values = metadata[value_col].values.astype(float)
            else:
                _session_to_auc = {
                    "ses-p30": "AUC_p30", "ses-p60": "AUC_p60", "ses-p90": "AUC_p90",
                }
                auc_values = []
                for _, row in metadata.iterrows():
                    auc_col = _session_to_auc.get(row["session"])
                    if auc_col and auc_col in metadata.columns:
                        auc_values.append(row.get(auc_col, np.nan))
                    else:
                        auc_values.append(np.nan)
                target_values = np.array(auc_values, dtype=float)
                if self.target == "log_auc":
                    target_values = np.log1p(target_values)

            valid_target = ~np.isnan(target_values)
            if not valid_target.all():
                n_drop = (~valid_target).sum()
                logger.warning("Dropping %d samples with NaN %s", n_drop, self.target)
                Xs = [X[valid_target] for X in Xs]
                metadata = metadata[valid_target].reset_index(drop=True)
                dose_labels = dose_labels[valid_target]
                target_values = target_values[valid_target]
                n_samples = len(target_values)
            target_name = "log(1+AUC)" if self.target == "log_auc" else "AUC"
        else:
            target_values = dose_labels.astype(float)
            target_name = "Ordinal dose"

        summary = {
            "status": "completed",
            "cohort": cohort_label,
            "feature_set": feature_set,
            "target": self.target,
            "target_name": target_name,
            "n_samples": n_samples,
            "view_dims": {vn: X.shape[1] for vn, X in zip(view_names, Xs)},
            "group_sizes": {
                name: int((dose_labels == dose_map[name]).sum()) for name in label_names
            },
            "confounds_residualized": self.confounds if self.confounds else None,
        }

        # Phase 2: Fit MCCA
        logger.info("[Phase 2] Fitting MCCA (n_components=%d, regs=%s)...", n_components, regs)
        actual_n_comp = min(n_components, n_samples - 1, min(X.shape[1] for X in Xs))
        result = run_mcca(Xs, n_components=actual_n_comp, regs=regs)
        result.view_names = view_names
        result.feature_names = feature_names

        summary["n_components"] = result.n_components
        summary["canonical_correlations"] = result.canonical_correlations.tolist()

        logger.info(
            "Canonical correlations: %s",
            ", ".join(f"CV{i+1}={r:.4f}" for i, r in enumerate(result.canonical_correlations)),
        )

        # Phase 3: Permutation test
        logger.info("[Phase 3] Permutation test (%d permutations)...", n_permutations)
        perm_result = permutation_test_mcca(
            Xs, result.canonical_correlations,
            n_components=result.n_components,
            regs=regs,
            n_permutations=n_permutations,
            seed=seed,
        )
        summary["permutation_p_values"] = perm_result.p_values.tolist()

        for i in range(result.n_components):
            logger.info(
                "  CV%d: r=%.4f, p=%.4f%s",
                i + 1, result.canonical_correlations[i], perm_result.p_values[i],
                " *" if perm_result.p_values[i] < 0.05 else "",
            )

        # Phase 4: Target association (dose or AUC)
        logger.info("[Phase 4] Testing %s association in canonical variate space...", target_name)
        dose_result = test_dose_association(
            result.scores, target_values,
            n_permutations=n_permutations, seed=seed,
        )
        assoc_key = "auc_association" if self.target == "auc" else "dose_association"
        summary[assoc_key] = {
            f"CV{i+1}": {
                "spearman_rho": float(dose_result.spearman_rho[i]),
                "p_value": float(dose_result.p_values[i]),
            }
            for i in range(result.n_components)
        }

        for i in range(result.n_components):
            logger.info(
                "  CV%d ~ %s: rho=%.4f, p=%.4f%s",
                i + 1, target_name, dose_result.spearman_rho[i], dose_result.p_values[i],
                " *" if dose_result.p_values[i] < 0.05 else "",
            )

        # Phase 5: PERMANOVA on scores
        logger.info("[Phase 5] PERMANOVA on MCCA score space...")
        permanova = test_group_differences(
            result.scores, dose_labels,
            n_permutations=min(n_permutations, 9999), seed=seed,
        )
        summary["permanova"] = permanova
        logger.info(
            "  PERMANOVA: F=%.4f, R²=%.4f, p=%.4f",
            permanova["pseudo_f"], permanova["r_squared"], permanova["p_value"],
        )

        # Phase 5b: Sex differences
        if "sex" in metadata.columns and metadata["sex"].nunique() == 2:
            logger.info("[Phase 5b] Testing sex differences in canonical variate space...")
            sex_result = test_sex_differences(
                result.scores, metadata["sex"].values,
                n_permutations=n_permutations, seed=seed,
            )
            summary["sex_differences"] = {
                "permanova": sex_result.permanova,
                "n_male": sex_result.n_male,
                "n_female": sex_result.n_female,
                "per_component": {
                    f"CV{i+1}": sex_result.per_component[i]
                    for i in range(result.n_components)
                },
            }
            logger.info(
                "  Sex PERMANOVA: F=%.4f, R²=%.4f, p=%.4f",
                sex_result.permanova["pseudo_f"],
                sex_result.permanova["r_squared"],
                sex_result.permanova["p_value"],
            )

        # Phase 6: Visualisations
        logger.info("[Phase 6] Generating visualisations...")

        plot_canonical_correlations(
            result, perm_result,
            title=f"Canonical Correlations — {cohort_label}",
            out_path=combo_dir / "canonical_correlations.png",
        )

        plot_scores_by_dose(
            result, dose_labels, label_names,
            title=f"MCCA Scores by Dose — {cohort_label}",
            out_path=combo_dir / "scores_by_dose.png",
        )

        plot_scores_by_cohort(
            result, metadata["cohort"].values,
            title=f"MCCA Scores by Cohort — {cohort_label}",
            out_path=combo_dir / "scores_by_cohort.png",
        )

        if "sex" in metadata.columns and metadata["sex"].nunique() == 2:
            plot_scores_by_sex(
                result, metadata["sex"].values,
                title=f"MCCA Scores by Sex — {cohort_label}",
                out_path=combo_dir / "scores_by_sex.png",
            )

        for k, vn in enumerate(view_names):
            plot_loadings_heatmap(
                result, view_idx=k,
                title=f"Top Loadings — {vn} — {cohort_label}",
                out_path=combo_dir / f"loadings_{vn}.png",
            )

        plot_cross_view_loadings(
            result, component=0,
            title=f"Cross-View Loadings CV1 — {cohort_label}",
            out_path=combo_dir / "cross_view_loadings_cv1.png",
        )

        plot_permutation_null(
            perm_result,
            title=f"Permutation Null — {cohort_label}",
            out_path=combo_dir / "permutation_null.png",
        )

        # Save detailed results
        mcca_json = {
            "n_samples": n_samples,
            "n_components": result.n_components,
            "regularisation": regs,
            "confounds_residualized": self.confounds if self.confounds else None,
            "view_names": view_names,
            "view_dims": {vn: X.shape[1] for vn, X in zip(view_names, Xs)},
            "canonical_correlations": result.canonical_correlations.tolist(),
            "permutation_p_values": perm_result.p_values.tolist(),
            assoc_key: summary[assoc_key],
            "permanova": permanova,
            "group_sizes": summary["group_sizes"],
        }
        if "sex_differences" in summary:
            mcca_json["sex_differences"] = summary["sex_differences"]
        with open(combo_dir / "mcca_results.json", "w") as f:
            json.dump(mcca_json, f, indent=2)

        # Save per-combo summary
        with open(combo_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    # ------------------------------------------------------------------
    # Design description
    # ------------------------------------------------------------------

    def write_design_description(
        self,
        feature_sets: list[str],
        n_components: int = 5,
        regs: str = "lw",
        n_permutations: int = 5000,
        seed: int = 42,
    ) -> None:
        """Write human-readable MCCA analysis description to output_dir.

        Parameters
        ----------
        feature_sets : list[str]
            Feature sets being analysed (e.g. ``["bilateral"]``).
        n_components : int
            Number of canonical components.
        regs : str
            Regularisation method.
        n_permutations : int
            Number of permutations.
        seed : int
            Random seed.
        """
        from datetime import datetime

        target_desc = (
            "AUC" if self.target in ("auc", "log_auc")
            else "Dose"
        )

        lines = [
            "ANALYSIS DESCRIPTION",
            "====================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Analysis: Multiset Canonical Correlation Analysis (MCCA)",
            "",
            "DATA SOURCE",
            "-----------",
            f"ROI directory: {self.roi_dir}",
            f"Exclusion list: {self.exclusion_csv or 'None'}",
            f"Feature sets: {', '.join(feature_sets)}",
            "",
            "VIEWS",
            "-----",
        ]

        for vn, metrics in self.views.items():
            lines.append(f"- {vn}: {', '.join(metrics)}")

        lines.extend([
            "",
            "EXPERIMENTAL DESIGN",
            "-------------------",
            "Grouping: Dose (C, L, M, H — 4 groups)",
            "Cohorts analysed: p30, p60, p90, and pooled",
            "",
            "STATISTICAL METHODS",
            "-------------------",
            "1. Regularised MCCA",
            f"   - Regularisation: {regs}",
            f"   - Components: {n_components}",
            "   - Generalised eigenvalue decomposition on block covariance matrices",
            "   - Subjects intersected across all views",
            "",
            "2. Permutation test for canonical correlations",
            f"   - {n_permutations} permutations (shuffle views 1..K, fix view 0)",
            "   - Empirical p-value per component",
            "",
            f"3. {target_desc} association test",
            "   - Average MCCA scores across views per component",
            f"   - Spearman correlation with {'continuous AUC (session-matched)' if self.target in ('auc', 'log_auc') else 'ordinal dose (C=0, L=1, M=2, H=3)'}",
            f"   - {n_permutations} permutation p-values (two-tailed)",
            "",
            "4. PERMANOVA on MCCA score space",
            "   - Euclidean distances on average canonical variate scores",
            "   - Tests group separability in fused multi-modal space",
            "",
            "5. Sex differences test (when sex data available)",
            "   - PERMANOVA on MCCA score space by sex",
            "   - Per-CV Cohen's d with permutation p-values",
            "",
            "PREPROCESSING",
            "-------------",
            f"- Confounds residualized: {self.confounds or 'None'}",
            "- Z-score standardisation per view (independent)",
            "- Median imputation for NaN values",
            "- ROIs with >20% zeros excluded",
            "- Subject intersection across all views",
            "",
            "PARAMETERS",
            "----------",
            f"Components: {n_components}",
            f"Regularisation: {regs}",
            f"Permutations: {n_permutations}",
            f"Random seed: {seed}",
        ])

        output_path = self.output_dir / "design_description.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info("Saved analysis description: %s", output_path)
