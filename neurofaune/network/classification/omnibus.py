"""
Omnibus multivariate tests for group discrimination.

PERMANOVA (Permutational MANOVA) is the primary test — non-parametric,
handles p > n, and requires no distributional assumptions. MANOVA is
available as an optional supplement when statsmodels is installed.
"""

import logging
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def _compute_pseudo_f(dist_sq: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute pseudo-F statistic from a squared distance matrix.

    Implements the PERMANOVA test statistic (Anderson, 2001):
    F = (SS_between / (k-1)) / (SS_within / (n-k))

    Parameters
    ----------
    dist_sq : ndarray, shape (n, n)
        Squared pairwise Euclidean distance matrix.
    y : ndarray, shape (n,)
        Integer group labels.

    Returns
    -------
    pseudo_f : float
        Pseudo-F statistic.
    r_squared : float
        Proportion of total variation explained by groups.
    """
    n = len(y)
    labels = np.unique(y)
    k = len(labels)

    # Total sum of squares: sum of squared distances / n
    ss_total = dist_sq.sum() / (2 * n)

    # Within-group sum of squares
    ss_within = 0.0
    for label in labels:
        mask = y == label
        ni = mask.sum()
        if ni < 2:
            continue
        group_dists = dist_sq[np.ix_(mask, mask)]
        ss_within += group_dists.sum() / (2 * ni)

    ss_between = ss_total - ss_within

    # Avoid division by zero
    if ss_within == 0 or (n - k) == 0:
        return 0.0, 0.0

    pseudo_f = (ss_between / (k - 1)) / (ss_within / (n - k))
    r_squared = ss_between / ss_total if ss_total > 0 else 0.0

    return pseudo_f, r_squared


def run_permanova(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    n_perm: int = 9999,
    seed: int = 42,
) -> dict:
    """PERMANOVA (Permutational Multivariate Analysis of Variance).

    Tests whether centroids differ among groups using Euclidean distances
    and permutation-based inference.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix (should be standardised).
    y : ndarray, shape (n_samples,)
        Integer group labels.
    label_names : sequence of str
        Group names.
    n_perm : int
        Number of permutations (default 9999).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        pseudo_f : float — observed pseudo-F statistic
        p_value : float — permutation p-value
        r_squared : float — proportion of variance explained
        n_permutations : int
        null_distribution : ndarray — null pseudo-F values
        group_sizes : dict — n per group
    """
    rng = np.random.default_rng(seed)

    # Compute squared Euclidean distance matrix
    dists = pdist(X, metric="euclidean")
    dist_sq = squareform(dists ** 2)

    # Observed statistic
    f_obs, r2_obs = _compute_pseudo_f(dist_sq, y)

    # Permutation test
    null_f = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null_f[i], _ = _compute_pseudo_f(dist_sq, y_perm)

    # p-value: proportion of null >= observed (include observed in count)
    p_value = (np.sum(null_f >= f_obs) + 1) / (n_perm + 1)

    group_sizes = {}
    for i, name in enumerate(label_names):
        group_sizes[name] = int((y == i).sum())

    logger.info(
        "PERMANOVA: F=%.3f, R²=%.4f, p=%.4f (n_perm=%d)",
        f_obs, r2_obs, p_value, n_perm,
    )

    return {
        "pseudo_f": float(f_obs),
        "p_value": float(p_value),
        "r_squared": float(r2_obs),
        "n_permutations": n_perm,
        "null_distribution": null_f,
        "group_sizes": group_sizes,
    }


def run_manova(
    X: np.ndarray,
    y: np.ndarray,
    label_names: Sequence[str],
    feature_names: Sequence[str],
) -> Optional[dict]:
    """Parametric MANOVA (optional, requires statsmodels).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Integer group labels.
    label_names : sequence of str
        Group names.
    feature_names : sequence of str
        Feature names.

    Returns
    -------
    dict or None
        If statsmodels is available: dict with Pillai's trace, Wilks' lambda,
        F-statistic, p-value, and test name. Otherwise None.
    """
    try:
        import pandas as pd
        from statsmodels.multivariate.manova import MANOVA
    except ImportError:
        logger.info("statsmodels not installed — skipping MANOVA")
        return None

    # Build DataFrame for statsmodels formula interface
    n_samples, n_features = X.shape
    if n_features > n_samples - len(np.unique(y)):
        logger.warning(
            "MANOVA: more features (%d) than residual df (%d) — skipping",
            n_features, n_samples - len(np.unique(y)),
        )
        return None

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["group"] = [label_names[yi] for yi in y]

    dep_vars = " + ".join(f"f{i}" for i in range(n_features))
    formula = f"{dep_vars} ~ group"

    try:
        mv = MANOVA.from_formula(formula, data=df)
        result = mv.mv_test()
        # Extract Pillai's trace (most robust to violations)
        group_result = result.results["group"]["stat"]
        pillai = group_result.loc["Pillai's trace"]

        return {
            "test": "MANOVA (Pillai's trace)",
            "pillai_trace": float(pillai["Value"]),
            "f_statistic": float(pillai["F Value"]),
            "df_num": float(pillai["Num DF"]),
            "df_den": float(pillai["Den DF"]),
            "p_value": float(pillai["Pr > F"]),
        }
    except Exception as exc:
        logger.warning("MANOVA failed: %s", exc)
        return None
