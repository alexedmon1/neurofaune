"""
Network distance tests for covariance network comparison.

Complements edge-level (NBS) and graph-metric approaches with holistic tests
that quantify global differences between group covariance networks.

Two analysis modes:
    - **Absolute distance** (``abs_distance_test``): directly compares two
      groups' networks.
    - **Relative distance** (``rel_distance_test``): compares how far two
      groups each are from a reference network (e.g., older controls).

Three distance statistics (used in both modes):
    1. Mantel test — Pearson correlation between vectorized upper triangles.
    2. Frobenius distance — L2 norm of the difference between upper triangles.
    3. Spectral divergence — L2 distance between sorted eigenvalue spectra.

All use a shared permutation framework: pool subjects, shuffle labels, split,
recompute group correlation matrices, recompute statistics, build null
distributions.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from neurofaune.network.matrices import default_dose_comparisons, spearman_matrix

logger = logging.getLogger(__name__)


def _fast_spearman(data: np.ndarray) -> np.ndarray:
    """Vectorized Spearman via scipy.stats.rankdata + np.corrcoef.

    Faster than spearman_matrix() for clean data (no NaN handling) because
    it avoids the Python-loop overhead of np.apply_along_axis.
    """
    ranked = stats.rankdata(data, axis=0)
    corr = np.corrcoef(ranked, rowvar=False)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


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


def abs_distance_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict:
    """Absolute network distance: compare two groups directly.

    Computes all three distance statistics (Mantel, Frobenius, Spectral)
    between two groups' covariance networks with permutation p-values.

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

        corr_pa = _fast_spearman(perm_a)
        corr_pb = _fast_spearman(perm_b)

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


def rel_distance_test(
    data_dose: np.ndarray,
    data_early_control: np.ndarray,
    data_late_control: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
    distance_fn: str = "frobenius",
) -> dict:
    """Relative network distance: compare two groups' distances to a reference.

    Tests whether a group's covariance network is shifted toward (or away
    from) a reference network relative to a comparison group.

    Compares d(group_A, reference) vs d(group_B, reference).
    If Δ = d(A, ref) − d(B, ref) < 0, group A is *closer* to the reference.
    When the reference is older controls, Δ < 0 suggests accelerated
    development; Δ > 0 suggests decelerated development.

    Permutation null: shuffle A/B labels (keeping reference fixed) and
    recompute Δ.

    Parameters
    ----------
    data_dose : ndarray (n_dose, n_rois)
        ROI values for group A (e.g., dosed group at the early timepoint).
    data_early_control : ndarray (n_ctrl, n_rois)
        ROI values for group B (e.g., controls at the early timepoint).
    data_late_control : ndarray (n_ref, n_rois)
        ROI values for the reference group (e.g., controls at a later
        timepoint).
    n_perm : int
        Number of permutations for the null distribution.
    seed : int
        Random seed.
    distance_fn : str
        Distance metric: "frobenius", "spectral", or "mantel".
        For Mantel, the delta is *negated* (1 − r) so that lower = more similar.

    Returns
    -------
    result : dict
        Keys: delta (observed), p_accelerated (one-sided: Δ < 0),
        p_decelerated (one-sided: Δ > 0), d_dose_to_ref, d_ctrl_to_ref,
        null_deltas, distance_fn, n_dose, n_ctrl, n_ref.
    """
    rng = np.random.default_rng(seed)

    if distance_fn == "frobenius":
        dist = frobenius_distance
    elif distance_fn == "spectral":
        dist = spectral_divergence
    elif distance_fn == "mantel":
        # For Mantel, convert correlation to distance: d = 1 - r
        def dist(a, b):
            r = mantel_test(a, b)
            return 1.0 - r if not np.isnan(r) else float("nan")
    else:
        raise ValueError(f"Unknown distance_fn: {distance_fn!r}")

    n_dose = data_dose.shape[0]
    n_ctrl = data_early_control.shape[0]

    # Reference covariance matrix (fixed across permutations)
    corr_ref = spearman_matrix(data_late_control)

    # Observed
    corr_dose = spearman_matrix(data_dose)
    corr_ctrl = spearman_matrix(data_early_control)

    d_dose_ref = dist(corr_dose, corr_ref)
    d_ctrl_ref = dist(corr_ctrl, corr_ref)
    obs_delta = d_dose_ref - d_ctrl_ref

    logger.info(
        f"  Observed: d(dose,ref)={d_dose_ref:.4f}, d(ctrl,ref)={d_ctrl_ref:.4f}, "
        f"Δ={obs_delta:.4f} ({'accelerated' if obs_delta < 0 else 'decelerated'})"
    )

    # Permutation: shuffle dose/control labels within early timepoint
    pooled_early = np.vstack([data_dose, data_early_control])
    n_early = n_dose + n_ctrl
    null_deltas = np.zeros(n_perm)

    for p in range(n_perm):
        perm_idx = rng.permutation(n_early)
        perm_dose = pooled_early[perm_idx[:n_dose]]
        perm_ctrl = pooled_early[perm_idx[n_dose:]]

        corr_pd = _fast_spearman(perm_dose)
        corr_pc = _fast_spearman(perm_ctrl)

        d_pd = dist(corr_pd, corr_ref)
        d_pc = dist(corr_pc, corr_ref)
        null_deltas[p] = d_pd - d_pc

    valid = null_deltas[~np.isnan(null_deltas)]
    if len(valid) == 0:
        p_accel = float("nan")
        p_decel = float("nan")
    else:
        # Accelerated: observed Δ is unusually negative
        p_accel = float(np.mean(valid <= obs_delta))
        # Decelerated: observed Δ is unusually positive
        p_decel = float(np.mean(valid >= obs_delta))

    return {
        "delta": obs_delta,
        "p_accelerated": p_accel,
        "p_decelerated": p_decel,
        "d_dose_to_ref": d_dose_ref,
        "d_ctrl_to_ref": d_ctrl_ref,
        "null_deltas": null_deltas,
        "distance_fn": distance_fn,
        "n_dose": n_dose,
        "n_ctrl": n_ctrl,
        "n_ref": data_late_control.shape[0],
    }


def rel_distance_comparisons(
    group_labels: list[str],
) -> list[tuple[str, str, str]]:
    """Generate relative distance triplets for developmental trajectory analysis.

    For each dose at each early PND, compare to controls at each later PND.

    Returns
    -------
    triplets : list of (dose_label, early_control_label, late_control_label)
    """
    pnds = ["p30", "p60", "p90"]
    dose_sets = [
        {"control": "C", "doses": ["L", "M", "H"]},
        {"control": "control", "doses": ["low", "medium", "high"]},
    ]

    naming = dose_sets[0]
    for candidate in dose_sets:
        if any(f"{pnds[0]}_{candidate['control']}" in group_labels
               for _ in [None]):
            naming = candidate
            break

    triplets = []
    for i, early_pnd in enumerate(pnds):
        early_ctrl = f"{early_pnd}_{naming['control']}"
        if early_ctrl not in group_labels:
            continue
        for later_pnd in pnds[i + 1:]:
            late_ctrl = f"{later_pnd}_{naming['control']}"
            if late_ctrl not in group_labels:
                continue
            for dose in naming["doses"]:
                dose_label = f"{early_pnd}_{dose}"
                if dose_label in group_labels:
                    triplets.append((dose_label, early_ctrl, late_ctrl))

    return triplets


def run_rel_distance(
    group_data: dict[str, np.ndarray],
    triplets: list[tuple[str, str, str]] | None = None,
    n_perm: int = 5000,
    seed: int = 42,
    distance_fns: list[str] | None = None,
    n_workers: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Run relative distance tests for all triplets and distance functions.

    Parameters
    ----------
    group_data : dict[str, ndarray]
        Mapping from group label to (n_subjects, n_rois) arrays.
    triplets : list of (dose_label, early_control, late_control), optional
        If None, auto-generated from group labels.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    distance_fns : list[str], optional
        Distance metrics to use. Default: ["frobenius", "spectral", "mantel"].
    n_workers : int
        Parallel workers (1 = sequential).

    Returns
    -------
    results_df : DataFrame
    null_dists : dict
        Mapping ``"{dose}_ref_{ref}__{distance_fn}"`` → null delta array.
    """
    if triplets is None:
        triplets = rel_distance_comparisons(list(group_data.keys()))

    if distance_fns is None:
        distance_fns = ["frobenius", "spectral", "mantel"]

    # Validate
    valid_triplets = []
    for dose_lbl, ctrl_lbl, ref_lbl in triplets:
        missing = [l for l in (dose_lbl, ctrl_lbl, ref_lbl) if l not in group_data]
        if missing:
            logger.warning(f"Skipping {dose_lbl}/{ctrl_lbl}/{ref_lbl}: missing {missing}")
            continue
        valid_triplets.append((dose_lbl, ctrl_lbl, ref_lbl))

    logger.info(
        f"Running relative distance: {len(valid_triplets)} triplets × "
        f"{len(distance_fns)} distance metrics = {len(valid_triplets) * len(distance_fns)} tests"
    )

    work_items = [
        (dose_lbl, ctrl_lbl, ref_lbl, dfn)
        for dose_lbl, ctrl_lbl, ref_lbl in valid_triplets
        for dfn in distance_fns
    ]

    rows = []
    null_dists: dict[str, np.ndarray] = {}

    def _do_one(dose_lbl, ctrl_lbl, ref_lbl, dfn):
        return rel_distance_test(
            data_dose=group_data[dose_lbl],
            data_early_control=group_data[ctrl_lbl],
            data_late_control=group_data[ref_lbl],
            n_perm=n_perm,
            seed=seed,
            distance_fn=dfn,
        )

    if n_workers > 1 and len(work_items) > 1:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for dose_lbl, ctrl_lbl, ref_lbl, dfn in work_items:
                future = executor.submit(_do_one, dose_lbl, ctrl_lbl, ref_lbl, dfn)
                futures[future] = (dose_lbl, ctrl_lbl, ref_lbl, dfn)
            for future in as_completed(futures):
                dose_lbl, ctrl_lbl, ref_lbl, dfn = futures[future]
                result = future.result()
                rows.append(_rel_dist_row(dose_lbl, ctrl_lbl, ref_lbl, result))
                key = f"{dose_lbl}_ref_{ref_lbl}__{dfn}"
                null_dists[key] = result["null_deltas"]
    else:
        for dose_lbl, ctrl_lbl, ref_lbl, dfn in work_items:
            comp_label = f"{dose_lbl}_ref_{ref_lbl}"
            logger.info(f"\n--- Relative distance: {comp_label} ({dfn}) ---")
            result = _do_one(dose_lbl, ctrl_lbl, ref_lbl, dfn)
            rows.append(_rel_dist_row(dose_lbl, ctrl_lbl, ref_lbl, result))
            null_dists[f"{comp_label}__{dfn}"] = result["null_deltas"]

    return pd.DataFrame(rows), null_dists


def _rel_dist_row(dose_lbl, ctrl_lbl, ref_lbl, result):
    """Build a row dict from relative distance test result."""
    return {
        "dose_group": dose_lbl,
        "early_control": ctrl_lbl,
        "reference": ref_lbl,
        "comparison": f"{dose_lbl}_ref_{ref_lbl}",
        "distance_fn": result["distance_fn"],
        "d_dose_to_ref": result["d_dose_to_ref"],
        "d_ctrl_to_ref": result["d_ctrl_to_ref"],
        "delta": result["delta"],
        "p_accelerated": result["p_accelerated"],
        "p_decelerated": result["p_decelerated"],
        "n_dose": result["n_dose"],
        "n_ctrl": result["n_ctrl"],
        "n_ref": result["n_ref"],
        "interpretation": (
            "accelerated" if result["delta"] < 0 else "decelerated"
        ),
    }


def run_all_comparisons(
    group_data: dict[str, np.ndarray],
    comparisons: Optional[list[tuple[str, str]]] = None,
    n_perm: int = 5000,
    seed: int = 42,
    n_workers: int = 1,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run absolute distance test for each pairwise comparison.

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
                    abs_distance_test,
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

            result = abs_distance_test(
                data_a=group_data[label_a],
                data_b=group_data[label_b],
                n_perm=n_perm,
                seed=seed,
            )
            _collect_result(comparison_label, label_a, label_b, result)

    results_df = pd.DataFrame(rows)
    return results_df, null_dists


# ---------------------------------------------------------------------------
#  Subject-level relative distance
# ---------------------------------------------------------------------------


def _subject_similarity(data: np.ndarray, ref_mean: np.ndarray,
                        similarity_fn: str) -> np.ndarray:
    """Compute per-subject similarity to a reference mean ROI profile.

    Parameters
    ----------
    data : ndarray (n_subjects, n_rois)
    ref_mean : ndarray (n_rois,)
    similarity_fn : {"spearman", "euclidean", "cosine"}

    Returns
    -------
    similarities : ndarray (n_subjects,)
    """
    if similarity_fn == "spearman":
        return np.array([
            stats.spearmanr(data[i], ref_mean).statistic
            for i in range(data.shape[0])
        ])
    elif similarity_fn == "euclidean":
        # Negate so higher = more similar (consistent with spearman)
        return -np.linalg.norm(data - ref_mean[np.newaxis, :], axis=1)
    elif similarity_fn == "cosine":
        norms = np.linalg.norm(data, axis=1) * np.linalg.norm(ref_mean)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        return (data @ ref_mean) / norms
    else:
        raise ValueError(f"Unknown similarity_fn: {similarity_fn!r}")


def subject_rel_distance_test(
    data_dose: np.ndarray,
    data_early_control: np.ndarray,
    data_late_control: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
    similarity_fn: str = "spearman",
) -> dict:
    """Subject-level relative distance: compare individual similarity to a
    reference group.

    For each subject in dose and early-control groups, computes the similarity
    of that subject's ROI profile to the reference group mean profile.  Then
    tests whether the distribution of similarities differs between groups.

    Positive delta → dose subjects are more similar to the reference
    (accelerated if reference = older controls).

    Parameters
    ----------
    data_dose : ndarray (n_dose, n_rois)
    data_early_control : ndarray (n_ctrl, n_rois)
    data_late_control : ndarray (n_ref, n_rois)
    n_perm : int
    seed : int
    similarity_fn : {"spearman", "euclidean", "cosine"}

    Returns
    -------
    dict with keys: delta, p_accelerated, p_decelerated, p_mann_whitney,
    mean_dose_sim, mean_ctrl_sim, dose_similarities, ctrl_similarities,
    similarity_fn, n_dose, n_ctrl, n_ref.
    """
    rng = np.random.default_rng(seed)

    # Reference mean profile
    ref_mean = np.nanmean(data_late_control, axis=0)

    # Per-subject similarities
    dose_sims = _subject_similarity(data_dose, ref_mean, similarity_fn)
    ctrl_sims = _subject_similarity(data_early_control, ref_mean, similarity_fn)

    mean_dose = float(np.nanmean(dose_sims))
    mean_ctrl = float(np.nanmean(ctrl_sims))
    obs_delta = mean_dose - mean_ctrl

    logger.info(
        f"  Subject-level ({similarity_fn}): "
        f"mean_dose={mean_dose:.4f}, mean_ctrl={mean_ctrl:.4f}, "
        f"Δ={obs_delta:.4f} ({'accelerated' if obs_delta > 0 else 'decelerated'})"
    )

    # Mann-Whitney U as a secondary non-parametric test
    valid_dose = dose_sims[~np.isnan(dose_sims)]
    valid_ctrl = ctrl_sims[~np.isnan(ctrl_sims)]
    if len(valid_dose) >= 2 and len(valid_ctrl) >= 2:
        mw_stat, mw_p = stats.mannwhitneyu(
            valid_dose, valid_ctrl, alternative="two-sided"
        )
        p_mw = float(mw_p)
    else:
        p_mw = float("nan")

    # Permutation test: shuffle dose/control labels among pre-computed scalars
    pooled = np.concatenate([dose_sims, ctrl_sims])
    n_dose = len(dose_sims)
    null_deltas = np.zeros(n_perm)

    for p in range(n_perm):
        perm_idx = rng.permutation(len(pooled))
        null_deltas[p] = (
            np.nanmean(pooled[perm_idx[:n_dose]])
            - np.nanmean(pooled[perm_idx[n_dose:]])
        )

    valid_null = null_deltas[~np.isnan(null_deltas)]
    if len(valid_null) == 0:
        p_accel = float("nan")
        p_decel = float("nan")
    else:
        # Accelerated: dose is MORE similar to reference (delta > 0)
        p_accel = float(np.mean(valid_null >= obs_delta))
        # Decelerated: dose is LESS similar to reference (delta < 0)
        p_decel = float(np.mean(valid_null <= obs_delta))

    return {
        "delta": obs_delta,
        "p_accelerated": p_accel,
        "p_decelerated": p_decel,
        "p_mann_whitney": p_mw,
        "mean_dose_sim": mean_dose,
        "mean_ctrl_sim": mean_ctrl,
        "dose_similarities": dose_sims,
        "ctrl_similarities": ctrl_sims,
        "similarity_fn": similarity_fn,
        "n_dose": data_dose.shape[0],
        "n_ctrl": data_early_control.shape[0],
        "n_ref": data_late_control.shape[0],
    }


def _subject_rel_dist_row(dose_lbl, ctrl_lbl, ref_lbl, result):
    """Build a summary row dict from subject-level rel distance result."""
    return {
        "dose_group": dose_lbl,
        "early_control": ctrl_lbl,
        "reference": ref_lbl,
        "comparison": f"{dose_lbl}_ref_{ref_lbl}",
        "similarity_fn": result["similarity_fn"],
        "mean_dose_sim": result["mean_dose_sim"],
        "mean_ctrl_sim": result["mean_ctrl_sim"],
        "delta": result["delta"],
        "p_accelerated": result["p_accelerated"],
        "p_decelerated": result["p_decelerated"],
        "p_mann_whitney": result["p_mann_whitney"],
        "n_dose": result["n_dose"],
        "n_ctrl": result["n_ctrl"],
        "n_ref": result["n_ref"],
        "interpretation": (
            "accelerated" if result["delta"] > 0 else "decelerated"
        ),
    }


def _subject_rel_dist_per_subject_rows(dose_lbl, ctrl_lbl, ref_lbl, result):
    """Build per-subject detail rows from subject-level rel distance result."""
    comp = f"{dose_lbl}_ref_{ref_lbl}"
    sim_fn = result["similarity_fn"]
    rows = []
    for i, s in enumerate(result["dose_similarities"]):
        rows.append({
            "comparison": comp,
            "similarity_fn": sim_fn,
            "group": dose_lbl,
            "subject_index": i,
            "similarity_to_ref": s,
        })
    for i, s in enumerate(result["ctrl_similarities"]):
        rows.append({
            "comparison": comp,
            "similarity_fn": sim_fn,
            "group": ctrl_lbl,
            "subject_index": i,
            "similarity_to_ref": s,
        })
    return rows


def run_subject_rel_distance(
    group_data: dict[str, np.ndarray],
    triplets: list[tuple[str, str, str]] | None = None,
    n_perm: int = 5000,
    seed: int = 42,
    similarity_fns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run subject-level relative distance tests for all triplets.

    Parameters
    ----------
    group_data : dict[str, ndarray]
        Mapping from group label to (n_subjects, n_rois) arrays.
    triplets : list of (dose_label, early_control, late_control), optional
    n_perm : int
    seed : int
    similarity_fns : list[str], optional
        Default: ["spearman", "euclidean", "cosine"].

    Returns
    -------
    summary_df : DataFrame
        One row per triplet × similarity_fn.
    per_subject_df : DataFrame
        One row per subject per triplet × similarity_fn.
    """
    if triplets is None:
        triplets = rel_distance_comparisons(list(group_data.keys()))

    if similarity_fns is None:
        similarity_fns = ["spearman", "euclidean", "cosine"]

    # Validate triplets
    valid_triplets = []
    for dose_lbl, ctrl_lbl, ref_lbl in triplets:
        missing = [l for l in (dose_lbl, ctrl_lbl, ref_lbl)
                   if l not in group_data]
        if missing:
            logger.warning(
                f"Skipping triplet ({dose_lbl}, {ctrl_lbl}, {ref_lbl}): "
                f"missing groups {missing}"
            )
            continue
        valid_triplets.append((dose_lbl, ctrl_lbl, ref_lbl))

    summary_rows = []
    per_subject_rows = []

    for dose_lbl, ctrl_lbl, ref_lbl in valid_triplets:
        comp_label = f"{dose_lbl}_ref_{ref_lbl}"
        for sim_fn in similarity_fns:
            logger.info(
                f"\n--- Subject rel-distance: {comp_label} ({sim_fn}) ---"
            )
            result = subject_rel_distance_test(
                data_dose=group_data[dose_lbl],
                data_early_control=group_data[ctrl_lbl],
                data_late_control=group_data[ref_lbl],
                n_perm=n_perm,
                seed=seed,
                similarity_fn=sim_fn,
            )
            summary_rows.append(
                _subject_rel_dist_row(dose_lbl, ctrl_lbl, ref_lbl, result)
            )
            per_subject_rows.extend(
                _subject_rel_dist_per_subject_rows(
                    dose_lbl, ctrl_lbl, ref_lbl, result
                )
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_subject_rows)


# Backward-compatible aliases
whole_network_test = abs_distance_test
maturation_distance_test = rel_distance_test
maturation_distance_comparisons = rel_distance_comparisons
run_maturation_distance = run_rel_distance
_mat_dist_row = _rel_dist_row
