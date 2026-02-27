"""
ME-ICA component classification for multi-echo fMRI.

Implements the core meica.py v2.5 approach (Kundu et al. 2014) using
MELODIC ICA components projected onto multi-echo data for kappa/rho
F-statistic computation. Classification follows the tedana 'minimal'
decision tree logic.

References
----------
- Kundu et al. 2014, "Differentiating BOLD and non-BOLD signals in fMRI
  time series from anesthetized rats using multi-echo EPI at 11.7 T"
- tedana minimal decision tree (tedana/resources/decision_trees/minimal.json)
- tedana/metrics/dependence.py for F-statistic models
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_meica_kappa_rho(
    echo_files: List[Path],
    echo_times_ms: List[float],
    mixing_matrix: np.ndarray,
    brain_mask_file: Path,
    optcom_file: Optional[Path] = None,
    f_max: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute kappa (TE-dependence) and rho (TE-independence) for MELODIC components.

    Uses multivariate OLS with intercept (matching tedana's approach) to compute
    per-echo partial regression betas, then fits S0 and T2* models across echoes.
    Weights are Fisher-z-transformed correlations from the optimally combined data,
    matching tedana's calculate_dependence_metrics().

    Parameters
    ----------
    echo_files : list of Path
        Motion-corrected, masked echo NIfTI files (one per echo).
    echo_times_ms : list of float
        Echo times in milliseconds.
    mixing_matrix : np.ndarray
        MELODIC mixing matrix, shape (T, n_components).
    brain_mask_file : Path
        Brain mask NIfTI file.
    optcom_file : Path, optional
        Optimally combined NIfTI file (used for weight computation).
        If None, a simple T2*-weighted combination is computed from echo data.
    f_max : float
        Cap for pseudo-F statistics (default 500).

    Returns
    -------
    kappas : np.ndarray
        TE-dependence metric per component, shape (n_components,).
    rhos : np.ndarray
        TE-independence metric per component, shape (n_components,).
    varex : np.ndarray
        Variance explained per component (%), shape (n_components,).
    """
    from scipy.stats import zscore as scipy_zscore

    mask_img = nib.load(brain_mask_file)
    mask = mask_img.get_fdata().astype(bool)

    n_echoes = len(echo_files)
    n_components = mixing_matrix.shape[1]
    tes = np.array(echo_times_ms, dtype=np.float64)

    # Load echo data into (E, V, T)
    echo_data = []
    for ef in echo_files:
        img = nib.load(ef)
        d = img.get_fdata()  # (x, y, z, T)
        echo_data.append(d[mask])  # (V, T)
    # Stack: (E, V, T)
    data_cat = np.array(echo_data, dtype=np.float64)
    n_voxels = data_cat.shape[1]
    n_timepoints = data_cat.shape[2]

    # Trim mixing matrix to match timepoints
    mix = mixing_matrix[:n_timepoints].astype(np.float64)

    # Mean signal per echo per voxel: (E, V)
    mu = data_cat.mean(axis=2)

    # ── Voxel selection: Otsu + echo decay filter ──
    mu_e1 = mu[0]
    e1_nonzero = mu_e1[mu_e1 > 0]
    if len(e1_nonzero) > 0:
        hist_vals, bin_edges = np.histogram(np.log1p(e1_nonzero), bins=100)
        total = hist_vals.sum()
        cum_sum = np.cumsum(hist_vals)
        cum_mean = np.cumsum(hist_vals * (bin_edges[:-1] + bin_edges[1:]) / 2)
        best_thresh_idx = 0
        best_var = 0
        for t in range(1, len(hist_vals)):
            w0 = cum_sum[t]
            w1 = total - w0
            if w0 == 0 or w1 == 0:
                continue
            m0 = cum_mean[t] / w0
            m1 = (cum_mean[-1] - cum_mean[t]) / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var = var
                best_thresh_idx = t
        otsu_log = (bin_edges[best_thresh_idx] + bin_edges[best_thresh_idx + 1]) / 2
        signal_thresh = np.expm1(otsu_log)
    else:
        signal_thresh = 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        decay_ratio = np.where(mu_e1 > 0, mu[-1] / mu_e1, 1.0)

    valid_voxels = (mu_e1 > signal_thresh) & (decay_ratio < 0.9)
    n_valid = valid_voxels.sum()
    print(f"  ME-ICA: {n_valid}/{n_voxels} voxels pass signal+decay filter "
          f"(signal>{signal_thresh:.0f}, decay<0.9)")

    # ── Multivariate OLS with intercept (matching tedana) ──
    # Design matrix: [mix | ones] → partial regression betas
    X = np.column_stack([mix, np.ones(n_timepoints)])  # (T, C+1)
    betas = np.zeros((n_echoes, n_voxels, n_components), dtype=np.float64)
    for e in range(n_echoes):
        # Solve: Y_e = X @ B_e where Y_e is (T, V) and B_e is (C+1, V)
        B, _, _, _ = np.linalg.lstsq(X, data_cat[e].T, rcond=None)
        betas[e] = B[:n_components].T  # (V, C), drop intercept row

    # ── Compute spatial weights from optcom (matching tedana) ──
    # tedana: z-score optcom + z-score mixing → OLS → correlations → Fisher z → z-score → clip → square
    if optcom_file is not None:
        optcom_data = nib.load(optcom_file).get_fdata()[mask]  # (V, T)
    else:
        # Simple T2*-weighted combination: weight_e = TE_e * exp(-TE_e / T2*)
        # Estimate T2* from log-linear fit: log(S) = log(S0) - TE/T2*
        with np.errstate(divide='ignore', invalid='ignore'):
            log_mu = np.log(np.maximum(mu, 1e-10))  # (E, V)
        # Linear fit: log(S) vs TE → slope = -1/T2*
        te_col = tes.reshape(-1, 1)
        te_design = np.column_stack([te_col, np.ones(n_echoes)])
        fit_coeffs = np.linalg.lstsq(te_design, log_mu, rcond=None)[0]  # (2, V)
        t2star = np.clip(-1.0 / np.where(fit_coeffs[0] != 0, fit_coeffs[0], -1e-10),
                         1.0, 200.0)  # (V,) in ms
        # Weights: TE * exp(-TE/T2*)
        oc_weights = tes[:, None] * np.exp(-tes[:, None] / t2star[None, :])  # (E, V)
        oc_weights /= oc_weights.sum(axis=0, keepdims=True)  # normalize
        optcom_data = (data_cat * oc_weights[:, :, None]).sum(axis=0)  # (V, T)

    optcom_valid = optcom_data[valid_voxels]  # (V_valid, T)
    optcom_valid = optcom_valid[:, :n_timepoints]

    # Z-score the optcom data across time (per voxel)
    optcom_std = optcom_valid.std(axis=1, keepdims=True)
    optcom_std[optcom_std == 0] = 1.0
    optcom_z = (optcom_valid - optcom_valid.mean(axis=1, keepdims=True)) / optcom_std

    # Z-score the mixing matrix across time (per component)
    mix_std = mix.std(axis=0, keepdims=True)
    mix_std[mix_std == 0] = 1.0
    mix_z = (mix - mix.mean(axis=0, keepdims=True)) / mix_std

    # OLS of z-scored mixing against z-scored optcom → approx correlations
    # optcom_z: (V_valid, T), mix_z: (T, C)
    corr_betas = np.linalg.lstsq(mix_z, optcom_z.T, rcond=None)[0].T  # (V_valid, C)

    # Clip and Fisher z-transform
    corr_betas = np.clip(corr_betas, -0.999, 0.999)
    fisher_z = np.arctanh(corr_betas)  # (V_valid, C)

    # Z-score across voxels per component, clip at ±8
    fz_std = fisher_z.std(axis=0, keepdims=True)
    fz_std[fz_std == 0] = 1.0
    z_maps = (fisher_z - fisher_z.mean(axis=0, keepdims=True)) / fz_std
    z_maps = np.clip(z_maps, -8, 8)

    # Square for final weights: (V_valid, C)
    weight_maps = z_maps ** 2

    # ── Compute total variance for varex (using optcom betas) ──
    # Match tedana: varex relative to sum of all component variances, not total data variance
    optcom_betas_all = np.linalg.lstsq(
        X, optcom_data[:, :n_timepoints].T, rcond=None
    )[0][:n_components].T  # (V, C)
    comp_variances = (optcom_betas_all ** 2).sum(axis=0)  # (C,)
    total_comp_var = comp_variances.sum()

    kappas = np.zeros(n_components)
    rhos = np.zeros(n_components)
    varex = np.zeros(n_components)

    for j in range(n_components):
        # Beta for this component across echoes: (E, V_valid)
        beta_v = betas[:, valid_voxels, j]  # (E, V_valid)
        mu_v = mu[:, valid_voxels]  # (E, V_valid)

        # T2* model (TE-dependent): beta_e(v) ~ c(v) * TE_e * mu_e(v)
        x2 = tes[:, None] * mu_v  # (E, V_valid)

        # S0 model (TE-independent): beta_e(v) ~ c(v) * mu_e(v)
        x1 = mu_v  # (E, V_valid)

        # Fit single coefficient per voxel for each model
        x2_sq = np.sum(x2 ** 2, axis=0)
        x2_sq[x2_sq == 0] = 1.0
        c_t2 = np.sum(beta_v * x2, axis=0) / x2_sq

        x1_sq = np.sum(x1 ** 2, axis=0)
        x1_sq[x1_sq == 0] = 1.0
        c_s0 = np.sum(beta_v * x1, axis=0) / x1_sq

        # Predicted betas
        pred_t2 = x2 * c_t2
        pred_s0 = x1 * c_s0

        # Sum of squares
        alpha = np.sum(beta_v ** 2, axis=0)
        sse_t2 = np.sum((beta_v - pred_t2) ** 2, axis=0)
        sse_s0 = np.sum((beta_v - pred_s0) ** 2, axis=0)

        # Pseudo-F: (alpha - sse) * (n_echoes - 1) / sse
        sse_t2[sse_t2 == 0] = 1e-10
        sse_s0[sse_s0 == 0] = 1e-10

        f_t2 = np.clip((alpha - sse_t2) * (n_echoes - 1) / sse_t2, 0, f_max)
        f_s0 = np.clip((alpha - sse_s0) * (n_echoes - 1) / sse_s0, 0, f_max)

        # Weighted average using Fisher-z weights for this component
        weights = weight_maps[:, j]
        w_sum = weights.sum()

        if w_sum > 0:
            kappas[j] = np.sum(f_t2 * weights) / w_sum
            rhos[j] = np.sum(f_s0 * weights) / w_sum
        else:
            kappas[j] = 0.0
            rhos[j] = 0.0

        # Variance explained (matching tedana: relative to sum of component variances)
        varex[j] = 100.0 * comp_variances[j] / total_comp_var if total_comp_var > 0 else 0.0

    return kappas, rhos, varex


def getelbow(arr: np.ndarray, return_val: bool = False):
    """Elbow detection using geometric perpendicular distance method.

    Sorts values in descending order and finds the point with maximum
    perpendicular distance from the line connecting the first (highest)
    and last (lowest) points.

    Matches tedana's selection_utils.getelbow().

    Parameters
    ----------
    arr : np.ndarray
        1D array of values.
    return_val : bool
        If True, return the value at the elbow; otherwise return the index.

    Returns
    -------
    int or float
        Elbow index (in the original sorted-descending array) or value.
    """
    if arr.ndim != 1:
        raise ValueError(f"Parameter arr should be 1d, not {arr.ndim}d")
    if arr.size == 0:
        raise ValueError("Empty array for elbow calculation")

    arr_sorted = np.sort(arr)[::-1]
    n = arr_sorted.shape[0]

    if n <= 2:
        return arr_sorted[0] if return_val else 0

    # 2D coordinates: (index, value)
    coords = np.array([np.arange(n, dtype=float), arr_sorted])

    # Translate so first point is at origin
    p = coords - coords[:, 0].reshape(2, 1)

    # Direction vector to last point
    b = p[:, -1]
    b_norm = np.sqrt((b ** 2).sum())
    if b_norm == 0:
        return arr_sorted[0] if return_val else 0
    b_hat = (b / b_norm).reshape(2, 1)

    # Perpendicular distance from each point to the line
    proj = p - np.dot(b_hat.T, p) * np.tile(b_hat, (1, n))
    d = np.sqrt((proj ** 2).sum(axis=0))

    k_min_ind = d.argmax()

    if return_val:
        return arr_sorted[k_min_ind]
    else:
        return k_min_ind


def classify_meica_components(
    echo_files: List[Path],
    echo_times_ms: List[float],
    melodic_dir: Path,
    brain_mask_file: Path,
    optcom_file: Optional[Path] = None,
    rho_kappa_scale: float = 1.0,
    kappa_elbow_scale: float = 1.0,
    rho_elbow_scale: float = 1.0,
) -> Dict[str, Any]:
    """Classify MELODIC components using ME-ICA kappa/rho approach.

    Follows the tedana 'minimal' decision tree logic:

    1. Reject if rho > rho_kappa_scale * kappa
    2. Compute kappa_elbow and rho_elbow via getelbow()
    3. Provisionally accept if kappa >= kappa_elbow * kappa_elbow_scale
    4. Accept provisionally accepted if kappa > 2*rho
    5. Provisionally reject if rho > rho_elbow * rho_elbow_scale
    6. Accept low-variance provisionally rejected (<0.1% individual, <1% cumulative)
    7. Finalize: remaining provisional accepts -> accepted, rest -> rejected
    8. Safety floor: if ALL rejected, retain highest-kappa component

    Parameters
    ----------
    echo_files : list of Path
        Motion-corrected, masked per-echo NIfTI files.
    echo_times_ms : list of float
        Echo times in milliseconds.
    melodic_dir : Path
        MELODIC output directory (contains melodic_mix, melodic_IC.nii.gz).
    brain_mask_file : Path
        Brain mask NIfTI.
    optcom_file : Path, optional
        Optimally combined NIfTI file (used for weight computation).
    rho_kappa_scale : float
        Scale factor for rho > scale*kappa rejection (default 1.0).
    kappa_elbow_scale : float
        Multiplier for kappa elbow threshold (default 1.0).
        Values < 1.0 lower the threshold, accepting more components.
    rho_elbow_scale : float
        Multiplier for rho elbow threshold (default 1.0).
        Values > 1.0 raise the threshold, making rho rejection less strict.

    Returns
    -------
    dict
        Classification results with per-component info, summary, and thresholds.
    """
    print("ME-ICA component classification...")

    # Load MELODIC mixing matrix
    mix_file = melodic_dir / 'melodic_mix'
    mixing_matrix = np.loadtxt(mix_file)
    if mixing_matrix.ndim == 1:
        mixing_matrix = mixing_matrix.reshape(-1, 1)
    n_components = mixing_matrix.shape[1]
    print(f"  {n_components} MELODIC components")

    # Compute kappa and rho
    print("  Computing kappa/rho F-statistics...")
    kappas, rhos, varex = compute_meica_kappa_rho(
        echo_files=echo_files,
        echo_times_ms=echo_times_ms,
        mixing_matrix=mixing_matrix,
        brain_mask_file=brain_mask_file,
        optcom_file=optcom_file,
    )

    print(f"  Kappa range: [{kappas.min():.1f}, {kappas.max():.1f}], "
          f"median={np.median(kappas):.1f}")
    print(f"  Rho   range: [{rhos.min():.1f}, {rhos.max():.1f}], "
          f"median={np.median(rhos):.1f}")
    print(f"  VarEx range: [{varex.min():.2f}%, {varex.max():.2f}%], "
          f"total={varex.sum():.1f}%")

    # Classification state: 'unclassified', 'rejected', 'provisionalaccept',
    # 'provisionalreject', 'accepted'
    # Use a Python list (not numpy array) to avoid string truncation
    status = ['unclassified'] * n_components
    reasons = [[] for _ in range(n_components)]

    # --- Node 1: Reject if rho > scale * kappa ---
    rho_gt_kappa = rhos > rho_kappa_scale * kappas
    for i in np.where(rho_gt_kappa)[0]:
        status[i] = 'rejected'
        reasons[i].append(f'rho ({rhos[i]:.1f}) > {rho_kappa_scale}*kappa ({kappas[i]:.1f})')

    n_rej_step1 = rho_gt_kappa.sum()
    print(f"  Step 1 (rho > {rho_kappa_scale}*kappa): {n_rej_step1} rejected")

    # --- Nodes 2-5: Skip cluster-based metrics (countsigFS0, dice, signal-noise_t) ---
    # These require cluster thresholding infrastructure we don't have.
    # The minimal tree already works without them being the primary drivers.

    # --- Node 6: Compute kappa_elbow ---
    kappa_elbow_raw = getelbow(kappas, return_val=True)
    kappa_elbow_val = kappa_elbow_raw * kappa_elbow_scale
    if kappa_elbow_scale != 1.0:
        print(f"  Kappa elbow: {kappa_elbow_raw:.1f} × {kappa_elbow_scale} = {kappa_elbow_val:.1f}")
    else:
        print(f"  Kappa elbow: {kappa_elbow_val:.1f}")

    # --- Node 7: Compute rho_elbow (liberal, on unclassified components) ---
    unclassified_idx = [i for i in range(n_components) if status[i] == 'unclassified']
    if len(unclassified_idx) > 2:
        rho_elbow_raw = getelbow(rhos[unclassified_idx], return_val=True)
    else:
        rho_elbow_raw = getelbow(rhos, return_val=True)
    rho_elbow_val = rho_elbow_raw * rho_elbow_scale
    if rho_elbow_scale != 1.0:
        print(f"  Rho elbow: {rho_elbow_raw:.1f} × {rho_elbow_scale} = {rho_elbow_val:.1f}")
    else:
        print(f"  Rho elbow (liberal): {rho_elbow_val:.1f}")

    # --- Node 8: Provisional accept/reject based on kappa_elbow ---
    for i in range(n_components):
        if status[i] != 'unclassified':
            continue
        if kappas[i] >= kappa_elbow_val:
            status[i] = 'provisionalaccept'
        else:
            status[i] = 'provisionalreject'
            reasons[i].append(f'kappa ({kappas[i]:.1f}) < kappa_elbow ({kappa_elbow_val:.1f})')

    n_prov_acc = sum(1 for s in status if s == 'provisionalaccept')
    n_prov_rej = sum(1 for s in status if s == 'provisionalreject')
    print(f"  Step 3 (kappa elbow): {n_prov_acc} provisional accept, "
          f"{n_prov_rej} provisional reject")

    # --- Node 9: Accept provisionally accepted if kappa > 2*rho ---
    for i in range(n_components):
        if status[i] != 'provisionalaccept':
            continue
        if kappas[i] > 2 * rhos[i]:
            status[i] = 'accepted'
            reasons[i].append(f'kappa ({kappas[i]:.1f}) > 2*rho ({2*rhos[i]:.1f})')

    n_accepted_rescue = sum(1 for s in status if s == 'accepted')
    print(f"  Step 4 (kappa > 2*rho rescue): {n_accepted_rescue} accepted")

    # --- Node 10: Provisionally reject if rho > rho_elbow ---
    for i in range(n_components):
        if status[i] not in ('provisionalaccept', 'provisionalreject'):
            continue
        if rhos[i] > rho_elbow_val:
            if status[i] != 'provisionalreject':
                reasons[i].append(
                    f'rho ({rhos[i]:.1f}) > rho_elbow ({rho_elbow_val:.1f})')
            status[i] = 'provisionalreject'

    # --- Node 11: Accept low-variance provisionally rejected ---
    cum_varex = 0.0
    for i in range(n_components):
        if status[i] != 'provisionalreject':
            continue
        if varex[i] < 0.1 and cum_varex + varex[i] < 1.0:
            status[i] = 'accepted'
            reasons[i].append(f'low variance ({varex[i]:.3f}%)')
            cum_varex += varex[i]

    # --- Nodes 12-13: Finalize ---
    for i in range(n_components):
        if status[i] == 'provisionalaccept':
            status[i] = 'accepted'
        elif status[i] in ('provisionalreject', 'unclassified'):
            if not reasons[i]:
                reasons[i].append('remaining unclassified/provisionalreject')
            status[i] = 'rejected'

    n_accepted = sum(1 for s in status if s == 'accepted')
    n_rejected = sum(1 for s in status if s == 'rejected')
    print(f"  Final: {n_accepted} accepted, {n_rejected} rejected")

    # --- Safety floor: if ALL rejected, retain highest-kappa component ---
    if n_accepted == 0 and n_components > 0:
        best = int(np.argmax(kappas))
        status[best] = 'accepted'
        reasons[best].append('safety floor: highest kappa retained')
        n_accepted = 1
        n_rejected -= 1
        print(f"  Safety floor: retained component {best} "
              f"(kappa={kappas[best]:.1f})")

    # Build results dict matching classify_ica_components format
    components = []
    for i in range(n_components):
        components.append({
            'index': i,
            'label': 'signal' if status[i] == 'accepted' else 'noise',
            'kappa': float(kappas[i]),
            'rho': float(rhos[i]),
            'kappa_rho_ratio': float(kappas[i] / rhos[i]) if rhos[i] > 0 else float('inf'),
            'variance_explained': float(varex[i]),
            'classification_reason': reasons[i],
        })

    signal_comps = [c['index'] for c in components if c['label'] == 'signal']
    noise_comps = [c['index'] for c in components if c['label'] == 'noise']

    results = {
        'n_components': n_components,
        'components': components,
        'summary': {
            'n_signal': len(signal_comps),
            'n_noise': len(noise_comps),
            'signal_components': signal_comps,
            'noise_components': noise_comps,
        },
        'thresholds': {
            'kappa_elbow': float(kappa_elbow_val),
            'kappa_elbow_raw': float(kappa_elbow_raw),
            'kappa_elbow_scale': float(kappa_elbow_scale),
            'rho_elbow': float(rho_elbow_val),
            'rho_elbow_raw': float(rho_elbow_raw),
            'rho_elbow_scale': float(rho_elbow_scale),
            'rho_kappa_scale': float(rho_kappa_scale),
        },
        'metrics': {
            'kappa_median': float(np.median(kappas)),
            'kappa_mean': float(np.mean(kappas)),
            'rho_median': float(np.median(rhos)),
            'rho_mean': float(np.mean(rhos)),
            'varex_total': float(varex.sum()),
        },
    }

    print(f"\n  ME-ICA classification complete:")
    print(f"    Signal: {len(signal_comps)} components "
          f"({100*len(signal_comps)/n_components:.0f}%)")
    print(f"    Noise:  {len(noise_comps)} components "
          f"({100*len(noise_comps)/n_components:.0f}%)")
    print(f"    Kappa elbow: {kappa_elbow_val:.1f}, Rho elbow: {rho_elbow_val:.1f}")

    return results


def generate_meica_qc(
    subject: str,
    session: str,
    classification_results: Dict[str, Any],
    melodic_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate QC report for ME-ICA classification.

    Produces:
    - Kappa vs Rho scatter plot
    - Kappa rank plot colored by classification
    - Classification summary bar chart
    - Component table with kappa, rho, ratio, varex, classification, reason

    Parameters
    ----------
    subject : str
        Subject identifier.
    session : str
        Session identifier.
    classification_results : dict
        Output from classify_meica_components().
    melodic_dir : Path
        MELODIC output directory.
    output_dir : Path
        QC output directory.

    Returns
    -------
    Path
        Path to HTML QC report.
    """
    print("Generating ME-ICA QC report...")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    components = classification_results['components']
    summary = classification_results['summary']
    thresholds = classification_results['thresholds']

    kappas = np.array([c['kappa'] for c in components])
    rhos_arr = np.array([c['rho'] for c in components])
    varex_arr = np.array([c['variance_explained'] for c in components])
    labels = [c['label'] for c in components]

    signal_mask = np.array([l == 'signal' for l in labels])
    noise_mask = ~signal_mask

    kappa_elbow = thresholds['kappa_elbow']
    rho_elbow = thresholds['rho_elbow']

    # =========================================================================
    # Figure 1: Kappa vs Rho scatter
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    if signal_mask.any():
        ax.scatter(kappas[signal_mask], rhos_arr[signal_mask],
                   c='green', s=60, alpha=0.7, label='Signal', edgecolors='k', linewidths=0.5)
    if noise_mask.any():
        ax.scatter(kappas[noise_mask], rhos_arr[noise_mask],
                   c='red', s=60, alpha=0.7, label='Noise', edgecolors='k', linewidths=0.5)

    # Diagonal: kappa = rho
    lim_max = max(kappas.max(), rhos_arr.max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.3, label='kappa = rho')

    # Elbow thresholds
    ax.axvline(x=kappa_elbow, color='blue', linestyle=':', alpha=0.5,
               label=f'kappa elbow ({kappa_elbow:.1f})')
    ax.axhline(y=rho_elbow, color='orange', linestyle=':', alpha=0.5,
               label=f'rho elbow ({rho_elbow:.1f})')

    # kappa = 2*rho line
    ax.plot([0, lim_max], [0, lim_max / 2], 'g--', alpha=0.3, label='kappa = 2*rho')

    ax.set_xlabel('Kappa (TE-dependence)', fontsize=12)
    ax.set_ylabel('Rho (TE-independence)', fontsize=12)
    ax.set_title(f'ME-ICA: Kappa vs Rho — {subject} {session}', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    plt.tight_layout()
    scatter_fig = figures_dir / f'{subject}_{session}_meica_kappa_rho.png'
    plt.savefig(scatter_fig, dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: Kappa rank plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 5))

    sorted_idx = np.argsort(kappas)[::-1]
    colors = ['green' if labels[i] == 'signal' else 'red' for i in sorted_idx]
    ax.bar(range(len(sorted_idx)), kappas[sorted_idx], color=colors, alpha=0.7,
           edgecolor='k', linewidth=0.3)
    ax.axhline(y=kappa_elbow, color='blue', linestyle=':', alpha=0.6,
               label=f'kappa elbow ({kappa_elbow:.1f})')
    ax.set_xlabel('Component (sorted by kappa)', fontsize=12)
    ax.set_ylabel('Kappa', fontsize=12)
    ax.set_title(f'ME-ICA: Kappa Rank Plot — {subject} {session}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    rank_fig = figures_dir / f'{subject}_{session}_meica_kappa_rank.png'
    plt.savefig(rank_fig, dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Classification summary
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(['Signal', 'Noise'],
                  [summary['n_signal'], summary['n_noise']],
                  color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax.set_ylabel('Number of Components')
    ax.set_title(f'ME-ICA Classification (Total: {len(components)})')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, [summary['n_signal'], summary['n_noise']]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    summary_fig = figures_dir / f'{subject}_{session}_meica_summary.png'
    plt.savefig(summary_fig, dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # HTML Report
    # =========================================================================
    component_rows = ""
    for comp in components:
        row_class = 'noise-row' if comp['label'] == 'noise' else 'signal-row'
        reason_str = '<br>'.join(comp['classification_reason']) if comp['classification_reason'] else '-'
        kr_ratio = f"{comp['kappa_rho_ratio']:.2f}" if comp['kappa_rho_ratio'] != float('inf') else 'inf'

        component_rows += f"""
            <tr class="{row_class}">
                <td>{comp['index'] + 1}</td>
                <td><strong>{comp['label'].upper()}</strong></td>
                <td>{comp['kappa']:.1f}</td>
                <td>{comp['rho']:.1f}</td>
                <td>{kr_ratio}</td>
                <td>{comp['variance_explained']:.2f}%</td>
                <td style="font-size: 0.85em;">{reason_str}</td>
            </tr>
"""

    metrics = classification_results.get('metrics', {})

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ME-ICA QC - {subject} {session}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        .metric-value {{
            color: #000;
            font-size: 1.1em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }}
        .signal-row {{
            background-color: #e8f5e9;
        }}
        .noise-row {{
            background-color: #ffebee;
        }}
        .info-box {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ME-ICA Denoising QC Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Session:</strong> {session}</p>
        <p><strong>Method:</strong> ME-ICA (Kundu et al. 2014 / tedana minimal tree)</p>

        <div class="summary">
            <h2>Classification Summary</h2>

            <div class="metric">
                <span class="metric-label">Total Components:</span>
                <span class="metric-value">{len(components)}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Signal:</span>
                <span class="metric-value" style="color: green;">{summary['n_signal']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Noise:</span>
                <span class="metric-value" style="color: red;">{summary['n_noise']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Signal %:</span>
                <span class="metric-value">{100*summary['n_signal']/len(components):.1f}%</span>
            </div>
        </div>

        <div class="summary">
            <h2>Thresholds & Metrics</h2>

            <div class="metric">
                <span class="metric-label">Kappa Elbow:</span>
                <span class="metric-value">{thresholds['kappa_elbow']:.1f}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Rho Elbow:</span>
                <span class="metric-value">{thresholds['rho_elbow']:.1f}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Rho/Kappa Scale:</span>
                <span class="metric-value">{thresholds['rho_kappa_scale']}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Median Kappa:</span>
                <span class="metric-value">{metrics.get('kappa_median', 0):.1f}</span>
            </div>

            <div class="metric">
                <span class="metric-label">Median Rho:</span>
                <span class="metric-value">{metrics.get('rho_median', 0):.1f}</span>
            </div>
        </div>

        <div class="info-box">
            <h3>ME-ICA Classification Logic (Minimal Tree)</h3>
            <ol>
                <li><strong>Reject</strong> if rho &gt; {thresholds['rho_kappa_scale']}&times;kappa (more S0-like than T2*-like)</li>
                <li><strong>Provisional accept</strong> if kappa &ge; kappa_elbow ({thresholds['kappa_elbow']:.1f}), else provisional reject</li>
                <li><strong>Accept</strong> if kappa &gt; 2&times;rho (strong BOLD rescue)</li>
                <li><strong>Provisional reject</strong> if rho &gt; rho_elbow ({thresholds['rho_elbow']:.1f})</li>
                <li><strong>Accept</strong> low-variance provisional rejects (&lt;0.1% single, &lt;1% cumulative)</li>
                <li><strong>Finalize</strong>: remaining provisional accepts &rarr; accepted, rest &rarr; rejected</li>
            </ol>
            <p><em>ICA run on optimally combined data (MELODIC), then kappa/rho computed from per-echo F-statistics.</em></p>
        </div>

        <h2>Kappa vs Rho</h2>
        <img src="figures/{scatter_fig.name}" alt="Kappa vs Rho scatter">

        <h2>Kappa Rank Plot</h2>
        <img src="figures/{rank_fig.name}" alt="Kappa rank plot">

        <h2>Classification Summary</h2>
        <img src="figures/{summary_fig.name}" alt="Classification summary">

        <h2>Component Details</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Class</th>
                    <th>Kappa</th>
                    <th>Rho</th>
                    <th>K/R Ratio</th>
                    <th>Var Expl</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
{component_rows}
            </tbody>
        </table>

        <h2>Signal Components (Retained)</h2>
        <p><strong>Indices (1-indexed):</strong> {', '.join(str(i+1) for i in summary['signal_components'])}</p>

        <h2>Noise Components (Removed)</h2>
        <p><strong>Indices (1-indexed):</strong> {', '.join(str(i+1) for i in summary['noise_components'])}</p>

        <h2>MELODIC Report</h2>
        <p>See <a href="{melodic_dir}/report.html">MELODIC HTML report</a> for spatial maps and timecourses.</p>

        <hr>
        <p style="text-align: center; color: #888; font-size: 0.9em;">
            Generated by Neurofaune ME-ICA denoising pipeline
        </p>
    </div>
</body>
</html>
"""

    report_file = output_dir / f'{subject}_{session}_meica_qc.html'
    with open(report_file, 'w') as f:
        f.write(html_content)

    # Save classification JSON
    json_file = output_dir / f'{subject}_{session}_meica_classification.json'

    # Make a serializable copy
    serializable = dict(classification_results)
    with open(json_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"  ME-ICA QC report: {report_file}")
    print(f"  Classification JSON: {json_file}")

    return report_file
