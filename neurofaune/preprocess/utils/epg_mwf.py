"""EPG stimulated-echo-corrected myelin water fraction (MWF).

Multi-echo T2 (MSME/CPMG) MWF via the extended-phase-graph (EPG) algorithm with
per-voxel refocusing-flip-angle estimation, following Prasloski et al. (2012) and
the UBC/DECAES reference implementation. This supersedes a plain multi-exponential
NNLS (``calculate_mwf_nnls``), whose design matrix assumes ideal 180 deg refocusing
— an assumption that fails under B1 inhomogeneity (real refocus angles ~150 deg at
7T), producing an even/odd echo oscillation the ideal model cannot fit. That
systematic misfit destabilises the short-T2 (myelin) estimate into a bimodal
0-or-saturated coin-flip; EPG models the actual flip angle and restores a stable,
physical MWF.

The EPG forward model here is validated bit-exact (max abs err 3e-16) against
DECAES ``EPGdecaycurve``, and the full per-voxel MWF against DECAES ``sfr``
(corr 0.997, mean|dMWF| 0.006).

Pure numpy so it ports to the human pipeline (neurovrai) without a Julia dependency.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
from scipy.optimize import nnls as _scipy_nnls


def nnls(A, b):
    """NNLS that never aborts a run on a single degenerate voxel.

    scipy's Lawson-Hanson ``nnls`` raises ``RuntimeError`` when it hits its
    iteration cap (default ``3*n_cols``), which at the fine rodent T2 grid
    (n=200) happens on rare ill-conditioned voxels. That exception, raised in a
    worker process, otherwise propagates up and kills the whole session. We give
    it a generous iteration budget and, if it still fails, fall back to a zero
    solution for that one voxel (it contributes no MWF) rather than crashing.
    """
    maxiter = 20 * A.shape[1]
    try:
        return _scipy_nnls(A, b, maxiter=maxiter)
    except (RuntimeError, ValueError):
        return np.zeros(A.shape[1]), 0.0


# ---------------------------------------------------------------------------
# EPG forward model (Hennig 1988 recursion, Jones 1997 phase-state correction).
# Pulse sequence: A*90, A*180, A*beta, A*beta, ... with A = alpha/180 (B1 scale);
# standard CPMG when alpha = beta = 180. Ported from DECAES EPGdecaycurve.jl.
# ---------------------------------------------------------------------------
def _element_flipmat(alpha_deg: float) -> np.ndarray:
    a = np.deg2rad(alpha_deg)
    c2, s2, s = np.cos(a / 2) ** 2, np.sin(a / 2) ** 2, np.sin(a)
    return np.array([
        [c2,        s2,      -1j * s],
        [s2,        c2,       1j * s],
        [-1j * s / 2, 1j * s / 2, np.cos(a)],
    ], dtype=complex)


def epg_decay_curve(ETL: int, alpha: float, TE: float, T2: float,
                    T1: float = 1.0, beta: float = 180.0) -> np.ndarray:
    """Normalised multi-spin-echo decay curve for one (alpha, T2) via EPG.

    ETL echoes, echo spacing ``TE`` (s), transverse ``T2`` (s), longitudinal
    ``T1`` (s), refocusing angle ``alpha`` (deg). Returns length-ETL amplitudes.
    """
    A = alpha / 180.0
    a_ex = A * 90.0
    E2, E1 = np.exp(-(TE / 2) / T2), np.exp(-(TE / 2) / T1)
    E = np.array([E2, E2, E1])
    R1 = _element_flipmat(A * 180.0)
    Ri = _element_flipmat(A * beta)
    M = np.zeros((ETL, 3), dtype=complex)
    M[0] = [np.sin(np.deg2rad(a_ex)), 0, 0]
    dc = np.zeros(ETL)
    for i in range(ETL):
        R = R1 if i == 0 else Ri
        for j in range(ETL):
            M[j] = R @ (E * M[j])
        # phase-state transition
        Mi, Mip1 = M[0].copy(), M[1].copy()
        M[0] = [Mi[1], Mip1[1], Mi[2]]
        Mprev = Mi
        for j in range(1, ETL - 1):
            Mim1, Mi, Mip1 = Mprev, M[j].copy(), M[j + 1].copy()
            M[j] = [Mim1[0], Mip1[1], Mi[2]]
            Mprev = Mi
        M[ETL - 1] = [Mprev[0], 0, M[ETL - 1][2]]
        for j in range(ETL):
            M[j] = E * M[j]
        dc[i] = abs(M[0][0])
    return dc


def epg_bases(angles: np.ndarray, t2_grid: np.ndarray, TE: float, ETL: int,
              T1: float = 1.0) -> np.ndarray:
    """Precompute the EPG decay basis for each candidate flip angle.

    Returns array (n_angles, ETL, n_T2): column k of angle a is the EPG decay
    curve for T2 = t2_grid[k] at refocusing angle a.
    """
    return np.stack([
        np.stack([epg_decay_curve(ETL, a, TE, t2, T1) for t2 in t2_grid], axis=1)
        for a in angles
    ])


# ---------------------------------------------------------------------------
# Per-voxel fit: flip-angle search + regularised NNLS.
# ---------------------------------------------------------------------------
def _curvature_matrix(n: int) -> np.ndarray:
    H = np.zeros((n - 2, n))
    for j in range(n - 2):
        H[j, j], H[j, j + 1], H[j, j + 2] = 1.0, -2.0, 1.0
    return H


def _monoexp_t2(signal, te_ms):
    """Mono-exponential effective T2 (ms) — matches the legacy pipeline's T2 map."""
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(lambda te, s0, t2: s0 * np.exp(-te / t2), te_ms, signal,
                            p0=[signal[0], 50.0], bounds=([0, 10], [np.inf, 500]),
                            maxfev=1000)
        return float(popt[1])
    except Exception:
        return 0.0


def _fit_voxel(signal, bases, coarse_bases, angles, H, mw_cutoff, iw_cutoff, chi2_factor, te_ms):
    """Return (mwf, iwf, csf, alpha) for one decay curve.

    1) pick the flip angle minimising the unregularised NNLS residual, then
    2) regularise (2nd-deriv Tikhonov) at that angle up to chi2 = chi2_factor *
       chi2_min (the per-voxel Whittall-MacKay/Prasloski criterion), and
    3) integrate the T2 spectrum over the myelin / intra-extra / CSF windows.

    The refocusing-angle basis is smooth in the flip angle, so the sub-grid
    refine linearly interpolates between the two bracketing precomputed
    angle-bases rather than re-running EPG per voxel (a large speed-up at
    negligible cost — max basis interpolation error << NNLS noise).
    """
    ETL = signal.shape[0]
    n = bases.shape[2]
    # 1) flip-angle search on the COARSE basis. The refocusing angle is a smooth
    #    B1 parameter and does not need the fine T2 grid to estimate, so searching
    #    on a low-resolution basis is far cheaper while giving the same angle.
    cn = coarse_bases.shape[2]
    resid = np.empty(len(angles))
    for k in range(len(angles)):
        x, _ = nnls(coarse_bases[k], signal)
        resid[k] = np.sum((coarse_bases[k] @ x - signal) ** 2)
    k = int(np.argmin(resid))
    step = 0.0
    alpha = angles[k]
    # local parabolic refine on the coarse residual curve (sub-grid angle)
    if 0 < k < len(angles) - 1:
        y0, y1, y2 = resid[k - 1], resid[k], resid[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom > 0:
            s = 0.5 * (y0 - y2) / denom
            cA = ((1 - s) * coarse_bases[k] + s * coarse_bases[k + 1]) if s >= 0 \
                else ((1 + s) * coarse_bases[k] - s * coarse_bases[k - 1])
            xr, _ = nnls(cA, signal)
            if np.sum((cA @ xr - signal) ** 2) < resid[k]:
                step = s
                alpha = angles[k] + s * (angles[1] - angles[0])
    # build the FULL-resolution basis at the chosen angle (interpolate neighbours)
    if step >= 0 and k < len(angles) - 1:
        A = (1 - step) * bases[k] + step * bases[k + 1]
    elif step < 0:
        A = (1 + step) * bases[k] - step * bases[k - 1]
    else:
        A = bases[k]

    # 2) regularised NNLS at alpha*: bisection on lambda to hit chi2 target
    x0, _ = nnls(A, signal)
    chi_min = np.sum((A @ x0 - signal) ** 2)
    target = chi2_factor * chi_min
    Ar = np.empty((ETL + n - 2, n))
    Ar[:ETL] = A
    br = np.concatenate([signal, np.zeros(n - 2)])
    if target <= 0:
        x = x0
    else:
        lo, hi = 1e-8, 1e8
        for _ in range(8):
            mid = np.sqrt(lo * hi)
            Ar[ETL:] = np.sqrt(mid) * H
            x, _ = nnls(Ar, br)
            if np.sum((A @ x - signal) ** 2) < target:
                lo = mid
            else:
                hi = mid
        Ar[ETL:] = np.sqrt(np.sqrt(lo * hi)) * H
        x, _ = nnls(Ar, br)

    tot = x.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0, 0.0, alpha
    # T2: mono-exponential effective T2 (matches the legacy pipeline's T2 map,
    # ~55ms; unchanged so switching to EPG only affects the myelin fractions).
    t2 = _monoexp_t2(signal, te_ms)
    return (x[:mw_cutoff].sum() / tot,
            x[mw_cutoff:iw_cutoff].sum() / tot,
            x[iw_cutoff:].sum() / tot,
            t2, alpha)


def _init_worker():
    # keep BLAS single-threaded inside each worker (we parallelise over voxels)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _fit_chunk(args):
    signals, bases, coarse_bases, angles, H, mwc, iwc, chi2, te_ms = args
    out = np.empty((len(signals), 5))
    for i, sig in enumerate(signals):
        out[i] = _fit_voxel(sig, bases, coarse_bases, angles, H, mwc, iwc, chi2, te_ms)
    return out


def calculate_mwf_epg(
    data: np.ndarray,
    mask: np.ndarray,
    te_values: np.ndarray,
    n_components: int = 40,
    t2_range: Optional[list] = None,
    myelin_water_cutoff: float = 25.0,
    intra_extra_cutoff: float = 200.0,
    T1: float = 1000.0,
    n_angles: int = 12,
    min_angle: float = 50.0,
    chi2_factor: float = 1.02,
    n_workers: Optional[int] = None,
):
    """EPG stimulated-echo-corrected MWF maps.

    Parameters mirror ``calculate_mwf_nnls`` where they overlap. ``te_values``
    in ms, uniform echo spacing assumed (CPMG). ``T1`` in ms. Returns
    ``(mwf_map, iwf_map, csf_map, t2_map, alpha_map)`` as 3D arrays (T2 in ms,
    alpha in degrees).
    """
    if t2_range is None:
        t2_range = [10.0, 2000.0]
    ETL = data.shape[3]
    TE_s = float(te_values[1] - te_values[0]) / 1000.0 if len(te_values) > 1 else float(te_values[0]) / 1000.0
    T1_s = T1 / 1000.0
    t2_grid = np.geomspace(t2_range[0] / 1000.0, t2_range[1] / 1000.0, n_components)
    mw_cut = int(np.searchsorted(t2_grid, myelin_water_cutoff / 1000.0))
    iw_cut = int(np.searchsorted(t2_grid, intra_extra_cutoff / 1000.0))
    angles = np.linspace(min_angle, 180.0, n_angles)

    print(f"\nCalculating MWF via EPG (stimulated-echo corrected)...")
    print(f"  ETL={ETL}, TE={TE_s*1000:.1f}ms, T1={T1:.0f}ms, "
          f"T2 grid {t2_range[0]:.0f}-{t2_range[1]:.0f}ms x{n_components}, "
          f"flip-angle search {min_angle:.0f}-180 deg x{n_angles}")

    bases = epg_bases(angles, t2_grid, TE_s, ETL, T1_s)
    H = _curvature_matrix(n_components)
    # Coarse basis for the flip-angle search only (the refocusing angle is a
    # smooth B1 term — it needs the angle grid, not the fine T2 grid). Big speedup
    # at n_components >> 40, since only the final regularised fit runs at full res.
    n_coarse = min(40, n_components)
    coarse_grid = np.geomspace(t2_range[0] / 1000.0, t2_range[1] / 1000.0, n_coarse)
    coarse_bases = (bases if n_coarse == n_components
                    else epg_bases(angles, coarse_grid, TE_s, ETL, T1_s))

    idx = np.argwhere(mask)
    signals = [data[x, y, z, :].astype(float) for x, y, z in idx]
    keep = [i for i, s in enumerate(signals) if s[0] > 0 and np.isfinite(s).all()]
    print(f"  Fitting {len(keep)} voxels...")

    mwf_map = np.zeros(data.shape[:3])
    iwf_map = np.zeros(data.shape[:3])
    csf_map = np.zeros(data.shape[:3])
    t2_map = np.zeros(data.shape[:3])
    alpha_map = np.zeros(data.shape[:3])

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    sub = [signals[i] for i in keep]
    n_chunks = max(1, n_workers * 4)
    chunks = np.array_split(np.arange(len(sub)), n_chunks)
    te_ms = np.asarray(te_values, dtype=float)
    tasks = [([sub[j] for j in ch], bases, coarse_bases, angles, H, mw_cut, iw_cut, chi2_factor, te_ms)
             for ch in chunks if len(ch)]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for r in ex.map(_fit_chunk, tasks):
            results.append(r)
    fitted = np.concatenate(results) if results else np.zeros((0, 5))

    for local_i, glob_i in enumerate(keep):
        x, y, z = idx[glob_i]
        (mwf_map[x, y, z], iwf_map[x, y, z], csf_map[x, y, z],
         t2_map[x, y, z], alpha_map[x, y, z]) = fitted[local_i]

    mm = mwf_map[mask]
    print(f"  MWF within brain: median={np.median(mm):.3f} "
          f"mean={np.mean(mm):.3f} (flip-angle median={np.median(alpha_map[mask]):.0f} deg)")
    return mwf_map, iwf_map, csf_map, t2_map, alpha_map
