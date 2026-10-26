"""Single-session resting-state fMRI in hub-atlas parcels (mouse), with implant-aware QC.

For a session already registered to the hub (:func:`neurofaune.atlas.subject.register_subject_to_hub`):

1. ``prepare_run`` -- motion correction (mcflirt) and topup on the whole time
   series, using the reverse-phase-encode pair acquired on the run's grid.
2. ``register_bold_to_t2w`` -- rigid ANTs registration of the corrected BOLD
   mean to the subject T2w, then subject-space atlas layers (parcels, CCFv3,
   RABIES priors, dorsal-depth) are resampled into BOLD space.
3. ``denoise`` -- 6 motion parameters plus derivatives and aCompCor (CCFv3
   fiber tracts and ventricles, eroded inside the brain), band-passed
   together with the data (0.01-0.1 Hz by default), first volumes dropped.
4. ``parcel_timeseries`` / ``connectivity`` -- parcel means, Fisher-z
   correlation, homotopic (left-right pair) vs heterotopic connectivity.
5. ``dual_regression`` -- spatial then temporal regression against prior
   maps (RABIES ICs). The subject network maps are scored by their spatial
   correlation with each prior, with and without the dorsal layer the
   implant affects.

The model is a plain least-squares GLM. No smoothing is applied: parcels
already average, and smoothing would bleed the dorsal dropout into deeper
voxels.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# --------------------------------------------------------------------------- #
# Preparation and registration
# --------------------------------------------------------------------------- #
def prepare_run(bold: Path, pair: Dict[str, Path], work: Path, n_threads: int = 2) -> Dict[str, Path]:
    """mcflirt, then topup (Jacobian) applied to every volume with the same-grid pair."""
    from neurofaune.preprocess.qc.artifacts.implant import motion_correct
    from neurofaune.preprocess.utils.topup import apply_topup, run_topup_pair, sidecar

    work = Path(work)
    mc = motion_correct(Path(bold), work / "mcflirt")
    tu = run_topup_pair(pair["fwd"], pair["rev"], work / "topup", n_threads=n_threads)
    bmeta, fmeta = sidecar(Path(bold)), sidecar(pair["fwd"])
    index = 1 if fmeta["PhaseEncodingDirection"] == bmeta["PhaseEncodingDirection"] else 2
    corrected = work / Path(bold).name.replace("_bold.nii.gz", "_desc-mctopup_bold.nii.gz")
    if not corrected.exists():
        apply_topup(mc["path"], tu, index, corrected)
    img = nib.load(str(corrected))
    mean = work / corrected.name.replace("_bold.nii.gz", "_mean.nii.gz")
    nib.save(nib.Nifti1Image(img.get_fdata().mean(axis=3).astype(np.float32), img.affine), str(mean))
    return {"bold": corrected, "mean": mean, "motion_par": Path(str(mc["path"]).replace(".nii.gz", ".par"))}


def register_bold_to_t2w(bold_mean: Path, t2w_brain: Path, t2w_mask: Path, out_prefix: Path,
                         n_threads: int = 2) -> Dict[str, Path]:
    """Rigid (MI) BOLD-mean -> T2w; the EPI partial coverage makes anything looser fragile."""
    from neurofaune.atlas.label_transfer import _run

    p = str(out_prefix)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    out = {"affine": Path(f"{p}0GenericAffine.mat"), "warped": Path(f"{p}Warped.nii.gz")}
    if not out["affine"].exists():
        _run(["antsRegistration", "-d", "3", "--float", "1", "-o", f"[{p},{p}Warped.nii.gz]",
              "-n", "Linear", "-u", "1", "-w", "[0.005,0.995]",
              "-r", f"[{t2w_brain},{bold_mean},1]",
              "-t", "Rigid[0.1]", "-m", f"MI[{t2w_brain},{bold_mean},1,32,Regular,0.5]",
              "-c", "[500x250x100,1e-6,10]", "-f", "4x2x1", "-s", "2x1x0vox",
              "-x", f"[{t2w_mask},NULL]"], n_threads)
    return out


def to_bold(src: Path, bold_ref: Path, rigid: Path, out: Path, interpolation: str = "GenericLabel",
            time_series: bool = False, n_threads: int = 2) -> Path:
    """Subject T2w-space image -> BOLD grid (inverse of the BOLD->T2w rigid)."""
    from neurofaune.atlas.label_transfer import apply_transforms
    return apply_transforms(src, bold_ref, [f"[{rigid},1]"], out, interpolation, n_threads, time_series=time_series)


# --------------------------------------------------------------------------- #
# Denoising
# --------------------------------------------------------------------------- #
def bandpass(x: np.ndarray, tr: float, low: Optional[float], high: Optional[float], order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 / tr
    if low and high:
        b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype="band")
    elif low:
        b, a = butter(order, low / nyq, btype="high")
    elif high:
        b, a = butter(order, min(high / nyq, 0.99), btype="low")
    else:
        return x
    return filtfilt(b, a, x, axis=0)


def motion_confounds(par: np.ndarray) -> np.ndarray:
    """6 parameters + their temporal derivatives (24-parameter models overfit 300 volumes)."""
    d = np.vstack([np.zeros((1, par.shape[1])), np.diff(par, axis=0)])
    return np.column_stack([par, d])


def compcor(data: np.ndarray, mask: np.ndarray, n: int = 5) -> np.ndarray:
    """Top ``n`` principal components of the (detrended, standardised) signals in ``mask``."""
    x = data[mask].T.astype(float)                      # time x voxels
    if x.shape[1] < n:
        return np.zeros((x.shape[0], 0))
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    x = x[:, sd > 0] / sd[sd > 0]
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :n]


def denoise(data: np.ndarray, brain: np.ndarray, confounds: np.ndarray, tr: float,
            low: float = 0.01, high: float = 0.1, drop: int = 5) -> np.ndarray:
    """Regress confounds from band-passed data (confounds band-passed identically); time is last axis."""
    y = data[brain].T[drop:].astype(float)               # time x voxels
    c = confounds[drop:]
    y = bandpass(y - y.mean(axis=0), tr, low, high)
    c = bandpass(c - c.mean(axis=0), tr, low, high)
    X = np.column_stack([np.ones(len(c)), c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out = np.zeros(brain.shape + (len(resid),), np.float32)
    out[brain] = resid.T
    return out


# --------------------------------------------------------------------------- #
# Parcels and connectivity
# --------------------------------------------------------------------------- #
def parcel_timeseries(data: np.ndarray, parcels: np.ndarray, labels: Sequence[int],
                      min_voxels: int = 5) -> Tuple[np.ndarray, List[int]]:
    """Mean time series per parcel (time x parcels); parcels with too few voxels are dropped."""
    ts, kept = [], []
    for lab in labels:
        m = parcels == lab
        if m.sum() >= min_voxels:
            ts.append(data[m].mean(axis=0))
            kept.append(int(lab))
    return (np.column_stack(ts) if ts else np.zeros((data.shape[-1], 0))), kept


def connectivity(ts: np.ndarray) -> np.ndarray:
    """Fisher-z Pearson correlation; diagonal set to 0."""
    r = np.corrcoef(ts.T)
    np.fill_diagonal(r, 0.0)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def homotopy(z: np.ndarray, labels: Sequence[int], names: Dict[int, str]) -> Dict[str, Any]:
    """Homotopic (X_L <-> X_R) vs heterotopic connectivity -- the classic rsfMRI sanity check."""
    idx = {names[lab]: i for i, lab in enumerate(labels)}
    pairs = [(n, n[:-2] + "_R") for n in idx if n.endswith("_L") and n[:-2] + "_R" in idx]
    homo = {a[:-2]: float(z[idx[a], idx[b]]) for a, b in pairs}
    hom_set = {(idx[a], idx[b]) for a, b in pairs} | {(idx[b], idx[a]) for a, b in pairs}
    het = [z[i, j] for i in range(len(labels)) for j in range(len(labels))
           if i < j and (i, j) not in hom_set]
    return {"homotopic": homo, "median_homotopic_z": float(np.median(list(homo.values()))) if homo else None,
            "median_heterotopic_z": float(np.median(het)) if het else None}


# --------------------------------------------------------------------------- #
# Dual regression
# --------------------------------------------------------------------------- #
def dual_regression(data: np.ndarray, priors: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Spatial regression (priors -> time courses), then temporal (time courses -> maps).

    ``data`` (x, y, z, t), ``priors`` (x, y, z, k) on the same grid. Time
    courses are standardised before the temporal stage so the maps are
    comparable across components. Returns ``timecourses`` (t x k) and
    ``maps`` (x, y, z, k), z-like (beta / residual SD).
    """
    Y = data[mask].astype(float)                         # voxels x time
    P = priors[mask].astype(float)                       # voxels x k
    P = (P - P.mean(axis=0)) / (P.std(axis=0) + 1e-12)
    tc, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(P)), P]), Y, rcond=None)
    tc = tc[1:].T                                        # time x k
    tc = (tc - tc.mean(axis=0)) / (tc.std(axis=0) + 1e-12)
    X = np.column_stack([np.ones(len(tc)), tc])
    beta, *_ = np.linalg.lstsq(X, Y.T, rcond=None)       # (k+1) x voxels
    resid = Y.T - X @ beta
    sd = resid.std(axis=0, ddof=X.shape[1]) + 1e-12
    maps = np.zeros(mask.shape + (P.shape[1],), np.float32)
    maps[mask] = (beta[1:] / sd).T
    return {"timecourses": tc, "maps": maps}


def spatial_match(maps: np.ndarray, priors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Spatial correlation of each subject map with its own prior inside ``mask``."""
    return np.array([np.corrcoef(maps[..., k][mask], priors[..., k][mask])[0, 1] for k in range(maps.shape[-1])])


def slice_overlap(parcels: np.ndarray, labels: Sequence[int], slice_axis: int = 2) -> np.ndarray:
    """Cosine similarity of parcels' distributions over acquisition slices (parcels x parcels).

    Parcels that occupy the same slices share slice-wise noise (motion, segment
    and respiration effects in multi-shot EPI), which shows up as spurious
    connectivity. On the mouse pilot, edge FC correlated with this overlap at
    r = 0.37 / 0.54. Left-right homotopic pairs overlap almost completely
    (0.96-0.98), so homotopic coupling must be judged against it.
    """
    p = np.moveaxis(np.asarray(parcels), slice_axis, -1)
    ns = p.shape[-1]
    H = np.array([np.bincount(np.nonzero(p == k)[-1], minlength=ns) / max(int((p == k).sum()), 1) for k in labels], float)
    norm = np.sqrt((H ** 2).sum(axis=1))
    return (H @ H.T) / np.outer(norm, norm).clip(1e-12)


def remove_overlap_trend(z: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Edge FC minus its linear dependence on slice overlap (a diagnostic, not a correction).

    Removing the trend also removes any true coupling that co-varies with
    overlap -- homotopic pairs are both the most overlapping and the most
    plausibly coupled -- so read it as a lower bound.
    """
    iu = np.triu_indices(len(z), 1)
    b = np.polyfit(overlap[iu], z[iu], 1)
    out = z.copy()
    out[iu] = z[iu] - np.polyval(b, overlap[iu])
    out.T[iu] = out[iu]
    np.fill_diagonal(out, 0.0)
    return out
