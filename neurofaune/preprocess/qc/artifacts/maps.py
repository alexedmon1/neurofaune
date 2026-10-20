"""Artifact metrics that need no external tools.

Everything here works on arrays already resampled onto one reference grid
(the distortion-free T2w). Depths are in true millimetres: the caller passes
``voxel_scale`` so that 10x-inflated headers are undone.
"""

from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np


def canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """RAS-canonical view of an image (axis 2 = dorsal for animal-RAS data)."""
    return nib.as_closest_canonical(img)


def dorsal_depth_mm(mask: np.ndarray, voxel_mm_z: float) -> np.ndarray:
    """Depth below the dorsal brain surface, per voxel, in mm.

    ``mask`` must be in RAS-canonical orientation, so axis 2 runs ventral ->
    dorsal. For every (x, y) column the dorsal surface is the highest mask
    voxel, and a voxel's depth is its distance below that surface. This is the
    natural axis for a skull-mounted implant: the array sits on the dorsal
    surface and everything it does to the image falls off with depth.

    Returns NaN outside the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    z = np.arange(mask.shape[2])[None, None, :]
    top = np.where(mask.any(axis=2), np.where(mask, z, -1).max(axis=2), -1)
    depth = (top[:, :, None] - z) * float(voxel_mm_z)
    return np.where(mask, depth, np.nan)


def ventral_height_mm(mask: np.ndarray, voxel_mm_z: float) -> np.ndarray:
    """Height above the ventral brain surface, per voxel, in mm (RAS-canonical).

    The internal control for :func:`dorsal_depth_mm`. The ventral surface has no
    implant but has its own susceptibility sources (skull base, sinuses, ear
    canals). An effect seen only near the dorsal surface points at the implant;
    one seen at both surfaces is generic surface/partial-volume behaviour.
    """
    mask = np.asarray(mask, dtype=bool)
    z = np.arange(mask.shape[2])[None, None, :]
    big = mask.shape[2] + 1
    bottom = np.where(mask.any(axis=2), np.where(mask, z, big).min(axis=2), big)
    height = (z - bottom[:, :, None]) * float(voxel_mm_z)
    return np.where(mask, height, np.nan)


def axis_position_mm(img: nib.Nifti1Image, axis: int, voxel_scale: float = 1.0) -> np.ndarray:
    """World coordinate along a RAS axis (0=R, 1=A, 2=S) per voxel, in true mm."""
    ijk = np.indices(img.shape[:3]).reshape(3, -1)
    xyz = img.affine[:3, :3] @ ijk + img.affine[:3, 3:4]
    return (xyz[axis] / voxel_scale).reshape(img.shape[:3])


def depth_profile(values: np.ndarray, depth_mm: np.ndarray, mask: np.ndarray,
                  bin_mm: float = 0.25, max_mm: Optional[float] = None) -> List[Dict[str, float]]:
    """Median / quartiles of ``values`` in depth bins inside ``mask``.

    NaNs in ``values`` are ignored (e.g. voxels outside an EPI slab).
    """
    sel = np.asarray(mask, dtype=bool) & np.isfinite(depth_mm) & np.isfinite(values)
    d, v = depth_mm[sel], values[sel]
    if d.size == 0:
        return []
    top = float(max_mm) if max_mm is not None else float(d.max())
    rows = []
    for lo in np.arange(0.0, top + 1e-9, bin_mm):
        inbin = (d >= lo) & (d < lo + bin_mm)
        if not inbin.any():
            continue
        x = v[inbin]
        rows.append({
            "depth_mm": float(lo + bin_mm / 2), "n": int(x.size),
            "median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "mean": float(np.mean(x)),
        })
    return rows


def tsnr(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """Temporal SNR: mean / standard deviation over time (0 where std is 0)."""
    data = np.asarray(data, dtype=float)
    mean = data.mean(axis=axis)
    std = data.std(axis=axis, ddof=1)
    out = np.zeros_like(mean)
    np.divide(mean, std, out=out, where=std > 0)
    return out


def normalised_ratio(a: np.ndarray, b: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(a / median_a) / (b / median_b)`` with medians over ``reference`` voxels.

    With ``a`` a gradient-echo and ``b`` a spin-echo image of the same slab,
    values well below 1 mark T2*-dephasing signal loss (dropout) that the spin
    echo refocuses. Normalising by a deep-brain reference removes the global
    contrast and scaling differences between the two sequences.
    """
    reference = np.asarray(reference, dtype=bool)
    ma, mb = np.median(a[reference]), np.median(b[reference])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (np.asarray(a, float) / ma) / (np.asarray(b, float) / mb)
    return np.where(np.isfinite(r), r, np.nan)


def masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson r between two images inside ``mask`` (finite voxels only)."""
    sel = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if sel.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[sel], b[sel])[0, 1])


def shell_above(mask: np.ndarray, voxel_mm_z: float, lo_mm: float, hi_mm: float) -> np.ndarray:
    """Voxels lying ``lo_mm``-``hi_mm`` above the dorsal brain surface, per column.

    In RAS-canonical orientation. This is where the implant sits.
    """
    mask = np.asarray(mask, dtype=bool)
    z = np.arange(mask.shape[2])[None, None, :]
    has = mask.any(axis=2)
    top = np.where(has, np.where(mask, z, -1).max(axis=2), -1)
    height = (z - top[:, :, None]) * float(voxel_mm_z)
    return has[:, :, None] & (height > lo_mm) & (height <= hi_mm)


def summarise(values: np.ndarray, mask: np.ndarray, quantiles: Sequence[float] = (5, 50, 95)) -> Dict[str, float]:
    sel = np.asarray(mask, dtype=bool) & np.isfinite(values)
    x = values[sel]
    out = {"n": int(x.size)}
    if x.size:
        for q in quantiles:
            out[f"p{int(q)}"] = float(np.percentile(x, q))
        out["mean"] = float(x.mean())
    return out
