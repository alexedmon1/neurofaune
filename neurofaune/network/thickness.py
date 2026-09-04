"""Plane-restricted cortical thickness (exploratory).

**Read this before using the output.** The native anatomical grid on this cohort is
0.125 x 0.125 x 0.8 mm true, i.e. 6.4x anisotropic. Rat cortex is roughly 1.2-2.0 mm
thick, so along the coarse axis the ribbon is crossed in 1.5-2.5 voxels — far too few
to place a surface. A full 3D thickness (FreeSurfer's surface method, or ANTs
DiReCT/KellyKapowski) run on this data would largely be measuring interpolation.

So thickness is solved **in the high-resolution plane only**, slice by slice along
the coarse axis, which is never differenced. In that plane the ribbon spans ~10-16
voxels, which does support the measurement. The cost is that the value is a
*plane-restricted apparent thickness*: it equals true thickness only where the
cortical normal lies in the plane, and overestimates it where the normal tilts out
of plane. Treat it as an exploratory regional index, comparable within a study and
across timepoints on the same acquisition — not as an absolute thickness, and not as
a FreeSurfer-equivalent number.

Method: Laplace field across the ribbon (Dirichlet 0 on the inner/white boundary, 1
on the outer/pial boundary), then the Yezzi-Prince pair of transport equations along
the normalised gradient, thickness = L0 + L1.

Returns a long-format table; the driver script writes it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from neurofaune.network.morphometry import coarse_axis

logger = logging.getLogger(__name__)

STRUCTURE = "cortical_gm"


def _laplace_2d(ribbon: np.ndarray, inner: np.ndarray, outer: np.ndarray,
                spacing: tuple[float, float], iters: int, tol: float) -> np.ndarray:
    """Solve grad^2 u = 0 on `ribbon` with u=0 on `inner`, u=1 on `outer` (2D)."""
    u = np.zeros(ribbon.shape, np.float64)
    u[outer] = 1.0
    hy2, hx2 = spacing[0] ** 2, spacing[1] ** 2
    denom = 2.0 * (1.0 / hy2 + 1.0 / hx2)
    for _ in range(iters):
        up = np.pad(u, 1, mode="edge")
        new = ((up[:-2, 1:-1] + up[2:, 1:-1]) / hy2
               + (up[1:-1, :-2] + up[1:-1, 2:]) / hx2) / denom
        new[inner] = 0.0
        new[outer] = 1.0
        delta = np.max(np.abs(new[ribbon] - u[ribbon])) if ribbon.any() else 0.0
        u = new
        if delta < tol:
            break
    u[inner] = 0.0
    u[outer] = 1.0
    return u


def _transport(nrm: tuple[np.ndarray, np.ndarray], ribbon: np.ndarray,
               start: np.ndarray, spacing: tuple[float, float], forward: bool,
               iters: int) -> np.ndarray:
    """Solve +/- N . grad L = 1 with L = 0 on `start`, by upwind iteration."""
    ny, nx = nrm
    if not forward:
        ny, nx = -ny, -nx
    hy, hx = spacing
    ay, ax = np.abs(ny) / hy, np.abs(nx) / hx
    L = np.zeros(ribbon.shape, np.float64)
    for _ in range(iters):
        Lp = np.pad(L, 1, mode="edge")
        # upwind neighbour: the one the streamline arrives from
        prev_y = np.where(ny > 0, Lp[:-2, 1:-1], Lp[2:, 1:-1])
        prev_x = np.where(nx > 0, Lp[1:-1, :-2], Lp[1:-1, 2:])
        new = np.divide(1.0 + ay * prev_y + ax * prev_x, ay + ax,
                        out=np.zeros_like(L), where=(ay + ax) > 1e-9)
        new[~ribbon] = 0.0
        new[start] = 0.0
        if np.max(np.abs(new - L)) < 1e-4:
            L = new
            break
        L = new
    return L


def slice_thickness(ribbon: np.ndarray, interior: np.ndarray, exterior: np.ndarray,
                    spacing: tuple[float, float], iters: int = 200,
                    tol: float = 1e-5) -> np.ndarray:
    """Thickness (mm) on one 2D slice; zero outside the ribbon."""
    if not ribbon.any():
        return np.zeros(ribbon.shape)
    u = _laplace_2d(ribbon, interior, exterior, spacing, iters, tol)
    gy, gx = np.gradient(u, spacing[0], spacing[1])
    mag = np.hypot(gy, gx)
    ok = mag > 1e-9
    ny = np.where(ok, gy / np.where(ok, mag, 1), 0.0)
    nx = np.where(ok, gx / np.where(ok, mag, 1), 0.0)
    L0 = _transport((ny, nx), ribbon, interior, spacing, True, iters)
    L1 = _transport((ny, nx), ribbon, exterior, spacing, False, iters)
    out = np.where(ribbon, L0 + L1, 0.0)
    return out


def thickness_volume(labels: np.ndarray, ribbon_ids: set[int], zooms, axis: int,
                     voxel_scale: float) -> np.ndarray:
    """Per-voxel plane-restricted thickness (mm) over the cortical ribbon."""
    ribbon = np.isin(labels, list(ribbon_ids))
    brain = labels > 0
    plane_axes = [a for a in range(3) if a != axis]
    spacing = tuple(float(zooms[a]) / voxel_scale for a in plane_axes)

    out = np.zeros(labels.shape, np.float32)
    for k in range(labels.shape[axis]):
        sl = [slice(None)] * 3
        sl[axis] = k
        sl = tuple(sl)
        rib = ribbon[sl]
        if not rib.any():
            continue
        out[sl] = slice_thickness(rib, brain[sl] & ~rib, ~brain[sl], spacing)
    return out


def compute_subject_thickness(
    dseg: Path,
    ribbon_ids: set[int],
    labels_df: pd.DataFrame | None = None,
    voxel_scale: float = 1.0,
    plane_axis: int | None = None,
) -> pd.DataFrame:
    """Per-cortical-region plane-restricted thickness for one subject-session.

    Long format: ``region_id, region, mean_thickness_mm, median_thickness_mm,
    sd_thickness_mm, n_voxels, method, excluded_axis, excluded_axis_mm``. The
    ``method`` and ``excluded_axis`` columns travel with the numbers on purpose —
    this is not a FreeSurfer-equivalent thickness and the output should not be able
    to lose that context.
    """
    if not ribbon_ids:
        logger.warning("No cortical ribbon labels supplied; skipping thickness")
        return pd.DataFrame()

    img = nib.load(str(dseg))
    labels = np.asarray(img.dataobj).astype(np.int32)
    zooms = img.header.get_zooms()[:3]
    axis = int(plane_axis) if plane_axis is not None else coarse_axis(zooms)
    true_mm = tuple(round(float(z) / voxel_scale, 4) for z in zooms)
    logger.info(
        "Thickness: true voxel %s mm, solving in-plane and excluding axis %d (%s mm)",
        true_mm, axis, true_mm[axis],
    )

    thickness = thickness_volume(labels, ribbon_ids, zooms, axis, voxel_scale)
    names = (dict(zip(labels_df["Labels"], labels_df["roi_name"], strict=False))
             if labels_df is not None else {})

    rows = []
    for region_id in sorted(ribbon_ids):
        values = thickness[(labels == region_id) & (thickness > 0)]
        if values.size == 0:
            continue
        rows.append({
            "region_id": region_id,
            "region": names.get(region_id, f"region_{region_id}"),
            "mean_thickness_mm": float(values.mean()),
            "median_thickness_mm": float(np.median(values)),
            "sd_thickness_mm": float(values.std()),
            "n_voxels": int(values.size),
            "method": "laplace-2d-inplane",
            "excluded_axis": axis,
            "excluded_axis_mm": true_mm[axis],
        })
    return pd.DataFrame(rows)
