"""Skull-derived stereotaxic frame (bregma origin) for a template, from coordinate maps.

Perens et al. 2023 ship, for every brain voxel of their MRI template, its
coordinates in a skull-derived stereotaxic system. Bregma and lambda were
found where fitted coronal, sagittal and lambdoid sutures meet on micro-CT,
then carried onto the MRI template (``mri_coords_{x,y,z}.nii.gz``, mm):

- x = |medio-lateral| distance from the midline (unsigned)
- y = antero-posterior from bregma (+ rostral)
- z = dorso-ventral below bregma (+ ventral)

The maps exist only inside the brain; bregma itself lies on the skull, just
above it. So the frame is recovered by least squares as an affine map from
world position to (ML, AP, DV) over the brain voxels, and bregma is the point
where that map is zero. The residual of the fit shows whether the maps really
are one rigid, scaled frame.
"""

from typing import Dict, Tuple

import numpy as np


def fit_frame(coords: Tuple[np.ndarray, np.ndarray, np.ndarray], hemisphere: np.ndarray,
              affine: np.ndarray, max_voxels: int = 400_000, seed: int = 0) -> Dict:
    """Fit world(mm, RAS) -> stereotaxic (ML, AP, DV) as an affine map.

    Parameters
    ----------
    coords : (x, y, z) arrays
        Unsigned ML, signed AP, signed DV, in mm; 0 outside the brain.
    hemisphere : array
        Non-zero on one hemisphere, used to sign ML.
    affine : 4x4
        Voxel index -> world (mm) for the arrays.

    Returns
    -------
    dict
        ``world_to_stereo`` (4x4), ``bregma_world_mm`` (3,), unit axis
        directions in world space (``ml_axis`` = +right after sign fixing,
        ``ap_axis`` = +rostral, ``dv_axis`` = +ventral), per-axis scale
        (stereotaxic mm per world mm) and fit residuals (mm).
    """
    x, y, z = (np.asarray(c, float) for c in coords)
    inside = (x != 0) | (y != 0) | (z != 0)
    idx = np.argwhere(inside)
    rng = np.random.default_rng(seed)
    if len(idx) > max_voxels:
        idx = idx[rng.choice(len(idx), max_voxels, replace=False)]
    i, j, k = idx.T
    side = np.where(np.asarray(hemisphere)[i, j, k] != 0, 1.0, -1.0)
    target = np.column_stack([x[i, j, k] * side, y[i, j, k], z[i, j, k]])
    world = (affine @ np.column_stack([idx, np.ones(len(idx))]).T).T[:, :3]
    design = np.column_stack([world, np.ones(len(world))])
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)      # (4, 3)
    m = np.eye(4)
    m[:3, :3] = sol[:3].T
    m[:3, 3] = sol[3]
    resid = target - design @ sol
    # +ML should point to the subject's right (+x world).
    if m[0, 0] < 0:
        m[0, :] *= -1
    bregma = np.linalg.solve(m[:3, :3], -m[:3, 3])
    inv = np.linalg.inv(m[:3, :3])
    axes = {name: inv[:, n] / np.linalg.norm(inv[:, n]) for n, name in enumerate(("ml_axis", "ap_axis", "dv_axis"))}
    return {
        "world_to_stereo": m,
        "bregma_world_mm": bregma,
        **axes,
        "scale_stereo_per_world": np.linalg.norm(m[:3, :3], axis=1),
        "residual_mm": {"rms": float(np.sqrt((resid ** 2).mean())),
                        "max_abs": float(np.abs(resid).max()),
                        "per_axis_rms": np.sqrt((resid ** 2).mean(axis=0)).tolist()},
    }


def stereo_to_world(frame: Dict, points_stereo: np.ndarray) -> np.ndarray:
    """Stereotaxic (ML, AP, DV) mm -> world mm."""
    inv = np.linalg.inv(frame["world_to_stereo"])
    p = np.atleast_2d(np.asarray(points_stereo, float))
    return (inv @ np.column_stack([p, np.ones(len(p))]).T).T[:, :3]


def world_to_stereo(frame: Dict, points_world: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points_world, float))
    return (frame["world_to_stereo"] @ np.column_stack([p, np.ones(len(p))]).T).T[:, :3]
