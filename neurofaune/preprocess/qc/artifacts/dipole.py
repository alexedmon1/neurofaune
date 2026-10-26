"""Point-dipole model of a compact susceptibility source (e.g. an implant connector) in a B0 map.

A small magnetised object in B0 adds a field offset (along B0) of

    dB(r) = m * (3 cos^2(theta) - 1) / |r - r0|^3

where theta is the angle between (r - r0) and B0. Fitting r0 and m, plus a
smooth background (constant + linear + quadratic terms), to a measured B0 map
says where the source is and how fast its field falls off with distance. That
fall-off is what carries over from a phantom to an animal: given how far a
connector sits from the cortex, it predicts the field there.

The model ignores the object's size and shape, so it is only meaningful a few
object radii away. Voxels nearer than ``exclude_radius_mm`` and voxels with no
field estimate (signal voids, where the B0 map is zero) are left out.
"""

from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares


def _design(X: np.ndarray, q: np.ndarray, b0_axis: np.ndarray) -> np.ndarray:
    r0, m, c, g, h = q[:3], q[3], q[4], q[5:8], q[8:11]
    d = X - r0
    r = np.linalg.norm(d, axis=1) + 1e-9
    cos = (d @ b0_axis) / r
    return m * (3 * cos ** 2 - 1) / r ** 3 + c + X @ g + (X ** 2) @ h


def background_only(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X, X ** 2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def fit_point_dipole(field_hz: np.ndarray, world_mm: np.ndarray, b0_axis=(0.0, 0.0, 1.0),
                     starts: Optional[np.ndarray] = None, exclude_radius_mm: float = 0.0,
                     loss: str = "soft_l1", f_scale: float = 20.0) -> Dict:
    """Fit dipole + smooth background to field samples ``field_hz`` at ``world_mm`` (N x 3)."""
    b0 = np.asarray(b0_axis, float) / np.linalg.norm(b0_axis)
    X, y = np.asarray(world_mm, float), np.asarray(field_hz, float)
    starts = np.atleast_2d(starts if starts is not None else X[np.argmax(np.abs(y))])
    best = None
    for s in starts:
        keep = np.linalg.norm(X - s, axis=1) >= exclude_radius_mm
        q0 = np.r_[s, np.sign(y[keep].mean() or 1) * 1e3, 0.0, np.zeros(3), np.zeros(3)]
        r = least_squares(lambda q: _design(X[keep], q, b0) - y[keep], q0, loss=loss, f_scale=f_scale)
        if best is None or r.cost < best[0].cost:
            best = (r, keep)
    r, keep = best
    resid = _design(X[keep], r.x, b0) - y[keep]
    bg = background_only(X[keep], y[keep]) - y[keep]
    return {"centre_mm": r.x[:3].tolist(), "moment_hz_mm3": float(r.x[3]),
            "rms_dipole_hz": float(np.sqrt((resid ** 2).mean())),
            "rms_background_only_hz": float(np.sqrt((bg ** 2).mean())),
            "rms_data_hz": float(y[keep].std()), "n": int(keep.sum()),
            "params": r.x.tolist(), "b0_axis": b0.tolist()}


def dipole_field(world_mm: np.ndarray, centre_mm, moment_hz_mm3: float, b0_axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The fitted dipole's own field (no background) at ``world_mm``."""
    b0 = np.asarray(b0_axis, float) / np.linalg.norm(b0_axis)
    d = np.asarray(world_mm, float) - np.asarray(centre_mm, float)
    r = np.linalg.norm(d, axis=1) + 1e-9
    return moment_hz_mm3 * (3 * ((d @ b0) / r) ** 2 - 1) / r ** 3


def max_field_at_distance(moment_hz_mm3: float, r_mm) -> np.ndarray:
    """Peak |dB| of a dipole at distance r (on the B0 axis, where 3cos^2-1 = 2)."""
    return 2 * abs(moment_hz_mm3) / np.asarray(r_mm, float) ** 3


def r2star_loglinear(echoes: np.ndarray, te_s: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """R2* (1/s) by log-linear fit over echoes (last axis), voxel-wise."""
    s = np.clip(np.asarray(echoes, float), floor, None)
    te = np.asarray(te_s, float)
    tc = te - te.mean()
    return -((np.log(s) - np.log(s).mean(axis=-1, keepdims=True)) @ tc) / (tc @ tc)
