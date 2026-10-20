"""Per-parcel comparison of two versions of a parcellation on the same grid.

Answers "how far did each region move?" when a parcellation is rebuilt, e.g.
CCFv3 labels re-warped into Antwerp space through a T2w bridge instead of a
direct registration of the CCFv3 autofluorescence template onto the MRI.
Distances are in world mm (RAS) from the affine.
"""

from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def _mean_surface_distance_mm(a: np.ndarray, b: np.ndarray, zooms) -> Optional[float]:
    """Symmetric mean distance between the boundaries of two masks (mm)."""
    if not a.any() or not b.any():
        return None
    sa, sb = _surface(a), _surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=zooms)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=zooms)[sb]
    return float((da.sum() + db.sum()) / (da.size + db.size))


def compare_parcellations(old: nib.Nifti1Image, new: nib.Nifti1Image,
                          names: Optional[Dict[int, str]] = None) -> List[Dict]:
    """One row per label present in either image.

    Columns: ``label``, ``name``, ``old_mm3``, ``new_mm3``, ``dice``,
    ``centroid_shift_mm`` and its RAS components ``shift_R_mm`` (+ = moved
    right), ``shift_A_mm`` (+ = rostral), ``shift_S_mm`` (+ = dorsal), and
    ``mean_boundary_distance_mm``.
    """
    if old.shape[:3] != new.shape[:3] or not np.allclose(old.affine, new.affine, atol=1e-4):
        raise ValueError("parcellations must share a grid")
    a = np.asanyarray(old.dataobj).astype(np.int64)
    b = np.asanyarray(new.dataobj).astype(np.int64)
    zooms = tuple(float(z) for z in old.header.get_zooms()[:3])
    vox = float(np.prod(zooms))
    labels = sorted((set(np.unique(a)) | set(np.unique(b))) - {0})
    rows = []
    for lab in labels:
        ma, mb = a == lab, b == lab
        row: Dict = {"label": int(lab), "name": (names or {}).get(int(lab), str(lab)),
                     "old_mm3": float(ma.sum() * vox), "new_mm3": float(mb.sum() * vox)}
        denom = ma.sum() + mb.sum()
        row["dice"] = float(2 * (ma & mb).sum() / denom) if denom else None
        if ma.any() and mb.any():
            ca = old.affine @ np.r_[np.argwhere(ma).mean(axis=0), 1]
            cb = new.affine @ np.r_[np.argwhere(mb).mean(axis=0), 1]
            d = (cb - ca)[:3]
            row.update({"centroid_shift_mm": float(np.linalg.norm(d)), "shift_R_mm": float(d[0]),
                        "shift_A_mm": float(d[1]), "shift_S_mm": float(d[2])})
        else:
            row.update({"centroid_shift_mm": None, "shift_R_mm": None, "shift_A_mm": None, "shift_S_mm": None})
        row["mean_boundary_distance_mm"] = _mean_surface_distance_mm(ma, mb, zooms)
        rows.append(row)
    return rows


def ap_landmark_scale(reference: nib.Nifti1Image, warped: nib.Nifti1Image,
                      landmark_ids: Dict[str, set], pairs: List[tuple]) -> List[Dict]:
    """Rostro-caudal (RAS y) distance between landmark-structure centroids, warped vs reference.

    A ratio below 1 means the warp compressed that span. Uses label centroids,
    so it needs no hand-placed points; ``landmark_ids`` maps a name to the
    label IDs making up that structure in both images.
    """
    def y(img, ids):
        d = np.asanyarray(img.dataobj)
        m = np.isin(d, list(ids))
        return float((img.affine @ np.r_[np.argwhere(m).mean(0), 1])[1]) if m.any() else None

    ref_y = {k: y(reference, v) for k, v in landmark_ids.items()}
    war_y = {k: y(warped, v) for k, v in landmark_ids.items()}
    rows = []
    for a, b in pairs:
        if None in (ref_y[a], ref_y[b], war_y[a], war_y[b]):
            continue
        r, w = ref_y[a] - ref_y[b], war_y[a] - war_y[b]
        rows.append({"pair": f"{a}->{b}", "reference_mm": r, "warped_mm": w, "ratio": w / r if r else None})
    return rows
