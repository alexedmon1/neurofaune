"""Place a skull EEG array on a subject (and on the hub) through the skull-derived stereotaxic frame.

Electrode layouts are specified in skull millimetres from bregma (e.g. the
H32-style 30-electrode mouse array, ``mouse_array_coords.csv`` columns
``Xbregmm`` = ML, ``Ybregmm`` = AP). Placing those offsets 1:1 in template
millimetres is wrong whenever the template is not skull-scaled (Antwerp is
~3% longer rostro-caudally than skull mm, ~9% near bregma). Here each
electrode is placed by its skull coordinates instead:

1. In the subject's (or hub's) stereotaxic ML/AP maps -- skull coordinates
   carried through the registration -- find the brain voxels whose (ML, AP) is
   nearest the electrode's (ML, AP).
2. Of those, take the most dorsal one: the **cortical projection** of the
   electrode, i.e. the brain surface directly beneath it.

The output is the cortical point under each electrode and the parcel there.
These are the quantities an MRI-EEG comparison needs. The electrode itself
lies on the skull, typically 0.2-0.5 mm above that point.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


def read_layout(path: Path, ml_col: str = "Xbregmm", ap_col: str = "Ybregmm",
                label_col: str = "Label", prefix: str = "E") -> List[Dict]:
    """Electrode (label, ML, AP) in skull mm from bregma; rows whose label lacks ``prefix`` are skipped."""
    with open(path, newline="") as f:
        return [{"label": r[label_col], "ml": float(r[ml_col]), "ap": float(r[ap_col])}
                for r in csv.DictReader(f) if r[label_col].startswith(prefix)]


def cortical_projection(layout: List[Dict], ml_map: nib.Nifti1Image, ap_map: nib.Nifti1Image,
                        brain: np.ndarray, parcels: Optional[nib.Nifti1Image] = None,
                        names: Optional[Dict[int, str]] = None, tolerance_mm: float = 0.15,
                        world_scale: float = 1.0) -> List[Dict]:
    """Cortical point beneath each electrode, from stereotaxic ML/AP maps on any grid.

    ``world_scale`` divides the image world coordinates (10 for 10x-inflated
    subject images) so reported positions are in true mm. Electrodes with no
    brain voxel within ``tolerance_mm`` of their (ML, AP) -- off the edge of
    the brain -- are reported with ``found = False``.
    """
    ml = ml_map.get_fdata()
    ap = ap_map.get_fdata()
    brain = np.asarray(brain, bool) & ((ml != 0) | (ap != 0))
    idx = np.argwhere(brain)
    mlv, apv = ml[brain], ap[brain]
    world = (ml_map.affine @ np.column_stack([idx, np.ones(len(idx))]).T).T[:, :3] / world_scale
    # "dorsal" = +S in world for RAS-ish data; use the world z component
    lab = np.asanyarray(parcels.dataobj) if parcels is not None else None
    rows = []
    for e in layout:
        d = np.hypot(mlv - e["ml"], apv - e["ap"])
        near = d <= tolerance_mm
        row = {"label": e["label"], "ml": e["ml"], "ap": e["ap"], "found": bool(near.any())}
        if near.any():
            k = np.flatnonzero(near)[np.argmax(world[near, 2])]
            row.update({"x_mm": float(world[k, 0]), "y_mm": float(world[k, 1]), "z_mm": float(world[k, 2]),
                        "voxel": idx[k].tolist(), "match_error_mm": float(d[k])})
            if lab is not None:
                # The surface voxel can sit in the unlabelled surface-CSF layer; take
                # the most dorsal *labelled* voxel under the electrode instead.
                labelled = near & (lab[brain] > 0)
                if labelled.any():
                    kl = np.flatnonzero(labelled)[np.argmax(world[labelled, 2])]
                    v = int(lab[tuple(idx[kl])])
                    row["parcel_depth_mm"] = float(world[k, 2] - world[kl, 2])
                else:
                    v = 0
                row["parcel_id"] = v
                row["parcel"] = (names or {}).get(v, str(v))
        rows.append(row)
    return rows


def write_rows(rows: List[Dict], path: Path) -> Path:
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    return Path(path)


def cement_footprint(t2: np.ndarray, brain: np.ndarray, affine: np.ndarray, noise_threshold: float,
                     band_vox: tuple = (4, 9), world_scale: float = 1.0) -> Dict:
    """Signal-free cement cap above the dorsal brain surface, in the horizontal plane.

    ``t2`` and ``brain`` in RAS-canonical orientation (axis 2 dorsal). For each
    column, the mean T2 in a band ``band_vox`` voxels above the dorsal brain
    surface (default 0.6-1.35 mm at 150 um) is compared with
    ``noise_threshold``; the largest connected signal-free region is the cap.

    This is a **weak** implant-route bregma cue: electrodes on a thin-film array
    are not visible on RARE, so all that can be measured is the cement over it.
    The cap's medio-lateral midpoint estimates the midline. Its centroid only
    estimates the array centre if the cement was laid symmetrically, which
    is often not the case where a connector is cemented behind the array.

    Returns the cap mask (2-D), its centroid and per-row ML midpoints in world mm.
    """
    from scipy import ndimage

    brain = np.asarray(brain, bool)
    z = np.arange(brain.shape[2])
    has = brain.any(axis=2)
    top = ndimage.median_filter(np.where(has, np.where(brain, z, -1).max(axis=2), -1), 5)
    band = np.full(has.shape, np.nan)
    for i, j in zip(*np.nonzero(has)):
        zz = top[i, j] + np.arange(*band_vox)
        zz = zz[zz < t2.shape[2]]
        band[i, j] = t2[i, j, zz].mean()
    dark = has & (band < noise_threshold)
    lab, n = ndimage.label(ndimage.binary_opening(dark, iterations=2))
    if n == 0:
        return {"found": False}
    cap = ndimage.binary_fill_holes(lab == (np.argmax(ndimage.sum(dark, lab, range(1, n + 1))) + 1))
    ij = np.argwhere(cap)
    zc = float(top[cap].mean())
    world = (affine @ np.column_stack([ij, np.full(len(ij), zc), np.ones(len(ij))]).T).T[:, :3] / world_scale
    rows = []
    for j in np.unique(ij[:, 1]):
        xs = world[ij[:, 1] == j, 0]
        if xs.size >= 5:
            rows.append({"y_mm": float(world[ij[:, 1] == j, 1].mean()), "x_min": float(xs.min()),
                         "x_max": float(xs.max()), "x_mid": float((xs.min() + xs.max()) / 2)})
    voxel_area = float(np.prod(np.abs(np.diag(affine)[:2]))) / world_scale ** 2
    return {"found": True, "mask": cap, "centroid_mm": world.mean(axis=0).tolist(),
            "area_mm2": float(cap.sum() * voxel_area), "row_midpoints": rows,
            "x_extent_mm": [float(world[:, 0].min()), float(world[:, 0].max())],
            "y_extent_mm": [float(world[:, 1].min()), float(world[:, 1].max())]}
