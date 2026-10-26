"""Read BrainGlobe atlas packages as NIfTI with a correct anatomical affine.

BrainGlobe (https://brainglobe.info) distributes atlases as a folder holding
``reference.tiff``, ``annotation.tiff``, ``structures.csv`` and ``metadata.json``.
The TIFF stacks carry no geometry. ``metadata.json`` gives the voxel size (µm)
and a three-letter ``orientation`` naming the corner the array starts from:
``asr`` = origin at Anterior-Superior-Right, so axis 0 runs toward posterior,
axis 1 toward inferior, axis 2 toward left. That is turned into a RAS affine in
millimetres here. The volume is centred on the world origin because a
BrainGlobe package has no stereotaxic origin.

Checked visually on ``perens_stereotaxic_mri_mouse_25um``: olfactory bulb at
low axis 0, dorsal cortex at low axis 1.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import nibabel as nib
import numpy as np

#: The world direction each orientation letter's *opposite* end points to: an
#: axis that starts at 'a' (anterior) increases toward posterior = -A.
_START_TO_RAS = {
    "r": np.array([-1.0, 0.0, 0.0]),   # starts right  -> runs to left
    "l": np.array([1.0, 0.0, 0.0]),
    "a": np.array([0.0, -1.0, 0.0]),   # starts anterior -> runs posterior
    "p": np.array([0.0, 1.0, 0.0]),
    "s": np.array([0.0, 0.0, -1.0]),   # starts superior -> runs inferior
    "i": np.array([0.0, 0.0, 1.0]),
}


def orientation_affine(orientation: str, resolution_um, shape) -> np.ndarray:
    """RAS affine (mm) for a BrainGlobe array, centred on the world origin."""
    orientation = orientation.lower()
    if len(orientation) != 3 or len({c for c in orientation} & set("rlaspi")) != 3:
        raise ValueError(f"not a BrainGlobe orientation: {orientation!r}")
    res = np.asarray(resolution_um, dtype=float) / 1000.0
    affine = np.eye(4)
    for axis, letter in enumerate(orientation):
        affine[:3, axis] = _START_TO_RAS[letter] * res[axis]
    centre = (np.asarray(shape[:3], dtype=float) - 1) / 2.0
    affine[:3, 3] = -affine[:3, :3] @ centre
    return affine


class BrainGlobeAtlas:
    """A BrainGlobe atlas folder: reference, annotation and structure tree."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        self.name = self.metadata["name"]
        self.structures = _read_structures(self.folder / "structures.csv")
        self._paths = {s["id"]: s["path"] for s in self.structures.values()}

    def _image(self, stem: str, dtype=None) -> nib.Nifti1Image:
        import tifffile
        data = tifffile.imread(self.folder / f"{stem}.tiff")
        if dtype is not None:
            data = data.astype(dtype)
        affine = orientation_affine(self.metadata["orientation"], self.metadata["resolution"], data.shape)
        img = nib.Nifti1Image(data, affine)
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
        return img

    def reference(self) -> nib.Nifti1Image:
        return self._image("reference", np.float32)

    def annotation(self) -> nib.Nifti1Image:
        return self._image("annotation", np.uint32)

    def descendants(self, structure_id: int, include_self: bool = True) -> Set[int]:
        """IDs of a structure and everything below it in the hierarchy."""
        token = f"/{int(structure_id)}/"
        out = {sid for sid, path in self._paths.items() if token in path}
        if not include_self:
            out.discard(int(structure_id))
        return out

    def by_acronym(self, acronym: str) -> int:
        for s in self.structures.values():
            if s["acronym"] == acronym:
                return s["id"]
        raise KeyError(f"{self.name}: no structure with acronym {acronym!r}")


def _read_structures(path: Path) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid = int(row["id"])
            parent = row.get("parent_structure_id") or ""
            out[sid] = {
                "id": sid, "name": row["name"], "acronym": row["acronym"],
                "path": row["structure_id_path"],
                "parent": int(float(parent)) if parent not in ("", "nan") else None,
            }
    return out


def composite_mask(annotation: np.ndarray, ids: Set[int]) -> np.ndarray:
    """Voxels whose label is any of ``ids``."""
    return np.isin(annotation, np.fromiter(ids, dtype=annotation.dtype if annotation.dtype.kind in "iu" else np.int64))


def structure_ids_present(annotation: np.ndarray) -> List[int]:
    return [int(v) for v in np.unique(annotation) if v != 0]


def dice(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else None
