"""Build the mouse hub atlas: Antwerp C57BL/6 MRI space with CCFv3 labels via a T2w bridge.

The hub space is the Antwerp template (UAntwerpen C57BL/6 in vivo T2w MRI) as
used by the EEG source-localization pipeline, so MRI and EEG share one
coordinate frame and one set of parcels (allen26 / allen32 / allen64).

Those parcels were originally made by warping Allen CCFv3 directly onto
Antwerp, Nissl/STPT to MRI (r = 0.57, ~20% AP compression). Here the CCFv3
annotation comes instead from a **T2w template that already carries CCFv3
labels** -- Perens et al. 2023, BrainGlobe ``perens_stereotaxic_mri_mouse`` --
so the transfer is a same-contrast T2w-to-T2w registration.

Outputs (``out_dir``), all in true millimetres:

- ``template_T2w.nii.gz``, ``brain.nii.gz``, ``brain_mask.nii.gz``,
  ``labels_antwerp46.nii.gz`` -- the Antwerp template on its native grid
  (0.203 x 0.080 x 0.200 mm), with the inflated headers corrected.
- ``ccfv3_annotation.nii.gz`` (native grid) and ``hub100um/`` (isotropic
  0.1 mm grid over the same field of view) -- CCFv3 structure IDs.
- ``allen32``, ``allen26``, ``allen64`` parcellations on both grids, rebuilt from
  their published structure-ID definitions, plus their ``roi_mapping`` JSON.
- ``scaled10/`` -- the same files with 10x-inflated headers, for neurofaune.
- ``validation.json`` -- registration similarity and Dice of CCFv3 composites
  against Antwerp's own MRI-drawn labels, for the old (direct) and new
  (Perens-bridge) annotations.
"""

import copy
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

from neurofaune.atlas.brainglobe import BrainGlobeAtlas, dice
from neurofaune.atlas.label_transfer import (
    forward_transforms,
    register_templates,
    warp_labels,
)

logger = logging.getLogger(__name__)

#: CCFv3 composites compared against Antwerp's MRI-drawn labels. Keys are
#: Antwerp label stems (``<stem>_L`` / ``<stem>_R``); values are CCFv3 acronyms
#: whose descendants make up the composite.
VALIDATION_COMPOSITES: Dict[str, List[str]] = {
    "Thalamus": ["TH"],
    "Hypothalamus": ["HY"],
    "Hippocampus": ["HIP"],
    "Cerebellum": ["CB"],
    "Striatum": ["CP", "ACB"],
    "Olfactory_Bulb": ["MOB", "AOB"],
    "Septal_Nucleus": ["LSX"],
    "Cortex_Motor": ["MO"],
    "Cortex_Somatosensory_1": ["SSp"],
    "Cortex_Cingulate": ["ACA"],
    "Cortex_Retrosplenial": ["RSP"],
    "Cortex_Auditory": ["AUD"],
    "Cortex_Visual": ["VIS"],
    "Cortex_Prelimbic": ["PL"],
    "Cortex_Orbital": ["ORB"],
    "Cortex_Piriform": ["PIR"],
}


def _true_grid(source_dir: Path) -> np.ndarray:
    """Affine of the Antwerp grid in true mm (taken from a correctly scaled header)."""
    return nib.load(str(source_dir / "Atlas_3DRoisLeftRight.Labels.nii")).affine


def load_antwerp_source(source_dir: Path) -> Dict[str, nib.Nifti1Image]:
    """The Antwerp atlas files from source-localization, all on one true-mm affine.

    Several of those files carry 10x-inflated voxel sizes in their headers; the
    voxel data share one grid, so every volume is re-wrapped on the affine of
    a correctly scaled file.
    """
    source_dir = Path(source_dir)
    affine = _true_grid(source_dir)

    def wrap(rel: str, dtype=np.float32) -> nib.Nifti1Image:
        data = np.asanyarray(nib.load(str(source_dir / rel)).dataobj).astype(dtype)
        return nib.Nifti1Image(data, affine)

    return {
        "template": wrap("Atlas_3DRois.nii"),
        "brain": wrap("Atlas_3DRois_brain.nii.gz"),
        "labels46": wrap("Atlas_3DRoisLeftRight.Labels.nii", np.uint32),
        "old_ccf": wrap("allen/allen_annotation_in_antwerp.nii.gz", np.uint32),
        "old_allen32": wrap("allen/allen_labels.nii.gz", np.uint32),
    }


def hub_grid(native: nib.Nifti1Image, voxel_mm: float = 0.1) -> nib.Nifti1Image:
    """Isotropic RAS grid covering the native Antwerp field of view."""
    shape = np.array(native.shape[:3])
    corners = np.array([[i, j, k, 1] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                        for k in (0, shape[2] - 1)], float)
    world = (native.affine @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    n = np.ceil((hi - lo) / voxel_mm).astype(int) + 1
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    affine[:3, 3] = lo
    return nib.Nifti1Image(np.zeros(tuple(n), np.uint8), affine)


def _save(img: nib.Nifti1Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.set_qform(img.affine, code=1)
    img.set_sform(img.affine, code=1)
    nib.save(img, str(path))
    return path


def _midline_x(annotation: nib.Nifti1Image) -> float:
    """World x of the labelled brain's centre of mass (the hemisphere split)."""
    ijk = np.argwhere(np.asanyarray(annotation.dataobj) > 0).mean(axis=0)
    return float((annotation.affine @ np.r_[ijk, 1])[0])


def _world_coords(img: nib.Nifti1Image, axis: int) -> np.ndarray:
    ijk = np.indices(img.shape[:3]).reshape(3, -1)
    return (img.affine[:3, :3] @ ijk + img.affine[:3, 3:4])[axis].reshape(img.shape[:3])


def parcellate(annotation: nib.Nifti1Image, mapping: Dict[str, Any],
               caudal_hippocampus_fraction: Optional[float] = None) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """Rebuild a source-localization parcellation (allen32/26/64) from CCFv3 IDs.

    Each ROI in ``mapping["rois"]`` lists ``allen_structure_ids`` and a
    ``hemisphere`` (L / R / B = bilateral). Hemispheres split at the labelled
    brain's centre-of-mass x (left = lower RAS x).

    Hippocampus_Ant / Hippocampus_Post share their structure IDs and are split
    along the rostro-caudal (RAS y) axis. In the original source-localization
    volumes the *caudal* part carries the "Ant" name -- the names are swapped
    relative to position. Here the split is placed so the caudal part keeps
    ``caudal_hippocampus_fraction`` of the hippocampal volume (0.35 in the
    original), and names follow position: Ant = rostral. The returned mapping
    records that in ``hippocampus_split``.
    """
    ann = np.asanyarray(annotation.dataobj)
    x = _world_coords(annotation, 0)
    y = _world_coords(annotation, 1)
    mid = _midline_x(annotation)
    left = x < mid
    out = np.zeros(ann.shape, np.uint16)
    rois = {k: v for k, v in mapping["rois"].items() if k != "0"}
    hippo = {k: v for k, v in rois.items() if v["name"].startswith(("Hippocampus_Ant", "Hippocampus_Post"))}
    split_y = None
    if hippo:
        ids = set().union(*(set(v["allen_structure_ids"]) for v in hippo.values()))
        hy = y[np.isin(ann, list(ids))]
        frac = 0.35 if caudal_hippocampus_fraction is None else caudal_hippocampus_fraction
        split_y = float(np.quantile(hy, frac)) if hy.size else None
    for key, roi in rois.items():
        sel = np.isin(ann, roi["allen_structure_ids"])
        hemi = roi.get("hemisphere", "B")
        if hemi == "L":
            sel &= left
        elif hemi == "R":
            sel &= ~left
        if roi["name"].startswith("Hippocampus_Ant") and split_y is not None:
            sel &= y >= split_y            # rostral
        elif roi["name"].startswith("Hippocampus_Post") and split_y is not None:
            sel &= y < split_y             # caudal
        out[sel & (out == 0)] = int(roi["id"])
    new_mapping = copy.deepcopy(mapping)
    new_mapping["hemisphere_split_x_mm"] = mid
    if split_y is not None:
        new_mapping["hippocampus_split"] = {
            "y_mm": split_y, "caudal_volume_fraction": frac,
            "naming": "Hippocampus_Ant = rostral (y >= split), Hippocampus_Post = caudal. "
                      "Source-localization's original volumes had these spatially swapped.",
        }
        new_mapping.pop("hpf_y_split_mm", None)
    return nib.Nifti1Image(out, annotation.affine), new_mapping


def composite_dice(ccf: np.ndarray, labels46: np.ndarray, names46: Dict[int, str],
                   atlas: BrainGlobeAtlas) -> Dict[str, Optional[float]]:
    """Dice of CCFv3 composites vs Antwerp's MRI-drawn labels (hemispheres pooled)."""
    out: Dict[str, Optional[float]] = {}
    for stem, acronyms in VALIDATION_COMPOSITES.items():
        ids = set().union(*(atlas.descendants(atlas.by_acronym(a)) for a in acronyms))
        target_ids = [i for i, n in names46.items() if n in (f"{stem}_L", f"{stem}_R")]
        if not target_ids:
            continue
        out[stem] = dice(np.isin(ccf, list(ids)), np.isin(labels46, target_ids))
    return out


def scale_header(img: nib.Nifti1Image, factor: float) -> nib.Nifti1Image:
    affine = img.affine.copy()
    affine[:3, :] *= factor
    return nib.Nifti1Image(np.asanyarray(img.dataobj), affine)


def build_antwerp_hub(source_dir: Path, perens_dir: Path, out_dir: Path, work_dir: Path,
                      mappings: Dict[str, Path], labels46_names: Path,
                      hub_voxel_mm: float = 0.1, registration_voxel_mm: float = 0.1,
                      n_threads: int = 4, voxel_scale: float = 10.0,
                      mask_weight: float = 0.0) -> Dict[str, Any]:
    """Build the Antwerp hub atlas package. See the module docstring for outputs."""
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    src = load_antwerp_source(Path(source_dir))
    perens = BrainGlobeAtlas(Path(perens_dir))

    # 1. Antwerp template files, true mm.
    brain = src["brain"]
    mask = nib.Nifti1Image((brain.get_fdata() > 0).astype(np.uint8), brain.affine)
    paths = {
        "template": _save(src["template"], out_dir / "template_T2w.nii.gz"),
        "brain": _save(brain, out_dir / "brain.nii.gz"),
        "brain_mask": _save(mask, out_dir / "brain_mask.nii.gz"),
        "labels46": _save(src["labels46"], out_dir / "labels_antwerp46.nii.gz"),
    }

    # 2. Perens reference (brain-only T2w), downsampled for registration.
    p_ref, p_ann = perens.reference(), perens.annotation()
    reg_grid = hub_grid(p_ref, registration_voxel_mm)
    p_small = resample_from_to(p_ref, reg_grid, order=1)
    p_small_path = _save(nib.Nifti1Image(p_small.get_fdata().astype(np.float32), p_small.affine),
                         work_dir / "perens_reference_reg.nii.gz")
    p_ann_path = _save(p_ann, work_dir / "perens_annotation_25um.nii.gz")
    p_mask = resample_from_to(nib.Nifti1Image((np.asanyarray(p_ann.dataobj) > 0).astype(np.float32),
                                              p_ann.affine), reg_grid, order=1)
    p_mask_path = _save(nib.Nifti1Image((p_mask.get_fdata() > 0.5).astype(np.float32), p_mask.affine),
                        work_dir / "perens_brain_mask_reg.nii.gz")

    # 3. Perens T2w -> Antwerp T2w brain.
    reg = register_templates(p_small_path, paths["brain"], work_dir / "perens_to_antwerp_",
                             fixed_mask=paths["brain_mask"], n_threads=n_threads,
                             mask_images=(paths["brain_mask"], p_mask_path), mask_weight=mask_weight)
    tf = forward_transforms(reg)
    tdir = out_dir / "transforms"
    tdir.mkdir(parents=True, exist_ok=True)
    import shutil
    for key in ("affine", "warp", "inverse_warp"):
        if key in reg:
            shutil.copy2(reg[key], tdir / f"perens_to_antwerp_{reg[key].name.split('_')[-1]}")
    warped = nib.load(str(reg["warped"])).get_fdata()
    m = mask.get_fdata() > 0
    r_new = float(np.corrcoef(warped[m], brain.get_fdata()[m])[0, 1])

    # 4. CCFv3 annotation onto the native and hub grids.
    hub = hub_grid(brain, hub_voxel_mm)
    hub_ref = _save(nib.Nifti1Image(np.zeros(hub.shape, np.float32), hub.affine), work_dir / "hub_ref.nii.gz")
    ccf_native = warp_labels(p_ann_path, paths["brain"], tf, out_dir / "ccfv3_annotation.nii.gz", n_threads)
    ccf_hub = warp_labels(p_ann_path, hub_ref, tf, out_dir / "hub100um" / "ccfv3_annotation.nii.gz", n_threads)
    _save(resample_from_to(src["template"], hub, order=1), out_dir / "hub100um" / "template_T2w.nii.gz")
    _save(nib.Nifti1Image((resample_from_to(mask, hub, order=1).get_fdata() > 0.5).astype(np.uint8),
                          hub.affine), out_dir / "hub100um" / "brain_mask.nii.gz")

    # 5. Parcellations, rebuilt from their structure-ID definitions.
    old32 = np.asanyarray(src["old_allen32"].dataobj)
    old_ant = np.isin(old32, [7, 23]).sum()
    old_post = np.isin(old32, [8, 24]).sum()
    caudal_frac = float(old_ant / (old_ant + old_post)) if old_ant + old_post else None
    parcels: Dict[str, Any] = {}
    for name, mpath in mappings.items():
        mapping = json.loads(Path(mpath).read_text())
        for grid_name, ann_path in (("native", ccf_native), ("hub100um", ccf_hub)):
            img, new_map = parcellate(nib.load(str(ann_path)), mapping, caudal_frac)
            sub = out_dir if grid_name == "native" else out_dir / "hub100um"
            _save(img, sub / f"{name}_labels.nii.gz")
            if grid_name == "native":
                new_map["source"] = (f"Rebuilt {date.today().isoformat()} by neurofaune.atlas.antwerp from "
                                     f"CCFv3 IDs warped via {perens.name} (T2w->T2w)")
                (out_dir / f"roi_mapping_{name}.json").write_text(json.dumps(new_map, indent=2))
                parcels[name] = {"n_labels": int(len(np.unique(np.asanyarray(img.dataobj))) - 1),
                                 "hippocampus_split": new_map.get("hippocampus_split")}

    # 6. Validation against Antwerp's own MRI labels.
    names_json = json.loads(Path(labels46_names).read_text())
    names46 = {int(k): v["name"] for k, v in names_json.get("rois", names_json).items()
               if isinstance(v, dict) and "name" in v}
    l46 = np.asanyarray(src["labels46"].dataobj)
    new_ccf = np.asanyarray(nib.load(str(ccf_native)).dataobj)
    old_ccf = np.asanyarray(src["old_ccf"].dataobj)
    d_new = composite_dice(new_ccf, l46, names46, perens)
    d_old = composite_dice(old_ccf, l46, names46, perens)
    brain_dice = {"new": dice(new_ccf > 0, m), "old": dice(old_ccf > 0, m)}
    validation = {
        "perens_to_antwerp_t2w_correlation_in_mask": r_new,
        "old_direct_warp_reported_correlation": 0.572,
        "brain_dice_vs_antwerp_mask": brain_dice,
        "composite_dice_vs_antwerp46": {k: {"new": d_new.get(k), "old": d_old.get(k)} for k in d_new},
        "median_composite_dice": {
            "new": float(np.median([v for v in d_new.values() if v is not None])),
            "old": float(np.median([v for v in d_old.values() if v is not None])),
        },
        "parcellations": parcels,
    }

    # 7. 10x-inflated copies for neurofaune.
    for f in sorted(out_dir.rglob("*.nii.gz")):
        if "scaled10" in f.parts:
            continue
        rel = f.relative_to(out_dir)
        _save(scale_header(nib.load(str(f)), voxel_scale), out_dir / "scaled10" / rel)

    meta = {
        "name": "Antwerp_CCFv3_hub", "built": date.today().isoformat(),
        "space": "Antwerp C57BL/6 in vivo T2w MRI template (UAntwerpen), RAS, true mm",
        "sources": {"antwerp": str(source_dir), "ccfv3_bridge": str(perens_dir),
                    "bridge_citation": perens.metadata.get("citation")},
        "transforms": {"dir": "transforms/", "moving": perens.name, "fixed": "Antwerp brain",
                       "forward_order": ["perens_to_antwerp_1Warp.nii.gz", "perens_to_antwerp_0GenericAffine.mat"]},
        "hub_voxel_mm": hub_voxel_mm, "scaled10_factor": voxel_scale,
        "registration": {"intensity_metric": "MI (rigid/affine), CC r=4 (SyN)",
                         "brain_mask_meansquares_weight": mask_weight},
        "validation": validation,
    }
    (out_dir / "atlas_metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    return meta
