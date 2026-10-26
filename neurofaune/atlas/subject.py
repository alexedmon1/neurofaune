"""Subject <-> mouse hub atlas: registration, bregma in subject space, atlas layers in subject space.

The subject's distortion-free T2w (brain-extracted) is registered to the
Antwerp hub brain in true millimetres. neurofaune keeps subject images
10x-inflated, so the registration runs on true-mm copies and every
subject-space output is written back on the subject's own inflated grid
(same voxels, header x10).

Outputs, BIDS-derivative style, in ``out_dir``:

- ``*_from-T2w_to-Antwerp_*`` transforms (ANTs affine + warp + inverse warp)
- ``*_desc-{allen26,allen32,allen64,ccfv3}_dseg.nii.gz`` atlas labels in T2w space
- ``*_desc-stereotaxic{ML,AP,DV}_mm.nii.gz`` skull-frame coordinate maps in T2w space
- ``*_desc-rabiesIC_components.nii.gz`` (4-D) + ``*_desc-rabiesCoverage_mask.nii.gz``
- ``*_bregma.json`` -- atlas-route bregma in subject world (true and inflated mm),
  in voxels, and the warp-vs-affine difference
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np

from neurofaune.atlas.label_transfer import (
    apply_transforms,
    forward_transforms,
    inverse_transforms,
    register_templates,
    transform_points_mm,
    warp_labels,
)


def rescale(img: nib.Nifti1Image, factor: float) -> nib.Nifti1Image:
    """Same voxels, world coordinates multiplied by ``factor`` (header-only)."""
    affine = img.affine.copy()
    affine[:3, :] *= factor
    out = nib.Nifti1Image(np.asanyarray(img.dataobj), affine)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    return out


def register_subject_to_hub(t2w: Path, brain_mask: Path, hub_dir: Path, out_dir: Path,
                            prefix: str, voxel_scale: float = 10.0, n_threads: int = 4,
                            work_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Register a subject T2w to the hub and bring the atlas layers and bregma into subject space."""
    t2w, brain_mask, hub_dir, out_dir = Path(t2w), Path(brain_mask), Path(hub_dir), Path(out_dir)
    work = Path(work_dir) if work_dir else out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # True-mm subject brain (and grid reference for outputs).
    t2 = nib.load(str(t2w))
    mask = nib.load(str(brain_mask))
    from nibabel.processing import resample_from_to
    m = resample_from_to(mask, t2, order=0).get_fdata() > 0.5
    brain_true = rescale(nib.Nifti1Image((t2.get_fdata() * m).astype(np.float32), t2.affine), 1 / voxel_scale)
    mask_true = rescale(nib.Nifti1Image(m.astype(np.uint8), t2.affine), 1 / voxel_scale)
    moving, moving_mask = work / "subject_brain_true.nii.gz", work / "subject_mask_true.nii.gz"
    nib.save(brain_true, str(moving))
    nib.save(mask_true, str(moving_mask))

    reg = register_templates(moving, hub_dir / "brain.nii.gz", work / f"{prefix}_from-T2w_to-Antwerp_",
                             fixed_mask=hub_dir / "brain_mask.nii.gz", n_threads=n_threads)
    warped = nib.load(str(reg["warped"])).get_fdata()
    hub_brain = nib.load(str(hub_dir / "brain.nii.gz")).get_fdata()
    hub_mask = nib.load(str(hub_dir / "brain_mask.nii.gz")).get_fdata() > 0
    r = float(np.corrcoef(warped[hub_mask], hub_brain[hub_mask])[0, 1])
    to_subject = inverse_transforms(reg)          # images: hub -> subject

    def back(src: Path, name: str, interp: str, labels: bool = False, series: bool = False) -> Path:
        tmp = work / f"_true_{name}"
        if labels:
            warp_labels(src, moving, to_subject, tmp, n_threads)
        else:
            apply_transforms(src, moving, to_subject, tmp, interp, n_threads, time_series=series)
        dest = out_dir / name
        nib.save(rescale(nib.load(str(tmp)), voxel_scale), str(dest))
        tmp.unlink(missing_ok=True)
        return dest

    outputs: Dict[str, str] = {}
    for key, rel in (("allen26", "allen26_labels.nii.gz"), ("allen32", "allen32_labels.nii.gz"),
                     ("allen64", "allen64_labels.nii.gz"), ("ccfv3", "ccfv3_annotation.nii.gz")):
        outputs[key] = str(back(hub_dir / rel, f"{prefix}_desc-{key}_dseg.nii.gz", "GenericLabel", labels=True))
    for axis in ("ml", "ap", "dv"):
        outputs[f"stereotaxic_{axis}"] = str(back(hub_dir / f"stereotaxic_{axis}_mm.nii.gz",
                                                  f"{prefix}_desc-stereotaxic{axis.upper()}_mm.nii.gz", "Linear"))
    func = hub_dir / "functional"
    if (func / "rabies_melodic_IC.nii.gz").exists():
        outputs["rabies_ic"] = str(back(func / "rabies_melodic_IC.nii.gz",
                                        f"{prefix}_desc-rabiesIC_components.nii.gz", "Linear", series=True))
        outputs["rabies_coverage"] = str(back(func / "rabies_coverage_mask.nii.gz",
                                              f"{prefix}_desc-rabiesCoverage_mask.nii.gz", "NearestNeighbor"))

    # Bregma: hub -> subject. Points fixed->moving use the image list for moving->fixed.
    stereo = json.loads((hub_dir / "stereotaxic.json").read_text())
    b_hub = np.array(stereo["bregma_antwerp_mm"])
    b_full = transform_points_mm(b_hub, forward_transforms(reg), work / "points")[0]
    b_aff = transform_points_mm(b_hub, [str(reg["affine"])], work / "points")[0]
    vox = (np.linalg.inv(t2.affine) @ np.r_[b_full * voxel_scale, 1])[:3]
    bregma = {
        "route": "atlas: Perens skull bregma in Antwerp, carried through subject<->Antwerp registration",
        "subject_world_true_mm": b_full.tolist(),
        "subject_world_scaled_mm": (b_full * voxel_scale).tolist(),
        "subject_voxel": vox.tolist(),
        "affine_only_true_mm": b_aff.tolist(),
        "warp_vs_affine_mm": float(np.linalg.norm(b_full - b_aff)),
        "registration_t2w_correlation_in_hub_mask": r,
    }
    (out_dir / f"{prefix}_bregma.json").write_text(json.dumps(bregma, indent=2))
    for key in ("affine", "warp", "inverse_warp"):
        dest = out_dir / reg[key].name
        dest.write_bytes(Path(reg[key]).read_bytes())
    return {"bregma": bregma, "outputs": outputs, "transforms": {k: str(v) for k, v in reg.items()}}
