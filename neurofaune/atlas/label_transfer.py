"""Template-to-template label transfer with ANTs (CLI), and its validation.

Moves a labelled template onto another template of the same contrast
(MRI-to-MRI), then applies the transform to the labels with ``GenericLabel``
interpolation. Same-contrast registration is the point: warping CCFv3 labels
from the CCFv3 average template (two-photon autofluorescence) onto an MRI
template is a cross-modality problem (the source-localization Antwerp warp
reached r = 0.57). A T2w template that already carries CCFv3 labels (e.g. Perens
et al. 2023) reduces that to a T2w-to-T2w registration.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np


def _run(cmd: Sequence[str], n_threads: int) -> None:
    env = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(n_threads))
    proc = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr[-3000:]}")


def register_templates(moving: Path, fixed: Path, out_prefix: Path,
                       fixed_mask: Optional[Path] = None, moving_mask: Optional[Path] = None,
                       n_threads: int = 4, syn: bool = True,
                       syn_metric: str = "CC", syn_radius: int = 4,
                       mask_images: Optional[Sequence[Path]] = None, mask_weight: float = 0.0,
                       force: bool = False) -> Dict[str, Path]:
    """Rigid -> affine -> (SyN) registration of ``moving`` onto ``fixed``.

    ``mask_images = (fixed_brain_mask, moving_brain_mask)`` with ``mask_weight > 0``
    adds a MeanSquares term on the two binary brain masks to the affine and SyN
    stages, so the brain outlines are driven to agree as well as the intensities.

    Smoothing is given in voxels and shrink factors are unit-free, so this
    works the same on true-mm and 10x-inflated headers. Returns the forward
    transforms (apply to moving-space data in reverse order: warp, affine) and
    the warped image.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    p = str(out_prefix)
    out = {"affine": Path(f"{p}0GenericAffine.mat"), "warped": Path(f"{p}Warped.nii.gz"),
           "inverse_warped": Path(f"{p}InverseWarped.nii.gz")}
    if syn:
        out["warp"] = Path(f"{p}1Warp.nii.gz")
        out["inverse_warp"] = Path(f"{p}1InverseWarp.nii.gz")
    if not force and all(v.exists() for v in out.values()):
        return out
    mask_term: List[str] = []
    if mask_images and mask_weight > 0:
        fm, mm = mask_images
        mask_term = ["--metric", f"MeanSquares[{fm},{mm},{mask_weight}]"]
    cmd: List[str] = [
        "antsRegistration", "--dimensionality", "3", "--float", "1",
        "--output", f"[{p},{p}Warped.nii.gz,{p}InverseWarped.nii.gz]",
        "--interpolation", "Linear", "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1",
        "--initial-moving-transform", f"[{fixed},{moving},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x0,1e-6,10]", "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
        *mask_term,
        "--convergence", "[1000x500x250x100,1e-6,10]", "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    if syn:
        cmd += [
            "--transform", "SyN[0.1,3,0]",
            "--metric", f"{syn_metric}[{fixed},{moving},1,{syn_radius}]",
            *mask_term,
            "--convergence", "[100x70x50x20,1e-6,10]", "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
        ]
    if fixed_mask or moving_mask:
        cmd += ["--masks", f"[{fixed_mask or 'NULL'},{moving_mask or 'NULL'}]"]
    _run(cmd, n_threads)
    return out


def forward_transforms(reg: Dict[str, Path]) -> List[str]:
    """``-t`` list mapping moving-space data into fixed space."""
    return ([str(reg["warp"])] if "warp" in reg else []) + [str(reg["affine"])]


def inverse_transforms(reg: Dict[str, Path]) -> List[str]:
    """``-t`` list mapping fixed-space data into moving space."""
    return [f"[{reg['affine']},1]"] + ([str(reg["inverse_warp"])] if "inverse_warp" in reg else [])


def apply_transforms(moving: Path, reference: Path, transforms: Sequence[str], out: Path,
                     interpolation: str = "Linear", n_threads: int = 4) -> Path:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["antsApplyTransforms", "-d", "3", "-i", moving, "-r", reference, "-o", out,
           "-n", interpolation, "--float", "1"]
    for t in transforms:
        cmd += ["-t", t]
    _run(cmd, n_threads)
    return Path(out)


def warp_labels(labels: Path, reference: Path, transforms: Sequence[str], out: Path,
                n_threads: int = 4) -> Path:
    """Warp an integer label volume with ``GenericLabel`` (no label mixing at boundaries).

    Labels are compacted to consecutive integers before warping and mapped back
    after. ANTs resamples in floating point, and CCFv3 structure IDs reach
    614,454,277 -- far beyond the 2**24 integers float32 holds exactly -- so
    warping raw IDs silently turns large ones into IDs that do not exist.
    """
    out = Path(out)
    img = nib.load(str(labels))
    data = np.asanyarray(img.dataobj)
    ids, compact = np.unique(data, return_inverse=True)
    if ids.size > 2 ** 24:
        raise ValueError(f"{labels}: {ids.size} labels cannot be warped exactly in float32")
    tmp = out.parent / f"_compact_{out.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(compact.reshape(data.shape).astype(np.float32), img.affine), str(tmp))
    try:
        apply_transforms(tmp, reference, transforms, out, "GenericLabel[Linear]", n_threads)
    finally:
        tmp.unlink(missing_ok=True)
    warped = nib.load(str(out))
    index = np.clip(np.rint(np.asanyarray(warped.dataobj)).astype(np.int64), 0, ids.size - 1)
    nib.save(nib.Nifti1Image(ids[index].astype(np.uint32), warped.affine), str(out))
    return out


def transform_points_mm(points: np.ndarray, transforms: Sequence[str], work: Path,
                        n_threads: int = 1) -> np.ndarray:
    """Map world points (RAS mm, as in the NIfTI affines) through ANTs transforms.

    ANTs works in LPS and transforms points with the *inverse* of the image
    transform list, so: to move a point from moving space to fixed space, pass
    the transforms that map fixed-space images into moving space (see
    :func:`inverse_transforms`).
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pts = np.atleast_2d(np.asarray(points, float)).copy()
    lps = pts * np.array([-1, -1, 1])
    src, dst = work / "points_in.csv", work / "points_out.csv"
    with open(src, "w") as f:
        f.write("x,y,z,t\n")
        for x, y, z in lps:
            f.write(f"{x},{y},{z},0\n")
    cmd = ["antsApplyTransformsToPoints", "-d", "3", "-i", src, "-o", dst]
    for t in transforms:
        cmd += ["-t", t]
    _run(cmd, n_threads)
    out = np.loadtxt(dst, delimiter=",", skiprows=1)[:, :3].reshape(-1, 3)
    return out * np.array([-1, -1, 1])


def masked_correlation(a: Path, b: Path, mask: Optional[np.ndarray] = None) -> float:
    x = nib.load(str(a)).get_fdata()
    y = nib.load(str(b)).get_fdata()
    sel = (x != 0) & (y != 0) if mask is None else mask
    return float(np.corrcoef(x[sel], y[sel])[0, 1])
