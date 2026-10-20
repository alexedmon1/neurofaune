"""Run FSL topup on a reverse-phase-encode pair, driven by BIDS sidecars.

The acquisition parameters come from the ``PhaseEncodingDirection`` and
``TotalReadoutTime`` fields neurofaune writes for Bruker EPI
(:mod:`neurofaune.utils.bruker_epi`), so the sign convention is the calibrated
one rather than an assumption about which scan is "forward".

topup's multi-resolution schedule (``b02b0.cnf`` subsamples by 2) needs even
image dimensions; rodent EPI slabs often have an odd slice count (13, 9). Odd
dimensions are padded by repeating the last plane, and every output is
cropped back to the input grid.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np

_PE_VECTORS = {
    "i": (1, 0, 0), "i-": (-1, 0, 0),
    "j": (0, 1, 0), "j-": (0, -1, 0),
    "k": (0, 0, 1), "k-": (0, 0, -1),
}


def sidecar(nifti: Path) -> Dict:
    return json.loads(Path(str(nifti).replace(".nii.gz", ".json")).read_text())


def acqparams_row(meta: Dict) -> str:
    """One topup ``--datain`` line from a BIDS sidecar."""
    try:
        vec = _PE_VECTORS[meta["PhaseEncodingDirection"]]
        trt = float(meta["TotalReadoutTime"])
    except KeyError as e:
        raise KeyError(f"sidecar lacks {e}; reconvert with a neurofaune that writes EPI readout fields")
    return f"{vec[0]} {vec[1]} {vec[2]} {trt:.6f}"


def _pad_even(data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    shape = data.shape[:3]
    for ax in range(3):
        if data.shape[ax] % 2:
            last = np.take(data, [-1], axis=ax)
            data = np.concatenate([data, last], axis=ax)
    return data, shape


def _crop(img_path: Path, shape: Tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    img = nib.load(str(img_path))
    data = np.asanyarray(img.dataobj)[: shape[0], : shape[1], : shape[2]]
    out = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(out, str(img_path))
    return out


def run_topup_pair(fwd: Path, rev: Path, out_dir: Path, n_threads: int = 2,
                   config: str = "b02b0.cnf", force: bool = False) -> Dict[str, Path]:
    """topup on one fwd/rev pair. Returns paths of the field (Hz), unwarped pair and mean.

    Each input may be 3-D or 4-D; only the first volume of each is used.
    """
    fwd, rev, out_dir = Path(fwd), Path(rev), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "field": out_dir / "topup_field.nii.gz",
        "unwarped": out_dir / "topup_unwarped.nii.gz",
        "unwarped_mean": out_dir / "topup_unwarped_mean.nii.gz",
        "fieldcoef": out_dir / "topup_fieldcoef.nii.gz",
        "movpar": out_dir / "topup_movpar.txt",
        "acqparams": out_dir / "acqparams.txt",
    }
    if not force and all(p.exists() for p in (outputs["field"], outputs["unwarped_mean"])):
        return outputs

    f_img, r_img = nib.load(str(fwd)), nib.load(str(rev))
    first = lambda im: np.asanyarray(im.dataobj)[..., 0] if im.ndim == 4 else np.asanyarray(im.dataobj)
    pair, shape = _pad_even(np.stack([first(f_img), first(r_img)], axis=-1).astype(np.float32))
    nib.save(nib.Nifti1Image(pair, f_img.affine), str(out_dir / "pair.nii.gz"))
    outputs["acqparams"].write_text(
        acqparams_row(sidecar(fwd)) + "\n" + acqparams_row(sidecar(rev)) + "\n")

    env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
    cmd = ["topup", "--imain=pair", "--datain=acqparams.txt", f"--config={config}",
           "--out=topup", "--fout=topup_field", "--iout=topup_unwarped", f"--nthr={n_threads}"]
    proc = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"topup failed ({proc.returncode}): {proc.stderr[-2000:]}")

    _crop(outputs["field"], shape, f_img.affine)
    unwarped = _crop(outputs["unwarped"], shape, f_img.affine)
    nib.save(nib.Nifti1Image(unwarped.get_fdata().mean(axis=-1).astype(np.float32), f_img.affine),
             str(outputs["unwarped_mean"]))
    (out_dir / "pair.nii.gz").unlink(missing_ok=True)
    return outputs


def apply_topup(image: Path, topup_outputs: Dict[str, Path], inindex: int, out: Path) -> Path:
    """applytopup (Jacobian modulation) of one image acquired like row ``inindex`` (1-based).

    The image must share the topup pair's grid. It is padded the same way the
    pair was and cropped back afterwards.
    """
    image, out = Path(image), Path(out)
    img = nib.load(str(image))
    data, shape = _pad_even(np.asarray(img.dataobj, dtype=np.float32))
    work = Path(topup_outputs["field"]).parent
    padded = work / f"_apply_in_{out.name}"
    nib.save(nib.Nifti1Image(data, img.affine), str(padded))
    env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
    stem = str(out).replace(".nii.gz", "")
    cmd = ["applytopup", f"--imain={padded}", f"--inindex={inindex}",
           f"--datain={topup_outputs['acqparams']}", f"--topup={work / 'topup'}",
           "--method=jac", f"--out={stem}"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    padded.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"applytopup failed ({proc.returncode}): {proc.stderr[-2000:]}")
    _crop(out, shape, img.affine)
    return out
