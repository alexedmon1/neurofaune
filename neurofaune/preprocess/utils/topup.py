"""Run FSL topup on a reverse-phase-encode pair, driven by BIDS sidecars.

The acquisition parameters come from the ``PhaseEncodingDirection`` and
``TotalReadoutTime`` fields neurofaune writes for Bruker EPI
(:mod:`neurofaune.utils.bruker_epi`), so the sign convention is the calibrated
one rather than an assumption about which scan is "forward".

topup's multi-resolution schedule (``b02b0.cnf`` subsamples by 2) needs even
image dimensions; rodent EPI slabs often have an odd slice count (13, 9). Odd
dimensions are padded by repeating the last plane, and every output is
cropped back to the input grid.

The dwi and func workflows use it through ``<modality>.topup.enabled``: the
reverse-PE pair is found in the session's BIDS ``fmap/`` (:func:`find_topup_pair`)
unless passed explicitly, and :func:`topup_correct_series` unwarps the series.
"""

import json
import os
import subprocess
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

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


def _crop(img_path: Path, shape: Tuple[int, int, int], affine: np.ndarray,
          header: "nib.Nifti1Header" = None) -> nib.Nifti1Image:
    """Crop back to ``shape``. ``header`` (the input's) keeps its zooms and units,
    so a 4-D series keeps its TR in pixdim[4] (FSL tools such as MELODIC read it)."""
    img = nib.load(str(img_path))
    data = np.asarray(np.asanyarray(img.dataobj)[: shape[0], : shape[1], : shape[2]], dtype=np.float32)
    if header is not None:
        header = header.copy()
        header.set_data_dtype(np.float32)
        out = nib.Nifti1Image(data, affine, header)
    else:
        out = nib.Nifti1Image(data, affine)
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
    _crop(out, shape, img.affine, header=img.header)
    return out


def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-2)


def _run_number(path: Path) -> Optional[int]:
    m = re.search(r"_run-(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def find_topup_pair(target: Path, fmap_dir: Optional[Path] = None) -> Optional[Tuple[Path, Path]]:
    """(same-PE, reverse-PE) EPIs that can unwarp ``target``, or None.

    Candidates are the ``*_epi.nii.gz`` in ``fmap_dir`` (default: the session's
    ``fmap/`` beside the target's modality folder) on the target's grid with
    its ``TotalReadoutTime``. One must share the target's
    ``PhaseEncodingDirection``, the other must be its reverse (j vs j-). BIDS
    ``IntendedFor`` is not required: neurofaune's Bruker conversion does not
    write it, and grid + readout + PE pin the pair. Several candidates are
    resolved by run number nearest the target's (the pair acquired next to it).
    """
    target = Path(target)
    fmap_dir = Path(fmap_dir) if fmap_dir is not None else target.parent.parent / "fmap"
    if not fmap_dir.is_dir():
        return None
    t_img, t_meta = nib.load(str(target)), sidecar(target)
    pe, trt = t_meta.get("PhaseEncodingDirection"), t_meta.get("TotalReadoutTime")
    if pe is None or trt is None:
        return None
    rev_pe = pe[:-1] if pe.endswith("-") else pe + "-"
    same, rev = [], []
    for epi in sorted(fmap_dir.glob("*_epi.nii.gz")):
        try:
            meta = sidecar(epi)
        except FileNotFoundError:
            continue
        if not np.isclose(float(meta.get("TotalReadoutTime", -1)), float(trt), rtol=1e-3):
            continue
        if not _same_grid(nib.load(str(epi)), t_img):
            continue
        if meta.get("PhaseEncodingDirection") == pe:
            same.append(epi)
        elif meta.get("PhaseEncodingDirection") == rev_pe:
            rev.append(epi)
    if not same or not rev:
        return None
    t_run = _run_number(target)
    near = lambda ref: (lambda p: abs((_run_number(p) or 0) - (ref or 0)))
    fwd = min(same, key=near(t_run))
    return fwd, min(rev, key=near(_run_number(fwd)))


def topup_correct_series(series: Sequence[Path], pair: Tuple[Path, Path], work_dir: Path,
                         n_threads: int = 1, config: str = "b02b0.cnf") -> Dict:
    """topup on ``pair`` and applytopup (Jacobian) to every 4-D ``series`` (e.g. echoes).

    Each series must share the pair's grid and the phase encoding of one of its
    images; it is corrected as acquired like that image. Returns the corrected
    paths (``series``), the topup outputs and a provenance record.
    """
    work_dir = Path(work_dir)
    tu = run_topup_pair(pair[0], pair[1], work_dir / "topup", n_threads=n_threads, config=config, force=True)
    rows = [acqparams_row(sidecar(Path(p))) for p in pair]
    corrected: List[Path] = []
    for s in series:
        s = Path(s)
        row = acqparams_row(sidecar(s)) if Path(str(s).replace(".nii.gz", ".json")).exists() else rows[0]
        if row not in rows:
            raise ValueError(f"{s.name} acquisition ({row}) matches neither topup EPI ({rows})")
        if not _same_grid(nib.load(str(s)), nib.load(str(pair[0]))):
            raise ValueError(f"{s.name} does not share the topup pair's grid")
        out = work_dir / s.name.replace(".nii.gz", "_topup.nii.gz")
        corrected.append(apply_topup(s, tu, rows.index(row) + 1, out))
    field = nib.load(str(tu["field"])).get_fdata()
    return {
        "series": corrected,
        "topup": tu,
        "record": {"Pair": [Path(p).name for p in pair],
                   "FieldHzP99Abs": round(float(np.percentile(np.abs(field), 99)), 1)},
    }
