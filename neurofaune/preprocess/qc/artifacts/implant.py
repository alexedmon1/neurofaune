"""Per-session implant artifact QC against a distortion-free T2w reference.

What an implant on the skull does to MRI, measured as a function of depth
below the dorsal brain surface (where the implant sits):

1. **B0 offset** -- the scanner's B0 map (Hz) inside the brain.
2. **EPI distortion** -- topup's field for every reverse-PE pair, converted to
   displacement along phase encoding (mm), and how well each EPI matches the
   T2w before and after topup.
3. **Dropout** -- gradient-echo / spin-echo signal ratio on the shared slab:
   T2* dephasing the spin echo refocuses.
4. **tSNR** -- of each BOLD run after motion correction.
5. **Chemical shift** -- predicted fat/petrolatum displacement in each EPI, and
   whether bright extracranial dorsal signal on the T2w reappears displaced in
   the EPI with opposite sign in the two phase-encode polarities.

Everything is resampled onto the RAS-canonical T2w grid through the scanner
affines (``bids.affine: scanner``); no registration is involved, so nothing
here can hide a distortion by absorbing it into a transform.

Outputs (``out_dir``): ``maps/*.nii.gz`` on the T2w grid,
``tables/depth_profiles.tsv`` (long format), ``summary.json``, and figures.
"""

import csv
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage

from neurofaune.preprocess.qc.artifacts import maps
from neurofaune.preprocess.utils.topup import apply_topup, run_topup_pair, sidecar
from neurofaune.utils.bruker_epi import chemical_shift_hz

logger = logging.getLogger(__name__)

#: Depth below the dorsal surface treated as "deep brain" -- the normalisation
#: reference for ratios, assumed beyond the implant's reach.
DEEP_REFERENCE_MM = 2.0


def _acq(path: Path) -> str:
    m = re.search(r"_acq-([A-Za-z0-9]+)", path.name)
    return m.group(1) if m else ""


def _resample(img: nib.Nifti1Image, ref: nib.Nifti1Image, order: int = 1,
              outside: float = np.nan, trim: int = 1) -> np.ndarray:
    """Resample ``img`` onto ``ref``; voxels outside ``img``'s field of view get ``outside``.

    The coverage is trimmed by ``trim`` source voxels at every face of the
    source grid: interpolation there mixes in the zero padding, which for a
    thin EPI slab shows up as a spurious band at its first and last slice.
    """
    data = resample_from_to(img, ref, order=order, cval=0.0).get_fdata()
    ones = np.ones(img.shape[:3], np.float32)
    if trim and order > 0:
        ones = ndimage.binary_erosion(ones, iterations=trim, border_value=0).astype(np.float32)
    cover = resample_from_to(nib.Nifti1Image(ones, img.affine), ref, order=0, cval=0.0).get_fdata() > 0.5
    return np.where(cover, data, outside)


def _first_volume(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return img.slicer[..., 0] if img.ndim == 4 else img


def _pe_voxel_mm(meta: Dict, img: nib.Nifti1Image, voxel_scale: float) -> float:
    axis = "ijk".index(meta.get("PhaseEncodingAxis", meta["PhaseEncodingDirection"][0]))
    return float(img.header.get_zooms()[axis]) / voxel_scale


def _n_pe(meta: Dict, img: nib.Nifti1Image) -> int:
    return int(img.shape["ijk".index(meta.get("PhaseEncodingAxis", meta["PhaseEncodingDirection"][0]))])


def find_epi_pairs(session_dir: Path) -> Dict[str, Dict[str, Path]]:
    """``{acq: {"fwd": path, "rev": path}}`` from ``fmap/*_dir-{fwd,rev}_*_epi.nii.gz``."""
    pairs: Dict[str, Dict[str, Path]] = {}
    for p in sorted((Path(session_dir) / "fmap").glob("*_epi.nii.gz")):
        m = re.search(r"_dir-(fwd|rev)_", p.name)
        if m:
            pairs.setdefault(_acq(p), {})[m.group(1)] = p
    return {k: v for k, v in pairs.items() if {"fwd", "rev"} <= set(v)}


def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-3)


def motion_correct(bold: Path, work: Path) -> Dict[str, Any]:
    """mcflirt to the middle volume; returns corrected path and framewise displacement."""
    work.mkdir(parents=True, exist_ok=True)
    out = work / (bold.name.replace(".nii.gz", "_mc"))
    if not Path(str(out) + ".nii.gz").exists():
        env = dict(os.environ, FSLOUTPUTTYPE="NIFTI_GZ")
        subprocess.run(["mcflirt", "-in", str(bold), "-out", str(out), "-plots"],
                       check=True, capture_output=True, env=env)
    par = np.loadtxt(Path(str(out) + ".par"))
    # Power FD with a 5 mm "head radius" in scaled units ~ 0.5 mm true for mouse
    rad = 5.0
    d = np.abs(np.diff(par, axis=0))
    fd = d[:, :3].sum(axis=1) * rad + d[:, 3:].sum(axis=1)
    return {"path": Path(str(out) + ".nii.gz"), "fd_scaled_mm": fd}


_ANATOMICAL = {"R": "right", "L": "left", "A": "rostral", "P": "caudal", "S": "dorsal", "I": "ventral"}
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


def edge_labels(img: nib.Nifti1Image, axis: int) -> Dict[str, str]:
    """Anatomical names of the low-index (start) and high-index (end) ends of a voxel axis."""
    code = nib.aff2axcodes(img.affine)[axis]          # direction of increasing index
    return {"start": _ANATOMICAL[_OPPOSITE[code]], "end": _ANATOMICAL[code]}


def edge_shift_table(epi: np.ndarray, grid: nib.Nifti1Image, ref: nib.Nifti1Image,
                     brain: np.ndarray, axis: int, voxel_scale: float) -> List[Dict[str, Any]]:
    """Per-slice brain-edge shifts of ``epi`` (on ``grid``) along ``axis``, edges named anatomically."""
    mask = resample_from_to(nib.Nifti1Image(brain.astype(np.float32), ref.affine), grid,
                            order=1).get_fdata() > 0.5
    vox = float(grid.header.get_zooms()[axis]) / voxel_scale
    labels = edge_labels(grid, axis)
    rows = []
    for r in maps.edge_shift_along_axis(epi, mask, axis, vox):
        out = {"slice": r["slice"], "n_columns": r["n_columns"]}
        for end in ("start", "end"):
            for stat in ("median_mm", "p90_mm", "frac_gt_2vox", "frac_unresolved"):
                out[f"{labels[end]}_{stat}"] = r[f"{end}_{stat}"]
        rows.append(out)
    return rows


def summarise_edges(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Across slices: median of slice medians, worst slice p90, worst slices."""
    out: Dict[str, Any] = {}
    edges = sorted({k.rsplit("_median_mm", 1)[0] for k in (rows[0] if rows else {}) if k.endswith("_median_mm")})
    for e in edges:
        med = np.array([r[f"{e}_median_mm"] for r in rows], float)
        p90 = np.array([r[f"{e}_p90_mm"] for r in rows], float)
        frac = np.array([r[f"{e}_frac_gt_2vox"] for r in rows], float)
        worst = np.argsort(-np.nan_to_num(p90, nan=-np.inf))[:3]
        unres = np.array([r[f"{e}_frac_unresolved"] for r in rows], float)
        out[e] = {"mean_frac_unresolved": float(np.nanmean(unres)),
                  "median_of_slice_medians_mm": float(np.nanmedian(med)),
                  "max_slice_p90_mm": float(np.nanmax(p90)),
                  "mean_frac_gt_2vox": float(np.nanmean(frac)),
                  "worst_slices": [int(rows[i]["slice"]) for i in worst]}
    return out


def _dorsal_bright_layer(t2: np.ndarray, brain: np.ndarray, vz: float,
                         max_height_mm: float = 3.0, percentile: float = 90.0) -> np.ndarray:
    """Bright extracranial signal above the dorsal brain surface (Neosporin/fat candidate)."""
    shell = maps.shell_above(brain, vz, 0.0, max_height_mm)
    thr = np.percentile(t2[brain], percentile)
    layer = shell & (t2 > thr)
    return ndimage.binary_opening(layer, iterations=1) if layer.sum() > 50 else layer


def run_implant_artifact_qc(
    session_dir: Path,
    out_dir: Path,
    t2w: Path,
    brain_mask: Path,
    voxel_scale: float = 10.0,
    spin_echo_prefix: str = "se",
    fieldmap_acq: Optional[str] = None,
    n_threads: int = 2,
    work_dir: Optional[Path] = None,
    bin_mm: float = 0.25,
) -> Dict[str, Any]:
    """Run the implant artifact QC for one BIDS session.

    Parameters
    ----------
    session_dir : Path
        BIDS ``ses-*`` directory written with ``bids.affine: scanner``.
    out_dir : Path
        Output folder (maps/, tables/, figures/, summary.json).
    t2w, brain_mask : Path
        Distortion-free anatomical reference and its brain mask (any
        orientation; both are taken to RAS-canonical).
    voxel_scale : float
        Header inflation to undo for mm reporting.
    spin_echo_prefix : str
        EPI pairs whose acq label starts with this are spin echo (dropout
        reference); the others are gradient echo.
    fieldmap_acq : str, optional
        acq label of the B0 map to compare topup against; default: the last
        ``*_fieldmap.nii.gz`` acquired (highest run number), i.e. post-shim.
    """
    session_dir, out_dir = Path(session_dir), Path(out_dir)
    work = Path(work_dir) if work_dir else out_dir / "work"
    for sub in ("maps", "tables", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    ref = maps.canonical(nib.load(str(t2w)))
    t2 = ref.get_fdata()
    brain = _resample(nib.load(str(brain_mask)), ref, order=0, outside=0) > 0.5
    vz = float(ref.header.get_zooms()[2]) / voxel_scale
    depth = maps.dorsal_depth_mm(brain, vz)
    height = maps.ventral_height_mm(brain, vz)
    # "Deep" = beyond the reach of both surfaces, so the normalisation
    # reference is neither implant- nor skull-base-affected.
    deep = brain & (depth > DEEP_REFERENCE_MM) & (height > DEEP_REFERENCE_MM)
    dorsal_1mm = brain & (depth <= 1.0)
    ventral_1mm = brain & (height <= 1.0)
    nib.save(nib.Nifti1Image(np.nan_to_num(depth, nan=-1).astype(np.float32), ref.affine),
             str(out_dir / "maps" / "dorsal_depth_mm.nii.gz"))

    profiles: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"reference": str(t2w), "brain_mm3": float(brain.sum() * vz ** 3),
                               "voxel_scale": voxel_scale, "deep_reference_mm": DEEP_REFERENCE_MM}

    def add_profile(metric: str, source: str, values: np.ndarray, mask: np.ndarray = brain):
        for surface, dist in (("dorsal", depth), ("ventral", height)):
            for row in maps.depth_profile(values, dist, mask, bin_mm=bin_mm):
                profiles.append({"metric": metric, "source": source, "surface": surface, **row})

    def surfaces(values: np.ndarray, mask: np.ndarray = brain) -> Dict[str, Any]:
        """Dorsal (implant) vs ventral (control) superficial 1 mm, and deep brain."""
        return {"dorsal_0_1mm": maps.summarise(values, mask & dorsal_1mm),
                "ventral_0_1mm": maps.summarise(values, mask & ventral_1mm),
                "deep": maps.summarise(values, mask & deep)}

    def save_map(name: str, values: np.ndarray):
        nib.save(nib.Nifti1Image(np.nan_to_num(values, nan=0).astype(np.float32), ref.affine),
                 str(out_dir / "maps" / f"{name}.nii.gz"))

    # 1. B0 maps -------------------------------------------------------------
    fieldmaps = sorted((session_dir / "fmap").glob("*_fieldmap.nii.gz"),
                       key=lambda p: int(re.search(r"_run-(\d+)", p.name).group(1)))
    summary["b0"] = {}
    for fm in fieldmaps:
        acq = _acq(fm)
        f = _resample(nib.load(str(fm)), ref)
        f = np.where(np.abs(f) > 0, f, np.nan)        # Bruker zeroes unmeasured voxels
        save_map(f"b0_{acq}_hz", f)
        add_profile("b0_hz", acq, f)
        add_profile("b0_abs_hz", acq, np.abs(f))
        summary["b0"][acq] = {
            "brain": maps.summarise(f, brain),
            "brain_abs": maps.summarise(np.abs(f), brain),
            "abs_by_surface": surfaces(np.abs(f)),
            "shell_0_1.5mm_above_brain_abs": maps.summarise(np.abs(f), maps.shell_above(brain, vz, 0, 1.5)),
        }
    if fieldmap_acq is None and fieldmaps:
        fieldmap_acq = _acq(fieldmaps[-1])
    fm_ref = next((p for p in fieldmaps if _acq(p) == fieldmap_acq), None)

    # 2. EPI pairs: topup field, displacement, fidelity ----------------------
    pairs = find_epi_pairs(session_dir)
    unwarped: Dict[str, np.ndarray] = {}
    topups: Dict[str, Any] = {}
    edge_rows: List[Dict[str, Any]] = []
    summary["epi"] = {}
    b0_t = None
    for acq, pair in pairs.items():
        meta = sidecar(pair["fwd"])
        b0_t = b0_t or float(meta.get("MagneticFieldStrength", 7.0))
        epi = nib.load(str(pair["fwd"]))
        tu = run_topup_pair(pair["fwd"], pair["rev"], work / f"topup_{acq}", n_threads=n_threads)
        field_img = nib.load(str(tu["field"]))
        n_pe = _n_pe(meta, epi)
        pe_mm = _pe_voxel_mm(meta, epi, voxel_scale)
        px_per_hz = float(meta["EffectiveEchoSpacing"]) * n_pe
        disp_mm = nib.Nifti1Image(field_img.get_fdata() * px_per_hz * pe_mm, field_img.affine)

        field_ref = _resample(field_img, ref)
        disp_ref = _resample(disp_mm, ref)
        save_map(f"topup_{acq}_field_hz", field_ref)
        save_map(f"topup_{acq}_displacement_mm", disp_ref)
        add_profile("topup_field_hz", acq, field_ref)
        add_profile("displacement_abs_mm", acq, np.abs(disp_ref))

        slab = np.isfinite(field_ref) & brain
        entry: Dict[str, Any] = {
            "phase_encoding": meta["PhaseEncodingDirection"],
            "total_readout_time_s": float(meta["TotalReadoutTime"]),
            "pe_bandwidth_hz_per_px": 1.0 / px_per_hz,
            "pe_voxel_mm": pe_mm,
            "displacement_abs_mm": maps.summarise(np.abs(disp_ref), slab),
            "displacement_abs_mm_by_surface": surfaces(np.abs(disp_ref), slab),
            "fat_shift_mm": chemical_shift_hz(b0_t) * px_per_hz * pe_mm,
        }
        if fm_ref is not None:
            fm_epi = resample_from_to(nib.load(str(fm_ref)), field_img, order=1).get_fdata()
            br_epi = ndimage.binary_erosion(
                resample_from_to(nib.Nifti1Image(brain.astype(np.float32), ref.affine), field_img,
                                 order=0).get_fdata() > 0.5, iterations=1) & (np.abs(fm_epi) > 0)
            a, b = field_img.get_fdata()[br_epi], fm_epi[br_epi]
            if a.size > 10:
                entry["topup_vs_b0map"] = {"b0map": fieldmap_acq, "n": int(a.size),
                                           "r": float(np.corrcoef(a, b)[0, 1]),
                                           "slope": float(np.polyfit(b, a, 1)[0])}
        fidelity = {}
        for label, img in (("fwd", _first_volume(epi)),
                           ("rev", _first_volume(nib.load(str(pair["rev"])))),
                           ("topup", nib.load(str(tu["unwarped_mean"])))):
            fidelity[label] = maps.masked_correlation(_resample(img, ref), t2, slab)
        entry["t2w_correlation"] = fidelity
        axis = "ijk".index(meta["PhaseEncodingDirection"][0])
        entry["edge_shift"] = {}
        for label, img in (("fwd", _first_volume(epi)),
                           ("rev", _first_volume(nib.load(str(pair["rev"])))),
                           ("topup", nib.load(str(tu["unwarped_mean"])))):
            rows = edge_shift_table(img.get_fdata(), epi, ref, brain, axis, voxel_scale)
            edge_rows += [{"image": f"{acq}_{label}", **r} for r in rows]
            entry["edge_shift"][label] = summarise_edges(rows)
        unwarped[acq] = _resample(nib.load(str(tu["unwarped_mean"])), ref)
        topups[acq] = (epi, tu)
        summary["epi"][acq] = entry

    # 3. Dropout: GE / SE ratio on the shared slab --------------------------
    summary["dropout"] = {}
    se = [a for a in unwarped if a.startswith(spin_echo_prefix)]
    ge = [a for a in unwarped if not a.startswith(spin_echo_prefix)]
    for g in ge:
        for s in se:
            both = brain & np.isfinite(unwarped[g]) & np.isfinite(unwarped[s])
            refv = both & deep
            if refv.sum() < 100:
                continue
            ratio = np.where(both, maps.normalised_ratio(unwarped[g], unwarped[s], refv), np.nan)
            name = f"{g}_over_{s}"
            save_map(f"dropout_ratio_{name}", ratio)
            add_profile("ge_se_ratio", name, ratio)
            frac = lambda m: float(np.mean(ratio[m] < 0.5)) if m.any() else None  # noqa: E731
            summary["dropout"][name] = {
                "ratio": maps.summarise(ratio, both),
                "ratio_by_surface": surfaces(ratio, both),
                "fraction_below_0.5": frac(both),
                "fraction_below_0.5_by_surface": {"dorsal_0_1mm": frac(both & dorsal_1mm),
                                                  "ventral_0_1mm": frac(both & ventral_1mm),
                                                  "deep": frac(both & deep)},
            }

    # 4. BOLD tSNR -----------------------------------------------------------
    summary["bold"] = {}
    for bold in sorted((session_dir / "func").glob("*_bold.nii.gz")):
        acq = _acq(bold)
        mc = motion_correct(bold, work / "mcflirt")
        mc_img = nib.load(str(mc["path"]))
        ts = maps.tsnr(mc_img.get_fdata())
        ts_ref = _resample(nib.Nifti1Image(ts.astype(np.float32), mc_img.affine), ref)
        save_map(f"tsnr_{acq}", ts_ref)
        add_profile("tsnr", acq, ts_ref)
        slab = brain & np.isfinite(ts_ref)
        # Local distortion of the run itself, raw and corrected with the
        # reverse-PE pair acquired on the same grid (if any).
        bmeta = sidecar(bold)
        edge = {}
        if "PhaseEncodingDirection" in bmeta:
            axis = "ijk".index(bmeta["PhaseEncodingDirection"][0])
            mean_img = nib.Nifti1Image(mc_img.get_fdata().mean(axis=3).astype(np.float32), mc_img.affine)
            mean_path = work / "mcflirt" / bold.name.replace("_bold.nii.gz", "_desc-mean_bold.nii.gz")
            nib.save(mean_img, str(mean_path))
            images = [("raw", mean_img)]
            match = next((a for a, (g, _) in topups.items() if _same_grid(g, mc_img.slicer[..., 0])), None)
            if match is not None:
                fmeta = sidecar(pairs[match]["fwd"])
                index = 1 if fmeta["PhaseEncodingDirection"] == bmeta["PhaseEncodingDirection"] else 2
                corrected = apply_topup(mean_path, topups[match][1], index,
                                        work / "mcflirt" / mean_path.name.replace("desc-mean", "desc-meantopup"))
                images.append(("topup", nib.load(str(corrected))))
                save_map(f"bold_{acq}_mean_topup_t2grid", _resample(nib.load(str(corrected)), ref))
            for label, img in images:
                rows = edge_shift_table(img.get_fdata(), mean_img, ref, brain, axis, voxel_scale)
                edge_rows += [{"image": f"bold-{acq}_{label}", **r} for r in rows]
                edge[label] = summarise_edges(rows)
            edge["topup_pair"] = match
        summary["bold"][acq] = {
            "edge_shift": edge,
            "n_volumes": int(mc_img.shape[3]),
            "tsnr": maps.summarise(ts_ref, slab),
            "tsnr_by_surface": surfaces(ts_ref, slab),
            "fd_scaled_mm": {"mean": float(mc["fd_scaled_mm"].mean()),
                             "max": float(mc["fd_scaled_mm"].max())},
        }

    # 5. Chemical shift: does the bright dorsal layer reappear displaced? ----
    layer = _dorsal_bright_layer(t2, brain, vz)
    save_map("dorsal_bright_layer", layer.astype(float))
    summary["chemical_shift"] = {"bright_layer_mm3": float(layer.sum() * vz ** 3), "pairs": {}}
    if layer.sum() and b0_t:
        for acq, pair in pairs.items():
            meta = sidecar(pair["fwd"])
            epi_f = _first_volume(nib.load(str(pair["fwd"])))
            epi_r = _first_volume(nib.load(str(pair["rev"])))
            axis = "ijk".index(meta["PhaseEncodingDirection"][0])
            shift_px = int(round(chemical_shift_hz(b0_t) * float(meta["EffectiveEchoSpacing"]) * _n_pe(meta, epi_f)))
            lay = resample_from_to(nib.Nifti1Image(layer.astype(np.float32), ref.affine), epi_f,
                                   order=0).get_fdata() > 0.5
            if not lay.any():
                continue
            bm = resample_from_to(nib.Nifti1Image(brain.astype(np.float32), ref.affine), epi_f,
                                  order=0).get_fdata() > 0.5
            rows = {}
            for label, img in (("fwd", epi_f), ("rev", epi_r)):
                d = img.get_fdata()
                norm = np.median(d[bm]) if bm.any() else 1.0
                rows[label] = {str(s): float(d[np.roll(lay, s, axis=axis)].mean() / norm)
                               for s in (-shift_px, 0, shift_px)}
            # A fat ghost moves to opposite sides in the two polarities, so it
            # shows up as antisymmetry; anatomy at +/-shift is the same in both.
            k = str(shift_px)
            asym = ((rows["fwd"][k] - rows["fwd"][f"-{k}"]) - (rows["rev"][k] - rows["rev"][f"-{k}"]))
            summary["chemical_shift"]["pairs"][acq] = {"shift_px": shift_px, "axis": "ijk"[axis],
                                                        "layer_signal_rel_brain": rows,
                                                        "polarity_antisymmetry": float(asym)}

    # Write -------------------------------------------------------------------
    with open(out_dir / "tables" / "depth_profiles.tsv", "w", newline="") as f:
        cols = ["metric", "source", "surface", "depth_mm", "n", "median", "p25", "p75", "mean"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in profiles:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in cols})
    if edge_rows:
        cols = ["image", "slice", "n_columns"] + sorted(
            {k for r in edge_rows for k in r} - {"image", "slice", "n_columns"})
        with open(out_dir / "tables" / "edge_shift_per_slice.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", restval="n/a")
            w.writeheader()
            for r in edge_rows:
                w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        from neurofaune.preprocess.qc.artifacts.figures import plot_depth_profiles, plot_map_montage
        plot_depth_profiles(profiles, out_dir / "figures" / "depth_profiles.png")
        plot_map_montage(out_dir / "maps", ref, brain, out_dir / "figures" / "maps.png")
        if edge_rows:
            from neurofaune.preprocess.qc.artifacts.figures import plot_edge_shift
            plot_edge_shift(edge_rows, out_dir / "figures" / "edge_shift_per_slice.png")
    except Exception as e:  # noqa: BLE001 - figures must never sink the QC numbers
        logger.warning("artifact QC figures failed: %s", e)
    return summary
