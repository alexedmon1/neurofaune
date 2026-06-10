"""
Per-subject bet4animal skull stripping with an automatic parameter sweep.

FSL's ``bet4animal`` fits a smooth ellipsoidal brain surface, which is well
suited to rodent T2w — but its sizing parameters must match each brain's
geometry, and the rat brain's rostro-caudal extent vs in-plane size is
subject-dependent (positioning, FOV). Two parameters do the work:

- ``-x 1,1,<zscale>`` — z-elongation of the ellipsoid. Counter-intuitively a
  *lower* zscale gives a *longer* z-extent. Too high clips the cerebellum
  (caudal) and olfactory bulb (rostral) poles; too low overshoots into
  non-brain. The optimum is subject-specific (≈1.4–1.6 for this protocol).
- ``-c`` — the brain centre of gravity, derived data-drivenly from the shared
  foreground module so it adapts to each subject's positioning.
- ``-f`` — bet's fractional intensity threshold. A *lower* ``-f`` grows the brain
  outline, recovering the thin brain-edge rim that the smooth surface otherwise
  sits inside; it is swept alongside zscale so each subject gets the threshold
  that best fits its edge (the air gate stops it growing into background/eyes).

Rather than hand-tune, this module sweeps the sizing parameters and selects the
best mask with a **reference-free** score (no brain segmentation needed — on this
contrast intensity cannot isolate brain: a low threshold connects brain to
muscle, a high threshold isolates the eyes):

- ``air_leakage`` — fraction of mask voxels below the noise floor. Rises sharply
  when the mask balloons into background / overshoots the poles. Also a useful
  per-subject QC flag (high → review).
- ``boundary_fit`` — mean normalized image-gradient on the mask surface. Peaks
  when the surface sits on the real brain/skull edge.

Selection: among candidates passing a validity gate (air below ``air_max``,
single component, coverage in range), take those within 95% of the peak
boundary-fit and pick the **largest z-extent** — this keeps the boundary on real
edges while guaranteeing the poles (cerebellum/olfactory) are not clipped.

Validated across 7 subjects / 3 cohorts (incl. a hard rescued case and an
off-centre brain): whole-brain capture with no clipped brain; selected zscale
1.2–1.4, f 0.4–0.5. The auto-QC has lied before on this data — always view the
overlay; subjects flagged ``qc='review'`` (air > air_warn) especially so.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
from scipy import ndimage

from neurofaune.preprocess.utils.foreground import estimate_noise_floor, foreground_mask


# Defaults tuned for Bruker rat T2w at 256x256x41 @ 1.25x1.25x8mm (10x-scaled).
DEFAULT_ZSCALES: Tuple[float, ...] = (1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
DEFAULT_RADII: Tuple[int, ...] = (125,)
DEFAULT_FS: Tuple[float, ...] = (0.3, 0.4, 0.5)  # bet -f; lower => larger brain outline
DEFAULT_W = 1.0
DEFAULT_AIR_MAX = 0.20
DEFAULT_AIR_WARN = 0.18          # QC flag threshold (review if exceeded)
DEFAULT_COV_RANGE = (4.0, 13.0)  # plausible brain coverage (% of FOV)
DEFAULT_FOREGROUND_K = 4.0


def cog_from_foreground(img_data: np.ndarray, k: float = DEFAULT_FOREGROUND_K) -> np.ndarray:
    """Data-driven centre of gravity: centroid of the largest foreground component."""
    fg = foreground_mask(img_data, k=k).astype(bool)
    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = lab == (int(np.argmax(sizes)) + 1)
    coords = np.argwhere(fg)
    if coords.size == 0:
        raise ValueError("No foreground voxels for COG estimation")
    return coords.mean(0)


def _run_bet4animal(input_file: Path, out_root: Path, cog: np.ndarray,
                    zscale: float, radius: int, f: float, w: float) -> Optional[Path]:
    """Run bet4animal once; return the mask path (or None on failure)."""
    env = dict(os.environ)
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    cmd = [
        "bet4animal", str(input_file), str(out_root),
        "-f", str(f),
        "-c", str(int(cog[0])), str(int(cog[1])), str(int(cog[2])),
        "-r", str(radius),
        "-x", f"1,1,{zscale}",
        "-w", str(w),
        "-m",
    ]
    subprocess.run(cmd, capture_output=True, env=env)
    mask = Path(str(out_root) + "_mask.nii.gz")
    return mask if mask.exists() else None


def score_mask(mask: np.ndarray, img_data: np.ndarray,
               gmag_norm: np.ndarray, fg_thr: float) -> Optional[Dict[str, Any]]:
    """Reference-free quality metrics for a candidate mask.

    Returns None for an empty mask. ``air`` = fraction of mask voxels below the
    noise floor (ballooning penalty); ``bgrad`` = mean normalized gradient on the
    mask surface (edge-fit reward).
    """
    mask = mask.astype(bool)
    if not mask.any():
        return None
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    frac_largest = float(sizes.max() / mask.sum())
    air = float((img_data[mask] < fg_thr).mean())
    surf = mask & ~ndimage.binary_erosion(mask)
    bgrad = float(gmag_norm[surf].mean())
    zs = np.where(mask.any(axis=(0, 1)))[0]
    return dict(cov=100.0 * mask.mean(), zlo=int(zs.min()), zhi=int(zs.max()),
                nsl=int(len(zs)), ncomp=int(n), frac_largest=frac_largest,
                air=air, bgrad=bgrad)


def select_best(candidates: Sequence[Dict[str, Any]], air_max: float = DEFAULT_AIR_MAX,
                cov_range: Tuple[float, float] = DEFAULT_COV_RANGE,
                near_peak: float = 0.90) -> Optional[Dict[str, Any]]:
    """Pick the best candidate (pure function — unit-testable without FSL).

    Validity gate: air < air_max, single dominant component, coverage in range.
    Boundary-fit alone leans tight (a tighter surface hugs the steep edge gradient
    more precisely), which under-includes the brain-edge rim. So we use boundary-
    fit only to define a band of edge-accurate candidates — those within
    ``near_peak`` of the peak — and within that band choose the **most complete**
    mask (largest coverage), tie-broken by fit. The air gate keeps "more complete"
    from meaning "ballooned into background". Falls back to all candidates if none
    pass the validity gate.
    """
    if not candidates:
        return None
    for c in candidates:
        c["valid"] = (c["air"] < air_max and c["frac_largest"] > 0.95
                      and cov_range[0] <= c["cov"] <= cov_range[1])
    pool = [c for c in candidates if c["valid"]] or list(candidates)
    peak = max(c["bgrad"] for c in pool)
    near = [c for c in pool if c["bgrad"] >= near_peak * peak]
    return max(near, key=lambda c: (c["cov"], c["bgrad"]))


def skull_strip_bet4animal(
    input_file: Path,
    output_file: Path,
    mask_file: Path,
    work_dir: Path,
    cog: Optional[Sequence[float]] = None,
    zscales: Sequence[float] = DEFAULT_ZSCALES,
    radii: Sequence[int] = DEFAULT_RADII,
    fs: Sequence[float] = DEFAULT_FS,
    w: float = DEFAULT_W,
    air_max: float = DEFAULT_AIR_MAX,
    air_warn: float = DEFAULT_AIR_WARN,
    cov_range: Tuple[float, float] = DEFAULT_COV_RANGE,
    foreground_k: float = DEFAULT_FOREGROUND_K,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Skull-strip a rodent T2w with bet4animal + an automatic parameter sweep.

    Sweeps ``zscales`` x ``radii`` (data-driven COG unless ``cog`` given), scores
    each mask reference-free, and selects the best. Writes the brain-extracted
    image to ``output_file`` and the mask to ``mask_file``.

    The returned info dict includes the selected params, the air-leakage QC value
    and a ``qc_flag`` ('ok' / 'review' when air > ``air_warn``), and every swept
    candidate's metrics. ALWAYS view the overlay — the auto-QC has been fooled on
    this data before.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(input_file)
    d = img.get_fdata()
    if not (d > 0).any():
        raise ValueError("Input image has no non-zero voxels")

    floor = estimate_noise_floor(d, mask=None)
    fg_thr = floor.mean + foreground_k * floor.sigma
    gz, gy, gx = np.gradient(d)
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    gmag_norm = gmag / np.percentile(gmag, 99)

    cog_arr = np.asarray(cog, dtype=float) if cog is not None else cog_from_foreground(d, foreground_k)

    candidates = []
    for zs in zscales:
        for r in radii:
            for fval in fs:
                mp = _run_bet4animal(input_file, work_dir / f"b4a_z{zs}_r{r}_f{fval}",
                                     cog_arr, zs, r, fval, w)
                if mp is None:
                    continue
                m = nib.load(mp).get_fdata() > 0
                sc = score_mask(m, d, gmag_norm, fg_thr)
                if sc is None:
                    continue
                sc.update(zscale=float(zs), radius=int(r), f=float(fval), mask=str(mp))
                candidates.append(sc)

    best = select_best(candidates, air_max=air_max, cov_range=cov_range)
    if best is None:
        raise RuntimeError("bet4animal sweep produced no usable mask")

    # Write selected mask + brain-extracted image (float32, matching other methods).
    sel = nib.load(best["mask"]).get_fdata() > 0
    nib.save(nib.Nifti1Image(sel.astype(np.float32), img.affine, img.header), mask_file)
    nib.save(nib.Nifti1Image((d * sel).astype(np.float32), img.affine, img.header), output_file)

    qc_flag = "review" if best["air"] > air_warn else "ok"
    print(f"  bet4animal sweep: selected zscale={best['zscale']} f={best['f']} radius={best['radius']} "
          f"(cov={best['cov']:.1f}% z{best['zlo']}-{best['zhi']} air={best['air']:.3f} "
          f"bgrad={best['bgrad']:.3f}) — QC: {qc_flag}")
    if qc_flag == "review":
        print(f"  ⚠ air-leakage {best['air']:.3f} > {air_warn} — review the overlay for "
              f"lateral/dorsal over-inclusion.")

    info = {
        "method": "bet4animal",
        "extraction_ratio": best["cov"] / 100.0,
        "brain_voxels": int(sel.sum()),
        "total_voxels": int(sel.size),
        "cog": [round(float(c), 1) for c in cog_arr],
        "zscale": best["zscale"],
        "radius": best["radius"],
        "bet_f": best["f"],
        "bet_w": w,
        "z_extent": [best["zlo"], best["zhi"], best["nsl"]],
        "air_leakage": best["air"],
        "boundary_fit": best["bgrad"],
        "qc_flag": qc_flag,
        "candidates": [{k: c[k] for k in ("zscale", "radius", "f", "cov", "zlo", "zhi",
                                          "nsl", "air", "bgrad", "valid")} for c in candidates],
    }
    return output_file, mask_file, info
