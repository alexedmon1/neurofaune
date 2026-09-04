"""Streamline generation (iFOD2) and SIFT2 filtering.

Two rodent-specific concerns shape this module.

**Units.** Rodent data in this package is stored with voxel sizes scaled 10x
for FSL/ANTs compatibility, so a header millimetre is a tenth of a real one.
Every length in this API — step size, minimum and maximum streamline length —
is specified in *real* millimetres and converted internally using
``voxel_scale``. Passing MRtrix defaults straight through would silently mean
something different here than the numbers suggest.

**Streamline counts are not connectivity.** Raw streamline counts between two
regions reflect tract length, curvature and seeding density as much as they
reflect any underlying connection, and the bias does not cancel in a group
contrast. SIFT2 solves for a per-streamline weight making the tractogram's
implied fibre density consistent with the FOD lobes that generated it, so
weighted edges are comparable across subjects. It is applied by default; the
weights it emits must then be carried into
:mod:`neurofaune.tractography.connectome`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from neurofaune.tractography.mrtrix import _run, require_mrtrix

logger = logging.getLogger(__name__)


def run_tractography(
    wm_fod: Path,
    output_dir: Path,
    subject: str,
    session: str,
    mask_file: Optional[Path] = None,
    fivett_file: Optional[Path] = None,
    n_streamlines: str = "1M",
    algorithm: str = "iFOD2",
    cutoff: float = 0.05,
    step_mm: Optional[float] = None,
    min_length_mm: Optional[float] = None,
    max_length_mm: Optional[float] = None,
    max_angle: Optional[float] = None,
    voxel_scale: float = 10.0,
    backtrack: bool = True,
    sift2: bool = True,
    config: Optional[dict] = None,
    force: bool = False,
    nthreads: Optional[int] = None,
    seed_strategy: str = "auto",
) -> Dict[str, Optional[Path]]:
    """Generate a tractogram from a WM FOD, optionally ACT-constrained.

    Parameters
    ----------
    wm_fod : Path
        White-matter FOD from :func:`~neurofaune.tractography.mrtrix.run_msmt_csd`.
        Should be the ``mtnormalise``-normalised one for group work.
    output_dir : Path
        Destination for the tractogram and SIFT2 weights.
    subject, session : str
        BIDS identifiers for naming.
    mask_file : Path, optional
        Brain mask (``.mif`` or ``.nii.gz``), used for seeding and propagation
        when no 5TT is supplied.
    fivett_file : Path, optional
        5TT image **already resampled into FOD space**. Enables ACT and
        GM/WM-interface seeding. See :mod:`neurofaune.tractography.fivett`.
    n_streamlines : str
        Target count, MRtrix shorthand (``"1M"``, ``"500k"``).
    algorithm : str
        ``tckgen`` algorithm. ``iFOD2`` (default) is probabilistic and
        second-order; use ``SD_STREAM`` for a deterministic comparison.
    cutoff : float
        FOD amplitude termination threshold.
    step_mm, min_length_mm, max_length_mm : float, optional
        Lengths in **real** millimetres. When omitted, derived from voxel size
        (step = half the smallest dimension) and, for lengths, from MRtrix
        defaults scaled appropriately.
    max_angle : float, optional
        Maximum angle per step, degrees.
    voxel_scale : float
        Header-to-real-mm divisor (``bids.voxel_scale``).
    backtrack : bool
        Allow ACT backtracking. Ignored without a 5TT.
    sift2 : bool
        Run ``tcksift2`` and emit per-streamline weights.
    seed_strategy : str
        ``"auto"`` uses the GM/WM interface when a 5TT is available and the
        brain mask otherwise; force with ``"gmwmi"`` or ``"mask"``.

    Returns
    -------
    dict
        ``tractogram``, ``weights`` (None unless ``sift2``), ``gmwmi``
        (None unless computed), ``mu`` (SIFT2 proportionality coefficient
        file, None unless ``sift2``).
    """
    bin_dir = require_mrtrix(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"

    out: Dict[str, Optional[Path]] = {
        "tractogram": output_dir / f"{prefix}_desc-{algorithm}{n_streamlines}_tractogram.tck",
        "weights": None,
        "gmwmi": None,
        "mu": None,
    }
    if sift2:
        out["weights"] = output_dir / f"{prefix}_desc-SIFT2_weights.txt"
        out["mu"] = output_dir / f"{prefix}_desc-SIFT2_mu.txt"

    if out["tractogram"].exists() and not force:
        logger.info("tractogram already exists for %s %s", subject, session)
        if not (sift2 and out["weights"] and not out["weights"].exists()):
            return out

    use_act = fivett_file is not None and Path(fivett_file).exists()

    # Lengths are quoted in real mm but MRtrix sees the scaled header, so
    # convert on the way in.
    def to_header(mm: Optional[float]) -> Optional[float]:
        return None if mm is None else mm * voxel_scale

    logger.info("=" * 70)
    logger.info(
        "Tractography: %s %s (%s, %s streamlines, ACT=%s)",
        subject, session, algorithm, n_streamlines, use_act,
    )
    logger.info("=" * 70)

    cmd: List[str] = ["tckgen", str(wm_fod), str(out["tractogram"])]

    # --- Seeding ------------------------------------------------------------
    want_gmwmi = seed_strategy == "gmwmi" or (seed_strategy == "auto" and use_act)
    if want_gmwmi and use_act:
        gmwmi = output_dir / f"{prefix}_desc-gmwmi_mask.mif"
        _run(
            ["5tt2gmwmi", str(fivett_file), str(gmwmi), "-force"],
            bin_dir, "5tt2gmwmi (seed interface)", nthreads,
        )
        out["gmwmi"] = gmwmi
        # Seeding the GM/WM interface rather than the whole white matter makes
        # streamline density reflect cortical surface area instead of tract
        # volume, which is the assumption SIFT2 is built on.
        cmd += ["-seed_gmwmi", str(gmwmi)]
    elif mask_file is not None:
        cmd += ["-seed_image", str(mask_file)]
    else:
        raise ValueError("need mask_file or fivett_file to seed tractography")

    if use_act:
        cmd += ["-act", str(fivett_file)]
        if backtrack and algorithm.startswith("iFOD"):
            cmd += ["-backtrack"]
    elif mask_file is not None:
        cmd += ["-mask", str(mask_file)]

    cmd += ["-algorithm", algorithm, "-select", n_streamlines, "-cutoff", str(cutoff)]

    if step_mm is not None:
        cmd += ["-step", str(to_header(step_mm))]
    if min_length_mm is not None:
        cmd += ["-minlength", str(to_header(min_length_mm))]
    if max_length_mm is not None:
        cmd += ["-maxlength", str(to_header(max_length_mm))]
    if max_angle is not None:
        cmd += ["-angle", str(max_angle)]
    cmd += ["-force"]

    _run(cmd, bin_dir, f"tckgen {algorithm}", nthreads)

    if sift2:
        # SIFT2 needs the same ACT information as tckgen; without it the
        # model cannot tell a genuine termination from a mask-edge truncation.
        sift_cmd: List[str] = [
            "tcksift2", str(out["tractogram"]), str(wm_fod), str(out["weights"]),
            "-out_mu", str(out["mu"]),
        ]
        if use_act:
            sift_cmd += ["-act", str(fivett_file)]
        sift_cmd += ["-force"]
        _run(sift_cmd, bin_dir, "tcksift2", nthreads)

    logger.info("Tractography complete: %s", out["tractogram"].name)
    return out
