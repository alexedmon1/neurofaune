"""Shared MRtrix3 front end: response estimation and FOD fitting.

Both downstream branches — fixel-based analysis
(:mod:`neurofaune.tractography.fixel`) and the structural connectome
(:mod:`neurofaune.tractography.connectome`) — consume the same fibre
orientation distributions, so that computation lives here once.

Why MSMT-CSD rather than single-shell CSD, for multi-shell rodent data:

- Three b-value shells plus b=0 let the three tissue responses (WM / GM / CSF)
  be solved simultaneously, which suppresses the GM and free-water partial
  volume that otherwise inflates apparent fibre density near ventricles and
  cortex — a large effect at rodent voxel sizes.
- MSMT fits the WM FOD against *every* shell jointly. With 30 directions per
  shell, single-shell CSD is capped near lmax=6 (28 coefficients, 2 degrees of
  freedom); pooling three shells gives 90 directions and makes lmax=8 well
  determined. Treating the shells separately would discard angular resolution
  the acquisition actually paid for.
- ``dwi2response dhollander`` derives all three responses unsupervised, without
  a tissue segmentation. This matters for rodents, where MRtrix's
  ``5ttgen fsl`` path depends on FAST/FIRST and their human priors do not
  transfer.

Intensity normalisation (``mtnormalise``) is applied by default because
cross-subject FOD amplitude comparison — the basis of fixel-based analysis and
of any group connectome contrast — is meaningless without it.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from neurofaune.config import get_config_value
from neurofaune.tractography.adequacy import (
    TractographyAdequacy,
    assess_tractography_adequacy,
)
from neurofaune.tractography.layout import resolve_output_dir, work_dir

logger = logging.getLogger(__name__)


class MRtrixNotFoundError(RuntimeError):
    """Raised when the MRtrix3 binaries cannot be located."""


def find_mrtrix_bin(config: Optional[dict] = None) -> Optional[Path]:
    """Locate the MRtrix3 ``bin`` directory.

    Resolution order: ``tractography.mrtrix_bin`` in config, then the
    ``MRTRIX_BIN`` environment variable, then ``PATH``.
    """
    if config is not None:
        configured = get_config_value(config, "tractography.mrtrix_bin", default=None)
        if configured:
            p = Path(str(configured))
            if (p / "dwi2fod").exists():
                return p
            logger.warning("configured tractography.mrtrix_bin has no dwi2fod: %s", p)

    env = os.environ.get("MRTRIX_BIN")
    if env and (Path(env) / "dwi2fod").exists():
        return Path(env)

    which = shutil.which("dwi2fod")
    if which:
        return Path(which).parent
    return None


def require_mrtrix(config: Optional[dict] = None) -> Path:
    """Return the MRtrix3 bin directory or raise with install guidance."""
    found = find_mrtrix_bin(config)
    if found is None:
        from neurofaune.utils.dependencies import INSTALL_HINTS

        raise MRtrixNotFoundError(
            "MRtrix3 not found: neither tractography.mrtrix_bin, MRTRIX_BIN, nor "
            "PATH resolved to a directory containing dwi2fod.\n\n"
            f"{INSTALL_HINTS['MRtrix3']}\n\n"
            "Check what is and is not installed with:\n"
            "    neurofaune check-deps --group tractography"
        )
    return found


def _run(
    cmd: Sequence[str],
    bin_dir: Path,
    description: str,
    nthreads: Optional[int] = None,
) -> None:
    """Run an MRtrix command, raising with captured stderr on failure."""
    argv: List[str] = [str(bin_dir / cmd[0]), *[str(c) for c in cmd[1:]]]
    if nthreads is not None:
        argv += ["-nthreads", str(nthreads)]

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

    logger.info("  %s", description)
    logger.debug("    %s", " ".join(argv))
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MRtrix step failed: {description}\n"
            f"command: {' '.join(argv)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def convert_to_mif(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_file: Path,
    bin_dir: Path,
    mask_file: Optional[Path] = None,
    nthreads: Optional[int] = None,
) -> Path:
    """Convert FSL-format DWI (+ gradient table) to a single MRtrix ``.mif``.

    Embedding the gradient scheme in the image removes the most common source
    of silent tractography error: a bvec/bval pair drifting out of sync with
    the volumes it describes.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "mrconvert", dwi_file, output_file,
            "-fslgrad", bvec_file, bval_file,
            "-datatype", "float32", "-force", "-quiet",
        ],
        bin_dir, f"mrconvert -> {output_file.name}", nthreads,
    )
    return output_file


def run_msmt_csd(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    config: Optional[dict] = None,
    adequacy: Optional[TractographyAdequacy] = None,
    wm_lmax: Optional[int] = None,
    normalise: bool = True,
    force: bool = False,
    nthreads: Optional[int] = None,
    voxel_scale: float = 10.0,
    derive_layout: bool = True,
) -> Dict[str, Path]:
    """Estimate tissue responses and fit FODs for one session.

    Runs ``dwi2response dhollander`` then ``dwi2fod`` (MSMT-CSD when the data
    is multi-shell, single-tissue CSD otherwise), followed by ``mtnormalise``.

    The acquisition is assessed first and the run refuses outright on data that
    cannot support an FOD — see
    :func:`neurofaune.tractography.adequacy.assess_tractography_adequacy`.

    Parameters
    ----------
    dwi_file, bval_file, bvec_file, mask_file : Path
        Preprocessed DWI, its gradient table, and a brain mask.
    output_dir : Path
        The **study root**. Per the package's workflow convention the layout
        below it is derived: kept outputs go to
        ``{study_root}/tractography/{subject}/{session}/`` and intermediates to
        ``{study_root}/work/{subject}/{session}/tractography/``. Pass
        ``derive_layout=False`` to treat this as a literal destination.
    subject, session : str
        BIDS identifiers, used for output naming.
    config : dict, optional
        Study config; supplies ``tractography.mrtrix_bin`` and defaults.
    adequacy : TractographyAdequacy, optional
        Precomputed verdict. Assessed here when omitted.
    wm_lmax : int, optional
        Override the WM FOD spherical harmonic order. Defaults to the largest
        order the acquisition determines, capped at 8.
    normalise : bool
        Apply ``mtnormalise``. Required for any cross-subject comparison.
    force : bool
        Recompute even when outputs exist.
    nthreads : int, optional
        Threads per MRtrix call.
    voxel_scale : float
        Header-to-real-mm divisor (see ``bids.voxel_scale``).
    derive_layout : bool
        Derive the study layout from ``output_dir``. Set False only for tests
        or one-off exploration.

    Returns
    -------
    dict
        ``wm_fod``, ``gm_fod``, ``csf_fod`` (normalised when ``normalise``),
        ``wm_response``, ``gm_response``, ``csf_response``, ``dwi_mif``,
        ``mask_mif``, and ``adequacy``.
    """
    bin_dir = require_mrtrix(config)
    session_out = resolve_output_dir(output_dir, subject, session, derive_layout)
    scratch_out = (
        work_dir(output_dir, subject, session) if derive_layout else session_out
    )
    session_out.mkdir(parents=True, exist_ok=True)
    scratch_out.mkdir(parents=True, exist_ok=True)

    if adequacy is None:
        adequacy = assess_tractography_adequacy(
            bval_file, bvec_file, dwi_file, voxel_scale=voxel_scale
        )
    adequacy.raise_if_infeasible()

    if wm_lmax is None:
        wm_lmax = adequacy.wm_lmax

    prefix = f"{subject}_{session}"
    model_tag = "MSMT" if adequacy.recommended_model == "msmt_csd" else "CSD"

    # The un-normalised FODs and the .mif copy of the DWI are intermediates:
    # the former is superseded by mtnormalise, the latter duplicates a NIfTI
    # that already exists (~263 MB/session on this study's geometry).
    out = {
        "wm_fod": scratch_out / f"{prefix}_model-{model_tag}_label-WM_fod.mif",
        "gm_fod": scratch_out / f"{prefix}_model-MSMT_label-GM_fod.mif",
        "csf_fod": scratch_out / f"{prefix}_model-MSMT_label-CSF_fod.mif",
        "wm_response": session_out / f"{prefix}_label-WM_response.txt",
        "gm_response": session_out / f"{prefix}_label-GM_response.txt",
        "csf_response": session_out / f"{prefix}_label-CSF_response.txt",
        "dwi_mif": scratch_out / f"{prefix}_desc-preproc_dwi.mif",
        "mask_mif": session_out / f"{prefix}_desc-brain_mask.mif",
    }
    # When normalising, the kept outputs are the normalised FODs in the session
    # directory; resolve their names now so the existence check tests the right
    # files rather than the intermediates.
    final_wm = (
        session_out / f"{prefix}_model-{model_tag}_label-WM_desc-norm_fod.mif"
        if normalise
        else out["wm_fod"]
    )

    if final_wm.exists() and not force:
        logger.info("FODs already exist for %s %s (use force=True)", subject, session)
        if normalise:
            out["wm_fod"] = final_wm
            for tissue in ("gm", "csf"):
                cand = session_out / (
                    f"{prefix}_model-MSMT_label-{tissue.upper()}_desc-norm_fod.mif"
                )
                out[f"{tissue}_fod"] = cand if cand.exists() else None
        out["adequacy"] = adequacy
        return out

    logger.info("=" * 70)
    logger.info("MSMT-CSD: %s %s (model=%s, WM lmax=%d)",
                subject, session, adequacy.recommended_model, wm_lmax)
    logger.info("=" * 70)

    convert_to_mif(dwi_file, bval_file, bvec_file, out["dwi_mif"], bin_dir,
                   nthreads=nthreads)
    _run(["mrconvert", mask_file, out["mask_mif"], "-force", "-quiet"],
         bin_dir, "mrconvert mask", nthreads)

    multishell = adequacy.recommended_model == "msmt_csd"

    # dhollander yields all three responses regardless of shell count; for
    # single-shell data only the WM response is usable downstream.
    scratch = output_dir / f".scratch_{prefix}"
    if scratch.exists():
        shutil.rmtree(scratch)
    _run(
        [
            "dwi2response", "dhollander", out["dwi_mif"],
            out["wm_response"], out["gm_response"], out["csf_response"],
            "-mask", out["mask_mif"], "-scratch", scratch, "-force",
        ],
        bin_dir, "dwi2response dhollander", nthreads,
    )
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)

    if multishell:
        # GM and CSF are isotropic compartments: lmax=0. Only WM carries
        # orientation, so only WM gets the full expansion.
        _run(
            [
                "dwi2fod", "msmt_csd", out["dwi_mif"],
                out["wm_response"], out["wm_fod"],
                out["gm_response"], out["gm_fod"],
                out["csf_response"], out["csf_fod"],
                "-mask", out["mask_mif"],
                "-lmax", f"{wm_lmax},0,0", "-force",
            ],
            bin_dir, f"dwi2fod msmt_csd (lmax {wm_lmax},0,0)", nthreads,
        )
        tissues = ["wm", "gm", "csf"]
    else:
        _run(
            [
                "dwi2fod", "csd", out["dwi_mif"],
                out["wm_response"], out["wm_fod"],
                "-mask", out["mask_mif"], "-lmax", str(wm_lmax), "-force",
            ],
            bin_dir, f"dwi2fod csd (lmax {wm_lmax})", nthreads,
        )
        out["gm_fod"] = out["csf_fod"] = None
        tissues = ["wm"]

    if normalise:
        # mtnormalise puts every subject on a common intensity scale. Without
        # it, group FOD-amplitude differences are dominated by receive-coil
        # and reconstruction scaling rather than by tissue.
        args: List[str] = ["mtnormalise"]
        norm_paths = {}
        for t in tissues:
            src = out[f"{t}_fod"]
            # Normalised FODs are the kept product, so they land in the session
            # directory even though their inputs live in work/.
            dst = session_out / src.name.replace("_fod.mif", "_desc-norm_fod.mif")
            norm_paths[t] = dst
            args += [str(src), str(dst)]
        args += ["-mask", str(out["mask_mif"]), "-force"]
        _run(args, bin_dir, "mtnormalise", nthreads)
        for t in tissues:
            out[f"{t}_fod"] = norm_paths[t]

    out["adequacy"] = adequacy
    logger.info("MSMT-CSD complete: %s", out["wm_fod"].name)
    return out
