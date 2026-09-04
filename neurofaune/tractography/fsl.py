"""FSL probabilistic tractography: BEDPOSTX and probtrackx2.

Provided as an alternative to the MRtrix3 path for two situations: studies
whose data is single-shell (where MSMT-CSD is unavailable), and cross-checking
an MRtrix result with an independent fibre model and tracking algorithm.

The model here is ball-and-sticks — an isotropic compartment plus ``n_fibres``
sticks — fit by MCMC. Its parameter count is ``2 + 3 * n_fibres`` (baseline
signal and diffusivity, then a volume fraction and two angles per stick), and
that number is what makes the model tractable or not for a given acquisition.
:func:`max_supported_fibres` derives the largest defensible ``n_fibres`` from
the number of measurements rather than trusting a config default, because the
failure is silent: FSL will fit two sticks to a seven-volume acquisition and
report nothing, even though the system is underdetermined and the second
fibre's orientation is unconstrained.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

from neurofaune.tractography.adequacy import (
    TractographyAdequacy,
    assess_tractography_adequacy,
)

logger = logging.getLogger(__name__)


# Measurements per free parameter required before a stick is considered
# supported. Three is conservative but keeps the MCMC posterior meaningful
# rather than prior-dominated.
MEASUREMENTS_PER_PARAMETER = 3.0


def ball_and_sticks_parameters(n_fibres: int) -> int:
    """Free parameters in a ball-and-``n_fibres``-sticks model."""
    return 2 + 3 * n_fibres


def max_supported_fibres(n_volumes: int, max_fibres: int = 3) -> int:
    """Largest ``n_fibres`` the measurement count can support.

    Returns 0 when not even one stick is supported.
    """
    for n in range(max_fibres, 0, -1):
        if n_volumes >= MEASUREMENTS_PER_PARAMETER * ball_and_sticks_parameters(n):
            return n
    return 0


def _run_fsl(cmd: Sequence[str], description: str, cwd: Optional[Path] = None) -> None:
    """Run an FSL command, raising with captured stderr on failure."""
    argv = [str(c) for c in cmd]
    logger.info("  %s", description)
    logger.debug("    %s", " ".join(argv))
    proc = subprocess.run(
        argv, capture_output=True, text=True,
        cwd=str(cwd) if cwd else None, env=dict(os.environ),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"FSL step failed: {description}\ncommand: {' '.join(argv)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )


def run_bedpostx(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    n_fibres: Optional[int] = None,
    model: int = 2,
    use_gpu: bool = True,
    adequacy: Optional[TractographyAdequacy] = None,
    require_csd: bool = False,
    voxel_scale: float = 10.0,
    force: bool = False,
) -> Path:
    """Fit the ball-and-sticks model with BEDPOSTX.

    Parameters
    ----------
    dwi_file, bval_file, bvec_file, mask_file : Path
        Preprocessed DWI, its gradient table, and a brain mask.
    output_dir : Path
        BEDPOSTX input directory; results land in ``<output_dir>.bedpostX``.
    n_fibres : int, optional
        Sticks per voxel. When omitted, set to the largest value the
        acquisition supports (see :func:`max_supported_fibres`), which is the
        safe default — a configured value that the data cannot support is
        reduced, with a warning.
    model : int
        1 = monoexponential, 2 = multiexponential (default), 3 = with
        gamma-distributed diffusivities. Model 2 or 3 suits multi-shell data.
    use_gpu : bool
        Prefer ``bedpostx_gpu`` when present.
    require_csd : bool
        Passed to the adequacy check. False by default here: unlike the MRtrix
        path, ball-and-sticks on a tensor-grade acquisition is a defensible
        (if limited) single-fibre model, so it warns rather than blocks.

    Returns
    -------
    Path
        The ``.bedpostX`` output directory.
    """
    bedpost_dir = Path(str(output_dir) + ".bedpostX")
    if (bedpost_dir / "mean_f1samples.nii.gz").exists() and not force:
        logger.info("BEDPOSTX output already exists: %s", bedpost_dir)
        return bedpost_dir

    if adequacy is None:
        adequacy = assess_tractography_adequacy(
            bval_file, bvec_file, dwi_file, voxel_scale=voxel_scale,
            require_csd=require_csd,
        )
    adequacy.raise_if_infeasible()

    n_volumes = int(nib.load(str(dwi_file)).shape[3])
    supported = max_supported_fibres(n_volumes)
    if supported == 0:
        raise ValueError(
            f"{n_volumes} volumes cannot support even a single-stick model "
            f"({ball_and_sticks_parameters(1)} parameters at "
            f"{MEASUREMENTS_PER_PARAMETER} measurements each)"
        )
    if n_fibres is None:
        n_fibres = supported
        logger.info("  n_fibres=%d chosen from %d volumes", n_fibres, n_volumes)
    elif n_fibres > supported:
        logger.warning(
            "  requested n_fibres=%d needs %d parameters, which %d volumes "
            "cannot constrain; reducing to %d",
            n_fibres, ball_and_sticks_parameters(n_fibres), n_volumes, supported,
        )
        n_fibres = supported

    # BEDPOSTX requires a rigidly-named input directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dwi_file, output_dir / "data.nii.gz")
    shutil.copy(mask_file, output_dir / "nodif_brain_mask.nii.gz")
    shutil.copy(bval_file, output_dir / "bvals")
    shutil.copy(bvec_file, output_dir / "bvecs")

    cmd_name = "bedpostx"
    if use_gpu and shutil.which("bedpostx_gpu"):
        cmd_name = "bedpostx_gpu"
    elif use_gpu:
        logger.warning("  bedpostx_gpu not found; falling back to CPU bedpostx")

    logger.info("=" * 70)
    logger.info("BEDPOSTX (%s): n_fibres=%d, model=%d", cmd_name, n_fibres, model)
    logger.info("=" * 70)
    _run_fsl(
        [cmd_name, str(output_dir), "-n", str(n_fibres), "-model", str(model)],
        f"{cmd_name} (slow)",
    )
    return bedpost_dir


def build_roi_seed_masks(
    parcellation: Path,
    output_dir: Path,
    labels: Optional[Sequence[int]] = None,
    min_voxels: int = 5,
) -> Path:
    """Split a parcellation into one binary mask per node for network tracking.

    Returns
    -------
    Path
        A text file listing the mask paths, in label order — the ``--network``
        seed list probtrackx2 expects. Node order in the resulting matrix
        follows this file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(parcellation))
    data = img.get_fdata().astype(np.int32)

    if labels is None:
        labels = [int(n) for n in np.unique(data) if n > 0]

    paths: List[str] = []
    for lab in labels:
        sel = data == lab
        if sel.sum() < min_voxels:
            logger.debug("  skipping label %d (%d voxels)", lab, int(sel.sum()))
            continue
        out = output_dir / f"node-{lab:04d}_mask.nii.gz"
        nib.save(
            nib.Nifti1Image(sel.astype(np.uint8), img.affine, img.header), str(out)
        )
        paths.append(str(out))

    seed_list = output_dir / "seed_list.txt"
    seed_list.write_text("\n".join(paths) + "\n")
    logger.info("  wrote %d ROI seed masks", len(paths))
    return seed_list


def run_probtrackx_connectome(
    bedpost_dir: Path,
    seed_list: Path,
    output_dir: Path,
    n_samples: int = 5000,
    step_length_mm: Optional[float] = None,
    curvature_threshold: float = 0.2,
    use_gpu: bool = True,
    distance_correction: bool = True,
    voxel_scale: float = 10.0,
    extra_args: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """Run probtrackx2 in network mode to produce a connectivity matrix.

    Parameters
    ----------
    bedpost_dir : Path
        ``.bedpostX`` directory from :func:`run_bedpostx`.
    seed_list : Path
        ROI mask list from :func:`build_roi_seed_masks`.
    n_samples : int
        Streamline samples per seed voxel.
    step_length_mm : float, optional
        Step length in **real** mm, converted internally. Defaults to half the
        smallest voxel dimension when omitted.
    distance_correction : bool
        Correct connectivity for distance travelled. Recommended: without it,
        nearby region pairs dominate the matrix purely through path length.

    Returns
    -------
    dict
        ``matrix`` (``fdt_network_matrix``) and ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "matrix": output_dir / "fdt_network_matrix",
        "output_dir": output_dir,
    }
    if out["matrix"].exists() and not force:
        logger.info("probtrackx network matrix already exists")
        return out

    cmd_name = "probtrackx2"
    if use_gpu and shutil.which("probtrackx2_gpu"):
        cmd_name = "probtrackx2_gpu"
    elif use_gpu:
        logger.warning("  probtrackx2_gpu not found; falling back to CPU probtrackx2")

    cmd: List[str] = [
        cmd_name,
        f"--samples={bedpost_dir / 'merged'}",
        f"--mask={bedpost_dir / 'nodif_brain_mask'}",
        f"--seed={seed_list}",
        "--network",
        f"--nsamples={n_samples}",
        f"--cthr={curvature_threshold}",
        f"--dir={output_dir}",
        "--forcedir", "--opd",
    ]
    if step_length_mm is not None:
        cmd.append(f"--steplength={step_length_mm * voxel_scale}")
    if distance_correction:
        cmd.append("--pd")
    if extra_args:
        cmd += list(extra_args)

    logger.info("=" * 70)
    logger.info("probtrackx2 network mode (%s)", cmd_name)
    logger.info("=" * 70)
    _run_fsl(cmd, f"{cmd_name} --network (slow)")
    return out
