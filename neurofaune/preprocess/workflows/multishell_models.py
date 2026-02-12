"""
Multi-shell diffusion model fitting: DKI and NODDI.

This module provides implementations of advanced diffusion models that
require multi-shell data (>= 2 non-zero b-value shells):

- **DKI** (Diffusion Kurtosis Imaging): Extends DTI with kurtosis terms
  quantifying non-Gaussian diffusion. Uses DIPY. Outputs: MK, AK, RK, KFA.

- **NODDI** (Neurite Orientation Dispersion and Density Imaging): Biophysical
  model of neurite density and dispersion. Uses AMICO (100x faster than
  traditional fitting). Outputs: FICVF, ODI, FISO.

These models are intended to be run *after* standard DWI preprocessing
(eddy correction, DTI fitting) has completed via ``run_dwi_preprocessing()``.

References
----------
Jensen et al. (2005) "Diffusional kurtosis imaging" MRM 53(6):1432-1440
Zhang et al. (2012) "NODDI" NeuroImage 61(4):1000-1016
Daducci et al. (2015) "AMICO" NeuroImage 105:32-44
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np

from ..utils.dwi_utils import round_bvals_to_shells

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------

def detect_multishell(
    bval_file: Path,
    b0_threshold: float = 50.0,
) -> dict:
    """Detect whether DWI data is multi-shell.

    Uses :func:`round_bvals_to_shells` to cluster Bruker-style per-direction
    b-values into discrete shells.

    Parameters
    ----------
    bval_file : Path
        FSL-format b-values file.
    b0_threshold : float
        Values below this are considered b=0.

    Returns
    -------
    dict
        ``is_multishell`` : bool – True if >= 2 non-zero shells.
        ``shells`` : list of (b-value, n_directions) tuples.
        ``n_shells`` : int – number of non-zero shells.
        ``max_bvalue`` : float – largest shell b-value.
    """
    bvals_raw = np.loadtxt(bval_file)
    if bvals_raw.ndim > 1:
        bvals_raw = bvals_raw.flatten()

    bvals = round_bvals_to_shells(bvals_raw, b0_threshold=b0_threshold)

    unique = np.unique(bvals)
    non_zero = unique[unique >= b0_threshold]

    shells = []
    # b=0 shell
    n_b0 = int(np.sum(bvals < b0_threshold))
    if n_b0 > 0:
        shells.append((0, n_b0))
    for bv in sorted(non_zero):
        shells.append((int(bv), int(np.sum(bvals == bv))))

    return {
        "is_multishell": len(non_zero) >= 2,
        "shells": shells,
        "n_shells": len(non_zero),
        "max_bvalue": float(non_zero.max()) if len(non_zero) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# DKI fitting (DIPY)
# ---------------------------------------------------------------------------

_MIN_DIRS_FULL_DKI = 15
"""Minimum directions per non-zero shell for reliable full DKI fitting.

Full DKI estimates 21 parameters (6 diffusion tensor + 15 kurtosis tensor).
Below this threshold we fall back to the more robust Mean Signal DKI (MSDKI)
which only estimates 2 parameters per voxel.
"""


def fit_dki(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    min_kurtosis: float = 0.0,
    max_kurtosis: float = 3.0,
) -> Dict[str, Path]:
    """Fit Diffusion Kurtosis Imaging model using DIPY.

    Automatically selects the fitting strategy based on the number of
    gradient directions per shell:

    - **>= 15 dirs/shell** : Full DKI (21-parameter tensor fit).
      Outputs MK, AK, RK, KFA.
    - **< 15 dirs/shell** : Mean Signal DKI (2-parameter robust fit).
      Outputs MSK (mean signal kurtosis) and MSD (mean signal diffusivity).
      This is common for Bruker rodent acquisitions with ~6 dirs/shell.

    Standard DTI metrics (FA, MD, AD, RD) are assumed to already exist from
    ``dtifit`` in the DWI preprocessing pipeline and are not re-computed.

    Parameters
    ----------
    dwi_file, bval_file, bvec_file, mask_file : Path
        Preprocessed DWI data and brain mask.
    output_dir : Path
        Directory for output NIfTI files.
    subject, session : str
        BIDS identifiers for file naming.
    min_kurtosis, max_kurtosis : float
        Clipping range for kurtosis values (default 0-3).

    Returns
    -------
    dict
        Mapping of metric name to output Path.
        Full DKI: ``{"MK": ..., "AK": ..., "RK": ..., "KFA": ...}``
        MSDKI:    ``{"MSK": ..., "MSD": ...}``
    """
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs

    logger.info("=" * 70)
    logger.info("DKI Fitting (DIPY)")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading DWI data...")
    img = nib.load(dwi_file)
    data = img.get_fdata()
    affine = img.affine

    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs=bvecs)

    logger.info(f"  Data shape: {data.shape}")
    logger.info(f"  Unique b-values: {np.unique(bvals.astype(int))}")

    mask_img = nib.load(mask_file)
    mask = mask_img.get_fdata().astype(bool)
    logger.info(f"  Mask voxels: {mask.sum()}")

    # Determine directions per shell to pick fitting strategy
    shelled = round_bvals_to_shells(bvals)
    non_zero_shells = np.unique(shelled[shelled >= 50])
    min_dirs = min(int(np.sum(shelled == bv)) for bv in non_zero_shells)
    logger.info(f"  Min directions per shell: {min_dirs}")

    if min_dirs >= _MIN_DIRS_FULL_DKI:
        return _fit_full_dki(
            data, gtab, mask, affine, output_dir, subject, session,
            min_kurtosis, max_kurtosis,
        )
    else:
        logger.info(
            f"  Directions per shell ({min_dirs}) < {_MIN_DIRS_FULL_DKI}: "
            "using Mean Signal DKI (MSDKI) for robust fitting."
        )
        return _fit_msdki(
            data, gtab, mask, affine, output_dir, subject, session,
            min_kurtosis, max_kurtosis,
        )


def _fit_full_dki(
    data: np.ndarray,
    gtab,
    mask: np.ndarray,
    affine: np.ndarray,
    output_dir: Path,
    subject: str,
    session: str,
    min_kurtosis: float,
    max_kurtosis: float,
) -> Dict[str, Path]:
    """Full 21-parameter DKI tensor fit."""
    from dipy.reconst import dki

    logger.info("Fitting full DKI model (this may take several minutes)...")
    dki_model = dki.DiffusionKurtosisModel(gtab)
    dki_fit = dki_model.fit(data, mask=mask)

    logger.info("Computing kurtosis metrics...")
    mk = dki_fit.mk(min_kurtosis=min_kurtosis, max_kurtosis=max_kurtosis)
    ak = dki_fit.ak(min_kurtosis=min_kurtosis, max_kurtosis=max_kurtosis)
    rk = dki_fit.rk(min_kurtosis=min_kurtosis, max_kurtosis=max_kurtosis)
    kfa = dki_fit.kfa

    prefix = f"{subject}_{session}_model-DKI"
    outputs: Dict[str, Path] = {}

    for name, metric_data in [("MK", mk), ("AK", ak), ("RK", rk), ("KFA", kfa)]:
        out_path = output_dir / f"{prefix}_{name}.nii.gz"
        nib.save(
            nib.Nifti1Image(metric_data.astype(np.float32), affine),
            out_path,
        )
        outputs[name] = out_path
        logger.info(f"  Saved {name}: {out_path.name}")

    logger.info("Full DKI fitting complete.")
    return outputs


def _fit_msdki(
    data: np.ndarray,
    gtab,
    mask: np.ndarray,
    affine: np.ndarray,
    output_dir: Path,
    subject: str,
    session: str,
    min_kurtosis: float,
    max_kurtosis: float,
) -> Dict[str, Path]:
    """Mean Signal DKI – robust 2-parameter fit for low-direction data.

    MSDKI estimates kurtosis from the powder-averaged (direction-averaged)
    signal decay, requiring only 2 parameters instead of 21.  This makes
    it far more stable when the number of gradient directions per shell
    is small (e.g. 6 directions in typical Bruker rodent acquisitions).

    Outputs
    -------
    MSK : Mean Signal Kurtosis (analogous to MK from full DKI)
    MSD : Mean Signal Diffusivity (analogous to MD from full DKI)
    """
    from dipy.reconst.msdki import MeanDiffusionKurtosisModel

    logger.info("Fitting Mean Signal DKI (MSDKI) model...")
    msdki_model = MeanDiffusionKurtosisModel(gtab)
    msdki_fit = msdki_model.fit(data, mask=mask)

    msk = msdki_fit.msk  # mean signal kurtosis
    msd = msdki_fit.msd  # mean signal diffusivity

    # Clip kurtosis to plausible range (same convention as full DKI)
    msk = np.clip(msk, min_kurtosis, max_kurtosis)

    # Clip MSD to non-negative (diffusivity must be >= 0)
    msd = np.clip(msd, 0, None)

    prefix = f"{subject}_{session}_model-DKI"
    outputs: Dict[str, Path] = {}

    for name, metric_data in [("MSK", msk), ("MSD", msd)]:
        out_path = output_dir / f"{prefix}_{name}.nii.gz"
        nib.save(
            nib.Nifti1Image(metric_data.astype(np.float32), affine),
            out_path,
        )
        outputs[name] = out_path
        logger.info(f"  Saved {name}: {out_path.name}")

    logger.info("MSDKI fitting complete.")
    return outputs


# ---------------------------------------------------------------------------
# NODDI fitting (AMICO)
# ---------------------------------------------------------------------------

def fit_noddi(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Optional[Path] = None,
    parallel_diffusivity: float = 1.7e-3,
    isotropic_diffusivity: float = 3.0e-3,
) -> Dict[str, Path]:
    """Fit NODDI model using AMICO.

    AMICO reformulates NODDI as a linear inverse problem with pre-computed
    dictionaries, making it ~100x faster than traditional fitting.

    Parameters
    ----------
    dwi_file, bval_file, bvec_file, mask_file : Path
        Preprocessed DWI data and brain mask.
    output_dir : Path
        Directory for final BIDS-named NIfTI files.
    subject, session : str
        BIDS identifiers for file naming.
    work_dir : Path, optional
        Temporary workspace for AMICO.  Defaults to ``output_dir/work``.
    parallel_diffusivity : float
        Intra-axonal diffusivity in mm²/s (default 1.7e-3, rodent-appropriate).
    isotropic_diffusivity : float
        CSF diffusivity in mm²/s (default 3.0e-3).

    Returns
    -------
    dict
        Mapping ``{"FICVF": Path, "ODI": Path, "FISO": Path}``.

    Raises
    ------
    ImportError
        If ``dmri-amico`` is not installed.
    """
    try:
        import amico
    except ImportError:
        raise ImportError(
            "NODDI fitting requires the dmri-amico package. "
            "Install with: uv pip install 'dmri-amico>=2.0.3'"
        )

    logger.info("=" * 70)
    logger.info("NODDI Fitting (AMICO)")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    if work_dir is None:
        work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Prepare AMICO workspace
    study_dir = work_dir / "amico_workspace"
    subject_id = "subject"
    subject_dir = study_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for src, dst_name in [
        (dwi_file, "dwi.nii.gz"),
        (bval_file, "dwi.bval"),
        (bvec_file, "dwi.bvec"),
        (mask_file, "mask.nii.gz"),
    ]:
        dst = subject_dir / dst_name
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    # AMICO setup
    logger.info("Initializing AMICO...")
    amico.core.setup()

    ae = amico.Evaluation(str(study_dir), subject_id)

    # Convert FSL bval/bvec to AMICO scheme
    logger.info("Converting bval/bvec to AMICO scheme format...")
    amico.util.fsl2scheme(
        bvalsFilename=str(subject_dir / "dwi.bval"),
        bvecsFilename=str(subject_dir / "dwi.bvec"),
        schemeFilename=str(subject_dir / "dwi.scheme"),
    )

    logger.info("Loading DWI data...")
    ae.load_data(
        dwi_filename="dwi.nii.gz",
        scheme_filename="dwi.scheme",
        mask_filename="mask.nii.gz",
        b0_thr=10,
    )

    # Configure NODDI model
    logger.info(f"  Parallel diffusivity: {parallel_diffusivity:.3e} mm^2/s")
    logger.info(f"  Isotropic diffusivity: {isotropic_diffusivity:.3e} mm^2/s")

    ae.set_model("NODDI")
    ae.model.set(dPar=parallel_diffusivity, dIso=isotropic_diffusivity)

    logger.info("Generating response function dictionary...")
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    logger.info("Fitting NODDI model...")
    ae.fit()

    logger.info("Saving NODDI outputs...")
    ae.save_results()

    # Map AMICO outputs to BIDS-named files
    amico_out = subject_dir / "AMICO" / "NODDI"
    prefix = f"{subject}_{session}_model-NODDI"
    outputs: Dict[str, Path] = {}

    metric_map = {
        "fit_NDI.nii.gz": "FICVF",
        "fit_ODI.nii.gz": "ODI",
        "fit_FWF.nii.gz": "FISO",
    }

    for amico_name, metric_name in metric_map.items():
        src = amico_out / amico_name
        dst = output_dir / f"{prefix}_{metric_name}.nii.gz"
        if src.exists():
            shutil.copy2(src, dst)
            outputs[metric_name] = dst
            logger.info(f"  Saved {metric_name}: {dst.name}")
        else:
            logger.warning(f"  {amico_name} not found in AMICO output")

    logger.info("NODDI fitting complete.")
    return outputs


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_multishell_fitting(
    output_dir: Path,
    subject: str,
    session: str,
    do_dki: bool = True,
    do_noddi: bool = True,
    work_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run DKI and/or NODDI fitting on preprocessed multi-shell DWI data.

    Discovers preprocessed inputs from the standard derivatives layout
    produced by :func:`run_dwi_preprocessing`.

    Parameters
    ----------
    output_dir : Path
        Study root directory (contains ``derivatives/``).
    subject, session : str
        BIDS identifiers (e.g. ``"sub-PRE10"``, ``"ses-01"``).
    do_dki : bool
        Fit DKI model (default True).
    do_noddi : bool
        Fit NODDI model (default True).
    work_dir : Path, optional
        Working directory for AMICO.  Defaults to ``output_dir/work/multishell``.
    force : bool
        Overwrite existing outputs (default False).

    Returns
    -------
    dict
        ``"dki"`` : dict of DKI metric paths (or None).
        ``"noddi"`` : dict of NODDI metric paths (or None).
        ``"qc"`` : dict with QC report paths (or None).
        ``"shell_info"`` : dict from :func:`detect_multishell`.
    """
    logger.info("=" * 70)
    logger.info(f"Multi-Shell Diffusion Model Fitting: {subject} {session}")
    logger.info("=" * 70)

    # Locate preprocessed inputs
    deriv_dir = output_dir / "derivatives" / subject / session / "dwi"
    prefix = f"{subject}_{session}"

    dwi_file = deriv_dir / f"{prefix}_desc-preproc_dwi.nii.gz"
    bval_file = deriv_dir / f"{prefix}_desc-preproc_dwi.bval"
    bvec_file = deriv_dir / f"{prefix}_desc-preproc_dwi.bvec"
    mask_file = deriv_dir / f"{prefix}_desc-brain_mask.nii.gz"

    for f in [dwi_file, bval_file, bvec_file, mask_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"Required preprocessed file not found: {f}\n"
                "Run DWI preprocessing first."
            )

    # Detect shells
    shell_info = detect_multishell(bval_file)
    logger.info(f"  Shells detected: {shell_info['n_shells']} non-zero")
    for bv, n in shell_info["shells"]:
        logger.info(f"    b={bv:>5d}: {n} volumes")

    if not shell_info["is_multishell"]:
        logger.warning(
            "Data is NOT multi-shell (< 2 non-zero shells). "
            "DKI/NODDI require multi-shell data."
        )
        return {
            "dki": None,
            "noddi": None,
            "qc": None,
            "shell_info": shell_info,
        }

    results: Dict[str, Any] = {"shell_info": shell_info, "dki": None, "noddi": None, "qc": None}

    # --- DKI ---
    if do_dki:
        dki_marker = deriv_dir / f"{prefix}_model-DKI_MK.nii.gz"
        if dki_marker.exists() and not force:
            logger.info("DKI outputs already exist. Use --force to overwrite.")
        else:
            results["dki"] = fit_dki(
                dwi_file=dwi_file,
                bval_file=bval_file,
                bvec_file=bvec_file,
                mask_file=mask_file,
                output_dir=deriv_dir,
                subject=subject,
                session=session,
            )

    # --- NODDI ---
    if do_noddi:
        noddi_marker = deriv_dir / f"{prefix}_model-NODDI_FICVF.nii.gz"
        if noddi_marker.exists() and not force:
            logger.info("NODDI outputs already exist. Use --force to overwrite.")
        else:
            if work_dir is None:
                work_dir = output_dir / "work" / "multishell" / subject / session
            try:
                results["noddi"] = fit_noddi(
                    dwi_file=dwi_file,
                    bval_file=bval_file,
                    bvec_file=bvec_file,
                    mask_file=mask_file,
                    output_dir=deriv_dir,
                    subject=subject,
                    session=session,
                    work_dir=work_dir,
                )
            except ImportError as exc:
                logger.warning(f"Skipping NODDI: {exc}")

    # --- QC ---
    dki_files = results.get("dki")
    noddi_files = results.get("noddi")
    if dki_files or noddi_files:
        try:
            from ..qc.dwi.multishell_qc import generate_multishell_qc_report

            qc_dir = output_dir / "qc" / subject / session / "dwi"
            results["qc"] = generate_multishell_qc_report(
                subject=subject,
                session=session,
                dki_files=dki_files,
                noddi_files=noddi_files,
                brain_mask=mask_file,
                output_dir=qc_dir,
            )
        except Exception:
            logger.exception("QC report generation failed")

    logger.info("")
    logger.info("Multi-shell fitting complete.")
    logger.info(f"  DKI:   {'done' if results['dki'] else 'skipped'}")
    logger.info(f"  NODDI: {'done' if results['noddi'] else 'skipped'}")
    logger.info(f"  QC:    {'done' if results['qc'] else 'skipped'}")

    return results
