#!/usr/bin/env python3
"""
MELODIC Group ICA and Dual Regression for Resting-State fMRI.

Implements group-level Independent Component Analysis (ICA) using FSL MELODIC,
followed by dual regression to obtain subject-specific spatial maps.

Intended workflow
-----------------
1. Warp each subject's cleaned BOLD to SIGMA space (uses the existing
   BOLD→template→SIGMA transform chain via ``warp_bold_to_sigma``).
2. Run group MELODIC on the SIGMA-space 4D timeseries to obtain IC maps.
3. Run dual regression to project IC maps back onto each subject's timeseries,
   yielding per-subject IC spatial maps for group statistics.

Input file convention
---------------------
``{derivatives}/{subject}/{session}/func/
    {subject}_{session}_space-SIGMA_desc-preproc_bold.nii.gz``

These are the fully preprocessed (ICA-denoised, aCompCor, smoothed,
bandpass-filtered) BOLD timeseries already warped to SIGMA atlas space.
If a subject's SIGMA-space file does not yet exist, use
``neurofaune.templates.registration.warp_bold_to_sigma`` to create it
before calling functions in this module.

Usage
-----
From the ``run_melodic_analysis.py`` script::

    bold_files = collect_bold_files(
        study_root=Path('/mnt/study'),
        cohort='p90',
        exclusions=exclusion_set,
        min_volumes=200,
    )
    melodic_results = run_group_melodic(
        subject_files=bold_files,
        output_dir=Path('/mnt/study/analysis/melodic/p90'),
        mask_file=sigma_mask,
        tr=0.5,
        n_components=20,
    )
    dr_results = run_dual_regression(
        group_ic_maps=melodic_results['component_maps'],
        subject_files=bold_files,
        output_dir=Path('/mnt/study/analysis/dual_regression/p90'),
    )
"""

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib

logger = logging.getLogger(__name__)

# SIGMA-space BOLD filename pattern produced by warp_bold_to_sigma
_SIGMA_BOLD_PATTERN = "{subject}_{session}_space-SIGMA_desc-preproc_bold.nii.gz"


def collect_bold_files(
    study_root: Path,
    cohort: Optional[str] = None,
    exclusions: Optional[set] = None,
    min_volumes: Optional[int] = None,
) -> List[Dict]:
    """
    Discover SIGMA-space preprocessed BOLD timeseries files.

    Parameters
    ----------
    study_root : Path
        Study root directory (contains ``derivatives/``).
    cohort : str, optional
        Restrict to one age cohort: ``'p30'``, ``'p60'``, or ``'p90'``.
        If None, all cohorts are returned.
    exclusions : set, optional
        Set of ``(subject, session)`` tuples to skip. Build from
        ``exclusions/*.csv`` via ``load_exclusions()``.
    min_volumes : int, optional
        Minimum number of volumes required. Sessions with fewer are skipped.

    Returns
    -------
    list of dict
        Each entry has keys:
        ``subject``, ``session``, ``cohort``, ``bold_file``, ``n_volumes``.
        Sorted by subject then session.
    """
    derivatives = study_root / "derivatives"
    sessions = []
    n_missing = 0
    n_excluded = 0
    n_short = 0

    for bold_file in sorted(
        derivatives.glob("sub-*/ses-*/func/*_space-SIGMA_desc-preproc_bold.nii.gz")
    ):
        func_dir = bold_file.parent
        subject = bold_file.name.split("_")[0]
        session = bold_file.name.split("_")[1]
        ses_cohort = session.replace("ses-", "")

        if cohort and ses_cohort != cohort:
            continue

        if exclusions and (subject, session) in exclusions:
            logger.debug("Excluding %s/%s (exclusion list)", subject, session)
            n_excluded += 1
            continue

        try:
            img = nib.load(bold_file)
            n_vols = img.shape[3] if len(img.shape) == 4 else 0
        except Exception as e:
            logger.warning("Cannot read %s: %s — skipping", bold_file.name, e)
            n_missing += 1
            continue

        if min_volumes and n_vols < min_volumes:
            logger.debug(
                "Excluding %s/%s: only %d volumes (< %d)",
                subject, session, n_vols, min_volumes,
            )
            n_short += 1
            continue

        sessions.append({
            "subject": subject,
            "session": session,
            "cohort": ses_cohort,
            "bold_file": bold_file,
            "n_volumes": n_vols,
        })

    label = f"cohort {cohort}" if cohort else "all cohorts"
    logger.info(
        "Found %d sessions (%s) | excluded: %d | short: %d | unreadable: %d",
        len(sessions), label, n_excluded, n_short, n_missing,
    )
    return sessions


def run_group_melodic(
    subject_files: List[Path],
    output_dir: Path,
    mask_file: Path,
    tr: float,
    n_components: Union[int, str] = "auto",
    approach: str = "concat",
    bg_image: Optional[Path] = None,
    sep_vn: bool = True,
    mm_thresh: float = 0.5,
    report: bool = True,
    n_threads: int = 1,
) -> Dict:
    """
    Run FSL MELODIC group ICA on SIGMA-space BOLD timeseries.

    Parameters
    ----------
    subject_files : list of Path
        SIGMA-space preprocessed BOLD images (4D, one per subject/session).
        All must share the same spatial dimensions and TR.
    output_dir : Path
        Directory for MELODIC outputs.
    mask_file : Path
        Brain mask in the same space as ``subject_files`` (SIGMA space).
    tr : float
        Repetition time in seconds.
    n_components : int or ``'auto'``
        Number of ICA components. ``'auto'`` lets MELODIC estimate via
        probabilistic PCA (recommended for exploratory analyses).
    approach : ``'concat'`` or ``'tica'``
        ICA approach. Temporal concatenation (``'concat'``) is standard
        for group ICA and substantially faster than tensor ICA.
    bg_image : Path, optional
        Background image for MELODIC HTML report.
    sep_vn : bool
        Separate variance normalisation per subject (default True).
    mm_thresh : float
        Mixture model threshold for component maps (default 0.5).
    report : bool
        Generate HTML QC report.
    n_threads : int
        Number of threads for MELODIC (default 1; MELODIC is largely
        single-threaded but some FSL routines can use OpenMP).

    Returns
    -------
    dict
        ``component_maps``, ``mixing_matrix``, ``ft_mixing_matrix``,
        ``subject_list``, ``n_components_actual``, ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MELODIC GROUP ICA")
    logger.info("=" * 70)
    logger.info("  Subjects   : %d", len(subject_files))
    logger.info("  Output     : %s", output_dir)
    logger.info("  Mask       : %s", mask_file.name)
    logger.info("  TR         : %.4f s", tr)
    logger.info("  Components : %s", n_components)
    logger.info("  Approach   : %s", approach)

    # Validate inputs
    ref_shape = None
    valid_files = []
    for f in subject_files:
        if not f.exists():
            logger.warning("Missing: %s — skipping", f)
            continue
        try:
            img = nib.load(f)
            if len(img.shape) != 4:
                logger.warning("Not 4D: %s — skipping", f.name)
                continue
            if ref_shape is None:
                ref_shape = img.shape[:3]
            elif img.shape[:3] != ref_shape:
                logger.warning(
                    "Dimension mismatch %s (%s vs %s) — skipping",
                    f.name, img.shape[:3], ref_shape,
                )
                continue
            valid_files.append(f)
        except Exception as e:
            logger.warning("Cannot read %s: %s — skipping", f, e)

    if len(valid_files) < 2:
        raise ValueError(
            f"Need at least 2 valid subjects for group ICA, found {len(valid_files)}"
        )
    logger.info("  Valid files: %d / %d", len(valid_files), len(subject_files))

    # Write subject list file
    subject_list_file = output_dir / "melodic_list.txt"
    with open(subject_list_file, "w") as fh:
        for f in valid_files:
            fh.write(f"{f}\n")
    logger.info("  Subject list: %s", subject_list_file)

    # Build melodic command
    cmd = [
        "melodic",
        "--in", str(subject_list_file),
        "--outdir", str(output_dir),
        "--mask", str(mask_file),
        "--tr", str(tr),
        "--approach", approach,
        "--nobet",       # data is already skull-stripped
        "--mmthresh", str(mm_thresh),
        "--Oall",        # save all output types
        "--out_all",
    ]
    if n_components != "auto":
        cmd += ["--dim", str(int(n_components))]
    if sep_vn:
        cmd += ["--sep_vn"]
    if report:
        cmd += ["--report"]
        if bg_image and bg_image.exists():
            cmd += ["--bgimage", str(bg_image)]
    if n_threads > 1:
        cmd += ["--numICs", "--num_threads", str(n_threads)]

    logger.info("Running: %s", " ".join(cmd))
    start = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (datetime.now() - start).total_seconds()

    # Save command log
    (output_dir / "melodic_cmd.log").write_text(
        f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    if result.returncode != 0:
        logger.error("MELODIC failed (exit %d)", result.returncode)
        logger.error("STDERR: %s", result.stderr[-2000:])
        raise RuntimeError(f"MELODIC failed with exit code {result.returncode}")

    logger.info("MELODIC completed in %.0f s", elapsed)

    # Collect outputs
    component_maps = output_dir / "melodic_IC.nii.gz"
    mixing_matrix = output_dir / "melodic_mix"
    ft_mixing_matrix = output_dir / "melodic_FTmix"

    # Determine actual number of components from output
    n_components_actual = None
    if component_maps.exists():
        img = nib.load(component_maps)
        n_components_actual = img.shape[3]
        logger.info("IC maps: %s (%d components)", component_maps.name, n_components_actual)

    outputs = {
        "component_maps": component_maps if component_maps.exists() else None,
        "mixing_matrix": mixing_matrix if mixing_matrix.exists() else None,
        "ft_mixing_matrix": ft_mixing_matrix if ft_mixing_matrix.exists() else None,
        "subject_list": subject_list_file,
        "n_components_actual": n_components_actual,
        "n_subjects": len(valid_files),
        "output_dir": str(output_dir),
        "elapsed_s": elapsed,
    }

    # Save summary
    summary = {k: str(v) if isinstance(v, Path) else v for k, v in outputs.items()}
    (output_dir / "melodic_summary.json").write_text(json.dumps(summary, indent=2))

    return outputs


def run_dual_regression(
    group_ic_maps: Path,
    subject_files: List[Path],
    output_dir: Path,
    design_mat: Optional[Path] = None,
    contrast_con: Optional[Path] = None,
    n_permutations: int = 5000,
    demean: bool = True,
    des_norm: bool = False,
) -> Dict:
    """
    Run FSL dual regression to obtain subject-specific IC spatial maps.

    Calls FSL's ``dual_regression`` tool, which:

    - **Stage 1**: regress group IC spatial maps against each subject's 4D
      timeseries → subject-specific IC time courses
    - **Stage 2**: regress subject-specific time courses against the 4D
      timeseries → subject-specific IC spatial maps
    - **Stage 3** (optional): run FSL ``randomise`` on the stage-2 maps
      using the provided design matrix and contrasts

    Parameters
    ----------
    group_ic_maps : Path
        ``melodic_IC.nii.gz`` from group MELODIC (4D, components in 4th dim).
    subject_files : list of Path
        SIGMA-space BOLD timeseries — must be the **same files** passed to
        MELODIC and in the same spatial space as ``group_ic_maps``.
    output_dir : Path
        Directory for dual regression outputs.
    design_mat : Path, optional
        FSL design matrix (``.mat``) for stage-3 group statistics. Pass
        ``None`` to skip randomise.
    contrast_con : Path, optional
        FSL contrast file (``.con``) paired with ``design_mat``.
    n_permutations : int
        Permutations for randomise (stage 3). Only used when ``design_mat``
        is provided. Default 5000.
    demean : bool
        Demean IC time courses before stage-2 regression (default True).
    des_norm : bool
        Variance-normalise the design matrix (default False).

    Returns
    -------
    dict
        ``stage1_dirs``, ``stage2_files``, ``stage3_files``,
        ``n_subjects``, ``n_components``, ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_ic_maps = Path(group_ic_maps)

    logger.info("=" * 70)
    logger.info("DUAL REGRESSION")
    logger.info("=" * 70)
    logger.info("  Group IC maps : %s", group_ic_maps.name)
    logger.info("  Subjects      : %d", len(subject_files))
    logger.info("  Output        : %s", output_dir)
    if design_mat:
        logger.info("  Design matrix : %s", design_mat.name)
        logger.info("  Permutations  : %d", n_permutations)

    # Validate and filter subject files
    if not group_ic_maps.exists():
        raise FileNotFoundError(f"Group IC maps not found: {group_ic_maps}")

    ic_img = nib.load(group_ic_maps)
    if len(ic_img.shape) != 4:
        raise ValueError(f"Group IC maps must be 4D, got shape {ic_img.shape}")
    n_components = ic_img.shape[3]
    ic_spatial_dims = ic_img.shape[:3]
    logger.info("  Components    : %d", n_components)

    valid_files = []
    for f in subject_files:
        if not f.exists():
            logger.warning("Missing: %s — skipping", f)
            continue
        try:
            img = nib.load(f)
            if img.shape[:3] != ic_spatial_dims:
                logger.warning(
                    "Dimension mismatch %s (%s vs IC %s) — skipping",
                    f.name, img.shape[:3], ic_spatial_dims,
                )
                continue
            valid_files.append(f)
        except Exception as e:
            logger.warning("Cannot read %s: %s — skipping", f, e)

    if len(valid_files) < 2:
        raise ValueError(
            f"Need at least 2 valid subjects for dual regression, found {len(valid_files)}"
        )
    logger.info("  Valid subjects: %d / %d", len(valid_files), len(subject_files))

    # Write subject list (dual_regression reads from the command line, but
    # we save it for provenance)
    subject_list_file = output_dir / "subject_list.txt"
    with open(subject_list_file, "w") as fh:
        for f in valid_files:
            fh.write(f"{f.resolve()}\n")

    # Build dual_regression command
    # Signature: dual_regression <IC_maps> <des_norm> <design.mat> <design.con> <n_perm> <output_dir> <subjects...>
    # When no design matrix: replace <design.mat> <design.con> with a single -1
    if design_mat and contrast_con:
        design_args = [str(design_mat.resolve()), str(contrast_con.resolve())]
        perm_arg = str(n_permutations)
    else:
        design_args = ["-1"]
        perm_arg = "0"   # skip randomise when no design matrix

    cmd = [
        "dual_regression",
        str(group_ic_maps.resolve()),
        "1" if des_norm else "0",
        *design_args,
        perm_arg,
        str(output_dir.resolve()),
        *(str(f.resolve()) for f in valid_files),
    ]

    logger.info("Running dual_regression...")
    start = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (datetime.now() - start).total_seconds()

    # Save command log
    (output_dir / "dual_regression_cmd.log").write_text(
        f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    if result.returncode != 0:
        logger.error("dual_regression failed (exit %d)", result.returncode)
        logger.error("STDERR: %s", result.stderr[-2000:])
        raise RuntimeError(
            f"dual_regression failed with exit code {result.returncode}"
        )

    logger.info("Dual regression completed in %.0f s", elapsed)

    # Collect outputs
    stage1_dirs = sorted(output_dir.glob("dr_stage1_subject*"))
    stage2_files = sorted(output_dir.glob("dr_stage2_subject*.nii.gz"))
    stage3_files: Dict[str, List[str]] = {}
    if design_mat:
        for ic_idx in range(n_components):
            ic_stat_files = sorted(
                output_dir.glob(f"dr_stage3_ic{ic_idx:04d}_*.nii.gz")
            )
            if ic_stat_files:
                stage3_files[f"ic_{ic_idx:04d}"] = [str(f) for f in ic_stat_files]

    logger.info(
        "  Stage 1 (time courses): %d dirs | Stage 2 (spatial maps): %d files | "
        "Stage 3 (statistics): %d components",
        len(stage1_dirs), len(stage2_files), len(stage3_files),
    )

    outputs = {
        "stage1_dirs": [str(d) for d in stage1_dirs],
        "stage2_files": [str(f) for f in stage2_files],
        "stage3_files": stage3_files,
        "n_subjects": len(valid_files),
        "n_components": n_components,
        "output_dir": str(output_dir),
        "subject_list": str(subject_list_file),
        "elapsed_s": elapsed,
    }

    (output_dir / "dual_regression_results.json").write_text(
        json.dumps(outputs, indent=2)
    )

    return outputs
