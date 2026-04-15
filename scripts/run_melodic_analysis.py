#!/usr/bin/env python3
"""
Group MELODIC ICA + Dual Regression for resting-state fMRI.

Discovers SIGMA-space preprocessed BOLD timeseries, warps any missing ones,
then runs FSL MELODIC group ICA followed by dual regression.

Subjects whose SIGMA-space BOLD does not yet exist are warped on-the-fly
using the existing BOLD→template→SIGMA transform chain. Warping is
parallelised across subjects via --n-workers.

Typical usage
-------------
  # Per-cohort (run separately for p30, p60, p90):
  uv run python scripts/run_melodic_analysis.py \\
      --study-root /mnt/arborea/bpa-rat \\
      --cohort p90 \\
      --output-dir /mnt/arborea/bpa-rat/analysis/melodic \\
      --exclusion-csv /mnt/arborea/bpa-rat/exclusions/func_exclusions.csv \\
      --min-volumes 200 \\
      --n-components 20 \\
      --n-workers 8

  # Pooled (all cohorts, subjects already in SIGMA space):
  uv run python scripts/run_melodic_analysis.py \\
      --study-root /mnt/arborea/bpa-rat \\
      --output-dir /mnt/arborea/bpa-rat/analysis/melodic \\
      --exclusion-csv /mnt/arborea/bpa-rat/exclusions/func_exclusions.csv \\
      --min-volumes 200 \\
      --n-components 20

Notes
-----
- Input BOLD: ``space-SIGMA_desc-preproc_bold.nii.gz`` (ICA-denoised,
  aCompCor, smoothed, bandpass-filtered, in SIGMA atlas space).
- Mask: SIGMA brain mask from ``atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz``.
- MELODIC output: ``{output_dir}/{cohort_label}/``
- Dual regression output: ``{output_dir}/{cohort_label}/dual_regression/``
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.analysis.func.melodic import (
    collect_bold_files,
    run_dual_regression,
    run_group_melodic,
)
from neurofaune.templates.registration import warp_bold_to_sigma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_exclusions(exclusion_csv: Path) -> set:
    df = pd.read_csv(exclusion_csv)
    return set(zip(df["subject"], df["session"]))


def _warp_one_session(args):
    """Worker function for parallel warping (must be module-level for pickle)."""
    subject, session, cohort, preproc_bold, transforms, sigma_template, sigma_mask, work_dir = args
    func_dir = preproc_bold.parent
    try:
        result = warp_bold_to_sigma(
            input_files={"desc-preproc_bold": preproc_bold},
            transforms=transforms,
            sigma_template=sigma_template,
            sigma_brain_mask=sigma_mask,
            output_dir=func_dir,
            subject=subject,
            session=session,
            low_memory=True,
            n_threads=2,
            work_dir=work_dir,
        )
        sigma_bold = result.get("desc-preproc_bold")
        return subject, session, sigma_bold, None
    except Exception as e:
        return subject, session, None, str(e)


def ensure_sigma_bold(
    sessions_needing_warp: list,
    study_root: Path,
    n_workers: int,
) -> int:
    """
    Warp native-space BOLD to SIGMA space for sessions that don't have it yet.

    Returns the number of successfully warped sessions.
    """
    transforms_root = study_root / "transforms"
    templates_root = study_root / "templates"
    sigma_template = study_root / "atlas" / "SIGMA_study_space" / "SIGMA_InVivo_Brain.nii.gz"
    sigma_mask = study_root / "atlas" / "SIGMA_study_space" / "SIGMA_InVivo_Brain_Mask.nii.gz"
    work_dir = study_root / "work" / "melodic_warp"
    work_dir.mkdir(parents=True, exist_ok=True)

    warp_args = []
    for sess in sessions_needing_warp:
        subject, session, cohort = sess["subject"], sess["session"], sess["cohort"]
        subj_transforms = transforms_root / subject / session
        tpl_transforms = templates_root / "anat" / cohort / "transforms"

        bold_to_template = subj_transforms / "BOLD_to_template_0GenericAffine.mat"
        tpl_to_sigma_affine = tpl_transforms / "tpl-to-SIGMA_0GenericAffine.mat"
        tpl_to_sigma_warp = tpl_transforms / "tpl-to-SIGMA_1Warp.nii.gz"

        if not bold_to_template.exists():
            logger.warning(
                "%s/%s: BOLD→template transform missing (%s) — skipping warp",
                subject, session, bold_to_template.name,
            )
            continue
        if not tpl_to_sigma_warp.exists():
            logger.warning(
                "%s/%s: template→SIGMA warp missing (%s) — skipping warp",
                subject, session, tpl_to_sigma_warp.name,
            )
            continue

        preproc_bold = (
            study_root / "derivatives" / subject / session / "func"
            / f"{subject}_{session}_desc-preproc_bold.nii.gz"
        )
        if not preproc_bold.exists():
            logger.warning(
                "%s/%s: native-space preproc_bold missing — skipping warp",
                subject, session,
            )
            continue

        # ANTs transform chain: outermost first (applied last-first)
        transforms = [
            str(tpl_to_sigma_warp),
            str(tpl_to_sigma_affine),
            str(bold_to_template),
        ]
        warp_args.append((
            subject, session, cohort, preproc_bold,
            transforms, sigma_template, sigma_mask, work_dir,
        ))

    if not warp_args:
        return 0

    logger.info("Warping %d sessions to SIGMA space (%d workers)...", len(warp_args), n_workers)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_warp_one_session, a): a for a in warp_args}
        for future in as_completed(futures):
            subject, session, sigma_bold, err = future.result()
            if err:
                logger.error("  %s/%s: warp FAILED — %s", subject, session, err)
            elif sigma_bold and sigma_bold.exists():
                logger.info("  %s/%s: warped → %s", subject, session, sigma_bold.name)
                n_ok += 1
            else:
                logger.warning("  %s/%s: warp returned no output", subject, session)

    return n_ok


def main():
    parser = argparse.ArgumentParser(
        description="Run MELODIC group ICA + dual regression on resting-state fMRI data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--study-root", type=Path, required=True,
        help="Study root directory (contains derivatives/, atlas/, transforms/)",
    )
    parser.add_argument(
        "--cohort", choices=["p30", "p60", "p90"],
        help="Age cohort to process. If omitted, pool all cohorts.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Base output directory. Results go to {output_dir}/{cohort_label}/melodic/ "
             "and {output_dir}/{cohort_label}/dual_regression/",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV with subject,session,reason,date_added columns for exclusions.",
    )
    parser.add_argument(
        "--min-volumes", type=int, default=200,
        help="Minimum number of BOLD volumes required (default: 200).",
    )
    parser.add_argument(
        "--n-components", default="auto",
        help="Number of ICA components (integer) or 'auto' for MELODIC estimation "
             "(default: auto).",
    )
    parser.add_argument(
        "--approach", choices=["concat", "tica"], default="concat",
        help="MELODIC approach: temporal concatenation (default) or tensor ICA.",
    )
    parser.add_argument(
        "--no-dual-regression", action="store_true",
        help="Run MELODIC only, skip dual regression.",
    )
    parser.add_argument(
        "--design-mat", type=Path, default=None,
        help="FSL design matrix (.mat) for dual regression stage 3 (optional).",
    )
    parser.add_argument(
        "--contrast-con", type=Path, default=None,
        help="FSL contrast file (.con) paired with --design-mat.",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Permutations for randomise in dual regression stage 3 (default: 5000).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Parallel workers for BOLD→SIGMA warping (default: 4).",
    )
    parser.add_argument(
        "--skip-warp", action="store_true",
        help="Skip warping step (assume all SIGMA-space BOLDs already exist).",
    )

    args = parser.parse_args()

    study_root = args.study_root
    cohort_label = args.cohort if args.cohort else "pooled"
    melodic_dir = args.output_dir / cohort_label
    dr_dir = args.output_dir / cohort_label / "dual_regression"

    # SIGMA atlas paths
    sigma_mask = (
        study_root / "atlas" / "SIGMA_study_space" / "SIGMA_InVivo_Brain_Mask.nii.gz"
    )
    sigma_template = (
        study_root / "atlas" / "SIGMA_study_space" / "SIGMA_InVivo_Brain.nii.gz"
    )

    if not sigma_mask.exists():
        logger.error("SIGMA brain mask not found: %s", sigma_mask)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("MELODIC + DUAL REGRESSION — %s", cohort_label.upper())
    logger.info("=" * 70)
    logger.info("  Study root  : %s", study_root)
    logger.info("  Cohort      : %s", cohort_label)
    logger.info("  MELODIC out : %s", melodic_dir)
    logger.info("  DR out      : %s", dr_dir)

    # Load exclusions
    exclusions = None
    if args.exclusion_csv:
        exclusions = load_exclusions(args.exclusion_csv)
        logger.info("  Exclusions  : %d sessions", len(exclusions))

    # Discover SIGMA-space BOLD files
    sessions = collect_bold_files(
        study_root=study_root,
        cohort=args.cohort,
        exclusions=exclusions,
        min_volumes=args.min_volumes,
    )

    if not sessions:
        logger.error("No SIGMA-space BOLD files found — have you run warp_bold_to_sigma?")
        logger.error(
            "Check that derivatives contain: *_space-SIGMA_desc-preproc_bold.nii.gz"
        )
        sys.exit(1)

    logger.info("")
    logger.info("[Phase 1] Checking SIGMA-space BOLD coverage...")

    # Find sessions missing SIGMA-space BOLD and warp them if needed
    if not args.skip_warp:
        # Identify sessions that have native-space preproc_bold but no SIGMA-space BOLD
        existing_sigma = {
            (s["subject"], s["session"]) for s in sessions
        }
        # Look for any sessions with native-space BOLD not yet warped
        derivatives = study_root / "derivatives"
        sessions_needing_warp = []
        for native_bold in sorted(
            derivatives.glob("sub-*/ses-*/func/*_desc-preproc_bold.nii.gz")
        ):
            if "space-" in native_bold.name:
                continue
            subject = native_bold.name.split("_")[0]
            session = native_bold.name.split("_")[1]
            ses_cohort = session.replace("ses-", "")

            if args.cohort and ses_cohort != args.cohort:
                continue
            if exclusions and (subject, session) in exclusions:
                continue
            if (subject, session) in existing_sigma:
                continue  # already have SIGMA-space BOLD

            sessions_needing_warp.append({
                "subject": subject,
                "session": session,
                "cohort": ses_cohort,
            })

        if sessions_needing_warp:
            logger.info(
                "  %d sessions need warping to SIGMA space...", len(sessions_needing_warp)
            )
            n_warped = ensure_sigma_bold(sessions_needing_warp, study_root, args.n_workers)
            logger.info("  Warped %d sessions successfully", n_warped)

            # Re-discover after warping
            sessions = collect_bold_files(
                study_root=study_root,
                cohort=args.cohort,
                exclusions=exclusions,
                min_volumes=args.min_volumes,
            )
        else:
            logger.info("  All sessions already have SIGMA-space BOLD.")

    bold_files = [s["bold_file"] for s in sessions]
    logger.info("  Total sessions for MELODIC: %d", len(bold_files))

    # Get TR from first file's native-space sidecar
    tr = None
    for sess in sessions:
        sidecar = (
            study_root / "derivatives" / sess["subject"] / sess["session"] / "func"
            / f"{sess['subject']}_{sess['session']}_desc-preproc_bold.json"
        )
        if sidecar.exists():
            import json
            meta = json.loads(sidecar.read_text())
            tr = meta.get("tr")
            if tr:
                logger.info("  TR: %.4f s (from %s)", tr, sidecar.name)
                break
    if tr is None:
        logger.error("Could not determine TR from preproc_bold.json sidecars")
        sys.exit(1)

    # Parse n_components
    try:
        n_components = int(args.n_components)
    except ValueError:
        n_components = "auto"

    # Phase 2: MELODIC
    logger.info("")
    logger.info("[Phase 2] Running MELODIC group ICA...")
    melodic_results = run_group_melodic(
        subject_files=bold_files,
        output_dir=melodic_dir,
        mask_file=sigma_mask,
        tr=tr,
        n_components=n_components,
        approach=args.approach,
        bg_image=sigma_template if sigma_template.exists() else None,
    )

    component_maps = melodic_results.get("component_maps")
    if not component_maps or not component_maps.exists():
        logger.error("MELODIC did not produce component maps — aborting")
        sys.exit(1)

    logger.info(
        "MELODIC complete: %d components → %s",
        melodic_results["n_components_actual"], component_maps,
    )

    # Phase 3: Dual Regression
    if args.no_dual_regression:
        logger.info("")
        logger.info("Skipping dual regression (--no-dual-regression).")
    else:
        logger.info("")
        logger.info("[Phase 3] Running dual regression...")
        dr_results = run_dual_regression(
            group_ic_maps=component_maps,
            subject_files=bold_files,
            output_dir=dr_dir,
            design_mat=args.design_mat,
            contrast_con=args.contrast_con,
            n_permutations=args.n_permutations,
        )

        logger.info(
            "Dual regression complete: %d subjects, %d stage-2 maps",
            dr_results["n_subjects"], len(dr_results["stage2_files"]),
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE — %s", cohort_label.upper())
    logger.info("  MELODIC   : %s", melodic_dir)
    if not args.no_dual_regression:
        logger.info("  DR        : %s", dr_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
