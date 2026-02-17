#!/usr/bin/env python3
"""
Batch resting-state fMRI analysis: fALFF, ReHo, SIGMA warping, and FC.

Discovers all sessions with preprocessed BOLD data and computes:
- fALFF (fractional ALFF) from unfiltered regressed data
- ReHo (Regional Homogeneity) from bandpass-filtered data
- SIGMA-space warping of BOLD, fALFF, ReHo (and z-scored versions)
- ROI-to-ROI functional connectivity matrices

For sessions missing unfiltered (regressed) data, reconstructs it from
bold_smooth.nii.gz + confounds TSV via OLS regression.
"""

import argparse
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.analysis.func import (
    compute_falff_map,
    compute_falff_zscore,
    compute_fc_matrix,
    compute_reho_map,
    compute_reho_zscore,
    extract_roi_timeseries,
    save_fc_matrix,
)
from neurofaune.config import get_config_value, load_config
from neurofaune.preprocess.workflows.func_preprocess import register_bold_to_template
from neurofaune.templates.registration import warp_bold_to_sigma


def find_preprocessed_sessions(study_root: Path) -> list:
    """
    Find all sessions with preprocessed BOLD data.

    Enriches each session dict with cohort, transform paths, and
    SIGMA transform availability for the SIGMA-warping phase.
    Also includes paths needed for on-the-fly BOLD->Template
    registration if the transform doesn't exist yet.

    Returns
    -------
    list of dict
        Session metadata including transform chain info.
    """
    derivatives = study_root / "derivatives"
    transforms_root = study_root / "transforms"
    templates_root = study_root / "templates"
    sessions = []

    for preproc_bold in sorted(derivatives.glob("sub-*/ses-*/func/*_desc-preproc_bold.nii.gz")):
        func_dir = preproc_bold.parent
        parts = preproc_bold.name.split("_")
        subject = parts[0]
        session = parts[1]
        cohort = session.replace("ses-", "")

        mask = func_dir / f"{subject}_{session}_desc-brain_mask.nii.gz"
        confounds = func_dir / f"{subject}_{session}_desc-confounds_timeseries.tsv"
        acompcor = func_dir / f"{subject}_{session}_desc-acompcor_timeseries.tsv"
        sidecar = func_dir / f"{subject}_{session}_desc-preproc_bold.json"
        work_dir = study_root / "work" / subject / session / "func_preproc"

        if not mask.exists() or not confounds.exists():
            print(f"  Warning: Skipping {subject}/{session} — missing mask or confounds")
            continue

        # Resolve transform paths for SIGMA warping (2-hop: BOLD -> Template -> SIGMA)
        subj_transforms = transforms_root / subject / session

        # BOLD -> Template (rigid, from register_bold_to_template)
        bold_to_template = subj_transforms / "BOLD_to_template_0GenericAffine.mat"

        # Template -> SIGMA (per-cohort, pre-computed)
        tpl_transforms = templates_root / "anat" / cohort / "transforms"
        tpl_to_sigma_affine = tpl_transforms / "tpl-to-SIGMA_0GenericAffine.mat"
        tpl_to_sigma_warp = tpl_transforms / "tpl-to-SIGMA_1Warp.nii.gz"

        # Cohort template (needed for on-the-fly registration)
        template_file = templates_root / "anat" / cohort / f"tpl-BPARat_{cohort}_T2w.nii.gz"

        # Motion-corrected BOLD (needed to compute temporal mean for registration)
        mcf_file = (
            study_root / "work" / subject / session
            / "func_preproc" / "motion_correction" / "bold_mcf.nii.gz"
        )

        # Can we reach SIGMA space?
        has_tpl_to_sigma = tpl_to_sigma_affine.exists()
        can_register = mcf_file.exists() and template_file.exists()

        sessions.append({
            "subject": subject,
            "session": session,
            "cohort": cohort,
            "derivatives_dir": func_dir,
            "preproc_bold": preproc_bold,
            "mask": mask,
            "confounds": confounds,
            "acompcor": acompcor if acompcor.exists() else None,
            "sidecar_json": sidecar if sidecar.exists() else None,
            "work_dir": work_dir,
            # Transform paths (raw, for building chain after registration)
            "study_root": study_root,
            "bold_to_template": bold_to_template,
            "tpl_to_sigma_affine": tpl_to_sigma_affine,
            "tpl_to_sigma_warp": tpl_to_sigma_warp if tpl_to_sigma_warp.exists() else None,
            "template_file": template_file if template_file.exists() else None,
            "mcf_file": mcf_file if mcf_file.exists() else None,
            "has_sigma_transforms": bold_to_template.exists() and has_tpl_to_sigma,
            "can_register": can_register and has_tpl_to_sigma,
        })

    return sessions


def get_tr(session_info: dict) -> float:
    """Read TR from sidecar JSON, fallback to 0.5s."""
    if session_info["sidecar_json"] and session_info["sidecar_json"].exists():
        with open(session_info["sidecar_json"]) as f:
            meta = json.load(f)
        return float(meta.get("tr", 0.5))
    return 0.5


def reconstruct_regressed(session_info: dict) -> Path:
    """
    Reconstruct nuisance-regressed (unfiltered) BOLD from bold_smooth + confounds.

    Replicates the OLS regression from func_preprocess.py (Step 9) without
    the subsequent bandpass filtering.

    Returns
    -------
    Path
        Path to the regressed BOLD file in derivatives.
    """
    subject = session_info["subject"]
    session = session_info["session"]
    deriv_dir = session_info["derivatives_dir"]
    work_dir = session_info["work_dir"]

    bold_smooth = work_dir / f"{subject}_{session}_bold_smooth.nii.gz"
    if not bold_smooth.exists():
        raise FileNotFoundError(
            f"Cannot reconstruct regressed data: {bold_smooth} not found"
        )

    print(f"    Reconstructing regressed BOLD from {bold_smooth.name}...")

    # Load smoothed BOLD and mask
    bold_img = nib.load(bold_smooth)
    bold_data = bold_img.get_fdata()
    mask_data = nib.load(session_info["mask"]).get_fdata().astype(bool)
    n_timepoints = bold_data.shape[3]

    # Build design matrix: 24 motion regressors (+ aCompCor if available)
    confounds_matrix = np.loadtxt(session_info["confounds"], skiprows=1)

    if session_info["acompcor"] is not None:
        acompcor_matrix = np.loadtxt(session_info["acompcor"], skiprows=1)
        confounds_matrix = np.column_stack([confounds_matrix, acompcor_matrix])

    design = np.column_stack([confounds_matrix[:n_timepoints], np.ones(n_timepoints)])

    # Voxelwise OLS regression within mask, keep residuals + intercept
    clean_data = bold_data.copy()
    voxels = np.where(mask_data)
    timeseries = bold_data[voxels[0], voxels[1], voxels[2], :]  # (n_voxels, n_timepoints)

    betas = np.linalg.lstsq(design, timeseries.T, rcond=None)[0]
    residuals = timeseries.T - design @ betas  # (n_timepoints, n_voxels)
    residuals = residuals + betas[-1, :]  # add intercept back
    clean_data[voxels[0], voxels[1], voxels[2], :] = residuals.T

    # Save to derivatives
    regressed_file = deriv_dir / f"{subject}_{session}_desc-regressed_bold.nii.gz"
    nib.save(
        nib.Nifti1Image(clean_data.astype(np.float32), bold_img.affine, bold_img.header),
        regressed_file,
    )
    print(f"    Saved reconstructed regressed BOLD: {regressed_file.name}")

    return regressed_file


def ensure_regressed_bold(session_info: dict) -> Path:
    """
    Get or reconstruct the unfiltered regressed BOLD for fALFF.

    Checks (in order):
    1. desc-regressed_bold in derivatives
    2. bold_regressed in work dir (copy to derivatives)
    3. Reconstruct from bold_smooth + confounds
    """
    subject = session_info["subject"]
    session = session_info["session"]
    deriv_dir = session_info["derivatives_dir"]
    work_dir = session_info["work_dir"]

    # 1. Already in derivatives
    regressed_deriv = deriv_dir / f"{subject}_{session}_desc-regressed_bold.nii.gz"
    if regressed_deriv.exists():
        return regressed_deriv

    # 2. In work dir — copy to derivatives
    regressed_work = work_dir / f"{subject}_{session}_bold_regressed.nii.gz"
    if regressed_work.exists():
        shutil.copy(str(regressed_work), str(regressed_deriv))
        print(f"    Copied regressed BOLD from work to derivatives")
        return regressed_deriv

    # 3. Reconstruct from smooth + confounds
    return reconstruct_regressed(session_info)


def process_session(session_info: dict, analysis: str, config: dict,
                    force: bool = False, skip_sigma: bool = False,
                    skip_fc: bool = False) -> dict:
    """
    Run resting-state analysis for a single session.

    Parameters
    ----------
    session_info : dict
        Session information from find_preprocessed_sessions.
    analysis : str
        'falff', 'reho', or 'both'.
    config : dict
        Loaded config.
    force : bool
        Recompute even if outputs exist.
    skip_sigma : bool
        Skip SIGMA-space warping.
    skip_fc : bool
        Skip functional connectivity matrices.

    Returns
    -------
    dict
        Result with status, paths, and statistics.
    """
    subject = session_info["subject"]
    session = session_info["session"]
    key = f"{subject}_{session}"
    deriv_dir = session_info["derivatives_dir"]

    result = {
        "status": "success",
        "key": key,
        "subject": subject,
        "session": session,
        "outputs": {},
        "statistics": {},
    }

    try:
        # Read config parameters
        low_freq = get_config_value(config, "functional.analysis.falff.low_freq", default=0.01)
        high_freq = get_config_value(config, "functional.analysis.falff.high_freq", default=0.08)
        neighborhood = get_config_value(config, "functional.analysis.reho.neighborhood", default=27)

        tr = get_tr(session_info)

        print(f"\n{'=' * 60}")
        print(f"Session: {key} (TR={tr}s)")
        print(f"{'=' * 60}")

        # ---- fALFF ----
        if analysis in ("falff", "both"):
            falff_output = deriv_dir / f"{subject}_{session}_desc-fALFF_bold.nii.gz"

            if falff_output.exists() and not force:
                print(f"  fALFF already exists, skipping (use --force to recompute)")
                result["outputs"]["falff"] = "skipped"
            else:
                regressed_bold = ensure_regressed_bold(session_info)

                falff_result = compute_falff_map(
                    func_file=regressed_bold,
                    mask_file=session_info["mask"],
                    output_dir=deriv_dir,
                    subject=subject,
                    session=session,
                    tr=tr,
                    low_freq=low_freq,
                    high_freq=high_freq,
                )

                zscore_result = compute_falff_zscore(
                    alff_file=falff_result["alff_file"],
                    falff_file=falff_result["falff_file"],
                    mask_file=session_info["mask"],
                    output_dir=deriv_dir,
                    subject=subject,
                    session=session,
                )

                result["outputs"]["falff"] = {
                    "alff": str(falff_result["alff_file"]),
                    "falff": str(falff_result["falff_file"]),
                    "alff_zscore": str(zscore_result["alff_zscore_file"]),
                    "falff_zscore": str(zscore_result["falff_zscore_file"]),
                }
                result["statistics"]["falff"] = falff_result["statistics"]

        # ---- ReHo ----
        if analysis in ("reho", "both"):
            reho_output = deriv_dir / f"{subject}_{session}_desc-ReHo_bold.nii.gz"

            if reho_output.exists() and not force:
                print(f"  ReHo already exists, skipping (use --force to recompute)")
                result["outputs"]["reho"] = "skipped"
            else:
                reho_result = compute_reho_map(
                    func_file=session_info["preproc_bold"],
                    mask_file=session_info["mask"],
                    output_dir=deriv_dir,
                    subject=subject,
                    session=session,
                    neighborhood=neighborhood,
                )

                zscore_result = compute_reho_zscore(
                    reho_file=reho_result["reho_file"],
                    mask_file=session_info["mask"],
                    output_dir=deriv_dir,
                    subject=subject,
                    session=session,
                )

                result["outputs"]["reho"] = {
                    "reho": str(reho_result["reho_file"]),
                    "reho_zscore": str(zscore_result["reho_zscore_file"]),
                }
                result["statistics"]["reho"] = reho_result["statistics"]

        # ---- BOLD-to-Template Registration (on-the-fly if needed) ----
        has_sigma = session_info["has_sigma_transforms"]

        if not skip_sigma and not has_sigma and session_info["can_register"]:
            print(f"\n  BOLD->Template transform missing — computing registration...")
            study_root = session_info["study_root"]
            work_reg_dir = study_root / "work" / subject / session / "bold_registration"
            work_reg_dir.mkdir(parents=True, exist_ok=True)

            # Compute temporal mean of motion-corrected BOLD
            mean_bold_file = work_reg_dir / f"{subject}_{session}_mean_mcf_brain.nii.gz"
            if not mean_bold_file.exists():
                mcf_img = nib.load(session_info["mcf_file"])
                mean_data = np.mean(mcf_img.get_fdata(), axis=3)
                mask_data = nib.load(session_info["mask"]).get_fdata() > 0
                mean_data *= mask_data
                nib.save(
                    nib.Nifti1Image(mean_data.astype(np.float32),
                                    mcf_img.affine, mcf_img.header),
                    mean_bold_file,
                )

            reg_results = register_bold_to_template(
                bold_ref_file=mean_bold_file,
                template_file=session_info["template_file"],
                output_dir=study_root,
                subject=subject,
                session=session,
                work_dir=work_reg_dir,
            )

            # Update session info with newly computed transform
            bold_to_template = session_info["bold_to_template"]
            if bold_to_template.exists():
                has_sigma = True
                result["outputs"]["registration"] = str(bold_to_template)
                print(f"  Registration complete: {bold_to_template.name}")
            else:
                print(f"  WARNING: registration ran but transform not found at {bold_to_template}")

        # ---- Warp to SIGMA ----
        if not skip_sigma and has_sigma:
            sigma_template = Path(get_config_value(config, "atlas.study_space.template"))
            sigma_brain_mask = Path(get_config_value(config, "atlas.study_space.brain_mask"))

            # Build transform chain from current files on disk
            sigma_transforms = []
            if session_info["tpl_to_sigma_warp"] is not None:
                sigma_transforms.append(str(session_info["tpl_to_sigma_warp"]))
            sigma_transforms.append(str(session_info["tpl_to_sigma_affine"]))
            sigma_transforms.append(str(session_info["bold_to_template"]))

            # Collect native-space maps to warp
            input_files = {}
            desc_map = {
                "desc-preproc_bold": f"{subject}_{session}_desc-preproc_bold.nii.gz",
                "desc-fALFF_bold": f"{subject}_{session}_desc-fALFF_bold.nii.gz",
                "desc-ALFF_bold": f"{subject}_{session}_desc-ALFF_bold.nii.gz",
                "desc-ReHo_bold": f"{subject}_{session}_desc-ReHo_bold.nii.gz",
                "desc-fALFFzscore_bold": f"{subject}_{session}_desc-fALFFzscore_bold.nii.gz",
                "desc-ALFFzscore_bold": f"{subject}_{session}_desc-ALFFzscore_bold.nii.gz",
                "desc-ReHozscore_bold": f"{subject}_{session}_desc-ReHozscore_bold.nii.gz",
            }

            for desc, filename in desc_map.items():
                p = deriv_dir / filename
                if p.exists():
                    input_files[desc] = p

            if input_files:
                sigma_outputs = warp_bold_to_sigma(
                    input_files=input_files,
                    transforms=sigma_transforms,
                    sigma_template=sigma_template,
                    sigma_brain_mask=sigma_brain_mask,
                    output_dir=deriv_dir,
                    subject=subject,
                    session=session,
                )
                result["outputs"]["sigma"] = {
                    k: str(v) for k, v in sigma_outputs.items()
                }
            else:
                print("  No native-space maps found for SIGMA warping")
        elif not skip_sigma and not has_sigma:
            if not session_info["can_register"]:
                print(f"  Skipping SIGMA warp — missing mcf or cohort template")
            else:
                print(f"  Skipping SIGMA warp — registration failed")
            result["outputs"]["sigma"] = "no_transforms"

        # ---- Functional Connectivity ----
        if not skip_fc and not skip_sigma and has_sigma:
            sigma_bold = deriv_dir / f"{subject}_{session}_space-SIGMA_desc-preproc_bold.nii.gz"

            if sigma_bold.exists():
                atlas_path = Path(get_config_value(config, "atlas.study_space.parcellation"))
                sigma_brain_mask = Path(get_config_value(config, "atlas.study_space.brain_mask"))

                fc_base = deriv_dir / f"{subject}_{session}_space-SIGMA_desc-fc_bold"
                fc_npy = fc_base.with_suffix(".npy")

                if fc_npy.exists() and not force:
                    print(f"  FC matrix already exists, skipping")
                    result["outputs"]["fc"] = "skipped"
                else:
                    print(f"\n  Computing functional connectivity...")
                    ts, roi_labels = extract_roi_timeseries(
                        bold_4d=sigma_bold,
                        atlas=atlas_path,
                        mask=sigma_brain_mask,
                    )
                    fc_z = compute_fc_matrix(ts)
                    npy_path, tsv_path = save_fc_matrix(fc_z, roi_labels, fc_base)

                    result["outputs"]["fc"] = {
                        "matrix": str(npy_path),
                        "labels": str(tsv_path),
                        "n_rois": int(len(roi_labels)),
                    }
            else:
                print(f"  Skipping FC — no SIGMA-space 4D BOLD")
                result["outputs"]["fc"] = "no_sigma_bold"

        # Save per-session metadata JSON
        metadata_file = deriv_dir / f"{subject}_{session}_desc-resting_analysis.json"
        metadata = {
            "subject": subject,
            "session": session,
            "analysis_date": datetime.now().isoformat(),
            "tr": tr,
            "parameters": {
                "falff": {"low_freq": low_freq, "high_freq": high_freq},
                "reho": {"neighborhood": neighborhood},
            },
            "statistics": result["statistics"],
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"  FAILED: {e}")

    return result


def _process_session_wrapper(args):
    """Wrapper for ProcessPoolExecutor (must be top-level picklable)."""
    session_info, analysis, config_path, force, skip_sigma, skip_fc = args
    config = load_config(config_path)
    return process_session(session_info, analysis, config, force, skip_sigma, skip_fc)


def main():
    parser = argparse.ArgumentParser(
        description="Batch resting-state fMRI analysis (fALFF + ReHo + SIGMA warp + FC)"
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("/mnt/arborea/bpa-rat"),
        help="Study root directory (default: /mnt/arborea/bpa-rat)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/edm9fd/sandbox/neurofaune/configs/default.yaml"),
        help="Configuration file",
    )
    parser.add_argument(
        "--analysis",
        choices=["falff", "reho", "both"],
        default="both",
        help="Which analysis to run (default: both)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=6,
        help="Number of parallel workers (default: 6)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if outputs exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sessions without processing",
    )
    parser.add_argument(
        "--skip-sigma",
        action="store_true",
        help="Skip SIGMA-space warping (only compute fALFF/ReHo in native space)",
    )
    parser.add_argument(
        "--skip-fc",
        action="store_true",
        help="Skip functional connectivity matrices",
    )

    args = parser.parse_args()

    print(f"Scanning for preprocessed BOLD in {args.study_root / 'derivatives'}...")
    sessions = find_preprocessed_sessions(args.study_root)
    print(f"Found {len(sessions)} preprocessed BOLD sessions")

    if not sessions:
        print("No sessions found. Exiting.")
        return 1

    if args.dry_run:
        print(f"\nSessions to analyze ({args.analysis}):")

        # Check data availability
        n_has_regressed = 0
        n_has_smooth = 0
        n_missing = 0
        n_has_sigma = 0
        n_will_register = 0
        n_cannot_register = 0

        for s in sessions:
            subject, session = s["subject"], s["session"]
            deriv_dir = s["derivatives_dir"]
            work_dir = s["work_dir"]

            regressed = deriv_dir / f"{subject}_{session}_desc-regressed_bold.nii.gz"
            regressed_work = work_dir / f"{subject}_{session}_bold_regressed.nii.gz"
            smooth = work_dir / f"{subject}_{session}_bold_smooth.nii.gz"

            if regressed.exists() or regressed_work.exists():
                source = "regressed"
                n_has_regressed += 1
            elif smooth.exists():
                source = "reconstruct"
                n_has_smooth += 1
            else:
                source = "MISSING"
                n_missing += 1

            if s["has_sigma_transforms"]:
                sigma_flag = "SIGMA"
                n_has_sigma += 1
            elif s["can_register"]:
                sigma_flag = "will-register"
                n_will_register += 1
            else:
                sigma_flag = "no-xfm"
                n_cannot_register += 1

            print(f"  {subject}/{session}  [fALFF: {source}] [{sigma_flag}]")

        print(f"\nfALFF data availability:")
        print(f"  Regressed available: {n_has_regressed}")
        print(f"  Will reconstruct:    {n_has_smooth}")
        print(f"  Missing (no smooth): {n_missing}")

        print(f"\nSIGMA transforms:")
        print(f"  Ready for warping:      {n_has_sigma}")
        print(f"  Will register on-the-fly: {n_will_register}")
        print(f"  Cannot register:        {n_cannot_register}")

        from collections import Counter

        cohort_counts = Counter(s["session"] for s in sessions)
        print(f"\nBy cohort:")
        for cohort, count in sorted(cohort_counts.items()):
            print(f"  {cohort}: {count} sessions")

        return 0

    # Load config for display
    config = load_config(args.config)

    print(f"\nAnalysis configuration:")
    print(f"  Study root: {args.study_root}")
    print(f"  Config: {args.config}")
    print(f"  Analysis: {args.analysis}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Force: {args.force}")
    print(f"  Skip SIGMA: {args.skip_sigma}")
    print(f"  Skip FC:    {args.skip_fc}")
    print(f"  fALFF band: {get_config_value(config, 'functional.analysis.falff.low_freq', 0.01)}"
          f" - {get_config_value(config, 'functional.analysis.falff.high_freq', 0.08)} Hz")
    print(f"  ReHo neighborhood: {get_config_value(config, 'functional.analysis.reho.neighborhood', 27)}")

    # Process sessions
    results = []
    completed = 0
    failed = 0
    skipped = 0

    # Build args for workers
    work_args = [
        (s, args.analysis, args.config, args.force, args.skip_sigma, args.skip_fc)
        for s in sessions
    ]

    if args.n_workers == 1:
        # Sequential — easier debugging
        for wa in work_args:
            result = _process_session_wrapper(wa)
            results.append(result)
            if result["status"] == "success":
                completed += 1
            elif result["status"] == "failed":
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(_process_session_wrapper, wa): wa
                for wa in work_args
            }

            for future in as_completed(futures):
                wa = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    total = completed + failed + skipped
                    if result["status"] == "success":
                        completed += 1
                        print(f"  [{total + 1}/{len(sessions)}] {result['key']}: SUCCESS")
                    elif result["status"] == "failed":
                        failed += 1
                        print(f"  [{total + 1}/{len(sessions)}] {result['key']}: FAILED — {result.get('error', '?')}")
                except Exception as e:
                    failed += 1
                    print(f"  EXCEPTION: {e}")
                    results.append({"status": "exception", "error": str(e)})

    # Save batch results
    log_dir = Path("/tmp/resting_state_analysis")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = log_dir / f"batch_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH RESTING-STATE ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total sessions: {len(sessions)}")
    print(f"Successful: {completed}")
    print(f"Failed: {failed}")
    print(f"Results: {results_file}")

    if failed > 0:
        print(f"\nFailed sessions:")
        for r in results:
            if r.get("status") == "failed":
                print(f"  {r.get('key', '?')}: {r.get('error', '?')}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
