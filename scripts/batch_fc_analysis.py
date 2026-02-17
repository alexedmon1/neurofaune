#!/usr/bin/env python3
"""
Batch functional connectivity analysis: ROI-to-ROI FC in SIGMA space.

Discovers all sessions with preprocessed BOLD data and computes:
- SIGMA-space warping of preprocessed BOLD (if not already done)
- ROI-to-ROI functional connectivity matrices (Pearson r → Fisher z)

Requires SIGMA-space BOLD for each session. If the SIGMA-space BOLD
doesn't exist yet, this script will warp it on-the-fly (computing the
BOLD-to-template registration first if needed).
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.analysis.func import (
    compute_fc_matrix,
    extract_roi_timeseries,
    save_fc_matrix,
)
from neurofaune.config import get_config_value, load_config
from neurofaune.preprocess.workflows.func_preprocess import register_bold_to_template
from neurofaune.templates.registration import warp_bold_to_sigma


def find_preprocessed_sessions(study_root: Path, subjects: list = None) -> list:
    """
    Find all sessions with preprocessed BOLD data.

    Enriches each session dict with cohort, transform paths, and
    SIGMA transform availability for the SIGMA-warping phase.
    Also includes paths needed for on-the-fly BOLD->Template
    registration if the transform doesn't exist yet.

    Parameters
    ----------
    study_root : Path
        Study root directory.
    subjects : list, optional
        If provided, only include these subjects (e.g. ['sub-Rat49']).

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

        # Apply subject filter
        if subjects and subject not in subjects:
            continue

        mask = func_dir / f"{subject}_{session}_desc-brain_mask.nii.gz"

        if not mask.exists():
            print(f"  Warning: Skipping {subject}/{session} — missing mask")
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

        # SIGMA-space BOLD (may already exist from a previous run)
        sigma_bold = func_dir / f"{subject}_{session}_space-SIGMA_desc-preproc_bold.nii.gz"

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
            "sigma_bold": sigma_bold,
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


def process_session(session_info: dict, config: dict,
                    force: bool = False) -> dict:
    """
    Run functional connectivity analysis for a single session.

    Ensures SIGMA-space BOLD exists (warping on-the-fly if needed),
    then computes ROI-to-ROI FC matrix.

    Parameters
    ----------
    session_info : dict
        Session information from find_preprocessed_sessions.
    config : dict
        Loaded config.
    force : bool
        Recompute even if outputs exist.

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
    }

    try:
        print(f"\n{'=' * 60}")
        print(f"Session: {key}")
        print(f"{'=' * 60}")

        # ---- Ensure SIGMA-space BOLD exists ----
        sigma_bold = session_info["sigma_bold"]
        has_sigma = session_info["has_sigma_transforms"]

        # BOLD-to-Template Registration (on-the-fly if needed)
        if not sigma_bold.exists() and not has_sigma and session_info["can_register"]:
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

            # Update with newly computed transform
            bold_to_template = session_info["bold_to_template"]
            if bold_to_template.exists():
                has_sigma = True
                result["outputs"]["registration"] = str(bold_to_template)
                print(f"  Registration complete: {bold_to_template.name}")
            else:
                print(f"  WARNING: registration ran but transform not found at {bold_to_template}")

        # Warp preproc_bold to SIGMA if not already present
        if not sigma_bold.exists() and has_sigma:
            print(f"\n  Warping preprocessed BOLD to SIGMA space...")
            sigma_template = Path(get_config_value(config, "atlas.study_space.template"))
            sigma_brain_mask = Path(get_config_value(config, "atlas.study_space.brain_mask"))

            # Build transform chain
            sigma_transforms = []
            if session_info["tpl_to_sigma_warp"] is not None:
                sigma_transforms.append(str(session_info["tpl_to_sigma_warp"]))
            sigma_transforms.append(str(session_info["tpl_to_sigma_affine"]))
            sigma_transforms.append(str(session_info["bold_to_template"]))

            input_files = {
                "desc-preproc_bold": session_info["preproc_bold"],
            }

            sigma_outputs = warp_bold_to_sigma(
                input_files=input_files,
                transforms=sigma_transforms,
                sigma_template=sigma_template,
                sigma_brain_mask=sigma_brain_mask,
                output_dir=deriv_dir,
                subject=subject,
                session=session,
                work_dir=session_info["study_root"] / "work",
            )
            result["outputs"]["sigma_warp"] = {
                k: str(v) for k, v in sigma_outputs.items()
            }

            # Re-check after warping
            if not sigma_bold.exists():
                result["status"] = "failed"
                result["error"] = "SIGMA warp ran but output not found"
                print(f"  FAILED: SIGMA warp produced no output at {sigma_bold.name}")
                return result

        if not sigma_bold.exists():
            if not session_info["can_register"]:
                msg = "No SIGMA-space BOLD and cannot register (missing mcf or cohort template)"
            else:
                msg = "No SIGMA-space BOLD and registration failed"
            result["status"] = "failed"
            result["error"] = msg
            print(f"  SKIPPED: {msg}")
            return result

        # ---- Functional Connectivity ----
        atlas_path = Path(get_config_value(config, "atlas.study_space.parcellation"))
        sigma_brain_mask = Path(get_config_value(config, "atlas.study_space.brain_mask"))

        fc_base = deriv_dir / f"{subject}_{session}_space-SIGMA_desc-fc_bold"
        fc_npy = fc_base.with_suffix(".npy")

        if fc_npy.exists() and not force:
            print(f"  FC matrix already exists, skipping (use --force to recompute)")
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

        # Save per-session metadata JSON
        metadata_file = deriv_dir / f"{subject}_{session}_desc-fc_analysis.json"
        metadata = {
            "subject": subject,
            "session": session,
            "analysis_date": datetime.now().isoformat(),
            "atlas": str(atlas_path),
            "sigma_bold": str(sigma_bold),
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
    session_info, config_path, force = args
    config = load_config(config_path)
    return process_session(session_info, config, force)


def main():
    parser = argparse.ArgumentParser(
        description="Batch ROI-to-ROI functional connectivity in SIGMA space"
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
        "--subjects",
        nargs="+",
        default=None,
        help="Only process these subjects (e.g. sub-Rat49 sub-Rat50)",
    )

    args = parser.parse_args()

    print(f"Scanning for preprocessed BOLD in {args.study_root / 'derivatives'}...")
    sessions = find_preprocessed_sessions(args.study_root, subjects=args.subjects)
    print(f"Found {len(sessions)} preprocessed BOLD sessions")

    if args.subjects:
        print(f"  Filtered to subjects: {', '.join(args.subjects)}")

    if not sessions:
        print("No sessions found. Exiting.")
        return 1

    if args.dry_run:
        print(f"\nSessions to analyze (FC):")

        n_has_sigma_bold = 0
        n_has_transforms = 0
        n_will_register = 0
        n_cannot = 0

        for s in sessions:
            subject, session = s["subject"], s["session"]
            sigma_bold = s["sigma_bold"]

            fc_npy = (s["derivatives_dir"]
                      / f"{subject}_{session}_space-SIGMA_desc-fc_bold.npy")

            if sigma_bold.exists():
                sigma_flag = "SIGMA-BOLD"
                n_has_sigma_bold += 1
            elif s["has_sigma_transforms"]:
                sigma_flag = "will-warp"
                n_has_transforms += 1
            elif s["can_register"]:
                sigma_flag = "will-register+warp"
                n_will_register += 1
            else:
                sigma_flag = "no-xfm"
                n_cannot += 1

            fc_status = "done" if fc_npy.exists() else "pending"
            print(f"  {subject}/{session}  [FC: {fc_status}] [{sigma_flag}]")

        print(f"\nSIGMA-space BOLD availability:")
        print(f"  Already warped:             {n_has_sigma_bold}")
        print(f"  Will warp (transforms OK):  {n_has_transforms}")
        print(f"  Will register + warp:       {n_will_register}")
        print(f"  Cannot reach SIGMA:         {n_cannot}")

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
    print(f"  Workers: {args.n_workers}")
    print(f"  Force: {args.force}")
    print(f"  Atlas: {get_config_value(config, 'atlas.study_space.parcellation', 'N/A')}")

    # Process sessions
    results = []
    completed = 0
    failed = 0

    # Build args for workers
    work_args = [
        (s, args.config, args.force)
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

                    total = completed + failed
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
    log_dir = Path("/tmp/fc_batch_analysis")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = log_dir / f"batch_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH FC ANALYSIS COMPLETE")
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
