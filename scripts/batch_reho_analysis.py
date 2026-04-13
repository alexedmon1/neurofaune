#!/usr/bin/env python3
"""
Batch ReHo analysis: compute ReHo maps and warp to SIGMA space.

Discovers all sessions with preprocessed BOLD data and computes:
- ReHo (Regional Homogeneity) from bandpass-filtered data
- Z-scored ReHo maps
- SIGMA-space warping of ReHo maps (optional)

ReHo operates on the bandpass-filtered preprocessed BOLD (desc-preproc_bold),
unlike fALFF which requires the unfiltered regressed data.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.analysis.func import (
    compute_reho_map,
    compute_reho_zscore,
)
from neurofaune.analysis.provenance import write_batch_run_manifest
from neurofaune.config import get_config_value, load_config
from neurofaune.preprocess.qc.func.motion_qc import compute_fd_from_confounds, get_scrub_indices
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
        confounds = func_dir / f"{subject}_{session}_desc-confounds_timeseries.tsv"
        sidecar = func_dir / f"{subject}_{session}_desc-preproc_bold.json"

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
            "confounds": confounds if confounds.exists() else None,
            "sidecar_json": sidecar if sidecar.exists() else None,
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


def load_qc_exclusions(qc_csv: Path) -> set:
    """Load (subject, session) pairs that failed QC from the summary CSV."""
    df = pd.read_csv(qc_csv)
    failed = df[~df["qc_pass"]]
    exclusions = set()
    for _, row in failed.iterrows():
        exclusions.add((row["subject"], row["session"]))
    print(f"QC exclusions: {len(exclusions)} sessions will skip SIGMA warping")
    for subj, ses in sorted(exclusions):
        print(f"  {subj}/{ses}")
    return exclusions


def process_session(session_info: dict, config: dict,
                    force: bool = False, skip_sigma: bool = False,
                    scrub_threshold: float = None,
                    min_clean_volumes: int = 0) -> dict:
    """
    Run ReHo analysis for a single session.

    Parameters
    ----------
    session_info : dict
        Session information from find_preprocessed_sessions.
    config : dict
        Loaded config.
    force : bool
        Recompute even if outputs exist.
    skip_sigma : bool
        Skip SIGMA-space warping.
    scrub_threshold : float, optional
        FD threshold (mm) for volume scrubbing. Volumes with FD above this
        value are dropped before computing KCC. If None, no scrubbing is applied.
    min_clean_volumes : int
        Minimum number of remaining volumes required after scrubbing. Sessions
        that fall below this threshold are skipped with a warning.

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
        neighborhood = get_config_value(config, "functional.analysis.reho.neighborhood", default=27)

        print(f"\n{'=' * 60}")
        print(f"Session: {key}")
        print(f"{'=' * 60}")

        # ---- Volume scrubbing ----
        scrub_indices = None
        if scrub_threshold is not None and session_info.get("confounds") is not None:
            confounds_path = session_info["confounds"]
            fd = compute_fd_from_confounds(confounds_path)
            scrub_indices = get_scrub_indices(fd, scrub_threshold)
            n_total = len(fd) + 1
            n_scrubbed = len(scrub_indices)
            n_clean = n_total - n_scrubbed
            print(f"  FD scrubbing: threshold={scrub_threshold}mm, "
                  f"{n_scrubbed}/{n_total} volumes flagged ({100*n_scrubbed/n_total:.1f}%)")
            if min_clean_volumes > 0 and n_clean < min_clean_volumes:
                print(f"  SKIPPING: only {n_clean} clean volumes < minimum {min_clean_volumes}")
                result["status"] = "skipped"
                result["reason"] = f"insufficient_clean_volumes ({n_clean} < {min_clean_volumes})"
                return result

        # ---- ReHo ----
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
                scrub_indices=scrub_indices,
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

            # Collect native-space ReHo maps to warp
            input_files = {}
            desc_map = {
                "desc-ReHo_bold": f"{subject}_{session}_desc-ReHo_bold.nii.gz",
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
                print("  No native-space ReHo maps found for SIGMA warping")
        elif not skip_sigma and not has_sigma:
            if not session_info["can_register"]:
                print(f"  Skipping SIGMA warp — missing mcf or cohort template")
            else:
                print(f"  Skipping SIGMA warp — registration failed")
            result["outputs"]["sigma"] = "no_transforms"

        # Save per-session metadata JSON
        metadata_file = deriv_dir / f"{subject}_{session}_desc-reho_analysis.json"
        metadata = {
            "subject": subject,
            "session": session,
            "analysis_date": datetime.now().isoformat(),
            "parameters": {
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
    session_info, config_path, force, skip_sigma, scrub_threshold, min_clean_volumes = args
    config = load_config(config_path)
    return process_session(session_info, config, force, skip_sigma,
                           scrub_threshold, min_clean_volumes)


def main():
    parser = argparse.ArgumentParser(
        description="Batch ReHo analysis with optional SIGMA-space warping"
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
    parser.add_argument(
        "--skip-sigma",
        action="store_true",
        help="Skip SIGMA-space warping (only compute ReHo in native space)",
    )
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help="QC summary CSV (from generate_func_registration_qc.py). "
             "Sessions with qc_pass=False skip SIGMA warping.",
    )
    parser.add_argument(
        "--scrub-threshold",
        type=float,
        default=None,
        help="FD threshold (mm) for volume scrubbing. Volumes with FD above "
             "this value are dropped before computing KCC. "
             "Default: no scrubbing. Typical value for rodent data: 0.05.",
    )
    parser.add_argument(
        "--min-clean-volumes",
        type=int,
        default=0,
        help="Minimum clean (non-scrubbed) volumes required after scrubbing. "
             "Sessions with fewer are skipped. Default: 0 (no check).",
    )

    args = parser.parse_args()

    # Load QC exclusions if provided
    qc_exclusions = set()
    if args.qc_csv:
        qc_exclusions = load_qc_exclusions(args.qc_csv)

    print(f"Scanning for preprocessed BOLD in {args.study_root / 'derivatives'}...")
    sessions = find_preprocessed_sessions(args.study_root, subjects=args.subjects)
    print(f"Found {len(sessions)} preprocessed BOLD sessions")

    if args.subjects:
        print(f"  Filtered to subjects: {', '.join(args.subjects)}")

    if not sessions:
        print("No sessions found. Exiting.")
        return 1

    if args.dry_run:
        print(f"\nSessions to analyze (ReHo):")

        n_has_sigma = 0
        n_will_register = 0
        n_cannot_register = 0

        for s in sessions:
            subject, session = s["subject"], s["session"]

            if s["has_sigma_transforms"]:
                sigma_flag = "SIGMA"
                n_has_sigma += 1
            elif s["can_register"]:
                sigma_flag = "will-register"
                n_will_register += 1
            else:
                sigma_flag = "no-xfm"
                n_cannot_register += 1

            reho_exists = (s["derivatives_dir"]
                           / f"{subject}_{session}_desc-ReHo_bold.nii.gz").exists()
            status = "done" if reho_exists else "pending"

            print(f"  {subject}/{session}  [ReHo: {status}] [{sigma_flag}]")

        print(f"\nSIGMA transforms:")
        print(f"  Ready for warping:        {n_has_sigma}")
        print(f"  Will register on-the-fly: {n_will_register}")
        print(f"  Cannot register:          {n_cannot_register}")

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
    print(f"  Skip SIGMA: {args.skip_sigma}")
    print(f"  QC exclusions: {len(qc_exclusions)} sessions")
    print(f"  ReHo neighborhood: {get_config_value(config, 'functional.analysis.reho.neighborhood', 27)}")
    if args.scrub_threshold is not None:
        print(f"  Scrub threshold: {args.scrub_threshold} mm FD")
        print(f"  Min clean volumes: {args.min_clean_volumes}")
    else:
        print(f"  Scrubbing: disabled")

    # Process sessions
    results = []
    completed = 0
    failed = 0

    # Build args for workers — QC-excluded sessions skip SIGMA warping
    work_args = []
    for s in sessions:
        session_skip_sigma = args.skip_sigma
        if (s["subject"], s["session"]) in qc_exclusions:
            session_skip_sigma = True
        work_args.append((s, args.config, args.force, session_skip_sigma,
                          args.scrub_threshold, args.min_clean_volumes))

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

    # Write batch run manifest (JSON + human-readable txt)
    log_dir = args.study_root / "logs" / "reho"
    input_files = [f for f in [args.qc_csv] if f is not None]
    json_path, txt_path = write_batch_run_manifest(
        output_dir=log_dir,
        analysis_name="ReHo",
        parameters=vars(args),
        session_results=results,
        input_files=input_files,
    )

    # Summary
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"\n{'=' * 60}")
    print("BATCH ReHo ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total sessions : {len(sessions)}")
    print(f"Successful     : {completed}")
    print(f"Skipped        : {skipped}")
    print(f"Failed         : {failed}")
    print(f"Manifest (txt) : {txt_path}")
    print(f"Manifest (json): {json_path}")

    if failed > 0:
        print(f"\nFailed sessions:")
        for r in results:
            if r.get("status") == "failed":
                print(f"  {r.get('key', '?')}: {r.get('error', '?')}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
