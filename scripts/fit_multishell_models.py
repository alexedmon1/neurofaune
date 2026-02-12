#!/usr/bin/env python3
"""
Fit multi-shell diffusion models (DKI + NODDI) on preprocessed DWI data.

This script runs *after* standard DWI preprocessing (eddy, dtifit) and
requires multi-shell data (>= 2 non-zero b-value shells).

Examples::

    # Fit both DKI and NODDI for a single subject
    uv run python scripts/fit_multishell_models.py \
        /mnt/d/scratch/Rat/dti/study_PRE10 \
        --subject sub-PRE10 --session ses-01

    # DKI only (no AMICO dependency)
    uv run python scripts/fit_multishell_models.py \
        /mnt/d/scratch/Rat/dti/study_PRE10 \
        --subject sub-PRE10 --session ses-01 --no-noddi

    # Batch: all subjects
    uv run python scripts/fit_multishell_models.py \
        /mnt/d/scratch/Rat/dti/study_PRE10 --all
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Fit DKI and/or NODDI on preprocessed multi-shell DWI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Study root directory containing derivatives/",
    )
    parser.add_argument("--subject", help="Subject identifier (e.g. sub-PRE10)")
    parser.add_argument("--session", help="Session identifier (e.g. ses-01)")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_subjects",
        help="Process all subjects found in derivatives/",
    )
    parser.add_argument("--no-dki", action="store_true", help="Skip DKI fitting")
    parser.add_argument("--no-noddi", action="store_true", help="Skip NODDI fitting")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from neurofaune.preprocess.workflows.multishell_models import run_multishell_fitting

    study_dir = args.study_dir.resolve()
    if not study_dir.exists():
        print(f"ERROR: Study directory does not exist: {study_dir}", file=sys.stderr)
        sys.exit(1)

    deriv_dir = study_dir / "derivatives"
    if not deriv_dir.exists():
        print(f"ERROR: No derivatives/ directory in {study_dir}", file=sys.stderr)
        sys.exit(1)

    # Build list of (subject, session) pairs
    pairs = []
    if args.all_subjects:
        for sub_dir in sorted(deriv_dir.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            for ses_dir in sorted(sub_dir.iterdir()):
                if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                    continue
                dwi_dir = ses_dir / "dwi"
                if dwi_dir.exists():
                    pairs.append((sub_dir.name, ses_dir.name))
        if not pairs:
            print("ERROR: No subject/session directories with DWI data found.", file=sys.stderr)
            sys.exit(1)
    elif args.subject:
        session = args.session or "ses-01"
        pairs.append((args.subject, session))
    else:
        parser.error("Specify --subject or --all")

    # Process
    success = 0
    failed = 0
    for subject, session in pairs:
        print(f"\n{'#' * 70}")
        print(f"# {subject}  {session}")
        print(f"{'#' * 70}\n")
        try:
            results = run_multishell_fitting(
                output_dir=study_dir,
                subject=subject,
                session=session,
                do_dki=not args.no_dki,
                do_noddi=not args.no_noddi,
                force=args.force,
            )
            if results.get("dki") or results.get("noddi"):
                success += 1
            else:
                logging.getLogger(__name__).warning(
                    f"No models fitted for {subject} {session} "
                    f"(multi-shell={results['shell_info']['is_multishell']})"
                )
        except Exception:
            logging.getLogger(__name__).exception(f"Failed: {subject} {session}")
            failed += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Done. Processed: {success}  Failed: {failed}  Total: {len(pairs)}")
    print(f"{'=' * 70}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
