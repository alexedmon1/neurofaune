#!/usr/bin/env python3
"""
CLI entry point for automated Bruker session processing.

Inventories a Bruker session directory, auto-selects the best scan for each
modality that has a preprocessing pipeline (T2w, DTI, fMRI, MSME), converts
to NIfTI, and runs each applicable pipeline.

Examples::

    # Full processing (all detected modalities)
    uv run python scripts/process_bruker_session.py \\
        /path/to/bruker_session /path/to/output \\
        --subject sub-PRE10 --session ses-01

    # Inventory only (list scans without processing)
    uv run python scripts/process_bruker_session.py \\
        /path/to/bruker_session /path/to/output \\
        --subject sub-PRE10 --session ses-01 --inventory-only

    # With custom config and CPU-only eddy
    uv run python scripts/process_bruker_session.py \\
        /path/to/bruker_session /path/to/output \\
        --subject sub-PRE10 --session ses-01 \\
        --config config.yaml --no-gpu
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Automated Bruker session processing (all modalities)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Bruker session directory (contains numbered scan sub-dirs)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Study root directory for outputs",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject identifier (e.g. sub-PRE10)",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session identifier (e.g. ses-01)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: use minimal built-in config)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU-accelerated eddy (use CPU eddy)",
    )
    parser.add_argument(
        "--run-registration",
        action="store_true",
        help="Run cross-modal registration to template (default: skip)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only inventory scans and write CSV; do not process",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate inputs
    if not args.session_dir.is_dir():
        print(f"ERROR: Session directory not found: {args.session_dir}", file=sys.stderr)
        sys.exit(1)

    if args.config and not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    from neurofaune.preprocess.workflows.bruker_session import (
        process_bruker_session,
    )

    try:
        results = process_bruker_session(
            session_dir=args.session_dir,
            output_dir=args.output_dir,
            subject=args.subject,
            session=args.session,
            config_path=args.config,
            use_gpu=not args.no_gpu,
            skip_registration=not args.run_registration,
            force=args.force,
            inventory_only=args.inventory_only,
        )
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
