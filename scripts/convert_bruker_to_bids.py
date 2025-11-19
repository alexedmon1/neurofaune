#!/usr/bin/env python3
"""
Convert Bruker MRI data to BIDS format.

Usage:
    python convert_bruker_to_bids.py --cohort-root /mnt/arborea/7T_Scanner_new \
                                      --output-root /mnt/arborea/7T_Scanner_new \
                                      --cohorts Cohort1 Cohort2

This script will:
1. Discover all Bruker sessions across specified cohorts
2. Classify scans by modality (T2w, DTI, fMRI, spectroscopy, MSME, MTR)
3. Convert to NIfTI format
4. Organize into BIDS-like structure:
   output_root/raw/bids/sub-Rat###/ses-p##/anat|dwi|func|spec/
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.utils.bruker_convert import process_all_cohorts


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert Bruker MRI data to BIDS format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--cohort-root',
        type=Path,
        required=True,
        help='Root directory containing Cohort# directories'
    )

    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Output root directory (will create raw/bids/ structure)'
    )

    parser.add_argument(
        '--cohorts',
        nargs='+',
        help='Specific cohorts to process (e.g., Cohort1 Cohort2). If not specified, processes all.'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Discover sessions but do not convert files'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate paths
    if not args.cohort_root.exists():
        logger.error(f"Cohort root directory not found: {args.cohort_root}")
        sys.exit(1)

    logger.info(f"Cohort root: {args.cohort_root}")
    logger.info(f"Output root: {args.output_root}")
    if args.cohorts:
        logger.info(f"Processing cohorts: {', '.join(args.cohorts)}")
    else:
        logger.info("Processing all cohorts")

    if args.dry_run:
        logger.info("DRY RUN MODE - files will not be converted")

    # Process cohorts
    try:
        stats = process_all_cohorts(
            cohort_root=args.cohort_root,
            output_root=args.output_root,
            cohorts=args.cohorts
        )

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Sessions processed: {stats['sessions_processed']}")
        logger.info(f"Scans converted: {stats['scans_converted']}")
        if stats['failures']:
            logger.warning(f"Failures: {len(stats['failures'])}")
            for failure in stats['failures']:
                logger.warning(f"  - {failure}")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
