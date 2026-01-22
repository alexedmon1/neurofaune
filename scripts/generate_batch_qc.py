#!/usr/bin/env python3
"""
Generate batch QC summary reports for preprocessed data.

This script collects QC metrics from all subjects and generates:
- Summary HTML dashboard with outlier flagging
- Metrics CSV for all subjects
- Thumbnail gallery of key QC images
- Distribution plots by cohort

Usage:
    # Generate DWI batch summary
    python scripts/generate_batch_qc.py /path/to/study --modality dwi

    # Generate for all modalities
    python scripts/generate_batch_qc.py /path/to/study --modality all

    # Generate with custom outlier threshold
    python scripts/generate_batch_qc.py /path/to/study --modality dwi --z-threshold 3.0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.qc import (
    generate_batch_qc_summary,
    generate_slice_qc_summary,
    BatchQCConfig,
    get_batch_summary_dir
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate batch QC summary reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate DWI batch summary for a study
    python scripts/generate_batch_qc.py /mnt/arborea/bpa-rat --modality dwi

    # Generate batch summaries for all modalities
    python scripts/generate_batch_qc.py /mnt/arborea/bpa-rat --modality all

    # Generate with custom z-score threshold for outlier detection
    python scripts/generate_batch_qc.py /mnt/arborea/bpa-rat --modality dwi --z-threshold 3.0

Output Structure:
    {study_root}/qc/{modality}_batch_summary/
    ├── summary.html           # Main dashboard
    ├── metrics.csv            # All metrics, all subjects
    ├── thumbnail_gallery.html # Image grid
    ├── outliers.json          # Flagged subjects
    └── figures/               # Distribution plots
        """
    )

    parser.add_argument(
        'study_root',
        type=Path,
        help='Study root directory containing qc/ folder'
    )

    parser.add_argument(
        '--modality', '-m',
        choices=['dwi', 'anat', 'func', 'msme', 'all'],
        default='all',
        help='Modality to generate summary for (default: all)'
    )

    parser.add_argument(
        '--z-threshold', '-z',
        type=float,
        default=2.5,
        help='Z-score threshold for outlier detection (default: 2.5)'
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        help='Specific subjects to include (default: all)'
    )

    parser.add_argument(
        '--qc-dir',
        type=Path,
        help='Override QC directory (default: {study_root}/qc/)'
    )

    parser.add_argument(
        '--slice-qc',
        action='store_true',
        help='Generate slice-level QC for DWI (identifies bad slices for TBSS)'
    )

    args = parser.parse_args()

    # Validate study root
    study_root = args.study_root.resolve()
    if not study_root.exists():
        print(f"Error: Study root does not exist: {study_root}")
        sys.exit(1)

    # Determine QC directory
    qc_dir = args.qc_dir if args.qc_dir else study_root / 'qc'
    if not qc_dir.exists():
        print(f"Error: QC directory does not exist: {qc_dir}")
        sys.exit(1)

    # Configure outlier detection
    config = BatchQCConfig(outlier_z_threshold=args.z_threshold)

    # Determine modalities to process
    if args.modality == 'all':
        modalities = ['dwi', 'anat', 'func', 'msme']
    else:
        modalities = [args.modality]

    print("=" * 60)
    print("Batch QC Summary Generator")
    print("=" * 60)
    print(f"Study root: {study_root}")
    print(f"QC directory: {qc_dir}")
    print(f"Modalities: {', '.join(modalities)}")
    print(f"Outlier z-threshold: {args.z_threshold}")
    print(f"Slice QC: {'Yes' if args.slice_qc else 'No'}")
    if args.subjects:
        print(f"Subjects filter: {len(args.subjects)} subjects")
    print()

    results = {}
    slice_results = {}

    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Processing {modality.upper()}")
        print('=' * 60)

        try:
            output_dir = get_batch_summary_dir(study_root, modality)

            result = generate_batch_qc_summary(
                qc_dir=qc_dir,
                modality=modality,
                output_dir=output_dir,
                subjects=args.subjects,
                config=config
            )

            if result:
                results[modality] = result
                print(f"\n  Summary: {result.get('summary_html', 'N/A')}")
            else:
                print(f"\n  Warning: No QC data found for {modality}")

        except Exception as e:
            print(f"\n  Error processing {modality}: {e}")
            import traceback
            traceback.print_exc()

    # Generate slice-level QC for DWI if requested
    if args.slice_qc and ('dwi' in modalities or args.modality == 'all'):
        print(f"\n{'='*60}")
        print("Generating Slice-Level QC for DWI")
        print('=' * 60)

        try:
            derivatives_dir = study_root / 'derivatives'
            slice_output_dir = get_batch_summary_dir(study_root, 'dwi') / 'slice_qc'

            slice_results = generate_slice_qc_summary(
                derivatives_dir=derivatives_dir,
                output_dir=slice_output_dir,
                subjects=args.subjects
            )

            if slice_results:
                print(f"\n  Slice metrics: {slice_results.get('slice_metrics_csv', 'N/A')}")
                print(f"  Slice exclusions: {slice_results.get('slice_exclusions_json', 'N/A')}")

        except Exception as e:
            print(f"\n  Error generating slice QC: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for modality, result in results.items():
        if result:
            print(f"\n{modality.upper()}:")
            print(f"  Dashboard: {result.get('summary_html', 'N/A')}")
            print(f"  Metrics CSV: {result.get('metrics_csv', 'N/A')}")
            print(f"  Gallery: {result.get('gallery_html', 'N/A')}")

    if not results:
        print("\nNo batch summaries generated. Check that QC data exists.")
        sys.exit(1)

    print("\nDone!")


if __name__ == '__main__':
    main()
