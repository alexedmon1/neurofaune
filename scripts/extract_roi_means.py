#!/usr/bin/env python3
"""
Extract ROI-level mean metrics from SIGMA atlas parcellation.

Produces wide and long CSV files with per-region and per-territory means
for all subjects with SIGMA-space metric maps.

Usage:
    uv run python scripts/extract_roi_means.py \
        --derivatives-dir /mnt/arborea/bpa-rat/derivatives \
        --parcellation /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
        --labels-csv /mnt/arborea/atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker_combined_250916.csv \
        --modality dwi \
        --metrics FA MD AD RD \
        --output-dir /mnt/arborea/bpa-rat/analysis/roi
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.connectome.roi_extraction import (
    extract_all_subjects,
    load_parcellation,
    merge_phenotype,
    to_long_format,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Extract ROI-level mean metrics from SIGMA atlas parcellation'
    )
    parser.add_argument(
        '--derivatives-dir', type=Path, required=True,
        help='Path to derivatives directory containing sub-*/ses-*/ folders',
    )
    parser.add_argument(
        '--parcellation', type=Path, required=True,
        help='Path to SIGMA parcellation NIfTI in study space',
    )
    parser.add_argument(
        '--labels-csv', type=Path, required=True,
        help='Path to SIGMA atlas labels CSV',
    )
    parser.add_argument(
        '--study-tracker', type=Path, default=None,
        help='Path to study tracker CSV for phenotype merge (optional)',
    )
    parser.add_argument(
        '--modality', type=str, required=True,
        choices=['dwi', 'msme', 'func'],
        help='Modality subdirectory to search',
    )
    parser.add_argument(
        '--metrics', type=str, nargs='+', required=True,
        help='Metric suffixes to extract (e.g. FA MD AD RD)',
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for CSV files',
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Add file handler for logging
    fh = logging.FileHandler(
        log_dir / f'extraction_{args.modality}_{datetime.now():%Y%m%d_%H%M%S}.log'
    )
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    logger.info(f"Extracting {args.metrics} from {args.modality}")
    logger.info(f"Derivatives: {args.derivatives_dir}")
    logger.info(f"Parcellation: {args.parcellation}")
    logger.info(f"Labels CSV: {args.labels_csv}")
    logger.info(f"Output: {args.output_dir}")

    # Extract all subjects
    wide_dfs = extract_all_subjects(
        derivatives_dir=args.derivatives_dir,
        parcellation_path=args.parcellation,
        labels_csv_path=args.labels_csv,
        modality=args.modality,
        metrics=args.metrics,
    )

    if not wide_dfs:
        logger.error("No data extracted — check paths and file patterns")
        sys.exit(1)

    # Load labels for long format conversion
    _, labels_df = load_parcellation(args.parcellation, args.labels_csv)

    summary = {
        'modality': args.modality,
        'extraction_time': datetime.now().isoformat(),
        'derivatives_dir': str(args.derivatives_dir),
        'parcellation': str(args.parcellation),
        'metrics': {},
    }

    for metric, wide_df in wide_dfs.items():
        # Merge phenotype if tracker provided
        if args.study_tracker:
            wide_df = merge_phenotype(wide_df, args.study_tracker)

        # Save wide format
        wide_path = args.output_dir / f'roi_{metric}_wide.csv'
        wide_df.to_csv(wide_path, index=False)
        logger.info(f"Saved wide: {wide_path} ({wide_df.shape})")

        # Convert and save long format
        long_df = to_long_format(wide_df, labels_df, metric)
        long_path = args.output_dir / f'roi_{metric}_long.csv'
        long_df.to_csv(long_path, index=False)
        logger.info(f"Saved long: {long_path} ({long_df.shape})")

        # Summary stats
        n_regions = len([c for c in wide_df.columns if not c.startswith('territory_')
                         and c not in ('subject', 'session', 'dose', 'sex')])
        n_territories = len([c for c in wide_df.columns if c.startswith('territory_')])
        summary['metrics'][metric] = {
            'n_subjects': len(wide_df),
            'n_regions': n_regions,
            'n_territories': n_territories,
            'wide_csv': str(wide_path),
            'long_csv': str(long_path),
        }

    # Save summary
    summary_path = args.output_dir / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    # Register with unified reporting system
    try:
        from neurofaune.analysis.reporting import register as report_register

        # Determine analysis root (parent of roi dir)
        analysis_root = args.output_dir.parent
        report_register(
            analysis_root=analysis_root,
            entry_id=f"roi_extraction_{args.modality}",
            analysis_type="roi_extraction",
            display_name=f"ROI Extraction ({args.modality.upper()}: {', '.join(args.metrics)})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                "modality": args.modality,
                "n_subjects": max(
                    (m.get("n_subjects", 0) for m in summary["metrics"].values()),
                    default=0,
                ),
                "metrics": args.metrics,
                "n_regions": next(iter(summary["metrics"].values()), {}).get("n_regions", 0),
                "n_territories": next(iter(summary["metrics"].values()), {}).get("n_territories", 0),
            },
            source_summary_json=str(summary_path.relative_to(analysis_root)),
        )
    except Exception as exc:
        logger.warning("Failed to register with reporting system: %s", exc)

    logger.info("Done.")


if __name__ == '__main__':
    main()
