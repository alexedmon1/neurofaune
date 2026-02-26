#!/usr/bin/env python3
"""
CovNet data preparation: load ROI data, bilateral average, compute correlation
matrices, and save everything to disk for downstream test scripts.

Runs Phases 1-4 of the CovNet pipeline:
  1. Load and filter ROI data
  2. Bilateral averaging
  3. Define groups and compute Spearman correlation matrices
  4. Generate correlation heatmaps

Usage:
    uv run python scripts/covnet_prepare.py \
        --roi-dir /mnt/arborea/bpa-rat/analysis/roi \
        --exclusion-csv /mnt/arborea/bpa-rat/dti_nonstandard_slices.csv \
        --output-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA MD AD RD
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.connectome import CovNetAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CovNet data preparation: load, average, compute matrices"
    )
    parser.add_argument(
        "--roi-dir", type=Path, required=True,
        help="Directory containing roi_*_wide.csv files",
    )
    parser.add_argument(
        "--exclusion-csv", type=Path, default=None,
        help="CSV of sessions to exclude (must have subject, session columns)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for CovNet results",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["FA", "MD", "AD", "RD"],
        help="DTI metrics to prepare (default: FA MD AD RD)",
    )
    parser.add_argument(
        "--labels-csv", type=Path, default=None,
        help="SIGMA atlas labels CSV for hybrid territory mapping (default: arborea path)",
    )
    args = parser.parse_args()

    if not args.roi_dir.exists():
        logger.error(f"ROI directory not found: {args.roi_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        try:
            analysis = CovNetAnalysis.prepare(
                args.roi_dir, args.exclusion_csv, args.output_dir, metric,
                labels_csv=args.labels_csv,
            )
            analysis.save()
        except FileNotFoundError as e:
            logger.warning(str(e))

    logger.info(f"\nPreparation complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
