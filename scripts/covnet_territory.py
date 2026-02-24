#!/usr/bin/env python3
"""
CovNet Phase 6: Territory-level Fisher z-tests with FDR correction.

Loads pre-computed territory correlation matrices and runs edge-level
Fisher z-tests for each comparison, with Benjamini-Hochberg FDR correction.

Usage:
    uv run python scripts/covnet_territory.py \
        --prep-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA MD AD RD \
        --comparisons dose cross-timepoint cross-dose-timepoint
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.analysis.covnet.pipeline import CovNetAnalysis
from scripts.covnet_common import add_common_args, add_comparison_args, parse_comparisons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CovNet Phase 6: Territory-level Fisher z-tests with FDR"
    )
    add_common_args(parser)
    add_comparison_args(parser)
    args = parser.parse_args()

    if not args.prep_dir.exists():
        logger.error(f"Prep directory not found: {args.prep_dir}")
        sys.exit(1)

    for metric in args.metrics:
        analysis = CovNetAnalysis.load(args.prep_dir, metric)
        comparisons = parse_comparisons(args, analysis)
        analysis.run_territory(comparisons)

    logger.info("\nTerritory analysis complete.")


if __name__ == "__main__":
    main()
