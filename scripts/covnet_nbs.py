#!/usr/bin/env python3
"""
CovNet Phase 5: Network-Based Statistic (NBS).

Loads prepared group arrays and runs NBS comparisons with permutation testing.
Supports named comparison sets (dose, cross-timepoint, cross-dose-timepoint)
or explicit group pairs via --groups.

Usage:
    uv run python scripts/covnet_nbs.py \
        --prep-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA MD AD RD \
        --comparisons dose cross-timepoint cross-dose-timepoint \
        --n-permutations 5000 \
        --nbs-threshold 3.0 \
        --seed 42 \
        --n-workers 8

    # Or with explicit pairs:
    uv run python scripts/covnet_nbs.py \
        --prep-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA \
        --groups p30_L p30_C p60_H p60_C \
        --n-workers 8
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
        description="CovNet Phase 5: Network-Based Statistic (NBS)"
    )
    add_common_args(parser)
    add_comparison_args(parser)
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Number of permutations for NBS (default: 5000)",
    )
    parser.add_argument(
        "--nbs-threshold", type=float, default=3.0,
        help="Z-statistic threshold for suprathreshold edges (default: 3.0)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    args = parser.parse_args()

    if not args.prep_dir.exists():
        logger.error(f"Prep directory not found: {args.prep_dir}")
        sys.exit(1)

    for metric in args.metrics:
        analysis = CovNetAnalysis.load(args.prep_dir, metric)
        comparisons = parse_comparisons(args, analysis)
        analysis.run_nbs(
            comparisons, args.n_permutations, args.nbs_threshold,
            args.seed, args.n_workers,
        )

    logger.info("\nNBS analysis complete.")


if __name__ == "__main__":
    main()
