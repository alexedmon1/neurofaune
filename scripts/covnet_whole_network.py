#!/usr/bin/env python3
"""
CovNet Phase 8: Whole-network similarity tests.

Loads prepared group arrays and compares covariance network structure between
groups using Mantel test, Frobenius distance, and spectral divergence, all
with permutation-based p-values.

Usage:
    uv run python scripts/covnet_whole_network.py \
        --prep-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA MD AD RD \
        --comparisons dose cross-timepoint cross-dose-timepoint \
        --n-permutations 5000 \
        --seed 42 \
        --n-workers 8

    # Or with explicit pairs:
    uv run python scripts/covnet_whole_network.py \
        --prep-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA \
        --groups p30_L p30_C \
        --n-workers 4
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.connectome import CovNetAnalysis
from scripts.covnet_common import add_common_args, add_comparison_args, parse_comparisons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CovNet Phase 8: Whole-network similarity tests"
    )
    add_common_args(parser)
    add_comparison_args(parser)
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Number of permutations (default: 5000)",
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
        analysis.run_whole_network(
            comparisons, args.n_permutations, args.seed, args.n_workers,
        )

    logger.info("\nWhole-network analysis complete.")


if __name__ == "__main__":
    main()
