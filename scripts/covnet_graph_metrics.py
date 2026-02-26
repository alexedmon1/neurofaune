#!/usr/bin/env python3
"""
CovNet Phase 7: Graph-theoretic metrics with permutation testing.

Loads prepared group arrays, computes graph metrics (efficiency, clustering,
modularity, path length, small-worldness) at multiple network densities,
and runs permutation-based pairwise group comparisons.

Usage:
    uv run python scripts/covnet_graph_metrics.py \
        --prep-dir /mnt/arborea/bpa-rat/analysis/covnet_dti \
        --metrics FA MD AD RD \
        --densities 0.10 0.15 0.20 0.25 \
        --n-permutations 5000 \
        --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.connectome import CovNetAnalysis
from scripts.covnet_common import add_common_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CovNet Phase 7: Graph-theoretic metrics with permutation testing"
    )
    add_common_args(parser)
    parser.add_argument(
        "--densities", nargs="+", type=float,
        default=[0.10, 0.15, 0.20, 0.25],
        help="Network densities to test (default: 0.10 0.15 0.20 0.25)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=5000,
        help="Number of permutations (default: 5000)",
    )
    args = parser.parse_args()

    if not args.prep_dir.exists():
        logger.error(f"Prep directory not found: {args.prep_dir}")
        sys.exit(1)

    for metric in args.metrics:
        analysis = CovNetAnalysis.load(args.prep_dir, metric)
        analysis.run_graph_metrics(args.densities, args.n_permutations, args.seed)

    logger.info("\nGraph metrics analysis complete.")


if __name__ == "__main__":
    main()
