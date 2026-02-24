"""Shared CLI helpers for CovNet analysis scripts.

Provides argparse argument groups and comparison resolution used by
the thin wrapper scripts (covnet_nbs.py, covnet_territory.py, etc.).
"""

import argparse
from pathlib import Path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --prep-dir, --metrics, --seed arguments."""
    parser.add_argument(
        "--prep-dir",
        type=Path,
        required=True,
        help="Directory with prepared CovNet data (from covnet_prepare.py)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["FA", "MD", "AD", "RD"],
        help="DTI metrics to process (default: FA MD AD RD)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )


def add_comparison_args(parser: argparse.ArgumentParser) -> None:
    """Add --comparisons and --groups arguments."""
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=["dose", "cross-timepoint", "cross-dose-timepoint"],
        choices=["dose", "cross-timepoint", "cross-dose-timepoint"],
        help=(
            "Named comparison sets to run. "
            "dose: each dose vs control within each PND (9 pairs). "
            "cross-timepoint: pairwise PND comparisons within each dose (12 pairs). "
            "cross-dose-timepoint: dosed groups vs later PND controls (9 pairs). "
            "Default: all three."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Explicit group pairs as flat list (parsed pairwise: A B C D -> "
            "A vs B, C vs D). Overrides --comparisons."
        ),
    )


def parse_comparisons(args, analysis):
    """Resolve --groups or --comparisons CLI args into comparison pairs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args (must have ``groups`` and ``comparisons``).
    analysis : CovNetAnalysis
        Loaded analysis instance.

    Returns
    -------
    list of (str, str) pairs.
    """
    if getattr(args, "groups", None):
        flat = args.groups
        if len(flat) % 2 != 0:
            raise ValueError(
                f"--groups needs even number of labels (got {len(flat)})"
            )
        return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    return analysis.resolve_comparisons(
        getattr(args, "comparisons", None)
    )
