#!/usr/bin/env python3
"""
TBSS Statistical Analysis Script

Runs group-level statistical analysis on prepared TBSS data using FSL randomise
with TFCE correction. Generates cluster reports with SIGMA atlas labels.

This is a convenience wrapper around:
    python -m neurofaune.analysis.tbss.run_tbss_stats

Prerequisites:
    - Completed TBSS data preparation (run_tbss_prepare.py)
    - Design matrix files (design.mat, design.con) from neuroaider

Usage:
    # Run analysis with pre-generated design
    uv run python scripts/run_tbss_stats.py \\
        --tbss-dir /study/analysis/tbss/ \\
        --design-dir /study/designs/dose_response/ \\
        --analysis-name dose_p60 \\
        --metrics FA MD AD RD

    # Quick test with fewer permutations
    uv run python scripts/run_tbss_stats.py \\
        --tbss-dir /study/analysis/tbss/ \\
        --design-dir /study/designs/dose_response/ \\
        --analysis-name test_run \\
        --n-permutations 100 \\
        --seed 42

    # With SIGMA atlas labels (requires config with atlas paths)
    uv run python scripts/run_tbss_stats.py \\
        --tbss-dir /study/analysis/tbss/ \\
        --design-dir /study/designs/dose_response/ \\
        --analysis-name dose_p60 \\
        --config config.yaml
"""

import sys

from neurofaune.analysis.tbss.run_tbss_stats import main

if __name__ == '__main__':
    main()
