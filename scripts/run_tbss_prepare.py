#!/usr/bin/env python3
"""
TBSS Data Preparation Script

Prepares DTI data for TBSS analysis by warping metrics to SIGMA space,
creating a tissue-informed WM skeleton, and projecting metrics onto it.

This is a convenience wrapper around:
    python -m neurofaune.analysis.tbss.prepare_tbss

Prerequisites:
    - Completed DTI preprocessing (batch_preprocess_dwi.py)
    - Completed FA-to-T2w registration (batch_register_fa_to_t2w.py)
    - Completed subject-to-template registration (batch_register_dwi.py)
    - Template-to-SIGMA registration

Usage:
    # Prepare all subjects with default settings
    uv run python scripts/run_tbss_prepare.py \\
        --config config.yaml \\
        --output-dir /study/analysis/tbss/

    # Use subject list from neuroaider design matrix
    uv run python scripts/run_tbss_prepare.py \\
        --config config.yaml \\
        --output-dir /study/analysis/tbss/ \\
        --subject-list /study/designs/model1/subject_list.txt

    # Specific cohort with QC exclusions
    uv run python scripts/run_tbss_prepare.py \\
        --config config.yaml \\
        --output-dir /study/analysis/tbss/ \\
        --cohorts p60 \\
        --exclude-file /study/qc/dwi_batch_summary/exclude_subjects.txt

    # Dry run to verify subject discovery
    uv run python scripts/run_tbss_prepare.py \\
        --config config.yaml \\
        --output-dir /study/analysis/tbss/ \\
        --dry-run
"""

import sys

from neurofaune.analysis.tbss.prepare_tbss import main

if __name__ == '__main__':
    main()
