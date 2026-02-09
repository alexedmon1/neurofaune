"""
ROI-level metric extraction using SIGMA atlas parcellation.

Provides functions to extract mean metric values within atlas-defined
regions of interest, aggregate by anatomical territory, and produce
tidy DataFrames for statistical analysis.
"""

from neurofaune.analysis.roi.extraction import (
    compute_territory_means,
    discover_sigma_metrics,
    extract_all_subjects,
    extract_roi_means,
    load_parcellation,
    merge_phenotype,
    to_long_format,
)

__all__ = [
    "load_parcellation",
    "extract_roi_means",
    "compute_territory_means",
    "discover_sigma_metrics",
    "extract_all_subjects",
    "to_long_format",
    "merge_phenotype",
]
