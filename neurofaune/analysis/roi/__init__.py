"""Deprecated: use neurofaune.connectome.roi_extraction instead."""

from neurofaune.connectome.roi_extraction import (  # noqa: F401
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
