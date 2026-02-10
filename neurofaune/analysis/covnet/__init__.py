"""
Covariance Network (CovNet) Analysis for ROI-level DTI metrics.

Builds Spearman correlation matrices per experimental group and statistically
compares them using the Network-Based Statistic (NBS) and graph-theoretic
metrics. Supports bilateral ROI averaging and territory-level analysis.

Submodules:
    matrices: Correlation matrix computation and grouping
    nbs: Network-Based Statistic for group comparison
    graph_metrics: Global efficiency, clustering, modularity
    visualization: Heatmaps, difference matrices, network plots
"""

from neurofaune.analysis.covnet.matrices import (
    bilateral_average,
    compute_spearman_matrices,
    define_groups,
    fisher_z_transform,
    load_and_prepare_data,
)

__all__ = [
    "load_and_prepare_data",
    "bilateral_average",
    "define_groups",
    "compute_spearman_matrices",
    "fisher_z_transform",
]
