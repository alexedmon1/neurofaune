"""
Covariance Network (CovNet) Analysis for ROI-level DTI metrics.

Builds Spearman correlation matrices per experimental group and statistically
compares them using the Network-Based Statistic (NBS), graph-theoretic
metrics, and whole-network similarity tests. Supports bilateral ROI averaging
and territory-level analysis.

Submodules:
    pipeline: CovNetAnalysis class — prepare once, run any test independently
    matrices: Correlation matrix computation and grouping
    nbs: Network-Based Statistic for group comparison
    graph_metrics: Global efficiency, clustering, modularity
    whole_network: Mantel test, Frobenius distance, spectral divergence
    visualization: Heatmaps, difference matrices, network plots
"""

from neurofaune.analysis.covnet.matrices import (
    bilateral_average,
    compute_spearman_matrices,
    define_groups,
    fisher_z_transform,
    load_and_prepare_data,
    spearman_matrix,
)
from neurofaune.analysis.covnet.pipeline import CovNetAnalysis
from neurofaune.analysis.covnet.whole_network import (
    frobenius_distance,
    mantel_test,
    spectral_divergence,
    whole_network_test,
)

__all__ = [
    "CovNetAnalysis",
    "load_and_prepare_data",
    "bilateral_average",
    "define_groups",
    "compute_spearman_matrices",
    "fisher_z_transform",
    "spearman_matrix",
    "mantel_test",
    "frobenius_distance",
    "spectral_divergence",
    "whole_network_test",
]
