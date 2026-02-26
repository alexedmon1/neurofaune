"""Deprecated: use neurofaune.connectome instead."""

from neurofaune.connectome import CovNetAnalysis  # noqa: F401
from neurofaune.connectome.matrices import (  # noqa: F401
    bilateral_average,
    compute_spearman_matrices,
    define_groups,
    fisher_z_transform,
    load_and_prepare_data,
    spearman_matrix,
)
from neurofaune.connectome.whole_network import (  # noqa: F401
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
