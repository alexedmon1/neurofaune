"""Deprecated: use neurofaune.network.covnet instead."""

from neurofaune.network.covnet import CovNetAnalysis  # noqa: F401
from neurofaune.network.covnet.whole_network import (  # noqa: F401
    frobenius_distance,
    mantel_test,
    spectral_divergence,
    whole_network_test,
)
from neurofaune.network.matrices import (  # noqa: F401
    compute_spearman_matrices,
    define_groups,
    fisher_z_transform,
    load_and_prepare_data,
    spearman_matrix,
)

__all__ = [
    "CovNetAnalysis",
    "load_and_prepare_data",
    "define_groups",
    "compute_spearman_matrices",
    "fisher_z_transform",
    "spearman_matrix",
    "mantel_test",
    "frobenius_distance",
    "spectral_divergence",
    "whole_network_test",
]
