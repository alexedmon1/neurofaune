"""
Resting-state fMRI analysis modules.

Submodules:
    falff: Amplitude of Low-Frequency Fluctuations (ALFF/fALFF) analysis
    reho: Regional Homogeneity (ReHo) analysis

Note: FC (connectivity) functions have moved to neurofaune.network.functional.
A backwards-compatible shim remains at neurofaune.analysis.func.connectivity.
"""

from .falff import compute_falff_map, compute_falff_zscore
from .melodic import collect_bold_files, run_dual_regression, run_group_melodic
from .reho import compute_reho_map, compute_reho_zscore

__all__ = [
    "compute_falff_map",
    "compute_falff_zscore",
    "compute_reho_map",
    "compute_reho_zscore",
    "collect_bold_files",
    "run_group_melodic",
    "run_dual_regression",
]
