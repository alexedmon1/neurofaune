"""
Resting-state fMRI analysis modules.

Submodules:
    falff: Amplitude of Low-Frequency Fluctuations (ALFF/fALFF) analysis
    reho: Regional Homogeneity (ReHo) analysis

Note: FC (connectivity) functions have moved to neurofaune.connectome.functional.
A backwards-compatible shim remains at neurofaune.analysis.func.connectivity.
"""

from .falff import compute_falff_map, compute_falff_zscore
from .reho import compute_reho_map, compute_reho_zscore

__all__ = [
    "compute_falff_map",
    "compute_falff_zscore",
    "compute_reho_map",
    "compute_reho_zscore",
]
