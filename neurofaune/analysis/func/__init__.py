"""
Resting-state fMRI analysis modules.

Submodules:
    falff: Amplitude of Low-Frequency Fluctuations (ALFF/fALFF) analysis
    reho: Regional Homogeneity (ReHo) analysis
    connectivity: ROI-to-ROI functional connectivity
"""

from .falff import compute_falff_map, compute_falff_zscore
from .reho import compute_reho_map, compute_reho_zscore
from .connectivity import compute_fc_matrix, extract_roi_timeseries, save_fc_matrix

__all__ = [
    "compute_falff_map",
    "compute_falff_zscore",
    "compute_reho_map",
    "compute_reho_zscore",
    "compute_fc_matrix",
    "extract_roi_timeseries",
    "save_fc_matrix",
]
