"""
MVPA (Multi-Voxel Pattern Analysis) module.

Whole-brain decoding and searchlight mapping using voxel-level SIGMA-space
DTI metrics. Complements ROI-level classification with spatially-resolved
pattern analysis.

Public API:
    discover_sigma_images: Find SIGMA-space NIfTIs in derivatives tree.
    load_mvpa_data: Stack images into 4D volume for analysis.
    load_design: Load NeuroAider design matrix.
    align_data_to_design: Match image data to design subject ordering.
    run_whole_brain_decoding: Decoder-based whole-brain classification/regression.
    run_searchlight: SearchLight mapping with FWER correction.
"""

from neurofaune.analysis.mvpa.data_loader import (
    align_data_to_design,
    discover_sigma_images,
    load_design,
    load_mvpa_data,
)
from neurofaune.analysis.mvpa.searchlight import run_searchlight
from neurofaune.analysis.mvpa.whole_brain import run_whole_brain_decoding

__all__ = [
    "discover_sigma_images",
    "load_mvpa_data",
    "load_design",
    "align_data_to_design",
    "run_whole_brain_decoding",
    "run_searchlight",
]
