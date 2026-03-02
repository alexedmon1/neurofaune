"""
Voxel-Based Morphometry (VBM) analysis pipeline.

Warps T2w-derived tissue probability maps (GM, WM) to SIGMA space,
applies Jacobian modulation for volume preservation, and prepares
smoothed 4D volumes for FSL randomise.
"""

from neurofaune.analysis.vbm.prepare_vbm import (
    warp_tissue_to_sigma,
    compute_jacobian,
    modulate_tissue,
    smooth_volume,
)
