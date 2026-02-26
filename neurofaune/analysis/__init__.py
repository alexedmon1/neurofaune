"""
Neurofaune Analysis Module

Group-level analysis pipelines for rodent MRI data.

Submodules:
    func: Resting-state fMRI metrics (ReHo, fALFF)
    tbss: Tract-Based Spatial Statistics for DTI metrics
    stats: Statistical utilities (randomise, cluster reporting)
    classification: Multivariate group classification (PERMANOVA, LDA, SVM, PCA)
    regression: Dose-response regression (SVR, Ridge, PLS with permutation testing)
    mvpa: Multi-Voxel Pattern Analysis (whole-brain decoding, searchlight)

Moved to top-level modules (re-export shims remain here):
    covnet -> neurofaune.connectome
    roi -> neurofaune.connectome.roi_extraction
    reporting -> neurofaune.reporting
"""
