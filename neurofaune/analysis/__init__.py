"""
Neurofaune Analysis Module

Group-level analysis pipelines for rodent MRI data.

Submodules:
    func: Resting-state fMRI metrics (ReHo, fALFF)
    tbss: Tract-Based Spatial Statistics for DTI metrics
    vbm: Voxel-based morphometry preparation (Jacobians, tissue warping)
    stats: Statistical utilities (randomise, cluster reporting, effect size, ICC)

Moved to top-level modules (re-export shims remain here):
    covnet -> neurofaune.network.covnet
    roi -> neurofaune.network.roi_extraction
    reporting -> neurofaune.reporting

Moved to MURINET (a separate repo, not importable from here):
    mvpa: Multi-Voxel Pattern Analysis (whole-brain decoding, searchlight)

    neurofaune owns the mass-univariate side -- randomise, TBSS, VBM, cluster and
    effect-size reporting. Anything that trains or cross-validates a model lives in
    murinet, which reads this package's outputs. The version removed from here also
    cross-validated with StratifiedKFold/KFold and never passed ``groups=``, so an
    animal contributing several sessions could land in train and test at once;
    murinet's replacement groups folds on animal. Designs built for VBM/TBSS are
    read directly by ``murinet.design``, so one design drives both analyses.
"""
