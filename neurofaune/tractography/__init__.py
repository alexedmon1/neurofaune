"""Structural connectivity: fibre modelling, tractography, connectomes, FBA.

Two pipelines share one data-adequacy gate.

**MRtrix3 (preferred for multi-shell data)** — MSMT-CSD fibre orientation
distributions feeding either a SIFT2-weighted structural connectome or
fixel-based analysis::

    from neurofaune.tractography import (
        assess_tractography_adequacy, run_msmt_csd, run_tractography,
        build_5tt_from_probseg, build_connectome,
    )

    fods = run_msmt_csd(dwi, bval, bvec, mask, out_dir, "sub-01", "ses-1")
    tck  = run_tractography(fods["wm_fod"], out_dir, "sub-01", "ses-1",
                            fivett_file=fivett, n_streamlines="1M")
    cn   = build_connectome(tck["tractogram"], parcellation, out_dir,
                            "sub-01", "ses-1", weights=tck["weights"])

**FSL (single-shell, or as an independent cross-check)** — BEDPOSTX
ball-and-sticks with probtrackx2 in network mode::

    from neurofaune.tractography import run_bedpostx, run_probtrackx_connectome

Both refuse acquisitions that cannot support the model they implement, rather
than emitting a matrix that looks valid.
"""

from neurofaune.tractography.adequacy import (
    InadequateDataError,
    TractographyAdequacy,
    assess_tractography_adequacy,
    max_feasible_lmax,
    sh_coefficients,
)
from neurofaune.tractography.connectome import build_connectome, compute_node_coverage
from neurofaune.tractography.fivett import build_5tt_from_probseg, check_5tt, warp_5tt
from neurofaune.tractography.fixel import (
    build_fod_template,
    compute_fixel_metrics,
    compute_group_response,
    register_fod_to_template,
    run_fixel_stats,
)
from neurofaune.tractography.layout import (
    connectome_dir,
    fixel_dir,
    session_dir,
    stage_dir,
    stats_dir,
    template_dir,
    work_dir,
)
from neurofaune.tractography.fsl import (
    ball_and_sticks_parameters,
    build_roi_seed_masks,
    max_supported_fibres,
    run_bedpostx,
    run_probtrackx_connectome,
)
from neurofaune.tractography.mrtrix import (
    MRtrixNotFoundError,
    convert_to_mif,
    find_mrtrix_bin,
    require_mrtrix,
    run_msmt_csd,
)
from neurofaune.tractography.tractogram import run_tractography

__all__ = [
    # adequacy
    "assess_tractography_adequacy",
    "TractographyAdequacy",
    "InadequateDataError",
    "sh_coefficients",
    "max_feasible_lmax",
    # mrtrix front end
    "run_msmt_csd",
    "convert_to_mif",
    "find_mrtrix_bin",
    "require_mrtrix",
    "MRtrixNotFoundError",
    # 5TT
    "build_5tt_from_probseg",
    "check_5tt",
    "warp_5tt",
    # tractography + connectome
    "run_tractography",
    "build_connectome",
    "compute_node_coverage",
    # fixel-based analysis
    "compute_group_response",
    "build_fod_template",
    "register_fod_to_template",
    "compute_fixel_metrics",
    "run_fixel_stats",
    # layout
    "stage_dir",
    "session_dir",
    "work_dir",
    "template_dir",
    "fixel_dir",
    "stats_dir",
    "connectome_dir",
    # FSL path
    "run_bedpostx",
    "run_probtrackx_connectome",
    "build_roi_seed_masks",
    "ball_and_sticks_parameters",
    "max_supported_fibres",
]
