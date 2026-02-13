"""
Quality Control modules for neurofaune preprocessing.

QC Directory Structure:
    {study_root}/qc/
    ├── subjects/{subject}/{session}/   # Per-subject QC
    │   ├── anat/
    │   ├── dwi/
    │   ├── func/
    │   └── msme/
    └── reports/                        # Module-wide reports
        ├── skull_strip_{modality}.html # Omnibus skull strip galleries
        ├── {modality}_batch_summary/   # Batch summaries
        └── templates/{cohort}/         # Template QC
"""

from neurofaune.preprocess.qc.batch_summary import (
    get_subject_qc_dir,
    get_batch_summary_dir,
    get_reports_dir,
    generate_batch_qc_summary,
    generate_exclusion_lists,
    generate_slice_qc_summary,
    generate_skull_strip_omnibus,
    compute_slice_metrics,
    flag_bad_slices,
    collect_qc_metrics,
    detect_outliers,
    BatchQCConfig,
)
from neurofaune.preprocess.qc.skull_strip_qc import (
    calculate_skull_strip_metrics,
    plot_slicesdir_mosaic,
    plot_mask_edge_triplanar,
    skull_strip_html_section,
)

__all__ = [
    'get_subject_qc_dir',
    'get_batch_summary_dir',
    'get_reports_dir',
    'generate_batch_qc_summary',
    'generate_exclusion_lists',
    'generate_slice_qc_summary',
    'generate_skull_strip_omnibus',
    'compute_slice_metrics',
    'flag_bad_slices',
    'collect_qc_metrics',
    'detect_outliers',
    'BatchQCConfig',
    'calculate_skull_strip_metrics',
    'plot_slicesdir_mosaic',
    'plot_mask_edge_triplanar',
    'skull_strip_html_section',
]
