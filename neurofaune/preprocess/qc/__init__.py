"""
Quality Control modules for neurofaune preprocessing.

QC Directory Structure:
    {study_root}/qc/
    ├── {subject}/{session}/        # Per-subject QC
    │   ├── anat/
    │   ├── dwi/
    │   ├── func/
    │   └── msme/
    └── {modality}_batch_summary/   # Batch summaries
"""

from neurofaune.preprocess.qc.batch_summary import (
    get_subject_qc_dir,
    get_batch_summary_dir,
    generate_batch_qc_summary,
    generate_exclusion_lists,
    generate_slice_qc_summary,
    compute_slice_metrics,
    flag_bad_slices,
    collect_qc_metrics,
    detect_outliers,
    BatchQCConfig,
)

__all__ = [
    'get_subject_qc_dir',
    'get_batch_summary_dir',
    'generate_batch_qc_summary',
    'generate_exclusion_lists',
    'generate_slice_qc_summary',
    'compute_slice_metrics',
    'flag_bad_slices',
    'collect_qc_metrics',
    'detect_outliers',
    'BatchQCConfig',
]
