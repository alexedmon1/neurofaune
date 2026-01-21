"""
Template building and management for neurofaune.

This module provides utilities for:
1. Building age-specific templates from preprocessed data
2. Automatic template→SIGMA registration (T2w only)
3. Subject selection based on QC metrics
4. Subject-to-template registration
5. Within-subject cross-modal registration
6. Label propagation
7. Template manifest tracking
8. Registration QC (Dice, correlation, overlays)
"""

from neurofaune.templates.builder import (
    build_template,
    select_subjects_for_template,
    extract_mean_bold,
    register_template_to_sigma,
    save_template_metadata
)

from neurofaune.templates.registration import (
    register_subject_to_template,
    register_within_subject,
    apply_transforms,
    propagate_labels_to_subject
)

from neurofaune.templates.manifest import (
    TemplateManifest,
    SubjectTransforms,
    find_template_manifest
)

from neurofaune.templates.anat_registration import (
    register_anat_to_template,
    propagate_atlas_to_anat,
    register_anat_to_sigma_direct,
    propagate_atlas_direct
)

from neurofaune.templates.registration_qc import (
    compute_dice_coefficient,
    compute_correlation,
    compute_registration_metrics,
    generate_registration_qc_figure,
    generate_atlas_overlay_figure,
    generate_template_qc_report
)

__all__ = [
    # Builder functions
    'build_template',
    'select_subjects_for_template',
    'extract_mean_bold',
    'register_template_to_sigma',
    'save_template_metadata',
    # Registration functions
    'register_subject_to_template',
    'register_within_subject',
    'apply_transforms',
    'propagate_labels_to_subject',
    # Manifest
    'TemplateManifest',
    'SubjectTransforms',
    'find_template_manifest',
    # Anatomical registration
    'register_anat_to_template',
    'propagate_atlas_to_anat',
    'register_anat_to_sigma_direct',
    'propagate_atlas_direct',
    # QC functions
    'compute_dice_coefficient',
    'compute_correlation',
    'compute_registration_metrics',
    'generate_registration_qc_figure',
    'generate_atlas_overlay_figure',
    'generate_template_qc_report',
]
