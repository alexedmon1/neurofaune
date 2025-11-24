"""
Template building and management for neurofaune.

This module provides utilities for:
1. Building age-specific templates from preprocessed data
2. Automatic template→SIGMA registration (T2w only)
3. Subject selection based on QC metrics
4. Subject-to-template registration
5. Within-subject cross-modal registration
6. Label propagation
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
    'propagate_labels_to_subject'
]
