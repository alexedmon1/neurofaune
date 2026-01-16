"""
Registration module for neurofaune.

Provides utilities for registering partial-coverage modalities to full anatomical
images and propagating atlas labels.
"""

from neurofaune.registration.slice_correspondence import (
    SliceCorrespondenceFinder,
    SliceCorrespondenceResult,
    IntensityMatcher,
    LandmarkDetector,
    find_slice_correspondence,
)

from neurofaune.registration.qc_visualization import (
    plot_slice_correspondence,
    plot_slice_correspondence_detailed,
    plot_registration_quality,
    create_registration_report,
    create_checkerboard,
    create_edge_overlay,
)

__all__ = [
    # Slice correspondence
    'SliceCorrespondenceFinder',
    'SliceCorrespondenceResult',
    'IntensityMatcher',
    'LandmarkDetector',
    'find_slice_correspondence',
    # QC visualization
    'plot_slice_correspondence',
    'plot_slice_correspondence_detailed',
    'plot_registration_quality',
    'create_registration_report',
    'create_checkerboard',
    'create_edge_overlay',
]
