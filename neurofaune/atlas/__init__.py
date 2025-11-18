"""
SIGMA Atlas Management Module.

Handles:
- Loading SIGMA rat brain atlas templates and labels
- Slice extraction for modality-specific registration
- ROI and tissue mask access
- Atlas metadata and parcellation information
"""

from neurofaune.atlas.manager import AtlasManager
from neurofaune.atlas.slice_extraction import extract_slices, get_slice_range

__all__ = [
    'AtlasManager',
    'extract_slices',
    'get_slice_range',
]
