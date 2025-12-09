"""
Quality control modules for functional MRI preprocessing.
"""

from .motion_qc import generate_motion_qc_report
from .confounds_qc import generate_confounds_qc_report
from .registration_qc import generate_registration_qc

__all__ = [
    'generate_motion_qc_report',
    'generate_confounds_qc_report',
    'generate_registration_qc'
]
