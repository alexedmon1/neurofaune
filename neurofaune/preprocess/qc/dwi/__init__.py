"""DWI/DTI quality control modules."""

from .eddy_qc import generate_eddy_qc_report
from .dti_qc import generate_dti_qc_report

__all__ = [
    'generate_eddy_qc_report',
    'generate_dti_qc_report'
]
