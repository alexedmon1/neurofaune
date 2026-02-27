"""
TBSS Analysis Pipeline for Rodent DTI

Implements Tract-Based Spatial Statistics adapted for rodent brains:
- Custom registration chain (FA→T2w→Template→SIGMA) instead of FSL tbss_2_reg
- Tissue-informed WM masking to remove exterior WM artifacts
- SIGMA atlas for anatomical labeling
- Slice-level QC with validity masks for partial-coverage DTI

Modules:
    prepare_tbss: Data preparation, skeleton creation, metric projection
    run_tbss_stats: Statistical analysis with FSL randomise
    slice_qc: Slice-level QC and validity mask creation
    reporting: SIGMA-based cluster reports and HTML generation
"""

from neurofaune.analysis.tbss.prepare_tbss import prepare_tbss_data
from neurofaune.analysis.tbss.prepare_template_tbss import prepare_template_tbss_data
from neurofaune.analysis.tbss.run_tbss_stats import run_tbss_statistical_analysis
from neurofaune.analysis.tbss.slice_qc import apply_slice_masking
from neurofaune.analysis.tbss.reporting import generate_tbss_report
