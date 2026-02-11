"""
Dose-Response Regression Module.

Cross-validated regression (SVR, Ridge, PLS) with permutation testing for
continuous dose-response relationships. Treats dose as ordinal and tests
whether joint ROI patterns predict dose level using LOOCV.

Complements classification (discrete group discrimination) by testing for
a graded, continuous dose-response trend.
"""

from neurofaune.analysis.classification.data_prep import prepare_classification_data
from neurofaune.analysis.regression.dose_response import run_regression

__all__ = [
    "prepare_classification_data",
    "run_regression",
]
