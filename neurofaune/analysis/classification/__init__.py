"""
Multivariate Group Classification Module.

Complements TBSS (mass-univariate) and CovNet (correlation structure) with
multivariate approaches: PERMANOVA omnibus test, LDA, cross-validated
classification (SVM + logistic), and PCA visualization.

For dose-response regression (SVR, Ridge, PLS), see neurofaune.analysis.regression.
"""

from neurofaune.analysis.classification.classifiers import run_classification
from neurofaune.analysis.classification.data_prep import prepare_classification_data
from neurofaune.analysis.classification.lda import run_lda
from neurofaune.analysis.classification.omnibus import run_manova, run_permanova
from neurofaune.analysis.classification.pca import run_pca

__all__ = [
    "prepare_classification_data",
    "run_permanova",
    "run_manova",
    "run_lda",
    "run_classification",
    "run_pca",
]
