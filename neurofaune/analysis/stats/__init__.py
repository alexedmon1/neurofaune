"""
Statistical Analysis Utilities

Provides wrappers for FSL statistical tools:
    randomise_wrapper: FSL randomise execution with TFCE
    cluster_report: Cluster extraction and anatomical labeling
    readout: Every test and cluster as rows -- effect size + CI, direction,
             extent, named regions, nulls included
"""

from neurofaune.analysis.stats.randomise_wrapper import run_randomise
from neurofaune.analysis.stats.cluster_report import generate_cluster_report
from neurofaune.analysis.stats.readout import Atlas, read_randomise
