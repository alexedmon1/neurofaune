"""Connectome module: ROI extraction, connectivity matrices, and network analysis.

Submodules:
    roi_extraction  — Atlas-based ROI value and timeseries extraction
    functional      — BOLD functional connectivity (Pearson, Fisher z)
    matrices        — Covariance network correlation matrices (Spearman, bilateral averaging)
    nbs             — Network-Based Statistic (edge-level permutation testing)
    graph_metrics   — Graph theory metrics (efficiency, clustering, modularity)
    whole_network   — Whole-network similarity (Mantel, Frobenius, spectral)
    visualization   — Heatmaps, difference matrices, network plots
    pipeline        — CovNetAnalysis orchestrator class
"""

from .pipeline import CovNetAnalysis

__all__ = ["CovNetAnalysis"]
