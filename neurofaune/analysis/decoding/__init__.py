"""Decoding on one harness: folds grouped by animal, a permutation null, and everything
fitted inside the fold.

ROI-level decoding (``run_decoding``), and searchlight and whole-brain decoding over
atlas-space maps (``run_searchlight``, ``run_whole_brain``, ``coverage_report``). Moved from
murinet (``decode.py``, ``design.py``, ``searchlight.py``; origin/main ade3d38). The rules
are in :mod:`neurofaune.analysis.decoding.harness`.
"""
from neurofaune.analysis.decoding.features import (
    build_design,
    morphometry_features,
    roi_stats_features,
    session_table,
)
from neurofaune.analysis.decoding.harness import (
    DecodeSettings,
    evaluate,
    make_pipeline,
    threshold_scan,
)
from neurofaune.analysis.decoding.pipeline import run_decoding
from neurofaune.analysis.decoding.results import write_decoding_results
from neurofaune.analysis.decoding.searchlight import (
    SearchlightSettings,
    coverage_report,
    run_searchlight,
    run_whole_brain,
)

__all__ = [
    "DecodeSettings",
    "SearchlightSettings",
    "build_design",
    "coverage_report",
    "evaluate",
    "make_pipeline",
    "morphometry_features",
    "roi_stats_features",
    "run_decoding",
    "run_searchlight",
    "run_whole_brain",
    "session_table",
    "threshold_scan",
    "write_decoding_results",
]
