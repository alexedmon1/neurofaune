"""ROI-level decoding on one harness: folds grouped by animal, a permutation null, and
everything fitted inside the fold.

Moved from murinet (``decode.py``, ``design.py``; origin/main ade3d38). The rules are in
:mod:`neurofaune.analysis.decoding.harness`.
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

__all__ = [
    "DecodeSettings",
    "build_design",
    "evaluate",
    "make_pipeline",
    "morphometry_features",
    "roi_stats_features",
    "run_decoding",
    "session_table",
    "threshold_scan",
    "write_decoding_results",
]
