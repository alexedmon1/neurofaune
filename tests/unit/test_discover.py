"""Dashboard discovery must find network/ outputs when --analysis-root is analysis/."""
import json
from pathlib import Path

from neurofaune.reporting.discover import (
    _discover_classification,
    _discover_mcca,
    _discover_roi,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_discovers_roi_and_mcca_under_sibling_network(tmp_path: Path):
    study = tmp_path / "study"
    analysis = study / "analysis"
    analysis.mkdir(parents=True)
    _write_json(
        study / "network" / "roi" / "extraction_summary.json",
        {"modality": "dwi", "metrics": {"FA": {"n_subjects": 6, "n_regions": 10, "n_territories": 3}}},
    )
    _write_json(
        study / "network" / "mcca" / "mcca_summary.json",
        {
            "views": {"dwi": ["FA"], "func": ["fALFF"]},
            "n_subjects_max": 12,
            "n_significant_canonical_variates": 2,
            "n_significant_dose_associations": 1,
        },
    )

    roi = _discover_roi(analysis)
    assert len(roi) == 1
    assert roi[0]["entry_id"] == "roi_extraction_dwi"
    assert roi[0]["output_dir"] == "../network/roi"

    mcca = _discover_mcca(analysis)
    assert len(mcca) == 1
    assert mcca[0]["analysis_type"] == "mcca"
    assert mcca[0]["summary_stats"]["n_subjects"] == 12


def test_discovers_classification_one_level_down(tmp_path: Path):
    study = tmp_path / "study"
    analysis = study / "analysis"
    analysis.mkdir(parents=True)
    _write_json(
        study / "network" / "classification" / "dwi" / "classification_summary.json",
        {
            "metrics": ["FA"],
            "feature_sets": ["all"],
            "n_subjects": 8,
            "n_significant_permanova": 1,
            "best_classification_accuracy": 0.75,
        },
    )
    entries = _discover_classification(analysis)
    assert len(entries) == 1
    assert entries[0]["entry_id"] == "classification_dwi"
    assert entries[0]["output_dir"] == "../network/classification/dwi"
