"""Classification/regression feature-set selection.

``feature_set='bilateral'`` must average paired L/R ROIs. Selecting the same
columns as ``'all'`` and skipping ``bilateral_average`` made the two sets
identical except for PCA, so published 'bilateral' runs were not bilateral.
"""
from pathlib import Path

import pandas as pd
import pytest

from neurofaune.network.classification.data_prep import (
    prepare_classification_data,
    prepare_regression_data,
)


def _wide_csv(tmp_path: Path) -> Path:
    rows = []
    for i, (session, dose) in enumerate(
        [("ses-p30", "C"), ("ses-p30", "L"), ("ses-p60", "C"),
         ("ses-p60", "H"), ("ses-p90", "M"), ("ses-p90", "C")]
    ):
        rows.append({
            "subject": f"sub-{i+1:02d}",
            "session": session,
            "dose": dose,
            "sex": "M" if i % 2 == 0 else "F",
            "Hip_L": 0.40 + 0.02 * i,
            "Hip_R": 0.60 + 0.02 * i,
            "Cortex_L": 0.30 + 0.01 * i,
            "Cortex_R": 0.50 + 0.01 * i,
            "Midline": 0.25 + 0.01 * i,
            "territory_Hip": 0.50 + 0.01 * i,
        })
    path = tmp_path / "roi_FA_wide.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_bilateral_averages_lr_pairs_and_keeps_unpaired(tmp_path):
    csv_path = _wide_csv(tmp_path)
    data = prepare_classification_data(
        csv_path, feature_set="bilateral", standardize=False,
    )

    names = data["feature_names"]
    assert "Hip" in names
    assert "Cortex" in names
    assert "Midline" in names
    assert "Hip_L" not in names
    assert "Hip_R" not in names
    assert all(not n.startswith("territory_") for n in names)

    hip = data["X"][:, names.index("Hip")]
    # First row: (0.40 + 0.60) / 2
    assert hip[0] == pytest.approx(0.50)


def test_all_keeps_individual_lr_rois(tmp_path):
    csv_path = _wide_csv(tmp_path)
    data = prepare_classification_data(
        csv_path, feature_set="all", standardize=False,
    )

    names = set(data["feature_names"])
    assert {"Hip_L", "Hip_R", "Cortex_L", "Cortex_R", "Midline"} <= names
    assert "Hip" not in names
    assert all(not n.startswith("territory_") for n in names)


def test_territory_uses_only_territory_columns(tmp_path):
    csv_path = _wide_csv(tmp_path)
    data = prepare_classification_data(
        csv_path, feature_set="territory", standardize=False,
    )
    assert data["feature_names"] == ["territory_Hip"]


def test_bilateral_has_fewer_features_than_all(tmp_path):
    csv_path = _wide_csv(tmp_path)
    bi = prepare_classification_data(csv_path, feature_set="bilateral", standardize=False)
    all_ = prepare_classification_data(csv_path, feature_set="all", standardize=False)
    assert len(bi["feature_names"]) < len(all_["feature_names"])


def test_regression_bilateral_matches_classification(tmp_path):
    csv_path = _wide_csv(tmp_path)
    clf = prepare_classification_data(csv_path, feature_set="bilateral", standardize=False)
    reg = prepare_regression_data(csv_path, feature_set="bilateral", standardize=False)
    assert clf["feature_names"] == reg["feature_names"]
