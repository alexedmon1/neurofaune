"""Pooled analyses refuse repeated animals; dropped sessions reach the outputs.

LeaveOneOut keeps an animal's other sessions in training when one is tested, and
PERMANOVA/MCCA permute rows as though each were its own animal. Pooling across
sessions of the same animals therefore overstates the result. The loader's
zero-ROI session filter used to be visible only in the log.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurofaune.network.classification import ClassificationAnalysis
from neurofaune.network.classification.data_prep import (
    RepeatedAnimalError,
    prepare_classification_data,
    prepare_regression_data,
)
from neurofaune.network.mcca import MCCAAnalysis, load_multiview_data
from neurofaune.network.regression import RegressionAnalysis

ROIS = [f"R{i}_{side}" for i in range(10) for side in "LR"]
DOSES = ["C", "L", "M", "H"]


def _rows(subjects, session, seed):
    rng = np.random.default_rng(seed)
    return [
        {
            "subject": subject,
            "session": session,
            "dose": DOSES[i % 4],
            "sex": "M" if i % 2 else "F",
            **dict(zip(ROIS, rng.uniform(0.2, 0.8, len(ROIS)), strict=True)),
        }
        for i, subject in enumerate(subjects)
    ]


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _animals(start=0, n=24):
    return [f"sub-{i:02d}" for i in range(start, start + n)]


@pytest.fixture
def repeated_csv(tmp_path):
    """The same 24 animals at p30 and p60."""
    return _write(
        tmp_path / "roi_FA_wide.csv",
        _rows(_animals(), "ses-p30", 1) + _rows(_animals(), "ses-p60", 2),
    )


PREPARERS = [prepare_classification_data, prepare_regression_data]


@pytest.mark.parametrize("prepare", PREPARERS)
def test_pooling_repeated_animals_is_refused(repeated_csv, prepare):
    with pytest.raises(RepeatedAnimalError, match=r"24 of 24 animals.*sub-00"):
        prepare(repeated_csv)


@pytest.mark.parametrize("prepare", PREPARERS)
def test_one_cohort_of_repeated_animals_is_allowed(repeated_csv, prepare):
    assert prepare(repeated_csv, cohort_filter="p30")["X"].shape[0] == 24


@pytest.mark.parametrize("prepare", PREPARERS)
def test_pooling_different_animals_is_allowed(tmp_path, prepare):
    csv = _write(
        tmp_path / "roi_FA_wide.csv",
        _rows(_animals(0), "ses-p30", 1) + _rows(_animals(24), "ses-p60", 2),
    )
    assert prepare(csv)["X"].shape[0] == 48


def test_refusal_is_on_rows_left_after_the_zero_roi_filter(tmp_path):
    """An animal whose other session was dropped is no longer repeated."""
    rows = _rows(_animals(), "ses-p30", 1) + _rows(["sub-00"], "ses-p60", 2)
    rows[-1].update(dict.fromkeys(ROIS[:6], 0.0))
    csv = _write(tmp_path / "roi_FA_wide.csv", rows)

    data = prepare_classification_data(csv)

    assert data["X"].shape[0] == 24
    assert data["dropped_sessions"] == [{
        "subject": "sub-00",
        "session": "ses-p60",
        "cohort": "p60",
        "n_zero_rois": 6,
        "n_valid_rois": 20,
        "max_zero_rois": 2,
    }]


def test_dropped_sessions_are_limited_to_the_cohort_analysed(tmp_path):
    rows = _rows(_animals(0), "ses-p30", 1) + _rows(_animals(24), "ses-p60", 2)
    rows[-1].update(dict.fromkeys(ROIS[:6], 0.0))
    csv = _write(tmp_path / "roi_FA_wide.csv", rows)

    assert prepare_classification_data(csv, cohort_filter="p30")["dropped_sessions"] == []
    dropped = prepare_classification_data(csv, cohort_filter="p60")["dropped_sessions"]
    assert [d["subject"] for d in dropped] == ["sub-47"]


def _classification(roi_dir, out):
    return ClassificationAnalysis(roi_dir=roi_dir, output_dir=out, metric="FA")


def _regression(roi_dir, out):
    return RegressionAnalysis(roi_dir=roi_dir, output_dir=out, metric="FA")


def _mcca(roi_dir, out):
    return MCCAAnalysis(roi_dir=roi_dir, output_dir=out, views={"a": ["FA"], "b": ["MD"]})


@pytest.fixture
def repeated_roi_dir(tmp_path):
    for metric, seed in (("FA", 1), ("MD", 3)):
        _write(
            tmp_path / "roi" / f"roi_{metric}_wide.csv",
            _rows(_animals(), "ses-p30", seed) + _rows(_animals(), "ses-p60", seed + 1),
        )
    return tmp_path / "roi"


@pytest.mark.parametrize("make", [_classification, _regression, _mcca])
def test_refused_run_says_so_and_writes_nothing(repeated_roi_dir, tmp_path, make):
    out = tmp_path / "out"
    summary = make(repeated_roi_dir, out).run(cohort=None, feature_set="bilateral")

    assert summary["status"] == "refused"
    assert "more than one row" in summary["reason"]
    assert not any(p.is_file() for p in out.rglob("*"))


@pytest.mark.parametrize("make", [_classification, _regression, _mcca])
def test_refused_run_keeps_existing_results_under_force(repeated_roi_dir, tmp_path, make):
    analysis = make(repeated_roi_dir, tmp_path / "out")
    analysis.force = True
    existing = analysis._result_dir(None, "bilateral") / "summary.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}")

    assert analysis.run(cohort=None, feature_set="bilateral")["status"] == "refused"
    assert existing.exists()


@pytest.fixture
def roi_dir_with_dropped_session(tmp_path):
    for metric, seed in (("FA", 1), ("MD", 3)):
        rows = _rows(_animals(), "ses-p30", seed)
        rows[-1].update(dict.fromkeys(ROIS[:6], 0.0))
        _write(tmp_path / "roi" / f"roi_{metric}_wide.csv", rows)
    return tmp_path / "roi"


@pytest.mark.parametrize("make", [_classification, _regression, _mcca])
def test_dropped_sessions_reach_summary_json(roi_dir_with_dropped_session, tmp_path, make):
    analysis = make(roi_dir_with_dropped_session, tmp_path / "out")
    summary = analysis.run(cohort="p30", feature_set="bilateral", n_permutations=2)

    assert summary["status"] == "completed"
    written = json.loads(
        (analysis._result_dir("p30", "bilateral") / "summary.json").read_text()
    )
    assert {d["subject"] for d in written["dropped_sessions"]} == {"sub-23"}


def test_mcca_dropped_sessions_name_their_metric(roi_dir_with_dropped_session):
    dropped = []
    load_multiview_data(
        roi_dir_with_dropped_session,
        {"a": ["FA"], "b": ["MD"]},
        cohort_filter="p30",
        dropped_sessions=dropped,
    )
    assert [(d["metric"], d["subject"]) for d in dropped] == [
        ("FA", "sub-23"),
        ("MD", "sub-23"),
    ]


def test_mcca_checks_the_rows_common_to_its_views(tmp_path):
    """One view repeats animals, but only one session of each is in every view."""
    _write(
        tmp_path / "roi_FA_wide.csv",
        _rows(_animals(), "ses-p30", 1) + _rows(_animals(), "ses-p60", 2),
    )
    _write(tmp_path / "roi_fALFF_wide.csv", _rows(_animals(), "ses-p60", 3))

    Xs, _, _, metadata = load_multiview_data(
        tmp_path, {"dwi": ["FA"], "func": ["fALFF"]}
    )

    assert Xs[0].shape[0] == 24
    assert set(metadata["session"]) == {"ses-p60"}


def test_mcca_refuses_repeated_animals_common_to_its_views(repeated_roi_dir):
    with pytest.raises(RepeatedAnimalError, match="MCCA"):
        load_multiview_data(repeated_roi_dir, {"a": ["FA"], "b": ["MD"]})
