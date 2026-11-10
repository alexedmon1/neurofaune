"""Decoding harness: animal-grouped folds, a permutation null, everything inside the fold."""
import numpy as np
import pandas as pd
import pytest

from neurofaune.analysis.decoding import (
    DecodeSettings,
    build_design,
    evaluate,
    make_pipeline,
    roi_stats_features,
    run_decoding,
    session_table,
    threshold_scan,
)
from neurofaune.analysis.decoding.harness import _animal_shuffles
from neurofaune.analysis.decoding.results import modality_of
from neurofaune.results import check
from neurofaune.study_design import StudyDesign

SESSIONS = pd.DataFrame({
    "subject": ["sub-1C_ses-1", "sub-1C_ses-2", "sub-1X_ses-1", "sub-1X_ses-2", "sub-2X_ses-1"],
    "animal": ["sub-1C", "sub-1C", "sub-1X", "sub-1X", "sub-2X"],
    "session": ["ses-1", "ses-2", "ses-1", "ses-2", "ses-1"],
    "group": ["control", "control", "treatment", "treatment", "treatment"],
})
FEATURES = pd.DataFrame(
    {"f1": [1.0, 4.0, 10.0, 12.0, 7.0], "f2": [0.0, 1.0, 5.0, 9.0, 3.0]},
    index=SESSIONS["subject"],
)
FAST = DecodeSettings(pca_components=None, permutations=0, workers=1)


def test_delta_subtracts_within_animal():
    X, _, groups = build_design(FEATURES, SESSIONS, "delta", "1", "2")
    assert list(X.index) == ["sub-1C", "sub-1X"]
    assert X.loc["sub-1C", "f1"] == pytest.approx(3.0)
    assert X.loc["sub-1X", "f2"] == pytest.approx(4.0)
    assert list(groups) == ["sub-1C", "sub-1X"]


def test_delta_drops_animals_missing_a_session():
    X, _, _ = build_design(FEATURES, SESSIONS, "delta", "1", "2")
    assert "sub-2X" not in X.index


def test_delta_labels_follow_the_animal():
    _, y, _ = build_design(FEATURES, SESSIONS, "delta", "1", "2")
    assert y.loc["sub-1C"] == "control" and y.loc["sub-1X"] == "treatment"


def test_delta_row_count_is_animals_not_sessions():
    X, labels, groups = build_design(FEATURES, SESSIONS, "delta", "1", "2")
    assert len(X) == len(labels) == len(groups) == groups.nunique()


def test_cross_sectional_selects_one_session():
    X, labels, groups = build_design(FEATURES, SESSIONS, "cross_sectional", "1", "2")
    assert sorted(X.index) == ["sub-1C_ses-2", "sub-1X_ses-2"]
    assert sorted(groups) == ["sub-1C", "sub-1X"]
    assert sorted(labels) == ["control", "treatment"]


def test_session_may_be_given_with_its_prefix():
    a, _, _ = build_design(FEATURES, SESSIONS, "delta", "ses-1", "ses-2")
    b, _, _ = build_design(FEATURES, SESSIONS, "delta", "1", "2")
    pd.testing.assert_frame_equal(a, b)


def test_no_shared_session_is_an_error():
    with pytest.raises(ValueError, match="no animal has both"):
        build_design(FEATURES, SESSIONS, "delta", "1", "3")


def test_bad_design_name_is_rejected():
    with pytest.raises(ValueError, match="delta"):
        build_design(FEATURES, SESSIONS, "longitudinal", "1", "2")


def test_delta_is_antisymmetric():
    fwd, _, _ = build_design(FEATURES, SESSIONS, "delta", "1", "2")
    rev, _, _ = build_design(FEATURES, SESSIONS, "delta", "2", "1")
    assert np.allclose(fwd.values, -rev.loc[fwd.index].values)


def test_delta_carries_a_continuous_target():
    sessions = SESSIONS.assign(exposure=[0.0, 0.0, 12.5, 12.5, 3.0])
    _, y, _ = build_design(FEATURES, sessions, "delta", "1", "2", target="exposure")
    assert list(y) == [0.0, 12.5]


def test_unknown_target_column_is_rejected():
    with pytest.raises(KeyError, match="exposure"):
        build_design(FEATURES, SESSIONS, "delta", "1", "2", target="exposure")


def test_regression_pipeline_uses_ridge():
    settings = DecodeSettings(alpha=2.0, pca_components=None)
    steps = dict(make_pipeline(settings, 9, analysis_type="regression").steps)
    assert "ridge" in steps and "svm" not in steps
    assert steps["ridge"].alpha == 2.0


def test_classification_pipeline_uses_svm():
    steps = dict(make_pipeline(DecodeSettings(pca_components=None), 9).steps)
    assert "svm" in steps and "ridge" not in steps


def test_pca_is_capped_below_the_smallest_training_fold():
    steps = dict(make_pipeline(DecodeSettings(pca_components=500), 9).steps)
    assert steps["pca"].n_components == 8


def test_pca_is_capped_at_the_feature_count():
    steps = dict(make_pipeline(DecodeSettings(pca_components=20), 33, n_features=5).steps)
    assert steps["pca"].n_components == 5


def _separable(n_animals=12, seed=0, shift=3.0):
    rng = np.random.default_rng(seed)
    animals = [f"sub-{i:02d}" for i in range(n_animals)]
    y = pd.Series(["control" if i % 2 else "treated" for i in range(n_animals)], index=animals)
    X = pd.DataFrame(rng.normal(size=(n_animals, 5)), index=animals)
    X.loc[y == "treated", 0] += shift
    return X, y, pd.Series(animals, index=animals)


def test_groups_are_required():
    X, y, _ = _separable()
    with pytest.raises(ValueError, match="needs groups"):
        evaluate(X, y, None, FAST)
    with pytest.raises(ValueError, match="groups for"):
        evaluate(X, y, pd.Series(["a"]), FAST)


def test_k_fold_classification_folds_hold_both_classes():
    """GroupKFold deals animals by sorted id; with ids alternating by group, each of its test
    folds held one class. The harness stratifies classification folds."""
    from neurofaune.analysis.decoding.harness import _splitter

    groups = np.array([f"sub-{i:02d}" for i in range(20)])
    y = np.array([0, 1] * 10)
    cv, n = _splitter(DecodeSettings(cv_folds=4), groups, stratify=True)
    assert n == 4
    assert all(set(y[test]) == {0, 1} for _, test in cv.split(np.zeros(20), y, groups=groups))


def test_one_animal_out_is_not_stratified():
    from sklearn.model_selection import LeaveOneGroupOut

    from neurofaune.analysis.decoding.harness import _splitter

    cv, n = _splitter(DecodeSettings(cv_folds="loo"), np.arange(10), stratify=True)
    assert isinstance(cv, LeaveOneGroupOut) and n == 10


def test_an_animal_is_never_in_train_and_test(monkeypatch):
    """Its second session must not help: an animal's rows are identical, labels random."""
    rng = np.random.default_rng(1)
    animals = [f"sub-{i:02d}" for i in range(16)]
    base = pd.DataFrame(rng.normal(size=(16, 8)), index=animals)
    labels = pd.Series(["a", "b"] * 8, index=animals)
    rows = pd.concat([base, base])
    groups = pd.Series(list(animals) * 2)
    y = pd.Series(list(labels) * 2)
    rows.index = groups.index = y.index = range(32)

    scores, _ = evaluate(rows, y, groups, FAST)

    assert scores["balanced_accuracy"] < 0.9
    assert scores["n_animals"] == 16 and scores["n_samples"] == 32


def test_separable_groups_beat_the_null():
    # 16 animals: at 12 there are so few labellings that 50 draws can hold the truth's inverse.
    # No PCA: it keeps variance, not labels, so 3 of 5 components can drop the one with the signal.
    X, y, groups = _separable(n_animals=16)
    settings = DecodeSettings(pca_components=None, permutations=50, workers=1)
    scores, predictions = evaluate(X, y, groups, settings)
    assert scores["balanced_accuracy"] > 0.8
    assert scores["p_permutation"] < 0.05
    assert abs(scores["null_mean"] - 0.5) < 0.2
    assert len(predictions) == 16


def test_noise_does_not_beat_the_null():
    X, y, groups = _separable(shift=0.0, seed=3)
    settings = DecodeSettings(pca_components=3, permutations=50, workers=1)
    assert evaluate(X, y, groups, settings)[0]["p_permutation"] > 0.05


def test_score_is_reproducible():
    X, y, groups = _separable()
    settings = DecodeSettings(pca_components=3, permutations=20, workers=1)
    assert evaluate(X, y, groups, settings)[0] == evaluate(X, y, groups, settings)[0]


def test_workers_do_not_change_the_null():
    X, y, groups = _separable()
    one = evaluate(X, y, groups, DecodeSettings(pca_components=3, permutations=20, workers=1))[0]
    two = evaluate(X, y, groups, DecodeSettings(pca_components=3, permutations=20, workers=2))[0]
    assert one["null_mean"] == two["null_mean"]


def test_permutation_moves_labels_between_animals():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    for shuffled in _animal_shuffles(y, groups, 20):
        for animal in "abcd":
            assert len(set(shuffled[groups == animal])) == 1


def test_one_row_per_animal_matches_a_plain_permutation():
    """So results from before the rule was stated are reproduced where it changes nothing."""
    y = np.array([0, 1, 1, 0, 1, 0])
    groups = np.array(list("abcdef"))
    rng = np.random.default_rng(0)
    expected = [rng.permutation(y) for _ in range(5)]
    for got, want in zip(_animal_shuffles(y, groups, 5), expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_target_that_varies_within_an_animal_cannot_be_permuted():
    with pytest.raises(ValueError, match="between animals"):
        _animal_shuffles(np.array([0, 1, 1, 1]), np.array(["a", "a", "b", "b"]), 2)


def test_classes_set_the_positive_class():
    X, y, groups = _separable()
    scores, _ = evaluate(X, y, groups, FAST, classes=["treated", "control"])
    assert scores["positive_class"] == "control"
    assert scores["negative_class"] == "treated"


def test_three_groups_are_refused():
    X, y, groups = _separable()
    y = y.copy()
    y.iloc[0] = "third"
    with pytest.raises(ValueError, match="exactly 2 groups"):
        evaluate(X, y, groups, FAST)


def test_regression_scores_r2():
    X, _, groups = _separable()
    y = pd.Series(X[0].values * 2.0, index=X.index)
    settings = DecodeSettings(pca_components=None, permutations=20, workers=1)
    scores, _ = evaluate(X, y, groups, settings, "regression")
    assert scores["r2"] > 0.5
    assert scores["analysis_type"] == "regression"


def test_threshold_scan_respects_the_minimum_per_side():
    X, _, groups = _separable()
    exposure = pd.Series(np.arange(12.0), index=X.index)
    scan = threshold_scan(X, exposure, groups, DecodeSettings(pca_components=None, permutations=0, workers=1))
    assert len(scan) == 5
    assert scan["threshold"].min() == 3.5 and scan["threshold"].max() == 7.5


def _long_table(path, source_metric="FA", n_animals=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_animals):
        for session, offset in (("ses-1", 0.0), ("ses-2", 1.0 if i % 2 else 0.0)):
            for roi in ("R1", "R2", "R3"):
                for statistic in ("mean", "max"):
                    rows.append({
                        "subject": f"sub-{i:02d}", "session": session, "metric": source_metric,
                        "roi_name": roi, "coverage": 0.2 if roi == "R3" else 1.0,
                        "statistic": statistic, "value": rng.normal() + offset,
                    })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_roi_stats_filter_by_coverage_and_statistic(tmp_path):
    _long_table(tmp_path / "roi_stats_dwi_long.csv")
    sets = roi_stats_features(tmp_path, min_coverage=0.5, statistics=["mean"])
    assert list(sets) == ["roi_stats|dwi:FA"]
    assert sets["roi_stats|dwi:FA"].shape == (24, 2)


def test_same_metric_from_two_sources_is_two_feature_sets(tmp_path):
    _long_table(tmp_path / "roi_stats_msme_long.csv", "MWF", seed=1)
    _long_table(tmp_path / "roi_stats_msme_common_long.csv", "MWF", seed=2)
    assert set(roi_stats_features(tmp_path)) == {"roi_stats|msme:MWF", "roi_stats|msme_common:MWF"}


def test_combined_set_needs_more_than_one_metric(tmp_path):
    _long_table(tmp_path / "a.csv")
    frames = [pd.read_csv(tmp_path / "a.csv"), pd.read_csv(tmp_path / "a.csv").assign(metric="MD")]
    pd.concat(frames).to_csv(tmp_path / "roi_stats_dwi_long.csv", index=False)
    assert "roi_stats|dwi:combined" in roi_stats_features(tmp_path)


@pytest.mark.parametrize("name,expected", [
    ("roi_stats|dwi:FA", "dwi"),
    ("roi_stats|msme_common:MWF", "msme"),
    ("morphometry_structures", "anat"),
])
def test_modality_of(name, expected):
    assert modality_of(name) == expected


def _design(n_animals=12, second_session=True):
    animals = [f"sub-{i:02d}" for i in range(n_animals)]
    participants = pd.DataFrame({
        "animal": animals,
        "group": ["treated" if i % 2 else "control" for i in range(n_animals)],
    })
    sessions = [{"animal": a, "session_id": "ses-1"} for a in animals]
    if second_session:
        sessions += [{"animal": a, "session_id": "ses-2"} for a in animals]
    return StudyDesign(participants, pd.DataFrame(sessions), control="control")


def _config(tmp_path, **decoding):
    _long_table(tmp_path / "roi_stats_dwi_long.csv")
    return {"decoding": {
        "roi_stats_dir": str(tmp_path),
        "contrasts": [{"name": "change", "session_a": "1", "session_b": "2"}],
        "pca_components": 3,
        "permutations": 20,
        "workers": 1,
        **decoding,
    }}


def test_run_decoding_writes_a_conforming_folder(tmp_path):
    result = run_decoding(_config(tmp_path, name="demo"), tmp_path / "out", design=_design())

    assert result["status"] == "completed"
    assert result["skipped"] == []
    row = result["summary"].iloc[0]
    assert row["positive_class"] == "treated" and row["n_animals"] == 12
    assert len(result["folders"]) == 1
    folder = result["folders"][0]
    assert folder.parts[-4:-1] == ("dwi", "decoding", "demo")
    reports = check(folder)
    assert not [e for r in reports for e in r.errors]
    assert not [w for r in reports for w in r.warnings]
    assert len(pd.read_csv(folder / "tests.csv")) == 1


def test_without_the_results_package_the_table_is_written_and_the_json_skipped(
    tmp_path, monkeypatch, caplog
):
    import neurofaune.results as nr

    monkeypatch.setattr(nr, "AVAILABLE", False)
    result = run_decoding(_config(tmp_path, name="demo"), tmp_path / "out", design=_design())

    folder = result["folders"][0]
    assert folder.parts[-4:-1] == ("dwi", "decoding", "demo")
    assert len(pd.read_csv(folder / "tests.csv")) == 1
    assert sorted(p.name for p in folder.iterdir()) == ["tests.csv"]
    assert "neurofaune[results]" in caplog.text


def test_every_decoder_is_in_the_table_significant_or_not(tmp_path):
    config = _config(tmp_path)
    config["decoding"]["contrasts"].append({"name": "same", "session_a": "1", "session_b": "1"})
    result = run_decoding(config, tmp_path / "out", design=_design())
    tests = pd.read_csv(result["folders"][0] / "tests.csv")
    assert set(tests["contrast"]) == {"change", "same"}


def test_decoder_that_cannot_run_is_reported(tmp_path):
    config = _config(tmp_path)
    config["decoding"]["contrasts"].append({"name": "none", "session_a": "1", "session_b": "9"})
    result = run_decoding(config, tmp_path / "out", design=_design())
    assert [s["contrast"] for s in result["skipped"]] == ["none"]
    assert "no animal has both" in result["skipped"][0]["reason"]


def test_nothing_to_decode_says_what_to_set(tmp_path):
    with pytest.raises(KeyError, match="decoding.contrasts"):
        run_decoding({}, tmp_path, design=_design())
    with pytest.raises(FileNotFoundError, match="roi_stats_dir"):
        run_decoding(
            {"decoding": {"contrasts": [{"session_a": "1", "session_b": "2"}]}},
            tmp_path,
            design=_design(),
        )


def test_session_table_takes_labels_from_a_design_directory(tmp_path):
    (tmp_path / "subject_order.txt").write_text("sub-00_ses-1\nsub-01_ses-1\n")
    (tmp_path / "design.mat").write_text("/NumWaves 2\n/NumPoints 2\n/Matrix\n1 0\n1 1\n")
    (tmp_path / "design_summary.json").write_text(
        '{"columns": ["intercept", "dose_H"], "contrasts": ["H_gt_C"],'
        ' "factors": [{"name": "dose", "reference": "C", "levels": ["C", "H"]}]}'
    )
    table = session_table(_design(), design_dir=tmp_path)
    assert list(table["subject"]) == ["sub-00_ses-1", "sub-01_ses-1"]
    assert list(table["group"]) == ["C", "H"]


def test_session_table_refuses_a_design_directory_that_matches_nothing(tmp_path):
    (tmp_path / "subject_order.txt").write_text("other_1\n")
    (tmp_path / "design.mat").write_text("/Matrix\n1 0\n")
    (tmp_path / "design_summary.json").write_text(
        '{"columns": ["intercept", "dose_H"], "contrasts": ["H_gt_C"],'
        ' "factors": [{"name": "dose", "reference": "C", "levels": ["C", "H"]}]}'
    )
    with pytest.raises(ValueError, match="no session"):
        session_table(_design(), design_dir=tmp_path)
