"""Reading an FSL-style design directory as a label source.

The tricky parts are (a) recovering the *reference* level of a dummy-coded factor,
which by construction has no column of its own, and (b) deciding whether a design is
categorical or continuous. The older implementation this replaces hardcoded one
study's column names (`dose_`, `dose_numeric`); these tests pin the general rule that
a design's contrasts declare what it tests.
"""
import json
from pathlib import Path

import pytest

from neurofaune.analysis.decoding.design_dir import (
    describe,
    infer_target,
    load_design,
    read_fsl_matrix,
)


def _make(tmp_path, summary, matrix, subjects, targets=None) -> Path:
    d = tmp_path / "design"
    d.mkdir()
    (d / "design_summary.json").write_text(json.dumps(summary))
    (d / "subject_order.txt").write_text("\n".join(subjects) + "\n")
    rows = "\n".join(" ".join(str(v) for v in row) for row in matrix)
    (d / "design.mat").write_text(f"/NumWaves {len(matrix[0])}\n/Matrix\n{rows}\n")
    if targets is not None:
        (d / "target_values.json").write_text(json.dumps(targets))
    return d


CATEGORICAL = {
    "columns": ["Intercept", "dose_H", "dose_L", "sex_M"],
    "factors": [{"name": "dose", "coding": "dummy", "reference": "C",
                 "levels": ["C", "H", "L"]},
                {"name": "sex", "coding": "effect", "reference": "F",
                 "levels": ["F", "M"]}],
    "covariates": [],
    "contrasts": ["H_gt_C", "C_gt_H", "L_gt_C"],
}
CAT_MATRIX = [[1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]]   # H, L, reference C
SUBJECTS = ["sub-1_ses-p60", "sub-2_ses-p60", "sub-3_ses-p60"]


def test_reference_level_is_recovered_from_all_zero_indicators(tmp_path):
    """The reference group has no column, so it is the all-zeros row."""
    d = _make(tmp_path, CATEGORICAL, CAT_MATRIX, SUBJECTS)
    assert list(load_design(d)["label"]) == ["H", "L", "C"]


def test_subject_order_is_preserved(tmp_path):
    d = _make(tmp_path, CATEGORICAL, CAT_MATRIX, SUBJECTS)
    assert list(load_design(d)["subject"]) == SUBJECTS


def test_contrasts_pick_the_factor_over_a_nuisance_covariate(tmp_path):
    """dose is tested; weight is a nuisance regressor and must not become the target."""
    summary = dict(CATEGORICAL,
                   columns=["Intercept", "dose_H", "dose_L", "weight"],
                   covariates=[{"name": "weight", "mean_center": True}])
    d = _make(tmp_path, summary, CAT_MATRIX, SUBJECTS)
    assert infer_target(summary) == ("factor", "dose")
    assert list(load_design(d)["label"]) == ["H", "L", "C"]


REGRESSION = {
    "columns": ["Intercept", "dose_numeric", "sex_M"],
    "factors": [{"name": "sex", "coding": "effect", "reference": "F",
                 "levels": ["F", "M"]}],
    "covariates": [{"name": "dose_numeric", "mean_center": True}],
    "contrasts": ["dose_pos", "dose_neg"],
}
REG_MATRIX = [[1, -1.5, 1], [1, 0.0, 0], [1, 1.5, 1]]


def test_contrast_stem_matches_a_longer_covariate_name(tmp_path):
    """`dose_pos` must resolve to the covariate `dose_numeric`, not to the sex factor."""
    assert infer_target(REGRESSION) == ("covariate", "dose_numeric")
    d = _make(tmp_path, REGRESSION, REG_MATRIX, SUBJECTS)
    assert list(load_design(d)["label"]) == pytest.approx([-1.5, 0.0, 1.5])


def test_factor_levels_match_as_tokens_not_substrings():
    """Level 'M' must match `M_gt_C` without matching the 'numeric' in `dose_numeric`."""
    summary = dict(REGRESSION, covariates=[], contrasts=["M_gt_F"])
    assert infer_target(summary) == ("factor", "sex")


def test_target_values_json_takes_precedence(tmp_path):
    """Those are uncentred per-subject values; nothing needs reconstructing."""
    targets = {s: float(i) for i, s in enumerate(SUBJECTS)}
    d = _make(tmp_path, REGRESSION, REG_MATRIX, SUBJECTS, targets=targets)
    assert list(load_design(d)["label"]) == pytest.approx([0.0, 1.0, 2.0])


def test_explicit_factor_overrides_inference(tmp_path):
    d = _make(tmp_path, CATEGORICAL, CAT_MATRIX, SUBJECTS)
    assert list(load_design(d, factor="sex")["label"]) == ["M", "F", "M"]


def test_explicit_target_overrides_inference(tmp_path):
    d = _make(tmp_path, CATEGORICAL, CAT_MATRIX, SUBJECTS)
    assert list(load_design(d, target="dose_H")["label"]) == pytest.approx([1, 0, 0])


def test_row_count_mismatch_is_rejected(tmp_path):
    """design.mat and subject_order.txt must correspond row for row."""
    d = _make(tmp_path, CATEGORICAL, CAT_MATRIX[:2], SUBJECTS)
    with pytest.raises(ValueError, match="row for row"):
        load_design(d)


def test_missing_summary_is_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="not a design directory"):
        load_design(tmp_path / "empty")


def test_unknown_factor_lists_the_available_ones(tmp_path):
    d = _make(tmp_path, CATEGORICAL, CAT_MATRIX, SUBJECTS)
    with pytest.raises(KeyError, match="dose"):
        load_design(d, factor="treatment")


def test_read_fsl_matrix_skips_the_header(tmp_path):
    p = tmp_path / "design.mat"
    p.write_text("/NumWaves 2\n/NumPoints 2\n/Matrix\n1 2\n3 4\n")
    assert read_fsl_matrix(p) == [[1.0, 2.0], [3.0, 4.0]]


def test_describe_states_what_is_being_predicted(tmp_path):
    d = _make(tmp_path, REGRESSION, REG_MATRIX, SUBJECTS)
    assert "regression on dose_numeric" in describe(d)


@pytest.mark.parametrize("name,expected", [
    ("per_pnd_p60", ("factor", "dose")),
    ("dose_response_p60", ("covariate", "dose_numeric")),
    ("log_auc_response_p60", ("covariate", "log_auc")),
])
def test_real_designs_if_present(name, expected):
    """Guard against the real design files drifting; skipped when not mounted."""
    from neurofaune.analysis.decoding.design_dir import load_summary

    d = Path("/mnt/arborea/bpa-rat/analysis/vbm/designs") / name
    if not (d / "design_summary.json").exists():
        pytest.skip("bpa-rat designs not mounted")
    assert infer_target(load_summary(d)) == expected
    assert len(load_design(d)) == load_summary(d)["n_subjects"]
