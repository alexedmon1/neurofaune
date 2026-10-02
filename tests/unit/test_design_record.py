"""Every randomise design says what it tests: design.json + design.md beside it."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from neurofaune.analysis.stats import design_record as dr

X = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=float)
COLUMNS = [("Intercept", "1 for every animal", "intercept"),
           ("group_cuprizone", "1 if the animal is cuprizone, 0 if control", "group"),
           ("age_days", "age at scan in days, mean-centred", "covariate")]
CONTRASTS = [
    {"name": "cuprizone>control", "vector": [0, 1, 0], "test_kind": "two_group",
     "tests": "cuprizone mean > control mean, adjusting for age",
     "group_a": "cuprizone", "group_b": "control"},
    {"name": "age+", "vector": [0, 0, 1], "test_kind": "regression",
     "tests": "positive association with age"},
]
ROWS = ["sub-1C", "sub-2C", "sub-1X", "sub-2X"]


def _write(tmp_path, **kw):
    return dr.write_design(tmp_path, X, COLUMNS, CONTRASTS, rows=ROWS,
                           groups={"control": 2, "cuprizone": 2},
                           summary="Does cuprizone change FA?", **kw)


def test_write_design_writes_matrices_and_what_they_mean(tmp_path):
    paths = _write(tmp_path)
    mat, _ = dr.read_vest(paths["design_mat"])
    con, names = dr.read_vest(paths["design_con"])
    assert np.allclose(mat, X)
    assert names == ["cuprizone>control", "age+"], "contrast names are in design.con"
    record = json.loads(paths["design_json"].read_text())
    assert record["schema"] == dr.SCHEMA
    assert record["rows"]["ids"] == ROWS
    assert record["columns"][1]["meaning"].startswith("1 if the animal is cuprizone")
    assert record["contrasts"][0]["group_a"] == "cuprizone"
    assert record["written_by"]["Name"] == "neurofaune"
    md = paths["design_md"].read_text()
    assert "`group_cuprizone`" in md and "cuprizone mean > control mean" in md
    assert "two_group (cuprizone vs control)" in md
    assert dr.validate_design_record(record, design_mat=paths["design_mat"],
                                     design_con=paths["design_con"]) == []


def test_f_tests_are_written_and_checked(tmp_path):
    paths = _write(tmp_path, ftests=[{"name": "any", "contrasts": [1, 2],
                                      "tests": "group or age effect"}])
    fts, _ = dr.read_vest(paths["design_fts"])
    assert fts.tolist() == [[1, 1]]
    record = dr.read_design_record(tmp_path)
    assert dr.validate_design_record(record, design_fts=paths["design_fts"]) == []


@pytest.mark.parametrize("break_it, expect", [
    (lambda c, k: c.__setitem__(1, ("group_cuprizone", "Design column")), "no meaning"),
    (lambda c, k: k[0].__setitem__("tests", ""), "no sentence"),
    (lambda c, k: k[0].pop("group_b"), "names group_a and group_b"),
    (lambda c, k: k[1].__setitem__("test_kind", "difference"), "test_kind"),
    (lambda c, k: k[1].__setitem__("vector", [0, 1]), "weights"),
])
def test_a_vague_or_incomplete_design_is_never_written(tmp_path, break_it, expect):
    columns = list(COLUMNS)
    contrasts = [dict(c) for c in CONTRASTS]
    break_it(columns, contrasts)
    with pytest.raises(dr.DesignRecordError, match=expect):
        dr.write_design(tmp_path, X, columns, contrasts, rows=ROWS)
    assert not (tmp_path / "design.mat").exists()


def test_a_null_run_says_its_labels_are_shuffled_and_what_each_row_is(tmp_path):
    dr.write_design(tmp_path, X, COLUMNS, CONTRASTS, rows=ROWS[::-1], label_ids=ROWS,
                    data={"file": "data.nii.gz", "meaning": "per-animal FA change, p90 minus p60"})
    record = dr.read_design_record(tmp_path)
    assert record["rows"]["label_ids"] == ROWS and record["rows"]["ids"] == ROWS[::-1]
    md = (tmp_path / "design.md").read_text()
    assert "deliberately shuffled" in md and "per-animal FA change, p90 minus p60" in md
    with pytest.raises(dr.DesignRecordError, match="label_ids"):
        dr.write_design(tmp_path / "x", X, COLUMNS, CONTRASTS, rows=ROWS, label_ids=ROWS[:3])
    with pytest.raises(dr.DesignRecordError, match="data: no meaning"):
        dr.write_design(tmp_path / "y", X, COLUMNS, CONTRASTS, rows=ROWS, data={"file": "d.nii.gz"})


def test_rows_must_be_named(tmp_path):
    with pytest.raises(dr.DesignRecordError, match="rows"):
        dr.write_design(tmp_path, X, COLUMNS, CONTRASTS)


def test_a_record_that_contradicts_its_matrices_is_refused(tmp_path):
    _write(tmp_path)
    record = dr.read_design_record(tmp_path)
    record["contrasts"][0]["vector"] = [0, -1, 0]       # describes the other direction
    with pytest.raises(dr.DesignRecordError, match="differ from the record"):
        dr.write_design_record(tmp_path, record)
    record = dr.read_design_record(tmp_path)
    record["contrasts"][0]["name"] = "control>cuprizone"
    with pytest.raises(dr.DesignRecordError, match="names its contrasts"):
        dr.write_design_record(tmp_path, record)


def test_attach_copies_the_design_and_its_record_into_the_run(tmp_path):
    design = tmp_path / "design"
    _write(design)
    out = tmp_path / "run"
    record = dr.attach_design(out, design / "design.mat", design / "design.con")
    assert record["contrasts"][0]["name"] == "cuprizone>control"
    for name in ("design.mat", "design.con", "design.json", "design.md"):
        assert (out / name).exists(), name


def test_attach_warns_when_a_design_says_nothing(tmp_path, caplog):
    design = tmp_path / "design"
    design.mkdir()
    dr.write_vest(design / "design.mat", X, "mat")
    dr.write_vest(design / "design.con", np.array([[0, 1, 0]]), "con")
    with caplog.at_level(logging.WARNING):
        record = dr.attach_design(tmp_path / "run", design / "design.mat", design / "design.con")
    assert record is None
    assert "not described" in caplog.text
    assert (tmp_path / "run" / "design.mat").exists(), "the matrices still travel"
    assert not (tmp_path / "run" / "design.json").exists()


def test_attach_refuses_a_record_that_contradicts_its_matrices(tmp_path):
    design = tmp_path / "design"
    _write(design)
    dr.write_vest(design / "design.con", np.array([[0, -1, 0], [0, 0, 1]]), "con",
                  ["cuprizone>control", "age+"])
    with pytest.raises(dr.DesignRecordError):
        dr.attach_design(tmp_path / "run", design / "design.mat", design / "design.con")


def test_runs_without_records_are_found(tmp_path):
    _write(tmp_path / "described")
    bare = tmp_path / "bare"
    bare.mkdir()
    dr.write_vest(bare / "design.con", np.array([[1.0]]), "con")
    assert list(dr.iter_runs_without_records(tmp_path)) == [bare]


def test_read_vest_reads_contrast_names_and_fsl_style_files(tmp_path):
    p = tmp_path / "design.con"
    p.write_text("/ContrastName1\ta>b\n/ContrastName2 b>a\n/NumWaves 2\n/NumContrasts 2\n"
                 "/PPheights 1 1\n\n/Matrix\n1 -1\n-1 1\n")
    m, names = dr.read_vest(p)
    assert names == ["a>b", "b>a"] and m.tolist() == [[1, -1], [-1, 1]]


def test_run_randomise_attaches_the_design_and_records_the_call(tmp_path, monkeypatch):
    import nibabel as nib

    from neurofaune.analysis.stats import randomise_wrapper as rw

    design = tmp_path / "design"
    _write(design)
    data = tmp_path / "all.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4), np.float32), np.eye(4)), data)
    seen = {}
    monkeypatch.setattr(rw, "check_fsl_available", lambda: True)

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return None
    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    out = tmp_path / "randomise_FA"
    result = rw.run_randomise(data, design / "design.mat", design / "design.con", out,
                              n_permutations=10, tfce_2d=True, seed=1)
    assert result["design_record"]["contrasts"][1]["name"] == "age+"
    assert (out / "design.json").exists() and (out / "design.md").exists()
    call = json.loads((out / "randomise.json").read_text())
    assert call["described"] is True
    assert call["inference"] == "TFCE, 2D (--T2, skeleton)"
    assert call["command"] == seen["cmd"]
