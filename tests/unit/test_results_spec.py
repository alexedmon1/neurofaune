"""The results specification (docs/RESULTS_SPEC.md): schemas, checker, writer, read-out adapter."""
import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofaune.results import (SPEC_VERSION, STANDARD_TERMS, NonConformingResults, check,
                                columns_for, provenance, write_analysis, write_columns)
from neurofaune.results import _schema, spec
from neurofaune.results.__main__ import main as cli

COLUMNS = {
    "measure": {"Description": "measure", "Standard": "measure"},
    "contrast": {"Description": "test id", "Standard": "contrast"},
    "direction": {"Description": "what positive means", "Standard": "tested_direction"},
    "n": {"Description": "subjects", "Standard": "n"},
    "d": {"Description": "Cohen's d", "Standard": "effect_size", "EffectMeasure": "d"},
    "lo": {"Description": "CI low", "Standard": "effect_ci_low"},
    "hi": {"Description": "CI high", "Standard": "effect_ci_high"},
    "p": {"Description": "p", "Standard": "p_value", "PKind": "fdr"},
}


def _analysis(**over):
    a = {"id": "dwi/roi/demo", "title": "Demo", "description": "a demonstration", "analysis_type": "roi",
         "modality": "dwi",
         "measures": ["FA", "MD"], "role": "exploratory", "design": {"n": 20, "groups": {"A": 10, "B": 10}},
         "inference": {"method": "Welch t", "correction": {"p_kind": "fdr", "family": "ROIs x measures",
                                                           "statement": "BH-FDR q < 0.05 over ROIs x measures"}},
         "effect": {"measure": "d", "definition": "pooled-SD Cohen's d", "ci_level": 0.95},
         "tables": [{"path": "tests.tsv", "role": "tests", "rows": "one measure in one ROI",
                     "description": "every test", "headline": True, "n_rows": 2}]}
    a.update(over)
    return a


def _prov():
    return provenance([{"Name": "demo", "Version": "1.0"}], status="completed", start="2026-10-06T00:00:00")


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "an"
    f.mkdir()
    pd.DataFrame({"measure": ["FA", "MD"], "contrast": ["A>B", "A>B"], "direction": ["A > B"] * 2,
                  "n": [20, 20], "d": [0.8, -0.1], "lo": [0.1, -0.8], "hi": [1.5, 0.6],
                  "p": [0.02, 0.7]}).to_csv(f / "tests.tsv", sep="\t", index=False)
    write_columns(f / "tests.tsv", COLUMNS)
    return f


def _errors(folder):
    (rep,) = check(folder)
    return " | ".join(rep.errors)


# ------------------------------------------------------------- the package ---
def test_package_needs_only_the_standard_library():
    code = ("import sys, neurofaune.results, neurofaune.results.__main__; "
            "heavy = {'numpy', 'pandas', 'nibabel', 'scipy', 'nipype'} & set(sys.modules); "
            "print(sorted(heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _walk(schema, where="$"):
    yield where, schema
    for key in ("properties",):
        for k, v in schema.get(key, {}).items():
            yield from _walk(v, f"{where}.{k}")
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            yield from _walk(schema[key], f"{where}.{key}")


@pytest.mark.parametrize("name", ["analysis", "provenance", "columns"])
def test_schemas_use_only_keywords_the_checker_understands(name):
    for where, node in _walk(_schema.load_schema(name)):
        unknown = set(node) - _schema._KEYWORDS - _schema._ANNOTATIONS
        assert not unknown, f"{name} {where}: {unknown}"


def test_schemas_and_vocabulary_agree():
    a, c = _schema.load_schema("analysis"), _schema.load_schema("columns")
    assert tuple(a["properties"]["analysis_type"]["enum"]) == spec.ANALYSIS_TYPES
    assert tuple(a["properties"]["role"]["enum"]) == spec.ROLES
    assert tuple(a["properties"]["tables"]["items"]["properties"]["role"]["enum"]) == spec.TABLE_ROLES
    assert tuple(a["properties"]["maps"]["items"]["properties"]["kind"]["enum"]) == spec.MAP_KINDS
    assert tuple(c["additionalProperties"]["properties"]["PKind"]["enum"]) == spec.P_KINDS
    assert set(c["additionalProperties"]["properties"]["Standard"]["enum"]) == set(STANDARD_TERMS)
    assert set(spec.QUALIFIERS) <= set(c["additionalProperties"]["properties"])


# ------------------------------------------------------------------ writing ---
def test_a_complete_folder_conforms_and_reads_without_neurofaune(folder):
    rep = write_analysis(folder, _analysis(), _prov())
    assert rep.ok and not rep.warnings, (rep.errors, rep.warnings)

    # The "no lightbox" read: plain json + pandas, knowing only the specification.
    a = json.loads((folder / "analysis.json").read_text())
    assert a["spec"] == "neurofaune.results"
    head = next(t for t in a["tables"] if t.get("headline"))
    cols = json.loads((folder / head["path"]).with_suffix(".json").read_text())
    role = {m["Standard"]: c for c, m in cols.items() if "Standard" in m}
    df = pd.read_csv(folder / head["path"], sep="\t" if head["path"].endswith(".tsv") else ",")
    assert df[role["effect_size"]].tolist() == [0.8, -0.1]
    assert cols[role["effect_size"]]["EffectMeasure"] == "d"
    assert cols[role["p_value"]]["PKind"] == "fdr"
    assert a["inference"]["correction"]["statement"].startswith("BH-FDR")
    prov = json.loads((folder / "provenance.json").read_text())
    assert prov["generated_by"][0]["Name"] == "demo" and prov["run"]["status"] == "completed"


def test_strict_writing_refuses_a_nonconforming_folder(folder):
    with pytest.raises(NonConformingResults, match="effect_ci_low"):
        cols = {k: v for k, v in COLUMNS.items() if k != "lo"}
        pd.read_csv(folder / "tests.tsv", sep="\t").drop(columns="lo").to_csv(
            folder / "tests.tsv", sep="\t", index=False)
        write_columns(folder / "tests.tsv", cols)
        write_analysis(folder, _analysis(), _prov())


def test_every_column_needs_a_dictionary_entry():
    with pytest.raises(KeyError, match="mystery"):
        columns_for(["d", "mystery"], COLUMNS)


# ------------------------------------------------------------------ the contract ---
@pytest.mark.parametrize("drop, message", [
    ("direction", "direction"),
    ("d", "magnitude"),
    ("hi", "uncertainty"),
    ("n", "sample size"),
    ("p", "significance"),
    ("measure", "which measure"),
    ("contrast", "which test"),
])
def test_tests_tables_must_state_the_contract(folder, drop, message):
    df = pd.read_csv(folder / "tests.tsv", sep="\t").drop(columns=drop)
    df.to_csv(folder / "tests.tsv", sep="\t", index=False)
    write_columns(folder / "tests.tsv", COLUMNS)
    write_analysis(folder, _analysis(), _prov(), strict=False)
    assert message in _errors(folder)


def test_a_p_value_must_name_its_correction(folder):
    write_columns(folder / "tests.tsv", COLUMNS, {"p": {"Description": "p", "Standard": "p_value"}})
    write_analysis(folder, _analysis(), _prov(), strict=False)
    assert "must state PKind" in _errors(folder)


def test_no_effect_needs_a_reason(folder):
    write_analysis(folder, _analysis(effect={"measure": None}), _prov(), strict=False)
    assert "without a reason" in _errors(folder)


def test_voxelwise_tests_must_state_extent(folder):
    write_analysis(folder, _analysis(analysis_type="vbm"), _prov(), strict=False)
    assert "extent" in _errors(folder)


def test_cluster_tables_must_state_location_and_selection(folder):
    pd.DataFrame({"measure": ["FA"], "contrast": ["A>B"], "n_voxels": [12], "d": [1.2],
                  "p": [0.01]}).to_csv(folder / "clusters.csv", index=False)
    write_columns(folder / "clusters.csv", {**COLUMNS, "n_voxels": {
        "Description": "extent", "Standard": "n_voxels", "Units": "voxels"}},
        {"p": {"Description": "p", "Standard": "p_value", "PKind": "fwe"}})
    tables = _analysis()["tables"] + [{"path": "clusters.csv", "role": "clusters",
                                        "rows": "one cluster", "description": "clusters"}]
    write_analysis(folder, _analysis(tables=tables), _prov(), strict=False)
    errs = _errors(folder)
    assert "location" in errs and "named location" in errs and "selected" in errs


# ---------------------------------------------------------------- structure ---
def test_dictionary_must_match_the_table(folder):
    cols = json.loads((folder / "tests.json").read_text())
    cols["ghost"] = {"Description": "not in the table"}
    del cols["n"]
    (folder / "tests.json").write_text(json.dumps(cols))
    write_analysis(folder, _analysis(), _prov(), strict=False)
    errs = _errors(folder)
    assert "without a dictionary entry: ['n']" in errs and "['ghost']" in errs


def test_listed_paths_must_stay_inside_and_exist(folder):
    maps = [{"path": "../elsewhere.nii.gz", "kind": "stat", "description": "x"},
            {"path": "missing.nii.gz", "kind": "stat", "description": "y"}]
    write_analysis(folder, _analysis(maps=maps), _prov(), strict=False)
    errs = _errors(folder)
    assert "stay inside" in errs and "not on disk" in errs


def test_row_count_is_checked_when_declared(folder):
    tables = [{**_analysis()["tables"][0], "n_rows": 3}]
    write_analysis(folder, _analysis(tables=tables), _prov(), strict=False)
    assert "2 rows, analysis.json says 3" in _errors(folder)


def test_a_standard_term_is_claimed_once(folder):
    write_columns(folder / "tests.tsv", COLUMNS, {"lo": {"Description": "x", "Standard": "effect_size",
                                                         "EffectMeasure": "d"}})
    write_analysis(folder, _analysis(), _prov(), strict=False)
    assert "claimed by both" in _errors(folder)


@pytest.mark.parametrize("version, ok", [(SPEC_VERSION, True), ("0.1.7", True), ("0.2.0", True),
                                         ("0.3.0", False), ("1.0.0", False)])
def test_spec_version(folder, version, ok):
    write_analysis(folder, {**_analysis(), "spec_version": version}, _prov(), strict=False)
    assert ("not readable" not in _errors(folder)) is ok


def test_analysis_folders_do_not_nest(folder):
    write_analysis(folder, _analysis(), _prov())
    inner = folder / "inner"
    inner.mkdir()
    (inner / "tests.tsv").write_bytes((folder / "tests.tsv").read_bytes())
    (inner / "tests.json").write_bytes((folder / "tests.json").read_bytes())
    write_analysis(inner, _analysis(id="dwi/roi/inner"), _prov())
    outer = next(r for r in check(folder.parent) if r.folder == folder)
    assert any("must not nest" in e for e in outer.errors)


def test_unfinished_runs_are_flagged(folder):
    write_analysis(folder, _analysis(), provenance([{"Name": "x", "Version": "1"}], status="partial",
                                                   start="2026-10-06T00:00:00"))
    (rep,) = check(folder)
    assert rep.ok and any("partial" in w for w in rep.warnings)


def test_cli(folder, capsys):
    write_analysis(folder, _analysis(), _prov())
    assert cli(["check", str(folder.parent)]) == 0
    assert "OK   dwi/roi/demo" in capsys.readouterr().out
    (folder / "provenance.json").unlink()
    assert cli(["check", str(folder), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)[0]["ok"] is False
    assert cli(["check", str(folder / "tests.tsv")]) == 1        # nothing to check is not a pass


# ------------------------------------------------- randomise read-out adapter ---
def test_one_sample_readout_is_fully_described(tmp_path):
    """Every column read_randomise writes for a one-sample design has a dictionary entry."""
    from neurofaune.analysis.stats import readout as ro
    from neurofaune.analysis.stats.readout_results import write_readout_results

    rng = np.random.default_rng(1)
    shape, aff = (6, 5, 3), np.eye(4)
    mask = np.zeros(shape, bool)
    mask[1:5, 1:4, 1] = True
    data = rng.normal(0.5, 1, shape + (10,)) * mask[..., None]
    rd = tmp_path / "an" / "randomise_FA"
    rd.mkdir(parents=True)
    for name, img in (("data", data), ("mask", mask.astype(float)),
                      ("randomise_tstat1", 3 * mask), ("randomise_tfce_corrp_tstat1", 0.99 * mask)):
        nib.save(nib.Nifti1Image(np.asarray(img, np.float32), aff), str(rd / f"{name}.nii.gz"))
    (rd / "design.mat").write_text("/NumWaves 1\n/NumPoints 10\n/Matrix\n" + "1\n" * 10)
    (rd / "design.con").write_text("/ContrastName1 mean>0\n/NumWaves 1\n/NumContrasts 1\n/Matrix\n1\n")
    atlas = ro.Atlas(mask.astype(np.int32), {1: "Region"}, {1: "L"})
    tests, clusters = ro.read_randomise(rd, rd / "data.nii.gz", rd / "design.mat", rd / "mask.nii.gz",
                                        atlas=atlas, labels={"metric": "FA"})
    assert "whole_mean" in tests and "mean" in clusters                  # one-sample columns
    rep = write_readout_results(
        tmp_path / "an", tests, clusters, analysis_id="dwi/voxelwise/one", title="one", description="one sample",
        analysis_type="voxelwise", modality="dwi", measure_column="metric", measures=["FA"], run_dirs={"FA": rd},
        n_permutations=10, alpha=0.05, mask_name="brain mask", space="test", inference="3-D TFCE",
        strict=True)
    assert rep.ok, rep.errors


def test_an_effect_needs_its_interval_values_not_just_the_columns(folder):
    df = pd.read_csv(folder / "tests.tsv", sep="\t")
    df.loc[0, "lo"] = None
    df.to_csv(folder / "tests.tsv", sep="\t", index=False)
    write_analysis(folder, _analysis(), _prov(), strict=False)
    assert "without its interval" in _errors(folder)


@pytest.mark.parametrize("code, ok", [("LIA", True), ("ras", True), ("RRA", False), ("LI", False),
                                      ("XYZ", False), ("SIA", False)])
def test_axes_codes(code, ok):
    assert spec.valid_axes(code) is ok


def test_a_bad_axes_code_fails(folder):
    maps = [{"path": "tests.tsv", "kind": "other", "description": "x", "axes": "LIX"}]
    write_analysis(folder, _analysis(maps=maps, display={"plane": "coronal"}), _prov(), strict=False)
    assert "three letters" in _errors(folder)


def test_study_space_axes_and_plane_come_from_the_config():
    from neurofaune.atlas.study_space import display_plane, study_space_axes

    cfg = {"atlas": {"study_space": {"axes": "lia", "display_plane": "coronal"}}}
    assert study_space_axes(cfg) == "LIA" and display_plane(cfg) == "coronal"
    assert study_space_axes({}) is None and display_plane(None) is None
    with pytest.raises(ValueError):
        study_space_axes({"atlas": {"study_space": {"axes": "LLA"}}})


def test_subgroup_terms_repeat_once_per_subgroup(folder):
    df = pd.read_csv(folder / "tests.tsv", sep="\t")
    df["d_X"], df["d_Y"], df["n_X"] = [0.6, -0.2], [0.9, 0.1], [8, 8]
    df.to_csv(folder / "tests.tsv", sep="\t", index=False)
    extra = {"d_X": {"Description": "d in X", "Standard": "subgroup_effect", "EffectMeasure": "d",
                     "Subgroup": "X"},
             "d_Y": {"Description": "d in Y", "Standard": "subgroup_effect", "EffectMeasure": "d",
                     "Subgroup": "Y"},
             "n_X": {"Description": "n in X", "Standard": "subgroup_n", "Subgroup": "X"}}
    write_columns(folder / "tests.tsv", COLUMNS, extra)
    assert write_analysis(folder, _analysis(), _prov()).ok
    extra["d_Y"]["Subgroup"] = "X"                                 # the same subgroup twice
    write_columns(folder / "tests.tsv", COLUMNS, extra)
    write_analysis(folder, _analysis(), _prov(), strict=False)
    assert "for subgroup 'X' claimed by both" in _errors(folder)
    del extra["d_Y"]["Subgroup"]                                   # a subgroup effect must name it
    write_columns(folder / "tests.tsv", COLUMNS, extra)
    write_analysis(folder, _analysis(), _prov(), strict=False)
    assert "must state Subgroup" in _errors(folder)


# ------------------------------------------------------------------- 0.2 ---
def test_a_0_1_folder_is_still_read_by_0_1_rules(folder):
    old = {k: v for k, v in _analysis(id="roi/demo").items() if k != "modality"}
    write_analysis(folder, {**old, "spec_version": "0.1.0"}, _prov(), strict=False)
    (rep,) = check(folder)
    assert rep.ok, rep.errors


@pytest.mark.parametrize("over, says", [
    ({"modality": None}, "modality is required"),
    ({"modality": "DWI"}, "is not one of"),
    ({"id": "roi/demo"}, "is not <modality>/<analysis_type>/<name>"),
    ({"id": "func/roi/demo"}, "not its modality"),
    ({"id": "dwi/tbss/demo"}, "not its analysis_type"),
    ({"id": "dwi/roi/Demo Run"}, "is not lowercase"),
    ({"modality": "multimodal", "id": "multimodal/roi/demo"}, "lists its modalities"),
    ({"modalities": ["dwi", "func"]}, "for multimodal analyses only"),
    ({"measures": ["FA", "NDI"]}, "is written 'FICVF'"),
    ({"measures": ["fa", "MD"]}, "is written 'FA'"),
])
def test_0_2_identity_and_vocabulary(folder, over, says):
    a = _analysis(**over)
    if a.get("modality") is None:
        a.pop("modality")
    write_analysis(folder, a, _prov(), strict=False)
    assert says in _errors(folder)


def test_0_2_multimodal_and_unknown_measures_warn_not_fail(folder):
    write_analysis(folder, _analysis(id="multimodal/roi/demo", modality="multimodal",
                                     modalities=["dwi", "func"], measures=["FA", "MD", "Zeta"]),
                   _prov(), strict=False)
    (rep,) = check(folder)
    assert rep.ok, rep.errors
    assert any("'Zeta' is not in the measure vocabulary" in w for w in rep.warnings)


def test_0_2_a_measure_of_another_modality_warns(folder):
    write_analysis(folder, _analysis(id="func/roi/demo", modality="func"), _prov(), strict=False)
    (rep,) = check(folder)
    assert rep.ok and any("is a dwi measure, in a func analysis" in w for w in rep.warnings)


def test_0_2_a_table_names_only_listed_measures(folder):
    write_analysis(folder, _analysis(measures=["FA"]), _prov(), strict=False)
    assert "measures ['MD'] are not in analysis.json's measures" in _errors(folder)


def test_analysis_id_helper():
    from neurofaune.results.spec import analysis_id, id_problem

    got = analysis_id("func", "voxelwise", "H1j ReHo", "p60 to p90")
    assert got == "func/voxelwise/h1j_reho/p60_to_p90"
    assert id_problem(got, "func", "voxelwise") is None


def test_the_measure_vocabulary_is_packaged_and_consistent():
    from neurofaune.results.spec import MODALITIES, canonical_measure, measure_vocabulary

    vocab = measure_vocabulary()
    assert {"FA", "MK", "FICVF", "T2", "MWF", "ReHo", "fALFF", "logJ_local", "Cr+PCr"} <= set(vocab)
    assert all(v["modality"] in MODALITIES and v["modality"] != "multimodal" for v in vocab.values())
    names = {k.lower() for k in vocab}
    aliases = [a.lower() for v in vocab.values() for a in v.get("aliases", [])]
    assert len(aliases) == len(set(aliases)) and not names & set(aliases)
    assert canonical_measure("CSWF") == ("CSFF", False) and canonical_measure("Glx") == ("Glu+Gln", False)
