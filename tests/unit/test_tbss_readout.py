"""Read-out of a randomise run: every test with its effect, clusters with named regions.

Synthetic: a one-slice "skeleton" in an 8x6x4 volume, two labelled regions (L, R),
12 subjects in two groups, a planted effect in a block spanning both hemispheres,
and hand-written randomise outputs (no FSL needed).
"""
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofaune.analysis.stats import readout as ro
from neurofaune.analysis.tbss import reporting

SHAPE = (8, 6, 4)
AFFINE = np.diag([0.5, 0.5, 0.5, 1.0])
VOX_MM3 = 0.125


def _save(path, data):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), AFFINE), str(path))
    return path


def _vest(path, M, header):
    with open(path, "w") as fh:
        fh.write("".join(header))
        fh.write(f"/NumWaves {M.shape[1]}\n/NumPoints {M.shape[0]}\n/Matrix\n" if "con" not in path.name
                 else f"/NumWaves {M.shape[1]}\n/NumContrasts {M.shape[0]}\n/Matrix\n")
        fh.writelines(" ".join(f"{v:g}" for v in row) + "\n" for row in M)
    return path


@pytest.fixture
def run(tmp_path):
    """A complete randomise run on disk; returns its paths and the planted truth."""
    rng = np.random.default_rng(0)
    mask = np.zeros(SHAPE, bool)
    mask[1:7, 1:6, 2] = True                                    # 30 skeleton voxels
    block = np.zeros(SHAPE, bool)
    block[2:6, 1:4, 2] = True                                   # 12 voxels, i=2..5 -> both hemispheres
    lone = (6, 5, 2)                                             # two rows clear of the block

    parc = np.zeros(SHAPE, np.int32)
    parc[:4][mask[:4]] = 1
    parc[4:][mask[4:]] = 2
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    _save(atlas_dir / "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz", parc)
    pd.DataFrame({"Original Atlas": ["x", "x"], "Labels": [1, 2], "Hemisphere": ["L", "R"],
                  "Matter": ["White Matter"] * 2, "Territories": ["Tract"] * 2, "System": ["s"] * 2,
                  "Region of interest": ["Tract.L", "Tract.R"]}).to_csv(
        atlas_dir / "SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv", index=False)

    n = 12
    group = np.array([1] * 6 + [0] * 6)                          # 1 = A, 0 = B
    data = rng.normal(0, 1, SHAPE + (n,))
    data[block] += 2.0 * group                                   # A higher in the block
    data[~mask] = 0
    rd = tmp_path / "rand"
    rd.mkdir()
    _save(rd / "data.nii.gz", data)
    _save(rd / "mask.nii.gz", mask)
    X = np.column_stack([group, 1 - group]).astype(float)
    C = np.array([[1.0, -1.0], [-1.0, 1.0]])
    _vest(rd / "design.mat", X, [])
    _vest(rd / "design.con", C, ["/ContrastName1 A>B\n", "/ContrastName2 B>A\n"])
    (rd / "design_summary.json").write_text(json.dumps({"columns": ["A", "B"], "contrasts": ["A>B", "B>A"]}))

    a, b = data[..., group == 1], data[..., group == 0]
    sp = np.sqrt((a.var(-1, ddof=1) + b.var(-1, ddof=1)) / 2)
    t = np.where(mask, (a.mean(-1) - b.mean(-1)) / (sp * np.sqrt(2 / 6) + 1e-12), 0)
    corrp1 = np.where(block, 0.99, 0.5) * mask
    p1 = np.where(block, 0.995, 0.5) * mask
    p1[lone] = 0.97
    for c, (tt, cp, pp) in enumerate([(t, corrp1, p1), (-t, 0.1 * mask, 0.2 * mask)], start=1):
        _save(rd / f"randomise_tstat{c}.nii.gz", tt)
        _save(rd / f"randomise_tfce_corrp_tstat{c}.nii.gz", cp)
        _save(rd / f"randomise_tfce_p_tstat{c}.nii.gz", pp)
    atlas = ro.Atlas.from_files(atlas_dir / "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz",
                                atlas_dir / "SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv")
    return dict(dir=rd, atlas=atlas, atlas_dir=atlas_dir, mask=mask, block=block, data=data,
                group=group, t=t, X=X, tmp=tmp_path)


def _read(run, **kw):
    rd = run["dir"]
    return ro.read_randomise(rd, rd / "data.nii.gz", rd / "design.mat", rd / "mask.nii.gz",
                             atlas=run["atlas"], **kw)


def _pooled_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


# ------------------------------------------------------------------ tests ---
def test_every_contrast_is_a_row_null_included(run):
    tests, _ = _read(run)
    assert list(tests.contrast_name) == ["A>B", "B>A"]
    assert list(tests.tested_direction) == ["A > B", "B > A"]
    assert tests.significant_fwe.tolist() == [True, False]
    assert tests.n_vox_fwe.tolist() == [int(run["block"].sum()), 0]
    assert (tests.n == 12).all() and (tests.df == 10).all()
    assert tests.mask_voxels.iloc[0] == int(run["mask"].sum())


def test_whole_mask_effect_is_unselected_and_signed_by_contrast(run):
    tests, _ = _read(run)
    m = run["data"][run["mask"]].mean(axis=0)
    d = _pooled_d(m[run["group"] == 1], m[run["group"] == 0])
    assert tests.whole_d.iloc[0] == pytest.approx(d)
    assert tests.whole_d_raw.iloc[0] == pytest.approx(d)       # no covariates: GLM d == raw d
    assert tests.whole_d.iloc[1] == pytest.approx(-d)
    assert tests.whole_d_ci_low.iloc[0] < d < tests.whole_d_ci_high.iloc[0]
    assert tests.whole_d_ci_low.iloc[1] == pytest.approx(-tests.whole_d_ci_high.iloc[0])
    # the null contrast still says which way the data went
    assert tests.whole_observed_direction.tolist() == ["A > B", "A > B"]
    assert tests.whole_mean_pos.iloc[0] == pytest.approx(m[run["group"] == 1].mean())


def test_cluster_extent_location_and_named_regions(run):
    _, cl = _read(run)
    assert len(cl) == 1
    c = cl.iloc[0]
    assert c.n_voxels == run["block"].sum()
    assert c.mm3 == pytest.approx(run["block"].sum() * VOX_MM3)
    assert set(c.regions.split("; ")) == {"Tract.L:6", "Tract.R:6"}
    assert c.crosses_midline and c.n_regions == 2
    assert c.peak_region in {"Tract.L", "Tract.R"}
    peak = run["t"][run["block"]].max()
    assert c.peak_t == pytest.approx(peak, rel=1e-5)
    assert c.effect_selected
    m = run["data"][run["block"]].mean(axis=0)
    assert c.d_raw == pytest.approx(_pooled_d(m[run["group"] == 1], m[run["group"] == 0]))


def test_uncorrected_cluster_definition_and_minimum_size(run):
    _, cl = _read(run, cluster_on="uncorrected")
    assert sorted(cl[cl.contrast == 1].n_voxels) == [1, 12]     # the lone voxel is its own cluster
    _, cl = _read(run, cluster_on="uncorrected", min_cluster_size=2)
    assert cl[cl.contrast == 1].n_voxels.tolist() == [12]
    tests, _ = _read(run, cluster_on="uncorrected", min_cluster_size=2)
    assert "uncorrected p < 0.05" in tests.cluster_definition.iloc[0]


def test_uncorrected_clusters_need_uncorrected_maps(run):
    for f in run["dir"].glob("randomise_tfce_p_tstat*"):
        f.unlink()
    tests, _ = _read(run)                                       # FWE read-out still works
    assert tests.n_vox_uncorr.isna().all()
    with pytest.raises(FileNotFoundError, match="--uncorrp"):
        _read(run, cluster_on="uncorrected")


def test_contrast_count_must_match_the_design(run):
    (run["dir"] / "randomise_tstat2.nii.gz").unlink()
    with pytest.raises(ValueError, match="1 t-stat maps but 2 contrasts"):
        _read(run)


def test_atlas_must_share_the_mask_space(run):
    bad = ro.Atlas(np.zeros((3, 3, 3), np.int32))
    rd = run["dir"]
    with pytest.raises(ValueError, match="atlas shape"):
        ro.read_randomise(rd, rd / "data.nii.gz", rd / "design.mat", rd / "mask.nii.gz", atlas=bad)


# ---------------------------------------------------------- design / effect ---
def test_describe_contrast_states_direction_in_design_terms():
    X = np.column_stack([np.ones(6), [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], np.linspace(-1, 1, 6)])
    cols = ["mean", "ctl", "cpz", "age"]
    assert ro.describe_contrast(np.array([0, -1, 1, 0]), X, cols)["tested"] == "cpz > ctl"
    assert ro.describe_contrast(np.array([-1, 0, 0, 0]), X, cols)["tested"] == "mean < 0"
    assert ro.describe_contrast(np.array([0, 0, 0, 1]), X, cols)["tested"] == "positive coefficient of age"
    general = ro.describe_contrast(np.array([0, 1, 1, 0]), X, cols)
    assert general["kind"] == "general" and general["tested"] == "+1*ctl +1*cpz > 0"


def test_one_sample_with_centred_covariate():
    rng = np.random.default_rng(1)
    batch = np.array([0] * 5 + [1] * 5, float)
    y = rng.normal(0.3, 1, 10) + 0.8 * batch
    X = np.column_stack([np.ones(10), batch - batch.mean()])
    c = np.array([1.0, 0.0])
    eff = ro.contrast_effect(y, X, c, ro.describe_contrast(c, X, ["mean", "batch"]))
    assert eff["estimate"] == pytest.approx(y.mean())          # centred covariate leaves the mean
    assert eff["d_raw"] == pytest.approx(y.mean() / y.std(ddof=1))
    assert abs(eff["d"]) > abs(eff["d_raw"])                    # batch variance left the error term
    assert eff["effect_df"] == 8


def test_ci_matches_large_sample_normal_approximation():
    n1 = n2 = 2000
    d, v = 0.3, 1 / n1 + 1 / n2
    lo, hi = ro._d_ci(d / np.sqrt(v), n1 + n2 - 2, v, 0.95)
    se = np.sqrt(v + d ** 2 / (2 * (n1 + n2)))
    assert lo == pytest.approx(d - 1.96 * se, abs=2e-3)
    assert hi == pytest.approx(d + 1.96 * se, abs=2e-3)


def test_ci_survives_scipy_nan_tails():
    """scipy's nct.cdf is NaN in far tails (seen at t=-3, df=32); the bound must still solve."""
    lo, hi = ro._d_ci(-3.0, 32, 0.1, 0.95)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < -3.0 * np.sqrt(0.1) < hi


def test_design_names_fall_back_to_design_con_then_defaults(tmp_path):
    _vest(tmp_path / "design.con", np.eye(2), ["/ContrastName1 up\n", "/ContrastName2 down\n"])
    cols, cons = ro.design_names(tmp_path, 2, 2)
    assert cols == ["EV1", "EV2"] and cons == ["up", "down"]


def test_design_record_names_the_columns_and_says_what_each_contrast_tests(run):
    """A design.json beside the design (neurofaune's design records) is the first
    source of names, and each test row carries the record's sentence."""
    from neurofaune.analysis.stats import design_record as dr
    rd = run["dir"]
    dr.write_design(rd, run["X"], [("drug", "1 if the animal got the drug", "group"),
                                   ("vehicle", "1 if the animal got vehicle", "group")],
                    [{"name": "drug>vehicle", "vector": [1, -1], "test_kind": "two_group",
                      "tests": "drug mean > vehicle mean", "group_a": "drug", "group_b": "vehicle"},
                     {"name": "vehicle>drug", "vector": [-1, 1], "test_kind": "two_group",
                      "tests": "vehicle mean > drug mean", "group_a": "vehicle", "group_b": "drug"}],
                    summary="Does the drug change FA?", groups={"drug": 6, "vehicle": 6},
                    rows=[f"sub-{i}" for i in range(12)])
    tests, _ = _read(run)
    assert tests.contrast_name.tolist() == ["drug>vehicle", "vehicle>drug"]
    assert tests.contrast_tests.tolist() == ["drug mean > vehicle mean", "vehicle mean > drug mean"]
    assert "drug" in tests.tested_direction.iloc[0]


def test_without_a_record_contrast_tests_is_empty(run):
    tests, _ = _read(run)
    assert tests.contrast_tests.isna().all()


# ----------------------------------------------------------------- report ---
def test_report_renders_from_readout_with_a_null_section(run, tmp_path):
    tests, clusters = _read(run, cluster_on="uncorrected")
    rd = tmp_path / "analysis"
    rd.mkdir()
    tests.assign(metric="FA").to_csv(rd / "tests.csv", index=False)
    clusters.assign(metric="FA").to_csv(rd / "clusters.csv", index=False)
    out = reporting.generate_tbss_report("demo", tmp_path / "no_tbss", rd, rd / "report.html", metrics=["FA"])
    html = out.read_text()
    assert "Null results" in html and "<td>B&gt;A</td>" in html
    assert "SIGNIFICANT" not in html and "n.s." not in html
    assert 'href="clusters.csv"' in html
    assert "1 of 2 tests had no voxel surviving FWE correction" in html
    assert "Tract.L" in html


def test_report_escapes_every_name(run, tmp_path):
    """Names reach the page as text, never as markup: <, > and & in an analysis,
    metric, contrast or region name are escaped wherever they appear."""
    tests, clusters = _read(run, cluster_on="uncorrected")
    bad_metric, bad_con, bad_region = "FA<i>&</i>", "A<b>&</b>B", "Tract<script>&x"
    tests = tests.assign(metric=bad_metric, contrast_name=bad_con)
    clusters = clusters.assign(metric=bad_metric, contrast_name=bad_con, peak_region=bad_region,
                               regions=f"{bad_region}:3; Other&<:1")
    rd = tmp_path / "analysis"
    rd.mkdir()
    tests.to_csv(rd / "tests.csv", index=False)
    clusters.to_csv(rd / "clusters.csv", index=False)
    (rd / "analysis_summary.json").write_text(json.dumps(
        {"metrics": [bad_metric], "cluster_definition": "FWE p < 0.05, 26-connected, >= 1 voxels"}))
    out = reporting.generate_tbss_report("dose<x>&y", tmp_path / "no_tbss", rd, rd / "report.html",
                                         metrics=[bad_metric])
    html = out.read_text()
    for raw in ("<i>", "</i>", "<b>", "</b>", "<script>", "dose<x>", "Other&<", "p < 0.05"):
        assert raw not in html, raw
    for escaped in ("FA&lt;i&gt;&amp;&lt;/i&gt;", "A&lt;b&gt;&amp;&lt;/b&gt;B", "Tract&lt;script&gt;&amp;x",
                    "dose&lt;x&gt;&amp;y", "p &lt; 0.05"):
        assert escaped in html, escaped


def test_cluster_report_escapes_the_contrast_name(tmp_path):
    from neurofaune.analysis.stats import cluster_report as cr
    df = pd.DataFrame({"cluster_id": [1], "size_voxels": [3], "peak_stat": [2.5], "peak_corrp": [0.99],
                       "peak_x_mm": [0.0], "peak_y_mm": [0.0], "peak_z_mm": [0.0], "region": ["R<&>"]})
    out = tmp_path / "c.html"
    cr._generate_html_report(df, "A<b>&B", out, 0.95)
    html = out.read_text()
    assert "A<b>" not in html and "R<&>" not in html
    assert "A&lt;b&gt;&amp;B" in html and "R&lt;&amp;&gt;" in html


def test_report_says_when_clusters_are_truncated(run):
    tests, clusters = _read(run, cluster_on="uncorrected")
    html = reporting._build_metric_results("FA", tests.assign(metric="FA"),
                                           clusters.assign(metric="FA"), max_rows=1)
    assert "Showing the 1 largest of 2 clusters" in html


# ------------------------------------------------------------ run_tbss_stats ---
def test_run_tbss_stats_writes_tests_and_named_clusters(run, tmp_path, monkeypatch):
    from neurofaune.analysis.tbss import run_tbss_stats as rts

    tbss = tmp_path / "tbss"
    (tbss / "stats").mkdir(parents=True)
    (tbss / "subject_manifest.json").write_text(json.dumps({"subjects_included": 12}))
    _save(tbss / "stats" / "analysis_mask.nii.gz", run["mask"])
    _save(tbss / "stats" / "all_FA.nii.gz", run["data"])
    design = tmp_path / "design"
    design.mkdir()
    for f in ("design.mat", "design.con", "design_summary.json"):
        (design / f).write_bytes((run["dir"] / f).read_bytes())

    def fake_randomise(input_file, design_mat, contrast_con, output_dir, **kw):
        assert kw.get("uncorrected_p") is True
        output_dir.mkdir(parents=True, exist_ok=True)
        for f in run["dir"].glob("randomise_*"):
            (output_dir / f.name).write_bytes(f.read_bytes())
        return {"success": True}

    monkeypatch.setattr(rts, "run_randomise", fake_randomise)
    out = tmp_path / "out"
    res = rts.run_tbss_statistical_analysis(
        tbss, design, out, "demo", metrics=["FA"], n_permutations=10,
        config={"atlas": {"study_space": {"base_path": str(run["atlas_dir"])}}})
    assert res["success"]
    tests = pd.read_csv(out / "tests.csv")
    clusters = pd.read_csv(out / "clusters.csv")
    assert tests.tested_direction.tolist() == ["A > B", "B > A"]
    assert clusters.peak_region.iloc[0] in {"Tract.L", "Tract.R"}   # names, not "SIGMA region 1"
    assert (out / "randomise_FA" / "randomise_cohend1.nii.gz").exists()
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["results"]["FA"]["tests"][1]["significant_fwe"] is False

    # ... and the folder is a results-specification analysis (docs/RESULTS_SPEC.md)
    from neurofaune.results import check
    reports = check(out)
    assert len(reports) == 1 and reports[0].ok, reports[0].errors
    analysis = json.loads((out / "analysis.json").read_text())
    assert analysis["analysis_type"] == "tbss" and analysis["inference"]["correction"]["p_kind"] == "fwe"
    assert {m["kind"] for m in analysis["maps"]} == {"stat", "p_corrected", "p_uncorrected", "mask"}
    assert (out / "mask.nii.gz").exists()          # the folder is complete on its own
