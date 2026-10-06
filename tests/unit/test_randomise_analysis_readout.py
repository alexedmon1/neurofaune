"""RandomiseAnalysis (VBM, voxelwise fMRI) reads every test out, nulls included.

randomise is stubbed: the maps it would write are planted, so the test checks
what the class does with them -- tests.csv / clusters.csv through
stats.readout.read_randomise, named regions from the parcellation's label table,
and the older outputs (cluster_reports_<metric>/, the per-contrast significance
summary) still written for the registry and the run scripts.
"""
from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofaune.analysis import randomise_analysis as ra
from neurofaune.reporting.discover import _discover_tbss
from neurofaune.reporting.section_renderers import render_tbss

SHAPE = (8, 7, 4)
PARC = "SIGMA_InVivo_Anatomical_Brain_Atlas"


def _save(path, data):
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), np.eye(4)), str(path))


def _vest(path, M, header):
    M = np.atleast_2d(M)
    with open(path, "w") as fh:
        fh.writelines(header)
        fh.write(f"/NumWaves {M.shape[1]}\n/{'NumContrasts' if 'con' in path.name else 'NumPoints'} "
                 f"{M.shape[0]}\n/Matrix\n")
        fh.writelines(" ".join(f"{v:g}" for v in row) + "\n" for row in M)


@pytest.fixture
def study(tmp_path):
    """An analysis_dir laid out as RandomiseAnalysis expects, with a planted effect."""
    rng = np.random.default_rng(1)
    n = 12
    subjects = [f"sub-{i:02d}" for i in range(n)]
    group = np.array([1] * 6 + [0] * 6)
    mask = np.zeros(SHAPE, bool)
    mask[1:7, 1:6, 1:3] = True
    block = np.zeros(SHAPE, bool)
    block[2:6, 1:4, 1] = True                       # spans i = 2..5 -> both "hemispheres"
    data = rng.normal(0, 1, SHAPE + (n,))
    data[block] += 2.0 * group[None, :]
    data[~mask] = 0

    ad = tmp_path / "vbm"
    (ad / "stats").mkdir(parents=True)
    (ad / "designs" / "groups").mkdir(parents=True)
    _save(ad / "stats" / "analysis_mask.nii.gz", mask)
    order = subjects[::-1]                          # master order differs from the design's
    _save(ad / "stats" / "all_GM.nii.gz", data[..., [subjects.index(s) for s in order]])
    (ad / "subject_list.txt").write_text("\n".join(order) + "\n")
    d = ad / "designs" / "groups"
    (d / "subject_order.txt").write_text("\n".join(subjects) + "\n")
    _vest(d / "design.mat", np.column_stack([group, 1 - group]).astype(float), [])
    _vest(d / "design.con", np.array([[1.0, -1.0], [-1.0, 1.0]]),
          ["/ContrastName1 A>B\n", "/ContrastName2 B>A\n"])
    (d / "design_summary.json").write_text(json.dumps({"columns": ["A", "B"], "contrasts": ["A>B", "B>A"]}))

    atlas = tmp_path / "atlas"
    atlas.mkdir()
    parc = np.zeros(SHAPE, np.int32)
    parc[:4][mask[:4]] = 1
    parc[4:][mask[4:]] = 2
    _save(atlas / f"{PARC}.nii.gz", parc)
    pd.DataFrame({"Original Atlas": ["x", "x"], "Labels": [1, 2], "Hemisphere": ["L", "R"],
                  "Matter": ["Grey Matter"] * 2, "Territories": ["Cortex"] * 2, "System": ["s"] * 2,
                  "Region of interest": ["Cortex<L>", "Cortex&R"]}).to_csv(
        atlas / f"{PARC}_Labels.csv", index=False)

    a, b = data[..., group == 1], data[..., group == 0]
    sp = np.sqrt((a.var(-1, ddof=1) + b.var(-1, ddof=1)) / 2)
    t = np.where(mask, (a.mean(-1) - b.mean(-1)) / (sp * np.sqrt(2 / 6) + 1e-12), 0)
    maps = {1: (t, np.where(block, 0.99, 0.5) * mask, np.where(block, 0.995, 0.5) * mask),
            2: (-t, 0.1 * mask, 0.2 * mask)}
    return dict(ad=ad, atlas=atlas, maps=maps, mask=mask, data=data, group=group)


def _fake_randomise(maps, calls):
    def fake(input_file, design_mat, contrast_con, output_dir, **kw):
        calls.append(kw)
        output_dir.mkdir(parents=True, exist_ok=True)
        for c, (tt, cp, pp) in maps.items():
            _save(output_dir / f"randomise_tstat{c}.nii.gz", tt)
            _save(output_dir / f"randomise_tfce_corrp_tstat{c}.nii.gz", cp)
            _save(output_dir / f"randomise_tfce_p_tstat{c}.nii.gz", pp)
        return {"success": True}
    return fake


def test_every_test_is_read_out_with_effect_and_named_clusters(study, monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "run_randomise", _fake_randomise(study["maps"], calls))
    an = ra.VBMAnalysis(analysis_dir=study["ad"])
    res = an.run("groups", metrics=["GM"], n_permutations=10,
                 parcellation_override=study["atlas"] / f"{PARC}.nii.gz")
    assert res["success"]
    assert calls[0]["tfce_2d"] is False and calls[0]["uncorrected_p"] is True   # 3-D TFCE
    out = study["ad"] / "randomise" / "groups"

    tests = pd.read_csv(out / "tests.csv")
    assert tests.contrast_name.tolist() == ["A>B", "B>A"]
    assert tests.tested_direction.tolist() == ["A > B", "B > A"]
    assert tests.significant_fwe.tolist() == [True, False]          # the null is a row too
    assert (tests.whole_d.iloc[0] > 0) and (tests.whole_d.iloc[1] < 0)
    assert tests.whole_d_ci_low.iloc[0] < tests.whole_d.iloc[0] < tests.whole_d_ci_high.iloc[0]
    assert tests.n_vox_uncorr.notna().all()                           # --uncorrp maps used

    clusters = pd.read_csv(out / "clusters.csv")
    assert len(clusters) == 1 and clusters.contrast_name.iloc[0] == "A>B"
    assert set(clusters.regions.iloc[0].split("; ")) == {"Cortex<L>:6", "Cortex&R:6"}
    assert bool(clusters.crosses_midline.iloc[0])

    summary = json.loads((out / "analysis_summary.json").read_text())
    r = summary["results"]["GM"]
    assert [t["contrast_name"] for t in r["tests"]] == ["A>B", "B>A"]
    assert r["n_significant_contrasts"] == 1 and len(r["contrasts"]) == 2   # kept for consumers
    assert summary["tests_csv"] == "tests.csv"
    assert (out / "cluster_reports_GM").is_dir()                            # kept for consumers

    from neurofaune.results import check                                    # docs/RESULTS_SPEC.md
    reports = check(out)
    assert len(reports) == 1 and reports[0].ok, reports[0].errors
    assert json.loads((out / "analysis.json").read_text())["analysis_type"] == "vbm"


def test_a_parcellation_on_another_grid_gives_unnamed_clusters_not_a_crash(study, monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "run_randomise", _fake_randomise(study["maps"], []))
    other = tmp_path / "other"
    other.mkdir()
    _save(other / f"{PARC}.nii.gz", np.ones((3, 3, 3)))
    res = ra.VoxelwiseFMRIAnalysis(analysis_dir=study["ad"]).run(
        "groups", metrics=["GM"], n_permutations=10, parcellation_override=other / f"{PARC}.nii.gz")
    assert res["success"]
    clusters = pd.read_csv(study["ad"] / "randomise" / "groups" / "clusters.csv")
    assert "peak_region" not in clusters.columns or clusters.peak_region.isna().all()


# ------------------------------------------------- the TBSS registry entry ---
def _tbss_summary(root, name, payload):
    d = root / "tbss" / "randomise" / name
    d.mkdir(parents=True)
    (d / "analysis_summary.json").write_text(json.dumps(payload))
    return d


def test_registry_reads_test_rows_from_new_tbss_summaries(tmp_path):
    d = _tbss_summary(tmp_path, "new", {
        "analysis_name": "new", "n_subjects": 12, "metrics": ["FA", "MD"], "n_contrasts": 2,
        "tests_csv": "tests.csv",
        "results": {"FA": {"tests": [{"significant_fwe": True}, {"significant_fwe": False}]},
                    "MD": {"tests": [{"significant_fwe": False}, {"significant_fwe": False}]}}})
    (d / "tests.csv").write_text("x\n")
    e = _discover_tbss(tmp_path)[0]["summary_stats"]
    assert e["n_tests"] == 4 and e["n_significant_contrasts"] == 1   # not a silent 0
    assert e["tests_csv"].endswith("new/tests.csv")
    html = render_tbss({"summary_stats": e, "output_dir": "tbss/randomise/new"}, tmp_path)
    assert "1 of 4" in html and "Tests with FWE-surviving voxels" in html and "tests.csv" in html


def test_registry_still_reads_old_tbss_summaries(tmp_path):
    _tbss_summary(tmp_path, "old", {
        "analysis_name": "old", "metrics": ["FA"],
        "results": {"FA": {"n_significant_contrasts": 2, "contrasts": []}}})
    e = _discover_tbss(tmp_path)[0]["summary_stats"]
    assert e["n_significant_contrasts"] == 2 and e["n_tests"] is None
    assert "Significant Contrasts" in render_tbss({"summary_stats": e}, tmp_path)
