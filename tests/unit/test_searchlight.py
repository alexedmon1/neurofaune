"""Searchlight and whole-brain decoding: geometry, coverage, the harness's rules, FWE, outputs.

The geometry, 4D, FWE and engine tests come from murinet (tests/test_searchlight.py,
origin/main ade3d38); the rest cover what changed on the way in.
"""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurofaune.analysis.decoding.searchlight import (
    SearchlightSettings,
    _as_3d,
    _estimator,
    _parse,
    _scale_in_fold,
    atlas_space_maps,
    coverage_report,
    covered,
    design_scans,
    fwe_pvalues,
    run_searchlight,
    run_whole_brain,
    score_sphere,
    sphere_neighbourhoods,
)
from neurofaune.study_design import StudyDesign

# --- sphere geometry ---------------------------------------------------------


def test_radius_is_true_mm_not_stored_mm():
    """Rodent voxels are stored x10; a radius given in true mm must survive that."""
    mask = np.ones((9, 9, 9), bool)
    zooms = (1.5, 1.5, 1.5)          # stored; true spacing is 0.15 mm
    _, scaled = sphere_neighbourhoods(mask, zooms, radius_mm=0.45, voxel_scale=10.0)
    _, unscaled = sphere_neighbourhoods(mask, zooms, radius_mm=0.45, voxel_scale=1.0)
    centre = 9 * 9 * 4 + 9 * 4 + 4
    assert len(scaled[centre]) > 1, "voxel_scale ignored: 0.45 mm should span 3 voxels"
    assert len(unscaled[centre]) == 1, "0.45 mm cannot reach past a 1.5 mm voxel"


def test_sphere_radius_grows_the_neighbourhood():
    mask = np.ones((11, 11, 11), bool)
    centre = 11 * 11 * 5 + 11 * 5 + 5
    sizes = [len(sphere_neighbourhoods(mask, (1.0, 1.0, 1.0), r, 1.0)[1][centre])
             for r in (1.0, 2.0, 3.0)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_spheres_never_reach_outside_the_mask():
    mask = np.zeros((9, 9, 9), bool)
    mask[4, 4, :] = True
    coords, neighbours = sphere_neighbourhoods(mask, (1.0, 1.0, 1.0), 3.0, 1.0)
    assert len(coords) == 9
    for idx in neighbours:
        assert set(coords[idx][:, 0]) <= {4} and set(coords[idx][:, 1]) <= {4}


def test_one_neighbourhood_per_mask_voxel():
    mask = np.zeros((6, 6, 6), bool)
    mask[1:4, 1:4, 1:4] = True
    coords, neighbours = sphere_neighbourhoods(mask, (1.0, 1.0, 1.0), 1.0, 1.0)
    assert len(coords) == len(neighbours) == 27


# --- 4D handling -------------------------------------------------------------


def _img(shape):
    return nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4))


def test_3d_map_passes_through():
    assert _as_3d(_img((4, 4, 4)), "x.nii.gz", None).shape == (4, 4, 4)


def test_4d_is_refused_by_default():
    """A 360-volume BOLD run is not a feature map; the 3D metrics are."""
    with pytest.raises(ValueError, match="fALFF"):
        _as_3d(_img((4, 4, 4, 10)), "bold.nii.gz", None)


@pytest.mark.parametrize("mode", ["mean", "first"])
def test_4d_reduction_is_opt_in(mode):
    assert _as_3d(_img((4, 4, 4, 10)), "bold.nii.gz", mode).shape == (4, 4, 4)


# --- family-wise error -------------------------------------------------------


def test_fwe_uses_the_max_statistic_across_voxels():
    observed = np.array([0.9, 0.5])
    null = np.array([[0.4, 0.8], [0.3, 0.85], [0.45, 0.6]])
    p, thresh = fwe_pvalues(observed, null)
    assert p[0] < p[1]
    assert thresh == pytest.approx(np.quantile(null.max(axis=1), 0.95))


def test_fwe_pvalue_is_never_zero():
    p, _ = fwe_pvalues(np.array([1.0]), np.zeros((99, 1)))
    assert p[0] == pytest.approx(1 / 100)


def test_fwe_pvalue_is_bounded_by_one():
    p, _ = fwe_pvalues(np.array([0.0]), np.ones((9, 1)))
    assert p[0] == pytest.approx(1.0)


def test_no_permutations_gives_nan_not_a_false_result():
    p, thresh = fwe_pvalues(np.array([0.9]), np.empty((0, 1)))
    assert np.isnan(p[0]) and np.isnan(thresh)


# --- estimator and scoring engine -------------------------------------------


def test_default_estimator_is_the_precomputed_kernel():
    mode, model = _estimator(SearchlightSettings())
    assert mode == "kernel" and model.kernel == "precomputed"


def test_lda_is_still_selectable():
    assert _estimator(SearchlightSettings(estimator="lda"))[0] == "dense"


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="svm"):
        _estimator(SearchlightSettings(estimator="forest"))


def test_scaling_equalises_feature_scales():
    """A sphere of several maps mixes units: unscaled, FA (0-1) would swamp MWF (0-0.3)."""
    x = np.array([[0.10, 100.0], [0.20, 200.0], [0.30, 300.0]])
    z = _scale_in_fold(x, np.arange(3))
    assert np.allclose(z[:, 0], z[:, 1])
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)


def test_scaling_is_fitted_on_the_training_rows_only():
    """The held-out animals must not move the scaling: harness rule 3."""
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    train = np.array([0, 1, 2])
    moved = x.copy()
    moved[3] = 1000.0
    assert np.allclose(_scale_in_fold(x, train)[train], _scale_in_fold(moved, train)[train])


def test_scaling_survives_a_constant_feature():
    z = _scale_in_fold(np.array([[1.0, 5.0], [2.0, 5.0]]), np.arange(2))
    assert np.isfinite(z).all()


_FOLDS = [(np.array([0, 1, 4, 5]), np.array([2, 3, 6, 7])),
          (np.array([2, 3, 6, 7]), np.array([0, 1, 4, 5]))]


def test_kernel_and_dense_paths_agree_on_a_separable_problem():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    features = rng.standard_normal((8, 5)) + y[:, None] * 6.0
    kernel, _ = score_sphere(features, np.vstack([y]), _FOLDS, _estimator(SearchlightSettings()))
    dense, _ = score_sphere(features, np.vstack([y]), _FOLDS,
                            _estimator(SearchlightSettings(estimator="lda")))
    assert kernel[0] == pytest.approx(1.0) and dense[0] == pytest.approx(1.0)


def test_permutation_row_zero_is_the_truth():
    rng = np.random.default_rng(1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    features = rng.standard_normal((8, 4)) + y[:, None] * 8.0
    labellings = np.vstack([y, np.array([1, 0, 1, 0, 1, 0, 1, 0])])
    scores, _ = score_sphere(features, labellings, _FOLDS, _estimator(SearchlightSettings()))
    assert scores[0] > scores[1]


def test_a_fold_with_one_class_is_counted_not_hidden():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    folds = [(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
             (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3]))]
    _, degenerate = score_sphere(np.ones((8, 2)), np.vstack([y]), folds,
                                 _estimator(SearchlightSettings()))
    assert degenerate == 2


# --- inputs --------------------------------------------------------------------


@pytest.mark.parametrize("folder, name, expected", [
    ("dwi", "sub-01_ses-1_space-SIGMA_FA.nii.gz", "FA"),
    ("dwi", "sub-01_ses-1_space-SIGMA_desc-syn_FA.nii.gz", "FA-syn"),
    ("func", "sub-01_ses-1_space-SIGMA_desc-fALFF_bold.nii.gz", "fALFF"),
    ("msme", "sub-01_ses-1_space-SIGMA_MWF.nii.gz", "MWF"),
])
def test_map_names(tmp_path, folder, name, expected):
    entry = _parse(tmp_path / folder / name)
    assert entry.name == expected and entry.subject == "sub-01_ses-1" and entry.modality == folder


@pytest.mark.parametrize("name", ["sub-01_ses-1_space-SIGMA_desc-brain_mask.nii.gz",
                                  "sub-01_ses-1_space-SIGMA_mask.nii.gz",
                                  "sub-01_ses-1_space-SIGMA_bold.nii.gz"])
def test_masks_and_timeseries_are_not_maps(tmp_path, name):
    assert _parse(tmp_path / "func" / name) is None


def test_two_files_of_one_map_are_refused_when_asked_for(tmp_path):
    from neurofaune.analysis.decoding.searchlight import load_maps

    for folder in ("dwi", "dwi2"):
        d = tmp_path / "sub-01" / "ses-1" / folder
        d.mkdir(parents=True)
        nib.save(_img((2, 2, 2)), str(d / "sub-01_ses-1_space-SIGMA_FA.nii.gz"))
    files = atlas_space_maps(tmp_path)
    with pytest.raises(ValueError, match="more than one file"):
        load_maps(files, ["sub-01_ses-1"], ["FA"], np.ones((2, 2, 2), bool), _img((2, 2, 2)))


def test_coverage_needs_data_in_every_scan_by_default():
    X = np.ones((4, 3, 1))
    X[0, 1, 0] = 0.0
    X[1, 2, 0] = np.nan
    assert covered(X, 1.0)[:, 0].tolist() == [True, False, False]
    assert covered(X, 0.75)[:, 0].tolist() == [True, True, True]


def _sessions(n=6):
    rows = []
    for i in range(n):
        for s in ("ses-1", "ses-2"):
            rows.append({"subject": f"sub-{i:02d}_{s}", "animal": f"sub-{i:02d}", "session": s,
                         "group": "treated" if i % 2 else "control"})
    return pd.DataFrame(rows)


def test_delta_pairs_each_animals_sessions():
    b, a, y = design_scans(_sessions(), "delta", "1", "2")
    assert [s.split("_")[0] for s in a] == [s.split("_")[0] for s in b] == list(y.index)
    assert all(s.endswith("ses-2") for s in b) and all(s.endswith("ses-1") for s in a)


def test_cross_sectional_takes_one_session():
    b, a, y = design_scans(_sessions(), "cross_sectional", "1", "2")
    assert a is None and len(b) == len(y) == 6


def test_no_shared_session_is_an_error():
    with pytest.raises(ValueError, match="no animal has both"):
        design_scans(_sessions(), "delta", "1", "9")


# --- a synthetic study ---------------------------------------------------------

N_ANIMALS = 20
SHAPE = (6, 6, 6)


def _study(tmp_path):
    """20 animals x 2 sessions, groups in blocks so the folds mix. FA changes in the left region of treated animals only; MWF
    is a slab covering the lower half of the brain."""
    rng = np.random.default_rng(0)
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    labels = np.zeros(SHAPE, np.int16)
    labels[1:5, 1:5, 1:5] = 2
    labels[1:3, 1:5, 1:5] = 1
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    parcellation = atlas_dir / "toy_atlas.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), str(parcellation))
    pd.DataFrame({"Labels": [1, 2], "Hemisphere": ["L", "R"], "Matter": ["WM", "WM"],
                  "Territories": ["Tract", "Tract"], "System": ["Fibre", "Fibre"],
                  "Region of interest": ["Left tract", "Right tract"]}
                 ).to_csv(atlas_dir / "toy_atlas_Labels.csv", index=False)
    groups = tmp_path / "groups.yaml"
    groups.write_text("structures:\n  left:\n    include_ids: [1]\n    tissue: any\n")

    derivatives = tmp_path / "derivatives"
    animals = [f"sub-{i:02d}" for i in range(N_ANIMALS)]
    for i, animal in enumerate(animals):
        treated = i >= N_ANIMALS // 2
        for session in ("ses-1", "ses-2"):
            fa = 0.5 + 0.1 * rng.standard_normal(SHAPE)
            if treated and session == "ses-2":
                fa[labels == 1] += 1.0
            mwf = 0.2 + 0.02 * rng.standard_normal(SHAPE)
            mwf[:, :, 3:] = 0.0
            for folder, name, data in (("dwi", "FA", fa), ("msme", "MWF", mwf)):
                d = derivatives / animal / session / folder
                d.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image((data * (labels > 0)).astype(np.float32), affine),
                         str(d / f"{animal}_{session}_space-SIGMA_{name}.nii.gz"))
    design = StudyDesign(
        pd.DataFrame({"animal": animals,
                      "group": ["treated" if i >= N_ANIMALS // 2 else "control" for i in range(N_ANIMALS)]}),
        pd.DataFrame([{"animal": a, "session_id": s} for a in animals for s in ("ses-1", "ses-2")]),
        control="control",
    )
    config = {
        "paths": {"derivatives": str(derivatives)},
        "bids": {"voxel_scale": 10.0},
        "searchlight": {
            "name": "toy", "maps": ["FA"], "parcellation": str(parcellation),
            "structure_groups": str(groups), "radius_mm": 0.2, "cv_folds": 4,
            "permutations": 39, "workers": 1,
            "contrasts": [{"name": "change", "session_a": "1", "session_b": "2"}],
        },
    }
    return config, design


def _assert_conforms(folder):
    from neurofaune.results import check

    reports = check(folder)
    assert not [e for r in reports for e in r.errors]
    return [w for r in reports for w in r.warnings]


def test_searchlight_finds_the_changed_region_and_writes_a_conforming_folder(tmp_path):
    config, design = _study(tmp_path)
    result = run_searchlight(config, tmp_path / "out", design=design)

    folder = result["folder"]
    assert folder.parts[-5:-1] == ("dwi", "decoding", "searchlight", "toy")
    _assert_conforms(folder)
    test = result["summary"].iloc[0]
    assert test["n"] == N_ANIMALS and test["usable_voxels"] == test["search_voxels"] == 64
    assert test["significant"] and test["positive_class"] == "treated"
    clusters = pd.read_csv(folder / "clusters.csv")
    assert clusters.iloc[0]["peak_region"] == "Left tract"
    regions = pd.read_csv(folder / "regions.csv").set_index("region")
    assert regions.loc["Left tract", "n_significant"] > regions.loc["Right tract", "n_significant"]
    assert (folder / "change_FA_score.nii.gz").exists() and (folder / "change_FA_pfwe.nii.gz").exists()


def test_a_slab_map_shrinks_the_search_volume(tmp_path):
    config, design = _study(tmp_path)
    config["searchlight"]["maps"] = ["FA", "MWF"]
    config["searchlight"]["permutations"] = 0
    result = run_searchlight(config, tmp_path / "out", design=design)

    test = result["summary"].iloc[0]
    assert test["search_voxels"] == 64 and test["usable_voxels"] == 32
    assert test["measure"] == "FA+MWF"
    assert result["folder"].parts[-5] == "multimodal"


def test_a_structure_scopes_the_search(tmp_path):
    config, design = _study(tmp_path)
    config["searchlight"]["mask"] = "left"
    config["searchlight"]["permutations"] = 0
    test = run_searchlight(config, tmp_path / "out", design=design)["summary"].iloc[0]
    assert test["search_voxels"] == 32


def test_an_unknown_structure_names_the_ones_there_are(tmp_path):
    config, design = _study(tmp_path)
    config["searchlight"]["mask"] = "hippocampus"
    with pytest.raises(KeyError, match="left"):
        run_searchlight(config, tmp_path / "out", design=design)


def test_whole_brain_decodes_and_writes_weight_maps(tmp_path):
    config, design = _study(tmp_path)
    config["searchlight"]["pca_components"] = 5
    result = run_whole_brain(config, tmp_path / "out", design=design)

    folder = result["folder"]
    assert folder.parts[-5:-1] == ("dwi", "decoding", "wholebrain", "toy")
    _assert_conforms(folder)
    test = result["summary"].iloc[0]
    assert test["stat"] > 0.8 and test["p_value"] < 0.05
    weights = nib.load(str(folder / "change_FA_weights-FA.nii.gz")).get_fdata()
    assert np.abs(weights[1:3]).mean() > np.abs(weights[3:5]).mean()


def test_coverage_report_names_the_limiting_map(tmp_path):
    config, design = _study(tmp_path)
    config["searchlight"]["maps"] = ["FA", "MWF"]
    report = coverage_report(config, design=design).set_index("region")
    assert report.loc["whole_brain", "pct_FA"] == 100.0
    assert report.loc["whole_brain", "pct_joint"] == 50.0
    assert report.loc["whole_brain", "limiting_map"] == "MWF"


def test_settings_are_required(tmp_path):
    config, design = _study(tmp_path)
    del config["searchlight"]["contrasts"]
    with pytest.raises(KeyError, match="searchlight.contrasts"):
        run_searchlight(config, tmp_path / "out", design=design)


def test_without_the_results_package_tables_and_maps_are_written(tmp_path, monkeypatch, caplog):
    import neurofaune.results as nr

    monkeypatch.setattr(nr, "AVAILABLE", False)
    config, design = _study(tmp_path)
    config["searchlight"]["permutations"] = 0
    folder = run_searchlight(config, tmp_path / "out", design=design)["folder"]
    names = sorted(p.name for p in folder.iterdir())
    assert "tests.csv" in names and "change_FA_score.nii.gz" in names
    assert not any(n.endswith(".json") for n in names)
    assert "neurofaune[results]" in caplog.text
