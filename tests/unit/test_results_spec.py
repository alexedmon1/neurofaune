"""What of the results specification is neurofaune's own: its read-out writer and its study space.

The specification's own tests (schemas, checker, writer) live in Lumière
(tests/test_spec.py), with the package, since it moved there on 2026-11-03.
"""
import nibabel as nib
import numpy as np
import pytest


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


def test_study_space_axes_and_plane_come_from_the_config():
    from neurofaune.atlas.study_space import display_plane, study_space_axes

    cfg = {"atlas": {"study_space": {"axes": "lia", "display_plane": "coronal"}}}
    assert study_space_axes(cfg) == "LIA" and display_plane(cfg) == "coronal"
    assert study_space_axes({}) is None and display_plane(None) is None
    with pytest.raises(ValueError):
        study_space_axes({"atlas": {"study_space": {"axes": "LLA"}}})
