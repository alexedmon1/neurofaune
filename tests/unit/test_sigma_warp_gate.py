"""SIGMA warp resolution: the required-gate and the coverage mask.

A printed "skipped" is not a safeguard in a cohort run. On the cuprizone study
it printed on all 52 sessions, every run exited 0, and the missing warps were
found later by grepping the log.
"""
import pytest

from neurofaune.templates.sigma_warp import (
    COVERAGE_MASK_NAME,
    sigma_targets_from_config,
)


def _config(tmp_path, template=True, required=False):
    study_space = {}
    if template:
        tpl = tmp_path / "SIGMA_template.nii.gz"
        tpl.write_bytes(b"")
        study_space["template"] = str(tpl)
    else:
        study_space["template"] = str(tmp_path / "does_not_exist.nii.gz")
    if required:
        study_space["required"] = True
    return {"atlas": {"study_space": study_space}}


def test_missing_template_is_not_ready_by_default(tmp_path):
    out = sigma_targets_from_config(
        _config(tmp_path, template=False), session="ses-1")
    assert out["ready"] is False
    assert "template" in out["reason"]


def test_missing_template_raises_when_required(tmp_path):
    with pytest.raises(RuntimeError, match="required"):
        sigma_targets_from_config(
            _config(tmp_path, template=False, required=True), session="ses-1")


def test_missing_transforms_is_not_ready_by_default(tmp_path):
    out = sigma_targets_from_config(_config(tmp_path), session="ses-1")
    assert out["ready"] is False
    assert "transforms not found" in out["reason"]


def test_missing_transforms_raises_when_required(tmp_path):
    with pytest.raises(RuntimeError, match="required"):
        sigma_targets_from_config(
            _config(tmp_path, required=True), session="ses-1")


def test_ready_when_transforms_present(tmp_path):
    tx = tmp_path / "transforms"
    tx.mkdir()
    (tx / "tpl-to-SIGMA_0GenericAffine.mat").write_bytes(b"")
    (tx / "tpl-to-SIGMA_1Warp.nii.gz").write_bytes(b"")
    tpl_file = tmp_path / "tpl.nii.gz"
    tpl_file.write_bytes(b"")

    out = sigma_targets_from_config(
        _config(tmp_path, required=True), session="ses-1",
        template_file=tpl_file)

    assert out["ready"] is True
    assert out["reason"] is None
    assert out["affine"].name == "tpl-to-SIGMA_0GenericAffine.mat"
    assert out["warp"].name == "tpl-to-SIGMA_1Warp.nii.gz"


def test_affine_only_registration_has_no_warp(tmp_path):
    tx = tmp_path / "transforms"
    tx.mkdir()
    (tx / "tpl-to-SIGMA_0GenericAffine.mat").write_bytes(b"")
    tpl_file = tmp_path / "tpl.nii.gz"
    tpl_file.write_bytes(b"")

    out = sigma_targets_from_config(
        _config(tmp_path), session="ses-1", template_file=tpl_file)
    assert out["ready"] is True
    assert out["warp"] is None


def test_coverage_mask_name_matches_what_roi_extraction_reads():
    """Both sides must agree or coverage silently falls back to nonzero."""
    assert COVERAGE_MASK_NAME == "desc-brain_mask"


def test_warp_coverage_mask_returns_none_when_absent(tmp_path):
    from neurofaune.templates.sigma_warp import warp_coverage_mask

    out = warp_coverage_mask(
        mask_file=tmp_path / "nope.nii.gz",
        moving_to_template=tmp_path / "x.mat",
        sigma_template=tmp_path / "t.nii.gz",
        output_dir=tmp_path,
        subject="sub-1X",
        session="ses-1",
        tpl_to_sigma_affine=tmp_path / "a.mat",
    )
    assert out is None
