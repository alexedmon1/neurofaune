"""Unit tests for ``refine_func_mask_with_anat`` — opt-in func mask refinement.

Guards the gating and the failure mode. The refinement itself is
``restrict_mask_to`` (see test_restrict_mask.py); what matters here is that it
stays OFF unless asked for, and degrades to a no-op rather than an exception
when the anat inputs are missing — a session without a preprocessed T2w must
still complete with its adaptive-BET mask intact.
"""
import nibabel as nib
import numpy as np

from neurofaune.preprocess.workflows.func_preprocess import refine_func_mask_with_anat

SHAPE = (8, 8, 3)


def _mask(tmp_path, name):
    m = np.zeros(SHAPE, np.uint8)
    m[2:6, 2:6, 1:2] = 1
    p = tmp_path / name
    nib.save(nib.Nifti1Image(m, np.eye(4)), p)
    return p


def test_disabled_by_default(tmp_path):
    """No config key -> untouched. The feature must never turn itself on."""
    mask = _mask(tmp_path, "brain_mask.nii.gz")
    before = nib.load(mask).get_fdata()
    out = refine_func_mask_with_anat(
        {}, "sub-1X", "ses-1", tmp_path / "derivatives" / "sub-1X" / "ses-1" / "func",
        tmp_path / "ref.nii.gz", mask, tmp_path / "work")
    assert out is None
    assert np.array_equal(nib.load(mask).get_fdata(), before)


def test_missing_anat_is_a_no_op_not_an_error(tmp_path):
    """Enabled but no preproc T2w: warn and keep the existing mask."""
    mask = _mask(tmp_path, "brain_mask.nii.gz")
    before = nib.load(mask).get_fdata()
    cfg = {"functional": {"second_mask": {"method": "anat_mask"}}}
    deriv = tmp_path / "derivatives" / "sub-1X" / "ses-1" / "func"
    deriv.mkdir(parents=True)
    out = refine_func_mask_with_anat(
        cfg, "sub-1X", "ses-1", deriv, tmp_path / "ref.nii.gz", mask,
        tmp_path / "work")
    assert out is None
    assert np.array_equal(nib.load(mask).get_fdata(), before)


def test_unknown_method_does_nothing(tmp_path):
    mask = _mask(tmp_path, "brain_mask.nii.gz")
    cfg = {"functional": {"second_mask": {"method": "something_else"}}}
    assert refine_func_mask_with_anat(
        cfg, "sub-1X", "ses-1", tmp_path, tmp_path / "ref.nii.gz", mask,
        tmp_path / "work") is None
