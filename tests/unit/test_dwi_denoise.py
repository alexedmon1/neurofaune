"""
Unit tests for pre-eddy DWI denoising + Gibbs removal (dipy wrappers).

Hermetic: small synthetic 4-D volumes through the real dipy routines (fast at
this size). We check the in/out NIfTI contract (shape/affine preserved, finite,
non-negative) and that MP-PCA actually reduces noise on a known-clean signal.
"""
import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.dwi_denoise import denoise_dwi_mppca, degibbs_dwi


def _save(tmp_path, data, name="dwi.nii.gz", affine=None):
    affine = np.eye(4) if affine is None else affine
    f = tmp_path / name
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(f))
    return f


def test_mppca_preserves_geometry_and_denoises(tmp_path):
    rng = np.random.default_rng(0)
    # smooth-ish clean signal over many volumes, plus strong gaussian noise
    base = np.zeros((20, 20, 10, 30), dtype=np.float32)
    base[5:15, 5:15, 2:8, :] = 100.0
    noisy = base + rng.normal(0, 15, base.shape).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    out = denoise_dwi_mppca(_save(tmp_path, noisy, affine=affine), tmp_path / "den.nii.gz")
    den = nib.load(str(out))
    d = den.get_fdata()
    assert d.shape == noisy.shape
    np.testing.assert_allclose(den.affine, affine)
    assert np.isfinite(d).all() and (d >= 0).all()
    # denoising should bring the data closer to the clean signal inside the block
    roi = (slice(6, 14), slice(6, 14), slice(3, 7), slice(None))
    err_noisy = np.abs(noisy[roi] - base[roi]).mean()
    err_den = np.abs(d[roi] - base[roi]).mean()
    assert err_den < err_noisy


def test_degibbs_preserves_geometry(tmp_path):
    rng = np.random.default_rng(1)
    data = np.zeros((24, 24, 6, 5), dtype=np.float32)
    data[6:18, 6:18, :, :] = 80.0  # sharp box -> Gibbs-prone edges
    data += rng.normal(0, 2, data.shape).astype(np.float32)
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    out = degibbs_dwi(_save(tmp_path, data, affine=affine), tmp_path / "dg.nii.gz")
    dg = nib.load(str(out))
    g = dg.get_fdata()
    assert g.shape == data.shape
    np.testing.assert_allclose(dg.affine, affine)
    assert np.isfinite(g).all() and (g >= 0).all()


def test_degibbs_does_not_mutate_input_array(tmp_path):
    # gibbs_removal defaults to inplace=True; our wrapper must pass inplace=False
    data = np.zeros((16, 16, 4, 3), dtype=np.float32)
    data[4:12, 4:12, :, :] = 50.0
    f = _save(tmp_path, data)
    before = nib.load(str(f)).get_fdata().copy()
    degibbs_dwi(f, tmp_path / "dg.nii.gz")
    after = nib.load(str(f)).get_fdata()
    np.testing.assert_array_equal(before, after)  # source file untouched
