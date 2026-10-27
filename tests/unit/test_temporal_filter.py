"""Unit tests for func_preprocess.temporal_filter: the configured cutoffs must be the real ones.

Before 2026-10-27 the filter was ``fslmaths -bptf`` with the low-pass sigma from FSL's high-pass formula,
which put a configured 0.1 Hz low-pass at about 0.027 Hz (TR 2 s). These tests measure the response.
"""

import shutil

import nibabel as nib
import numpy as np
import pytest
from scipy import signal

from neurofaune.preprocess.workflows.func_preprocess import temporal_filter

TR = 2.0
T = 600


def _noise_image(tmp_path, offset=100.0):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((6, 5, 4, T)) + offset).astype(np.float32)
    path = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path, data


def _gain(x, y):
    """Amplitude gain per frequency (output / input), averaged over voxels."""
    x, y = x.reshape(-1, T), y.reshape(-1, T)
    f, px = signal.welch(x - x.mean(1, keepdims=True), fs=1 / TR, nperseg=200, axis=1)
    _, py = signal.welch(y - y.mean(1, keepdims=True), fs=1 / TR, nperseg=200, axis=1)
    return f, np.sqrt(py.mean(0) / px.mean(0))


def _at(f, g, hz):
    return float(np.interp(hz, f, g))


def test_bandpass_cutoffs_are_where_configured(tmp_path):
    src, x = _noise_image(tmp_path)
    out = temporal_filter(src, tmp_path / "out.nii.gz", TR, highpass=0.01, lowpass=0.1)
    y = nib.load(str(out)).get_fdata()
    f, g = _gain(x, y)
    assert _at(f, g, 0.03) > 0.9 and _at(f, g, 0.06) > 0.9      # pass band
    assert 0.4 < _at(f, g, 0.1) < 0.6                             # zero-phase: 1/2 at the cutoff
    assert _at(f, g, 0.2) < 0.1                                   # stop band
    assert _at(f, g, 0.0025) < 0.15
    assert np.abs(y.mean(-1)).max() < 1e-3                        # band-pass removes the mean, as -bptf did


def test_lowpass_only_keeps_the_mean_and_highpass_only_removes_it(tmp_path):
    src, x = _noise_image(tmp_path)
    lp = nib.load(str(temporal_filter(src, tmp_path / "lp.nii.gz", TR, lowpass=0.1))).get_fdata()
    hp = nib.load(str(temporal_filter(src, tmp_path / "hp.nii.gz", TR, highpass=0.01))).get_fdata()
    assert np.allclose(lp.mean(-1), x.mean(-1), atol=1e-3)
    assert np.abs(hp.mean(-1)).max() < 1e-3
    f, g = _gain(x, hp)
    assert _at(f, g, 0.1) > 0.95 and _at(f, g, 0.2) > 0.95


def test_lowpass_at_nyquist_is_ignored_and_bad_bands_raise(tmp_path):
    src, x = _noise_image(tmp_path)
    y = nib.load(str(temporal_filter(src, tmp_path / "o.nii.gz", TR, highpass=0.01, lowpass=0.25))).get_fdata()
    f, g = _gain(x, y)
    assert _at(f, g, 0.2) > 0.95
    with pytest.raises(ValueError):
        temporal_filter(src, tmp_path / "bad.nii.gz", TR, highpass=0.1, lowpass=0.05)
    with pytest.raises(ValueError):
        temporal_filter(src, tmp_path / "bad.nii.gz", TR, highpass=0.01, lowpass=0.1, method="gaussian")


@pytest.mark.skipif(shutil.which("fslmaths") is None, reason="fslmaths not installed")
def test_legacy_bptf_low_pass_sits_far_below_its_configured_cutoff(tmp_path):
    """Documents the pre-fix behaviour that method='fsl_bptf' reproduces."""
    src, x = _noise_image(tmp_path)
    out = temporal_filter(src, tmp_path / "old.nii.gz", TR, highpass=0.01, lowpass=0.1, method="fsl_bptf")
    f, g = _gain(x, nib.load(str(out)).get_fdata())
    assert _at(f, g, 0.06) < 0.25 and _at(f, g, 0.1) < 0.05
