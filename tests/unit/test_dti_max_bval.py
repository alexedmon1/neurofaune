"""Unit tests for restricting the DTI tensor fit to the Gaussian regime.

The DTI model assumes mono-exponential (Gaussian) decay, which only holds to
b ~= 1000. Fitting a single tensor across high-b shells deflates diffusivity and
inflates FA. Measured on the cuprizone rat protocol (b0/1018/2021/3025),
direction-averaged ADC falls 0.643 -> 0.541 -> 0.438 e-3 mm2/s across the three
shells - a 32% drop, i.e. strongly non-Gaussian.

``fit_dti(max_bval=...)`` therefore subsets to b0 + b <= max_bval before calling
dtifit. DKI/NODDI keep all shells; they model the non-Gaussianity deliberately.

These tests intercept the dtifit call and inspect exactly what was handed to it.
"""
import subprocess as _subprocess

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.workflows import dwi_preprocess
from neurofaune.preprocess.workflows.dwi_preprocess import fit_dti

SHAPE = (6, 6, 3)
# b0 x5 + 30 each at b1018 / b2021 / b3025, as acquired
BVALS = np.concatenate([
    np.zeros(5), np.full(30, 1018.0), np.full(30, 2021.0), np.full(30, 3025.0)
])


@pytest.fixture
def dwi_inputs(tmp_path):
    rng = np.random.default_rng(0)
    n = BVALS.size
    data = rng.uniform(100, 1000, SHAPE + (n,))
    dwi = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), dwi)

    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.uint8), np.eye(4)), mask)

    bval = tmp_path / "dwi.bval"
    np.savetxt(bval, BVALS[None, :], fmt="%g")

    bvecs = rng.normal(size=(3, n))
    bvecs /= np.linalg.norm(bvecs, axis=0)
    bvecs[:, BVALS < 100] = 0.0
    bvec = tmp_path / "dwi.bvec"
    np.savetxt(bvec, bvecs, fmt="%.6f")
    return dwi, mask, bval, bvec


@pytest.fixture
def capture_dtifit(monkeypatch, tmp_path):
    """Replace dtifit with a stub that records its args and emits the outputs."""
    seen = {}

    def fake_run(cmd, **kwargs):
        args = {c.split("=", 1)[0].lstrip("-"): c.split("=", 1)[1]
                for c in cmd[1:] if "=" in c}
        seen["cmd"] = cmd
        seen["bvals"] = np.atleast_1d(np.loadtxt(args["bvals"]))
        seen["bvecs"] = np.loadtxt(args["bvecs"])
        seen["n_vols"] = nib.load(args["data"]).shape[-1]
        for suffix in ("FA", "MD", "L1", "L2", "L3"):
            nib.save(nib.Nifti1Image(np.ones(SHAPE, np.float32), np.eye(4)),
                     f"{args['out']}_{suffix}.nii.gz")
        return _subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(dwi_preprocess.subprocess, "run", fake_run)
    return seen


def test_subsets_to_b0_plus_low_shell(dwi_inputs, capture_dtifit, tmp_path):
    dwi, mask, bval, bvec = dwi_inputs
    fit_dti(dwi, mask, bval, bvec, tmp_path / "sub", max_bval=1500.0)

    got = capture_dtifit["bvals"]
    # 5 b0 + 30 at b1018; b2021/b3025 excluded
    assert got.size == 35
    assert capture_dtifit["n_vols"] == 35
    assert capture_dtifit["bvecs"].shape == (3, 35)
    assert (got <= 1500.0).all()
    assert (got < 100).sum() == 5, "all b0 volumes must be retained"
    assert not np.isin(got, [2021.0, 3025.0]).any()


def test_none_keeps_all_shells(dwi_inputs, capture_dtifit, tmp_path):
    """Legacy behaviour — and what DKI/NODDI rely on."""
    dwi, mask, bval, bvec = dwi_inputs
    fit_dti(dwi, mask, bval, bvec, tmp_path / "sub", max_bval=None)

    assert capture_dtifit["bvals"].size == 95
    assert capture_dtifit["n_vols"] == 95


def test_threshold_above_all_shells_is_a_noop(dwi_inputs, capture_dtifit, tmp_path):
    dwi, mask, bval, bvec = dwi_inputs
    fit_dti(dwi, mask, bval, bvec, tmp_path / "sub", max_bval=9999.0)

    assert capture_dtifit["bvals"].size == 95
    assert capture_dtifit["n_vols"] == 95


def test_bvec_columns_track_the_retained_volumes(dwi_inputs, capture_dtifit, tmp_path):
    """A bval/bvec/volume mismatch would silently corrupt the tensor."""
    dwi, mask, bval, bvec = dwi_inputs
    full = np.loadtxt(bvec)
    fit_dti(dwi, mask, bval, bvec, tmp_path / "sub", max_bval=1500.0)

    expected = full[:, BVALS <= 1500.0]
    assert np.allclose(capture_dtifit["bvecs"], expected, atol=1e-6)
