"""Unit tests for single-session resting-state helpers (neurofaune.analysis.func.hub_rest)."""

import numpy as np
import pytest

from neurofaune.analysis.func.hub_rest import (
    bandpass, connectivity, denoise, dual_regression, homotopy, motion_confounds, parcel_timeseries,
    spatial_match,
)


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    tr = 2.0
    t = np.arange(300) * tr
    slow, fast, mid = np.sin(2 * np.pi * 0.002 * t), np.sin(2 * np.pi * 0.2 * t), np.sin(2 * np.pi * 0.04 * t)
    y = bandpass((slow + fast + mid)[:, None], tr, 0.01, 0.1)[:, 0]
    assert np.corrcoef(y[20:-20], mid[20:-20])[0, 1] > 0.98


def test_denoise_removes_a_confound_shared_by_all_voxels():
    rng = np.random.default_rng(1)
    n = 300
    conf = bandpass(rng.normal(size=(n, 1)), 2.0, 0.01, 0.1)
    sig = bandpass(rng.normal(size=(n, 1)), 2.0, 0.01, 0.1)
    data = np.zeros((4, 1, 1, n))
    for v in range(4):
        data[v, 0, 0] = 5 * conf[:, 0] + (sig[:, 0] if v < 2 else rng.normal(size=n) * 0.1)
    out = denoise(data, np.ones((4, 1, 1), bool), conf, tr=2.0, drop=0)
    assert abs(np.corrcoef(out[0, 0, 0], conf[:, 0])[0, 1]) < 0.05
    assert np.corrcoef(out[0, 0, 0], out[1, 0, 0])[0, 1] > 0.9


def test_motion_confounds_shape():
    assert motion_confounds(np.zeros((10, 6))).shape == (10, 12)


def test_parcels_connectivity_and_homotopy():
    rng = np.random.default_rng(2)
    shared = rng.normal(size=200)
    data = np.zeros((4, 2, 1, 200))
    data[0] = shared + 0.3 * rng.normal(size=(2, 1, 200))       # A_L
    data[1] = shared + 0.3 * rng.normal(size=(2, 1, 200))       # A_R
    data[2] = rng.normal(size=(2, 1, 200))                      # B_L
    data[3] = rng.normal(size=(2, 1, 200))                      # B_R
    parcels = np.array([1, 2, 3, 4])[:, None, None] * np.ones((4, 2, 1), int)
    ts, kept = parcel_timeseries(data, parcels, [1, 2, 3, 4], min_voxels=2)
    z = connectivity(ts)
    h = homotopy(z, kept, {1: "A_L", 2: "A_R", 3: "B_L", 4: "B_R"})
    assert h["homotopic"]["A"] > 1.0 and abs(h["homotopic"]["B"]) < 0.3
    assert h["median_heterotopic_z"] < h["homotopic"]["A"]


def test_dual_regression_recovers_network_maps():
    rng = np.random.default_rng(3)
    shape = (10, 10, 4)
    priors = np.zeros(shape + (2,))
    priors[:4, :, :, 0] = 1.0          # two networks plus background voxels in neither,
    priors[6:, :, :, 1] = 1.0          # so the demeaned priors are not collinear
    tcs = rng.normal(size=(2, 300))
    data = np.einsum("xyzk,kt->xyzt", priors, tcs) + 0.5 * rng.normal(size=shape + (300,))
    mask = np.ones(shape, bool)
    dr = dual_regression(data, priors, mask)
    match = spatial_match(dr["maps"], priors, mask)
    assert (match > 0.9).all()
    assert abs(np.corrcoef(dr["timecourses"][:, 0], tcs[0])[0, 1]) > 0.95


def test_slice_overlap_and_trend_removal():
    from neurofaune.analysis.func.hub_rest import remove_overlap_trend, slice_overlap
    parc = np.zeros((4, 4, 6), int)
    parc[:2, :, :3] = 1; parc[2:, :, :3] = 2      # same slices
    parc[:2, :, 3:] = 3                           # different slices
    S = slice_overlap(parc, [1, 2, 3])
    assert S[0, 1] == pytest.approx(1.0) and S[0, 2] == pytest.approx(0.0)
    z = np.array([[0, .5, .1], [.5, 0, .1], [.1, .1, 0]])
    zr = remove_overlap_trend(z, S)
    assert abs(zr[0, 1]) < 1e-9 and abs(zr[0, 2]) < 1e-9
