"""Unit tests for DWI eddy per-volume SNR (``_plot_volume_metrics``).

Pins the corrected estimator so the historical failure cannot return: SNR is
mean(brain signal) / std(background noise), using a supplied brain mask (or a
hot-voxel-robust percentile fallback) — not mean/std *within* the brain behind a
``0.1 * max(data)`` threshold, which collapsed to 0 (empty mask) or exploded to
~1e5 (tiny near-uniform mask) depending on a single bright voxel.
"""
import math

import numpy as np
import pytest

from neurofaune.preprocess.qc.dwi.eddy_qc import _plot_volume_metrics

SHAPE = (40, 40, 12)


def _phantom(sigma=50.0, signal=1000.0, n=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, sigma, size=SHAPE + (n,))  # background noise ~ N(0, sigma)
    brain = np.zeros(SHAPE, dtype=bool)
    brain[15:25, 15:25, 4:8] = True
    data[brain] += signal
    return data, brain, sigma, signal


def test_snr_is_signal_over_background_noise(tmp_path):
    data, brain, sigma, signal = _phantom()
    metrics, _ = _plot_volume_metrics(data, "s", "ses-1", tmp_path, brain_mask=brain)
    # SNR ~ mean(brain ~ signal) / std(background ~ sigma) = 1000/50 = 20
    assert metrics["mean_snr"] == pytest.approx(signal / sigma, rel=0.2)
    assert np.isfinite(metrics["min_snr"]) and np.isfinite(metrics["snr_std"])


def test_fallback_mask_robust_to_hot_voxel(tmp_path):
    # No mask passed + a single hot voxel: the old 0.1*max(data) threshold would
    # empty the mask (-> SNR 0). The percentile fallback must stay non-degenerate.
    data, brain, sigma, signal = _phantom(seed=1)
    data = np.abs(data)               # magnitude-like, so background is positive
    data[0, 0, 0, :] = 1e6            # hot voxel
    metrics, _ = _plot_volume_metrics(data, "s", "ses-1", tmp_path, brain_mask=None)
    assert np.isfinite(metrics["mean_snr"])
    assert metrics["mean_snr"] > 1.0   # not the old 0.000


def test_degenerate_no_background_returns_nan(tmp_path):
    # Brain mask covers the whole FOV -> no background -> NaN, never 0 or huge.
    data = np.ones((10, 10, 4, 2)) * 100.0
    brain = np.ones((10, 10, 4), dtype=bool)
    metrics, _ = _plot_volume_metrics(data, "s", "ses-1", tmp_path, brain_mask=brain)
    assert math.isnan(metrics["mean_snr"])


def test_hot_voxel_does_not_explode_snr_with_mask(tmp_path):
    # With a real mask, a hot voxel inside the brain must not produce ~1e5 SNR.
    data, brain, sigma, signal = _phantom(seed=2)
    data[16, 16, 5, :] = 1e6  # hot voxel within brain
    metrics, _ = _plot_volume_metrics(data, "s", "ses-1", tmp_path, brain_mask=brain)
    assert metrics["mean_snr"] < 1e3  # bounded; old within-brain ratio could blow up
