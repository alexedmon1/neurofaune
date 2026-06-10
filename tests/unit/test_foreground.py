"""
Unit tests for the shared foreground / noise-floor module (gh #12).

These are hermetic (no ANTs/FSL): they exercise the noise-floor estimate and the
initial foreground mask directly. The end-to-end "atropos_bet gives a brain-tight
mask" check belongs to the integration tier (real ANTs) and the per-subject
visual QC, not here.
"""

import numpy as np
import pytest

from neurofaune.preprocess.utils.foreground import (
    NoiseFloor,
    estimate_noise_floor,
    foreground_mask,
)


def _rayleigh_floor(shape, sigma, seed=0):
    """Rayleigh-distributed magnitude noise (what air looks like in |MR|)."""
    rng = np.random.RandomState(seed)
    re = rng.normal(0.0, sigma, size=shape)
    im = rng.normal(0.0, sigma, size=shape)
    return np.sqrt(re ** 2 + im ** 2).astype(np.float32)


def _head_phantom(shape=(48, 48, 24), sigma=8.0, seed=1):
    """Non-zero air floor + a bright central 'head' blob.

    Mimics the failure-mode geometry: most of the FOV is low-but-non-zero air,
    with a compact bright structure in the middle.
    """
    data = _rayleigh_floor(shape, sigma, seed=seed)
    cx, cy, cz = (s // 2 for s in shape)
    rx, ry, rz = shape[0] // 5, shape[1] // 5, shape[2] // 4
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    blob = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2 <= 1.0
    data[blob] += 2000.0
    return data, blob


class TestEstimateNoiseFloor:
    def test_scale_aware_no_mask(self):
        # The no-mask lowest-quartile estimate is deliberately biased low (so is
        # MSME's lowest-quartile-of-background); k absorbs the bias. The contract
        # is *scale-linearity*: double the intensity scale -> double the floor.
        base = _rayleigh_floor((40, 40, 40), 12.0, seed=3)
        f1 = estimate_noise_floor(base, mask=None)
        f2 = estimate_noise_floor(base * 2.0, mask=None)
        assert isinstance(f1, NoiseFloor)
        assert f1.mean > 0 and f1.sigma > 0
        assert f2.mean == pytest.approx(2.0 * f1.mean, rel=1e-6)
        assert f2.sigma == pytest.approx(2.0 * f1.sigma, rel=1e-6)
        assert f1.sigma2 == pytest.approx(f1.sigma ** 2)

    def test_raises_on_all_zero(self):
        with pytest.raises(ValueError):
            estimate_noise_floor(np.zeros((8, 8, 8)), mask=None)

    def test_masked_fallback_triggers_when_mask_covers_image(self):
        data, _ = _head_phantom()
        full = np.ones(data.shape, dtype=bool)  # mask covers everything
        floor = estimate_noise_floor(data, mask=full)
        assert floor.used_fallback is True
        assert floor.n_noise_voxels > 0

    def test_masked_path_no_fallback_when_background_present(self):
        data, blob = _head_phantom()
        floor = estimate_noise_floor(data, mask=blob)
        assert floor.used_fallback is False


class TestMsmeBitIdentical:
    """The shared estimator must reproduce the MSME background arithmetic
    exactly, so msme_preprocess can delegate without moving any number."""

    def test_matches_inline_msme_computation(self):
        data, blob = _head_phantom(shape=(50, 50, 30), sigma=15.0, seed=7)
        mask_3d = blob

        # --- verbatim MSME logic (msme_preprocess.py lines ~600-621) ---
        first_echo_all = data
        first_echo_bg = first_echo_all[~mask_3d]
        first_echo_bg = first_echo_bg[first_echo_bg > 0]
        if len(first_echo_bg) < 100:
            all_nonzero = first_echo_all[first_echo_all > 0]
            q10 = np.percentile(all_nonzero, 10)
            noise_voxels = all_nonzero[all_nonzero <= q10]
        else:
            q25 = np.percentile(first_echo_bg, 25)
            noise_voxels = first_echo_bg[first_echo_bg <= q25]
        noise_mean = float(np.mean(noise_voxels))
        noise_sigma2 = noise_mean ** 2 * (2.0 / np.pi)
        # ----------------------------------------------------------------

        floor = estimate_noise_floor(first_echo_all, mask=mask_3d)
        assert floor.mean == noise_mean
        assert floor.sigma2 == noise_sigma2  # exact, not approx
        assert floor.n_noise_voxels == len(noise_voxels)

    def test_matches_inline_msme_fallback(self):
        # Force the < 100 background-voxel fallback branch.
        data, _ = _head_phantom(shape=(20, 20, 10), sigma=10.0, seed=9)
        mask_3d = np.ones(data.shape, dtype=bool)
        mask_3d.flat[:50] = False  # 50 background voxels (< 100) -> fallback

        first_echo_all = data
        first_echo_bg = first_echo_all[~mask_3d]
        first_echo_bg = first_echo_bg[first_echo_bg > 0]
        assert len(first_echo_bg) < 100
        all_nonzero = first_echo_all[first_echo_all > 0]
        q10 = np.percentile(all_nonzero, 10)
        noise_voxels = all_nonzero[all_nonzero <= q10]
        noise_mean = float(np.mean(noise_voxels))
        noise_sigma2 = noise_mean ** 2 * (2.0 / np.pi)

        floor = estimate_noise_floor(first_echo_all, mask=mask_3d)
        assert floor.used_fallback is True
        assert floor.mean == noise_mean
        assert floor.sigma2 == noise_sigma2


class TestForegroundMask:
    def test_excludes_nonzero_floor(self):
        # THE bug: 5th-percentile seed keeps ~95% here; foreground_mask must not.
        data, blob = _head_phantom()
        nonzero = data[data > 0]
        old_seed = (data > np.percentile(nonzero, 5)).mean()
        assert old_seed > 0.9  # reproduce the broken behavior

        fg = foreground_mask(data, k=4.0)
        coverage = fg.mean()
        blob_frac = blob.mean()
        # Tight: a few× the blob, nowhere near whole-image.
        assert coverage < 0.25
        assert coverage >= blob_frac
        # Nearly all blob voxels survive the threshold.
        assert fg[blob].mean() > 0.95

    def test_scale_invariance(self):
        # mu and sigma both scale linearly with intensity, so the relative
        # threshold mask is invariant to a global intensity rescale.
        data, _ = _head_phantom()
        a = foreground_mask(data, k=4.0)
        b = foreground_mask(data * 260.0, k=4.0)
        assert np.array_equal(a, b)

    def test_accepts_precomputed_floor(self):
        data, _ = _head_phantom()
        floor = estimate_noise_floor(data, mask=None)
        a = foreground_mask(data, k=3.5, floor=floor)
        b = foreground_mask(data, k=3.5)
        assert np.array_equal(a, b)

    def test_higher_k_shrinks_mask(self):
        data, _ = _head_phantom()
        assert foreground_mask(data, k=6.0).sum() <= foreground_mask(data, k=3.0).sum()
