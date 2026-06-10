"""
Unit tests for the bet4animal sweep's reference-free scoring + selection.

Hermetic: the bet4animal subprocess (FSL) is integration-tier and not exercised
here. We test the pure logic — score_mask metrics and select_best — which is what
turns the parameter sweep into a choice.
"""
import numpy as np
import pytest

from neurofaune.preprocess.utils.bet4animal import (
    score_mask,
    select_best,
    cog_from_foreground,
)


def _sphere(shape, center, radius):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius ** 2


class TestScoreMask:
    def test_empty_mask_returns_none(self):
        d = np.ones((10, 10, 10))
        g = np.ones_like(d)
        assert score_mask(np.zeros((10, 10, 10), bool), d, g, fg_thr=0.5) is None

    def test_metrics_basic(self):
        shape = (40, 40, 40)
        # bright sphere on a dark (sub-threshold) background
        d = np.full(shape, 1.0)
        m = _sphere(shape, (20, 20, 20), 10)
        d[m] = 100.0
        gz, gy, gx = np.gradient(d)
        gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        gmag_n = gmag / np.percentile(gmag, 99)
        sc = score_mask(m, d, gmag_n, fg_thr=50.0)
        assert sc["ncomp"] == 1
        assert sc["frac_largest"] == pytest.approx(1.0)
        assert sc["air"] == pytest.approx(0.0)        # whole sphere is bright
        assert sc["cov"] == pytest.approx(100.0 * m.mean())
        assert sc["bgrad"] > 0

    def test_air_leakage_detects_balloon(self):
        shape = (40, 40, 40)
        d = np.full(shape, 1.0)
        brain = _sphere(shape, (20, 20, 20), 8)
        d[brain] = 100.0
        g = np.ones_like(d)
        tight = brain
        ballooned = _sphere(shape, (20, 20, 20), 16)  # extends into dark background
        sc_tight = score_mask(tight, d, g, fg_thr=50.0)
        sc_balloon = score_mask(ballooned, d, g, fg_thr=50.0)
        assert sc_tight["air"] < sc_balloon["air"]
        assert sc_balloon["air"] > 0.3


class TestSelectBest:
    def _cand(self, zscale, cov, nsl, air, bgrad, frac=1.0):
        return dict(zscale=zscale, radius=125, cov=cov, zlo=0, zhi=nsl - 1,
                    nsl=nsl, ncomp=1, frac_largest=frac, air=air, bgrad=bgrad)

    def test_picks_peak_boundary_fit(self):
        cands = [
            self._cand(2.0, 8.0, 24, 0.16, 0.300),
            self._cand(1.5, 8.7, 29, 0.16, 0.326),  # peak fit
            self._cand(1.0, 12.0, 38, 0.22, 0.300),
        ]
        best = select_best([dict(c) for c in cands])
        assert best["zscale"] == 1.5

    def test_largest_extent_among_near_peak(self):
        # Two within 95% of peak fit -> prefer the larger z-extent (poles kept).
        cands = [
            self._cand(1.8, 7.6, 26, 0.13, 0.342),  # peak fit but tighter
            self._cand(1.6, 8.0, 28, 0.14, 0.326),  # within 95%, more brain
        ]
        best = select_best([dict(c) for c in cands])
        assert best["zscale"] == 1.6  # bias toward coverage, no pole clipping

    def test_rejects_ballooned_via_air_gate(self):
        # The ballooned candidate has the highest fit but fails the air gate.
        cands = [
            self._cand(1.0, 16.0, 41, 0.42, 0.400),  # invalid: air too high
            self._cand(1.5, 8.7, 29, 0.16, 0.320),   # valid
        ]
        best = select_best([dict(c) for c in cands], air_max=0.20)
        assert best["zscale"] == 1.5

    def test_rejects_out_of_coverage_range(self):
        cands = [
            self._cand(3.0, 1.5, 10, 0.10, 0.500),   # too small (clipped)
            self._cand(1.5, 8.7, 29, 0.16, 0.320),
        ]
        best = select_best([dict(c) for c in cands], cov_range=(4.0, 13.0))
        assert best["zscale"] == 1.5

    def test_empty_returns_none(self):
        assert select_best([]) is None

    def test_fallback_when_none_valid(self):
        # All fail the gate -> still return the best-scoring rather than None.
        cands = [
            self._cand(1.0, 16.0, 41, 0.42, 0.40),
            self._cand(0.5, 20.0, 41, 0.55, 0.25),
        ]
        best = select_best([dict(c) for c in cands])
        assert best is not None
        assert best["zscale"] == 1.0  # higher fit among the invalid pool


class TestCogFromForeground:
    def test_centroid_of_bright_blob(self):
        shape = (40, 40, 40)
        d = np.full(shape, 5.0)          # low but non-zero floor
        blob = _sphere(shape, (18, 22, 20), 6)
        d[blob] = 3000.0
        cog = cog_from_foreground(d, k=4.0)
        # COG should be near the blob centre, not the FOV centre.
        assert abs(cog[0] - 18) < 3
        assert abs(cog[1] - 22) < 3
        assert abs(cog[2] - 20) < 3
