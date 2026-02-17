#!/usr/bin/env python3
"""
Unit tests for whole-network similarity tests.

Tests Mantel test, Frobenius distance, spectral divergence,
and the permutation-based whole_network_test function.
"""

import numpy as np
import pytest

from neurofaune.analysis.covnet.whole_network import (
    _upper_tri,
    frobenius_distance,
    mantel_test,
    run_all_comparisons,
    spectral_divergence,
    whole_network_test,
)


@pytest.fixture
def identical_matrices():
    """Two identical correlation matrices."""
    rng = np.random.default_rng(0)
    n = 10
    A = rng.standard_normal((n, n))
    corr = np.corrcoef(A.T)
    return corr, corr.copy()


@pytest.fixture
def different_matrices():
    """Two clearly different correlation matrices."""
    rng = np.random.default_rng(1)
    n = 10
    A = rng.standard_normal((50, n))
    B = rng.standard_normal((50, n))
    corr_a = np.corrcoef(A.T)
    corr_b = np.corrcoef(B.T)
    return corr_a, corr_b


@pytest.fixture
def group_data():
    """Synthetic group data for permutation tests."""
    rng = np.random.default_rng(42)
    n_rois = 8
    # Group A: correlated structure
    base_a = rng.standard_normal((20, n_rois))
    # Group B: same distribution (null should not reject)
    base_b = rng.standard_normal((18, n_rois))
    return base_a, base_b


class TestUpperTri:
    def test_shape(self):
        n = 5
        mat = np.arange(25).reshape(5, 5).astype(float)
        result = _upper_tri(mat)
        expected_len = n * (n - 1) // 2
        assert result.shape == (expected_len,)

    def test_values(self):
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = _upper_tri(mat)
        np.testing.assert_array_equal(result, [2, 3, 6])


class TestMantelTest:
    def test_identical_matrices_r_equals_1(self, identical_matrices):
        corr_a, corr_b = identical_matrices
        r = mantel_test(corr_a, corr_b)
        assert abs(r - 1.0) < 1e-10

    def test_different_matrices_r_below_1(self, different_matrices):
        corr_a, corr_b = different_matrices
        r = mantel_test(corr_a, corr_b)
        assert r < 1.0

    def test_returns_float(self, identical_matrices):
        corr_a, corr_b = identical_matrices
        r = mantel_test(corr_a, corr_b)
        assert isinstance(r, float)

    def test_symmetric(self, different_matrices):
        corr_a, corr_b = different_matrices
        r_ab = mantel_test(corr_a, corr_b)
        r_ba = mantel_test(corr_b, corr_a)
        assert abs(r_ab - r_ba) < 1e-10


class TestFrobeniusDistance:
    def test_identical_is_zero(self, identical_matrices):
        corr_a, corr_b = identical_matrices
        d = frobenius_distance(corr_a, corr_b)
        assert abs(d) < 1e-10

    def test_different_is_positive(self, different_matrices):
        corr_a, corr_b = different_matrices
        d = frobenius_distance(corr_a, corr_b)
        assert d > 0

    def test_symmetric(self, different_matrices):
        corr_a, corr_b = different_matrices
        d_ab = frobenius_distance(corr_a, corr_b)
        d_ba = frobenius_distance(corr_b, corr_a)
        assert abs(d_ab - d_ba) < 1e-10

    def test_triangle_inequality(self):
        """Frobenius distance satisfies the triangle inequality."""
        rng = np.random.default_rng(99)
        n = 6
        A = np.corrcoef(rng.standard_normal((20, n)).T)
        B = np.corrcoef(rng.standard_normal((20, n)).T)
        C = np.corrcoef(rng.standard_normal((20, n)).T)
        d_ab = frobenius_distance(A, B)
        d_bc = frobenius_distance(B, C)
        d_ac = frobenius_distance(A, C)
        assert d_ac <= d_ab + d_bc + 1e-10


class TestSpectralDivergence:
    def test_identical_is_zero(self, identical_matrices):
        corr_a, corr_b = identical_matrices
        d = spectral_divergence(corr_a, corr_b)
        assert abs(d) < 1e-10

    def test_different_is_positive(self, different_matrices):
        corr_a, corr_b = different_matrices
        d = spectral_divergence(corr_a, corr_b)
        assert d > 0

    def test_symmetric(self, different_matrices):
        corr_a, corr_b = different_matrices
        d_ab = spectral_divergence(corr_a, corr_b)
        d_ba = spectral_divergence(corr_b, corr_a)
        assert abs(d_ab - d_ba) < 1e-10

    def test_identity_vs_full_correlation(self):
        """Identity matrix (all independent) vs fully correlated should differ."""
        n = 5
        identity = np.eye(n)
        full = np.ones((n, n))
        d = spectral_divergence(identity, full)
        assert d > 0


class TestWholeNetworkTest:
    def test_basic_output_structure(self, group_data):
        data_a, data_b = group_data
        result = whole_network_test(data_a, data_b, n_perm=50, seed=42)

        assert "mantel_r" in result
        assert "mantel_p" in result
        assert "frobenius_d" in result
        assert "frobenius_p" in result
        assert "spectral_d" in result
        assert "spectral_p" in result
        assert "null_distributions" in result
        assert "n_a" in result
        assert "n_b" in result

    def test_p_values_in_range(self, group_data):
        data_a, data_b = group_data
        result = whole_network_test(data_a, data_b, n_perm=50, seed=42)
        for key in ["mantel_p", "frobenius_p", "spectral_p"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]}"

    def test_null_distribution_length(self, group_data):
        data_a, data_b = group_data
        n_perm = 100
        result = whole_network_test(data_a, data_b, n_perm=n_perm, seed=42)
        for key in ["mantel", "frobenius", "spectral"]:
            assert len(result["null_distributions"][key]) == n_perm

    def test_group_sizes(self, group_data):
        data_a, data_b = group_data
        result = whole_network_test(data_a, data_b, n_perm=50, seed=42)
        assert result["n_a"] == data_a.shape[0]
        assert result["n_b"] == data_b.shape[0]

    def test_reproducible_with_same_seed(self, group_data):
        data_a, data_b = group_data
        r1 = whole_network_test(data_a, data_b, n_perm=50, seed=123)
        r2 = whole_network_test(data_a, data_b, n_perm=50, seed=123)
        assert r1["mantel_r"] == r2["mantel_r"]
        assert r1["mantel_p"] == r2["mantel_p"]

    def test_identical_groups_high_mantel_r(self):
        """When both groups share correlated structure, Mantel r should be
        positive (more similar than random)."""
        rng = np.random.default_rng(7)
        n_rois = 6
        # Create latent factor structure so correlations are non-trivial
        latent = rng.standard_normal((80, 2))
        loadings = rng.standard_normal((2, n_rois))
        data = latent @ loadings + 0.5 * rng.standard_normal((80, n_rois))
        data_a = data[:40]
        data_b = data[40:]
        result = whole_network_test(data_a, data_b, n_perm=50, seed=42)
        # Shared structure → Mantel r should be positive
        assert result["mantel_r"] > 0.0


class TestRunAllComparisons:
    def test_with_explicit_comparisons(self):
        rng = np.random.default_rng(42)
        group_data = {
            "p60_C": rng.standard_normal((15, 6)),
            "p60_L": rng.standard_normal((12, 6)),
            "p60_H": rng.standard_normal((10, 6)),
        }
        comparisons = [("p60_L", "p60_C"), ("p60_H", "p60_C")]
        df, nulls = run_all_comparisons(
            group_data, comparisons=comparisons, n_perm=50, seed=42
        )
        assert len(df) == 2
        assert set(df.columns) == {
            "comparison", "group_a", "group_b", "n_a", "n_b",
            "mantel_r", "mantel_p", "frobenius_d", "frobenius_p",
            "spectral_d", "spectral_p",
        }
        assert len(nulls) == 2

    def test_default_comparisons(self):
        rng = np.random.default_rng(42)
        group_data = {
            "p60_C": rng.standard_normal((15, 5)),
            "p60_L": rng.standard_normal((12, 5)),
            "p60_M": rng.standard_normal((10, 5)),
            "p60_H": rng.standard_normal((10, 5)),
        }
        df, nulls = run_all_comparisons(
            group_data, n_perm=50, seed=42
        )
        # 3 dose vs control for p60
        assert len(df) == 3

    def test_missing_group_skipped(self):
        rng = np.random.default_rng(42)
        group_data = {
            "p60_C": rng.standard_normal((15, 5)),
            "p60_L": rng.standard_normal((12, 5)),
        }
        comparisons = [("p60_L", "p60_C"), ("p60_MISSING", "p60_C")]
        df, _ = run_all_comparisons(
            group_data, comparisons=comparisons, n_perm=50, seed=42
        )
        assert len(df) == 1
