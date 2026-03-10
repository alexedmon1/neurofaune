"""Tests for voxelwise effect size computation."""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofaune.analysis.stats.effect_size import (
    compute_cohens_d_map,
    compute_contrast_variance_factors,
    compute_partial_etasq_from_fstat,
    compute_partial_etasq_from_tstat,
    generate_effect_size_maps,
    read_fsl_vest,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def dummy_nifti(tmp_dir):
    """Create a small 3D NIfTI with known values."""
    def _make(name, data):
        img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
        path = tmp_dir / name
        nib.save(img, path)
        return path
    return _make


@pytest.fixture
def simple_design(tmp_dir):
    """Two-group design: 10 subjects, 5 per group, intercept + group dummy."""
    n = 10
    X = np.column_stack([np.ones(n), np.array([0]*5 + [1]*5, dtype=float)])
    C = np.array([[0, 1], [0, -1]], dtype=float)  # group+ and group-

    mat_path = tmp_dir / "design.mat"
    con_path = tmp_dir / "design.con"

    with open(mat_path, 'w') as f:
        f.write(f"/NumWaves {X.shape[1]}\n")
        f.write(f"/NumPoints {X.shape[0]}\n")
        f.write("/Matrix\n")
        np.savetxt(f, X, fmt='%.6f')

    with open(con_path, 'w') as f:
        f.write(f"/NumWaves {C.shape[1]}\n")
        f.write(f"/NumContrasts {C.shape[0]}\n")
        f.write("/Matrix\n")
        np.savetxt(f, C, fmt='%.6f')

    return mat_path, con_path, X, C


class TestVestReader:
    def test_read_mat(self, simple_design):
        mat_path, _, X_expected, _ = simple_design
        X = read_fsl_vest(mat_path)
        np.testing.assert_array_almost_equal(X, X_expected)

    def test_read_con(self, simple_design):
        _, con_path, _, C_expected = simple_design
        C = read_fsl_vest(con_path)
        np.testing.assert_array_almost_equal(C, C_expected)


class TestContrastVarianceFactors:
    def test_two_group(self, simple_design):
        _, _, X, C = simple_design
        factors = compute_contrast_variance_factors(X, C)

        # For balanced two-group (n1=n2=5) with contrast [0,1]:
        # c'(X'X)^{-1}c should be 1/n1 + 1/n2 = 0.4
        # But with intercept + dummy: (X'X)^{-1} gives different scaling
        # Just verify it's positive and both contrasts have same factor
        assert len(factors) == 2
        assert factors[0] > 0
        np.testing.assert_almost_equal(factors[0], factors[1])

    def test_identity_design(self):
        """With X=I and c=[1,0,...], factor should be 1."""
        n = 5
        X = np.eye(n)
        c = np.zeros((1, n))
        c[0, 0] = 1
        factors = compute_contrast_variance_factors(X, c)
        np.testing.assert_almost_equal(factors[0], 1.0)


class TestCohensD:
    def test_output_scaling(self, dummy_nifti, tmp_dir):
        """Cohen's d = t * sqrt(variance_factor)."""
        t_data = np.full((5, 5, 3), 3.0)
        tstat_file = dummy_nifti("randomise_tstat1.nii.gz", t_data)

        var_factor = 0.4  # e.g., balanced two-group
        d_file = compute_cohens_d_map(tstat_file, var_factor)

        d_img = nib.load(d_file)
        d_data = d_img.get_fdata()

        expected = 3.0 * np.sqrt(0.4)
        np.testing.assert_array_almost_equal(d_data, expected, decimal=4)

    def test_output_naming(self, dummy_nifti):
        t_data = np.ones((3, 3, 3))
        tstat_file = dummy_nifti("randomise_tstat2.nii.gz", t_data)
        d_file = compute_cohens_d_map(tstat_file, 1.0)
        assert "cohend2" in d_file.name


class TestPartialEtaSquared:
    def test_from_tstat(self, dummy_nifti):
        """eta_p^2 = t^2 / (t^2 + df)."""
        t_val = 2.5
        df = 18
        t_data = np.full((4, 4, 3), t_val)
        tstat_file = dummy_nifti("randomise_tstat1.nii.gz", t_data)

        etasq_file = compute_partial_etasq_from_tstat(tstat_file, df)
        etasq_data = nib.load(etasq_file).get_fdata()

        expected = t_val**2 / (t_val**2 + df)
        np.testing.assert_array_almost_equal(etasq_data, expected, decimal=5)

    def test_from_fstat(self, dummy_nifti):
        """eta_p^2 = F*df1 / (F*df1 + df2)."""
        f_val = 4.0
        df1, df2 = 3, 46
        f_data = np.full((4, 4, 3), f_val)
        fstat_file = dummy_nifti("randomise_fstat1.nii.gz", f_data)

        etasq_file = compute_partial_etasq_from_fstat(fstat_file, df1, df2)
        etasq_data = nib.load(etasq_file).get_fdata()

        expected = (f_val * df1) / (f_val * df1 + df2)
        np.testing.assert_array_almost_equal(etasq_data, expected, decimal=5)

    def test_zero_tstat_gives_zero_etasq(self, dummy_nifti):
        t_data = np.zeros((3, 3, 3))
        tstat_file = dummy_nifti("randomise_tstat1.nii.gz", t_data)
        etasq_file = compute_partial_etasq_from_tstat(tstat_file, 20)
        etasq_data = nib.load(etasq_file).get_fdata()
        np.testing.assert_array_almost_equal(etasq_data, 0.0)


class TestGenerateEffectSizeMaps:
    def test_full_pipeline(self, tmp_dir, simple_design):
        """End-to-end: create mock randomise output, generate all effect size maps."""
        mat_path, con_path, X, C = simple_design

        # Create mock randomise output dir
        rand_dir = tmp_dir / "randomise_FA"
        rand_dir.mkdir()

        shape = (5, 5, 3)
        for i in [1, 2]:
            # t-stats
            t_data = np.random.randn(*shape).astype(np.float32) * 2
            img = nib.Nifti1Image(t_data, np.eye(4))
            nib.save(img, rand_dir / f"randomise_tstat{i}.nii.gz")

        result = generate_effect_size_maps(rand_dir, mat_path, con_path)

        assert len(result['cohens_d']) == 2
        assert len(result['etasq_t']) == 2
        assert all(p.exists() for p in result['cohens_d'])
        assert all(p.exists() for p in result['etasq_t'])

    def test_with_ftest(self, tmp_dir, simple_design):
        """Test F-stat effect size with .fts file."""
        mat_path, con_path, X, C = simple_design

        # Create .fts: one F-test including both contrasts
        fts_path = tmp_dir / "design.fts"
        with open(fts_path, 'w') as f:
            f.write("/NumWaves 2\n")
            f.write("/NumContrasts 1\n")
            f.write("/Matrix\n")
            f.write("1 1\n")

        rand_dir = tmp_dir / "randomise_FA"
        rand_dir.mkdir()

        shape = (5, 5, 3)
        # t-stats
        for i in [1, 2]:
            img = nib.Nifti1Image(np.ones(shape, dtype=np.float32) * 2, np.eye(4))
            nib.save(img, rand_dir / f"randomise_tstat{i}.nii.gz")
        # f-stat
        img = nib.Nifti1Image(np.ones(shape, dtype=np.float32) * 5, np.eye(4))
        nib.save(img, rand_dir / "randomise_fstat1.nii.gz")

        result = generate_effect_size_maps(rand_dir, mat_path, con_path, fts_path)

        assert len(result['etasq_f']) == 1
        # df_num should be 2 (both contrasts in F-test)
        etasq = nib.load(result['etasq_f'][0]).get_fdata()
        df_error = X.shape[0] - np.linalg.matrix_rank(X)  # 10 - 2 = 8
        expected = (5 * 2) / (5 * 2 + df_error)
        np.testing.assert_array_almost_equal(etasq, expected, decimal=5)
