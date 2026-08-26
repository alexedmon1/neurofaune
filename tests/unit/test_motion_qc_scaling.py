"""DVARS units and the framewise-displacement threshold on 10x-scaled headers.

Both defects come from human-scale defaults meeting rodent data: DVARS reported
in scanner units (so it ranks brightness, not instability) and an FD cut typed in
millimetres against headers whose voxel sizes are scaled 10x for FSL/ANTs.
"""
import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.qc.func.motion_qc import (
    FD_VOXEL_FRACTION,
    calculate_dvars,
    fd_threshold_from_voxel_size,
)


@pytest.fixture
def bold_and_mask(tmp_path):
    """A 4-D series with a known mean and a known volume-to-volume step."""
    def _make(scale=1.0, zooms=(4.0, 4.0, 8.0)):
        rng = np.random.default_rng(0)
        base = rng.normal(100.0, 1.0, (6, 6, 4, 20)) * scale
        aff = np.diag(list(zooms) + [1.0])
        # Key the filenames on BOTH knobs -- keying on scale alone made two
        # different-geometry requests collide on one file.
        tag = f'{scale}_{"x".join(str(z) for z in zooms)}'
        bold = tmp_path / f'bold_{tag}.nii.gz'
        mask = tmp_path / f'mask_{tag}.nii.gz'
        img = nib.Nifti1Image(base, aff)
        img.header.set_zooms(tuple(zooms) + (1.0,))
        nib.save(img, bold)
        nib.save(nib.Nifti1Image(np.ones((6, 6, 4), dtype=np.uint8), aff), mask)
        return bold, mask
    return _make


class TestDvarsNormalization:
    def test_raw_dvars_scales_with_session_brightness(self, bold_and_mask):
        """The defect: two identical sessions differing only in gain disagree."""
        b1, m1 = bold_and_mask(scale=1.0)
        b2, m2 = bold_and_mask(scale=10.0)
        raw1 = calculate_dvars(b1, m1).mean()
        raw2 = calculate_dvars(b2, m2).mean()
        assert raw2 == pytest.approx(10.0 * raw1, rel=1e-6)

    def test_standardized_dvars_is_invariant_to_brightness(self, bold_and_mask):
        b1, m1 = bold_and_mask(scale=1.0)
        b2, m2 = bold_and_mask(scale=10.0)
        std1 = calculate_dvars(b1, m1, normalize=True).mean()
        std2 = calculate_dvars(b2, m2, normalize=True).mean()
        assert std2 == pytest.approx(std1, rel=1e-6)

    def test_default_stays_raw_for_existing_callers(self, bold_and_mask):
        b, m = bold_and_mask()
        assert np.allclose(calculate_dvars(b, m),
                           calculate_dvars(b, m, normalize=False))

    def test_standardized_is_scaled_by_the_temporal_difference_sd(self, bold_and_mask):
        b, m = bold_and_mask()
        raw = calculate_dvars(b, m)
        std = calculate_dvars(b, m, normalize=True)
        data = nib.load(b).get_fdata().reshape(-1, 20)
        scale = np.median(np.std(np.diff(data, axis=1), axis=1))
        assert np.allclose(std, raw / scale)

    def test_standardized_is_near_one_for_a_typical_volume(self, bold_and_mask):
        """The point of standardizing: ~1.0 means 'an ordinary volume'."""
        b, m = bold_and_mask()
        assert calculate_dvars(b, m, normalize=True).mean() == pytest.approx(1.0, abs=0.25)

    def test_demeaned_data_still_standardizes(self, tmp_path):
        """Preprocessed BOLD is demeaned (this cohort: in-mask mean -0.22), which
        is exactly where a percent-of-mean normalization breaks down."""
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 5.0, (5, 5, 3, 15))     # mean ~ 0
        aff = np.eye(4)
        bold, mask = tmp_path / 'd.nii.gz', tmp_path / 'dm.nii.gz'
        nib.save(nib.Nifti1Image(data, aff), bold)
        nib.save(nib.Nifti1Image(np.ones((5, 5, 3), dtype=np.uint8), aff), mask)
        out = calculate_dvars(bold, mask, normalize=True)
        assert np.all(np.isfinite(out))
        assert out.mean() == pytest.approx(1.0, abs=0.3)

    def test_empty_mask_does_not_produce_inf(self, tmp_path):
        aff = np.eye(4)
        bold = tmp_path / 'b.nii.gz'
        mask = tmp_path / 'm.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 2, 5)), aff), bold)
        nib.save(nib.Nifti1Image(np.ones((4, 4, 2), dtype=np.uint8), aff), mask)
        assert np.all(np.isfinite(calculate_dvars(bold, mask, normalize=True)))


class TestFdThreshold:
    def test_threshold_follows_the_voxel(self, bold_and_mask):
        """0.05 against a 4.0 header-unit voxel was 1.25% of a voxel."""
        b, _ = bold_and_mask(zooms=(4.0, 4.0, 8.0))
        assert fd_threshold_from_voxel_size(b) == pytest.approx(0.17 * 4.0)

    def test_scaling_the_header_scales_the_threshold(self, bold_and_mask):
        """The whole point: 10x-scaled headers get a 10x-scaled cut, for free."""
        small, _ = bold_and_mask(zooms=(0.4, 0.4, 0.8))
        big, _ = bold_and_mask(zooms=(4.0, 4.0, 8.0))
        assert (fd_threshold_from_voxel_size(big)
                == pytest.approx(10.0 * fd_threshold_from_voxel_size(small)))

    def test_ignores_the_coarse_through_plane_dimension(self, bold_and_mask):
        """FD is dominated by in-plane motion; slice thickness must not inflate it."""
        thin, _ = bold_and_mask(zooms=(4.0, 4.0, 1.0))
        thick, _ = bold_and_mask(zooms=(4.0, 4.0, 20.0))
        assert (fd_threshold_from_voxel_size(thin)
                == pytest.approx(fd_threshold_from_voxel_size(thick)))

    def test_human_geometry_lands_near_the_power_default(self, bold_and_mask):
        """3mm human voxels should give ~0.5mm, the value the literature uses."""
        b, _ = bold_and_mask(zooms=(3.0, 3.0, 3.0))
        assert fd_threshold_from_voxel_size(b) == pytest.approx(0.51, abs=0.02)

    def test_fraction_is_overridable(self, bold_and_mask):
        b, _ = bold_and_mask(zooms=(4.0, 4.0, 8.0))
        assert fd_threshold_from_voxel_size(b, fraction=0.5) == pytest.approx(2.0)

    def test_default_fraction_is_the_documented_one(self):
        assert FD_VOXEL_FRACTION == 0.17
