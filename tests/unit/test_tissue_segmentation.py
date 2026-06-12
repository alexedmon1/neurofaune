"""
Unit tests for decoupled tissue segmentation (GM/WM/CSF).

Hermetic: the Atropos subprocess (ANTs) is integration-tier and not exercised
here. We test the pure logic that turns Atropos posteriors into labeled tissue
maps — the intensity-based class labeling (which class is WM/GM/CSF) and the
hard-segmentation finalization — since that is where a labeling bug would land.
"""
import numpy as np
import nibabel as nib
import pytest

from neurofaune.preprocess.workflows.anat_preprocess import (
    _order_tissue_by_intensity,
    _finalize_tissue_maps,
)


class TestOrderTissueByIntensity:
    def test_orders_wm_gm_csf_by_ascending_intensity(self):
        # Three disjoint regions with WM darkest, GM mid, CSF brightest on T2w.
        shape = (10, 10, 9)
        img = np.zeros(shape)
        img[:, :, 0:3] = 30.0    # WM-like (dark)
        img[:, :, 3:6] = 90.0    # GM-like
        img[:, :, 6:9] = 200.0   # CSF-like (bright)
        mask = np.ones(shape)

        # Posteriors handed in deliberately OUT of WM/GM/CSF order (CSF, WM, GM)
        post_csf = np.zeros(shape); post_csf[:, :, 6:9] = 1.0
        post_wm = np.zeros(shape); post_wm[:, :, 0:3] = 1.0
        post_gm = np.zeros(shape); post_gm[:, :, 3:6] = 1.0

        ordered, mean_int, order = _order_tissue_by_intensity(
            [post_csf, post_wm, post_gm], img, mask
        )
        # ordered = (WM, GM, CSF) regardless of input order
        np.testing.assert_allclose(ordered[0], post_wm)
        np.testing.assert_allclose(ordered[1], post_gm)
        np.testing.assert_allclose(ordered[2], post_csf)
        # ascending mean intensity
        assert mean_int[order[0]] < mean_int[order[1]] < mean_int[order[2]]
        assert mean_int[order[0]] == pytest.approx(30.0)
        assert mean_int[order[2]] == pytest.approx(200.0)

    def test_mask_restricts_intensity_estimate(self):
        # A bright voxel outside the mask must not pull a class's mean up.
        shape = (6, 6, 6)
        img = np.full(shape, 50.0)
        img[0, 0, 0] = 10000.0   # bright, but masked out
        mask = np.ones(shape); mask[0, 0, 0] = 0.0
        post = np.ones(shape)
        _, mean_int, _ = _order_tissue_by_intensity([post], img, mask)
        assert mean_int[0] == pytest.approx(50.0)

    def test_empty_posterior_gets_zero_intensity(self):
        shape = (4, 4, 4)
        img = np.full(shape, 100.0)
        mask = np.ones(shape)
        empty = np.zeros(shape)
        _, mean_int, _ = _order_tissue_by_intensity([empty], img, mask)
        assert mean_int[0] == 0.0


class TestFinalizeTissueMaps:
    def _affine_img(self, shape):
        # Match the production brain mask: float32 header (slope 1.0), so a
        # label/prob map saved with this header roundtrips without quantization.
        return nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))

    def test_writes_full_output_set_and_argmax_labels(self, tmp_path):
        shape = (8, 8, 6)
        mask_data = np.ones(shape)
        mask_img = self._affine_img(shape)
        # WM wins in z0:2, GM in z2:4, CSF in z4:6
        wm = np.zeros(shape); wm[:, :, 0:2] = 0.9
        gm = np.zeros(shape); gm[:, :, 2:4] = 0.9
        csf = np.zeros(shape); csf[:, :, 4:6] = 0.9

        out = _finalize_tissue_maps(
            wm, gm, csf, mask_img, mask_data, tmp_path, "sub-T", "ses-1",
            tissue_confidence_threshold=0.35,
        )
        for key in ("segmentation", "gm_prob", "wm_prob", "csf_prob"):
            assert out[key].exists(), f"missing output {key}"

        seg = nib.load(str(out["segmentation"])).get_fdata()
        # Label convention 1=WM, 2=GM, 3=CSF
        assert set(np.unique(seg)) <= {0, 1, 2, 3}
        assert (seg[:, :, 0:2] == 1).all()
        assert (seg[:, :, 2:4] == 2).all()
        assert (seg[:, :, 4:6] == 3).all()

    def test_low_confidence_voxels_unlabeled(self, tmp_path):
        shape = (5, 5, 5)
        mask_data = np.ones(shape)
        mask_img = self._affine_img(shape)
        # All probabilities below threshold -> dseg should be 0 everywhere
        wm = np.full(shape, 0.1)
        gm = np.full(shape, 0.05)
        csf = np.full(shape, 0.02)
        out = _finalize_tissue_maps(
            wm, gm, csf, mask_img, mask_data, tmp_path, "sub-T", "ses-1",
            tissue_confidence_threshold=0.35,
        )
        seg = nib.load(str(out["segmentation"])).get_fdata()
        assert (seg == 0).all()

    def test_segmentation_confined_to_mask(self, tmp_path):
        shape = (6, 6, 6)
        mask_data = np.zeros(shape); mask_data[1:5, 1:5, 1:5] = 1.0
        mask_img = self._affine_img(shape)
        wm = np.full(shape, 0.9)  # high prob everywhere
        gm = np.zeros(shape)
        csf = np.zeros(shape)
        out = _finalize_tissue_maps(
            wm, gm, csf, mask_img, mask_data, tmp_path, "sub-T", "ses-1",
        )
        seg = nib.load(str(out["segmentation"])).get_fdata()
        assert (seg[mask_data == 0] == 0).all()
        assert (seg[mask_data == 1] == 1).all()
