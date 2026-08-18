#!/usr/bin/env python3
"""
Unit tests for the geometry-validation guards and the placement check.

These exist because three separate sign errors in the voxel localisation each
survived visual inspection, all for the same reason: the error is proportional
to an offset that is zero on most sessions. The guards refuse to extrapolate
silently to unvalidated acquisition geometry, and the placement check turns a
misplaced voxel into a number rather than a plausible-looking picture.
"""

import numpy as np
import pytest

from neurofaune.preprocess.utils.mrs.geometry_support import (
    check_geometry_support,
    check_reco_transposition,
)
from neurofaune.preprocess.utils.mrs.voxel_geometry import (
    labels_for_structure,
    target_overlap,
)

VALIDATED_PARAMS = {
    'PVM_SPackArrSliceOrient': ['axial'],
    'PVM_SpatDimEnum': '2D',
    'PVM_NSPacks': 1,
    'PVM_SPackArrPhase1Offset': np.array([0.0]),
}


class TestGeometryGuards:
    def test_validated_acquisition_passes_silently(self):
        assert check_geometry_support(VALIDATED_PARAMS) == []
        assert check_reco_transposition(1) == []

    def test_non_axial_orientation_is_flagged(self):
        params = {**VALIDATED_PARAMS, 'PVM_SPackArrSliceOrient': ['coronal']}
        problems = check_geometry_support(params)
        assert len(problems) == 1
        assert 'coronal' in problems[0]

    def test_three_d_acquisition_is_flagged(self):
        params = {**VALIDATED_PARAMS, 'PVM_SpatDimEnum': '3D'}
        assert any('3D' in p for p in check_geometry_support(params))

    def test_multiple_slice_packages_are_flagged(self):
        params = {**VALIDATED_PARAMS, 'PVM_NSPacks': 2}
        assert any('slice packages' in p for p in check_geometry_support(params))

    def test_phase_offset_is_flagged_as_untested(self):
        # 52 of 53 validation sessions had no phase offset, and the analogous
        # slice-offset sign turned out to be wrong -- so a non-zero one is
        # worth surfacing rather than trusting.
        params = {**VALIDATED_PARAMS, 'PVM_SPackArrPhase1Offset': np.array([1.5])}
        problems = check_geometry_support(params)
        assert any('phase offset' in p for p in problems)

    def test_zero_phase_offset_is_not_flagged(self):
        assert check_geometry_support(VALIDATED_PARAMS) == []

    def test_unvalidated_transposition_is_flagged(self):
        problems = check_reco_transposition(0)
        assert len(problems) == 1
        assert 'RECO_transposition=0' in problems[0]

    def test_scan_name_appears_in_the_message(self):
        params = {**VALIDATED_PARAMS, 'PVM_SPackArrSliceOrient': ['sagittal']}
        assert 'scan-5' in check_geometry_support(params, scan='scan-5')[0]

    def test_missing_parameters_do_not_raise(self):
        # An acquisition that simply lacks these fields should not crash the
        # check; absence is not evidence of a problem.
        assert check_geometry_support({}) == []


class TestLabelLookup:
    def _csv(self, tmp_path):
        path = tmp_path / 'labels.csv'
        path.write_text(
            'Original Atlas,Labels,Hemisphere,Matter,Territories,System,Region of interest\n'
            'Waxholm,71,L,Grey Matter,Cortex,Hippocampus Fomation,Cornu.Ammonis.1.L\n'
            'Waxholm,72,R,Grey Matter,Cortex,Hippocampus Fomation,Cornu.Ammonis.1.R\n'
            'Tohoku,11,L,Grey Matter,Cortex,Insular System,Agranular.Insular.Cortex.L\n')
        return path

    def test_matches_across_columns(self, tmp_path):
        # The SIGMA table puts "Hippocampus Fomation" (sic) in System, not
        # Territories -- searching one column silently returns nothing.
        assert sorted(labels_for_structure(self._csv(tmp_path), 'hippocamp')) == [71, 72]

    def test_match_is_case_insensitive(self, tmp_path):
        assert labels_for_structure(self._csv(tmp_path), 'HIPPOCAMP') == [71, 72]

    def test_matches_a_region_name(self, tmp_path):
        assert labels_for_structure(self._csv(tmp_path), 'insular') == [11]

    def test_no_match_returns_empty(self, tmp_path):
        assert labels_for_structure(self._csv(tmp_path), 'cerebellum') == []


class TestTargetOverlap:
    @pytest.fixture
    def parcellation(self, tmp_path):
        import nibabel as nib

        seg = np.zeros((20, 20, 10), dtype=np.int16)
        seg[5:15, 5:10, 3:7] = 71          # the target
        seg[5:15, 10:15, 3:7] = 99         # a neighbour
        path = tmp_path / 'dseg.nii.gz'
        nib.save(nib.Nifti1Image(seg, np.eye(4)), path)
        return path

    def test_voxel_fully_on_target(self, parcellation):
        mask = np.zeros((20, 20, 10)); mask[6:14, 6:9, 4:6] = 1.0
        result = target_overlap(mask, parcellation, [71])
        assert result['overlap'] == pytest.approx(1.0)
        assert 0.0 < result['target_captured'] < 1.0

    def test_voxel_on_the_wrong_structure(self, parcellation):
        # This is the failure a wrong geometry convention produces, and what
        # the QC threshold is there to catch.
        mask = np.zeros((20, 20, 10)); mask[6:14, 11:14, 4:6] = 1.0
        assert target_overlap(mask, parcellation, [71])['overlap'] == pytest.approx(0.0)

    def test_partial_overlap_is_proportional(self, parcellation):
        mask = np.zeros((20, 20, 10)); mask[6:14, 8:12, 4:6] = 1.0   # half on target
        assert target_overlap(mask, parcellation, [71])['overlap'] == pytest.approx(0.5)

    def test_fractional_mask_is_weighted(self, parcellation):
        mask = np.zeros((20, 20, 10))
        mask[6:14, 6:9, 4:6] = 0.5      # on target, half weight
        mask[6:14, 11:14, 4:6] = 1.0    # off target, full weight
        assert target_overlap(mask, parcellation, [71])['overlap'] == pytest.approx(1 / 3)

    def test_empty_mask_is_zero_not_an_error(self, parcellation):
        result = target_overlap(np.zeros((20, 20, 10)), parcellation, [71])
        assert result == {'overlap': 0.0, 'target_captured': 0.0}

    def test_absent_label_is_zero(self, parcellation):
        mask = np.zeros((20, 20, 10)); mask[6:14, 6:9, 4:6] = 1.0
        assert target_overlap(mask, parcellation, [12345])['overlap'] == 0.0

    def test_shape_mismatch_is_rejected(self, parcellation):
        with pytest.raises(ValueError, match='same space'):
            target_overlap(np.zeros((8, 8, 8)), parcellation, [71])
