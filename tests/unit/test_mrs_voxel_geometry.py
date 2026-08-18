#!/usr/bin/env python3
"""
Unit tests for placing the SVS voxel on the anatomical image.

The mapping from Bruker read/phase/slice coordinates to the array axes
brukerapi produces is the part that is easy to get subtly wrong -- a
transposed or sign-flipped axis still yields a plausible-looking mask, just in
the wrong place. These tests pin the round trip and the tissue-fraction
arithmetic; the real-data check is the QC voxel-placement overlay.
"""

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.utils.mrs.bruker_mrs import BrukerSVS
from neurofaune.preprocess.utils.mrs.voxel_geometry import (
    AnatGeometry,
    index_to_magnet,
    magnet_to_index,
    make_voxel_mask,
    tissue_fractions,
    voxel_corners,
    write_tissue_fraction_json,
)

MATRIX = 64
N_SLICES = 20
FOV = 32.0
SLICE_DISTANCE = 0.8


def make_geometry(transposition=1, centre_offsets=(0.0, 0.0, 0.0)):
    """Geometry matching the CPZ T2w: axial, read along +y, phase along +x."""
    grad_orient = np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
    return AnatGeometry(
        grad_orient=grad_orient,
        centre=grad_orient.T @ np.asarray(centre_offsets, dtype=float),
        fov=np.array([FOV, FOV]),
        matrix=np.array([MATRIX, MATRIX]),
        n_slices=N_SLICES,
        slice_distance=SLICE_DISTANCE,
        transposition=transposition,
    )


def make_svs(size=(7.5, 2.0, 2.0), position=(0.0, 0.0, 0.0), orientation=None):
    return BrukerSVS(
        metab=np.zeros((4, 1, 1), dtype=complex),
        water_ref=None,
        dwelltime=3e-4,
        spectrometer_frequency=300.32,
        echo_time=0.02,
        repetition_time=2.0,
        nucleus='1H',
        voxel_size=np.asarray(size, dtype=float),
        voxel_position=np.asarray(position, dtype=float),
        voxel_orientation=np.eye(3) if orientation is None else np.asarray(orientation),
        source='rawdata',
    )


@pytest.fixture
def anat_image(tmp_path):
    """An anatomical NIfTI whose grid matches make_geometry()."""
    geometry = make_geometry()
    data = np.zeros(geometry.shape, dtype=np.float32)
    affine = np.diag([1.25, 1.25, 8.0, 1.0])
    path = tmp_path / 'T2w.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


class TestAxisMapping:
    def test_round_trip(self):
        geometry = make_geometry()
        points = np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 1.5], [-7.0, 5.0, -4.0]])
        recovered = index_to_magnet(geometry, magnet_to_index(geometry, points))
        assert recovered == pytest.approx(points)

    def test_package_centre_maps_to_array_centre(self):
        geometry = make_geometry()
        index = magnet_to_index(geometry, np.zeros((1, 3)))[0]
        expected = np.array(geometry.shape, dtype=float) / 2.0 - 0.5
        assert index == pytest.approx(expected)

    def test_transposition_swaps_the_in_plane_axes(self):
        # Move along magnet x (the phase direction). With transposition=1 that
        # is array axis 0; untransposed it is array axis 1.
        point = np.array([[4.0, 0.0, 0.0]])
        centre = np.array([MATRIX, MATRIX, N_SLICES]) / 2.0 - 0.5

        transposed = magnet_to_index(make_geometry(transposition=1), point)[0] - centre
        plain = magnet_to_index(make_geometry(transposition=0), point)[0] - centre

        assert transposed[0] != pytest.approx(0.0) and transposed[1] == pytest.approx(0.0)
        assert plain[1] != pytest.approx(0.0) and plain[0] == pytest.approx(0.0)

    def test_read_axis_is_reversed_when_transposed(self):
        # The verified mapping flips the read direction; without that the voxel
        # lands mirrored about the package centre.
        point = np.array([[0.0, 4.0, 0.0]])  # +4 mm along read
        centre = np.array([MATRIX, MATRIX, N_SLICES]) / 2.0 - 0.5
        offset = magnet_to_index(make_geometry(transposition=1), point)[0] - centre
        assert offset[1] < 0

    def test_slice_axis_is_reversed_when_transposed(self):
        # Regression: this sign was wrong, and because it only shows on
        # sessions whose voxel is prescribed away from isocentre it survived a
        # visual check on one session whose slice offset was 0.26 mm. It
        # displaced 36 of 52 cuprizone sessions by more than two slices.
        point = np.array([[0.0, 0.0, 4.0]])  # +4 mm along slice
        centre = np.array([MATRIX, MATRIX, N_SLICES]) / 2.0 - 0.5
        offset = magnet_to_index(make_geometry(transposition=1), point)[0] - centre
        assert offset[2] < 0

    def test_a_sign_error_doubles_the_displacement(self):
        # Why a near-isocentre session cannot validate the convention: the
        # error is proportional to the offset, so it vanishes at zero.
        centre = np.array([MATRIX, MATRIX, N_SLICES]) / 2.0 - 0.5
        geometry = make_geometry(transposition=1)
        near = magnet_to_index(geometry, np.array([[0.0, 0.0, 0.2]]))[0] - centre
        far = magnet_to_index(geometry, np.array([[0.0, 0.0, 4.0]]))[0] - centre
        assert abs(near[2]) < 0.5           # invisible near isocentre
        assert abs(far[2]) > 4.0            # obvious away from it

    def test_unsupported_transposition_is_rejected(self):
        with pytest.raises(ValueError, match='RECO_transposition'):
            _ = make_geometry(transposition=3).shape

    def test_slice_offset_shifts_the_package(self):
        # A 1.6 mm slice offset is two slices at 0.8 mm spacing. The direction
        # follows the verified slice sign: shifting the package one way moves
        # magnet isocentre the other way within the array.
        geometry = make_geometry(centre_offsets=(0.0, 0.0, 1.6))
        index = magnet_to_index(geometry, np.zeros((1, 3)))[0]
        assert index[2] == pytest.approx(N_SLICES / 2.0 - 0.5 + 2.0)


class TestSliceOffsetSign:
    """PVM_SPackArrSliceOffset is negated; the in-plane offsets are not."""

    def _geometry(self, read=0.0, phase=0.0, slice_=0.0, tmp_path=None):
        from neurofaune.preprocess.utils.mrs.voxel_geometry import read_anat_geometry
        scan = tmp_path / '5'
        (scan / 'pdata' / '1').mkdir(parents=True)
        (scan / 'method').write_text(
            '##$PVM_SPackArrGradOrient=( 1, 3, 3 )\n0 1 0 1 0 0 0 0 1\n'
            f'##$PVM_SPackArrReadOffset=( 1 )\n{read}\n'
            f'##$PVM_SPackArrPhase1Offset=( 1 )\n{phase}\n'
            f'##$PVM_SPackArrSliceOffset=( 1 )\n{slice_}\n'
            '##$PVM_Fov=( 2 )\n32 32\n##$PVM_Matrix=( 2 )\n64 64\n'
            '##$PVM_SPackArrNSlices=( 1 )\n20\n'
            '##$PVM_SPackArrSliceDistance=( 1 )\n0.8\n##END=\n')
        (scan / 'pdata' / '1' / 'reco').write_text(
            '##$RECO_transposition=( 20 )\n@20*(1)\n##END=\n')
        return read_anat_geometry(scan)

    def test_slice_offset_is_negated(self, tmp_path):
        # Regression: this sign error displaced every session with a non-zero
        # slice offset -- 15 of 50 here, by up to 2.5 mm -- while leaving the
        # other 35 untouched, so it hid behind the majority that were fine.
        g = self._geometry(slice_=2.0, tmp_path=tmp_path)
        assert g.centre[2] == pytest.approx(-2.0)

    def test_in_plane_offsets_are_not_negated(self, tmp_path):
        # Flipping these too breaks the sessions that carry a read offset.
        g = self._geometry(read=-1.0, tmp_path=tmp_path)
        # grad_orient row 0 (read) is +y, so a -1 mm read offset lands at y=-1.
        assert g.centre[1] == pytest.approx(-1.0)


class TestVoxelCorners:
    def test_eight_corners_span_the_voxel(self):
        svs = make_svs(size=(7.5, 2.0, 2.0), position=(1.0, 2.0, 3.0))
        corners = voxel_corners(svs)
        assert corners.shape == (8, 3)
        assert corners.max(axis=0) - corners.min(axis=0) == pytest.approx([7.5, 2.0, 2.0])
        assert corners.mean(axis=0) == pytest.approx([1.0, 2.0, 3.0])

    def test_rotation_is_applied(self):
        angle = np.deg2rad(90.0)
        rotation = np.array([[np.cos(angle), np.sin(angle), 0.0],
                             [-np.sin(angle), np.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
        svs = make_svs(size=(8.0, 2.0, 2.0), orientation=rotation)
        extent = voxel_corners(svs).max(axis=0) - voxel_corners(svs).min(axis=0)
        # The 8 mm axis now lies along magnet y.
        assert extent == pytest.approx([2.0, 8.0, 2.0])


class TestVoxelMask:
    def test_volume_matches_the_voxel(self, anat_image):
        svs = make_svs(size=(7.5, 2.0, 2.0))
        mask, _ = make_voxel_mask(svs, None, anat_image, supersample=5, geometry=make_geometry())

        voxel_volume = np.prod(make_geometry().spacing)
        assert mask.sum() * voxel_volume == pytest.approx(7.5 * 2.0 * 2.0, rel=0.05)

    def test_mask_is_centred_on_the_voxel(self, anat_image):
        svs = make_svs(size=(4.0, 4.0, 2.0), position=(0.0, 0.0, 0.0))
        mask, _ = make_voxel_mask(svs, None, anat_image, supersample=3, geometry=make_geometry())

        weighted = np.array(np.nonzero(mask)).T
        centre = weighted.mean(axis=0)
        expected = np.array(make_geometry().shape, dtype=float) / 2.0 - 0.5
        assert centre == pytest.approx(expected, abs=0.6)

    def test_fractional_edges(self, anat_image):
        # Anti-aliasing: partially covered voxels must take intermediate values,
        # otherwise the coarse 0.8 mm slices quantise the tissue fractions.
        svs = make_svs(size=(7.5, 2.0, 1.2))
        mask, _ = make_voxel_mask(svs, None, anat_image, supersample=4, geometry=make_geometry())
        assert np.any((mask > 0) & (mask < 1))

    def test_writes_a_mask_file(self, anat_image, tmp_path):
        svs = make_svs()
        out = tmp_path / 'mask.nii.gz'
        _, path = make_voxel_mask(svs, None, anat_image, output_file=out, geometry=make_geometry())
        assert path == out
        assert nib.load(out).shape == make_geometry().shape

    def test_shape_mismatch_is_rejected(self, anat_image, tmp_path):
        wrong = tmp_path / 'wrong.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), wrong)
        with pytest.raises(ValueError, match='not the same acquisition'):
            make_voxel_mask(make_svs(), None, wrong, geometry=make_geometry())

    def test_voxel_outside_the_slab(self, anat_image):
        # Far outside the field of view: an empty mask, not a crash.
        svs = make_svs(position=(0.0, 0.0, 500.0))
        mask, _ = make_voxel_mask(svs, None, anat_image, geometry=make_geometry())
        assert mask.sum() == 0


class TestTissueFractions:
    @pytest.fixture
    def tissue_maps(self, tmp_path):
        shape = make_geometry().shape
        maps = {}
        for label, value in (('GM', 0.6), ('WM', 0.3), ('CSF', 0.1)):
            path = tmp_path / f'{label}.nii.gz'
            nib.save(nib.Nifti1Image(np.full(shape, value, np.float32), np.eye(4)), path)
            maps[label] = path
        return maps

    def test_fractions_normalise_to_one(self, tissue_maps):
        mask = np.zeros(make_geometry().shape)
        mask[20:30, 20:30, 8:12] = 1.0
        fractions = tissue_fractions(mask, tissue_maps['GM'],
                                     tissue_maps['WM'], tissue_maps['CSF'])
        assert fractions['GM'] == pytest.approx(0.6)
        assert fractions['WM'] == pytest.approx(0.3)
        assert fractions['CSF'] == pytest.approx(0.1)
        assert sum(fractions[k] for k in ('GM', 'WM', 'CSF')) == pytest.approx(1.0)

    def test_coverage_reports_unsegmented_voxel(self, tissue_maps, tmp_path):
        # Half the voxel falls outside the brain mask: coverage must say so,
        # because fractions from a half-covered voxel are extrapolated.
        shape = make_geometry().shape
        for label in ('GM', 'WM', 'CSF'):
            data = nib.load(tissue_maps[label]).get_fdata()
            data[:, :, :10] = 0.0
            nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), tissue_maps[label])

        mask = np.zeros(shape)
        mask[20:30, 20:30, 8:12] = 1.0
        fractions = tissue_fractions(mask, tissue_maps['GM'],
                                     tissue_maps['WM'], tissue_maps['CSF'])
        assert fractions['voxel_volume_ratio'] == pytest.approx(0.5)

    def test_empty_mask_is_rejected(self, tissue_maps):
        with pytest.raises(ValueError, match='does not intersect'):
            tissue_fractions(np.zeros(make_geometry().shape), tissue_maps['GM'],
                             tissue_maps['WM'], tissue_maps['CSF'])

    def test_json_is_written_in_fsl_mrs_layout(self, tmp_path):
        out = write_tissue_fraction_json(
            {'GM': 0.5, 'WM': 0.4, 'CSF': 0.1, 'measured': True, 'mask': None},
            tmp_path / 'frac.json',
        )
        import json

        payload = json.loads(out.read_text())
        assert set(payload) == {'GM', 'WM', 'CSF'}
        assert payload['GM'] == pytest.approx(0.5)
