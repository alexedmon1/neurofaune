#!/usr/bin/env python3
"""
Unit tests for deriving NIfTI affines from Bruker's own image geometry.

This is the systematic replacement for reconstructing the index-to-world
mapping from ``PVM_`` parameters by hand. ``visu_pars`` carries the
DICOM-equivalent fields, so the affine comes out with its signs and axis order
included, and locating an SVS voxel becomes affine composition -- the way it
works in human MRS.
"""

import numpy as np
import pytest

from neurofaune.preprocess.utils.mrs.bruker_affine import (
    SUBJECT_POSITION_TRANSFORMS,
    affine_from_visu,
    index_to_world,
    pvm_to_world,
    read_visu_geometry,
    world_to_index,
)


def write_visu(scan_dir, orientation=None, first_corner=(-16.0, -15.0, -16.0),
               n_frames=41, size=(256, 256), extent=(32.0, 32.0),
               slice_step=0.8, subject_position='Head_Supine', spacing_jitter=0.0):
    """Write a visu_pars describing an evenly spaced axial slice package."""
    rotation = np.eye(3) if orientation is None else np.asarray(orientation)
    positions = []
    for frame in range(n_frames):
        step = rotation[2] * (slice_step * frame)
        if spacing_jitter and frame == n_frames - 1:
            step = step + rotation[2] * spacing_jitter
        positions.append(np.asarray(first_corner) + step)
    proc = scan_dir / 'pdata' / '1'
    proc.mkdir(parents=True, exist_ok=True)
    lines = [
        '##TITLE=synthetic',
        f'##$VisuCoreSize=( 2 )\n{size[0]} {size[1]}',
        f'##$VisuCoreExtent=( 2 )\n{extent[0]} {extent[1]}',
        '##$VisuCoreFrameThickness=( 1 )\n0.8',
        f'##$VisuSubjectPosition={subject_position}',
        f'##$VisuCoreOrientation=( {n_frames}, 9 )',
        ' '.join(' '.join(f'{v:.10g}' for v in rotation.ravel()) for _ in range(n_frames)),
        f'##$VisuCorePosition=( {n_frames}, 3 )',
        ' '.join(' '.join(f'{v:.10g}' for v in p) for p in positions),
        '##END=',
    ]
    (proc / 'visu_pars').write_text('\n'.join(lines) + '\n')
    return scan_dir


def write_method(scan_dir, grad_orient=None, read=0.0, phase=0.0, slice_=0.0):
    grad = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float) \
        if grad_orient is None else np.asarray(grad_orient, dtype=float)
    (scan_dir / 'method').write_text(
        '##$PVM_SPackArrGradOrient=( 1, 3, 3 )\n'
        + ' '.join(f'{v:.10g}' for v in grad.ravel()) + '\n'
        f'##$PVM_SPackArrReadOffset=( 1 )\n{read}\n'
        f'##$PVM_SPackArrPhase1Offset=( 1 )\n{phase}\n'
        f'##$PVM_SPackArrSliceOffset=( 1 )\n{slice_}\n'
        '##END=\n')
    return scan_dir


@pytest.fixture
def scan(tmp_path):
    d = tmp_path / '5'
    d.mkdir()
    write_visu(d)
    write_method(d)
    return d


class TestVisuGeometry:
    def test_reads_the_geometry_fields(self, scan):
        geometry = read_visu_geometry(scan)
        assert geometry['orientation'].shape == (41, 3, 3)
        assert geometry['position'].shape == (41, 3)
        assert geometry['extent'][:2] == pytest.approx([32.0, 32.0])

    def test_missing_visu_pars_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='visu_pars'):
            read_visu_geometry(tmp_path / 'absent')

    def test_missing_fields_are_an_error(self, tmp_path):
        d = tmp_path / '5' / 'pdata' / '1'
        d.mkdir(parents=True)
        (d / 'visu_pars').write_text('##$VisuCoreSize=( 2 )\n256 256\n##END=\n')
        # No silent fallback: a partial visu_pars must not yield a wrong affine.
        with pytest.raises(KeyError, match='cannot be derived'):
            read_visu_geometry(tmp_path / '5')


class TestAffine:
    def test_spacing_comes_from_extent_and_size(self, scan):
        affine = affine_from_visu(scan)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        assert spacing == pytest.approx([0.125, 0.125, 0.8])

    def test_affine_is_on_voxel_centres(self, scan):
        # VisuCorePosition is the first voxel's corner, so the affine must add
        # half a voxel. Otherwise the whole volume sits half a voxel out.
        affine = affine_from_visu(scan)
        assert affine[:3, 3] == pytest.approx([-16 + 0.0625, -15 + 0.0625, -16.0])

    def test_volume_centre_reflects_the_read_offset(self, scan):
        # The validation session had a -1 mm read offset and its volume centre
        # lands on exactly (0, +1, 0): the sign relationship falls out of the
        # geometry rather than being chosen.
        affine = affine_from_visu(scan)
        centre = index_to_world(affine, np.array([[127.5, 127.5, 20.0]]))[0]
        assert centre == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)

    def test_round_trip(self, scan):
        affine = affine_from_visu(scan)
        index = np.array([[10.0, 20.0, 5.0], [200.0, 30.0, 39.0]])
        assert world_to_index(affine, index_to_world(affine, index)) == pytest.approx(index)

    def test_uneven_slice_spacing_is_rejected(self, tmp_path):
        d = tmp_path / '5'; d.mkdir()
        write_visu(d, spacing_jitter=0.4)
        with pytest.raises(ValueError, match='evenly spaced'):
            affine_from_visu(d)

    def test_varying_orientation_is_rejected(self, tmp_path):
        d = tmp_path / '5' / 'pdata' / '1'
        d.mkdir(parents=True)
        (d / 'visu_pars').write_text(
            '##$VisuCoreSize=( 2 )\n64 64\n##$VisuCoreExtent=( 2 )\n32 32\n'
            '##$VisuCoreOrientation=( 2, 9 )\n1 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 1\n'
            '##$VisuCorePosition=( 2, 3 )\n0 0 0 0 0 1\n##END=\n')
        with pytest.raises(ValueError, match='orientation varies'):
            affine_from_visu(tmp_path / '5')

    def test_single_frame_uses_the_thickness(self, tmp_path):
        d = tmp_path / '5'; d.mkdir()
        write_visu(d, n_frames=1)
        affine = affine_from_visu(d)
        assert np.linalg.norm(affine[:3, 2]) == pytest.approx(0.8)


class TestSubjectPositionTransform:
    def test_known_position_returns_the_calibrated_rotation(self, scan):
        rotation = pvm_to_world(scan)
        assert np.allclose(rotation[:3, :3], SUBJECT_POSITION_TRANSFORMS['Head_Supine'])
        assert np.linalg.det(rotation[:3, :3]) == pytest.approx(1.0)

    def test_unknown_position_refuses_rather_than_guesses(self, tmp_path):
        d = tmp_path / '5'; d.mkdir()
        write_visu(d, subject_position='Head_Prone')
        write_method(d)
        # Guessing this is what produced the earlier sign errors.
        with pytest.raises(KeyError, match='no calibrated'):
            pvm_to_world(d)

    def test_misaligned_frames_are_rejected(self, tmp_path):
        # A gradient orientation that cannot be carried onto the image axes by
        # the calibrated rotation means the frames are unrelated as assumed.
        d = tmp_path / '5'; d.mkdir()
        write_visu(d)
        skew = np.array([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
        write_method(d, grad_orient=skew)
        with pytest.raises(ValueError, match='does not align'):
            pvm_to_world(d)

    def test_offsets_do_not_affect_the_check(self, tmp_path):
        # The check is deliberately offset-free: reconstructing the package
        # centre from PVM_SPackArr*Offset would reintroduce the sign
        # conventions this replaces.
        d = tmp_path / '5'; d.mkdir()
        write_visu(d)
        write_method(d, slice_=2.5, read=-1.0)
        assert np.allclose(pvm_to_world(d)[:3, :3],
                           SUBJECT_POSITION_TRANSFORMS['Head_Supine'])
