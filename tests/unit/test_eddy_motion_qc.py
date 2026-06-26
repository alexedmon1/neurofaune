"""Unit tests for DWI eddy motion-QC math (framewise displacement).

Pins the contract of ``_plot_motion_parameters`` so the three historical bugs
cannot silently return:
  1. FSL eddy column order — translations in cols 0-2, rotations in cols 3-5
     (a swap turns small rotations into huge bogus degrees, e.g. 7 rad -> 426 deg).
  2. voxel_scale correction — rodent images carry a 10x-inflated header, so eddy
     motion estimates are inflated by that factor and must be divided back out.
  3. rotation->mm via the head radius (arc length), consistent with the fMRI path.
"""
import numpy as np
import pytest

from neurofaune.preprocess.qc.dwi.eddy_qc import _plot_motion_parameters

RADIUS = 50.0
VOXEL_SCALE = 10.0


def _write_params(path, translations, rotations):
    """Write a minimal FSL-eddy-style .eddy_parameters file (16 cols):
    translations (mm, header-scaled) in cols 0-2, rotations (rad) in cols 3-5."""
    n = len(translations)
    arr = np.zeros((n, 16))
    arr[:, 0:3] = translations
    arr[:, 3:6] = rotations
    np.savetxt(path, arr)
    return path


def test_fd_uses_radius_and_voxel_scale(tmp_path):
    # vol0 = origin; vol1 = 10 header-mm translation in x AND 0.1 rad rotation in pitch.
    trans = np.array([[0, 0, 0], [10.0, 0, 0], [10.0, 0, 0]])
    rot = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])
    p = _write_params(tmp_path / "eddy_corrected.eddy_parameters", trans, rot)

    metrics, _ = _plot_motion_parameters(p, "sub-x", "ses-1", tmp_path,
                                         radius=RADIUS, voxel_scale=VOXEL_SCALE)

    # FD[1] = (|Δtrans| + radius*|Δrot|) / voxel_scale = (10 + 50*0.1)/10 = 1.5 mm
    assert metrics["max_fd"] == pytest.approx(1.5, rel=1e-6)
    assert metrics["mean_fd"] == pytest.approx(0.5, rel=1e-6)  # (1.5 + 0 + 0)/3


def test_translations_reported_in_real_mm(tmp_path):
    # 20 header-mm translation -> 2.0 real mm after /voxel_scale; rotations zero.
    trans = np.array([[0, 0, 0], [20.0, 0, 0]])
    rot = np.zeros((2, 3))
    p = _write_params(tmp_path / "e.eddy_parameters", trans, rot)

    metrics, _ = _plot_motion_parameters(p, "s", "ses-1", tmp_path,
                                         radius=RADIUS, voxel_scale=VOXEL_SCALE)

    assert metrics["max_translation"] == pytest.approx(2.0, rel=1e-6)
    assert metrics["max_rotation"] == pytest.approx(0.0, abs=1e-9)


def test_column_order_not_swapped(tmp_path):
    """A pure-rotation run must show rotation (deg) and ~zero translation — the
    swap bug would mislabel the 0.2 rad rotation as a 0.02 mm translation and
    report a ~11 deg 'rotation' from the (zero) translation column."""
    trans = np.zeros((2, 3))
    rot = np.array([[0, 0, 0], [0.2, 0, 0]])  # 0.2 rad = 11.46 deg
    p = _write_params(tmp_path / "e.eddy_parameters", trans, rot)

    metrics, _ = _plot_motion_parameters(p, "s", "ses-1", tmp_path,
                                         radius=RADIUS, voxel_scale=VOXEL_SCALE)

    assert metrics["max_rotation"] == pytest.approx(np.degrees(0.2), rel=1e-6)
    assert metrics["max_translation"] == pytest.approx(0.0, abs=1e-9)


def test_voxel_scale_one_is_identity(tmp_path):
    # With no scaling, a 3 mm translation stays 3 mm.
    trans = np.array([[0, 0, 0], [3.0, 0, 0]])
    rot = np.zeros((2, 3))
    p = _write_params(tmp_path / "e.eddy_parameters", trans, rot)

    metrics, _ = _plot_motion_parameters(p, "s", "ses-1", tmp_path,
                                         radius=RADIUS, voxel_scale=1.0)

    assert metrics["max_translation"] == pytest.approx(3.0, rel=1e-6)
    assert metrics["max_fd"] == pytest.approx(3.0, rel=1e-6)
