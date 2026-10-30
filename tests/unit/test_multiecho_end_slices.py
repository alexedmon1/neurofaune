"""Multi-echo motion correction keeps the end slices of the slab.

applyxfm4D has nothing to sample outside the field of view, so a volume whose
through-slice shift points outward lost its outermost slice: on the cuprizone
cohort a median 48% of echo 1's signal in slices 0 and 20, which then held 74% of
the optimally combined series' temporal variance.
"""
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.workflows import func_preprocess as fp
from neurofaune.preprocess.workflows.func_preprocess import (
    _apply_transforms_padded,
    _largest_through_slice_shift,
    _shift_mats_to_padded_frame,
)

needs_fsl = pytest.mark.skipif(
    shutil.which("applyxfm4D") is None, reason="applyxfm4D not on the path"
)

ZOOMS = (4.0, 4.0, 8.0)
AFFINE = np.diag([*ZOOMS, 1.0])
SHAPE = (24, 24, 21)
REAL_RUN = subprocess.run
_x, _y = np.mgrid[:SHAPE[0], :SHAPE[1]]
DISC = ((_x - 11.5) ** 2 + (_y - 11.5) ** 2) < 81


def _translation(z_mm):
    mat = np.eye(4)
    mat[2, 3] = z_mm
    return mat


def _rotation_x(degrees, z_mm=0.0):
    """Rotation about x through the slab's corner, so its effect depends on z."""
    t = np.deg2rad(degrees)
    mat = np.eye(4)
    mat[1:3, 1:3] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    mat[2, 3] = z_mm
    return mat


def _write_mats(directory: Path, mats) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for i, mat in enumerate(mats):
        path = directory / f"MAT_{i:04d}"
        np.savetxt(path, mat, fmt="%.10f")
        files.append(path)
    return files


def _constant_slab(path: Path, n_vols) -> Path:
    """Constant signal in a disc on zero background.

    A wholly constant image cannot show the defect: applyxfm4D fills outside the
    field of view with a value taken from the image, which there equals the signal.
    """
    data = np.zeros((*SHAPE, n_vols), dtype=np.float32)
    data[DISC] = 1000.0
    nib.save(nib.Nifti1Image(data, AFFINE), path)
    return path


def _structured_slab(path: Path, n_vols, affine=AFFINE) -> Path:
    x, y, z = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
    volume = 1000 + 200 * np.sin(x / 3) + 150 * np.cos(y / 4) + 300 * np.sin(z / 2.5)
    data = np.repeat(volume[..., None], n_vols, axis=3).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _plain_applyxfm(echo: Path, mat_dir: Path, out: Path) -> np.ndarray:
    subprocess.run(
        ["applyxfm4D", str(echo), str(echo), str(out), str(mat_dir), "-fourdigit"],
        check=True, capture_output=True, text=True,
        env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
    )
    return np.asarray(nib.load(out).dataobj, dtype=np.float64)


def _end_slice_means(data):
    """Per-volume mean inside the disc of each end slice, as (first, last)."""
    return data[:, :, 0][DISC].mean(axis=0), data[:, :, -1][DISC].mean(axis=0)


def _fail_applyxfm4d(monkeypatch):
    def run(cmd, *args, **kwargs):
        if cmd[0] == "applyxfm4D":
            raise subprocess.CalledProcessError(1, cmd)
        return REAL_RUN(cmd, *args, **kwargs)

    monkeypatch.setattr(fp.subprocess, "run", run)
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI_GZ")


def test_frame_shift_commutes(tmp_path):
    """Moving by T then shifting equals shifting then moving by T'."""
    mat = _rotation_x(3.0, z_mm=5.0)
    mat[0, 3], mat[1, 3] = 1.5, -2.5
    source = _write_mats(tmp_path / "in", [mat])
    pad, dz = 2, 8.0

    shifted = _shift_mats_to_padded_frame(source, tmp_path / "out", dz, pad)

    assert [p.name for p in shifted] == ["MAT_0000"]
    shift = _translation(pad * dz)
    point = np.array([12.0, 40.0, 56.0, 1.0])
    np.testing.assert_allclose(
        shift @ mat @ point, np.loadtxt(shifted[0]) @ shift @ point, atol=1e-6
    )


def test_largest_through_slice_shift_is_in_slices(tmp_path):
    files = _write_mats(tmp_path, [np.eye(4), _translation(-12.0), _translation(4.0)])
    assert _largest_through_slice_shift(files, SHAPE, ZOOMS) == pytest.approx(1.5)


@needs_fsl
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_end_slices_are_kept_where_plain_applyxfm_loses_them(tmp_path, sign):
    echo = _constant_slab(tmp_path / "echo.nii.gz", 4)
    mats = [np.eye(4), _translation(sign * 4.0), np.eye(4), _translation(sign * 6.0)]
    files = _write_mats(tmp_path / "mats", mats)

    plain = _plain_applyxfm(echo, tmp_path / "mats", tmp_path / "plain.nii.gz")
    plain_first, plain_last = _end_slice_means(plain)
    assert min(plain_first.min(), plain_last.min()) < 0.95 * 1000  # the defect

    out = _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", tmp_path, 0)

    first, last = _end_slice_means(np.asarray(nib.load(out).dataobj, dtype=np.float64))
    np.testing.assert_allclose(first, 1000, rtol=0.01)
    np.testing.assert_allclose(last, 1000, rtol=0.01)


@needs_fsl
@pytest.mark.parametrize("x_scale", [4.0, -4.0], ids=["x-positive", "x-flipped"])
def test_slices_away_from_the_ends_match_plain_applyxfm(tmp_path, x_scale):
    """FSL flips x for one handedness; that must be the same in both frames."""
    echo = _structured_slab(
        tmp_path / "echo.nii.gz", 4, affine=np.diag([x_scale, 4.0, 8.0, 1.0])
    )
    mats = [np.eye(4), _translation(4.0), _translation(-4.0), _translation(6.0)]
    files = _write_mats(tmp_path / "mats", mats)

    plain = _plain_applyxfm(echo, tmp_path / "mats", tmp_path / "plain.nii.gz")
    out = _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", tmp_path, 0)

    padded = np.asarray(nib.load(out).dataobj, dtype=np.float64)
    np.testing.assert_allclose(padded[:, :, 3:18], plain[:, :, 3:18], rtol=0.01)


@needs_fsl
def test_pixdim_that_disagrees_with_the_affine_is_kept(tmp_path):
    """The matrices are in pixdim units, whatever the affine says."""
    source = nib.load(_structured_slab(tmp_path / "base.nii.gz", 4))
    img = nib.Nifti1Image(np.asarray(source.dataobj), np.eye(4))
    img.header.set_zooms((*ZOOMS, 1.0))
    echo = tmp_path / "echo.nii.gz"
    nib.save(img, echo)
    files = _write_mats(
        tmp_path / "mats", [np.eye(4), _translation(4.0), _translation(-4.0), _translation(6.0)]
    )

    plain = _plain_applyxfm(echo, tmp_path / "mats", tmp_path / "plain.nii.gz")
    out = _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", tmp_path, 0)

    padded = np.asarray(nib.load(out).dataobj, dtype=np.float64)
    np.testing.assert_allclose(padded[:, :, 3:18], plain[:, :, 3:18], rtol=0.01)


@needs_fsl
def test_geometry_equals_the_input(tmp_path):
    echo = _constant_slab(tmp_path / "echo.nii.gz", 2)
    source = nib.load(echo)
    files = _write_mats(tmp_path / "mats", [np.eye(4), _translation(4.0)])

    out = nib.load(_apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", tmp_path, 0))

    assert out.shape == source.shape
    np.testing.assert_array_equal(out.affine, source.affine)
    assert out.header.get_zooms() == source.header.get_zooms()


@needs_fsl
def test_pad_grows_with_a_shift_beyond_it(tmp_path, capsys):
    echo = _constant_slab(tmp_path / "echo.nii.gz", 2)
    files = _write_mats(tmp_path / "mats", [np.eye(4), _translation(2.4 * 8.0)])

    out = _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", tmp_path, 0)

    assert "padding 4 slices" in capsys.readouterr().out
    first, last = _end_slice_means(np.asarray(nib.load(out).dataobj, dtype=np.float64))
    np.testing.assert_allclose(first, 1000, rtol=0.01)
    np.testing.assert_allclose(last, 1000, rtol=0.01)


@needs_fsl
def test_fallback_keeps_the_end_slices(tmp_path, monkeypatch, capsys):
    _fail_applyxfm4d(monkeypatch)
    echo = _constant_slab(tmp_path / "echo.nii.gz", 3)
    files = _write_mats(tmp_path / "mats", [np.eye(4), _translation(4.0), _translation(-4.0)])

    out = _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", tmp_path, 0)

    assert "fallback" in capsys.readouterr().out
    loaded = nib.load(out)
    assert loaded.shape == (*SHAPE, 3)
    first, last = _end_slice_means(np.asarray(loaded.dataobj, dtype=np.float64))
    np.testing.assert_allclose(first, 1000, rtol=0.01)
    np.testing.assert_allclose(last, 1000, rtol=0.01)


@needs_fsl
def test_no_pad_folder_left_on_success(tmp_path):
    echo = _constant_slab(tmp_path / "echo.nii.gz", 2)
    files = _write_mats(tmp_path / "mats", [np.eye(4), _translation(4.0)])
    work = tmp_path / "work"
    work.mkdir()

    _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", work, 0)

    assert list(work.iterdir()) == []


def test_no_pad_folder_left_on_failure(tmp_path, monkeypatch):
    def run(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(fp.subprocess, "run", run)
    echo = _constant_slab(tmp_path / "echo.nii.gz", 2)
    files = _write_mats(tmp_path / "mats", [np.eye(4), _translation(4.0)])
    work = tmp_path / "work"
    work.mkdir()

    with pytest.raises(subprocess.CalledProcessError):
        _apply_transforms_padded(echo, files, tmp_path / "out.nii.gz", work, 0)

    assert list(work.iterdir()) == []


@needs_fsl
def test_frame_uses_dz_on_unequal_voxels(tmp_path, monkeypatch):
    """Rotation makes the result depend on the frame, so a wrong dz shows."""
    echo = _structured_slab(tmp_path / "echo.nii.gz", 2)
    files = _write_mats(tmp_path / "mats", [np.eye(4), _rotation_x(10.0)])
    plain = _plain_applyxfm(echo, tmp_path / "mats", tmp_path / "plain.nii.gz")
    middle = (slice(4, 20), slice(4, 20), slice(7, 14))

    out = _apply_transforms_padded(echo, files, tmp_path / "right.nii.gz", tmp_path, 0)
    right = np.asarray(nib.load(out).dataobj, dtype=np.float64)
    np.testing.assert_allclose(right[middle], plain[middle], rtol=0.01)

    shift = fp._shift_mats_to_padded_frame
    monkeypatch.setattr(
        fp, "_shift_mats_to_padded_frame", lambda m, o, dz, pad: shift(m, o, ZOOMS[0], pad)
    )
    out = _apply_transforms_padded(echo, files, tmp_path / "wrong.nii.gz", tmp_path, 0)
    wrong = np.asarray(nib.load(out).dataobj, dtype=np.float64)
    assert np.abs(wrong[middle] / plain[middle] - 1).max() > 0.02


def test_call_site_pads_other_echoes_and_keeps_mcflirt_for_the_reference(tmp_path, monkeypatch):
    echoes = [_constant_slab(tmp_path / f"echo{i}.nii.gz", 2) for i in range(3)]
    work = tmp_path / "work"
    padded_calls = []

    class Stop(Exception):
        pass

    def fake_run(cmd, *args, **kwargs):
        assert cmd[0] == "mcflirt", "only MCFLIRT may run before the padded step"
        base = Path(cmd[cmd.index("-out") + 1])
        _write_mats(Path(f"{base}.mat"), [np.eye(4)] * 2)
        return subprocess.CompletedProcess(cmd, 0)

    def fake_padded(echo_file, mat_files, output_file, work_dir, ref_vol):
        padded_calls.append((echo_file, output_file.name, len(mat_files), ref_vol))
        return output_file

    def stop_at_mean(path, *args, **kwargs):
        raise Stop

    monkeypatch.setattr(fp.subprocess, "run", fake_run)
    monkeypatch.setattr(fp, "_apply_transforms_padded", fake_padded)
    monkeypatch.setattr(fp.nib, "load", _load_unless_mcf(stop_at_mean))

    with pytest.raises(Stop):
        fp.run_multiecho_motion_correction(echoes, work, tmp_path / "mask.nii.gz")

    assert padded_calls == [
        (echoes[0], "echo0_mcf.nii.gz", 2, 1),
        (echoes[2], "echo2_mcf.nii.gz", 2, 1),
    ]


def _load_unless_mcf(on_mcf, real_load=nib.load):
    def load(path, *args, **kwargs):
        if str(path).endswith("_mcf.nii.gz"):
            return on_mcf(path)
        return real_load(path, *args, **kwargs)

    return load
