"""Unit tests for implant artifact QC metrics and EPI readout geometry."""

import json

import nibabel as nib
import numpy as np
import pytest

from neurofaune.preprocess.qc.artifacts import maps
from neurofaune.preprocess.qc.artifacts.topup import _pad_even, acqparams_row
from neurofaune.utils.bruker_epi import (
    chemical_shift_hz,
    displacement_voxels,
    epi_readout,
    phase_encoding_axis,
)


def box_mask(shape=(10, 10, 20), z0=5, z1=15):
    m = np.zeros(shape, bool)
    m[2:8, 2:8, z0:z1] = True
    return m


class TestDepth:
    def test_dorsal_depth_counts_down_from_the_top_voxel(self):
        m = box_mask()
        d = maps.dorsal_depth_mm(m, 0.15)
        assert d[4, 4, 14] == pytest.approx(0.0)
        assert d[4, 4, 5] == pytest.approx(9 * 0.15)
        assert np.isnan(d[0, 0, 0])

    def test_ventral_height_is_the_mirror(self):
        m = box_mask()
        h = maps.ventral_height_mm(m, 0.15)
        d = maps.dorsal_depth_mm(m, 0.15)
        assert h[4, 4, 5] == pytest.approx(0.0)
        assert (h + d)[m] == pytest.approx(9 * 0.15)

    def test_shell_above_sits_over_the_dorsal_surface_only(self):
        m = box_mask()
        shell = maps.shell_above(m, 0.15, 0.0, 0.45)
        assert shell[4, 4, 15:18].all() and not shell[4, 4, 18]
        assert not (shell & m).any()
        assert not shell[0, 0].any()          # no brain in that column


class TestMetrics:
    def test_tsnr(self):
        rng = np.random.default_rng(0)
        x = 100 + rng.normal(0, 5, size=(2, 2, 2, 2000))
        assert maps.tsnr(x).mean() == pytest.approx(20, rel=0.05)
        assert maps.tsnr(np.ones((1, 1, 1, 5)))[0, 0, 0, ] == 0

    def test_normalised_ratio_is_one_in_reference_and_flags_dropout(self):
        b = np.full((4, 4, 4), 200.0)
        a = np.full((4, 4, 4), 50.0)
        a[0] = 10.0                            # dropout in the GE image
        ref = np.zeros_like(a, bool); ref[2:] = True
        r = maps.normalised_ratio(a, b, ref)
        assert r[3, 0, 0] == pytest.approx(1.0)
        assert r[0, 0, 0] == pytest.approx(0.2)

    def test_depth_profile_bins_and_ignores_nan(self):
        m = box_mask()
        d = maps.dorsal_depth_mm(m, 0.25)
        v = np.where(m, d * 2, np.nan)
        v[4, 4, 14] = np.nan
        rows = maps.depth_profile(v, d, m, bin_mm=0.5)
        assert rows[0]["depth_mm"] == pytest.approx(0.25)
        assert all(r["median"] == pytest.approx(2 * (r["depth_mm"] - 0.25) + 0.25, abs=0.3) for r in rows)


class TestReadout:
    def test_displacement_and_chemical_shift(self):
        # mouse pilot GE-EPI: 0.1808 ms effective spacing, 90 PE lines.
        assert displacement_voxels(100.0, 0.1808e-3, 90) == pytest.approx(1.627, abs=1e-3)
        assert chemical_shift_hz(7.0) == pytest.approx(1013.3, abs=1)

    def test_phase_axis_from_visu(self):
        assert phase_encoding_axis({"VisuAcqGradEncoding": ["read_enc", "phase_enc"]}) == 1
        assert phase_encoding_axis({"VisuAcqGradEncoding": "<phase_enc> <read_enc>"}) == 0
        assert phase_encoding_axis({}) == 1

    def _scan(self, tmp_path, blip, encoding="read_enc phase_enc", blips_off="No"):
        (tmp_path / "pdata" / "1").mkdir(parents=True)
        (tmp_path / "method").write_text(
            "##$PVM_EpiEchoSpacing=0.3616\n##$PVM_EpiNShots=2\n"
            f"##$PVM_EpiBlipOddGrad={blip}\n##$PVM_EpiBlipsOff={blips_off}\n"
            "##$PVM_EncPpi=( 2 )\n1 1\n##END=\n")
        (tmp_path / "pdata" / "1" / "visu_pars").write_text(
            f"##$VisuCoreSize=( 2 )\n90 90\n##$VisuAcqGradEncoding=( 2 )\n{encoding}\n##END=\n")
        return tmp_path

    def test_positive_blip_is_negative_direction(self, tmp_path):
        # Calibrated on the mouse pilot: topup vs B0 map anti-correlated
        # for every pair when the positive-blip scan was labelled '+'.
        out = epi_readout(self._scan(tmp_path, 0.137))
        assert out["PhaseEncodingDirection"] == "j-"
        assert out["EffectiveEchoSpacing"] == pytest.approx(0.1808e-3)
        assert out["TotalReadoutTime"] == pytest.approx(0.1808e-3 * 89)

    def test_coronal_phase_axis_and_reverse_blip(self, tmp_path):
        out = epi_readout(self._scan(tmp_path, -0.137, encoding="phase_enc read_enc"))
        assert out["PhaseEncodingDirection"] == "i"

    def test_blips_off_reference_has_no_direction(self, tmp_path):
        out = epi_readout(self._scan(tmp_path, 0.0, blips_off="Yes"))
        assert "PhaseEncodingDirection" not in out and out["PhaseEncodingAxis"] == "j"


class TestTopupHelpers:
    def test_acqparams_row(self):
        assert acqparams_row({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.0161}) == "0 -1 0 0.016100"
        with pytest.raises(KeyError, match="readout"):
            acqparams_row({"PhaseEncodingDirection": "j"})

    def test_pad_even_repeats_last_plane(self):
        data = np.arange(3 * 4 * 13 * 2, dtype=float).reshape(3, 4, 13, 2)
        padded, shape = _pad_even(data)
        assert shape == (3, 4, 13) and padded.shape == (4, 4, 14, 2)
        assert np.array_equal(padded[:, :, 13], padded[:, :, 12])
