"""Unit tests for the BIDS RepetitionTime resolved from Bruker method parameters.

Hermetic: synthetic brukerapi-style method dicts ({key: {'value': ...}}). The two
segmented cases reproduce real protocols: a 2-shot multi-echo T2S_EPI BOLD
(1140 ms per shot, 360 volumes, scan time 820.8 s -> 2.28 s per volume) and a 4-shot
DtiEpi DWI (2000 ms per shot, 95 diffusion experiments x 2 repetitions, scan time
1520 s -> 8 s per volume).
"""
import logging

import pytest

from neurofaune.utils.bruker_convert import _resolve_repetition_time


def _m(**kw):
    return {k: {"value": v} for k, v in kw.items()}


def test_no_repetition_time():
    assert _resolve_repetition_time(_m(PVM_EchoTime=10)) == {}


def test_non_epi_keeps_per_excitation_tr():
    # RARE / FLASH / MSME carry no PVM_EpiNShots: TR unchanged, no shot fields
    out = _resolve_repetition_time(_m(PVM_RepetitionTime=4800, PVM_NAverages=2))
    assert out == {"RepetitionTime": pytest.approx(4.8)}


def test_single_shot_epi_unchanged():
    out = _resolve_repetition_time(_m(PVM_RepetitionTime=1500, PVM_EpiNShots=1, PVM_NAverages=1,
                                      PVM_NRepetitions=300, PVM_ScanTime=450000))
    assert out == {"RepetitionTime": pytest.approx(1.5)}


def test_two_shot_bold(caplog):
    with caplog.at_level(logging.WARNING):
        out = _resolve_repetition_time(_m(PVM_RepetitionTime=1140, PVM_EpiNShots=2, PVM_NAverages=1,
                                          PVM_NRepetitions=360, PVM_ScanTime=820800))
    assert out["RepetitionTime"] == pytest.approx(2.28)
    assert out["NumberShots"] == out["NumberOfShots"] == 2
    assert out["RepetitionTimeExcitation"] == out["RepetitionTimePerShot"] == pytest.approx(1.14)
    assert "disagrees" not in caplog.text


def test_four_shot_dwi_counts_diffusion_experiments(caplog):
    with caplog.at_level(logging.WARNING):
        out = _resolve_repetition_time(_m(PVM_RepetitionTime=2000, PVM_EpiNShots=4, PVM_NAverages=1,
                                          PVM_NRepetitions=2, PVM_DwNDiffExp=95, PVM_ScanTime=1520000))
    assert out["RepetitionTime"] == pytest.approx(8.0)
    assert out["NumberShots"] == 4
    assert "disagrees" not in caplog.text


@pytest.mark.parametrize("shots", [2, 3, 4, 8, 16])
def test_any_number_of_shots_and_averages(shots):
    out = _resolve_repetition_time(_m(PVM_RepetitionTime=1000, PVM_EpiNShots=shots, PVM_NAverages=2))
    assert out["RepetitionTime"] == pytest.approx(1.0 * shots * 2)
    assert out["NumberShots"] == shots
    assert out["RepetitionTimeExcitation"] == pytest.approx(1.0)


def test_scan_time_mismatch_is_logged_not_fatal(caplog):
    with caplog.at_level(logging.WARNING):
        out = _resolve_repetition_time(_m(PVM_RepetitionTime=1140, PVM_EpiNShots=2, PVM_NAverages=1,
                                          PVM_NRepetitions=360, PVM_ScanTime=410400))   # implies 1.14 s
    assert out["RepetitionTime"] == pytest.approx(2.28)
    assert "disagrees" in caplog.text


def test_scalar_values_in_lists():
    # brukerapi sometimes returns single values as one-element lists
    out = _resolve_repetition_time({"PVM_RepetitionTime": {"value": [1140]}, "PVM_EpiNShots": {"value": [2]}})
    assert out["RepetitionTime"] == pytest.approx(2.28)
