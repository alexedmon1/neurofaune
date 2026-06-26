"""Unit tests for Bruker -> BIDS echo-time resolution (_resolve_echo_time).

Pins the multi-echo fix: per-echo TEs come from the EffectiveEchoTime array, and
PVM_EchoTime (the echo SPACING, ~0.29 ms, for T2S_EPI) is never used as a TE.
"""
import pytest

from neurofaune.utils.bruker_convert import _resolve_echo_time


def _method(**kw):
    """Build a brukerapi-style parsed method dict: {key: {'value': v}}."""
    return {k: {"value": v} for k, v in kw.items()}


# Real cuprizone T2S_EPI resting-state values.
ME = dict(PVM_NEchoImages=3, EffectiveEchoTime=[8.0, 20.2, 32.4], PVM_EchoTime=0.29)


def test_multiecho_per_echo_te():
    m = _method(**ME)
    assert _resolve_echo_time(m, 0) == {"EchoTime": 0.008, "EchoNumber": 1}
    assert _resolve_echo_time(m, 1)["EchoTime"] == pytest.approx(0.0202)
    assert _resolve_echo_time(m, 1)["EchoNumber"] == 2
    assert _resolve_echo_time(m, 2)["EchoTime"] == pytest.approx(0.0324)
    assert _resolve_echo_time(m, 2)["EchoNumber"] == 3


def test_multiecho_never_uses_echo_spacing():
    m = _method(**ME)
    for ei in (0, 1, 2):
        assert _resolve_echo_time(m, ei)["EchoTime"] != pytest.approx(0.00029)


def test_multiecho_default_is_first_echo_no_echonumber():
    m = _method(PVM_NEchoImages=3, EffectiveEchoTime=[8.0, 20.2, 32.4])
    out = _resolve_echo_time(m, None)
    assert out["EchoTime"] == pytest.approx(0.008)
    assert "EchoNumber" not in out


def test_out_of_range_echo_index_falls_back_to_first():
    m = _method(**ME)
    assert _resolve_echo_time(m, 9)["EchoTime"] == pytest.approx(0.008)


def test_singleecho_uses_pvm_echotime():
    assert _resolve_echo_time(_method(PVM_EchoTime=15.0), None) == {"EchoTime": 0.015}


def test_no_echo_info_returns_empty():
    assert _resolve_echo_time(_method(), None) == {}
