"""Unit tests for the config-driven Bruker→BIDS converter (gh #3/#4/#5/#8).

Hermetic: the dimensionality logic is a pure numpy function, so we test it on
synthetic frame-group arrays without any real Bruker data.
"""
import re

import numpy as np
import pytest

from neurofaune.utils.bids import (
    DEFAULT_SEQUENCE_MAP,
    BidsConfig,
    assemble_modality,
    bids_filename,
    clean_fg_name,
    frame_group_names,
    parse_session_name,
)


def test_bids_filename_no_double_entity():
    # multi-echo: entity inserted exactly once, suffix stays last
    assert bids_filename("sub-1Y_ses-1_run-9", "echo-1", "bold") == "sub-1Y_ses-1_run-9_echo-1_bold"
    # no entity -> just suffix
    assert bids_filename("sub-1Y_ses-1_run-5", "", "T2w") == "sub-1Y_ses-1_run-5_T2w"


def test_clean_fg_name():
    assert clean_fg_name("<FG_SLICE>") == "slice"
    assert clean_fg_name("<FG_DIFFUSION>") == "diffusion"


def test_frame_group_names():
    fg = [[27, "<FG_SLICE>", "<>", 0, 2], [95, "<FG_DIFFUSION>", "<>", 2, 3], [2, "<FG_CYCLE>", "<>", 5, 1]]
    assert frame_group_names(fg) == ["slice", "diffusion", "cycle"]
    assert frame_group_names(None) == []


def test_assemble_dwi_averages_cycle():
    """DWI (x,y,slice,diffusion,cycle): cycle axis (the 2 acquisitions) is AVERAGED."""
    X, Y, Z, NDIFF = 4, 5, 3, 7
    a = np.zeros((X, Y, Z, NDIFF, 2), dtype=float)
    a[..., 0] = 2.0   # cycle 0
    a[..., 1] = 4.0   # cycle 1  -> mean should be 3.0
    out = assemble_modality(a, ["slice", "diffusion", "cycle"], "dwi")
    assert set(out) == {""}
    vol = out[""]
    assert vol.shape == (X, Y, Z, NDIFF)        # 4-D, aligned to bval/bvec
    assert np.allclose(vol, 3.0)                # mean of the 2 cycles


def test_assemble_func_multiecho_splits_echoes_keeps_time():
    """func (x,y,echo,slice,cycle): split echoes, keep cycle as 4-D time axis."""
    X, Y, NECHO, Z, T = 4, 5, 3, 6, 8
    a = np.zeros((X, Y, NECHO, Z, T), dtype=float)
    for e in range(NECHO):
        a[:, :, e, :, :] = e + 1            # tag each echo
    out = assemble_modality(a, ["echo", "slice", "cycle"], "func")
    assert set(out) == {"echo-1", "echo-2", "echo-3"}
    for e in range(NECHO):
        v = out[f"echo-{e + 1}"]
        assert v.shape == (X, Y, Z, T)         # 4-D (x,y,z,t) per echo
        assert np.allclose(v, e + 1)


def test_assemble_anat_plain():
    a = np.arange(4 * 5 * 7).reshape(4, 5, 7).astype(float)
    out = assemble_modality(a, ["slice"], "anat")
    assert set(out) == {""}
    assert out[""].shape == (4, 5, 7)


def test_assemble_msme_keeps_echo():
    """MSME/MESE (x,y,echo,slice) -> (x,y,z,echo)."""
    X, Y, NECHO, Z = 4, 5, 32, 11
    a = np.zeros((X, Y, NECHO, Z), dtype=float)
    out = assemble_modality(a, ["echo", "slice"], "anat")
    assert set(out) == {""}
    assert out[""].shape == (X, Y, Z, NECHO)


def test_parse_session_name_and_relabel():
    rgx = re.compile(
        r"^IRC\d+_\w+_CageCPZ(?P<cage>\w+?)_Rat(?P<subject>\d+[A-Za-z])_(?P<session>\d+[a-z]?)__"
    )
    m = parse_session_name(
        "IRC1200_Cuprizone_CageCPZ1_Rat1Y_1__Cage_CPZ1__Rat_1Y_1_1_20260309_083156",
        rgx, {})
    assert m["subject"] == "1Y" and m["session"] == "1"
    # lowercase rat id is upper-cased; '1a' relabel applies
    m2 = parse_session_name("IRC1200_X_CageCPZ2_Rat4y_1a__x", rgx, {"1a": "1b"})
    assert m2["subject"] == "4Y" and m2["session"] == "1b"
    assert parse_session_name("not_a_session", rgx, {}) is None


def test_parse_session_name_requires_named_groups():
    bad = re.compile(r"^(?P<subject>\d+)")  # missing 'session'
    with pytest.raises(ValueError):
        parse_session_name("12abc", bad, {})


def test_sequence_map_defaults_and_override():
    # default now maps the BOLD + FISP sequences that used to be dropped (gh #4)
    assert DEFAULT_SEQUENCE_MAP["Bruker:T2S_EPI"] == ("func", "bold")
    assert "Bruker:FISP" in DEFAULT_SEQUENCE_MAP
    cfg = BidsConfig(
        raw_root=".", bids_root=".",
        sequence_map={"Bruker:T2S_EPI": ("anat", "T2starw")},
    )
    assert cfg.sequence_map["Bruker:T2S_EPI"] == ("anat", "T2starw")   # override wins
    assert cfg.sequence_map["Bruker:RARE"] == ("anat", "T2w")          # default kept


def test_bids_config_from_config_requires_roots():
    with pytest.raises(ValueError):
        BidsConfig.from_config({"bids": {"raw_root": "/x"}})  # no bids_root
