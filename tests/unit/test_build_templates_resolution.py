#!/usr/bin/env python3
"""Tests that build_templates.py carries no study's design in its code.

It previously hardcoded one study's cohorts (p30/p60/p90), assumed a cohort named
its own session, and wrote files called ``tpl-BPARat_*`` whatever data it was given.
Those are the three things pinned here.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "build_templates", Path(__file__).parents[2] / "scripts" / "build_templates.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def test_prefix_comes_from_the_study_code():
    assert bt.resolve_prefix({"study": {"code": "CPZ001"}}, "p60") == "tpl-CPZ001p60"


def test_prefix_falls_back_to_a_neutral_stem_not_a_study_name():
    """No study code must not mean another study's name."""
    got = bt.resolve_prefix({}, "p60")
    assert got == "tpl-p60"
    assert "BPA" not in got


def test_prefix_template_placeholder_is_honoured():
    cfg = {"templates": {"prefix": "tpl-CPZ{cohort}_T2w"}}
    assert bt.resolve_prefix(cfg, "p120") == "tpl-CPZp120_T2w"


def test_cli_prefix_overrides_config():
    cfg = {"study": {"code": "CPZ001"}, "templates": {"prefix": "x"}}
    assert bt.resolve_prefix(cfg, "p60", override="tpl-mine") == "tpl-mine"


def test_declared_cohort_session_map_wins():
    """This study's cohorts are p60/p90/p120 but its sessions are ses-1/2/3."""
    cfg = {"templates": {"cohort_sessions": {"p60": "ses-1", "p90": "ses-2",
                                             "p120": "ses-3"}}}
    assert bt.resolve_session(cfg, "p120") == "ses-3"


def test_session_falls_back_to_the_cohort_name():
    """Only correct for studies that label sessions by age -- but it is the old default."""
    assert bt.resolve_session({}, "p60") == "ses-p60"


def test_a_cohort_already_named_as_a_session_is_not_double_prefixed():
    assert bt.resolve_session({}, "ses-2") == "ses-2"


def test_cohorts_are_declared_not_hardcoded():
    cfg = {"templates": {"cohorts": ["p60", "p90", "p120"]}}
    assert bt.discover_cohorts(cfg, Path("/nonexistent")) == ["p60", "p90", "p120"]


def test_cohorts_are_discovered_from_the_tree_when_undeclared(tmp_path):
    for sub in ("sub-1C", "sub-2C"):
        for ses in ("ses-1", "ses-2", "ses-3"):
            (tmp_path / sub / ses).mkdir(parents=True)
    assert bt.discover_cohorts({}, tmp_path) == ["ses-1", "ses-2", "ses-3"]


def test_an_empty_tree_is_an_error_not_a_guess():
    with pytest.raises(SystemExit, match="templates.cohorts"):
        bt.discover_cohorts({}, Path("/nonexistent"))
