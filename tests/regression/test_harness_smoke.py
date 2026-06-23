"""Smoke test for the regression harness itself.

This keeps `make check` green out of the box and proves the golden
mechanism works: matching arrays pass, drifted arrays fail, and the
NEUROFAUNE_UPDATE_GOLDEN path writes a fresh golden.

Real per-change regression tests (comparing a candidate implementation
against a committed golden) are added as needed — see
tests/regression/README.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression import _equivalence as eq

pytestmark = pytest.mark.regression


def _deterministic_array() -> np.ndarray:
    rng = np.random.default_rng(seed=20260526)
    return rng.standard_normal((8, 8))


def test_golden_create_then_match(tmp_path, monkeypatch):
    """First run creates the golden; an identical second run matches it."""
    monkeypatch.setattr(eq, "GOLDEN", tmp_path)

    # Create-on-missing: writes the golden and signals it must be reviewed.
    monkeypatch.setattr(eq, "UPDATE", True)
    eq.assert_array_matches_golden(_deterministic_array(), "smoke")
    assert (tmp_path / "smoke.npy").exists()

    # Subsequent run compares against the frozen golden and passes.
    monkeypatch.setattr(eq, "UPDATE", False)
    eq.assert_array_matches_golden(_deterministic_array(), "smoke")


def test_golden_detects_drift(tmp_path, monkeypatch):
    """A candidate that drifts outside tolerance fails the boundary."""
    monkeypatch.setattr(eq, "GOLDEN", tmp_path)
    monkeypatch.setattr(eq, "UPDATE", True)
    eq.assert_array_matches_golden(_deterministic_array(), "smoke")

    monkeypatch.setattr(eq, "UPDATE", False)
    drifted = _deterministic_array() + 1e-3
    with pytest.raises(AssertionError):
        eq.assert_array_matches_golden(drifted, "smoke")
