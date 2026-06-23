"""Helpers for behavior-preservation (golden) regression tests.

The *boundary* of an implementation is a frozen contract:
    given a fixed input (``fixtures/``), a candidate implementation must
    reproduce a fixed output (``golden/``) within tolerance.

Any number of candidate implementations can be compared against the same
golden output to verify behavior is preserved across changes.

Updating golden outputs is deliberate and explicit:
    NEUROFAUNE_UPDATE_GOLDEN=1 uv run pytest -m regression
Only do this when you have *decided* the new behavior is correct, and commit
the changed golden files in their own reviewable commit.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
GOLDEN = HERE / "golden"

UPDATE = os.environ.get("NEUROFAUNE_UPDATE_GOLDEN") == "1"


def fixture_path(name: str) -> Path:
    """Resolve a frozen input fixture by name."""
    return FIXTURES / name


def golden_path(name: str) -> Path:
    """Resolve a golden-output file by name."""
    return GOLDEN / name


def assert_array_matches_golden(
    actual: np.ndarray,
    name: str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Assert ``actual`` matches the golden array ``name`` within tolerance.

    Writes/refreshes the golden file instead of asserting when
    ``NEUROFAUNE_UPDATE_GOLDEN=1`` is set.
    """
    actual = np.asarray(actual)
    path = golden_path(f"{name}.npy")
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, actual)
        if not UPDATE:
            raise AssertionError(
                f"golden '{name}' did not exist and was created at {path}; "
                "review and commit it, then re-run."
            )
        return
    expected = np.load(path)
    assert actual.shape == expected.shape, (
        f"shape drift for '{name}': candidate {actual.shape} vs golden {expected.shape}"
    )
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"candidate output for '{name}' drifted outside tolerance from golden",
    )
