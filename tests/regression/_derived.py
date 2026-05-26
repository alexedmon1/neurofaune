"""Derived-metric comparison for the integration tier.

End-to-end preprocessing/stats output (ANTs/FSL) is not bit-reproducible across
tool versions, so comparing raw voxels causes false failures. Instead we freeze
robust *derived scalars* — mask overlap (Dice), map correlation, intensity
summary — and check a candidate stays close to golden within a generous
tolerance. This catches genuine regressions without tracking numeric jitter.

These tests are @pytest.mark.integration (run via `make integration`, locally
or on HPC where ANTs/FSL exist), never part of the fast PR gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GOLDEN = HERE / "golden"
UPDATE = os.environ.get("NEUROFAUNE_UPDATE_GOLDEN") == "1"


def dice(a, b) -> float:
    """Dice overlap of two binary masks."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    denom = int(a.sum() + b.sum())
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def map_correlation(a, b) -> float:
    """Pearson correlation between two maps (flattened)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def summary(arr) -> dict[str, float]:
    """Robust intensity summary of a map."""
    arr = np.asarray(arr, dtype=float).ravel()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def assert_metrics_close(actual: dict[str, float], name: str, *, rtol: float = 1e-2, atol: float = 1e-3) -> None:
    """Assert derived metrics match the committed golden within loose tolerance."""
    path = GOLDEN / f"{name}.metrics.json"
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        if not UPDATE:
            raise AssertionError(
                f"metrics golden '{name}' did not exist and was created at {path}; "
                "review and commit it, then re-run."
            )
        return
    expected = json.loads(path.read_text())
    assert set(actual) == set(expected), (
        f"metric keys changed for '{name}': {sorted(actual)} vs golden {sorted(expected)}"
    )
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=rtol, atol=atol,
            err_msg=f"derived metric '{key}' drifted for '{name}'",
        )
