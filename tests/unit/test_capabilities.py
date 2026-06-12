"""
Unit tests for the capability catalog (the "nothing missed" mechanism).

Hermetic: pure introspection of the installed package. Two guarantees:
  1. The walk imports every module — IMPORT_ERRORS must be empty, so a capability
     hidden behind a broken import is a test failure, not a silent omission.
  2. The committed CAPABILITIES.md is current — adding/removing/renaming an entry
     point without running `make capabilities` fails the gate (content compared
     version-insensitively, so a release bump alone doesn't break it).
"""
from pathlib import Path

import pytest

from neurofaune import capabilities as cap

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "CAPABILITIES.md"


def _strip_version(md: str) -> str:
    # the only version-dependent line; ignore it so a pin bump alone is fine
    return "\n".join(l for l in md.splitlines() if not l.startswith("_Generated"))


def test_catalog_non_empty_and_complete():
    caps = cap.collect_capabilities()
    assert len(caps) > 50  # sanity floor; ~119 today
    names = {c.name for c in caps}
    # representative entry points across stages must be present
    for expected in [
        "run_anatomical_preprocessing",
        "segment_brain_tissue_atropos",
        "build_template",
        "register_template_to_sigma",
        "fit_dti",
        "fit_noddi",
        "compute_fc_matrix",
        "compute_cohens_d_map",
    ]:
        assert expected in names, f"missing capability: {expected}"


def test_no_import_errors():
    # populates IMPORT_ERRORS as a side effect of the walk
    cap.collect_capabilities()
    assert not cap.IMPORT_ERRORS, (
        "modules failed to import during capability scan (capabilities may be "
        f"hidden): {cap.IMPORT_ERRORS}"
    )


def test_every_capability_has_a_summary():
    for c in cap.collect_capabilities():
        assert c.summary and c.summary != "(no docstring)", (
            f"{c.module}.{c.name} has no docstring — add a one-line summary"
        )


def test_committed_catalog_is_current():
    assert CATALOG.exists(), "CAPABILITIES.md missing — run `make capabilities`"
    on_disk = _strip_version(CATALOG.read_text())
    regenerated = _strip_version(cap.render_markdown())
    assert on_disk == regenerated, (
        "CAPABILITIES.md is stale — run `make capabilities` and commit the result"
    )
