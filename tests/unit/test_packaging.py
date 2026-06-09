"""Packaging completeness — the wheel must actually ship the whole library.

Regression guard for the bug where `[tool.setuptools] packages = [...]` was a
hardcoded list that silently dropped most subpackages (preprocess.workflows,
registration, templates, analysis.*, ...), so a tag install contained almost
none of the pipeline. We now use auto-discovery; these tests fail loudly if a
package on disk is not discoverable, or if any subpackage isn't importable
(i.e. not self-contained).
"""
import importlib
from pathlib import Path

import pytest
from setuptools import find_packages

ROOT = Path(__file__).resolve().parents[2]


def _on_disk_packages() -> set[str]:
    return {
        str(p.parent.relative_to(ROOT)).replace("/", ".")
        for p in ROOT.glob("neurofaune/**/__init__.py")
    }


def test_all_on_disk_packages_are_discovered():
    """Every neurofaune package on disk must be picked up by setuptools discovery."""
    discovered = set(find_packages(where=str(ROOT), include=["neurofaune*"]))
    missing = _on_disk_packages() - discovered
    assert not missing, f"packages on disk but NOT shipped by the build: {sorted(missing)}"


@pytest.mark.parametrize("pkg", sorted(_on_disk_packages()))
def test_subpackage_is_importable(pkg):
    """Each shipped subpackage must import cleanly (self-contained, no missing modules)."""
    importlib.import_module(pkg)
