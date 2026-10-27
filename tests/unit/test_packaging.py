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


# --- Data files ---------------------------------------------------------------------------------
# An install copies the neurofaune/ package and only the non-Python files package-data declares.
# Until 2026-10-27 the species profiles and default.yaml sat in a top-level configs/ folder, outside
# the package: a checkout found them, every installed copy did not, and load_config failed there with
# "No species profile for 'rat'" (and had never merged default.yaml at all).

import os
import re
import shutil
import subprocess
import sys
import zipfile

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib


def _package_data_patterns() -> list[str]:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["setuptools"]["package-data"]["neurofaune"]


def _glob_to_regex(pattern: str) -> re.Pattern:
    """setuptools package-data globbing: '*' stays within one directory."""
    return re.compile("^" + re.escape(pattern).replace(r"\*", "[^/]*") + "$")


def test_every_data_file_in_the_package_is_declared():
    """A non-Python file under neurofaune/ that no package-data pattern matches is missing from every install."""
    pats = [_glob_to_regex(p) for p in _package_data_patterns()]
    pkg = ROOT / "neurofaune"
    undeclared = sorted(
        p.relative_to(pkg).as_posix()
        for p in pkg.rglob("*")
        if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts
        and not any(r.match(p.relative_to(pkg).as_posix()) for r in pats)
    )
    assert not undeclared, f"data files an install would leave out (add them to package-data): {undeclared}"


def test_an_installed_copy_loads_a_rat_config(tmp_path):
    """Build the wheel, unpack it as an installer would, and load a rat study config from that copy alone."""
    src = tmp_path / "src"
    src.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, src / name)
    shutil.copytree(ROOT / "neurofaune", src / "neurofaune", ignore=shutil.ignore_patterns("__pycache__"))
    dist = tmp_path / "dist"
    dist.mkdir()
    build = subprocess.run([sys.executable, "-c", "import sys; from setuptools import build_meta as b; b.build_wheel(sys.argv[1])",
                            str(dist)], cwd=src, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-2000:]
    wheel = next(dist.glob("*.whl"))
    names = set(zipfile.ZipFile(wheel).namelist())
    for f in ("neurofaune/configs/default.yaml", "neurofaune/configs/species/rat.yaml",
              "neurofaune/configs/species/mouse.yaml", "neurofaune/atlas/groups/sigma_invivo.yaml"):
        assert f in names, f"{f} is not in the wheel"

    site = tmp_path / "site"
    zipfile.ZipFile(wheel).extractall(site)
    cfg = tmp_path / "study.yaml"
    cfg.write_text("study:\n  name: packaging test\n  code: PKG\n  species: rat\n")
    probe = ("import sys, neurofaune; from neurofaune.config import load_config; "
             "c = load_config(sys.argv[1], validate=False); "
             "print(neurofaune.__file__); print(c['functional']['filtering']['method'])")
    run = subprocess.run([sys.executable, "-c", probe, str(cfg)], cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(site)})
    assert run.returncode == 0, run.stderr[-2000:]
    loaded_from, method = run.stdout.split()[:2]
    assert loaded_from.startswith(str(site)), f"the probe imported {loaded_from}, not the unpacked wheel"
    assert method == "butterworth", "default.yaml was not merged (its functional.filtering.method is missing)"
