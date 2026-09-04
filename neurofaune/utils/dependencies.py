"""External tool discovery and install guidance.

neurofaune wraps several neuroimaging suites that cannot be installed from
PyPI, so ``uv sync`` succeeding tells a new user nothing about whether the
pipeline will run. The failure that follows is usually a ``FileNotFoundError``
from deep inside a workflow, naming one binary and offering no route to
getting it.

This module declares what each feature area needs, checks it, and carries the
install guidance next to the requirement so the two cannot drift apart. It
backs ``neurofaune check-deps`` and the error messages raised by the modules
themselves.

Tools are grouped by feature area rather than listed flat, because most
studies need only some of them: a lab doing anatomical and functional
preprocessing needs FSL and ANTs and should not be told MRtrix3 is missing as
though the install were broken.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# --- Install guidance -------------------------------------------------------
# Keyed by package. Kept here so a single edit updates the CLI report, the
# README generator and every exception message.

INSTALL_HINTS: Dict[str, str] = {
    "FSL": (
        "FSL 6.0+ — https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index\n"
        "  Official installer (Linux/macOS):\n"
        "    python <(curl -sL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/getfsl.sh)"
    ),
    "ANTs": (
        "ANTs 2.3+ — https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS\n"
        "  conda:  conda install -c aramislab ants\n"
        "  Or use a prebuilt release: https://github.com/ANTsX/ANTs/releases"
    ),
    "MRtrix3": (
        "MRtrix3 3.0+ — https://www.mrtrix.org/download/\n"
        "  conda/mamba (no root, all platforms — recommended):\n"
        "    conda install -c mrtrix3 mrtrix3\n"
        "  Ubuntu/Debian:  sudo apt install mrtrix3\n"
        "  macOS:          brew install mrtrix3\n"
        "  container:      docker pull mrtrix3/mrtrix3\n"
        "  Then either put its bin/ on PATH, export MRTRIX_BIN=<prefix>/bin,\n"
        "  or set tractography.mrtrix_bin in your study config.yaml."
    ),
    "CUDA": (
        "CUDA (optional) — enables eddy_cuda and bedpostx_gpu.\n"
        "  Provided by your NVIDIA driver + FSL's CUDA-enabled binaries."
    ),
}


@dataclass(frozen=True)
class Tool:
    """One external binary neurofaune may invoke."""

    binary: str
    package: str
    purpose: str
    optional: bool = False
    version_flag: Optional[str] = "--version"


@dataclass
class ToolStatus:
    """Result of looking for a :class:`Tool`."""

    tool: Tool
    found: bool
    path: Optional[str] = None
    version: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True when present, or absent but optional."""
        return self.found or self.tool.optional


# --- Requirements by feature area -------------------------------------------

DEPENDENCY_GROUPS: Dict[str, List[Tool]] = {
    "core": [
        Tool("antsRegistration", "ANTs", "registration", version_flag="--version"),
        Tool("N4BiasFieldCorrection", "ANTs", "bias field correction"),
        Tool("Atropos", "ANTs", "tissue segmentation"),
        Tool("antsApplyTransforms", "ANTs", "resampling / transform application"),
        Tool("fslmaths", "FSL", "image arithmetic", version_flag=None),
        Tool("bet", "FSL", "brain extraction", version_flag=None),
    ],
    "diffusion": [
        Tool("eddy", "FSL", "eddy-current / motion correction", version_flag=None),
        Tool("dtifit", "FSL", "tensor fitting", version_flag=None),
        Tool("eddy_cuda", "CUDA", "GPU eddy (10-50x faster)",
             optional=True, version_flag=None),
    ],
    "functional": [
        Tool("mcflirt", "FSL", "motion correction", version_flag=None),
        Tool("melodic", "FSL", "ICA denoising", version_flag=None),
    ],
    "tractography": [
        Tool("dwi2response", "MRtrix3", "response function estimation"),
        Tool("dwi2fod", "MRtrix3", "FOD fitting (MSMT-CSD)"),
        Tool("tckgen", "MRtrix3", "streamline generation"),
        Tool("tcksift2", "MRtrix3", "streamline weighting"),
        Tool("tck2connectome", "MRtrix3", "connectivity matrices"),
        Tool("fixelcfestats", "MRtrix3", "fixel-based analysis statistics"),
        Tool("bedpostx", "FSL", "ball-and-sticks fitting (FSL path)",
             optional=True, version_flag=None),
        Tool("probtrackx2", "FSL", "probabilistic tracking (FSL path)",
             optional=True, version_flag=None),
    ],
}

MRTRIX_MINIMUM_VERSION = "3.0.0"
"""Earliest MRtrix3 providing every command the tractography module calls.
Development and verification were done against 3.0.8."""


def _probe_version(tool: Tool, path: str) -> Optional[str]:
    """Best-effort version string; None when unavailable or slow to obtain."""
    if tool.version_flag is None:
        return None
    try:
        proc = subprocess.run(
            [path, tool.version_flag], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def check_dependencies(
    groups: Optional[Sequence[str]] = None,
    probe_versions: bool = True,
    extra_paths: Optional[Sequence[str]] = None,
) -> Dict[str, List[ToolStatus]]:
    """Check external tools, grouped by feature area.

    Parameters
    ----------
    groups : sequence of str, optional
        Feature areas to check. Defaults to all of :data:`DEPENDENCY_GROUPS`.
    probe_versions : bool
        Run each binary to read its version. Disable for a faster check.
    extra_paths : sequence of str, optional
        Additional directories to search before ``PATH`` — e.g. a configured
        ``tractography.mrtrix_bin``.

    Returns
    -------
    dict
        Feature area -> list of :class:`ToolStatus`.
    """
    selected = list(groups) if groups else list(DEPENDENCY_GROUPS)
    unknown = [g for g in selected if g not in DEPENDENCY_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown dependency group(s): {unknown}; "
            f"known groups are {sorted(DEPENDENCY_GROUPS)}"
        )

    search_path = None
    if extra_paths:
        import os
        search_path = os.pathsep.join(
            [*[str(p) for p in extra_paths], os.environ.get("PATH", "")]
        )

    results: Dict[str, List[ToolStatus]] = {}
    for group in selected:
        statuses: List[ToolStatus] = []
        for tool in DEPENDENCY_GROUPS[group]:
            found_path = shutil.which(tool.binary, path=search_path)
            version = (
                _probe_version(tool, found_path)
                if found_path and probe_versions
                else None
            )
            statuses.append(
                ToolStatus(
                    tool=tool,
                    found=found_path is not None,
                    path=found_path,
                    version=version,
                )
            )
        results[group] = statuses
    return results


def missing_packages(results: Dict[str, List[ToolStatus]]) -> List[str]:
    """Distinct packages with at least one missing required binary."""
    out: List[str] = []
    for statuses in results.values():
        for st in statuses:
            if not st.found and not st.tool.optional and st.tool.package not in out:
                out.append(st.tool.package)
    return out


def format_report(results: Dict[str, List[ToolStatus]], color: bool = True) -> str:
    """Render a human-readable dependency report with install guidance."""
    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    lines: List[str] = []
    for group, statuses in results.items():
        lines.append(f"\n{group}:")
        for st in statuses:
            if st.found:
                mark, colour = "OK", "32"
            elif st.tool.optional:
                mark, colour = "--", "33"
            else:
                mark, colour = "MISSING", "31"
            detail = st.tool.purpose
            if st.version:
                detail += f", v{st.version}"
            if not st.found and st.tool.optional:
                detail += ", optional"
            lines.append(
                f"  [{paint(mark, colour):>{7 + (9 if color else 0)}}] "
                f"{st.tool.binary:<22} {st.tool.package:<9} ({detail})"
            )

    missing = missing_packages(results)
    if missing:
        lines.append("\n" + "=" * 70)
        lines.append("Missing required packages: " + ", ".join(missing))
        lines.append("=" * 70)
        for pkg in missing:
            hint = INSTALL_HINTS.get(pkg)
            if hint:
                lines.append("\n" + hint)
    else:
        lines.append("\n" + paint("All required tools found.", "32"))
    return "\n".join(lines)
