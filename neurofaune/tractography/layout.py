"""Output layout for the tractography stage.

Tractography is a study-root stage directory of its own, alongside
``analysis/``, ``network/`` and ``qc/`` — not a subdirectory of
``derivatives/``. Two reasons:

- **Volume.** At 1M streamlines the stage produces roughly 300 MB per session
  that is kept and another 390 MB that is not, which is an order of magnitude
  more than the scalar derivatives it sits next to. Keeping it separate leaves
  ``derivatives/`` small enough to sync, archive and reason about, and lets the
  whole tractography tree be deleted or moved in one operation.
- **Optionality.** It is the only stage requiring MRtrix3, and many studies
  will never run it. A study that does not should not have empty
  ``tractography/`` directories interleaved through its subject derivatives.

Every path the stage writes is derived here so the convention lives in one
place rather than in each entry point's defaults.

Layout::

    {study_root}/
    ├── tractography/
    │   ├── sub-{subject}/ses-{session}/     per-session fibre model + tracts
    │   ├── template/                        group FOD template (FBA)
    │   ├── fixel/                           group fixel metrics (FD/logFC/FDC)
    │   └── stats/                           fixelcfestats output
    ├── network/connectome/{atlas}/          group-stacked ROI x ROI matrices
    └── work/sub-{subject}/ses-{session}/tractography/    discardable

Intermediates go to ``work/``: a ``.mif`` copy of the DWI duplicates a NIfTI
that already exists (263 MB/session here), and the pre-``mtnormalise`` FOD is
superseded by the normalised one. Neither belongs beside the results.
"""
from __future__ import annotations

from pathlib import Path

TRACTOGRAPHY_DIRNAME = "tractography"


def stage_dir(study_root: Path) -> Path:
    """``{study_root}/tractography``."""
    return Path(study_root) / TRACTOGRAPHY_DIRNAME


def session_dir(study_root: Path, subject: str, session: str) -> Path:
    """Per-session outputs: ``{study_root}/tractography/{subject}/{session}``.

    ``subject`` and ``session`` are used verbatim, so pass the full BIDS
    entities (``"sub-7Z"``, ``"ses-1"``) as the rest of the package does.
    """
    return stage_dir(study_root) / subject / session


def work_dir(study_root: Path, subject: str, session: str) -> Path:
    """Discardable intermediates, under the study's existing ``work/`` tree."""
    return (
        Path(study_root) / "work" / subject / session / TRACTOGRAPHY_DIRNAME
    )


def template_dir(study_root: Path) -> Path:
    """Group FOD template and fixel mask for fixel-based analysis."""
    return stage_dir(study_root) / "template"


def fixel_dir(study_root: Path) -> Path:
    """Group fixel metrics (FD, log FC, FDC) in template correspondence."""
    return stage_dir(study_root) / "fixel"


def stats_dir(study_root: Path) -> Path:
    """``fixelcfestats`` output."""
    return stage_dir(study_root) / "stats"


def connectome_dir(study_root: Path, atlas: str = "SIGMA") -> Path:
    """Group-stacked connectivity matrices.

    These live under ``network/`` rather than ``tractography/`` because a
    connectome is an ROI x ROI matrix consumed by the existing network
    machinery (``network.graph_theory``, CovNet, NBS), which is what
    ``network/`` is for. Per-session matrices stay in the session directory.
    """
    return Path(study_root) / "network" / "connectome" / atlas


def resolve_output_dir(
    output_dir: Path,
    subject: str,
    session: str,
    derive: bool = True,
) -> Path:
    """Resolve a caller's ``output_dir`` to the session directory.

    Matches the package's workflow convention (see CLAUDE.md, "Workflow
    Pattern"): ``output_dir`` is the **study root** and the layout below it is
    derived, so placement is a property of the package rather than of each
    call site.

    Parameters
    ----------
    output_dir : Path
        Study root.
    derive : bool
        Set False to treat ``output_dir`` as a literal destination instead.
        Intended for tests and one-off exploration, not for study runs.
    """
    if not derive:
        return Path(output_dir)
    return session_dir(output_dir, subject, session)
