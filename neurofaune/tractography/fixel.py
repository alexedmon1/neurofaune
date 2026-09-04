"""Fixel-based analysis (FBA): fibre density, cross-section, and their product.

For a demyelination model this is the more direct measurement than a
connectome. A *fixel* is one fibre population within one voxel, so the metrics
are specific to a fibre bundle rather than averaged over everything crossing a
voxel — which is exactly where FA fails. FA falls both when myelin is lost and
when a second crossing population is gained, and rises when one population
atrophies away; in voxels with crossing fibres (most of white matter) it cannot
distinguish these. FBA separates them:

- **FD** (fibre density) — the FOD lobe integral, sensitive to intra-axonal
  volume within that specific bundle. This is the microstructural term.
- **FC** (fibre cross-section) — from the Jacobian of the subject-to-template
  warp perpendicular to the fibre, a macrostructural term capturing bundle
  thinning. Analysed as ``log(FC)`` so it is centred and symmetric.
- **FDC** = FD x FC — the combined effect, and usually the most sensitive to a
  real reduction in total information carried by the bundle.

Two constraints the MRtrix FBA pipeline imposes, both enforced here:

1. **A single group-average response function per tissue, shared by every
   subject.** Per-subject responses would make FD differences partly reflect
   the response estimate rather than the data.
2. **Statistics must use connectivity-based fixel enhancement**
   (``fixelcfestats``), which smooths evidence along fibre-connected fixels
   rather than through space, and corrects across the fixel mask by
   permutation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from neurofaune.tractography.mrtrix import _run, require_mrtrix

logger = logging.getLogger(__name__)


def compute_group_response(
    response_files: Sequence[Path],
    output_file: Path,
    config: Optional[dict] = None,
    force: bool = False,
) -> Path:
    """Average per-subject response functions into one group response.

    Every subject's FOD must be fit with the *same* response function, or
    between-subject FD differences partly encode differences in the response
    estimate rather than in the tissue.

    Parameters
    ----------
    response_files : sequence of Path
        Per-subject responses for one tissue, from ``dwi2response dhollander``.
    output_file : Path
        Destination for the averaged response.
    """
    if output_file.exists() and not force:
        logger.info("group response already exists: %s", output_file)
        return output_file
    bin_dir = require_mrtrix(config)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["responsemean", *[str(f) for f in response_files], str(output_file), "-force"],
        bin_dir, f"responsemean over {len(response_files)} subjects",
    )
    return output_file


def build_fod_template(
    fod_files: Sequence[Path],
    mask_files: Sequence[Path],
    output_file: Path,
    config: Optional[dict] = None,
    voxel_size_mm: Optional[float] = None,
    voxel_scale: float = 10.0,
    force: bool = False,
    nthreads: Optional[int] = None,
) -> Path:
    """Build a study-specific FOD template with ``population_template``.

    An FOD template is required rather than a scalar (FA) one: fixel
    correspondence needs the template to carry orientation information so that
    each subject's fibre populations can be matched to the template's.

    MRtrix recommends building from a subset (30-40 subjects) rather than the
    whole cohort; a balanced subset across groups avoids biasing the template
    toward the larger or more affected group.

    Parameters
    ----------
    fod_files, mask_files : sequence of Path
        Per-subject WM FODs (normalised, fit with the group response) and their
        brain masks, in matching order.
    voxel_size_mm : float, optional
        Template voxel size in **real** mm; converted internally.
    """
    if output_file.exists() and not force:
        logger.info("FOD template already exists: %s", output_file)
        return output_file
    if len(fod_files) != len(mask_files):
        raise ValueError("fod_files and mask_files must correspond one-to-one")

    bin_dir = require_mrtrix(config)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # population_template consumes directories, not file lists.
    scratch = output_file.parent / ".template_inputs"
    fod_dir, mask_dir = scratch / "fod", scratch / "mask"
    fod_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i, (f, m) in enumerate(zip(fod_files, mask_files)):
        # Symlink under a shared stem so population_template pairs them up.
        stem = f"sub{i:04d}.mif"
        for src, dst in ((f, fod_dir / stem), (m, mask_dir / stem)):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(Path(src).resolve())

    cmd: List[str] = [
        "population_template", str(fod_dir), str(output_file),
        "-mask_dir", str(mask_dir), "-voxel_size",
    ]
    cmd.append(
        str(voxel_size_mm * voxel_scale) if voxel_size_mm is not None else "1.25"
    )
    cmd += ["-force"]
    logger.info("Building FOD template from %d subjects (slow)", len(fod_files))
    _run(cmd, bin_dir, "population_template", nthreads)
    return output_file


def register_fod_to_template(
    fod_file: Path,
    template_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    mask_file: Optional[Path] = None,
    config: Optional[dict] = None,
    force: bool = False,
    nthreads: Optional[int] = None,
) -> Dict[str, Path]:
    """Register one subject's FOD to the template, keeping both warps.

    Both directions are retained because the pipeline needs each: the forward
    warp carries subject data into template space, and the *inverse* warp's
    Jacobian is what FC is computed from.
    """
    bin_dir = require_mrtrix(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"
    out = {
        "warp": output_dir / f"{prefix}_from-subject_to-template_warp.mif",
        "inverse_warp": output_dir / f"{prefix}_from-template_to-subject_warp.mif",
    }
    if out["warp"].exists() and not force:
        logger.info("FOD registration already exists for %s %s", subject, session)
        return out

    cmd: List[str] = [
        "mrregister", str(fod_file), str(template_file),
        "-nl_warp", str(out["warp"]), str(out["inverse_warp"]),
    ]
    if mask_file is not None:
        cmd += ["-mask1", str(mask_file)]
    cmd += ["-force"]
    _run(cmd, bin_dir, f"mrregister {prefix} -> template", nthreads)
    return out


def compute_fixel_metrics(
    fod_file: Path,
    warp_file: Path,
    template_fixel_mask: Path,
    output_dir: Path,
    subject: str,
    session: str,
    config: Optional[dict] = None,
    force: bool = False,
    nthreads: Optional[int] = None,
) -> Dict[str, Path]:
    """Compute FD, log(FC) and FDC for one subject in template fixel space.

    The subject FOD is warped into template space *without* reorientation, then
    reoriented explicitly, because the two operations must not be combined:
    ``mrtransform``'s modulation would otherwise alter the lobe integrals that
    FD is derived from.

    Returns
    -------
    dict
        ``fd``, ``fc``, ``log_fc``, ``fdc`` fixel data files, plus the
        subject's ``fixel_dir`` in template correspondence.
    """
    bin_dir = require_mrtrix(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_{session}"

    warped_fod = output_dir / f"{prefix}_space-template_desc-noreorient_fod.mif"
    subj_fixel = output_dir / f"{prefix}_fixel"
    corr_fixel = output_dir / f"{prefix}_fixel_corr"

    out = {
        "fd": corr_fixel / "fd.mif",
        "fc": corr_fixel / "fc.mif",
        "log_fc": corr_fixel / "log_fc.mif",
        "fdc": corr_fixel / "fdc.mif",
        "fixel_dir": corr_fixel,
    }
    if out["fdc"].exists() and not force:
        logger.info("fixel metrics already exist for %s %s", subject, session)
        return out

    _run(
        [
            "mrtransform", str(fod_file), "-warp", str(warp_file),
            "-reorient_fod", "no", str(warped_fod), "-force",
        ],
        bin_dir, f"mrtransform {prefix} -> template (no reorient)", nthreads,
    )
    # Segment the warped FOD into fixels, then reorient them to account for the
    # rotational part of the warp.
    _run(
        [
            "fod2fixel", "-mask", str(template_fixel_mask), str(warped_fod),
            str(subj_fixel), "-afd", "fd.mif", "-force",
        ],
        bin_dir, "fod2fixel (subject in template space)", nthreads,
    )
    _run(
        [
            "fixelreorient", str(subj_fixel), str(warp_file), str(subj_fixel),
            "-force",
        ],
        bin_dir, "fixelreorient", nthreads,
    )
    # Match each subject fixel to the corresponding template fixel, so every
    # subject's data indexes the same fixel set.
    _run(
        [
            "fixelcorrespondence", str(subj_fixel / "fd.mif"),
            str(template_fixel_mask), str(corr_fixel), "fd.mif", "-force",
        ],
        bin_dir, "fixelcorrespondence", nthreads,
    )
    # FC comes from the warp Jacobian perpendicular to each fixel direction.
    _run(
        [
            "warp2metric", str(warp_file), "-fc", str(template_fixel_mask),
            str(corr_fixel), "fc.mif", "-force",
        ],
        bin_dir, "warp2metric (FC)", nthreads,
    )
    # log() so FC is centred on zero and symmetric about no change; raw FC is
    # multiplicative and its distribution is skewed.
    _run(
        ["mrcalc", str(out["fc"]), "-log", str(out["log_fc"]), "-force"],
        bin_dir, "log(FC)", nthreads,
    )
    _run(
        [
            "mrcalc", str(out["fd"]), str(out["fc"]), "-mult",
            str(out["fdc"]), "-force",
        ],
        bin_dir, "FDC = FD x FC", nthreads,
    )
    return out


def run_fixel_stats(
    fixel_data_dir: Path,
    subject_files: Sequence[str],
    design_matrix: Path,
    contrast_matrix: Path,
    connectivity_matrix: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    n_permutations: int = 5000,
    force: bool = False,
    nthreads: Optional[int] = None,
) -> Path:
    """Run connectivity-based fixel enhancement statistics.

    Parameters
    ----------
    fixel_data_dir : Path
        Fixel directory holding one file per subject for the metric tested.
    subject_files : sequence of str
        Subject filenames, in the same row order as ``design_matrix``.
    design_matrix, contrast_matrix : Path
        FSL-style design and contrast files.
    connectivity_matrix : Path
        Output of ``fixelconnectivity``, defining which fixels smooth together.
    n_permutations : int
        Permutations for family-wise error control.
    """
    bin_dir = require_mrtrix(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_list = output_dir / "subject_files.txt"
    files_list.write_text("\n".join(subject_files) + "\n")

    _run(
        [
            "fixelcfestats", str(fixel_data_dir), str(files_list),
            str(design_matrix), str(contrast_matrix), str(connectivity_matrix),
            str(output_dir), "-nperms", str(n_permutations), "-force",
        ],
        bin_dir, f"fixelcfestats ({n_permutations} permutations)", nthreads,
    )
    return output_dir
