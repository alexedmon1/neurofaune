"""Five-tissue-type (5TT) image construction for anatomically-constrained tractography.

MRtrix's ``5ttgen fsl`` pipeline runs FSL FAST and FIRST, both of which carry
human anatomical priors — FIRST's subcortical shape models in particular have
no rodent equivalent — so it cannot be used here. Instead this module assembles
the 5TT directly from the tissue posteriors neurofaune's anatomical workflow
already produces with Atropos (``label-{GM,WM,CSF}_probseg.nii.gz``), which are
derived from the study's own data and carry no cross-species assumption.

The cortical/subcortical grey matter split that ACT requires is recovered from
the SIGMA atlas already propagated into subject space
(``atlas-SIGMA_dseg.nii.gz``), using the atlas' ``Territories`` column. That
split matters for ACT: streamlines are permitted to terminate in subcortical
grey matter but are required to reach a GM boundary elsewhere, so collapsing
the two causes systematic premature termination in deep structures.

MRtrix 5TT volume order (fixed by the format):

===  ==========================
  0  cortical grey matter
  1  sub-cortical grey matter
  2  white matter
  3  CSF
  4  pathological tissue
===  ==========================
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# SIGMA ``Territories`` values that are subcortical grey matter. Cortex and the
# olfactory bulb are cortical; fibre tracts and CSF are handled by their own
# tissue classes and never contribute grey matter.
SUBCORTICAL_TERRITORIES = {
    "Diencephalon",
    "Basal ganglia",
    "Mesencephalon",
    "Brainstem",
    "Cerebellum",
    "Forebrain",
}

CORTICAL_TERRITORIES = {
    "Cortex",
    "Olfactive Bulb",   # SIGMA ships both spellings
    "Olfactory Bulb",
}


def _subcortical_mask_from_atlas(
    atlas_dseg: Path,
    atlas_labels_csv: Path,
) -> np.ndarray:
    """Boolean mask of subcortical grey-matter labels in atlas (dseg) space."""
    import pandas as pd

    labels = pd.read_csv(atlas_labels_csv)
    grey = labels["Matter"].astype(str).str.strip().str.lower() == "grey matter"
    territory = labels["Territories"].astype(str).str.strip()
    subcortical = grey & territory.isin(SUBCORTICAL_TERRITORIES)

    label_ids = (
        labels.loc[subcortical, "Labels"]
        .astype(int)
        .unique()
        .tolist()
    )
    logger.info("  %d SIGMA labels classed as subcortical grey matter", len(label_ids))

    dseg = nib.load(str(atlas_dseg)).get_fdata().astype(np.int32)
    return np.isin(dseg, label_ids)


def build_5tt_from_probseg(
    gm_probseg: Path,
    wm_probseg: Path,
    csf_probseg: Path,
    output_file: Path,
    atlas_dseg: Optional[Path] = None,
    atlas_labels_csv: Optional[Path] = None,
    brain_mask: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Assemble an MRtrix-compatible 5TT image from Atropos tissue posteriors.

    Parameters
    ----------
    gm_probseg, wm_probseg, csf_probseg : Path
        Tissue probability maps from neurofaune's anatomical workflow. All
        three must share a grid.
    output_file : Path
        Destination ``.nii.gz``. A 4-D image with 5 volumes.
    atlas_dseg : Path, optional
        SIGMA parcellation resampled into the same space as the posteriors.
        When given (with ``atlas_labels_csv``), grey matter is split into
        cortical and subcortical. When omitted, all grey matter is placed in
        the cortical volume, which is valid but makes ACT more likely to
        truncate streamlines in deep structures.
    atlas_labels_csv : Path, optional
        SIGMA labels CSV supplying the ``Matter`` and ``Territories`` columns.
    brain_mask : Path, optional
        Restricts the 5TT to the brain; voxels outside are set to zero, as the
        format requires.
    force : bool
        Overwrite an existing output.

    Returns
    -------
    Path
        ``output_file``.

    Notes
    -----
    The five volumes are normalised to sum to 1 wherever their sum is
    positive, and to 0 elsewhere. ``5ttcheck`` enforces this, and ACT
    silently misbehaves without it.
    """
    if output_file.exists() and not force:
        logger.info("5TT already exists: %s", output_file)
        return output_file

    logger.info("Building 5TT from Atropos posteriors")
    gm_img = nib.load(str(gm_probseg))
    gm = gm_img.get_fdata().astype(np.float32)
    wm = nib.load(str(wm_probseg)).get_fdata().astype(np.float32)
    csf = nib.load(str(csf_probseg)).get_fdata().astype(np.float32)

    if not (gm.shape == wm.shape == csf.shape):
        raise ValueError(
            f"tissue posteriors differ in shape: GM {gm.shape}, WM {wm.shape}, "
            f"CSF {csf.shape}"
        )

    gm_cortical = gm
    gm_subcortical = np.zeros_like(gm)

    if atlas_dseg is not None and atlas_labels_csv is not None:
        sub_mask = _subcortical_mask_from_atlas(atlas_dseg, atlas_labels_csv)
        if sub_mask.shape != gm.shape:
            raise ValueError(
                f"atlas dseg shape {sub_mask.shape} does not match tissue "
                f"posteriors {gm.shape}; resample the atlas into the posterior "
                "space first"
            )
        gm_subcortical = np.where(sub_mask, gm, 0.0).astype(np.float32)
        gm_cortical = np.where(sub_mask, 0.0, gm).astype(np.float32)
        logger.info(
            "  grey matter split: %.1f%% subcortical by volume",
            100.0 * gm_subcortical.sum() / max(gm.sum(), 1e-9),
        )
    else:
        logger.warning(
            "  no atlas supplied: all grey matter assigned to the cortical "
            "volume (ACT may truncate streamlines in deep structures)"
        )

    pathology = np.zeros_like(gm)
    stack = np.stack([gm_cortical, gm_subcortical, wm, csf, pathology], axis=-1)

    if brain_mask is not None:
        mask = nib.load(str(brain_mask)).get_fdata() > 0
        if mask.shape != gm.shape:
            raise ValueError(
                f"brain mask shape {mask.shape} does not match posteriors {gm.shape}"
            )
        stack[~mask] = 0.0

    # The format requires the tissue fractions to be a partition of unity
    # inside the brain and exactly zero outside it.
    total = stack.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        stack = np.where(total > 0, stack / total, 0.0).astype(np.float32)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(stack, gm_img.affine, gm_img.header), str(output_file))
    logger.info("  wrote %s %s", output_file.name, stack.shape)
    return output_file


def check_5tt(fivett_file: Path, bin_dir: Path) -> bool:
    """Validate a 5TT image with ``5ttcheck``. Returns True when clean."""
    proc = subprocess.run(
        [str(bin_dir / "5ttcheck"), str(fivett_file)],
        capture_output=True, text=True,
    )
    output = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0 and "WARNING" not in output.upper()
    if ok:
        logger.info("  5ttcheck: valid")
    else:
        logger.warning("  5ttcheck reported issues:\n%s", output[-2000:])
    return ok


def warp_5tt(
    fivett_file: Path,
    reference: Path,
    transforms: Sequence[Path],
    output_file: Path,
    invert: Optional[Sequence[bool]] = None,
    force: bool = False,
) -> Path:
    """Resample a 5TT image into another space with ANTs, then renormalise.

    Interpolation breaks the partition-of-unity constraint, so the tissue
    fractions are renormalised after resampling — otherwise ``5ttcheck`` fails
    and ACT's termination logic operates on fractions that do not sum to 1.

    Parameters
    ----------
    fivett_file : Path
        5TT image to resample.
    reference : Path
        Image defining the target grid (e.g. the DWI b0 or FA map).
    transforms : sequence of Path
        ANTs transforms, in the order ``antsApplyTransforms`` expects
        (last applied first).
    output_file : Path
        Destination.
    invert : sequence of bool, optional
        Per-transform inversion flags, aligned with ``transforms``.
    force : bool
        Overwrite an existing output.
    """
    if output_file.exists() and not force:
        logger.info("warped 5TT already exists: %s", output_file)
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)
    invert = list(invert) if invert is not None else [False] * len(transforms)
    if len(invert) != len(transforms):
        raise ValueError("invert flags must align with transforms")

    cmd: List[str] = [
        "antsApplyTransforms", "-d", "3", "-e", "3",  # -e 3: 4-D, per-volume
        "-i", str(fivett_file), "-r", str(reference),
        "-o", str(output_file), "-n", "Linear",
    ]
    for tf, inv in zip(transforms, invert):
        cmd += ["-t", f"[{tf},1]" if inv else str(tf)]

    logger.info("  antsApplyTransforms -> %s", output_file.name)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed for 5TT warp\n"
            f"command: {' '.join(cmd)}\nstderr:\n{proc.stderr[-3000:]}"
        )

    img = nib.load(str(output_file))
    data = np.clip(img.get_fdata().astype(np.float32), 0.0, None)
    total = data.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        data = np.where(total > 0, data / total, 0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(output_file))
    return output_file
