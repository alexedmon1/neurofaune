"""Cross-sectional & longitudinal registration-consistency QC.

Group-level QC that answers "do the subjects' brains land in the same place on the
atlas?" by warping each subject's brain mask into the common atlas (SIGMA) space and
measuring spatial overlap (Dice). It complements the *per-registration* metrics in
:mod:`neurofaune.templates.registration_qc` (correlation-after, single-pair Dice),
which cannot see batch/session effects because they look at one scan at a time.

Two measures:

* **Cross-sectional Dice** — *within a timepoint*. All subjects at a timepoint share
  one study template and one template→SIGMA registration, so their warped brains are
  directly comparable. Reported as (a) each subject vs the SIGMA brain, and (b) mean
  pairwise Dice across subjects. Low/scattered values flag unstable registration or
  inconsistent brain extraction at that timepoint.

* **Longitudinal Dice** — *within an animal, across its timepoints*. For each animal,
  Dice between its own brain at timepoint A vs B (both in SIGMA space). This is the
  prerequisite for trusting longitudinal per-region readouts: if an animal's brain
  does not occupy the same atlas location over time, regional trajectories are not
  measuring the same tissue.

Both metrics live in the common SIGMA space so that different per-timepoint templates
are compared on equal footing (which is why the comparison is *not* done in native
subject space).

Layout assumed (as written by ``scripts/batch_preprocess_anat.py``)::

    <transforms_dir>/<subject>/<session>/<subject>_<session>_T2w_to_template_{0GenericAffine.mat,1Warp.nii.gz}
    <templates_dir>/anat/<cohort>/transforms/tpl-to-SIGMA_{0GenericAffine.mat,1Warp.nii.gz}
    <derivatives_dir>/<subject>/<session>/anat/<subject>_<session>_desc-brain_mask.nii.gz

where ``cohort`` is the session label without the ``ses-`` prefix (e.g. ``ses-1`` -> ``1``).
"""
from __future__ import annotations

import itertools
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

from neurofaune.templates.registration_qc import compute_dice_coefficient
from neurofaune.templates.sigma_warp import resolve_tpl_to_sigma_for_cohort

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Warping subject → common atlas space
# --------------------------------------------------------------------------- #
def warp_brain_to_atlas(
    brain_mask: Path,
    to_template_affine: Path,
    tpl_to_sigma_affine: Path,
    atlas_reference: Path,
    output_file: Path,
    to_template_warp: Optional[Path] = None,
    tpl_to_sigma_warp: Optional[Path] = None,
    interpolation: str = "NearestNeighbor",
) -> Path:
    """Warp a subject brain mask into atlas (SIGMA) space via subject→template→SIGMA.

    ANTs applies transforms in reverse of the list order, so the chain is listed
    SIGMA-first (template→SIGMA, then subject→template) — matching
    :func:`neurofaune.analysis.tbss.prepare_tbss.warp_metric_to_sigma`.
    """
    transforms: List[str] = []
    if tpl_to_sigma_warp:
        transforms.append(str(tpl_to_sigma_warp))
    transforms.append(str(tpl_to_sigma_affine))
    if to_template_warp:
        transforms.append(str(to_template_warp))
    transforms.append(str(to_template_affine))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "antsApplyTransforms", "-d", "3",
        "-i", str(brain_mask), "-r", str(atlas_reference),
        "-o", str(output_file), "-n", interpolation,
    ]
    for t in transforms:
        cmd += ["-t", t]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"antsApplyTransforms failed for {brain_mask.name}: {result.stderr}")
    return output_file


def _load_bool(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0


# --------------------------------------------------------------------------- #
# The two measures (pure — operate on already-warped masks)
# --------------------------------------------------------------------------- #
def cross_sectional_dice(
    warped_by_subject: Dict[str, Path],
    atlas_mask: Optional[Path] = None,
) -> Dict[str, object]:
    """Within one timepoint: agreement of subjects' warped brains.

    Parameters
    ----------
    warped_by_subject
        ``{subject_id: path to that subject's brain warped into atlas space}``.
    atlas_mask
        Optional atlas brain mask (same grid) for the subject-vs-atlas fit.

    Returns
    -------
    dict with keys:
        ``pairwise``   : list of ``(subject_a, subject_b, dice)``
        ``vs_atlas``   : ``{subject: dice}`` vs the atlas brain (empty if no atlas_mask)
        ``mean_pairwise``, ``std_pairwise``, ``mean_vs_atlas``, ``std_vs_atlas``, ``n_subjects``
    """
    masks = {s: _load_bool(p) for s, p in warped_by_subject.items()}
    pairwise: List[Tuple[str, str, float]] = [
        (a, b, float(compute_dice_coefficient(masks[a], masks[b])))
        for a, b in itertools.combinations(sorted(masks), 2)
    ]
    vs_atlas: Dict[str, float] = {}
    if atlas_mask is not None:
        am = _load_bool(atlas_mask)
        vs_atlas = {s: float(compute_dice_coefficient(m, am)) for s, m in masks.items()}

    pw = np.array([d for *_, d in pairwise], dtype=float)
    va = np.array(list(vs_atlas.values()), dtype=float)
    _mean = lambda x: float(x.mean()) if x.size else float("nan")
    _std = lambda x: float(x.std()) if x.size else float("nan")
    return {
        "pairwise": pairwise,
        "vs_atlas": vs_atlas,
        "n_subjects": len(masks),
        "mean_pairwise": _mean(pw),
        "std_pairwise": _std(pw),
        "mean_vs_atlas": _mean(va),
        "std_vs_atlas": _std(va),
    }


def longitudinal_dice(warped_by_session: Dict[str, Path]) -> List[Tuple[str, str, float]]:
    """Within one animal: Dice between its brain at each pair of timepoints.

    Parameters
    ----------
    warped_by_session
        ``{session_label: path to that session's brain warped into atlas space}``.

    Returns
    -------
    list of ``(session_a, session_b, dice)`` for every timepoint pair (sorted).
    """
    masks = {s: _load_bool(p) for s, p in warped_by_session.items()}
    return [
        (a, b, float(compute_dice_coefficient(masks[a], masks[b])))
        for a, b in itertools.combinations(sorted(masks), 2)
    ]


# --------------------------------------------------------------------------- #
# Driver: discover transforms, warp everything, compute both measures
# --------------------------------------------------------------------------- #
def _cohort(session: str) -> str:
    return session.replace("ses-", "")


def run_consistency_qc(
    derivatives_dir: Path,
    transforms_dir: Path,
    templates_dir: Path,
    atlas_mask: Path,
    output_dir: Path,
    brain_mask_suffix: str = "desc-brain_mask",
    interpolation: str = "NearestNeighbor",
) -> Dict[str, "object"]:
    """Warp every registered subject brain into atlas space and compute the two measures.

    Returns a dict with ``cross_sectional`` (per-timepoint summaries), ``longitudinal``
    (per-animal pair Dice), and the paths of the CSVs written under ``output_dir``.
    Requires ``antsApplyTransforms`` on PATH and pandas.
    """
    import pandas as pd

    derivatives_dir, transforms_dir = Path(derivatives_dir), Path(transforms_dir)
    templates_dir, output_dir = Path(templates_dir), Path(output_dir)
    warped_dir = output_dir / "brains_in_atlas"
    warped_dir.mkdir(parents=True, exist_ok=True)

    # 1. discover (subject, session) with both a brain mask and a subject→template transform
    warped: Dict[Tuple[str, str], Path] = {}
    for mask in sorted(derivatives_dir.glob(f"sub-*/ses-*/anat/*_{brain_mask_suffix}.nii.gz")):
        subject, session = mask.parts[-4], mask.parts[-3]
        coh = _cohort(session)
        subtx = transforms_dir / subject / session
        aff1 = subtx / f"{subject}_{session}_T2w_to_template_0GenericAffine.mat"
        warp1 = subtx / f"{subject}_{session}_T2w_to_template_1Warp.nii.gz"
        tpl_to_sigma = resolve_tpl_to_sigma_for_cohort(templates_dir, coh)
        aff2 = tpl_to_sigma["affine"]
        warp2 = tpl_to_sigma["warp"]
        if not (aff1.exists() and aff2 is not None):
            logger.warning("missing transforms for %s %s — skipped", subject, session)
            continue
        out = warped_dir / f"{subject}_{session}_brain_in_atlas.nii.gz"
        try:
            warp_brain_to_atlas(
                mask, aff1, aff2, atlas_mask, out,
                to_template_warp=warp1 if warp1.exists() else None,
                tpl_to_sigma_warp=warp2,
                interpolation=interpolation,
            )
            warped[(subject, session)] = out
        except RuntimeError as e:
            logger.error("warp failed for %s %s: %s", subject, session, e)

    logger.info("warped %d subject-sessions into atlas space", len(warped))

    # 2. cross-sectional: within each timepoint
    cs_rows, cs_pairs = [], []
    by_session: Dict[str, Dict[str, Path]] = {}
    for (sub, ses), p in warped.items():
        by_session.setdefault(ses, {})[sub] = p
    for ses, subs in sorted(by_session.items()):
        res = cross_sectional_dice(subs, atlas_mask=atlas_mask)
        cs_rows.append({
            "session": ses, "n_subjects": res["n_subjects"],
            "mean_pairwise_dice": round(res["mean_pairwise"], 4),
            "std_pairwise_dice": round(res["std_pairwise"], 4),
            "mean_dice_vs_atlas": round(res["mean_vs_atlas"], 4),
            "std_dice_vs_atlas": round(res["std_vs_atlas"], 4),
        })
        for s, d in res["vs_atlas"].items():
            cs_pairs.append({"session": ses, "subject": s, "dice_vs_atlas": round(d, 4)})

    # 3. longitudinal: within each animal across its timepoints
    by_animal: Dict[str, Dict[str, Path]] = {}
    for (sub, ses), p in warped.items():
        by_animal.setdefault(sub, {})[ses] = p
    long_rows = []
    for sub, sess in sorted(by_animal.items()):
        for a, b, d in longitudinal_dice(sess):
            long_rows.append({"animal": sub, "pair": f"{a}_vs_{b}", "dice": round(d, 4)})

    # 4. write CSVs
    cs_df = pd.DataFrame(cs_rows)
    cs_subj_df = pd.DataFrame(cs_pairs)
    long_df = pd.DataFrame(long_rows)
    paths = {
        "cross_sectional_summary": output_dir / "cross_sectional_dice_summary.csv",
        "cross_sectional_subjects": output_dir / "cross_sectional_dice_per_subject.csv",
        "longitudinal": output_dir / "longitudinal_dice.csv",
    }
    cs_df.to_csv(paths["cross_sectional_summary"], index=False)
    cs_subj_df.to_csv(paths["cross_sectional_subjects"], index=False)
    long_df.to_csv(paths["longitudinal"], index=False)
    logger.info("wrote consistency-QC CSVs to %s", output_dir)

    return {
        "cross_sectional": cs_df,
        "longitudinal_by_pair": long_df.groupby("pair").dice.agg(["count", "mean", "std"]) if long_rows else None,
        "n_warped": len(warped),
        "paths": paths,
    }
