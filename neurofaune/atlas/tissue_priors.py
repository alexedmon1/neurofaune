"""Atlas-derived tissue priors for Atropos, aligned via the propagated parcellation.

Atropos segments far more reliably when it is initialised from spatial priors than
from KMeans, but getting the SIGMA tissue maps into a subject's native space is the
part that is easy to get wrong.

The obvious route — resample ``SIGMA_InVivo_{GM,WM,CSF}.nii`` onto the subject grid —
is wrong: a geometric resample uses only the two affines and so **skips the
registration entirely**. Measured on the cuprizone cohort, a geometrically resampled
WM prior reaches only 0.20 inside the corpus callosum, where it should be ~0.9.

The route taken here needs no transform chain at all. The subject already has the
atlas parcellation warped into its native space (``*_atlas-SIGMA_dseg.nii.gz``), so:

1. reduce the atlas tissue maps to a **per-label tissue composition** in atlas space,
   where labels and tissue maps share a grid (corpus callosum 0.80 WM, ventricular
   system 0.82 CSF, CA1 0.95 GM — 222 of SIGMA's 234 labels have a dominant class);
2. paint that composition onto the subject's warped ``dseg``.

The result is aligned by construction. The prior is piecewise-constant within each
region, which is exactly what a label-based prior is, and Atropos refines it against
the subject's own intensities.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ANTs numbers prior images from 1 and returns posteriors in the same order, so this
# tuple defines which class each `prior_%02d` file is.
TISSUE_ORDER = ("CSF", "GM", "WM")


def compute_label_tissue_fractions(
    atlas_labels: Path,
    prior_paths: dict[str, Path],
    tissue_order: Sequence[str] = TISSUE_ORDER,
) -> dict[int, np.ndarray]:
    """Each atlas region's mean tissue composition, normalised to sum to 1.

    Parameters
    ----------
    atlas_labels : Path
        Atlas parcellation volume, in atlas space.
    prior_paths : dict
        ``{tissue: path}`` for the atlas tissue probability maps, on the same grid
        as `atlas_labels`.
    tissue_order : sequence of str
        Order the fractions are returned in; must match the order priors are
        written to disk.

    Returns
    -------
    dict
        ``{label_id: array of fractions in `tissue_order`}``. A region the tissue
        maps say nothing about falls back to a uniform composition.
    """
    labels = np.asarray(nib.load(str(atlas_labels)).dataobj).astype(np.int32)

    priors = {}
    for tissue in tissue_order:
        if tissue not in prior_paths:
            raise KeyError(f"prior_paths is missing {tissue!r}; need {list(tissue_order)}")
        arr = np.asarray(nib.load(str(prior_paths[tissue])).dataobj, dtype=np.float32)
        if arr.shape != labels.shape:
            raise ValueError(
                f"tissue prior {prior_paths[tissue]} is {arr.shape} but the atlas "
                f"parcellation is {labels.shape}; both must be in the same atlas space"
            )
        priors[tissue] = arr

    fractions: dict[int, np.ndarray] = {}
    ids = np.unique(labels)
    n = len(tissue_order)
    for label_id in ids[ids > 0]:
        region = labels == label_id
        values = np.array(
            [float(priors[t][region].mean()) for t in tissue_order], dtype=np.float32
        )
        total = values.sum()
        fractions[int(label_id)] = (
            values / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float32)
        )

    dominant = sum(1 for f in fractions.values() if f.max() > 0.5)
    logger.info(
        "Atlas tissue composition: %d regions, %d with a dominant tissue",
        len(fractions), dominant,
    )
    return fractions


def paint_priors(
    dseg_data: np.ndarray,
    fractions: dict[int, np.ndarray],
    n_tissues: int = len(TISSUE_ORDER),
) -> list[np.ndarray]:
    """Native-space prior per tissue, painted onto the warped parcellation.

    Aligned by construction: `dseg_data` is the atlas already carried into subject
    space by the upstream registration, so no further transform is applied.
    """
    stack = [np.zeros(dseg_data.shape, np.float32) for _ in range(n_tissues)]
    for label_id, fraction in fractions.items():
        region = dseg_data == label_id
        if not region.any():
            continue
        for k in range(n_tissues):
            stack[k][region] = fraction[k]
    return stack


def build_native_priors(
    dseg: Path,
    fractions: dict[int, np.ndarray],
    output_dir: Path,
    tissue_order: Sequence[str] = TISSUE_ORDER,
) -> str:
    """Write ``prior_01..prior_0N`` next to each other and return the ANTs pattern.

    Atropos takes priors as a printf-style path (``prior_%02d.nii.gz``) and returns
    posteriors in the same order, so the file numbering *is* the class mapping.
    """
    dseg_img = nib.load(str(dseg))
    dseg_data = np.asarray(dseg_img.dataobj).astype(np.int32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (tissue, arr) in enumerate(
        zip(tissue_order, paint_priors(dseg_data, fractions, len(tissue_order)),
            strict=True), start=1
    ):
        path = output_dir / f"prior_{index:02d}.nii.gz"
        nib.save(nib.Nifti1Image(arr, dseg_img.affine, dseg_img.header), str(path))
        logger.debug("Wrote %s prior -> %s", tissue, path.name)

    logger.info(
        "Painted %d atlas-derived priors (%s) onto %s",
        len(tissue_order), ", ".join(tissue_order), Path(dseg).name,
    )
    return str(output_dir / "prior_%02d.nii.gz")


def build_atlas_extent_mask(dseg: Path, output_path: Path) -> Path:
    """Binary mask of the labelled brain, for use instead of the brain mask.

    A skull-strip mask is deliberately generous, and the surplus is scored as CSF:
    on the cuprizone cohort it covers 2774 mm3 against 2156 mm3 of labelled brain,
    and that ~600 mm3 rim inflates every CSF-derived number downstream. Bounding
    the segmentation by the parcellation removes it.
    """
    img = nib.load(str(dseg))
    mask = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(output_path))
    logger.info("Atlas-extent mask: %d voxels -> %s", int(mask.sum()), output_path.name)
    return output_path
