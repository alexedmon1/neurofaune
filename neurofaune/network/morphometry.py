"""Composite (FreeSurfer-aseg-style) morphometry from the propagated parcellation.

:mod:`neurofaune.network.roi_extraction` answers "what is the mean of this metric in
this region". This module answers "how much tissue is there", which needs two things
that a metric mean does not:

1. **Partial-volume weighting.** Volume is the tissue posterior integrated over a
   label set, ``sum_v P_tissue(v) * voxel_mm3``, not a voxel count. Rodent anatomicals
   are strongly anisotropic — 0.125 x 0.125 x 0.8 mm true on the cuprizone cohort,
   6.4x — so through-plane partial volume is severe and unweighted counts overstate
   cortex by roughly a third.
2. **An anatomical rollup.** SIGMA's 234 regions are grouped into composite
   structures (cortical GM, subcortical, cerebellum, brainstem, white matter,
   ventricles, ...) by a group file under ``neurofaune/atlas/groups/``.

SIGMA's own ``Matter`` column is *not* used to derive tissue volumes. It is a coarse
territory annotation that assigns the entire brainstem, pons and thalamus to "White
Matter"; the label set says only *where* to measure, and the subject's tissue
posterior says how much of what is there.

Everything is computed in **subject-native space**. In atlas space every subject
shares one mask, so per-region volumes would be identical by construction.

Voxel scaling: neurofaune stores rodent voxels scaled by ``voxel_scale`` (x10) for
tool compatibility, so volumes are divided by ``voxel_scale ** 3`` to come out in true
physical mm3.
"""

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from neurofaune.network.roi_extraction import load_parcellation

logger = logging.getLogger(__name__)

TISSUES = ("GM", "WM", "CSF")

DEFAULT_GROUPS = Path(__file__).resolve().parent.parent / "atlas" / "groups" / "sigma_invivo.yaml"

# Map the group file's canonical selector names onto SIGMA's CSV column names, which
# is what `load_parcellation` returns.
_COLUMNS = {
    "id": "Labels",
    "name": "roi_name",
    "hemisphere": "Hemisphere",
    "matter": "Matter",
    "territory": "Territories",
    "system": "System",
}


def _norm(text) -> str:
    """Collapse the LUT's inconsistent free text to a comparable key."""
    return " ".join(str(text).split())


def load_structure_groups(path: Path | None = None) -> dict:
    """Read a composite-structure definition file."""
    path = Path(path) if path else DEFAULT_GROUPS
    with open(path) as fh:
        spec = yaml.safe_load(fh) or {}
    if "structures" not in spec:
        raise KeyError(f"group file {path} has no 'structures' block")
    logger.info("Loaded %d composite structures from %s",
                len(spec["structures"]), path.name)
    return spec


def normalise_labels(labels_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the group file's text aliases on top of `load_parcellation`'s frame.

    `load_parcellation` already fixes ``Olfactive Bulb``; the group file carries the
    remaining SIGMA text inconsistencies (``Hippocampus Fomation``, a trailing space
    in ``Spinocerebellar Pathway``) so they live with the rules that depend on them.
    """
    df = labels_df.copy()
    if not spec.get("normalize_lut_text", True):
        return df
    aliases = {_norm(k): _norm(v) for k, v in (spec.get("aliases") or {}).items()}
    for column in ("Hemisphere", "Matter", "Territories", "System", "roi_name"):
        if column in df.columns:
            df[column] = df[column].map(lambda v: aliases.get(_norm(v), _norm(v)))
    return df


def resolve_labels(labels_df: pd.DataFrame, rule: dict) -> set[int]:
    """Resolve one structure's selector rule to a set of atlas label ids.

    Selectors intersect by default; ``combine: union`` ORs the territory and system
    selectors instead, which is how a structure spanning both is expressed.
    ``include_ids`` is always a union, ``exclude_*`` always a difference.
    """
    ids_col = _COLUMNS["id"]
    if rule.get("all_labels"):
        picked = set(labels_df[ids_col])
    else:
        selectors = []
        for key, canon in (("territories", "territory"), ("systems", "system"),
                           ("matter", "matter")):
            if key in rule:
                wanted = {_norm(v) for v in rule[key]}
                column = _COLUMNS[canon]
                selectors.append(set(labels_df.loc[labels_df[column].isin(wanted), ids_col]))
        if not selectors:
            picked = set()
        elif rule.get("combine") == "union":
            picked = set().union(*selectors)
        else:
            picked = set.intersection(*selectors)

    picked |= set(rule.get("include_ids", []))
    for key, canon in (("exclude_territories", "territory"), ("exclude_systems", "system")):
        if key in rule:
            wanted = {_norm(v) for v in rule[key]}
            picked -= set(labels_df.loc[labels_df[_COLUMNS[canon]].isin(wanted), ids_col])
    picked -= set(rule.get("exclude_ids", []))
    return {int(i) for i in picked}


def coarse_axis(zooms: Sequence[float]) -> int:
    """Index of the worst-sampled axis. Dilation must not grow along it."""
    return int(np.argmax(zooms[:3]))


def structure_volume(
    labels: np.ndarray,
    ids: Iterable[int],
    weight: np.ndarray | None,
    voxel_mm3: float,
    dilate: int = 0,
    plane_axis: int = 2,
) -> tuple:
    """``(volume_mm3, n_voxels)`` for one label set.

    `weight` is the tissue posterior; None gives an unweighted voxel count, which is
    the plain region volume. Dilation grows the mask **in-plane only**, never along
    `plane_axis`: one step along a 0.8 mm axis moves the boundary six times further
    than one step in-plane, which would turn a ventricle ROI into a slab.
    """
    ids = list(ids)
    mask = np.isin(labels, ids) if ids else np.zeros(labels.shape, bool)
    if dilate:
        from scipy import ndimage as ndi

        structure = np.ones((3, 3, 3), bool)
        structure[(slice(None),) * plane_axis + (slice(None, None, 2),)] = False
        mask = ndi.binary_dilation(mask, structure=structure, iterations=int(dilate))
    n_voxels = int(mask.sum())
    if weight is None:
        return n_voxels * voxel_mm3, n_voxels
    return float(weight[mask].sum()) * voxel_mm3, n_voxels


def voxel_volume_mm3(img, voxel_scale: float = 1.0) -> float:
    """True physical voxel volume, undoing neurofaune's x10 storage scaling."""
    return float(np.prod(img.header.get_zooms()[:3])) / (voxel_scale ** 3)


def compute_region_volumes(
    dseg: Path,
    posteriors: dict[str, np.ndarray] | None = None,
    voxel_scale: float = 1.0,
    labels_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-region volumes, unweighted and (when posteriors are given) per tissue.

    Long format, one row per region: ``region_id, region, hemisphere, n_voxels,
    volume_mm3[, volume_GM_mm3, volume_WM_mm3, volume_CSF_mm3]``.
    """
    img = nib.load(str(dseg))
    labels = np.asarray(img.dataobj).astype(np.int32)
    voxel_mm3 = voxel_volume_mm3(img, voxel_scale)

    names, hemis = {}, {}
    if labels_df is not None:
        names = dict(zip(labels_df["Labels"], labels_df["roi_name"], strict=False))
        hemis = dict(zip(labels_df["Labels"], labels_df["Hemisphere"], strict=False))

    ids, counts = np.unique(labels, return_counts=True)
    rows = []
    for region_id, count in zip(ids, counts, strict=True):
        region_id = int(region_id)
        if region_id == 0:
            continue
        row = {
            "region_id": region_id,
            "region": names.get(region_id, f"region_{region_id}"),
            "hemisphere": hemis.get(region_id, ""),
            "n_voxels": int(count),
            "volume_mm3": int(count) * voxel_mm3,
        }
        if posteriors:
            region = labels == region_id
            for tissue in TISSUES:
                row[f"volume_{tissue}_mm3"] = float(posteriors[tissue][region].sum()) * voxel_mm3
        rows.append(row)
    return pd.DataFrame(rows)


def compute_structure_volumes(
    dseg: Path,
    labels_df: pd.DataFrame,
    spec: dict,
    posteriors: dict[str, np.ndarray],
    voxel_scale: float = 1.0,
) -> pd.DataFrame:
    """Composite-structure volumes, one row per (structure, tissue).

    ``n_voxels_labelset`` is the size of the label SET, not of the tissue: the same
    label set is integrated once per requested tissue, so e.g. cerebellum GM and
    cerebellum WM share it.
    """
    img = nib.load(str(dseg))
    labels = np.asarray(img.dataobj).astype(np.int32)
    voxel_mm3 = voxel_volume_mm3(img, voxel_scale)
    axis = coarse_axis(img.header.get_zooms())

    label_sets = {
        name: resolve_labels(labels_df, rule) for name, rule in spec["structures"].items()
    }
    for name, ids in label_sets.items():
        if not ids:
            logger.warning("Structure '%s' resolved to no labels; check its selectors", name)

    rows = []
    for name, rule in spec["structures"].items():
        wanted = rule.get("tissue", "any")
        wanted = [wanted] if isinstance(wanted, str) else list(wanted)
        for tissue in wanted:
            weight = None if tissue == "any" else posteriors[tissue]
            volume, n_voxels = structure_volume(
                labels, label_sets[name], weight, voxel_mm3,
                int(rule.get("dilate", 0)), axis,
            )
            rows.append({
                "structure": name,
                "tissue": tissue,
                "volume_mm3": volume,
                "n_voxels_labelset": n_voxels,
            })

    df = pd.DataFrame(rows)
    total = df.loc[df["structure"] == "total_brain", "volume_mm3"].sum()
    df["pct_of_total_brain"] = 100.0 * df["volume_mm3"] / total if total else np.nan
    return df


def compute_asymmetry_index(
    regions: pd.DataFrame,
    min_volume_mm3: float = 1.0,
    measures: list[str] | None = None,
) -> pd.DataFrame:
    """Per-region-pair ``(L-R)/(0.5*(L+R))``, one row per measure.

    Emitted for the unweighted volume and each tissue because no single choice suits
    every region: weighting a fibre tract by the GM posterior builds an index out of
    near-zero numbers. ``above_floor`` flags pairs whose mean volume clears
    `min_volume_mm3` — below it the index is dominated by single-voxel differences,
    which at 0.8 mm slice thickness is a third of a small structure. Rows are kept
    rather than dropped so the filter stays visible.
    """
    df = regions[regions["hemisphere"].isin(["L", "R"])].copy()
    if df.empty:
        return pd.DataFrame()
    df["pair"] = df["region"].str.replace(r"_[LR]$", "", regex=True)

    measures = measures or (["volume_mm3"] + [f"volume_{t}_mm3" for t in TISSUES])
    out = []
    for measure in measures:
        if measure not in df.columns:
            continue
        wide = df.pivot_table(index="pair", columns="hemisphere", values=measure,
                              aggfunc="sum").dropna(subset=["L", "R"])
        if wide.empty:
            continue
        mean = 0.5 * (wide["L"] + wide["R"])
        out.append(wide.assign(
            measure=measure,
            mean_mm3=mean,
            compute_asymmetry_index=np.where(mean > 0, (wide["L"] - wide["R"]) / mean, np.nan),
            above_floor=mean >= min_volume_mm3,
        ).reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_posteriors(paths: dict[str, Path], expected_shape) -> dict[str, np.ndarray]:
    """Load GM/WM/CSF posteriors and check they sit on the parcellation's grid."""
    arrays = {}
    for tissue in TISSUES:
        if tissue not in paths:
            raise KeyError(f"missing {tissue} posterior; need all of {TISSUES}")
        arr = np.asarray(nib.load(str(paths[tissue])).dataobj, dtype=np.float32)
        if arr.shape != expected_shape:
            raise ValueError(
                f"{paths[tissue]} is {arr.shape} but the parcellation is "
                f"{expected_shape}; tissue maps must be on the subject's native grid"
            )
        arrays[tissue] = arr
    return arrays


def compute_subject_morphometry(
    dseg: Path,
    posterior_paths: dict[str, Path],
    parcellation: Path,
    labels_csv: Path,
    spec: dict | None = None,
    voxel_scale: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """All morphometry tables for one subject-session.

    Returns ``{'structures': ..., 'regions': ..., 'asymmetry': ...}``.
    """
    spec = spec or load_structure_groups()
    _, labels_df = load_parcellation(parcellation, labels_csv)
    labels_df = normalise_labels(labels_df, spec)

    shape = nib.load(str(dseg)).shape[:3]
    posteriors = load_posteriors(posterior_paths, shape)

    regions = compute_region_volumes(dseg, posteriors, voxel_scale, labels_df)
    structures = compute_structure_volumes(dseg, labels_df, spec, posteriors, voxel_scale)
    return {
        "structures": structures,
        "regions": regions,
        "asymmetry": compute_asymmetry_index(regions),
    }
