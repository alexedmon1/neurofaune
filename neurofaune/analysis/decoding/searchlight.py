"""Searchlight and whole-brain decoding over atlas-space maps, on the decoding harness's rules.

Moved from murinet (``searchlight.py``, origin/main ade3d38). Three entry points:

- :func:`run_searchlight` slides a sphere through the brain and, at each centre,
  cross-validates a classifier on the voxels inside it. The result is a map of *where*
  group membership is locally decodable, which ROI averages cannot give: averaging over a
  region destroys the pattern within it.
- :func:`run_whole_brain` decodes from the whole search volume at once and projects the
  model's weights back to voxels. It asks whether the pattern is decodable at all.
- :func:`coverage_report` says what a combination of maps leaves of each structure, before
  hours go into a searchlight over a volume that turns out to be a slab.

**Several maps make one feature vector, and the search volume shrinks to match.** A
sphere's features are its voxels across every map in ``searchlight.maps``, so
``[MWF, FA, RD]`` asks whether the joint local signature decodes. The maps do not share a
field of view (on the cuprizone cohort, inside the atlas brain: T2w 99.9%, resting state
90.3%, DWI 73.6%, MSME-derived MWF 50.1%, acquired as a slab), and a feature vector means
the same thing for every animal only where every map has data in every animal. So coverage
is intersected over the scans each contrast uses before a single sphere is placed, and the
cost is logged. Averaging over missing data would manufacture a signal; it is refused.

**The harness's rules hold here** (:mod:`neurofaune.analysis.decoding.harness`): folds are
grouped by animal, labels are permuted between animals, and everything fitted is fitted
inside the fold. In murinet each sphere was z-scored once over all animals, on the grounds
that scaling is label-blind; here it is scaled on each fold's training animals. The kernel
trick survives: each fold's Gram matrix depends on the features and the fold, not on the
labels, so it is built once per sphere and fold and reused by every relabelling.

**Family-wise error by the maximum statistic.** For each permutation the best score
anywhere in the search volume is kept; a voxel's FWE p is the share of those maxima at or
above its score. This needs every voxel to see the same relabellings, so they are drawn once
and shared.

Geometry: every map must be on the atlas grid. A map on another grid is resampled onto it
(linear) and the substitution is logged. Radii are in true mm: header zooms are divided by
``bids.voxel_scale`` first, because rodent voxels are stored x10.

Scope is ``searchlight.mask``: ``whole_brain`` (exploratory), or a structure of the group
file (``fiber_tracts`` is the confirmatory scope for a demyelination model). Corpus
callosum recovery is the positive control in cuprizone: if it does not light up, distrust
the run before interpreting anything else.

Settings are under ``searchlight:`` and not in ``default.yaml``, whose contents go into the
provenance digest. ``contrasts`` (a list of ``{name, session_a, session_b}``) and ``maps``
are required.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from neurofaune.analysis.decoding.features import _session_id, session_table
from neurofaune.analysis.decoding.harness import RANDOM_STATE, _animal_shuffles, _splitter
from neurofaune.analysis.stats.readout_results import now
from neurofaune.config import get_config_value
from neurofaune.study_design import StudyDesign, load_study_design

logger = logging.getLogger(__name__)

#: Derivative folders and the modality the results specification calls them.
_MODALITY = {"anat": "anat", "dwi": "dwi", "func": "func", "msme": "msme", "mrs": "mrs"}


@dataclass(frozen=True)
class SearchlightSettings:
    """What to decode and how; ``from_config`` reads them from ``searchlight:``."""

    maps: tuple[str, ...] = ()
    mask: str = "whole_brain"
    radius_mm: float = 0.45
    estimator: str = "svm"
    C: float = 1.0
    alpha: float = 1.0
    cv_folds: Any = 5
    permutations: int = 200
    workers: int = 8
    min_coverage_fraction: float = 1.0
    reduce_4d: str | None = None
    space: str = "SIGMA"
    pca_components: int = 20
    fwe_alpha: float = 0.05
    voxel_scale: float = 10.0

    @classmethod
    def from_config(cls, config: dict) -> SearchlightSettings:
        maps = get_config_value(config, "searchlight.maps", default=None) or ()
        return cls(
            maps=tuple(maps),
            mask=get_config_value(config, "searchlight.mask", default=cls.mask),
            radius_mm=float(get_config_value(config, "searchlight.radius_mm", default=cls.radius_mm)),
            estimator=get_config_value(config, "searchlight.estimator", default=cls.estimator),
            C=float(get_config_value(config, "searchlight.C", default=cls.C)),
            alpha=float(get_config_value(config, "searchlight.alpha", default=cls.alpha)),
            cv_folds=get_config_value(config, "searchlight.cv_folds", default=cls.cv_folds),
            permutations=int(
                get_config_value(config, "searchlight.permutations", default=cls.permutations)
            ),
            workers=int(get_config_value(config, "searchlight.workers", default=cls.workers)),
            min_coverage_fraction=float(get_config_value(
                config, "searchlight.min_coverage_fraction", default=cls.min_coverage_fraction
            )),
            reduce_4d=get_config_value(config, "searchlight.reduce_4d", default=None),
            space=get_config_value(config, "searchlight.space", default=cls.space),
            pca_components=int(
                get_config_value(config, "searchlight.pca_components", default=cls.pca_components)
            ),
            fwe_alpha=float(get_config_value(config, "searchlight.fwe_alpha", default=cls.fwe_alpha)),
            voxel_scale=float(
                get_config_value(config, "bids.voxel_scale", default=cls.voxel_scale) or 10.0
            ),
        )


# --- inputs ------------------------------------------------------------------

_ENTITY = re.compile(r"(?P<key>[a-zA-Z]+)-(?P<value>[a-zA-Z0-9.]+)")


@dataclass(frozen=True)
class MapFile:
    """One atlas-space derivative: whose it is, which map, and its modality folder."""

    path: Path
    subject: str
    name: str
    modality: str


def _parse(path: Path) -> MapFile | None:
    stem = path.name.split(".nii")[0]
    parts = stem.split("_")
    suffix = parts[-1] if "-" not in parts[-1] else ""
    ents = {m["key"]: m["value"] for p in parts for m in [_ENTITY.fullmatch(p)] if m}
    if "sub" not in ents or "ses" not in ents or not suffix:
        return None
    desc = ents.get("desc")
    if suffix == "mask" or desc == "brain":
        return None
    # func puts the metric in desc (desc-fALFF_bold); elsewhere the suffix is the metric and a
    # desc qualifies it (desc-syn_FA is a variant of FA, not a map called "syn")
    if suffix == "bold":
        if not desc:
            return None
        name = desc
    else:
        name = suffix if not desc else f"{suffix}-{desc}"
    return MapFile(path, f"sub-{ents['sub']}_ses-{ents['ses']}", name,
                   _MODALITY.get(path.parent.name, path.parent.name))


def atlas_space_maps(derivatives: Path, space: str = "SIGMA") -> dict[str, dict[str, MapFile]]:
    """``{"sub-X_ses-Y": {map name: MapFile}}`` for every ``space-<space>`` image.

    The map name is the BIDS suffix (``FA``, ``MWF``), the ``desc`` for functional metrics
    (``desc-fALFF_bold`` -> ``fALFF``), and ``<suffix>-<desc>`` for a variant
    (``desc-syn_FA`` -> ``FA-syn``). Masks are skipped. Two files giving one session the same
    map name are refused when that map is asked for, not silently resolved.
    """
    out: dict[str, dict[str, MapFile]] = {}
    clashes: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(Path(derivatives).glob(f"sub-*/ses-*/*/*_space-{space}_*.nii*")):
        entry = _parse(path)
        if entry is None:
            continue
        here = out.setdefault(entry.subject, {})
        if entry.name in here:
            clashes.setdefault((entry.subject, entry.name), [here[entry.name].path]).append(path)
        here[entry.name] = entry
    for (subject, name), paths in clashes.items():
        out[subject][name] = MapFile(Path(), subject, name, "")
        logger.warning("%s: %d files are map %r: %s", subject, len(paths), name,
                       ", ".join(p.name for p in paths))
    return out


@dataclass
class AtlasInputs:
    """The atlas the maps are in: labels on their grid, region names, structure groups."""

    image: nib.Nifti1Image
    labels: np.ndarray
    names: Any = None  # neurofaune.analysis.stats.readout.Atlas
    structures: dict[str, set[int]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> AtlasInputs:
        """The parcellation (``searchlight.parcellation``, default the study-space atlas's),
        its ``<name>_Labels.csv`` beside it, and the structure group file
        (``searchlight.structure_groups``, default SIGMA InVivo's)."""
        from neurofaune.analysis.stats.readout import Atlas
        from neurofaune.network.morphometry import (
            load_structure_groups,
            normalise_labels,
            resolve_labels,
        )
        from neurofaune.network.roi_extraction import load_parcellation

        parcellation = get_config_value(config, "searchlight.parcellation", default=None) \
            or get_config_value(config, "atlas.study_space.parcellation", default=None)
        if not parcellation:
            raise KeyError("set searchlight.parcellation or atlas.study_space.parcellation")
        parcellation = Path(parcellation)
        image = nib.load(str(parcellation))
        labels = np.asarray(image.dataobj).astype(np.int32)
        stem = parcellation.name.replace(".nii.gz", "").replace(".nii", "")
        table = Path(get_config_value(config, "searchlight.labels", default=None)
                     or parcellation.with_name(f"{stem}_Labels.csv"))
        if not table.exists():
            logger.warning("No label table %s: regions are label numbers, and only "
                           "searchlight.mask: whole_brain is available", table)
            return cls(image, labels, Atlas(labels))
        _, labels_df = load_parcellation(parcellation, table)
        groups_file = get_config_value(config, "searchlight.structure_groups", default=None)
        spec = load_structure_groups(Path(groups_file) if groups_file else None)
        labels_df = normalise_labels(labels_df, spec)
        structures = {name: resolve_labels(labels_df, rule)
                      for name, rule in spec["structures"].items()}
        return cls(image, labels, Atlas.from_files(parcellation, table), structures)

    def search_mask(self, name: str) -> np.ndarray:
        """Voxels to centre spheres on: every labelled voxel, or one structure's."""
        if name == "whole_brain":
            return self.labels > 0
        if name not in self.structures:
            raise KeyError(f"searchlight.mask {name!r} is not whole_brain or a structure of the "
                           f"group file ({', '.join(sorted(self.structures)) or 'none loaded'})")
        return np.isin(self.labels, sorted(self.structures[name]))


def _as_3d(img, path, reduce_4d: str | None) -> np.ndarray:
    """A 3D array from a possibly 4D image; 4D is refused unless ``reduce_4d`` opts in.

    The resting-state metrics (fALFF, ReHo, ALFF) are the 3D summaries meant for this;
    taking one volume of a BOLD run, or averaging it, is a different measurement.
    """
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim <= 3:
        return arr
    if reduce_4d == "mean":
        return np.nanmean(arr, axis=-1)
    if reduce_4d == "first":
        return arr[..., 0]
    raise ValueError(
        f"{path} is 4D {arr.shape}. Use a 3D derived metric (fALFF / ReHo / ALFF) "
        "instead, or set searchlight.reduce_4d to 'mean' or 'first' to opt in."
    )


def _on_grid(img, path, ref: nib.Nifti1Image, reduce_4d: str | None) -> np.ndarray:
    """The image as a 3D array on the atlas grid, resampled (and logged) when it is not."""
    from scipy.ndimage import affine_transform

    arr = _as_3d(img, path, reduce_4d)
    if arr.shape == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3):
        return arr
    transform = np.linalg.inv(img.affine) @ ref.affine
    logger.warning("resampling %s from %s onto the atlas grid %s (linear)",
                   Path(path).name, arr.shape, ref.shape[:3])
    return affine_transform(arr, transform[:3, :3], offset=transform[:3, 3],
                            output_shape=ref.shape[:3], order=1,
                            mode="constant", cval=0.0).astype(np.float32)


def load_maps(files: Mapping[str, Mapping[str, MapFile]], subjects: Sequence[str],
              maps: Sequence[str], mask: np.ndarray, ref: nib.Nifti1Image,
              reduce_4d: str | None = None) -> np.ndarray:
    """``(n_subjects, n_voxels, n_maps)`` float32 over the ``mask`` voxels."""
    out = np.full((len(subjects), int(mask.sum()), len(maps)), np.nan, dtype=np.float32)
    for i, subject in enumerate(subjects):
        entries = files.get(subject, {})
        for j, name in enumerate(maps):
            entry = entries.get(name)
            if entry is None:
                raise FileNotFoundError(f"{subject} has no atlas-space map {name!r}")
            if not entry.modality:
                raise ValueError(f"{subject}: more than one file is map {name!r}; see the log")
            out[i, :, j] = _on_grid(nib.load(str(entry.path)), entry.path, ref, reduce_4d)[mask]
    return out


def covered(X: np.ndarray, min_fraction: float) -> np.ndarray:
    """``(n_voxels, n_maps)``: where each map has data in at least ``min_fraction`` of rows.

    Data is finite and non-zero, the convention neurofaune writes outside a modality's field
    of view. The default 1.0 demands every scan: only then does a sphere's feature vector
    mean the same thing for each animal.
    """
    return (np.isfinite(X) & (X != 0)).mean(axis=0) >= min_fraction


# --- the design --------------------------------------------------------------

def design_scans(sessions: pd.DataFrame, design: str, session_a: str, session_b: str,
                 target: str = "group") -> tuple[list[str], list[str] | None, pd.Series]:
    """``(scans of session_b, scans of session_a or None, target per animal)``.

    ``delta`` pairs each animal's two sessions (its row is b - a); ``cross_sectional`` takes
    session_b alone. The target is indexed by animal, in the order of the scans: one row per
    animal, so the folds and the permutations are both between animals.
    """
    if target not in sessions.columns:
        raise KeyError(f"target {target!r} is not a column of the study design")
    a, b = _session_id(session_a), _session_id(session_b)
    by = {s: sessions[sessions["session"] == s].drop_duplicates("animal").set_index("animal")
          for s in (a, b)}
    if design == "cross_sectional":
        if by[b].empty:
            raise ValueError(f"no scans of {b}")
        animals = sorted(by[b].index)
        return by[b].loc[animals, "subject"].tolist(), None, by[b].loc[animals, target]
    if design != "delta":
        raise ValueError(f"design must be 'delta' or 'cross_sectional', got {design!r}")
    animals = sorted(by[a].index.intersection(by[b].index))
    if not animals:
        raise ValueError(f"no animal has both {a} and {b}")
    return (by[b].loc[animals, "subject"].tolist(), by[a].loc[animals, "subject"].tolist(),
            by[a].loc[animals, target])


@dataclass
class ContrastData:
    """One contrast's features over its usable voxels, and what was lost to coverage."""

    name: str
    X: np.ndarray            # (n_animals, n_voxels, n_maps)
    y: pd.Series             # target per animal
    mask: np.ndarray         # usable voxels, on the atlas grid
    search_voxels: int
    coverage_pct: dict[str, float]


def contrast_data(files, sessions, atlas: AtlasInputs, settings: SearchlightSettings,
                  design: str, contrast: Mapping[str, Any], target: str = "group"
                  ) -> ContrastData:
    """Load one contrast's scans once, drop voxels any map lacks, and build its rows."""
    a, b = str(contrast["session_a"]), str(contrast["session_b"])
    name = contrast.get("name") or (f"{a}_to_{b}" if design == "delta" else str(b))
    scans_b, scans_a, y = design_scans(sessions, design, a, b, target)
    search = atlas.search_mask(settings.mask)
    maps = list(settings.maps)
    Xb = load_maps(files, scans_b, maps, search, atlas.image, settings.reduce_4d)
    Xa = None if scans_a is None else load_maps(files, scans_a, maps, search, atlas.image,
                                                settings.reduce_4d)
    scans = Xb if Xa is None else np.concatenate([Xa, Xb])
    per_map = covered(scans, settings.min_coverage_fraction)
    keep = per_map.all(axis=1)
    pct = {m: round(100 * float(per_map[:, j].mean()), 2) for j, m in enumerate(maps)}
    for m, p in pct.items():
        logger.info("[searchlight] %s: %s covers %.1f%% of the %s", name, m, p, settings.mask)
    mask = np.zeros(search.shape, bool)
    mask[search] = keep
    X = Xb[:, keep] if Xa is None else Xb[:, keep] - Xa[:, keep]
    return ContrastData(name, np.nan_to_num(X), y, mask, int(search.sum()), pct)


def sphere_neighbourhoods(mask: np.ndarray, zooms, radius_mm: float,
                          voxel_scale: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(centre coords, [indices into the mask's voxels])``, one sphere per mask voxel.

    ``radius_mm`` is in true mm: header zooms are divided by ``voxel_scale``. Neighbours come
    from the mask only, so a sphere never reaches outside the search volume.
    """
    from scipy.spatial import cKDTree

    coords = np.argwhere(mask)
    spacing = np.asarray(zooms[:3], dtype=float) / voxel_scale
    tree = cKDTree(coords * spacing)
    return coords, [np.asarray(n, dtype=np.int32)
                    for n in tree.query_ball_point(coords * spacing, r=radius_mm)]


# --- one sphere ----------------------------------------------------------------

def _estimator(settings: SearchlightSettings):
    """``(mode, estimator)`` for each sphere.

    ``svm`` (default) is an SVC on a precomputed linear kernel: each fold's Gram matrix does
    not depend on the labels, so it is built once per sphere and fold and reused by every
    relabelling, and each fit works on an n-by-n matrix however many features the sphere
    has. In murinet, at 138 features and 34 samples, this was 9.3 ms per sphere-labelling
    against 294 ms for shrinkage LDA. ``lda`` is kept for comparison on small masks.
    """
    if settings.estimator == "svm":
        from sklearn.svm import SVC

        return "kernel", SVC(kernel="precomputed", C=settings.C, class_weight="balanced")
    if settings.estimator == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return "dense", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"searchlight.estimator must be 'svm' or 'lda', got {settings.estimator!r}")


def _scale_in_fold(features: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Z-score every row on the training rows' mean and SD.

    A sphere of several maps mixes units (MWF ~0-0.3, FA ~0-1), so an unscaled linear model
    would weigh FA most. Fitting on the training rows keeps the held-out animals out of it.
    """
    mean = features[train].mean(axis=0, keepdims=True)
    sd = features[train].std(axis=0, keepdims=True)
    return (features - mean) / np.where(sd > 1e-12, sd, 1.0)


def _fold_sets(folds, n_labellings: int) -> list:
    """One list of ``(train, test)`` per labelling, from a shared list or a list per labelling."""
    if len(folds) and isinstance(folds[0], tuple):
        return [folds] * n_labellings
    if len(folds) != n_labellings:
        raise ValueError(f"{len(folds)} fold sets for {n_labellings} labellings")
    return folds


def label_folds(cv, labellings: np.ndarray, animals: np.ndarray) -> list:
    """The folds of each labelling. Stratified folds depend on the labels, so each
    permutation gets folds balanced on its own labels, as the observed labels do; folds that
    do not depend on them come out the same every time and their work is reused."""
    zeros = np.zeros(len(animals))
    return [list(cv.split(zeros, y, groups=animals)) for y in labellings]


def score_sphere(features: np.ndarray, labellings: np.ndarray, folds, estimator
                 ) -> tuple[np.ndarray, int]:
    """Balanced accuracy for every labelling (row 0 the truth), and the degenerate fits.

    ``folds`` is one list of ``(train, test)`` for every labelling, or one list per
    labelling. Scaling and the kernel depend on the fold, not the labels, so each distinct
    fold is prepared once. A fold whose training rows hold one class cannot be fitted; its
    test rows get that class and the fit is counted, so a run of tens of thousands of
    spheres does not stop on one, and the count is reported.
    """
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score

    mode, model = estimator
    prepared: dict[bytes, tuple] = {}

    def prepare(train, test):
        key = train.tobytes() + b"|" + test.tobytes()
        if key not in prepared:
            Z = _scale_in_fold(features, train)
            if mode == "kernel":
                K = Z @ Z[train].T
                prepared[key] = (K[train], K[test])
            else:
                prepared[key] = (Z[train], Z[test])
        return prepared[key]

    out = np.empty(len(labellings), dtype=np.float32)
    degenerate = 0
    for k, (y, fold_set) in enumerate(zip(labellings, _fold_sets(folds, len(labellings)),
                                          strict=True)):
        pred = np.empty_like(y)
        for train, test in fold_set:
            if len(np.unique(y[train])) < 2:
                pred[test] = y[train][0]
                degenerate += 1
                continue
            x_train, x_test = prepare(train, test)
            pred[test] = clone(model).fit(x_train, y[train]).predict(x_test)
        out[k] = balanced_accuracy_score(y, pred)
    return out, degenerate


def _chunk(X, neighbours, labellings, folds, estimator) -> tuple[np.ndarray, int]:
    scores = np.empty((len(neighbours), len(labellings)), dtype=np.float32)
    degenerate = 0
    for i, idx in enumerate(neighbours):
        scores[i], d = score_sphere(X[:, idx, :].reshape(X.shape[0], -1), labellings, folds,
                                    estimator)
        degenerate += d
    return scores, degenerate


def searchlight_scores(X: np.ndarray, y: pd.Series, neighbours: list[np.ndarray],
                       settings: SearchlightSettings, classes: Sequence[str] | None = None
                       ) -> dict[str, Any]:
    """Observed scores ``(n_voxels,)``, the null ``(n_perm, n_voxels)``, and how they were got.

    ``y`` is indexed by animal, one row each. ``classes`` orders the two groups, negative
    first (default sorted).
    """
    from joblib import Parallel, delayed

    found = sorted(y.unique())
    if len(found) != 2:
        raise ValueError(f"the searchlight needs exactly 2 groups, found {found}")
    if classes is not None:
        found = [c for c in classes if c in found]
    y_bin = (y.values == found[1]).astype(np.int8)
    animals = np.asarray(y.index)
    cv, n_splits = _splitter(settings, animals, stratify=True)
    # one set of relabellings, and their folds, for every voxel: the max-statistic null
    # needs every voxel to see the same ones
    labellings = np.vstack([y_bin, *_animal_shuffles(y_bin, animals, settings.permutations)])
    folds = label_folds(cv, labellings, animals)

    estimator = _estimator(settings)
    workers = max(1, settings.workers)
    size = max(1, len(neighbours) // (workers * 8))
    chunks = [neighbours[i:i + size] for i in range(0, len(neighbours), size)]
    _log_runtime(X, neighbours, labellings, folds, estimator, workers)
    parts = Parallel(n_jobs=workers)(
        delayed(_chunk)(X, c, labellings, folds, estimator) for c in chunks)
    scores = np.vstack([p[0] for p in parts])
    return {"observed": scores[:, 0], "null": scores[:, 1:].T, "n_splits": n_splits,
            "negative_class": found[0], "positive_class": found[1],
            "class_balance": f"{int(y_bin.sum())}/{len(y_bin)}",
            "degenerate_fits": int(sum(p[1] for p in parts))}


def _log_runtime(X, neighbours, labellings, folds, estimator, workers, sample: int = 24) -> None:
    """Time a few spheres and log the projected total before committing hours to it."""
    probe = neighbours[:: max(1, len(neighbours) // sample)][:sample]
    if not probe:
        return
    start = time.perf_counter()
    _chunk(X, probe, labellings, folds, estimator)
    per_sphere = (time.perf_counter() - start) / len(probe)
    logger.info("[searchlight] %d spheres x %d folds x %d labellings: ~%.1f ms/sphere, "
                "about %.1f min on %d worker(s)", len(neighbours), len(folds[0]), len(labellings),
                per_sphere * 1e3, per_sphere * len(neighbours) / workers / 60, workers)


def fwe_pvalues(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, float]:
    """Max-statistic FWE p per voxel, and the 95th percentile of the maxima.

    Each permutation contributes its best score anywhere in the volume, so beating that
    distribution accounts for every voxel tested.
    """
    if null.size == 0:
        return np.full_like(observed, np.nan, dtype=np.float32), float("nan")
    max_null = null.max(axis=1)
    p = (1 + (max_null[None, :] >= observed[:, None]).sum(axis=1)) / (1 + len(max_null))
    return p.astype(np.float32), float(np.quantile(max_null, 0.95))


# --- whole brain -------------------------------------------------------------

def _fold_features(X: np.ndarray, n_components: int, folds, cache: dict):
    """Per fold: scaling and PCA fitted on the training animals, applied to both sides.

    Both are unsupervised, so a fold's transformed features do not depend on the labels and
    are computed once (``cache``) however many relabellings use that fold. Fitting them
    inside the fold is still required: fitted on every row, the held-out animals would shape
    the components.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    out = []
    for train, test in folds:
        key = train.tobytes() + b"|" + test.tobytes()
        if key not in cache:
            scaler = StandardScaler().fit(X[train])
            pca = PCA(n_components=min(n_components, len(train) - 1, X.shape[1]),
                      random_state=RANDOM_STATE)
            cache[key] = (pca.fit_transform(scaler.transform(X[train])),
                          pca.transform(scaler.transform(X[test])))
        out.append((train, test, *cache[key]))
    return out


def _fold_score(estimator, y, prepared, score_fn) -> float:
    from sklearn.base import clone

    pred = np.empty(len(y), dtype=float)
    for train, test, x_train, x_test in prepared:
        pred[test] = clone(estimator).fit(x_train, y[train]).predict(x_test)
    return float(score_fn(y, pred))


def whole_brain_scores(X: np.ndarray, y: pd.Series, settings: SearchlightSettings,
                       analysis_type: str = "classification",
                       classes: Sequence[str] | None = None) -> dict[str, Any]:
    """Cross-validated score of one model over every usable voxel, with its permutation p.

    ``X`` is ``(n_animals, n_features)``. Returns the scores, and the weights of a model
    fitted on every animal, one per feature, for display only: a linear model's weights are
    not a significance map.
    """
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler

    animals = np.asarray(y.index)
    row = {"analysis_type": analysis_type}
    if analysis_type == "regression":
        from sklearn.linear_model import Ridge

        target = pd.to_numeric(y, errors="coerce")
        usable = target.notna().values
        X, target, animals = X[usable], target[usable].to_numpy(float), animals[usable]
        estimator, score_fn, metric = Ridge(alpha=settings.alpha), r2_score, "r2"
    elif analysis_type == "classification":
        from sklearn.svm import LinearSVC

        found = sorted(y.unique())
        if len(found) != 2:
            raise ValueError(f"whole-brain decoding needs exactly 2 groups, found {found}")
        if classes is not None:
            found = [c for c in classes if c in found]
        target = (y.values == found[1]).astype(int)
        estimator = LinearSVC(C=settings.C, class_weight="balanced", dual="auto",
                              max_iter=10000)
        score_fn, metric = balanced_accuracy_score, "balanced_accuracy"
        row.update(negative_class=found[0], positive_class=found[1],
                   class_balance=f"{int(target.sum())}/{len(target)}")
    else:
        raise ValueError(f"analysis_type must be classification or regression: {analysis_type!r}")

    cv, n_splits = _splitter(settings, animals, stratify=analysis_type == "classification")
    labellings = [target, *_animal_shuffles(target, animals, settings.permutations)]
    cache: dict[bytes, tuple] = {}
    scores = [_fold_score(estimator, labels,
                          _fold_features(X, settings.pca_components,
                                         cv.split(X, labels, groups=animals), cache), score_fn)
              for labels in labellings]
    observed, null = scores[0], np.asarray(scores[1:])

    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=min(settings.pca_components, len(target) - 1, X.shape[1]),
              random_state=RANDOM_STATE)
    model = clone(estimator).fit(pca.fit_transform(scaler.transform(X)), target)
    weights = np.asarray(model.coef_).reshape(-1) @ pca.components_
    row.update(metric=metric, score=observed, n_samples=len(target), n_features=X.shape[1],
               n_splits=n_splits, permutations=len(null),
               null_mean=float(null.mean()) if len(null) else float("nan"),
               p_permutation=((1 + int((null >= observed).sum())) / (1 + len(null))
                              if len(null) else float("nan")),
               pca_components=pca.n_components_)
    return {"scores": row, "weights": weights}


# --- runs --------------------------------------------------------------------

def _setup(config: dict, design: StudyDesign | None, atlas: AtlasInputs | None,
           files: Mapping | None):
    contrasts = get_config_value(config, "searchlight.contrasts", default=None)
    if not contrasts:
        raise KeyError("set searchlight.contrasts: a list of {name, session_a, session_b}")
    settings = SearchlightSettings.from_config(config)
    if not settings.maps:
        raise KeyError("set searchlight.maps: the atlas-space maps to decode from")
    design = design or load_study_design(config)
    sessions = session_table(design, get_config_value(config, "searchlight.design_dir",
                                                      default=None))
    atlas = atlas or AtlasInputs.from_config(config)
    atlas.search_mask(settings.mask)  # a mask that does not exist is a config error, not a skip
    if files is None:
        derivatives = get_config_value(config, "paths.derivatives", default=None)
        if not derivatives:
            raise KeyError("set paths.derivatives: where the space-<space> maps are")
        files = atlas_space_maps(Path(derivatives), settings.space)
    kind = get_config_value(config, "searchlight.design", default="delta")
    return contrasts, settings, design, sessions, atlas, files, kind


def _axes(config: dict) -> str | None:
    """The study-space atlas's declared orientation, written on every map."""
    from neurofaune.atlas.study_space import study_space_axes

    return study_space_axes(config)


def _modality(files: Mapping[str, Mapping[str, MapFile]], maps: Sequence[str]
              ) -> tuple[str, list[str]]:
    found = sorted({e.modality for entries in files.values() for name, e in entries.items()
                    if name in maps and e.modality})
    if len(found) == 1:
        return found[0], found
    return "multimodal", found


def run_searchlight(config: dict, output_dir: Path, design: StudyDesign | None = None,
                    atlas: AtlasInputs | None = None, files: Mapping | None = None
                    ) -> dict[str, Any]:
    """Searchlight every contrast over ``searchlight.maps``; write one results folder.

    The folder is ``<output_dir>/<modality>/decoding/searchlight/<name>/<run>/``: a tests
    table (one row per contrast), FWE clusters with their regions, a by-region table, and
    the score and FWE p maps.
    """
    started = now()
    contrasts, settings, design, sessions, atlas, files, kind = _setup(config, design, atlas,
                                                                       files)
    classes = design.levels
    zooms = atlas.image.header.get_zooms()
    rows, skipped = [], []
    for contrast in contrasts:
        try:
            data = contrast_data(files, sessions, atlas, settings, kind, contrast)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping %s: %s", contrast.get("name"), exc)
            skipped.append({"contrast": contrast.get("name"), "reason": str(exc)})
            continue
        if not data.mask.any():
            reason = (f"no voxel has all of {list(settings.maps)} in every animal; drop the most "
                      "restrictive map or lower searchlight.min_coverage_fraction")
            logger.warning("Skipping %s: %s", data.name, reason)
            skipped.append({"contrast": data.name, "reason": reason})
            continue
        _, neighbours = sphere_neighbourhoods(data.mask, zooms, settings.radius_mm,
                                              settings.voxel_scale)
        sizes = np.array([len(n) for n in neighbours])
        result = searchlight_scores(data.X, data.y, neighbours, settings, classes)
        p_fwe, threshold = fwe_pvalues(result["observed"], result["null"])
        rows.append({"data": data, "result": result, "p_fwe": p_fwe, "threshold": threshold,
                     "sphere_voxels": int(np.median(sizes))})
    if not rows:
        raise ValueError(f"no contrast could run: {skipped[:3]}")

    from neurofaune.analysis.decoding.searchlight_results import write_searchlight_results

    modality, modalities = _modality(files, settings.maps)
    folder = write_searchlight_results(
        Path(output_dir), get_config_value(config, "searchlight.name", default="searchlight"),
        rows, atlas, settings, kind, modality, modalities, design, axes=_axes(config),
        started=started)
    return {"status": "completed", "folder": folder, "skipped": skipped,
            "summary": pd.read_csv(folder / "tests.csv")}


def run_whole_brain(config: dict, output_dir: Path, design: StudyDesign | None = None,
                    atlas: AtlasInputs | None = None, files: Mapping | None = None
                    ) -> dict[str, Any]:
    """Whole-brain decoding of every contrast; one results folder with the weight maps.

    ``searchlight.analysis_type`` is ``classification`` (on ``group``) or ``regression``
    (on ``searchlight.target``).
    """
    started = now()
    contrasts, settings, design, sessions, atlas, files, kind = _setup(config, design, atlas,
                                                                       files)
    analysis_type = get_config_value(config, "searchlight.analysis_type",
                                     default="classification")
    target = get_config_value(config, "searchlight.target", default="group")
    rows, skipped = [], []
    for contrast in contrasts:
        try:
            data = contrast_data(files, sessions, atlas, settings, kind, contrast, target)
            if not data.mask.any():
                raise ValueError(f"no voxel has all of {list(settings.maps)} in every animal")
            result = whole_brain_scores(data.X.reshape(len(data.y), -1), data.y, settings,
                                        analysis_type,
                                        design.levels if target == "group" else None)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping %s: %s", contrast.get("name"), exc)
            skipped.append({"contrast": contrast.get("name"), "reason": str(exc)})
            continue
        rows.append({"data": data, "result": result})
    if not rows:
        raise ValueError(f"no contrast could run: {skipped[:3]}")

    from neurofaune.analysis.decoding.searchlight_results import write_whole_brain_results

    modality, modalities = _modality(files, settings.maps)
    folder = write_whole_brain_results(
        Path(output_dir), get_config_value(config, "searchlight.name", default="searchlight"),
        rows, atlas, settings, kind, target, modality, modalities, design, axes=_axes(config),
        started=started)
    return {"status": "completed", "folder": folder, "skipped": skipped,
            "summary": pd.read_csv(folder / "tests.csv")}


def coverage_report(config: dict, design: StudyDesign | None = None,
                    atlas: AtlasInputs | None = None, files: Mapping | None = None
                    ) -> pd.DataFrame:
    """What each of ``searchlight.maps``, and all of them together, leave of each structure.

    Counted over every session of the design, one at a time (stacking them would take
    gigabytes). Rows are the whole brain and every structure of the group file; ``pct_joint``
    is the share where every map has data in ``min_coverage_fraction`` of sessions, and
    ``limiting_map`` the map that costs the most there.
    """
    settings = SearchlightSettings.from_config(config)
    if not settings.maps:
        raise KeyError("set searchlight.maps to the maps you want a coverage report for")
    design = design or load_study_design(config)
    sessions = session_table(design)
    atlas = atlas or AtlasInputs.from_config(config)
    if files is None:
        derivatives = get_config_value(config, "paths.derivatives", default=None)
        if not derivatives:
            raise KeyError("set paths.derivatives: where the space-<space> maps are")
        files = atlas_space_maps(Path(derivatives), settings.space)
    maps = list(settings.maps)
    counts = {m: np.zeros(atlas.labels.shape, np.uint16) for m in maps}
    subjects = sessions["subject"].tolist()
    for subject in subjects:
        for m in maps:
            entry = files.get(subject, {}).get(m)
            if entry is None:
                raise FileNotFoundError(f"{subject} has no atlas-space map {m!r}")
            arr = _on_grid(nib.load(str(entry.path)), entry.path, atlas.image, settings.reduce_4d)
            counts[m] += (np.isfinite(arr) & (arr != 0)).astype(np.uint16)
    has = {m: c / max(1, len(subjects)) >= settings.min_coverage_fraction
           for m, c in counts.items()}
    joint = np.logical_and.reduce(list(has.values()))
    regions = {"whole_brain": atlas.labels > 0}
    regions.update({n: np.isin(atlas.labels, sorted(ids)) for n, ids in atlas.structures.items()})
    rows = []
    for name, region in regions.items():
        if not region.any():
            continue
        row = {"region": name, "n_voxels": int(region.sum())}
        row.update({f"pct_{m}": round(100 * float(has[m][region].mean()), 1) for m in maps})
        row["pct_joint"] = round(100 * float(joint[region].mean()), 1)
        row["n_voxels_joint"] = int(joint[region].sum())
        row["limiting_map"] = min(maps, key=lambda m: has[m][region].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pct_joint", ascending=False, ignore_index=True)


__all__ = [
    "AtlasInputs",
    "MapFile",
    "SearchlightSettings",
    "atlas_space_maps",
    "coverage_report",
    "fwe_pvalues",
    "run_searchlight",
    "run_whole_brain",
    "score_sphere",
    "searchlight_scores",
    "sphere_neighbourhoods",
    "whole_brain_scores",
]
