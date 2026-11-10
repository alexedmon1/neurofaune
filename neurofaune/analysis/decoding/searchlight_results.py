"""Write searchlight and whole-brain decoding as results-specification folders.

The tables and maps are written either way. Their column dictionaries, ``analysis.json``
and ``provenance.json`` need the optional results package, as for every other analysis.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from neurofaune.analysis.decoding.results import _default_run_id, _run_folder
from neurofaune.analysis.stats.readout import describe_cluster, label_clusters
from neurofaune.analysis.stats.readout_results import analysis_id, now

logger = logging.getLogger(__name__)

_NO_MDE = ("not computed: no power calculation exists for a cross-validated, "
           "permutation-tested decoder")

_COMMON = {
    "contrast": {"Description": "What is decoded and between which sessions", "Standard": "contrast"},
    "measure": {"Description": "The maps decoded together, joined by '+'", "Standard": "measure"},
}

SEARCHLIGHT_TESTS = {
    **_COMMON,
    "design": {"Description": "delta (within-animal change) or cross_sectional"},
    "tested_direction": {"Description": "What the test asks of each sphere's score",
                         "Standard": "tested_direction"},
    "n": {"Description": "Animals decoded, one row each", "Standard": "n"},
    "negative_class": {"Description": "Class coded 0"},
    "positive_class": {"Description": "Class coded 1"},
    "class_balance": {"Description": "Positive animals / all animals"},
    "mask": {"Description": "The search volume: whole_brain or a structure of the group file"},
    "search_voxels": {"Description": "Voxels in the search volume", "Units": "voxels"},
    "usable_voxels": {"Description": "Search voxels where every map has data in every animal; "
                                     "one sphere is centred on each", "Units": "voxels"},
    "coverage": {"Description": "Percent of the search volume each map covers, 'map:pct; ...'"},
    "radius_mm": {"Description": "Sphere radius in true mm", "Units": "mm"},
    "sphere_voxels": {"Description": "Median voxels per sphere", "Units": "voxels"},
    "features_per_sphere": {"Description": "sphere_voxels times the number of maps"},
    "n_splits": {"Description": "Cross-validation folds, grouped by animal"},
    "peak_score": {"Description": "Highest cross-validated balanced accuracy of any sphere",
                   "Standard": "stat", "StatName": "balanced accuracy", "StatScope": "peak"},
    "mean_score": {"Description": "Mean balanced accuracy over the spheres"},
    "fwe_threshold": {"Description": "95th percentile of the permutation maxima: the score a "
                                     "sphere must beat for FWE p < 0.05"},
    "n_significant": {"Description": "Spheres with FWE-corrected p below fwe_alpha",
                      "Units": "voxels", "Standard": "n_significant", "PKind": "fwe"},
    "frac_significant": {"Description": "n_significant as a fraction of usable_voxels",
                         "Standard": "frac_significant", "PKind": "fwe"},
    "p_value": {"Description": "Smallest FWE-corrected p of any sphere (maximum statistic over "
                               "the usable voxels)", "Standard": "p_value", "PKind": "fwe",
                "PScope": "minimum over the search volume"},
    "significant": {"Description": "Any sphere with FWE-corrected p below fwe_alpha",
                    "Standard": "significant"},
    "permutations": {"Description": "Relabellings between animals, shared by every sphere"},
    "degenerate_fits": {"Description": "Fits skipped because a training fold held one class, "
                                       "over every sphere and labelling; their test animals "
                                       "were given that class"},
    "mde_note": {"Description": "Why no minimum detectable effect", "Standard": "mde_note"},
}

SEARCHLIGHT_CLUSTERS = {
    **_COMMON,
    "cluster": {"Description": "Cluster id within its contrast", "Standard": "element"},
    "n_voxels": {"Description": "Cluster extent", "Units": "voxels", "Standard": "n_voxels"},
    "mm3": {"Description": "Cluster extent, in the maps' stored (x voxel_scale) units",
            "Units": "mm^3", "Standard": "volume_mm3"},
    "peak_score": {"Description": "Highest balanced accuracy in the cluster", "Standard": "stat",
                   "StatName": "balanced accuracy", "StatScope": "peak"},
    "peak_ijk": {"Description": "Peak voxel indices, 'i j k'"},
    "peak_xyz_mm": {"Description": "Peak coordinate, 'x y z' in mm (image affine)",
                    "Standard": "peak_xyz_mm"},
    "cog_xyz_mm": {"Description": "Centre of gravity, 'x y z' in mm"},
    "p_value": {"Description": "Smallest FWE-corrected p in the cluster", "Standard": "p_value",
                "PKind": "fwe", "PScope": "minimum within the cluster"},
    "peak_region": {"Description": "Atlas region at the peak", "Standard": "peak_region"},
    "top_region": {"Description": "Atlas region with the most cluster voxels"},
    "regions": {"Description": "Every region the cluster covers, 'name:voxels; ...', largest "
                               "first", "Standard": "regions"},
    "n_regions": {"Description": "Regions the cluster covers"},
    "outside_atlas_vox": {"Description": "Cluster voxels with no atlas label", "Units": "voxels"},
    "crosses_midline": {"Description": "The cluster covers left and right regions",
                        "Standard": "crosses_midline"},
}

SEARCHLIGHT_REGIONS = {
    **_COMMON,
    "region": {"Description": "Atlas region"},
    "region_id": {"Description": "Atlas label id"},
    "n_voxels": {"Description": "Usable voxels of the region", "Units": "voxels"},
    "peak_score": {"Description": "Highest balanced accuracy of a sphere centred in the region"},
    "mean_score": {"Description": "Mean balanced accuracy of the spheres centred in the region"},
    "n_significant": {"Description": "Spheres in the region with FWE-corrected p below fwe_alpha",
                      "Units": "voxels"},
}

WHOLE_BRAIN_TESTS = {
    **_COMMON,
    "design": {"Description": "delta (within-animal change) or cross_sectional"},
    "target": {"Description": "What is decoded: a column of the study design"},
    "analysis_type": {"Description": "classification or regression"},
    "tested_direction": {"Description": "What the test asks of the score",
                         "Standard": "tested_direction"},
    "n": {"Description": "Animals decoded, one row each", "Standard": "n"},
    "negative_class": {"Description": "Class coded 0"},
    "positive_class": {"Description": "Class coded 1"},
    "class_balance": {"Description": "Positive animals / all animals"},
    "usable_voxels": {"Description": "Voxels where every map has data in every animal",
                      "Units": "voxels"},
    "n_features": {"Description": "usable_voxels times the number of maps"},
    "n_splits": {"Description": "Cross-validation folds, grouped by animal"},
    "pca_components": {"Description": "Principal components, fitted inside each fold"},
    "metric": {"Description": "What 'stat' is: balanced_accuracy or r2"},
    "stat": {"Description": "Cross-validated score", "Standard": "stat",
             "StatName": "cross-validated score"},
    "null_mean": {"Description": "Mean score under relabelling between animals"},
    "permutations": {"Description": "Relabellings between animals"},
    "p_value": {"Description": "Permutation p-value of the score", "Standard": "p_value",
                "PKind": "perm"},
    "mde_note": {"Description": "Why no minimum detectable effect", "Standard": "mde_note"},
}


def _tag(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", "_".join(parts))


def _write_map(values: np.ndarray, mask: np.ndarray, image: nib.Nifti1Image, path: Path) -> None:
    volume = np.zeros(mask.shape, np.float32)
    volume[mask] = values
    nib.save(nib.Nifti1Image(volume, image.affine, image.header), str(path))


def _modality_fields(modality: str, modalities: Sequence[str]) -> dict:
    return {"modality": modality,
            **({"modalities": list(modalities)} if modality == "multimodal" else {})}


def _design_block(design, n: int) -> dict:
    groups = design.table().drop_duplicates("animal")["group"].value_counts().to_dict()
    return {"n": int(n), "groups": {str(k): int(v) for k, v in groups.items()}}


def _finish(folder: Path, tables: dict[str, tuple[pd.DataFrame, dict]], analysis: dict,
            started: str | None) -> None:
    """Write the tables; with the results package, their dictionaries and the JSON too."""
    from neurofaune import results

    for name, (table, _) in tables.items():
        table.to_csv(folder / name, index=False)
    if not results.AVAILABLE:
        logger.warning("Wrote %s; not its column dictionaries, analysis.json or "
                       "provenance.json: %s", folder, results.INSTALL_HINT)
        return
    from neurofaune.provenance import generated_by
    from neurofaune.results import provenance as provenance_record
    from neurofaune.results import write_analysis, write_columns

    for name, (_, columns) in tables.items():
        write_columns(folder / name, {k: v for k, v in columns.items()
                                      if k in pd.read_csv(folder / name, nrows=0).columns})
    report = write_analysis(folder, analysis,
                            provenance_record(generated_by(), status="completed",
                                              start=started or now(), end=now()),
                            strict=False)
    for e in report.errors:
        logger.error("results spec: %s", e)
    for w in report.warnings:
        logger.warning("results spec: %s", w)


def write_searchlight_results(root: Path, name: str, rows: list[dict], atlas, settings,
                              design_kind: str, modality: str, modalities: Sequence[str],
                              design, *, axes: str | None = None, run_id: str | None = None,
                              started: str | None = None) -> Path:
    """``<root>/<modality>/decoding/searchlight/<name>/<run>/``; returns the folder."""
    identifier = analysis_id(modality, "decoding", "searchlight", name)
    folder = _run_folder(root, identifier, run_id or _default_run_id())
    folder.mkdir(parents=True, exist_ok=False)
    measure = "+".join(settings.maps)
    affine = atlas.image.affine
    vox_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    tests, clusters, regions, maps = [], [], [], []
    for row in rows:
        data, result, p_fwe = row["data"], row["result"], row["p_fwe"]
        observed = result["observed"]
        significant = p_fwe < settings.fwe_alpha
        tests.append({
            "contrast": data.name, "measure": measure, "design": design_kind,
            "tested_direction": "balanced accuracy above the permutation maximum",
            "n": len(data.y), "negative_class": result["negative_class"],
            "positive_class": result["positive_class"], "class_balance": result["class_balance"],
            "mask": settings.mask, "search_voxels": data.search_voxels,
            "usable_voxels": int(data.mask.sum()),
            "coverage": "; ".join(f"{m}:{p}" for m, p in data.coverage_pct.items()),
            "radius_mm": settings.radius_mm, "sphere_voxels": row["sphere_voxels"],
            "features_per_sphere": row["sphere_voxels"] * len(settings.maps),
            "n_splits": result["n_splits"], "peak_score": float(observed.max()),
            "mean_score": float(observed.mean()), "fwe_threshold": row["threshold"],
            "n_significant": int(significant.sum()),
            "frac_significant": float(significant.mean()),
            "p_value": float(np.nanmin(p_fwe)) if np.isfinite(p_fwe).any() else np.nan,
            "significant": bool(significant.any()), "permutations": settings.permutations,
            "degenerate_fits": result["degenerate_fits"], "mde_note": _NO_MDE,
        })
        ijk = np.argwhere(data.mask)
        labelled = atlas.names.masked(data.mask)
        lab, _, keep = label_clusters(significant, data.mask)
        for cid in keep:
            where, _ = describe_cluster(lab == cid, observed, ijk, affine, vox_mm3, labelled)
            clusters.append({"contrast": data.name, "measure": measure, "cluster": int(cid),
                             "n_voxels": where.pop("n_voxels"), "mm3": where.pop("mm3"),
                             "peak_score": float(observed[lab == cid].max()),
                             "p_value": float(np.nanmin(p_fwe[lab == cid])), **where})
        by_region = (pd.DataFrame({"region_id": labelled.labels, "score": observed,
                                   "significant": significant})
                     .groupby("region_id")
                     .agg(n_voxels=("score", "size"), peak_score=("score", "max"),
                          mean_score=("score", "mean"), n_significant=("significant", "sum"))
                     .reset_index())
        by_region.insert(0, "region", by_region["region_id"].map(
            lambda r: atlas.names.name(int(r)) or "outside the atlas"))
        by_region.insert(0, "measure", measure)
        by_region.insert(0, "contrast", data.name)
        regions.append(by_region.sort_values(["n_significant", "peak_score"], ascending=False))

        tag = _tag(data.name, measure)
        for kind, values, file, what in (
                ("stat", observed, f"{tag}_score.nii.gz", "cross-validated balanced accuracy "
                                                          "of the sphere centred on each voxel"),
                ("p_corrected", p_fwe, f"{tag}_pfwe.nii.gz", "FWE-corrected p (maximum "
                                                             "statistic) of each sphere")):
            _write_map(values, data.mask, atlas.image, folder / file)
            maps.append({"path": file, "kind": kind, "space": settings.space, "measure": measure,
                         "contrast": data.name, "description": f"{what}, {data.name}",
                         **({"values": "p"} if kind == "p_corrected" else {}),
                         **({"axes": axes} if axes else {})})

    tests_df = pd.DataFrame(tests)
    clusters_df = pd.DataFrame(clusters, columns=None if clusters else
                               ["contrast", "measure", "cluster", "n_voxels", "mm3", "peak_score",
                                "peak_xyz_mm", "peak_region", "regions", "p_value"])
    regions_df = pd.concat(regions, ignore_index=True)
    analysis = {
        "id": identifier, "run": {"id": folder.name},
        "title": f"Searchlight decoding of {measure}: {name}",
        "description": (
            f"A sphere of radius {settings.radius_mm} mm centred on every usable voxel of the {settings.mask}, decoded by "
            f"{settings.estimator} with folds grouped by animal and scaling fitted inside each fold; FWE "
            "controlled by the maximum score over the volume under relabelling between animals."
        ),
        "analysis_type": "decoding", **_modality_fields(modality, modalities),
        "measures": [measure], "role": "exploratory",
        "design": _design_block(design, int(tests_df["n"].max())),
        "inference": {
            "method": "searchlight, label permutation between animals, maximum statistic",
            "n_permutations": int(settings.permutations),
            "correction": {
                "p_kind": "fwe", "method": "maxT", "alpha": float(settings.fwe_alpha),
                "family": f"spheres in the {settings.mask}, per contrast",
                "statement": (f"FWE p < {settings.fwe_alpha:g} by the maximum balanced accuracy "
                              f"over the search volume in {settings.permutations} relabellings, "
                              "per contrast"),
            },
        },
        "effect": {"measure": None, "reason": "the cross-validated balanced accuracy in the "
                                              "score map and 'peak_score' is the magnitude"},
        "tables": [
            {"path": "tests.csv", "role": "tests", "headline": True, "n_rows": len(tests_df),
             "rows": "one contrast, significant or not",
             "description": "every contrast with its search volume, coverage, peak and extent"},
            {"path": "clusters.csv", "role": "clusters", "n_rows": len(clusters_df),
             "rows": "one cluster of FWE-significant spheres",
             "description": "clusters with extent, peak and the regions they cover"},
            {"path": "regions.csv", "role": "descriptives", "n_rows": len(regions_df),
             "rows": "one atlas region in one contrast",
             "description": "spheres summarised by the region of their centre"},
        ],
        "maps": maps,
        "caveats": [
            "Cluster extents (mm3) are in the maps' stored units, x voxel_scale per axis.",
            f"{sum(t['degenerate_fits'] for t in tests)} fits had one class in training; "
            "their test animals were given that class (degenerate_fits)."
        ],
    }
    _finish(folder, {"tests.csv": (tests_df, SEARCHLIGHT_TESTS),
                     "clusters.csv": (clusters_df, {
                         **SEARCHLIGHT_CLUSTERS,
                         "peak_xyz_mm": {**SEARCHLIGHT_CLUSTERS["peak_xyz_mm"],
                                         "Space": settings.space}}),
                     "regions.csv": (regions_df, SEARCHLIGHT_REGIONS)}, analysis, started)
    return folder


def write_whole_brain_results(root: Path, name: str, rows: list[dict], atlas, settings,
                              design_kind: str, target: str, modality: str,
                              modalities: Sequence[str], design, *, axes: str | None = None,
                              run_id: str | None = None, started: str | None = None) -> Path:
    """``<root>/<modality>/decoding/wholebrain/<name>/<run>/``; returns the folder."""
    identifier = analysis_id(modality, "decoding", "wholebrain", name)
    folder = _run_folder(root, identifier, run_id or _default_run_id())
    folder.mkdir(parents=True, exist_ok=False)
    measure = "+".join(settings.maps)
    tests, maps = [], []
    for row in rows:
        data, scores, weights = row["data"], row["result"]["scores"], row["result"]["weights"]
        tests.append({
            "contrast": data.name, "measure": measure, "design": design_kind, "target": target,
            "analysis_type": scores["analysis_type"],
            "tested_direction": "score above the permutation null", "n": scores["n_samples"],
            **{k: scores[k] for k in ("negative_class", "positive_class", "class_balance")
               if k in scores},
            "usable_voxels": int(data.mask.sum()), "n_features": scores["n_features"],
            "n_splits": scores["n_splits"], "pca_components": scores["pca_components"],
            "metric": scores["metric"], "stat": scores["score"], "null_mean": scores["null_mean"],
            "permutations": scores["permutations"], "p_value": scores["p_permutation"],
            "mde_note": _NO_MDE,
        })
        per_map = weights.reshape(int(data.mask.sum()), len(settings.maps))
        for j, map_name in enumerate(settings.maps):
            file = f"{_tag(data.name, measure)}_weights-{map_name}.nii.gz"
            _write_map(per_map[:, j], data.mask, atlas.image, folder / file)
            maps.append({"path": file, "kind": "other", "space": settings.space,
                         "measure": measure, "contrast": data.name,
                         "description": f"weights of the model fitted on every animal, on "
                                        f"{map_name} (z-scored), for display only: not a "
                                        f"significance map; {data.name}",
                         **({"axes": axes} if axes else {})})
    tests_df = pd.DataFrame(tests)
    analysis = {
        "id": identifier, "run": {"id": folder.name},
        "title": f"Whole-brain decoding of {measure}: {name}",
        "description": (
            "One linear model over every usable voxel, with folds grouped by animal and "
            "scaling and PCA fitted inside each fold; tested against relabelling between "
            "animals. Weight maps are for display only."),
        "analysis_type": "decoding", **_modality_fields(modality, modalities),
        "measures": [measure], "role": "exploratory",
        "design": _design_block(design, int(tests_df["n"].max())),
        "inference": {
            "method": "label permutation between animals",
            "n_permutations": int(settings.permutations),
            "correction": {"p_kind": "perm", "family": "each contrast alone",
                           "statement": (f"Permutation p of each contrast; not corrected over "
                                         f"the {len(tests_df)} contrasts in this table")},
        },
        "effect": {"measure": None, "reason": "the cross-validated score in 'stat' is the "
                                              "magnitude; no interval is computed"},
        "tables": [{"path": "tests.csv", "role": "tests", "headline": True,
                    "n_rows": len(tests_df), "rows": "one contrast",
                    "description": "every contrast, significant or not"}],
        "maps": maps,
    }
    _finish(folder, {"tests.csv": (tests_df, WHOLE_BRAIN_TESTS)}, analysis, started)
    return folder
