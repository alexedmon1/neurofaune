"""A randomise read-out written as a results-specification folder.

`read_randomise` (readout.py) returns a tests table and a clusters table. This
module writes them -- ``tests.csv`` and ``clusters.csv``, where the TBSS and
VBM / voxelwise-fMRI pipelines always wrote them -- with their column
dictionaries, ``analysis.json`` and ``provenance.json`` (docs/RESULTS_SPEC.md), so
any reader can use the results without neurofaune.

The column dictionary below covers every column `read_randomise` can produce;
a test fails if it ever writes one that is not described here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from neurofaune.results import provenance as provenance_record
from neurofaune.results import write_analysis, write_columns
from neurofaune.results.write import now

logger = logging.getLogger("neurofaune.stats")

_DIRECTION = {"Description": "what a positive statistic means, in the design's terms "
                             "(randomise tests each contrast one-sided, t > 0)",
              "Standard": "tested_direction"}

#: Columns common to both tables (from the design and the contrast).
_BASE: dict[str, dict[str, Any]] = {
    "contrast": {"Description": "contrast number in design.con (1-based)"},
    "contrast_name": {"Description": "contrast name, from the design record or design.con",
                      "Standard": "contrast"},
    "contrast_vector": {"Description": "the contrast weights, one per design column"},
    "design": {"Description": "how the contrast reads in the design",
               "Levels": {"two-group": "+1/-1 on two indicator columns",
                          "one-sample": "a weight on a constant column",
                          "single-column": "the sign of one column's coefficient",
                          "general": "any other weighting, written out"}},
    "tested_direction": _DIRECTION,
    "contrast_tests": {"Description": "what the contrast tests, in words, from the design record "
                                      "(empty when the design has no record)",
                       "Standard": "contrast_label"},
    "n": {"Description": "rows (subjects) in the design", "Standard": "n"},
    "df": {"Description": "error degrees of freedom, n - rank(X)", "Standard": "df"},
}

TESTS_COLUMNS: dict[str, dict[str, Any]] = {
    **_BASE,
    "mask_voxels": {"Description": "voxels in the analysis mask", "Units": "voxels"},
    "n_vox_fwe": {"Description": "voxels with FWE-corrected p below alpha", "Units": "voxels",
                  "Standard": "n_significant", "PKind": "fwe"},
    "frac_mask_fwe": {"Description": "n_vox_fwe as a fraction of the mask",
                      "Standard": "frac_significant", "PKind": "fwe"},
    "min_p_fwe": {"Description": "smallest FWE-corrected voxel p in the map",
                  "Standard": "p_value", "PKind": "fwe", "PScope": "minimum over the mask"},
    "n_vox_uncorr": {"Description": "voxels with uncorrected p below alpha (empty when randomise "
                                    "wrote no uncorrected maps)", "Units": "voxels",
                     "PKind": "uncorrected"},
    "frac_mask_uncorr": {"Description": "n_vox_uncorr as a fraction of the mask", "PKind": "uncorrected"},
    "min_p_uncorr": {"Description": "smallest uncorrected voxel p in the map", "PKind": "uncorrected"},
    "peak_t": {"Description": "largest t in the map", "Standard": "stat", "StatName": "t",
               "StatScope": "peak"},
    "cluster_definition": {"Description": "how clusters were formed for this test"},
    "n_clusters": {"Description": "clusters under cluster_definition"},
    "largest_cluster_vox": {"Description": "voxels in the largest cluster", "Units": "voxels"},
    "significant_fwe": {"Description": "any voxel with FWE-corrected p below alpha",
                        "Standard": "significant"},
    "whole_effect_n": {"Description": "subjects with a finite whole-mask mean"},
    "whole_effect_df": {"Description": "degrees of freedom of the whole-mask effect"},
    "whole_estimate": {"Description": "the contrast estimate c'b on each subject's mean over the "
                                      "whole mask, in the measure's units", "Standard": "estimate"},
    "whole_d": {"Description": "c'b / residual SD on each subject's mean over the whole mask: pooled-SD "
                               "Cohen's d for two groups without covariates, mean / SD for one sample; "
                               "never selected", "Standard": "effect_size", "EffectMeasure": "d",
                "EffectScope": "whole mask, unselected"},
    "whole_d_ci_low": {"Description": "lower bound of whole_d (exact, noncentral t)",
                       "Standard": "effect_ci_low"},
    "whole_d_ci_high": {"Description": "upper bound of whole_d (exact, noncentral t)",
                        "Standard": "effect_ci_high"},
    "whole_d_raw": {"Description": "whole-mask d without covariate adjustment (pooled-SD for two "
                                   "groups, mean / SD for one sample)"},
    "whole_mean_pos": {"Description": "whole-mask mean of the group weighted +1", "Standard": "mean_a"},
    "whole_mean_neg": {"Description": "whole-mask mean of the group weighted -1", "Standard": "mean_b"},
    "whole_n_pos": {"Description": "subjects in the group weighted +1", "Standard": "n_a"},
    "whole_n_neg": {"Description": "subjects in the group weighted -1", "Standard": "n_b"},
    "whole_mean": {"Description": "whole-mask mean (one-sample tests)", "Standard": "mean"},
    "whole_observed_direction": {"Description": "the direction the whole-mask effect went",
                                 "Standard": "observed_direction"},
}

CLUSTERS_COLUMNS: dict[str, dict[str, Any]] = {
    **_BASE,
    "cluster": {"Description": "cluster id within its test", "Standard": "element"},
    "n_voxels": {"Description": "cluster extent", "Units": "voxels", "Standard": "n_voxels"},
    "mm3": {"Description": "cluster extent", "Units": "mm^3", "Standard": "volume_mm3"},
    "peak_t": {"Description": "largest t in the cluster", "Standard": "stat", "StatName": "t",
               "StatScope": "peak"},
    "peak_ijk": {"Description": "peak voxel indices, 'i j k'"},
    "peak_xyz_mm": {"Description": "peak coordinate, 'x y z' in mm (image affine)",
                    "Standard": "peak_xyz_mm"},
    "cog_xyz_mm": {"Description": "centre of gravity, 'x y z' in mm"},
    "min_p_fwe": {"Description": "smallest FWE-corrected p in the cluster", "Standard": "p_value",
                  "PKind": "fwe", "PScope": "minimum within the cluster"},
    "min_p_uncorr": {"Description": "smallest uncorrected p in the cluster", "PKind": "uncorrected"},
    "fwe_significant": {"Description": "the cluster holds a voxel with FWE-corrected p below alpha",
                        "Standard": "significant"},
    "peak_region": {"Description": "atlas region at the peak", "Standard": "peak_region"},
    "top_region": {"Description": "atlas region with the most cluster voxels"},
    "regions": {"Description": "every region the cluster covers, 'name:voxels; ...', largest first",
                "Standard": "regions"},
    "n_regions": {"Description": "regions the cluster covers"},
    "outside_atlas_vox": {"Description": "cluster voxels with no atlas label", "Units": "voxels"},
    "crosses_midline": {"Description": "the cluster covers left and right regions",
                        "Standard": "crosses_midline"},
    "effect_n": {"Description": "subjects with a finite cluster mean"},
    "effect_df": {"Description": "degrees of freedom of the cluster effect"},
    "estimate": {"Description": "the contrast estimate on each subject's mean over the cluster",
                 "Standard": "estimate"},
    "d": {"Description": "d on each subject's mean over the cluster -- computed on voxels chosen "
                         "for significance, so inflated", "Standard": "effect_size",
          "EffectMeasure": "d", "EffectScope": "cluster mean, selected"},
    "d_ci_low": {"Description": "lower bound of d (does not hold its coverage: selected)",
                 "Standard": "effect_ci_low"},
    "d_ci_high": {"Description": "upper bound of d (does not hold its coverage: selected)",
                  "Standard": "effect_ci_high"},
    "d_raw": {"Description": "cluster d without covariate adjustment"},
    "mean_pos": {"Description": "cluster mean of the group weighted +1", "Standard": "mean_a"},
    "mean_neg": {"Description": "cluster mean of the group weighted -1", "Standard": "mean_b"},
    "n_pos": {"Description": "subjects in the group weighted +1", "Standard": "n_a"},
    "n_neg": {"Description": "subjects in the group weighted -1", "Standard": "n_b"},
    "mean": {"Description": "cluster mean (one-sample tests)", "Standard": "mean"},
    "observed_direction": {"Description": "the direction the cluster effect went",
                           "Standard": "observed_direction"},
    "effect_selected": {"Description": "the effect was computed on voxels chosen for significance",
                        "Standard": "effect_selected"},
}

#: Header of an empty clusters table, so a run with no clusters still writes a readable file.
EMPTY_CLUSTERS = ["contrast", "contrast_name", "tested_direction", "n", "df", "cluster", "n_voxels",
                  "mm3", "peak_t", "peak_xyz_mm", "peak_region", "regions", "min_p_fwe", "d",
                  "effect_selected"]


def _fsl_version() -> dict | None:
    import os
    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        return None
    f = Path(fsldir) / "etc" / "fslversion"
    try:
        return {"Name": "FSL", "Version": f.read_text().split(":")[0].strip(),
                "Description": "randomise"}
    except OSError:
        return None


def _same_grid(a: Path, b: Path) -> bool:
    import nibabel as nib
    import numpy as np

    ia, ib = nib.load(str(a)), nib.load(str(b))
    return ia.shape[:3] == ib.shape[:3] and np.allclose(ia.affine, ib.affine, atol=1e-3)


def write_readout_results(
    output_dir: Path,
    tests: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    analysis_id: str,
    title: str,
    description: str,
    analysis_type: str,
    measure_column: str,
    measures: Sequence[str],
    run_dirs: Mapping[str, Path] | None = None,
    run_dir_of: Callable[[pd.Series], Path] | None = None,
    facet_column: str | None = None,
    prefix: str = "randomise",
    n_permutations: int,
    alpha: float,
    mask_name: str,
    space: str,
    inference: str,
    role: str = "exploratory",
    modality: str | None = None,
    extra_columns: Mapping[str, Mapping[str, Any]] | None = None,
    design_record: Mapping[str, Any] | None = None,
    started: str | None = None,
    inputs: Sequence[Mapping[str, Any]] = (),
    settings: Mapping[str, Any] | None = None,
    caveats: Sequence[str] = (),
    mask: Path | None = None,
    background: Path | None = None,
    design: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    references: Sequence[Mapping[str, str]] = (),
    tests_file: str = "tests.csv",
    clusters_file: str = "clusters.csv",
    strict: bool = False,
):
    """Write tests.csv, clusters.csv, their dictionaries, analysis.json and provenance.json.

    Args:
        output_dir: the analysis folder (holds ``run_dirs``).
        tests, clusters: from `read_randomise`, concatenated over measures.
        measure_column: the `labels` key that names the measure (e.g. "metric").
        run_dirs: measure -> its randomise folder (inside ``output_dir``), when each
            measure has one run; otherwise ``run_dir_of(row)`` gives a test row's folder.
        facet_column: a labels column that splits the tests further (e.g. window); it is
            marked as the standard ``facet`` and maps carry it.
        prefix: randomise's output prefix in the run folders.
        design: the analysis's design block (n, groups, ...), for a battery of designs;
            by default it is taken from ``design_record`` and the first test row.
        decision, references: passed into analysis.json as the specification defines them.
        tests_file, clusters_file: table names, so a folder that already holds other
            tables of those names can adopt the specification without losing them.
        mask_name: what the mask is, in words ("TBSS skeleton", "brain mask").
        space: template space of the maps (e.g. "SIGMA").
        inference: how the maps were corrected, as a noun phrase: "2-D TFCE" (skeleton),
            "3-D TFCE", or "voxel-wise maximum t".
        extra_columns: dictionary entries for other constant `labels` columns.
        design_record: the run's design.json content, when it has one.
        mask, background: the analysis mask, and the intensity template of the atlas the
            maps are in (``neurofaune.atlas.study_space.study_space_template``), copied into
            the folder as mask.nii.gz / background.nii.gz so the folder is complete on its
            own. A background on another grid than the mask is left out, with a warning:
            readers draw maps on it voxel for voxel.
        strict: raise when the folder does not conform; otherwise log and return.

    Returns:
        The conformance report.
    """
    output_dir = Path(output_dir)
    if (run_dirs is None) == (run_dir_of is None):
        raise ValueError("give exactly one of run_dirs and run_dir_of")
    if run_dir_of is None:
        def run_dir_of(row):
            return Path(run_dirs[row[measure_column]])
    extra = {measure_column: {"Description": "the measure tested", "Standard": "measure"},
             "analysis": {"Description": "analysis name"}, **(extra_columns or {})}
    if facet_column:
        extra[facet_column] = {**extra.get(facet_column, {"Description": "the test within the battery"}),
                               "Standard": "facet"}
    clusters = clusters if len(clusters.columns) else pd.DataFrame(columns=EMPTY_CLUSTERS)
    tests.to_csv(output_dir / tests_file, index=False)
    clusters.to_csv(output_dir / clusters_file, index=False)
    space_q = {"peak_xyz_mm": {**CLUSTERS_COLUMNS["peak_xyz_mm"], "Space": space}}
    write_columns(output_dir / tests_file, TESTS_COLUMNS, extra)
    write_columns(output_dir / clusters_file, CLUSTERS_COLUMNS, {**extra, **space_q})

    from neurofaune.analysis.stats.readout import _map
    maps = []
    keys = ["contrast", measure_column] + ([facet_column] if facet_column else [])
    for _, r in tests.drop_duplicates(keys).iterrows():
        rd = Path(run_dir_of(r))
        for kind, key, what in (("stat", "tstat", "t statistic"),
                                ("p_corrected", "corrp", "1 - FWE-corrected p (randomise's convention)"),
                                ("p_uncorrected", "p", "1 - uncorrected p (randomise's convention)")):
            f = _map(rd, prefix, key, int(r.contrast))
            if f is not None:
                maps.append({"path": f.relative_to(output_dir).as_posix(), "kind": kind, "space": space,
                             "measure": str(r[measure_column]), "contrast": str(r.contrast_name),
                             **({"facet": str(r[facet_column])} if facet_column else {}),
                             "description": f"{what}, {r.contrast_name} on {r[measure_column]}",
                             **({"values": "one_minus_p"} if kind != "stat" else {})})
    import shutil
    if background is not None and mask is not None and Path(background).exists() \
            and Path(mask).exists() and not _same_grid(Path(background), Path(mask)):
        logger.warning(f"results spec: {background} is not on the grid of {mask}; "
                       "no background listed")
        background = None
    for src, name, kind, what in ((mask, "mask.nii.gz", "mask", f"the {mask_name}"),
                                  (background, "background.nii.gz", "background",
                                   f"intensity template of the {space} atlas, on the maps' grid")):
        if src is not None and Path(src).exists():
            if Path(src).resolve() != (output_dir / name).resolve():
                shutil.copyfile(src, output_dir / name)
            maps.append({"path": name, "kind": kind, "space": space, "description": what})
    run_folders = sorted({Path(run_dir_of(r)) for _, r in tests.iterrows()})
    records = [p.relative_to(output_dir).as_posix()
               for p in [output_dir / "design.json", *(d / "design.json" for d in run_folders)]
               if p.exists()]

    groups = dict((design_record or {}).get("groups") or {})
    kinds = sorted({c.get("test_kind") for c in (design_record or {}).get("contrasts", [])
                    if c.get("test_kind")})
    analysis = {
        "id": analysis_id, "title": title, "description": description,
        "analysis_type": analysis_type, "measures": list(measures), "role": role,
        "design": {**(dict(design) if design else
                      {"n": int(tests["n"].iloc[0]), **({"groups": groups} if groups else {}),
                       **({"test_kinds": kinds} if kinds else {})}),
                   **({"records": records} if records else {})},
        "inference": {
            "method": f"FSL randomise, {inference}, {n_permutations} permutations",
            "n_permutations": int(n_permutations),
            "correction": {"p_kind": "fwe", "alpha": float(alpha),
                           "family": f"voxels in the {mask_name}, per contrast and measure",
                           "statement": f"{inference}, FWE p < {alpha:g} by {n_permutations} permutations, "
                                        f"per contrast over the {mask_name}"}},
        "effect": {"measure": "d", "ci_level": 0.95, "scope": "whole mask, unselected (tests); "
                   "cluster mean, selected (clusters)",
                   "definition": "contrast estimate over the residual SD of each subject's mean over "
                                 "the voxels: pooled-SD Cohen's d for two groups without covariates, "
                                 "mean / SD for one sample; 95% CI exact under normal residuals"},
        "tables": [
            {"path": tests_file, "role": "tests", "headline": True, "n_rows": int(len(tests)),
             "rows": "one t-contrast on one measure, significant or not",
             "description": "every test with its whole-mask effect, CI, extent and peak"},
            {"path": clusters_file, "role": "clusters", "n_rows": int(len(clusters)),
             "rows": "one cluster of one test",
             "description": "clusters with extent, peak, named regions and the (selected) cluster effect"},
        ],
        "maps": maps,
        **({"modality": modality} if modality else {}),
        **({"caveats": list(caveats)} if caveats else {}),
        **({"decision": dict(decision)} if decision else {}),
        **({"references": [dict(r) for r in references]} if references else {}),
    }
    from neurofaune.provenance import generated_by
    gen = generated_by()
    fsl = _fsl_version()
    prov = provenance_record(gen + ([fsl] if fsl else []), status="completed",
                             start=started or now(), end=now(), inputs=[dict(i) for i in inputs],
                             subjects={"n": int((design or {}).get("n", tests["n"].iloc[0])),
                                       **({"groups": dict((design or {}).get("groups") or groups)}
                                          if (design or {}).get("groups") or groups else {})},
                             settings=dict(settings or {}))
    report = write_analysis(output_dir, analysis, prov, strict=strict)
    for e in report.errors:
        logger.error(f"results spec: {e}")
    for w in report.warnings:
        logger.warning(f"results spec: {w}")
    return report
