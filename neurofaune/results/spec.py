"""The vocabulary of the results specification (docs/RESULTS_SPEC.md).

Standard library only: this package is meant to run -- or be copied -- where
neurofaune's imaging dependencies are not installed (neurovrai, a reader's laptop).
"""
from __future__ import annotations

SPEC = "neurofaune.results"
SPEC_VERSION = "0.1.0"

ANALYSIS_JSON = "analysis.json"
PROVENANCE_JSON = "provenance.json"

ANALYSIS_TYPES = ("tbss", "vbm", "tbm", "voxelwise", "roi", "covariance_network", "nbs", "graph",
                  "connectome", "fixel", "other")
ROLES = ("confirmatory", "exploratory", "descriptive", "diagnostic")
P_KINDS = ("fwe", "fdr", "perm", "uncorrected", "none")
TABLE_ROLES = ("tests", "clusters", "elements", "descriptives", "other")
CONTRACT_ROLES = ("tests", "clusters", "elements")
MAP_KINDS = ("stat", "p_corrected", "p_uncorrected", "effect", "mask", "background", "input",
             "other")
RUN_STATUSES = ("completed", "partial", "failed", "running")
DECISION_OUTCOMES = ("holds", "does_not_hold", "not_assessed")
TEST_KINDS = ("two_group", "one_sample", "regression", "interaction", "custom")
TABLE_SUFFIXES = {".csv": ",", ".tsv": "\t"}

#: Standard column terms -> the qualifiers each one requires in its dictionary entry.
STANDARD_TERMS: dict[str, tuple[str, ...]] = {
    "measure": (), "contrast": (), "contrast_label": (), "facet": (), "element": (),
    "test_kind": (), "tested_direction": (), "observed_direction": (),
    "group_a": (), "group_b": (), "n": (), "n_a": (), "n_b": (), "df": (),
    "effect_size": ("EffectMeasure",), "effect_ci_low": (), "effect_ci_high": (),
    "effect_selected": (), "estimate": (), "mean_a": (), "mean_b": (), "mean": (),
    "stat": ("StatName",), "p_value": ("PKind",), "significant": (),
    "n_significant": (), "frac_significant": (), "n_voxels": (), "volume_mm3": (),
    "peak_xyz_mm": ("Space",), "peak_region": (), "regions": (), "crosses_midline": (),
}

#: Qualifier keys a dictionary entry may carry, besides BIDS's Description / Units / Levels.
QUALIFIERS = ("Standard", "EffectMeasure", "EffectScope", "CILevel", "StatName", "StatScope",
              "PKind", "PScope", "Alpha", "Space")
