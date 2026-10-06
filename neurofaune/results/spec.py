"""The vocabulary of the results specification (docs/RESULTS_SPEC.md).

Standard library only: this package is meant to run -- or be copied -- where
neurofaune's imaging dependencies are not installed (neurovrai, a reader's laptop).
"""
from __future__ import annotations

SPEC = "neurofaune.results"
SPEC_VERSION = "0.2.0"
#: The (major, minor) versions this package reads: every 0.x up to its own. A 0.1 folder
#: is checked by 0.1's rules; the rules added in 0.2 apply to folders that declare 0.2.
READS = ((0, 1), (0, 2))

ANALYSIS_JSON = "analysis.json"
PROVENANCE_JSON = "provenance.json"

ANALYSIS_TYPES = ("tbss", "vbm", "tbm", "voxelwise", "roi", "covariance_network", "nbs", "graph",
                  "connectome", "fixel", "decoding", "radiomics", "spectroscopy", "other")
#: The modality an analysis's data come from (0.2: required, one of these). An analysis
#: on several modalities says "multimodal" and lists them in ``modalities``.
MODALITIES = ("anat", "dwi", "func", "msme", "mrs", "asl", "multimodal")
#: ``references`` labels with an agreed meaning (others are allowed, and opaque):
#: a pre-registered hypothesis the analysis tests, a recorded finding it reports, a DOI.
REFERENCE_LABELS = ("registration", "finding", "doi")
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
PLANES = ("axial", "coronal", "sagittal")
#: Each voxel axis runs toward one end of one of these pairs (S = superior / dorsal,
#: I = inferior / ventral for animals as for people).
AXIS_PAIRS = ("RL", "AP", "SI")


def valid_axes(code: str) -> bool:
    """Three letters, one from each of R/L, A/P, S/I, e.g. "LIA"."""
    if not isinstance(code, str) or len(code) != 3:
        return False
    used = [next((i for i, pair in enumerate(AXIS_PAIRS) if ch in pair), None) for ch in code.upper()]
    return None not in used and sorted(used) == [0, 1, 2]

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
    "subgroup_effect": ("EffectMeasure", "Subgroup"), "subgroup_n": ("Subgroup",),
}

#: Standard terms a table may carry once per subgroup (each column names its Subgroup).
PER_SUBGROUP = ("subgroup_effect", "subgroup_n")

#: Qualifier keys a dictionary entry may carry, besides BIDS's Description / Units / Levels.
QUALIFIERS = ("Standard", "EffectMeasure", "EffectScope", "CILevel", "StatName", "StatScope",
              "PKind", "PScope", "Alpha", "Space", "Subgroup")


# ── 0.2: identity and vocabulary ──────────────────────────────────────────────────

_SEGMENT = "abcdefghijklmnopqrstuvwxyz0123456789_"


def version_tuple(version: str) -> tuple[int, int] | None:
    """(major, minor) of a semver string, or None."""
    try:
        major, minor = (int(x) for x in str(version).split(".")[:2])
    except ValueError:
        return None
    return major, minor


def id_problem(analysis_id: str, modality: str, analysis_type: str) -> str | None:
    """Why an id does not follow ``<modality>/<analysis_type>/<name>`` (0.2), else None.

    The name may itself have ``/``-separated parts; every part is lowercase letters,
    digits and underscores (dots and hyphens also allowed in the name).
    """
    parts = str(analysis_id).split("/")
    if len(parts) < 3:
        return "is not <modality>/<analysis_type>/<name>"
    if parts[0] != modality:
        return f"starts with {parts[0]!r}, not its modality {modality!r}"
    if parts[1] != analysis_type:
        return f"has {parts[1]!r} second, not its analysis_type {analysis_type!r}"
    for part in parts[2:]:
        if not part or any(ch not in _SEGMENT + ".-" for ch in part):
            return f"name part {part!r} is not lowercase letters, digits, '_', '.' or '-'"
    return None


def measure_vocabulary() -> dict[str, dict]:
    """The measure vocabulary (vocab/measures.json): canonical name -> {modality,
    description, aliases}."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / "vocab" / "measures.json").read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def canonical_measure(name: str, vocabulary: dict[str, dict] | None = None) -> tuple[str | None, bool]:
    """(the canonical name ``name`` stands for, whether ``name`` already is it).

    (None, False) for a name the vocabulary does not know. A case variant or an alias of
    a known measure returns its canonical name and False.
    """
    vocab = vocabulary if vocabulary is not None else measure_vocabulary()
    if name in vocab:
        return name, True
    low = str(name).lower()
    for canon, entry in vocab.items():
        if low == canon.lower() or low in {a.lower() for a in entry.get("aliases", [])}:
            return canon, False
    return None, False


def analysis_id(modality: str, analysis_type: str, *name: str) -> str:
    """``<modality>/<analysis_type>/<name>``, each name part lowercased and anything but
    letters, digits, '_', '.' and '-' turned into '_' -- the id 0.2 asks for."""
    import re

    parts = [re.sub(r"[^a-z0-9_.-]+", "_", str(p).lower()).strip("_") for p in name]
    return "/".join([modality, analysis_type, *(p for p in parts if p)])
