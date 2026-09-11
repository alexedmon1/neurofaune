"""Structural covariance networks: morphometry tables -> CovNet input.

:mod:`neurofaune.network.morphometry` produces long-format per-subject volumes.
:class:`neurofaune.network.covnet.CovNetAnalysis` consumes a wide ROI table and
already implements everything a structural covariance analysis needs — group-wise
Spearman matrices, NBS, graph-theoretic density curves, network distance. This
module is the bridge between them; it adds no new statistics.

Three things have to happen in between, and only the first is bookkeeping:

1. **Reshape.** ``extract_morphometry.py`` keys rows by a combined ``sub-X_ses-Y``
   stem, and CovNet needs separate ``subject``/``session`` columns (its exclusion
   CSVs and its ``cohort`` derivation both depend on them).

2. **Remove global head size.** This is the one that decides whether the result
   means anything. Regional volumes all scale with the animal, so a covariance
   matrix built from raw volumes is dominated by a single size factor and nearly
   every edge comes out strongly positive — the network is a picture of how big the
   rats are. The standard correction (He et al. 2007) regresses each region on
   total brain volume, plus sex and age, and correlates the residuals. That is the
   default here (``method="residual"``); ``"proportion"`` (value / total brain) is
   offered because it is common in the rodent literature, but it imposes a fixed
   slope of 1 in log space and is the weaker choice. ``"none"`` exists so the
   inflation can be demonstrated rather than asserted.

   Residualisation is **pooled across the analysis sample**, not run within each
   group. Within a group the only confound that varies is total brain volume, and
   fitting 3 nuisance parameters inside a cell of n≈10 costs more than it removes.
   Group differences in mean survive pooled residualisation, which is harmless:
   a within-group correlation does not see the group's mean.

3. **Choose a node set.** Every edge is a correlation *across subjects within one
   group*, so the sample size is the group size, not the session count. At the
   234-region level with n≈10-12 per cell the matrix is noise. Prefer
   ``nodes="structures"`` (composite structures, ~20 nodes) or bilateral regions,
   and treat full unilateral region-level SCN as needing a much larger cohort.

Morphometry is native-space only, so this inherits that: in atlas space every
subject shares one mask and the volumes are identical by construction.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ``extract_morphometry.py`` writes ``subject`` as a combined session stem.
STEM = re.compile(r"^(?P<subject>sub-[^_\s]+)_(?P<session>ses-[^_\s]+)$")

#: Structures resolved from ``all_labels: true`` — whole-brain summaries. They are
#: excluded from the node set because correlating a global total against its own
#: parts is circular, and ``total_brain`` is the normalisation denominator.
GLOBAL_STRUCTURES = ("total_brain", "supratentorial", "white_matter", "total_csf")

TBV_COLUMN = "total_brain_mm3"
DEFAULT_CONFOUNDS = (TBV_COLUMN, "sex", "cohort")
NORMALISATIONS = ("residual", "proportion", "none")

META_COLS = ("subject", "session", "cohort", "dose", "sex", TBV_COLUMN)


# ---------------------------------------------------------------------------
# Reshaping
# ---------------------------------------------------------------------------

def split_session_stem(df: pd.DataFrame, column: str = "subject") -> pd.DataFrame:
    """Split a combined ``sub-X_ses-Y`` key into ``subject`` and ``session``.

    Raises rather than dropping unparseable rows: a silently missing session is
    indistinguishable downstream from one that was legitimately excluded.
    """
    if {"subject", "session"}.issubset(df.columns) and column == "subject":
        stems = df["subject"].astype(str)
        if not stems.str.contains("_ses-").any():
            return df.copy()  # already split

    parsed = df[column].astype(str).str.extract(STEM)
    bad = df.loc[parsed["subject"].isna(), column].unique()
    if len(bad):
        raise ValueError(
            f"{len(bad)} key(s) in column {column!r} are not 'sub-X_ses-Y': "
            f"{sorted(bad)[:5]}"
        )
    out = df.copy()
    out["subject"] = parsed["subject"]
    out["session"] = parsed["session"]
    cols = ["subject", "session"] + [c for c in out.columns if c not in ("subject", "session")]
    return out[cols]


def _pivot(long_df: pd.DataFrame, name_col: str, value_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Long -> wide on (subject, session), one column per `name_col` value."""
    if value_col not in long_df.columns:
        raise KeyError(
            f"measure {value_col!r} not in the table; available: "
            f"{[c for c in long_df.columns if c.startswith(('volume', 'mean', 'median'))]}"
        )
    df = split_session_stem(long_df)
    wide = df.pivot_table(
        index=["subject", "session"], columns=name_col, values=value_col, aggfunc="mean"
    )
    wide.columns = [str(c) for c in wide.columns]
    node_cols = sorted(wide.columns)
    wide = wide[node_cols].reset_index()
    wide.columns.name = None
    return wide, node_cols


def region_nodes(
    regions: pd.DataFrame, measure: str = "volume_GM_mm3"
) -> tuple[pd.DataFrame, list[str]]:
    """Nodes = the 234 SIGMA regions, from ``morphometry_regions.csv``.

    Column names come from ``load_parcellation``'s ``roi_name`` (dots already
    replaced by underscores, ``_L``/``_R`` retained), so CovNet's territory
    mapping resolves them unchanged.
    """
    wide, cols = _pivot(regions, "region", measure)
    logger.info("Region nodes: %d regions x %d sessions (%s)", len(cols), len(wide), measure)
    return wide, cols


def structure_nodes(
    structures: pd.DataFrame,
    measure: str = "volume_mm3",
    exclude: Sequence[str] = GLOBAL_STRUCTURES,
    tissue: str | None = None,
    structure_labels: dict[str, Sequence[int]] | None = None,
    allow_nested: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Nodes = composite structures, from ``morphometry_structures.csv``.

    A structure is emitted once per tissue it was measured with, so the node name
    is ``{structure}_{tissue}`` (``cerebellum_GM``, ``thalamus_any``). Pass
    `tissue` to keep only one. Whole-brain summaries in `exclude` are dropped —
    see :data:`GLOBAL_STRUCTURES`.

    Pass `structure_labels` (the contents of the extraction's
    ``structure_labels.json``) to also drop nodes that nest inside one another;
    see :func:`prune_overlapping_nodes` for why that matters. `allow_nested`
    keeps them.
    """
    df = structures.copy()
    if tissue is not None:
        df = df[df["tissue"] == tissue]
    df = df[~df["structure"].isin(set(exclude))]
    if df.empty:
        raise ValueError("no structures left after filtering; check --tissue / exclude")
    df["node"] = df["structure"] + "_" + df["tissue"].astype(str)
    wide, cols = _pivot(df, "node", measure)
    if structure_labels and not allow_nested:
        cols, _ = prune_overlapping_nodes(cols, structure_labels)
        wide = wide[["subject", "session"] + cols]
    logger.info("Structure nodes: %d structures x %d sessions (%s)",
                len(cols), len(wide), measure)
    return wide, cols


def prune_overlapping_nodes(
    node_cols: Sequence[str],
    structure_labels: dict[str, Sequence[int]],
) -> tuple[list[str], dict[str, str]]:
    """Drop nodes that overlap another node by construction.

    A covariance network needs nodes that could in principle vary independently.
    The composite structures deliberately do not: ``subcortical`` contains
    hippocampus, amygdala, thalamus and striatum; ``fiber_tracts`` contains the
    named tracts; ``cerebellum_any`` is the sum of its own GM and WM split. Edges
    between a structure and its own parts are high because of the label sets, not
    because of the anatomy, and they would dominate any network statistic.

    Two rules, applied to the resolved label sets that ``extract_morphometry.py``
    writes to ``structure_labels.json``:

    1. A node whose label set is a strict superset of another node's is dropped —
       the finer partition is kept.
    2. One node per label set. Where a structure carries both a tissue-weighted
       and an unweighted measure the two are the same region measured twice, so
       the aseg convention the group file documents decides which survives: a
       structure with a GM/WM split keeps the split and loses the ``_any`` sum,
       while a discrete structure measured one way keeps the ``_any`` total and
       loses the single weighted variant. Force a different choice with the
       `tissue` argument.

    Returns ``(kept, {dropped_node: reason})``. Nodes with no entry in
    `structure_labels` are kept untouched.
    """
    def label_set(node: str) -> set[int] | None:
        for structure, ids in structure_labels.items():
            for tissue_suffix in ("_any", "_GM", "_WM", "_CSF"):
                if node == f"{structure}{tissue_suffix}":
                    return set(ids)
        return set(structure_labels[node]) if node in structure_labels else None

    sets = {node: label_set(node) for node in node_cols}
    known = {n: v for n, v in sets.items() if v}
    dropped: dict[str, str] = {}

    # Rule 1: strict supersets.
    for node, ids in known.items():
        smaller = [
            other for other, other_ids in known.items()
            if other != node and other_ids < ids
        ]
        if smaller:
            dropped[node] = f"label set contains {', '.join(sorted(smaller)[:3])}"

    # Rule 2: one node per label set, following the aseg convention.
    by_labels: dict[frozenset, list[str]] = {}
    for node, ids in known.items():
        if node not in dropped:
            by_labels.setdefault(frozenset(ids), []).append(node)
    for nodes in by_labels.values():
        weighted = sorted(n for n in nodes if not n.endswith("_any"))
        totals = sorted(n for n in nodes if n.endswith("_any"))
        if not weighted or not totals:
            continue
        if len(weighted) >= 2:
            # A GM/WM split: keep it, drop the sum.
            for node in totals:
                dropped[node] = f"sum of {', '.join(weighted)}"
        else:
            # A discrete structure measured one way: keep the aseg-style total.
            for node in weighted:
                dropped[node] = f"same label set as {totals[0]}"

    kept = [n for n in node_cols if n not in dropped]
    if dropped:
        logger.info(
            "Pruned %d overlapping node(s) of %d:\n  %s",
            len(dropped), len(node_cols),
            "\n  ".join(f"{n}: {r}" for n, r in sorted(dropped.items())),
        )
    if not kept:
        raise ValueError("pruning removed every node; pass allow_nested=True to keep them")
    return kept, dropped


def thickness_nodes(
    thickness: pd.DataFrame, measure: str = "mean_thickness_mm"
) -> tuple[pd.DataFrame, list[str]]:
    """Nodes = cortical regions, from ``morphometry_thickness.csv``.

    EXPLORATORY. The underlying measure is plane-restricted apparent thickness —
    read :mod:`neurofaune.network.thickness` before interpreting anything built
    on it.
    """
    wide, cols = _pivot(thickness, "region", measure)
    logger.warning(
        "Thickness nodes are plane-restricted apparent thickness (%d regions) — "
        "exploratory, not a FreeSurfer-equivalent measure", len(cols),
    )
    return wide, cols


def total_brain_volume(
    structures: pd.DataFrame, structure: str = "total_brain", tissue: str = "any"
) -> pd.DataFrame:
    """``subject, session, total_brain_mm3`` — the normalisation denominator.

    ``any`` is the full labelled extent. There is no skull in this data, so this
    is not an eTIV and every ratio derived from it is a fraction of labelled
    brain, not of intracranial volume.
    """
    df = split_session_stem(structures)
    sel = df[(df["structure"] == structure) & (df["tissue"] == tissue)]
    if sel.empty:
        raise ValueError(
            f"structure {structure!r} / tissue {tissue!r} not in the structures "
            f"table; cannot normalise for head size"
        )
    out = (sel.groupby(["subject", "session"], as_index=False)["volume_mm3"]
              .mean().rename(columns={"volume_mm3": TBV_COLUMN}))
    logger.info("Total brain volume: n=%d, %.1f-%.1f mm3", len(out),
                out[TBV_COLUMN].min(), out[TBV_COLUMN].max())
    return out


def load_participants(
    path: Path,
    subject_col: str = "participant_id",
    group_col: str = "group",
    sex_col: str = "sex",
) -> pd.DataFrame:
    """Read a BIDS ``participants.tsv`` as ``subject, dose, sex``.

    ``neurofaune.network.roi_extraction.merge_phenotype`` reads the bpa-rat study
    tracker (``irc.ID`` / ``dose.level``); this reads the BIDS sidecar instead, so
    a study that never had a tracker can still be grouped. The group column is
    renamed to ``dose`` because that is the name CovNet's grouping uses — it is the
    experimental factor, not necessarily a dose.
    """
    path = Path(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    table = pd.read_csv(path, sep=sep)

    missing = [c for c in (subject_col, group_col) if c not in table.columns]
    if missing:
        raise KeyError(
            f"{path} has no column(s) {missing}; available: {list(table.columns)}"
        )

    out = pd.DataFrame({
        "subject": table[subject_col].astype(str),
        "dose": table[group_col],
    })
    out["subject"] = out["subject"].where(
        out["subject"].str.startswith("sub-"), "sub-" + out["subject"]
    )
    if sex_col in table.columns:
        out["sex"] = table[sex_col]
    else:
        logger.warning("%s has no %r column; sex will not be available as a "
                       "confound or a stratifier", path, sex_col)

    logger.info("Participants: %d subjects, groups=%s", len(out),
                sorted(out["dose"].dropna().unique()))
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _design_matrix(df: pd.DataFrame, confounds: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Intercept + numeric confounds + dummy-coded categoricals."""
    missing = [c for c in confounds if c not in df.columns]
    if missing:
        raise KeyError(
            f"confound column(s) not available: {missing}. Supply them by joining "
            f"a phenotype table (--participants / --study-tracker), or drop them "
            f"from the confound list."
        )
    block = pd.get_dummies(df[list(confounds)], drop_first=True, dummy_na=False)
    block = block.astype(float)
    # A confound that is constant in this sample carries no information and makes
    # the design rank-deficient; drop it rather than relying on lstsq's pinv.
    constant = [c for c in block.columns if block[c].nunique(dropna=True) <= 1]
    if constant:
        logger.info("Dropping constant confound term(s): %s", ", ".join(constant))
        block = block.drop(columns=constant)
    design = np.column_stack([np.ones(len(block)), block.to_numpy(dtype=float)])
    return design, list(block.columns)


def residualise(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    confounds: Sequence[str] = DEFAULT_CONFOUNDS,
    add_mean_back: bool = True,
) -> pd.DataFrame:
    """OLS-residualise each node column on `confounds`, pooled over all rows.

    `add_mean_back` restores each column's mean so the values stay in the original
    units and read as volumes; it has no effect on any correlation.

    Rows with a missing confound cannot be residualised and are set to NaN — they
    are then dropped per-edge by CovNet's pairwise-complete correlation rather
    than silently entering the model with an imputed value.
    """
    design, terms = _design_matrix(df, confounds)
    usable = np.isfinite(design).all(axis=1)
    if not usable.all():
        logger.warning("%d/%d rows have a missing confound; their nodes become NaN",
                       int((~usable).sum()), len(df))

    out = df.copy()
    for col in value_cols:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        fit_rows = usable & np.isfinite(y)
        if fit_rows.sum() <= design.shape[1]:
            logger.warning("Node %r has %d usable rows for %d design columns; left as NaN",
                           col, int(fit_rows.sum()), design.shape[1])
            out[col] = np.nan
            continue
        beta, *_ = np.linalg.lstsq(design[fit_rows], y[fit_rows], rcond=None)
        resid = np.full(len(df), np.nan)
        resid[fit_rows] = y[fit_rows] - design[fit_rows] @ beta
        if add_mean_back:
            resid[fit_rows] += float(np.mean(y[fit_rows]))
        out[col] = resid

    logger.info("Residualised %d nodes on [intercept, %s] (n=%d)",
                len(value_cols), ", ".join(terms), int(usable.sum()))
    return out


def normalise_nodes(
    df: pd.DataFrame,
    node_cols: Sequence[str],
    method: str = "residual",
    confounds: Sequence[str] = DEFAULT_CONFOUNDS,
    tbv_col: str = TBV_COLUMN,
) -> pd.DataFrame:
    """Apply the chosen head-size correction. See the module docstring."""
    if method not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {method!r}; use one of {NORMALISATIONS}")

    if method == "none":
        logger.warning(
            "normalise='none': regional volumes share a global size factor, so "
            "expect a uniformly positive covariance matrix. Use for comparison only."
        )
        return df.copy()

    if method == "proportion":
        if tbv_col not in df.columns:
            raise KeyError(f"{tbv_col!r} required for proportion normalisation")
        out = df.copy()
        denom = pd.to_numeric(out[tbv_col], errors="coerce").replace(0, np.nan)
        for col in node_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce") / denom
        logger.info("Normalised %d nodes as a fraction of %s", len(node_cols), tbv_col)
        return out

    return residualise(df, node_cols, confounds)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_covnet_table(
    nodes: pd.DataFrame,
    node_cols: Sequence[str],
    structures: pd.DataFrame | None = None,
    phenotype: pd.DataFrame | None = None,
    study_tracker: Path | None = None,
    method: str = "residual",
    confounds: Sequence[str] = DEFAULT_CONFOUNDS,
    bilateral: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Join phenotype + total brain volume onto a node table and normalise.

    Returns ``(df, node_cols)`` in the layout ``CovNetAnalysis.prepare`` expects:
    ``subject, session, cohort, dose, sex, <nodes...>``.
    """
    df = nodes.copy()
    node_cols = list(node_cols)

    if "cohort" not in df.columns:
        df["cohort"] = df["session"].str.extract(r"ses-(\w+)")[0]

    if structures is not None:
        df = df.merge(total_brain_volume(structures), on=["subject", "session"], how="left")
        n_missing = int(df[TBV_COLUMN].isna().sum())
        if n_missing:
            logger.warning("%d session(s) have no total_brain volume", n_missing)

    if phenotype is not None:
        df = df.merge(phenotype, on="subject", how="left")
    elif study_tracker is not None:
        from neurofaune.network.roi_extraction import merge_phenotype
        df = merge_phenotype(df, Path(study_tracker))

    for col in ("dose", "sex"):
        if col not in df.columns:
            logger.warning("No %r column; CovNet grouping will fail without it", col)

    if bilateral:
        from neurofaune.network.matrices import bilateral_average
        df, node_cols = bilateral_average(df, node_cols)

    df = normalise_nodes(df, node_cols, method=method, confounds=confounds)

    lead = [c for c in META_COLS if c in df.columns]
    df = df[lead + [c for c in node_cols if c in df.columns]]
    return df, [c for c in node_cols if c in df.columns]


def export_covnet_wide(
    df: pd.DataFrame, output_dir: Path, metric: str
) -> Path:
    """Write ``roi_{metric}_wide.csv`` — the file ``CovNetAnalysis`` reads."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"roi_{metric}_wide.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d sessions x %d columns)", path, len(df), df.shape[1])
    return path
