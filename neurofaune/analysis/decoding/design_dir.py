"""Read a design directory as an alternative label source.

``participants.tsv`` (see :mod:`neurofaune.study_design`) is the simple way to say who is in
which group. A **design directory** is the other way: an FSL-style design matrix
plus metadata, as produced by the design-building tools used for VBM and TBSS. Being
able to read it means the same design drives the mass-univariate and the multivariate
analyses, so they cannot silently disagree about which animal was in which group.

Expected layout::

    <design_dir>/design_summary.json   columns, factors (name/coding/reference/levels)
                 subject_order.txt     one subject id per row, matching design.mat
                 design.mat            FSL design matrix
                 target_values.json    optional; per-subject continuous target

Labels are reconstructed rather than stored: for a dummy-coded factor, a row with all
its indicator columns at zero is the **reference** level. That is the only way to
recover the reference group, since by construction it has no column of its own.

This is deliberately factor-agnostic. The design tooling names things per study
(``dose``, ``group``, ``log_auc``), so what to predict is inferred from the design
itself rather than hardcoded.

**How the target is inferred:** a design's ``contrasts`` declare what it tests, so
they decide. ``H_gt_C`` names factor *levels* → categorical on that factor;
``log_auc_pos`` names a declared *covariate* → regression on that covariate. This
generalises the older convention of looking for one specific column name.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_fsl_matrix(path: Path) -> list[list[float]]:
    """Numeric rows of an FSL ``.mat`` design file (everything after ``/Matrix``)."""
    rows: list[list[float]] = []
    in_matrix = False
    with open(path) as fh:
        for line in fh:
            if line.startswith("/Matrix"):
                in_matrix = True
                continue
            if in_matrix and line.strip():
                rows.append([float(x) for x in line.split()])
    return rows


def load_summary(design_dir: Path) -> dict:
    path = Path(design_dir) / "design_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"not a design directory: {path} is missing")
    with open(path) as fh:
        return json.load(fh)


def column_names(summary: dict) -> list[str]:
    """Design-matrix column names; entries may be plain strings or ``{"name": ...}``."""
    return [c if isinstance(c, str) else c.get("name", "")
            for c in summary.get("columns", [])]


def factors(summary: dict) -> dict[str, dict]:
    """``{factor name: spec}`` from ``design_summary.json``."""
    return {f["name"]: f for f in summary.get("factors", []) if "name" in f}


def covariates(summary: dict) -> dict[str, dict]:
    """``{covariate name: spec}`` — the continuous regressors a design declares."""
    return {c["name"]: c for c in summary.get("covariates", [])
            if isinstance(c, dict) and "name" in c}


# Contrast names carry a direction suffix that is not part of the variable name.
_DIRECTION_SUFFIXES = ("_pos", "_neg", "_positive", "_negative", "_up", "_down")


def _contrast_stems(summary: dict) -> set[str]:
    """Contrast names with their direction suffix removed (``dose_pos`` -> ``dose``)."""
    stems = set()
    for contrast in summary.get("contrasts", []):
        name = str(contrast)
        for suffix in _DIRECTION_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        stems.add(name)
    return stems


def _contrast_tokens(summary: dict) -> set[str]:
    """Underscore-separated pieces of every contrast name.

    Tokenised rather than substring-matched so a factor level ``M`` matches
    ``M_gt_C`` without also matching the ``numeric`` in ``dose_numeric``.
    """
    return {token for contrast in summary.get("contrasts", [])
            for token in str(contrast).split("_")}


def infer_target(summary: dict) -> tuple[str, str]:
    """``(kind, name)`` where kind is ``'covariate'`` (regression) or ``'factor'``.

    Decided by the contrasts, because that is where a design states what it is
    testing: ``log_auc_pos`` and ``dose_pos`` name covariates, ``H_gt_C`` names
    factor levels. A design with a factor of interest plus nuisance covariates
    therefore resolves to the factor, which plain name-matching would get wrong.

    Matching is stem-based in both directions: the contrast ``dose_pos`` has stem
    ``dose``, which must match the covariate ``dose_numeric``.
    """
    covs, facs = covariates(summary), factors(summary)
    stems, tokens = _contrast_stems(summary), _contrast_tokens(summary)

    for name in covs:
        if any(stem == name or name.startswith(f"{stem}_") or stem.startswith(name)
               for stem in stems):
            return "covariate", name
    for name, spec in facs.items():
        if any(str(level) in tokens for level in spec.get("levels", [])):
            return "factor", name

    if len(covs) == 1 and not facs:
        return "covariate", next(iter(covs))
    if facs:
        return "factor", next(iter(facs))
    if covs:
        return "covariate", next(iter(covs))
    raise KeyError("design declares neither factors nor covariates")


def subject_order(design_dir: Path) -> list[str]:
    path = Path(design_dir) / "subject_order.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; it defines the design.mat row order")
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_design(design_dir: Path, factor: str | None = None,
                target: str | None = None) -> pd.DataFrame:
    """``DataFrame(subject, label)`` — categorical levels, or a continuous target.

    Parameters
    ----------
    design_dir : Path
        Directory holding ``design_summary.json``, ``subject_order.txt`` and
        ``design.mat``.
    factor : str, optional
        Which factor to read levels for. Overrides inference.
    target : str, optional
        Name of a numeric design column to use as a continuous target. Overrides
        inference. ``target_values.json``, when present, takes precedence over both
        — it holds uncentred per-subject values.

    With neither given, :func:`infer_target` decides from the design's contrasts.

    Returns
    -------
    DataFrame
        ``subject`` (as written in ``subject_order.txt``) and ``label``. The label
        dtype is float for a continuous target and str for a factor.
    """
    design_dir = Path(design_dir)
    summary = load_summary(design_dir)
    subjects = subject_order(design_dir)
    columns = column_names(summary)

    # 1. explicit per-subject targets win: no reconstruction, no centring to undo
    values_path = design_dir / "target_values.json"
    if values_path.exists():
        with open(values_path) as fh:
            raw = json.load(fh)
        missing = [s for s in subjects if s not in raw]
        if missing:
            raise KeyError(f"target_values.json is missing {len(missing)} subject(s), "
                           f"e.g. {missing[:3]}")
        return pd.DataFrame({"subject": subjects,
                             "label": [float(raw[s]) for s in subjects]})

    matrix = read_fsl_matrix(design_dir / "design.mat")
    if len(matrix) != len(subjects):
        raise ValueError(
            f"design.mat has {len(matrix)} rows but subject_order.txt lists "
            f"{len(subjects)} subjects; they must correspond row for row"
        )

    # 2. a named numeric column
    if target is not None:
        if target not in columns:
            raise KeyError(f"target column {target!r} is not in the design; "
                           f"available: {columns}")
        index = columns.index(target)
        return pd.DataFrame({"subject": subjects,
                             "label": [float(row[index]) for row in matrix]})

    # 3. otherwise let the design say what it is testing
    available = factors(summary)
    name = factor
    if name is None:
        kind, inferred = infer_target(summary)
        if kind == "covariate":
            if inferred not in columns:
                raise KeyError(f"covariate {inferred!r} has no design column; "
                               f"columns are {columns}")
            index = columns.index(inferred)
            # A mean-centred covariate is fine as a regression target: centring
            # shifts the intercept, not the fit.
            return pd.DataFrame({"subject": subjects,
                                 "label": [float(row[index]) for row in matrix]})
        name = inferred

    if name not in available:
        raise KeyError(f"factor {name!r} is not in the design; "
                       f"available: {sorted(available)}")

    spec = available[name]
    reference = spec.get("reference")
    indicators = [(columns.index(c), c[len(name) + 1:])
                  for c in columns if c.startswith(f"{name}_") and c in columns]
    if not indicators:
        raise KeyError(f"factor {name!r} has no '{name}_<level>' columns in {columns}")

    labels = []
    for row in matrix:
        level = reference          # all indicators zero => the reference level
        for index, value in indicators:
            if index < len(row) and row[index] == 1.0:
                level = value
                break
        labels.append(level)
    return pd.DataFrame({"subject": subjects, "label": labels})


def describe(design_dir: Path) -> str:
    """One-line summary of a design directory, for logging."""
    summary = load_summary(design_dir)
    try:
        kind, name = infer_target(summary)
        what = f"{'regression on' if kind == 'covariate' else 'classification on'} {name}"
    except KeyError:
        what = "no target inferable"
    return (f"{Path(design_dir).name}: {summary.get('n_subjects', '?')} subjects, "
            f"{what}, {len(summary.get('contrasts', []))} contrasts")
