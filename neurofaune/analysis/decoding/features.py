"""Feature tables and the (X, y, groups) of a decoding design.

Designs:

``delta``
    One row per animal: the within-animal change ``X(session_b) - X(session_a)``. An animal
    missing either session drops out, since imputing it would invent the change measured.

``cross_sectional``
    One row per session at ``session_b``. Every between-animal difference is noise here.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurofaune.study_design import StudyDesign


def _wide(df: pd.DataFrame, index: list[str], columns: list[str], value: str) -> pd.DataFrame:
    """Long -> wide, with a flat single-string column name per feature."""
    wide = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    wide.columns = [
        "|".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]
    return wide


def roi_stats_features(
    roi_dir: Path,
    min_coverage: float = 0.0,
    statistics: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """``{name: wide frame}`` from ``roi_stats_*_long.csv``: one set per metric, one combined.

    Tagged by source table, not metric alone: the same metric exported under another
    coverage rule (``msme`` and ``msme_common``) is a different measurement, and keyed on
    metric alone the pivot would silently average them.
    """
    frames = []
    for path in sorted(Path(roi_dir).glob("roi_stats_*_long.csv")):
        frame = pd.read_csv(path)
        frame["source"] = path.name[len("roi_stats_"):-len("_long.csv")]
        frames.append(frame)
    if not frames:
        return {}
    stats = pd.concat(frames, ignore_index=True)
    stats["metric"] = stats["source"] + ":" + stats["metric"]
    stats["subject"] = stats["subject"] + "_" + stats["session"]
    if min_coverage > 0:
        stats = stats[stats["coverage"] >= min_coverage]
    if statistics:
        stats = stats[stats["statistic"].isin(list(statistics))]

    out = {}
    for metric, sub in stats.groupby("metric"):
        out[f"roi_stats|{metric}"] = _wide(sub, ["subject"], ["roi_name", "statistic"], "value")
    for source, sub in stats.groupby("source"):
        if sub["metric"].nunique() > 1:
            out[f"roi_stats|{source}:combined"] = _wide(
                sub, ["subject"], ["metric", "roi_name", "statistic"], "value"
            )
    return {name: frame for name, frame in out.items() if not frame.empty}


def morphometry_features(tables: Path) -> dict[str, pd.DataFrame]:
    """``{name: wide frame}`` from the morphometry tables that exist under ``tables``."""
    out = {}
    path = Path(tables) / "morphometry_structures.csv"
    if path.exists():
        out["morphometry_structures"] = _wide(
            pd.read_csv(path), ["subject"], ["structure", "tissue"], "volume_mm3"
        )
    path = Path(tables) / "morphometry_regions.csv"
    if path.exists():
        regions = pd.read_csv(path)
        long = regions.melt(
            id_vars=["subject", "region"],
            value_vars=[
                c for c in regions.columns if c.startswith("volume_") and c != "volume_mm3"
            ],
            var_name="tissue",
            value_name="v",
        )
        out["morphometry_regions"] = _wide(long, ["subject"], ["region", "tissue"], "v")
    return {name: frame for name, frame in out.items() if not frame.empty}


def session_table(design: StudyDesign, design_dir: Path | None = None) -> pd.DataFrame:
    """The design's sessions, with ``subject`` = ``<animal>_<session>`` as the feature key.

    With ``design_dir``, its labels replace ``group``: the same design then drives the
    mass-univariate and the multivariate analyses, so they cannot disagree. A design covers
    one analysis, so sessions outside it are dropped rather than left unlabelled.
    """
    table = design.table()
    table.insert(0, "subject", table["animal"] + "_" + table["session"])
    if design_dir is None:
        return table

    from neurofaune.analysis.decoding.design_dir import load_design

    labels = load_design(design_dir).rename(columns={"label": "group"})
    merged = table.drop(columns="group").merge(labels, on="subject", how="inner")
    if merged.empty:
        raise ValueError(
            "no session in the study design matches a subject in the design directory; "
            f"its ids look like {labels['subject'].head(2).tolist()}"
        )
    return merged


def _session_id(session: str) -> str:
    return session if str(session).startswith("ses-") else f"ses-{session}"


def build_design(
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    design: str,
    session_a: str,
    session_b: str,
    target: str = "group",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """``(X, y, groups)`` for one design; ``groups`` is the animal of each row."""
    meta = sessions.set_index("subject")
    common = features.index.intersection(meta.index)
    feats, meta = features.loc[common], meta.loc[common]
    session_a, session_b = _session_id(session_a), _session_id(session_b)

    if target not in meta.columns:
        raise KeyError(f"target {target!r} is not a column of the study design")

    if design == "cross_sectional":
        keep = meta["session"] == session_b
        return feats[keep.values], meta.loc[keep.values, target], meta.loc[keep.values, "animal"]

    if design != "delta":
        raise ValueError(f"design must be 'delta' or 'cross_sectional', got {design!r}")

    by_session = {}
    for session in (session_a, session_b):
        here = meta["session"] == session
        frame = feats[here.values].copy()
        frame.index = meta.loc[here.values, "animal"].values
        by_session[session] = frame

    animals = by_session[session_a].index.intersection(by_session[session_b].index)
    if len(animals) == 0:
        raise ValueError(f"no animal has both {session_a} and {session_b}")
    X = by_session[session_b].loc[animals] - by_session[session_a].loc[animals]
    labels = meta.drop_duplicates("animal").set_index("animal")[target]
    return X, labels.loc[animals], pd.Series(animals, index=animals, name="animal")
