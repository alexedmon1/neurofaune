"""Run decoding for a study from its config. Settings are under ``decoding:``.

Keys, beyond those of :class:`~neurofaune.analysis.decoding.harness.DecodeSettings`:

- ``name``: the analysis name in the results id (default ``decoding``);
- ``roi_stats_dir``, ``morphometry_dir``: where the feature tables are; a set is used only
  when its folder is given;
- ``roi_stats_min_coverage`` (default 0), ``roi_stats_statistics`` (default all);
- ``design``: ``delta`` (default) or ``cross_sectional``;
- ``analysis_type``, ``target``: classification on ``group`` by default;
- ``contrasts``: required, a list of ``{name, session_a, session_b}``;
- ``design_dir``: an FSL-style design directory to take the labels from.

They are not in ``default.yaml``, whose contents go into the provenance digest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from neurofaune.analysis.decoding.features import (
    build_design,
    morphometry_features,
    roi_stats_features,
    session_table,
)
from neurofaune.analysis.decoding.harness import DecodeSettings, evaluate
from neurofaune.analysis.decoding.results import write_decoding_results
from neurofaune.analysis.stats.readout_results import now
from neurofaune.config import get_config_value
from neurofaune.study_design import StudyDesign, load_study_design

logger = logging.getLogger(__name__)


def _feature_sets(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    sets: dict[str, pd.DataFrame] = {}
    roi_dir = get_config_value(config, "decoding.roi_stats_dir")
    if roi_dir:
        sets.update(roi_stats_features(
            Path(roi_dir),
            float(get_config_value(config, "decoding.roi_stats_min_coverage", default=0.0)),
            get_config_value(config, "decoding.roi_stats_statistics"),
        ))
    morphometry_dir = get_config_value(config, "decoding.morphometry_dir")
    if morphometry_dir:
        sets.update(morphometry_features(Path(morphometry_dir)))
    return sets


def run_decoding(
    config: dict[str, Any],
    output_dir: Path,
    design: StudyDesign | None = None,
    feature_sets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Decode every feature set in every contrast, and write one results folder per modality.

    ``summary`` holds every decoder run; ``skipped`` the ones that could not run, and why.
    """
    started = now()
    contrasts = get_config_value(config, "decoding.contrasts")
    if not contrasts:
        raise KeyError("set decoding.contrasts: a list of {name, session_a, session_b}")
    design = design or load_study_design(config)
    sets = feature_sets if feature_sets is not None else _feature_sets(config)
    if not sets:
        raise FileNotFoundError(
            "no feature tables: set decoding.roi_stats_dir or decoding.morphometry_dir"
        )

    settings = DecodeSettings.from_config(config)
    kind = get_config_value(config, "decoding.design", default="delta")
    analysis_type = get_config_value(config, "decoding.analysis_type", default="classification")
    target = get_config_value(config, "decoding.target", default="group")
    sessions = session_table(design, get_config_value(config, "decoding.design_dir"))
    classes = design.levels if target == "group" else None

    summary, predictions, skipped = [], [], []
    for contrast in contrasts:
        a, b = str(contrast["session_a"]), str(contrast["session_b"])
        name = contrast.get("name") or f"{a}_to_{b}"
        for set_name, features in sorted(sets.items()):
            try:
                X, y, groups = build_design(features, sessions, kind, a, b, target)
                scores, preds = evaluate(X, y, groups, settings, analysis_type, classes=classes)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping %s / %s: %s", set_name, name, exc)
                skipped.append({"feature_set": set_name, "contrast": name, "reason": str(exc)})
                continue
            summary.append(
                {"feature_set": set_name, "design": kind, "contrast": name, "target": target,
                 **scores}
            )
            predictions.append(preds.assign(feature_set=set_name, contrast=name))
    if not summary:
        raise ValueError(f"no decoder could run: {skipped[:3]}")

    table = pd.DataFrame(summary)
    n_groups = design.table().drop_duplicates("animal")["group"].value_counts().to_dict()
    folders = write_decoding_results(
        Path(output_dir),
        get_config_value(config, "decoding.name", default="decoding"),
        table,
        n_groups,
        started=started,
    )
    return {
        "status": "completed",
        "summary": table,
        "predictions": pd.concat(predictions, ignore_index=True),
        "skipped": skipped,
        "folders": folders,
    }
