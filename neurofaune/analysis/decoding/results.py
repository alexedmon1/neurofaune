"""Write decoding scores as results-specification folders, one per modality.

``tests.csv`` is written either way. Its column dictionary, ``analysis.json`` and
``provenance.json`` need the optional results package, as for TBSS and VBM
(:mod:`neurofaune.analysis.stats.readout_results`).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from neurofaune.analysis.stats.readout_results import analysis_id, now

logger = logging.getLogger(__name__)

_NOT_COMPUTED = "not computed: no power calculation exists for a cross-validated decoder"

_COLUMNS = {
    "contrast": {"Description": "What is decoded and between which sessions", "Standard": "contrast"},
    "measure": {"Description": "The feature set", "Standard": "measure"},
    "design": {"Description": "delta (within-animal change) or cross_sectional"},
    "tested_direction": {
        "Description": "What the test asks of the score", "Standard": "tested_direction",
    },
    "n": {"Description": "Samples decoded", "Standard": "n"},
    "n_animals": {"Description": "Animals; folds are grouped on these"},
    "n_features": {"Description": "Features before PCA"},
    "n_splits": {"Description": "Cross-validation folds"},
    "negative_class": {"Description": "Class coded 0"},
    "positive_class": {"Description": "Class coded 1"},
    "class_balance": {"Description": "Positive samples / all samples"},
    "balanced_accuracy": {"Description": "Cross-validated balanced accuracy"},
    "roc_auc": {"Description": "Cross-validated area under the ROC curve"},
    "r2": {"Description": "Cross-validated R2"},
    "stat": {
        "Description": "The score tested: balanced accuracy, or R2 for regression",
        "Standard": "stat",
        "StatName": "cross-validated score",
    },
    "null_mean": {"Description": "Mean score under label permutation between animals"},
    "permutations": {"Description": "Label permutations"},
    "p_value": {
        "Description": "Permutation p-value of the score",
        "Standard": "p_value",
        "PKind": "perm",
    },
    "mde_note": {"Description": "Why no minimum detectable effect", "Standard": "mde_note"},
}


def modality_of(feature_set: str) -> str:
    """``roi_stats|msme_common:MWF`` -> ``msme``; morphometry -> ``anat``."""
    if feature_set.startswith("morphometry"):
        return "anat"
    source = feature_set.split("|", 1)[1].split(":", 1)[0]
    return source.split("_", 1)[0]


def _measure_of(feature_set: str) -> str:
    return feature_set.split(":", 1)[1] if ":" in feature_set else feature_set


def _default_run_id() -> str:
    """A run id from the current UTC time, e.g. ``2026-10-06_183248`` (the spec's own)."""
    from neurofaune import results

    if results.AVAILABLE:
        return results.spec.default_run_id()
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _run_folder(root: Path, identifier: str, run_id: str) -> Path:
    """``<root>/<analysis id>/<run id>``: checked by the spec when it is installed."""
    from neurofaune import results

    if results.AVAILABLE:
        return results.spec.run_folder(root, identifier, run_id)
    return Path(root).joinpath(*identifier.split("/"), run_id)


def write_decoding_results(
    root: Path,
    name: str,
    summary: pd.DataFrame,
    n_groups: Mapping[str, int],
    run_id: str | None = None,
    started: str | None = None,
) -> list[Path]:
    """Write ``<root>/<modality>/decoding/<name>/<run>/``; returns the folders written."""
    from neurofaune import results
    from neurofaune.provenance import generated_by

    run_id = run_id or _default_run_id()
    folders = []
    table = summary.assign(modality=summary["feature_set"].map(modality_of))
    for modality, rows in table.groupby("modality"):
        identifier = analysis_id(modality, "decoding", name)
        folder = _run_folder(root, identifier, run_id)
        folder.mkdir(parents=True, exist_ok=False)
        tests = _tests_table(rows)
        tests.to_csv(folder / "tests.csv", index=False)
        folders.append(folder)
        if not results.AVAILABLE:
            logger.warning("Wrote %s; not its column dictionary, analysis.json or "
                           "provenance.json: %s", folder / "tests.csv", results.INSTALL_HINT)
            continue
        from neurofaune.results import provenance as provenance_record
        from neurofaune.results import write_analysis, write_columns

        write_columns(folder / "tests.csv", _COLUMNS)
        write_analysis(
            folder,
            {
                "id": identifier,
                "run": {"id": run_id},
                "title": f"{modality} decoding: {name}",
                "description": (
                    "Cross-validated decoding from ROI-level features, with folds grouped by "
                    "animal and imputation, scaling and PCA fitted inside each fold."
                ),
                "analysis_type": "decoding",
                "modality": modality,
                "measures": sorted(tests["measure"].unique()),
                "role": "exploratory",
                "design": {"n": int(tests["n"].max()), "groups": dict(n_groups)},
                "inference": {
                    "method": "label permutation between animals",
                    "n_permutations": int(tests["permutations"].max()),
                    "correction": {
                        "p_kind": "perm",
                        "family": "each decoder alone",
                        "statement": (
                            f"Permutation p of each decoder; not corrected over the {len(tests)} "
                            "decoders in this table"
                        ),
                    },
                },
                "effect": {
                    "measure": None,
                    "reason": "the cross-validated score in 'stat' is the magnitude; no interval is computed",
                },
                "tables": [{
                    "path": "tests.csv",
                    "role": "tests",
                    "headline": True,
                    "n_rows": len(tests),
                    "rows": "one decoder: a feature set in a contrast",
                    "description": "every decoder run, significant or not",
                }],
            },
            provenance_record(
                generated_by(), status="completed", start=started or now(), end=now()
            ),
        )
    return folders


def _tests_table(rows: pd.DataFrame) -> pd.DataFrame:
    classification = rows["analysis_type"] == "classification"
    score = rows["balanced_accuracy"] if classification.all() else rows["r2"]
    tests = pd.DataFrame({
        "contrast": rows["contrast"],
        "measure": rows["feature_set"].map(_measure_of),
        "design": rows["design"],
        "tested_direction": "score above the permutation null",
        "n": rows["n_samples"],
        "n_animals": rows["n_animals"],
        "n_features": rows["n_features"],
        "n_splits": rows["n_splits"],
    })
    for column in ("negative_class", "positive_class", "class_balance", "balanced_accuracy",
                   "roc_auc", "r2"):
        if column in rows.columns:
            tests[column] = rows[column]
    tests["stat"] = score
    tests["null_mean"] = rows["null_mean"]
    tests["permutations"] = rows["permutations"]
    tests["p_value"] = rows["p_permutation"]
    tests["mde_note"] = _NOT_COMPUTED
    return tests
