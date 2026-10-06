"""Writing an analysis folder (docs/RESULTS_SPEC.md).

The producer writes its tables (pandas ``to_csv`` is fine); this module writes the
rest -- each table's column dictionary, ``analysis.json`` and ``provenance.json`` --
and checks the folder before it returns, so nothing non-conforming is left behind
silently. Standard library only.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .check import Report, check_analysis, read_table
from .spec import ANALYSIS_JSON, PROVENANCE_JSON, SPEC, SPEC_VERSION


class NonConformingResults(ValueError):
    """An analysis folder that does not meet the specification."""

    def __init__(self, report: Report):
        self.report = report
        super().__init__(f"{report.folder}: " + "; ".join(report.errors))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def columns_for(columns: Iterable[str], master: Mapping[str, Mapping[str, Any]],
                extra: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict]:
    """The dictionary entries for exactly ``columns``, from ``master`` then ``extra``.

    A column with no entry anywhere is an error: a table must not carry a column
    nobody described.
    """
    extra = extra or {}
    out, missing = {}, []
    for c in columns:
        entry = extra.get(c, master.get(c))
        if entry is None:
            missing.append(c)
        else:
            out[c] = dict(entry)
    if missing:
        raise KeyError(f"no column dictionary entry for {missing}")
    return out


def write_columns(table: Path, master: Mapping[str, Mapping[str, Any]],
                  extra: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """Write ``<table stem>.json`` for the columns ``table`` actually has."""
    table = Path(table)
    cols, _rows = read_table(table)
    out = table.with_suffix(".json")
    out.write_text(json.dumps(columns_for(cols, master, extra), indent=2) + "\n")
    return out


def provenance(generated_by: list[dict], *, status: str, start: str, end: str | None = None,
               run_id: str | None = None, inputs: list[dict] | None = None,
               subjects: dict | None = None, settings: dict | None = None) -> dict:
    """A ``provenance.json`` body; ``generated_by`` is BIDS GeneratedBy, producer first."""
    rec: dict[str, Any] = {
        "spec": SPEC, "spec_version": SPEC_VERSION, "generated_by": generated_by,
        "run": {"status": status, "start": start, "end": end},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    if run_id:
        rec["run"]["id"] = run_id
    for key, value in (("inputs", inputs), ("subjects", subjects), ("settings", settings)):
        if value is not None:
            rec[key] = value
    return rec


def write_analysis(folder: Path, analysis: Mapping[str, Any], provenance_record: Mapping[str, Any],
                   *, strict: bool = True) -> Report:
    """Write ``analysis.json`` and ``provenance.json`` into ``folder`` and check it.

    ``spec`` / ``spec_version`` are filled in. With ``strict`` a non-conforming
    folder raises :class:`NonConformingResults` (the files stay on disk, so the
    problem can be inspected); otherwise the report is returned either way.
    """
    folder = Path(folder)
    a = {"spec": SPEC, "spec_version": SPEC_VERSION, **analysis}
    p = {"spec": SPEC, "spec_version": SPEC_VERSION, **provenance_record}
    (folder / ANALYSIS_JSON).write_text(json.dumps(a, indent=2, default=str) + "\n")
    (folder / PROVENANCE_JSON).write_text(json.dumps(p, indent=2, default=str) + "\n")
    report = check_analysis(folder)
    if strict and not report.ok:
        raise NonConformingResults(report)
    return report
