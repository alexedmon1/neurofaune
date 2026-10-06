"""Conformance checker for the results specification (docs/RESULTS_SPEC.md §7).

``check(path)`` finds every analysis folder under ``path`` and returns one
:class:`Report` per folder: the JSON files against the schemas, every listed file
present and inside the folder, every table against its column dictionary, the
standard terms and their qualifiers, and the reporting contract (§6).

Standard library only.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ._schema import load_schema, validate
from .spec import (ANALYSIS_JSON, CONTRACT_ROLES, MODALITIES, PER_SUBGROUP, PROVENANCE_JSON, READS,
                   SPEC, SPEC_VERSION, STANDARD_TERMS, TABLE_SUFFIXES, canonical_measure, id_problem,
                   measure_vocabulary, run_id_problem, valid_axes, version_tuple)

#: Analysis types whose tests are voxelwise, so a test row must state its extent.
VOXELWISE = ("tbss", "vbm", "tbm", "voxelwise", "fixel")
_TRUE, _FALSE = {"true", "True", "TRUE", "1"}, {"false", "False", "FALSE", "0"}


@dataclass
class Report:
    folder: Path
    id: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: what the cross-run check needs: version, modality, analysis_type, run, supersedes, tests
    meta: dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def run(self) -> str | None:
        return self.meta.get("run")

    def as_dict(self) -> dict:
        return {"folder": str(self.folder), "id": self.id, "run": self.run, "ok": self.ok,
                "errors": self.errors, "warnings": self.warnings}


# ----------------------------------------------------------------- finding ---
def is_analysis(folder: Path) -> bool:
    f = Path(folder) / ANALYSIS_JSON
    if not f.is_file():
        return False
    try:
        return json.loads(f.read_text()).get("spec") == SPEC
    except (OSError, ValueError, AttributeError):
        return False


def find_analyses(root: Path) -> list[Path]:
    """Every analysis folder at or under ``root``, sorted."""
    root = Path(root)
    if is_analysis(root):
        return [root]
    return sorted(p.parent for p in root.rglob(ANALYSIS_JSON) if is_analysis(p.parent))


# ------------------------------------------------------------------ helpers ---
def _load_json(path: Path, rep: Report) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        rep.errors.append(f"{path.name}: missing")
        return None
    except (OSError, ValueError) as exc:
        rep.errors.append(f"{path.name}: not valid JSON ({exc})")
        return None
    if not isinstance(value, dict):
        rep.errors.append(f"{path.name}: must be a JSON object")
        return None
    return value


def _inside(folder: Path, rel: str, what: str, rep: Report) -> Path | None:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or "\\" in rel:
        rep.errors.append(f"{what} {rel!r}: paths must be relative and stay inside the folder")
        return None
    full = folder / rel
    if not full.exists():
        rep.errors.append(f"{what} {rel!r}: listed but not on disk")
        return None
    return full


def _version_ok(version: str, rep: Report) -> tuple[int, int] | None:
    """The folder's (major, minor) when this checker reads it, else None (and an error)."""
    got = version_tuple(version)
    if got is None:
        rep.errors.append(f"spec_version {version!r} is not semver")
        return None
    if got not in READS:
        rep.errors.append(f"spec_version {version} is not readable by checker {SPEC_VERSION}")
        return None
    return got


def _rules_0_2(analysis: dict, rep: Report) -> None:
    """What 0.2 adds (RESULTS_SPEC.md §3.1): identity and vocabulary."""
    modality, atype = analysis.get("modality"), analysis.get("analysis_type")
    if modality is None:
        rep.errors.append(f"{ANALYSIS_JSON}: modality is required from 0.2 (one of {', '.join(MODALITIES)})")
    elif modality not in MODALITIES:
        rep.errors.append(f"{ANALYSIS_JSON}: modality {modality!r} is not one of {', '.join(MODALITIES)}")
    mods = analysis.get("modalities")
    if modality == "multimodal":
        if not mods:
            rep.errors.append(f"{ANALYSIS_JSON}: a multimodal analysis lists its modalities")
        else:
            bad = [m for m in mods if m not in MODALITIES or m == "multimodal"]
            if bad:
                rep.errors.append(f"{ANALYSIS_JSON}: modalities {bad} are not modalities")
    elif mods:
        rep.errors.append(f"{ANALYSIS_JSON}: modalities is for multimodal analyses only")
    if isinstance(analysis.get("id"), str) and isinstance(modality, str) and isinstance(atype, str):
        why = id_problem(analysis["id"], modality, atype)
        if why:
            rep.errors.append(f"{ANALYSIS_JSON}: id {analysis['id']!r} {why}")
    run = analysis.get("run")
    if not isinstance(run, dict) or not run.get("id"):
        rep.errors.append(f"{ANALYSIS_JSON}: run is required from 0.2 ({{id, label, supersedes}}; §3.2)")
    else:
        why = run_id_problem(run["id"])
        if why:
            rep.errors.append(f"{ANALYSIS_JSON}: run id {run['id']!r} {why}")
        if run["id"] in (run.get("supersedes") or []):
            rep.errors.append(f"{ANALYSIS_JSON}: run {run['id']!r} supersedes itself")
    vocab = measure_vocabulary()
    own = set(mods or []) if modality == "multimodal" else {modality}
    for m in analysis.get("measures") or []:
        canon, exact = canonical_measure(m, vocab)
        if canon is None:
            rep.warnings.append(f"{ANALYSIS_JSON}: measure {m!r} is not in the measure vocabulary "
                                "(neurofaune/results/vocab/measures.json) -- add it there")
        elif not exact:
            rep.errors.append(f"{ANALYSIS_JSON}: measure {m!r} is written {canon!r} in the vocabulary")
        elif vocab[canon]["modality"] not in own:
            rep.warnings.append(f"{ANALYSIS_JSON}: measure {m!r} is a {vocab[canon]['modality']} measure, "
                                f"in a {modality} analysis")


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Header and rows of a spec table (CSV or TSV by extension)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=TABLE_SUFFIXES[path.suffix.lower()])
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _in_unit(v: str) -> bool:
    x = _num(v)
    return x is not None and 0 <= x <= 1


# ------------------------------------------------------------------ tables ---
def _check_table(folder: Path, entry: dict, analysis: dict, rep: Report) -> tuple[str, dict, list] | None:
    rel = entry.get("path", "")
    path = _inside(folder, rel, "table", rep)
    if path is None:
        return None
    if path.suffix.lower() not in TABLE_SUFFIXES:
        rep.errors.append(f"table {rel!r}: must be .csv or .tsv")
        return None
    where = f"table {rel!r}"
    cols, rows = read_table(path)
    if not cols:
        rep.errors.append(f"{where}: no header row")
        return None
    dict_path = path.with_suffix(".json")
    dictionary = _load_json(dict_path, rep) if dict_path.exists() else None
    if dictionary is None:
        if not dict_path.exists():
            rep.errors.append(f"{where}: no column dictionary {dict_path.name} beside it")
        return None
    rep.errors += [f"{dict_path.name}{e[1:]}" for e in validate(dictionary, load_schema("columns"))]
    missing = [c for c in cols if c not in dictionary]
    absent = [c for c in dictionary if c not in cols]
    if missing:
        rep.errors.append(f"{where}: columns without a dictionary entry: {missing}")
    if absent:
        rep.errors.append(f"{dict_path.name}: entries for columns the table does not have: {absent}")

    std: dict[str, str] = {}
    for col, meta in dictionary.items():
        term = meta.get("Standard") if isinstance(meta, dict) else None
        if not term:
            continue
        if term in std and term not in PER_SUBGROUP:
            rep.errors.append(f"{dict_path.name}: standard term {term!r} claimed by both "
                              f"{std[term]!r} and {col!r}")
            continue
        for q in STANDARD_TERMS.get(term, ()):
            if not meta.get(q):
                rep.errors.append(f"{dict_path.name}: {col!r} is {term!r} and must state {q}")
        if term in PER_SUBGROUP:                     # once per subgroup, not once per table
            key = f"{term}:{meta.get('Subgroup')}"
            if key in std:
                rep.errors.append(f"{dict_path.name}: {term!r} for subgroup {meta.get('Subgroup')!r} "
                                  f"claimed by both {std[key]!r} and {col!r}")
            std[key] = col
            continue
        std[term] = col

    if "n_rows" in entry and entry["n_rows"] != len(rows):
        rep.errors.append(f"{where}: {len(rows)} rows, analysis.json says {entry['n_rows']}")
    for term in ("p_value", "frac_significant"):
        col = std.get(term)
        if col and col in cols:
            bad = [r[col] for r in rows if r[col] != "" and not _in_unit(r[col])]
            if bad:
                rep.errors.append(f"{where}: {col!r} ({term}) has values outside [0, 1]: {bad[:3]}")
    for term in ("significant", "effect_selected", "crosses_midline"):
        col = std.get(term)
        if col and col in cols:
            bad = [r[col] for r in rows if r[col] not in _TRUE | _FALSE | {""}]
            if bad:
                rep.errors.append(f"{where}: {col!r} ({term}) is not true/false: {bad[:3]}")
    if entry.get("role") in CONTRACT_ROLES:
        _contract(entry["role"], std, rows, analysis, where, rep)
    return entry.get("role", ""), std, rows


def _contract(role: str, std: dict, rows: list, analysis: dict, where: str, rep: Report) -> None:
    """The reporting contract, §6."""
    def need(*terms, why):
        if not any(t in std for t in terms):
            rep.errors.append(f"{where} ({role}): {why} -- needs a column marked "
                              + " or ".join(repr(t) for t in terms))

    need("contrast", why="which test")
    if len(analysis.get("measures") or []) > 1:
        need("measure", why="which measure (the analysis has several)")
    has_effect = (analysis.get("effect") or {}).get("measure") is not None

    if role == "tests":
        if "tested_direction" not in std and not ("test_kind" in std and "group_a" in std
                                                    and "group_b" in std):
            rep.errors.append(f"{where} (tests): direction -- needs 'tested_direction', or "
                              "'test_kind' with 'group_a' and 'group_b'")
        if has_effect:
            need("effect_size", why="magnitude")
            if "effect_size" in std:
                need("effect_ci_low", why="uncertainty of the effect")
                need("effect_ci_high", why="uncertainty of the effect")
                blank = sum(1 for r in rows if r.get(std["effect_size"], "") == "")
                if blank:
                    rep.warnings.append(f"{where}: {blank} test row(s) with no effect size")
                lo, hi = std.get("effect_ci_low"), std.get("effect_ci_high")
                if lo and hi:
                    no_ci = sum(1 for r in rows if r.get(std["effect_size"], "") != ""
                                and (r.get(lo, "") == "" or r.get(hi, "") == ""))
                    if no_ci:
                        rep.errors.append(f"{where} (tests): {no_ci} row(s) give an effect without its "
                                          "interval -- the columns exist but the values are blank")
        if "n" not in std and not ("n_a" in std and "n_b" in std):
            rep.errors.append(f"{where} (tests): sample size -- needs 'n', or 'n_a' and 'n_b'")
        need("p_value", why="significance")
        if analysis.get("analysis_type") in VOXELWISE:
            need("n_significant", "frac_significant", why="extent of a voxelwise test")
    elif role == "clusters":
        need("n_voxels", "volume_mm3", why="extent")
        need("peak_xyz_mm", why="location")
        need("peak_region", why="named location")
        need("stat", "effect_size", why="magnitude")
        if "effect_size" in std:
            need("effect_selected", why="cluster effects are selected; say so")
        need("p_value", why="significance")
    elif role == "elements":
        need("element", why="which element")
        need("effect_size", "stat", why="magnitude")
        need("p_value", why="significance")


# ----------------------------------------------------------------- analysis ---
def check_analysis(folder: Path) -> Report:
    folder = Path(folder)
    rep = Report(folder)
    analysis = _load_json(folder / ANALYSIS_JSON, rep)
    if analysis is None:
        return rep
    rep.id = analysis.get("id")
    rep.errors += [f"{ANALYSIS_JSON}{e[1:]}" for e in validate(analysis, load_schema("analysis"))]
    known = set(load_schema("analysis")["properties"])
    unknown = sorted(set(analysis) - known)
    if unknown:
        rep.warnings.append(f"{ANALYSIS_JSON}: fields this checker does not know: {unknown}")
    version = _version_ok(analysis["spec_version"], rep) if isinstance(analysis.get("spec_version"), str) else None
    v02 = version is not None and version >= (0, 2)
    if v02:
        _rules_0_2(analysis, rep)

    prov = _load_json(folder / PROVENANCE_JSON, rep)
    if prov is not None:
        rep.errors += [f"{PROVENANCE_JSON}{e[1:]}" for e in validate(prov, load_schema("provenance"))]
        status = (prov.get("run") or {}).get("status")
        if status and status != "completed":
            rep.warnings.append(f"run status is {status!r}, not completed")

    effect = analysis.get("effect") or {}
    if isinstance(effect, dict):
        if effect.get("measure") is None and not effect.get("reason"):
            rep.errors.append(f"{ANALYSIS_JSON}: effect.measure is null without a reason")
        if effect.get("measure") is not None and not effect.get("definition"):
            rep.errors.append(f"{ANALYSIS_JSON}: effect.definition is required with a measure")

    tables = [t for t in analysis.get("tables") or [] if isinstance(t, dict)]
    if sum(1 for t in tables if t.get("headline")) > 1:
        rep.errors.append(f"{ANALYSIS_JSON}: more than one headline table")
    tested: set[tuple[str, str]] = set()
    listed = set(analysis.get("measures") or [])
    for entry in tables:
        got = _check_table(folder, entry, analysis, rep)
        if v02 and got and got[0] in CONTRACT_ROLES and "measure" in got[1] and listed:
            # 0.2: a table names its measures as analysis.json lists them
            col = got[1]["measure"]
            unlisted = sorted({r.get(col, "") for r in got[2]} - listed - {""})
            if unlisted:
                rep.errors.append(f"table {entry.get('path')!r}: measures {unlisted} are not in "
                                  "analysis.json's measures")
        if got and got[0] == "tests":
            _role, std, rows = got
            mcol, ccol, fcol = std.get("measure"), std.get("contrast"), std.get("facet")
            if ccol:
                tested |= {(r.get(mcol, "") if mcol else "", r.get(ccol, ""),
                            r.get(fcol, "") if fcol else "") for r in rows}
    for item in analysis.get("maps") or []:
        if isinstance(item, dict) and "axes" in item and not valid_axes(item["axes"]):
            rep.errors.append(f"map {item.get('path')!r}: axes {item['axes']!r} is not three letters, "
                              "one from each of R/L, A/P, S/I")
        if isinstance(item, dict) and item.get("kind") in ("p_corrected", "p_uncorrected") \
                and "values" not in item:
            rep.warnings.append(f"map {item.get('path')!r}: a p map should say whether it holds "
                                "p or 1 - p ('values')")
    for kind in ("maps", "figures"):
        for item in analysis.get(kind) or []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            _inside(folder, item["path"], kind[:-1], rep)
            if tested and item.get("contrast"):
                key = (item.get("measure", ""), item["contrast"], item.get("facet", ""))
                if key not in tested and ("", item["contrast"], item.get("facet", "")) not in tested:
                    rep.warnings.append(f"{kind[:-1]} {item['path']!r}: names a test with no "
                                        f"row in the tests table: {key}")
    for rec in (analysis.get("design") or {}).get("records") or []:
        if isinstance(rec, str):
            _inside(folder, rec, "design record", rep)
    nested = sorted(p.parent for p in folder.rglob(ANALYSIS_JSON)
                    if p.parent != folder and is_analysis(p.parent))
    if nested:
        rep.errors.append(f"analysis folders must not nest: {[str(p) for p in nested[:3]]}")
    run = analysis.get("run") if isinstance(analysis.get("run"), dict) else {}
    rep.meta = {"v02": v02, "modality": analysis.get("modality"), "analysis_type": analysis.get("analysis_type"),
                "run": run.get("id"), "supersedes": list(run.get("supersedes") or []), "tests": tested}
    return rep


def check_runs(reports: list[Report]) -> None:
    """Across folders (§3.2): an id names one analysis; each of its folders is one run.

    Runs of one id share modality and analysis_type and have distinct run ids; a run
    supersedes only runs of its own id; a test (measure, contrast, facet) in two runs is
    an error unless the later run lists the earlier in ``supersedes`` -- then the earlier
    run's copy of that test is superseded, and its other tests stay current. 0.1 folders
    have no runs: their ids must be unique. Errors are added to the reports in place.
    """
    from itertools import combinations

    by_id: dict[str, list[Report]] = {}
    for r in reports:
        if r.id:
            by_id.setdefault(r.id, []).append(r)
    for aid, reps in by_id.items():
        if len(reps) < 2:
            continue
        if not all(r.meta.get("v02") for r in reps):
            for r in reps:
                r.errors.append(f"id {aid!r} is used by {len(reps)} folders; only 0.2 folders, each one "
                                "run (§3.2), may share an id")
            continue
        for key in ("modality", "analysis_type"):
            values = {r.meta.get(key) for r in reps}
            if len(values) > 1:
                for r in reps:
                    r.errors.append(f"runs of {aid!r} disagree on {key}: {sorted(map(str, values))}")
        runs: dict[str, Report] = {}
        for r in reps:
            if r.run in runs:
                for x in (r, runs[r.run]):
                    x.errors.append(f"run {r.run!r} of {aid!r} is in two folders: {runs[r.run].folder}, {r.folder}")
            elif r.run:
                runs[r.run] = r
        superseded: set[tuple[str, tuple]] = set()
        for r in reps:
            for old in r.meta.get("supersedes") or []:
                if old not in runs:
                    r.errors.append(f"run {r.run!r} supersedes {old!r}, which is not a run of {aid!r}")
                    continue
                if r.run in (runs[old].meta.get("supersedes") or []):
                    r.errors.append(f"runs {r.run!r} and {old!r} of {aid!r} supersede each other")
                    continue
                overlap = r.meta["tests"] & runs[old].meta["tests"]
                if not overlap:
                    r.warnings.append(f"run {r.run!r} supersedes {old!r} but repeats none of its tests")
                superseded |= {(old, k) for k in overlap}
        for a, b in combinations([r for r in reps if r.run], 2):
            clash = sorted(k for k in a.meta["tests"] & b.meta["tests"]
                           if (a.run, k) not in superseded and (b.run, k) not in superseded)
            if clash:
                for x in (a, b):
                    x.errors.append(f"runs {a.run!r} and {b.run!r} of {aid!r} both hold {len(clash)} test(s), "
                                    f"e.g. {clash[0]}; the later run must list the earlier in supersedes")


def check(path: Path) -> list[Report]:
    """One report per analysis folder at or under ``path`` (empty when there are none),
    the runs of each analysis checked against each other (``check_runs``)."""
    reports = [check_analysis(f) for f in find_analyses(Path(path))]
    check_runs(reports)
    return reports
