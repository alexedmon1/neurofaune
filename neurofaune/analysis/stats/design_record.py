"""What a randomise design tests, written beside it: ``design.json`` + ``design.md``.

A ``design.mat`` / ``design.con`` pair is a matrix of numbers. Which subject each
row is, what each column codes, and what each contrast asks are decided when the
design is built and then live only in the code that built it -- or in someone's
memory. A results folder that cannot say what its contrasts test cannot be read,
reported, or rebuilt, and the question always comes up after the code has moved
on.

So every design carries a **design record**: ``design.json`` (machine-readable,
schema ``neurofaune.design/1``) and ``design.md`` (the same, as a README a person
reads). The record names every row (subject), says what every column is, and
for every contrast gives its vector, a plain sentence of what it tests, and its
test kind (two-group, one-sample, regression, ...). It is checked against the
matrices it describes -- a record that disagrees with its ``design.mat`` or
``design.con`` is refused, because a wrong description is worse than none.

Three ways to get one:

- :func:`write_design` -- build the matrices *and* the record in one call (for
  designs built by hand, e.g. a study's own orchestration);
- :func:`write_design_record` -- describe matrices that already exist (a
  ``neuroaider.DesignHelper.describe()`` dict, or a backfill);
- :func:`attach_design` -- what :func:`~neurofaune.analysis.stats.randomise_wrapper.run_randomise`
  calls: copy the design and its record into the run's output folder, so every
  results folder explains itself.

Schema ``neurofaune.design/1`` (fields marked * are required)::

    schema*        "neurofaune.design/1"
    summary        one sentence: what the analysis asks
    n*             number of rows (observations) in design.mat
    rows*          {"ids": [...], "file": "subject_order.txt"} -- row i is ids[i];
                   "label_ids": [...] only when the design's labels were deliberately
                   decoupled from the data (a null / shuffled run): row i holds the
                   data of ids[i] but carries the group / covariate values of
                   label_ids[i]
    data           {"file": "all_FA.nii.gz", "meaning": "..."} -- what each row's
                   observation is (the 4D input to randomise), e.g. "per-animal FA
                   change, p90 minus p60, on the TBSS skeleton"
    groups         {group: n} -- how many rows each group has
    columns*       [{index, name, meaning, kind}] -- one per design.mat column;
                   kind: intercept | group | covariate | nuisance | interaction
    contrasts*     [{index, name, vector, tests, test_kind, group_a, group_b}]
                   test_kind: two_group | one_sample | regression | interaction | custom;
                   two_group names group_a (higher when the statistic is positive)
                   and group_b
    ftests         [{index, name, contrasts: [contrast indices], tests}]
    inference      free-text notes on the permutation scheme (sign-flipping,
                   exchangeability blocks, ...)
    notes          [free text]
    written_by     {Name, Version, CommitID} of the writer
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA = "neurofaune.design/1"
RECORD_JSON = "design.json"
RECORD_MD = "design.md"
TEST_KINDS = ("two_group", "one_sample", "regression", "interaction", "custom")
COLUMN_KINDS = ("intercept", "group", "covariate", "nuisance", "interaction")

#: Meanings that say nothing (neuroaider's fallback, empty strings, ...).
_PLACEHOLDERS = {"", "design column", "column", "unknown", "todo", "n/a", "na"}

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)


_CONTRAST_NAME = re.compile(r"^/ContrastName(\d+)\s+(.*)$")


class DesignRecordError(ValueError):
    """A design record that is missing, incomplete, or contradicts its matrices."""


# --------------------------------------------------------------------------- #
# FSL VEST files
# --------------------------------------------------------------------------- #
def read_vest(path: PathLike) -> Tuple[np.ndarray, Optional[List[str]]]:
    """A VEST matrix (``.mat`` / ``.con`` / ``.fts``) and its ``/ContrastName``
    names, if it has them."""
    names: Dict[int, str] = {}
    rows: List[List[float]] = []
    in_matrix = False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        named = _CONTRAST_NAME.match(line)
        if named:                                   # FSL writes a tab; others a space
            names[int(named.group(1))] = named.group(2).strip()
        elif line.startswith("/Matrix"):
            in_matrix = True
        elif line.startswith("/"):
            continue
        elif in_matrix:
            rows.append([float(v) for v in line.split()])
    matrix = np.array(rows, dtype=float) if rows else np.zeros((0, 0))
    ordered = [names[i] for i in sorted(names)] if names else None
    return matrix, ordered


def write_vest(path: PathLike, matrix: np.ndarray, kind: str,
               names: Optional[Sequence[str]] = None) -> Path:
    """Write a VEST file: ``kind`` is ``mat``, ``con`` or ``fts``. Contrast files
    carry their names as ``/ContrastName`` lines."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    path = Path(path)
    lines = []
    if names and kind in ("con", "fts"):
        lines += [f"/ContrastName{i + 1} {n}" for i, n in enumerate(names)]
    lines.append(f"/NumWaves {matrix.shape[1]}")
    lines.append(f"/{'NumPoints' if kind == 'mat' else 'NumContrasts'} {matrix.shape[0]}")
    lines.append("/Matrix")
    fmt = "%d" if kind == "fts" else "%.6g"
    lines += [" ".join(fmt % v for v in row) for row in matrix]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Building a record
# --------------------------------------------------------------------------- #
def _column(i: int, col: Union[Dict[str, Any], Sequence]) -> Dict[str, Any]:
    if isinstance(col, dict):
        out = {"index": i, "name": col.get("name"), "meaning": col.get("meaning")}
        if col.get("kind"):
            out["kind"] = col["kind"]
        return out
    name, meaning, *kind = col
    out = {"index": i, "name": name, "meaning": meaning}
    if kind and kind[0]:
        out["kind"] = kind[0]
    return out


def describe_design(
    columns: Sequence[Union[Dict[str, Any], Sequence]],
    contrasts: Sequence[Dict[str, Any]],
    *,
    rows: Optional[Sequence[str]] = None,
    rows_file: Optional[str] = None,
    label_ids: Optional[Sequence[str]] = None,
    data: Optional[Dict[str, str]] = None,
    groups: Optional[Dict[str, int]] = None,
    summary: Optional[str] = None,
    ftests: Optional[Sequence[Dict[str, Any]]] = None,
    inference: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    n: Optional[int] = None,
) -> Dict[str, Any]:
    """A design record from plain Python values.

    ``columns`` are ``(name, meaning[, kind])`` tuples or dicts, in design.mat
    order. ``contrasts`` are dicts with ``name``, ``vector``, ``tests`` (a
    sentence), ``test_kind`` and, for a two-group test, ``group_a`` (higher when
    the statistic is positive) and ``group_b``. ``rows`` are the subject ids of
    design.mat's rows, in order; ``label_ids``, only for a deliberately shuffled
    (null) run, the subject whose labels each row carries. ``data`` is
    ``{"file", "meaning"}``: what each row's observation is. ``ftests`` are
    dicts with ``name``, ``contrasts`` (1-based contrast indices) and ``tests``.
    """
    record: Dict[str, Any] = {"schema": SCHEMA}
    if summary:
        record["summary"] = summary
    record["n"] = int(n if n is not None else (len(rows) if rows is not None else 0))
    record["rows"] = {}
    if rows is not None:
        record["rows"]["ids"] = [str(r) for r in rows]
    if rows_file:
        record["rows"]["file"] = rows_file
    if label_ids is not None:
        record["rows"]["label_ids"] = [str(r) for r in label_ids]
    if data:
        record["data"] = {k: str(v) for k, v in data.items() if v is not None}
    if groups:
        record["groups"] = {str(k): int(v) for k, v in groups.items()}
    record["columns"] = [_column(i + 1, c) for i, c in enumerate(columns)]
    record["contrasts"] = []
    for i, c in enumerate(contrasts):
        entry = {"index": i + 1, "name": c.get("name"),
                 "vector": [float(v) for v in c.get("vector", [])],
                 "tests": c.get("tests"), "test_kind": c.get("test_kind")}
        for key in ("group_a", "group_b"):
            if c.get(key) is not None:
                entry[key] = str(c[key])
        record["contrasts"].append(entry)
    if ftests:
        record["ftests"] = [{"index": i + 1, "name": f.get("name"),
                             "contrasts": [int(x) for x in f.get("contrasts", [])],
                             "tests": f.get("tests")}
                            for i, f in enumerate(ftests)]
    if inference:
        record["inference"] = inference
    if notes:
        record["notes"] = [str(x) for x in notes]
    return record


# --------------------------------------------------------------------------- #
# Checking a record
# --------------------------------------------------------------------------- #
def _vague(text: Any) -> bool:
    return not isinstance(text, str) or text.strip().lower() in _PLACEHOLDERS


def validate_design_record(
    record: Dict[str, Any],
    design_mat: Optional[PathLike] = None,
    design_con: Optional[PathLike] = None,
    design_fts: Optional[PathLike] = None,
) -> List[str]:
    """Everything wrong with ``record``, and with it against the matrix files
    given; an empty list means the record is complete and agrees with them."""
    problems: List[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}, expected {SCHEMA!r}")
    columns = record.get("columns") or []
    contrasts = record.get("contrasts") or []
    n = record.get("n")
    if not columns:
        problems.append("no columns described")
    if not contrasts:
        problems.append("no contrasts described")
    ids = (record.get("rows") or {}).get("ids")
    if not ids and not (record.get("rows") or {}).get("file"):
        problems.append("rows: neither the subject ids nor the file listing them in order")
    if ids is not None and n is not None and len(ids) != n:
        problems.append(f"rows: {len(ids)} ids for n = {n}")
    label_ids = (record.get("rows") or {}).get("label_ids")
    if label_ids is not None and n is not None and len(label_ids) != n:
        problems.append(f"rows: {len(label_ids)} label_ids for n = {n}")
    if record.get("data") is not None and _vague(record["data"].get("meaning")):
        problems.append("data: no meaning -- say what each row's observation is")
    if record.get("groups") and n is not None and sum(record["groups"].values()) != n:
        problems.append(f"groups add up to {sum(record['groups'].values())}, not n = {n}")

    for i, col in enumerate(columns, 1):
        if col.get("index") != i:
            problems.append(f"column {i}: index is {col.get('index')}")
        if _vague(col.get("name")):
            problems.append(f"column {i}: no name")
        if _vague(col.get("meaning")):
            problems.append(f"column {i} ({col.get('name')}): no meaning -- say what it codes")
        if col.get("kind") is not None and col["kind"] not in COLUMN_KINDS:
            problems.append(f"column {i}: kind {col['kind']!r} not one of {COLUMN_KINDS}")

    names = [c.get("name") for c in contrasts]
    if len(set(names)) != len(names):
        problems.append(f"contrast names repeat: {names}")
    for i, con in enumerate(contrasts, 1):
        label = f"contrast {i} ({con.get('name')})"
        if con.get("index") != i:
            problems.append(f"{label}: index is {con.get('index')}")
        if _vague(con.get("name")):
            problems.append(f"contrast {i}: no name")
        if len(con.get("vector") or []) != len(columns):
            problems.append(f"{label}: vector has {len(con.get('vector') or [])} weights "
                            f"for {len(columns)} columns")
        if _vague(con.get("tests")):
            problems.append(f"{label}: no sentence saying what it tests")
        if con.get("test_kind") not in TEST_KINDS:
            problems.append(f"{label}: test_kind {con.get('test_kind')!r} not one of {TEST_KINDS}")
        if con.get("test_kind") == "two_group" and not (con.get("group_a") and con.get("group_b")):
            problems.append(f"{label}: a two-group test names group_a and group_b")

    for i, ft in enumerate(record.get("ftests") or [], 1):
        bad = [x for x in ft.get("contrasts", []) if not 1 <= x <= len(contrasts)]
        if bad or not ft.get("contrasts"):
            problems.append(f"F-test {i} ({ft.get('name')}): contrasts {ft.get('contrasts')} "
                            f"not in 1..{len(contrasts)}")
        if _vague(ft.get("tests")):
            problems.append(f"F-test {i} ({ft.get('name')}): no sentence saying what it tests")

    if design_mat is not None:
        mat, _ = read_vest(design_mat)
        if mat.shape != (n, len(columns)):
            problems.append(f"{Path(design_mat).name} is {mat.shape[0]} x {mat.shape[1]}; "
                            f"the record describes {n} x {len(columns)}")
    if design_con is not None:
        con, con_names = read_vest(design_con)
        vectors = np.array([c.get("vector") or [] for c in contrasts], dtype=float)
        if con.shape != vectors.shape:
            problems.append(f"{Path(design_con).name} is {con.shape[0]} x {con.shape[1]}; "
                            f"the record has {vectors.shape[0]} contrasts of {vectors.shape[1] if vectors.ndim == 2 else 0}")
        elif not np.allclose(con, vectors, atol=1e-6):
            rows_off = [i + 1 for i in range(len(con)) if not np.allclose(con[i], vectors[i], atol=1e-6)]
            problems.append(f"{Path(design_con).name}: contrast(s) {rows_off} differ from the record")
        if con_names and con_names != names:
            problems.append(f"{Path(design_con).name} names its contrasts {con_names}; the record {names}")
    if design_fts is not None:
        fts, _ = read_vest(design_fts)
        listed = record.get("ftests") or []
        expected = np.zeros((len(listed), len(contrasts)))
        for i, ft in enumerate(listed):
            for x in ft.get("contrasts", []):
                if 1 <= x <= len(contrasts):
                    expected[i, x - 1] = 1
        if fts.shape != expected.shape or not np.allclose(fts, expected):
            problems.append(f"{Path(design_fts).name} does not match the record's F-tests")
    return problems


def check_design_record(record: Dict[str, Any], **files) -> None:
    """Raise :class:`DesignRecordError` listing every problem, if there are any."""
    problems = validate_design_record(record, **files)
    if problems:
        raise DesignRecordError("design record problems:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# The README
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    return f"{v + 0.0:g}"                        # never "-0"


def render_design_markdown(record: Dict[str, Any]) -> str:
    """``design.md``: the record as a README."""
    cols = record.get("columns") or []
    out = ["# What this randomise design tests", ""]
    if record.get("summary"):
        out += [record["summary"], ""]
    groups = record.get("groups") or {}
    gtxt = (" (" + "; ".join(f"{g}: {n}" for g, n in groups.items()) + ")") if groups else ""
    rows = record.get("rows") or {}
    where = ("listed in order in `design.json` (`rows.ids`)" if rows.get("ids") else "")
    if rows.get("file"):
        where += (" and " if where else "listed in order in ") + f"`{rows['file']}`"
    if rows.get("label_ids"):
        out += [f"**{record.get('n')} rows**{gtxt}. **The labels are deliberately shuffled "
                f"against the data (a null run):** row *i* holds the data of `rows.ids[i]` but "
                f"carries the design values of `rows.label_ids[i]` (both in `design.json`).", ""]
    else:
        out += [f"**{record.get('n')} rows**{gtxt}, one per observation; row *i* of `design.mat` "
                f"is the *i*-th subject {where}.", ""]
    data = record.get("data") or {}
    if data.get("meaning"):
        out += [f"**Each row's observation:** {data['meaning']}"
                + (f" (`{data['file']}`)" if data.get("file") else "") + ".", ""]

    out += [f"## Design matrix — `design.mat`, {record.get('n')} × {len(cols)}", "",
            "| # | Column | What it codes |", "|---|---|---|"]
    out += [f"| {c['index']} | `{c['name']}` | {c['meaning']}"
            + (f" *({c['kind']})*" if c.get("kind") else "") + " |" for c in cols]

    out += ["", "## Contrasts — `design.con`", "",
            "Each is a t-test; randomise's `tstat<i>` / `*_tstat<i>` maps are contrast *i*. "
            "A positive statistic means what the contrast says.", "",
            "| # | Name | Tests | Kind | Vector |", "|---|---|---|---|---|"]
    for c in record.get("contrasts") or []:
        kind = c.get("test_kind", "")
        if kind == "two_group":
            kind += f" ({c.get('group_a')} vs {c.get('group_b')})"
        vec = "[" + ", ".join(_fmt(v) for v in c.get("vector") or []) + "]"
        out.append(f"| {c['index']} | `{c['name']}` | {c.get('tests')} | {kind} | `{vec}` |")

    if record.get("ftests"):
        out += ["", "## F-tests — `design.fts`", "", "| # | Name | Combines contrasts | Tests |",
                "|---|---|---|---|"]
        for f in record["ftests"]:
            out.append(f"| {f['index']} | `{f['name']}` | {', '.join(str(x) for x in f['contrasts'])} "
                       f"| {f.get('tests')} |")
    if record.get("inference"):
        out += ["", "## Inference", "", record["inference"]]
    if record.get("notes"):
        out += ["", "## Notes", ""] + [f"- {x}" for x in record["notes"]]
    wb = record.get("written_by") or {}
    stamp = " ".join(str(x) for x in (wb.get("Name"), wb.get("Version"),
                                      (wb.get("CommitID") or "")[:7]) if x)
    out += ["", f"_Schema `{record.get('schema')}`"
            + (f"; written by {stamp}" if stamp else "")
            + (f" on {record['written']}" if record.get("written") else "") + "._", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _stamp(record: Dict[str, Any]) -> Dict[str, Any]:
    from neurofaune.provenance import package_provenance

    out = dict(record)
    out.setdefault("written", datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    out.setdefault("written_by", {k: v for k, v in package_provenance().items()
                                  if k in ("Name", "Version", "CommitID")})
    return out


def write_design_record(directory: PathLike, record: Dict[str, Any], *,
                        check_files: bool = True) -> Tuple[Path, Path]:
    """Write ``design.json`` and ``design.md`` into ``directory``.

    With ``check_files`` (the default) the record is checked against the
    ``design.mat`` / ``design.con`` / ``design.fts`` already in ``directory``;
    any disagreement, or a vague record, raises :class:`DesignRecordError` and
    nothing is written.
    """
    directory = Path(directory)
    files = {}
    if check_files:
        for key, name in (("design_mat", "design.mat"), ("design_con", "design.con"),
                          ("design_fts", "design.fts")):
            if (directory / name).exists():
                files[key] = directory / name
    check_design_record(record, **files)
    record = _stamp(record)
    for c in record.get("contrasts") or []:          # -0.0 is 0
        c["vector"] = [float(v) + 0.0 for v in c.get("vector") or []]
    json_path = directory / RECORD_JSON
    md_path = directory / RECORD_MD
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    md_path.write_text(render_design_markdown(record))
    return json_path, md_path


def write_design(
    directory: PathLike,
    X: np.ndarray,
    columns: Sequence[Union[Dict[str, Any], Sequence]],
    contrasts: Sequence[Dict[str, Any]],
    **describe,
) -> Dict[str, Path]:
    """Write a design and what it means in one call: ``design.mat``,
    ``design.con`` (contrasts named), ``design.fts`` when F-tests are given,
    ``design.json`` and ``design.md``. Keyword arguments are those of
    :func:`describe_design` (``rows``, ``groups``, ``summary``, ``ftests``, ...).

    The record is checked before anything is written, so a design that cannot
    say what its columns and contrasts are is never written.
    """
    directory = Path(directory)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    describe.setdefault("n", X.shape[0])
    record = describe_design(columns, contrasts, **describe)
    check_design_record(record)
    if X.shape != (record["n"], len(record["columns"])):
        raise DesignRecordError(f"X is {X.shape[0]} x {X.shape[1]}; the description has "
                                f"{record['n']} rows x {len(record['columns'])} columns")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"design_mat": write_vest(directory / "design.mat", X, "mat"),
             "design_con": write_vest(directory / "design.con",
                                      np.array([c["vector"] for c in record["contrasts"]]),
                                      "con", [c["name"] for c in record["contrasts"]])}
    if record.get("ftests"):
        fts = np.zeros((len(record["ftests"]), len(record["contrasts"])))
        for i, ft in enumerate(record["ftests"]):
            for x in ft["contrasts"]:
                fts[i, x - 1] = 1
        paths["design_fts"] = write_vest(directory / "design.fts", fts, "fts")
    paths["design_json"], paths["design_md"] = write_design_record(directory, record)
    return paths


def read_design_record(directory: PathLike) -> Optional[Dict[str, Any]]:
    """The ``design.json`` in ``directory``, or None."""
    path = Path(directory) / RECORD_JSON
    return json.loads(path.read_text()) if path.exists() else None


def attach_design(output_dir: PathLike, design_mat: PathLike, design_con: PathLike,
                  fts_file: Optional[PathLike] = None,
                  log: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """Put a run's design, and what it tests, into its output folder.

    Copies ``design.mat`` / ``design.con`` (/ ``design.fts``) into ``output_dir``
    and, when a ``design.json`` sits beside ``design_mat``, checks it against them
    and copies it with its ``design.md``. A record that contradicts its matrices
    raises :class:`DesignRecordError` -- the run would be described wrongly. A
    missing record is warned about (the run's contrasts are then undescribed)
    and None is returned.
    """
    log = log or logger
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_mat, design_con = Path(design_mat), Path(design_con)
    sources = {"design.mat": design_mat, "design.con": design_con}
    if fts_file is not None:
        sources["design.fts"] = Path(fts_file)

    record = read_design_record(design_mat.parent)
    if record is not None:
        check_design_record(record, design_mat=design_mat, design_con=design_con,
                            design_fts=Path(fts_file) if fts_file is not None else None)
    else:
        log.warning(
            "No design.json beside %s: this run's design columns and contrasts are not "
            "described, so nothing in its results says what it tested. Write one with "
            "neurofaune.analysis.stats.design_record (write_design, or "
            "write_design_record for an existing design).", design_mat)

    for name, src in sources.items():
        dst = output_dir / name
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    if record is not None:
        for name in (RECORD_JSON, RECORD_MD):
            src = design_mat.parent / name
            dst = output_dir / name
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
        if not (output_dir / RECORD_MD).exists():
            (output_dir / RECORD_MD).write_text(render_design_markdown(record))
    return record


def summarize_contrasts(record: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``[{index, name, tests, test_kind, group_a, group_b}]`` for a results summary."""
    if not record:
        return []
    keys = ("index", "name", "tests", "test_kind", "group_a", "group_b")
    return [{k: c[k] for k in keys if k in c} for c in record.get("contrasts") or []]


def iter_runs_without_records(root: PathLike) -> Iterable[Path]:
    """Folders under ``root`` holding a ``design.con`` but no ``design.json``."""
    for con in sorted(Path(root).rglob("design.con")):
        if not (con.parent / RECORD_JSON).exists():
            yield con.parent
