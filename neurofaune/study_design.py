"""Who each animal is, what group it is in, and what each session is.

The imaging tree does not carry this, so it used to be guessed from session labels and
literals. :class:`StudyDesign` reads it from the study's tables instead and raises when
something is missing or unrecognised.

There is deliberately no ``cohort`` field. The word can mean an age band, a recruited
batch or a separate study, and the code has used it for several jobs that coincide in
only some designs. Each job has its own name here:

- ``animal``, ``session``: identity. Cross-validation groups on ``animal``.
- ``group``: the experimental group, ordered by :attr:`StudyDesign.levels`.
- ``sex``: ``M`` or ``F``, or blank when not recorded.
- ``age_days``: numeric, or blank when not recorded.
- ``template``: which template the session registers to.

Any other column (``timepoint``, ``batch``, a study's own ``cohort``) is carried through
and can be filtered on, but nothing here interprets it.

Sources, read-only:

- participants: ``design.participants``, else ``<paths.bids>/participants.tsv``;
- sessions: ``design.sessions``, else ``<paths.study_root>/design/sessions.tsv``, else the
  BIDS ``sub-*/sub-*_sessions.tsv`` files. They are not merged: the first found is used.

Config keys, all under ``design:``. They are not in ``default.yaml``, because the config
digest written into provenance covers that file and adding them would change it for every
study:

- ``group_column``: column holding the group (default ``group``);
- ``levels``: group levels in order (default: sorted, control first);
- ``control``: the control level;
- ``template_column``: column holding the template, or ``session_label`` for the session
  label without ``ses-``, which an existing study may state explicitly;
- ``template``: one template name for every session;
- ``sex_codes``: extra sex codes, e.g. ``{1: M, 2: F}``.

A blank group excludes the animal deliberately, and :attr:`StudyDesign.excluded` lists it.
An animal with sessions but no row in the participants table is an error, because that
means the metadata is incomplete.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurofaune.config import get_config_value

logger = logging.getLogger(__name__)

SESSION_LABEL = "session_label"
_FIXED = ("animal", "session", "group", "sex", "age_days", "template")
_SEX_CODES = {"m": "M", "male": "M", "f": "F", "female": "F"}
_SUBJECT_COLUMNS = ("participant_id", "animal", "subject")
_SESSION_COLUMNS = ("session_id", "session")
_NA_VALUES = ["", "n/a", "N/A", "NA", "nan"]


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=_NA_VALUES)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def _prefixed(values: pd.Series, prefix: str) -> pd.Series:
    return values.map(lambda v: v if pd.isna(v) or v.startswith(prefix) else f"{prefix}{v}")


def _find_column(df: pd.DataFrame, names: Sequence[str], path: Path) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"{path} needs one of the columns {list(names)}; it has {list(df.columns)}")


def _read_participants(path: Path, group_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set design.participants to its location")
    df = _read_tsv(path)
    animal = _find_column(df, _SUBJECT_COLUMNS, path)
    if group_column not in df.columns:
        raise KeyError(
            f"{path} has no column {group_column!r} (design.group_column); "
            f"it has {list(df.columns)}"
        )
    if group_column != "group" and "group" in df.columns:
        raise ValueError(
            f"{path} has a 'group' column but design.group_column is {group_column!r}; "
            "rename one so that 'group' has one meaning"
        )
    df = df.rename(columns={animal: "animal"})
    df["animal"] = _prefixed(df["animal"], "sub-")
    dupes = df.loc[df["animal"].duplicated(), "animal"].tolist()
    if dupes:
        raise ValueError(f"{path} has duplicate animals: {dupes[:5]}")
    if group_column != "group":
        df["group"] = df[group_column]
    return df


def _read_session_file(path: Path) -> pd.DataFrame:
    """One table holding every animal's sessions."""
    df = _read_tsv(path)
    animal = _find_column(df, _SUBJECT_COLUMNS, path)
    return df.rename(columns={animal: "animal"})


def _read_bids_sessions(bids_root: Path) -> pd.DataFrame:
    """BIDS keeps one file per subject, and the subject is in its name, not its columns."""
    tables = []
    for path in sorted(bids_root.glob("sub-*/sub-*_sessions.tsv")):
        df = _read_tsv(path)
        df["animal"] = path.parent.name
        tables.append(df)
    if not tables:
        raise FileNotFoundError(
            f"no sessions table: no sub-*/sub-*_sessions.tsv under {bids_root}. "
            "Set design.sessions or write design/sessions.tsv"
        )
    return pd.concat(tables, ignore_index=True)


def _join(sessions: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    """Join on animal; a column in both tables must agree wherever both have a value."""
    absent = sorted(set(sessions["animal"]) - set(participants["animal"]))
    if absent:
        raise ValueError(
            f"{len(absent)} animal(s) have sessions but no row in participants: {absent[:5]}"
        )
    shared = [c for c in participants.columns if c in sessions.columns and c != "animal"]
    df = sessions.merge(participants, on="animal", how="left", suffixes=("", "__participant"))
    for column in shared:
        other = f"{column}__participant"
        conflict = df[column].notna() & df[other].notna() & (df[column] != df[other])
        if conflict.any():
            example = df.loc[conflict].iloc[0]
            raise ValueError(
                f"{column!r} disagrees between the sessions and participants tables at "
                f"{conflict.sum()} row(s), e.g. {example['animal']} {example['session']}: "
                f"{example[column]!r} against {example[other]!r}"
            )
        df[column] = df[column].fillna(df[other])
        df = df.drop(columns=other)
    return df


def _coded_sex(values: pd.Series, codes: dict[Any, str] | None) -> pd.Series:
    known = dict(_SEX_CODES)
    known.update({str(k).lower(): str(v).upper() for k, v in (codes or {}).items()})
    lowered = values.str.lower()
    unknown = sorted(values[lowered.notna() & ~lowered.isin(known)].unique())
    if unknown:
        raise ValueError(f"unknown sex code(s) {unknown}; declare them in design.sex_codes")
    return lowered.map(known)


def _numeric_age(values: pd.Series) -> pd.Series:
    age = pd.to_numeric(values, errors="coerce")
    bad = sorted(values[values.notna() & age.isna()].unique())
    if bad:
        raise ValueError(f"age_days must be numeric; found {bad[:5]}")
    return age


def _template_column(
    df: pd.DataFrame, template_column: str | None, template: str | None
) -> pd.Series:
    if template_column and template:
        raise ValueError("set design.template_column or design.template, not both")
    if template:
        return pd.Series(str(template), index=df.index)
    if template_column == SESSION_LABEL:
        return df["session"].str.replace(r"^ses-", "", regex=True)
    if template_column:
        if template_column not in df.columns:
            raise KeyError(f"design.template_column {template_column!r} is not a column")
        return df[template_column]
    if "template" in df.columns:
        return df["template"]
    return pd.Series(np.nan, index=df.index, dtype=object)


class StudyDesign:
    """The study's session table and the group order. Build with :func:`load_study_design`."""

    def __init__(
        self,
        participants: pd.DataFrame,
        sessions: pd.DataFrame,
        levels: Sequence[Any] | None = None,
        control: Any | None = None,
        template_column: str | None = None,
        template: str | None = None,
        sex_codes: dict[Any, str] | None = None,
    ):
        column = _find_column(sessions, _SESSION_COLUMNS, Path("sessions table"))
        sessions = sessions.rename(columns={column: "session"}).copy()
        sessions["animal"] = _prefixed(sessions["animal"], "sub-")
        sessions["session"] = _prefixed(sessions["session"], "ses-")
        repeated = sessions.duplicated(["animal", "session"])
        if repeated.any():
            first = sessions.loc[repeated].iloc[0]
            raise ValueError(
                f"{repeated.sum()} duplicate session row(s), e.g. "
                f"{first['animal']} {first['session']}"
            )

        df = _join(sessions, participants)
        excluded = df["group"].isna()
        self.excluded: list[str] = sorted(df.loc[excluded, "animal"].unique())
        if self.excluded:
            logger.info("Excluding %d animal(s) with a blank group", len(self.excluded))
        df = df.loc[~excluded].reset_index(drop=True)
        if df.empty:
            raise ValueError("no session is left: every animal with sessions has a blank group")

        self.control = None if control is None else str(control)
        self.levels = self._ordered_levels(df["group"], levels)

        df["sex"] = _coded_sex(df["sex"], sex_codes) if "sex" in df.columns else np.nan
        df["age_days"] = _numeric_age(df["age_days"]) if "age_days" in df.columns else np.nan
        df["template"] = _template_column(df, template_column, template)
        others = [c for c in df.columns if c not in _FIXED]
        self._table = df[list(_FIXED) + others]

    def _ordered_levels(self, groups: pd.Series, levels: Sequence[Any] | None) -> list[str]:
        found = set(groups.unique())
        if levels is None:
            ordered = sorted(found)
        else:
            ordered = [str(level) for level in levels]
            unknown = sorted(found - set(ordered))
            if unknown:
                raise ValueError(
                    f"group value(s) {unknown} are not in design.levels {ordered}; "
                    "add them, or leave the group blank to exclude the animal"
                )
        if self.control is None:
            return ordered
        if self.control not in ordered:
            raise ValueError(f"design.control {self.control!r} is not a level: {ordered}")
        return [self.control] + [level for level in ordered if level != self.control]

    def table(self) -> pd.DataFrame:
        """One row per session, excluded animals left out."""
        return self._table.copy()

    def where(self, **filters: Any) -> pd.DataFrame:
        """Rows matching every filter; a value may be a list. An empty result raises."""
        df = self._table
        for column, value in filters.items():
            if column not in df.columns:
                raise KeyError(f"no column {column!r}; columns are {list(df.columns)}")
            wanted = value if isinstance(value, (list, tuple, set)) else [value]
            df = df[df[column].isin(list(wanted))]
        if df.empty:
            raise ValueError(f"no session matches {filters}")
        return df.copy()

    def _row(self, animal: str, session: str) -> pd.Series:
        hit = self._table[(self._table["animal"] == animal) & (self._table["session"] == session)]
        if hit.empty:
            raise KeyError(f"{animal} {session} is not in the study design")
        return hit.iloc[0]

    def template_of(self, animal: str, session: str) -> str:
        """The session's template. Never derived from the label unless the config says so."""
        value = self._row(animal, session)["template"]
        if pd.isna(value):
            raise ValueError(
                f"{animal} {session} has no template; set design.template, "
                "design.template_column, or a 'template' column"
            )
        return str(value)

    def age_of(self, animal: str, session: str) -> float:
        """Age in days at this session. Blank raises: age is not guessed."""
        value = self._row(animal, session)["age_days"]
        if pd.isna(value):
            raise ValueError(f"{animal} {session} has no age_days in the sessions table")
        return float(value)

    def groups_for(self, rows: pd.DataFrame) -> np.ndarray:
        """The animal behind each row, for grouped cross-validation."""
        column = "animal" if "animal" in rows.columns else "subject"
        if column not in rows.columns:
            raise KeyError("rows need an 'animal' or 'subject' column")
        animals = _prefixed(rows[column].astype(str), "sub-")
        unknown = sorted(set(animals) - set(self._table["animal"]))
        if unknown:
            raise ValueError(f"{len(unknown)} animal(s) not in the study design: {unknown[:5]}")
        return animals.to_numpy()


def load_study_design(config: dict[str, Any]) -> StudyDesign:
    """Build the :class:`StudyDesign` a study's config describes."""
    bids_value = get_config_value(config, "paths.bids")
    bids_root = Path(bids_value) if bids_value else None
    participants = get_config_value(config, "design.participants")
    if participants is not None:
        participants_path = Path(participants)
    elif bids_root is not None:
        participants_path = bids_root / "participants.tsv"
    else:
        raise KeyError("set design.participants or paths.bids")

    sessions = get_config_value(config, "design.sessions")
    study_root = get_config_value(config, "paths.study_root")
    if sessions is not None:
        sessions_path = Path(sessions)
        if not sessions_path.exists():
            raise FileNotFoundError(f"design.sessions: {sessions_path} not found")
    else:
        sessions_path = Path(study_root) / "design" / "sessions.tsv" if study_root else None

    participants_table = _read_participants(
        participants_path, get_config_value(config, "design.group_column", default="group")
    )
    if sessions_path is not None and sessions_path.exists():
        sessions_table = _read_session_file(sessions_path)
    elif bids_root is not None:
        sessions_table = _read_bids_sessions(bids_root)
    else:
        raise FileNotFoundError("no sessions table: set design.sessions or paths.bids")
    return StudyDesign(
        participants_table,
        sessions_table,
        levels=get_config_value(config, "design.levels"),
        control=get_config_value(config, "design.control"),
        template_column=get_config_value(config, "design.template_column"),
        template=get_config_value(config, "design.template"),
        sex_codes=get_config_value(config, "design.sex_codes"),
    )
