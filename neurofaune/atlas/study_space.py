"""The study-space atlas named by the config: its intensity template and its name.

Group results are drawn on the intensity template of the atlas the study registered
to, in the grid the analyses ran on (``atlas.study_space``). Which atlas that is
belongs to the study's config -- SIGMA for the rat studies so far, another atlas for
a mouse study -- so nothing here names one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from neurofaune.config import get_config_value


def study_space_template(config: Optional[Dict[str, Any]]) -> Optional[Path]:
    """The study-space intensity template: ``atlas.study_space.template_masked``, else
    ``template``; None when the config names neither or the file is absent."""
    if not config:
        return None
    for key in ("atlas.study_space.template_masked", "atlas.study_space.template"):
        value = get_config_value(config, key, default=None)
        if value and Path(str(value)).is_file():
            return Path(str(value))
    return None


def atlas_space_name(config: Optional[Dict[str, Any]], default: str = "study atlas space") -> str:
    """The atlas the study registered to, by name (``atlas.name``, e.g. "SIGMA")."""
    name = get_config_value(config, "atlas.name", default=None) if config else None
    return str(name) if name else default


def study_space_axes(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The anatomical direction each voxel axis of the study-space atlas runs toward
    (``atlas.study_space.axes``, e.g. "LIA"), or None when the config does not say.

    Rodent image headers often follow the scanner rather than the animal, so the
    orientation a reader should trust is declared, not read from the header. A value
    that is not three letters, one from each of R/L, A/P, S/I, is an error.
    """
    from neurofaune.results.spec import valid_axes

    value = get_config_value(config, "atlas.study_space.axes", default=None) if config else None
    if value is None:
        return None
    if not valid_axes(str(value)):
        raise ValueError(f"atlas.study_space.axes = {value!r}: three letters, one from each of "
                         "R/L, A/P, S/I (S = dorsal, I = ventral)")
    return str(value).upper()


def display_plane(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The plane group maps are shown in (``atlas.study_space.display_plane``:
    axial, coronal or sagittal), or None for the reader's default."""
    value = get_config_value(config, "atlas.study_space.display_plane", default=None) if config else None
    if value is None:
        return None
    if str(value) not in ("axial", "coronal", "sagittal"):
        raise ValueError(f"atlas.study_space.display_plane = {value!r}: axial, coronal or sagittal")
    return str(value)
