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
