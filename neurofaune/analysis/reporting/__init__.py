"""Deprecated: use neurofaune.reporting instead."""

from neurofaune.reporting import (  # noqa: F401
    backfill_registry,
    generate_index_html,
    list_entries,
    load_registry,
    register,
    remove_entry,
    save_registry,
)

__all__ = [
    "register",
    "load_registry",
    "save_registry",
    "list_entries",
    "remove_entry",
    "backfill_registry",
    "generate_index_html",
]
