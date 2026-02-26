"""Deprecated: use neurofaune.reporting instead."""

from neurofaune.reporting import (  # noqa: F401
    register,
    load_registry,
    save_registry,
    list_entries,
    remove_entry,
    backfill_registry,
    generate_index_html,
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
