"""Unified analysis reporting: registry, discovery, and HTML dashboard generation."""

from .registry import register, load_registry, save_registry, list_entries, remove_entry
from .discover import backfill_registry
from .index_generator import generate_index_html

__all__ = [
    "register",
    "load_registry",
    "save_registry",
    "list_entries",
    "remove_entry",
    "backfill_registry",
    "generate_index_html",
]
