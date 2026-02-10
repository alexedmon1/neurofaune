"""
Report Registry — central JSON store for analysis results.

Stores entries in {analysis_root}/report_registry.json with file-locking
for concurrent-safe writes and atomic rename for NFS safety.
"""

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

VALID_STATUSES = ("completed", "partial", "failed")


def _empty_registry(analysis_root: str, study_name: str = "") -> Dict[str, Any]:
    """Return a minimal empty registry dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "study_name": study_name,
        "analysis_root": str(analysis_root),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "entries": {},
    }


def _registry_path(analysis_root: Path) -> Path:
    return Path(analysis_root) / "report_registry.json"


def load_registry(analysis_root: Path) -> Dict[str, Any]:
    """
    Load the registry JSON, returning an empty registry if not found.

    Args:
        analysis_root: Root directory of analysis outputs.

    Returns:
        Registry dict.
    """
    path = _registry_path(analysis_root)
    if not path.exists():
        return _empty_registry(str(analysis_root))
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read registry %s: %s — starting fresh", path, exc)
        return _empty_registry(str(analysis_root))


def save_registry(analysis_root: Path, registry: Dict[str, Any]) -> Path:
    """
    Atomically write the registry JSON with file-locking.

    Uses fcntl.flock for local safety and atomic write-rename for NFS.

    Args:
        analysis_root: Root directory of analysis outputs.
        registry: Full registry dict.

    Returns:
        Path to the written registry file.
    """
    dest = _registry_path(analysis_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    registry["last_updated"] = datetime.now().isoformat(timespec="seconds")

    # Write to temp file in same directory, then rename (atomic on POSIX).
    fd, tmp_path = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".tmp", prefix=".registry_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(registry, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp_path, str(dest))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return dest


def register(
    analysis_root: Path,
    entry_id: str,
    analysis_type: str,
    display_name: str,
    output_dir: str,
    *,
    status: str = "completed",
    summary_stats: Optional[Dict[str, Any]] = None,
    figures: Optional[List[str]] = None,
    report_html: Optional[str] = None,
    source_summary_json: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    notes: str = "",
    study_name: str = "",
    auto_generate_index: bool = True,
) -> Dict[str, Any]:
    """
    Register (or update) an analysis entry in the report registry.

    All paths stored are relative to *analysis_root* for portability.

    Args:
        analysis_root: Absolute path to analysis root directory.
        entry_id: Stable unique key for this entry.
        analysis_type: One of 'tbss', 'roi_extraction', 'covnet', 'batch_qc'.
        display_name: Human-readable label shown in the dashboard.
        output_dir: Relative path (from analysis_root) to output directory.
        status: 'completed', 'partial', or 'failed'.
        summary_stats: Arbitrary dict of key metrics.
        figures: List of relative paths to figure files.
        report_html: Relative path to a detailed HTML report (if any).
        source_summary_json: Relative path to the source summary JSON.
        config: Snapshot of analysis configuration.
        warnings: List of warning strings.
        notes: Free-text notes.
        study_name: Study name (only used when creating a fresh registry).
        auto_generate_index: If True, regenerate index.html after registering.

    Returns:
        The entry dict that was stored.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")

    analysis_root = Path(analysis_root)
    registry = load_registry(analysis_root)

    if study_name:
        registry["study_name"] = study_name

    entry = {
        "entry_id": entry_id,
        "analysis_type": analysis_type,
        "display_name": display_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "output_dir": output_dir,
        "summary_stats": summary_stats or {},
        "figures": figures or [],
        "report_html": report_html,
        "source_summary_json": source_summary_json,
        "config": config or {},
        "warnings": warnings or [],
        "notes": notes,
    }

    registry["entries"][entry_id] = entry
    save_registry(analysis_root, registry)
    logger.info("Registered analysis entry: %s (%s)", entry_id, analysis_type)

    if auto_generate_index:
        try:
            from neurofaune.analysis.reporting.index_generator import (
                generate_index_html,
            )
            generate_index_html(analysis_root, registry=registry)
        except Exception as exc:
            logger.warning("Failed to auto-generate index.html: %s", exc)

    return entry


def remove_entry(analysis_root: Path, entry_id: str) -> bool:
    """
    Remove an entry from the registry.

    Args:
        analysis_root: Analysis root directory.
        entry_id: Entry to remove.

    Returns:
        True if the entry was found and removed, False otherwise.
    """
    analysis_root = Path(analysis_root)
    registry = load_registry(analysis_root)
    if entry_id in registry["entries"]:
        del registry["entries"][entry_id]
        save_registry(analysis_root, registry)
        logger.info("Removed registry entry: %s", entry_id)
        return True
    return False


def list_entries(
    analysis_root: Path,
    analysis_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List registry entries, optionally filtered by analysis type.

    Args:
        analysis_root: Analysis root directory.
        analysis_type: If given, only return entries of this type.

    Returns:
        List of entry dicts, sorted by timestamp descending.
    """
    registry = load_registry(analysis_root)
    entries = list(registry["entries"].values())
    if analysis_type:
        entries = [e for e in entries if e["analysis_type"] == analysis_type]
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries
