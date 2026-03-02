"""Provenance tracking for ROI-based and voxelwise analyses.

Provides reusable functions for hashing source data files and writing/validating
provenance.json records. Consolidates patterns previously duplicated across
prepare_tbss_designs.py, prepare_fmri_voxelwise.py, and others.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file (8 KB chunked)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_roi_provenance(
    output_dir: Path,
    roi_dir: Path,
    metrics: List[str],
    exclusion_csv: Optional[Path],
    n_subjects: int,
    analysis_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write provenance.json tracking ROI CSV and exclusion CSV hashes.

    Parameters
    ----------
    output_dir : Path
        Analysis output directory (provenance.json is written here).
    roi_dir : Path
        Directory containing roi_*_wide.csv files.
    metrics : list[str]
        Metric names (e.g. ["FA", "MD"]).
    exclusion_csv : Path or None
        Optional exclusion list CSV.
    n_subjects : int
        Number of subjects included in the analysis.
    analysis_type : str
        Label for the analysis (e.g. "classification", "connectome").
    extra : dict or None
        Additional key/value pairs to include in provenance.

    Returns
    -------
    Path
        Path to the written provenance.json file.
    """
    source_hashes: Dict[str, str] = {}

    for metric in metrics:
        csv_path = roi_dir / f"roi_{metric}_wide.csv"
        if csv_path.exists():
            source_hashes[f"roi_{metric}_wide.csv"] = sha256_file(csv_path)

    provenance: Dict[str, Any] = {
        "analysis_type": analysis_type,
        "roi_dir": str(roi_dir),
        "metrics": metrics,
        "n_subjects": n_subjects,
        "source_hashes": source_hashes,
        "date_created": datetime.now().isoformat(),
    }

    if exclusion_csv is not None and exclusion_csv.exists():
        provenance["exclusion_csv"] = str(exclusion_csv)
        provenance["exclusion_csv_sha256"] = sha256_file(exclusion_csv)

    if extra:
        provenance.update(extra)

    prov_path = output_dir / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Wrote provenance.json → %s", prov_path)
    return prov_path


def validate_roi_provenance(
    output_dir: Path,
    roi_dir: Path,
    metrics: List[str],
    exclusion_csv: Optional[Path],
    log: Optional[logging.Logger] = None,
) -> bool:
    """Validate provenance against current source data.

    Checks that the SHA256 hashes recorded in provenance.json still match
    the files on disk. Returns True if valid (or if no provenance exists,
    with a warning). Returns False if any hash mismatch is detected.

    Parameters
    ----------
    output_dir : Path
        Directory containing provenance.json.
    roi_dir : Path
        Directory containing roi_*_wide.csv files.
    metrics : list[str]
        Metric names to check.
    exclusion_csv : Path or None
        Optional exclusion list CSV.
    log : logging.Logger or None
        Logger instance (uses module logger if None).

    Returns
    -------
    bool
        True if provenance is valid or missing; False if stale.
    """
    _log = log or logger
    prov_path = output_dir / "provenance.json"

    if not prov_path.exists():
        _log.warning("No provenance.json found in %s — cannot validate", output_dir)
        return True  # permissive: missing provenance is a warning, not a failure

    with open(prov_path) as f:
        provenance = json.load(f)

    source_hashes = provenance.get("source_hashes", {})
    valid = True

    for metric in metrics:
        csv_name = f"roi_{metric}_wide.csv"
        csv_path = roi_dir / csv_name
        recorded = source_hashes.get(csv_name)

        if recorded and csv_path.exists():
            current = sha256_file(csv_path)
            if current != recorded:
                _log.warning(
                    "PROVENANCE MISMATCH: %s hash changed "
                    "(recorded=%s... current=%s...)",
                    csv_name,
                    recorded[:16],
                    current[:16],
                )
                valid = False

    # Check exclusion CSV
    recorded_excl = provenance.get("exclusion_csv_sha256")
    if recorded_excl and exclusion_csv and exclusion_csv.exists():
        current_excl = sha256_file(exclusion_csv)
        if current_excl != recorded_excl:
            _log.warning(
                "PROVENANCE MISMATCH: exclusion CSV hash changed "
                "(recorded=%s... current=%s...)",
                recorded_excl[:16],
                current_excl[:16],
            )
            valid = False

    if valid:
        _log.info("Provenance validated OK for %s", output_dir.name)

    return valid
