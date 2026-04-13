"""Provenance tracking for ROI-based, voxelwise, and batch analyses.

Provides reusable functions for hashing source data files and writing/validating
provenance.json records. Consolidates patterns previously duplicated across
prepare_tbss_designs.py, prepare_fmri_voxelwise.py, and others.

Also provides write_batch_run_manifest() for a standardized batch-run record
that every neurofaune batch script should call on completion.
"""

import hashlib
import json
import logging
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# Batch run manifest
# ---------------------------------------------------------------------------

def _neurofaune_git_hash() -> Optional[str]:
    """Return the short git hash of the neurofaune repo, or None if unavailable."""
    try:
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def write_batch_run_manifest(
    output_dir: Path,
    analysis_name: str,
    parameters: Dict[str, Any],
    session_results: List[Dict],
    input_files: Optional[List[Path]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Write a standardised batch-run manifest for any neurofaune batch script.

    Produces two files in output_dir:
    - ``run_manifest_{timestamp}.json``  — full machine-readable record
    - ``run_manifest_{timestamp}.txt``   — human-readable summary

    Designed to be called at the end of every batch preprocessing or analysis
    script so that every run has a durable, self-describing record.

    Parameters
    ----------
    output_dir : Path
        Directory where manifest files are written (created if needed).
    analysis_name : str
        Short label for the run, e.g. ``"fALFF"``, ``"ReHo"``,
        ``"func_preprocess"``, ``"dwi_preprocess"``.
    parameters : dict
        Key/value pairs describing all CLI arguments and config settings used
        for this run (pass ``vars(args)`` from argparse).
    session_results : list of dict
        One dict per processed session/subject. Each dict must have at minimum
        a ``"status"`` key (``"success"``, ``"failed"``, ``"skipped"``, or
        ``"exception"``). Common optional keys: ``"key"``, ``"subject"``,
        ``"session"``, ``"outputs"``, ``"statistics"``, ``"error"``.
    input_files : list of Path, optional
        Shared input files referenced by the run (e.g. exclusion CSVs,
        config files). Their paths and SHA-256 hashes are recorded.
    extra : dict, optional
        Any additional top-level fields to include in the JSON manifest.

    Returns
    -------
    tuple[Path, Path]
        ``(json_path, txt_path)`` — paths to the two written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dt = datetime.now().isoformat()

    # --- Tally outcomes ---
    counts: Dict[str, int] = {}
    for r in session_results:
        status = r.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    # --- Hash shared input files ---
    hashed_inputs: Dict[str, str] = {}
    if input_files:
        for p in input_files:
            p = Path(p)
            if p.exists():
                hashed_inputs[str(p)] = sha256_file(p)
            else:
                hashed_inputs[str(p)] = "not_found"

    # --- Collect failed sessions for quick reference ---
    failures = [
        {
            "key": r.get("key", r.get("subject", "?")),
            "error": r.get("error", ""),
        }
        for r in session_results
        if r.get("status") in ("failed", "exception")
    ]

    # --- Build JSON manifest ---
    manifest: Dict[str, Any] = {
        "analysis_name": analysis_name,
        "run_datetime": run_dt,
        "neurofaune_git_hash": _neurofaune_git_hash(),
        "command": sys.argv,
        "parameters": {k: str(v) for k, v in parameters.items()},
        "n_sessions": len(session_results),
        "outcome_counts": counts,
        "input_files": hashed_inputs,
        "failures": failures,
        "session_results": session_results,
    }
    if extra:
        manifest.update(extra)

    json_path = output_dir / f"run_manifest_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # --- Build human-readable text summary ---
    lines = [
        "=" * 70,
        f"neurofaune batch run manifest",
        "=" * 70,
        f"Analysis  : {analysis_name}",
        f"Date/time : {run_dt}",
        f"Git hash  : {manifest['neurofaune_git_hash'] or 'unavailable'}",
        f"Command   : {' '.join(str(a) for a in sys.argv)}",
        "",
        "Parameters",
        "----------",
    ]
    for k, v in parameters.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "Outcome summary",
        "---------------",
        f"  Total sessions : {len(session_results)}",
    ]
    for status, n in sorted(counts.items()):
        lines.append(f"  {status:<12}: {n}")

    if hashed_inputs:
        lines += ["", "Input files", "-----------"]
        for path, digest in hashed_inputs.items():
            lines.append(f"  {path}")
            lines.append(f"    sha256: {digest}")

    if failures:
        lines += ["", "Failed sessions", "---------------"]
        for f in failures:
            lines.append(f"  {f['key']}: {f['error']}")

    lines += [
        "",
        "Output files (first 20 sessions)",
        "--------------------------------",
    ]
    for r in session_results[:20]:
        key = r.get("key", r.get("subject", "?"))
        status = r.get("status", "?")
        outputs = r.get("outputs", {})
        lines.append(f"  {key}  [{status}]")
        if isinstance(outputs, dict):
            for out_key, out_val in outputs.items():
                if isinstance(out_val, dict):
                    for sub_key, sub_val in out_val.items():
                        lines.append(f"    {out_key}/{sub_key}: {sub_val}")
                elif isinstance(out_val, str) and out_val not in ("skipped", "no_transforms"):
                    lines.append(f"    {out_key}: {out_val}")
    if len(session_results) > 20:
        lines.append(f"  ... and {len(session_results) - 20} more (see JSON manifest)")

    lines += [
        "",
        "Manifest files",
        "--------------",
        f"  JSON : {json_path}",
    ]

    txt_path = output_dir / f"run_manifest_{timestamp}.txt"
    lines.append(f"  Text : {txt_path}")
    lines.append("=" * 70)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote batch run manifest → %s", txt_path)
    return json_path, txt_path
