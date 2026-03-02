#!/usr/bin/env python3
"""Extract MRS (PRESS spectroscopy) data from Bruker Cohort 7+8 into the MRS folder.

Cohorts 7-8 use an IRC1200 scanner with a flat directory layout:
    /mnt/arborea/bruker/Cohort7/
        IRC1200_Cohort7_Rat83_1__Rat_83__p90_1_1_20250507_112422/
            10/  11/  ...  (numbered scan directories)

This script:
1. Iterates over IRC1200_* session directories in Cohort7 and Cohort8
2. Parses rat ID and timepoint from the directory name
3. Finds the metabolite scan (ACQ_scan_name contains PRESS_1H_Hippo + PVM_WsMode=VAPOR)
4. Copies the entire scan directory as metabolite/ into the MRS tree
5. Skips sessions that already exist
6. Reports sessions where no metabolite scan was found

Usage:
    uv run python scripts/extract_bruker_mrs.py [--dry-run]
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BRUKER_ROOT = Path("/mnt/arborea/bruker")
MRS_ROOT = Path("/mnt/arborea/bpa-rat/mrs")
COHORTS = ["Cohort7", "Cohort8"]

# Pattern for IRC1200 directory names:
#   IRC1200_Cohort7_Rat83_1__Rat_83__p90_1_1_20250507_112422
#   IRC1200_Cohort8_Rat1_1__Rat_1__p60_1_1_20250620_084158
# We extract the first RatNNN and the timepoint (p30/p60/p90).
SESSION_RE = re.compile(
    r"IRC1200_Cohort\d+_(Rat\d+)_.*_(p\d+)_\d+_\d+_\d{8}_\d{6}$"
)


def parse_session_dir(dirname: str) -> tuple[str, str] | None:
    """Extract (rat_id, timepoint) from an IRC1200 session directory name."""
    m = SESSION_RE.match(dirname)
    if m:
        return m.group(1), m.group(2)
    return None


def read_bruker_param(filepath: Path, key: str) -> str | None:
    """Read a Bruker parameter value from acqp or method file.

    For simple params like ##$PVM_WsMode=VAPOR, returns 'VAPOR'.
    For array params like ##$ACQ_scan_name=( 64 ), returns the next line stripped of <>.
    """
    if not filepath.exists():
        return None
    lines = filepath.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"##${key}="):
            value = line.split("=", 1)[1].strip()
            # Array parameter — value is on the next line
            if value.startswith("(") and i + 1 < len(lines):
                return lines[i + 1].strip().strip("<>")
            return value
    return None


def find_metabolite_scan(session_dir: Path) -> Path | None:
    """Find the PRESS_1H_Hippo metabolite scan with VAPOR water suppression."""
    for child in sorted(session_dir.iterdir()):
        if not child.is_dir() or not child.name.isdigit():
            continue
        acqp = child / "acqp"
        method = child / "method"
        scan_name = read_bruker_param(acqp, "ACQ_scan_name")
        if scan_name and "PRESS_1H_Hippo" in scan_name:
            ws_mode = read_bruker_param(method, "PVM_WsMode")
            if ws_mode == "VAPOR":
                return child
    return None


def extract_mrs(dry_run: bool = False) -> None:
    """Main extraction loop."""
    extracted = []
    skipped_existing = []
    skipped_no_press = []
    errors = []

    for cohort_name in COHORTS:
        cohort_dir = BRUKER_ROOT / cohort_name
        if not cohort_dir.exists():
            log.warning("Cohort directory not found: %s", cohort_dir)
            continue

        sessions = sorted(
            d for d in cohort_dir.iterdir()
            if d.is_dir() and d.name.startswith("IRC1200_")
        )
        log.info("Found %d sessions in %s", len(sessions), cohort_name)

        for session_dir in sessions:
            parsed = parse_session_dir(session_dir.name)
            if not parsed:
                log.warning("Could not parse session name: %s", session_dir.name)
                errors.append(session_dir.name)
                continue

            rat_id, timepoint = parsed
            target_dir = MRS_ROOT / rat_id / timepoint / "metabolite"

            if target_dir.exists():
                log.debug("Already exists, skipping: %s/%s", rat_id, timepoint)
                skipped_existing.append(f"{rat_id}/{timepoint}")
                continue

            metabolite_scan = find_metabolite_scan(session_dir)
            if metabolite_scan is None:
                log.warning(
                    "No PRESS_1H_Hippo (VAPOR) found: %s (%s/%s)",
                    session_dir.name, rat_id, timepoint,
                )
                skipped_no_press.append(f"{rat_id}/{timepoint}")
                continue

            if dry_run:
                log.info(
                    "[DRY RUN] Would copy scan %s → %s",
                    metabolite_scan, target_dir,
                )
            else:
                log.info(
                    "Copying scan %s → %s", metabolite_scan.name, target_dir
                )
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(metabolite_scan, target_dir)

            extracted.append(f"{rat_id}/{timepoint}")

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    action = "Would extract" if dry_run else "Extracted"
    print(f"{action}: {len(extracted)} sessions")
    if extracted:
        for s in sorted(extracted):
            print(f"  + {s}")
    print(f"Skipped (already exists): {len(skipped_existing)}")
    print(f"Skipped (no PRESS_1H_Hippo): {len(skipped_no_press)}")
    if skipped_no_press:
        for s in sorted(skipped_no_press):
            print(f"  ! {s}")
    if errors:
        print(f"Errors (unparseable names): {len(errors)}")
        for e in errors:
            print(f"  ? {e}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MRS data from Bruker Cohort 7+8"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files",
    )
    args = parser.parse_args()
    extract_mrs(dry_run=args.dry_run)
