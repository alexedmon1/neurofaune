#!/usr/bin/env python3
"""
Utilities for managing preprocessing exclusion markers.

When preprocessing fails for a subject/session, an exclusion marker file
is created to prevent repeated attempts and exclusion from group analyses.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import traceback


def create_exclusion_marker(
    subject: str,
    session: str,
    output_dir: Path,
    reason: str,
    error_traceback: Optional[str] = None
) -> Path:
    """
    Create a preprocessing exclusion marker file.

    This marker indicates that preprocessing failed for this subject/session
    and they should be excluded from template building and group analyses.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    output_dir : Path
        Study root directory
    reason : str
        Brief reason for exclusion
    error_traceback : str, optional
        Full error traceback for debugging

    Returns
    -------
    Path
        Path to created marker file
    """
    # Create marker in derivatives directory for this subject/session
    marker_dir = output_dir / 'derivatives' / subject / session / 'anat'
    marker_dir.mkdir(parents=True, exist_ok=True)

    marker_file = marker_dir / '.preprocessing_failed'

    # Write marker with metadata
    marker_data = {
        'subject': subject,
        'session': session,
        'timestamp': datetime.now().isoformat(),
        'reason': reason,
        'traceback': error_traceback
    }

    with open(marker_file, 'w') as f:
        json.dump(marker_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"⚠ PREPROCESSING FAILED - EXCLUSION MARKER CREATED")
    print(f"{'='*60}")
    print(f"Subject: {subject}")
    print(f"Session: {session}")
    print(f"Reason: {reason}")
    print(f"Marker: {marker_file}")
    print(f"{'='*60}\n")

    return marker_file


def check_exclusion_marker(
    subject: str,
    session: str,
    output_dir: Path
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if a preprocessing exclusion marker exists.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    output_dir : Path
        Study root directory

    Returns
    -------
    Tuple[bool, Optional[dict]]
        (marker_exists, marker_data)
    """
    marker_file = output_dir / 'derivatives' / subject / session / 'anat' / '.preprocessing_failed'

    if not marker_file.exists():
        return False, None

    try:
        with open(marker_file, 'r') as f:
            marker_data = json.load(f)
        return True, marker_data
    except:
        # Marker file exists but can't read it
        return True, {'reason': 'Unknown (marker file corrupted)'}


def remove_exclusion_marker(
    subject: str,
    session: str,
    output_dir: Path
) -> bool:
    """
    Remove an exclusion marker to retry preprocessing.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    output_dir : Path
        Study root directory

    Returns
    -------
    bool
        True if marker was removed, False if it didn't exist
    """
    marker_file = output_dir / 'derivatives' / subject / session / 'anat' / '.preprocessing_failed'

    if marker_file.exists():
        marker_file.unlink()
        print(f"Removed exclusion marker for {subject} {session}")
        return True
    return False
