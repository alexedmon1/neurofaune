#!/usr/bin/env python3
"""
Generate a clean AUC lookup CSV from ROI wide CSVs.

Extracts session-matched AUC (area under the plasma concentration curve)
values from the ROI wide CSV's AUC_p30, AUC_p60, AUC_p90 columns. Each
subject's AUC is matched to their scan timepoint (ses-p60 -> AUC_p60).
Controls have AUC=0 at all timepoints.

Output CSV format:
    subject,session,auc
    sub-Rat001,ses-p30,12.5
    sub-Rat001,ses-p60,38.2
    sub-Rat002,ses-p30,0.0

Usage:
    uv run python scripts/prepare_auc_lookup.py \
        --roi-csv /mnt/arborea/bpa-rat/network/roi/roi_FA_wide.csv \
        --output /mnt/arborea/bpa-rat/auc_lookup.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Session -> AUC column mapping
SESSION_TO_AUC = {
    "ses-p30": "AUC_p30",
    "ses-p60": "AUC_p60",
    "ses-p90": "AUC_p90",
}


def extract_auc_lookup(roi_csv: Path) -> pd.DataFrame:
    """Extract session-matched AUC from a ROI wide CSV.

    Parameters
    ----------
    roi_csv : Path
        Path to any ROI wide CSV containing AUC_p30, AUC_p60, AUC_p90.

    Returns
    -------
    DataFrame with columns: subject, session, auc.
    """
    df = pd.read_csv(roi_csv)

    # Verify AUC columns exist
    auc_cols = [c for c in SESSION_TO_AUC.values() if c in df.columns]
    if not auc_cols:
        raise ValueError(
            f"No AUC columns found in {roi_csv}. "
            f"Expected: {list(SESSION_TO_AUC.values())}"
        )
    logger.info("Found AUC columns: %s", auc_cols)

    rows = []
    for _, row in df.iterrows():
        subject = row["subject"]
        session = row["session"]

        auc_col = SESSION_TO_AUC.get(session)
        if auc_col is None or auc_col not in df.columns:
            logger.debug("Skipping %s %s — no AUC column for session", subject, session)
            continue

        auc_val = row.get(auc_col)
        if pd.isna(auc_val):
            logger.debug("Skipping %s %s — AUC is NaN", subject, session)
            continue

        rows.append({
            "subject": subject,
            "session": session,
            "auc": float(auc_val),
        })

    result = pd.DataFrame(rows)
    logger.info(
        "Extracted %d AUC values (%d unique subjects)",
        len(result),
        result["subject"].nunique(),
    )

    # Summary stats
    for session, auc_col in SESSION_TO_AUC.items():
        subset = result[result["session"] == session]
        if len(subset) > 0:
            logger.info(
                "  %s: n=%d, AUC range=[%.2f, %.2f], mean=%.2f",
                session,
                len(subset),
                subset["auc"].min(),
                subset["auc"].max(),
                subset["auc"].mean(),
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate AUC lookup CSV from ROI wide CSV"
    )
    parser.add_argument(
        "--roi-csv",
        type=Path,
        required=True,
        help="Path to any ROI wide CSV with AUC columns (e.g. roi_FA_wide.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for auc_lookup.csv",
    )
    args = parser.parse_args()

    if not args.roi_csv.exists():
        logger.error("ROI CSV not found: %s", args.roi_csv)
        sys.exit(1)

    lookup = extract_auc_lookup(args.roi_csv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(args.output, index=False)
    logger.info("Saved AUC lookup: %s (%d rows)", args.output, len(lookup))


if __name__ == "__main__":
    main()
