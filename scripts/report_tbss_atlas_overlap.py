#!/usr/bin/env python3
"""
Report SIGMA atlas overlap for significant TBSS clusters.

For each significant contrast in a randomise output directory, intersects the
thresholded TFCE corrected-p map with the SIGMA parcellation atlas and reports
which regions contain significant voxels, ranked by voxel count.

This answers the question: "Where in the brain are the significant TBSS clusters?"
at the level of named atlas regions rather than voxel coordinates.

Usage:
    python scripts/report_tbss_atlas_overlap.py \
        --randomise-dir /study/analysis/tbss/template/dwi/p90/randomise/per_pnd_p90_M \
        --atlas /atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
        --labels-csv /atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv

    # Restrict to specific metrics, raise voxel floor, suppress terminal output
    python scripts/report_tbss_atlas_overlap.py \
        --randomise-dir /study/.../per_pnd_p90_M \
        --atlas /atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
        --labels-csv /atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv \
        --metrics RD MD \
        --min-voxels 50 \
        --output-csv /study/results/p90_M_atlas_overlap.csv \
        --no-report

Output CSV columns:
    metric               Imaging metric (FA, MD, RD, MWF, ...)
    contrast             Contrast name (e.g. C_gt_H, M_gt_C)
    n_sig_voxels         Total significant voxels in this contrast
    region_id            SIGMA atlas label integer
    region_name          SIGMA region name
    n_voxels             Voxels from this contrast inside this region
    fraction_of_cluster  Fraction of the full significant cluster in this region
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = [
    "metric",
    "contrast",
    "n_sig_voxels",
    "region_id",
    "region_name",
    "n_voxels",
    "fraction_of_cluster",
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_label_map(labels_csv: Path) -> dict[int, str]:
    """Load SIGMA label integer → region name from the SIGMA labels CSV.

    The SIGMA labels CSV is expected to have at minimum two columns:
      - ``Labels``: integer label value stored in the atlas NIfTI
      - ``Region of interest``: human-readable name (dots used as word separators)

    Dot separators are replaced with spaces in the returned names.
    """
    label_map: dict[int, str] = {}
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_id = int(row["Labels"])
            name = row["Region of interest"].replace(".", " ")
            label_map[label_id] = name
    return label_map


def load_contrast_names(randomise_dir: Path) -> list[str]:
    """Read ordered contrast names from analysis_summary.json.

    Returns an empty list if the file is absent (contrast indices are used
    as fallback names).
    """
    summary_path = randomise_dir / "analysis_summary.json"
    if not summary_path.exists():
        return []
    with open(summary_path) as f:
        data = json.load(f)
    return data.get("contrast_names", [])


def compute_atlas_overlap(
    corrp_data: np.ndarray,
    atlas_data: np.ndarray,
    label_map: dict[int, str],
    threshold: float = 0.95,
) -> list[dict]:
    """Return per-region voxel counts for voxels exceeding *threshold*.

    Parameters
    ----------
    corrp_data:
        3-D array of TFCE corrected p-values (1 − p, as output by FSL randomise).
    atlas_data:
        3-D integer array with the same shape as ``corrp_data`` containing SIGMA
        atlas label values (0 = outside atlas).
    label_map:
        Mapping from label integer to region name.
    threshold:
        Significance threshold applied to corrp values (default 0.95 = p < 0.05).

    Returns
    -------
    list of dicts, one per atlas region with keys:
        region_id, region_name, n_voxels, fraction_of_cluster.
    Regions are sorted descending by n_voxels.  A synthetic "Outside atlas"
    entry is appended if any significant voxels fall outside the parcellation.
    """
    if corrp_data.shape != atlas_data.shape:
        raise ValueError(
            f"Shape mismatch: corrp {corrp_data.shape} vs atlas {atlas_data.shape}. "
            "Both images must be in the same voxel space."
        )

    sig_mask = corrp_data > threshold
    n_sig_total = int(sig_mask.sum())
    if n_sig_total == 0:
        return []

    sig_labels = atlas_data[sig_mask]
    label_counts = Counter(sig_labels[sig_labels > 0].astype(int))
    n_outside = n_sig_total - int(sum(label_counts.values()))

    rows = []
    for label_id, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        rows.append(
            {
                "region_id": label_id,
                "region_name": label_map.get(label_id, f"SIGMA_{label_id}"),
                "n_voxels": count,
                "fraction_of_cluster": round(count / n_sig_total, 4),
            }
        )
    if n_outside > 0:
        rows.append(
            {
                "region_id": 0,
                "region_name": "Outside atlas",
                "n_voxels": n_outside,
                "fraction_of_cluster": round(n_outside / n_sig_total, 4),
            }
        )
    return rows


def process_metric_dir(
    metric_dir: Path,
    atlas_data: np.ndarray,
    label_map: dict[int, str],
    contrast_names: list[str],
    threshold: float,
    min_voxels: int,
) -> list[dict]:
    """Process all corrp maps in one ``randomise_<metric>`` directory.

    Parameters
    ----------
    metric_dir:
        Directory named ``randomise_<metric>`` containing
        ``*_tfce_corrp_tstat*.nii.gz`` files.
    contrast_names:
        Ordered list of contrast names from analysis_summary.json.  Used to
        label tstat indices.  Falls back to ``"tstat<n>"`` if empty.
    min_voxels:
        Skip contrasts with fewer than this many significant voxels.

    Returns
    -------
    list of flat dicts ready to write to CSV.
    """
    corrp_maps = sorted(metric_dir.glob("*_tfce_corrp_tstat*.nii.gz"))
    if not corrp_maps:
        logger.debug("No corrp maps found in %s", metric_dir)
        return []

    metric_name = metric_dir.name.removeprefix("randomise_")
    rows = []

    for corrp_path in corrp_maps:
        # Parse 1-based tstat index from filename (strip both .nii and .gz)
        stem = corrp_path.name.replace(".nii.gz", "").replace(".nii", "")
        try:
            tstat_idx_0 = int(stem.rsplit("tstat", 1)[1]) - 1
        except (IndexError, ValueError):
            tstat_idx_0 = -1

        contrast_name = (
            contrast_names[tstat_idx_0]
            if 0 <= tstat_idx_0 < len(contrast_names)
            else f"tstat{tstat_idx_0 + 1}"
        )

        corrp_data = nib.load(corrp_path).get_fdata()
        n_sig = int((corrp_data > threshold).sum())

        if n_sig < min_voxels:
            logger.debug(
                "  %s / %s: %d voxels < min_voxels=%d, skipping",
                metric_name, contrast_name, n_sig, min_voxels,
            )
            continue

        logger.info("  %s / %s: %d significant voxels", metric_name, contrast_name, n_sig)

        region_rows = compute_atlas_overlap(corrp_data, atlas_data, label_map, threshold)
        for r in region_rows:
            rows.append({"metric": metric_name, "contrast": contrast_name, "n_sig_voxels": n_sig, **r})

    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        logger.warning("No significant contrasts found — CSV not written.")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)


def print_report(rows: list[dict], top_n: int = 20) -> None:
    """Print a human-readable summary to stdout."""
    if not rows:
        print("No significant contrasts above threshold.")
        return

    for (metric, contrast), group in groupby(rows, key=lambda r: (r["metric"], r["contrast"])):
        group = list(group)
        n_sig = group[0]["n_sig_voxels"]
        print(f"\n{'=' * 66}")
        print(f"  {metric} / {contrast}   ({n_sig:,} significant voxels)")
        print(f"{'=' * 66}")
        print(f"  {'Region':<52} {'Voxels':>8}  {'Fraction':>8}")
        print(f"  {'-' * 52} {'-' * 8}  {'-' * 8}")
        for r in group[:top_n]:
            print(f"  {r['region_name']:<52} {r['n_voxels']:>8,}  {r['fraction_of_cluster']:>8.3f}")
        if len(group) > top_n:
            remaining = sum(r["n_voxels"] for r in group[top_n:])
            print(f"  {'...' + str(len(group) - top_n) + ' more regions':<52} {remaining:>8,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Map significant TBSS clusters to SIGMA atlas regions. "
            "Produces a CSV of per-region voxel counts for each significant contrast."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--randomise-dir", type=Path, required=True,
        metavar="DIR",
        help=(
            "Randomise output directory containing randomise_<metric>/ subdirs "
            "and optionally analysis_summary.json for contrast names."
        ),
    )
    parser.add_argument(
        "--atlas", type=Path, required=True,
        metavar="NII",
        help=(
            "SIGMA anatomical atlas NIfTI (integer-labelled parcellation, must be "
            "in the same voxel space as the TBSS results)."
        ),
    )
    parser.add_argument(
        "--labels-csv", type=Path, required=True,
        metavar="CSV",
        help=(
            "SIGMA labels CSV with 'Labels' (integer) and 'Region of interest' "
            "columns. Dot separators in region names are replaced with spaces."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        metavar="FLOAT",
        help="TFCE corrp threshold for significance (default: 0.95 → p < 0.05).",
    )
    parser.add_argument(
        "--min-voxels", type=int, default=1,
        metavar="INT",
        help="Skip contrasts with fewer than this many significant voxels (default: 1).",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        metavar="METRIC",
        help=(
            "Restrict to these metric names (e.g. RD MD). "
            "Default: all randomise_<metric> subdirs found."
        ),
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        metavar="CSV",
        help="Output CSV path (default: <randomise-dir>/atlas_overlap.csv).",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        metavar="INT",
        help="Maximum regions to print per contrast in terminal report (default: 20).",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Suppress terminal report; only write CSV.",
    )

    args = parser.parse_args()

    # --- Validate paths -------------------------------------------------------
    if not args.randomise_dir.exists():
        logger.error("Randomise directory not found: %s", args.randomise_dir)
        sys.exit(1)
    if not args.atlas.exists():
        logger.error("Atlas not found: %s", args.atlas)
        sys.exit(1)
    if not args.labels_csv.exists():
        logger.error("Labels CSV not found: %s", args.labels_csv)
        sys.exit(1)

    # --- Load atlas & labels --------------------------------------------------
    logger.info("Loading atlas: %s", args.atlas)
    atlas_data = nib.load(str(args.atlas)).get_fdata().astype(int)
    logger.info("Atlas shape: %s", atlas_data.shape)

    logger.info("Loading label map: %s", args.labels_csv)
    label_map = load_label_map(args.labels_csv)
    logger.info("Loaded %d named regions", len(label_map))

    # --- Contrast names -------------------------------------------------------
    contrast_names = load_contrast_names(args.randomise_dir)
    if contrast_names:
        logger.info("Contrast names from summary: %s", contrast_names)
    else:
        logger.warning(
            "analysis_summary.json not found in %s — tstat indices used as contrast names",
            args.randomise_dir,
        )

    # --- Find metric dirs -----------------------------------------------------
    metric_dirs = sorted(
        d for d in args.randomise_dir.iterdir()
        if d.is_dir() and d.name.startswith("randomise_")
    )
    if args.metrics:
        wanted = set(args.metrics)
        metric_dirs = [d for d in metric_dirs if d.name.removeprefix("randomise_") in wanted]
    if not metric_dirs:
        logger.error(
            "No randomise_<metric> subdirectories found in %s%s",
            args.randomise_dir,
            f" matching {args.metrics}" if args.metrics else "",
        )
        sys.exit(1)
    logger.info(
        "Processing %d metric dir(s): %s",
        len(metric_dirs),
        [d.name for d in metric_dirs],
    )

    # --- Process --------------------------------------------------------------
    all_rows: list[dict] = []
    for metric_dir in metric_dirs:
        rows = process_metric_dir(
            metric_dir,
            atlas_data,
            label_map,
            contrast_names,
            args.threshold,
            args.min_voxels,
        )
        all_rows.extend(rows)

    # --- Output ---------------------------------------------------------------
    output_csv = args.output_csv or (args.randomise_dir / "atlas_overlap.csv")
    write_csv(all_rows, output_csv)

    if not args.no_report:
        print_report(all_rows, top_n=args.top_n)


if __name__ == "__main__":
    main()
