#!/usr/bin/env python3
"""
MELODIC RSN Cleaning — identify resting-state networks from group ICA output.

Two modes
---------
auto
    Score each IC on spectral and spatial features; keep components whose
    noise score falls below ``--noise-score-thresh``.

manual
    Provide ``--components 1,3,7,12`` (1-based indices); those components
    are kept unconditionally.

In both modes the tool writes three files to the output directory:

- ``melodic_IC_RSN.nii.gz``  — 4D volume (RSN components only)
- ``rsn_index.json``         — per-component features and classification
- ``rsn_mosaic.png``         — axial-slice mosaic of RSN spatial maps

Typical usage
-------------
  # Automated (default thresholds):
  uv run python scripts/run_melodic_clean.py \\
      --melodic-dir /mnt/arborea/bpa-rat/analysis/melodic/p90 \\
      --mask /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \\
      --tr 0.5 \\
      --bg-image /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Template.nii.gz

  # Manual component selection:
  uv run python scripts/run_melodic_clean.py \\
      --melodic-dir /mnt/arborea/bpa-rat/analysis/melodic/p90 \\
      --mask /mnt/arborea/bpa-rat/atlas/SIGMA_study_space/SIGMA_InVivo_Brain_Mask.nii.gz \\
      --tr 0.5 \\
      --mode manual --components 1,4,7,9,12,15

  # Write outputs to a separate directory:
  uv run python scripts/run_melodic_clean.py \\
      --melodic-dir /mnt/arborea/bpa-rat/analysis/melodic/p90 \\
      --mask .../SIGMA_InVivo_Brain_Mask.nii.gz \\
      --tr 0.5 \\
      --output-dir /mnt/arborea/bpa-rat/analysis/melodic/p90/rsn
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.analysis.func.melodic_clean import clean_melodic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_component_list(value: str):
    """Parse a comma-separated list of 1-based component indices."""
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid component list '{value}'. Expected comma-separated integers, e.g. 1,3,7,12"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Identify RSN components from MELODIC output and write cleaned maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--melodic-dir", type=Path, required=True,
        help="MELODIC output directory (contains melodic_IC.nii.gz, melodic_mix, melodic_FTmix).",
    )
    parser.add_argument(
        "--mask", type=Path, required=True,
        help="Brain mask in the same space as the IC maps (e.g. SIGMA_InVivo_Brain_Mask.nii.gz).",
    )
    parser.add_argument(
        "--tr", type=float, required=True,
        help="Repetition time in seconds (used for low-frequency power calculation).",
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["auto", "manual"], default="auto",
        help="Classification mode: 'auto' (default) or 'manual'.",
    )
    parser.add_argument(
        "--components", type=parse_component_list, default=None,
        metavar="1,3,7,...",
        help="Comma-separated 1-based component indices to keep (required for --mode manual).",
    )

    # Optional outputs
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for cleaned files. Defaults to --melodic-dir.",
    )
    parser.add_argument(
        "--bg-image", type=Path, default=None,
        help="Background anatomical image for the mosaic (e.g. SIGMA template). "
             "If omitted, the mean IC magnitude is used.",
    )

    # Auto-mode thresholds
    thresholds = parser.add_argument_group("auto-mode thresholds")
    thresholds.add_argument(
        "--low-freq-thresh", type=float, default=0.3,
        help="Minimum low-frequency power ratio to be considered signal (default: 0.3). "
             "Components below this get +1.0 noise score.",
    )
    thresholds.add_argument(
        "--edge-thresh", type=float, default=0.4,
        help="Maximum edge fraction tolerated for RSNs (default: 0.4). "
             "Components above this get +1.0 noise score.",
    )
    thresholds.add_argument(
        "--bg-thresh", type=float, default=0.1,
        help="Maximum background (outside-mask) fraction tolerated (default: 0.1). "
             "Components above this get +1.0 noise score.",
    )
    thresholds.add_argument(
        "--roughness-thresh", type=float, default=1.5,
        help="Maximum spatial roughness (normalised Sobel gradient) tolerated (default: 1.5). "
             "Components above this get +0.5 noise score.",
    )
    thresholds.add_argument(
        "--noise-score-thresh", type=float, default=1.5,
        help="Components with noise score strictly below this are kept as RSNs (default: 1.5).",
    )

    args = parser.parse_args()

    # Validate
    if not args.melodic_dir.exists():
        logger.error("--melodic-dir not found: %s", args.melodic_dir)
        sys.exit(1)
    ic_maps = args.melodic_dir / "melodic_IC.nii.gz"
    if not ic_maps.exists():
        logger.error("melodic_IC.nii.gz not found in %s", args.melodic_dir)
        sys.exit(1)
    if not args.mask.exists():
        logger.error("--mask not found: %s", args.mask)
        sys.exit(1)
    if args.mode == "manual" and not args.components:
        logger.error("--components is required when --mode manual is selected")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("MELODIC RSN CLEANING")
    logger.info("=" * 70)
    logger.info("  MELODIC dir : %s", args.melodic_dir)
    logger.info("  Mask        : %s", args.mask)
    logger.info("  TR          : %.4f s", args.tr)
    logger.info("  Mode        : %s", args.mode)
    if args.mode == "manual":
        logger.info("  Components  : %s", args.components)
    else:
        logger.info(
            "  Thresholds  : low_freq=%.2f  edge=%.2f  bg=%.2f  roughness=%.2f  noise_score=%.2f",
            args.low_freq_thresh, args.edge_thresh, args.bg_thresh,
            args.roughness_thresh, args.noise_score_thresh,
        )
    logger.info("  Output dir  : %s", args.output_dir or args.melodic_dir)
    logger.info("")

    result = clean_melodic(
        melodic_dir=args.melodic_dir,
        brain_mask_file=args.mask,
        tr=args.tr,
        mode=args.mode,
        component_indices=args.components,
        bg_image=args.bg_image,
        output_dir=args.output_dir,
        low_freq_thresh=args.low_freq_thresh,
        edge_thresh=args.edge_thresh,
        bg_thresh=args.bg_thresh,
        roughness_thresh=args.roughness_thresh,
        noise_score_thresh=args.noise_score_thresh,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("  RSNs identified : %d / %d", result["n_rsn"], result["n_components"])
    logger.info("  RSN indices     : %s", [i + 1 for i in result["rsn_indices"]])
    if result.get("rsn_volume"):
        logger.info("  RSN volume      : %s", result["rsn_volume"])
    if result.get("rsn_mosaic"):
        logger.info("  Mosaic          : %s", result["rsn_mosaic"])
    logger.info("  Index JSON      : %s", result["rsn_index"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
