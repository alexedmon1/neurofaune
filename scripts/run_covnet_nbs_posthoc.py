#!/usr/bin/env python3
"""
Post-hoc characterization of NBS components.

Loads previously saved NBS results and runs two analyses:

1. Edge direction test (unthresholded): binomial sign test and Wilcoxon
   signed-rank test on ALL edges to assess whether the overall covariance
   difference is directionally biased, independent of NBS thresholding.

2. Component characterization (for significant components): edge
   directionality and node centrality within each component.

Usage:
    uv run python scripts/run_covnet_nbs_posthoc.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --nbs-dir $STUDY_ROOT/network/covnet/nbs/pooled \
        --modality dwi \
        --metrics FA MD AD RD \
        --design pooled
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet import CovNetAnalysis
from neurofaune.network.covnet.nbs import characterize_components, edge_direction_test
from neurofaune.network.covnet.pipeline import _save_nbs_characterization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc characterization of NBS components"
    )
    parser.add_argument("--roi-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=["dwi", "msme"])
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--exclusion-csv", type=Path, default=None)
    parser.add_argument("--design", default="pooled",
                        choices=["pooled", "sex_stratified"])
    parser.add_argument("--nbs-dir", type=Path, default=None,
                        help="Direct path to NBS results dir")
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    for metric in args.metrics:
        logger.info(f"=== Post-hoc: {args.modality} {metric} ({args.design}) ===")

        covnet_root = args.output_dir / args.design
        analysis = CovNetAnalysis.prepare(
            roi_dir=args.roi_dir,
            exclusion_csv=args.exclusion_csv,
            covnet_root=covnet_root,
            modality=args.modality,
            metric=metric,
        )

        if args.nbs_dir:
            nbs_dir = args.nbs_dir / args.modality / metric
            nbs_results = analysis._load_nbs_results(nbs_dir)

            direction_rows = []
            posthoc = {}

            for comp_label, result in nbs_results.items():
                out_dir = nbs_dir / comp_label / "posthoc"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Edge direction test on full matrix
                dir_test = edge_direction_test(result["test_stat"])
                dir_test["comparison"] = comp_label
                direction_rows.append(dir_test)

                # Component characterization
                characterized = characterize_components(
                    result,
                    roi_cols=result.get("roi_cols", analysis.region_cols),
                    p_threshold=args.p_threshold,
                )
                if characterized:
                    _save_nbs_characterization(characterized, result, out_dir)

                posthoc[comp_label] = {
                    "direction_test": dir_test,
                    "components": characterized,
                }

            # Save direction summary
            if direction_rows:
                dir_df = pd.DataFrame(direction_rows)
                cols = ["comparison", "n_edges", "n_positive", "n_negative",
                        "frac_positive", "mean_z", "median_z",
                        "sign_test_p", "wilcoxon_p"]
                dir_df[cols].to_csv(
                    nbs_dir / "edge_direction_summary.csv", index=False
                )
        else:
            posthoc = analysis.run_nbs_posthoc(p_threshold=args.p_threshold)

        # Report results
        logger.info(f"  {metric} Edge Direction Tests:")
        for comp_label, data in sorted(posthoc.items()):
            dt = data["direction_test"]
            sig = "*" if dt["sign_test_p"] < 0.05 else " "
            direction = "POS" if dt["frac_positive"] > 0.55 else (
                "NEG" if dt["frac_positive"] < 0.45 else "BAL")
            logger.info(
                f"  {sig} {comp_label:30s} | "
                f"{direction} ({dt['frac_positive']:.0%} pos) | "
                f"mean z={dt['mean_z']:+.3f} | "
                f"sign p={dt['sign_test_p']:.4f} | "
                f"wilcoxon p={dt['wilcoxon_p']:.4f}"
            )

        n_with_components = sum(
            1 for v in posthoc.values() if v["components"]
        )
        logger.info(
            f"  {metric}: {n_with_components} comparisons with "
            f"significant NBS components"
        )


if __name__ == "__main__":
    main()
