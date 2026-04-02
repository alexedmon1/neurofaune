#!/usr/bin/env python3
"""
Post-hoc characterization of NBS components.

Loads previously saved NBS results and computes edge directionality
(increased vs. decreased covariance) and node centrality (degree,
betweenness) within each significant component. Does not re-run
permutation testing.

Usage:
    uv run python scripts/run_covnet_nbs_posthoc.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality dwi \
        --metrics FA MD AD RD \
        --exclusion-csv $STUDY_ROOT/dti_nonstandard_slices.csv \
        --design pooled

    uv run python scripts/run_covnet_nbs_posthoc.py \
        --roi-dir $STUDY_ROOT/network/roi \
        --output-dir $STUDY_ROOT/network/covnet \
        --modality msme \
        --metrics T2 MWF IWF CSFF \
        --design pooled
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet import CovNetAnalysis

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
                        help="Direct path to NBS results dir (overrides "
                             "default path from output-dir/design)")
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

        # Allow direct NBS path for legacy directory structures
        if args.nbs_dir:
            nbs_dir = args.nbs_dir / args.modality / metric
            nbs_results = analysis._load_nbs_results(nbs_dir)
            posthoc = {}
            for comp_label, result in nbs_results.items():
                from neurofaune.network.covnet.nbs import characterize_components
                from neurofaune.network.covnet.pipeline import _save_nbs_characterization as _save_nbs_posthoc
                characterized = characterize_components(
                    result,
                    roi_cols=result.get("roi_cols", analysis.region_cols),
                    p_threshold=args.p_threshold,
                )
                if characterized:
                    out_dir = nbs_dir / comp_label / "posthoc"
                    _save_nbs_posthoc(characterized, result, out_dir)
                    posthoc[comp_label] = characterized
        else:
            posthoc = analysis.run_nbs_posthoc(p_threshold=args.p_threshold)

        n_comparisons = len(posthoc)
        n_components = sum(len(v) for v in posthoc.values())
        logger.info(
            f"  {metric}: {n_components} significant components "
            f"across {n_comparisons} comparisons"
        )

        for comp_label, components in posthoc.items():
            for i, comp in enumerate(components):
                logger.info(
                    f"  {comp_label} component {i}: "
                    f"{comp['n_nodes']} nodes, {comp['n_edges']} edges, "
                    f"{comp['n_increased']} increased / {comp['n_decreased']} decreased, "
                    f"mean z={comp['mean_z']:.3f}, "
                    f"hubs: {', '.join(comp['hub_nodes'][:5]) or 'none'}"
                )


if __name__ == "__main__":
    main()
