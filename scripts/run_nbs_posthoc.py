#!/usr/bin/env python3
"""
NBS post-hoc analyses: centrality and hub vulnerability.

Loads existing NBS results from disk (components.json + test_statistics.csv)
and runs post-hoc characterisation on every significant component (p < 0.05):

  - **Centrality** (centrality.csv): degree, betweenness, eigenvector centrality
    and mean |z| per node — identifies the most influential nodes in the component.

  - **Hub vulnerability** (hub_vulnerability.csv): leave-one-node-out analysis
    measuring how much the component shrinks when each node is removed — identifies
    load-bearing hubs whose removal fragments the network.

Results are written to ``{comparison}/posthoc/`` inside the existing NBS output tree.

Usage:
    python scripts/run_nbs_posthoc.py \\
        --nbs-dir /mnt/arborea/bpa-rat/network/covnet/nbs \\
        --modality dwi msme func \\
        --p-threshold 0.05
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet.nbs import nbs_posthoc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_component(raw: dict) -> dict:
    """Normalise a component dict from JSON to the format nbs_posthoc expects.

    Handles both storage formats:
    - Old: ``"edges": [[u, v], ...]``  (raw index pairs)
    - New: ``"edges": [{"edge": [u, v], ...}, ...]``  (dicts with per-edge stats)
    """
    edges_raw = raw.get("edges", [])
    if edges_raw and isinstance(edges_raw[0], dict):
        edges = [tuple(e["edge"]) for e in edges_raw]
    else:
        edges = [tuple(e) for e in edges_raw]

    return {
        "nodes": raw["nodes"],
        "edges": edges,
        "size": raw["size"],
        "pvalue": raw["pvalue"],
    }


def run_posthoc_for_comparison(
    comp_dir: Path,
    p_threshold: float,
) -> int:
    """Run post-hoc analyses for one comparison directory.

    Returns the number of significant components processed.
    """
    comp_json = comp_dir / "components.json"
    stat_csv = comp_dir / "test_statistics.csv"

    if not comp_json.exists():
        return 0
    if not stat_csv.exists():
        logger.warning("No test_statistics.csv in %s — skipping", comp_dir)
        return 0

    with open(comp_json) as f:
        data = json.load(f)

    sig_comps = [
        _load_component(c)
        for c in data.get("components", [])
        if c.get("pvalue", 1.0) < p_threshold
    ]
    if not sig_comps:
        return 0

    stat_df = pd.read_csv(stat_csv, index_col=0)
    test_stat = stat_df.values
    roi_cols = list(stat_df.index)

    posthoc_dir = comp_dir / "posthoc"
    posthoc_dir.mkdir(exist_ok=True)

    for idx, comp in enumerate(sig_comps):
        ph = nbs_posthoc(comp, test_stat, roi_cols)
        prefix = f"comp{idx:02d}_"

        pd.DataFrame(ph["centrality"]).to_csv(
            posthoc_dir / f"{prefix}centrality.csv", index=False
        )
        pd.DataFrame(ph["hub_vulnerability"]).to_csv(
            posthoc_dir / f"{prefix}hub_vulnerability.csv", index=False
        )

    logger.info(
        "  %s: %d significant component(s) → posthoc/",
        comp_dir.name, len(sig_comps),
    )
    return len(sig_comps)


def main():
    parser = argparse.ArgumentParser(
        description="NBS post-hoc centrality and hub-vulnerability analyses"
    )
    parser.add_argument(
        "--nbs-dir", type=Path, required=True,
        help="Root NBS output directory (contains modality/metric/comparison subdirs)",
    )
    parser.add_argument(
        "--modality", nargs="+", default=None,
        help="Modalities to process (e.g. dwi msme func). Default: all found.",
    )
    parser.add_argument(
        "--metric", nargs="+", default=None,
        help="Metrics to process (e.g. FA MD). Default: all found.",
    )
    parser.add_argument(
        "--p-threshold", type=float, default=0.05,
        help="p-value threshold for significant components (default: 0.05)",
    )

    args = parser.parse_args()

    if not args.nbs_dir.exists():
        logger.error("NBS directory not found: %s", args.nbs_dir)
        sys.exit(1)

    total_comparisons = 0
    total_components = 0

    # New structure: {nbs_dir}/{variant}/{modality}/{metric}/{comparison}/
    # variant is e.g. "pooled", "sex_stratified/F", "sex_stratified/M", "interaction"
    for comp_json in sorted(args.nbs_dir.rglob("components.json")):
        # Skip posthoc dirs
        if "posthoc" in comp_json.parts:
            continue
        comp_dir = comp_json.parent
        # Log path relative to nbs_dir for clarity
        rel = comp_dir.relative_to(args.nbs_dir)
        parts = rel.parts
        # Expect at least variant/modality/metric/comparison (4 parts)
        # sex_stratified/F/modality/metric/comparison = 5 parts
        if len(parts) < 4:
            continue
        # Filter by modality/metric if requested (modality is 2nd-to-last-3 or 3rd-to-last-2)
        # Modality is parts[-3], metric is parts[-2], comparison is parts[-1]
        modality_name = parts[-3]
        metric_name = parts[-2]
        if args.modality and modality_name not in args.modality:
            continue
        if args.metric and metric_name not in args.metric:
            continue

        variant = "/".join(parts[:-3])
        logger.info("%s / %s / %s", variant, modality_name.upper(), metric_name)
        n = run_posthoc_for_comparison(comp_dir, args.p_threshold)
        total_comparisons += 1
        total_components += n

    logger.info(
        "\nDone. Processed %d comparison directories; "
        "%d significant component(s) had post-hoc analyses written.",
        total_comparisons, total_components,
    )


if __name__ == "__main__":
    main()
