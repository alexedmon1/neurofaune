#!/usr/bin/env python3
"""
Structural covariance networks from composite morphometry.

Bridges scripts/extract_morphometry.py to neurofaune.network.covnet: reshapes the
long morphometry tables into the wide table CovNetAnalysis reads, removes global
head size, then runs the covariance tests.

The default node set is the composite structures, NOT the 234 regions. Every edge
is a correlation across subjects WITHIN one group, so the usable sample size is the
group size, not the session count; at region level with n~10-12 per cell the matrix
is noise. Use --nodes regions --bilateral only with a cohort that supports it.

Head-size correction is on by default (--normalise residual). Without it regional
volumes share a global size factor and nearly every edge comes out positive —
see neurofaune/network/structural_covariance.py.

Usage:
    # export the CovNet input only, to inspect it first
    uv run python scripts/run_covnet_morphometry.py \
        --morphometry-dir /mnt/arborea/bpa-rat/network/morphometry \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker.csv \
        --config /mnt/arborea/bpa-rat/config.yaml \
        --export-only

    # export and run the tests that survive small n
    uv run python scripts/run_covnet_morphometry.py \
        --morphometry-dir /mnt/arborea/bpa-rat/network/morphometry \
        --study-tracker /mnt/arborea/bpa-rat/study_tracker.csv \
        --config /mnt/arborea/bpa-rat/config.yaml \
        --exclusion-csv /mnt/arborea/bpa-rat/exclusions/anat_exclusions.csv \
        --tests abs-distance graph --n-perm 5000 --force
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.covnet import CovNetAnalysis  # noqa: E402
from neurofaune.network.structural_covariance import (  # noqa: E402
    DEFAULT_CONFOUNDS,
    NORMALISATIONS,
    build_covnet_table,
    export_covnet_wide,
    load_participants,
    region_nodes,
    structure_nodes,
    thickness_nodes,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

NODE_BUILDERS = {
    'structures': (structure_nodes, 'morphometry_structures.csv', 'volume_mm3'),
    'regions': (region_nodes, 'morphometry_regions.csv', 'volume_GM_mm3'),
    'thickness': (thickness_nodes, 'morphometry_thickness.csv', 'mean_thickness_mm'),
}

TESTS = ('abs-distance', 'graph', 'nbs', 'territory')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--morphometry-dir', type=Path, required=True,
                   help='Output directory of extract_morphometry.py')
    p.add_argument('--nodes', choices=sorted(NODE_BUILDERS), default='structures',
                   help='Node set (default: structures — see the module docstring '
                        'on why region-level SCN needs a large cohort)')
    p.add_argument('--measure', default=None,
                   help='Column to use as the node value (default depends on --nodes)')
    p.add_argument('--tissue', default=None,
                   help='For --nodes structures: keep only this tissue (GM/WM/CSF/any)')
    p.add_argument('--allow-nested', action='store_true',
                   help='Keep composite structures that nest inside one another '
                        '(subcortical contains hippocampus, fiber_tracts contains '
                        'the named tracts). Off by default: those edges are high '
                        'because of the label sets, not the anatomy')
    p.add_argument('--bilateral', action='store_true',
                   help='Average _L/_R pairs before correlating (halves the node count)')

    p.add_argument('--normalise', choices=NORMALISATIONS, default='residual',
                   help='Head-size correction (default: residual)')
    p.add_argument('--confounds', nargs='*', default=list(DEFAULT_CONFOUNDS),
                   help=f'Confounds for --normalise residual (default: '
                        f'{" ".join(DEFAULT_CONFOUNDS)})')

    p.add_argument('--study-tracker', type=Path, default=None,
                   help='bpa-rat study tracker CSV (irc.ID / dose.level / sex)')
    p.add_argument('--participants', type=Path, default=None,
                   help='BIDS participants.tsv supplying the group and sex '
                        '(alternative to --study-tracker)')
    p.add_argument('--group-col', default='group',
                   help='Column of --participants holding the experimental group '
                        '(default: group)')
    p.add_argument('--exclusion-csv', type=Path, default=None,
                   help='Sessions to exclude (subject,session,...). For volumes '
                        'this is normally anat_exclusions.csv')
    p.add_argument('--cohorts', nargs='*', default=None,
                   help='Cohorts to analyse (default: all present except unknown)')
    p.add_argument('--cohort-order', nargs='*', default=None,
                   help='Chronological cohort order for cross-timepoint '
                        'comparisons (default: sorted order)')
    p.add_argument('--sex', choices=['F', 'M'], default=None,
                   help='Restrict to one sex')

    p.add_argument('--config', type=Path, default=None,
                   help='Study config.yaml (supplies covnet root and atlas labels)')
    p.add_argument('--covnet-root', type=Path, default=None)
    p.add_argument('--labels-csv', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Where to write the CovNet input table '
                        '(default: {morphometry-dir}/covnet_input)')
    p.add_argument('--metric-name', default=None,
                   help='Metric label for the CovNet output tree '
                        '(default: derived from --nodes and --measure)')

    p.add_argument('--tests', nargs='*', choices=TESTS, default=['abs-distance', 'graph'],
                   help='Tests to run (default: abs-distance graph)')
    p.add_argument('--export-only', action='store_true',
                   help='Write the CovNet input table and stop')
    p.add_argument('--n-perm', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-workers', type=int, default=1)
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing CovNet results')
    return p.parse_args(argv)


def load_table(morphometry_dir: Path, filename: str) -> pd.DataFrame:
    path = morphometry_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found — run scripts/extract_morphometry.py first'
            + (' with --thickness' if 'thickness' in filename else '')
        )
    return pd.read_csv(path)


def main(argv=None) -> int:
    args = parse_args(argv)

    builder, filename, default_measure = NODE_BUILDERS[args.nodes]
    measure = args.measure or default_measure
    cleaned = measure.replace('_mm3', '').replace('_mm', '')
    metric = args.metric_name or (
        cleaned if args.nodes in cleaned else f'{args.nodes}_{cleaned}'
    )

    long_df = load_table(args.morphometry_dir, filename)
    kwargs = {'measure': measure}
    if args.nodes == 'structures':
        kwargs['tissue'] = args.tissue
        kwargs['allow_nested'] = args.allow_nested
        # extract_morphometry.py exports the resolved label sets precisely so the
        # rules stay defined in one place; use them to detect nesting.
        labels_json = args.morphometry_dir / 'structure_labels.json'
        if labels_json.exists():
            with open(labels_json) as fh:
                kwargs['structure_labels'] = json.load(fh)
        elif not args.allow_nested:
            logger.warning('%s not found; cannot detect nested structures. '
                           'Re-run extract_morphometry.py to produce it.', labels_json)
    nodes, node_cols = builder(long_df, **kwargs)

    # total_brain always comes from the structures table, whatever the node set is.
    structures = load_table(args.morphometry_dir, 'morphometry_structures.csv')

    if args.participants and args.study_tracker:
        logger.warning('Both --participants and --study-tracker given; '
                       'using --participants')
    phenotype = (load_participants(args.participants, group_col=args.group_col)
                 if args.participants else None)

    df, node_cols = build_covnet_table(
        nodes=nodes,
        node_cols=node_cols,
        structures=structures,
        phenotype=phenotype,
        study_tracker=None if phenotype is not None else args.study_tracker,
        method=args.normalise,
        confounds=args.confounds,
        bilateral=args.bilateral,
    )

    output_dir = args.output_dir or (args.morphometry_dir / 'covnet_input')
    wide_csv = export_covnet_wide(df, output_dir, metric)

    provenance = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'morphometry_dir': str(args.morphometry_dir),
        'source_table': filename,
        'nodes': args.nodes,
        'measure': measure,
        'tissue': args.tissue,
        'n_nodes': len(node_cols),
        'n_sessions': len(df),
        'bilateral': args.bilateral,
        'allow_nested': args.allow_nested,
        'normalise': args.normalise,
        'phenotype': str(args.participants or args.study_tracker or ''),
        'confounds': args.confounds if args.normalise == 'residual' else None,
        'metric': metric,
        'wide_csv': str(wide_csv),
    }
    with open(output_dir / f'roi_{metric}_provenance.json', 'w') as fh:
        json.dump(provenance, fh, indent=2)
    logger.info('%d nodes x %d sessions, normalise=%s', len(node_cols), len(df),
                args.normalise)

    if args.export_only:
        logger.info('--export-only: stopping before the CovNet tests')
        return 0

    analysis = CovNetAnalysis.prepare(
        wide_csv=wide_csv,
        exclusion_csv=args.exclusion_csv,
        covnet_root=args.covnet_root,
        config_path=args.config,
        modality='anat',
        metric=metric,
        labels_csv=args.labels_csv,
        sex=args.sex,
        cohorts=args.cohorts,
        cohort_order=args.cohort_order,
        force=args.force,
    )
    analysis.save()

    smallest = min(analysis.group_sizes.values()) if analysis.group_sizes else 0
    if smallest < 15:
        logger.warning(
            'Smallest group has n=%d. Each edge is a correlation over that many '
            'subjects, so treat these networks as exploratory.', smallest,
        )

    if 'abs-distance' in args.tests:
        analysis.run_abs_distance(n_perm=args.n_perm, seed=args.seed,
                                  n_workers=args.n_workers)
    if 'graph' in args.tests:
        analysis.run_graph_metrics(n_perm=min(args.n_perm, 1000), seed=args.seed,
                                   n_workers=args.n_workers)
    if 'nbs' in args.tests:
        analysis.run_nbs(n_perm=args.n_perm, seed=args.seed,
                         n_workers=args.n_workers, posthoc=True)
    if 'territory' in args.tests:
        if analysis.territory_cols:
            analysis.run_territory(n_perm=args.n_perm, seed=args.seed)
        else:
            logger.warning('No territory nodes for this node set; skipping')

    logger.info('Structural covariance complete: %s', analysis.covnet_root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
