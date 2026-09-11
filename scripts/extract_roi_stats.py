#!/usr/bin/env python3
"""
Extract per-ROI distributions of parametric maps from the SIGMA parcellation.

Where `extract_roi_means.py` reports one number per ROI, this reports the
distribution: mean, SD, median and percentiles. A mean hides *where* in an ROI a
change happened, and demyelination is patchy — the lower percentiles of MWF move
before the mean does.

Coverage is handled by neurofaune.network.roi_extraction: voxels outside the
acquisition slab are excluded rather than averaged in as zeros. That matters a lot
here — measured on this data, corr(coverage, ROI mean) was 0.932 including zeros
against 0.03 excluding them. Each session's own `space-SIGMA_desc-brain_mask` is
used as the coverage mask when present, which is stricter than the non-zero
fallback because it can tell "outside the slab" from a genuine zero.

**--common-coverage goes further, and for a group comparison you probably want it.**
Excluding out-of-slab voxels fixes the *amount* of an ROI that is averaged; it does
not fix *which part*. When the slab sits differently between scans, different
sessions average different sub-volumes of the same ROI, and if those sub-volumes
differ in the metric then coverage is still confounded with the value. Measured on
this cohort at baseline: corr(ROI coverage, corpus-callosum MWF) = +0.53 (p<0.001)
even after out-of-slab voxels were excluded, with the control group systematically
better covered than the treated group. `--common-coverage` intersects every
session's mask first and measures every session over that identical sub-volume, so
the comparison is like for like. It costs volume — only what all sessions share
survives — which is the honest price.

Output is one tidy long CSV: subject, session, metric, roi_name, statistic, value,
plus the per-ROI coverage so a reader can threshold on it.

Usage:
    uv run python scripts/extract_roi_stats.py \
        --derivatives-dir /mnt/arborea/cuprizone/preprocessing/derivatives \
        --parcellation /mnt/arborea/cuprizone/preprocessing/atlas/SIGMA_study_space/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
        --labels-csv /mnt/arborea/atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv \
        --modality msme --metrics MWF IWF T2 \
        --output-dir /mnt/arborea/cuprizone/network/roi_stats
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.roi_extraction import (  # noqa: E402
    DEFAULT_PERCENTILES,
    discover_sigma_metrics,
    extract_roi_stats,
    load_parcellation,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Which subdirectory a modality's coverage mask lives in, alongside its maps.
MODALITY_DIRS = {'msme': 'msme', 'dwi': 'dwi', 'func': 'func', 'anat': 'anat'}


def coverage_mask_for(derivatives_dir: Path, subject: str, session: str,
                      modality: str, shape) -> np.ndarray | None:
    """The session's own brain mask in SIGMA space, or None to fall back to non-zero.

    Per-modality on purpose: an MSME slab and a DWI slab reach different parts of
    the atlas, so a single shared mask would either overstate MSME coverage or
    understate DWI's.
    """
    subdir = MODALITY_DIRS.get(modality, modality)
    path = (derivatives_dir / subject / session / subdir
            / f'{subject}_{session}_space-SIGMA_desc-brain_mask.nii.gz')
    if not path.exists():
        return None
    mask = np.asarray(nib.load(str(path)).dataobj) > 0
    if mask.shape != shape:
        logger.warning('%s %s: coverage mask is %s but the map is %s; falling back '
                       'to the non-zero rule', subject, session, mask.shape, shape)
        return None
    return mask


def main():
    parser = argparse.ArgumentParser(
        description='Extract per-ROI distributions of parametric maps')
    parser.add_argument('--derivatives-dir', type=Path, required=True)
    parser.add_argument('--parcellation', type=Path, required=True)
    parser.add_argument('--labels-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--modality', required=True,
                        choices=sorted(MODALITY_DIRS), help='where the maps live')
    parser.add_argument('--metrics', nargs='+', required=True,
                        help='e.g. MWF IWF T2 (msme) or FA MD AD RD MK ODI (dwi)')
    parser.add_argument('--min-coverage', type=float, default=0.0,
                        help='NaN out any ROI covered below this fraction (0-1); '
                             'coverage is reported regardless so you can threshold later')
    parser.add_argument('--percentiles', type=float, nargs='+',
                        default=list(DEFAULT_PERCENTILES))
    parser.add_argument('--no-coverage-mask', action='store_true',
                        help='use the non-zero fallback instead of the brain mask')
    parser.add_argument('--common-coverage', action='store_true',
                        help='measure every session over the INTERSECTION of all '
                             'sessions coverage masks, so a group difference cannot '
                             'come from the slab sitting differently between scans')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parcellation, labels_df = load_parcellation(args.parcellation, args.labels_csv)

    found = discover_sigma_metrics(args.derivatives_dir, args.modality, args.metrics)
    if not found:
        logger.error('No space-SIGMA %s maps found for %s under %s',
                     args.metrics, args.modality, args.derivatives_dir)
        return 1
    logger.info('Found %d (session x metric) map(s) for %s', len(found), args.metrics)

    shared = None
    if args.common_coverage:
        shared = np.ones(parcellation.shape, bool)
        seen = set()
        for entry in found:
            key = (entry['subject'], entry['session'])
            if key in seen:
                continue
            seen.add(key)
            mask = coverage_mask_for(args.derivatives_dir, entry['subject'],
                                     entry['session'], args.modality,
                                     parcellation.shape)
            if mask is None:
                logger.warning('%s %s has no coverage mask; --common-coverage needs '
                               'one for every session', *key)
                continue
            shared &= mask
        in_brain = int((shared & (parcellation > 0)).sum())
        total_brain = int((parcellation > 0).sum())
        logger.info('Common coverage across %d session(s): %d of %d atlas-brain '
                    'voxels (%.1f%%)', len(seen), in_brain, total_brain,
                    100.0 * in_brain / max(1, total_brain))

    frames, no_mask = [], 0
    for index, entry in enumerate(found, start=1):
        subject, session = entry['subject'], entry['session']
        metric, path = entry['metric'], Path(entry['path'])
        arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
        if arr.ndim > 3:
            logger.warning('%s: %dD map, skipped (use a 3D derived metric)',
                           path.name, arr.ndim)
            continue
        if arr.shape != parcellation.shape:
            logger.warning('%s is %s but the parcellation is %s; skipped',
                           path.name, arr.shape, parcellation.shape)
            continue

        if shared is not None:
            mask = shared
        elif args.no_coverage_mask:
            mask = None
        else:
            mask = coverage_mask_for(args.derivatives_dir, subject, session,
                                     args.modality, arr.shape)
            if mask is None:
                no_mask += 1

        stats = extract_roi_stats(arr, parcellation, labels_df, coverage_mask=mask,
                                  min_coverage=args.min_coverage,
                                  percentiles=args.percentiles)
        stats.insert(0, 'subject', subject)
        stats.insert(1, 'session', session)
        stats.insert(2, 'metric', metric)
        # Self-describing: the same metric under a different coverage rule is a
        # different measurement, and a consumer must not be able to merge them.
        stats.insert(3, 'coverage_mode',
                     'common' if args.common_coverage else 'per_session')
        frames.append(stats)
        if index % 25 == 0 or index == len(found):
            logger.info('  [%d/%d] %s %s %s', index, len(found), subject, session, metric)

    if not frames:
        logger.error('Nothing extracted')
        return 1

    wide = pd.concat(frames, ignore_index=True)
    suffix = '_common' if args.common_coverage else ''
    wide_path = args.output_dir / f'roi_stats_{args.modality}{suffix}.csv'
    wide.to_csv(wide_path, index=False)

    # Long format for the modelling side: one row per statistic.
    stat_columns = ['mean', 'sd', 'median'] + [f'p{int(q)}' for q in args.percentiles]
    long = wide.melt(
        id_vars=['subject', 'session', 'metric', 'coverage_mode', 'roi_name',
                 'label_id', 'n_voxels', 'n_covered', 'coverage'],
        value_vars=[c for c in stat_columns if c in wide.columns],
        var_name='statistic', value_name='value')
    long_path = args.output_dir / f'roi_stats_{args.modality}{suffix}_long.csv'
    long.to_csv(long_path, index=False)

    covered = wide['coverage'].dropna()
    summary = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'modality': args.modality,
        'metrics': args.metrics,
        'n_session_metric_pairs': int(wide.groupby(['subject', 'session', 'metric'])
                                      .ngroups),
        'n_rois': int(wide['roi_name'].nunique()),
        'percentiles': args.percentiles,
        'min_coverage': args.min_coverage,
        'coverage_mask': ('common-intersection' if args.common_coverage
                          else 'nonzero-fallback' if args.no_coverage_mask
                          else 'per-session-brain_mask'),
        'sessions_without_a_coverage_mask': no_mask,
        'median_roi_coverage': float(covered.median()) if len(covered) else None,
        'pct_roi_measurements_fully_covered':
            float(100.0 * (covered >= 0.999).mean()) if len(covered) else None,
        'outputs': {'wide': str(wide_path), 'long': str(long_path)},
    }
    summary_path = args.output_dir / f'roi_stats_{args.modality}{suffix}_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info('Wrote %s and %s', wide_path, long_path)
    logger.info('Median ROI coverage %.3f; %.1f%% of ROI measurements fully covered',
                summary['median_roi_coverage'] or float('nan'),
                summary['pct_roi_measurements_fully_covered'] or float('nan'))
    if no_mask:
        logger.warning('%d map(s) had no space-SIGMA brain mask and used the '
                       'non-zero fallback, which cannot tell "outside the slab" '
                       'from a genuine zero', no_mask)

    try:
        from neurofaune.reporting import register as report_register

        analysis_root = args.output_dir.parent
        report_register(
            analysis_root=analysis_root,
            entry_id=f'roi_stats_{args.modality}{suffix}',
            analysis_type='roi_stats',
            display_name=f'ROI distributions ({args.modality.upper()}: '
                         f"{', '.join(args.metrics)})",
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={k: summary[k] for k in
                           ('metrics', 'n_session_metric_pairs', 'n_rois',
                            'median_roi_coverage')},
            source_summary_json=str(summary_path.relative_to(analysis_root)),
        )
    except Exception as exc:  # reporting is optional plumbing
        logger.debug('Reporting registration skipped: %s', exc)

    return 0


if __name__ == '__main__':
    sys.exit(main())
