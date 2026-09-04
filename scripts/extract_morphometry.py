#!/usr/bin/env python3
"""
Extract composite (FreeSurfer-aseg-style) morphometry from the SIGMA parcellation.

Produces per-structure and per-region volumes for every subject-session that has
both a propagated parcellation and tissue posteriors. Volumes are partial-volume
weighted by the subject's own tissue posterior, so they are not a restatement of the
registration; see neurofaune.network.morphometry for why that matters on anisotropic
rodent anatomicals.

Optionally adds plane-restricted cortical thickness (--thickness), which is
EXPLORATORY: read neurofaune/network/thickness.py before using those numbers.

Usage:
    uv run python scripts/extract_morphometry.py \
        --derivatives-dir /mnt/arborea/cuprizone/preprocessing/derivatives \
        --parcellation /mnt/arborea/cuprizone/preprocessing/atlas/SIGMA_study_space/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
        --labels-csv /mnt/arborea/atlases/SIGMA/SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv \
        --voxel-scale 10 \
        --output-dir /mnt/arborea/cuprizone/network/morphometry
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.network.morphometry import (  # noqa: E402
    TISSUES,
    compute_subject_morphometry,
    load_structure_groups,
    normalise_labels,
    resolve_labels,
)
from neurofaune.network.roi_extraction import load_parcellation  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

RIBBON_STRUCTURE = 'cortical_gm'


def find_sessions(derivatives_dir: Path, tissue_dir: Path = None):
    """Yield ``(subject, session, dseg, {tissue: posterior})`` for usable sessions.

    Posteriors are taken from `tissue_dir` when given, else from the session's own
    ``label-*_probseg.nii.gz``. The source is reported per session because the two
    are not interchangeable: a KMeans-initialised segmentation computed inside a
    generous brain mask misplaces white matter (it scores the corpus callosum as
    grey), whereas the atlas-prior path does not.
    """
    for anat in sorted(derivatives_dir.glob('sub-*/ses-*/anat')):
        session_dir = anat.parent
        subject, session = session_dir.parent.name, session_dir.name
        stem = f'{subject}_{session}'

        dseg = next(iter(anat.glob(f'{stem}_atlas-*_dseg.nii.gz')), None)
        if dseg is None:
            logger.warning('%s: no propagated parcellation; skipped', stem)
            continue

        if tissue_dir:
            posteriors = {t: tissue_dir / f'{stem}_label-{t}_prob.nii.gz' for t in TISSUES}
            source = 'external'
        else:
            posteriors = {t: anat / f'{stem}_label-{t}_probseg.nii.gz' for t in TISSUES}
            source = 'derivatives'

        missing = [t for t, p in posteriors.items() if not p.exists()]
        if missing:
            logger.warning('%s: missing %s posterior(s); skipped', stem, missing)
            continue
        yield subject, session, dseg, posteriors, source


def main():
    parser = argparse.ArgumentParser(
        description='Extract composite morphometry from the SIGMA parcellation'
    )
    parser.add_argument('--derivatives-dir', type=Path, required=True,
                        help='Derivatives directory containing sub-*/ses-*/ folders')
    parser.add_argument('--parcellation', type=Path, required=True,
                        help='SIGMA parcellation NIfTI in study space')
    parser.add_argument('--labels-csv', type=Path, required=True,
                        help='SIGMA atlas labels CSV')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--groups', type=Path, default=None,
                        help='Composite structure definitions (default: SIGMA InVivo)')
    parser.add_argument('--tissue-dir', type=Path, default=None,
                        help='Directory of externally computed tissue posteriors '
                             '(<sub>_<ses>_label-{GM,WM,CSF}_prob.nii.gz)')
    parser.add_argument('--voxel-scale', type=float, default=1.0,
                        help='Linear voxel scaling in the headers (10 for neurofaune '
                             'x10-scaled rodent data); volumes are divided by its cube')
    parser.add_argument('--thickness', action='store_true',
                        help='Also compute plane-restricted cortical thickness '
                             '(EXPLORATORY - see neurofaune/network/thickness.py)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N sessions (smoke run)')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_structure_groups(args.groups)
    _, labels_df = load_parcellation(args.parcellation, args.labels_csv)
    labels_df = normalise_labels(labels_df, spec)

    ribbon_ids = set()
    if args.thickness:
        ribbon_ids = resolve_labels(labels_df, spec['structures'][RIBBON_STRUCTURE])
        logger.info('Thickness over %d cortical regions', len(ribbon_ids))

    structures, regions, asymmetry, thickness_rows = [], [], [], []
    sources = {}
    for i, (subject, session, dseg, posteriors, source) in enumerate(
        find_sessions(args.derivatives_dir, args.tissue_dir), start=1
    ):
        stem = f'{subject}_{session}'
        tables = compute_subject_morphometry(
            dseg=dseg, posterior_paths=posteriors,
            parcellation=args.parcellation, labels_csv=args.labels_csv,
            spec=spec, voxel_scale=args.voxel_scale,
        )
        for name, frame in tables.items():
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, 'subject', stem)
            frame['tissue_source'] = source
            {'structures': structures, 'regions': regions,
             'asymmetry': asymmetry}[name].append(frame)

        if args.thickness:
            from neurofaune.network.thickness import compute_subject_thickness

            frame = compute_subject_thickness(dseg, ribbon_ids, labels_df, args.voxel_scale)
            if not frame.empty:
                frame.insert(0, 'subject', stem)
                thickness_rows.append(frame)

        total = tables['structures'].query("structure == 'total_brain' and tissue == 'any'")
        sources[stem] = source
        logger.info('%s: %s posteriors, total_brain=%.1f mm3', stem, source,
                    float(total['volume_mm3'].iloc[0]) if len(total) else float('nan'))

        if args.limit and i >= args.limit:
            logger.info('Stopping at --limit %d', args.limit)
            break

    if not structures:
        logger.error('No sessions produced morphometry; nothing written')
        return 1

    written = {}
    for name, frames in (('structures', structures), ('regions', regions),
                         ('asymmetry', asymmetry), ('thickness', thickness_rows)):
        if not frames:
            continue
        path = args.output_dir / f'morphometry_{name}.csv'
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        written[name] = str(path)
        logger.info('Wrote %s', path)

    summary = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'n_sessions': len(sources),
        'voxel_scale': args.voxel_scale,
        'groups': str(args.groups or 'sigma_invivo (default)'),
        'n_structures': len(spec['structures']),
        'tissue_sources': sorted(set(sources.values())),
        'thickness': bool(thickness_rows),
        'outputs': written,
    }
    summary_path = args.output_dir / 'morphometry_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info('Summary: %s', summary_path)

    try:
        from neurofaune.reporting import register as report_register

        analysis_root = args.output_dir.parent
        report_register(
            analysis_root=analysis_root,
            entry_id='morphometry',
            analysis_type='morphometry',
            display_name='Composite morphometry (aseg-style volumes)',
            output_dir=str(args.output_dir.relative_to(analysis_root)),
            summary_stats={
                'n_sessions': summary['n_sessions'],
                'n_structures': summary['n_structures'],
                'tissue_sources': summary['tissue_sources'],
                'thickness': summary['thickness'],
            },
            source_summary_json=str(summary_path.relative_to(analysis_root)),
        )
    except Exception as exc:  # reporting is optional plumbing, not the deliverable
        logger.debug('Reporting registration skipped: %s', exc)

    return 0


if __name__ == '__main__':
    sys.exit(main())
