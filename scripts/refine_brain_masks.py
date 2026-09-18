#!/usr/bin/env python3
"""
Second-pass atlas-guided refinement of brain masks, with QC, for an existing study.

The same stage runs automatically in `batch_preprocess_anat.py` phase 2 when
`anatomical.skull_strip.refine.method` is `atlas_iterative`; this script re-runs it
over sessions that are already registered and have a propagated atlas. All logic
lives in `neurofaune.preprocess.workflows.anat_mask_refinement`.

    initial strip -> register to atlas -> this script -> (optionally re-register)

Settings come from the study config (`anatomical.skull_strip.refine.*`); the flags
below override them. Nothing is overwritten unless --apply is given (or
`refine.apply: true`): by default refined masks land beside the originals as
`desc-refinedbrain_mask` so the montages can be reviewed first.

Usage:
    uv run python scripts/refine_brain_masks.py \
        --config /mnt/arborea/cuprizone/preprocessing/config.yaml \
        --bids-dir /mnt/arborea/cuprizone/preprocessing/bids
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.config import get_config_value, load_config  # noqa: E402
from neurofaune.preprocess.workflows.anat_mask_refinement import (  # noqa: E402
    PARCELLATION_NAMES,
    get_refine_settings,
    run_brain_mask_refinement,
    write_refinement_summary,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Atlas-guided brain mask refinement')
    parser.add_argument('--config', type=Path, required=True, help='study config.yaml')
    parser.add_argument('--bids-dir', type=Path, default=None,
                        help='BIDS root holding the unstripped T2w '
                             '(default: paths.bids from the config)')
    parser.add_argument('--study-root', type=Path, default=None,
                        help='default: paths.study_root from the config')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='CSV/summary/montages (default: <study>/qc/mask_refinement)')
    parser.add_argument('--apply', action='store_true',
                        help='replace desc-brain_mask and re-strip the T2w '
                             '(default: refine.apply from the config)')
    parser.add_argument('--direct-to-sigma', action='store_true',
                        help='sessions were registered straight to SIGMA, no template')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='restrict to these subject ids (e.g. sub-4C sub-10C)')
    parser.add_argument('--montage-every', type=int, default=1,
                        help='write a comparison montage for every Nth session')
    parser.add_argument('--iterations', type=int, default=None,
                        help='override refine.iterations. 1 reuses the existing '
                             'transform and does not re-register; >1 re-registers from '
                             'pass 2 and keeps the best-covering pass')
    parser.add_argument('--target-coverage', type=float, default=None,
                        help='override refine.target_coverage')
    parser.add_argument('--tolerance-mm3', type=float, default=None,
                        help='override refine.tolerance_mm3 (diagnostic only)')
    args = parser.parse_args()

    config = load_config(args.config)
    study_root = args.study_root or Path(get_config_value(config, 'paths.study_root'))
    bids_dir = args.bids_dir or Path(get_config_value(config, 'paths.bids'))
    output_dir = args.output_dir or study_root / 'qc' / 'mask_refinement'

    settings = get_refine_settings(config)
    if settings['method'] == 'none':
        logger.warning('refine.method is "none" in the config; running atlas_iterative '
                       'because this script was invoked explicitly')
        settings['method'] = 'atlas_iterative'
    for key in ('iterations', 'target_coverage', 'tolerance_mm3'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.apply:
        settings['apply'] = True

    derivatives = study_root / 'derivatives'
    sessions = sorted({(p.parent.parent.parent.name, p.parent.parent.name)
                       for name in PARCELLATION_NAMES
                       for p in derivatives.glob(f'sub-*/ses-*/anat/*_{name}.nii.gz')})
    if args.subjects:
        wanted = set(args.subjects)
        sessions = [(sub, ses) for sub, ses in sessions if sub in wanted]

    rows = []
    for i, (sub, ses) in enumerate(sessions, 1):
        row = run_brain_mask_refinement(
            config, sub, ses, study_root,
            raw_t2w=sorted((bids_dir / sub / ses / 'anat').glob('*_T2w.nii.gz')),
            direct=args.direct_to_sigma, qc_dir=output_dir,
            montage=i % args.montage_every == 0, settings=settings)
        rows.append(row)
        if row['status'] == 'refined':
            logger.info('[%d/%d] %s %s %s', i, len(sessions), sub, ses,
                        'ok' if row['qc_passed'] else 'FAIL')
        if args.limit and sum(r['status'] == 'refined' for r in rows) >= args.limit:
            break

    table = write_refinement_summary(rows, output_dir)
    if table is None:
        logger.error('No sessions refined')
        return 1
    logger.info('Wrote %s', table)
    failed = [f"{r['subject']}/{r['session']}" for r in rows
              if r['status'] == 'refined' and not r['qc_passed']]
    if failed:
        logger.warning('%d session(s) FAILED QC -- review their montages before '
                       'using them: %s', len(failed), ', '.join(failed))
    if not settings['apply']:
        logger.info('Originals untouched; refined masks written as '
                    'desc-refinedbrain_mask. Re-run with --apply to replace them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
