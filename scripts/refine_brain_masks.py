#!/usr/bin/env python3
"""
Second-pass atlas-guided refinement of brain masks, with QC.

Run AFTER an initial skull strip and the registration it enables:

    initial strip -> register to atlas -> this script -> (optionally re-register)

For every session it warps the atlas brain mask through the existing transform
chain, refines the boundary against the UNSTRIPPED anatomical, unions in atlas
voxels that carry real tissue, gates the result, and writes a comparison montage
showing the old and new outlines on the full image.

Nothing is overwritten unless --apply is given; by default the refined masks land
beside the originals with a `desc-refinedbrain` suffix so the montages can be
reviewed first.

Usage:
    uv run python scripts/refine_brain_masks.py \
        --derivatives-dir /mnt/arborea/cuprizone/preprocessing/derivatives \
        --bids-dir /mnt/arborea/cuprizone/preprocessing/bids \
        --transforms-dir /mnt/arborea/cuprizone/preprocessing/transforms \
        --templates-dir /mnt/arborea/cuprizone/preprocessing/templates/anat \
        --atlas-dir /mnt/arborea/cuprizone/preprocessing/atlas/SIGMA_study_space \
        --output-dir /mnt/arborea/cuprizone/preprocessing/qc/mask_refinement
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurofaune.preprocess.qc.skull_strip_qc import (  # noqa: E402
    plot_mask_comparison_mosaic,
)
from neurofaune.preprocess.utils.atlas_guided_strip import (  # noqa: E402
    DEFAULT_TARGET_COVERAGE,
    DEFAULT_TOLERANCE_MM3,
    compute_brain_mask_qc,
    refine_iterative,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transform_chain(transforms_dir, templates_dir, sub, ses):
    """Atlas -> subject transforms, plus which entries to invert.

    The stored chain runs subject -> template -> atlas, so it is applied inverted
    here. The ordering was established by checking which composition reproduces the
    existing parcellation: this one matches at 97.2-97.3% of voxels (brain Dice
    0.908), while every forward ordering gives Dice below 0.01.
    """
    tp = ses.replace('ses-', '')
    s2t = Path(transforms_dir) / sub / ses
    t2s = Path(templates_dir) / tp / 'transforms'
    parts = [s2t / f'{sub}_{ses}_T2w_to_template_0GenericAffine.mat',
             s2t / f'{sub}_{ses}_T2w_to_template_1InverseWarp.nii.gz',
             t2s / 'tpl-to-SIGMA_0GenericAffine.mat',
             t2s / 'tpl-to-SIGMA_1InverseWarp.nii.gz']
    if not all(p.exists() for p in parts):
        return None, None
    return [str(p) for p in parts], [True, False, True, False]


def warp_atlas_mask(atlas_mask, reference, transforms, invert, output):
    """Warp the atlas brain mask into subject space with antsApplyTransforms.

    The CLI rather than a Python binding, matching how neurofaune applies
    transforms elsewhere (see tractography/fivett.py) and avoiding a heavyweight
    dependency for one call. antsApplyTransforms applies -t entries in reverse
    order, which is why the chain is passed as given.
    """
    command = ['antsApplyTransforms', '-d', '3', '-i', str(atlas_mask),
               '-r', str(reference), '-o', str(output),
               '-n', 'NearestNeighbor', '--float', '1']
    for path, flip in zip(transforms, invert, strict=True):
        command += ['-t', f'[{path},1]' if flip else str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'antsApplyTransforms failed:\n{result.stderr[-600:]}')
    return output


def _register_fn(args, anat, sub, ses, raw_p, work):
    """Build register(mask) -> (atlas_mask, parcellation) for the iterative loop.

    Brain-to-brain registration of the atlas onto the subject stripped with the
    current mask. Well conditioned, unlike registering the atlas's full head to the
    subject's -- that settles either on the brain or on a scaled-up fit to the wider
    subject FOV, and repeats of the identical command returned 2243, 3956, 3986 and
    2394 mm3.

    Uses antsRegistrationSyN.sh rather than a Python binding, matching how
    neurofaune drives ANTs elsewhere and avoiding a heavyweight dependency.
    """
    template = args.atlas_dir / 'SIGMA_InVivo_Brain_Template_Masked.nii.gz'
    atlas_mask = args.atlas_dir / 'SIGMA_InVivo_Brain_Mask.nii.gz'
    labels = args.atlas_dir / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz'
    raw_img = nib.load(str(raw_p))
    raw = np.asarray(raw_img.dataobj, dtype=np.float32)
    counter = {'n': 0}

    def register(mask):
        counter['n'] += 1
        tag = f'{sub}_{ses}_it{counter["n"]}'
        stripped = work / f'{tag}_stripped.nii.gz'
        nib.save(nib.Nifti1Image((raw * mask).astype(np.float32), raw_img.affine,
                                 raw_img.header), str(stripped))
        prefix = work / f'{tag}_'
        result = subprocess.run(
            ['antsRegistrationSyN.sh', '-d', '3', '-f', str(stripped),
             '-m', str(template), '-o', str(prefix), '-t', 's', '-n', '4'],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'antsRegistrationSyN failed:\n{result.stderr[-600:]}')

        transforms = [f'{prefix}1Warp.nii.gz', f'{prefix}0GenericAffine.mat']
        out = {}
        for name, source, interp in (('mask', atlas_mask, 'NearestNeighbor'),
                                     ('labels', labels, 'GenericLabel')):
            dest = work / f'{tag}_{name}.nii.gz'
            command = ['antsApplyTransforms', '-d', '3', '-i', str(source),
                       '-r', str(stripped), '-o', str(dest), '-n', interp, '--float', '1']
            for t in transforms:
                command += ['-t', t]
            r2 = subprocess.run(command, capture_output=True, text=True)
            if r2.returncode != 0:
                raise RuntimeError(f'antsApplyTransforms failed:\n{r2.stderr[-400:]}')
            out[name] = np.asarray(nib.load(str(dest)).dataobj)
        return out['mask'] > 0.5, out['labels'].astype(np.int32)

    return register


def find_raw(bids_dir, sub, ses, shape):
    """The unstripped anatomical whose grid matches the derivatives."""
    for p in sorted((Path(bids_dir) / sub / ses / 'anat').glob(
            f'{sub}_{ses}_run-*_T2w.nii.gz')):
        if nib.load(str(p)).shape[:3] == shape:
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description='Atlas-guided brain mask refinement')
    parser.add_argument('--derivatives-dir', type=Path, required=True)
    parser.add_argument('--bids-dir', type=Path, required=True)
    parser.add_argument('--transforms-dir', type=Path, required=True)
    parser.add_argument('--templates-dir', type=Path, required=True)
    parser.add_argument('--atlas-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true',
                        help='overwrite desc-brain_mask (default: write '
                             'desc-refinedbrain_mask alongside and leave the original)')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='restrict to these subject ids (e.g. sub-4C sub-10C)')
    parser.add_argument('--montage-every', type=int, default=1,
                        help='write a comparison montage for every Nth session')
    parser.add_argument('--iterations', type=int, default=1,
                        help='alternate registration and refinement up to N times. '
                             '1 (default) reuses the existing transform and does not '
                             're-register. >1 re-registers between passes, which is '
                             'what makes it converge -- refinement alone drifts '
                             'outward. 3 is normally ample; the tolerance stops it '
                             'sooner.')
    parser.add_argument('--target-coverage', type=float, default=DEFAULT_TARGET_COVERAGE,
                        help='stop once this fraction of the fixed reference '
                             'parcellation is inside the mask')
    parser.add_argument('--tolerance-mm3', type=float, default=DEFAULT_TOLERANCE_MM3,
                        help='stop when the volume change falls below this')
    args = parser.parse_args()

    figures = args.output_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    atlas_mask_p = args.atlas_dir / 'SIGMA_InVivo_Brain_Mask.nii.gz'
    work = Path(tempfile.mkdtemp(prefix='mask_refine_'))

    sessions = sorted({(p.parent.parent.parent.name, p.parent.parent.name)
                       for p in args.derivatives_dir.glob(
                           'sub-*/ses-*/anat/*_atlas-*_dseg.nii.gz')})
    if args.subjects:
        wanted = set(args.subjects)
        sessions = [(sub, ses) for sub, ses in sessions if sub in wanted]
    rows = []
    for i, (sub, ses) in enumerate(sessions, 1):
        anat = args.derivatives_dir / sub / ses / 'anat'
        dseg_p = anat / f'{sub}_{ses}_atlas-SIGMA_dseg.nii.gz'
        mask_p = anat / f'{sub}_{ses}_desc-brain_mask.nii.gz'
        if not mask_p.exists():
            continue
        tl, inv = transform_chain(args.transforms_dir, args.templates_dir, sub, ses)
        if tl is None:
            logger.warning('%s %s: transform chain incomplete; skipped', sub, ses)
            continue

        dseg_img = nib.load(str(dseg_p))
        parcellation = np.asarray(dseg_img.dataobj).astype(np.int32)
        raw_p = find_raw(args.bids_dir, sub, ses, parcellation.shape)
        if raw_p is None:
            logger.warning('%s %s: no unstripped T2w on the derivatives grid; skipped',
                           sub, ses)
            continue
        raw = np.asarray(nib.load(str(raw_p)).dataobj, dtype=np.float32)
        current = np.asarray(nib.load(str(mask_p)).dataobj) > 0

        warped_p = warp_atlas_mask(
            atlas_mask_p, anat / f'{sub}_{ses}_desc-preproc_T2w.nii.gz',
            tl, inv, work / f'{sub}_{ses}_seed.nii.gz')
        seed = np.asarray(nib.load(str(warped_p)).dataobj) > 0.5

        voxel_mm3 = float(np.prod(dseg_img.header.get_zooms()[:3])) / 1000.0
        # One code path regardless of `iterations`: with a reference guide, pass 1
        # refines against the existing transform chain and registers nothing, so
        # `--iterations 1` is the plain single pass with a history attached.
        refined, history = refine_iterative(
            raw, seed, _register_fn(args, anat, sub, ses, raw_p, work),
            voxel_mm3, reference_parcellation=parcellation, reference_guide=seed,
            iterations=args.iterations, target_coverage=args.target_coverage,
            tolerance_mm3=args.tolerance_mm3)
        qc = compute_brain_mask_qc(refined, parcellation, voxel_mm3,
                                   raw=raw, initial_mask=current)
        before = compute_brain_mask_qc(current, parcellation, voxel_mm3, raw=raw)

        name = 'desc-brain_mask' if args.apply else 'desc-refinedbrain_mask'
        out_p = anat / f'{sub}_{ses}_{name}.nii.gz'
        nib.save(nib.Nifti1Image(refined.astype(np.uint8), dseg_img.affine,
                                 dseg_img.header), str(out_p))

        montage = ''
        if i % args.montage_every == 0:
            montage = str(plot_mask_comparison_mosaic(
                raw, {'current': current, 'refined': refined},
                sub, ses, figures))

        rows.append({'subject': sub, 'session': ses,
                     'volume_before_mm3': before['volume_mm3'],
                     'volume_after_mm3': qc['volume_mm3'],
                     'atlas_coverage_before': before['atlas_coverage'],
                     'atlas_coverage_after': qc['atlas_coverage'],
                     'non_brain_before_mm3': before['non_brain_mm3'],
                     'non_brain_after_mm3': qc['non_brain_mm3'],
                     'tissue_fraction_before': before['tissue_fraction'],
                     'tissue_fraction_after': qc['tissue_fraction'],
                     'dice_with_initial': qc['dice_with_initial'],
                     'n_iterations': len(history) if history else 1,
                     # which pass was kept, and whether the last one was discarded
                     # for contracting -- see refine_iterative's docstring
                     'selected_iteration': next(
                         (h['iteration'] for h in history if h['selected']), 1),
                     'reached_target_coverage': (
                         history[-1]['atlas_coverage'] >= args.target_coverage
                         if history else None),
                     'coverage_by_iteration': ';'.join(
                         f"{h['atlas_coverage']:.4f}" for h in history),
                     'n_registrations': sum(h['registered'] for h in history),
                     'qc_passed': qc['passed'],
                     'qc_failures': '; '.join(qc['failures']),
                     'mask': str(out_p), 'montage': montage})
        flag = 'ok  ' if qc['passed'] else 'FAIL'
        logger.info('[%d/%d] %s %s %s  %.0f -> %.0f mm3, coverage %.1f%% -> %.1f%%',
                    i, len(sessions), sub, ses, flag,
                    before['volume_mm3'], qc['volume_mm3'],
                    100 * before['atlas_coverage'], 100 * qc['atlas_coverage'])
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        logger.error('No sessions refined')
        return 1

    df = pd.DataFrame(rows)
    table = args.output_dir / 'mask_refinement.csv'
    df.to_csv(table, index=False)

    summary = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'n_sessions': len(df),
        'n_qc_failed': int((~df.qc_passed).sum()),
        'applied_in_place': bool(args.apply),
        'volume_mm3': {'before_median': float(df.volume_before_mm3.median()),
                       'after_median': float(df.volume_after_mm3.median()),
                       'before_sd': float(df.volume_before_mm3.std()),
                       'after_sd': float(df.volume_after_mm3.std())},
        'atlas_coverage': {'before_median': float(df.atlas_coverage_before.median()),
                           'after_median': float(df.atlas_coverage_after.median())},
        'non_brain_mm3': {'before_median': float(df.non_brain_before_mm3.median()),
                          'after_median': float(df.non_brain_after_mm3.median())},
        'tissue_fraction': {'before_median': float(df.tissue_fraction_before.median()),
                            'after_median': float(df.tissue_fraction_after.median())},
        'dice_with_initial': {'median': float(df.dice_with_initial.median()),
                              'min': float(df.dice_with_initial.min())},
    }
    with open(args.output_dir / 'mask_refinement_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info('Wrote %s and %d montage(s) in %s', table,
                int((df.montage != '').sum()), figures)
    logger.info('volume   median %.0f -> %.0f mm3 (SD %.0f -> %.0f)',
                summary['volume_mm3']['before_median'],
                summary['volume_mm3']['after_median'],
                summary['volume_mm3']['before_sd'], summary['volume_mm3']['after_sd'])
    logger.info('coverage median %.1f%% -> %.1f%%',
                100 * summary['atlas_coverage']['before_median'],
                100 * summary['atlas_coverage']['after_median'])
    logger.info('non-brain median %.0f -> %.0f mm3',
                summary['non_brain_mm3']['before_median'],
                summary['non_brain_mm3']['after_median'])
    logger.info('tissue fraction median %.1f%% -> %.1f%% (registration-independent)',
                100 * summary['tissue_fraction']['before_median'],
                100 * summary['tissue_fraction']['after_median'])
    logger.info('Dice vs initial mask: median %.3f, min %.3f',
                summary['dice_with_initial']['median'],
                summary['dice_with_initial']['min'])
    if summary['n_qc_failed']:
        logger.warning('%d session(s) FAILED QC -- review their montages before '
                       'using them: %s', summary['n_qc_failed'],
                       ', '.join(df[~df.qc_passed].subject + '/' + df[~df.qc_passed].session))
    if not args.apply:
        logger.info('Originals untouched; refined masks written as '
                    'desc-refinedbrain_mask. Re-run with --apply to replace them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
