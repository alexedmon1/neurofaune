#!/usr/bin/env python3
"""Rebuild desc-skullstrip_T2w / desc-preproc_T2w from the refined brain masks.

Why this cannot read the derivatives it replaces
------------------------------------------------
``desc-preproc_T2w`` is not an unstripped volume. anat_preprocess writes it as the
*already skull-stripped* brain scaled by ``anatomical.intensity_normalization.factor``
(1000), and ``desc-skullstrip_T2w`` is that same brain unscaled. Both are therefore
zero everywhere the old mask excluded -- including every voxel the refinement
recovered. Measured on sub-10C/ses-1 the refined mask adds 4171 voxels and all 4171
are exactly zero in ``desc-preproc_T2w``, so masking it with the new mask would
reproduce the old strip and silently discard the olfactory bulb this whole exercise
was about.

The unstripped, N4-corrected volume survives in the work tree as
``work/<sub>/<ses>/anat/<sub>_<ses>_T2w_n4.nii.gz``; all 92 sessions have one. Those
same 4171 voxels sit at 6.7x background there, which is the tissue being recovered.

Reconstruction is verified rather than assumed: inside the OLD mask
``preproc / n4`` must equal the normalisation factor, which pins that `n4` really is
the source these products were derived from. A session failing that check is
skipped, not guessed at.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NORM_FACTOR = 1000.0
RATIO_TOL = 0.01


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--derivatives-dir', type=Path, required=True)
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--old-masks-dir', type=Path, required=True,
                    help='backup of the pre-refinement desc-brain_mask tree, for the '
                         'preproc/n4 consistency check')
    ap.add_argument('--norm-factor', type=float, default=NORM_FACTOR)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    masks = sorted(args.derivatives_dir.glob('sub-*/ses-*/anat/*_desc-brain_mask.nii.gz'))
    logger.info('%d session(s)', len(masks))
    written = skipped = 0

    for i, mask_p in enumerate(masks, 1):
        stem = mask_p.name.replace('_desc-brain_mask.nii.gz', '')
        sub, ses = stem.split('_')[0], stem.split('_')[1]
        anat = mask_p.parent
        n4_p = args.work_dir / sub / ses / 'anat' / f'{stem}_T2w_n4.nii.gz'
        old_p = args.old_masks_dir / sub / ses / 'anat' / mask_p.name
        preproc_p = anat / f'{stem}_desc-preproc_T2w.nii.gz'

        if not n4_p.exists():
            logger.warning('[%d/%d] %s: no N4 volume; skipped', i, len(masks), stem)
            skipped += 1
            continue

        n4_img = nib.load(str(n4_p))
        n4 = np.asarray(n4_img.dataobj, dtype=np.float32)
        new = np.asarray(nib.load(str(mask_p)).dataobj) > 0
        if n4.shape != new.shape:
            logger.warning('[%d/%d] %s: N4 %s vs mask %s; skipped',
                           i, len(masks), stem, n4.shape, new.shape)
            skipped += 1
            continue

        # Pin that n4 really is what the existing products came from.
        if old_p.exists() and preproc_p.exists():
            old = np.asarray(nib.load(str(old_p)).dataobj) > 0
            pre = np.asarray(nib.load(str(preproc_p)).dataobj, dtype=np.float32)
            denom = float(n4[old].sum())
            ratio = float(pre[old].sum()) / denom if denom else float('nan')
            if not np.isfinite(ratio) or abs(ratio - args.norm_factor) > args.norm_factor * RATIO_TOL:
                logger.warning('[%d/%d] %s: preproc/n4 = %.1f inside the old mask, '
                               'expected %.0f -- N4 is not the source; skipped',
                               i, len(masks), stem, ratio, args.norm_factor)
                skipped += 1
                continue
            gained = int((new & ~old).sum())
        else:
            gained = -1

        stripped = (n4 * new).astype(np.float32)
        if not args.dry_run:
            nib.save(nib.Nifti1Image(stripped, n4_img.affine, n4_img.header),
                     str(anat / f'{stem}_desc-skullstrip_T2w.nii.gz'))
            nib.save(nib.Nifti1Image(stripped * args.norm_factor, n4_img.affine,
                                     n4_img.header), str(preproc_p))
        written += 1
        logger.info('[%d/%d] %s ok  %d voxel(s) in mask, %s recovered vs old',
                    i, len(masks), stem, int(new.sum()),
                    str(gained) if gained >= 0 else 'n/a')

    logger.info('%s %d session(s), skipped %d',
                'would write' if args.dry_run else 'wrote', written, skipped)
    if skipped:
        logger.warning('%d session(s) skipped -- templates must not be rebuilt until '
                       'every session is consistent', skipped)


if __name__ == '__main__':
    main()
