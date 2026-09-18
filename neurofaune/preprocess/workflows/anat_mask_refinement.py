"""Second-pass brain mask refinement as a pipeline stage.

`neurofaune.preprocess.utils.atlas_guided_strip` holds the image logic; this module
is the file-level stage that runs it inside anatomical preprocessing. It needs the
registration and atlas propagation to exist already, so it runs AFTER them:

    preprocess (initial strip) -> register -> propagate atlas -> refine (this)

Selected by `anatomical.skull_strip.refine.method` (`atlas_iterative` or `none`).

By default the refined mask is written beside the original as
`desc-refinedbrain_mask` and nothing downstream changes. With
`anatomical.skull_strip.refine.apply: true` it replaces `desc-brain_mask`, the
original is kept once as `desc-initialbrain_mask`, and `desc-skullstrip_T2w` /
`desc-preproc_T2w` are re-stripped from the N4-corrected image so the recovered
tissue is actually in them. Tissue probsegs and the T2w->template registration
still predate the refined mask; re-registration is deliberately not triggered
(see `refine_iterative`: it never improved coverage over 16 sessions).
"""

import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neurofaune.config import get_config_value
from neurofaune.preprocess.utils.atlas_guided_strip import (
    DEFAULT_ITERATIONS,
    DEFAULT_QC,
    DEFAULT_TARGET_COVERAGE,
    DEFAULT_TOLERANCE_MM3,
    REFINE_METHODS,
    compute_brain_mask_qc,
    refine_iterative,
)
from neurofaune.templates.sigma_warp import (
    inverse_transform_args,
    resolve_tpl_to_sigma_for_cohort,
)

logger = logging.getLogger(__name__)

# Derivatives carry voxel sizes scaled 10x for FSL/ANTs (check_and_scale_voxel_size),
# so one derivative voxel's volume in real mm3 is its header volume / 10**3.
VOXEL_SCALE = 10.0

# Parcellation names in use: `atlas-SIGMA_dseg` is what morphometry, tissue priors,
# MRS and 5TT read; batch_preprocess_anat.py writes `space-T2w_atlas-SIGMA`.
PARCELLATION_NAMES = ('atlas-SIGMA_dseg', 'space-T2w_atlas-SIGMA')


def get_refine_settings(config: dict[str, Any], modality: str = 'anatomical') -> dict[str, Any]:
    """Read `<modality>.skull_strip.refine.*` with the module defaults filled in."""
    key = f'{modality}.skull_strip.refine'
    method = get_config_value(config, f'{key}.method', default='none')
    if method not in REFINE_METHODS:
        raise ValueError(f'{key}.method must be one of {REFINE_METHODS}, got {method!r}')
    qc = dict(DEFAULT_QC)
    qc.update(get_config_value(config, f'{key}.qc', default={}) or {})
    qc['volume_mm3'] = tuple(qc['volume_mm3'])
    return {
        'method': method,
        'apply': bool(get_config_value(config, f'{key}.apply', default=False)),
        'iterations': int(get_config_value(config, f'{key}.iterations',
                                           default=DEFAULT_ITERATIONS)),
        'target_coverage': float(get_config_value(config, f'{key}.target_coverage',
                                                  default=DEFAULT_TARGET_COVERAGE)),
        'tolerance_mm3': float(get_config_value(config, f'{key}.tolerance_mm3',
                                                default=DEFAULT_TOLERANCE_MM3)),
        'tissue_factor': float(get_config_value(config, f'{key}.tissue_factor', default=2.0)),
        'qc': qc,
    }


def atlas_to_subject_chain(
    transforms_dir: Path,
    templates_dir: Path,
    subject: str,
    session: str,
    cohort: str | None = None,
    direct: bool = False,
) -> list[str] | None:
    """`-t` arguments taking an atlas-space image into subject T2w space.

    Returned in antsApplyTransforms order: each leg from `inverse_transform_args`,
    the subject->template leg before the template->SIGMA leg.

    Returns None if any required transform is missing.
    """
    subj = Path(transforms_dir) / subject / session
    if direct:
        affine = subj / f'{subject}_{session}_T2w_to_SIGMA_0GenericAffine.mat'
        inverse = subj / f'{subject}_{session}_T2w_to_SIGMA_1InverseWarp.nii.gz'
        if not affine.exists():
            return None
        return inverse_transform_args(affine, inverse)

    affine = subj / f'{subject}_{session}_T2w_to_template_0GenericAffine.mat'
    inverse = subj / f'{subject}_{session}_T2w_to_template_1InverseWarp.nii.gz'
    tpl = resolve_tpl_to_sigma_for_cohort(templates_dir, cohort or session.replace('ses-', ''))
    if not affine.exists() or not tpl['found']:
        return None
    return (inverse_transform_args(affine, inverse)
            + inverse_transform_args(tpl['affine'], tpl['inverse_warp']))


def apply_chain(source: Path, reference: Path, chain: Sequence[str], output: Path,
                interpolation: str = 'NearestNeighbor') -> Path:
    """antsApplyTransforms with `chain` passed through in order."""
    command = ['antsApplyTransforms', '-d', '3', '-i', str(source), '-r', str(reference),
               '-o', str(output), '-n', interpolation, '--float', '1']
    for t in chain:
        command += ['-t', t]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'antsApplyTransforms failed:\n{result.stderr[-600:]}')
    return Path(output)


def make_atlas_register(
    raw_img: nib.Nifti1Image,
    atlas_template: Path,
    atlas_mask: Path,
    atlas_labels: Path,
    work_dir: Path,
    tag: str,
    n_threads: int = 4,
) -> Callable[[np.ndarray], tuple]:
    """Build `register(mask) -> (atlas_mask, parcellation)` for passes 2+.

    Brain-to-brain SyN of the masked atlas template onto the subject stripped with
    the current mask. Only called when `iterations > 1`. Registering the atlas's
    full head instead is not usable: repeats of the identical command returned
    2243, 3956, 3986 and 2394 mm3.
    """
    raw = np.asarray(raw_img.dataobj, dtype=np.float32)
    counter = {'n': 0}

    def register(mask):
        counter['n'] += 1
        stem = f'{tag}_it{counter["n"]}'
        stripped = work_dir / f'{stem}_stripped.nii.gz'
        nib.save(nib.Nifti1Image((raw * mask).astype(np.float32), raw_img.affine,
                                 raw_img.header), str(stripped))
        prefix = work_dir / f'{stem}_'
        result = subprocess.run(
            ['antsRegistrationSyN.sh', '-d', '3', '-f', str(stripped),
             '-m', str(atlas_template), '-o', str(prefix), '-t', 's', '-n', str(n_threads)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'antsRegistrationSyN failed:\n{result.stderr[-600:]}')
        chain = [f'{prefix}1Warp.nii.gz', f'{prefix}0GenericAffine.mat']
        warped_mask = apply_chain(atlas_mask, stripped, chain,
                                  work_dir / f'{stem}_mask.nii.gz', 'NearestNeighbor')
        warped_labels = apply_chain(atlas_labels, stripped, chain,
                                    work_dir / f'{stem}_labels.nii.gz', 'GenericLabel')
        return (np.asarray(nib.load(str(warped_mask)).dataobj) > 0.5,
                np.asarray(nib.load(str(warped_labels)).dataobj).astype(np.int32))

    return register


def find_unstripped_t2w(candidates: Sequence[Path], shape: tuple) -> Path | None:
    """The unstripped T2w whose grid matches the derivatives, or None.

    A session can hold several T2w runs (e.g. a 5-slice scout beside the 41-slice
    anatomical); only the one on the derivatives grid is usable. 3D acquisitions
    resampled to 2D geometry have no such run and are skipped by the caller.
    """
    for p in sorted(Path(c) for c in candidates):
        if nib.load(str(p)).shape[:3] == tuple(shape[:3]):
            return p
    return None


def _find_parcellation(anat_dir: Path, subject: str, session: str) -> Path | None:
    for name in PARCELLATION_NAMES:
        p = anat_dir / f'{subject}_{session}_{name}.nii.gz'
        if p.exists():
            return p
    return None


def _find_n4(study_root: Path, subject: str, session: str) -> Path | None:
    # run_anatomical_preprocessing defaults to work/.../anat_preproc; studies that
    # pass their own work_dir commonly use work/.../anat.
    for sub in ('anat_preproc', 'anat'):
        p = study_root / 'work' / subject / session / sub / f'{subject}_{session}_T2w_n4.nii.gz'
        if p.exists():
            return p
    return None


def run_brain_mask_refinement(
    config: dict[str, Any],
    subject: str,
    session: str,
    study_root: Path,
    raw_t2w: Sequence[Path],
    cohort: str | None = None,
    direct: bool = False,
    parcellation_file: Path | None = None,
    n4_file: Path | None = None,
    apply: bool | None = None,
    qc_dir: Path | None = None,
    work_dir: Path | None = None,
    montage: bool = True,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine one session's T2w brain mask per `anatomical.skull_strip.refine`.

    Parameters
    ----------
    raw_t2w : sequence of Path
        Candidate UNSTRIPPED T2w images (e.g. every `*_T2w.nii.gz` in the BIDS anat
        dir); the one on the derivatives grid is used.
    apply : bool, optional
        Overrides `refine.apply` from the config.
    settings : dict, optional
        Overrides the config entirely (as returned by `get_refine_settings`).

    Returns
    -------
    dict
        `status` is 'refined', 'disabled' or 'skipped' (with `reason`); a refined
        session also carries the before/after QC metrics, the per-pass history
        and the paths written -- one row of `mask_refinement.csv`.
    """
    s = settings or get_refine_settings(config)
    if apply is not None:
        s = {**s, 'apply': bool(apply)}
    base = {'subject': subject, 'session': session}
    if s['method'] == 'none':
        return {**base, 'status': 'disabled'}

    study_root = Path(study_root)
    anat = study_root / 'derivatives' / subject / session / 'anat'
    mask_p = anat / f'{subject}_{session}_desc-brain_mask.nii.gz'
    initial_p = anat / f'{subject}_{session}_desc-initialbrain_mask.nii.gz'
    preproc_p = anat / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    parcellation_file = parcellation_file or _find_parcellation(anat, subject, session)

    def skipped(reason):
        logger.warning('%s %s: mask refinement skipped -- %s', subject, session, reason)
        return {**base, 'status': 'skipped', 'reason': reason}

    if not mask_p.exists() or not preproc_p.exists():
        return skipped('no preprocessed T2w / brain mask')
    if parcellation_file is None or not Path(parcellation_file).exists():
        return skipped('no atlas parcellation in subject space; propagate the atlas first')
    chain = atlas_to_subject_chain(study_root / 'transforms', study_root / 'templates',
                                   subject, session, cohort=cohort, direct=direct)
    if chain is None:
        return skipped('atlas->subject transform chain incomplete')

    atlas_mask_p = Path(get_config_value(config, 'atlas.study_space.brain_mask'))
    atlas_template_p = Path(get_config_value(config, 'atlas.study_space.template_masked'))
    atlas_labels_p = Path(get_config_value(config, 'atlas.study_space.parcellation'))
    if not atlas_mask_p.exists():
        return skipped(f'study-space atlas brain mask not found: {atlas_mask_p}')

    dseg_img = nib.load(str(parcellation_file))
    parcellation = np.asarray(dseg_img.dataobj).astype(np.int32)
    raw_p = find_unstripped_t2w(raw_t2w, parcellation.shape)
    if raw_p is None:
        return skipped('no unstripped T2w on the derivatives grid (3D acquisition?)')
    raw_img = nib.load(str(raw_p))
    raw = np.asarray(raw_img.dataobj, dtype=np.float32)

    if s['apply']:
        n4_file = n4_file or _find_n4(study_root, subject, session)
        if n4_file is None or nib.load(str(n4_file)).shape[:3] != parcellation.shape:
            return skipped('refine.apply needs the N4-corrected T2w on the derivatives '
                           'grid to re-strip from, and none was found')

    # The original first-pass mask, even when an earlier applied run replaced it.
    initial_src = initial_p if initial_p.exists() else mask_p
    initial = np.asarray(nib.load(str(initial_src)).dataobj) > 0
    voxel_mm3 = float(np.prod(dseg_img.header.get_zooms()[:3])) / VOXEL_SCALE ** 3

    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix='mask_refine_')
        work_dir = Path(tmp.name)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        seed_p = apply_chain(atlas_mask_p, preproc_p, chain,
                             work_dir / f'{subject}_{session}_seed.nii.gz')
        seed = np.asarray(nib.load(str(seed_p)).dataobj) > 0.5
        register = make_atlas_register(raw_img, atlas_template_p, atlas_mask_p,
                                       atlas_labels_p, work_dir, f'{subject}_{session}')
        # One path for every iteration count: pass 1 refines against the existing
        # chain and registers nothing, so iterations=1 is the plain single pass.
        refined, history = refine_iterative(
            raw, seed, register, voxel_mm3,
            reference_parcellation=parcellation, reference_guide=seed,
            iterations=s['iterations'], target_coverage=s['target_coverage'],
            tolerance_mm3=s['tolerance_mm3'], qc_thresholds=s['qc'],
            tissue_factor=s['tissue_factor'])
    finally:
        if tmp is not None:
            tmp.cleanup()

    qc = compute_brain_mask_qc(refined, parcellation, voxel_mm3, raw=raw,
                               initial_mask=initial, thresholds=s['qc'],
                               tissue_factor=s['tissue_factor'])
    before = compute_brain_mask_qc(initial, parcellation, voxel_mm3, raw=raw,
                                   thresholds=s['qc'], tissue_factor=s['tissue_factor'])

    mask_hdr = nib.load(str(mask_p))
    if s['apply']:
        if not initial_p.exists():
            shutil.copy(mask_p, initial_p)
        out_p = mask_p
        _restrip(n4_file, refined, anat, subject, session, config)
    else:
        out_p = anat / f'{subject}_{session}_desc-refinedbrain_mask.nii.gz'
    nib.save(nib.Nifti1Image(refined.astype(np.uint8), mask_hdr.affine, mask_hdr.header),
             str(out_p))

    montage_p = ''
    if montage:
        from neurofaune.preprocess.qc.skull_strip_qc import plot_mask_comparison_mosaic
        figures = (Path(qc_dir) if qc_dir else study_root / 'qc' / 'mask_refinement') / 'figures'
        figures.mkdir(parents=True, exist_ok=True)
        montage_p = str(plot_mask_comparison_mosaic(
            raw, {'initial': initial, 'refined': refined}, subject, session, figures))

    flag = 'ok' if qc['passed'] else 'FAILED QC: ' + '; '.join(qc['failures'])
    logger.info('%s %s: refined mask %.0f -> %.0f mm3, coverage %.1f%% -> %.1f%% (%s)',
                subject, session, before['volume_mm3'], qc['volume_mm3'],
                100 * before['atlas_coverage'], 100 * qc['atlas_coverage'], flag)
    return {
        **base, 'status': 'refined', 'method': s['method'], 'applied': s['apply'],
        'volume_before_mm3': before['volume_mm3'], 'volume_after_mm3': qc['volume_mm3'],
        'atlas_coverage_before': before['atlas_coverage'],
        'atlas_coverage_after': qc['atlas_coverage'],
        'non_brain_before_mm3': before['non_brain_mm3'],
        'non_brain_after_mm3': qc['non_brain_mm3'],
        'tissue_fraction_before': before['tissue_fraction'],
        'tissue_fraction_after': qc['tissue_fraction'],
        'dice_with_initial': qc['dice_with_initial'],
        'n_iterations': len(history),
        # which pass was kept, and whether later ones were discarded for
        # contracting -- see refine_iterative's docstring
        'selected_iteration': next((h['iteration'] for h in history if h['selected']), 1),
        'reached_target_coverage': history[-1]['atlas_coverage'] >= s['target_coverage'],
        'coverage_by_iteration': ';'.join(f"{h['atlas_coverage']:.4f}" for h in history),
        'n_registrations': sum(h['registered'] for h in history),
        'qc_passed': qc['passed'], 'qc_failures': '; '.join(qc['failures']),
        'raw_t2w': str(raw_p), 'parcellation': str(parcellation_file),
        'mask': str(out_p), 'montage': montage_p,
    }


def _restrip(n4_file: Path, mask: np.ndarray, anat: Path, subject: str, session: str,
             config: dict[str, Any]) -> None:
    """Rewrite desc-skullstrip / desc-preproc T2w from the N4 image and `mask`.

    Mirrors run_anatomical_preprocessing: skullstrip = N4 * mask, preproc =
    skullstrip * `anatomical.intensity_normalization.factor`. Multiplying the old
    stripped image would not do: the tissue the refinement recovers was zeroed in it.
    """
    n4 = nib.load(str(n4_file))
    brain = np.asarray(n4.dataobj, dtype=np.float32) * mask
    factor = get_config_value(config, 'anatomical.intensity_normalization.factor',
                              default=1000.0)
    nib.save(nib.Nifti1Image(brain, n4.affine, n4.header),
             str(anat / f'{subject}_{session}_desc-skullstrip_T2w.nii.gz'))
    nib.save(nib.Nifti1Image(brain * factor, n4.affine, n4.header),
             str(anat / f'{subject}_{session}_desc-preproc_T2w.nii.gz'))
    logger.warning('%s %s: refined mask applied; tissue probsegs and the T2w->template '
                   'registration still come from the first-pass mask', subject, session)


def write_refinement_summary(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path | None:
    """Write `mask_refinement.csv` and `mask_refinement_summary.json` for refined rows."""
    import pandas as pd

    refined = [r for r in rows if r.get('status') == 'refined']
    if not refined:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(refined)
    table = output_dir / 'mask_refinement.csv'
    df.to_csv(table, index=False)

    def med(col):
        return float(df[col].median())

    summary = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'n_sessions': len(df),
        'n_qc_failed': int((~df.qc_passed.astype(bool)).sum()),
        'qc_failed': [f'{r.subject}/{r.session}' for r in df.itertuples() if not r.qc_passed],
        'n_skipped': sum(r.get('status') == 'skipped' for r in rows),
        'applied_in_place': bool(df.applied.any()),
        'volume_mm3': {'before_median': med('volume_before_mm3'),
                       'after_median': med('volume_after_mm3'),
                       'before_sd': float(df.volume_before_mm3.std()),
                       'after_sd': float(df.volume_after_mm3.std())},
        'atlas_coverage': {'before_median': med('atlas_coverage_before'),
                           'after_median': med('atlas_coverage_after')},
        'non_brain_mm3': {'before_median': med('non_brain_before_mm3'),
                          'after_median': med('non_brain_after_mm3')},
        'tissue_fraction': {'before_median': med('tissue_fraction_before'),
                            'after_median': med('tissue_fraction_after')},
        'dice_with_initial': {'median': med('dice_with_initial'),
                              'min': float(df.dice_with_initial.min())},
    }
    with open(output_dir / 'mask_refinement_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    return table
