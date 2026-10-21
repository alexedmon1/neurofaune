"""Subject T2w <-> atlas template, for species registered straight to the atlas (no cohort template).

The rat pipeline builds a study (cohort) template and registers each T2w to it in
``scripts/batch_preprocess_anat.py``; ``func``/``dwi``/``msme`` then reach template
space by composing their own moving->T2w registration with that
``T2w_to_template_*`` transform. A mouse study registers to the atlas template
directly (``atlas.study_space.template_masked``), so here the "template" those
workflows are handed is the atlas template and ``T2w_to_template_*`` is T2w ->
atlas. Writing it under the same names keeps every consumer unchanged.

:func:`atlas_tissue_to_anat` brings the atlas WM/CSF masks
(``atlas.study_space.wm_prob`` / ``csf_prob``) into T2w space as
``label-{WM,CSF}_desc-atlas_probseg``, for aCompCor and the ICA CSF feature when
``functional.denoising.acompcor.tissue_source`` is ``atlas``.
"""

from pathlib import Path
from typing import Any, Dict

from neurofaune.config import get_config_value
from neurofaune.templates.sigma_warp import inverse_transform_args

TEMPLATE_NAME = 'T2w_to_template'
ATLAS_TISSUES = {'WM': 'wm_prob', 'CSF': 'csf_prob'}


def _anat_dir(study_root: Path, subject: str, session: str) -> Path:
    return Path(study_root) / 'derivatives' / subject / session / 'anat'


def _transforms_dir(study_root: Path, subject: str, session: str) -> Path:
    return Path(study_root) / 'transforms' / subject / session


def atlas_template(config: Dict[str, Any]) -> Path:
    """The atlas template a T2w is registered to: brain-only, matching the skull-stripped preproc T2w."""
    return Path(get_config_value(config, 'atlas.study_space.template_masked'))


def register_anat_to_atlas_template(
    config: Dict[str, Any],
    subject: str,
    session: str,
    study_root: Path,
    n_cores: int = 4,
    force: bool = False,
) -> Dict[str, Path]:
    """SyN-register the preproc (skull-stripped) T2w to the atlas template.

    Writes ``transforms/<sub>/<ses>/<sub>_<ses>_T2w_to_template_{0GenericAffine.mat,
    1Warp, 1InverseWarp, Warped}``. Skips the registration when the inverse warp
    already exists, unless ``force``.
    """
    from neurofaune.templates.anat_registration import register_anat_to_template

    out = _transforms_dir(study_root, subject, session)
    prefix = out / f'{subject}_{session}_{TEMPLATE_NAME}_'
    paths = {
        'affine_transform': Path(f'{prefix}0GenericAffine.mat'),
        'warp_transform': Path(f'{prefix}1Warp.nii.gz'),
        'inverse_warp': Path(f'{prefix}1InverseWarp.nii.gz'),
        'warped_t2w': Path(f'{prefix}Warped.nii.gz'),
    }
    if paths['inverse_warp'].exists() and paths['affine_transform'].exists() and not force:
        return paths
    anat = _anat_dir(study_root, subject, session)
    t2w = anat / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    mask = anat / f'{subject}_{session}_desc-brain_mask.nii.gz'
    return register_anat_to_template(
        t2w_file=t2w, template_file=atlas_template(config), output_dir=out,
        subject=subject, session=session, mask_file=mask,
        template_mask=Path(get_config_value(config, 'atlas.study_space.brain_mask')),
        n_cores=n_cores, generate_qc=True,
    )


def atlas_tissue_path(study_root: Path, subject: str, session: str, label: str) -> Path:
    return _anat_dir(study_root, subject, session) / f'{subject}_{session}_label-{label}_desc-atlas_probseg.nii.gz'


def atlas_tissue_to_anat(
    config: Dict[str, Any],
    subject: str,
    session: str,
    study_root: Path,
    force: bool = False,
) -> Dict[str, Path]:
    """Warp the atlas WM/CSF masks into T2w space (Linear, so the result is a partial-volume fraction).

    Needs :func:`register_anat_to_atlas_template` first. Returns ``{'WM': path, 'CSF': path}``.
    """
    import subprocess

    tdir = _transforms_dir(study_root, subject, session)
    affine = tdir / f'{subject}_{session}_{TEMPLATE_NAME}_0GenericAffine.mat'
    inv_warp = tdir / f'{subject}_{session}_{TEMPLATE_NAME}_1InverseWarp.nii.gz'
    if not affine.exists():
        raise FileNotFoundError(
            f'{affine} not found: run register_anat_to_atlas_template() first')
    anat = _anat_dir(study_root, subject, session)
    ref = anat / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    inv = [a for t in inverse_transform_args(affine, inv_warp if inv_warp.exists() else None)
           for a in ('-t', t)]
    out: Dict[str, Path] = {}
    for label, key in ATLAS_TISSUES.items():
        src = get_config_value(config, f'atlas.study_space.{key}', default=None)
        if not src:
            raise ValueError(f'atlas.study_space.{key} is not set; the atlas tissue source needs it')
        dest = atlas_tissue_path(study_root, subject, session, label)
        if force or not dest.exists():
            subprocess.run(['antsApplyTransforms', '-d', '3', '-i', str(src), '-r', str(ref), *inv,
                            '-o', str(dest), '--interpolation', 'Linear'],
                           check=True, capture_output=True, text=True)
        out[label] = dest
    return out
