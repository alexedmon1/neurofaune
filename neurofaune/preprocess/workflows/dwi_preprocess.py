"""
DTI/DWI preprocessing workflow.

This module provides a complete preprocessing pipeline for diffusion MRI data,
including eddy correction, DTI fitting, FA→Template registration, and
SIGMA atlas warping.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import subprocess
import json

from neurofaune.config import get_config_value
from neurofaune.preprocess.utils.dwi_utils import (
    convert_5d_to_4d,
    validate_gradient_table,
    extract_b0_volume,
    check_dwi_data_quality,
    pad_slices_for_eddy,
    pad_mask_for_eddy,
    crop_slices_after_eddy,
    normalize_dwi_intensity,
    normalize_for_brain_extraction
)
from neurofaune.preprocess.utils.skull_strip import skull_strip
from neurofaune.preprocess.utils.registration_utils import (
    propagate_anat_mask,
    restrict_mask_to,
)
from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.preprocess.utils.orientation import (
    match_orientation_to_reference,
    save_orientation_metadata,
    print_orientation_info
)
from neurofaune.atlas.manager import AtlasManager
from neurofaune.utils.transforms import TransformRegistry
from neurofaune.preprocess.qc.dwi import generate_eddy_qc_report, generate_dti_qc_report
from neurofaune.preprocess.qc import get_subject_qc_dir
from neurofaune.provenance import write_provenance, write_dataset_description


def _sigma_warp_variants(
    registration_results: Dict[str, Any],
    canonical: str = 'affine',
) -> List[Dict[str, Any]]:
    """Decide which transform set(s) to warp with, and how to name each.

    ``register_fa_to_template(transform_type='both')`` yields two *independent*
    registrations. Each has its own affine, and a warp is only valid alongside
    the affine from the same run -- crossing them produces a chain that still
    runs and still looks plausible. This function keeps those pairings intact.

    The canonical variant is written without a ``desc-`` entity, so it is the
    one ``roi_extraction.discover_sigma_metrics`` finds. Flipping ``canonical``
    to ``'syn'`` promotes the nonlinear outputs to the analysed set without any
    change to analysis code.

    Returns
    -------
    list of dict
        ``{'affine': Path, 'warp': Path|None, 'desc': str|None}``, canonical
        first.
    """
    if canonical not in ('affine', 'syn'):
        raise ValueError(
            f"diffusion.registration.canonical must be 'affine' or 'syn', "
            f"got {canonical!r}"
        )

    affine_set = (
        {'affine': registration_results.get('affine_transform'),
         'warp': None, 'name': 'affine'}
        if registration_results.get('affine_transform') else None
    )
    syn_set = (
        {'affine': registration_results.get('syn_affine_transform'),
         'warp': registration_results.get('warp_transform'), 'name': 'syn'}
        if registration_results.get('warp_transform') else None
    )

    available = [v for v in (affine_set, syn_set) if v]
    if not available:
        raise RuntimeError("no usable DWI->template transform in registration results")

    # Fall back gracefully: asking for a canonical set that was not produced
    # should not drop the outputs entirely.
    if not any(v['name'] == canonical for v in available):
        canonical = available[0]['name']

    ordered = sorted(available, key=lambda v: v['name'] != canonical)
    return [
        {'affine': v['affine'], 'warp': v['warp'],
         'desc': None if v['name'] == canonical else v['name']}
        for v in ordered
    ]


def _run_registration(cmd: List[str], label: str) -> None:
    """Run an ANTs registration command, raising with captured output on failure."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {label} registration failed!")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        raise RuntimeError(f"FA to Template {label} registration failed")


def _report_coverage(warped: Path) -> None:
    """Report which template slices the warped partial-coverage volume reaches."""
    if not warped.exists():
        return
    print(f"  Warped: {warped.name}")
    data = nib.load(warped).get_fdata()
    slices = [z for z in range(data.shape[2]) if np.sum(data[:, :, z] > 0.1) > 1000]
    if slices:
        print(f"  Covers template slices {slices[0]}-{slices[-1]} ({len(slices)} slices)")


def _build_syn_registration_cmd(
    fixed: Path,
    moving: Path,
    output_prefix: Path,
    metric: str = 'MI',
    n_cores: int = 4,
) -> List[str]:
    """Build a rigid + affine + SyN ``antsRegistration`` call with a chosen metric.

    ``antsRegistrationSyN.sh`` is not usable here: its SyN stage is fixed to
    cross-correlation and it has no metric flag (confirmed against ANTs 2.6.3),
    so it cannot register FA to a T2w template without chasing the inverted
    white-matter contrast. Mutual information is the default because it is
    valid whether or not the two images share contrast.
    """
    metric = metric.upper()
    if metric == 'MI':
        lin = f'MI[{fixed},{moving},1,32,Regular,0.25]'
        syn = f'MI[{fixed},{moving},1,32]'
    elif metric == 'CC':
        # Only valid within modality (e.g. FA against an FA template).
        lin = f'MI[{fixed},{moving},1,32,Regular,0.25]'
        syn = f'CC[{fixed},{moving},1,4]'
    else:
        raise ValueError(f"metric must be 'MI' or 'CC', got {metric!r}")

    return [
        'antsRegistration',
        '--dimensionality', '3',
        '--float', '1',
        '--output', f'[{output_prefix},{output_prefix}Warped.nii.gz,'
                    f'{output_prefix}InverseWarped.nii.gz]',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '0',
        '--winsorize-image-intensities', '[0.005,0.995]',
        # Center of mass: partial-coverage DWI sits at an arbitrary Z offset
        # relative to a whole-brain template.
        '--initial-moving-transform', f'[{fixed},{moving},1]',
        '--transform', 'Rigid[0.1]',
        '--metric', lin,
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--transform', 'Affine[0.1]',
        '--metric', lin,
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--transform', 'SyN[0.1,3,0]',
        '--metric', syn,
        '--convergence', '[100x70x50x20,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
    ]


def register_fa_to_template(
    fa_file: Path,
    template_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4,
    transform_type: str = 'a',
    moving_file: Optional[Path] = None,
    metric: str = 'MI',
) -> Dict[str, Any]:
    """
    Register FA (or another DWI-space volume) to the cohort template.

    Center-of-mass initialization handles the Z-offset from partial-coverage
    DWI. Registering to the cohort template rather than to SIGMA directly is
    deliberate: subject→template is a small deformation between similar brains,
    while template→SIGMA is one large cross-population deformation solved once
    on an averaged image. See ``neurofaune.templates.sigma_warp``.

    **On going nonlinear.** ``transform_type='a'`` (the default) is affine and
    is what this function has always done. Nonlinear is available but the
    choice of moving image matters more than it looks:

    - Registering **FA to a T2w template is cross-contrast with an inverted
      intensity relationship** — white matter is bright in FA and dark in T2w.
      ``antsRegistrationSyN.sh`` uses cross-correlation for its SyN stage and
      exposes no metric flag, so a nonlinear run through it will chase that
      mismatch, deform white matter, and still exit 0. This function therefore
      drops to an explicit ``antsRegistration`` call with **mutual
      information** (the default here) whenever a deformable stage is
      requested.
    - Better still, register **within modality**: build an FA template with
      :func:`neurofaune.templates.builder.build_dwi_template` and point
      ``template_file`` at it, or pass a b0 as ``moving_file`` against a T2w
      template. Both avoid the contrast problem rather than compensating for it.

    See ``docs/DWI_DISTORTION.md`` §3.

    Parameters
    ----------
    fa_file : Path
        FA map from DTI fitting. Used as the moving image unless
        ``moving_file`` is given; either way the transform is named
        ``FA_to_template_*`` and applies to every DWI-space volume.
    template_file : Path
        Cohort template. A T2w template implies a cross-contrast registration;
        an FA template does not.
    output_dir : Path
        Study root directory (transforms saved to transforms/{subject}/{session}/)
    subject, session : str
        BIDS identifiers.
    work_dir : Path
        Working directory for intermediate files
    n_cores : int
        Number of CPU cores for ANTs
    transform_type : str
        ``'a'`` affine only (default, unchanged behaviour), ``'s'`` rigid +
        affine + SyN. Anything containing ``'s'`` triggers the deformable path.
    moving_file : Path, optional
        Use this instead of ``fa_file`` as the moving image. Pass a mean b0 to
        register against a T2w template in matched contrast.
    metric : str
        Similarity metric for the deformable path: ``'MI'`` (default, safe
        across contrasts) or ``'CC'`` (sharper, but only valid when moving and
        fixed share contrast, e.g. FA against an FA template). Ignored when
        ``transform_type='a'``.

    Parameters
    ----------
    fa_file : Path
        FA map from DTI fitting
    template_file : Path
        Cohort template (e.g., tpl-BPARat_p60_T2w.nii.gz)
    output_dir : Path
        Study root directory (transforms saved to transforms/{subject}/{session}/)
    subject : str
        Subject ID
    session : str
        Session ID
    work_dir : Path
        Working directory for intermediate files
    n_cores : int
        Number of CPU cores for ANTs

    Returns
    -------
    dict
        Dictionary with transform paths and metadata. ``warp_transform`` is
        present only when a deformable stage ran.
    """
    print("\n" + "="*60)
    print("FA to Template Registration")
    print("="*60)

    moving = Path(moving_file) if moving_file is not None else fa_file

    # Load images to get info
    fa_img = nib.load(moving)
    template_img = nib.load(template_file)
    print(f"\n  Moving: {fa_img.shape} voxels, {fa_img.header.get_zooms()[:3]} mm")
    print(f"  Template: {template_img.shape} voxels, {template_img.header.get_zooms()[:3]} mm")
    print(f"  Moving file: {moving.name}")
    print(f"  Fixed file: {template_file.name}")

    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)

    want = transform_type.lower()
    if want not in ('a', 's', 'both'):
        raise ValueError(
            f"transform_type must be 'a', 's' or 'both', got {transform_type!r}"
        )
    run_affine = want in ('a', 'both')
    run_syn = want in ('s', 'both')

    results: Dict[str, Any] = {
        'template_file': template_file,
        'transform_type': transform_type,
        'metric': metric if run_syn else None,
        'moving_file': moving,
        'fa_shape': fa_img.shape,
        'template_shape': template_img.shape,
    }

    if run_affine:
        # Canonical prefix, unchanged since this function was affine-only.
        affine_prefix = transforms_dir / 'FA_to_template_'
        print("\nRunning ANTs Affine registration (moving -> Template)...")
        _run_registration(
            [
                'antsRegistrationSyN.sh',
                '-d', '3',
                '-f', str(template_file),
                '-m', str(moving),
                '-o', str(affine_prefix),
                '-n', str(n_cores),
                '-t', 'a',
            ],
            label='affine',
        )
        affine_transform = Path(str(affine_prefix) + '0GenericAffine.mat')
        if not affine_transform.exists():
            raise RuntimeError(f"Expected transform not found: {affine_transform}")
        warped_affine = Path(str(affine_prefix) + 'Warped.nii.gz')
        print(f"  Affine transform: {affine_transform.name}")
        _report_coverage(warped_affine)
        results.update({
            'affine_transform': affine_transform,
            'warped_fa': warped_affine if warped_affine.exists() else None,
        })

    if run_syn:
        # A separate prefix, so running both does not overwrite. The SyN run
        # emits its OWN affine, and that affine is a matched pair with its warp
        # -- pairing the standalone affine above with this warp would be wrong,
        # so they are returned under distinct keys.
        syn_prefix = transforms_dir / 'FA_to_template_desc-SyN_'
        print(f"\nRunning ANTs SyN registration (moving -> Template, {metric})...")
        _run_registration(
            _build_syn_registration_cmd(
                fixed=template_file, moving=moving,
                output_prefix=syn_prefix, metric=metric, n_cores=n_cores,
            ),
            label='SyN',
        )
        syn_affine = Path(str(syn_prefix) + '0GenericAffine.mat')
        syn_warp = Path(str(syn_prefix) + '1Warp.nii.gz')
        syn_inverse = Path(str(syn_prefix) + '1InverseWarp.nii.gz')
        warped_syn = Path(str(syn_prefix) + 'Warped.nii.gz')
        if not syn_warp.exists():
            raise RuntimeError(
                f"deformable registration requested (transform_type={transform_type!r}) "
                f"but no warp was produced at {syn_warp}"
            )
        print(f"  SyN affine: {syn_affine.name}")
        print(f"  SyN warp:   {syn_warp.name}")
        _report_coverage(warped_syn)
        results.update({
            'syn_affine_transform': syn_affine,
            'warp_transform': syn_warp,
            'inverse_warp': syn_inverse,
            'warped_fa_syn': warped_syn if warped_syn.exists() else None,
        })

    results.setdefault('affine_transform', None)
    results.setdefault('warp_transform', None)
    results.setdefault('inverse_warp', None)
    results.setdefault('syn_affine_transform', None)
    results.setdefault('warped_fa', None)
    return results


def warp_dti_to_sigma(
    metric_files: Dict[str, Path],
    fa_to_template_affine: Path,
    tpl_to_sigma_affine: Path,
    tpl_to_sigma_warp: Optional[Path],
    sigma_template: Path,
    output_dir: Path,
    subject: str,
    session: str,
    fa_to_template_warp: Optional[Path] = None,
    desc: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Warp DTI metric maps to SIGMA atlas space.

    Applies the FA→Template + Template→SIGMA transform chain to each DTI
    metric (FA, MD, AD, RD) to produce SIGMA-space outputs for group analysis.

    Parameters
    ----------
    metric_files : dict
        Mapping of metric name to path, e.g. {'FA': path, 'MD': path, ...}
    fa_to_template_affine : Path
        FA_to_template_0GenericAffine.mat from Step 7
    tpl_to_sigma_affine : Path
        tpl-to-SIGMA_0GenericAffine.mat (pre-computed)
    tpl_to_sigma_warp : Path or None
        tpl-to-SIGMA_1Warp.nii.gz (may not exist for affine-only registrations)
    sigma_template : Path
        SIGMA reference image for output geometry
    output_dir : Path
        Directory for SIGMA-space outputs (derivatives dwi dir)
    subject : str
        Subject ID
    session : str
        Session ID

    fa_to_template_warp : Path, optional
        Subject warp from a deformable DWI->template registration. Must be the
        matched pair of ``fa_to_template_affine`` -- pairing a warp with the
        affine from a *different* registration run silently produces a wrong
        chain, so pass ``syn_affine_transform`` and ``warp_transform`` together.
    desc : str, optional
        BIDS ``desc-`` entity for the outputs, e.g. ``'syn'`` giving
        ``..._space-SIGMA_desc-syn_FA.nii.gz``. ``None`` (default) writes the
        canonical ``..._space-SIGMA_FA.nii.gz``, which is what
        ``neurofaune.network.roi_extraction.discover_sigma_metrics`` globs for.
        Use a desc for a variant you want to compare against but not analyse.

    Returns
    -------
    dict
        Mapping of metric name to SIGMA-space output path
    """
    print("\n" + "="*60)
    print(f"Warp DTI Metrics to SIGMA Space{f' (desc-{desc})' if desc else ''}")
    print("="*60)

    # Build transform chain (ANTs applies in reverse order: last listed = first applied)
    transforms: List[str] = []
    if tpl_to_sigma_warp is not None and tpl_to_sigma_warp.exists():
        transforms.append(str(tpl_to_sigma_warp))
    transforms.append(str(tpl_to_sigma_affine))
    if fa_to_template_warp is not None and Path(fa_to_template_warp).exists():
        transforms.append(str(fa_to_template_warp))
    transforms.append(str(fa_to_template_affine))

    print(f"\n  Transform chain ({len(transforms)} transforms):")
    for t in transforms:
        print(f"    {Path(t).name}")
    print(f"  Reference: {sigma_template.name}")

    sigma_outputs = {}

    for metric, input_path in metric_files.items():
        suffix = f'desc-{desc}_{metric}' if desc else metric
        output_path = output_dir / f'{subject}_{session}_space-SIGMA_{suffix}.nii.gz'

        if output_path.exists():
            print(f"\n  {metric}: already exists, skipping")
            sigma_outputs[metric] = output_path
            continue

        if not input_path.exists():
            print(f"\n  {metric}: input not found ({input_path.name}), skipping")
            continue

        print(f"\n  {metric}: warping to SIGMA space...")

        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(input_path),
            '-r', str(sigma_template),
            '-o', str(output_path),
            '-n', 'Linear',
        ]
        for t in transforms:
            cmd.extend(['-t', t])

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"    ERROR: antsApplyTransforms failed for {metric}")
            print(f"    {result.stderr[:500]}")
            continue

        sigma_outputs[metric] = output_path
        print(f"    {output_path.name}")

    print(f"\n  Warped {len(sigma_outputs)}/{len(metric_files)} metrics to SIGMA space")
    return sigma_outputs


def register_fa_to_t2w(
    fa_file: Path,
    t2w_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    work_dir: Path,
    n_cores: int = 4
) -> Dict[str, Any]:
    """
    Register FA to T2w within the same subject.

    .. deprecated::
        Use :func:`register_fa_to_template` instead for better atlas overlap.

    Registers FA directly to the full T2w volume, letting ANTs find the
    optimal 3D alignment including the Z-offset for partial coverage DWI.

    Parameters
    ----------
    fa_file : Path
        FA map from DTI fitting
    t2w_file : Path
        Preprocessed T2w from anatomical pipeline
    output_dir : Path
        Study root directory (transforms saved to transforms/{subject}/{session}/)
    subject : str
        Subject ID
    session : str
        Session ID
    work_dir : Path
        Working directory for intermediate files
    n_cores : int
        Number of CPU cores for ANTs

    Returns
    -------
    dict
        Dictionary with transform paths and metadata
    """
    import warnings
    warnings.warn(
        "register_fa_to_t2w() is deprecated. Use register_fa_to_template() "
        "for better SIGMA atlas overlap.",
        DeprecationWarning,
        stacklevel=2
    )

    print("\n" + "="*60)
    print("FA to T2w Registration")
    print("="*60)

    # Load images to get info
    fa_img = nib.load(fa_file)
    t2w_img = nib.load(t2w_file)
    print(f"\n  FA: {fa_img.shape} voxels, {fa_img.header.get_zooms()[:3]} mm")
    print(f"  T2w: {t2w_img.shape} voxels, {t2w_img.header.get_zooms()[:3]} mm")

    # Register FA directly to full T2w - let ANTs find optimal alignment
    print("\nRunning ANTs Affine registration (FA → full T2w)...")
    transforms_dir = output_dir / 'transforms' / subject / session
    transforms_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = transforms_dir / 'FA_to_T2w_'

    # Use antsRegistrationSyN.sh with affine only
    cmd = [
        'antsRegistrationSyN.sh',
        '-d', '3',
        '-f', str(t2w_file),
        '-m', str(fa_file),
        '-o', str(output_prefix),
        '-n', str(n_cores),
        '-t', 'a'  # Affine only
    ]

    print(f"  Moving: {fa_file.name}")
    print(f"  Fixed: {t2w_file.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:
        print(f"  ERROR: Registration failed!")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        raise RuntimeError("FA to T2w registration failed")

    # Check outputs
    affine_transform = Path(str(output_prefix) + '0GenericAffine.mat')
    warped_fa = Path(str(output_prefix) + 'Warped.nii.gz')

    if not affine_transform.exists():
        raise RuntimeError(f"Expected transform not found: {affine_transform}")

    print(f"  ✓ Affine transform: {affine_transform.name}")
    if warped_fa.exists():
        print(f"  ✓ Warped FA: {warped_fa.name}")

        # Report which T2w slices have FA coverage
        warped_data = nib.load(warped_fa).get_fdata()
        slices_with_fa = [z for z in range(warped_data.shape[2])
                         if np.sum(warped_data[:, :, z] > 0.1) > 1000]
        if slices_with_fa:
            print(f"  FA covers T2w slices {slices_with_fa[0]}-{slices_with_fa[-1]} ({len(slices_with_fa)} slices)")

    return {
        'affine_transform': affine_transform,
        'warped_fa': warped_fa if warped_fa.exists() else None,
        't2w_file': t2w_file,
        'fa_shape': fa_img.shape,
        't2w_shape': t2w_img.shape,
    }


def run_dwi_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_dir: Path,
    transform_registry: TransformRegistry,
    work_dir: Optional[Path] = None,
    use_gpu: bool = True,
    template_file: Optional[Path] = None,
    t2w_file: Optional[Path] = None,
    run_registration: bool = True
) -> Dict[str, Any]:
    """
    Run complete DTI/DWI preprocessing workflow.

    This workflow performs:
    1. Image validation and 5D→4D conversion (if needed)
    2. Gradient table validation
    3. GPU-accelerated eddy correction (motion + distortion)
    4. Brain masking from b0 volume
    5. DTI fitting (FA, MD, AD, RD)
    6. Save preprocessed outputs
    7. (Optional) Register FA to template for atlas propagation
    8. (Optional) Warp DTI metrics to SIGMA space via template

    Parameters
    ----------
    config : dict
        Configuration dictionary from load_config()
    subject : str
        Subject identifier (e.g., 'sub-Rat207')
    session : str
        Session identifier (e.g., 'ses-p60')
    dwi_file : Path
        Input DWI/DTI NIfTI file (may be 4D or 5D)
    bval_file : Path
        FSL-format bval file
    bvec_file : Path
        FSL-format bvec file (3xN)
    output_dir : Path
        Study root directory (will create derivatives/{subject}/{session}/dwi/)
    transform_registry : TransformRegistry
        Transform registry for saving spatial transforms
    work_dir : Path, optional
        Working directory (defaults to output_dir/work/{subject}/{session}/dwi_preproc)
    use_gpu : bool
        Use GPU-accelerated eddy_cuda (default: True)
    template_file : Path, optional
        Cohort template for direct FA→Template registration (preferred)
    t2w_file : Path, optional
        Preprocessed T2w (deprecated, use template_file instead)
    run_registration : bool
        Whether to run FA→Template registration (default: True)

    Returns
    -------
    dict
        Dictionary with output file paths and processing info:
        - 'dwi_preproc': Path to preprocessed DWI
        - 'dwi_mask': Path to brain mask
        - 'bval': Path to output bval file
        - 'bvec': Path to eddy-corrected bvec file
        - 'fa': Path to FA map
        - 'md': Path to MD map
        - 'ad': Path to AD map
        - 'rd': Path to RD map
        - 'qc_metrics': Dict with QC metrics
        - 'registration': Dict with registration outputs (if run_registration=True)

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> from neurofaune.utils.transforms import create_transform_registry
    >>> from pathlib import Path
    >>>
    >>> config = load_config(Path('config.yaml'))
    >>> registry = create_transform_registry(config, 'sub-Rat207', cohort='p60')
    >>>
    >>> results = run_dwi_preprocessing(
    ...     config=config,
    ...     subject='sub-Rat207',
    ...     session='ses-p60',
    ...     dwi_file=Path('dwi.nii.gz'),
    ...     bval_file=Path('dwi.bval'),
    ...     bvec_file=Path('dwi.bvec'),
    ...     output_dir=Path('/study'),
    ...     transform_registry=registry,
    ...     use_gpu=True
    ... )
    """
    print("="*80)
    print(f"DTI/DWI Preprocessing Workflow")
    print(f"Subject: {subject}, Session: {session}")
    print("="*80)

    # Setup directories
    derivatives_dir = output_dir / 'derivatives' / subject / session / 'dwi'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = get_subject_qc_dir(output_dir, subject, session, 'dwi')

    if work_dir is None:
        work_dir = output_dir / 'work' / subject / session / 'dwi_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDirectories:")
    print(f"  Derivatives: {derivatives_dir}")
    print(f"  QC: {qc_dir}")
    print(f"  Work: {work_dir}")

    # Define output files
    dwi_4d_file = work_dir / f'{subject}_{session}_dwi_4d.nii.gz'
    b0_file = work_dir / f'{subject}_{session}_b0.nii.gz'
    brain_mask_file = derivatives_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'

    # Eddy-corrected outputs
    dwi_eddy_file = derivatives_dir / f'{subject}_{session}_desc-preproc_dwi.nii.gz'
    eddy_rotated_bvecs = derivatives_dir / f'{subject}_{session}_desc-preproc_dwi.bvec'

    # DTI outputs
    fa_file = derivatives_dir / f'{subject}_{session}_FA.nii.gz'
    md_file = derivatives_dir / f'{subject}_{session}_MD.nii.gz'
    ad_file = derivatives_dir / f'{subject}_{session}_AD.nii.gz'
    rd_file = derivatives_dir / f'{subject}_{session}_RD.nii.gz'

    # ==========================================================================
    # Step 1: Image validation and 5D→4D conversion
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 1: Image Validation and 5D→4D Conversion")
    print("="*80)

    # Validate input image
    validation = validate_image(dwi_file, modality='dwi', strict=False)
    print_validation_results(validation, name=f"{subject}_{session} DWI")

    if not validation['valid']:
        raise ValueError(f"DWI validation failed: {validation['errors']}")

    # Check if 5D and convert to 4D
    img = nib.load(dwi_file)
    if len(img.shape) == 5:
        print(f"\nDetected 5D data: {img.shape}")
        print("Converting 5D → 4D by averaging across 5th dimension...")
        convert_5d_to_4d(dwi_file, dwi_4d_file, method='mean')
        dwi_input = dwi_4d_file
    elif len(img.shape) == 4:
        print(f"\nData is already 4D: {img.shape}")
        dwi_input = dwi_file
    else:
        raise ValueError(f"Unexpected DWI shape: {img.shape}")

    # ==========================================================================
    # Step 1.5: MP-PCA denoising + Gibbs-ringing removal (pre-eddy, on raw 4-D)
    # ==========================================================================
    denoise_cfg = config.get('diffusion', {}).get('denoise', {})
    degibbs_cfg = config.get('diffusion', {}).get('degibbs', {})
    do_denoise = denoise_cfg.get('enabled', True)
    do_degibbs = degibbs_cfg.get('enabled', True)
    if do_denoise or do_degibbs:
        from neurofaune.preprocess.utils.dwi_denoise import denoise_dwi_mppca, degibbs_dwi
        print("\n" + "="*80)
        print("Step 1.5: Denoising (MP-PCA) + Gibbs Removal")
        print("="*80)
        if do_denoise:
            den_file = work_dir / f'{subject}_{session}_dwi_4d_denoised.nii.gz'
            print("\n  MP-PCA denoising (dipy mppca)...")
            denoise_dwi_mppca(dwi_input, den_file,
                              patch_radius=denoise_cfg.get('patch_radius', 2))
            dwi_input = den_file
            print(f"  Denoised DWI: {den_file.name}")
        if do_degibbs:
            dg_file = work_dir / f'{subject}_{session}_dwi_4d_degibbs.nii.gz'
            print("\n  Gibbs-ringing removal (dipy gibbs_removal)...")
            degibbs_dwi(dwi_input, dg_file,
                        num_processes=degibbs_cfg.get('num_processes', 1))
            dwi_input = dg_file
            print(f"  De-Gibbs DWI: {dg_file.name}")
    else:
        print("\n  [INFO] DWI denoising + Gibbs removal disabled in config")

    # ==========================================================================
    # Step 2: Gradient table validation
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 2: Gradient Table Validation")
    print("="*80)

    img_4d = nib.load(dwi_input)
    n_volumes = img_4d.shape[3]

    bvals, bvecs = validate_gradient_table(bval_file, bvec_file, n_volumes)

    # Save validated gradient tables to work directory
    bval_validated = work_dir / 'dwi.bval'
    bvec_validated = work_dir / 'dwi.bvec'
    np.savetxt(bval_validated, bvals.reshape(1, -1), fmt='%d')
    np.savetxt(bvec_validated, bvecs, fmt='%.6f')

    # ==========================================================================
    # Step 2.5: Intensity Normalization (for robust brain extraction)
    # ==========================================================================
    # Check if intensity normalization is enabled (default: True)
    norm_config = config.get('diffusion', {}).get('intensity_normalization', {})
    normalize_enabled = norm_config.get('enabled', True)

    if normalize_enabled:
        print("\n" + "="*80)
        print("Step 2.5: Intensity Normalization")
        print("="*80)
        print("\nNormalizing DWI intensity for robust brain extraction...")
        print("  (Different Bruker ParaVision settings can cause vastly different")
        print("   intensity scales - normalization ensures consistent BET performance)")

        target_max = norm_config.get('target_max', 10000.0)
        dwi_normalized_file = work_dir / f'{subject}_{session}_dwi_4d_normalized.nii.gz'

        normalized_file, norm_params = normalize_dwi_intensity(
            dwi_input, dwi_normalized_file, target_max=target_max
        )

        print(f"\n  Original intensity range: [{norm_params['original_min']:.2f}, {norm_params['original_max']:.2f}]")
        print(f"  Percentile range used: [{norm_params['original_p_min']:.2f}, {norm_params['original_p_max']:.2f}]")
        print(f"  Scale factor applied: {norm_params['scale_factor']:.4f}")
        print(f"  Target max intensity: {norm_params['target_max']:.0f}")
        print(f"  Normalized DWI: {normalized_file}")

        # Use normalized data for subsequent steps
        dwi_input = normalized_file
    else:
        print("\n  [INFO] Intensity normalization disabled in config")

    # ==========================================================================
    # Step 3: Extract b0 volume and create brain mask
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 3: b0 Extraction and Brain Masking")
    print("="*80)

    extract_b0_volume(dwi_input, bval_validated, b0_file)

    # Create brain mask using unified skull strip dispatcher
    # DTI has 11 slices (>=10), so will auto-select atropos_bet two-pass method
    print(f"\nCreating brain mask (auto-selects method based on slice count)...")
    b0_brain_file = work_dir / f'{subject}_{session}_b0_brain.nii.gz'
    skull_strip_work_dir = work_dir / 'skull_strip'
    skull_strip_work_dir.mkdir(exist_ok=True)

    # Brain extraction gets its OWN range-compressed copy of the b0. The data
    # path must never be clipped (clipping truncates S0 and wrecks the decay
    # curve), but atropos_bet's "brightest class is brain" heuristic needs the
    # bright tissue saturated onto a common plateau -- on an unclipped b0 the
    # brain splits across intensity classes and the mask collapses (observed:
    # ~46k-voxel brain -> 794-voxel mask, which then starved eddy of the voxels
    # it needs for hyperparameter estimation). Percentiles come from the full 4D
    # exactly as the legacy path did: pooled over all volumes, p_max sits below
    # the b0 brain level, so the b0 brain saturates into that plateau. This copy
    # is a throwaway -- only the MASK is kept, applied to unclipped data.
    b0_strip_file = work_dir / f'{subject}_{session}_b0_for-strip.nii.gz'
    normalize_for_brain_extraction(b0_file, b0_strip_file, reference_file=dwi_input)

    cohort = session.split('-')[1] if '-' in session else 'p60'
    dwi_ss_method = get_config_value(config, 'diffusion.skull_strip.method', default='atropos_bet')
    dwi_ss_n_classes = get_config_value(config, 'diffusion.skull_strip.n_classes', default=3)
    _, _, skull_strip_info = skull_strip(
        input_file=b0_strip_file,
        output_file=b0_brain_file,
        mask_file=brain_mask_file,
        work_dir=skull_strip_work_dir,
        method=dwi_ss_method,
        cohort=cohort,
        n_classes=dwi_ss_n_classes,
    )
    print(f"  Method: {skull_strip_info.get('method', 'unknown')}")
    print(f"  Extraction ratio: {skull_strip_info.get('extraction_ratio', 0):.3f}")

    # skull_strip wrote the brain image from the range-compressed copy; rewrite
    # it from the real b0 so QC (which pairs it with the unclipped b0_file) and
    # anything else reading it see true intensities. Only the mask carries over.
    _b0_img = nib.load(str(b0_file))
    _b0_mask = nib.load(str(brain_mask_file)).get_fdata() > 0
    nib.save(nib.Nifti1Image(
        (_b0_img.get_fdata() * _b0_mask).astype(np.float32),
        _b0_img.affine, _b0_img.header), str(b0_brain_file))

    # ==========================================================================
    # Step 4: GPU-accelerated eddy correction (with slice padding)
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 4: Eddy Correction (Motion + Distortion)")
    print("="*80)

    # Get slice padding config (default: 2 slices on each side)
    n_pad_slices = config.get('diffusion', {}).get('eddy', {}).get('slice_padding', 2)

    # Pad DWI and mask to prevent edge slice loss during eddy
    # This is critical for thin-slice acquisitions where motion correction
    # can cause edge slices to be interpolated from outside the volume
    print(f"\nPadding slices for eddy protection (n_pad={n_pad_slices})...")

    dwi_padded_file = work_dir / f'{subject}_{session}_dwi_4d_padded.nii.gz'
    mask_padded_file = work_dir / f'{subject}_{session}_mask_padded.nii.gz'

    dwi_padded_file, original_n_slices = pad_slices_for_eddy(
        dwi_input, dwi_padded_file, n_pad=n_pad_slices, method='reflect'
    )
    pad_mask_for_eddy(brain_mask_file, mask_padded_file, n_pad=n_pad_slices)

    # Check for eddy_cuda availability
    eddy_cmd = 'eddy_cuda' if use_gpu else 'eddy'

    try:
        # eddy_cuda doesn't support --version, so just check the binary exists
        import shutil
        if shutil.which(eddy_cmd) is not None:
            print(f"Using {eddy_cmd} for eddy correction")
        else:
            raise FileNotFoundError(f"{eddy_cmd} not found in PATH")
    except (FileNotFoundError,):
        print(f"Warning: {eddy_cmd} not available, falling back to eddy")
        eddy_cmd = 'eddy'

    # Create index file (all volumes use same phase encoding)
    index_file = work_dir / 'index.txt'
    with open(index_file, 'w') as f:
        f.write(' '.join(['1'] * n_volumes))

    # Create acquisition parameters file from config
    pe_direction = get_config_value(config, 'diffusion.eddy.phase_encoding_direction', default='0 -1 0')
    readout_time = get_config_value(config, 'diffusion.eddy.readout_time',
                                    default=get_config_value(config, 'diffusion.topup.readout_time', default=0.05))
    acqparams_file = work_dir / 'acqparams.txt'
    with open(acqparams_file, 'w') as f:
        f.write(f'{pe_direction} {readout_time}\n')

    # Run eddy on PADDED data
    eddy_basename = work_dir / 'eddy_corrected'
    eddy_cmd_full = [
        eddy_cmd,
        f'--imain={dwi_padded_file}',
        f'--mask={mask_padded_file}',
        f'--acqp={acqparams_file}',
        f'--index={index_file}',
        f'--bvecs={bvec_validated}',
        f'--bvals={bval_validated}',
        f'--out={eddy_basename}',
        '--verbose'
    ]

    # Add optional eddy flags from config
    if get_config_value(config, 'diffusion.eddy.repol', default=True):
        eddy_cmd_full.append('--repol')
    if get_config_value(config, 'diffusion.eddy.data_is_shelled', default=True):
        eddy_cmd_full.append('--data_is_shelled')

    if use_gpu and eddy_cmd == 'eddy_cuda':
        eddy_cmd_full.append('--very_verbose')

    print(f"\nRunning eddy correction on padded data...")
    print(f"Command: {' '.join(eddy_cmd_full)}")

    result = subprocess.run(eddy_cmd_full,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True)

    if result.returncode != 0:
        print(f"Eddy correction failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Eddy correction failed")

    print("Eddy correction completed successfully")

    # Crop eddy output back to original slice count
    eddy_output_padded = work_dir / 'eddy_corrected.nii.gz'
    eddy_output_cropped = work_dir / 'eddy_corrected_cropped.nii.gz'

    crop_slices_after_eddy(
        eddy_output_padded, eddy_output_cropped,
        original_n_slices=original_n_slices, n_pad=n_pad_slices
    )

    # Copy cropped eddy output to derivatives
    eddy_bvecs_rotated = work_dir / 'eddy_corrected.eddy_rotated_bvecs'

    import shutil
    shutil.copy(eddy_output_cropped, dwi_eddy_file)

    # Fix eddy rotated bvecs - replace NaN with 0 (occurs for b0 volumes)
    bvecs_rotated = np.loadtxt(eddy_bvecs_rotated)
    if np.any(np.isnan(bvecs_rotated)):
        print("  Fixing NaN values in rotated bvecs (b0 volumes)...")
        bvecs_rotated = np.nan_to_num(bvecs_rotated, nan=0.0)
    np.savetxt(eddy_rotated_bvecs, bvecs_rotated, fmt='%.10g')

    # Copy bvals (unchanged by eddy)
    bval_output = derivatives_dir / f'{subject}_{session}_desc-preproc_dwi.bval'
    shutil.copy(bval_validated, bval_output)

    print(f"\nEddy-corrected DWI saved to: {dwi_eddy_file}")
    print(f"Rotated bvecs saved to: {eddy_rotated_bvecs}")

    # ==========================================================================
    # Step 4.5: Post-eddy brain mask refinement (same-session T2w, SyN)
    # ==========================================================================
    # The atropos_bet mask retains some non-brain tissue (muscle, ear canals).
    # Once eddy has produced a clean, motion-corrected b0, the same-session T2w
    # brain mask is propagated in via SyN (nonlinear = absorbs EPI susceptibility
    # distortion, which a rigid fit can't) and INTERSECTED with the atropos mask
    # to strip that residue.
    #
    # Intersect, not replace: this is a cleanup pass over atropos_bet, not a
    # substitute for it. atropos_bet still produces the mask eddy itself uses
    # (Step 3); this only refines the mask that Step 5 / DKI / NODDI fit within.
    #
    # NO erosion — the FA maps stay full-brain and unbiased, and the brain-edge
    # rim is handled at the analysis stage (TBSS erodes the template WM mask;
    # ROI analyses use interior atlas WM ROIs).
    # Off by default (legacy atropos mask preserved).
    second_mask = get_config_value(config, 'diffusion.second_mask.method', default=None)
    if second_mask == 'anat_mask':
        print("\n" + "="*80)
        print("Step 4.5: Post-eddy brain mask refinement (T2w SyN propagation)")
        print("="*80)
        anat_t2w = (output_dir / 'derivatives' / subject / session / 'anat'
                    / f'{subject}_{session}_desc-preproc_T2w.nii.gz')
        anat_mask = (output_dir / 'derivatives' / subject / session / 'anat'
                     / f'{subject}_{session}_desc-brain_mask.nii.gz')
        if not (anat_t2w.exists() and anat_mask.exists()):
            print(f"  WARNING: preproc T2w / brain mask not found for {subject} "
                  f"{session}; keeping the atropos_bet mask.")
        else:
            mean_b0 = work_dir / f'{subject}_{session}_meanb0_eddy.nii.gz'
            ec = nib.load(str(dwi_eddy_file))
            bvals_ec = np.atleast_1d(np.loadtxt(bval_output))
            b0_idx = bvals_ec < 100
            nib.save(nib.Nifti1Image(
                ec.get_fdata()[..., b0_idx].mean(-1).astype(np.float32),
                ec.affine, ec.header), mean_b0)
            # INTERSECT, never replace. The propagated T2w mask is written to
            # its own file and ANDed with the atropos mask, so this step can
            # only ever REMOVE tissue.
            #
            # Replacing outright was measured on sub-7Y ses-1 and is wrong here:
            # the propagated mask ADDS 22,583 voxels (46.5% of the atropos mask)
            # whose b0 signal is 3.1x dimmer than brain core (4,728 vs 14,458)
            # and which carry 9x the rate of degenerate FA>0.8 (1.8% vs 0.2%) --
            # a dorsal region of EPI signal dropout, not recoverable brain.
            # Dice(atropos, propagated) is only 0.78, so the propagated boundary
            # is not reliable enough to be taken as truth in its own right.
            propagated = work_dir / 'second_mask' / 'anat_mask_in_dwi.nii.gz'
            propagated.parent.mkdir(parents=True, exist_ok=True)
            propagate_anat_mask(
                moving_ref=mean_b0, anat_t2w=anat_t2w, anat_mask=anat_mask,
                out_mask=propagated,
                work_dir=work_dir / 'second_mask', nonlinear=True,
            )
            info = restrict_mask_to(brain_mask_file, propagated, brain_mask_file)
            print(f"  Brain mask refined (T2w SyN, intersect): "
                  f"{info['n_before']} -> {info['n_after']} voxels "
                  f"({info['n_removed']} removed = "
                  f"{info['fraction_removed']*100:.1f}%, 0 added by construction)")

    # ==========================================================================
    # Step 5: DTI fitting
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 5: DTI Fitting (FA, MD, AD, RD)")
    print("="*80)

    # Restrict the tensor fit to the Gaussian regime (b0 + b<=max_bval). The DTI
    # model assumes mono-exponential decay; high-b shells are non-Gaussian and
    # bias the tensor (deflate diffusivity, inflate FA). DKI/NODDI keep all shells.
    dti_max_bval = get_config_value(config, 'diffusion.dti.max_bval', default=None)
    fit_dti(
        dwi_file=dwi_eddy_file,
        mask_file=brain_mask_file,
        bval_file=bval_output,
        bvec_file=eddy_rotated_bvecs,
        output_prefix=derivatives_dir / f'{subject}_{session}',
        max_bval=dti_max_bval,
    )

    print(f"\nDTI maps created:")
    print(f"  FA: {fa_file}")
    print(f"  MD: {md_file}")
    print(f"  AD: {ad_file}")
    print(f"  RD: {rd_file}")

    # SIGMA warping is done in Step 8 (after FA→Template registration in Step 7)

    # ==========================================================================
    # Step 6: Quality control
    # ==========================================================================
    print("\n" + "="*80)
    print("Step 6: Quality Control")
    print("="*80)

    qc_results = {}

    # Eddy QC (motion, signal quality)
    eddy_params_file = work_dir / 'eddy_corrected.eddy_parameters'
    if eddy_params_file.exists():
        eddy_qc = generate_eddy_qc_report(
            subject=subject,
            session=session,
            dwi_preproc=dwi_eddy_file,
            eddy_params=eddy_params_file,
            output_dir=qc_dir,
            original_file=b0_file,
            brain_file=b0_brain_file,
            mask_file=brain_mask_file,
            skull_strip_info=skull_strip_info,
        )
        qc_results['eddy_qc'] = eddy_qc
    else:
        print("Warning: Eddy parameters file not found, skipping motion QC")

    # DTI metrics QC
    dti_qc = generate_dti_qc_report(
        subject=subject,
        session=session,
        fa_file=fa_file,
        md_file=md_file,
        ad_file=ad_file,
        rd_file=rd_file,
        brain_mask=brain_mask_file,
        output_dir=qc_dir
    )
    qc_results['dti_qc'] = dti_qc

    # Basic data quality metrics
    qc_metrics = check_dwi_data_quality(dwi_eddy_file, brain_mask_file)
    qc_json = qc_dir / f'{subject}_{session}_dwi_basic_qc.json'
    with open(qc_json, 'w') as f:
        json.dump(qc_metrics, f, indent=2)

    print(f"\n✓ QC reports generated:")
    if 'eddy_qc' in qc_results:
        print(f"  - Eddy/Motion QC: {qc_results['eddy_qc']['html_report']}")
    print(f"  - DTI Metrics QC: {qc_results['dti_qc']['html_report']}")
    print(f"  - Basic metrics: {qc_json}")

    # ==========================================================================
    # Step 7: FA to Template Registration (optional)
    # ==========================================================================
    registration_results = None

    if run_registration:
        # Prefer template_file; fall back to t2w_file (deprecated)
        reg_target = template_file or t2w_file
        use_template = template_file is not None

        if reg_target is None:
            print("\n  Registration requested but no template/T2w file provided - skipping")
        elif not reg_target.exists():
            print(f"\n  Registration target not found: {reg_target} - skipping registration")
        else:
            if use_template:
                print("\n" + "="*80)
                print("Step 7: FA to Template Registration")
                print("="*80)

                try:
                    # Affine by default, preserving long-standing behaviour.
                    # Set diffusion.registration.transform_type: 's' to go
                    # nonlinear -- see docs/DWI_DISTORTION.md for why the
                    # metric and the choice of moving image matter when the
                    # template is T2w rather than FA.
                    reg_transform_type = get_config_value(
                        config, 'diffusion.registration.transform_type', default='a')
                    reg_metric = get_config_value(
                        config, 'diffusion.registration.metric', default='MI')
                    registration_results = register_fa_to_template(
                        fa_file=fa_file,
                        template_file=template_file,
                        output_dir=output_dir,
                        subject=subject,
                        session=session,
                        work_dir=work_dir,
                        n_cores=4,
                        transform_type=reg_transform_type,
                        metric=reg_metric,
                    )

                    # Save registration metadata to JSON
                    reg_metadata_file = derivatives_dir / f'{subject}_{session}_FA_to_template_registration.json'
                    reg_metadata = {
                        'fa_file': str(fa_file),
                        'template_file': str(registration_results['template_file']),
                        'affine_transform': str(registration_results['affine_transform']),
                        'warped_fa': str(registration_results['warped_fa']) if registration_results.get('warped_fa') else None,
                        'fa_shape': list(registration_results['fa_shape']),
                        'template_shape': list(registration_results['template_shape']),
                    }
                    with open(reg_metadata_file, 'w') as f:
                        json.dump(reg_metadata, f, indent=2)

                    print(f"\n  Registration complete:")
                    print(f"  - Transform: {registration_results['affine_transform']}")
                    print(f"  - Metadata: {reg_metadata_file}")

                except Exception as e:
                    print(f"\n  Registration failed: {e}")
                    print("  Continuing without registration...")
            else:
                # Legacy path: FA → T2w (deprecated)
                print("\n" + "="*80)
                print("Step 7: FA to T2w Registration (deprecated)")
                print("="*80)

                try:
                    registration_results = register_fa_to_t2w(
                        fa_file=fa_file,
                        t2w_file=t2w_file,
                        output_dir=output_dir,
                        subject=subject,
                        session=session,
                        work_dir=work_dir,
                        n_cores=4
                    )

                    reg_metadata_file = derivatives_dir / f'{subject}_{session}_FA_to_T2w_registration.json'
                    reg_metadata = {
                        'fa_file': str(fa_file),
                        't2w_file': str(registration_results['t2w_file']),
                        'affine_transform': str(registration_results['affine_transform']),
                        'warped_fa': str(registration_results['warped_fa']) if registration_results.get('warped_fa') else None,
                        'fa_shape': list(registration_results['fa_shape']),
                        't2w_shape': list(registration_results['t2w_shape']),
                    }
                    with open(reg_metadata_file, 'w') as f:
                        json.dump(reg_metadata, f, indent=2)

                    print(f"\n  Registration complete:")
                    print(f"  - Transform: {registration_results['affine_transform']}")
                    print(f"  - Metadata: {reg_metadata_file}")

                except Exception as e:
                    print(f"\n  Registration failed: {e}")
                    print("  Continuing without registration...")

    # ==========================================================================
    # Step 8: Warp DTI metrics to SIGMA space (optional)
    # ==========================================================================
    sigma_outputs = None

    if registration_results is not None:
        from neurofaune.templates.sigma_warp import (
            DWI_SIGMA_METRICS, build_metric_files, sigma_targets_from_config,
            warp_coverage_mask, warp_maps_to_sigma,
        )

        cohort_name = session.split('-')[1] if '-' in session else None
        targets = sigma_targets_from_config(
            config, session=session, cohort=cohort_name,
            study_root=output_dir, template_file=template_file)

        if not targets["ready"]:
            # Loudly: analysis reads space-SIGMA_* from derivatives, so skipping
            # here leaves the analysis stage with nothing to find.
            print(f"\n  Step 8: SIGMA warp SKIPPED — {targets['reason']}")
            print("  (group analysis reads space-SIGMA_* from derivatives; "
                  "these metrics will be unavailable downstream)")
        else:
            print("\n" + "="*80)
            print("Step 8: Warp DWI Metrics to SIGMA Space")
            print("="*80)
            metric_files = build_metric_files(
                derivatives_dir, f"{subject}_{session}", DWI_SIGMA_METRICS)
            missing = sorted(set(DWI_SIGMA_METRICS) - set(metric_files))
            if missing:
                print(f"  (not yet fitted, skipping: {', '.join(missing)})")
            try:
                warp_coverage_mask(
                    mask_file=(derivatives_dir /
                               f"{subject}_{session}_desc-brain_mask.nii.gz"),
                    moving_to_template=registration_results['affine_transform'],
                    sigma_template=targets["sigma_template"],
                    output_dir=derivatives_dir,
                    subject=subject, session=session,
                    tpl_to_sigma_affine=targets["affine"],
                    tpl_to_sigma_warp=targets["warp"],
                    force=True,
                )
                # With transform_type='both' the registration produced two
                # independent transform sets, so warp twice. Whichever is
                # canonical gets the plain space-SIGMA_{metric} name that
                # roi_extraction globs for; the other carries a desc- entity so
                # it sits alongside for comparison without entering the
                # analysis glob.
                canonical = get_config_value(
                    config, 'diffusion.registration.canonical', default='affine')
                variants = _sigma_warp_variants(registration_results, canonical)
                sigma_outputs = {}
                for v in variants:
                    produced = warp_maps_to_sigma(
                        metric_files=metric_files,
                        moving_to_template=v['affine'],
                        moving_to_template_warp=v['warp'],
                        sigma_template=targets["sigma_template"],
                        output_dir=derivatives_dir,
                        subject=subject, session=session,
                        tpl_to_sigma_affine=targets["affine"],
                        tpl_to_sigma_warp=targets["warp"],
                        desc=v['desc'],
                        force=True,   # metrics were just refitted
                    )
                    if v['desc'] is None:
                        sigma_outputs = produced
            except Exception as e:
                print(f"\n  SIGMA warping failed: {e}")
                print("  Continuing without SIGMA outputs...")

    # ==========================================================================
    # Workflow complete
    # ==========================================================================
    print("\n" + "="*80)
    print("DTI/DWI Preprocessing Complete!")
    print("="*80)
    print(f"\nPreprocessed DWI: {dwi_eddy_file}")
    print(f"Brain mask: {brain_mask_file}")
    print(f"FA map: {fa_file}")
    print(f"MD map: {md_file}")
    print(f"AD map: {ad_file}")
    print(f"RD map: {rd_file}")
    if registration_results is not None:
        print(f"FA→Template transform: {registration_results['affine_transform']}")
    else:
        print("\nNOTE: FA registration was skipped. Run with template_file to enable.")
    if sigma_outputs:
        print(f"SIGMA-space outputs: {len(sigma_outputs)} metrics")
        for metric, path in sigma_outputs.items():
            print(f"  {metric}: {path.name}")
    print("="*80 + "\n")

    results = {
        'dwi_preproc': dwi_eddy_file,
        'dwi_mask': brain_mask_file,
        'bval': bval_output,
        'bvec': eddy_rotated_bvecs,
        'fa': fa_file,
        'md': md_file,
        'ad': ad_file,
        'rd': rd_file,
        'qc_results': qc_results,
        'qc_metrics': qc_metrics
    }

    if registration_results is not None:
        results['registration'] = registration_results

    if sigma_outputs:
        for metric, path in sigma_outputs.items():
            results[f'sigma_{metric.lower()}'] = path

    # Record which neurofaune and which settings produced these outputs. A
    # derivative that cannot name its code cannot be reproduced, and the
    # question always arrives later than the run does.
    write_provenance(derivatives_dir, subject, session, 'dwi', config=config)
    write_dataset_description(output_dir / 'derivatives', config=config)

    return results


def fit_dti(
    dwi_file: Path,
    mask_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_prefix: Path,
    max_bval: Optional[float] = None,
) -> Tuple[Path, Path, Path, Path]:
    """
    Fit DTI model and compute FA, MD, AD, RD maps using FSL's dtifit.

    Parameters
    ----------
    dwi_file : Path
        Preprocessed DWI file
    mask_file : Path
        Brain mask
    bval_file : Path
        b-values
    bvec_file : Path
        b-vectors
    output_prefix : Path
        Output prefix (will create {prefix}_FA.nii.gz, etc.)
    max_bval : float, optional
        Upper b-value (s/mm^2) for the tensor fit. The single-tensor (DTI) model
        assumes mono-exponential (Gaussian) decay, which only holds at low b;
        above ~1000-1500 the signal is non-Gaussian (kurtosis), so including
        high-b shells biases the tensor (deflates diffusivity, inflates FA). When
        set, only b0 + volumes with b <= max_bval are used for the fit. None keeps
        all volumes (legacy behaviour; correct only for single-shell b<=1000 data).
        Higher shells remain available to DKI/NODDI, which model non-Gaussianity.

    Returns
    -------
    tuple
        Paths to (FA, MD, AD, RD) files
    """
    print("\nFitting DTI model with FSL dtifit...")

    # Restrict the tensor fit to the Gaussian regime (b0 + b <= max_bval).
    if max_bval is not None:
        bvals = np.atleast_1d(np.loadtxt(bval_file))
        keep = bvals <= float(max_bval)
        n_keep, n_tot = int(keep.sum()), keep.size
        if n_keep < n_tot:
            import tempfile
            work = Path(tempfile.mkdtemp(prefix='dti_lowb_'))
            img = nib.load(str(dwi_file))
            sub_dwi = work / 'dwi_lowb.nii.gz'
            nib.save(nib.Nifti1Image(img.get_fdata()[..., keep], img.affine, img.header), sub_dwi)
            sub_bval = work / 'lowb.bval'
            np.savetxt(sub_bval, bvals[keep][None, :], fmt='%g')
            bvecs = np.loadtxt(bvec_file)
            if bvecs.shape[0] != 3:
                bvecs = bvecs.T
            sub_bvec = work / 'lowb.bvec'
            np.savetxt(sub_bvec, bvecs[:, keep], fmt='%.6f')
            n_b0 = int((bvals[keep] < 100).sum())
            print(f"  DTI tensor fit restricted to b<={max_bval:g}: "
                  f"{n_keep}/{n_tot} volumes ({n_b0} b0 + {n_keep - n_b0} DW)")
            dwi_file, bval_file, bvec_file = sub_dwi, sub_bval, sub_bvec
        else:
            print(f"  (all {n_tot} volumes have b<={max_bval:g}; no subsetting)")

    # Use FSL's dtifit
    cmd = [
        'dtifit',
        f'--data={dwi_file}',
        f'--mask={mask_file}',
        f'--bvecs={bvec_file}',
        f'--bvals={bval_file}',
        f'--out={output_prefix}',
        '--sse',  # Save sum of squared errors
        '--save_tensor'  # Save tensor
    ]

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True)

    if result.returncode != 0:
        print(f"DTI fitting failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("FSL dtifit failed")

    print("  DTI fitting completed successfully")

    # Define output file paths (dtifit naming convention)
    fa_file = Path(str(output_prefix) + '_FA.nii.gz')
    md_file = Path(str(output_prefix) + '_MD.nii.gz')
    l1_file = Path(str(output_prefix) + '_L1.nii.gz')
    l2_file = Path(str(output_prefix) + '_L2.nii.gz')
    l3_file = Path(str(output_prefix) + '_L3.nii.gz')

    # Calculate AD and RD from eigenvalues
    # AD = L1 (axial diffusivity)
    # RD = (L2 + L3) / 2 (radial diffusivity)
    print("  Computing AD and RD from eigenvalues...")

    l1_img = nib.load(l1_file)
    l1_data = l1_img.get_fdata()

    l2_img = nib.load(l2_file)
    l2_data = l2_img.get_fdata()

    l3_img = nib.load(l3_file)
    l3_data = l3_img.get_fdata()

    # AD = L1
    ad_data = l1_data
    ad_file = Path(str(output_prefix) + '_AD.nii.gz')
    nib.save(nib.Nifti1Image(ad_data, l1_img.affine, l1_img.header), ad_file)

    # RD = (L2 + L3) / 2
    rd_data = (l2_data + l3_data) / 2.0
    rd_file = Path(str(output_prefix) + '_RD.nii.gz')
    nib.save(nib.Nifti1Image(rd_data, l1_img.affine, l1_img.header), rd_file)

    # Load FA and MD to check ranges
    fa_img = nib.load(fa_file)
    fa_data = fa_img.get_fdata()

    md_img = nib.load(md_file)
    md_data = md_img.get_fdata()

    print(f"  FA range: [{fa_data.min():.3f}, {fa_data.max():.3f}]")
    print(f"  MD range: [{md_data.min():.6f}, {md_data.max():.6f}]")
    print(f"  AD range: [{ad_data.min():.6f}, {ad_data.max():.6f}]")
    print(f"  RD range: [{rd_data.min():.6f}, {rd_data.max():.6f}]")

    return fa_file, md_file, ad_file, rd_file


def register_to_atlas_slices(
    moving_image: Path,
    fixed_image: Path,
    output_prefix: Path,
    output_warped: Path
) -> Path:
    """
    Register moving image to fixed atlas slices using ANTs SyN.

    Parameters
    ----------
    moving_image : Path
        Image to register (e.g., FA map)
    fixed_image : Path
        Fixed atlas image (slice-specific)
    output_prefix : Path
        Output prefix for transform files
    output_warped : Path
        Output path for warped image

    Returns
    -------
    Path
        Path to composite transform
    """
    print(f"\n  Moving: {moving_image.name}")
    print(f"  Fixed: {fixed_image.name}")

    # ANTs registration command (SyN)
    cmd = [
        'antsRegistration',
        '--dimensionality', '3',
        '--float', '1',
        '--output', f'[{output_prefix}_,{output_warped}]',
        '--interpolation', 'Linear',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--use-histogram-matching', '0',
        # Initial moving transform (center of mass)
        '--initial-moving-transform', f'[{fixed_image},{moving_image},1]',
        # Rigid registration
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{fixed_image},{moving_image},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        # Affine registration
        '--transform', 'Affine[0.1]',
        '--metric', f'MI[{fixed_image},{moving_image},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        # SyN deformable registration
        '--transform', 'SyN[0.1,3,0]',
        '--metric', f'CC[{fixed_image},{moving_image},1,4]',
        '--convergence', '[100x70x50x20,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox'
    ]

    print("\n  Running ANTs registration...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"Registration failed!")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("ANTs registration failed")

    print("  Registration completed successfully")

    composite_transform = Path(str(output_prefix) + '_Composite.h5')
    return composite_transform
