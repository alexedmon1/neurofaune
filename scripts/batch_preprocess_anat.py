#!/usr/bin/env python3
"""
Batch anatomical preprocessing with template building and registration.

This script implements the two-phase anatomical preprocessing workflow:

Phase 1: Template Building (subset of subjects)
- Select best subjects based on quality metrics
- Run preprocessing on selected subjects
- Build age-specific template
- Register template to SIGMA atlas

Phase 2: Full Processing (all subjects)
- Preprocess remaining subjects
- Register all subjects to cohort template
- Propagate SIGMA atlas to each subject

Usage:
    # Standard workflow with quality-based template building
    python scripts/batch_preprocess_anat.py /path/to/bids /path/to/output --config config.yaml

    # Skip template building (use existing templates)
    python scripts/batch_preprocess_anat.py /path/to/bids /path/to/output --skip-template-build

    # Direct-to-SIGMA mode (not recommended)
    python scripts/batch_preprocess_anat.py /path/to/bids /path/to/output --direct-to-sigma
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing
from neurofaune.templates.manifest import TemplateManifest, find_template_manifest
from neurofaune.templates.builder import (
    select_subjects_for_template,
    build_template,
    register_template_to_sigma,
    save_template_metadata
)
from neurofaune.templates.anat_registration import (
    register_anat_to_template,
    propagate_atlas_to_anat,
    register_anat_to_sigma_direct,
    propagate_atlas_direct
)
from neurofaune.templates.registration_qc import generate_template_qc_report
from neurofaune.utils.select_anatomical import is_3d_only_subject


def discover_subjects(bids_dir: Path, cohort: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Discover subjects in BIDS directory.

    Parameters
    ----------
    bids_dir : Path
        BIDS root directory
    cohort : str, optional
        Filter by cohort (e.g., 'p60')

    Returns
    -------
    list
        List of dicts with subject, session, and paths
    """
    subjects = []

    for subject_dir in sorted(bids_dir.glob('sub-*')):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        # Find sessions
        for session_dir in sorted(subject_dir.glob('ses-*')):
            if not session_dir.is_dir():
                continue

            session = session_dir.name

            # Filter by cohort if specified
            if cohort and cohort not in session:
                continue

            # Check for anatomical data
            anat_dir = session_dir / 'anat'
            if not anat_dir.exists():
                continue

            # Find T2w files
            t2w_files = list(anat_dir.glob('*_T2w.nii.gz'))
            if not t2w_files:
                continue

            subjects.append({
                'subject': subject,
                'session': session,
                'subject_dir': subject_dir,
                'anat_dir': anat_dir,
                't2w_files': t2w_files,
                'is_3d_only': is_3d_only_subject(subject_dir, session)
            })

    return subjects


def check_preprocessing_complete(
    output_dir: Path,
    subject: str,
    session: str
) -> Tuple[bool, Optional[Path]]:
    """
    Check if preprocessing outputs exist for a subject.

    Returns
    -------
    tuple
        (is_complete, preprocessed_t2w_path)
    """
    derivatives_dir = output_dir / 'derivatives' / subject / session / 'anat'
    preproc_t2w = derivatives_dir / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    brain_mask = derivatives_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'

    if preproc_t2w.exists() and brain_mask.exists():
        return True, preproc_t2w
    return False, None


def check_registration_complete(
    output_dir: Path,
    subject: str,
    session: str,
    mode: str = 'template'
) -> Tuple[bool, Dict[str, Path]]:
    """
    Check if registration outputs exist for a subject.

    Parameters
    ----------
    output_dir : Path
        Study root directory
    subject : str
        Subject ID
    session : str
        Session ID
    mode : str
        'template' or 'direct'

    Returns
    -------
    tuple
        (is_complete, paths_dict)
    """
    transforms_dir = output_dir / 'transforms' / subject / session

    if mode == 'template':
        affine = transforms_dir / f'{subject}_{session}_T2w_to_template_0GenericAffine.mat'
        warp = transforms_dir / f'{subject}_{session}_T2w_to_template_1Warp.nii.gz'
    else:
        affine = transforms_dir / f'{subject}_{session}_T2w_to_SIGMA_0GenericAffine.mat'
        warp = transforms_dir / f'{subject}_{session}_T2w_to_SIGMA_1Warp.nii.gz'

    if affine.exists():
        return True, {'affine': affine, 'warp': warp if warp.exists() else None}
    return False, {}


def select_template_subjects_quality(
    subjects: List[Dict[str, Any]],
    output_dir: Path,
    fraction: float = 0.2,
    min_subjects: int = 5,
    max_subjects: int = 30
) -> List[Dict[str, Any]]:
    """
    Select subjects for template building based on quality metrics.

    Uses SNR, CNR, and brain coverage metrics to rank subjects.

    Parameters
    ----------
    subjects : list
        List of subject info dicts
    output_dir : Path
        Study root for accessing QC data
    fraction : float
        Fraction of subjects to use (default: 20%)
    min_subjects : int
        Minimum subjects regardless of fraction
    max_subjects : int
        Maximum subjects regardless of fraction

    Returns
    -------
    list
        Selected subject info dicts, sorted by quality
    """
    print("\n" + "="*60)
    print("QUALITY-BASED TEMPLATE SUBJECT SELECTION")
    print("="*60)

    # Calculate target number
    n_total = len(subjects)
    n_target = max(min_subjects, min(max_subjects, int(n_total * fraction)))

    print(f"Total subjects: {n_total}")
    print(f"Target for template: {n_target} ({fraction*100:.0f}%)")

    # Score each subject based on available quality metrics
    scored_subjects = []

    for subj in subjects:
        subject = subj['subject']
        session = subj['session']

        # Check if preprocessed (required for quality assessment)
        is_preproc, preproc_path = check_preprocessing_complete(
            output_dir, subject, session
        )

        if not is_preproc:
            # Not yet preprocessed - assign neutral score
            scored_subjects.append({
                **subj,
                'quality_score': 0.5,
                'metrics': {'status': 'not_preprocessed'}
            })
            continue

        # Load preprocessed image and compute quality metrics
        try:
            import nibabel as nib
            img = nib.load(preproc_path)
            data = img.get_fdata()

            # Mask file
            mask_path = preproc_path.parent / f'{subject}_{session}_desc-brain_mask.nii.gz'
            if mask_path.exists():
                mask = nib.load(mask_path).get_fdata() > 0
            else:
                mask = data > 0

            # Compute metrics
            brain_data = data[mask]

            # SNR: mean / std
            snr = np.mean(brain_data) / (np.std(brain_data) + 1e-6)

            # Brain volume (number of voxels)
            brain_volume = np.sum(mask)

            # Intensity homogeneity (lower std = more homogeneous)
            homogeneity = 1.0 / (np.std(brain_data) / np.mean(brain_data) + 0.1)

            # Composite score (weighted combination)
            # Higher SNR, moderate volume, higher homogeneity = better
            score = (
                0.4 * min(snr / 10, 1.0) +  # SNR contribution (capped)
                0.3 * min(brain_volume / 50000, 1.0) +  # Volume contribution
                0.3 * min(homogeneity, 1.0)  # Homogeneity contribution
            )

            scored_subjects.append({
                **subj,
                'quality_score': score,
                'preprocessed_t2w': preproc_path,
                'brain_mask': mask_path,
                'metrics': {
                    'snr': float(snr),
                    'brain_volume': int(brain_volume),
                    'homogeneity': float(homogeneity)
                }
            })

        except Exception as e:
            print(f"  Warning: Could not score {subject} {session}: {e}")
            scored_subjects.append({
                **subj,
                'quality_score': 0.3,  # Low score for errors
                'metrics': {'error': str(e)}
            })

    # Sort by quality score (descending)
    scored_subjects.sort(key=lambda x: x['quality_score'], reverse=True)

    # Select top subjects
    selected = scored_subjects[:n_target]

    print(f"\nSelected {len(selected)} subjects for template building:")
    for i, subj in enumerate(selected):
        metrics = subj.get('metrics', {})
        print(f"  {i+1}. {subj['subject']} {subj['session']}: "
              f"score={subj['quality_score']:.3f}, "
              f"SNR={metrics.get('snr', 'N/A'):.1f}" if isinstance(metrics.get('snr'), (int, float)) else f"  {i+1}. {subj['subject']} {subj['session']}: score={subj['quality_score']:.3f}")

    return selected


def preprocess_subject(
    subj: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
    force: bool = False
) -> Dict[str, Any]:
    """
    Run preprocessing on a single subject.

    Returns
    -------
    dict
        Preprocessing results with status and paths
    """
    subject = subj['subject']
    session = subj['session']
    subject_dir = subj['subject_dir']

    # Check if already complete
    if not force:
        is_complete, preproc_path = check_preprocessing_complete(
            output_dir, subject, session
        )
        if is_complete:
            print(f"  {subject} {session}: Already preprocessed, skipping")
            return {
                'status': 'skipped',
                'preprocessed_t2w': preproc_path,
                'brain_mask': preproc_path.parent / f'{subject}_{session}_desc-brain_mask.nii.gz'
            }

    # Create transform registry
    cohort = session.replace('ses-', '')
    registry = create_transform_registry(config, subject, cohort=cohort)

    try:
        results = run_anatomical_preprocessing(
            config=config,
            subject=subject,
            session=session,
            output_dir=output_dir,
            transform_registry=registry,
            subject_dir=subject_dir
        )

        return {
            'status': 'success',
            'preprocessed_t2w': results['brain'],
            'brain_mask': results['mask'],
            'results': results
        }

    except Exception as e:
        print(f"  ERROR: {subject} {session}: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def run_phase1_template_building(
    subjects: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
    cohort: str,
    template_fraction: float = 0.2,
    force: bool = False
) -> Tuple[TemplateManifest, Path]:
    """
    Phase 1: Build template from subset of best subjects.

    Returns
    -------
    tuple
        (manifest, template_path)
    """
    print("\n" + "="*80)
    print("PHASE 1: TEMPLATE BUILDING")
    print("="*80)

    templates_dir = output_dir / 'templates' / 'anat' / cohort
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing template
    template_path = templates_dir / f'tpl-{cohort}_T2w.nii.gz'
    manifest_path = templates_dir / 'template_manifest.json'

    if template_path.exists() and manifest_path.exists() and not force:
        print(f"Template already exists: {template_path}")
        print(f"Loading existing manifest...")
        manifest = TemplateManifest.load(manifest_path)
        return manifest, template_path

    # Select subjects for template
    template_subjects = select_template_subjects_quality(
        subjects, output_dir, fraction=template_fraction
    )

    # Preprocess template subjects first
    print("\n" + "-"*60)
    print("Step 1.1: Preprocessing template subjects")
    print("-"*60)

    preprocessed_subjects = []
    for subj in template_subjects:
        print(f"\nProcessing {subj['subject']} {subj['session']}...")
        result = preprocess_subject(subj, output_dir, config, force=force)

        if result['status'] in ['success', 'skipped']:
            subj['preprocessed_t2w'] = result['preprocessed_t2w']
            subj['brain_mask'] = result['brain_mask']
            preprocessed_subjects.append(subj)
        else:
            print(f"  Skipping from template: preprocessing failed")

    if len(preprocessed_subjects) < 3:
        raise ValueError(
            f"Not enough preprocessed subjects for template building "
            f"(got {len(preprocessed_subjects)}, need at least 3)"
        )

    # Create manifest
    manifest = TemplateManifest(
        study_name=config.get('study', {}).get('name', 'unknown'),
        cohort=cohort,
        modality='anat',
        template_path=template_path,
        n_subjects=len(preprocessed_subjects),
        fraction_used=template_fraction,
        selection_method='quality'
    )

    # Add subjects to manifest
    for subj in preprocessed_subjects:
        manifest.add_subject(
            subject=subj['subject'],
            session=subj['session'],
            preprocessed_t2w=subj['preprocessed_t2w'],
            brain_mask=subj['brain_mask'],
            qc_metrics=subj.get('metrics')
        )

    # Build template
    print("\n" + "-"*60)
    print("Step 1.2: Building template")
    print("-"*60)

    input_files = [str(s['preprocessed_t2w']) for s in preprocessed_subjects]

    template_result = build_template(
        input_files=input_files,
        output_dir=templates_dir,
        template_name=f'tpl-{cohort}_T2w',
        n_iterations=4
    )

    template_path = Path(template_result['template'])
    print(f"Template built: {template_path}")

    # Register template to SIGMA
    print("\n" + "-"*60)
    print("Step 1.3: Registering template to SIGMA")
    print("-"*60)

    from neurofaune.atlas import AtlasManager
    atlas_mgr = AtlasManager(config)

    # Get SIGMA template path
    sigma_t2w = atlas_mgr.get_template_path('T2', masked=False)

    sigma_result = register_template_to_sigma(
        template_file=template_path,
        sigma_template=sigma_t2w,
        output_dir=templates_dir / 'transforms'
    )

    # Update manifest with SIGMA registration
    manifest.sigma_affine = Path(sigma_result['affine'])
    manifest.sigma_warp = Path(sigma_result['warp']) if sigma_result.get('warp') else None
    manifest.sigma_inverse_warp = Path(sigma_result['inverse_warp']) if sigma_result.get('inverse_warp') else None
    manifest.sigma_warped_template = Path(sigma_result['warped_template'])

    # Generate template QC
    print("\n" + "-"*60)
    print("Step 1.4: Generating template QC report")
    print("-"*60)

    qc_dir = output_dir / 'qc' / 'reports' / 'templates' / cohort
    qc_dir.mkdir(parents=True, exist_ok=True)

    generate_template_qc_report(
        template_file=template_path,
        sigma_template=sigma_t2w,
        warped_template=sigma_result['warped_template'],
        output_dir=qc_dir
    )

    # Save manifest
    manifest.save(manifest_path)

    print(f"\nPhase 1 complete!")
    print(f"  Template: {template_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  QC: {qc_dir}")

    return manifest, template_path


def run_phase2_full_processing(
    subjects: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
    manifest: TemplateManifest,
    template_path: Path,
    direct_mode: bool = False,
    force: bool = False
) -> Dict[str, Any]:
    """
    Phase 2: Process all subjects with registration.

    Returns
    -------
    dict
        Summary of processing results
    """
    print("\n" + "="*80)
    print("PHASE 2: FULL PROCESSING WITH REGISTRATION")
    print("="*80)

    cohort = manifest.cohort

    # Get SIGMA paths - use STUDY-SPACE atlas (already reoriented to match study)
    from neurofaune.atlas import AtlasManager
    atlas_mgr = AtlasManager(config)

    # Use study-space parcellation from config (already reoriented)
    if 'study_space' in config.get('atlas', {}) and 'parcellation' in config['atlas']['study_space']:
        sigma_parcellation = Path(config['atlas']['study_space']['parcellation'])
    else:
        # Fallback to standard location
        study_root = Path(config['paths']['study_root'])
        sigma_parcellation = study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz'

    if not sigma_parcellation.exists():
        raise FileNotFoundError(
            f"Study-space SIGMA parcellation not found at {sigma_parcellation}\n"
            "Run setup_study_atlas() first to create the reoriented atlas."
        )

    # Template mask
    template_mask = template_path.parent / f'tpl-{cohort}_T2w_mask.nii.gz'
    if not template_mask.exists():
        template_mask = None

    results = {
        'total': len(subjects),
        'preprocessed': 0,
        'registered': 0,
        'atlas_propagated': 0,
        'failed': [],
        'skipped_template_subjects': 0
    }

    for i, subj in enumerate(subjects):
        subject = subj['subject']
        session = subj['session']

        print(f"\n[{i+1}/{len(subjects)}] {subject} {session}")
        print("-" * 40)

        # Check if this was a template subject
        is_template_subject = manifest.is_template_subject(subject, session)
        if is_template_subject:
            print(f"  Template subject - checking if registration needed")
            # Template subjects already have preprocessing done
            subj_transforms = manifest.get_subject_transforms(subject, session)
            if subj_transforms:
                subj['preprocessed_t2w'] = subj_transforms.preprocessed_t2w
                subj['brain_mask'] = subj_transforms.brain_mask
                results['skipped_template_subjects'] += 1

        # Step 1: Preprocessing
        print(f"  Step 1: Preprocessing")

        if 'preprocessed_t2w' not in subj:
            preproc_result = preprocess_subject(subj, output_dir, config, force=force)

            if preproc_result['status'] == 'failed':
                results['failed'].append({
                    'subject': subject,
                    'session': session,
                    'stage': 'preprocessing',
                    'error': preproc_result.get('error', 'Unknown')
                })
                continue

            subj['preprocessed_t2w'] = preproc_result['preprocessed_t2w']
            subj['brain_mask'] = preproc_result['brain_mask']

        results['preprocessed'] += 1
        print(f"    ✓ Preprocessed")

        # Step 2: Registration
        print(f"  Step 2: Registration")

        transforms_dir = output_dir / 'transforms' / subject / session
        transforms_dir.mkdir(parents=True, exist_ok=True)

        # Check if already registered
        is_registered, reg_paths = check_registration_complete(
            output_dir, subject, session,
            mode='direct' if direct_mode else 'template'
        )

        if is_registered and not force:
            print(f"    ✓ Already registered, skipping")
            results['registered'] += 1
        else:
            try:
                if direct_mode:
                    # Direct to SIGMA (not recommended)
                    sigma_t2w = atlas_mgr.get_template_path('T2', masked=False)
                    reg_result = register_anat_to_sigma_direct(
                        t2w_file=subj['preprocessed_t2w'],
                        sigma_template=sigma_t2w,
                        output_dir=transforms_dir,
                        subject=subject,
                        session=session,
                        mask_file=subj['brain_mask']
                    )
                else:
                    # Template-based registration
                    reg_result = register_anat_to_template(
                        t2w_file=subj['preprocessed_t2w'],
                        template_file=template_path,
                        output_dir=transforms_dir,
                        subject=subject,
                        session=session,
                        mask_file=subj['brain_mask'],
                        template_mask=template_mask
                    )

                results['registered'] += 1
                print(f"    ✓ Registered (corr: {reg_result.get('metrics', {}).get('correlation_after', 'N/A'):.3f})")

            except Exception as e:
                print(f"    ✗ Registration failed: {e}")
                results['failed'].append({
                    'subject': subject,
                    'session': session,
                    'stage': 'registration',
                    'error': str(e)
                })
                continue

        # Step 3: Atlas propagation
        print(f"  Step 3: Atlas propagation")

        derivatives_dir = output_dir / 'derivatives' / subject / session / 'anat'
        atlas_output = derivatives_dir / f'{subject}_{session}_space-T2w_atlas-SIGMA.nii.gz'

        if atlas_output.exists() and not force:
            print(f"    ✓ Atlas already propagated, skipping")
            results['atlas_propagated'] += 1
        else:
            try:
                if direct_mode:
                    atlas_result = propagate_atlas_direct(
                        atlas_file=sigma_parcellation,
                        t2w_reference=subj['preprocessed_t2w'],
                        transforms_dir=output_dir / 'transforms',
                        subject=subject,
                        session=session,
                        output_file=atlas_output,
                        qc_dir=output_dir / 'qc' / 'subjects' / subject / session / 'anat'
                    )
                else:
                    atlas_result = propagate_atlas_to_anat(
                        atlas_file=sigma_parcellation,
                        t2w_reference=subj['preprocessed_t2w'],
                        transforms_dir=output_dir / 'transforms',
                        templates_dir=output_dir / 'templates',
                        subject=subject,
                        session=session,
                        output_file=atlas_output,
                        qc_dir=output_dir / 'qc' / 'subjects' / subject / session / 'anat'
                    )

                results['atlas_propagated'] += 1
                print(f"    ✓ Atlas propagated ({atlas_result['n_labels']} labels, {atlas_result['coverage']:.1%} coverage)")

            except Exception as e:
                print(f"    ✗ Atlas propagation failed: {e}")
                results['failed'].append({
                    'subject': subject,
                    'session': session,
                    'stage': 'atlas_propagation',
                    'error': str(e)
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch anatomical preprocessing with template building',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('bids_dir', type=Path, help='BIDS root directory')
    parser.add_argument('output_dir', type=Path, help='Output study directory')
    parser.add_argument('--config', type=Path, default=None, help='Config file path')
    parser.add_argument('--cohort', type=str, default=None, help='Filter by cohort (e.g., p60)')
    parser.add_argument('--template-fraction', type=float, default=0.2,
                        help='Fraction of subjects for template building (default: 0.2)')
    parser.add_argument('--skip-template-build', action='store_true',
                        help='Skip template building (use existing)')
    parser.add_argument('--direct-to-sigma', action='store_true',
                        help='Register directly to SIGMA (not recommended)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if outputs exist')
    parser.add_argument('--phase', choices=['1', '2', 'all'], default='all',
                        help='Run specific phase (1=template, 2=full, all=both)')
    parser.add_argument('--exclude-3d', action='store_true',
                        help='Exclude subjects that only have 3D T2w scans (no 2D available)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (not yet implemented)')

    args = parser.parse_args()

    # Validate paths
    if not args.bids_dir.exists():
        print(f"ERROR: BIDS directory not found: {args.bids_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config and args.config.exists():
        config = load_config(args.config)
    else:
        # Try default locations
        default_configs = [
            Path('config.yaml'),
            Path('configs/default.yaml'),
            args.output_dir / 'config.yaml'
        ]
        config = None
        for cfg_path in default_configs:
            if cfg_path.exists():
                config = load_config(cfg_path)
                print(f"Using config: {cfg_path}")
                break

        if config is None:
            print("ERROR: No config file found. Specify with --config")
            sys.exit(1)

    # Discover subjects
    print("\n" + "="*80)
    print("BATCH ANATOMICAL PREPROCESSING")
    print("="*80)
    print(f"BIDS directory: {args.bids_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cohort filter: {args.cohort or 'all'}")
    print(f"Template fraction: {args.template_fraction}")
    print(f"Mode: {'Direct-to-SIGMA' if args.direct_to_sigma else 'Template-based'}")

    subjects = discover_subjects(args.bids_dir, args.cohort)
    print(f"\nDiscovered {len(subjects)} subject/sessions")

    # Filter out 3D-only subjects if requested
    if args.exclude_3d:
        n_3d = sum(1 for s in subjects if s.get('is_3d_only', False))
        if n_3d > 0:
            subjects = [s for s in subjects if not s.get('is_3d_only', False)]
            print(f"Excluding {n_3d} 3D-only subject/sessions (--exclude-3d)")
            print(f"Remaining: {len(subjects)} subject/sessions")
    else:
        n_3d = sum(1 for s in subjects if s.get('is_3d_only', False))
        if n_3d > 0:
            print(f"Note: {n_3d} subject/sessions have only 3D T2w "
                  f"(use --exclude-3d to skip them)")

    if not subjects:
        print("No subjects found. Check BIDS directory structure.")
        sys.exit(1)

    # Group by cohort
    cohorts = {}
    for subj in subjects:
        cohort = subj['session'].replace('ses-', '')
        if cohort not in cohorts:
            cohorts[cohort] = []
        cohorts[cohort].append(subj)

    print(f"Cohorts: {list(cohorts.keys())}")
    for cohort, subj_list in cohorts.items():
        print(f"  {cohort}: {len(subj_list)} subjects")

    # Process each cohort
    all_results = {}

    for cohort, cohort_subjects in cohorts.items():
        print(f"\n\n{'#'*80}")
        print(f"# Processing cohort: {cohort}")
        print(f"{'#'*80}")

        # Phase 1: Template building
        if args.phase in ['1', 'all'] and not args.skip_template_build and not args.direct_to_sigma:
            manifest, template_path = run_phase1_template_building(
                subjects=cohort_subjects,
                output_dir=args.output_dir,
                config=config,
                cohort=cohort,
                template_fraction=args.template_fraction,
                force=args.force
            )
        elif args.direct_to_sigma:
            print("\nDirect-to-SIGMA mode: Skipping template building")
            manifest = None
            template_path = None
        else:
            # Load existing manifest
            manifest_path = find_template_manifest(
                args.output_dir / 'templates', cohort, 'anat'
            )
            if manifest_path:
                manifest = TemplateManifest.load(manifest_path)
                template_path = manifest.template_path
            else:
                print(f"ERROR: No template found for {cohort}. "
                      f"Run without --skip-template-build first.")
                continue

        # Phase 2: Full processing
        if args.phase in ['2', 'all']:
            if args.direct_to_sigma:
                # Create dummy manifest for direct mode
                manifest = TemplateManifest(
                    study_name=config.get('study', {}).get('name', 'unknown'),
                    cohort=cohort,
                    modality='anat',
                    template_path=Path('/dev/null'),
                    n_subjects=0,
                    fraction_used=0,
                    selection_method='direct'
                )

            results = run_phase2_full_processing(
                subjects=cohort_subjects,
                output_dir=args.output_dir,
                config=config,
                manifest=manifest,
                template_path=template_path,
                direct_mode=args.direct_to_sigma,
                force=args.force
            )

            all_results[cohort] = results

    # Print summary
    print("\n\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)

    for cohort, results in all_results.items():
        print(f"\n{cohort}:")
        print(f"  Total subjects: {results['total']}")
        print(f"  Preprocessed: {results['preprocessed']}")
        print(f"  Registered: {results['registered']}")
        print(f"  Atlas propagated: {results['atlas_propagated']}")
        print(f"  Template subjects (fast-tracked): {results['skipped_template_subjects']}")
        print(f"  Failed: {len(results['failed'])}")

        if results['failed']:
            print(f"\n  Failures:")
            for fail in results['failed']:
                print(f"    - {fail['subject']} {fail['session']}: "
                      f"{fail['stage']} - {fail['error'][:50]}...")

    # Save summary
    summary_file = args.output_dir / 'batch_anat_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'bids_dir': str(args.bids_dir),
            'output_dir': str(args.output_dir),
            'template_fraction': args.template_fraction,
            'direct_mode': args.direct_to_sigma,
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\nSummary saved to: {summary_file}")

    # Generate omnibus skull strip QC report
    total_preprocessed = sum(r['preprocessed'] for r in all_results.values())
    if total_preprocessed > 0:
        try:
            from neurofaune.preprocess.qc.batch_summary import generate_skull_strip_omnibus
            report = generate_skull_strip_omnibus(args.output_dir, 'anat')
            print(f"\nSkull strip omnibus report: {report}")
        except Exception as e:
            print(f"\nWarning: Could not generate omnibus QC report: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
