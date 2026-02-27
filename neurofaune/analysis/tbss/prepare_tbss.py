#!/usr/bin/env python3
"""
TBSS Data Preparation Workflow for Rodent DTI

Prepares DTI data for group-level TBSS analysis using the neurofaune
registration chain (FA→T2w→Template→SIGMA) instead of FSL's standard
tbss_2_reg pipeline.

This workflow:
1. Discovers subjects with completed DTI preprocessing and transforms
2. Warps FA/MD/AD/RD to SIGMA study-space using existing transform chain
3. Creates tissue-informed WM mask to remove exterior WM artifacts
4. Creates mean FA skeleton using FSL tbss_skeleton
5. Projects all metrics onto the FA skeleton
6. Generates subject manifest and analysis-ready 4D volumes

Workflow Order:
1. Run prepare_tbss_data() to create warped maps and skeleton
2. Run statistical analysis with run_tbss_stats.py

Usage:
    python -m neurofaune.analysis.tbss.prepare_tbss \\
        --config config.yaml \\
        --output-dir /study/analysis/tbss/ \\
        --metrics FA MD AD RD

    # With subject list from neuroaider design matrix:
    python -m neurofaune.analysis.tbss.prepare_tbss \\
        --config config.yaml \\
        --output-dir /study/analysis/tbss/ \\
        --subject-list /study/designs/model1/subject_list.txt
"""

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from neurofaune.config import load_config, get_config_value


# Supported DTI metrics
DTI_METRICS = ['FA', 'MD', 'AD', 'RD']


@dataclass
class SubjectData:
    """Container for subject data paths and validation status."""
    subject: str
    session: str
    cohort: str
    included: bool = False
    exclusion_reason: Optional[str] = None
    metric_files: Dict[str, Path] = field(default_factory=dict)
    # Pre-warped SIGMA-space files (from batch_register_dwi.py direct pipeline)
    sigma_metric_files: Dict[str, Path] = field(default_factory=dict)
    # Legacy transform paths (only used when use_prewarped=False)
    fa_to_t2w_affine: Optional[Path] = None
    t2w_to_tpl_affine: Optional[Path] = None
    t2w_to_tpl_warp: Optional[Path] = None
    tpl_to_sigma_affine: Optional[Path] = None
    tpl_to_sigma_warp: Optional[Path] = None


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prepare_tbss_{timestamp}.log"

    logger = logging.getLogger("neurofaune.tbss")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def discover_tbss_subjects(
    config: Dict,
    cohorts: List[str] = None,
    exclude_file: Optional[Path] = None,
    subject_list: Optional[List[str]] = None,
    metrics: List[str] = None,
    use_prewarped: bool = True
) -> List[SubjectData]:
    """
    Discover subjects with completed DTI preprocessing.

    When use_prewarped=True (default), looks for existing space-SIGMA files
    in derivatives/ produced by batch_register_dwi.py (direct FA→Template→SIGMA
    pipeline). This avoids needing the old FA→T2w→Template→SIGMA transform chain.

    When use_prewarped=False, validates the legacy transform chain.

    Args:
        config: Configuration dictionary
        cohorts: Cohorts to include (default: ['p30', 'p60', 'p90'])
        exclude_file: Path to exclusion list (one subject_session per line)
        subject_list: Explicit subject list (format: 'sub-Rat1_ses-p60' per line)
        metrics: DTI metrics to validate (default: ['FA', 'MD', 'AD', 'RD'])
        use_prewarped: If True, use existing space-SIGMA files (default: True)

    Returns:
        List of SubjectData with validation status
    """
    logger = logging.getLogger("neurofaune.tbss")

    if cohorts is None:
        cohorts = ['p30', 'p60', 'p90']
    if metrics is None:
        metrics = DTI_METRICS

    derivatives_dir = Path(get_config_value(config, 'paths.derivatives'))
    transforms_dir = Path(get_config_value(config, 'paths.transforms'))
    templates_dir = Path(get_config_value(config, 'paths.templates'))

    # Load exclusion list
    excluded_ids = set()
    if exclude_file and exclude_file.exists():
        with open(exclude_file) as f:
            excluded_ids = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(excluded_ids)} excluded subjects from {exclude_file}")

    # Determine subject/session pairs to check
    if subject_list:
        # Use explicit subject list (from neuroaider design matrix)
        pairs = []
        for entry in subject_list:
            entry = entry.strip()
            if '_ses-' in entry:
                subject, session = entry.split('_ses-')
                session = f'ses-{session}'
            else:
                subject = entry
                session = None
            pairs.append((subject, session))
        logger.info(f"Using provided subject list: {len(pairs)} entries")
    else:
        # Discover from derivatives directory
        pairs = []
        if not derivatives_dir.exists():
            logger.error(f"Derivatives directory not found: {derivatives_dir}")
            return []

        for subject_dir in sorted(derivatives_dir.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
                continue
            subject = subject_dir.name
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
                    continue
                session = session_dir.name
                cohort = session.replace('ses-', '')
                if cohort in cohorts:
                    pairs.append((subject, session))

        logger.info(f"Discovered {len(pairs)} subject/session pairs in {derivatives_dir}")

    # Validate each subject
    subjects_data = []
    for subject, session in pairs:
        if session is None:
            # Try to find session from derivatives
            subj_dir = derivatives_dir / subject
            if subj_dir.exists():
                sessions = [d.name for d in subj_dir.iterdir()
                            if d.is_dir() and d.name.startswith('ses-')]
                if sessions:
                    session = sessions[0]
                else:
                    subjects_data.append(SubjectData(
                        subject=subject, session='', cohort='',
                        included=False, exclusion_reason='No session found'
                    ))
                    continue

        cohort = session.replace('ses-', '')
        subject_session = f"{subject}_{session}"

        # Check exclusion
        if subject_session in excluded_ids or subject in excluded_ids:
            subjects_data.append(SubjectData(
                subject=subject, session=session, cohort=cohort,
                included=False, exclusion_reason='In exclusion list'
            ))
            continue

        # Validate metric files
        sd = SubjectData(subject=subject, session=session, cohort=cohort)
        dwi_dir = derivatives_dir / subject / session / 'dwi'

        if use_prewarped:
            # Look for existing space-SIGMA files from batch_register_dwi.py
            missing_sigma = []
            for metric in metrics:
                sigma_file = dwi_dir / f'{subject}_{session}_space-SIGMA_{metric}.nii.gz'
                if sigma_file.exists():
                    sd.sigma_metric_files[metric] = sigma_file
                else:
                    missing_sigma.append(metric)

            if missing_sigma:
                sd.exclusion_reason = f"Missing SIGMA-space maps: {', '.join(missing_sigma)}"
                subjects_data.append(sd)
                continue

            # Also record native-space files if they exist (for reference)
            for metric in metrics:
                metric_file = _find_metric_file(dwi_dir, subject, session, metric)
                if metric_file:
                    sd.metric_files[metric] = metric_file

            sd.included = True
            subjects_data.append(sd)

        else:
            # Legacy mode: validate native metrics + full transform chain
            missing_metrics = []
            for metric in metrics:
                metric_file = _find_metric_file(dwi_dir, subject, session, metric)
                if metric_file:
                    sd.metric_files[metric] = metric_file
                else:
                    missing_metrics.append(metric)

            if missing_metrics:
                sd.exclusion_reason = f"Missing metrics: {', '.join(missing_metrics)}"
                subjects_data.append(sd)
                continue

            # Validate transforms
            subj_transforms = transforms_dir / subject / session
            tpl_transforms = templates_dir / 'anat' / cohort / 'transforms'

            sd.fa_to_t2w_affine = _find_transform(
                subj_transforms, 'FA_to_T2w_0GenericAffine.mat'
            )
            sd.t2w_to_tpl_affine = _find_transform(
                subj_transforms,
                f'{subject}_{session}_T2w_to_template_0GenericAffine.mat',
                'T2w_to_template_0GenericAffine.mat'
            )
            sd.t2w_to_tpl_warp = _find_transform(
                subj_transforms,
                f'{subject}_{session}_T2w_to_template_1Warp.nii.gz',
                'T2w_to_template_1Warp.nii.gz'
            )
            sd.tpl_to_sigma_affine = _find_transform(
                tpl_transforms, 'tpl-to-SIGMA_0GenericAffine.mat'
            )
            sd.tpl_to_sigma_warp = _find_transform(
                tpl_transforms, 'tpl-to-SIGMA_1Warp.nii.gz'
            )

            missing_transforms = []
            if not sd.fa_to_t2w_affine:
                missing_transforms.append('FA_to_T2w')
            if not sd.t2w_to_tpl_affine:
                missing_transforms.append('T2w_to_template_affine')
            if not sd.tpl_to_sigma_affine:
                missing_transforms.append('tpl_to_SIGMA_affine')

            if missing_transforms:
                sd.exclusion_reason = f"Missing transforms: {', '.join(missing_transforms)}"
                subjects_data.append(sd)
                continue

            sd.included = True
            subjects_data.append(sd)

    included = sum(1 for s in subjects_data if s.included)
    excluded = sum(1 for s in subjects_data if not s.included)
    logger.info(f"Validation: {included} included, {excluded} excluded")

    return subjects_data


def _find_metric_file(
    dwi_dir: Path, subject: str, session: str, metric: str
) -> Optional[Path]:
    """Find DTI metric file trying multiple naming patterns."""
    patterns = [
        dwi_dir / f'{subject}_{session}_{metric}.nii.gz',
        dwi_dir / f'{metric}.nii.gz',
        dwi_dir / f'{subject}_{metric}.nii.gz',
        dwi_dir / 'dti' / f'{metric}.nii.gz',
        dwi_dir / 'dti' / f'dtifit__{metric}.nii.gz',
    ]
    for p in patterns:
        if p.exists():
            return p
    return None


def _find_transform(directory: Path, *filenames: str) -> Optional[Path]:
    """Find transform file trying multiple naming patterns."""
    for name in filenames:
        path = directory / name
        if path.exists():
            return path
    return None


def warp_metric_to_sigma(
    metric_file: Path,
    subject_data: SubjectData,
    config: Dict,
    output_file: Path,
    interpolation: str = 'Linear'
) -> Path:
    """
    Warp a DTI metric map to SIGMA study-space using the full transform chain.

    Transform chain (forward): FA → T2w → Template → SIGMA
    ANTs applies transforms in reverse order, so they're listed SIGMA-first.

    Args:
        metric_file: Input metric NIfTI (in FA/DWI space)
        subject_data: SubjectData with transform paths
        config: Configuration dictionary
        output_file: Output path in SIGMA space
        interpolation: Interpolation method ('Linear', 'NearestNeighbor')

    Returns:
        Path to warped metric in SIGMA space
    """
    logger = logging.getLogger("neurofaune.tbss")

    # Reference image: SIGMA study-space template
    atlas_config = get_config_value(config, 'atlas.study_space', default={})
    sigma_template = Path(atlas_config.get('template', ''))

    if not sigma_template.exists():
        # Try default location
        study_root = Path(get_config_value(config, 'paths.study_root'))
        sigma_template = study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_Brain_Template_Masked.nii.gz'

    if not sigma_template.exists():
        raise FileNotFoundError(f"SIGMA template not found: {sigma_template}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build transform chain (listed in ANTs order: last-applied first)
    # Goal: FA-space → SIGMA-space
    # Application order: FA→T2w → T2w→Template → Template→SIGMA
    # ANTs list order: Template→SIGMA, T2w→Template, FA→T2w
    transforms = []

    # 3. Template → SIGMA (applied last, listed first)
    if subject_data.tpl_to_sigma_warp:
        transforms.append(str(subject_data.tpl_to_sigma_warp))
    transforms.append(str(subject_data.tpl_to_sigma_affine))

    # 2. T2w → Template
    if subject_data.t2w_to_tpl_warp:
        transforms.append(str(subject_data.t2w_to_tpl_warp))
    transforms.append(str(subject_data.t2w_to_tpl_affine))

    # 1. FA → T2w (applied first, listed last)
    transforms.append(str(subject_data.fa_to_t2w_affine))

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(metric_file),
        '-r', str(sigma_template),
        '-o', str(output_file),
        '-n', interpolation
    ]
    for t in transforms:
        cmd.extend(['-t', t])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"antsApplyTransforms failed for {metric_file.name}: {result.stderr}")
        raise RuntimeError(f"Transform failed for {subject_data.subject}/{subject_data.session}")

    return output_file


def create_wm_mask(
    config: Dict,
    output_dir: Path,
    reference_img: Path,
    wm_prob_threshold: float = 0.3,
    erosion_voxels: int = 2
) -> Tuple[Path, Path]:
    """
    Create interior WM mask excluding exterior WM artifacts.

    Uses SIGMA study-space WM probability template to identify true WM,
    then erodes from brain boundary to remove exterior voxels. No FA
    gating — the mask is purely tissue-informed so it can be shared
    across modalities (DTI, MSME, etc.).

    Args:
        config: Configuration dictionary
        output_dir: Output directory for masks
        reference_img: Path to a reference NIfTI in SIGMA space (for
            affine/shape; typically mean FA or the SIGMA template)
        wm_prob_threshold: Minimum WM probability (default: 0.3)
        erosion_voxels: Voxels to erode from brain boundary

    Returns:
        Tuple of (interior_wm_mask, exterior_removed_mask) paths
    """
    logger = logging.getLogger("neurofaune.tbss")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(reference_img)

    # Load SIGMA WM probability
    study_root = Path(get_config_value(config, 'paths.study_root'))
    wm_prob_file = study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_WM.nii.gz'

    if not wm_prob_file.exists():
        raise FileNotFoundError(
            f"SIGMA WM probability template not found: {wm_prob_file}"
        )

    wm_prob_img = nib.load(wm_prob_file)
    wm_prob_data = wm_prob_img.get_fdata()

    # Normalize WM probability to [0,1] if needed
    if wm_prob_data.max() > 1.0:
        wm_prob_data = wm_prob_data / wm_prob_data.max()

    # Create WM mask from probability only
    wm_mask = (wm_prob_data > wm_prob_threshold).astype(np.uint8)

    # Save WM probability in SIGMA space for reference
    wm_prob_out = output_dir / 'wm_probability_sigma.nii.gz'
    nib.save(nib.Nifti1Image(wm_prob_data, ref_img.affine), wm_prob_out)

    # Erode from brain boundary to remove exterior WM
    if erosion_voxels > 0:
        from scipy import ndimage

        # Use SIGMA brain mask for erosion (covers full brain, not just WM)
        brain_mask_file = study_root / 'atlas' / 'SIGMA_study_space' / 'SIGMA_InVivo_Brain_Mask.nii.gz'
        if brain_mask_file.exists():
            brain_mask = (nib.load(brain_mask_file).get_fdata() > 0).astype(np.uint8)
        else:
            logger.warning("SIGMA brain mask not found, using WM probability extent")
            brain_mask = (wm_prob_data > 0).astype(np.uint8)

        # Erode brain mask
        struct = ndimage.generate_binary_structure(3, 1)
        eroded_brain = ndimage.binary_erosion(
            brain_mask, structure=struct, iterations=erosion_voxels
        ).astype(np.uint8)

        # Interior WM = WM mask AND eroded brain
        interior_wm = (wm_mask & eroded_brain).astype(np.uint8)

        # What was removed
        exterior_removed = (wm_mask & ~eroded_brain).astype(np.uint8)
    else:
        interior_wm = wm_mask
        exterior_removed = np.zeros_like(wm_mask)

    # Remove small isolated clusters
    from scipy import ndimage as ndi
    labeled, n_features = ndi.label(interior_wm)
    if n_features > 0:
        component_sizes = ndi.sum(interior_wm, labeled, range(1, n_features + 1))
        min_component_size = 50  # Remove clusters < 50 voxels
        for i, size in enumerate(component_sizes):
            if size < min_component_size:
                interior_wm[labeled == (i + 1)] = 0

    # Save masks
    interior_mask_file = output_dir / 'interior_wm_mask.nii.gz'
    exterior_mask_file = output_dir / 'exterior_wm_removed.nii.gz'

    nib.save(nib.Nifti1Image(interior_wm, ref_img.affine), interior_mask_file)
    nib.save(nib.Nifti1Image(exterior_removed, ref_img.affine), exterior_mask_file)

    n_interior = int(np.sum(interior_wm))
    n_exterior = int(np.sum(exterior_removed))
    logger.info(f"WM mask: {n_interior} interior voxels, {n_exterior} exterior removed")

    return interior_mask_file, exterior_mask_file


def create_skeleton(
    mean_fa: Path,
    wm_mask: Path,
    output_dir: Path,
    skeleton_threshold: float = 0.2,
    smoothing_sigma: float = 0.7,
    min_cluster_size: int = 20
) -> Dict[str, Path]:
    """
    Create FA skeleton using FSL tbss_skeleton with connectivity-aware processing.

    Strategy for rodent data (warped from thick coronal slices):
    1. Smooth the unmasked mean FA to bridge small gaps from resampling
    2. Run tbss_skeleton on the smoothed FA
    3. Intersect skeleton with WM mask to keep only WM voxels
    4. Remove small disconnected fragments

    Args:
        mean_fa: Path to mean FA in SIGMA space
        wm_mask: Path to interior WM mask
        output_dir: Output directory for skeleton files
        skeleton_threshold: FA threshold for skeleton (default: 0.2)
        smoothing_sigma: Gaussian smoothing sigma in voxels before
            skeletonization (default: 0.7, bridges gaps from anisotropic
            resampling without over-smoothing)
        min_cluster_size: Remove skeleton fragments smaller than this
            (default: 20 voxels)

    Returns:
        Dict with paths to skeleton files
    """
    from scipy import ndimage as ndi

    logger = logging.getLogger("neurofaune.tbss")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mean FA and WM mask
    fa_img = nib.load(mean_fa)
    fa_data = fa_img.get_fdata()
    mask_data = nib.load(wm_mask).get_fdata() > 0

    # Save WM-masked FA for reference
    masked_fa = (fa_data * mask_data).astype(np.float32)
    mean_fa_masked = output_dir / 'mean_FA_masked.nii.gz'
    nib.save(nib.Nifti1Image(masked_fa, fa_img.affine), mean_fa_masked)

    # Smooth FA within WM mask to bridge gaps from anisotropic resampling
    # Uses normalized convolution: smooth(FA*mask) / smooth(mask) to avoid
    # edge darkening while keeping values constrained to WM
    if smoothing_sigma > 0:
        smooth_numerator = ndi.gaussian_filter(masked_fa, sigma=smoothing_sigma)
        smooth_denominator = ndi.gaussian_filter(mask_data.astype(np.float32), sigma=smoothing_sigma)
        smooth_denominator[smooth_denominator < 0.01] = 1.0  # avoid div-by-zero
        smoothed_fa = smooth_numerator / smooth_denominator
        smoothed_fa[~mask_data] = 0  # zero outside WM mask
        logger.info(f"Smoothed mean FA within WM mask (sigma={smoothing_sigma})")
    else:
        smoothed_fa = masked_fa

    smoothed_fa_file = output_dir / 'mean_FA_smoothed.nii.gz'
    nib.save(nib.Nifti1Image(smoothed_fa.astype(np.float32), fa_img.affine), smoothed_fa_file)

    # Run FSL tbss_skeleton on smoothed WM-masked FA
    skeleton_file = output_dir / 'mean_FA_skeleton.nii.gz'
    dist_file = output_dir / 'mean_FA_skeleton_mask_dst.nii.gz'

    cmd = [
        'tbss_skeleton',
        '-i', str(smoothed_fa_file),
        '-o', str(skeleton_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"tbss_skeleton failed: {result.stderr}")
        raise RuntimeError("FSL tbss_skeleton failed")

    # Threshold skeleton AND intersect with WM mask
    skel_img = nib.load(skeleton_file)
    skel_data = skel_img.get_fdata()
    skel_mask = ((skel_data >= skeleton_threshold) & mask_data).astype(np.uint8)

    # Topology-preserving gap bridging for skeleton connectivity
    # 1. Dilate skeleton by 1 within WM (connects all components)
    # 2. Remove added voxels that don't contribute to connectivity
    #    (simple-point removal, farthest from original skeleton first)
    from scipy.ndimage import distance_transform_edt as dist_edt
    struct_26 = ndi.generate_binary_structure(3, 2)  # 26-connectivity

    dilated = ndi.binary_dilation(skel_mask > 0, structure=struct_26, iterations=1)
    dilated = dilated & mask_data  # Stay within WM

    # Only thin if dilation actually helped connectivity
    labeled_pre, n_pre = ndi.label(skel_mask)
    labeled_dil, n_dil = ndi.label(dilated)
    if n_dil < n_pre:
        # Identify added voxels, sort by distance from original (remove farthest first)
        added_mask = dilated & ~(skel_mask > 0)
        added_coords = np.argwhere(added_mask)
        dist_from_skel = dist_edt(~(skel_mask > 0))
        coord_distances = np.array([
            dist_from_skel[c[0], c[1], c[2]] for c in added_coords
        ])
        sort_idx = np.argsort(-coord_distances)  # Farthest first
        added_coords = added_coords[sort_idx]

        # Remove each added voxel if it doesn't disconnect its neighborhood
        current = dilated.copy()
        for coord in added_coords:
            x, y, z = coord
            if not current[x, y, z]:
                continue
            x_min = max(0, x - 1)
            x_max = min(current.shape[0] - 1, x + 1) + 1
            y_min = max(0, y - 1)
            y_max = min(current.shape[1] - 1, y + 1) + 1
            z_min = max(0, z - 1)
            z_max = min(current.shape[2] - 1, z + 1) + 1
            neighborhood = current[x_min:x_max, y_min:y_max, z_min:z_max].copy()
            cx, cy, cz = x - x_min, y - y_min, z - z_min
            neighborhood[cx, cy, cz] = False
            _, n_components = ndi.label(neighborhood)
            if n_components <= 1:
                current[x, y, z] = False

        skel_mask = current.astype(np.uint8)
        n_added = int(skel_mask.sum() - (labeled_pre > 0).sum())
        logger.info(f"Topology-preserving bridging: added {n_added} voxels for connectivity")
    else:
        logger.info("Skeleton already maximally connected, no bridging needed")

    # Remove small disconnected fragments
    if min_cluster_size > 1:
        labeled, n_clusters = ndi.label(skel_mask)
        if n_clusters > 0:
            sizes = ndi.sum(skel_mask, labeled, range(1, n_clusters + 1))
            n_removed = 0
            for i, size in enumerate(sizes):
                if size < min_cluster_size:
                    skel_mask[labeled == (i + 1)] = 0
                    n_removed += 1
            logger.info(f"Removed {n_removed} skeleton fragments < {min_cluster_size} voxels")

    skeleton_mask_file = output_dir / 'mean_FA_skeleton_mask.nii.gz'
    nib.save(nib.Nifti1Image(skel_mask, skel_img.affine), skeleton_mask_file)

    # Report connectivity
    labeled_final, n_final = ndi.label(skel_mask)
    if n_final > 0:
        sizes_final = ndi.sum(skel_mask, labeled_final, range(1, n_final + 1))
        largest_pct = max(sizes_final) / np.sum(skel_mask) * 100
        logger.info(f"Skeleton connectivity: {n_final} clusters, "
                    f"largest = {largest_pct:.1f}%")

    # Create distance map from skeleton mask using FSL distancemap
    cmd_dist = [
        'distancemap',
        '-i', str(skeleton_mask_file),
        '-o', str(dist_file)
    ]
    result_dist = subprocess.run(cmd_dist, capture_output=True, text=True)
    if result_dist.returncode != 0:
        # Fallback: compute distance map with scipy
        logger.warning(f"FSL distancemap failed, using scipy fallback: {result_dist.stderr}")
        from scipy.ndimage import distance_transform_edt
        dist_data = distance_transform_edt(1 - skel_mask)
        nib.save(nib.Nifti1Image(dist_data.astype(np.float32), skel_img.affine), dist_file)

    # Create search rule mask (zeros = no direction constraints)
    # FSL uses LowerCingulum_1mm for humans; rodents don't need this
    search_rule_file = output_dir / 'search_rule_mask.nii.gz'
    search_rule = np.zeros_like(skel_mask)
    nib.save(nib.Nifti1Image(search_rule, skel_img.affine), search_rule_file)

    n_skel_voxels = int(np.sum(skel_mask))
    logger.info(f"Skeleton: {n_skel_voxels} voxels at threshold {skeleton_threshold}")

    return {
        'mean_FA_masked': mean_fa_masked,
        'mean_FA_skeleton': skeleton_file,
        'mean_FA_skeleton_mask': skeleton_mask_file,
        'distance_map': dist_file,
        'search_rule_mask': search_rule_file,
        'skeleton_threshold': skeleton_threshold
    }


def project_to_skeleton(
    metric_4d: Path,
    mean_fa: Path,
    skeleton_mask: Path,
    output_file: Path,
    skeleton_threshold: float = 0.2,
    distance_map: Optional[Path] = None,
    search_rule_mask: Optional[Path] = None
) -> Path:
    """
    Project DTI metric onto FA skeleton using FSL tbss_skeleton -p.

    Uses perpendicular search from skeleton to find peak metric values,
    similar to FSL's tbss_4_prestats.

    FSL syntax: tbss_skeleton -i mean_FA -p thresh distmap searchrule data4D output

    Args:
        metric_4d: 4D metric volume (subjects concatenated)
        mean_fa: Mean FA used to create skeleton
        skeleton_mask: Binary skeleton mask
        output_file: Output skeletonised 4D volume
        skeleton_threshold: FA threshold for skeleton
        distance_map: Pre-computed distance map from skeleton
        search_rule_mask: Search direction constraint mask

    Returns:
        Path to skeletonised 4D output
    """
    logger = logging.getLogger("neurofaune.tbss")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats_dir = output_file.parent
    if distance_map is None:
        distance_map = stats_dir / 'mean_FA_skeleton_mask_dst.nii.gz'
    if search_rule_mask is None:
        search_rule_mask = stats_dir / 'search_rule_mask.nii.gz'

    if not Path(distance_map).exists():
        raise FileNotFoundError(f"Distance map not found: {distance_map}")

    # FSL tbss_skeleton -p: thresh distmap searchrule data4D output
    cmd = [
        'tbss_skeleton',
        '-i', str(mean_fa),
        '-p', str(skeleton_threshold),
        str(distance_map),
        str(search_rule_mask),
        str(metric_4d),
        str(output_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"tbss_skeleton projection failed: {result.stderr}")
        logger.error(f"Command: {' '.join(cmd)}")
        raise RuntimeError(f"Skeleton projection failed for {metric_4d.name}")

    if output_file.exists():
        img = nib.load(output_file)
        logger.info(f"Projected {metric_4d.name} -> {output_file.name} (shape: {img.shape})")
    else:
        raise RuntimeError(f"Expected output not found: {output_file}")

    # Mask output with custom skeleton and fill gaps with direct metric values
    _apply_skeleton_mask_and_fill(output_file, skeleton_mask, metric_4d)

    return output_file


def _apply_skeleton_mask_and_fill(
    skeletonised_file: Path,
    skeleton_mask: Path,
    raw_metric_4d: Path,
) -> None:
    """
    Mask FSL's projection output with our custom skeleton mask, and fill
    skeleton voxels that got zero values from FSL (bridge voxels not in FSL's
    internal skeleton) with direct values from the warped metric maps.
    """
    skel_mask_data = nib.load(skeleton_mask).get_fdata() > 0
    skel_img = nib.load(skeletonised_file)
    skel_vals = skel_img.get_fdata()
    raw_vals = nib.load(raw_metric_4d).get_fdata()

    mask_4d = skel_mask_data[:, :, :, np.newaxis]
    # Mask to skeleton
    masked = skel_vals * mask_4d
    # Fill missing skeleton voxels with direct values
    for subj in range(skel_vals.shape[3]):
        missing = skel_mask_data & (masked[:, :, :, subj] == 0)
        masked[:, :, :, subj][missing] = raw_vals[:, :, :, subj][missing]

    nib.save(
        nib.Nifti1Image(masked.astype(np.float32), skel_img.affine, skel_img.header),
        skeletonised_file,
    )


def project_nonfa_to_skeleton(
    metric_4d: Path,
    fa_4d: Path,
    mean_fa: Path,
    skeleton_mask: Path,
    output_file: Path,
    skeleton_threshold: float = 0.2,
    distance_map: Optional[Path] = None,
    search_rule_mask: Optional[Path] = None
) -> Path:
    """
    Project non-FA metric onto FA skeleton using -a flag.

    Uses the FA skeleton structure but projects alternate metric values.
    FSL syntax: tbss_skeleton -i mean_FA -p thresh distmap searchrule FA_4D FA_skel -a alt_4D alt_output

    Args:
        metric_4d: 4D non-FA metric volume (MD, AD, or RD)
        fa_4d: 4D FA volume (reference for skeleton search)
        mean_fa: Mean FA (defines skeleton structure)
        output_file: Output skeletonised 4D volume
        skeleton_threshold: FA threshold for skeleton
        distance_map: Pre-computed distance map
        search_rule_mask: Search direction constraint mask

    Returns:
        Path to skeletonised 4D output
    """
    logger = logging.getLogger("neurofaune.tbss")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats_dir = output_file.parent
    if distance_map is None:
        distance_map = stats_dir / 'mean_FA_skeleton_mask_dst.nii.gz'
    if search_rule_mask is None:
        search_rule_mask = stats_dir / 'search_rule_mask.nii.gz'

    # FSL: tbss_skeleton -i mean_FA -p thresh distmap searchrule FA_4D output -a alt_4D
    # The output goes in the projected_4Ddata position; -a specifies alt data to sample
    cmd = [
        'tbss_skeleton',
        '-i', str(mean_fa),
        '-p', str(skeleton_threshold),
        str(distance_map),
        str(search_rule_mask),
        str(fa_4d),
        str(output_file),
        '-a', str(metric_4d),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Non-FA projection failed: {result.stderr}")
        logger.error(f"Command: {' '.join(cmd)}")
        raise RuntimeError(f"Non-FA skeleton projection failed for {metric_4d.name}")

    if output_file.exists():
        img = nib.load(output_file)
        logger.info(f"Projected {metric_4d.name} -> {output_file.name} (shape: {img.shape})")
    else:
        raise RuntimeError(f"Expected output not found: {output_file}")

    # Mask output with custom skeleton and fill gaps with direct metric values
    _apply_skeleton_mask_and_fill(output_file, skeleton_mask, metric_4d)

    return output_file


def generate_manifest(
    subjects_data: List[SubjectData],
    output_dir: Path,
    metrics: List[str]
) -> Dict:
    """
    Generate manifest file documenting included/excluded subjects.

    Matches neurovrai format: subject_manifest.json + subject_list.txt.

    Args:
        subjects_data: List of SubjectData with validation status
        output_dir: Output directory
        metrics: Metrics prepared

    Returns:
        Manifest dictionary
    """
    included = [s for s in subjects_data if s.included]
    excluded = [s for s in subjects_data if not s.included]

    manifest = {
        "analysis_type": "TBSS",
        "pipeline": "neurofaune",
        "metrics": metrics,
        "date_prepared": datetime.now().isoformat(),
        "total_subjects_discovered": len(subjects_data),
        "subjects_included": len(included),
        "subjects_excluded": len(excluded),
        "included_subjects": [
            {
                "subject_id": f"{s.subject}_{s.session}",
                "subject": s.subject,
                "session": s.session,
                "cohort": s.cohort
            }
            for s in included
        ],
        "excluded_subjects": [
            {
                "subject_id": f"{s.subject}_{s.session}",
                "subject": s.subject,
                "session": s.session,
                "cohort": s.cohort,
                "reason": s.exclusion_reason
            }
            for s in excluded
        ]
    }

    # Write manifest JSON
    manifest_file = output_dir / "subject_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Write simple subject list (matches neurovrai format)
    subject_list_file = output_dir / "subject_list.txt"
    with open(subject_list_file, 'w') as f:
        for s in included:
            f.write(f"{s.subject}_{s.session}\n")

    return manifest


def prepare_tbss_data(
    config: Dict,
    output_dir: Path,
    metrics: List[str] = None,
    cohorts: List[str] = None,
    subjects: Optional[List[str]] = None,
    exclude_file: Optional[Path] = None,
    wm_prob_threshold: float = 0.3,
    erosion_voxels: int = 2,
    dry_run: bool = False,
    use_prewarped: bool = True
) -> Dict:
    """
    Main workflow: Prepare TBSS-style analysis from preprocessed DTI data.

    Uses a tissue-informed WM mask (SIGMA WM probability with boundary
    erosion) as the analysis mask. The mask is modality-agnostic and can
    be shared across DTI, MSME, etc.

    Phases:
    1. Discover and validate subjects
    2. Collect SIGMA-space metrics (prewarped) or warp from native space
    3. Create mean FA and WM mask
    4. Create analysis mask (interior WM mask)
    5. Mask metrics with analysis mask
    6. Generate manifest

    Args:
        config: Configuration dictionary
        output_dir: Output directory for TBSS analysis
        metrics: Metrics to prepare (default: ['FA', 'MD', 'AD', 'RD'])
        cohorts: Cohorts to include (default: all)
        subjects: Explicit subject list (format: 'sub-Rat1_ses-p60')
        exclude_file: Path to exclusion list
        wm_prob_threshold: WM probability threshold
        erosion_voxels: Erosion from brain boundary
        dry_run: If True, only discover subjects without processing
        use_prewarped: If True, use existing space-SIGMA files from
            batch_register_dwi.py instead of re-warping (default: True)

    Returns:
        Dictionary with preparation results
    """
    if metrics is None:
        metrics = DTI_METRICS
    if cohorts is None:
        cohorts = ['p30', 'p60', 'p90']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("TBSS Data Preparation (neurofaune)")
    logger.info("=" * 80)
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Cohorts: {cohorts}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"WM probability threshold: {wm_prob_threshold}")
    logger.info(f"Erosion voxels: {erosion_voxels}")
    logger.info(f"Use prewarped SIGMA files: {use_prewarped}")

    # Phase 1: Discover subjects
    logger.info("\n[Phase 1] Discovering subjects...")
    subjects_data = discover_tbss_subjects(
        config=config,
        cohorts=cohorts,
        exclude_file=exclude_file,
        subject_list=subjects,
        metrics=metrics,
        use_prewarped=use_prewarped
    )

    included = [s for s in subjects_data if s.included]
    if not included:
        logger.error("No valid subjects found!")
        return {"success": False, "error": "No valid subjects found"}

    logger.info(f"Found {len(included)} valid subjects")

    if dry_run:
        manifest = generate_manifest(subjects_data, output_dir, metrics)
        logger.info("Dry run complete. Subject manifest generated.")
        return {"success": True, "manifest": manifest, "dry_run": True}

    # Phase 2: Collect SIGMA-space metrics
    stats_dir = output_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    if use_prewarped:
        logger.info("\n[Phase 2] Linking prewarped SIGMA-space metrics...")
        import shutil

        for metric in metrics:
            metric_dir = output_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)

            # Clean stale files from previous runs that are not in the
            # current included list, so Phase 3/5 globs stay consistent
            included_filenames = {
                f'{sd.subject}_{sd.session}_{metric}_sigma.nii.gz'
                for sd in included
            }
            for existing in metric_dir.glob('*_sigma.nii.gz'):
                if existing.name not in included_filenames:
                    existing.unlink()
                    logger.info(f"  Removed stale: {existing.name}")

            for i, sd in enumerate(included):
                out_file = metric_dir / f'{sd.subject}_{sd.session}_{metric}_sigma.nii.gz'
                if out_file.exists():
                    logger.info(f"  [{i+1}/{len(included)}] {sd.subject}/{sd.session} {metric} (exists)")
                    continue

                source = sd.sigma_metric_files[metric]
                logger.info(f"  [{i+1}/{len(included)}] {sd.subject}/{sd.session} {metric}")
                try:
                    shutil.copy2(source, out_file)
                except Exception as e:
                    logger.error(f"  Failed to copy {source}: {e}")
                    sd.included = False
                    sd.exclusion_reason = f"Copy failed: {e}"

    else:
        logger.info("\n[Phase 2] Warping metrics to SIGMA space...")

        for metric in metrics:
            metric_dir = output_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)

            for i, sd in enumerate(included):
                out_file = metric_dir / f'{sd.subject}_{sd.session}_{metric}_sigma.nii.gz'
                if out_file.exists():
                    logger.info(f"  [{i+1}/{len(included)}] {sd.subject}/{sd.session} {metric} (exists)")
                    continue

                logger.info(f"  [{i+1}/{len(included)}] {sd.subject}/{sd.session} {metric}")
                try:
                    warp_metric_to_sigma(
                        metric_file=sd.metric_files[metric],
                        subject_data=sd,
                        config=config,
                        output_file=out_file
                    )
                except Exception as e:
                    logger.error(f"  Failed: {e}")
                    sd.included = False
                    sd.exclusion_reason = f"Warp failed: {e}"

    # Recompute included after potential failures
    included = [s for s in subjects_data if s.included]
    if not included:
        logger.error("No subjects remaining after Phase 2!")
        return {"success": False, "error": "All subjects failed in Phase 2"}

    # Phase 3: Create mean FA and WM mask
    logger.info("\n[Phase 3] Creating mean FA and WM mask...")

    # Build mean FA from included subjects (not from globbing, which may
    # pick up stale files from previous runs)
    fa_dir = output_dir / 'FA'
    fa_files = [
        fa_dir / f'{sd.subject}_{sd.session}_FA_sigma.nii.gz'
        for sd in included
    ]
    missing_fa = [f for f in fa_files if not f.exists()]
    if missing_fa:
        logger.error(f"Missing FA files for {len(missing_fa)} included subjects")
        return {"success": False, "error": f"Missing FA files: {missing_fa[:5]}"}

    if not fa_files:
        logger.error("No warped FA files found!")
        return {"success": False, "error": "No warped FA files"}

    # Stack and compute mean
    first_img = nib.load(fa_files[0])
    shape_3d = first_img.shape[:3]
    affine = first_img.affine

    fa_stack = np.zeros((*shape_3d, len(fa_files)), dtype=np.float32)
    for i, f in enumerate(fa_files):
        fa_stack[..., i] = nib.load(f).get_fdata()

    mean_fa_data = np.mean(fa_stack, axis=-1)
    mean_fa_file = stats_dir / 'mean_FA.nii.gz'
    nib.save(nib.Nifti1Image(mean_fa_data, affine), mean_fa_file)
    logger.info(f"Mean FA created from {len(fa_files)} subjects")

    # Create WM mask (purely tissue-informed, no FA gating)
    wm_mask_dir = output_dir / 'wm_mask'
    interior_mask, exterior_mask = create_wm_mask(
        config=config,
        output_dir=wm_mask_dir,
        reference_img=mean_fa_file,
        wm_prob_threshold=wm_prob_threshold,
        erosion_voxels=erosion_voxels
    )

    # Phase 4: Analysis mask = interior WM mask
    # The WM mask is based on SIGMA WM probability with boundary erosion,
    # making it modality-agnostic (usable for DTI, MSME, etc.).
    logger.info("\n[Phase 4] Creating analysis mask...")
    analysis_mask_data = (nib.load(interior_mask).get_fdata() > 0).astype(np.uint8)
    analysis_mask_file = stats_dir / 'analysis_mask.nii.gz'
    nib.save(nib.Nifti1Image(analysis_mask_data, affine), analysis_mask_file)
    n_mask_voxels = int(analysis_mask_data.sum())
    logger.info(f"Analysis mask: {n_mask_voxels} voxels (interior WM)")

    # Phase 5: Mask metrics with analysis mask
    logger.info("\n[Phase 5] Masking metrics...")

    for metric in metrics:
        metric_dir = output_dir / metric
        # Build file list from included subjects to match subject_list order
        metric_files = [
            metric_dir / f'{sd.subject}_{sd.session}_{metric}_sigma.nii.gz'
            for sd in included
        ]
        missing = [f for f in metric_files if not f.exists()]
        if missing:
            logger.warning(
                f"  {metric}: {len(missing)} missing files for included subjects"
            )

        # Create 4D volume from warped maps (in subject_list order)
        metric_4d_data = np.zeros((*shape_3d, len(metric_files)), dtype=np.float32)
        for i, f in enumerate(metric_files):
            metric_4d_data[..., i] = nib.load(f).get_fdata()

        # Apply analysis mask
        mask_4d = analysis_mask_data[:, :, :, np.newaxis]
        metric_4d_masked = metric_4d_data * mask_4d

        metric_4d_file = stats_dir / f'all_{metric}.nii.gz'
        nib.save(nib.Nifti1Image(metric_4d_masked, affine), metric_4d_file)
        logger.info(f"  {metric}: masked 4D volume ({metric_4d_masked.shape})")

    # Phase 6: Generate manifest
    logger.info("\n[Phase 6] Generating manifest...")
    manifest = generate_manifest(subjects_data, output_dir, metrics)

    logger.info("\n" + "=" * 80)
    logger.info("TBSS PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Subjects included: {manifest['subjects_included']}")
    logger.info(f"Subjects excluded: {manifest['subjects_excluded']}")
    logger.info(f"Metrics prepared: {metrics}")
    logger.info(f"Output: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Review subject_manifest.json")
    logger.info("  2. Run statistical analysis:")
    logger.info(f"     python -m neurofaune.analysis.tbss.run_tbss_stats \\")
    logger.info(f"         --tbss-dir {output_dir} \\")
    logger.info(f"         --design-dir /path/to/designs/ \\")
    logger.info(f"         --analysis-name my_analysis")

    return {
        "success": True,
        "manifest": manifest,
        "output_dir": str(output_dir),
        "analysis_mask": str(analysis_mask_file),
        "n_mask_voxels": n_mask_voxels,
        "n_subjects": len(included)
    }


def main():
    """Command-line interface for TBSS preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare DTI data for TBSS analysis using neurofaune registration chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all cohorts with default settings
  uv run python -m neurofaune.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --output-dir /study/analysis/tbss/

  # Use subject list from neuroaider design matrix
  uv run python -m neurofaune.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --output-dir /study/analysis/tbss/ \\
      --subject-list /study/designs/model1/subject_list.txt

  # Specific cohort with exclusions
  uv run python -m neurofaune.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --output-dir /study/analysis/tbss/ \\
      --cohorts p60 \\
      --exclude-file /study/qc/dwi_batch_summary/exclude_subjects.txt

  # Dry run to check subject discovery
  uv run python -m neurofaune.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --output-dir /study/analysis/tbss/ \\
      --dry-run
        """
    )

    parser.add_argument('--config', type=Path, required=True,
                        help='Path to config.yaml')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for TBSS analysis')
    parser.add_argument('--metrics', nargs='+', default=DTI_METRICS,
                        choices=DTI_METRICS,
                        help='Metrics to prepare (default: FA MD AD RD)')
    parser.add_argument('--cohorts', nargs='+', default=['p30', 'p60', 'p90'],
                        help='Cohorts to include (default: p30 p60 p90)')
    parser.add_argument('--subject-list', type=Path,
                        help='Path to subject list file (one per line, format: sub-Rat1_ses-p60)')
    parser.add_argument('--exclude-file', type=Path,
                        help='Path to exclusion list from batch QC')
    parser.add_argument('--wm-prob-threshold', type=float, default=0.3,
                        help='WM probability threshold (default: 0.3)')
    parser.add_argument('--erosion-voxels', type=int, default=2,
                        help='Erosion from brain boundary in voxels (default: 2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only discover subjects, do not process')
    parser.add_argument('--no-prewarped', action='store_true',
                        help='Use legacy warp mode instead of prewarped SIGMA files')

    args = parser.parse_args()

    # Load subject list if provided
    subjects = None
    if args.subject_list:
        if not args.subject_list.exists():
            parser.error(f"Subject list not found: {args.subject_list}")
        with open(args.subject_list) as f:
            subjects = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(subjects)} subjects from {args.subject_list}")

    config = load_config(args.config)

    result = prepare_tbss_data(
        config=config,
        output_dir=args.output_dir,
        metrics=args.metrics,
        cohorts=args.cohorts,
        subjects=subjects,
        exclude_file=args.exclude_file,
        wm_prob_threshold=args.wm_prob_threshold,
        erosion_voxels=args.erosion_voxels,
        dry_run=args.dry_run,
        use_prewarped=not args.no_prewarped
    )

    exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
