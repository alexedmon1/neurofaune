#!/usr/bin/env python3
"""
Anatomical T2w preprocessing workflow.

This workflow implements:
1. Image validation and preprocessing
2. Skull stripping (two-pass: Atropos + BET)
3. Bias field correction (N4)
4. Intensity normalization

NOTE: Registration to study-specific template and SIGMA atlas will be added
separately in template building/registration modules.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import nibabel as nib
import numpy as np
from nipype.interfaces import fsl, ants
import subprocess

from neurofaune.utils.transforms import TransformRegistry
from neurofaune.preprocess.utils.validation import validate_image, print_validation_results
from neurofaune.preprocess.qc import get_subject_qc_dir


def check_and_scale_voxel_size(
    input_file: Path,
    output_file: Path,
    min_voxel_size: float = 1.0,
    scale_factor: float = 10.0
) -> Tuple[Path, bool, float]:
    """
    Check voxel sizes and scale if needed for FSL/ANTs compatibility.

    Rodent MRI often has sub-millimeter voxels (e.g., 0.1x0.1x0.8mm) which
    can cause issues with FSL and ANTs. This function scales the voxel sizes
    by a factor (default 10x) if any dimension is below threshold.

    Parameters
    ----------
    input_file : Path
        Input NIfTI file
    output_file : Path
        Output scaled file (only created if scaling needed)
    min_voxel_size : float
        Minimum voxel size threshold (mm)
    scale_factor : float
        Scaling factor to apply if needed

    Returns
    -------
    Tuple[Path, bool, float]
        (output_file, was_scaled, actual_scale_factor)
    """
    img = nib.load(input_file)
    header = img.header
    voxel_sizes = header.get_zooms()[:3]  # Get x, y, z voxel sizes

    print(f"Original voxel sizes: {voxel_sizes[0]:.3f} x {voxel_sizes[1]:.3f} x {voxel_sizes[2]:.3f} mm")

    # Check if any dimension is below threshold
    if any(v < min_voxel_size for v in voxel_sizes):
        print(f"Voxel size < {min_voxel_size}mm detected. Scaling by {scale_factor}x for FSL/ANTs compatibility...")

        # Create new affine with scaled voxel sizes
        affine = img.affine.copy()
        affine[:3, :3] = affine[:3, :3] * scale_factor

        # Create new image with scaled affine (data stays same)
        scaled_img = nib.Nifti1Image(img.get_fdata(), affine, img.header)

        # Update header voxel sizes
        scaled_header = scaled_img.header
        new_voxel_sizes = tuple(v * scale_factor for v in voxel_sizes)
        scaled_header.set_zooms(new_voxel_sizes)

        # Save scaled image
        nib.save(scaled_img, output_file)

        print(f"Scaled voxel sizes: {new_voxel_sizes[0]:.3f} x {new_voxel_sizes[1]:.3f} x {new_voxel_sizes[2]:.3f} mm")

        return output_file, True, scale_factor
    else:
        print(f"Voxel sizes OK (>= {min_voxel_size}mm). No scaling needed.")
        return input_file, False, 1.0


def extract_slices_from_volume(
    input_file: Path,
    slice_indices: list,
    output_file: Path
) -> Path:
    """
    Extract specific slices from a 3D volume and merge them.

    Parameters
    ----------
    input_file : Path
        Input NIfTI file
    slice_indices : list
        List of slice indices to extract (0-indexed)
    output_file : Path
        Output merged file

    Returns
    -------
    Path
        Path to merged output file
    """
    import tempfile
    from glob import glob
    from natsort import natsorted

    tmpdir = Path(tempfile.mkdtemp())
    tmp_slices = []

    # Extract each slice
    for idx in slice_indices:
        roi = fsl.ExtractROI()
        roi.inputs.in_file = str(input_file)
        roi.inputs.z_min = idx
        roi.inputs.z_size = 1
        roi.inputs.x_min = -1
        roi.inputs.x_size = -1
        roi.inputs.y_min = -1
        roi.inputs.y_size = -1
        roi.inputs.output_type = 'NIFTI_GZ'
        roi.inputs.roi_file = str(tmpdir / f'tmp_slice_{idx:04d}.nii.gz')
        roi.run()
        tmp_slices.append(str(tmpdir / f'tmp_slice_{idx:04d}.nii.gz'))

    # Merge slices
    merger = fsl.Merge()
    merger.inputs.in_files = natsorted(tmp_slices)
    merger.inputs.dimension = 'z'
    merger.inputs.output_type = 'NIFTI_GZ'
    merger.inputs.merged_file = str(output_file)
    merger.run()

    # Clean up
    for tmp_file in tmp_slices:
        Path(tmp_file).unlink()
    tmpdir.rmdir()

    return output_file


def calculate_adaptive_bet_frac(
    input_file: Path,
    atropos_mask: np.ndarray,
    default_frac: float = 0.25,
    min_frac: float = 0.1,
    max_frac: float = 0.4
) -> float:
    """
    Calculate adaptive BET fractional intensity parameter based on image statistics.

    The frac parameter controls where BET cuts between brain and non-brain.
    This function adapts frac based on:
    - Brain tissue intensity distribution
    - Contrast between brain and background
    - Contrast-to-noise ratio (CNR)

    Parameters
    ----------
    input_file : Path
        Input T2w image
    atropos_mask : np.ndarray
        Rough brain mask from Atropos
    default_frac : float
        Default frac if calculation fails
    min_frac : float
        Minimum allowed frac (more conservative)
    max_frac : float
        Maximum allowed frac (more aggressive)

    Returns
    -------
    float
        Calculated frac parameter
    """
    img = nib.load(input_file)
    img_data = img.get_fdata()

    # Get brain intensities from Atropos mask region
    brain_intensities = img_data[atropos_mask > 0]

    # Get background intensities (everything outside Atropos mask but > 0)
    # This includes skull, eyes, scalp, etc.
    background_mask = (atropos_mask == 0) & (img_data > 0)
    if background_mask.sum() > 0:
        background_intensities = img_data[background_mask]
    else:
        # Fallback if no background (shouldn't happen)
        print(f"  Warning: No background found, using default frac={default_frac}")
        return default_frac

    # Calculate statistics
    brain_median = np.median(brain_intensities)
    brain_std = np.std(brain_intensities)
    background_median = np.median(background_intensities)
    background_std = np.std(background_intensities)

    # Calculate contrast-to-noise ratio (CNR)
    # Higher CNR = easier to separate brain from skull = can use higher frac
    # Lower CNR = harder to separate = need lower frac (more conservative)
    cnr = abs(brain_median - background_median) / (brain_std + background_std + 1e-6)

    # Map CNR to frac based on empirical rodent T2w ranges:
    # CNR < 1.5: Poor contrast (similar intensities) -> frac 0.15-0.20 (very conservative)
    # CNR 1.5-3.0: Good contrast -> frac 0.25-0.30 (moderate)
    # CNR > 3.0: Excellent contrast -> frac 0.35-0.40 (still conservative)

    if cnr < 1.5:
        frac = 0.20  # Very conservative for low contrast
    elif cnr < 3.0:
        frac = 0.28  # Moderate for good contrast
    else:
        frac = 0.38  # More aggressive for excellent contrast

    # Clamp to allowed range
    frac = np.clip(frac, min_frac, max_frac)

    print(f"  Adaptive BET parameter calculation:")
    print(f"    Brain median intensity: {brain_median:.1f}")
    print(f"    Background median intensity: {background_median:.1f}")
    print(f"    Contrast-to-noise ratio (CNR): {cnr:.2f}")
    print(f"    -> Selected frac: {frac:.2f}")

    return frac


def skull_strip_rodent(
    input_file: Path,
    output_file: Path,
    cohort: str = 'p60',
    method: str = 'atropos',
    template_file: Optional[Path] = None,
    template_mask: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Skull strip rodent brain with age-specific parameters.

    Parameters
    ----------
    input_file : Path
        Input T2w image
    output_file : Path
        Output brain-extracted image
    template_file : Path
        Atlas template for ANTs brain extraction
    template_mask : Path
        Atlas brain mask for ANTs brain extraction
    cohort : str
        Age cohort ('p30', 'p60', 'p90')
    method : str
        Skull stripping method ('ants', 'atropos', or 'bet')

    Returns
    -------
    Tuple[Path, Path]
        (brain_file, mask_file)
    """
    if method == 'ants':
        # Use ANTs brain extraction with SIGMA template
        # This is more robust for cylindrical rodent brains
        from nipype.interfaces.ants import BrainExtraction

        brain_extract = BrainExtraction()
        brain_extract.inputs.dimension = 3
        brain_extract.inputs.anatomical_image = str(input_file)
        brain_extract.inputs.brain_template = str(template_file)
        brain_extract.inputs.brain_probability_mask = str(template_mask)
        brain_extract.inputs.out_prefix = str(output_file.parent / output_file.stem.replace('.nii', ''))

        # Keep existing image instead of warping (faster)
        brain_extract.inputs.keep_temporary_files = 0

        print(f"Running ANTs brain extraction...")
        print(f"  Template: {template_file.name}")
        print(f"  Mask: {template_mask.name}")

        result = brain_extract.run()

        # ANTs BrainExtraction outputs: <prefix>BrainExtractionBrain.nii.gz and <prefix>BrainExtractionMask.nii.gz
        brain_file = Path(result.outputs.BrainExtractionBrain)
        mask_file = Path(result.outputs.BrainExtractionMask)

        # Rename to expected output
        import shutil
        shutil.move(brain_file, output_file)
        expected_mask = output_file.parent / output_file.name.replace('.nii.gz', '_mask.nii.gz')
        shutil.move(mask_file, expected_mask)

        # ANTs BrainExtraction doesn't provide posteriors
        return output_file, expected_mask, None

    elif method == 'atropos':
        # Use Atropos 5-component segmentation to separate brain from skull
        from nipype.interfaces.ants import Atropos
        import os

        # Create a simple foreground mask from the subject image itself
        # This excludes background air but includes head/skull
        print(f"Creating foreground mask from subject image...")
        img = nib.load(input_file)
        img_data = img.get_fdata()

        # Use Otsu thresholding to separate foreground from background
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(img_data[img_data > 0])
        foreground_mask = img_data > (threshold * 0.3)  # Liberal threshold to include skull

        # Save foreground mask
        foreground_mask_file = output_file.parent / 'foreground_mask.nii.gz'
        nib.save(nib.Nifti1Image(foreground_mask.astype(np.uint8), img.affine, img.header),
                 foreground_mask_file)

        # Change to output directory so Atropos writes files there
        original_dir = os.getcwd()
        os.chdir(output_file.parent)

        try:
            atropos = Atropos()
            atropos.inputs.dimension = 3
            atropos.inputs.intensity_images = [str(input_file)]
            atropos.inputs.mask_image = str(foreground_mask_file)  # Use subject's own foreground mask
            atropos.inputs.number_of_tissue_classes = 5  # Background, CSF, GM, WM, Skull/other
            atropos.inputs.n_iterations = 5
            atropos.inputs.convergence_threshold = 0.0
            atropos.inputs.mrf_smoothing_factor = 0.1
            atropos.inputs.mrf_radius = [1, 1, 1]
            atropos.inputs.initialization = 'KMeans'
            atropos.inputs.save_posteriors = True

            print(f"Running Atropos 5-component segmentation...")
            print(f"  Working directory: {output_file.parent}")
            result = atropos.run()

            # DYNAMIC POSTERIOR CLASSIFICATION
            # Atropos with KMeans has non-deterministic ordering
            # Original code excluded both extremes: darkest (background) and brightest (CSF/eyes/scalp)
            # Exclude both LARGEST and SMALLEST posteriors by volume
            posteriors = [Path(p) for p in result.outputs.posteriors]

            print(f"\nAnalyzing Atropos posteriors...")

            # Calculate volume for each posterior
            post_volumes = []
            for idx, post_path in enumerate(posteriors):
                post_data = nib.load(post_path).get_fdata()
                volume = (post_data > 0.5).sum()
                post_volumes.append((idx, volume))
                print(f"  Posterior {idx+1}: {volume:,} voxels")

            # Exclude both largest and smallest (like original code excluded extremes)
            smallest_idx = min(post_volumes, key=lambda x: x[1])[0]
            largest_idx = max(post_volumes, key=lambda x: x[1])[0]
            brain_indices = [idx for idx, _ in post_volumes
                           if idx != smallest_idx and idx != largest_idx]

            print(f"\n  Excluding largest: Posterior {largest_idx+1} (likely background/CSF)")
            print(f"  Excluding smallest: Posterior {smallest_idx+1} (likely peripheral tissue)")
            print(f"  Including: Posteriors {[i+1 for i in brain_indices]}")

            # Load and combine brain tissue posteriors
            atropos_mask = np.zeros(nib.load(posteriors[0]).shape)
            for i in brain_indices:
                post_img = nib.load(posteriors[i])
                atropos_mask += post_img.get_fdata() > 0.5

            atropos_mask = atropos_mask > 0

            # Save Atropos mask for BET refinement
            atropos_mask_file = output_file.parent / 'atropos_rough_mask.nii.gz'
            img = nib.load(input_file)
            nib.save(nib.Nifti1Image(atropos_mask.astype(np.uint8), img.affine, img.header),
                    atropos_mask_file)

            # Calculate center of gravity of Atropos brain mask
            # This helps BET initialize its spherical model at the correct location
            brain_coords = np.argwhere(atropos_mask > 0)
            center_of_gravity = brain_coords.mean(axis=0)
            print(f"Atropos brain mask center of gravity: [{center_of_gravity[0]:.1f}, {center_of_gravity[1]:.1f}, {center_of_gravity[2]:.1f}]")

            # Calculate adaptive BET frac parameter based on image contrast
            adaptive_frac = calculate_adaptive_bet_frac(input_file, atropos_mask)

            # Refine with BET using Atropos mask as prior and adaptive frac
            print(f"Refining brain extraction with BET...")
            from nipype.interfaces import fsl

            bet = fsl.BET()
            bet.inputs.in_file = str(input_file)
            bet.inputs.out_file = str(output_file)
            bet.inputs.mask = True
            bet.inputs.frac = adaptive_frac  # Adaptive frac based on image contrast
            bet.inputs.center = [int(center_of_gravity[0]), int(center_of_gravity[1]), int(center_of_gravity[2])]

            # Apply Atropos mask by masking the input first
            masked_input = output_file.parent / 'atropos_masked_input.nii.gz'
            masked_data = img.get_fdata() * atropos_mask
            nib.save(nib.Nifti1Image(masked_data, img.affine, img.header), masked_input)

            bet.inputs.in_file = str(masked_input)
            bet_result = bet.run()

            # Get the refined mask
            mask_file = Path(bet_result.outputs.mask_file)

            # Apply morphological closing to mask (dilate → erode)
            # This fills small holes and removes external speckles
            from scipy import ndimage
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata().astype(bool)

            # Closing: dilate then erode (fills holes, smooths boundaries)
            mask_data = ndimage.binary_dilation(mask_data, iterations=2)
            mask_data = ndimage.binary_erosion(mask_data, iterations=2)

            # Save cleaned mask
            nib.save(nib.Nifti1Image(mask_data.astype(np.uint8), mask_img.affine, mask_img.header),
                    mask_file)

            # Return brain, mask, and atropos posteriors for tissue segmentation reuse
            return output_file, mask_file, posteriors
        finally:
            # Always change back to original directory
            os.chdir(original_dir)

    elif method == 'bet':
        # FSL BET - can be problematic for cylindrical rodent brains
        bet_params = {
            'p30': {'frac': 0.3, 'radius': 100},
            'p60': {'frac': 0.6, 'radius': 125},
            'p90': {'frac': 0.6, 'radius': 125}
        }

        params = bet_params.get(cohort, bet_params['p60'])

        bet = fsl.BET()
        bet.inputs.in_file = str(input_file)
        bet.inputs.out_file = str(output_file)
        bet.inputs.frac = params['frac']
        bet.inputs.robust = True
        bet.inputs.mask = True
        bet.inputs.output_type = 'NIFTI_GZ'
        result = bet.run()

        mask_file = output_file.parent / output_file.name.replace('.nii.gz', '_mask.nii.gz')

        # BET method doesn't provide posteriors
        return output_file, mask_file, None
    else:
        raise NotImplementedError(f"Method {method} not yet implemented")


def segment_brain_tissue(
    mask_file: Path,
    atropos_posteriors: List[Path],
    output_dir: Path,
    subject: str,
    session: str
) -> Dict[str, Path]:
    """
    Extract tissue probability maps from Atropos skull stripping posteriors.

    This function reuses the 5-component Atropos segmentation from skull stripping
    and applies the refined BET brain mask to extract GM, WM, CSF probability maps.

    The 5 Atropos posteriors from skull stripping are:
    - Index 0 (Posterior 1): Background/scalp (darkest)
    - Index 1 (Posterior 2): White matter (dark on T2w)
    - Index 2 (Posterior 3): Grey matter (intermediate)
    - Index 3 (Posterior 4): CSF within brain (bright)
    - Index 4 (Posterior 5): Very bright CSF/eyes/scalp (excluded)

    Parameters
    ----------
    mask_file : Path
        Refined brain mask from BET
    atropos_posteriors : List[Path]
        List of 5 posterior probability maps from Atropos skull stripping
    output_dir : Path
        Output directory for tissue maps
    subject : str
        Subject identifier
    session : str
        Session identifier

    Returns
    -------
    dict
        Dictionary with paths to:
        - 'segmentation': Hard segmentation (discrete labels)
        - 'gm_prob': Grey matter probability map
        - 'wm_prob': White matter probability map
        - 'csf_prob': CSF probability map
    """
    print("\n" + "="*60)
    print("TISSUE SEGMENTATION (GM, WM, CSF)")
    print("="*60)
    print("Extracting tissue maps from Atropos 5-component segmentation...")
    print("Using posteriors from skull stripping step (no redundant Atropos run)")

    # Load brain mask
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # Load brain tissue posteriors (indices 1-3) and apply brain mask
    # Index 1 = WM, Index 2 = GM, Index 3 = CSF
    wm_data = nib.load(atropos_posteriors[1]).get_fdata() * mask_data
    gm_data = nib.load(atropos_posteriors[2]).get_fdata() * mask_data
    csf_data = nib.load(atropos_posteriors[3]).get_fdata() * mask_data

    # Create hard segmentation (argmax of tissue probabilities)
    # Only assign tissue label if max probability exceeds threshold (reduces speckling)
    # Label 1 = WM, Label 2 = GM, Label 3 = CSF
    tissue_stack = np.stack([wm_data, gm_data, csf_data], axis=-1)
    max_prob = np.max(tissue_stack, axis=-1)
    segmentation_data = np.argmax(tissue_stack, axis=-1) + 1  # Add 1 so labels are 1,2,3
    segmentation_data = segmentation_data * mask_data  # Apply brain mask

    # Remove low-confidence assignments (reduces speckling)
    min_prob_threshold = 0.35  # Only assign tissue if confidence >= 35%
    segmentation_data[max_prob < min_prob_threshold] = 0

    # Save tissue probability maps
    wm_prob = output_dir / f'{subject}_{session}_label-WM_probseg.nii.gz'
    gm_prob = output_dir / f'{subject}_{session}_label-GM_probseg.nii.gz'
    csf_prob = output_dir / f'{subject}_{session}_label-CSF_probseg.nii.gz'
    seg_final = output_dir / f'{subject}_{session}_dseg.nii.gz'

    nib.save(nib.Nifti1Image(wm_data, mask_img.affine, mask_img.header), wm_prob)
    nib.save(nib.Nifti1Image(gm_data, mask_img.affine, mask_img.header), gm_prob)
    nib.save(nib.Nifti1Image(csf_data, mask_img.affine, mask_img.header), csf_prob)
    nib.save(nib.Nifti1Image(segmentation_data, mask_img.affine, mask_img.header), seg_final)

    print(f"\nTissue segmentation complete!")
    print(f"  Segmentation: {seg_final.name}")
    print(f"  GM probability: {gm_prob.name}")
    print(f"  WM probability: {wm_prob.name}")
    print(f"  CSF probability: {csf_prob.name}")

    return {
        'segmentation': seg_final,
        'gm_prob': gm_prob,
        'wm_prob': wm_prob,
        'csf_prob': csf_prob
    }


def bias_field_correction(
    input_file: Path,
    output_file: Path,
    mask_file: Optional[Path] = None
) -> Path:
    """
    Perform N4 bias field correction.

    Parameters
    ----------
    input_file : Path
        Input image
    output_file : Path
        Output bias-corrected image
    mask_file : Path, optional
        Brain mask

    Returns
    -------
    Path
        Bias-corrected image
    """
    n4 = ants.N4BiasFieldCorrection()
    n4.inputs.input_image = str(input_file)
    n4.inputs.output_image = str(output_file)
    if mask_file:
        n4.inputs.mask_image = str(mask_file)
    n4.inputs.dimension = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.shrink_factor = 3
    n4.inputs.convergence_threshold = 1e-6
    n4.run()

    return output_file


def register_to_atlas_ants(
    moving_image: Path,
    fixed_image: Path,
    output_prefix: Path,
    cohort: str = 'p60',
    registration_type: str = 'SyN'
) -> Dict[str, Path]:
    """
    Register subject to atlas using ANTs.

    Parameters
    ----------
    moving_image : Path
        Subject brain image
    fixed_image : Path
        Atlas template
    output_prefix : Path
        Prefix for output files
    cohort : str
        Age cohort for parameter selection
    registration_type : str
        'Rigid', 'Affine', or 'SyN'

    Returns
    -------
    Dict[str, Path]
        Dictionary with warped_image, inverse_warped_image,
        composite_transform, inverse_composite_transform
    """
    # Age-specific registration parameters
    # P30 needs more regularization due to size/shape differences
    reg_params = {
        'p30': {
            'smoothing_sigmas': [[3, 2, 1, 0], [2, 1, 0]],
            'shrink_factors': [[8, 4, 2, 1], [4, 2, 1]],
            'transform_parameters': [(0.1,), (0.1, 3.0, 0.0)]  # SyN regularization
        },
        'p60': {
            'smoothing_sigmas': [[3, 2, 1, 0], [2, 1, 0]],
            'shrink_factors': [[8, 4, 2, 1], [4, 2, 1]],
            'transform_parameters': [(0.1,), (0.1, 3.0, 0.0)]
        },
        'p90': {
            'smoothing_sigmas': [[3, 2, 1, 0], [2, 1, 0]],
            'shrink_factors': [[8, 4, 2, 1], [4, 2, 1]],
            'transform_parameters': [(0.1,), (0.1, 3.0, 0.0)]
        }
    }

    params = reg_params.get(cohort, reg_params['p60'])

    # ANTs Registration
    reg = ants.Registration()
    reg.inputs.fixed_image = str(fixed_image)
    reg.inputs.moving_image = str(moving_image)
    reg.inputs.output_transform_prefix = str(output_prefix)
    reg.inputs.output_warped_image = str(output_prefix) + '_Warped.nii.gz'
    reg.inputs.output_inverse_warped_image = str(output_prefix) + '_InverseWarped.nii.gz'

    # Initialize with center alignment to ensure overlap
    reg.inputs.initial_moving_transform_com = 1  # Align centers of mass

    # Metric
    reg.inputs.metric = ['MI', 'MI']
    reg.inputs.metric_weight = [1.0, 1.0]
    reg.inputs.radius_or_number_of_bins = [32, 32]

    # Transforms: Affine + SyN
    reg.inputs.transforms = ['Affine', 'SyN']
    reg.inputs.transform_parameters = params['transform_parameters']

    # Iterations (4 levels for both transforms)
    reg.inputs.number_of_iterations = [[1000, 500, 250, 100], [100, 70, 50, 20]]

    # Smoothing and shrinking (4 levels for both transforms)
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0]]
    reg.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1]]

    # Convergence
    reg.inputs.convergence_threshold = [1e-6, 1e-6]
    reg.inputs.convergence_window_size = [10, 10]

    # Interpolation
    reg.inputs.interpolation = 'Linear'
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.verbose = True

    print(f"Running ANTs registration...")
    print(f"  Moving image: {moving_image}")
    print(f"  Fixed image: {fixed_image}")
    print(f"  Output prefix: {output_prefix}")
    print()

    result = reg.run()

    # Debug: print what files were actually created
    import os
    prefix_dir = output_prefix.parent
    prefix_name = output_prefix.name
    print(f"\nLooking for output files in: {prefix_dir}")
    print(f"With prefix: {prefix_name}")
    matching_files = [f for f in os.listdir(prefix_dir) if prefix_name in f]
    print(f"Found files: {matching_files}")
    print()

    # Check what attributes are available
    print("Available outputs:")
    for attr in dir(result.outputs):
        if not attr.startswith('_'):
            try:
                val = getattr(result.outputs, attr)
                print(f"  {attr}: {val}")
            except:
                pass
    print()

    return {
        'warped_image': Path(result.outputs.warped_image),
        'inverse_warped_image': Path(result.outputs.inverse_warped_image),
        'composite_transform': Path(result.outputs.composite_transform),
        'inverse_composite_transform': Path(result.outputs.inverse_composite_transform)
    }


def run_anatomical_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    output_dir: Path,
    transform_registry: TransformRegistry,
    t2w_file: Optional[Path] = None,
    subject_dir: Optional[Path] = None,
    modality: str = 'anat',
    slice_range: Optional[Tuple[int, int]] = None,
    work_dir: Optional[Path] = None,
    prefer_orientation: str = 'axial'
) -> Dict[str, Any]:
    """
    Run anatomical T2w preprocessing workflow.

    This workflow:
    0. Selects best T2w scan if subject_dir provided (automatic selection)
    1. Image validation (voxel size, orientation, data quality)
    2. Bias field correction (N4)
    3. Skull stripping (two-pass: Atropos 5-component + BET refinement)
    4. Intensity normalization (×1000)
    5. Save preprocessed outputs

    NOTE: This workflow NO LONGER performs registration to SIGMA atlas.
    Registration will be done separately:
    - First to age-matched study template (T2w template)
    - Template auto-registers to SIGMA (for parcellation access)

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier (e.g., 'sub-Rat207')
    session : str
        Session identifier (e.g., 'ses-p30')
    output_dir : Path
        Study root directory
    transform_registry : TransformRegistry
        Transform registry for saving spatial transforms
    t2w_file : Path, optional
        Input T2w image (if not provided, will auto-select from subject_dir)
    subject_dir : Path, optional
        Subject directory for automatic T2w selection
        (e.g., /path/to/bids/sub-Rat207)
    modality : str
        Modality identifier for slice-specific registration
    slice_range : Tuple[int, int], optional
        (start, end) slice indices for extraction
    work_dir : Path, optional
        Working directory
    prefer_orientation : str
        Preferred scan orientation for automatic selection ('axial', 'coronal', 'sagittal')

    Returns
    -------
    dict
        Dictionary with output file paths and QC reports

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> from neurofaune.utils.transforms import create_transform_registry
    >>>
    >>> config = load_config(Path('config.yaml'))
    >>> registry = create_transform_registry(
    ...     config, 'sub-Rat207', cohort='p30'
    ... )
    >>>
    >>> # Option 1: Specify exact T2w file
    >>> results = run_anatomical_preprocessing(
    ...     config=config,
    ...     subject='sub-Rat207',
    ...     session='ses-p30',
    ...     t2w_file=Path('sub-Rat207_ses-p30_T2w.nii.gz'),
    ...     output_dir=Path('/mnt/arborea/bpa-rat'),
    ...     transform_registry=registry,
    ...     slice_range=(0, 11)  # 11 slices for DTI
    ... )
    >>>
    >>> # Option 2: Automatic selection from subject directory
    >>> results = run_anatomical_preprocessing(
    ...     config=config,
    ...     subject='sub-Rat207',
    ...     session='ses-p30',
    ...     subject_dir=Path('/mnt/arborea/bpa-rat/raw/bids/sub-Rat207'),
    ...     output_dir=Path('/mnt/arborea/bpa-rat'),
    ...     transform_registry=registry
    ... )
    """
    # Check for existing exclusion marker
    from neurofaune.utils.exclusion import check_exclusion_marker

    marker_exists, marker_data = check_exclusion_marker(subject, session, output_dir)
    if marker_exists:
        print(f"\n{'='*60}")
        print(f"⚠ SUBJECT EXCLUDED - Preprocessing previously failed")
        print(f"{'='*60}")
        print(f"Subject: {subject}")
        print(f"Session: {session}")
        print(f"Reason: {marker_data.get('reason', 'Unknown')}")
        if 'timestamp' in marker_data:
            print(f"Failed at: {marker_data['timestamp']}")
        print(f"\nSkipping preprocessing for this subject/session.")
        print(f"To retry, use: neurofaune.utils.exclusion.remove_exclusion_marker()")
        print(f"{'='*60}\n")
        return {
            'status': 'excluded',
            'reason': marker_data.get('reason', 'Unknown'),
            'marker_file': output_dir / 'derivatives' / subject / session / 'anat' / '.preprocessing_failed'
        }

    # Step 0: Select best T2w scan if subject_dir provided
    if t2w_file is None and subject_dir is None:
        raise ValueError("Either t2w_file or subject_dir must be provided")

    if t2w_file is None:
        # Automatic selection
        from neurofaune.utils.select_anatomical import select_best_anatomical

        print(f"Auto-selecting best T2w scan for {subject} {session}...")
        best_scan = select_best_anatomical(
            subject_dir=subject_dir,
            session=session,
            modality=modality,
            prefer_orientation=prefer_orientation
        )

        if best_scan is None:
            raise FileNotFoundError(
                f"No suitable T2w scans found in {subject_dir}/{session}/{modality}"
            )

        t2w_file = best_scan['nifti']
        print(f"Selected: {t2w_file.name}")
        print(f"  Score: {best_scan['score']:.2f}")
        print(f"  Scan: {best_scan['info']['scan_name']}")
        print(f"  Orientation: {best_scan['info']['orientation']}")
        print(f"  Slices: {best_scan['info']['slices']}")
        print()

    # Step 0: Validate input image
    print("="*60)
    print("STEP 0: Validating Input Image")
    print("="*60)
    validation_result = validate_image(
        img_file=t2w_file,
        modality='anat',
        min_voxel_size=0.05,
        max_voxel_size=2.0,
        check_orientation=True,
        strict=False
    )
    print_validation_results(validation_result, name=f"{subject} {session} T2w")

    # Check validation status
    if not validation_result['valid']:
        error_msg = f"Image validation failed for {t2w_file}:\n"
        error_msg += "\n".join(validation_result['errors'])
        raise ValueError(error_msg)

    if validation_result['warnings']:
        print(f"⚠ Found {len(validation_result['warnings'])} warnings during validation.")
        print("Proceeding with preprocessing, but review warnings above.\n")

    # Setup directories
    derivatives_dir = output_dir / 'derivatives' / subject / session / modality
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = get_subject_qc_dir(output_dir, subject, session, modality)

    if work_dir is None:
        work_dir = output_dir / 'work' / subject / session / 'anat_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Extract cohort from session
    cohort = session.split('-')[1] if 'ses-' in session else 'p60'

    # Step 1: Check and scale voxel sizes if needed
    print("Checking voxel sizes...")
    t2w_scaled_file = work_dir / f'{subject}_{session}_T2w_scaled.nii.gz'
    t2w_processed, was_scaled, scale_factor = check_and_scale_voxel_size(
        t2w_file, t2w_scaled_file, min_voxel_size=1.0, scale_factor=10.0
    )

    # Load atlas
    from neurofaune.atlas import AtlasManager
    atlas_mgr = AtlasManager(config)

    # Get atlas template and mask for skull stripping
    atlas_template_img = atlas_mgr.get_template(modality=None, masked=False, coronal=True)

    # Create brain mask by combining GM and WM (excluding CSF)
    gm_mask_img = atlas_mgr.get_tissue_mask('gm', probability=False)
    wm_mask_img = atlas_mgr.get_tissue_mask('wm', probability=False)

    # Combine masks
    brain_mask_data = (gm_mask_img.get_fdata() > 0) | (wm_mask_img.get_fdata() > 0)
    atlas_mask_img = nib.Nifti1Image(brain_mask_data.astype(np.uint8), gm_mask_img.affine, gm_mask_img.header)

    # Save to work directory
    atlas_template_file = work_dir / 'SIGMA_template_unmasked.nii.gz'
    atlas_mask_file = work_dir / 'SIGMA_brain_mask.nii.gz'
    nib.save(atlas_template_img, atlas_template_file)
    nib.save(atlas_mask_img, atlas_mask_file)

    # Scale atlas if subject was scaled
    if was_scaled:
        atlas_template_scaled = work_dir / 'SIGMA_template_unmasked_scaled.nii.gz'
        atlas_mask_scaled = work_dir / 'SIGMA_brain_mask_scaled.nii.gz'
        atlas_template_file, _, _ = check_and_scale_voxel_size(
            atlas_template_file, atlas_template_scaled, min_voxel_size=1.0, scale_factor=scale_factor
        )
        atlas_mask_file, _, _ = check_and_scale_voxel_size(
            atlas_mask_file, atlas_mask_scaled, min_voxel_size=1.0, scale_factor=scale_factor
        )

    # NOTE: Atlas registration removed - will be done separately in template registration module

    # Step 1: Extract slices from subject if needed
    if slice_range:
        print(f"Extracting slices {slice_range[0]}-{slice_range[1]} from subject T2w...")
        t2w_sliced = work_dir / f'{subject}_{session}_T2w_sliced.nii.gz'
        slice_indices = list(range(slice_range[0], slice_range[1]))
        extract_slices_from_volume(t2w_processed, slice_indices, t2w_sliced)
        t2w_input = t2w_sliced
    else:
        t2w_input = t2w_processed

    # Step 2: Bias field correction (before skull stripping!)
    print("Bias field correction...")
    t2w_n4_file = work_dir / f'{subject}_{session}_T2w_n4.nii.gz'
    bias_field_correction(t2w_input, t2w_n4_file, mask_file=None)

    # Step 3: Skull stripping (after bias correction)
    print(f"Skull stripping with cohort={cohort} using Atropos 5-component segmentation...")
    brain_file = work_dir / f'{subject}_{session}_brain.nii.gz'
    brain_file, mask_file, atropos_posteriors = skull_strip_rodent(
        input_file=t2w_n4_file,
        output_file=brain_file,
        cohort=cohort,
        method='atropos'
    )

    # Check if posteriors are available
    if atropos_posteriors is None:
        raise ValueError(
            "Atropos posteriors not available. "
            "Tissue segmentation currently requires method='atropos' for skull stripping."
        )

    # Step 4: Tissue segmentation (GM, WM, CSF)
    print("\n" + "="*60)
    print("STEP 4: Tissue Segmentation")
    print("="*60)

    tissue_results = segment_brain_tissue(
        mask_file=mask_file,
        atropos_posteriors=atropos_posteriors,
        output_dir=work_dir,
        subject=subject,
        session=session
    )

    # Step 5: Intensity normalization (multiply by 1000)
    print("\n" + "="*60)
    print("STEP 5: Intensity Normalization")
    print("="*60)
    print("Normalizing intensity...")
    img = nib.load(brain_file)
    data = img.get_fdata()
    data_norm = data * 1000.0
    img_norm = nib.Nifti1Image(data_norm, img.affine, img.header)
    brain_norm_file = work_dir / f'{subject}_{session}_brain_norm.nii.gz'
    nib.save(img_norm, brain_norm_file)
    print(f"  Saved normalized brain: {brain_norm_file.name}")

    # Step 6: Save outputs to derivatives
    print("\n" + "="*60)
    print("STEP 6: Saving Outputs")
    print("="*60)
    print("Copying files to derivatives directory...")

    final_brain_skullstrip = derivatives_dir / f'{subject}_{session}_desc-skullstrip_T2w.nii.gz'
    final_brain = derivatives_dir / f'{subject}_{session}_desc-preproc_T2w.nii.gz'
    final_mask = derivatives_dir / f'{subject}_{session}_desc-brain_mask.nii.gz'
    final_seg = derivatives_dir / f'{subject}_{session}_dseg.nii.gz'
    final_gm = derivatives_dir / f'{subject}_{session}_label-GM_probseg.nii.gz'
    final_wm = derivatives_dir / f'{subject}_{session}_label-WM_probseg.nii.gz'
    final_csf = derivatives_dir / f'{subject}_{session}_label-CSF_probseg.nii.gz'

    import shutil
    shutil.copy(brain_file, final_brain_skullstrip)
    shutil.copy(brain_norm_file, final_brain)
    shutil.copy(mask_file, final_mask)
    shutil.copy(tissue_results['segmentation'], final_seg)
    shutil.copy(tissue_results['gm_prob'], final_gm)
    shutil.copy(tissue_results['wm_prob'], final_wm)
    shutil.copy(tissue_results['csf_prob'], final_csf)

    print(f"  ✓ Skull-stripped T2w: {final_brain_skullstrip.name}")
    print(f"  ✓ Preprocessed T2w: {final_brain.name}")
    print(f"  ✓ Brain mask: {final_mask.name}")
    print(f"  ✓ Tissue segmentation: {final_seg.name}")
    print(f"  ✓ GM probability: {final_gm.name}")
    print(f"  ✓ WM probability: {final_wm.name}")
    print(f"  ✓ CSF probability: {final_csf.name}")

    # Step 7: Generate QC
    print("\n" + "="*60)
    print("STEP 7: Quality Control")
    print("="*60)
    from neurofaune.preprocess.qc.anat import generate_anatomical_qc_report

    qc_report = generate_anatomical_qc_report(
        subject=subject,
        session=session,
        t2w_file=t2w_input,
        brain_file=final_brain_skullstrip,
        mask_file=final_mask,
        gm_file=final_gm,
        wm_file=final_wm,
        csf_file=final_csf,
        output_dir=qc_dir
    )
    print(f"  ✓ QC report: {qc_report}")

    print("\n" + "="*80)
    print("Anatomical preprocessing complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Preprocessed T2w: {final_brain}")
    print(f"  Brain mask: {final_mask}")
    print(f"  Tissue segmentation: {final_seg}")
    print(f"  GM/WM/CSF probabilities: {derivatives_dir}")
    print("\nNOTE: Registration to study template and SIGMA will be done separately.")
    print("="*80 + "\n")

    return {
        'brain': final_brain,
        'mask': final_mask,
        'segmentation': final_seg,
        'gm_prob': final_gm,
        'wm_prob': final_wm,
        'csf_prob': final_csf,
        'was_scaled': was_scaled,
        'scale_factor': scale_factor,
        'qc_reports': [qc_report]
    }
