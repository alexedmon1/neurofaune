"""
Robust skull stripping for functional BOLD data using Atropos segmentation.

This module adapts the anatomical skull stripping approach (Atropos + BET + morphological
operations) for functional BOLD contrast.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from scipy import ndimage


def calculate_adaptive_bet_frac_bold(
    input_file: Path,
    atropos_mask: np.ndarray,
    default_frac: float = 0.25,
    min_frac: float = 0.1,
    max_frac: float = 0.4
) -> float:
    """
    Calculate adaptive BET fractional intensity parameter for BOLD data.

    Parameters
    ----------
    input_file : Path
        Input BOLD reference image (3D)
    atropos_mask : np.ndarray
        Rough brain mask from Atropos
    default_frac : float
        Default frac if calculation fails
    min_frac : float
        Minimum allowed frac
    max_frac : float
        Maximum allowed frac

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
    background_mask = (atropos_mask == 0) & (img_data > 0)
    if background_mask.sum() > 0:
        background_intensities = img_data[background_mask]
    else:
        print(f"  Warning: No background found, using default frac={default_frac}")
        return default_frac

    # Calculate statistics
    brain_median = np.median(brain_intensities)
    brain_std = np.std(brain_intensities)
    background_median = np.median(background_intensities)

    # Calculate contrast-to-noise ratio (CNR)
    # Higher CNR = easier to separate brain from background = can use higher frac
    cnr = abs(brain_median - background_median) / brain_std if brain_std > 0 else 1.0

    # Map CNR to frac parameter
    # Low CNR (< 2): Use lower frac (more conservative)
    # High CNR (> 4): Use higher frac (more aggressive)
    if cnr < 2.0:
        frac = max_frac  # Low contrast, be conservative (higher frac = smaller mask)
    elif cnr > 4.0:
        frac = min_frac  # High contrast, be aggressive (lower frac = larger mask)
    else:
        # Linear interpolation between min and max
        frac = max_frac - (cnr - 2.0) * (max_frac - min_frac) / 2.0

    frac = np.clip(frac, min_frac, max_frac)

    print(f"  Adaptive BET parameter calculation:")
    print(f"    Brain median intensity: {brain_median:.1f}")
    print(f"    Background median intensity: {background_median:.1f}")
    print(f"    Contrast-to-noise ratio (CNR): {cnr:.2f}")
    print(f"    -> Selected frac: {frac:.2f}")

    return frac


def skull_strip_bold_atropos(
    input_file: Path,
    output_file: Path,
    mask_file: Optional[Path] = None,
    use_bet_refinement: bool = False
) -> Tuple[Path, Path]:
    """
    Robust skull stripping for BOLD data using Atropos segmentation.

    This function adapts the anatomical skull stripping approach:
    1. Create foreground mask
    2. Run Atropos 5-class segmentation
    3. Exclude largest (background) and smallest (peripheral) posteriors
    4. Apply morphological closing (dilate 2x, erode 2x)
    5. [Optional] Refine with BET (usually not needed for BOLD)

    Parameters
    ----------
    input_file : Path
        Input BOLD reference volume (3D - mean or single TR)
    output_file : Path
        Output brain-extracted image
    mask_file : Path, optional
        Output brain mask file (default: output_file with _mask suffix)
    use_bet_refinement : bool
        Whether to use BET refinement (default: False, as BOLD has low contrast)

    Returns
    -------
    Tuple[Path, Path]
        (brain_file, mask_file)
    """
    from nipype.interfaces.ants import Atropos
    from nipype.interfaces import fsl
    import os

    if mask_file is None:
        mask_file = output_file.parent / output_file.name.replace('.nii.gz', '_mask.nii.gz')

    print("="*80)
    print("ROBUST BOLD SKULL STRIPPING (Atropos + BET + Morphology)")
    print("="*80)

    # Step 1: Create foreground mask
    print("\nStep 1: Creating foreground mask...")
    img = nib.load(input_file)
    img_data = img.get_fdata()

    # Simple thresholding: anything > 0 is foreground
    # This excludes background air but includes head/skull
    foreground_mask = img_data > 0

    foreground_mask_file = output_file.parent / 'foreground_mask.nii.gz'
    nib.save(
        nib.Nifti1Image(foreground_mask.astype(np.uint8), img.affine, img.header),
        foreground_mask_file
    )
    print(f"  Foreground mask: {foreground_mask.sum():,} voxels")

    # Step 2: Run Atropos segmentation
    print("\nStep 2: Running Atropos 5-class segmentation...")
    print("  This separates brain from skull/CSF/background")

    # Change to output directory for Atropos
    original_dir = os.getcwd()
    os.chdir(output_file.parent)

    try:
        atropos = Atropos()
        atropos.inputs.dimension = 3
        atropos.inputs.intensity_images = [str(input_file)]
        atropos.inputs.mask_image = str(foreground_mask_file)
        atropos.inputs.number_of_tissue_classes = 5
        atropos.inputs.n_iterations = 5
        atropos.inputs.convergence_threshold = 0.0
        atropos.inputs.mrf_smoothing_factor = 0.1
        atropos.inputs.mrf_radius = [1, 1, 1]
        atropos.inputs.initialization = 'KMeans'
        atropos.inputs.save_posteriors = True

        result = atropos.run()
        posteriors = [Path(p) for p in result.outputs.posteriors]

        # Step 3: Intelligent posterior classification for BOLD
        print("\nStep 3: Analyzing Atropos posteriors...")

        # Calculate volume AND mean intensity for each posterior
        # For BOLD data, we need to identify brain based on BOTH spatial and intensity patterns
        post_info = []
        for idx, post_file in enumerate(posteriors):
            post_data = nib.load(post_file).get_fdata()

            # Volume: number of voxels with probability > 0.5
            volume = (post_data > 0.5).sum()

            # Mean intensity in this tissue class (weighted by probability)
            mask = post_data > 0.5
            if mask.sum() > 0:
                mean_intensity = (img_data[mask] * post_data[mask]).sum() / post_data[mask].sum()
            else:
                mean_intensity = 0

            # Spatial distribution: center of mass (to identify peripheral vs central tissues)
            if mask.sum() > 0:
                coords = np.argwhere(mask)
                center_of_mass = coords.mean(axis=0)
                # Distance from image center
                img_center = np.array(img_data.shape) / 2
                dist_from_center = np.linalg.norm(center_of_mass - img_center)
            else:
                center_of_mass = np.zeros(3)
                dist_from_center = 0

            post_info.append({
                'idx': idx,
                'volume': volume,
                'mean_intensity': mean_intensity,
                'center_of_mass': center_of_mass,
                'dist_from_center': dist_from_center
            })

            print(f"  Posterior {idx+1}: {volume:,} voxels, "
                  f"mean intensity: {mean_intensity:.1f}, "
                  f"distance from center: {dist_from_center:.1f}")

        # Strategy for BOLD: Keep posteriors that are:
        # 1. NOT the largest volume (likely background air)
        # 2. NOT extremely peripheral (skull/scalp)
        # 3. Have reasonable BOLD signal intensity

        # Find largest posterior (background)
        largest_idx = max(post_info, key=lambda x: x['volume'])['idx']

        # Find most peripheral posterior (skull/scalp) - highest distance from center
        # But only among non-background posteriors
        non_background = [p for p in post_info if p['idx'] != largest_idx]
        if len(non_background) > 1:
            most_peripheral_idx = max(non_background, key=lambda x: x['dist_from_center'])['idx']
        else:
            most_peripheral_idx = None

        # Include all posteriors EXCEPT background and most peripheral
        brain_indices = [p['idx'] for p in post_info
                        if p['idx'] != largest_idx and p['idx'] != most_peripheral_idx]

        print(f"\n  Excluding Posterior {largest_idx+1} (largest volume = background)")
        if most_peripheral_idx is not None:
            print(f"  Excluding Posterior {most_peripheral_idx+1} (most peripheral = skull/scalp)")
        print(f"  Including: Posteriors {[i+1 for i in brain_indices]} as brain tissue")

        # Combine brain tissue posteriors
        atropos_mask = np.zeros(nib.load(posteriors[0]).shape)
        for i in brain_indices:
            post_img = nib.load(posteriors[i])
            atropos_mask += post_img.get_fdata() > 0.5

        atropos_mask = atropos_mask > 0

        # Step 4: Apply morphological closing directly to Atropos mask
        print("\nStep 4: Applying morphological closing (dilate 2x → erode 2x)...")
        print(f"  Before morphology: {atropos_mask.sum():,} voxels")

        # Closing: dilate then erode (fills holes, smooths boundaries)
        mask_data = ndimage.binary_dilation(atropos_mask, iterations=2)
        mask_data = ndimage.binary_erosion(mask_data, iterations=2)

        print(f"  After morphology: {mask_data.sum():,} voxels")

        # Save Atropos+morphology mask
        atropos_mask_file = output_file.parent / 'atropos_morphology_mask.nii.gz'
        nib.save(
            nib.Nifti1Image(atropos_mask.astype(np.uint8), img.affine, img.header),
            atropos_mask_file
        )

        # Optional BET refinement (usually not needed for BOLD)
        if use_bet_refinement:
            print("\nStep 5: [Optional] Refining with BET...")

            # Calculate center of gravity and adaptive frac
            brain_coords = np.argwhere(mask_data > 0)
            center_of_gravity = brain_coords.mean(axis=0)
            print(f"  Brain center of gravity: [{center_of_gravity[0]:.1f}, "
                  f"{center_of_gravity[1]:.1f}, {center_of_gravity[2]:.1f}]")

            adaptive_frac = calculate_adaptive_bet_frac_bold(input_file, mask_data)

            # Mask the input with Atropos mask
            masked_input = output_file.parent / 'atropos_masked_input.nii.gz'
            masked_data = img_data * mask_data
            nib.save(
                nib.Nifti1Image(masked_data, img.affine, img.header),
                masked_input
            )

            bet = fsl.BET()
            bet.inputs.in_file = str(masked_input)
            bet.inputs.out_file = str(output_file)
            bet.inputs.mask = True
            bet.inputs.frac = adaptive_frac
            bet.inputs.center = [
                int(center_of_gravity[0]),
                int(center_of_gravity[1]),
                int(center_of_gravity[2])
            ]

            bet_result = bet.run()
            bet_mask_file = Path(bet_result.outputs.mask_file)

            # Load BET-refined mask
            mask_img = nib.load(bet_mask_file)
            mask_data = mask_img.get_fdata().astype(bool)

            print(f"  After BET refinement: {mask_data.sum():,} voxels")

        # Save final mask
        nib.save(
            nib.Nifti1Image(mask_data.astype(np.uint8), img.affine, img.header),
            mask_file
        )

        # Apply mask to create brain-extracted image
        brain_data = img_data * mask_data
        nib.save(
            nib.Nifti1Image(brain_data, img.affine, img.header),
            output_file
        )

        print("\n" + "="*80)
        print("SKULL STRIPPING COMPLETE")
        print("="*80)
        print(f"  Brain: {output_file}")
        print(f"  Mask: {mask_file}")
        print(f"  Mask voxels: {mask_data.sum():,}")

        return output_file, mask_file

    finally:
        # Always restore original directory
        os.chdir(original_dir)
