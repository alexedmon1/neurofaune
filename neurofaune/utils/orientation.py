"""
Orientation verification and correction utilities.

This module provides functions to detect and correct orientation mismatches
between images before registration. This is critical for rodent MRI where
different acquisition protocols may result in different data orientations
even when headers report the same orientation code.
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np


def get_intensity_distribution(data: np.ndarray, axis: int) -> Tuple[float, float]:
    """
    Compute intensity distribution along an axis.

    Returns the ratio of intensity in the first vs second half,
    which can indicate anatomical orientation.

    Parameters
    ----------
    data : np.ndarray
        3D image data
    axis : int
        Axis to analyze (0=X, 1=Y, 2=Z)

    Returns
    -------
    tuple
        (first_half_sum, second_half_sum)
    """
    mid = data.shape[axis] // 2
    first_half = np.take(data, range(0, mid), axis=axis).sum()
    second_half = np.take(data, range(mid, data.shape[axis]), axis=axis).sum()
    return first_half, second_half


def detect_anterior_posterior_flip(
    moving_img: nib.Nifti1Image,
    reference_img: nib.Nifti1Image,
    threshold: float = 0.3
) -> bool:
    """
    Detect if the moving image is flipped in the anterior-posterior axis.

    Uses the fact that the anterior brain (olfactory bulb region) typically
    has more intensity than the posterior (cerebellum) in T2w images due to
    the larger tissue volume anteriorly.

    Parameters
    ----------
    moving_img : Nifti1Image
        Image to check
    reference_img : Nifti1Image
        Reference image with known correct orientation
    threshold : float
        Minimum ratio difference to consider a flip (default 0.3 = 30%)

    Returns
    -------
    bool
        True if flip is detected, False otherwise
    """
    moving_data = moving_img.get_fdata()
    ref_data = reference_img.get_fdata()

    # For Y axis (typically A-P in RAS)
    # In correct orientation: anterior (high Y) should have more intensity
    mov_first, mov_second = get_intensity_distribution(moving_data, axis=1)
    ref_first, ref_second = get_intensity_distribution(ref_data, axis=1)

    # Compute ratios (positive = more in second half)
    mov_ratio = (mov_second - mov_first) / (mov_second + mov_first)
    ref_ratio = (ref_second - ref_first) / (ref_second + ref_first)

    # If ratios have opposite signs and difference is significant, it's flipped
    flip_detected = (mov_ratio * ref_ratio < 0) and (abs(mov_ratio - ref_ratio) > threshold)

    return flip_detected


def compute_orientation_metrics(
    moving_img: nib.Nifti1Image,
    reference_img: nib.Nifti1Image
) -> Dict[str, any]:
    """
    Compute orientation metrics between two images.

    Parameters
    ----------
    moving_img : Nifti1Image
        Image to analyze
    reference_img : Nifti1Image
        Reference image

    Returns
    -------
    dict
        Dictionary with orientation metrics
    """
    moving_data = moving_img.get_fdata()
    ref_data = reference_img.get_fdata()

    metrics = {
        'moving_shape': moving_data.shape,
        'reference_shape': ref_data.shape,
        'moving_orientation': nib.aff2axcodes(moving_img.affine),
        'reference_orientation': nib.aff2axcodes(reference_img.affine),
    }

    # Intensity distribution along each axis
    for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
        mov_first, mov_second = get_intensity_distribution(moving_data, axis)
        ref_first, ref_second = get_intensity_distribution(ref_data, axis)

        mov_ratio = (mov_second - mov_first) / (mov_second + mov_first + 1e-10)
        ref_ratio = (ref_second - ref_first) / (ref_second + ref_first + 1e-10)

        metrics[f'{name}_moving_ratio'] = mov_ratio
        metrics[f'{name}_reference_ratio'] = ref_ratio
        metrics[f'{name}_flip_indicated'] = (mov_ratio * ref_ratio < 0) and abs(mov_ratio - ref_ratio) > 0.2

    return metrics


def verify_orientation(
    moving_path: Path,
    reference_path: Path,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify orientation compatibility between moving and reference images.

    Parameters
    ----------
    moving_path : Path
        Path to moving image
    reference_path : Path
        Path to reference image
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Dictionary with:
        - 'compatible': bool - True if orientations are compatible
        - 'flips_needed': list - Axes that need flipping ['x', 'y', 'z']
        - 'metrics': dict - Detailed orientation metrics
    """
    moving_img = nib.load(moving_path)
    ref_img = nib.load(reference_path)

    metrics = compute_orientation_metrics(moving_img, ref_img)

    flips_needed = []

    # Check each axis
    if metrics.get('Y_flip_indicated', False):
        flips_needed.append('y')
    if metrics.get('X_flip_indicated', False):
        flips_needed.append('x')
    if metrics.get('Z_flip_indicated', False):
        flips_needed.append('z')

    compatible = len(flips_needed) == 0

    if verbose:
        print(f"Orientation Verification")
        print(f"=" * 50)
        print(f"Moving:    {moving_path.name}")
        print(f"Reference: {reference_path.name}")
        print(f"")
        print(f"Header orientations:")
        print(f"  Moving:    {metrics['moving_orientation']}")
        print(f"  Reference: {metrics['reference_orientation']}")
        print(f"")
        print(f"Intensity distribution analysis:")
        for axis in ['X', 'Y', 'Z']:
            mov_ratio = metrics[f'{axis}_moving_ratio']
            ref_ratio = metrics[f'{axis}_reference_ratio']
            flip = metrics[f'{axis}_flip_indicated']
            status = "⚠️  FLIP NEEDED" if flip else "✓"
            print(f"  {axis}: moving={mov_ratio:+.2f}, ref={ref_ratio:+.2f} {status}")
        print(f"")
        if compatible:
            print(f"✓ Orientations are compatible")
        else:
            print(f"⚠️  Orientation mismatch detected!")
            print(f"   Flips needed: {flips_needed}")

    return {
        'compatible': compatible,
        'flips_needed': flips_needed,
        'metrics': metrics
    }


def flip_image(
    input_path: Path,
    output_path: Path,
    axes: list
) -> Path:
    """
    Flip image along specified axes.

    Parameters
    ----------
    input_path : Path
        Input image path
    output_path : Path
        Output image path
    axes : list
        List of axes to flip ('x', 'y', 'z')

    Returns
    -------
    Path
        Path to flipped image
    """
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine.copy()

    for axis in axes:
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

        # Flip the data
        data = np.flip(data, axis=axis_idx)

        # Update the affine to reflect the flip
        # Negate the corresponding column and adjust origin
        affine[:3, axis_idx] = -affine[:3, axis_idx]
        affine[:3, 3] = affine[:3, 3] - affine[:3, axis_idx] * (data.shape[axis_idx] - 1)

    # Save flipped image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flipped_img = nib.Nifti1Image(data, affine, img.header)
    nib.save(flipped_img, output_path)

    print(f"Flipped image along {axes}: {output_path}")

    return output_path


def detect_flip_by_correlation(
    moving_img: nib.Nifti1Image,
    reference_img: nib.Nifti1Image,
    resample_size: int = 32
) -> Dict[str, any]:
    """
    Detect required flips by testing correlations at different flip configurations.

    Resamples both images to a common low-resolution grid and tests all 8
    possible flip combinations to find the best match.

    Parameters
    ----------
    moving_img : Nifti1Image
        Moving image
    reference_img : Nifti1Image
        Reference image
    resample_size : int
        Size to resample to for quick correlation (default 32)

    Returns
    -------
    dict
        Dictionary with best flip configuration and correlations
    """
    from scipy.ndimage import zoom

    moving_data = moving_img.get_fdata()
    ref_data = reference_img.get_fdata()

    # Resample both to same low-res grid for quick comparison
    mov_zoom = [resample_size / s for s in moving_data.shape[:3]]
    ref_zoom = [resample_size / s for s in ref_data.shape[:3]]

    # Handle 4D images
    if moving_data.ndim > 3:
        moving_data = moving_data[..., 0]
    if ref_data.ndim > 3:
        ref_data = ref_data[..., 0]

    mov_resampled = zoom(moving_data, mov_zoom, order=1)
    ref_resampled = zoom(ref_data, ref_zoom, order=1)

    # Normalize
    mov_resampled = (mov_resampled - mov_resampled.mean()) / (mov_resampled.std() + 1e-10)
    ref_resampled = (ref_resampled - ref_resampled.mean()) / (ref_resampled.std() + 1e-10)

    # Test all 8 flip combinations
    flip_configs = [
        [],
        ['x'],
        ['y'],
        ['z'],
        ['x', 'y'],
        ['x', 'z'],
        ['y', 'z'],
        ['x', 'y', 'z']
    ]

    results = []
    for flips in flip_configs:
        test_data = mov_resampled.copy()
        for axis in flips:
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            test_data = np.flip(test_data, axis=axis_idx)

        # Compute correlation
        corr = np.corrcoef(test_data.flatten(), ref_resampled.flatten())[0, 1]
        results.append({
            'flips': flips,
            'correlation': corr
        })

    # Sort by correlation
    results.sort(key=lambda x: x['correlation'], reverse=True)

    best = results[0]
    second = results[1] if len(results) > 1 else None

    return {
        'best_flips': best['flips'],
        'best_correlation': best['correlation'],
        'second_best_flips': second['flips'] if second else None,
        'second_best_correlation': second['correlation'] if second else None,
        'all_results': results,
        'confidence': best['correlation'] - (second['correlation'] if second else 0)
    }


def verify_orientation_correlation(
    moving_path: Path,
    reference_path: Path,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify orientation using correlation-based flip detection.

    More robust than intensity distribution method for images with
    different slice coverage or acquisition parameters.

    Parameters
    ----------
    moving_path : Path
        Path to moving image
    reference_path : Path
        Path to reference image
    verbose : bool
        Print detailed output

    Returns
    -------
    dict
        Dictionary with orientation analysis results
    """
    moving_img = nib.load(moving_path)
    ref_img = nib.load(reference_path)

    corr_result = detect_flip_by_correlation(moving_img, ref_img)

    compatible = len(corr_result['best_flips']) == 0
    flips_needed = corr_result['best_flips']

    if verbose:
        print(f"Orientation Verification (Correlation-Based)")
        print(f"=" * 50)
        print(f"Moving:    {moving_path.name}")
        print(f"Reference: {reference_path.name}")
        print(f"")
        print(f"Header orientations:")
        print(f"  Moving:    {nib.aff2axcodes(moving_img.affine)}")
        print(f"  Reference: {nib.aff2axcodes(ref_img.affine)}")
        print(f"")
        print(f"Correlation analysis (top 4 flip configurations):")
        for i, r in enumerate(corr_result['all_results'][:4]):
            flips_str = str(r['flips']) if r['flips'] else 'none'
            marker = " ← BEST" if i == 0 else ""
            print(f"  {flips_str:20s}: r={r['correlation']:.3f}{marker}")
        print(f"")
        print(f"Confidence: {corr_result['confidence']:.3f}")
        print(f"")
        if compatible:
            print(f"✓ Orientations are compatible (no flips needed)")
        else:
            print(f"⚠️  Orientation mismatch detected!")
            print(f"   Flips needed: {flips_needed}")

    return {
        'compatible': compatible,
        'flips_needed': flips_needed,
        'correlation': corr_result['best_correlation'],
        'confidence': corr_result['confidence'],
        'all_results': corr_result['all_results']
    }


def generate_orientation_comparison(
    moving_path: Path,
    reference_path: Path,
    output_path: Optional[Path] = None,
    test_flips: bool = True
) -> Path:
    """
    Generate visual comparison of orientations with optional flip testing.

    Creates a figure showing the moving image, reference image, and
    optionally the moving image with various flips applied for comparison.

    Parameters
    ----------
    moving_path : Path
        Path to moving image
    reference_path : Path
        Path to reference image
    output_path : Path, optional
        Output path for figure. If None, saves next to moving image.
    test_flips : bool
        If True, also show Y-flipped version for comparison

    Returns
    -------
    Path
        Path to saved figure
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    moving_img = nib.load(moving_path)
    ref_img = nib.load(reference_path)

    moving_data = moving_img.get_fdata()
    ref_data = ref_img.get_fdata()

    # Handle 4D
    if moving_data.ndim > 3:
        moving_data = moving_data[..., 0]
    if ref_data.ndim > 3:
        ref_data = ref_data[..., 0]

    # Find middle slices with data
    def find_slice_with_max_data(data, axis):
        sums = data.sum(axis=tuple(i for i in range(3) if i != axis))
        return np.argmax(sums)

    ref_z = find_slice_with_max_data(ref_data, 2)
    mov_z = find_slice_with_max_data(moving_data, 2)

    if test_flips:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.reshape(1, -1)

    # Reference axial
    ax = axes[0, 0]
    ax.imshow(ref_data[:, :, ref_z].T, cmap='gray', origin='lower')
    ax.set_title(f'Reference: {reference_path.name}\nAxial z={ref_z}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Anterior should be HIGH)')
    ax.text(5, ref_data.shape[1]-15, 'HIGH Y', color='yellow', fontsize=9)
    ax.text(5, 10, 'LOW Y', color='yellow', fontsize=9)

    # Moving axial (original)
    ax = axes[0, 1]
    ax.imshow(moving_data[:, :, mov_z].T, cmap='gray', origin='lower')
    ax.set_title(f'Moving (original): {moving_path.name}\nAxial z={mov_z}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Overlay (original)
    ax = axes[0, 2]
    ref_slice = ref_data[:, :, ref_z]
    mov_slice = moving_data[:, :, mov_z]
    zoom_factor = [ref_slice.shape[0]/mov_slice.shape[0], ref_slice.shape[1]/mov_slice.shape[1]]
    mov_resized = zoom(mov_slice, zoom_factor, order=1)
    ref_norm = ref_slice / (ref_slice.max() + 1e-10)
    mov_norm = mov_resized / (mov_resized.max() + 1e-10)
    ax.imshow(ref_norm.T, cmap='Reds', alpha=0.6, origin='lower')
    ax.imshow(mov_norm.T, cmap='Blues', alpha=0.5, origin='lower')
    ax.set_title('Overlay (original)\nRed=Reference, Blue=Moving')
    ax.axis('off')

    if test_flips:
        # Y-flipped moving
        mov_flipped = np.flip(moving_data, axis=1)

        ax = axes[1, 0]
        ax.imshow(mov_flipped[:, :, mov_z].T, cmap='gray', origin='lower')
        ax.set_title('Moving FLIPPED in Y')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Overlay (Y-flipped)
        ax = axes[1, 1]
        mov_flip_slice = mov_flipped[:, :, mov_z]
        mov_flip_resized = zoom(mov_flip_slice, zoom_factor, order=1)
        mov_flip_norm = mov_flip_resized / (mov_flip_resized.max() + 1e-10)
        ax.imshow(ref_norm.T, cmap='Reds', alpha=0.6, origin='lower')
        ax.imshow(mov_flip_norm.T, cmap='Blues', alpha=0.5, origin='lower')
        ax.set_title('Overlay (Y-flipped)\nRed=Reference, Blue=Moving')
        ax.axis('off')

        # Compute correlations
        corr_orig = np.corrcoef(ref_norm.flatten(), mov_norm.flatten())[0, 1]
        corr_flip = np.corrcoef(ref_norm.flatten(), mov_flip_norm.flatten())[0, 1]

        ax = axes[1, 2]
        ax.text(0.5, 0.6, f'Original correlation: {corr_orig:.3f}',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.4, f'Y-flipped correlation: {corr_flip:.3f}',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        recommendation = "Y-FLIP RECOMMENDED" if corr_flip > corr_orig + 0.05 else "No flip needed"
        color = 'red' if 'FLIP' in recommendation else 'green'
        ax.text(0.5, 0.2, recommendation, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color, transform=ax.transAxes)
        ax.axis('off')

    plt.suptitle('Orientation Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = moving_path.parent / f'{moving_path.stem}_orientation_check.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved orientation comparison: {output_path}")

    return output_path


def reorient_to_reference(
    moving_path: Path,
    reference_path: Path,
    output_path: Optional[Path] = None,
    flips: Optional[list] = None,
    auto_detect: bool = False
) -> Tuple[Path, Dict]:
    """
    Reorient moving image to match reference orientation.

    Due to the structural differences between rodent brain images
    (different slice thicknesses, coverage, etc.), automated detection
    is unreliable. Use generate_orientation_comparison() first to
    visually verify the needed flips, then specify them explicitly.

    Parameters
    ----------
    moving_path : Path
        Path to moving image
    reference_path : Path
        Path to reference image
    output_path : Path, optional
        Output path for reoriented image. If None, creates in same directory
        with '_reoriented' suffix
    flips : list, optional
        List of axes to flip, e.g., ['y'] or ['x', 'y'].
        If None and auto_detect=False, no flipping is done.
    auto_detect : bool
        If True, attempt automatic detection (unreliable for very
        different image structures). Default False.

    Returns
    -------
    tuple
        (output_path, info_dict)

    Examples
    --------
    >>> # First, generate visual comparison
    >>> generate_orientation_comparison(moving, reference)
    >>> # View the output, determine flips needed
    >>> # Then apply the flip
    >>> reorient_to_reference(moving, reference, flips=['y'])
    """
    info = {
        'moving': str(moving_path),
        'reference': str(reference_path),
        'flips_applied': [],
        'method': 'manual' if flips else ('auto' if auto_detect else 'none')
    }

    if flips is None and auto_detect:
        # Try auto-detection (may not be reliable)
        result = verify_orientation(moving_path, reference_path, verbose=True)
        flips = result.get('flips_needed', [])
        info['auto_detected_flips'] = flips

    if not flips:
        print(f"No flips specified or detected. Image unchanged.")
        return moving_path, info

    if output_path is None:
        stem = moving_path.stem.replace('.nii', '')
        output_path = moving_path.parent / f"{stem}_reoriented.nii.gz"

    flip_image(moving_path, output_path, flips)
    info['flips_applied'] = flips
    info['output'] = str(output_path)

    # Generate verification comparison
    verify_path = output_path.parent / f"{output_path.stem}_verify.png"
    generate_orientation_comparison(output_path, reference_path, verify_path, test_flips=False)
    info['verification_image'] = str(verify_path)

    return output_path, info


def verify_and_fix_orientation(
    moving_path: Path,
    reference_path: Path,
    output_dir: Optional[Path] = None
) -> Tuple[Path, str]:
    """
    Interactive workflow for orientation verification and correction.

    Generates a comparison figure and prompts for user decision.
    For batch processing, use generate_orientation_comparison() and
    reorient_to_reference() separately.

    Parameters
    ----------
    moving_path : Path
        Path to moving image
    reference_path : Path
        Path to reference image
    output_dir : Path, optional
        Directory for outputs

    Returns
    -------
    tuple
        (final_image_path, action_taken)
    """
    if output_dir is None:
        output_dir = moving_path.parent

    # Generate comparison
    comparison_path = output_dir / f"{moving_path.stem}_orientation_check.png"
    generate_orientation_comparison(moving_path, reference_path, comparison_path)

    print(f"\n" + "="*60)
    print("ORIENTATION VERIFICATION")
    print("="*60)
    print(f"Comparison saved to: {comparison_path}")
    print(f"\nPlease view the comparison and determine if a flip is needed.")
    print(f"The top row shows original, bottom row shows Y-flipped version.")
    print(f"\nTo apply a Y-flip, run:")
    print(f"  from neurofaune.utils.orientation import reorient_to_reference")
    print(f"  reorient_to_reference('{moving_path}', '{reference_path}', flips=['y'])")

    return moving_path, 'verification_generated'
