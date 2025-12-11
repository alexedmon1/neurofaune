#!/usr/bin/env python3
"""
Parameter sweep to optimize skull stripping for functional BOLD data.

This script tests various parameter combinations for both FSL BET and bet4animal
to find optimal settings for rodent functional MRI preprocessing.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import subprocess
import shutil
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from typing import Dict, List, Tuple, Optional

# Add neurofaune to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_reference_volume(
    bold_file: Path,
    output_file: Path,
    method: str = "mean"
) -> Path:
    """
    Extract a reference volume from 4D BOLD timeseries.

    Parameters
    ----------
    bold_file : Path
        Input 4D BOLD file
    output_file : Path
        Output reference volume
    method : str
        Method to extract reference: "mean" (temporal mean), "middle" (middle volume),
        or integer volume index

    Returns
    -------
    Path
        Path to reference volume file
    """
    print(f"Extracting reference volume using method: {method}")

    img = nib.load(bold_file)
    data = img.get_fdata()

    if len(data.shape) == 4:
        n_vols = data.shape[3]

        if method == "mean":
            print(f"  Computing temporal mean of {n_vols} volumes...")
            ref_data = data.mean(axis=3)
        elif method == "middle":
            middle_vol = n_vols // 2
            print(f"  Extracting middle volume (volume {middle_vol} of {n_vols})...")
            ref_data = data[..., middle_vol]
        else:
            # Assume it's an integer index
            try:
                vol_idx = int(method)
                if vol_idx < 0 or vol_idx >= n_vols:
                    raise ValueError(f"Volume index {vol_idx} out of range [0, {n_vols-1}]")
                print(f"  Extracting volume {vol_idx} of {n_vols}...")
                ref_data = data[..., vol_idx]
            except ValueError:
                raise ValueError(f"Unknown method: {method}. Use 'mean', 'middle', or integer index")
    else:
        # Already 3D
        print(f"  Input is already 3D, using as-is")
        ref_data = data

    ref_img = nib.Nifti1Image(ref_data, img.affine, img.header)
    nib.save(ref_img, output_file)

    print(f"  Saved reference volume: {output_file}")
    return output_file


def test_bet_basic(
    input_file: Path,
    output_dir: Path,
    frac: float,
    use_functional_flag: bool = False
) -> Dict:
    """
    Test FSL BET with given parameters.

    Parameters
    ----------
    input_file : Path
        Input mean BOLD image
    output_dir : Path
        Output directory for this test
    frac : float
        Fractional intensity threshold
    use_functional_flag : bool
        Whether to use -F flag for functional optimization

    Returns
    -------
    dict
        Results including file paths and quality metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    method_name = f"bet_frac{frac}_F{use_functional_flag}"
    brain_file = output_dir / f"brain_{method_name}.nii.gz"

    # Build command
    cmd = [
        "bet",
        str(input_file),
        str(brain_file.with_suffix('')),  # BET adds .nii.gz
        "-f", str(frac),
        "-R",  # Robust brain center estimation
        "-m",  # Create mask
        "-n"   # Don't generate surface
    ]

    if use_functional_flag:
        cmd.append("-F")  # Functional-specific optimization

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Find mask file (BET creates it with _mask suffix)
        mask_file = brain_file.with_name(brain_file.stem.replace('.nii', '') + '_mask.nii.gz')

        if not brain_file.exists() or not mask_file.exists():
            return {
                'method': method_name,
                'status': 'failed',
                'error': 'Output files not created'
            }

        # Calculate quality metrics
        metrics = calculate_quality_metrics(input_file, brain_file, mask_file)

        return {
            'method': method_name,
            'tool': 'bet',
            'params': {
                'frac': frac,
                'functional_flag': use_functional_flag
            },
            'brain_file': brain_file,
            'mask_file': mask_file,
            'status': 'success',
            **metrics
        }

    except subprocess.CalledProcessError as e:
        return {
            'method': method_name,
            'status': 'failed',
            'error': str(e),
            'stderr': e.stderr if hasattr(e, 'stderr') else None
        }


def test_bet4animal(
    input_file: Path,
    output_dir: Path,
    frac: float,
    center: Tuple[int, int, int],
    radius: int = 125,
    scale: Tuple[float, float, float] = (1, 1, 1.5),
    width: float = 2.5
) -> Dict:
    """
    Test bet4animal with given parameters.

    Parameters
    ----------
    input_file : Path
        Input mean BOLD image
    output_dir : Path
        Output directory for this test
    frac : float
        Fractional intensity threshold
    center : tuple
        Brain center coordinates (x, y, z)
    radius : int
        Brain radius in voxels
    scale : tuple
        Voxel scaling factors
    width : float
        Smoothness parameter

    Returns
    -------
    dict
        Results including file paths and quality metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    method_name = f"bet4animal_frac{frac}_c{center[0]}_{center[1]}_{center[2]}"
    brain_file = output_dir / f"brain_{method_name}.nii.gz"

    # Build command
    cmd = [
        "bet4animal",
        str(input_file),
        str(brain_file.with_suffix('')),  # bet4animal adds .nii.gz
        "-f", str(frac),
        "-c", f"{center[0]} {center[1]} {center[2]}",
        "-r", str(radius),
        "-x", f"{scale[0]},{scale[1]},{scale[2]}",
        "-w", str(width),
        "-m"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Find mask file
        mask_file = brain_file.with_name(brain_file.stem.replace('.nii', '') + '_mask.nii.gz')

        if not brain_file.exists() or not mask_file.exists():
            return {
                'method': method_name,
                'status': 'failed',
                'error': 'Output files not created'
            }

        # Calculate quality metrics
        metrics = calculate_quality_metrics(input_file, brain_file, mask_file)

        return {
            'method': method_name,
            'tool': 'bet4animal',
            'params': {
                'frac': frac,
                'center': center,
                'radius': radius,
                'scale': scale,
                'width': width
            },
            'brain_file': brain_file,
            'mask_file': mask_file,
            'status': 'success',
            **metrics
        }

    except subprocess.CalledProcessError as e:
        return {
            'method': method_name,
            'status': 'failed',
            'error': str(e),
            'stderr': e.stderr if hasattr(e, 'stderr') else None
        }


def calculate_quality_metrics(
    original_file: Path,
    brain_file: Path,
    mask_file: Path
) -> Dict:
    """
    Calculate quality metrics for brain extraction.

    Parameters
    ----------
    original_file : Path
        Original image
    brain_file : Path
        Brain-extracted image
    mask_file : Path
        Binary brain mask

    Returns
    -------
    dict
        Quality metrics
    """
    # Load images
    orig_img = nib.load(original_file)
    orig_data = orig_img.get_fdata()

    brain_img = nib.load(brain_file)
    brain_data = brain_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # Calculate metrics
    orig_nonzero = (orig_data > 0).sum()
    mask_voxels = (mask_data > 0).sum()

    extraction_ratio = mask_voxels / orig_nonzero if orig_nonzero > 0 else 0

    # Brain intensity statistics
    brain_nonzero = brain_data[brain_data > 0]

    metrics = {
        'orig_nonzero_voxels': int(orig_nonzero),
        'mask_voxels': int(mask_voxels),
        'extraction_ratio': float(extraction_ratio),
        'brain_intensity_mean': float(brain_nonzero.mean()) if len(brain_nonzero) > 0 else 0,
        'brain_intensity_std': float(brain_nonzero.std()) if len(brain_nonzero) > 0 else 0,
        'brain_intensity_min': float(brain_nonzero.min()) if len(brain_nonzero) > 0 else 0,
        'brain_intensity_max': float(brain_nonzero.max()) if len(brain_nonzero) > 0 else 0
    }

    return metrics


def create_comparison_visualization(
    original_file: Path,
    results: List[Dict],
    output_file: Path,
    subject: str
):
    """
    Create comprehensive comparison visualization of all methods.

    Parameters
    ----------
    original_file : Path
        Original mean BOLD image
    results : list
        List of result dictionaries from all tests
    output_file : Path
        Output figure file
    subject : str
        Subject identifier
    """
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']

    if not successful:
        print("No successful results to visualize")
        return

    # Load original image
    orig_img = nib.load(original_file)
    orig_data = orig_img.get_fdata()

    # Get middle slices
    mid_axial = orig_data.shape[2] // 2
    mid_coronal = orig_data.shape[1] // 2
    mid_sagittal = orig_data.shape[0] // 2

    # Create figure
    n_methods = len(successful)
    n_cols = 4  # Original + 3 views per method
    n_rows = n_methods + 1  # Header row + one per method

    fig = plt.figure(figsize=(20, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)

    fig.suptitle(f'{subject} - Functional Skull Stripping Parameter Sweep',
                 fontsize=16, fontweight='bold')

    # Header row - show original
    ax = fig.add_subplot(gs[0, :])
    ax.text(0.5, 0.5, 'Original Mean BOLD',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Show original in three views
    axes = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]

    axes[0].imshow(orig_data[:, :, mid_axial].T, cmap='gray', origin='lower')
    axes[0].set_title('Axial', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(orig_data[:, mid_coronal, :].T, cmap='gray', origin='lower')
    axes[1].set_title('Coronal', fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(orig_data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
    axes[2].set_title('Sagittal', fontsize=10)
    axes[2].axis('off')

    # Plot each method
    for idx, result in enumerate(successful):
        row = idx + 1

        # Load brain and mask
        brain_img = nib.load(result['brain_file'])
        brain_data = brain_img.get_fdata()

        mask_img = nib.load(result['mask_file'])
        mask_data = mask_img.get_fdata()

        # Method name and metrics
        ax_label = fig.add_subplot(gs[row, 0])

        method_text = f"{result['method']}\n\n"
        method_text += f"Extraction Ratio: {result['extraction_ratio']:.3f}\n"
        method_text += f"Mask Voxels: {result['mask_voxels']:,}\n"
        method_text += f"Brain Mean: {result['brain_intensity_mean']:.1f}"

        ax_label.text(0.1, 0.5, method_text,
                     ha='left', va='center', fontsize=9, family='monospace')
        ax_label.axis('off')

        # Axial view with mask overlay
        ax_axial = fig.add_subplot(gs[row, 1])
        ax_axial.imshow(orig_data[:, :, mid_axial].T, cmap='gray', origin='lower')
        ax_axial.imshow(mask_data[:, :, mid_axial].T, cmap='Reds', alpha=0.3, origin='lower')
        ax_axial.set_title('Axial + Mask', fontsize=9)
        ax_axial.axis('off')

        # Coronal view with mask overlay
        ax_coronal = fig.add_subplot(gs[row, 2])
        ax_coronal.imshow(orig_data[:, mid_coronal, :].T, cmap='gray', origin='lower')
        ax_coronal.imshow(mask_data[:, mid_coronal, :].T, cmap='Reds', alpha=0.3, origin='lower')
        ax_coronal.set_title('Coronal + Mask', fontsize=9)
        ax_coronal.axis('off')

        # Sagittal view with mask overlay
        ax_sagittal = fig.add_subplot(gs[row, 3])
        ax_sagittal.imshow(orig_data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
        ax_sagittal.imshow(mask_data[mid_sagittal, :, :].T, cmap='Reds', alpha=0.3, origin='lower')
        ax_sagittal.set_title('Sagittal + Mask', fontsize=9)
        ax_sagittal.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison figure: {output_file}")


def run_parameter_sweep(
    subject: str,
    session: str,
    bold_file: Path,
    output_dir: Path,
    ref_method: str = "middle"
) -> List[Dict]:
    """
    Run parameter sweep for skull stripping optimization.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier
    bold_file : Path
        Input 4D BOLD file
    output_dir : Path
        Output directory for results
    ref_method : str
        Reference volume extraction method: "mean", "middle", or volume index

    Returns
    -------
    list
        List of result dictionaries from all tests
    """
    print("="*80)
    print(f"PARAMETER SWEEP: {subject} {session}")
    print("="*80)
    print(f"BOLD file: {bold_file}")
    print(f"Reference method: {ref_method}")
    print(f"Output directory: {output_dir}")
    print()

    # Create work directory
    work_dir = output_dir / 'work'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Extract reference volume
    ref_bold = work_dir / f"{subject}_{session}_bold_ref.nii.gz"
    extract_reference_volume(bold_file, ref_bold, method=ref_method)

    # For backwards compatibility, also save as mean_bold
    mean_bold = ref_bold

    # Load mean BOLD to get dimensions for center estimation
    mean_img = nib.load(mean_bold)
    dims = mean_img.shape

    # Estimate brain center (rough approximation)
    center_default = (dims[0] // 2, dims[1] // 2, dims[2] // 2)

    print(f"\nImage dimensions: {dims}")
    print(f"Estimated brain center: {center_default}")
    print()

    results = []

    # Test FSL BET with different parameters
    print("\n" + "="*80)
    print("TESTING FSL BET")
    print("="*80)

    bet_fracs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for frac in bet_fracs:
        print(f"\nTesting BET frac={frac} (standard)...")
        result = test_bet_basic(
            mean_bold,
            output_dir / 'bet_standard',
            frac=frac,
            use_functional_flag=False
        )
        results.append(result)

        if result['status'] == 'success':
            print(f"  ✓ Success - Extraction ratio: {result['extraction_ratio']:.3f}")
        else:
            print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")

    # Test with -F flag
    for frac in [0.3, 0.35, 0.4]:
        print(f"\nTesting BET frac={frac} (with -F flag)...")
        result = test_bet_basic(
            mean_bold,
            output_dir / 'bet_functional',
            frac=frac,
            use_functional_flag=True
        )
        results.append(result)

        if result['status'] == 'success':
            print(f"  ✓ Success - Extraction ratio: {result['extraction_ratio']:.3f}")
        else:
            print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")

    # Test bet4animal with different parameters
    print("\n" + "="*80)
    print("TESTING BET4ANIMAL")
    print("="*80)

    # Test different frac values (lower than anatomical default of 0.7)
    bet4animal_fracs = [0.3, 0.4, 0.5, 0.6]

    # Test different centers (estimate based on image dimensions)
    centers = [
        center_default,
        (center_default[0], center_default[1], center_default[2] + 2),  # Shifted dorsally
        (center_default[0], center_default[1], center_default[2] - 2),  # Shifted ventrally
    ]

    for frac in bet4animal_fracs:
        for center in centers:
            center_str = f"({center[0]},{center[1]},{center[2]})"
            print(f"\nTesting bet4animal frac={frac} center={center_str}...")

            result = test_bet4animal(
                mean_bold,
                output_dir / 'bet4animal',
                frac=frac,
                center=center,
                radius=125,
                scale=(1, 1, 1.5),
                width=2.5
            )
            results.append(result)

            if result['status'] == 'success':
                print(f"  ✓ Success - Extraction ratio: {result['extraction_ratio']:.3f}")
            else:
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")

    return results, mean_bold


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize skull stripping parameters for functional BOLD data'
    )
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-Rat108)')
    parser.add_argument('--session', required=True, help='Session ID (e.g., ses-p30)')
    parser.add_argument('--bold-file', required=True, type=Path,
                       help='Path to 4D BOLD file')
    parser.add_argument('--output-dir', required=True, type=Path,
                       help='Output directory for results')
    parser.add_argument('--ref-method', default='middle', type=str,
                       help='Reference volume method: "mean" (temporal average), "middle" (middle volume), or volume index (default: middle)')

    args = parser.parse_args()

    # Run parameter sweep
    results, mean_bold = run_parameter_sweep(
        args.subject,
        args.session,
        args.bold_file,
        args.output_dir,
        ref_method=args.ref_method
    )

    # Save results to JSON
    results_file = args.output_dir / 'results.json'

    # Convert Path objects to strings for JSON serialization
    results_serializable = []
    for r in results:
        r_copy = r.copy()
        if 'brain_file' in r_copy:
            r_copy['brain_file'] = str(r_copy['brain_file'])
        if 'mask_file' in r_copy:
            r_copy['mask_file'] = str(r_copy['mask_file'])
        results_serializable.append(r_copy)

    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nSaved results to: {results_file}")

    # Create comparison visualization
    qc_file = args.output_dir / f'{args.subject}_{args.session}_skull_strip_comparison.png'
    create_comparison_visualization(
        mean_bold,
        results,
        qc_file,
        f"{args.subject} {args.session}"
    )

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']

    if successful:
        # Sort by extraction ratio (higher is generally better, but need visual confirmation)
        successful_sorted = sorted(successful, key=lambda x: x['extraction_ratio'], reverse=True)

        print(f"\nTotal tests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(results) - len(successful)}")
        print()

        print("Top 5 results by extraction ratio:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Method':<40} {'Ratio':<10} {'Mask Voxels':<15}")
        print("-" * 80)

        for idx, result in enumerate(successful_sorted[:5], 1):
            print(f"{idx:<6} {result['method']:<40} {result['extraction_ratio']:<10.3f} {result['mask_voxels']:<15,}")

        print()
        print("Please visually inspect the comparison figure to select the best method.")
        print(f"QC Figure: {qc_file}")
    else:
        print("\n✗ All tests failed!")
        print("\nFailed results:")
        for r in results:
            print(f"  {r['method']}: {r.get('error', 'Unknown error')}")

    print("="*80)


if __name__ == '__main__':
    main()
