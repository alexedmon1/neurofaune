"""
Slice timing correction for fMRI data.

Corrects for temporal differences in slice acquisition within each TR.
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Union


def calculate_slice_times(
    n_slices: int,
    tr: float,
    slice_order: str = 'interleaved',
    custom_order: Optional[List[int]] = None
) -> np.ndarray:
    """
    Calculate slice acquisition times based on acquisition order.

    Parameters
    ----------
    n_slices : int
        Number of slices
    tr : float
        Repetition time in seconds
    slice_order : str
        Slice acquisition order:
        - 'sequential_ascending': 0, 1, 2, ... (bottom to top)
        - 'sequential_descending': n-1, n-2, ..., 0 (top to bottom)
        - 'interleaved': 0, 2, 4, ..., 1, 3, 5, ... (odd first)
        - 'interleaved_even_first': 1, 3, 5, ..., 0, 2, 4, ... (even first)
        - 'custom': Use custom_order parameter
    custom_order : list of int, optional
        Custom slice acquisition order (0-indexed)

    Returns
    -------
    np.ndarray
        Slice acquisition times in seconds (shape: n_slices)

    Examples
    --------
    >>> # Interleaved acquisition, 9 slices, TR=0.5s
    >>> times = calculate_slice_times(9, 0.5, 'interleaved')
    >>> # For Bruker interleaved: slices acquired at times proportional to their order
    """
    # Time between slice acquisitions
    slice_duration = tr / n_slices

    if slice_order == 'custom':
        if custom_order is None:
            raise ValueError("custom_order must be provided when slice_order='custom'")
        if len(custom_order) != n_slices:
            raise ValueError(f"custom_order length ({len(custom_order)}) != n_slices ({n_slices})")

        # Create acquisition time array
        slice_times = np.zeros(n_slices)
        for acq_idx, slice_idx in enumerate(custom_order):
            slice_times[slice_idx] = acq_idx * slice_duration

    elif slice_order == 'sequential_ascending':
        # Sequential acquisition: slice 0, 1, 2, ..., n-1
        slice_times = np.arange(n_slices) * slice_duration

    elif slice_order == 'sequential_descending':
        # Sequential acquisition: slice n-1, n-2, ..., 0
        slice_times = np.arange(n_slices)[::-1] * slice_duration

    elif slice_order == 'interleaved':
        # Interleaved odd-first: 0, 2, 4, ..., 1, 3, 5, ...
        odd_slices = list(range(0, n_slices, 2))  # 0, 2, 4, ...
        even_slices = list(range(1, n_slices, 2))  # 1, 3, 5, ...
        acq_order = odd_slices + even_slices

        slice_times = np.zeros(n_slices)
        for acq_idx, slice_idx in enumerate(acq_order):
            slice_times[slice_idx] = acq_idx * slice_duration

    elif slice_order == 'interleaved_even_first':
        # Interleaved even-first: 1, 3, 5, ..., 0, 2, 4, ...
        even_slices = list(range(1, n_slices, 2))  # 1, 3, 5, ...
        odd_slices = list(range(0, n_slices, 2))  # 0, 2, 4, ...
        acq_order = even_slices + odd_slices

        slice_times = np.zeros(n_slices)
        for acq_idx, slice_idx in enumerate(acq_order):
            slice_times[slice_idx] = acq_idx * slice_duration

    else:
        raise ValueError(f"Unknown slice_order: {slice_order}")

    return slice_times


def run_slice_timing_correction(
    input_file: Path,
    output_file: Path,
    tr: float,
    slice_order: str = 'interleaved',
    custom_order: Optional[List[int]] = None,
    reference_slice: Union[int, str] = 'middle'
) -> Path:
    """
    Perform slice timing correction using FSL slicetimer.

    IMPORTANT: This should be run BEFORE motion correction, as motion correction
    involves spatial interpolation which assumes all slices were acquired simultaneously.

    Parameters
    ----------
    input_file : Path
        Input 4D fMRI file
    output_file : Path
        Output slice-time corrected file
    tr : float
        Repetition time in seconds
    slice_order : str
        Slice acquisition order (see calculate_slice_times for options)
    custom_order : list of int, optional
        Custom slice acquisition order (0-indexed)
    reference_slice : int or 'middle'
        Reference slice to align to (default: 'middle' = middle slice in time)

    Returns
    -------
    Path
        Path to output file

    Examples
    --------
    >>> # Standard interleaved acquisition
    >>> run_slice_timing_correction(
    ...     input_file=Path('bold.nii.gz'),
    ...     output_file=Path('bold_stc.nii.gz'),
    ...     tr=0.5,
    ...     slice_order='interleaved'
    ... )

    >>> # Bruker custom acquisition order
    >>> run_slice_timing_correction(
    ...     input_file=Path('bold.nii.gz'),
    ...     output_file=Path('bold_stc.nii.gz'),
    ...     tr=0.5,
    ...     slice_order='custom',
    ...     custom_order=[0, 2, 4, 6, 8, 1, 3, 5, 7]
    ... )
    """
    print(f"\nPerforming slice timing correction...")
    print(f"  Input: {input_file.name}")
    print(f"  TR: {tr}s")
    print(f"  Slice order: {slice_order}")

    # Get number of slices
    import nibabel as nib
    img = nib.load(input_file)
    n_slices = img.shape[2]
    print(f"  Number of slices: {n_slices}")

    # Calculate slice times
    slice_times = calculate_slice_times(
        n_slices=n_slices,
        tr=tr,
        slice_order=slice_order,
        custom_order=custom_order
    )

    # Determine reference slice
    if reference_slice == 'middle':
        # Reference to middle slice in acquisition time
        ref_slice_idx = int(n_slices / 2)
        ref_time = slice_times[ref_slice_idx]
    else:
        ref_time = slice_times[reference_slice]

    print(f"  Reference slice: {reference_slice} (time: {ref_time:.3f}s)")

    # Create custom timing file for FSL slicetimer
    timing_file = output_file.parent / f"{output_file.stem}_slice_times.txt"

    # FSL slicetimer expects slice times in fractions of TR
    slice_times_frac = slice_times / tr

    with open(timing_file, 'w') as f:
        for time_frac in slice_times_frac:
            f.write(f"{time_frac:.6f}\n")

    print(f"  Created timing file: {timing_file}")

    # Run FSL slicetimer
    cmd = [
        'slicetimer',
        '-i', str(input_file),
        '-o', str(output_file),
        '-r', str(tr),
        f'--tcustom={timing_file}'
    ]

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR: slicetimer failed!")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"Slice timing correction failed")

    # Clean up timing file
    timing_file.unlink()

    print(f"  ✓ Slice timing corrected: {output_file}")

    return output_file


def extract_slice_order_from_json(json_file: Path) -> Optional[List[int]]:
    """
    Extract slice timing information from BIDS JSON sidecar.

    Parameters
    ----------
    json_file : Path
        BIDS JSON sidecar file

    Returns
    -------
    list of int or None
        Slice acquisition order (0-indexed), or None if not found

    Notes
    -----
    BIDS spec defines SliceTiming field as a list of slice acquisition times
    in seconds, which can be used to infer slice order.
    """
    import json

    with open(json_file, 'r') as f:
        metadata = json.load(f)

    # Check for SliceTiming field (BIDS standard)
    if 'SliceTiming' in metadata:
        slice_times = np.array(metadata['SliceTiming'])
        # Get slice order by argsort of times
        slice_order = np.argsort(slice_times).tolist()
        return slice_order

    # Check for Bruker-specific fields
    # (These would need to be added during BIDS conversion)
    if 'SliceAcquisitionOrder' in metadata:
        return metadata['SliceAcquisitionOrder']

    return None


def detect_slice_order(
    json_file: Optional[Path] = None,
    method: str = 'interleaved'
) -> tuple:
    """
    Detect or infer slice acquisition order.

    Parameters
    ----------
    json_file : Path, optional
        BIDS JSON sidecar to extract slice order from
    method : str
        Default method to use if not found in JSON

    Returns
    -------
    tuple
        (slice_order, custom_order) where slice_order is a string
        and custom_order is None or a list

    Examples
    --------
    >>> slice_order, custom = detect_slice_order(
    ...     json_file=Path('bold.json'),
    ...     method='interleaved'
    ... )
    """
    if json_file and json_file.exists():
        custom_order = extract_slice_order_from_json(json_file)
        if custom_order is not None:
            return 'custom', custom_order

    # Fall back to provided method
    return method, None
