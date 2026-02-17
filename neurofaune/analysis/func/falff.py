#!/usr/bin/env python3
"""
fALFF (fractional Amplitude of Low-Frequency Fluctuations) Analysis

ALFF measures the amplitude of low-frequency fluctuations in the BOLD signal.
fALFF is the ratio of ALFF to total amplitude across all frequencies, which
makes it more robust to physiological noise.

Uses vectorized FFT across all brain voxels simultaneously for ~100x speedup
over voxel-by-voxel computation.

References:
- Zou et al. (2008). An improved approach to detection of amplitude of
  low-frequency fluctuation (ALFF) for resting-state fMRI. Journal of
  Neuroscience Methods, 172(1), 137-141.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional


def compute_falff_map(func_file: Path,
                      mask_file: Path,
                      output_dir: Path,
                      subject: str,
                      session: str,
                      tr: float,
                      low_freq: float = 0.01,
                      high_freq: float = 0.08) -> dict:
    """
    Compute ALFF and fALFF maps for whole brain using vectorized FFT.

    Parameters
    ----------
    func_file : Path
        Preprocessed 4D functional image (UNFILTERED, post-regression).
        Must NOT be bandpass filtered — fALFF requires full spectrum.
    mask_file : Path
        Brain mask.
    output_dir : Path
        Directory to save output maps.
    subject : str
        Subject ID (e.g., 'sub-Rat49').
    session : str
        Session ID (e.g., 'ses-p90').
    tr : float
        Repetition time in seconds.
    low_freq : float
        Lower bound of low-frequency range (Hz). Default: 0.01.
    high_freq : float
        Upper bound of low-frequency range (Hz). Default: 0.08.

    Returns
    -------
    dict
        Dictionary with keys:
        - alff_file: Path to ALFF map
        - falff_file: Path to fALFF map
        - statistics: dict with ALFF/fALFF summary statistics
    """
    func_file = Path(func_file)
    mask_file = Path(mask_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Computing ALFF and fALFF")
    print("=" * 60)
    print(f"  Input: {func_file.name}")
    print(f"  Frequency range: {low_freq} - {high_freq} Hz")
    print(f"  TR: {tr} s")

    # Load data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()

    if func_data.ndim != 4:
        raise ValueError(f"Expected 4D functional data, got {func_data.ndim}D")

    nx, ny, nz, nt = func_data.shape
    print(f"  Dimensions: {nx} x {ny} x {nz} x {nt} timepoints")

    mask_data = nib.load(mask_file).get_fdata().astype(bool)
    n_voxels = int(np.sum(mask_data))
    print(f"  Brain voxels: {n_voxels}")

    # Frequency resolution
    freq_resolution = 1.0 / (nt * tr)
    nyquist_freq = 1.0 / (2 * tr)
    print(f"  Frequency resolution: {freq_resolution:.4f} Hz")
    print(f"  Nyquist frequency: {nyquist_freq:.4f} Hz")

    if high_freq > nyquist_freq:
        print(f"  Warning: high_freq ({high_freq} Hz) exceeds Nyquist ({nyquist_freq:.4f} Hz), clamping")
        high_freq = nyquist_freq

    # Extract all brain voxel timeseries: (n_voxels, n_timepoints)
    voxel_indices = np.where(mask_data)
    timeseries = func_data[voxel_indices[0], voxel_indices[1], voxel_indices[2], :]

    # Remove mean from each voxel (detrend constant)
    timeseries = timeseries - timeseries.mean(axis=1, keepdims=True)

    # Identify constant voxels
    voxel_std = np.std(timeseries, axis=1)
    valid_mask = voxel_std > 0

    # Vectorized FFT: rfft across all voxels at once
    print(f"  Computing FFT for {np.sum(valid_mask)} valid voxels...")
    fft_vals = np.fft.rfft(timeseries[valid_mask], axis=1)
    power = np.abs(fft_vals) ** 2

    # Frequency axis for rfft
    freqs = np.fft.rfftfreq(nt, d=tr)

    # Exclude DC component (freq=0)
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    power_pos = power[:, positive_mask]

    # Low-frequency band indices
    lf_mask = (freqs_pos >= low_freq) & (freqs_pos <= high_freq)
    n_lf_bins = int(np.sum(lf_mask))
    print(f"  Low-frequency bins: {n_lf_bins} (of {len(freqs_pos)} total)")

    # ALFF = sqrt(sum of power in LF band)
    alff_vals = np.sqrt(np.sum(power_pos[:, lf_mask], axis=1))

    # Total amplitude = sqrt(sum of all power)
    total_amp = np.sqrt(np.sum(power_pos, axis=1))

    # fALFF = ALFF / total_amplitude
    falff_vals = np.zeros_like(alff_vals)
    nonzero = total_amp > 0
    falff_vals[nonzero] = alff_vals[nonzero] / total_amp[nonzero]

    # Map back to 3D volumes
    alff_data = np.zeros((nx, ny, nz), dtype=np.float32)
    falff_data = np.zeros((nx, ny, nz), dtype=np.float32)

    valid_indices = (voxel_indices[0][valid_mask],
                     voxel_indices[1][valid_mask],
                     voxel_indices[2][valid_mask])
    alff_data[valid_indices] = alff_vals.astype(np.float32)
    falff_data[valid_indices] = falff_vals.astype(np.float32)

    # Save output maps
    alff_file = output_dir / f"{subject}_{session}_desc-ALFF_bold.nii.gz"
    falff_file = output_dir / f"{subject}_{session}_desc-fALFF_bold.nii.gz"

    # Use 3D header (drop time dimension)
    hdr = func_img.header.copy()
    hdr.set_data_shape((nx, ny, nz))

    nib.save(nib.Nifti1Image(alff_data, func_img.affine, hdr), alff_file)
    nib.save(nib.Nifti1Image(falff_data, func_img.affine, hdr), falff_file)

    print(f"  Saved ALFF: {alff_file.name}")
    print(f"  Saved fALFF: {falff_file.name}")

    # Statistics within brain mask
    brain_alff = alff_data[mask_data]
    brain_falff = falff_data[mask_data]

    statistics = {
        'alff': {
            'mean': float(np.mean(brain_alff)),
            'std': float(np.std(brain_alff)),
            'min': float(np.min(brain_alff)),
            'max': float(np.max(brain_alff)),
        },
        'falff': {
            'mean': float(np.mean(brain_falff)),
            'std': float(np.std(brain_falff)),
            'min': float(np.min(brain_falff)),
            'max': float(np.max(brain_falff)),
        },
        'parameters': {
            'low_freq': low_freq,
            'high_freq': high_freq,
            'tr': tr,
            'n_timepoints': nt,
            'n_brain_voxels': n_voxels,
            'n_lf_bins': n_lf_bins,
            'freq_resolution': float(freq_resolution),
        },
    }

    print(f"\n  ALFF  — mean: {statistics['alff']['mean']:.4f}, "
          f"std: {statistics['alff']['std']:.4f}, "
          f"range: [{statistics['alff']['min']:.4f}, {statistics['alff']['max']:.4f}]")
    print(f"  fALFF — mean: {statistics['falff']['mean']:.4f}, "
          f"std: {statistics['falff']['std']:.4f}, "
          f"range: [{statistics['falff']['min']:.4f}, {statistics['falff']['max']:.4f}]")
    print("=" * 60)

    return {
        'alff_file': alff_file,
        'falff_file': falff_file,
        'statistics': statistics,
    }


def compute_falff_zscore(alff_file: Path,
                         falff_file: Path,
                         mask_file: Path,
                         output_dir: Path,
                         subject: str,
                         session: str) -> dict:
    """
    Standardize ALFF and fALFF maps to z-scores within the brain mask.

    Parameters
    ----------
    alff_file : Path
        Path to ALFF map.
    falff_file : Path
        Path to fALFF map.
    mask_file : Path
        Brain mask.
    output_dir : Path
        Directory to save z-scored maps.
    subject : str
        Subject ID.
    session : str
        Session ID.

    Returns
    -------
    dict
        Dictionary with keys:
        - alff_zscore_file: Path to z-scored ALFF map
        - falff_zscore_file: Path to z-scored fALFF map
    """
    alff_file = Path(alff_file)
    falff_file = Path(falff_file)
    mask_file = Path(mask_file)
    output_dir = Path(output_dir)

    print("  Standardizing ALFF/fALFF to z-scores...")

    alff_img = nib.load(alff_file)
    falff_img = nib.load(falff_file)
    alff_data = alff_img.get_fdata()
    falff_data = falff_img.get_fdata()
    mask_data = nib.load(mask_file).get_fdata().astype(bool)

    # Z-score ALFF
    brain_alff = alff_data[mask_data]
    mean_alff = np.mean(brain_alff)
    std_alff = np.std(brain_alff)

    alff_zscore = np.zeros_like(alff_data, dtype=np.float32)
    if std_alff > 0:
        alff_zscore[mask_data] = ((brain_alff - mean_alff) / std_alff).astype(np.float32)

    # Z-score fALFF
    brain_falff = falff_data[mask_data]
    mean_falff = np.mean(brain_falff)
    std_falff = np.std(brain_falff)

    falff_zscore = np.zeros_like(falff_data, dtype=np.float32)
    if std_falff > 0:
        falff_zscore[mask_data] = ((brain_falff - mean_falff) / std_falff).astype(np.float32)

    # Save
    alff_zscore_file = output_dir / f"{subject}_{session}_desc-ALFFzscore_bold.nii.gz"
    falff_zscore_file = output_dir / f"{subject}_{session}_desc-fALFFzscore_bold.nii.gz"

    nib.save(nib.Nifti1Image(alff_zscore, alff_img.affine, alff_img.header), alff_zscore_file)
    nib.save(nib.Nifti1Image(falff_zscore, falff_img.affine, falff_img.header), falff_zscore_file)

    print(f"  Saved z-scored ALFF: {alff_zscore_file.name}")
    print(f"  Saved z-scored fALFF: {falff_zscore_file.name}")

    return {
        'alff_zscore_file': alff_zscore_file,
        'falff_zscore_file': falff_zscore_file,
    }
