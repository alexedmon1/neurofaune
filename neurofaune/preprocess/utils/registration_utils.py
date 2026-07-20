"""
Shared registration utilities for partial-coverage MRI modalities.

These functions are used across fMRI, MSME, and DTI preprocessing workflows
when aligning partial-slab acquisitions to full-brain templates.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np


def register_via_anat_composition(
    moving_ref: Path,
    anat_t2w: Path,
    anat_to_tpl_affine: Path,
    anat_to_tpl_warp: Optional[Path],
    template_file: Path,
    output_prefix: Path,
    work_dir: Path,
    n_cores: int = 4,
) -> Dict[str, Any]:
    """Register a partial-slab moving reference to template VIA the same-session anat.

    moving -> anat (ANTs rigid, centre-of-mass initialised, MI) composed with the
    already-computed anat -> template transform. A within-subject moving->anat
    registration is far more tractable than a direct partial-slab -> group-template
    one (which is cross-subject, cross-contrast EPI/T2w, and has an ambiguous
    z-placement search), and it inherits the high-quality anat->template SyN.

    Writes ``<output_prefix>Warped.nii.gz`` (composed, for QC) and
    ``<output_prefix>CompositeWarp.nii.gz`` (a single displacement field the
    downstream stage applies as one transform).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    f2a_prefix = work_dir / 'moving_to_anat_'
    f2a_affine = Path(str(f2a_prefix) + '0GenericAffine.mat')

    # 1) moving -> anat: rigid, COM-initialised, mutual information (within-subject).
    subprocess.run([
        'antsRegistration', '-d', '3',
        '--output', f'[{f2a_prefix},{work_dir}/moving_in_anat.nii.gz]',
        '--initial-moving-transform', f'[{anat_t2w},{moving_ref},1]',
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{anat_t2w},{moving_ref},1,32,Regular,0.25]',
        '--convergence', '[500x250x100,1e-6,10]',
        '--shrink-factors', '4x2x1', '--smoothing-sigmas', '2x1x0vox',
        '--interpolation', 'Linear', '-v', '0',
    ], check=True, capture_output=True, text=True)

    # transform chain moving->template = (anat->tpl warp) o (anat->tpl affine) o (moving->anat)
    chain = []
    if anat_to_tpl_warp is not None and Path(anat_to_tpl_warp).exists():
        chain += ['-t', str(anat_to_tpl_warp)]
    chain += ['-t', str(anat_to_tpl_affine), '-t', str(f2a_affine)]

    warped = Path(str(output_prefix) + 'Warped.nii.gz')
    subprocess.run(['antsApplyTransforms', '-d', '3', '-i', str(moving_ref),
                    '-r', str(template_file), *chain, '-o', str(warped),
                    '--interpolation', 'Linear'], check=True, capture_output=True, text=True)
    # single composite displacement field for the downstream stage
    comp = Path(str(output_prefix) + 'CompositeWarp.nii.gz')
    subprocess.run(['antsApplyTransforms', '-d', '3', '-i', str(moving_ref),
                    '-r', str(template_file), *chain, '-o', f'[{comp},1]'],
                   check=True, capture_output=True, text=True)
    return {
        'moving_to_anat_affine': f2a_affine,
        'composite_warp': comp if comp.exists() else None,
        'warped': warped if warped.exists() else None,
        'method': 'anat_composition',
    }


def propagate_anat_mask(
    moving_ref: Path,
    anat_t2w: Path,
    anat_mask: Path,
    out_mask: Path,
    work_dir: Path,
    out_brain: Optional[Path] = None,
) -> Dict[str, Any]:
    """Derive a partial-slab brain mask by warping the same-session anat mask in.

    A within-subject rigid registration of the moving reference to the preproc
    T2w (centre-of-mass initialised, mutual information) followed by inverse-
    warping the anatomical brain mask (nearest-neighbour) into the moving image's
    native space. Unlike an intensity-threshold strip (e.g. slice-wise BET, which
    on low-contrast MSME degenerates to a fixed-area oval that clips cortex and
    swallows muscle), this inherits the true anatomical boundary from the T2w
    skull-strip, so the mask edge follows the brain.

    Writes ``out_mask`` (uint8, in moving space); if ``out_brain`` is given, also
    writes the masked moving reference. Returns the moving->anat affine so a
    caller can reuse it.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    m2a_prefix = work_dir / 'mask_moving_to_anat_'
    m2a_affine = Path(str(m2a_prefix) + '0GenericAffine.mat')

    # moving -> anat: rigid, COM-initialised, mutual information (within-subject).
    subprocess.run([
        'antsRegistration', '-d', '3',
        '--output', f'[{m2a_prefix},{work_dir}/mask_moving_in_anat.nii.gz]',
        '--initial-moving-transform', f'[{anat_t2w},{moving_ref},1]',
        '--transform', 'Rigid[0.1]',
        '--metric', f'MI[{anat_t2w},{moving_ref},1,32,Regular,0.25]',
        '--convergence', '[500x250x100,1e-6,10]',
        '--shrink-factors', '4x2x1', '--smoothing-sigmas', '2x1x0vox',
        '--interpolation', 'Linear', '-v', '0',
    ], check=True, capture_output=True, text=True)

    # anat mask -> moving space = inverse of (moving->anat), nearest-neighbour.
    subprocess.run([
        'antsApplyTransforms', '-d', '3', '-i', str(anat_mask),
        '-r', str(moving_ref), '-t', f'[{m2a_affine},1]',
        '-o', str(out_mask), '--interpolation', 'NearestNeighbor',
    ], check=True, capture_output=True, text=True)

    if out_brain is not None:
        m = nib.load(str(out_mask))
        ref = nib.load(str(moving_ref))
        brain = ref.get_fdata() * (m.get_fdata() > 0)
        nib.save(nib.Nifti1Image(brain.astype(np.float32), ref.affine, ref.header),
                 str(out_brain))

    return {
        'moving_to_anat_affine': m2a_affine,
        'mask': out_mask if Path(out_mask).exists() else None,
        'brain': out_brain if (out_brain and Path(out_brain).exists()) else None,
        'method': 'anat_mask',
    }


def find_z_offset_ncc(
    moving_img: nib.Nifti1Image,
    fixed_img: nib.Nifti1Image,
    work_dir: Path,
    z_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Path, Dict]:
    """
    Find the optimal Z offset for a partial-slab acquisition relative to a
    full-brain reference image via normalized cross-correlation (NCC) scan.

    Resamples the moving image in-plane to the fixed image resolution, then
    slides it along Z and picks the offset with the highest mean per-slice NCC.
    Writes an ITK-format affine transform encoding the Z-translation so that
    ``antsRegistration --initial-moving-transform`` can consume it directly.

    Parameters
    ----------
    moving_img : Nifti1Image
        Partial-slab reference volume (e.g. mean BOLD, MSME first echo).
    fixed_img : Nifti1Image
        Full-brain reference (e.g. cohort T2w template).
    work_dir : Path
        Directory for the output transform file.
    z_range : (int, int), optional
        ``(min_slice, max_slice)`` to constrain the Z search to a known
        sub-range of the fixed image.  If None, the full valid range is
        searched.

    Returns
    -------
    transform_file : Path
        Path to the ITK-format ``.txt`` initial transform.
    info : dict
        ``z_offset_slice`` – best template slice index for moving slice 0.
        ``z_offset_mm``    – corresponding Z translation in physical units.
        ``ncc``            – NCC score at best offset.
        ``bold_fixed_range`` – (start, end) template slice indices covered.
    """
    from scipy.ndimage import zoom as scipy_zoom

    moving_data = moving_img.get_fdata()
    fixed_data = fixed_img.get_fdata()
    moving_zooms = moving_img.header.get_zooms()
    fixed_zooms = fixed_img.header.get_zooms()

    # Resample moving in-plane to fixed resolution for slice-by-slice comparison
    scale_xy = float(moving_zooms[0]) / float(fixed_zooms[0])
    moving_resampled = scipy_zoom(moving_data, (scale_xy, scale_xy, 1), order=1)

    # How many fixed slices does one moving slice span?
    z_scale = float(moving_zooms[2]) / float(fixed_zooms[2])
    n_moving_in_fixed = int(round(moving_data.shape[2] * z_scale))

    max_offset = fixed_data.shape[2] - n_moving_in_fixed
    if z_range is not None:
        z_start = max(0, z_range[0])
        z_end = min(max_offset, z_range[1]) + 1
        print(f"  Z search range: slices {z_start}-{min(max_offset, z_range[1])} "
              f"(constrained; full range 0-{max_offset})")
    else:
        z_start = 0
        z_end = max(max_offset, 1)

    best_ncc = -1.0
    best_offset = max(0, max_offset // 2)  # sensible fallback

    for z_offset in range(z_start, z_end):
        total_ncc = 0.0
        n_valid = 0

        for mz in range(moving_data.shape[2]):
            fz = int(round(z_offset + mz * z_scale))
            if fz < 0 or fz >= fixed_data.shape[2]:
                continue

            min_x = min(moving_resampled.shape[0], fixed_data.shape[0])
            min_y = min(moving_resampled.shape[1], fixed_data.shape[1])

            m_sl = moving_resampled[:min_x, :min_y, mz]
            f_sl = fixed_data[:min_x, :min_y, fz]

            mask = m_sl > 0
            if mask.sum() < 500:
                continue

            m_vals = m_sl[mask]
            f_vals = f_sl[mask]
            m_norm = (m_vals - m_vals.mean()) / (m_vals.std() + 1e-10)
            f_norm = (f_vals - f_vals.mean()) / (f_vals.std() + 1e-10)
            total_ncc += float(np.mean(m_norm * f_norm))
            n_valid += 1

        if n_valid > 0:
            avg_ncc = total_ncc / n_valid
            if avg_ncc > best_ncc:
                best_ncc = avg_ncc
                best_offset = z_offset

    z_offset_mm = best_offset * float(fixed_zooms[2])
    print(f"  Best Z offset: fixed slice {best_offset} ({z_offset_mm:.1f} mm), NCC={best_ncc:.4f}")
    print(f"  Moving maps to fixed slices {best_offset}-{best_offset + n_moving_in_fixed}")

    # Write ITK initial transform with Z translation only.
    # Convention (ANTs moving→fixed): fixed = R*moving + t
    # R = I, t_z = -z_offset_mm  →  moving at Z=0 aligns with fixed at Z=z_offset_mm
    work_dir.mkdir(parents=True, exist_ok=True)
    transform_file = work_dir / "initial_z_offset.txt"
    with open(transform_file, "w") as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")
        f.write(f"Parameters: 1 0 0 0 1 0 0 0 1 0 0 {-z_offset_mm}\n")
        f.write("FixedParameters: 0 0 0\n")

    return transform_file, {
        "z_offset_slice": best_offset,
        "z_offset_mm": z_offset_mm,
        "ncc": best_ncc,
        "bold_fixed_range": (best_offset, best_offset + n_moving_in_fixed),
    }
